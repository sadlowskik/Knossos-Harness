//! The Claude Code adapter against real processes: the permission bridge
//! subcommand driven end to end over MCP stdio against a Field API stub, and
//! process-generation ownership (`field/server/test/process-generation.test.mjs`).

use knossos::field::adapter::{Adapter, EventSink};
use knossos::field::claude_session::{ClaudeOptions, ClaudeSession};
use knossos::field::permission_bridge::{SUBCOMMAND, TOOL_NAME};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Lines, Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

type Seen = Arc<Mutex<Vec<(String, Value)>>>;

/// A one-route Field API stub. Each queued decision answers one
/// `POST /api/internal/permission`; `null` answers with a 403 instead.
fn field_stub(decisions: Vec<Value>) -> (String, Seen) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    std::thread::spawn(move || {
        for decision in decisions {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            let mut authorization = String::new();
            let mut length = 0usize;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let Some((name, value)) = line.trim_end().split_once(':') else {
                    break;
                };
                if name.eq_ignore_ascii_case("authorization") {
                    authorization = value.trim().to_string();
                } else if name.eq_ignore_ascii_case("content-length") {
                    length = value.trim().parse().unwrap();
                }
            }
            let mut body = vec![0u8; length];
            reader.read_exact(&mut body).unwrap();
            let body: Value = serde_json::from_slice(&body).unwrap();
            assert!(
                request_line.starts_with("POST /api/internal/permission "),
                "{request_line}"
            );
            log.lock().unwrap().push((authorization, body));
            let (status, payload) = if decision.is_null() {
                ("403 Forbidden", json!({ "error": "forbidden" }).to_string())
            } else {
                ("200 OK", decision.to_string())
            };
            write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                payload.len()
            )
            .unwrap();
            stream.flush().unwrap();
        }
    });
    (base, seen)
}

fn roundtrip(
    stdin: &mut ChildStdin,
    lines: &mut Lines<BufReader<ChildStdout>>,
    msg: Value,
) -> Value {
    writeln!(stdin, "{msg}").unwrap();
    stdin.flush().unwrap();
    let line = lines
        .next()
        .expect("the bridge answers")
        .expect("the bridge's stdout is readable");
    serde_json::from_str(&line).unwrap()
}

fn tool_text(reply: &Value) -> Value {
    serde_json::from_str(reply["result"]["content"][0]["text"].as_str().unwrap()).unwrap()
}

#[test]
fn permission_bridge_forwards_tools_call_to_the_field_api() {
    let (base, seen) = field_stub(vec![
        json!({ "decision": "allow", "updatedInput": { "command": "ls -la" } }),
        json!({ "decision": "deny", "message": "outside scope" }),
        Value::Null,
    ]);
    let mut child = Command::new(env!("CARGO_BIN_EXE_knossos"))
        .arg(SUBCOMMAND)
        .env("FIELD_API", &base)
        .env("FIELD_SESSION", "s-1")
        .env("FIELD_INTERNAL_TOKEN", "cap-token")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();

    let init = roundtrip(
        &mut stdin,
        &mut lines,
        json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": { "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": { "name": "claude-code", "version": "test" } } }),
    );
    assert_eq!(init["id"], 1);
    assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(init["result"]["serverInfo"]["name"], "field");
    writeln!(
        stdin,
        "{}",
        json!({ "jsonrpc": "2.0", "method": "notifications/initialized" })
    )
    .unwrap();

    let list = roundtrip(
        &mut stdin,
        &mut lines,
        json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }),
    );
    assert_eq!(list["result"]["tools"][0]["name"], TOOL_NAME);

    let allowed = roundtrip(
        &mut stdin,
        &mut lines,
        json!({ "jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": { "name": TOOL_NAME, "arguments": { "tool_name": "Bash", "input": { "command": "ls" }, "tool_use_id": "toolu_1" } } }),
    );
    assert_eq!(allowed["id"], 3);
    assert_eq!(
        tool_text(&allowed),
        json!({ "behavior": "allow", "updatedInput": { "command": "ls -la" } })
    );

    let denied = roundtrip(
        &mut stdin,
        &mut lines,
        json!({ "jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": { "name": TOOL_NAME, "arguments": { "tool_name": "Edit", "input": { "file_path": "/etc/passwd" } } } }),
    );
    assert_eq!(
        tool_text(&denied),
        json!({ "behavior": "deny", "message": "outside scope" })
    );

    let refused = roundtrip(
        &mut stdin,
        &mut lines,
        json!({ "jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": { "name": TOOL_NAME, "arguments": { "tool_name": "Write", "input": {} } } }),
    );
    let text = tool_text(&refused);
    assert_eq!(
        text["behavior"], "deny",
        "a gate that cannot reach Field denies"
    );
    assert!(text["message"]
        .as_str()
        .unwrap()
        .contains("refused the request (403)"));

    let unsupported = roundtrip(
        &mut stdin,
        &mut lines,
        json!({ "jsonrpc": "2.0", "id": 6, "method": "prompts/list" }),
    );
    assert_eq!(unsupported["error"]["code"], -32603);

    drop(stdin);
    let status = child.wait().unwrap();
    assert!(
        status.success(),
        "the bridge exits cleanly when stdin closes: {status}"
    );

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 3);
    assert_eq!(seen[0].0, "Bearer cap-token");
    assert_eq!(
        seen[0].1,
        json!({ "sessionId": "s-1", "toolName": "Bash", "input": { "command": "ls" }, "toolUseId": "toolu_1" })
    );
    assert_eq!(seen[1].1["toolName"], "Edit");
    assert_eq!(seen[1].1["toolUseId"], Value::Null);
}

fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn process_generation_preserves_replacement_ownership() {
    let events = Arc::new(Mutex::new(Vec::<(String, Value)>::new()));
    let log = Arc::clone(&events);
    let sink: EventSink =
        Arc::new(move |kind: &str, data: Value| log.lock().unwrap().push((kind.to_string(), data)));
    // The bridge subcommand is a long-lived stdin reader: a stand-in for
    // `claude` that lives until it is killed or its stdin closes.
    let session = ClaudeSession::new(
        ClaudeOptions {
            id: "generation".into(),
            cwd: std::env::current_dir().unwrap(),
            binary: Some(PathBuf::from(env!("CARGO_BIN_EXE_knossos"))),
            args_override: Some(vec![SUBCOMMAND.into()]),
            ..Default::default()
        },
        sink,
    );
    let ended = || {
        events
            .lock()
            .unwrap()
            .iter()
            .filter(|(k, _)| k == "session.ended")
            .count()
    };

    session.start(None).unwrap();
    assert!(session.is_running());
    assert_eq!(session.owned_processes(), 1);
    assert!(session.start(None).unwrap_err().contains("already running"));

    assert!(session.pause());
    assert!(!session.is_running());
    assert_eq!(session.info().state, "paused");
    assert!(session.resume(None).unwrap());
    assert!(session.is_running(), "the replacement is owned immediately");
    assert!(session.owned_processes() >= 1);

    // The old process drains; its close must not clear or end the replacement.
    wait_until("the old process to close", || {
        session.owned_processes() == 1
    });
    std::thread::sleep(Duration::from_millis(100));
    assert!(
        session.is_running(),
        "old close cannot clear the replacement"
    );
    assert_eq!(
        ended(),
        0,
        "old close cannot terminate the replacement session"
    );
    assert!(
        session.send("still here"),
        "the replacement's stdin is live"
    );

    assert!(session.cancel());
    wait_until("the replacement to close", || {
        session.owned_processes() == 0
    });
    assert!(!session.is_running());
    assert_eq!(ended(), 1);
    let events = events.lock().unwrap();
    let last = events
        .iter()
        .rev()
        .find(|(k, _)| k == "session.ended")
        .unwrap();
    assert_eq!(last.1["reason"], "cancelled");
    assert!(events
        .iter()
        .any(|(k, d)| k == "session.state" && d["state"] == "paused"));
    assert!(events
        .iter()
        .any(|(k, d)| k == "session.state" && d["detail"] == "resumed"));
}
