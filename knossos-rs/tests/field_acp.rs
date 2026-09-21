//! The ACP adapter against a real ACP agent over stdio: the knossos binary's
//! own `acp` subcommand on its scripted mock engine (`KNOSSOS_SCRIPT`), so
//! no model, key or network is involved. The live half of
//! `field/server/test/acp-session.test.mjs`, with Knossos standing in for
//! the Node mock agent.

use knossos::field::acp_session::{AcpOptions, AcpSession};
use knossos::field::adapter::{Adapter, EventSink, CANONICAL_EVENTS};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

type Events = Arc<Mutex<Vec<(String, Value)>>>;

fn workspace() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir_all(dir.path().join("src")).expect("src");
    std::fs::write(
        dir.path().join("Cargo.toml"),
        "[package]\nname = \"acp-fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("manifest");
    std::fs::write(dir.path().join("src/lib.rs"), "pub fn one() -> u32 { 1 }\n").expect("lib");
    let root = std::fs::canonicalize(dir.path()).expect("canonicalize");
    (dir, root)
}

fn wait_until(what: &str, timeout: Duration, mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + timeout;
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(25));
    }
}

fn has(events: &Events, predicate: impl Fn(&str, &Value) -> bool) -> bool {
    events.lock().unwrap().iter().any(|(k, d)| predicate(k, d))
}

fn session(id: &str, root: &Path, script: &[&str]) -> (Arc<AcpSession>, Events) {
    let events: Events = Arc::new(Mutex::new(Vec::new()));
    let sink_events = Arc::clone(&events);
    let sink: EventSink = Arc::new(move |kind: &str, data: Value| {
        sink_events.lock().unwrap().push((kind.to_string(), data))
    });
    let s = AcpSession::new(
        AcpOptions {
            id: id.into(),
            agent_id: Some("a1".into()),
            name: Some("Ada".into()),
            role: Some("builder".into()),
            model: Some("mock-model".into()),
            endpoint_id: Some("acp-1".into()),
            cwd: root.to_path_buf(),
            workspace_id: Some("ws".into()),
            system_prompt: "BE CORRECT".into(),
            command: Some(env!("CARGO_BIN_EXE_knossos").into()),
            args: Some(
                [
                    "acp",
                    "--no-judge",
                    "--no-context",
                    "--max-steps",
                    "4",
                    "--target-steps",
                    "2",
                ]
                .iter()
                .map(|a| a.to_string())
                .collect(),
            ),
            launch_env: [("KNOSSOS_SCRIPT".to_string(), json!(script).to_string())]
                .into_iter()
                .collect(),
            ..Default::default()
        },
        sink,
    );
    (s, events)
}

#[test]
fn drives_the_knossos_binary_as_an_acp_agent_end_to_end() {
    let (_dir, root) = workspace();
    let (s, events) = session(
        "acp-e2e",
        &root,
        &["Here is what I found: the crate exposes one() returning 1."],
    );
    assert_eq!(s.capabilities()["kind"], "acp");

    s.start(Some("inspect the workspace and report".into()))
        .expect("start");
    assert!(s.is_running());
    assert_eq!(s.owned_processes(), 1);

    // Handshake: initialize + session/new -> ready + endpoint.routed.
    wait_until("ready state", Duration::from_secs(60), || {
        has(&events, |k, d| {
            k == "session.state" && d["state"] == "ready"
        })
    });
    assert!(has(&events, |k, d| k == "endpoint.routed"
        && d["endpointId"] == "acp-1"
        && d["reason"] == "ACP session/new"));
    assert!(
        s.acp_session_id().is_some(),
        "adapter captured the ACP-side session id"
    );

    // The queued orders become the first prompt and the turn round-trips.
    wait_until("first turn_complete", Duration::from_secs(120), || {
        has(&events, |k, _| k == "session.turn_complete")
    });
    let stderr_hint = || {
        events
            .lock()
            .unwrap()
            .iter()
            .map(|(k, d)| format!("{k}: {d}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert!(
        has(&events, |k, d| k == "session.message"
            && d["role"] == "user"
            && d["text"] == "inspect the workspace and report"),
        "user turn recorded\n{}",
        stderr_hint()
    );
    assert!(
        has(&events, |k, d| k == "session.message"
            && d["role"] == "assistant"
            && d["text"]
                .as_str()
                .is_some_and(|t| t.contains("Here is what I found"))),
        "agent_message_chunk -> session.message\n{}",
        stderr_hint()
    );
    let done = events
        .lock()
        .unwrap()
        .iter()
        .find(|(k, _)| k == "session.turn_complete")
        .map(|(_, d)| d.clone())
        .unwrap();
    assert_eq!(done["isError"], false, "{done}");
    assert!(done["stopReason"].is_string(), "{done}");
    assert!(has(&events, |k, d| k == "session.state" && d["state"] == "idle"));
    assert!(s.has_turn());
    assert_eq!(s.info().state, "idle");

    // Every event is canonical and stamped with the Field session id.
    {
        let ev = events.lock().unwrap();
        for (kind, data) in ev.iter() {
            assert!(CANONICAL_EVENTS.contains(&kind.as_str()), "{kind}");
            assert_eq!(data["sessionId"], "acp-e2e");
        }
    }

    // cancel() tells the agent, tears the process down and ends the session;
    // the waiter thread reaps the child and releases its capacity.
    assert!(s.cancel());
    assert!(has(&events, |k, d| k == "session.ended"
        && d["reason"] == "cancelled"));
    assert!(!s.is_running());
    wait_until("child reaped", Duration::from_secs(30), || {
        s.owned_processes() == 0
    });
    assert_eq!(s.info().state, "cancelled");
    assert_eq!(
        events
            .lock()
            .unwrap()
            .iter()
            .filter(|(k, _)| k == "session.ended")
            .count(),
        1,
        "the reaped process does not end the session a second time"
    );
}

#[test]
fn pause_and_resume_re_handshake_with_a_fresh_agent_process() {
    let (_dir, root) = workspace();
    let (s, events) = session("acp-pause", &root, &["Ready.", "Resumed."]);
    s.start(Some("stand by".into())).expect("start");
    wait_until("first turn", Duration::from_secs(120), || {
        has(&events, |k, _| k == "session.turn_complete")
    });

    assert!(s.pause());
    assert_eq!(s.info().state, "paused");
    assert!(!s.is_running());
    assert!(has(&events, |k, d| k == "session.state"
        && d["state"] == "paused"));
    wait_until("paused child reaped", Duration::from_secs(30), || {
        s.owned_processes() == 0
    });
    assert!(
        !has(&events, |k, _| k == "session.ended"),
        "a paused process exiting does not end the session"
    );

    // Resume spawns a new agent and re-runs the handshake.
    let ready_before = events
        .lock()
        .unwrap()
        .iter()
        .filter(|(k, d)| k == "session.state" && d["state"] == "ready")
        .count();
    assert_eq!(s.resume(None), Ok(true));
    assert!(s.is_running());
    wait_until("second ready", Duration::from_secs(60), || {
        events
            .lock()
            .unwrap()
            .iter()
            .filter(|(k, d)| k == "session.state" && d["state"] == "ready")
            .count()
            > ready_before
    });
    wait_until("second turn", Duration::from_secs(120), || {
        events
            .lock()
            .unwrap()
            .iter()
            .filter(|(k, _)| k == "session.turn_complete")
            .count()
            >= 2
    });
    assert!(has(&events, |k, d| {
        k == "session.message"
            && d["role"] == "user"
            && d["text"]
                .as_str()
                .is_some_and(|t| t.starts_with("Resume from the workspace"))
    }));
    s.cancel();
    wait_until("resumed child reaped", Duration::from_secs(30), || {
        s.owned_processes() == 0
    });
}

#[test]
fn a_missing_agent_binary_ends_the_session_with_an_error() {
    let (_dir, root) = workspace();
    let events: Events = Arc::new(Mutex::new(Vec::new()));
    let sink_events = Arc::clone(&events);
    let sink: EventSink = Arc::new(move |kind: &str, data: Value| {
        sink_events.lock().unwrap().push((kind.to_string(), data))
    });
    let s = AcpSession::new(
        AcpOptions {
            id: "acp-missing".into(),
            cwd: root,
            command: Some("definitely-not-an-acp-agent-binary".into()),
            args: Some(Vec::new()),
            ..Default::default()
        },
        sink,
    );
    s.start(None)
        .expect("a spawn failure is reported, not returned");
    assert!(!s.is_running());
    assert_eq!(s.info().state, "error");
    assert!(has(&events, |k, d| k == "session.ended"
        && d["reason"] == "error"
        && d["error"]
            .as_str()
            .is_some_and(|e| e.starts_with("ACP spawn failed"))));
}
