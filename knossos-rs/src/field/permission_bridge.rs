//! The Field permission gate as a stdio MCP server. Port of
//! `field/server/src/harness/permission-mcp.mjs`.
//!
//! Claude Code is pointed at it with `--permission-prompt-tool` and calls it
//! before any consequential action. Its only tool parks the request with the
//! Field server (`POST /api/internal/permission`, authorised by the session's
//! harness capability) and blocks until a human or the policy decides. stdout
//! carries JSON-RPC only; anything else would corrupt the protocol stream.
//! A gate that cannot reach Field denies: a permission gate that fails open
//! is not a permission gate.

use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::UnboundedReceiver;

use super::js::get_str;

/// The MCP server name Claude Code sees; tools are `mcp__<server>__<tool>`.
pub const SERVER_NAME: &str = "field";
/// The one tool the bridge serves.
pub const TOOL_NAME: &str = "approve";
/// What the registry passes as `--permission-prompt-tool`.
pub const PERMISSION_TOOL: &str = "mcp__field__approve";
/// The `knossos` subcommand that runs the bridge.
pub const SUBCOMMAND: &str = "field-permission-bridge";
pub const DEFAULT_API: &str = "http://127.0.0.1:7749";
const DEFAULT_PROTOCOL: &str = "2024-11-05";

/// What the bridge needs from its environment: `FIELD_API` (or
/// `FIELD_API_BASE`), `FIELD_SESSION`, `FIELD_INTERNAL_TOKEN`.
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    pub api: String,
    pub session: String,
    pub token: Option<String>,
}

impl BridgeConfig {
    pub fn from_env() -> BridgeConfig {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        BridgeConfig {
            api: var("FIELD_API_BASE")
                .or_else(|| var("FIELD_API"))
                .unwrap_or_else(|| DEFAULT_API.into()),
            session: var("FIELD_SESSION").unwrap_or_else(|| "unknown".into()),
            token: var("FIELD_INTERNAL_TOKEN"),
        }
    }
}

/// The `--mcp-config` document the registry hands Claude Code, exactly as
/// `registry.js` builds it: one server, `field`, running this bridge with
/// the session's API base, id and harness capability in its environment.
pub fn mcp_config(bridge_binary: &Path, api_base: &str, session_id: &str, token: &str) -> String {
    json!({
        "mcpServers": {
            SERVER_NAME: {
                "command": bridge_binary.to_string_lossy(),
                "args": [SUBCOMMAND],
                "env": {
                    "FIELD_API": api_base,
                    "FIELD_SESSION": session_id,
                    "FIELD_INTERNAL_TOKEN": token,
                },
            },
        },
    })
    .to_string()
}

/// Rewrite the capability in an existing config (a restart mints a fresh one).
pub fn mcp_config_with_token(config: &str, token: &str) -> Option<String> {
    let mut parsed: Value = serde_json::from_str(config).ok()?;
    let env = parsed
        .get_mut("mcpServers")?
        .get_mut(SERVER_NAME)?
        .get_mut("env")?;
    env.as_object_mut()?
        .insert("FIELD_INTERNAL_TOKEN".into(), Value::String(token.into()));
    Some(parsed.to_string())
}

pub fn initialize_result(params: Option<&Value>) -> Value {
    let protocol = params
        .and_then(|p| get_str(p, "protocolVersion"))
        .unwrap_or(DEFAULT_PROTOCOL);
    json!({
        "protocolVersion": protocol,
        "capabilities": { "tools": { "listChanged": false } },
        "serverInfo": { "name": SERVER_NAME, "version": env!("CARGO_PKG_VERSION") },
    })
}

pub fn tools_list() -> Value {
    json!({
        "tools": [{
            "name": TOOL_NAME,
            "description": "Ask the Field operator to approve a consequential tool call. Blocks until a human decides.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tool_name": { "type": "string" },
                    "input": { "type": "object" },
                    "tool_use_id": { "type": "string" },
                },
                "required": ["tool_name", "input"],
            },
        }],
    })
}

/// The decision in the shape Claude Code expects from a permission prompt tool.
pub fn decision_payload(decision: &Value, input: &Value) -> Value {
    if get_str(decision, "decision") == Some("allow") {
        json!({
            "behavior": "allow",
            "updatedInput": decision.get("updatedInput").cloned().unwrap_or_else(|| input.clone()),
        })
    } else {
        let message = get_str(decision, "message")
            .filter(|m| !m.is_empty())
            .unwrap_or("Denied by the Field operator.");
        json!({ "behavior": "deny", "message": message })
    }
}

pub fn unreachable_payload(error: &str) -> Value {
    json!({ "behavior": "deny", "message": format!("Field permission gate unreachable: {error}") })
}

/// Wrap a payload as an MCP tool result.
pub fn tool_result(payload: &Value) -> Value {
    json!({ "content": [{ "type": "text", "text": payload.to_string() }] })
}

fn response(id: &Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn failure(id: &Value, message: String) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": -32603, "message": message } })
}

/// What one inbound line asks for.
#[derive(Debug, Clone, PartialEq)]
pub enum Handled {
    /// Nothing to send back (a notification, or noise).
    Ignore,
    /// Send this response now.
    Reply(Value),
    /// A permission call: ask Field, then answer `id`.
    Ask { id: Value, arguments: Value },
}

/// Dispatch one JSON-RPC line, without performing the network call.
pub fn handle_line(line: &str) -> Handled {
    let text = line.trim();
    if text.is_empty() {
        return Handled::Ignore;
    }
    let Ok(msg) = serde_json::from_str::<Value>(text) else {
        return Handled::Ignore;
    };
    let id = msg.get("id").cloned();
    let method = get_str(&msg, "method").unwrap_or("");
    let params = msg.get("params");
    if method == "initialize" {
        return Handled::Reply(response(
            &id.unwrap_or(Value::Null),
            initialize_result(params),
        ));
    }
    let Some(id) = id else {
        return Handled::Ignore;
    };
    if method == "notifications/initialized" {
        return Handled::Ignore;
    }
    match method {
        "tools/list" => Handled::Reply(response(&id, tools_list())),
        "tools/call" => Handled::Ask {
            id,
            arguments: params
                .and_then(|p| p.get("arguments"))
                .cloned()
                .unwrap_or_else(|| json!({})),
        },
        other => Handled::Reply(failure(&id, format!("unsupported method: {other}"))),
    }
}

/// Park the request with the Field server; the answer is
/// `{ decision: 'allow' | 'deny', message?, updatedInput? }`.
pub async fn ask_operator(
    client: &reqwest::Client,
    config: &BridgeConfig,
    tool_name: &Value,
    input: &Value,
    tool_use_id: &Value,
) -> Result<Value, String> {
    let token = config
        .token
        .as_deref()
        .ok_or("internal Field authorization is not configured")?;
    let response = client
        .post(format!("{}/api/internal/permission", config.api))
        .header("authorization", format!("Bearer {token}"))
        .header("content-type", "application/json")
        .body(
            json!({
                "sessionId": config.session, "toolName": tool_name, "input": input, "toolUseId": tool_use_id,
            })
            .to_string(),
        )
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(format!(
            "field server refused the request ({})",
            response.status().as_u16()
        ));
    }
    response.json::<Value>().await.map_err(|e| e.to_string())
}

/// Answer one `tools/call` with the operator's decision.
pub async fn answer_call(
    client: &reqwest::Client,
    config: &BridgeConfig,
    id: &Value,
    arguments: &Value,
) -> Value {
    let input = arguments.get("input").cloned().unwrap_or(Value::Null);
    let tool_name = arguments.get("tool_name").cloned().unwrap_or(Value::Null);
    let tool_use_id = arguments.get("tool_use_id").cloned().unwrap_or(Value::Null);
    let payload = match ask_operator(client, config, &tool_name, &input, &tool_use_id).await {
        Ok(decision) => decision_payload(&decision, &input),
        Err(error) => unreachable_payload(&error),
    };
    response(id, tool_result(&payload))
}

fn send(out: &Arc<Mutex<dyn Write + Send>>, message: &Value) {
    if let Ok(mut out) = out.lock() {
        let _ = out.write_all(format!("{message}\n").as_bytes());
        let _ = out.flush();
    }
}

/// Serve the bridge over the given streams until the input closes. Each
/// permission call is answered from its own task, so one request waiting
/// on a human never blocks the protocol.
pub async fn serve(
    mut lines: UnboundedReceiver<String>,
    out: Arc<Mutex<dyn Write + Send>>,
    config: BridgeConfig,
) -> std::io::Result<()> {
    let client = reqwest::Client::new();
    let config = Arc::new(config);
    let mut in_flight = Vec::new();
    while let Some(line) = lines.recv().await {
        match handle_line(&line) {
            Handled::Ignore => {}
            Handled::Reply(message) => send(&out, &message),
            Handled::Ask { id, arguments } => {
                let client = client.clone();
                let config = Arc::clone(&config);
                let out = Arc::clone(&out);
                in_flight.push(tokio::spawn(async move {
                    let message = answer_call(&client, &config, &id, &arguments).await;
                    send(&out, &message);
                }));
            }
        }
        in_flight.retain(|task| !task.is_finished());
    }
    for task in in_flight {
        let _ = task.await;
    }
    Ok(())
}

/// `knossos field-permission-bridge`: stdio in, stdio out, config from the
/// environment.
pub async fn run_stdio() -> std::io::Result<()> {
    let out: Arc<Mutex<dyn Write + Send>> = Arc::new(Mutex::new(std::io::stdout()));
    // stdin is read on a plain thread; the async side only sees lines.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    serve(rx, out, BridgeConfig::from_env()).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcp_config_matches_registry_js() {
        let config = mcp_config(
            Path::new("/opt/knossos"),
            "http://127.0.0.1:7749",
            "s1",
            "tok",
        );
        let parsed: Value = serde_json::from_str(&config).unwrap();
        let field = &parsed["mcpServers"]["field"];
        assert_eq!(field["command"], "/opt/knossos");
        assert_eq!(field["args"], json!([SUBCOMMAND]));
        assert_eq!(field["env"]["FIELD_API"], "http://127.0.0.1:7749");
        assert_eq!(field["env"]["FIELD_SESSION"], "s1");
        assert_eq!(field["env"]["FIELD_INTERNAL_TOKEN"], "tok");
        let refreshed: Value =
            serde_json::from_str(&mcp_config_with_token(&config, "fresh").unwrap()).unwrap();
        assert_eq!(
            refreshed["mcpServers"]["field"]["env"]["FIELD_INTERNAL_TOKEN"],
            "fresh"
        );
        assert_eq!(
            refreshed["mcpServers"]["field"]["env"]["FIELD_SESSION"],
            "s1"
        );
        assert!(mcp_config_with_token("{}", "x").is_none());
    }

    #[test]
    fn protocol_dispatch_matches_permission_mcp_mjs() {
        assert_eq!(handle_line(""), Handled::Ignore);
        assert_eq!(handle_line("not json"), Handled::Ignore);
        let Handled::Reply(init) = handle_line(
            &json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": { "protocolVersion": "2025-06-18" } }).to_string(),
        ) else {
            panic!("initialize replies")
        };
        assert_eq!(init["id"], 1);
        assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(init["result"]["serverInfo"]["name"], "field");
        assert_eq!(
            init["result"]["capabilities"]["tools"]["listChanged"],
            false
        );
        let Handled::Reply(init) =
            handle_line(&json!({ "id": 2, "method": "initialize" }).to_string())
        else {
            panic!()
        };
        assert_eq!(init["result"]["protocolVersion"], DEFAULT_PROTOCOL);
        assert_eq!(
            handle_line(&json!({ "method": "notifications/initialized" }).to_string()),
            Handled::Ignore
        );
        assert_eq!(
            handle_line(&json!({ "method": "tools/list" }).to_string()),
            Handled::Ignore,
            "a request without an id is ignored, as in the Node bridge"
        );
        let Handled::Reply(list) =
            handle_line(&json!({ "id": 3, "method": "tools/list" }).to_string())
        else {
            panic!()
        };
        assert_eq!(list["result"]["tools"][0]["name"], TOOL_NAME);
        assert_eq!(
            list["result"]["tools"][0]["inputSchema"]["required"],
            json!(["tool_name", "input"])
        );
        assert_eq!(
            handle_line(&json!({ "id": 4, "method": "tools/call", "params": { "arguments": { "tool_name": "Bash", "input": { "command": "ls" } } } }).to_string()),
            Handled::Ask { id: json!(4), arguments: json!({ "tool_name": "Bash", "input": { "command": "ls" } }) }
        );
        let Handled::Reply(fail) =
            handle_line(&json!({ "id": 5, "method": "resources/list" }).to_string())
        else {
            panic!()
        };
        assert_eq!(fail["error"]["code"], -32603);
        assert_eq!(
            fail["error"]["message"],
            "unsupported method: resources/list"
        );
    }

    #[test]
    fn decisions_take_the_shape_claude_code_expects() {
        let input = json!({ "command": "ls" });
        assert_eq!(
            decision_payload(&json!({ "decision": "allow" }), &input),
            json!({ "behavior": "allow", "updatedInput": { "command": "ls" } })
        );
        assert_eq!(
            decision_payload(
                &json!({ "decision": "allow", "updatedInput": { "command": "ls -la" } }),
                &input
            ),
            json!({ "behavior": "allow", "updatedInput": { "command": "ls -la" } })
        );
        assert_eq!(
            decision_payload(&json!({ "decision": "deny" }), &input),
            json!({ "behavior": "deny", "message": "Denied by the Field operator." })
        );
        assert_eq!(
            decision_payload(
                &json!({ "decision": "deny", "message": "outside scope" }),
                &input
            ),
            json!({ "behavior": "deny", "message": "outside scope" })
        );
        let unreachable = unreachable_payload("boom");
        assert_eq!(unreachable["behavior"], "deny");
        assert!(unreachable["message"].as_str().unwrap().contains("boom"));
        let wrapped = tool_result(&unreachable);
        assert_eq!(wrapped["content"][0]["type"], "text");
        let text: Value =
            serde_json::from_str(wrapped["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(text, unreachable);
    }

    #[tokio::test]
    async fn a_missing_token_denies_without_a_network_call() {
        let config = BridgeConfig {
            api: "http://127.0.0.1:1".into(),
            session: "s".into(),
            token: None,
        };
        let answer = answer_call(
            &reqwest::Client::new(),
            &config,
            &json!(9),
            &json!({ "tool_name": "Edit", "input": {} }),
        )
        .await;
        assert_eq!(answer["id"], 9);
        let text: Value =
            serde_json::from_str(answer["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(text["behavior"], "deny");
        assert!(text["message"].as_str().unwrap().contains("not configured"));
    }
}
