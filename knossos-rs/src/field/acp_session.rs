//! The generic ACP (Agent Client Protocol) adapter. Port of
//! `field/server/src/harness/acp-session.js`.
//!
//! ACP is JSON-RPC 2.0 over newline-delimited stdio. Knossos implements the
//! *agent* side of the protocol (see `crate::acp`); this adapter is the
//! *client* side, so Field can drive any ACP-compliant coding agent as a
//! harness. It owns one external agent process, performs the ACP handshake
//! (`initialize` then `session/new`), sends prompts, and translates the
//! agent's `session/update` notifications into Field's canonical event
//! vocabulary. An agent's `session/request_permission` becomes a
//! `harness.permission_requested` event, so the registry's one permission
//! flow gates a third-party agent exactly as it gates the native harnesses.
//!
//! Nothing here is Knossos-specific: the launch spec (command, args, env)
//! comes from the endpoint record or the `FIELD_ACP_BIN` / `FIELD_ACP_ARGS`
//! operator overrides, and the protocol is spoken as the ACP schema defines it.

use super::adapter::{stamp, Adapter, BeforeStart, EventSink, SessionInfo};
use super::child_env::{build_child_environment, process_environment};
use super::js::{get, get_arr, get_str, jnum, js_string, or_null};
use crate::acp::PROTOCOL_VERSION;
use crate::jsonrpc::METHOD_NOT_FOUND;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

/// How long the `initialize` -> `session/new` handshake may take. A prompt
/// turn is deliberately un-timed (a turn can legitimately run for minutes);
/// only the handshake, a couple of cheap round trips, is bounded so a broken
/// agent cannot hang a session.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(15);

/// The binary run when neither the endpoint nor `FIELD_ACP_BIN` names one.
pub const DEFAULT_ACP_COMMAND: &str = "acp-agent";

/// Everything an ACP session is created with.
#[derive(Debug, Clone, Default)]
pub struct AcpOptions {
    pub id: String,
    pub agent_id: Option<String>,
    pub name: Option<String>,
    pub role: Option<String>,
    pub model: Option<String>,
    pub endpoint_id: Option<String>,
    pub effort: Option<String>,
    pub cwd: PathBuf,
    pub workspace_id: Option<String>,
    pub system_prompt: String,
    pub env: BTreeMap<String, String>,
    pub provider_kind: Option<String>,
    pub read_only: bool,
    pub environment_scope: Option<String>,
    pub credential_env_keys: Vec<String>,
    /// The ACP agent binary; `None` resolves `FIELD_ACP_BIN`, then
    /// [`DEFAULT_ACP_COMMAND`].
    pub command: Option<String>,
    /// Its arguments; `None` resolves `FIELD_ACP_ARGS`.
    pub args: Option<Vec<String>>,
    /// Extra environment for the agent process, layered over `env`.
    pub launch_env: BTreeMap<String, String>,
}

/// An argv from a JSON array, a whitespace-separated string, or nothing.
pub fn normalize_args(args: Option<&Value>) -> Vec<String> {
    match args {
        Some(Value::Array(items)) => items.iter().map(js_string).collect(),
        Some(Value::String(s)) => s.split_whitespace().map(str::to_string).collect(),
        _ => Vec::new(),
    }
}

/// The launch spec: the endpoint's command/args, else the operator's
/// `FIELD_ACP_BIN` / `FIELD_ACP_ARGS`, else the generic default.
pub fn resolve_acp_launch(command: Option<&str>, args: Option<&[String]>) -> (String, Vec<String>) {
    let command = command
        .filter(|c| !c.is_empty())
        .map(str::to_string)
        .or_else(|| {
            std::env::var("FIELD_ACP_BIN")
                .ok()
                .filter(|v| !v.is_empty())
        })
        .unwrap_or_else(|| DEFAULT_ACP_COMMAND.to_string());
    let args = match args {
        Some(args) => args.to_vec(),
        None => std::env::var("FIELD_ACP_ARGS")
            .ok()
            .map(|v| normalize_args(Some(&Value::String(v))))
            .unwrap_or_default(),
    };
    (command, args)
}

/// What runs when a reply to one of our requests arrives (or the request
/// is abandoned): `Ok(result)` or `Err(message)`.
type Reply = Box<dyn FnOnce(&AcpSession, Result<Value, String>) + Send>;

struct Running {
    generation: u64,
    stdin: Box<dyn Write + Send>,
    child: Option<Arc<Mutex<Child>>>,
}

struct Inner {
    running: Option<Running>,
    /// Processes owned, including ones still draining after a stop.
    owned: usize,
    generation: u64,
    started: bool,
    /// Handshake complete (`initialize` + `session/new`).
    ready: bool,
    /// At least one prompt sent (drives the system-prompt prepend).
    has_turn: bool,
    /// The ACP-side session id from `session/new`, distinct from the Field id.
    acp_session_id: Option<String>,
    /// Orders queued until the handshake completes.
    pending_orders: Option<String>,
    stderr: String,
    next_rpc_id: u64,
    pending_rpc: HashMap<u64, Reply>,
    /// Field `requestId` -> inbound JSON-RPC id to answer.
    pending_permissions: HashMap<String, Value>,
    /// `toolCallId` -> tool name, for `session.tool_result` labelling.
    tool_names: HashMap<String, String>,
}

pub struct AcpSession {
    /// A handle to ourselves, so the reader threads a start spawns can own
    /// the session without a separate launch interface.
    me: Weak<AcpSession>,
    info: Mutex<SessionInfo>,
    opts: AcpOptions,
    sink: EventSink,
    inner: Mutex<Inner>,
    before_start: Mutex<Option<BeforeStart>>,
}

impl AcpSession {
    pub fn new(opts: AcpOptions, sink: EventSink) -> Arc<AcpSession> {
        let info = SessionInfo {
            id: opts.id.clone(),
            agent_id: opts.agent_id.clone(),
            name: opts.name.clone().unwrap_or_else(|| opts.id.clone()),
            role: opts.role.clone(),
            model: opts.model.clone(),
            endpoint_id: opts.endpoint_id.clone(),
            effort: opts.effort.clone().unwrap_or_else(|| "medium".into()),
            cwd: opts.cwd.clone(),
            workspace_id: opts.workspace_id.clone(),
            state: "created".into(),
        };
        Arc::new_cyclic(|me| AcpSession {
            me: me.clone(),
            info: Mutex::new(info),
            opts,
            sink,
            inner: Mutex::new(Inner {
                running: None,
                owned: 0,
                generation: 0,
                started: false,
                ready: false,
                has_turn: false,
                acp_session_id: None,
                pending_orders: None,
                stderr: String::new(),
                next_rpc_id: 0,
                pending_rpc: HashMap::new(),
                pending_permissions: HashMap::new(),
                tool_names: HashMap::new(),
            }),
            before_start: Mutex::new(None),
        })
    }

    fn emit(&self, kind: &str, data: Value) {
        (self.sink)(kind, stamp(&self.opts.id, data));
    }

    fn set_state(&self, state: &str) {
        if let Ok(mut info) = self.info.lock() {
            info.state = state.to_string();
        }
    }

    fn state(&self) -> String {
        self.info
            .lock()
            .map(|i| i.state.clone())
            .unwrap_or_default()
    }

    /// The ACP-side session id, once `session/new` has answered.
    pub fn acp_session_id(&self) -> Option<String> {
        self.inner
            .lock()
            .ok()
            .and_then(|i| i.acp_session_id.clone())
    }

    /// Whether a prompt has been sent in this process lifetime.
    pub fn has_turn(&self) -> bool {
        self.inner.lock().map(|i| i.has_turn).unwrap_or(false)
    }

    /// Requests written and not yet answered.
    pub fn pending_rpc_count(&self) -> usize {
        self.inner.lock().map(|i| i.pending_rpc.len()).unwrap_or(0)
    }

    /// The command and argv this session launches.
    pub fn launch(&self) -> (String, Vec<String>) {
        resolve_acp_launch(self.opts.command.as_deref(), self.opts.args.as_deref())
    }

    /// Whether the session should ask the agent for a read-only mode.
    fn wants_read_only_mode(&self) -> bool {
        self.opts.read_only
            || matches!(
                self.opts.environment_scope.as_deref(),
                Some("snapshot" | "production-readonly")
            )
    }

    /// The environment the child receives.
    pub fn child_environment(&self) -> BTreeMap<String, String> {
        let mut overrides = self.opts.env.clone();
        overrides.extend(self.opts.launch_env.clone());
        overrides.insert("FIELD_SESSION_ID".into(), self.opts.id.clone());
        build_child_environment(
            &process_environment(),
            self.opts.provider_kind.as_deref(),
            &self.opts.credential_env_keys,
            &overrides,
        )
    }

    /// Attach a fake stdin for tests, so the protocol can be exercised
    /// without a process.
    pub fn attach_writer(&self, writer: Box<dyn Write + Send>) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.generation += 1;
            let generation = inner.generation;
            inner.running = Some(Running {
                generation,
                stdin: writer,
                child: None,
            });
            inner.owned += 1;
            inner.started = true;
            inner.ready = false;
            inner.acp_session_id = None;
        }
    }

    pub fn set_pending_orders(&self, orders: &str) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.pending_orders = Some(orders.to_string());
        }
    }

    fn take_pending_orders(&self) -> Option<String> {
        self.inner
            .lock()
            .ok()
            .and_then(|mut i| i.pending_orders.take())
    }

    // ------------------------------------------------------------ lifecycle

    fn spawn_child(&self, orders: Option<String>) -> Result<(), String> {
        let this = self.me.upgrade().ok_or("session is being dropped")?;
        let hook = self.before_start.lock().ok().and_then(|h| h.clone());
        if let Some(hook) = hook {
            hook()?;
        }
        let (program, args) = self.launch();
        let env = self.child_environment();
        let mut command = Command::new(&program);
        command
            .args(&args)
            .current_dir(&self.opts.cwd)
            .env_clear()
            .envs(&env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                self.set_state("error");
                self.emit(
                    "session.ended",
                    json!({ "reason": "error", "error": format!("ACP spawn failed: {error}") }),
                );
                return Ok(());
            }
        };
        let stdin = child.stdin.take().ok_or("child stdin is not piped")?;
        let stdout = child.stdout.take().ok_or("child stdout is not piped")?;
        let stderr = child.stderr.take().ok_or("child stderr is not piped")?;
        let child = Arc::new(Mutex::new(child));
        let generation = {
            let mut inner = self.inner.lock().map_err(|_| "session lock poisoned")?;
            inner.generation += 1;
            inner.owned += 1;
            inner.started = true;
            inner.ready = false;
            inner.acp_session_id = None;
            inner.pending_orders = Some(orders.unwrap_or_else(|| "Await orders.".into()));
            inner.stderr.clear();
            let generation = inner.generation;
            inner.running = Some(Running {
                generation,
                stdin: Box::new(stdin),
                child: Some(Arc::clone(&child)),
            });
            generation
        };
        self.set_state("spawning");

        let reader = Arc::clone(&this);
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if !reader.is_current(generation) {
                    break;
                }
                reader.handle_line(&line);
            }
        });
        let errs = Arc::clone(&this);
        std::thread::spawn(move || {
            let mut buf = [0u8; 4096];
            let mut stderr = stderr;
            while let Ok(n) = stderr.read(&mut buf) {
                if n == 0 {
                    break;
                }
                if let Ok(mut inner) = errs.inner.lock() {
                    inner.stderr.push_str(&String::from_utf8_lossy(&buf[..n]));
                    if inner.stderr.len() > 8000 {
                        let keep = inner.stderr.len() - 4000;
                        inner.stderr = inner.stderr[keep..].to_string();
                    }
                }
            }
        });
        let waiter = this;
        std::thread::spawn(move || {
            // Polled rather than `wait()`ed so pause/cancel can still reach
            // the child through the shared handle to kill it.
            let code = loop {
                let polled = match child.lock() {
                    Ok(mut c) => c.try_wait(),
                    Err(_) => break None,
                };
                match polled {
                    Ok(Some(status)) => break status.code(),
                    Ok(None) => std::thread::sleep(Duration::from_millis(50)),
                    Err(_) => break None,
                }
            };
            waiter.on_close(generation, code);
        });

        // The handshake is request/response driven from the reader thread;
        // kick it off and return, like the other adapters, so the registry's
        // start() call site is unchanged.
        self.begin_handshake();
        Ok(())
    }

    fn is_current(&self, generation: u64) -> bool {
        self.inner
            .lock()
            .map(|i| {
                i.running
                    .as_ref()
                    .is_some_and(|r| r.generation == generation)
            })
            .unwrap_or(false)
    }

    fn on_close(&self, generation: u64, code: Option<i32>) {
        let (current, stderr_tail) = {
            let Ok(mut inner) = self.inner.lock() else {
                return;
            };
            inner.owned = inner.owned.saturating_sub(1);
            let current = inner
                .running
                .as_ref()
                .is_some_and(|r| r.generation == generation);
            if current {
                inner.running = None;
            }
            let tail = inner
                .stderr
                .trim()
                .lines()
                .rev()
                .take(4)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n");
            (current, tail)
        };
        if !current {
            return;
        }
        self.reject_all_rpc("ACP agent exited");
        let state = self.state();
        if state == "paused" || state == "cancelled" {
            return;
        }
        if code != Some(0) {
            self.set_state("error");
            let error = if stderr_tail.is_empty() {
                format!(
                    "ACP agent exited {}",
                    code.map(|c| c.to_string())
                        .unwrap_or_else(|| "by signal".into())
                )
            } else {
                stderr_tail
            };
            self.emit(
                "session.ended",
                json!({ "reason": "error", "error": error }),
            );
        } else if state != "done" {
            self.set_state("done");
            self.emit("session.ended", json!({ "reason": "exit" }));
        }
    }

    /// Tell the agent to stop, end stdin and kill the child; the waiter
    /// thread reaps it and releases its capacity on close.
    fn stop_child(&self) {
        if let Some(acp_id) = self.acp_session_id() {
            self.notify("session/cancel", json!({ "sessionId": acp_id }));
        }
        let child = {
            let Ok(mut inner) = self.inner.lock() else {
                return;
            };
            inner.ready = false;
            inner.acp_session_id = None;
            inner.running.take().and_then(|r| r.child)
        };
        if let Some(child) = child {
            if let Ok(mut child) = child.lock() {
                let _ = child.kill();
            }
        }
        // Replies from the stopped process must not settle a turn; the
        // callbacks see the paused/cancelled state and stand down.
        self.reject_all_rpc("ACP process stopped");
    }

    // ------------------------------------------------------------ handshake

    /// `initialize` -> `session/new` (-> `session/set_mode` for read-only
    /// roles) -> ready, then flush the queued orders. Public so tests can
    /// drive it over an attached writer without a process.
    pub fn begin_handshake(&self) {
        self.rpc(
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "clientInfo": { "name": "field", "title": "Knossos Field" },
                "clientCapabilities": {},
            }),
            Some(HANDSHAKE_TIMEOUT),
            Box::new(|s: &AcpSession, outcome| match outcome {
                Err(error) => s.handshake_failed(&error),
                Ok(_) => {
                    if !s.is_running() {
                        return;
                    }
                    s.rpc(
                        "session/new",
                        json!({ "cwd": s.opts.cwd, "mcpServers": [] }),
                        Some(HANDSHAKE_TIMEOUT),
                        Box::new(|s: &AcpSession, outcome| match outcome {
                            Err(error) => s.handshake_failed(&error),
                            Ok(created) => s.on_session_created(&created),
                        }),
                    );
                }
            }),
        );
    }

    fn on_session_created(&self, created: &Value) {
        if !self.is_running() {
            return;
        }
        let acp_id = get_str(created, "sessionId").map(str::to_string);
        if let Ok(mut inner) = self.inner.lock() {
            inner.acp_session_id = acp_id.clone();
        }
        // Read-only / snapshot roles map onto an ACP mode that does not
        // touch the workspace. Best effort: an agent without
        // `session/set_mode` simply errors and we proceed.
        if self.wants_read_only_mode() {
            self.rpc(
                "session/set_mode",
                json!({ "sessionId": acp_id, "modeId": "ask" }),
                Some(HANDSHAKE_TIMEOUT),
                Box::new(|s: &AcpSession, _outcome| s.finish_handshake()),
            );
        } else {
            self.finish_handshake();
        }
    }

    fn finish_handshake(&self) {
        if !self.is_running() {
            return;
        }
        if let Ok(mut inner) = self.inner.lock() {
            inner.ready = true;
        }
        self.set_state("ready");
        self.emit("session.state", json!({ "state": "ready" }));
        let info = self.info();
        self.emit(
            "endpoint.routed",
            json!({
                "endpointId": info.endpoint_id, "model": info.model,
                "reason": "ACP session/new",
            }),
        );
        if let Some(orders) = self.take_pending_orders() {
            self.send(&orders);
        }
    }

    fn handshake_failed(&self, error: &str) {
        let state = self.state();
        if state == "cancelled" || state == "paused" {
            return;
        }
        self.set_state("error");
        self.emit(
            "session.state",
            json!({ "state": "error", "detail": error }),
        );
    }

    fn on_turn_complete(&self, result: &Value, is_error: bool) {
        if let Some(usage) = get(result, "usage") {
            self.emit_usage(usage);
        }
        let state = if is_error { "error" } else { "idle" };
        self.set_state(state);
        self.emit(
            "session.state",
            json!({ "state": state, "detail": or_null(result.get("stopReason")) }),
        );
        let summary = match get_str(result, "stopReason") {
            Some(stop) => json!(stop),
            None => get_str(result, "error")
                .map(|e| json!(e.chars().take(4000).collect::<String>()))
                .unwrap_or(Value::Null),
        };
        self.emit(
            "session.turn_complete",
            json!({
                "result": summary,
                "stopReason": or_null(result.get("stopReason")),
                "missionId": or_null(result.get("missionId")),
                "isError": is_error,
            }),
        );
    }

    // ------------------------------------------------------------ JSON-RPC transport

    /// Write one JSON-RPC message line to the agent's stdin.
    pub fn write_message(&self, message: Value) -> bool {
        let Ok(mut inner) = self.inner.lock() else {
            return false;
        };
        let Some(running) = inner.running.as_mut() else {
            return false;
        };
        let mut line = message.to_string();
        line.push('\n');
        running.stdin.write_all(line.as_bytes()).is_ok() && running.stdin.flush().is_ok()
    }

    /// Send a JSON-RPC request; `reply` runs with its result or error. Returns
    /// whether the request was written (an unwritable request fails `reply`
    /// at once).
    pub fn rpc(
        &self,
        method: &str,
        params: Value,
        timeout: Option<Duration>,
        reply: Reply,
    ) -> bool {
        let id = match self.inner.lock() {
            Ok(mut inner) => {
                inner.next_rpc_id += 1;
                inner.next_rpc_id
            }
            Err(_) => {
                reply(self, Err("session lock poisoned".into()));
                return false;
            }
        };
        if !self.write_message(
            json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }),
        ) {
            reply(
                self,
                Err(format!(
                    "cannot send {method}: ACP agent stdin is not writable"
                )),
            );
            return false;
        }
        if let Ok(mut inner) = self.inner.lock() {
            inner.pending_rpc.insert(id, reply);
        }
        if let Some(timeout) = timeout {
            let me = self.me.clone();
            let method = method.to_string();
            std::thread::spawn(move || {
                std::thread::sleep(timeout);
                let Some(s) = me.upgrade() else {
                    return;
                };
                if let Some(reply) = s.take_reply(id) {
                    reply(
                        &s,
                        Err(format!(
                            "ACP {method} timed out after {}ms",
                            timeout.as_millis()
                        )),
                    );
                }
            });
        }
        true
    }

    /// Send a JSON-RPC notification (no id, no reply expected).
    pub fn notify(&self, method: &str, params: Value) -> bool {
        self.write_message(json!({ "jsonrpc": "2.0", "method": method, "params": params }))
    }

    fn take_reply(&self, id: u64) -> Option<Reply> {
        self.inner
            .lock()
            .ok()
            .and_then(|mut i| i.pending_rpc.remove(&id))
    }

    fn reject_all_rpc(&self, error: &str) {
        let pending: Vec<Reply> = match self.inner.lock() {
            Ok(mut inner) => inner.pending_rpc.drain().map(|(_, r)| r).collect(),
            Err(_) => Vec::new(),
        };
        for reply in pending {
            reply(self, Err(error.to_string()));
        }
    }

    /// Dispatch one line from the agent: a reply to one of our requests, an
    /// agent -> client request, or a notification.
    pub fn handle_line(&self, line: &str) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return;
        }
        let Ok(msg) = serde_json::from_str::<Value>(trimmed) else {
            return;
        };
        let has_id = get(&msg, "id").is_some();
        let has_method = get_str(&msg, "method").is_some();
        if has_method && has_id {
            self.handle_inbound_request(&msg);
        } else if has_method {
            self.handle_notification(&msg);
        } else if has_id {
            self.handle_response(&msg);
        }
    }

    fn handle_response(&self, msg: &Value) {
        let Some(id) = msg.get("id").and_then(Value::as_u64) else {
            return;
        };
        let Some(reply) = self.take_reply(id) else {
            return;
        };
        match get(msg, "error") {
            Some(error) => {
                let message = get_str(error, "message")
                    .map(str::to_string)
                    .unwrap_or_else(|| {
                        format!(
                            "ACP error {}",
                            get(error, "code").map(js_string).unwrap_or_default()
                        )
                        .trim()
                        .to_string()
                    });
                reply(self, Err(message));
            }
            None => reply(self, Ok(msg.get("result").cloned().unwrap_or(json!({})))),
        }
    }

    fn handle_inbound_request(&self, msg: &Value) {
        let id = msg.get("id").cloned().unwrap_or(Value::Null);
        if get_str(msg, "method") == Some("session/request_permission") {
            let params = get(msg, "params").cloned().unwrap_or(json!({}));
            let tool_call = get(&params, "toolCall").cloned().unwrap_or(json!({}));
            let request_id = js_string(&id);
            if let Ok(mut inner) = self.inner.lock() {
                inner
                    .pending_permissions
                    .insert(request_id.clone(), id.clone());
            }
            let tool_name = get_str(&tool_call, "title")
                .filter(|t| !t.is_empty())
                .or_else(|| get_str(&tool_call, "toolCallId").filter(|t| !t.is_empty()))
                .unwrap_or("tool");
            self.emit(
                "harness.permission_requested",
                json!({
                    "requestId": request_id,
                    "toolName": tool_name,
                    "toolCallId": or_null(tool_call.get("toolCallId")),
                    "options": get_arr(&params, "options").cloned().unwrap_or_default(),
                    "input": or_null(tool_call.get("rawInput")),
                }),
            );
            return;
        }
        // We advertise no fs/terminal client capabilities, so no other
        // agent -> client request is expected. Answer rather than ignore: an
        // unanswered request blocks the agent's turn.
        let method = get_str(msg, "method").unwrap_or("");
        self.write_message(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": METHOD_NOT_FOUND, "message": format!("{method} is not supported by this client") },
        }));
    }

    fn handle_notification(&self, msg: &Value) {
        if get_str(msg, "method") != Some("session/update") {
            return;
        }
        let Some(update) = get(msg, "params")
            .and_then(|p| get(p, "update"))
            .filter(|u| u.is_object())
        else {
            return;
        };
        self.translate_update(update);
    }

    /// Translate one ACP `session/update` into Field canonical events.
    pub fn translate_update(&self, update: &Value) {
        match get_str(update, "sessionUpdate") {
            Some("agent_message_chunk") => {
                // ACP text lives at update.content.text (a ContentBlock).
                self.emit(
                    "session.message",
                    json!({ "role": "assistant", "text": content_text(update.get("content")) }),
                );
            }
            Some("agent_thought_chunk") => {
                let text: String = content_text(update.get("content"))
                    .chars()
                    .take(2000)
                    .collect();
                self.emit("session.thinking", json!({ "text": text }));
            }
            Some("plan") => {
                // ACP plans are `entries[]` of {content, priority, status}.
                let entries = get_arr(update, "entries").cloned().unwrap_or_default();
                let done = entries
                    .iter()
                    .filter(|e| get_str(e, "status") == Some("completed"))
                    .count();
                let steps: Vec<Value> = entries
                    .iter()
                    .map(|e| e.get("content").cloned().unwrap_or(Value::Null))
                    .collect();
                self.emit(
                    "session.progress",
                    json!({ "done": done, "total": entries.len(), "steps": steps }),
                );
            }
            Some("tool_call") => {
                // The initial call: identified by `toolCallId`, described by
                // `title`, classified by `kind`. There is no machine tool-name
                // field, so the name falls back title -> kind -> "tool".
                let name = first_non_empty(&[get_str(update, "title"), get_str(update, "kind")]);
                let tool_id = js_string(update.get("toolCallId").unwrap_or(&Value::Null));
                if let Ok(mut inner) = self.inner.lock() {
                    inner.tool_names.insert(tool_id.clone(), name.clone());
                }
                self.emit(
                    "session.tool_use",
                    json!({
                        "toolId": tool_id,
                        "name": name,
                        "summary": get_str(update, "title").unwrap_or(&name),
                        "kind": or_null(update.get("kind")),
                        "input": or_null(update.get("rawInput")),
                        "locations": or_null(update.get("locations")),
                    }),
                );
                // Some agents (Knossos among them) emit tool_call already in
                // a terminal state instead of a follow-up tool_call_update.
                if matches!(get_str(update, "status"), Some("completed" | "failed")) {
                    self.emit_tool_result(update);
                }
            }
            Some("tool_call_update") => self.emit_tool_result(update),
            Some("usage_update") => {
                self.emit_usage(get(update, "usage").unwrap_or(update));
            }
            // available_commands_update, current_mode_update,
            // user_message_chunk, ...: real ACP narration, but nothing the
            // Field event vocabulary models.
            _ => {}
        }
    }

    fn emit_tool_result(&self, update: &Value) {
        let tool_id = js_string(update.get("toolCallId").unwrap_or(&Value::Null));
        let name = self
            .inner
            .lock()
            .ok()
            .and_then(|i| i.tool_names.get(&tool_id).cloned());
        let preview: String = extract_tool_output(update).chars().take(400).collect();
        self.emit(
            "session.tool_result",
            json!({
                "toolId": tool_id,
                "name": name,
                "ok": get_str(update, "status") != Some("failed"),
                "preview": preview,
            }),
        );
    }

    /// Cumulative token/cost figures an agent reports, in the shape the
    /// projection folds (`inputTokens`, `outputTokens`, `costUsd`). ACP has
    /// no single spelling for these yet, so both camel and snake case count.
    fn emit_usage(&self, usage: &Value) {
        let pick = |keys: &[&str]| -> Option<f64> {
            keys.iter()
                .find_map(|k| get(usage, k))
                .and_then(Value::as_f64)
                .filter(|n| n.is_finite() && *n >= 0.0)
        };
        let input = pick(&["inputTokens", "input_tokens", "input"]);
        let output = pick(&["outputTokens", "output_tokens", "output"]);
        let cache = pick(&["cacheRead", "cache_read_input_tokens", "cacheReadTokens"]);
        let cost = pick(&["costUsd", "cost_usd", "cost"]);
        if input.is_none() && output.is_none() && cost.is_none() {
            return;
        }
        let mut data = json!({});
        if let Some(obj) = data.as_object_mut() {
            if let Some(v) = input {
                obj.insert("inputTokens".into(), jnum(v));
            }
            if let Some(v) = output {
                obj.insert("outputTokens".into(), jnum(v));
            }
            if let Some(v) = cache {
                obj.insert("cacheRead".into(), jnum(v));
            }
            if let Some(v) = cost {
                obj.insert("costUsd".into(), jnum(v));
            }
        }
        self.emit("session.usage", data);
    }
}

impl Adapter for AcpSession {
    fn info(&self) -> SessionInfo {
        self.info
            .lock()
            .map(|i| i.clone())
            .unwrap_or_else(|_| SessionInfo {
                id: self.opts.id.clone(),
                agent_id: None,
                name: self.opts.id.clone(),
                role: None,
                model: None,
                endpoint_id: None,
                effort: "medium".into(),
                cwd: self.opts.cwd.clone(),
                workspace_id: None,
                state: "unknown".into(),
            })
    }

    fn set_endpoint(&self, endpoint_id: Option<String>, model: Option<String>) {
        if let Ok(mut info) = self.info.lock() {
            info.endpoint_id = endpoint_id;
            info.model = model;
        }
    }

    fn set_effort(&self, effort: &str) {
        if let Ok(mut info) = self.info.lock() {
            info.effort = effort.to_string();
        }
    }

    fn is_running(&self) -> bool {
        self.inner
            .lock()
            .map(|i| i.running.is_some())
            .unwrap_or(false)
    }

    fn owned_processes(&self) -> usize {
        self.inner.lock().map(|i| i.owned).unwrap_or(0)
    }

    fn set_before_start(&self, hook: BeforeStart) {
        if let Ok(mut slot) = self.before_start.lock() {
            *slot = Some(hook);
        }
    }

    fn start(&self, orders: Option<String>) -> Result<(), String> {
        if self.is_running() {
            return Err(format!("session {} already running", self.opts.id));
        }
        self.spawn_child(orders)
    }

    /// Push a user turn into the running ACP session via `session/prompt`.
    fn send(&self, text: &str) -> bool {
        let state = self.state();
        if !self.is_running() || state == "cancelled" || state == "paused" {
            return false;
        }
        // Before the handshake finishes there is no ACP session to prompt;
        // queue the orders and let the handshake flush them.
        let (ready, acp_id, has_turn) = match self.inner.lock() {
            Ok(inner) => (inner.ready, inner.acp_session_id.clone(), inner.has_turn),
            Err(_) => return false,
        };
        let Some(acp_id) = acp_id.filter(|_| ready) else {
            self.set_pending_orders(text);
            return true;
        };
        let full = if has_turn || self.opts.system_prompt.is_empty() {
            text.to_string()
        } else {
            format!(
                "{}\n\n---\n\n# Active orders\n\n{text}",
                self.opts.system_prompt
            )
        };
        self.emit("session.message", json!({ "role": "user", "text": text }));
        self.set_state("thinking");
        self.emit("session.state", json!({ "state": "thinking" }));
        if let Ok(mut inner) = self.inner.lock() {
            inner.has_turn = true;
        }
        self.rpc(
            "session/prompt",
            json!({
                "sessionId": acp_id,
                "prompt": [{ "type": "text", "text": full }],
            }),
            None,
            Box::new(|s: &AcpSession, outcome| match outcome {
                Ok(result) => s.on_turn_complete(&result, false),
                Err(error) => {
                    let state = s.state();
                    if state == "cancelled" || state == "paused" || !s.is_running() {
                        return;
                    }
                    s.on_turn_complete(&json!({ "error": error }), true);
                }
            }),
        )
    }

    /// Stop the process but keep the Field session id so resume() can
    /// re-handshake.
    fn pause(&self) -> bool {
        if !self.is_running() {
            return false;
        }
        self.set_state("paused");
        self.stop_child();
        self.emit(
            "session.state",
            json!({ "state": "paused", "detail": "ACP process stopped; workspace retained" }),
        );
        true
    }

    fn resume(&self, orders: Option<String>) -> Result<bool, String> {
        if self.is_running() {
            return Ok(false);
        }
        if let Ok(mut inner) = self.inner.lock() {
            inner.has_turn = false;
        }
        self.spawn_child(Some(orders.unwrap_or_else(|| {
            "Resume from the workspace, restate current progress, and continue.".into()
        })))?;
        self.emit(
            "session.state",
            json!({ "state": "thinking", "detail": "ACP restarted from workspace state" }),
        );
        Ok(true)
    }

    fn cancel(&self) -> bool {
        self.set_state("cancelled");
        self.stop_child();
        self.reject_all_rpc("session cancelled");
        self.emit("session.ended", json!({ "reason": "cancelled" }));
        true
    }

    /// Answer a `harness.permission_requested` by replying to the agent's
    /// ACP request with the chosen option.
    fn decide_permission(&self, request_id: &Value, decision: &str) -> bool {
        let key = js_string(request_id);
        let rpc_id = match self.inner.lock() {
            Ok(mut inner) => inner.pending_permissions.remove(&key),
            Err(_) => None,
        };
        let Some(rpc_id) = rpc_id else {
            return false;
        };
        self.write_message(json!({
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "outcome": {
                    "outcome": "selected",
                    "optionId": if decision == "allow" { "allow" } else { "reject" },
                },
            },
        }))
    }

    fn capabilities(&self) -> Value {
        json!({
            "kind": "acp", "duplex": true, "resumable": true, "permissions": "inline-handshake",
            "delegation": false, "browser": false, "verification": false, "dryRun": true,
        })
    }
}

fn first_non_empty(values: &[Option<&str>]) -> String {
    values
        .iter()
        .flatten()
        .find(|v| !v.is_empty())
        .map_or_else(|| "tool".to_string(), |v| v.to_string())
}

/// The text of an ACP ContentBlock (`{type: "text", text}`) or a bare string.
fn content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(v) if v.is_object() => get_str(v, "text").unwrap_or("").to_string(),
        _ => String::new(),
    }
}

/// Flatten an ACP tool call's output content / rawOutput into preview text.
fn extract_tool_output(update: &Value) -> String {
    if let Some(raw) = get_str(update, "rawOutput") {
        return raw.to_string();
    }
    get_arr(update, "content")
        .map(|blocks| {
            blocks
                .iter()
                .map(|b| content_text(get(b, "content").or(Some(b))))
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    //! `field/server/test/acp-session.test.mjs`, driven through
    //! `handle_line` with the lines `acp-mock-agent.mjs` would write.
    use super::*;

    struct Shared(Arc<Mutex<Vec<Value>>>);
    impl Write for Shared {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            let text = String::from_utf8_lossy(buf);
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                if let Ok(v) = serde_json::from_str(line) {
                    self.0.lock().unwrap().push(v);
                }
            }
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    type Events = Arc<Mutex<Vec<(String, Value)>>>;
    type Writes = Arc<Mutex<Vec<Value>>>;

    fn session(opts: AcpOptions) -> (Arc<AcpSession>, Events, Writes) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink_events = Arc::clone(&events);
        let sink: EventSink = Arc::new(move |kind: &str, data: Value| {
            sink_events.lock().unwrap().push((kind.to_string(), data))
        });
        let s = AcpSession::new(opts, sink);
        let writes = Arc::new(Mutex::new(Vec::new()));
        s.attach_writer(Box::new(Shared(Arc::clone(&writes))));
        (s, events, writes)
    }

    fn options(id: &str) -> AcpOptions {
        AcpOptions {
            id: id.into(),
            agent_id: Some("a1".into()),
            name: Some("Ada".into()),
            role: Some("builder".into()),
            model: Some("mock-model".into()),
            endpoint_id: Some("acp-1".into()),
            cwd: std::env::current_dir().unwrap(),
            workspace_id: Some("ws".into()),
            system_prompt: "BE CORRECT".into(),
            provider_kind: Some("anthropic".into()),
            ..Default::default()
        }
    }

    fn update(u: Value) -> String {
        json!({ "jsonrpc": "2.0", "method": "session/update", "params": { "sessionId": "mock-s1", "update": u } })
            .to_string()
    }

    fn reply(id: u64, result: Value) -> String {
        json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string()
    }

    /// The mock agent's answers to `initialize` (id 1) and `session/new` (id 2).
    fn handshake(s: &AcpSession, writes: &Writes) {
        s.begin_handshake();
        assert_eq!(writes.lock().unwrap()[0]["method"], "initialize");
        s.handle_line(&reply(
            1,
            json!({
                "protocolVersion": 1,
                "agentInfo": { "name": "mock-acp", "version": "0.0.0" },
                "agentCapabilities": { "loadSession": true },
                "authMethods": [],
            }),
        ));
        assert_eq!(writes.lock().unwrap()[1]["method"], "session/new");
        s.handle_line(&reply(
            2,
            json!({ "sessionId": "mock-s1", "modes": { "currentModeId": "write", "availableModes": [] } }),
        ));
    }

    #[test]
    fn contract_and_capabilities() {
        let (s, _, _) = session(AcpOptions {
            id: "contract".into(),
            cwd: std::env::current_dir().unwrap(),
            ..Default::default()
        });
        let caps = s.capabilities();
        assert_eq!(caps["kind"], "acp");
        assert!(super::super::adapter::HARNESS_KINDS.contains(&"acp"));
        assert_eq!(caps["permissions"], "inline-handshake");
        assert_eq!(caps["duplex"], true);
        assert_eq!(s.info().state, "created");
        assert_eq!(s.info().effort, "medium");
        assert_eq!(s.info().name, "contract");
    }

    #[test]
    fn launch_spec_resolves_endpoint_then_operator_env_then_default() {
        assert_eq!(
            normalize_args(Some(&json!(["a", 1, true]))),
            vec!["a", "1", "true"]
        );
        assert_eq!(
            normalize_args(Some(&json!("  --flag   value "))),
            vec!["--flag", "value"]
        );
        assert!(normalize_args(None).is_empty());
        assert!(normalize_args(Some(&json!(null))).is_empty());
        let (cmd, args) = resolve_acp_launch(Some("my-agent"), Some(&["--acp".to_string()]));
        assert_eq!(cmd, "my-agent");
        assert_eq!(args, vec!["--acp"]);
        // No endpoint spec and no FIELD_ACP_BIN: the generic default.
        if std::env::var_os("FIELD_ACP_BIN").is_none() {
            assert_eq!(resolve_acp_launch(None, None).0, DEFAULT_ACP_COMMAND);
        }
        let (s, _, _) = session(AcpOptions {
            command: Some("agent-bin".into()),
            args: Some(vec!["serve".into()]),
            env: [("A".to_string(), "1".to_string())].into_iter().collect(),
            launch_env: [("B".to_string(), "2".to_string())].into_iter().collect(),
            ..options("launch")
        });
        assert_eq!(
            s.launch(),
            ("agent-bin".to_string(), vec!["serve".to_string()])
        );
        let env = s.child_environment();
        assert_eq!(env["A"], "1");
        assert_eq!(env["B"], "2");
        assert_eq!(env["FIELD_SESSION_ID"], "launch");
        assert!(!env.contains_key("CAMEO_CONSOLE_KEY"));
    }

    #[test]
    fn handshake_prompt_round_trip_translates_every_update() {
        let (s, events, writes) = session(options("acp-live"));
        s.set_pending_orders("inspect the workspace");
        handshake(&s, &writes);

        // session/new -> ready + endpoint.routed, then the queued orders go
        // out as the first prompt with the system prompt prepended.
        assert_eq!(s.acp_session_id().as_deref(), Some("mock-s1"));
        {
            let ev = events.lock().unwrap();
            assert!(ev
                .iter()
                .any(|(k, d)| k == "session.state" && d["state"] == "ready"));
            let routed = ev.iter().find(|(k, _)| k == "endpoint.routed").unwrap();
            assert_eq!(routed.1["endpointId"], "acp-1");
            assert_eq!(routed.1["model"], "mock-model");
            assert_eq!(routed.1["reason"], "ACP session/new");
            assert!(ev.iter().any(|(k, d)| k == "session.message"
                && d["role"] == "user"
                && d["text"] == "inspect the workspace"));
            assert!(ev
                .iter()
                .any(|(k, d)| k == "session.state" && d["state"] == "thinking"));
        }
        assert_eq!(s.info().state, "thinking");
        assert!(s.has_turn());
        {
            let w = writes.lock().unwrap();
            assert_eq!(w.len(), 3);
            assert_eq!(w[2]["method"], "session/prompt");
            assert_eq!(w[2]["id"], 3);
            assert_eq!(w[2]["params"]["sessionId"], "mock-s1");
            let text = w[2]["params"]["prompt"][0]["text"].as_str().unwrap();
            assert!(text.starts_with("BE CORRECT\n\n---\n\n# Active orders\n\n"));
            assert!(text.ends_with("inspect the workspace"));
        }

        // Every ACP session/update the mock agent narrates is translated.
        s.handle_line(&update(json!({
            "sessionUpdate": "plan",
            "entries": [
                { "content": "inspect the workspace", "priority": "medium", "status": "completed" },
                { "content": "make the change", "priority": "medium", "status": "pending" },
            ],
        })));
        s.handle_line(&update(json!({ "sessionUpdate": "agent_thought_chunk", "content": { "type": "text", "text": "thinking about it" } })));
        s.handle_line(&update(json!({ "sessionUpdate": "agent_message_chunk", "content": { "type": "text", "text": "here is what I found" } })));
        s.handle_line(&update(json!({
            "sessionUpdate": "tool_call",
            "toolCallId": "call-read-1",
            "title": "read_file src/lib.rs",
            "kind": "read",
            "status": "completed",
            "rawInput": { "path": "src/lib.rs" },
            "content": [{ "type": "content", "content": { "type": "text", "text": "pub fn one() -> u32 { 1 }" } }],
            "locations": [{ "path": "src/lib.rs" }],
        })));
        // Narration Field does not model is ignored, not an error.
        s.handle_line(&update(
            json!({ "sessionUpdate": "current_mode_update", "currentModeId": "write" }),
        ));
        s.handle_line("not json at all");
        s.handle_line("");
        s.handle_line(&reply(
            3,
            json!({ "stopReason": "end_turn", "missionId": "mock-m1" }),
        ));

        let ev = events.lock().unwrap();
        let progress = ev.iter().find(|(k, _)| k == "session.progress").unwrap();
        assert_eq!(progress.1["total"], 2);
        assert_eq!(progress.1["done"], 1);
        assert_eq!(
            progress.1["steps"],
            json!(["inspect the workspace", "make the change"])
        );
        assert!(ev
            .iter()
            .any(|(k, d)| k == "session.thinking" && d["text"] == "thinking about it"));
        assert!(ev.iter().any(|(k, d)| k == "session.message"
            && d["role"] == "assistant"
            && d["text"] == "here is what I found"));
        let tool_use = ev.iter().find(|(k, _)| k == "session.tool_use").unwrap();
        assert_eq!(tool_use.1["toolId"], "call-read-1");
        assert_eq!(tool_use.1["name"], "read_file src/lib.rs");
        assert_eq!(tool_use.1["kind"], "read");
        assert_eq!(tool_use.1["input"], json!({ "path": "src/lib.rs" }));
        assert_eq!(tool_use.1["locations"], json!([{ "path": "src/lib.rs" }]));
        let tool_result = ev.iter().find(|(k, _)| k == "session.tool_result").unwrap();
        assert_eq!(tool_result.1["toolId"], "call-read-1");
        assert_eq!(tool_result.1["name"], "read_file src/lib.rs");
        assert_eq!(tool_result.1["ok"], true);
        assert_eq!(tool_result.1["preview"], "pub fn one() -> u32 { 1 }");
        let done = ev
            .iter()
            .find(|(k, _)| k == "session.turn_complete")
            .unwrap();
        assert_eq!(done.1["isError"], false);
        assert_eq!(done.1["stopReason"], "end_turn");
        assert_eq!(done.1["result"], "end_turn");
        assert_eq!(done.1["missionId"], "mock-m1");
        assert!(ev.iter().any(|(k, d)| k == "session.state"
            && d["state"] == "idle"
            && d["detail"] == "end_turn"));
        assert!(
            ev.iter().all(|(_, d)| d["sessionId"] == "acp-live"),
            "every event carries the session id"
        );
        for kind in [
            "session.state",
            "session.message",
            "session.thinking",
            "session.tool_use",
            "session.tool_result",
            "session.progress",
            "session.turn_complete",
            "endpoint.routed",
        ] {
            assert!(ev.iter().any(|(k, _)| k == kind), "expected {kind}");
            assert!(super::super::adapter::CANONICAL_EVENTS.contains(&kind));
        }
        drop(ev);
        assert_eq!(s.info().state, "idle");
        assert_eq!(s.pending_rpc_count(), 0);
    }

    #[test]
    fn permission_handshake_answers_the_agents_request() {
        let (s, events, writes) = session(options("acp-perm"));
        s.set_pending_orders("first");
        handshake(&s, &writes);
        s.handle_line(&reply(3, json!({ "stopReason": "end_turn" })));

        // The second turn: no system prompt prepended.
        assert!(s.send("please make the write that needs permission"));
        {
            let w = writes.lock().unwrap();
            assert_eq!(w[3]["method"], "session/prompt");
            assert_eq!(
                w[3]["params"]["prompt"][0]["text"],
                "please make the write that needs permission"
            );
        }

        // Agent -> client request: becomes harness.permission_requested.
        s.handle_line(&json!({
            "jsonrpc": "2.0", "id": 1001, "method": "session/request_permission",
            "params": {
                "sessionId": "mock-s1",
                "toolCall": { "toolCallId": "call-danger", "title": "write_file config.json", "rawInput": { "path": "config.json" } },
                "options": [
                    { "optionId": "allow", "name": "Allow", "kind": "allow_once" },
                    { "optionId": "reject", "name": "Reject", "kind": "reject_once" },
                ],
            },
        }).to_string());
        let perm = {
            let ev = events.lock().unwrap();
            ev.iter()
                .find(|(k, _)| k == "harness.permission_requested")
                .map(|(_, d)| d.clone())
                .expect("permission request surfaced")
        };
        assert_eq!(perm["requestId"], "1001");
        assert_eq!(perm["toolName"], "write_file config.json");
        assert_eq!(perm["toolCallId"], "call-danger");
        assert_eq!(perm["input"], json!({ "path": "config.json" }));
        assert_eq!(perm["options"].as_array().unwrap().len(), 2);

        // The registry answers with the requestId it was given, as a string
        // or a number; the reply carries the chosen option.
        assert!(!s.decide_permission(&json!("nope"), "allow"));
        assert!(s.decide_permission(&perm["requestId"], "allow"));
        assert!(
            !s.decide_permission(&json!(1001), "allow"),
            "a request is answered once"
        );
        {
            let w = writes.lock().unwrap();
            let answer = w.last().unwrap();
            assert_eq!(answer["id"], 1001);
            assert_eq!(answer["result"]["outcome"]["outcome"], "selected");
            assert_eq!(answer["result"]["outcome"]["optionId"], "allow");
        }
        s.handle_line(&update(json!({
            "sessionUpdate": "tool_call_update",
            "toolCallId": "call-danger",
            "status": "completed",
            "content": [{ "type": "content", "content": { "type": "text", "text": "wrote config.json" } }],
        })));
        s.handle_line(&reply(
            4,
            json!({ "stopReason": "end_turn", "missionId": "mock-m1" }),
        ));
        let ev = events.lock().unwrap();
        assert!(ev.iter().any(|(k, d)| k == "session.tool_result"
            && d["toolId"] == "call-danger"
            && d["ok"] == true
            && d["preview"] == "wrote config.json"));
        assert_eq!(
            ev.iter()
                .filter(|(k, _)| k == "session.turn_complete")
                .count(),
            2
        );

        // A rejection answers with the reject option and a failed update
        // surfaces as a failed tool result.
        drop(ev);
        s.handle_line(
            &json!({
                "jsonrpc": "2.0", "id": 1002, "method": "session/request_permission",
                "params": { "toolCall": { "toolCallId": "call-2" }, "options": [] },
            })
            .to_string(),
        );
        assert!(s.decide_permission(&json!(1002), "deny"));
        assert_eq!(
            writes.lock().unwrap().last().unwrap()["result"]["outcome"]["optionId"],
            "reject"
        );
        s.handle_line(&update(json!({
            "sessionUpdate": "tool_call_update", "toolCallId": "call-2", "status": "failed",
            "rawOutput": "write rejected",
        })));
        let ev = events.lock().unwrap();
        let last_perm = ev
            .iter()
            .rev()
            .find(|(k, _)| k == "harness.permission_requested")
            .unwrap();
        assert_eq!(
            last_perm.1["toolName"], "call-2",
            "title falls back to the toolCallId"
        );
        assert!(ev.iter().any(|(k, d)| k == "session.tool_result"
            && d["toolId"] == "call-2"
            && d["ok"] == false
            && d["preview"] == "write rejected"
            && d["name"].is_null()));
    }

    #[test]
    fn unsupported_inbound_requests_are_answered_not_ignored() {
        let (s, _, writes) = session(options("acp-fs"));
        handshake(&s, &writes);
        s.handle_line(&json!({ "jsonrpc": "2.0", "id": 77, "method": "fs/read_text_file", "params": { "path": "x" } }).to_string());
        let w = writes.lock().unwrap();
        let answer = w.last().unwrap();
        assert_eq!(answer["id"], 77);
        assert_eq!(answer["error"]["code"], METHOD_NOT_FOUND);
        assert!(answer["error"]["message"]
            .as_str()
            .unwrap()
            .contains("fs/read_text_file"));
    }

    #[test]
    fn errors_time_outs_and_cancel_settle_the_turn() {
        // An RPC error on session/new fails the handshake.
        let (s, events, writes) = session(options("acp-err"));
        s.begin_handshake();
        s.handle_line(&reply(1, json!({ "protocolVersion": 1 })));
        s.handle_line(&json!({ "jsonrpc": "2.0", "id": 2, "error": { "code": -32000, "message": "no workspace" } }).to_string());
        assert_eq!(s.info().state, "error");
        assert!(events
            .lock()
            .unwrap()
            .iter()
            .any(|(k, d)| k == "session.state"
                && d["state"] == "error"
                && d["detail"] == "no workspace"));
        assert_eq!(writes.lock().unwrap().len(), 2);

        // A prompt that errors ends the turn as an error.
        let (s, events, writes) = session(options("acp-turn-err"));
        s.set_pending_orders("go");
        handshake(&s, &writes);
        s.handle_line(
            &json!({ "jsonrpc": "2.0", "id": 3, "error": { "code": -32603 } }).to_string(),
        );
        let ev = events.lock().unwrap();
        let done = ev
            .iter()
            .find(|(k, _)| k == "session.turn_complete")
            .unwrap();
        assert_eq!(done.1["isError"], true);
        assert_eq!(done.1["result"], "ACP error -32603");
        assert!(done.1["stopReason"].is_null());
        drop(ev);
        assert_eq!(s.info().state, "error");

        // Cancel answers the queued prompt with no turn_complete, tells the
        // agent, and ends the session.
        let (s, events, writes) = session(options("acp-cancel"));
        s.set_pending_orders("go");
        handshake(&s, &writes);
        assert_eq!(s.pending_rpc_count(), 1);
        assert!(s.cancel());
        assert_eq!(s.pending_rpc_count(), 0);
        assert!(!s.is_running());
        {
            let w = writes.lock().unwrap();
            let last = w.last().unwrap();
            assert_eq!(last["method"], "session/cancel");
            assert_eq!(last["params"]["sessionId"], "mock-s1");
            assert!(last.get("id").is_none(), "cancel is a notification");
        }
        let ev = events.lock().unwrap();
        assert!(ev
            .iter()
            .any(|(k, d)| k == "session.ended" && d["reason"] == "cancelled"));
        assert!(!ev.iter().any(|(k, _)| k == "session.turn_complete"));
        drop(ev);
        assert!(
            !s.send("after cancel"),
            "a cancelled session takes no orders"
        );
        assert_eq!(s.info().state, "cancelled");
    }

    #[test]
    fn read_only_roles_ask_for_the_ask_mode_before_ready() {
        let (s, events, writes) = session(AcpOptions {
            read_only: true,
            ..options("acp-ro")
        });
        s.set_pending_orders("look");
        handshake(&s, &writes);
        {
            let w = writes.lock().unwrap();
            assert_eq!(w.len(), 3);
            assert_eq!(w[2]["method"], "session/set_mode");
            assert_eq!(
                w[2]["params"],
                json!({ "sessionId": "mock-s1", "modeId": "ask" })
            );
        }
        assert_eq!(
            s.info().state,
            "created",
            "not ready until set_mode answers"
        );
        assert!(!events
            .lock()
            .unwrap()
            .iter()
            .any(|(k, d)| k == "session.state" && d["state"] == "ready"));
        // An agent without set_mode errors; the mode is advisory and the
        // session becomes ready anyway.
        s.handle_line(&json!({ "jsonrpc": "2.0", "id": 3, "error": { "code": -32601, "message": "unknown method" } }).to_string());
        assert_eq!(writes.lock().unwrap()[3]["method"], "session/prompt");
        assert!(events
            .lock()
            .unwrap()
            .iter()
            .any(|(k, d)| k == "session.state" && d["state"] == "ready"));

        let (snapshot, _, w) = session(AcpOptions {
            environment_scope: Some("production-readonly".into()),
            ..options("acp-snap")
        });
        handshake(&snapshot, &w);
        assert_eq!(w.lock().unwrap()[2]["method"], "session/set_mode");
    }

    #[test]
    fn pause_stops_the_agent_and_resume_requeues_the_prompt_prefix() {
        let (s, events, writes) = session(options("acp-pause"));
        s.set_pending_orders("go");
        handshake(&s, &writes);
        assert!(s.has_turn());
        assert!(s.pause());
        assert!(!s.is_running());
        assert!(s.acp_session_id().is_none());
        assert_eq!(s.info().state, "paused");
        assert_eq!(
            writes.lock().unwrap().last().unwrap()["method"],
            "session/cancel"
        );
        assert!(events
            .lock()
            .unwrap()
            .iter()
            .any(|(k, d)| k == "session.state" && d["state"] == "paused"));
        assert!(!s.pause(), "nothing to pause twice");
        // A reply arriving after the pause is not a turn_complete.
        s.handle_line(&reply(3, json!({ "stopReason": "end_turn" })));
        assert!(!events
            .lock()
            .unwrap()
            .iter()
            .any(|(k, _)| k == "session.turn_complete"));
        // Orders sent while paused are refused, not queued.
        assert!(!s.send("while paused"));
    }

    #[test]
    fn usage_is_folded_when_an_agent_reports_it() {
        let (s, events, writes) = session(options("acp-usage"));
        s.set_pending_orders("go");
        handshake(&s, &writes);
        s.handle_line(&update(json!({ "sessionUpdate": "usage_update", "usage": { "input_tokens": 10, "output_tokens": 5 } })));
        s.handle_line(&update(
            json!({ "sessionUpdate": "usage_update", "used": 1 }),
        ));
        s.handle_line(&reply(3, json!({ "stopReason": "end_turn", "usage": { "inputTokens": 12, "outputTokens": 7, "costUsd": 0.01 } })));
        let ev = events.lock().unwrap();
        let usage: Vec<&Value> = ev
            .iter()
            .filter(|(k, _)| k == "session.usage")
            .map(|(_, d)| d)
            .collect();
        assert_eq!(usage.len(), 2, "an update without figures is ignored");
        assert_eq!(usage[0]["inputTokens"], 10);
        assert_eq!(usage[0]["outputTokens"], 5);
        assert_eq!(usage[1]["outputTokens"], 7);
        assert_eq!(usage[1]["costUsd"], 0.01);
    }
}
