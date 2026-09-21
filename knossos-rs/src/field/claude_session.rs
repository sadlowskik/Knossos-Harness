//! The Claude Code adapter: one real `claude` process in stream-json duplex
//! mode. Port of `field/server/src/harness/session.js` and `tools.js`.
//!
//! Every Field event about a Claude Code agent originates here, parsed from
//! that process's actual stdout. Nothing is synthesized. Permissions do not
//! travel over stdin: the child is pointed at the Field permission bridge
//! (`knossos field-permission-bridge`, an MCP server) through
//! `--permission-prompt-tool`, and that bridge posts each request to the
//! Field API with the session's harness capability.

use super::adapter::{stamp, Adapter, BeforeStart, EventSink, SessionInfo};
use super::adapters::{manifest_by_id, CLAUDE_CODE};
use super::child_env::{build_child_environment, process_environment};
use super::js::{get, get_str, jnum, js_string, truthy};
use super::knossos_session::resolve_knossos_binary;
use super::permission_bridge::{mcp_config, mcp_config_with_token};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

/// Field exposes low / medium / high / adaptive; the harness takes an effort
/// level. `adaptive` is resolved by the registry before it reaches here.
const EFFORT: [&str; 5] = ["low", "medium", "high", "xhigh", "max"];

/// Where the `claude` CLI is: `FIELD_CLAUDE_BIN`, then `claude` (and its
/// Windows shims) on the PATH, else the bare name for the OS to resolve.
pub fn resolve_claude_binary() -> PathBuf {
    if let Some(explicit) = std::env::var_os("FIELD_CLAUDE_BIN").filter(|v| !v.is_empty()) {
        return PathBuf::from(explicit);
    }
    let names: &[&str] = if cfg!(windows) {
        &["claude.cmd", "claude.exe", "claude.bat", "claude"]
    } else {
        &["claude"]
    };
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            for name in names {
                let candidate = dir.join(name);
                if candidate.is_file() {
                    return candidate;
                }
            }
        }
    }
    PathBuf::from("claude")
}

/// Mints a session-scoped harness capability; `None` when unavailable.
pub type TokenMint = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;
/// Registers a secret with the log redactor.
pub type SecretRegistrar = Arc<dyn Fn(&str) + Send + Sync>;

/// How a restart re-arms the permission bridge: every process start gets a
/// fresh harness capability, exactly as the registry's `beforeStart` did.
#[derive(Clone)]
pub struct PermissionBridge {
    pub api_base: String,
    pub mint: TokenMint,
    pub register_secret: Option<SecretRegistrar>,
}

/// Everything a Claude Code session is created with.
#[derive(Clone, Default)]
pub struct ClaudeOptions {
    /// uuid, also the harness `--session-id`.
    pub id: String,
    pub agent_id: Option<String>,
    pub name: Option<String>,
    pub role: Option<String>,
    pub model: Option<String>,
    pub endpoint_id: Option<String>,
    pub effort: Option<String>,
    pub cwd: PathBuf,
    pub workspace_id: Option<String>,
    pub system_prompt: Option<String>,
    pub add_dirs: Vec<String>,
    pub allowed_tools: Option<Vec<String>>,
    pub disallowed_tools: Option<Vec<String>>,
    /// JSON document for `--mcp-config`.
    pub mcp_config: Option<String>,
    pub permission_tool: Option<String>,
    pub permission_mode: Option<String>,
    /// The mounted workspaces, for locating tool activity on the map.
    pub workspaces: Vec<Value>,
    pub env: BTreeMap<String, String>,
    pub provider_kind: Option<String>,
    pub credential_env_keys: Vec<String>,
    pub permission_bridge: Option<PermissionBridge>,
    /// The binary to run; `None` resolves at start.
    pub binary: Option<PathBuf>,
    /// Replace the argument list entirely (a test seam, as the Node tests
    /// monkeypatched `buildArgs`).
    pub args_override: Option<Vec<String>>,
}

struct Running {
    generation: u64,
    stdin: Box<dyn Write + Send>,
    child: Option<Arc<Mutex<Child>>>,
}

#[derive(Default)]
struct LifetimeUsage {
    input: f64,
    output: f64,
    cache_read: f64,
}

struct Inner {
    running: Option<Running>,
    /// Processes owned, including ones still draining after a stop.
    owned: usize,
    generation: u64,
    started: bool,
    first_start: bool,
    stderr: String,
    pending_tools: HashMap<String, String>,
    usage: LifetimeUsage,
    turn_saw_usage: bool,
    mcp_config: Option<String>,
}

pub struct ClaudeSession {
    me: Weak<ClaudeSession>,
    info: Mutex<SessionInfo>,
    opts: ClaudeOptions,
    sink: EventSink,
    inner: Mutex<Inner>,
    before_start: Mutex<Option<BeforeStart>>,
}

impl ClaudeSession {
    pub fn new(opts: ClaudeOptions, sink: EventSink) -> Arc<ClaudeSession> {
        let effort = opts
            .effort
            .clone()
            .filter(|e| EFFORT.contains(&e.as_str()))
            .unwrap_or_else(|| "medium".into());
        let info = SessionInfo {
            id: opts.id.clone(),
            agent_id: opts.agent_id.clone(),
            name: opts.name.clone().unwrap_or_else(|| opts.id.clone()),
            role: opts.role.clone(),
            model: opts.model.clone(),
            endpoint_id: opts.endpoint_id.clone(),
            effort,
            cwd: opts.cwd.clone(),
            workspace_id: opts.workspace_id.clone(),
            state: "created".into(),
        };
        let mcp_config = opts.mcp_config.clone();
        Arc::new_cyclic(|me| ClaudeSession {
            me: me.clone(),
            info: Mutex::new(info),
            opts,
            sink,
            inner: Mutex::new(Inner {
                running: None,
                owned: 0,
                generation: 0,
                started: false,
                first_start: true,
                stderr: String::new(),
                pending_tools: HashMap::new(),
                usage: LifetimeUsage::default(),
                turn_saw_usage: false,
                mcp_config,
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

    /// The current `--mcp-config` document.
    pub fn mcp_config(&self) -> Option<String> {
        self.inner.lock().ok().and_then(|i| i.mcp_config.clone())
    }

    pub fn set_mcp_config(&self, config: Option<String>) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.mcp_config = config;
        }
    }

    /// The `claude` argument list.
    pub fn build_args(&self, resume: bool) -> Vec<String> {
        let info = self.info();
        let mut a: Vec<String> = [
            "-p",
            "--output-format",
            "stream-json",
            "--input-format",
            "stream-json",
            "--verbose",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        if resume {
            a.push("--resume".into());
        } else {
            a.push("--session-id".into());
        }
        a.push(self.opts.id.clone());
        if let Some(model) = info.model.filter(|m| !m.is_empty()) {
            a.push("--model".into());
            a.push(model);
        }
        a.push("--effort".into());
        a.push(info.effort);
        if let Some(prompt) = self.opts.system_prompt.as_ref().filter(|p| !p.is_empty()) {
            a.push("--append-system-prompt".into());
            a.push(prompt.clone());
        }
        for dir in &self.opts.add_dirs {
            a.push("--add-dir".into());
            a.push(dir.clone());
        }
        // --tools is the actual role capability boundary. Do not also pass
        // --allowedTools: that flag auto-approves consequential tools and
        // would bypass Field's permission bridge for writes, commands, and
        // network requests.
        a.push("--tools".into());
        match &self.opts.allowed_tools {
            Some(tools) if !tools.is_empty() => a.push(tools.join(",")),
            _ => a.push(String::new()),
        }
        if let Some(denied) = self
            .opts
            .disallowed_tools
            .as_ref()
            .filter(|d| !d.is_empty())
        {
            a.push("--disallowedTools".into());
            a.push(denied.join(","));
        }
        let mcp_config = self.mcp_config();
        match (&self.opts.permission_tool, mcp_config) {
            (Some(tool), Some(config)) if !tool.is_empty() && !config.is_empty() => {
                a.push("--mcp-config".into());
                a.push(config);
                a.push("--permission-prompt-tool".into());
                a.push(tool.clone());
            }
            _ => {
                if let Some(mode) = self.opts.permission_mode.as_ref().filter(|m| !m.is_empty()) {
                    a.push("--permission-mode".into());
                    a.push(mode.clone());
                }
            }
        }
        a
    }

    /// The environment the child receives.
    pub fn child_environment(&self) -> BTreeMap<String, String> {
        let mut overrides = self.opts.env.clone();
        overrides.insert("FIELD_SESSION_ID".into(), self.opts.id.clone());
        build_child_environment(
            &process_environment(),
            self.opts.provider_kind.as_deref(),
            &self.opts.credential_env_keys,
            &overrides,
        )
    }

    /// Write one stream-json line to the child's stdin.
    fn write(&self, message: &Value) -> bool {
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
        }
    }

    /// A restart mints a fresh harness capability for the bridge, as the
    /// registry's `beforeStart` did; the first start uses the one it was
    /// created with.
    fn refresh_permission_token(&self) -> Result<(), String> {
        let first = self.inner.lock().map(|i| i.first_start).unwrap_or(true);
        if first {
            return Ok(());
        }
        let Some(bridge) = &self.opts.permission_bridge else {
            return Ok(());
        };
        let token = (bridge.mint)(&self.opts.id)
            .ok_or("session-scoped permission capability service is unavailable")?;
        if let Some(register) = &bridge.register_secret {
            register(&token);
        }
        let current = self.mcp_config();
        let refreshed = current
            .as_deref()
            .and_then(|c| mcp_config_with_token(c, &token))
            .unwrap_or_else(|| {
                mcp_config(
                    &resolve_knossos_binary(),
                    &bridge.api_base,
                    &self.opts.id,
                    &token,
                )
            });
        self.set_mcp_config(Some(refreshed));
        Ok(())
    }

    fn spawn_child(&self, orders: Option<String>, resume: bool) -> Result<(), String> {
        let this = self.me.upgrade().ok_or("session is being dropped")?;
        let hook = self.before_start.lock().ok().and_then(|h| h.clone());
        if let Some(hook) = hook {
            hook()?;
        }
        self.refresh_permission_token()?;
        let args = self
            .opts
            .args_override
            .clone()
            .unwrap_or_else(|| self.build_args(resume));
        let binary = self
            .opts
            .binary
            .clone()
            .unwrap_or_else(resolve_claude_binary);
        let env = self.child_environment();
        let mut command = Command::new(&binary);
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
                if let Ok(mut inner) = self.inner.lock() {
                    inner.started = true;
                    inner.first_start = false;
                }
                self.set_state("error");
                self.emit(
                    "session.ended",
                    json!({ "reason": "error", "error": format!("spawn failed: {error}") }),
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
            inner.first_start = false;
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
        if let Some(orders) = orders.filter(|o| !o.is_empty()) {
            self.send(&orders);
        }
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
        let state = self.state();
        if state == "paused" || state == "cancelled" {
            return;
        }
        if code != Some(0) {
            self.set_state("error");
            let error = if stderr_tail.is_empty() {
                format!(
                    "harness exited {}",
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

    /// End stdin and kill the child; the waiter thread reaps it and releases
    /// its capacity on close.
    fn stop_child(&self) {
        let child = self
            .inner
            .lock()
            .ok()
            .and_then(|mut inner| inner.running.take().and_then(|r| r.child));
        if let Some(child) = child {
            if let Ok(mut child) = child.lock() {
                let _ = child.kill();
            }
        }
    }

    /// Translate one stream-json line into Field events.
    pub fn handle_line(&self, line: &str) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return;
        }
        let Ok(msg) = serde_json::from_str::<Value>(trimmed) else {
            return;
        };
        match get_str(&msg, "type") {
            Some("system") => {
                if get_str(&msg, "subtype") == Some("init") {
                    self.set_state("ready");
                    self.emit("session.state", json!({ "state": "ready" }));
                    let info = self.info();
                    self.emit(
                        "endpoint.routed",
                        json!({
                            "endpointId": info.endpoint_id,
                            "model": get(&msg, "model").cloned().or(info.model.map(Value::String)),
                            "reason": "session start",
                        }),
                    );
                }
            }
            Some("assistant") => self.handle_assistant(msg.get("message").unwrap_or(&Value::Null)),
            Some("user") => self.handle_tool_results(msg.get("message").unwrap_or(&Value::Null)),
            Some("result") => self.handle_result(&msg),
            _ => {}
        }
    }

    fn handle_assistant(&self, message: &Value) {
        let content = message
            .get("content")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for part in &content {
            match get_str(part, "type") {
                Some("text") => {
                    if let Some(text) = get_str(part, "text").filter(|t| !t.trim().is_empty()) {
                        self.emit(
                            "session.message",
                            json!({ "role": "assistant", "text": text }),
                        );
                    }
                }
                Some("thinking") => {
                    if let Some(thinking) = get(part, "thinking").filter(|t| truthy(t)) {
                        let text: String = js_string(thinking).chars().take(2000).collect();
                        self.emit("session.thinking", json!({ "text": text }));
                    }
                }
                Some("tool_use") => self.handle_tool_use(part),
                _ => {}
            }
        }
        if let Some(usage) = get(message, "usage").filter(|u| truthy(u)) {
            self.emit_usage(usage, None);
        }
    }

    fn handle_tool_use(&self, part: &Value) {
        let name = get_str(part, "name").unwrap_or("").to_string();
        let empty = json!({});
        let input = get(part, "input").unwrap_or(&empty);
        let tool_id = part.get("id").cloned().unwrap_or(Value::Null);
        let info = describe_tool(&name, input, &self.opts.workspaces, &self.opts.cwd);
        if let (Ok(mut inner), Some(id)) = (self.inner.lock(), tool_id.as_str()) {
            inner.pending_tools.insert(id.to_string(), name.clone());
        }
        self.emit(
            "session.tool_use",
            json!({
                "toolId": tool_id,
                "name": name,
                "summary": info.summary,
                "workspaceId": info.workspace_id,
                "dir": info.dir,
                "path": info.path,
                "command": info.command,
                "input": truncate_input(part.get("input")),
            }),
        );
        if let Some(browser) = &info.browser {
            self.emit(
                "browser.navigated",
                json!({ "url": browser.url, "domain": browser.domain }),
            );
        }
        if let Some(delegation) = &info.delegation {
            // A real child harness session, created by the agent itself,
            // inside the delegation limits the Field configured for it.
            self.emit(
                "session.delegated",
                json!({
                    "parentSessionId": self.opts.id,
                    "childSessionId": format!("{}:{}", self.opts.id, js_string(&tool_id)),
                    "description": delegation.description,
                    "subagentType": delegation.subagent_type,
                }),
            );
        }
        if let Some((done, total)) = info.progress {
            self.emit("session.progress", json!({ "done": done, "total": total }));
        }
    }

    fn handle_tool_results(&self, message: &Value) {
        let content = message
            .get("content")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for part in &content {
            if get_str(part, "type") != Some("tool_result") {
                continue;
            }
            let tool_use_id = part.get("tool_use_id").cloned().unwrap_or(Value::Null);
            let pending = self.inner.lock().ok().and_then(|mut inner| {
                tool_use_id
                    .as_str()
                    .and_then(|id| inner.pending_tools.remove(id))
            });
            let text = match part.get("content") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Array(items)) => items
                    .iter()
                    .map(|c| get_str(c, "text").unwrap_or("").to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
                _ => String::new(),
            };
            let preview: String = text.chars().take(400).collect();
            self.emit(
                "session.tool_result",
                json!({
                    "toolId": tool_use_id,
                    "name": pending,
                    "ok": !part.get("is_error").is_some_and(truthy),
                    "preview": preview,
                }),
            );
        }
    }

    /// Fold one usage report into the lifetime totals and report both.
    pub fn emit_usage(&self, usage: &Value, cost_usd: Option<f64>) {
        let count = |key: &str| {
            usage
                .get(key)
                .and_then(Value::as_f64)
                .filter(|n| n.is_finite())
                .unwrap_or(0.0)
        };
        let delta_input = count("input_tokens").max(0.0);
        let delta_output = count("output_tokens").max(0.0);
        let delta_cache_read = count("cache_read_input_tokens").max(0.0);
        let (input, output, cache_read) = {
            let Ok(mut inner) = self.inner.lock() else {
                return;
            };
            inner.usage.input += delta_input;
            inner.usage.output += delta_output;
            inner.usage.cache_read += delta_cache_read;
            inner.turn_saw_usage = true;
            (
                inner.usage.input,
                inner.usage.output,
                inner.usage.cache_read,
            )
        };
        let context = count("input_tokens")
            + count("cache_read_input_tokens")
            + count("cache_creation_input_tokens")
            + count("output_tokens");
        let mut data = json!({
            "inputTokens": jnum(input),
            "outputTokens": jnum(output),
            "cacheRead": jnum(cache_read),
            "contextTokens": jnum(context),
            "deltaInput": jnum(delta_input),
            "deltaOutput": jnum(delta_output),
        });
        if let (Some(cost), Some(obj)) = (cost_usd, data.as_object_mut()) {
            obj.insert("costUsd".into(), jnum(cost));
        }
        self.emit("session.usage", data);
    }

    /// The end of a turn: usage the stream did not already report, the
    /// resting state, and the turn summary.
    pub fn handle_result(&self, msg: &Value) {
        let cost = msg.get("total_cost_usd").and_then(Value::as_f64);
        let saw_usage = self.inner.lock().map(|i| i.turn_saw_usage).unwrap_or(false);
        match get(msg, "usage").filter(|u| truthy(u)) {
            Some(usage) if !saw_usage => self.emit_usage(usage, cost),
            _ => {
                if let Some(cost) = cost {
                    self.emit("session.usage", json!({ "costUsd": jnum(cost) }));
                }
            }
        }
        let is_error = msg.get("is_error").is_some_and(truthy);
        let state = if is_error { "error" } else { "idle" };
        self.set_state(state);
        self.emit(
            "session.state",
            json!({ "state": state, "detail": msg.get("subtype").cloned().unwrap_or(Value::Null) }),
        );
        let result = get_str(msg, "result").map(|r| r.chars().take(4000).collect::<String>());
        self.emit(
            "session.turn_complete",
            json!({
                "result": result,
                "turns": msg.get("num_turns").cloned().unwrap_or(Value::Null),
                "durationMs": msg.get("duration_ms").cloned().unwrap_or(Value::Null),
                "isError": is_error,
            }),
        );
    }
}

impl Adapter for ClaudeSession {
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
            info.effort = if EFFORT.contains(&effort) {
                effort.to_string()
            } else {
                "medium".into()
            };
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
        self.spawn_child(orders, false)
    }

    /// Push a real user turn into the running harness session.
    fn send(&self, text: &str) -> bool {
        if !self.is_running() {
            return false;
        }
        let message = json!({
            "type": "user",
            "message": { "role": "user", "content": [{ "type": "text", "text": text }] },
        });
        if !self.write(&message) {
            return false;
        }
        if let Ok(mut inner) = self.inner.lock() {
            inner.turn_saw_usage = false;
        }
        self.emit("session.message", json!({ "role": "user", "text": text }));
        self.set_state("thinking");
        self.emit("session.state", json!({ "state": "thinking" }));
        true
    }

    /// Stop the process but keep the session id so `--resume` can pick it
    /// back up.
    fn pause(&self) -> bool {
        if !self.is_running() {
            return false;
        }
        self.set_state("paused");
        self.stop_child();
        self.emit("session.state", json!({ "state": "paused" }));
        true
    }

    fn resume(&self, orders: Option<String>) -> Result<bool, String> {
        if self.is_running() {
            return Ok(false);
        }
        self.set_state("spawning");
        self.spawn_child(Some(orders.unwrap_or_else(|| "Continue.".into())), true)?;
        self.emit(
            "session.state",
            json!({ "state": "thinking", "detail": "resumed" }),
        );
        Ok(true)
    }

    fn cancel(&self) -> bool {
        self.set_state("cancelled");
        self.stop_child();
        self.emit("session.ended", json!({ "reason": "cancelled" }));
        true
    }

    /// Permissions reach Claude Code through the MCP bridge and the Field
    /// API, never over stdin; there is nothing to write here.
    fn decide_permission(&self, _request_id: &Value, _decision: &str) -> bool {
        false
    }

    fn capabilities(&self) -> Value {
        manifest_by_id(CLAUDE_CODE)
            .map(|m| m.capabilities())
            .unwrap_or_else(|| json!({ "kind": Value::Null }))
    }
}

/// `truncateInput`: long string arguments are cut for the event record.
pub fn truncate_input(input: Option<&Value>) -> Value {
    match input {
        Some(Value::Object(fields)) => {
            let mut out = Map::new();
            for (k, v) in fields {
                let value = match v {
                    Value::String(s) if s.chars().count() > 600 => {
                        Value::String(s.chars().take(600).collect::<String>() + "…")
                    }
                    other => other.clone(),
                };
                out.insert(k.clone(), value);
            }
            Value::Object(out)
        }
        Some(other) => other.clone(),
        None => Value::Null,
    }
}

// ---- tools.js: a real harness tool call in Field's spatial vocabulary ----

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Browser {
    pub url: String,
    pub domain: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Delegation {
    pub description: String,
    pub subagent_type: String,
}

/// What a tool call means on the map. Nothing here invents activity: every
/// field is derived from the actual tool name and arguments.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolInfo {
    pub summary: String,
    pub workspace_id: Option<String>,
    pub dir: Option<String>,
    pub path: Option<String>,
    pub command: Option<String>,
    pub browser: Option<Browser>,
    pub delegation: Option<Delegation>,
    pub progress: Option<(usize, usize)>,
    pub is_edit: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Location {
    pub workspace_id: String,
    pub rel: String,
    /// Always a directory: the path itself when it is one, otherwise its parent.
    pub dir: String,
    pub is_dir: bool,
}

const EDIT_TOOLS: [&str; 3] = ["Edit", "Write", "NotebookEdit"];

fn parent_of(rel: &str) -> String {
    match rel.rfind('/') {
        Some(i) => rel[..i].to_string(),
        None => String::new(),
    }
}

/// `path.resolve`: absolute against the process cwd, `.` and `..` folded.
fn resolve(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };
    let mut out = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Forward slashes, without Windows' verbatim prefix.
fn normalize(path: &Path) -> String {
    let text = path.to_string_lossy().replace('\\', "/");
    text.strip_prefix("//?/")
        .map(str::to_string)
        .unwrap_or(text)
}

/// Which mounted workspace does this path belong to, and where inside it?
pub fn locate(abs_path: &str, workspaces: &[Value]) -> Option<Location> {
    if abs_path.is_empty() {
        return None;
    }
    let resolved = resolve(Path::new(abs_path));
    // Report non-existent targets lexically.
    let candidate = std::fs::canonicalize(&resolved).unwrap_or(resolved);
    let norm = normalize(&candidate);
    let mut best: Option<(&Value, String)> = None;
    for w in workspaces {
        let root = match get_str(w, "canonicalPath") {
            Some(canonical) => normalize(Path::new(canonical)),
            None => normalize(&resolve(Path::new(get_str(w, "path").unwrap_or("")))),
        };
        let inside = norm == root || norm.starts_with(&format!("{root}/"));
        if inside && best.as_ref().is_none_or(|(_, r)| root.len() > r.len()) {
            best = Some((w, root));
        }
    }
    let (workspace, root) = best?;
    let rel = norm[root.len()..].trim_start_matches('/').to_string();
    let is_dir = match std::fs::metadata(&norm) {
        Ok(meta) => meta.is_dir(),
        // Gone already: fall back to the shape.
        Err(_) => Path::new(&rel).extension().is_none(),
    };
    Some(Location {
        workspace_id: get_str(workspace, "id").unwrap_or("").to_string(),
        dir: if is_dir { rel.clone() } else { parent_of(&rel) },
        rel,
        is_dir,
    })
}

fn domain_of(url: &str) -> Option<String> {
    url::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(str::to_string))
}

fn short(p: &str, n: usize) -> String {
    let count = p.chars().count();
    if count <= n {
        p.to_string()
    } else {
        let tail: String = p.chars().skip(count - (n - 1)).collect();
        format!("…{tail}")
    }
}

fn take(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}

fn arg_string(input: &Value, key: &str) -> String {
    get(input, key).map(js_string).unwrap_or_default()
}

fn place(out: &mut ToolInfo, loc: Option<Location>, with_path: bool) {
    if let Some(loc) = loc {
        out.workspace_id = Some(loc.workspace_id);
        out.dir = Some(loc.dir);
        if with_path {
            out.path = Some(loc.rel);
        }
    }
}

/// `describeTool`: the spatial reading of one tool call.
pub fn describe_tool(name: &str, input: &Value, workspaces: &[Value], cwd: &Path) -> ToolInfo {
    let mut out = ToolInfo {
        summary: name.to_string(),
        is_edit: EDIT_TOOLS.contains(&name),
        ..Default::default()
    };
    let cwd_text = cwd.to_string_lossy().into_owned();
    let file_path = get(input, "file_path")
        .or_else(|| get(input, "filePath"))
        .or_else(|| get(input, "notebook_path"))
        .map(js_string);
    let search_path = get(input, "path").map(js_string);

    if let Some(file_path) = file_path {
        let loc = locate(&file_path, workspaces);
        let label = loc.as_ref().map(|l| l.rel.clone()).unwrap_or(file_path);
        place(&mut out, loc, true);
        out.summary = format!("{name} {}", short(&label, 48));
        return out;
    }

    match name {
        "Grep" => {
            let loc = locate(search_path.as_deref().unwrap_or(&cwd_text), workspaces);
            place(&mut out, loc, false);
            out.summary = format!("Grep /{}/", take(&arg_string(input, "pattern"), 32));
        }
        "Glob" => {
            let loc = locate(search_path.as_deref().unwrap_or(&cwd_text), workspaces);
            place(&mut out, loc, false);
            out.summary = format!("Glob {}", take(&arg_string(input, "pattern"), 40));
        }
        "Bash" | "PowerShell" => {
            place(&mut out, locate(&cwd_text, workspaces), false);
            let cmd = arg_string(input, "command")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            out.summary = format!("$ {}", take(&cmd, 56));
            out.command = Some(cmd);
        }
        "WebFetch" => {
            let url = get_str(input, "url").map(str::to_string);
            let domain = url.as_deref().and_then(domain_of);
            if let (Some(url), Some(domain)) = (&url, &domain) {
                out.browser = Some(Browser {
                    url: url.clone(),
                    domain: domain.clone(),
                });
            }
            out.summary = format!("Fetch {}", domain.or(url).unwrap_or_default());
        }
        "WebSearch" => {
            out.summary = format!("Search \"{}\"", take(&arg_string(input, "query"), 40));
        }
        "Task" | "Agent" => {
            let description = get(input, "description")
                .map(js_string)
                .or_else(|| get_str(input, "prompt").map(|p| take(p, 80)))
                .unwrap_or_else(|| "subagent".into());
            let subagent_type = get(input, "subagent_type")
                .map(js_string)
                .unwrap_or_else(|| "general-purpose".into());
            out.summary = format!("Delegate → {subagent_type}");
            out.delegation = Some(Delegation {
                description,
                subagent_type,
            });
        }
        "TodoWrite" => {
            let todos = input
                .get("todos")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let done = todos
                .iter()
                .filter(|t| get_str(t, "status") == Some("completed"))
                .count();
            out.progress = Some((done, todos.len()));
            out.summary = format!("Plan {done}/{}", todos.len());
        }
        _ => {
            // Browser-driving MCP tools carry a url; treat them as real
            // browser routes.
            if let Some(url) = get_str(input, "url") {
                let domain = domain_of(url);
                if let Some(domain) = &domain {
                    out.browser = Some(Browser {
                        url: url.to_string(),
                        domain: domain.clone(),
                    });
                }
                out.summary = format!("{name} {}", domain.unwrap_or_default())
                    .trim()
                    .to_string();
                return out;
            }
            place(&mut out, locate(&cwd_text, workspaces), false);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    //! `adapter-contract.test.mjs` (the adapter half), `session-usage.test.mjs`
    //! and the stream-json translation of `session.js`, case for case.
    use super::*;
    use crate::field::adapter::HARNESS_KINDS;

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

    fn session(opts: ClaudeOptions) -> (Arc<ClaudeSession>, Events) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink_events = Arc::clone(&events);
        let sink: EventSink = Arc::new(move |kind: &str, data: Value| {
            sink_events.lock().unwrap().push((kind.to_string(), data))
        });
        (ClaudeSession::new(opts, sink), events)
    }

    fn cwd() -> PathBuf {
        std::env::current_dir().unwrap()
    }

    fn workspace() -> Value {
        json!({ "id": "cameo", "name": "Cameo", "path": cwd().to_string_lossy(), "mounted": true })
    }

    #[test]
    fn contract_capabilities_and_stamp() {
        let (s, events) = session(ClaudeOptions {
            id: "contract".into(),
            cwd: cwd(),
            ..Default::default()
        });
        let caps = s.capabilities();
        assert!(HARNESS_KINDS.contains(&caps["kind"].as_str().unwrap()));
        assert_eq!(caps, manifest_by_id(CLAUDE_CODE).unwrap().capabilities());
        assert_eq!(caps["permissions"], "mcp-bridge");
        assert_eq!(
            s.info().effort,
            "medium",
            "an unknown effort falls back to medium"
        );
        assert!(!s.is_running());
        assert!(!s.send("nothing to write to"));
        assert!(!s.pause());
        s.emit("session.state", json!({ "state": "ready" }));
        let events = events.lock().unwrap();
        assert_eq!(events[0].0, "session.state");
        assert_eq!(events[0].1["sessionId"], "contract");
        assert_eq!(events[0].1["state"], "ready");
    }

    #[test]
    fn build_args_matches_session_js() {
        let (s, _) = session(ClaudeOptions {
            id: "args".into(),
            model: Some("claude-sonnet".into()),
            effort: Some("high".into()),
            cwd: cwd(),
            system_prompt: Some("constitution".into()),
            add_dirs: vec!["/extra".into()],
            allowed_tools: Some(vec!["Read".into(), "Edit".into()]),
            disallowed_tools: Some(vec!["WebFetch".into()]),
            mcp_config: Some("{\"mcpServers\":{}}".into()),
            permission_tool: Some("mcp__field__approve".into()),
            permission_mode: Some("plan".into()),
            ..Default::default()
        });
        let args = s.build_args(false);
        let expected: Vec<String> = [
            "-p",
            "--output-format",
            "stream-json",
            "--input-format",
            "stream-json",
            "--verbose",
            "--session-id",
            "args",
            "--model",
            "claude-sonnet",
            "--effort",
            "high",
            "--append-system-prompt",
            "constitution",
            "--add-dir",
            "/extra",
            "--tools",
            "Read,Edit",
            "--disallowedTools",
            "WebFetch",
            "--mcp-config",
            "{\"mcpServers\":{}}",
            "--permission-prompt-tool",
            "mcp__field__approve",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(args, expected);
        let resumed = s.build_args(true);
        assert!(resumed
            .windows(2)
            .any(|w| w[0] == "--resume" && w[1] == "args"));
        assert!(!resumed.iter().any(|a| a == "--session-id"));
        assert!(
            !args.iter().any(|a| a == "--allowedTools"),
            "never auto-approve tools"
        );

        // No tools: an explicit empty boundary. No bridge: the permission mode.
        let (bare, _) = session(ClaudeOptions {
            id: "bare".into(),
            cwd: cwd(),
            permission_mode: Some("plan".into()),
            ..Default::default()
        });
        let args = bare.build_args(false);
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--tools" && w[1].is_empty()));
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--permission-mode" && w[1] == "plan"));
        assert!(!args.iter().any(|a| a == "--mcp-config"));
        assert!(!args.iter().any(|a| a == "--model"));
        assert_eq!(
            bare.child_environment()
                .get("FIELD_SESSION_ID")
                .map(String::as_str),
            Some("bare")
        );
    }

    #[test]
    fn session_usage_cumulative_totals_and_non_duplicated_result_accounting() {
        let (s, events) = session(ClaudeOptions {
            id: "usage-1".into(),
            agent_id: Some("agent".into()),
            name: Some("Agent".into()),
            role: Some("builder".into()),
            model: Some("model".into()),
            endpoint_id: Some("endpoint".into()),
            cwd: cwd(),
            workspace_id: Some("workspace".into()),
            provider_kind: Some("anthropic".into()),
            ..Default::default()
        });
        s.emit_usage(
            &json!({ "input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 2 }),
            None,
        );
        s.handle_result(&json!({
            "usage": { "input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 2 },
            "total_cost_usd": 0.25, "result": "done", "num_turns": 1, "duration_ms": 10,
        }));
        {
            let events = events.lock().unwrap();
            let usage: Vec<&Value> = events
                .iter()
                .filter(|(k, _)| k == "session.usage")
                .map(|(_, d)| d)
                .collect();
            let delta_output: f64 = usage
                .iter()
                .map(|d| d["deltaOutput"].as_f64().unwrap_or(0.0))
                .sum();
            assert_eq!(
                delta_output, 5.0,
                "result usage cannot double-count streamed completion tokens"
            );
            assert_eq!(usage.last().unwrap()["costUsd"], 0.25);
            // input + cache read + cache creation + output, as the Node adapter sums it.
            assert_eq!(usage[0]["contextTokens"], 17);
            assert!(events
                .iter()
                .any(|(k, d)| k == "session.state" && d["state"] == "idle"));
            assert!(events.iter().any(|(k, d)| k == "session.turn_complete"
                && d["result"] == "done"
                && d["turns"] == 1
                && d["durationMs"] == 10
                && d["isError"] == false));
        }
        s.emit_usage(
            &json!({ "input_tokens": 4, "output_tokens": 7, "cache_read_input_tokens": 0 }),
            None,
        );
        let events = events.lock().unwrap();
        let latest = &events
            .iter()
            .rfind(|(k, _)| k == "session.usage")
            .unwrap()
            .1;
        assert_eq!(latest["inputTokens"], 14);
        assert_eq!(latest["outputTokens"], 12);
        assert_eq!(latest["deltaOutput"], 7);
        assert_eq!(latest["cacheRead"], 2);
    }

    #[test]
    fn stream_json_translates_into_canonical_events() {
        let (s, events) = session(ClaudeOptions {
            id: "c1".into(),
            model: Some("configured".into()),
            endpoint_id: Some("cloud".into()),
            cwd: cwd(),
            workspaces: vec![workspace()],
            ..Default::default()
        });
        let writes = Arc::new(Mutex::new(Vec::new()));
        s.attach_writer(Box::new(Shared(Arc::clone(&writes))));
        assert!(s.is_running());
        assert!(s.send("build the gateway"));
        assert_eq!(s.info().state, "thinking");
        assert_eq!(
            writes.lock().unwrap()[0],
            json!({ "type": "user", "message": { "role": "user", "content": [{ "type": "text", "text": "build the gateway" }] } })
        );

        s.handle_line("");
        s.handle_line("not json");
        s.handle_line(
            &json!({ "type": "system", "subtype": "init", "model": "claude-real" }).to_string(),
        );
        assert_eq!(s.info().state, "ready");
        let file = cwd().join("src").join("lib.rs");
        let long = "x".repeat(700);
        s.handle_line(&json!({ "type": "assistant", "message": {
            "content": [
                { "type": "thinking", "thinking": "let me think" },
                { "type": "text", "text": "   " },
                { "type": "text", "text": "On it." },
                { "type": "tool_use", "id": "t1", "name": "Edit", "input": { "file_path": file.to_string_lossy(), "old_string": long, "new_string": "y" } },
                { "type": "tool_use", "id": "t2", "name": "Bash", "input": { "command": "cargo   test\n  --lib" } },
                { "type": "tool_use", "id": "t3", "name": "WebFetch", "input": { "url": "https://docs.rs/serde/latest" } },
                { "type": "tool_use", "id": "t4", "name": "Task", "input": { "description": "audit", "subagent_type": "Explore", "prompt": "look" } },
                { "type": "tool_use", "id": "t5", "name": "TodoWrite", "input": { "todos": [{ "status": "completed" }, { "status": "pending" }] } },
            ],
            "usage": { "input_tokens": 3, "output_tokens": 4, "cache_read_input_tokens": 1, "cache_creation_input_tokens": 2 },
        } }).to_string());
        s.handle_line(&json!({ "type": "user", "message": { "content": [
            { "type": "tool_result", "tool_use_id": "t1", "content": "ok" },
            { "type": "tool_result", "tool_use_id": "t2", "content": [{ "type": "text", "text": "boom" }, { "type": "text", "text": "bang" }], "is_error": true },
            { "type": "text", "text": "ignored" },
        ] } }).to_string());
        s.handle_line(&json!({ "type": "result", "subtype": "success", "result": "shipped", "num_turns": 3, "duration_ms": 42, "total_cost_usd": 0.5 }).to_string());

        let events = events.lock().unwrap();
        let find = |kind: &str| -> Vec<&Value> {
            events
                .iter()
                .filter(|(k, _)| k == kind)
                .map(|(_, d)| d)
                .collect()
        };
        assert!(
            events.iter().all(|(_, d)| d["sessionId"] == "c1"),
            "every event carries the session id"
        );
        assert!(find("session.state").iter().any(|d| d["state"] == "ready"));
        let routed = find("endpoint.routed");
        assert_eq!(routed[0]["endpointId"], "cloud");
        assert_eq!(
            routed[0]["model"], "claude-real",
            "the harness's model wins over the configured one"
        );
        assert_eq!(routed[0]["reason"], "session start");
        assert_eq!(find("session.thinking")[0]["text"], "let me think");
        let messages = find("session.message");
        assert_eq!(
            messages.len(),
            2,
            "user turn plus one non-blank assistant text"
        );
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["text"], "On it.");

        let tools = find("session.tool_use");
        assert_eq!(tools.len(), 5);
        assert_eq!(tools[0]["toolId"], "t1");
        assert_eq!(tools[0]["name"], "Edit");
        assert_eq!(tools[0]["workspaceId"], "cameo");
        assert_eq!(tools[0]["dir"], "src");
        assert_eq!(tools[0]["path"], "src/lib.rs");
        assert_eq!(tools[0]["summary"], "Edit src/lib.rs");
        assert_eq!(tools[0]["command"], Value::Null);
        let truncated = tools[0]["input"]["old_string"].as_str().unwrap();
        assert_eq!(truncated.chars().count(), 601);
        assert!(truncated.ends_with('…'));
        assert_eq!(tools[1]["command"], "cargo test --lib");
        assert_eq!(tools[1]["summary"], "$ cargo test --lib");
        assert_eq!(tools[1]["workspaceId"], "cameo");
        assert_eq!(tools[1]["dir"], "");
        assert_eq!(tools[2]["summary"], "Fetch docs.rs");
        assert_eq!(tools[3]["summary"], "Delegate → Explore");
        assert_eq!(tools[4]["summary"], "Plan 1/2");
        let browser = find("browser.navigated");
        assert_eq!(browser[0]["url"], "https://docs.rs/serde/latest");
        assert_eq!(browser[0]["domain"], "docs.rs");
        let delegated = find("session.delegated");
        assert_eq!(delegated[0]["parentSessionId"], "c1");
        assert_eq!(delegated[0]["childSessionId"], "c1:t4");
        assert_eq!(delegated[0]["description"], "audit");
        assert_eq!(delegated[0]["subagentType"], "Explore");
        let progress = find("session.progress");
        assert_eq!(progress[0]["done"], 1);
        assert_eq!(progress[0]["total"], 2);

        let results = find("session.tool_result");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["toolId"], "t1");
        assert_eq!(results[0]["name"], "Edit");
        assert_eq!(results[0]["ok"], true);
        assert_eq!(results[0]["preview"], "ok");
        assert_eq!(results[1]["name"], "Bash");
        assert_eq!(results[1]["ok"], false);
        assert_eq!(results[1]["preview"], "boom bang");

        let usage = find("session.usage");
        assert_eq!(usage.len(), 2, "streamed usage plus the result's cost only");
        assert_eq!(usage[0]["contextTokens"], 10);
        assert_eq!(usage[0]["deltaInput"], 3);
        assert_eq!(usage[1], &json!({ "sessionId": "c1", "costUsd": 0.5 }));
        assert_eq!(s.info().state, "idle");
        let complete = find("session.turn_complete");
        assert_eq!(complete[0]["result"], "shipped");
        assert_eq!(complete[0]["turns"], 3);
        assert_eq!(find("session.state").last().unwrap()["detail"], "success");
        drop(events);

        assert!(
            !s.decide_permission(&json!(1), "allow"),
            "permissions go through the bridge"
        );
        assert!(s.cancel());
        assert_eq!(s.info().state, "cancelled");
        assert!(!s.is_running());
    }

    #[test]
    fn describe_tool_covers_the_tools_js_cases() {
        let ws = vec![workspace()];
        let root = cwd();
        let src = root.join("src");
        let grep = describe_tool(
            "Grep",
            &json!({ "pattern": "fn main", "path": src.to_string_lossy() }),
            &ws,
            &root,
        );
        assert_eq!(grep.summary, "Grep /fn main/");
        assert_eq!(grep.dir.as_deref(), Some("src"));
        assert_eq!(grep.workspace_id.as_deref(), Some("cameo"));
        let glob = describe_tool("Glob", &json!({ "pattern": "**/*.rs" }), &ws, &root);
        assert_eq!(glob.summary, "Glob **/*.rs");
        assert_eq!(glob.dir.as_deref(), Some(""));
        let missing = describe_tool(
            "Write",
            &json!({ "file_path": src.join("gone").join("new.rs").to_string_lossy() }),
            &ws,
            &root,
        );
        assert!(missing.is_edit);
        assert_eq!(missing.path.as_deref(), Some("src/gone/new.rs"));
        assert_eq!(missing.dir.as_deref(), Some("src/gone"));
        let outside = describe_tool(
            "Read",
            &json!({ "file_path": "/definitely/elsewhere/file.txt" }),
            &ws,
            &root,
        );
        assert!(outside.workspace_id.is_none());
        assert_eq!(outside.summary, "Read /definitely/elsewhere/file.txt");
        let longname = describe_tool(
            "Read",
            &json!({ "file_path": format!("/x/{}", "a".repeat(60)) }),
            &ws,
            &root,
        );
        assert!(longname.summary.starts_with("Read …"));
        assert_eq!(longname.summary.chars().count(), "Read ".len() + 48);
        let search = describe_tool("WebSearch", &json!({ "query": "rust" }), &ws, &root);
        assert_eq!(search.summary, "Search \"rust\"");
        let mcp = describe_tool(
            "mcp__browser__navigate",
            &json!({ "url": "https://example.org/x" }),
            &ws,
            &root,
        );
        assert_eq!(mcp.summary, "mcp__browser__navigate example.org");
        assert_eq!(mcp.browser.as_ref().unwrap().domain, "example.org");
        let bad_url = describe_tool("WebFetch", &json!({ "url": "not a url" }), &ws, &root);
        assert!(bad_url.browser.is_none());
        assert_eq!(bad_url.summary, "Fetch not a url");
        let agent = describe_tool("Agent", &json!({ "prompt": "p".repeat(100) }), &ws, &root);
        let delegation = agent.delegation.unwrap();
        assert_eq!(delegation.description.len(), 80);
        assert_eq!(delegation.subagent_type, "general-purpose");
        let other = describe_tool("Unknown", &json!({}), &ws, &root);
        assert_eq!(other.summary, "Unknown");
        assert_eq!(other.workspace_id.as_deref(), Some("cameo"));
        assert_eq!(truncate_input(None), Value::Null);
        assert_eq!(truncate_input(Some(&json!("s"))), json!("s"));
        assert!(!locate("", &ws).is_some_and(|_| true));
        let stem = resolve_claude_binary();
        assert!(stem.file_stem().is_some());
    }

    #[test]
    fn restart_re_mints_the_permission_capability() {
        let minted = Arc::new(Mutex::new(Vec::new()));
        let registered = Arc::new(Mutex::new(Vec::new()));
        let mint_log = Arc::clone(&minted);
        let reg_log = Arc::clone(&registered);
        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let bridge = PermissionBridge {
            api_base: "http://127.0.0.1:7749".into(),
            mint: Arc::new(move |id: &str| {
                let n = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let token = format!("{id}-token-{n}");
                mint_log.lock().unwrap().push(token.clone());
                Some(token)
            }),
            register_secret: Some(Arc::new(move |secret: &str| {
                reg_log.lock().unwrap().push(secret.to_string())
            })),
        };
        let initial = mcp_config(
            Path::new("/bin/knossos"),
            "http://127.0.0.1:7749",
            "r1",
            "first",
        );
        let (s, _) = session(ClaudeOptions {
            id: "r1".into(),
            cwd: cwd(),
            mcp_config: Some(initial.clone()),
            permission_tool: Some("mcp__field__approve".into()),
            permission_bridge: Some(bridge),
            ..Default::default()
        });
        // The first start keeps the capability it was created with.
        s.refresh_permission_token().unwrap();
        assert_eq!(s.mcp_config().as_deref(), Some(initial.as_str()));
        s.inner.lock().unwrap().first_start = false;
        s.refresh_permission_token().unwrap();
        let config: Value = serde_json::from_str(&s.mcp_config().unwrap()).unwrap();
        assert_eq!(
            config["mcpServers"]["field"]["env"]["FIELD_INTERNAL_TOKEN"],
            "r1-token-0"
        );
        assert_eq!(config["mcpServers"]["field"]["env"]["FIELD_SESSION"], "r1");
        assert_eq!(*registered.lock().unwrap(), vec!["r1-token-0".to_string()]);
        assert_eq!(minted.lock().unwrap().len(), 1);
        let args = s.build_args(true);
        let pos = args.iter().position(|a| a == "--mcp-config").unwrap();
        assert!(args[pos + 1].contains("r1-token-0"));

        // A mint that fails refuses the restart rather than launching unguarded.
        let (unguarded, _) = session(ClaudeOptions {
            id: "r2".into(),
            cwd: cwd(),
            permission_bridge: Some(PermissionBridge {
                api_base: "http://127.0.0.1:7749".into(),
                mint: Arc::new(|_| None),
                register_secret: None,
            }),
            ..Default::default()
        });
        unguarded.inner.lock().unwrap().first_start = false;
        assert!(unguarded
            .refresh_permission_token()
            .unwrap_err()
            .contains("capability service is unavailable"));
    }
}
