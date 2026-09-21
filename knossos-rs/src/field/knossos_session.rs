//! The Knossos adapter: `knossos serve` over NDJSON. Port of
//! `field/server/src/harness/knossos-session.js`.
//!
//! `serve` is a long-lived protocol, commands on stdin and events on stdout.
//! This adapter translates those events into the Field vocabulary, so a
//! campaign does not care which harness powers a unit. The child is the
//! `knossos` binary: `FIELD_KNOSSOS_BIN`, this very executable when it is
//! `knossos`, a sibling build, or `knossos` on the PATH. Running the loop
//! in-process is the planned follow-up once the Talos builder moves into
//! the library.

use super::adapter::{stamp, Adapter, BeforeStart, EventSink, SessionInfo};
use super::child_env::{build_child_environment, process_environment};
use super::js::{get, get_arr, get_str, js_string, or_null};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, Weak};

/// Where the `knossos` binary is.
pub fn resolve_knossos_binary() -> PathBuf {
    if let Some(explicit) = std::env::var_os("FIELD_KNOSSOS_BIN").filter(|v| !v.is_empty()) {
        return PathBuf::from(explicit);
    }
    if let Ok(exe) = std::env::current_exe() {
        if exe.file_stem().and_then(|s| s.to_str()) == Some("knossos") {
            return exe;
        }
    }
    let exe_name = if cfg!(windows) {
        "knossos.exe"
    } else {
        "knossos"
    };
    let legacy = if cfg!(windows) {
        "daedalus.exe"
    } else {
        "daedalus"
    };
    let cwd = std::env::current_dir().unwrap_or_default();
    let roots = [
        cwd.join("..").join("knossos-rs"),
        cwd.join("..").join("Knossos-Harness").join("knossos-rs"),
        cwd.join("..").join("knossos-harness").join("knossos-rs"),
        cwd.join("..").join("daedalus").join("knossos-rs"),
        cwd.join("..")
            .join("..")
            .join("daedalus")
            .join("knossos-rs"),
    ];
    for root in roots {
        for (profile, name) in [
            ("release", exe_name),
            ("debug", exe_name),
            ("release", legacy),
            ("debug", legacy),
        ] {
            let candidate = root.join("target").join(profile).join(name);
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    PathBuf::from("knossos")
}

/// Everything a Knossos session is created with.
#[derive(Debug, Clone, Default)]
pub struct KnossosOptions {
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
    pub engine: Option<String>,
    pub provider_kind: Option<String>,
    pub read_only: bool,
    pub environment_scope: Option<String>,
    pub credential_env_keys: Vec<String>,
    /// The binary to run; `None` resolves at start.
    pub binary: Option<PathBuf>,
}

struct Running {
    generation: u64,
    stdin: Box<dyn Write + Send>,
    child: Option<Child>,
}

struct Inner {
    running: Option<Running>,
    /// Processes owned, including ones still draining after a stop.
    owned: usize,
    generation: u64,
    started: bool,
    has_task: bool,
    pending_orders: Option<String>,
    stderr: String,
}

pub struct KnossosSession {
    /// A handle to ourselves, so the reader threads a start spawns can own
    /// the session without a separate launch interface.
    me: Weak<KnossosSession>,
    info: Mutex<SessionInfo>,
    opts: KnossosOptions,
    sink: EventSink,
    inner: Mutex<Inner>,
    before_start: Mutex<Option<BeforeStart>>,
}

impl KnossosSession {
    pub fn new(opts: KnossosOptions, sink: EventSink) -> Arc<KnossosSession> {
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
        Arc::new_cyclic(|me| KnossosSession {
            me: me.clone(),
            info: Mutex::new(info),
            opts,
            sink,
            inner: Mutex::new(Inner {
                running: None,
                owned: 0,
                generation: 0,
                started: false,
                has_task: false,
                pending_orders: None,
                stderr: String::new(),
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

    /// The `serve` argument list.
    pub fn build_args(&self) -> Vec<String> {
        // Every Field-launched mission is checkpointed: without the flag
        // `serve` keeps the mission in memory only, so a restart lost it and
        // accept/revert had nothing to decide.
        let mut args = vec![
            "serve".to_string(),
            "--workspace".into(),
            self.opts.cwd.to_string_lossy().into_owned(),
            "--engine".into(),
            self.opts.engine.clone().unwrap_or_else(|| "cameo".into()),
            "--persist-conversation".into(),
        ];
        if let Some(model) = &self.opts.model {
            args.push("--model".into());
            args.push(model.clone());
        }
        if self.opts.read_only
            || matches!(
                self.opts.environment_scope.as_deref(),
                Some("snapshot" | "production-readonly")
            )
        {
            args.push("--dry-run".into());
        }
        args
    }

    /// The environment the child receives.
    pub fn child_environment(&self) -> BTreeMap<String, String> {
        let mut overrides = self.opts.env.clone();
        overrides.insert("KNOSSOS_SESSION_ID".into(), self.opts.id.clone());
        overrides.insert("FIELD_SESSION_ID".into(), self.opts.id.clone());
        build_child_environment(
            &process_environment(),
            self.opts.provider_kind.as_deref(),
            &self.opts.credential_env_keys,
            &overrides,
        )
    }

    /// Write one command line to the child's stdin.
    pub fn write(&self, command: Value) -> bool {
        let Ok(mut inner) = self.inner.lock() else {
            return false;
        };
        let Some(running) = inner.running.as_mut() else {
            return false;
        };
        let mut line = command.to_string();
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

    fn has_task(&self) -> bool {
        self.inner.lock().map(|i| i.has_task).unwrap_or(false)
    }

    fn spawn_child(&self, orders: Option<String>) -> Result<(), String> {
        let this = self.me.upgrade().ok_or("session is being dropped")?;
        let hook = self.before_start.lock().ok().and_then(|h| h.clone());
        if let Some(hook) = hook {
            hook()?;
        }
        let binary = self
            .opts
            .binary
            .clone()
            .unwrap_or_else(resolve_knossos_binary);
        let env = self.child_environment();
        let mut command = Command::new(&binary);
        command
            .args(self.build_args())
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
                    json!({ "reason": "error", "error": format!("Knossos spawn failed: {error}") }),
                );
                return Ok(());
            }
        };
        let stdin = child.stdin.take().ok_or("child stdin is not piped")?;
        let stdout = child.stdout.take().ok_or("child stdout is not piped")?;
        let stderr = child.stderr.take().ok_or("child stderr is not piped")?;
        let generation = {
            let mut inner = self.inner.lock().map_err(|_| "session lock poisoned")?;
            inner.generation += 1;
            inner.owned += 1;
            inner.started = true;
            inner.pending_orders = Some(orders.unwrap_or_else(|| "Await orders.".into()));
            inner.stderr.clear();
            let generation = inner.generation;
            inner.running = Some(Running {
                generation,
                stdin: Box::new(stdin),
                child: None,
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
            let status = child.wait();
            waiter.on_close(generation, status.ok().and_then(|s| s.code()));
        });
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
                    "Knossos exited {}",
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

    fn stop_child(&self) {
        let child = self
            .inner
            .lock()
            .ok()
            .and_then(|mut inner| inner.running.take().and_then(|r| r.child));
        if let Some(mut child) = child {
            let _ = child.kill();
        }
        // The child's stdin is dropped with `running`; a `serve` loop whose
        // command channel closes shuts down, and the waiter thread reaps it.
    }

    /// Translate one `serve` event line into Field events.
    pub fn handle_line(&self, line: &str) {
        let Ok(msg) = serde_json::from_str::<Value>(line.trim()) else {
            return;
        };
        match get_str(&msg, "event") {
            Some("ready") => {
                self.set_state("ready");
                self.emit("session.state", json!({ "state": "ready" }));
                let engine = get_str(&msg, "engine")
                    .map(str::to_string)
                    .or_else(|| self.opts.engine.clone())
                    .unwrap_or_else(|| "cameo".into());
                self.emit(
                    "endpoint.routed",
                    json!({
                        "endpointId": self.opts.endpoint_id, "model": self.opts.model,
                        "reason": format!("Knossos ready via {engine}"),
                    }),
                );
                self.write(json!({ "cmd": "capabilities", "permissions": true }));
                if let Some(orders) = self.take_pending_orders() {
                    self.send(&orders);
                }
            }
            Some("plan") => {
                let steps = get_arr(&msg, "steps").cloned().unwrap_or_default();
                self.emit(
                    "session.progress",
                    json!({ "done": 0, "total": steps.len(), "steps": steps }),
                );
            }
            Some("outcome") => {
                let succeeded = msg
                    .get("succeeded")
                    .is_some_and(|v| v.as_bool().unwrap_or(false));
                let state = if succeeded { "idle" } else { "blocked" };
                self.set_state(state);
                if let Some(summary) = get_str(&msg, "summary").filter(|s| !s.is_empty()) {
                    self.emit(
                        "session.message",
                        json!({ "role": "assistant", "text": summary }),
                    );
                }
                self.emit(
                    "session.state",
                    json!({ "state": state, "detail": or_null(msg.get("halt")) }),
                );
                let summary: String = get(&msg, "summary")
                    .map(js_string)
                    .unwrap_or_default()
                    .chars()
                    .take(4000)
                    .collect();
                self.emit(
                    "session.turn_complete",
                    json!({
                        "result": summary,
                        "turns": or_null(msg.get("steps_used")), "isError": !succeeded,
                        "changedFiles": get_arr(&msg, "changed").cloned().unwrap_or_default(),
                        "dryRun": msg.get("dry_run").is_some_and(|v| v.as_bool().unwrap_or(false)),
                        "residualRisk": get_arr(&msg, "residual_risk").cloned().unwrap_or_default(),
                        "recovery": get_arr(&msg, "recovery").cloned().unwrap_or_default(),
                    }),
                );
            }
            Some("verdict") => {
                self.emit(
                    "session.verification",
                    json!({
                        "passed": msg.get("passed").is_some_and(|v| v.as_bool().unwrap_or(false)),
                        "summary": get(&msg, "summary").cloned().unwrap_or(json!("")),
                        "tiers": get_arr(&msg, "tiers").cloned().unwrap_or_default(),
                        "dryRun": msg.get("dry_run").is_some_and(|v| v.as_bool().unwrap_or(false)),
                    }),
                );
            }
            Some("permission_request") => {
                self.emit(
                    "harness.permission_requested",
                    json!({ "requestId": msg.get("id"), "toolName": msg.get("tool"), "input": msg.get("input") }),
                );
            }
            Some("error") => {
                self.set_state("error");
                self.emit(
                    "session.state",
                    json!({ "state": "error", "detail": get(&msg, "message").cloned().unwrap_or(json!("Knossos error")) }),
                );
            }
            Some("idle") => {
                let state = self.state();
                if state != "blocked" && state != "error" {
                    self.set_state("idle");
                }
                self.emit("session.state", json!({ "state": self.state() }));
            }
            _ => {}
        }
    }
}

impl Adapter for KnossosSession {
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

    fn send(&self, text: &str) -> bool {
        if !self.is_running() || self.state() == "paused" {
            return false;
        }
        let has_task = self.has_task();
        let full = if has_task || self.opts.system_prompt.is_empty() {
            text.to_string()
        } else {
            format!(
                "{}\n\n---\n\n# Active orders\n\n{text}",
                self.opts.system_prompt
            )
        };
        let cmd = if has_task { "resume" } else { "task" };
        if !self.write(json!({ "cmd": cmd, "text": full })) {
            return false;
        }
        if let Ok(mut inner) = self.inner.lock() {
            inner.has_task = true;
        }
        self.emit("session.message", json!({ "role": "user", "text": text }));
        self.set_state("thinking");
        self.emit("session.state", json!({ "state": "thinking" }));
        true
    }

    fn pause(&self) -> bool {
        if !self.is_running() {
            return false;
        }
        // `serve` cannot cancel an in-flight turn through its queued command
        // channel; stopping the process is the only honest immediate pause.
        // Workspace and trace are durable; resume starts a fresh turn and
        // says so.
        self.set_state("paused");
        self.stop_child();
        self.emit(
            "session.state",
            json!({ "state": "paused", "detail": "Knossos process stopped; workspace and trace retained" }),
        );
        true
    }

    fn resume(&self, orders: Option<String>) -> Result<bool, String> {
        if self.is_running() {
            return Ok(false);
        }
        if let Ok(mut inner) = self.inner.lock() {
            inner.has_task = false;
        }
        self.spawn_child(Some(orders.unwrap_or_else(|| {
            "Recover from the durable workspace and trace, restate current progress, and continue.".into()
        })))?;
        self.emit(
            "session.state",
            json!({ "state": "thinking", "detail": "Knossos restarted from durable workspace state" }),
        );
        Ok(true)
    }

    fn cancel(&self) -> bool {
        self.set_state("cancelled");
        self.stop_child();
        self.emit("session.ended", json!({ "reason": "cancelled" }));
        true
    }

    fn decide_permission(&self, request_id: &Value, decision: &str) -> bool {
        self.write(json!({ "cmd": "permission", "id": request_id, "allow": decision == "allow" }))
    }

    fn capabilities(&self) -> Value {
        json!({
            "kind": "ndjson-serve", "duplex": true, "resumable": true, "permissions": "inline-handshake",
            "delegation": false, "browser": false, "verification": true, "dryRun": true,
        })
    }
}

#[allow(dead_code)]
fn _path_probe(p: &Path) -> bool {
    p.is_file()
}

#[cfg(test)]
mod tests {
    //! `field/server/test/knossos-session.test.mjs`, case for case.
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

    fn session(opts: KnossosOptions) -> (Arc<KnossosSession>, Events) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink_events = Arc::clone(&events);
        let sink: EventSink = Arc::new(move |kind: &str, data: Value| {
            sink_events.lock().unwrap().push((kind.to_string(), data))
        });
        (KnossosSession::new(opts, sink), events)
    }

    #[test]
    fn cameo_adapter_event_translation_and_permission_handshake() {
        let cwd = std::env::current_dir().unwrap();
        let (s, events) = session(KnossosOptions {
            id: "k1".into(),
            agent_id: Some("ornith-1".into()),
            name: Some("Ornith".into()),
            role: Some("builder".into()),
            model: Some("ornith-35b-a3b".into()),
            endpoint_id: Some("cameod-local".into()),
            cwd: cwd.clone(),
            workspace_id: Some("cameo".into()),
            system_prompt: "constitution".into(),
            engine: Some("cameo".into()),
            ..Default::default()
        });
        let writes = Arc::new(Mutex::new(Vec::new()));
        s.attach_writer(Box::new(Shared(Arc::clone(&writes))));
        s.set_pending_orders("build the gateway");

        s.handle_line(&json!({ "event": "ready", "workspace": cwd, "engine": "cameo", "files": 10, "symbols": 20 }).to_string());
        assert_eq!(s.info().state, "thinking");
        {
            let w = writes.lock().unwrap();
            assert_eq!(w[0], json!({ "cmd": "capabilities", "permissions": true }));
            assert_eq!(w[1]["cmd"], "task");
            let text = w[1]["text"].as_str().unwrap();
            assert!(text.contains("constitution") && text.contains("build the gateway"));
        }

        s.handle_line(
            &json!({ "event": "plan", "steps": ["inspect", "edit", "verify"] }).to_string(),
        );
        s.handle_line(&json!({ "event": "verdict", "passed": true, "summary": "green", "tiers": [{ "tier": 1, "passed": true }] }).to_string());
        s.handle_line(
            &json!({
                "event": "outcome", "halt": "done", "succeeded": true, "steps_used": 3,
                "summary": "complete\nFIELD_REPORT: {\"kind\":\"objective_satisfied\",\"evidence\":[\"green\"]}",
                "changed": ["gateway.rs"], "dry_run": false,
            })
            .to_string(),
        );
        s.handle_line(&json!({ "event": "permission_request", "id": 7, "tool": "write", "input": { "path": "x" } }).to_string());

        let events = events.lock().unwrap();
        assert!(events
            .iter()
            .any(|(k, d)| k == "session.progress" && d["total"] == 3));
        assert!(events
            .iter()
            .any(|(k, d)| k == "session.verification" && d["passed"] == true));
        assert!(events.iter().any(|(k, d)| k == "session.turn_complete"
            && d["result"].as_str().unwrap().contains("FIELD_REPORT")));
        assert!(events
            .iter()
            .any(|(k, d)| k == "harness.permission_requested" && d["requestId"] == 7));
        assert!(
            events.iter().all(|(_, d)| d["sessionId"] == "k1"),
            "every event carries the session id"
        );
        drop(events);

        assert!(s.decide_permission(&json!(7), "allow"));
        assert_eq!(
            writes.lock().unwrap().last().unwrap(),
            &json!({ "cmd": "permission", "id": 7, "allow": true })
        );
        assert!(!s.build_args().contains(&"--dry-run".to_string()));

        let (read_only, _) = session(KnossosOptions {
            id: "k2".into(),
            cwd: cwd.clone(),
            engine: Some("cameo".into()),
            read_only: true,
            ..Default::default()
        });
        assert!(read_only.build_args().contains(&"--dry-run".to_string()));
        let (snapshot, _) = session(KnossosOptions {
            id: "k3".into(),
            cwd,
            engine: Some("cameo".into()),
            environment_scope: Some("production-readonly".into()),
            ..Default::default()
        });
        assert!(snapshot.build_args().contains(&"--dry-run".to_string()));
        let resolved = resolve_knossos_binary();
        let stem = resolved.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        assert!(matches!(stem, "knossos" | "daedalus"), "{resolved:?}");
    }
}
