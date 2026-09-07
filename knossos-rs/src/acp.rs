//! ACP: speak the Agent Client Protocol so an editor can drive the harness.
//!
//! ACP is JSON-RPC 2.0 over newline-delimited stdio, which
//! [`jsonrpc::Peer`](crate::jsonrpc::Peer) already does. What is here is the
//! agent half of the conversation: sessions, a turn, and the narration an
//! editor renders while the turn runs.
//!
//! | Direction | Method | Meaning |
//! |---|---|---|
//! | in | `initialize` | version and capability exchange |
//! | in | `session/new` | a workspace to work in |
//! | in | `session/load` | replay a session's updates without re-running tools |
//! | in | `session/prompt` | run a turn, block until it ends |
//! | in | `session/cancel` | stop the running turn (a notification) |
//! | in | `session/list` | ids of live sessions |
//! | in | `session/close` | drop a session (alias: `session/delete`) |
//! | in | `session/set_mode` | `ask` / `preview` / `write` |
//! | in | `session/interject` | steer a running turn at the next step boundary |
//! | out | `session/update` | what the agent is doing, as it happens |
//! | out | `session/request_permission` | ask before something consequential |
//!
//! # stdout is the protocol
//!
//! Every human-readable byte goes to stderr. A single stray `println!` on this
//! path corrupts the stream and the editor drops the connection, which is why
//! progress reaches the client through [`crate::session::Session::with_sink`] rather than
//! through the stdout streaming `serve` uses.
//!
//! # Cancellation is why the fast path exists
//!
//! `session/cancel` arrives *during* the `session/prompt` it is meant to
//! interrupt. Queued behind that prompt on the worker thread it could only be
//! delivered once the turn had already finished, so it is dispatched on the
//! reader thread instead — see [`Peer::with_fast_path`](crate::jsonrpc::Peer::with_fast_path).
//! It sets [`Talos::cancel`], which the loop reads at its next step boundary.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use serde_json::{json, Value};

use crate::ariadne::Halt;
use crate::jsonrpc::{PeerHandle, RpcError, INVALID_PARAMS, METHOD_NOT_FOUND};
use crate::metis::{self, Plan};
use crate::session::TraceEvent;
use crate::talos::{Approver, Talos};

/// The protocol version this agent implements.
pub const PROTOCOL_VERSION: u32 = 1;

/// Methods that must not wait behind a running turn.
pub fn is_fast_path(method: &str) -> bool {
    method == "session/cancel" || method == "session/interject"
}

/// `KNOSSOS_SCRIPT` sessions start in ask/preview/write the way Python
/// `scripted_agent.py` mapped `KNOSSOS_EXECUTE` / `KNOSSOS_WRITE`.
fn scripted_default_mode() -> (String, bool) {
    if std::env::var("KNOSSOS_SCRIPT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .is_none()
    {
        return ("write".into(), false);
    }
    let write = std::env::var("KNOSSOS_WRITE").ok().as_deref() == Some("1");
    let execute = std::env::var("KNOSSOS_EXECUTE").ok().as_deref() == Some("1");
    if !execute {
        ("ask".into(), true)
    } else if write {
        ("write".into(), false)
    } else {
        ("preview".into(), true)
    }
}

fn mode_state(current: &str) -> Value {
    json!({
        "currentModeId": current,
        "availableModes": [
            {
                "id": "ask",
                "name": "Ask",
                "description": "Investigate and answer without writing files."
            },
            {
                "id": "preview",
                "name": "Preview",
                "description": "Stage proposed edits in memory for review."
            },
            {
                "id": "write",
                "name": "Write",
                "description": "Apply approved edits to the workspace."
            }
        ]
    })
}

/// Builds the agent for one workspace. Supplied by the binary, which is what
/// knows how to assemble a [`Talos`].
pub type BuildAgent =
    dyn Fn(&Path, Arc<AtomicBool>, Arc<dyn Approver>) -> Result<Talos> + Send + Sync;

struct Live {
    talos: Mutex<Talos>,
    cancel: Arc<AtomicBool>,
    /// Whether a turn has already run, which decides `run` versus `resume`.
    started: AtomicBool,
    cwd: PathBuf,
    mode: Mutex<String>,
    /// Cloned handle so `session/interject` on the fast path never waits on
    /// the Talos lock held by a running prompt.
    interjections: crate::interject::Interjections,
    /// Every `session/update` already sent. `session/load` replays this
    /// list; it does not re-execute the turn.
    history: Arc<Mutex<Vec<Value>>>,
}

/// The agent side of an ACP conversation.
pub struct Agent {
    peer: PeerHandle,
    build: Arc<BuildAgent>,
    sessions: Mutex<HashMap<String, Arc<Live>>>,
    next_session: AtomicU64,
    /// Captured at construction, not looked up when a turn starts.
    ///
    /// Handlers run on the peer's worker thread, which is a plain OS thread —
    /// `Handle::current()` there finds nothing however many runtimes exist
    /// elsewhere in the process. Being a foreign thread is also what makes
    /// `block_on` legal: it parks that thread rather than a runtime worker.
    ///
    /// The runtime must be multi-threaded, since the thread that would
    /// otherwise drive a current-thread runtime is the one blocking on it.
    runtime: tokio::runtime::Handle,
    caps: Mutex<crate::client_io::ClientCaps>,
}

impl Agent {
    pub fn new(peer: PeerHandle, build: Arc<BuildAgent>, runtime: tokio::runtime::Handle) -> Self {
        Agent {
            peer,
            build,
            sessions: Mutex::new(HashMap::new()),
            next_session: AtomicU64::new(0),
            runtime,
            caps: Mutex::new(crate::client_io::ClientCaps::default()),
        }
    }

    /// Route one request. Errors become JSON-RPC errors; the peer replies.
    pub fn handle(&self, method: &str, params: Option<Value>) -> Result<Value, RpcError> {
        let params = params.unwrap_or(Value::Null);
        match method {
            "initialize" => Ok(self.initialize(&params)),
            "authenticate" => Ok(json!({})),
            "session/new" => self.new_session(&params),
            "session/load" => self.load_session(&params),
            "session/fork" => self.fork_session(&params),
            "session/prompt" => self.prompt(&params),
            "session/cancel" => {
                self.cancel(&params);
                Ok(Value::Null)
            }
            "session/list" => Ok(self.list_sessions(&params)),
            "session/close" | "session/delete" => self.close_session(&params),
            "session/set_mode" => self.set_mode(&params),
            "session/set_context" => self.set_context(&params),
            "session/interject" => self.interject(&params),
            // Answered rather than ignored: an unanswered request blocks the
            // editor for as long as it is willing to wait.
            other => Err(RpcError::new(
                METHOD_NOT_FOUND,
                format!("{other} is not supported by this agent"),
            )),
        }
    }

    fn initialize(&self, params: &Value) -> Value {
        *self.caps.lock().expect("caps") = crate::client_io::ClientCaps::from_initialize(params);
        json!({
            "protocolVersion": PROTOCOL_VERSION,
            "agentInfo": {
                "name": "knossos",
                "title": "Knossos",
                "version": env!("CARGO_PKG_VERSION"),
            },
            "agentCapabilities": {
                "loadSession": true,
                "unstable_forkSession": true,
                "sessionCapabilities": {"list": {}, "close": {}, "delete": {}},
                "daedalusCapabilities": {
                    "contextPolicy": true,
                    "interject": true,
                    "checkpointReplay": true,
                },
                "promptCapabilities": {
                    "image": false,
                    "audio": false,
                    "embeddedContext": true,
                },
            },
            "authMethods": [],
        })
    }

    fn new_session(&self, params: &Value) -> Result<Value, RpcError> {
        let cwd = params
            .get("cwd")
            .and_then(Value::as_str)
            .ok_or_else(|| RpcError::new(INVALID_PARAMS, "session/new needs an absolute cwd"))?;
        let root = PathBuf::from(cwd);
        if !root.is_absolute() {
            return Err(RpcError::new(
                INVALID_PARAMS,
                format!("cwd is not absolute: {cwd}"),
            ));
        }

        let id = format!("s{}", self.next_session.fetch_add(1, Ordering::SeqCst) + 1);
        let cancel = Arc::new(AtomicBool::new(false));
        let approver: Arc<dyn Approver> = Arc::new(AcpApprover {
            peer: self.peer.clone(),
            session: id.clone(),
            cancel: Arc::clone(&cancel),
        });

        let mut talos = (self.build)(&root, Arc::clone(&cancel), approver)
            .map_err(|e| RpcError::new(INVALID_PARAMS, format!("cannot open {cwd}: {e}")))?;
        if let Some((window, compact_at)) = context_overrides(params)? {
            talos.set_context_limits(window, compact_at);
        }
        if scripted_default_mode().1 {
            talos.ctx.set_dry_run(true);
        }
        let caps = self.caps.lock().expect("caps").clone();
        if caps.any_fs() || caps.terminal || caps.elicit {
            talos
                .ctx
                .set_client(std::sync::Arc::new(crate::client_io::ClientIo::new(
                    self.peer.clone(),
                    id.clone(),
                    Arc::clone(&cancel),
                    caps,
                )));
        }

        // Narration. Installed on the session rather than printed, because
        // stdout is the protocol channel.
        let history = Arc::new(Mutex::new(Vec::new()));
        let notifier = Notifier {
            peer: self.peer.clone(),
            session: id.clone(),
            root: root.clone(),
            history: Arc::clone(&history),
        };
        talos.session = std::mem::replace(
            &mut talos.session,
            crate::session::Session::new(&root, "placeholder"),
        )
        .with_sink(Arc::new(move |event: &TraceEvent| notifier.emit(event)));

        let interjections = talos.interjections();
        let context = talos.context_budget();
        let initial_mode = scripted_default_mode().0;
        self.sessions.lock().expect("sessions").insert(
            id.clone(),
            Arc::new(Live {
                talos: Mutex::new(talos),
                cancel,
                started: AtomicBool::new(false),
                cwd: root,
                mode: Mutex::new(initial_mode.clone()),
                interjections,
                history,
            }),
        );
        crate::cameo_board::report(json!({
            "id": id,
            "name": id,
            "role": "executor",
            "mode": "ask",
            "state": "idle",
            "model": "",
            "native_context": context.engine_tokens,
            "context_window": context.assigned_tokens,
            "compact_at": context.compact_at_tokens,
        }));
        Ok(json!({
            "sessionId": id,
            "modes": mode_state(&initial_mode),
            "context": context_json(context),
        }))
    }

    fn live(&self, params: &Value) -> Result<Arc<Live>, RpcError> {
        let id = params
            .get("sessionId")
            .and_then(Value::as_str)
            .ok_or_else(|| RpcError::new(INVALID_PARAMS, "sessionId is required"))?;
        self.sessions
            .lock()
            .expect("sessions")
            .get(id)
            .cloned()
            .ok_or_else(|| RpcError::new(INVALID_PARAMS, format!("no such session: {id}")))
    }

    /// Replay already-sent updates. Not re-execution: files are not touched.
    fn load_session(&self, params: &Value) -> Result<Value, RpcError> {
        let live = self.live(params)?;
        let id = params
            .get("sessionId")
            .and_then(Value::as_str)
            .unwrap_or("");
        let history = live.history.lock().expect("history").clone();
        for update in history {
            self.peer.notify(
                "session/update",
                Some(json!({"sessionId": id, "update": update})),
            );
        }
        Ok(json!({}))
    }

    fn fork_session(&self, params: &Value) -> Result<Value, RpcError> {
        let source = self.live(params)?;
        let cwd = params
            .get("cwd")
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .unwrap_or_else(|| source.cwd.clone());
        if !cwd.is_absolute() {
            return Err(RpcError::new(
                INVALID_PARAMS,
                "cwd must be an absolute path",
            ));
        }
        let (messages, task, dry) = {
            let t = source.talos.lock().expect("session in use");
            (t.messages.clone(), t.task.clone(), t.ctx.is_dry_run())
        };
        let mode = source.mode.lock().expect("mode").clone();
        let history = source.history.lock().expect("history").clone();

        let id = format!("s{}", self.next_session.fetch_add(1, Ordering::SeqCst) + 1);
        let cancel = Arc::new(AtomicBool::new(false));
        let approver: Arc<dyn Approver> = Arc::new(AcpApprover {
            peer: self.peer.clone(),
            session: id.clone(),
            cancel: Arc::clone(&cancel),
        });
        let mut talos = (self.build)(&cwd, Arc::clone(&cancel), approver)
            .map_err(|e| RpcError::new(INVALID_PARAMS, format!("cannot fork: {e}")))?;
        let started = !messages.is_empty();
        talos.messages = messages;
        talos.task = task;
        talos.ctx.set_dry_run(dry);
        let caps = self.caps.lock().expect("caps").clone();
        if caps.any_fs() || caps.terminal || caps.elicit {
            talos
                .ctx
                .set_client(std::sync::Arc::new(crate::client_io::ClientIo::new(
                    self.peer.clone(),
                    id.clone(),
                    Arc::clone(&cancel),
                    caps,
                )));
        }
        let sink_hist = Arc::new(Mutex::new(history));
        let notifier = Notifier {
            peer: self.peer.clone(),
            session: id.clone(),
            root: cwd.clone(),
            history: Arc::clone(&sink_hist),
        };
        talos.session = std::mem::replace(
            &mut talos.session,
            crate::session::Session::new(&cwd, "placeholder"),
        )
        .with_sink(Arc::new(move |event: &TraceEvent| notifier.emit(event)));
        let interjections = talos.interjections();
        self.sessions.lock().expect("sessions").insert(
            id.clone(),
            Arc::new(Live {
                talos: Mutex::new(talos),
                cancel,
                started: AtomicBool::new(started),
                cwd,
                mode: Mutex::new(mode),
                interjections,
                history: sink_hist,
            }),
        );
        Ok(json!({"sessionId": id}))
    }

    /// Run one turn and block until it ends.
    ///
    /// Blocking is correct: ACP defines `session/prompt` as returning when the
    /// turn is over, and everything the editor should see in the meantime goes
    /// out as `session/update`.
    fn prompt(&self, params: &Value) -> Result<Value, RpcError> {
        let live = self.live(params)?;
        let text = prompt_text(params);
        if text.trim().is_empty() {
            return Err(RpcError::new(
                INVALID_PARAMS,
                "an empty prompt has nothing to do",
            ));
        }

        // A cancel that arrived between turns must not kill this one.
        live.cancel.store(false, Ordering::SeqCst);

        let mut talos = live.talos.lock().expect("session in use");
        let first = !live.started.swap(true, Ordering::SeqCst);

        // The worker thread this runs on is a plain OS thread, not a runtime
        // worker, so blocking on the async loop here starves nothing.
        let outcome = self.runtime.block_on(async {
            if first {
                talos.capture_environment()?;
                // A one-word "fix it" is not worth a planner turn. Anything
                // longer gets a real Metis plan so stepwise execution can
                // engage — the previous ACP path stuffed the prompt in as a
                // single step and the planner never ran.
                // A conformance script is defined as one reply per executor
                // turn. Spending its first reply on Metis makes the scripted
                // wire test depend on an unrelated planner call.
                let scripted = std::env::var("KNOSSOS_SCRIPT")
                    .ok()
                    .is_some_and(|value| !value.trim().is_empty());
                let plan = if !scripted && metis::worth_planning(&text) {
                    metis::plan(
                        talos.engine.as_ref(),
                        &talos.themis,
                        &talos.scribe,
                        &text,
                        talos.max_tokens,
                    )
                    .await
                    .unwrap_or_else(|_| Plan {
                        steps: vec![text.clone()],
                    })
                } else {
                    Plan {
                        steps: vec![text.clone()],
                    }
                };
                talos.run(&text, &plan).await
            } else {
                talos.resume(&text).await
            }
        });

        let outcome =
            outcome.map_err(|e| RpcError::new(-32603, format!("the turn failed: {e}")))?;
        let sid = params
            .get("sessionId")
            .and_then(Value::as_str)
            .unwrap_or("session");
        crate::cameo_board::report(json!({
            "id": sid,
            "name": sid,
            "role": "executor",
            "mode": "write",
            "state": outcome.halt.label(),
            "model": talos.engine.name(),
            "halt": outcome.halt.label(),
            "summary": outcome.summary,
            "files": outcome.changed.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
            "native_context": talos.context_budget().engine_tokens,
            "context_window": talos.context_budget().assigned_tokens,
            "compact_at": talos.context_budget().compact_at_tokens,
        }));
        Ok(json!({"stopReason": stop_reason(outcome.halt)}))
    }

    /// A notification, so nothing is returned and nothing waits.
    fn cancel(&self, params: &Value) {
        if let Ok(live) = self.live(params) {
            live.cancel.store(true, Ordering::SeqCst);
        }
    }

    fn list_sessions(&self, params: &Value) -> Value {
        let cwd = params.get("cwd").and_then(Value::as_str).map(PathBuf::from);
        let sessions = self.sessions.lock().expect("sessions");
        let list: Vec<Value> = sessions
            .iter()
            .filter(|(_, live)| cwd.as_ref().is_none_or(|wanted| wanted == &live.cwd))
            .map(|(id, live)| {
                let talos = live.talos.lock().expect("session in use");
                let title = talos
                    .task
                    .lines()
                    .next()
                    .map(str::trim)
                    .filter(|title| !title.is_empty())
                    .unwrap_or("New session");
                json!({
                    "sessionId": id,
                    "cwd": live.cwd,
                    "title": title,
                })
            })
            .collect();
        json!({ "sessions": list })
    }

    fn close_session(&self, params: &Value) -> Result<Value, RpcError> {
        let id = params
            .get("sessionId")
            .and_then(Value::as_str)
            .ok_or_else(|| RpcError::new(INVALID_PARAMS, "sessionId is required"))?;
        let mut sessions = self.sessions.lock().expect("sessions");
        if sessions.remove(id).is_none() {
            return Err(RpcError::new(
                INVALID_PARAMS,
                format!("no such session: {id}"),
            ));
        }
        Ok(json!({}))
    }

    fn set_mode(&self, params: &Value) -> Result<Value, RpcError> {
        let live = self.live(params)?;
        let mode = params
            .get("modeId")
            .or_else(|| params.get("mode"))
            .and_then(Value::as_str)
            .ok_or_else(|| RpcError::new(INVALID_PARAMS, "session/set_mode needs modeId"))?;
        let dry = match mode {
            "write" => false,
            "preview" | "ask" => true,
            other => {
                return Err(RpcError::new(
                    INVALID_PARAMS,
                    format!("unknown mode {other:?}; want ask, preview, or write"),
                ));
            }
        };
        {
            let mut talos = live.talos.lock().expect("session in use");
            talos.ctx.set_dry_run(dry);
        }
        *live.mode.lock().expect("mode") = mode.to_string();
        Ok(json!({ "mode": mode }))
    }

    fn set_context(&self, params: &Value) -> Result<Value, RpcError> {
        let live = self.live(params)?;
        let (context_window, compact_at) = context_overrides(params)?.ok_or_else(|| {
            RpcError::new(
                INVALID_PARAMS,
                "session/set_context needs contextWindow or compactAt",
            )
        })?;
        let mut talos = live.talos.try_lock().map_err(|_| {
            RpcError::new(
                INVALID_PARAMS,
                "session is busy; change context between turns",
            )
        })?;
        let context = talos.set_context_limits(context_window, compact_at);
        Ok(json!({"context": context_json(context)}))
    }

    fn interject(&self, params: &Value) -> Result<Value, RpcError> {
        let live = self.live(params)?;
        let text = params
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| {
                params
                    .get("prompt")
                    .and_then(Value::as_array)
                    .map(|blocks| {
                        blocks
                            .iter()
                            .filter_map(|b| b.get("text").and_then(Value::as_str))
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
            })
            .unwrap_or_default();
        let accepted = live.interjections.push(text);
        if !accepted {
            return Err(RpcError::new(
                INVALID_PARAMS,
                "interjection refused (empty or queue full)",
            ));
        }
        Ok(json!({ "ok": true }))
    }
}

/// Why the turn ended, in the protocol's vocabulary.
///
/// `Stuck` maps to `end_turn` because ACP has no word for it and the summary
/// already says so in prose. Reporting `refusal` would be a lie about intent —
/// the agent tried and got nowhere, which is not the same as declining.
fn stop_reason(halt: Halt) -> &'static str {
    match halt {
        Halt::Done | Halt::Stuck | Halt::Continue => "end_turn",
        Halt::BudgetExhausted => "max_turn_requests",
        Halt::Cancelled => "cancelled",
    }
}

fn context_value(params: &Value, key: &str) -> Result<Option<u32>, RpcError> {
    match params.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => {
            let number = value.as_u64().filter(|number| *number > 0).ok_or_else(|| {
                RpcError::new(
                    INVALID_PARAMS,
                    format!("{key} must be a positive 32-bit integer or null"),
                )
            })?;
            let number = u32::try_from(number).map_err(|_| {
                RpcError::new(
                    INVALID_PARAMS,
                    format!("{key} must be a positive 32-bit integer or null"),
                )
            })?;
            Ok(Some(number))
        }
    }
}

type ContextOverrides = (Option<u32>, Option<u32>);

fn context_overrides(params: &Value) -> Result<Option<ContextOverrides>, RpcError> {
    if params.get("contextWindow").is_none() && params.get("compactAt").is_none() {
        return Ok(None);
    }
    Ok(Some((
        context_value(params, "contextWindow")?,
        context_value(params, "compactAt")?,
    )))
}

fn context_json(context: crate::context::ContextBudget) -> Value {
    json!({
        "nativeContext": context.engine_tokens,
        "assignedContext": context.assigned_tokens,
        "inputLimit": context.input_limit_tokens,
        "compactAt": context.compact_at_tokens,
        "completionReserve": context.completion_reserve,
        "protocolReserve": context.protocol_reserve,
    })
}

/// Flatten an ACP prompt into text. Non-text blocks are named rather than
/// dropped, so a model told `[image]` can ask for something else.
fn prompt_text(params: &Value) -> String {
    let Some(blocks) = params.get("prompt").and_then(Value::as_array) else {
        return String::new();
    };
    blocks
        .iter()
        .map(|b| match b.get("type").and_then(Value::as_str) {
            Some("text") => b
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            Some("resource_link") => b
                .get("uri")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            other => format!("[{} content]", other.unwrap_or("unknown")),
        })
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Turns trace events into `session/update` notifications.
struct Notifier {
    peer: PeerHandle,
    session: String,
    root: PathBuf,
    history: Arc<Mutex<Vec<Value>>>,
}

impl Notifier {
    fn send(&self, update: Value) {
        self.history.lock().expect("history").push(update.clone());
        self.peer.notify(
            "session/update",
            Some(json!({"sessionId": self.session, "update": update})),
        );
    }

    fn emit(&self, event: &TraceEvent) {
        match event {
            TraceEvent::PlanProduced { steps } => self.send(json!({
                "sessionUpdate": "plan",
                "entries": steps.iter().map(|s| json!({
                    "content": s, "priority": "medium", "status": "pending",
                })).collect::<Vec<_>>(),
            })),
            TraceEvent::AgentMessage { text, .. } => self.send(json!({
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": text},
            })),
            TraceEvent::Thought { text, .. } => self.send(json!({
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": text},
            })),
            // Logged once the call has returned, so it is reported in its final
            // state rather than as pending followed by an update. An editor
            // renders one settled row instead of two.
            TraceEvent::ToolCall {
                tool,
                input,
                output,
                is_error,
                changed,
                ..
            } => self.send(json!({
                "sessionUpdate": "tool_call",
                "toolCallId": format!("{tool}-{}", short_hash(input)),
                "title": tool_title(tool, input),
                "kind": tool_kind(tool),
                "status": if *is_error { "failed" } else { "completed" },
                "content": [{
                    "type": "content",
                    "content": {"type": "text", "text": output},
                }],
                "rawInput": input,
                "rawOutput": output,
                "locations": tool_locations(&self.root, input, changed),
            })),
            // Everything else is harness bookkeeping — verdicts, compaction,
            // context decisions. Real, but not narration an editor should
            // render as agent activity.
            _ => {}
        }
    }
}

fn tool_locations(root: &Path, input: &Value, changed: &[String]) -> Vec<Value> {
    let mut paths = changed.iter().map(PathBuf::from).collect::<Vec<PathBuf>>();
    if let Some(path) = input.get("path").and_then(Value::as_str) {
        let path = PathBuf::from(path);
        let path = if path.is_absolute() {
            path
        } else {
            root.join(path)
        };
        if !paths.iter().any(|known| known == &path) {
            paths.push(path);
        }
    }
    paths
        .into_iter()
        .map(|path| json!({"path": crate::client_io::wire_path(&path)}))
        .collect()
}

/// A short, stable discriminator so two calls to the same tool are two rows.
fn short_hash(input: &Value) -> String {
    let text = input.to_string();
    let mut hash: u64 = 1469598103934665603;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    format!("{hash:x}")
}

/// ACP's tool taxonomy, which is what picks the icon in an editor.
fn tool_kind(tool: &str) -> &'static str {
    match tool {
        "read_file" | "list_dir" => "read",
        "write_file" | "edit_file" => "edit",
        "search" | "search_code" => "search",
        "run" | "verify" => "execute",
        "ask_user" | "delegate" => "think",
        _ => "other",
    }
}

fn tool_title(tool: &str, input: &Value) -> String {
    match input.get("path").and_then(Value::as_str) {
        Some(path) => format!("{tool} {path}"),
        None => match input.get("task").and_then(Value::as_str) {
            Some(task) => format!("{tool}: {}", truncate(task, 60)),
            None => tool.to_string(),
        },
    }
}

fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    text.chars().take(max).collect::<String>() + "…"
}

/// Asks the editor before anything consequential.
struct AcpApprover {
    peer: PeerHandle,
    session: String,
    cancel: Arc<AtomicBool>,
}

#[async_trait::async_trait]
impl Approver for AcpApprover {
    async fn approve(&self, tool: &str, input: &Value) -> bool {
        let peer = self.peer.clone();
        let session = self.session.clone();
        let cancel = Arc::clone(&self.cancel);
        let title = tool_title(tool, input);
        let call_id = format!("{tool}-{}", short_hash(input));

        // The request blocks on a human, so it leaves the async worker.
        tokio::task::spawn_blocking(move || {
            let reply = peer.request_with(
                "session/request_permission",
                Some(json!({
                    "sessionId": session,
                    "toolCall": {"toolCallId": call_id, "title": title},
                    "options": [
                        {"optionId": "allow", "name": "Allow", "kind": "allow_once"},
                        {"optionId": "reject", "name": "Reject", "kind": "reject_once"},
                    ],
                })),
                // No deadline: a person reading a diff is not a timeout. Still
                // interruptible — `abort` releases it if the turn is cancelled
                // or the editor goes away.
                None,
                Some(&cancel),
            );

            // Anything other than an explicit allow is a refusal. A malformed
            // answer, a closed pipe or a cancelled prompt must not read as
            // consent for a write.
            reply.is_ok_and(|r| {
                r.pointer("/outcome/outcome").and_then(Value::as_str) == Some("selected")
                    && r.pointer("/outcome/optionId").and_then(Value::as_str) == Some("allow")
            })
        })
        .await
        .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ariadne::Ariadne;
    use crate::engine::mock::{text_response, MockEngine};
    use crate::oracle::Oracle;
    use crate::scribe::SymbolIndex;
    use crate::session::Session;
    use crate::themis::Themis;
    use crate::tools::{ToolCtx, ToolRegistry};
    use std::io::{BufRead, BufReader, Write};
    use std::net::{Shutdown, TcpListener, TcpStream};
    use tempfile::TempDir;

    fn workspace() -> (TempDir, PathBuf) {
        let dir = TempDir::new().expect("tempdir");
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

    /// An editor on the other end of a real socket.
    struct Editor {
        peer: crate::jsonrpc::Peer,
        handle: PeerHandle,
        tx: TcpStream,
        rx: BufReader<TcpStream>,
        next_id: u64,
    }

    impl Editor {
        fn connect(scripted: Vec<crate::engine::Response>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
            let addr = listener.local_addr().expect("addr");
            let client = TcpStream::connect(addr).expect("connect");
            let (server, _) = listener.accept().expect("accept");

            let scripted = Arc::new(Mutex::new(scripted));
            let build: Arc<BuildAgent> = Arc::new(move |root: &Path, cancel, approver| {
                let queued = std::mem::take(&mut *scripted.lock().expect("scripted"));
                let mut t = Talos::new(
                    Box::new(MockEngine::new(queued)),
                    ToolRegistry::standard(),
                    ToolCtx::new(root),
                    Oracle::new(root).without_baseline(),
                    SymbolIndex::build(root)?,
                    Themis::from_text("Be correct."),
                    Ariadne::new(2, 1),
                    Session::new(root, "mock"),
                    1024,
                    false,
                );
                t.cancel = cancel;
                t.approver = Some(approver);
                Ok(t)
            });

            let rx = Box::new(BufReader::new(server.try_clone().expect("clone")));
            let mut peer =
                crate::jsonrpc::Peer::new(rx, Box::new(server)).with_fast_path(is_fast_path);
            // Captured here, on a runtime thread. The worker that later calls
            // `block_on` has no runtime of its own to find.
            let agent = Arc::new(Agent::new(
                peer.handle(),
                build,
                tokio::runtime::Handle::current(),
            ));
            peer.start(move |method: &str, params: Option<Value>, _| agent.handle(method, params));

            Editor {
                handle: peer.handle(),
                peer,
                rx: BufReader::new(client.try_clone().expect("clone")),
                tx: client,
                next_id: 0,
            }
        }

        fn call(&mut self, method: &str, params: Value) -> Value {
            self.next_id += 1;
            let id = self.next_id;
            let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
            writeln!(self.tx, "{msg}").expect("write");
            self.tx.flush().expect("flush");

            // Skip the narration until the reply to `id` arrives.
            loop {
                let msg = self.read();
                if msg.get("id").and_then(Value::as_u64) == Some(id) {
                    return msg;
                }
            }
        }

        fn read(&mut self) -> Value {
            let mut line = String::new();
            self.rx.read_line(&mut line).expect("read");
            serde_json::from_str(&line).expect("json")
        }

        /// Run a turn, returning everything narrated along the way and the
        /// reply that ended it.
        ///
        /// Bounded by the reply rather than by a message count: reading a fixed
        /// number blocks forever the moment the agent sends one fewer, which
        /// turns any failure in here into a hung suite instead of a failed test.
        fn turn(&mut self, session: &str, text: &str) -> (Vec<Value>, Value) {
            self.collect(
                "session/prompt",
                json!({"sessionId": session, "prompt": [{"type": "text", "text": text}]}),
            )
        }

        fn collect(&mut self, method: &str, params: Value) -> (Vec<Value>, Value) {
            self.next_id += 1;
            let id = self.next_id;
            let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
            writeln!(self.tx, "{msg}").expect("write");
            self.tx.flush().expect("flush");

            let mut updates = Vec::new();
            loop {
                let msg = self.read();
                if msg.get("id").and_then(Value::as_u64) == Some(id) {
                    return (updates, msg);
                }
                if msg.get("method").and_then(Value::as_str) == Some("session/update") {
                    updates.push(msg["params"]["update"].clone());
                }
            }
        }

        fn open(&mut self, root: &Path) -> String {
            self.call("initialize", json!({"protocolVersion": 1}));
            let reply = self.call("session/new", json!({"cwd": root, "mcpServers": []}));
            reply["result"]["sessionId"]
                .as_str()
                .expect("sessionId")
                .to_string()
        }
    }

    impl Drop for Editor {
        fn drop(&mut self) {
            self.handle.close();
            let _ = self.tx.shutdown(Shutdown::Both);
            self.peer.wait();
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn initialize_states_a_version_and_claims_no_capability_it_lacks() {
        let (_dir, _root) = workspace();
        let mut ed = Editor::connect(vec![]);
        let reply = ed.call("initialize", json!({"protocolVersion": 1}));

        assert_eq!(reply["result"]["protocolVersion"], json!(PROTOCOL_VERSION));
        assert_eq!(reply["result"]["agentInfo"]["name"], json!("knossos"));
        // Claiming `fs` would advertise a callback this agent never makes.
        assert!(reply["result"]["agentCapabilities"].get("fs").is_none());
        assert_eq!(
            reply["result"]["agentCapabilities"]["loadSession"],
            json!(true)
        );
        assert_eq!(
            reply["result"]["agentCapabilities"]["unstable_forkSession"],
            json!(true)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_session_needs_an_absolute_cwd() {
        let mut ed = Editor::connect(vec![]);
        ed.call("initialize", json!({"protocolVersion": 1}));

        let reply = ed.call("session/new", json!({"cwd": "relative/path"}));
        assert_eq!(reply["error"]["code"], json!(INVALID_PARAMS));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn context_limits_are_clamped_and_change_only_between_turns() {
        let (_dir, root) = workspace();
        let mut editor = Editor::connect(vec![]);
        editor.call("initialize", json!({"protocolVersion": 1}));
        let opened = editor.call(
            "session/new",
            json!({"cwd": root, "contextWindow": 8_000, "compactAt": 7_000}),
        );
        let session = opened["result"]["sessionId"].as_str().unwrap().to_string();
        assert_eq!(opened["result"]["context"]["assignedContext"], json!(8_000));
        assert_eq!(opened["result"]["context"]["compactAt"], json!(6_464));

        let changed = editor.call(
            "session/set_context",
            json!({"sessionId": session, "contextWindow": 4_000, "compactAt": 99_000}),
        );
        assert_eq!(
            changed["result"]["context"]["assignedContext"],
            json!(4_000)
        );
        assert_eq!(changed["result"]["context"]["compactAt"], json!(2_488));

        let bad = editor.call(
            "session/set_context",
            json!({"sessionId": session, "contextWindow": 0}),
        );
        assert_eq!(bad["error"]["code"], json!(INVALID_PARAMS));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_prompt_runs_a_turn_and_reports_why_it_stopped() {
        let (_dir, root) = workspace();
        let mut ed = Editor::connect(vec![text_response("had a look"); 2]);
        let session = ed.open(&root);

        let reply = ed.call(
            "session/prompt",
            json!({"sessionId": session, "prompt": [{"type": "text", "text": "look around"}]}),
        );
        assert!(reply["result"]["stopReason"].is_string(), "{reply}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_editor_is_told_what_the_model_said() {
        let (_dir, root) = workspace();
        let mut ed = Editor::connect(vec![text_response("here is my answer"); 2]);
        let session = ed.open(&root);

        let (updates, _) = ed.turn(&session, "look around");

        assert!(
            updates
                .iter()
                .any(|u| u["sessionUpdate"] == "agent_message_chunk"
                    && u["content"]["text"] == "here is my answer"),
            "no agent message in {updates:#?}",
        );
        assert!(
            updates.iter().any(|u| u["sessionUpdate"] == "plan"),
            "no plan in {updates:#?}",
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn load_replays_history_without_doubling_it() {
        let (_dir, root) = workspace();
        let mut ed = Editor::connect(vec![text_response("here is my answer"); 2]);
        let session = ed.open(&root);
        let (first, _) = ed.turn(&session, "look around");
        assert!(!first.is_empty(), "a turn must produce updates to replay");

        let (replayed, reply) = ed.collect("session/load", json!({"sessionId": session}));
        assert!(reply.get("error").is_none(), "{reply}");
        assert_eq!(replayed, first, "load must replay the same updates");

        let (again, _) = ed.collect("session/load", json!({"sessionId": session}));
        assert_eq!(again, first, "a second load must not double the history");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_unknown_session_is_an_error_rather_than_a_panic() {
        let mut ed = Editor::connect(vec![]);
        ed.call("initialize", json!({"protocolVersion": 1}));

        let reply = ed.call("session/prompt", json!({"sessionId": "nope", "prompt": []}));
        assert_eq!(reply["error"]["code"], json!(INVALID_PARAMS));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_unsupported_method_is_answered_not_ignored() {
        let mut ed = Editor::connect(vec![]);
        // Silence would block the editor for as long as it is willing to wait.
        let reply = ed.call("session/telepathy", json!({}));
        assert_eq!(reply["error"]["code"], json!(METHOD_NOT_FOUND));
    }

    #[test]
    fn cancel_is_on_the_fast_path() {
        // The reason it is: it arrives during the prompt it interrupts, so
        // queueing it behind that prompt delivers it after the turn it was
        // meant to stop.
        assert!(is_fast_path("session/cancel"));
        assert!(is_fast_path("session/interject"));
        assert!(!is_fast_path("session/prompt"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn list_and_close_sessions() {
        let (_dir, root) = workspace();
        let mut ed = Editor::connect(vec![]);
        let session = ed.open(&root);

        let listed = ed.call("session/list", json!({}));
        let ids: Vec<_> = listed["result"]["sessions"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|s| s["sessionId"].as_str())
            .collect();
        assert!(ids.contains(&session.as_str()), "{listed}");

        let closed = ed.call("session/close", json!({"sessionId": session}));
        assert!(closed.get("error").is_none(), "{closed}");
        let listed = ed.call("session/list", json!({}));
        assert_eq!(listed["result"]["sessions"].as_array().unwrap().len(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn set_mode_preview_stages_writes() {
        let (_dir, root) = workspace();
        let mut ed = Editor::connect(vec![]);
        let session = ed.open(&root);
        let reply = ed.call(
            "session/set_mode",
            json!({"sessionId": session, "mode": "preview"}),
        );
        assert_eq!(reply["result"]["mode"], json!("preview"), "{reply}");
        let bad = ed.call(
            "session/set_mode",
            json!({"sessionId": session, "mode": "explode"}),
        );
        assert_eq!(bad["error"]["code"], json!(INVALID_PARAMS));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_prompt_of_no_text_is_refused() {
        let (_dir, root) = workspace();
        let mut ed = Editor::connect(vec![]);
        let session = ed.open(&root);

        let reply = ed.call(
            "session/prompt",
            json!({"sessionId": session, "prompt": []}),
        );
        assert_eq!(reply["error"]["code"], json!(INVALID_PARAMS));
    }

    // ------------------------------------------------------------ mapping

    #[test]
    fn a_halt_becomes_the_protocols_own_word_for_it() {
        assert_eq!(stop_reason(Halt::Done), "end_turn");
        assert_eq!(stop_reason(Halt::Cancelled), "cancelled");
        assert_eq!(stop_reason(Halt::BudgetExhausted), "max_turn_requests");
        // No ACP word for it, and `refusal` would misreport intent.
        assert_eq!(stop_reason(Halt::Stuck), "end_turn");
    }

    #[test]
    fn tool_kinds_match_what_an_editor_draws() {
        assert_eq!(tool_kind("read_file"), "read");
        assert_eq!(tool_kind("write_file"), "edit");
        assert_eq!(tool_kind("run"), "execute");
        assert_eq!(tool_kind("delegate"), "think");
        assert_eq!(tool_kind("github.create_issue"), "other");
    }

    #[test]
    fn two_calls_to_one_tool_are_two_rows() {
        let a = short_hash(&json!({"path": "a.rs"}));
        let b = short_hash(&json!({"path": "b.rs"}));
        assert_ne!(a, b, "identical ids would collapse two calls into one row");
        assert_eq!(
            a,
            short_hash(&json!({"path": "a.rs"})),
            "and must be stable"
        );
    }

    #[test]
    fn a_prompt_keeps_text_and_names_what_it_cannot_read() {
        let params = json!({"prompt": [
            {"type": "text", "text": "fix this"},
            {"type": "image", "data": "..."},
        ]});
        let text = prompt_text(&params);
        assert!(text.contains("fix this"));
        assert!(
            text.contains("[image content]"),
            "silently dropping it looks like nothing \
                                                    was attached: {text}"
        );
    }

    #[test]
    fn a_delegation_is_titled_by_its_task() {
        let title = tool_title("delegate", &json!({"task": "find every caller of Router"}));
        assert!(title.contains("find every caller"), "{title}");
    }
}
