//! Long-lived NDJSON server, for editor front ends.
//!
//! One JSON object per line in each direction: commands on stdin, events on
//! stdout. The process stays alive across turns, which is what lets a chat
//! panel hold a conversation instead of re-invoking a one-shot command and
//! losing all context.
//!
//! **stdout is the protocol.** Every human-readable byte goes to stderr; a
//! single stray `println!` on this path corrupts the stream. Progress events
//! reuse [`crate::session::TraceEvent`] verbatim, because the trace log
//! already is an event stream — the front end reads exactly the lines the log
//! file receives.

use std::io::Write;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::metis;
use crate::talos::{Outcome, Talos};

#[derive(Debug, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Command {
    /// Start a fresh task, discarding the previous conversation.
    Task {
        text: String,
    },
    /// Continue the existing conversation.
    Resume {
        text: String,
    },
    /// Plan without executing.
    Plan {
        text: String,
    },
    /// Current staged changes, with full proposed content.
    Diffs,
    /// Write staged changes to disk.
    Apply,
    /// Write only the selected hunks, leaving the rest staged.
    ApplyHunks {
        selection: Vec<HunkSelection>,
    },
    /// Throw staged changes away.
    Discard,
    /// Run the verification ladder now.
    Verify,
    /// Symbol counts, or an exact lookup.
    Index {
        #[serde(default)]
        name: Option<String>,
    },
    /// Clear the conversation, keep the workspace.
    Reset,
    /// Report context allocation and durable mission status.
    State,
    /// Change this agent's context allocation between turns. Values are
    /// clamped to the engine/server ceiling.
    SetContext {
        #[serde(default)]
        context_window: Option<u32>,
        #[serde(default)]
        compact_at: Option<u32>,
    },
    /// Put the workspace back as it was before the last turn began.
    ///
    /// The inverse of `Reset`: that keeps the files and drops the conversation,
    /// this keeps the conversation and drops the files. Dispatched normally —
    /// there is no turn running to undo while a turn is running.
    Undo,
    /// What this front end can do. Send before anything else.
    ///
    /// **Permission gating is opt-in, and it has to be.** A front end that does
    /// not understand `permission_request` will drop it — the VS Code panel in
    /// this repository dispatches events by name and silently ignores unknown
    /// ones — and the agent would then wait forever for a reply nobody is going
    /// to send. Gating by default would turn every existing front end into a
    /// hang, which is a worse failure than the one the gate prevents.
    ///
    /// So a front end declares that it can answer, and only then is the gate
    /// installed. Not declaring leaves the run unattended, which is exactly what
    /// `approver: None` means everywhere else. Routed, never dispatched.
    Capabilities {
        #[serde(default)]
        permissions: bool,
    },
    /// Say something to a task that is already running. **Routed, never
    /// dispatched.**
    ///
    /// Same reason as a permission reply: the dispatch loop is inside
    /// `talos.run` for the whole turn, so anything that must reach a running
    /// task cannot be a command it handles. Unlike a permission reply this does
    /// not block the agent — it is queued and picked up at the next step
    /// boundary. See [`interject`](crate::interject).
    ///
    /// Sending this with no task running is harmless: it waits, and the next
    /// task begins by reading it.
    Interject {
        text: String,
    },
    /// Answer to a `permission_request`. **Routed, never dispatched.**
    ///
    /// This is the one command that must be handled while another command is
    /// still running, so it is intercepted by the reader task and completes the
    /// waiting request directly. If it went through the dispatch loop it could
    /// never arrive: that loop is inside `talos.run`, waiting for this.
    Permission {
        id: u64,
        allow: bool,
    },
    Shutdown,
}

#[derive(Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    Ready {
        workspace: String,
        engine: String,
        constitution: String,
        symbols: usize,
        files: usize,
        dry_run: bool,
        max_steps: usize,
    },
    Plan {
        steps: Vec<String>,
    },
    Outcome {
        halt: String,
        succeeded: bool,
        steps_used: usize,
        summary: String,
        changed: Vec<String>,
        dry_run: bool,
    },
    Diffs {
        files: Vec<DiffPayload>,
    },
    Applied {
        files: Vec<String>,
    },
    Discarded,
    Verdict {
        passed: bool,
        summary: String,
        dry_run: bool,
        tiers: Vec<TierPayload>,
    },
    Index {
        symbols: usize,
        files: usize,
        hits: Vec<String>,
    },
    Reset,
    State {
        engine_context_tokens: Option<u32>,
        assigned_context_tokens: u32,
        input_limit_tokens: u32,
        compact_at_tokens: u32,
        completion_reserve_tokens: u32,
        protocol_reserve_tokens: u32,
        estimated_conversation_tokens: usize,
        compaction_enabled: bool,
        mission_id: Option<String>,
        mission_phase: Option<String>,
        workspace_revision: Option<u64>,
        pending_actions: usize,
    },
    /// Files put back by an `undo`, workspace-relative.
    Undone {
        files: Vec<String>,
    },
    Error {
        message: String,
    },
    /// The agent wants to do something consequential and is waiting.
    ///
    /// Emitted *mid-command*, so it is not followed by `Idle` — the command has
    /// not finished, and telling the front end otherwise would have it re-enable
    /// input while a turn is still running. Answer with
    /// `{"cmd":"permission","id":<id>,"allow":true|false}`.
    PermissionRequest {
        id: u64,
        tool: String,
        input: serde_json::Value,
    },
    /// Every command ends with exactly one of these, so the front end always
    /// knows when it can re-enable input.
    Idle,
}

#[derive(Debug, Deserialize)]
pub struct HunkSelection {
    pub path: String,
    /// Hunk ids being accepted, as sent in the matching `diffs` event.
    pub hunks: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct DiffPayload {
    pub path: String,
    pub unified: String,
    pub added: usize,
    pub removed: usize,
    /// The full proposed file, so the editor can render a real side-by-side
    /// diff rather than parsing the unified text back apart.
    pub content: String,
    pub existed: bool,
    /// Individually acceptable pieces of this file's change.
    pub hunks: Vec<HunkPayload>,
}

#[derive(Debug, Serialize)]
pub struct HunkPayload {
    pub id: usize,
    pub header: String,
    pub body: String,
    pub added: usize,
    pub removed: usize,
}

#[derive(Debug, Serialize)]
pub struct TierPayload {
    pub tier: u8,
    pub label: String,
    pub passed: bool,
    pub detail: String,
}

/// Where events go. Cloneable, so the permission approver can hold one too.
///
/// Events used to be written to `std::io::stdout()` from a free function, which
/// made the protocol untestable: nothing could observe the stream without
/// capturing the process's real stdout. Sending them instead means a test reads
/// exactly what a front end would.
#[derive(Clone)]
pub struct Emitter(tokio::sync::mpsc::UnboundedSender<Event>);

impl Emitter {
    pub fn new(tx: tokio::sync::mpsc::UnboundedSender<Event>) -> Self {
        Emitter(tx)
    }

    fn send(&self, event: Event) {
        // A closed receiver means the front end is gone. The loop notices via
        // the command channel; dropping the event here is right, and panicking
        // on it would take down a session that is merely finishing.
        let _ = self.0.send(event);
    }
}

/// Drain events to a writer, one JSON object per line.
///
/// The `main`-side half of the split. Kept out of `run` so tests never touch a
/// real stream, and so **stdout stays protocol-only** — the invariant the module
/// docs open with.
pub async fn write_events<W: Write + Send + 'static>(
    mut events: tokio::sync::mpsc::UnboundedReceiver<Event>,
    mut out: W,
) {
    while let Some(event) = events.recv().await {
        match serde_json::to_string(&event) {
            Ok(line) => {
                let _ = writeln!(out, "{line}");
                let _ = out.flush();
            }
            Err(e) => eprintln!("could not serialize event: {e}"),
        }
    }
}

/// Requests waiting for the front end to answer, by id.
type Pending = std::sync::Arc<
    std::sync::Mutex<std::collections::HashMap<u64, tokio::sync::oneshot::Sender<bool>>>,
>;

/// Puts a consequential call to the front end and waits for the answer.
struct FrontEndApprover {
    events: Emitter,
    pending: Pending,
    next_id: std::sync::atomic::AtomicU64,
    /// Set when the front end declares it can answer. Until then every call is
    /// allowed, because asking something that cannot reply is just a hang.
    enabled: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

#[async_trait::async_trait]
impl crate::talos::Approver for FrontEndApprover {
    async fn approve(&self, tool: &str, input: &serde_json::Value) -> bool {
        if !self.enabled.load(std::sync::atomic::Ordering::Relaxed) {
            return true; // this front end cannot answer; asking would hang it
        }
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (tx, rx) = tokio::sync::oneshot::channel();
        // Registered *before* the event goes out, so an instant reply cannot
        // arrive before there is anywhere to put it.
        match self.pending.lock() {
            Ok(mut map) => {
                map.insert(id, tx);
            }
            // A poisoned lock means a router thread panicked. Nothing can answer
            // after that, so refusing is the only honest result.
            Err(_) => return false,
        }
        self.events.send(Event::PermissionRequest {
            id,
            tool: tool.to_string(),
            input: input.clone(),
        });
        // Untimed, like the ACP side: a user reading a diff is not a failure,
        // and a timeout that denies would silently reject work they meant to
        // approve. The realistic failure is the front end going away, which
        // drops the sender and resolves this as a refusal rather than a hang.
        rx.await.unwrap_or(false)
    }
}

/// Read lines, answer permission requests directly, forward everything else.
///
/// This is the whole reason the loop is split. A permission reply has to be
/// processed *while* a command is still running — the dispatch loop is inside
/// `talos.run`, waiting for exactly this — so it can never be a command the
/// dispatch loop handles. Routing it here also means the dispatch loop is never
/// reentrant: it sees one command at a time and nothing else.
async fn route(
    mut lines: tokio::sync::mpsc::UnboundedReceiver<String>,
    commands: tokio::sync::mpsc::UnboundedSender<Result<Command, String>>,
    pending: Pending,
    gating: std::sync::Arc<std::sync::atomic::AtomicBool>,
    interjections: crate::interject::Interjections,
    events: Emitter,
) {
    while let Some(line) = lines.recv().await {
        let trimmed = line.trim_start_matches('\u{feff}').trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<Command>(trimmed) {
            Ok(Command::Capabilities { permissions }) => {
                // Routed rather than dispatched for the same reason as a
                // permission reply: it changes how the *approver* behaves, and
                // the approver lives outside the dispatch loop.
                gating.store(permissions, std::sync::atomic::Ordering::Relaxed);
            }
            Ok(Command::Interject { text }) => {
                // Refusal is reported rather than swallowed: the person typed
                // something and is entitled to know the agent will not see it.
                if !interjections.push(text) {
                    events.send(Event::Error {
                        message: "interjection not accepted: empty, or too many are already queued"
                            .to_string(),
                    });
                }
            }
            Ok(Command::Permission { id, allow }) => {
                let waiting = pending.lock().ok().and_then(|mut m| m.remove(&id));
                match waiting {
                    Some(tx) => {
                        let _ = tx.send(allow);
                    }
                    // A reply to a request that already resolved, or an id that
                    // never existed. Neither is worth ending a session over.
                    None => eprintln!("permission reply for unknown request {id}"),
                }
            }
            Ok(command) => {
                if commands.send(Ok(command)).is_err() {
                    return; // dispatch loop is gone
                }
            }
            Err(e) => {
                if commands.send(Err(format!("bad command: {e}"))).is_err() {
                    return;
                }
            }
        }
    }
    deny_outstanding(&pending);
}

/// Refuse everything still waiting. Called when the front end goes away.
///
/// Without this the loop deadlocks, and it is worth being precise about why,
/// because dropping the router's own handle is *not* enough: `pending` is an
/// `Arc` and the approver holds a clone, so the `oneshot::Sender` inside the map
/// stays alive after the router returns. `rx.await` then never resolves,
/// `talos.run` never returns, dispatch never returns, and the loop never reaches
/// the `recv()` that would have noticed the disconnect. The server hangs holding
/// a half-finished turn.
///
/// Denying rather than approving is the only defensible resolution: nobody
/// answered, and treating silence as consent is how an unattended write happens
/// in the one code path built to prevent it.
fn deny_outstanding(pending: &Pending) {
    let waiting = match pending.lock() {
        Ok(mut map) => std::mem::take(&mut *map),
        Err(_) => return, // poisoned; the approver's own lock will refuse too
    };
    for (_, tx) in waiting {
        let _ = tx.send(false);
    }
}

/// Drive the protocol over a line source and an event sink.
///
/// Takes channels rather than the real streams so the loop can be driven from a
/// test. `main` supplies a thread reading stdin and a task writing stdout; the
/// tests supply channels they control, which is what makes the permission
/// handshake something that can be *proved* not to deadlock rather than hoped
/// about.
pub async fn run(
    mut talos: Talos,
    max_tokens: u32,
    lines: tokio::sync::mpsc::UnboundedReceiver<String>,
    events: Emitter,
) -> Result<()> {
    let pending: Pending = Default::default();
    let gating = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    talos.approver = Some(std::sync::Arc::new(FrontEndApprover {
        events: events.clone(),
        pending: pending.clone(),
        next_id: std::sync::atomic::AtomicU64::new(1),
        enabled: gating.clone(),
    }));

    events.send(Event::Ready {
        workspace: talos.oracle.root().display().to_string(),
        engine: talos.session.engine_name.clone(),
        constitution: talos.themis.source().to_string(),
        symbols: talos.scribe.symbol_count(),
        files: talos.scribe.file_count(),
        dry_run: talos.ctx.is_dry_run(),
        max_steps: talos.ariadne.max_steps,
    });
    events.send(Event::Idle);

    let (command_tx, mut command_rx) = tokio::sync::mpsc::unbounded_channel();
    let router = tokio::spawn(route(
        lines,
        command_tx,
        pending,
        gating,
        talos.interjections(),
        events.clone(),
    ));

    while let Some(incoming) = command_rx.recv().await {
        let command = match incoming {
            Ok(c) => c,
            Err(message) => {
                events.send(Event::Error { message });
                events.send(Event::Idle);
                continue;
            }
        };

        if matches!(command, Command::Shutdown) {
            break;
        }

        // A failing command must not kill the server: the conversation and any
        // staged work would go with it.
        if let Err(e) = dispatch(&mut talos, command, max_tokens, &events).await {
            events.send(Event::Error {
                message: format!("{e:#}"),
            });
        }
        events.send(Event::Idle);
    }

    router.abort();
    Ok(())
}

async fn dispatch(
    talos: &mut Talos,
    command: Command,
    max_tokens: u32,
    events: &Emitter,
) -> Result<()> {
    match command {
        Command::Task { text } => {
            talos.capture_environment()?;
            let plan = metis::plan(
                talos.engine.as_ref(),
                &talos.themis,
                &talos.scribe,
                &text,
                max_tokens,
            )
            .await?;
            events.send(Event::Plan {
                steps: plan.steps.clone(),
            });
            let outcome = talos.run(&text, &plan).await?;
            finish_turn(talos, &outcome, events);
        }
        Command::Resume { text } => {
            let outcome = talos.resume(&text).await?;
            finish_turn(talos, &outcome, events);
        }
        Command::Plan { text } => {
            let plan = metis::plan(
                talos.engine.as_ref(),
                &talos.themis,
                &talos.scribe,
                &text,
                max_tokens,
            )
            .await?;
            events.send(Event::Plan { steps: plan.steps });
        }
        Command::Diffs => emit_diffs(talos, events),
        Command::Apply => {
            let written = talos.apply()?;
            events.send(Event::Applied {
                files: written.iter().map(|p| rel(talos, p)).collect(),
            });
        }
        Command::ApplyHunks { selection } => {
            let pairs: Vec<(String, Vec<usize>)> =
                selection.into_iter().map(|s| (s.path, s.hunks)).collect();
            let written = talos.apply_hunks(&pairs)?;
            events.send(Event::Applied {
                files: written.iter().map(|p| rel(talos, p)).collect(),
            });
            // Anything partially accepted is still staged; re-send so the
            // front end shows what remains rather than a stale list.
            emit_diffs(talos, events);
        }
        Command::Discard => {
            talos.discard();
            events.send(Event::Discarded);
        }
        Command::Verify => {
            let files: Vec<std::path::PathBuf> = talos.changed.iter().cloned().collect();
            let verdict = if talos.ctx.is_dry_run() {
                talos
                    .oracle
                    .verify_staged(talos.scribe.adapter(), &talos.ctx.staged_contents())
            } else {
                talos.oracle.verify(talos.scribe.adapter(), &files).await?
            };
            events.send(Event::Verdict {
                passed: verdict.passed,
                summary: verdict.summary(),
                dry_run: verdict.dry_run,
                tiers: verdict
                    .tiers
                    .iter()
                    .map(|t| TierPayload {
                        tier: t.tier,
                        label: t.label.clone(),
                        passed: t.passed,
                        detail: t.detail.clone(),
                    })
                    .collect(),
            });
        }
        Command::Index { name } => {
            let hits = match &name {
                Some(n) => talos
                    .scribe
                    .lookup(n)
                    .iter()
                    .map(|s| format!("{}:{}: {}", s.file.display(), s.line, s.signature))
                    .collect(),
                None => Vec::new(),
            };
            events.send(Event::Index {
                symbols: talos.scribe.symbol_count(),
                files: talos.scribe.file_count(),
                hits,
            });
        }
        Command::Reset => {
            talos.messages.clear();
            talos.changed.clear();
            events.send(Event::Reset);
        }
        Command::State => emit_state(talos, events),
        Command::SetContext {
            context_window,
            compact_at,
        } => {
            talos.set_context_limits(context_window, compact_at);
            emit_state(talos, events);
        }
        Command::Undo => match talos.undo_turn() {
            Ok(restored) => {
                let files = restored.iter().map(|p| talos.ctx.display(p)).collect();
                events.send(Event::Undone { files });
            }
            // "No turn has run yet" is a legitimate answer, not a session-ending
            // fault, so it reports like any other refused command.
            Err(e) => events.send(Event::Error {
                message: format!("{e:#}"),
            }),
        },
        // Both are intercepted before they get here: `Shutdown` by the loop,
        // `Permission` by the router. Reaching either would mean a reply was
        // queued behind the very command that is waiting for it — the deadlock
        // this whole split exists to prevent — so it is worth a loud failure
        // rather than a silent no-op.
        Command::Shutdown => unreachable!("handled by the caller"),
        Command::Permission { .. } => {
            unreachable!("permission replies are routed, never dispatched")
        }
        Command::Capabilities { .. } => {
            unreachable!("capabilities are routed, never dispatched")
        }
        Command::Interject { .. } => {
            unreachable!("interjections are routed, never dispatched")
        }
    }
    Ok(())
}

fn finish_turn(talos: &Talos, outcome: &Outcome, events: &Emitter) {
    events.send(Event::Outcome {
        halt: outcome.halt.label().to_string(),
        succeeded: outcome.succeeded(),
        steps_used: outcome.steps_used,
        summary: outcome.summary.clone(),
        changed: outcome.changed.iter().map(|p| rel(talos, p)).collect(),
        dry_run: outcome.dry_run,
    });
    if outcome.dry_run {
        emit_diffs(talos, events);
    }
}

fn emit_state(talos: &Talos, events: &Emitter) {
    let context = talos.context_budget();
    let mission = talos.mission_state();
    events.send(Event::State {
        engine_context_tokens: context.engine_tokens,
        assigned_context_tokens: context.assigned_tokens,
        input_limit_tokens: context.input_limit_tokens,
        compact_at_tokens: if talos.context_policy.compaction_enabled {
            context.compact_at_tokens
        } else {
            0
        },
        completion_reserve_tokens: context.completion_reserve,
        protocol_reserve_tokens: context.protocol_reserve,
        estimated_conversation_tokens: crate::lethe::estimate_tokens(&talos.messages),
        compaction_enabled: talos.context_policy.compaction_enabled,
        mission_id: talos.mission_id().map(str::to_string),
        mission_phase: mission.map(|state| state.focus.phase.label().to_string()),
        workspace_revision: mission.map(|state| state.workspace.revision),
        pending_actions: mission.map_or(0, |state| state.pending_actions.len()),
    });
}

fn emit_diffs(talos: &Talos, events: &Emitter) {
    let staged: std::collections::BTreeMap<_, _> =
        talos.ctx.staged_contents().into_iter().collect();

    let files = talos
        .diffs()
        .into_iter()
        .map(|d| {
            // `d.path` is workspace-relative; staging is keyed by absolute path.
            let absolute = talos.oracle.root().join(&d.path);
            DiffPayload {
                path: d.path.display().to_string(),
                unified: d.unified,
                added: d.added,
                removed: d.removed,
                content: staged.get(&absolute).cloned().unwrap_or_default(),
                existed: d.existed,
                hunks: d
                    .hunks
                    .iter()
                    .map(|h| HunkPayload {
                        id: h.id,
                        header: h.header.clone(),
                        body: h.body.clone(),
                        added: h.added,
                        removed: h.removed,
                    })
                    .collect(),
            }
        })
        .collect();

    events.send(Event::Diffs { files });
}

fn rel(talos: &Talos, path: &std::path::Path) -> String {
    talos.ctx.display(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_command_shape() {
        let cases = [
            r#"{"cmd":"task","text":"add a flag"}"#,
            r#"{"cmd":"resume","text":"now undo it"}"#,
            r#"{"cmd":"plan","text":"something"}"#,
            r#"{"cmd":"diffs"}"#,
            r#"{"cmd":"apply"}"#,
            r#"{"cmd":"apply_hunks","selection":[{"path":"src/lib.rs","hunks":[0,2]}]}"#,
            r#"{"cmd":"discard"}"#,
            r#"{"cmd":"verify"}"#,
            r#"{"cmd":"index"}"#,
            r#"{"cmd":"index","name":"Adder"}"#,
            r#"{"cmd":"reset"}"#,
            r#"{"cmd":"state"}"#,
            r#"{"cmd":"set_context","context_window":8000,"compact_at":6000}"#,
            r#"{"cmd":"undo"}"#,
            r#"{"cmd":"interject","text":"use the existing helper"}"#,
            r#"{"cmd":"shutdown"}"#,
        ];
        for c in cases {
            serde_json::from_str::<Command>(c).unwrap_or_else(|e| panic!("{c} -> {e}"));
        }
    }

    #[test]
    fn an_unknown_command_is_a_parse_error_not_a_panic() {
        assert!(serde_json::from_str::<Command>(r#"{"cmd":"launch_missiles"}"#).is_err());
        assert!(serde_json::from_str::<Command>("not json").is_err());
    }

    #[test]
    fn events_serialize_with_a_discriminating_tag() {
        let e = Event::Applied {
            files: vec!["src/lib.rs".into()],
        };
        let v: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&e).unwrap()).unwrap();
        assert_eq!(v["event"], "applied");
        assert_eq!(v["files"][0], "src/lib.rs");

        let idle: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&Event::Idle).unwrap()).unwrap();
        assert_eq!(idle["event"], "idle");
    }

    #[test]
    fn diff_payload_carries_full_content_for_the_editor() {
        let p = DiffPayload {
            path: "src/lib.rs".into(),
            unified: "@@".into(),
            added: 1,
            removed: 0,
            content: "pub fn a() {}\n".into(),
            existed: true,
            hunks: vec![HunkPayload {
                id: 0,
                header: "@@ -1,1 +1,1 @@".into(),
                body: "+pub fn a() {}\n".into(),
                added: 1,
                removed: 0,
            }],
        };
        let v: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!(v["content"], "pub fn a() {}\n");
        assert_eq!(v["existed"], true);
    }
}
