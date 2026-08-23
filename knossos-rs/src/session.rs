//! Conversation state and the trajectory trace.
//!
//! The trace is not logging. Every plan, step, tool call, Oracle verdict and
//! halt decision is appended as one JSON object per line, which makes a
//! completed run a structured record of what the agent did and whether it
//! worked.
//!
//! Echo — distilling deep trajectories into shallow ones — is out of scope for
//! v1, but its training data is exactly this file. Capturing it now costs
//! almost nothing; reconstructing it later from logs costs a great deal. This
//! is the bridge from the harness back to the model architecture.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use serde::Serialize;

use crate::engine::Message;

pub const TRACE_SCHEMA_VERSION: &str = "daedalus-trace/v2";

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum TraceEvent {
    /// Reproducibility facts for an evaluation trajectory. Kept separate from
    /// `TaskStarted` so ordinary interactive traces stay compact.
    ExperimentMetadata {
        arm: String,
        case_id: String,
        expected_action: String,
        provider: String,
        model: String,
        harness_commit: String,
        suite_digest: String,
        max_steps: usize,
        max_output_tokens: u32,
        started_at: String,
    },
    TaskStarted {
        task: String,
        engine: String,
        workspace: String,
        max_steps: usize,
        target_steps: usize,
    },
    PlanProduced {
        steps: Vec<String>,
    },
    StepStarted {
        index: usize,
        description: String,
    },
    /// What the model actually said, per step.
    ///
    /// The trace recorded every tool call but never the prose around them, so a
    /// reader could see what the agent *did* and not what it claimed to be
    /// doing. [`Exchange`](Self::Exchange) carries it, but only when collecting,
    /// and at the cost of the whole conversation per step. This is the cheap
    /// half, and it is what a front end renders as the agent's reply.
    AgentMessage {
        step: usize,
        text: String,
    },
    /// Reasoning the model produced while generating. Editors render it
    /// collapsed; it is not a tool call and must not be parsed as one.
    Thought {
        step: usize,
        text: String,
    },
    /// Whether repository context was injected for this turn, and the signals
    /// that decided it.
    ///
    /// Logged even when nothing is injected. A skip is the harder case to
    /// debug — the model simply answers without the code it needed, and
    /// nothing in the trace would otherwise say why.
    ContextConsidered {
        query: String,
        injected: bool,
        confidence: f64,
        reasons: Vec<String>,
        /// Slices actually injected, and their total size in characters.
        slices: usize,
        chars: usize,
    },
    /// Exactly what was sent to the engine, and exactly what came back.
    ///
    /// This is the difference between a trace you can *read* and a trace you
    /// can *train on*. Every other event here records what the harness did;
    /// only this one records what the model was shown and what it answered,
    /// which is the sole pair an SFT example can be built from. The verdict
    /// and halt events already in the stream are the label — so a collected
    /// trace carries both the example and whether it worked.
    ///
    /// Off unless the session is collecting, because a `Request` holds the
    /// whole conversation and recording one per step makes the trace roughly
    /// quadratic in run length. A run that exists to be audited does not need
    /// it; a run that exists to produce training data is the only kind that
    /// does, and it opts in.
    ///
    /// Written to the trace file only, never streamed — see `streamable`.
    Exchange {
        step: usize,
        request: crate::engine::Request,
        response: crate::engine::Response,
    },
    /// Linear-size training record. The first event (and any reset after
    /// compaction) carries system/tools plus the complete current messages;
    /// later events carry only messages appended since the preceding request.
    ExchangeDelta {
        step: usize,
        reset: bool,
        system: Option<String>,
        tools: Option<Vec<crate::engine::ToolDef>>,
        messages_start: usize,
        messages: Vec<Message>,
        response: crate::engine::Response,
    },
    ToolCall {
        step: usize,
        tool: String,
        input: serde_json::Value,
        output: String,
        is_error: bool,
        changed: Vec<String>,
    },
    OracleVerdict {
        step: usize,
        tier: String,
        passed: bool,
        summary: String,
    },
    /// The conversation was shrunk to fit its budget.
    ///
    /// Traced because it is otherwise invisible: the run continues normally and
    /// the only evidence that context was given up is that the model stops
    /// referring to something it was told earlier.
    ContextCompacted {
        step: usize,
        tokens: usize,
    },
    /// The user said something to a run that was already going.
    ///
    /// Traced for the same reason as `ContextCompacted`: without it the agent
    /// visibly changes course at some step and the trace gives no reason, which
    /// is exactly the kind of thing a run is analysed to find out.
    Interjected {
        step: usize,
        notes: Vec<String>,
    },
    Halt {
        step: usize,
        reason: String,
        detail: String,
    },
    /// A failed or futile call, persisted so the next prompt (and the next
    /// process) can refuse to retry it. Distinct from [`Self::ToolCall`]
    /// because that event is per-dispatch and this one is what recall reads.
    Hypothesis {
        step: usize,
        signature: String,
        detail: String,
    },
    /// First `Stuck` of a drive: the loop continues rather than finishing.
    Redirected {
        step: usize,
        attempt: String,
        forbidden: Vec<String>,
    },
    TaskFinished {
        steps_used: usize,
        outcome: String,
    },
    /// Orthogonal labels for corpus selection. A provider failure, an
    /// unverifiable grader, and an incorrect patch must never collapse into
    /// the same `failed` bucket.
    EvaluationFinished {
        case_id: String,
        grader_pass: bool,
        verifier_pass: bool,
        halt_reason: String,
        provider_status: String,
        infrastructure_status: String,
        tamper: bool,
    },
}

pub struct Session {
    pub root: PathBuf,
    pub engine_name: String,
    /// The running conversation handed to the engine each turn.
    pub messages: Vec<Message>,
    trace_path: Option<PathBuf>,
    /// Also write each event to stdout, for `daedalus serve`.
    ///
    /// The trace format already *is* an event stream, so a front end that
    /// wants live progress needs no second mechanism — it reads the same
    /// lines the log file gets.
    stream_stdout: bool,
    /// Hand each event to a caller-supplied sink as well.
    ///
    /// Exists because [`stream_stdout`](Self::streaming) is not universal: under
    /// ACP, stdout *is* the protocol channel, so a progress line written there
    /// corrupts the stream and the editor drops the connection. A front end that
    /// owns stdout needs the events without them being printed.
    sink: Option<Arc<EventSink>>,
    /// Record the full request and response at every engine call.
    ///
    /// Turns the trace from an audit log into a training corpus. Costs a great
    /// deal of disk, so it is off unless a caller asks — see
    /// [`TraceEvent::Exchange`].
    collect_exchanges: bool,
    run_id: String,
    exchange_cursor: Mutex<usize>,
}

/// Whether an event is fit to send down a front end's progress channel.
///
/// [`TraceEvent::Exchange`] is not. It carries the entire prompt, which is
/// training data rather than progress; pushing tens of thousands of tokens per
/// step at a UI would drown every other event and stall whoever is reading.
/// Where live events go besides the trace file. See [`Session::with_sink`].
pub type EventSink = dyn Fn(&TraceEvent) + Send + Sync;

fn streamable(event: &TraceEvent) -> bool {
    !matches!(
        event,
        TraceEvent::Exchange { .. } | TraceEvent::ExchangeDelta { .. }
    )
}

impl Session {
    pub fn new(root: impl Into<PathBuf>, engine_name: impl Into<String>) -> Self {
        Session {
            root: root.into(),
            engine_name: engine_name.into(),
            messages: Vec::new(),
            trace_path: None,
            stream_stdout: false,
            sink: None,
            collect_exchanges: false,
            run_id: format!(
                "{}-{}",
                std::process::id(),
                chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
            ),
            exchange_cursor: Mutex::new(0),
        }
    }

    /// Send every streamable event to `sink` as well as the trace file.
    ///
    /// The sink runs on whatever thread logged the event and inside the run, so
    /// it must not block: queue the event and return.
    pub fn with_sink(mut self, sink: Arc<EventSink>) -> Self {
        self.sink = Some(sink);
        self
    }

    /// Record the full prompt and completion at every engine call.
    ///
    /// Requires a trace file: exchanges are never streamed, so with nothing to
    /// write to there would be nowhere for them to go.
    pub fn collecting(mut self) -> Self {
        self.collect_exchanges = true;
        self
    }

    /// Whether to build an [`TraceEvent::Exchange`] at all.
    ///
    /// Checked by the caller before cloning a request, so a run that is not
    /// collecting does not pay to copy the conversation on every step.
    pub fn collects_exchanges(&self) -> bool {
        self.collect_exchanges && self.trace_path.is_some()
    }

    /// Stream events to stdout as well as to any trace file.
    ///
    /// Callers that enable this must keep every other byte of output on
    /// stderr: stdout becomes a protocol channel, and one stray `println!`
    /// corrupts it.
    pub fn streaming(mut self) -> Self {
        self.stream_stdout = true;
        self
    }

    /// Write the trajectory to `path`, creating parent directories.
    pub fn with_trace(mut self, path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        self.trace_path = Some(path);
        Ok(self)
    }

    pub fn trace_path(&self) -> Option<&Path> {
        self.trace_path.as_deref()
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Record one engine exchange without repeating the entire prefix on every
    /// turn. Consumers reconstruct the request by appending `messages` at
    /// `messages_start`; `reset` starts a new prefix after compaction.
    pub fn log_exchange(
        &self,
        step: usize,
        request: &crate::engine::Request,
        response: &crate::engine::Response,
    ) {
        if !self.collects_exchanges() {
            return;
        }
        let mut cursor = self
            .exchange_cursor
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let reset = *cursor == 0 || request.messages.len() < *cursor;
        let start = if reset { 0 } else { *cursor };
        self.log(&TraceEvent::ExchangeDelta {
            step,
            reset,
            system: reset.then(|| request.system.clone()),
            tools: reset.then(|| request.tools.clone()),
            messages_start: start,
            messages: request.messages[start..].to_vec(),
            response: response.clone(),
        });
        *cursor = request.messages.len();
    }

    pub fn push(&mut self, m: Message) {
        self.messages.push(m);
    }

    /// Append one event. Trace failures are reported but never abort a run —
    /// losing the record is bad, losing the work is worse.
    pub fn log(&self, event: &TraceEvent) {
        let live = streamable(event);
        if live {
            if let Some(sink) = &self.sink {
                sink(event);
            }
        }

        let to_stdout = self.stream_stdout && live;
        if self.trace_path.is_none() && !to_stdout {
            return;
        }
        let line = match serde_json::to_string(&Envelope {
            at: chrono::Utc::now().to_rfc3339(),
            schema_version: TRACE_SCHEMA_VERSION,
            run_id: &self.run_id,
            event,
        }) {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!("could not serialize trace event: {e}");
                return;
            }
        };

        if to_stdout {
            let mut out = std::io::stdout();
            let _ = writeln!(out, "{line}");
            let _ = out.flush();
        }

        let Some(path) = &self.trace_path else {
            return;
        };

        let result = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .and_then(|mut f| writeln!(f, "{line}"));

        if let Err(e) = result {
            tracing::warn!("could not write trace to {}: {e}", path.display());
        }
    }
}

#[derive(Serialize)]
struct Envelope<'a> {
    at: String,
    schema_version: &'static str,
    run_id: &'a str,
    #[serde(flatten)]
    event: &'a TraceEvent,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_are_appended_one_json_object_per_line() {
        let dir = tempfile::tempdir().unwrap();
        let trace = dir.path().join("nested").join("trace.jsonl");
        let s = Session::new(dir.path(), "mock").with_trace(&trace).unwrap();

        s.log(&TraceEvent::StepStarted {
            index: 0,
            description: "first".into(),
        });
        s.log(&TraceEvent::Halt {
            step: 1,
            reason: "done".into(),
            detail: "oracle passed".into(),
        });

        let body = std::fs::read_to_string(&trace).unwrap();
        let lines: Vec<_> = body.lines().collect();
        assert_eq!(lines.len(), 2);

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["event"], "step_started");
        assert_eq!(first["description"], "first");
        assert!(first["at"].is_string());

        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["event"], "halt");
    }

    /// Collect what a sink is handed, for the tests below.
    fn recording() -> (Arc<EventSink>, Arc<std::sync::Mutex<Vec<String>>>) {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let into = Arc::clone(&seen);
        let sink: Arc<EventSink> = Arc::new(move |event: &TraceEvent| {
            let tag = serde_json::to_value(event)
                .ok()
                .and_then(|v| v["event"].as_str().map(str::to_string))
                .unwrap_or_default();
            into.lock().expect("seen").push(tag);
        });
        (sink, seen)
    }

    /// A sink fires with no trace file configured.
    ///
    /// `log` returns early when there is nowhere to write, and the sink call has
    /// to happen before that check — under ACP the sink is the *only* consumer,
    /// so getting this order wrong makes the editor go silent while the run
    /// proceeds invisibly.
    #[test]
    fn a_sink_receives_events_with_no_trace_file() {
        let dir = tempfile::tempdir().unwrap();
        let (sink, seen) = recording();
        let s = Session::new(dir.path(), "mock").with_sink(sink);

        s.log(&TraceEvent::AgentMessage {
            step: 1,
            text: "hello".into(),
        });

        assert_eq!(*seen.lock().unwrap(), ["agent_message"]);
    }

    /// The sink is fed alongside the trace, not instead of it.
    #[test]
    fn a_sink_does_not_displace_the_trace_file() {
        let dir = tempfile::tempdir().unwrap();
        let trace = dir.path().join("trace.jsonl");
        let (sink, seen) = recording();
        let s = Session::new(dir.path(), "mock")
            .with_trace(&trace)
            .unwrap()
            .with_sink(sink);

        s.log(&TraceEvent::AgentMessage {
            step: 1,
            text: "hello".into(),
        });

        assert_eq!(*seen.lock().unwrap(), ["agent_message"]);
        assert!(std::fs::read_to_string(&trace)
            .unwrap()
            .contains("agent_message"));
    }

    /// `Exchange` carries the whole conversation. It belongs in the trace and
    /// nowhere near a live consumer — the same reason it is kept off stdout.
    #[test]
    fn a_sink_is_spared_the_exchange_event() {
        let dir = tempfile::tempdir().unwrap();
        let (sink, seen) = recording();
        let s = Session::new(dir.path(), "mock").with_sink(sink);

        s.log(&TraceEvent::Exchange {
            step: 1,
            request: crate::engine::Request::new("sys", vec![]),
            response: crate::engine::Response {
                content: Vec::new(),
                stop_reason: crate::engine::StopReason::EndTurn,
                usage: crate::engine::Usage::default(),
            },
        });
        s.log(&TraceEvent::AgentMessage {
            step: 1,
            text: "hello".into(),
        });

        assert_eq!(
            *seen.lock().unwrap(),
            ["agent_message"],
            "the whole conversation must not reach a live consumer",
        );
    }

    #[test]
    fn a_session_without_a_trace_path_is_silent() {
        let s = Session::new(".", "mock");
        s.log(&TraceEvent::StepStarted {
            index: 0,
            description: "x".into(),
        });
        assert!(s.trace_path().is_none());
    }
}
