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

use anyhow::Result;
use serde::Serialize;

use crate::engine::Message;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum TraceEvent {
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
    TaskFinished {
        steps_used: usize,
        outcome: String,
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
    /// Record the full request and response at every engine call.
    ///
    /// Turns the trace from an audit log into a training corpus. Costs a great
    /// deal of disk, so it is off unless a caller asks — see
    /// [`TraceEvent::Exchange`].
    collect_exchanges: bool,
}

/// Whether an event is fit to send down a front end's progress channel.
///
/// [`TraceEvent::Exchange`] is not. It carries the entire prompt, which is
/// training data rather than progress; pushing tens of thousands of tokens per
/// step at a UI would drown every other event and stall whoever is reading.
fn streamable(event: &TraceEvent) -> bool {
    !matches!(event, TraceEvent::Exchange { .. })
}

impl Session {
    pub fn new(root: impl Into<PathBuf>, engine_name: impl Into<String>) -> Self {
        Session {
            root: root.into(),
            engine_name: engine_name.into(),
            messages: Vec::new(),
            trace_path: None,
            stream_stdout: false,
            collect_exchanges: false,
        }
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

    pub fn push(&mut self, m: Message) {
        self.messages.push(m);
    }

    /// Append one event. Trace failures are reported but never abort a run —
    /// losing the record is bad, losing the work is worse.
    pub fn log(&self, event: &TraceEvent) {
        let to_stdout = self.stream_stdout && streamable(event);
        if self.trace_path.is_none() && !to_stdout {
            return;
        }
        let line = match serde_json::to_string(&Envelope {
            at: chrono::Utc::now().to_rfc3339(),
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

        s.log(&TraceEvent::StepStarted { index: 0, description: "first".into() });
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

    #[test]
    fn a_session_without_a_trace_path_is_silent() {
        let s = Session::new(".", "mock");
        s.log(&TraceEvent::StepStarted { index: 0, description: "x".into() });
        assert!(s.trace_path().is_none());
    }
}
