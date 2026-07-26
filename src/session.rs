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
}

impl Session {
    pub fn new(root: impl Into<PathBuf>, engine_name: impl Into<String>) -> Self {
        Session {
            root: root.into(),
            engine_name: engine_name.into(),
            messages: Vec::new(),
            trace_path: None,
            stream_stdout: false,
        }
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
        if self.trace_path.is_none() && !self.stream_stdout {
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

        if self.stream_stdout {
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
