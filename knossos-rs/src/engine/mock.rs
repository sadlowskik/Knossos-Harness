//! A scripted engine.
//!
//! Exists so the entire test suite runs with no network, no API key and no
//! local model server. Tests that exercise Talos, Ariadne or the tool
//! dispatcher script the exact reply sequence they need and then assert on
//! what the harness did with it.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use async_trait::async_trait;

use crate::engine::types::{Content, Request, Response, StopReason, Usage};
use crate::engine::Engine;

pub struct MockEngine {
    name: String,
    native_tools: bool,
    scripted: Mutex<VecDeque<Response>>,
    /// Every request the harness made, in order — assert against this.
    pub seen: Mutex<Vec<Request>>,
}

impl MockEngine {
    pub fn new(scripted: Vec<Response>) -> Self {
        MockEngine {
            name: "mock".to_string(),
            native_tools: true,
            scripted: Mutex::new(scripted.into()),
            seen: Mutex::new(Vec::new()),
        }
    }

    /// Convenience: an engine that replies with one plain text turn.
    pub fn text(reply: &str) -> Self {
        MockEngine::new(vec![Response {
            content: vec![Content::text(reply)],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        }])
    }

    pub fn without_native_tools(mut self) -> Self {
        self.native_tools = false;
        self
    }

    pub fn call_count(&self) -> usize {
        self.seen.lock().unwrap().len()
    }

    /// Conformance / `KNOSSOS_SCRIPT`: each entry is prose or a fenced tool call.
    pub fn from_script(replies: Vec<String>) -> Self {
        let scripted = replies
            .into_iter()
            .map(|s| {
                let mut r = text_response(&s);
                crate::engine::prompt_fallback::extract_tool_calls(&mut r);
                r
            })
            .collect();
        MockEngine::new(scripted)
    }

    /// `KNOSSOS_SCRIPT` is a JSON array of strings, same as Python `scripted_agent.py`.
    pub fn from_env() -> Option<Self> {
        let raw = std::env::var("KNOSSOS_SCRIPT").ok()?;
        if raw.trim().is_empty() {
            return None;
        }
        let values: Vec<serde_json::Value> = serde_json::from_str(&raw).ok()?;
        let replies = values
            .into_iter()
            .map(|v| match v {
                serde_json::Value::String(s) => s,
                other => other.to_string(),
            })
            .collect();
        Some(Self::from_script(replies))
    }

    /// Replay provider responses captured in a collected v2 trace.
    /// Requests are rebuilt by each ablation so the provider output remains
    /// fixed while retrieval, memory, or compaction changes independently.
    pub fn from_trace(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading replay trace {}", path.display()))?;
        let mut responses = Vec::new();
        for (index, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let event: serde_json::Value = serde_json::from_str(line).with_context(|| {
                format!(
                    "decoding replay trace {} line {}",
                    path.display(),
                    index + 1
                )
            })?;
            if !matches!(
                event.get("event").and_then(serde_json::Value::as_str),
                Some("exchange") | Some("exchange_delta")
            ) {
                continue;
            }
            let response = event.get("response").cloned().ok_or_else(|| {
                anyhow::anyhow!("replay exchange has no response at line {}", index + 1)
            })?;
            responses.push(
                serde_json::from_value(response)
                    .with_context(|| format!("decoding replay response at line {}", index + 1))?,
            );
        }
        if responses.is_empty() {
            bail!("replay trace {} contains no exchanges", path.display());
        }
        let mut engine = Self::new(responses);
        engine.name = format!("replay:{}", path.display());
        Ok(engine)
    }
}

#[async_trait]
impl Engine for MockEngine {
    async fn complete(&self, req: &Request) -> Result<Response> {
        self.seen.lock().unwrap().push(req.clone());
        match self.scripted.lock().unwrap().pop_front() {
            Some(r) => Ok(r),
            None => bail!("MockEngine ran out of scripted responses"),
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports_native_tools(&self) -> bool {
        self.native_tools
    }
}

/// Build a response containing a single tool call.
pub fn tool_call(id: &str, name: &str, input: serde_json::Value) -> Response {
    Response {
        content: vec![Content::ToolUse {
            id: id.to_string(),
            name: name.to_string(),
            input,
        }],
        stop_reason: StopReason::ToolUse,
        usage: Usage::default(),
    }
}

/// Build a plain text response that ends the turn.
pub fn text_response(text: &str) -> Response {
    Response {
        content: vec![Content::text(text)],
        stop_reason: StopReason::EndTurn,
        usage: Usage::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_script_extracts_a_fenced_call_with_legacy_args() {
        let fence = "```json\n{\"tool\":\"write_file\",\"args\":{\"path\":\"a.py\",\"content\":\"x\"}}\n```";
        let eng = MockEngine::from_script(vec![fence.into(), "Done.".into()]);
        let first = eng.scripted.lock().unwrap().pop_front().unwrap();
        let uses = first.tool_uses();
        assert_eq!(uses.len(), 1);
        assert_eq!(uses[0].1, "write_file");
        assert_eq!(uses[0].2["path"], "a.py");
    }

    #[test]
    fn a_collected_trace_replays_exchange_responses() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.jsonl");
        std::fs::write(
            &path,
            concat!(
                "{\"event\":\"task_started\"}\n",
                "{\"event\":\"exchange_delta\",\"response\":{\"content\":[{\"kind\":\"text\",\"text\":\"one\"}],\"stop_reason\":\"end_turn\",\"usage\":{\"input_tokens\":2,\"output_tokens\":1}}}\n"
            ),
        )
        .unwrap();

        let engine = MockEngine::from_trace(&path).unwrap();
        let response = engine.scripted.lock().unwrap().pop_front().unwrap();
        assert_eq!(response.text(), "one");
        assert!(engine.name().starts_with("replay:"));
    }
}
