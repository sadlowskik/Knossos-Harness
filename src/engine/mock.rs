//! A scripted engine.
//!
//! Exists so the entire test suite runs with no network, no API key and no
//! local model server. Tests that exercise Talos, Ariadne or the tool
//! dispatcher script the exact reply sequence they need and then assert on
//! what the harness did with it.

use std::collections::VecDeque;
use std::sync::Mutex;

use anyhow::{bail, Result};
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
