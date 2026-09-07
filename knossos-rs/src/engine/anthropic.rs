//! Anthropic Messages API backend.
//!
//! The reference implementation of the engine slot: native tool use, so no
//! prompted-JSON shim is involved and tool calls round-trip exactly.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::engine::types::{
    Content, Message, Request, Response, Role, StopReason, StreamDelta, Usage,
};
use crate::engine::Engine;

const API_VERSION: &str = "2023-06-01";
/// Names this backend in [`EngineError`](crate::engine::EngineError). Kept as
/// the exact wording the old `bail!` used, so messages do not change.
const PROVIDER: &str = "Anthropic API";
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// Most capable model in the current family. Sonnet is the cheaper swap for
/// long agent loops — set `KNOSSOS_MODEL` to override.
pub const DEFAULT_MODEL: &str = "claude-opus-5";

pub struct AnthropicEngine {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    name: String,
}

impl AnthropicEngine {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let model = model.into();
        AnthropicEngine {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            name: format!("anthropic:{model}"),
            model,
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

#[async_trait]
impl Engine for AnthropicEngine {
    async fn complete(&self, req: &Request) -> Result<Response> {
        let body = WireRequest {
            model: &self.model,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            system: &req.system,
            messages: req.messages.iter().map(to_wire_message).collect(),
            tools: req
                .tools
                .iter()
                .map(|t| WireTool {
                    name: &t.name,
                    description: &t.description,
                    input_schema: &t.input_schema,
                })
                .collect(),
            stream: None,
        };

        let resp = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::engine::EngineError::Transport {
                provider: PROVIDER,
                detail: format!("request failed: {e}"),
            })?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .context("reading Anthropic response body")?;
        if !status.is_success() {
            // Typed rather than `bail!`ed so the retry policy can read the
            // status instead of the sentence — see `engine::error`.
            return Err(crate::engine::EngineError::Status {
                provider: PROVIDER,
                status: status.as_u16(),
                body: text,
            }
            .into());
        }

        let wire: WireResponse = serde_json::from_str(&text)
            .with_context(|| format!("decoding Anthropic response: {text}"))?;

        Ok(Response {
            content: wire.content.into_iter().map(from_wire_content).collect(),
            stop_reason: match wire.stop_reason.as_deref() {
                Some("end_turn") | Some("stop_sequence") => StopReason::EndTurn,
                Some("tool_use") => StopReason::ToolUse,
                Some("max_tokens") => StopReason::MaxTokens,
                Some(other) => StopReason::Other(other.to_string()),
                None => StopReason::EndTurn,
            },
            usage: Usage {
                input_tokens: wire.usage.input_tokens,
                output_tokens: wire.usage.output_tokens,
            },
        })
    }

    async fn complete_stream(
        &self,
        req: &Request,
        on_delta: &(dyn Fn(StreamDelta) + Send + Sync),
    ) -> Result<Response> {
        let body = WireRequest {
            model: &self.model,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            system: &req.system,
            messages: req.messages.iter().map(to_wire_message).collect(),
            tools: req
                .tools
                .iter()
                .map(|t| WireTool {
                    name: &t.name,
                    description: &t.description,
                    input_schema: &t.input_schema,
                })
                .collect(),
            stream: Some(true),
        };

        let mut resp = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::engine::EngineError::Transport {
                provider: PROVIDER,
                detail: format!("stream request failed: {e}"),
            })?;

        if !resp.status().is_success() {
            return self.complete(req).await;
        }

        let mut assembler = AnthropicAssembler::default();
        let mut buf = String::new();
        loop {
            let chunk = resp.chunk().await.context("reading Anthropic stream")?;
            let Some(bytes) = chunk else {
                break;
            };
            buf.push_str(&String::from_utf8_lossy(&bytes));
            drain_anthropic(&mut buf, &mut assembler, on_delta);
        }
        if !buf.trim().is_empty() {
            buf.push_str("\n\n");
            drain_anthropic(&mut buf, &mut assembler, on_delta);
        }
        Ok(assembler.finish())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports_native_tools(&self) -> bool {
        true
    }
}

// ---- wire format (private; nothing below escapes this module) ----

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    temperature: f32,
    system: &'a str,
    messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Serialize)]
struct WireTool<'a> {
    name: &'a str,
    description: &'a str,
    input_schema: &'a serde_json::Value,
}

#[derive(Serialize)]
struct WireMessage {
    role: &'static str,
    content: Vec<WireContent>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireContent {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

#[derive(Deserialize)]
struct WireResponse {
    content: Vec<WireContent>,
    stop_reason: Option<String>,
    usage: WireUsage,
}

#[derive(Deserialize)]
struct WireUsage {
    input_tokens: u64,
    output_tokens: u64,
}

fn to_wire_message(m: &Message) -> WireMessage {
    WireMessage {
        role: match m.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        },
        content: m.content.iter().map(to_wire_content).collect(),
    }
}

fn to_wire_content(c: &Content) -> WireContent {
    match c {
        Content::Text { text } => WireContent::Text { text: text.clone() },
        Content::ToolUse { id, name, input } => WireContent::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(),
        },
        Content::ToolResult {
            id,
            content,
            is_error,
        } => WireContent::ToolResult {
            tool_use_id: id.clone(),
            content: content.clone(),
            is_error: *is_error,
        },
    }
}

#[derive(Default)]
struct AnthropicAssembler {
    text: String,
    tools: std::collections::BTreeMap<usize, (String, String, String)>, // id, name, json
    thinking: BTreeMap<usize, bool>,
    stop_reason: Option<String>,
    usage: Usage,
}

impl AnthropicAssembler {
    fn ingest(&mut self, data: &str, on_delta: &(dyn Fn(StreamDelta) + Send + Sync)) {
        let data = data.trim();
        if data.is_empty() {
            return;
        }
        let Ok(ev) = serde_json::from_str::<serde_json::Value>(data) else {
            return;
        };
        let kind = ev.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match kind {
            "content_block_start" => {
                let index = ev.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let block = ev
                    .get("content_block")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                match block.get("type").and_then(|v| v.as_str()) {
                    Some("tool_use") => {
                        let id = block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = block
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        self.tools.insert(index, (id, name, String::new()));
                    }
                    Some("thinking") => {
                        self.thinking.insert(index, true);
                    }
                    _ => {}
                }
            }
            "content_block_delta" => {
                let index = ev.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let delta = ev.get("delta").cloned().unwrap_or(serde_json::Value::Null);
                match delta.get("type").and_then(|v| v.as_str()) {
                    Some("text_delta") => {
                        if let Some(piece) = delta.get("text").and_then(|v| v.as_str()) {
                            if !piece.is_empty() {
                                self.text.push_str(piece);
                                on_delta(StreamDelta::Text(piece.to_string()));
                            }
                        }
                    }
                    Some("thinking_delta") | Some("reasoning_delta") => {
                        if let Some(piece) = delta
                            .get("thinking")
                            .or(delta.get("text"))
                            .and_then(|v| v.as_str())
                        {
                            if !piece.is_empty() {
                                on_delta(StreamDelta::Thought(piece.to_string()));
                            }
                        }
                    }
                    Some("input_json_delta") => {
                        if let Some(piece) = delta.get("partial_json").and_then(|v| v.as_str()) {
                            if let Some(slot) = self.tools.get_mut(&index) {
                                slot.2.push_str(piece);
                            }
                        }
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                if let Some(reason) = ev.pointer("/delta/stop_reason").and_then(|v| v.as_str()) {
                    self.stop_reason = Some(reason.to_string());
                }
                if let Some(out) = ev.pointer("/usage/output_tokens").and_then(|v| v.as_u64()) {
                    self.usage.output_tokens = out;
                }
            }
            "message_start" => {
                if let Some(inp) = ev
                    .pointer("/message/usage/input_tokens")
                    .and_then(|v| v.as_u64())
                {
                    self.usage.input_tokens = inp;
                }
            }
            _ => {}
        }
    }

    fn finish(self) -> Response {
        let mut content = Vec::new();
        if !self.text.is_empty() {
            content.push(Content::text(self.text));
        }
        for (_, (id, name, json)) in self.tools {
            if name.is_empty() {
                continue;
            }
            let input = serde_json::from_str(&json).unwrap_or(serde_json::json!({}));
            content.push(Content::ToolUse { id, name, input });
        }
        let wants_tools = content.iter().any(|c| matches!(c, Content::ToolUse { .. }));
        Response {
            content,
            stop_reason: match self.stop_reason.as_deref() {
                Some("tool_use") => StopReason::ToolUse,
                Some("max_tokens") => StopReason::MaxTokens,
                Some("end_turn") | Some("stop_sequence") | None if wants_tools => {
                    StopReason::ToolUse
                }
                Some("end_turn") | Some("stop_sequence") | None => StopReason::EndTurn,
                Some(other) => StopReason::Other(other.to_string()),
            },
            usage: self.usage,
        }
    }
}

fn drain_anthropic(
    buf: &mut String,
    assembler: &mut AnthropicAssembler,
    on_delta: &(dyn Fn(StreamDelta) + Send + Sync),
) {
    loop {
        let (pos, sep_len) = if let Some(p) = buf.find("\r\n\r\n") {
            (p, 4)
        } else if let Some(p) = buf.find("\n\n") {
            (p, 2)
        } else {
            break;
        };
        let frame = buf[..pos].to_string();
        buf.replace_range(..pos + sep_len, "");
        for line in frame.lines() {
            let line = line.trim_end_matches('\r');
            if let Some(data) = line.strip_prefix("data:") {
                assembler.ingest(data.trim(), on_delta);
            }
        }
    }
}

fn from_wire_content(w: WireContent) -> Content {
    match w {
        WireContent::Text { text } => Content::Text { text },
        WireContent::ToolUse { id, name, input } => Content::ToolUse { id, name, input },
        WireContent::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => Content::ToolResult {
            id: tool_use_id,
            content,
            is_error,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_result_maps_to_tool_use_id() {
        let c = Content::ToolResult {
            id: "abc".into(),
            content: "ok".into(),
            is_error: false,
        };
        let json = serde_json::to_value(to_wire_content(&c)).unwrap();
        assert_eq!(json["type"], "tool_result");
        assert_eq!(json["tool_use_id"], "abc");
    }

    #[test]
    fn round_trips_a_tool_use_block() {
        let wire: WireContent = serde_json::from_value(serde_json::json!({
            "type": "tool_use", "id": "x", "name": "read", "input": {"path": "a.rs"}
        }))
        .unwrap();
        match from_wire_content(wire) {
            Content::ToolUse { id, name, input } => {
                assert_eq!(id, "x");
                assert_eq!(name, "read");
                assert_eq!(input["path"], "a.rs");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn stream_text_deltas_join() {
        let mut a = AnthropicAssembler::default();
        let seen = std::sync::Mutex::new(Vec::new());
        a.ingest(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}"#,
            &|d| seen.lock().unwrap().push(d),
        );
        a.ingest(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}"#,
            &|d| seen.lock().unwrap().push(d),
        );
        a.ingest(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
            &|_| {},
        );
        let resp = a.finish();
        assert_eq!(resp.text(), "Hello");
        assert_eq!(seen.lock().unwrap().len(), 2);
    }
}
