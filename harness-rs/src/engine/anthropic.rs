//! Anthropic Messages API backend.
//!
//! The reference implementation of the engine slot: native tool use, so no
//! prompted-JSON shim is involved and tool calls round-trip exactly.

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::engine::types::{
    Content, Message, Request, Response, Role, StopReason, Usage,
};
use crate::engine::Engine;

const API_VERSION: &str = "2023-06-01";
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// Most capable model in the current family. Sonnet is the cheaper swap for
/// long agent loops — set `DAEDALUS_MODEL` to override.
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
            .context("request to Anthropic API failed")?;

        let status = resp.status();
        let text = resp.text().await.context("reading Anthropic response body")?;
        if !status.is_success() {
            bail!("Anthropic API returned {status}: {text}");
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
        Content::ToolResult { id, content, is_error } => WireContent::ToolResult {
            tool_use_id: id.clone(),
            content: content.clone(),
            is_error: *is_error,
        },
    }
}

fn from_wire_content(w: WireContent) -> Content {
    match w {
        WireContent::Text { text } => Content::Text { text },
        WireContent::ToolUse { id, name, input } => Content::ToolUse { id, name, input },
        WireContent::ToolResult { tool_use_id, content, is_error } => Content::ToolResult {
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
}
