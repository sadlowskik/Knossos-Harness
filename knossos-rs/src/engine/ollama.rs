//! Local Ollama backend.
//!
//! Two shape mismatches with the neutral types are handled here rather than
//! leaking upward:
//!
//! 1. Ollama's `/api/chat` has no separate system field, so the system prompt
//!    becomes a leading `system` message.
//! 2. Ollama messages carry a single content string, so one neutral message
//!    holding several tool results expands into several wire messages.
//!
//! Native tool support varies by model, which is why `native_tools` is a
//! constructor flag rather than a constant: point it at a model without tool
//! calling, set it false, and the harness routes through the prompted-JSON
//! shim in `prompt_fallback`.

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::engine::types::{
    Content, Message, Request, Response, Role, StopReason, Usage,
};
use crate::engine::Engine;

pub const DEFAULT_BASE_URL: &str = "http://localhost:11434";
pub const DEFAULT_MODEL: &str = "qwen3-coder:30b";

pub struct OllamaEngine {
    client: reqwest::Client,
    model: String,
    base_url: String,
    name: String,
    native_tools: bool,
}

impl OllamaEngine {
    pub fn new(model: impl Into<String>) -> Self {
        let model = model.into();
        OllamaEngine {
            client: reqwest::Client::new(),
            name: format!("ollama:{model}"),
            model,
            base_url: DEFAULT_BASE_URL.to_string(),
            native_tools: true,
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Declare that this model cannot call tools natively, routing it through
    /// the prompted-JSON shim.
    pub fn without_native_tools(mut self) -> Self {
        self.native_tools = false;
        self
    }
}

#[async_trait]
impl Engine for OllamaEngine {
    async fn complete(&self, req: &Request) -> Result<Response> {
        let mut messages = vec![WireMessage {
            role: "system",
            content: req.system.clone(),
            tool_calls: Vec::new(),
        }];
        for m in &req.messages {
            messages.extend(to_wire_messages(m));
        }

        let body = WireRequest {
            model: &self.model,
            messages,
            stream: false,
            tools: req
                .tools
                .iter()
                .map(|t| WireTool {
                    kind: "function",
                    function: WireFunction {
                        name: &t.name,
                        description: &t.description,
                        parameters: &t.input_schema,
                    },
                })
                .collect(),
            options: WireOptions {
                temperature: req.temperature,
                num_predict: req.max_tokens,
            },
        };

        let resp = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await
            .context("request to Ollama failed (is `ollama serve` running?)")?;

        let status = resp.status();
        let text = resp.text().await.context("reading Ollama response body")?;
        if !status.is_success() {
            bail!("Ollama returned {status}: {text}");
        }

        let wire: WireResponse = serde_json::from_str(&text)
            .with_context(|| format!("decoding Ollama response: {text}"))?;

        let mut content = Vec::new();
        if !wire.message.content.trim().is_empty() {
            content.push(Content::text(wire.message.content));
        }
        // Ollama tool calls carry no id, so the harness assigns one.
        for (i, call) in wire.message.tool_calls.into_iter().enumerate() {
            content.push(Content::ToolUse {
                id: format!("call_{i}"),
                name: call.function.name,
                input: call.function.arguments,
            });
        }

        let wants_tools = content
            .iter()
            .any(|c| matches!(c, Content::ToolUse { .. }));

        Ok(Response {
            content,
            stop_reason: if wants_tools {
                StopReason::ToolUse
            } else {
                match wire.done_reason.as_deref() {
                    Some("length") => StopReason::MaxTokens,
                    _ => StopReason::EndTurn,
                }
            },
            usage: Usage {
                input_tokens: wire.prompt_eval_count.unwrap_or(0),
                output_tokens: wire.eval_count.unwrap_or(0),
            },
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports_native_tools(&self) -> bool {
        self.native_tools
    }
}

// ---- wire format (private) ----

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    messages: Vec<WireMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool<'a>>,
    options: WireOptions,
}

#[derive(Serialize)]
struct WireOptions {
    temperature: f32,
    num_predict: u32,
}

#[derive(Serialize)]
struct WireTool<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    function: WireFunction<'a>,
}

#[derive(Serialize)]
struct WireFunction<'a> {
    name: &'a str,
    description: &'a str,
    parameters: &'a serde_json::Value,
}

#[derive(Serialize)]
struct WireMessage {
    role: &'static str,
    content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<WireToolCall>,
}

#[derive(Serialize, Deserialize)]
struct WireToolCall {
    function: WireToolCallFunction,
}

#[derive(Serialize, Deserialize)]
struct WireToolCallFunction {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

#[derive(Deserialize)]
struct WireResponse {
    message: WireResponseMessage,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    eval_count: Option<u64>,
}

#[derive(Deserialize)]
struct WireResponseMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    tool_calls: Vec<WireToolCall>,
}

/// One neutral message becomes one or more Ollama messages: tool results must
/// each be their own `role: "tool"` entry.
fn to_wire_messages(m: &Message) -> Vec<WireMessage> {
    let mut out = Vec::new();
    let mut text = String::new();
    let mut tool_calls = Vec::new();

    for c in &m.content {
        match c {
            Content::Text { text: t } => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(t);
            }
            Content::ToolUse { name, input, .. } => tool_calls.push(WireToolCall {
                function: WireToolCallFunction {
                    name: name.clone(),
                    arguments: input.clone(),
                },
            }),
            Content::ToolResult { content, .. } => out.push(WireMessage {
                role: "tool",
                content: content.clone(),
                tool_calls: Vec::new(),
            }),
        }
    }

    if !text.is_empty() || !tool_calls.is_empty() {
        let msg = WireMessage {
            role: match m.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            },
            content: text,
            tool_calls,
        };
        // Assistant tool calls must precede their results in the transcript.
        out.insert(0, msg);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_results_become_separate_tool_messages() {
        let m = Message::user(vec![
            Content::ToolResult {
                id: "a".into(),
                content: "first".into(),
                is_error: false,
            },
            Content::ToolResult {
                id: "b".into(),
                content: "second".into(),
                is_error: false,
            },
        ]);
        let wire = to_wire_messages(&m);
        assert_eq!(wire.len(), 2);
        assert!(wire.iter().all(|w| w.role == "tool"));
        assert_eq!(wire[0].content, "first");
    }

    #[test]
    fn assistant_tool_calls_are_flattened_into_one_message() {
        let m = Message::assistant(vec![
            Content::text("working on it"),
            Content::ToolUse {
                id: "x".into(),
                name: "read".into(),
                input: serde_json::json!({"path": "a.rs"}),
            },
        ]);
        let wire = to_wire_messages(&m);
        assert_eq!(wire.len(), 1);
        assert_eq!(wire[0].role, "assistant");
        assert_eq!(wire[0].tool_calls.len(), 1);
        assert_eq!(wire[0].tool_calls[0].function.name, "read");
    }
}
