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

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::engine::types::{
    Content, Message, Request, Response, Role, StopReason, Usage,
};
use crate::engine::Engine;

/// Names this backend in [`EngineError`](crate::engine::EngineError).
const PROVIDER: &str = "Ollama";
pub const DEFAULT_BASE_URL: &str = "http://localhost:11434";
pub const DEFAULT_MODEL: &str = "qwen3-coder:30b";

/// Context window requested of Ollama when the caller states no preference.
///
/// Ollama's own default is 4096, which is far below what a harness prompt needs
/// once the constitution, the symbol index and retrieved context are in it — and
/// the overflow is discarded silently. 32k matches the `-32k` model variants
/// that were previously being built by hand to work around exactly this.
pub const DEFAULT_NUM_CTX: u32 = 32_768;

pub struct OllamaEngine {
    client: reqwest::Client,
    model: String,
    base_url: String,
    name: String,
    native_tools: bool,
    num_ctx: Option<u32>,
    think: Option<bool>,
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
            num_ctx: Some(DEFAULT_NUM_CTX),
            think: None,
        }
    }

    /// Ask a reasoning model to answer without reasoning first.
    ///
    /// `None` leaves the model's own default. `Some(false)` is worth measuring
    /// in a harness: reasoning is generated *before* the answer and billed as
    /// output, and much of what it does — decide an approach, check the work —
    /// this harness already does with a plan it can inspect and a verifier that
    /// actually compiles the result. Paying twice for that is a choice rather
    /// than a given, which is why it is a flag and not a default.
    pub fn with_think(mut self, think: Option<bool>) -> Self {
        self.think = think;
        self
    }

    /// Set the context window, or `None` to accept whatever the model declares.
    pub fn with_num_ctx(mut self, num_ctx: Option<u32>) -> Self {
        self.num_ctx = num_ctx;
        self
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
                num_ctx: self.num_ctx,
            },
            think: self.think,
        };

        let resp = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::engine::EngineError::Transport {
                provider: PROVIDER,
                detail: format!("request failed ({e}) — is `ollama serve` running?"),
            })?;

        let status = resp.status();
        let text = resp.text().await.context("reading Ollama response body")?;
        if !status.is_success() {
            return Err(crate::engine::EngineError::Status {
                provider: PROVIDER,
                status: status.as_u16(),
                body: text,
            }
            .into());
        }

        let wire: WireResponse = serde_json::from_str(&text)
            .with_context(|| format!("decoding Ollama response: {text}"))?;

        let mut content = Vec::new();
        if !wire.message.content.trim().is_empty() {
            content.push(Content::text(wire.message.content));
        } else if wire.message.tool_calls.is_empty() {
            // Nothing to say and nothing to do, but it may still have thought.
            // Reasoning is not an answer and is never used *alongside* one — it
            // would pollute the conversation replayed to the model next turn.
            // As the sole fallback it is strictly better than an empty turn: the
            // user sees what the model was doing instead of a blank reply, and
            // Ariadne sees a step rather than silence.
            if let Some(thinking) = wire.message.thinking.filter(|t| !t.trim().is_empty()) {
                content.push(Content::text(thinking));
            }
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
    /// Top-level, not an option: Ollama treats reasoning as a mode rather than
    /// a sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
}

#[derive(Serialize)]
struct WireOptions {
    temperature: f32,
    num_predict: u32,
    /// Context window. Omitted leaves Ollama's own default, which is 4096
    /// regardless of what the model supports.
    ///
    /// That default is a silent failure, not a loud one: the prompt is truncated
    /// from the left, so the constitution and the symbol index disappear first
    /// and the model answers a question it was never fully asked. A 262k-context
    /// model behaves like a 4k one and nothing in the response says so.
    #[serde(skip_serializing_if = "Option::is_none")]
    num_ctx: Option<u32>,
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
    /// Reasoning, which newer Ollama returns *separately* from the answer.
    ///
    /// A thinking model can finish a turn having filled this and left `content`
    /// empty. Read only `content` and that turn looks like the model said
    /// nothing at all — which is what a front end renders as "no response was
    /// returned", and what the loop would otherwise score as a step that did no
    /// work.
    #[serde(default)]
    thinking: Option<String>,
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

    /// Decode a response body the way `complete` does, minus the HTTP.
    fn decode(body: &str) -> Vec<Content> {
        let wire: WireResponse = serde_json::from_str(body).expect("valid body");
        let mut content = Vec::new();
        if !wire.message.content.trim().is_empty() {
            content.push(Content::text(wire.message.content));
        } else if wire.message.tool_calls.is_empty() {
            if let Some(t) = wire.message.thinking.filter(|t| !t.trim().is_empty()) {
                content.push(Content::text(t));
            }
        }
        content
    }

    /// The failure this was written for: a thinking model that fills `thinking`
    /// and leaves `content` empty reads as having said nothing at all.
    #[test]
    fn a_reply_that_is_only_thinking_is_not_an_empty_turn() {
        let out = decode(
            r#"{"message":{"role":"assistant","content":"",
                "thinking":"weighing two approaches"}}"#,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].as_text(), Some("weighing two approaches"));
    }

    /// Reasoning never rides along with an answer: it would be replayed to the
    /// model next turn as though it had said it out loud.
    #[test]
    fn thinking_is_dropped_when_there_is_a_real_answer() {
        let out = decode(
            r#"{"message":{"role":"assistant","content":"the answer",
                "thinking":"scratchpad"}}"#,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].as_text(), Some("the answer"));
    }

    /// A turn that is entirely tool calls is doing work, not going silent, so
    /// its reasoning must not be turned into prose.
    #[test]
    fn thinking_is_dropped_when_the_model_called_a_tool() {
        let out = decode(
            r#"{"message":{"role":"assistant","content":"","thinking":"I should read it",
                "tool_calls":[{"function":{"name":"read_file","arguments":{}}}]}}"#,
        );
        assert!(out.is_empty(), "reasoning must not become the assistant's prose");
    }

    /// Omitting `num_ctx` hands Ollama its own 4096 default, which truncates the
    /// prompt from the left and says nothing about having done so.
    #[test]
    fn a_request_states_its_context_window() {
        let opts = WireOptions { temperature: 0.0, num_predict: 10, num_ctx: Some(32_768) };
        let v = serde_json::to_value(&opts).expect("serialise");
        assert_eq!(v["num_ctx"], 32_768);

        let unset = WireOptions { temperature: 0.0, num_predict: 10, num_ctx: None };
        let v = serde_json::to_value(&unset).expect("serialise");
        assert!(v.get("num_ctx").is_none(), "unset must mean absent, not null");
    }

    /// Absent must mean absent. Sending `think: true` to match a model whose
    /// default is already to reason would override a setting rather than leave
    /// it, and there is no way back from that at the call site.
    #[test]
    fn reasoning_is_only_mentioned_when_a_caller_asked() {
        let engine = OllamaEngine::new("m");
        assert_eq!(engine.think, None);
        assert_eq!(OllamaEngine::new("m").with_think(Some(false)).think, Some(false));

        let wire = serde_json::to_string(&WireRequest {
            model: "m",
            messages: vec![],
            stream: false,
            tools: vec![],
            options: WireOptions { temperature: 0.0, num_predict: 1, num_ctx: None },
            think: None,
        })
        .expect("serialise");
        assert!(!wire.contains("think"), "an unset think field must not reach the wire: {wire}");
    }

    #[test]
    fn the_default_engine_asks_for_more_than_ollamas_4k() {
        assert_eq!(OllamaEngine::new("m").num_ctx, Some(DEFAULT_NUM_CTX));
        const { assert!(DEFAULT_NUM_CTX > 4096) };
        assert_eq!(OllamaEngine::new("m").with_num_ctx(None).num_ctx, None);
    }
}
