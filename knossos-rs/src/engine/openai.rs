//! OpenAI-compatible `/chat/completions` backend.
//!
//! Groq, Gemini, OpenRouter, Qwen, Fireworks and the rest all speak this
//! shape. One adapter covers them, so switching provider is a flag rather
//! than a new backend — the same engine-slot argument one level down.
//!
//! # Degrade-on-400
//!
//! Capability is discovered by sending the request and reading the refusal,
//! one wasted request per engine, sticky for its life. A 400 is not a
//! failed run: it is how a provider says "not that field". The branches
//! match `OpenAICompatEngine._run` on the Python side, in the same order:
//!
//! 1. shrink `max_tokens` to a stated reply cap (400) or a 413 token limit
//! 2. flatten native tool history
//! 3. drop sampling parameters
//! 4. fold a rejected system role into the first user message
//!
//! `stream_options` is not sent. Usage arrives on a non-stream response, or
//! on the last SSE chunk when streaming. The field that caused a whole family
//! of Python 400s stays off the wire.
//!
//! Streaming is opt-in per instance: a 400 on `stream: true` sticks the
//! engine to [`complete`](OpenAICompatEngine::complete) for the rest of its
//! life, so a host that cannot stream does not fail every turn.
//!
//! 429 is left to [`crate::resilience::Resilient`]. Degrade is about the
//! *shape* of the request; a rate limit is about when it was sent.

use std::sync::Mutex;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::engine::sse;
use crate::engine::types::{
    Content, Message, Request, Response, Role, StopReason, StreamDelta, Usage,
};
use crate::engine::{Engine, EngineError};

const PROVIDER_LABEL: &str = "OpenAI-compat API";

/// A provider call must never be able to pin an agent turn forever.  This is
/// deliberately longer than a normal completion but shorter than the eval
/// case timeout; callers can tighten it for canaries and tests.
pub const DEFAULT_HTTP_TIMEOUT: Duration = Duration::from_secs(120);

/// Smallest reply worth asking for. Below this the model cannot finish a
/// tool call, so shrinking further trades one failure for a worse one.
pub const MIN_OUTPUT_TOKENS: u32 = 512;

/// Headroom left under a provider's stated 413 limit. The limit is
/// enforced on their tokenizer, not ours.
const LIMIT_MARGIN: u32 = 256;

/// How many times one `complete` may degrade and retry. Each sticky flag
/// flips at most once; shrinks are bounded separately.
const MAX_DEGRADES: usize = 6;

#[derive(Debug, Clone)]
pub struct Provider {
    pub name: &'static str,
    pub base_url: &'static str,
    pub key_env: Option<&'static str>,
    pub default_model: &'static str,
}

/// Same table as `engine.py` `PROVIDERS`. Model ids drift; treat
/// `default_model` as a starting guess and pass `--model` when it 404s.
pub const PROVIDERS: &[Provider] = &[
    Provider {
        name: "anthropic",
        base_url: "https://api.anthropic.com/v1",
        key_env: Some("ANTHROPIC_API_KEY"),
        default_model: "claude-sonnet-5",
    },
    Provider {
        name: "groq",
        base_url: "https://api.groq.com/openai/v1",
        key_env: Some("GROQ_API_KEY"),
        default_model: "qwen/qwen3.6-27b",
    },
    Provider {
        name: "nvidia",
        base_url: "https://integrate.api.nvidia.com/v1",
        key_env: Some("NVIDIA_API_KEY"),
        default_model: "qwen/qwen3-coder-480b-a35b-instruct",
    },
    Provider {
        name: "gemini",
        base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
        key_env: Some("GEMINI_API_KEY"),
        default_model: "gemini-3.6-flash",
    },
    Provider {
        name: "cerebras",
        base_url: "https://api.cerebras.ai/v1",
        key_env: Some("CEREBRAS_API_KEY"),
        default_model: "gpt-oss-120b",
    },
    Provider {
        name: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        key_env: Some("OPENROUTER_API_KEY"),
        default_model: "meta-llama/llama-3.3-70b-instruct:free",
    },
    Provider {
        name: "qwen",
        base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        key_env: Some("DASHSCOPE_API_KEY"),
        default_model: "qwen3-coder-flash",
    },
    Provider {
        name: "fireworks",
        base_url: "https://api.fireworks.ai/inference/v1",
        key_env: Some("FIREWORKS_API_KEY"),
        default_model: "accounts/fireworks/models/deepseek-v4-flash",
    },
    Provider {
        name: "mistral",
        base_url: "https://api.mistral.ai/v1",
        key_env: Some("MISTRAL_API_KEY"),
        default_model: "mistral-small-latest",
    },
    Provider {
        name: "together",
        base_url: "https://api.together.xyz/v1",
        key_env: Some("TOGETHER_API_KEY"),
        default_model: "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    Provider {
        name: "ollama",
        base_url: "http://localhost:11434/v1",
        key_env: None,
        default_model: "qwen2.5-coder:7b",
    },
    Provider {
        name: "openwebui",
        base_url: "http://localhost:3000/api",
        key_env: Some("OPENWEBUI_API_KEY"),
        default_model: "",
    },
    Provider {
        name: "local",
        base_url: "http://localhost:8000/v1",
        key_env: None,
        default_model: "local-model",
    },
];

pub fn provider(name: &str) -> Option<&'static Provider> {
    PROVIDERS.iter().find(|p| p.name == name)
}

pub fn provider_names() -> impl Iterator<Item = &'static str> {
    PROVIDERS.iter().map(|p| p.name)
}

#[derive(Debug, Clone)]
struct Caps {
    max_tokens: u32,
    hard_output_cap: Option<u32>,
    context_window: Option<u32>,
    send_sampling: bool,
    native_history: bool,
    fold_system: bool,
    output_shrinks: u32,
    limit_restores: u32,
    /// False after a stream request is refused. Sticky, like the other caps.
    stream: bool,
}

pub struct OpenAICompatEngine {
    client: reqwest::Client,
    provider: String,
    model: String,
    base_url: String,
    api_key: Option<String>,
    name: String,
    configured_max_tokens: u32,
    caps: Mutex<Caps>,
}

impl OpenAICompatEngine {
    pub fn from_provider(
        provider_name: &str,
        model: Option<String>,
        base_url: Option<String>,
        max_tokens: u32,
    ) -> Result<Self> {
        let preset = provider(provider_name).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown provider {provider_name:?}; pick one of: {}",
                provider_names().collect::<Vec<_>>().join(", ")
            )
        })?;

        let base_url = base_url
            .or_else(|| local_base_url(provider_name))
            .unwrap_or_else(|| preset.base_url.to_string());

        let api_key = match preset.key_env {
            Some(var) => Some(std::env::var(var).with_context(|| {
                format!("{var} is not set (needed for --engine / --provider {provider_name})")
            })?),
            None => None,
        };

        let model = model
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| preset.default_model.to_string());
        if model.is_empty() {
            bail!("provider {provider_name} has no default model; pass --model");
        }

        Ok(Self::new(
            provider_name,
            model,
            base_url,
            api_key,
            max_tokens,
        ))
    }

    pub fn new(
        provider: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<String>,
        max_tokens: u32,
    ) -> Self {
        let provider = provider.into();
        let model = model.into();
        let base_url = base_url.into().trim_end_matches('/').to_string();
        OpenAICompatEngine {
            client: reqwest::Client::builder()
                .timeout(DEFAULT_HTTP_TIMEOUT)
                .build()
                .expect("the static OpenAI-compatible HTTP client configuration must build"),
            name: format!("{provider}:{model}"),
            provider,
            model,
            base_url,
            api_key,
            configured_max_tokens: max_tokens,
            caps: Mutex::new(Caps {
                max_tokens,
                hard_output_cap: None,
                context_window: std::env::var("KNOSSOS_CONTEXT_WINDOW")
                    .ok()
                    .and_then(|s| s.parse().ok()),
                send_sampling: true,
                native_history: true,
                fold_system: false,
                output_shrinks: 0,
                limit_restores: 0,
                stream: true,
            }),
        }
    }

    /// Override the total request deadline. It covers connecting, response
    /// headers, and consuming a streaming body.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.client = reqwest::Client::builder()
            .timeout(timeout.max(Duration::from_millis(1)))
            .build()
            .expect("the OpenAI-compatible HTTP client configuration must build");
        self
    }

    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub fn output_shrinks(&self) -> u32 {
        self.caps.lock().expect("caps").output_shrinks
    }

    /// Apply a server-advertised context window. It may tighten an unknown or
    /// larger inferred limit, but never widen one learned from a real refusal.
    pub fn bound_context_window(&self, window: u32) {
        if window == 0 {
            return;
        }
        let mut caps = self.caps.lock().expect("caps");
        if caps.context_window.map(|old| window < old).unwrap_or(true) {
            caps.context_window = Some(window);
        }
    }

    pub fn limit_restores(&self) -> u32 {
        self.caps.lock().expect("caps").limit_restores
    }

    async fn post(&self, payload: &Value) -> Result<(u16, String)> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut req = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .json(payload);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await.map_err(|e| EngineError::Transport {
            provider: PROVIDER_LABEL,
            detail: format!("request failed: {e}"),
        })?;
        let status = resp.status().as_u16();
        let text = resp
            .text()
            .await
            .context("reading OpenAI-compat response")?;
        Ok((status, text))
    }
}

#[async_trait]
impl Engine for OpenAICompatEngine {
    async fn complete(&self, req: &Request) -> Result<Response> {
        let mut caps = self.caps.lock().expect("caps").clone();
        let mut last_err: Option<(u16, String)> = None;

        for _ in 0..MAX_DEGRADES {
            let messages = to_openai_messages(req, caps.fold_system, caps.native_history);
            let max_tokens = req.max_tokens.min(caps.max_tokens);
            let mut payload = json!({
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
            });
            if caps.send_sampling {
                payload["temperature"] = json!(req.temperature);
            }
            if !req.tools.is_empty() {
                payload["tools"] = Value::Array(req.tools.iter().map(to_openai_tool).collect());
            }

            let (status, text) = self.post(&payload).await?;
            if (200..300).contains(&status) {
                *self.caps.lock().expect("caps") = caps;
                return parse_response(&text);
            }

            last_err = Some((status, text.clone()));
            let body = text.to_ascii_lowercase();
            match diagnose(status, &body, &caps, &messages) {
                Some(Action::Shrink {
                    max_tokens,
                    hard_cap,
                    context_window,
                }) => {
                    caps.max_tokens = max_tokens;
                    if hard_cap {
                        caps.hard_output_cap = Some(match caps.hard_output_cap {
                            Some(old) => old.min(max_tokens),
                            None => max_tokens,
                        });
                    }
                    if let Some(window) = context_window {
                        if caps.context_window.map(|w| window < w).unwrap_or(true) {
                            caps.context_window = Some(window);
                        }
                    }
                    caps.output_shrinks += 1;
                }
                Some(Action::BoundWindow { context_window }) => {
                    if caps
                        .context_window
                        .map(|w| context_window < w)
                        .unwrap_or(true)
                    {
                        caps.context_window = Some(context_window);
                    }
                    *self.caps.lock().expect("caps") = caps;
                    return Err(EngineError::Status {
                        provider: PROVIDER_LABEL,
                        status,
                        body: text,
                    }
                    .into());
                }
                Some(Action::FlattenTools) => caps.native_history = false,
                Some(Action::DropSampling) => caps.send_sampling = false,
                Some(Action::FoldSystem) => caps.fold_system = true,
                None => {
                    *self.caps.lock().expect("caps") = caps;
                    return Err(EngineError::Status {
                        provider: PROVIDER_LABEL,
                        status,
                        body: text,
                    }
                    .into());
                }
            }
        }

        *self.caps.lock().expect("caps") = caps;
        let (status, body) = last_err.unwrap_or((400, "degrade loop exhausted".into()));
        Err(EngineError::Status {
            provider: PROVIDER_LABEL,
            status,
            body,
        }
        .into())
    }

    async fn complete_stream(
        &self,
        req: &Request,
        on_delta: &(dyn Fn(StreamDelta) + Send + Sync),
    ) -> Result<Response> {
        let stream_ok = self.caps.lock().expect("caps").stream;
        if !stream_ok {
            return self.complete(req).await;
        }

        let caps = self.caps.lock().expect("caps").clone();
        let messages = to_openai_messages(req, caps.fold_system, caps.native_history);
        let max_tokens = req.max_tokens.min(caps.max_tokens);
        let mut payload = json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": true,
        });
        if caps.send_sampling {
            payload["temperature"] = json!(req.temperature);
        }
        if !req.tools.is_empty() {
            payload["tools"] = Value::Array(req.tools.iter().map(to_openai_tool).collect());
        }

        let url = format!("{}/chat/completions", self.base_url);
        let mut http = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .json(&payload);
        if let Some(key) = &self.api_key {
            http = http.bearer_auth(key);
        }
        let mut resp = http.send().await.map_err(|e| EngineError::Transport {
            provider: PROVIDER_LABEL,
            detail: format!("stream request failed: {e}"),
        })?;
        let status = resp.status().as_u16();
        if !(200..300).contains(&status) {
            let body = resp
                .text()
                .await
                .context("reading OpenAI-compat stream refusal")?;
            // Only request-shape failures justify spending another request on
            // the non-stream path. Retrying a 401/402/429/5xx here doubles key,
            // quota, and outage traffic before `Resilient` even sees it.
            if matches!(status, 400 | 413 | 422) {
                self.caps.lock().expect("caps").stream = false;
                return self.complete(req).await;
            }
            return Err(EngineError::Status {
                provider: PROVIDER_LABEL,
                status,
                body,
            }
            .into());
        }

        let mut assembler = crate::engine::sse::Assembler::new();
        let mut buf = String::new();
        loop {
            let chunk = resp.chunk().await.map_err(|e| EngineError::Transport {
                provider: PROVIDER_LABEL,
                detail: format!("stream body: {e}"),
            })?;
            let Some(bytes) = chunk else {
                break;
            };
            buf.push_str(&String::from_utf8_lossy(&bytes));
            sse::drain_frames(&mut buf, &mut assembler, on_delta);
        }
        if !buf.trim().is_empty() {
            buf.push_str("\n\n");
            sse::drain_frames(&mut buf, &mut assembler, on_delta);
        }
        Ok(assembler.finish())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports_native_tools(&self) -> bool {
        true
    }

    fn restore_limits(&self) -> bool {
        let mut caps = self.caps.lock().expect("caps");
        let mut ceiling = self.configured_max_tokens;
        if let Some(hard) = caps.hard_output_cap {
            ceiling = ceiling.min(hard);
        }
        if caps.max_tokens >= ceiling {
            return false;
        }
        caps.max_tokens = ceiling;
        caps.limit_restores += 1;
        true
    }

    fn context_window(&self) -> Option<u32> {
        self.caps.lock().expect("caps").context_window
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Action {
    Shrink {
        max_tokens: u32,
        hard_cap: bool,
        context_window: Option<u32>,
    },
    /// Record the server's window and give up: the prompt itself does not
    /// fit, so shrinking the reply cannot help.
    BoundWindow {
        context_window: u32,
    },
    FlattenTools,
    DropSampling,
    FoldSystem,
}

fn diagnose(status: u16, body: &str, caps: &Caps, messages: &[Value]) -> Option<Action> {
    if let Some(action) = shrink_to_token_limit(status, body, caps.max_tokens) {
        return Some(action);
    }
    if !matches!(status, 400 | 422) {
        return None;
    }
    if caps.native_history && mentions(body, NO_TOOL_MESSAGES) && has_tool_shape(messages) {
        return Some(Action::FlattenTools);
    }
    if caps.send_sampling && mentions(body, NO_SAMPLING) {
        return Some(Action::DropSampling);
    }
    if !caps.fold_system && mentions(body, NO_SYSTEM) {
        return Some(Action::FoldSystem);
    }
    None
}

const NO_SYSTEM: &[&str] = &[
    "system role not supported",
    "does not support system",
    "system messages are not",
    "only user and assistant",
    "system instruction",
];

const NO_TOOL_MESSAGES: &[&str] = &[
    "tool_calls",
    "tool_call_id",
    "role 'tool'",
    "role \"tool\"",
    "invalid role",
    "unsupported role",
    "tool messages",
    "tool role",
    "thought_signature",
    "thought signature",
];

const NO_SAMPLING: &[&str] = &["temperature", "top_p", "top_k", "sampling"];

fn mentions(body: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| body.contains(n))
}

fn has_tool_shape(messages: &[Value]) -> bool {
    messages.iter().any(|m| {
        m.get("role").and_then(Value::as_str) == Some("tool") || m.get("tool_calls").is_some()
    })
}

/// Adapt to a provider's token ceiling. `None` if the 400/413 is about
/// something else and must not be retried unchanged.
fn shrink_to_token_limit(status: u16, body: &str, current: u32) -> Option<Action> {
    if matches!(status, 400 | 422) {
        if let Some(allowed) = parse_output_cap(body) {
            if allowed < MIN_OUTPUT_TOKENS || allowed >= current {
                return None;
            }
            return Some(Action::Shrink {
                max_tokens: allowed,
                hard_cap: true,
                context_window: None,
            });
        }
    }
    if status != 413 || !body.contains("token") {
        return None;
    }
    let allowance = parse_after(body, "limit")?;
    let room = if let Some(requested) = parse_after(body, "requested") {
        let prompt_tokens = requested.saturating_sub(current);
        if prompt_tokens + MIN_OUTPUT_TOKENS >= allowance {
            return Some(Action::BoundWindow {
                context_window: allowance,
            });
        }
        allowance
            .saturating_sub(prompt_tokens)
            .saturating_sub(LIMIT_MARGIN)
    } else {
        current / 2
    };

    let new_max = MIN_OUTPUT_TOKENS.max(room.min(current));
    if new_max >= current {
        return Some(Action::BoundWindow {
            context_window: allowance,
        });
    }
    Some(Action::Shrink {
        max_tokens: new_max,
        hard_cap: false,
        context_window: Some(allowance),
    })
}

fn parse_output_cap(body: &str) -> Option<u32> {
    // `max_tokens` must be less than or equal to `16384`
    let re = regex::Regex::new(r"max_tokens.{0,120}?less than or equal to\D{0,8}(\d+)")
        .expect("static pattern");
    re.captures(body)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse().ok())
}

fn parse_after(body: &str, label: &str) -> Option<u32> {
    let re = regex::Regex::new(&format!(r"{label}\s+(\d+)")).expect("static pattern");
    re.captures(body)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse().ok())
}

fn to_openai_tool(t: &crate::engine::ToolDef) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.input_schema,
        }
    })
}

fn to_openai_messages(req: &Request, fold_system: bool, native_history: bool) -> Vec<Value> {
    let mut out = Vec::new();
    if !req.system.is_empty() && !fold_system {
        out.push(json!({"role": "system", "content": req.system}));
    }
    for message in &req.messages {
        out.extend(message_to_openai(message));
    }
    if fold_system && !req.system.is_empty() {
        out = fold_system_into(&out, &req.system);
    }
    if !native_history {
        out = flatten_tool_messages(&out);
    }
    out
}

fn message_to_openai(m: &Message) -> Vec<Value> {
    match m.role {
        Role::User => {
            let mut texts = Vec::new();
            let mut tools = Vec::new();
            for c in &m.content {
                match c {
                    Content::Text { text } => texts.push(text.clone()),
                    Content::ToolResult { id, content, .. } => {
                        tools.push(json!({
                            "role": "tool",
                            "tool_call_id": id,
                            "content": content,
                        }));
                    }
                    Content::ToolUse { .. } => {}
                }
            }
            let mut out = Vec::new();
            if !texts.is_empty() {
                out.push(json!({"role": "user", "content": texts.join("\n")}));
            }
            out.extend(tools);
            if out.is_empty() {
                out.push(json!({"role": "user", "content": ""}));
            }
            out
        }
        Role::Assistant => {
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
                    Content::ToolUse { id, name, input } => {
                        tool_calls.push(json!({
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": input.to_string(),
                            }
                        }));
                    }
                    Content::ToolResult { .. } => {}
                }
            }
            let mut msg = json!({"role": "assistant", "content": text});
            if !tool_calls.is_empty() {
                msg["tool_calls"] = Value::Array(tool_calls);
            }
            vec![msg]
        }
    }
}

/// Render native `tool_calls` / `tool` messages back into plain text.
///
/// For providers whose OpenAI-compatible surface does not accept them.
/// Adjacent tool results merge into one user turn, matching the prompted
/// path the rest of the harness already speaks.
fn flatten_tool_messages(messages: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    let mut pending: Vec<String> = Vec::new();

    let flush = |pending: &mut Vec<String>, out: &mut Vec<Value>| {
        if pending.is_empty() {
            return;
        }
        out.push(json!({"role": "user", "content": pending.join("\n")}));
        pending.clear();
    };

    for message in messages {
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");
        if role == "tool" {
            if pending.is_empty() {
                pending.push("## Tool results".into());
            }
            let name = message
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("tool");
            let content = message.get("content").and_then(Value::as_str).unwrap_or("");
            pending.push(format!("\n### {name}\n{content}"));
            continue;
        }
        flush(&mut pending, &mut out);
        if role == "assistant" && message.get("tool_calls").is_some() {
            let mut parts = vec![message
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()];
            if let Some(Value::Array(calls)) = message.get("tool_calls") {
                for call in calls {
                    let function = call.get("function").cloned().unwrap_or(json!({}));
                    let name = function.get("name").and_then(Value::as_str).unwrap_or("");
                    let args = function
                        .get("arguments")
                        .and_then(Value::as_str)
                        .and_then(|s| serde_json::from_str::<Value>(s).ok())
                        .unwrap_or_else(|| json!({}));
                    parts.push(format!(
                        "\n```json\n{}\n```\n",
                        json!({"tool": name, "input": args})
                    ));
                }
            }
            out.push(json!({
                "role": "assistant",
                "content": parts.join("").trim(),
            }));
            continue;
        }
        out.push(message.clone());
    }
    flush(&mut pending, &mut out);
    out
}

fn fold_system_into(messages: &[Value], system: &str) -> Vec<Value> {
    let mut out = Vec::new();
    let mut carried = system.to_string();
    for message in messages {
        if message.get("role").and_then(Value::as_str) == Some("system") {
            let extra = message.get("content").and_then(Value::as_str).unwrap_or("");
            if !extra.is_empty() {
                if !carried.is_empty() {
                    carried.push_str("\n\n");
                }
                carried.push_str(extra);
            }
            continue;
        }
        if !carried.is_empty() && message.get("role").and_then(Value::as_str) == Some("user") {
            let content = message.get("content").and_then(Value::as_str).unwrap_or("");
            out.push(json!({"role": "user", "content": format!("{carried}\n\n{content}")}));
            carried.clear();
            continue;
        }
        out.push(message.clone());
    }
    if !carried.is_empty() {
        out.insert(0, json!({"role": "user", "content": carried}));
    }
    out
}

fn parse_response(text: &str) -> Result<Response> {
    let wire: WireResponse = serde_json::from_str(text)
        .with_context(|| format!("decoding OpenAI-compat response: {text}"))?;
    let choice = wire
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("OpenAI-compat response had no choices"))?;
    let mut content = Vec::new();
    if let Some(text) = choice.message.content {
        if !text.is_empty() {
            content.push(Content::text(text));
        }
    }
    for call in choice.message.tool_calls {
        let input = serde_json::from_str(&call.function.arguments).unwrap_or_else(|_| json!({}));
        content.push(Content::ToolUse {
            id: call.id,
            name: call.function.name,
            input,
        });
    }
    Ok(Response {
        content,
        stop_reason: match choice.finish_reason.as_deref() {
            Some("stop") | None => StopReason::EndTurn,
            Some("tool_calls") | Some("function_call") => StopReason::ToolUse,
            Some("length") => StopReason::MaxTokens,
            Some(other) => StopReason::Other(other.to_string()),
        },
        usage: Usage {
            input_tokens: wire.usage.prompt_tokens,
            output_tokens: wire.usage.completion_tokens,
        },
    })
}

#[derive(Deserialize)]
struct WireResponse {
    choices: Vec<WireChoice>,
    #[serde(default)]
    usage: WireUsage,
}

#[derive(Deserialize)]
struct WireChoice {
    message: WireOutMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct WireOutMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<WireToolCall>,
}

#[derive(Deserialize)]
struct WireToolCall {
    id: String,
    function: WireFn,
}

#[derive(Deserialize)]
struct WireFn {
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Deserialize, Default)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

/// Resolve a homelab host variable the way Python `_local_base_url` does.
fn local_base_url(provider: &str) -> Option<String> {
    let (var, port, suffix) = match provider {
        "ollama" => ("OLLAMA_HOST", 11434, "/v1"),
        "openwebui" => ("OPENWEBUI_HOST", 3000, "/api"),
        "local" => ("LOCAL_LLM_HOST", 8000, "/v1"),
        _ => return None,
    };
    let raw = std::env::var(var).ok()?;
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let mut url = if raw.contains("://") {
        raw.to_string()
    } else {
        format!("http://{raw}")
    };
    url = url.trim_end_matches('/').to_string();
    if url.matches(':').count() < 2 {
        url = format!("{url}:{port}");
    }
    if url.ends_with(suffix) {
        Some(url)
    } else {
        Some(format!("{url}{suffix}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GROQ_413: &str = "request too large for model `x` in organization `y` \
         service tier `on_demand` on tokens per minute (tpm): limit 8000, \
         requested 8263, please reduce your message size and try again";

    const GROQ_400_OUTPUT_CAP: &str = "`max_tokens` must be less than or equal to `16384`, \
         the maximum value for `max_tokens` is less than the `context_window` \
         for this model";

    fn caps(max_tokens: u32) -> Caps {
        Caps {
            max_tokens,
            hard_output_cap: None,
            context_window: None,
            send_sampling: true,
            native_history: true,
            fold_system: false,
            output_shrinks: 0,
            limit_restores: 0,
            stream: true,
        }
    }

    #[test]
    fn a_token_limit_shrinks_the_reply_allowance() {
        let action = shrink_to_token_limit(413, GROQ_413, 8192).unwrap();
        match action {
            Action::Shrink {
                max_tokens,
                hard_cap,
                context_window,
            } => {
                assert!(!hard_cap);
                assert_eq!(context_window, Some(8000));
                // 8263 requested - 8192 asked for = 71 prompt tokens.
                assert_eq!(max_tokens, 8000 - 71 - LIMIT_MARGIN);
            }
            other => panic!("expected shrink, got {other:?}"),
        }
    }

    #[test]
    fn a_per_model_reply_cap_is_clamped_not_reported() {
        let action = shrink_to_token_limit(400, GROQ_400_OUTPUT_CAP, 32_000).unwrap();
        match action {
            Action::Shrink {
                max_tokens,
                hard_cap,
                ..
            } => {
                assert!(hard_cap);
                assert_eq!(max_tokens, 16384);
            }
            other => panic!("expected shrink, got {other:?}"),
        }
    }

    #[test]
    fn a_reply_cap_we_are_already_under_is_not_a_shrink() {
        assert!(shrink_to_token_limit(400, GROQ_400_OUTPUT_CAP, 8192).is_none());
    }

    #[test]
    fn a_tool_schema_400_is_left_to_the_native_history_path() {
        let body = "invalid value for 'tools[0].function.parameters': expected object";
        assert!(shrink_to_token_limit(400, body, 32_000).is_none());
    }

    #[test]
    fn a_prompt_that_does_not_fit_binds_the_window_and_does_not_shrink() {
        // 8000 limit, 9000 requested, current max_tokens 512: prompt is 8488.
        let body = "limit 8000, requested 9000 tokens per minute";
        match shrink_to_token_limit(413, body, 512) {
            Some(Action::BoundWindow { context_window }) => {
                assert_eq!(context_window, 8000);
            }
            other => panic!("expected BoundWindow, got {other:?}"),
        }
    }

    #[test]
    fn a_rate_limit_is_never_read_as_a_reply_cap() {
        assert!(shrink_to_token_limit(
            429,
            "max_tokens must be less than or equal to 16384",
            32_000
        )
        .is_none());
    }

    #[test]
    fn a_sampling_refusal_beats_a_generic_unknown_field() {
        let c = caps(8192);
        let action = diagnose(400, "unknown field `temperature`", &c, &[]);
        assert_eq!(action, Some(Action::DropSampling));
    }

    #[test]
    fn a_system_role_refusal_folds() {
        let c = caps(8192);
        let action = diagnose(400, "system role not supported", &c, &[]);
        assert_eq!(action, Some(Action::FoldSystem));
    }

    #[test]
    fn a_tool_history_refusal_requires_tool_shaped_messages() {
        let c = caps(8192);
        assert!(diagnose(400, "invalid role 'tool'", &c, &[]).is_none());
        let msgs = vec![json!({"role": "tool", "content": "ok"})];
        assert_eq!(
            diagnose(400, "invalid role 'tool'", &c, &msgs),
            Some(Action::FlattenTools)
        );
    }

    #[test]
    fn flatten_renders_tool_calls_as_fenced_json() {
        let msgs = vec![json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{\"path\":\"a.rs\"}"}
            }]
        })];
        let flat = flatten_tool_messages(&msgs);
        let text = flat[0]["content"].as_str().unwrap();
        assert!(text.contains("read_file"), "{text}");
        assert!(text.contains("```json"), "{text}");
        assert!(text.contains("\"input\""), "emit input not args: {text}");
        assert!(!text.contains("\"args\""), "do not emit args: {text}");
        assert!(!flat[0].get("tool_calls").is_some());
    }

    #[test]
    fn fold_merges_system_into_the_first_user_turn() {
        let folded = fold_system_into(
            &[json!({"role": "user", "content": "hello"})],
            "CONSTITUTION",
        );
        assert_eq!(folded.len(), 1);
        assert_eq!(folded[0]["role"], "user");
        assert!(folded[0]["content"]
            .as_str()
            .unwrap()
            .starts_with("CONSTITUTION"));
        assert!(folded[0]["content"].as_str().unwrap().contains("hello"));
    }

    #[test]
    fn restore_limits_undoes_a_transient_shrink_but_not_a_hard_cap() {
        let eng = OpenAICompatEngine::new("groq", "x", "http://example.invalid", None, 8192);
        {
            let mut c = eng.caps.lock().unwrap();
            c.max_tokens = 2000;
        }
        assert!(eng.restore_limits());
        assert_eq!(eng.caps.lock().unwrap().max_tokens, 8192);

        {
            let mut c = eng.caps.lock().unwrap();
            c.max_tokens = 2000;
            c.hard_output_cap = Some(4096);
        }
        assert!(eng.restore_limits());
        assert_eq!(eng.caps.lock().unwrap().max_tokens, 4096);
        assert!(!eng.restore_limits());
    }

    #[test]
    fn every_named_provider_is_in_the_table() {
        for name in [
            "groq",
            "gemini",
            "openrouter",
            "qwen",
            "fireworks",
            "nvidia",
            "cerebras",
            "mistral",
            "together",
            "ollama",
            "local",
        ] {
            assert!(provider(name).is_some(), "missing {name}");
        }
    }

    #[test]
    fn parse_response_reads_tool_calls_and_usage() {
        let raw = r#"{
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{\"path\":\"a.rs\"}"}
                    }]
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4}
        }"#;
        let resp = parse_response(raw).unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.tool_uses().len(), 1);
        assert_eq!(resp.tool_uses()[0].1, "read_file");
    }
}
