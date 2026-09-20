//! OpenAI-shaped SSE (`text/event-stream`) assembly.
//!
//! Isolated so the parser can be tested with a byte string and no network.
//! Tool-call arguments are concatenated off-stream and only become a
//! [`Content::ToolUse`](crate::engine::Content::ToolUse) when [`Assembler::finish`]
//! runs — a truncated `arguments` fragment is never a call.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

use crate::engine::types::{Content, Response, StopReason, StreamDelta, Usage};

#[derive(Default)]
struct PartialTool {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Default)]
pub struct Assembler {
    text: String,
    tools: BTreeMap<usize, PartialTool>,
    finish_reason: Option<String>,
    usage: Usage,
}

impl Assembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one SSE `data:` payload (the JSON, or `[DONE]`).
    pub fn ingest(&mut self, data: &str, on_delta: &(dyn Fn(StreamDelta) + Send + Sync)) {
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            return;
        }
        let Ok(chunk) = serde_json::from_str::<WireChunk>(data) else {
            return;
        };
        if let Some(u) = chunk.usage {
            if u.prompt_tokens > 0 {
                self.usage.input_tokens = u.prompt_tokens;
            }
            if u.completion_tokens > 0 {
                self.usage.output_tokens = u.completion_tokens;
            }
        }
        let Some(choice) = chunk.choices.into_iter().next() else {
            return;
        };
        if choice.finish_reason.is_some() {
            self.finish_reason = choice.finish_reason;
        }
        let delta = choice.delta;
        if let Some(piece) = delta.content {
            if !piece.is_empty() {
                self.text.push_str(&piece);
                on_delta(StreamDelta::Text(piece));
            }
        }
        // OpenAI-compat reasoning models (DeepSeek, etc.)
        if let Some(piece) = delta.reasoning_content.or(delta.reasoning) {
            if !piece.is_empty() {
                on_delta(StreamDelta::Thought(piece));
            }
        }
        for call in delta.tool_calls {
            let slot = self.tools.entry(call.index).or_default();
            if let Some(id) = call.id {
                slot.id = id;
            }
            if let Some(function) = call.function {
                if let Some(name) = function.name {
                    slot.name.push_str(&name);
                }
                if let Some(args) = function.arguments {
                    slot.arguments.push_str(&args);
                }
            }
        }
    }

    pub fn finish(self) -> Response {
        let mut content = Vec::new();
        if !self.text.is_empty() {
            content.push(Content::text(self.text));
        }
        for (_, tool) in self.tools {
            if tool.name.is_empty() {
                continue;
            }
            let input = serde_json::from_str(&tool.arguments)
                .unwrap_or_else(|_| Value::Object(Default::default()));
            content.push(Content::ToolUse {
                id: if tool.id.is_empty() {
                    format!("call-{}", tool.name)
                } else {
                    tool.id
                },
                name: tool.name,
                input,
            });
        }
        let stop_reason = match self.finish_reason.as_deref() {
            Some("stop") | None if content.iter().any(|c| matches!(c, Content::ToolUse { .. })) => {
                StopReason::ToolUse
            }
            Some("stop") | None => StopReason::EndTurn,
            Some("tool_calls") | Some("function_call") => StopReason::ToolUse,
            Some("length") => StopReason::MaxTokens,
            Some(other) => StopReason::Other(other.to_string()),
        };
        Response {
            content,
            stop_reason,
            usage: self.usage,
        }
    }
}

/// Split an SSE body into `data:` payloads and assemble a response.
pub fn parse_body(body: &str, on_delta: &(dyn Fn(StreamDelta) + Send + Sync)) -> Response {
    let mut assembler = Assembler::new();
    let mut data = String::new();
    for line in body.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.trim_start());
            continue;
        }
        if line.is_empty() && !data.is_empty() {
            assembler.ingest(&data, on_delta);
            data.clear();
        }
    }
    if !data.is_empty() {
        assembler.ingest(&data, on_delta);
    }
    assembler.finish()
}

/// Pull complete SSE frames (`\n\n` or `\r\n\r\n`) out of `buf`.
pub fn drain_frames(
    buf: &mut String,
    assembler: &mut Assembler,
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

#[derive(Deserialize)]
struct WireChunk {
    #[serde(default)]
    choices: Vec<WireChoice>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct WireChoice {
    #[serde(default)]
    delta: WireDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct WireDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Vec<WireToolDelta>,
}

#[derive(Deserialize, Default)]
struct WireToolDelta {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<WireFnDelta>,
}

#[derive(Deserialize, Default)]
struct WireFnDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Deserialize, Default)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn collect(body: &str) -> (Response, Vec<StreamDelta>) {
        let seen = Mutex::new(Vec::new());
        let resp = parse_body(body, &|d| seen.lock().unwrap().push(d));
        (resp, seen.into_inner().unwrap())
    }

    #[test]
    fn text_deltas_join_into_the_final_response() {
        let body = "\
data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\
\n\
data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\
\n\
data: [DONE]\n\
";
        let (resp, deltas) = collect(body);
        assert_eq!(resp.text(), "Hello");
        assert_eq!(
            deltas,
            vec![
                StreamDelta::Text("Hel".into()),
                StreamDelta::Text("lo".into())
            ]
        );
        assert!(resp.tool_uses().is_empty());
    }

    #[test]
    fn tool_calls_assemble_only_at_the_end() {
        let body = "\
data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"1\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"pa\"}}]}}]}\n\
\n\
data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"th\\\":\\\"a.rs\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\
\n\
data: [DONE]\n\
";
        let (resp, deltas) = collect(body);
        assert!(
            deltas.is_empty(),
            "partial JSON must not become text: {deltas:?}"
        );
        let uses = resp.tool_uses();
        assert_eq!(uses.len(), 1);
        assert_eq!(uses[0].1, "read_file");
        assert_eq!(uses[0].2["path"], "a.rs");
    }

    #[test]
    fn reasoning_is_a_thought_not_the_answer() {
        let body = "\
data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hmm\"}}]}\n\
\n\
data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\
\n\
";
        let (resp, deltas) = collect(body);
        assert_eq!(resp.text(), "ok");
        assert_eq!(
            deltas,
            vec![
                StreamDelta::Thought("hmm".into()),
                StreamDelta::Text("ok".into())
            ]
        );
    }
}
