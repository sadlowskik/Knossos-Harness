//! Backend-neutral conversation types.
//!
//! Nothing in here is shaped by a particular provider's wire format. Each
//! backend owns its own private wire structs and converts at the boundary,
//! which is what makes the engine slot a real seam rather than a nominal one:
//! Metis, Talos and Oracle never touch a provider-specific field.
//!
//! The `serde` derives exist for the JSONL trace log (`session.rs`), not for
//! any HTTP body.

use serde::{Deserialize, Serialize};

/// One piece of a reply, delivered while the engine is still generating.
///
/// Tool calls are **not** a delta. They are assembled off-stream and appear
/// only on the finished [`Response`], so a truncated JSON argument cannot
/// become a dispatched tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamDelta {
    Text(String),
    Thought(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// One block inside a message. A single assistant turn can mix prose with
/// several tool calls, so content is always a list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Content {
    Text {
        text: String,
    },
    /// The engine asking for a tool to be run.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// The harness answering a `ToolUse`. Carried in a `User` message.
    ToolResult {
        id: String,
        content: String,
        is_error: bool,
    },
}

impl Content {
    pub fn text(s: impl Into<String>) -> Self {
        Content::Text { text: s.into() }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Content::Text { text } => Some(text),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
}

impl Message {
    pub fn user(content: Vec<Content>) -> Self {
        Message {
            role: Role::User,
            content,
        }
    }

    pub fn assistant(content: Vec<Content>) -> Self {
        Message {
            role: Role::Assistant,
            content,
        }
    }

    pub fn user_text(s: impl Into<String>) -> Self {
        Message::user(vec![Content::text(s)])
    }

    /// All text blocks joined — prose only, tool calls excluded.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(Content::as_text)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// A tool the engine is allowed to call. `input_schema` is JSON Schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    /// Themis renders the constitution into this on every single call.
    pub system: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDef>,
    pub max_tokens: u32,
    pub temperature: f32,
}

impl Request {
    pub fn new(system: impl Into<String>, messages: Vec<Message>) -> Self {
        Request {
            system: system.into(),
            messages,
            tools: Vec::new(),
            max_tokens: 4096,
            temperature: 0.0,
        }
    }

    pub fn with_tools(mut self, tools: Vec<ToolDef>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The engine finished its turn with nothing outstanding.
    EndTurn,
    /// The engine wants tools run before it continues.
    ToolUse,
    /// Output was truncated by the token cap.
    MaxTokens,
    Other(String),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub content: Vec<Content>,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

impl Response {
    /// Prose only. Tool calls are deliberately excluded — callers that want
    /// them should ask for `tool_uses()` so the two never get confused.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(Content::as_text)
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn tool_uses(&self) -> Vec<(&str, &str, &serde_json::Value)> {
        self.content
            .iter()
            .filter_map(|c| match c {
                Content::ToolUse { id, name, input } => Some((id.as_str(), name.as_str(), input)),
                _ => None,
            })
            .collect()
    }

    pub fn wants_tools(&self) -> bool {
        self.content
            .iter()
            .any(|c| matches!(c, Content::ToolUse { .. }))
    }

    /// Turn this response into the assistant message that goes back into
    /// history on the next call.
    pub fn as_message(&self) -> Message {
        Message::assistant(self.content.clone())
    }
}
