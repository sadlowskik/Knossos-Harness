//! `ask_user`: a structured question, when the client can elicit.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use super::{Tool, ToolCtx, ToolOutput};

pub struct AskUser;

#[async_trait]
impl Tool for AskUser {
    fn name(&self) -> &str {
        "ask_user"
    }

    fn description(&self) -> &str {
        "Ask the person using the editor a question and wait for their answer. \
         Use when you need a choice you cannot infer from the repository."
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "choices": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["question"]
        })
    }

    fn consequential(&self) -> bool {
        false
    }

    async fn run(&self, input: &serde_json::Value, ctx: &ToolCtx) -> Result<ToolOutput> {
        let question = match input.get("question").and_then(|v| v.as_str()) {
            Some(q) if !q.trim().is_empty() => q,
            _ => return Ok(ToolOutput::error("ask_user needs a question")),
        };
        let choices: Vec<String> = input
            .get("choices")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        match ctx.ask_user(question, &choices) {
            Some(answer) => Ok(ToolOutput::ok(answer)),
            None => Ok(ToolOutput::error(
                "nobody to ask: the client has no elicitation, the user declined, or the turn was cancelled",
            )),
        }
    }
}
