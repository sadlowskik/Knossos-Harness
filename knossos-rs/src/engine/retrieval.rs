//! No model. Reports that retrieval is all this process can do.
//!
//! The ACP server must still speak when no API key is configured. This engine
//! answers with the retrieved excerpts already in the prompt, and says so.
//! It never claims to have edited a file.

use anyhow::Result;
use async_trait::async_trait;

use crate::engine::types::{Request, Response, StopReason, Usage};
use crate::engine::Engine;

pub struct RetrievalEngine;

#[async_trait]
impl Engine for RetrievalEngine {
    async fn complete(&self, req: &Request) -> Result<Response> {
        let last = req
            .messages
            .iter()
            .rev()
            .find_map(|m| {
                let t = m.text();
                if t.trim().is_empty() {
                    None
                } else {
                    Some(t)
                }
            })
            .unwrap_or_default();
        let text = if last.trim().is_empty() {
            "Nothing in the index matched that. Name a symbol, file, or identifier.".to_string()
        } else {
            format!(
                "No language model is loaded, so I cannot edit the tree. \
                 Retrieved context from the workspace follows.\n\n{last}"
            )
        };
        Ok(Response {
            content: vec![crate::engine::Content::text(text)],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        })
    }

    fn name(&self) -> &str {
        "retrieval"
    }

    fn supports_native_tools(&self) -> bool {
        false
    }
}
