//! `verify`: ask Oracle mid-loop, without claiming the task is done.
//!
//! The engine calling this is *inspecting*. Halt::Done still requires a turn
//! with no tool calls and a passing verdict — a passing `verify` is evidence,
//! not completion. Dispatch is intercepted in `Talos` because Oracle lives
//! there (baseline, dry-run, staged contents). `run` exists so the schema is
//! on the registry; it is not the path a live loop takes.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use super::{Tool, ToolCtx, ToolOutput};

pub const NAME: &str = "verify";

pub struct Verify;

#[async_trait]
impl Tool for Verify {
    fn name(&self) -> &str {
        NAME
    }

    fn description(&self) -> &str {
        "Run the verifier on the current tree. Default is a cheap syntax check. \
         Set full=true for the compiler/test ladder. A passing result is not \
         completion — stop calling tools when the task is actually done."
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "full": {
                    "type": "boolean",
                    "description": "Run the full ladder instead of syntax-only"
                }
            }
        })
    }

    fn consequential(&self) -> bool {
        // Does not change the tree. Counting it as work would let a run
        // "complete" by only asking Oracle, which is the empty-change-set
        // hole the loop already closed for `read_file`.
        false
    }

    async fn run(&self, _input: &serde_json::Value, _ctx: &ToolCtx) -> Result<ToolOutput> {
        // Reached only if a caller dispatches through the registry instead of
        // Talos. A live run never does.
        Ok(ToolOutput::error(
            "verify is applied by the harness; the loop intercepts this tool",
        ))
    }
}

pub fn wants_full(input: &serde_json::Value) -> bool {
    input.get("full").and_then(|v| v.as_bool()).unwrap_or(false)
}
