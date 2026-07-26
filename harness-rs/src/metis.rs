//! Metis: planning.
//!
//! The plan is a separate artifact from its execution. That is the whole point
//! of splitting Metis from Talos — a plan you can print, log and disagree with
//! before any file is touched is worth more than an intention buried in a
//! model's first turn.
//!
//! The plan is requested through a tool call rather than parsed out of prose,
//! so its structure is enforced by the engine's schema validation. Local
//! models being what they are, a prose fallback exists — but it is a fallback,
//! not the design.

use anyhow::Result;
use serde_json::json;

use crate::engine::{self, Engine, Message, Request, ToolDef};
use crate::scribe::SymbolIndex;
use crate::themis::{Themis, PLANNER_ROLE};

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Plan {
    pub steps: Vec<String>,
}

impl Plan {
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Render for injection into Talos's opening message.
    pub fn render(&self) -> String {
        self.steps
            .iter()
            .enumerate()
            .map(|(i, s)| format!("{}. {}", i + 1, s))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn plan_tool() -> ToolDef {
    ToolDef {
        name: "submit_plan".to_string(),
        description: "Submit the ordered steps for this task. Between 1 and 8 steps, each a \
                      concrete action on the codebase."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered, concrete steps"
                }
            },
            "required": ["steps"]
        }),
    }
}

pub async fn plan(
    eng: &dyn Engine,
    themis: &Themis,
    scribe: &SymbolIndex,
    task: &str,
    max_tokens: u32,
) -> Result<Plan> {
    let system = themis.system_prompt(PLANNER_ROLE, Some(scribe));
    let req = Request::new(
        system,
        vec![Message::user_text(format!(
            "Task: {task}\n\nProduce a short plan by calling submit_plan. Keep it to the \
             smallest number of steps that actually completes the task."
        ))],
    )
    .with_tools(vec![plan_tool()])
    .with_max_tokens(max_tokens);

    let resp = engine::complete(eng, &req).await?;

    for (_, name, input) in resp.tool_uses() {
        if name != "submit_plan" {
            continue;
        }
        if let Some(steps) = input.get("steps").and_then(|v| v.as_array()) {
            let steps: Vec<String> = steps
                .iter()
                .filter_map(|s| s.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if !steps.is_empty() {
                return Ok(Plan { steps });
            }
        }
    }

    // Fallback: the engine described a plan instead of submitting one.
    let steps = parse_prose_plan(&resp.text());
    if steps.is_empty() {
        // A single-step plan is still a plan; the task itself is the step.
        return Ok(Plan { steps: vec![task.to_string()] });
    }
    Ok(Plan { steps })
}

/// Pull numbered or bulleted lines out of prose.
fn parse_prose_plan(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }

        let stripped = t
            .strip_prefix("- ")
            .or_else(|| t.strip_prefix("* "))
            .map(str::to_string)
            .or_else(|| {
                // "1. step" / "2) step"
                let mut chars = t.char_indices();
                let digits: String = chars
                    .by_ref()
                    .take_while(|(_, c)| c.is_ascii_digit())
                    .map(|(_, c)| c)
                    .collect();
                if digits.is_empty() {
                    return None;
                }
                let rest = t[digits.len()..].trim_start();
                rest.strip_prefix('.')
                    .or_else(|| rest.strip_prefix(')'))
                    .map(|r| r.trim().to_string())
            });

        if let Some(s) = stripped {
            if !s.is_empty() {
                out.push(s);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mock::{text_response, MockEngine};
    use crate::engine::{Content, Response, StopReason, Usage};

    fn scribe() -> SymbolIndex {
        let dir = tempfile::tempdir().unwrap();
        SymbolIndex::build(dir.path()).unwrap()
    }

    #[tokio::test]
    async fn uses_the_submitted_plan_when_the_tool_is_called() {
        let eng = MockEngine::new(vec![Response {
            content: vec![Content::ToolUse {
                id: "1".into(),
                name: "submit_plan".into(),
                input: json!({"steps": ["read lib.rs", "add the flag", "run cargo check"]}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        }]);

        let p = plan(&eng, &Themis::from_text("x"), &scribe(), "add a flag", 1024)
            .await
            .unwrap();
        assert_eq!(p.steps.len(), 3);
        assert_eq!(p.steps[0], "read lib.rs");
        assert!(p.render().starts_with("1. read lib.rs"));
    }

    #[tokio::test]
    async fn falls_back_to_parsing_prose() {
        let eng = MockEngine::new(vec![text_response(
            "Here is my plan:\n1. Read the file\n2. Make the edit\n- Verify with cargo",
        )]);
        let p = plan(&eng, &Themis::from_text("x"), &scribe(), "task", 1024)
            .await
            .unwrap();
        assert_eq!(p.steps, vec!["Read the file", "Make the edit", "Verify with cargo"]);
    }

    #[tokio::test]
    async fn an_unparseable_reply_still_yields_a_usable_plan() {
        let eng = MockEngine::new(vec![text_response("I'm not sure what to do.")]);
        let p = plan(&eng, &Themis::from_text("x"), &scribe(), "add a flag", 1024)
            .await
            .unwrap();
        assert_eq!(p.steps, vec!["add a flag"]);
        assert!(!p.is_empty());
    }

    #[tokio::test]
    async fn the_planner_prompt_carries_the_constitution() {
        let eng = MockEngine::new(vec![text_response("- do it")]);
        plan(&eng, &Themis::from_text("CONSTITUTION MARKER"), &scribe(), "t", 512)
            .await
            .unwrap();
        let seen = eng.seen.lock().unwrap();
        assert!(seen[0].system.contains("CONSTITUTION MARKER"));
        assert!(seen[0].system.contains("Metis"));
    }

    #[test]
    fn prose_parser_handles_both_numbering_styles() {
        let s = parse_prose_plan("1. alpha\n2) beta\n- gamma\n* delta\nnot a step");
        assert_eq!(s, vec!["alpha", "beta", "gamma", "delta"]);
    }
}
