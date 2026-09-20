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

/// A plan longer than this is a symptom — either the task wants splitting, or
/// the model is narrating rather than planning. Truncated rather than refused,
/// because a too-long plan is still more use than none.
pub const MAX_STEPS: usize = 8;

/// Tasks below this word count rarely gain from a plan, and planning them
/// costs an engine turn that execution could have had.
const TRIVIAL_WORDS: usize = 6;

/// Whether a task is big enough that a plan earns its turn.
///
/// Deliberately crude. The cost of planning a trivial task is one wasted
/// turn; the cost of *not* planning a large one is a flat loop with no
/// structure. So this errs toward planning, and only skips what is
/// obviously atomic.
pub fn worth_planning(task: &str) -> bool {
    task.split_whitespace().count() >= TRIVIAL_WORDS
}

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
        description: format!(
            "Submit the ordered steps for this task. Between 1 and {MAX_STEPS} steps, each a \
             concrete action on the codebase."
        ),
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

    let resp = match engine::complete(eng, &req).await {
        Ok(resp) => resp,
        // A planner must not end a run. The task as its own step keeps every
        // caller on one path — Talos always takes a plan.
        Err(_) => return Ok(degenerate(task)),
    };

    for (_, name, input) in resp.tool_uses() {
        if name != "submit_plan" {
            continue;
        }
        let steps = tidy(&steps_from_value(input));
        if !steps.is_empty() {
            return Ok(Plan { steps });
        }
    }

    let text = resp.text();
    let steps = tidy(&recover_truncated_json(&text));
    if !steps.is_empty() {
        return Ok(Plan { steps });
    }

    // Fallback: the engine described a plan instead of submitting one.
    let steps = tidy(&parse_prose_plan(&text));
    if steps.is_empty() {
        return Ok(degenerate(task));
    }
    Ok(Plan { steps })
}

fn degenerate(task: &str) -> Plan {
    let collapsed = task.split_whitespace().collect::<Vec<_>>().join(" ");
    Plan {
        steps: vec![if collapsed.is_empty() {
            task.to_string()
        } else {
            collapsed
        }],
    }
}

fn steps_from_value(input: &serde_json::Value) -> Vec<String> {
    let raw = input.get("steps").or_else(|| {
        input
            .get("args")
            .or_else(|| input.get("input"))
            .and_then(|inner| inner.get("steps"))
    });
    match raw {
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|s| s.as_str())
            .map(|s| s.to_string())
            .collect(),
        _ => Vec::new(),
    }
}

/// Strip, drop blanks and duplicates, and cap the length.
fn tidy(steps: &[String]) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for step in steps {
        let text = step.split_whitespace().collect::<Vec<_>>().join(" ");
        if text.is_empty() {
            continue;
        }
        let key = text.to_ascii_lowercase();
        if !seen.insert(key) {
            continue;
        }
        out.push(text);
        if out.len() >= MAX_STEPS {
            break;
        }
    }
    out
}

/// Recover a plan whose JSON lost its closing brackets.
///
/// Small models drop the last `}` remarkably often. Guessing at a
/// half-parsed *action* is how the wrong file gets written; a plan is
/// inert text the executor may ignore, so the same caution costs a whole
/// planning turn and buys nothing.
///
/// Narrow on purpose. Only unclosed brackets are appended, in the order
/// that closes them; nothing in the content is altered, and a block that
/// still will not parse is abandoned.
fn recover_truncated_json(reply: &str) -> Vec<String> {
    for body in fence_bodies(reply) {
        if !body.contains("submit_plan") {
            continue;
        }
        let Some(closers) = unclosed_brackets(body) else {
            continue;
        };
        if closers.is_empty() {
            // Complete JSON in a fence: still try to parse it, because a
            // model that wrote a valid `submit_plan` block as text (no
            // native tool call) should not have to fall through to prose.
            if let Some(steps) = parse_plan_json(body) {
                return steps;
            }
            continue;
        }
        let mut repaired = body.to_string();
        repaired.extend(closers.into_iter().rev());
        if let Some(steps) = parse_plan_json(&repaired) {
            return steps;
        }
    }
    Vec::new()
}

fn parse_plan_json(body: &str) -> Option<Vec<String>> {
    let payload: serde_json::Value = serde_json::from_str(body).ok()?;
    let steps = steps_from_value(&payload);
    if steps.is_empty() {
        None
    } else {
        Some(steps)
    }
}

/// Bodies of fenced blocks, including an unterminated fence — a truncated
/// reply may never close it.
fn fence_bodies(reply: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = reply;
    while let Some(start) = rest.find("```") {
        let after_ticks = &rest[start + 3..];
        let after_lang = match after_ticks.find('\n') {
            Some(i) => &after_ticks[i + 1..],
            None => break,
        };
        match after_lang.find("```") {
            Some(end) => {
                out.push(&after_lang[..end]);
                rest = &after_lang[end + 3..];
            }
            None => {
                out.push(after_lang);
                break;
            }
        }
    }
    out
}

/// Closers needed to balance `{`/`[`, or `None` if a string is unclosed
/// (we cannot decide) or the brackets are already balanced / over-closed.
fn unclosed_brackets(body: &str) -> Option<Vec<char>> {
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for c in body.chars() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => match stack.pop() {
                Some(expected) if expected == c => {}
                _ => return None,
            },
            _ => {}
        }
    }
    if in_string {
        return None;
    }
    Some(stack)
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
        assert_eq!(
            p.steps,
            vec!["Read the file", "Make the edit", "Verify with cargo"]
        );
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
        plan(
            &eng,
            &Themis::from_text("CONSTITUTION MARKER"),
            &scribe(),
            "t",
            512,
        )
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

    #[test]
    fn a_plan_missing_its_closing_brace_is_recovered() {
        let reply = "```json\n{\"tool\": \"submit_plan\", \"args\": {\"steps\": [\"read it\", \"fix it\"]}\n```";
        assert_eq!(recover_truncated_json(reply), ["read it", "fix it"]);
    }

    #[test]
    fn a_plan_truncated_mid_array_is_recovered() {
        let reply = "```json\n{\"tool\": \"submit_plan\", \"args\": {\"steps\": [\"read it\"";
        assert_eq!(recover_truncated_json(reply), ["read it"]);
    }

    #[test]
    fn a_brace_inside_a_string_is_not_counted() {
        let reply = "```json\n{\"tool\": \"submit_plan\", \"args\": {\"steps\": [\"handle the { case\"]}\n```";
        assert_eq!(recover_truncated_json(reply), ["handle the { case"]);
    }

    #[test]
    fn genuinely_broken_json_is_not_guessed_at() {
        let reply = "```json\n{\"tool\": \"submit_plan\", \"args\": {\"steps\": [oh dear]}}\n```";
        assert!(recover_truncated_json(reply).is_empty());
    }

    #[test]
    fn a_truncated_block_for_another_tool_is_ignored() {
        let reply = "```json\n{\"tool\": \"write_file\", \"args\": {\"path\": \"a.py\"";
        assert!(recover_truncated_json(reply).is_empty());
    }

    #[test]
    fn blank_and_duplicate_steps_are_dropped() {
        let steps = tidy(&[
            "read it".into(),
            "  ".into(),
            "read it".into(),
            "READ IT".into(),
            "fix it".into(),
        ]);
        assert_eq!(steps, ["read it", "fix it"]);
    }

    #[test]
    fn a_runaway_plan_is_capped_not_refused() {
        let steps: Vec<String> = (0..50).map(|i| format!("step {i}")).collect();
        assert_eq!(tidy(&steps).len(), MAX_STEPS);
    }

    #[test]
    fn trivial_tasks_skip_the_planning_turn() {
        assert!(!worth_planning("fix it"));
        assert!(!worth_planning("add a flag"));
        assert!(worth_planning("add a flag to the config loader"));
    }

    #[tokio::test]
    async fn a_broken_engine_does_not_raise() {
        let eng = MockEngine::new(vec![]); // no scripted replies: complete fails
        let p = plan(
            &eng,
            &Themis::from_text("x"),
            &scribe(),
            "a task long enough to plan",
            512,
        )
        .await
        .unwrap();
        assert_eq!(p.steps, ["a task long enough to plan"]);
    }
}
