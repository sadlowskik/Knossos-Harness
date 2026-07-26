//! Tool calling for engines that do not have it.
//!
//! Local models vary wildly: some expose native tool calling, some emit JSON
//! when asked nicely, some do neither reliably. When an engine reports
//! `supports_native_tools() == false`, the harness renders the tool
//! definitions into the system prompt and parses fenced JSON back out of the
//! reply itself.
//!
//! This is strictly a shim. It is less reliable than native tool use, and the
//! honest reason it exists is that "trait + both backends" would otherwise be
//! a claim rather than a fact.

use crate::engine::types::{Content, Response, StopReason, ToolDef};

const PROTOCOL: &str = r#"
# Calling tools

You do not have native tool access. To call a tool, emit a fenced JSON block:

```json
{"tool": "<tool_name>", "input": { ... }}
```

Rules:
- One tool call per block. You may emit several blocks in one reply.
- `input` must satisfy that tool's schema exactly.
- Emit nothing after a tool call block except further tool call blocks.
- When you are done and need no tools, reply with prose and no JSON block.
"#;

/// Append tool documentation and the call protocol to a system prompt.
pub fn augment_system(system: &str, tools: &[ToolDef]) -> String {
    let mut out = String::from(system);
    out.push_str("\n\n# Available tools\n");
    for t in tools {
        out.push_str(&format!(
            "\n## {}\n{}\n\nInput schema:\n```json\n{}\n```\n",
            t.name,
            t.description,
            serde_json::to_string_pretty(&t.input_schema)
                .unwrap_or_else(|_| "{}".to_string())
        ));
    }
    out.push_str(PROTOCOL);
    out
}

/// Rewrite a response in place, promoting fenced JSON tool calls into real
/// `ToolUse` blocks so downstream code cannot tell the difference.
pub fn extract_tool_calls(resp: &mut Response) {
    let mut rewritten: Vec<Content> = Vec::new();
    let mut found = 0usize;

    for block in resp.content.drain(..) {
        let text = match &block {
            Content::Text { text } => text.clone(),
            other => {
                rewritten.push(other.clone());
                continue;
            }
        };

        let (prose, calls) = split_fenced_calls(&text, &mut found);
        if !prose.trim().is_empty() {
            rewritten.push(Content::text(prose.trim()));
        }
        rewritten.extend(calls);
    }

    resp.content = rewritten;
    if found > 0 {
        resp.stop_reason = StopReason::ToolUse;
    }
}

/// Split text into (prose without tool-call fences, extracted ToolUse blocks).
fn split_fenced_calls(text: &str, counter: &mut usize) -> (String, Vec<Content>) {
    let mut prose = String::new();
    let mut calls = Vec::new();
    let mut rest = text;

    while let Some(start) = find_fence_start(rest) {
        let after_open = &rest[start..];
        let body_start = match after_open.find('\n') {
            Some(i) => start + i + 1,
            None => break,
        };
        let close = match rest[body_start..].find("```") {
            Some(i) => body_start + i,
            None => break,
        };

        let body = &rest[body_start..close];
        match parse_call(body, counter) {
            Some(call) => {
                prose.push_str(&rest[..start]);
                calls.push(call);
            }
            // Not a tool call — a plain JSON code block. Keep it as prose.
            None => prose.push_str(&rest[..close + 3.min(rest.len() - close)]),
        }
        rest = &rest[(close + 3).min(rest.len())..];
    }

    prose.push_str(rest);
    (prose, calls)
}

/// Locate the next ```json (or bare ```) fence opening.
fn find_fence_start(s: &str) -> Option<usize> {
    s.find("```")
}

fn parse_call(body: &str, counter: &mut usize) -> Option<Content> {
    let value: serde_json::Value = serde_json::from_str(body.trim()).ok()?;
    let obj = value.as_object()?;
    let name = obj.get("tool")?.as_str()?.to_string();
    let input = obj.get("input").cloned().unwrap_or(serde_json::json!({}));
    let id = format!("call_{}", *counter);
    *counter += 1;
    Some(Content::ToolUse { id, name, input })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::types::Usage;

    fn resp(text: &str) -> Response {
        Response {
            content: vec![Content::text(text)],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        }
    }

    #[test]
    fn extracts_a_fenced_tool_call() {
        let mut r = resp("Let me look.\n```json\n{\"tool\": \"read\", \"input\": {\"path\": \"a.rs\"}}\n```");
        extract_tool_calls(&mut r);
        assert_eq!(r.stop_reason, StopReason::ToolUse);
        let calls = r.tool_uses();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, "read");
        assert_eq!(calls[0].2["path"], "a.rs");
    }

    #[test]
    fn leaves_plain_json_blocks_as_prose() {
        let mut r = resp("Here is config:\n```json\n{\"debug\": true}\n```");
        extract_tool_calls(&mut r);
        assert_eq!(r.stop_reason, StopReason::EndTurn);
        assert!(r.tool_uses().is_empty());
    }

    #[test]
    fn extracts_several_calls() {
        let mut r = resp(
            "```json\n{\"tool\":\"read\",\"input\":{\"path\":\"a\"}}\n```\n```json\n{\"tool\":\"read\",\"input\":{\"path\":\"b\"}}\n```",
        );
        extract_tool_calls(&mut r);
        assert_eq!(r.tool_uses().len(), 2);
    }

    #[test]
    fn augmented_system_names_every_tool() {
        let tools = vec![ToolDef {
            name: "read".into(),
            description: "Read a file".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let s = augment_system("base", &tools);
        assert!(s.contains("base"));
        assert!(s.contains("## read"));
        assert!(s.contains("Calling tools"));
    }
}
