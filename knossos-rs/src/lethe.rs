//! Lethe: keeping the conversation inside a budget.
//!
//! The river of forgetting. Rust's `Talos` kept every message of every turn and
//! rebuilt the whole request each step, so the conversation grew without bound —
//! measured at roughly 96 000 tokens by the end of turn one and past 270 000 by
//! turn three, since the system prompt is rebuilt every step as well. The step
//! ceiling was the only thing keeping that finite, which is a blunt instrument
//! rather than a bound.
//!
//! # Why this does not summarise, and does not drop messages
//!
//! The Python side compacts by replacing a middle band of the transcript with an
//! extractive summary. That is safe there because a Python transcript is a list
//! of strings: losing one loses information and nothing else.
//!
//! Here it would be a correctness bug, not a fidelity loss. A `ToolUse` block
//! carries an id, and the `ToolResult` answering it carries the same id in the
//! *next* message. Providers pair them by that id, and an id present without its
//! partner is a hard error — Anthropic rejects the request outright rather than
//! degrading. Any strategy that removes or merges messages can split a pair:
//! summarise a middle band containing the call while the result survives in the
//! kept tail, and the next request fails. That exact bug was hit on the Python
//! side, where it merely produced a malformed prompt; here it would end runs.
//!
//! So nothing is removed. **Message count, block count, block order, every id
//! and every `ToolUse` input are all invariant.** The only thing that changes is
//! the *text inside* `Text` and `ToolResult` blocks, which shrinks in place with
//! a marker. Pairing is therefore preserved structurally rather than by a check
//! that has to be kept correct — there is no code path here that could break it.
//!
//! What that costs: the floor is higher than a summarising compactor's, because
//! a conversation of five hundred tiny messages cannot be shrunk much by
//! eliding any of them. That is the right trade for a loop whose entries are
//! file reads and cargo output — a transcript goes over budget because two
//! entries are enormous, not because five hundred are slightly too long.

use crate::engine::{Content, Message};

pub const DEFAULT_MAX_TOKENS: usize = 24_000;

/// Rough token estimate: four characters per token.
///
/// Deliberately the same crude ratio as the Python side, and deliberately an
/// *under*-estimate for code and JSON, which is what a tool result is full of.
/// The caller compensates with a margin; a cleverer estimate here would still
/// be wrong on the provider's tokenizer and would hide that it was guessing.
pub fn estimate_tokens(messages: &[Message]) -> usize {
    messages.iter().map(message_chars).sum::<usize>() / 4
}

fn message_chars(message: &Message) -> usize {
    message
        .content
        .iter()
        .map(|block| match block {
            Content::Text { text } => text.len(),
            Content::ToolResult { content, .. } => content.len(),
            // Counted but never elided: the input *is* the call. Shrinking it
            // would misreport what the agent did, and it is small anyway.
            Content::ToolUse { input, .. } => input.to_string().len(),
        })
        .sum::<usize>()
        + 8 // per-message framing overhead, roughly
}

pub struct Lethe {
    pub max_tokens: usize,
    /// Messages at the end left completely alone — the model's working context.
    pub keep_recent: usize,
    /// Messages at the start left completely alone — the task statement.
    ///
    /// A conversation that has forgotten what it was asked to do is worse than
    /// one that is over budget, which is the entire reason this exists.
    pub pin_opening: usize,
}

impl Default for Lethe {
    fn default() -> Self {
        Lethe {
            max_tokens: DEFAULT_MAX_TOKENS,
            keep_recent: 6,
            pin_opening: 1,
        }
    }
}

/// Smallest a block is worth shrinking to. Below this the marker costs more
/// than the elision saves.
const MIN_BLOCK_CHARS: usize = 400;

/// Hard stop, so a pathological conversation cannot spin. Each pass halves the
/// largest block, so this is far more than enough to reach any budget.
const MAX_PASSES: usize = 64;

impl Lethe {
    /// Bring `messages` within budget. Returns whether anything changed.
    ///
    /// When nothing further can be given up this returns what it has: an
    /// over-budget request is still better than a gutted one, and the caller
    /// can see the shortfall by asking `estimate_tokens` again.
    /// Bring `messages` within budget. Returns whether anything changed.
    ///
    /// Two phases, because `keep_recent` is a preference and the budget is not.
    /// The first pass spares the recent tail, since that is the context the
    /// model is actually working from. If the conversation is still over budget
    /// afterwards the second pass includes it, because the alternative is
    /// sending a request the server will truncate from the front — losing the
    /// task statement instead of the middle of one tool result, and losing it
    /// silently.
    ///
    /// The second phase is not an edge case. `keep_recent` is 6 by default, so
    /// a four-message conversation has no middle band at all — and that is
    /// exactly the shape of turn one, where a single enormous file read arrives.
    /// Stopping at phase one would mean the bound did not apply precisely when
    /// it was first needed.
    pub fn compact(&self, messages: &mut [Message]) -> bool {
        if estimate_tokens(messages) <= self.max_tokens {
            return false;
        }
        let sparing_the_tail = messages.len().saturating_sub(self.keep_recent);
        let mut changed = self.shrink_within(messages, sparing_the_tail);
        if estimate_tokens(messages) > self.max_tokens {
            changed |= self.shrink_within(messages, messages.len());
        }
        changed
    }

    /// Elide largest-first across `messages[pin_opening..end]`.
    fn shrink_within(&self, messages: &mut [Message], end: usize) -> bool {
        if end <= self.pin_opening {
            return false;
        }
        let mut changed = false;
        for _ in 0..MAX_PASSES {
            if estimate_tokens(messages) <= self.max_tokens {
                break;
            }
            let Some((mi, bi, len)) = largest_block(&messages[self.pin_opening..end])
                .map(|(mi, bi, len)| (mi + self.pin_opening, bi, len))
            else {
                break;
            };
            if len <= MIN_BLOCK_CHARS {
                break; // nothing left worth taking
            }
            let target = (len / 2).max(MIN_BLOCK_CHARS);
            if !elide_block(&mut messages[mi].content[bi], target) {
                break; // the marker would cost more than the elision saves
            }
            changed = true;
        }
        changed
    }
}

/// `(message index, block index, length)` of the largest elidable block.
fn largest_block(messages: &[Message]) -> Option<(usize, usize, usize)> {
    let mut best: Option<(usize, usize, usize)> = None;
    for (mi, message) in messages.iter().enumerate() {
        for (bi, block) in message.content.iter().enumerate() {
            let len = match block {
                Content::Text { text } => text.len(),
                Content::ToolResult { content, .. } => content.len(),
                // Never a candidate. See the module docs.
                Content::ToolUse { .. } => continue,
            };
            if best.is_none_or(|(_, _, b)| len > b) {
                best = Some((mi, bi, len));
            }
        }
    }
    best
}

/// Shrink one block's text toward `target`. False if it would not help.
fn elide_block(block: &mut Content, target: usize) -> bool {
    let text = match block {
        Content::Text { text } => text,
        Content::ToolResult { content, .. } => content,
        Content::ToolUse { .. } => return false,
    };
    let Some(shorter) = elide(text, target) else {
        return false;
    };
    *text = shorter;
    true
}

/// Keep both ends of `text`, dropping the middle. `None` if that is no shorter.
///
/// Both ends, because the two halves answer different questions. The head says
/// what this block *is* — which tool ran, against which file. The tail carries
/// the conclusion: cargo's error summary, the last frames of a panic, the end of
/// a diff. Keeping only the head, which is what plain truncation does, throws
/// away the part the model has to act on.
fn elide(text: &str, target: usize) -> Option<String> {
    if text.len() <= target {
        return None;
    }
    let half = (target / 2).max(1);
    // Char boundaries, not byte offsets: a tool result is arbitrary UTF-8 and
    // slicing mid-character panics.
    let head_end = floor_boundary(text, half);
    let tail_start = ceil_boundary(text, text.len() - half);
    if tail_start <= head_end {
        return None;
    }
    let dropped = tail_start - head_end;
    let out = format!(
        "{}\n\n[… {dropped} characters elided …]\n\n{}",
        &text[..head_end],
        &text[tail_start..]
    );
    if out.len() >= text.len() {
        return None;
    }
    Some(out)
}

fn floor_boundary(text: &str, mut index: usize) -> usize {
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn ceil_boundary(text: &str, mut index: usize) -> usize {
    while index < text.len() && !text.is_char_boundary(index) {
        index += 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    fn big(n: usize) -> String {
        "x".repeat(n)
    }

    fn conversation() -> Vec<Message> {
        vec![
            Message::user_text("## Task\nfix the bug"),
            Message::assistant(vec![Content::ToolUse {
                id: "call_1".into(),
                name: "read_file".into(),
                input: serde_json::json!({ "path": "src/lib.rs" }),
            }]),
            Message::user(vec![Content::ToolResult {
                id: "call_1".into(),
                content: big(200_000),
                is_error: false,
            }]),
            Message::assistant(vec![Content::text("I see the problem.")]),
        ]
    }

    #[test]
    fn an_over_budget_conversation_is_brought_within_budget() {
        let mut messages = conversation();
        let lethe = Lethe {
            max_tokens: 2_000,
            keep_recent: 1,
            pin_opening: 1,
        };

        assert!(lethe.compact(&mut messages));
        assert!(estimate_tokens(&messages) <= 2_000);
    }

    #[test]
    fn no_message_and_no_block_is_ever_removed() {
        // The load-bearing property. A `ToolUse` id without its `ToolResult` is
        // a hard provider error, not a degradation, so pairing is preserved by
        // construction rather than by a check.
        let mut messages = conversation();
        let before: Vec<(usize, usize)> = messages.iter().map(|m| (m.content.len(), 0)).collect();
        let lethe = Lethe {
            max_tokens: 500,
            keep_recent: 1,
            pin_opening: 1,
        };

        lethe.compact(&mut messages);

        assert_eq!(messages.len(), 4);
        let after: Vec<(usize, usize)> = messages.iter().map(|m| (m.content.len(), 0)).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn every_tool_use_keeps_its_result() {
        let mut messages = conversation();
        Lethe {
            max_tokens: 500,
            keep_recent: 1,
            pin_opening: 1,
        }
        .compact(&mut messages);

        let uses: Vec<&str> = messages
            .iter()
            .flat_map(|m| &m.content)
            .filter_map(|c| match c {
                Content::ToolUse { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        let results: Vec<&str> = messages
            .iter()
            .flat_map(|m| &m.content)
            .filter_map(|c| match c {
                Content::ToolResult { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();

        assert_eq!(uses, results);
    }

    #[test]
    fn a_tool_uses_input_is_never_elided() {
        // The input *is* the call. Shrinking it would misreport what the agent
        // did, and would corrupt the record the trace log is built from.
        let mut messages = conversation();
        messages[1].content = vec![Content::ToolUse {
            id: "call_1".into(),
            name: "write_file".into(),
            input: serde_json::json!({ "path": "a.rs", "content": big(50_000) }),
        }];
        let lethe = Lethe {
            max_tokens: 100,
            keep_recent: 1,
            pin_opening: 1,
        };

        lethe.compact(&mut messages);

        match &messages[1].content[0] {
            Content::ToolUse { input, .. } => {
                assert_eq!(input["content"].as_str().unwrap().len(), 50_000);
            }
            other => panic!("the tool use became {other:?}"),
        }
    }

    #[test]
    fn the_task_statement_survives() {
        let mut messages = conversation();
        messages[0] = Message::user_text(format!("## Task\n{}", big(100_000)));
        let original = messages[0].clone();
        let lethe = Lethe {
            max_tokens: 500,
            keep_recent: 1,
            pin_opening: 1,
        };

        lethe.compact(&mut messages);

        // Byte-identical, not merely recognisable: a conversation that has
        // forgotten what it was asked to do is worse than one over budget.
        assert_eq!(messages[0], original);
    }

    #[test]
    fn a_short_conversation_is_still_bounded() {
        // The case that matters most and that `keep_recent` alone would miss:
        // turn one, four messages, one enormous file read. With a six-message
        // tail there is no middle band, so a single-phase compactor returns
        // early and the bound does not apply exactly when it is first needed.
        let mut messages = conversation();
        assert!(messages.len() < Lethe::default().keep_recent);
        let lethe = Lethe {
            max_tokens: 2_000,
            ..Lethe::default()
        };

        assert!(lethe.compact(&mut messages));
        assert!(estimate_tokens(&messages) <= 2_000);
        // ...and it is still a well-formed conversation.
        assert_eq!(messages.len(), 4);
    }

    #[test]
    fn the_tail_is_spared_when_sparing_it_is_enough() {
        // Phase two exists for when phase one is not enough, not instead of it.
        let mut messages = vec![
            Message::user_text("## Task\nfix the bug"),
            Message::user_text(big(200_000)),
            Message::user_text("recent 1"),
            Message::user_text("recent 2"),
        ];
        let tail = messages[3].clone();
        let lethe = Lethe {
            max_tokens: 2_000,
            keep_recent: 2,
            pin_opening: 1,
        };

        lethe.compact(&mut messages);

        assert_eq!(messages[3], tail);
        assert!(estimate_tokens(&messages) <= 2_000);
    }

    #[test]
    fn the_recent_tail_is_untouched() {
        let mut messages = conversation();
        let tail = messages.last().unwrap().clone();
        Lethe {
            max_tokens: 500,
            keep_recent: 1,
            pin_opening: 1,
        }
        .compact(&mut messages);

        assert_eq!(messages.last().unwrap(), &tail);
    }

    #[test]
    fn a_conversation_within_budget_is_left_alone() {
        let mut messages = vec![Message::user_text("hello")];
        let before = messages.clone();

        assert!(!Lethe::default().compact(&mut messages));
        assert_eq!(messages, before);
    }

    #[test]
    fn compaction_is_idempotent_once_it_fits() {
        let mut messages = conversation();
        let lethe = Lethe {
            max_tokens: 2_000,
            keep_recent: 1,
            pin_opening: 1,
        };

        lethe.compact(&mut messages);
        let once = messages.clone();
        lethe.compact(&mut messages);

        assert_eq!(messages, once);
    }

    #[test]
    fn nothing_left_to_take_terminates_rather_than_spinning() {
        // Many tiny messages: the floor this strategy accepts in exchange for
        // never being able to break a pair.
        let mut messages: Vec<Message> = (0..500)
            .map(|i| Message::user_text(format!("turn {i}")))
            .collect();
        let lethe = Lethe {
            max_tokens: 1,
            keep_recent: 1,
            pin_opening: 1,
        };

        lethe.compact(&mut messages);

        assert_eq!(messages.len(), 500);
    }

    #[test]
    fn multibyte_content_is_not_sliced_mid_character() {
        let mut messages = conversation();
        messages[2].content = vec![Content::ToolResult {
            id: "call_1".into(),
            content: "é".repeat(50_000),
            is_error: false,
        }];
        let lethe = Lethe {
            max_tokens: 500,
            keep_recent: 1,
            pin_opening: 1,
        };

        lethe.compact(&mut messages); // panics on a bad boundary

        match &messages[2].content[0] {
            Content::ToolResult { content, .. } => assert!(content.contains("elided")),
            other => panic!("became {other:?}"),
        }
    }

    #[test]
    fn the_marker_says_something_was_dropped() {
        let mut messages = conversation();
        Lethe {
            max_tokens: 2_000,
            keep_recent: 1,
            pin_opening: 1,
        }
        .compact(&mut messages);

        match &messages[2].content[0] {
            Content::ToolResult { content, .. } => {
                assert!(
                    content.contains("elided"),
                    "the model must be told, not lied to"
                );
            }
            other => panic!("became {other:?}"),
        }
    }
}
