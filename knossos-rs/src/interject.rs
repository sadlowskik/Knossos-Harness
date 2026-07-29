//! Saying something to a run that is already going.
//!
//! Cancellation is the only mid-run control the harness had: the loop either
//! finished or it was killed. That is a poor fit for the failure it actually
//! sees most — the agent is doing something reasonable and slightly wrong, and
//! the person watching knows the one sentence that would fix it. Killing the
//! run throws away the context that made it nearly right; waiting for it to
//! finish spends the whole step budget being wrong on purpose.
//!
//! So: a queue the loop drains between steps.
//!
//! # Why only between steps
//!
//! A step is not interruptible in the middle, and the reason is structural
//! rather than an implementation shortcut. Inside a step the conversation
//! passes through states that are not valid to append to — most sharply,
//! between an assistant message carrying `tool_use` blocks and the user message
//! carrying their `tool_result`s. A message inserted there produces a request
//! the provider rejects, and the agent would lose the whole turn to a protocol
//! error rather than gain a correction.
//!
//! The step boundary is the one point where the conversation is a complete,
//! well-formed exchange. Everything queued lands there, in order, as a single
//! user message — so three corrections typed in quick succession cost one turn
//! rather than three.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// How many pending interjections are held before new ones are refused.
///
/// A person typing will never reach this. A programmatic producer — a front end
/// forwarding notifications, a test in a loop — can, and an unbounded queue
/// behind an agent that may be busy for minutes is a slow leak with a
/// respectable-looking name.
const CAPACITY: usize = 64;

/// A handle for putting words into a running loop.
///
/// Cheap to clone and safe to hold across threads: the front end keeps one
/// while [`Talos`](crate::talos::Talos) holds another, and they address the
/// same queue.
#[derive(Debug, Clone, Default)]
pub struct Interjections {
    queue: Arc<Mutex<VecDeque<String>>>,
}

impl Interjections {
    pub fn new() -> Self {
        Interjections::default()
    }

    /// Queue text for the agent to read at the next step boundary.
    ///
    /// Returns `false` if the queue is full, so a front end can tell the person
    /// their message was not accepted. Silently dropping it would be worse than
    /// refusing: they would believe the agent had been told.
    pub fn push(&self, text: impl Into<String>) -> bool {
        let text = text.into();
        if text.trim().is_empty() {
            // An empty interjection would spend a turn saying nothing.
            return false;
        }
        let mut q = self.queue.lock().unwrap();
        if q.len() >= CAPACITY {
            return false;
        }
        q.push_back(text);
        true
    }

    /// Take everything queued, oldest first.
    pub fn drain(&self) -> Vec<String> {
        self.queue.lock().unwrap().drain(..).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.lock().unwrap().is_empty()
    }

    pub fn len(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    /// Everything queued, rendered as one user message.
    ///
    /// `None` when nothing is waiting, so the caller can skip the push without
    /// having to decide what an empty interjection means.
    ///
    /// Marked as coming from the person rather than presented as bare text: the
    /// agent is mid-task and needs to be able to tell a new instruction from
    /// the tool output and pressure notes around it.
    pub fn take_message(&self) -> Option<String> {
        let pending = self.drain();
        if pending.is_empty() {
            return None;
        }
        Some(Interjections::render(&pending))
    }

    /// Render already-drained notes as the message the agent sees.
    ///
    /// Split out because the loop needs the notes twice — once for the trace,
    /// once for the conversation — and draining them a second time would return
    /// nothing.
    pub fn render(notes: &[String]) -> String {
        let mut out = String::from(
            "The user interrupted with the following. Take it into account before continuing:\n",
        );
        for note in notes {
            out.push_str("\n- ");
            out.push_str(note.trim());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_queued_produces_no_message() {
        let i = Interjections::new();
        assert!(i.take_message().is_none());
        assert!(i.is_empty());
    }

    #[test]
    fn interjections_arrive_in_the_order_they_were_said() {
        let i = Interjections::new();
        assert!(i.push("use the existing helper"));
        assert!(i.push("and do not touch the tests"));

        let msg = i.take_message().unwrap();
        let first = msg.find("use the existing helper").unwrap();
        let second = msg.find("and do not touch the tests").unwrap();
        assert!(first < second, "order must be preserved:\n{msg}");
    }

    #[test]
    fn several_corrections_become_one_message_not_several_turns() {
        let i = Interjections::new();
        i.push("one");
        i.push("two");
        i.push("three");

        let msg = i.take_message().unwrap();
        assert!(msg.contains("one") && msg.contains("two") && msg.contains("three"));
        assert!(i.is_empty(), "taking the message must clear the queue");
        assert!(i.take_message().is_none(), "and must not deliver it twice");
    }

    #[test]
    fn the_message_says_it_came_from_the_person() {
        // Without this the agent reads a bare instruction sandwiched between
        // tool results and has no way to weigh it differently.
        let i = Interjections::new();
        i.push("stop rewriting the parser");
        assert!(i.take_message().unwrap().contains("user interrupted"));
    }

    #[test]
    fn empty_and_whitespace_input_is_refused_rather_than_queued() {
        let i = Interjections::new();
        assert!(!i.push(""));
        assert!(!i.push("   \n  "));
        assert!(i.is_empty());
    }

    #[test]
    fn a_full_queue_refuses_rather_than_growing() {
        let i = Interjections::new();
        for n in 0..CAPACITY {
            assert!(i.push(format!("note {n}")), "should accept up to capacity");
        }
        assert!(!i.push("one too many"));
        assert_eq!(i.len(), CAPACITY);
    }

    #[test]
    fn a_cloned_handle_addresses_the_same_queue() {
        // The whole point: the front end holds one, the loop holds another.
        let front_end = Interjections::new();
        let loop_side = front_end.clone();

        front_end.push("from the user");
        assert_eq!(loop_side.len(), 1);

        let msg = loop_side.take_message().unwrap();
        assert!(msg.contains("from the user"));
        assert!(front_end.is_empty(), "draining on one side clears the other");
    }

    #[test]
    fn pushing_from_another_thread_is_visible_to_the_loop() {
        let loop_side = Interjections::new();
        let front_end = loop_side.clone();
        let t = std::thread::spawn(move || front_end.push("said from elsewhere"));
        assert!(t.join().unwrap());
        assert!(loop_side.take_message().unwrap().contains("said from elsewhere"));
    }
}
