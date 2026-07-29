//! What went wrong, in a form something other than a human can read.
//!
//! Both backends used to report failure as prose — `bail!("Anthropic API
//! returned {status}: {text}")`. That reads well in a terminal and is useless
//! to a caller that has to *decide* something, which is exactly what
//! [`Resilient`](crate::resilience::Resilient) does: it must tell a rate limit
//! worth waiting for from an API key that will be just as wrong in eight
//! seconds. Recovering a status code by searching the message for `"returned
//! 429"` couples the retry policy to another module's phrasing, and the coupling
//! is invisible — nothing breaks at compile time when the wording changes.
//!
//! So the status survives as a number, and the judgement about what a number
//! means lives here rather than in the wrapper. The backend knows its own
//! protocol; the retry policy should not have to.
//!
//! The [`Engine`](crate::engine::Engine) trait still returns `anyhow::Result`,
//! so this is additive: an engine that does not use these variants keeps
//! working, and [`Resilient`](crate::resilience::Resilient) treats an untyped
//! failure as retryable.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    /// The backend answered, with a status saying it would not serve.
    ///
    /// Rendered the way the old `bail!` rendered it, so traces and tests that
    /// match on the message text still read the same.
    #[error("{provider} returned {status}: {body}")]
    Status {
        provider: &'static str,
        status: u16,
        body: String,
    },

    /// The request never got an answer: connection refused, DNS, a timeout.
    #[error("{provider}: {detail}")]
    Transport {
        provider: &'static str,
        detail: String,
    },
}

impl EngineError {
    /// Whether trying again could plausibly produce a different answer.
    ///
    /// The listed codes are the ones that describe the *request* rather than
    /// the server's mood: a malformed body, a bad key, a forbidden or missing
    /// endpoint, an unprocessable payload. Repeating any of them repeats the
    /// same rejection. Everything else — 429, 5xx, and anything unrecognised —
    /// is treated as worth another attempt, because the cost of retrying a
    /// permanent failure is bounded by the circuit breaker while the cost of
    /// *not* retrying a transient one is a lost run.
    pub fn is_transient(&self) -> bool {
        match self {
            EngineError::Transport { .. } => true,
            EngineError::Status { status, .. } => {
                !matches!(status, 400 | 401 | 403 | 404 | 422)
            }
        }
    }

    /// The HTTP status, when the failure had one.
    pub fn status(&self) -> Option<u16> {
        match self {
            EngineError::Status { status, .. } => Some(*status),
            EngineError::Transport { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status(code: u16) -> EngineError {
        EngineError::Status { provider: "Test API", status: code, body: "b".into() }
    }

    #[test]
    fn a_rate_limit_and_a_server_fault_are_worth_retrying() {
        assert!(status(429).is_transient());
        assert!(status(500).is_transient());
        assert!(status(503).is_transient());
        assert!(status(529).is_transient(), "Anthropic's overloaded status");
    }

    #[test]
    fn a_rejected_request_is_not_worth_retrying() {
        assert!(!status(400).is_transient());
        assert!(!status(401).is_transient());
        assert!(!status(403).is_transient());
        assert!(!status(404).is_transient());
        assert!(!status(422).is_transient());
    }

    #[test]
    fn a_request_that_never_arrived_is_always_worth_retrying() {
        // `ollama serve` starting a second later is the ordinary case.
        let e = EngineError::Transport { provider: "Ollama", detail: "refused".into() };
        assert!(e.is_transient());
        assert_eq!(e.status(), None);
    }

    #[test]
    fn an_unrecognised_status_defaults_to_retryable() {
        // Guessing "permanent" about a code nobody listed turns one odd reply
        // into an ended run; guessing "transient" costs a bounded few seconds.
        assert!(status(418).is_transient());
    }

    #[test]
    fn the_message_still_reads_the_way_it_used_to() {
        let e = EngineError::Status {
            provider: "Anthropic API",
            status: 429,
            body: "rate limited".into(),
        };
        assert_eq!(e.to_string(), "Anthropic API returned 429: rate limited");
    }
}
