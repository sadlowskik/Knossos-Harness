//! Surviving an engine that is having a bad minute.
//!
//! Neither backend retries. [`anthropic`](crate::engine::anthropic) and
//! [`ollama`](crate::engine::ollama) both `bail!` on any non-success status,
//! `engine::complete` propagates it, and `Talos::drive` takes it with `?` — so a
//! single 429 ends a run that may be twenty steps deep, and the trace records a
//! failure that is not about the model, the harness, or the task.
//!
//! That failure is already being paid for downstream: the Python side
//! reconstructs an `unreachable` flag by scanning traces for the API failure
//! prefix, precisely so those runs can be excluded from a score. Absorbing the
//! blip here means fewer runs to exclude.
//!
//! # Two mechanisms, and why neither is enough alone
//!
//! **Retry with backoff** handles the blip. On its own it turns a genuinely
//! dead endpoint into a run that sleeps through the full schedule on every step
//! before failing — slower than not retrying at all, and every step pays it.
//!
//! **A breaker** handles the outage: after enough consecutive failures it stops
//! calling and fails immediately. On its own it does nothing for the blip that a
//! single retry would have absorbed.
//!
//! Together the cost of being wrong is bounded, which is what makes the crude
//! error classification below acceptable: misjudging a terminal error as
//! transient costs a few retries and then the breaker stops it.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use async_trait::async_trait;

use crate::engine::{Engine, EngineError, Request, Response};

#[derive(Debug, Clone)]
pub struct Policy {
    /// Attempts per call, including the first. 1 disables retrying.
    pub max_attempts: usize,
    /// Wait before the second attempt; doubles each time, capped.
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    /// Consecutive failed calls before the breaker opens.
    pub trip_after: usize,
    /// How long the breaker stays open before allowing a trial call.
    pub cooldown: Duration,
}

impl Default for Policy {
    fn default() -> Self {
        Policy {
            // Three attempts covers the overwhelming majority of transient
            // failures without turning a step into a minute.
            max_attempts: 3,
            initial_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(8),
            trip_after: 3,
            cooldown: Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Calls pass through. Carries the consecutive-failure count.
    Closed(usize),
    /// Calls fail immediately until this instant.
    Open(Instant),
}

/// Whether an error will fail the same way however many times it is repeated.
///
/// The judgement itself lives on [`EngineError`], next to the backend that
/// knows what its own status codes mean. This function only decides what to do
/// about an error that is not one — which is to retry it.
///
/// That default is the deliberate half. An engine outside this crate, or the
/// scripted mock, reports failure however it likes, and treating an unreadable
/// error as permanent would end a run over a message this module failed to
/// parse. Retrying instead costs at most `max_attempts` and then the breaker.
fn is_terminal(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<EngineError>()
        .is_some_and(|e| !e.is_transient())
}

/// An [`Engine`] that retries transient failures and stops calling a dead one.
///
/// Wraps any engine, so it composes with the prompted-JSON shim rather than
/// competing with it: `engine::complete` still decides which protocol to speak,
/// and this decides whether the call happens at all.
pub struct Resilient {
    inner: Box<dyn Engine>,
    policy: Policy,
    state: Mutex<State>,
    /// Calls the breaker refused. Read in tests, and worth surfacing in a
    /// summary — a run that failed with a tripped breaker failed for a reason
    /// that has nothing to do with the task.
    short_circuited: Mutex<usize>,
}

impl Resilient {
    pub fn new(inner: Box<dyn Engine>) -> Self {
        Resilient::with_policy(inner, Policy::default())
    }

    pub fn with_policy(inner: Box<dyn Engine>, policy: Policy) -> Self {
        Resilient {
            inner,
            policy,
            state: Mutex::new(State::Closed(0)),
            short_circuited: Mutex::new(0),
        }
    }

    pub fn short_circuited(&self) -> usize {
        *self.short_circuited.lock().unwrap()
    }

    /// Whether the breaker is currently refusing calls.
    pub fn is_open(&self) -> bool {
        matches!(*self.state.lock().unwrap(), State::Open(until) if Instant::now() < until)
    }

    /// Take the breaker's verdict on whether this call may proceed.
    ///
    /// Separated so the lock is released before any `await`. Holding a `std`
    /// mutex across an await point is how an async harness deadlocks itself.
    fn admit(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        match *state {
            State::Closed(_) => true,
            State::Open(until) => {
                if Instant::now() >= until {
                    // Half-open: one call is let through to find out whether the
                    // endpoint came back. Recorded as Closed with the failure
                    // count retained, so a single failure re-opens it rather
                    // than granting a fresh budget of attempts.
                    *state = State::Closed(self.policy.trip_after.saturating_sub(1));
                    true
                } else {
                    false
                }
            }
        }
    }

    fn record_success(&self) {
        *self.state.lock().unwrap() = State::Closed(0);
    }

    fn record_failure(&self) {
        let mut state = self.state.lock().unwrap();
        let failures = match *state {
            State::Closed(n) => n + 1,
            State::Open(_) => self.policy.trip_after,
        };
        *state = if failures >= self.policy.trip_after {
            State::Open(Instant::now() + self.policy.cooldown)
        } else {
            State::Closed(failures)
        };
    }
}

#[async_trait]
impl Engine for Resilient {
    async fn complete(&self, req: &Request) -> Result<Response> {
        if !self.admit() {
            *self.short_circuited.lock().unwrap() += 1;
            bail!(
                "engine circuit is open after {} consecutive failures; not calling `{}`",
                self.policy.trip_after,
                self.inner.name()
            );
        }

        let mut backoff = self.policy.initial_backoff;
        let mut last: Option<anyhow::Error> = None;

        for attempt in 1..=self.policy.max_attempts.max(1) {
            match self.inner.complete(req).await {
                Ok(resp) => {
                    self.record_success();
                    return Ok(resp);
                }
                Err(e) if is_terminal(&e) => {
                    // Counted as a failure even though it is not retried: a
                    // wrong key should trip the breaker rather than be
                    // rediscovered on every step for the rest of the run.
                    self.record_failure();
                    return Err(e);
                }
                Err(e) => {
                    last = Some(e);
                    if attempt < self.policy.max_attempts {
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(self.policy.max_backoff);
                    }
                }
            }
        }

        self.record_failure();
        Err(last.expect("a loop that ran at least once without succeeding has an error"))
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn supports_native_tools(&self) -> bool {
        self.inner.supports_native_tools()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::engine::types::{Content, Request, Response, StopReason, Usage};

    /// How a `Flaky` fails.
    #[derive(Clone, Copy)]
    enum Failure {
        Status(u16),
        Transport,
        /// Not an [`EngineError`] at all — what a third-party engine or the
        /// scripted mock produces.
        Untyped,
    }

    impl Failure {
        fn build(self) -> anyhow::Error {
            match self {
                Failure::Status(status) => EngineError::Status {
                    provider: "Test API",
                    status,
                    body: "b".into(),
                }
                .into(),
                Failure::Transport => EngineError::Transport {
                    provider: "Test API",
                    detail: "connection refused".into(),
                }
                .into(),
                Failure::Untyped => anyhow::anyhow!("something went wrong"),
            }
        }
    }

    /// Fails its first `failures` calls, then succeeds. Counts every call, so a
    /// test can assert the breaker stopped one from happening at all.
    struct Flaky {
        failures: usize,
        calls: AtomicUsize,
        failure: Failure,
    }

    impl Flaky {
        fn new(failures: usize) -> Self {
            Flaky {
                failures,
                calls: AtomicUsize::new(0),
                failure: Failure::Status(429),
            }
        }

        fn always(failure: Failure) -> Self {
            Flaky { failures: usize::MAX, calls: AtomicUsize::new(0), failure }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Engine for Flaky {
        async fn complete(&self, _req: &Request) -> Result<Response> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n < self.failures {
                return Err(self.failure.build());
            }
            Ok(Response {
                content: vec![Content::text("ok")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        }
        fn name(&self) -> &str {
            "flaky"
        }
        fn supports_native_tools(&self) -> bool {
            true
        }
    }

    /// Backoff short enough that the tests are not slow, everything else real.
    fn fast() -> Policy {
        Policy {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(4),
            trip_after: 2,
            cooldown: Duration::from_millis(30),
        }
    }

    fn req() -> Request {
        Request::new("sys".to_string(), Vec::new())
    }

    #[tokio::test]
    async fn a_transient_failure_is_absorbed_rather_than_ending_the_run() {
        // The case this module exists for: one 429 twenty steps into a run.
        let e = Resilient::with_policy(Box::new(Flaky::new(1)), fast());
        assert!(e.complete(&req()).await.is_ok());
    }

    #[tokio::test]
    async fn retrying_is_bounded_by_max_attempts() {
        let flaky = std::sync::Arc::new(Flaky::new(usize::MAX));
        let e = Resilient::with_policy(Box::new(FlakyRef(flaky.clone())), fast());

        assert!(e.complete(&req()).await.is_err());
        assert_eq!(flaky.calls(), 3, "three attempts, not an unbounded loop");
    }

    /// A shared handle to the same `Flaky`, so a test can inspect the call count
    /// after the engine has been boxed into the wrapper.
    struct FlakyRef(std::sync::Arc<Flaky>);

    #[async_trait]
    impl Engine for FlakyRef {
        async fn complete(&self, req: &Request) -> Result<Response> {
            self.0.complete(req).await
        }
        fn name(&self) -> &str {
            self.0.name()
        }
        fn supports_native_tools(&self) -> bool {
            true
        }
    }

    #[tokio::test]
    async fn a_dead_endpoint_stops_being_called_at_all() {
        let flaky = std::sync::Arc::new(Flaky::new(usize::MAX));
        let e = Resilient::with_policy(Box::new(FlakyRef(flaky.clone())), fast());

        // Two failed calls at trip_after = 2 opens the breaker.
        assert!(e.complete(&req()).await.is_err());
        assert!(e.complete(&req()).await.is_err());
        let before = flaky.calls();
        assert!(e.is_open(), "the breaker should be open");

        let err = e.complete(&req()).await.unwrap_err();
        assert!(format!("{err:#}").contains("circuit is open"), "{err:#}");
        assert_eq!(flaky.calls(), before, "no call may reach a dead endpoint");
        assert_eq!(e.short_circuited(), 1);
    }

    #[tokio::test]
    async fn the_breaker_lets_one_call_through_after_the_cooldown() {
        // Two failures then success, so the trial call finds a working endpoint.
        let e = Resilient::with_policy(Box::new(Flaky::new(6)), fast());

        assert!(e.complete(&req()).await.is_err());
        assert!(e.complete(&req()).await.is_err());
        assert!(e.is_open());

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!e.is_open(), "the cooldown has passed");
        assert!(e.complete(&req()).await.is_ok(), "the trial call should succeed");
    }

    #[tokio::test]
    async fn a_success_clears_the_failure_count() {
        // Otherwise failures accumulated across an entire long run would
        // eventually trip a breaker on an endpoint that is working fine.
        let e = Resilient::with_policy(Box::new(Flaky::new(1)), fast());
        assert!(e.complete(&req()).await.is_ok());

        assert!(
            !e.is_open(),
            "one absorbed blip must not leave the breaker part-way to open"
        );
    }

    /// Attempts made against an engine that always fails this way.
    async fn attempts(failure: Failure) -> usize {
        let flaky = std::sync::Arc::new(Flaky::always(failure));
        let e = Resilient::with_policy(Box::new(FlakyRef(flaky.clone())), fast());
        assert!(e.complete(&req()).await.is_err());
        flaky.calls()
    }

    #[tokio::test]
    async fn a_wrong_api_key_is_not_retried() {
        assert_eq!(
            attempts(Failure::Status(401)).await,
            1,
            "retrying a bad key just wastes the wait"
        );
    }

    #[tokio::test]
    async fn a_rate_limit_is_retried() {
        assert_eq!(
            attempts(Failure::Status(429)).await,
            3,
            "a 429 is exactly what retrying is for"
        );
    }

    #[tokio::test]
    async fn a_connection_refused_is_retried() {
        // Ollama not being up yet is transient in the way that matters.
        assert_eq!(attempts(Failure::Transport).await, 3);
    }

    /// The half of the classification that lives in this module: an error that
    /// is not an `EngineError` is retried rather than guessed about.
    #[tokio::test]
    async fn an_error_this_module_cannot_read_is_retried_not_assumed_permanent() {
        assert_eq!(attempts(Failure::Untyped).await, 3);
    }

    #[tokio::test]
    async fn the_wrapper_is_transparent_about_what_it_wraps() {
        // Talos logs the engine name into the trace; a wrapper that renamed it
        // would make every trace say "resilient" instead of the model.
        let e = Resilient::new(Box::new(Flaky::new(0)));
        assert_eq!(e.name(), "flaky");
        assert!(e.supports_native_tools());
    }
}
