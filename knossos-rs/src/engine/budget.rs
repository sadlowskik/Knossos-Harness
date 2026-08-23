//! Shared local quota and concurrency enforcement.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::Semaphore;

use super::{Engine, EngineError, Request, Response, StreamDelta};

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct QuotaSnapshot {
    pub requests: u64,
    pub tokens: u64,
}

#[derive(Debug, Default)]
struct State {
    requests: u64,
    tokens: u64,
    reserved_tokens: u64,
}

pub struct Quota {
    max_requests: Option<u64>,
    max_tokens: Option<u64>,
    state: Mutex<State>,
    semaphore: Semaphore,
}

impl Quota {
    pub fn new(
        max_requests: Option<u64>,
        max_tokens: Option<u64>,
        concurrency: usize,
    ) -> Arc<Self> {
        Self::with_spent(
            max_requests,
            max_tokens,
            concurrency,
            QuotaSnapshot::default(),
        )
    }

    pub fn with_spent(
        max_requests: Option<u64>,
        max_tokens: Option<u64>,
        concurrency: usize,
        spent: QuotaSnapshot,
    ) -> Arc<Self> {
        Arc::new(Self {
            max_requests,
            max_tokens,
            state: Mutex::new(State {
                requests: spent.requests,
                tokens: spent.tokens,
                reserved_tokens: 0,
            }),
            semaphore: Semaphore::new(concurrency.max(1)),
        })
    }

    pub fn snapshot(&self) -> QuotaSnapshot {
        let state = self.state.lock().expect("quota");
        QuotaSnapshot {
            requests: state.requests,
            tokens: state.tokens,
        }
    }

    fn reserve(&self, request: &Request) -> Result<u64> {
        let request_tokens =
            (serde_json::to_vec(request).map(|v| v.len()).unwrap_or(0) as u64).div_ceil(4);
        // Reserve the full possible completion. This makes the cap strict even
        // with concurrent calls; unused capacity is refunded after the reply.
        let reservation = request_tokens.saturating_add(request.max_tokens as u64);
        let mut state = self.state.lock().expect("quota");
        if self.max_requests.is_some_and(|max| state.requests >= max) {
            return Err(EngineError::Budget {
                kind: "request",
                detail: format!("{} request(s) already spent", state.requests),
            }
            .into());
        }
        if self.max_tokens.is_some_and(|max| {
            state
                .tokens
                .saturating_add(state.reserved_tokens)
                .saturating_add(reservation)
                > max
        }) {
            return Err(EngineError::Budget {
                kind: "token",
                detail: format!(
                    "{} spent + {} reserved + {} requested",
                    state.tokens, state.reserved_tokens, reservation
                ),
            }
            .into());
        }
        state.requests += 1;
        state.reserved_tokens += reservation;
        Ok(reservation)
    }

    fn finish(&self, reservation: u64, response: Option<&Response>) {
        let mut state = self.state.lock().expect("quota");
        state.reserved_tokens = state.reserved_tokens.saturating_sub(reservation);
        if let Some(response) = response {
            let measured = response.usage.input_tokens + response.usage.output_tokens;
            let fallback = (response.text().len() as u64).div_ceil(4);
            state.tokens = state.tokens.saturating_add(measured.max(fallback));
        }
    }
}

pub struct BudgetedEngine {
    inner: Box<dyn Engine>,
    quota: Arc<Quota>,
}

impl BudgetedEngine {
    pub fn new(inner: Box<dyn Engine>, quota: Arc<Quota>) -> Self {
        Self { inner, quota }
    }
}

#[async_trait]
impl Engine for BudgetedEngine {
    async fn complete(&self, request: &Request) -> Result<Response> {
        let _permit = self
            .quota
            .semaphore
            .acquire()
            .await
            .map_err(|_| EngineError::Budget {
                kind: "concurrency",
                detail: "quota was closed".into(),
            })?;
        let reservation = self.quota.reserve(request)?;
        let result = self.inner.complete(request).await;
        self.quota.finish(reservation, result.as_ref().ok());
        result
    }

    async fn complete_stream(
        &self,
        request: &Request,
        on_delta: &(dyn Fn(StreamDelta) + Send + Sync),
    ) -> Result<Response> {
        let _permit = self
            .quota
            .semaphore
            .acquire()
            .await
            .map_err(|_| EngineError::Budget {
                kind: "concurrency",
                detail: "quota was closed".into(),
            })?;
        let reservation = self.quota.reserve(request)?;
        let result = self.inner.complete_stream(request, on_delta).await;
        self.quota.finish(reservation, result.as_ref().ok());
        result
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
    fn supports_native_tools(&self) -> bool {
        self.inner.supports_native_tools()
    }
    fn restore_limits(&self) -> bool {
        self.inner.restore_limits()
    }
    fn context_window(&self) -> Option<u32> {
        self.inner.context_window()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{mock::MockEngine, Message};

    #[tokio::test]
    async fn request_budget_fails_closed_before_an_extra_call() {
        let quota = Quota::new(Some(1), None, 1);
        let engine = BudgetedEngine::new(
            Box::new(MockEngine::from_script(vec!["one".into(), "two".into()])),
            quota,
        );
        let request = Request::new("s", vec![Message::user_text("u")]).with_max_tokens(8);
        assert_eq!(engine.complete(&request).await.unwrap().text(), "one");
        let error = engine.complete(&request).await.unwrap_err();
        assert!(matches!(
            error.downcast_ref::<EngineError>(),
            Some(EngineError::Budget { .. })
        ));
    }

    #[tokio::test]
    async fn token_reservation_prevents_overshoot() {
        let quota = Quota::new(None, Some(4), 1);
        let engine = BudgetedEngine::new(Box::new(MockEngine::text("never")), quota);
        let request = Request::new("system", vec![Message::user_text("user")]).with_max_tokens(8);
        assert!(engine.complete(&request).await.is_err());
    }
}
