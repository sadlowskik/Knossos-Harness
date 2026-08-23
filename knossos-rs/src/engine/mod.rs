//! The engine slot.
//!
//! Daedalus is the *system*; the reasoning engine is swappable. This trait is
//! that swap point. Implementations ship for Anthropic's Messages API, a
//! local Ollama server, an OpenAI-compatible `/chat/completions` adapter
//! (Groq, Gemini, OpenRouter, …), a Cameo node that discovers `/api/engines`
//! and fail-closes unless the model is resident (or the operator seam can
//! ensure it), and a scripted mock so the whole test suite runs without network.
//!
//! `supports_native_tools()` is not decoration. Anthropic has first-class tool
//! use; Ollama's varies by model, and some have none at all. Rather than
//! pretend the backends are identical, engines that lack native tool calling
//! declare it, and the harness falls back to a prompted-JSON protocol
//! (`prompt_fallback`) that it parses itself.

pub mod anthropic;
pub mod budget;
pub mod cameo;
pub mod error;
pub mod mock;
pub mod ollama;
pub mod openai;
pub mod prompt_fallback;
pub mod retrieval;
pub mod sse;
pub mod types;

use anyhow::Result;
use async_trait::async_trait;

pub use error::EngineError;
pub use types::{
    Content, Message, Request, Response, Role, StopReason, StreamDelta, ToolDef, Usage,
};

#[async_trait]
pub trait Engine: Send + Sync {
    /// One turn. Implementations must translate `Request` into their own wire
    /// format and the reply back into `Response` — no provider types escape.
    async fn complete(&self, req: &Request) -> Result<Response>;

    /// One turn, with live text. Default is [`complete`](Self::complete) and
    /// no deltas, so `MockEngine` and tests that never opted into streaming
    /// keep the one-shot path.
    ///
    /// Implementations that stream must still return a finished
    /// [`Response`] whose text is the concatenation of every `Text` delta,
    /// and must not put a `ToolUse` on the wire until this future resolves.
    async fn complete_stream(
        &self,
        req: &Request,
        on_delta: &(dyn Fn(StreamDelta) + Send + Sync),
    ) -> Result<Response> {
        let _ = on_delta;
        self.complete(req).await
    }

    /// Human-readable identifier, e.g. `anthropic:claude-opus-5`. Goes into
    /// the trace log so a trajectory records which engine produced it.
    fn name(&self) -> &str;

    /// Whether the backend can be handed tool definitions directly. When
    /// false, callers should route through `prompt_fallback`.
    fn supports_native_tools(&self) -> bool;

    /// Undo a transient `max_tokens` shrink. True if anything changed.
    ///
    /// A 413 is a per-minute window that reopens; a 400 reply-cap is a
    /// property of the model and must not be restored past. Called at a
    /// case boundary, never mid-run. Default is a no-op so a backend
    /// that does not shrink does not have to know this exists.
    fn restore_limits(&self) -> bool {
        false
    }

    /// The server's context window, when the engine has learned it.
    ///
    /// Distinct from the model's advertised maximum: a card saying 256K
    /// is irrelevant if the server was started with 16K. `None` means
    /// unknown, which is deliberately not "unlimited".
    fn context_window(&self) -> Option<u32> {
        None
    }
}

/// Run a request against an engine, transparently handling the prompted-JSON
/// path for backends without native tool use.
///
/// This is the only place the two paths are reconciled, so callers upstream
/// (Talos, Metis) can stay ignorant of which kind of engine they hold.
pub async fn complete(engine: &dyn Engine, req: &Request) -> Result<Response> {
    if engine.supports_native_tools() || req.tools.is_empty() {
        return engine.complete(req).await;
    }

    let mut shimmed = req.clone();
    shimmed.system = prompt_fallback::augment_system(&req.system, &req.tools);
    shimmed.tools = Vec::new();

    let mut resp = engine.complete(&shimmed).await?;
    prompt_fallback::extract_tool_calls(&mut resp);
    Ok(resp)
}

/// Streaming twin of [`complete`]. Prompt-fallback still runs on the **final**
/// text, never on a partial chunk.
pub async fn complete_stream(
    engine: &dyn Engine,
    req: &Request,
    on_delta: &(dyn Fn(StreamDelta) + Send + Sync),
) -> Result<Response> {
    if engine.supports_native_tools() || req.tools.is_empty() {
        return engine.complete_stream(req, on_delta).await;
    }

    let mut shimmed = req.clone();
    shimmed.system = prompt_fallback::augment_system(&req.system, &req.tools);
    shimmed.tools = Vec::new();

    let mut resp = engine.complete_stream(&shimmed, on_delta).await?;
    prompt_fallback::extract_tool_calls(&mut resp);
    Ok(resp)
}
