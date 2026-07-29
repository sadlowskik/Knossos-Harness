//! The engine slot.
//!
//! Daedalus is the *system*; the reasoning engine is swappable. This trait is
//! that swap point. Two implementations ship in v1 — Anthropic's Messages API
//! and a local Ollama server — plus a scripted mock so the whole test suite
//! runs without network.
//!
//! `supports_native_tools()` is not decoration. Anthropic has first-class tool
//! use; Ollama's varies by model, and some have none at all. Rather than
//! pretend the backends are identical, engines that lack native tool calling
//! declare it, and the harness falls back to a prompted-JSON protocol
//! (`prompt_fallback`) that it parses itself.

pub mod anthropic;
pub mod error;
pub mod mock;
pub mod ollama;
pub mod prompt_fallback;
pub mod types;

use anyhow::Result;
use async_trait::async_trait;

pub use error::EngineError;
pub use types::{
    Content, Message, Request, Response, Role, StopReason, ToolDef, Usage,
};

#[async_trait]
pub trait Engine: Send + Sync {
    /// One turn. Implementations must translate `Request` into their own wire
    /// format and the reply back into `Response` — no provider types escape.
    async fn complete(&self, req: &Request) -> Result<Response>;

    /// Human-readable identifier, e.g. `anthropic:claude-opus-5`. Goes into
    /// the trace log so a trajectory records which engine produced it.
    fn name(&self) -> &str;

    /// Whether the backend can be handed tool definitions directly. When
    /// false, callers should route through `prompt_fallback`.
    fn supports_native_tools(&self) -> bool;
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
