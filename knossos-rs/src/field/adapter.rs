//! The harness adapter contract. Port of `field/server/src/harness/adapter.js`.
//!
//! An adapter owns one real agent process and translates its native output
//! into Field's canonical event vocabulary, so the registry, the projection
//! and the director treat every harness alike. Two "kind" axes exist and
//! must not be conflated: the endpoint kind (anthropic, openai-compatible,
//! cameo) describes the inference backend; the harness kind describes how
//! Field launches and speaks to the agent process.

use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;

/// The canonical Field event kinds every adapter emits; mirrors the `kind`
/// enum in `contracts/field-event-v1.schema.json`.
pub const CANONICAL_EVENTS: [&str; 14] = [
    "session.state",
    "session.message",
    "session.thinking",
    "session.tool_use",
    "session.tool_result",
    "session.usage",
    "session.progress",
    "session.turn_complete",
    "session.verification",
    "session.delegated",
    "session.ended",
    "endpoint.routed",
    "browser.navigated",
    "harness.permission_requested",
];

pub const HARNESS_KINDS: [&str; 5] = ["cli-stream-json", "ndjson-serve", "acp", "http-api", "mcp"];

/// Where an adapter's events go: `(kind, data)`, with `sessionId` already
/// stamped into `data`.
pub type EventSink = Arc<dyn Fn(&str, Value) + Send + Sync>;

/// A hook the registry attaches so admission and budget checks run before
/// every (re)start; an error refuses the start.
pub type BeforeStart = Arc<dyn Fn() -> Result<(), String> + Send + Sync>;

/// The mutable identity of a live session, as the registry sees it.
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub id: String,
    pub agent_id: Option<String>,
    pub name: String,
    pub role: Option<String>,
    pub model: Option<String>,
    pub endpoint_id: Option<String>,
    pub effort: String,
    pub cwd: PathBuf,
    pub workspace_id: Option<String>,
    pub state: String,
}

/// What the registry can do with a session, whichever harness backs it.
pub trait Adapter: Send + Sync {
    fn info(&self) -> SessionInfo;
    fn set_endpoint(&self, endpoint_id: Option<String>, model: Option<String>);
    fn set_effort(&self, effort: &str);
    /// A process (or a process still draining) is owned right now.
    fn is_running(&self) -> bool;
    /// Owned processes, including ones still draining after a stop.
    fn owned_processes(&self) -> usize;
    fn set_before_start(&self, hook: BeforeStart);
    fn start(&self, orders: Option<String>) -> Result<(), String>;
    fn send(&self, text: &str) -> bool;
    fn pause(&self) -> bool;
    fn resume(&self, orders: Option<String>) -> Result<bool, String>;
    fn cancel(&self) -> bool;
    fn decide_permission(&self, request_id: &Value, decision: &str) -> bool;
    fn capabilities(&self) -> Value;
}

/// Stamp `sessionId` into an event payload, the canonical emission point.
pub fn stamp(session_id: &str, data: Value) -> Value {
    let mut out = json!({ "sessionId": session_id });
    if let (Value::Object(o), Value::Object(d)) = (&mut out, data) {
        for (k, v) in d {
            o.insert(k, v);
        }
    }
    out
}
