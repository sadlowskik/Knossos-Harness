//! Field's core, ported from the Node server one responsibility at a time.
//!
//! Field is the operator surface: many agents, many workspaces, one map. Its
//! spine is an append-only event log; every piece of operational state is a
//! fold over that log, and nothing operational is ever updated in place. The
//! Node server (`field/server`) is the reference implementation while the port
//! is in progress; each module here carries the Node file it replaces and the
//! tests that were its parity oracle.
//!
//! Order of the port (strangler): event log and replay (this commit), then
//! the projection fold, then the API and WebSocket hub, then the adapters,
//! then the director and routines. The web client and the `field-event-v1`
//! vocabulary do not change.

pub mod acp_session;
pub mod adapter;
pub mod adapters;
pub mod budget_ledger;
pub mod campaign_projection;
pub mod child_env;
pub mod cities;
pub mod claude_session;
pub mod config;
pub mod director;
pub mod eventlog;
pub mod git;
pub mod graph_projection;
pub mod hub;
pub mod js;
pub mod knossos_session;
pub mod model;
pub mod page;
pub mod permission_bridge;
pub mod policy;
pub mod projection;
pub mod registry;
pub mod replay;
pub mod routines;
pub mod sanitize;
pub mod security;
pub mod server;
pub mod simulation;
pub mod stores;
pub mod terminal;
pub mod watch;
pub mod workspace;

pub use eventlog::{Event, EventLog, Health, Source, EVENT_LOG_SCHEMA_VERSION};
pub use page::{page_result, parse_event_page, EventPage, PageResult, PaginationError};
pub use projection::{FieldConfig, Projection};
pub use replay::{read_event_range, replay_into, ApplyEvent, Range, ReadEvents, Replayed};
pub use security::{Authority, Bootstrap, ControlSecurity, Denied, RequestFacts, SecurityOptions};
pub use server::{Running, ServerOptions};
