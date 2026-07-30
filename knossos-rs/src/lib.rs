//! # Daedalus Harness
//!
//! An agentic coding harness whose components are the system-level form of the
//! mechanisms in the Daedalus model architecture.
//!
//! The engine is a swappable slot. Everything around it — exact symbol memory,
//! tiered verification, the constitution, the halting policy — is
//! engine-agnostic by construction and survives an engine swap.
//!
//! | Component | Model-level origin | The design decision it forces |
//! |---|---|---|
//! | [`scribe`] | exact symbol table | identifiers are injected verbatim, never summarized |
//! | [`themis`] | always-on shared expert | the constitution applies to every call |
//! | [`ariadne`] | PonderNet halting | explicit stopping pressure and a hard ceiling |
//! | [`oracle`] | — | deterministic tiers before any model judgement |
//! | [`metis`] / [`talos`] | — | the plan is an artifact separate from execution |
//!
//! Mappings that would only rename a standard pattern were left out.

pub mod acp;
pub mod argus;
pub mod ariadne;
pub mod config;
pub mod delegate;
pub mod diff;
pub mod engine;
pub mod gate;
pub mod hooks;
pub mod interject;
pub mod jsonrpc;
pub mod lethe;
pub mod lsp;
pub mod mcp;
pub mod metis;
pub mod mnemosyne;
pub mod oracle;
pub mod repl;
pub mod resilience;
pub mod sandbox;
pub mod scribe;
pub mod serve;
pub mod session;
pub mod talos;
pub mod themis;
pub mod tools;

pub use config::Config;
pub use engine::Engine;
