//! Runtime configuration: which engine, which workspace, what budgets.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use crate::engine::{anthropic, ollama, Engine};

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum EngineKind {
    Anthropic,
    Ollama,
    /// No engine. Lets `index` and `verify` run with nothing configured.
    None,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub engine: EngineKind,
    pub model: Option<String>,
    pub anthropic_base_url: String,
    pub ollama_base_url: String,
    /// Set false for a local model whose tool calling is unreliable; the
    /// harness then uses the prompted-JSON shim.
    pub ollama_native_tools: bool,
    pub workspace: PathBuf,
    /// Ariadne's hard ceiling — the forced halt.
    pub max_steps: usize,
    /// Ariadne's target mean depth; pressure begins past this point.
    pub target_steps: usize,
    pub max_tokens: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            engine: EngineKind::Anthropic,
            model: None,
            anthropic_base_url: env_or("ANTHROPIC_BASE_URL", anthropic::DEFAULT_BASE_URL),
            ollama_base_url: env_or("OLLAMA_HOST", ollama::DEFAULT_BASE_URL),
            ollama_native_tools: true,
            workspace: PathBuf::from("."),
            // Matches `Ariadne::default()` and Python's `acp.py` default. Raised
            // from 12 on measurement; `target_steps` stays where it was so
            // pressure starts in the same place with further to escalate.
            max_steps: 20,
            target_steps: 6,
            max_tokens: 8192,
        }
    }
}

impl Config {
    /// Instantiate the configured backend.
    ///
    /// The only place a concrete engine type is named. Everything downstream
    /// holds `Box<dyn Engine>`, which is what makes the slot swappable.
    pub fn build_engine(&self) -> Result<Box<dyn Engine>> {
        match self.engine {
            EngineKind::Anthropic => {
                let key = std::env::var("ANTHROPIC_API_KEY").context(
                    "ANTHROPIC_API_KEY is not set (use --engine ollama for a local model)",
                )?;
                let model = self
                    .model
                    .clone()
                    .unwrap_or_else(|| env_or("DAEDALUS_MODEL", anthropic::DEFAULT_MODEL));
                Ok(Box::new(
                    anthropic::AnthropicEngine::new(key, model)
                        .with_base_url(&self.anthropic_base_url),
                ))
            }
            EngineKind::Ollama => {
                let model = self
                    .model
                    .clone()
                    .unwrap_or_else(|| env_or("DAEDALUS_MODEL", ollama::DEFAULT_MODEL));
                let mut e = ollama::OllamaEngine::new(model).with_base_url(&self.ollama_base_url);
                if !self.ollama_native_tools {
                    e = e.without_native_tools();
                }
                Ok(Box::new(e))
            }
            EngineKind::None => bail!("this command needs an engine; pass --engine"),
        }
    }

    /// Canonical workspace root. Every file tool is jailed inside it, so it
    /// must resolve before anything else runs.
    pub fn workspace_root(&self) -> Result<PathBuf> {
        std::fs::canonicalize(&self.workspace).with_context(|| {
            format!("workspace does not exist: {}", self.workspace.display())
        })
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}
