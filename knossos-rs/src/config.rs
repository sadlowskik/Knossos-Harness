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
            ollama_base_url: ollama_host().unwrap_or_else(|| ollama::DEFAULT_BASE_URL.to_string()),
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
    ///
    /// The result is wrapped in [`Resilient`](crate::resilience::Resilient), so
    /// every configured backend retries transient failures and stops calling a
    /// dead one. Wrapped here rather than inside each backend because it is a
    /// property of *using* an engine over a network, not of the wire format —
    /// and doing it once is what keeps the two implementations from drifting
    /// into two different retry policies.
    pub fn build_engine(&self) -> Result<Box<dyn Engine>> {
        Ok(Box::new(crate::resilience::Resilient::new(
            self.build_backend()?,
        )))
    }

    /// The bare backend, without retrying. Exposed for a caller that wants to
    /// choose its own [`Policy`](crate::resilience::Policy).
    pub fn build_backend(&self) -> Result<Box<dyn Engine>> {
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

/// `OLLAMA_HOST` as a base URL, however it was written.
///
/// People write the same address several ways -- `10.0.0.5`, `10.0.0.5:11434`,
/// `http://10.0.0.5:11434` -- and mean one thing by all of them. Python's
/// `_local_base_url` already normalises; this side used the variable raw, so
/// the most natural form to type produced a base URL with no scheme and no
/// port, and the failure surfaced much later as an opaque transport error.
///
/// Returns `None` when the variable is unset or blank, so the caller keeps its
/// own default rather than being handed an empty string.
fn ollama_host() -> Option<String> {
    normalise_host(&std::env::var("OLLAMA_HOST").ok()?)
}

/// The normalising half, kept free of the environment.
///
/// Environment variables are process-global, so a test that sets one races
/// every other test in the binary. Taking the string as an argument makes the
/// rule testable without that.
fn normalise_host(raw: &str) -> Option<String> {
    let raw = raw.trim().trim_end_matches('/');
    if raw.is_empty() {
        return None;
    }

    let with_scheme =
        if raw.contains("://") { raw.to_string() } else { format!("http://{raw}") };

    // One colon is the scheme's own; a second means a port was given.
    Some(if with_scheme.matches(':').count() < 2 {
        format!("{with_scheme}:11434")
    } else {
        with_scheme
    })
}

#[cfg(test)]
mod tests {
    use super::normalise_host;

    #[test]
    fn a_bare_address_gains_a_scheme_and_the_default_port() {
        assert_eq!(
            normalise_host("192.168.4.103").unwrap(),
            "http://192.168.4.103:11434"
        );
    }

    #[test]
    fn an_address_with_a_port_keeps_it() {
        assert_eq!(
            normalise_host("192.168.4.103:11500").unwrap(),
            "http://192.168.4.103:11500"
        );
    }

    #[test]
    fn a_full_url_is_left_alone_but_for_a_trailing_slash() {
        assert_eq!(
            normalise_host("http://192.168.4.103:11434/").unwrap(),
            "http://192.168.4.103:11434"
        );
    }

    #[test]
    fn a_blank_variable_is_not_a_host() {
        // So the caller keeps its own default instead of being handed "".
        assert!(normalise_host("   ").is_none());
        assert!(normalise_host("").is_none());
    }
}
