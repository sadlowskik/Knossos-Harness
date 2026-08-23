//! Crash-safe suite progress for paid/quota-limited evaluations.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

pub const SCHEMA: &str = "knossos-eval-checkpoint/v1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvalCheckpoint {
    pub schema: String,
    pub suite_digest: String,
    pub experiment_arm: String,
    pub completed: BTreeSet<String>,
    pub passed: BTreeSet<String>,
    pub requests_spent: u64,
    pub tokens_spent: u64,
}

impl EvalCheckpoint {
    pub fn new(suite_digest: impl Into<String>, arm: impl Into<String>) -> Self {
        Self {
            schema: SCHEMA.into(),
            suite_digest: suite_digest.into(),
            experiment_arm: arm.into(),
            completed: BTreeSet::new(),
            passed: BTreeSet::new(),
            requests_spent: 0,
            tokens_spent: 0,
        }
    }

    pub fn load(path: &Path, digest: &str, arm: &str) -> Result<Self> {
        let value: Self = serde_json::from_slice(
            &std::fs::read(path)
                .with_context(|| format!("reading eval checkpoint {}", path.display()))?,
        )?;
        if value.schema != SCHEMA || value.suite_digest != digest || value.experiment_arm != arm {
            bail!("checkpoint does not match this suite digest and experiment arm");
        }
        Ok(value)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let temp = path.with_extension("tmp");
        let bytes = serde_json::to_vec_pretty(self)?;
        {
            use std::io::Write;
            let mut file = std::fs::File::create(&temp)?;
            file.write_all(&bytes)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
        }
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        std::fs::rename(&temp, path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_is_bound_to_suite_and_arm() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("checkpoint.json");
        let mut value = EvalCheckpoint::new("digest", "full");
        value.completed.insert("case-a".into());
        value.save(&path).unwrap();
        assert_eq!(
            EvalCheckpoint::load(&path, "digest", "full").unwrap(),
            value
        );
        assert!(EvalCheckpoint::load(&path, "other", "full").is_err());
        assert!(EvalCheckpoint::load(&path, "digest", "other").is_err());
    }
}
