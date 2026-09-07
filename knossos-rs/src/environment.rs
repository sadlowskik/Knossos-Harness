//! Reproducible, explicitly host-mode mission environment discovery.
//!
//! Discovery is read-only except for an operator-requested export. It never
//! runs a suggested setup command, so a lockfile cannot silently authorize a
//! registry request or a write to a package cache.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};

use crate::mission::EnvironmentRecord;

const INSTRUCTIONS: &[&str] = &["AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md"];
const LOCKFILES: &[&str] = &[
    "Cargo.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "requirements.txt",
    "go.sum",
];
const MARKERS: &[&str] = &[
    "Cargo.toml",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "go.mod",
    ".git",
];
/// Enough for ordinary dependency locks, while still bounding a hostile file.
const MAX_FINGERPRINT_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// The record is kept separately from `MissionState` so callers can inspect
/// and explicitly export it before a model gets a chance to plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvironmentFingerprint(EnvironmentRecord);

impl EnvironmentFingerprint {
    pub fn discover(root: &Path) -> Result<Self> {
        let root = root
            .canonicalize()
            .with_context(|| format!("cannot fingerprint workspace {}", root.display()))?;
        if !root.is_dir() {
            bail!("workspace is not a directory: {}", root.display());
        }

        let instruction_files = present(&root, INSTRUCTIONS);
        let lockfiles = present(&root, LOCKFILES);
        let workspace_markers = present(&root, MARKERS);
        // Running `cargo --version` or `python --version` is not a harmless
        // read: PATH can resolve to a workspace-controlled executable or a
        // platform app shim. Probe execution belongs behind a later explicit
        // capability. Do not inherit credentials or turn reconnaissance into
        // arbitrary host code execution merely to decorate a mission record.
        let runtime_versions = BTreeMap::from([(
            "host_runtime_probe".into(),
            "not run during untrusted workspace discovery".into(),
        )]);
        let source_names: Vec<String> = instruction_files
            .iter()
            .chain(lockfiles.iter())
            .chain(workspace_markers.iter())
            .filter(|name| name.as_str() != ".git")
            .cloned()
            .collect();
        let source_hashes = hashes(&root, source_names.iter())?;

        let (setup_recipe, verification_recipe) = recipes(&workspace_markers, &lockfiles);
        let mut discovery_notes = vec![
            "adapter=host: commands run directly on the current machine".into(),
            "isolation=none: discovery does not claim a container, VM, or sandbox".into(),
            "recipes are descriptive and require a separate approved execution capability".into(),
        ];
        if !workspace_markers.iter().any(|marker| marker == ".git") {
            discovery_notes.push("Git metadata was not found at the workspace root".into());
        }

        Ok(Self(EnvironmentRecord {
            adapter: "host".into(),
            isolation: "none".into(),
            os: std::env::consts::OS.into(),
            architecture: std::env::consts::ARCH.into(),
            runtime_versions,
            source_hashes,
            instruction_files,
            lockfiles,
            workspace_markers,
            setup_recipe,
            verification_recipe,
            setup_status: "not_run_requires_explicit_capability".into(),
            verification_status: "pending".into(),
            verification_evidence_hash: None,
            discovery_notes,
        }))
    }

    pub fn into_record(self) -> EnvironmentRecord {
        self.0
    }

    pub fn record(&self) -> &EnvironmentRecord {
        &self.0
    }

    /// Compare only the durable, source-derived parts of a previous record.
    /// This is safe to run before a resumed turn because it reads bounded files
    /// and never launches a program from PATH or the workspace.
    pub fn unchanged_since(root: &Path, previous: &EnvironmentRecord) -> Result<bool> {
        let current = Self::discover(root)?;
        Ok(current.0.adapter == previous.adapter
            && current.0.isolation == previous.isolation
            && current.0.os == previous.os
            && current.0.architecture == previous.architecture
            && current.0.runtime_versions == previous.runtime_versions
            && current.0.instruction_files == previous.instruction_files
            && current.0.lockfiles == previous.lockfiles
            && current.0.workspace_markers == previous.workspace_markers
            && current.0.source_hashes == previous.source_hashes)
    }

    /// Export only to a new file. Replacing a previous recipe obscures the
    /// environment that produced earlier evidence.
    pub fn export(&self, path: &Path) -> Result<()> {
        if path.exists() {
            bail!(
                "refusing to overwrite environment export {}",
                path.display()
            );
        }
        let parent = path
            .parent()
            .context("environment export has no parent directory")?;
        let parent = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        if !parent.is_dir() {
            bail!(
                "environment export parent does not exist: {}",
                parent.display()
            );
        }
        let canonical_parent = parent.canonicalize()?;
        let target = canonical_parent.join(
            path.file_name()
                .context("environment export needs a file name")?,
        );
        let bytes = serde_json::to_vec_pretty(&self.0)?;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&target)
            .with_context(|| format!("creating environment export {}", target.display()))?;
        use std::io::Write;
        file.write_all(&bytes)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        Ok(())
    }
}

fn present(root: &Path, names: &[&str]) -> Vec<String> {
    names
        .iter()
        .filter(|name| root.join(name).exists())
        .map(|name| (*name).to_string())
        .collect()
}

fn hashes<'a>(
    root: &Path,
    names: impl Iterator<Item = &'a String>,
) -> Result<BTreeMap<String, String>> {
    let mut hashes = BTreeMap::new();
    for name in names {
        let path = root.join(name);
        let metadata = std::fs::symlink_metadata(&path)?;
        if !metadata.file_type().is_file() {
            bail!(
                "fingerprint input is not a regular file: {}",
                path.display()
            );
        }
        if metadata.len() > MAX_FINGERPRINT_FILE_BYTES {
            bail!(
                "fingerprint input exceeds the {} byte limit: {}",
                MAX_FINGERPRINT_FILE_BYTES,
                path.display()
            );
        }
        let mut file = std::fs::File::open(&path)?;
        let mut hash = Sha256::new();
        let mut total = 0u64;
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            total = total.saturating_add(read as u64);
            if total > MAX_FINGERPRINT_FILE_BYTES {
                bail!(
                    "fingerprint input grew beyond its {} byte limit: {}",
                    MAX_FINGERPRINT_FILE_BYTES,
                    path.display()
                );
            }
            hash.update(&buffer[..read]);
        }
        let after = std::fs::symlink_metadata(&path)?;
        if after.len() != metadata.len() || after.modified().ok() != metadata.modified().ok() {
            bail!(
                "fingerprint input changed while it was read: {}",
                path.display()
            );
        }
        hashes.insert(name.clone(), format!("sha256:{:x}", hash.finalize()));
    }
    Ok(hashes)
}

fn recipes(markers: &[String], locks: &[String]) -> (Vec<String>, Vec<String>) {
    let mut setup = Vec::new();
    let mut verify = Vec::new();
    if markers.iter().any(|marker| marker == "Cargo.toml") {
        setup.push(if locks.iter().any(|lock| lock == "Cargo.lock") {
            "cargo fetch --locked (network; writes Cargo cache)".into()
        } else {
            "cargo fetch (network; writes Cargo cache; no lockfile was found)".into()
        });
        verify.push(if locks.iter().any(|lock| lock == "Cargo.lock") {
            "cargo test --locked".into()
        } else {
            "cargo test".into()
        });
    }
    if markers.iter().any(|marker| marker == "package.json") {
        setup.push(if locks.iter().any(|lock| lock == "package-lock.json") {
            "npm ci (network; writes node_modules and npm cache)".into()
        } else if locks.iter().any(|lock| lock == "pnpm-lock.yaml") {
            "pnpm install --frozen-lockfile (network; writes dependency store)".into()
        } else if locks.iter().any(|lock| lock == "yarn.lock") {
            "yarn install --frozen-lockfile (network; writes node_modules/cache)".into()
        } else {
            "package manager install (network; lockfile missing, result is not reproducible)".into()
        });
        verify.push("npm test (confirm project script before execution)".into());
    }
    if markers.iter().any(|marker| marker == "pyproject.toml")
        || markers.iter().any(|marker| marker == "requirements.txt")
    {
        setup.push("create an isolated Python environment and install the declared lock/requirements (network and cache writes)".into());
        verify.push("pytest (when the project declares it)".into());
    }
    if markers.iter().any(|marker| marker == "go.mod") {
        setup.push("go mod download (network; writes module cache)".into());
        verify.push("go test ./...".into());
    }
    if setup.is_empty() {
        setup.push(
            "no supported package manifest discovered; inspect project instructions before setup"
                .into(),
        );
        verify.push(
            "no default verifier inferred; inspect project instructions before verification".into(),
        );
    }
    (setup, verify)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_workspace_has_explicit_nonexecuting_recipe() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
        std::fs::write(dir.path().join("Cargo.lock"), "# locked\n").unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "instructions\n").unwrap();
        let record = EnvironmentFingerprint::discover(dir.path())
            .unwrap()
            .into_record();
        assert_eq!(record.adapter, "host");
        assert_eq!(record.isolation, "none");
        assert_eq!(record.instruction_files, ["AGENTS.md"]);
        assert!(record.source_hashes["AGENTS.md"].starts_with("sha256:"));
        assert!(record.source_hashes.contains_key("Cargo.toml"));
        assert_eq!(
            record.runtime_versions["host_runtime_probe"],
            "not run during untrusted workspace discovery"
        );
        assert!(record
            .setup_recipe
            .iter()
            .any(|line| line.contains("cargo fetch --locked")));
        assert!(record
            .verification_recipe
            .iter()
            .any(|line| line == "cargo test --locked"));
    }

    #[test]
    fn export_is_json_and_never_overwrites() {
        let dir = tempfile::tempdir().unwrap();
        let fingerprint = EnvironmentFingerprint::discover(dir.path()).unwrap();
        let path = dir.path().join("environment.json");
        fingerprint.export(&path).unwrap();
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(value["adapter"], "host");
        assert!(fingerprint.export(&path).is_err());
    }

    #[test]
    fn drift_detects_instruction_or_lockfile_changes_without_running_tools() {
        let dir = tempfile::tempdir().unwrap();
        let instructions = dir.path().join("AGENTS.md");
        std::fs::write(&instructions, "first\n").unwrap();
        let previous = EnvironmentFingerprint::discover(dir.path())
            .unwrap()
            .into_record();
        assert!(EnvironmentFingerprint::unchanged_since(dir.path(), &previous).unwrap());
        std::fs::write(&instructions, "changed\n").unwrap();
        assert!(!EnvironmentFingerprint::unchanged_since(dir.path(), &previous).unwrap());
    }

    #[test]
    fn a_common_sized_lockfile_is_stream_hashed_and_manifest_drift_is_detected() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname=\"x\"\n").unwrap();
        std::fs::write(dir.path().join("Cargo.lock"), vec![b'x'; 1024 * 1024]).unwrap();
        let previous = EnvironmentFingerprint::discover(dir.path())
            .unwrap()
            .into_record();
        assert!(EnvironmentFingerprint::unchanged_since(dir.path(), &previous).unwrap());
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname=\"changed\"\n",
        )
        .unwrap();
        assert!(!EnvironmentFingerprint::unchanged_since(dir.path(), &previous).unwrap());
    }
}
