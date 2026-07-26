//! The tool layer, and the guardrails around it.
//!
//! Two safety properties are structural rather than advisory, because an agent
//! with filesystem and process access needs them from the first commit:
//!
//! 1. **Path jail.** Every filesystem tool resolves paths through
//!    [`ToolCtx::resolve`], which rejects anything landing outside the
//!    workspace root — `../` traversal, absolute paths, and symlinks that
//!    point out.
//! 2. **No shell.** [`shell`] splits commands into program and arguments and
//!    executes them directly. There is no shell interpreter, so `&&`, `|`,
//!    backticks and redirection are inert, and the program name is checked
//!    against an allowlist.
//!
//! Tools also report which files they changed. That is not bookkeeping: it is
//! what lets Ariadne distinguish a step that did work from one that spun, and
//! so decide whether the loop is `Stuck`.

pub mod fs;
pub mod search;
pub mod shell;

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use async_trait::async_trait;

use crate::diff::{diff_file, FileDiff};
use crate::engine::ToolDef;

/// What a tool is allowed to touch.
///
/// In dry-run mode no write reaches disk. Edits are staged in memory instead,
/// and reads consult the staging area first — so a sequence of edits to the
/// same file behaves exactly as it would on disk, and the agent sees its own
/// work. That is what makes a previewed multi-step change trustworthy rather
/// than a guess about what would have happened.
#[derive(Debug, Clone)]
pub struct ToolCtx {
    root: PathBuf,
    dry_run: bool,
    staged: Arc<Mutex<BTreeMap<PathBuf, String>>>,
}

impl ToolCtx {
    /// `root` must already be canonical — see `Config::workspace_root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        ToolCtx {
            root: root.into(),
            dry_run: false,
            staged: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    /// Stage writes in memory instead of applying them.
    pub fn dry_run(mut self) -> Self {
        self.dry_run = true;
        self
    }

    pub fn is_dry_run(&self) -> bool {
        self.dry_run
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Read a file, preferring staged content over what is on disk.
    pub fn read(&self, path: &Path) -> std::io::Result<String> {
        if let Some(staged) = self.staged.lock().unwrap().get(path) {
            return Ok(staged.clone());
        }
        std::fs::read_to_string(path)
    }

    /// Write a file, or stage it when running dry.
    pub fn write(&self, path: &Path, content: &str) -> std::io::Result<()> {
        if self.dry_run {
            self.staged
                .lock()
                .unwrap()
                .insert(path.to_path_buf(), content.to_string());
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)
    }

    /// Staged (path, content) pairs. Empty unless running dry.
    pub fn staged_contents(&self) -> Vec<(PathBuf, String)> {
        self.staged
            .lock()
            .unwrap()
            .iter()
            .map(|(p, c)| (p.clone(), c.clone()))
            .collect()
    }

    /// Diffs for everything staged, against what is currently on disk.
    pub fn diffs(&self) -> Vec<FileDiff> {
        self.staged
            .lock()
            .unwrap()
            .iter()
            .map(|(path, after)| {
                let before = std::fs::read_to_string(path).ok();
                let rel = path.strip_prefix(&self.root).unwrap_or(path);
                diff_file(rel, before.as_deref(), after)
            })
            .filter(|d| !d.is_empty())
            .collect()
    }

    /// Write every staged change to disk and clear the staging area.
    pub fn apply_staged(&self) -> Result<Vec<PathBuf>> {
        let mut staged = self.staged.lock().unwrap();
        let mut written = Vec::new();
        for (path, content) in staged.iter() {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, content)?;
            written.push(path.clone());
        }
        staged.clear();
        Ok(written)
    }

    pub fn discard_staged(&self) {
        self.staged.lock().unwrap().clear();
    }

    /// Resolve a caller-supplied path against the workspace root, refusing
    /// anything that escapes it.
    ///
    /// Works for paths that do not exist yet (a file about to be written), so
    /// it cannot rely on `canonicalize` alone. Two checks run: a lexical one
    /// on the requested path, and a canonical one on the nearest existing
    /// ancestor, which is what catches a symlink pointing out of the tree.
    pub fn resolve(&self, requested: &str) -> Result<PathBuf> {
        let p = Path::new(requested);
        let joined = if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.root.join(p)
        };

        let normalized = normalize(&joined);
        if !normalized.starts_with(&self.root) {
            bail!(
                "path escapes the workspace: {requested} resolves outside {}",
                self.root.display()
            );
        }

        // Symlink check: canonicalize the deepest ancestor that exists.
        let mut probe = normalized.as_path();
        loop {
            if probe.exists() {
                let real = std::fs::canonicalize(probe)?;
                if !real.starts_with(&self.root) {
                    bail!("path escapes the workspace via a symlink: {requested}");
                }
                break;
            }
            match probe.parent() {
                Some(parent) => probe = parent,
                None => break,
            }
        }

        Ok(normalized)
    }

    /// Render a path for display, relative to the root where possible.
    pub fn display(&self, p: &Path) -> String {
        p.strip_prefix(&self.root).unwrap_or(p).display().to_string()
    }
}

/// Lexical path normalization — resolves `.` and `..` without consulting the
/// filesystem, so it works on paths that do not exist yet.
fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[derive(Debug, Clone, Default)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
    /// Files this call created or modified. Feeds Ariadne's `Stuck` check.
    pub changed: Vec<PathBuf>,
}

impl ToolOutput {
    pub fn ok(content: impl Into<String>) -> Self {
        ToolOutput { content: content.into(), is_error: false, changed: Vec::new() }
    }

    pub fn error(content: impl Into<String>) -> Self {
        ToolOutput { content: content.into(), is_error: true, changed: Vec::new() }
    }

    pub fn changed(mut self, path: PathBuf) -> Self {
        self.changed.push(path);
        self
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// JSON Schema for this tool's input.
    fn schema(&self) -> serde_json::Value;
    async fn run(&self, input: &serde_json::Value, ctx: &ToolCtx) -> Result<ToolOutput>;

    fn def(&self) -> ToolDef {
        ToolDef {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.schema(),
        }
    }
}

pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new(tools: Vec<Box<dyn Tool>>) -> Self {
        ToolRegistry { tools }
    }

    /// The v1 tool set: read, write, edit, list, search, run.
    pub fn standard() -> Self {
        ToolRegistry::new(vec![
            Box::new(fs::ReadFile),
            Box::new(fs::WriteFile),
            Box::new(fs::EditFile),
            Box::new(fs::ListDir),
            Box::new(search::Search),
            Box::new(shell::Run::default()),
        ])
    }

    pub fn defs(&self) -> Vec<ToolDef> {
        self.tools.iter().map(|t| t.def()).collect()
    }

    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }

    /// Run a tool by name. An unknown name or a failing tool becomes an error
    /// `ToolOutput` rather than an `Err`, so the engine sees the failure and
    /// can correct itself instead of the loop collapsing.
    pub async fn dispatch(
        &self,
        name: &str,
        input: &serde_json::Value,
        ctx: &ToolCtx,
    ) -> ToolOutput {
        let Some(tool) = self.tools.iter().find(|t| t.name() == name) else {
            return ToolOutput::error(format!(
                "unknown tool `{name}`; available: {}",
                self.names().join(", ")
            ));
        };
        match tool.run(input, ctx).await {
            Ok(out) => out,
            Err(e) => ToolOutput::error(format!("{name} failed: {e:#}")),
        }
    }
}

/// Read a required string field from a tool input object.
pub(crate) fn req_str<'a>(input: &'a serde_json::Value, key: &str) -> Result<&'a str> {
    match input.get(key).and_then(|v| v.as_str()) {
        Some(s) => Ok(s),
        None => bail!("missing required string field `{key}`"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> (tempfile::TempDir, ToolCtx) {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        (dir, ToolCtx::new(root))
    }

    #[test]
    fn resolves_a_plain_relative_path() {
        let (_d, c) = ctx();
        let p = c.resolve("src/main.rs").unwrap();
        assert!(p.starts_with(c.root()));
        assert!(p.ends_with("main.rs"));
    }

    #[test]
    fn rejects_parent_traversal() {
        let (_d, c) = ctx();
        assert!(c.resolve("../secrets.txt").is_err());
        assert!(c.resolve("src/../../secrets.txt").is_err());
        assert!(c.resolve("a/b/c/../../../../out").is_err());
    }

    #[test]
    fn rejects_absolute_paths_outside_root() {
        let (_d, c) = ctx();
        let outside = if cfg!(windows) { "C:\\Windows\\System32\\drivers\\etc\\hosts" } else { "/etc/passwd" };
        assert!(c.resolve(outside).is_err());
    }

    #[test]
    fn allows_absolute_paths_inside_root() {
        let (_d, c) = ctx();
        let inside = c.root().join("nested").join("f.rs");
        assert!(c.resolve(&inside.to_string_lossy()).is_ok());
    }

    #[test]
    fn interior_parent_segments_that_stay_inside_are_fine() {
        let (_d, c) = ctx();
        let p = c.resolve("src/deep/../main.rs").unwrap();
        assert_eq!(p, c.root().join("src").join("main.rs"));
    }

    #[tokio::test]
    async fn unknown_tool_is_an_error_output_not_a_panic() {
        let (_d, c) = ctx();
        let reg = ToolRegistry::standard();
        let out = reg.dispatch("nope", &serde_json::json!({}), &c).await;
        assert!(out.is_error);
        assert!(out.content.contains("unknown tool"));
    }
}
