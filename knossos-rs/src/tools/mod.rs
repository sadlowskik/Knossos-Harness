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
//!    against an allowlist. What the resulting process can *see* is a separate
//!    question, answered by [`sandbox`](crate::sandbox).
//! 3. **Freshness.** Reads record a [`Stamp`] of what was on disk. A whole-file
//!    write whose target changed since then is refused rather than applied, so
//!    an edit made outside the harness — by the user, by a formatter, by a
//!    rebase — is not silently overwritten by content the model composed from
//!    a stale read. `edit_file` needs no such check: it re-reads at edit time
//!    and requires its `old_string` to still match, which fails naturally when
//!    the region it targeted has moved.
//!
//! Tools also report which files they changed. That is not bookkeeping: it is
//! what lets Ariadne distinguish a step that did work from one that spun, and
//! so decide whether the loop is `Stuck`.
//!
//! Every tool also declares whether it is [consequential][Tool::consequential]
//! — whether it can change anything outside the conversation. Talos counts only
//! those toward having done the task, so that reading a file cannot be mistaken
//! for carrying one out. The trait method has no default on purpose: a new tool
//! does not compile until the question is answered, whereas a list of names
//! kept elsewhere would quietly classify the next writing tool as a read.

pub mod fs;
pub mod history;
pub mod search;
pub mod shell;

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use async_trait::async_trait;

use crate::diff::{diff_file, FileDiff};
use crate::engine::ToolDef;

pub use history::{Conflict, History, JournalEntry, Stamp};

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
    /// Freshness stamps and the undo journal — what already happened, as
    /// opposed to what is permitted, which is the rest of this type. Shared
    /// with every clone of the context, so a front end holding one sees the
    /// same history as the loop holding another. See [`history`].
    history: History,
}

impl ToolCtx {
    /// `root` must already be canonical — see `Config::workspace_root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        ToolCtx {
            root: root.into(),
            dry_run: false,
            staged: Arc::new(Mutex::new(BTreeMap::new())),
            history: History::new(),
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
            // Deliberately not stamped: this is the harness's own pending work,
            // not an observation of disk, and recording it would make every
            // later freshness check compare against the wrong thing.
            return Ok(staged.clone());
        }
        let content = std::fs::read_to_string(path)?;
        self.history.observed(path, &content);
        Ok(content)
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
        self.history.record(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)?;
        // The harness's own write is not an external change. Without this, the
        // second write to a file would report a conflict against the first.
        self.history.wrote(path, content);
        Ok(())
    }

    /// Mark a point the workspace can be rewound to. See [`History::checkpoint`].
    pub fn checkpoint(&self, label: impl Into<String>) -> String {
        self.history.checkpoint(label)
    }

    /// Restore every file to its state at `label`. See [`History::rewind`].
    pub fn rewind(&self, label: &str) -> Result<Vec<PathBuf>> {
        self.history.rewind(label)
    }

    /// Every recorded write, oldest first.
    pub fn journal(&self) -> Vec<JournalEntry> {
        self.history.entries()
    }

    /// Whether `path` still holds what the harness last read from it.
    /// See [`History::conflict`].
    pub fn conflict(&self, path: &Path) -> Option<Conflict> {
        self.history.conflict(path)
    }

    /// Accept whatever is on disk now as the new baseline.
    /// See [`History::accept_current`].
    pub fn accept_current(&self, path: &Path) {
        self.history.accept_current(path)
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

    /// Write only the selected hunks of the selected files.
    ///
    /// `selection` maps a workspace-relative path to the hunk ids being
    /// accepted. Files absent from the selection are left staged, so a partial
    /// review can be continued rather than lost. Accepting every hunk of a file
    /// is equivalent to applying it whole.
    pub fn apply_hunks(&self, selection: &[(String, Vec<usize>)]) -> Result<Vec<PathBuf>> {
        let mut staged = self.staged.lock().unwrap();
        let mut written = Vec::new();

        for (relative, accepted) in selection {
            let path = self.root.join(relative);
            // Cloned so the map is not borrowed while it is mutated below.
            let Some(proposed) = staged.get(&path).cloned() else {
                continue;
            };

            let original = std::fs::read_to_string(&path).unwrap_or_default();
            let d = diff_file(Path::new(relative), Some(&original), &proposed);
            let merged = crate::diff::apply_hunks(&original, &d.hunks, accepted);

            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            self.history.record(&path);
            std::fs::write(&path, &merged)?;
            self.history.wrote(&path, &merged);
            written.push(path.clone());

            // Fully accepted means nothing is left to review for this file.
            if accepted.len() == d.hunks.len() {
                staged.remove(&path);
            } else {
                // Keep the rest staged; `diffs()` re-measures against the file
                // as it now stands, so the remaining hunks stay accurate.
                staged.insert(path, proposed);
            }
        }

        Ok(written)
    }

    /// Write every staged change to disk and clear the staging area.
    pub fn apply_staged(&self) -> Result<Vec<PathBuf>> {
        let mut staged = self.staged.lock().unwrap();
        let mut written = Vec::new();
        for (path, content) in staged.iter() {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            self.history.record(path);
            std::fs::write(path, content)?;
            self.history.wrote(path, content);
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
    /// Whether this tool can change something outside the conversation.
    ///
    /// True for writing to the workspace or running a command; false for
    /// reading, listing and searching. Deliberately not defaulted — see the
    /// module docs.
    fn consequential(&self) -> bool;
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
    hooks: Vec<Box<dyn crate::hooks::Hook>>,
}

impl ToolRegistry {
    pub fn new(tools: Vec<Box<dyn Tool>>) -> Self {
        ToolRegistry { tools, hooks: Vec::new() }
    }

    /// Add a policy hook. They run in the order they are added.
    pub fn with_hook(mut self, hook: Box<dyn crate::hooks::Hook>) -> Self {
        self.hooks.push(hook);
        self
    }

    /// The base tool set: read, write, edit, list, search, run.
    ///
    /// Carries [`ProtectPaths`](crate::hooks::ProtectPaths) by default, because
    /// the hole it closes — `write_file` into `.git/`, which the path jail
    /// permits since `.git` is under the root — is present in every workspace
    /// and is not something a caller should have to know to opt into.
    pub fn standard() -> Self {
        ToolRegistry::new(vec![
            Box::new(fs::ReadFile),
            Box::new(fs::WriteFile),
            Box::new(fs::EditFile),
            Box::new(fs::ListDir),
            Box::new(search::Search),
            Box::new(shell::Run::default()),
        ])
        .with_hook(Box::new(crate::hooks::ProtectPaths::default()))
    }

    /// The standard set plus `search_code`, backed by the Mnemosyne index.
    ///
    /// Retrieval is additive rather than a replacement: `search` stays for when
    /// the agent knows the exact string, and `search_code` covers when it does
    /// not yet know what to grep for.
    pub fn with_retrieval(index: std::sync::Arc<crate::mnemosyne::Mnemosyne>) -> Self {
        let mut registry = ToolRegistry::standard();
        registry.tools.push(Box::new(search::SearchCode::new(index)));
        registry
    }

    /// Add tools discovered at runtime — an MCP server's, typically.
    ///
    /// Takes them already boxed rather than one generic tool at a time, because
    /// the caller has a heterogeneous set from one connection and splitting it
    /// up would only make the collision check below harder to do once.
    pub fn extend(&mut self, tools: impl IntoIterator<Item = Box<dyn Tool>>) {
        self.tools.extend(tools);
    }

    pub fn defs(&self) -> Vec<ToolDef> {
        self.tools.iter().map(|t| t.def()).collect()
    }

    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }

    /// Whether the named tool can change something outside the conversation.
    ///
    /// An unknown name is not consequential: `dispatch` turns it into an error
    /// output, and an error is not work done.
    pub fn is_consequential(&self, name: &str) -> bool {
        self.tools.iter().any(|t| t.name() == name && t.consequential())
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
        // One exit point, so `after` sees every call without exception. An
        // early return for any outcome — a denial, an unknown name, a failing
        // tool — would make the hook contract "every call except the ones we
        // forgot", which is not a contract an audit hook can be built on.
        let chained = crate::hooks::before_chain(&self.hooks, name, input);
        let effective = chained.input.as_ref();

        let out = match &chained.denial {
            Some(reason) => ToolOutput::error(reason.clone()),
            None => match self.tools.iter().find(|t| t.name() == name) {
                None => ToolOutput::error(format!(
                    "unknown tool `{name}`; available: {}",
                    self.names().join(", ")
                )),
                Some(tool) => match tool.run(effective, ctx).await {
                    Ok(out) => out,
                    Err(e) => ToolOutput::error(format!("{name} failed: {e:#}")),
                },
            },
        };

        for hook in &self.hooks {
            hook.after(name, effective, &out);
        }
        out
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
    use serde_json::json;

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

    /// The hole the default hook closes, checked through `dispatch` rather than
    /// against the hook in isolation — the path jail admits `.git` because it
    /// is under the root, so nothing else would stop this.
    #[tokio::test]
    async fn the_standard_registry_refuses_to_write_into_dot_git() {
        let (_d, c) = ctx();
        std::fs::create_dir_all(c.root().join(".git")).unwrap();

        let reg = ToolRegistry::standard();
        let out = reg
            .dispatch(
                "write_file",
                &serde_json::json!({"path": ".git/config", "content": "[remote]\n"}),
                &c,
            )
            .await;

        assert!(out.is_error, "{}", out.content);
        assert!(out.content.contains("protect-paths"), "{}", out.content);
        assert!(!c.root().join(".git/config").exists(), "nothing may be written");
    }

    #[tokio::test]
    async fn ordinary_writes_are_unaffected_by_the_default_hook() {
        let (_d, c) = ctx();
        let reg = ToolRegistry::standard();
        let out = reg
            .dispatch(
                "write_file",
                &serde_json::json!({"path": "src/main.rs", "content": "fn main() {}\n"}),
                &c,
            )
            .await;
        assert!(!out.is_error, "{}", out.content);
    }

    #[test]
    fn rewinding_restores_content_a_write_replaced() {
        let (_d, c) = ctx();
        let f = c.root().join("a.rs");
        std::fs::write(&f, "original\n").unwrap();

        c.checkpoint("turn-1");
        c.write(&f, "the agent's idea\n").unwrap();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "the agent's idea\n");

        let restored = c.rewind("turn-1").unwrap();
        assert_eq!(restored, vec![f.clone()]);
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "original\n");
    }

    #[test]
    fn rewinding_deletes_a_file_that_did_not_exist_before() {
        // Restoring empty content instead would leave a file nobody asked for.
        let (_d, c) = ctx();
        let f = c.root().join("invented.rs");

        c.checkpoint("turn-1");
        c.write(&f, "pub fn nobody_wanted_this() {}\n").unwrap();
        assert!(f.exists());

        c.rewind("turn-1").unwrap();
        assert!(!f.exists(), "a file created after the mark must not survive it");
    }

    #[test]
    fn several_writes_to_one_path_rewind_to_the_oldest_state() {
        // The reason replay is newest-first. Oldest-first would leave the file
        // holding whatever the *second* write replaced.
        let (_d, c) = ctx();
        let f = c.root().join("a.rs");
        std::fs::write(&f, "v0\n").unwrap();

        c.checkpoint("turn-1");
        c.write(&f, "v1\n").unwrap();
        c.write(&f, "v2\n").unwrap();
        c.write(&f, "v3\n").unwrap();

        let restored = c.rewind("turn-1").unwrap();
        assert_eq!(restored.len(), 1, "one path, reported once");
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "v0\n");
    }

    #[test]
    fn a_checkpoint_only_rewinds_what_came_after_it() {
        let (_d, c) = ctx();
        let kept = c.root().join("kept.rs");
        let undone = c.root().join("undone.rs");

        c.write(&kept, "written before the mark\n").unwrap();
        c.checkpoint("turn-2");
        c.write(&undone, "written after\n").unwrap();

        c.rewind("turn-2").unwrap();
        assert_eq!(
            std::fs::read_to_string(&kept).unwrap(),
            "written before the mark\n",
            "earlier work must survive"
        );
        assert!(!undone.exists());
    }

    #[test]
    fn marks_taken_after_a_rewind_point_stop_being_valid() {
        // They index into journal entries that no longer exist; keeping them
        // would let a later rewind truncate an unrelated part of the journal.
        let (_d, c) = ctx();
        c.checkpoint("early");
        c.write(&c.root().join("a.rs"), "one\n").unwrap();
        c.checkpoint("late");
        c.write(&c.root().join("b.rs"), "two\n").unwrap();

        c.rewind("early").unwrap();
        assert!(c.rewind("late").is_err(), "a stale mark must not be usable");
    }

    #[test]
    fn rewinding_an_unknown_label_is_an_error_not_a_silent_no_op() {
        let (_d, c) = ctx();
        assert!(c.rewind("never-taken").is_err());
    }

    #[test]
    fn a_dry_run_write_is_not_journalled() {
        // Nothing reached disk, so there is nothing to undo; `discard_staged`
        // already covers it, and a journal entry here would make a rewind
        // delete a file the dry run never created.
        let (_d, c) = ctx();
        let c = c.dry_run();
        c.checkpoint("turn-1");
        c.write(&c.root().join("a.rs"), "staged only\n").unwrap();
        assert!(c.journal().is_empty());
    }

    #[test]
    fn applying_staged_changes_is_journalled_and_undoable() {
        // The case the Python docstring calls out: staging protects a dry run,
        // but once `apply` has written, only a journal can put it back.
        let (_d, c) = ctx();
        let f = c.root().join("a.rs");
        std::fs::write(&f, "original\n").unwrap();

        let c = c.dry_run();
        c.checkpoint("before-apply");
        c.write(&f, "proposed\n").unwrap();
        c.apply_staged().unwrap();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "proposed\n");

        c.rewind("before-apply").unwrap();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "original\n");
    }

    #[test]
    fn a_rewound_file_can_be_written_again_without_a_stale_conflict() {
        // rewind changes the file behind the freshness check's back; if it did
        // not re-baseline, every write after an undo would be refused.
        let (_d, c) = ctx();
        let f = c.root().join("a.rs");
        std::fs::write(&f, "original\n").unwrap();

        c.checkpoint("turn-1");
        c.write(&f, "attempt\n").unwrap();
        c.rewind("turn-1").unwrap();

        assert!(c.conflict(&f).is_none(), "the rewound state is the new baseline");
    }

    /// Counts what `after` was shown, so the contract can be checked rather
    /// than asserted in a doc comment.
    #[derive(Default)]
    struct Counting {
        seen: std::sync::Mutex<Vec<(String, bool)>>,
    }

    impl crate::hooks::Hook for std::sync::Arc<Counting> {
        fn name(&self) -> &str {
            "counting"
        }
        fn before(&self, _t: &str, input: &serde_json::Value) -> crate::hooks::Decision {
            if input.get("path").and_then(|v| v.as_str()) == Some("denied.rs") {
                return crate::hooks::Decision::Deny("no".into());
            }
            crate::hooks::Decision::Allow
        }
        fn after(&self, tool: &str, _i: &serde_json::Value, out: &ToolOutput) {
            self.seen.lock().unwrap().push((tool.to_string(), out.is_error));
        }
    }

    #[tokio::test]
    async fn after_hooks_see_every_call_including_the_ones_that_failed() {
        let (_d, c) = ctx();
        let counter = std::sync::Arc::new(Counting::default());
        let reg = ToolRegistry::standard().with_hook(Box::new(counter.clone()));

        // 1: ordinary success.
        reg.dispatch("write_file", &json!({"path": "ok.rs", "content": "x\n"}), &c).await;
        // 2: denied by a hook before the tool ran.
        reg.dispatch("write_file", &json!({"path": "denied.rs", "content": "x\n"}), &c).await;
        // 3: a name no tool answers to.
        reg.dispatch("no_such_tool", &json!({}), &c).await;
        // 4: the tool ran and reported failure.
        reg.dispatch("read_file", &json!({"path": "missing.rs"}), &c).await;

        let seen = counter.seen.lock().unwrap().clone();
        assert_eq!(
            seen.len(),
            4,
            "an audit hook that misses a call is not an audit hook: {seen:?}"
        );
        assert_eq!(seen[0], ("write_file".into(), false));
        assert_eq!(seen[1], ("write_file".into(), true), "the denial");
        assert_eq!(seen[2], ("no_such_tool".into(), true), "the unknown name");
        assert_eq!(seen[3], ("read_file".into(), true), "the tool's own failure");
    }

    #[tokio::test]
    async fn after_hooks_are_shown_the_input_the_tool_actually_received() {
        // Auditing the pre-rewrite value would describe a call that never
        // happened.
        struct Rewrite;
        impl crate::hooks::Hook for Rewrite {
            fn name(&self) -> &str {
                "rewrite"
            }
            fn before(&self, _t: &str, _i: &serde_json::Value) -> crate::hooks::Decision {
                crate::hooks::Decision::Rewrite(json!({"path": "rewritten.rs", "content": "x\n"}))
            }
        }

        let (_d, c) = ctx();
        let counter = std::sync::Arc::new(Counting::default());
        let reg = ToolRegistry::new(vec![Box::new(fs::WriteFile)])
            .with_hook(Box::new(Rewrite))
            .with_hook(Box::new(counter.clone()));

        reg.dispatch("write_file", &json!({"path": "original.rs", "content": "x\n"}), &c).await;

        assert!(c.root().join("rewritten.rs").exists());
        assert!(!c.root().join("original.rs").exists());
        assert_eq!(counter.seen.lock().unwrap().len(), 1);
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
