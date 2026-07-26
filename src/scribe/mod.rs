//! Scribe: the exact tier of memory.
//!
//! The two-tier split this comes from is the one idea in the Daedalus
//! architecture that transfers to a harness without dilution: approximate the
//! prose, but keep anything a compiler cares about bit-exact. Most agent
//! frameworks summarize the whole context window — identifiers and paths
//! included — and then the model invents a method name that never existed.
//!
//! Scribe never summarizes. Symbols are parsed, stored, and rendered verbatim.
//! It is a parser, not a model: it returns ground truth, not a best guess.

pub mod adapter;
pub mod rust;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use ignore::WalkBuilder;

pub use adapter::{LanguageAdapter, Symbol, SymbolKind, VerifyCommand, Visibility};
pub use rust::RustAdapter;

/// Pick an adapter for a workspace. Rust is the only one in v1.
pub fn adapter_for(root: &Path) -> Box<dyn LanguageAdapter> {
    let _ = root;
    Box::new(RustAdapter)
}

pub struct SymbolIndex {
    root: PathBuf,
    adapter: Box<dyn LanguageAdapter>,
    /// Per-file, so a single edit re-parses one file rather than the tree.
    files: BTreeMap<PathBuf, Vec<Symbol>>,
}

impl SymbolIndex {
    pub fn new(root: impl Into<PathBuf>, adapter: Box<dyn LanguageAdapter>) -> Self {
        SymbolIndex { root: root.into(), adapter, files: BTreeMap::new() }
    }

    /// Parse every file the adapter handles, honouring .gitignore.
    pub fn build(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        let adapter = adapter_for(&root);
        let mut index = SymbolIndex::new(root, adapter);
        index.rebuild()?;
        Ok(index)
    }

    pub fn rebuild(&mut self) -> Result<()> {
        self.files.clear();
        let walker = WalkBuilder::new(&self.root).git_ignore(true).build();
        for entry in walker.flatten() {
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            if !self.adapter.handles(entry.path()) {
                continue;
            }
            let _ = self.refresh(entry.path());
        }
        Ok(())
    }

    /// Re-parse a single file. Called after every write so the index tracks
    /// the agent's own edits.
    pub fn refresh(&mut self, path: &Path) -> Result<()> {
        let source = std::fs::read_to_string(path)?;
        let rel = path.strip_prefix(&self.root).unwrap_or(path).to_path_buf();
        let symbols = self.adapter.symbols(&source, &rel);
        self.files.insert(rel, symbols);
        Ok(())
    }

    pub fn adapter(&self) -> &dyn LanguageAdapter {
        self.adapter.as_ref()
    }

    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    pub fn symbol_count(&self) -> usize {
        self.files.values().map(Vec::len).sum()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Symbol> {
        self.files.values().flatten()
    }

    /// Exact lookup. This is what grounds "does this identifier exist?".
    pub fn lookup(&self, name: &str) -> Vec<&Symbol> {
        self.iter().filter(|s| s.name == name).collect()
    }

    /// Every declared name — the vocabulary the agent is allowed to assume.
    pub fn names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self
            .iter()
            .filter(|s| s.kind != SymbolKind::Import)
            .map(|s| s.name.as_str())
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Render the index for injection into a prompt.
    ///
    /// Verbatim signatures grouped by file, imports omitted (they are noise at
    /// this level), truncated by whole files so a signature is never cut in
    /// half — a half-rendered signature is worse than an absent one, because
    /// the model will confidently complete it.
    pub fn render(&self, max_bytes: usize) -> String {
        let mut out = String::new();
        for (file, symbols) in &self.files {
            let interesting: Vec<&Symbol> = symbols
                .iter()
                .filter(|s| s.kind != SymbolKind::Import)
                .collect();
            if interesting.is_empty() {
                continue;
            }

            let mut block = format!("\n## {}\n", file.display());
            for s in interesting {
                // No kind label: a Rust signature already begins with its
                // keyword, so prefixing it produces "struct pub struct Adder".
                block.push_str(&format!("  {}: {}\n", s.line, s.signature));
            }

            if out.len() + block.len() > max_bytes {
                out.push_str(&format!(
                    "\n[symbol index truncated — {} files, {} symbols total]\n",
                    self.file_count(),
                    self.symbol_count()
                ));
                break;
            }
            out.push_str(&block);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
        std::fs::write(
            dir.path().join("src/lib.rs"),
            "pub fn alpha() -> u32 { 1 }\npub struct Beta;\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("src/other.rs"), "fn gamma() {}\n").unwrap();
        dir
    }

    #[test]
    fn indexes_every_rust_file_in_the_tree() {
        let dir = workspace();
        let idx = SymbolIndex::build(dir.path()).unwrap();
        assert_eq!(idx.file_count(), 2);
        assert!(idx.names().contains(&"alpha"));
        assert!(idx.names().contains(&"gamma"));
    }

    #[test]
    fn lookup_returns_the_exact_declaration() {
        let dir = workspace();
        let idx = SymbolIndex::build(dir.path()).unwrap();
        let hits = idx.lookup("alpha");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].signature, "pub fn alpha() -> u32");
        assert_eq!(hits[0].line, 1);
        assert!(idx.lookup("does_not_exist").is_empty());
    }

    #[test]
    fn refresh_picks_up_an_edit_to_one_file() {
        let dir = workspace();
        let mut idx = SymbolIndex::build(dir.path()).unwrap();
        assert!(idx.lookup("delta").is_empty());

        let f = dir.path().join("src/lib.rs");
        std::fs::write(&f, "pub fn delta() {}\n").unwrap();
        idx.refresh(&f).unwrap();

        assert_eq!(idx.lookup("delta").len(), 1);
        assert!(idx.lookup("alpha").is_empty(), "stale symbols must be dropped");
    }

    #[test]
    fn render_groups_by_file_and_omits_imports() {
        let dir = workspace();
        std::fs::write(
            dir.path().join("src/lib.rs"),
            "use std::fmt;\npub fn alpha() -> u32 { 1 }\n",
        )
        .unwrap();
        let idx = SymbolIndex::build(dir.path()).unwrap();
        let r = idx.render(10_000);
        assert!(r.contains("## src"));
        assert!(r.contains("pub fn alpha() -> u32"));
        assert!(!r.contains("use std::fmt"));
    }

    #[test]
    fn render_does_not_repeat_the_declaration_keyword() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "pub struct Adder;\nimpl Adder {}\n")
            .unwrap();
        let r = SymbolIndex::build(dir.path()).unwrap().render(10_000);
        assert!(r.contains("pub struct Adder"));
        assert!(!r.contains("struct pub struct"));
        assert!(!r.contains("impl impl"));
    }

    #[test]
    fn render_truncates_on_whole_files_never_mid_signature() {
        let dir = workspace();
        let idx = SymbolIndex::build(dir.path()).unwrap();
        let r = idx.render(40);
        assert!(r.contains("truncated"));
        // Any signature present must be complete.
        for line in r.lines().filter(|l| l.trim_start().starts_with(char::is_numeric)) {
            assert!(line.contains("fn ") || line.contains("struct "), "partial line: {line}");
        }
    }
}
