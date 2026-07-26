//! The language seam.
//!
//! Everything language-specific in the harness lives behind this trait: a
//! parser that turns source into symbols, and the command chain Oracle runs to
//! verify a workspace. Nothing above this line — not Metis, not Talos, not the
//! halting policy — should ever name a toolchain.
//!
//! Rust is the only implementation in v1. The trait is here because the cost
//! of adding it now is an hour, and the cost of *not* having it is that
//! `cargo` string literals spread into the agent loop, which is what makes a
//! second language expensive later. It will need revision when that second
//! implementation arrives; an interface designed against one example usually
//! does.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Module,
    Const,
    TypeAlias,
    Import,
}

impl SymbolKind {
    pub fn label(self) -> &'static str {
        match self {
            SymbolKind::Function => "fn",
            SymbolKind::Struct => "struct",
            SymbolKind::Enum => "enum",
            SymbolKind::Trait => "trait",
            SymbolKind::Impl => "impl",
            SymbolKind::Module => "mod",
            SymbolKind::Const => "const",
            SymbolKind::TypeAlias => "type",
            SymbolKind::Import => "use",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    Public,
    Restricted,
    Private,
}

/// One exact fact about the code. Never summarized, never paraphrased.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Symbol {
    pub kind: SymbolKind,
    pub name: String,
    /// The declaration as written, minus the body.
    pub signature: String,
    pub file: PathBuf,
    pub line: usize,
    /// Last line of the declaration including its body, 1-indexed inclusive.
    /// Gives every symbol a span, which is what lets Mnemosyne chunk on
    /// declaration boundaries instead of arbitrary line windows.
    pub end_line: usize,
    pub visibility: Visibility,
}

impl Symbol {
    /// Number of source lines this declaration spans.
    pub fn line_count(&self) -> usize {
        self.end_line.saturating_sub(self.line) + 1
    }

    /// Whether this symbol's span fully contains another's.
    pub fn contains(&self, other: &Symbol) -> bool {
        self.file == other.file && self.line <= other.line && self.end_line >= other.end_line
    }
}

/// One rung of Oracle's deterministic ladder.
#[derive(Debug, Clone)]
pub struct VerifyCommand {
    /// Lower runs first. Tier 0 is reserved for in-process parsing.
    pub tier: u8,
    pub label: &'static str,
    pub program: &'static str,
    pub args: Vec<String>,
    /// Whether stdout carries machine-readable diagnostics.
    pub structured: bool,
}

pub trait LanguageAdapter: Send + Sync {
    fn name(&self) -> &str;

    /// File extensions this adapter parses, without the dot.
    fn extensions(&self) -> &[&str];

    /// Whether this adapter recognises `root` as one of its projects.
    fn detect(&self, root: &Path) -> bool;

    /// Extract symbols from one file.
    ///
    /// Implementations **must** return whatever parsed successfully when the
    /// input is syntactically broken, rather than failing. An agent mid-edit
    /// leaves files in a broken state constantly, and that is exactly when
    /// knowing which symbols exist matters most.
    fn symbols(&self, source: &str, path: &Path) -> Vec<Symbol>;

    /// True if the source parses with no error regions. Oracle tier 0.
    fn parses_cleanly(&self, source: &str) -> bool;

    /// Deterministic verification chain, cheapest first.
    fn verify_commands(&self) -> Vec<VerifyCommand>;

    fn handles(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| self.extensions().contains(&e))
    }
}
