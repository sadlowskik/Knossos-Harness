//! The language seam.
//!
//! Everything language-specific in the harness lives behind this trait: a
//! parser that turns source into symbols, and the command chain Oracle runs to
//! verify a workspace. Nothing above this line — not Metis, not Talos, not the
//! halting policy — should ever name a toolchain.
//!
//! Python, Rust, Go and Node each have an implementation. Detection is by
//! marker file at the workspace root; a polyglot tree gets every applicable
//! ladder, cheapest-first across languages. See [`crate::scribe::detect`].

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    Function,
    Class,
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
            SymbolKind::Class => "class",
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    Public,
    Restricted,
    Private,
}

/// One exact fact about the code. Never summarized, never paraphrased.
///
/// `Deserialize` is here for Argus, which persists its index so a rescan costs
/// a hash per file rather than a parse.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
    pub label: String,
    pub program: String,
    pub args: Vec<String>,
    /// Whether stdout carries machine-readable diagnostics.
    pub structured: bool,
    /// Treat a non-zero exit as failure. Linters that warn by default set this
    /// false so advice is surfaced without blocking.
    pub fail_on_nonzero: bool,
    /// File suffixes this tier can be pointed at (e.g. `".py"`). Non-empty
    /// means the tier is *scopable*: it runs over the changed files of these
    /// kinds instead of the whole tree. Empty means it can only run over
    /// everything, and leans on the baseline instead.
    pub scopes: Vec<String>,
    /// Insert `-P` before `-m` so the workspace is kept off `sys.path[0]`.
    /// Only safe for tiers that take explicit file arguments.
    pub safe_path: bool,
    /// This command is part of the project's acceptance boundary. If it is
    /// unavailable, verification is unverifiable and must fail closed.
    pub required: bool,
}

impl VerifyCommand {
    pub fn new(
        tier: u8,
        label: impl Into<String>,
        program: impl Into<String>,
        args: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            tier,
            label: label.into(),
            program: program.into(),
            args: args.into_iter().map(Into::into).collect(),
            structured: false,
            fail_on_nonzero: true,
            scopes: Vec::new(),
            safe_path: true,
            required: false,
        }
    }

    pub fn structured(mut self) -> Self {
        self.structured = true;
        self
    }

    pub fn advisory(mut self) -> Self {
        self.fail_on_nonzero = false;
        self
    }

    pub fn scopes(mut self, scopes: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.scopes = scopes.into_iter().map(Into::into).collect();
        self
    }

    pub fn unsafe_path(mut self) -> Self {
        self.safe_path = false;
        self
    }

    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }
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
    ///
    /// `path` is how a polyglot adapter decides which parser to ask; a
    /// single-language adapter may ignore it.
    fn parses_cleanly(&self, source: &str, path: &Path) -> bool;

    /// Deterministic verification chain, cheapest first.
    fn verify_commands(&self) -> Vec<VerifyCommand>;

    fn handles(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| self.extensions().contains(&e))
    }
}
