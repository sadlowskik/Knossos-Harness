//! Argus: the hundred-eyed watchman — he sees the whole repository at once.
//!
//! Scribe parses one file exactly. Argus watches every file and answers the
//! question the harness asks before it can do anything useful:
//!
//! > given this task, which parts of this codebase belong in the context window?
//!
//! That is the retrieval problem Cursor solves with embeddings over chunked
//! files. Argus solves it structurally instead, for three reasons that matter at
//! this project's scale: it needs no model and no vector store, it is
//! incremental enough to run on every keystroke, and every result carries a
//! *reason* — which is what lets Oracle audit a retrieval instead of trusting
//! it.
//!
//! Three layers, in order of how much they can be trusted:
//!
//! 1. **Symbols** — what is defined, where. From [`crate::scribe`]'s parser.
//! 2. **Edges** — what imports what. Same parse, no extra cost.
//! 3. **Ranking** — multi-field BM25 against the request.
//!
//! Retrieval seeds from layer 3, expands along layer 2, and returns slices of
//! layer 1 until a character budget is spent.
//!
//! # Why this is not Mnemosyne
//!
//! Both rank text with BM25 and the resemblance stops there. Mnemosyne scores
//! *chunks* on one field and answers "show me code that reads like this".
//! Argus scores *files* on three fields — what a file says, what it defines,
//! what it is called — and answers "which files does this task touch", then
//! walks the import graph one hop and packs slices to a budget with provenance
//! attached. The three-field split is the part that earns its keep: single-field
//! scoring ranks a symbol's *user* above its *definition*, because the user is
//! usually the shorter file and says the name more often.
//!
//! # What the Python original had and this does not
//!
//! `FileRecord.exact` recorded whether a real parser or a regex scanner
//! produced the symbols, because `tree-sitter` was an optional dependency
//! there. Here it is not optional and the scanner does not exist, so the field
//! would be a constant `true` — and a provenance flag that cannot vary is worse
//! than none, since it invites callers to branch on it.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::Result;
use ignore::WalkBuilder;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};

use crate::lsp::LspClient;
use crate::scribe::{adapter_for, LanguageAdapter, Symbol, SymbolKind};

/// Term -> occurrences. Ordered so a rendered index is byte-stable.
pub type Bag = BTreeMap<String, usize>;

/// Files past this size are indexed as neither symbols nor text. A megabyte of
/// source is a generated file or a vendored blob, and either way it is not what
/// the question is about.
const MAX_FILE_BYTES: usize = 1_000_000;

/// Indexed lexically even though no adapter parses them. A README answers
/// "where is X documented" and `Cargo.toml` answers "what is this crate called",
/// and neither needs a symbol table to do it.
const PROSE_EXTENSIONS: &[&str] = &["md", "toml"];

/// Directories never worth descending. `.gitignore` covers most of these in a
/// real repository and none of them in a fresh one, so both are applied.
const EXCLUDED_DIRS: &[&str] = &[
    ".git",
    ".argus",
    "target",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
];

/// BM25 (Robertson & Walker 1994). `K1` saturates term frequency: the 50th
/// "file" in a file about files adds almost nothing over the 5th.
const BM25_K1: f64 = 1.5;
/// How hard long documents are penalised.
const BM25_B: f64 = 0.75;

/// Field weights. Mentioning a name is weak evidence; defining it is strong;
/// being named after it is stronger still.
const W_TEXT: f64 = 1.0;
const W_DEFS: f64 = 2.5;
const W_PATH: f64 = 2.0;

/// Files whose job is to talk *about* code quote identifiers without being
/// their home. They stay findable — ask about a test by name and you still get
/// it — just outranked by the real thing.
const META_PENALTY: f64 = 0.6;

/// Bumped whenever a scored field is added or its meaning changes. An index
/// written before a field existed would otherwise load without complaint and
/// leave that signal silently switched off, which is the kind of bug that never
/// gets noticed because nothing crashes.
const INDEX_VERSION: u32 = 1;

static IDENT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Za-z_][A-Za-z0-9_]*").expect("static pattern"));

/// Identifiers this common carry no signal about *which* file you want.
///
/// Rust-flavoured, unlike the Python original's list: `self`, `crate` and `impl`
/// are in every file here, while `def`, `cls` and `torch` never appear. A stop
/// list inherited from the wrong language is dead weight in one direction and a
/// hole in the other.
static STOP: LazyLock<BTreeSet<&'static str>> = LazyLock::new(|| {
    "self cls the and not for while let mut use mod impl struct enum trait match \
     where crate super pub fn dyn ref move box type const static async await \
     return break continue loop else true false none some ptr str usize isize \
     u8 u16 u32 u64 i8 i16 i32 i64 f32 f64 bool vec string option result err ok"
        .split_whitespace()
        .collect()
});

fn is_stop(word: &str) -> bool {
    STOP.contains(word)
}

/// `RecurrentMoECore` -> `["recurrentmoecore", "recurrent", "core"]`.
///
/// Both halves matter: the whole token matches an exact symbol name, the parts
/// match a request phrased in prose. Parts of one or two characters are dropped
/// — `mo` and `e` identify nothing.
pub(crate) fn split_identifier(token: &str) -> Vec<String> {
    let whole = token.to_lowercase();
    let mut parts = vec![whole.clone()];
    for chunk in token.split('_').filter(|c| !c.is_empty()) {
        for piece in crate::mnemosyne::split_camel(chunk) {
            if piece.len() > 2 && !parts.contains(&piece) {
                parts.push(piece);
            }
        }
    }
    parts
}

/// Every identifier in `text`, in order, without repeats.
fn identifiers(text: &str) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for m in IDENT.find_iter(text) {
        if seen.insert(m.as_str().to_string()) {
            out.push(m.as_str().to_string());
        }
    }
    out
}

/// Text -> weighted terms, stop words and one/two-character noise removed.
pub fn tokenize(text: &str) -> Bag {
    let mut bag = Bag::new();
    for m in IDENT.find_iter(text) {
        let token = m.as_str();
        if token.len() < 3 || is_stop(&token.to_lowercase()) {
            continue;
        }
        for part in split_identifier(token) {
            if !is_stop(&part) {
                *bag.entry(part).or_insert(0) += 1;
            }
        }
    }
    bag
}

/// One definition, plus the container that qualifies it.
///
/// [`crate::scribe::Symbol`] has no parent field, because the exact tier renders
/// one file at a time and never needs one. Argus does: two files defining
/// `forward` are only telling you something once you can say which is
/// `Router::forward` and which is `Expert::forward`. Recovered from the spans
/// rather than added to `Symbol`, since the parser already emits items in
/// pre-order and containment is therefore decidable without re-parsing.
#[derive(Debug, Clone)]
pub struct Located<'a> {
    pub symbol: &'a Symbol,
    pub parent: Option<&'a str>,
    /// Whether this definition lives under a `#[cfg(test)] mod tests`.
    ///
    /// Callers that build repository vocabulary want it out: a Rust test is
    /// named after the sentence it asserts, so its parts are ordinary English.
    pub in_test: bool,
}

impl Located<'_> {
    pub fn qualname(&self) -> String {
        match self.parent {
            Some(p) => format!("{p}::{}", self.symbol.name),
            None => self.symbol.name.clone(),
        }
    }

    /// A `file:line` string — clickable in every editor worth forking.
    pub fn reference(&self) -> String {
        format!("{}:{}", self.symbol.file.display(), self.symbol.line)
    }
}

/// Everything Argus knows about one file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRecord {
    /// Repository-relative, forward slashes on every platform.
    pub path: String,
    pub sha: String,
    pub language: String,
    pub n_lines: usize,
    pub symbols: Vec<Symbol>,
    pub imports: Vec<String>,
    /// What the file says.
    pub idents: Bag,
    /// Words from the names this file *defines*, as opposed to merely mentions.
    /// Scored as its own field because flat term frequency cannot tell a
    /// definition from a use, and ranked the user above the definition.
    pub defs: Bag,
    /// Words from the file's own path. "oracle" in a question should favour
    /// `src/oracle/mod.rs` over a file that happens to say "oracle" a lot.
    pub path_terms: Bag,
}

impl FileRecord {
    /// The nearest enclosing `impl`/`trait`/`mod` for the symbol at `idx`.
    fn container_of(&self, idx: usize) -> Option<&Symbol> {
        let target = &self.symbols[idx];
        self.symbols[..idx].iter().rev().find(|candidate| {
            matches!(
                candidate.kind,
                SymbolKind::Impl | SymbolKind::Trait | SymbolKind::Module
            ) && candidate.contains(target)
        })
    }

    /// Whether the symbol at `idx` lives inside a test module.
    ///
    /// Rust unit tests sit in the same file as the code they exercise, so
    /// without this every module's definition terms would be polluted by the
    /// sentences its tests are named after — `deleting_a_test_fails_verification`
    /// contributes "deleting", "fails" and "verification" to what the file
    /// claims to define. Python got the same protection from a `test_` prefix
    /// rule, which does not transfer: names here carry no prefix.
    fn under_test_module(&self, idx: usize) -> bool {
        let target = &self.symbols[idx];
        self.symbols[..idx].iter().any(|candidate| {
            candidate.kind == SymbolKind::Module
                && (candidate.name == "tests" || candidate.name == "test")
                && candidate.contains(target)
        })
    }

    /// Every symbol with its qualifying container resolved.
    pub fn located(&self) -> Vec<Located<'_>> {
        (0..self.symbols.len())
            .map(|i| Located {
                symbol: &self.symbols[i],
                parent: self.container_of(i).map(|c| c.name.as_str()),
                in_test: self.under_test_module(i),
            })
            .collect()
    }
}

/// One slice of source, and the reason it was pulled.
#[derive(Debug, Clone)]
pub struct Retrieved {
    pub file: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub score: f64,
    pub reason: String,
    pub symbol: Option<String>,
}

impl Retrieved {
    pub fn reference(&self) -> String {
        format!("{}:{}", self.file, self.start_line)
    }
}

/// Rendered excerpts that still remember what they were made of.
///
/// An engine that reasons over prose wants the flat string; an engine that
/// reports on the retrieval wants the hits. Recovering the structure by
/// re-parsing the rendered text breaks the moment a retrieved file contains a
/// line that looks like a header — a markdown README is enough to do it.
#[derive(Debug, Clone, Default)]
pub struct Context {
    pub text: String,
    pub hits: Vec<Retrieved>,
}

impl std::fmt::Display for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text)
    }
}

/// Pack retrieved slices into a prompt block, provenance on every one.
pub fn render(hits: &[Retrieved]) -> Context {
    let text = hits
        .iter()
        .map(|h| {
            let symbol = match &h.symbol {
                Some(s) => format!("  ({s})"),
                None => String::new(),
            };
            format!(
                "# {}:{}-{}{symbol}  [{}]\n{}",
                h.file, h.start_line, h.end_line, h.reason, h.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    Context {
        text,
        hits: hits.to_vec(),
    }
}

#[derive(Debug, Default, Clone)]
pub struct ScanReport {
    pub scanned: usize,
    /// Re-parsed because the content hash changed.
    pub parsed: usize,
    pub unchanged: usize,
    pub removed: usize,
    pub failed: Vec<(String, String)>,
}

impl std::fmt::Display for ScanReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "scanned {} | parsed {} | unchanged {} | removed {}",
            self.scanned, self.parsed, self.unchanged, self.removed
        )?;
        if !self.failed.is_empty() {
            write!(f, " | failed {}", self.failed.len())?;
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct StoredIndex {
    version: u32,
    root: String,
    files: BTreeMap<String, FileRecord>,
}

/// A repository index that answers "what should I load to do this?".
pub struct Argus {
    root: PathBuf,
    adapter: Box<dyn LanguageAdapter>,
    files: BTreeMap<String, FileRecord>,
    /// An optional language server. When present, [`users_of`](Argus::users_of)
    /// answers with resolved references instead of nothing.
    lsp: Option<LspClient>,
}

impl Argus {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let adapter = adapter_for(&root);
        Argus {
            root,
            adapter,
            files: BTreeMap::new(),
            lsp: None,
        }
    }

    pub fn with_adapter(root: impl Into<PathBuf>, adapter: Box<dyn LanguageAdapter>) -> Self {
        Argus {
            root: root.into(),
            adapter,
            files: BTreeMap::new(),
            lsp: None,
        }
    }

    /// Attach a language server, for the questions name matching cannot answer.
    pub fn with_lsp(mut self, lsp: LspClient) -> Self {
        self.lsp = Some(lsp);
        self
    }

    pub fn lsp(&self) -> Option<&LspClient> {
        self.lsp.as_ref()
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn files(&self) -> &BTreeMap<String, FileRecord> {
        &self.files
    }

    // ------------------------------------------------------------- scanning

    /// Every indexable file, code before prose.
    ///
    /// The order is not cosmetic: [`Argus::lookup`] returns hits in index order,
    /// so a name defined in both a source file and a README resolves to the
    /// source file first.
    fn walk(&self) -> Vec<(String, PathBuf)> {
        let mut code = Vec::new();
        let mut prose = Vec::new();

        let walker = WalkBuilder::new(&self.root)
            .git_ignore(true)
            .filter_entry(|entry| {
                let is_dir = entry.file_type().is_some_and(|t| t.is_dir());
                let name = entry.file_name().to_string_lossy().to_string();
                !(is_dir && EXCLUDED_DIRS.contains(&name.as_str()))
            })
            .build();

        for entry in walker.flatten() {
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            let path = entry.path();
            let Some(rel) = relative_posix(&self.root, path) else {
                continue;
            };
            if self.adapter.handles(path) {
                code.push((rel, path.to_path_buf()));
            } else if extension_of(path).is_some_and(|e| PROSE_EXTENSIONS.contains(&e.as_str())) {
                prose.push((rel, path.to_path_buf()));
            }
        }

        code.sort();
        prose.sort();
        code.extend(prose);
        code
    }

    /// Index the repo, re-parsing only what changed since the last scan.
    pub fn scan(&mut self) -> ScanReport {
        let mut report = ScanReport::default();
        let mut seen: BTreeSet<String> = BTreeSet::new();

        for (rel, path) in self.walk() {
            seen.insert(rel.clone());
            report.scanned += 1;

            let raw = match std::fs::read(&path) {
                Ok(raw) => raw,
                Err(err) => {
                    report.failed.push((rel, err.to_string()));
                    continue;
                }
            };
            if raw.len() > MAX_FILE_BYTES {
                continue;
            }

            let sha = format!("{:x}", Sha1::digest(&raw));
            if self.files.get(&rel).is_some_and(|cached| cached.sha == sha) {
                report.unchanged += 1;
                continue;
            }

            let text = String::from_utf8_lossy(&raw).into_owned();
            let record = self.parse(&rel, &text, sha);
            self.files.insert(rel, record);
            report.parsed += 1;
        }

        let gone: Vec<String> = self
            .files
            .keys()
            .filter(|k| !seen.contains(*k))
            .cloned()
            .collect();
        report.removed = gone.len();
        for key in gone {
            self.files.remove(&key);
        }
        report
    }

    fn parse(&self, rel: &str, text: &str, sha: String) -> FileRecord {
        let language = extension_of(Path::new(rel)).unwrap_or_else(|| "text".to_string());
        let handled = self.adapter.handles(Path::new(rel));

        let symbols = if handled {
            self.adapter.symbols(text, Path::new(rel))
        } else {
            Vec::new()
        };
        let imports = symbols
            .iter()
            .filter(|s| s.kind == SymbolKind::Import)
            .filter_map(|s| normalize_import(&s.name))
            .collect();

        let mut record = FileRecord {
            path: rel.to_string(),
            sha,
            language,
            n_lines: text.matches('\n').count() + 1,
            symbols,
            imports,
            idents: tokenize(text),
            defs: Bag::new(),
            path_terms: path_terms(rel),
        };
        record.defs = definition_terms(&record);
        record
    }

    // -------------------------------------------------------------- queries

    /// Every definition of `name`, across the repo.
    ///
    /// Returns a *list* — the thing [`crate::scribe::SymbolIndex`] cannot do
    /// per-file. Two files both defining `forward` give you two symbols, not one
    /// silently overwriting the other. Matches a bare name or a qualified one.
    pub fn lookup(&self, name: &str) -> Vec<Located<'_>> {
        self.files
            .values()
            .flat_map(FileRecord::located)
            .filter(|l| l.symbol.kind != SymbolKind::Import)
            .filter(|l| l.symbol.name == name || l.qualname() == name)
            .collect()
    }

    pub fn symbols_in(&self, path: &str) -> Vec<&Symbol> {
        self.files
            .get(path)
            .map(|r| r.symbols.iter().collect())
            .unwrap_or_default()
    }

    /// Files that actually reference `symbol`, via the language server.
    ///
    /// This is the question [`importers_of`](Self::importers_of) can only
    /// approximate. Name matching says "some file imports something called
    /// `config`"; a language server says which `config`, having resolved it.
    ///
    /// Returns empty when no server is attached, so callers must treat "no
    /// answer" and "no users" as the same — which is why `importers_of` remains
    /// the fallback rather than being replaced.
    pub fn users_of(&self, symbol: &Symbol) -> Vec<String> {
        let Some(lsp) = self.lsp.as_ref().filter(|l| l.available()) else {
            return Vec::new();
        };
        let own_file = symbol.file.to_string_lossy().replace('\\', "/");
        let locations = lsp.references(
            &self.root.join(&symbol.file),
            symbol.line.saturating_sub(1), // LSP counts lines from zero
            0,
            false,
        );

        let mut out: Vec<String> = Vec::new();
        for location in locations {
            let Some(rel) = self.relative(&location.path) else {
                continue; // outside the workspace
            };
            if rel != own_file && !out.contains(&rel) {
                out.push(rel);
            }
        }
        out
    }

    /// A path inside the workspace, as the forward-slashed key `files` uses.
    fn relative(&self, path: &Path) -> Option<String> {
        let resolved = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let root = self
            .root
            .canonicalize()
            .unwrap_or_else(|_| self.root.clone());
        let rel = resolved
            .strip_prefix(&root)
            .or_else(|_| path.strip_prefix(&self.root))
            .ok()?;
        Some(rel.to_string_lossy().replace('\\', "/"))
    }

    /// Files whose imports plausibly resolve to `path`.
    ///
    /// Module-path matching, not real resolution: a file's module path must
    /// appear as a contiguous run inside the `use`. `crate::oracle::diagnostics`
    /// therefore matches `src/oracle/diagnostics.rs` and not `src/scribe/diagnostics.rs`.
    /// Good enough to walk one hop out, which is all it is used for.
    pub fn importers_of(&self, path: &str) -> Vec<String> {
        let target = module_path(path);
        if target.is_empty() {
            return Vec::new();
        }
        self.files
            .iter()
            .filter(|(rel, _)| rel.as_str() != path)
            .filter(|(_, rec)| {
                rec.imports
                    .iter()
                    .any(|imp| contains_run(&import_segments(imp), &target))
            })
            .map(|(rel, _)| rel.clone())
            .collect()
    }

    /// Inverse document frequency, computed per field.
    ///
    /// A term common in prose can be rare among definitions, and that contrast
    /// is exactly the signal — so the fields cannot share one table.
    fn idf(&self, field: fn(&FileRecord) -> &Bag) -> BTreeMap<String, f64> {
        let n_docs = self.files.len().max(1) as f64;
        let mut df: BTreeMap<&str, usize> = BTreeMap::new();
        for rec in self.files.values() {
            for term in field(rec).keys() {
                *df.entry(term.as_str()).or_insert(0) += 1;
            }
        }
        df.into_iter()
            .map(|(term, count)| {
                let c = count as f64;
                (
                    term.to_string(),
                    (1.0 + (n_docs - c + 0.5) / (c + 0.5)).ln(),
                )
            })
            .collect()
    }

    /// BM25 over one field. Empty documents are skipped, not scored zero.
    fn bm25_field(&self, query: &Bag, field: fn(&FileRecord) -> &Bag) -> BTreeMap<String, f64> {
        let idf = self.idf(field);
        let lengths: BTreeMap<&str, usize> = self
            .files
            .iter()
            .map(|(rel, rec)| (rel.as_str(), field(rec).values().sum()))
            .collect();
        let non_empty: Vec<usize> = lengths.values().copied().filter(|v| *v > 0).collect();
        let avg_len = if non_empty.is_empty() {
            1.0
        } else {
            non_empty.iter().sum::<usize>() as f64 / non_empty.len() as f64
        };

        let mut out = BTreeMap::new();
        for (rel, rec) in &self.files {
            let length = lengths[rel.as_str()];
            if length == 0 {
                continue;
            }
            let norm = BM25_K1 * (1.0 - BM25_B + BM25_B * length as f64 / avg_len);
            let bag = field(rec);
            let mut score = 0.0;
            for term in query.keys() {
                let tf = bag.get(term).copied().unwrap_or(0) as f64;
                if tf > 0.0 {
                    score += idf.get(term).copied().unwrap_or(0.0) * (tf * (BM25_K1 + 1.0))
                        / (tf + norm);
                }
            }
            if score > 0.0 {
                out.insert(rel.clone(), score);
            }
        }
        out
    }

    /// Multi-field BM25, highest first. The seeding step.
    ///
    /// Three fields, because they are three different kinds of evidence:
    /// what the file says (weak — any file can mention anything), what it
    /// *defines* (strong), and what it is called (strong). Scored separately and
    /// summed, a simplified BM25F, each with its own idf.
    ///
    /// The history is worth keeping, since each version failed differently. v1
    /// was tf-idf, and a long file repeating "file" and "defines" outranked the
    /// short one that defined the thing asked about — fixed by BM25 term
    /// saturation. v2 was single-field BM25, and it still ranked a symbol's
    /// *user* above its *definition*. This is v3.
    pub fn score_files(&self, query: &str) -> Vec<(String, f64)> {
        let q = tokenize(query);
        if q.is_empty() {
            return Vec::new();
        }

        let text = self.bm25_field(&q, |r| &r.idents);
        let defs = self.bm25_field(&q, |r| &r.defs);
        let paths = self.bm25_field(&q, |r| &r.path_terms);

        let mut scored: Vec<(String, f64)> = self
            .files
            .keys()
            .filter_map(|rel| {
                let get = |m: &BTreeMap<String, f64>| m.get(rel).copied().unwrap_or(0.0);
                let mut s = W_TEXT * get(&text) + W_DEFS * get(&defs) + W_PATH * get(&paths);
                if s <= 0.0 {
                    return None;
                }
                if is_meta(rel) {
                    s *= META_PENALTY;
                }
                Some((rel.clone(), s))
            })
            .collect();

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored
    }

    /// Turn a request into the slice of repo it needs.
    ///
    /// 1. Any identifier in `query` naming a real symbol is an exact hit.
    /// 2. Remaining budget goes to files ranked by [`Argus::score_files`].
    /// 3. `hops` expands to files that import a hit file.
    /// 4. Slices are packed by score until `budget` characters are spent.
    ///
    /// Every result carries a `reason`, so a bad retrieval is debuggable and
    /// Oracle can check what was pulled and why.
    pub fn retrieve(&self, query: &str, budget: usize, hops: usize) -> Vec<Retrieved> {
        const MAX_SLICES: usize = 24;
        let mut packer = Packer::new(&self.root, budget, MAX_SLICES);
        let mut seeds: Vec<String> = Vec::new();

        // 1. Exact symbol hits.
        for token in identifiers(query) {
            for hit in self.lookup(&token) {
                let file = hit.symbol.file.display().to_string();
                packer.take(
                    &file,
                    hit.symbol.line,
                    hit.symbol.end_line,
                    100.0,
                    format!("exact symbol match: {token}"),
                    Some(hit.qualname()),
                );
                seeds.push(file);
            }
        }

        // 2. Lexically ranked files.
        let q = tokenize(query);
        for (rel, score) in self.score_files(query).into_iter().take(MAX_SLICES) {
            if packer.spent >= budget {
                break;
            }
            let Some(rec) = self.files.get(&rel) else {
                continue;
            };
            seeds.push(rel.clone());

            let located: Vec<Located<'_>> = rec
                .located()
                .into_iter()
                .filter(|l| l.symbol.kind != SymbolKind::Import)
                .collect();

            if located.is_empty() {
                packer.take(
                    &rel,
                    1,
                    rec.n_lines.min(60),
                    score,
                    format!("bm25 {score:.2} in {rel}"),
                    None,
                );
                continue;
            }

            // Rank symbols by what their *body* says, not just their name:
            // `Router::new` and `load_balance_loss` score identically on the
            // name alone. The source is already cached, so this costs a
            // tokenize of the top files only, not the whole repo.
            let lines = packer.source(&rel).to_vec();
            let mut ranked: Vec<(f64, &Located<'_>)> = located
                .iter()
                .map(|l| (relevance(&q, l.symbol, &lines), l))
                .collect();
            ranked.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.symbol.line.cmp(&b.1.symbol.line))
            });

            for (rel_score, hit) in ranked.into_iter().take(3) {
                if rel_score <= 0.0 {
                    break;
                }
                packer.take(
                    &rel,
                    hit.symbol.line,
                    hit.symbol.end_line,
                    score,
                    format!("bm25 {score:.2} in {rel}"),
                    Some(hit.qualname()),
                );
            }
        }

        // 3. Graph expansion.
        let mut frontier: Vec<String> = dedup(seeds);
        for _ in 0..hops {
            let mut next = Vec::new();
            for rel in &frontier {
                if packer.spent >= budget {
                    break;
                }
                for neighbour in self.importers_of(rel) {
                    let Some(rec) = self.files.get(&neighbour) else {
                        continue;
                    };
                    if let Some(head) = rec
                        .located()
                        .into_iter()
                        .find(|l| l.symbol.kind != SymbolKind::Import)
                    {
                        packer.take(
                            &neighbour,
                            head.symbol.line,
                            head.symbol.end_line,
                            1.0,
                            format!("one hop from {rel}"),
                            Some(head.qualname()),
                        );
                    }
                    next.push(neighbour);
                }
            }
            frontier = dedup(next);
        }

        let mut out = packer.out;
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.file.cmp(&b.file))
                .then(a.start_line.cmp(&b.start_line))
        });
        out
    }

    /// [`Argus::retrieve`], rendered as a prompt block with provenance.
    pub fn context(&self, query: &str, budget: usize, hops: usize) -> Context {
        render(&self.retrieve(query, budget, hops))
    }

    // ---------------------------------------------------------- persistence

    fn default_index_path(&self) -> PathBuf {
        self.root.join(".argus").join("index.json")
    }

    pub fn save(&self, path: Option<&Path>) -> Result<PathBuf> {
        let target = path
            .map(Path::to_path_buf)
            .unwrap_or_else(|| self.default_index_path());
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let blob = StoredIndex {
            version: INDEX_VERSION,
            root: self.root.display().to_string(),
            files: self.files.clone(),
        };
        std::fs::write(&target, serde_json::to_string(&blob)?)?;
        Ok(target)
    }

    /// Restore a saved index. `false` means there was nothing usable to restore.
    ///
    /// Stale entries are not a correctness problem: [`Argus::scan`] re-parses
    /// anything whose hash no longer matches and drops anything that
    /// disappeared. A *wrong-schema* index is a different matter, and is refused
    /// rather than half-loaded — see `INDEX_VERSION`.
    pub fn load(&mut self, path: Option<&Path>) -> Result<bool> {
        let target = path
            .map(Path::to_path_buf)
            .unwrap_or_else(|| self.default_index_path());
        let Ok(raw) = std::fs::read_to_string(&target) else {
            return Ok(false);
        };
        let Ok(blob) = serde_json::from_str::<StoredIndex>(&raw) else {
            return Ok(false);
        };
        if blob.version != INDEX_VERSION {
            return Ok(false);
        }
        self.files = blob.files;
        Ok(true)
    }
}

/// Packs slices to a character budget, refusing overlaps and repeats.
struct Packer<'a> {
    root: &'a Path,
    budget: usize,
    max_slices: usize,
    spent: usize,
    out: Vec<Retrieved>,
    claimed: BTreeSet<(String, usize, usize)>,
    cache: BTreeMap<String, Vec<String>>,
}

impl<'a> Packer<'a> {
    fn new(root: &'a Path, budget: usize, max_slices: usize) -> Self {
        Packer {
            root,
            budget,
            max_slices,
            spent: 0,
            out: Vec::new(),
            claimed: BTreeSet::new(),
            cache: BTreeMap::new(),
        }
    }

    fn source(&mut self, rel: &str) -> &[String] {
        if !self.cache.contains_key(rel) {
            let lines = std::fs::read_to_string(self.root.join(rel))
                .map(|s| s.lines().map(str::to_string).collect())
                .unwrap_or_default();
            self.cache.insert(rel.to_string(), lines);
        }
        &self.cache[rel]
    }

    fn take(
        &mut self,
        rel: &str,
        start: usize,
        end: usize,
        score: f64,
        reason: String,
        symbol: Option<String>,
    ) {
        if self.out.len() >= self.max_slices || start == 0 {
            return;
        }
        let key = (rel.to_string(), start, end);
        if self.claimed.contains(&key) {
            return;
        }
        // A method's span sits inside its impl block's. Returning both spends
        // the budget twice on the same lines.
        if self
            .claimed
            .iter()
            .any(|(c_rel, c_start, c_end)| c_rel == rel && *c_start <= start && end <= *c_end)
        {
            return;
        }

        let body = {
            let lines = self.source(rel);
            let stop = end.min(lines.len());
            if start > stop {
                return;
            }
            lines[start - 1..stop].join("\n")
        };
        if body.trim().is_empty() {
            return;
        }
        if self.spent + body.len() > self.budget && !self.out.is_empty() {
            return;
        }

        self.claimed.insert(key);
        self.spent += body.len();
        self.out.push(Retrieved {
            file: rel.to_string(),
            start_line: start,
            end_line: end,
            text: body,
            score,
            reason,
            symbol,
        });
    }
}

/// How well one declaration answers the query.
///
/// The name counts triple: a symbol called what you asked about is a stronger
/// answer than one that merely mentions it. Body overlap is divided by the
/// square root of the body's length so a long function cannot win on volume.
fn relevance(query: &Bag, symbol: &Symbol, lines: &[String]) -> f64 {
    let name_hit: usize = split_identifier(&symbol.name)
        .iter()
        .filter_map(|p| query.get(p))
        .sum();

    let stop = symbol.end_line.min(lines.len());
    if symbol.line == 0 || symbol.line > stop {
        return 3.0 * name_hit as f64;
    }
    let body = tokenize(&lines[symbol.line - 1..stop].join("\n"));
    if body.is_empty() {
        return 3.0 * name_hit as f64;
    }
    let overlap: usize = query
        .iter()
        .map(|(term, count)| count * body.get(term).copied().unwrap_or(0))
        .sum();
    let length: usize = body.values().sum();
    3.0 * name_hit as f64 + overlap as f64 / (length as f64).sqrt()
}

/// Words from the names this file defines.
///
/// A type is a stronger landmark than a method, so it counts twice. Imports
/// define nothing and test functions define nothing anyone will ask for — see
/// [`FileRecord::under_test_module`].
fn definition_terms(record: &FileRecord) -> Bag {
    let mut bag = Bag::new();
    for (i, symbol) in record.symbols.iter().enumerate() {
        if symbol.kind == SymbolKind::Import || record.under_test_module(i) {
            continue;
        }
        let weight = match symbol.kind {
            SymbolKind::Struct | SymbolKind::Enum | SymbolKind::Trait => 2,
            _ => 1,
        };
        for part in split_identifier(&symbol.name) {
            if !is_stop(&part) {
                *bag.entry(part).or_insert(0) += weight;
            }
        }
    }
    bag
}

fn path_terms(rel: &str) -> Bag {
    let mut bag = Bag::new();
    for chunk in rel.split(['/', '\\', '.']) {
        for part in split_identifier(chunk) {
            if part.len() >= 2 && !is_stop(&part) {
                *bag.entry(part).or_insert(0) += 1;
            }
        }
    }
    bag
}

/// Files that discuss code rather than being it.
///
/// Matched on whole path components rather than a substring, so `src/latest.rs`
/// is not mistaken for a test.
pub(crate) fn is_meta(rel: &str) -> bool {
    let mut components = rel.split('/').peekable();
    while let Some(component) = components.next() {
        let is_file = components.peek().is_none();
        let name = component.to_lowercase();
        if !is_file && matches!(name.as_str(), "tests" | "test" | "fixtures" | "benches") {
            return true;
        }
        if is_file {
            let stem = name.rsplit_once('.').map(|(s, _)| s).unwrap_or(&name);
            if stem == "tests" || stem.starts_with("test_") || stem.ends_with("_test") {
                return true;
            }
        }
    }
    false
}

/// `use std::collections::HashMap` -> `std::collections::HashMap`.
fn normalize_import(name: &str) -> Option<String> {
    let mut rest = name.trim();
    for prefix in ["pub(crate)", "pub(super)", "pub(self)", "pub"] {
        if let Some(stripped) = rest.strip_prefix(prefix) {
            rest = stripped.trim_start();
            break;
        }
    }
    rest = rest
        .strip_prefix("use")?
        .trim()
        .trim_end_matches(';')
        .trim();
    (!rest.is_empty()).then(|| rest.to_string())
}

/// The `::`-separated head of a `use`, with the braced tail dropped.
fn import_segments(import: &str) -> Vec<String> {
    let head = import.split('{').next().unwrap_or(import);
    head.split("::")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s != "crate" && s != "self" && s != "super")
        .collect()
}

/// `src/oracle/diagnostics.rs` -> `["oracle", "diagnostics"]`.
///
/// Empty for a crate root, which no `use` names.
fn module_path(rel: &str) -> Vec<String> {
    let without_ext = rel.rsplit_once('.').map(|(s, _)| s).unwrap_or(rel);
    let mut parts: Vec<String> = without_ext
        .split('/')
        .filter(|p| !p.is_empty() && *p != "src")
        .map(str::to_string)
        .collect();
    match parts.last().map(String::as_str) {
        // `mod.rs` is named by its directory, `lib.rs`/`main.rs` by nothing.
        Some("mod") => {
            parts.pop();
        }
        Some("lib") | Some("main") => return Vec::new(),
        _ => {}
    }
    parts
}

/// Whether `needle` appears as a contiguous run inside `haystack`.
fn contains_run(haystack: &[String], needle: &[String]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

fn dedup(items: Vec<String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    items
        .into_iter()
        .filter(|i| seen.insert(i.clone()))
        .collect()
}

fn extension_of(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
}

/// Repository-relative with forward slashes, so an index written on Windows
/// reads the same as one written anywhere else.
fn relative_posix(root: &Path, path: &Path) -> Option<String> {
    let rel = path.strip_prefix(root).ok()?;
    Some(
        rel.components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two files defining `forward`, one importing the other.
    fn repo() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"demo\"\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("src/alpha.rs"),
            "use std::collections::HashMap;\n\
             \n\
             /// Picks which experts speak.\n\
             pub struct Router {\n\
             \x20   weights: Vec<f32>,\n\
             }\n\
             \n\
             impl Router {\n\
             \x20   pub fn forward(&self, x: f32) -> f32 {\n\
             \x20       x\n\
             \x20   }\n\
             \n\
             \x20   pub fn balance(&self, scores: &[f32]) -> f32 {\n\
             \x20       scores.iter().sum()\n\
             \x20   }\n\
             }\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("src/beta.rs"),
            "use crate::alpha::Router;\n\
             \n\
             pub struct Expert;\n\
             \n\
             impl Expert {\n\
             \x20   pub fn forward(&self, x: f32) -> f32 {\n\
             \x20       x * 2.0\n\
             \x20   }\n\
             }\n",
        )
        .unwrap();
        dir
    }

    fn scanned(dir: &tempfile::TempDir) -> Argus {
        let mut argus = Argus::new(dir.path());
        argus.scan();
        argus
    }

    #[test]
    fn two_files_defining_the_same_name_both_survive() {
        let dir = repo();
        let argus = scanned(&dir);
        let hits = argus.lookup("forward");
        let mut files: Vec<String> = hits
            .iter()
            .map(|h| h.symbol.file.display().to_string())
            .collect();
        files.sort();
        assert_eq!(files, vec!["src/alpha.rs", "src/beta.rs"]);
    }

    /// The disambiguation a flat symbol table cannot provide.
    #[test]
    fn a_method_carries_its_impl_as_its_qualified_name() {
        let dir = repo();
        let argus = scanned(&dir);
        let mut names: Vec<String> = argus
            .lookup("forward")
            .iter()
            .map(Located::qualname)
            .collect();
        names.sort();
        assert_eq!(names, vec!["Expert::forward", "Router::forward"]);

        let qualified = argus.lookup("Router::forward");
        assert_eq!(qualified.len(), 1);
        assert_eq!(
            qualified[0].symbol.file.display().to_string(),
            "src/alpha.rs"
        );
    }

    #[test]
    fn a_rescan_reparses_only_what_changed() {
        let dir = repo();
        let mut argus = scanned(&dir);

        let again = argus.scan();
        assert_eq!(again.parsed, 0, "{again}");
        assert_eq!(again.unchanged, again.scanned);

        std::fs::write(
            dir.path().join("src/beta.rs"),
            "pub fn only_thing() -> u32 { 1 }\n",
        )
        .unwrap();
        let after = argus.scan();
        assert_eq!(after.parsed, 1, "{after}");
        assert_eq!(
            argus
                .symbols_in("src/beta.rs")
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>(),
            vec!["only_thing"]
        );
    }

    #[test]
    fn a_deleted_file_leaves_the_index() {
        let dir = repo();
        let mut argus = scanned(&dir);

        std::fs::remove_file(dir.path().join("src/beta.rs")).unwrap();
        let report = argus.scan();

        assert_eq!(report.removed, 1);
        assert!(!argus.files().contains_key("src/beta.rs"));
        assert!(argus
            .lookup("forward")
            .iter()
            .all(|h| h.symbol.file.display().to_string() != "src/beta.rs"));
    }

    /// The case a regex scanner cannot get right. Citing a definition that is
    /// not there is worse than missing one that is.
    #[test]
    fn declarations_inside_strings_and_comments_are_not_symbols() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("lib.rs"),
            "//! A doc comment mentioning `fn ghost_in_the_comment()`.\n\
             pub fn real() -> u32 {\n\
             \x20   let sql = \"fn ghost_in_a_string(&self) -> ()\";\n\
             \x20   sql.len() as u32\n\
             }\n\
             // fn ghost_in_a_line_comment() {}\n",
        )
        .unwrap();

        let argus = scanned(&dir);
        let names: Vec<&str> = argus
            .symbols_in("lib.rs")
            .iter()
            .map(|s| s.name.as_str())
            .collect();
        assert!(names.contains(&"real"), "{names:?}");
        for ghost in [
            "ghost_in_a_string",
            "ghost_in_a_line_comment",
            "ghost_in_the_comment",
        ] {
            assert!(!names.contains(&ghost), "{ghost} is not a real definition");
        }
    }

    #[test]
    fn importers_are_found() {
        let dir = repo();
        let argus = scanned(&dir);
        assert_eq!(argus.importers_of("src/alpha.rs"), vec!["src/beta.rs"]);
        assert!(argus.importers_of("src/beta.rs").is_empty());
    }

    /// Regression: BM25 saturation, without which prose drowns out the answer.
    ///
    /// Under plain tf-idf a long file repeating "file" and "defines" hundreds of
    /// times outranked the short file that actually defines the thing asked
    /// about — a real retrieval returned the retrieval module itself for a
    /// question about the router.
    #[test]
    fn a_rare_term_beats_a_repeated_common_one() {
        let dir = tempfile::tempdir().unwrap();
        let chatter = "This file defines the file that defines files. ".repeat(200);
        std::fs::write(
            dir.path().join("chatty.rs"),
            format!("//! {chatter}\npub fn unrelated_helper() -> u32 {{ 1 }}\n"),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("moe.rs"),
            "/// Route tokens to experts.\npub fn apollo_router(scores: u32) -> u32 { scores }\n",
        )
        .unwrap();

        let argus = scanned(&dir);
        let ranked = argus.score_files("which file defines the apollo_router?");
        assert_eq!(ranked[0].0, "moe.rs", "ranked {ranked:?}");
    }

    /// The failure that motivated the `defs` field: a file that *uses* a name
    /// outranking the one that *defines* it, purely by being shorter.
    #[test]
    fn a_definition_outranks_a_mention() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("home.rs"),
            "pub struct Beacon {\n\
             \x20   power: u32,\n\
             }\n\
             impl Beacon {\n\
             \x20   pub fn new() -> Self { Beacon { power: 1 } }\n\
             }\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("user.rs"),
            "// Beacon, Beacon, Beacon.\npub fn light() -> u32 { 1 }\n",
        )
        .unwrap();

        let argus = scanned(&dir);
        let ranked = argus.score_files("Beacon");
        assert_eq!(ranked[0].0, "home.rs", "ranked {ranked:?}");
    }

    /// Rust unit tests share a file with the code they exercise, so their
    /// sentence-shaped names would otherwise become that file's definitions.
    #[test]
    fn test_functions_do_not_count_as_definitions() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("thing.rs"),
            "pub fn widget() -> u32 { 1 }\n\
             #[cfg(test)]\n\
             mod tests {\n\
             \x20   #[test]\n\
             \x20   fn deleting_a_sprocket_fails_verification() {}\n\
             }\n",
        )
        .unwrap();

        let argus = scanned(&dir);
        let defs = &argus.files()["thing.rs"].defs;
        assert!(defs.contains_key("widget"), "{defs:?}");
        assert!(
            !defs.contains_key("sprocket"),
            "a test name is not a definition: {defs:?}"
        );
        assert!(!defs.contains_key("verification"), "{defs:?}");
    }

    #[test]
    fn retrieval_prefers_an_exact_symbol_match() {
        let dir = repo();
        let argus = scanned(&dir);
        let hits = argus.retrieve("fix the balance method", 4000, 1);
        assert!(!hits.is_empty(), "expected at least one slice");
        assert_eq!(hits[0].symbol.as_deref(), Some("Router::balance"));
        assert!(
            hits[0].reason.contains("exact symbol match"),
            "{}",
            hits[0].reason
        );
    }

    #[test]
    fn retrieval_respects_its_budget() {
        let dir = repo();
        let argus = scanned(&dir);
        let hits = argus.retrieve("router expert forward balance", 200, 1);
        assert!(!hits.is_empty());
        // One slice may exceed the budget alone; the sum of the rest may not.
        let rest: usize = hits[1..].iter().map(|h| h.text.len()).sum();
        assert!(rest <= 200, "spent {rest} beyond the first slice");
    }

    #[test]
    fn every_slice_carries_provenance() {
        let dir = repo();
        let argus = scanned(&dir);
        let hits = argus.retrieve("router balance", 4000, 1);
        assert!(!hits.is_empty());
        for hit in &hits {
            assert!(!hit.reason.is_empty());
            assert!(!hit.file.is_empty());
            assert!(hit.start_line >= 1);
            assert!(hit.reference().starts_with(&hit.file));
        }
    }

    #[test]
    fn a_rendered_context_names_the_source_of_every_slice() {
        let dir = repo();
        let argus = scanned(&dir);
        let context = argus.context("balance", 4000, 0);
        assert!(context.text.contains("src/alpha.rs:"), "{}", context.text);
        assert!(context.text.contains("Router::balance"), "{}", context.text);
        assert_eq!(context.hits.len(), context.text.matches("# src/").count());
    }

    #[test]
    fn the_index_survives_a_save_load_round_trip() {
        let dir = repo();
        let argus = scanned(&dir);
        let before: Vec<String> = argus
            .lookup("forward")
            .iter()
            .map(Located::reference)
            .collect();
        let record = &argus.files()["src/alpha.rs"];
        assert!(
            !record.idents.is_empty() && !record.defs.is_empty() && !record.path_terms.is_empty(),
            "the fixture must be non-empty for this to mean anything"
        );

        let path = argus.save(None).unwrap();
        assert!(path.exists());

        let mut restored = Argus::new(dir.path());
        assert!(restored.load(Some(&path)).unwrap());

        let after: Vec<String> = restored
            .lookup("forward")
            .iter()
            .map(Located::reference)
            .collect();
        assert_eq!(after, before);
        assert_eq!(restored.files()["src/alpha.rs"].defs, record.defs);
        assert_eq!(
            restored.files()["src/alpha.rs"].path_terms,
            record.path_terms
        );

        // Nothing changed on disk, so a scan of a restored index re-parses nothing.
        assert_eq!(restored.scan().parsed, 0);
    }

    /// A stale index must force a rebuild, not silently rank with blank signals.
    #[test]
    fn an_index_from_an_older_schema_is_refused() {
        let dir = repo();
        let argus = scanned(&dir);
        let path = argus.save(None).unwrap();

        let mut blob: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        blob["version"] = serde_json::json!(0);
        std::fs::write(&path, serde_json::to_string(&blob).unwrap()).unwrap();

        let mut stale = Argus::new(dir.path());
        assert!(
            !stale.load(Some(&path)).unwrap(),
            "an older schema must be refused"
        );
        assert!(
            stale.files().is_empty(),
            "nothing should be half-loaded from a refused index"
        );
    }

    #[test]
    fn a_missing_index_is_a_false_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let mut argus = Argus::new(dir.path());
        assert!(!argus.load(None).unwrap());
    }

    #[test]
    fn prose_files_are_searchable_without_symbols() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("README.md"),
            "# Deployment\n\nRun the migration playbook before promoting a release.\n",
        )
        .unwrap();

        let argus = scanned(&dir);
        let ranked = argus.score_files("migration playbook");
        assert_eq!(ranked[0].0, "README.md", "ranked {ranked:?}");
        assert!(argus.files()["README.md"].symbols.is_empty());
    }

    /// An agent mid-edit leaves files broken constantly, and that is exactly
    /// when knowing what is in them matters most.
    #[test]
    fn a_broken_file_is_still_lexically_searchable() {
        let dir = repo();
        let mut argus = scanned(&dir);
        std::fs::write(dir.path().join("src/broken.rs"), "pub fn nope( {\n").unwrap();
        argus.scan();

        let record = &argus.files()["src/broken.rs"];
        assert!(
            !record.idents.is_empty(),
            "a broken file must stay findable"
        );
    }

    #[test]
    fn tokenizing_drops_stop_words_and_keeps_both_halves_of_a_name() {
        let bag = tokenize("pub fn verify_token(&self) -> bool");
        assert!(bag.contains_key("verify_token"), "{bag:?}");
        assert!(bag.contains_key("verify"), "{bag:?}");
        assert!(bag.contains_key("token"), "{bag:?}");
        for noise in ["pub", "fn", "self", "bool"] {
            assert!(
                !bag.contains_key(noise),
                "{noise} should be a stop word: {bag:?}"
            );
        }
    }

    #[test]
    fn a_module_path_names_the_directory_for_mod_rs_and_nothing_for_a_crate_root() {
        assert_eq!(
            module_path("src/oracle/diagnostics.rs"),
            vec!["oracle", "diagnostics"]
        );
        assert_eq!(module_path("src/oracle/mod.rs"), vec!["oracle"]);
        assert!(module_path("src/lib.rs").is_empty());
        assert!(module_path("src/main.rs").is_empty());
    }

    #[test]
    fn a_file_named_latest_is_not_mistaken_for_a_test() {
        assert!(is_meta("tests/harness_loop.rs"));
        assert!(is_meta("src/fixtures/passing.rs"));
        assert!(is_meta("src/thing_test.rs"));
        assert!(!is_meta("src/latest.rs"));
        assert!(!is_meta("src/oracle/mod.rs"));
    }

    #[test]
    fn an_empty_query_ranks_nothing_rather_than_everything() {
        let dir = repo();
        let argus = scanned(&dir);
        assert!(argus.score_files("").is_empty());
        assert!(argus.score_files("!!!").is_empty());
    }
}
