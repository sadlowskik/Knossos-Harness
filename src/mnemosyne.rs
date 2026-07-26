//! Mnemosyne: the lossy tier of memory.
//!
//! Scribe holds declarations exactly; Mnemosyne holds *bodies* approximately,
//! retrieved on demand. That is the two-tier split the architecture describes —
//! approximate the prose, keep bit-exact what a compiler cares about — and it
//! is what lets the agent answer "where is authentication handled" rather than
//! only "does this identifier exist".
//!
//! # Why BM25 and not embeddings
//!
//! The tensor-level Mnemosyne compresses with learned cross-attention. The
//! obvious system-level translation is a vector index, and I deliberately did
//! not build one. Code queries are overwhelmingly *lexical* — a name, a call, a
//! literal, an error string — and lexical scoring is very hard to beat on those
//! while being exact, explainable, dependency-free and instant to rebuild.
//! Embeddings earn their cost on natural-language paraphrase, which is a small
//! fraction of what an agent asks a codebase.
//!
//! What is carried over is the *role* — fuzzy, lossy recall over context too
//! large to hold — not the mechanism. A vector index can slot in behind
//! [`Mnemosyne::search`] later if lexical retrieval is measurably insufficient.
//! `fastembed` would run locally, which matters: the alternative ships your
//! source to a third party.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use ignore::WalkBuilder;

use crate::scribe::{LanguageAdapter, Symbol};

/// Standard BM25 term-frequency saturation.
const K1: f64 = 1.2;
/// Standard BM25 length normalization.
const B: f64 = 0.75;

/// Chunks longer than this are split, so one enormous function cannot dominate.
const MAX_CHUNK_LINES: usize = 120;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub path: PathBuf,
    /// 1-indexed, inclusive.
    pub start_line: usize,
    pub end_line: usize,
    /// The declaration this chunk came from, when it came from one.
    pub symbol: Option<String>,
    pub text: String,
    /// Term -> count, precomputed at index time.
    terms: HashMap<String, usize>,
    length: usize,
}

impl Chunk {
    pub fn location(&self) -> String {
        match &self.symbol {
            Some(s) => format!("{}:{}-{} ({s})", self.path.display(), self.start_line, self.end_line),
            None => format!("{}:{}-{}", self.path.display(), self.start_line, self.end_line),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Hit<'a> {
    pub chunk: &'a Chunk,
    pub score: f64,
}

#[derive(Debug, Default)]
pub struct Mnemosyne {
    root: PathBuf,
    chunks: Vec<Chunk>,
    /// Document frequency per term.
    df: HashMap<String, usize>,
    avg_length: f64,
}

impl Mnemosyne {
    /// Index every file the adapter handles, chunked on declaration boundaries.
    pub fn build(root: impl Into<PathBuf>, adapter: &dyn LanguageAdapter) -> Result<Self> {
        let root = root.into();
        let mut chunks = Vec::new();

        for entry in WalkBuilder::new(&root).git_ignore(true).build().flatten() {
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            if !adapter.handles(entry.path()) {
                continue;
            }
            let Ok(source) = std::fs::read_to_string(entry.path()) else {
                continue;
            };
            let rel = entry.path().strip_prefix(&root).unwrap_or(entry.path());
            let symbols = adapter.symbols(&source, rel);
            chunks.extend(chunk_file(rel, &source, &symbols));
        }

        Ok(Self::from_chunks(root, chunks))
    }

    fn from_chunks(root: PathBuf, chunks: Vec<Chunk>) -> Self {
        let mut df: HashMap<String, usize> = HashMap::new();
        for chunk in &chunks {
            for term in chunk.terms.keys() {
                *df.entry(term.clone()).or_insert(0) += 1;
            }
        }
        let avg_length = if chunks.is_empty() {
            0.0
        } else {
            chunks.iter().map(|c| c.length as f64).sum::<f64>() / chunks.len() as f64
        };

        Mnemosyne { root, chunks, df, avg_length }
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Top-k chunks for a query, best first. Zero-scoring chunks are dropped.
    pub fn search(&self, query: &str, limit: usize) -> Vec<Hit<'_>> {
        let terms = tokenize(query);
        if terms.is_empty() || self.chunks.is_empty() {
            return Vec::new();
        }

        let n = self.chunks.len() as f64;
        let mut hits: Vec<Hit<'_>> = self
            .chunks
            .iter()
            .map(|chunk| {
                let mut score = 0.0;
                for term in &terms {
                    let Some(&tf) = chunk.terms.get(term) else {
                        continue;
                    };
                    let df = *self.df.get(term).unwrap_or(&0) as f64;
                    // BM25 IDF, the +1 keeping it non-negative for common terms.
                    let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                    let tf = tf as f64;
                    let norm = 1.0 - B + B * (chunk.length as f64 / self.avg_length.max(1.0));
                    score += idf * (tf * (K1 + 1.0)) / (tf + K1 * norm);
                }
                Hit { chunk, score }
            })
            .filter(|h| h.score > 0.0)
            .collect();

        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(limit);
        hits
    }
}

/// Split a file into chunks on declaration boundaries.
///
/// `symbols` arrives in pre-order, so an `impl` block precedes and spans the
/// methods inside it. Greedy non-overlapping selection over that order
/// therefore yields top-level items without needing a depth field, and any
/// lines outside every declaration (imports, module docs) become their own
/// chunk rather than being dropped.
fn chunk_file(path: &Path, source: &str, symbols: &[Symbol]) -> Vec<Chunk> {
    let lines: Vec<&str> = source.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut spans: Vec<(usize, usize, Option<String>)> = Vec::new();
    let mut covered_to = 0usize;

    for symbol in symbols {
        if symbol.line <= covered_to || symbol.end_line > lines.len() {
            continue; // inside a span already taken
        }
        // Lines between the last declaration and this one.
        if symbol.line > covered_to + 1 {
            spans.push((covered_to + 1, symbol.line - 1, None));
        }
        spans.push((symbol.line, symbol.end_line, Some(symbol.name.clone())));
        covered_to = symbol.end_line;
    }
    if covered_to < lines.len() {
        spans.push((covered_to + 1, lines.len(), None));
    }

    let mut chunks = Vec::new();
    for (start, end, name) in spans {
        // Split anything oversized so one huge function cannot dominate.
        let mut from = start;
        while from <= end {
            let to = (from + MAX_CHUNK_LINES - 1).min(end);
            let text = lines[from - 1..to].join("\n");
            if !text.trim().is_empty() {
                chunks.push(make_chunk(path, from, to, name.clone(), text));
            }
            from = to + 1;
        }
    }
    chunks
}

fn make_chunk(
    path: &Path,
    start_line: usize,
    end_line: usize,
    symbol: Option<String>,
    text: String,
) -> Chunk {
    // The symbol name is indexed alongside the body so a search for the name
    // ranks its own declaration highly.
    let mut tokens = tokenize(&text);
    if let Some(name) = &symbol {
        tokens.extend(tokenize(name));
    }

    let length = tokens.len();
    let mut terms: HashMap<String, usize> = HashMap::new();
    for token in tokens {
        *terms.entry(token).or_insert(0) += 1;
    }

    Chunk { path: path.to_path_buf(), start_line, end_line, symbol, text, terms, length }
}

/// Code-aware tokenization.
///
/// Splitting `snake_case` and `camelCase` into their parts *and* keeping the
/// whole identifier is what makes lexical search work on code: a query for
/// "parse config" then matches `parse_config_file`, while a query for the exact
/// identifier still scores it highest.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();

    for raw in text.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if raw.is_empty() {
            continue;
        }
        let lower = raw.to_lowercase();
        if lower.len() > 1 {
            out.push(lower.clone());
        }

        // snake_case parts
        let snake: Vec<&str> = raw.split('_').filter(|s| !s.is_empty()).collect();
        for part in &snake {
            for sub in split_camel(part) {
                if sub.len() > 1 && sub != lower {
                    out.push(sub);
                }
            }
        }
    }
    out
}

/// `parseHTTPResponse` -> ["parse", "http", "response"]
fn split_camel(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut parts = Vec::new();
    let mut current = String::new();

    for i in 0..chars.len() {
        let c = chars[i];
        let boundary = i > 0
            && c.is_uppercase()
            // lower->Upper, or the end of an acronym run (HTTPResponse)
            && (chars[i - 1].is_lowercase()
                || chars[i - 1].is_ascii_digit()
                || chars.get(i + 1).is_some_and(|n| n.is_lowercase()));

        if boundary && !current.is_empty() {
            parts.push(std::mem::take(&mut current));
        }
        current.push(c.to_ascii_lowercase());
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scribe::RustAdapter;

    fn workspace() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(
            dir.path().join("src/auth.rs"),
            "use std::fmt;\n\n\
             /// Verify a bearer token against the session store.\n\
             pub fn verify_token(token: &str) -> bool {\n\
             \x20   !token.is_empty()\n\
             }\n\n\
             pub fn hash_password(pw: &str) -> String {\n\
             \x20   format!(\"hashed:{pw}\")\n\
             }\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("src/render.rs"),
            "pub fn draw_triangle(sides: u32) -> u32 {\n\
             \x20   sides * 3\n\
             }\n",
        )
        .unwrap();
        dir
    }

    fn index(dir: &tempfile::TempDir) -> Mnemosyne {
        Mnemosyne::build(dir.path(), &RustAdapter).unwrap()
    }

    #[test]
    fn indexes_chunks_from_every_file() {
        let dir = workspace();
        let m = index(&dir);
        assert!(m.chunk_count() >= 3, "got {} chunks", m.chunk_count());
        assert!(!m.is_empty());
    }

    #[test]
    fn finds_the_right_function_for_a_lexical_query() {
        let dir = workspace();
        let m = index(&dir);
        let hits = m.search("verify token", 5);
        assert!(!hits.is_empty());
        assert!(
            hits[0].chunk.text.contains("verify_token"),
            "best hit was: {}",
            hits[0].chunk.location()
        );
    }

    /// The point of retrieval: a question Scribe structurally cannot answer.
    #[test]
    fn answers_a_question_about_where_something_lives() {
        let dir = workspace();
        let m = index(&dir);
        let hits = m.search("password hashing", 3);
        assert!(!hits.is_empty());
        assert!(hits[0].chunk.path.ends_with("auth.rs"), "{}", hits[0].chunk.location());
    }

    #[test]
    fn chunks_carry_their_symbol_and_line_span() {
        let dir = workspace();
        let m = index(&dir);
        let hit = &m.search("draw triangle", 1)[0];
        assert_eq!(hit.chunk.symbol.as_deref(), Some("draw_triangle"));
        assert_eq!(hit.chunk.start_line, 1);
        assert!(hit.chunk.end_line >= 3);
        assert!(hit.chunk.location().contains("draw_triangle"));
    }

    #[test]
    fn an_unrelated_query_returns_nothing_rather_than_noise() {
        let dir = workspace();
        assert!(index(&dir).search("kubernetes helm chart", 5).is_empty());
    }

    #[test]
    fn ranking_prefers_the_more_relevant_chunk() {
        let dir = workspace();
        let m = index(&dir);
        let hits = m.search("triangle", 5);
        assert!(hits[0].chunk.text.contains("draw_triangle"));
        if hits.len() > 1 {
            assert!(hits[0].score >= hits[1].score, "results must be sorted");
        }
    }

    #[test]
    fn tokenizer_splits_snake_case_and_keeps_the_whole_identifier() {
        let t = tokenize("verify_token");
        assert!(t.contains(&"verify_token".to_string()));
        assert!(t.contains(&"verify".to_string()));
        assert!(t.contains(&"token".to_string()));
    }

    #[test]
    fn tokenizer_splits_camel_case_including_acronyms() {
        let t = tokenize("parseHTTPResponse");
        assert!(t.contains(&"parse".to_string()), "{t:?}");
        assert!(t.contains(&"http".to_string()), "{t:?}");
        assert!(t.contains(&"response".to_string()), "{t:?}");
    }

    #[test]
    fn single_characters_are_dropped() {
        let t = tokenize("let x = 1;");
        assert!(!t.contains(&"x".to_string()));
        assert!(t.contains(&"let".to_string()));
    }

    #[test]
    fn an_empty_workspace_searches_without_panicking() {
        let dir = tempfile::tempdir().unwrap();
        let m = Mnemosyne::build(dir.path(), &RustAdapter).unwrap();
        assert!(m.is_empty());
        assert!(m.search("anything", 5).is_empty());
    }

    #[test]
    fn oversized_declarations_are_split() {
        let dir = tempfile::tempdir().unwrap();
        let body: String = (0..300).map(|i| format!("    let v{i} = {i};\n")).collect();
        std::fs::write(
            dir.path().join("big.rs"),
            format!("pub fn huge() {{\n{body}}}\n"),
        )
        .unwrap();

        let m = Mnemosyne::build(dir.path(), &RustAdapter).unwrap();
        assert!(m.chunk_count() > 1, "a 300-line function must not be one chunk");
        assert!(m
            .chunks
            .iter()
            .all(|c| c.end_line - c.start_line < MAX_CHUNK_LINES));
    }
}
