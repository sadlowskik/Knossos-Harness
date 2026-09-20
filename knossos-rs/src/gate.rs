//! The retrieval gate: decide whether context belongs in the prompt at all.
//!
//! Measured on gemma4:e4b, injecting retrieved excerpts into a *general* question
//! ("what problem does rotary positional embedding solve?") dropped the answer
//! score from 1.00 to 0.00 — the model stopped answering the question and started
//! describing this repository instead. The same injection on llama-3.3-70b cost
//! nothing. So irrelevant context is not free, and it is least free exactly where
//! this project is headed: small models.
//!
//! Always retrieving is therefore the wrong default. The gate decides.
//!
//! Signals, each named in the decision so a wrong call can be argued with rather
//! than guessed at — the same reason [`Retrieved::reason`] exists:
//!
//! | Signal | What it detects |
//! |---|---|
//! | anchor | "which file", "in this codebase" — asks about *here* |
//! | distinctive | a query word naming a symbol that occurs in very few files |
//! | concentration | retrieval that spikes on one file rather than smearing over ten |
//! | generality | "in general", "conceptually" — asks about the idea, not the code |
//!
//! `distinctive` is why raw symbol matching is not enough: "Mnemosyne" means this
//! repo, while "Router" or "forward" could mean anything.
//!
//! The hard cases are questions whose vocabulary is shared with the repo: "in a
//! mixture-of-experts layer, what is expert collapse?" matches `moe.rs` on every
//! content word and is still a general question. That is what `generality` is
//! for, and why it outweighs a bare symbol match.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, LazyLock, OnceLock};

use regex::Regex;

use crate::argus::{is_meta, split_identifier, Argus, Retrieved};

/// A gate ruling, and the signals that produced it.
#[derive(Debug, Clone)]
pub struct GateDecision {
    pub inject: bool,
    /// 0..1; at or above the threshold means inject.
    pub confidence: f64,
    pub reasons: Vec<String>,
}

impl std::fmt::Display for GateDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let verb = if self.inject { "inject" } else { "skip" };
        let reasons = if self.reasons.is_empty() {
            "no signal".to_string()
        } else {
            self.reasons.join("; ")
        };
        write!(f, "{verb} ({:.2}): {reasons}", self.confidence)
    }
}

/// Phrases that point at *this* codebase.
static ANCHORS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"\bwhich file\b",
        r"\bwhat file\b",
        r"\bwhich module\b",
        r"\bwhere is\b",
        r"\bwhere do(?:es)?\b",
        r"\bthis (?:repo|repository|codebase|project|crate)\b",
        r"\bour\b",
        r"\bwe (?:use|do|have|handle|call)\b",
        r"\bin the code\b",
        r"\bdefined? in\b",
        r"\bimplemented? in\b",
        r"\bthe codebase\b",
    ]
    .iter()
    .map(|p| Regex::new(p).expect("anchor pattern"))
    .collect()
});

/// Phrases that ask about an idea rather than an implementation.
///
/// Kept to high-precision ones only. An earlier draft included `what are`, which
/// skipped "What are Moirai's two gates called?" — about as repo-specific as a
/// question gets. A generality marker that fires on ordinary question grammar is
/// not a generality marker.
static GENERAL: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"\bin general\b",
        r"\bgenerally\b",
        r"\bconceptually\b",
        r"\bin theory\b",
        r"\bexplain the concept\b",
        r"\btypically\b",
        r"\busually\b",
        r"\bas a concept\b",
        r"\bwhat does .{1,40} mean\b",
    ]
    .iter()
    .map(|p| Regex::new(p).expect("generality pattern"))
    .collect()
});

/// Weaker: indefinite framing ("in *a* mixture-of-experts layer") describes a
/// category, where "the"/"our" would point at an instance. Half weight, because
/// it is a hint about grammar rather than a statement about intent.
static INDEFINITE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?i)\b(?:in|for|with|within)\s+an?\s+\w+[\w-]*\s+",
        r"(?:layer|model|network|system|transformer|architecture|module)\b",
    ))
    .expect("indefinite pattern")
});

/// Extensions kept in step with what the scanner actually indexes, plus the
/// prose formats a question is likely to name.
static FILENAME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b\w+\.(?:rs|py|toml|md|json|ya?ml|txt|cfg|ini|lock)\b")
        .expect("filename pattern")
});

static WORD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Za-z_][A-Za-z0-9_]*").expect("word pattern"));

/// Should this query get repository context?
///
/// Holds its [`Argus`] by [`Arc`] rather than borrowing it, because the real
/// consumer is long-lived: the executor decides once per turn and the name index
/// below costs a pass over every symbol in the repository to build. A borrowing
/// gate would have to be rebuilt at each call site that owns the index, which
/// throws that cache away every turn.
pub struct RetrievalGate {
    argus: Arc<Argus>,
    threshold: f64,
    /// Lowercased name fragment -> files defining it. Built once, on first use.
    ///
    /// `OnceLock` rather than `OnceCell` so the gate stays `Sync` and can sit in
    /// a struct held across an await.
    index: OnceLock<BTreeMap<String, BTreeSet<String>>>,
}

impl RetrievalGate {
    /// The gate STARTS above the threshold: injecting is the default and
    /// skipping must be argued for.
    ///
    /// The costs are asymmetric — a wrong skip loses a repo answer, which is the
    /// whole job, while a wrong inject loses a general answer the user could have
    /// asked anywhere. An earlier draft defaulted to skip and scored 60%, below
    /// the 87% of injecting unconditionally.
    pub const BASE: f64 = 0.75;

    const W_ANCHOR: f64 = 0.20;
    const W_DISTINCTIVE: f64 = 0.20;
    const W_FILENAME: f64 = 0.20;
    const W_CONCENTRATION: f64 = 0.05;
    const W_GENERAL: f64 = -0.45;
    const W_INDEFINITE: f64 = -0.22;

    /// A name occurring in at most this many files counts as distinctive.
    const DISTINCTIVE_MAX_FILES: usize = 3;
    /// Below this length a token is too generic to identify a codebase.
    const MIN_TOKEN: usize = 4;
    /// A term occurring as prose in more than this fraction of files is a word,
    /// not a name, and cannot make a question repo-specific.
    const COMMON_FRACTION: f64 = 0.25;

    pub fn new(argus: Arc<Argus>) -> Self {
        Self::with_threshold(argus, 0.5)
    }

    pub fn with_threshold(argus: Arc<Argus>, threshold: f64) -> Self {
        RetrievalGate {
            argus,
            threshold,
            index: OnceLock::new(),
        }
    }

    pub fn argus(&self) -> &Argus {
        &self.argus
    }

    /// Lowercased name fragment -> files it is defined in.
    ///
    /// Includes file stems and the camel/snake parts of every symbol, because
    /// people write "Moirai" for `MoiraiMixer` and "Proteus" for `ProteusBlock`.
    /// Exact-name matching alone misses the way names are actually spoken.
    pub fn name_index(&self) -> &BTreeMap<String, BTreeSet<String>> {
        self.index.get_or_init(|| self.build_name_index())
    }

    fn build_name_index(&self) -> BTreeMap<String, BTreeSet<String>> {
        // Test definitions are excluded. A Rust test is named after the sentence
        // it asserts — `syntax_error_degrades_instead_of_crashing` — so indexing
        // its parts fills the table with ordinary English ("error", "instead")
        // and makes every question look like it names something. Python got this
        // from a `test_` prefix rule; here the signal is the enclosing
        // `#[cfg(test)] mod tests`, which `Located::in_test` already resolves.
        let mut index: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (rel, rec) in self.argus.files() {
            if !is_meta(rel) {
                let stem = rel.rsplit('/').next().unwrap_or(rel);
                let stem = stem.rsplit_once('.').map(|(s, _)| s).unwrap_or(stem);
                index
                    .entry(stem.to_lowercase())
                    .or_default()
                    .insert(rel.clone());
            }
            for located in rec.located() {
                if located.in_test {
                    continue;
                }
                let file = located.symbol.file.to_string_lossy().replace('\\', "/");
                for part in split_identifier(&located.symbol.name) {
                    if part.len() >= Self::MIN_TOKEN {
                        index.entry(part).or_default().insert(file.clone());
                    }
                }
            }
        }

        // A word that appears as prose all over the corpus is not a name, even
        // if some symbol happens to contain it. "expert" and "layer" are in half
        // these files; "Mnemosyne" is in a handful.
        let common = (Self::COMMON_FRACTION * self.argus.files().len() as f64).max(3.0) as usize;
        let mut text_df: BTreeMap<&str, usize> = BTreeMap::new();
        for rec in self.argus.files().values() {
            for term in rec.idents.keys() {
                *text_df.entry(term.as_str()).or_insert(0) += 1;
            }
        }
        index
            .into_iter()
            .filter(|(name, _)| text_df.get(name.as_str()).copied().unwrap_or(0) <= common)
            .collect()
    }

    /// Query words naming something that lives in very few files.
    ///
    /// Frequency separates a name from a word: `Mnemosyne` is defined once and
    /// means this repository, while `forward` is defined in a dozen files and
    /// means nothing in particular.
    pub fn distinctive_hits(&self, query: &str) -> Vec<String> {
        let index = self.name_index();
        let mut seen = BTreeSet::new();
        let mut found = Vec::new();
        for m in WORD.find_iter(query) {
            let token = m.as_str();
            if token.len() < Self::MIN_TOKEN || !seen.insert(token) {
                continue;
            }
            if index.get(&token.to_lowercase()).is_some_and(|files| {
                !files.is_empty() && files.len() <= Self::DISTINCTIVE_MAX_FILES
            }) {
                found.push(token.to_string());
            }
        }
        found
    }

    /// True when retrieval spiked rather than smeared.
    ///
    /// A question about one thing in the repo scores one file far above the
    /// rest. A question whose words are merely common in the repo scores many
    /// files similarly.
    fn concentrated(scores: &[f64]) -> bool {
        let mut positive: Vec<f64> = scores.iter().copied().filter(|s| *s > 0.0).collect();
        if positive.len() < 3 {
            return !positive.is_empty();
        }
        positive.sort_by(|a, b| a.partial_cmp(b).expect("scores are finite"));
        let top = *positive.last().expect("non-empty");
        let mid = positive.len() / 2;
        let median = if positive.len().is_multiple_of(2) {
            (positive[mid - 1] + positive[mid]) / 2.0
        } else {
            positive[mid]
        };
        median > 0.0 && top >= 2.0 * median
    }

    /// Rule on a query, optionally reusing hits already retrieved for it.
    pub fn decide(&self, query: &str, hits: Option<&[Retrieved]>) -> GateDecision {
        let low = query.to_lowercase();
        let mut score = Self::BASE;
        let mut reasons: Vec<String> = Vec::new();

        let anchored = ANCHORS.iter().any(|p| p.is_match(&low));
        if anchored {
            score += Self::W_ANCHOR;
            reasons.push("asks about this codebase".into());
        }

        let named_file = FILENAME.is_match(query);
        if named_file {
            score += Self::W_FILENAME;
            reasons.push("names a file".into());
        }

        let distinctive = self.distinctive_hits(query);
        if !distinctive.is_empty() {
            score += Self::W_DISTINCTIVE;
            let shown: Vec<&str> = distinctive.iter().take(3).map(|s| s.as_str()).collect();
            reasons.push(format!("names {}", shown.join(", ")));
        }

        let general = GENERAL.iter().any(|p| p.is_match(&low));
        if general {
            score += Self::W_GENERAL;
            reasons.push("generality phrasing".into());
        }

        let indefinite = INDEFINITE.is_match(query);
        if indefinite {
            score += Self::W_INDEFINITE;
            reasons.push("indefinite framing (a category, not this instance)".into());
        }

        // Decisive rule, because additive weights got this wrong: an incidental
        // name match ("post" -> `_post`, "Explain" -> `_explain`) was cancelling
        // an explicit "in general" and dragging the score back over the line.
        // Evidence of a general question wins unless something points at *here*
        // — an anchor phrase, a filename, or a proper name that only means
        // something in this repo. The first word is excluded from the
        // proper-name test, since every sentence capitalises it.
        let proper: Vec<&String> = distinctive
            .iter()
            .filter(|d| {
                d.chars().next().is_some_and(char::is_uppercase)
                    && !low.starts_with(&d.to_lowercase())
            })
            .collect();
        let veto = anchored || named_file || !proper.is_empty();
        if (general || indefinite) && !veto {
            reasons.push("no anchor, filename or proper name to override it".into());
            return GateDecision {
                inject: false,
                confidence: score.min(0.3),
                reasons,
            };
        }
        if let Some(first) = proper.first() {
            reasons.push(format!("proper name: {first}"));
        }

        let file_scores: Vec<f64> = match hits {
            Some(hits) => hits.iter().map(|h| h.score).collect(),
            None => self
                .argus
                .score_files(query)
                .into_iter()
                .map(|(_, s)| s)
                .collect(),
        };
        if Self::concentrated(&file_scores) {
            score += Self::W_CONCENTRATION;
            reasons.push("retrieval concentrated".into());
        }

        let score = score.clamp(0.0, 1.0);
        GateDecision {
            inject: score >= self.threshold,
            confidence: score,
            reasons,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// A repo with one distinctive name and one thoroughly generic one.
    fn fixture() -> (TempDir, Arc<Argus>) {
        let dir = TempDir::new().expect("tempdir");
        let root = dir.path();
        fs::write(
            root.join("mnemosyne.rs"),
            "/// Gist memory.\n\
             pub struct MnemosyneBank;\n\
             impl MnemosyneBank {\n\
             \x20   pub fn forward(&self, x: usize) -> usize { x }\n\
             }\n",
        )
        .expect("write mnemosyne.rs");
        fs::write(
            root.join("moe.rs"),
            "pub struct Router;\n\
             impl Router {\n\
             \x20   pub fn forward(&self, x: usize) -> usize { x }\n\
             }\n\
             \n\
             pub fn load_balance_loss(scores: &[f64]) -> f64 {\n\
             \x20   scores.iter().sum::<f64>() / scores.len() as f64\n\
             }\n",
        )
        .expect("write moe.rs");
        // Unit tests in the same file as the code, which is where Rust puts
        // them — the gate has to strip these without the help of a filename.
        fs::write(
            root.join("things.rs"),
            "pub fn thing() {}\n\
             \n\
             #[cfg(test)]\n\
             mod tests {\n\
             \x20   #[test]\n\
             \x20   fn router_does_not_collapse_when_training_a_layer() {}\n\
             \x20   #[test]\n\
             \x20   fn gradient_clipping_helps_stabilise_general_training() {}\n\
             }\n",
        )
        .expect("write things.rs");

        let mut argus = Argus::new(root);
        argus.scan();
        (dir, Arc::new(argus))
    }

    // --------------------------------------------------------------- defaults

    #[test]
    fn injecting_is_the_default() {
        let (_dir, argus) = fixture();
        let gate = RetrievalGate::new(Arc::clone(&argus));
        let d = gate.decide("how is the halting probability computed here", None);
        assert!(d.inject, "{d}");
    }

    #[test]
    fn a_bare_question_still_injects() {
        let (_dir, argus) = fixture();
        let gate = RetrievalGate::new(Arc::clone(&argus));
        assert!(gate.decide("what does forward do", None).inject);
    }

    // ------------------------------------------------------------------ skips

    #[test]
    fn generality_phrasing_skips() {
        let (_dir, argus) = fixture();
        let gate = RetrievalGate::new(Arc::clone(&argus));
        for query in [
            "what problem does rotary positional embedding solve, in general?",
            "how does KV caching speed up inference, typically?",
            "explain the concept of 4-bit quantization",
            "what is attention, conceptually?",
        ] {
            let d = gate.decide(query, None);
            assert!(!d.inject, "{query}: {d}");
        }
    }

    #[test]
    fn indefinite_framing_skips() {
        let (_dir, argus) = fixture();
        let gate = RetrievalGate::new(Arc::clone(&argus));
        let d = gate.decide(
            "in a mixture-of-experts layer, what is expert collapse?",
            None,
        );
        assert!(!d.inject, "{d}");
    }

    // ----------------------------------------------------------------- vetoes

    #[test]
    fn an_anchor_overrides_generality() {
        let (_dir, argus) = fixture();
        let gate = RetrievalGate::new(Arc::clone(&argus));
        let d = gate.decide("which file handles quantization, in general terms?", None);
        assert!(d.inject, "{d}");
    }

    #[test]
    fn a_filename_overrides_generality() {
        let (_dir, argus) = fixture();
        let gate = RetrievalGate::new(Arc::clone(&argus));
        let d = gate.decide("what does moe.rs do, generally?", None);
        assert!(d.inject, "{d}");
    }

    #[test]
    fn a_proper_name_overrides_generality() {
        let (_dir, argus) = fixture();
        let gate = RetrievalGate::new(Arc::clone(&argus));
        let d = gate.decide("how does MnemosyneBank work, in general?", None);
        assert!(d.inject, "{d}");
    }

    /// Every sentence capitalises its first word; that is grammar, not a name.
    ///
    /// Regression: "Explain the concept of quantization" matched `_explain` and
    /// the leading capital made it look like a deliberate proper noun, vetoing a
    /// skip that should have happened.
    #[test]
    fn a_sentence_initial_capital_is_not_a_proper_name() {
        let (_dir, argus) = fixture();
        let gate = RetrievalGate::new(Arc::clone(&argus));
        let d = gate.decide("Router the concept of expert collapse, in general?", None);
        assert!(!d.inject, "{d}");
    }

    // ---------------------------------------------------------- name indexing

    /// Test names are sentences, so indexing them makes everything look local.
    #[test]
    fn test_function_names_do_not_become_repo_vocabulary() {
        let (_dir, argus) = fixture();
        let gate = RetrievalGate::new(Arc::clone(&argus));
        let index = gate.name_index();
        for word in ["gradient", "clipping", "stabilise", "collapse", "training"] {
            assert!(!index.contains_key(word), "{word} leaked into the index");
        }
    }

    #[test]
    fn a_distinctive_name_is_recognised() {
        let (_dir, argus) = fixture();
        let gate = RetrievalGate::new(Arc::clone(&argus));
        assert!(gate.name_index().contains_key("mnemosyne"));
        assert!(!gate
            .distinctive_hits("how does Mnemosyne compress a segment")
            .is_empty());
    }

    #[test]
    fn short_tokens_are_never_distinctive() {
        let (_dir, argus) = fixture();
        let gate = RetrievalGate::new(Arc::clone(&argus));
        assert!(gate.distinctive_hits("what is x").is_empty());
    }

    // ------------------------------------------------------------- reporting

    /// A gate that cannot be argued with cannot be debugged.
    #[test]
    fn every_decision_explains_itself() {
        let (_dir, argus) = fixture();
        let gate = RetrievalGate::new(Arc::clone(&argus));
        for query in [
            "which file defines the router",
            "what is attention, in general?",
        ] {
            let d = gate.decide(query, None);
            assert!(!d.reasons.is_empty(), "{query}");
            assert!((0.0..=1.0).contains(&d.confidence), "{query}");
            let shown = d.to_string();
            assert!(
                shown.contains("inject") || shown.contains("skip"),
                "{shown}"
            );
        }
    }

    #[test]
    fn threshold_is_adjustable() {
        let (_dir, argus) = fixture();
        let query = "how is the halting probability computed here";
        assert!(
            RetrievalGate::with_threshold(Arc::clone(&argus), 0.0)
                .decide(query, None)
                .inject
        );
        assert!(
            !RetrievalGate::with_threshold(Arc::clone(&argus), 1.01)
                .decide(query, None)
                .inject
        );
    }

    /// Passing hits should not change the ruling for a query that already
    /// retrieves well — the gate reuses them only to skip a second scoring pass.
    #[test]
    fn supplied_hits_stand_in_for_a_second_scoring_pass() {
        let (_dir, argus) = fixture();
        let gate = RetrievalGate::new(Arc::clone(&argus));
        let query = "how does MnemosyneBank store a gist";
        let hits = argus.retrieve(query, 2000, 0);
        assert_eq!(
            gate.decide(query, Some(&hits)).inject,
            gate.decide(query, None).inject,
        );
    }
}
