//! Language detection and the workspace-wide adapter.
//!
//! Mirrors `oracle.detect` / `oracle.tiers_for` on the Python side. Two
//! different questions share this module because they must not drift:
//!
//! * **What is this project?** Marker files at the workspace root, never a
//!   source-file count. A vendored `.py` under `node_modules` is not a
//!   Python project.
//! * **What should we check?** Every detected language's ladder, cheapest
//!   first *across* languages, so a syntax error is not preceded by someone
//!   else's test suite. No marker at all still gets the Python ladder —
//!   verifying nothing would make every task pass vacuously.
//!
//! Symbol extraction is deliberately wider than verification. Scribe still
//! has to index a `.rs` file in an unmarked tree; Oracle must not invent a
//! cargo run there.

use std::path::Path;

use super::adapter::{LanguageAdapter, Symbol, VerifyCommand};
use super::go::GoAdapter;
use super::node::NodeAdapter;
use super::python::PythonAdapter;
use super::rust::RustAdapter;

/// Every language present at `root`. Possibly none, possibly several.
pub fn detect(root: &Path) -> Vec<Box<dyn LanguageAdapter>> {
    all_adapters()
        .into_iter()
        .filter(|adapter| adapter.detect(root))
        .collect()
}

/// The ladder for whatever kind of project this is.
///
/// Polyglot repositories get every applicable ladder rather than a guess at
/// which language "really" owns the tree. Tier numbers are reassigned to
/// stay unique and dense, because the baseline is keyed on them.
pub fn tiers_for(root: &Path) -> Vec<VerifyCommand> {
    adapter_for(root).verify_commands()
}

/// Pick the adapter the rest of the harness talks to.
///
/// Always a [`WorkspaceAdapter`]: parsers for every language (so Scribe and
/// Argus keep working on mixed trees), and the verification ladder of the
/// languages the markers actually name.
pub fn adapter_for(root: &Path) -> Box<dyn LanguageAdapter> {
    Box::new(WorkspaceAdapter::for_root(root))
}

fn all_adapters() -> Vec<Box<dyn LanguageAdapter>> {
    // Order matches `oracle.py` ADAPTERS — detection is independent, but
    // stable order keeps test expectations and polyglot label prefixes
    // deterministic.
    vec![
        Box::new(PythonAdapter),
        Box::new(RustAdapter),
        Box::new(GoAdapter),
        Box::new(NodeAdapter),
    ]
}

/// One adapter that speaks every language the harness knows.
///
/// Verification is narrower than parsing: see [`WorkspaceAdapter::for_root`].
pub struct WorkspaceAdapter {
    name: String,
    parsers: Vec<Box<dyn LanguageAdapter>>,
    verifiers: Vec<Box<dyn LanguageAdapter>>,
}

impl WorkspaceAdapter {
    pub fn for_root(root: &Path) -> Self {
        let parsers = all_adapters();
        let detected = detect(root);
        let verifiers = if detected.is_empty() {
            vec![Box::new(PythonAdapter) as Box<dyn LanguageAdapter>]
        } else {
            detected
        };
        let name = verifiers
            .iter()
            .map(|a| a.name())
            .collect::<Vec<_>>()
            .join("+");
        WorkspaceAdapter {
            name,
            parsers,
            verifiers,
        }
    }

    fn parser_for(&self, path: &Path) -> Option<&dyn LanguageAdapter> {
        self.parsers
            .iter()
            .find(|adapter| adapter.handles(path))
            .map(|adapter| adapter.as_ref())
    }
}

impl LanguageAdapter for WorkspaceAdapter {
    fn name(&self) -> &str {
        &self.name
    }

    fn extensions(&self) -> &[&str] {
        // The union of every parser. `handles` does not use this — it asks
        // the parsers — but a caller that reads the list still needs it.
        &["py", "rs", "go", "js", "jsx", "ts", "tsx", "mjs", "cjs"]
    }

    fn detect(&self, root: &Path) -> bool {
        self.parsers.iter().any(|adapter| adapter.detect(root))
    }

    fn symbols(&self, source: &str, path: &Path) -> Vec<Symbol> {
        match self.parser_for(path) {
            Some(adapter) => adapter.symbols(source, path),
            None => Vec::new(),
        }
    }

    fn parses_cleanly(&self, source: &str, path: &Path) -> bool {
        self.parser_for(path)
            .map(|adapter| adapter.parses_cleanly(source, path))
            .unwrap_or(true)
    }

    fn verify_commands(&self) -> Vec<VerifyCommand> {
        merge_ladders(&self.verifiers)
    }

    fn handles(&self, path: &Path) -> bool {
        self.parsers.iter().any(|adapter| adapter.handles(path))
    }
}

/// Cheapest-first across languages, then rename and renumber.
fn merge_ladders(adapters: &[Box<dyn LanguageAdapter>]) -> Vec<VerifyCommand> {
    let multilingual = adapters.len() > 1;
    let mut pairs: Vec<(VerifyCommand, String)> = adapters
        .iter()
        .flat_map(|adapter| {
            let lang = adapter.name().to_string();
            adapter
                .verify_commands()
                .into_iter()
                .map(move |cmd| (cmd, lang.clone()))
        })
        .collect();
    pairs.sort_by(|a, b| a.0.tier.cmp(&b.0.tier).then_with(|| a.1.cmp(&b.1)));
    pairs
        .into_iter()
        .enumerate()
        .map(|(index, (mut cmd, lang))| {
            cmd.tier = (index + 1) as u8;
            if multilingual {
                cmd.label = format!("{lang}: {}", cmd.label);
            }
            cmd
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(root: &Path) -> Vec<String> {
        detect(root)
            .into_iter()
            .map(|a| a.name().to_string())
            .collect()
    }

    fn labels(root: &Path) -> Vec<String> {
        tiers_for(root).into_iter().map(|c| c.label).collect()
    }

    #[test]
    fn a_cargo_project_gets_the_rust_ladder() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname='x'\n").unwrap();
        assert_eq!(names(dir.path()), ["rust"]);
        assert_eq!(labels(dir.path()), ["cargo check", "clippy", "cargo test"]);
    }

    #[test]
    fn a_go_module_gets_the_go_ladder() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("go.mod"), "module x\n").unwrap();
        assert_eq!(labels(dir.path()), ["go build", "go vet", "go test"]);
    }

    #[test]
    fn a_node_project_gets_the_node_ladder() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("package.json"), "{}\n").unwrap();
        assert_eq!(labels(dir.path()), ["tsc", "eslint", "npm test"]);
    }

    #[test]
    fn a_polyglot_repository_gets_every_applicable_ladder() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname='x'\n").unwrap();
        std::fs::write(dir.path().join("pyproject.toml"), "[project]\nname='x'\n").unwrap();
        let mut found = names(dir.path());
        found.sort();
        assert_eq!(found, ["python", "rust"]);
        let labels = labels(dir.path());
        assert!(labels.iter().any(|l| l.contains("ruff")), "{labels:?}");
        assert!(
            labels.iter().any(|l| l.contains("cargo test")),
            "{labels:?}"
        );
    }

    #[test]
    fn ladders_interleave_cheapest_first_across_languages() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname='x'\n").unwrap();
        std::fs::write(dir.path().join("go.mod"), "module x\n").unwrap();
        let labels = labels(dir.path());
        let build = labels
            .iter()
            .position(|l| l == "rust: cargo check")
            .max(labels.iter().position(|l| l == "go: go build"))
            .unwrap();
        let tests = labels
            .iter()
            .position(|l| l == "rust: cargo test")
            .min(labels.iter().position(|l| l == "go: go test"))
            .unwrap();
        assert!(build < tests, "{labels:?}");
    }

    #[test]
    fn tier_numbers_stay_unique_and_dense() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname='x'\n").unwrap();
        std::fs::write(dir.path().join("go.mod"), "module x\n").unwrap();
        std::fs::write(dir.path().join("package.json"), "{}\n").unwrap();
        let numbers: Vec<u8> = tiers_for(dir.path()).into_iter().map(|c| c.tier).collect();
        let expected: Vec<u8> = (1..=numbers.len() as u8).collect();
        assert_eq!(numbers, expected);
    }

    #[test]
    fn labels_stay_distinct_when_several_languages_have_a_test_tier() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname='x'\n").unwrap();
        std::fs::write(dir.path().join("go.mod"), "module x\n").unwrap();
        let labels = labels(dir.path());
        let unique = labels
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(labels.len(), unique.len(), "{labels:?}");
    }

    #[test]
    fn an_unmarked_tree_still_gets_the_python_ladder() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(labels(dir.path()), ["ruff", "mypy", "pytest"]);
        assert!(names(dir.path()).is_empty());
    }

    #[test]
    fn an_unmarked_tree_still_indexes_rust_files() {
        let dir = tempfile::tempdir().unwrap();
        let adapter = adapter_for(dir.path());
        assert!(adapter.handles(Path::new("src/lib.rs")));
        assert!(adapter.handles(Path::new("pkg/mod.py")));
        assert_eq!(adapter.name(), "python");
    }
}
