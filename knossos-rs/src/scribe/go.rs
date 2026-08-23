//! Go language adapter.

use std::path::Path;

use super::adapter::{LanguageAdapter, Symbol, SymbolKind, VerifyCommand, Visibility};

pub struct GoAdapter;

impl LanguageAdapter for GoAdapter {
    fn name(&self) -> &str {
        "go"
    }

    fn extensions(&self) -> &[&str] {
        &["go"]
    }

    fn detect(&self, root: &Path) -> bool {
        root.join("go.mod").exists()
    }

    fn symbols(&self, source: &str, path: &Path) -> Vec<Symbol> {
        let mut out = Vec::new();
        for (idx, line) in source.lines().enumerate() {
            let trimmed = line.trim_start();
            if let Some(name) = func_name(trimmed) {
                out.push(Symbol {
                    kind: SymbolKind::Function,
                    name: name.to_string(),
                    signature: trimmed.trim_end().to_string(),
                    file: path.to_path_buf(),
                    line: idx + 1,
                    end_line: idx + 1,
                    visibility: visibility(name),
                });
            } else if let Some(name) = type_name(trimmed) {
                out.push(Symbol {
                    kind: SymbolKind::Struct,
                    name: name.to_string(),
                    signature: trimmed.trim_end().to_string(),
                    file: path.to_path_buf(),
                    line: idx + 1,
                    end_line: idx + 1,
                    visibility: visibility(name),
                });
            }
        }
        out
    }

    fn parses_cleanly(&self, _source: &str, _path: &Path) -> bool {
        // No in-process Go parser. Fail open: `go build` is the real check.
        true
    }

    fn verify_commands(&self) -> Vec<VerifyCommand> {
        // `./...` is the whole module by design — Go has no cheaper
        // granularity that is still correct, since a change to one package
        // can break any package that imports it.
        vec![
            VerifyCommand::new(1, "go build", "go", ["build", "./..."]).required(),
            VerifyCommand::new(2, "go vet", "go", ["vet", "./..."]).advisory(),
            VerifyCommand::new(3, "go test", "go", ["test", "./..."]).required(),
        ]
    }
}

fn visibility(name: &str) -> Visibility {
    match name.chars().next() {
        Some(c) if c.is_uppercase() => Visibility::Public,
        _ => Visibility::Private,
    }
}

fn func_name(trimmed: &str) -> Option<&str> {
    let rest = trimmed.strip_prefix("func ")?;
    let rest = if let Some(after) = rest.strip_prefix('(') {
        after.split_once(')')?.1.trim_start()
    } else {
        rest
    };
    ident_at_start(rest)
}

fn type_name(trimmed: &str) -> Option<&str> {
    ident_at_start(trimmed.strip_prefix("type ")?)
}

fn ident_at_start(s: &str) -> Option<&str> {
    let end = s
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .unwrap_or(s.len());
    let name = &s[..end];
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn detects_go_mod() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!GoAdapter.detect(dir.path()));
        std::fs::write(dir.path().join("go.mod"), "module x\n").unwrap();
        assert!(GoAdapter.detect(dir.path()));
    }

    #[test]
    fn symbols_include_methods() {
        let src = "func (s *Srv) Serve() {}\nfunc helper() {}\n";
        let found = GoAdapter.symbols(src, &PathBuf::from("s.go"));
        let names: Vec<_> = found.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["Serve", "helper"]);
        assert_eq!(found[0].visibility, Visibility::Public);
        assert_eq!(found[1].visibility, Visibility::Private);
    }

    #[test]
    fn ladder_is_build_then_vet_then_test() {
        let labels: Vec<_> = GoAdapter
            .verify_commands()
            .into_iter()
            .map(|c| c.label)
            .collect();
        assert_eq!(labels, ["go build", "go vet", "go test"]);
    }
}
