//! Node / TypeScript language adapter.
//!
//! Deliberately `node_modules/.bin/…` and never `npx`: npx will download and
//! execute a package that is not installed, which turns a verification step
//! into arbitrary code execution sourced from a name in a config file. If the
//! project has not installed its own toolchain, the tier is skipped — absent
//! tooling is not evidence of broken code.

use std::path::Path;

use super::adapter::{LanguageAdapter, Symbol, SymbolKind, VerifyCommand, Visibility};

pub struct NodeAdapter;

impl LanguageAdapter for NodeAdapter {
    fn name(&self) -> &str {
        "node"
    }

    fn extensions(&self) -> &[&str] {
        &["js", "jsx", "ts", "tsx", "mjs", "cjs"]
    }

    fn detect(&self, root: &Path) -> bool {
        root.join("package.json").exists()
    }

    fn symbols(&self, source: &str, path: &Path) -> Vec<Symbol> {
        let mut out = Vec::new();
        for (idx, line) in source.lines().enumerate() {
            let trimmed = line.trim_start();
            if let Some(name) = function_name(trimmed) {
                out.push(Symbol {
                    kind: SymbolKind::Function,
                    name: name.to_string(),
                    signature: trimmed.trim_end_matches('{').trim().to_string(),
                    file: path.to_path_buf(),
                    line: idx + 1,
                    end_line: idx + 1,
                    visibility: Visibility::Public,
                });
            } else if let Some(name) = class_name(trimmed) {
                out.push(Symbol {
                    kind: SymbolKind::Class,
                    name: name.to_string(),
                    signature: trimmed.trim_end_matches('{').trim().to_string(),
                    file: path.to_path_buf(),
                    line: idx + 1,
                    end_line: idx + 1,
                    visibility: Visibility::Public,
                });
            }
        }
        out
    }

    fn parses_cleanly(&self, _source: &str, _path: &Path) -> bool {
        true
    }

    fn verify_commands(&self) -> Vec<VerifyCommand> {
        vec![
            VerifyCommand::new(1, "tsc", local_bin("tsc"), ["--noEmit"]).required(),
            VerifyCommand::new(2, "eslint", local_bin("eslint"), ["."])
                .advisory()
                .scopes([".js", ".jsx", ".ts", ".tsx"]),
            VerifyCommand::new(3, "npm test", "npm", ["test", "--silent"]).required(),
        ]
    }
}

/// Path to an on-disk binary the project itself installed.
///
/// On Windows the shim is `name.cmd`. Using the Unix name there fails to
/// spawn and the tier is skipped — which is safe, but means a Windows
/// machine with a fully installed toolchain would never verify. Prefer the
/// shim that will actually start.
fn local_bin(name: &str) -> String {
    let dir = Path::new("node_modules").join(".bin");
    let path = if cfg!(windows) {
        dir.join(format!("{name}.cmd"))
    } else {
        dir.join(name)
    };
    path.to_string_lossy().into_owned()
}

fn function_name(trimmed: &str) -> Option<&str> {
    let rest = trimmed
        .strip_prefix("export async function ")
        .or_else(|| trimmed.strip_prefix("export function "))
        .or_else(|| trimmed.strip_prefix("async function "))
        .or_else(|| trimmed.strip_prefix("function "))?;
    ident_at_start(rest)
}

fn class_name(trimmed: &str) -> Option<&str> {
    let rest = trimmed
        .strip_prefix("export class ")
        .or_else(|| trimmed.strip_prefix("class "))?;
    ident_at_start(rest)
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

    #[test]
    fn detects_package_json() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!NodeAdapter.detect(dir.path()));
        std::fs::write(dir.path().join("package.json"), "{}\n").unwrap();
        assert!(NodeAdapter.detect(dir.path()));
    }

    #[test]
    fn the_ladder_never_reaches_for_npx() {
        let programs: Vec<_> = NodeAdapter
            .verify_commands()
            .into_iter()
            .map(|c| c.program)
            .collect();
        assert!(
            !programs.iter().any(|p| p.contains("npx")),
            "npx downloads and executes a package named in a config file: {programs:?}"
        );
        assert!(
            programs.iter().any(|p| p.contains("node_modules")),
            "expected an on-disk toolchain, got {programs:?}"
        );
    }

    #[test]
    fn ladder_labels_match_the_python_side() {
        let labels: Vec<_> = NodeAdapter
            .verify_commands()
            .into_iter()
            .map(|c| c.label)
            .collect();
        assert_eq!(labels, ["tsc", "eslint", "npm test"]);
    }
}
