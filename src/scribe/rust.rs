//! Rust language adapter: tree-sitter for symbols, cargo for verification.
//!
//! tree-sitter rather than `syn` or regex for one specific reason: it is
//! error-tolerant. A file with a syntax error still yields a parse tree where
//! the broken region is an `ERROR` node and every valid sibling is intact, so
//! Scribe keeps answering "what symbols exist here" while the agent is
//! mid-edit. `syn` returns `Err` for the whole file, and the Python `ast`
//! module in the research repo raises — both give you nothing exactly when you
//! need something.

use std::path::Path;

use tree_sitter::{Node, Parser};

use super::adapter::{LanguageAdapter, Symbol, SymbolKind, VerifyCommand, Visibility};

pub struct RustAdapter;

impl RustAdapter {
    fn parser() -> Parser {
        let mut p = Parser::new();
        p.set_language(&tree_sitter_rust::LANGUAGE.into())
            .expect("tree-sitter-rust grammar is incompatible with the linked tree-sitter");
        p
    }
}

impl LanguageAdapter for RustAdapter {
    fn name(&self) -> &str {
        "rust"
    }

    fn extensions(&self) -> &[&str] {
        &["rs"]
    }

    fn detect(&self, root: &Path) -> bool {
        root.join("Cargo.toml").exists()
    }

    fn symbols(&self, source: &str, path: &Path) -> Vec<Symbol> {
        let mut parser = Self::parser();
        let Some(tree) = parser.parse(source, None) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        walk(tree.root_node(), source, path, &mut out);
        out
    }

    fn parses_cleanly(&self, source: &str) -> bool {
        let mut parser = Self::parser();
        match parser.parse(source, None) {
            Some(tree) => !tree.root_node().has_error(),
            None => false,
        }
    }

    fn verify_commands(&self) -> Vec<VerifyCommand> {
        vec![
            VerifyCommand {
                tier: 1,
                label: "cargo check",
                program: "cargo",
                args: vec![
                    "check".into(),
                    "--message-format=json".into(),
                    "--quiet".into(),
                ],
                structured: true,
            },
            VerifyCommand {
                tier: 2,
                label: "cargo clippy",
                program: "cargo",
                args: vec![
                    "clippy".into(),
                    "--message-format=json".into(),
                    "--quiet".into(),
                ],
                structured: true,
            },
            VerifyCommand {
                tier: 3,
                label: "cargo test",
                program: "cargo",
                args: vec!["test".into(), "--quiet".into()],
                structured: false,
            },
        ]
    }
}

fn walk(node: Node, src: &str, path: &Path, out: &mut Vec<Symbol>) {
    if let Some(sym) = symbol_for(node, src, path) {
        out.push(sym);
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk(child, src, path, out);
    }
}

fn symbol_for(node: Node, src: &str, path: &Path) -> Option<Symbol> {
    let kind = match node.kind() {
        "function_item" => SymbolKind::Function,
        "struct_item" => SymbolKind::Struct,
        "enum_item" => SymbolKind::Enum,
        "trait_item" => SymbolKind::Trait,
        "impl_item" => SymbolKind::Impl,
        "mod_item" => SymbolKind::Module,
        "const_item" | "static_item" => SymbolKind::Const,
        "type_item" => SymbolKind::TypeAlias,
        "use_declaration" => SymbolKind::Import,
        _ => return None,
    };

    let name = match kind {
        SymbolKind::Import => text(node, src).trim_end_matches(';').to_string(),
        // `impl Foo for Bar` names the type being implemented.
        SymbolKind::Impl => node
            .child_by_field_name("type")
            .map(|n| text(n, src).to_string())
            .unwrap_or_else(|| "impl".to_string()),
        _ => node
            .child_by_field_name("name")
            .map(|n| text(n, src).to_string())?,
    };

    Some(Symbol {
        kind,
        name,
        signature: signature(node, src),
        file: path.to_path_buf(),
        line: node.start_position().row + 1,
        visibility: visibility(node, src),
    })
}

fn text<'a>(node: Node, src: &'a str) -> &'a str {
    node.utf8_text(src.as_bytes()).unwrap_or("")
}

/// The declaration without its body, collapsed onto one line.
fn signature(node: Node, src: &str) -> String {
    let start = node.start_byte();
    let end = node
        .child_by_field_name("body")
        .map(|b| b.start_byte())
        .unwrap_or_else(|| node.end_byte());

    let raw = src.get(start..end).unwrap_or("").trim();
    let collapsed: String = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed.trim_end_matches(['{', ';']).trim().to_string()
}

fn visibility(node: Node, src: &str) -> Visibility {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "visibility_modifier" {
            let t = text(child, src);
            return if t == "pub" { Visibility::Public } else { Visibility::Restricted };
        }
    }
    Visibility::Private
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const SAMPLE: &str = r#"
use std::collections::HashMap;

pub struct Config {
    pub name: String,
}

pub(crate) enum Mode { Fast, Slow }

pub trait Runner {
    fn run(&self) -> u32;
}

impl Runner for Config {
    fn run(&self) -> u32 { 7 }
}

pub fn build(name: &str, retries: u32) -> Config {
    Config { name: name.to_string() }
}

fn helper() {}

const LIMIT: usize = 10;
"#;

    fn syms(src: &str) -> Vec<Symbol> {
        RustAdapter.symbols(src, &PathBuf::from("lib.rs"))
    }

    fn find<'a>(s: &'a [Symbol], name: &str) -> &'a Symbol {
        s.iter().find(|x| x.name == name).unwrap_or_else(|| panic!("no symbol named {name}"))
    }

    #[test]
    fn extracts_every_declaration_kind() {
        let s = syms(SAMPLE);
        assert_eq!(find(&s, "Config").kind, SymbolKind::Struct);
        assert_eq!(find(&s, "Mode").kind, SymbolKind::Enum);
        assert_eq!(find(&s, "Runner").kind, SymbolKind::Trait);
        assert_eq!(find(&s, "build").kind, SymbolKind::Function);
        assert_eq!(find(&s, "LIMIT").kind, SymbolKind::Const);
        assert!(s.iter().any(|x| x.kind == SymbolKind::Import));
        assert!(s.iter().any(|x| x.kind == SymbolKind::Impl));
    }

    #[test]
    fn signature_excludes_the_body() {
        let s = syms(SAMPLE);
        let build = find(&s, "build");
        assert_eq!(build.signature, "pub fn build(name: &str, retries: u32) -> Config");
        assert!(!build.signature.contains("to_string"));
    }

    #[test]
    fn records_visibility_and_line_numbers() {
        let s = syms(SAMPLE);
        assert_eq!(find(&s, "build").visibility, Visibility::Public);
        assert_eq!(find(&s, "helper").visibility, Visibility::Private);
        assert_eq!(find(&s, "Mode").visibility, Visibility::Restricted);
        assert!(find(&s, "build").line > 1);
    }

    /// The claim that justified choosing tree-sitter. Tested, not asserted.
    #[test]
    fn still_extracts_symbols_from_a_file_with_a_syntax_error() {
        let broken = r#"
pub fn good_one() -> u32 { 1 }

pub fn broken( {{{ ~~~ not rust at all

pub fn also_good(x: u32) -> u32 { x }

pub struct StillHere;
"#;
        let s = syms(broken);
        let names: Vec<_> = s.iter().map(|x| x.name.as_str()).collect();
        assert!(names.contains(&"good_one"), "got {names:?}");
        assert!(names.contains(&"StillHere"), "got {names:?}");
        assert!(!s.is_empty());
    }

    #[test]
    fn parses_cleanly_distinguishes_valid_from_broken() {
        assert!(RustAdapter.parses_cleanly("pub fn a() {}"));
        assert!(!RustAdapter.parses_cleanly("pub fn a( {{{ ~~~"));
    }

    #[test]
    fn verify_chain_is_ordered_cheapest_first() {
        let cmds = RustAdapter.verify_commands();
        let tiers: Vec<u8> = cmds.iter().map(|c| c.tier).collect();
        let mut sorted = tiers.clone();
        sorted.sort_unstable();
        assert_eq!(tiers, sorted);
        assert_eq!(cmds[0].label, "cargo check");
        assert!(cmds[0].structured, "cargo check must emit JSON diagnostics");
    }

    #[test]
    fn handles_only_rust_files() {
        assert!(RustAdapter.handles(Path::new("a/b.rs")));
        assert!(!RustAdapter.handles(Path::new("a/b.py")));
    }
}
