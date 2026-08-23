//! Python language adapter: a line scanner for symbols, ruff/mypy/pytest for
//! verification.
//!
//! Symbols are scanned rather than parsed with `ast`. A file mid-edit is
//! constantly syntactically broken, and a parser that raises on that gives
//! you nothing exactly when the index is most useful. The scanner returns
//! every `def` / `class` it can still read.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use super::adapter::{LanguageAdapter, Symbol, SymbolKind, VerifyCommand, Visibility};

/// Markers whose presence at the workspace root identifies a Python project.
///
/// Detection is by marker, not by counting `.py` files: a single vendored
/// file under `node_modules` must not make a React app look like a Python
/// project. Same list as `oracle.py`.
const MARKERS: &[&str] = &[
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "tox.ini",
];

pub struct PythonAdapter;

/// Interpreter and launcher prefix the Oracle should invoke for `python -m …`.
///
/// The Python harness uses `sys.executable` so a virtualenv is honoured.
/// This process is not a Python interpreter, so we take `$PY`, then `$PYTHON`.
/// On Windows the Store `python.exe` alias can appear healthy in the parent
/// environment but fail after the sandbox scrubs it, so the standard `py -3`
/// launcher is preferred. Other platforms use a working `python`.
pub fn python_command() -> (String, Vec<String>) {
    for key in ["PY", "PYTHON"] {
        if let Ok(value) = std::env::var(key) {
            if !value.is_empty() {
                return (direct_interpreter(&value, &[]).unwrap_or(value), Vec::new());
            }
        }
    }
    #[cfg(windows)]
    if let Some(path) = installed_windows_python() {
        return (path, Vec::new());
    }
    #[cfg(windows)]
    if let Some(path) = direct_interpreter("py", &["-3"]) {
        return (path, Vec::new());
    }
    if let Some(path) = direct_interpreter("python", &[]) {
        return (path, Vec::new());
    }
    ("python".into(), Vec::new())
}

#[cfg(windows)]
fn installed_windows_python() -> Option<String> {
    let local = std::env::var_os("LOCALAPPDATA")?;
    let roots = [
        PathBuf::from(&local).join("Python"),
        PathBuf::from(local).join("Programs").join("Python"),
    ];
    let mut candidates = Vec::new();
    for root in roots {
        let Ok(entries) = std::fs::read_dir(root) else {
            continue;
        };
        for entry in entries.flatten() {
            let candidate = entry.path().join("python.exe");
            if candidate.is_file() {
                candidates.push(candidate);
            }
        }
    }
    // Python's install directory contains the version, so lexical descending
    // order selects the newest available interpreter deterministically.
    candidates.sort_by(|a, b| b.cmp(a));
    candidates.into_iter().find_map(|candidate| {
        let program = candidate.to_string_lossy();
        direct_interpreter(&program, &[])
    })
}

pub fn python_interpreter() -> String {
    python_command().0
}

fn direct_interpreter(program: &str, prefix: &[&str]) -> Option<String> {
    let output = Command::new(program)
        .args(prefix)
        .args(["-c", "import sys; print(sys.executable)"])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let reported = String::from_utf8(output.stdout).ok()?;
    let path = PathBuf::from(reported.trim());
    if !path.is_file() {
        return None;
    }
    Some(
        std::fs::canonicalize(&path)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned(),
    )
}

impl LanguageAdapter for PythonAdapter {
    fn name(&self) -> &str {
        "python"
    }

    fn extensions(&self) -> &[&str] {
        &["py"]
    }

    fn detect(&self, root: &Path) -> bool {
        MARKERS.iter().any(|marker| root.join(marker).exists())
    }

    fn symbols(&self, source: &str, path: &Path) -> Vec<Symbol> {
        scan_symbols(source, path)
    }

    fn parses_cleanly(&self, source: &str, _path: &Path) -> bool {
        // Fail open on anything we cannot decide. A false "broken" blocks
        // the whole ladder at tier 0; a false "clean" is caught by ruff.
        brackets_balance(source)
    }

    fn verify_commands(&self) -> Vec<VerifyCommand> {
        let (py, prefix) = python_command();
        let args = |tail: &[&str]| {
            prefix
                .iter()
                .cloned()
                .chain(tail.iter().map(|arg| (*arg).to_string()))
                .collect::<Vec<_>>()
        };
        vec![
            // `--output-format=concise` is load-bearing: the default block
            // form puts the location on a `-->` line, which per-diagnostic
            // baselining cannot parse.
            VerifyCommand::new(
                1,
                "ruff",
                &py,
                args(&["-m", "ruff", "check", "--output-format=concise"]),
            )
            .scopes([".py"]),
            VerifyCommand::new(2, "mypy", &py, args(&["-m", "mypy"]))
                .advisory()
                .scopes([".py"]),
            // pytest's standard layout imports the package via the cwd
            // entry that `-P` removes, so this tier must keep it.
            VerifyCommand::new(3, "pytest", &py, args(&["-m", "pytest", "-q"]))
                .unsafe_path()
                .required(),
        ]
    }
}

fn scan_symbols(source: &str, path: &Path) -> Vec<Symbol> {
    let mut out = Vec::new();
    for (idx, line) in source.lines().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            continue;
        }
        if let Some(name) = def_name(trimmed) {
            out.push(Symbol {
                kind: SymbolKind::Function,
                name: name.to_string(),
                signature: trimmed.trim_end_matches(':').trim().to_string(),
                file: path.to_path_buf(),
                line: idx + 1,
                end_line: idx + 1,
                visibility: if name.starts_with('_') {
                    Visibility::Private
                } else {
                    Visibility::Public
                },
            });
            continue;
        }
        if let Some(name) = class_name(trimmed) {
            out.push(Symbol {
                kind: SymbolKind::Class,
                name: name.to_string(),
                signature: trimmed.trim_end_matches(':').trim().to_string(),
                file: path.to_path_buf(),
                line: idx + 1,
                end_line: idx + 1,
                visibility: if name.starts_with('_') {
                    Visibility::Private
                } else {
                    Visibility::Public
                },
            });
        }
    }
    out
}

fn def_name(trimmed: &str) -> Option<&str> {
    let rest = trimmed
        .strip_prefix("async def ")
        .or_else(|| trimmed.strip_prefix("def "))?;
    ident_at_start(rest)
}

fn class_name(trimmed: &str) -> Option<&str> {
    ident_at_start(trimmed.strip_prefix("class ")?)
}

fn ident_at_start(s: &str) -> Option<&str> {
    let end = s
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .unwrap_or(s.len());
    let name = &s[..end];
    if name.is_empty() || name.as_bytes()[0].is_ascii_digit() {
        None
    } else {
        Some(name)
    }
}

/// Cheap syntax probe: unmatched brackets, ignoring strings and comments.
///
/// Not a parser. Triple-quoted strings, f-string expressions and implicit
/// concatenation are the usual ways this can be wrong; on those we return
/// `true` rather than invent a syntax error ruff would not agree with.
fn brackets_balance(source: &str) -> bool {
    let bytes = source.as_bytes();
    let mut i = 0;
    let mut stack: Vec<u8> = Vec::new();
    while i < bytes.len() {
        let b = bytes[i];
        match b {
            b'#' => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'\'' | b'"' => {
                let quote = b;
                let triple = i + 2 < bytes.len() && bytes[i + 1] == quote && bytes[i + 2] == quote;
                i += if triple { 3 } else { 1 };
                while i < bytes.len() {
                    if bytes[i] == b'\\' {
                        i += 2;
                        continue;
                    }
                    if triple
                        && i + 2 < bytes.len()
                        && bytes[i] == quote
                        && bytes[i + 1] == quote
                        && bytes[i + 2] == quote
                    {
                        i += 3;
                        break;
                    }
                    if !triple && bytes[i] == quote {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
                if i > bytes.len() {
                    return true; // unclosed string we could not decide
                }
            }
            b'(' | b'[' | b'{' => {
                stack.push(b);
                i += 1;
            }
            b')' | b']' | b'}' => {
                let expected = match b {
                    b')' => b'(',
                    b']' => b'[',
                    _ => b'{',
                };
                match stack.pop() {
                    Some(open) if open == expected => {}
                    _ => return false,
                }
                i += 1;
            }
            _ => i += 1,
        }
    }
    stack.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    #[test]
    fn detects_by_marker_not_by_source_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("app.py"), "def f():\n    pass\n").unwrap();
        assert!(!PythonAdapter.detect(dir.path()));
        std::fs::write(dir.path().join("pyproject.toml"), "[project]\nname='x'\n").unwrap();
        assert!(PythonAdapter.detect(dir.path()));
    }

    #[test]
    fn symbols_survive_a_syntax_error() {
        let src = "def good():\n    pass\n\ndef broken(\nclass StillHere:\n    pass\n";
        let found = PythonAdapter.symbols(src, &PathBuf::from("mod.py"));
        let names: Vec<_> = found.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"good"), "{names:?}");
        assert!(names.contains(&"StillHere"), "{names:?}");
    }

    #[test]
    fn parses_cleanly_distinguishes_valid_from_broken() {
        assert!(PythonAdapter.parses_cleanly("def a():\n    return 1\n", Path::new("a.py")));
        assert!(!PythonAdapter.parses_cleanly("def a(\n", Path::new("a.py")));
    }

    #[test]
    fn verify_chain_never_uses_bare_ruff() {
        let cmds = PythonAdapter.verify_commands();
        assert_eq!(cmds[0].label, "ruff");
        assert!(cmds[0].args.iter().any(|a| a == "-m"));
        assert_eq!(cmds[2].label, "pytest");
        assert!(!cmds[2].safe_path, "pytest needs the cwd on sys.path");
    }

    #[test]
    fn selected_interpreter_is_an_existing_executable_when_python_is_available() {
        let (program, prefix) = python_command();
        assert!(
            Path::new(&program).is_file(),
            "selected {program:?} with prefix {prefix:?}"
        );
    }
}
