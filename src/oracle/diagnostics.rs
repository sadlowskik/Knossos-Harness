//! Parsing cargo's machine-readable diagnostics.
//!
//! `--message-format=json` emits one JSON object per line. Structured
//! diagnostics mean Oracle can tell an error from a warning, and can hand the
//! engine a precise `file:line: message` instead of a wall of terminal output.
//! Getting this for free is a concrete reason the Rust-first choice was cheap.

use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub level: String,
    pub message: String,
    pub file: Option<String>,
    pub line: Option<u64>,
    /// cargo's own pretty-printed form, which is what a human would see.
    pub rendered: Option<String>,
}

impl Diagnostic {
    pub fn is_error(&self) -> bool {
        self.level == "error" || self.level.starts_with("error")
    }

    pub fn location(&self) -> String {
        match (&self.file, self.line) {
            (Some(f), Some(l)) => format!("{f}:{l}"),
            (Some(f), None) => f.clone(),
            _ => "<no location>".to_string(),
        }
    }
}

#[derive(Deserialize)]
struct Line {
    #[serde(default)]
    reason: String,
    #[serde(default)]
    message: Option<Message>,
}

#[derive(Deserialize)]
struct Message {
    #[serde(default)]
    level: String,
    #[serde(default)]
    message: String,
    #[serde(default)]
    rendered: Option<String>,
    #[serde(default)]
    spans: Vec<Span>,
}

#[derive(Deserialize)]
struct Span {
    #[serde(default)]
    file_name: String,
    #[serde(default)]
    line_start: u64,
    #[serde(default)]
    is_primary: bool,
}

/// Extract diagnostics from a cargo JSON stream.
///
/// Non-JSON lines and non-diagnostic records are skipped rather than treated
/// as failures: cargo interleaves build progress with diagnostics, and a
/// parser that chokes on that would report false errors.
pub fn parse_cargo_json(stdout: &str) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    for raw in stdout.lines() {
        let raw = raw.trim();
        if !raw.starts_with('{') {
            continue;
        }
        let Ok(line) = serde_json::from_str::<Line>(raw) else {
            continue;
        };
        if line.reason != "compiler-message" {
            continue;
        }
        let Some(msg) = line.message else { continue };
        if msg.level.is_empty() {
            continue;
        }

        let primary = msg
            .spans
            .iter()
            .find(|s| s.is_primary)
            .or_else(|| msg.spans.first());

        out.push(Diagnostic {
            level: msg.level,
            message: msg.message,
            file: primary.map(|s| s.file_name.clone()),
            line: primary.map(|s| s.line_start),
            rendered: msg.rendered,
        });
    }
    out
}

/// Condense diagnostics into something worth putting in a prompt: errors
/// first, with cargo's rendered form, capped so one broken macro cannot
/// consume the context window.
pub fn summarize(diags: &[Diagnostic], max_bytes: usize) -> String {
    let errors: Vec<&Diagnostic> = diags.iter().filter(|d| d.is_error()).collect();
    let warnings = diags.len() - errors.len();

    if errors.is_empty() && warnings == 0 {
        return "no diagnostics".to_string();
    }

    let mut out = String::new();
    if !errors.is_empty() {
        out.push_str(&format!("{} error(s):\n", errors.len()));
        for d in &errors {
            let block = d
                .rendered
                .clone()
                .unwrap_or_else(|| format!("{}: {}\n", d.location(), d.message));
            if out.len() + block.len() > max_bytes {
                out.push_str("\n[remaining diagnostics truncated]\n");
                break;
            }
            out.push_str(&block);
        }
    }
    if warnings > 0 {
        out.push_str(&format!("\n{warnings} warning(s)\n"));
    }
    out
}

pub fn error_count(diags: &[Diagnostic]) -> usize {
    diags.iter().filter(|d| d.is_error()).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    const STREAM: &str = r#"{"reason":"compiler-artifact","target":{"name":"x"}}
{"reason":"compiler-message","message":{"level":"error","message":"cannot find value `y`","rendered":"error[E0425]: cannot find value `y`\n --> src/lib.rs:3:5\n","spans":[{"file_name":"src/lib.rs","line_start":3,"is_primary":true}]}}
{"reason":"compiler-message","message":{"level":"warning","message":"unused variable: `z`","rendered":"warning: unused variable\n","spans":[{"file_name":"src/lib.rs","line_start":9,"is_primary":true}]}}
   Compiling something v0.1.0
{"reason":"build-finished","success":false}"#;

    #[test]
    fn extracts_errors_and_warnings_and_ignores_the_rest() {
        let d = parse_cargo_json(STREAM);
        assert_eq!(d.len(), 2);
        assert_eq!(error_count(&d), 1);
        assert_eq!(d[0].level, "error");
        assert_eq!(d[0].location(), "src/lib.rs:3");
        assert_eq!(d[1].level, "warning");
    }

    #[test]
    fn non_json_progress_lines_do_not_break_parsing() {
        let d = parse_cargo_json("   Compiling foo v0.1.0\nnot json at all\n");
        assert!(d.is_empty());
    }

    #[test]
    fn summary_leads_with_errors_and_counts_warnings() {
        let s = summarize(&parse_cargo_json(STREAM), 10_000);
        assert!(s.starts_with("1 error(s)"));
        assert!(s.contains("E0425"));
        assert!(s.contains("1 warning(s)"));
    }

    #[test]
    fn summary_respects_the_byte_cap() {
        let d = parse_cargo_json(STREAM);
        let s = summarize(&d, 20);
        assert!(s.contains("truncated"));
        assert!(s.len() < 400);
    }

    #[test]
    fn empty_stream_reports_no_diagnostics() {
        assert_eq!(summarize(&[], 100), "no diagnostics");
    }
}
