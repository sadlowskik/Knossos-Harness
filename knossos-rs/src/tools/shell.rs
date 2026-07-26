//! Command execution, without a shell.
//!
//! The command string is tokenized here and passed to the OS as an explicit
//! program plus argument vector. No shell interpreter is involved, so `&&`,
//! `;`, `|`, `>` and backticks are inert — they arrive at the program as
//! literal arguments rather than being executed. That removes the entire class
//! of "the model appended `&& rm -rf`" failures structurally instead of by
//! pattern-matching for dangerous strings.
//!
//! On top of that, the program name must be on an allowlist, and `git` is
//! restricted to read-only subcommands.

use std::process::Stdio;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use super::{req_str, Tool, ToolCtx, ToolOutput};

const MAX_OUTPUT: usize = 30_000;

/// Programs the agent may invoke.
const ALLOWED: &[&str] = &["cargo", "rustc", "rustfmt", "git"];

/// `git` subcommands that cannot modify the repository or reach the network.
const GIT_READONLY: &[&str] = &[
    "status", "diff", "log", "show", "ls-files", "blame", "rev-parse", "branch",
];

/// `cargo` subcommands that publish or install outside the workspace.
const CARGO_DENIED: &[&str] = &["publish", "install", "login", "owner", "yank"];

pub struct Run {
    allowed: Vec<String>,
}

impl Default for Run {
    fn default() -> Self {
        Run { allowed: ALLOWED.iter().map(|s| s.to_string()).collect() }
    }
}

#[async_trait]
impl Tool for Run {
    fn name(&self) -> &str {
        "run"
    }

    fn description(&self) -> &str {
        "Run a build or inspection command in the workspace. Allowed programs: cargo, rustc, rustfmt, git (read-only subcommands). No shell is used, so operators like && and | do not work."
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "A single command, e.g. `cargo check` or `cargo test --lib`"
                }
            },
            "required": ["command"]
        })
    }

    async fn run(&self, input: &serde_json::Value, ctx: &ToolCtx) -> Result<ToolOutput> {
        let raw = req_str(input, "command")?;
        let argv = tokenize(raw);
        let Some((program, args)) = argv.split_first() else {
            return Ok(ToolOutput::error("empty command"));
        };

        if let Err(reason) = self.check(program, args) {
            return Ok(ToolOutput::error(reason));
        }

        let output = tokio::process::Command::new(program)
            .args(args)
            .current_dir(ctx.root())
            .stdin(Stdio::null())
            .output()
            .await;

        let output = match output {
            Ok(o) => o,
            Err(e) => return Ok(ToolOutput::error(format!("failed to run `{program}`: {e}"))),
        };

        let mut body = String::new();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stdout.trim().is_empty() {
            body.push_str(&stdout);
        }
        if !stderr.trim().is_empty() {
            if !body.is_empty() {
                body.push('\n');
            }
            body.push_str(&stderr);
        }
        if body.trim().is_empty() {
            body.push_str("(no output)");
        }

        let code = output.status.code().unwrap_or(-1);
        let body = format!("exit {code}\n\n{}", cap(body));

        Ok(if output.status.success() {
            ToolOutput::ok(body)
        } else {
            // A failing command is information, not a harness error: the engine
            // should see the compiler output and react to it.
            ToolOutput { content: body, is_error: true, changed: Vec::new() }
        })
    }
}

impl Run {
    fn check(&self, program: &str, args: &[String]) -> Result<(), String> {
        let base = program.rsplit(['/', '\\']).next().unwrap_or(program);
        let base = base.strip_suffix(".exe").unwrap_or(base);

        if !self.allowed.iter().any(|a| a == base) {
            return Err(format!(
                "`{base}` is not on the allowlist ({})",
                self.allowed.join(", ")
            ));
        }

        let sub = args.iter().find(|a| !a.starts_with('-')).map(String::as_str);
        match (base, sub) {
            ("git", Some(s)) if !GIT_READONLY.contains(&s) => Err(format!(
                "git {s} is not allowed; read-only subcommands only ({})",
                GIT_READONLY.join(", ")
            )),
            ("git", None) => Err("git needs a subcommand".to_string()),
            ("cargo", Some(s)) if CARGO_DENIED.contains(&s) => {
                Err(format!("cargo {s} is not allowed"))
            }
            _ => Ok(()),
        }
    }
}

fn cap(s: String) -> String {
    if s.len() <= MAX_OUTPUT {
        return s;
    }
    // Compiler errors cluster at the start; keep the head.
    let mut cut = MAX_OUTPUT;
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}\n\n[output truncated at {MAX_OUTPUT} bytes]", &s[..cut])
}

/// Split a command line into argv, honouring double quotes. Deliberately not a
/// shell: no expansion, no substitution, no operators.
fn tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;

    for ch in s.chars() {
        match ch {
            '"' => quoted = !quoted,
            c if c.is_whitespace() && !quoted => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> (tempfile::TempDir, ToolCtx) {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        (dir, ToolCtx::new(root))
    }

    #[test]
    fn tokenizer_keeps_quoted_arguments_together() {
        assert_eq!(tokenize(r#"cargo test "my test name""#), vec!["cargo", "test", "my test name"]);
    }

    #[test]
    fn shell_operators_are_not_interpreted() {
        // They become literal argv entries; they are never executed.
        let argv = tokenize("cargo check && rm -rf /");
        assert_eq!(argv[0], "cargo");
        assert!(argv.contains(&"&&".to_string()));
    }

    #[test]
    fn allowlist_rejects_arbitrary_programs() {
        let r = Run::default();
        assert!(r.check("rm", &["-rf".into()]).is_err());
        assert!(r.check("powershell", &[]).is_err());
        assert!(r.check("cargo", &["check".into()]).is_ok());
    }

    #[test]
    fn git_is_limited_to_readonly_subcommands() {
        let r = Run::default();
        assert!(r.check("git", &["status".into()]).is_ok());
        assert!(r.check("git", &["push".into()]).is_err());
        assert!(r.check("git", &["reset".into(), "--hard".into()]).is_err());
    }

    #[test]
    fn cargo_publish_is_denied() {
        let r = Run::default();
        assert!(r.check("cargo", &["publish".into()]).is_err());
        assert!(r.check("cargo", &["--offline".into(), "check".into()]).is_ok());
    }

    #[tokio::test]
    async fn disallowed_command_returns_an_error_output() {
        let (_d, c) = ctx();
        let out = Run::default()
            .run(&json!({"command": "rm -rf ."}), &c)
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("not on the allowlist"));
    }
}
