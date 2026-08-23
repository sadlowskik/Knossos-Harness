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
//!
//! The allowlist bounds *which* program runs, which is as far as it can go:
//! `cargo test` is on it, and `cargo test` executes whatever the agent wrote.
//! [`Sandbox`] covers the other half — what that code can see once it is
//! running — and is the reason the harness's own API key is not readable from
//! inside a test the agent authored.

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use super::{req_str, Tool, ToolCtx, ToolOutput};
use crate::sandbox::Sandbox;

const MAX_OUTPUT: usize = 30_000;

/// How long a single command may run. Matches `oracle::COMMAND_TIMEOUT`: both
/// bound workspace-controlled work, and a build that is too slow for one is too
/// slow for the other.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(300);

/// Programs the agent may invoke.
const ALLOWED: &[&str] = &[
    "cargo", "rustc", "rustfmt", "git", "python", "python3", "pytest",
];

/// `git` subcommands that cannot modify the repository or reach the network.
const GIT_READONLY: &[&str] = &[
    "status",
    "diff",
    "log",
    "show",
    "ls-files",
    "blame",
    "rev-parse",
    "branch",
];

/// `cargo` subcommands that publish or install outside the workspace.
const CARGO_DENIED: &[&str] = &["publish", "install", "login", "owner", "yank"];

pub struct Run {
    allowed: Vec<String>,
    sandbox: Sandbox,
}

impl Default for Run {
    fn default() -> Self {
        Run {
            allowed: ALLOWED.iter().map(|s| s.to_string()).collect(),
            sandbox: Sandbox::default(),
        }
    }
}

impl Run {
    /// Run commands under a different environment policy.
    ///
    /// The Oracle spawns cargo too and holds its own [`Sandbox`]; see
    /// [`Oracle::with_sandbox`](crate::oracle::Oracle::with_sandbox). The two
    /// share a spawn path so their *mechanism* cannot diverge, but the policy
    /// values are independent — setting one and not the other leaves the
    /// verification ladder on the default.
    pub fn with_sandbox(mut self, sandbox: Sandbox) -> Self {
        self.sandbox = sandbox;
        self
    }

    pub fn sandbox(&self) -> &Sandbox {
        &self.sandbox
    }
}

#[async_trait]
impl Tool for Run {
    fn name(&self) -> &str {
        "run"
    }

    fn description(&self) -> &str {
        "Run a build or test command in the workspace. Allowed programs: cargo, rustc, rustfmt, git (read-only subcommands), python, python3, and pytest. No shell is used, so operators like && and | do not work."
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

    /// The allowlist keeps commands to builds and inspections, but a build
    /// writes `target/` and a command is the only way to answer "run the tests
    /// and tell me what breaks" — so this is work, not reconnaissance.
    fn consequential(&self) -> bool {
        true
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

        if let Some((code, body)) = ctx.run_in_editor(&argv, COMMAND_TIMEOUT) {
            let body = cap(body);
            return Ok(if code == 0 {
                ToolOutput::ok(format!("exit {code}\n\n{body}"))
            } else {
                ToolOutput {
                    content: format!("exit {code}\n\n{body}"),
                    is_error: true,
                    changed: Vec::new(),
                }
            });
        }

        // Environment scrubbing, no stdin, the deadline and the tree kill all
        // come from here. `cargo run` and `cargo test` are both on the
        // allowlist, so this executes workspace code — which is entitled to
        // loop forever. Every other subprocess in either harness is bounded;
        // this one was not, and `serve` awaits dispatch inline on the
        // stdin-reading thread, so a single hang wedged the server permanently
        // against every later command, `shutdown` included.
        let (exec_program, exec_args) = resolve_python_command(program, args);
        let finished = match self
            .sandbox
            .run_bounded(&exec_program, &exec_args, ctx.root(), COMMAND_TIMEOUT)
            .await
        {
            Ok(f) => f,
            Err(e) => return Ok(ToolOutput::error(format!("failed to run `{program}`: {e}"))),
        };

        let mut body = String::new();
        if !finished.stdout.trim().is_empty() {
            body.push_str(&finished.stdout);
        }
        if !finished.stderr.trim().is_empty() {
            if !body.is_empty() {
                body.push('\n');
            }
            body.push_str(&finished.stderr);
        }
        if body.trim().is_empty() {
            body.push_str("(no output)");
        }

        if finished.timed_out {
            // Whatever it managed to print before the deadline is kept: a build
            // that hung after emitting three errors has told the agent
            // something, and discarding it would send it back to guess.
            return Ok(ToolOutput::error(format!(
                "`{program}` timed out after {COMMAND_TIMEOUT:?}; it and everything \
                 it started were killed\n\n{}",
                cap(body)
            )));
        }

        let body = format!("exit {}\n\n{}", finished.code(), cap(body));

        Ok(if finished.success() {
            ToolOutput::ok(body)
        } else {
            // A failing command is information, not a harness error: the engine
            // should see the compiler output and react to it.
            ToolOutput {
                content: body,
                is_error: true,
                changed: Vec::new(),
            }
        })
    }
}

fn resolve_python_command(program: &str, args: &[String]) -> (String, Vec<String>) {
    let basename = program.rsplit(['/', '\\']).next().unwrap_or(program);
    let base = basename
        .strip_suffix(".exe")
        .unwrap_or(basename)
        .to_ascii_lowercase();
    if matches!(base.as_str(), "python" | "python3") {
        let (resolved, mut prefix) = crate::scribe::python::python_command();
        prefix.extend_from_slice(args);
        return (resolved, prefix);
    }
    if base == "pytest" {
        let (resolved, mut prefix) = crate::scribe::python::python_command();
        prefix.extend(["-m".to_string(), "pytest".to_string()]);
        prefix.extend_from_slice(args);
        return (resolved, prefix);
    }
    (program.to_string(), args.to_vec())
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

        let sub = args
            .iter()
            .find(|a| !a.starts_with('-'))
            .map(String::as_str);
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
        assert_eq!(
            tokenize(r#"cargo test "my test name""#),
            vec!["cargo", "test", "my test name"]
        );
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
        assert!(r
            .check("cargo", &["--offline".into(), "check".into()])
            .is_ok());
    }

    #[test]
    fn python_aliases_use_the_verified_interpreter() {
        let args = vec!["-m".into(), "pytest".into(), "-q".into()];
        let (program, resolved) = resolve_python_command(r"C:\\tools\\python.exe", &args);
        let (expected_program, expected_prefix) = crate::scribe::python::python_command();

        assert_eq!(program, expected_program);
        assert_eq!(
            &resolved[..expected_prefix.len()],
            expected_prefix.as_slice()
        );
        assert_eq!(&resolved[expected_prefix.len()..], args.as_slice());
    }

    #[test]
    fn pytest_alias_is_rewritten_as_a_python_module() {
        let args = vec!["-q".into()];
        let (program, resolved) = resolve_python_command("pytest", &args);
        let (expected_program, expected_prefix) = crate::scribe::python::python_command();

        assert_eq!(program, expected_program);
        assert_eq!(
            &resolved[..expected_prefix.len()],
            expected_prefix.as_slice()
        );
        assert_eq!(&resolved[expected_prefix.len()..], ["-m", "pytest", "-q"]);
    }

    #[test]
    fn non_python_commands_are_unchanged() {
        let args = vec!["test".into(), "--offline".into()];
        assert_eq!(
            resolve_python_command("cargo", &args),
            ("cargo".into(), args)
        );
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

    #[tokio::test]
    async fn python_alias_runs_inside_the_scrubbed_sandbox() {
        let (_d, c) = ctx();
        let out = Run::default()
            .run(&json!({"command": "python --version"}), &c)
            .await
            .unwrap();

        assert!(!out.is_error, "{}", out.content);
        assert!(out.content.to_ascii_lowercase().contains("python"));
    }
}
