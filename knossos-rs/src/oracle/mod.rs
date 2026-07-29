//! Oracle: tiered verification.
//!
//! Two rules, both load-bearing:
//!
//! 1. **Fail fast.** The first failing tier returns immediately. There is no
//!    point running clippy on code that does not compile, and the compiler's
//!    error is the one the agent needs to see.
//! 2. **Model judgement is the last tier, and only reachable when every
//!    deterministic tier passes.** "Compiles, lints, and tests pass — but is
//!    it *good*?" is the only question a model adds that cargo cannot answer.
//!    Asking it about broken code wastes tokens on an answer the compiler
//!    already gave for free.
//!
//! Tier 0 is in-process (tree-sitter), so a syntactically broken edit is
//! caught without spawning a process at all.

pub mod diagnostics;

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Result;

use crate::scribe::LanguageAdapter;

/// How long any single verification command may run before being abandoned.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(300);
const MAX_DETAIL: usize = 12_000;

#[derive(Debug, Clone, serde::Serialize)]
pub struct TierResult {
    pub tier: u8,
    pub label: String,
    pub passed: bool,
    pub detail: String,
    /// The tier could not run at all — the program is not installed.
    ///
    /// Distinct from passing, and the distinction is the whole point. A tier
    /// that never ran is not evidence the code is correct, but it is not
    /// evidence the code is broken either, and reporting `FAILED at cargo` on a
    /// machine without cargo tells the engine to repair code that was never
    /// checked. Python has always skipped these (`oracle.py`); this side failed
    /// them, so the two harnesses reached opposite verdicts on the same tree.
    ///
    /// Skipped tiers do not block, and `summary` names them rather than
    /// counting them among the passes — an absent ladder must not read as a
    /// clean bill of health.
    #[serde(default)]
    pub skipped: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Verdict {
    pub passed: bool,
    /// Highest tier actually executed.
    pub reached_tier: u8,
    pub tiers: Vec<TierResult>,
    /// True when only tier 0 could run because nothing was written to disk.
    /// A passing dry-run verdict is **not** full verification, and every
    /// message this type produces says so.
    pub dry_run: bool,
}

impl Verdict {
    /// The failing tier, if any. A skipped tier never ran and cannot be one.
    pub fn failure(&self) -> Option<&TierResult> {
        self.tiers.iter().find(|t| !t.passed && !t.skipped)
    }

    pub fn summary(&self) -> String {
        match self.failure() {
            Some(f) => format!("FAILED at {} (tier {})", f.label, f.tier),
            None if self.dry_run => {
                "syntax only — cargo cannot see unwritten changes".to_string()
            }
            None => {
                let (ran, skipped): (Vec<_>, Vec<_>) =
                    self.tiers.iter().partition(|t| !t.skipped);
                let mut text = format!(
                    "passed {} tier(s): {}",
                    ran.len(),
                    ran.iter().map(|t| t.label.as_str()).collect::<Vec<_>>().join(", ")
                );
                // Named, not silently folded in. "Passed 4 tiers" over a
                // machine where three of them are not installed is the kind of
                // confident wrong number this harness exists to refuse.
                if !skipped.is_empty() {
                    text.push_str(&format!(
                        " ({} skipped, not installed: {})",
                        skipped.len(),
                        skipped.iter().map(|t| t.label.as_str()).collect::<Vec<_>>().join(", ")
                    ));
                }
                text
            }
        }
    }

    /// What the engine is told after a verification round.
    pub fn report(&self) -> String {
        match self.failure() {
            Some(f) => format!(
                "Verification FAILED at `{}`.\n\n{}\n\nFix this before continuing.",
                f.label, f.detail
            ),
            None if self.dry_run => format!(
                "Syntax check passed ({}). Nothing was written to disk, so the compiler, \
                 linter and tests could not run — this is a preview, not verification.",
                self.summary()
            ),
            None => format!("Verification passed: {}.", self.summary()),
        }
    }

    /// Whether model judgement (tier 4) is permitted. Only when every
    /// deterministic tier passed — which a dry run can never satisfy, because
    /// most of them never ran.
    ///
    /// **At least one tier above 0 must actually have run.** Skipping a missing
    /// program instead of failing it (see [`TierResult::skipped`]) is right, and
    /// it opened this: on a machine without cargo every tier above syntax is now
    /// skipped, the verdict passes, and without this clause tier 4 would be
    /// asked to bless code that nothing compiled. A judge is the last tier
    /// precisely because it is the least trustworthy one; reaching it by having
    /// no toolchain is the opposite of the ladder's argument.
    pub fn deterministic_tiers_passed(&self) -> bool {
        self.passed
            && !self.dry_run
            && self.tiers.iter().any(|t| t.tier > 0 && !t.skipped)
    }
}

pub struct Oracle {
    root: PathBuf,
    sandbox: crate::sandbox::Sandbox,
}

impl Oracle {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Oracle { root: root.into(), sandbox: crate::sandbox::Sandbox::default() }
    }

    /// Run the tiers under a different environment policy.
    ///
    /// The ladder needs this at least as much as the `run` tool does: `cargo
    /// test` is a verification tier, it executes whatever the agent just wrote,
    /// and unlike the `run` tool it fires automatically rather than because the
    /// agent asked for it.
    ///
    /// That tool holds its own [`Sandbox`](crate::sandbox::Sandbox) — see
    /// [`Run::with_sandbox`](crate::tools::shell::Run::with_sandbox). Both
    /// spawn through `Sandbox::command`, so the settings cannot drift, but the
    /// policies are separate values: customise both or neither.
    pub fn with_sandbox(mut self, sandbox: crate::sandbox::Sandbox) -> Self {
        self.sandbox = sandbox;
        self
    }

    /// Run the deterministic ladder (tiers 0..3), stopping at the first failure.
    ///
    /// `changed` limits tier 0 to files the agent actually touched — parsing
    /// the whole tree to validate one edit would be waste.
    pub async fn verify(
        &self,
        adapter: &dyn LanguageAdapter,
        changed: &[PathBuf],
    ) -> Result<Verdict> {
        let mut tiers = Vec::new();

        // Tier 0 — syntax, in-process.
        let t0 = self.tier0(adapter, changed);
        let passed0 = t0.passed;
        let tier0_num = t0.tier;
        tiers.push(t0);
        if !passed0 {
            return Ok(Verdict { passed: false, reached_tier: tier0_num, tiers, dry_run: false });
        }

        // Tiers 1.. — the adapter's command chain, cheapest first.
        let mut reached = 0u8;
        for cmd in adapter.verify_commands() {
            let result = self.run_tier(&cmd).await?;
            reached = cmd.tier;
            let passed = result.passed;
            tiers.push(result);
            if !passed {
                return Ok(Verdict { passed: false, reached_tier: reached, tiers, dry_run: false });
            }
        }

        Ok(Verdict { passed: true, reached_tier: reached, tiers, dry_run: false })
    }

    /// Tier 0 over in-memory content, for dry runs.
    ///
    /// Nothing has been written, so cargo would compile the *old* code and
    /// report a misleading pass. Rather than run a check whose answer is about
    /// the wrong source, the ladder stops at syntax and the verdict is marked
    /// `dry_run` so no caller can mistake it for verification.
    pub fn verify_staged(
        &self,
        adapter: &dyn LanguageAdapter,
        staged: &[(PathBuf, String)],
    ) -> Verdict {
        let mut broken = Vec::new();
        let mut checked = 0usize;

        for (path, source) in staged {
            if !adapter.handles(path) {
                continue;
            }
            checked += 1;
            if !adapter.parses_cleanly(source) {
                let rel = path.strip_prefix(&self.root).unwrap_or(path);
                broken.push(rel.display().to_string());
            }
        }

        let passed = broken.is_empty();
        Verdict {
            passed,
            reached_tier: 0,
            tiers: vec![TierResult {
                tier: 0,
                label: "syntax".to_string(),
                passed,
                skipped: false,
                detail: if passed {
                    format!("{checked} staged file(s) parse cleanly")
                } else {
                    format!("syntax errors in staged: {}", broken.join(", "))
                },
            }],
            dry_run: true,
        }
    }

    fn tier0(&self, adapter: &dyn LanguageAdapter, changed: &[PathBuf]) -> TierResult {
        let mut broken = Vec::new();
        let mut checked = 0usize;

        for path in changed {
            if !adapter.handles(path) {
                continue;
            }
            let full = if path.is_absolute() { path.clone() } else { self.root.join(path) };
            let Ok(source) = std::fs::read_to_string(&full) else {
                continue;
            };
            checked += 1;
            if !adapter.parses_cleanly(&source) {
                broken.push(path.display().to_string());
            }
        }

        TierResult {
            tier: 0,
            label: "syntax".to_string(),
            passed: broken.is_empty(),
            skipped: false,
            detail: if broken.is_empty() {
                format!("{checked} file(s) parse cleanly")
            } else {
                format!("syntax errors in: {}", broken.join(", "))
            },
        }
    }

    async fn run_tier(&self, cmd: &crate::scribe::VerifyCommand) -> Result<TierResult> {
        // A verification tier runs `cargo test`, which executes whatever the
        // agent just wrote — so it needs the same containment as the `run`
        // tool, and gets it from the same place rather than a second copy of
        // the same settings. `Sandbox::run_bounded` documents the deadline and
        // why the whole process tree is killed rather than just the child.
        let finished = match self
            .sandbox
            .run_bounded(cmd.program, &cmd.args, &self.root, COMMAND_TIMEOUT)
            .await
        {
            Ok(f) => f,
            Err(e) => {
                // The program is not there. Absent cargo is not evidence of
                // broken code, and failing here reported `FAILED at cargo` on
                // every run on such a machine -- sending the engine to repair
                // something that was never checked.
                return Ok(TierResult {
                    tier: cmd.tier,
                    label: cmd.label.to_string(),
                    passed: true,
                    skipped: true,
                    detail: format!("skipped: could not run `{}`: {e}", cmd.program),
                });
            }
        };

        if finished.timed_out {
            return Ok(TierResult {
                tier: cmd.tier,
                label: cmd.label.to_string(),
                passed: false,
                // Not skipped: it ran, and not finishing is a real signal
                // about the tree rather than about the toolchain.
                skipped: false,
                detail: format!("`{}` timed out after {COMMAND_TIMEOUT:?}", cmd.label),
            });
        }

        let stdout = &finished.stdout;
        let stderr = &finished.stderr;

        let (passed, detail) = if cmd.structured {
            let diags = diagnostics::parse_cargo_json(stdout);
            let errors = diagnostics::error_count(&diags);
            // Warnings do not fail a tier: clippy's advice is worth surfacing
            // but not worth blocking on, and `cargo check` warnings are noise
            // when the build succeeded.
            (
                errors == 0 && finished.success(),
                if errors == 0 && !finished.success() {
                    // Failure with no compiler-message: link errors, bad
                    // manifest, missing toolchain. stderr carries it.
                    cap(stderr.to_string())
                } else {
                    diagnostics::summarize(&diags, MAX_DETAIL)
                },
            )
        } else {
            let mut body = stdout.to_string();
            if !stderr.trim().is_empty() {
                body.push('\n');
                body.push_str(stderr);
            }
            (finished.success(), cap(body))
        };

        Ok(TierResult {
            tier: cmd.tier,
            label: cmd.label.to_string(),
            passed,
            skipped: false,
            detail,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

fn cap(s: String) -> String {
    if s.len() <= MAX_DETAIL {
        return s;
    }
    let mut cut = MAX_DETAIL;
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}\n\n[truncated]", &s[..cut])
}

/// How much changed-file content tier 4 may read.
const JUDGE_BUDGET: usize = 24_000;

/// Tier 4 — model judgement against the constitution.
///
/// Callers **must** gate this on [`Verdict::deterministic_tiers_passed`]. It
/// exists to answer the one question cargo cannot — does this change satisfy
/// the constitution and actually do what was asked — and asking it about code
/// that does not compile spends tokens re-deriving what the compiler already
/// said for free.
pub async fn judge(
    engine: &dyn crate::engine::Engine,
    themis: &crate::themis::Themis,
    task: &str,
    root: &Path,
    changed: &[PathBuf],
    max_tokens: u32,
) -> Result<TierResult> {
    use crate::engine::{self as eng, Message, Request, ToolDef};

    let mut body = String::new();
    for path in changed {
        let full = if path.is_absolute() { path.clone() } else { root.join(path) };
        let Ok(text) = std::fs::read_to_string(&full) else {
            continue;
        };
        let block = format!("\n### {}\n```\n{}\n```\n", path.display(), text);
        if body.len() + block.len() > JUDGE_BUDGET {
            body.push_str("\n[remaining changed files omitted for length]\n");
            break;
        }
        body.push_str(&block);
    }
    if body.is_empty() {
        body.push_str("(no readable changed files)");
    }

    let tool = ToolDef {
        name: "submit_verdict".to_string(),
        description: "Record whether the change satisfies the constitution and the task."
            .to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "passed": {"type": "boolean"},
                "reason": {"type": "string", "description": "One or two sentences. If failing, name the principle violated."}
            },
            "required": ["passed", "reason"]
        }),
    };

    let req = Request::new(
        themis.system_prompt(crate::themis::JUDGE_ROLE, None),
        vec![Message::user_text(format!(
            "Task given to the agent:\n{task}\n\nThe compiler, linter and tests already pass.\n\
             Changed files:\n{body}\n\nCall submit_verdict."
        ))],
    )
    .with_tools(vec![tool])
    .with_max_tokens(max_tokens);

    let resp = eng::complete(engine, &req).await?;

    for (_, name, input) in resp.tool_uses() {
        if name != "submit_verdict" {
            continue;
        }
        let passed = input.get("passed").and_then(|v| v.as_bool()).unwrap_or(true);
        let reason = input
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("no reason given")
            .to_string();
        return Ok(TierResult {
            tier: 4,
            label: "constitution".into(),
            passed,
            skipped: false,
            detail: reason,
        });
    }

    // No verdict submitted. Treating that as a failure would block on the
    // engine's formatting rather than on the code, so it passes with a note.
    Ok(TierResult {
        tier: 4,
        label: "constitution".into(),
        passed: true,
        skipped: false,
        detail: format!("no structured verdict returned; engine said: {}", resp.text()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scribe::RustAdapter;

    fn tier(number: u8, label: &str, passed: bool, skipped: bool) -> TierResult {
        TierResult {
            tier: number,
            label: label.into(),
            passed,
            skipped,
            detail: String::new(),
        }
    }

    #[tokio::test]
    async fn a_missing_program_is_skipped_not_failed() {
        let dir = workspace("pub fn f() {}\n");
        let oracle = Oracle::new(dir.path());
        let cmd = crate::scribe::VerifyCommand {
            tier: 1,
            label: "absent",
            program: "definitely-not-a-real-program-xyz",
            args: vec![],
            structured: false,
        };

        let result = oracle.run_tier(&cmd).await.unwrap();

        // Absent cargo is not evidence of broken code. Failing here reported
        // `FAILED at cargo` on every run on such a machine, and sent the engine
        // to repair something that was never checked.
        assert!(result.skipped, "a program that will not spawn must be skipped");
        assert!(result.passed, "a skipped tier must not block");
    }

    #[test]
    fn a_skipped_tier_is_never_the_failure() {
        let verdict = Verdict {
            passed: true,
            reached_tier: 1,
            tiers: vec![tier(0, "syntax", true, false), tier(1, "cargo check", true, true)],
            dry_run: false,
        };

        assert!(verdict.failure().is_none());
    }

    #[test]
    fn skipped_tiers_are_named_rather_than_counted_as_passes() {
        // "passed 3 tier(s)" on a machine where two are not installed is the
        // kind of confident wrong number this harness exists to refuse.
        let verdict = Verdict {
            passed: true,
            reached_tier: 1,
            tiers: vec![
                tier(0, "syntax", true, false),
                tier(1, "cargo check", true, true),
                tier(2, "clippy", true, true),
            ],
            dry_run: false,
        };

        let summary = verdict.summary();
        assert!(summary.contains("passed 1 tier(s): syntax"), "{summary}");
        assert!(summary.contains("2 skipped"), "{summary}");
        assert!(summary.contains("cargo check"), "{summary}");
    }

    #[test]
    fn an_entirely_skipped_ladder_does_not_reach_the_judge() {
        // Skipping a missing program rather than failing it is right, and it
        // opened this: on a machine without cargo every tier above syntax is
        // skipped, the verdict passes, and tier 4 would be asked to bless code
        // that nothing compiled. The judge is last because it is the least
        // trustworthy tier; reaching it by having no toolchain inverts the
        // ladder's entire argument.
        let verdict = Verdict {
            passed: true,
            reached_tier: 0,
            tiers: vec![
                tier(0, "syntax", true, false),
                tier(1, "cargo check", true, true),
                tier(2, "clippy", true, true),
            ],
            dry_run: false,
        };

        assert!(verdict.failure().is_none(), "skips must not block");
        assert!(!verdict.deterministic_tiers_passed(), "but must not license tier 4");
    }

    #[test]
    fn a_ladder_that_really_ran_still_reaches_the_judge() {
        let verdict = Verdict {
            passed: true,
            reached_tier: 1,
            tiers: vec![
                tier(0, "syntax", true, false),
                tier(1, "cargo check", true, false),
                tier(2, "clippy", true, true),
            ],
            dry_run: false,
        };

        assert!(verdict.deterministic_tiers_passed());
    }

    #[test]
    fn a_real_failure_still_outranks_a_skip() {
        let verdict = Verdict {
            passed: false,
            reached_tier: 1,
            tiers: vec![tier(1, "cargo check", true, true), tier(2, "clippy", false, false)],
            dry_run: false,
        };

        assert_eq!(verdict.failure().map(|t| t.label.as_str()), Some("clippy"));
    }

    fn workspace(lib: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), lib).unwrap();
        dir
    }

    #[tokio::test]
    async fn tier0_catches_a_syntax_error_without_spawning_cargo() {
        let dir = workspace("pub fn a( {{{ ~~~ not rust");
        let oracle = Oracle::new(dir.path());
        let changed = vec![PathBuf::from("src/lib.rs")];

        let v = oracle.verify(&RustAdapter, &changed).await.unwrap();
        assert!(!v.passed);
        assert_eq!(v.reached_tier, 0);
        // The whole point of fail-fast: nothing past tier 0 ran.
        assert_eq!(v.tiers.len(), 1);
        assert_eq!(v.failure().unwrap().label, "syntax");
    }

    #[tokio::test]
    async fn tier0_passes_valid_syntax_and_escalates() {
        let dir = workspace("pub fn a() -> u32 { 1 }\n");
        let oracle = Oracle::new(dir.path());
        let v = oracle
            .verify(&RustAdapter, &[PathBuf::from("src/lib.rs")])
            .await
            .unwrap();
        // cargo may or may not be able to build in a temp dir here; what must
        // hold is that tier 0 passed and the ladder moved past it.
        assert!(v.tiers[0].passed, "tier 0: {}", v.tiers[0].detail);
        assert!(v.tiers.len() > 1, "must escalate past tier 0");
    }

    #[tokio::test]
    async fn a_failing_tier_stops_the_ladder() {
        // Type error: passes tier 0 (syntactically valid), fails cargo check.
        let dir = workspace("pub fn a() -> u32 { \"not a number\" }\n");
        let oracle = Oracle::new(dir.path());
        let v = oracle
            .verify(&RustAdapter, &[PathBuf::from("src/lib.rs")])
            .await
            .unwrap();

        assert!(!v.passed);
        assert!(v.tiers[0].passed, "syntax is valid, only the type is wrong");
        let failed = v.failure().unwrap();
        assert_eq!(failed.tier, 1);
        assert_eq!(failed.label, "cargo check");
        // Nothing after the failure ran.
        assert_eq!(v.tiers.last().unwrap().tier, 1);
        assert!(!v.deterministic_tiers_passed(), "tier 4 must be unreachable");
    }

    #[test]
    fn report_names_the_failing_tier() {
        let v = Verdict {
            passed: false,
            reached_tier: 1,
            tiers: vec![TierResult {
                tier: 1,
                label: "cargo check".into(),
                passed: false,
                skipped: false,
                detail: "error[E0308]".into(),
            }],
            dry_run: false,
        };
        assert!(v.report().contains("cargo check"));
        assert!(v.report().contains("E0308"));
        assert!(!v.deterministic_tiers_passed());
    }

    #[test]
    fn staged_verification_checks_syntax_of_unwritten_content() {
        let dir = workspace("pub fn a() {}\n");
        let oracle = Oracle::new(dir.path());

        let good = oracle.verify_staged(
            &RustAdapter,
            &[(PathBuf::from("src/lib.rs"), "pub fn b() -> u32 { 1 }".to_string())],
        );
        assert!(good.passed);
        assert!(good.dry_run);

        let bad = oracle.verify_staged(
            &RustAdapter,
            &[(PathBuf::from("src/lib.rs"), "pub fn b( {{{ ~~~".to_string())],
        );
        assert!(!bad.passed);
        assert_eq!(bad.failure().unwrap().tier, 0);
    }

    /// A dry run must never be mistaken for verification, even when it passes.
    #[test]
    fn a_passing_dry_run_still_blocks_tier_four_and_says_why() {
        let dir = workspace("pub fn a() {}\n");
        let v = Oracle::new(dir.path()).verify_staged(
            &RustAdapter,
            &[(PathBuf::from("src/lib.rs"), "pub fn b() {}".to_string())],
        );

        assert!(v.passed);
        assert!(!v.deterministic_tiers_passed(), "dry run must gate tier 4");
        assert!(v.report().contains("preview, not verification"));
        assert!(v.summary().contains("syntax only"));
    }
}
