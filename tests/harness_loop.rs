//! End-to-end tests for the agent loop.
//!
//! Every one of these runs against `MockEngine`, so the suite needs no API
//! key, no network and no local model server — but the tools, the path jail,
//! Scribe, Oracle and Ariadne are all real. `cargo check` genuinely runs.

use std::path::{Path, PathBuf};

use daedalus_harness::ariadne::{Ariadne, Halt};
use daedalus_harness::engine::mock::{text_response, tool_call, MockEngine};
use daedalus_harness::metis::Plan;
use daedalus_harness::oracle::Oracle;
use daedalus_harness::scribe::SymbolIndex;
use daedalus_harness::session::Session;
use daedalus_harness::talos::{Outcome, Talos};
use daedalus_harness::themis::Themis;
use daedalus_harness::tools::{ToolCtx, ToolRegistry};

/// Copy a fixture crate into a temp dir so tests can edit it freely.
fn fixture(name: &str) -> tempfile::TempDir {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
    let dir = tempfile::tempdir().unwrap();
    copy_tree(&src, dir.path()).unwrap();
    dir
}

fn copy_tree(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let dest = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            // `target/` would make the copy enormous and slow.
            if entry.file_name() == "target" {
                continue;
            }
            copy_tree(&entry.path(), &dest)?;
        } else {
            std::fs::copy(entry.path(), dest)?;
        }
    }
    Ok(())
}

struct Harness {
    _dir: tempfile::TempDir,
    root: PathBuf,
    trace: PathBuf,
}

impl Harness {
    fn new(name: &str) -> Self {
        let dir = fixture(name);
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let trace = root.join("trace.jsonl");
        Harness { _dir: dir, root, trace }
    }

    fn talos(&self, scripted: Vec<daedalus_harness::engine::Response>, max_steps: usize, dry: bool) -> Talos {
        let mut ctx = ToolCtx::new(&self.root);
        if dry {
            ctx = ctx.dry_run();
        }
        Talos::new(
            Box::new(MockEngine::new(scripted)),
            ToolRegistry::standard(),
            ctx,
            Oracle::new(&self.root),
            SymbolIndex::build(&self.root).unwrap(),
            Themis::from_text("Be correct."),
            Ariadne::new(max_steps, max_steps.saturating_sub(1).max(1)),
            Session::new(&self.root, "mock").with_trace(&self.trace).unwrap(),
            1024,
            // Tier 4 needs an engine turn of its own; the tests that exercise
            // it script that turn explicitly.
            false,
        )
    }

    async fn run(&self, scripted: Vec<daedalus_harness::engine::Response>, max_steps: usize) -> Outcome {
        let mut talos = self.talos(scripted, max_steps, false);
        let plan = Plan { steps: vec!["do the thing".into()] };
        talos.run("test task", &plan).await.unwrap()
    }

    fn trace_events(&self) -> Vec<serde_json::Value> {
        let body = std::fs::read_to_string(&self.trace).unwrap_or_default();
        body.lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect()
    }
}

#[tokio::test]
async fn a_verified_change_halts_done() {
    let h = Harness::new("passing");

    let outcome = h
        .run(
            vec![
                tool_call(
                    "1",
                    "write_file",
                    serde_json::json!({
                        "path": "src/lib.rs",
                        "content": "pub fn double(x: u32) -> u32 { x * 2 }\npub fn triple(x: u32) -> u32 { x * 3 }\n"
                    }),
                ),
                text_response("Added triple. Done."),
            ],
            8,
        )
        .await;

    assert_eq!(outcome.halt, Halt::Done, "{}", outcome.summary);
    assert!(outcome.succeeded());
    assert_eq!(outcome.changed.len(), 1);
    assert_eq!(outcome.steps_used, 2);

    let verdict = outcome.verdict.expect("a Done outcome must carry a verdict");
    assert!(verdict.passed);
    // The full deterministic ladder ran: syntax, check, clippy, test.
    assert!(verdict.reached_tier >= 3, "reached tier {}", verdict.reached_tier);
}

#[tokio::test]
async fn a_broken_edit_fails_verification_and_the_loop_keeps_going() {
    let h = Harness::new("passing");

    let outcome = h
        .run(
            vec![
                // Type error: parses fine, fails cargo check.
                tool_call(
                    "1",
                    "write_file",
                    serde_json::json!({
                        "path": "src/lib.rs",
                        "content": "pub fn double(x: u32) -> u32 { \"not a number\" }\n"
                    }),
                ),
                text_response("Done."),
                text_response("Still done, I insist."),
            ],
            3,
        )
        .await;

    assert_ne!(outcome.halt, Halt::Done);
    let verdict = outcome.verdict.expect("verification ran");
    assert!(!verdict.passed);

    let failed = verdict.failure().unwrap();
    assert_eq!(failed.tier, 1, "syntax is valid; cargo check is what must fail");
    assert_eq!(failed.label, "cargo check");
    // Fail-fast: clippy and test never ran.
    assert!(!verdict.tiers.iter().any(|t| t.label == "cargo test"));
    // And tier 4 was never reachable.
    assert!(!verdict.deterministic_tiers_passed());
}

#[tokio::test]
async fn tier_zero_catches_a_syntax_error_before_cargo_runs() {
    let h = Harness::new("passing");

    let outcome = h
        .run(
            vec![
                tool_call(
                    "1",
                    "write_file",
                    serde_json::json!({"path": "src/lib.rs", "content": "pub fn a( {{{ ~~~"}),
                ),
                text_response("Done."),
                text_response("Done again."),
            ],
            3,
        )
        .await;

    let verdict = outcome.verdict.expect("verification ran");
    let failed = verdict.failure().unwrap();
    assert_eq!(failed.tier, 0);
    assert_eq!(failed.label, "syntax");
    assert_eq!(verdict.tiers.len(), 1, "nothing past tier 0 should have run");
}

#[tokio::test]
async fn the_step_ceiling_forces_a_halt() {
    let h = Harness::new("passing");

    // An engine that never stops calling tools. Without Ariadne's ceiling
    // this loop would not terminate.
    let forever: Vec<_> = (0..5)
        .map(|i| {
            tool_call(
                &i.to_string(),
                "read_file",
                serde_json::json!({"path": "src/lib.rs"}),
            )
        })
        .collect();

    let outcome = h.run(forever, 3).await;

    assert_eq!(outcome.halt, Halt::BudgetExhausted);
    assert_eq!(outcome.steps_used, 3);
    assert!(outcome.summary.contains("budget"), "{}", outcome.summary);
}

#[tokio::test]
async fn repeated_empty_steps_are_reported_as_stuck() {
    let h = Harness::new("broken");

    // The engine keeps saying it is done without changing anything, and
    // verification keeps failing. That is the `Stuck` signature.
    let outcome = h
        .run(
            vec![
                text_response("Done."),
                text_response("Done."),
                text_response("Done."),
            ],
            8,
        )
        .await;

    assert_eq!(outcome.halt, Halt::Stuck, "{}", outcome.summary);
    assert!(outcome.changed.is_empty());
}

#[tokio::test]
async fn the_path_jail_survives_a_hostile_tool_call() {
    let h = Harness::new("passing");

    let outcome = h
        .run(
            vec![
                tool_call(
                    "1",
                    "write_file",
                    serde_json::json!({"path": "../../escaped.rs", "content": "pwned"}),
                ),
                text_response("Done."),
            ],
            4,
        )
        .await;

    // The write was refused, so nothing changed and the file does not exist.
    assert!(outcome.changed.is_empty(), "jail must refuse the write");
    assert!(!h.root.parent().unwrap().join("escaped.rs").exists());

    // The refusal reached the engine as a tool error rather than killing the run.
    let errored = h
        .trace_events()
        .iter()
        .any(|e| e["event"] == "tool_call" && e["is_error"] == true);
    assert!(errored, "the refusal should be recorded as a tool error");
}

#[tokio::test]
async fn disallowed_shell_commands_are_refused() {
    let h = Harness::new("passing");

    let outcome = h
        .run(
            vec![
                tool_call("1", "run", serde_json::json!({"command": "rm -rf ."})),
                text_response("Done."),
            ],
            4,
        )
        .await;

    let refused = h.trace_events().iter().any(|e| {
        e["event"] == "tool_call"
            && e["is_error"] == true
            && e["output"].as_str().unwrap_or("").contains("not on the allowlist")
    });
    assert!(refused, "the allowlist must reject `rm`");
    assert!(outcome.changed.is_empty());
}

#[tokio::test]
async fn the_trace_records_the_whole_trajectory() {
    let h = Harness::new("passing");

    h.run(
        vec![
            tool_call(
                "1",
                "write_file",
                serde_json::json!({
                    "path": "src/lib.rs",
                    "content": "pub fn double(x: u32) -> u32 { x * 2 }\n"
                }),
            ),
            text_response("Done."),
        ],
        6,
    )
    .await;

    let events = h.trace_events();
    let kinds: Vec<&str> = events.iter().filter_map(|e| e["event"].as_str()).collect();

    for expected in ["plan_produced", "step_started", "tool_call", "oracle_verdict", "halt", "task_finished"] {
        assert!(kinds.contains(&expected), "trace missing `{expected}`; got {kinds:?}");
    }

    // Every event carries a timestamp, and tool calls carry their input and
    // which files they touched — this is the SFT/RL record, not just a log.
    assert!(events.iter().all(|e| e["at"].is_string()));
    let tool = events.iter().find(|e| e["event"] == "tool_call").unwrap();
    assert_eq!(tool["tool"], "write_file");
    assert!(tool["input"]["path"].is_string());
    assert_eq!(tool["changed"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn a_dry_run_proposes_changes_without_writing_them() {
    let h = Harness::new("passing");
    let before = std::fs::read_to_string(h.root.join("src/lib.rs")).unwrap();

    let mut talos = h.talos(
        vec![
            tool_call(
                "1",
                "write_file",
                serde_json::json!({
                    "path": "src/lib.rs",
                    "content": "pub fn double(x: u32) -> u32 { x * 2 }\npub fn triple(x: u32) -> u32 { x * 3 }\n"
                }),
            ),
            text_response("Done."),
        ],
        6,
        true,
    );

    let outcome = talos
        .run("test task", &Plan { steps: vec!["s".into()] })
        .await
        .unwrap();

    assert!(outcome.dry_run);
    assert_eq!(
        std::fs::read_to_string(h.root.join("src/lib.rs")).unwrap(),
        before,
        "dry run must leave the file untouched"
    );

    let diffs = talos.diffs();
    assert_eq!(diffs.len(), 1);
    assert!(diffs[0].unified.contains("+pub fn triple"));

    // A dry run can only check syntax, and must not be mistaken for more.
    let verdict = outcome.verdict.unwrap();
    assert!(verdict.dry_run);
    assert!(!verdict.deterministic_tiers_passed());
}

#[tokio::test]
async fn applying_a_dry_run_writes_the_staged_content() {
    let h = Harness::new("passing");
    let mut talos = h.talos(
        vec![
            tool_call(
                "1",
                "write_file",
                serde_json::json!({"path": "src/added.rs", "content": "pub fn added() {}\n"}),
            ),
            text_response("Done."),
        ],
        6,
        true,
    );

    talos.run("t", &Plan { steps: vec!["s".into()] }).await.unwrap();
    assert!(!h.root.join("src/added.rs").exists());

    let written = talos.apply().unwrap();
    assert_eq!(written.len(), 1);
    assert_eq!(
        std::fs::read_to_string(h.root.join("src/added.rs")).unwrap(),
        "pub fn added() {}\n"
    );
    // And the exact index picked it up.
    assert_eq!(talos.scribe.lookup("added").len(), 1);
}

#[tokio::test]
async fn resume_continues_the_same_conversation() {
    let h = Harness::new("passing");
    let mut talos = h.talos(
        vec![
            // First turn.
            tool_call(
                "1",
                "write_file",
                serde_json::json!({"path": "src/lib.rs", "content": "pub fn one() -> u32 { 1 }\n"}),
            ),
            text_response("Added one."),
            // Second turn, after the user speaks again.
            tool_call(
                "2",
                "write_file",
                serde_json::json!({
                    "path": "src/lib.rs",
                    "content": "pub fn one() -> u32 { 1 }\npub fn two() -> u32 { 2 }\n"
                }),
            ),
            text_response("Added two as well."),
        ],
        6,
        false,
    );

    let first = talos
        .run("add one()", &Plan { steps: vec!["add one".into()] })
        .await
        .unwrap();
    assert_eq!(first.halt, Halt::Done, "{}", first.summary);
    let messages_after_first = talos.messages.len();

    let second = talos.resume("now add two() as well").await.unwrap();
    assert_eq!(second.halt, Halt::Done, "{}", second.summary);

    // Context was kept, not restarted.
    assert!(talos.messages.len() > messages_after_first);
    assert!(talos.task == "add one()", "the original task is retained for tier 4");

    // Both functions exist, so the second turn built on the first.
    let src = std::fs::read_to_string(h.root.join("src/lib.rs")).unwrap();
    assert!(src.contains("pub fn one"));
    assert!(src.contains("pub fn two"));

    // The step budget resets per turn rather than draining across the session.
    assert_eq!(second.steps_used, 2);
}

#[tokio::test]
async fn resume_on_a_fresh_talos_behaves_like_a_first_task() {
    let h = Harness::new("passing");
    let mut talos = h.talos(vec![text_response("Nothing to do.")], 4, false);

    let outcome = talos.resume("look around").await.unwrap();
    assert!(!talos.messages.is_empty());
    assert_eq!(talos.task, "look around");
    assert!(outcome.steps_used >= 1);
}

#[tokio::test]
async fn scribe_tracks_edits_made_during_the_run() {
    let h = Harness::new("passing");

    // `triple` does not exist at the start; after the write it must be in the
    // index, because the exact tier has to stay exact as the agent edits.
    let before = SymbolIndex::build(&h.root).unwrap();
    assert!(before.lookup("triple").is_empty());

    h.run(
        vec![
            tool_call(
                "1",
                "write_file",
                serde_json::json!({
                    "path": "src/lib.rs",
                    "content": "pub fn triple(x: u32) -> u32 { x * 3 }\n"
                }),
            ),
            text_response("Done."),
        ],
        6,
    )
    .await;

    let after = SymbolIndex::build(&h.root).unwrap();
    let hits = after.lookup("triple");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].signature, "pub fn triple(x: u32) -> u32");
}
