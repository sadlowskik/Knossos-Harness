//! End-to-end tests for the agent loop.
//!
//! Every one of these runs against `MockEngine`, so the suite needs no API
//! key, no network and no local model server — but the tools, the path jail,
//! Scribe, Oracle and Ariadne are all real. `cargo check` genuinely runs.

use std::path::{Path, PathBuf};

use knossos::ariadne::{Ariadne, Halt};
use knossos::engine::mock::{text_response, tool_call, MockEngine};
use knossos::hooks::Hook;
use knossos::interject::Interjections;
use knossos::metis::Plan;
use knossos::oracle::Oracle;
use knossos::scribe::SymbolIndex;
use knossos::session::Session;
use knossos::talos::{Outcome, Talos};
use knossos::themis::Themis;
use knossos::tools::{ToolCtx, ToolOutput, ToolRegistry};

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

    fn talos(&self, scripted: Vec<knossos::engine::Response>, max_steps: usize, dry: bool) -> Talos {
        let mut ctx = ToolCtx::new(&self.root);
        if dry {
            ctx = ctx.dry_run();
        }
        Talos::new(
            Box::new(MockEngine::new(scripted)),
            ToolRegistry::standard(),
            ctx,
            // No baseline. These tests are about the loop, and paying for a
            // full ladder run before each one costs the suite far more than it
            // proves — forgiveness is covered directly in `oracle`'s own tests.
            // The trade is that `drive`'s call to `prepare` is not exercised
            // here; it is one line, and the alternative was 83 extra seconds on
            // every run of this file.
            Oracle::new(&self.root).without_baseline(),
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

    async fn run(&self, scripted: Vec<knossos::engine::Response>, max_steps: usize) -> Outcome {
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

/// Speaks once, from inside a tool call.
///
/// This is how the test gets an interjection to land *between* engine turns
/// without depending on timing. A `push` before `run` would prove only that the
/// queue drains at step 1, which is indistinguishable from an ordinary
/// instruction; this pushes while step 1 is being carried out, so a delivery at
/// step 2 is the real behaviour under test.
struct SpeakDuringFirstToolCall {
    handle: Interjections,
    spoken: std::sync::atomic::AtomicBool,
}

impl Hook for SpeakDuringFirstToolCall {
    fn name(&self) -> &str {
        "test-speaker"
    }

    fn after(&self, _tool: &str, _input: &serde_json::Value, _out: &ToolOutput) {
        use std::sync::atomic::Ordering;
        if !self.spoken.swap(true, Ordering::SeqCst) {
            self.handle.push("actually, name it quadruple");
        }
    }
}

#[tokio::test]
async fn a_word_from_the_user_reaches_the_next_step_without_ending_the_run() {
    let h = Harness::new("passing");

    let interjections = Interjections::new();
    let registry = ToolRegistry::standard().with_hook(Box::new(SpeakDuringFirstToolCall {
        handle: interjections.clone(),
        spoken: std::sync::atomic::AtomicBool::new(false),
    }));

    let mut talos = Talos::new(
        Box::new(MockEngine::new(vec![
            tool_call("1", "write_file", serde_json::json!({
                "path": "src/added.rs",
                "content": "pub fn triple(n: i32) -> i32 { n * 3 }\n"
            })),
            tool_call("2", "read_file", serde_json::json!({"path": "src/added.rs"})),
            text_response("done"),
        ])),
        registry,
        ToolCtx::new(&h.root),
        Oracle::new(&h.root).without_baseline(),
        SymbolIndex::build(&h.root).unwrap(),
        Themis::from_text("Be correct."),
        Ariadne::new(6, 5),
        Session::new(&h.root, "mock").with_trace(&h.trace).unwrap(),
        1024,
        false,
    )
    // The hook already holds a queue and was moved into the registry above, so
    // Talos adopts that one rather than the queue `new` would have made.
    .with_interjections(interjections.clone());

    let plan = Plan { steps: vec!["do the thing".into()] };
    let outcome = talos.run("test task", &plan).await.unwrap();

    let interjected: Vec<_> = h
        .trace_events()
        .into_iter()
        .filter(|e| e["event"] == "interjected")
        .collect();

    assert_eq!(interjected.len(), 1, "exactly one delivery, not one per step");
    assert_eq!(
        interjected[0]["notes"][0], "actually, name it quadruple",
        "the user's words, verbatim"
    );
    assert!(
        interjected[0]["step"].as_u64().unwrap() >= 2,
        "delivered at a later step, not folded into the opening request"
    );

    // The point of interjecting rather than cancelling: the run keeps going.
    assert!(outcome.steps_used >= 2, "the run continued past the interruption");
    assert!(
        talos.interjections.is_empty(),
        "nothing may be left queued when the run ends"
    );
}

/// A collected trace has to contain a training example, not a summary of one.
///
/// The bar is reconstruction: from the trace alone, can you recover what the
/// model was shown and what it produced, at every step? Presence of the event
/// is not enough — step 5 has to carry the conversation as it stood at step 5,
/// which is the part a naive implementation gets wrong by logging only the
/// latest turn.
#[tokio::test]
async fn a_collected_trace_carries_the_prompt_and_the_completion() {
    let h = Harness::new("passing");

    let mut talos = Talos::new(
        Box::new(MockEngine::new(vec![
            tool_call(
                "1",
                "write_file",
                serde_json::json!({
                    "path": "src/added.rs",
                    "content": "pub fn triple(n: i32) -> i32 { n * 3 }\n"
                }),
            ),
            text_response("done"),
        ])),
        ToolRegistry::standard(),
        ToolCtx::new(&h.root),
        Oracle::new(&h.root).without_baseline(),
        SymbolIndex::build(&h.root).unwrap(),
        Themis::from_text("Be correct."),
        Ariadne::new(6, 5),
        Session::new(&h.root, "mock")
            .with_trace(&h.trace)
            .unwrap()
            .collecting(),
        1024,
        false,
    );

    let plan = Plan { steps: vec!["do the thing".into()] };
    talos.run("add a triple function", &plan).await.unwrap();

    let exchanges: Vec<_> = h
        .trace_events()
        .into_iter()
        .filter(|e| e["event"] == "exchange")
        .collect();

    assert!(exchanges.len() >= 2, "one exchange per engine call");

    let first = &exchanges[0];
    assert!(
        first["request"]["system"].as_str().unwrap().contains("Be correct."),
        "the system prompt has to be the one the model actually saw"
    );
    let opening = first["request"]["messages"].as_array().unwrap();
    assert!(!opening.is_empty(), "the prompt side of the pair");
    assert_eq!(
        first["response"]["content"][0]["kind"], "tool_use",
        "the completion side of the pair, verbatim"
    );

    // The reconstruction check. A later step must show the history it was
    // actually given, or the example is a different one from the one that ran.
    let last = exchanges.last().unwrap();
    let later = last["request"]["messages"].as_array().unwrap();
    assert!(
        later.len() > opening.len(),
        "step N must record the conversation as it stood at step N"
    );
    assert!(
        serde_json::to_string(later).unwrap().contains("triple"),
        "the earlier turn's work has to appear in the later prompt"
    );
}

#[tokio::test]
async fn a_trace_carries_no_prompts_unless_asked() {
    // The default stays an audit log. Exchanges change the size of a trace by
    // orders of magnitude, so every ordinary run must not pay for them.
    // One step, one scripted reply: enough to produce a trace, not enough to
    // outrun the mock.
    let h = Harness::new("passing");
    h.run(vec![text_response("nothing to do")], 1).await;

    let events = h.trace_events();
    assert!(!events.is_empty(), "the run has to have traced something");
    assert!(
        events.iter().all(|e| e["event"] != "exchange"),
        "collecting is opt-in"
    );
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

// ------------------------------------------------- doing nothing is not done

#[tokio::test]
async fn an_empty_reply_is_not_a_claim_of_completion() {
    // An empty reply has no tool calls, so it used to take the "engine believes
    // it is finished" branch -- where an empty change set satisfies every tier
    // vacuously. Found on the Python side against a live reasoning model that
    // spent its whole budget before producing content; the same shape was here.
    let h = Harness::new("passing");

    let outcome = h
        .run(
            vec![
                text_response(""),
                text_response(""),
                text_response(""),
                text_response(""),
            ],
            4,
        )
        .await;

    assert!(!outcome.succeeded(), "a run that said nothing did not succeed");
    assert_ne!(outcome.halt, Halt::Done);
    assert!(outcome.changed.is_empty());
}

#[tokio::test]
async fn a_passing_verdict_cannot_end_a_run_that_did_nothing() {
    // `passing` compiles cleanly, so the real cargo ladder passes over an empty
    // change set. That is a verdict about the repository, not about the task.
    let h = Harness::new("passing");

    let outcome = h
        .run(
            vec![
                text_response("Nothing needs doing."),
                text_response("Still nothing."),
                text_response("Still nothing."),
                text_response("Still nothing."),
            ],
            4,
        )
        .await;

    assert!(!outcome.succeeded());
    assert!(outcome.changed.is_empty());
}

#[tokio::test]
async fn a_task_that_changes_no_file_can_still_succeed() {
    // The discriminator is not whether a *file* changed -- otherwise "run the
    // tests and tell me what breaks" could never finish. It is whether a tool
    // that can change something outside the conversation succeeded, which `run`
    // does and `read_file` does not.
    let h = Harness::new("passing");

    let outcome = h
        .run(
            vec![
                tool_call("1", "run", serde_json::json!({"command": "cargo --version"})),
                text_response("It builds with the stable toolchain."),
            ],
            4,
        )
        .await;

    assert!(outcome.succeeded(), "running a command is real work: {}", outcome.summary);
    assert!(outcome.changed.is_empty());
}

#[tokio::test]
async fn reading_a_file_is_not_doing_the_task() {
    // The vacuous pass this check exists to prevent, reached through
    // `read_file`. `acted` used to count any successful call, so an engine
    // could answer "add a triple() to src/lib.rs" by reading src/lib.rs and
    // asking to be verified: the change set is empty, `passing` compiles so
    // every tier is vacuously satisfied, and the run halted Done reporting
    // "Completed and verified" having written nothing at all.
    let h = Harness::new("passing");

    let mut talos = h.talos(
        vec![
            tool_call("1", "read_file", serde_json::json!({"path": "src/lib.rs"})),
            text_response("I have added triple()."),
            text_response("Still added."),
            text_response("Truly added."),
        ],
        4,
        false,
    );

    let outcome = talos
        .run(
            "add a triple(x: u32) -> u32 to src/lib.rs",
            &Plan { steps: vec!["add triple".into()] },
        )
        .await
        .unwrap();

    assert_ne!(outcome.halt, Halt::Done, "{}", outcome.summary);
    assert!(!outcome.succeeded());
    assert!(outcome.changed.is_empty());

    // And the engine was told why, rather than being left to repeat itself.
    let conversation: String =
        talos.messages.iter().map(|m| m.text()).collect::<Vec<_>>().join("\n");
    assert!(
        conversation.contains("nothing to verify"),
        "the engine should have been told reading is not doing: {conversation}"
    );
}

/// Records what it was asked, and answers with a fixed verdict.
struct Recording {
    allow: bool,
    asked: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

#[async_trait::async_trait]
impl knossos::talos::Approver for Recording {
    async fn approve(&self, tool: &str, _input: &serde_json::Value) -> bool {
        self.asked.lock().unwrap().push(tool.to_string());
        self.allow
    }
}

#[tokio::test]
async fn without_an_approver_a_run_is_unattended() {
    // The default, and it is the right one for `daedalus task`: a one-shot CLI
    // is non-interactive by design, and prompting there hangs CI and every
    // scripted use. Pinned so it cannot drift into a silent full-allow that
    // nobody chose.
    let h = Harness::new("passing");
    let mut talos = h.talos(
        vec![
            tool_call(
                "1",
                "write_file",
                serde_json::json!({"path": "src/added.rs", "content": "pub fn f() {}\n"}),
            ),
            text_response("Done."),
        ],
        3,
        false,
    );
    assert!(talos.approver.is_none());

    talos
        .run("add a file", &Plan { steps: vec!["add".into()] })
        .await
        .unwrap();

    assert!(h.root.join("src/added.rs").exists());
}

#[tokio::test]
async fn an_approver_is_asked_only_about_consequential_calls() {
    let h = Harness::new("passing");
    let asked = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut talos = h.talos(
        vec![
            tool_call("1", "read_file", serde_json::json!({"path": "src/lib.rs"})),
            tool_call(
                "2",
                "write_file",
                serde_json::json!({"path": "src/added.rs", "content": "pub fn f() {}\n"}),
            ),
            text_response("Done."),
        ],
        4,
        false,
    );
    talos.approver = Some(std::sync::Arc::new(Recording {
        allow: true,
        asked: asked.clone(),
    }));

    talos
        .run("read then write", &Plan { steps: vec!["do it".into()] })
        .await
        .unwrap();

    // Asking about every read trains people to approve without looking, which
    // is worse than not asking. `Tool::consequential` is the discriminator and
    // has no default, so a new tool cannot be silently unclassified.
    assert_eq!(*asked.lock().unwrap(), vec!["write_file"]);
}

#[tokio::test]
async fn a_refused_call_does_not_count_as_work_done() {
    // The interaction that matters most: `acted` gates `Halt::Done`, so if a
    // refusal counted as acting, denying every write would still let a run
    // report success over an empty change set.
    let h = Harness::new("passing");
    let asked = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut talos = h.talos(
        vec![
            tool_call(
                "1",
                "write_file",
                serde_json::json!({"path": "src/nope.rs", "content": "pub fn f() {}\n"}),
            ),
            text_response("I could not write it."),
            text_response("Still could not."),
            text_response("Nothing further."),
        ],
        4,
        false,
    );
    talos.approver = Some(std::sync::Arc::new(Recording {
        allow: false,
        asked: asked.clone(),
    }));

    let outcome = talos
        .run("add a file", &Plan { steps: vec!["add".into()] })
        .await
        .unwrap();

    assert!(!h.root.join("src/nope.rs").exists(), "a refused write happened anyway");
    assert_ne!(outcome.halt, Halt::Done, "{}", outcome.summary);
    assert!(outcome.changed.is_empty());

    // And the engine was told, so it can propose something else rather than
    // repeating the same call into the same refusal.
    let conversation: String = talos
        .messages
        .iter()
        .flat_map(|m| &m.content)
        .filter_map(|c| match c {
            knossos::engine::Content::ToolResult { content, .. } => Some(content.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(conversation.contains("not permitted"), "{conversation}");
}

#[tokio::test]
async fn the_conversation_is_bounded_across_a_run() {
    // `messages` kept every block of every turn and the whole request was
    // rebuilt each step, so a run grew without bound -- measured at roughly
    // 96 000 tokens by the end of turn one and past 270 000 by turn three. The
    // step ceiling was the only thing keeping it finite, which is a blunt
    // instrument rather than a bound.
    let h = Harness::new("passing");
    // Two *different* files, deliberately. Reading the same one twice changes
    // nothing on the second read, which `is_futile` now recognises as a repeat
    // and answers with a feedback note -- correct behaviour, but it lands after
    // the final compaction and so shows up in the post-run total, which is not
    // what this test is about. Distinct files exercise the same bound (a
    // conversation far larger than the ceiling) without the entanglement.
    // The repeat path has its own test: `a_repeated_call_is_told_it_is_repeating`.
    let big = "x".repeat(300_000);
    std::fs::write(h.root.join("src/huge.rs"), &big).unwrap();
    std::fs::write(h.root.join("src/huger.rs"), &big).unwrap();

    let mut talos = h.talos(
        vec![
            tool_call("1", "read_file", serde_json::json!({"path": "src/huge.rs"})),
            tool_call("2", "read_file", serde_json::json!({"path": "src/huger.rs"})),
            text_response("Done."),
        ],
        3,
        false,
    );
    talos.lethe = knossos::lethe::Lethe { max_tokens: 4_000, ..Default::default() };

    talos
        .run("read the big file", &Plan { steps: vec!["read".into()] })
        .await
        .unwrap();

    assert!(
        knossos::lethe::estimate_tokens(&talos.messages) <= 4_000,
        "conversation was {} tokens",
        knossos::lethe::estimate_tokens(&talos.messages)
    );

    // The load-bearing invariant: every tool call still has its result. An id
    // without its partner is a hard provider error, not a degradation.
    let uses = talos
        .messages
        .iter()
        .flat_map(|m| &m.content)
        .filter(|c| matches!(c, knossos::engine::Content::ToolUse { .. }))
        .count();
    let results = talos
        .messages
        .iter()
        .flat_map(|m| &m.content)
        .filter(|c| matches!(c, knossos::engine::Content::ToolResult { .. }))
        .count();
    assert_eq!(uses, results, "compaction split a tool call from its result");
}

#[tokio::test]
async fn compaction_is_visible_in_the_trace() {
    // Otherwise it is invisible: the run continues normally and the only
    // evidence that context was given up is the model failing to refer to
    // something it was told earlier.
    let h = Harness::new("passing");
    std::fs::write(h.root.join("src/huge.rs"), "x".repeat(300_000)).unwrap();

    let mut talos = h.talos(
        vec![
            tool_call("1", "read_file", serde_json::json!({"path": "src/huge.rs"})),
            text_response("Done."),
        ],
        2,
        false,
    );
    talos.lethe = knossos::lethe::Lethe { max_tokens: 4_000, ..Default::default() };
    talos
        .run("read it", &Plan { steps: vec!["read".into()] })
        .await
        .unwrap();

    // `#[serde(tag = "event", rename_all = "snake_case")]`, so this is the
    // exact discriminant -- a looser match could pass on any event mentioning
    // the word and would not be evidence of anything.
    let events = h.trace_events();
    let compactions: Vec<_> = events
        .iter()
        .filter(|e| e["event"] == "context_compacted")
        .collect();

    assert!(!compactions.is_empty(), "no compaction event in {events:?}");
    assert!(
        compactions[0]["tokens"].as_u64().unwrap() <= 4_000,
        "the trace should record the size it reached: {:?}",
        compactions[0]
    );
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
                // Three, not one: a refused call accomplishes nothing, so the
                // engine claiming to be done no longer ends the run. It keeps
                // being asked until Ariadne calls it stuck.
                text_response("Done."),
                text_response("Done."),
                text_response("Done."),
            ],
            4,
        )
        .await;

    // The write was refused, so nothing changed and the file does not exist.
    assert!(!outcome.succeeded(), "a refused run did not accomplish the task");
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
                text_response("Done."),
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
    // Repeated because "nothing to do" no longer ends a run on its own.
    let mut talos = h.talos(
        vec![
            text_response("Nothing to do."),
            text_response("Nothing to do."),
            text_response("Nothing to do."),
            text_response("Nothing to do."),
        ],
        4,
        false,
    );

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

// ------------------------------------------------------------------- futility
//
// Ported from the Python side, where the check existed but compared only
// against the immediately previous step. Here it did not exist at all:
// `is_noop` was the whole staleness test, and an engine re-issuing one failing
// `edit_file` calls a tool every step, so it was never a noop and every such
// run went to the ceiling.

/// A call that always fails: `old_string` is not in the file, so nothing changes.
fn doomed_edit(id: &str, marker: &str) -> knossos::engine::Response {
    tool_call(
        id,
        "edit_file",
        serde_json::json!({
            "path": "src/lib.rs",
            "old_string": format!("absent-{marker}"),
            "new_string": "x"
        }),
    )
}

#[tokio::test]
async fn an_engine_repeating_one_failing_call_is_stuck() {
    // `is_noop` cannot see this: the engine *is* calling a tool every step.
    let h = Harness::new("passing");
    let script: Vec<_> = (0..12).map(|i| doomed_edit(&i.to_string(), "a")).collect();

    let outcome = h.run(script, 12).await;

    assert_eq!(outcome.halt, Halt::Stuck);
    assert_eq!(outcome.steps_used, 3, "one attempt, then two repeats");
}

#[tokio::test]
async fn an_engine_alternating_between_two_failing_calls_is_stuck() {
    // The loop an adjacent-only check cannot see, and which nothing here saw at
    // all: A, B, A, B never repeats itself consecutively.
    let h = Harness::new("passing");
    let script: Vec<_> = (0..20)
        .map(|i| doomed_edit(&i.to_string(), if i % 2 == 0 { "a" } else { "b" }))
        .collect();

    let outcome = h.run(script, 20).await;

    assert_eq!(outcome.halt, Halt::Stuck);
    assert_eq!(outcome.steps_used, 4, "A B then A B again: caught on the second B");
}

#[tokio::test]
async fn a_three_step_ritual_that_achieves_nothing_is_stuck() {
    let h = Harness::new("passing");
    let marks = ["a", "b", "c"];
    let script: Vec<_> = (0..20)
        .map(|i| doomed_edit(&i.to_string(), marks[i % 3]))
        .collect();

    let outcome = h.run(script, 20).await;

    assert_eq!(outcome.halt, Halt::Stuck);
    assert_eq!(outcome.steps_used, 5, "three to establish the cycle, two to confirm");
}

#[tokio::test]
async fn a_repeated_call_is_told_it_is_repeating() {
    // Halting is the backstop; the cheaper outcome is the engine noticing it is
    // going in a circle while it still has budget left.
    let h = Harness::new("passing");
    let script: Vec<_> = (0..4).map(|i| doomed_edit(&i.to_string(), "a")).collect();
    let mut talos = h.talos(script, 6, false);

    talos
        .run("edit something", &Plan { steps: vec!["edit".into()] })
        .await
        .unwrap();

    let conversation = format!("{:?}", talos.messages);
    assert!(
        conversation.contains("same tool call"),
        "the engine was never told it was repeating"
    );
}

#[tokio::test]
async fn real_work_between_repeats_restarts_the_window() {
    // The half that keeps a wider window safe: exploring, changing something,
    // then exploring the same way again is ordinary work, not a loop.
    let h = Harness::new("passing");
    let read = |id: &str| tool_call(id, "read_file", serde_json::json!({"path": "src/lib.rs"}));
    let write = |id: &str, body: &str| {
        tool_call(
            id,
            "write_file",
            serde_json::json!({"path": "src/added.rs", "content": body}),
        )
    };

    let outcome = h
        .run(
            vec![
                read("1"),
                read("2"),
                write("3", "pub fn a() {}\n"),
                read("4"),
                read("5"),
                write("6", "pub fn b() {}\n"),
                text_response("Done."),
            ],
            10,
        )
        .await;

    assert_ne!(outcome.halt, Halt::Stuck, "real work happened between the reads");
}

// ----------------------------- every step sequence, not just the ones anyone
//                                thought to write a test for
//
// `reference_halt` states the intended rule without reference to `Talos`.
// Agreement across every sequence in the space means a transcription error in
// either one surfaces as a disagreement on a specific input, which the failure
// message prints. What it cannot pin is the window *width* -- both sides read
// `FUTILE_WINDOW` -- so the two-cycle and three-cycle tests above hardcode the
// step at which a loop must be noticed, and those are what fix the value.

/// `true` when the symbol changes a file.
fn changes(symbol: char) -> bool {
    symbol == 'W'
}

fn script_for(symbols: &str) -> Vec<knossos::engine::Response> {
    symbols
        .chars()
        .enumerate()
        .map(|(i, c)| {
            let id = i.to_string();
            if changes(c) {
                tool_call(
                    &id,
                    "write_file",
                    serde_json::json!({"path": "src/sweep.rs", "content": "pub fn s() {}\n"}),
                )
            } else {
                doomed_edit(&id, &c.to_string())
            }
        })
        .collect()
}

/// The rule as specified: futile when a signature seen within the last
/// `FUTILE_WINDOW` steps recurs and nothing changed; a step that changes
/// something restarts the window while remaining in it. `assess` checks the
/// ceiling before staleness, so a run that would be stuck on its final
/// permitted step reports budget exhaustion — reproduced here deliberately.
fn reference_halt(symbols: &str) -> (Halt, usize) {
    use std::collections::VecDeque;
    let max_steps = symbols.chars().count();
    let mut recent: VecDeque<char> = VecDeque::new();
    let mut noops = 0usize;
    for (i, c) in symbols.chars().enumerate() {
        let step = i + 1;
        let futile = recent.contains(&c) && !changes(c);
        if changes(c) {
            recent.clear();
        }
        if recent.len() == knossos::talos::FUTILE_WINDOW {
            recent.pop_front();
        }
        recent.push_back(c);
        if futile {
            noops += 1;
        } else {
            noops = 0;
        }
        if step >= max_steps {
            return (Halt::BudgetExhausted, step);
        }
        if noops >= 2 {
            return (Halt::Stuck, step);
        }
    }
    unreachable!("the ceiling equals the script length")
}

async fn drive(symbols: &str) -> (Halt, usize) {
    let h = Harness::new("passing");
    let n = symbols.chars().count();
    let outcome = h.run(script_for(symbols), n).await;
    (outcome.halt, outcome.steps_used)
}

async fn sweep(alphabet: &[char], length: usize) {
    let total = alphabet.len().pow(length as u32);
    for n in 0..total {
        let mut rest = n;
        let mut symbols = vec![alphabet[0]; length];
        for slot in symbols.iter_mut() {
            *slot = alphabet[rest % alphabet.len()];
            rest /= alphabet.len();
        }
        let word: String = symbols.iter().collect();
        assert_eq!(drive(&word).await, reference_halt(&word), "disagreed on {word}");
    }
}

#[tokio::test]
async fn the_loop_matches_the_specified_rule_on_every_short_sequence() {
    // Two useless calls and one that works, exhaustively to depth 4.
    sweep(&['a', 'b', 'W'], 4).await;
}

#[tokio::test]
async fn the_loop_matches_the_specified_rule_on_every_alternation() {
    // The alternation family, at depth: every sequence over two failing calls.
    sweep(&['a', 'b'], 5).await;
}

#[tokio::test]
async fn no_sequence_of_real_work_is_ever_called_stuck() {
    // The false-positive direction. If the `files_changed == 0` half of
    // `is_futile` were ever dropped, this is what would start failing.
    for length in 1..=5 {
        let word: String = std::iter::repeat('W').take(length).collect();
        let (halt, _) = drive(&word).await;
        assert_ne!(halt, Halt::Stuck, "{word} did real work every step");
    }
}
