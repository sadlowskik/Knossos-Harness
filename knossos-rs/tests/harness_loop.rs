//! End-to-end tests for the agent loop.
//!
//! Every one of these runs against `MockEngine`, so the suite needs no API
//! key, no network and no local model server — but the tools, the path jail,
//! Scribe, Oracle and Ariadne are all real. `cargo check` genuinely runs.

use std::path::{Path, PathBuf};

use knossos::argus::Argus;
use knossos::ariadne::{Ariadne, Halt};
use knossos::engine::mock::{text_response, tool_call, MockEngine};
use knossos::gate::RetrievalGate;
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
    let src = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
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
        Harness {
            _dir: dir,
            root,
            trace,
        }
    }

    fn talos(
        &self,
        scripted: Vec<knossos::engine::Response>,
        max_steps: usize,
        dry: bool,
    ) -> Talos {
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
            Session::new(&self.root, "mock")
                .with_trace(&self.trace)
                .unwrap(),
            1024,
            // Tier 4 needs an engine turn of its own; the tests that exercise
            // it script that turn explicitly.
            false,
        )
    }

    async fn run(&self, scripted: Vec<knossos::engine::Response>, max_steps: usize) -> Outcome {
        let mut talos = self.talos(scripted, max_steps, false);
        let plan = Plan {
            steps: vec!["do the thing".into()],
        };
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
            tool_call(
                "1",
                "write_file",
                serde_json::json!({
                    "path": "src/added.rs",
                    "content": "pub fn triple(n: i32) -> i32 { n * 3 }\n"
                }),
            ),
            tool_call(
                "2",
                "read_file",
                serde_json::json!({"path": "src/added.rs"}),
            ),
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

    let plan = Plan {
        steps: vec!["do the thing".into()],
    };
    let outcome = talos.run("test task", &plan).await.unwrap();

    let interjected: Vec<_> = h
        .trace_events()
        .into_iter()
        .filter(|e| e["event"] == "interjected")
        .collect();

    assert_eq!(
        interjected.len(),
        1,
        "exactly one delivery, not one per step"
    );
    assert_eq!(
        interjected[0]["notes"][0], "actually, name it quadruple",
        "the user's words, verbatim"
    );
    assert!(
        interjected[0]["step"].as_u64().unwrap() >= 2,
        "delivered at a later step, not folded into the opening request"
    );

    // The point of interjecting rather than cancelling: the run keeps going.
    assert!(
        outcome.steps_used >= 2,
        "the run continued past the interruption"
    );
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

    let plan = Plan {
        steps: vec!["do the thing".into()],
    };
    talos.run("add a triple function", &plan).await.unwrap();

    let exchanges: Vec<_> = h
        .trace_events()
        .into_iter()
        .filter(|e| e["event"] == "exchange_delta")
        .collect();

    assert!(exchanges.len() >= 2, "one exchange per engine call");

    let first = &exchanges[0];
    assert!(
        first["system"].as_str().unwrap().contains("Be correct."),
        "the system prompt has to be the one the model actually saw"
    );
    let opening = first["messages"].as_array().unwrap();
    assert!(!opening.is_empty(), "the prompt side of the pair");
    assert_eq!(
        first["response"]["content"][0]["kind"], "tool_use",
        "the completion side of the pair, verbatim"
    );

    // The reconstruction check. Deltas must reproduce the history the later
    // engine call actually received without storing the whole prefix each time.
    let mut later = Vec::new();
    for exchange in &exchanges {
        if exchange["reset"].as_bool().unwrap_or(false) {
            later.clear();
        }
        assert_eq!(
            exchange["messages_start"].as_u64().unwrap() as usize,
            later.len()
        );
        later.extend(exchange["messages"].as_array().unwrap().iter().cloned());
    }
    assert!(
        later.len() > opening.len(),
        "step N must record the conversation as it stood at step N"
    );
    assert!(
        serde_json::to_string(&later).unwrap().contains("triple"),
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
    assert!(
        events.iter().all(|e| e["event"] != "exchange_delta"),
        "delta collection is opt-in"
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

    let verdict = outcome
        .verdict
        .expect("a Done outcome must carry a verdict");
    assert!(verdict.passed);
    // The full deterministic ladder ran: syntax, check, clippy, test.
    assert!(
        verdict.reached_tier >= 3,
        "reached tier {}",
        verdict.reached_tier
    );
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
    assert_eq!(
        failed.tier, 1,
        "syntax is valid; cargo check is what must fail"
    );
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
    assert_eq!(
        verdict.tiers.len(),
        1,
        "nothing past tier 0 should have run"
    );
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

    assert!(
        !outcome.succeeded(),
        "a run that said nothing did not succeed"
    );
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
                tool_call(
                    "1",
                    "run",
                    serde_json::json!({"command": "cargo --version"}),
                ),
                text_response("It builds with the stable toolchain."),
            ],
            4,
        )
        .await;

    assert!(
        outcome.succeeded(),
        "running a command is real work: {}",
        outcome.summary
    );
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
            &Plan {
                steps: vec!["add triple".into()],
            },
        )
        .await
        .unwrap();

    assert_ne!(outcome.halt, Halt::Done, "{}", outcome.summary);
    assert!(!outcome.succeeded());
    assert!(outcome.changed.is_empty());

    // And the engine was told why, rather than being left to repeat itself.
    let conversation: String = talos
        .messages
        .iter()
        .map(|m| m.text())
        .collect::<Vec<_>>()
        .join("\n");
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
        .run(
            "add a file",
            &Plan {
                steps: vec!["add".into()],
            },
        )
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
        .run(
            "read then write",
            &Plan {
                steps: vec!["do it".into()],
            },
        )
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
        .run(
            "add a file",
            &Plan {
                steps: vec!["add".into()],
            },
        )
        .await
        .unwrap();

    assert!(
        !h.root.join("src/nope.rs").exists(),
        "a refused write happened anyway"
    );
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
            tool_call(
                "2",
                "read_file",
                serde_json::json!({"path": "src/huger.rs"}),
            ),
            text_response("Done."),
        ],
        3,
        false,
    );
    talos.lethe = knossos::lethe::Lethe {
        max_tokens: 4_000,
        ..Default::default()
    };

    talos
        .run(
            "read the big file",
            &Plan {
                steps: vec!["read".into()],
            },
        )
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
    assert_eq!(
        uses, results,
        "compaction split a tool call from its result"
    );
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
    talos.lethe = knossos::lethe::Lethe {
        max_tokens: 4_000,
        ..Default::default()
    };
    talos
        .run(
            "read it",
            &Plan {
                steps: vec!["read".into()],
            },
        )
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
    assert!(
        !outcome.succeeded(),
        "a refused run did not accomplish the task"
    );
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
            && e["output"]
                .as_str()
                .unwrap_or("")
                .contains("not on the allowlist")
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

    for expected in [
        "plan_produced",
        "step_started",
        "tool_call",
        "oracle_verdict",
        "halt",
        "task_finished",
    ] {
        assert!(
            kinds.contains(&expected),
            "trace missing `{expected}`; got {kinds:?}"
        );
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
        .run(
            "test task",
            &Plan {
                steps: vec!["s".into()],
            },
        )
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

    talos
        .run(
            "t",
            &Plan {
                steps: vec!["s".into()],
            },
        )
        .await
        .unwrap();
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
        .run(
            "add one()",
            &Plan {
                steps: vec!["add one".into()],
            },
        )
        .await
        .unwrap();
    assert_eq!(first.halt, Halt::Done, "{}", first.summary);
    let messages_after_first = talos.messages.len();

    let second = talos.resume("now add two() as well").await.unwrap();
    assert_eq!(second.halt, Halt::Done, "{}", second.summary);

    // Context was kept, not restarted.
    assert!(talos.messages.len() > messages_after_first);
    assert!(
        talos.task == "add one()",
        "the original task is retained for tier 4"
    );

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
    assert_eq!(
        outcome.steps_used, 5,
        "one attempt, two repeats, one redirect, two more"
    );
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
    assert_eq!(
        outcome.steps_used, 6,
        "A B A B redirects, then A B forbidden is stuck"
    );
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
    assert_eq!(
        outcome.steps_used, 7,
        "three to establish, two to confirm, redirect, two more"
    );
}

#[tokio::test]
async fn a_repeated_call_is_told_it_is_repeating() {
    // Halting is the backstop; the cheaper outcome is the engine noticing it is
    // going in a circle while it still has budget left.
    let h = Harness::new("passing");
    let script: Vec<_> = (0..8).map(|i| doomed_edit(&i.to_string(), "a")).collect();
    let mut talos = h.talos(script, 8, false);

    talos
        .run(
            "edit something",
            &Plan {
                steps: vec!["edit".into()],
            },
        )
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

    assert_ne!(
        outcome.halt,
        Halt::Stuck,
        "real work happened between the reads"
    );
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
    let mut forbidden: Vec<char> = Vec::new();
    let mut noops = 0usize;
    let mut redirects = 0usize;
    for (i, c) in symbols.chars().enumerate() {
        let step = i + 1;
        let futile = (recent.contains(&c) || forbidden.contains(&c)) && !changes(c);
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
            if redirects < 1 {
                redirects += 1;
                forbidden.extend(recent.iter().copied());
                noops = 0;
                recent.clear();
                continue;
            }
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
        assert_eq!(
            drive(&word).await,
            reference_halt(&word),
            "disagreed on {word}"
        );
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
        let word = "W".repeat(length);
        let (halt, _) = drive(&word).await;
        assert_ne!(halt, Halt::Stuck, "{word} did real work every step");
    }
}

// ------------------------------------------------------ proactive retrieval
//
// Context the harness offers unasked, as opposed to the `search_code` tool the
// model asks for. The gate is what makes offering it safe, so both directions
// are pinned: the task that should get code, and the question that should not.

impl Harness {
    /// A Talos that offers repository context unasked.
    ///
    /// One step per turn: these pin the shape of the first prompt, and a longer
    /// budget only buys more engine calls to script.
    fn talos_with_retrieval(&self, scripted: Vec<knossos::engine::Response>) -> Talos {
        let mut argus = Argus::new(&self.root);
        argus.scan();
        self.talos(scripted, 1, false)
            .with_retrieval(RetrievalGate::new(std::sync::Arc::new(argus)))
    }
}

/// The `ContextConsidered` record for a run, which must exist either way — a
/// wrong skip is invisible without it.
fn context_event(h: &Harness) -> serde_json::Value {
    h.trace_events()
        .into_iter()
        .find(|e| e["event"] == "context_considered")
        .expect("every gated turn records its decision")
}

#[tokio::test]
async fn a_task_naming_repository_code_is_given_it_unasked() {
    let h = Harness::new("passing");
    let mut talos = h.talos_with_retrieval(vec![text_response("done")]);
    let plan = Plan {
        steps: vec!["look at the adder".into()],
    };

    talos
        .run("fix the rounding in Adder::add", &plan)
        .await
        .unwrap();

    let first = talos.messages[0].text();
    assert!(
        first.contains("retrieved automatically"),
        "no context block:\n{first}"
    );
    assert!(
        first.contains("Adder"),
        "the named type was not retrieved:\n{first}"
    );
    // The instruction has to stay last, or the injected code becomes the most
    // recent thing the model read and starts looking like the request.
    assert!(
        first
            .trim_end()
            .ends_with("verification runs automatically."),
        "context displaced the instruction:\n{first}",
    );
    assert_eq!(context_event(&h)["injected"], serde_json::json!(true));
}

#[tokio::test]
async fn a_general_question_is_not_given_repository_context() {
    let h = Harness::new("passing");
    let mut talos = h.talos_with_retrieval(vec![text_response("done")]);
    let plan = Plan {
        steps: vec!["explain".into()],
    };

    talos
        .run("what is a mixture-of-experts layer, in general?", &plan)
        .await
        .unwrap();

    let first = talos.messages[0].text();
    assert!(
        !first.contains("retrieved automatically"),
        "context was forced in:\n{first}"
    );
    let event = context_event(&h);
    assert_eq!(event["injected"], serde_json::json!(false));
    assert!(
        event["reasons"].as_array().is_some_and(|r| !r.is_empty()),
        "a skip with no stated reason cannot be argued with: {event}",
    );
}

/// A second turn is gated on its own words, not on the ones that opened the
/// session — the conversation may have moved to a different part of the tree.
#[tokio::test]
async fn a_resumed_turn_is_gated_on_its_own_instruction() {
    let h = Harness::new("passing");
    let mut talos =
        h.talos_with_retrieval(vec![text_response("done"), text_response("done again")]);
    let plan = Plan {
        steps: vec!["start".into()],
    };

    talos
        .run("what is attention, conceptually?", &plan)
        .await
        .unwrap();
    let before = talos.messages.len();
    talos
        .resume("now change Adder::add to saturate")
        .await
        .unwrap();

    let resumed = talos.messages[before].text();
    assert!(
        resumed.contains("retrieved automatically"),
        "second turn got nothing:\n{resumed}"
    );
    assert!(
        resumed.starts_with("now change Adder::add"),
        "the instruction moved:\n{resumed}"
    );
}

/// Without a gate attached nothing changes — no injection, and no decision to
/// record. The proactive path is opt-in and must stay invisible when it is off.
#[tokio::test]
async fn retrieval_left_unattached_changes_nothing() {
    let h = Harness::new("passing");
    let mut talos = h.talos(vec![text_response("done")], 1, false);
    let plan = Plan {
        steps: vec!["do the thing".into()],
    };

    talos
        .run("fix the rounding in Adder::add", &plan)
        .await
        .unwrap();

    let first = talos.messages[0].text();
    assert_eq!(
        first,
        "Task: fix the rounding in Adder::add\n\nPlan:\n1. do the thing\n\nWork through it. \
         When everything is complete, reply with a short summary and no tool calls — \
         verification runs automatically.",
    );
    assert!(
        h.trace_events()
            .iter()
            .all(|e| e["event"] != "context_considered"),
        "a disabled gate must not report decisions it never made",
    );
}

/// The trace records what the model *said*, not only what it did.
///
/// Tool calls were logged from the start; the prose around them was not, unless
/// the run was collecting whole exchanges. A reader could see every edit and
/// still not know what the agent claimed it was doing.
#[tokio::test]
async fn the_trace_records_what_the_model_said() {
    let h = Harness::new("passing");
    let mut talos = h.talos(vec![text_response("Looks fine to me.")], 1, false);
    let plan = Plan {
        steps: vec!["do the thing".into()],
    };

    talos.run("test task", &plan).await.unwrap();

    let said: Vec<String> = h
        .trace_events()
        .into_iter()
        .filter(|e| e["event"] == "agent_message")
        .filter_map(|e| e["text"].as_str().map(str::to_string))
        .collect();
    assert_eq!(said, ["Looks fine to me."]);
}

/// An empty reply is not a message. A model that answers with tool calls only
/// would otherwise produce a stream of blank bubbles in a front end.
#[tokio::test]
async fn an_empty_reply_is_not_recorded_as_a_message() {
    let h = Harness::new("passing");
    let mut talos = h.talos(vec![text_response("   ")], 1, false);
    let plan = Plan {
        steps: vec!["do the thing".into()],
    };

    talos.run("test task", &plan).await.unwrap();

    assert!(h
        .trace_events()
        .iter()
        .all(|e| e["event"] != "agent_message"));
}

// ------------------------------------------------------------- cancellation

/// Cancelling stops the next engine call rather than letting the turn run out.
///
/// The budget here is 5 with only one scripted reply: without the check the
/// loop would ask for a second turn and the mock would error. Finishing cleanly
/// is the evidence that no further request went out.
#[tokio::test]
async fn a_cancelled_turn_stops_at_the_next_step_boundary() {
    let h = Harness::new("passing");
    let mut talos = h.talos(vec![text_response("working on it")], 5, false);
    talos
        .cancel
        .store(true, std::sync::atomic::Ordering::SeqCst);

    let plan = Plan {
        steps: vec!["do the thing".into()],
    };
    let outcome = talos.run("test task", &plan).await.unwrap();

    assert_eq!(outcome.halt, Halt::Cancelled);
    assert_eq!(outcome.steps_used, 0, "cancelled before the first request");
}

/// A cancellation applies to one turn, not to the session.
///
/// Latching would mean the next thing the user asked for died on arrival, with
/// no obvious cause — the flag was set minutes earlier for something else.
#[tokio::test]
async fn cancelling_one_turn_does_not_poison_the_next() {
    let h = Harness::new("passing");
    // The cancelled turn consumes none of these; the resumed one may use its
    // whole budget, since a reply that changes nothing gets pushed back on.
    let mut talos = h.talos(
        vec![
            text_response("second turn ran"),
            text_response("still nothing"),
            text_response("done"),
        ],
        3,
        false,
    );
    talos
        .cancel
        .store(true, std::sync::atomic::Ordering::SeqCst);

    let plan = Plan {
        steps: vec!["do the thing".into()],
    };
    assert_eq!(
        talos.run("first", &plan).await.unwrap().halt,
        Halt::Cancelled
    );

    // The flag cleared itself, so this turn reaches the engine.
    let second = talos.resume("carry on").await.unwrap();
    assert_ne!(
        second.halt,
        Halt::Cancelled,
        "the flag latched into the next turn"
    );
    assert!(second.steps_used >= 1);
}

/// A cancelled run is nobody's fault, and must not read as a failure.
#[tokio::test]
async fn cancelling_is_not_reported_as_a_failure() {
    let h = Harness::new("passing");
    let mut talos = h.talos(vec![text_response("x")], 3, false);
    talos
        .cancel
        .store(true, std::sync::atomic::Ordering::SeqCst);

    let plan = Plan {
        steps: vec!["do the thing".into()],
    };
    let outcome = talos.run("test task", &plan).await.unwrap();

    assert!(!outcome.succeeded(), "cancelled is not success either");
    assert!(
        outcome.summary.starts_with("Cancelled"),
        "{}",
        outcome.summary
    );
    let halts: Vec<_> = h
        .trace_events()
        .into_iter()
        .filter(|e| e["event"] == "halt")
        .collect();
    assert_eq!(halts.len(), 1, "exactly one halt event, not a duplicate");
    assert_eq!(halts[0]["reason"], "cancelled");
}

/// A multi-step plan is executed step by step, not pasted into one prompt.
#[tokio::test]
async fn a_multi_step_plan_is_driven_stepwise() {
    let h = Harness::new("passing");
    let valid = std::fs::read_to_string(h.root.join("src/lib.rs")).unwrap();
    let mut talos = h.talos(
        vec![
            // Step 1: act, then ask to be verified.
            tool_call(
                "1",
                "write_file",
                serde_json::json!({"path": "src/lib.rs", "content": valid.clone()}),
            ),
            text_response("step one done"),
            // Step 2: act again so this drive has its own `acted`.
            tool_call(
                "2",
                "write_file",
                serde_json::json!({"path": "src/lib.rs", "content": valid}),
            ),
            text_response("step two done"),
            // Closing, plus a couple of spares if a drive takes an extra turn.
            text_response("the task is done"),
            text_response("still done"),
            text_response("still done"),
        ],
        12,
        false,
    );

    let plan = Plan {
        steps: vec!["touch the crate".into(), "confirm it".into()],
    };
    let outcome = talos
        .run("add nothing, just walk the plan", &plan)
        .await
        .unwrap();

    let dumped = talos
        .messages
        .iter()
        .map(|m| m.text())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        dumped.contains("Step 1 of 2"),
        "step 1 was never framed:\n{dumped}"
    );
    assert!(
        dumped.contains("Step 2 of 2"),
        "step 2 was never framed:\n{dumped}"
    );
    assert!(
        dumped.contains("Plan complete"),
        "the closing drive never ran:\n{dumped}"
    );
    // Closing used the full ladder, not just syntax.
    let verdict = outcome.verdict.expect("closing must verify");
    assert!(verdict.reached_tier >= 1 || verdict.dry_run, "{verdict:?}");
}

/// A hard ceiling remains hard when there are too many plan slices to fund.
/// In that case the rendered plan is driven flat instead of silently flooring
/// every slice and spending more turns than the caller allowed.
#[tokio::test]
async fn a_short_budget_flattens_a_long_plan() {
    let h = Harness::new("passing");
    let mut talos = h.talos(
        vec![valid_write("1"), text_response("implemented and ready")],
        8,
        false,
    );
    let plan = Plan {
        steps: vec![
            "read the crate".into(),
            "add the module".into(),
            "verify it".into(),
        ],
    };

    let outcome = talos.run("add a harmless module", &plan).await.unwrap();

    assert_eq!(outcome.halt, Halt::Done, "{}", outcome.summary);
    assert!(
        outcome.steps_used <= 8,
        "hard ceiling exceeded: {outcome:?}"
    );
    let conversation = talos
        .messages
        .iter()
        .map(|message| message.text())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !conversation.contains("## Step 1 of 3"),
        "an underfunded plan must be driven flat: {conversation}"
    );
}

/// A plan step that leaves the tree unparseable is rolled back before the
/// next step starts, so later work is not built on broken source.
#[tokio::test]
async fn a_broken_plan_step_is_rewound() {
    let h = Harness::new("passing");
    let original = std::fs::read_to_string(h.root.join("src/lib.rs")).unwrap();
    let mut talos = h.talos(
        vec![
            tool_call(
                "1",
                "write_file",
                serde_json::json!({
                    "path": "src/lib.rs",
                    "content": "pub fn a( {{{ ~~~ not rust"
                }),
            ),
            text_response("done"),
            // After a failed interim verdict the drive keeps going until
            // stuck (2 noops) or the per-step ceiling. A redirect consumes
            // one more engine turn (Metis) and then two more noops, so
            // pad generously.
            text_response("still done"),
            text_response("still done"),
            text_response("still done"),
            text_response("ok"),
            text_response("ok"),
            text_response("ok"),
            text_response("ok"),
            text_response("ok"),
            text_response("ok"),
            text_response("ok"),
            text_response("ok"),
        ],
        12,
        false,
    );

    let plan = Plan {
        steps: vec!["break the crate".into(), "leave it".into()],
    };
    talos.run("break then stop", &plan).await.unwrap();

    let now = std::fs::read_to_string(h.root.join("src/lib.rs")).unwrap();
    assert_eq!(now, original, "the broken edit must have been rolled back");
}

// --------------------------------------------------------------- 70-point loop

fn valid_write(id: &str) -> knossos::engine::Response {
    tool_call(
        id,
        "write_file",
        serde_json::json!({
            "path": "src/added.rs",
            "content": "pub fn extra() -> u32 { 1 }\n"
        }),
    )
}

#[tokio::test]
async fn the_first_stuck_redirects_instead_of_halting() {
    let h = Harness::new("passing");
    let outcome = h
        .run(
            vec![
                text_response("Done."),
                text_response("Done."),
                valid_write("w"),
                text_response("done for real"),
            ],
            8,
        )
        .await;

    let redirected: Vec<_> = h
        .trace_events()
        .into_iter()
        .filter(|e| e["event"] == "redirected")
        .collect();
    assert_eq!(redirected.len(), 1, "exactly one redirect: {redirected:?}");
    assert_eq!(outcome.halt, Halt::Done, "{}", outcome.summary);
}

#[tokio::test]
async fn a_second_stuck_still_halts() {
    let h = Harness::new("broken");
    let outcome = h.run(vec![text_response("Done."); 6], 8).await;
    assert_eq!(outcome.halt, Halt::Stuck, "{}", outcome.summary);
    let redirected = h
        .trace_events()
        .iter()
        .filter(|e| e["event"] == "redirected")
        .count();
    assert_eq!(redirected, 1);
}

#[tokio::test]
async fn a_failed_hypothesis_survives_process_restart() {
    let h = Harness::new("passing");
    let first = h
        .run(
            (0..12).map(|i| doomed_edit(&i.to_string(), "a")).collect(),
            12,
        )
        .await;
    assert_eq!(first.halt, Halt::Stuck);

    let mut talos = h.talos(vec![text_response("Done."); 6], 6, false);
    talos
        .run(
            "edit something",
            &Plan {
                steps: vec!["edit".into()],
            },
        )
        .await
        .unwrap();

    let conversation = talos
        .messages
        .iter()
        .map(|m| m.text())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        conversation.contains("Previously failed"),
        "run B must see run A's hypothesis:\n{conversation}"
    );
    assert!(
        conversation.contains("edit_file"),
        "the failed call must be named:\n{conversation}"
    );
}

#[tokio::test]
async fn verify_mid_loop_does_not_complete_the_run() {
    let h = Harness::new("passing");
    let outcome = h
        .run(
            vec![
                tool_call("1", "verify", serde_json::json!({})),
                valid_write("2"),
                text_response("done"),
            ],
            6,
        )
        .await;
    assert_eq!(outcome.halt, Halt::Done, "{}", outcome.summary);
    let verifies = h
        .trace_events()
        .iter()
        .filter(|e| e["event"] == "tool_call" && e["tool"] == "verify")
        .count();
    assert_eq!(verifies, 1);
}

#[tokio::test]
async fn verify_report_reaches_the_engine() {
    let h = Harness::new("passing");
    let mut talos = h.talos(
        vec![
            tool_call("1", "verify", serde_json::json!({"full": true})),
            valid_write("2"),
            text_response("done"),
        ],
        6,
        false,
    );
    talos
        .run(
            "check then add",
            &Plan {
                steps: vec!["do".into()],
            },
        )
        .await
        .unwrap();

    let conversation = format!("{:?}", talos.messages);
    assert!(
        conversation.contains("Verification") || conversation.contains("passed"),
        "Oracle's report must land in a tool result: {conversation}"
    );
}

#[tokio::test]
async fn dry_run_verify_uses_staged_contents() {
    let h = Harness::new("passing");
    let mut talos = h.talos(
        vec![
            tool_call(
                "1",
                "write_file",
                serde_json::json!({
                    "path": "src/lib.rs",
                    "content": "pub fn a( {{{"
                }),
            ),
            tool_call("2", "verify", serde_json::json!({})),
            text_response("previewed"),
            text_response("previewed"),
            text_response("previewed"),
            text_response("previewed"),
        ],
        8,
        true,
    );
    let outcome = talos
        .run(
            "preview a break",
            &Plan {
                steps: vec!["preview".into()],
            },
        )
        .await
        .unwrap();
    assert!(outcome.dry_run);
    let conversation = format!("{:?}", talos.messages);
    assert!(
        conversation.contains("preview")
            || conversation.contains("Syntax")
            || conversation.contains("syntax"),
        "staged verify must not pretend cargo ran:\n{conversation}"
    );
}

#[tokio::test]
async fn compaction_still_carries_a_failed_hypothesis() {
    let h = Harness::new("passing");
    std::fs::write(h.root.join("src/huge.rs"), "x".repeat(300_000)).unwrap();
    let mut talos = h.talos(
        vec![
            doomed_edit("1", "a"),
            tool_call("2", "read_file", serde_json::json!({"path": "src/huge.rs"})),
            text_response("Done."),
            text_response("Done."),
            text_response("Done."),
            text_response("Done."),
            text_response("Done."),
            text_response("Done."),
        ],
        10,
        false,
    );
    talos.lethe = knossos::lethe::Lethe {
        max_tokens: 4_000,
        ..Default::default()
    };
    talos
        .run(
            "edit then read",
            &Plan {
                steps: vec!["edit".into()],
            },
        )
        .await
        .unwrap();

    let conversation = talos
        .messages
        .iter()
        .map(|m| m.text())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        conversation.contains("Previously failed"),
        "compacted output must not erase the durable brief:\n{}",
        &conversation[..conversation.len().min(800)]
    );
}

#[tokio::test]
async fn forbidden_signatures_are_in_the_redirect_prompt() {
    let h = Harness::new("passing");
    let mut talos = h.talos(
        (0..12).map(|i| doomed_edit(&i.to_string(), "a")).collect(),
        12,
        false,
    );
    talos
        .run(
            "edit something",
            &Plan {
                steps: vec!["edit".into()],
            },
        )
        .await
        .unwrap();

    let conversation = talos
        .messages
        .iter()
        .map(|m| m.text())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        conversation.contains("SUPERVISOR"),
        "redirect must speak:\n{conversation}"
    );
    assert!(
        conversation.contains("Already tried") || conversation.contains("edit_file"),
        "the failed signature must be named:\n{conversation}"
    );
}

#[tokio::test]
async fn plan_redirect_asks_metis_for_a_new_tail() {
    let h = Harness::new("passing");
    let mut talos = h.talos(
        vec![
            text_response("Done."),
            text_response("Done."),
            tool_call(
                "p",
                "submit_plan",
                serde_json::json!({"steps": ["try a different file"]}),
            ),
            text_response("still stuck"),
            text_response("still stuck"),
            valid_write("w"),
            text_response("step two done"),
            text_response("closing"),
            text_response("closing"),
        ],
        20,
        false,
    );
    let plan = Plan {
        steps: vec!["first approach".into(), "original second".into()],
    };
    talos.run("add extra()", &plan).await.unwrap();

    let dumped = talos
        .messages
        .iter()
        .map(|m| m.text())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        dumped.contains("try a different file") || dumped.contains("SUPERVISOR"),
        "replan or supervisor must land:\n{dumped}"
    );
}
