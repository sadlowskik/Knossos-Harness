//! End-to-end tests for the NDJSON server loop.
//!
//! `serve.rs` previously had tests for command and event *serialisation* only —
//! nothing drove the loop. That is the gap this file closes, and it had to close
//! before the permission gate went in rather than after: a handshake that waits
//! for a reply on a stream nobody is reading is a deadlock, and a deadlock in an
//! untested async loop is one that first appears in somebody's editor.
//!
//! Every test here therefore runs under a timeout. A hang is a *failure*, not a
//! test suite that never finishes.

use std::path::{Path, PathBuf};
use std::time::Duration;

use knossos::ariadne::Ariadne;
use knossos::engine::mock::{text_response, tool_call, MockEngine};
use knossos::oracle::Oracle;
use knossos::scribe::SymbolIndex;
use knossos::serve::{Emitter, Event};
use knossos::session::Session;
use knossos::talos::Talos;
use knossos::themis::Themis;
use knossos::tools::{ToolCtx, ToolRegistry};

/// Nothing here should take anywhere near this. It exists so a deadlock reports
/// as a failed assertion rather than a suite that hangs forever.
const LIMIT: Duration = Duration::from_secs(20);

fn fixture(name: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let src = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    copy_tree(&src, dir.path()).unwrap();
    dir
}

fn copy_tree(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

/// A running server plus the two ends a front end would hold.
struct Server {
    lines: tokio::sync::mpsc::UnboundedSender<String>,
    events: tokio::sync::mpsc::UnboundedReceiver<Event>,
    handle: tokio::task::JoinHandle<anyhow::Result<()>>,
    _dir: tempfile::TempDir,
    root: PathBuf,
}

/// The turn Metis spends before Talos gets one.
///
/// `Command::Task` plans first, so the *first* scripted response is consumed by
/// the planner and never reaches the executor. Forgetting this makes a test look
/// like it exercised the loop when the engine actually ran out of script — and
/// a test that drains the wrong events and then waits forever reads as a
/// deadlock, which on this particular file is the worst possible false alarm.
fn plan_turn() -> knossos::engine::Response {
    tool_call(
        "plan",
        "submit_plan",
        serde_json::json!({ "steps": ["do the thing"] }),
    )
}

impl Server {
    /// `scripted` is what the *executor* sees; the planner's turn is prepended.
    fn start(scripted: Vec<knossos::engine::Response>, dry: bool) -> Server {
        let mut script = vec![plan_turn()];
        script.extend(scripted);
        let scripted = script;
        let dir = fixture("passing");
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let trace = root.join("trace.jsonl");

        let mut ctx = ToolCtx::new(&root);
        if dry {
            ctx = ctx.dry_run();
        }
        let talos = Talos::new(
            Box::new(MockEngine::new(scripted)),
            ToolRegistry::standard(),
            ctx,
            // See the note in `harness_loop.rs`: the baseline is measured in
            // `oracle`'s own tests, not paid for on every loop test here.
            Oracle::new(&root).without_baseline(),
            SymbolIndex::build(&root).unwrap(),
            Themis::from_text("Be correct."),
            Ariadne::new(4, 3),
            Session::new(&root, "mock").with_trace(&trace).unwrap(),
            1024,
            false,
        );

        let (line_tx, line_rx) = tokio::sync::mpsc::unbounded_channel();
        let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = tokio::spawn(knossos::serve::run(
            talos,
            1024,
            line_rx,
            Emitter::new(event_tx),
        ));

        Server {
            lines: line_tx,
            events: event_rx,
            handle,
            _dir: dir,
            root,
        }
    }

    fn send(&self, raw: &str) {
        self.lines.send(format!("{raw}\n")).unwrap();
    }

    /// Declare that this front end can answer permission requests.
    ///
    /// Gating is opt-in, so a test that forgets this is testing an *ungated*
    /// server and will pass whatever it asserts about prompts not appearing.
    fn declare_permissions(&self) {
        self.send(r#"{"cmd":"capabilities","permissions":true}"#);
    }

    /// Next event, or a failed assertion if the server has gone quiet.
    async fn next(&mut self) -> Event {
        match tokio::time::timeout(LIMIT, self.events.recv()).await {
            Ok(Some(e)) => e,
            Ok(None) => panic!("the server closed its event channel"),
            Err(_) => panic!("timed out waiting for an event — the loop is stuck"),
        }
    }

    /// Drain until a predicate matches, collecting what went past.
    async fn until<F>(&mut self, mut matches: F) -> (Event, Vec<Event>)
    where
        F: FnMut(&Event) -> bool,
    {
        let mut seen = Vec::new();
        loop {
            let event = self.next().await;
            if matches(&event) {
                return (event, seen);
            }
            seen.push(event);
        }
    }
}

fn is_idle(e: &Event) -> bool {
    matches!(e, Event::Idle)
}

/// Routed, not dispatched — the same property permission replies need.
///
/// The dispatch loop is inside `talos.run` for the whole turn, so an
/// interjection sent during a task can only arrive if the router handles it
/// without touching that loop. If it were an ordinary command it would sit in
/// the queue until the task finished, by which point saying it was pointless.
#[tokio::test]
async fn a_word_typed_during_a_task_reaches_the_run_without_stopping_it() {
    let mut server = Server::start(
        vec![
            tool_call(
                "1",
                "write_file",
                serde_json::json!({
                    "path": "src/added.rs",
                    "content": "pub fn triple(n: i32) -> i32 { n * 3 }\n"
                }),
            ),
            text_response("done"),
        ],
        false,
    );

    assert!(matches!(server.next().await, Event::Ready { .. }));
    assert!(matches!(server.next().await, Event::Idle));

    server.send(r#"{"cmd":"task","text":"add a helper"}"#);
    server.send(r#"{"cmd":"interject","text":"call it quadruple instead"}"#);

    let (_, seen) = server.until(is_idle).await;
    assert!(
        !seen.iter().any(|e| matches!(e, Event::Error { .. })),
        "the interjection produced an error event"
    );

    let trace = std::fs::read_to_string(server.root.join("trace.jsonl")).unwrap();
    assert!(
        trace.contains("interjected"),
        "the run never saw it:\n{trace}"
    );
    assert!(trace.contains("call it quadruple instead"), "{trace}");
}

#[tokio::test]
async fn an_interjection_that_cannot_be_queued_is_reported_not_swallowed() {
    // Dropping it silently would leave the person believing the agent had been
    // told something it will never read.
    let mut server = Server::start(vec![text_response("idle")], false);

    assert!(matches!(server.next().await, Event::Ready { .. }));
    assert!(matches!(server.next().await, Event::Idle));

    server.send(r#"{"cmd":"interject","text":"   "}"#);

    match server.next().await {
        Event::Error { message } => {
            assert!(message.contains("not accepted"), "{message}")
        }
        _ => panic!("an unqueueable interjection must produce an error event"),
    }
}

#[tokio::test]
async fn a_turns_work_can_be_put_back_after_it_has_been_written() {
    // Not a dry run: staging already covers the previewed case. This is the
    // one the journal exists for — the write reached disk and the only way
    // back is the recorded `before`.
    let mut server = Server::start(
        vec![
            tool_call(
                "1",
                "write_file",
                serde_json::json!({
                    "path": "src/lib.rs",
                    "content": "// the agent replaced everything\n"
                }),
            ),
            text_response("done"),
        ],
        false,
    );

    assert!(matches!(server.next().await, Event::Ready { .. }));
    assert!(matches!(server.next().await, Event::Idle));

    let lib = server.root.join("src/lib.rs");
    let original = std::fs::read_to_string(&lib).unwrap();

    server.send(r#"{"cmd":"task","text":"rewrite the lib"}"#);
    server.until(is_idle).await;
    assert_ne!(
        std::fs::read_to_string(&lib).unwrap(),
        original,
        "the turn should have changed the file"
    );

    server.send(r#"{"cmd":"undo"}"#);
    let (event, _) = server.until(|e| matches!(e, Event::Undone { .. })).await;
    match event {
        Event::Undone { files } => assert!(!files.is_empty(), "something was put back"),
        _ => unreachable!(),
    }

    assert_eq!(
        std::fs::read_to_string(&lib).unwrap(),
        original,
        "undo must restore the exact prior content"
    );
}

#[tokio::test]
async fn undoing_before_any_turn_is_an_error_not_a_crash() {
    let mut server = Server::start(vec![text_response("idle")], false);

    assert!(matches!(server.next().await, Event::Ready { .. }));
    assert!(matches!(server.next().await, Event::Idle));

    server.send(r#"{"cmd":"undo"}"#);
    match server.next().await {
        Event::Error { message } => assert!(message.contains("no checkpoint"), "{message}"),
        _ => panic!("undo with nothing to undo should report an error"),
    }
}

#[tokio::test]
async fn the_loop_answers_a_command_and_returns_to_idle() {
    // The baseline the rest of the file depends on: the protocol works when
    // driven over channels rather than the process's real streams.
    let mut server = Server::start(vec![text_response("nothing to do")], false);

    assert!(matches!(server.next().await, Event::Ready { .. }));
    assert!(matches!(server.next().await, Event::Idle));

    server.send(r#"{"cmd":"index"}"#);
    let (_, seen) = server.until(is_idle).await;

    assert!(
        seen.iter().any(|e| matches!(e, Event::Index { .. })),
        "{seen:?}"
    );
}

#[tokio::test]
async fn a_bad_command_is_reported_and_the_loop_survives() {
    let mut server = Server::start(vec![text_response("fine")], false);
    server.until(is_idle).await;

    server.send("not json at all");
    let (_, seen) = server.until(is_idle).await;
    assert!(
        seen.iter().any(|e| matches!(e, Event::Error { .. })),
        "{seen:?}"
    );

    // Still answering afterwards.
    server.send(r#"{"cmd":"index"}"#);
    let (_, seen) = server.until(is_idle).await;
    assert!(
        seen.iter().any(|e| matches!(e, Event::Index { .. })),
        "{seen:?}"
    );
}

#[tokio::test]
async fn a_consequential_call_asks_before_it_writes() {
    let mut server = Server::start(
        vec![
            tool_call(
                "1",
                "write_file",
                serde_json::json!({"path": "src/added.rs", "content": "pub fn f() {}\n"}),
            ),
            text_response("Added it."),
        ],
        false,
    );
    server.until(is_idle).await;
    server.declare_permissions();
    server.send(r#"{"cmd":"task","text":"add a file"}"#);

    // The request must arrive *before* the file exists.
    let (request, _) = server
        .until(|e| matches!(e, Event::PermissionRequest { .. }))
        .await;
    let Event::PermissionRequest { id, tool, .. } = request else {
        unreachable!()
    };
    assert_eq!(tool, "write_file");
    assert!(
        !server.root.join("src/added.rs").exists(),
        "the write happened before anyone was asked"
    );

    server.send(&format!(r#"{{"cmd":"permission","id":{id},"allow":true}}"#));
    server.until(is_idle).await;

    assert!(
        server.root.join("src/added.rs").exists(),
        "approval did not let it through"
    );
}

#[tokio::test]
async fn refusing_a_call_prevents_it_and_the_run_continues() {
    // A refusal is a result, not an error. The engine sees it and gets another
    // turn; ending the run would discard the conversation over a decision the
    // user is entitled to make.
    let mut server = Server::start(
        vec![
            tool_call(
                "1",
                "write_file",
                serde_json::json!({"path": "src/nope.rs", "content": "pub fn f() {}\n"}),
            ),
            // Enough turns to reach the ceiling: a refused call leaves the task
            // undone, so the engine keeps being asked. Running the script out
            // would surface as an Error event and read like a protocol fault.
            text_response("Understood, I will not write it."),
            text_response("Still not writing it."),
            text_response("Nothing further."),
            text_response("Nothing further."),
        ],
        false,
    );
    server.until(is_idle).await;
    server.declare_permissions();
    server.send(r#"{"cmd":"task","text":"add a file"}"#);

    let (request, _) = server
        .until(|e| matches!(e, Event::PermissionRequest { .. }))
        .await;
    let Event::PermissionRequest { id, .. } = request else {
        unreachable!()
    };
    server.send(&format!(
        r#"{{"cmd":"permission","id":{id},"allow":false}}"#
    ));

    let (_, seen) = server.until(is_idle).await;

    assert!(
        !server.root.join("src/nope.rs").exists(),
        "a refused write still happened"
    );
    assert!(
        seen.iter().any(|e| matches!(e, Event::Outcome { .. })),
        "the run should have finished rather than aborting: {seen:?}"
    );
}

#[tokio::test]
async fn a_read_is_never_put_to_the_user() {
    // Prompting for every read trains people to approve without looking, which
    // is worse than not asking. `Tool::consequential` is the discriminator.
    let mut server = Server::start(
        vec![
            tool_call("1", "read_file", serde_json::json!({"path": "src/lib.rs"})),
            text_response("Read it."),
        ],
        false,
    );
    server.until(is_idle).await;
    server.declare_permissions();
    server.send(r#"{"cmd":"task","text":"read the file"}"#);

    let (_, seen) = server.until(is_idle).await;

    assert!(
        !seen
            .iter()
            .any(|e| matches!(e, Event::PermissionRequest { .. })),
        "a read should not have been put to the user: {seen:?}"
    );
}

#[tokio::test]
async fn the_loop_keeps_reading_while_a_permission_is_outstanding() {
    // The property the whole reader/dispatch split exists for. If the loop only
    // read stdin between commands, the reply would queue behind the very command
    // waiting for it and neither would ever complete.
    //
    // A plain `permission` reply proves routing; this also proves the router is
    // alive and consuming *while* dispatch is blocked, by sending junk first and
    // seeing the reply still land.
    let mut server = Server::start(
        vec![
            tool_call(
                "1",
                "write_file",
                serde_json::json!({"path": "src/added.rs", "content": "pub fn f() {}\n"}),
            ),
            text_response("Done."),
        ],
        false,
    );
    server.until(is_idle).await;
    server.declare_permissions();
    server.send(r#"{"cmd":"task","text":"add a file"}"#);

    let (request, _) = server
        .until(|e| matches!(e, Event::PermissionRequest { .. }))
        .await;
    let Event::PermissionRequest { id, .. } = request else {
        unreachable!()
    };

    // Arrives while the dispatch loop is inside `talos.run`.
    server.send("   ");
    server.send(&format!(r#"{{"cmd":"permission","id":{id},"allow":true}}"#));

    let (_, seen) = server.until(is_idle).await;
    assert!(
        seen.iter().any(|e| matches!(e, Event::Outcome { .. })),
        "{seen:?}"
    );
}

#[tokio::test]
async fn a_reply_to_an_unknown_request_is_ignored() {
    let mut server = Server::start(vec![text_response("fine")], false);
    server.until(is_idle).await;

    // No Idle for this: it is routed, not dispatched, so it is not a command
    // that completes. The loop must simply carry on.
    server.send(r#"{"cmd":"permission","id":9999,"allow":true}"#);
    server.send(r#"{"cmd":"index"}"#);

    let (_, seen) = server.until(is_idle).await;
    assert!(
        seen.iter().any(|e| matches!(e, Event::Index { .. })),
        "{seen:?}"
    );
}

#[tokio::test]
async fn a_disconnected_front_end_denies_rather_than_hanging() {
    // The realistic failure. Untimed waiting is right for a user reading a diff,
    // but if the front end goes away the request must resolve — as a refusal,
    // never as an approval, and never as a hang.
    let mut server = Server::start(
        vec![
            tool_call(
                "1",
                "write_file",
                serde_json::json!({"path": "src/ghost.rs", "content": "pub fn f() {}\n"}),
            ),
            text_response("Could not write."),
            text_response("Nothing further."),
            text_response("Nothing further."),
            text_response("Nothing further."),
        ],
        false,
    );
    server.until(is_idle).await;
    server.declare_permissions();
    server.send(r#"{"cmd":"task","text":"add a file"}"#);
    server
        .until(|e| matches!(e, Event::PermissionRequest { .. }))
        .await;

    // The front end vanishes mid-request.
    drop(std::mem::replace(&mut server.lines, {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        tx
    }));

    let finished = tokio::time::timeout(LIMIT, server.handle).await;
    assert!(
        finished.is_ok(),
        "the server hung after the front end disconnected"
    );
    assert!(
        !server.root.join("src/ghost.rs").exists(),
        "a request nobody answered must not be treated as approval"
    );
}

#[tokio::test]
async fn a_front_end_that_never_declares_is_not_gated() {
    // The compatibility guarantee, and the reason gating is opt-in.
    //
    // The VS Code panel in this repository dispatches events by name and
    // silently ignores anything it does not recognise, so an unannounced
    // `permission_request` would be dropped and the agent would wait forever for
    // a reply nobody was going to send. Gating by default would have turned
    // every existing front end into a hang — a worse failure than the one the
    // gate prevents, introduced by the gate.
    let mut server = Server::start(
        vec![
            tool_call(
                "1",
                "write_file",
                serde_json::json!({"path": "src/added.rs", "content": "pub fn f() {}\n"}),
            ),
            text_response("Done."),
        ],
        false,
    );
    server.until(is_idle).await;
    // No `capabilities` line: an older front end.
    server.send(r#"{"cmd":"task","text":"add a file"}"#);

    let (_, seen) = server.until(is_idle).await;

    assert!(
        !seen
            .iter()
            .any(|e| matches!(e, Event::PermissionRequest { .. })),
        "an undeclared front end was asked something it cannot answer: {seen:?}"
    );
    assert!(
        server.root.join("src/added.rs").exists(),
        "the run did not complete"
    );
}

#[tokio::test]
async fn declaring_permissions_false_is_the_same_as_not_declaring() {
    let mut server = Server::start(
        vec![
            tool_call(
                "1",
                "write_file",
                serde_json::json!({"path": "src/added.rs", "content": "pub fn f() {}\n"}),
            ),
            text_response("Done."),
        ],
        false,
    );
    server.until(is_idle).await;
    server.send(r#"{"cmd":"capabilities","permissions":false}"#);
    server.send(r#"{"cmd":"task","text":"add a file"}"#);

    let (_, seen) = server.until(is_idle).await;

    assert!(
        !seen
            .iter()
            .any(|e| matches!(e, Event::PermissionRequest { .. })),
        "{seen:?}"
    );
}

#[tokio::test]
async fn shutdown_ends_the_loop() {
    let mut server = Server::start(vec![text_response("fine")], false);
    server.until(is_idle).await;

    server.send(r#"{"cmd":"shutdown"}"#);

    let finished = tokio::time::timeout(LIMIT, server.handle).await;
    assert!(finished.is_ok(), "shutdown did not end the loop");
}

#[tokio::test]
async fn every_dispatched_command_ends_with_exactly_one_idle() {
    // The documented contract the front end relies on to re-enable input. A
    // permission request is emitted mid-command and must *not* count as the end
    // of one.
    let mut server = Server::start(
        vec![
            tool_call(
                "1",
                "write_file",
                serde_json::json!({"path": "src/added.rs", "content": "pub fn f() {}\n"}),
            ),
            text_response("Done."),
        ],
        false,
    );
    server.until(is_idle).await; // the Ready/Idle preamble

    server.declare_permissions();
    server.send(r#"{"cmd":"task","text":"add a file"}"#);
    let (request, before) = server
        .until(|e| matches!(e, Event::PermissionRequest { .. }))
        .await;
    assert!(
        !before.iter().any(is_idle),
        "an Idle arrived before the command finished: {before:?}"
    );

    let Event::PermissionRequest { id, .. } = request else {
        unreachable!()
    };
    server.send(&format!(r#"{{"cmd":"permission","id":{id},"allow":true}}"#));

    let (_, after) = server.until(is_idle).await;
    assert!(
        !after.iter().any(is_idle),
        "more than one Idle for one command: {after:?}"
    );
}
