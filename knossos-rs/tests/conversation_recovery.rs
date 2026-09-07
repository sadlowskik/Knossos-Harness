//! Drive two real CLI processes through a persisted safe boundary. No provider
//! credentials or live engines are used; only recorded mock responses execute.
use std::path::Path;
use std::process::{Command, Output};

use knossos::engine::mock::{text_response, tool_call};
use knossos::engine::Response;
use knossos::mission::MissionStore;

fn copy_tree(source: &Path, target: &Path) {
    std::fs::create_dir_all(target).unwrap();
    for entry in std::fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name() == "target" {
            continue;
        }
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target.join(entry.file_name()));
        } else {
            std::fs::copy(entry.path(), target.join(entry.file_name())).unwrap();
        }
    }
}

fn script(path: &Path, responses: Vec<Response>) {
    let text = responses
        .into_iter()
        .map(|response| serde_json::json!({"event":"exchange","response":response}).to_string())
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(path, text).unwrap();
}

fn run(root: &Path, replay: &Path, mission: Option<&str>, task: &str, requests: &str) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_knossos"));
    command
        .args(["--engine", "none", "--workspace"])
        .arg(root)
        .args([
            "task",
            task,
            "--no-judge",
            "--no-context",
            "--no-memory",
            "--max-steps",
            "6",
            "--target-steps",
            "3",
            "--max-requests",
            requests,
            "--replay-responses",
        ])
        .arg(replay)
        .arg("--collect-exchanges")
        .arg("--persist-conversation");
    if let Some(mission) = mission {
        command.args(["--resume-mission", mission]);
    }
    command.output().unwrap()
}

fn success(output: &Output) {
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn cli_continuation_preserves_conversation_authority_and_quota_across_processes() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("workspace");
    copy_tree(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/passing"),
        &root,
    );
    let replay = directory.path().join("responses.jsonl");
    script(
        &replay,
        vec![
            tool_call(
                "plan",
                "submit_plan",
                serde_json::json!({"steps":["add the requested source"]}),
            ),
            tool_call(
                "original-write",
                "write_file",
                serde_json::json!({"path":"src/first.rs", "content":"pub const FIRST: u32 = 41;\n"}),
            ),
            text_response("Original checkpoint answer: FIRST is 41."),
        ],
    );
    success(&run(
        &root,
        &replay,
        None,
        "add a source file with FIRST set to 41",
        "10",
    ));
    let missions = root.join(".knossos/missions");
    let mission = std::fs::read_dir(&missions)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .file_name()
        .to_string_lossy()
        .into_owned();
    let before = MissionStore::open(&root, &mission).unwrap();
    let capsule = before.load_conversation().unwrap();
    let spent = capsule["quota"]["spent"]["requests"].as_u64().unwrap();
    assert!(spent >= 3);
    let original_modified = std::fs::metadata(root.join("src/first.rs"))
        .unwrap()
        .modified()
        .unwrap();

    // Policy changes fail before consuming the capsule or spawning work.
    let rejected = run(&root, &replay, Some(&mission), "continue", "100");
    assert!(!rejected.status.success());
    assert!(
        String::from_utf8_lossy(&rejected.stderr).contains("execution policy or provider differs")
    );
    // A second executor cannot claim the same mission.
    let lock = before.lock_execution().unwrap();
    let rejected = run(&root, &replay, Some(&mission), "continue", "10");
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("active executor"));
    drop(lock);

    // The resumed process skips Metis. Its first response is a new action,
    // and the old write remains evidence in the conversation, not replayed I/O.
    script(
        &replay,
        vec![
            tool_call(
                "followup-write",
                "write_file",
                serde_json::json!({"path":"src/second.rs", "content":"pub const SECOND: u32 = 42;\n"}),
            ),
            text_response("Added SECOND while retaining FIRST."),
        ],
    );
    success(&run(
        &root,
        &replay,
        Some(&mission),
        "add another source file with SECOND set to 42",
        "10",
    ));
    assert_eq!(
        std::fs::metadata(root.join("src/first.rs"))
            .unwrap()
            .modified()
            .unwrap(),
        original_modified
    );
    assert_eq!(
        std::fs::read_to_string(root.join("src/second.rs")).unwrap(),
        "pub const SECOND: u32 = 42;\n"
    );
    let after = MissionStore::open(&root, &mission).unwrap();
    assert_eq!(after.state().contract, before.state().contract);
    let resumed = after.load_conversation().unwrap();
    assert!(resumed["quota"]["spent"]["requests"].as_u64().unwrap() > spent);
    assert!(resumed["messages"]
        .to_string()
        .contains("Original checkpoint answer"));

    // Both corrupt bytes and a stale journal are rejected before restoration.
    let id = after
        .state()
        .identity
        .last_safe_checkpoint
        .as_ref()
        .unwrap();
    let file = missions.join(&mission).join(format!("{id}.json"));
    let bytes = std::fs::read(&file).unwrap();
    std::fs::write(&file, b"corrupt").unwrap();
    assert!(after
        .load_conversation()
        .unwrap_err()
        .to_string()
        .contains("hash mismatch"));
    std::fs::write(file, bytes).unwrap();
    let mut advanced = MissionStore::open(&root, &mission).unwrap();
    advanced
        .append(knossos::mission::MissionEvent::Checkpoint {
            id: "interrupted-new-work".into(),
            safe: false,
        })
        .unwrap();
    assert!(advanced
        .load_conversation()
        .unwrap_err()
        .to_string()
        .contains("advanced beyond"));
}
