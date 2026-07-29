//! The sandbox has to leave a toolchain that still works.
//!
//! `sandbox.rs`'s unit tests run against a constructed environment, which is
//! the right way to assert that a credential is dropped but says nothing about
//! whether the survivors are sufficient. An allowlist that is one variable too
//! narrow does not fail loudly — it fails as a linker error in the middle of an
//! eval run, on whichever machine happens to need the variable nobody listed.
//!
//! So this compiles an actual crate through the sandboxed environment.

use std::path::Path;
use std::time::Duration;

use knossos::sandbox::Sandbox;

fn write_crate(dir: &Path) {
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        // No dependencies: offline is the sandbox default, and this test is
        // about the environment, not about cargo's network behaviour.
        "[package]\nname = \"sandbox-probe\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n[dependencies]\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("src").join("lib.rs"),
        "pub fn answer() -> u32 { 42 }\n",
    )
    .unwrap();
}

#[tokio::test]
async fn a_real_crate_still_compiles_under_the_sandbox() {
    let tmp = tempfile::tempdir().unwrap();
    write_crate(tmp.path());

    let mut cmd = tokio::process::Command::new("cargo");
    cmd.arg("check").arg("--quiet").current_dir(tmp.path());
    Sandbox::default().apply(&mut cmd);

    let out = cmd.output().await.expect("cargo is not on PATH");

    assert!(
        out.status.success(),
        "the allowlist is too narrow to build with on this machine.\n\
         stderr:\n{}\n\nwithheld: {:?}",
        String::from_utf8_lossy(&out.stderr),
        Sandbox::default().withheld(),
    );
}

/// A grandchild must not outlive the process that was killed for hanging.
///
/// This is the failure both `shell.rs` and the Oracle documented and neither
/// could fix: `kill_on_drop` reaps the process that was started, and `cargo
/// test` is a launcher whose children are separate processes. They kept the
/// `target/` lock and the CPU, Ariadne granted another step, and one wedged
/// tree accumulated an orphan per step.
///
/// Windows only, because it is written with `cmd`. The mechanism under test
/// differs by platform anyway — `taskkill /T` here, a process-group signal
/// elsewhere — so a single portable test would not have covered both.
#[cfg(windows)]
#[tokio::test]
async fn a_grandchild_does_not_outlive_the_process_that_was_killed() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    // Waits, then leaves proof it was still running.
    std::fs::write(
        dir.join("grandchild.bat"),
        "@echo off\r\nping -n 5 127.0.0.1 >nul\r\necho survived> marker.txt\r\n",
    )
    .unwrap();
    // One level of indirection, so the thing that writes the marker is a
    // *grandchild* of the process `run_bounded` starts and kills.
    std::fs::write(
        dir.join("parent.bat"),
        "@echo off\r\ncmd /c grandchild.bat\r\n",
    )
    .unwrap();

    let marker = dir.join("marker.txt");
    let sandbox = Sandbox::default();

    // Control. Without this the real assertion below could pass because the
    // batch file never worked, rather than because the kill worked.
    let control = sandbox
        .run_bounded("cmd", &["/c", "grandchild.bat"], dir, Duration::from_secs(60))
        .await
        .unwrap();
    assert!(!control.timed_out, "the control run should finish on its own");
    assert!(marker.exists(), "the control run must produce the marker");
    std::fs::remove_file(&marker).unwrap();

    // The real case: kill the parent well before the grandchild is done.
    let finished = sandbox
        .run_bounded("cmd", &["/c", "parent.bat"], dir, Duration::from_millis(700))
        .await
        .unwrap();
    assert!(finished.timed_out, "should have hit the deadline");

    // Long enough that a surviving grandchild would certainly have written it.
    tokio::time::sleep(Duration::from_secs(8)).await;
    assert!(
        !marker.exists(),
        "the grandchild outlived the process that was killed"
    );
}

#[tokio::test]
async fn a_test_binary_cannot_read_the_harnesss_api_key() {
    // The end-to-end version of the leak: agent-authored code, compiled and
    // executed by an allowlisted command, trying to read the credential the
    // harness itself uses. `cargo test` printing it would put it in stdout,
    // which shell.rs folds into the transcript.
    std::env::set_var("ANTHROPIC_API_KEY", "sk-ant-integration-canary");

    let tmp = tempfile::tempdir().unwrap();
    write_crate(tmp.path());
    std::fs::write(
        tmp.path().join("src").join("lib.rs"),
        r#"
#[test]
fn exfiltrate() {
    println!("KEY={:?}", std::env::var("ANTHROPIC_API_KEY"));
}
"#,
    )
    .unwrap();

    let mut cmd = tokio::process::Command::new("cargo");
    cmd.arg("test").arg("--").arg("--nocapture").current_dir(tmp.path());
    Sandbox::default().apply(&mut cmd);

    let out = cmd.output().await.expect("cargo is not on PATH");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    std::env::remove_var("ANTHROPIC_API_KEY");

    assert!(
        text.contains("KEY="),
        "the probe test did not run, so this proves nothing:\n{text}"
    );
    assert!(
        !text.contains("sk-ant-integration-canary"),
        "the API key reached agent-authored code:\n{text}"
    );
}
