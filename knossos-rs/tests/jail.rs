//! The workspace jail, exercised by real children that try to get out.
//!
//! Linux and macOS only. Windows has no unprivileged filesystem confinement;
//! `confine::tests` covers that it is refused under `require` and reported
//! under `prefer`.
#![cfg(any(target_os = "linux", target_os = "macos"))]

use std::path::{Path, PathBuf};
use std::time::Duration;

use knossos::confine::{self, Policy};
use knossos::sandbox::{Finished, Sandbox};

const LIMIT: Duration = Duration::from_secs(120);

struct Fixture {
    _dir: tempfile::TempDir,
    root: PathBuf,
    ws: PathBuf,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let ws = root.join("ws");
    std::fs::create_dir(&ws).unwrap();
    Fixture {
        _dir: dir,
        root,
        ws,
    }
}

/// Confinement is demanded, so a host that cannot provide it fails the test
/// rather than passing it vacuously.
fn jailed() -> Sandbox {
    Sandbox::default().with_policy(Policy::Require)
}

/// Run `script` under `sh -c` with one positional argument, so paths never
/// pass through shell quoting.
async fn sh(sandbox: &Sandbox, ws: &Path, script: &str, arg: &Path) -> Finished {
    let arg = arg.to_string_lossy().into_owned();
    sandbox
        .run_bounded("sh", &["-c", script, "sh", arg.as_str()], ws, LIMIT)
        .await
        .expect("sh spawns")
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").expect("HOME is set"))
}

fn probe_name(tag: &str) -> String {
    format!("knossos-jail-probe-{tag}-{}", std::process::id())
}

/// A probe file that must not exist afterwards; removes it and fails if it does.
fn assert_never_written(path: &Path) {
    if path.exists() {
        let _ = std::fs::remove_file(path);
        panic!("{} was written from inside the jail", path.display());
    }
}

#[tokio::test]
async fn a_child_runs_confined_when_confinement_is_required() {
    let f = fixture();
    let out = sh(&jailed(), &f.ws, "true", &f.ws).await;
    assert!(out.success(), "{}", out.stderr);
    assert!(out.confinement.is_confined(), "{:?}", out.confinement);
}

#[tokio::test]
async fn the_system_stays_readable() {
    let f = fixture();
    let out = sh(&jailed(), &f.ws, "cat /etc/hosts > /dev/null", &f.ws).await;
    assert!(out.success(), "{}", out.stderr);
}

#[tokio::test]
async fn the_workspace_is_writable() {
    let f = fixture();
    let inside = f.ws.join("inside.txt");
    let out = sh(&jailed(), &f.ws, "echo hi > \"$1\"", &inside).await;
    assert!(out.success(), "{}", out.stderr);
    assert!(inside.exists());
}

#[tokio::test]
async fn a_sibling_of_the_workspace_is_not() {
    let f = fixture();
    let sibling = f.root.join("sibling");
    std::fs::create_dir(&sibling).unwrap();
    let probe = sibling.join("probe.txt");
    let out = sh(&jailed(), &f.ws, "echo x > \"$1\"", &probe).await;
    assert!(!out.success(), "a write next to the workspace succeeded");
    assert_never_written(&probe);
}

#[tokio::test]
async fn the_home_directory_is_closed_for_writing() {
    let f = fixture();
    let probe = home().join(probe_name("write"));
    let out = sh(&jailed(), &f.ws, "echo x > \"$1\"", &probe).await;
    assert!(!out.success(), "a write into $HOME succeeded");
    assert_never_written(&probe);
}

#[tokio::test]
async fn the_home_directory_is_closed_for_reading() {
    let f = fixture();
    let out = sh(&jailed(), &f.ws, "ls \"$1\" > /dev/null", &home()).await;
    assert!(!out.success(), "listing $HOME succeeded:\n{}", out.stdout);
}

#[tokio::test]
async fn a_symlink_out_of_the_workspace_does_not_reach_home() {
    let f = fixture();
    let escape = f.ws.join("escape");
    std::os::unix::fs::symlink(home(), &escape).unwrap();
    let name = probe_name("symlink");
    let out = sh(&jailed(), &f.ws, "echo x > \"$1\"", &escape.join(&name)).await;
    assert!(
        !out.success(),
        "a write through a symlink into $HOME succeeded"
    );
    assert_never_written(&home().join(&name));
}

#[tokio::test]
async fn the_per_run_temp_directory_is_the_one_the_child_sees() {
    let f = fixture();
    let out = sh(
        &jailed(),
        &f.ws,
        "echo x > \"$TMPDIR/probe\" && cat \"$TMPDIR/probe\"",
        &f.ws,
    )
    .await;
    assert!(out.success(), "{}", out.stderr);
    assert_eq!(out.stdout.trim(), "x");
}

/// The toolchain path list is complete if a build and a test run pass under
/// the jail. Skipped, loudly, where cargo is not on PATH.
#[tokio::test]
async fn cargo_test_passes_inside_the_jail() {
    let f = fixture();
    std::fs::write(
        f.ws.join("Cargo.toml"),
        "[package]\nname = \"jailed\"\nversion = \"0.0.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::create_dir(f.ws.join("src")).unwrap();
    std::fs::write(
        f.ws.join("src/lib.rs"),
        "#[test]\nfn it_works() {\n    assert_eq!(2 + 2, 4);\n}\n",
    )
    .unwrap();
    let out = match jailed()
        .run_bounded("cargo", &["test", "--offline", "-q"], &f.ws, LIMIT)
        .await
    {
        Ok(out) => out,
        Err(e) => {
            eprintln!("skipped: cargo is not runnable here ({e})");
            return;
        }
    };
    assert!(
        out.success() && !out.timed_out,
        "cargo test failed inside the jail\n--- stdout\n{}\n--- stderr\n{}",
        out.stdout,
        out.stderr
    );
}

#[tokio::test]
async fn tcp_is_open_by_default_and_closed_on_request() {
    let f = fixture();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port().to_string();
    let code = "import socket, sys\nsocket.create_connection(('127.0.0.1', int(sys.argv[1])), timeout=5)\n";

    let open = match jailed()
        .run_bounded("python3", &["-c", code, port.as_str()], &f.ws, LIMIT)
        .await
    {
        Ok(out) => out,
        Err(e) => {
            eprintln!("skipped: python3 is not runnable here ({e})");
            return;
        }
    };
    assert!(
        open.success(),
        "loopback connect must work by default\n{}",
        open.stderr
    );

    let closed = jailed()
        .deny_network()
        .run_bounded("python3", &["-c", code, port.as_str()], &f.ws, LIMIT)
        .await
        .unwrap();
    if closed.confinement.network_denied() {
        assert!(
            !closed.success(),
            "connect succeeded although the jail reports tcp denied"
        );
    } else {
        eprintln!(
            "skipped: this host cannot deny tcp ({})",
            closed.confinement.describe()
        );
        assert!(
            confine::residual_risk()
                .iter()
                .any(|line| line.contains("network denial was requested")),
            "an unhonoured network request must show in residual risk"
        );
    }
}

#[tokio::test]
async fn off_runs_environment_only_and_the_result_says_so() {
    let f = fixture();
    let out = sh(
        &Sandbox::default().with_policy(Policy::Off),
        &f.ws,
        "true",
        &f.ws,
    )
    .await;
    assert!(out.success());
    assert!(!out.confinement.is_confined());
    assert!(
        confine::residual_risk()
            .iter()
            .any(|line| line.contains("without filesystem confinement")),
        "{:?}",
        confine::residual_risk()
    );
}
