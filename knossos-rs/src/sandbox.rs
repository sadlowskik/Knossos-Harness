//! What a command inherits once it is allowed to run.
//!
//! [`shell`](crate::tools::shell) answers *which* programs may run. That
//! question has a ceiling: `cargo test` is on the allowlist, and `cargo test`
//! compiles and executes whatever is in the workspace. The allowlist therefore
//! admits a program that runs agent-authored code by design, and no amount of
//! tightening it changes that.
//!
//! What is left to control is the environment that code lands in. A child
//! process inherits its parent's entire environment by default, and this
//! harness reads `ANTHROPIC_API_KEY` out of exactly that environment — see
//! [`Config::load`](crate::config::Config). A test that prints
//! `std::env::var("ANTHROPIC_API_KEY")` puts the key in the command's stdout,
//! which `shell` folds into the transcript and sends to the model provider on
//! the next turn. That is a credential leak reachable without escaping the path
//! jail or defeating the allowlist, using only tools the agent is supposed to
//! have.
//!
//! So the child gets an explicitly named environment rather than an inherited
//! one. Anything not named does not exist as far as the subprocess is
//! concerned.
//!
//! # What this does not do
//!
//! This is environment isolation, not process isolation. A sandboxed child can
//! still:
//!
//! - write anywhere the user account can write — the path jail in
//!   [`ToolCtx::resolve`](crate::tools::ToolCtx::resolve) binds the harness's
//!   own tools, not a subprocess those tools start;
//! - open network sockets, unless it is cargo itself honouring
//!   `CARGO_NET_OFFLINE`;
//! - outlive a timeout, because `kill_on_drop` reaps the direct child while the
//!   test binaries `cargo test` spawned are separately parented.
//!
//! Closing those needs OS-level containment — a job object on Windows, a
//! process group plus namespaces on Linux — and a platform crate this workspace
//! does not depend on. Left undone deliberately rather than approximated, so
//! that the guarantee this module *does* make stays believable.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use tokio::io::AsyncReadExt;

#[cfg(windows)]
struct WindowsJob(usize);

#[cfg(windows)]
impl WindowsJob {
    fn assign(child: &tokio::process::Child) -> Option<Self> {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
            SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        };

        unsafe {
            let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if handle.is_null() {
                return None;
            }
            let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let configured = SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                (&raw const info).cast(),
                std::mem::size_of_val(&info) as u32,
            ) != 0;
            let Some(process) = child.raw_handle() else {
                CloseHandle(handle);
                return None;
            };
            let assigned = configured && AssignProcessToJobObject(handle, process.cast()) != 0;
            if !assigned {
                CloseHandle(handle);
                return None;
            }
            Some(Self(handle as usize))
        }
    }

    fn terminate(&self) {
        unsafe {
            windows_sys::Win32::System::JobObjects::TerminateJobObject(self.0 as _, 1);
        }
    }
}

#[cfg(windows)]
impl Drop for WindowsJob {
    fn drop(&mut self) {
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.0 as _);
        }
    }
}

/// Variables without which nothing runs at all.
const BASE: &[&str] = &["PATH"];

/// Toolchain families admitted wholesale, because rustup and cargo route a
/// great deal of ordinary configuration through them (`CARGO_BUILD_JOBS`,
/// `RUSTUP_TOOLCHAIN`, `RUST_BACKTRACE`) and enumerating each one would break
/// on the next release.
const PREFIXES: &[&str] = &["CARGO_", "RUSTUP_", "RUST_"];

/// Toolchain variables that carry no prefix.
const TOOLCHAIN: &[&str] = &["RUSTC", "RUSTDOC", "RUSTFLAGS", "TERM"];

/// Platform variables the compiler and linker need to function.
#[cfg(windows)]
const PLATFORM: &[&str] = &[
    // Win32 itself fails in odd ways without these two.
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    // rustup and cargo live under the user profile.
    "USERPROFILE",
    "LOCALAPPDATA",
    "APPDATA",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROCESSOR_ARCHITECTURE",
    "NUMBER_OF_PROCESSORS",
    // The MSVC linker reads its search paths from the environment; without
    // these a `cargo build` on the msvc toolchain fails at the link step.
    "LIB",
    "INCLUDE",
    "VCINSTALLDIR",
    "VCTOOLSINSTALLDIR",
    "WINDOWSSDKDIR",
    "WINDOWSSDKVERSION",
    "UNIVERSALCRTSDKDIR",
    "UCRTVERSION",
];

#[cfg(not(windows))]
const PLATFORM: &[&str] = &["HOME", "TMPDIR", "LANG", "LC_ALL", "TERM"];

/// Substrings that disqualify a name even when a rule above would admit it.
///
/// This exists because [`PREFIXES`] is generous and one of the things it would
/// otherwise admit is `CARGO_REGISTRY_TOKEN`, a crates.io publish credential.
/// A prefix rule wide enough to be maintainable is wide enough to leak, so the
/// two rules are layered rather than merged.
const SECRET_MARKERS: &[&str] = &[
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "APIKEY",
    "_KEY",
];

/// What a bounded run produced.
#[derive(Debug, Clone)]
pub struct Finished {
    /// `None` when the process could not be reaped — in practice, when it was
    /// killed for running too long.
    pub status: Option<std::process::ExitStatus>,
    /// Whether the deadline was reached. Distinct from a non-zero exit: a
    /// command that failed said something, and one that hung did not.
    pub timed_out: bool,
    pub stdout: String,
    pub stderr: String,
}

impl Finished {
    pub fn success(&self) -> bool {
        !self.timed_out && self.status.map(|s| s.success()).unwrap_or(false)
    }

    /// The exit code, or -1 where there was none (a signal, or a kill).
    pub fn code(&self) -> i32 {
        self.status.and_then(|s| s.code()).unwrap_or(-1)
    }
}

/// Kill a process and everything it started.
///
/// Shells out to the platform's own tool rather than taking a dependency on
/// `windows-sys` or `libc` for two calls. The correct Windows mechanism is a
/// job object — it kills the tree atomically and cannot be outrun — and this is
/// not that; it is a best effort that asks the OS to walk the tree for us. The
/// honest summary is that a process which re-parents itself deliberately can
/// still escape. Nothing in a Rust toolchain does.
async fn kill_tree(pid: u32) {
    let mut killer;

    #[cfg(windows)]
    {
        // /T takes the children with it, /F does not ask twice.
        killer = tokio::process::Command::new("taskkill");
        killer.args(["/F", "/T", "/PID", &pid.to_string()]);
    }

    #[cfg(not(windows))]
    {
        // A negative pid addresses the process group, which `command` arranged
        // for this child to lead.
        killer = tokio::process::Command::new("kill");
        killer.args(["-KILL", &format!("-{pid}")]);
    }

    killer
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);

    // Bounded: a cleanup step that can itself hang has moved the problem
    // rather than solved it.
    let _ = tokio::time::timeout(Duration::from_secs(5), killer.status()).await;
}

/// The environment policy applied to every command `shell` runs.
#[derive(Debug, Clone)]
pub struct Sandbox {
    /// Names the operator added, for toolchains this module did not anticipate.
    extra: Vec<String>,
    /// Whether cargo may reach the network.
    offline: bool,
}

impl Default for Sandbox {
    /// Offline, because a run that silently fetches a new dependency has
    /// changed the build in a way the diff does not show.
    fn default() -> Self {
        Sandbox {
            extra: Vec::new(),
            offline: true,
        }
    }
}

impl Sandbox {
    /// Admit one further variable by exact name.
    ///
    /// The secret check still applies: this widens the allowlist, it does not
    /// override the layer beneath it.
    pub fn allow(mut self, name: impl Into<String>) -> Self {
        self.extra.push(name.into().to_ascii_uppercase());
        self
    }

    /// Let cargo reach the network. Needed to add a dependency; otherwise off.
    pub fn networked(mut self) -> Self {
        self.offline = false;
        self
    }

    pub fn is_offline(&self) -> bool {
        self.offline
    }

    /// Whether a variable of this name reaches the child.
    pub fn admits(&self, name: &str) -> bool {
        let upper = name.to_ascii_uppercase();

        // Checked first so that no rule below can be used to reach a secret.
        if SECRET_MARKERS.iter().any(|m| upper.contains(m)) {
            return false;
        }

        BASE.contains(&upper.as_str())
            || PLATFORM.contains(&upper.as_str())
            || TOOLCHAIN.contains(&upper.as_str())
            || PREFIXES.iter().any(|p| upper.starts_with(p))
            || self.extra.contains(&upper)
    }

    /// The environment a child receives, given the one this process holds.
    ///
    /// Takes the source environment as an argument rather than reading the
    /// real one, so the policy can be tested against a constructed environment
    /// containing secrets that are not actually present on the machine.
    pub fn env_for<I, K, V>(&self, source: I) -> BTreeMap<String, String>
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let mut out: BTreeMap<String, String> = source
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .filter(|(k, _)| self.admits(k))
            .collect();

        // Colour codes are noise in a transcript the model has to read, and
        // they cost tokens on every line of compiler output.
        out.insert("CARGO_TERM_COLOR".into(), "never".into());

        if self.offline {
            out.insert("CARGO_NET_OFFLINE".into(), "true".into());
        }

        out
    }

    /// Replace the command's environment with the sandboxed one.
    ///
    /// Prefer [`Sandbox::command`], which applies this along with the other
    /// settings every child in this harness needs. This stays public for a
    /// caller building a process some other way.
    pub fn apply(&self, cmd: &mut tokio::process::Command) {
        cmd.env_clear();
        cmd.envs(self.env_for(std::env::vars()));
    }

    /// Build a child process under this policy, rooted at `cwd`.
    ///
    /// The only place the harness constructs a subprocess. Both callers — the
    /// `run` tool and the Oracle's verification ladder — spawn cargo against
    /// the workspace and need the same three things, and two call sites each
    /// remembering three settings is two call sites that can drift. One
    /// constructor cannot.
    ///
    /// Beyond the environment, those settings are:
    ///
    /// * **No stdin.** There is nobody to type at it. A child that blocks
    ///   reading stdin would hang the step until the timeout, which then
    ///   reports it as slow rather than stuck.
    /// * **Killed on drop.** Without it a timeout bounds the *wait* and not the
    ///   process: dropping the future leaves the child running, so a hanging
    ///   `cargo test` is reported as timed out while it keeps its `target/`
    ///   lock and its CPU. Ariadne then grants another step, and one wedged
    ///   tree accumulates an orphan per step, all outliving the harness.
    pub fn command(&self, program: &str, cwd: &Path) -> tokio::process::Command {
        let mut cmd = tokio::process::Command::new(program);
        cmd.current_dir(cwd).stdin(Stdio::null()).kill_on_drop(true);
        self.apply(&mut cmd);

        // On Unix the child leads its own process group, so a timeout has a
        // group to signal. Windows needs no equivalent: `taskkill /T` walks the
        // parent-child chain instead, and putting the child in a new *console*
        // group there would only stop Ctrl+C reaching it, which is not the
        // problem being solved.
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.as_std_mut().process_group(0);
        }

        cmd
    }

    /// Run a command and collect its output, killing the whole process tree if
    /// it outlives `limit`.
    ///
    /// `kill_on_drop` is not enough on its own, and both call sites had already
    /// written down why: it reaps the process that was started, while `cargo
    /// test` is a *launcher*. The test binaries it spawns are separate
    /// processes, they survive their parent, and they keep the `target/` lock
    /// and the CPU. Ariadne then grants another step, so one wedged tree
    /// accumulates an orphan per step, all outliving the harness.
    ///
    /// The output pipes are drained concurrently rather than one after the
    /// other. Reading stdout to EOF first deadlocks the moment a chatty child
    /// fills the stderr pipe, because it then blocks writing to a pipe nobody
    /// is reading — a hang that looks exactly like the one the timeout exists
    /// to catch, and would be mistaken for it.
    ///
    /// Waiting on the pipes rather than on exit is deliberate: they close when
    /// every holder exits, so a process that forks a daemon and returns still
    /// counts as running. That is the case worth catching.
    pub async fn run_bounded<S: AsRef<OsStr>>(
        &self,
        program: &str,
        args: &[S],
        cwd: &Path,
        limit: Duration,
    ) -> std::io::Result<Finished> {
        let mut cmd = self.command(program, cwd);
        cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::piped());

        let mut child = cmd.spawn()?;
        let pid = child.id();
        #[cfg(windows)]
        let job = WindowsJob::assign(&child);
        let mut out = child.stdout.take().expect("stdout was piped");
        let mut err = child.stderr.take().expect("stderr was piped");

        let mut stdout = Vec::new();
        let mut stderr = Vec::new();

        let drained = tokio::time::timeout(limit, async {
            tokio::join!(out.read_to_end(&mut stdout), err.read_to_end(&mut stderr))
        })
        .await;

        let timed_out = drained.is_err();
        if timed_out {
            // Before reaping the direct child: once it is gone the children it
            // launched are orphans, and on Windows `taskkill /T` finds them by
            // asking who their parent is.
            #[cfg(windows)]
            if let Some(job) = &job {
                job.terminate();
            } else if let Some(pid) = pid {
                kill_tree(pid).await;
            }
            #[cfg(not(windows))]
            if let Some(pid) = pid {
                kill_tree(pid).await;
            }
            // `taskkill /T` is best effort. Always signal the direct child as
            // a fallback, then bound reaping too: a deadline that can spend the
            // child's remaining 30 seconds in `wait` is not a deadline.
            let _ = child.start_kill();
        }

        // Whatever was read before the deadline is kept. Partial compiler
        // output from a run that hung is more use to the agent than nothing.
        let status = if timed_out {
            tokio::time::timeout(Duration::from_secs(2), child.wait())
                .await
                .ok()
                .and_then(Result::ok)
        } else {
            child.wait().await.ok()
        };

        Ok(Finished {
            status,
            timed_out,
            stdout: String::from_utf8_lossy(&stdout).into_owned(),
            stderr: String::from_utf8_lossy(&stderr).into_owned(),
        })
    }

    /// The names that would be dropped from the current environment.
    ///
    /// For explaining the policy — `daedalus` has no way to show its own
    /// guardrails otherwise, and a sandbox nobody can inspect is one nobody
    /// trusts.
    pub fn withheld(&self) -> Vec<String> {
        let mut names: Vec<String> = std::env::vars()
            .map(|(k, _)| k)
            .filter(|k| !self.admits(k))
            .collect();
        names.sort();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An environment shaped like a developer machine that has been used for
    /// real work: a toolchain, and a pile of credentials.
    fn realistic() -> Vec<(&'static str, &'static str)> {
        vec![
            ("PATH", "/usr/bin"),
            ("CARGO_HOME", "/home/k/.cargo"),
            ("RUSTUP_TOOLCHAIN", "stable"),
            ("RUST_BACKTRACE", "1"),
            ("ANTHROPIC_API_KEY", "sk-ant-should-never-appear"),
            ("OPENAI_API_KEY", "sk-should-never-appear"),
            ("AWS_SECRET_ACCESS_KEY", "should-never-appear"),
            ("GITHUB_TOKEN", "ghp_should-never-appear"),
            ("CARGO_REGISTRY_TOKEN", "cio-should-never-appear"),
            ("DATABASE_PASSWORD", "hunter2"),
            ("SSH_AUTH_SOCK", "/tmp/ssh-agent"),
        ]
    }

    #[test]
    fn the_harnesss_own_api_key_does_not_reach_the_child() {
        // The leak this module exists for: config.rs reads this variable, and
        // `cargo test` runs code the agent wrote.
        let env = Sandbox::default().env_for(realistic());
        assert!(!env.contains_key("ANTHROPIC_API_KEY"));
    }

    #[test]
    fn no_value_marked_secret_survives() {
        let env = Sandbox::default().env_for(realistic());
        for (name, value) in &env {
            assert!(
                !value.contains("should-never-appear"),
                "{name} leaked a credential into the child environment"
            );
        }
    }

    #[test]
    fn a_prefix_rule_cannot_be_used_to_reach_a_token() {
        // CARGO_REGISTRY_TOKEN matches the CARGO_ prefix and is a publish
        // credential; the secret layer has to win.
        let s = Sandbox::default();
        assert!(s.admits("CARGO_BUILD_JOBS"));
        assert!(!s.admits("CARGO_REGISTRY_TOKEN"));
    }

    #[test]
    fn the_toolchain_still_gets_what_it_needs() {
        let env = Sandbox::default().env_for(realistic());
        assert_eq!(env.get("PATH").map(String::as_str), Some("/usr/bin"));
        assert_eq!(
            env.get("CARGO_HOME").map(String::as_str),
            Some("/home/k/.cargo")
        );
        assert_eq!(
            env.get("RUSTUP_TOOLCHAIN").map(String::as_str),
            Some("stable")
        );
        assert_eq!(env.get("RUST_BACKTRACE").map(String::as_str), Some("1"));
    }

    #[test]
    fn unrecognized_variables_are_dropped_rather_than_passed() {
        // The default is deny: a name nobody thought about does not get through
        // just because it looks harmless.
        let s = Sandbox::default();
        assert!(!s.admits("SSH_AUTH_SOCK"));
        assert!(!s.admits("KUBECONFIG"));
        assert!(!s.admits("SOME_INTERNAL_ENDPOINT"));
    }

    #[test]
    fn an_operator_can_widen_the_list_but_not_past_the_secret_check() {
        let s = Sandbox::default().allow("KUBECONFIG").allow("MY_TOKEN");
        assert!(s.admits("KUBECONFIG"));
        assert!(
            !s.admits("MY_TOKEN"),
            "allow() must not override the secret layer"
        );
    }

    #[test]
    fn matching_is_case_insensitive() {
        // Windows environment names are case-insensitive, and `Path` is how it
        // is actually spelled there.
        let s = Sandbox::default();
        assert!(s.admits("Path"));
        assert!(!s.admits("anthropic_api_key"));
    }

    #[test]
    fn offline_is_the_default_and_can_be_turned_off() {
        let offline = Sandbox::default().env_for(realistic());
        assert_eq!(
            offline.get("CARGO_NET_OFFLINE").map(String::as_str),
            Some("true")
        );

        let networked = Sandbox::default().networked().env_for(realistic());
        assert!(!networked.contains_key("CARGO_NET_OFFLINE"));
    }

    #[test]
    fn colour_is_forced_off_regardless_of_the_parent() {
        let env = Sandbox::default().env_for(vec![("CARGO_TERM_COLOR", "always")]);
        assert_eq!(
            env.get("CARGO_TERM_COLOR").map(String::as_str),
            Some("never")
        );
    }

    /// The single spawn point has to carry the policy, or consolidating on it
    /// would have quietly removed the scrubbing from both call sites.
    #[tokio::test]
    async fn command_builds_a_child_that_is_scrubbed_and_rooted() {
        std::env::set_var("KNOSSOS_COMMAND_CANARY_KEY", "canary-must-not-appear");
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();

        let (program, args): (&str, &[&str]) = if cfg!(windows) {
            ("cmd", &["/c", "set"])
        } else {
            ("env", &[])
        };

        let mut cmd = Sandbox::default().command(program, &root);
        cmd.args(args);
        let out = cmd.output().await.expect("could not run the probe");
        let text = String::from_utf8_lossy(&out.stdout);

        assert!(
            text.contains("PATH=") || text.contains("Path="),
            "the probe produced nothing, so this proves nothing: {text}"
        );
        assert!(
            !text.contains("canary-must-not-appear"),
            "command() must scrub"
        );

        std::env::remove_var("KNOSSOS_COMMAND_CANARY_KEY");
    }

    #[tokio::test]
    async fn command_runs_in_the_directory_it_was_given() {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();

        let (program, args): (&str, &[&str]) = if cfg!(windows) {
            ("cmd", &["/c", "cd"])
        } else {
            ("pwd", &[])
        };

        let mut cmd = Sandbox::default().command(program, &root);
        cmd.args(args);
        let out = cmd.output().await.expect("could not run the probe");
        let printed = String::from_utf8_lossy(&out.stdout).trim().to_string();

        assert_eq!(
            std::fs::canonicalize(&printed).unwrap(),
            root,
            "a child rooted outside the workspace would escape the jail entirely"
        );
    }

    /// A program that finishes immediately, printing `word`.
    fn echoing(word: &str) -> (&str, Vec<String>) {
        if cfg!(windows) {
            ("cmd", vec!["/c".into(), "echo".into(), word.into()])
        } else {
            ("echo", vec![word.into()])
        }
    }

    /// A program that runs for roughly `seconds` and produces nothing.
    fn sleeping(seconds: u32) -> (&'static str, Vec<String>) {
        if cfg!(windows) {
            // `timeout` needs a console; ping against loopback does not, and
            // waits about one second per count.
            (
                "cmd",
                vec![
                    "/c".into(),
                    "ping".into(),
                    "-n".into(),
                    (seconds + 1).to_string(),
                    "127.0.0.1".into(),
                ],
            )
        } else {
            ("sleep", vec![seconds.to_string()])
        }
    }

    #[tokio::test]
    async fn a_command_that_finishes_reports_its_output_and_code() {
        let dir = tempfile::tempdir().unwrap();
        let (program, args) = echoing("hello");

        let f = Sandbox::default()
            .run_bounded(program, &args, dir.path(), Duration::from_secs(30))
            .await
            .unwrap();

        assert!(!f.timed_out);
        assert!(f.success(), "stderr: {}", f.stderr);
        assert_eq!(f.code(), 0);
        assert!(f.stdout.contains("hello"), "stdout was {:?}", f.stdout);
    }

    #[tokio::test]
    async fn a_command_that_outstays_its_limit_is_reported_as_timed_out() {
        let dir = tempfile::tempdir().unwrap();
        let (program, args) = sleeping(30);

        let started = std::time::Instant::now();
        let f = Sandbox::default()
            .run_bounded(program, &args, dir.path(), Duration::from_millis(700))
            .await
            .unwrap();

        assert!(f.timed_out, "should have hit the deadline");
        assert!(!f.success(), "a timeout is not a success");
        // The point of the deadline: it returns near it, not near the
        // program's own runtime.
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "waited {:?}, which means the deadline did not bound anything",
            started.elapsed()
        );
    }

    /// The tests above check the policy; this one checks that it is actually
    /// attached to a process. `env_for` could be perfect and `apply` could
    /// still forget to call `env_clear`, in which case every real command
    /// inherits everything and nothing above would notice.
    #[tokio::test]
    async fn apply_clears_the_environment_of_a_real_child() {
        std::env::set_var("KNOSSOS_SANDBOX_CANARY_KEY", "canary-must-not-appear");

        // A program that prints its own environment. `set` is a cmd builtin.
        let (program, args): (&str, &[&str]) = if cfg!(windows) {
            ("cmd", &["/c", "set"])
        } else {
            ("env", &[])
        };

        let mut cmd = tokio::process::Command::new(program);
        cmd.args(args);
        Sandbox::default().apply(&mut cmd);
        let out = cmd
            .output()
            .await
            .expect("could not run the environment probe");
        let text = String::from_utf8_lossy(&out.stdout);

        // Guards against passing vacuously: if the probe printed nothing, the
        // absence of the canary would prove nothing at all.
        assert!(
            text.contains("PATH=") || text.contains("Path="),
            "the probe produced no environment listing, so this test proves nothing: {text}"
        );
        assert!(
            !text.contains("canary-must-not-appear"),
            "apply() did not clear the inherited environment"
        );

        std::env::remove_var("KNOSSOS_SANDBOX_CANARY_KEY");
    }
}
