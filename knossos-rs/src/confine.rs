//! What a child process can *reach*, as opposed to what it inherits.
//!
//! [`sandbox`](crate::sandbox) hands every child a scrubbed environment, so a
//! test that prints `ANTHROPIC_API_KEY` prints nothing. That closes the
//! credential leak and nothing else: `cargo test` still runs agent-authored
//! code with the operator's whole filesystem and network. A test that copies
//! `~/.ssh` into the workspace, or posts it somewhere, needs no escape from
//! the path jail because the path jail binds the harness's own tools, not the
//! programs those tools start.
//!
//! This module confines the child itself, using what the operating system
//! offers without privileges or a daemon:
//!
//! - **Linux: Landlock.** A ruleset is built in the parent and applied in the
//!   forked child just before `exec`, so every descendant inherits it and none
//!   can shed it. The child may read and execute the system roots and the
//!   toolchain homes, write the cargo home (cargo locks its package cache), a
//!   per-run temp directory and `/dev`, and do anything inside the workspace.
//!   Nothing else exists for it. Landlock resolves the real inode, so a
//!   symlink in the workspace that points at `$HOME` is refused too. With
//!   kernel ABI 4 or newer, TCP bind and connect can be denied as well.
//! - **macOS: Seatbelt.** The command is wrapped in `/usr/bin/sandbox-exec`
//!   with a profile that allows everything by default, then denies writes
//!   outside the same set of paths and reads under `$HOME` except the same
//!   toolchain homes. `sandbox-exec` execs the command in place, so the pid
//!   and the process group are still the child's. The interface is
//!   deprecated but shipped on current macOS and relied on by Chromium and
//!   Bazel; if it ever disappears the run degrades to environment-only and
//!   says so.
//! - **Windows: nothing.** There is no unprivileged filesystem confinement.
//!   The Job Object in `sandbox` still bounds the process tree. Runs are
//!   marked [`Confinement::EnvOnly`].
//!
//! [`Policy`] decides what happens when confinement is unavailable, and the
//! weakest level observed in this process is folded into the mission's
//! residual risk, so nobody mistakes an environment jail for a workspace one.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// What to do when the operating system cannot confine a child.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Policy {
    /// Refuse to spawn. What CI and the Cameo appliance run.
    Require,
    /// Spawn anyway and record the level actually applied.
    #[default]
    Prefer,
    /// Never confine. An operator debugging aid; the run says so.
    Off,
}

impl Policy {
    /// Environment variable that selects the policy: `require`, `prefer`, `off`.
    pub const ENV: &'static str = "KNOSSOS_CONFINE";

    pub fn parse(text: &str) -> Option<Policy> {
        match text.trim().to_ascii_lowercase().as_str() {
            "require" => Some(Policy::Require),
            "prefer" => Some(Policy::Prefer),
            "off" => Some(Policy::Off),
            _ => None,
        }
    }

    /// The policy named by [`Policy::ENV`], or `Prefer`.
    pub fn from_env() -> Policy {
        std::env::var(Self::ENV)
            .ok()
            .and_then(|v| Self::parse(&v))
            .unwrap_or_default()
    }

    pub fn name(self) -> &'static str {
        match self {
            Policy::Require => "require",
            Policy::Prefer => "prefer",
            Policy::Off => "off",
        }
    }
}

/// What an admitted path gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    ReadOnly,
    ReadWrite,
}

/// The confinement a child actually ran under.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Confinement {
    /// Linux Landlock at this kernel ABI. `network_denied` is true only when
    /// denial was requested and the kernel (ABI 4+) could honour it.
    Landlock { abi: i32, network_denied: bool },
    /// macOS Seatbelt through `sandbox-exec`.
    Seatbelt { network_denied: bool },
    /// Only the environment was controlled; the reason says why.
    EnvOnly { reason: String },
}

impl Confinement {
    pub fn is_confined(&self) -> bool {
        !matches!(self, Confinement::EnvOnly { .. })
    }

    pub fn network_denied(&self) -> bool {
        match self {
            Confinement::Landlock { network_denied, .. }
            | Confinement::Seatbelt { network_denied } => *network_denied,
            Confinement::EnvOnly { .. } => false,
        }
    }

    /// One line for logs, results and the `exec` banner.
    pub fn describe(&self) -> String {
        let net = |denied: bool| if denied { ", tcp denied" } else { "" };
        match self {
            Confinement::Landlock {
                abi,
                network_denied,
            } => format!("landlock abi {abi}{}", net(*network_denied)),
            Confinement::Seatbelt { network_denied } => {
                format!("seatbelt{}", net(*network_denied))
            }
            Confinement::EnvOnly { reason } => format!("environment only: {reason}"),
        }
    }
}

/// Paths outside the workspace that a confined child may read and execute.
/// Missing ones are dropped at build time, so the list can name every
/// distribution's layout at once.
#[cfg(target_os = "linux")]
const SYSTEM_READ: &[&str] = &[
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib32",
    "/lib64",
    "/etc",
    "/opt",
    "/proc",
    "/sys",
    "/run",
    "/var",
    "/nix",
    "/snap",
    "/home/linuxbrew",
];
/// macOS allows reads by default; only `$HOME` is closed, so nothing outside
/// it needs naming.
#[cfg(not(target_os = "linux"))]
const SYSTEM_READ: &[&str] = &[];

/// Paths outside the workspace a confined child may write. Device files only:
/// `/dev/null`, `/dev/urandom`, the pty.
const SYSTEM_WRITE: &[&str] = &["/dev"];

/// Toolchain homes under `$HOME` that stay readable. Everything else under
/// `$HOME` — SSH keys, shell history, other repositories — is closed.
const HOME_READ: &[&str] = &[".local", ".pyenv", ".nvm", ".volta", ".rye", ".bun"];

/// Caches under `$HOME` that tools insist on writing.
const HOME_WRITE: &[&str] = &[".cache", ".npm", "Library/Caches"];

/// What a confined child may reach. Built per spawn from the workspace, the
/// toolchain homes and any operator additions.
#[derive(Debug, Clone)]
pub struct Jail {
    pub workspace: PathBuf,
    /// Per-run temporary directory, handed to the child as `TMPDIR`.
    pub tmp: PathBuf,
    pub read_only: Vec<PathBuf>,
    pub read_write: Vec<PathBuf>,
    pub deny_network: bool,
}

impl Jail {
    /// Build the jail for `workspace`, creating the per-run temp directory.
    pub fn for_workspace(
        workspace: &Path,
        extra: &[(PathBuf, Access)],
        deny_network: bool,
    ) -> io::Result<Jail> {
        let workspace = workspace.canonicalize()?;
        let tmp = per_run_tmp()?;
        let home = std::env::var_os("HOME").map(PathBuf::from);

        let mut read_only: Vec<PathBuf> = SYSTEM_READ.iter().map(PathBuf::from).collect();
        let mut read_write: Vec<PathBuf> = SYSTEM_WRITE.iter().map(PathBuf::from).collect();

        if let Some(p) = home_dir(&home, "RUSTUP_HOME", ".rustup") {
            read_only.push(p);
        }
        if let Some(p) = home_dir(&home, "CARGO_HOME", ".cargo") {
            read_write.push(p);
        }
        if let Some(home) = &home {
            read_only.extend(HOME_READ.iter().map(|d| home.join(d)));
            read_write.extend(HOME_WRITE.iter().map(|d| home.join(d)));
        }
        // macOS puts the per-user temp directory under /var/folders and the
        // toolchain assumes it can write there regardless of TMPDIR.
        if cfg!(target_os = "macos") {
            read_write.push(std::env::temp_dir());
        }
        for (path, access) in extra {
            match access {
                Access::ReadOnly => read_only.push(path.clone()),
                Access::ReadWrite => read_write.push(path.clone()),
            }
        }
        read_write.push(tmp.clone());
        read_write.push(workspace.clone());

        let existing = |paths: Vec<PathBuf>| -> Vec<PathBuf> {
            let mut out: Vec<PathBuf> = paths
                .into_iter()
                .filter_map(|p| p.canonicalize().ok())
                .collect();
            out.sort();
            out.dedup();
            out
        };

        Ok(Jail {
            workspace,
            tmp,
            read_only: existing(read_only),
            read_write: existing(read_write),
            deny_network,
        })
    }
}

fn home_dir(home: &Option<PathBuf>, var: &str, default: &str) -> Option<PathBuf> {
    std::env::var_os(var)
        .map(PathBuf::from)
        .or_else(|| home.as_ref().map(|h| h.join(default)))
}

fn per_run_tmp() -> io::Result<PathBuf> {
    let dir = std::env::temp_dir().join(format!("knossos-{}", std::process::id()));
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        match std::fs::DirBuilder::new().mode(0o700).create(&dir) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e),
        }
    }
    #[cfg(not(unix))]
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Read extra admitted paths from `KNOSSOS_CONFINE_ALLOW_RO` and
/// `KNOSSOS_CONFINE_ALLOW_RW`, each a `PATH`-style list.
pub fn extra_paths_from_env() -> Vec<(PathBuf, Access)> {
    let mut out = Vec::new();
    for (var, access) in [
        ("KNOSSOS_CONFINE_ALLOW_RO", Access::ReadOnly),
        ("KNOSSOS_CONFINE_ALLOW_RW", Access::ReadWrite),
    ] {
        if let Some(list) = std::env::var_os(var) {
            out.extend(
                std::env::split_paths(&list)
                    .filter(|p| !p.as_os_str().is_empty())
                    .map(|p| (p, access)),
            );
        }
    }
    out
}

/// A command ready to spawn, and the confinement it will run under.
#[derive(Debug)]
pub struct Confined {
    pub command: tokio::process::Command,
    pub level: Confinement,
    /// The per-run temp directory, to hand the child as `TMPDIR`.
    pub tmp: Option<PathBuf>,
}

/// Build the command for `program`, confined to `workspace` as far as this
/// platform and `policy` allow.
///
/// `Err(Unsupported)` only under [`Policy::Require`] when the platform cannot
/// confine; every other failure to build the jail (a workspace that cannot be
/// canonicalized, a temp directory that cannot be created) is an ordinary
/// I/O error regardless of policy.
pub fn confined_command(
    policy: Policy,
    workspace: &Path,
    extra: &[(PathBuf, Access)],
    deny_network: bool,
    program: &str,
) -> io::Result<Confined> {
    if policy == Policy::Off {
        return Ok(Confined {
            command: tokio::process::Command::new(program),
            level: Confinement::EnvOnly {
                reason: format!("{}=off", Policy::ENV),
            },
            tmp: None,
        });
    }

    let jail = Jail::for_workspace(workspace, extra, deny_network)?;
    let tmp = Some(jail.tmp.clone());

    match platform::prepare(&jail, program) {
        Ok((command, level)) => Ok(Confined {
            command,
            level,
            tmp,
        }),
        Err(reason) if policy == Policy::Require => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "{}=require but this host cannot confine child processes: {reason}",
                Policy::ENV
            ),
        )),
        Err(reason) => Ok(Confined {
            command: tokio::process::Command::new(program),
            level: Confinement::EnvOnly { reason },
            tmp,
        }),
    }
}

// ---------------------------------------------------------------------------
// Observation: the weakest level any child in this process ran under.

static WEAKEST: Mutex<Option<Confinement>> = Mutex::new(None);
static NETWORK_UNCONFINED: Mutex<bool> = Mutex::new(false);

fn rank(level: &Confinement) -> u8 {
    match level {
        Confinement::EnvOnly { .. } => 0,
        Confinement::Landlock { .. } | Confinement::Seatbelt { .. } => 1,
    }
}

/// Record the level a child ran under. `wanted_network_denied` is what the
/// caller asked for, so a request the kernel could not honour is noted.
pub fn observe(level: &Confinement, wanted_network_denied: bool) {
    if let Ok(mut weakest) = WEAKEST.lock() {
        let replace = match &*weakest {
            None => true,
            Some(current) => rank(level) < rank(current),
        };
        if replace {
            *weakest = Some(level.clone());
        }
    }
    if wanted_network_denied && !level.network_denied() {
        if let Ok(mut flag) = NETWORK_UNCONFINED.lock() {
            *flag = true;
        }
    }
}

/// The weakest confinement any child of this process has run under.
pub fn weakest_observed() -> Option<Confinement> {
    WEAKEST.lock().ok().and_then(|w| w.clone())
}

/// Whether a child asked to be cut off from the network was not.
pub fn network_was_requested_but_open() -> bool {
    NETWORK_UNCONFINED.lock().map(|f| *f).unwrap_or(false)
}

/// Lines for the result object's residual risk, if the process ran anything
/// it could not confine.
pub fn residual_risk() -> Vec<String> {
    let mut out = Vec::new();
    if let Some(level) = weakest_observed() {
        if !level.is_confined() {
            out.push(format!(
                "child processes ran without filesystem confinement on {} ({})",
                std::env::consts::OS,
                level.describe()
            ));
        }
    }
    if network_was_requested_but_open() {
        out.push("network denial was requested but this host cannot enforce it".into());
    }
    out
}

// ---------------------------------------------------------------------------
// Platform back ends. Each returns the command to spawn or the reason it
// cannot confine.

#[cfg(target_os = "linux")]
mod platform {
    use super::{Confinement, Jail};
    use landlock::{
        path_beneath_rules, Access, AccessFs, AccessNet, CompatLevel, Compatible, Ruleset,
        RulesetAttr, RulesetCreated, RulesetCreatedAttr, ABI,
    };
    use std::io;
    use std::sync::Mutex;

    /// The highest Landlock ABI this kernel supports, or `None` when it has
    /// no Landlock at all (kernel < 5.13, or `landlock` missing from the LSM
    /// list). Asked of the kernel directly: `landlock_create_ruleset` with
    /// the version flag returns the ABI and creates nothing.
    fn kernel_abi() -> Option<i32> {
        const LANDLOCK_CREATE_RULESET_VERSION: libc::c_uint = 1;
        // SAFETY: a null attribute pointer with size 0 and the version flag
        // is the documented query form; it touches no memory.
        let n = unsafe {
            libc::syscall(
                libc::SYS_landlock_create_ruleset,
                std::ptr::null::<libc::c_void>(),
                0usize,
                LANDLOCK_CREATE_RULESET_VERSION,
            )
        };
        (n > 0).then_some(n as i32)
    }

    pub fn prepare(
        jail: &Jail,
        program: &str,
    ) -> Result<(tokio::process::Command, Confinement), String> {
        let abi_number = kernel_abi().ok_or_else(|| {
            "this kernel has no Landlock (needs 5.13+ with landlock in the LSM list)".to_string()
        })?;
        // The crate degrades a newer request to what the kernel has, but
        // asking for exactly the detected ABI keeps the rule set identical
        // from one spawn to the next on the same host.
        let abi = ABI::from(abi_number);
        let net_capable = jail.deny_network && abi_number >= 4;

        let ruleset = build(jail, abi).map_err(|e| format!("landlock ruleset: {e}"))?;

        let mut command = tokio::process::Command::new(program);
        attach(&mut command, ruleset);
        Ok((
            command,
            Confinement::Landlock {
                abi: abi_number,
                network_denied: net_capable,
            },
        ))
    }

    fn build(jail: &Jail, abi: ABI) -> Result<RulesetCreated, landlock::RulesetError> {
        let mut ruleset = Ruleset::default()
            .set_compatibility(CompatLevel::BestEffort)
            .handle_access(AccessFs::from_all(abi))?;
        if jail.deny_network {
            ruleset = ruleset.handle_access(AccessNet::from_all(ABI::V4))?;
        }
        ruleset
            .create()?
            .add_rules(path_beneath_rules(
                &jail.read_only,
                AccessFs::from_read(abi),
            ))?
            .add_rules(path_beneath_rules(
                &jail.read_write,
                AccessFs::from_all(abi),
            ))
    }

    fn attach(command: &mut tokio::process::Command, ruleset: RulesetCreated) {
        use std::os::unix::process::CommandExt;
        let cell = Mutex::new(Some(ruleset));
        // SAFETY: the hook runs in the forked child, single-threaded, before
        // exec. It takes an uncontended lock, moves the ruleset out and makes
        // two syscalls (prctl no_new_privs, landlock_restrict_self). Nothing
        // on the success path allocates or touches state another thread of
        // the parent could hold.
        unsafe {
            command.as_std_mut().pre_exec(move || {
                let ruleset = cell
                    .lock()
                    .ok()
                    .and_then(|mut slot| slot.take())
                    .ok_or_else(|| io::Error::other("landlock ruleset already applied"))?;
                ruleset
                    .restrict_self()
                    .map(|_| ())
                    .map_err(io::Error::other)
            });
        }
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::{Confinement, Jail};
    use std::path::{Path, PathBuf};

    const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";

    pub fn prepare(
        jail: &Jail,
        program: &str,
    ) -> Result<(tokio::process::Command, Confinement), String> {
        if !Path::new(SANDBOX_EXEC).is_file() {
            return Err(format!("{SANDBOX_EXEC} is not present on this macOS"));
        }
        let profile = profile(jail);
        let mut command = tokio::process::Command::new(SANDBOX_EXEC);
        command.arg("-p").arg(profile).arg(program);
        Ok((
            command,
            Confinement::Seatbelt {
                network_denied: jail.deny_network,
            },
        ))
    }

    fn quote(path: &Path) -> String {
        let text = path
            .to_string_lossy()
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        format!("\"{text}\"")
    }

    fn subpaths(paths: &[PathBuf]) -> String {
        paths
            .iter()
            .map(|p| format!(" (subpath {})", quote(p)))
            .collect()
    }

    /// Seatbelt profile. Later rules win, so: allow everything, deny all
    /// writes, re-allow writes on the admitted paths, close `$HOME` for
    /// reading, re-open the toolchain homes and the workspace if it lives
    /// there. An `allow` with no filter would allow everything, so each
    /// re-allow is emitted only with at least one path.
    pub(super) fn profile(jail: &Jail) -> String {
        let mut out = String::from("(version 1)\n(allow default)\n(deny file-write*)\n");
        if !jail.read_write.is_empty() {
            out.push_str(&format!(
                "(allow file-write*{})\n",
                subpaths(&jail.read_write)
            ));
        }
        if let Some(home) = std::env::var_os("HOME")
            .map(PathBuf::from)
            .and_then(|h| h.canonicalize().ok())
        {
            out.push_str(&format!(
                "(deny file-read* (subpath {h}))\n(allow file-read-metadata (literal {h}))\n",
                h = quote(&home)
            ));
            let reopened: Vec<PathBuf> = jail
                .read_only
                .iter()
                .chain(jail.read_write.iter())
                .filter(|p| p.starts_with(&home))
                .cloned()
                .collect();
            if !reopened.is_empty() {
                out.push_str(&format!("(allow file-read*{})\n", subpaths(&reopened)));
            }
        }
        if jail.deny_network {
            out.push_str("(deny network*)\n");
        }
        out
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod platform {
    use super::{Confinement, Jail};

    pub fn prepare(
        _jail: &Jail,
        _program: &str,
    ) -> Result<(tokio::process::Command, Confinement), String> {
        Err(format!(
            "no unprivileged filesystem confinement on {}",
            std::env::consts::OS
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_parses_its_three_words_and_nothing_else() {
        assert_eq!(Policy::parse(" Require "), Some(Policy::Require));
        assert_eq!(Policy::parse("prefer"), Some(Policy::Prefer));
        assert_eq!(Policy::parse("OFF"), Some(Policy::Off));
        assert_eq!(Policy::parse("yes"), None);
        assert_eq!(Policy::default(), Policy::Prefer);
    }

    #[test]
    fn env_only_is_never_confined_and_says_why() {
        let level = Confinement::EnvOnly {
            reason: "test".into(),
        };
        assert!(!level.is_confined());
        assert!(!level.network_denied());
        assert!(level.describe().contains("test"));
        assert!(Confinement::Landlock {
            abi: 5,
            network_denied: true
        }
        .network_denied());
    }

    #[test]
    fn off_yields_an_unconfined_command_without_touching_the_disk() {
        let dir = tempfile::tempdir().unwrap();
        let confined = confined_command(Policy::Off, dir.path(), &[], false, "true").unwrap();
        assert!(!confined.level.is_confined());
        assert!(confined.tmp.is_none());
    }

    #[test]
    fn the_jail_admits_the_workspace_and_a_temp_directory_it_created() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().canonicalize().unwrap();
        let jail = Jail::for_workspace(&ws, &[], false).unwrap();
        assert!(jail.tmp.is_dir());
        assert!(jail.read_write.contains(&ws));
        assert!(jail.read_write.contains(&jail.tmp.canonicalize().unwrap()));
    }

    #[test]
    fn an_operator_path_lands_in_the_list_it_asked_for() {
        let dir = tempfile::tempdir().unwrap();
        let extra_ro = dir.path().join("ro");
        let extra_rw = dir.path().join("rw");
        std::fs::create_dir_all(&extra_ro).unwrap();
        std::fs::create_dir_all(&extra_rw).unwrap();
        let jail = Jail::for_workspace(
            dir.path(),
            &[
                (extra_ro.clone(), Access::ReadOnly),
                (extra_rw.clone(), Access::ReadWrite),
            ],
            false,
        )
        .unwrap();
        assert!(jail.read_only.contains(&extra_ro.canonicalize().unwrap()));
        assert!(jail.read_write.contains(&extra_rw.canonicalize().unwrap()));
    }

    #[test]
    fn a_missing_path_is_dropped_rather_than_failing_the_jail() {
        let dir = tempfile::tempdir().unwrap();
        let ghost = dir.path().join("does-not-exist");
        let jail =
            Jail::for_workspace(dir.path(), &[(ghost.clone(), Access::ReadOnly)], false).unwrap();
        assert!(!jail.read_only.iter().any(|p| p.ends_with("does-not-exist")));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn the_seatbelt_profile_never_emits_an_unfiltered_allow() {
        let dir = tempfile::tempdir().unwrap();
        let jail = Jail::for_workspace(dir.path(), &[], true).unwrap();
        let profile = platform::profile(&jail);
        assert!(profile.contains("(deny file-write*)"));
        assert!(profile.contains("(deny network*)"));
        for line in profile.lines() {
            if line.starts_with("(allow file-") {
                assert!(
                    line.contains("(subpath ") || line.contains("(literal "),
                    "{line}"
                );
            }
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    #[test]
    fn require_refuses_where_the_platform_cannot_confine() {
        let dir = tempfile::tempdir().unwrap();
        let err = confined_command(Policy::Require, dir.path(), &[], false, "cmd").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
        let ok = confined_command(Policy::Prefer, dir.path(), &[], false, "cmd").unwrap();
        assert!(!ok.level.is_confined());
    }
}
