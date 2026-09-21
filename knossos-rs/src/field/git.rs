//! Real Git state per workspace. Port of the read side of
//! `field/server/src/watch/git.js`: `readStatus`, `diffFile` and `log`.
//!
//! `git` runs as a process with explicit arguments, never through a shell,
//! with hooks, fsmonitor, credential helpers and terminal prompts disabled
//! and the same minimal environment every Field child gets.

use super::child_env::{build_child_environment, process_environment};
use regex::Regex;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

/// Run `git` in `cwd`; `Err` carries the message Node's `execFile` produced.
pub fn git(cwd: &Path, args: &[&str]) -> Result<String, String> {
    let hooks = if cfg!(windows) { "NUL" } else { "/dev/null" };
    let hooks_path = format!("core.hooksPath={hooks}");
    let mut safe_args: Vec<&str> = vec![
        "-c",
        &hooks_path,
        "-c",
        "core.fsmonitor=false",
        "-c",
        "credential.helper=",
    ];
    safe_args.extend_from_slice(args);
    let overrides: BTreeMap<String, String> = [
        ("GIT_OPTIONAL_LOCKS".to_string(), "0".to_string()),
        ("GIT_TERMINAL_PROMPT".to_string(), "0".to_string()),
    ]
    .into_iter()
    .collect();
    let env = build_child_environment(&process_environment(), None, &[], &overrides);
    let mut command = Command::new("git");
    command
        .args(&safe_args)
        .current_dir(cwd)
        .env_clear()
        .envs(env)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    hide_window(&mut command);
    let output = command.output().map_err(|e| {
        let code = match e.kind() {
            std::io::ErrorKind::NotFound => "ENOENT".to_string(),
            std::io::ErrorKind::PermissionDenied => "EACCES".to_string(),
            other => format!("{other:?}"),
        };
        format!("spawn git {code}")
    })?;
    if !output.status.success() {
        return Err(format!(
            "Command failed: git {}\n{}",
            safe_args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(windows)]
fn hide_window(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn hide_window(_command: &mut Command) {}

fn status_label(code: char) -> &'static str {
    match code {
        'M' => "modified",
        'A' => "added",
        'D' => "deleted",
        'R' => "renamed",
        'C' => "copied",
        'U' => "conflicted",
        '?' => "untracked",
        '!' => "ignored",
        _ => "changed",
    }
}

/// `readCleanRevision`: the HEAD revision and branch of a workspace with no
/// uncommitted changes; a dirty tree is an error naming a few of the files.
pub fn read_clean_revision(cwd: &Path) -> Result<(String, Value), String> {
    let status = read_status(cwd)?;
    let files: Vec<String> = status
        .get("files")
        .and_then(Value::as_array)
        .map(|files| {
            files
                .iter()
                .filter_map(|f| f.get("path").and_then(Value::as_str))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    if !files.is_empty() {
        let sample = files.iter().take(5).cloned().collect::<Vec<_>>().join(", ");
        let remainder = if files.len() > 5 {
            format!(" and {} more", files.len() - 5)
        } else {
            String::new()
        };
        return Err(format!(
            "workspace has uncommitted changes ({sample}{remainder})"
        ));
    }
    let revision = git(cwd, &["rev-parse", "--verify", "HEAD"])?
        .trim()
        .to_string();
    if !(40..=64).contains(&revision.len()) || !revision.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err("workspace HEAD is not a valid Git revision".into());
    }
    Ok((
        revision,
        status.get("branch").cloned().unwrap_or(Value::Null),
    ))
}

/// `readStatus`: branch, ahead/behind, and the changed files, scoped to the
/// workspace subtree and reported relative to it.
pub fn read_status(cwd: &Path) -> Result<Value, String> {
    let prefix = git(cwd, &["rev-parse", "--show-prefix"])
        .map(|out| out.trim().replace('\\', "/"))
        .unwrap_or_default();
    let out = git(cwd, &["status", "--porcelain=v1", "-b", "--", "."])?;
    Ok(parse_status(&out, &prefix))
}

fn parse_status(out: &str, prefix: &str) -> Value {
    static AHEAD: OnceLock<Regex> = OnceLock::new();
    static BEHIND: OnceLock<Regex> = OnceLock::new();
    let ahead_re = AHEAD.get_or_init(|| Regex::new(r"ahead (\d+)").expect("static regex"));
    let behind_re = BEHIND.get_or_init(|| Regex::new(r"behind (\d+)").expect("static regex"));
    let mut branch = Value::Null;
    let mut ahead = 0u64;
    let mut behind = 0u64;
    let mut files = Vec::new();
    for line in out.split('\n').filter(|l| !l.is_empty()) {
        if let Some(head) = line.strip_prefix("## ") {
            branch = json!(head.split("...").next().unwrap_or("").trim());
            if let Some(m) = ahead_re.captures(head) {
                ahead = m[1].parse().unwrap_or(0);
            }
            if let Some(m) = behind_re.captures(head) {
                behind = m[1].parse().unwrap_or(0);
            }
            continue;
        }
        let mut chars = line.chars();
        let x = chars.next().unwrap_or(' ');
        let y = chars.next().unwrap_or(' ');
        let file = line.chars().skip(3).collect::<String>().trim().to_string();
        let code = if x != ' ' && x != '?' {
            x
        } else if y != ' ' {
            y
        } else {
            x
        };
        let norm = file.replace('\\', "/");
        let scoped = if !prefix.is_empty() {
            norm.strip_prefix(prefix).unwrap_or(&norm).to_string()
        } else {
            norm
        };
        // An untracked workspace is reported by git as its own directory,
        // which strips to an empty string. Show it as the workspace root.
        let path = if scoped.is_empty() || scoped == "/" {
            ".".to_string()
        } else {
            scoped
        };
        files.push(json!({
            "path": path,
            "status": status_label(code),
            "staged": x != ' ' && x != '?',
        }));
    }
    json!({ "branch": branch, "ahead": ahead, "behind": behind, "files": files })
}

/// `diffFile`: staged then unstaged hunks, or a note when there are none.
pub fn diff_file(cwd: &Path, file: &str) -> String {
    let attempt = || -> Result<String, String> {
        let staged = git(cwd, &["diff", "--no-ext-diff", "--cached", "--", file])?;
        let unstaged = git(cwd, &["diff", "--no-ext-diff", "--", file])?;
        let text = format!("{staged}{unstaged}").trim().to_string();
        if !text.is_empty() {
            return Ok(text);
        }
        // A brand new untracked file has no diff; show it as an addition.
        let show = git(cwd, &["status", "--porcelain=v1", "--", file])?;
        Ok(if show.trim().starts_with("??") {
            "(untracked — no diff yet)".to_string()
        } else {
            "(no changes)".to_string()
        })
    };
    attempt().unwrap_or_else(|e| format!("(git diff failed: {e})"))
}

/// `log`: the last `limit` commits, empty when this is not a repository.
pub fn git_log(cwd: &Path, limit: usize) -> Vec<Value> {
    let count = format!("-{limit}");
    // %x1f is a literal unit-separator byte, which cannot appear in a subject.
    let Ok(out) = git(
        cwd,
        &["log", &count, "--pretty=format:%h%x1f%an%x1f%ar%x1f%s"],
    ) else {
        return Vec::new();
    };
    parse_log(&out)
}

fn parse_log(out: &str) -> Vec<Value> {
    out.split('\n')
        .filter(|l| !l.is_empty())
        .map(|l| {
            let mut fields = l.split('\x1f');
            let mut next = || fields.next().map(|s| json!(s)).unwrap_or(Value::Null);
            let hash = next();
            let author = next();
            let when = next();
            let subject = next();
            json!({ "hash": hash, "author": author, "when": when, "subject": subject })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_lines_map_to_labels_and_staging() {
        let out = "## main...origin/main [ahead 2, behind 1]\n M src/a.rs\nA  new.txt\n?? untracked.txt\nR  old.txt -> new-name.txt\nD  gone.txt\n";
        let status = parse_status(out, "");
        assert_eq!(status["branch"], "main");
        assert_eq!(status["ahead"], 2);
        assert_eq!(status["behind"], 1);
        let files = status["files"].as_array().unwrap();
        assert_eq!(
            files[0],
            json!({ "path": "src/a.rs", "status": "modified", "staged": false })
        );
        assert_eq!(
            files[1],
            json!({ "path": "new.txt", "status": "added", "staged": true })
        );
        assert_eq!(
            files[2],
            json!({ "path": "untracked.txt", "status": "untracked", "staged": false })
        );
        assert_eq!(files[3]["status"], "renamed");
        assert_eq!(files[4]["status"], "deleted");
    }

    #[test]
    fn status_scopes_paths_to_the_workspace_prefix() {
        let status = parse_status("## feature\n?? sub/\n M sub/file.txt\n", "sub/");
        assert_eq!(status["branch"], "feature");
        assert_eq!(status["files"][0]["path"], ".");
        assert_eq!(status["files"][1]["path"], "file.txt");
        let bare = parse_status("## HEAD (no branch)\n", "");
        assert_eq!(bare["branch"], "HEAD (no branch)");
        assert_eq!(bare["files"], json!([]));
    }

    #[test]
    fn log_lines_split_on_the_unit_separator() {
        let commits = parse_log(
            "abc1234\x1fAda\x1f2 days ago\x1fFix: the thing\n\ndef5678\x1fBob\x1fnow\x1f",
        );
        assert_eq!(commits.len(), 2);
        assert_eq!(
            commits[0],
            json!({ "hash": "abc1234", "author": "Ada", "when": "2 days ago", "subject": "Fix: the thing" })
        );
        assert_eq!(commits[1]["subject"], "");
        assert!(parse_log("").is_empty());
    }

    #[test]
    fn not_a_repository_is_reported_the_node_way() {
        let dir = tempfile::tempdir().unwrap();
        let error = read_status(dir.path()).unwrap_err();
        assert!(
            error.starts_with("Command failed: git ") || error.starts_with("spawn git "),
            "{error}"
        );
        assert!(git_log(dir.path(), 5).is_empty());
        assert!(diff_file(dir.path(), "x.txt").starts_with("(git diff failed: "));
    }
}
