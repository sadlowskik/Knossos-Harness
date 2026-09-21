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

// ---- change review: changes, revert, commit ------------------------------

/// Root-relative path with the workspace prefix stripped and `/` separators.
fn scope_path(path: &str, prefix: &str) -> String {
    let norm = path.replace('\\', "/");
    if prefix.is_empty() {
        norm
    } else {
        norm.strip_prefix(prefix).unwrap_or(&norm).to_string()
    }
}

/// `git diff --numstat -M` output as `path -> (additions, deletions)`, with a
/// rename (`old => new`, `dir/{old => new}/file`) keyed by its new path.
fn parse_numstat(out: &str, prefix: &str) -> BTreeMap<String, (u64, u64)> {
    let mut counts = BTreeMap::new();
    for line in out.split('\n').filter(|l| !l.trim().is_empty()) {
        let mut fields = line.splitn(3, '\t');
        let adds = fields.next().unwrap_or("-").trim();
        let dels = fields.next().unwrap_or("-").trim();
        let Some(path) = fields.next() else { continue };
        let path = numstat_new_path(path.trim());
        let adds: u64 = adds.parse().unwrap_or(0);
        let dels: u64 = dels.parse().unwrap_or(0);
        let entry = counts.entry(scope_path(&path, prefix)).or_insert((0, 0));
        entry.0 += adds;
        entry.1 += dels;
    }
    counts
}

fn numstat_new_path(path: &str) -> String {
    if let (Some(open), Some(close)) = (path.find('{'), path.rfind('}')) {
        if open < close {
            let inner = &path[open + 1..close];
            if let Some((_, new)) = inner.split_once(" => ") {
                return format!("{}{}{}", &path[..open], new, &path[close + 1..]);
            }
        }
    }
    match path.split_once(" => ") {
        Some((_, new)) => new.to_string(),
        None => path.to_string(),
    }
}

/// Lines in an untracked file; 0 for binaries and unreadable files.
fn count_lines(path: &Path) -> u64 {
    let Ok(bytes) = std::fs::read(path) else {
        return 0;
    };
    if bytes.contains(&0) {
        return 0;
    }
    let newlines = bytes.iter().filter(|b| **b == b'\n').count() as u64;
    if !bytes.is_empty() && bytes.last() != Some(&b'\n') {
        newlines + 1
    } else {
        newlines
    }
}

fn has_head(cwd: &Path) -> bool {
    git(cwd, &["rev-parse", "--verify", "-q", "HEAD"]).is_ok()
}

/// The HEAD revision as JSON, `null` outside a repository or before the first commit.
pub fn head(cwd: &Path) -> Value {
    git(cwd, &["rev-parse", "--verify", "-q", "HEAD"])
        .map(|r| json!(r.trim()))
        .unwrap_or(Value::Null)
}

fn show_prefix(cwd: &Path) -> String {
    git(cwd, &["rev-parse", "--show-prefix"])
        .map(|out| out.trim().replace('\\', "/"))
        .unwrap_or_default()
}

/// `GET /api/git/changes`: every changed file with its line counts, untracked
/// files listed one by one and counted as pure additions.
pub fn read_changes(cwd: &Path) -> Result<Value, String> {
    let prefix = show_prefix(cwd);
    let out = git(cwd, &["status", "--porcelain=v1", "-b", "-uall", "--", "."])?;
    let status = parse_status(&out, &prefix);
    let numstat = if has_head(cwd) {
        git(cwd, &["diff", "--numstat", "-M", "HEAD", "--", "."])?
    } else {
        let staged = git(cwd, &["diff", "--numstat", "-M", "--cached", "--", "."])?;
        let unstaged = git(cwd, &["diff", "--numstat", "-M", "--", "."])?;
        format!("{staged}{unstaged}")
    };
    let counts = parse_numstat(&numstat, &prefix);
    Ok(build_changes(&status, &counts, |rel| {
        count_lines(&cwd.join(rel))
    }))
}

fn build_changes(
    status: &Value,
    counts: &BTreeMap<String, (u64, u64)>,
    untracked_lines: impl Fn(&str) -> u64,
) -> Value {
    let mut files = Vec::new();
    let (mut total_add, mut total_del) = (0u64, 0u64);
    for f in status
        .get("files")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let raw = f.get("path").and_then(Value::as_str).unwrap_or("");
        let label = f.get("status").and_then(Value::as_str).unwrap_or("changed");
        if raw.is_empty() || raw == "." || label == "ignored" {
            continue;
        }
        let (path, from) = match (label, raw.split_once(" -> ")) {
            ("renamed" | "copied", Some((old, new))) => (new.to_string(), Some(old.to_string())),
            _ => (raw.to_string(), None),
        };
        let (additions, deletions) = if label == "untracked" {
            (untracked_lines(&path), 0)
        } else {
            counts.get(&path).copied().unwrap_or((0, 0))
        };
        total_add += additions;
        total_del += deletions;
        let mut entry = json!({
            "path": path,
            "status": label,
            "staged": f.get("staged").cloned().unwrap_or(Value::Bool(false)),
            "additions": additions,
            "deletions": deletions,
        });
        if let Some(from) = from {
            entry["from"] = json!(from);
        }
        files.push(entry);
    }
    json!({
        "branch": status.get("branch").cloned().unwrap_or(Value::Null),
        "files": files,
        "additions": total_add,
        "deletions": total_del,
    })
}

/// `POST /api/git/revert`: put the listed workspace-relative paths back to
/// HEAD. Tracked files are unstaged and checked out; a staged addition is
/// removed from the index and deleted; an untracked file is deleted.
/// Returns the paths actually reverted.
pub fn revert_paths(cwd: &Path, paths: &[String]) -> Result<Vec<String>, String> {
    let changes = read_changes(cwd)?;
    let mut by_path: BTreeMap<String, Value> = BTreeMap::new();
    for f in changes
        .get("files")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(p) = f.get("path").and_then(Value::as_str) {
            by_path.insert(p.to_string(), f.clone());
        }
    }
    let head = has_head(cwd);
    let mut reverted = Vec::new();
    for path in paths {
        let Some(entry) = by_path.get(path) else {
            return Err(format!("{path} has no changes to revert"));
        };
        let label = entry.get("status").and_then(Value::as_str).unwrap_or("");
        match label {
            "untracked" => remove_file(&cwd.join(path))?,
            "added" => {
                git(cwd, &["rm", "-q", "--cached", "--force", "--", path])?;
                remove_file(&cwd.join(path))?;
            }
            "renamed" | "copied" => {
                let from = entry.get("from").and_then(Value::as_str).unwrap_or(path);
                if head {
                    git(cwd, &["reset", "-q", "HEAD", "--", from, path])?;
                } else {
                    git(cwd, &["reset", "-q", "--", from, path])?;
                }
                git(cwd, &["checkout", "--", from])?;
                let tracked = git(cwd, &["ls-files", "--", path]).unwrap_or_default();
                if tracked.trim().is_empty() {
                    remove_file(&cwd.join(path))?;
                }
            }
            _ => {
                if head {
                    git(cwd, &["reset", "-q", "HEAD", "--", path])?;
                } else {
                    git(cwd, &["reset", "-q", "--", path])?;
                }
                git(cwd, &["checkout", "--", path])?;
            }
        }
        reverted.push(path.clone());
    }
    Ok(reverted)
}

fn remove_file(path: &Path) -> Result<(), String> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("remove {}: {e}", path.display())),
    }
}

/// `POST /api/git/commit`: stage the listed paths (or everything) and commit
/// them. Author comes from git config; without one, a local `Field`
/// identity is set for this commit only.
pub fn commit_paths(cwd: &Path, message: &str, paths: Option<&[String]>) -> Result<Value, String> {
    let changes = read_changes(cwd)?;
    if changes
        .get("files")
        .and_then(Value::as_array)
        .is_some_and(Vec::is_empty)
    {
        return Err("nothing to commit: the workspace is clean".into());
    }
    let prefix = show_prefix(cwd);
    let has_identity = |key: &str| {
        git(cwd, &["config", key])
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
    };
    let mut args: Vec<&str> = vec!["-c", "commit.gpgsign=false"];
    if !(has_identity("user.name") && has_identity("user.email")) {
        args.extend_from_slice(&[
            "-c",
            "user.name=Field",
            "-c",
            "user.email=field@knossos.invalid",
        ]);
    }
    args.extend_from_slice(&["commit", "-q", "-m", message]);
    let listed: Vec<&str> = paths.into_iter().flatten().map(String::as_str).collect();
    if listed.is_empty() {
        git(cwd, &["add", "-A", "--", "."])?;
    } else {
        let mut add = vec!["add", "-A", "--"];
        add.extend_from_slice(&listed);
        git(cwd, &add)?;
        args.push("--");
        args.extend_from_slice(&listed);
    }
    git(cwd, &args)?;
    let revision = git(cwd, &["rev-parse", "--verify", "HEAD"])?
        .trim()
        .to_string();
    let branch = git(cwd, &["rev-parse", "--abbrev-ref", "HEAD"])
        .map(|b| json!(b.trim()))
        .unwrap_or(Value::Null);
    let files: Vec<String> = git(
        cwd,
        &[
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            "--root",
            "HEAD",
        ],
    )?
    .split('\n')
    .filter(|l| !l.is_empty())
    .map(|l| scope_path(l, &prefix))
    .collect();
    Ok(json!({ "revision": revision, "branch": branch, "files": files }))
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
    fn numstat_counts_key_renames_by_their_new_path() {
        let out = "3\t1\tsrc/a.rs\n-\t-\timg.png\n2\t2\tsrc/{old => new}/x.rs\n5\t0\told.txt => new.txt\n1\t0\tsub/inner.txt\n";
        let counts = parse_numstat(out, "sub/");
        assert_eq!(counts["src/a.rs"], (3, 1));
        assert_eq!(counts["img.png"], (0, 0));
        assert_eq!(counts["src/new/x.rs"], (2, 2));
        assert_eq!(counts["new.txt"], (5, 0));
        assert_eq!(counts["inner.txt"], (1, 0));
    }

    #[test]
    fn changes_merge_status_with_counts_and_count_untracked_lines() {
        let status = parse_status(
            "## main\n M src/a.rs\n?? fresh.txt\nR  old.txt -> new.txt\n?? .\n",
            "",
        );
        let mut counts = BTreeMap::new();
        counts.insert("src/a.rs".to_string(), (3, 1));
        counts.insert("new.txt".to_string(), (0, 0));
        let changes = build_changes(&status, &counts, |p| if p == "fresh.txt" { 4 } else { 0 });
        assert_eq!(changes["branch"], "main");
        assert_eq!(changes["additions"], 7);
        assert_eq!(changes["deletions"], 1);
        let files = changes["files"].as_array().unwrap();
        assert_eq!(files.len(), 3, "{changes}");
        assert_eq!(files[0]["path"], "src/a.rs");
        assert_eq!(files[0]["additions"], 3);
        assert_eq!(files[1]["status"], "untracked");
        assert_eq!(files[1]["additions"], 4);
        assert_eq!(files[2]["path"], "new.txt");
        assert_eq!(files[2]["from"], "old.txt");
        assert_eq!(files[2]["status"], "renamed");

        let dir = tempfile::tempdir().unwrap();
        let text = dir.path().join("t.txt");
        std::fs::write(&text, "a\nb\nc").unwrap();
        assert_eq!(count_lines(&text), 3);
        std::fs::write(&text, "a\nb\n").unwrap();
        assert_eq!(count_lines(&text), 2);
        std::fs::write(&text, b"a\0b\n").unwrap();
        assert_eq!(count_lines(&text), 0);
        assert_eq!(count_lines(&dir.path().join("missing")), 0);
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
