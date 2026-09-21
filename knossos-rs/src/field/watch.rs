//! Real filesystem activity for every mounted workspace. Port of
//! `field/server/src/watch/fs.js`.
//!
//! Every change becomes an `fs.changed` event (`workspaceId`, `path`, `dir`,
//! `change` of `add` | `change` | `unlink`), which the projection folds into
//! the workspace file tree and routines use as a trigger. Writes are
//! coalesced per path for 250 ms so an editor's save (truncate, write,
//! rename) arrives as one change, matching chokidar's `awaitWriteFinish`.

use super::config::FieldSettings;
use super::eventlog::AppendOptions;
use super::git::read_status;
use super::js::{get, get_str, js_string};
use super::policy::build_matcher;
use super::registry::Emit;
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// `.field-state` matters more than it looks: the event store writes its
/// SQLite WAL there on every append. Watching it would turn each event into a
/// filesystem event and feed the log back into itself.
const ALWAYS_IGNORE: &[&str] = &[
    "**/.git/**",
    "**/node_modules/**",
    "**/target/**",
    "**/dist/**",
    "**/.field-state/**",
    "**/.pytest_cache/**",
    "**/__pycache__/**",
    "**/.venv/**",
    "**/*.pyc",
    "**/*.swp",
    "**/*.tmp",
    "**/.DS_Store",
];
const MAX_DEPTH: usize = 14;
const SETTLE: Duration = Duration::from_millis(250);
const POLL: Duration = Duration::from_millis(50);

/// Keeps the watchers alive; dropping it stops them.
pub struct FsWatchers {
    _watchers: Vec<RecommendedWatcher>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Change {
    Added,
    Modified,
    Removed,
}

impl Change {
    fn as_str(self) -> &'static str {
        match self {
            Change::Added => "add",
            Change::Modified => "change",
            Change::Removed => "unlink",
        }
    }
}

fn relative(root: &Path, abs: &Path) -> Option<String> {
    let rel = abs.strip_prefix(root).ok()?;
    let text = rel.to_string_lossy().replace('\\', "/");
    if text.is_empty() || text.starts_with("..") {
        return None;
    }
    Some(text)
}

/// Translates one notify event into the chokidar vocabulary. A rename
/// arrives as an unlink of the old name and an add of the new one.
fn changes(event: &Event) -> Vec<(PathBuf, Change)> {
    use notify::event::{ModifyKind, RenameMode};
    match &event.kind {
        EventKind::Create(_) => event
            .paths
            .iter()
            .map(|p| (p.clone(), Change::Added))
            .collect(),
        EventKind::Remove(_) => event
            .paths
            .iter()
            .map(|p| (p.clone(), Change::Removed))
            .collect(),
        EventKind::Modify(ModifyKind::Name(mode)) => match mode {
            RenameMode::From => event
                .paths
                .iter()
                .map(|p| (p.clone(), Change::Removed))
                .collect(),
            RenameMode::To => event
                .paths
                .iter()
                .map(|p| (p.clone(), Change::Added))
                .collect(),
            RenameMode::Both => {
                let mut out = Vec::new();
                if let Some(from) = event.paths.first() {
                    out.push((from.clone(), Change::Removed));
                }
                if let Some(to) = event.paths.get(1) {
                    out.push((to.clone(), Change::Added));
                }
                out
            }
            _ => event
                .paths
                .iter()
                .map(|p| (p.clone(), Change::Modified))
                .collect(),
        },
        EventKind::Modify(_) => event
            .paths
            .iter()
            .map(|p| (p.clone(), Change::Modified))
            .collect(),
        _ => Vec::new(),
    }
}

/// Starts one recursive watcher per mounted workspace. Errors on a single
/// workspace are reported and skipped; the others still run.
pub fn start_fs_watchers(settings: &FieldSettings, emit: Emit, state_dir: &Path) -> FsWatchers {
    // The state directory is normally `.field-state` and already ignored by
    // name; when it lives elsewhere inside a workspace (tests, custom
    // FIELD_STATE) the event store's own writes must still never come back
    // as filesystem events.
    let mut skip_roots = vec![state_dir.to_path_buf()];
    if let Ok(canonical) = state_dir.canonicalize() {
        skip_roots.push(canonical);
    }
    let mut watchers = Vec::new();
    for w in &settings.workspaces {
        if !get(w, "mounted").is_some_and(super::js::truthy) {
            continue;
        }
        let id = get_str(w, "id").unwrap_or("").to_string();
        let root = PathBuf::from(get(w, "path").map(js_string).unwrap_or_default());
        let mut patterns: Vec<String> = ALWAYS_IGNORE.iter().map(|s| s.to_string()).collect();
        if let Some(extra) = get(w, "watch")
            .and_then(|watch| get(watch, "ignore"))
            .and_then(|v| v.as_array())
        {
            patterns.extend(extra.iter().map(js_string));
        }
        match watch_workspace(
            id.clone(),
            root.clone(),
            patterns,
            skip_roots.clone(),
            emit.clone(),
        ) {
            Ok(watcher) => watchers.push(watcher),
            Err(error) => eprintln!("[fs:{id}] cannot watch {}: {error}", root.display()),
        }
    }
    FsWatchers {
        _watchers: watchers,
    }
}

fn watch_workspace(
    id: String,
    root: PathBuf,
    patterns: Vec<String>,
    skip_roots: Vec<PathBuf>,
    emit: Emit,
) -> notify::Result<RecommendedWatcher> {
    let (tx, rx) = mpsc::channel::<(PathBuf, Change)>();
    let mut watcher = RecommendedWatcher::new(
        move |result: notify::Result<Event>| {
            if let Ok(event) = result {
                for change in changes(&event) {
                    let _ = tx.send(change);
                }
            }
        },
        notify::Config::default(),
    )?;
    watcher.watch(&root, RecursiveMode::Recursive)?;

    // Coalesce per path: the last change wins once the path has been quiet
    // for SETTLE. Ignore rules run here, off the watcher's own thread.
    let ignore = build_matcher(&patterns);
    let canonical_root = root.canonicalize().unwrap_or_else(|_| root.clone());
    std::thread::Builder::new()
        .name(format!("fs-watch-{id}"))
        .spawn(move || {
            let mut pending: BTreeMap<String, (Change, Instant)> = BTreeMap::new();
            loop {
                match rx.recv_timeout(POLL) {
                    Ok((abs, change)) => {
                        if skip_roots.iter().any(|r| abs.starts_with(r)) {
                            continue;
                        }
                        let rel = match relative(&root, &abs)
                            .or_else(|| relative(&canonical_root, &abs))
                        {
                            Some(rel) => rel,
                            None => continue,
                        };
                        if rel.split('/').count() > MAX_DEPTH {
                            continue;
                        }
                        // Patterns may be absolute-ish (`**/node_modules/**`) or
                        // workspace-relative (`RTS/**`); test both forms.
                        let abs_text = abs.to_string_lossy().replace('\\', "/");
                        if ignore(&abs_text) || ignore(&rel) {
                            continue;
                        }
                        pending.insert(rel, (change, Instant::now()));
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
                let now = Instant::now();
                let due: Vec<String> = pending
                    .iter()
                    .filter(|(_, (_, at))| now.duration_since(*at) >= SETTLE)
                    .map(|(rel, _)| rel.clone())
                    .collect();
                for rel in due {
                    let Some((change, _)) = pending.remove(&rel) else {
                        continue;
                    };
                    let dir = rel
                        .rfind('/')
                        .map(|i| rel[..i].to_string())
                        .unwrap_or_default();
                    emit(
                        "fs.changed",
                        json!({ "workspaceId": id, "path": rel, "dir": dir, "change": change.as_str() }),
                        AppendOptions {
                            actor: None,
                            subject: Some(id.clone()),
                            source: None,
                            simulated: false,
                        },
                    );
                }
            }
        })
        .map_err(|e| notify::Error::generic(&e.to_string()))?;
    Ok(watcher)
}

/// Polls `git status` for every mounted Git workspace and emits `git.status`
/// when it changes. Port of `startGitWatchers` in `watch/git.js`: a workspace
/// that is not a repository is reported once, as branchless, then left alone.
pub struct GitWatchers {
    stop: Arc<AtomicBool>,
}

impl Drop for GitWatchers {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}

pub fn start_git_watchers(settings: &FieldSettings, emit: Emit, interval: Duration) -> GitWatchers {
    let stop = Arc::new(AtomicBool::new(false));
    let workspaces: Vec<(String, PathBuf)> = settings
        .workspaces
        .iter()
        .filter(|w| {
            get(w, "mounted").is_some_and(super::js::truthy)
                && get(w, "git") != Some(&serde_json::Value::Bool(false))
        })
        .filter_map(|w| {
            Some((
                get_str(w, "id")?.to_string(),
                PathBuf::from(get(w, "path").map(js_string)?),
            ))
        })
        .collect();
    let flag = Arc::clone(&stop);
    let _ = std::thread::Builder::new()
        .name("git-watch".into())
        .spawn(move || {
            let mut last: BTreeMap<String, String> = BTreeMap::new();
            while !flag.load(Ordering::SeqCst) {
                for (id, path) in &workspaces {
                    match read_status(path) {
                        Ok(status) => {
                            let fingerprint = status.to_string();
                            if last.get(id) == Some(&fingerprint) {
                                continue;
                            }
                            last.insert(id.clone(), fingerprint);
                            let mut data = status;
                            if let Some(obj) = data.as_object_mut() {
                                obj.insert("workspaceId".into(), json!(id));
                            }
                            emit("git.status", data, subject(id));
                        }
                        Err(_) => {
                            if !last.contains_key(id) {
                                last.insert(id.clone(), "nogit".into());
                                emit(
                                    "git.status",
                                    json!({ "workspaceId": id, "branch": null, "ahead": 0, "behind": 0, "files": [] }),
                                    subject(id),
                                );
                            }
                        }
                    }
                }
                // Sleep in short slices so a stop request is honoured promptly.
                let until = Instant::now() + interval;
                while Instant::now() < until && !flag.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        });
    GitWatchers { stop }
}

fn subject(id: &str) -> AppendOptions {
    AppendOptions {
        actor: None,
        subject: Some(id.to_string()),
        source: None,
        simulated: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::sync::{Arc, Mutex};

    #[test]
    fn a_written_file_becomes_one_fs_changed_event() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(root.join("node_modules").join("pkg")).unwrap();
        let events: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&events);
        let emit: Emit = Arc::new(move |kind, data, _| {
            sink.lock()
                .unwrap()
                .push(json!({ "kind": kind, "data": data }));
            None
        });
        let settings = FieldSettings {
            workspaces: vec![
                json!({ "id": "ws", "path": root.to_string_lossy(), "mounted": true }),
            ],
            ..Default::default()
        };
        let _watchers = start_fs_watchers(&settings, emit, &root.join("state"));
        // Give the OS watcher a moment to arm before writing.
        std::thread::sleep(Duration::from_millis(300));
        std::fs::write(root.join("src").join("a.txt"), "one").unwrap();
        std::fs::write(root.join("src").join("a.txt"), "two").unwrap();
        std::fs::write(
            root.join("node_modules").join("pkg").join("x.js"),
            "ignored",
        )
        .unwrap();

        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let seen = events.lock().unwrap().clone();
            if seen.iter().any(|e| e["data"]["path"] == "src/a.txt")
                && Instant::now() > deadline - Duration::from_secs(4)
            {
                assert!(
                    seen.iter().all(|e| e["kind"] == "fs.changed"),
                    "only fs.changed events: {seen:?}"
                );
                assert!(
                    seen.iter()
                        .all(|e| !js_string(&e["data"]["path"]).contains("node_modules")),
                    "node_modules is ignored: {seen:?}"
                );
                let for_a: Vec<&Value> = seen
                    .iter()
                    .filter(|e| e["data"]["path"] == "src/a.txt")
                    .collect();
                assert_eq!(for_a.len(), 1, "two quick writes coalesce into one event");
                assert_eq!(for_a[0]["data"]["dir"], "src");
                assert_eq!(for_a[0]["data"]["workspaceId"], "ws");
                return;
            }
            assert!(
                Instant::now() < deadline,
                "no fs.changed event arrived: {seen:?}"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}
