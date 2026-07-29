//! What has happened to the workspace's files, so it can be checked or undone.
//!
//! Split out of [`ToolCtx`](super::ToolCtx), which describes *what a tool is
//! allowed to touch* — a question about permission, answered before anything
//! happens. Freshness and undo are the opposite kind of question: they are
//! about what already did happen, and they only have answers after the fact.
//! Keeping both on one type meant a struct whose own docstring described half
//! of it.
//!
//! The two live together here because they are the same bookkeeping seen from
//! two directions. A write needs the previous content to journal it, and the
//! new content to re-baseline the freshness check; both are recorded at the
//! same instant from the same values. Splitting *these* two would mean reading
//! every file twice to answer one question.
//!
//! # Locking
//!
//! One mutex over all three maps, and no lock is ever held across filesystem
//! I/O. As three separate locks these had an ordering — `rewind` took the
//! journal and then reached for the stamps — that nothing recorded and nothing
//! enforced. One lock has no ordering to get wrong.

use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};

/// What a file held the last time the harness read it.
///
/// A content hash rather than a modification time. mtime resolution varies by
/// filesystem — whole seconds on some — and the case this has to catch is an
/// editor saving a file of the same length inside one tick, which is precisely
/// where mtime says nothing. The content is already in memory when a read
/// happens, so hashing it there costs nothing; the check re-reads the file,
/// which is one syscall weighed against silently discarding someone's work.
///
/// `DefaultHasher` is not stable across Rust releases. That is fine: stamps are
/// compared only against other stamps taken in the same process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stamp {
    hash: u64,
    len: u64,
}

impl Stamp {
    fn of(content: &str) -> Self {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        content.hash(&mut h);
        Stamp { hash: h.finish(), len: content.len() as u64 }
    }
}

/// How a file on disk diverged from what the harness last read from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conflict {
    /// Something wrote to it after the harness read it.
    Modified,
    /// It existed when the harness read it and does not now.
    Deleted,
}

impl Conflict {
    pub fn describe(self) -> &'static str {
        match self {
            Conflict::Modified => "changed on disk since it was read",
            Conflict::Deleted => "was deleted after it was read",
        }
    }
}

/// What a path held immediately before the harness changed it.
///
/// `before == None` means the file did not exist, so undoing the entry means
/// deleting it rather than restoring empty content — a distinction that matters
/// the first time an agent creates a file you did not want.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JournalEntry {
    pub path: PathBuf,
    pub before: Option<String>,
}

#[derive(Debug, Default)]
struct Inner {
    /// What each file held when the harness last read it from disk.
    observed: BTreeMap<PathBuf, Stamp>,
    /// Every disk write, oldest first, with the content it replaced.
    journal: Vec<JournalEntry>,
    /// Label to journal length when the mark was taken.
    marks: BTreeMap<String, usize>,
}

/// Shared, cheap to clone: every clone of a [`ToolCtx`](super::ToolCtx) sees
/// the same history, which is what lets a front end hold one while the loop
/// holds another.
#[derive(Debug, Clone, Default)]
pub struct History {
    inner: Arc<Mutex<Inner>>,
}

impl History {
    pub fn new() -> Self {
        History::default()
    }

    // ------------------------------------------------------------ freshness

    /// Record what a disk read returned, as the baseline for later checks.
    pub fn observed(&self, path: &Path, content: &str) {
        self.inner
            .lock()
            .unwrap()
            .observed
            .insert(path.to_path_buf(), Stamp::of(content));
    }

    /// Whether `path` still holds what the harness last read from it.
    ///
    /// `None` when the file has never been read — a first write, or a file the
    /// agent is creating, has nothing to be stale against and is not the case
    /// this guards.
    pub fn conflict(&self, path: &Path) -> Option<Conflict> {
        let expected = *self.inner.lock().unwrap().observed.get(path)?;

        match std::fs::read_to_string(path) {
            Ok(current) if Stamp::of(&current) == expected => None,
            Ok(_) => Some(Conflict::Modified),
            // Anything unreadable that was readable before is, from here,
            // indistinguishable from deletion and equally worth stopping on.
            Err(_) => Some(Conflict::Deleted),
        }
    }

    /// Accept whatever is on disk now as the new baseline.
    ///
    /// The escape hatch for a conflict the caller has decided is fine —
    /// otherwise a file that changed once would refuse writes forever.
    pub fn accept_current(&self, path: &Path) {
        // Read before locking: this is I/O, and the lock covers three maps.
        let current = std::fs::read_to_string(path).ok();
        let mut inner = self.inner.lock().unwrap();
        match current {
            Some(content) => {
                inner.observed.insert(path.to_path_buf(), Stamp::of(&content));
            }
            None => {
                inner.observed.remove(path);
            }
        }
    }

    // -------------------------------------------------------------- journal

    /// Capture what a path holds before it is overwritten.
    ///
    /// An unreadable file that is not simply missing records *nothing*, and
    /// that is the one place this diverges from the Python journal, which
    /// collapses every read error to "did not exist". The consequence there is
    /// that rewinding deletes a file it was never able to read — a permissions
    /// blip or a non-UTF-8 file becomes data loss during an undo. Recording
    /// nothing means such a file is not restored, which is the strictly safer
    /// side of the same uncertainty.
    pub fn record(&self, path: &Path) {
        let before = match std::fs::read_to_string(path) {
            Ok(content) => Some(content),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(_) => return,
        };
        self.inner
            .lock()
            .unwrap()
            .journal
            .push(JournalEntry { path: path.to_path_buf(), before });
    }

    /// Note that the harness itself wrote this content.
    ///
    /// Re-baselines the freshness check: without it, the second write to a file
    /// would be refused as somebody else's edit.
    pub fn wrote(&self, path: &Path, content: &str) {
        self.observed(path, content);
    }

    /// Mark a point the workspace can be rewound to.
    ///
    /// Cheap: it records a position in the journal rather than copying files,
    /// so marking before every turn costs nothing.
    pub fn checkpoint(&self, label: impl Into<String>) -> String {
        let label = label.into();
        let mut inner = self.inner.lock().unwrap();
        let at = inner.journal.len();
        inner.marks.insert(label.clone(), at);
        label
    }

    /// Restore every file to its state at `label`, returning what changed.
    ///
    /// Replayed newest-first, so a path written several times during the
    /// interval lands on the oldest content rather than an intermediate one.
    pub fn rewind(&self, label: &str) -> Result<Vec<PathBuf>> {
        // Take everything the rewind needs, then let go of the lock: the loop
        // below is filesystem I/O and must not run under it.
        let tail = {
            let mut inner = self.inner.lock().unwrap();
            let Some(mark) = inner.marks.get(label).copied() else {
                bail!("no checkpoint named `{label}`");
            };
            let tail = inner.journal.split_off(mark);
            // Marks taken after this one no longer refer to anything real.
            inner.marks.retain(|_, at| *at <= mark);
            tail
        };

        let mut restored = Vec::new();
        let mut seen = BTreeSet::new();

        for entry in tail.iter().rev() {
            let ok = match &entry.before {
                // It did not exist before; undoing means removing it. An error
                // here — already gone, or never created — means nothing
                // changed, so it is not reported as restored.
                None => std::fs::remove_file(&entry.path).is_ok(),
                Some(before) => std::fs::write(&entry.path, before).is_ok(),
            };
            if ok && seen.insert(entry.path.clone()) {
                restored.push(entry.path.clone());
            }
            // What is on disk now is the baseline again, so a later write is
            // not reported as somebody else's edit.
            self.accept_current(&entry.path);
        }

        Ok(restored)
    }

    /// Every recorded write, oldest first.
    pub fn entries(&self) -> Vec<JournalEntry> {
        self.inner.lock().unwrap().journal.clone()
    }
}
