//! Durable episode memory: failed hypotheses and attempts that survive Lethe.
//!
//! The conversation is not memory. Lethe shrinks tool results in place so a
//! compiler error that just taught the agent what not to do can vanish from the
//! next prompt. The JSONL trace is write-only. This module is the store that
//! *consumes* those events and hands a compact reminder back at the seams
//! where forgetting actually happens: a new run, a compaction, a redirect.
//!
//! JSONL rather than SQLite, matching [`Session`](crate::session::Session). The
//! file is append-only; open rebuilds the in-memory index. The agent must not
//! write here — `.knossos` is on [`ProtectPaths`](crate::hooks::ProtectPaths).

use std::collections::HashSet;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

const MAX_BRIEF_CHARS: usize = 1_500;
const MAX_DETAIL_CHARS: usize = 400;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Record {
    Hypothesis {
        task: String,
        signature: String,
        detail: String,
        step: usize,
    },
    Attempt {
        id: String,
        parent: Option<String>,
        halt: String,
        changed: Vec<String>,
        summary: String,
        task: String,
    },
    Redirect {
        from_attempt: String,
        forbidden: Vec<String>,
        reason: String,
        task: String,
    },
}

impl Record {
    pub fn task(&self) -> &str {
        match self {
            Record::Hypothesis { task, .. }
            | Record::Attempt { task, .. }
            | Record::Redirect { task, .. } => task,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EpisodeStore {
    path: PathBuf,
    records: Vec<Record>,
}

impl EpisodeStore {
    /// Open the store for `root`. Missing files are empty, not an error;
    /// creating the directory is deferred until the first write so constructing
    /// a `Talos` does not plant a `.knossos` in a workspace that never ran.
    pub fn open(root: impl AsRef<Path>) -> Self {
        let path = root.as_ref().join(".knossos").join("episodes.jsonl");
        let records = load(&path);
        EpisodeStore { path, records }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn records(&self) -> &[Record] {
        &self.records
    }

    /// Append one record. Failures are logged and swallowed: losing the
    /// reminder is bad, aborting the run over a log line is worse.
    pub fn record(&mut self, rec: Record) {
        let rec = cap_detail(rec);
        if let Err(e) = append(&self.path, &rec) {
            tracing::warn!("could not write episode to {}: {e}", self.path.display());
            return;
        }
        self.records.push(rec);
    }

    /// Workspace records that look like `task`, then recent failures.
    pub fn recall(&self, task: &str, limit: usize) -> Vec<Record> {
        if self.records.is_empty() || limit == 0 {
            return Vec::new();
        }
        let mut scored: Vec<(usize, usize)> = self
            .records
            .iter()
            .enumerate()
            .map(|(i, r)| (overlap(task, r.task()), i))
            .collect();
        // Overlap first, then recency (later index wins ties).
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for (_, i) in scored {
            if out.len() >= limit {
                break;
            }
            let rec = &self.records[i];
            let key = record_key(rec);
            if seen.insert(key) {
                out.push(rec.clone());
            }
        }
        out
    }
}

/// Compact reminder for the next prompt. Empty when there is nothing to say.
pub fn render_brief(records: &[Record]) -> String {
    if records.is_empty() {
        return String::new();
    }
    let mut lines = vec![
        "Previously failed in this workspace — do not retry these without new evidence:"
            .to_string(),
    ];
    let mut attempts = Vec::new();
    for rec in records {
        match rec {
            Record::Hypothesis {
                signature, detail, ..
            } => {
                let detail = if detail.is_empty() {
                    String::new()
                } else {
                    format!(": {detail}")
                };
                lines.push(format!("- {signature}{detail}"));
            }
            Record::Attempt {
                id,
                parent,
                halt,
                summary,
                ..
            } => {
                let parent = parent
                    .as_deref()
                    .map(|p| format!(", parent {p}"))
                    .unwrap_or_default();
                attempts.push(format!("- {id}{parent}: {halt}; {summary}"));
            }
            Record::Redirect {
                from_attempt,
                forbidden,
                reason,
                ..
            } => {
                lines.push(format!(
                    "- redirect from {from_attempt} ({reason}); forbidden: {}",
                    forbidden.join("; ")
                ));
            }
        }
    }
    if !attempts.is_empty() {
        lines.push("Attempts:".to_string());
        lines.extend(attempts);
    }
    let mut brief = lines.join("\n");
    if brief.len() > MAX_BRIEF_CHARS {
        brief.truncate(MAX_BRIEF_CHARS);
        brief.push_str("\n[truncated]");
    }
    brief
}

fn record_key(rec: &Record) -> String {
    match rec {
        Record::Hypothesis {
            signature, detail, ..
        } => format!("h:{signature}:{detail}"),
        Record::Attempt { id, .. } => format!("a:{id}"),
        Record::Redirect { from_attempt, .. } => format!("r:{from_attempt}"),
    }
}

fn cap_detail(rec: Record) -> Record {
    match rec {
        Record::Hypothesis {
            task,
            signature,
            detail,
            step,
        } => Record::Hypothesis {
            task,
            signature,
            detail: clip(&detail, MAX_DETAIL_CHARS),
            step,
        },
        Record::Attempt {
            id,
            parent,
            halt,
            changed,
            summary,
            task,
        } => Record::Attempt {
            id,
            parent,
            halt,
            changed,
            summary: clip(&summary, MAX_DETAIL_CHARS),
            task,
        },
        other => other,
    }
}

fn clip(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
    }
}

fn overlap(task: &str, stored: &str) -> usize {
    let a: HashSet<String> = tokenize(task);
    if a.is_empty() {
        return 0;
    }
    tokenize(stored)
        .into_iter()
        .filter(|t| a.contains(t))
        .count()
}

fn tokenize(s: &str) -> HashSet<String> {
    s.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| t.len() > 1)
        .map(|t| t.to_ascii_lowercase())
        .collect()
}

fn load(path: &Path) -> Vec<Record> {
    let Ok(file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let mut records = Vec::new();
    for line in std::io::BufReader::new(file).lines() {
        let Ok(line) = line else {
            continue;
        };
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Envelope>(&line) {
            Ok(env) => records.push(env.record),
            Err(e) => tracing::warn!("skipping malformed episode line: {e}"),
        }
    }
    records
}

fn append(path: &Path, rec: &Record) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let line = serde_json::to_string(&Envelope {
        at: chrono::Utc::now().to_rfc3339(),
        record: rec.clone(),
    })?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{line}")?;
    Ok(())
}

#[derive(Serialize, Deserialize)]
struct Envelope {
    at: String,
    #[serde(flatten)]
    record: Record,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hyp(task: &str, sig: &str) -> Record {
        Record::Hypothesis {
            task: task.into(),
            signature: sig.into(),
            detail: "old_string not found".into(),
            step: 2,
        }
    }

    #[test]
    fn a_record_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = EpisodeStore::open(dir.path());
        store.record(hyp(
            "add triple",
            r#"[["edit_file",{"path":"src/lib.rs"}]]"#,
        ));
        drop(store);

        let again = EpisodeStore::open(dir.path());
        assert_eq!(again.records().len(), 1);
        match &again.records()[0] {
            Record::Hypothesis {
                task, signature, ..
            } => {
                assert_eq!(task, "add triple");
                assert!(signature.contains("edit_file"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn opening_does_not_create_the_directory() {
        let dir = tempfile::tempdir().unwrap();
        let _ = EpisodeStore::open(dir.path());
        assert!(!dir.path().join(".knossos").exists());
    }

    #[test]
    fn recall_prefers_the_matching_task() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = EpisodeStore::open(dir.path());
        store.record(hyp("unrelated chore", "other"));
        store.record(hyp("add a triple function", "edit_file triple"));
        let hits = store.recall("please add triple", 4);
        assert!(!hits.is_empty());
        assert!(hits[0].task().contains("triple"), "{:?}", hits[0]);
    }

    #[test]
    fn render_brief_is_empty_when_there_is_nothing_to_say() {
        assert!(render_brief(&[]).is_empty());
    }

    #[test]
    fn render_brief_names_the_failed_call() {
        let text = render_brief(&[hyp("t", "edit_file src/lib.rs")]);
        assert!(text.contains("Previously failed"));
        assert!(text.contains("edit_file src/lib.rs"));
        assert!(text.contains("do not retry"));
    }

    #[test]
    fn attempt_lineage_is_in_the_brief() {
        let rec = Record::Attempt {
            id: "attempt-2".into(),
            parent: Some("attempt-1".into()),
            halt: "stuck".into(),
            changed: vec!["src/lib.rs".into()],
            summary: "FAILED at cargo".into(),
            task: "add triple".into(),
        };
        let text = render_brief(&[rec]);
        assert!(text.contains("attempt-2"));
        assert!(text.contains("parent attempt-1"));
        assert!(text.contains("FAILED at cargo"));
    }
}
