//! Long-lived NDJSON server, for editor front ends.
//!
//! One JSON object per line in each direction: commands on stdin, events on
//! stdout. The process stays alive across turns, which is what lets a chat
//! panel hold a conversation instead of re-invoking a one-shot command and
//! losing all context.
//!
//! **stdout is the protocol.** Every human-readable byte goes to stderr; a
//! single stray `println!` on this path corrupts the stream. Progress events
//! reuse [`crate::session::TraceEvent`] verbatim, because the trace log
//! already is an event stream — the front end reads exactly the lines the log
//! file receives.

use std::io::{BufRead, Write};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::metis;
use crate::talos::{Outcome, Talos};

#[derive(Debug, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Command {
    /// Start a fresh task, discarding the previous conversation.
    Task { text: String },
    /// Continue the existing conversation.
    Resume { text: String },
    /// Plan without executing.
    Plan { text: String },
    /// Current staged changes, with full proposed content.
    Diffs,
    /// Write staged changes to disk.
    Apply,
    /// Write only the selected hunks, leaving the rest staged.
    ApplyHunks { selection: Vec<HunkSelection> },
    /// Throw staged changes away.
    Discard,
    /// Run the verification ladder now.
    Verify,
    /// Symbol counts, or an exact lookup.
    Index {
        #[serde(default)]
        name: Option<String>,
    },
    /// Clear the conversation, keep the workspace.
    Reset,
    Shutdown,
}

#[derive(Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    Ready {
        workspace: String,
        engine: String,
        constitution: String,
        symbols: usize,
        files: usize,
        dry_run: bool,
        max_steps: usize,
    },
    Plan {
        steps: Vec<String>,
    },
    Outcome {
        halt: String,
        succeeded: bool,
        steps_used: usize,
        summary: String,
        changed: Vec<String>,
        dry_run: bool,
    },
    Diffs {
        files: Vec<DiffPayload>,
    },
    Applied {
        files: Vec<String>,
    },
    Discarded,
    Verdict {
        passed: bool,
        summary: String,
        dry_run: bool,
        tiers: Vec<TierPayload>,
    },
    Index {
        symbols: usize,
        files: usize,
        hits: Vec<String>,
    },
    Reset,
    Error {
        message: String,
    },
    /// Every command ends with exactly one of these, so the front end always
    /// knows when it can re-enable input.
    Idle,
}

#[derive(Debug, Deserialize)]
pub struct HunkSelection {
    pub path: String,
    /// Hunk ids being accepted, as sent in the matching `diffs` event.
    pub hunks: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct DiffPayload {
    pub path: String,
    pub unified: String,
    pub added: usize,
    pub removed: usize,
    /// The full proposed file, so the editor can render a real side-by-side
    /// diff rather than parsing the unified text back apart.
    pub content: String,
    pub existed: bool,
    /// Individually acceptable pieces of this file's change.
    pub hunks: Vec<HunkPayload>,
}

#[derive(Debug, Serialize)]
pub struct HunkPayload {
    pub id: usize,
    pub header: String,
    pub body: String,
    pub added: usize,
    pub removed: usize,
}

#[derive(Debug, Serialize)]
pub struct TierPayload {
    pub tier: u8,
    pub label: String,
    pub passed: bool,
    pub detail: String,
}

fn emit(event: &Event) {
    let mut out = std::io::stdout();
    match serde_json::to_string(event) {
        Ok(line) => {
            let _ = writeln!(out, "{line}");
            let _ = out.flush();
        }
        Err(e) => eprintln!("could not serialize event: {e}"),
    }
}

pub async fn run(mut talos: Talos, max_tokens: u32) -> Result<()> {
    emit(&Event::Ready {
        workspace: talos.oracle.root().display().to_string(),
        engine: talos.session.engine_name.clone(),
        constitution: talos.themis.source().to_string(),
        symbols: talos.scribe.symbol_count(),
        files: talos.scribe.file_count(),
        dry_run: talos.ctx.is_dry_run(),
        max_steps: talos.ariadne.max_steps,
    });
    emit(&Event::Idle);

    let stdin = std::io::stdin();
    let mut line = String::new();

    loop {
        line.clear();
        if stdin.lock().read_line(&mut line)? == 0 {
            break; // front end closed the pipe
        }
        let trimmed = line.trim_start_matches('\u{feff}').trim();
        if trimmed.is_empty() {
            continue;
        }

        let command: Command = match serde_json::from_str(trimmed) {
            Ok(c) => c,
            Err(e) => {
                emit(&Event::Error { message: format!("bad command: {e}") });
                emit(&Event::Idle);
                continue;
            }
        };

        if matches!(command, Command::Shutdown) {
            break;
        }

        // A failing command must not kill the server: the conversation and any
        // staged work would go with it.
        if let Err(e) = dispatch(&mut talos, command, max_tokens).await {
            emit(&Event::Error { message: format!("{e:#}") });
        }
        emit(&Event::Idle);
    }

    Ok(())
}

async fn dispatch(talos: &mut Talos, command: Command, max_tokens: u32) -> Result<()> {
    match command {
        Command::Task { text } => {
            let plan = metis::plan(
                talos.engine.as_ref(),
                &talos.themis,
                &talos.scribe,
                &text,
                max_tokens,
            )
            .await?;
            emit(&Event::Plan { steps: plan.steps.clone() });
            let outcome = talos.run(&text, &plan).await?;
            finish_turn(talos, &outcome);
        }
        Command::Resume { text } => {
            let outcome = talos.resume(&text).await?;
            finish_turn(talos, &outcome);
        }
        Command::Plan { text } => {
            let plan = metis::plan(
                talos.engine.as_ref(),
                &talos.themis,
                &talos.scribe,
                &text,
                max_tokens,
            )
            .await?;
            emit(&Event::Plan { steps: plan.steps });
        }
        Command::Diffs => emit_diffs(talos),
        Command::Apply => {
            let written = talos.apply()?;
            emit(&Event::Applied {
                files: written.iter().map(|p| rel(talos, p)).collect(),
            });
        }
        Command::ApplyHunks { selection } => {
            let pairs: Vec<(String, Vec<usize>)> = selection
                .into_iter()
                .map(|s| (s.path, s.hunks))
                .collect();
            let written = talos.apply_hunks(&pairs)?;
            emit(&Event::Applied {
                files: written.iter().map(|p| rel(talos, p)).collect(),
            });
            // Anything partially accepted is still staged; re-send so the
            // front end shows what remains rather than a stale list.
            emit_diffs(talos);
        }
        Command::Discard => {
            talos.discard();
            emit(&Event::Discarded);
        }
        Command::Verify => {
            let files: Vec<std::path::PathBuf> = talos.changed.iter().cloned().collect();
            let verdict = if talos.ctx.is_dry_run() {
                talos
                    .oracle
                    .verify_staged(talos.scribe.adapter(), &talos.ctx.staged_contents())
            } else {
                talos.oracle.verify(talos.scribe.adapter(), &files).await?
            };
            emit(&Event::Verdict {
                passed: verdict.passed,
                summary: verdict.summary(),
                dry_run: verdict.dry_run,
                tiers: verdict
                    .tiers
                    .iter()
                    .map(|t| TierPayload {
                        tier: t.tier,
                        label: t.label.clone(),
                        passed: t.passed,
                        detail: t.detail.clone(),
                    })
                    .collect(),
            });
        }
        Command::Index { name } => {
            let hits = match &name {
                Some(n) => talos
                    .scribe
                    .lookup(n)
                    .iter()
                    .map(|s| format!("{}:{}: {}", s.file.display(), s.line, s.signature))
                    .collect(),
                None => Vec::new(),
            };
            emit(&Event::Index {
                symbols: talos.scribe.symbol_count(),
                files: talos.scribe.file_count(),
                hits,
            });
        }
        Command::Reset => {
            talos.messages.clear();
            talos.changed.clear();
            emit(&Event::Reset);
        }
        Command::Shutdown => unreachable!("handled by the caller"),
    }
    Ok(())
}

fn finish_turn(talos: &Talos, outcome: &Outcome) {
    emit(&Event::Outcome {
        halt: outcome.halt.label().to_string(),
        succeeded: outcome.succeeded(),
        steps_used: outcome.steps_used,
        summary: outcome.summary.clone(),
        changed: outcome.changed.iter().map(|p| rel(talos, p)).collect(),
        dry_run: outcome.dry_run,
    });
    if outcome.dry_run {
        emit_diffs(talos);
    }
}

fn emit_diffs(talos: &Talos) {
    let staged: std::collections::BTreeMap<_, _> =
        talos.ctx.staged_contents().into_iter().collect();

    let files = talos
        .diffs()
        .into_iter()
        .map(|d| {
            // `d.path` is workspace-relative; staging is keyed by absolute path.
            let absolute = talos.oracle.root().join(&d.path);
            DiffPayload {
                path: d.path.display().to_string(),
                unified: d.unified,
                added: d.added,
                removed: d.removed,
                content: staged.get(&absolute).cloned().unwrap_or_default(),
                existed: d.existed,
                hunks: d
                    .hunks
                    .iter()
                    .map(|h| HunkPayload {
                        id: h.id,
                        header: h.header.clone(),
                        body: h.body.clone(),
                        added: h.added,
                        removed: h.removed,
                    })
                    .collect(),
            }
        })
        .collect();

    emit(&Event::Diffs { files });
}

fn rel(talos: &Talos, path: &std::path::Path) -> String {
    talos.ctx.display(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_command_shape() {
        let cases = [
            r#"{"cmd":"task","text":"add a flag"}"#,
            r#"{"cmd":"resume","text":"now undo it"}"#,
            r#"{"cmd":"plan","text":"something"}"#,
            r#"{"cmd":"diffs"}"#,
            r#"{"cmd":"apply"}"#,
            r#"{"cmd":"apply_hunks","selection":[{"path":"src/lib.rs","hunks":[0,2]}]}"#,
            r#"{"cmd":"discard"}"#,
            r#"{"cmd":"verify"}"#,
            r#"{"cmd":"index"}"#,
            r#"{"cmd":"index","name":"Adder"}"#,
            r#"{"cmd":"reset"}"#,
            r#"{"cmd":"shutdown"}"#,
        ];
        for c in cases {
            serde_json::from_str::<Command>(c).unwrap_or_else(|e| panic!("{c} -> {e}"));
        }
    }

    #[test]
    fn an_unknown_command_is_a_parse_error_not_a_panic() {
        assert!(serde_json::from_str::<Command>(r#"{"cmd":"launch_missiles"}"#).is_err());
        assert!(serde_json::from_str::<Command>("not json").is_err());
    }

    #[test]
    fn events_serialize_with_a_discriminating_tag() {
        let e = Event::Applied { files: vec!["src/lib.rs".into()] };
        let v: serde_json::Value = serde_json::from_str(&serde_json::to_string(&e).unwrap()).unwrap();
        assert_eq!(v["event"], "applied");
        assert_eq!(v["files"][0], "src/lib.rs");

        let idle: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&Event::Idle).unwrap()).unwrap();
        assert_eq!(idle["event"], "idle");
    }

    #[test]
    fn diff_payload_carries_full_content_for_the_editor() {
        let p = DiffPayload {
            path: "src/lib.rs".into(),
            unified: "@@".into(),
            added: 1,
            removed: 0,
            content: "pub fn a() {}\n".into(),
            existed: true,
            hunks: vec![HunkPayload {
                id: 0,
                header: "@@ -1,1 +1,1 @@".into(),
                body: "+pub fn a() {}\n".into(),
                added: 1,
                removed: 0,
            }],
        };
        let v: serde_json::Value = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!(v["content"], "pub fn a() {}\n");
        assert_eq!(v["existed"], true);
    }
}
