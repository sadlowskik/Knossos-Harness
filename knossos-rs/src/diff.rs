//! Unified diffs for previewing changes before they touch disk.

use std::path::{Path, PathBuf};

use similar::{ChangeTag, TextDiff};

/// Lines of unchanged context kept around each hunk.
const CONTEXT: usize = 3;

/// One independently acceptable change within a file.
///
/// Hunks are addressable so a reviewer can take some and leave others. The
/// spans are what make that reconstructable: `old_start`/`old_len` say which
/// original lines this hunk replaces, and `new_lines` is what replaces them.
#[derive(Debug, Clone)]
pub struct Hunk {
    /// Index within the file's hunk list. Stable for one diff.
    pub id: usize,
    /// `@@ -a,b +c,d @@`
    pub header: String,
    /// The hunk rendered with +/-/space prefixes, for display.
    pub body: String,
    /// 0-indexed first original line this hunk covers.
    pub old_start: usize,
    /// How many original lines it covers.
    pub old_len: usize,
    /// The replacement lines, without prefixes or trailing newlines.
    pub new_lines: Vec<String>,
    pub added: usize,
    pub removed: usize,
}

#[derive(Debug, Clone)]
pub struct FileDiff {
    pub path: PathBuf,
    /// `None` when the file did not exist before.
    pub existed: bool,
    pub unified: String,
    pub added: usize,
    pub removed: usize,
    pub hunks: Vec<Hunk>,
}

impl FileDiff {
    pub fn is_empty(&self) -> bool {
        self.added == 0 && self.removed == 0
    }

    /// One-line summary, e.g. `src/lib.rs  +12 -3 (new)`.
    pub fn stat(&self) -> String {
        format!(
            "{}  +{} -{}{}",
            self.path.display(),
            self.added,
            self.removed,
            if self.existed { "" } else { " (new)" }
        )
    }
}

pub fn diff_file(path: &Path, before: Option<&str>, after: &str) -> FileDiff {
    let existed = before.is_some();
    let before = before.unwrap_or("");
    let td = TextDiff::from_lines(before, after);

    let mut added = 0usize;
    let mut removed = 0usize;
    for change in td.iter_all_changes() {
        match change.tag() {
            ChangeTag::Insert => added += 1,
            ChangeTag::Delete => removed += 1,
            ChangeTag::Equal => {}
        }
    }

    let label = path.display().to_string();
    let unified = td
        .unified_diff()
        .context_radius(CONTEXT)
        .header(
            &if existed {
                format!("a/{label}")
            } else {
                "/dev/null".to_string()
            },
            &format!("b/{label}"),
        )
        .to_string();

    let hunks = build_hunks(&td);

    FileDiff {
        path: path.to_path_buf(),
        existed,
        unified,
        added,
        removed,
        hunks,
    }
}

/// Split a diff into independently acceptable hunks.
///
/// `grouped_ops` already clusters changes with their surrounding context, which
/// is exactly hunk segmentation — so this walks those groups and records, for
/// each, which original lines it replaces and with what.
fn build_hunks(td: &TextDiff<'_, '_, str>) -> Vec<Hunk> {
    let mut hunks = Vec::new();

    for (id, group) in td.grouped_ops(CONTEXT).into_iter().enumerate() {
        let Some(first) = group.first() else { continue };
        let Some(last) = group.last() else { continue };

        let old_start = first.old_range().start;
        let old_end = last.old_range().end;
        let new_start = first.new_range().start;
        let new_end = last.new_range().end;

        let mut body = String::new();
        let mut new_lines = Vec::new();
        let mut added = 0usize;
        let mut removed = 0usize;

        for op in &group {
            for change in td.iter_changes(op) {
                let value = change.value();
                let text = value.strip_suffix('\n').unwrap_or(value);
                match change.tag() {
                    ChangeTag::Equal => {
                        body.push_str(&format!(" {text}\n"));
                        new_lines.push(text.to_string());
                    }
                    ChangeTag::Insert => {
                        body.push_str(&format!("+{text}\n"));
                        new_lines.push(text.to_string());
                        added += 1;
                    }
                    ChangeTag::Delete => {
                        body.push_str(&format!("-{text}\n"));
                        removed += 1;
                    }
                }
            }
        }

        hunks.push(Hunk {
            id,
            header: format!(
                "@@ -{},{} +{},{} @@",
                old_start + 1,
                old_end - old_start,
                new_start + 1,
                new_end - new_start
            ),
            body,
            old_start,
            old_len: old_end - old_start,
            new_lines,
            added,
            removed,
        });
    }

    hunks
}

/// Rebuild a file from its original content plus only the accepted hunks.
///
/// Hunks are disjoint and ordered, so this is a single pass: copy original
/// lines up to each hunk, then either emit the hunk's replacement (accepted) or
/// the original span (rejected). Selecting every hunk reproduces the full
/// proposed file; selecting none reproduces the original exactly.
pub fn apply_hunks(original: &str, hunks: &[Hunk], accepted: &[usize]) -> String {
    let lines: Vec<&str> = original.lines().collect();
    let ends_with_newline = original.is_empty() || original.ends_with('\n');

    let mut out: Vec<String> = Vec::new();
    let mut cursor = 0usize;

    for hunk in hunks {
        // Untouched lines before this hunk.
        while cursor < hunk.old_start && cursor < lines.len() {
            out.push(lines[cursor].to_string());
            cursor += 1;
        }

        if accepted.contains(&hunk.id) {
            out.extend(hunk.new_lines.iter().cloned());
        } else {
            let end = (hunk.old_start + hunk.old_len).min(lines.len());
            for line in &lines[cursor.min(end)..end] {
                out.push((*line).to_string());
            }
        }
        cursor = (hunk.old_start + hunk.old_len).max(cursor);
    }

    while cursor < lines.len() {
        out.push(lines[cursor].to_string());
        cursor += 1;
    }

    let mut text = out.join("\n");
    if ends_with_newline && !text.is_empty() {
        text.push('\n');
    }
    text
}

/// Render a set of diffs as a review block.
pub fn render(diffs: &[FileDiff]) -> String {
    if diffs.is_empty() {
        return "No changes proposed.".to_string();
    }

    let (add, rem): (usize, usize) = diffs
        .iter()
        .fold((0, 0), |(a, r), d| (a + d.added, r + d.removed));

    let mut out = format!("{} file(s) changed, +{} -{}\n\n", diffs.len(), add, rem);
    for d in diffs {
        out.push_str(&format!("  {}\n", d.stat()));
    }
    out.push('\n');
    for d in diffs {
        out.push_str(&d.unified);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_added_and_removed_lines() {
        let d = diff_file(
            Path::new("a.rs"),
            Some("one\ntwo\nthree\n"),
            "one\nTWO\nthree\nfour\n",
        );
        assert_eq!(d.added, 2);
        assert_eq!(d.removed, 1);
        assert!(d.existed);
        assert!(!d.is_empty());
    }

    #[test]
    fn a_new_file_reads_as_all_additions() {
        let d = diff_file(Path::new("new.rs"), None, "alpha\nbeta\n");
        assert_eq!(d.added, 2);
        assert_eq!(d.removed, 0);
        assert!(!d.existed);
        assert!(d.unified.contains("/dev/null"));
        assert!(d.stat().contains("(new)"));
    }

    #[test]
    fn identical_content_produces_an_empty_diff() {
        let d = diff_file(Path::new("a.rs"), Some("same\n"), "same\n");
        assert!(d.is_empty());
    }

    #[test]
    fn unified_output_carries_both_labels_and_the_change() {
        let d = diff_file(Path::new("src/lib.rs"), Some("old\n"), "new\n");
        assert!(d.unified.contains("a/src/lib.rs"));
        assert!(d.unified.contains("b/src/lib.rs"));
        assert!(d.unified.contains("-old"));
        assert!(d.unified.contains("+new"));
    }

    #[test]
    fn render_summarizes_then_details() {
        let diffs = vec![
            diff_file(Path::new("a.rs"), Some("x\n"), "y\n"),
            diff_file(Path::new("b.rs"), None, "z\n"),
        ];
        let r = render(&diffs);
        assert!(r.starts_with("2 file(s) changed"));
        assert!(r.contains("a.rs  +1 -1"));
        assert!(r.contains("b.rs  +1 -0 (new)"));
        assert!(r.contains("+z"));
    }

    #[test]
    fn render_handles_nothing_proposed() {
        assert_eq!(render(&[]), "No changes proposed.");
    }

    // ---- hunks ----

    const ORIGINAL: &str = "one\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\nten\n\
                            eleven\ntwelve\nthirteen\nfourteen\nfifteen\n";

    /// Two edits far enough apart to become separate hunks.
    fn two_hunk_diff() -> FileDiff {
        let after = "ONE\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\nten\n\
                     eleven\ntwelve\nthirteen\nfourteen\nFIFTEEN\n";
        diff_file(Path::new("a.rs"), Some(ORIGINAL), after)
    }

    #[test]
    fn distant_edits_become_separate_hunks() {
        let d = two_hunk_diff();
        assert_eq!(d.hunks.len(), 2, "got {} hunks", d.hunks.len());
        assert!(d.hunks[0].header.starts_with("@@"));
        assert_eq!(d.hunks[0].id, 0);
        assert_eq!(d.hunks[1].id, 1);
    }

    #[test]
    fn hunk_body_carries_prefixed_lines() {
        let d = two_hunk_diff();
        assert!(d.hunks[0].body.contains("-one"));
        assert!(d.hunks[0].body.contains("+ONE"));
        assert!(d.hunks[0].body.contains(" two"));
        assert_eq!(d.hunks[0].added, 1);
        assert_eq!(d.hunks[0].removed, 1);
    }

    #[test]
    fn accepting_every_hunk_reproduces_the_proposed_file() {
        let d = two_hunk_diff();
        let all: Vec<usize> = d.hunks.iter().map(|h| h.id).collect();
        let rebuilt = apply_hunks(ORIGINAL, &d.hunks, &all);
        assert!(rebuilt.starts_with("ONE\n"));
        assert!(rebuilt.ends_with("FIFTEEN\n"));
    }

    #[test]
    fn accepting_no_hunk_reproduces_the_original_exactly() {
        let d = two_hunk_diff();
        assert_eq!(apply_hunks(ORIGINAL, &d.hunks, &[]), ORIGINAL);
    }

    /// The whole point of per-hunk review.
    #[test]
    fn accepting_one_hunk_takes_only_that_change() {
        let d = two_hunk_diff();
        let rebuilt = apply_hunks(ORIGINAL, &d.hunks, &[0]);

        assert!(rebuilt.starts_with("ONE\n"), "first hunk should apply");
        assert!(rebuilt.contains("fifteen"), "second hunk should not");
        assert!(!rebuilt.contains("FIFTEEN"));

        let other = apply_hunks(ORIGINAL, &d.hunks, &[1]);
        assert!(other.starts_with("one\n"), "first hunk should not apply");
        assert!(other.contains("FIFTEEN"), "second hunk should");
    }

    #[test]
    fn a_new_file_is_a_single_hunk_of_pure_additions() {
        let d = diff_file(Path::new("new.rs"), None, "alpha\nbeta\n");
        assert_eq!(d.hunks.len(), 1);
        assert_eq!(d.hunks[0].removed, 0);
        assert_eq!(d.hunks[0].added, 2);
        assert_eq!(apply_hunks("", &d.hunks, &[0]), "alpha\nbeta\n");
        assert_eq!(apply_hunks("", &d.hunks, &[]), "");
    }

    #[test]
    fn a_file_without_a_trailing_newline_keeps_that_shape() {
        let before = "a\nb";
        let d = diff_file(Path::new("x.rs"), Some(before), "a\nB");
        let rebuilt = apply_hunks(before, &d.hunks, &[0]);
        assert_eq!(rebuilt, "a\nB");
        assert!(!rebuilt.ends_with('\n'));
    }

    #[test]
    fn hunk_spans_locate_the_original_lines_they_replace() {
        let d = two_hunk_diff();
        assert_eq!(d.hunks[0].old_start, 0, "first hunk starts at line 1");
        assert!(d.hunks[1].old_start > 0);
        assert!(d.hunks[0].old_len > 0);
    }
}
