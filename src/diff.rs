//! Unified diffs for previewing changes before they touch disk.

use std::path::{Path, PathBuf};

use similar::{ChangeTag, TextDiff};

#[derive(Debug, Clone)]
pub struct FileDiff {
    pub path: PathBuf,
    /// `None` when the file did not exist before.
    pub existed: bool,
    pub unified: String,
    pub added: usize,
    pub removed: usize,
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
        .context_radius(3)
        .header(
            &if existed { format!("a/{label}") } else { "/dev/null".to_string() },
            &format!("b/{label}"),
        )
        .to_string();

    FileDiff { path: path.to_path_buf(), existed, unified, added, removed }
}

/// Render a set of diffs as a review block.
pub fn render(diffs: &[FileDiff]) -> String {
    if diffs.is_empty() {
        return "No changes proposed.".to_string();
    }

    let (add, rem): (usize, usize) = diffs
        .iter()
        .fold((0, 0), |(a, r), d| (a + d.added, r + d.removed));

    let mut out = format!(
        "{} file(s) changed, +{} -{}\n\n",
        diffs.len(),
        add,
        rem
    );
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
}
