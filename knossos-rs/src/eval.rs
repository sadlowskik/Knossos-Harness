//! Coding eval: a task is done only if the tests say so.
//!
//! Faithful port of `model/knossos/codeval.py` grading, not a new instrument.
//!
//! - `fail_to_pass` must go from failing to passing.
//! - `pass_to_pass` must still pass.
//! - `tests/` is restored before grading (`tamper` records that they changed).
//! - Node ids, not process exit codes. A deleted test is a failure.
//! - Held-out files are written only after the agent finishes.
//!
//! A case passes only if all three sets hold. Partial counts are reported
//! and do not make a case pass.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::Path;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::ariadne::Halt;
use crate::sandbox::Sandbox;

const TEST_TIMEOUT_SECS: u64 = 120;

#[derive(Debug, Clone)]
pub struct CodingCase {
    pub id: String,
    pub prompt: String,
    pub files: BTreeMap<String, String>,
    pub fail_to_pass: Vec<String>,
    pub pass_to_pass: Vec<String>,
    pub held_out: BTreeMap<String, String>,
    pub held_out_pass: Vec<String>,
    pub kind: String,
    pub tier: String,
    pub note: String,
    /// `edit` (default), `edit_preserve`, `no_op`, or `clarify`.
    pub expected_action: String,
}

impl CodingCase {
    /// Files restored before grading. Everything under `tests/`.
    pub fn test_files(&self) -> Vec<&str> {
        self.files
            .keys()
            .filter(|p| p.starts_with("tests/"))
            .map(String::as_str)
            .collect()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Grade {
    pub fixed: usize,
    pub fixed_total: usize,
    pub kept: usize,
    pub kept_total: usize,
    pub held: usize,
    pub held_total: usize,
    pub passed: bool,
    pub tamper: bool,
    pub timed_out: usize,
    pub unverifiable: usize,
    pub changed_files: usize,
    pub action_passed: bool,
}

impl Grade {
    /// Stable, machine-readable reason for a non-pass.
    pub fn failure_kind(&self) -> &'static str {
        if self.passed {
            "none"
        } else if self.tamper {
            "boundary_violation"
        } else if self.unverifiable > 0 {
            "infrastructure"
        } else if self.timed_out > 0 {
            "timeout"
        } else if !self.action_passed {
            "action_violation"
        } else {
            "test_failure"
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TestStatus {
    Passed,
    Failed,
    TimedOut,
    Unverifiable,
}

#[derive(Debug, Clone, Default)]
pub struct TestRun {
    pub nodes: BTreeMap<String, TestStatus>,
    pub diagnostics: Vec<String>,
}

/// The three-set rule, extracted so it can be tested without pytest.
pub fn verdict(
    fixed: usize,
    fixed_total: usize,
    kept: usize,
    kept_total: usize,
    held: usize,
    held_total: usize,
) -> bool {
    fixed == fixed_total && kept == kept_total && held == held_total
}

/// Check the behavioral contract independently from the test verdict.
pub fn action_verdict(
    expected_action: &str,
    changed_files: usize,
    agent_response: Option<&str>,
) -> bool {
    match expected_action {
        "no_op" => changed_files == 0,
        "edit_preserve" => changed_files > 0,
        "clarify" => {
            changed_files == 0
                && agent_response.is_some_and(|text| text.contains('?') && !text.trim().is_empty())
        }
        "edit" => true,
        _ => false,
    }
}

/// A green repository is not enough: the agent must itself finish the task.
/// Clarification is the one exception because an unattended eval cannot answer
/// the question, so a valid question is the successful terminal behavior.
pub fn completion_verdict(expected_action: &str, halt: Halt, action_passed: bool) -> bool {
    halt == Halt::Done || (expected_action == "clarify" && action_passed)
}

#[derive(Deserialize)]
struct RawCase {
    id: String,
    prompt: String,
    files: BTreeMap<String, String>,
    fail_to_pass: Vec<String>,
    #[serde(default)]
    pass_to_pass: Vec<String>,
    #[serde(default)]
    held_out: BTreeMap<String, String>,
    #[serde(default)]
    held_out_pass: Vec<String>,
    #[serde(default)]
    kind: String,
    #[serde(default)]
    tier: String,
    #[serde(default)]
    note: String,
    #[serde(default = "default_expected_action")]
    expected_action: String,
}

fn default_expected_action() -> String {
    "edit".into()
}

/// Read cases from JSON. Same schema as Python `load_cases`.
pub fn load_cases(path: &Path) -> Result<Vec<CodingCase>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)?;
    let list = if value.is_array() {
        value
    } else {
        value.get("cases").cloned().ok_or_else(|| {
            anyhow::anyhow!("{}: expected a list or {{cases: [...]}}", path.display())
        })?
    };
    let raw: Vec<RawCase> = serde_json::from_value(list)?;
    if raw.is_empty() {
        bail!("{}: expected a non-empty list of cases", path.display());
    }
    let mut cases = Vec::new();
    let mut seen = BTreeMap::new();
    for (index, entry) in raw.into_iter().enumerate() {
        let where_ = format!("{}[{index}]", path.display());
        if entry.id.is_empty() || entry.prompt.is_empty() || entry.files.is_empty() {
            bail!("{where_}: missing id, prompt, or files");
        }
        if entry.fail_to_pass.is_empty() && entry.expected_action == "edit" {
            bail!("{where_}: edit cases need at least one fail_to_pass node");
        }
        if !matches!(
            entry.expected_action.as_str(),
            "edit" | "edit_preserve" | "no_op" | "clarify"
        ) {
            bail!(
                "{where_}: unknown expected_action {}",
                entry.expected_action
            );
        }
        if !entry.held_out_pass.is_empty() && entry.held_out.is_empty() {
            bail!("{where_}: `held_out_pass` without `held_out` files");
        }
        let clash: Vec<_> = entry
            .held_out
            .keys()
            .filter(|k| entry.files.contains_key(k.as_str()))
            .cloned()
            .collect();
        if !clash.is_empty() {
            bail!(
                "{where_}: `held_out` may not overwrite visible files: {}",
                clash.join(", ")
            );
        }
        if seen.insert(entry.id.clone(), ()).is_some() {
            bail!("{}: duplicate case id {}", path.display(), entry.id);
        }
        cases.push(CodingCase {
            kind: if entry.kind.is_empty() {
                "bugfix".into()
            } else {
                entry.kind
            },
            tier: if entry.tier.is_empty() {
                "core".into()
            } else {
                entry.tier
            },
            id: entry.id,
            prompt: entry.prompt,
            files: entry.files,
            fail_to_pass: entry.fail_to_pass,
            pass_to_pass: entry.pass_to_pass,
            held_out: entry.held_out,
            held_out_pass: entry.held_out_pass,
            note: entry.note,
            expected_action: entry.expected_action,
        });
    }
    Ok(cases)
}

/// Bundled core suite, compiled in so `knossos eval` works without a path.
pub fn bundled_core() -> Result<Vec<CodingCase>> {
    let value: serde_json::Value = serde_json::from_str(include_str!("../cases/core.json"))?;
    let tmp = std::env::temp_dir().join("knossos-bundled-core.json");
    std::fs::write(&tmp, serde_json::to_vec(&value)?)?;
    load_cases(&tmp)
}

pub fn materialise(case: &CodingCase, root: &Path) -> Result<()> {
    std::fs::create_dir_all(root)?;
    for (rel, content) in &case.files {
        let path = root.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, content)?;
    }
    Ok(())
}

/// Put fixture test files back. True if any had changed (`tamper`).
pub fn restore_tests(case: &CodingCase, root: &Path) -> Result<bool> {
    let mut tampered = false;
    for rel in case.test_files() {
        let path = root.join(rel);
        let original = &case.files[rel];
        let current = std::fs::read_to_string(&path).ok();
        if current.as_deref() != Some(original.as_str()) {
            tampered = true;
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, original)?;
        }
    }
    Ok(tampered)
}

pub fn reveal_held_out(case: &CodingCase, root: &Path) -> Result<()> {
    for (rel, content) in &case.held_out {
        let path = root.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, content)?;
    }
    Ok(())
}

async fn pytest(root: &Path, node_ids: &[String]) -> (TestStatus, String) {
    if node_ids.is_empty() {
        return (TestStatus::Passed, String::new());
    }
    let (py, prefix) = crate::scribe::python::python_command();
    let mut args: Vec<OsString> = prefix.into_iter().map(OsString::from).collect();
    args.extend(["-m", "pytest"].into_iter().map(OsString::from));
    args.extend(node_ids.iter().map(OsString::from));
    args.extend(
        ["-q", "--no-header", "-rA", "-p", "no:cacheprovider"]
            .into_iter()
            .map(OsString::from),
    );
    match Sandbox::default()
        .run_bounded(&py, &args, root, Duration::from_secs(TEST_TIMEOUT_SECS))
        .await
    {
        Ok(out) if out.timed_out => (
            TestStatus::TimedOut,
            format!(
                "pytest timed out after {TEST_TIMEOUT_SECS}s\n{}\n{}",
                out.stdout, out.stderr
            ),
        ),
        Ok(out) => {
            let detail = format!("pytest exit={}\n{}\n{}", out.code(), out.stdout, out.stderr);
            let report = format!("{}\n{}", out.stdout, out.stderr).to_ascii_lowercase();
            // Pytest exits zero when every selected node was skipped. That is
            // not evidence that the requested behaviour works, so skips and
            // expected failures are deliberately non-passes.
            let non_pass = [" skipped", " xfailed", " deselected"]
                .iter()
                .any(|marker| report.contains(marker));
            if out.success() && !non_pass {
                (TestStatus::Passed, detail)
            } else {
                (TestStatus::Failed, detail)
            }
        }
        Err(error) => (
            TestStatus::Unverifiable,
            format!("could not launch pytest with {py}: {error}"),
        ),
    }
}

/// Run node ids; batch first, then one process per node if the batch is not green.
pub async fn run_tests(root: &Path, node_ids: &[String]) -> TestRun {
    if node_ids.is_empty() {
        return TestRun::default();
    }
    let (batch, batch_detail) = pytest(root, node_ids).await;
    if batch == TestStatus::Passed {
        return TestRun {
            nodes: node_ids
                .iter()
                .cloned()
                .map(|n| (n, TestStatus::Passed))
                .collect(),
            diagnostics: Vec::new(),
        };
    }
    if matches!(batch, TestStatus::TimedOut | TestStatus::Unverifiable) {
        return TestRun {
            nodes: node_ids.iter().cloned().map(|n| (n, batch)).collect(),
            diagnostics: vec![batch_detail],
        };
    }

    let mut run = TestRun {
        nodes: BTreeMap::new(),
        diagnostics: vec![batch_detail],
    };
    for node in node_ids {
        let (status, detail) = pytest(root, std::slice::from_ref(node)).await;
        if status != TestStatus::Passed {
            run.diagnostics.push(format!("{node}: {detail}"));
        }
        run.nodes.insert(node.clone(), status);
    }
    run
}

/// Restore tests, run both sets, reveal held-out, decide.
pub async fn grade(
    case: &CodingCase,
    root: &Path,
    tampered: bool,
    agent_response: Option<&str>,
) -> Result<Grade> {
    let fixed = run_tests(root, &case.fail_to_pass).await;
    let kept = run_tests(root, &case.pass_to_pass).await;
    reveal_held_out(case, root)?;
    let held = run_tests(root, &case.held_out_pass).await;
    let all = fixed
        .nodes
        .values()
        .chain(kept.nodes.values())
        .chain(held.nodes.values());
    let n_fixed = fixed
        .nodes
        .values()
        .filter(|v| **v == TestStatus::Passed)
        .count();
    let n_kept = kept
        .nodes
        .values()
        .filter(|v| **v == TestStatus::Passed)
        .count();
    let n_held = held
        .nodes
        .values()
        .filter(|v| **v == TestStatus::Passed)
        .count();
    let timed_out = all.clone().filter(|v| **v == TestStatus::TimedOut).count();
    let unverifiable = all.filter(|v| **v == TestStatus::Unverifiable).count();
    let changed_files = case
        .files
        .iter()
        .filter(|(rel, original)| {
            !rel.starts_with("tests/")
                && std::fs::read_to_string(root.join(rel)).ok().as_deref()
                    != Some(original.as_str())
        })
        .count();
    let tests_passed = verdict(
        n_fixed,
        case.fail_to_pass.len(),
        n_kept,
        case.pass_to_pass.len(),
        n_held,
        case.held_out_pass.len(),
    );
    let action_passed = action_verdict(&case.expected_action, changed_files, agent_response);
    Ok(Grade {
        passed: !tampered && tests_passed && action_passed,
        fixed: n_fixed,
        fixed_total: case.fail_to_pass.len(),
        kept: n_kept,
        kept_total: case.pass_to_pass.len(),
        held: n_held,
        held_total: case.held_out_pass.len(),
        tamper: tampered,
        timed_out,
        unverifiable,
        changed_files,
        action_passed,
    })
}

/// Materialise, let `run` edit the tree, restore tests, grade.
pub async fn run_case<F>(case: &CodingCase, root: &Path, run: F) -> Result<Grade>
where
    F: FnOnce(&Path) -> Result<()>,
{
    materialise(case, root)?;
    run(root)?;
    let tampered = restore_tests(case, root)?;
    grade(case, root, tampered, None).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fix_that_breaks_the_suite_is_not_a_pass() {
        assert!(!verdict(1, 1, 0, 1, 0, 0));
    }

    #[test]
    fn all_three_sets_must_hold() {
        assert!(verdict(2, 2, 1, 1, 0, 0));
        assert!(!verdict(2, 2, 1, 1, 0, 1));
        assert!(verdict(1, 1, 0, 0, 1, 1));
    }

    #[test]
    fn action_contracts_reward_restraint_and_questions() {
        assert!(action_verdict("no_op", 0, None));
        assert!(!action_verdict("no_op", 1, None));
        assert!(action_verdict("edit_preserve", 2, None));
        assert!(!action_verdict("edit_preserve", 0, None));
        assert!(action_verdict(
            "clarify",
            0,
            Some("Which format should I use?")
        ));
        assert!(!action_verdict("clarify", 1, Some("Which format?")));
        assert!(!action_verdict("clarify", 0, Some("I assumed JSON.")));
        assert!(action_verdict("edit", 0, None));
        assert!(!action_verdict("unknown", 0, None));
    }

    #[test]
    fn a_green_starting_tree_does_not_rescue_a_stuck_agent() {
        assert!(!completion_verdict("no_op", Halt::Stuck, true));
        assert!(!completion_verdict("edit", Halt::BudgetExhausted, true));
        assert!(completion_verdict("no_op", Halt::Done, true));
        assert!(completion_verdict("clarify", Halt::Stuck, true));
        assert!(!completion_verdict("clarify", Halt::Stuck, false));
    }

    #[test]
    fn action_violation_has_a_distinct_failure_kind() {
        let grade = Grade {
            action_passed: false,
            ..Grade::default()
        };
        assert_eq!(grade.failure_kind(), "action_violation");
    }

    #[test]
    fn restore_tests_overwrites_tampering() {
        let dir = tempfile::tempdir().unwrap();
        let mut files = BTreeMap::new();
        files.insert("src/lib.py".into(), "x = 1\n".into());
        files.insert(
            "tests/test_x.py".into(),
            "def test_ok():\n    assert True\n".into(),
        );
        let case = CodingCase {
            id: "t".into(),
            prompt: "fix".into(),
            files,
            fail_to_pass: vec!["tests/test_x.py::test_ok".into()],
            pass_to_pass: vec![],
            held_out: BTreeMap::new(),
            held_out_pass: vec![],
            kind: "bugfix".into(),
            tier: "core".into(),
            note: String::new(),
            expected_action: "edit".into(),
        };
        materialise(&case, dir.path()).unwrap();
        std::fs::write(
            dir.path().join("tests/test_x.py"),
            "def test_ok():\n    assert False\n",
        )
        .unwrap();
        let tampered = restore_tests(&case, dir.path()).unwrap();
        assert!(tampered);
        let body = std::fs::read_to_string(dir.path().join("tests/test_x.py")).unwrap();
        assert!(body.contains("assert True"));
    }

    #[test]
    fn load_cases_rejects_an_empty_list() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.json");
        std::fs::write(&path, "[]").unwrap();
        assert!(load_cases(&path).is_err());
    }

    #[test]
    fn load_cases_rejects_held_out_clobbering_visible_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.json");
        std::fs::write(
            &path,
            r#"[{
                "id": "x",
                "prompt": "p",
                "files": {"tests/a.py": "a"},
                "fail_to_pass": ["tests/a.py::t"],
                "held_out": {"tests/a.py": "b"}
            }]"#,
        )
        .unwrap();
        let err = load_cases(&path).unwrap_err().to_string();
        assert!(err.contains("held_out"), "{err}");
    }

    #[test]
    fn held_out_files_are_not_on_disk_until_revealed() {
        let dir = tempfile::tempdir().unwrap();
        let mut files = BTreeMap::new();
        files.insert("src/a.py".into(), "x=1\n".into());
        let mut held = BTreeMap::new();
        held.insert(
            "tests/hidden.py".into(),
            "def test_h():\n    assert True\n".into(),
        );
        let case = CodingCase {
            id: "h".into(),
            prompt: "p".into(),
            files,
            fail_to_pass: vec!["x".into()],
            pass_to_pass: vec![],
            held_out: held,
            held_out_pass: vec!["tests/hidden.py::test_h".into()],
            kind: "bugfix".into(),
            tier: "core".into(),
            note: String::new(),
            expected_action: "edit".into(),
        };
        materialise(&case, dir.path()).unwrap();
        assert!(!dir.path().join("tests/hidden.py").exists());
        reveal_held_out(&case, dir.path()).unwrap();
        assert!(dir.path().join("tests/hidden.py").exists());
    }
}
