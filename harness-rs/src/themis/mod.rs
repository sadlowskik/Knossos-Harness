//! Themis: the always-on shared expert.
//!
//! In the model architecture, Themis is the expert that runs for every token
//! regardless of what Apollo routed. That is the property being carried over
//! here, and it is the design decision this mapping forces: the constitution
//! is rendered into *every* engine call — planning and execution alike — not
//! attached to a "review" step at the end. A rule applied only at review time
//! is a rule the agent spent the whole task violating.
//!
//! The same text serves two roles, which is how the user specified it:
//! a behaviour guide that shapes generation, and the rubric Oracle's final
//! tier judges against.

use std::path::Path;

use crate::scribe::SymbolIndex;

/// Shipped default, used when the workspace has no `constitution.md`.
const DEFAULT: &str = include_str!("../../constitution.md");

/// How much of the symbol index to render into a prompt.
const SCRIBE_BUDGET: usize = 24_000;

#[derive(Debug, Clone)]
pub struct Themis {
    principles: String,
    source: String,
}

impl Themis {
    /// Load `constitution.md` from the workspace, falling back to the default.
    pub fn load(root: &Path) -> Self {
        let path = root.join("constitution.md");
        match std::fs::read_to_string(&path) {
            Ok(text) => Themis { principles: text, source: path.display().to_string() },
            Err(_) => Themis {
                principles: DEFAULT.to_string(),
                source: "built-in default".to_string(),
            },
        }
    }

    /// Build from literal text rather than a file. Used by tests and by
    /// callers that carry their own constitution.
    pub fn from_text(text: impl Into<String>) -> Self {
        Themis { principles: text.into(), source: "inline".to_string() }
    }

    pub fn principles(&self) -> &str {
        &self.principles
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    /// Build a system prompt for a given role.
    ///
    /// Scribe's symbols go in verbatim. They are the exact tier of memory: a
    /// summarized signature is a signature the model will confidently get
    /// wrong, so this text is never paraphrased or compressed by a model.
    pub fn system_prompt(&self, role: &str, scribe: Option<&SymbolIndex>) -> String {
        let mut s = String::new();
        s.push_str(role);
        s.push_str("\n\n# Constitution\n\nThese principles apply to everything you do.\n\n");
        s.push_str(&self.principles);

        if let Some(index) = scribe {
            if index.symbol_count() > 0 {
                s.push_str(&format!(
                    "\n\n# Symbol index (exact)\n\nParsed from source — these declarations are \
                     ground truth, not a summary. {} symbols across {} files. Do not use an \
                     identifier that is neither listed here nor created by your own change.\n",
                    index.symbol_count(),
                    index.file_count()
                ));
                s.push_str(&index.render(SCRIBE_BUDGET));
            }
        }
        s
    }
}

pub const PLANNER_ROLE: &str = "You are Metis, the planning half of the Daedalus harness. \
You break a coding task into a short ordered list of concrete steps. You do not write code \
and you do not call tools other than submitting the plan.";

pub const EXECUTOR_ROLE: &str = "You are Talos, the executing half of the Daedalus harness. \
You carry out a plan by reading and editing files in the workspace and running build and test \
commands. Work one step at a time. When the whole task is complete and verified, reply with a \
short summary and no tool calls.";

pub const JUDGE_ROLE: &str = "You are Oracle, the final verification tier of the Daedalus \
harness. The compiler, linter and test suite have already passed — do not re-check those. \
Judge only what they cannot: whether the change satisfies the constitution and actually \
accomplishes the stated task.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn falls_back_to_the_builtin_constitution() {
        let dir = tempfile::tempdir().unwrap();
        let t = Themis::load(dir.path());
        assert!(t.principles().contains("Verify, do not assert"));
        assert_eq!(t.source(), "built-in default");
    }

    #[test]
    fn a_workspace_constitution_overrides_the_default() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("constitution.md"), "# Local\n\nOnly rule: be brief.")
            .unwrap();
        let t = Themis::load(dir.path());
        assert!(t.principles().contains("be brief"));
        assert!(!t.principles().contains("Verify, do not assert"));
    }

    #[test]
    fn every_role_carries_the_constitution() {
        let t = Themis::from_text("RULE ONE");
        for role in [PLANNER_ROLE, EXECUTOR_ROLE, JUDGE_ROLE] {
            let p = t.system_prompt(role, None);
            assert!(p.contains("RULE ONE"), "constitution missing from a role prompt");
            assert!(p.contains(role));
        }
    }

    #[test]
    fn symbol_index_is_included_verbatim() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "pub fn exact_name() -> u32 { 1 }\n")
            .unwrap();
        let idx = SymbolIndex::build(dir.path()).unwrap();

        let p = Themis::from_text("x").system_prompt(EXECUTOR_ROLE, Some(&idx));
        assert!(p.contains("pub fn exact_name() -> u32"));
        assert!(p.contains("ground truth"));
    }

    #[test]
    fn an_empty_index_adds_no_symbol_section() {
        let dir = tempfile::tempdir().unwrap();
        let idx = SymbolIndex::build(dir.path()).unwrap();
        let p = Themis::from_text("x").system_prompt(EXECUTOR_ROLE, Some(&idx));
        assert!(!p.contains("Symbol index"));
    }
}
