//! Policy that runs around every tool call.
//!
//! The guardrails in [`tools`](crate::tools) are structural: they hold for
//! every workspace because they are properties of the code. A hook is the other
//! kind of rule — the one that is true for *this* repository, or this run, and
//! that the harness has no business hard-coding. "Do not touch the migrations
//! directory" is a real constraint and a bad constant.
//!
//! Hooks sit inside [`ToolRegistry::dispatch`](crate::tools::ToolRegistry) so
//! that every caller gets them. Talos is not the only thing that dispatches
//! tools, and a policy that only applies on one path is not a policy.
//!
//! # Why `after` cannot change a result
//!
//! [`Hook::before`] can deny or rewrite; [`Hook::after`] can only observe. That
//! asymmetry is deliberate. A denied call is visible to the engine as an error
//! and it adapts — that is an ordinary failure it already knows how to handle.
//! A *rewritten result* is a lie: the engine is told a file contains something
//! it does not, and every subsequent step reasons from it. Refusing to build
//! that road at all is cheaper than policing who walks down it.

use std::borrow::Cow;
use std::path::Path;

use crate::tools::ToolOutput;

/// What a hook decided about a call it was shown.
#[derive(Debug, Clone)]
pub enum Decision {
    /// Run it unchanged.
    Allow,
    /// Run it against different input.
    Rewrite(serde_json::Value),
    /// Do not run it. The reason reaches the engine as the tool's error output.
    Deny(String),
}

pub trait Hook: Send + Sync {
    /// Named so a denial can say what stopped it. An agent told only "denied"
    /// will retry the same call; one told which rule denied it will not.
    fn name(&self) -> &str;

    /// Runs before the tool, in registration order. The first [`Decision::Deny`]
    /// wins and the hooks after it do not run.
    fn before(&self, _tool: &str, _input: &serde_json::Value) -> Decision {
        Decision::Allow
    }

    /// Runs after **every** dispatched call, in registration order.
    ///
    /// Every call means every call: one that ran, one a hook denied, one whose
    /// tool name does not exist, and one whose tool returned an error. A hook
    /// that sees only successful calls cannot be used to audit anything, since
    /// the interesting cases are exactly the ones that went wrong.
    ///
    /// `input` is the call as the `before` chain left it: rewritten if a hook
    /// rewrote it, original otherwise. On a denial it is still the rewritten
    /// value, because the rewritten call is the one that was attempted.
    ///
    /// Observation only; it cannot change the result. See the module docs.
    fn after(&self, _tool: &str, _input: &serde_json::Value, _out: &ToolOutput) {}
}

/// Refuse changes to paths that are inside the workspace but not the work.
///
/// The path jail stops at the workspace boundary, which is correct and is not
/// the same as saying everything inside is fair game. `.git` is the sharpest
/// case: it sits under the root, so [`ToolCtx::resolve`](crate::tools::ToolCtx)
/// admits it, and `write_file` to `.git/config` would therefore be allowed —
/// while [`shell`](crate::tools::shell) goes to the trouble of restricting
/// `git` to read-only subcommands. The intent there is plain, and the
/// filesystem tools route straight around it.
///
/// Matching is on path components, not substrings: `.git` protects `.git/` and
/// everything under it without also catching a file named `mygit.rs`.
pub struct ProtectPaths {
    components: Vec<String>,
}

impl Default for ProtectPaths {
    fn default() -> Self {
        ProtectPaths::new([".git", ".knossos"])
    }
}

impl ProtectPaths {
    pub fn new<I, S>(components: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        ProtectPaths {
            components: components.into_iter().map(Into::into).collect(),
        }
    }

    /// The protected component this path falls under, if any.
    fn violated(&self, path: &str) -> Option<&str> {
        let p = Path::new(path);
        self.components.iter().find_map(|c| {
            p.components()
                .any(|part| part.as_os_str().eq_ignore_ascii_case(c.as_str()))
                .then_some(c.as_str())
        })
    }
}

impl Hook for ProtectPaths {
    fn name(&self) -> &str {
        "protect-paths"
    }

    fn before(&self, tool: &str, input: &serde_json::Value) -> Decision {
        // Reads are not the concern: knowing what is in `.git/config` is
        // occasionally useful and never destructive. Only writes are stopped,
        // so the rule costs the agent nothing it legitimately needs.
        if !matches!(tool, "write_file" | "edit_file") {
            return Decision::Allow;
        }

        let Some(path) = input.get("path").and_then(|v| v.as_str()) else {
            return Decision::Allow;
        };

        match self.violated(path) {
            Some(component) => Decision::Deny(format!(
                "`{component}` is protected; {path} is not part of the working tree"
            )),
            None => Decision::Allow,
        }
    }
}

/// What the `before` chain decided.
pub(crate) struct Chained<'a> {
    /// The input as the chain left it, whether or not the call went ahead.
    ///
    /// Carried even on a denial, and that is the point: if one hook rewrites a
    /// path and a later one refuses it, the call that was *attempted* is the
    /// rewritten one. Reporting the original to [`Hook::after`] would make an
    /// audit log describe something nobody tried to do.
    pub input: Cow<'a, serde_json::Value>,
    /// Set when a hook refused the call.
    pub denial: Option<String>,
}

/// Run the `before` chain.
///
/// Separated from `dispatch` so the ordering rules — first denial wins, a
/// rewrite is visible to the hooks after it — can be tested without a
/// filesystem or a tool.
pub(crate) fn before_chain<'a>(
    hooks: &[Box<dyn Hook>],
    tool: &str,
    input: &'a serde_json::Value,
) -> Chained<'a> {
    let mut current = Cow::Borrowed(input);
    for hook in hooks {
        match hook.before(tool, &current) {
            Decision::Allow => {}
            Decision::Rewrite(v) => current = Cow::Owned(v),
            Decision::Deny(reason) => {
                return Chained {
                    input: current,
                    denial: Some(format!("{tool} was refused by `{}`: {reason}", hook.name())),
                }
            }
        }
    }
    Chained {
        input: current,
        denial: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn writes_into_the_git_directory_are_refused() {
        let h = ProtectPaths::default();
        let d = h.before("write_file", &json!({"path": ".git/config"}));
        assert!(matches!(d, Decision::Deny(_)), "{d:?}");
    }

    #[test]
    fn writes_into_the_episode_store_are_refused() {
        let h = ProtectPaths::default();
        let d = h.before("write_file", &json!({"path": ".knossos/episodes.jsonl"}));
        assert!(matches!(d, Decision::Deny(_)), "{d:?}");
    }

    #[test]
    fn the_working_tree_is_untouched_by_the_rule() {
        let h = ProtectPaths::default();
        assert!(matches!(
            h.before("write_file", &json!({"path": "src/main.rs"})),
            Decision::Allow
        ));
        // A substring match would have caught this one.
        assert!(matches!(
            h.before("write_file", &json!({"path": "src/mygit.rs"})),
            Decision::Allow
        ));
        assert!(matches!(
            h.before("write_file", &json!({"path": "gitignore.rs"})),
            Decision::Allow
        ));
    }

    #[test]
    fn reads_are_never_blocked() {
        let h = ProtectPaths::default();
        assert!(matches!(
            h.before("read_file", &json!({"path": ".git/config"})),
            Decision::Allow
        ));
        assert!(matches!(
            h.before("list_dir", &json!({"path": ".git"})),
            Decision::Allow
        ));
    }

    #[test]
    fn nested_paths_under_a_protected_component_are_caught() {
        let h = ProtectPaths::default();
        let d = h.before("edit_file", &json!({"path": "a/b/.git/hooks/pre-commit"}));
        assert!(matches!(d, Decision::Deny(_)), "{d:?}");
    }

    #[test]
    fn the_protected_set_is_configurable() {
        let h = ProtectPaths::new(["migrations", "vendor"]);
        assert!(matches!(
            h.before("write_file", &json!({"path": "migrations/003.sql"})),
            Decision::Deny(_)
        ));
        assert!(matches!(
            h.before("write_file", &json!({"path": ".git/config"})),
            Decision::Allow
        ));
    }

    struct Rewriter;
    impl Hook for Rewriter {
        fn name(&self) -> &str {
            "rewriter"
        }
        fn before(&self, _t: &str, _i: &serde_json::Value) -> Decision {
            Decision::Rewrite(json!({"path": ".git/config"}))
        }
    }

    struct Blocker;
    impl Hook for Blocker {
        fn name(&self) -> &str {
            "blocker"
        }
        fn before(&self, _t: &str, _i: &serde_json::Value) -> Decision {
            Decision::Deny("no".into())
        }
    }

    #[test]
    fn a_rewrite_is_visible_to_the_hooks_that_follow_it() {
        // Otherwise a rewrite could be used to smuggle a path past a later
        // policy hook, which would make hook order a security boundary.
        let hooks: Vec<Box<dyn Hook>> = vec![Box::new(Rewriter), Box::new(ProtectPaths::default())];
        let input = json!({"path": "src/ok.rs"});
        let out = before_chain(&hooks, "write_file", &input);
        assert!(
            out.denial.is_some(),
            "the rewritten path should have been judged"
        );
    }

    /// What an audit log has to say about a call that was rewritten and then
    /// refused: the attempt was on the rewritten path, not the original.
    #[test]
    fn a_denial_reports_the_call_that_was_actually_attempted() {
        let hooks: Vec<Box<dyn Hook>> = vec![Box::new(Rewriter), Box::new(ProtectPaths::default())];
        let input = json!({"path": "src/harmless.rs"});
        let out = before_chain(&hooks, "write_file", &input);

        assert!(out.denial.is_some());
        assert_eq!(
            out.input["path"], ".git/config",
            "reporting the original would describe an attempt nobody made"
        );
    }

    #[test]
    fn the_first_denial_wins_and_names_itself() {
        let hooks: Vec<Box<dyn Hook>> = vec![Box::new(Blocker), Box::new(Rewriter)];
        let input = json!({});
        let out = before_chain(&hooks, "write_file", &input);
        let denial = out.denial.expect("should have been denied");
        assert!(denial.contains("blocker"), "{denial}");
    }

    #[test]
    fn an_empty_chain_passes_the_input_through_untouched() {
        let hooks: Vec<Box<dyn Hook>> = vec![];
        let input = json!({"path": "a.rs"});
        let out = before_chain(&hooks, "write_file", &input);
        assert!(out.denial.is_none());
        assert_eq!(*out.input, input);
    }
}
