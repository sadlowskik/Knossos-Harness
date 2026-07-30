//! Delegation: run one scoped subtask in a child agent, keep only its summary.
//!
//! # What this is for
//!
//! **Not speed.** The child runs to completion before the parent's turn
//! continues, so nothing here is concurrent — and it must stay that way while
//! the workspace is shared, because two children writing to one staging area
//! would interleave edits to the same file.
//!
//! It is for **context**. The expensive part of a subtask is usually the
//! reading: ten files opened to discover that one function needed changing. In
//! a single agent all ten land in the transcript and are re-sent every
//! subsequent step, which is what makes a long run cost quadratically. A child
//! reads them into its own transcript, which is discarded when it finishes, and
//! hands back a paragraph. The parent pays for the paragraph.
//!
//! # What a child shares, and why
//!
//! **The workspace, by reference.** [`ToolCtx`] clones share their staging area
//! and undo journal, so a child's edits are ordinary edits — visible to the
//! parent's verifier, revertible by the parent's checkpoints, and counted in the
//! parent's change set. The alternative, a private copy merged afterwards, means
//! writing a merge algorithm and being wrong about conflicts.
//!
//! **The permission callback, by reference.** A child must not be a way around a
//! gate the user is watching: `write_file` from a child prompts exactly as it
//! would from the parent.
//!
//! **Not the transcript.** That is the entire point.
//!
//! # Why the child is built by a closure
//!
//! Python could spawn from inside `Talos` because `_spawn` was a method with
//! everything already in scope. A [`Tool`] here sees only its input and a
//! [`ToolCtx`], and assembling a `Talos` needs an engine, a constitution, a
//! symbol index and a verifier. So the thing that knows how to build one — the
//! caller that built the parent — supplies a [`SpawnChild`]. That also keeps the
//! decision about *what tools a role gets* where the tools are chosen, rather
//! than hidden in here.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::metis::Plan;
use crate::talos::Talos;
use crate::tools::{Tool, ToolCtx, ToolOutput};

/// The name of the delegation tool, kept here so a child's registry can be
/// built without it.
pub const DELEGATE: &str = "delegate";

/// How deep delegation may go. One, deliberately.
///
/// A child that can delegate is a fork bomb with a token budget: each level
/// multiplies the number of live agents, and the failure is not a crash but a
/// bill. Depth one buys the thing worth having — an independent subtask whose
/// reading does not land in the parent's context — and nothing beyond it has
/// paid for itself in any harness the author is aware of.
pub const MAX_DEPTH: usize = 1;

/// Floor on a child's step budget.
///
/// Halving the parent's budget is the rule, but a child with one step cannot
/// read anything and then act on it, so it would fail for a reason that has
/// nothing to do with the task.
pub const MIN_CHILD_STEPS: usize = 3;

/// One kind of child agent.
///
/// Data rather than an enum so a new speciality is configuration, not a code
/// change — and so the set can differ between a CLI run and an editor session.
#[derive(Debug, Clone)]
pub struct Role {
    /// What the parent selects it by.
    pub name: String,
    /// Shown to the parent in the tool schema, so the model can choose.
    pub description: String,
    /// The system role handed to the child, composed with the constitution by
    /// [`Themis::system_prompt`](crate::themis::Themis::system_prompt).
    pub prompt: String,
}

impl Role {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        prompt: impl Into<String>,
    ) -> Self {
        Role {
            name: name.into(),
            description: description.into(),
            prompt: prompt.into(),
        }
    }

    /// The default set: a generalist, plus the two specialities whose value is
    /// that they *cannot* write.
    ///
    /// A reviewer that can edit will fix what it finds and report success, which
    /// destroys the only thing a separate reviewer was for — an opinion formed
    /// without a stake in the work. Enforcing that is the caller's job, in the
    /// registry it hands the child; naming it here is what makes the caller
    /// remember to.
    pub fn defaults() -> Vec<Role> {
        vec![
            Role::new(
                "general",
                "A capable generalist. Use when the subtask is ordinary work.",
                "You are a Daedalus subagent handling one scoped subtask. Do exactly what \
                 you were asked and nothing adjacent to it. Report what you changed and \
                 what you could not.",
            ),
            Role::new(
                "reviewer",
                "Reads and critiques without editing. Use to get an opinion on work \
                 already done.",
                "You are a Daedalus review subagent. You read and judge; you do not edit. \
                 Report what is wrong, where, and why it matters. Say plainly when \
                 something is fine — a review that invents problems is worse than none.",
            ),
            Role::new(
                "investigator",
                "Searches and reads to answer a question. Use when finding the answer \
                 costs more reading than the answer is worth remembering.",
                "You are a Daedalus investigation subagent. Find the answer and report it \
                 with exact file and line references. Do not change anything. If the \
                 answer is not in the workspace, say so rather than guessing.",
            ),
        ]
    }
}

/// What the parent is asking for. Passed to the caller's factory.
pub struct ChildRequest {
    /// The subtask, as the parent wrote it.
    pub task: String,
    /// The chosen role. Whoever builds the child decides what tools it gets.
    pub role: Role,
    /// How far down the chain this child sits. Zero is the top-level agent, so
    /// a child is always at least one.
    pub depth: usize,
    /// Already halved and floored — see [`MIN_CHILD_STEPS`].
    pub max_steps: usize,
    /// The parent's context, cloned. Shares the staging area and undo journal.
    pub ctx: ToolCtx,
}

/// Builds a child agent. Supplied by whoever built the parent.
///
/// Returning an error fails the delegation rather than the run: a parent that
/// cannot spawn should be told so and carry on itself.
pub type SpawnChild = dyn Fn(ChildRequest) -> Result<Talos> + Send + Sync;

/// Hand one self-contained subtask to a child agent.
pub struct Delegate {
    spawn: Arc<SpawnChild>,
    roles: Vec<Role>,
    /// The *parent's* depth. A child is spawned at `depth + 1`.
    depth: usize,
    /// The parent's ceiling, which the child's budget is derived from.
    parent_max_steps: usize,
}

impl Delegate {
    pub fn new(spawn: Arc<SpawnChild>, parent_max_steps: usize) -> Self {
        Delegate {
            spawn,
            roles: Role::defaults(),
            depth: 0,
            parent_max_steps,
        }
    }

    pub fn with_roles(mut self, roles: Vec<Role>) -> Self {
        self.roles = roles;
        self
    }

    /// Whether an agent at `depth` may be given this tool at all.
    ///
    /// Checked by the caller when building a registry. The tool also refuses at
    /// dispatch, but that is a backstop: a child that can *see* a delegate tool
    /// will try to use it and waste a step discovering it cannot.
    pub fn allowed_at(depth: usize) -> bool {
        depth < MAX_DEPTH
    }

    /// The budget a child of this parent gets.
    pub fn child_steps(parent_max_steps: usize) -> usize {
        (parent_max_steps / 2).max(MIN_CHILD_STEPS)
    }

    fn role(&self, requested: Option<&str>) -> Result<Role, String> {
        match requested {
            None => self
                .roles
                .first()
                .cloned()
                .ok_or_else(|| "no roles are configured".to_string()),
            Some(name) => self
                .roles
                .iter()
                .find(|r| r.name == name)
                .cloned()
                .ok_or_else(|| {
                    let known: Vec<&str> = self.roles.iter().map(|r| r.name.as_str()).collect();
                    format!("unknown role `{name}`; available: {}", known.join(", "))
                }),
        }
    }
}

#[async_trait]
impl Tool for Delegate {
    fn name(&self) -> &str {
        DELEGATE
    }

    fn description(&self) -> &str {
        "Hand one self-contained subtask to a child agent that works in the same \
         workspace but keeps its own context, and reports back a summary. Use it when a \
         piece of work needs a lot of reading you do not need to remember afterwards — \
         locating something across many files, or a mechanical change in a part of the \
         tree you are not otherwise touching. The child cannot delegate further and \
         cannot ask you questions, so give it everything it needs in the task."
    }

    fn schema(&self) -> Value {
        let names: Vec<&str> = self.roles.iter().map(|r| r.name.as_str()).collect();
        let catalogue: Vec<String> = self
            .roles
            .iter()
            .map(|r| format!("{}: {}", r.name, r.description))
            .collect();
        json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The complete subtask. The child sees none of this \
                                    conversation, so it must stand alone.",
                },
                "role": {
                    "type": "string",
                    "enum": names,
                    "description": catalogue.join(" | "),
                },
            },
            "required": ["task"],
        })
    }

    /// A child can write, so the parent's permission gate applies to the
    /// delegation itself.
    ///
    /// The child's own calls prompt as well — it shares the callback — but the
    /// parent asking to spawn one is its own decision to approve, and it is the
    /// only point at which the whole subtask can still be declined.
    fn consequential(&self) -> bool {
        true
    }

    async fn run(&self, input: &Value, ctx: &ToolCtx) -> Result<ToolOutput> {
        if !Self::allowed_at(self.depth) {
            return Ok(ToolOutput::error(format!(
                "delegation is limited to depth {MAX_DEPTH}; do this subtask yourself"
            )));
        }

        let Some(task) = input.get("task").and_then(Value::as_str).filter(|t| !t.trim().is_empty())
        else {
            return Ok(ToolOutput::error("`task` is required and must not be empty"));
        };

        let role = match self.role(input.get("role").and_then(Value::as_str)) {
            Ok(role) => role,
            Err(why) => return Ok(ToolOutput::error(why)),
        };

        let request = ChildRequest {
            task: task.to_string(),
            role: role.clone(),
            depth: self.depth + 1,
            max_steps: Self::child_steps(self.parent_max_steps),
            // Cloned, so the child stages into the parent's workspace.
            ctx: ctx.clone(),
        };

        let mut child = match (self.spawn)(request) {
            Ok(child) => child,
            Err(err) => {
                return Ok(ToolOutput::error(format!("could not start a child agent: {err}")))
            }
        };

        // A child gets the subtask as its whole plan. Planning it again would
        // spend one of its few steps restating what the parent already decided.
        let plan = Plan { steps: vec![task.to_string()] };
        let outcome = match child.run(task, &plan).await {
            Ok(outcome) => outcome,
            Err(err) => return Ok(ToolOutput::error(format!("the child agent failed: {err}"))),
        };

        // Only the summary crosses back. The transcript is discarded with the
        // child, which is the entire reason this exists.
        let report = format!(
            "[{} subagent, {} step(s)] {}",
            role.name, outcome.steps_used, outcome.summary
        );

        // Unverified work is reported as an error so the parent treats it as
        // something to check rather than something done. The edits still stand:
        // they are in the shared staging area either way, and hiding them would
        // leave the parent's change set lying about what is on disk.
        let mut out = if outcome.succeeded() {
            ToolOutput::ok(report)
        } else {
            ToolOutput::error(report)
        };
        // Folded into the parent's change set, so its verifier sees them and its
        // checkpoint can revert them.
        for path in outcome.changed {
            out = out.changed(path);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ariadne::Ariadne;
    use crate::engine::mock::{text_response, tool_call, MockEngine};
    use crate::engine::Response;
    use crate::oracle::Oracle;
    use crate::scribe::SymbolIndex;
    use crate::session::Session;
    use crate::themis::Themis;
    use crate::tools::ToolRegistry;
    use std::sync::Mutex;
    use tempfile::TempDir;

    /// A real crate, not just a file: Oracle's ladder runs `cargo check`, so a
    /// child working in a directory that is not a crate would fail verification
    /// for a reason that has nothing to do with what it was asked to do.
    ///
    /// The root is returned canonicalised because [`ToolCtx::new`] requires it:
    /// on Windows a temp path is not, so every write would resolve outside the
    /// jail and be refused.
    fn workspace() -> (TempDir, std::path::PathBuf) {
        let dir = TempDir::new().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("src")).expect("src");
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"child-fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .expect("manifest");
        std::fs::write(dir.path().join("src/lib.rs"), "pub fn one() -> u32 { 1 }\n")
            .expect("write");
        let root = std::fs::canonicalize(dir.path()).expect("canonicalize");
        (dir, root)
    }

    /// What a child was actually asked for.
    #[derive(Debug, Clone, PartialEq)]
    struct Spawned {
        task: String,
        role: String,
        max_steps: usize,
    }

    type Spawns = Arc<Mutex<Vec<Spawned>>>;

    /// Build a `Delegate` whose children run a scripted engine.
    ///
    /// The returned log records each request, which is how the tests below
    /// check what the child was given.
    fn delegate_with(
        root: &std::path::Path,
        scripted: Vec<Response>,
        parent_max_steps: usize,
    ) -> (Delegate, Spawns) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::clone(&seen);
        let root = root.to_path_buf();
        let scripted = Arc::new(Mutex::new(scripted));

        let spawn: Arc<SpawnChild> = Arc::new(move |req: ChildRequest| {
            recorder.lock().expect("seen").push(Spawned {
                task: req.task.clone(),
                role: req.role.name.clone(),
                max_steps: req.max_steps,
            });
            let queued = std::mem::take(&mut *scripted.lock().expect("scripted"));
            Ok(Talos::new(
                Box::new(MockEngine::new(queued)),
                // No delegate tool: a child cannot spawn another.
                ToolRegistry::standard(),
                req.ctx,
                Oracle::new(&root).without_baseline(),
                SymbolIndex::build(&root).expect("index"),
                Themis::from_text("Be correct."),
                Ariadne::new(req.max_steps, req.max_steps.saturating_sub(1).max(1)),
                Session::new(&root, "mock"),
                1024,
                false,
            ))
        });

        (Delegate::new(spawn, parent_max_steps), seen)
    }

    #[tokio::test]
    async fn a_child_reports_back_only_a_summary() {
        let (_dir, root) = workspace();
        // A child that only reads has changed nothing, so it gets pushed back
        // on and spends its whole budget saying so. Scripted to the budget.
        let (d, _) = delegate_with(
            &root,
            vec![text_response("had a look, all fine"); Delegate::child_steps(8)],
            8,
        );
        let ctx = ToolCtx::new(&root);

        let out = d.run(&json!({"task": "look at src/lib.rs"}), &ctx).await.expect("run");

        assert!(out.content.contains("had a look"), "{}", out.content);
        assert!(out.content.contains("general subagent"), "{}", out.content);
        assert!(
            !out.content.contains("look at src/lib.rs\n\nPlan"),
            "the child's transcript must not come back: {}",
            out.content,
        );
    }

    /// The child's budget is half the parent's, floored so it can still read
    /// something and then act on it.
    #[tokio::test]
    async fn a_child_gets_half_the_parents_budget() {
        assert_eq!(Delegate::child_steps(20), 10);
        assert_eq!(Delegate::child_steps(8), 4);
        assert_eq!(Delegate::child_steps(4), MIN_CHILD_STEPS);
        assert_eq!(Delegate::child_steps(1), MIN_CHILD_STEPS);

        let (_dir, root) = workspace();
        let (d, seen) = delegate_with(&root, vec![text_response("done")], 20);
        let ctx = ToolCtx::new(&root);
        d.run(&json!({"task": "x"}), &ctx).await.expect("run");

        assert_eq!(seen.lock().unwrap()[0].max_steps, 10);
    }

    /// The whole point: a child's edits land in the parent's staging area, so
    /// the parent's verifier sees them and its checkpoint can revert them.
    #[tokio::test]
    async fn a_childs_edits_reach_the_parents_change_set() {
        let (_dir, root) = workspace();
        let (d, _) = delegate_with(
            &root,
            vec![
                tool_call(
                    "1",
                    "write_file",
                    json!({"path": "src/added.rs", "content": "pub fn two() -> u32 { 2 }\n"}),
                ),
                text_response("added it"),
            ],
            8,
        );
        let ctx = ToolCtx::new(&root);

        let out = d.run(&json!({"task": "add src/added.rs"}), &ctx).await.expect("run");

        assert!(
            out.changed.iter().any(|p| p.ends_with("added.rs")),
            "the parent must be told what the child touched: {:?}",
            out.changed,
        );
        assert!(root.join("src/added.rs").exists());
    }

    /// A child that did not verify is reported as an error, so the parent treats
    /// it as something to check rather than something finished.
    #[tokio::test]
    async fn unverified_work_comes_back_as_an_error() {
        let (_dir, root) = workspace();
        let (d, _) = delegate_with(
            &root,
            vec![
                tool_call(
                    "1",
                    "write_file",
                    json!({"path": "src/lib.rs", "content": "pub fn one() -> u32 { \"no\" }\n"}),
                ),
                text_response("done"),
                text_response("still done"),
            ],
            8,
        );
        let ctx = ToolCtx::new(&root);

        let out = d.run(&json!({"task": "break it"}), &ctx).await.expect("run");
        assert!(out.is_error, "a child that failed verification must not read as success");
    }

    // ------------------------------------------------------------- roles

    #[tokio::test]
    async fn a_role_is_passed_through_to_the_child() {
        let (_dir, root) = workspace();
        let (d, seen) = delegate_with(&root, vec![text_response("reviewed")], 8);
        let ctx = ToolCtx::new(&root);

        d.run(&json!({"task": "review it", "role": "reviewer"}), &ctx).await.expect("run");

        assert_eq!(seen.lock().unwrap()[0].role, "reviewer");
    }

    #[tokio::test]
    async fn an_unknown_role_names_the_ones_that_exist() {
        let (_dir, root) = workspace();
        let (d, _) = delegate_with(&root, vec![text_response("x")], 8);
        let ctx = ToolCtx::new(&root);

        let out = d.run(&json!({"task": "x", "role": "wizard"}), &ctx).await.expect("run");

        assert!(out.is_error);
        assert!(out.content.contains("reviewer"), "{}", out.content);
    }

    #[test]
    fn the_schema_advertises_every_role() {
        let (_dir, root) = workspace();
        let (d, _) = delegate_with(&root, vec![], 8);
        let schema = d.schema();

        let names: Vec<&str> = schema["properties"]["role"]["enum"]
            .as_array()
            .expect("enum")
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert_eq!(names, ["general", "reviewer", "investigator"]);
        // Without the descriptions the model is picking a name out of a list.
        assert!(schema["properties"]["role"]["description"]
            .as_str()
            .is_some_and(|d| d.contains("without editing")));
    }

    // -------------------------------------------------------------- limits

    #[test]
    fn depth_one_is_the_limit() {
        assert!(Delegate::allowed_at(0), "the top-level agent may delegate");
        assert!(!Delegate::allowed_at(1), "a child may not delegate further");
        assert_eq!(MAX_DEPTH, 1);
    }

    /// The backstop, for a caller that hands the tool to a child anyway.
    #[tokio::test]
    async fn a_child_that_somehow_has_the_tool_is_still_refused() {
        let (_dir, root) = workspace();
        let (mut d, seen) = delegate_with(&root, vec![text_response("x")], 8);
        d.depth = MAX_DEPTH;
        let ctx = ToolCtx::new(&root);

        let out = d.run(&json!({"task": "spawn another"}), &ctx).await.expect("run");

        assert!(out.is_error);
        assert!(seen.lock().unwrap().is_empty(), "no child may be built at the limit");
    }

    #[tokio::test]
    async fn an_empty_task_is_refused_before_a_child_is_built() {
        let (_dir, root) = workspace();
        let (d, seen) = delegate_with(&root, vec![text_response("x")], 8);
        let ctx = ToolCtx::new(&root);

        for input in [json!({}), json!({"task": "   "})] {
            let out = d.run(&input, &ctx).await.expect("run");
            assert!(out.is_error, "{input}");
        }
        assert!(seen.lock().unwrap().is_empty(), "an empty task must not cost a child");
    }

    /// Spawning is the last point at which a whole subtask can be declined.
    #[test]
    fn delegating_is_consequential() {
        let (_dir, root) = workspace();
        let (d, _) = delegate_with(&root, vec![], 8);
        assert!(d.consequential());
    }
}
