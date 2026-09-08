//! Talos: execution.
//!
//! The agent loop, governed by Ariadne. Every iteration is one engine turn:
//! either the engine calls tools (work happens, the loop continues) or it
//! stops calling them (it believes the task is done, and Oracle is asked
//! whether that is true).
//!
//! The important structural choice is that **completion is decided by Oracle,
//! not by the engine**. The engine saying "done" is a request for
//! verification, not a halt. That is what makes `Halt::Done` mean something.
//!
//! Conversation state lives on the struct rather than inside [`Talos::run`],
//! so a finished run can be continued with [`Talos::resume`]. That is what
//! turns a one-shot command into an interactive session: the user disagrees,
//! and the agent keeps its whole context instead of starting over.

use std::collections::{BTreeSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::ariadne::{Ariadne, Halt, StepOutcome};
use crate::diff::FileDiff;
use crate::engine::{self, Content, Engine, Message, Request};
use crate::episode::{self, EpisodeStore, Record};
use crate::metis::Plan;
use crate::mission::{
    evidence_digest, ActionIntent, EnvironmentRecord, FinalVerdict, MissionContract, MissionEvent,
    MissionPhase, MissionState, MissionStore, OutcomeDecision, PlanNode, ProofRecord, ProofStatus,
};
use crate::oracle::{Oracle, Verdict};
use crate::scribe::SymbolIndex;
use crate::session::{Session, TraceEvent};
use crate::themis::{Themis, EXECUTOR_ROLE};
use crate::tools::{ToolCtx, ToolRegistry};

/// Fed back when the engine returns nothing at all.
///
/// An empty reply has no tool calls, so it used to take the "the engine
/// believes it is finished" branch, where an empty change set satisfies every
/// tier vacuously and the run reported success having done nothing. Found on
/// the Python side against a live reasoning model that spent its whole token
/// budget before producing any content; the same shape was here.
const EMPTY_REPLY_NOTE: &str =
    "Your last reply was empty. This usually means the response was truncated \
     before any content was produced. Reply with a tool call, or a short \
     statement of what you intend to do.";

/// How many recent tool-call signatures count as "again".
///
/// Ported from the Python side, where this was a one-step lookback and could
/// not see a loop that alternates. Here it was worse: there was no repeat check
/// at all, so `is_noop` was the whole staleness test and an engine re-issuing
/// one failing `edit_file` ran to the ceiling — it calls a tool every step, so
/// it is never a noop.
///
/// Four, because the window only has to be as long as the cycle it must close,
/// and the cycles that actually occur are short: re-issuing one failing edit,
/// alternating between two, walking a three-step ritual. Longer buys nothing
/// and `max_steps` remains the backstop for anything more baroque. Bounding it
/// at all keeps "again" meaning *recently* — an unbounded window would grow
/// more sensitive the longer a run went without progress, a coupling nothing
/// would test.
pub const FUTILE_WINDOW: usize = 4;

/// Floor for a plan-step budget. A step given fewer turns cannot both act
/// and ask to be verified, so the plan would be decorative.
const MIN_STEP_BUDGET: usize = 3;

/// First `Stuck` is a redirect, not a halt. A second one in the same drive
/// is the honest stop the constitution requires. One, not more: extra
/// redirects would spend the ceiling the way β=0.01 spent every loop.
const MAX_REDIRECTS: usize = 1;

/// How thoroughly a drive consults Oracle.
#[derive(Clone, Copy)]
enum VerifyMode {
    /// The full ladder, plus tier 4 when the caller asked for a judge.
    Full,
    /// Tier 0 only — used between plan steps, where cargo test after every
    /// step would cost more than the plan saves.
    Quick,
}

/// Fed back when the engine repeats a call that achieved nothing.
///
/// Halting is the backstop; the cheaper outcome is the engine noticing it is
/// going in a circle and trying something else while it still has budget.
const REPEATED_CALL_NOTE: &str =
    "That was the same tool call, with the same arguments, as one you made a \
     moment ago, and it changed nothing. Cycling back to it will not produce a \
     different result. Read the error above and do something different: check \
     the file's actual contents, try a different approach, or state plainly \
     what is blocking you.";

/// Fed back when the engine asks to be verified without having done anything.
const NOTHING_DONE_NOTE: &str =
    "Nothing has been changed or run this turn, so there is nothing to verify \
     and the task cannot be considered done. Reading a file tells you what to \
     do; it is not doing it. Either carry out the task with write_file, \
     edit_file or run, or state plainly what is blocking you.";

/// A stable identity for a step's tool calls.
///
/// Name and arguments, in the order issued. The provider's call `id` is
/// deliberately excluded: it is fresh on every turn, so including it would make
/// every step unique and the repeat check dead on arrival — the same reason the
/// Python side excludes the raw text of the call.
///
/// `serde_json::Value` keeps object keys in a `BTreeMap` unless `preserve_order`
/// is enabled, which this crate does not enable, so two calls with the same
/// arguments in a different order already serialise identically. That is what
/// makes this comparable without a hand-rolled canonicaliser.
pub(crate) fn signature(calls: &[(String, String, serde_json::Value)]) -> String {
    let pairs: Vec<serde_json::Value> = calls
        .iter()
        .map(|(_, name, input)| serde_json::json!([name, input]))
        .collect();
    serde_json::Value::Array(pairs).to_string()
}

/// Consulted before a tool call that can change something outside the
/// conversation.
///
/// # Why this is a seam and not a policy
///
/// The Rust harness had no permission boundary at all: `drive` dispatched
/// straight to the registry, and without `--dry-run` a run wrote to disk
/// unattended. The Python side has had one since the ACP server existed
/// (`talos.py`), so the two harnesses disagreed about whether the agent may act
/// without being asked.
///
/// It is an `Option` on `Talos`, and `None` means **allow**, matching Python's
/// `ask_permission=None`. That default is deliberate rather than lax: the
/// one-shot CLI (`knossos task`) is non-interactive by design, and a prompt
/// there would hang CI and every scripted use. Front ends that *can* ask —
/// `repl`, which owns stdin, and `serve`, which has a front end to ask — install
/// one. Which front ends do and do not is documented at each call site rather
/// than inferred from this type.
///
/// Implementations must **fail closed**: a disconnected front end, a dropped
/// channel or a broken asker denies. An approver that cannot ask is not an
/// approver that says yes.
#[async_trait::async_trait]
pub trait Approver: Send + Sync {
    async fn approve(&self, tool: &str, input: &serde_json::Value) -> bool;
}

pub struct Talos {
    pub engine: Box<dyn Engine>,
    pub tools: ToolRegistry,
    pub ctx: ToolCtx,
    pub oracle: Oracle,
    pub scribe: SymbolIndex,
    pub themis: Themis,
    pub ariadne: Ariadne,
    pub session: Session,
    pub max_tokens: u32,
    /// Whether to run tier 4 once the deterministic ladder passes.
    pub judge: bool,
    /// Bounds `messages`. Public so a caller can size it to the server's window
    /// or disable it by raising `max_tokens`.
    pub lethe: crate::lethe::Lethe,
    /// Per-agent allocation and compaction policy. A known server window
    /// always clamps this policy; an unknown window is never unlimited.
    pub context_policy: crate::context::ContextPolicy,
    /// Consulted before every consequential tool call. `None` is unattended:
    /// the executor proceeds, which is right for a scripted run and wrong for
    /// an editor — so the front ends that can ask, do. See [`Approver`].
    pub approver: Option<std::sync::Arc<dyn Approver>>,
    /// Words from the user, delivered at the next step boundary.
    ///
    /// Clone the handle out with [`Talos::interjections`] before starting a
    /// run; pushing to it afterwards steers the loop without ending it.
    pub interjections: crate::interject::Interjections,
    /// The system role this agent plays, composed with the constitution.
    ///
    /// Defaults to [`EXECUTOR_ROLE`]. A child agent overrides it, which is what
    /// makes a subagent *specialised* rather than merely separate — though the
    /// role only states the intent, and what it can actually do is decided by
    /// the registry it was given.
    pub role: String,
    /// Raised to stop the current turn at the next step boundary.
    ///
    /// A boundary rather than immediately: a step is a model call, its tool
    /// calls, and the verification of what they changed. Tearing out of the
    /// middle would leave the workspace half-edited with no verdict on it,
    /// which is worse than the extra few seconds — the point of stopping is to
    /// regain control, not to create a mess someone else has to reason about.
    ///
    /// Held by `Arc` because whoever cancels is by definition not the thread
    /// inside `drive`.
    pub cancel: Arc<AtomicBool>,
    /// Proactive repository context, and the gate that decides whether it
    /// belongs in the prompt at all.
    ///
    /// Distinct from the `search_code` tool, which is retrieval the *model*
    /// asks for. This is retrieval the harness offers unasked, which is exactly
    /// why it is gated: a model that did not ask for context cannot be assumed
    /// to want it, and injecting it into a general question measurably makes
    /// small models worse. `None` disables the whole path.
    pub retrieval: Option<crate::gate::RetrievalGate>,

    // --- session state, persisted across `run`/`resume` ---
    /// The running conversation. Survives between turns.
    pub messages: Vec<Message>,
    /// Every file touched so far, across all turns.
    pub changed: BTreeSet<PathBuf>,
    /// The original task, kept so tier 4 can still judge against it on resume.
    pub task: String,
    /// Whether a passing verifier must be backed by a consequential action.
    /// `None` derives the answer from task language; eval cases set it from
    /// their explicit `expected_action` contract.
    require_action: Option<bool>,
    /// Failed hypotheses and attempts that survive Lethe and process restart.
    pub episodes: EpisodeStore,
    /// Ablation switch: when false, neither recall nor durable writes occur.
    episodes_enabled: bool,
    attempt_seq: usize,
    current_attempt: String,
    attempt_parent: Option<String>,
    /// Unused tail of a multi-step plan. Empty on a flat drive. The supervisor
    /// replaces this on redirect; `drive_plan` splices it back.
    plan_remainder: Vec<String>,
    /// Provider-independent state for the current task. Tool output is never
    /// copied here; only evidence hashes and changed paths cross this boundary.
    mission: Option<MissionStore>,
    mission_sequence: u64,
    /// Captured by the caller before Metis plans. It becomes immutable mission
    /// evidence at creation; discovery itself cannot execute a setup recipe.
    environment: Option<EnvironmentRecord>,
    recovery_identity: Option<String>,
    recovery_lock: Option<std::fs::File>,
    shared_quota: Option<Arc<crate::engine::budget::Quota>>,
}

#[derive(Serialize, Deserialize)]
struct ConversationState {
    policy_hash: String,
    task: String,
    messages: Vec<Message>,
    changed: BTreeSet<PathBuf>,
    changed_hashes: std::collections::BTreeMap<PathBuf, Option<String>>,
    require_action: Option<bool>,
    attempt_seq: usize,
    current_attempt: String,
    attempt_parent: Option<String>,
    plan_remainder: Vec<String>,
    quota: Option<crate::engine::budget::RecoveryQuota>,
}

impl Talos {
    /// Build with empty session state.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        engine: Box<dyn Engine>,
        tools: ToolRegistry,
        ctx: ToolCtx,
        oracle: Oracle,
        scribe: SymbolIndex,
        themis: Themis,
        ariadne: Ariadne,
        session: Session,
        max_tokens: u32,
        judge: bool,
    ) -> Self {
        let episodes = EpisodeStore::open(ctx.root());
        Talos {
            engine,
            tools,
            ctx,
            oracle,
            scribe,
            themis,
            ariadne,
            session,
            max_tokens,
            judge,
            lethe: crate::lethe::Lethe::default(),
            context_policy: crate::context::ContextPolicy::for_completion(max_tokens),
            approver: None,
            interjections: crate::interject::Interjections::new(),
            role: EXECUTOR_ROLE.to_string(),
            cancel: Arc::new(AtomicBool::new(false)),
            retrieval: None,
            messages: Vec::new(),
            changed: BTreeSet::new(),
            task: String::new(),
            require_action: None,
            episodes,
            episodes_enabled: true,
            attempt_seq: 0,
            current_attempt: String::new(),
            attempt_parent: None,
            plan_remainder: Vec::new(),
            mission: None,
            mission_sequence: 0,
            environment: None,
            recovery_identity: None,
            recovery_lock: None,
            shared_quota: None,
        }
    }

    pub fn set_require_action(&mut self, require: bool) {
        self.require_action = Some(require);
    }

    pub fn disable_memory(&mut self) {
        self.episodes_enabled = false;
    }

    pub fn disable_compaction(&mut self) {
        self.context_policy.compaction_enabled = false;
        self.lethe.max_tokens = usize::MAX;
    }

    pub fn mission_state(&self) -> Option<&MissionState> {
        self.mission.as_ref().map(MissionStore::state)
    }

    pub fn mission_id(&self) -> Option<&str> {
        self.mission.as_ref().map(MissionStore::mission_id)
    }

    /// Accept a verified handoff. The decision is journaled against the exact
    /// contract and workspace revisions rather than inferred from a new prompt.
    pub fn accept_mission(&mut self, reason: impl Into<String>) -> Result<()> {
        let reason = reason.into();
        anyhow::ensure!(!reason.trim().is_empty(), "acceptance reason is required");
        anyhow::ensure!(
            self.mission
                .as_ref()
                .is_some_and(|mission| mission.state().focus.phase == MissionPhase::Handoff),
            "accept requires a mission at handoff"
        );
        self.append_mission(MissionEvent::OutcomeDecided {
            decision: OutcomeDecision::Accepted,
            reason,
            checkpoint: None,
        })
    }

    /// Revert the last delivered turn and journal the decision only after the
    /// workspace rewind succeeds. A restarted executor has no in-memory file
    /// undo log and therefore fails closed instead of pretending to revert.
    pub fn revert_mission(&mut self, reason: impl Into<String>) -> Result<Vec<PathBuf>> {
        let reason = reason.into();
        anyhow::ensure!(!reason.trim().is_empty(), "revert reason is required");
        anyhow::ensure!(
            self.mission
                .as_ref()
                .is_some_and(|mission| mission.state().focus.phase == MissionPhase::Handoff),
            "revert requires a mission at handoff"
        );
        let checkpoint = self
            .mission
            .as_ref()
            .and_then(|mission| mission.state().workspace.snapshots.last())
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("mission has no safe handoff checkpoint"))?;
        let restored = self.undo_turn()?;
        self.append_mission(MissionEvent::OutcomeDecided {
            decision: OutcomeDecision::Reverted,
            reason,
            checkpoint: Some(checkpoint),
        })?;
        Ok(restored)
    }

    /// Attach a bounded host/container fingerprint captured before planning.
    pub fn set_environment(&mut self, environment: EnvironmentRecord) {
        self.environment = Some(environment);
    }

    /// Capture host-mode facts before a front end asks Metis to spend a model
    /// turn. The idempotent fallback in `start_mission` protects callers that
    /// construct Talos directly, but front ends should call this before plan
    /// generation so the plan itself can be attributed to the environment.
    pub fn capture_environment(&mut self) -> Result<()> {
        if self.environment.is_none() {
            self.environment = Some(
                crate::environment::EnvironmentFingerprint::discover(self.ctx.root())
                    .map(crate::environment::EnvironmentFingerprint::into_record)?,
            );
        }
        Ok(())
    }

    pub fn attach_quota(&mut self, quota: Arc<crate::engine::budget::Quota>) {
        self.shared_quota = Some(quota);
    }

    /// Explicit opt-in: portable checkpoints contain prompt and tool-result
    /// content. The caller supplies an identity covering its external policy
    /// configuration (including tool/MCP settings); only its hash is retained.
    pub fn enable_conversation_checkpoints(&mut self, policy_identity: String) -> Result<()> {
        anyhow::ensure!(
            self.mission.is_none(),
            "enable checkpoints before starting a mission"
        );
        anyhow::ensure!(
            !self.ctx.is_dry_run(),
            "preview overlays cannot yet be restored from conversation checkpoints"
        );
        anyhow::ensure!(
            !policy_identity.is_empty(),
            "checkpoint policy identity is required"
        );
        self.recovery_identity = Some(policy_identity);
        Ok(())
    }

    fn recovery_policy_hash(&self) -> Result<String> {
        let value = serde_json::json!({
            "runtime_version": env!("CARGO_PKG_VERSION"),
            "external": self.recovery_identity.as_ref().ok_or_else(|| anyhow::anyhow!("conversation persistence is not enabled"))?,
            "engine": self.engine.name(), "native_tools": self.engine.supports_native_tools(),
            "role": self.role, "constitution": self.themis.principles(), "tools": self.tools.defs(),
            "max_tokens": self.max_tokens, "ariadne": self.ariadne, "judge": self.judge,
            "context": self.context_policy, "approvals": self.approver.is_some(),
            "memory": self.episodes_enabled,
        });
        Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(&value)?)))
    }

    fn changed_file_hashes(
        &self,
        paths: &BTreeSet<PathBuf>,
    ) -> Result<std::collections::BTreeMap<PathBuf, Option<String>>> {
        use std::io::Read;
        let root = self.ctx.root().canonicalize()?;
        let mut result = std::collections::BTreeMap::new();
        for path in paths {
            let path = self.ctx.resolve(
                path.to_str()
                    .ok_or_else(|| anyhow::anyhow!("non-UTF8 changed path"))?,
            )?;
            anyhow::ensure!(
                path.starts_with(&root),
                "checkpoint changed path escaped workspace"
            );
            let hash = match std::fs::symlink_metadata(&path) {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
                Err(e) => return Err(e.into()),
                Ok(meta) => {
                    anyhow::ensure!(
                        meta.is_file() && !meta.file_type().is_symlink(),
                        "checkpoint changed path is not a regular file"
                    );
                    let mut stream = std::fs::File::open(&path)?.take(64 * 1024 * 1024 + 1);
                    let mut hash = Sha256::new();
                    let mut buffer = [0u8; 65536];
                    let mut size = 0u64;
                    loop {
                        let n = stream.read(&mut buffer)?;
                        if n == 0 {
                            break;
                        }
                        size += n as u64;
                        anyhow::ensure!(
                            size <= 64 * 1024 * 1024,
                            "changed file exceeds checkpoint hash limit"
                        );
                        hash.update(&buffer[..n]);
                    }
                    let after = std::fs::metadata(&path)?;
                    anyhow::ensure!(
                        meta.len() == size
                            && meta.len() == after.len()
                            && meta.modified()? == after.modified()?,
                        "changed file mutated while checkpointing"
                    );
                    Some(format!("{:x}", hash.finalize()))
                }
            };
            result.insert(path, hash);
        }
        Ok(result)
    }

    fn save_conversation(&mut self) -> Result<()> {
        if self.recovery_identity.is_none() {
            return Ok(());
        }
        validate_conversation(&self.messages)?;
        let saved = ConversationState {
            policy_hash: self.recovery_policy_hash()?,
            task: self.task.clone(),
            messages: self.messages.clone(),
            changed: self.changed.clone(),
            changed_hashes: self.changed_file_hashes(&self.changed)?,
            require_action: self.require_action,
            attempt_seq: self.attempt_seq,
            current_attempt: self.current_attempt.clone(),
            attempt_parent: self.attempt_parent.clone(),
            plan_remainder: self.plan_remainder.clone(),
            quota: self
                .shared_quota
                .as_ref()
                .map(|quota| quota.recovery_snapshot())
                .transpose()?,
        };
        self.mission
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("no mission to checkpoint"))?
            .save_conversation(&serde_json::to_value(saved)?)
    }

    /// Restore a completed safe boundary. A newer journal or pending action
    /// refuses replay, preserving uncertain effects for operator reconciliation.
    pub fn restore_conversation(&mut self, mission_id: &str) -> Result<()> {
        anyhow::ensure!(
            self.messages.is_empty() && self.mission.is_none(),
            "restore requires a fresh executor"
        );
        let store = MissionStore::open_for_resume(self.ctx.root(), mission_id)?;
        let lock = store.lock_execution()?;
        let store = MissionStore::open_for_resume(self.ctx.root(), mission_id)?;
        let saved: ConversationState = serde_json::from_value(store.load_conversation()?)?;
        anyhow::ensure!(
            saved.policy_hash == self.recovery_policy_hash()?,
            "checkpoint execution policy or provider differs"
        );
        anyhow::ensure!(
            saved.task == store.state().contract.outcome,
            "checkpoint task/contract mismatch"
        );
        validate_conversation(&saved.messages)?;
        anyhow::ensure!(
            saved.changed_hashes == self.changed_file_hashes(&saved.changed)?,
            "checkpoint workspace files changed; review before resuming"
        );
        match (&self.shared_quota, &saved.quota) {
            (Some(quota), Some(saved)) => quota.restore_spent(saved)?,
            (None, None) => {}
            _ => anyhow::bail!("checkpoint quota configuration differs"),
        }
        self.task = saved.task;
        self.messages = saved.messages;
        self.changed = saved.changed;
        self.require_action = saved.require_action;
        self.attempt_seq = saved.attempt_seq;
        self.current_attempt = saved.current_attempt;
        self.attempt_parent = saved.attempt_parent;
        self.plan_remainder = saved.plan_remainder;
        self.environment = store.state().environment.clone();
        self.mission = Some(store);
        self.recovery_lock = Some(lock);
        // A fresh baseline must not forgive failures introduced by the prior
        // process. Until baseline serialization lands, resumed verification is
        // strict and can require repair of pre-existing failures as well.
        self.oracle = Oracle::new(self.ctx.root()).without_baseline();
        Ok(())
    }
}

fn validate_conversation(messages: &[Message]) -> Result<()> {
    let mut pending = BTreeSet::new();
    anyhow::ensure!(!messages.is_empty(), "empty conversation checkpoint");
    for message in messages {
        for block in &message.content {
            match block {
                Content::ToolUse { id, .. } => {
                    anyhow::ensure!(
                        message.role == crate::engine::types::Role::Assistant
                            && !id.is_empty()
                            && pending.insert(id.clone()),
                        "invalid checkpoint tool call"
                    );
                }
                Content::ToolResult { id, .. } => {
                    anyhow::ensure!(
                        message.role == crate::engine::types::Role::User && pending.remove(id),
                        "unmatched checkpoint tool result"
                    );
                }
                Content::Text { .. } => {}
            }
        }
    }
    anyhow::ensure!(
        pending.is_empty(),
        "checkpoint contains unresolved tool calls"
    );
    Ok(())
}

fn task_requires_action(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    let trimmed = lower.trim();
    // Pronoun-only imperatives do not identify an object or desired outcome.
    // Completing them without first asking would be a guess, so clarification
    // itself is allowed to be the successful action.
    if matches!(trimmed, "fix it" | "change it" | "update it" | "do it") {
        return false;
    }
    if lower.contains("fix it if ")
        || lower.contains("change it if ")
        || lower.contains("only if reproducible")
    {
        return false;
    }
    if trimmed.ends_with('?')
        || [
            "where ",
            "what ",
            "which ",
            "why ",
            "how ",
            "explain ",
            "tell me ",
            "read ",
            "investigate ",
            "find where ",
            "locate ",
        ]
        .iter()
        .any(|prefix| trimmed.starts_with(prefix))
    {
        return false;
    }
    let words: std::collections::BTreeSet<_> = lower
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .filter(|word| !word.is_empty())
        .collect();
    [
        "add",
        "build",
        "change",
        "configure",
        "create",
        "delete",
        "disable",
        "enable",
        "extract",
        "fix",
        "implement",
        "make",
        "migrate",
        "refactor",
        "remove",
        "rename",
        "repair",
        "replace",
        "route",
        "run",
        "ship",
        "test",
        "update",
        "verify",
        "write",
    ]
    .iter()
    .any(|word| words.contains(word))
}

#[derive(Debug, Clone)]
pub struct Outcome {
    pub halt: Halt,
    pub steps_used: usize,
    pub changed: Vec<PathBuf>,
    pub verdict: Option<Verdict>,
    pub summary: String,
    /// True when nothing was written to disk.
    pub dry_run: bool,
}

impl Outcome {
    pub fn succeeded(&self) -> bool {
        self.halt == Halt::Done
    }
}

impl Talos {
    /// Play a different role than the default executor.
    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.role = role.into();
        self
    }

    /// Attach proactive retrieval. Without it the harness offers no unasked
    /// context and the `search_code` tool remains the only way in.
    pub fn with_retrieval(mut self, gate: crate::gate::RetrievalGate) -> Self {
        self.retrieval = Some(gate);
        self
    }

    pub fn with_context_limits(
        mut self,
        assigned_tokens: Option<u32>,
        compact_at_tokens: Option<u32>,
    ) -> Self {
        self.context_policy.assigned_tokens = assigned_tokens;
        self.context_policy.compact_at_tokens = compact_at_tokens;
        self
    }

    pub fn set_context_limits(
        &mut self,
        assigned_tokens: Option<u32>,
        compact_at_tokens: Option<u32>,
    ) -> crate::context::ContextBudget {
        self.context_policy.assigned_tokens = assigned_tokens;
        self.context_policy.compact_at_tokens = compact_at_tokens;
        self.context_budget()
    }

    pub fn context_budget(&self) -> crate::context::ContextBudget {
        self.context_policy.resolve(
            self.engine.context_window(),
            crate::lethe::DEFAULT_MAX_TOKENS as u32,
        )
    }

    /// How much retrieved source may be injected, in characters.
    ///
    /// Deliberately small next to the context window. This is a head start, not
    /// a substitute for the tools: the model can still read any file it wants,
    /// and a large injection buys a worse trade than it looks — it displaces
    /// conversation Lethe would otherwise have kept, to supply code that may not
    /// be the code that mattered.
    const CONTEXT_BUDGET: usize = 6_000;
    /// How far to follow the import graph out from a seed file.
    const CONTEXT_HOPS: usize = 1;

    /// Repository context for `query`, when the gate says it belongs.
    ///
    /// The decision is logged either way, because a wrong *skip* is the harder
    /// failure to see: the model just answers without the code it needed, and
    /// nothing else in the trace would say why.
    fn repository_context(&self, query: &str) -> Option<String> {
        let gate = self.retrieval.as_ref()?;
        let decision = gate.decide(query, None);

        let context = decision
            .inject
            .then(|| {
                gate.argus()
                    .context(query, Self::CONTEXT_BUDGET, Self::CONTEXT_HOPS)
            })
            .filter(|c| !c.hits.is_empty());

        self.session.log(&TraceEvent::ContextConsidered {
            query: query.to_string(),
            injected: context.is_some(),
            confidence: decision.confidence,
            reasons: decision.reasons,
            slices: context.as_ref().map_or(0, |c| c.hits.len()),
            chars: context.as_ref().map_or(0, |c| c.text.len()),
        });

        context.map(|c| c.text)
    }

    /// Wrap retrieved source so it cannot be mistaken for part of the task.
    ///
    /// The framing is load-bearing. Unlabelled code in the prompt reads as
    /// something to act on, and the measured failure this whole path guards
    /// against is exactly that: a model handed repository excerpts started
    /// describing the repository instead of doing what was asked.
    fn frame_context(context: &str) -> String {
        format!(
            "\n\nExisting code retrieved automatically because it looked relevant. \
             It is background, not part of the task, and it may be incomplete — \
             read the files directly if you need more:\n\n{context}"
        )
    }

    fn mission_plan(plan: &Plan, prefix: &str) -> Vec<PlanNode> {
        plan.steps
            .iter()
            .enumerate()
            .map(|(index, step)| PlanNode {
                id: format!("{prefix}-{}", index + 1),
                dependencies: if index > 0 {
                    vec![format!("{prefix}-{index}")]
                } else {
                    Vec::new()
                },
                status: "pending".into(),
                owner: Some("talos".into()),
                expected_output: step.clone(),
                proof: "oracle verification bound to the resulting workspace revision".into(),
                rollback: Some("rewind the step checkpoint".into()),
            })
            .collect()
    }

    fn append_mission(&mut self, event: MissionEvent) -> Result<()> {
        if let Some(mission) = &mut self.mission {
            mission.append(event)?;
        }
        Ok(())
    }

    fn transition_mission(&mut self, to: MissionPhase, cause: impl Into<String>) -> Result<()> {
        let Some(from) = self
            .mission
            .as_ref()
            .map(|mission| mission.state().focus.phase)
        else {
            return Ok(());
        };
        if from == to {
            return Ok(());
        }
        self.append_mission(MissionEvent::Transition {
            from,
            to,
            cause: cause.into(),
        })
    }

    fn start_mission(&mut self, task: &str, plan: &Plan) -> Result<()> {
        self.capture_environment()?;
        self.mission_sequence = self.mission_sequence.saturating_add(1);
        let mission_id = format!("{}-{}", self.session.run_id(), self.mission_sequence);
        let root = self
            .ctx
            .root()
            .canonicalize()
            .unwrap_or_else(|_| self.ctx.root().to_path_buf());
        let contract = MissionContract {
            outcome: task.to_string(),
            scope: vec![root.display().to_string()],
            exclusions: vec!["paths outside the workspace".into()],
            constraints: vec![
                "consequential actions obey the active approval policy".into(),
                "tool output is retained by hash, not copied into MissionState".into(),
            ],
            risk: if self.ctx.is_dry_run() {
                "preview_only".into()
            } else {
                "workspace_mutation".into()
            },
            definition_of_done: vec![
                "the requested outcome is implemented".into(),
                "Oracle verification is current for the final workspace revision".into(),
                "the operator receives an evidence-backed handoff".into(),
            ],
        };
        let mut state = MissionState::new(mission_id, contract, root.display().to_string())?;
        state.policy.execution_mode = if self.ctx.is_dry_run() {
            "preview"
        } else if self.approver.is_some() {
            "ask"
        } else {
            "unattended"
        }
        .into();
        state.policy.tool_grants = self
            .tools
            .defs()
            .into_iter()
            .map(|tool| tool.name)
            .collect();
        state.budget.token_limit = Some(self.max_tokens as u64);
        state.budget.tool_limit = Some(self.ariadne.max_steps as u64);
        state.budget.retry_limit = Some(MAX_REDIRECTS as u64);
        state.environment = self.environment.clone();

        self.mission = Some(MissionStore::create(self.ctx.root(), state)?);
        self.recovery_lock = if self.recovery_identity.is_some() {
            Some(
                self.mission
                    .as_ref()
                    .expect("created mission")
                    .lock_execution()?,
            )
        } else {
            None
        };
        self.transition_mission(MissionPhase::Recon, "workspace reconnaissance began")?;
        self.transition_mission(MissionPhase::Contract, "task contract captured")?;
        self.transition_mission(MissionPhase::Baseline, "pre-change baseline required")?;
        let nodes = Self::mission_plan(plan, "step");
        let active_node = nodes.first().map(|node| node.id.clone());
        self.append_mission(MissionEvent::PlanRevised {
            nodes,
            active_node,
            cause: "Metis produced the initial plan".into(),
        })
    }

    fn prepare_mission_execution(&mut self) -> Result<()> {
        let phase = self
            .mission
            .as_ref()
            .map(|mission| mission.state().focus.phase);
        if phase == Some(MissionPhase::Baseline) {
            self.transition_mission(MissionPhase::Plan, "baseline captured")?;
        }
        if self
            .mission
            .as_ref()
            .map(|mission| mission.state().focus.phase)
            == Some(MissionPhase::Plan)
        {
            self.transition_mission(MissionPhase::Execute, "execution began")?;
        }
        Ok(())
    }

    fn resume_mission(&mut self, instruction: &str) -> Result<()> {
        let Some(phase) = self
            .mission
            .as_ref()
            .map(|mission| mission.state().focus.phase)
        else {
            return Ok(());
        };
        match phase {
            MissionPhase::Handoff => {
                self.append_mission(MissionEvent::OutcomeDecided {
                    decision: OutcomeDecision::Revise,
                    reason: instruction.to_string(),
                    checkpoint: None,
                })?;
                let (expected_revision, mut contract) = self
                    .mission
                    .as_ref()
                    .map(|mission| {
                        (
                            mission.state().contract_revision,
                            mission.state().contract.clone(),
                        )
                    })
                    .expect("handoff phase has a mission");
                contract
                    .constraints
                    .push(format!("operator revision: {instruction}"));
                self.append_mission(MissionEvent::ContractAmended {
                    expected_revision,
                    contract,
                    reason: "operator revised the delivered outcome".into(),
                })?;
            }
            MissionPhase::Verify => {
                self.transition_mission(MissionPhase::Replan, "verification needed revision")?;
            }
            MissionPhase::Blocked => {
                self.transition_mission(MissionPhase::Recon, "operator supplied new direction")?;
                self.transition_mission(MissionPhase::Contract, "revision scope captured")?;
                self.transition_mission(MissionPhase::Baseline, "existing baseline reused")?;
            }
            MissionPhase::Paused => {
                self.transition_mission(MissionPhase::Execute, "operator resumed the mission")?;
            }
            _ => {}
        }
        let plan = Plan {
            steps: vec![instruction.to_string()],
        };
        let nodes = Self::mission_plan(&plan, "revision");
        let active_node = nodes.first().map(|node| node.id.clone());
        self.append_mission(MissionEvent::PlanRevised {
            nodes,
            active_node,
            cause: "operator supplied follow-up instructions".into(),
        })?;
        let phase = self
            .mission
            .as_ref()
            .map(|mission| mission.state().focus.phase);
        if phase == Some(MissionPhase::Revise) || phase == Some(MissionPhase::Replan) {
            self.transition_mission(MissionPhase::Plan, "revision plan accepted for execution")?;
            self.transition_mission(MissionPhase::Execute, "revision execution began")?;
        }
        Ok(())
    }

    fn record_compaction(&mut self, before: usize, after: usize) -> Result<()> {
        let Some(mission) = self.mission.as_ref() else {
            return Ok(());
        };
        let generation = mission.state().conversation.compaction_generation + 1;
        let source_start = mission.state().conversation.recent_message_start;
        let source_end = source_start.saturating_add(before.saturating_sub(1) as u64);
        let manifest = format!(
            "messages:{before}->{after};tokens:{}",
            crate::lethe::estimate_tokens(&self.messages)
        );
        self.append_mission(MissionEvent::Compacted {
            generation,
            source_start,
            source_end,
            manifest_hash: evidence_digest(manifest.as_bytes()),
        })
    }

    fn record_consequential_result(
        &mut self,
        call_id: &str,
        tool: &str,
        output: &crate::tools::ToolOutput,
    ) -> Result<()> {
        let changed_paths = output
            .changed
            .iter()
            .map(|path| {
                path.strip_prefix(self.ctx.root())
                    .unwrap_or(path)
                    .display()
                    .to_string()
            })
            .collect();
        self.append_mission(MissionEvent::ConsequentialResult {
            call_id: call_id.to_string(),
            tool: tool.to_string(),
            succeeded: !output.is_error,
            result_hash: evidence_digest(output.content.as_bytes()),
            changed_paths,
        })
    }

    fn record_verdict(&mut self, verdict: &Verdict, full: bool) -> Result<()> {
        let Some(mission) = self.mission.as_ref() else {
            return Ok(());
        };
        let revision = mission.state().workspace.revision;
        let environment_current = self.environment.as_ref().is_some_and(|environment| {
            crate::environment::EnvironmentFingerprint::unchanged_since(
                self.ctx.root(),
                environment,
            )
            .unwrap_or(false)
        });
        let conclusive = full && verdict.deterministic_tiers_passed() && environment_current;
        let status = if !verdict.passed || !environment_current {
            ProofStatus::Failed
        } else {
            ProofStatus::Passed
        };
        let final_verdict = if !verdict.passed || !environment_current {
            FinalVerdict::Rejected
        } else if conclusive {
            FinalVerdict::Verified
        } else {
            FinalVerdict::PartiallyVerified
        };
        let summary = if environment_current {
            verdict.summary()
        } else {
            format!(
                "{}\nEnvironment changed or could not be re-fingerprinted before verification.",
                verdict.summary()
            )
        };
        self.append_mission(MissionEvent::VerifierVerdict {
            proof: ProofRecord {
                id: "oracle".into(),
                status,
                bound_revision: Some(revision),
                evidence_id: Some(evidence_digest(summary.as_bytes())),
            },
            verdict: final_verdict,
        })?;
        if full {
            self.append_mission(MissionEvent::EnvironmentVerification {
                status: match final_verdict {
                    FinalVerdict::Verified => "passed",
                    FinalVerdict::PartiallyVerified => "partial",
                    FinalVerdict::Rejected | FinalVerdict::Unverified => "failed",
                }
                .into(),
                evidence_hash: evidence_digest(summary.as_bytes()),
            })?;
        }
        Ok(())
    }

    /// Start a fresh task, discarding any previous conversation.
    pub async fn run(&mut self, task: &str, plan: &Plan) -> Result<Outcome> {
        self.start_mission(task, plan)?;
        self.session.log(&TraceEvent::PlanProduced {
            steps: plan.steps.clone(),
        });

        self.task = task.to_string();
        self.changed.clear();

        // Composed before the closing instruction rather than after it, so what
        // the model is being asked to do stays the last thing it reads.
        let mut prompt = format!("Task: {task}\n\nPlan:\n{}", plan.render());
        if let Some(context) = self.repository_context(task) {
            prompt.push_str(&Self::frame_context(&context));
        }
        prompt.push_str(
            "\n\nWork through it. When everything is complete, reply with a short summary \
             and no tool calls — verification runs automatically.",
        );
        if let Some(brief) = self.recall_brief() {
            prompt.push_str("\n\n");
            prompt.push_str(&brief);
        }
        self.messages = vec![Message::user_text(prompt)];

        self.mark_turn();
        // Stepwise execution needs enough room for every slice plus a closing
        // verifier pass. When the configured hard ceiling cannot fund that,
        // flatten the already-rendered plan into one ordinary drive. Flooring
        // every slice at `MIN_STEP_BUDGET` used to turn an 8-turn ceiling and a
        // three-step plan into as many as 12 turns, while also forcing simple
        // read/read/edit work to stop at each artificial boundary.
        let stepwise_min = plan
            .steps
            .len()
            .saturating_add(1)
            .saturating_mul(MIN_STEP_BUDGET);
        if plan.steps.len() > 1 && self.ariadne.max_steps >= stepwise_min {
            self.drive_plan(plan).await
        } else {
            self.drive().await
        }
    }

    /// Continue the existing conversation with new instructions.
    ///
    /// The step budget resets: each thing the user asks for gets its own
    /// allowance, rather than one budget draining across a long session.
    pub async fn resume(&mut self, instruction: &str) -> Result<Outcome> {
        if self.messages.is_empty() {
            let plan = Plan {
                steps: vec![instruction.to_string()],
            };
            return self.run(instruction, &plan).await;
        }
        if let Some(environment) = self.environment.as_ref() {
            if !crate::environment::EnvironmentFingerprint::unchanged_since(
                self.ctx.root(),
                environment,
            )? {
                anyhow::bail!(
                    "workspace instructions or lockfiles changed since this mission began; review the new environment and start a revised mission"
                );
            }
        }
        self.resume_mission(instruction)?;
        // Gated per turn, not once per session: the conversation may have moved
        // to a different part of the tree, and the turn that needs context is
        // rarely the one that opened the session.
        let mut prompt = instruction.to_string();
        if let Some(context) = self.repository_context(instruction) {
            prompt.push_str(&Self::frame_context(&context));
        }
        if let Some(brief) = self.recall_brief() {
            prompt.push_str("\n\n");
            prompt.push_str(&brief);
        }
        self.messages.push(Message::user_text(prompt));
        self.mark_turn();
        self.drive().await
    }

    /// Run each plan step under its own budget, checking as it goes.
    ///
    /// Two things a flat loop cannot do:
    ///
    /// **A step cannot spend the whole budget.** Each gets an equal slice of
    /// the allowance minus a reserve for closing, floored at
    /// [`MIN_STEP_BUDGET`]. One step thrashing can no longer starve the rest.
    ///
    /// **Breakage is caught where it happened.** Between steps only tier 0
    /// runs. A step that leaves the tree unparseable is rolled back to the
    /// checkpoint taken before it, so the next step is not built on broken
    /// source. The full ladder is reserved for the closing drive.
    async fn drive_plan(&mut self, plan: &Plan) -> Result<Outcome> {
        let mut steps = plan.steps.clone();
        let reserve = MIN_STEP_BUDGET.max(self.ariadne.max_steps / 4);
        let available = MIN_STEP_BUDGET.max(self.ariadne.max_steps.saturating_sub(reserve));
        let mut per_step = MIN_STEP_BUDGET.max(available / steps.len().max(1));

        let mut used = 0usize;
        let mut worked = false;
        let mut index = 0usize;

        while index < steps.len() {
            if self.cancel.swap(false, Ordering::SeqCst) {
                return self.finish(Halt::Cancelled, used, None, "cancelled by the caller", true);
            }

            let step = steps[index].clone();
            self.messages.push(Message::user_text(format!(
                "## Step {} of {}\n{step}\n\nDo only this step. The remaining steps \
                 are not yours to start.",
                index + 1,
                steps.len()
            )));

            let budget = Ariadne {
                max_steps: per_step,
                target_steps: per_step.saturating_sub(1).max(1),
                stuck_after: self.ariadne.stuck_after,
            };
            let mark = self.ctx.checkpoint(format!("step-{}", index + 1));
            self.plan_remainder = steps[index + 1..].to_vec();

            let require_action = self
                .require_action
                .unwrap_or_else(|| task_requires_action(&self.task));
            let outcome = self
                .drive_with(budget, VerifyMode::Quick, require_action)
                .await?;
            used = used.saturating_add(outcome.steps_used);

            // Supervisor may have replaced the unused tail.
            let mut rebuilt = steps[..=index].to_vec();
            rebuilt.extend(self.plan_remainder.iter().cloned());
            steps = rebuilt;

            if outcome.succeeded() {
                worked = true;
            } else {
                let reverted = self.revert_broken_step(&mark, &outcome);
                if !reverted.is_empty() {
                    let names = reverted
                        .iter()
                        .map(|p| p.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    self.messages.push(Message::user_text(format!(
                        "Step {} left the tree unparseable, so it was undone: {names} \
                         {} back to the state before that step. Nothing after it was \
                         built on the broken version. Try a different approach.",
                        index + 1,
                        if reverted.len() == 1 { "is" } else { "are" }
                    )));
                } else {
                    self.messages.push(Message::user_text(format!(
                        "Step {} ended as {} rather than verified. Carry what you \
                         learned into the next step; do not restart the plan.",
                        index + 1,
                        outcome.halt.label()
                    )));
                }
                // Re-divide whatever budget is left if the tail is still long.
                // Replanning is not wired yet; this keeps the arithmetic ready
                // so a revision cannot starve every remaining step.
                let left = steps.len().saturating_sub(index + 1).max(1);
                per_step = MIN_STEP_BUDGET.max(available.saturating_sub(used) / left);
            }

            index += 1;
        }

        if self.cancel.swap(false, Ordering::SeqCst) {
            return self.finish(Halt::Cancelled, used, None, "cancelled by the caller", true);
        }

        self.messages.push(Message::user_text(format!(
            "## Plan complete\nAll {} steps have been attempted. Confirm the \
             original task is done, repair anything outstanding, then reply \
             without a tool call so verification can run.",
            steps.len()
        )));

        let closing = Ariadne {
            max_steps: reserve,
            target_steps: reserve.saturating_sub(1).max(1),
            stuck_after: self.ariadne.stuck_after,
        };
        // Relaxed only when the plan actually accomplished something.
        // Otherwise this phase is a second door onto "every step did
        // nothing, then a verdict over an empty change set reports success."
        let task_requires_action = self
            .require_action
            .unwrap_or_else(|| task_requires_action(&self.task));
        let final_outcome = self
            .drive_with(closing, VerifyMode::Full, task_requires_action && !worked)
            .await?;

        Ok(Outcome {
            halt: final_outcome.halt,
            steps_used: used.saturating_add(final_outcome.steps_used),
            changed: final_outcome.changed,
            verdict: final_outcome.verdict,
            summary: final_outcome.summary,
            dry_run: final_outcome.dry_run,
        })
    }

    /// Undo a plan step that left the tree unparseable. Returns what moved.
    ///
    /// Narrow on purpose:
    ///
    /// * Only on a real failed verdict. A step that ran out of budget without
    ///   reaching the verifier proves nothing about the tree.
    /// * Only what that step wrote. The mark is a journal position.
    /// * Only on disk. A dry run stages rather than writes, so the journal
    ///   is empty and this is a no-op — staging is already the undo.
    fn revert_broken_step(&mut self, mark: &str, outcome: &Outcome) -> Vec<PathBuf> {
        let Some(verdict) = &outcome.verdict else {
            return Vec::new();
        };
        if verdict.passed {
            return Vec::new();
        }
        let Ok(restored) = self.ctx.rewind(mark) else {
            return Vec::new();
        };
        for path in &restored {
            let _ = self.scribe.refresh(path);
            let gone = !path.exists() && !self.ctx.root().join(path).exists();
            if gone {
                self.changed.remove(path);
            }
        }
        restored
    }

    /// The label every turn is checkpointed under.
    ///
    /// One rolling mark rather than one per turn: undoing is something people
    /// want for the turn that just went wrong, and a growing set of labels
    /// nobody names is a leak with a filing system. Rewinding further back is
    /// what version control is for.
    pub const LAST_TURN: &'static str = "turn";

    /// Take the checkpoint a later [`Talos::undo_turn`] rewinds to.
    ///
    /// Costs a `Vec::len()` — see [`ToolCtx::checkpoint`] — so doing it on
    /// every turn is free.
    fn mark_turn(&mut self) {
        self.ctx.checkpoint(Self::LAST_TURN);
    }

    /// Put the workspace back as it was before the last turn began.
    ///
    /// Returns the files that changed. Errors when no turn has run yet, which
    /// is the honest answer to "undo what?".
    pub fn undo_turn(&mut self) -> Result<Vec<PathBuf>> {
        let restored = self.ctx.rewind(Self::LAST_TURN)?;
        for path in &restored {
            // The index has to follow the files back, or the next turn reasons
            // about symbols that no longer exist.
            let _ = self.scribe.refresh(path);
            self.changed.remove(path);
        }
        Ok(restored)
    }

    /// Adopt a queue that already exists, instead of the one `new` built.
    ///
    /// Needed because of a construction order that cannot be avoided: anything
    /// which pushes interjections may have to exist *before* Talos does — a
    /// hook inside the [`ToolRegistry`] is built and moved in by `new`, and a
    /// front end may want the handle before the first run. Without this the
    /// only way to share one queue is to overwrite the field afterwards, which
    /// works and reads like a mistake.
    pub fn with_interjections(mut self, handle: crate::interject::Interjections) -> Self {
        self.interjections = handle;
        self
    }

    /// A handle for steering this run from outside it.
    ///
    /// Clone it before calling `run`, then push to it while the loop is going;
    /// what you push arrives at the next step boundary. See
    /// [`interject`](crate::interject) for why not sooner.
    pub fn interjections(&self) -> crate::interject::Interjections {
        self.interjections.clone()
    }

    /// Proposed changes, when running dry.
    pub fn diffs(&self) -> Vec<FileDiff> {
        self.ctx.diffs()
    }

    /// Write staged changes to disk and re-index them.
    pub fn apply(&mut self) -> Result<Vec<PathBuf>> {
        let written = self.ctx.apply_staged()?;
        for path in &written {
            let _ = self.scribe.refresh(path);
        }
        Ok(written)
    }

    /// Write only the selected hunks, leaving the rest staged for review.
    pub fn apply_hunks(&mut self, selection: &[(String, Vec<usize>)]) -> Result<Vec<PathBuf>> {
        let written = self.ctx.apply_hunks(selection)?;
        for path in &written {
            let _ = self.scribe.refresh(path);
        }
        Ok(written)
    }

    pub fn discard(&mut self) {
        self.ctx.discard_staged();
        self.changed.clear();
    }

    /// Whether this call may run.
    ///
    /// Only consequential calls are put to the user. Asking about every read
    /// would train them to approve without looking, which is worse than not
    /// asking at all — the same reasoning as `Talos._permitted` on the Python
    /// side, and it leans on `Tool::consequential`, which has no default, so a
    /// tool added later cannot compile without being classified.
    async fn permitted(&self, name: &str, input: &serde_json::Value) -> bool {
        let Some(approver) = &self.approver else {
            return true; // unattended
        };
        if !self.tools.is_consequential(name) {
            return true;
        }
        approver.approve(name, input).await
    }

    fn recall_brief(&self) -> Option<String> {
        if !self.episodes_enabled {
            return None;
        }
        let records = self.episodes.recall(&self.task, 8);
        let brief = episode::render_brief(&records);
        if brief.is_empty() {
            None
        } else {
            Some(brief)
        }
    }

    fn inject_recall(&mut self) {
        if let Some(brief) = self.recall_brief() {
            self.messages.push(Message::user_text(brief));
        }
    }

    fn begin_attempt(&mut self) {
        self.attempt_seq += 1;
        let id = format!("attempt-{}", self.attempt_seq);
        self.ctx.checkpoint(&id);
        self.current_attempt = id;
    }

    fn record_hypothesis(&mut self, step: usize, signature: &str, detail: &str) {
        if self.episodes_enabled {
            self.episodes.record(Record::Hypothesis {
                task: self.task.clone(),
                signature: signature.to_string(),
                detail: detail.to_string(),
                step,
            });
        }
        self.session.log(&TraceEvent::Hypothesis {
            step,
            signature: signature.to_string(),
            detail: detail.to_string(),
        });
    }

    fn persist_attempt(&mut self, halt: Halt, verdict: &Option<Verdict>) {
        if self.current_attempt.is_empty() {
            return;
        }
        let summary = match verdict {
            Some(v) => v.summary(),
            None => "never reached verification".to_string(),
        };
        if self.episodes_enabled {
            self.episodes.record(Record::Attempt {
                id: self.current_attempt.clone(),
                parent: self.attempt_parent.clone(),
                halt: halt.label().to_string(),
                changed: self
                    .changed
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect(),
                summary,
                task: self.task.clone(),
            });
        }
    }

    /// One redirect: remember what failed, optionally rewind a broken tree,
    /// then keep going on a fresh attempt under the same budget.
    async fn redirect(
        &mut self,
        step: usize,
        forbidden: &[String],
        last_verdict: &Option<Verdict>,
    ) {
        let from = self.current_attempt.clone();
        self.persist_attempt(Halt::Stuck, last_verdict);
        if self.episodes_enabled {
            self.episodes.record(Record::Redirect {
                from_attempt: from.clone(),
                forbidden: forbidden.to_vec(),
                reason: "stuck".to_string(),
                task: self.task.clone(),
            });
        }

        let broken = last_verdict.as_ref().is_some_and(|v| !v.passed);
        if broken && !from.is_empty() {
            if let Ok(restored) = self.ctx.rewind(&from) {
                for path in &restored {
                    let _ = self.scribe.refresh(path);
                    self.changed.remove(path);
                }
            }
        }

        self.attempt_parent = Some(from.clone());
        self.begin_attempt();

        if !self.plan_remainder.is_empty() {
            let remaining = self.plan_remainder.join("\n");
            let forbid = if forbidden.is_empty() {
                String::new()
            } else {
                format!("\nDo not retry these calls:\n{}", forbidden.join("\n"))
            };
            let task = format!("{}\n\nRemaining work:\n{remaining}{forbid}", self.task);
            if let Ok(plan) = crate::metis::plan(
                self.engine.as_ref(),
                &self.themis,
                &self.scribe,
                &task,
                self.max_tokens,
            )
            .await
            {
                if !plan.is_empty() {
                    self.plan_remainder = plan.steps;
                }
            }
        }

        let mut note = String::from(
            "SUPERVISOR: consecutive steps made no progress. Do not retry the \
             approaches below. Take a different approach, or state plainly what \
             is blocking you.",
        );
        if !forbidden.is_empty() {
            note.push_str("\nAlready tried:\n");
            for sig in forbidden {
                note.push_str("- ");
                note.push_str(sig);
                note.push('\n');
            }
        }
        if let Some(brief) = self.recall_brief() {
            note.push('\n');
            note.push_str(&brief);
        }
        self.messages.push(Message::user_text(note));
        self.session.log(&TraceEvent::Redirected {
            step,
            attempt: from,
            forbidden: forbidden.to_vec(),
        });
    }

    async fn apply_verify(&mut self, step: usize, full: bool) -> Verdict {
        let files: Vec<PathBuf> = self.changed.iter().cloned().collect();
        let verdict = if self.ctx.is_dry_run() {
            self.oracle
                .verify_staged(self.scribe.adapter(), &self.ctx.staged_contents())
        } else if full {
            match self.oracle.verify(self.scribe.adapter(), &files).await {
                Ok(v) => v,
                Err(e) => Verdict {
                    passed: false,
                    reached_tier: 0,
                    tiers: vec![crate::oracle::TierResult {
                        tier: 0,
                        label: "verify".into(),
                        passed: false,
                        detail: format!("verify failed to run: {e:#}"),
                        skipped: false,
                        forgiven: false,
                    }],
                    dry_run: false,
                },
            }
        } else {
            self.oracle.quick(self.scribe.adapter(), &files)
        };
        for tier in &verdict.tiers {
            self.session.log(&TraceEvent::OracleVerdict {
                step,
                tier: tier.label.clone(),
                passed: tier.passed,
                summary: tier.detail.clone(),
            });
        }
        verdict
    }

    async fn drive(&mut self) -> Result<Outcome> {
        self.plan_remainder.clear();
        let require_action = self
            .require_action
            .unwrap_or_else(|| task_requires_action(&self.task));
        self.drive_with(self.ariadne, VerifyMode::Full, require_action)
            .await
    }

    async fn drive_with(
        &mut self,
        budget: Ariadne,
        verify: VerifyMode,
        require_action: bool,
    ) -> Result<Outcome> {
        // Before the agent changes anything: whatever fails now is not its
        // doing, and this is the only moment at which that can be established.
        // Idempotent, so a resumed session pays for it once. Skipped in a dry
        // run, where nothing reaches disk for a tier to look at.
        if !self.ctx.is_dry_run() {
            self.oracle.prepare(self.scribe.adapter()).await?;
        }
        self.prepare_mission_execution()?;

        self.begin_attempt();

        let mut noops = 0usize;
        let mut last_verdict: Option<Verdict> = None;
        let mut last_text = String::new();
        let mut redirects_used = 0usize;
        let mut forbidden: Vec<String> = Vec::new();
        let mut reusable_verdict: Option<Verdict> = None;
        // Successful calls to tools that can change something outside the
        // conversation, this turn. The discriminator between "verified" and
        // "verified nothing": every tier is satisfied vacuously by an empty
        // change set, so a passing verdict is only evidence of completion if
        // something actually happened.
        //
        // Not derived from `changed`, because a task whose work is a command --
        // "run the tests and tell me what breaks" -- legitimately writes no
        // file and is still real work. `run` is consequential, so those count.
        //
        // Not "any successful call" either, which is what this used to be. That
        // let a single `read_file` satisfy the check, so an engine could answer
        // "add a triple() to lib.rs" by reading lib.rs, asking to be verified,
        // and being told it had completed and verified the task -- over an
        // empty change set, having written nothing. Reading is how you find out
        // what to do; it is not doing it. Caught by
        // `reading_a_file_is_not_doing_the_task`.
        let mut acted = 0usize;
        // Recent tool-call signatures, for spotting a repeat. Local to this
        // drive rather than to the struct: each run gets a fresh budget, so it
        // should get a fresh idea of what "again" means.
        let mut recent: VecDeque<String> = VecDeque::with_capacity(FUTILE_WINDOW);

        for step in 1..=budget.max_steps {
            // Checked before the engine call rather than after, so cancelling
            // stops the next request going out instead of paying for a turn
            // whose answer is already unwanted.
            // Cleared as it is read, so a cancellation stops exactly one turn
            // rather than latching and killing the next thing the user asks for.
            if self.cancel.swap(false, Ordering::SeqCst) {
                return self.finish(
                    Halt::Cancelled,
                    step - 1,
                    last_verdict,
                    &last_text,
                    matches!(verify, VerifyMode::Full),
                );
            }

            if let Some(note) = budget.pressure(step) {
                self.messages.push(Message::user_text(note));
            }

            // Drained here, at the one point in the step where the conversation
            // is a complete exchange. Pushed after the pressure note so the
            // user's words are the last thing before the request rather than
            // buried behind the harness's own prompting.
            let interjected = self.interjections.drain();
            if !interjected.is_empty() {
                self.session.log(&TraceEvent::Interjected {
                    step,
                    notes: interjected.clone(),
                });
                self.messages
                    .push(Message::user_text(crate::interject::Interjections::render(
                        &interjected,
                    )));
            }

            self.session.log(&TraceEvent::StepStarted {
                index: step,
                description: format!("engine turn {step}"),
            });

            // Local serving backends may only know the resident model's real
            // window after setup/discovery. Resolve the prompt budget after
            // that point, and never treat an unknown window as unlimited.
            self.engine.prepare().await?;
            let mut context = self.context_policy.resolve(
                self.engine.context_window(),
                crate::lethe::DEFAULT_MAX_TOKENS as u32,
            );
            if self.context_policy.compaction_enabled {
                self.lethe.max_tokens = if self.context_policy.compact_at_tokens.is_some() {
                    context.compact_at_tokens as usize
                } else {
                    self.lethe
                        .max_tokens
                        .min(context.compact_at_tokens as usize)
                };
                context.compact_at_tokens = self.lethe.max_tokens.min(u32::MAX as usize) as u32;
            } else {
                context.compact_at_tokens = 0;
            }
            self.session.log(&TraceEvent::ContextBudgeted {
                step,
                estimated_tokens: crate::lethe::estimate_tokens(&self.messages),
                assigned_tokens: context.assigned_tokens,
                input_limit_tokens: context.input_limit_tokens,
                compact_at_tokens: context.compact_at_tokens,
                completion_reserve: context.completion_reserve,
                protocol_reserve: context.protocol_reserve,
                engine_tokens: context.engine_tokens,
            });

            // Bounded here, immediately before the conversation is spent,
            // rather than after each push -- the same placement as the Python
            // side, and for the same reason: this is the one point where the
            // whole request is known.
            let messages_before_compaction = self.messages.len();
            if self.context_policy.compaction_enabled && self.lethe.compact(&mut self.messages) {
                self.session.log(&TraceEvent::ContextCompacted {
                    step,
                    tokens: crate::lethe::estimate_tokens(&self.messages),
                    assigned_tokens: context.assigned_tokens,
                    input_limit_tokens: context.input_limit_tokens,
                    compact_at_tokens: context.compact_at_tokens,
                    engine_tokens: context.engine_tokens,
                });
                self.record_compaction(messages_before_compaction, self.messages.len())?;
                // Compacted tool output is exactly the evidence the next step
                // would have learned from. Put the durable remainder back
                // after shrinking, not before: Lethe would elide it too.
                self.inject_recall();
            }

            let req = Request::new(
                self.themis.system_prompt(&self.role, Some(&self.scribe)),
                self.messages.clone(),
            )
            .with_tools(self.tools.defs())
            .with_max_tokens(context.completion_reserve.max(1));

            let streamed = std::sync::atomic::AtomicBool::new(false);
            let resp = {
                let session = &self.session;
                engine::complete_stream(self.engine.as_ref(), &req, &|delta| match delta {
                    crate::engine::StreamDelta::Text(text) if !text.is_empty() => {
                        streamed.store(true, Ordering::SeqCst);
                        session.log(&TraceEvent::AgentMessage { step, text });
                    }
                    crate::engine::StreamDelta::Thought(text) if !text.is_empty() => {
                        session.log(&TraceEvent::Thought { step, text });
                    }
                    _ => {}
                })
                .await?
            };

            // Recorded before anything downstream touches either side, so the
            // pair is exactly what crossed the wire. Guarded rather than
            // logged unconditionally: the clone copies the whole conversation,
            // and a run that is not collecting should not pay for it.
            if self.session.collects_exchanges() {
                self.session.log_exchange(step, &req, &resp);
            }

            self.messages.push(resp.as_message());
            let reply_text = resp.text();
            if !reply_text.trim().is_empty() {
                last_text = reply_text.clone();
                // One-shot engines (MockEngine) never fire deltas; keep the
                // existing single AgentMessage so halt tests and traces stay
                // the same. Streaming engines already logged each chunk.
                if !streamed.load(Ordering::SeqCst) {
                    self.session.log(&TraceEvent::AgentMessage {
                        step,
                        text: reply_text.clone(),
                    });
                }
            }

            // Own the calls so `resp` is not borrowed across the awaits below.
            let calls: Vec<(String, String, serde_json::Value)> = resp
                .tool_uses()
                .into_iter()
                .map(|(id, name, input)| (id.to_string(), name.to_string(), input.clone()))
                .collect();

            let mut outcome = StepOutcome::default();

            if calls.is_empty() && reply_text.trim().is_empty() {
                // An engine that said nothing has not claimed to be finished,
                // so there is nothing to verify. Verifying here is what turns a
                // truncated reply into a *passing* run.
                self.messages
                    .push(Message::user_text(EMPTY_REPLY_NOTE.to_string()));
            } else if calls.is_empty() {
                // The engine believes it is finished. Oracle decides.
                // A `verify` call on the immediately previous step already ran
                // the ladder; running it twice would tax the suite and teach
                // the engine nothing.
                let files: Vec<PathBuf> = self.changed.iter().cloned().collect();
                let mut verdict = if let Some(prior) = reusable_verdict.take() {
                    prior
                } else if self.ctx.is_dry_run() {
                    // Nothing is on disk, so cargo would compile the old code
                    // and report a pass about the wrong source.
                    self.oracle
                        .verify_staged(self.scribe.adapter(), &self.ctx.staged_contents())
                } else if matches!(verify, VerifyMode::Quick) {
                    self.oracle.quick(self.scribe.adapter(), &files)
                } else {
                    self.oracle.verify(self.scribe.adapter(), &files).await?
                };

                if last_verdict.as_ref().map(|v| v.summary()) != Some(verdict.summary()) {
                    for tier in &verdict.tiers {
                        self.session.log(&TraceEvent::OracleVerdict {
                            step,
                            tier: tier.label.clone(),
                            passed: tier.passed,
                            summary: tier.detail.clone(),
                        });
                    }
                }

                // Tier 4 is reachable only once every deterministic tier passed
                // — which `deterministic_tiers_passed` denies for a dry run.
                if matches!(verify, VerifyMode::Full)
                    && verdict.deterministic_tiers_passed()
                    && self.judge
                {
                    let t4 = crate::oracle::judge(
                        self.engine.as_ref(),
                        &self.themis,
                        &self.task,
                        self.oracle.root(),
                        &files,
                        self.max_tokens,
                    )
                    .await?;
                    self.session.log(&TraceEvent::OracleVerdict {
                        step,
                        tier: t4.label.clone(),
                        passed: t4.passed,
                        summary: t4.detail.clone(),
                    });
                    verdict.passed = t4.passed;
                    verdict.reached_tier = 4;
                    verdict.tiers.push(t4);
                }
                self.record_verdict(&verdict, matches!(verify, VerifyMode::Full))?;

                // A pass over a run that did nothing is not a completion. The
                // engine may request verification at any point; it may not be
                // told it succeeded merely by declining to act.
                outcome.verdict_passed = Some(verdict.passed && (acted > 0 || !require_action));
                if !verdict.passed {
                    self.messages.push(Message::user_text(verdict.report()));
                } else if acted == 0 && require_action {
                    self.messages
                        .push(Message::user_text(NOTHING_DONE_NOTE.to_string()));
                }
                last_verdict = Some(verdict);
            } else {
                let mut results = Vec::new();
                for (id, name, input) in &calls {
                    let consequential = self.tools.is_consequential(name);
                    let governed = consequential && self.approver.is_some();
                    let permitted = self.permitted(name, input).await;
                    if governed {
                        self.append_mission(MissionEvent::ApprovalDecision {
                            lease_id: id.clone(),
                            capability: name.clone(),
                            allowed: permitted,
                        })?;
                    }
                    if !permitted {
                        // A refusal is a result, not an error: the engine sees
                        // it and can propose something else on its next step.
                        // Ending the run would discard a whole conversation
                        // over a decision the user is entitled to make.
                        //
                        // "not permitted" rather than "the user said no": this
                        // also covers a disconnected front end, and telling the
                        // engine a human rejected it when none was consulted
                        // invites it to abandon a task nobody declined.
                        outcome.tool_calls += 1;
                        self.session.log(&TraceEvent::ToolCall {
                            step,
                            tool: name.clone(),
                            input: input.clone(),
                            output: format!("{name} was not permitted"),
                            is_error: true,
                            changed: Vec::new(),
                        });
                        results.push(Content::ToolResult {
                            id: id.clone(),
                            content: format!("{name} was not permitted"),
                            is_error: true,
                        });
                        continue;
                    }
                    if consequential {
                        let revision = self
                            .mission
                            .as_ref()
                            .map_or(0, |mission| mission.state().workspace.revision);
                        self.append_mission(MissionEvent::ConsequentialIntent {
                            intent: ActionIntent {
                                call_id: id.clone(),
                                tool: name.clone(),
                                input_hash: evidence_digest(input.to_string().as_bytes()),
                                at_workspace_revision: revision,
                            },
                        })?;
                    }
                    let out = if name == crate::tools::verify::NAME {
                        let full = crate::tools::verify::wants_full(input);
                        let verdict = self.apply_verify(step, full).await;
                        self.record_verdict(&verdict, full)?;
                        let content = verdict.report();
                        last_verdict = Some(verdict.clone());
                        reusable_verdict = Some(verdict);
                        crate::tools::ToolOutput::ok(content)
                    } else {
                        self.tools.dispatch(name, input, &self.ctx).await
                    };
                    if consequential {
                        self.record_consequential_result(id, name, &out)?;
                    }
                    outcome.tool_calls += 1;
                    outcome.files_changed += out.changed.len();
                    if !out.is_error && self.tools.is_consequential(name) {
                        acted += 1;
                    }

                    for path in &out.changed {
                        self.changed.insert(path.clone());
                        // Keep the exact index in step with the agent's edits.
                        // Read through the context so a dry run indexes the
                        // staged content rather than the stale file on disk.
                        if let Ok(source) = self.ctx.read(path) {
                            self.scribe.refresh_from(path, &source);
                        }
                    }

                    if out.is_error {
                        let one = signature(&[(id.clone(), name.clone(), input.clone())]);
                        self.record_hypothesis(step, &one, &out.content);
                    }

                    self.session.log(&TraceEvent::ToolCall {
                        step,
                        tool: name.clone(),
                        input: input.clone(),
                        output: out.content.clone(),
                        is_error: out.is_error,
                        changed: out
                            .changed
                            .iter()
                            .map(|p| p.display().to_string())
                            .collect(),
                    });

                    results.push(Content::ToolResult {
                        id: id.clone(),
                        content: out.content,
                        is_error: out.is_error,
                    });
                }
                self.messages.push(Message::user(results));

                let sig = signature(&calls);
                outcome.repeated = recent.contains(&sig) || forbidden.iter().any(|f| f == &sig);
                if outcome.files_changed > 0 {
                    reusable_verdict = None;
                }
                // A step that changed something is where "again" starts over:
                // whatever the model was circling, it is no longer circling it,
                // and the reads that led up to the change must not be held
                // against the reads that follow it.
                //
                // The window is emptied *and* this step is then remembered, not
                // skipped. Dropping it would lose the plainest loop of all: one
                // edit that lands, then the identical edit re-issued forever,
                // each retry changing nothing because the first one worked.
                if outcome.files_changed > 0 {
                    recent.clear();
                }
                if recent.len() == FUTILE_WINDOW {
                    recent.pop_front();
                }
                recent.push_back(sig.clone());
                if outcome.is_futile() {
                    self.record_hypothesis(step, &sig, "repeated call changed nothing");
                    self.messages
                        .push(Message::user_text(REPEATED_CALL_NOTE.to_string()));
                }
            }

            // `made_progress`, not `is_noop`: a step that repeated a recent
            // call and changed nothing is unproductive even though it called a
            // tool, and before `is_futile` existed here that step reset the
            // counter and bought the loop another turn.
            if outcome.made_progress() {
                noops = 0;
            } else {
                noops += 1;
            }

            let halt = budget.assess(step, &outcome, noops);
            if halt == Halt::Stuck && redirects_used < MAX_REDIRECTS {
                redirects_used += 1;
                // Snapshot the window, not every idle call of the drive:
                // a write that made progress must still be allowed to retry
                // an earlier read. After the redirect those signatures are
                // the thing not to repeat.
                let snap: Vec<String> = recent.iter().cloned().collect();
                self.redirect(step, &snap, &last_verdict).await;
                forbidden = snap;
                noops = 0;
                recent.clear();
                continue;
            }
            if halt.is_terminal() {
                return self.finish(
                    halt,
                    step,
                    last_verdict,
                    &last_text,
                    matches!(verify, VerifyMode::Full),
                );
            }
        }

        // Unreachable in practice: `assess` returns BudgetExhausted at
        // max_steps. Kept so the function is total rather than relying on it.
        let steps = budget.max_steps;
        self.finish(
            Halt::BudgetExhausted,
            steps,
            last_verdict,
            &last_text,
            matches!(verify, VerifyMode::Full),
        )
    }

    fn finish_mission(&mut self, halt: Halt, final_drive: bool) -> Result<()> {
        if !final_drive || self.mission.is_none() {
            return Ok(());
        }
        if halt == Halt::Done {
            if self
                .mission
                .as_ref()
                .map(|mission| mission.state().focus.phase)
                == Some(MissionPhase::Execute)
            {
                self.transition_mission(MissionPhase::Evaluate, "execution loop completed")?;
                self.transition_mission(
                    MissionPhase::Challenge,
                    "deterministic evaluation completed",
                )?;
                self.transition_mission(MissionPhase::Verify, "final proof was evaluated")?;
            }
            let verified = self.mission.as_ref().is_some_and(|mission| {
                mission.state().verification.final_verdict == FinalVerdict::Verified
            });
            if verified
                && self
                    .mission
                    .as_ref()
                    .map(|mission| mission.state().focus.phase)
                    == Some(MissionPhase::Verify)
            {
                self.transition_mission(
                    MissionPhase::Handoff,
                    "current proof permits operator handoff",
                )?;
            }
        } else {
            let to = match halt {
                Halt::Stuck => MissionPhase::Blocked,
                Halt::BudgetExhausted | Halt::Cancelled => MissionPhase::Paused,
                Halt::Done | Halt::Continue => unreachable!(),
            };
            self.transition_mission(to, halt.label())?;
        }
        let safe = self
            .mission
            .as_ref()
            .is_some_and(|mission| mission.state().focus.phase == MissionPhase::Handoff);
        let revision = self
            .mission
            .as_ref()
            .map_or(0, |mission| mission.state().workspace.revision);
        self.append_mission(MissionEvent::Checkpoint {
            id: format!("finish-revision-{revision}"),
            safe,
        })
    }

    fn finish(
        &mut self,
        halt: Halt,
        step: usize,
        verdict: Option<Verdict>,
        last_text: &str,
        final_drive: bool,
    ) -> Result<Outcome> {
        self.finish_mission(halt, final_drive)?;
        if final_drive {
            self.save_conversation()?;
        }
        self.persist_attempt(halt, &verdict);
        let summary = self.summarize(halt, &verdict, last_text);
        self.session.log(&TraceEvent::Halt {
            step,
            reason: halt.label().to_string(),
            detail: summary.clone(),
        });
        self.session.log(&TraceEvent::TaskFinished {
            steps_used: step,
            outcome: halt.label().to_string(),
        });

        Ok(Outcome {
            halt,
            steps_used: step,
            changed: self.changed.iter().cloned().collect(),
            verdict,
            summary,
            dry_run: self.ctx.is_dry_run(),
        })
    }

    fn summarize(&self, halt: Halt, verdict: &Option<Verdict>, last_text: &str) -> String {
        let verdict_line = match verdict {
            Some(v) => v.summary(),
            None => "never reached verification".to_string(),
        };
        let base = match halt {
            Halt::Done if self.ctx.is_dry_run() => format!(
                "Finished (preview only, nothing written) — {verdict_line}. \
                 Final answer: {last_text}"
            ),
            Halt::Done => {
                format!("Completed and verified — {verdict_line}. Final answer: {last_text}")
            }
            Halt::Stuck => format!(
                "Stopped: consecutive steps made no progress. Verification: {verdict_line}. \
                 Last message: {last_text}"
            ),
            Halt::BudgetExhausted => format!(
                "Stopped: step budget of {} exhausted. Verification: {verdict_line}. \
                 Last message: {last_text}",
                self.ariadne.max_steps
            ),
            // Says what was kept, not just that it stopped. A cancelled turn
            // may have already edited files, and the person who cancelled needs
            // to know that before deciding whether to accept or undo.
            Halt::Cancelled => format!(
                "Cancelled; {} file(s) already changed. Verification: {verdict_line}. \
                 Last message: {last_text}",
                self.changed.len(),
            ),
            Halt::Continue => "still running".to_string(),
        };
        base
    }
}
