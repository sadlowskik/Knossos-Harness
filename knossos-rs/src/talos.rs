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

use crate::ariadne::{Ariadne, Halt, StepOutcome};
use crate::diff::FileDiff;
use crate::engine::{self, Content, Engine, Message, Request};
use crate::metis::Plan;
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
fn signature(calls: &[(String, String, serde_json::Value)]) -> String {
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
/// one-shot CLI (`daedalus task`) is non-interactive by design, and a prompt
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
            approver: None,
            interjections: crate::interject::Interjections::new(),
            role: EXECUTOR_ROLE.to_string(),
            cancel: Arc::new(AtomicBool::new(false)),
            retrieval: None,
            messages: Vec::new(),
            changed: BTreeSet::new(),
            task: String::new(),
        }
    }
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
            .then(|| gate.argus().context(query, Self::CONTEXT_BUDGET, Self::CONTEXT_HOPS))
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

    /// Start a fresh task, discarding any previous conversation.
    pub async fn run(&mut self, task: &str, plan: &Plan) -> Result<Outcome> {
        self.session.log(&TraceEvent::PlanProduced { steps: plan.steps.clone() });

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
        self.messages = vec![Message::user_text(prompt)];

        self.mark_turn();
        self.drive().await
    }

    /// Continue the existing conversation with new instructions.
    ///
    /// The step budget resets: each thing the user asks for gets its own
    /// allowance, rather than one budget draining across a long session.
    pub async fn resume(&mut self, instruction: &str) -> Result<Outcome> {
        if self.messages.is_empty() {
            let plan = Plan { steps: vec![instruction.to_string()] };
            return self.run(instruction, &plan).await;
        }
        // Gated per turn, not once per session: the conversation may have moved
        // to a different part of the tree, and the turn that needs context is
        // rarely the one that opened the session.
        let mut prompt = instruction.to_string();
        if let Some(context) = self.repository_context(instruction) {
            prompt.push_str(&Self::frame_context(&context));
        }
        self.messages.push(Message::user_text(prompt));
        self.mark_turn();
        self.drive().await
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

    async fn drive(&mut self) -> Result<Outcome> {
        // Before the agent changes anything: whatever fails now is not its
        // doing, and this is the only moment at which that can be established.
        // Idempotent, so a resumed session pays for it once. Skipped in a dry
        // run, where nothing reaches disk for a tier to look at.
        if !self.ctx.is_dry_run() {
            self.oracle.prepare(self.scribe.adapter()).await?;
        }

        let mut noops = 0usize;
        let mut last_verdict: Option<Verdict> = None;
        let mut last_text = String::new();
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

        for step in 1..=self.ariadne.max_steps {
            // Checked before the engine call rather than after, so cancelling
            // stops the next request going out instead of paying for a turn
            // whose answer is already unwanted.
            // Cleared as it is read, so a cancellation stops exactly one turn
            // rather than latching and killing the next thing the user asks for.
            if self.cancel.swap(false, Ordering::SeqCst) {
                return Ok(self.finish(Halt::Cancelled, step - 1, last_verdict, &last_text));
            }

            if let Some(note) = self.ariadne.pressure(step) {
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
                self.messages.push(Message::user_text(
                    crate::interject::Interjections::render(&interjected),
                ));
            }

            self.session.log(&TraceEvent::StepStarted {
                index: step,
                description: format!("engine turn {step}"),
            });

            // Bounded here, immediately before the conversation is spent,
            // rather than after each push -- the same placement as the Python
            // side, and for the same reason: this is the one point where the
            // whole request is known.
            if self.lethe.compact(&mut self.messages) {
                self.session.log(&TraceEvent::ContextCompacted {
                    step,
                    tokens: crate::lethe::estimate_tokens(&self.messages),
                });
            }

            let req = Request::new(
                self.themis.system_prompt(&self.role, Some(&self.scribe)),
                self.messages.clone(),
            )
            .with_tools(self.tools.defs())
            .with_max_tokens(self.max_tokens);

            let resp = engine::complete(self.engine.as_ref(), &req).await?;

            // Recorded before anything downstream touches either side, so the
            // pair is exactly what crossed the wire. Guarded rather than
            // logged unconditionally: the clone copies the whole conversation,
            // and a run that is not collecting should not pay for it.
            if self.session.collects_exchanges() {
                self.session.log(&TraceEvent::Exchange {
                    step,
                    request: req.clone(),
                    response: resp.clone(),
                });
            }

            self.messages.push(resp.as_message());
            let reply_text = resp.text();
            if !reply_text.trim().is_empty() {
                last_text = reply_text.clone();
                self.session
                    .log(&TraceEvent::AgentMessage { step, text: reply_text.clone() });
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
                self.messages.push(Message::user_text(EMPTY_REPLY_NOTE.to_string()));
            } else if calls.is_empty() {
                // The engine believes it is finished. Oracle decides.
                let files: Vec<PathBuf> = self.changed.iter().cloned().collect();
                let mut verdict = if self.ctx.is_dry_run() {
                    // Nothing is on disk, so cargo would compile the old code
                    // and report a pass about the wrong source.
                    self.oracle
                        .verify_staged(self.scribe.adapter(), &self.ctx.staged_contents())
                } else {
                    self.oracle.verify(self.scribe.adapter(), &files).await?
                };

                for tier in &verdict.tiers {
                    self.session.log(&TraceEvent::OracleVerdict {
                        step,
                        tier: tier.label.clone(),
                        passed: tier.passed,
                        summary: tier.detail.clone(),
                    });
                }

                // Tier 4 is reachable only once every deterministic tier passed
                // — which `deterministic_tiers_passed` denies for a dry run.
                if verdict.deterministic_tiers_passed() && self.judge {
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

                // A pass over a run that did nothing is not a completion. The
                // engine may request verification at any point; it may not be
                // told it succeeded merely by declining to act.
                outcome.verdict_passed = Some(verdict.passed && acted > 0);
                if !verdict.passed {
                    self.messages.push(Message::user_text(verdict.report()));
                } else if acted == 0 {
                    self.messages.push(Message::user_text(NOTHING_DONE_NOTE.to_string()));
                }
                last_verdict = Some(verdict);
            } else {
                let mut results = Vec::new();
                for (id, name, input) in &calls {
                    if !self.permitted(name, input).await {
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
                    let out = self.tools.dispatch(name, input, &self.ctx).await;
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

                    self.session.log(&TraceEvent::ToolCall {
                        step,
                        tool: name.clone(),
                        input: input.clone(),
                        output: out.content.clone(),
                        is_error: out.is_error,
                        changed: out.changed.iter().map(|p| p.display().to_string()).collect(),
                    });

                    results.push(Content::ToolResult {
                        id: id.clone(),
                        content: out.content,
                        is_error: out.is_error,
                    });
                }
                self.messages.push(Message::user(results));

                let sig = signature(&calls);
                outcome.repeated = recent.contains(&sig);
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
                recent.push_back(sig);
                if outcome.is_futile() {
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

            let halt = self.ariadne.assess(step, &outcome, noops);
            if halt.is_terminal() {
                return Ok(self.finish(halt, step, last_verdict, &last_text));
            }
        }

        // Unreachable in practice: `assess` returns BudgetExhausted at
        // max_steps. Kept so the function is total rather than relying on it.
        let steps = self.ariadne.max_steps;
        Ok(self.finish(Halt::BudgetExhausted, steps, last_verdict, &last_text))
    }

    fn finish(
        &self,
        halt: Halt,
        step: usize,
        verdict: Option<Verdict>,
        last_text: &str,
    ) -> Outcome {
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

        Outcome {
            halt,
            steps_used: step,
            changed: self.changed.iter().cloned().collect(),
            verdict,
            summary,
            dry_run: self.ctx.is_dry_run(),
        }
    }

    fn summarize(&self, halt: Halt, verdict: &Option<Verdict>, last_text: &str) -> String {
        let verdict_line = match verdict {
            Some(v) => v.summary(),
            None => "never reached verification".to_string(),
        };
        let base = match halt {
            Halt::Done if self.ctx.is_dry_run() => {
                format!("Finished (preview only, nothing written) — {verdict_line}.")
            }
            Halt::Done => format!("Completed and verified — {verdict_line}."),
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
