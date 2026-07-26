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

use std::collections::BTreeSet;
use std::path::PathBuf;

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
    /// Start a fresh task, discarding any previous conversation.
    pub async fn run(&mut self, task: &str, plan: &Plan) -> Result<Outcome> {
        self.session.log(&TraceEvent::PlanProduced { steps: plan.steps.clone() });

        self.task = task.to_string();
        self.changed.clear();
        self.messages = vec![Message::user_text(format!(
            "Task: {task}\n\nPlan:\n{}\n\nWork through it. When everything is complete, reply \
             with a short summary and no tool calls — verification runs automatically.",
            plan.render()
        ))];

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
        self.messages.push(Message::user_text(instruction.to_string()));
        self.drive().await
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

    async fn drive(&mut self) -> Result<Outcome> {
        let mut noops = 0usize;
        let mut last_verdict: Option<Verdict> = None;
        let mut last_text = String::new();

        for step in 1..=self.ariadne.max_steps {
            if let Some(note) = self.ariadne.pressure(step) {
                self.messages.push(Message::user_text(note));
            }

            self.session.log(&TraceEvent::StepStarted {
                index: step,
                description: format!("engine turn {step}"),
            });

            let req = Request::new(
                self.themis.system_prompt(EXECUTOR_ROLE, Some(&self.scribe)),
                self.messages.clone(),
            )
            .with_tools(self.tools.defs())
            .with_max_tokens(self.max_tokens);

            let resp = engine::complete(self.engine.as_ref(), &req).await?;
            self.messages.push(resp.as_message());
            if !resp.text().trim().is_empty() {
                last_text = resp.text();
            }

            // Own the calls so `resp` is not borrowed across the awaits below.
            let calls: Vec<(String, String, serde_json::Value)> = resp
                .tool_uses()
                .into_iter()
                .map(|(id, name, input)| (id.to_string(), name.to_string(), input.clone()))
                .collect();

            let mut outcome = StepOutcome::default();

            if calls.is_empty() {
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

                outcome.verdict_passed = Some(verdict.passed);
                if !verdict.passed {
                    self.messages.push(Message::user_text(verdict.report()));
                }
                last_verdict = Some(verdict);
            } else {
                let mut results = Vec::new();
                for (id, name, input) in &calls {
                    let out = self.tools.dispatch(name, input, &self.ctx).await;
                    outcome.tool_calls += 1;
                    outcome.files_changed += out.changed.len();

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
            }

            if outcome.is_noop() {
                noops += 1;
            } else {
                noops = 0;
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
            Halt::Continue => "still running".to_string(),
        };
        base
    }
}
