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

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::Result;

use crate::ariadne::{Ariadne, Halt, StepOutcome};
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
}

#[derive(Debug, Clone)]
pub struct Outcome {
    pub halt: Halt,
    pub steps_used: usize,
    pub changed: Vec<PathBuf>,
    pub verdict: Option<Verdict>,
    pub summary: String,
}

impl Outcome {
    pub fn succeeded(&self) -> bool {
        self.halt == Halt::Done
    }
}

impl Talos {
    pub async fn run(&mut self, task: &str, plan: &Plan) -> Result<Outcome> {
        self.session.log(&TraceEvent::PlanProduced { steps: plan.steps.clone() });

        let mut messages = vec![Message::user_text(format!(
            "Task: {task}\n\nPlan:\n{}\n\nWork through it. When everything is complete, reply \
             with a short summary and no tool calls — verification runs automatically.",
            plan.render()
        ))];

        let mut changed: BTreeSet<PathBuf> = BTreeSet::new();
        let mut noops = 0usize;
        let mut last_verdict: Option<Verdict> = None;
        let mut last_text = String::new();

        for step in 1..=self.ariadne.max_steps {
            if let Some(note) = self.ariadne.pressure(step) {
                messages.push(Message::user_text(note));
            }

            self.session.log(&TraceEvent::StepStarted {
                index: step,
                description: format!("engine turn {step}"),
            });

            let req = Request::new(
                self.themis.system_prompt(EXECUTOR_ROLE, Some(&self.scribe)),
                messages.clone(),
            )
            .with_tools(self.tools.defs())
            .with_max_tokens(self.max_tokens);

            let resp = engine::complete(self.engine.as_ref(), &req).await?;
            messages.push(resp.as_message());
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
                let files: Vec<PathBuf> = changed.iter().cloned().collect();
                let mut verdict = self.oracle.verify(self.scribe.adapter(), &files).await?;

                for tier in &verdict.tiers {
                    self.session.log(&TraceEvent::OracleVerdict {
                        step,
                        tier: tier.label.clone(),
                        passed: tier.passed,
                        summary: tier.detail.clone(),
                    });
                }

                // Tier 4 is reachable only once every deterministic tier passed.
                if verdict.deterministic_tiers_passed() && self.judge {
                    let t4 = crate::oracle::judge(
                        self.engine.as_ref(),
                        &self.themis,
                        task,
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
                    messages.push(Message::user_text(verdict.report()));
                }
                last_verdict = Some(verdict);
            } else {
                let mut results = Vec::new();
                for (id, name, input) in &calls {
                    let out = self.tools.dispatch(name, input, &self.ctx).await;
                    outcome.tool_calls += 1;
                    outcome.files_changed += out.changed.len();

                    for path in &out.changed {
                        changed.insert(path.clone());
                        // Keep the exact index in step with the agent's edits.
                        let _ = self.scribe.refresh(path);
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
                messages.push(Message::user(results));
            }

            if outcome.is_noop() {
                noops += 1;
            } else {
                noops = 0;
            }

            let halt = self.ariadne.assess(step, &outcome, noops);
            if halt.is_terminal() {
                let summary = self.summarize(halt, &last_verdict, &last_text);
                self.session.log(&TraceEvent::Halt {
                    step,
                    reason: halt.label().to_string(),
                    detail: summary.clone(),
                });
                self.session.log(&TraceEvent::TaskFinished {
                    steps_used: step,
                    outcome: halt.label().to_string(),
                });

                return Ok(Outcome {
                    halt,
                    steps_used: step,
                    changed: changed.into_iter().collect(),
                    verdict: last_verdict,
                    summary,
                });
            }
        }

        // Unreachable in practice: `assess` returns BudgetExhausted at
        // max_steps. Kept so the function is total rather than relying on it.
        let summary = self.summarize(Halt::BudgetExhausted, &last_verdict, &last_text);
        Ok(Outcome {
            halt: Halt::BudgetExhausted,
            steps_used: self.ariadne.max_steps,
            changed: changed.into_iter().collect(),
            verdict: last_verdict,
            summary,
        })
    }

    fn summarize(&self, halt: Halt, verdict: &Option<Verdict>, last_text: &str) -> String {
        let verdict_line = match verdict {
            Some(v) => v.summary(),
            None => "never reached verification".to_string(),
        };
        match halt {
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
        }
    }
}
