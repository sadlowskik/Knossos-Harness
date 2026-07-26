//! Ariadne: the halting policy.
//!
//! The model-level Ariadne is a PonderNet halting head — a learned
//! per-token depth allocation trained with a KL penalty toward a geometric
//! prior. None of that math transfers to an agent loop, and pretending it does
//! would be cosplay.
//!
//! What transfers is the **measured failure mode**. In the research repo, at
//! β=0.01 the halting distribution collapsed to maximum depth (7.5 of 8 steps,
//! no adaptivity at all); at β=0.1 it settled at a healthy ~5 steps for a
//! small accuracy cost. The lesson is that without explicit, *increasing*
//! pressure to stop, an adaptive-compute system spends its entire budget on
//! every input regardless of difficulty. That is exactly the agent pathology
//! of burning twelve iterations on a one-line fix.
//!
//! So this module is not PonderNet. It is three things that lesson says you
//! need:
//!
//! - a **hard ceiling** (`max_steps`), the analogue of forcing `λ = 1` at the
//!   final loop, so termination is guaranteed rather than hoped for;
//! - **escalating pressure** past `target_steps`, the analogue of the KL term,
//!   applied as prompt text since there is no gradient here;
//! - a **strong deterministic stop signal** — Oracle passing — so the common
//!   case ends on evidence rather than on the model's self-assessment.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Halt {
    /// Keep going.
    Continue,
    /// Verified complete. The only halt that means success.
    Done,
    /// Consecutive steps achieved nothing; more iterations will not help.
    Stuck,
    /// Hit the ceiling. Mirrors the forced halt at the final loop.
    BudgetExhausted,
}

impl Halt {
    pub fn is_terminal(self) -> bool {
        !matches!(self, Halt::Continue)
    }

    pub fn label(self) -> &'static str {
        match self {
            Halt::Continue => "continue",
            Halt::Done => "done",
            Halt::Stuck => "stuck",
            Halt::BudgetExhausted => "budget_exhausted",
        }
    }
}

/// What one step actually accomplished. `Ariadne` decides on evidence, not on
/// the engine's opinion of its own progress.
#[derive(Debug, Clone, Copy, Default)]
pub struct StepOutcome {
    pub tool_calls: usize,
    pub files_changed: usize,
    /// `Some(true)` only when Oracle's deterministic ladder passed.
    pub verdict_passed: Option<bool>,
}

impl StepOutcome {
    /// A step that called no tools and changed no files produced nothing.
    pub fn is_noop(&self) -> bool {
        self.tool_calls == 0 && self.files_changed == 0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Ariadne {
    /// Hard ceiling. Termination is guaranteed by this, nothing else.
    pub max_steps: usize,
    /// Where pressure starts. The analogue of the geometric prior's mean.
    pub target_steps: usize,
    /// Consecutive no-op steps tolerated before declaring `Stuck`.
    pub stuck_after: usize,
}

impl Default for Ariadne {
    fn default() -> Self {
        Ariadne { max_steps: 12, target_steps: 6, stuck_after: 2 }
    }
}

impl Ariadne {
    pub fn new(max_steps: usize, target_steps: usize) -> Self {
        Ariadne { max_steps, target_steps: target_steps.min(max_steps), ..Default::default() }
    }

    /// Decide whether to continue after `step` (1-indexed).
    ///
    /// Order matters: success is checked before exhaustion so a run that
    /// verifies on its final permitted step reports `Done`, not
    /// `BudgetExhausted`.
    pub fn assess(&self, step: usize, outcome: &StepOutcome, consecutive_noops: usize) -> Halt {
        if outcome.verdict_passed == Some(true) {
            return Halt::Done;
        }
        if step >= self.max_steps {
            return Halt::BudgetExhausted;
        }
        if consecutive_noops >= self.stuck_after {
            return Halt::Stuck;
        }
        Halt::Continue
    }

    /// Pressure to inject into the next prompt, once past `target_steps`.
    ///
    /// There is no gradient to add a KL term to, so the penalty is stated in
    /// words and escalates as the ceiling approaches.
    pub fn pressure(&self, next_step: usize) -> Option<String> {
        if next_step <= self.target_steps {
            return None;
        }
        let remaining = self.max_steps.saturating_sub(next_step - 1);

        Some(if remaining <= 1 {
            format!(
                "BUDGET: this is your final step ({} of {}). Stop making changes. Summarize \
                 what you completed and state plainly what remains unfinished.",
                next_step, self.max_steps
            )
        } else if remaining <= 3 {
            format!(
                "BUDGET: step {} of {}, {} remaining. Finish the current change and verify it. \
                 Do not begin anything new.",
                next_step, self.max_steps, remaining
            )
        } else {
            format!(
                "BUDGET: step {} of {} (target was {}). Prefer finishing over exploring. If you \
                 are blocked, say so rather than trying another angle.",
                next_step, self.max_steps, self.target_steps
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn passed() -> StepOutcome {
        StepOutcome { tool_calls: 1, files_changed: 1, verdict_passed: Some(true) }
    }

    fn worked() -> StepOutcome {
        StepOutcome { tool_calls: 2, files_changed: 1, verdict_passed: None }
    }

    fn nothing() -> StepOutcome {
        StepOutcome::default()
    }

    #[test]
    fn oracle_passing_ends_the_run() {
        let a = Ariadne::new(12, 6);
        assert_eq!(a.assess(1, &passed(), 0), Halt::Done);
    }

    #[test]
    fn productive_steps_continue() {
        let a = Ariadne::new(12, 6);
        assert_eq!(a.assess(3, &worked(), 0), Halt::Continue);
    }

    /// The hard ceiling. Without this the loop has no termination guarantee.
    #[test]
    fn the_ceiling_forces_a_halt() {
        let a = Ariadne::new(5, 3);
        assert_eq!(a.assess(5, &worked(), 0), Halt::BudgetExhausted);
        assert_eq!(a.assess(99, &worked(), 0), Halt::BudgetExhausted);
    }

    #[test]
    fn success_on_the_final_step_reports_done_not_exhausted() {
        let a = Ariadne::new(5, 3);
        assert_eq!(a.assess(5, &passed(), 0), Halt::Done);
    }

    #[test]
    fn repeated_noops_are_stuck() {
        let a = Ariadne::new(12, 6);
        assert_eq!(a.assess(4, &nothing(), 1), Halt::Continue);
        assert_eq!(a.assess(4, &nothing(), 2), Halt::Stuck);
    }

    #[test]
    fn a_failed_verdict_does_not_end_the_run() {
        let a = Ariadne::new(12, 6);
        let failed = StepOutcome { tool_calls: 1, files_changed: 1, verdict_passed: Some(false) };
        assert_eq!(a.assess(2, &failed, 0), Halt::Continue);
    }

    #[test]
    fn no_pressure_before_the_target() {
        let a = Ariadne::new(12, 6);
        assert!(a.pressure(1).is_none());
        assert!(a.pressure(6).is_none());
    }

    #[test]
    fn pressure_escalates_toward_the_ceiling() {
        let a = Ariadne::new(12, 6);
        let mid = a.pressure(7).unwrap();
        let late = a.pressure(10).unwrap();
        let last = a.pressure(12).unwrap();

        assert!(mid.contains("Prefer finishing"));
        assert!(late.contains("Do not begin anything new"));
        assert!(last.contains("final step"));
    }

    #[test]
    fn noop_detection_is_about_evidence_not_opinion() {
        assert!(nothing().is_noop());
        assert!(!worked().is_noop());
        assert!(!StepOutcome { tool_calls: 1, files_changed: 0, verdict_passed: None }.is_noop());
    }

    #[test]
    fn target_cannot_exceed_the_ceiling() {
        let a = Ariadne::new(4, 99);
        assert_eq!(a.target_steps, 4);
    }
}
