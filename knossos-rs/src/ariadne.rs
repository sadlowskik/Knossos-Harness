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
    /// The caller asked for the turn to stop.
    ///
    /// Not a judgement about progress, which is why Ariadne never returns it —
    /// it comes from outside the loop entirely. Distinct from every other halt
    /// because it is the one that is nobody's fault: a cancelled run must not
    /// look like a failed one in a trace or a front end.
    Cancelled,
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
            Halt::Cancelled => "cancelled",
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
    /// This step made exactly the same calls, with the same arguments, as a
    /// *recent* step — not necessarily the one immediately before it. Set by
    /// the caller, which is the only party that can see more than one step.
    ///
    /// See `talos::FUTILE_WINDOW` for how far back "recent" reaches. Comparing
    /// against the previous step alone cannot see a loop that alternates: a
    /// model going A, B, A, B never repeats itself consecutively.
    pub repeated: bool,
}

impl StepOutcome {
    /// A step that called no tools and changed no files produced nothing.
    pub fn is_noop(&self) -> bool {
        self.tool_calls == 0 && self.files_changed == 0
    }

    /// A step that called tools, repeated a recent step, and changed nothing.
    ///
    /// `is_noop` alone cannot see this, and until now it was the whole of the
    /// staleness check on this side. It requires `tool_calls == 0`, so an
    /// engine stuck re-issuing one failing `edit_file` looks productive on
    /// every step — it *is* calling a tool — and the run goes to the ceiling.
    /// Twenty identical failures cost the same as twenty useful steps and
    /// teach nobody anything.
    ///
    /// The conjunction is what keeps it safe. Repetition alone is not failure:
    /// reading the same file twice while working toward different edits is
    /// ordinary. Repetition that *also* changed nothing is the loop. That is
    /// what lets `repeated` look back further than one step without turning
    /// ordinary revisiting into a stall.
    pub fn is_futile(&self) -> bool {
        self.repeated && self.files_changed == 0
    }

    /// Whether this step is worth granting another one after.
    pub fn made_progress(&self) -> bool {
        !(self.is_noop() || self.is_futile())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
        // 20, not 12, matching Python. The ceiling was raised on measurement:
        // hard tasks were hitting the wall mid-fix. `target_steps` deliberately
        // stays at 6, so pressure begins in the same place and simply has
        // further to escalate — raising the wall without raising the target
        // buys persistence on hard tasks without licensing sprawl on easy ones.
        Ariadne {
            max_steps: 20,
            target_steps: 6,
            stuck_after: 2,
        }
    }
}

impl Ariadne {
    pub fn new(max_steps: usize, target_steps: usize) -> Self {
        Ariadne {
            max_steps,
            target_steps: target_steps.min(max_steps),
            ..Default::default()
        }
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
    ///
    /// **The bands are fractions of the ceiling, not step counts.** They used to
    /// be absolute — `remaining <= 3` and everything else — which was correct
    /// while the ceiling was 12 and silently wrong once it became 20. Two things
    /// go wrong at the larger size, both invisible to tests that pin
    /// `max_steps = 12`:
    ///
    /// * The widest band returns one constant string, so from step 7 to step 17
    ///   the engine hears the same thing eleven times running. That is a plateau,
    ///   not escalation — the β=0.01 failure this module exists to avoid,
    ///   reintroduced at a different scale.
    /// * That constant string says *"prefer finishing over exploring"* and *"say
    ///   so rather than trying another angle"*, which is sound advice with three
    ///   steps left and actively harmful with thirteen. Trying another angle is
    ///   the correct move most of the way through a long budget.
    ///
    /// Expressed as fractions the same four sentences land in the same places at
    /// `max_steps = 12` and spread out properly at 20.
    pub fn pressure(&self, next_step: usize) -> Option<String> {
        if next_step <= self.target_steps {
            return None;
        }
        let remaining = self.max_steps.saturating_sub(next_step - 1);
        // How much of the whole allowance is still ahead. `max_steps` is at
        // least 1 wherever an `Ariadne` is usable, so this cannot divide by zero
        // in practice; guard anyway rather than rely on it.
        let share = if self.max_steps == 0 {
            0.0
        } else {
            remaining as f64 / self.max_steps as f64
        };

        Some(if remaining <= 1 {
            format!(
                "BUDGET: this is your final step ({} of {}). Stop making changes. Summarize \
                 what you completed and state plainly what remains unfinished.",
                next_step, self.max_steps
            )
        } else if share <= 0.25 {
            format!(
                "BUDGET: step {} of {}, {} remaining. Finish the current change and verify it. \
                 Do not begin anything new.",
                next_step, self.max_steps, remaining
            )
        } else if share <= 0.5 {
            format!(
                "BUDGET: step {} of {} (target was {}). Prefer finishing over exploring. If you \
                 are blocked, say so rather than trying another angle.",
                next_step, self.max_steps, self.target_steps
            )
        } else {
            // Past the target but still holding most of the budget. This band
            // exists so the escalation has somewhere to start that is not
            // already the closing instruction: it reports the cost and asks for
            // convergence without withdrawing permission to change approach,
            // which at this point is usually the right move rather than a
            // symptom.
            format!(
                "BUDGET: step {} of {} (target was {}), {} remaining. You are past the target, \
                 so favour converging on a working change. Changing approach is still worth it \
                 if the current one is not working; repeating it unchanged is not.",
                next_step, self.max_steps, self.target_steps, remaining
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn passed() -> StepOutcome {
        StepOutcome {
            tool_calls: 1,
            files_changed: 1,
            verdict_passed: Some(true),
            ..Default::default()
        }
    }

    fn worked() -> StepOutcome {
        StepOutcome {
            tool_calls: 2,
            files_changed: 1,
            verdict_passed: None,
            ..Default::default()
        }
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
        let failed = StepOutcome {
            tool_calls: 1,
            files_changed: 1,
            verdict_passed: Some(false),
            ..Default::default()
        };
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

    /// The bands must scale with `max_steps`, not sit at absolute counts.
    ///
    /// Banded on absolute counts, raising the ceiling 12 -> 20 put steps 7
    /// through 17 in one band and handed the engine the same sentence eleven
    /// times. That is a constant penalty, which the module docs identify as
    /// equivalent to no penalty at all — and the suite did not notice, because
    /// every other pressure test pins `max_steps` at 12.
    #[test]
    fn pressure_does_not_plateau_at_the_shipped_ceiling() {
        let a = Ariadne::default(); // the shipped 20 / 6 / 2
        assert_eq!(a.max_steps, 20);

        let seen: Vec<String> = (7..=20).map(|s| a.pressure(s).unwrap()).collect();
        let distinct: std::collections::HashSet<&String> = seen.iter().collect();
        // Four distinct messages across the run, not one repeated.
        assert!(
            distinct.len() >= 4,
            "only {} distinct messages",
            distinct.len()
        );
    }

    /// With most of the budget left, trying another angle is the right move.
    ///
    /// The closing instruction ("say so rather than trying another angle") is
    /// sound with three steps remaining and wrong with thirteen. It escaped
    /// notice because at `max_steps = 12` step 7 really is halfway; at 20 it is
    /// barely a third.
    #[test]
    fn early_pressure_does_not_forbid_changing_approach() {
        let a = Ariadne::new(20, 6);
        let early = a.pressure(7).unwrap();

        assert!(!early.contains("rather than trying another angle"));
        assert!(early.contains("Changing approach is still worth it"));
        // ...and the closing bands still say it, at the point where it is true.
        assert!(a
            .pressure(17)
            .unwrap()
            .contains("Do not begin anything new"));
        assert!(a.pressure(20).unwrap().contains("final step"));
    }

    /// The whole point of fractions: the same four sentences land in the same
    /// places at the old ceiling, so this is a generalisation and not a retune.
    #[test]
    fn the_old_ceiling_still_bands_where_it_used_to() {
        let a = Ariadne::new(12, 6);
        assert!(a.pressure(7).unwrap().contains("Prefer finishing"));
        assert!(a
            .pressure(10)
            .unwrap()
            .contains("Do not begin anything new"));
        assert!(a.pressure(12).unwrap().contains("final step"));
    }

    #[test]
    fn noop_detection_is_about_evidence_not_opinion() {
        assert!(nothing().is_noop());
        assert!(!worked().is_noop());
        assert!(!StepOutcome {
            tool_calls: 1,
            files_changed: 0,
            verdict_passed: None,
            ..Default::default()
        }
        .is_noop());
    }

    #[test]
    fn target_cannot_exceed_the_ceiling() {
        let a = Ariadne::new(4, 99);
        assert_eq!(a.target_steps, 4);
    }
}
