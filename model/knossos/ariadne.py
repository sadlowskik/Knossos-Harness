"""Ariadne: when to stop.

The model-level Ariadne is a PonderNet halting head -- a learned per-token depth
allocation trained with a KL penalty toward a geometric prior. None of that
math transfers to an agent loop, and pretending it does would be decoration.

**What transfers is the measured failure mode.** In `daedalus/ariadne.py`, at
β=0.01 the halting distribution collapsed to maximum depth -- 7.5 of 8 steps, no
adaptivity at all -- while at β=0.1 it settled near 5 steps for a small accuracy
cost. The lesson is that without explicit and *increasing* pressure to stop, an
adaptive-compute system spends its whole budget on every input regardless of
difficulty. That is exactly the agent pathology of burning twelve iterations on
a one-line change.

So this module is not PonderNet. It is the three things that lesson says are
needed:

  * a **hard ceiling**, the analogue of forcing λ=1 at the final loop, so
    termination is guaranteed rather than hoped for;
  * **escalating pressure** past a target, the analogue of the KL term, stated
    in words because there is no gradient here;
  * a **deterministic stop signal** -- verification passing -- so the common case
    ends on evidence rather than on the model's opinion of its own work.

Stdlib only.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

__all__ = ["Halt", "StepOutcome", "Ariadne"]


class Halt(Enum):
    """Why a loop stopped, or that it did not."""

    CONTINUE = "continue"
    #: Verified complete. The only halt that means success.
    DONE = "done"
    #: Consecutive steps achieved nothing; more iterations will not help.
    STUCK = "stuck"
    #: Hit the ceiling. Mirrors the forced halt at the final loop.
    BUDGET_EXHAUSTED = "budget_exhausted"

    @property
    def is_terminal(self) -> bool:
        return self is not Halt.CONTINUE


@dataclass
class StepOutcome:
    """What one step actually accomplished.

    Ariadne decides on evidence, not on the engine's description of its own
    progress -- an engine that says "done" is making a request for
    verification, not a decision.
    """

    tool_calls: int = 0
    files_changed: int = 0
    #: Only `True` when the deterministic verification tiers passed.
    verdict_passed: Optional[bool] = None
    #: This step made exactly the same calls, with the same arguments, as the
    #: step before it. Set by the caller, which is the only party that can see
    #: more than one step.
    repeated: bool = False

    @property
    def is_noop(self) -> bool:
        """A step that called no tools and changed no files produced nothing."""
        return self.tool_calls == 0 and self.files_changed == 0

    @property
    def is_futile(self) -> bool:
        """A step that called tools, repeated itself exactly, and changed nothing.

        `is_noop` alone cannot see this. It requires `tool_calls == 0`, so an
        engine stuck re-issuing one failing `edit_file` looks productive on
        every step -- it *is* calling a tool -- and the loop runs to the ceiling
        instead of stopping. Twelve identical failures cost the same as twelve
        useful steps and teach nobody anything.

        The conjunction is what keeps this safe. Repetition alone is not
        failure: reading the same file twice while working toward different
        edits is ordinary. Repetition that also changed nothing is the loop.
        """
        return self.repeated and self.files_changed == 0

    @property
    def made_progress(self) -> bool:
        """Whether this step is worth granting another one after."""
        return not (self.is_noop or self.is_futile)


@dataclass
class Ariadne:
    #: Hard ceiling. Termination is guaranteed by this and nothing else.
    #:
    #: Raised from 12 on measurement, not on taste. On the hard eval tier the
    #: ceiling was binding rather than the model: doubling it to 24 took the
    #: score from 11/15 to 13/15, and the cases that flipped needed 15, 18, 19
    #: and 23 steps. Twelve was cutting off work that was going to succeed.
    #:
    #: The PonderNet lesson in the module docstring is about *pressure*, not
    #: about the wall -- it says a system with no escalating cost will spend its
    #: whole budget on every input. `target_steps` is what applies that cost and
    #: is deliberately left where it was, so the pressure begins at the same
    #: place and simply has further to escalate. Raising the wall without
    #: raising the target buys persistence on hard tasks without licensing
    #: sprawl on easy ones, which the core tier confirms: it still finishes in
    #: 3-7 steps.
    #:
    #: That "further to escalate" was the intent and was not what the code did:
    #: `pressure` banded on absolute counts, so the extra steps all fell into one
    #: band and got one repeated sentence. The bands are fractions of this
    #: ceiling now, which is what makes leaving the target here actually correct
    #: rather than merely intended.
    max_steps: int = 20
    #: Where pressure begins -- the analogue of the geometric prior's mean.
    target_steps: int = 6
    #: Consecutive unproductive steps tolerated before declaring `STUCK`. A step
    #: is unproductive when it achieved nothing (`is_noop`) or merely repeated
    #: the one before it (`is_futile`).
    stuck_after: int = 2

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        # A target above the ceiling would silently disable pressure entirely.
        self.target_steps = min(self.target_steps, self.max_steps)

    def assess(self, step: int, outcome: StepOutcome, consecutive_noops: int = 0) -> Halt:
        """Decide whether to continue after `step` (1-indexed).

        Order matters: success is checked before exhaustion, so a run that
        verifies on its last permitted step reports DONE rather than
        BUDGET_EXHAUSTED.
        """
        if outcome.verdict_passed is True:
            return Halt.DONE
        if step >= self.max_steps:
            return Halt.BUDGET_EXHAUSTED
        if consecutive_noops >= self.stuck_after:
            return Halt.STUCK
        return Halt.CONTINUE

    def pressure(self, next_step: int) -> Optional[str]:
        """Text to inject into the next prompt, once past `target_steps`.

        There is no gradient to attach a KL term to, so the penalty is stated
        and escalates as the ceiling approaches.

        **The bands are fractions of the ceiling, not step counts.** They used to
        be absolute -- `remaining <= 3` and everything else -- which was correct
        while the ceiling was 12 and silently wrong once it became 20. Two things
        went wrong at the larger size, both invisible to the tests, which pin
        `max_steps=12`:

        * The widest band returned one constant string, so from step 7 to step 17
          the engine was told the same thing eleven times running. That is a
          plateau, not escalation -- the β=0.01 failure this module exists to
          avoid, reintroduced at a different scale.
        * That constant string said *"prefer finishing over exploring"* and
          *"say so rather than trying another angle"*, which is sound advice with
          three steps left and actively harmful with thirteen. Trying another
          angle is the correct move most of the way through a long budget, and
          the harness was telling a capable model not to.

        Expressed as fractions the same four sentences land in the same places at
        `max_steps=12` and spread out properly at 20, so raising the ceiling
        again needs no further retuning here.
        """
        if next_step <= self.target_steps:
            return None

        remaining = max(0, self.max_steps - next_step + 1)
        # How much of the whole allowance is still ahead. The ceiling is at least
        # 1 by `__post_init__`, so this cannot divide by zero.
        share = remaining / self.max_steps

        if remaining <= 1:
            return (f"BUDGET: this is your final step ({next_step} of {self.max_steps}). "
                    f"Stop making changes. Summarise what you completed and state plainly "
                    f"what remains unfinished.")
        if share <= 0.25:
            return (f"BUDGET: step {next_step} of {self.max_steps}, {remaining} remaining. "
                    f"Finish the current change and verify it. Do not begin anything new.")
        if share <= 0.5:
            return (f"BUDGET: step {next_step} of {self.max_steps} (target was "
                    f"{self.target_steps}). Prefer finishing over exploring. If you are "
                    f"blocked, say so rather than trying another angle.")
        # Past the target but still holding most of the budget. This band exists
        # so the escalation has somewhere to start that is not already the
        # closing instruction: it reports the cost and asks for convergence
        # without withdrawing permission to change approach, which at this point
        # is usually the right move rather than a symptom.
        return (f"BUDGET: step {next_step} of {self.max_steps} (target was "
                f"{self.target_steps}), {remaining} remaining. You are past the "
                f"target, so favour converging on a working change. Changing "
                f"approach is still worth it if the current one is not working; "
                f"repeating it unchanged is not.")
