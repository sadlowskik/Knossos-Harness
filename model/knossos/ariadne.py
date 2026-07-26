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

    @property
    def is_noop(self) -> bool:
        """A step that called no tools and changed no files produced nothing."""
        return self.tool_calls == 0 and self.files_changed == 0


@dataclass
class Ariadne:
    #: Hard ceiling. Termination is guaranteed by this and nothing else.
    max_steps: int = 12
    #: Where pressure begins -- the analogue of the geometric prior's mean.
    target_steps: int = 6
    #: Consecutive no-op steps tolerated before declaring `STUCK`.
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
        """
        if next_step <= self.target_steps:
            return None

        remaining = max(0, self.max_steps - next_step + 1)

        if remaining <= 1:
            return (f"BUDGET: this is your final step ({next_step} of {self.max_steps}). "
                    f"Stop making changes. Summarise what you completed and state plainly "
                    f"what remains unfinished.")
        if remaining <= 3:
            return (f"BUDGET: step {next_step} of {self.max_steps}, {remaining} remaining. "
                    f"Finish the current change and verify it. Do not begin anything new.")
        return (f"BUDGET: step {next_step} of {self.max_steps} (target was "
                f"{self.target_steps}). Prefer finishing over exploring. If you are "
                f"blocked, say so rather than trying another angle.")
