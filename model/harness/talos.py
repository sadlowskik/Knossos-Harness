"""Talos: the executor.

The bronze automaton that walked the shore of Crete three times a day. It does
the rounds; it does not decide what the rounds are.

# The one rule that shapes everything else

**Completion is decided by the verifier, not by the engine.** An engine that
stops calling tools is making a *request* for verification, not announcing
success. Only a passing verdict produces `Halt.DONE`. Without that rule
"finished" means "the model felt finished", which is precisely the claim the
whole harness exists to stop trusting.

# Why the transcript is re-rendered every turn

`Engine.generate(prompt, context, cancelled)` takes two strings and has no
concept of message history -- deliberately, because the slot must one day hold a
model that only maps bytes to bytes. So the conversation is re-rendered into
`prompt` on each step.

That is quadratic in tokens over a long run, and it is the honest cost of
keeping the engine interface narrow. Lethe (bounded context with
summarise-and-reset) is the intended fix; until it exists, the step ceiling is
what keeps the growth bounded, which is a blunt instrument rather than a
solution.

# What is not here

No planner: Metis produces a plan, Talos executes one. No verifier: Oracle will
implement the `Verifier` protocol below. Both are seams rather than omissions --
Talos is testable today against a scripted engine and a trivial verifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Protocol, Sequence

from .ariadne import Ariadne, Halt, StepOutcome
from .tools import ToolCall, ToolRegistry, ToolResult, parse_calls
from .workspace import Workspace

__all__ = ["Verdict", "Verifier", "accept_everything", "Event", "Outcome", "Talos",
           "EXECUTOR_ROLE"]


EXECUTOR_ROLE = (
    "You are Talos, the executing half of the Daedalus harness. You carry out a "
    "task by reading and editing files in the workspace and running commands. "
    "Work one step at a time and prefer the smallest change that works. When the "
    "task is complete, reply in prose with no tool call -- verification runs "
    "automatically at that point, and only a passing verification ends the run."
)


@dataclass
class Verdict:
    """The result of checking the work. Oracle will produce these."""

    passed: bool
    summary: str
    detail: str = ""

    def report(self) -> str:
        """What the engine is told after a verification round."""
        if self.passed:
            return f"Verification passed: {self.summary}"
        body = f"Verification FAILED: {self.summary}"
        return f"{body}\n\n{self.detail}\n\nFix this before continuing." if self.detail else body


class Verifier(Protocol):
    """Anything that can judge whether the work is done.

    Oracle's tiered ladder will satisfy this. Keeping it a protocol means Talos
    can be tested without one, and that a project with a different notion of
    "correct" can supply its own.
    """

    def __call__(self, ws: Workspace, changed: Sequence[Path]) -> Verdict:
        ...


def accept_everything(ws: Workspace, changed: Sequence[Path]) -> Verdict:
    """The default verifier: agrees with the engine.

    Deliberately named to be uncomfortable. A run using this has *no* check on
    correctness -- it is for testing the loop, not for doing work.
    """
    return Verdict(True, f"no verifier configured ({len(changed)} file(s) changed)")


@dataclass
class Event:
    """Progress, for a caller that wants to stream it (the ACP server does)."""

    kind: str                       # step | text | tool | verdict | halt
    step: int = 0
    text: str = ""
    call: Optional[ToolCall] = None
    result: Optional[ToolResult] = None
    verdict: Optional[Verdict] = None
    halt: Optional[Halt] = None


@dataclass
class Outcome:
    halt: Halt
    steps_used: int
    changed: List[Path] = field(default_factory=list)
    verdict: Optional[Verdict] = None
    summary: str = ""
    dry_run: bool = False

    @property
    def succeeded(self) -> bool:
        return self.halt is Halt.DONE


class Talos:
    def __init__(self, engine, workspace: Workspace,
                 tools: Optional[ToolRegistry] = None,
                 ariadne: Optional[Ariadne] = None,
                 verifier: Optional[Verifier] = None,
                 constitution: str = "") -> None:
        self.engine = engine
        self.ws = workspace
        self.tools = tools or ToolRegistry.default()
        self.ariadne = ariadne or Ariadne()
        self.verify = verifier or accept_everything
        self.constitution = constitution

        #: The rendered conversation so far. Persisted across `run` and
        #: `resume` so a user can redirect without losing context.
        self.transcript: List[str] = []
        self.changed: List[Path] = []
        self.task: str = ""

    # ------------------------------------------------------------------ api

    def run(self, task: str, context: str = "", plan: Sequence[str] = (),
            cancelled: Optional[Callable[[], bool]] = None,
            on_event: Optional[Callable[[Event], None]] = None) -> Outcome:
        """Start a fresh task, discarding any previous conversation."""
        self.task = task
        self.changed = []
        self.transcript = [self._opening(task, plan)]
        return self._drive(context, cancelled, on_event)

    def resume(self, instruction: str, context: str = "",
               cancelled: Optional[Callable[[], bool]] = None,
               on_event: Optional[Callable[[Event], None]] = None) -> Outcome:
        """Continue the existing conversation with new instructions.

        The step budget resets: each thing the user asks for gets its own
        allowance rather than one budget draining across a whole session.
        """
        if not self.transcript:
            return self.run(instruction, context, cancelled=cancelled, on_event=on_event)
        self.transcript.append(f"\n## User\n{instruction}")
        return self._drive(context, cancelled, on_event)

    # --------------------------------------------------------------- prompt

    def _opening(self, task: str, plan: Sequence[str]) -> str:
        parts = [f"## Task\n{task}"]
        if plan:
            steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan))
            parts.append(f"\n## Plan\n{steps}")
        return "\n".join(parts)

    def _prompt(self) -> str:
        header = [EXECUTOR_ROLE]
        if self.constitution:
            header.append(f"\n# Constitution\n\nThese apply to everything you do.\n\n"
                          f"{self.constitution}")
        header.append(f"\n{self.tools.render()}")
        return "\n".join(header) + "\n\n" + "\n".join(self.transcript)

    # ----------------------------------------------------------------- loop

    def _drive(self, context: str,
               cancelled: Optional[Callable[[], bool]],
               on_event: Optional[Callable[[Event], None]]) -> Outcome:
        alive = cancelled or (lambda: False)
        emit = on_event or (lambda _event: None)

        noops = 0
        last_verdict: Optional[Verdict] = None
        last_text = ""

        for step in range(1, self.ariadne.max_steps + 1):
            if alive():
                return self._finish(Halt.STUCK, step, last_verdict,
                                    "cancelled by the caller", emit)

            nudge = self.ariadne.pressure(step)
            if nudge:
                self.transcript.append(f"\n## Budget\n{nudge}")

            emit(Event(kind="step", step=step))

            reply = "".join(self._stream(self._prompt(), context, alive))
            prose, calls = parse_calls(reply)
            self.transcript.append(f"\n## Assistant\n{reply.strip()}")
            if prose:
                last_text = prose
                emit(Event(kind="text", step=step, text=prose))

            outcome = StepOutcome()

            if calls:
                results = self._run_calls(calls, step, emit)
                outcome.tool_calls = len(calls)
                outcome.files_changed = sum(len(r.changed) for r in results)
                self.transcript.append(self._render_results(calls, results))
            else:
                # The engine believes it is finished. The verifier decides.
                verdict = self.verify(self.ws, list(self.changed))
                last_verdict = verdict
                outcome.verdict_passed = verdict.passed
                emit(Event(kind="verdict", step=step, verdict=verdict))
                if not verdict.passed:
                    self.transcript.append(f"\n## Verification\n{verdict.report()}")

            noops = noops + 1 if outcome.is_noop else 0

            halt = self.ariadne.assess(step, outcome, noops)
            if halt.is_terminal:
                return self._finish(halt, step, last_verdict, last_text, emit)

        # Unreachable: `assess` returns BUDGET_EXHAUSTED at max_steps. Kept so
        # the function is total rather than relying on that.
        return self._finish(Halt.BUDGET_EXHAUSTED, self.ariadne.max_steps,
                            last_verdict, last_text, emit)

    def _stream(self, prompt: str, context: str,
                alive: Callable[[], bool]) -> Iterable[str]:
        for chunk in self.engine.generate(prompt, context, alive):
            # Reasoning text is not the answer; it must not be parsed for calls.
            if type(chunk).__name__ == "Thought":
                continue
            yield str(chunk)

    def _run_calls(self, calls: Sequence[ToolCall], step: int,
                   emit: Callable[[Event], None]) -> List[ToolResult]:
        results: List[ToolResult] = []
        for call in calls:
            result = self.tools.dispatch(call, self.ws)
            for path in result.changed:
                if path not in self.changed:
                    self.changed.append(path)
            emit(Event(kind="tool", step=step, call=call, result=result))
            results.append(result)
        return results

    def _render_results(self, calls: Sequence[ToolCall],
                        results: Sequence[ToolResult]) -> str:
        blocks = ["\n## Tool results"]
        for call, result in zip(calls, results):
            status = "ERROR" if result.is_error else "ok"
            blocks.append(f"\n### {call.name} [{status}]\n{result.content}")
        return "\n".join(blocks)

    # ------------------------------------------------------------- finishing

    def _finish(self, halt: Halt, step: int, verdict: Optional[Verdict],
                last_text: str, emit: Callable[[Event], None]) -> Outcome:
        summary = self._summarise(halt, verdict, last_text)
        emit(Event(kind="halt", step=step, halt=halt, text=summary))
        return Outcome(halt=halt, steps_used=step, changed=list(self.changed),
                       verdict=verdict, summary=summary, dry_run=self.ws.dry_run)

    def _summarise(self, halt: Halt, verdict: Optional[Verdict], last_text: str) -> str:
        checked = verdict.summary if verdict else "never reached verification"
        if halt is Halt.DONE:
            if self.ws.dry_run:
                return f"Finished (preview only, nothing written) -- {checked}."
            return f"Completed and verified -- {checked}."
        if halt is Halt.STUCK:
            return (f"Stopped: consecutive steps made no progress. "
                    f"Verification: {checked}. Last message: {last_text}")
        if halt is Halt.BUDGET_EXHAUSTED:
            return (f"Stopped: step budget of {self.ariadne.max_steps} exhausted. "
                    f"Verification: {checked}. Last message: {last_text}")
        return "still running"

    # ------------------------------------------------------------- reviewing

    def apply(self, only: Optional[List[Path]] = None) -> List[Path]:
        """Write staged changes to disk. Only meaningful after a dry run."""
        return self.ws.apply(only)

    def discard(self) -> None:
        self.ws.discard()
        self.changed = []
