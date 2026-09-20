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

import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Protocol, Sequence

from .ariadne import Ariadne, Halt, StepOutcome
from .engine import Usage
from .interject import Interjections
from .jsonrpc import log
from .lethe import Lethe, estimate_tokens
from .tools import (Tool, ToolCall, ToolRegistry, ToolResult, ToolSpec,
                    parse_calls)
from .workspace import Workspace

__all__ = ["Verdict", "Verifier", "accept_everything", "Event", "Outcome", "Talos",
           "EXECUTOR_ROLE"]


EXECUTOR_ROLE = (
    "You are Talos, the executing half of Knossos. You carry out a "
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

    Oracle's tiered ladder satisfies this. Keeping it a protocol means Talos can
    be tested without one, and that a project with a different notion of
    "correct" can supply its own.

    A verifier may *optionally* also define `prepare(ws)`, which Talos calls
    once before the first step so the verifier can look at the tree before the
    agent changes it -- Oracle uses it to record which checks were already
    failing. It is deliberately not part of the protocol: one required method
    keeps a bare function a valid verifier.
    """

    def __call__(self, ws: Workspace, changed: Sequence[Path]) -> Verdict:
        ...


class Replanner(Protocol):
    """Anything that can revise the rest of a plan once a step has failed.

    The plan is produced before the agent has read a single file, so it is a
    guess about a codebase nobody has looked at yet. Until this existed the
    guess was binding: `_drive_plan` walked the original list to the end and the
    transcript said, in as many words, *do not restart the plan*. A plan that
    was wrong about the shape of the problem therefore consumed the entire
    budget being wrong.

    `done` is what has already been attempted, `remaining` what has not, and
    `learned` is the failure text the engine just produced. Returning an empty
    sequence means "no revision" and leaves the original plan intact -- which is
    the right answer when the plan was fine and the step simply needs another
    attempt.
    """

    def __call__(self, task: str, learned: str, done: Sequence[str],
                 remaining: Sequence[str]) -> Sequence[str]:
        ...


#: Ceiling on a revised plan's length, as a multiple of what it replaced. A
#: planner handed its own failure can answer with twenty steps, and each one
#: would take a slice of a budget that is not growing to match.
REPLAN_GROWTH_LIMIT = 2

#: Transcript entries handed to a re-planner. Enough to carry the failure and
#: what produced it, without re-sending the whole run to a second model call.
REPLAN_CONTEXT_ENTRIES = 6

#: The name of the delegation tool, kept here so `Talos` can strip it from a
#: child's registry without importing the class into the registry module.
DELEGATE = "delegate"

#: How deep delegation may go. One, deliberately.
#:
#: A child that can delegate is a fork bomb with a token budget: each level
#: multiplies the number of live agents, and the failure is not a crash but a
#: bill. Depth one buys the thing worth having -- an independent subtask whose
#: reading does not land in the parent's context -- and nothing beyond it has
#: paid for itself in any harness the author is aware of.
MAX_DELEGATION_DEPTH = 1

#: How many recent tool-call signatures count as "again".
#:
#: This used to be one -- a step was a repeat only of the step immediately
#: before it -- which cannot see a loop that alternates. A model going
#: A, B, A, B, A, B never produces two consecutive identical signatures, so
#: `is_futile` never fired, `stuck_after` never tripped, and the run spent its
#: whole ceiling achieving nothing. That is the exact pathology Ariadne exists
#: to prevent, arriving through the one door the check did not cover.
#:
#: Four, because the window only has to be as long as the cycle it must close:
#: a cycle of length k is caught once k signatures fit, and the cycles that
#: actually occur are short (re-issuing one failing edit, alternating between
#: two, walking a three-step ritual). Longer than the cycle buys nothing, and
#: the ceiling remains the backstop for anything more baroque.
#:
#: Bounding it at all -- rather than remembering everything since the last
#: change -- keeps "again" meaning *recently*. An unbounded window would make
#: the detector more sensitive the longer a run had gone without progress,
#: which is a coupling nobody asked for and nothing would test.
FUTILE_WINDOW = 4


class Delegate(Tool):
    """Run one scoped subtask in a child agent, and return only its summary.

    # What this is actually for

    Not speed. The child runs to completion before the parent's turn continues,
    so nothing here is concurrent -- `parallel_safe` is False and must stay
    False, because two children sharing one `Workspace` would interleave writes
    to the same staging area.

    It is for **context**. The expensive part of a subtask is usually the
    reading: ten files opened to discover that one function needed changing. In
    a single agent all ten land in the transcript and are re-sent every
    subsequent step, which is what makes a long run cost quadratically. A child
    reads them into its own transcript, which is discarded when it finishes, and
    hands back a paragraph. The parent pays for the paragraph.

    # What it shares, and why

    **The workspace, by reference.** The child writes to the same staging area
    and the same undo journal, so its edits are ordinary edits -- visible to the
    parent's verifier, revertible by the parent's checkpoints, and counted in
    the parent's change set. The alternative, a private copy merged afterwards,
    means writing a merge algorithm and being wrong about conflicts.

    **The permission callback, by reference.** A child must not be a way around
    a gate the user is watching. `write_file` from a subagent prompts exactly as
    it would from the parent.

    **Not the transcript.** That is the entire point.
    """

    spec = ToolSpec(
        name=DELEGATE,
        description=(
            "Hand one self-contained subtask to a child agent that works in the "
            "same workspace but keeps its own context, and reports back a "
            "summary. Use it when a piece of work needs a lot of reading that "
            "you do not need to remember afterwards -- locating something "
            "across many files, or a mechanical change in a part of the tree "
            "you are not otherwise touching. The child cannot delegate further, "
            "and cannot ask you questions, so give it everything it needs in "
            "one description."),
        schema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": ("The complete subtask. The child sees none "
                                    "of this conversation, so state the goal, "
                                    "the relevant paths, and what done means."),
                },
            },
            "required": ["task"],
        },
    )

    #: Two children sharing one workspace would interleave writes into the same
    #: staging dict. See the class docstring: this tool buys context, not speed.
    parallel_safe = False

    def __init__(self, spawn: Callable[[str], "Outcome"]) -> None:
        self._spawn = spawn

    def run(self, args: Dict[str, Any], ws: Workspace) -> ToolResult:
        task = str(args.get("task") or "").strip()
        if not task:
            return ToolResult("delegate needs a `task`", is_error=True)
        try:
            outcome = self._spawn(task)
        except Exception as exc:                       # noqa: BLE001 - reported
            # A child that crashes is a failed subtask, not a failed run. The
            # parent still has its budget and can do the work itself.
            log(f"[talos] delegated task failed: {exc}")
            return ToolResult(f"the delegated task failed: {exc}", is_error=True)

        changed = list(outcome.changed)
        body = [f"Delegated task finished as {outcome.halt.value}.",
                outcome.summary]
        if changed:
            body.append("Files it changed: "
                        + ", ".join(ws.display(p) for p in changed))
        else:
            body.append("It changed no files.")
        # `is_error` on an unverified child, so the parent does not read a
        # halt-by-exhaustion as a completed subtask. It still gets the summary.
        return ToolResult("\n".join(body), is_error=not outcome.succeeded,
                          changed=changed)


def _signature(calls: Sequence[ToolCall]) -> str:
    """A stable identity for a step's tool calls.

    Name and arguments, in the order issued. `raw` is deliberately excluded:
    the same call formatted differently is the same call, and an engine caught
    in a loop often reformats slightly between attempts.

    `sort_keys` because two dicts with the same contents in a different order
    are the same arguments; `default=str` because an argument that is not JSON
    serialisable should degrade to a comparable string rather than raise inside
    the halting path.
    """
    return json.dumps([[c.name, c.args] for c in calls],
                      sort_keys=True, default=str)


def accept_everything(ws: Workspace, changed: Sequence[Path]) -> Verdict:
    """The default verifier: agrees with the engine.

    Deliberately named to be uncomfortable. A run using this has *no* check on
    correctness -- it is for testing the loop, not for doing work.
    """
    return Verdict(True, f"no verifier configured ({len(changed)} file(s) changed)")


class Entry(str):
    """A transcript entry that also remembers what it was.

    # Why a `str` subclass rather than a second structure

    The transcript is the harness's memory, and everything already reads it as a
    list of strings: `Lethe.compact` summarises it, `_prompt` joins it, and the
    tests assert on it. A structured conversation held *alongside* it would be a
    second copy of the same facts, and the way those two would drift is the
    specific failure this project exists to prevent -- compaction shrinks the
    transcript, the structured copy keeps growing, and the harness quietly sends
    a different conversation than the one it believes it is bounded to.

    So there is one list. `Entry` is a `str`, so `List[Entry]` *is* the
    `List[str]` every existing caller wants, byte for byte; the structure rides
    along as attributes for the one caller that can use it. `Thought`
    (`engine.py`) is the same trick for the same reason.

    Compaction turns summarised regions back into plain `str`, which drops the
    attributes. That is the correct degradation rather than a bug: a summarised
    region is no longer a faithful record of who said what, and `_history` falls
    back to reading its `## Assistant` headings exactly as it did before any of
    this existed.
    """

    #: `user`, `assistant`, or `tool`.
    role: str
    #: The calls this entry carries, when it is an assistant turn or the results
    #: of one.
    calls: "tuple"
    results: "tuple"
    #: The assistant's prose with tool-call fences removed. Used as the message
    #: content when the calls travel natively and the fences would be duplication.
    text: str
    #: Every call here arrived with a provider-assigned id, so this turn can be
    #: replayed as native `tool_calls` rather than as text.
    native: bool

    #: The model's reasoning for this turn, when the provider both produced it
    #: and accepts it back. Usually empty; see `Talos._reasoning_for`.
    reasoning: str

    def __new__(cls, value: str, *, role: str = "user",
                calls: Sequence[ToolCall] = (),
                results: Sequence[ToolResult] = (),
                text: str = "", native: bool = False,
                reasoning: str = "") -> "Entry":
        entry = super().__new__(cls, value)
        entry.role = role
        entry.calls = tuple(calls)
        entry.results = tuple(results)
        entry.text = text
        entry.native = native
        entry.reasoning = reasoning
        return entry


@dataclass
class Event:
    """Progress, for a caller that wants to stream it (the ACP server does)."""

    kind: str        # step | plan_step | text | thought | tool | verdict |
                     # halt | interjected
    step: int = 0
    text: str = ""
    call: Optional[ToolCall] = None
    result: Optional[ToolResult] = None
    verdict: Optional[Verdict] = None
    halt: Optional[Halt] = None


@dataclass
class Outcome:
    #: Talos is bounded by Ariadne's step ceiling, never by wall clock.
    budget_kind = "steps"

    halt: Halt
    steps_used: int
    #: Where `succeeded` came from: `verifier` when a real verdict decided it,
    #: `unverified` when the run used `accept_everything` and DONE therefore
    #: means only that the engine stopped calling tools. The distinction
    #: `CaseResult.claim_source` keeps visible, so a control arm's `honest` is
    #: never read beside a verified arm's as though they measured the same thing.
    claim_source: str = "verifier"
    #: Tool calls issued across the run, so `steps_used` can be read against the
    #: work done rather than as though a step were a fixed unit.
    tools_used: int = 0
    changed: List[Path] = field(default_factory=list)
    verdict: Optional[Verdict] = None
    summary: str = ""
    dry_run: bool = False
    #: What this run cost, when the provider reported it. Covers only the
    #: executor's own turns -- Metis plans on the same engine but before Talos is
    #: entered, so a caller wanting the whole session's spend should read
    #: `engine.total_usage` instead of adding these up.
    usage: Usage = field(default_factory=Usage)

    @property
    def succeeded(self) -> bool:
        return self.halt is Halt.DONE


#: Tools that change something outside the conversation. Everything else --
#: reading, listing, searching -- is safe to run unasked, and prompting for it
#: would train the user to click yes without reading.
CONSEQUENTIAL = frozenset({"write_file", "edit_file", "run_command",
                           # Touches many files at once, so if anything here
                           # needs asking about, this does.
                           "rename_symbol"})

#: Floor on a plan step's allowance. Below two, a step cannot both act and be
#: told the result, so it could never repair anything it broke.
MIN_STEP_BUDGET = 3

#: Floor on the transcript's share of the context window. A budget below this is
#: not a budget, it is a guarantee of thrashing -- so the floor is applied and
#: the real problem (too small a window, or too large a tool schema) is logged
#: rather than hidden behind endless compaction.
MIN_TRANSCRIPT_BUDGET = 1_000

#: Fed back to the engine when it returns nothing, so the next step has some
#: chance of going differently.
EMPTY_REPLY_NOTE = (
    "\n## Note\nYour last reply was empty. This usually means the response was "
    "truncated before any content was produced -- typically a reasoning model "
    "spending its entire token budget before answering. Reply with a tool call "
    "or a short statement of what you intend to do.")

#: Shown to the user. A blank turn otherwise looks like the agent is thinking.
EMPTY_REPLY_TEXT = (
    "The engine returned an empty reply (likely truncated before it produced "
    "any content). Retrying.")

#: Fed back when the engine asks to be verified without having done anything.
#: Verification of an empty change set succeeds vacuously, so without this the
#: engine can end a run simply by declining to act.
#: Fed back when a step repeats the one before it exactly and changes nothing.
#: The halting policy will stop this on its own, but stopping is the expensive
#: outcome: the engine has budget left at this point and only needs to notice.
REPEATED_CALL_NOTE = (
    "\n## Note\nThat was the same tool call, with the same arguments, as one "
    "you made a moment ago, and it changed nothing. Cycling back to it will "
    "not produce a different result. Read the error above and do something "
    "different: check the file's actual contents, try a different approach, or "
    "state plainly what is blocking you.")

NOTHING_DONE_NOTE = (
    "\n## Verification\nNothing has been changed or run this turn, so there is "
    "nothing to verify and the task cannot be considered done. Reading a file "
    "tells you what to do; it is not doing it. Either carry out the task with "
    "write_file, edit_file, rename_symbol or run_command, or state plainly what "
    "is blocking you.")


class Talos:
    def __init__(self, engine, workspace: Workspace,
                 tools: Optional[ToolRegistry] = None,
                 ariadne: Optional[Ariadne] = None,
                 verifier: Optional[Verifier] = None,
                 constitution: str = "",
                 lethe: Optional[Lethe] = None,
                 ask_permission: Optional[Callable[[ToolCall], bool]] = None,
                 interim: Optional[Verifier] = None,
                 replan: Optional["Replanner"] = None,
                 max_replans: int = 1,
                 delegation: bool = False,
                 depth: int = 0) -> None:
        self.engine = engine
        self.ws = workspace
        self.tools = tools or ToolRegistry.default()
        #: How far down the delegation chain this agent is. 0 is the one the
        #: user is talking to.
        self.depth = depth
        if delegation and depth < MAX_DELEGATION_DEPTH:
            # Registered last so a workspace that already defines `delegate`
            # -- an MCP server, say -- keeps its own meaning rather than being
            # silently replaced by ours.
            self.tools = ToolRegistry.combined(
                [Delegate(self._spawn)], self.tools)
        self.ariadne = ariadne or Ariadne()
        self.verify = verifier or accept_everything
        #: Whether `Outcome.succeeded` is backed by evidence. `accept_everything`
        #: agrees with the engine by construction, so a run using it reports DONE
        #: on the model's say-so and `honest` becomes a tautology. The eval needs
        #: to know that to avoid printing such a run beside a verified one -- and
        #: the control arm depends on exactly this configuration, so the
        #: distinction has to be recorded rather than assumed away.
        self.claim_source = ("unverified" if self.verify is accept_everything
                             else "verifier")
        self.constitution = constitution
        #: Revises the *remaining* plan after a step fails. `None` keeps the
        #: original behaviour: the plan is a guess made before anything was
        #: known, and it is followed to the end regardless of what the first
        #: steps discovered.
        self.replan = replan
        #: Re-planning costs an engine turn and can thrash -- a planner handed
        #: its own failure can produce a new plan that fails the same way. One
        #: revision per run is the useful case (the plan was wrong about the
        #: shape of the problem); a second is usually the model arguing with
        #: the verifier.
        self.max_replans = max_replans
        #: Bounds the transcript. The repair loop makes this necessary rather
        #: than merely nice: every failed attempt appends a verification report,
        #: and resending all of them each step is quadratic.
        self.lethe = lethe or Lethe()
        #: What the caller asked for, kept so a per-turn shrink never becomes
        #: permanent: the budget is recomputed from this each turn rather than
        #: from whatever the last turn happened to leave behind.
        self._default_transcript_budget = self.lethe.max_tokens
        self._warned_tiny_window = False
        #: Consulted before a consequential tool call. `None` means unattended:
        #: the executor proceeds, which is the right default for a scripted run
        #: and the wrong one for an editor -- so the ACP server always sets it.
        self.ask_permission = ask_permission
        #: Run between plan steps, where the full ladder would be too expensive.
        #: Cheap and structural: "did this step leave the tree intact", not "is
        #: the task done". Falls back to the real verifier when unset, which is
        #: correct but slow -- the ACP layer passes `Oracle.quick`.
        self.interim = interim or self.verify

        #: Words from the user, delivered at the next step boundary. Hold this
        #: object before starting a run and push to it while the loop is going
        #: to steer it without ending it. See `knossos.interject`.
        self.interjections = Interjections()

        #: The rendered conversation so far. Persisted across `run` and
        #: `resume` so a user can redirect without losing context.
        self.transcript: List[str] = []
        self.changed: List[Path] = []
        #: Tool calls issued across the run. Reported on `Outcome` so the eval
        #: can read `steps_used` against the work a step contained: this loop
        #: batches adjacent parallel-safe calls into one turn, so step count
        #: alone rewards a model that happens to emit them together.
        self.tools_used = 0
        self.task: str = ""
        #: Reasoning emitted during the turn in flight. Reset per turn by
        #: `_turn`, read once the stream has been drained.
        self._turn_reasoning: List[str] = []
        #: What the current request has cost. Reset by `run` and by `resume`, so
        #: each thing the user asked for is costed separately -- the same
        #: reasoning that resets the step budget there. The engine keeps the
        #: session total.
        self.usage = Usage()

    # ------------------------------------------------------------------ api

    def run(self, task: str, context: str = "", plan: Sequence[str] = (),
            cancelled: Optional[Callable[[], bool]] = None,
            on_event: Optional[Callable[[Event], None]] = None) -> Outcome:
        """Start a fresh task, discarding any previous conversation.

        A plan of more than one step is *executed* step by step rather than
        merely quoted at the engine -- see `_drive_plan`.
        """
        self.task = task
        self.changed = []
        self.usage = Usage()
        self.transcript = [self._opening(task, plan)]
        if len(plan) > 1:
            return self._drive_plan(list(plan), context, cancelled, on_event)
        return self._drive(context, cancelled, on_event)

    # ----------------------------------------------------------- plan driving

    def _spawn(self, task: str) -> Outcome:
        """Run `task` in a child agent that shares this workspace.

        The child gets half the parent's step ceiling. Not a tunable fraction
        with a flag: the parent still has to verify and repair whatever comes
        back, and a child permitted the whole budget can leave nothing for the
        turn that has to make sense of it.

        Everything the child changes is folded into this run's change set before
        the outcome is returned, so the parent's verifier sees one change set --
        the child's edits are ordinary edits that happened to be made by
        something else.
        """
        budget = max(MIN_STEP_BUDGET, self.ariadne.max_steps // 2)
        child = Talos(
            self.engine, self.ws,
            # Delegation removed by construction. A child that can delegate
            # multiplies agents per level, and the failure mode is a bill
            # rather than a crash.
            tools=self.tools.without(DELEGATE),
            ariadne=Ariadne(max_steps=budget, target_steps=max(1, budget - 1),
                            stuck_after=self.ariadne.stuck_after),
            verifier=self.verify,
            constitution=self.constitution,
            # The same gate, by reference: a subagent must not be a route
            # around a permission prompt the user is watching.
            ask_permission=self.ask_permission,
            interim=self.interim,
            delegation=False,
            depth=self.depth + 1)

        log(f"[talos] delegating at depth {child.depth}: {task[:60]}")
        outcome = child.run(task)

        for path in outcome.changed:
            if path not in self.changed:
                self.changed.append(path)
        self.usage = self.usage + outcome.usage
        return outcome

    def _recent_transcript(self, entries: int) -> str:
        """The tail of the transcript, for a caller that needs recent context."""
        return "\n".join(str(entry) for entry in self.transcript[-entries:])

    def _revise_plan(self, steps: List[str], index: int, replans: int,
                     acted: bool,
                     emit: Callable[[Event], None]) -> Optional[List[str]]:
        """A new plan after a failed step, or None to keep the current one.

        **`acted` is the precondition that makes this worth its cost.** A step
        that wrote code and still failed verification is evidence that the
        approach was wrong, which is exactly what a plan encodes. A step that
        produced nothing -- an engine that stalled, replied empty, or thrashed
        without calling a tool -- is evidence about the *engine*, and says
        nothing whatever about whether the plan was right. Re-planning on
        silence spends an engine turn to learn nothing, and worse, it spends the
        turn that the next step needed.

        Fails closed in every other direction too. No planner, budget spent,
        nothing left to revise, a planner that raises, a planner that answers
        with something that is not a list of non-empty strings -- all of them
        leave the original plan in place, because continuing with a stale plan
        is a known quantity and continuing with a malformed one is not.

        Only the *tail* is rewritten. The steps already attempted are history:
        they changed files, and a revision that renumbers them would make the
        transcript disagree with what was done.
        """
        remaining = steps[index + 1:]
        if (self.replan is None
                or not acted
                or replans >= self.max_replans
                or not remaining):
            return None

        done = steps[:index + 1]
        # The tail of the transcript is what the engine just said about why the
        # step did not work. It is the whole reason to re-plan, so it is what
        # the planner is given.
        learned = self._recent_transcript(REPLAN_CONTEXT_ENTRIES)
        try:
            proposed = self.replan(self.task, learned, done, remaining)
        except Exception as exc:                       # noqa: BLE001 - reported
            log(f"[talos] re-planning failed, keeping the original plan: {exc}")
            return None

        clean = [str(item).strip() for item in (proposed or [])
                 if str(item).strip()]
        if not clean:
            return None
        ceiling = max(1, len(remaining) * REPLAN_GROWTH_LIMIT)
        if len(clean) > ceiling:
            log(f"[talos] revised plan had {len(clean)} steps for a tail of "
                f"{len(remaining)}; truncated to {ceiling}")
            clean = clean[:ceiling]
        if clean == remaining:
            return None                                # nothing actually changed

        emit(Event(kind="plan_revised", step=index + 1,
                   text="\n".join(f"{n}. {s}" for n, s in enumerate(clean, 1))))
        self.transcript.append(
            f"\n## Plan revised\nWhat the last step found changed the remaining "
            f"plan. The steps already attempted stand; from here the plan is:\n"
            + "\n".join(f"{n}. {s}" for n, s in enumerate(clean, index + 2)))
        return done + clean

    def _drive_plan(self, steps: List[str], context: str,
                    cancelled: Optional[Callable[[], bool]],
                    on_event: Optional[Callable[[Event], None]]) -> Outcome:
        """Run each plan step under its own budget, checking as it goes.

        Two things here that a flat loop cannot do.

        **A step cannot spend the whole budget.** Each gets `max_steps //
        len(plan)`, floored so a long plan still gives every step a real
        allowance. One step thrashing can no longer starve the rest -- which is
        the usual way a plan-shaped task fails.

        **Breakage is caught where it happened.** Between steps the *interim*
        check runs -- tier 0 only, in-process, reading through the workspace so
        staged content counts. A step that leaves the tree unparseable is
        reported inside its own step, while the engine still has the context
        that produced it, instead of surfacing five steps later as a mystery.
        The full ladder is reserved for the last step, because running pytest
        after every step would cost more than the plan saves.

        A failed step does not abort the plan: a plan is a guess, and the
        verifier at the end is what decides. But every failure is recorded, and
        the final verdict is the real one.
        """
        emit = on_event or (lambda _event: None)
        alive = cancelled or (lambda: False)

        # Held back for the closing phase, so proving the task is done is never
        # squeezed out by the plan itself.
        reserve = max(MIN_STEP_BUDGET, self.ariadne.max_steps // 4)
        available = max(MIN_STEP_BUDGET, self.ariadne.max_steps - reserve)
        per_step = max(MIN_STEP_BUDGET, available // max(1, len(steps)))
        used, failures, worked = 0, 0, False
        replans = 0

        # A `while` rather than `for ... in steps`, because `steps` can now be
        # replaced underneath the walk. The index is the position of the step
        # about to run, so a revision only ever rewrites the tail.
        index = 0
        while index < len(steps):
            step = steps[index]
            if alive():
                return self._finish(Halt.STUCK, used, None, "cancelled by the caller", emit)

            emit(Event(kind="plan_step", step=index + 1, text=step))
            self.transcript.append(
                f"\n## Step {index + 1} of {len(steps)}\n{step}\n\n"
                f"Do only this step. The remaining steps are not yours to start.")

            budget = Ariadne(max_steps=per_step,
                             target_steps=max(1, per_step - 1),
                             stuck_after=self.ariadne.stuck_after)
            # Marked before the step so a step that breaks the tree can be put
            # back. Cheap -- a position in the journal, not a copy of anything.
            mark = self.ws.checkpoint(f"step-{index + 1}")
            # Whether this step produced any evidence at all, which is what
            # decides if a failure is worth re-planning over -- see
            # `_revise_plan`.
            changed_before = len(self.changed)
            # Every step is held only to "did this leave the tree intact". What
            # the *task* required is settled once, below.
            outcome = self._drive(context, cancelled, on_event, ariadne=budget,
                                  verifier=self.interim)
            used += outcome.steps_used
            # Sampled here, before `_revert_broken_step` can undo the writes and
            # `self.changed` drops them. A step whose changes were rolled back
            # for breaking the tree is not "did nothing" -- it is the strongest
            # evidence available that the approach was wrong, which is exactly
            # what re-planning is for.
            acted = len(self.changed) > changed_before

            if outcome.succeeded:
                worked = True
            else:
                failures += 1
                reverted = self._revert_broken_step(mark, outcome)
                if reverted:
                    names = ", ".join(self.ws.display(p) for p in reverted)
                    self.transcript.append(
                        f"\n## Note\nStep {index + 1} left the tree unparseable, so "
                        f"it was undone: {names} {'is' if len(reverted) == 1 else 'are'} "
                        f"back to the state before that step. Nothing after it was "
                        f"built on the broken version. Try a different approach.")
                    emit(Event(kind="text", step=index + 1,
                               text=f"Step {index + 1} broke the tree and was rolled "
                                    f"back ({len(reverted)} file(s))."))
                else:
                    self.transcript.append(
                        f"\n## Note\nStep {index + 1} ended as {outcome.halt.value} "
                        f"rather than verified. Carry what you learned into the next "
                        f"step; do not restart the plan.")

                # The plan was a guess made before anything had been read. A
                # step failing is the first real evidence about whether the
                # guess was right, and it is the only point in the run where
                # acting on that evidence is still cheap -- after the last step
                # the budget is gone.
                revised = self._revise_plan(steps, index, replans,
                                            acted=acted, emit=emit)
                if revised is not None:
                    steps = revised
                    replans += 1
                    # The tail changed length, so the remaining budget has to be
                    # redivided. Without this a revision that adds steps starves
                    # every one of them, and a revision that removes steps
                    # leaves the saved budget unspent.
                    left = max(1, len(steps) - (index + 1))
                    per_step = max(MIN_STEP_BUDGET, (available - used) // left)

            index += 1

        if replans:
            log(f"[talos] the plan was revised {replans} time(s)")
        if failures:
            log(f"[talos] {failures} of {len(steps)} plan step(s) did not verify")

        # The plan was a guess about how to get there; the verifier decides
        # whether it did. A run whose last *step* stalled but whose work is
        # complete must not report failure -- and the reserve gives the engine
        # room to repair whatever the real ladder turns up.
        if alive():
            return self._finish(Halt.STUCK, used, None, "cancelled by the caller", emit)
        self.transcript.append(
            f"\n## Plan complete\nAll {len(steps)} steps have been attempted. "
            f"Confirm the original task is done, repair anything outstanding, "
            f"then reply without a tool call so verification can run.")
        closing = Ariadne(max_steps=reserve, target_steps=max(1, reserve - 1),
                          stuck_after=self.ariadne.stuck_after)
        # Relaxed only when the plan actually accomplished something. Otherwise
        # this phase becomes a second door onto the bug the rule exists to
        # close: every step doing nothing, then a verdict over an empty change
        # set reporting success. Caught by
        # `test_an_unfinished_run_leaves_the_plan_pending`.
        final = self._drive(context, cancelled, on_event, ariadne=closing,
                            require_action=not worked)

        return Outcome(halt=final.halt, steps_used=used + final.steps_used,
                       # Not `used + final.tools_used`: the counter lives on the
                       # executor and already spans every plan step, the same
                       # reason `usage` is passed whole rather than summed.
                       tools_used=self.tools_used,
                       claim_source=self.claim_source,
                       changed=list(self.changed), verdict=final.verdict,
                       summary=final.summary, dry_run=self.ws.dry_run,
                       # `self.usage` already spans every plan step: it is reset
                       # per request, not per drive, so adding the per-step
                       # outcomes here would double-count the whole plan.
                       usage=self.usage)

    def resume(self, instruction: str, context: str = "",
               cancelled: Optional[Callable[[], bool]] = None,
               on_event: Optional[Callable[[Event], None]] = None) -> Outcome:
        """Continue the existing conversation with new instructions.

        The step budget resets: each thing the user asks for gets its own
        allowance rather than one budget draining across a whole session.
        """
        if not self.transcript:
            return self.run(instruction, context, cancelled=cancelled, on_event=on_event)
        self.usage = Usage()
        self.transcript.append(f"\n## User\n{instruction}")
        return self._drive(context, cancelled, on_event)

    def _revert_broken_step(self, mark: str, outcome: Outcome) -> List[Path]:
        """Undo a plan step that left the tree unparseable. Returns what moved.

        Narrow on purpose, in three ways, because undoing the engine's work is
        a strong action and doing it wrongly is worse than not doing it.

        **Only on a real failed verdict.** A step that ran out of budget without
        ever reaching the verifier proves nothing about the tree, so there is
        nothing to undo. The interim check is tier 0 -- syntax -- so a failure
        here means the files genuinely do not parse, not merely that the task is
        unfinished.

        **Only what that step wrote.** The mark is a journal position, so a file
        edited in an earlier step and again in this one goes back to its
        *earlier* content, not to its state at the start of the run.

        **Only on disk.** A dry run stages rather than writes, so the journal is
        empty and this is a no-op -- staging is already the undo.

        Why undo at all rather than just reporting it: the next plan step
        otherwise builds on code that does not parse, and every later failure
        becomes a consequence of this one. That is the same reasoning the
        ladder's fail-fast rule is built on, applied across steps.
        """
        if outcome.verdict is None or outcome.verdict.passed:
            return []
        try:
            restored = self.ws.rewind(mark)
        except KeyError:
            return []                       # the mark was consumed by an earlier rewind
        # A file the step *created* no longer exists, so it is no longer a
        # change. One it merely edited still differs from where the run started
        # and must stay in the set.
        gone = {p for p in restored if not p.exists()}
        if gone:
            self.changed = [p for p in self.changed if p not in gone]
        return restored

    def _prepare_verifier(self) -> None:
        """Let the verifier look at the tree before the agent changes it.

        Optional, and absent from the `Verifier` protocol's required surface, so
        a plain callable stays a valid verifier -- which is the reason the
        protocol is one method in the first place. Oracle uses it to record
        which checks were already failing; a verifier that needs no such
        snapshot simply does not define it.

        **Timed to the first consequential call, not to the start of the run.**
        Both are pristine, so both are correct, but a baseline costs a whole
        test suite and most turns never write anything -- asking a question
        should not spend thirty seconds proving the repository was already
        green. Implementations must be idempotent, because this is reached once
        per consequential call rather than once per run.
        """
        prepare = getattr(self.verify, "prepare", None)
        if not callable(prepare):
            return
        try:
            prepare(self.ws)
        except Exception as exc:
            # A verifier that cannot take a baseline still verifies; it just
            # cannot tell pre-existing failures apart from new ones. That is a
            # worse verdict, not a reason to abandon the run.
            log(f"[talos] verifier baseline failed, continuing without one: {exc}")

    # --------------------------------------------------------------- prompt

    def _opening(self, task: str, plan: Sequence[str]) -> str:
        parts = [f"## Task\n{task}"]
        if plan:
            steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan))
            parts.append(f"\n## Plan\n{steps}")
        return "\n".join(parts)

    def _system(self) -> str:
        """The standing instructions: role, constitution, tools.

        Byte-identical on every step of a run, which is what makes it worth
        keeping separate. Sent as a real system message it is a stable prefix a
        provider's cache can hit; concatenated onto a growing transcript it is
        just more input to pay for, every turn.

        # Why there are no explicit cache breakpoints

        Anthropic's API wants `cache_control` markers naming what to cache. This
        engine speaks OpenAI's chat-completions schema, where prefix caching is
        *automatic* -- OpenAI and DeepSeek match the longest common prefix with
        no annotation, and Groq does not cache at all. Sending `cache_control`
        into that schema is not a no-op: providers that validate unknown fields
        reject the request, so the marker would buy nothing and cost runs.

        So the whole of cache control here is a property rather than an API, and
        it is two things, both of which are now tested rather than assumed:
        this string not drifting between steps
        (`test_the_system_message_is_stable_across_steps`), and the message array
        growing by append only, so no earlier message is rewritten and
        invalidates everything after it
        (`test_the_whole_prefix_grows_by_append_only`). Compaction breaks the
        second deliberately and unavoidably, which is a reason to compact as
        late as possible, not a reason to annotate.

        Whether any of it works is now measurable rather than asserted:
        `Usage.cached` reports what the provider actually served from cache.
        """
        parts = [EXECUTOR_ROLE]
        if self.constitution:
            parts.append(f"\n# Constitution\n\nThese apply to everything you do.\n\n"
                         f"{self.constitution}")
        parts.append(f"\n{self.tools.render()}")
        return "\n".join(parts)

    def _prompt(self) -> str:
        preamble = self._system()
        # Sized here, where the header is known: the transcript's share of the
        # window is whatever the header and the reply do not take.
        self._size_transcript_budget(preamble)
        # Compaction happens here rather than after each append, so the
        # transcript is bounded exactly where it is about to be spent.
        self.compact()
        return preamble + "\n\n" + "\n".join(self.transcript)

    #: Transcript entries beginning with this are the engine's own words. Talos
    #: writes every entry itself, so this is reading back its own formatting
    #: rather than parsing anything a model produced.
    _ASSISTANT_HEADING = "## Assistant"

    def _history(self) -> List[dict]:
        """The conversation as a role-tagged message array.

        Flattened, a run is one enormous user message in which the model's own
        prior replies appear as `## Assistant` headings -- so the model is being
        asked to infer the structure of its own conversation from markdown. That
        is not the shape any instruction-tuned model was trained on, and it gets
        worse the longer the run goes.

        The mapping is deliberately crude, because it does not need to be
        clever: everything Talos appends is either the engine's reply or
        something being told *to* the engine. Tool results, verification
        reports, budget pressure and notes are all the latter, and all become
        user turns.

        **Tool calls travel natively when they arrived natively.** A turn whose
        calls all carry a provider id is replayed as an `assistant` message with
        a `tool_calls` array and one `tool` message per result, which is the
        shape the provider sent and the shape its model was trained on. A turn
        whose calls have no ids came from a model writing the fenced convention
        by hand, and is replayed as the text it was -- reflecting it back as
        native calls would show a model a format it never produced.

        That test is per turn rather than per run on purpose. A run can contain
        both: the same model may answer natively on one step and fall back to a
        fence on the next, and each step should go back the way it came.

        **A call and its results go natively only together.** They are appended
        adjacently, so at append time the pair always holds -- but compaction
        runs between then and here, and it does not preserve pairs. Lethe elides
        the largest entry, which returns a plain `str` and drops the structure
        from one half; or it summarises a middle band containing the assistant
        turn while the results survive in the kept tail. Either way one half
        loses its `Entry`-ness and the other does not.

        Emitting the surviving half natively is worse than not doing it at all:
        an assistant message carrying `tool_calls` with no matching `tool`
        messages, or a `tool` message answering nothing, is rejected outright by
        the providers this shape exists to please. So the pairing is checked
        here, against what compaction actually left, and a broken pair falls back
        to text on *both* halves.
        """
        preamble = self._system()
        self._size_transcript_budget(preamble)
        self.compact()

        entries = list(self.transcript)
        paired = self._native_pairs(entries)

        messages: List[dict] = [{"role": "system", "content": preamble}]
        for index, entry in enumerate(entries):
            if index in paired and isinstance(entry, Entry):
                messages.extend(self._native_messages(entry))
                continue
            # Either a plain string (a note, or a region Lethe has summarised
            # and so flattened), a turn whose calls were written as text, or a
            # half-pair whose other half compaction took.
            stripped = entry.lstrip()
            if stripped.startswith(self._ASSISTANT_HEADING):
                body = stripped[len(self._ASSISTANT_HEADING):].strip()
                # An empty assistant turn is not a message. Some providers
                # reject one outright, and it tells the model nothing.
                if body:
                    messages.append({"role": "assistant", "content": body})
            else:
                messages.append({"role": "user", "content": entry.strip()})
        return messages

    @staticmethod
    def _native_pairs(entries: Sequence[str]) -> "set":
        """Indices safe to send as native tool messages.

        An assistant turn qualifies only when the entry directly after it is the
        `tool` entry answering it, with the same call ids in the same order.
        Anything else -- a missing half, a reordering, a summarised gap between
        them -- disqualifies both.
        """
        safe: set = set()
        for index in range(len(entries) - 1):
            first, second = entries[index], entries[index + 1]
            if not (isinstance(first, Entry) and isinstance(second, Entry)):
                continue
            if not (first.native and second.native):
                continue
            if first.role != "assistant" or second.role != "tool":
                continue
            # Ids, not just counts: a result attributed to the wrong call is a
            # quieter failure than a missing one and a worse one.
            if [c.id for c in first.calls] != [c.id for c in second.calls]:
                continue
            if len(second.results) != len(second.calls):
                continue
            safe.add(index)
            safe.add(index + 1)
        return safe

    @staticmethod
    def _native_messages(entry: "Entry") -> List[dict]:
        """Render one natively-called turn as OpenAI-shaped messages.

        An assistant turn becomes `content` plus a `tool_calls` array; the
        matching results become one `tool` message each, keyed by the same id.
        Providers pair them by that id, so a result whose id does not match a
        call is worse than no result at all -- which is why only turns where
        *every* call has an id take this path.
        """
        if entry.role == "assistant":
            message: dict = {
                "role": "assistant",
                # May legitimately be empty when the model only called tools.
                # OpenAI accepts an empty string alongside `tool_calls`; it is
                # `None` that some providers reject.
                "content": entry.text,
                "tool_calls": [
                    {"id": call.id, "type": "function",
                     "function": {"name": call.name,
                                  "arguments": json.dumps(call.args)}}
                    for call in entry.calls],
            }
            # Only ever populated for an engine that declared it accepts this;
            # most OpenAI-compatible providers reject it outright.
            if entry.reasoning:
                message["reasoning_content"] = entry.reasoning
            return [message]

        # Results. Zipped rather than indexed so a short list from a refused or
        # failed call cannot raise inside prompt assembly.
        return [{"role": "tool", "tool_call_id": call.id, "name": call.name,
                 "content": result.content}
                for call, result in zip(entry.calls, entry.results)]

    #: Left for the engine's own reply. The header and transcript are the input
    #: side of the window; the output has to come out of the same budget.
    OUTPUT_RESERVE = 4096

    #: Slack for the difference between `estimate_tokens` and the server's real
    #: tokenizer. The estimate is chars/4, which is close for English prose and
    #: optimistic for code and JSON -- exactly what a transcript is full of. An
    #: over-estimate here costs a little context; an under-estimate costs the
    #: front of the prompt, so the asymmetry decides the direction.
    SAFETY_MARGIN = 0.9

    def _size_transcript_budget(self, preamble: str) -> None:
        """Fit Lethe's budget to the window the server will actually accept.

        Lethe's default is 24 000 tokens, chosen as a reasonable guess and
        applied regardless of what was on the other end. That is the wrong shape
        for a harness whose whole point is not making claims it cannot support:
        against a server handing out 16 384 -- measured, on an ordinary Ollama
        deployment serving models that advertise far more -- the harness
        compacted to 24 000, declared the prompt within budget, and the *server*
        then dropped the front of it. Which is the task statement.

        So the budget is derived rather than assumed, and derived per turn
        because the header is not constant: `tools.render()` grows when MCP
        tools are registered, and a constitution can be any size.

        Does nothing when the engine cannot say what its window is -- most
        hosted providers cannot -- and never *raises* the budget above Lethe's
        configured default. A larger window is not a reason to send more than
        the operator asked for; a smaller one is a reason to send less.
        """
        window = getattr(self.engine, "context_window", None)
        if not isinstance(window, int) or window <= 0:
            return

        available = window - self.OUTPUT_RESERVE - estimate_tokens(preamble)
        budget = int(available * self.SAFETY_MARGIN)
        if budget < self._default_transcript_budget:
            if budget < MIN_TRANSCRIPT_BUDGET:
                # The header alone has nearly filled the window. Compaction
                # cannot fix that -- the fix is a bigger window or fewer tools --
                # so say so once rather than silently thrashing at a budget no
                # transcript can meet.
                if not self._warned_tiny_window:
                    self._warned_tiny_window = True
                    log(f"[talos] context window {window} leaves only {budget} "
                        f"tokens for the transcript after the tool schema and "
                        f"reply reserve; raise the server's window")
                budget = MIN_TRANSCRIPT_BUDGET
            self.lethe.max_tokens = budget
        else:
            self.lethe.max_tokens = self._default_transcript_budget

    def compact(self) -> bool:
        """Bring the transcript within Lethe's budget. Returns True if it acted."""
        result = self.lethe.compact(self.transcript)
        if result.compacted:
            self.transcript = result.transcript
        return result.compacted

    # ----------------------------------------------------------------- loop

    def _drive(self, context: str,
               cancelled: Optional[Callable[[], bool]],
               on_event: Optional[Callable[[Event], None]],
               ariadne: Optional[Ariadne] = None,
               verifier: Optional[Verifier] = None,
               require_action: bool = True) -> Outcome:
        """One budgeted loop.

        `ariadne`/`verifier` override the defaults so a single plan step can run
        under its own allowance and its own check.

        `require_action` is what stops an engine passing by declining to act. It
        is relaxed for exactly one caller: the closing phase of a plan, whose
        job is to *confirm* work already done in earlier steps. Everywhere else
        a verdict over a loop that accomplished nothing is not a completion.
        """
        alive = cancelled or (lambda: False)
        emit = on_event or (lambda _event: None)
        ariadne = ariadne or self.ariadne
        verify = verifier or self.verify

        noops = 0
        last_verdict: Optional[Verdict] = None
        last_text = ""
        #: Recent tool-call signatures, for spotting a repeat. Local to this
        #: drive rather than to the instance: each plan step and each `resume`
        #: gets a fresh budget, so it should get a fresh idea of what "again"
        #: means.
        #:
        #: A window rather than a single previous value, because a loop does not
        #: have to be adjacent to be a loop -- see `FUTILE_WINDOW`.
        recent_signatures: "deque[str]" = deque(maxlen=FUTILE_WINDOW)
        #: Successful calls to tools that can change something outside the
        #: conversation, this turn. The discriminator between "verified" and
        #: "verified nothing": every tier is satisfied vacuously by an empty
        #: change set, so a passing verdict is only evidence of completion if
        #: something actually happened.
        #:
        #: Not `len(self.changed)`, because a read-only task -- running the
        #: tests, checking a build -- legitimately writes nothing and is still a
        #: real piece of work. `run_command` is consequential, so those still
        #: count.
        #:
        #: Not "any successful call" either, which is what this used to be. That
        #: let a single `read_file` satisfy the check, so an engine could answer
        #: "add a subtract() to foo.py" by reading foo.py, asking to be verified,
        #: and being told it had completed and verified the task -- over an empty
        #: change set, having written nothing. Reading is how you find out what
        #: to do; it is not doing it. Caught by
        #: `test_reading_a_file_is_not_doing_the_task`.
        acted = 0

        self._advertise_tools()

        for step in range(1, ariadne.max_steps + 1):
            if alive():
                return self._finish(Halt.STUCK, step, last_verdict,
                                    "cancelled by the caller", emit)

            nudge = ariadne.pressure(step)
            if nudge:
                self.transcript.append(f"\n## Budget\n{nudge}")

            # Drained here, at the one point in the step where the transcript is
            # a complete exchange -- see `knossos.interject` for why not sooner.
            # After the budget note, so the user's words are the last thing
            # before the request rather than buried behind the harness's own
            # prompting.
            interjected = self.interjections.drain()
            if interjected:
                self.transcript.append(
                    f"\n## User\n{Interjections.render(interjected)}")
                emit(Event(kind="interjected", step=step,
                           text="\n".join(interjected)))

            emit(Event(kind="step", step=step))

            reply = "".join(self._turn(context, alive, step, emit))
            # After the join, so the stream is fully drained and the provider's
            # usage chunk -- which arrives last -- has been read.
            self._bank_usage()
            prose, calls = parse_calls(reply)
            # Only when every call carries a provider id. A turn that mixes them
            # cannot be replayed natively without inventing an id for the half
            # that has none, so the whole turn stays text.
            native = bool(calls) and all(call.id for call in calls)
            self.transcript.append(
                Entry(f"\n## Assistant\n{reply.strip()}", role="assistant",
                      calls=calls, text=prose, native=native,
                      reasoning=self._reasoning_for(native)))
            if prose:
                last_text = prose
                emit(Event(kind="text", step=step, text=prose))

            outcome = StepOutcome()

            if calls:
                results = self._run_calls(calls, step, emit)
                outcome.tool_calls = len(calls)
                outcome.files_changed = sum(len(r.changed) for r in results)
                acted += sum(1 for c, r in zip(calls, results)
                             if not r.is_error and c.name in CONSEQUENTIAL)
                self.transcript.append(
                    Entry(self._render_results(calls, results), role="tool",
                          calls=calls, results=results, native=native))

                signature = _signature(calls)
                outcome.repeated = signature in recent_signatures
                # A step that changed something is where "again" starts over:
                # whatever the model was circling, it is no longer circling it,
                # and the reads that led up to the change must not be held
                # against the reads that follow it.
                #
                # The window is emptied *and* this step is then remembered, not
                # skipped. Dropping it would lose the plainest loop of all --
                # one successful edit followed by the same edit re-issued
                # forever, each retry changing nothing because the first one
                # already landed.
                if outcome.files_changed:
                    recent_signatures.clear()
                recent_signatures.append(signature)
                if outcome.is_futile:
                    # Say so in the transcript as well as counting it. Halting
                    # is the backstop; the cheaper outcome is the engine
                    # noticing it is going in a circle and trying something
                    # else while it still has budget left.
                    self.transcript.append(REPEATED_CALL_NOTE)
            elif not reply.strip():
                # The window is deliberately *not* cleared here. A silent turn
                # is not progress -- it is already a noop by `is_noop`, and it
                # counts toward `stuck_after` on its own. Forgetting the loop
                # because the model paused inside it is how A, silence, A reads
                # as two unrelated steps; the old single-slot version reset here
                # and did exactly that.
                # An engine that said nothing has not claimed to be finished, so
                # there is nothing to verify. Falling through to the verifier
                # here is what turns a truncated reply into a *passing* run: the
                # change set is empty, an empty change set satisfies every tier,
                # and the turn reports success having done nothing. Observed
                # against a live reasoning model, which spent its whole token
                # budget inside `<think>` and returned no content at all.
                self.transcript.append(EMPTY_REPLY_NOTE)
                emit(Event(kind="text", step=step, text=EMPTY_REPLY_TEXT))
            else:
                # The engine believes it is finished. The verifier decides.
                verdict = verify(self.ws, list(self.changed))
                last_verdict = verdict
                # A pass over a run that did nothing is not a completion. The
                # engine is entitled to *request* verification at any point;
                # it is not entitled to be told it succeeded by doing so.
                enough = acted > 0 or not require_action
                outcome.verdict_passed = verdict.passed and enough
                emit(Event(kind="verdict", step=step, verdict=verdict))
                if not verdict.passed:
                    self.transcript.append(f"\n## Verification\n{verdict.report()}")
                elif not enough:
                    self.transcript.append(NOTHING_DONE_NOTE)

            noops = 0 if outcome.made_progress else noops + 1

            halt = ariadne.assess(step, outcome, noops)
            if halt.is_terminal:
                return self._finish(halt, step, last_verdict, last_text, emit)

        # Unreachable: `assess` returns BUDGET_EXHAUSTED at max_steps. Kept so
        # the function is total rather than relying on that.
        return self._finish(Halt.BUDGET_EXHAUSTED, ariadne.max_steps,
                            last_verdict, last_text, emit)

    def _bank_usage(self) -> None:
        """Add the turn just finished to this request's running cost.

        Duck-typed, like every other optional thing on the engine slot: an engine
        that reports nothing leaves the total at zero, which `Usage.known`
        reports as *unmeasured* rather than as free.
        """
        turn = getattr(self.engine, "usage", None)
        if isinstance(turn, Usage):
            self.usage = self.usage + turn

    def _advertise_tools(self) -> None:
        """Hand the engine the real tool schema, if it can carry one.

        The prose protocol stays: it is what a model without native tool calling
        follows, and it is what says *when* to stop calling tools. This is the
        machine-readable half, and it is why argument names stop being guesses.
        """
        if hasattr(self.engine, "tool_schema"):
            try:
                self.engine.tool_schema = self.tools.openai_schema()
            except Exception as exc:                  # a custom registry may not
                log(f"[talos] could not build a tool schema: {exc}")

    def _turn(self, context: str, alive: Callable[[], bool], step: int,
              emit: Callable[[Event], None]) -> Iterable[str]:
        """Ask the engine for one reply, structured if it can take one.

        `generate_messages` is optional on the engine, so this probes for it and
        falls back. The fallback is not a lesser path in any way that matters
        for correctness -- it is the same conversation, flattened -- which is
        what keeps a bytes-to-bytes engine a valid occupant of the slot.
        """
        self._turn_reasoning = []
        structured = getattr(self.engine, "generate_messages", None)
        if not callable(structured):
            return self._stream(self._prompt(), context, alive, step, emit)

        history = self._history()
        if context.strip():
            # Retrieval belongs with the task, not bolted to the latest turn:
            # it describes the repository, which does not change mid-run.
            history.insert(1, {"role": "user",
                               "content": f"<repository_excerpts>\n{context}\n"
                                          f"</repository_excerpts>"})
        return self._forward(structured(history, alive), step, emit)

    def _stream(self, prompt: str, context: str, alive: Callable[[], bool],
                step: int, emit: Callable[[Event], None]) -> Iterable[str]:
        """Yield the reply, forwarding reasoning as it arrives.

        Reasoning is not the answer and must never be parsed for tool calls --
        but dropping it entirely leaves the user watching a blank turn during
        exactly the part of a run where the agent is deciding what to do, which
        is indistinguishable from a hang. It goes out on its own event kind so
        the caller can put it on the channel editors render collapsed.
        """
        return self._forward(self.engine.generate(prompt, context, alive), step, emit)

    def _forward(self, chunks: Iterable[Any], step: int,
                 emit: Callable[[Event], None]) -> Iterable[str]:
        """Split reasoning off onto its own event kind. Shared by both paths.

        Reasoning is also *kept*, on `_turn_reasoning`, which it was not before:
        it went out as an event and was dropped on the floor, so the harness had
        no record of the part of a run where the model decided what to do. That
        cost two things -- a truncated turn left nothing at all to look at, and
        the eval could not see reasoning length against outcome.

        Keeping it is not the same as sending it back, and this deliberately
        does not. See `Talos._reasoning_for`.
        """
        for chunk in chunks:
            if type(chunk).__name__ == "Thought":
                text = str(chunk)
                if text:
                    self._turn_reasoning.append(text)
                    emit(Event(kind="thought", step=step, text=text))
                continue
            yield str(chunk)

    def _reasoning_for(self, entry_is_native: bool) -> str:
        """The reasoning to replay with the turn just taken. Usually none.

        It is tempting to treat this like tool calls -- the provider sent it, so
        send it back -- and for an Anthropic-shaped engine with extended
        thinking that is exactly right, because a thinking block must accompany
        the tool use it produced or the chain is rejected. This engine is
        OpenAI-shaped, and there the same move is an error: DeepSeek documents
        that `reasoning_content` fed back in a later request is refused outright,
        and the o-series never exposes raw reasoning through this endpoint at
        all.

        So the default is to keep reasoning in the record and out of the
        request, and an engine that genuinely accepts it opts in by setting
        `send_reasoning`. Off by default because the failure is a 400 on every
        turn after the first, which is a broken run rather than a degraded one.
        """
        if not self._turn_reasoning:
            return ""
        if not entry_is_native or not getattr(self.engine, "send_reasoning", False):
            return ""
        return "".join(self._turn_reasoning)

    def _run_calls(self, calls: Sequence[ToolCall], step: int,
                   emit: Callable[[Event], None]) -> List[ToolResult]:
        """Run a turn's calls, overlapping the ones that may overlap.

        Counted here rather than at the emit site: this is where a turn's calls
        are known as a set, and it runs for permitted and refused calls alike --
        a call the user declined was still a call the model chose to make.

        # What runs together

        A *run* of adjacent calls that are all `parallel_safe` goes at once;
        anything else goes on its own, in order. Adjacency rather than
        "gather all the reads first" is deliberate -- reordering a turn's calls
        would change what they see. A read after a write must still observe that
        write, and a model that issued them in that order is entitled to assume
        so.

        # What deliberately does not move into the workers

        Only `tools.dispatch` runs off-thread. Three things stay here, and each
        of them would be a real defect if it did not:

        * **Events.** The ACP server turns these into JSON-RPC notifications on
          stdout, one JSON value per line, with an assertion on the framing.
          Two threads writing at once interleaves them and takes the protocol
          down -- a much worse failure than a slow turn.
        * **`self.changed`.** Appended to from a worker it would race, and it is
          the input to every verification decision.
        * **Permission.** Prompts are put to one person through one channel, so
          they are asked before anything starts and strictly in order.

        Ordering of results is by input index, never by completion, so a turn's
        transcript reads the same whether or not anything overlapped.
        """
        results: List[Optional[ToolResult]] = [None] * len(calls)

        # Asked up front, in order, before any work starts. A refusal is a
        # result, not an error: the engine sees it in the transcript and can
        # propose something else on its next step. Raising would end the run
        # over a decision the user is entitled to make.
        # "not permitted" rather than "the user said no": a refusal also covers
        # a missing or broken asker, and telling the engine a human rejected it
        # when none was consulted invites it to give up on a task that was never
        # actually declined.
        self.tools_used += len(calls)
        allowed: List[bool] = []
        for call in calls:
            permitted = self._permitted(call)
            allowed.append(permitted)
            if not permitted:
                results[len(allowed) - 1] = ToolResult(
                    f"{call.name} was not permitted", is_error=True)

        index = 0
        while index < len(calls):
            if not allowed[index]:
                # Already refused above; deliver it in its turn so the ordering
                # of what the user sees matches the ordering they were asked in.
                self._deliver(calls, results, [index], step, emit)
                index += 1
                continue
            group = self._parallel_group(calls, allowed, index)
            if len(group) > 1:
                for offset, got in zip(group, self._dispatch_many(
                        [calls[i] for i in group])):
                    results[offset] = got
            else:
                call = calls[index]
                if call.name in CONSEQUENTIAL:
                    # The last moment the tree is still as the user left it. A
                    # session that only ever reads never pays for this.
                    self._prepare_verifier()
                results[index] = self.tools.dispatch(call, self.ws)
            # Per group, not once at the end. Draining every event after the
            # whole turn would make a multi-call turn appear to happen all at
            # once in the editor, which is a step backwards from watching it
            # work -- and for the overwhelmingly common single-call turn it
            # would have delayed the only event there is. A group that really
            # did run together is the one case where reporting it together is
            # also the honest description.
            self._deliver(calls, results, group, step, emit)
            index = group[-1] + 1

        return [result for result in results if result is not None]

    def _deliver(self, calls: Sequence[ToolCall],
                 results: Sequence[Optional[ToolResult]], group: Sequence[int],
                 step: int, emit: Callable[[Event], None]) -> None:
        """Reconcile the change set and narrate, on the calling thread only.

        Both halves are here rather than in the workers deliberately.
        `self.changed` feeds every verification decision and would race; events
        become JSON-RPC notifications on a stdout that asserts its own framing,
        and two threads writing at once takes the protocol down.
        """
        for index in group:
            result = results[index]
            if result is None:
                continue
            for path in result.changed:
                if path not in self.changed:
                    self.changed.append(path)
            emit(Event(kind="tool", step=step, call=calls[index], result=result))

    def _parallel_group(self, calls: Sequence[ToolCall], allowed: Sequence[bool],
                        start: int) -> List[int]:
        """Indices of the adjacent, permitted, parallel-safe run at `start`.

        Returns `[start]` when the call there is not safe to overlap, so the
        caller has one shape to handle rather than two.
        """
        if not self.tools.parallel_safe(calls[start].name):
            return [start]
        group = []
        index = start
        while (index < len(calls) and allowed[index]
               and self.tools.parallel_safe(calls[index].name)):
            group.append(index)
            index += 1
        return group

    #: Ceiling on concurrent tool calls. Bounded because these are filesystem
    #: reads: past a handful the disk is the limit and more threads only add
    #: contention. Also bounds the damage if a tool is wrongly marked safe.
    MAX_PARALLEL = 8

    def _dispatch_many(self, calls: Sequence[ToolCall]) -> List[ToolResult]:
        """Run several calls at once, returning results in input order.

        `ThreadPoolExecutor.map` preserves input order regardless of completion
        order, which is the property the transcript depends on. `dispatch`
        already turns every failure into an error result, so a worker cannot
        raise into the pool.
        """
        with ThreadPoolExecutor(max_workers=min(len(calls), self.MAX_PARALLEL),
                                thread_name_prefix="talos-tool") as pool:
            return list(pool.map(lambda c: self.tools.dispatch(c, self.ws), calls))

    def _permitted(self, call: ToolCall) -> bool:
        """Whether this call may run.

        Only consequential calls are put to the user. Asking about every read
        would train them to approve without looking, which is worse than not
        asking at all.
        """
        if call.name not in CONSEQUENTIAL or self.ask_permission is None:
            return True
        try:
            return bool(self.ask_permission(call))
        except Exception as exc:
            # A broken asker must not become an implicit yes.
            log(f"[talos] permission check failed, refusing: {exc}")
            return False

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
        return Outcome(halt=halt, steps_used=step, tools_used=self.tools_used,
                       claim_source=self.claim_source,
                       changed=list(self.changed),
                       verdict=verdict, summary=summary, dry_run=self.ws.dry_run,
                       usage=self.usage)

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
