"""Metis: planning, as an artifact separate from its execution.

That separation is the whole point of splitting this from Talos. A plan you can
read, log, and disagree with *before* any file is touched is worth more than an
intention buried in the model's first turn -- and once it exists as a list, the
editor can render it as a checklist and the executor can be held to it.

The plan is requested as a tool call rather than parsed out of prose, so its
shape is constrained by a schema the engine already knows how to emit. Local
models being what they are, a prose fallback exists; it is a fallback, not the
design. And if both fail, the task itself becomes a one-step plan, because a
planner that can return *nothing* would make planning a new way for a run to
fail before it starts.

Planning costs one engine turn. `Ariadne`'s budget is for execution, so a task
small enough not to need a plan should skip it -- see `worth_planning`.

Ported from `knossos-rs/src/metis.rs`, which is the reference implementation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .jsonrpc import log
from .tools import ToolRegistry, parse_calls

__all__ = ["Plan", "Metis", "PLANNER_ROLE", "MAX_STEPS", "worth_planning"]

#: A plan longer than this is a symptom -- either the task wants splitting, or
#: the model is narrating rather than planning. Truncated rather than refused,
#: because a too-long plan is still more use than none.
MAX_STEPS = 8

PLANNER_ROLE = """\
You are Metis, the planner. You do not edit files and you do not run commands.
You produce the shortest ordered list of concrete actions that completes the
task, and nothing else.

Rules:
- Each step is one concrete action on this codebase, phrased so someone else
  could carry it out without asking you what you meant.
- Prefer fewer steps. A one-step plan is correct for a one-step task.
- Do not include "verify" or "run the tests" as a final step: verification runs
  automatically after execution and is not yours to schedule.
- Do not restate the task as step 1.
"""

PLAN_TOOL = {
    "type": "object",
    "properties": {
        "steps": {"type": "array", "items": {"type": "string"},
                  "description": "Ordered, concrete steps"},
    },
    "required": ["steps"],
}

#: Bullets and both numbering styles: "- x", "* x", "1. x", "2) x".
_BULLET = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.*\S)\s*$")

#: A fenced block's body. Matches an *unterminated* fence too, which is the
#: whole point -- a truncated reply may never close it.
_FENCE_BODY = re.compile(r"```(?:json)?\s*\n(.*?)(?:```|\Z)", re.DOTALL)

#: Tasks below this word count rarely gain from a plan, and planning them costs
#: an engine turn that execution could have had.
_TRIVIAL_WORDS = 6


def worth_planning(task: str) -> bool:
    """Whether a task is big enough that a plan earns its turn.

    Deliberately crude. The cost of planning a trivial task is one wasted turn;
    the cost of *not* planning a large one is a flat loop with no structure. So
    this errs toward planning, and only skips what is obviously atomic.
    """
    words = task.split()
    if len(words) < _TRIVIAL_WORDS:
        return False
    # A task naming several things, or several sentences, wants a plan.
    return True


@dataclass
class Plan:
    """An ordered list of concrete steps. Possibly empty; never None."""

    steps: List[str] = field(default_factory=list)
    #: True when no plan could be extracted and the task became its own step.
    degenerate: bool = False

    def __bool__(self) -> bool:
        return bool(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def render(self) -> str:
        return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(self.steps))


class Metis:
    """Turns a task into a plan, using one engine turn.

    Holds no state between calls: a plan is a function of the task and the
    context it was given, and caching one across tasks is how a stale plan gets
    executed against the wrong request.
    """

    def __init__(self, engine, tools: Optional[ToolRegistry] = None,
                 constitution: str = "", max_steps: int = MAX_STEPS) -> None:
        self.engine = engine
        self.tools = tools
        self.constitution = constitution
        self.max_steps = max_steps

    # ------------------------------------------------------------------ api

    def plan(self, task: str, context: str = "",
             cancelled: Optional[Callable[[], bool]] = None) -> Plan:
        """Produce a plan for `task`. Always returns a usable one."""
        alive = cancelled or (lambda: False)
        try:
            reply = "".join(self._stream(self._prompt(task), context, alive))
        except Exception as exc:                     # a planner must not end a run
            log(f"[metis] planning failed, falling back to the task itself: {exc}")
            return self._degenerate(task)

        if alive():
            return self._degenerate(task)

        steps = (self._from_tool_call(reply)
                 or self._from_truncated_json(reply)
                 or self._from_prose(reply))
        if not steps:
            log("[metis] no plan could be extracted from the reply")
            return self._degenerate(task)
        return Plan(steps=self._tidy(steps))

    # -------------------------------------------------------------- prompt

    def _prompt(self, task: str) -> str:
        parts = [PLANNER_ROLE]
        if self.constitution:
            parts.append(f"\n# Constitution\n\nThese apply to everything you do.\n\n"
                         f"{self.constitution}")
        if self.tools is not None:
            # The planner does not call these, but a plan written without
            # knowing what the executor can do is a plan it cannot carry out.
            parts.append("\n# The executor has these tools\n\n"
                         + ", ".join(self.tools.names))
        parts.append(
            "\n# Submitting the plan\n\n"
            "Emit exactly one fenced JSON block and no other JSON:\n\n"
            '```json\n{"tool": "submit_plan", "args": {"steps": ["...", "..."]}}\n```\n'
            f"\nBetween 1 and {self.max_steps} steps.")
        parts.append(f"\n# Task\n\n{task}")
        return "\n".join(parts)

    def _advertise(self) -> None:
        """Offer `submit_plan` as a real tool, not only as prose.

        A model that prefers native tool calls will make one whatever the prompt
        says -- and if the only tool it has been *given* is nothing, it invents
        a shape and the plan is unparseable. Observed with gemma4:e4b, which
        called tools happily all through execution and produced no usable plan
        at all, so stepwise execution never engaged and the whole planning path
        silently did nothing.

        Only `submit_plan` is offered. Handing the planner the executor's tools
        would invite it to start editing files during the turn whose entire
        purpose is to decide what to do before anything is touched.
        """
        if not hasattr(self.engine, "tool_schema"):
            return
        self.engine.tool_schema = [{
            "type": "function",
            "function": {
                "name": "submit_plan",
                "description": (f"Submit the ordered steps for this task. Between "
                                f"1 and {self.max_steps} steps, each a concrete "
                                f"action on the codebase."),
                "parameters": PLAN_TOOL,
            },
        }]

    def _stream(self, prompt: str, context: str, alive):
        self._advertise()
        try:
            for chunk in self.engine.generate(prompt, context, alive):
                if type(chunk).__name__ == "Thought":
                    continue                         # reasoning is not the plan
                yield str(chunk)
        finally:
            # Talos re-advertises its own tools before executing, but a planner
            # that leaves `submit_plan` on the engine would let the executor
            # call it -- so clear rather than rely on that ordering.
            if hasattr(self.engine, "tool_schema"):
                self.engine.tool_schema = None

    # ------------------------------------------------------------ extraction

    @staticmethod
    def _from_tool_call(reply: str) -> List[str]:
        """The designed path: a `submit_plan` call, native or fenced.

        The engine normalises native `tool_calls` into fenced blocks, so both
        wire formats arrive here identically.
        """
        _prose, calls = parse_calls(reply)
        for call in calls:
            if call.name != "submit_plan":
                continue
            steps = call.args.get("steps")
            if isinstance(steps, list):
                return [str(s) for s in steps]
        return []

    @staticmethod
    def _from_truncated_json(reply: str) -> List[str]:
        """Recover a plan whose JSON lost its closing brackets.

        Small models drop the last `}` remarkably often. `parse_calls` refuses
        such a block, and for a *tool call* that is exactly right -- guessing at
        a half-parsed action is how the wrong file gets written. A plan is not
        an action: it is inert text that the executor may ignore, so the same
        caution costs a whole planning turn and buys nothing.

        Narrow on purpose. Only unclosed brackets are appended, in the order
        that closes them; nothing in the content is altered, and a block that
        still will not parse is abandoned. Observed with gemma4:e4b, which
        emitted a complete and correct plan missing one closing brace.
        """
        for match in _FENCE_BODY.finditer(reply):
            body = match.group(1).strip()
            if "submit_plan" not in body:
                continue
            stack: List[str] = []
            in_string = escaped = False
            for char in body:
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char in "{[":
                    stack.append("}" if char == "{" else "]")
                elif char in "}]" and stack:
                    stack.pop()
            if in_string or not stack:
                continue                             # not a truncation we can fix
            try:
                payload = json.loads(body + "".join(reversed(stack)))
            except json.JSONDecodeError:
                continue
            steps = (payload.get("args") or {}).get("steps")
            if isinstance(steps, list) and steps:
                log("[metis] recovered a plan from truncated JSON")
                return [str(s) for s in steps]
        return []

    @staticmethod
    def _from_prose(reply: str) -> List[str]:
        """The fallback: numbered or bulleted lines.

        Only lines that are *marked* as list items count. Treating every line as
        a step turns a paragraph of hedging into an eight-step plan.
        """
        out: List[str] = []
        for line in reply.splitlines():
            match = _BULLET.match(line)
            if match:
                out.append(match.group(1))
        return out

    def _tidy(self, steps: Sequence[str]) -> List[str]:
        """Strip, drop blanks and duplicates, and cap the length."""
        seen, out = set(), []
        for step in steps:
            text = " ".join(str(step).split())
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            out.append(text)
            if len(out) >= self.max_steps:
                break
        return out

    @staticmethod
    def _degenerate(task: str) -> Plan:
        """The task as its own single step.

        A plan that is only the task restates nothing useful, but it keeps every
        downstream caller on one code path -- `Talos.run` takes a plan, the ACP
        layer renders one, and neither has to special-case an absent plan.
        """
        return Plan(steps=[" ".join(task.split())], degenerate=True)
