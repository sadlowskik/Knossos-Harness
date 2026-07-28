"""Isolation tests for Metis, the planner.

The claims, in order of how much they matter:

  1. **A planner can never end a run.** Planning happens before any file is
     touched, so every failure path -- a broken engine, an unparseable reply, a
     cancelled turn -- must still yield a usable plan rather than an exception.
  2. The designed path is a tool call; prose is a fallback, not the design.
  3. Prose parsing takes only *marked* list items. Treating every line as a step
     turns a paragraph of hedging into an eight-step plan.

    pytest -q tests/test_metis.py
"""
import json

import pytest

from knossos.metis import MAX_STEPS, Metis, Plan, worth_planning
from knossos.tools import ToolRegistry


class ScriptedEngine:
    name = "scripted"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt, context, cancelled):
        self.prompts.append(prompt)
        yield self.replies.pop(0) if self.replies else ""


class BrokenEngine:
    name = "broken"

    def generate(self, prompt, context, cancelled):
        raise RuntimeError("the provider is down")
        yield  # pragma: no cover - unreachable, keeps this a generator


def submit(*steps):
    return ("```json\n"
            + json.dumps({"tool": "submit_plan", "args": {"steps": list(steps)}})
            + "\n```")


# ------------------------------------------------------------ the designed path

def test_a_submitted_plan_is_used():
    metis = Metis(ScriptedEngine(submit("read lib.py", "add the flag")))

    plan = metis.plan("add a --verbose flag to the CLI entry point")

    assert plan.steps == ["read lib.py", "add the flag"]
    assert not plan.degenerate
    assert plan.render().startswith("1. read lib.py")


def test_a_native_tool_call_arrives_as_a_fenced_block():
    """The engine normalises `tool_calls` into fences, so both formats land here
    identically -- Metis needs no second parser."""
    metis = Metis(ScriptedEngine("prose first\n" + submit("one", "two")))

    assert metis.plan("do a reasonably sized thing here").steps == ["one", "two"]


def test_a_plan_call_for_another_tool_is_ignored():
    reply = ("```json\n" + json.dumps(
        {"tool": "write_file", "args": {"path": "a.py", "content": "x"}}) + "\n```")
    metis = Metis(ScriptedEngine(reply))

    plan = metis.plan("do a reasonably sized thing here")

    assert plan.degenerate, "a write_file call is not a plan"


# ------------------------------------------------------- truncated json

def test_a_plan_missing_its_closing_brace_is_recovered():
    """Observed with gemma4:e4b: a complete, correct plan minus one `}`.

    `parse_calls` refuses it, which is right for a tool call and wrong for a
    plan -- a plan is inert text the executor may ignore, so failing closed
    costs a whole turn and buys nothing.
    """
    reply = ('```json\n'
             '{"tool": "submit_plan", "args": {"steps": ["read it", "fix it"]}\n'
             '```')
    assert Metis(ScriptedEngine(reply)).plan("a task long enough").steps == [
        "read it", "fix it"]


def test_a_plan_truncated_mid_array_is_recovered():
    reply = '```json\n{"tool": "submit_plan", "args": {"steps": ["read it"'
    assert Metis(ScriptedEngine(reply)).plan("a task long enough").steps == ["read it"]


def test_a_brace_inside_a_string_is_not_counted():
    """Otherwise the repair appends closers for brackets that were never open."""
    reply = ('```json\n'
             '{"tool": "submit_plan", "args": {"steps": ["handle the { case"]}\n'
             '```')
    assert Metis(ScriptedEngine(reply)).plan("a task long enough").steps == [
        "handle the { case"]


def test_genuinely_broken_json_is_not_guessed_at():
    """Only unclosed brackets are appended; nothing in the content is altered."""
    reply = '```json\n{"tool": "submit_plan", "args": {"steps": [oh dear]}}\n```'

    assert Metis(ScriptedEngine(reply)).plan("a task long enough").degenerate


def test_a_truncated_block_for_another_tool_is_ignored():
    reply = '```json\n{"tool": "write_file", "args": {"path": "a.py"'

    assert Metis(ScriptedEngine(reply)).plan("a task long enough").degenerate


# ------------------------------------------------------------- advertising

class SchemaEngine(ScriptedEngine):
    tool_schema = None

    def generate(self, prompt, context, cancelled):
        self.seen_schema = self.tool_schema
        yield from super().generate(prompt, context, cancelled)


def test_submit_plan_is_offered_as_a_real_tool():
    """A model that prefers native tool calls will make one regardless of the
    prompt; given no tools it invents a shape and the plan is unparseable."""
    engine = SchemaEngine(submit("x"))
    Metis(engine).plan("a task long enough to plan")

    names = [t["function"]["name"] for t in engine.seen_schema]
    assert names == ["submit_plan"]


def test_the_planner_is_not_given_the_executors_tools():
    """Otherwise it can start editing files during the turn whose whole purpose
    is to decide what to do before anything is touched."""
    engine = SchemaEngine(submit("x"))
    Metis(engine, tools=ToolRegistry.default()).plan("a task long enough to plan")

    names = [t["function"]["name"] for t in engine.seen_schema]
    assert "write_file" not in names


def test_the_plan_tool_is_cleared_afterwards():
    """Leaving it on the engine would let the executor call `submit_plan`."""
    engine = SchemaEngine(submit("x"))
    Metis(engine).plan("a task long enough to plan")

    assert engine.tool_schema is None


# ------------------------------------------------------------------- fallbacks

@pytest.mark.parametrize("reply,expected", [
    ("1. alpha\n2) beta\n- gamma\n* delta", ["alpha", "beta", "gamma", "delta"]),
    ("Here is the plan:\n1. Read it\n2. Edit it", ["Read it", "Edit it"]),
])
def test_prose_is_parsed_when_no_tool_call_arrives(reply, expected):
    metis = Metis(ScriptedEngine(reply))

    assert metis.plan("a task long enough to be worth planning").steps == expected


def test_unmarked_lines_are_not_steps():
    """Otherwise a paragraph of hedging becomes a plan."""
    metis = Metis(ScriptedEngine(
        "I am not sure what you want.\nThere are several options.\nLet me think."))

    plan = metis.plan("a task long enough to be worth planning")

    assert plan.degenerate


def test_an_unparseable_reply_still_yields_a_usable_plan():
    metis = Metis(ScriptedEngine("no idea"))

    plan = metis.plan("add   a   flag")

    assert plan.steps == ["add a flag"], "the task itself becomes the step"
    assert plan.degenerate
    assert plan


def test_a_broken_engine_does_not_raise():
    """Planning runs before anything is touched; it must not end the run."""
    plan = Metis(BrokenEngine()).plan("a task long enough to be worth planning")

    assert plan.degenerate
    assert len(plan) == 1


def test_a_cancelled_turn_stops_planning():
    metis = Metis(ScriptedEngine(submit("one", "two")))

    plan = metis.plan("a task long enough", cancelled=lambda: True)

    assert plan.degenerate


# ----------------------------------------------------------------------- tidying

def test_blank_and_duplicate_steps_are_dropped():
    metis = Metis(ScriptedEngine(submit("read it", "  ", "read it", "READ IT", "fix it")))

    assert metis.plan("a task long enough to plan").steps == ["read it", "fix it"]


def test_a_runaway_plan_is_capped_not_refused():
    metis = Metis(ScriptedEngine(submit(*[f"step {i}" for i in range(50)])))

    plan = metis.plan("a task long enough to plan")

    assert len(plan) == MAX_STEPS


def test_whitespace_inside_a_step_is_normalised():
    metis = Metis(ScriptedEngine(submit("read    the\n   file")))

    assert metis.plan("a task long enough to plan").steps == ["read the file"]


# ------------------------------------------------------------------ the prompt

def test_the_prompt_carries_the_constitution_and_the_tool_names():
    engine = ScriptedEngine(submit("x"))
    Metis(engine, tools=ToolRegistry.default(),
          constitution="CONSTITUTION MARKER").plan("a task long enough to plan")

    prompt = engine.prompts[0]
    assert "CONSTITUTION MARKER" in prompt
    assert "Metis" in prompt
    assert "write_file" in prompt, "a plan written blind to the tools cannot be run"


def test_the_planner_is_told_not_to_schedule_verification():
    """Verification runs automatically; a plan that ends in "run the tests"
    spends a step on something the harness does anyway."""
    engine = ScriptedEngine(submit("x"))
    Metis(engine).plan("a task long enough to plan")

    assert "verify" in engine.prompts[0].lower()


# --------------------------------------------------------------- worth planning

@pytest.mark.parametrize("task,expected", [
    ("fix typo", False),
    ("add a file", False),
    ("add a --verbose flag to the CLI and document it", True),
])
def test_trivial_tasks_skip_the_planning_turn(task, expected):
    assert worth_planning(task) is expected


def test_an_empty_plan_is_falsey_but_renders():
    assert not Plan()
    assert Plan().render() == ""
