"""Isolation tests for the executor loop.

Driven by a scripted engine, so there is no model and no network -- but the
workspace, the tools and the halting policy are all real, and files genuinely
get written.

The claims that matter:

  1. **Completion is decided by the verifier, not the engine.** An engine that
     stops calling tools is requesting verification, not announcing success.
  2. The loop always terminates, even against an engine that never stops.
  3. A failing verification feeds back into the conversation rather than ending
     the run.
  4. The jail holds when the calls come through the loop rather than directly.
  5. Consequential calls go through the permission gate, and every way of
     failing to get an answer means no.

    pytest -q tests/test_talos.py
"""
from pathlib import Path
from typing import List, Sequence

import pytest

from knossos.ariadne import Ariadne, Halt
from knossos.lethe import Lethe, estimate_tokens
from knossos.talos import (DELEGATE, FUTILE_WINDOW, MAX_DELEGATION_DEPTH,
                           MIN_TRANSCRIPT_BUDGET, Event, Talos, Verdict)
from knossos.tools import (AskUser, ReadFile, ToolRegistry, ToolResult,
                           WriteFile)
from knossos.workspace import Workspace


class ScriptedEngine:
    """Replays a fixed list of replies, one per call."""

    name = "scripted"

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)
        self.prompts: List[str] = []

    def generate(self, prompt, context, cancelled):
        self.prompts.append(prompt)
        if not self.replies:
            yield "I have nothing left to say."
            return
        yield self.replies.pop(0)


def call(tool: str, **args) -> str:
    import json
    return f'```json\n{json.dumps({"tool": tool, "args": args})}\n```'


def always_fails(ws, changed):
    return Verdict(False, "the check never passes", "detail about why")


def passes_when_anything_changed(ws, changed):
    return Verdict(bool(changed), "changed something" if changed else "nothing changed")


@pytest.fixture()
def ws(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.py").write_text("value = 1\n", encoding="utf-8")
    return Workspace(tmp_path)


# ------------------------------------------------------- the load-bearing rule

def test_the_engine_saying_done_does_not_end_the_run(ws):
    """The whole point: only the verifier can produce DONE."""
    engine = ScriptedEngine(["All finished!", "Really, finished.", "Truly done."])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=3, target_steps=2),
                  verifier=always_fails)

    outcome = talos.run("do something")

    assert outcome.halt is not Halt.DONE
    assert outcome.verdict is not None and not outcome.verdict.passed


def test_a_passing_verdict_ends_the_run(ws):
    engine = ScriptedEngine([
        call("write_file", path="src/new.py", content="x = 1\n"),
        "Done.",
    ])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)

    outcome = talos.run("add a file")

    assert outcome.halt is Halt.DONE
    assert outcome.succeeded
    assert outcome.steps_used == 2
    assert (ws.root / "src" / "new.py").read_text(encoding="utf-8") == "x = 1\n"


def test_a_failed_verdict_is_fed_back_into_the_conversation(ws):
    engine = ScriptedEngine(["Done.", "Done again.", "Still done."])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=3, target_steps=2),
                  verifier=always_fails)

    talos.run("do something")

    joined = "\n".join(talos.transcript)
    assert "Verification FAILED" in joined
    assert "detail about why" in joined
    # And the engine actually saw it on a later turn.
    assert "Verification FAILED" in engine.prompts[-1]


# ------------------------------------------------------------------ stopping

def test_the_loop_terminates_against_an_engine_that_never_stops(ws):
    engine = ScriptedEngine([call("read_file", path="src/lib.py")] * 10)
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=3, target_steps=2))

    outcome = talos.run("read forever")

    assert outcome.halt is Halt.BUDGET_EXHAUSTED
    assert outcome.steps_used == 3
    assert "budget" in outcome.summary.lower()


def test_an_engine_repeating_one_failing_call_is_stuck(ws):
    """`is_noop` cannot see this: the engine *is* calling a tool every step.

    An edit whose `old_string` does not match fails identically forever. Before
    `is_futile`, twelve of them cost twelve steps and the run ended as
    BUDGET_EXHAUSTED, having learned nothing after the first.
    """
    doomed = call("edit_file", path="src/lib.py",
                  old_string="not in the file", new_string="x")
    engine = ScriptedEngine([doomed] * 12)
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=12, target_steps=6),
                  verifier=always_fails)

    outcome = talos.run("edit something")

    assert outcome.halt is Halt.STUCK
    assert outcome.steps_used == 3, "one attempt, then two repeats"
    assert outcome.changed == []


def test_a_repeating_engine_is_told_it_is_repeating(ws):
    """Halting is the backstop; noticing while there is budget left is cheaper."""
    doomed = call("edit_file", path="src/lib.py",
                  old_string="not in the file", new_string="x")
    engine = ScriptedEngine([doomed] * 4)
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=6, target_steps=4),
                  verifier=always_fails)

    talos.run("edit something")

    assert "same tool call" in "\n".join(talos.transcript)


def test_repeating_a_call_that_still_changes_something_is_not_futile(ws):
    """Repetition alone is not failure -- only repetition that achieved nothing.

    Writing the same path twice is a real edit each time, and a run doing it
    must not be cut short for lack of imagination.
    """
    writing = call("write_file", path="out.py", content="x = 1\n")
    engine = ScriptedEngine([writing, writing, writing, "Done."])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=6, target_steps=5),
                  verifier=passes_when_anything_changed)

    outcome = talos.run("write it repeatedly")

    assert outcome.halt is Halt.DONE
    assert (ws.root / "out.py").exists()


def test_reading_the_same_file_then_acting_is_not_stuck(ws):
    """Two reads then an edit is ordinary work, not a loop."""
    reading = call("read_file", path="src/lib.py")
    engine = ScriptedEngine([
        reading, reading,
        call("write_file", path="src/new.py", content="x = 1\n"),
        "Done.",
    ])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=6, target_steps=5),
                  verifier=passes_when_anything_changed)

    outcome = talos.run("look twice then write")

    assert outcome.halt is Halt.DONE, outcome.summary


def test_an_engine_alternating_between_two_failing_calls_is_stuck(ws):
    """The loop the adjacent-only check could not see.

    A, B, A, B never produces two consecutive identical signatures, so
    `repeated` stayed False on every step, `is_futile` never fired, and the run
    spent its entire ceiling alternating between two calls that each changed
    nothing. Ariadne exists to prevent exactly this, and it arrived through the
    one door the check did not cover.
    """
    a = call("edit_file", path="src/lib.py", old_string="absent", new_string="x")
    b = call("edit_file", path="src/lib.py", old_string="missing", new_string="y")
    engine = ScriptedEngine([a, b] * 10)
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=20, target_steps=6),
                  verifier=always_fails)

    outcome = talos.run("alternate uselessly")

    assert outcome.halt is Halt.STUCK
    assert outcome.steps_used == 4, "A B then A B again: caught on the second B"
    assert outcome.changed == []


def test_a_three_step_ritual_that_achieves_nothing_is_stuck(ws):
    """Cycles longer than two, up to the window, close the same way."""
    calls = [call("read_file", path=f"src/{name}.py") for name in ("a", "b", "c")]
    engine = ScriptedEngine(calls * 7)
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=20, target_steps=6),
                  verifier=always_fails)

    outcome = talos.run("walk in a circle")

    assert outcome.halt is Halt.STUCK
    assert outcome.steps_used == 5, "three to establish the cycle, two to confirm"


def test_a_loop_interrupted_by_silence_is_still_a_loop(ws):
    """The old single slot was reset by an empty reply, which masked the repeat.

    A, silence, A read as two unrelated steps. Silence is not progress -- it is
    already a noop -- so it must not erase what the model was circling.
    """
    a = call("edit_file", path="src/lib.py", old_string="absent", new_string="x")
    engine = ScriptedEngine([a, "", a, ""])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=8, target_steps=6),
                  verifier=always_fails)

    outcome = talos.run("loop around a silence")

    assert outcome.halt is Halt.STUCK
    assert outcome.steps_used == 3, "the third step repeats the first"


def test_real_work_between_repeats_restarts_the_window(ws):
    """A revisit after a change must not be held against one before it.

    This is the half that keeps a wider window safe: exploring, changing
    something, then exploring the same way again is ordinary work.
    """
    reading = call("read_file", path="src/lib.py")
    writing = call("write_file", path="out.py", content="x = 1\n")
    engine = ScriptedEngine([reading, reading, writing,
                             reading, reading, writing, "Done."])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=10, target_steps=8),
                  verifier=passes_when_anything_changed)

    outcome = talos.run("explore, act, explore, act")

    assert outcome.halt is Halt.DONE, outcome.summary


def test_one_landed_edit_then_the_same_edit_forever_is_stuck(ws):
    """The window is emptied on a change *and* keeps that change's signature.

    Dropping it would lose the plainest loop there is: an edit that succeeds,
    then the identical edit re-issued forever, each retry changing nothing
    because the first one already landed.
    """
    writing = call("write_file", path="out.py", content="x = 1\n")
    engine = ScriptedEngine([writing] * 8)
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=10, target_steps=8),
                  verifier=always_fails)

    outcome = talos.run("write the same thing forever")

    # `write_file` reports a change every time, so this must *not* be cut short
    # for repetition -- it is doing something, however pointlessly.
    assert outcome.halt is Halt.BUDGET_EXHAUSTED


# ------------------------------------- every step sequence, not just the ones
#                                        anyone thought to write a test for
#
# The bug this closes survived because every futility test in this file used a
# single repeated call. A, B, A, B is not an exotic input -- it is what a model
# does when it has two plausible fixes and neither works -- and nothing here
# exercised it. Enumerating the space is the answer to "which case did we not
# think of", and it is cheap: the loop is deterministic given the script.
#
# `_reference` states the intended rule independently of `Talos`, in terms of
# the specification rather than the implementation. Agreement across every
# sequence in the space means a transcription error in either one shows up as a
# disagreement on some specific input, which the failure message prints.
#
# What the sweep does *not* pin is the width itself: both sides read
# `FUTILE_WINDOW`, so narrowing it to 1 leaves them agreeing with each other
# about the wrong rule. The width is pinned behaviourally instead, by the
# two-cycle and three-cycle tests above, which hardcode the step at which the
# loop must be noticed. Verified by setting the constant to 1 and confirming
# those two fail while the sweep still passes.

#: symbol -> (scripted reply, files it changes). Every symbol issues exactly one
#: tool call, so `is_noop` never fires and the sweep isolates `is_futile`.
_STEP_KINDS = {
    "A": (call("edit_file", path="src/lib.py",
               old_string="absent-a", new_string="x"), 0),
    "B": (call("edit_file", path="src/lib.py",
               old_string="absent-b", new_string="y"), 0),
    "C": (call("edit_file", path="src/lib.py",
               old_string="absent-c", new_string="z"), 0),
    "W": (call("write_file", path="out.py", content="x = 1\n"), 1),
}


def _reference(symbols: Sequence[str], stuck_after: int = 2):
    """The rule as specified, written without reference to `Talos`.

    A step is futile when it repeats a signature seen within the last
    `FUTILE_WINDOW` steps *and* changed nothing; a step that changes something
    restarts the window while remaining in it. `Ariadne.assess` checks the
    ceiling before staleness, so a run that would be stuck on its final
    permitted step reports budget exhaustion instead -- the order matters and
    is reproduced here deliberately.
    """
    from collections import deque

    recent: "deque[str]" = deque(maxlen=FUTILE_WINDOW)
    noops = 0
    max_steps = len(symbols)
    for step, symbol in enumerate(symbols, 1):
        changed = _STEP_KINDS[symbol][1]
        futile = symbol in recent and not changed
        if changed:
            recent.clear()
        recent.append(symbol)
        noops = 0 if not futile else noops + 1
        if step >= max_steps:
            return Halt.BUDGET_EXHAUSTED, step
        if noops >= stuck_after:
            return Halt.STUCK, step
    raise AssertionError("unreachable: the ceiling equals the script length")


def _drive(ws, symbols: Sequence[str]):
    engine = ScriptedEngine([_STEP_KINDS[s][0] for s in symbols])
    talos = Talos(engine, ws,
                  ariadne=Ariadne(max_steps=len(symbols),
                                  target_steps=len(symbols)),
                  verifier=always_fails)
    outcome = talos.run("sweep")
    return outcome.halt, outcome.steps_used


@pytest.mark.parametrize("alphabet,length", [
    ("ABW", 5),      # 243 sequences: two useless calls and one that works
    ("AB", 6),       # 64: the alternation family, at depth
    ("ABCW", 4),     # 256: cycles up to the window width
])
def test_the_loop_matches_the_specified_rule_on_every_sequence(
        ws, alphabet, length):
    """Exhaustive over the space, not a sample of it."""
    import itertools

    for symbols in itertools.product(alphabet, repeat=length):
        assert _drive(ws, symbols) == _reference(symbols), \
            f"disagreed on {''.join(symbols)}"


def test_no_sequence_of_real_work_is_ever_called_stuck(ws):
    """The false-positive direction, exhaustively.

    A wider window is only safe because of the `files_changed == 0` half. If
    that conjunction were ever dropped, this is what would start failing: runs
    that repeat themselves *and get something done each time* are not loops.
    """
    import itertools

    for length in range(1, 7):
        for symbols in itertools.product("W", repeat=length):
            halt, _ = _drive(ws, symbols)
            assert halt is not Halt.STUCK, f"{''.join(symbols)} did real work"


def test_every_short_cycle_that_achieves_nothing_is_caught(ws):
    """The false-negative direction, for every cycle the window should close.

    A cycle of period k is detectable once k signatures fit in the window, so
    every k up to `FUTILE_WINDOW` must halt STUCK -- and must do it promptly,
    within the cycle plus the two steps `stuck_after` requires, rather than
    merely somewhere before the ceiling.
    """
    import itertools

    useless = [s for s, (_, changed) in _STEP_KINDS.items() if not changed]
    for period in range(1, FUTILE_WINDOW + 1):
        for base in itertools.permutations(useless, min(period, len(useless))):
            if len(base) != period:
                continue
            symbols = (list(base) * 6)[:20]
            halt, steps = _drive(ws, symbols)
            assert halt is Halt.STUCK, f"cycle {''.join(base)} ran to the ceiling"
            assert steps <= period + 2, \
                f"cycle {''.join(base)} took {steps} steps to notice"


def test_a_cycle_longer_than_the_window_is_left_to_the_ceiling(ws):
    """Stated so the boundary is a decision rather than an accident.

    The window is deliberately finite: "again" means recently. A cycle wider
    than it is not detected here and the step ceiling remains the backstop --
    which is the documented division of labour, not an oversight.
    """
    useless = [s for s, (_, changed) in _STEP_KINDS.items() if not changed]
    assert len(useless) < FUTILE_WINDOW + 1, \
        "this test needs a cycle wider than the window to be inexpressible " \
        "with the current alphabet; widen _STEP_KINDS if that changes"


def test_repeated_empty_steps_are_stuck(ws):
    """No tools, no changes, verification failing -- more turns will not help."""
    engine = ScriptedEngine(["thinking", "still thinking", "yet more thinking"])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=8, target_steps=6),
                  verifier=always_fails)

    outcome = talos.run("achieve nothing")

    assert outcome.halt is Halt.STUCK
    assert outcome.changed == []


def test_budget_pressure_reaches_the_engine(ws):
    engine = ScriptedEngine([call("read_file", path="src/lib.py")] * 6)
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=4, target_steps=1))

    talos.run("keep going")

    assert any("BUDGET:" in p for p in engine.prompts)


# ------------------------------------------------------------------- safety

def test_the_jail_holds_through_the_loop(ws):
    engine = ScriptedEngine([
        call("write_file", path="../escaped.py", content="pwned"),
        "Done.",
    ])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)

    outcome = talos.run("try to escape")

    assert not (ws.root.parent / "escaped.py").exists()
    assert outcome.changed == []
    assert "refused" in "\n".join(talos.transcript)


def test_a_disallowed_command_is_refused_and_reported(ws):
    engine = ScriptedEngine([call("run_command", command="rm -rf ."), "Done."])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)

    talos.run("delete everything")

    assert "not on the allowlist" in "\n".join(talos.transcript)


# --------------------------------------------------- step-wise plan execution

def test_a_multi_step_plan_runs_step_by_step(ws):
    """A plan is executed, not merely quoted at the engine."""
    engine = ScriptedEngine([
        call("write_file", path="one.py", content="one\n"), "One done.",
        call("write_file", path="two.py", content="two\n"), "Two done.",
    ])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)
    seen = []

    talos.run("do both", plan=["write one", "write two"],
              on_event=lambda e: seen.append(e))

    steps = [e.text for e in seen if e.kind == "plan_step"]
    assert steps == ["write one", "write two"]
    assert (ws.root / "one.py").exists() and (ws.root / "two.py").exists()


def test_one_step_cannot_spend_the_whole_budget(ws):
    """The usual way a plan-shaped task fails: step one thrashes to exhaustion."""
    engine = ScriptedEngine([
        "thinking", "still thinking",                # step one burns out
        call("write_file", path="two.py", content="two\n"), "Two done.",
    ])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=12, target_steps=6),
                  verifier=passes_when_anything_changed)

    talos.run("do both", plan=["thrash", "write two"])

    assert (ws.root / "two.py").exists(), "step two still got its own allowance"


def test_breakage_is_caught_between_steps_not_at_the_end(ws):
    """The interim check is what makes stepwise execution worth doing.

    A step that leaves the tree unparseable is reported inside its own step,
    while the engine still has the context that produced it -- rather than
    surfacing several steps later as a mystery.
    """
    checked = []

    def interim(w, changed):
        broken = [p for p in changed
                  if p.suffix == ".py" and "def (" in w.read(p)]
        checked.append(list(changed))
        return Verdict(not broken, "syntax broken" if broken else "parses")

    engine = ScriptedEngine([
        call("write_file", path="bad.py", content="def (\n"), "Step one done.",
        call("write_file", path="bad.py", content="def ok():\n    pass\n"), "Fixed.",
        call("write_file", path="two.py", content="two\n"), "Two done.",
    ])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=12, target_steps=6),
                  verifier=passes_when_anything_changed, interim=interim)

    talos.run("do both", plan=["write one", "write two"])

    assert checked, "the interim check must run between steps"
    assert "def (" not in (ws.root / "bad.py").read_text(encoding="utf-8"), \
        "the breakage was repaired inside its own step"


def test_the_last_step_faces_the_real_verifier(ws):
    """Intermediate steps only have to leave the tree intact; the last one
    answers for the whole task."""
    full, interim = [], []

    def full_verifier(w, changed):
        full.append(1)
        return Verdict(True, "the real ladder")

    def quick(w, changed):
        interim.append(1)
        return Verdict(True, "parses")

    engine = ScriptedEngine([
        call("write_file", path="one.py", content="one\n"), "One done.",
        call("write_file", path="two.py", content="two\n"), "Two done.",
    ])
    talos = Talos(engine, ws, verifier=full_verifier, interim=quick)

    talos.run("do both", plan=["write one", "write two"])

    assert interim, "intermediate steps used the cheap check"
    assert full, "the last step used the real verifier"


def test_the_closing_phase_settles_the_task_not_the_last_step(ws):
    """A plan is a guess about how to get there; the verifier decides arrival.

    Without this, a run whose work is complete but whose *last step* stalled
    reports failure -- which is what a live run did: both files written, four
    steps planned, `refusal` returned.
    """
    engine = ScriptedEngine([
        call("write_file", path="one.py", content="one\n"), "One done.",
        "nothing left for me to do here",             # step two stalls
        "nothing left for me to do here",
        "Everything the task asked for is present.",  # the closing phase
    ])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=12, target_steps=6),
                  verifier=passes_when_anything_changed)

    outcome = talos.run("do both", plan=["write one", "do nothing"])

    assert outcome.succeeded, outcome.summary
    assert (ws.root / "one.py").exists()


def test_a_plan_where_nothing_happened_still_cannot_pass(ws):
    """The closing phase must not become a second door onto the vacuous pass.

    It is allowed to verify without acting *because* earlier steps acted. When
    none did, that reasoning does not apply and the rule stands.
    """
    engine = ScriptedEngine(["nothing", "nothing", "nothing",
                             "nothing", "nothing", "nothing", "all done"])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=12, target_steps=6),
                  verifier=lambda w, c: Verdict(True, "vacuously fine"))

    outcome = talos.run("do both", plan=["do nothing", "also nothing"])

    assert not outcome.succeeded
    assert outcome.changed == []


def test_a_step_that_breaks_the_tree_is_rolled_back(ws):
    """The undo journal, actually reached.

    Without this the next step builds on code that does not parse, and every
    later failure is a consequence of this one.
    """
    def parses(w, changed):
        for path in changed:
            try:
                compile(w.read(path), str(path), "exec")
            except SyntaxError:
                return Verdict(False, "does not parse")
        return Verdict(bool(changed), "fine")

    engine = ScriptedEngine([
        call("write_file", path="good.py", content="a = 1\n"), "Step one done.",
        call("write_file", path="broken.py", content="def (:\n"), "Step two done.",
        "Everything is in place.",
    ])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=12, target_steps=8),
                  verifier=parses, interim=parses)

    talos.run("build both", plan=["write good", "write broken"])

    assert (ws.root / "good.py").exists(), "the good step survives"
    assert not (ws.root / "broken.py").exists(), "the broken step was undone"
    assert "was undone" in "\n".join(talos.transcript)


def test_a_rolled_back_creation_leaves_the_change_set(ws):
    """A file that no longer exists is not a change the run made."""
    def parses(w, changed):
        for path in changed:
            try:
                compile(w.read(path), str(path), "exec")
            except SyntaxError:
                return Verdict(False, "does not parse")
        return Verdict(bool(changed), "fine")

    engine = ScriptedEngine([
        call("write_file", path="good.py", content="a = 1\n"), "One.",
        call("write_file", path="broken.py", content="def (:\n"), "Two.",
        "Done.",
    ])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=12, target_steps=8),
                  verifier=parses, interim=parses)

    outcome = talos.run("build both", plan=["write good", "write broken"])

    names = [p.name for p in outcome.changed]
    assert "good.py" in names
    assert "broken.py" not in names


def test_a_step_that_merely_ran_out_of_budget_is_not_undone(ws):
    """Unfinished is not broken. Undoing here would throw away real work."""
    engine = ScriptedEngine([
        call("write_file", path="one.py", content="one\n"), "Step one done.",
        call("write_file", path="two.py", content="two\n"),
        call("write_file", path="two.py", content="two\n"),
        call("write_file", path="two.py", content="two\n"),
        "Everything is in place.",
    ])
    # Never reaches a verdict on the second step: it keeps calling tools.
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=12, target_steps=8),
                  verifier=passes_when_anything_changed,
                  interim=passes_when_anything_changed)

    talos.run("build both", plan=["write one", "write two"])

    assert (ws.root / "one.py").exists()
    assert (ws.root / "two.py").exists(), "budget exhaustion is not a rollback"


def test_a_dry_run_step_is_not_rolled_back_from_disk(tmp_path):
    """Staging is already the undo; the journal is empty in a dry run."""
    (tmp_path / "src").mkdir()
    ws = Workspace(tmp_path, dry_run=True)

    def always_broken(w, changed):
        return Verdict(False, "does not parse")

    engine = ScriptedEngine([
        call("write_file", path="a.py", content="x = 1\n"), "One.",
        call("write_file", path="b.py", content="y = 2\n"), "Two.",
        "Done.",
    ])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=12, target_steps=8),
                  verifier=always_broken, interim=always_broken)

    talos.run("stage both", plan=["stage a", "stage b"])

    assert not (tmp_path / "a.py").exists(), "nothing was written in the first place"
    assert not (tmp_path / "b.py").exists()


def test_a_single_step_plan_stays_on_the_flat_path(ws):
    """Splitting a one-step plan buys nothing and costs an extra prompt."""
    engine = ScriptedEngine([call("write_file", path="one.py", content="one\n"),
                             "Done."])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)
    seen = []

    talos.run("do one", plan=["write one"], on_event=lambda e: seen.append(e))

    assert not [e for e in seen if e.kind == "plan_step"]
    assert (ws.root / "one.py").exists()


def test_the_engine_is_told_which_step_it_is_on(ws):
    engine = ScriptedEngine([
        call("write_file", path="one.py", content="one\n"), "One done.",
        call("write_file", path="two.py", content="two\n"), "Two done.",
    ])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)

    talos.run("do both", plan=["write one", "write two"])

    assert "Step 1 of 2" in "\n".join(talos.transcript)
    assert "Step 2 of 2" in "\n".join(talos.transcript)


# ------------------------------------------------------- doing nothing

def test_a_passing_verdict_cannot_end_a_run_that_did_nothing(ws):
    """The engine may request verification; it may not pass by declining to act.

    `accept_everything` and the real Oracle both approve an empty change set,
    so without this an engine that simply never calls a tool is told it
    succeeded.
    """
    engine = ScriptedEngine(["Nothing needs doing.", "Still nothing.", "Done."])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=3, target_steps=2),
                  verifier=lambda w, c: Verdict(True, "vacuously fine"))

    outcome = talos.run("add a feature")

    assert outcome.halt is not Halt.DONE
    assert not outcome.succeeded


def test_the_engine_is_told_why_its_verification_was_rejected(ws):
    engine = ScriptedEngine(["Nothing needs doing.", "Still nothing."])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=2, target_steps=1),
                  verifier=lambda w, c: Verdict(True, "vacuously fine"))

    talos.run("add a feature")

    assert "nothing to verify" in "\n".join(talos.transcript)


def test_a_read_only_task_can_still_succeed(ws):
    """Running something changes no file and is still real work.

    The discriminator is not whether a *file* changed -- otherwise "run the
    tests and tell me what breaks" could never finish. It is whether a tool
    that can change something outside the conversation succeeded, which
    `run_command` does and `read_file` does not.
    """
    engine = ScriptedEngine([call("run_command", command="python -c pass"),
                             "The tests pass."])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=3, target_steps=2),
                  verifier=lambda w, c: Verdict(True, "nothing to check"))

    outcome = talos.run("run the tests and tell me what breaks")

    assert outcome.halt is Halt.DONE
    assert outcome.changed == []


def test_reading_a_file_is_not_doing_the_task(ws):
    """The vacuous pass this loop exists to prevent, reached through `read_file`.

    `acted` used to count any successful call, so an engine could answer "add a
    subtract() to foo.py" by reading foo.py and asking to be verified: the
    change set is empty, every tier is vacuously satisfied, and the run reported
    "Completed and verified -- nothing was changed, so there was nothing to
    verify" having written nothing at all. Reading is how you find out what to
    do; it is not doing it.
    """
    engine = ScriptedEngine([call("read_file", path="src/lib.py"),
                             "I have added subtract().",
                             "Still added.", "Truly added."])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=4, target_steps=3),
                  verifier=lambda w, c: Verdict(True, "vacuously fine"))

    outcome = talos.run("add a subtract(a, b) to src/lib.py")

    assert outcome.halt is not Halt.DONE
    assert not outcome.succeeded
    assert outcome.changed == []
    assert "nothing to verify" in "\n".join(talos.transcript)


# ------------------------------------------------- structured message history

class MessageEngine:
    """An engine that takes a message array. Records what it was handed."""

    name = "messages"

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def generate(self, prompt, context, cancelled):
        raise AssertionError("the structured path should have been preferred")

    def generate_messages(self, messages, cancelled):
        self.seen.append([dict(m) for m in messages])
        yield self.replies.pop(0) if self.replies else "Nothing further."


def test_a_message_engine_is_preferred_over_the_flat_one(ws):
    engine = MessageEngine(["Done."])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=1, target_steps=1),
                  verifier=passes_when_anything_changed)

    talos.run("do a thing")

    assert engine.seen, "generate_messages was never called"


def test_the_conversation_arrives_role_tagged(ws):
    """Flattened, the model's own replies are `## Assistant` headings inside one
    enormous user turn -- it has to infer the structure of its own conversation
    from markdown."""
    engine = MessageEngine([
        call("write_file", path="a.py", content="x = 1\n"),
        "All done.",
    ])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=4, target_steps=3),
                  verifier=passes_when_anything_changed)

    talos.run("write a file")

    last = engine.seen[-1]
    assert last[0]["role"] == "system"
    roles = [m["role"] for m in last]
    assert "assistant" in roles, "the engine's own prior reply must be an assistant turn"
    assert not any(m["role"] == "assistant" and not m["content"] for m in last)


def test_the_system_message_is_stable_across_steps(ws):
    """A prefix that changes every turn is a prefix no cache can ever hit."""
    engine = MessageEngine(["thinking", "still thinking", "done"])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=3, target_steps=2),
                  verifier=always_fails)

    talos.run("do a thing")

    systems = {m[0]["content"] for m in engine.seen}
    assert len(engine.seen) >= 2
    assert len(systems) == 1, "the system message drifted between steps"


def test_the_whole_prefix_grows_by_append_only(ws):
    """A stable system message is necessary for caching and not sufficient.

    Providers key a prefix cache on the longest matching *prefix of the whole
    request*, not on message[0]. If any earlier message is rewritten between
    steps, everything after the rewrite is a miss no matter how stable the
    system prompt was. Nothing tested that, so the property could have been lost
    to any change that touched history assembly -- including this one.

    Compaction genuinely does rewrite the middle, which is an unavoidable miss
    and the reason it should not run more often than it must. This asserts the
    append-only property in the case where compaction has not fired.
    """
    for name in ("a", "b", "c"):
        (ws.root / "src" / f"{name}.py").write_text("x = 1\n", encoding="utf-8")
    # Distinct calls each step: prose-only steps are noops and identical calls
    # are futile, and either would halt the run before it built a prefix worth
    # asserting on.
    engine = MessageEngine([call("read_file", path=f"src/{n}.py") for n in "abc"])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=3, target_steps=9),
                  lethe=Lethe(max_tokens=1_000_000),      # never compacts
                  verifier=always_fails)

    talos.run("do a thing")

    assert len(engine.seen) >= 3
    for earlier, later in zip(engine.seen, engine.seen[1:]):
        assert later[:len(earlier)] == earlier, (
            "a message before the end was rewritten; every token after it is a "
            "cache miss")


def test_the_system_message_carries_the_tools_and_constitution(ws):
    engine = MessageEngine(["Done."])
    talos = Talos(engine, ws, constitution="Never delete a test.",
                  ariadne=Ariadne(max_steps=1, target_steps=1))

    talos.run("do a thing")

    system = engine.seen[0][0]["content"]
    assert "Never delete a test." in system
    assert "write_file" in system, "the tool protocol belongs in the system message"


def test_retrieval_context_is_sent_once_next_to_the_task(ws):
    engine = MessageEngine(["Done."])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=1, target_steps=1))

    talos.run("do a thing", context="# a.py:1-2\ncode")

    contents = [m["content"] for m in engine.seen[0]]
    assert sum("<repository_excerpts>" in c for c in contents) == 1


def test_an_engine_without_the_method_still_works(ws):
    """The slot must stay open to something that only maps bytes to bytes."""
    engine = ScriptedEngine([call("write_file", path="a.py", content="x\n"), "Done."])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)

    assert talos.run("write a file").succeeded


# ---------------------------------------------------------- parallel dispatch

def test_independent_reads_in_one_turn_overlap(ws):
    """A frontier model answers with several reads at once and expects them to
    overlap. Serialised, the wall-clock is the sum and the model learns to emit
    one call per step, which spends the step budget instead."""
    import threading
    import time

    seen = []
    barrier = threading.Barrier(3, timeout=5)

    class SlowRead(ReadFile):
        def run(self, args, ws):
            seen.append(threading.current_thread().name)
            # Passes only if all three are in flight at once; times out and
            # raises if they are run one after another.
            barrier.wait()
            return super().run(args, ws)

    registry = ToolRegistry([SlowRead(), WriteFile(), AskUser()])
    engine = ScriptedEngine(["\n".join(
        call("read_file", path="src/lib.py") for _ in range(3)), "Done."])
    talos = Talos(engine, ws, tools=registry,
                  ariadne=Ariadne(max_steps=2, target_steps=2),
                  verifier=passes_when_anything_changed)

    talos.run("read it three times")

    assert len(seen) == 3
    assert len(set(seen)) > 1, "every call ran on the same thread"


def test_results_are_ordered_by_issue_not_by_completion(ws):
    """The transcript must read the same whether or not anything overlapped."""
    import threading

    class Staggered(ReadFile):
        def run(self, args, ws):
            # The last call issued finishes first.
            delay = {"src/a.py": 0.05, "src/b.py": 0.02, "src/c.py": 0.0}
            threading.Event().wait(delay.get(args.get("path"), 0))
            return ToolResult(f"content of {args['path']}")

    for name in ("a", "b", "c"):
        (ws.root / "src" / f"{name}.py").write_text("x\n", encoding="utf-8")

    registry = ToolRegistry([Staggered(), WriteFile()])
    engine = ScriptedEngine(["\n".join(
        call("read_file", path=f"src/{n}.py") for n in "abc"), "Done."])
    talos = Talos(engine, ws, tools=registry,
                  ariadne=Ariadne(max_steps=2, target_steps=2),
                  verifier=passes_when_anything_changed)

    talos.run("read three files")

    results = "\n".join(talos.transcript)
    assert results.index("src/a.py") < results.index("src/b.py") < results.index("src/c.py")


def test_a_write_between_reads_still_separates_them(ws):
    """Adjacency, not "gather the reads first".

    Reordering would change what the calls see: a read issued after a write is
    entitled to observe that write.
    """
    order = []

    class Recording(ReadFile):
        def run(self, args, ws):
            order.append(f"read:{args.get('path')}")
            return super().run(args, ws)

    class RecordingWrite(WriteFile):
        def run(self, args, ws):
            order.append("write")
            return super().run(args, ws)

    registry = ToolRegistry([Recording(), RecordingWrite()])
    engine = ScriptedEngine([
        call("read_file", path="src/lib.py")
        + "\n" + call("write_file", path="src/lib.py", content="value = 2\n")
        + "\n" + call("read_file", path="src/lib.py"),
        "Done."])
    talos = Talos(engine, ws, tools=registry,
                  ariadne=Ariadne(max_steps=2, target_steps=2),
                  verifier=passes_when_anything_changed)

    talos.run("read, write, read")

    assert order == ["read:src/lib.py", "write", "read:src/lib.py"]


def test_ask_user_is_never_run_concurrently(ws):
    """Not consequential -- it changes nothing -- and still must not overlap.

    Two questions at once go to one person through one channel. This is why the
    predicate is `parallel_safe` rather than a reuse of `CONSEQUENTIAL`.
    """
    from knossos.tools import AskUser

    assert not AskUser.parallel_safe
    assert not ToolRegistry.default().parallel_safe("ask_user")
    # ...while the reads it sits beside are.
    assert ToolRegistry.default().parallel_safe("read_file")
    assert ToolRegistry.default().parallel_safe("search")


def test_an_unknown_tool_runs_alone(ws):
    """So its failure stays attributable to the call that caused it."""
    assert not ToolRegistry.default().parallel_safe("no_such_tool")


def test_tool_events_stream_rather_than_arriving_all_at_once(ws):
    """A single-call turn is the common case and must not be delayed.

    Collecting every event and draining them after the whole turn is the easy
    way to keep emission off the worker threads, and it makes the editor show a
    turn happening all at once instead of watching it work.
    """
    order = []

    class Narrating(ReadFile):
        def run(self, args, ws):
            order.append(f"ran:{args.get('path')}")
            return super().run(args, ws)

    def watch(event):
        if event.kind == "tool":
            order.append(f"event:{event.call.args.get('path')}")

    registry = ToolRegistry([Narrating(), WriteFile()])
    engine = ScriptedEngine([
        call("read_file", path="src/lib.py")
        + "\n" + call("write_file", path="src/lib.py", content="value = 2\n")
        + "\n" + call("read_file", path="src/lib.py"),
        "Done."])
    talos = Talos(engine, ws, tools=registry,
                  ariadne=Ariadne(max_steps=2, target_steps=2),
                  verifier=passes_when_anything_changed)

    talos.run("read, write, read", on_event=watch)

    # Each serial call is narrated before the next one starts.
    assert order.index("event:src/lib.py") < len(order) - 1
    assert order[0] == "ran:src/lib.py"
    assert order[1] == "event:src/lib.py"


def test_a_refused_call_does_not_block_its_neighbours(ws):
    """Permission is decided for the whole turn before anything runs."""
    asked = []

    def refuse_writes(call):
        asked.append(call.name)
        return call.name != "write_file"

    engine = ScriptedEngine([
        call("read_file", path="src/lib.py")
        + "\n" + call("write_file", path="src/new.py", content="x\n"),
        "Done."])
    talos = Talos(engine, ws, ask_permission=refuse_writes,
                  ariadne=Ariadne(max_steps=2, target_steps=2),
                  verifier=passes_when_anything_changed)

    talos.run("read and write")

    assert "write_file was not permitted" in "\n".join(talos.transcript)
    assert "value = 1" in "\n".join(talos.transcript)      # the read still ran


# ------------------------------------------------------- native tool messages

def native_call(tool: str, call_id: str, **args) -> str:
    """A fenced block carrying a provider id, as the engine emits after
    normalising a native `tool_calls` response."""
    import json
    return (f'```json\n'
            f'{json.dumps({"tool": tool, "args": args, "id": call_id})}\n```')


def test_calls_with_provider_ids_travel_as_native_tool_calls(ws):
    """The shape the provider sent is the shape it gets back.

    Re-rendered as markdown, a model is shown its own tool use as prose it
    wrote, and a multi-call turn cannot be represented at all -- there is
    nothing to pair a result with.
    """
    engine = MessageEngine([
        native_call("write_file", "call_abc", path="a.py", content="x = 1\n"),
        "All done.",
    ])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)

    talos.run("write a file")

    final = engine.seen[-1]
    assistant = [m for m in final if m["role"] == "assistant"]
    results = [m for m in final if m["role"] == "tool"]

    assert len(assistant) == 1
    assert assistant[0]["tool_calls"][0]["id"] == "call_abc"
    assert assistant[0]["tool_calls"][0]["function"]["name"] == "write_file"
    # The result is paired by the same id, which is how providers match them.
    assert len(results) == 1
    assert results[0]["tool_call_id"] == "call_abc"
    # The fenced block is gone from the content -- it is the tool_calls array now.
    assert "```json" not in assistant[0]["content"]


def test_a_hand_written_fence_is_not_reflected_back_as_native(ws):
    """No id means the model wrote the convention itself.

    Showing such a model an `assistant` turn with a `tool_calls` array would be
    reflecting a format it never produced, which is how a working prompted-JSON
    run gets broken by an improvement aimed at somebody else.
    """
    engine = MessageEngine([
        call("write_file", path="a.py", content="x = 1\n"),      # no id
        "All done.",
    ])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)

    talos.run("write a file")

    final = engine.seen[-1]
    assert not any(m.get("tool_calls") for m in final)
    assert not any(m["role"] == "tool" for m in final)
    assert any("```json" in str(m.get("content", "")) for m in final)


def test_a_turn_mixing_identified_and_bare_calls_stays_text(ws):
    """Half a turn cannot go natively: the bare call has no id to pair on, and
    inventing one produces a result matching nothing."""
    import json
    mixed = (f'```json\n{json.dumps({"tool": "read_file", "args": {"path": "src/lib.py"}, "id": "call_1"})}\n```'
             f'\n```json\n{json.dumps({"tool": "read_file", "args": {"path": "src/lib.py"}})}\n```')
    engine = MessageEngine([mixed, "Done."])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=2, target_steps=2),
                  verifier=passes_when_anything_changed)

    talos.run("read it twice")

    assert not any(m.get("tool_calls") for m in engine.seen[-1])


def test_multiple_native_calls_in_one_turn_each_keep_their_result(ws):
    """The thing the markdown rendering could not express at all."""
    engine = MessageEngine([
        native_call("read_file", "call_1", path="src/lib.py")
        + "\n" + native_call("list_dir", "call_2", path="src"),
        "Done.",
    ])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=2, target_steps=2),
                  verifier=passes_when_anything_changed)

    talos.run("look around")

    final = engine.seen[-1]
    assistant = [m for m in final if m.get("tool_calls")][0]
    results = [m for m in final if m["role"] == "tool"]

    assert [c["id"] for c in assistant["tool_calls"]] == ["call_1", "call_2"]
    assert [r["tool_call_id"] for r in results] == ["call_1", "call_2"]


def test_compaction_never_leaves_a_call_without_its_result(ws):
    """The pair is adjacent when appended and not afterwards.

    Lethe elides the largest entry, which slices the string and so returns a
    plain `str` -- dropping the structure from one half of the pair while the
    other keeps it. An `assistant` message carrying `tool_calls` with no
    matching `tool` messages is rejected outright by the providers this shape
    exists for, so a broken pair must fall back to text on both halves.
    """
    big = "x" * 40_000
    (ws.root / "src" / "big.py").write_text(big, encoding="utf-8")

    engine = MessageEngine([
        native_call("read_file", "call_big", path="src/big.py"),
        "Done.",
    ])
    talos = Talos(engine, ws, lethe=Lethe(max_tokens=400, keep_recent=2),
                  ariadne=Ariadne(max_steps=2, target_steps=2),
                  verifier=passes_when_anything_changed)

    talos.run("read the big file")

    for sent in engine.seen:
        called = [c["id"] for m in sent for c in m.get("tool_calls", [])]
        answered = [m["tool_call_id"] for m in sent if m["role"] == "tool"]
        assert called == answered, f"orphaned tool call in {called} vs {answered}"


def test_a_result_whose_call_was_summarised_away_is_not_sent_natively(ws):
    """The orphan in the other direction: a `tool` message answering nothing."""
    from knossos.talos import Entry
    from knossos.tools import ToolCall, ToolResult

    calls = (ToolCall("read_file", {"path": "a.py"}, id="call_1"),)
    results = (ToolResult("contents"),)
    talos = Talos(ScriptedEngine([]), ws)
    talos.transcript = [
        "## Task\ndo a thing",
        "## Earlier (summarised)\nfiles touched: a.py",   # the assistant turn, gone
        Entry("\n## Tool results\n...", role="tool", calls=calls,
              results=results, native=True),
    ]

    messages = talos._history()

    assert not any(m["role"] == "tool" for m in messages)
    assert any("Tool results" in str(m.get("content", "")) for m in messages)


def test_the_flat_prompt_is_unchanged_by_any_of_this(ws):
    """`Entry` is a `str` subclass precisely so this stays true.

    The flattened path is what keeps a bytes-to-bytes model a valid occupant of
    the engine slot, and it must not notice that the transcript learned to carry
    structure.
    """
    engine = ScriptedEngine([
        native_call("write_file", "call_abc", path="a.py", content="x = 1\n"),
        "Done.",
    ])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)

    talos.run("write a file")

    assert "## Assistant" in engine.prompts[-1]
    assert "## Tool results" in engine.prompts[-1]
    assert all(isinstance(entry, str) for entry in talos.transcript)


# ------------------------------------------------------------ prefix stability
#
# Every provider's prompt caching keys on an exact prefix: the request is billed
# at a fraction for however many leading tokens match the previous one, and at
# full price from the first byte that differs. A run re-sends the whole
# conversation each turn -- measured at +1,341 tokens per step on a read-heavy
# run, so a 14-step run costs 3.8x a 7-step one, not 2x -- which makes the
# cached prefix the single largest lever on what a run costs.
#
# Nothing enforced that prefix. Anything volatile placed early in the prompt --
# a timestamp, a step counter, the budget nudge moved to the front, retrieval
# re-run per turn -- silently drops the hit rate to zero. Nothing would fail;
# the run would simply cost several times more, and the only symptom is a
# number in `_report_cost` that nobody is watching.


def _prompts_over(ws, steps):
    """Drive a real run and capture the prompt built for each step."""
    engine = ScriptedEngine([call("read_file", path=f"src/f{i}.py")
                             for i in range(steps)])
    for i in range(steps):
        (ws.root / "src" / f"f{i}.py").write_text("# x\n" * 40, encoding="utf-8")

    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=steps, target_steps=steps),
                  verifier=always_fails)
    seen = []
    real = talos._prompt
    talos._prompt = lambda: (lambda p: (seen.append(p), p)[1])(real())
    talos.run("read them")
    return seen


def test_each_turn_extends_the_previous_prompt_rather_than_rewriting_it(ws):
    """The property prompt caching is billed on, asserted rather than assumed."""
    seen = _prompts_over(ws, 5)
    assert len(seen) >= 3, "need several turns to compare"

    for i in range(1, len(seen)):
        assert seen[i].startswith(seen[i - 1]), (
            f"step {i + 1} rewrote the prompt instead of appending to it; "
            f"every cached token from the first difference on is re-billed")


def test_the_structured_history_is_append_only_too(ws):
    """Hosted providers take `generate_messages`, so this is the path they bill.

    Same property, expressed on the message array: earlier messages must be
    untouched, because a provider matches the prefix message by message.
    """
    engine = ScriptedEngine([call("read_file", path="src/lib.py")] * 4)
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=4, target_steps=4),
                  verifier=always_fails)

    first = talos._history()
    talos.transcript.append("\n## Assistant\nsomething new")
    second = talos._history()

    assert second[:len(first)] == first, \
        "an appended turn must not disturb the messages before it"


# --------------------------------------------------- fitting the context window

class Sized(ScriptedEngine):
    """A scripted engine that also advertises a context window."""

    def __init__(self, replies, context_window):
        super().__init__(replies)
        self.context_window = context_window


def test_the_budget_shrinks_to_fit_a_small_window(ws):
    """Lethe's 24 000 default was applied regardless of what was on the far end.

    Against a server handing out 16 384 the harness compacted to 24 000, called
    the prompt within budget, and the *server* then dropped the front of it --
    which is the task statement.
    """
    talos = Talos(Sized(["done"], context_window=8_000), ws,
                  ariadne=Ariadne(max_steps=1, target_steps=1))

    talos.run("do a thing")

    assert talos.lethe.max_tokens < 8_000
    assert talos.lethe.max_tokens >= MIN_TRANSCRIPT_BUDGET


def test_the_budget_accounts_for_the_header_and_the_reply(ws):
    """The transcript gets what the tool schema and the reply reserve leave."""
    window = 20_000
    talos = Talos(Sized(["done"], context_window=window), ws,
                  ariadne=Ariadne(max_steps=1, target_steps=1))

    talos.run("do a thing")

    header_tokens = estimate_tokens(talos.tools.render())
    assert talos.lethe.max_tokens <= window - Talos.OUTPUT_RESERVE - header_tokens


def test_a_large_window_does_not_raise_the_configured_budget(ws):
    """A bigger window is not a reason to send more than was asked for."""
    talos = Talos(Sized(["done"], context_window=1_000_000), ws,
                  lethe=Lethe(max_tokens=5_000),
                  ariadne=Ariadne(max_steps=1, target_steps=1))

    talos.run("do a thing")

    assert talos.lethe.max_tokens == 5_000


def test_an_engine_that_cannot_say_leaves_the_budget_alone(ws):
    """Most hosted providers do not publish a per-deployment window."""
    talos = Talos(ScriptedEngine(["done"]), ws, lethe=Lethe(max_tokens=24_000),
                  ariadne=Ariadne(max_steps=1, target_steps=1))

    talos.run("do a thing")

    assert talos.lethe.max_tokens == 24_000


def test_a_shrink_is_recomputed_not_accumulated(ws):
    """Each turn is sized from the configured default, not from last turn.

    Otherwise a tool schema that grows once would ratchet the budget down for
    the rest of the session and never let it back up.
    """
    engine = Sized(["a", "b", "c"], context_window=9_000)
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=3, target_steps=2),
                  verifier=always_fails)

    talos.run("do a thing")
    first = talos.lethe.max_tokens

    engine.context_window = 1_000_000
    talos.resume("carry on")

    assert first < 24_000
    assert talos.lethe.max_tokens == 24_000, "it must be able to go back up"


def test_a_window_too_small_for_any_transcript_is_reported(ws, capsys):
    """Compaction cannot fix a window the tool schema has already filled."""
    talos = Talos(Sized(["done"], context_window=4_200), ws,
                  ariadne=Ariadne(max_steps=1, target_steps=1))

    talos.run("do a thing")

    assert talos.lethe.max_tokens == MIN_TRANSCRIPT_BUDGET
    assert "raise the server's window" in capsys.readouterr().err


# ------------------------------------------------- the verifier's baseline hook

class PreparedVerifier:
    """A verifier that wants a look at the tree before anything changes."""

    def __init__(self) -> None:
        self.prepared_at: List[int] = []
        self.calls = 0

    def prepare(self, ws):
        self.prepared_at.append(len(list(ws.root.glob("*.new"))))

    def __call__(self, ws, changed):
        self.calls += 1
        return Verdict(bool(changed), "changed something")


def test_the_verifier_is_prepared_before_the_first_change(ws):
    """A baseline taken after the agent writes measures the agent's own work."""
    verifier = PreparedVerifier()
    engine = ScriptedEngine([call("write_file", path="a.new", content="x\n"), "Done."])
    talos = Talos(engine, ws, verifier=verifier)

    talos.run("write a file")

    assert verifier.prepared_at == [0], "prepared before anything was written"


def test_a_run_that_never_changes_anything_never_pays_for_a_baseline(ws):
    """A baseline costs a whole test suite; a question should not buy one."""
    verifier = PreparedVerifier()
    engine = ScriptedEngine([call("read_file", path="src/lib.py"), "It sets value to 1."])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=2, target_steps=1),
                  verifier=verifier)

    talos.run("what does lib.py do?")

    assert verifier.prepared_at == []


def test_a_verifier_without_the_hook_is_still_valid(ws):
    """One required method is what keeps a bare function a valid verifier."""
    engine = ScriptedEngine([call("write_file", path="a.new", content="x\n"), "Done."])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)

    assert talos.run("write a file").succeeded


def test_a_broken_baseline_does_not_end_the_run(ws):
    """A worse verdict is not a reason to abandon the task."""
    class Exploding(PreparedVerifier):
        def prepare(self, ws):
            raise RuntimeError("no baseline for you")

    engine = ScriptedEngine([call("write_file", path="a.new", content="x\n"), "Done."])
    talos = Talos(engine, ws, verifier=Exploding())

    assert talos.run("write a file").succeeded


def test_a_run_whose_every_call_was_refused_did_not_succeed(ws):
    """Refusals are errors, so nothing was accomplished."""
    engine = ScriptedEngine([
        call("write_file", path="src/new.py", content="x = 1\n"),
        "I could not proceed.",
    ])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=3, target_steps=2),
                  verifier=lambda w, c: Verdict(True, "vacuously fine"),
                  ask_permission=Asker(answer=False))

    outcome = talos.run("add a file")

    assert not outcome.succeeded


# ------------------------------------------------------------------ thoughts

class Thought(str):
    """Stands in for `engine.Thought`, which Talos detects by type name."""


class ThinkingEngine:
    name = "thinking"

    def __init__(self, script):
        self.script = list(script)

    def generate(self, prompt, context, cancelled):
        yield from self.script.pop(0) if self.script else ["nothing left"]


def test_reasoning_is_forwarded_not_discarded(ws):
    """Execute mode used to drop reasoning entirely.

    A turn deciding what to do then looked identical to a hung one, in the
    mode where the user most needs to see why something is happening.
    """
    engine = ThinkingEngine([[Thought("let me look at the tests"), "All done."]])
    seen = []
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=1, target_steps=1))

    talos.run("do something", on_event=lambda e: seen.append(e))

    thoughts = [e.text for e in seen if e.kind == "thought"]
    assert thoughts == ["let me look at the tests"]


def test_reasoning_is_never_parsed_for_tool_calls(ws):
    """A scratchpad that mentions a tool call is not a request to run one."""
    engine = ThinkingEngine([[
        Thought(call("write_file", path="from_thought.py", content="x = 1\n")),
        "I have not decided yet.",
    ]])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=1, target_steps=1),
                  ask_permission=Asker(answer=True))

    talos.run("think about it")

    assert not (ws.root / "from_thought.py").exists()


def test_reasoning_is_kept_out_of_the_reply(ws):
    engine = ThinkingEngine([[Thought("scratch"), "the answer"]])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=1, target_steps=1))

    talos.run("ask")

    assert "scratch" not in "\n".join(talos.transcript)


# ------------------------------------------------------------- empty replies

def test_an_empty_reply_is_not_a_claim_of_completion(ws):
    """The bug this guards against was found against a live reasoning model.

    It spent its whole token budget inside `<think>` and returned no content.
    Talos read "no tool calls" as "the engine believes it is finished", verified
    an empty change set -- which every tier passes, vacuously -- and reported
    success having done nothing at all.
    """
    engine = ScriptedEngine(["", "", ""])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=4, target_steps=2),
                  verifier=passes_when_anything_changed)

    outcome = talos.run("create a file")

    assert outcome.halt is not Halt.DONE
    assert not outcome.succeeded
    assert outcome.changed == []


def test_an_empty_reply_never_reaches_the_verifier(ws):
    """Nothing was proposed, so there is nothing to verify."""
    asked = []

    def recording_verifier(w, changed):
        asked.append(list(changed))
        return Verdict(True, "would pass vacuously")

    engine = ScriptedEngine(["", ""])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=2, target_steps=1),
                  verifier=recording_verifier)

    talos.run("create a file")

    assert asked == [], "an empty reply must not be verified as a finished run"


def test_an_empty_reply_is_explained_rather_than_silent(ws):
    """A blank turn is indistinguishable from a hang unless it is narrated."""
    seen = []
    engine = ScriptedEngine(["", ""])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=2, target_steps=1))

    talos.run("create a file", on_event=lambda e: seen.append(e))

    texts = [e.text for e in seen if e.kind == "text"]
    assert any("empty reply" in t for t in texts)
    assert any("truncated" in entry for entry in talos.transcript), \
        "the engine must be told why its turn was rejected"


def test_a_non_empty_reply_still_reaches_the_verifier(ws):
    """The fix must not stop an engine from requesting verification."""
    engine = ScriptedEngine([
        call("write_file", path="src/new.py", content="x = 1\n"),
        "That is everything.",
    ])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)

    outcome = talos.run("add a file")

    assert outcome.halt is Halt.DONE


# -------------------------------------------------------------- permissions

class Asker:
    """Records what it was asked and answers the same way every time."""

    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.calls: List = []

    def __call__(self, call) -> bool:
        self.calls.append(call)
        return self.answer

    def names(self) -> List[str]:
        return [c.name for c in self.calls]


def test_without_an_asker_the_loop_runs_unattended(ws):
    """A scripted run has nobody to ask; it must not deadlock or refuse."""
    engine = ScriptedEngine([call("write_file", path="src/new.py", content="x = 1\n"),
                             "Done."])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)

    talos.run("add a file")

    assert (ws.root / "src" / "new.py").exists()


def test_a_write_is_put_to_the_asker(ws):
    asker = Asker(answer=True)
    engine = ScriptedEngine([call("write_file", path="src/new.py", content="x = 1\n"),
                             "Done."])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed,
                  ask_permission=asker)

    talos.run("add a file")

    assert asker.names() == ["write_file"]
    assert (ws.root / "src" / "new.py").exists()


def test_reads_are_never_put_to_the_asker(ws):
    """Asking about harmless calls trains the user to approve without reading."""
    asker = Asker(answer=True)
    engine = ScriptedEngine([call("read_file", path="src/lib.py"),
                             call("list_dir", path="src"), "Done."])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=3, target_steps=2),
                  ask_permission=asker)

    talos.run("look around")

    assert asker.calls == []


def test_a_refused_write_does_not_happen(ws):
    engine = ScriptedEngine([call("write_file", path="src/new.py", content="x = 1\n"),
                             "Done."])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed,
                  ask_permission=Asker(answer=False))

    outcome = talos.run("add a file")

    assert not (ws.root / "src" / "new.py").exists()
    assert outcome.changed == []


def test_a_refusal_is_visible_to_the_engine(ws):
    """A refusal is a result, not an error: the engine can propose something else."""
    engine = ScriptedEngine([call("write_file", path="src/new.py", content="x = 1\n"),
                             "Understood, I will not."])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=3, target_steps=2),
                  ask_permission=Asker(answer=False))

    talos.run("add a file")

    assert "was not permitted" in "\n".join(talos.transcript)
    assert len(engine.prompts) > 1, "the run continues after a refusal"


def test_a_broken_asker_is_not_an_implicit_yes(ws):
    def explodes(call):
        raise RuntimeError("the dialog never appeared")

    engine = ScriptedEngine([call("write_file", path="src/new.py", content="x = 1\n"),
                             "Done."])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed,
                  ask_permission=explodes)

    talos.run("add a file")

    assert not (ws.root / "src" / "new.py").exists()


def test_commands_are_consequential_too(ws):
    """Refused, so the test does not actually shell out."""
    asker = Asker(answer=False)
    engine = ScriptedEngine([call("run_command", command="pytest -q"), "Done."])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=2, target_steps=1),
                  ask_permission=asker)

    talos.run("run the tests")

    assert asker.names() == ["run_command"]


# ------------------------------------------------------------------ dry runs

def test_a_dry_run_writes_nothing_until_applied(tmp_path):
    dry = Workspace(tmp_path, dry_run=True)
    engine = ScriptedEngine([
        call("write_file", path="a.py", content="staged = True\n"),
        "Done.",
    ])
    talos = Talos(engine, dry, verifier=passes_when_anything_changed)

    outcome = talos.run("propose a file")

    assert outcome.dry_run
    assert not (tmp_path / "a.py").exists()

    written = talos.apply()
    assert len(written) == 1
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "staged = True\n"


# ------------------------------------------------------------------ sessions

def test_resume_keeps_the_conversation(ws):
    engine = ScriptedEngine([
        call("write_file", path="one.py", content="one\n"), "Added one.",
        call("write_file", path="two.py", content="two\n"), "Added two.",
    ])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)

    first = talos.run("add one")
    assert first.halt is Halt.DONE
    length_after_first = len(talos.transcript)

    second = talos.resume("now add two")
    assert second.halt is Halt.DONE

    assert len(talos.transcript) > length_after_first
    assert talos.task == "add one", "the original task is retained"
    assert (ws.root / "one.py").exists() and (ws.root / "two.py").exists()
    assert second.steps_used == 2, "the budget resets per turn"


def test_resume_on_a_fresh_talos_behaves_like_a_first_task(ws):
    talos = Talos(ScriptedEngine(["Nothing to do."]), ws,
                  verifier=passes_when_anything_changed)
    outcome = talos.resume("look around")
    assert talos.task == "look around"
    assert outcome.steps_used >= 1


# ------------------------------------------------------------------- events

def test_events_describe_the_whole_run(ws):
    engine = ScriptedEngine([
        call("write_file", path="a.py", content="x\n"),
        "Done.",
    ])
    seen: List[Event] = []
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)

    talos.run("add a file", on_event=seen.append)

    kinds = [e.kind for e in seen]
    for expected in ("step", "tool", "verdict", "halt"):
        assert expected in kinds, f"missing {expected}; got {kinds}"

    tool_event = next(e for e in seen if e.kind == "tool")
    assert tool_event.call is not None and tool_event.call.name == "write_file"
    assert tool_event.result is not None and not tool_event.result.is_error

    halt_event = next(e for e in seen if e.kind == "halt")
    assert halt_event.halt is Halt.DONE


def test_the_default_verifier_is_named_to_be_uncomfortable():
    """A run with no verifier has no check on correctness; that should show."""
    from knossos.talos import accept_everything
    verdict = accept_everything(None, [])
    assert verdict.passed
    assert "no verifier configured" in verdict.summary


# ------------------------------------------------------------- re-planning
#
# The plan is produced before the agent has read a single file, so it is a guess
# about a codebase nobody has looked at. Until `replan` existed that guess was
# binding: `_drive_plan` walked the original list to the end, and the transcript
# said in as many words *do not restart the plan*. A plan that was wrong about
# the shape of the problem therefore spent the entire budget being wrong.
#
# Re-planning is a model call made from inside the loop, so most of what matters
# is what happens when it misbehaves.


def _acting_replies(count: int = 40) -> List[str]:
    """Replies where every step writes something and then stops.

    `_revise_plan` only fires for a step that produced evidence, so a scripted
    engine that merely talks would exercise the "did nothing, do not re-plan"
    path instead of the one under test.
    """
    replies: List[str] = []
    for index in range(count):
        replies.append(call("write_file", path=f"f{index}.py",
                            content=f"value = {index}\n"))
        replies.append("that step is done")
    return replies


def _plan_run(ws, replies, plan, replan=None, **kwargs):
    # `interim` explicitly, because it is the *interim* verdict that decides
    # whether a plan step failed -- and a step that never fails never re-plans.
    engine = ScriptedEngine(replies)
    talos = Talos(engine, ws, verifier=always_fails, interim=always_fails,
                  replan=replan, ariadne=Ariadne(max_steps=12), **kwargs)
    outcome = talos.run("task", plan=plan)
    return talos, outcome


def test_a_failed_step_revises_the_rest_of_the_plan(ws):
    seen = {}

    def replan(task, learned, done, remaining):
        seen["done"] = list(done)
        seen["remaining"] = list(remaining)
        return ["revised A", "revised B"]

    talos, _ = _plan_run(ws, _acting_replies(), ["one", "two", "three"], replan=replan)
    assert seen["done"] == ["one"]
    assert seen["remaining"] == ["two", "three"]
    assert "Plan revised" in "\n".join(talos.transcript)
    assert "revised A" in "\n".join(talos.transcript)


def test_steps_already_attempted_are_never_rewritten(ws):
    """They changed files; renumbering them would make the transcript lie."""
    def replan(task, learned, done, remaining):
        return ["something else"]

    talos, _ = _plan_run(ws, _acting_replies(), ["one", "two"], replan=replan)
    joined = "\n".join(talos.transcript)
    assert "Step 1 of 2\none" in joined


def test_re_planning_is_capped(ws):
    calls = []

    def replan(task, learned, done, remaining):
        calls.append(1)
        return ["a", "b", "c"]

    _plan_run(ws, _acting_replies(60), ["one", "two", "three"], replan=replan,
              max_replans=1)
    assert len(calls) == 1


def test_a_planner_that_raises_leaves_the_plan_alone(ws):
    """A stale plan is a known quantity; a half-applied revision is not."""
    def replan(task, learned, done, remaining):
        raise RuntimeError("planner exploded")

    talos, outcome = _plan_run(ws, _acting_replies(), ["one", "two"], replan=replan)
    assert "Plan revised" not in "\n".join(talos.transcript)
    assert outcome is not None


def test_a_planner_that_returns_junk_leaves_the_plan_alone(ws):
    for junk in ([], ["", "   "], None):
        talos, _ = _plan_run(ws, _acting_replies(), ["one", "two"],
                             replan=lambda *a, **k: junk)
        assert "Plan revised" not in "\n".join(talos.transcript)


def test_an_unchanged_revision_is_not_announced(ws):
    """Returning the same tail is 'no revision', not a revision to itself."""
    talos, _ = _plan_run(ws, _acting_replies(), ["one", "two", "three"],
                         replan=lambda task, learned, done, remaining: list(remaining))
    assert "Plan revised" not in "\n".join(talos.transcript)


def test_a_runaway_plan_is_truncated(ws):
    """A planner handed its own failure can answer with twenty steps.

    The budget does not grow to match, so each would get a starvation slice.
    """
    talos, _ = _plan_run(ws, _acting_replies(80), ["one", "two", "three"],
                         replan=lambda *a, **k: [f"s{i}" for i in range(20)])
    revised = [line for line in "\n".join(talos.transcript).splitlines()
               if line.startswith("## Step")]
    # tail of 2 * growth limit 2 == 4, plus the one already attempted
    assert len(revised) <= 5


def test_no_replanner_keeps_the_original_behaviour(ws):
    talos, _ = _plan_run(ws, _acting_replies(), ["one", "two", "three"], replan=None)
    joined = "\n".join(talos.transcript)
    assert "Plan revised" not in joined
    # The original three steps, in order, and nothing else.
    assert ["one", "two", "three"] == [
        line.strip() for line in joined.splitlines()
        if line.strip() in {"one", "two", "three"}]


def test_a_revision_emits_an_event_the_editor_can_show(ws):
    events = []
    engine = ScriptedEngine(_acting_replies())
    talos = Talos(engine, ws, verifier=always_fails, interim=always_fails,
                  replan=lambda *a, **k: ["revised"],
                  ariadne=Ariadne(max_steps=12))
    talos.run("task", plan=["one", "two"], on_event=events.append)
    assert any(e.kind == "plan_revised" for e in events)


# ------------------------------------------------------------------ delegation
#
# A subagent buys context, not speed: it runs to completion inline. The
# expensive part of a subtask is usually the reading, and in a single agent
# every file opened stays in the transcript and is re-sent every later step.
# A child reads into its own transcript, which is thrown away, and returns a
# paragraph.
#
# Most of what matters is what it shares and what it does not.


def _delegating(ws, replies, **kwargs):
    engine = ScriptedEngine(replies)
    talos = Talos(engine, ws, verifier=passes_when_anything_changed,
                  delegation=True, ariadne=Ariadne(max_steps=6), **kwargs)
    return talos, engine


def test_delegation_is_off_unless_asked_for(ws):
    plain = Talos(ScriptedEngine([]), ws)
    assert DELEGATE not in plain.tools.names
    assert DELEGATE in Talos(ScriptedEngine([]), ws, delegation=True).tools.names


def test_a_child_cannot_delegate_further(ws):
    """Not by instruction -- by construction. Each level multiplies agents."""
    parent = Talos(ScriptedEngine([]), ws, delegation=True)
    child = Talos(ScriptedEngine([]), ws, tools=parent.tools.without(DELEGATE),
                  delegation=False, depth=1)
    assert DELEGATE not in child.tools.names


def test_delegation_stops_at_the_depth_limit(ws):
    """Even asked for explicitly, a child at the ceiling gets no delegate tool."""
    deep = Talos(ScriptedEngine([]), ws, delegation=True,
                 depth=MAX_DELEGATION_DEPTH)
    assert DELEGATE not in deep.tools.names


def test_what_the_child_changed_becomes_the_parents_change_set(ws):
    """One change set, so the parent's verifier sees the child's edits."""
    talos, _ = _delegating(ws, [
        call(DELEGATE, task="write the helper"),
        call("write_file", path="child.py", content="x = 1\n"),
        "child done",
        "parent done",
    ])
    outcome = talos.run("build it")
    assert (ws.root / "child.py").exists()
    assert any(p.name == "child.py" for p in outcome.changed)


def test_the_childs_reading_never_reaches_the_parents_transcript(ws):
    """The whole reason the tool exists."""
    talos, _ = _delegating(ws, [
        call(DELEGATE, task="go and read a great many files"),
        call("read_file", path="src/lib.py"),
        "child done",
        "parent done",
    ])
    talos.run("build it")
    joined = "\n".join(talos.transcript)
    assert "Delegated task finished" in joined
    assert "value = 1" not in joined, "the child's file contents leaked upward"


def test_the_child_answers_to_the_same_permission_gate(ws):
    """A subagent must not be a way around a prompt the user is watching."""
    asked = []

    def deny(call_):
        asked.append(call_.name)
        return False

    talos, _ = _delegating(ws, [
        call(DELEGATE, task="write something"),
        call("write_file", path="sneaky.py", content="x = 1\n"),
        "child done",
        "parent done",
    ], ask_permission=deny)
    talos.run("build it")
    assert "write_file" in asked
    assert not (ws.root / "sneaky.py").exists()


def test_a_crashing_child_is_a_failed_subtask_not_a_failed_run(ws):
    from knossos.talos import Delegate

    def explode(task):
        raise RuntimeError("child fell over")

    result = Delegate(explode).run({"task": "do a thing"}, ws)
    assert result.is_error
    assert "child fell over" in result.content


def test_delegate_needs_a_task(ws):
    from knossos.talos import Delegate
    result = Delegate(lambda task: None).run({}, ws)
    assert result.is_error


def test_delegation_is_not_parallel_safe(ws):
    """Two children sharing one workspace would interleave staged writes."""
    from knossos.talos import Delegate
    assert Delegate.parallel_safe is False


def test_without_removes_a_tool_and_tolerates_absent_names(ws):
    registry = ToolRegistry.default()
    assert "read_file" not in registry.without("read_file").names
    assert registry.without("nothing_called_this").names == registry.names


# --------------------------------------------------- what the claim is worth
#
# `succeeded` means different things depending on what decided it, and the
# coding eval compares this harness against foreign ones whose only signal is a
# process exit code. Recording the source is what stops three incomparable
# quantities being printed under one `honest` heading.


def test_a_real_verifier_marks_the_claim_as_verified(ws):
    talos = Talos(ScriptedEngine(["done"]), ws,
                  verifier=passes_when_anything_changed)
    assert talos.claim_source == "verifier"


def test_the_default_verifier_marks_the_claim_as_unverified(ws):
    """`accept_everything` agrees with the engine, so DONE is its say-so.

    This is the control arm's configuration. Without the distinction its
    `honest` -- a tautology, since nothing can disagree -- would print beside a
    verified run's as though they measured the same thing.
    """
    talos = Talos(ScriptedEngine(["done"]), ws)
    assert talos.claim_source == "unverified"
    assert talos.run("anything").claim_source == "unverified"


def test_tool_calls_are_counted_across_the_run(ws):
    """`steps_used` alone ranks a batching habit as capability."""
    engine = ScriptedEngine([
        call("write_file", path="src/a.py", content="a = 1\n"),
        call("write_file", path="src/b.py", content="b = 2\n"),
        "Done.",
    ])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)

    outcome = talos.run("write two files")

    assert outcome.tools_used == 2
    assert outcome.steps_used >= 2
