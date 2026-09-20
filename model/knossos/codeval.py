"""The coding evaluation: tasks that are only done if the code works.

# Why this exists

`evalset.py` measures retrieval: 28 natural-language questions about this
repository, graded by substring and citation matching. It never constructs a
`Talos`, a `Workspace` or an `Oracle`, and not one of its cases requires editing
a file. So everything the project knew empirically was about retrieval-augmented
question answering, and the agentic half -- the loop, the verifier, the tools,
the halting policy -- had no measured results at all.

That is the gap this closes. Every case here is a task that is only complete if
the code afterwards behaves differently, and the grade comes from running tests,
not from reading the agent's prose.

# How a case is graded

Borrowed from SWE-bench, because the design is right:

    fail_to_pass    tests that fail before the change and must pass after.
                    This is what "did the task" means.
    pass_to_pass    tests that pass before and must still pass after. This is
                    what stops a fix that breaks something else scoring as
                    success -- the usual way an agent "solves" a bug.

A case passes only if **both** hold. Partial credit is reported (each set has
its own count) but does not make a case pass, because a change that fixes the
bug and breaks the suite is not a smaller success; it is a different failure.

# Anti-gaming

Grading by "run the tests" invites several obvious cheats. The grader closes
the ones it can structurally instead of hoping the evaluated agent behaves.

**Test files are restored before grading.** Whatever the agent did to
`tests/`, the originals are written back before pytest runs. An agent that
deletes the failing test, weakens its assertion, or adds a passing duplicate
gains exactly nothing -- the graded run uses the fixture's tests. This is
`restore_tests`, and it is not optional.

**The Python process is isolated from the fixture.** Safe-path startup, ignored
Python environment variables, disabled plugin autoload, an empty pytest config,
and `--noconftest` prevent workspace files from redefining how grading starts
or interprets tests.

**Node ids, not exit codes.** Each expectation names a specific test. A deleted
or renamed test does not "pass", it fails to collect, which is a failure.

**Added product files count as changes.** No-op and clarification cases cannot
be gamed by leaving the original files untouched while adding a second
implementation elsewhere.

`tamper` records whether the agent touched the test files even though it did not
help, because an agent that repeatedly tries is worth knowing about separately
from one that does not.

# What a score here does and does not mean

These fixtures are small and self-contained: a handful of files, a clear bug, a
test that names it. They measure whether the loop can find a defect, change the
right file, and not break the rest -- and they measure it honestly. They do not
measure performance on a large unfamiliar codebase, and a number from here
should never be compared to a SWE-bench number. See `scripts/coding_eval.py`
for what the harness will and will not claim.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

from . import sandbox
from .jsonrpc import log
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = ["CodingCase", "CaseResult", "CODING_CASES", "materialise",
           "restore_tests", "run_tests", "grade", "run_case", "run_suite"]


#: Long enough for a tiny fixture suite, short enough that a hung run is not a
#: coffee break. These fixtures run in well under a second.
TEST_TIMEOUT = 120

#: What `OpenAICompatEngine._explain` prefixes a refused request with. Matching
#: on it is coupling, but the alternative is silently scoring a rate-limited run
#: as though the model had answered badly.
_API_FAILURE = "Request failed:"

#: Consecutive provider-refused cases before the suite gives up. Two, because
#: one can be a single unlucky window and the third case has never told anyone
#: anything they did not already know by the second.
STARVED_CASES_BEFORE_ABORT = 2


@dataclass(frozen=True)
class CodingCase:
    """One task, its starting repository, and what proves it was done."""

    id: str
    #: What the user would type. Deliberately phrased as a person would, not as
    #: a specification -- if the task only works when spelled out precisely,
    #: that is worth discovering here rather than in use.
    prompt: str
    #: Relative path -> contents. The whole fixture repository.
    files: Dict[str, str]
    #: pytest node ids that must go from failing to passing.
    fail_to_pass: Sequence[str]
    #: pytest node ids that must be passing before and after.
    pass_to_pass: Sequence[str] = ()
    #: Tests the agent never sees. Written into the tree *after* it finishes and
    #: before grading, so they cannot be read, edited, or fitted.
    #:
    #: `restore_tests` closes one channel -- an agent that rewrites the
    #: assertions gets them overwritten and is recorded as `tamper`. It cannot
    #: close the other: an agent that writes code shaped to the assertions it
    #: was shown passes honestly by every measure this file had. Held-out tests
    #: are the only structural answer, because the thing being defended against
    #: is the agent optimising against exactly what it can see.
    held_out: Dict[str, str] = field(default_factory=dict)
    #: pytest node ids inside `held_out` that must pass. Empty is allowed and
    #: means this case asks the question of nobody -- most fixtures predate it.
    held_out_pass: Sequence[str] = ()
    #: bugfix | feature | cross_file | regression_trap
    kind: str = "bugfix"
    #: `core` or `hard`. The core set separates a working harness from a broken
    #: one and saturates quickly -- every model that can drive the loop at all
    #: scores 5/5. The hard set exists because a saturated benchmark measures a
    #: floor: it cannot tell an adequate agent from a good one, and six
    #: consecutive perfect runs is a statement about the fixtures, not the agent.
    tier: str = "core"
    #: What this case is really testing about the harness.
    note: str = ""
    #: `edit` (default), `edit_preserve`, `no_op`, or `clarify`.
    expected_action: str = "edit"

    def __post_init__(self) -> None:
        """Enforce the held-out invariants on *every* construction path.

        `load_cases` already checks these and reports them with the offending
        entry's index, which is the better message and stays. But the built-in
        suite is not built through `load_cases` -- `CODING_CASES` constructs
        `CodingCase` directly, and so does every test and every future generator
        script. The invariants were reachable only through the JSON door.

        That matters more here than the usual argument for validating at the
        boundary, because of *how* these two mistakes fail. Neither raises and
        neither produces a wrong-looking number: a shadowed file is shown to the
        agent and silently overwritten before grading, and a node id with no
        file behind it is collected from a tree that does not contain it and
        scores zero. Both surface as `overfit` -- the suite accusing the model
        of fitting the visible tests when the fault is in the fixture. A false
        accusation of gaming is the worst failure this file can produce, and it
        is indistinguishable from the real thing by inspection of the score.
        """
        clash = sorted(set(self.held_out) & set(self.files))
        if clash:
            raise ValueError(
                f"case {self.id!r}: `held_out` may not overwrite visible "
                f"files: {', '.join(clash)}")
        if self.held_out_pass and not self.held_out:
            raise ValueError(
                f"case {self.id!r}: `held_out_pass` without `held_out` files -- "
                f"these node ids would score a silent zero and read as overfit")
        if self.expected_action not in {"edit", "edit_preserve", "no_op", "clarify"}:
            raise ValueError(
                f"case {self.id!r}: unknown expected_action {self.expected_action!r}")
        if self.expected_action == "edit" and not self.fail_to_pass:
            raise ValueError(
                f"case {self.id!r}: edit cases need at least one fail_to_pass node")

    @property
    def test_files(self) -> List[str]:
        """Files restored before grading. Everything under `tests/`."""
        return [p for p in self.files if p.startswith("tests/")]


def load_cases(path: str | Path) -> List[CodingCase]:
    """Read cases from a JSON file, so a suite need not be Python.

    # Why this exists

    The built-in cases are the calibration set: small, self-contained, and
    saturated -- every model that can drive the loop at all scores 5/5 on the
    core tier. A saturated benchmark measures a floor. Comparing harness changes
    on anything harder means adding cases, and until now that meant editing
    `CODING_CASES` in this file, which makes the suite a property of the source
    tree rather than something a run can point at.

    The schema is exactly `CodingCase`: an id, a prompt, a `files` map that is
    the entire starting repository, and the two node-id lists that decide the
    grade. Everything that makes the grader trustworthy -- test-file restoration,
    node-id grading rather than exit codes, `pass_to_pass` catching a fix that
    breaks something else -- applies unchanged, because this produces the same
    objects the built-in set does.

    # What it is not

    It is not a SWE-bench runner. SWE-bench instances name a repository and a
    commit rather than carrying their files, and running them honestly needs
    per-instance environment setup and container isolation that this harness
    does not have. Converting one into this schema is possible and is the
    obvious next step; pretending the two are the same benchmark is not, and a
    number produced here must never be quoted as a SWE-bench score.

    Raises rather than returning a partial suite: a benchmark that silently
    dropped the cases it could not parse would report a rate over a denominator
    nobody chose.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):                    # tolerate {"cases": [...]}
        raw = raw.get("cases", [])
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty list of cases")

    cases: List[CodingCase] = []
    for index, entry in enumerate(raw):
        where = f"{path}[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: expected an object")
        missing = [key for key in ("id", "prompt", "files")
                   if not entry.get(key)]
        if "fail_to_pass" not in entry:
            missing.append("fail_to_pass")
        if missing:
            raise ValueError(f"{where}: missing {', '.join(missing)}")
        if not isinstance(entry["files"], dict):
            raise ValueError(f"{where}: `files` must be a path -> contents map")
        held_files = entry.get("held_out") or {}
        if not isinstance(held_files, dict):
            raise ValueError(f"{where}: `held_out` must be a path -> contents map")
        # Node ids without the files they live in would be collected from a tree
        # that does not contain them and score a silent zero, which reads as an
        # agent that overfitted rather than a suite that was written wrong.
        if entry.get("held_out_pass") and not held_files:
            raise ValueError(f"{where}: `held_out_pass` without `held_out` files")
        # A held-out file that collides with a visible one would be shown to the
        # agent by `materialise` and then overwritten before grading -- held out
        # in name only, and worse, silently.
        clash = sorted(set(held_files) & set(entry["files"]))
        if clash:
            raise ValueError(
                f"{where}: `held_out` may not overwrite visible files: "
                f"{', '.join(clash)}")
        # A case with nothing to fix cannot be passed or failed, which is worse
        # than a missing case: it inflates the denominator with a free point.
        cases.append(CodingCase(
            id=str(entry["id"]),
            prompt=str(entry["prompt"]),
            files={str(k): str(v) for k, v in entry["files"].items()},
            fail_to_pass=[str(n) for n in entry["fail_to_pass"]],
            pass_to_pass=[str(n) for n in entry.get("pass_to_pass", [])],
            held_out={str(k): str(v) for k, v in held_files.items()},
            held_out_pass=[str(n) for n in entry.get("held_out_pass", [])],
            kind=str(entry.get("kind", "bugfix")),
            tier=str(entry.get("tier", "core")),
            note=str(entry.get("note", "")),
            expected_action=str(entry.get("expected_action", "edit"))))

    duplicates = {c.id for c in cases if [x.id for x in cases].count(c.id) > 1}
    if duplicates:
        raise ValueError(f"{path}: duplicate case id(s): {', '.join(sorted(duplicates))}")
    return cases


@dataclass
class CaseResult:
    case_id: str
    kind: str
    passed: bool
    #: Of `fail_to_pass`, how many now pass.
    fixed: int = 0
    fixed_total: int = 0
    #: Of `pass_to_pass`, how many still pass.
    kept: int = 0
    kept_total: int = 0
    #: Of `held_out_pass`, how many pass. These tests were never on disk while
    #: the agent was running, so they are the only evidence here that separates
    #: solving the task from fitting the assertions that were visible.
    held: int = 0
    held_total: int = 0
    #: The agent edited a file under `tests/`. Did not help -- see module docs.
    tamper: bool = False
    #: Turns the provider refused outright -- HTTP errors, rate limits, quota.
    #: Tracked because a starved run and an incapable model produce the same
    #: zero, and reporting them the same way turns an infrastructure problem
    #: into a false claim about a model.
    api_errors: int = 0
    #: What the harness itself reported, for comparison against the truth.
    halt: str = ""
    harness_said_done: bool = False
    steps_used: int = 0
    #: Tool calls issued across the run. Reported beside `steps_used` because a
    #: step is not a fixed unit of work: a harness that batches five calls into
    #: one turn spends a fifth of the steps for the same work, so steps alone
    #: rank tool-batching habits as if they were capability.
    tools_used: int = 0
    #: Where `harness_said_done` came from, because the two sources are not the
    #: same measurement and must not be averaged into one `honest` column.
    #:
    #: * `verifier` -- a deterministic verdict from Oracle. A real self-assessment.
    #: * `exit_code` -- a foreign CLI returned 0. Process health, not a claim:
    #:   most agent CLIs exit 0 unless they crash, so `honest` collapses into
    #:   `passed` and a false fail becomes impossible to score.
    claim_source: str = ""
    #: Which budget the run was actually bounded by, so a steps-limited arm is
    #: never silently compared against a wall-clock-limited one.
    budget_kind: str = ""
    #: Configuration the provider forced the engine to give up mid-run. A case
    #: that finished with a halved reply allowance, or after minutes of backoff,
    #: is not the same configuration as one that ran clean -- and the console
    #: warning that says so does not survive into the trace.
    output_shrinks: int = 0
    throttle_waits: int = 0
    throttled_seconds: float = 0.0
    changed_files: List[str] = field(default_factory=list)
    action_passed: bool = True
    seconds: float = 0.0
    error: str = ""
    #: What this case cost, when the provider reported it.
    #:
    #: Stored as plain integers rather than an engine type, because
    #: `AgentFactory` is deliberately "anything with `.run(prompt)`" -- importing
    #: `engine.Usage` here would make the eval require a Knossos agent and
    #: quietly close the seam that makes a harness-vs-harness comparison
    #: possible at all.
    #:
    #: `requests == 0` means *unmeasured*, which is distinct from free: a
    #: scripted calibration agent has no engine, and some providers report
    #: nothing. The report says "not reported" rather than printing a zero.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    requests: int = 0
    #: How many samples were drawn before this one was kept. 1 unless
    #: `--best-of` was used.
    attempts: int = 1
    #: Whether the *first* sample would have solved it, tracked separately so a
    #: best-of-k score can never be quoted as if it were single-shot. They are
    #: different numbers about different things and one is k times the price.
    first_passed: Optional[bool] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def honest(self) -> bool:
        """Whether the harness's own verdict matched reality.

        The most useful number in the whole suite. A harness that fails a task
        and says so is merely weak; one that fails and reports success is
        actively misleading, and that is the failure mode this project exists
        to prevent. Counted separately from `passed` because they are different
        properties: a run can be honest and unsuccessful, which is fine.
        """
        return self.harness_said_done == self.passed

    @property
    def unreachable(self) -> bool:
        """Whether the provider refused, so this case measured nothing.

        A case that never reached the model is not a capability result and must
        not be counted as one. It is scored zero by every other property here --
        `passed` is False, `halt` reads `stuck` because an agent that received
        an error string in place of a reply makes no tool calls -- and those
        zeros are indistinguishable from an incapable model unless something
        says so explicitly.

        `api_errors and not passed` rather than `api_errors` alone: a run that
        hit one rate limit, retried, and went on to solve the task did measure
        the model, and forgiving it would discard a real result.

        Two ways to never reach the model, and the second was found by running
        this suite with no API key configured:

        * **The provider refused** -- `api_errors`, counted from the engine's
          failure text in the event stream.
        * **The request was never made.** A missing key, an unreachable host, a
          bad base URL: the engine raises before the first turn, `run_case`
          records it in `error`, and `api_errors` stays zero because no reply
          ever arrived to count. Every case reported `0/5 solved` and `5/5
          honest` -- a clean capability zero for a model nobody called.

        `steps_used == 0` is what separates that from an ordinary crash. A
        harness bug that kills a run mid-way still measured the turns it took;
        one that dies before the first turn measured nothing. The error text is
        printed either way, so nothing is hidden by excluding it from the rate.
        """
        never_ran = bool(self.error) and not self.steps_used
        return (bool(self.api_errors) or never_ran) and not self.passed

    @property
    def degraded(self) -> bool:
        """Whether the provider forced the engine to run in a smaller shape.

        The third state between `unreachable` and a clean result, and the one
        with no home before now. A refused request is visible in `api_errors`;
        a request that *succeeded* after ninety seconds of backoff with the
        reply allowance halved leaves no mark on the transcript at all. Both
        produce a number, and only one of them is a capability result.

        Kept separate from `unreachable` rather than folded into it: a degraded
        case did measure the model, just not the model as configured. Excluding
        it would discard real evidence; counting it silently is how a provider's
        free tier gets published as a model's ceiling.
        """
        return bool(self.output_shrinks or self.throttle_waits)

    @property
    def overfit(self) -> bool:
        """Passed the visible tests and failed the held-out ones.

        The failure `tamper` cannot see. `restore_tests` catches an agent that
        edits the assertions; nothing catches an agent that writes code shaped
        to the assertions it was shown. This is that signal, and it only exists
        for cases that carry held-out tests -- for the rest it is False because
        the question was never asked, not because the answer was no.
        """
        return bool(self.held_total) and self.held < self.held_total and (
            self.fixed == self.fixed_total and self.kept == self.kept_total)

    def summary(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        claim = "" if self.honest else "  <-- HARNESS DISAGREED"
        if self.overfit:
            claim += "  <-- FITTED THE VISIBLE TESTS"
        held = f" held {self.held}/{self.held_total}" if self.held_total else ""
        return (f"[{mark}] {self.case_id:<22} "
                f"fix {self.fixed}/{self.fixed_total} "
                f"keep {self.kept}/{self.kept_total}{held} "
                f"steps {self.steps_used:>2} "
                f"{self.seconds:5.1f}s{claim}")


# --------------------------------------------------------------------- cases

_CHUNK = '''\
def chunk(items, size):
    """Split `items` into lists of length `size`."""
    out = []
    for start in range(0, len(items) - size + 1, size):
        out.append(items[start:start + size])
    return out
'''

_STATS = '''\
def mean(values):
    """Arithmetic mean of `values`."""
    return sum(values) / len(values)


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if not ordered:
        raise ValueError("median of an empty sequence")
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2
'''

_REGISTRY = '''\
from .errors import UnknownPlugin

_PLUGINS = {}


def register(name, factory):
    _PLUGINS[name] = factory


def build(name):
    """Instantiate a registered plugin by name."""
    if name not in _PLUGINS:
        raise ValueError(f"no plugin named {name}")
    return _PLUGINS[name]()
'''

_ERRORS = '''\
class UnknownPlugin(KeyError):
    """Raised when a plugin name is not registered."""
'''


CODING_CASES: List[CodingCase] = [

    CodingCase(
        id="chunk-drops-remainder",
        kind="bugfix",
        prompt=("chunk() is dropping the last group when the list doesn't "
                "divide evenly. Fix it."),
        note=("The simplest real bug: an off-by-one in a range bound. Tests "
              "the loop can locate a defect from a description and edit one "
              "file. If this fails, nothing below it is meaningful."),
        files={
            "pkg/__init__.py": "",
            "pkg/collections_util.py": _CHUNK,
            "tests/test_chunk.py": '''\
from pkg.collections_util import chunk


def test_even_split():
    assert chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_keeps_the_remainder():
    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_size_larger_than_input():
    assert chunk([1, 2], 5) == [[1, 2]]
''',
        },
        fail_to_pass=["tests/test_chunk.py::test_keeps_the_remainder",
                      "tests/test_chunk.py::test_size_larger_than_input"],
        pass_to_pass=["tests/test_chunk.py::test_even_split"],
    ),

    CodingCase(
        id="mean-of-empty",
        kind="bugfix",
        prompt="mean() crashes on an empty list. It should return 0.0 instead.",
        note=("A guard clause, with a neighbouring function that already "
              "handles the empty case differently. An agent that 'fixes' both "
              "to be consistent breaks `pass_to_pass`."),
        files={
            "pkg/__init__.py": "",
            "pkg/stats.py": _STATS,
            "tests/test_stats.py": '''\
import pytest

from pkg.stats import mean, median


def test_mean_of_values():
    assert mean([1, 2, 3]) == 2


def test_mean_of_empty_is_zero():
    assert mean([]) == 0.0


def test_median_of_empty_still_raises():
    with pytest.raises(ValueError):
        median([])
''',
        },
        fail_to_pass=["tests/test_stats.py::test_mean_of_empty_is_zero"],
        pass_to_pass=["tests/test_stats.py::test_mean_of_values",
                      "tests/test_stats.py::test_median_of_empty_still_raises"],
    ),

    CodingCase(
        id="add-a-function",
        kind="feature",
        prompt=("Add a `clamp(value, low, high)` to pkg/numeric.py that returns "
                "value limited to the range, and raise ValueError if low > high."),
        note=("Pure addition -- nothing to find, only something to write. The "
              "control for the bugfix cases: if these pass and the bugfixes do "
              "not, the weakness is in locating code, not in editing it."),
        files={
            "pkg/__init__.py": "",
            "pkg/numeric.py": '''\
def scale(value, factor):
    return value * factor
''',
            "tests/test_numeric.py": '''\
import pytest

from pkg.numeric import clamp, scale


def test_scale_still_works():
    assert scale(3, 2) == 6


def test_clamp_within_range():
    assert clamp(5, 0, 10) == 5


def test_clamp_below_and_above():
    assert clamp(-1, 0, 10) == 0
    assert clamp(99, 0, 10) == 10


def test_clamp_rejects_an_inverted_range():
    with pytest.raises(ValueError):
        clamp(1, 10, 0)
''',
        },
        fail_to_pass=["tests/test_numeric.py::test_clamp_within_range",
                      "tests/test_numeric.py::test_clamp_below_and_above",
                      "tests/test_numeric.py::test_clamp_rejects_an_inverted_range"],
        # Deliberately in the same file: it fails to *collect* while `clamp` is
        # missing, so it is not a valid `pass_to_pass` until the import works.
        pass_to_pass=(),
    ),

    CodingCase(
        id="wrong-exception-across-files",
        kind="cross_file",
        prompt=("build() raises ValueError for an unregistered plugin, but "
                "callers expect the UnknownPlugin error this package already "
                "defines. Make it raise that instead."),
        note=("The type to raise is defined in a *different* file than the one "
              "to edit, and the test names neither. This is the case that "
              "needs `search`: without it the agent must guess filenames or "
              "read the tree one file at a time."),
        files={
            "pkg/__init__.py": "",
            "pkg/errors.py": _ERRORS,
            "pkg/registry.py": _REGISTRY,
            "tests/test_registry.py": '''\
import pytest

from pkg.errors import UnknownPlugin
from pkg.registry import build, register


def test_builds_a_registered_plugin():
    register("ok", lambda: "made")
    assert build("ok") == "made"


def test_unknown_plugin_raises_the_package_error():
    with pytest.raises(UnknownPlugin):
        build("nope")
''',
        },
        fail_to_pass=["tests/test_registry.py::test_unknown_plugin_raises_the_package_error"],
        pass_to_pass=["tests/test_registry.py::test_builds_a_registered_plugin"],
    ),

    CodingCase(
        id="fix-without-breaking",
        kind="regression_trap",
        prompt=("normalise() should strip whitespace and lowercase its input. "
                "It's not stripping. Fix it."),
        note=("The tempting one-line fix (`return text.strip().lower()`) is "
              "correct. The tempting *over*-fix -- also collapsing internal "
              "whitespace, which looks like part of 'normalise' -- breaks a "
              "passing test. Measures whether the agent does what was asked."),
        files={
            "pkg/__init__.py": "",
            "pkg/text.py": '''\
def normalise(text):
    """Strip surrounding whitespace and lowercase."""
    return text.lower()
''',
            "tests/test_text.py": '''\
from pkg.text import normalise


def test_lowercases():
    assert normalise("ABC") == "abc"


def test_strips_surrounding_whitespace():
    assert normalise("  Hello  ") == "hello"


def test_leaves_internal_spacing_alone():
    assert normalise("a  b") == "a  b"
''',
        },
        fail_to_pass=["tests/test_text.py::test_strips_surrounding_whitespace"],
        pass_to_pass=["tests/test_text.py::test_lowercases",
                      "tests/test_text.py::test_leaves_internal_spacing_alone"],
    ),
]


# ----------------------------------------------------------------- hard cases
#
# The core set above saturates: every model that can drive the loop scores 5/5,
# so it measures a floor and cannot rank anything. These are built to fail in
# *specific* ways, so that a failure says something:
#
#   the shallow fix          fixing where the test points, not where the bug is
#   the second defect        stopping after the first thing that looked wrong
#   the incomplete change    editing one call site of three
#   the search               finding one wrong thing among many similar ones
#
# Each still grades on `fail_to_pass` and `pass_to_pass`, so nothing here relies
# on judging *how* the agent worked -- only on whether the code ends up right.

_JOINER_BASE = '''\
class Joiner:
    """Joins items with a separator. Subclasses choose the separator."""

    sep = ", "

    def join(self, items):
        return self.sep.join(str(i) for i in items[:-1])
'''

_HUMANISE = '''\
UNITS = ["bytes", "KB", "MB", "GB"]


def humanise(count):
    """Render a byte count in the largest unit that keeps it >= 1."""
    return f"{count} {UNITS[0]}"
'''

_REPORT = '''\
from .format import humanise


def build_report(entries):
    """One line per entry: name, then its size."""
    return "\\n".join(f"{name}: {humanise(size)}" for name, size in entries)
'''


def _handler(index: int, code: int) -> str:
    return (f'CODE = {code}\n\n\n'
            f'def handle(payload):\n'
            f'    """Handler {index}."""\n'
            f'    return {{"handler": {index}, "code": CODE}}\n')


CODING_CASES += [

    CodingCase(
        id="fix-belongs-in-the-base-class",
        kind="cross_file",
        tier="hard",
        prompt=("Both the CSV and text joiners are dropping the last item. "
                "Fix it."),
        note=("The shallow fix -- overriding `join` in whichever subclass the "
              "agent looked at first -- makes one named test pass and leaves "
              "the other failing, so the case scores 1/2 and fails. Only a "
              "change in the shared base fixes both. Measures whether the agent "
              "locates the *cause* or patches at the point of the symptom."),
        files={
            "pkg/__init__.py": "",
            "pkg/base.py": _JOINER_BASE,
            "pkg/csvout.py": "from .base import Joiner\n\n\n"
                             "class CsvJoiner(Joiner):\n    sep = \",\"\n",
            "pkg/txtout.py": "from .base import Joiner\n\n\n"
                             "class TxtJoiner(Joiner):\n    sep = \" | \"\n",
            "tests/test_csvout.py": '''\
from pkg.csvout import CsvJoiner


def test_separator_is_a_comma():
    assert CsvJoiner().sep == ","


def test_keeps_every_item():
    assert CsvJoiner().join([1, 2, 3]) == "1,2,3"
''',
            "tests/test_txtout.py": '''\
from pkg.txtout import TxtJoiner


def test_empty_input_is_blank():
    assert TxtJoiner().join([]) == ""


def test_keeps_every_item():
    assert TxtJoiner().join(["a", "b"]) == "a | b"
''',
        },
        fail_to_pass=["tests/test_csvout.py::test_keeps_every_item",
                      "tests/test_txtout.py::test_keeps_every_item"],
        pass_to_pass=["tests/test_csvout.py::test_separator_is_a_comma",
                      "tests/test_txtout.py::test_empty_input_is_blank"],
    ),

    CodingCase(
        id="bug-two-files-from-the-test",
        kind="cross_file",
        tier="hard",
        prompt="The report always says 'bytes', even for large sizes. Fix it.",
        note=("The failing test is in `test_report.py`, the function it calls is "
              "in `report.py`, and the defect is in `format.py` -- which the "
              "test never mentions. Reachable by reading one import, or by "
              "`search`; not reachable by editing the file the test names."),
        files={
            "pkg/__init__.py": "",
            "pkg/format.py": _HUMANISE,
            "pkg/report.py": _REPORT,
            "tests/test_report.py": '''\
from pkg.report import build_report


def test_lists_every_entry():
    report = build_report([("a", 1), ("b", 2), ("c", 3)])
    assert len(report.splitlines()) == 3


def test_scales_to_larger_units():
    report = build_report([("big", 2048), ("huge", 5 * 1024 * 1024)])
    assert "2.0 KB" in report
    assert "5.0 MB" in report
''',
        },
        fail_to_pass=["tests/test_report.py::test_scales_to_larger_units"],
        pass_to_pass=["tests/test_report.py::test_lists_every_entry"],
    ),

    CodingCase(
        id="a-second-defect-behind-the-first",
        kind="bugfix",
        tier="hard",
        prompt=("parse_config isn't returning what it should for "
                "' timeout = 30 '. Fix it."),
        note=("Two defects, and fixing either alone leaves the test red with a "
              "*different* failure: keys keep their surrounding whitespace, and "
              "values stay strings. Rewards actually reading the new error "
              "rather than declaring victory after the first plausible edit -- "
              "which is what the repair loop and the verifier feedback exist "
              "for."),
        files={
            "pkg/__init__.py": "",
            "pkg/parser.py": '''\
def parse_config(text):
    """Parse `key = value` lines into a dict.

    Keys are stripped. Values that look like integers become integers.
    """
    out = {}
    for line in text.splitlines():
        if not line.strip() or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key] = value.strip()
    return out
''',
            "tests/test_parser.py": '''\
from pkg.parser import parse_config


def test_empty_text_is_an_empty_config():
    assert parse_config("") == {}


def test_comments_and_blanks_are_skipped():
    assert parse_config("\\n\\nnot-a-pair\\n") == {}


def test_keys_are_stripped_and_ints_are_ints():
    assert parse_config("  timeout = 30 ") == {"timeout": 30}


def test_non_numeric_values_stay_strings():
    assert parse_config("name = daedalus") == {"name": "daedalus"}
''',
        },
        fail_to_pass=["tests/test_parser.py::test_keys_are_stripped_and_ints_are_ints",
                      "tests/test_parser.py::test_non_numeric_values_stay_strings"],
        pass_to_pass=["tests/test_parser.py::test_empty_text_is_an_empty_config",
                      "tests/test_parser.py::test_comments_and_blanks_are_skipped"],
    ),

    CodingCase(
        id="every-call-site-must-change",
        kind="cross_file",
        tier="hard",
        prompt=("Add a `level` argument to emit() that defaults to \"info\". The "
                "alert module should pass \"error\" and the audit module should "
                "pass \"warn\"."),
        note=("Three files, and a partial job is visibly partial: each call site "
              "has its own named test. Measures whether the agent finishes a "
              "change rather than stopping at the first file, which is the "
              "usual failure on a real refactor."),
        files={
            "pkg/__init__.py": "",
            "pkg/core.py": '''\
def emit(message):
    """Render a log line."""
    return f"[info] {message}"
''',
            "pkg/alert.py": "from .core import emit\n\n\n"
                            "def alert(message):\n    return emit(message)\n",
            "pkg/audit.py": "from .core import emit\n\n\n"
                            "def audit(message):\n    return emit(message)\n",
            "tests/test_core.py": '''\
from pkg.core import emit


def test_message_is_rendered():
    assert "hello" in emit("hello")


def test_level_defaults_to_info():
    assert emit("hello") == "[info] hello"


def test_level_can_be_given():
    assert emit("hello", "debug") == "[debug] hello"
''',
            "tests/test_callers.py": '''\
from pkg.alert import alert
from pkg.audit import audit


def test_alert_is_an_error():
    assert alert("disk full") == "[error] disk full"


def test_audit_is_a_warning():
    assert audit("login") == "[warn] login"
''',
        },
        fail_to_pass=["tests/test_core.py::test_level_can_be_given",
                      "tests/test_callers.py::test_alert_is_an_error",
                      "tests/test_callers.py::test_audit_is_a_warning"],
        pass_to_pass=["tests/test_core.py::test_message_is_rendered",
                      "tests/test_core.py::test_level_defaults_to_info"],
    ),

    CodingCase(
        id="one-wrong-handler-among-many",
        kind="cross_file",
        tier="hard",
        prompt=("One of the handlers in pkg/handlers/ returns the wrong code. "
                "Each handler N should return code N * 100. Find it and fix it."),
        note=("Twelve near-identical files, one wrong. Reading them all costs "
              "most of the step budget and a large slice of the context window; "
              "`search` answers it in one call. This is the case that most "
              "directly measures whether the retrieval tooling is being used, "
              "and it is deliberately sized so that brute force is expensive "
              "rather than impossible."),
        files={
            "pkg/__init__.py": "",
            "pkg/handlers/__init__.py": "",
            **{f"pkg/handlers/h{i}.py": _handler(i, i * 100 if i != 7 else 650)
               for i in range(1, 13)},
            "tests/test_handlers.py": '''\
import importlib

import pytest

MODULES = [f"pkg.handlers.h{i}" for i in range(1, 13)]


def test_every_handler_is_importable():
    for name in MODULES:
        assert importlib.import_module(name) is not None


@pytest.mark.parametrize("index", range(1, 13))
def test_each_handler_returns_its_own_code(index):
    module = importlib.import_module(f"pkg.handlers.h{index}")
    assert module.handle({})["code"] == index * 100
''',
        },
        fail_to_pass=["tests/test_handlers.py::test_each_handler_returns_its_own_code[7]"],
        pass_to_pass=[
            "tests/test_handlers.py::test_every_handler_is_importable",
            "tests/test_handlers.py::test_each_handler_returns_its_own_code[1]",
            "tests/test_handlers.py::test_each_handler_returns_its_own_code[12]",
        ],
    ),
    # ------------------------------------------------------------ hard, added
    #
    # The first ten cases all reward *finding* the defect. These three reward
    # not making the obvious change, which is a different skill and the one a
    # step budget is most often spent failing at.
    CodingCase(
        id="the-obvious-fix-overshoots",
        kind="regression_trap",
        tier="hard",
        prompt=("`windows` never yields the final window -- see the failing "
                "test. Fix it."),
        note=("The bug is an off-by-one in a range bound, and the *obvious* "
              "correction overshoots: widening it to `range(len(items))` makes "
              "the failing test pass and starts emitting short windows at the "
              "end, which a different, currently-passing test forbids. Only "
              "`- size + 1` satisfies both. Verified to trap: the naive patch "
              "really does turn `test_every_window_is_full_length` red, which "
              "is the whole reason this case is in the suite."),
        files={
            "pkg/__init__.py": "",
            "pkg/windows.py": '''\
def windows(items, size):
    """Every contiguous run of exactly `size` items, left to right.

    windows([1, 2, 3, 4], 2) is [[1, 2], [2, 3], [3, 4]]. Every window has
    exactly `size` items; a sequence shorter than `size` has none.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    return [items[i:i + size] for i in range(len(items) - size)]
''',
            "tests/test_windows.py": '''\
import pytest

from pkg.windows import windows


def test_includes_the_final_window():
    assert windows([1, 2, 3, 4], 2) == [[1, 2], [2, 3], [3, 4]]


def test_every_window_is_full_length():
    for window in windows([1, 2, 3, 4, 5], 3):
        assert len(window) == 3


def test_too_short_a_sequence_has_no_windows():
    assert windows([1], 3) == []


def test_a_non_positive_size_is_an_error():
    with pytest.raises(ValueError):
        windows([1, 2], 0)
''',
        },
        fail_to_pass=["tests/test_windows.py::test_includes_the_final_window"],
        pass_to_pass=[
            "tests/test_windows.py::test_every_window_is_full_length",
            "tests/test_windows.py::test_too_short_a_sequence_has_no_windows",
            "tests/test_windows.py::test_a_non_positive_size_is_an_error",
        ],
    ),
    CodingCase(
        id="the-defect-is-not-where-it-fails",
        kind="cross_file",
        tier="hard",
        prompt=("Updating a record leaves `lookup` returning the old value. Fix it."),
        note=("The failing assertion is in `lookup`, and `lookup` is correct. "
              "The defect is that `update` never invalidates the cache, in "
              "another module. Patching where the failure surfaces produces a "
              "fix that passes the failing test by disabling the cache, which "
              "the performance test then catches. Measures whether the agent "
              "traces a cause or patches a symptom."),
        files={
            "pkg/__init__.py": "",
            "pkg/cache.py": '''\
class Cache:
    """A tiny memo table. `invalidate` must be called when a key changes."""

    def __init__(self):
        self._values = {}
        self.hits = 0
        self.misses = 0

    def get(self, key, compute):
        if key in self._values:
            self.hits += 1
            return self._values[key]
        self.misses += 1
        self._values[key] = compute()
        return self._values[key]

    def invalidate(self, key):
        self._values.pop(key, None)
''',
            "pkg/store.py": '''\
from pkg.cache import Cache


class Store:
    def __init__(self):
        self._rows = {}
        self.cache = Cache()

    def update(self, key, value):
        self._rows[key] = value

    def lookup(self, key):
        return self.cache.get(key, lambda: self._rows.get(key))
''',
            "tests/test_store.py": '''\
from pkg.store import Store


def test_lookup_sees_an_update():
    store = Store()
    store.update("a", 1)
    assert store.lookup("a") == 1
    store.update("a", 2)
    assert store.lookup("a") == 2


def test_repeated_lookups_are_cached():
    store = Store()
    store.update("a", 1)
    store.lookup("a")
    store.lookup("a")
    assert store.cache.hits >= 1


def test_missing_keys_are_none():
    assert Store().lookup("nope") is None
''',
        },
        fail_to_pass=["tests/test_store.py::test_lookup_sees_an_update"],
        pass_to_pass=[
            "tests/test_store.py::test_repeated_lookups_are_cached",
            "tests/test_store.py::test_missing_keys_are_none",
        ],
    ),
]


# -------------------------------------------------------------------- running

def materialise(case: CodingCase, root: Path) -> Path:
    """Write the fixture repository into `root`. Returns `root`."""
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in case.files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def restore_tests(case: CodingCase, root: Path) -> bool:
    """Put the fixture's test files back. Returns True if any had changed.

    Called after the agent and before grading, so nothing the agent did to
    `tests/` can affect the score. The return value is recorded as `tamper`:
    it changes no outcome, but an agent that keeps reaching for the test file
    is worth being able to see.
    """
    tampered = False
    expected = set(case.test_files)
    tests = root / "tests"
    if tests.exists():
        for path in sorted(item for item in tests.rglob("*") if item.is_file()):
            rel = path.relative_to(root).as_posix()
            if rel not in expected:
                tampered = True
                path.unlink()
    for rel in case.test_files:
        path = root / rel
        original = case.files[rel]
        try:
            current: Optional[str] = path.read_text(encoding="utf-8")
        except OSError:
            current = None
        if current != original:
            tampered = True
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(original, encoding="utf-8")
    return tampered


def reveal_held_out(case: CodingCase, root: Path) -> None:
    """Write the held-out tests, after the agent and before grading.

    Deliberately not part of `materialise`: the whole value of these tests is
    that they were never on disk while the agent was working, so it could not
    read them, could not edit them, and could not shape its solution to them.
    Writing them here is what makes `overfit` mean anything.
    """
    for rel, content in case.held_out.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _pytest(root: Path, node_ids: Sequence[str]) -> bool:
    """Run these node ids in one process. True only if every one passed."""
    try:
        # The sharpest instance of the leak this guards: these are tests an
        # agent wrote, run by the eval loop, and their stdout is graded.
        # Import pytest before putting the fixture root on sys.path, then add
        # that root explicitly so its product modules remain importable.
        bootstrap = (
            "import os,sys; "
            "os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'; "
            "import pytest; sys.path.insert(0,os.getcwd()); "
            "raise SystemExit(pytest.main())"
        )
        proc = sandbox.DEFAULT.run(
            [sys.executable, "-E", "-P", "-c", bootstrap, *node_ids, "-q",
             "--no-header", "-c", os.devnull, "--noconftest",
             "-p", "no:cacheprovider"],
            cwd=root, timeout=TEST_TIMEOUT)
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def run_tests(root: Path, node_ids: Sequence[str]) -> Dict[str, bool]:
    """Run the node ids and report pass/fail for each.

    **The batch first, one process per node only if it is not green.** Isolation
    is needed to *attribute* a failure -- a collection error in one file takes
    the whole batch down and cannot be told apart from a test that failed -- and
    a green batch has nothing to attribute. So the expensive path is paid only
    when something is wrong, which on a solved case is never.

    That is the dominant cost of the whole eval: it was one process per node,
    per case, per run, and `--repeat` multiplies it while held-out tests add a
    fourth set. The result is never worse than the old behaviour, because the
    fallback *is* the old behaviour.

    One deliberate difference: a test that passes only because another ran first
    is graded as passing here, where per-node isolation would have failed it.
    Order-dependence is a property of the fixture rather than of the agent, and
    a suite is how these tests are meant to be run -- but it is a difference,
    not an equivalence.
    """
    # Never call pytest with no arguments: it would collect the entire tree and
    # report on tests nobody asked about. The empty set has no results.
    if not node_ids:
        return {}
    if _pytest(root, node_ids):
        return {node: True for node in node_ids}
    return {node: _pytest(root, [node]) for node in node_ids}


def _action_verdict(expected_action: str, changed_files: Sequence[str],
                    agent_response: Optional[str]) -> bool:
    """Check the requested behavior independently of the test outcome."""
    if expected_action == "no_op":
        return not changed_files
    if expected_action == "edit_preserve":
        return bool(changed_files)
    if expected_action == "clarify":
        return not changed_files and bool(agent_response and "?" in agent_response)
    return expected_action == "edit"


def grade(case: CodingCase, root: Path, tampered: bool,
          agent_response: Optional[str] = None) -> Dict[str, Any]:
    """Run both expectation sets and decide. Assumes tests are restored."""
    fixed = run_tests(root, case.fail_to_pass)
    kept = run_tests(root, case.pass_to_pass)
    reveal_held_out(case, root)
    held = run_tests(root, case.held_out_pass)
    n_fixed = sum(fixed.values())
    n_kept = sum(kept.values())
    n_held = sum(held.values())
    changed_files = [
        rel for rel, original in case.files.items()
        if not rel.startswith("tests/") and (
            not (root / rel).is_file()
            or (root / rel).read_text(encoding="utf-8") != original)
    ]
    known = set(case.files)
    ignored = {".knossos", ".pytest_cache", "__pycache__", "target"}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if (rel.startswith("tests/") or rel in known
                or any(part in ignored for part in Path(rel).parts)):
            continue
        changed_files.append(rel)
    action_passed = _action_verdict(
        case.expected_action, changed_files, agent_response)
    return {
        "fixed": n_fixed,
        "fixed_total": len(case.fail_to_pass),
        "kept": n_kept,
        "kept_total": len(case.pass_to_pass),
        "held": n_held,
        "held_total": len(case.held_out_pass),
        # All three, not any. A change that fixes the bug and breaks the suite is
        # not partial success, it is a different failure -- and one that passes
        # everything it was shown while failing what it was not has not solved
        # the task, it has solved the assertions.
        #
        # Cases with no held-out tests are unaffected: an empty set sums to zero
        # against a length of zero, which is True.
        "passed": (not tampered
                   and action_passed
                   and n_fixed == len(case.fail_to_pass)
                   and n_kept == len(case.pass_to_pass)
                   and n_held == len(case.held_out_pass)),
        "tamper": tampered,
        "changed_files": changed_files,
        "action_passed": action_passed,
    }


#: Builds the thing under test. Given a workspace root, return an object with
#: `.run(prompt) -> outcome`, where outcome has `halt`, `succeeded`,
#: `steps_used` and `changed`. `Talos` satisfies this; so does a scripted stand-in.
AgentFactory = Callable[[Path], Any]


def _degradation(agent: Any) -> tuple:
    """The engine's give-up counters, or zeros for an agent without one.

    Reached through the agent rather than passed in, because `AgentFactory` is
    deliberately "anything with `.run(prompt)`" and requiring an engine here
    would close the seam that lets a foreign harness be graded by the same
    instrument. A scripted agent has no engine and truthfully reports nothing.
    """
    engine = getattr(agent, "engine", None)
    if engine is None:
        return (0, 0, 0.0)
    return (int(getattr(engine, "output_shrinks", 0) or 0),
            int(getattr(engine, "throttle_waits", 0) or 0),
            float(getattr(engine, "throttled_seconds", 0.0) or 0.0))


def run_case(case: CodingCase, make_agent: AgentFactory, root: Path,
             trace: Optional[Path] = None) -> CaseResult:
    """Materialise, run the agent, restore the tests, grade."""
    materialise(case, root)
    result = CaseResult(case_id=case.id, kind=case.kind, passed=False,
                        fixed_total=len(case.fail_to_pass),
                        kept_total=len(case.pass_to_pass),
                        held_total=len(case.held_out_pass))
    started = time.perf_counter()

    events: List[Dict[str, Any]] = []
    agent_response: Optional[str] = None
    try:
        agent = make_agent(root)
        # Snapshotted before the run and differenced after, so the counters are
        # per case rather than per suite. The engine is shared across cases, so
        # reading it raw would attribute every earlier case's backoff to this
        # one and make the last case in a throttled run look catastrophic.
        before = _degradation(agent)
        outcome = agent.run(case.prompt, on_event=_recorder(events)) \
            if _takes_events(agent) else agent.run(case.prompt)
        agent_response = str(getattr(outcome, "summary", "") or "") or None
        result.halt = getattr(getattr(outcome, "halt", None), "value", "") or ""
        result.harness_said_done = bool(getattr(outcome, "succeeded", False))
        result.steps_used = int(getattr(outcome, "steps_used", 0) or 0)
        result.tools_used = int(getattr(outcome, "tools_used", 0) or 0)
        result.changed_files = [Path(p).name for p in getattr(outcome, "changed", [])]
        # Duck-typed like the rest: an agent that reports neither leaves the
        # defaults, and `report` prints "not reported" rather than a zero.
        result.claim_source = str(getattr(outcome, "claim_source", "") or "")
        result.budget_kind = str(getattr(outcome, "budget_kind", "") or "")
        after = _degradation(agent)
        (result.output_shrinks, result.throttle_waits,
         result.throttled_seconds) = (after[0] - before[0], after[1] - before[1],
                                      after[2] - before[2])
        # Duck-typed for the same reason the fields are plain ints: an agent
        # that reports nothing simply leaves these at zero.
        usage = getattr(outcome, "usage", None)
        if usage is not None:
            result.prompt_tokens = int(getattr(usage, "prompt", 0) or 0)
            result.completion_tokens = int(getattr(usage, "completion", 0) or 0)
            result.cached_tokens = int(getattr(usage, "cached", 0) or 0)
            result.requests = int(getattr(usage, "requests", 0) or 0)
    except Exception as exc:                      # noqa: BLE001 - reported, not raised
        # A crashed run is a failed case, not a failed suite. The whole point
        # is to get a number across every case.
        result.error = f"{type(exc).__name__}: {exc}"

    # The engine reports a refused request as ordinary reply text, because that
    # is what reaches the user in a live session. Here it has to be told apart
    # from the model's own output, or a provider outage reads as a capability
    # score.
    result.api_errors = sum(
        1 for e in events
        if e.get("kind") == "text" and _API_FAILURE in (e.get("text") or ""))
    # Counted from the stream rather than asked of the outcome, because this is
    # the number that keeps `steps_used` honest: a harness batching five calls
    # into one turn is not five times more capable than one that serialises.
    # An agent that emits no events leaves it at zero, which `report` excludes
    # rather than averaging in.
    if not result.tools_used:
        result.tools_used = sum(1 for e in events if e.get("kind") == "tool")

    tampered = restore_tests(case, root)
    if not agent_response:
        agent_response = next(
            (str(e.get("text")) for e in reversed(events)
             if e.get("kind") == "text" and e.get("text")), None)
    verdict = grade(case, root, tampered, agent_response)
    result.passed = verdict["passed"]
    result.fixed = verdict["fixed"]
    result.kept = verdict["kept"]
    result.held = verdict["held"]
    result.tamper = verdict["tamper"]
    result.changed_files = verdict["changed_files"]
    result.action_passed = verdict["action_passed"]
    result.seconds = time.perf_counter() - started
    # Set here so a single-attempt run already carries it; `run_best_of`
    # overwrites it with the *first* sample's outcome when it keeps a later one.
    result.first_passed = result.passed

    if trace is not None:
        _write_trace(trace, case, result, events)
    return result


def run_best_of(case: CodingCase, make_agent: AgentFactory, root: Path,
                attempts: int, trace: Optional[Path] = None) -> CaseResult:
    """Run a case `attempts` times and keep the one the *harness* accepted.

    # Why this is worth having

    A verifier with no false passes is the precondition for test-time compute,
    and most harnesses do not have one: if the check can be fooled, sampling
    repeatedly until something passes is a machine for finding the route that
    fools it. This project's verifier is calibrated (`--agent vandal`), so
    selection is available to it in a way it is not to most.

    # The rule that keeps it honest

    **Selection reads only `harness_said_done`.** It never touches
    `fail_to_pass`, `kept`, or anything else the grader computes. Selecting on
    the graded outcome would be running the answer key in the loop, and the
    resulting number would describe an oracle that does not exist at inference
    time. The first accepted attempt wins; if none is accepted, the first
    attempt is reported, because "the harness accepted nothing" is the honest
    outcome rather than "pick the closest".

    # What this does to the verifier

    Sampling k times is *adversarial pressure* on the check in a way one attempt
    is not: it selects precisely for whatever the verifier blesses. Zero false
    passes single-shot therefore does not imply zero under selection, which is
    why `--best-of` is applied to the calibration agents too — the vandal gets k
    chances to find a route past the Oracle instead of one.
    """
    drawn: List[CaseResult] = []
    for attempt in range(attempts):
        # A fresh tree per attempt: sampling into a directory a previous attempt
        # already edited measures a sequence of repairs, not k independent
        # samples, and would let a later attempt inherit an earlier one's work.
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        per_attempt = (trace.with_suffix(f".{attempt + 1}.jsonl")
                       if trace is not None else None)
        drawn.append(run_case(case, make_agent, root, trace=per_attempt))
        if drawn[-1].harness_said_done:
            break

    # The kept sample is the first the harness accepted, or the first drawn when
    # it accepted none -- "nothing was accepted" being the honest outcome rather
    # than "pick whichever came closest", which would need the grader.
    kept = next((r for r in drawn if r.harness_said_done), drawn[0])
    # Every sample was paid for whether or not it was used, so the cost is the
    # sum over all of them. Charging only the winner would make best-of-k look
    # free, which is the one thing it certainly is not.
    kept.prompt_tokens = sum(r.prompt_tokens for r in drawn)
    kept.completion_tokens = sum(r.completion_tokens for r in drawn)
    kept.cached_tokens = sum(r.cached_tokens for r in drawn)
    kept.requests = sum(r.requests for r in drawn)
    kept.seconds = sum(r.seconds for r in drawn)
    # Samples *drawn*, not the index of the winner: a run that drew three and
    # accepted none has still spent three.
    kept.attempts = len(drawn)
    kept.first_passed = drawn[0].passed
    return kept


def run_suite(make_agent: AgentFactory, workdir: Path,
              cases: Optional[Sequence[CodingCase]] = None,
              trace_dir: Optional[Path] = None,
              on_result: Optional[Callable[[CaseResult], None]] = None,
              attempts: int = 1,
              ) -> List[CaseResult]:
    """Run every case in its own clean directory."""
    out: List[CaseResult] = []
    starved = 0
    for case in (cases if cases is not None else CODING_CASES):
        root = workdir / case.id
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        trace = (trace_dir / f"{case.id}.jsonl") if trace_dir else None
        result = (run_case(case, make_agent, root, trace=trace) if attempts <= 1
                  else run_best_of(case, make_agent, root, attempts, trace=trace))
        out.append(result)
        if on_result is not None:
            on_result(result)

        # Stop once the provider is plainly not answering. Every case after this
        # would pay the full retry cost to produce the same zero: measured at
        # 480s per case against an exhausted free tier, 82.7 minutes to learn
        # nothing. The cases already run are still reported -- they are what
        # shows *why* the run stopped.
        starved = starved + 1 if (result.api_errors and not result.passed) else 0
        if starved >= STARVED_CASES_BEFORE_ABORT:
            log(f"[codeval] {starved} consecutive cases were refused by the "
                f"provider; abandoning the remaining cases")
            break
    return out


# -------------------------------------------------------------------- tracing

def _takes_events(agent: Any) -> bool:
    try:
        import inspect
        return "on_event" in inspect.signature(agent.run).parameters
    except (TypeError, ValueError):
        return False


def _recorder(sink: List[Dict[str, Any]]) -> Callable[[Any], None]:
    def record(event: Any) -> None:
        entry: Dict[str, Any] = {"kind": getattr(event, "kind", "?"),
                                 "step": getattr(event, "step", 0)}
        text = getattr(event, "text", "")
        if text:
            entry["text"] = text[:2000]
        call = getattr(event, "call", None)
        if call is not None:
            entry["tool"] = call.name
            entry["args"] = {k: str(v)[:400] for k, v in call.args.items()}
        result = getattr(event, "result", None)
        if result is not None:
            entry["error"] = result.is_error
            entry["output"] = result.content[:2000]
        verdict = getattr(event, "verdict", None)
        if verdict is not None:
            entry["verdict"] = {"passed": verdict.passed, "summary": verdict.summary}
        sink.append(entry)
    return record


def _write_trace(path: Path, case: CodingCase, result: CaseResult,
                 events: Sequence[Dict[str, Any]]) -> None:
    """One JSONL file per case: the header, then every event.

    Python logs to stderr and keeps nothing, so a failed eval case could only
    be investigated by running it again and watching. This is the replayable
    record that makes a number actionable instead of merely discouraging.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        header = {"case": case.id, "kind": case.kind, "prompt": case.prompt,
                  "passed": result.passed, "halt": result.halt,
                  "harness_said_done": result.harness_said_done,
                  "fixed": f"{result.fixed}/{result.fixed_total}",
                  "kept": f"{result.kept}/{result.kept_total}",
                  "tamper": result.tamper, "error": result.error,
                  # Without these the header cannot distinguish a model that
                  # failed from a provider that was never reached: both leave
                  # `passed: false, halt: "stuck"`. The console report warns
                  # about it at run time, but the console is ephemeral and this
                  # file is what anyone aggregates months later -- so the
                  # distinction has to survive here or the number lies.
                  "api_errors": result.api_errors,
                  "unreachable": result.unreachable,
                  # The console warning about a throttled, shrunk run does not
                  # survive the terminal scrolling, and this file is what gets
                  # aggregated months later. Recording only `unreachable` meant
                  # a run that finished with a halved reply allowance and
                  # minutes of backoff aggregated as a clean capability number
                  # -- the same failure `trace_summary` exists to prevent, on a
                  # field it did not yet cover.
                  "output_shrinks": result.output_shrinks,
                  "throttle_waits": result.throttle_waits,
                  "throttled_seconds": round(result.throttled_seconds, 1),
                  "degraded": result.degraded,
                  # Named so `honest` is never read across harnesses as though
                  # a verifier's verdict and a process exit code were the same
                  # measurement.
                  "claim_source": result.claim_source,
                  "budget_kind": result.budget_kind,
                  "held": f"{result.held}/{result.held_total}",
                  "overfit": result.overfit,
                  "tools": result.tools_used,
                  # Recorded rather than left to be counted from the events
                  # below, because for an external harness there are no events
                  # to count. `--harness-cmd` runs write a header and nothing
                  # else, so a summariser that derives the turn count by
                  # tallying `step` events reads every foreign harness as zero
                  # -- which is indistinguishable from a harness that solved
                  # the case instantly. This is the one field that makes a
                  # harness-vs-harness comparison legible on the axis that
                  # still separates runs once `solved` saturates.
                  "steps": result.steps_used}
        fh.write(json.dumps(header) + "\n")
        for event in events:
            fh.write(json.dumps(event) + "\n")
