"""Tests for Lethe, bounded context with summarise-and-reset.

The load-bearing claims, each the inverse of a way compaction goes wrong:

  1. The opening survives. An agent that forgets the task will confidently do
     something else, and it will look like a model failure rather than a
     context one.
  2. The most recent turns survive verbatim. They hold the failure being
     repaired, and "the tests failed" is not enough to fix anything.
  3. A transcript inside budget is untouched, so compaction never costs
     anything when it is not needed.
  4. Elision is visible. The model must be able to tell recollection from
     record.

    pytest -q tests/test_lethe.py
"""
import pytest

from knossos.lethe import (
    CompactionResult,
    Lethe,
    estimate_tokens,
    extractive_summary,
)


def turn(label: str, size: int = 400) -> str:
    return f"## {label}\n" + ("filler prose that carries no structure. " * (size // 40))


@pytest.fixture()
def long_transcript():
    return [turn("Task: add a triple() function")] + [
        turn(f"Assistant step {i}") for i in range(1, 21)
    ]


# ------------------------------------------------------------------ no-op path

def test_a_transcript_within_budget_is_untouched(long_transcript):
    lethe = Lethe(max_tokens=10_000_000)
    result = lethe.compact(long_transcript)

    assert result.compacted is False
    assert result.transcript == long_transcript
    assert result.saved_tokens == 0
    assert "no compaction needed" in result.report()


def test_needs_compaction_matches_what_compact_does(long_transcript):
    lethe = Lethe(max_tokens=200, keep_recent=3)
    assert lethe.needs_compaction(long_transcript) is True
    assert lethe.compact(long_transcript).compacted is True


# --------------------------------------------------------------- what survives

def test_the_opening_is_pinned(long_transcript):
    """It states the task. Forgetting it is the worst possible loss."""
    result = Lethe(max_tokens=200, keep_recent=3).compact(long_transcript)
    assert result.transcript[0] == long_transcript[0]
    assert "add a triple() function" in result.transcript[0]


def test_recent_turns_are_kept_verbatim(long_transcript):
    """They hold the failure being repaired; a summary of it is not enough."""
    keep = 4
    result = Lethe(max_tokens=200, keep_recent=keep).compact(long_transcript)
    assert result.transcript[-keep:] == long_transcript[-keep:]


def test_the_elision_is_visible_to_the_model(long_transcript):
    result = Lethe(max_tokens=200, keep_recent=3).compact(long_transcript)
    joined = "\n".join(result.transcript)
    assert Lethe.MARKER in joined, "a silent gap is worse than a marked one"


def test_compaction_actually_reduces_size(long_transcript):
    result = Lethe(max_tokens=200, keep_recent=3).compact(long_transcript)
    assert result.tokens_after < result.tokens_before
    assert result.saved_tokens > 0
    assert result.summarised_turns > 0
    assert "saved" in result.report()


# ------------------------------------------------------------------ edge cases

def test_nothing_between_head_and_tail_is_left_alone():
    """Refusing to act beats gutting the turns the model needs.

    An over-budget prompt fails honestly; one missing its recent history fails
    mysteriously.
    """
    transcript = [turn("Task"), turn("A"), turn("B")]
    result = Lethe(max_tokens=1, keep_recent=2, pin_opening=1).compact(transcript)
    assert result.compacted is False
    assert result.transcript == transcript


def test_keep_recent_must_be_positive():
    with pytest.raises(ValueError):
        Lethe(keep_recent=0)


def test_one_enormous_entry_is_still_brought_within_budget():
    """Summarising the middle is not a bound on its own.

    `_render_results` packs a whole turn's tool output into one entry, so a
    single `read_file` of a large file lands in the kept tail intact and the
    result comes back over budget -- measured at 50 100 tokens against 24 000
    in a live run, which the engine reports as a context-length error rather
    than as anything Lethe did.
    """
    lethe = Lethe(max_tokens=2_000, keep_recent=3, pin_opening=1)
    transcript = ["## Task\ndo the thing"]
    transcript += [f"## Assistant\nstep {i}" for i in range(10)]
    transcript.append("## Tool results\n" + ("x" * 200_000))

    result = lethe.compact(transcript)

    assert result.compacted
    assert lethe.tokens(result.transcript) <= lethe.max_tokens, (
        f"still {lethe.tokens(result.transcript)} tokens against a "
        f"{lethe.max_tokens} budget")


def test_fitting_keeps_both_ends_of_what_it_elides():
    """The tail of a tool result is the verdict; the head says what ran."""
    lethe = Lethe(max_tokens=1_000, keep_recent=2, pin_opening=1)
    body = "RUNNING pytest\n" + ("filler\n" * 40_000) + "FAILED test_x - AssertionError"
    transcript = ["## Task\nrun the tests", "## Assistant\nok", "## Middle\nnoise",
                  "## Tool results\n" + body]

    kept = "\n".join(lethe.compact(transcript).transcript)

    assert "RUNNING pytest" in kept
    assert "FAILED test_x - AssertionError" in kept, "the verdict was dropped"
    assert "elided" in kept, "the gap must be visible, not silent"


def test_reported_tokens_after_reflect_the_fitting_pass():
    """`tokens_after` is what the caller uses to see any shortfall."""
    lethe = Lethe(max_tokens=2_000, keep_recent=3, pin_opening=1)
    transcript = ["## Task\ndo it"]
    transcript += [f"## Assistant\nstep {i}" for i in range(10)]
    transcript.append("## Tool results\n" + ("y" * 100_000))

    result = lethe.compact(transcript)

    assert result.tokens_after == lethe.tokens(result.transcript)


def test_fitting_stops_rather_than_gutting_the_transcript():
    """An impossible budget must terminate, not spin or empty the transcript."""
    lethe = Lethe(max_tokens=1, keep_recent=2, pin_opening=1)
    transcript = ["## Task\nx", "## A\ny", "## B\nz", "## C\nw"]

    result = lethe.compact(transcript)

    assert result.transcript, "the transcript must not be emptied"
    assert result.tokens_after > 0


def test_compaction_is_idempotent_once_within_budget(long_transcript):
    lethe = Lethe(max_tokens=400, keep_recent=3)
    once = lethe.compact(long_transcript)
    twice = lethe.compact(once.transcript)

    if not twice.compacted:
        assert twice.transcript == once.transcript
    # Either way, summaries must not nest into a chain of summaries.
    assert "\n".join(twice.transcript).count(Lethe.MARKER) <= 1


def test_an_empty_transcript_is_handled():
    result = Lethe(max_tokens=1).compact([])
    assert result.compacted is False
    assert result.transcript == []


# ---------------------------------------------------------------- summarising

def test_the_summary_keeps_structure_and_drops_prose():
    turns = [
        "## Assistant\nI think the best approach here is probably to consider",
        "<call>write_file</call>\nfile: daedalus/moe.py",
        "## Verification\nFAILED tests/test_moe.py::test_router",
    ]
    summary = extractive_summary(turns)

    assert "daedalus/moe.py" in summary, "which files were touched is not recoverable"
    assert "FAILED" in summary, "the outcome is the point"
    assert "I think the best approach" not in summary


@pytest.mark.parametrize("line,expected", [
    ("file: daedalus/moe.py", "daedalus/moe.py"),
    ("daedalus/moe.py at the very start", "daedalus/moe.py"),
    ('opened "pkg/thing.py" for writing', "pkg/thing.py"),
    ("(src/lib.rs)", "src/lib.rs"),
    ["edited pkg\\win.py on windows", "pkg\\win.py"],
    ("see config.toml, then README.md", "config.toml"),
    ("FAILED tests/test_moe.py::test_router", "tests/test_moe.py"),
])
def test_paths_are_still_found_after_the_delimiter_change(line, expected):
    """The regex gained a required leading delimiter for speed.

    That is only acceptable if it still finds the paths that matter, so the
    delimiters a path actually appears after are pinned here.
    """
    assert expected in extractive_summary([f"## Step\n{line}"])


def test_a_pathological_line_does_not_stall_the_summariser():
    """A performance regression test, because no correctness test caught this.

    The original pattern retried `[\\w./\\\\-]+` at every offset of a long run
    with no dot in it, which is quadratic. Measured at 8.6 ms for one such line,
    1 240 ms per executor step, ~15 s across a 12-step run -- all inside
    `findall`, and entirely invisible to a suite that only checked output.

    The bound is generous on purpose: this asserts "not quadratic", not a
    particular machine's speed.
    """
    import time

    turns = ["## Tool results\n" + ("x" * 20_000)] * 5

    started = time.perf_counter()
    extractive_summary(turns)
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, f"summarising took {elapsed:.2f}s; the regex is backtracking"


def test_a_long_data_line_is_not_scanned_for_paths():
    """Structural lines are short; something this long is data."""
    blob = "y" * 5_000 + " embedded/inside.py"

    assert "embedded/inside.py" not in extractive_summary([f"## Step\n{blob}"])


def test_the_summary_is_capped():
    turns = ["## Step\nerror: something failed\n" * 500]
    summary = extractive_summary(turns, max_chars=300)
    assert len(summary) <= 302          # cap plus the elision marker


def test_a_summary_with_no_structure_says_so():
    assert "no recoverable structure" in extractive_summary(["just prose here"])


def test_a_custom_summariser_is_used(long_transcript):
    """An engine-backed summariser plugs in where a model call is worth it."""
    calls = []

    def summarise(turns):
        calls.append(len(turns))
        return "MODEL SUMMARY"

    result = Lethe(max_tokens=200, keep_recent=3,
                   summariser=summarise).compact(long_transcript)

    assert calls, "the custom summariser should have been called"
    assert "MODEL SUMMARY" in "\n".join(result.transcript)


def test_token_estimate_scales_with_length():
    assert estimate_tokens("") >= 1
    assert estimate_tokens("x" * 400) > estimate_tokens("x" * 40)
