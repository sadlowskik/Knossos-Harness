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
