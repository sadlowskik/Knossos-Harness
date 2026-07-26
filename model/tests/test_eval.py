"""Tests for the eval grader.

An eval you have not tested reports numbers you cannot trust, and a grading bug
is worse than no eval -- it produces a confident wrong measurement that you then
make decisions against. The claims pinned here:

  1. A provider error (429, unreachable host) is never scored as a wrong answer.
     Free tiers fail partway through constantly; if that counted as a miss, the
     harness would look worse the more you tested it.
  2. A known fabrication zeroes a case even when every other signal is good.
  3. Negative cases pass only on admitting absence.

    pytest -q tests/test_eval.py
"""
import pytest

from harness.eval import AnswerGrade, grade_answer, grade_retrieval, report_answers
from harness.evalset import Case, CASES


def case(**kw):
    base = dict(id="t", prompt="p", kind="repo_specific")
    base.update(kw)
    return Case(**base)


# -------------------------------------------------------------------- errors

@pytest.mark.parametrize("reply", [
    "Request failed: HTTP 429 — rate limit or free-tier daily cap reached.",
    "Could not reach http://localhost:8000/v1 (connection refused).",
    "   ",
])
def test_provider_failures_are_not_scored_as_wrong(reply):
    g = grade_answer(case(expect_files=["a.py"], expect_terms=[["x"]]), reply, "harness")
    assert g.error is not None
    assert g.score == 0.0 and g.fabricated is False


def test_a_real_answer_is_not_mistaken_for_an_error():
    g = grade_answer(case(expect_terms=[["ok"]]), "everything is ok here", "raw")
    assert g.error is None and g.score == 1.0


# ------------------------------------------------------------------- scoring

def test_citation_and_terms_weigh_equally():
    c = case(expect_files=["daedalus/moe.py"], expect_terms=[["router"]])
    both = grade_answer(c, "the router lives in daedalus/moe.py", "harness")
    terms_only = grade_answer(c, "the router does the routing", "raw")
    cite_only = grade_answer(c, "see daedalus/moe.py", "harness")
    assert both.score == 1.0
    assert terms_only.score == pytest.approx(0.5)
    assert cite_only.score == pytest.approx(0.5)


def test_citation_matches_on_basename_too():
    c = case(expect_files=["daedalus/moe.py"])
    assert grade_answer(c, "it is in moe.py", "harness").cited is True


def test_windows_separators_still_match():
    c = case(expect_files=["daedalus/moe.py"])
    assert grade_answer(c, r"see daedalus\moe.py", "harness").cited is True


def test_term_groups_are_any_within_and_all_across():
    c = case(expect_terms=[["alpha", "beta"], ["gamma"]])
    assert grade_answer(c, "alpha and gamma", "raw").terms == 1.0
    assert grade_answer(c, "beta and gamma", "raw").terms == 1.0
    assert grade_answer(c, "alpha only", "raw").terms == pytest.approx(0.5)


def test_fabrication_zeroes_an_otherwise_perfect_answer():
    c = case(expect_files=["daedalus/ariadne.py"],
             expect_terms=[["pondernet"]],
             forbid_terms=["so et al"])
    g = grade_answer(c, "PonderNet, from daedalus/ariadne.py, by So et al. 2017",
                     "harness")
    assert g.cited is True and g.terms == 1.0
    assert g.fabricated is True and g.score == 0.0


def test_cases_without_expected_files_score_on_terms_alone():
    g = grade_answer(case(kind="general", expect_terms=[["relative"]]),
                     "it encodes relative positions", "raw")
    assert g.cited is None and g.score == 1.0


# ------------------------------------------------------------------ negatives

def test_negative_case_passes_only_on_admitting_absence():
    c = case(kind="negative")
    assert grade_answer(c, "There is no retry logic in this codebase.", "harness").score == 1.0
    assert grade_answer(c, "Retries are handled in daedalus/net.py.", "harness").score == 0.0


def test_negative_fabrication_is_flagged():
    g = grade_answer(case(kind="negative"), "It uses exponential backoff.", "raw")
    assert g.fabricated is True and g.honest is False


# ----------------------------------------------------------------- retrieval

def test_retrieval_rank_is_the_first_expected_file():
    c = case(expect_files=["daedalus/moe.py"])
    g = grade_retrieval(c, ["README.md", "daedalus/naiads.py", "daedalus/moe.py"])
    assert g.hit is True and g.rank == 3


def test_retrieval_miss_has_no_rank():
    g = grade_retrieval(case(expect_files=["daedalus/moe.py"]), ["README.md"])
    assert g.hit is False and g.rank is None


def test_retrieval_dedupes_repeated_files():
    """Several excerpts from one file are one retrieved file, not three."""
    c = case(expect_files=["daedalus/moe.py"])
    g = grade_retrieval(c, ["a.py", "a.py", "daedalus/moe.py"])
    assert g.rank == 2


# -------------------------------------------------------------------- pairing

def _grade(case_id, kind, condition, score, error=None):
    return AnswerGrade(case_id, kind, condition, None, score, False, None, score,
                       error=error)


def test_unpaired_cases_are_excluded_from_the_by_kind_block(capsys):
    """A case only one condition answered must not enter either mean.

    Observed: gemma4:e4b exhausted its token budget thinking on 4 of 10 harness
    cases. Averaging all 10 raw scores against the 6 surviving harness scores
    compares different questions and reports the difference as an effect.
    """
    grades = [
        _grade("a", "repo_specific", "raw", 0.0),
        _grade("a", "repo_specific", "harness", 1.0),
        # `b` failed in the harness arm: neither column may count it
        _grade("b", "repo_specific", "raw", 1.0),
        _grade("b", "repo_specific", "harness", 0.0, error="empty reply"),
    ]
    report_answers(grades, ["raw", "harness"])
    out = capsys.readouterr().out

    line = next(l for l in out.splitlines()
                if l.strip().startswith("repo_specific") and "n=" in l)
    assert "n=1/" in line, line
    assert "0.00" in line and "1.00" in line     # case `a` only
    assert "+1.00" in line
    assert "excluded from the paired block" in out
    assert " b" in out.split("excluded from the paired block")[1]


def test_pairing_survives_repeats(capsys):
    grades = [
        _grade("a", "repo_specific", "raw", 0.0),
        _grade("a", "repo_specific", "raw", 0.5),
        _grade("a", "repo_specific", "harness", 1.0),
        _grade("a", "repo_specific", "harness", 1.0),
    ]
    report_answers(grades, ["raw", "harness"])
    line = next(l for l in capsys.readouterr().out.splitlines()
                if l.strip().startswith("repo_specific") and "n=" in l)
    assert "0.25" in line and "1.00" in line     # per-case mean, then across cases


# ------------------------------------------------------------------- the set

def test_evalset_is_well_formed():
    assert len({c.id for c in CASES}) == len(CASES), "duplicate case id"
    for c in CASES:
        assert c.kind in ("repo_specific", "general", "negative"), c.id
        if c.kind == "negative":
            assert not c.expect_files and not c.expect_terms, c.id
        else:
            assert c.expect_terms, f"{c.id} would pass trivially"


def test_evalset_covers_both_sides_of_the_comparison():
    kinds = {c.kind for c in CASES}
    assert {"repo_specific", "general", "negative"} <= kinds, (
        "the raw-vs-harness comparison is uninterpretable without general "
        "controls and negative cases")
