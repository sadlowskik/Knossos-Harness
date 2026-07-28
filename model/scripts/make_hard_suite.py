"""Generate a harder external suite for `coding_eval --cases`.

# Why this exists

The built-in set is saturated. Measured on 2026-07-28: Knossos on Gemini 3.1
Pro scored 12/12, Claude Code on Sonnet 5 scored 12/12, and Knossos on Gemini
3.1 Flash-Lite *also* solved 12/12 -- a benchmark that a lite model aces cannot
rank a frontier one. `codeval.py` predicted this in its own docstring: "six
consecutive perfect runs is a statement about the fixtures, not the agent."

# What makes these harder

Not size. The one signal that survived saturation was `honest` -- Flash-Lite
solved every case but ran two of them to the step ceiling without recognising
it was finished. So these cases target the *completion decision* rather than
the edit:

  * The obvious fix passes the named test and breaks a different one.
  * Two defects mask each other, so fixing one looks like failure.
  * The contract is stated in a docstring and enforced only from another file.
  * The symptom is wrong values rather than an exception.
  * The first call is correct and the second is not.

Each is a bug a competent engineer writes on a tired afternoon, not a puzzle.
That matters: a fixture that is merely obscure measures patience, and a
benchmark of patience does not transfer to real work.

# Calibration

`--agent oracle` cannot check an external suite (its solutions are keyed to the
built-in ids), so `--check` verifies the two properties that make a case
gradeable at all: every `fail_to_pass` test must fail on the untouched fixture,
and every `pass_to_pass` test must pass. A case that violates either is worse
than absent -- the first is unwinnable, the second is a free point.

    python -m scripts.make_hard_suite            # write the suite
    python -m scripts.make_hard_suite --check    # write, then verify the states
    python -m scripts.coding_eval --cases fixtures/hard_suite.json --engine api ...
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CASES = []


def case(**kw):
    CASES.append(kw)


# ---------------------------------------------------------------- 1. aliasing
#
# The classic mutable default. Each call is correct in isolation, so a test
# suite that exercises one call per test never sees it. The completion trap:
# the agent fixes the reported symptom by clearing the list at the top, which
# also passes -- and silently breaks the caller that relies on accumulating
# across an explicitly passed-in list.
case(
    id="the-second-call-remembers-the-first",
    kind="bugfix",
    tier="hard",
    prompt=("collect_errors() returns entries from a previous call when it is "
            "called more than once. Fix it."),
    files={
        "pkg/__init__.py": "",
        "pkg/validate.py": '''
def collect_errors(fields, into=[]):
    """Append a message for each empty field and return the accumulator.

    Callers may pass their own list to gather errors across several calls.
    """
    for name, value in fields.items():
        if not value:
            into.append(f"{name} is required")
    return into
''',
        "tests/__init__.py": "",
        "tests/test_validate.py": '''
from pkg.validate import collect_errors


def test_a_second_call_does_not_see_the_first():
    collect_errors({"a": ""})
    assert collect_errors({"b": ""}) == ["b is required"]


def test_one_call_still_works():
    assert collect_errors({"a": ""}) == ["a is required"]
''',
        "tests/test_accumulating.py": '''
from pkg.validate import collect_errors


def test_an_explicit_accumulator_is_still_appended_to():
    """The documented use: gather errors across several calls."""
    bucket = ["existing"]
    collect_errors({"a": ""}, bucket)
    collect_errors({"b": ""}, bucket)
    assert bucket == ["existing", "a is required", "b is required"]
''',
    },
    fail_to_pass=["tests/test_validate.py::test_a_second_call_does_not_see_the_first"],
    pass_to_pass=["tests/test_validate.py::test_one_call_still_works",
                  "tests/test_accumulating.py::test_an_explicit_accumulator_is_still_appended_to"],
    note="Clearing `into` at the top also passes the named test and breaks the caller.",
)

# ------------------------------------------------------------- 2. the easy fix
#
# `set()` is the reflexive dedupe and it is wrong here. The failing test says
# nothing about order; the order requirement lives in a different file that is
# already passing, so an agent that stops at the named test regresses it.
case(
    id="dedupe-must-not-reorder",
    kind="regression_trap",
    tier="hard",
    prompt="unique() keeps duplicates. Make it return each item once.",
    files={
        "pkg/__init__.py": "",
        "pkg/seq.py": '''
def unique(items):
    """Return the items with duplicates removed, in first-seen order."""
    return [item for item in items]
''',
        "tests/__init__.py": "",
        "tests/test_seq.py": '''
from pkg.seq import unique


def test_duplicates_are_removed():
    assert len(unique([3, 1, 3, 2, 1])) == 3
''',
        # These pass on the untouched fixture (it returns its input unchanged)
        # and pass again under a correct first-seen fix -- but not under
        # `list(set(items))`, which reorders. Integers, deliberately: CPython
        # iterates a small-int set in value order every time, while string set
        # order moves with hash randomisation. A trap that only springs on some
        # runs is worse than no trap.
        "tests/test_ordering.py": '''
from pkg.seq import unique


def test_order_is_preserved_when_there_are_no_duplicates():
    """Downstream rendering depends on this; see the docstring."""
    assert unique([3, 1, 2]) == [3, 1, 2]


def test_a_longer_run_keeps_its_order():
    assert unique([5, 3, 9, 1]) == [5, 3, 9, 1]
''',
    },
    fail_to_pass=["tests/test_seq.py::test_duplicates_are_removed"],
    pass_to_pass=["tests/test_ordering.py::test_order_is_preserved_when_there_are_no_duplicates",
                  "tests/test_ordering.py::test_a_longer_run_keeps_its_order"],
    note="`list(set(items))` passes the named test and breaks two already-passing ones.",
)

# --------------------------------------------------- 3. two defects, one symptom
#
# Fixing either alone leaves the test red, which reads as "my fix did not
# work" and invites reverting a correct change. This is the case that punishes
# a loop that treats a still-failing test as evidence the last edit was wrong.
case(
    id="one-fix-is-not-enough",
    kind="bugfix",
    tier="hard",
    prompt=("parse_row() returns the wrong fields for a quoted CSV row. "
            "Make it handle the sample correctly."),
    files={
        "pkg/__init__.py": "",
        "pkg/csvparse.py": '''
def parse_row(line):
    """Split one CSV row on commas, honouring double quotes.

    A quoted field may contain commas. Quotes are stripped from the result.
    """
    out, field, quoted = [], "", False
    for char in line:
        if char == "'":                 # defect 1: wrong quote character
            quoted = not quoted
        elif char == "," and quoted:    # defect 2: inverted condition
            out.append(field)
            field = ""
        else:
            field += char
    out.append(field)
    return out
''',
        "tests/__init__.py": "",
        "tests/test_csvparse.py": '''
from pkg.csvparse import parse_row


def test_a_quoted_field_may_contain_a_comma():
    assert parse_row('a,"b,c",d') == ["a", "b,c", "d"]


def test_a_plain_row_still_splits():
    assert parse_row("a,b,c") == ["a", "b", "c"]
''',
    },
    fail_to_pass=["tests/test_csvparse.py::test_a_quoted_field_may_contain_a_comma",
                  "tests/test_csvparse.py::test_a_plain_row_still_splits"],
    pass_to_pass=[],
    note="Two independent defects; fixing one leaves both tests red.",
)

# ------------------------------------------------------- 4. the stated contract
#
# The "do not mutate" rule is in the docstring and enforced only from a second
# file. The shortest fix mutates in place and passes the named test.
case(
    id="the-contract-is-in-the-docstring",
    kind="cross_file",
    tier="hard",
    prompt="normalise_tags() lowercases but leaves duplicates. Remove them.",
    files={
        "pkg/__init__.py": "",
        "pkg/tags.py": '''
def normalise_tags(tags):
    """Return a new lowercased, de-duplicated list, in first-seen order.

    Never mutates the argument: callers reuse the list they passed in.
    """
    return [t.lower() for t in tags]
''',
        "pkg/report.py": '''
from pkg.tags import normalise_tags


def summarise(tags):
    """Report on the caller's tags without disturbing them."""
    return {"normalised": normalise_tags(tags), "original_count": len(tags)}
''',
        "tests/__init__.py": "",
        "tests/test_tags.py": '''
from pkg.tags import normalise_tags


def test_duplicates_are_removed():
    assert normalise_tags(["A", "b", "a"]) == ["a", "b"]
''',
        "tests/test_report.py": '''
from pkg.report import summarise


def test_the_callers_list_is_left_alone():
    tags = ["A", "b", "a"]
    result = summarise(tags)
    assert result["original_count"] == 3
    assert tags == ["A", "b", "a"], "summarise must not mutate its argument"
''',
    },
    fail_to_pass=["tests/test_tags.py::test_duplicates_are_removed"],
    pass_to_pass=["tests/test_report.py::test_the_callers_list_is_left_alone"],
    note="An in-place fix passes the named test and violates the documented contract.",
)

# ------------------------------------------------------- 5. no exception at all
#
# Nothing raises. The totals are simply wrong, and only for inputs that need
# more than two decimal places -- so a spot check with clean numbers agrees.
case(
    id="wrong-answers-without-an-error",
    kind="bugfix",
    tier="hard",
    prompt=("running_total() disagrees with the expected totals on the sample "
            "invoice. Fix it."),
    files={
        "pkg/__init__.py": "",
        "pkg/money.py": '''
def running_total(amounts):
    """Cumulative totals in cents, rounded to whole cents at each step.

    Amounts arrive as floating point dollars.
    """
    total, out = 0, []
    for amount in amounts:
        total += int(amount * 100)          # truncates instead of rounding
        out.append(total)
    return out
''',
        "tests/__init__.py": "",
        "tests/test_money.py": '''
from pkg.money import running_total


def test_fractional_cents_round_rather_than_truncate():
    assert running_total([0.29, 0.29, 0.29]) == [29, 58, 87]


def test_whole_amounts_are_unaffected():
    assert running_total([1.00, 2.00]) == [100, 300]
''',
    },
    fail_to_pass=["tests/test_money.py::test_fractional_cents_round_rather_than_truncate"],
    pass_to_pass=["tests/test_money.py::test_whole_amounts_are_unaffected"],
    note="No exception; 0.29*100 is 28.999... and int() truncates to 28.",
)

# --------------------------------------------------------- 6. correct once only
#
# The first call is right, which is what a quick manual check exercises. The
# cache key ignores an argument, so the second call with different arguments
# returns the first answer.
case(
    id="right-the-first-time-only",
    kind="bugfix",
    tier="hard",
    prompt=("price_for() returns the wrong currency after the first lookup. "
            "Fix it."),
    files={
        "pkg/__init__.py": "",
        "pkg/pricing.py": '''
_RATES = {"USD": 1.0, "EUR": 0.9, "GBP": 0.8}
_cache = {}


def price_for(sku, currency):
    """Price of `sku` in `currency`, memoised."""
    if sku in _cache:                       # key ignores currency
        return _cache[sku]
    price = round(100 * _RATES[currency], 2)
    _cache[sku] = price
    return price
''',
        "tests/__init__.py": "",
        "tests/test_pricing.py": '''
from pkg.pricing import price_for


def test_a_second_currency_is_not_served_from_the_first():
    assert price_for("widget", "USD") == 100.0
    assert price_for("widget", "EUR") == 90.0


def test_the_first_lookup_is_still_right():
    assert price_for("gadget", "GBP") == 80.0
''',
    },
    fail_to_pass=["tests/test_pricing.py::test_a_second_currency_is_not_served_from_the_first"],
    pass_to_pass=["tests/test_pricing.py::test_the_first_lookup_is_still_right"],
    note="Module-level cache; the agent must notice the key, not add a clear().",
)


def check(cases) -> int:
    """Verify every case starts from the state its grade assumes."""
    from knossos.codeval import CodingCase, materialise, run_tests

    total_bad = 0
    for raw in cases:
        c = CodingCase(id=raw["id"], prompt=raw["prompt"], files=raw["files"],
                       fail_to_pass=raw["fail_to_pass"],
                       pass_to_pass=raw.get("pass_to_pass", ()),
                       kind=raw.get("kind", "bugfix"), tier=raw.get("tier", "core"))
        root = Path(tempfile.mkdtemp(prefix=f"check-{c.id}-"))
        materialise(c, root)

        bad = 0
        for node, passing in run_tests(root, c.fail_to_pass).items():
            if passing:
                print(f"  !! {c.id}: {node} already passes — the case is a free point")
                bad += 1
        for node, passing in run_tests(root, c.pass_to_pass).items():
            if not passing:
                print(f"  !! {c.id}: {node} already fails — the case is unwinnable")
                bad += 1
        # Per case, not unconditionally: printing `ok` after `!!` lines for the
        # same case reported it both broken and fine, which is how a calibration
        # failure gets skimmed past.
        if not bad:
            print(f"  ok  {c.id:38} "
                  f"{len(c.fail_to_pass)} to fix, {len(c.pass_to_pass)} to keep")
        total_bad += bad
    return total_bad


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="fixtures/hard_suite.json", type=Path)
    parser.add_argument("--check", action="store_true",
                        help="verify each fixture starts red where it must")
    args = parser.parse_args(argv)

    ids = [c["id"] for c in CASES]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate case ids")

    # Checked before writing. Writing first left an ungradeable suite on disk
    # after a failed check -- and a JSON file that exists is a file someone
    # runs.
    if args.check:
        bad = check(CASES)
        if bad:
            print(f"\n  !! {bad} problem(s): not written. The suite is not "
                  f"gradeable as written.\n")
            return 1
        print("\n  every case starts red where it must and green where it must.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(CASES, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {len(CASES)} case(s) to {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
