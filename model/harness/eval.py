"""Measure what the harness is worth: the same model, with and without it.

Two evaluations, kept separate because they fail for different reasons.

    retrieval   Did Argus surface the right files? No model, no network, fully
                deterministic. Run this after any ranking change -- it is the
                cheap signal, and a ranking regression shows up here first.

    answer      Does the model's reply cite the right file and contain the right
                facts? Run twice per case:
                    raw      the question alone
                    harness  the question plus what Argus retrieved
                The delta between those two is the only number that actually
                says whether the harness earns its keep.

The interesting result is not "harness scores higher". It is *where* it scores
higher. `repo_specific` cases should show a large gap and `general` cases should
show none, because a retrieval system cannot help a model recall what RoPE is. If
the general gap is large too, suspect the grader before believing the harness.

    python -m harness.eval --mode retrieval
    python -m harness.eval --mode answer --repeat 3

Rate limits are a real hazard here: a free tier will start refusing partway
through and, ungraded, those refusals look exactly like wrong answers. Provider
errors are counted separately and never scored.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .argus import Argus, render
from .engine import Engine, OpenAICompatEngine, Thought
from .evalset import CASES, Case

__all__ = ["grade_answer", "grade_retrieval", "AnswerGrade", "RetrievalGrade"]

#: Phrases that count as admitting absence, for `negative` cases. Crude by
#: construction -- a model can decline in ways this list does not contain, which
#: makes the negative score a *lower bound* on honesty, not a measurement of it.
ABSENCE_MARKERS = (
    "not present", "does not", "doesn't", "no such", "not in the", "not found",
    "could not find", "couldn't find", "no evidence", "there is no", "there's no",
    "not appear", "no retry", "no database", "no authentication", "no auth",
    "not contain", "nothing in", "do not contain", "isn't any", "is not any",
    "not implemented", "no implementation", "unable to find", "not included",
)

#: Text the engine yields when the *provider* failed. Never graded as an answer.
ERROR_MARKERS = ("request failed: http", "could not reach")


@dataclass
class RetrievalGrade:
    case_id: str
    expected: List[str]
    got: List[str]
    hit: bool
    rank: Optional[int]                 # 1-indexed position of the first expected file


@dataclass
class AnswerGrade:
    case_id: str
    kind: str
    condition: str
    cited: Optional[bool]               # None when the case expects no files
    terms: float                        # 0..1 fraction of term groups satisfied
    fabricated: bool
    honest: Optional[bool]              # negative cases only
    score: float                        # 0..1 headline number
    error: Optional[str] = None
    reply: str = ""


def _norm(text: str) -> str:
    return text.lower().replace("\\", "/")


def grade_retrieval(case: Case, hit_files: Sequence[str]) -> RetrievalGrade:
    """Did the expected files appear, and how far down?"""
    got = list(dict.fromkeys(hit_files))
    rank = None
    for i, path in enumerate(got, 1):
        if any(_norm(path).endswith(_norm(exp)) for exp in case.expect_files):
            rank = i
            break
    return RetrievalGrade(case.id, list(case.expect_files), got,
                          hit=rank is not None, rank=rank)


def grade_answer(case: Case, reply: str, condition: str) -> AnswerGrade:
    """Score one reply. Provider errors short-circuit to `error`, unscored."""
    low = _norm(reply)

    for marker in ERROR_MARKERS:
        if marker in low:
            return AnswerGrade(case.id, case.kind, condition, None, 0.0, False,
                               None, 0.0, error=reply.strip()[:120], reply=reply)
    if not reply.strip():
        return AnswerGrade(case.id, case.kind, condition, None, 0.0, False, None,
                           0.0, error="empty reply", reply=reply)

    if case.kind == "negative":
        honest = any(m in low for m in ABSENCE_MARKERS)
        return AnswerGrade(case.id, case.kind, condition, None, 0.0, not honest,
                           honest, 1.0 if honest else 0.0, reply=reply)

    # ANY-of within a group, ALL-of across groups.
    groups = case.expect_terms or ()
    satisfied = sum(1 for group in groups if any(_norm(t) in low for t in group))
    terms = satisfied / len(groups) if groups else 1.0

    cited = None
    if case.expect_files:
        cited = any(_norm(f) in low or _norm(Path(f).name) in low
                    for f in case.expect_files)

    fabricated = any(_norm(f) in low for f in case.forbid_terms)

    # Citation and content weigh equally; a fabricated citation zeroes the case,
    # because a confident wrong reference is worse than no reference.
    if cited is None:
        score = terms
    else:
        score = 0.5 * terms + 0.5 * (1.0 if cited else 0.0)
    if fabricated:
        score = 0.0

    return AnswerGrade(case.id, case.kind, condition, cited, terms, fabricated,
                       None, score, reply=reply)


# --------------------------------------------------------------------- running

def report_gate(argus: Argus, threshold: float, budget: int, hops: int) -> float:
    """Score the gate against the labelled eval set. No model, no network.

    Ground truth: `general` cases should be skipped, everything else injected --
    including `negative`, which are questions about this repo that happen to have
    no answer in it. Sending those through retrieval is what made gemma say
    "there is no retry logic" instead of inventing one.
    """
    from .gate import RetrievalGate

    gate = RetrievalGate(argus, threshold=threshold)
    print(f"\nGATE  (threshold {threshold:.2f}; no model)")
    print(f"  {'case':<24} {'want':<8} {'got':<8} {'conf':<6} why")
    print("  " + "-" * 88)

    correct = 0
    for case in CASES:
        want = case.kind != "general"
        hits = argus.retrieve(case.prompt, budget=budget, hops=hops)
        decision = gate.decide(case.prompt, hits)
        ok = decision.inject == want
        correct += ok
        print(f"  {case.id:<24} {'inject' if want else 'skip':<8} "
              f"{'inject' if decision.inject else 'skip':<8} "
              f"{decision.confidence:<6.2f} "
              f"{'' if ok else 'WRONG - '}{'; '.join(decision.reasons)[:52]}")

    accuracy = correct / len(CASES) if CASES else 0.0
    print(f"\n  accuracy {correct}/{len(CASES)} ({accuracy:.0%})")
    skips = [c for c in CASES if c.kind == "general"]
    caught = sum(1 for c in skips
                 if not gate.decide(c.prompt,
                                    argus.retrieve(c.prompt, budget=budget,
                                                   hops=hops)).inject)
    print(f"  general questions correctly skipped {caught}/{len(skips)}")
    print("  A wrong skip costs a repo answer; a wrong inject costs a general "
          "one.\n  On a small model both are real -- that is why this exists.")
    return accuracy


def run_retrieval(argus: Argus, budget: int, hops: int) -> List[RetrievalGrade]:
    grades = []
    for case in CASES:
        if not case.expect_files:
            continue
        hits = argus.retrieve(case.prompt, budget=budget, hops=hops)
        grades.append(grade_retrieval(case, [h.file for h in hits]))
    return grades


def _collect(engine: Engine, prompt: str, context) -> tuple[str, int]:
    """Drain a generation. Returns the reply and how much was spent thinking.

    Reasoning is not graded -- the answer is. But the thinking length has to come
    back, because "no reply at all" and "no reply because the whole budget went
    to <think>" are different failures and only one of them is about the model.
    """
    out: List[str] = []
    thought = 0
    for chunk in engine.generate(prompt, context, lambda: False):
        if isinstance(chunk, Thought):
            thought += len(str(chunk))
        else:
            out.append(str(chunk))
    return "".join(out), thought


def run_answers(engine: Engine, argus: Argus, conditions: Sequence[str],
                budget: int, hops: int, repeat: int, sleep: float,
                verbose: bool) -> List[AnswerGrade]:
    grades: List[AnswerGrade] = []
    total = len(CASES) * len(conditions) * repeat
    done = 0

    for trial in range(repeat):
        for case in CASES:
            context = render(argus.retrieve(case.prompt, budget=budget, hops=hops))
            for condition in conditions:
                done += 1
                supplied = context if condition == "harness" else ""
                reply, thought_chars = _collect(engine, case.prompt, supplied)
                grade = grade_answer(case, reply, condition)
                # An empty reply after a long think is a budget failure, not a
                # model failure. Naming it stops the run being scored as though
                # the model had nothing to say.
                if grade.error == "empty reply" and thought_chars > 0:
                    grade.error = (f"empty reply: spent {thought_chars} chars "
                                   f"thinking, budget exhausted "
                                   f"(stop_reason="
                                   f"{getattr(engine, 'stop_reason', None)!r}) "
                                   f"-- raise --max-tokens")
                grades.append(grade)

                flag = "ERR" if grade.error else f"{grade.score:.2f}"
                print(f"  [{done:>3}/{total}] {case.id:<22} {condition:<8} {flag}",
                      file=sys.stderr)
                if verbose and not grade.error:
                    print("        " + reply.strip()[:300].replace("\n", "\n        "),
                          file=sys.stderr)
                if grade.error:
                    print(f"        {grade.error}", file=sys.stderr)
                    # A daily cap will not clear during this run. Continuing
                    # produces an hour of identical failures and a report made
                    # entirely of errors -- stop and keep what was scored.
                    if "quota" in grade.error.lower():
                        print("\n  daily quota exhausted -- stopping early. "
                              "Scores below cover completed cases only.",
                              file=sys.stderr)
                        return grades
                if sleep:
                    time.sleep(sleep)
    return grades


# ---------------------------------------------------------------- reporting

def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def report_retrieval(grades: Sequence[RetrievalGrade]) -> None:
    print("\nRETRIEVAL  (no model; deterministic)")
    print(f"  {'case':<24} {'hit':<5} {'rank':<5} first expected file")
    print("  " + "-" * 66)
    for g in grades:
        mark = "yes" if g.hit else "NO"
        rank = str(g.rank) if g.rank else "-"
        print(f"  {g.case_id:<24} {mark:<5} {rank:<5} {g.expected[0]}")
    hits = [g for g in grades if g.hit]
    top3 = [g for g in hits if g.rank and g.rank <= 3]
    print(f"\n  recall     {len(hits)}/{len(grades)}")
    print(f"  in top 3   {len(top3)}/{len(grades)}")
    if hits:
        print(f"  mean rank  {_mean([float(g.rank) for g in hits if g.rank]):.1f}")


def report_answers(grades: Sequence[AnswerGrade], conditions: Sequence[str]) -> None:
    scored = [g for g in grades if g.error is None]
    errors = [g for g in grades if g.error is not None]

    # Driven by the grades, not by CASES: the report should describe the run it
    # was handed, including cases that errored out of one condition entirely.
    by_case: Dict[str, Dict[str, List[float]]] = {}
    kinds: Dict[str, str] = {}
    order: List[str] = []
    for g in grades:
        if g.case_id not in kinds:
            kinds[g.case_id] = g.kind
            order.append(g.case_id)
    for g in scored:
        by_case.setdefault(g.case_id, {}).setdefault(g.condition, []).append(g.score)

    print("\nANSWERS")
    header = f"  {'case':<24} {'kind':<14}" + "".join(f"{c:>10}" for c in conditions)
    if len(conditions) == 2:
        header += f"{'delta':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for case_id in order:
        row = by_case.get(case_id)
        if not row:
            continue
        cells = ""
        means = {}
        for cond in conditions:
            vals = row.get(cond, [])
            means[cond] = _mean(vals)
            cells += f"{means[cond]:>10.2f}" if vals else f"{'-':>10}"
        if len(conditions) == 2 and all(row.get(c) for c in conditions):
            delta = means[conditions[1]] - means[conditions[0]]
            cells += f"{delta:>+9.2f}"
        print(f"  {case_id:<24} {kinds[case_id]:<14}{cells}")

    # Paired: only cases where EVERY condition produced a gradeable reply.
    # Averaging 10 raw cases against the 6 harness ones that survived compares
    # different questions and calls the difference an effect.
    complete = {cid for cid, row in by_case.items()
                if all(row.get(c) for c in conditions)}
    dropped = sorted(set(by_case) - complete)

    print("\n  by kind (paired: cases where every condition answered)")
    for kind in ("repo_specific", "general", "negative"):
        seen = [cid for cid in order if kinds[cid] == kind]
        if not seen:
            continue
        ids = [cid for cid in seen if cid in complete]
        cells = ""
        means = {}
        for cond in conditions:
            vals = [_mean(by_case[cid][cond]) for cid in ids]
            means[cond] = _mean(vals)
            cells += f"{means[cond]:>10.2f}" if vals else f"{'-':>10}"
        if len(conditions) == 2 and ids:
            cells += f"{means[conditions[1]] - means[conditions[0]]:>+9.2f}"
        label = f"n={len(ids)}/{len(seen)}"
        print(f"  {kind:<24} {label:<14}{cells}")

    if dropped:
        print(f"\n  excluded from the paired block ({len(dropped)} case(s) where a "
              f"condition failed):")
        print(f"    {', '.join(dropped)}")
        print("    An unpaired mean compares different questions per column.")

    fabs = [g for g in scored if g.fabricated]
    if fabs:
        print("\n  fabrications (known-wrong citation present):")
        for g in fabs:
            print(f"    {g.case_id} [{g.condition}]")

    if errors:
        print(f"\n  provider errors: {len(errors)} (excluded from all scores)")
        for g in errors[:5]:
            print(f"    {g.case_id} [{g.condition}] {g.error}")

    print("\n  Read the by-kind block, not the totals: the harness should lift")
    print("  repo_specific and leave general flat. A large general lift means the")
    print("  grader is rewarding something other than retrieval.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harness.eval",
        description="Score Argus retrieval, and the same model with and without it.")
    parser.add_argument("--mode", choices=("retrieval", "answer", "gate", "both"),
                        default="retrieval")
    parser.add_argument("--gate-threshold", type=float, default=0.5,
                        help="confidence at or above which context is injected")
    parser.add_argument("--repo", default=".", help="repository to evaluate against")
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--conditions", default="raw,harness",
                        help="comma-separated: raw, harness")
    parser.add_argument("--repeat", type=int, default=1,
                        help="samples per case per condition; >1 exposes variance")
    parser.add_argument("--budget", type=int, default=8000)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 for reproducibility; raise to measure spread")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="must leave room for a reasoning model's <think> "
                             "block, which is charged against the same budget; "
                             "too low and the reply is empty, not short")
    parser.add_argument("--sleep", type=float, default=4.0,
                        help="seconds between calls; free tiers cap tokens per "
                             "minute, so long-context calls exhaust the window fast")
    parser.add_argument("--json", default=None, help="write raw results here")
    parser.add_argument("--verbose", action="store_true", help="print each reply")
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    argus = Argus(repo)
    report = argus.scan()
    print(f"indexed {repo}: {report}", file=sys.stderr)

    payload: Dict[str, object] = {"repo": str(repo), "cases": len(CASES)}

    if args.mode in ("gate", "both"):
        payload["gate_accuracy"] = report_gate(argus, args.gate_threshold,
                                               args.budget, args.hops)

    retrieval_grades: List[RetrievalGrade] = []
    if args.mode in ("retrieval", "both"):
        retrieval_grades = run_retrieval(argus, args.budget, args.hops)
        report_retrieval(retrieval_grades)
        payload["retrieval"] = [vars(g) for g in retrieval_grades]

    if args.mode in ("answer", "both"):
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
        engine = OpenAICompatEngine(provider=args.provider, model=args.model,
                                    base_url=args.base_url,
                                    temperature=args.temperature,
                                    max_tokens=args.max_tokens)
        print(f"\nengine: {engine.name}  conditions={conditions}  "
              f"repeat={args.repeat}", file=sys.stderr)
        grades = run_answers(engine, argus, conditions, args.budget, args.hops,
                             args.repeat, args.sleep, args.verbose)
        report_answers(grades, conditions)
        payload["answers"] = [vars(g) for g in grades]

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str),
                                   encoding="utf-8")
        print(f"\nwrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
