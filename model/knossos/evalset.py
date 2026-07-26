"""The evaluation set: questions about this repo with checkable answers.

Every case is tagged `repo_specific` or `general`, and that split is the whole
design of the experiment. A retrieval harness should win big on facts that exist
only in this codebase -- which file defines what, what the Fates' two gates are
called -- and win *nothing* on facts the model already learned in pretraining,
like what RoPE is. If the harness appears to help on `general` cases too,
something is wrong with the grader, not right with the harness.

`negative` cases have no answer in the repo at all. They exist to measure
fabrication: the pass condition is admitting the absence. A retrieval system that
never says "not here" is not a retrieval system, it is a random file generator
with good manners.

Graders are keyword and citation matchers, not judges. They measure "did the
right file get cited and the right term appear", which is a proxy for a correct
answer and not the same thing. Read `expect_terms` as necessary, not sufficient.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

__all__ = ["Case", "CASES"]


@dataclass(frozen=True)
class Case:
    id: str
    prompt: str
    kind: str                                   # repo_specific | general | negative
    #: Files retrieval should surface. Also what a cited answer should mention.
    expect_files: Sequence[str] = ()
    #: Substrings a correct answer contains (case-insensitive). ANY-of within a
    #: group, ALL-of across groups: [["a","b"], ["c"]] means (a or b) and c.
    expect_terms: Sequence[Sequence[str]] = ()
    #: Substrings whose presence indicates a specific known fabrication.
    forbid_terms: Sequence[str] = ()
    note: str = ""


CASES: List[Case] = [

    # ------------------------------------------------ repo-specific retrieval

    Case(
        id="moe-router",
        kind="repo_specific",
        prompt="Which file defines the MoE router, and what stops one expert "
               "from taking all the traffic?",
        expect_files=["daedalus/moe.py"],
        # Stems, not whole words: "load-balancing" does not contain the
        # substring "load-balance", and scoring a correct answer as wrong
        # because of an inflection makes the whole measurement noise.
        expect_terms=[["load balanc", "load-balanc", "load_balanc", "auxiliary", "aux loss"],
                      ["nois", "top-k", "top_k"]],
        note="Two mechanisms: noisy top-k gating and the Switch aux loss.",
    ),
    Case(
        id="mnemosyne-compression",
        kind="repo_specific",
        prompt="How does Mnemosyne compress a segment, and by what factor?",
        expect_files=["daedalus/memory.py"],
        expect_terms=[["gist", "learned quer", "cross-atten", "cross atten"], ["8", "16"]],
        note="16 learned queries cross-attend a T=128 segment -> 8x.",
    ),
    Case(
        id="moirai-gates",
        kind="repo_specific",
        prompt="What are Moirai's two gates called and why are they decoupled?",
        expect_files=["daedalus/moirai.py"],
        expect_terms=[["clotho"], ["lachesis"], ["erase", "write"]],
        note="Naming exists nowhere but this repo -- a pure retrieval win.",
    ),
    Case(
        id="echo-distillation",
        kind="repo_specific",
        prompt="In Echo, which pass is the teacher and which is the student?",
        expect_files=["daedalus/echo.py"],
        expect_terms=[["teacher"], ["student"], ["detach", "stopgrad", "stop-grad"]],
        note="Deep R-loop pass teaches the shallow k-loop pass; teacher detached.",
    ),
    Case(
        id="naiads-banks",
        kind="repo_specific",
        prompt="How does Naiads stop one memory bank absorbing every segment?",
        expect_files=["daedalus/naiads.py"],
        expect_terms=[["bank"], ["load balanc", "load-balanc", "load_balanc", "aux"]],
    ),
    Case(
        id="proteus-srwm",
        kind="repo_specific",
        prompt="What makes Proteus level 2 self-referential, and whose paper is it?",
        expect_files=["daedalus/proteus.py"],
        expect_terms=[["irie"], ["2022"], ["srwm", "self-referential", "self referential"]],
    ),
    Case(
        id="scribe-exactness",
        kind="repo_specific",
        prompt="Why is Scribe described as exact rather than approximate?",
        expect_files=["daedalus/memory.py"],
        expect_terms=[["ast", "pars"], ["exact", "ground truth"]],
    ),
    Case(
        id="forward-collision",
        kind="repo_specific",
        prompt="Which classes define a forward method? Name as many as you can, "
               "with their files.",
        expect_files=["daedalus/moe.py", "daedalus/layers.py", "daedalus/full.py"],
        expect_terms=[["forward"]],
        note="28 definitions across the repo. Tests that retrieval does not "
             "collapse same-named symbols into one.",
    ),
    Case(
        id="halting-location",
        kind="repo_specific",
        prompt="Where is the per-token halting probability computed?",
        expect_files=["daedalus/ariadne.py"],
        expect_terms=[["sigmoid", "lam", "halt"]],
    ),

    # ---------------------------------------------------- general knowledge

    Case(
        id="rope-general",
        kind="general",
        prompt="What problem does rotary positional embedding solve, in general?",
        expect_terms=[["relative", "position"]],
        note="Pretraining knowledge. The harness should add nothing here.",
    ),
    Case(
        id="moe-general",
        kind="general",
        prompt="In a mixture-of-experts layer, what is expert collapse?",
        expect_terms=[["collapse", "few", "same expert", "imbalance"]],
        note="Hard: every content word also occurs in moe.py. Only the "
             "indefinite framing marks it as a question about the category.",
    ),
    Case(
        id="prenorm-general",
        kind="general",
        prompt="What is the difference between pre-norm and post-norm "
               "transformer blocks, in general?",
        expect_terms=[["pre-norm", "prenorm", "before"], ["stab", "gradient", "deep"]],
    ),
    Case(
        id="quantization-general",
        kind="general",
        prompt="Explain the concept of 4-bit quantization for neural network weights.",
        expect_terms=[["4-bit", "precision", "memory", "smaller"]],
    ),
    Case(
        id="kvcache-general",
        kind="general",
        prompt="How does KV caching speed up autoregressive inference, typically?",
        expect_terms=[["cach", "reuse", "recompute", "previous"]],
    ),
    Case(
        id="gradclip-general",
        kind="general",
        prompt="Why does gradient clipping help stabilise training?",
        expect_terms=[["explod", "large", "norm", "stab"]],
        note="Deliberately hard for the gate: no generality marker, no "
             "indefinite framing, no repo symbol. Expected to be mis-injected. "
             "Kept so the reported accuracy is honest rather than flattering.",
    ),

    # ------------------------------------------------------ fabrication traps

    Case(
        id="pondernet-citation",
        kind="repo_specific",
        prompt="Which paper is the adaptive halting mechanism taken from? "
               "Give the authors and year.",
        # README.md carries the reference list, so citing it is as correct as
        # citing the implementation. The original expectation named only
        # ariadne.py and marked a right answer wrong.
        expect_files=["daedalus/ariadne.py", "README.md"],
        expect_terms=[["pondernet"], ["banino"]],
        forbid_terms=["so et al", "graves 2016 halting"],
        note="Observed failure: qwen3.6 confidently answered 'So et al. 2017'. "
             "Correct is Banino et al. 2021.",
    ),
    Case(
        id="no-retries",
        kind="negative",
        prompt="Where does this codebase handle network retries and backoff?",
        note="There is no retry logic. Pass = says so.",
    ),
    Case(
        id="no-database",
        kind="negative",
        prompt="Which module opens the database connection, and what pooling "
               "settings does it use?",
        note="There is no database. Pass = says so.",
    ),
    Case(
        id="no-auth",
        kind="negative",
        prompt="How does the user authentication flow validate passwords?",
        note="There is no auth. Pass = says so.",
    ),
]
