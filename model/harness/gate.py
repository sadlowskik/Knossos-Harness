"""The retrieval gate: decide whether context belongs in the prompt at all.

Measured on gemma4:e4b, injecting retrieved excerpts into a *general* question
("what problem does rotary positional embedding solve?") dropped the answer score
from 1.00 to 0.00 -- the model stopped answering the question and started
describing this repository instead. The same injection on llama-3.3-70b cost
nothing. So irrelevant context is not free, and it is least free exactly where
this project is headed: small models.

Always retrieving is therefore the wrong default. The gate decides.

Signals, each named in the decision so a wrong call can be argued with rather
than guessed at -- the same reason `Retrieved.reason` exists:

    anchor        "which file", "in this codebase", "moe.py" -- asks about *here*
    distinctive   a query word naming a symbol that occurs in very few files.
                  "Mnemosyne" means this repo; "Router" or "forward" could mean
                  anything, which is why raw symbol matching is not enough.
    concentration retrieval that spikes on one file rather than smearing over ten
    generality    "in general", "conceptually", "what is a ..." -- asks about the
                  idea, not the code

The hard cases are questions whose vocabulary is shared with the repo: "in a
mixture-of-experts layer, what is expert collapse?" matches `moe.py` on every
content word and is still a general question. That is what `generality` is for,
and why it outweighs a bare symbol match.

Thresholds here are tuned against `harness.evalset` and re-measurable at any time:

    python -m harness.eval --mode gate
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .argus import Argus, Retrieved, _split_identifier, _tokenize

__all__ = ["RetrievalGate", "GateDecision"]


@dataclass
class GateDecision:
    inject: bool
    confidence: float                   # 0..1; >= threshold means inject
    reasons: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        verb = "inject" if self.inject else "skip"
        return f"{verb} ({self.confidence:.2f}): {'; '.join(self.reasons) or 'no signal'}"


#: Phrases that point at *this* codebase.
_ANCHORS = (
    r"\bwhich file\b", r"\bwhat file\b", r"\bwhich module\b", r"\bwhere is\b",
    r"\bwhere do(?:es)?\b", r"\bthis (?:repo|repository|codebase|project)\b",
    r"\bour\b", r"\bwe (?:use|do|have|handle|call)\b", r"\bin the code\b",
    r"\bdefined? in\b", r"\bimplemented? in\b", r"\bthe codebase\b",
)

#: Phrases that ask about an idea rather than an implementation. Kept to
#: high-precision ones only. An earlier draft included `what are`, which skipped
#: "What are Moirai's two gates called?" -- about as repo-specific as a question
#: gets. A generality marker that fires on ordinary question grammar is not a
#: generality marker.
_GENERAL = (
    r"\bin general\b", r"\bgenerally\b", r"\bconceptually\b", r"\bin theory\b",
    r"\bexplain the concept\b", r"\btypically\b", r"\busually\b",
    r"\bas a concept\b", r"\bwhat does .{1,40} mean\b",
)

#: Weaker: indefinite framing ("in *a* mixture-of-experts layer") describes a
#: category, where "the"/"our" would point at an instance. Half weight, because
#: it is a hint about grammar rather than a statement about intent.
_INDEFINITE = re.compile(
    r"\b(?:in|for|with|within)\s+an?\s+\w+[\w-]*\s+(?:layer|model|network|"
    r"system|transformer|architecture|module)\b", re.I)

_FILENAME = re.compile(r"\b\w+\.(?:py|rs|toml|md|json|ya?ml|txt|cfg|ini)\b", re.I)


class RetrievalGate:
    """Should this query get repository context?

    >>> gate = RetrievalGate(argus)
    >>> gate.decide("what is expert collapse, in general?")
    skip (0.25): generality phrasing; no distinctive repo symbol
    """

    #: The gate STARTS above the threshold: injecting is the default and skipping
    #: must be argued for. The costs are asymmetric -- a wrong skip loses a repo
    #: answer, which is the whole job, while a wrong inject loses a general answer
    #: the user could have asked anywhere. An earlier draft defaulted to skip and
    #: scored 60%, below the 87% of injecting unconditionally.
    BASE = 0.75

    W_ANCHOR = 0.20
    W_DISTINCTIVE = 0.20
    W_FILENAME = 0.20
    W_CONCENTRATION = 0.05
    W_GENERAL = -0.45
    W_INDEFINITE = -0.22

    #: A name occurring in at most this many files counts as distinctive.
    DISTINCTIVE_MAX_FILES = 3
    #: Below this length a token is too generic to identify a codebase.
    MIN_TOKEN = 4
    #: A term occurring as prose in more than this fraction of files is a word,
    #: not a name, and cannot make a question repo-specific.
    COMMON_FRACTION = 0.25

    def __init__(self, argus: Argus, threshold: float = 0.5) -> None:
        self.argus = argus
        self.threshold = threshold
        self._index: Optional[dict] = None

    def _name_index(self) -> dict:
        """Lowercased name fragment -> files it is defined in.

        Includes file stems and the camel/snake parts of every symbol, because
        people write "Moirai" for `MoiraiMixer` and "Proteus" for `ProteusBlock`.
        Exact-name matching alone misses the way names are actually spoken.
        """
        if self._index is not None:
            return self._index

        # Test functions are excluded. Their names are sentences --
        # `test_syntax_error_degrades_instead_of_crashing` -- so indexing their
        # parts fills the table with ordinary English ("does", "instead",
        # "error") and makes every question look like it names something.
        index: dict = {}
        for rel, rec in self.argus.files.items():
            stem = rel.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
            if not stem.startswith("test"):
                index.setdefault(stem, set()).add(rel)
            for sym in rec.symbols:
                if sym.name.startswith("test") or (sym.parent or "").startswith("Test"):
                    continue
                for part in {sym.name.lower(), *_split_identifier(sym.name)}:
                    if len(part) >= self.MIN_TOKEN:
                        index.setdefault(part, set()).add(sym.file)

        # A word that appears as prose all over the corpus is not a name, even
        # if some symbol happens to contain it. "expert" and "layer" are in half
        # these files; "Mnemosyne" is in a handful.
        common = int(max(3, self.COMMON_FRACTION * len(self.argus.files)))
        text_df: dict = {}
        for rec in self.argus.files.values():
            for term in rec.idents:
                text_df[term] = text_df.get(term, 0) + 1
        self._index = {name: files for name, files in index.items()
                       if text_df.get(name, 0) <= common}
        return self._index

    def _distinctive_hits(self, query: str) -> List[str]:
        """Query words naming something that lives in very few files.

        Frequency separates a name from a word: `Mnemosyne` is defined once and
        means this repository, while `forward` is defined in a dozen files and
        means nothing in particular.
        """
        index = self._name_index()
        found = []
        for token in dict.fromkeys(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)):
            if len(token) < self.MIN_TOKEN:
                continue
            files = index.get(token.lower())
            if files and len(files) <= self.DISTINCTIVE_MAX_FILES:
                found.append(token)
        return found

    def _concentrated(self, scores: Sequence[float]) -> bool:
        """True when retrieval spiked rather than smeared.

        A question about one thing in the repo scores one file far above the
        rest. A question whose words are merely common in the repo scores many
        files similarly.
        """
        positive = [s for s in scores if s > 0]
        if len(positive) < 3:
            return bool(positive)
        top = max(positive)
        median = statistics.median(positive)
        return median > 0 and top >= 2.0 * median

    def decide(self, query: str, hits: Optional[Sequence[Retrieved]] = None) -> GateDecision:
        low = query.lower()
        score = self.BASE
        reasons: List[str] = []

        anchored = any(re.search(p, low) for p in _ANCHORS)
        if anchored:
            score += self.W_ANCHOR
            reasons.append("asks about this codebase")

        named_file = bool(_FILENAME.search(query))
        if named_file:
            score += self.W_FILENAME
            reasons.append("names a file")

        distinctive = self._distinctive_hits(query)
        if distinctive:
            score += self.W_DISTINCTIVE
            reasons.append(f"names {', '.join(distinctive[:3])}")

        general = any(re.search(p, low) for p in _GENERAL)
        if general:
            score += self.W_GENERAL
            reasons.append("generality phrasing")

        indefinite = bool(_INDEFINITE.search(query))
        if indefinite:
            score += self.W_INDEFINITE
            reasons.append("indefinite framing (a category, not this instance)")

        # Decisive rule, because additive weights got this wrong: an incidental
        # name match ("post" -> `_post`, "Explain" -> `_explain`) was cancelling
        # an explicit "in general" and dragging the score back over the line.
        # Evidence of a general question wins unless something points at *here*
        # -- an anchor phrase, a filename, or a proper name that only means
        # something in this repo. The first word is excluded from the proper-name
        # test, since every sentence capitalises it.
        proper = [d for d in distinctive
                  if d[:1].isupper() and not low.startswith(d.lower())]
        veto = anchored or named_file or bool(proper)
        if (general or indefinite) and not veto:
            reasons.append("no anchor, filename or proper name to override it")
            return GateDecision(inject=False, confidence=min(score, 0.3),
                                reasons=reasons)
        if proper:
            reasons.append(f"proper name: {proper[0]}")

        if hits is None:
            file_scores = [s for _, s in self.argus.score_files(query)]
        else:
            file_scores = [h.score for h in hits]
        if self._concentrated(file_scores):
            score += self.W_CONCENTRATION
            reasons.append("retrieval concentrated")

        score = max(0.0, min(1.0, score))
        return GateDecision(inject=score >= self.threshold, confidence=score,
                            reasons=reasons)
