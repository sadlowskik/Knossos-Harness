"""Grammatical error correction: corruption operators, format, and scoring.

The training signal for a corrector is free. Take clean text -- which the
phase-1 corpus already is -- damage it in ways a human plausibly would, and the
original is the label. This module holds the damage operators, the wire format
that `scripts/make_gec.py` writes and `scripts/evaluate.py` reads, and the
metrics. Keeping all three here is deliberate: a separator that drifts between
the builder and the evaluator produces a model that scores zero for a reason no
metric can show you.

Why the operator mix leans on confusables and agreement rather than typos: a
1MB dictionary plus edit distance already fixes `teh` faster and more reliably
than any neural model, so spelling is not where a 45M model earns its place.
`their`/`there`, `its`/`it's` and subject-verb agreement are errors where every
word is spelled correctly and only the context disambiguates -- which is the
one thing a language model has that a dictionary does not.
"""
from __future__ import annotations

import difflib
import re
from typing import Dict, Iterator, List, Sequence, Tuple

# ---------------------------------------------------------------- wire format
# The prompt ends with GEC_SEP and the completion ends with GEC_END. A tab is
# used because it is essentially absent from prose, so the model never has to
# guess whether a separator is content.
GEC_SEP = "\t"
GEC_END = "\n"


def build_prompt(corrupted: str) -> str:
    return corrupted + GEC_SEP


def parse_completion(generated: str) -> str:
    """Take the model's raw continuation and cut it at the first boundary."""
    for stop in (GEC_END, GEC_SEP, "<|endoftext|>"):
        i = generated.find(stop)
        if i != -1:
            generated = generated[:i]
    return generated.strip()


# ------------------------------------------------------------------ operators

CONFUSABLE_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("their", "there", "they're"), ("your", "you're"), ("its", "it's"),
    ("to", "too", "two"), ("then", "than"), ("affect", "effect"),
    ("lose", "loose"), ("whose", "who's"), ("accept", "except"),
    ("were", "we're", "where"), ("hear", "here"), ("piece", "peace"),
    ("principal", "principle"), ("complement", "compliment"),
    ("stationary", "stationery"), ("advice", "advise"), ("passed", "past"),
    ("bare", "bear"), ("cite", "site", "sight"), ("desert", "dessert"),
    ("elicit", "illicit"), ("farther", "further"), ("later", "latter"),
    ("quiet", "quite"), ("weather", "whether"), ("breath", "breathe"),
    ("choose", "chose"), ("lead", "led"),
)
# Single words only: lookup is keyed on one whitespace-delimited token, so a
# multi-word entry here would never match and would look like a live rule.
assert all(" " not in w for g in CONFUSABLE_GROUPS for w in g)
CONFUSABLE_LOOKUP: Dict[str, int] = {
    w: i for i, group in enumerate(CONFUSABLE_GROUPS) for w in group
}

AGREEMENT_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("is", "are"), ("was", "were"), ("has", "have"), ("does", "do"),
    ("doesn't", "don't"), ("isn't", "aren't"), ("wasn't", "weren't"),
    ("this", "these"), ("that", "those"), ("hasn't", "haven't"),
)
AGREEMENT_LOOKUP: Dict[str, int] = {
    w: i for i, group in enumerate(AGREEMENT_GROUPS) for w in group
}

FUNCTION_WORDS = ("a", "an", "the", "of", "to", "in", "for", "on", "at", "and")

_KEYBOARD = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str, min_chars: int = 40,
                    max_chars: int = 300) -> Iterator[str]:
    """Yield whitespace-normalised sentences inside a length band.

    Normalisation matters more than it looks: the corrupted and clean strings
    must differ *only* by the introduced errors, or every metric silently
    measures whitespace as well.
    """
    for para in text.split("\n"):
        for sent in _SENTENCE_BREAK.split(para):
            sent = " ".join(sent.split())
            if min_chars <= len(sent) <= max_chars:
                yield sent


def _split_word(word: str) -> Tuple[str, str, str]:
    """`'"Hello,'` -> `('"', 'Hello', ',')`. Interior apostrophes survive."""
    i, j = 0, len(word)
    while i < j and not word[i].isalnum():
        i += 1
    while j > i and not word[j - 1].isalnum():
        j -= 1
    return word[:i], word[i:j], word[j:]


def _match_case(source: str, replacement: str) -> str:
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _swap_from(lookup, groups, words, rng) -> bool:
    cands = [i for i, w in enumerate(words) if _split_word(w)[1].lower() in lookup]
    if not cands:
        return False
    i = rng.choice(cands)
    pre, core, suf = _split_word(words[i])
    alts = [a for a in groups[lookup[core.lower()]] if a != core.lower()]
    if not alts:
        return False
    words[i] = pre + _match_case(core, rng.choice(alts)) + suf
    return True


def _op_confusable(words: List[str], rng) -> bool:
    return _swap_from(CONFUSABLE_LOOKUP, CONFUSABLE_GROUPS, words, rng)


def _op_agreement(words: List[str], rng) -> bool:
    return _swap_from(AGREEMENT_LOOKUP, AGREEMENT_GROUPS, words, rng)


def _op_article(words: List[str], rng) -> bool:
    cands = [i for i, w in enumerate(words) if _split_word(w)[1].lower() in ("a", "an")]
    if not cands:
        return False
    i = rng.choice(cands)
    pre, core, suf = _split_word(words[i])
    words[i] = pre + _match_case(core, "an" if core.lower() == "a" else "a") + suf
    return True


def _op_drop_word(words: List[str], rng) -> bool:
    if len(words) < 6:
        return False
    cands = [i for i, w in enumerate(words)
             if _split_word(w)[1].lower() in FUNCTION_WORDS and not _split_word(w)[2]]
    if not cands:
        return False
    del words[rng.choice(cands)]
    return True


def _op_double_word(words: List[str], rng) -> bool:
    if len(words) < 4:
        return False
    i = rng.randrange(1, len(words))
    if _split_word(words[i])[2]:            # never duplicate across punctuation
        return False
    words.insert(i, words[i])
    return True


def _op_apostrophe(words: List[str], rng) -> bool:
    cands = [i for i, w in enumerate(words) if "'" in _split_word(w)[1]]
    if not cands:
        return False
    i = rng.choice(cands)
    pre, core, suf = _split_word(words[i])
    words[i] = pre + core.replace("'", "") + suf
    return True


def _op_swap_adjacent(words: List[str], rng) -> bool:
    if len(words) < 5:
        return False
    i = rng.randrange(1, len(words) - 2)
    if _split_word(words[i])[2] or _split_word(words[i + 1])[2]:
        return False
    words[i], words[i + 1] = words[i + 1], words[i]
    return True


def _op_typo(words: List[str], rng) -> bool:
    cands = [i for i, w in enumerate(words) if len(_split_word(w)[1]) >= 4]
    if not cands:
        return False
    i = rng.choice(cands)
    pre, core, suf = _split_word(words[i])
    kind = rng.choice(("adjacent", "transpose", "double", "drop"))
    j = rng.randrange(len(core))
    if kind == "adjacent" and core[j].lower() in _KEYBOARD:
        core = core[:j] + rng.choice(_KEYBOARD[core[j].lower()]) + core[j + 1:]
    elif kind == "transpose" and j < len(core) - 1:
        core = core[:j] + core[j + 1] + core[j] + core[j + 2:]
    elif kind == "double":
        core = core[:j] + core[j] + core[j:]
    elif kind == "drop" and len(core) > 4:
        core = core[:j] + core[j + 1:]
    else:
        return False
    words[i] = pre + core + suf
    return True


def _op_lowercase_start(words: List[str], rng) -> bool:
    pre, core, suf = _split_word(words[0])
    if not core[:1].isupper():
        return False
    words[0] = pre + core[0].lower() + core[1:] + suf
    return True


# Weighted toward the errors a dictionary cannot see. Typos are kept in at a
# low rate so the model is not thrown by them, not because they are the point.
OPERATORS: Tuple[Tuple[str, object, float], ...] = (
    ("confusable", _op_confusable, 3.0),
    ("agreement", _op_agreement, 3.0),
    ("article", _op_article, 1.5),
    ("drop_word", _op_drop_word, 1.5),
    ("apostrophe", _op_apostrophe, 1.5),
    ("double_word", _op_double_word, 1.0),
    ("swap_adjacent", _op_swap_adjacent, 1.0),
    ("lowercase_start", _op_lowercase_start, 0.75),
    ("typo", _op_typo, 1.0),
)


def corrupt_sentence(text: str, rng, max_edits: int = 2) -> Tuple[str, int]:
    """Damage `text`, returning `(corrupted, n_edits)`.

    `n_edits` is 0 when nothing changed -- including when an operator fired but
    happened to produce the original string. Callers rely on `corrupted != clean`
    being an exact test for "this example contains an error", so the comparison
    is made here rather than trusting the operator return values.
    """
    words = text.split()
    if len(words) < 5:
        return text, 0
    names = [o[0] for o in OPERATORS]
    fns = {o[0]: o[1] for o in OPERATORS}
    weights = [o[2] for o in OPERATORS]
    target = rng.randint(1, max(max_edits, 1))
    applied = 0
    for _ in range(target * 4):
        if applied >= target:
            break
        if fns[rng.choices(names, weights=weights, k=1)[0]](words, rng):
            applied += 1
    out = " ".join(words)
    return (out, applied) if out != text else (text, 0)


# -------------------------------------------------------------------- scoring

def gec_score(corrupted: Sequence[str], clean: Sequence[str],
              predicted: Sequence[str]) -> Dict[str, float]:
    """Score corrections against references.

    `copy_rate` is the metric that matters most and the one a naive setup omits.
    The degenerate solution for a corrector is the identity function: echo the
    input and you are right on every already-correct sentence, which is most of
    them. A model doing that can post a respectable overall exact-match score
    while fixing nothing, so `exact_match_errored` and `copy_rate` have to be
    read together -- it is the same trap as loss-looks-great-output-is-whitespace.

    `false_positive_rate` is its mirror: corrections invented on text that was
    already correct. A corrector that damages clean input is worse than none.
    """
    if not (len(corrupted) == len(clean) == len(predicted)):
        raise ValueError("corrupted, clean and predicted must be the same length")
    errored = [i for i in range(len(clean)) if corrupted[i] != clean[i]]
    intact = [i for i in range(len(clean)) if corrupted[i] == clean[i]]

    def _ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a.split(), b.split()).ratio()

    fixed = sum(1 for i in errored if predicted[i].strip() == clean[i].strip())
    copied = sum(1 for i in errored if predicted[i].strip() == corrupted[i].strip())
    kept = sum(1 for i in intact if predicted[i].strip() == clean[i].strip())
    return {
        "exact_match_errored": fixed / len(errored) if errored else 0.0,
        "copy_rate": copied / len(errored) if errored else 0.0,
        "exact_match_clean": kept / len(intact) if intact else 0.0,
        "false_positive_rate": 1.0 - (kept / len(intact)) if intact else 0.0,
        "word_similarity": (sum(_ratio(predicted[i], clean[i])
                                for i in range(len(clean))) / len(clean)
                            if clean else 0.0),
        "n_errored": len(errored),
        "n_clean": len(intact),
    }
