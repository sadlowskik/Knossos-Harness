"""Lethe: the river of forgetting -- bounded context with summarise-and-reset.

Talos resends the whole transcript every step, which is quadratic in tokens
over a long run. The repair loop makes that worse rather than better: every
failed attempt appends a verification report, and a task that takes eight
repairs pays for the first seven on every subsequent step.

Lethe bounds it. When the transcript exceeds a budget, the oldest middle is
replaced by a summary and the recent turns are kept verbatim:

    [opening]  [====== summarised ======]  [recent turns kept intact]
     pinned            forgotten                    verbatim

Three rules, each of which exists because dropping the wrong thing is worse
than keeping too much:

  The opening is pinned. It states the task; an agent that forgets what it was
  asked will confidently do something else.

  The most recent turns are never summarised. They hold the failure the model
  is currently repairing, and a summary of "the tests failed" is not enough to
  fix them -- the traceback is the whole point.

  Summarisation is lossy and says so. The replacement is explicitly marked as
  elided history so the model treats it as recollection rather than as the
  record.

The summariser is pluggable. The default is extractive and needs no model:
it keeps tool calls, file paths and verification outcomes, which is the
skeleton of what happened. An engine-backed summariser can be supplied when
the cost of a model call is worth better recall -- this is the same
lossy-gist idea as Mnemosyne, applied to a conversation instead of a segment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

__all__ = ["Lethe", "CompactionResult", "extractive_summary", "estimate_tokens"]

#: Rough chars-per-token for code-heavy English. Good enough for a budget, and
#: it avoids a tokenizer dependency in a package that has none.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class CompactionResult:
    """What a compaction did, so the caller can report or test it."""

    transcript: List[str]
    compacted: bool
    turns_before: int
    turns_after: int
    tokens_before: int
    tokens_after: int
    summarised_turns: int = 0

    @property
    def saved_tokens(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)

    def report(self) -> str:
        if not self.compacted:
            return f"no compaction needed ({self.tokens_before} tokens)"
        return (f"compacted {self.summarised_turns} turns: "
                f"{self.tokens_before} -> {self.tokens_after} tokens "
                f"({self.saved_tokens} saved)")


#: Lines worth keeping when summarising extractively: what was done, to what,
#: and whether it worked. Prose about intent is the part safe to lose.
_KEEP = re.compile(
    r"^\s*(#{1,3}\s|<call\b|</call>|tool:|file:|error|failed|passed|traceback|"
    r"assert|\+\+\+|---|@@)",
    re.IGNORECASE,
)
_PATH = re.compile(r"[\w./\\-]+\.(?:py|rs|toml|md|json|ya?ml|txt|cfg|ini)\b")


def extractive_summary(turns: Sequence[str], max_chars: int = 1200) -> str:
    """Summarise without a model.

    Keeps structural lines -- headings, tool calls, file paths, pass/fail
    outcomes -- and drops reasoning prose. That is deliberately the opposite of
    what a language model would keep, and it is the right trade here: the model
    is about to re-reason anyway, but it cannot re-derive which files it already
    touched.
    """
    kept: List[str] = []
    paths: List[str] = []
    for turn in turns:
        for line in turn.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            for path in _PATH.findall(stripped):
                if path not in paths:
                    paths.append(path)
            if _KEEP.match(stripped):
                kept.append(stripped)

    body: List[str] = []
    if paths:
        body.append("files touched: " + ", ".join(paths[:20]))
    body.extend(kept)

    text = "\n".join(body)
    if len(text) > max_chars:
        # Keep the tail: later events supersede earlier ones.
        text = "…\n" + text[-max_chars:]
    return text or "(no recoverable structure)"


class Lethe:
    """Keeps a transcript under a token budget.

    >>> lethe = Lethe(max_tokens=8000, keep_recent=6)
    >>> result = lethe.compact(transcript)
    >>> transcript = result.transcript
    """

    #: Marker for elided history. Present in the transcript so the model can see
    #: that something was forgotten rather than silently receiving a gap.
    MARKER = "## Earlier (summarised)"

    def __init__(self, max_tokens: int = 24_000, keep_recent: int = 6,
                 pin_opening: int = 1,
                 summariser: Optional[Callable[[Sequence[str]], str]] = None,
                 summary_max_chars: int = 1200) -> None:
        if keep_recent < 1:
            raise ValueError("keep_recent must be at least 1")
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.pin_opening = max(0, pin_opening)
        self.summary_max_chars = summary_max_chars
        self._summarise = summariser or (
            lambda turns: extractive_summary(turns, summary_max_chars)
        )

    def tokens(self, transcript: Sequence[str]) -> int:
        return estimate_tokens("\n".join(transcript))

    def needs_compaction(self, transcript: Sequence[str]) -> bool:
        return self.tokens(transcript) > self.max_tokens

    def compact(self, transcript: Sequence[str]) -> CompactionResult:
        """Return a transcript within budget, summarising the middle if needed.

        Idempotent: compacting an already-compacted transcript that still fits
        changes nothing, and one that does not fit summarises the next band
        rather than nesting summaries indefinitely.
        """
        turns = list(transcript)
        before_tokens = self.tokens(turns)
        before_turns = len(turns)

        if before_tokens <= self.max_tokens:
            return CompactionResult(turns, False, before_turns, before_turns,
                                    before_tokens, before_tokens)

        head = turns[: self.pin_opening]
        tail = turns[-self.keep_recent:] if self.keep_recent else []
        middle_start = len(head)
        middle_end = len(turns) - len(tail)

        if middle_end <= middle_start:
            # Nothing between the pinned opening and the kept tail. Refusing to
            # act is correct: the alternative is dropping the very turns the
            # model needs, and an over-budget prompt fails more honestly than a
            # silently gutted one.
            return CompactionResult(turns, False, before_turns, before_turns,
                                    before_tokens, before_tokens)

        # Unwrap any summary already in the middle so its content is folded into
        # the new one. Without this a long session accumulates a chain of
        # summaries of summaries: each pass adds a marker and re-summarises the
        # previous marker's text, which grows the transcript instead of
        # shrinking it.
        middle = [self._unwrap(t) for t in turns[middle_start:middle_end]]
        summary = self._summarise(middle)
        compacted = head + [f"\n{self.MARKER}\n{summary}"] + tail

        after_tokens = self.tokens(compacted)
        if after_tokens >= before_tokens:
            # Summarising made it bigger -- possible when the turns are short or
            # structural, so nearly every line is worth keeping. Compacting
            # anyway would spend a marker and lose fidelity for nothing.
            return CompactionResult(turns, False, before_turns, before_turns,
                                    before_tokens, before_tokens)

        return CompactionResult(
            transcript=compacted,
            compacted=True,
            turns_before=before_turns,
            turns_after=len(compacted),
            tokens_before=before_tokens,
            tokens_after=after_tokens,
            summarised_turns=len(middle),
        )

    def _unwrap(self, turn: str) -> str:
        """Strip the summary marker so a previous summary is treated as content."""
        stripped = turn.lstrip()
        if stripped.startswith(self.MARKER):
            return stripped[len(self.MARKER):].lstrip("\n")
        return turn
