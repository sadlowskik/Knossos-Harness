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
#: A file path mentioned in a line. Two details here are performance, not
#: pedantry, and they are the difference between this module costing nothing and
#: costing more than the model.
#:
#: **The leading delimiter.** Without it the engine tries to match at *every*
#: offset in a line, and `[\w./\\-]+` happily consumes a long run of word
#: characters before discovering there is no `.py` at the end -- then backs off
#: one character and does it again. On a single long line with no dot that is
#: quadratic. Measured on a 3 000-character line: 8.6 ms per call, 1 240 ms per
#: executor step, ~15 s across a 12-step run, spent entirely inside `findall`.
#: Requiring a delimiter collapses the number of start positions to the number
#: of word boundaries, which is what makes it linear in practice.
#:
#: **The bounded repeat.** A real path is not 3 000 characters. Capping the run
#: puts a ceiling on the work done at any single start position, so the worst
#: case degrades gracefully instead of exploding.
_PATH = re.compile(
    r"(?:^|[\s\"'`(\[{,:=])"
    r"([\w.\\/-]{1,200}\.(?:py|rs|toml|md|json|ya?ml|txt|cfg|ini))\b")

#: Lines longer than this are not scanned for paths. A structural line -- a
#: heading, a tool call, a traceback frame -- is short; something this long is
#: data (minified source, a base64 blob, one enormous log line), and scanning it
#: costs far more than the paths it might yield are worth.
_MAX_SCANNED_LINE = 2_000


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
    #: `paths` stays a list because order is meaningful -- the first files
    #: mentioned are the ones the run started from -- but membership testing on
    #: a list is linear, and this runs once per line of the whole middle band.
    seen_paths: set = set()
    for turn in turns:
        for line in turn.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if len(stripped) <= _MAX_SCANNED_LINE:
                for path in _PATH.findall(stripped):
                    if path not in seen_paths:
                        seen_paths.add(path)
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


def _elide(text: str, budget: int) -> str:
    """Shrink one entry to roughly `budget` chars, keeping both ends.

    Both ends, because the two halves answer different questions. The head says
    what this entry *is* -- which tool ran, with what arguments, against which
    file. The tail carries the conclusion: pytest's failure summary, the last
    lines of a traceback, the end of a diff. Keeping only the head, which is
    what a plain truncation does, throws away the part the model needs to act
    on and leaves the part it already knew.
    """
    if len(text) <= budget:
        return text
    half = max(1, budget // 2)
    head, tail = text[:half], text[-half:]
    dropped = len(text) - len(head) - len(tail)
    return f"{head}\n\n[… {dropped} characters elided …]\n\n{tail}"


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
            # Nothing between the pinned opening and the kept tail, so there is
            # no band to summarise. Dropping whole turns here would be wrong --
            # they are the ones the model needs -- but the transcript is still
            # over budget, and an oversized *entry* can be elided without losing
            # a turn. That is what `_fit` does.
            return self._fitted(turns, before_turns, before_tokens)

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
            # structural, so nearly every line is worth keeping. Spending a
            # marker and losing fidelity for nothing would be the wrong trade,
            # so the summary is discarded. The transcript is still over budget
            # though, so this is not a place to stop: elide instead.
            #
            # This is the path the measured 50 100-vs-24 000 case actually took.
            # A transcript dominated by one huge tool result has a *small*
            # middle, so summarising it saves nothing, the early return fired,
            # and the caller received the original transcript unchanged and
            # twice over budget.
            return self._fitted(turns, before_turns, before_tokens)

        # Summarising the middle is not, on its own, a bound either: the pinned
        # opening and the kept tail are whatever size they happen to be.
        compacted = self._fit(compacted)

        return CompactionResult(
            transcript=compacted,
            compacted=True,
            turns_before=before_turns,
            turns_after=len(compacted),
            tokens_before=before_tokens,
            tokens_after=self.tokens(compacted),
            summarised_turns=len(middle),
        )

    def _fitted(self, turns: List[str], before_turns: int,
                before_tokens: int) -> CompactionResult:
        """Elide oversized entries without summarising anything.

        For the two paths where summarisation is not available or not worth it,
        but the transcript is still over budget. `compacted` reports whether
        anything actually changed, so a transcript nothing could be done for is
        still reported as untouched.
        """
        fitted = self._fit(turns)
        return CompactionResult(fitted, fitted != turns, before_turns,
                                len(fitted), before_tokens, self.tokens(fitted))

    #: Smallest an entry is worth eliding to. Below this the marker and the
    #: surviving fragments cost more than they explain.
    MIN_ENTRY_CHARS = 400

    #: Hard stop on the fitting loop, so a pathological transcript cannot spin.
    #: Each pass halves the largest entry, so this is far more than enough.
    _FIT_PASSES = 40

    def _fit(self, turns: List[str]) -> List[str]:
        """Bring `turns` within budget by eliding the middle of large entries.

        Elides rather than drops, and elides the *biggest* entry each pass, on
        the reasoning that a transcript goes over budget because one or two
        entries are enormous -- a file read, a pytest dump -- not because forty
        turns are each slightly too long. Dropping whole turns would be the
        other option and is worse: the turns at risk are the most recent ones,
        which is precisely the context the model is working from.

        Every elision leaves a marker, so the model is told something is
        missing rather than silently receiving a seamless-looking lie. When
        nothing further can be given up, this returns what it has: an
        over-budget prompt is still better than a gutted one, and the caller
        can see the shortfall in `CompactionResult.tokens_after`.
        """
        out = list(turns)
        # The pinned opening is exempt. It holds the task statement, and a
        # transcript that has forgotten what it was asked to do is worse than
        # one that is over budget -- pinning it is the entire reason
        # `pin_opening` exists, so eliding it here would quietly undo that.
        candidates = range(self.pin_opening, len(out))
        if not candidates:
            return out

        for _ in range(self._FIT_PASSES):
            if self.tokens(out) <= self.max_tokens:
                break
            index = max(candidates, key=lambda i: len(out[i]))
            longest = out[index]
            if len(longest) <= self.MIN_ENTRY_CHARS:
                break                       # nothing left worth taking
            candidate = _elide(longest, max(self.MIN_ENTRY_CHARS,
                                            len(longest) // 2))
            if len(candidate) >= len(longest):
                # The marker costs more than the elision saves. Since the
                # largest entry is the one being cut, no smaller one can do
                # better, so there is nothing further to try -- and pressing on
                # would *grow* the transcript, which is the opposite of the job.
                break
            out[index] = candidate
        return out

    def _unwrap(self, turn: str) -> str:
        """Strip the summary marker so a previous summary is treated as content."""
        stripped = turn.lstrip()
        if stripped.startswith(self.MARKER):
            return stripped[len(self.MARKER):].lstrip("\n")
        return turn
