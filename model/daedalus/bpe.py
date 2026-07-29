"""Byte-level BPE, written from scratch (no `tokenizers` dependency).

Why this exists
---------------
`ByteTokenizer` (vocab 256) has a fatal cost at scale: one byte = one token, so a
2048-token context holds ~2KB of text. Real language modelling needs ~4x more
text per position. Byte-level BPE (Sennrich et al. 2016; GPT-2, Radford et al.
2019) merges frequent byte pairs into single tokens, giving ~4.2 bytes/token on
English prose and ~3.3 on code -- so the same compute sees ~4x more material,
and there is still no UNK token, because every merge bottoms out in raw bytes.

The algorithm
-------------
1. Pre-tokenize the corpus with a regex so merges never cross a word boundary
   (without this, BPE learns tokens like " the def" that generalise badly).
2. Count how often each chunk occurs. BPE then runs over *unique* chunks
   weighted by frequency, not over the raw stream.
3. Repeatedly merge the most frequent adjacent symbol pair, assigning it the
   next free id. Pair counts are updated **incrementally** -- only the chunks
   that actually contained the merged pair are touched -- which is what makes
   training in pure Python tractable rather than quadratic.

Encoding replays the merges in the order they were learned (lowest rank first),
so encode/decode are exact inverses of the training procedure by construction.
"""
from __future__ import annotations

import heapq
import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# A cl100k-flavoured split pattern using only the stdlib `re` module.
#   - `[^\W\d_]` is "unicode letter" (\p{L}) without needing the `regex` package
#   - digits are capped at 3 per token so numbers do not blow up the vocabulary
#   - `\s*[\r\n]` and `\s+` keep runs of indentation together, which matters far
#     more for code than for prose
#
# The punctuation branch is written as a tempered `(?!...)`. rather than the
# obvious `[^\s\w]+`, because `\w` *includes* `_` -- and with `[^\s\w]+` the
# underscore matches no alternative at all and is silently dropped. Every
# `__init__`, `snake_case` and `self._x` in the corpus would lose a character,
# with nothing to indicate it. The trailing `[\s\S]` is a catch-all that makes
# the same class of bug impossible: every character matches something.
SPLIT_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)"""                    # common English contractions
    r"""|[^\r\n\w]?[^\W\d_]+"""                    # optional lead symbol + letters
    r"""|\d{1,3}"""                                # short digit runs
    r"""| ?(?:(?![^\W\d_]|\d|\s).)+[\r\n]*"""      # punctuation, including `_`
    r"""|\s*[\r\n]"""                              # newline with leading indent
    r"""|\s+(?!\S)"""                              # trailing whitespace run
    r"""|\s+"""                                    # any other whitespace
    r"""|[\s\S]"""                                 # catch-all: never drop a char
)

Pair = Tuple[int, int]


class BPETokenizer:
    """Byte-level BPE with a learned merge table.

    Ids 0..255 are the raw bytes. Ids 256.. are merges, in learned order.
    Special tokens occupy the ids immediately after the merges.
    """

    def __init__(self, merges: Optional[Sequence[Pair]] = None,
                 specials: Optional[Dict[str, int]] = None):
        self.merges: List[Pair] = [tuple(m) for m in (merges or [])]
        # (a, b) -> new id.  Rank is implied by position: earlier = merged first.
        self.ranks: Dict[Pair, int] = {p: 256 + i for i, p in enumerate(self.merges)}
        self.specials: Dict[str, int] = dict(specials or {})
        self._vocab: Dict[int, bytes] = self._build_vocab()
        self._cache: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------ vocab

    def _build_vocab(self) -> Dict[int, bytes]:
        vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            vocab[256 + i] = vocab[a] + vocab[b]
        for text, tid in self.specials.items():
            vocab[tid] = text.encode("utf-8")
        return vocab

    @property
    def vocab_size(self) -> int:
        base = 256 + len(self.merges)
        return max([base] + [i + 1 for i in self.specials.values()])

    def token_bytes(self, tid: int) -> bytes:
        return self._vocab[tid]

    # --------------------------------------------------------------- encoding

    def _encode_chunk(self, chunk: str) -> List[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        # Replay merges lowest-rank-first. Each pass applies the single
        # earliest-learned merge present, which is what training did.
        while len(ids) >= 2:
            best, best_rank = None, None
            for pair in zip(ids, ids[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = pair, rank
            if best is None:
                break
            ids = _merge(ids, best, best_rank)
        if len(self._cache) < 500_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> List[int]:
        out: List[int] = []
        for chunk in SPLIT_PATTERN.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode_with_specials(self, text: str) -> List[int]:
        """Like `encode`, but special-token strings are matched literally.

        Only use this on trusted text -- it lets the input assert token
        boundaries (e.g. a document separator) that raw `encode` cannot.
        """
        if not self.specials:
            return self.encode(text)
        pattern = "(" + "|".join(re.escape(s) for s in self.specials) + ")"
        out: List[int] = []
        for part in re.split(pattern, text):
            if not part:
                continue
            if part in self.specials:
                out.append(self.specials[part])
            else:
                out.extend(self.encode(part))
        return out

    def decode(self, ids: Sequence[int]) -> str:
        raw = b"".join(self._vocab.get(int(i), b"") for i in ids)
        return raw.decode("utf-8", errors="replace")

    # --------------------------------------------------------------- training

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 32768,
              specials: Sequence[str] = ("<|endoftext|>",),
              min_frequency: int = 2, verbose: bool = False) -> "BPETokenizer":
        """Learn a merge table from an iterable of text chunks.

        `vocab_size` counts everything: 256 bytes + merges + specials. It is an
        **upper bound** -- a corpus can run out of repeated pairs before the
        budget is spent, which is common on small samples. Always read the
        resulting `.vocab_size` rather than assuming the requested number, or
        the model will be built with an output layer that does not match.
        """
        n_merges = vocab_size - 256 - len(specials)
        if n_merges < 0:
            raise ValueError(f"vocab_size must be >= {256 + len(specials)}")

        # 1. chunk frequencies -----------------------------------------------
        freqs: Counter = Counter()
        for text in texts:
            freqs.update(SPLIT_PATTERN.findall(text))
        if verbose:
            print(f"  {len(freqs):,} unique chunks, {sum(freqs.values()):,} total")

        words: List[List[int]] = []
        counts: List[int] = []
        for chunk, n in freqs.items():
            if n < min_frequency and len(freqs) > 10_000:
                continue
            words.append(list(chunk.encode("utf-8")))
            counts.append(n)

        # 2. pair statistics + an index of which words contain each pair ------
        pair_counts: Counter = Counter()
        where: Dict[Pair, set] = {}
        for wi, sym in enumerate(words):
            c = counts[wi]
            for pair in zip(sym, sym[1:]):
                pair_counts[pair] += c
                where.setdefault(pair, set()).add(wi)

        # 3. merge loop -------------------------------------------------------
        # `max(pair_counts, ...)` would be O(unique pairs) *per merge* -- on a
        # real corpus that is ~1e6 x 3e4 comparisons and dominates everything.
        # A max-heap with lazy deletion replaces it: stale entries are skipped
        # on pop, and every count update pushes a fresh entry.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        def bump(pair: Pair, delta: int) -> None:
            pair_counts[pair] += delta
            c = pair_counts[pair]
            if c <= 0:
                pair_counts.pop(pair, None)
            elif delta > 0:
                heapq.heappush(heap, (-c, pair))

        merges: List[Pair] = []
        for step in range(n_merges):
            best = None
            while heap:
                neg_c, pair = heapq.heappop(heap)
                if pair_counts.get(pair, 0) == -neg_c:      # not stale
                    best = pair
                    break
            if best is None:
                break
            new_id = 256 + len(merges)
            merges.append(best)

            # Only words containing `best` can change.
            for wi in list(where.get(best, ())):
                sym = words[wi]
                c = counts[wi]
                if len(sym) < 2:
                    continue
                # Remove this word's old pair contributions...
                for pair in zip(sym, sym[1:]):
                    bump(pair, -c)
                    s = where.get(pair)
                    if s is not None:
                        s.discard(wi)
                new_sym = _merge(sym, best, new_id)
                words[wi] = new_sym
                # ...and add the new ones.
                for pair in zip(new_sym, new_sym[1:]):
                    bump(pair, c)
                    where.setdefault(pair, set()).add(wi)

            pair_counts.pop(best, None)
            where.pop(best, None)
            if verbose and (step + 1) % 1000 == 0:
                print(f"  merge {step + 1:,}/{n_merges:,}")

        if len(merges) < n_merges:
            print(f"  note: corpus exhausted after {len(merges):,} merges "
                  f"(asked for {n_merges:,}) -> vocab "
                  f"{256 + len(merges) + len(specials):,}, not {vocab_size:,}")

        special_ids = {s: 256 + len(merges) + i for i, s in enumerate(specials)}
        return cls(merges, special_ids)

    # ------------------------------------------------------------ persistence

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": [list(m) for m in self.merges],
                       "specials": self.specials}, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls([tuple(m) for m in d["merges"]], d.get("specials", {}))


def _merge(ids: Sequence[int], pair: Pair, new_id: int) -> List[int]:
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`."""
    out: List[int] = []
    i, a, b = 0, pair[0], pair[1]
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
