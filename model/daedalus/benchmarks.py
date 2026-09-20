"""Pretraining benchmarks: the numbers that decide whether a run actually worked.

This exists because of the 37M Rust run, which reached 0.46 bits/byte and then
generated pure whitespace. A loss curve alone cannot tell a good run from a
degenerate one, and neither can a loss measured in *bits per token* -- change the
tokenizer and that number moves without the model changing at all.

So the headline metric here is **bits per byte**:

    bits_per_byte = sum(-log2 P(token)) / sum(len(token_bytes))

which is invariant to tokenization and therefore the one loss number that can be
compared against another model, another vocab size, or your own byte-level runs
from Phase 0. Everything else in this module is a *capability* probe, because
compression and capability come apart at small scale.

Three families, in increasing order of how much they need scale to move:

    bits/byte          moves from step one; the run-health signal
    multiple choice    HellaSwag/ARC/PIQA-style, near chance below ~100M
    parse rate         does generated code actually parse; moves at ~50M

`harness/eval.py` is a different thing entirely -- it measures the agent harness
(retrieval and answer quality) around a finished model, not the model itself.

References: Gao et al. (2021) `lm-evaluation-harness` for the scoring protocol;
Zellers et al. (2019) HellaSwag; Radford et al. (2019) for bits-per-byte.
"""
from __future__ import annotations

import ast
import math
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

__all__ = ["model_log_probs", "token_byte_lengths", "evaluate_bits_per_byte",
           "score_continuations", "multiple_choice", "parse_rate",
           "balanced_delimiters", "halting_stats"]


# --------------------------------------------------------------------------
# Uniform access to a model's predictive distribution
# --------------------------------------------------------------------------

def model_log_probs(model, x: torch.Tensor, vocab_size: Optional[int] = None
                    ) -> torch.Tensor:
    """(B, T) ids -> (B, T, V) log-probabilities, for any Daedalus model.

    The models do not share a return signature: most give `(logits, ...)`, but
    Ariadne gives `(halting_probs, per_step_logits)`. For Ariadne the correct
    predictive distribution is not any single step's logits -- it is the
    PonderNet mixture `sum_n p_n * softmax(logits_n)`, weighted by the
    probability of halting at that step. Taking the last step instead would
    score a model that never actually commits to it.
    """
    out = model(x)
    if torch.is_tensor(out):
        return F.log_softmax(out.float(), dim=-1)

    first = out[0]
    if (len(out) > 1 and torch.is_tensor(out[1]) and out[1].dim() == 4
            and first.dim() == 3):
        p, step_logits = first, out[1]                    # (N,B,T), (N,B,T,V)
        probs = (p.unsqueeze(-1) * F.softmax(step_logits.float(), dim=-1)).sum(0)
        return probs.clamp_min(1e-12).log()
    return F.log_softmax(first.float(), dim=-1)


def token_byte_lengths(tok, vocab_size: int) -> torch.Tensor:
    """Lookup table: token id -> number of bytes it represents.

    Special tokens map to 0. They are structural markers, not text, so counting
    the literal bytes of "<|endoftext|>" would inflate the byte total and
    silently flatter the bits-per-byte number.
    """
    special = set(getattr(tok, "specials", {}).values())
    lengths = torch.ones(vocab_size, dtype=torch.long)
    if hasattr(tok, "token_bytes"):
        for i in range(vocab_size):
            lengths[i] = 0 if i in special else len(tok.token_bytes(i))
    return lengths


# --------------------------------------------------------------------------
# Bits per byte on a held-out split
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bits_per_byte(model, ids, tok, block_size: int, stride: Optional[int] = None,
                           batch_size: int = 8, max_tokens: Optional[int] = None,
                           device: str = "cpu", vocab_size: Optional[int] = None
                           ) -> Dict[str, float]:
    """Compression of a held-out split, in bits per byte.

    Windows overlap by `block_size - stride` and only the *unseen* tail of each
    window is scored, so every token is counted exactly once and almost all of
    them are predicted with a full context. Scoring non-overlapping windows
    instead would grade every token that lands at the start of a window with
    almost no context, which makes a model look worse than it is -- by an amount
    that depends on block_size, so the number stops being comparable.
    """
    model.eval()
    if not torch.is_tensor(ids):
        ids = torch.from_numpy(ids.astype("int64")) if hasattr(ids, "astype") \
            else torch.tensor(ids, dtype=torch.long)
    ids = ids.long()
    n = len(ids)
    if max_tokens:
        n = min(n, int(max_tokens))
        ids = ids[:n]
    if n < block_size + 1:
        raise ValueError(f"split has {n} tokens, needs > block_size ({block_size})")

    stride = stride or max(block_size // 2, 1)
    vocab_size = vocab_size or int(ids.max().item()) + 1
    blen = token_byte_lengths(tok, max(vocab_size, int(ids.max().item()) + 1)).to(device)

    total_nats = torch.zeros((), dtype=torch.float64)
    total_bytes = torch.zeros((), dtype=torch.long)
    total_tokens = 0

    starts = list(range(0, n - block_size - 1, stride))
    batch: List[int] = []

    def flush(chunk: List[int]) -> None:
        nonlocal total_nats, total_bytes, total_tokens
        if not chunk:
            return
        x = torch.stack([ids[s:s + block_size] for s in chunk]).to(device)
        y = torch.stack([ids[s + 1:s + block_size + 1] for s in chunk]).to(device)
        logp = model_log_probs(model, x, vocab_size)
        nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)          # (B, T)

        keep = torch.zeros_like(nll, dtype=torch.bool)
        for i, s in enumerate(chunk):
            # first window is scored whole; later ones only past the overlap
            lo = 0 if s == 0 else block_size - stride
            keep[i, lo:] = True
        nbytes = blen[y]
        keep &= nbytes > 0                       # special tokens score nothing
        total_nats += nll[keep].double().sum().cpu()
        total_bytes += nbytes[keep].sum().cpu()
        total_tokens += int(keep.sum().item())

    for s in starts:
        batch.append(s)
        if len(batch) == batch_size:
            flush(batch)
            batch = []
    flush(batch)

    if total_bytes.item() == 0:
        raise ValueError("no scorable bytes -- is the tokenizer the right one?")
    nats = total_nats.item()
    n_bytes = int(total_bytes.item())
    return {
        "bits_per_byte": nats / math.log(2) / n_bytes,
        "bits_per_token": nats / math.log(2) / total_tokens,
        "nats_per_token": nats / total_tokens,
        "perplexity": math.exp(nats / total_tokens),
        "tokens": total_tokens,
        "bytes": n_bytes,
    }


# --------------------------------------------------------------------------
# Multiple choice (HellaSwag / ARC / PIQA style)
# --------------------------------------------------------------------------

@torch.no_grad()
def score_continuations(model, pairs: Sequence, block_size: int, device: str = "cpu",
                        batch_size: int = 8, vocab_size: Optional[int] = None
                        ) -> List[float]:
    """Total log P(continuation | context) for each (context_ids, cont_ids) pair.

    Only continuation positions are scored. Over-long sequences are truncated
    from the **left**, keeping the continuation and as much recent context as
    fits -- truncating from the right would delete the thing being scored.

    Batches are padded on the **right**. These models take no attention mask, so
    left-padding would put real token ids into the context of every scored
    position and quietly corrupt the comparison between choices of different
    lengths. Right-padding is safe for free: under a causal mask, position t
    cannot see anything after it, so trailing pad tokens are unreachable.
    """
    model.eval()
    out: List[float] = []
    for i in range(0, len(pairs), batch_size):
        chunk = list(pairs[i:i + batch_size])
        seqs, n_cont = [], []
        for ctx, cont in chunk:
            ctx, cont = list(ctx), list(cont)
            if not cont:
                seqs.append(list(ctx) or [0, 0])
                n_cont.append(0)
                continue
            seq = (ctx + cont)[-(block_size + 1):]
            if len(seq) < 2:                      # need at least one prediction
                seq = [0] + seq
            seqs.append(seq)
            n_cont.append(min(len(cont), len(seq) - 1))
        width = max(len(s) for s in seqs)
        x = torch.zeros(len(seqs), width - 1, dtype=torch.long)
        y = torch.zeros(len(seqs), width - 1, dtype=torch.long)
        mask = torch.zeros(len(seqs), width - 1, dtype=torch.bool)
        for j, seq in enumerate(seqs):
            L = len(seq)
            x[j, :L - 1] = torch.tensor(seq[:-1])
            y[j, :L - 1] = torch.tensor(seq[1:])
            if n_cont[j]:
                # y[t] holds seq[t+1], so the continuation's targets are the
                # last n_cont positions of the unpadded region.
                mask[j, L - 1 - n_cont[j]:L - 1] = True
        logp = model_log_probs(model, x.to(device), vocab_size)
        lp = logp.gather(-1, y.to(device).unsqueeze(-1)).squeeze(-1)
        out.extend((lp * mask.to(device)).sum(-1).double().cpu().tolist())
    return out


def multiple_choice(model, tok, examples: Sequence[Dict], block_size: int,
                    device: str = "cpu", batch_size: int = 8,
                    vocab_size: Optional[int] = None) -> Dict[str, float]:
    """Zero-shot accuracy on `{context, choices, label}` examples.

    Two numbers, because the naive one is biased:

        acc       argmax of the raw total log-probability, which systematically
                  prefers *short* continuations -- every extra token adds
                  another negative number.
        acc_norm  argmax of log-probability per byte, which removes that bias.

    `acc_norm` is the figure HellaSwag results are normally quoted with, and the
    one to compare against other models. Below ~100M parameters expect both to
    sit near chance; that is a fact about the scale, not a bug in the harness.
    """
    pairs, spans, byte_lens = [], [], []
    for ex in examples:
        ctx_ids = tok.encode(ex["context"])
        start = len(pairs)
        for choice in ex["choices"]:
            cont_ids = tok.encode(choice)
            pairs.append((ctx_ids, cont_ids))
            byte_lens.append(max(len(choice.encode("utf-8")), 1))
        spans.append((start, len(pairs)))

    scores = score_continuations(model, pairs, block_size, device, batch_size, vocab_size)

    n_correct = n_correct_norm = 0
    for ex, (lo, hi) in zip(examples, spans):
        raw = scores[lo:hi]
        norm = [s / b for s, b in zip(raw, byte_lens[lo:hi])]
        n_correct += int(max(range(len(raw)), key=raw.__getitem__) == ex["label"])
        n_correct_norm += int(max(range(len(norm)), key=norm.__getitem__) == ex["label"])
    n = max(len(examples), 1)
    chance = sum(1.0 / max(len(ex["choices"]), 1) for ex in examples) / n
    return {"acc": n_correct / n, "acc_norm": n_correct_norm / n,
            "chance": chance, "n": len(examples)}


# --------------------------------------------------------------------------
# Code: does the output actually parse
# --------------------------------------------------------------------------

@torch.no_grad()
def halting_stats(model, ids, block_size: int, device: str, batch_size: int = 4,
                  max_batches: int = 16) -> Optional[Dict[str, float]]:
    """Does the adaptive depth actually adapt? Returns None if there is no halting.

    This is the experiment that decides whether the architecture's central claim
    is supported, and it is separate from every other metric here. Recurrent
    depth plus PonderNet halting is a bet that **hard tokens want more compute
    than easy ones**. Nothing in the loss checks that bet: a model whose halting
    collapses to depth 1, or saturates at max_loops for every token, will still
    train to a perfectly respectable perplexity. It has simply become a fixed-
    depth model that pays for a halting head.

    Three numbers decide it:

        depth_std        0 means the depth dial is dead. Not adaptive at all.
        frac_at_floor    all tokens exiting immediately -- the loops are unused
        frac_at_ceiling  all tokens running to max -- halting learned nothing
        depth_nll_corr   the one that matters. Positive means tokens the model
                         finds hard (high NLL) get more depth, which is the
                         thesis. Near zero means depth varies but not with
                         difficulty, i.e. it is noise wearing a mechanism's hat.

    A high `depth_nll_corr` is not proof the mechanism *helps* -- that needs the
    dense control at matched inference cost. It is proof the mechanism is alive.
    """
    n_windows = (len(ids) - 1) // block_size
    if n_windows < 1:
        return None
    depths: List[torch.Tensor] = []
    nlls: List[torch.Tensor] = []
    n_steps = 0
    model.eval()
    for b0 in range(0, min(n_windows, max_batches * batch_size), batch_size):
        rows = []
        for i in range(b0, min(b0 + batch_size, n_windows)):
            rows.append(ids[i * block_size: i * block_size + block_size + 1])
        if not rows:
            break
        win = torch.tensor([list(r) for r in rows], dtype=torch.long, device=device)
        x, y = win[:, :-1], win[:, 1:]
        out = model(x, y)
        # Only the halting models return (logits, loss, extras{"p": ...}).
        if not (isinstance(out, tuple) and len(out) == 3
                and isinstance(out[2], dict) and "p" in out[2]):
            return None
        p = out[2]["p"]                                   # (steps, B, T)
        n_steps = p.shape[0]
        idx = torch.arange(1, n_steps + 1, device=p.device,
                           dtype=p.dtype).view(-1, 1, 1)
        depths.append((p * idx).sum(0).float().reshape(-1).cpu())
        logits = out[0]
        v = logits.shape[-1]
        nlls.append(F.cross_entropy(logits.reshape(-1, v), y.reshape(-1),
                                    reduction="none").float().cpu())
    if not depths:
        return None
    d, l = torch.cat(depths), torch.cat(nlls)
    dc, lc = d - d.mean(), l - l.mean()
    denom = float(dc.norm() * lc.norm())
    corr = float((dc * lc).sum() / denom) if denom > 1e-9 else 0.0
    hist = torch.histc(d, bins=n_steps, min=1.0, max=float(n_steps))
    return {
        "max_loops": n_steps,
        "depth_mean": float(d.mean()),
        "depth_std": float(d.std()),
        "depth_min": float(d.min()),
        "depth_max": float(d.max()),
        "frac_at_floor": float((d <= 1.05).float().mean()),
        "frac_at_ceiling": float((d >= n_steps - 0.05).float().mean()),
        "depth_nll_corr": corr,
        "histogram": [float(h) / max(float(hist.sum()), 1.0) for h in hist],
        "tokens": int(d.numel()),
    }


def parse_rate(samples: Sequence[str], language: str = "python") -> Dict[str, float]:
    """Fraction of generated samples that are syntactically valid.

    This is the cheap code metric that actually moves at ~50M parameters, where
    HumanEval pass@1 is still a flat zero and tells you nothing. Python uses the
    real grammar via `ast.parse`; other languages fall back to delimiter
    balance, which is a much weaker proxy and is reported under its own name so
    the two are never confused.
    """
    if not samples:
        return {"parse_rate": 0.0, "n": 0, "method": "none"}
    if language == "python":
        ok = 0
        for s in samples:
            try:
                ast.parse(s)
                ok += 1
            except (SyntaxError, ValueError, MemoryError, RecursionError):
                pass
        return {"parse_rate": ok / len(samples), "n": len(samples), "method": "ast"}
    ok = sum(1 for s in samples if balanced_delimiters(s))
    return {"parse_rate": ok / len(samples), "n": len(samples),
            "method": "delimiter-balance (weak proxy, not a parser)"}


def balanced_delimiters(text: str) -> bool:
    """Are (), [] and {} balanced, ignoring those inside strings and comments?

    Deliberately crude: it understands `//` and `/* */` comments and double- and
    single-quoted strings with backslash escapes, which covers Rust and C-family
    code well enough to be a signal, and nothing more.
    """
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: List[str] = []
    i, n = 0, len(text)
    quote: Optional[str] = None
    while i < n:
        c = text[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c in "\"'":
            quote = c
        elif c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        elif c in "([{":
            stack.append(c)
        elif c in pairs:
            if not stack or stack.pop() != pairs[c]:
                return False
        i += 1
    return not stack and quote is None
