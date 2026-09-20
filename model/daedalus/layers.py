"""Core transformer building blocks shared across all Daedalus models.

Reference: "Attention Is All You Need" (Vaswani et al., 2017); pre-norm
placement follows Xiong et al. (2020); residual connections follow He et al.
(2015).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embeddings(nn.Module):
    """Token embeddings + learned absolute position embeddings.

    Learned-absolute positions are used for clarity. RoPE (Su et al., 2021) is
    the recommended upgrade: it is relative, extrapolates to longer contexts,
    and interacts better with recurrent depth.
    """

    def __init__(self, vocab_size: int, n_embd: int, block_size: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        return self.tok_emb(idx) + self.pos_emb(torch.arange(t, device=idx.device))


class Head(nn.Module):
    """A single causal self-attention head."""

    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        t = x.shape[1]
        k, q, v = self.key(x), self.query(x), self.value(x)
        att = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)          # scaled scores
        att = att.masked_fill(self.tril[:t, :t] == 0, float("-inf"))    # causal mask
        att = F.softmax(att, dim=-1)
        out = att @ v
        return (out, att) if return_weights else out


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, concatenated and projected.

    Mathematically identical to stacking `Head` above, but computed as one fused
    QKV projection and dispatched to `F.scaled_dot_product_attention`, which
    picks a fused kernel (FlashAttention, Dao et al. 2022) when one is available.
    That matters at training scale for two reasons: the per-head Python loop is
    replaced by a single batched matmul, and Flash never materialises the
    (T, T) attention matrix, so memory stops growing quadratically with context.

    `Head` is kept as the readable reference implementation -- read that one to
    understand the mechanism, run this one.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head, self.hd = n_head, n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = dropout
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.hd).transpose(1, 2)   # (B, H, T, hd)
        k = k.view(b, t, self.n_head, self.hd).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(b, t, c))


class FeedForward(nn.Module):
    """Position-wise MLP with a `mult`x hidden expansion. In MoE models this is
    the module that is replaced by a mixture of experts."""

    def __init__(self, n_embd: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, mult * n_embd), nn.GELU(), nn.Linear(mult * n_embd, n_embd)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-norm transformer block: attention + MLP, each with a residual.

    Maps (B, T, n_embd) -> (B, T, n_embd), which is what makes it stackable AND
    loopable (see Labyrinth).

    `mixer` selects how tokens talk to each other:

        softmax         standard causal self-attention (default, unchanged)
        moirai          gated fast weights, one gate for erase and write
        moirai-untied   the same with the gates decoupled

    All three are (B, T, C) -> (B, T, C) and interchangeable; only "softmax"
    keeps a growing KV cache.

    `moirai` is the tied form -- plain Gated DeltaNet. The decoupled variant is
    reachable but is no longer what the plain name selects: at n=5 it measured
    nominally worse in both regimes and cost 5% more parameters
    (scripts/moirai_sweep.py). It stays available so the ablation can be re-run,
    not because it is recommended.
    """

    MIXERS = ("softmax", "moirai", "moirai-untied")

    def __init__(self, n_embd: int, n_head: int, block_size: int, mixer: str = "softmax"):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        if mixer == "softmax":
            self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        elif mixer in ("moirai", "moirai-untied"):
            from .moirai import MoiraiMixer          # local: avoids an import cycle
            self.attn = MoiraiMixer(n_embd, n_head, block_size,
                                    tied=mixer == "moirai")
        else:
            raise ValueError(
                f"unknown mixer: {mixer!r} (expected one of {', '.join(self.MIXERS)})")
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
