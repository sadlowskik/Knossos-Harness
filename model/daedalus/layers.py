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
    """Several attention heads in parallel, concatenated and projected."""

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


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

    `mixer` selects how tokens talk to each other: "softmax" (default, standard
    causal self-attention) or "moirai" (gated fast-weight linear attention, see
    moirai.MoiraiMixer). Both are (B, T, C) -> (B, T, C), so they are
    interchangeable; only "softmax" keeps a growing KV cache.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, mixer: str = "softmax"):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        if mixer == "softmax":
            self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        elif mixer == "moirai":
            from .moirai import MoiraiMixer          # local: avoids an import cycle
            self.attn = MoiraiMixer(n_embd, n_head, block_size)
        else:
            raise ValueError(f"unknown mixer: {mixer!r} (expected 'softmax' or 'moirai')")
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
