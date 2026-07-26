"""Rotary Position Embeddings (RoPE, Su et al. 2021).

Instead of adding a learned vector per absolute position, RoPE rotates query and
key vectors by an angle proportional to their position. The dot product of a
rotated query at position m and rotated key at position n then depends only on
the relative offset m - n, giving relative-position awareness that extrapolates
to lengths not seen in training.
"""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rope_cache(seq_len: int, head_dim: int, base: int = 10000
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), theta)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


class RoPEAttention(nn.Module):
    """Multi-head causal self-attention with rotary positions.

    The per-head dimension (n_embd // n_head) must be even.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, base: int = 10000):
        super().__init__()
        assert (n_embd // n_head) % 2 == 0, "head dim must be even for RoPE"
        self.n_head, self.hd = n_head, n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        cos, sin = build_rope_cache(block_size, self.hd, base)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.hd).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.hd).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.hd).transpose(1, 2)
        q = apply_rope(q, self.cos[:t], self.sin[:t])
        k = apply_rope(k, self.cos[:t], self.sin[:t])
        att = (q @ k.transpose(-2, -1)) * (self.hd ** -0.5)
        att = att.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        out = (F.softmax(att, dim=-1) @ v).transpose(1, 2).reshape(b, t, c)
        return self.proj(out)
