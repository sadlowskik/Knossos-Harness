"""Moirai: the Fates -- they spin, measure, and cut the thread of memory.

A gated fast-weight sequence mixer (Gated DeltaNet, Yang et al. 2024; delta rule
from Schlag et al. 2021 "Linear Transformers Are Secretly Fast Weight
Programmers"). Instead of a KV cache that grows with sequence length, each head
carries one fixed-size fast-weight matrix `W` (d_value x d_key) that is
rewritten at every token:

    v_hat_t = W_{t-1} k_t          what is currently stored under this key
    r_t     = v_t - v_hat_t        the residual: what is new about this write
    W_t     = b_t * W_{t-1} + w_t * (r_t (x) k_t)
    y_t     = W_t q_t

Two decoupled channel-wise gates, both read off the token:

    Clotho  -- `b_t`, the erase gate: how much of the old thread survives.
    Lachesis -- `w_t`, the write gate: how strongly the new residual is committed.

Decoupling them is what lets the model drop stale content *before* deciding how
hard to write, so new writes can cannibalize the space that low-value ones held.
Set `tied=True` for plain Gated DeltaNet (b_t = w_t).

State is O(n_head * d_head^2) regardless of sequence length, and the scan is
strictly causal by construction -- no mask needed.

This is a v1: a plain sequential `for t in range(T)` scan. That is fine at this
project's toy scale (byte-level, single T4). The chunk-wise parallel form is a
later optimization, not a correctness issue.
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoiraiMixer(nn.Module):
    """Gated fast-weight token mixer. Drop-in for `MultiHeadAttention`.

    Maps (B, T, n_embd) -> (B, T, n_embd), same as the softmax mixer it
    replaces. `block_size` is accepted and ignored (there is no mask and no
    positional table to size) so the two are interchangeable at the call site.

    Args:
        tied: if True, one gate drives both erase and write (plain Gated
            DeltaNet). If False (default), erase and write are separate
            projections.
        erase_bias: init bias of the erase gate. The default (+4.0, sigmoid
            ~0.982) makes the state forget slowly at init, which is what keeps
            the recurrence from washing out before it has learned anything.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: Optional[int] = None,
                 tied: bool = False, erase_bias: float = 4.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must divide evenly into heads"
        self.n_head, self.hd, self.tied = n_head, n_embd // n_head, tied

        self.q = nn.Linear(n_embd, n_embd, bias=False)
        self.k = nn.Linear(n_embd, n_embd, bias=False)
        self.v = nn.Linear(n_embd, n_embd, bias=False)

        self.erase = nn.Linear(n_embd, n_embd)          # Clotho: keep-fraction
        nn.init.zeros_(self.erase.weight)
        nn.init.constant_(self.erase.bias, erase_bias)
        if not tied:
            self.write = nn.Linear(n_embd, n_embd)      # Lachesis: write strength
            nn.init.zeros_(self.write.weight)
            nn.init.zeros_(self.write.bias)              # sigmoid(0) = 0.5

        self.ln = nn.LayerNorm(n_embd)                  # Atropos: cut the outliers
        self.proj = nn.Linear(n_embd, n_embd)

    def _heads(self, t: torch.Tensor, b: int, n: int) -> torch.Tensor:
        return t.view(b, n, self.n_head, self.hd).transpose(1, 2)   # (B, H, T, hd)

    def forward(self, x: torch.Tensor, return_state: bool = False):
        b, n, c = x.shape
        # L2-normalising keys keeps ||k||=1, which is what makes the delta rule a
        # well-behaved read-modify-write instead of an unbounded accumulation.
        k = F.normalize(self._heads(self.k(x), b, n), dim=-1)
        q = F.normalize(self._heads(self.q(x), b, n), dim=-1)
        v = self._heads(self.v(x), b, n)

        gb = torch.sigmoid(self._heads(self.erase(x), b, n))
        gw = gb if self.tied else torch.sigmoid(self._heads(self.write(x), b, n))

        W = x.new_zeros(b, self.n_head, self.hd, self.hd)            # (B, H, d_v, d_k)
        outs = []
        for t in range(n):
            k_t, v_t, q_t = k[:, :, t], v[:, :, t], q[:, :, t]       # (B, H, hd)
            v_hat = torch.einsum("bhvk,bhk->bhv", W, k_t)            # current content
            r_t = v_t - v_hat                                        # delta-rule residual
            outer = r_t.unsqueeze(-1) * k_t.unsqueeze(-2)            # (B, H, d_v, d_k)
            W = gb[:, :, t].unsqueeze(-1) * W + gw[:, :, t].unsqueeze(-1) * outer
            outs.append(torch.einsum("bhvk,bhk->bhv", W, q_t))

        y = torch.stack(outs, dim=2).transpose(1, 2).reshape(b, n, c)
        out = self.proj(self.ln(y))
        return (out, W) if return_state else out
