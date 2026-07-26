"""Proteus: the sea-god who changes his own shape -- weights that rewrite themselves.

The most experimental piece in the repo, and deliberately kept off the main line:
`DaedalusProteus` is its own model class and touches none of Daedalus, Labyrinth,
DaedalusMoE, UnifiedDaedalus, DaedalusFull or DaedalusFullAdaptive.

Two levels, both using Moirai's delta rule, but pointed at the layer's *own*
transform instead of a separate memory:

Level 1 -- `SelfModifyingLinear` (self_referential=False).
    A slow weight `W0` (learned by gradient descent, frozen at run time) plus a
    fast weight `F` that the layer edits itself, token by token:

        v_hat_t = F_{t-1} k_t ; r_t = v_t - v_hat_t
        F_t     = b_t * F_{t-1} + beta_t * (r_t (x) k_t)
        y_t     = (W0 + F_t) x_t

    `k_t`, `v_t`, `beta_t` come from ordinary (slow) projections of the token.

Level 2 -- fully self-referential (self_referential=True), the SRWM of Irie et
    al. 2022 ("A Modern Self-Referential Weight Matrix That Learns to Modify
    Itself"), building on Schlag et al. 2021. One matrix `W` produces *everything*
    in a single read -- the output, the query, the key, and the learning rate:

        [y_t, q_t, k_t, beta_t] = W_t x_t
        v_t     = W_t q_t          (the matrix asks itself what to write)
        v_hat_t = W_t k_t          (what is already stored there)
        W_{t+1} = b_t * W_t + beta_t * (v_t - v_hat_t) (x) k_t

    The rows that generate the update are inside `W`, so the update rule modifies
    the update rule. That is the interesting part and also the unstable part.

Honest status: the known failure mode is unbounded growth of ||W|| -- the layer
teaches itself to write ever harder. Two guards are on by default: L2-normalised
q/k (so the delta rule stays a bounded read-modify-write) and Moirai's erase gate
`b_t` initialised near 1.0 so forgetting is available but off at init. Watch
`weight_norms()` over training, not just the loss; the norm diverges before the
loss does. See tests/test_proteus.py and scripts/proteus_probe.py.
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Embeddings, FeedForward


class SelfModifyingLinear(nn.Module):
    """A linear map that edits its own weights as it reads the sequence.

    (B, T, n_embd) -> (B, T, n_embd), causal by construction (the state at step
    t depends only on tokens <= t), so it is a drop-in sequence mixer like
    MoiraiMixer.
    """

    def __init__(self, n_embd: int, n_head: int = 4, self_referential: bool = False,
                 erase_bias: float = 4.0, beta_scale: float = 1.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.hd = n_head, n_embd // n_head
        self.self_referential, self.beta_scale = self_referential, beta_scale

        if self_referential:
            # One matrix per head, rows: [output | query | key | beta].
            self.rows = 3 * self.hd + 1
            w = torch.randn(n_head, self.rows, self.hd) / (self.hd ** 0.5)
            self.W0 = nn.Parameter(w)
            self.erase = nn.Linear(n_embd, n_head)          # one erase gate per head
        else:
            self.W0 = nn.Parameter(torch.randn(n_head, self.hd, self.hd) / (self.hd ** 0.5))
            self.k = nn.Linear(n_embd, n_embd, bias=False)
            self.v = nn.Linear(n_embd, n_embd, bias=False)
            self.beta = nn.Linear(n_embd, n_head)
            self.erase = nn.Linear(n_embd, n_head)
        nn.init.zeros_(self.erase.weight)
        nn.init.constant_(self.erase.bias, erase_bias)      # sigmoid(4) ~ 0.982: slow forgetting

        self.ln = nn.LayerNorm(n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def _heads(self, t: torch.Tensor, b: int, n: int) -> torch.Tensor:
        return t.view(b, n, self.n_head, self.hd).transpose(1, 2)

    def forward(self, x: torch.Tensor, return_norms: bool = False):
        b, n, c = x.shape
        xh = self._heads(x, b, n)                                    # (B, H, T, hd)
        gb = torch.sigmoid(self.erase(x)).transpose(1, 2)            # (B, H, T)
        norms: List[float] = []

        if self.self_referential:
            W = self.W0.unsqueeze(0).expand(b, -1, -1, -1)           # (B, H, rows, hd)
            outs = []
            for t in range(n):
                read = torch.einsum("bhrd,bhd->bhr", W, xh[:, :, t])  # one read, four uses
                y_t, q_t, k_t, beta_t = read.split([self.hd, self.hd, self.hd, 1], dim=-1)
                q_t = F.normalize(q_t, dim=-1)
                k_t = F.normalize(k_t, dim=-1)
                v_t = torch.einsum("bhrd,bhd->bhr", W, q_t)          # what to write
                v_hat = torch.einsum("bhrd,bhd->bhr", W, k_t)        # what is stored
                beta = torch.sigmoid(beta_t) * self.beta_scale       # (B, H, 1)
                delta = (v_t - v_hat).unsqueeze(-1) * k_t.unsqueeze(-2)
                W = gb[:, :, t].view(b, self.n_head, 1, 1) * W + beta.unsqueeze(-1) * delta
                outs.append(y_t)
                if return_norms:
                    norms.append(W.norm().item())
        else:
            k = F.normalize(self._heads(self.k(x), b, n), dim=-1)
            v = self._heads(self.v(x), b, n)
            beta = torch.sigmoid(self.beta(x)).transpose(1, 2) * self.beta_scale   # (B, H, T)
            Fw = x.new_zeros(b, self.n_head, self.hd, self.hd)       # fast weights start empty
            outs = []
            for t in range(n):
                k_t, v_t, x_t = k[:, :, t], v[:, :, t], xh[:, :, t]
                v_hat = torch.einsum("bhvd,bhd->bhv", Fw, k_t)
                delta = (v_t - v_hat).unsqueeze(-1) * k_t.unsqueeze(-2)
                Fw = (gb[:, :, t].view(b, self.n_head, 1, 1) * Fw
                      + beta[:, :, t].view(b, self.n_head, 1, 1) * delta)
                eff = self.W0.unsqueeze(0) + Fw                      # slow + self-written fast
                outs.append(torch.einsum("bhvd,bhd->bhv", eff, x_t))
                if return_norms:
                    norms.append(Fw.norm().item())

        y = torch.stack(outs, dim=2).transpose(1, 2).reshape(b, n, c)
        out = self.proj(self.ln(y))
        return (out, norms) if return_norms else out


class ProteusBlock(nn.Module):
    """Pre-norm block whose token mixer is a self-modifying weight matrix."""

    def __init__(self, n_embd: int, n_head: int, self_referential: bool = False):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.mix = SelfModifyingLinear(n_embd, n_head, self_referential)
        self.ff = FeedForward(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mix(self.ln1(x))
        return x + self.ff(self.ln2(x))


class DaedalusProteus(nn.Module):
    """Decoder-only LM whose token mixer rewrites its own weights at run time.

    Isolated on purpose: this shares nothing with the main model line, so an
    instability here cannot contaminate DaedalusFull. forward() matches the
    dense baseline's signature -- (logits, ce_loss) -- so train.py, generate.py
    and the checkpoint format work unchanged.

    Set `self_referential=True` for the full SRWM (the update rule modifies the
    rows that produce the update). Try it only after the level-1 version is
    stable on your data.
    """

    def __init__(self, vocab_size: int = 256, n_embd: int = 128, n_head: int = 4,
                 n_layer: int = 3, block_size: int = 128, self_referential: bool = False):
        super().__init__()
        self.emb = Embeddings(vocab_size, n_embd, block_size)
        self.blocks = nn.ModuleList([
            ProteusBlock(n_embd, n_head, self_referential) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size, self.self_referential = block_size, self_referential

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            b, t, v = logits.shape
            loss = F.cross_entropy(logits.view(b * t, v), targets.reshape(b * t))
        return logits, loss

    @torch.no_grad()
    def weight_norms(self, idx: torch.Tensor) -> List[float]:
        """Per-token Frobenius norm of the self-written matrix in the first block.

        The diagnostic to watch: divergence shows up here before it shows up in
        the loss.
        """
        x = self.emb(idx)
        return self.blocks[0].mix(self.blocks[0].ln1(x), return_norms=True)[1]
