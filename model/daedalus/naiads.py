"""Naiads: many springs, each holding its own water -- a mixture of memories.

Mnemosyne is one gist bank: every segment writes into the same vectors, so
unrelated context interferes. Naiads splits that into `n_banks` independent
Mnemosyne-style states and lets Apollo (moe.Router) decide, per segment, which
top-k banks to read from and write to.

    score the segment against all banks  ->  top-k  ->  softmax over just those k
    selected banks    : state <- update(state, gist of this segment)
    unselected banks  : state unchanged, bit for bit
    read              : router-weighted combination of the selected banks

The bit-exactness of unselected banks is the whole point -- it is what keeps a
segment about one subject from smearing over a bank that holds another. The same
Switch-Transformer aux loss that keeps the Muses balanced keeps one bank from
absorbing all the traffic (`load_balance_loss`).

`n_banks=1, top_k=1` reduces to the single-bank behaviour Mnemosyne already had.
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .memory import Mnemosyne
from .moe import Router, load_balance_loss


class Naiads(nn.Module):
    """`n_banks` independently-updated gist banks with top-k segment routing.

    forward(x, state) -> (readout, new_state, scores)
        x        (B, T, C)          the segment
        state    (B, n_banks, G, C) or None to start from the learned init
        readout  (B, T, C)          what the segment reads back from memory
        scores   (B, 1, n_banks)    router logits, shaped for `load_balance_loss`

    The returned state is a *new* tensor; rows for banks this segment did not
    select are copied through unchanged (`torch.where`), never recomputed.
    """

    def __init__(self, n_embd: int, n_gist: int = 16, n_head: int = 4,
                 n_banks: int = 4, top_k: int = 2):
        super().__init__()
        assert 1 <= top_k <= n_banks, "top_k must be between 1 and n_banks"
        self.n_banks, self.top_k, self.n_gist = n_banks, top_k, n_gist
        self.compress = nn.ModuleList([Mnemosyne(n_embd, n_gist, n_head) for _ in range(n_banks)])
        self.router = Router(n_embd, n_banks, top_k)
        self.init_state = nn.Parameter(torch.randn(n_banks, n_gist, n_embd) * 0.02)
        self.read = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ln = nn.LayerNorm(n_embd)

    def initial_state(self, batch: int, device=None, dtype=None) -> torch.Tensor:
        s = self.init_state.unsqueeze(0).expand(batch, -1, -1, -1)
        return s.to(device=device, dtype=dtype) if device is not None else s

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, c = x.shape
        if state is None:
            state = self.initial_state(b, x.device, x.dtype)

        # One routing decision per segment, so the router sees a segment summary.
        summary = x.mean(dim=1, keepdim=True)                     # (B, 1, C)
        idx, w, scores = self.router(summary)                     # (B,1,k), (B,1,k), (B,1,n_banks)
        idx, w = idx.squeeze(1), w.squeeze(1)                     # (B, k)

        new_banks, readouts = [], x.new_zeros(b, t, c)
        for e in range(self.n_banks):
            sel = (idx == e)                                      # (B, k)
            chosen = sel.any(-1)                                  # (B,)
            prev = state[:, e]                                    # (B, G, C)
            if not bool(chosen.any()):
                new_banks.append(prev)                            # untouched: pass through
                continue
            updated = self.ln(prev + self.compress[e](x))         # write this segment's gist
            gate = chosen.view(b, 1, 1)
            new_banks.append(torch.where(gate, updated, prev))    # exact copy where not chosen

            r, _ = self.read(x, updated, updated)                 # (B, T, C)
            weight = (w * sel).sum(-1).view(b, 1, 1)              # router weight, 0 if not chosen
            readouts = readouts + r * weight

        return readouts, torch.stack(new_banks, dim=1), scores

    def aux_loss(self, scores: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer load-balancing loss over the banks (1.0 = balanced)."""
        idx = scores.topk(self.top_k, -1)[1]
        return load_balance_loss(scores, idx, self.n_banks)[0]
