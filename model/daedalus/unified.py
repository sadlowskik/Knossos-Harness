"""UnifiedDaedalus: recurrent depth + mixture-of-experts.

Loops a shared MoE core, combining weight-sharing across depth (Labyrinth) with
sparse expert routing (Muses). This is the Mixture-of-Recursions design
(Bae et al., 2025): few unique parameters, large effective depth, sparse
activation per token.
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Embeddings
from .moe import MoEBlock, load_balance_loss


class UnifiedDaedalus(nn.Module):
    """Shared MoE core looped `n_loops` times.

    forward() returns (logits, ce_loss, aux_loss). The load-balancing aux loss
    is accumulated over every loop iteration and block. Pass `n_loops` to vary
    depth at inference (train with a variable loop count first).
    """

    def __init__(self, vocab_size: int = 256, n_embd: int = 128, n_head: int = 4,
                 core_layers: int = 2, n_loops: int = 4, block_size: int = 128,
                 n_experts: int = 8, top_k: int = 2, n_shared: int = 1,
                 hidden: Optional[int] = None):
        super().__init__()
        self.emb = Embeddings(vocab_size, n_embd, block_size)
        self.core = nn.ModuleList([
            MoEBlock(n_embd, n_head, block_size, n_experts, top_k, n_shared, hidden)
            for _ in range(core_layers)
        ])
        self.n_loops = n_loops
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size, self.n_experts, self.top_k = block_size, n_experts, top_k

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                n_loops: Optional[int] = None):
        r = n_loops if n_loops is not None else self.n_loops
        x = self.emb(idx)
        aux = x.new_zeros(())
        for _ in range(r):
            for blk in self.core:
                x, scores = blk(x)
                aux = aux + load_balance_loss(scores, scores.topk(self.top_k, -1)[1], self.n_experts)[0]
        logits = self.lm_head(self.ln_f(x))
        ce = None
        if targets is not None:
            b, t, v = logits.shape
            ce = F.cross_entropy(logits.view(b * t, v), targets.reshape(b * t))
        return logits, ce, aux
