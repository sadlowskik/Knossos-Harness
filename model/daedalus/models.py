"""Core Daedalus models: the dense baseline and the recurrent-depth Labyrinth."""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Embeddings, Block


class Daedalus(nn.Module):
    """Dense decoder-only transformer. The baseline every other model is
    measured against."""

    def __init__(self, vocab_size: int = 256, n_embd: int = 128, n_head: int = 4,
                 n_layer: int = 3, block_size: int = 128):
        super().__init__()
        self.emb = Embeddings(vocab_size, n_embd, block_size)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

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


class Labyrinth(nn.Module):
    """Recurrent-depth transformer: a shared core looped `n_loops` times.

    Structure: prelude (embeddings) -> shared core (a stack of `core_layers`
    blocks, looped) -> coda (final norm + head). Looping the shared core gives
    the effective depth of `core_layers * n_loops` while paying for only
    `core_layers` blocks of parameters.

    Pass a different `n_loops` at inference to scale test-time compute -- but
    only if trained with a *variable* loop count (see train.py --variable-loops),
    otherwise the model overfits to one depth and diverges at others.

    `mixer` chooses how the core mixes tokens: "softmax" (default, unchanged
    behaviour) or "moirai" (gated fast-weight linear attention). The mixer is an
    axis orthogonal to loop depth and expert routing.

    References: Universal Transformer (Dehghani et al., 2018); Huginn (Geiping
    et al., 2025); Ouro; Deep Equilibrium Models (Bai et al., 2019).
    """

    def __init__(self, vocab_size: int = 256, n_embd: int = 128, n_head: int = 4,
                 core_layers: int = 3, n_loops: int = 4, block_size: int = 128,
                 mixer: str = "softmax"):
        super().__init__()
        self.emb = Embeddings(vocab_size, n_embd, block_size)
        self.core = nn.ModuleList([Block(n_embd, n_head, block_size, mixer)
                                   for _ in range(core_layers)])
        self.n_loops = n_loops
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size, self.mixer = block_size, mixer

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                n_loops: Optional[int] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r = n_loops if n_loops is not None else self.n_loops
        x = self.emb(idx)
        for _ in range(r):                 # recurrent depth: re-apply the SAME core
            for blk in self.core:
                x = blk(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            b, t, v = logits.shape
            loss = F.cross_entropy(logits.view(b * t, v), targets.reshape(b * t))
        return logits, loss
