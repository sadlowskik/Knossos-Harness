"""Ariadne: per-token adaptive halting for the recurrent-depth core.

Implements PonderNet (Banino et al., 2021): a learned halting head produces,
at each loop step, the probability of halting there. Training runs all steps
and weights each step's loss by its halting probability (a differentiable
expectation), plus a KL regularizer toward a geometric prior that sets the
average compute budget. At inference the model can halt early per token.

Related: Adaptive Computation Time (Graves, 2016); Mixture-of-Recursions
(Bae et al., 2025).
"""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Embeddings, Block


class Ariadne(nn.Module):
    """Recurrent-depth core with a halting head that allocates depth per token.

    forward() returns:
        p:      (n_steps, B, T) halting distribution (sums to 1 over steps)
        logits: (n_steps, B, T, vocab) per-step output logits
    Use `ponder_loss` to train, and `expected_steps` to inspect allocation.
    """

    def __init__(self, vocab_size: int = 256, n_embd: int = 128, n_head: int = 4,
                 core_layers: int = 3, max_loops: int = 8, block_size: int = 128):
        super().__init__()
        self.emb = Embeddings(vocab_size, n_embd, block_size)
        self.core = nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(core_layers)])
        self.halt = nn.Linear(n_embd, 1)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.max_loops = max_loops
        self.block_size = block_size

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.emb(idx)
        still = torch.ones(idx.shape, device=idx.device)   # P(not yet halted)
        p_list, logits_list = [], []
        for n in range(1, self.max_loops + 1):
            for blk in self.core:
                x = blk(x)
            if n < self.max_loops:
                lam = torch.sigmoid(self.halt(x)).squeeze(-1)          # P(halt here | not yet)
            else:
                lam = torch.ones(idx.shape, device=idx.device)         # force halt at the end
            p_list.append(still * lam)                                 # P(halt exactly at n)
            still = still * (1 - lam)
            logits_list.append(self.lm_head(self.ln_f(x)))
        return torch.stack(p_list, 0), torch.stack(logits_list, 0)


def ponder_loss(p: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor,
                lambda_prior: float = 0.2, beta: float = 0.01
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PonderNet loss = expected reconstruction CE + beta * KL(p || geometric prior).

    `lambda_prior` sets the target mean number of steps (~1/lambda_prior);
    `beta` is the accuracy-vs-compute dial. Returns (total, L_rec, L_kl).
    """
    n_steps, b, t, v = logits.shape
    ce = torch.stack([
        F.cross_entropy(logits[n].reshape(-1, v), targets.reshape(-1), reduction="none").reshape(b, t)
        for n in range(n_steps)
    ], dim=0)
    l_rec = (p * ce).sum(0).mean()

    steps = torch.arange(1, n_steps + 1, device=p.device, dtype=torch.float)
    prior = lambda_prior * (1 - lambda_prior) ** (steps - 1)
    prior = (prior / prior.sum()).view(n_steps, 1, 1)
    eps = 1e-9
    l_kl = (p * (torch.log(p + eps) - torch.log(prior + eps))).sum(0).mean()
    return l_rec + beta * l_kl, l_rec, l_kl


def expected_steps(p: torch.Tensor) -> torch.Tensor:
    """Per-token expected halting step (B, T) from a halting distribution."""
    n_steps = p.shape[0]
    idx = torch.arange(1, n_steps + 1, device=p.device).view(-1, 1, 1)
    return (p * idx).sum(0)
