"""Mixture-of-Experts: the Muses (routed experts), Apollo (router), Themis
(always-on shared experts).

Fine-grained experts with an always-on shared expert follow DeepSeekMoE
(Dai et al., 2024). The router uses noisy top-k gating (Shazeer et al., 2017)
for exploration, and training adds a load-balancing auxiliary loss (Switch
Transformer, Fedus et al., 2021) to prevent expert collapse.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Embeddings, MultiHeadAttention


class Expert(nn.Module):
    """One small feed-forward expert (a Muse)."""

    def __init__(self, n_embd: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, hidden), nn.GELU(), nn.Linear(hidden, n_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Router(nn.Module):
    """Apollo: noisy top-k gating. Noise is applied only during training to give
    under-used experts a chance to be selected (exploration)."""

    def __init__(self, n_embd: int, n_experts: int, top_k: int, noise: float = 1.0):
        super().__init__()
        self.gate = nn.Linear(n_embd, n_experts, bias=False)
        self.noise = nn.Linear(n_embd, n_experts, bias=False)   # learned per-token noise scale
        self.n_experts, self.top_k, self.noise_eps = n_experts, top_k, noise

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clean = self.gate(x)
        if self.training:
            scores = clean + torch.randn_like(clean) * (F.softplus(self.noise(x)) * self.noise_eps)
        else:
            scores = clean
        vals, idx = scores.topk(self.top_k, -1)
        return idx, F.softmax(vals, -1), scores


def load_balance_loss(scores: torch.Tensor, idx: torch.Tensor, n_experts: int
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Switch-Transformer aux loss = N * sum_e (load_e * importance_e).

    Minimum 1.0 when perfectly balanced, approaches N when collapsed onto one
    expert. Returns (aux, load, importance).
    """
    importance = F.softmax(scores, dim=-1).mean(dim=(0, 1))                 # soft, differentiable
    load = F.one_hot(idx, n_experts).float().sum(dim=(0, 1, 2)) / idx.numel()  # hard fraction
    return n_experts * (load * importance).sum(), load, importance


class MoELayer(nn.Module):
    """Routed Muses (top-k) plus always-on Themis shared experts."""

    def __init__(self, n_embd: int, n_experts: int = 8, top_k: int = 2,
                 n_shared: int = 1, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or n_embd
        self.router = Router(n_embd, n_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embd, hidden) for _ in range(n_experts)])
        self.shared = nn.ModuleList([Expert(n_embd, hidden) for _ in range(n_shared)])
        self.top_k = top_k
        self.n_experts = n_experts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, c = x.shape
        idx, w, scores = self.router(x)
        out = torch.zeros_like(x)
        for se in self.shared:                                 # Themis: every token
            out = out + se(x)
        fx = x.reshape(-1, c)
        fidx = idx.reshape(-1, self.top_k)
        fw = w.reshape(-1, self.top_k)
        fout = torch.zeros_like(fx)
        for e, expert in enumerate(self.experts):              # sparse dispatch
            mask = (fidx == e)
            sel = mask.any(-1)
            if sel.any():
                we = (fw * mask).sum(-1)
                fout[sel] += expert(fx[sel]) * we[sel].unsqueeze(-1)
        return out + fout.reshape(b, t, c), scores


class MoEBlock(nn.Module):
    """Pre-norm block with attention + an MoE layer in place of the dense MLP."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, n_experts: int = 8,
                 top_k: int = 2, n_shared: int = 1, hidden: Optional[int] = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.moe = MoELayer(n_embd, n_experts, top_k, n_shared, hidden)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x))
        moe_out, scores = self.moe(self.ln2(x))
        return x + moe_out, scores


class DaedalusMoE(nn.Module):
    """Dense-depth transformer whose feed-forward is a mixture of experts.

    forward() returns (logits, ce_loss, aux_loss). Train with
    `ce_loss + alpha * aux_loss` (alpha ~ 0.01).
    """

    def __init__(self, vocab_size: int = 256, n_embd: int = 128, n_head: int = 4,
                 n_layer: int = 3, block_size: int = 128, n_experts: int = 8,
                 top_k: int = 2, n_shared: int = 1, hidden: Optional[int] = None):
        super().__init__()
        self.emb = Embeddings(vocab_size, n_embd, block_size)
        self.blocks = nn.ModuleList([
            MoEBlock(n_embd, n_head, block_size, n_experts, top_k, n_shared, hidden)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size, self.n_experts, self.top_k = block_size, n_experts, top_k

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        x = self.emb(idx)
        aux = x.new_zeros(())
        for blk in self.blocks:
            x, scores = blk(x)
            aux = aux + load_balance_loss(scores, scores.topk(self.top_k, -1)[1], self.n_experts)[0]
        logits = self.lm_head(self.ln_f(x))
        ce = None
        if targets is not None:
            b, t, v = logits.shape
            ce = F.cross_entropy(logits.view(b * t, v), targets.reshape(b * t))
        return logits, ce, aux
