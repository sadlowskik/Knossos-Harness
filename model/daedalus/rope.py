"""Rotary Position Embeddings (RoPE, Su et al. 2021).

Instead of adding a learned vector per absolute position, RoPE rotates query and
key vectors by an angle proportional to their position. The dot product of a
rotated query at position m and rotated key at position n then depends only on
the relative offset m - n, giving relative-position awareness that extrapolates
to lengths not seen in training.
"""
from __future__ import annotations
from typing import Optional, Tuple
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


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last dimension, in fp32 for stability under autocast."""
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight.float()).to(dt)


class RoPEAttention(nn.Module):
    """Multi-head causal self-attention with rotary positions.

    The per-head dimension (n_embd // n_head) must be even.

    The attention itself goes through `F.scaled_dot_product_attention` so a
    fused (FlashAttention) kernel can be used when available; RoPE is applied to
    q and k beforehand, which is exactly where it belongs -- the rotation only
    ever touches the score computation, never the values.

    Three optional behaviours, all off by default so existing checkpoints keep
    their shapes:

    `n_kv_head` -- grouped-query attention. Queries keep `n_head` heads while
    keys and values use fewer, so the KV cache shrinks by `n_head/n_kv_head`.
    At 45M this is irrelevant; at the scale where "runs on modest hardware" is a
    claim, the KV cache exceeds the weights and this is the binding constraint.

    `qk_norm` -- RMSNorm on q and k before the dot product. Attention logits can
    grow without bound across a shared-weight recurrence, because there are no
    per-iteration parameters to absorb the growth; normalising q and k caps the
    logit scale directly. This is the failure MuonClip's QK-clip addresses at
    scale, and recurrent depth is exactly the regime that invites it.

    `doc_ids` (per call) -- block-diagonal document masking. Packed corpora put
    several documents in one window, and plain causal attention lets a token
    attend across the boundary into an unrelated document. Passing segment ids
    restricts attention to the token's own document.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, base: int = 10000,
                 dropout: float = 0.0, n_kv_head: Optional[int] = None,
                 qk_norm: bool = False):
        super().__init__()
        assert (n_embd // n_head) % 2 == 0, "head dim must be even for RoPE"
        self.n_head, self.hd = n_head, n_embd // n_head
        self.n_kv_head = n_kv_head or n_head
        assert n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.n_rep = self.n_head // self.n_kv_head
        self.qkv = nn.Linear(n_embd, n_embd + 2 * self.n_kv_head * self.hd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = dropout
        self.q_norm = nn.Parameter(torch.ones(self.hd)) if qk_norm else None
        self.k_norm = nn.Parameter(torch.ones(self.hd)) if qk_norm else None
        cos, sin = build_rope_cache(block_size, self.hd, base)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def init_cache(self, batch_size: int, max_len: int, *, device=None,
                   dtype=None) -> dict:
        """Preallocate an append-only cache; avoids concatenating every token."""
        device = device or self.cos.device
        dtype = dtype or self.qkv.weight.dtype
        shape = (batch_size, self.n_kv_head, max_len, self.hd)
        return {"k_buf": torch.empty(shape, device=device, dtype=dtype),
                "v_buf": torch.empty(shape, device=device, dtype=dtype),
                "length": 0}

    def forward(self, x: torch.Tensor, doc_ids: Optional[torch.Tensor] = None,
                cache: Optional[dict] = None, pos_offset: int = 0) -> torch.Tensor:
        b, t, c = x.shape
        qs = self.n_head * self.hd
        ks = self.n_kv_head * self.hd
        q, k, v = self.qkv(x).split([qs, ks, ks], dim=2)
        q = q.view(b, t, self.n_head, self.hd).transpose(1, 2)
        k = k.view(b, t, self.n_kv_head, self.hd).transpose(1, 2)
        v = v.view(b, t, self.n_kv_head, self.hd).transpose(1, 2)
        if self.q_norm is not None:
            q = rms_norm(q, self.q_norm)
            k = rms_norm(k, self.k_norm)
        # RoPE is keyed to absolute position, so incremental decoding has to
        # rotate by the position in the *full* sequence, not within this chunk.
        cos = self.cos[pos_offset:pos_offset + t].to(q.dtype)
        sin = self.sin[pos_offset:pos_offset + t].to(q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if cache is not None:
            if "k_buf" in cache:
                start = int(cache.get("length", 0))
                end = start + t
                if end > cache["k_buf"].shape[2]:
                    raise ValueError("KV cache capacity exceeded")
                cache["k_buf"][:, :, start:end].copy_(k)
                cache["v_buf"][:, :, start:end].copy_(v)
                cache["length"] = end
                k = cache["k_buf"][:, :, :end]
                v = cache["v_buf"][:, :, :end]
            elif cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
                cache["k"], cache["v"] = k, v
            else:
                cache["k"], cache["v"] = k, v
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        drop = self.dropout if self.training else 0.0
        if doc_ids is not None:
            # Bool mask: True = attend. Causal AND same-document.
            causal = torch.ones(t, k.shape[2], dtype=torch.bool,
                                device=x.device).tril(diagonal=k.shape[2] - t)
            same = doc_ids[:, None, :, None] == doc_ids[:, None, None, :]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal & same,
                                                 dropout_p=drop)
        elif t == k.shape[2]:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        else:
            # Incremental decode: the query block is shorter than the cached
            # keys, so every query legitimately sees all of them. is_causal here
            # would mask top-left and silently blind the model to its own prefix.
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        return self.proj(out.transpose(1, 2).reshape(b, t, c))
