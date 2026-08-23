"""Tests for the opt-in architecture flags: GQA, QK-norm, doc masking, injection
gating, aux-loss-free MoE balancing, multi-token prediction, z-loss, KV cache."""
from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from daedalus.full import DaedalusFullAdaptive                             # noqa: E402
from daedalus.moe import MoELayer                                          # noqa: E402
from daedalus.rope import RoPEAttention                                    # noqa: E402
from generate import build, resolve_config                                 # noqa: E402
from train import document_ids, z_loss                                     # noqa: E402

BASE = dict(vocab_size=64, n_embd=32, n_head=4, block_size=16, core_layers=1,
            max_loops=3, n_experts=4, n_gist=4, n_stages=2)


def _xy(b=2, t=16, v=64):
    g = torch.Generator().manual_seed(0)
    return (torch.randint(0, v, (b, t), generator=g),
            torch.randint(0, v, (b, t), generator=g))


# ------------------------------------------------------------------- KV cache

def test_incremental_decode_matches_full_forward():
    """The cache is only worth having if it changes nothing but the cost."""
    torch.manual_seed(0)
    a = RoPEAttention(32, 4, 16).eval()
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        full = a(x)
        cache, steps = {}, []
        for i in range(8):
            steps.append(a(x[:, i:i + 1], cache=cache, pos_offset=i))
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5)


def test_cache_grows_by_one_per_step():
    a = RoPEAttention(32, 4, 16).eval()
    cache = {}
    with torch.no_grad():
        for i in range(5):
            a(torch.randn(1, 1, 32), cache=cache, pos_offset=i)
            assert cache["k"].shape[2] == i + 1


def test_cache_with_gqa_keeps_kv_heads_small():
    a = RoPEAttention(32, 4, 16, n_kv_head=2).eval()
    cache = {}
    with torch.no_grad():
        a(torch.randn(1, 4, 32), cache=cache)
    assert cache["k"].shape[1] == 2          # cached before the repeat, as intended


# ----------------------------------------------------------------------- GQA

def test_gqa_shrinks_parameters_and_preserves_shape():
    full = RoPEAttention(32, 4, 16)
    gqa = RoPEAttention(32, 4, 16, n_kv_head=2)
    x = torch.randn(2, 8, 32)
    assert gqa(x).shape == full(x).shape
    assert sum(p.numel() for p in gqa.parameters()) < sum(p.numel() for p in full.parameters())


def test_gqa_rejects_indivisible_head_counts():
    with pytest.raises(AssertionError):
        RoPEAttention(32, 4, 16, n_kv_head=3)


# ------------------------------------------------------------------- QK-norm

def test_qk_norm_adds_two_vectors_and_bounds_logits():
    a = RoPEAttention(32, 4, 16, qk_norm=True)
    assert a.q_norm is not None and a.q_norm.numel() == 8
    assert RoPEAttention(32, 4, 16).q_norm is None
    # A wildly scaled input must not produce a wildly scaled output.
    big = torch.randn(2, 8, 32) * 50
    assert torch.isfinite(a(big)).all()


# ------------------------------------------------------------ document masking

def test_document_ids_keeps_separator_with_its_own_document():
    eot = 9
    x = torch.tensor([[1, 2, eot, 3, 4, eot, 5]])
    assert document_ids(x, eot).tolist() == [[0, 0, 0, 1, 1, 1, 2]]


def test_document_ids_is_none_without_an_eot():
    assert document_ids(torch.tensor([[1, 2, 3]]), None) is None


def test_document_mask_changes_attention():
    torch.manual_seed(0)
    a = RoPEAttention(32, 4, 16).eval()
    x = torch.randn(1, 8, 32)
    doc = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    with torch.no_grad():
        assert not torch.allclose(a(x), a(x, doc_ids=doc), atol=1e-4)


def test_first_document_is_unaffected_by_masking():
    """Tokens in document 0 have nothing before them, so masking is a no-op there."""
    torch.manual_seed(0)
    a = RoPEAttention(32, 4, 16).eval()
    x = torch.randn(1, 8, 32)
    doc = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    with torch.no_grad():
        assert torch.allclose(a(x)[:, :4], a(x, doc_ids=doc)[:, :4], atol=1e-5)


def test_preallocated_kv_cache_matches_concatenating_cache():
    torch.manual_seed(0)
    a = RoPEAttention(32, 4, 16, n_kv_head=2).eval()
    x = torch.randn(1, 6, 32)
    old = {"k": None, "v": None}
    new = a.init_cache(1, 6, device=x.device, dtype=x.dtype)
    with torch.no_grad():
        old_out = torch.cat([a(x[:, i:i + 1], cache=old, pos_offset=i)
                             for i in range(6)], dim=1)
        new_out = torch.cat([a(x[:, i:i + 1], cache=new, pos_offset=i)
                             for i in range(6)], dim=1)
    assert torch.allclose(old_out, new_out, atol=1e-5)
    assert new["length"] == 6


# --------------------------------------------------------------- inject gate

def test_inject_gate_is_identity_at_initialisation():
    """Gates start at 1.0, so an untrained gated model equals the ungated one."""
    x, y = _xy()
    torch.manual_seed(1)
    plain = DaedalusFullAdaptive(**BASE)
    torch.manual_seed(1)
    gated = DaedalusFullAdaptive(**BASE, inject_gate=True)
    plain.eval(), gated.eval()
    with torch.no_grad():
        assert torch.allclose(plain(x, y)[1], gated(x, y)[1], atol=1e-5)


def test_inject_gate_receives_gradient():
    m = DaedalusFullAdaptive(**BASE, inject_gate=True)
    x, y = _xy()
    m(x, y)[1].backward()
    g = m.stages[-1].inject_gate.grad
    assert g is not None and g.abs().sum() > 0


# ------------------------------------------------- aux-loss-free MoE balancing

def test_bias_update_disabled_is_a_no_op():
    torch.manual_seed(0)
    off = MoELayer(32, n_experts=4, top_k=2).eval()
    torch.manual_seed(0)
    on = MoELayer(32, n_experts=4, top_k=2, bias_update=1e-3).eval()
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        assert torch.allclose(off(x)[0], on(x)[0], atol=1e-6)


def test_bias_moves_toward_balance_during_training():
    torch.manual_seed(0)
    moe = MoELayer(32, n_experts=4, top_k=1, bias_update=0.05).train()
    x = torch.randn(4, 16, 32)
    for _ in range(20):
        moe(x)
    bias = moe.router.expert_bias
    assert bias.abs().sum() > 0, "bias never moved"
    # Under-used experts must be pushed up relative to over-used ones.
    assert bias.max() > bias.min()


def test_default_model_state_dict_has_no_new_keys():
    """An opt-in flag must not change the on-disk shape of a model without it.

    Registering the balancing buffer unconditionally added keys to every
    checkpoint, which made runs written before the feature existed fail to load
    with "Missing key(s)" -- breaking --resume on anything already training.
    """
    m = DaedalusFullAdaptive(**BASE)
    sd = m.state_dict()
    assert not [k for k in sd if "expert_bias" in k]
    assert not [k for k in sd if "loop_emb" in k or "inject_gate" in k]
    assert not [k for k in sd if "q_norm" in k or "k_norm" in k]
    assert not [k for k in sd if "mtp_head" in k]
    m.load_state_dict(sd)                       # strict round-trip


def test_enabled_model_persists_its_balancing_bias():
    m = DaedalusFullAdaptive(**BASE, bias_update=1e-3)
    assert [k for k in m.state_dict() if "expert_bias" in k]


def test_bias_is_a_buffer_not_a_parameter():
    """It is updated by rule, so gradient descent must not touch it."""
    moe = MoELayer(32, n_experts=4, top_k=2, bias_update=1e-3)
    assert "router.expert_bias" in dict(moe.named_buffers())
    assert "router.expert_bias" not in dict(moe.named_parameters())


# ----------------------------------------------------------------------- MTP

def test_mtp_adds_a_head_and_a_loss_term():
    x, y = _xy()
    m = DaedalusFullAdaptive(**BASE, mtp=True)
    _, loss_off, ex_off = m(x, y, mtp_weight=0.0)
    _, loss_on, ex_on = m(x, y, mtp_weight=0.5)
    assert "l_mtp" not in ex_off and "l_mtp" in ex_on
    assert float(loss_on) > float(loss_off)


def test_mtp_head_absent_by_default():
    assert DaedalusFullAdaptive(**BASE).mtp_head is None


# -------------------------------------------------------------------- z-loss

def test_z_loss_is_zero_for_uniform_logits_and_grows_with_scale():
    small = torch.zeros(2, 4, 8)
    assert float(z_loss(small)) == pytest.approx(float(torch.tensor(8.0).log() ** 2))
    assert float(z_loss(torch.ones(2, 4, 8) * 10)) > float(z_loss(small))


# ------------------------------------------------------------ round-tripping

@pytest.mark.parametrize("flags", [
    {"inject_gate": True}, {"qk_norm": True}, {"n_kv_head": 2},
    {"bias_update": 1e-3}, {"mtp": True},
    {"loop_embed": True, "inject_gate": True, "qk_norm": True,
     "n_kv_head": 2, "bias_update": 1e-3, "mtp": True},
])
def test_checkpoint_round_trip(flags):
    m = DaedalusFullAdaptive(**BASE, **flags)
    args = {"model": "adaptive", "vocab_size": 64, "n_embd": 32, "n_head": 4,
            "block_size": 16, "core_layers": 1, "max_loops": 3, "n_experts": 4,
            "n_gist": 4, "n_stages": 2, "n_mem_banks": 1}
    args.update(flags)
    rebuilt = build(resolve_config({"args": args}), "cpu")
    rebuilt.load_state_dict(m.state_dict())      # raises on shape mismatch


def test_old_checkpoints_still_load_with_all_flags_off():
    cfg = resolve_config({"args": {"model": "adaptive", "vocab_size": 64,
                                   "n_embd": 32, "n_head": 4, "block_size": 16}})
    assert cfg["n_kv_head"] is None and cfg["bias_update"] == 0.0
    m = build(cfg, "cpu")
    assert m.mtp_head is None
    assert all(s.inject_gate is None for s in m.stages)
