"""Isolation tests for Naiads, the mixture-of-memories.

The load-bearing claim is non-interference: a segment routed to banks {a, b}
must leave every other bank bit-identical. That is asserted exactly (`torch.equal`,
not `allclose`) both after a forward and after a backward.

    pytest -q tests/test_naiads.py
"""
import torch

from daedalus import Naiads, MemoryLayer, DaedalusFull, DaedalusFullAdaptive, Mnemosyne

B, T, C, V = 4, 16, 64, 256
G, N_BANKS = 8, 4


def test_naiads_shapes():
    nai = Naiads(C, n_gist=G, n_head=4, n_banks=N_BANKS, top_k=2)
    x = torch.randn(B, T, C)
    read, state, scores = nai(x)
    assert read.shape == (B, T, C)
    assert state.shape == (B, N_BANKS, G, C)
    assert scores.shape == (B, 1, N_BANKS)


def test_unselected_banks_are_bit_identical():
    torch.manual_seed(0)
    nai = Naiads(C, n_gist=G, n_head=4, n_banks=N_BANKS, top_k=1).eval()
    x = torch.randn(B, T, C)
    state = nai.initial_state(B).clone()

    read, new_state, scores = nai(x)
    idx = scores.topk(1, -1)[1].squeeze(1)                 # (B, 1)

    touched = 0
    for bi in range(B):
        for e in range(N_BANKS):
            if e in idx[bi].tolist():
                touched += 1
                continue
            assert torch.equal(new_state[bi, e], state[bi, e]), \
                f"bank {e} changed for row {bi} without being routed to"
    assert touched == B                                    # top_k=1: exactly one per row


def test_unselected_banks_survive_a_backward_step():
    """A gradient step must not move banks the router never selected."""
    torch.manual_seed(0)
    nai = Naiads(C, n_gist=G, n_head=4, n_banks=N_BANKS, top_k=1)
    nai.eval()                                             # disable router noise
    x = torch.randn(B, T, C)
    state = nai.initial_state(B).detach().clone()

    read, new_state, scores = nai(x)
    read.pow(2).mean().backward()
    idx = scores.topk(1, -1)[1].squeeze(1)

    for bi in range(B):
        for e in range(N_BANKS):
            if e not in idx[bi].tolist():
                assert torch.equal(new_state[bi, e].detach(), state[bi, e])


def test_bank_load_balance_bounds():
    """Mirrors the Muses expert-balance check: 1.0 balanced, larger when collapsed."""
    nai = Naiads(C, n_gist=G, n_head=4, n_banks=N_BANKS, top_k=2)
    balanced = torch.randn(B, 1, N_BANKS)
    assert nai.aux_loss(balanced).item() >= 1.0 - 1e-3

    collapsed = torch.zeros(B, 1, N_BANKS); collapsed[..., 0] = 20.0
    assert nai.aux_loss(collapsed).item() > nai.aux_loss(balanced).item()


def test_state_carries_across_segments():
    """Feeding a second segment must build on the first segment's state."""
    torch.manual_seed(0)
    nai = Naiads(C, n_gist=G, n_head=4, n_banks=N_BANKS, top_k=2).eval()
    with torch.no_grad():
        _, s1, _ = nai(torch.randn(B, T, C))
        _, s2, _ = nai(torch.randn(B, T, C), s1)
    assert not torch.equal(s1, s2)


def test_memory_layer_single_bank_is_the_old_path():
    """n_banks=1 must still be a plain Mnemosyne with zero aux loss."""
    layer = MemoryLayer(C, G, 4)                           # default n_banks=1
    assert isinstance(layer.compress, Mnemosyne)
    out, aux = layer(torch.randn(B, T, C))
    assert out.shape == (B, T, C) and aux.item() == 0.0


def test_full_models_accept_memory_banks():
    x = torch.randint(0, V, (B, T)); y = torch.randint(0, V, (B, T))
    full = DaedalusFull(n_embd=C, n_head=4, block_size=T, core_layers=2,
                        n_stages=2, n_gist=G, n_mem_banks=N_BANKS)
    logits, ce, aux = full(x, y)
    assert logits.shape == (B, T, V) and torch.isfinite(ce) and torch.isfinite(aux)

    ada = DaedalusFullAdaptive(n_embd=C, n_head=4, block_size=T, core_layers=2,
                               max_loops=3, n_stages=2, n_gist=G, n_mem_banks=N_BANKS)
    logits, loss, ex = ada(x, y, beta=0.1)
    assert logits.shape == (B, T, V) and torch.isfinite(loss)
    assert torch.allclose(ex["p"].sum(0), torch.ones_like(ex["p"].sum(0)), atol=1e-5)
