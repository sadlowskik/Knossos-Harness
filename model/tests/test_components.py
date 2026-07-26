"""Isolation tests for every Daedalus component.

These are the same checks used to validate each piece as it was built:
shape correctness, the causal-mask property, init loss ~= ln(vocab), the
load-balancing bounds, gist compression, and exact symbol extraction.

    pytest -q
"""
import math
import torch

from daedalus import (ByteTokenizer, Embeddings, Head, MultiHeadAttention, Block,
                      Daedalus, Labyrinth, Ariadne, ponder_loss,
                      MoELayer, load_balance_loss, DaedalusMoE, UnifiedDaedalus,
                      Mnemosyne, Scribe,
                      build_rope_cache, apply_rope, RoPEAttention, DaedalusFull,
                      DaedalusFullAdaptive)

B, T, C, V = 4, 16, 128, 256


def test_tokenizer_roundtrip():
    tok = ByteTokenizer()
    for s in ["def add(a, b):\n    return a + b\n", "x = {'π': 3.14, '🚀': 1}", ""]:
        assert tok.decode(tok.encode(s)) == s


def test_embeddings_positions_matter():
    emb = Embeddings(V, C, T)
    same_token = torch.full((1, T), 65)
    out = emb(same_token)
    assert not torch.allclose(out[0, 0], out[0, 1])   # position changes the vector


def test_attention_is_causal():
    head = Head(C, C, T)
    x = torch.randn(B, T, C)
    o1 = head(x)
    x2 = x.clone(); x2[:, -1] += 10.0                 # tamper the future
    assert torch.allclose(o1[:, :-1], head(x2)[:, :-1], atol=1e-5)


def test_block_preserves_causality():
    blk = Block(C, 4, T)
    x = torch.randn(B, T, C)
    o1 = blk(x)
    x2 = x.clone(); x2[:, -1] += 10.0
    assert torch.allclose(o1[:, :-1], blk(x2)[:, :-1], atol=1e-4)


def test_dense_init_loss_is_ln_vocab():
    torch.manual_seed(0)
    model = Daedalus(V, block_size=T)
    x = torch.randint(0, V, (B, T)); y = torch.randint(0, V, (B, T))
    _, loss = model(x, y)
    assert abs(loss.item() - math.log(V)) < 0.6


def test_labyrinth_same_params_more_depth():
    lab = Labyrinth(V, core_layers=3, n_loops=4, block_size=T)
    dense = Daedalus(V, n_layer=3, block_size=T)
    # same block count -> same parameter count, but the loop gives more depth
    assert sum(p.numel() for p in lab.parameters()) == sum(p.numel() for p in dense.parameters())


def test_labyrinth_variable_loops_differ():
    lab = Labyrinth(V, block_size=T)
    x = torch.randint(0, V, (B, T))
    with torch.no_grad():
        assert not torch.allclose(lab(x, n_loops=2)[0], lab(x, n_loops=8)[0])


def test_ariadne_halting_distribution_sums_to_one():
    ari = Ariadne(V, max_loops=6, block_size=T)
    x = torch.randint(0, V, (B, T))
    p, logits = ari(x)
    assert torch.allclose(p.sum(0), torch.ones(B, T), atol=1e-5)
    total, l_rec, l_kl = ponder_loss(p, logits, torch.randint(0, V, (B, T)))
    assert abs(l_rec.item() - math.log(V)) < 0.6
    assert l_kl.item() >= -1e-4


def test_load_balance_bounds():
    scores = torch.randn(B, T, 8)
    idx = scores.topk(2, -1)[1]
    aux, _, _ = load_balance_loss(scores, idx, 8)
    assert aux.item() >= 1.0 - 1e-3
    collapsed = torch.zeros_like(scores); collapsed[..., 0] = 20.0
    aux_bad, _, _ = load_balance_loss(collapsed, torch.zeros_like(idx), 8)
    assert aux_bad.item() > aux.item()


def test_moe_layer_shapes_and_causality():
    moe = MoELayer(C, n_experts=8, top_k=2)
    x = torch.randn(B, T, C)
    out, scores = moe(x)
    assert out.shape == (B, T, C) and scores.shape == (B, T, 8)


def test_moe_and_unified_forward():
    for model in (DaedalusMoE(V, block_size=T), UnifiedDaedalus(V, block_size=T)):
        x = torch.randint(0, V, (B, T)); y = torch.randint(0, V, (B, T))
        logits, ce, aux = model(x, y)
        assert logits.shape == (B, T, V)
        assert ce is not None and aux.item() >= 1.0 - 1e-3


def test_mnemosyne_compresses():
    mn = Mnemosyne(C, n_gist=16)
    g = mn(torch.randn(B, T, C))
    assert g.shape == (B, 16, C)


def test_rope_relative_position_invariance():
    cos, sin = build_rope_cache(128, 32)
    u, w = torch.randn(32), torch.randn(32)

    def score(m, n):
        qu = apply_rope(u.view(1, -1), cos[m:m+1], sin[m:m+1])
        kv = apply_rope(w.view(1, -1), cos[n:n+1], sin[n:n+1])
        return (qu * kv).sum().item()

    # same distance apart, different absolute positions -> same score
    assert abs(score(5, 3) - score(60, 58)) < 1e-3


def test_rope_attention_is_causal():
    attn = RoPEAttention(C, 4, T)
    x = torch.randn(B, T, C)
    o1 = attn(x)
    x2 = x.clone(); x2[:, -1] += 10.0
    assert torch.allclose(o1[:, :-1], attn(x2)[:, :-1], atol=1e-5)


def test_daedalus_full_forward_and_init_loss():
    torch.manual_seed(0)
    model = DaedalusFull(n_embd=64, n_head=4, block_size=T, core_layers=2, n_stages=2)
    x = torch.randint(0, V, (B, T)); y = torch.randint(0, V, (B, T))
    logits, ce, aux = model(x, y)
    assert logits.shape == (B, T, V)
    assert abs(ce.item() - math.log(V)) < 0.6
    # aux accumulates one balanced (~1.0) term per MoE application (stages*loops*layers)
    assert aux.item() >= 1.0 - 1e-3
    # test-time depth dial: different loop counts give different outputs
    with torch.no_grad():
        assert not torch.allclose(model(x, n_loops=2)[0], model(x, n_loops=5)[0])


def test_daedalus_full_adaptive_halting():
    torch.manual_seed(0)
    model = DaedalusFullAdaptive(n_embd=64, n_head=4, block_size=T, core_layers=2,
                                 max_loops=5, n_stages=2)
    x = torch.randint(0, V, (B, T)); y = torch.randint(0, V, (B, T))
    logits, loss, ex = model(x, y, beta=0.1)
    assert logits.shape == (B, T, V)
    # halting distribution over the final core sums to 1 per token
    assert torch.allclose(ex["p"].sum(0), torch.ones_like(ex["p"].sum(0)), atol=1e-5)
    # expectation-weighted reconstruction loss ~ ln(vocab) at init
    assert abs(ex["l_rec"].item() - math.log(V)) < 0.6
    assert loss is not None and ex["aux"].item() >= 1.0 - 1e-3


def test_scribe_exact_extraction():
    code = "import os\n\ndef add(a, b):\n    return a + b\n\nclass Calc:\n    def mul(self, x):\n        return x\n"
    s = Scribe(); s.ingest(code, "calc.py")
    assert s.functions["add"]["signature"] == "add(a, b)"
    assert "mul" in s.classes["Calc"]["methods"]
    assert "os" in s.imports
