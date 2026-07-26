"""Isolation tests for Moirai, the gated fast-weight mixer.

The central test re-derives the recurrence step by step from the module's own
projections and checks the module matches it numerically -- if the scan, the
gating, or the delta rule is wrong, this fails.

    pytest -q tests/test_moirai.py
"""
import torch
import torch.nn.functional as F

from daedalus import MoiraiMixer, MultiHeadAttention, Labyrinth

B, T, C, V = 2, 4, 32, 256
H = 4


def _reference(mix: MoiraiMixer, x: torch.Tensor) -> torch.Tensor:
    """The recurrence written out longhand, independently of forward()."""
    b, n, c = x.shape
    hd = c // mix.n_head

    def heads(t):
        return t.view(b, n, mix.n_head, hd).transpose(1, 2)

    k = F.normalize(heads(mix.k(x)), dim=-1)
    q = F.normalize(heads(mix.q(x)), dim=-1)
    v = heads(mix.v(x))
    gb = torch.sigmoid(heads(mix.erase(x)))
    gw = gb if mix.tied else torch.sigmoid(heads(mix.write(x)))

    W = torch.zeros(b, mix.n_head, hd, hd)
    ys = []
    for t in range(n):
        y_t = torch.zeros(b, mix.n_head, hd)
        for bi in range(b):
            for h in range(mix.n_head):
                k_t, v_t, q_t = k[bi, h, t], v[bi, h, t], q[bi, h, t]
                r_t = v_t - W[bi, h] @ k_t                       # delta-rule residual
                W[bi, h] = (gb[bi, h, t].unsqueeze(-1) * W[bi, h]
                            + gw[bi, h, t].unsqueeze(-1) * torch.outer(r_t, k_t))
                y_t[bi, h] = W[bi, h] @ q_t
        ys.append(y_t)
    y = torch.stack(ys, dim=2).transpose(1, 2).reshape(b, n, c)
    return mix.proj(mix.ln(y))


def test_moirai_matches_hand_computed_recurrence():
    torch.manual_seed(0)
    mix = MoiraiMixer(C, H).eval()
    x = torch.randn(B, T, C)
    with torch.no_grad():
        assert torch.allclose(mix(x), _reference(mix, x), atol=1e-5)


def test_moirai_tied_gates_match_recurrence():
    torch.manual_seed(0)
    mix = MoiraiMixer(C, H, tied=True).eval()
    x = torch.randn(B, T, C)
    with torch.no_grad():
        assert torch.allclose(mix(x), _reference(mix, x), atol=1e-5)


def test_moirai_is_drop_in_for_attention():
    x = torch.randn(B, T, C)
    attn, mix = MultiHeadAttention(C, H, T), MoiraiMixer(C, H, T)
    assert attn(x).shape == mix(x).shape == (B, T, C)


def test_moirai_is_causal():
    mix = MoiraiMixer(C, H).eval()
    x = torch.randn(B, T, C)
    with torch.no_grad():
        o1 = mix(x)
        x2 = x.clone(); x2[:, -1] += 10.0            # tamper the future
        assert torch.allclose(o1[:, :-1], mix(x2)[:, :-1], atol=1e-5)


def test_moirai_gradients_reach_the_gates():
    mix = MoiraiMixer(C, H)
    x = torch.randn(B, T, C, requires_grad=True)
    mix(x).pow(2).mean().backward()
    for name, p in mix.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient in {name}"
    assert mix.erase.weight.grad.abs().sum() > 0      # erase gate is trained
    assert mix.write.weight.grad.abs().sum() > 0      # write gate is trained
    assert torch.isfinite(x.grad).all()


def test_moirai_state_is_bounded_on_a_long_sequence():
    """The fixed-size state must not blow up as the sequence grows."""
    mix = MoiraiMixer(C, H).eval()
    with torch.no_grad():
        _, W = mix(torch.randn(1, 256, C), return_state=True)
    assert torch.isfinite(W).all() and W.norm().item() < 1e4


def test_labyrinth_accepts_the_moirai_mixer():
    lab = Labyrinth(V, n_embd=C, n_head=H, core_layers=2, n_loops=2,
                    block_size=T, mixer="moirai")
    x = torch.randint(0, V, (B, T)); y = torch.randint(0, V, (B, T))
    logits, loss = lab(x, y)
    assert logits.shape == (B, T, V) and torch.isfinite(loss)


def test_labyrinth_default_mixer_is_unchanged():
    """Default construction must still build softmax attention."""
    lab = Labyrinth(V, n_embd=C, n_head=H, block_size=T)
    assert isinstance(lab.core[0].attn, MultiHeadAttention)
