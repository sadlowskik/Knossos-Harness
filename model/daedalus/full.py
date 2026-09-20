"""DaedalusFull -- the fully integrated architecture.

Combines, in one model:
  - RoPE positions (rope.RoPEAttention)
  - fine-grained MoE (moe.MoELayer: Muses / Apollo / Themis)
  - input injection (the original embedding is re-added every loop)
  - interleaved memory (memory.Mnemosyne between recurrent stages)
  - variable-loop recurrent depth

Structure:
    Prelude(token emb)
      -> RecurrentMoECore (looped, input injection)
      -> MemoryLayer      (compress -> read back)
      -> RecurrentMoECore
      -> Coda(norm + head)

Adaptive halting (Ariadne) is intentionally NOT fused here: PonderNet's
per-step output weighting does not compose cleanly with the interleaved
core -> memory -> core stack. Use it as a separate model when you want
per-token adaptive depth.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RoPEAttention
from .moe import MoELayer, load_balance_loss
from .memory import Mnemosyne
from .naiads import Naiads
from .ariadne import ponder_loss


class RoPEMoEBlock(nn.Module):
    """Pre-norm block: RoPE attention + MoE feed-forward."""

    def __init__(self, n_embd, n_head, block_size, n_experts, top_k, n_shared, hidden,
                 qk_norm: bool = False, n_kv_head=None, bias_update: float = 0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = RoPEAttention(n_embd, n_head, block_size, qk_norm=qk_norm,
                                  n_kv_head=n_kv_head)
        self.moe = MoELayer(n_embd, n_experts, top_k, n_shared, hidden,
                            bias_update=bias_update)

    def forward(self, x: torch.Tensor, doc_ids=None, cache=None,
                pos_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x), doc_ids=doc_ids, cache=cache,
                          pos_offset=pos_offset)
        moe_out, scores = self.moe(self.ln2(x))
        return x + moe_out, scores


class RecurrentMoECore(nn.Module):
    """A shared stack of MoE blocks, looped with input injection.

    The prelude embedding `e` is re-added before every loop so deep recurrence
    stays anchored to the input (Huginn, Geiping et al. 2025).

    `loop_embed > 0` allocates a learned per-iteration vector, added alongside
    `e`. Without it the block has **no way to tell which iteration it is on**:
    `e` is identical on every pass, and the only signal distinguishing loop 1
    from loop 4 is the hidden state itself. Universal Transformer (Dehghani et
    al. 2018) shares weights across depth and adds an explicit timestep
    embedding for exactly this reason. The cost is `loop_embed x n_embd`
    parameters -- 2,048 at max_loops=4, n_embd=512, or 0.005% of a 45M model --
    which is why this is the cheapest way to test whether the loops "want" to
    be different functions before paying for per-loop weights.
    """

    def __init__(self, n_embd, n_head, block_size, core_layers, n_experts, top_k,
                 n_shared, hidden, loop_embed: int = 0, inject_gate: int = 0,
                 qk_norm: bool = False, n_kv_head=None, bias_update: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            RoPEMoEBlock(n_embd, n_head, block_size, n_experts, top_k, n_shared,
                         hidden, qk_norm=qk_norm, n_kv_head=n_kv_head,
                         bias_update=bias_update)
            for _ in range(core_layers)
        ])
        self.loop_emb = nn.Embedding(loop_embed, n_embd) if loop_embed else None
        # Input injection re-adds `e` on every loop -- seven times at the default
        # config, with nothing scaling it. The residual stream therefore grows
        # monotonically, and each block's contribution shrinks relative to it
        # (the deep pre-norm pathology, made worse by recurrence because the same
        # block writes into an ever-larger stream). A learned per-iteration gate
        # lets the model decide how much input to re-inject at each depth.
        # Initialised to 1.0, so at step zero this is exactly the old behaviour.
        self.inject_gate = nn.Parameter(torch.ones(inject_gate)) if inject_gate else None

    def step(self, i: int):
        """Per-iteration offset for loop `i`, or 0.0 when disabled.

        Clamped, because `--variable-loops` samples loop counts above the
        trained maximum and the test-time depth dial is meant to keep working
        past it. Indexing past the table would crash instead.
        """
        if self.loop_emb is None:
            return 0.0
        return self.loop_emb.weight[min(i, self.loop_emb.num_embeddings - 1)]

    def gate(self, i: int):
        """Per-iteration injection strength, or 1.0 when disabled."""
        if self.inject_gate is None:
            return 1.0
        return self.inject_gate[min(i, self.inject_gate.numel() - 1)]

    def forward(self, x: torch.Tensor, e: torch.Tensor, n_loops: int,
                doc_ids=None) -> Tuple[torch.Tensor, torch.Tensor]:
        aux = x.new_zeros(())
        for i in range(n_loops):
            x = x + self.gate(i) * e + self.step(i)        # input injection
            for blk in self.blocks:
                x, scores = blk(x, doc_ids=doc_ids)
                # load_balance_loss returns (aux, load, importance); only the
                # scalar is summed. Dropping the [0] adds a tuple to a tensor.
                aux = aux + load_balance_loss(
                    scores, scores.topk(blk.moe.top_k, -1)[1], blk.moe.n_experts)[0]
        return x, aux


class MemoryLayer(nn.Module):
    """Interleaved memory: compress the running state into gist vectors, then
    let every position read from them via cross-attention.

    With `n_banks > 1` the single Mnemosyne bank is replaced by Naiads -- several
    independently-updated banks with top-k segment routing (see naiads.py).
    `n_banks=1` is the original single-bank path, unchanged.

    forward() returns (x, aux). `aux` is zero for the single-bank path and the
    bank load-balancing loss for Naiads.
    """

    def __init__(self, n_embd, n_gist, n_head, n_banks: int = 1, mem_top_k: int = 2):
        super().__init__()
        self.n_banks = n_banks
        if n_banks == 1:
            self.compress = Mnemosyne(n_embd, n_gist, n_head)
            self.read = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
            self.ln = nn.LayerNorm(n_embd)
        else:
            self.naiads = Naiads(n_embd, n_gist, n_head, n_banks, min(mem_top_k, n_banks))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n_banks == 1:
            gist = self.compress(x)
            readout, _ = self.read(x, gist, gist)
            return x + self.ln(readout), x.new_zeros(())
        readout, _state, scores = self.naiads(x)
        return x + readout, self.naiads.aux_loss(scores)


class DaedalusFull(nn.Module):
    """The whole architecture in one model.

    forward() returns (logits, ce_loss, aux_loss). Train with
    `ce_loss + alpha * aux_loss` and a variable loop count (sample `n_loops`
    each step) so the test-time depth dial stays usable.
    """

    def __init__(self, vocab_size: int = 256, n_embd: int = 128, n_head: int = 4,
                 block_size: int = 256, core_layers: int = 2, n_loops: int = 3,
                 n_experts: int = 8, top_k: int = 2, n_shared: int = 1,
                 hidden: Optional[int] = None, n_gist: int = 16, n_stages: int = 2,
                 n_mem_banks: int = 1, loop_embed: bool = False):
        super().__init__()
        hidden = hidden or n_embd
        self.tok_emb = nn.Embedding(vocab_size, n_embd)      # RoPE handles position
        self.stages = nn.ModuleList([
            RecurrentMoECore(n_embd, n_head, block_size, core_layers,
                             n_experts, top_k, n_shared, hidden,
                             loop_embed=(n_loops if loop_embed else 0))
            for _ in range(n_stages)
        ])
        self.memories = nn.ModuleList([
            MemoryLayer(n_embd, n_gist, n_head, n_mem_banks) for _ in range(n_stages - 1)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.n_loops, self.block_size = n_loops, block_size

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                n_loops: Optional[int] = None):
        r = n_loops if n_loops is not None else self.n_loops
        e = self.tok_emb(idx)
        x = e
        aux = x.new_zeros(())
        for i, stage in enumerate(self.stages):
            x, a = stage(x, e, r)
            aux = aux + a
            if i < len(self.memories):
                x, mem_aux = self.memories[i](x)             # interleaved memory
                aux = aux + mem_aux                          # zero unless Naiads
        logits = self.lm_head(self.ln_f(x))
        ce = None
        if targets is not None:
            b, t, v = logits.shape
            ce = F.cross_entropy(logits.view(b * t, v), targets.reshape(b * t))
        return logits, ce, aux


class DaedalusFullAdaptive(nn.Module):
    """DaedalusFull with PonderNet adaptive halting on the FINAL recurrent core.

    Every stage but the last runs a fixed number of loops (with interleaved
    memory). The final core loops up to `max_loops`, with a halting head
    producing a per-token halting distribution; its output is applied per step
    via the coda. This localizes adaptive depth to where the output is produced,
    which composes cleanly with the interleaved memory stack.

    forward() returns (expected_logits, loss, extras). `extras` holds the aux
    (MoE load-balance) loss, the halting distribution `p`, and `l_rec`/`l_kl`.
    Train with a stronger `beta` (e.g. 0.1) so halting does not collapse to
    max depth.
    """

    def __init__(self, vocab_size: int = 256, n_embd: int = 128, n_head: int = 4,
                 block_size: int = 256, core_layers: int = 2, fixed_loops: int = 3,
                 max_loops: int = 6, n_experts: int = 8, top_k: int = 2, n_shared: int = 1,
                 hidden: Optional[int] = None, n_gist: int = 16, n_stages: int = 2,
                 n_mem_banks: int = 1, loop_embed: bool = False,
                 inject_gate: bool = False, qk_norm: bool = False,
                 n_kv_head=None, bias_update: float = 0.0, mtp: bool = False):
        super().__init__()
        hidden = hidden or n_embd
        n_steps = max(fixed_loops, max_loops)
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.stages = nn.ModuleList([
            RecurrentMoECore(n_embd, n_head, block_size, core_layers,
                             n_experts, top_k, n_shared, hidden,
                             loop_embed=(n_steps if loop_embed else 0),
                             inject_gate=(n_steps if inject_gate else 0),
                             qk_norm=qk_norm, n_kv_head=n_kv_head,
                             bias_update=bias_update)
            for _ in range(n_stages)
        ])
        # Multi-token prediction (DeepSeek-V3): a second head predicting token
        # t+2 from position t. It costs one vocab-sized matrix and is discarded
        # at inference, but the extra supervision per position measurably
        # improves sample efficiency -- which is the whole point when the token
        # budget, not the parameter count, is what you are short of.
        self.mtp_head = nn.Linear(n_embd, vocab_size) if mtp else None
        self.memories = nn.ModuleList([
            MemoryLayer(n_embd, n_gist, n_head, n_mem_banks) for _ in range(n_stages - 1)
        ])
        self.halt = nn.Linear(n_embd, 1)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.fixed_loops, self.max_loops, self.block_size = fixed_loops, max_loops, block_size

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                lambda_prior: float = 0.2, beta: float = 0.01, alpha: float = 0.01,
                doc_ids: Optional[torch.Tensor] = None, mtp_weight: float = 0.0,
                return_step_logits: bool = True):
        e = self.tok_emb(idx)
        x = e
        aux = x.new_zeros(())
        # fixed preprocessing stages + interleaved memory
        for i in range(len(self.stages) - 1):
            x, a = self.stages[i](x, e, self.fixed_loops, doc_ids=doc_ids)
            aux = aux + a
            x, mem_aux = self.memories[i](x)
            aux = aux + mem_aux
        # adaptive final core (PonderNet halting)
        final = self.stages[-1]
        still = torch.ones(idx.shape, device=idx.device)
        p_list, logits_list = [], []
        exp_logits = None
        for n in range(1, self.max_loops + 1):
            # The final core's loop is unrolled here rather than delegated to
            # RecurrentMoECore.forward, so the per-step offset and injection gate
            # have to be applied explicitly -- otherwise --loop-embed and
            # --inject-gate would silently do nothing on the one stage whose
            # depth actually varies.
            x = x + final.gate(n - 1) * e + final.step(n - 1)
            for blk in final.blocks:
                x, scores = blk(x, doc_ids=doc_ids)
                aux = aux + load_balance_loss(
                    scores, scores.topk(blk.moe.top_k, -1)[1], blk.moe.n_experts)[0]
            lam = (torch.sigmoid(self.halt(x)).squeeze(-1) if n < self.max_loops
                   else torch.ones(idx.shape, device=idx.device))
            prob = still * lam
            p_list.append(prob)
            still = still * (1 - lam)
            step_logits = self.lm_head(self.ln_f(x))
            exp_logits = (step_logits * prob.unsqueeze(-1) if exp_logits is None
                          else exp_logits + step_logits * prob.unsqueeze(-1))
            if targets is not None or return_step_logits:
                logits_list.append(step_logits)
        p = torch.stack(p_list, 0)
        logits = torch.stack(logits_list, 0) if logits_list else None
        # Accumulate the halting-weighted mixture one step at a time rather than
        # as `(p.unsqueeze(-1) * logits).sum(0)`. That expression materialises a
        # second (steps, B, T, vocab) tensor -- and because `p` is fp32 while
        # `logits` is fp16 under autocast, the product promotes to fp32 and the
        # copy is *twice* the size of the stack it came from. At max_loops=4,
        # B=16, T=1024, vocab=16384 that single temporary is 4 GiB, which is
        # what caps the batch size on a 16GB card. Accumulating peaks at one
        # (B, T, vocab) term instead of `steps` of them.
        # `step_logits` is kept so Echo (loop self-distillation) can use step R as a
        # teacher for step k without a second forward pass.
        extras = {"aux": aux, "p": p, "step_logits": logits}
        loss = None
        if targets is not None:
            assert logits is not None
            pond, l_rec, l_kl = ponder_loss(p, logits, targets, lambda_prior, beta)
            loss = pond + alpha * aux
            extras.update(l_rec=l_rec, l_kl=l_kl)
            if self.mtp_head is not None and mtp_weight > 0.0:
                # Position t already predicts t+1 through lm_head; this head
                # predicts t+2, so its label is `targets` shifted left by one and
                # the final position has no label to learn from.
                v = self.mtp_head(self.ln_f(x))
                l_mtp = F.cross_entropy(v[:, :-1].reshape(-1, v.shape[-1]),
                                        targets[:, 1:].reshape(-1))
                loss = loss + mtp_weight * l_mtp
                extras["l_mtp"] = l_mtp
        return exp_logits, loss, extras
