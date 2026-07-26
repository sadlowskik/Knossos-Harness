"""Echo: the nymph condemned to repeat what was already said.

Loop self-distillation for recurrent-depth models. `--variable-loops` teaches the
core to *survive* unseen depths; it never teaches a shallow pass to *agree* with
a deep one. Echo does exactly that: run the batch at a short loop count `k`, run
it at the max loop count `R`, and pull the k-loop logits toward the (detached)
R-loop logits.

    loss = ce(k-loop) + echo_weight * distill(k-loop, stopgrad(R-loop))

The teacher is the model's own deeper pass, so this is self-distillation with no
second model and no extra parameters -- the price is one extra forward per step
for fixed-loop models. For Ariadne / DaedalusFullAdaptive the per-step logits are
already materialised by the halting machinery, so the teacher is free: take
step `k` as the student and step `R` as the teacher out of the same stack.

This is the direct answer to "can you force it into fewer loops?" -- measure it
with scripts/echo_sweep.py, which sweeps loop count 1..R at eval.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def echo_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
              kind: str = "kl", temperature: float = 1.0) -> torch.Tensor:
    """Distillation term between a shallow pass and a deep one.

    The teacher is detached here, not at the call site, so it cannot be
    forgotten: gradient flows only into the student.

    kind="kl"  -- KL(teacher || student) on softened distributions, scaled by
                  T^2 so the gradient magnitude is temperature-independent
                  (Hinton et al., 2015).
    kind="mse" -- plain MSE on raw logits; no temperature, and it also matches
                  the teacher's confidence scale, which KL discards.
    """
    teacher = teacher_logits.detach()
    if kind == "mse":
        return F.mse_loss(student_logits, teacher)
    if kind != "kl":
        raise ValueError(f"unknown distill kind: {kind!r} (expected 'kl' or 'mse')")
    t = temperature
    # Flatten to (positions, vocab) first. `batchmean` divides by dim 0, so on a
    # (B, T, V) tensor it would divide by B and leave the term T times too
    # large -- at T=64 that is a distillation loss 64x the intended scale, which
    # swamps the CE and collapses training. Neither "zero when they agree" nor
    # "positive when they disagree" can catch a scale error, so this was invisible
    # to the unit tests and only showed up in the loop sweep.
    v = student_logits.shape[-1]
    s_log = F.log_softmax(student_logits.reshape(-1, v) / t, dim=-1)
    t_prob = F.softmax(teacher.reshape(-1, v) / t, dim=-1)
    return F.kl_div(s_log, t_prob, reduction="batchmean") * (t * t)


def echo_step(model, x: torch.Tensor, y: torch.Tensor, max_loops: int,
              weight: float, min_loops: int = 1, kind: str = "kl",
              temperature: float = 1.0, generator=None,
              ce_depth: str = "deep") -> torch.Tensor:
    """One Echo training step for a fixed-loop model (Labyrinth / full).

    Samples a student depth k in [min_loops, max_loops-1] and adds a
    distillation pull from the deep pass onto the shallow one.

    `ce_depth` decides where the cross-entropy is applied, and it matters more
    than it looks:

    "deep" (default)
        loss = ce(R-loop) + weight * distill(k-loop, stopgrad(R-loop))

        The deep pass is trained exactly as it would be with Echo off, and the
        distillation term is *added*. Turning `weight` up therefore changes one
        thing, which is what makes an --echo-weight sweep an ablation.

    "student"
        loss = ce(k-loop) + weight * distill(k-loop, stopgrad(R-loop))

        The original formulation. Kept because it is what the first
        implementation did, but note that `weight > 0` also moves the CE off
        `max_loops` onto a random shallow depth -- so a sweep against it varies
        two things at once, and the deep pass is never directly optimised. The
        first controlled run of the loop sweep failed this way: loop-1 val loss
        3.34 with Echo against 0.44 without, which is a confound, not a
        refutation of the idea.

    Returns plain CE at `max_loops` if `weight` is 0 or no shallower depth
    exists, so the two arms of a sweep stay comparable.
    """
    if weight <= 0.0 or max_loops <= min_loops:
        return _ce(model, x, y, max_loops)

    k = int(torch.randint(min_loops, max_loops, (1,), generator=generator).item())

    if ce_depth == "deep":
        # Both passes carry gradients: CE trains the deep one, distillation
        # pulls the shallow one toward it. `echo_loss` detaches the teacher, so
        # the deep pass never learns to be shallow.
        deep = _logits(model, x, max_loops)
        student = _logits(model, x, k)
        b, t, v = deep.shape
        ce = F.cross_entropy(deep.reshape(b * t, v), y.reshape(b * t))
        return ce + weight * echo_loss(student, deep, kind, temperature)

    if ce_depth != "student":
        raise ValueError(f"unknown ce_depth: {ce_depth!r} (expected 'deep' or 'student')")

    with torch.no_grad():
        teacher = _logits(model, x, max_loops)
    student = _logits(model, x, k)
    b, t, v = student.shape
    ce = F.cross_entropy(student.reshape(b * t, v), y.reshape(b * t))
    return ce + weight * echo_loss(student, teacher, kind, temperature)


def echo_from_steps(step_logits: torch.Tensor, k: int, weight: float,
                    kind: str = "kl", temperature: float = 1.0) -> torch.Tensor:
    """Echo term for models that already expose per-loop logits.

    `step_logits` is (n_steps, B, T, V) as produced by Ariadne and
    DaedalusFullAdaptive: student is step `k` (1-indexed), teacher the last step.
    Costs nothing extra -- those tensors were computed for the halting loss.
    """
    if weight <= 0.0 or step_logits.shape[0] < 2:
        return step_logits.new_zeros(())
    k = max(1, min(k, step_logits.shape[0] - 1))
    return weight * echo_loss(step_logits[k - 1], step_logits[-1], kind, temperature)


def _logits(model, x: torch.Tensor, n_loops: int) -> torch.Tensor:
    out = model(x, n_loops=n_loops)
    return out[0] if isinstance(out, tuple) else out


def _ce(model, x: torch.Tensor, y: torch.Tensor, n_loops: int) -> torch.Tensor:
    logits = _logits(model, x, n_loops)
    b, t, v = logits.shape
    return F.cross_entropy(logits.reshape(b * t, v), y.reshape(b * t))
