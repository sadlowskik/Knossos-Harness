"""Running an arm over several seeds, and reporting it honestly.

PLAN.md rule 4: *no number without its `n`, and no `n` below 5*. That rule was
not written in the abstract -- an Echo result was reported as "the loss halves"
from seed 0 alone, and seed 0's baseline turned out to be an outlier that
overstated the effect about twofold. The corrected three-seed table said
something much smaller and much more defensible.

So every arm here reports mean, spread and per-seed values, and the per-seed
values are printed rather than summarised away: with n=5 the individual numbers
are still small enough to read, and a mean hiding one wild seed is exactly the
failure this module exists to prevent.

A paired design throughout -- every arm sees the same seeds, and within a seed
the same data order -- so arm-to-arm deltas are not competing with seed noise.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

#: Rule 4's floor. Scripts may take more; none should take fewer without saying so.
DEFAULT_SEEDS = 5


@dataclass
class Arm:
    """One condition, measured across seeds."""

    label: str
    values: List[float] = field(default_factory=list)
    #: Seeds whose run was excluded, and why. Reported, never silently dropped:
    #: a lost sample that correlates with the condition is a confound, not noise.
    excluded: Dict[int, str] = field(default_factory=dict)
    #: Anything constant across seeds worth showing, e.g. a parameter count.
    note: str = ""

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.values) if self.values else float("nan")

    @property
    def stdev(self) -> float:
        """Sample standard deviation; 0.0 below two samples rather than raising."""
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    def summary(self) -> str:
        if not self.values:
            return "no successful runs"
        return f"{self.mean:.4f} +/- {self.stdev:.4f} (n={self.n})"


def run_arm(label: str, fn: Callable[[int], Optional[float]],
            seeds: Sequence[int], note: str = "",
            progress: bool = True) -> Arm:
    """Run `fn(seed)` for each seed. Returning None excludes that seed.

    An exception is recorded as an exclusion rather than killing the sweep --
    losing one arm to a diverged run should not cost the other arms too, and
    the exclusion is visible in the report.
    """
    arm = Arm(label=label, note=note)
    for seed in seeds:
        if progress:
            print(f"  {label:<24} seed {seed} ...", end="", flush=True)
        try:
            value = fn(seed)
        except Exception as exc:                     # noqa: BLE001 - recorded below
            arm.excluded[seed] = f"{type(exc).__name__}: {exc}"
            if progress:
                print(f" EXCLUDED ({type(exc).__name__})")
            continue
        if value is None:
            arm.excluded[seed] = "run reported itself unusable"
            if progress:
                print(" EXCLUDED")
            continue
        arm.values.append(value)
        if progress:
            print(f" {value:.4f}")
    return arm


def table(arms: Sequence[Arm], metric: str = "values") -> str:
    """A fixed-width table: mean, spread, n, and every seed's value."""
    width = max((len(a.label) for a in arms), default=10)
    head = (f"{'arm':<{width}} | {'mean':>9} | {'stdev':>8} | {'n':>2} | "
            f"per-seed {metric}")
    lines = [head, "-" * len(head)]
    for arm in arms:
        per = " ".join(f"{v:.4f}" for v in arm.values) or "--"
        lines.append(f"{arm.label:<{width}} | {arm.mean:>9.4f} | {arm.stdev:>8.4f} | "
                     f"{arm.n:>2} | {per}")
        if arm.note:
            lines.append(f"{'':<{width}} | {arm.note}")
        for seed, why in arm.excluded.items():
            lines.append(f"{'':<{width}} | seed {seed} EXCLUDED: {why}")
    return "\n".join(lines)


def delta(a: Arm, b: Arm) -> str:
    """`b - a`, with a plain statement of whether the spread swamps it.

    Not a significance test -- at n=5 that would be dressing up a small sample.
    It is the one comparison worth making by eye: if the difference between two
    arms is smaller than the seed-to-seed variation inside them, the difference
    has not been measured yet, and saying so is the whole point of rule 4.
    """
    if not a.values or not b.values:
        return f"{b.label} vs {a.label}: not enough successful runs to compare"
    d = b.mean - a.mean
    noise = max(a.stdev, b.stdev)
    verdict = ("smaller than the seed-to-seed spread -- not measured"
               if abs(d) <= noise else "larger than the spread")
    return (f"{b.label} vs {a.label}: {d:+.4f} "
            f"(spread {noise:.4f}) -- {verdict}")
