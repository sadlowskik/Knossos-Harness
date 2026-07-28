#!/usr/bin/env python
"""Profile the executor's per-step overhead, with the engine taken out.

    python scripts/profile_loop.py            # summary
    python scripts/profile_loop.py --full     # full cProfile table

A real turn is dominated by waiting on the model, which hides everything else.
That is exactly why this drives a scripted engine instead: what is left is the
harness's own cost, which is the part that can be fixed. The transcript is grown
to a realistic size first, because several of the suspected hot paths are
quadratic in transcript length and invisible on a short one.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knossos.ariadne import Ariadne                     # noqa: E402
from knossos.lethe import Lethe, estimate_tokens        # noqa: E402
from knossos.talos import Talos, Verdict                # noqa: E402
from knossos.workspace import Workspace                 # noqa: E402


class Engine:
    """Answers instantly, so the profile is all harness."""

    name = "profile"
    context_window = 16384

    def generate(self, prompt, context, cancelled):
        yield "Thinking about it."


def build(root: Path, turns: int, entry_chars: int):
    ws = Workspace(root)
    talos = Talos(Engine(), ws, lethe=Lethe(max_tokens=24_000),
                  ariadne=Ariadne(max_steps=12, target_steps=8),
                  verifier=lambda w, c: Verdict(False, "not yet"))
    # A transcript the size of a long session: tool results dominate, and
    # `_render_results` puts a whole turn's output into a single entry.
    talos.transcript = ["## Task\nfix the thing"]
    for i in range(turns):
        talos.transcript.append(f"\n## Assistant\nstep {i}")
        talos.transcript.append("\n## Tool results\n" + ("x" * entry_chars))
    return talos


def bench(label, fn, repeat):
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    elapsed = time.perf_counter() - start
    print(f"  {label:<34} {elapsed / repeat * 1000:8.2f} ms/call")
    return elapsed / repeat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--entry-chars", type=int, default=3_000)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    import tempfile
    root = Path(tempfile.mkdtemp(prefix="profile-"))
    (root / "a.py").write_text("x = 1\n", encoding="utf-8")

    talos = build(root, args.turns, args.entry_chars)
    size = sum(len(t) for t in talos.transcript)
    print(f"\n  transcript: {len(talos.transcript)} entries, {size:,} chars, "
          f"~{estimate_tokens(chr(10).join(talos.transcript)):,} tokens")
    print(f"  budget:     {talos.lethe.max_tokens:,} tokens\n")

    print("  component timings")
    bench("tools.render()", lambda: talos.tools.render(), args.repeat)
    bench("lethe.tokens(transcript)",
          lambda: talos.lethe.tokens(talos.transcript), args.repeat)

    # `_prompt` mutates the transcript by compacting it, so each call gets a
    # fresh copy -- otherwise the first call shrinks it and the rest measure
    # a different, smaller problem.
    saved = list(talos.transcript)

    def prompt_once():
        talos.transcript = list(saved)
        talos._prompt()

    per_step = bench("_prompt() [full per-step cost]", prompt_once, args.repeat)
    print(f"\n  a 12-step run spends {per_step * 12 * 1000:.0f} ms here, "
          f"outside the model\n")

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(args.repeat):
        prompt_once()
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(30 if args.full else 12)
    print("  cProfile (cumulative)")
    for line in stream.getvalue().splitlines():
        if line.strip():
            print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
