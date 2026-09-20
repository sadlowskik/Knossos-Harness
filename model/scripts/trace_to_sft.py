"""Turn collected traces into supervised fine-tuning records.

    python scripts/trace_to_sft.py --traces .knossos --out corpus.jsonl

Input is JSONL written by `knossos-rs` with `--collect-exchanges`: one
`exchange_delta` event per engine call, carrying only the request suffix and
the exact reply, plus the evaluation labels that say whether the run worked.
Legacy full-prefix `exchange` events remain readable.

Output is one JSON object per line in the OpenAI chat shape, because that is
what every SFT trainer already reads:

    {"messages": [{"role": "system", ...}, ..., {"role": "assistant", ...}]}

**The last message is the target.** Everything before it is context. Mask the
loss to that final assistant turn; training on the whole record teaches the
model to produce the tool output it was given, which is both wrong and easy to
do by accident.

What "successful" means here
----------------------------

For v2 traces, the deterministic `evaluation_finished` label must report both
`grader_pass` and `verifier_pass`, with no provider, infrastructure, or tamper
failure. This label outranks the harness halt because historical traces proved
that a halt can disagree with the external grader. Legacy traces fall back to
`Halt.Done`; use the stricter curator before training on them.

Where the input format is defined
---------------------------------

`knossos-rs/src/session.rs` (`TraceEvent::ExchangeDelta`) and
`knossos-rs/src/engine/types.rs` (`Content`, `Message`, `Role`). This reader
was written from those and its tests use events constructed by hand to match
them -- so the two sides are pinned to the same shape by inspection, not by a
shared fixture. If `Content` gains a variant or renames its `kind` tag, nothing
here fails; the corpus just quietly loses those blocks. Run the check in
`--help` against a real collected trace after any change to either file.

The limitation worth knowing before you train
---------------------------------------------

Filtering is per *trajectory*, not per *step*. A run that took a wrong turn at
step 3, noticed at step 4 and recovered by step 6 is kept whole, so the wrong
turn is in the corpus as something to imitate. Rejection sampling always has
this property; fixing it needs a per-step reward, which the trace does not
carry and the Oracle does not produce. Worth measuring before assuming it is
harmless: a corpus of recoveries may teach recovery, or may teach the mistake.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from typing import Any, Dict, Iterable, Iterator, List, Optional

#: The only halt that counts as success. See `ariadne::Halt`.
SUCCESS = "done"


def read_trace(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Every well-formed event in one trace, in order.

    A truncated final line is normal -- a killed run leaves one -- and is
    skipped rather than treated as a corrupt file.
    """
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def succeeded(events: Iterable[Dict[str, Any]]) -> bool:
    """Whether the strongest available label says the trajectory succeeded."""
    events = list(events)
    evaluations = [e for e in events if e.get("event") == "evaluation_finished"]
    if evaluations:
        label = evaluations[-1]
        halts = [e for e in events if e.get("event") == "halt"]
        metadata = [e for e in events if e.get("event") == "experiment_metadata"]
        clarification = bool(
            metadata and metadata[-1].get("expected_action") == "clarify"
        )
        completion_consistent = (
            not halts or halts[-1].get("reason") == SUCCESS or clarification
        )
        return (
            completion_consistent
            and label.get("grader_pass") is True
            and label.get("verifier_pass") is True
            and label.get("provider_status") == "ok"
            and label.get("infrastructure_status") == "ok"
            and label.get("tamper") is False
        )
    for event in events:
        if event.get("event") == "halt" and event.get("reason") == SUCCESS:
            return True
        if event.get("event") == "task_finished" and event.get("outcome") == SUCCESS:
            return True
    return False


def _tool_calls(blocks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """`tool_use` blocks in OpenAI's `tool_calls` shape.

    `arguments` is a JSON *string* there, not an object. Getting that wrong
    produces a corpus that parses but trains the model to emit the wrong type.
    """
    return [
        {
            "id": b["id"],
            "type": "function",
            "function": {
                "name": b["name"],
                "arguments": json.dumps(b.get("input", {}), ensure_ascii=False),
            },
        }
        for b in blocks
        if b.get("kind") == "tool_use"
    ]


def _text(blocks: Iterable[Dict[str, Any]]) -> str:
    return "\n".join(b["text"] for b in blocks if b.get("kind") == "text")


def to_chat(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One harness message as one or more OpenAI messages.

    The shapes do not correspond one to one. A single user message here can
    carry several tool results, and OpenAI wants one `tool` message each,
    immediately after the assistant turn that requested them -- so results are
    emitted before any prose in the same message, or the ordering constraint
    breaks.
    """
    blocks = message.get("content") or []
    role = message.get("role")
    out: List[Dict[str, Any]] = []

    if role == "assistant":
        calls = _tool_calls(blocks)
        entry: Dict[str, Any] = {"role": "assistant", "content": _text(blocks) or None}
        if calls:
            entry["tool_calls"] = calls
        out.append(entry)
        return out

    for b in blocks:
        if b.get("kind") == "tool_result":
            out.append({
                "role": "tool",
                "tool_call_id": b["id"],
                # `is_error` has no home in this schema. The content already
                # reads as a failure, and inventing a field would break the
                # trainers this format exists to satisfy.
                "content": b.get("content", ""),
            })
    prose = _text(blocks)
    if prose:
        out.append({"role": "user", "content": prose})
    return out


def records(events: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """One SFT record per exchange: the prompt as sent, the reply as target."""
    system = ""
    request_messages: List[Dict[str, Any]] = []
    for event in events:
        kind = event.get("event")
        if kind not in {"exchange", "exchange_delta"}:
            continue
        if kind == "exchange":
            request, response = event.get("request") or {}, event.get("response") or {}
            system = request.get("system") or ""
            request_messages = list(request.get("messages") or [])
        else:
            response = event.get("response") or {}
            if event.get("reset"):
                system = event.get("system") or ""
                request_messages = []
            start = event.get("messages_start")
            if not isinstance(start, int) or start < 0 or start > len(request_messages):
                # A missing prefix is corruption, not a reason to silently
                # train on a different prompt than the model actually saw.
                continue
            request_messages[start:] = list(event.get("messages") or [])

        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        for m in request_messages:
            messages.extend(to_chat(m))

        target = to_chat({"role": "assistant", "content": response.get("content") or []})
        if not target:
            continue
        # An assistant turn with neither prose nor a call is not something to
        # teach; it is the empty reply the harness already has a note for.
        if not target[0].get("content") and not target[0].get("tool_calls"):
            continue
        messages.extend(target)

        yield {
            "messages": messages,
            "step": event.get("step"),
            "run_id": event.get("run_id"),
            "trace_schema": event.get("schema_version", "knossos-trace/v1"),
        }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, type=pathlib.Path,
                    help="a trace file, or a directory of them")
    ap.add_argument("--out", type=pathlib.Path,
                    help="where to write; stdout by default")
    ap.add_argument("--all", action="store_true",
                    help="keep failed trajectories too. Off by default: a run "
                         "that did not reach Done is a demonstration of "
                         "something you do not want imitated")
    args = ap.parse_args(argv)

    if args.traces.is_dir():
        paths = sorted(args.traces.rglob("*.jsonl"))
    else:
        paths = [args.traces]
    if not paths:
        print(f"no traces under {args.traces}", file=sys.stderr)
        return 1

    stats = Counter()
    out = args.out.open("w", encoding="utf-8") if args.out else sys.stdout
    try:
        for path in paths:
            events = read_trace(path)
            if not any(e.get("event") in {"exchange", "exchange_delta"} for e in events):
                # Almost always the real cause: the run was not collecting.
                stats["no exchanges (run without --collect-exchanges?)"] += 1
                continue
            stats["trajectories"] += 1
            if not succeeded(events) and not args.all:
                stats["dropped (did not reach Done)"] += 1
                continue
            stats["kept"] += 1
            for record in records(events):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["records"] += 1
    finally:
        if args.out:
            out.close()

    for key, n in stats.most_common():
        print(f"  {n:>6}  {key}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
