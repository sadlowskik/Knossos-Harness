"""A Knossos ACP server whose engine is scripted rather than a model.

Everything below the engine is the real thing: the real `KnossosAgent`, the
real `Talos` loop, the real workspace jail and permission gate. Only the model's
replies are fixed, which is what makes execute-mode conformance checks
deterministic -- a real model may or may not decide to write a file on any given
run, and a conformance suite that depends on that is measuring the model.

The script comes in as JSON on `KNOSSOS_SCRIPT`, one reply per engine turn:

    KNOSSOS_SCRIPT='["```json\\n{\\"tool\\": ...}\\n```", "Done."]'

Not importable by the package itself, and not on the CLI, deliberately: a
`--engine scripted` flag would be a way to drive the agent from outside with
arbitrary tool calls, which is exactly the thing the permission gate exists to
prevent.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from knossos.acp import KnossosAgent            # noqa: E402
from knossos.jsonrpc import _configure_stdio    # noqa: E402


class ScriptedEngine:
    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)

    def generate(self, prompt, context, cancelled):
        yield self.replies.pop(0) if self.replies else "Nothing further."


def main() -> int:
    replies = json.loads(os.environ.get("KNOSSOS_SCRIPT", "[]"))
    agent = KnossosAgent(
        engine=ScriptedEngine(replies),
        execute=os.environ.get("KNOSSOS_EXECUTE") == "1",
        dry_run=os.environ.get("KNOSSOS_WRITE") != "1",
        gate=False,
    )
    _configure_stdio()
    agent.serve().serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
