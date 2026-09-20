"""Saying something to a run that is already going.

Cancellation was the only mid-run control the harness had: the loop either
finished or it was killed. That is a poor fit for the failure it actually sees
most -- the agent is doing something reasonable and slightly wrong, and the
person watching knows the one sentence that would fix it. Killing the run throws
away the context that made it nearly right; waiting for it to finish spends the
whole step budget being wrong on purpose.

So: a queue the loop drains between steps.

Why only between steps
----------------------

A step is not interruptible in the middle, and the reason is structural rather
than an implementation shortcut. Inside a step the conversation passes through
states that are not valid to append to -- most sharply, between an assistant
turn carrying tool calls and the tool results answering them. Text inserted
there produces a request the provider rejects, and the agent loses the whole
turn to a protocol error rather than gaining a correction.

The step boundary is the one point where the transcript is a complete exchange.
Everything queued lands there, in order, as a single note -- so three
corrections typed in quick succession cost one turn rather than three.

Mirrors `knossos-rs/src/interject.rs`.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Deque, List, Optional

__all__ = ["Interjections", "CAPACITY"]

#: How many pending interjections are held before new ones are refused.
#:
#: A person typing will never reach this. A programmatic producer -- a front end
#: forwarding notifications, a test in a loop -- can, and an unbounded queue
#: behind an agent that may be busy for minutes is a slow leak with a
#: respectable-looking name.
CAPACITY = 64


class Interjections:
    """A place to put words into a running loop.

    Safe to share across threads: the ACP server reads its socket on one and the
    executor runs on another, and both address the same queue.
    """

    def __init__(self) -> None:
        self._queue: Deque[str] = deque()
        self._lock = threading.Lock()

    def push(self, text: str) -> bool:
        """Queue text for the agent to read at the next step boundary.

        Returns False if it was not accepted, so a front end can say so.
        Silently dropping it would be worse than refusing: the person would
        believe the agent had been told.
        """
        if not text or not text.strip():
            # An empty interjection would spend a turn saying nothing.
            return False
        with self._lock:
            if len(self._queue) >= CAPACITY:
                return False
            self._queue.append(text)
            return True

    def drain(self) -> List[str]:
        """Take everything queued, oldest first."""
        with self._lock:
            out = list(self._queue)
            self._queue.clear()
        return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)

    def __bool__(self) -> bool:
        return len(self) > 0

    @staticmethod
    def render(notes: List[str]) -> str:
        """Already-drained notes, as the block the agent sees.

        Split from `drain` because the loop needs the notes twice -- once for
        the transcript, once for the trace -- and draining twice returns
        nothing the second time.

        Marked as coming from the person rather than presented as bare text: the
        agent is mid-task and needs to tell a new instruction from the tool
        output and budget notes around it.
        """
        lines = ["The user interrupted with the following. Take it into "
                 "account before continuing:"]
        lines += [f"- {note.strip()}" for note in notes]
        return "\n".join(lines)

    def take_note(self) -> Optional[str]:
        """Everything queued as one block, or None if nothing is waiting."""
        pending = self.drain()
        return self.render(pending) if pending else None
