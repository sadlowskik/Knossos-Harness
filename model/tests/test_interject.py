"""Steering a run without ending it.

Cancellation was the only mid-run control: the loop finished or it was killed.
That fits the failure it sees least. The common one is that the agent is doing
something reasonable and slightly wrong, and the person watching knows the
sentence that would fix it.

No torch, no network.

    pytest -q tests/test_interject.py
"""
import threading

from knossos.interject import CAPACITY, Interjections


def test_nothing_queued_produces_no_note():
    i = Interjections()
    assert i.take_note() is None
    assert not i


def test_interjections_arrive_in_the_order_they_were_said():
    i = Interjections()
    assert i.push("use the existing helper")
    assert i.push("and do not touch the tests")

    note = i.take_note()
    assert note.index("use the existing helper") < note.index("and do not touch")


def test_several_corrections_become_one_note_not_several_turns():
    i = Interjections()
    for word in ("one", "two", "three"):
        i.push(word)

    note = i.take_note()
    assert all(word in note for word in ("one", "two", "three"))
    assert not i, "taking the note must clear the queue"
    assert i.take_note() is None, "and must not deliver it twice"


def test_the_note_says_it_came_from_the_person():
    # Without this the agent reads a bare instruction sandwiched between tool
    # results and has no way to weigh it differently.
    i = Interjections()
    i.push("stop rewriting the parser")
    assert "user interrupted" in i.take_note()


def test_empty_and_whitespace_input_is_refused_rather_than_queued():
    i = Interjections()
    assert not i.push("")
    assert not i.push("   \n  ")
    assert not i


def test_a_full_queue_refuses_rather_than_growing():
    i = Interjections()
    for n in range(CAPACITY):
        assert i.push(f"note {n}")
    assert not i.push("one too many")
    assert len(i) == CAPACITY


def test_pushing_from_another_thread_is_visible_to_the_loop():
    # The whole point: the server reads its socket on one thread and the
    # executor runs on another.
    i = Interjections()
    t = threading.Thread(target=lambda: i.push("said from elsewhere"))
    t.start()
    t.join()
    assert "said from elsewhere" in i.take_note()


def test_the_loop_delivers_a_queued_note_at_the_next_step(tmp_path):
    """End to end through Talos, not just the queue in isolation."""
    from knossos.ariadne import Ariadne
    from knossos.talos import Talos, Verdict
    from knossos.workspace import Workspace

    class Engine:
        """Records the transcript it was handed on each turn."""

        name = "probe"
        context_window = 16384

        def __init__(self):
            self.prompts = []

        def generate(self, prompt, context, cancelled):
            self.prompts.append(prompt)
            yield "still thinking"

    engine = Engine()
    talos = Talos(engine, Workspace(tmp_path),
                  ariadne=Ariadne(max_steps=3, target_steps=2),
                  verifier=lambda w, c: Verdict(False, "not yet"))

    talos.interjections.push("actually, name it quadruple")
    talos.run("do the thing", ["step one"])

    assert engine.prompts, "the engine should have been called"
    assert any("quadruple" in p for p in engine.prompts), (
        "the note never reached a request")
    assert any("user interrupted" in p for p in engine.prompts), (
        "it must be marked as coming from the person")
    assert not talos.interjections, "nothing may be left queued"


def test_render_does_not_consume_so_the_loop_can_trace_it_too():
    # The loop needs the notes twice, once for the transcript and once for the
    # trace; draining twice would return nothing the second time.
    i = Interjections()
    i.push("a correction")
    notes = i.drain()
    assert Interjections.render(notes) == Interjections.render(notes)
    assert "a correction" in Interjections.render(notes)
