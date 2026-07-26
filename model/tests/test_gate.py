"""Tests for the retrieval gate.

The gate's job is asymmetric and the tests are weighted to match: a wrong skip
loses a repository answer, which is the whole point of the harness, while a wrong
inject loses a general answer the user could have got anywhere. So injecting is
the default and every skip must be argued for.

Measured motivation: on gemma4:e4b, injecting repo excerpts into a general
question dropped the answer score from 1.00 to 0.00 -- the model described this
codebase instead of answering. On llama-3.3-70b the same injection cost nothing.

    pytest -q tests/test_gate.py
"""
import textwrap

import pytest

from harness import Argus
from harness.gate import RetrievalGate


@pytest.fixture()
def gate(tmp_path):
    """A repo with one distinctive name and one thoroughly generic one."""
    (tmp_path / "mnemosyne.py").write_text(textwrap.dedent('''
        class MnemosyneBank:
            """Gist memory."""
            def forward(self, x):
                return x
    ''').strip(), encoding="utf-8")
    (tmp_path / "moe.py").write_text(textwrap.dedent('''
        class Router:
            def forward(self, x):
                return x

        def load_balance_loss(scores):
            return scores.mean()
    ''').strip(), encoding="utf-8")
    (tmp_path / "test_things.py").write_text(textwrap.dedent('''
        def test_router_does_not_collapse_when_training_a_layer():
            assert True

        def test_gradient_clipping_helps_stabilise_general_training():
            assert True
    ''').strip(), encoding="utf-8")

    argus = Argus(tmp_path)
    argus.scan()
    return RetrievalGate(argus)


# ------------------------------------------------------------------- defaults

def test_injecting_is_the_default(gate):
    """An unremarkable question gets context. Skipping is the exception."""
    d = gate.decide("how is the halting probability computed here")
    assert d.inject is True


def test_a_bare_question_still_injects(gate):
    assert gate.decide("what does forward do").inject is True


# ---------------------------------------------------------------------- skips

@pytest.mark.parametrize("query", [
    "what problem does rotary positional embedding solve, in general?",
    "how does KV caching speed up inference, typically?",
    "explain the concept of 4-bit quantization",
    "what is attention, conceptually?",
])
def test_generality_phrasing_skips(gate, query):
    d = gate.decide(query)
    assert d.inject is False, d


def test_indefinite_framing_skips(gate):
    """'in a mixture-of-experts layer' names a category, not this instance."""
    d = gate.decide("in a mixture-of-experts layer, what is expert collapse?")
    assert d.inject is False, d


# -------------------------------------------------------------------- vetoes

def test_an_anchor_overrides_generality(gate):
    """'which file' means here, whatever else the sentence says."""
    d = gate.decide("which file handles quantization, in general terms?")
    assert d.inject is True, d


def test_a_filename_overrides_generality(gate):
    d = gate.decide("what does moe.py do, generally?")
    assert d.inject is True, d


def test_a_proper_name_overrides_generality(gate):
    d = gate.decide("how does MnemosyneBank work, in general?")
    assert d.inject is True, d


def test_a_sentence_initial_capital_is_not_a_proper_name(gate):
    """Every sentence capitalises its first word; that is grammar, not a name.

    Regression: "Explain the concept of quantization" matched `_explain` and the
    leading capital made it look like a deliberate proper noun, vetoing a skip
    that should have happened.
    """
    d = gate.decide("Router the concept of expert collapse, in general?")
    assert d.inject is False, d


# ------------------------------------------------------------- name indexing

def test_test_function_names_do_not_become_repo_vocabulary(gate):
    """Their names are sentences, so indexing them makes everything look local.

    `test_gradient_clipping_helps_stabilise_general_training` would otherwise
    make "gradient", "clipping" and "training" read as distinctive repo terms.
    """
    index = gate._name_index()
    for word in ("gradient", "clipping", "stabilise", "collapse", "training"):
        assert word not in index, word


def test_a_distinctive_name_is_recognised(gate):
    assert "mnemosyne" in gate._name_index()
    assert gate._distinctive_hits("how does Mnemosyne compress a segment")


def test_short_tokens_are_never_distinctive(gate):
    assert gate._distinctive_hits("what is x") == []


# ------------------------------------------------------------------ reporting

def test_every_decision_explains_itself(gate):
    """A gate that cannot be argued with cannot be debugged."""
    for query in ("which file defines the router",
                  "what is attention, in general?"):
        d = gate.decide(query)
        assert d.reasons, query
        assert 0.0 <= d.confidence <= 1.0
        assert ("inject" in str(d)) or ("skip" in str(d))


def test_threshold_is_adjustable(gate):
    query = "how is the halting probability computed here"
    assert RetrievalGate(gate.argus, threshold=0.0).decide(query).inject is True
    assert RetrievalGate(gate.argus, threshold=1.01).decide(query).inject is False
