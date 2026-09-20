"""Isolation tests for Argus, the repo-wide index.

Three load-bearing claims, in order of how much they matter:

  1. Two files defining the same name both survive. This is the bug Scribe has
     -- its flat `Dict[str, dict]` lets the second `forward` overwrite the first
     -- and the whole point of a repo-wide index is that it cannot happen.
  2. A rescan with nothing changed re-parses nothing, and touching one file
     re-parses exactly one. Without this, indexing cannot run in an editor.
  3. Nothing approximate is labelled exact: `FileRecord.exact` is True only when
     a real parser produced the symbols.

No torch import here -- the harness does not depend on the model.

    pytest -q tests/test_argus.py
"""
import json
import textwrap
from collections import Counter

import pytest

from knossos import Argus
from knossos.argus import _rust_parser


@pytest.fixture()
def repo(tmp_path):
    """A miniature repo: two files defining `forward`, one importing the other."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "alpha.py").write_text(textwrap.dedent('''
        import math

        CONST = 3

        class Router:
            """Picks which experts speak."""
            def forward(self, x):
                return x

            def balance(self, scores):
                return scores.mean()

        def helper(a, b=2, *args, **kwargs):
            return a + b
    ''').strip(), encoding="utf-8")

    (tmp_path / "pkg" / "beta.py").write_text(textwrap.dedent('''
        from pkg.alpha import Router

        class Expert:
            def forward(self, x):
                return x * 2
    ''').strip(), encoding="utf-8")

    (tmp_path / "lib.rs").write_text(textwrap.dedent('''
        use std::collections::HashMap;

        pub struct Router {
            weights: Vec<f32>,
        }

        impl Router {
            pub fn forward(&self, x: f32) -> f32 {
                x
            }
        }

        pub fn balance(scores: &[f32]) -> f32 {
            scores.iter().sum()
        }
    ''').strip(), encoding="utf-8")

    argus = Argus(tmp_path)
    argus.scan()
    return argus


def test_same_name_in_two_files_both_survive(repo):
    hits = repo.lookup("forward")
    files = sorted(h.file for h in hits)
    assert files == ["lib.rs", "pkg/alpha.py", "pkg/beta.py"]
    assert len({h.ref for h in hits}) == 3           # three distinct locations


def test_qualname_disambiguates(repo):
    assert {h.qualname for h in repo.lookup("forward") if h.language == "python"} == {
        "Router.forward", "Expert.forward"}
    assert repo.lookup("Router.forward")[0].file == "pkg/alpha.py"


def test_python_symbols_are_exact(repo):
    rec = repo.files["pkg/alpha.py"]
    assert rec.exact is True
    by_name = {s.name: s for s in rec.symbols}
    assert by_name["helper"].signature == "def helper(a, b, *args, **kwargs)"
    assert by_name["Router"].kind == "class"
    assert by_name["forward"].kind == "method" and by_name["forward"].parent == "Router"
    # spans are real: the class body ends after its last method
    assert by_name["Router"].end_line > by_name["balance"].start_line


def test_rust_exactness_reflects_the_parser_that_ran(repo):
    """`exact` is a claim about provenance, not a constant.

    It was False unconditionally when the line scanner was the only Rust
    backend. With tree-sitter installed a real parse runs, and reporting that
    as approximate would understate what Oracle can rely on.
    """
    rec = repo.files["lib.rs"]
    assert rec.exact is (_rust_parser() is not None)
    kinds = {(s.name, s.kind) for s in rec.symbols}
    assert ("Router", "struct") in kinds
    assert ("balance", "fn") in kinds
    assert "std::collections::HashMap" in rec.imports


def test_rescan_is_incremental(repo, tmp_path):
    again = repo.scan()
    assert again.parsed == 0 and again.unchanged == again.scanned

    (tmp_path / "pkg" / "beta.py").write_text("def only_thing():\n    return 1\n", encoding="utf-8")
    after = repo.scan()
    assert after.parsed == 1
    assert [s.name for s in repo.symbols_in("pkg/beta.py")] == ["only_thing"]


def test_deleted_files_leave_the_index(repo, tmp_path):
    (tmp_path / "pkg" / "beta.py").unlink()
    report = repo.scan()
    assert report.removed == 1
    assert "pkg/beta.py" not in repo.files
    assert all(h.file != "pkg/beta.py" for h in repo.lookup("forward"))


def test_syntax_error_degrades_instead_of_crashing(repo, tmp_path):
    (tmp_path / "pkg" / "broken.py").write_text("def nope(:\n", encoding="utf-8")
    repo.scan()
    rec = repo.files["pkg/broken.py"]
    assert rec.exact is False and "SyntaxError" in rec.parse_error
    assert rec.idents                                # still lexically searchable


def test_importers_are_found(repo):
    assert repo.importers_of("pkg/alpha.py") == ["pkg/beta.py"]


def test_every_counter_field_survives_a_save_load_round_trip(repo, tmp_path):
    """Regression: `save()` used to drop two of the three Counter fields.

    `dataclasses.asdict` rebuilds a dict subclass by handing its constructor
    (key, value) tuples. `dict` accepts that; `Counter` counts the tuples as
    *elements*, producing `{("router", 3): 1}` -- tuple keys, which JSON
    refuses. Only `idents` had an explicit `dict()`, so adding `defs` and
    `path_terms` made every `session/new` raise TypeError.

    It only triggers on a non-empty Counter, so the fixture must have real
    content for this test to mean anything.
    """
    before = repo.files["pkg/alpha.py"]
    assert before.idents and before.defs and before.path_terms, "fixture must be non-empty"

    target = repo.save(tmp_path / "idx" / "index.json")
    assert target.exists()

    restored = Argus(repo.root)
    assert restored.load(target) is True

    after = restored.files["pkg/alpha.py"]
    for field in ("idents", "defs", "path_terms"):
        original, round_tripped = getattr(before, field), getattr(after, field)
        assert dict(round_tripped) == dict(original), f"{field} did not survive"
        assert all(isinstance(k, str) for k in round_tripped), f"{field} has non-str keys"
        # Counter, not a plain dict -- ranking calls Counter-only methods.
        assert isinstance(round_tripped, Counter), f"{field} came back as {type(round_tripped)}"


def test_an_index_from_an_older_schema_is_refused(repo, tmp_path):
    """A stale index must force a rebuild, not silently rank with blank signals.

    `load()` used to ignore the version it wrote. An index saved before `defs`
    existed would load happily, leave that Counter empty, and quietly switch off
    the signal that stops a symbol's *user* outranking its *definition*. A crash
    gets noticed; this did not.
    """
    target = repo.save(tmp_path / "idx" / "index.json")
    blob = json.loads(target.read_text(encoding="utf-8"))

    blob["version"] = 0                       # as if written by older code
    for rec in blob["files"].values():
        rec.pop("defs", None)
        rec.pop("path_terms", None)
    target.write_text(json.dumps(blob), encoding="utf-8")

    stale = Argus(repo.root)
    assert stale.load(target) is False, "an older schema must be refused"
    assert not stale.files, "nothing should be half-loaded from a refused index"


def test_retrieval_prefers_an_exact_symbol_match(repo):
    hits = repo.retrieve("fix the balance method", budget=4000)
    assert hits, "expected at least one slice"
    top = hits[0]
    assert top.symbol in {"Router.balance", "balance"}
    assert "exact symbol match" in top.reason


def test_a_rare_term_beats_a_repeated_common_one(tmp_path):
    """Regression: BM25 saturation, without which prose drowns out the answer.

    Under plain tf-idf a long file repeating "file" and "defines" hundreds of
    times outranked the short file that actually defines the thing asked about
    -- a real retrieval on this repo returned `knossos/argus.py` for a question
    about the MoE router. Term frequency has to stop paying after a few hits.
    """
    (tmp_path / "chatty.py").write_text(
        '"""' + "This file defines the file that defines files. " * 200 + '"""\n'
        "def unrelated_helper():\n    return 1\n", encoding="utf-8")
    (tmp_path / "moe.py").write_text(
        "def apollo_router(scores):\n"
        '    """Route tokens to experts."""\n'
        "    return scores\n", encoding="utf-8")

    argus = Argus(tmp_path)
    argus.scan()
    ranked = argus.score_files("which file defines the apollo_router?")
    assert ranked[0][0] == "moe.py", f"ranked {ranked}"


def test_retrieval_respects_its_budget(repo):
    hits = repo.retrieve("router expert forward balance", budget=200)
    assert len(hits) >= 1
    # one slice may exceed the budget alone; the sum of the rest may not
    assert sum(len(h.text) for h in hits[1:]) <= 200


def test_every_slice_carries_provenance(repo):
    for hit in repo.retrieve("router balance", budget=4000):
        assert hit.reason and hit.file and hit.start_line >= 1
        assert hit.ref.startswith(hit.file)


def test_save_load_round_trip(repo, tmp_path):
    before = {s.ref for s in repo.lookup("forward")}
    path = repo.save()

    fresh = Argus(tmp_path)
    assert fresh.load(path) is True
    assert {s.ref for s in fresh.lookup("forward")} == before

    report = fresh.scan()                            # nothing changed on disk
    assert report.parsed == 0


# --------------------------------------------------------------- rust parsing

RUST_DECOY = """
//! A doc comment mentioning `fn ghost_in_the_comment()`.

use std::collections::HashMap;

pub struct Router {
    weights: Vec<f32>,
}

impl Router {
    pub fn forward(&self, x: f32) -> f32 {
        let sql = "fn ghost_in_a_string(&self) -> ()";
        let _ = sql;
        x
    }
}

// fn ghost_in_a_line_comment() {}

pub fn balance(scores: &[f32]) -> f32 {
    scores.iter().sum()
}
"""


@pytest.fixture()
def rust_repo(tmp_path):
    (tmp_path / "lib.rs").write_text(RUST_DECOY, encoding="utf-8")
    argus = Argus(tmp_path)
    argus.scan()
    return argus


@pytest.mark.skipif(_rust_parser() is None, reason="tree-sitter not installed")
def test_rust_is_exact_when_tree_sitter_is_available(rust_repo):
    rec = rust_repo.files["lib.rs"]
    assert rec.exact is True, "a real parse must not be labelled approximate"


@pytest.mark.skipif(_rust_parser() is None, reason="tree-sitter not installed")
def test_declarations_inside_strings_and_comments_are_not_symbols(rust_repo):
    """The case the line scanner cannot get right.

    A regex sees `fn ghost_in_a_string` in a string literal and reports a
    function that does not exist. Retrieval then cites a definition that is not
    there, which is worse than missing it.
    """
    names = {s.name for s in rust_repo.files["lib.rs"].symbols}
    for ghost in (
        "ghost_in_a_string",
        "ghost_in_a_line_comment",
        "ghost_in_the_comment",
    ):
        assert ghost not in names, f"{ghost} is not a real definition"


@pytest.mark.skipif(_rust_parser() is None, reason="tree-sitter not installed")
def test_rust_methods_carry_their_impl_as_parent(rust_repo):
    """Same guarantee Python methods get: `Router.forward`, not a bare `forward`."""
    forward = [s for s in rust_repo.files["lib.rs"].symbols if s.name == "forward"]
    assert forward, "expected the method to be found"
    assert forward[0].parent == "Router"
    assert forward[0].qualname == "Router.forward"


@pytest.mark.skipif(_rust_parser() is None, reason="tree-sitter not installed")
def test_rust_kinds_and_extents_are_real(rust_repo):
    # `struct Router` and `impl Router` share a name -- key by kind as well, or
    # one silently shadows the other.
    by_key = {(s.name, s.kind): s for s in rust_repo.files["lib.rs"].symbols}
    assert ("Router", "struct") in by_key
    assert ("Router", "impl") in by_key
    assert ("balance", "fn") in by_key
    by_name = {s.name: s for s in rust_repo.files["lib.rs"].symbols}
    # A real parse knows where the impl block ends; brace-counting guesses.
    impl = [s for s in rust_repo.files["lib.rs"].symbols if s.kind == "impl"][0]
    assert impl.end_line > by_name["forward"].start_line


@pytest.mark.skipif(_rust_parser() is None, reason="tree-sitter not installed")
def test_rust_imports_are_extracted(rust_repo):
    assert "std::collections::HashMap" in rust_repo.files["lib.rs"].imports


def test_the_lexical_fallback_still_finds_declarations(tmp_path):
    """Without tree-sitter, Rust must degrade rather than disappear.

    The scanner is allowed to be wrong about ghosts; it is not allowed to find
    nothing, and it must never claim to be exact.
    """
    argus = Argus(tmp_path)
    rec = argus._parse("lib.rs", RUST_DECOY, "sha")
    rec.symbols.clear()
    rec.imports.clear()
    rec.exact = False
    argus._scan_rust_lexical(rec, RUST_DECOY)

    names = {s.name for s in rec.symbols}
    assert {"Router", "balance", "forward"} <= names
    assert rec.exact is False, "the fallback must never be reported as exact"


def test_definitions_field_is_populated_for_rust(rust_repo):
    """Ranking leans hardest on `defs`; Rust must contribute to it too."""
    defs = rust_repo.files["lib.rs"].defs
    assert defs.get("router", 0) > 0
    assert defs.get("balance", 0) > 0


def test_a_byte_order_mark_does_not_make_a_file_unindexable(tmp_path):
    """A BOM is legal UTF-8 and Windows editors write them.

    `ast.parse` rejects U+FEFF at position 0, so without utf-8-sig decoding the
    file parses as nothing: no symbols, no `defs`, no ranking contribution --
    and no error anyone sees. This repo's own acp.py, oracle.py and
    workspace.py were silently unindexable for exactly this reason.
    """
    source = "class Router:\n    def forward(self, x):\n        return x\n"
    (tmp_path / "bom.py").write_bytes(b"\xef\xbb\xbf" + source.encode("utf-8"))
    (tmp_path / "plain.py").write_text(source, encoding="utf-8")

    argus = Argus(tmp_path)
    argus.scan()

    bom = argus.files["bom.py"]
    assert bom.parse_error is None, f"BOM broke parsing: {bom.parse_error}"
    assert bom.exact is True
    assert {s.name for s in bom.symbols} == {"Router", "forward"}
    # And it contributes to ranking exactly like the plain file.
    assert bom.defs == argus.files["plain.py"].defs
