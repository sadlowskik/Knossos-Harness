"""Argus: the hundred-eyed watchman -- he sees the whole repository at once.

Scribe parses one file exactly. Argus watches every file and answers the question
the harness actually asks before it can do anything useful:

    "given this task, which parts of this codebase belong in the context window?"

That is the retrieval problem Cursor solves with embeddings over chunked files.
Argus solves the same problem structurally instead, for three reasons that matter
at this project's scale: it needs no model and no vector store, it is incremental
enough to run on every keystroke, and every result carries a *reason* -- which is
what lets Oracle audit a retrieval instead of trusting it.

Three layers, in order of how much they can be trusted:

    1. Symbols   what is defined, where. From a real parser where one exists.
    2. Edges     what imports what, what mentions what. Same parse, no extra cost.
    3. Ranking   identifier overlap with the request, BM25 across the repo.

Retrieval seeds from layer 3, expands along layer 2, and returns slices of layer 1
until a character budget is spent. Nothing is approximated silently: every
`FileRecord` carries `exact`, and it is False whenever the symbols came from the
line scanner rather than a parser.

Incremental by content hash: a scan re-parses only files whose SHA-1 changed, so
the second scan of a large repo costs one `stat` and one read per file. This is
the same idea as Cursor's Merkle sync, minus the tree.

Language support:

    Python    exact, via the stdlib `ast` module.
    Rust      approximate, via a declaration scanner (see `_scan_rust`). Rust
              declarations are regular enough that this finds them reliably, but
              it does not resolve types, generics, or cross-file references.
              `exact=False` says so. The upgrade path is a tree-sitter backend
              behind the same `_parse` seam; inside a Lapce fork the better
              upgrade is rust-analyzer, which already has a real index.
    other     lexical only -- indexed for identifier search, no symbols.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = ["Argus", "Symbol", "FileRecord", "Retrieved", "ScanReport", "Context", "render"]

DEFAULT_INCLUDE = ("*.py", "*.rs", "*.toml", "*.md")
DEFAULT_EXCLUDE = (
    ".git", ".argus", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv",
    "venv", "node_modules", "target", "build", "dist", ".ipynb_checkpoints",
)
MAX_FILE_BYTES = 1_000_000

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Identifiers this common carry no signal about *which* file you want.
_STOP = frozenset("""
self cls the a an and or not if else for while in is to of it as with return def
class import from pub fn let mut use mod impl struct enum trait match where crate
super Self None True False true false new int str dtype torch nn F x y z i j k n
""".split())


@dataclass(frozen=True)
class Symbol:
    """One definition, located exactly."""
    name: str
    kind: str                       # function | class | method | struct | enum | trait | impl | mod | const
    file: str                       # repo-relative, posix separators
    start_line: int                 # 1-indexed, inclusive
    end_line: int                   # 1-indexed, inclusive
    signature: str
    parent: Optional[str] = None
    language: str = "python"

    @property
    def qualname(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name

    @property
    def ref(self) -> str:
        """A `file:line` string -- clickable in every editor worth forking."""
        return f"{self.file}:{self.start_line}"


@dataclass
class FileRecord:
    """Everything Argus knows about one file, plus how much to trust it."""
    path: str
    sha: str
    language: str
    n_lines: int
    exact: bool                     # did a real parser produce these symbols?
    symbols: List[Symbol] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    idents: Counter = field(default_factory=Counter)
    #: Words from the names this file *defines*, as opposed to merely mentions.
    #: Scored as its own field: `moe.py` defines `Router`, `naiads.py` only uses
    #: one, and flat term frequency cannot tell those apart -- it ranked the
    #: user of a symbol above its definition.
    defs: Counter = field(default_factory=Counter)
    #: Words from the file's own path. "moe" in a question should favour
    #: `daedalus/moe.py` over a file that happens to say "moe" a lot.
    path_terms: Counter = field(default_factory=Counter)
    parse_error: Optional[str] = None


@dataclass
class Retrieved:
    """One slice of source, and the reason it was pulled."""
    file: str
    start_line: int
    end_line: int
    text: str
    score: float
    reason: str
    symbol: Optional[str] = None

    @property
    def ref(self) -> str:
        return f"{self.file}:{self.start_line}"


class Context(str):
    """Rendered excerpts that still remember what they were made of.

    An engine that reasons over prose wants the flat string; an engine that
    reports on the retrieval wants the hits. Subclassing `str` means the string
    contract is unchanged -- `context.strip()`, f-strings and tokenizers all work
    -- while `.hits` stays available to anything that needs the structure.

    The alternative, recovering structure by re-parsing the rendered text, breaks
    the moment a retrieved file contains a line that looks like a header. A
    markdown README is enough to do it.
    """

    hits: "List[Retrieved]"

    def __new__(cls, text: str, hits: "Sequence[Retrieved]" = ()) -> "Context":
        obj = super().__new__(cls, text)
        obj.hits = list(hits)
        return obj


def render(hits: "Sequence[Retrieved]") -> Context:
    """Pack retrieved slices into a prompt block, provenance on every one."""
    body = "\n\n".join(
        f"# {h.ref}-{h.end_line}"
        f"{'  (' + h.symbol + ')' if h.symbol else ''}  [{h.reason}]\n{h.text}"
        for h in hits)
    return Context(body, hits)


@dataclass
class ScanReport:
    scanned: int = 0
    parsed: int = 0                 # re-parsed because the hash changed
    unchanged: int = 0
    removed: int = 0
    failed: List[Tuple[str, str]] = field(default_factory=list)

    def __str__(self) -> str:
        s = (f"scanned {self.scanned} | parsed {self.parsed} | "
             f"unchanged {self.unchanged} | removed {self.removed}")
        return s + f" | failed {len(self.failed)}" if self.failed else s


def _split_identifier(tok: str) -> List[str]:
    """`RecurrentMoECore` -> [recurrentmoecore, recurrent, mo, e, core]-ish.

    Both halves matter: the whole token matches an exact symbol name, the parts
    match a request phrased in prose.
    """
    parts = [tok.lower()]
    for chunk in _CAMEL.sub(" ", tok).replace("_", " ").split():
        low = chunk.lower()
        if len(low) > 2 and low not in parts:
            parts.append(low)
    return parts


def _tokenize(text: str) -> Counter:
    bag: Counter = Counter()
    for tok in _IDENT.findall(text):
        if tok in _STOP or len(tok) < 3:
            continue
        for part in _split_identifier(tok):
            if part not in _STOP:
                bag[part] += 1
    return bag


class Argus:
    """A repository index that answers 'what should I load to do this?'.

    >>> argus = Argus("/path/to/repo")
    >>> argus.scan()
    >>> for hit in argus.retrieve("how does the router balance experts", budget=4000):
    ...     print(hit.ref, hit.reason)
    """

    def __init__(self, root: str | Path,
                 include: Sequence[str] = DEFAULT_INCLUDE,
                 exclude: Sequence[str] = DEFAULT_EXCLUDE) -> None:
        self.root = Path(root).resolve()
        self.include = tuple(include)
        self.exclude = set(exclude)
        self.files: Dict[str, FileRecord] = {}

    # ---------------------------------------------------------------- scanning

    def _walk(self) -> Iterable[Path]:
        for pattern in self.include:
            for path in self.root.rglob(pattern):
                if not path.is_file():
                    continue
                if any(part in self.exclude for part in path.relative_to(self.root).parts):
                    continue
                yield path

    def scan(self) -> ScanReport:
        """Index the repo, re-parsing only what changed since the last scan."""
        report = ScanReport()
        seen: Set[str] = set()

        for path in self._walk():
            rel = path.relative_to(self.root).as_posix()
            seen.add(rel)
            report.scanned += 1
            try:
                raw = path.read_bytes()
            except OSError as exc:
                report.failed.append((rel, str(exc)))
                continue
            if len(raw) > MAX_FILE_BYTES:
                continue

            sha = hashlib.sha1(raw).hexdigest()
            cached = self.files.get(rel)
            if cached is not None and cached.sha == sha:
                report.unchanged += 1
                continue

            text = raw.decode("utf-8", errors="replace")
            record = self._parse(rel, text, sha)
            self.files[rel] = record
            report.parsed += 1
            if record.parse_error:
                report.failed.append((rel, record.parse_error))

        for gone in set(self.files) - seen:
            del self.files[gone]
            report.removed += 1
        return report

    def _parse(self, rel: str, text: str, sha: str) -> FileRecord:
        suffix = Path(rel).suffix
        language = {".py": "python", ".rs": "rust", ".toml": "toml", ".md": "markdown"}.get(suffix, "text")
        record = FileRecord(path=rel, sha=sha, language=language,
                            n_lines=text.count("\n") + 1,
                            exact=(language == "python"))
        record.idents = _tokenize(text)

        if language == "python":
            try:
                self._scan_python(record, text)
            except SyntaxError as exc:
                record.exact = False
                record.parse_error = f"SyntaxError: {exc}"
        elif language == "rust":
            self._scan_rust(record, text)

        record.defs = self._definition_terms(record)
        record.path_terms = self._path_terms(rel)
        return record

    @staticmethod
    def _definition_terms(record: FileRecord) -> Counter:
        """Words from the names this file defines.

        Test functions are excluded: their names are sentences
        (`test_router_does_not_collapse`), so counting them as definitions makes
        a test file look like the definitive source for every word in its own
        name.
        """
        bag: Counter = Counter()
        for sym in record.symbols:
            if sym.name.startswith("test") or (sym.parent or "").startswith("Test"):
                continue
            # A type is a stronger landmark than a method, so it counts twice.
            weight = 2 if sym.kind in ("class", "struct", "enum", "trait") else 1
            for part in _split_identifier(sym.name):
                if part not in _STOP:
                    bag[part] += weight
        return bag

    @staticmethod
    def _path_terms(rel: str) -> Counter:
        bag: Counter = Counter()
        for chunk in re.split(r"[/\\.]", rel):
            for part in _split_identifier(chunk):
                if len(part) >= 2 and part not in _STOP:
                    bag[part] += 1
        return bag

    # ------------------------------------------------------- language backends

    def _scan_python(self, record: FileRecord, text: str) -> None:
        """Exact, via the stdlib parser. Same ground-truth guarantee as Scribe."""
        tree = ast.parse(text)

        def signature(node) -> str:
            a = node.args
            names = [x.arg for x in (*a.posonlyargs, *a.args)]
            if a.vararg:
                names.append("*" + a.vararg.arg)
            names += [x.arg for x in a.kwonlyargs]
            if a.kwarg:
                names.append("**" + a.kwarg.arg)
            prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
            return f"{prefix}{node.name}({', '.join(names)})"

        def end_of(node) -> int:
            return getattr(node, "end_lineno", None) or node.lineno

        def visit(node, parent: Optional[str]) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    record.symbols.append(Symbol(
                        name=child.name, kind="method" if parent else "function",
                        file=record.path, start_line=child.lineno, end_line=end_of(child),
                        signature=signature(child), parent=parent, language="python"))
                    visit(child, parent)
                elif isinstance(child, ast.ClassDef):
                    bases = [b.id for b in child.bases if isinstance(b, ast.Name)]
                    record.symbols.append(Symbol(
                        name=child.name, kind="class", file=record.path,
                        start_line=child.lineno, end_line=end_of(child),
                        signature=f"class {child.name}({', '.join(bases)})" if bases else f"class {child.name}",
                        parent=parent, language="python"))
                    visit(child, child.name)
                else:
                    visit(child, parent)

        visit(tree, None)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                record.imports.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                # A relative import resolves against this file's package.
                if node.level:
                    pkg = Path(record.path).parent.as_posix().replace("/", ".")
                    record.imports.append(f"{pkg}.{node.module}" if pkg else node.module)
                else:
                    record.imports.append(node.module)

    _RUST_DECL = re.compile(
        r"""^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?(?:default[ \t]+)?(?:const[ \t]+)?
            (?:async[ \t]+)?(?:unsafe[ \t]+)?(?:extern[ \t]+"[^"]*"[ \t]+)?
            (?P<kind>fn|struct|enum|trait|impl|mod|type|const|static)[ \t]+
            (?P<rest>[^\n{;]*)""",
        re.VERBOSE | re.MULTILINE,
    )
    _RUST_USE = re.compile(r"^[ \t]*(?:pub[ \t]+)?use[ \t]+([^;]+);", re.MULTILINE)

    def _scan_rust(self, record: FileRecord, text: str) -> None:
        """Approximate. `record.exact` is False and stays False.

        A declaration scanner, not a parser: it finds `fn`/`struct`/`impl`/... at
        the start of a line and takes the name that follows. It does not resolve
        types or generics, and it will happily report a declaration that appears
        inside a string or a `cfg`-disabled block. Good enough to seed retrieval,
        not good enough for Oracle to assert anything about.
        """
        lines = text.splitlines()

        def block_end(start_idx: int) -> int:
            """Brace-match forward from the declaration line. Naive about strings."""
            depth, started = 0, False
            for i in range(start_idx, min(len(lines), start_idx + 2000)):
                for ch in lines[i]:
                    if ch == "{":
                        depth += 1
                        started = True
                    elif ch == "}":
                        depth -= 1
                if started and depth <= 0:
                    return i + 1
                if not started and lines[i].rstrip().endswith(";"):
                    return i + 1
            return min(len(lines), start_idx + 1)

        for m in self._RUST_DECL.finditer(text):
            kind, rest = m.group("kind"), m.group("rest").strip()
            name_match = _IDENT.search(rest)
            if not name_match:
                continue
            name = name_match.group(0)
            if kind == "impl":
                # `impl Trait for Type` / `impl<T> Type` -- the last ident is the type.
                idents = [t for t in _IDENT.findall(rest) if t != "for"]
                name = idents[-1] if idents else name
            start = text.count("\n", 0, m.start()) + 1
            record.symbols.append(Symbol(
                name=name, kind=kind, file=record.path,
                start_line=start, end_line=block_end(start - 1),
                signature=f"{kind} {rest}".strip(), language="rust"))

        for m in self._RUST_USE.finditer(text):
            record.imports.append(m.group(1).strip().replace("\n", " "))

    # ----------------------------------------------------------------- queries

    def lookup(self, name: str) -> List[Symbol]:
        """Every definition of `name`, across the repo.

        Returns a *list* -- the thing Scribe could not do. Two files both
        defining `forward` give you two symbols, not one silently overwriting
        the other.
        """
        return [s for rec in self.files.values() for s in rec.symbols
                if s.name == name or s.qualname == name]

    def symbols_in(self, path: str) -> List[Symbol]:
        rec = self.files.get(path)
        return list(rec.symbols) if rec else []

    def importers_of(self, path: str) -> List[str]:
        """Files whose imports plausibly resolve to `path`.

        Module-path matching, not real resolution: `daedalus/moe.py` matches an
        import of `daedalus.moe` or `moe`. Good enough to walk one hop out.
        """
        stem = Path(path).with_suffix("").as_posix()
        dotted = stem.replace("/", ".")
        tail = Path(path).stem
        out = []
        for rel, rec in self.files.items():
            if rel == path:
                continue
            for imp in rec.imports:
                norm = imp.replace("::", ".")
                if norm == dotted or norm.endswith("." + tail) or norm == tail:
                    out.append(rel)
                    break
        return out

    # BM25 parameters (Robertson & Walker 1994). `k1` saturates term frequency:
    # the 50th "file" in a file about files adds almost nothing over the 5th.
    # `b` controls how hard long documents are penalised.
    BM25_K1 = 1.5
    BM25_B = 0.75

    #: Field weights. Mentioning a name is weak evidence; defining it is strong;
    #: being named after it is stronger still. Single-field scoring ranked
    #: `naiads.py` -- which imports `Router` -- above `moe.py`, which defines it,
    #: purely because naiads.py is shorter and says the word more often.
    W_TEXT = 1.0
    W_DEFS = 2.5
    W_PATH = 2.0

    #: Files whose job is to talk *about* code -- tests, fixtures, eval sets --
    #: quote identifiers without being their home. They stay findable (ask about
    #: a test by name and you still get it), just outranked by the real thing.
    META_PENALTY = 0.6

    def _idf(self, field: str = "idents") -> Dict[str, float]:
        """Inverse document frequency, computed per field.

        A term common in prose can be rare among definitions, and that contrast
        is exactly the signal -- so the fields cannot share one idf table.
        """
        n_docs = max(1, len(self.files))
        df: Counter = Counter()
        for rec in self.files.values():
            df.update(getattr(rec, field).keys())
        return {t: math.log(1 + (n_docs - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def _bm25_field(self, q: Counter, field: str) -> Dict[str, float]:
        """BM25 over one field. Empty documents are skipped, not scored zero."""
        idf = self._idf(field)
        lengths = {rel: sum(getattr(rec, field).values())
                   for rel, rec in self.files.items()}
        non_empty = [v for v in lengths.values() if v]
        avg_len = (sum(non_empty) / len(non_empty)) if non_empty else 1.0
        k1, b = self.BM25_K1, self.BM25_B

        out: Dict[str, float] = {}
        for rel, rec in self.files.items():
            length = lengths[rel]
            if not length:
                continue
            norm = k1 * (1 - b + b * length / avg_len)
            bag = getattr(rec, field)
            s = 0.0
            for term in q:
                tf = bag.get(term, 0)
                if tf:
                    s += idf.get(term, 0.0) * (tf * (k1 + 1)) / (tf + norm)
            if s:
                out[rel] = s
        return out

    @staticmethod
    def _is_meta(rel: str) -> bool:
        """Files that discuss code rather than being it."""
        low = rel.lower()
        return ("test" in low or low.endswith("evalset.py")
                or "fixture" in low or "conftest" in low)

    def score_files(self, query: str) -> List[Tuple[str, float]]:
        """Multi-field BM25, highest first. The seeding step.

        Three fields, because they are three different kinds of evidence:

            text   what the file says. Weak -- any file can mention anything.
            defs   what the file *defines*. Strong -- `moe.py` defines `Router`.
            path   what the file is called. "moe" in a question means `moe.py`.

        Scored separately and summed (a simplified BM25F), each with its own idf,
        because a word that is common in prose may be rare among definitions and
        that contrast is the whole point.

        The history is worth keeping, since each version failed differently:
        v1 was tf-idf, and a long file repeating "file" and "defines" outranked
        the short one that defined the thing asked about -- fixed by BM25 term
        saturation. v2 was single-field BM25, and it still ranked a symbol's
        *user* above its *definition*. This is v3.
        """
        q = _tokenize(query)
        if not q:
            return []

        text = self._bm25_field(q, "idents")
        defs = self._bm25_field(q, "defs")
        paths = self._bm25_field(q, "path_terms")

        scored = []
        for rel in self.files:
            s = (self.W_TEXT * text.get(rel, 0.0)
                 + self.W_DEFS * defs.get(rel, 0.0)
                 + self.W_PATH * paths.get(rel, 0.0))
            if s > 0:
                if self._is_meta(rel):
                    s *= self.META_PENALTY
                scored.append((rel, s))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored

    def retrieve(self, query: str, budget: int = 8000, hops: int = 1,
                 max_slices: int = 24) -> List[Retrieved]:
        """The whole point: turn a request into the slice of repo it needs.

        1. Any identifier in `query` that names a real symbol is an exact hit.
        2. Remaining budget goes to files ranked by tf-idf against the query.
        3. `hops` expands to files that import a hit file, or that it imports.
        4. Slices are packed by score until `budget` characters are spent.

        Every result carries `reason`, so a bad retrieval is debuggable and
        Oracle can check what was pulled and why.
        """
        out: List[Retrieved] = []
        spent = 0
        claimed: Set[Tuple[str, int, int]] = set()
        text_cache: Dict[str, List[str]] = {}

        def source(rel: str) -> List[str]:
            if rel not in text_cache:
                try:
                    text_cache[rel] = (self.root / rel).read_text(
                        encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    text_cache[rel] = []
            return text_cache[rel]

        def take(rel: str, start: int, end: int, score: float, reason: str,
                 symbol: Optional[str] = None) -> None:
            nonlocal spent
            key = (rel, start, end)
            if key in claimed or len(out) >= max_slices:
                return
            # A method's span sits inside its class's. Returning both spends the
            # budget twice on the same lines.
            for c_rel, c_start, c_end in claimed:
                if c_rel == rel and c_start <= start and end <= c_end:
                    return
            lines = source(rel)
            if not lines:
                return
            body = "\n".join(lines[start - 1:end])
            if not body.strip():
                return
            if spent + len(body) > budget and out:
                return
            claimed.add(key)
            spent += len(body)
            out.append(Retrieved(file=rel, start_line=start, end_line=end,
                                 text=body, score=score, reason=reason, symbol=symbol))

        # 1. exact symbol hits
        seed_files: List[str] = []
        for tok in dict.fromkeys(_IDENT.findall(query)):
            for sym in self.lookup(tok):
                take(sym.file, sym.start_line, sym.end_line, 100.0,
                     f"exact symbol match: {tok}", sym.qualname)
                seed_files.append(sym.file)

        # 2. lexically ranked files
        ranked = self.score_files(query)
        for rel, score in ranked[:max_slices]:
            if spent >= budget:
                break
            rec = self.files[rel]
            seed_files.append(rel)
            if rec.symbols:
                # Rank symbols by what their *body* says, not just their name --
                # `Expert.__init__` and `load_balance_loss` score identically on
                # the name alone. The source is already in the cache, so this
                # costs a tokenize of the top files only, not the whole repo.
                q = _tokenize(query)
                lines = source(rel)

                def relevance(sym: Symbol) -> float:
                    body = _tokenize("\n".join(lines[sym.start_line - 1:sym.end_line]))
                    name_hit = sum(q.get(p, 0) for p in _split_identifier(sym.name))
                    if not body:
                        return 3.0 * name_hit
                    overlap = sum(qc * body.get(t, 0) for t, qc in q.items())
                    return 3.0 * name_hit + overlap / math.sqrt(sum(body.values()))

                best = sorted(rec.symbols, key=lambda s: (-relevance(s), s.start_line))
                for sym in best[:3]:
                    if relevance(sym) <= 0:
                        break
                    take(rel, sym.start_line, sym.end_line, score,
                         f"bm25 {score:.2f} in {rel}", sym.qualname)
            else:
                take(rel, 1, min(rec.n_lines, 60), score, f"bm25 {score:.2f} in {rel}")

        # 3. graph expansion
        frontier = list(dict.fromkeys(seed_files))
        for _ in range(max(0, hops)):
            nxt: List[str] = []
            for rel in frontier:
                if spent >= budget:
                    break
                neighbours = set(self.importers_of(rel))
                rec = self.files.get(rel)
                if rec:
                    for imp in rec.imports:
                        tail = imp.replace("::", ".").split(".")[-1]
                        neighbours.update(
                            r for r in self.files if Path(r).stem == tail and r != rel)
                for nb in sorted(neighbours):
                    nb_rec = self.files.get(nb)
                    if not nb_rec:
                        continue
                    head = nb_rec.symbols[0] if nb_rec.symbols else None
                    if head:
                        take(nb, head.start_line, head.end_line, 1.0,
                             f"one hop from {rel}", head.qualname)
                    nxt.append(nb)
            frontier = nxt

        out.sort(key=lambda r: (-r.score, r.file, r.start_line))
        return out

    def context(self, query: str, budget: int = 8000, hops: int = 1) -> Context:
        """`retrieve`, rendered as a prompt block with provenance on every slice."""
        return render(self.retrieve(query, budget=budget, hops=hops))

    # ------------------------------------------------------------- persistence

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.root / ".argus" / "index.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            # Bump whenever a scored field is added. v2 introduced `defs` and
            # `path_terms`; a v1 file loads cleanly and leaves both empty, which
            # turns off two thirds of the ranking without any visible error.
            "version": 2,
            "root": str(self.root),
            # Every Counter field needs an explicit dict(): asdict() rebuilds a
            # dict subclass by handing its constructor (key, value) tuples, and
            # Counter counts those as *elements* -- producing tuple keys, which
            # JSON refuses. Missing one here is silent until a Counter is
            # non-empty, then it takes down session/new.
            "files": {
                rel: {**asdict(rec),
                      "symbols": [asdict(s) for s in rec.symbols],
                      "idents": dict(rec.idents),
                      "defs": dict(rec.defs),
                      "path_terms": dict(rec.path_terms)}
                for rel, rec in self.files.items()
            },
        }
        target.write_text(json.dumps(blob), encoding="utf-8")
        return target

    def load(self, path: str | Path | None = None) -> bool:
        """Restore a saved index. Returns False if there is nothing to restore.

        Stale entries are not a correctness problem: `scan` re-parses anything
        whose SHA no longer matches and drops anything that disappeared.
        """
        target = Path(path) if path else self.root / ".argus" / "index.json"
        if not target.exists():
            return False
        blob = json.loads(target.read_text(encoding="utf-8"))

        # An index written before a field existed loads without complaint and
        # leaves that Counter empty -- ranking then runs with a scoring signal
        # silently switched off. Refusing the file forces a rebuild instead.
        if blob.get("version") != 2:
            return False

        self.files = {}
        for rel, raw in blob.get("files", {}).items():
            symbols = [Symbol(**s) for s in raw.pop("symbols", [])]
            idents = Counter(raw.pop("idents", {}))
            defs = Counter(raw.pop("defs", {}))
            path_terms = Counter(raw.pop("path_terms", {}))
            self.files[rel] = FileRecord(symbols=symbols, idents=idents,
                                         defs=defs, path_terms=path_terms, **raw)
        return True
