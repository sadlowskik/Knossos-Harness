"""Collection rules for the Python suite.

The harness package is stdlib-only, but the Daedalus research modules (and the
tests that import them, directly or through `daedalus`) need torch at import
time. CI's "Fast suite" job installs only the harness extras, so without this
those modules failed at import and the whole job was red. When torch is absent,
any test module whose source imports torch or the daedalus package is skipped
at collection and the report header says so; with torch installed nothing
changes. Detected from source rather than a hand-kept list, so a new research
test cannot silently break the job.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

TORCH_PRESENT = importlib.util.find_spec("torch") is not None
_RESEARCH_IMPORT = re.compile(
    r"^\s*(?:import\s+(?:torch|daedalus)\b|from\s+(?:torch|daedalus)\b)", re.M
)


def _needs_torch(path: Path) -> bool:
    if path.suffix != ".py" or not path.name.startswith("test_"):
        return False
    try:
        return bool(_RESEARCH_IMPORT.search(path.read_text(encoding="utf-8")))
    except OSError:
        return False


def pytest_ignore_collect(collection_path, config):
    if TORCH_PRESENT:
        return None
    return True if _needs_torch(Path(collection_path)) else None


def pytest_report_header(config):
    if TORCH_PRESENT:
        return None
    skipped = sorted(p.name for p in Path(__file__).parent.glob("test_*.py") if _needs_torch(p))
    if skipped:
        return (
            f"torch not installed: {len(skipped)} Daedalus research modules not collected "
            "(pip install -r requirements.txt to run them)"
        )
    return None
