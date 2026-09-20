"""Collection rules for the Python suite.

The harness package is stdlib-only, but the Daedalus research tests import
torch at module level. CI's "Fast suite" job installs only the harness extras,
so without this those modules failed at import and the whole job was red.
When torch is absent the research modules are left out of collection and the
reason is printed once; with torch installed nothing changes.
"""
from __future__ import annotations

import importlib.util

RESEARCH_MODULES = [
    "test_arch_flags.py",
    "test_benchmarks.py",
    "test_components.py",
    "test_echo.py",
    "test_gec.py",
    "test_loop_embed.py",
    "test_moirai.py",
    "test_naiads.py",
    "test_proteus.py",
    "test_training.py",
]

collect_ignore: list[str] = []
if importlib.util.find_spec("torch") is None:
    collect_ignore = list(RESEARCH_MODULES)


def pytest_report_header(config):
    if collect_ignore:
        return (
            f"torch not installed: {len(collect_ignore)} Daedalus research modules "
            "not collected (pip install -r requirements.txt to run them)"
        )
    return None
