"""Audit and freeze every evaluation suite used by Knossos.

This is deliberately a CI command, not a report generator.  A fixture that is
green before the agent starts, a hidden test that is visible in the starting
tree, or an unnoticed suite edit makes the resulting score and training trace
untrustworthy, so any of those conditions exits non-zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knossos.codeval import load_cases, materialise, reveal_held_out, run_tests


SCHEMA = "knossos-suite-lock/v1"
DEFAULT_SUITES = (
    Path("../knossos-rs/cases/core.json"),
    Path("fixtures/hard_suite.json"),
    Path("fixtures/agentic_behavior_suite.json"),
)


def _canonical_digest(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _fingerprint(case) -> str:
    # The ID is intentionally excluded: renaming a duplicated task must not
    # disguise train/test leakage.
    value = {"prompt": case.prompt, "files": case.files}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_lock(paths: Iterable[Path], base: Path) -> dict:
    suites = []
    for path in paths:
        resolved = path.resolve()
        cases = load_cases(resolved)
        try:
            display = resolved.relative_to(base.resolve()).as_posix()
        except ValueError:
            display = resolved.as_posix()
        suites.append(
            {
                "path": display,
                "sha256": _canonical_digest(resolved),
                "cases": len(cases),
                "case_ids": [case.id for case in cases],
            }
        )
    return {"schema": SCHEMA, "suites": suites}


def audit(paths: Iterable[Path]) -> list[str]:
    errors: list[str] = []
    ids: dict[str, str] = {}
    fingerprints: dict[str, str] = {}

    for path in paths:
        cases = load_cases(path)
        for case in cases:
            where = f"{path}:{case.id}"
            if case.id in ids:
                errors.append(f"duplicate case id {case.id!r}: {ids[case.id]} and {path}")
            ids[case.id] = str(path)

            fingerprint = _fingerprint(case)
            if fingerprint in fingerprints:
                errors.append(
                    f"cross-suite task leak: {where} duplicates {fingerprints[fingerprint]}"
                )
            fingerprints[fingerprint] = where

            visible_blob = "\n".join(case.files.values())
            for hidden_path, hidden_text in case.held_out.items():
                if hidden_path in case.files:
                    errors.append(f"hidden path is visible at start: {where}:{hidden_path}")
                if hidden_text and hidden_text in visible_blob:
                    errors.append(f"hidden verifier content leaked into visible files: {where}")

            # Keep fixtures under the checkout.  Besides making CI artefacts
            # discoverable, this works in restricted runners where the system
            # temp directory is intentionally not writable by subprocesses.
            with tempfile.TemporaryDirectory(
                prefix="knossos-suite-audit-", dir=Path.cwd()
            ) as tmp:
                root = Path(tmp)
                materialise(case, root)
                leaked = [name for name in case.held_out if (root / name).exists()]
                if leaked:
                    errors.append(f"held-out files materialised early for {where}: {leaked}")

                pass_state = run_tests(root, case.pass_to_pass)
                failed_preservation = [node for node, ok in pass_state.items() if not ok]
                if failed_preservation:
                    errors.append(
                        f"pass_to_pass is red before the run for {where}: {failed_preservation}"
                    )

                red_state = run_tests(root, case.fail_to_pass)
                unexpectedly_green = [node for node, ok in red_state.items() if ok]
                if unexpectedly_green:
                    errors.append(
                        f"fail_to_pass is green before the run for {where}: {unexpectedly_green}"
                    )
                if case.expected_action == "edit" and not red_state:
                    errors.append(f"edit task has no red starting check: {where}")

                reveal_held_out(case, root)
                hidden_first = run_tests(root, case.held_out_pass)
                hidden_second = run_tests(root, case.held_out_pass)
                if hidden_first != hidden_second:
                    errors.append(f"non-deterministic hidden verifier for {where}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", action="append", type=Path, dest="suites")
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--write-lock", action="store_true")
    args = parser.parse_args()

    paths = tuple(args.suites or DEFAULT_SUITES)
    errors = audit(paths)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    expected = build_lock(paths, Path.cwd().parent)
    if args.write_lock:
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        args.lock.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.lock}")
        return 0

    if not args.lock.exists():
        print(f"ERROR: missing suite lock {args.lock}; run with --write-lock")
        return 1
    actual = json.loads(args.lock.read_text(encoding="utf-8"))
    if actual != expected:
        print("ERROR: evaluation suites do not match the immutable lock")
        print(json.dumps(expected, indent=2))
        return 1
    print(f"audited {sum(s['cases'] for s in expected['suites'])} locked cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
