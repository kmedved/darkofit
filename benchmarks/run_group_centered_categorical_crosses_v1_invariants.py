#!/usr/bin/env python3
"""Run and record the pre-quality group-centered-cross invariant suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from paired_evidence_contract import write_create_only
except ImportError:  # pragma: no cover
    from benchmarks.paired_evidence_contract import write_create_only


IDENTITY = "group-centered-categorical-crosses-v1-invariants-20260722"
CONTRACT_ID = "group-centered-categorical-crosses-v1-development-20260722"
CANDIDATE_HEAD = "c3f2608cd3033cfc00aa0737897a92ed868b5865"
RUNNER_PATH = Path(__file__).resolve()
ROOT = RUNNER_PATH.parents[1]
CONTRACT_PATH = RUNNER_PATH.with_name(
    "group_centered_categorical_crosses_v1_development_contract.md"
)
TESTS = (
    "tests/test_group_centered_preprocessing.py",
    "tests/test_group_centered_selector.py",
    "tests/test_thread_state_restoration.py",
    "tests/test_payload_hardening.py",
    "tests/test_ensemble_api.py",
    "tests/test_public_ensemble_v3.py",
    "tests/test_private_ensemble_v3.py",
    "tests/test_deprecations.py",
    "tests/test_basketball_native_ordinal.py::test_normalized_archive_identity_removes_only_inactive_declaration",
    "tests/test_native_ordinal_c2.py::test_empty_explicit_ordinal_archive_normalizes_to_control",
)
CANDIDATE_FILES = (
    "darkofit/booster.py",
    "darkofit/preprocessing.py",
    "darkofit/serialization.py",
    "darkofit/sklearn_api.py",
    "tests/test_darkofit.py",
    "tests/test_group_centered_preprocessing.py",
    "tests/test_group_centered_selector.py",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or process.stdout.strip())
    return process.stdout.strip()


def source_state(repository: Path) -> dict[str, Any]:
    repository = repository.expanduser().resolve()
    if Path(_git(repository, "rev-parse", "--show-toplevel")).resolve() != repository:
        raise RuntimeError("invariant source must name its Git root")
    status = _git(
        repository, "status", "--porcelain=v1", "--untracked-files=all"
    ).splitlines()
    return {
        "path": str(repository),
        "head": _git(repository, "rev-parse", "HEAD"),
        "tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "clean": not status,
        "status": status,
    }


def run(args: argparse.Namespace) -> Path:
    candidate = args.candidate.expanduser().resolve()
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("invariant output must be outside the harness")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    before = {
        "harness": source_state(ROOT),
        "candidate": source_state(candidate),
    }
    if (
        not all(state["clean"] for state in before.values())
        or before["candidate"]["head"] != CANDIDATE_HEAD
    ):
        raise RuntimeError("invariants require clean frozen sources")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--disable-warnings",
        *TESTS,
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(candidate)
    process = subprocess.run(
        command,
        cwd=candidate,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise RuntimeError(
            "group-centered invariant suite failed:\n"
            + (process.stdout + "\n" + process.stderr).strip()
        )
    after = {
        "harness": source_state(ROOT),
        "candidate": source_state(candidate),
    }
    if after != before:
        raise RuntimeError("invariant source changed during execution")
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": IDENTITY,
        "contract_id": CONTRACT_ID,
        "quality_outcomes_inspected": False,
        "shipping_or_default_claim_authorized": False,
        "sources": before,
        "bindings": {
            "contract_sha256": file_sha256(CONTRACT_PATH),
            "runner_sha256": file_sha256(RUNNER_PATH),
            "candidate_files": {
                relative: file_sha256(candidate / relative)
                for relative in CANDIDATE_FILES
            },
        },
        "execution": {
            "command": command,
            "tests": list(TESTS),
            "stdout": process.stdout.strip(),
            "stderr": process.stderr.strip(),
        },
        "analysis": {
            "focused_invariants_passed": True,
            "mechanism_synthetic_engaged": True,
            "safe_npz_roundtrip_and_corruption_checks_passed": True,
            "nested_ensemble_and_thread_checks_passed": True,
        },
    }
    write_create_only(
        output,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(f"wrote group-centered invariants to {output}")
    print(f"artifact sha256: {file_sha256(output)}")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
