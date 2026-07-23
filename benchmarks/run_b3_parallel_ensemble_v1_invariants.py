#!/usr/bin/env python3
"""Create the source-bound pre-timing invariant record for private B3 v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "benchmarks/b3_parallel_ensemble_v1_contract.md"
CONTROL_HEAD = "c4dae58fcf7a8d456533ba2d9b469f039adc453c"
CANDIDATE_BASE_HEAD = "4073bb971cd09f7aeb776f7cce52c497c7ba3bd1"
CANDIDATE_HEAD = "5116470e21675f8a869ee7a84145eb2a663ed809"
CANDIDATE_FILES = {
    "darkofit/sklearn_api.py",
    "tests/test_b3_parallel_ensemble_candidate.py",
}
TESTS = (
    "tests/test_b3_parallel_ensemble_candidate.py",
    "tests/test_b3_parallel_ensemble_contract.py",
    "tests/test_ensemble_api.py",
    "tests/test_private_ensemble_v3.py",
    "tests/test_public_ensemble_v3.py",
    "tests/test_ensemble_v3_release_candidate.py",
    "tests/test_ensemble_archive_components.py",
    "tests/test_ensemble_v3_public_contract.py",
    "tests/test_thread_state_restoration.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=repo, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def source_state(repo: Path, expected_head: str | None = None) -> dict:
    repo = repo.expanduser().resolve()
    if Path(git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        raise RuntimeError(f"source must be a Git root: {repo}")
    head = git(repo, "rev-parse", "HEAD")
    status = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(f"source is dirty: {repo}")
    if expected_head is not None and head != expected_head:
        raise RuntimeError(f"source head drifted: {repo}: {head}")
    remotes = git(repo, "branch", "-r", "--contains", head).splitlines()
    if not remotes:
        raise RuntimeError(f"source head is not published: {repo}: {head}")
    return {
        "path": str(repo),
        "head": head,
        "tree": git(repo, "rev-parse", "HEAD^{tree}"),
        "published_refs": [value.strip() for value in remotes],
    }


def write_create_only(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        )
        handle.flush()
        os.fsync(handle.fileno())


def run(control: Path, candidate: Path, output: Path) -> Path:
    harness_state = source_state(ROOT)
    control_state = source_state(control, CONTROL_HEAD)
    candidate_state = source_state(candidate, CANDIDATE_HEAD)
    changed = set(
        git(
            candidate,
            "diff",
            "--name-only",
            f"{CANDIDATE_BASE_HEAD}..{CANDIDATE_HEAD}",
        )
        .splitlines()
    )
    if changed != CANDIDATE_FILES:
        raise RuntimeError(f"candidate file allowlist drifted: {sorted(changed)}")
    command = [sys.executable, "-m", "pytest", "-q", *TESTS]
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": str(candidate),
        "PYTHONDONTWRITEBYTECODE": "1",
        "DARKOFIT_WARMUP": "0",
    })
    completed = subprocess.run(
        command,
        cwd=candidate,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"B3 invariant suite failed\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    payload = {
        "schema_version": 1,
        "contract_id": "b3-parallel-ensemble-members-v1-20260723",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "harness": harness_state,
            "control": control_state,
            "candidate": candidate_state,
        },
        "candidate_file_allowlist": sorted(CANDIDATE_FILES),
        "contract": {
            "path": str(CONTRACT_PATH),
            "sha256": sha256(CONTRACT_PATH),
        },
        "implementation": {
            "path": "darkofit/sklearn_api.py",
            "sha256": sha256(candidate / "darkofit/sklearn_api.py"),
        },
        "candidate_tests": {
            "path": "tests/test_b3_parallel_ensemble_candidate.py",
            "sha256": sha256(
                candidate / "tests/test_b3_parallel_ensemble_candidate.py"
            ),
        },
        "command": command,
        "test_paths": list(TESTS),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": True,
    }
    write_create_only(output, payload)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = run(args.control, args.candidate, args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
