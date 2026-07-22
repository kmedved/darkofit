#!/usr/bin/env python3
"""Execute one exact M6 quality-successor v2 development comparison."""

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
    import m6_quality_rule_v2 as rule
    from bench_compare_revisions import EVIDENCE_CSV_FIELDS
    from paired_evidence_contract import load_and_validate_csv, write_create_only
except ImportError:  # pragma: no cover
    from benchmarks import m6_quality_rule_v2 as rule
    from benchmarks.bench_compare_revisions import EVIDENCE_CSV_FIELDS
    from benchmarks.paired_evidence_contract import (
        load_and_validate_csv,
        write_create_only,
    )


RUNNER_PATH = Path(__file__).resolve()
REPO_ROOT = RUNNER_PATH.parents[1]
CONTRACT_PATH = RUNNER_PATH.with_name("m6_quality_successor_v2_contract.md")
RULE_PATH = RUNNER_PATH.with_name("m6_quality_rule_v2.py")
COMPARISON_PATH = RUNNER_PATH.with_name("bench_compare_revisions.py")
EXECUTION_PATH = RUNNER_PATH.with_name("paired_evidence_contract.py")
BACKTEST_RESULT_PATH = RUNNER_PATH.with_name(
    "m6_quality_successor_v2_backtest_result.json"
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repository: Path, *args: str, binary: bool = False):
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=False,
        capture_output=True,
        text=not binary,
    )
    if result.returncode:
        stderr = result.stderr if not binary else result.stderr.decode(errors="replace")
        stdout = result.stdout if not binary else result.stdout.decode(errors="replace")
        raise RuntimeError(stderr.strip() or stdout.strip())
    return result.stdout if binary else result.stdout.strip()


def source_state(repository: Path) -> dict[str, Any]:
    repository = repository.expanduser().resolve()
    root = Path(_git(repository, "rev-parse", "--show-toplevel")).resolve()
    if root != repository:
        raise RuntimeError(
            f"M6 v2 source must name its Git root: {repository} (root is {root})"
        )
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


def _tracked_head_bytes(path: Path) -> bytes:
    relative = path.resolve().relative_to(REPO_ROOT)
    return _git(REPO_ROOT, "show", f"HEAD:{relative}", binary=True)


def validate_backtest_binding() -> dict[str, Any]:
    if not BACKTEST_RESULT_PATH.is_file():
        raise RuntimeError("M6 v2 backtest result is not committed")
    harness = source_state(REPO_ROOT)
    if not harness["clean"]:
        raise RuntimeError("M6 v2 execution requires a clean harness")
    if BACKTEST_RESULT_PATH.read_bytes() != _tracked_head_bytes(BACKTEST_RESULT_PATH):
        raise RuntimeError("M6 v2 backtest result differs from HEAD")
    payload = json.loads(BACKTEST_RESULT_PATH.read_text())
    expected_bindings = {
        "contract": file_sha256(CONTRACT_PATH),
        "rule": file_sha256(RULE_PATH),
        "execution_runner": file_sha256(RUNNER_PATH),
        "comparison_runner": file_sha256(COMPARISON_PATH),
        "paired_execution": file_sha256(EXECUTION_PATH),
    }
    actual = {
        name: payload.get("bindings", {}).get(name, {}).get("sha256")
        for name in expected_bindings
    }
    if (
        payload.get("contract_id") != rule.CONTRACT_ID
        or payload.get("backtest_complete") is not True
        or payload.get("candidate_ranking_eligible") is not True
        or actual != expected_bindings
    ):
        raise RuntimeError("M6 v2 backtest binding is invalid or drifted")
    return {"harness": harness, "result_sha256": file_sha256(BACKTEST_RESULT_PATH)}


def comparison_command(
    *, control: Path, candidate: Path, raw_csv: Path
) -> list[str]:
    return [
        sys.executable,
        str(COMPARISON_PATH),
        "--policy-suite",
        "standing-slice",
        "--control",
        str(control.resolve()),
        "--candidate",
        str(candidate.resolve()),
        "--datasets",
        *rule.DATASETS,
        "--sizes",
        *rule.SIZES,
        "--seeds",
        str(len(rule.SEEDS)),
        "--repeat",
        str(rule.REPEAT),
        "--threads",
        str(rule.THREADS),
        "--weight-modes",
        *rule.WEIGHT_MODES,
        "--models",
        *rule.ARMS,
        "--evidence-contract",
        "paired-evidence-v1",
        "--csv",
        str(raw_csv.resolve()),
    ]


def _require_external_output(path: Path) -> None:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        return
    raise ValueError("M6 v2 output paths must be outside the harness checkout")


def run(args: argparse.Namespace) -> int:
    binding = validate_backtest_binding()
    control = args.control.expanduser().resolve()
    candidate = args.candidate.expanduser().resolve()
    raw_csv = args.raw_csv.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    for path in (raw_csv, output, manifest_path):
        _require_external_output(path)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"M6 v2 output is create-only: {path}")
    if (
        not args.mechanism_id
        or any(
            char not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
            for char in args.mechanism_id
        )
    ):
        raise ValueError("mechanism_id must be a stable lowercase slug")
    if args.inspection_index < 1:
        raise ValueError("inspection_index must be positive")

    before = {
        "control_default": source_state(control),
        "candidate_default": source_state(candidate),
    }
    if any(not state["clean"] for state in before.values()):
        raise RuntimeError("M6 v2 control and candidate must be clean")
    if before["control_default"]["head"] == before["candidate_default"]["head"]:
        raise RuntimeError("M6 v2 control and candidate commits must differ")

    command = comparison_command(control=control, candidate=candidate, raw_csv=raw_csv)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    rows, validation = load_and_validate_csv(
        raw_csv,
        expected_fields=EVIDENCE_CSV_FIELDS,
        expected_sources={"control_default": control, "candidate_default": candidate},
        threads=rule.THREADS,
        expected_pair_keys=rule.expected_pair_keys(),
    )
    after = {
        "control_default": source_state(control),
        "candidate_default": source_state(candidate),
    }
    if after != before:
        raise RuntimeError("M6 v2 source state changed during execution")
    analysis = rule.analyze_rows(rows)
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": rule.CONTRACT_ID,
        "mechanism_id": args.mechanism_id,
        "inspection_index": args.inspection_index,
        "evidence_scope": "spent_general_quality_development_slice",
        "candidate_ranking_eligible": True,
        "shipping_or_default_claim_eligible": False,
        "analysis": analysis,
    }
    manifest = {
        "schema_version": 1,
        "contract_id": rule.CONTRACT_ID,
        "mechanism_id": args.mechanism_id,
        "inspection_index": args.inspection_index,
        "inspection_spent": True,
        "repeat_count": rule.REPEAT,
        "command": command,
        "raw_csv": {"path": str(raw_csv), "sha256": file_sha256(raw_csv)},
        "backtest": binding,
        "bindings": {
            "contract_sha256": file_sha256(CONTRACT_PATH),
            "rule_sha256": file_sha256(RULE_PATH),
            "execution_runner_sha256": file_sha256(RUNNER_PATH),
            "comparison_runner_sha256": file_sha256(COMPARISON_PATH),
            "paired_execution_sha256": file_sha256(EXECUTION_PATH),
        },
        "sources_before_and_after": before,
        "validation": validation,
    }
    result_bytes = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    write_create_only(manifest_path, manifest_bytes)
    try:
        write_create_only(output, result_bytes)
    except BaseException:
        manifest_path.unlink(missing_ok=True)
        raise
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--mechanism-id", required=True)
    parser.add_argument("--inspection-index", type=int, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
