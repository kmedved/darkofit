#!/usr/bin/env python3
"""Frozen quality-only M6 successor analyzer and contract state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from bench_compare_revisions import EVIDENCE_CSV_FIELDS
    from paired_evidence_contract import (
        CONTRACT_THREADS,
        CONTRACT_VERSION as EXECUTION_CONTRACT,
        load_and_validate_csv,
        write_create_only,
    )
except ImportError:  # pragma: no cover - supports module execution
    from benchmarks.bench_compare_revisions import EVIDENCE_CSV_FIELDS
    from benchmarks.paired_evidence_contract import (
        CONTRACT_THREADS,
        CONTRACT_VERSION as EXECUTION_CONTRACT,
        load_and_validate_csv,
        write_create_only,
    )


CONTRACT_ID = "m6-quality-successor-v1"
CONTRACT_PATH = Path(__file__).with_name("m6_quality_successor_contract.md")
ANALYZER_PATH = Path(__file__).resolve()
REPO_ROOT = ANALYZER_PATH.parents[1]

DATASETS = (
    "diabetes_resampled",
    "friedman_numeric",
    "wide_numeric_reg",
    "categorical_reg",
    "breast_cancer_resampled",
    "numeric_binary",
    "wine_resampled",
    "numeric_multiclass",
    "categorical_binary",
    "categorical_multiclass",
)
SIZES = ("medium",)
SEEDS = (0, 1, 2)
WEIGHT_MODES = ("none", "stress")
REPEAT = 3
THREADS = 4
ARMS = ("control_default", "candidate_default")

MAX_GEOMEAN_RATIO = 0.98
MIN_WIN_FRACTION = 0.60
MAX_CELL_RATIO = 1.02

BACKTEST_COMPLETE = False
BACKTEST_RESULT_PATH = ""
BACKTEST_RESULT_SHA256 = ""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _geomean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all() or np.any(array < 0.0):
        raise RuntimeError("M6 quality ratios must be nonnegative and finite")
    if np.any(array == 0.0):
        return 0.0
    return float(np.exp(np.mean(np.log(array))))


def quality_decision(ratios: Mapping[str, float]) -> dict[str, Any]:
    """Apply the frozen development-quality rule to named paired ratios."""
    if not isinstance(ratios, Mapping) or not ratios:
        raise RuntimeError("M6 quality decision requires named ratios")
    names = list(ratios)
    if any(not isinstance(name, str) or not name for name in names):
        raise RuntimeError("M6 quality ratio names must be non-empty strings")
    if len(names) != len(set(names)):
        raise RuntimeError("M6 quality ratio names must be unique")
    numeric = {}
    for name, value in ratios.items():
        if isinstance(value, bool):
            raise RuntimeError("M6 quality ratios must be numeric")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("M6 quality ratios must be numeric") from exc
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError("M6 quality ratios must be nonnegative and finite")
        numeric[name] = value
    count = len(numeric)
    wins = sum(value < 1.0 for value in numeric.values())
    minimum_wins = int(math.ceil(MIN_WIN_FRACTION * count))
    geomean = _geomean(list(numeric.values()))
    worst_name, worst_ratio = max(numeric.items(), key=lambda item: item[1])
    gates = {
        "geomean_at_most_0_98": geomean <= MAX_GEOMEAN_RATIO,
        "wins_at_least_60_percent": wins >= minimum_wins,
        "no_cell_above_1_02": worst_ratio <= MAX_CELL_RATIO,
    }
    return {
        "disposition": "advance" if all(gates.values()) else "kill",
        "gates": gates,
        "case_count": count,
        "geometric_mean_ratio": geomean,
        "wins": wins,
        "minimum_wins": minimum_wins,
        "worst_case": worst_name,
        "worst_ratio": worst_ratio,
        "ratios": numeric,
    }


def expected_pair_keys() -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (dataset, size, str(seed), weight)
        for size in SIZES
        for dataset in DATASETS
        for seed in SEEDS
        for weight in WEIGHT_MODES
    )


def _pair_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["dataset"]),
        str(row["size"]),
        str(row["seed"]),
        str(row["weight_mode"]),
    )


def analyze_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = set(expected_pair_keys())
    pairs: dict[tuple[str, str, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = _pair_key(row)
        arm = str(row["variant"])
        if key not in expected or arm not in ARMS or arm in pairs.setdefault(key, {}):
            raise RuntimeError("M6 quality rows contain an unexpected identity")
        pairs[key][arm] = row
    if set(pairs) != expected or any(set(pair) != set(ARMS) for pair in pairs.values()):
        raise RuntimeError("M6 quality rows do not cover the exact paired grid")

    ratios = {}
    dataset_ratios = {dataset: [] for dataset in DATASETS}
    for key in expected_pair_keys():
        control = pairs[key]["control_default"]
        candidate = pairs[key]["candidate_default"]
        if control["primary_metric"] != candidate["primary_metric"]:
            raise RuntimeError("M6 paired primary metrics differ")
        try:
            denominator = float(control["primary_value"])
            numerator = float(candidate["primary_value"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("M6 paired primary values are invalid") from exc
        if (
            not math.isfinite(denominator)
            or not math.isfinite(numerator)
            or denominator <= 0.0
            or numerator < 0.0
        ):
            raise RuntimeError("M6 paired primary values are invalid")
        ratio = numerator / denominator
        name = "/".join(key)
        ratios[name] = ratio
        dataset_ratios[key[0]].append(ratio)

    decision = quality_decision(ratios)
    per_dataset = {
        dataset: _geomean(values)
        for dataset, values in dataset_ratios.items()
    }
    leave_one_dataset_out = {}
    for omitted in DATASETS:
        retained = [
            ratio
            for dataset, values in dataset_ratios.items()
            if dataset != omitted
            for ratio in values
        ]
        leave_one_dataset_out[omitted] = _geomean(retained)
    return {
        **decision,
        "per_dataset_geometric_mean_ratio": per_dataset,
        "leave_one_dataset_out_geometric_mean_ratio": leave_one_dataset_out,
    }


def source_state(repository: Path) -> dict[str, Any]:
    repository = repository.expanduser().resolve()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result.stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    return {
        "path": str(repository),
        "head": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "clean": not status,
        "status": status,
    }


def validate_backtest_binding() -> None:
    if not BACKTEST_COMPLETE:
        raise RuntimeError("M6 quality successor backtest is not complete")
    if (
        not BACKTEST_RESULT_PATH
        or len(BACKTEST_RESULT_SHA256) != 64
        or set(BACKTEST_RESULT_SHA256) - set("0123456789abcdef")
    ):
        raise RuntimeError("M6 quality successor backtest binding is invalid")
    result = REPO_ROOT / BACKTEST_RESULT_PATH
    if not result.is_file() or file_sha256(result) != BACKTEST_RESULT_SHA256:
        raise RuntimeError("M6 quality successor backtest result drifted")
    payload = json.loads(result.read_text())
    if (
        payload.get("contract_id") != CONTRACT_ID
        or payload.get("backtest_complete") is not True
        or payload.get("candidate_ranking_eligible") is not True
    ):
        raise RuntimeError("M6 quality successor backtest result is invalid")


def analyze_csv(
    csv_path: Path,
    *,
    control: Path,
    candidate: Path,
    mechanism_id: str,
    inspection_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_backtest_binding()
    if (
        not isinstance(mechanism_id, str)
        or not mechanism_id.strip()
        or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in mechanism_id)
    ):
        raise ValueError("mechanism_id must be a stable lowercase slug")
    if isinstance(inspection_index, bool) or not isinstance(inspection_index, int) or inspection_index < 1:
        raise ValueError("inspection_index must be a positive integer")
    sources = {
        "control_default": control.expanduser().resolve(),
        "candidate_default": candidate.expanduser().resolve(),
    }
    rows, validation = load_and_validate_csv(
        csv_path,
        expected_fields=EVIDENCE_CSV_FIELDS,
        expected_sources=sources,
        threads=THREADS,
        expected_pair_keys=expected_pair_keys(),
    )
    analysis = analyze_rows(rows)
    states = {name: source_state(path) for name, path in sources.items()}
    if any(not state["clean"] for state in states.values()):
        raise RuntimeError("M6 quality source trees must be clean")
    if states["control_default"]["head"] == states["candidate_default"]["head"]:
        raise RuntimeError("M6 quality control and candidate commits must differ")
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": CONTRACT_ID,
        "execution_contract": EXECUTION_CONTRACT,
        "mechanism_id": mechanism_id,
        "inspection_index": inspection_index,
        "evidence_scope": "spent_general_development_slice",
        "candidate_ranking_eligible": True,
        "shipping_or_default_claim_eligible": False,
        "analysis": analysis,
    }
    manifest = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "mechanism_id": mechanism_id,
        "inspection_index": inspection_index,
        "csv": {"path": str(csv_path), "sha256": file_sha256(csv_path)},
        "contract_sha256": file_sha256(CONTRACT_PATH),
        "analyzer_sha256": file_sha256(ANALYZER_PATH),
        "sources": states,
        "validation": validation,
        "inspection_spent": True,
    }
    return result, manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--mechanism-id", required=True)
    parser.add_argument("--inspection-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if args.output.exists() or args.output.is_symlink() or manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError("M6 quality result and manifest are create-only")
    result, manifest = analyze_csv(
        args.csv,
        control=args.control,
        candidate=args.candidate,
        mechanism_id=args.mechanism_id,
        inspection_index=args.inspection_index,
    )
    result_bytes = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    write_create_only(manifest_path, manifest_bytes)
    try:
        write_create_only(args.output, result_bytes)
    except BaseException:
        manifest_path.unlink(missing_ok=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
