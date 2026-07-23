#!/usr/bin/env python3
"""Analyze the automatic-depth CTR23 holdout ship-check."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path[:1]:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_t7b_automatic_depth_ctr23_ship_check_v1 as runner
from benchmarks import run_t7b_automatic_depth_fresh_tier_d as helpers


def _geomean(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or np.any(array <= 0) or not np.isfinite(array).all():
        raise RuntimeError("invalid geomean input")
    return float(np.exp(np.mean(np.log(array))))


def analyze(raw: Mapping[str, Any]) -> dict[str, Any]:
    if (
        raw.get("ship_check_id") != runner.SHIP_CHECK_ID
        or raw.get("complete") is not True
        or len(raw.get("rows", ())) != 54
    ):
        raise RuntimeError("CTR23 raw artifact is incomplete")
    indexed = defaultdict(dict)
    for row in raw["rows"]:
        key = (int(row["task_id"]), int(row["fold"]))
        if row["arm"] in indexed[key]:
            raise RuntimeError("duplicate CTR23 arm row")
        indexed[key][row["arm"]] = row
    if len(indexed) != 27:
        raise RuntimeError("CTR23 pair census changed")
    by_task = defaultdict(list)
    depth_counts: Counter[str] = Counter()
    fit_ratios = []
    predict_ratios = []
    for key in sorted(indexed):
        arms = indexed[key]
        if set(arms) != {"control", "candidate"}:
            raise RuntimeError("incomplete CTR23 pair")
        control, candidate = arms["control"], arms["candidate"]
        immutable = (
            "task_id",
            "dataset_id",
            "dataset_name",
            "fold",
            "train_rows",
            "test_rows",
            "input_features",
            "train_index_sha256",
            "test_index_sha256",
        )
        if any(control[name] != candidate[name] for name in immutable):
            raise RuntimeError("CTR23 paired metadata mismatch")
        if not control["integrity_passes"] or not candidate["integrity_passes"]:
            raise RuntimeError("CTR23 integrity failed")
        ratio = float(candidate["rmse"]) / float(control["rmse"])
        by_task[int(control["task_id"])].append(ratio)
        fit_ratios.append(float(candidate["fit_seconds"]) / float(control["fit_seconds"]))
        predict_ratios.append(
            float(candidate["predict_seconds"]) / float(control["predict_seconds"])
        )
        depth_counts[str(int(candidate["fitted_depth"]))] += 1
    task_rows = [
        {
            "task_id": task_id,
            "quality_ratio": _geomean(values),
            "fold_ratios": values,
        }
        for task_id, values in sorted(by_task.items())
    ]
    task_ratios = np.asarray(
        [row["quality_ratio"] for row in task_rows], dtype=np.float64
    )
    rng = np.random.default_rng(20260723)
    draws = rng.integers(0, len(task_ratios), size=(20_000, len(task_ratios)))
    boot = np.exp(np.mean(np.log(task_ratios)[draws], axis=1))
    point = _geomean(task_ratios)
    leave_one_out = [
        _geomean(np.delete(task_ratios, index))
        for index in range(len(task_ratios))
    ]
    return {
        "schema_version": 1,
        "ship_check_id": runner.SHIP_CHECK_ID,
        "kind": "holdout_ship_check",
        "holdout": "CTR23 lockbox; observed release-validation after this run",
        "quality": {
            "task_geomean_ratio": point,
            "bootstrap_upper_ratio": float(np.quantile(boot, 0.95)),
            "worst_task_ratio": float(np.max(task_ratios)),
            "leave_one_out_max_ratio": float(np.max(leave_one_out)),
            "task_wins": int(np.sum(task_ratios < 1.0)),
            "task_ties": int(np.sum(task_ratios == 1.0)),
            "task_losses": int(np.sum(task_ratios > 1.0)),
        },
        "costs": {
            "fit_pair_geomean_ratio": _geomean(fit_ratios),
            "predict_pair_geomean_ratio": _geomean(predict_ratios),
        },
        "candidate_depth_counts": dict(sorted(depth_counts.items())),
        "tasks": task_rows,
        "integrity": {
            "passes": True,
            "rows": 54,
            "pairs": 27,
            "tasks": 9,
        },
        "interpretation": [
            "Holdout evidence; never tune this candidate from these outcomes.",
            "CTR23 is no longer pristine after this ship-check.",
            "The newest untouched sports season is a separate required ship-check.",
        ],
    }


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(helpers.canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    result = analyze(helpers._load_json(args.raw))
    result["source_hashes"] = {
        "raw": helpers.file_sha256(args.raw),
        "analyzer": helpers.file_sha256(Path(__file__)),
    }
    _write_create_only(args.output, result)
    print(json.dumps({"output": str(args.output), "quality": result["quality"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
