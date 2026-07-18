#!/usr/bin/env python3
"""Derive a cost-aware engagement margin from the spent cross-feature screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks" / "smooth_cross_features.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "smooth_cross_margin_analysis.json"
MARGIN_GRID = tuple(index / 100 for index in range(0, 11))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _geomean(values):
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0 or np.any(values <= 0.0):
        raise RuntimeError("geomean requires positive values")
    return float(np.exp(np.mean(np.log(values))))


def _margin(row):
    base = float(row["base"]["best_validation_rmse"])
    selected = float(row["selected"]["best_validation_rmse"])
    return float((base - selected) / base)


def evaluate_margin(rows, threshold):
    per_dataset = {}
    split_records = []
    for row in rows:
        engage = bool(row["cross_selected"]) and _margin(row) >= threshold
        ratio = (
            float(row["selected"]["test_rmse"])
            / float(row["base"]["test_rmse"])
            if engage
            else 1.0
        )
        per_dataset.setdefault(row["dataset_name"], []).append(ratio)
        split_records.append(
            {
                "task_id": int(row["task_id"]),
                "dataset_name": row["dataset_name"],
                "fold": int(row["fold"]),
                "validation_improvement": _margin(row),
                "engaged": engage,
                "test_ratio": ratio,
            }
        )
    dataset_ratios = {
        name: _geomean(ratios) for name, ratios in per_dataset.items()
    }
    leave_one_out = {
        omitted: _geomean(
            [
                ratio
                for name, ratio in dataset_ratios.items()
                if name != omitted
            ]
        )
        for omitted in dataset_ratios
    }
    return {
        "minimum_validation_improvement": float(threshold),
        "engaged_coordinates": int(
            sum(record["engaged"] for record in split_records)
        ),
        "equal_dataset_geomean_ratio": _geomean(dataset_ratios.values()),
        "worst_dataset_ratio": float(max(dataset_ratios.values())),
        "worst_split_ratio": float(
            max(record["test_ratio"] for record in split_records)
        ),
        "leave_one_out_equal_dataset_ratios": leave_one_out,
        "dataset_ratios": dataset_ratios,
        "split_records": split_records,
    }


def analyze(source):
    rows = source["results"]
    grid = [evaluate_margin(rows, threshold) for threshold in MARGIN_GRID]
    zero_harm = [
        record for record in grid if record["worst_split_ratio"] <= 1.0
    ]
    if not zero_harm:
        nominee = None
    else:
        nominee = min(
            zero_harm,
            key=lambda record: record["minimum_validation_improvement"],
        )
    return {
        "claim_tier": "development_policy_nomination",
        "fresh_claim_eligible": False,
        "margin_grid": list(MARGIN_GRID),
        "selection_rule": (
            "smallest whole-percentage validation margin on the declared "
            "grid with no observed split regression"
        ),
        "grid_results": grid,
        "nominee": nominee,
        "nominee_requires_fresh_confirmation": nominee is not None,
        "caveats": [
            "margin chosen on spent development outcomes",
            "three datasets are insufficient for a shipping claim",
            "full crossed audition cost is paid even when the guard declines",
            "zero observed harm is not a population guarantee",
        ],
    }


def run(output):
    source = json.loads(SOURCE.read_text())
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(SOURCE.relative_to(ROOT)),
            "sha256": _sha256(SOURCE),
            "source_head": source["sources"]["darkofit"]["head"],
            "source_protocol_sha256": source["protocol"]["sha256"],
        },
        "analysis": analyze(source),
    }
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = run(args.output.resolve())
    print(json.dumps(artifact["analysis"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
