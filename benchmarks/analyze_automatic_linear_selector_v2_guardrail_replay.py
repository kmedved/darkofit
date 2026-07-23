#!/usr/bin/env python3
"""Recompute the selector's historical guardrails without fitting models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
FRESH_RESULT = REPO_ROOT / "benchmarks" / "fresh_selector_confirmation.json"
SPORTS_RESULT = REPO_ROOT / "benchmarks" / "basketball_group_linear_selector.json"
EXPECTED_SHA256 = {
    FRESH_RESULT: "4dc158ec4fd11cf29a5822dc2a09aa76715ce9446773673fa9a2828da1b71a7d",
    SPORTS_RESULT: "cb56ab34769609cd9639245f9ca6ea2012a0ea1a2b532aae458a9e6cfd9f2f25",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_bound_json(path: Path) -> dict[str, Any]:
    actual = _sha256(path)
    expected = EXPECTED_SHA256[path]
    if actual != expected:
        raise RuntimeError(
            f"historical artifact hash changed for {path}: {actual} != {expected}"
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _summarize(lineages: list[dict[str, Any]]) -> dict[str, Any]:
    lineage_ratios = [float(lineage["ratio"]) for lineage in lineages]
    split_ratios = [
        float(ratio)
        for lineage in lineages
        for ratio in lineage["split_ratios"]
    ]
    leave_one_out = (
        {
            lineage["lineage"]: _geomean(
                [
                    float(other["ratio"])
                    for other in lineages
                    if other["lineage"] != lineage["lineage"]
                ]
            )
            for lineage in lineages
        }
        if len(lineages) > 1
        else {}
    )
    return {
        "lineage_count": len(lineages),
        "split_count": len(split_ratios),
        "equal_lineage_geomean_ratio": _geomean(lineage_ratios),
        "worst_lineage": max(lineages, key=lambda item: float(item["ratio"]))[
            "lineage"
        ],
        "worst_lineage_ratio": max(lineage_ratios),
        "worst_split_ratio": max(split_ratios),
        "leave_one_lineage_out": leave_one_out,
        "worst_leave_one_lineage_out": (
            max(leave_one_out, key=leave_one_out.__getitem__)
            if leave_one_out
            else None
        ),
        "worst_leave_one_lineage_out_ratio": (
            max(leave_one_out.values()) if leave_one_out else None
        ),
    }


def _fresh_lineages(data: dict[str, Any]) -> list[dict[str, Any]]:
    by_key = {
        (row["stratum"], row["lineage_cluster"], row["config"]): row
        for row in data["results"]
        if row["config"] in {"darko_default", "smooth_margin_selector"}
    }
    keys = sorted({(stratum, lineage) for stratum, lineage, _ in by_key})
    lineages: list[dict[str, Any]] = []
    for stratum, lineage in keys:
        control = by_key[(stratum, lineage, "darko_default")]
        selector = by_key[(stratum, lineage, "smooth_margin_selector")]
        control_folds = {row["fold"]: row for row in control["folds"]}
        selector_folds = {row["fold"]: row for row in selector["folds"]}
        if control_folds.keys() != selector_folds.keys():
            raise RuntimeError(f"fold identity mismatch for {lineage}")
        lineages.append(
            {
                "lineage": lineage,
                "stratum": stratum,
                "ratio": float(selector["geomean_rmse"])
                / float(control["geomean_rmse"]),
                "split_ratios": [
                    float(selector_folds[fold]["rmse"])
                    / float(control_folds[fold]["rmse"])
                    for fold in sorted(control_folds)
                ],
                "source": FRESH_RESULT.relative_to(REPO_ROOT).as_posix(),
            }
        )
    return lineages


def _sports_lineage(data: dict[str, Any]) -> dict[str, Any]:
    exactness = data["exactness"]
    if not exactness["passes"] or not all(exactness["gates"].values()):
        raise RuntimeError("historical group-safe sports exactness no longer passes")
    control = data["canonical_results"]["control"]
    selector = data["canonical_results"]["group_margin_selector"]
    if control["fold_scores"] != selector["fold_scores"]:
        raise RuntimeError("historical group-safe sports fold scores changed")
    if control["holdout"]["scores"] != selector["holdout"]["scores"]:
        raise RuntimeError("historical group-safe sports holdout scores changed")
    return {
        "lineage": "basketball_group_safe",
        "stratum": "group_safe_sports",
        "ratio": 1.0,
        "split_ratios": [1.0] * (len(control["fold_scores"]) + 1),
        "source": SPORTS_RESULT.relative_to(REPO_ROOT).as_posix(),
        "ratio_interpretation": (
            "exact prediction, model-state, fold-score, and held-team/cold-player "
            "score equality; 1.0 records equality rather than dividing R-squared"
        ),
    }


def build_replay() -> dict[str, Any]:
    fresh = _read_bound_json(FRESH_RESULT)
    sports = _read_bound_json(SPORTS_RESULT)
    lineages = _fresh_lineages(fresh) + [_sports_lineage(sports)]
    strata = sorted({lineage["stratum"] for lineage in lineages})
    return {
        "schema_version": "automatic-linear-selector-v2-guardrail-replay-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_class": "artifact_only_historical_recomputation",
        "source_artifacts": {
            path.relative_to(REPO_ROOT).as_posix(): {"sha256": expected}
            for path, expected in EXPECTED_SHA256.items()
        },
        "limitations": {
            "dependent_on_prior_artifacts": True,
            "prior_outcomes_known": True,
            "fresh_evidence": False,
            "candidate_code_executed": False,
            "can_reverse_protein_terminal_close": False,
            "note": (
                "This replay is a consistency check over spent evidence. It is "
                "not independent confirmation and cannot rescue a terminal Protein failure."
            ),
        },
        "lineages": lineages,
        "analysis": {
            "combined": _summarize(lineages),
            "by_stratum": {
                stratum: _summarize(
                    [item for item in lineages if item["stratum"] == stratum]
                )
                for stratum in strata
            },
        },
    }


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _write_create_only(args.output.resolve(), build_replay())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
