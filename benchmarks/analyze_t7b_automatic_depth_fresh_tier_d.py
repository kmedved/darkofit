#!/usr/bin/env python3
"""Analyze the one-shot automatic-depth fresh Tier-D artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from benchmarks.tier_d_fresh_power_design import evaluate_panel_logs


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "benchmarks" / ("t7b_automatic_depth_fresh_tier_d_execution_contract.json")
)
POWER_CONTRACT = (
    ROOT
    / "benchmarks"
    / ("t7b_automatic_depth_fresh_tier_d_power_design_contract.json")
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _geomean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 1
        or array.size == 0
        or not np.isfinite(array).all()
        or np.any(array <= 0.0)
    ):
        raise RuntimeError("geomean input is invalid")
    return float(np.exp(np.mean(np.log(array))))


def _pairs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["lineage_id"]), int(row["coordinate"]))
        arm = str(row["arm"])
        if arm in indexed[key]:
            raise RuntimeError(f"duplicate arm row: {key}/{arm}")
        indexed[key][arm] = row
    if len(indexed) != 96:
        raise RuntimeError("raw artifact must contain 96 paired coordinates")

    pairs = []
    for key in sorted(indexed):
        arms = indexed[key]
        if set(arms) != {"control", "candidate"}:
            raise RuntimeError(f"incomplete arm pair: {key}")
        control = arms["control"]
        candidate = arms["candidate"]
        immutable = (
            "lineage_id",
            "slot",
            "stratum",
            "branch",
            "coordinate",
            "weight_mode",
            "task_id",
            "dataset_id",
            "split_sha256",
            "train_rows",
            "test_rows",
            "input_features",
        )
        if any(control[name] != candidate[name] for name in immutable):
            raise RuntimeError(f"paired metadata mismatch: {key}")
        if control["status"] != "ok" or candidate["status"] != "ok":
            raise RuntimeError(f"non-ok row in complete raw artifact: {key}")
        prediction_control = np.asarray(
            control["predict_seconds_repeats"], dtype=np.float64
        )
        prediction_candidate = np.asarray(
            candidate["predict_seconds_repeats"], dtype=np.float64
        )
        if (
            prediction_control.shape != (3,)
            or prediction_candidate.shape != (3,)
            or not np.isfinite(prediction_control).all()
            or not np.isfinite(prediction_candidate).all()
            or np.any(prediction_control <= 0.0)
            or np.any(prediction_candidate <= 0.0)
        ):
            raise RuntimeError(f"invalid prediction timings: {key}")
        pairs.append(
            {
                **{name: control[name] for name in immutable},
                "quality_ratio": float(candidate["rmse"]) / float(control["rmse"]),
                "fit_ratio": float(candidate["fit_seconds"])
                / float(control["fit_seconds"]),
                "predict_ratio": float(
                    np.median(prediction_candidate) / np.median(prediction_control)
                ),
                "rss_ratio": float(candidate["peak_process_tree_rss_bytes"])
                / float(control["peak_process_tree_rss_bytes"]),
                "rss_delta_bytes": int(candidate["peak_process_tree_rss_bytes"])
                - int(control["peak_process_tree_rss_bytes"]),
                "candidate_peak_rss_bytes": int(
                    candidate["peak_process_tree_rss_bytes"]
                ),
                "control_peak_rss_bytes": int(control["peak_process_tree_rss_bytes"]),
                "integrity_passes": bool(
                    control["integrity_passes"]
                    and candidate["integrity_passes"]
                    and control["ambient_thread_restored"]
                    and candidate["ambient_thread_restored"]
                    and control["safe_npz_exact"]
                    and candidate["safe_npz_exact"]
                ),
            }
        )
    return pairs


def analyze(
    raw: Mapping[str, Any],
    contract: Mapping[str, Any],
    power_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if raw.get("contract_id") != contract.get("contract_id"):
        raise RuntimeError("raw/contract identity mismatch")
    rows = raw.get("rows")
    if not isinstance(rows, list) or len(rows) != 192:
        raise RuntimeError("raw artifact must contain exactly 192 arm rows")
    pairs = _pairs(rows)

    by_lineage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_lineage[str(pair["lineage_id"])].append(pair)
    if len(by_lineage) != 32 or any(len(values) != 3 for values in by_lineage.values()):
        raise RuntimeError("raw lineage/coordinate census changed")

    lineage_rows = []
    for lineage_id in sorted(by_lineage):
        values = sorted(by_lineage[lineage_id], key=lambda row: row["coordinate"])
        lineage_rows.append(
            {
                "lineage_id": lineage_id,
                "slot": values[0]["slot"],
                "stratum": values[0]["stratum"],
                "branch": values[0]["branch"],
                "quality_ratio": _geomean([row["quality_ratio"] for row in values]),
                "fit_ratio": _geomean([row["fit_ratio"] for row in values]),
                "predict_ratio": _geomean([row["predict_ratio"] for row in values]),
                "rss_ratio": _geomean([row["rss_ratio"] for row in values]),
                "rss_delta_bytes": int(
                    round(sum(row["rss_delta_bytes"] for row in values) / len(values))
                ),
                "candidate_peak_rss_bytes": max(
                    row["candidate_peak_rss_bytes"] for row in values
                ),
                "integrity_passes": all(row["integrity_passes"] for row in values),
                "coordinate_ratios": {
                    str(row["coordinate"]): {
                        "quality": row["quality_ratio"],
                        "fit": row["fit_ratio"],
                        "predict": row["predict_ratio"],
                        "rss": row["rss_ratio"],
                    }
                    for row in values
                },
            }
        )

    quality_logs = np.log(
        np.asarray(
            [[row["quality_ratio"] for row in lineage_rows]],
            dtype=np.float64,
        )
    )
    simulation = power_contract["simulation"]
    rng = np.random.default_rng(simulation["lineage_bootstrap_seed"])
    counts = rng.multinomial(
        len(lineage_rows),
        np.full(len(lineage_rows), 1.0 / len(lineage_rows)),
        size=simulation["lineage_bootstrap_replicates"],
    ).astype(np.float64)
    evaluated = evaluate_panel_logs(
        quality_logs,
        [row["branch"] for row in lineage_rows],
        counts,
        power_contract["quality_gates"],
        bootstrap_percentile=simulation["lineage_bootstrap_percentile"],
    )
    quality = {
        "equal_lineage_geomean_ratio": float(evaluated["point"][0]),
        "bootstrap_upper_ratio": float(evaluated["bootstrap_upper"][0]),
        "leave_one_favorable_lineage_out_ratio": float(
            evaluated["leave_one_favorable_out"][0]
        ),
        "worst_lineage_ratio": float(evaluated["worst_lineage"][0]),
        "branch_geomean_ratio": {
            "depth_4": float(evaluated["branch_depth_4"][0]),
            "depth_8": float(evaluated["branch_depth_8"][0]),
        },
        "component_passes": {
            "equal_lineage_geomean": bool(evaluated["component_point"][0]),
            "bootstrap_upper": bool(evaluated["component_bootstrap_upper"][0]),
            "leave_one_favorable_out": bool(
                evaluated["component_leave_one_favorable_out"][0]
            ),
            "worst_lineage": bool(evaluated["component_worst_lineage"][0]),
            "each_branch_direction": bool(
                evaluated["component_each_branch_direction"][0]
            ),
        },
    }
    quality["passes"] = all(quality["component_passes"].values())

    cost_gates = contract["cost_gates"]
    fit_ratio = _geomean([row["fit_ratio"] for row in lineage_rows])
    predict_ratio = _geomean([row["predict_ratio"] for row in lineage_rows])
    rss_ratio = _geomean([row["rss_ratio"] for row in lineage_rows])
    rss_delta = int(
        round(sum(row["rss_delta_bytes"] for row in lineage_rows) / len(lineage_rows))
    )
    maximum_candidate_rss = max(row["candidate_peak_rss_bytes"] for row in lineage_rows)
    rss_hard_pass = maximum_candidate_rss <= int(
        raw["environment"]["physical_memory_bytes"]
    ) * float(cost_gates["rss_hard_fraction_of_physical_memory"])
    rss_hybrid_pass = not (
        rss_ratio > float(cost_gates["rss_ratio_allowance"])
        and rss_delta > int(cost_gates["rss_absolute_delta_allowance_bytes"])
    )
    costs = {
        "fit_equal_lineage_geomean_ratio": fit_ratio,
        "predict_equal_lineage_geomean_ratio": predict_ratio,
        "rss_equal_lineage_geomean_ratio": rss_ratio,
        "rss_equal_lineage_mean_delta_bytes": rss_delta,
        "maximum_candidate_peak_process_tree_rss_bytes": (maximum_candidate_rss),
        "component_passes": {
            "fit": fit_ratio <= float(cost_gates["fit_ratio_at_most"]),
            "predict": predict_ratio <= float(cost_gates["predict_ratio_at_most"]),
            "rss_hard_ceiling": rss_hard_pass,
            "rss_hybrid_allowance": rss_hybrid_pass,
        },
    }
    costs["passes"] = all(costs["component_passes"].values())
    integrity = {
        "all_rows_pass": all(row["integrity_passes"] for row in lineage_rows),
        "raw_declares_complete": raw.get("complete") is True,
        "row_count": len(rows),
        "pair_count": len(pairs),
        "lineage_count": len(lineage_rows),
    }
    integrity["passes"] = (
        integrity["all_rows_pass"]
        and integrity["raw_declares_complete"]
        and integrity["row_count"] == 192
        and integrity["pair_count"] == 96
        and integrity["lineage_count"] == 32
    )
    go = bool(quality["passes"] and costs["passes"] and integrity["passes"])
    return {
        "schema_version": 1,
        "contract_id": contract["contract_id"],
        "mechanism_id": contract["candidate"]["mechanism_id"],
        "disposition": (
            "go_promote_automatic_depth_default_v0_12"
            if go
            else "no_go_close_automatic_depth_default_candidate"
        ),
        "go": go,
        "quality": quality,
        "costs": costs,
        "integrity": integrity,
        "lineages": lineage_rows,
        "non_claims": [
            "Development-panel values are historical and are not shipping evidence.",
            "This result says nothing about TabArena, CTR23, classification quality, or lockboxes.",
            "Archive bytes are telemetry and are not a gate.",
        ],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    power_contract = json.loads(POWER_CONTRACT.read_text(encoding="utf-8"))
    result = analyze(raw, contract, power_contract)
    result["source_hashes"] = {
        "raw": file_sha256(args.raw),
        "execution_contract": file_sha256(CONTRACT),
        "power_contract": file_sha256(POWER_CONTRACT),
        "analyzer": file_sha256(Path(__file__)),
    }
    args.output.write_bytes(canonical_json_bytes(result))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "disposition": result["disposition"],
                "quality_ratio": result["quality"]["equal_lineage_geomean_ratio"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
