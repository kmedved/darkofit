#!/usr/bin/env python3
"""Analyze the frozen T7b automatic-depth spent-sports successor."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from . import paired_evidence_contract as paired
    from . import run_t7b_automatic_depth_sports_v1 as campaign
except ImportError:  # direct script execution
    import paired_evidence_contract as paired
    import run_t7b_automatic_depth_sports_v1 as campaign


def _geomean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or not len(array) or np.any(~np.isfinite(array)) or np.any(array <= 0):
        raise RuntimeError("geometric mean requires positive finite values")
    return float(np.exp(np.mean(np.log(array))))


def _ratio(numerator: Any, denominator: Any) -> float:
    left = float(numerator)
    right = float(denominator)
    if not math.isfinite(left) or not math.isfinite(right) or left <= 0 or right <= 0:
        raise RuntimeError("sports losses and resources must be positive and finite")
    return left / right


def _validate_structure(row: Mapping[str, Any], arm: str) -> None:
    structure = row.get("auto_structure")
    if not isinstance(structure, Mapping):
        raise RuntimeError("sports row is missing automatic-structure metadata")
    resolved = structure.get("resolved", {}).get("depth", {})
    if arm == campaign.CANDIDATE:
        policy = structure.get("candidates", {}).get("depth", {})
        if (
            resolved
            != {"input": None, "resolved": 4, "source": "auto"}
            or policy.get("rule") != campaign.DEPTH_RULE
            or policy.get("branch") != "low_density"
            or policy.get("input_feature_count") != 15
            or policy.get("low_threshold") != 100.0
            or policy.get("high_threshold") != 2_500.0
            or not 0.0 < float(policy.get("effective_rows_per_feature", 0.0)) < 100.0
        ):
            raise RuntimeError("candidate automatic-depth policy did not engage as frozen")
    elif resolved.get("resolved") != 6 or resolved.get("input") is not None:
        raise RuntimeError("control depth resolution drifted")


def validate_raw(
    raw: Mapping[str, Any], contract: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    if (
        raw.get("schema_version") != 1
        or raw.get("contract_id") != campaign.CONTRACT_ID
        or raw.get("status") != "complete"
        or raw.get("contract_sha256") != campaign.file_sha256(campaign.CONTRACT_PATH)
        or raw.get("execution") != campaign.execution_spec()
        or raw.get("claims") != campaign.claim_spec()
        or raw.get("sources") != contract["sources"]
        or raw.get("case_manifests") != contract["case_manifests"]
    ):
        raise RuntimeError("spent-sports raw envelope is invalid")
    rows = raw.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("spent-sports rows are missing")
    expected = {
        (spec["case_id"], arm)
        for spec in campaign.case_specs()
        for arm in campaign.ARMS
    }
    observed = set()
    for row in rows:
        key = (row.get("case_id"), row.get("arm"))
        if key in observed:
            raise RuntimeError(f"duplicate spent-sports row: {key}")
        observed.add(key)
        manifest = contract["case_manifests"].get(str(row.get("case_id")))
        arm = str(row.get("arm"))
        source = contract["sources"].get(arm, {})
        if (
            key not in expected
            or row.get("primary_metric") != "cold_player_rmse"
            or row.get("secondary_metric") != "held_team_rmse"
            or row.get("fingerprints") != manifest["fingerprints"]
            or row.get("fit_rows") != manifest["fit_rows"]
            or row.get("primary_rows") != manifest["primary_rows"]
            or row.get("test_rows") != manifest["test_rows"]
            or row.get("feature_count") != manifest["feature_count"]
            or row.get("source_head") != source.get("head")
            or row.get("source_tree") != source.get("tree")
            or row.get("safe_roundtrip_exact") is not True
            or row.get("requested_depth") is not None
            or row.get("requested_threads") != campaign.THREADS
            or row.get("fitted_thread_counts") != [campaign.THREADS]
            or row.get("runtime_before", {}).get("current") != campaign.THREADS
            or row.get("runtime_after", {}).get("current") != campaign.THREADS
            or row.get("ambient_thread_count_before_fit") != campaign.THREADS
            or row.get("ambient_thread_count_after_predict") != campaign.THREADS
            or not row.get("prediction_sha256")
            or row.get("rss_errors") != []
        ):
            raise RuntimeError(f"spent-sports row contract failed: {key}")
        for name in (
            "primary_loss",
            "secondary_loss",
            "fit_seconds",
            "predict_seconds",
            "peak_rss_bytes",
            "archive_bytes",
        ):
            _ratio(row.get(name), 1.0)
        _validate_structure(row, arm)
    if observed != expected or len(rows) != len(expected):
        raise RuntimeError("spent-sports grid is incomplete")
    return rows


def _bootstrap(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(campaign.SEASONS),):
        raise RuntimeError("sports bootstrap requires exactly three season clusters")
    rng = np.random.default_rng(campaign.BOOTSTRAP_SEED)
    draws = np.empty(campaign.BOOTSTRAP_DRAWS, dtype=np.float64)
    for start in range(0, campaign.BOOTSTRAP_DRAWS, 10_000):
        count = min(10_000, campaign.BOOTSTRAP_DRAWS - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        draws[start : start + count] = np.exp(
            np.mean(np.log(array[indices]), axis=1)
        )
    p2_5, p50, p95, p97_5 = np.percentile(draws, [2.5, 50.0, 95.0, 97.5])
    return {
        "cluster_count": 3,
        "draws": campaign.BOOTSTRAP_DRAWS,
        "seed": campaign.BOOTSTRAP_SEED,
        "percentiles": {
            "p2_5": float(p2_5),
            "p50": float(p50),
            "p95": float(p95),
            "p97_5": float(p97_5),
        },
    }


def analyze_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        pairs[str(row["case_id"])][str(row["arm"])] = row
    expected_cases = {spec["case_id"] for spec in campaign.case_specs()}
    if set(pairs) != expected_cases or any(set(pair) != set(campaign.ARMS) for pair in pairs.values()):
        raise RuntimeError("spent-sports row pairing is incomplete")

    primary = {}
    secondary = {}
    telemetry = {name: {} for name in ("fit_seconds", "predict_seconds", "peak_rss_bytes", "archive_bytes")}
    for case_id, pair in pairs.items():
        control = pair[campaign.CONTROL]
        candidate = pair[campaign.CANDIDATE]
        primary[case_id] = _ratio(candidate["primary_loss"], control["primary_loss"])
        secondary[case_id] = _ratio(candidate["secondary_loss"], control["secondary_loss"])
        if candidate["l2_leaf_reg"] != control["l2_leaf_reg"]:
            raise RuntimeError(f"automatic L2 drifted in {case_id}")
        for name in telemetry:
            telemetry[name][case_id] = _ratio(candidate[name], control[name])

    season_primary = {
        str(season): _geomean(
            value
            for case_id, value in primary.items()
            if case_id.startswith(f"sports_{season}_")
        )
        for season in campaign.SEASONS
    }
    season_secondary = {
        str(season): _geomean(
            value
            for case_id, value in secondary.items()
            if case_id.startswith(f"sports_{season}_")
        )
        for season in campaign.SEASONS
    }
    leave_one_out = {
        f"omit_{season}": _geomean(
            value for key, value in season_primary.items() if key != str(season)
        )
        for season in campaign.SEASONS
    }
    bootstrap = _bootstrap(list(season_primary.values()))
    primary_point = _geomean(primary.values())
    secondary_point = _geomean(secondary.values())
    worst_case = max(primary, key=primary.get)
    worst_season = max(season_primary, key=season_primary.get)
    worst_loo = max(leave_one_out, key=leave_one_out.get)
    rules = campaign.decision_rules()
    gates = {
        "cold_player_aggregate_at_most_1_000": primary_point
        <= rules["cold_player_aggregate_at_most"],
        "held_team_aggregate_at_most_1_010": secondary_point
        <= rules["held_team_aggregate_at_most"],
        "cluster_bootstrap_p95_at_most_1_010": bootstrap["percentiles"]["p95"]
        <= rules["cluster_bootstrap_p95_at_most"],
        "worst_season_at_most_1_020": season_primary[worst_season]
        <= rules["worst_season_at_most"],
        "worst_lineage_at_most_1_030": primary[worst_case]
        <= rules["worst_lineage_at_most"],
        "worst_loo_at_most_1_003": leave_one_out[worst_loo]
        <= rules["worst_loo_at_most"],
    }
    return {
        "disposition": (
            "eligible_for_fresh_tier_d_design"
            if all(gates.values())
            else "closed_after_spent_sports"
        ),
        "gates": gates,
        "case_count": len(primary),
        "season_cluster_count": len(season_primary),
        "cold_player": {
            "point_ratio": primary_point,
            "per_lineage_ratio": primary,
            "season_ratio": season_primary,
            "cluster_bootstrap": bootstrap,
            "leave_one_season_out": leave_one_out,
            "worst_lineage": worst_case,
            "worst_lineage_ratio": primary[worst_case],
            "worst_season": worst_season,
            "worst_season_ratio": season_primary[worst_season],
            "worst_loo_omission": worst_loo,
            "worst_loo_ratio": leave_one_out[worst_loo],
        },
        "held_team": {
            "point_ratio": secondary_point,
            "per_lineage_ratio": secondary,
            "season_ratio": season_secondary,
        },
        "telemetry_only": {
            name: {
                "per_lineage_ratio": values,
                "equal_lineage_geomean_ratio": _geomean(values.values()),
            }
            for name, values in telemetry.items()
        },
        "shipping_or_default_claim_eligible": False,
        "fresh_confirmation_authorized": False,
    }


def analyze_raw_payload(
    raw: Mapping[str, Any], contract: Mapping[str, Any], *, raw_sha256: str
) -> dict[str, Any]:
    rows = validate_raw(raw, contract)
    return {
        "schema_version": 1,
        "contract_id": campaign.CONTRACT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": campaign.file_sha256(campaign.CONTRACT_PATH),
        "raw_sha256": raw_sha256,
        "analysis": analyze_rows(rows),
        "evidence_scope": "spent_player_disjoint_sports_development",
        "shipping_or_default_claim_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=campaign.CONTRACT_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = campaign.load_contract(args.contract)
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    result = analyze_raw_payload(raw, contract, raw_sha256=campaign.file_sha256(args.raw))
    campaign.write_create_only_json(args.output, result)
    print(json.dumps({"output": str(args.output), "disposition": result["analysis"]["disposition"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
