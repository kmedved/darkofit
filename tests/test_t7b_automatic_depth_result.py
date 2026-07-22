from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from benchmarks import m6_quality_rule_v3 as rule


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"

ARTIFACTS = {
    "invariants": (
        BENCH / "t7b_automatic_depth_v1_invariants_20260722.json",
        "02362e5d7080c155add0846a58b6960db997bd29a0374e936a16a5a5364e5aff",
    ),
    "m5": (
        BENCH / "t7b_automatic_depth_v1_m5_20260722.json",
        "1d3eac70f81babeb628850cf19844d7b4c590c6df67ded723fcf7caba019bca1",
    ),
    "launch": (
        BENCH
        / "t7b_automatic_depth_v1_m6_inspection1_launch_manifest_20260722.json",
        "7eb95710c761f0682c00cf4b5971233089c70e654c5e5adc316d5388d933dc46",
    ),
    "raw": (
        BENCH / "t7b_automatic_depth_v1_m6_inspection1_raw_20260722.csv",
        "e8e651459fafdea7ace0d298ccedd2c8d87145b945928111d475a007b955bafe",
    ),
    "result": (
        BENCH / "t7b_automatic_depth_v1_m6_inspection1_result_20260722.json",
        "7af0c480221b5886c7bbf41f810147663d9da6e2c4171a70bc9db3a431eebb28",
    ),
    "manifest": (
        BENCH
        / "t7b_automatic_depth_v1_m6_inspection1_result_20260722.json.manifest.json",
        "dbb47702f4e7992f34e653ea1155a8638e4e1945dbda0da1eb582345c73c32c7",
    ),
    "terminal": (
        BENCH
        / "t7b_automatic_depth_v1_m6_inspection1_terminal_attestation_20260722.json",
        "b925aab09fdd71ca0f8887e1d3a4023c20412b2eefc337f2a2a7c1d5a267f598",
    ),
}


def _load(name: str):
    return json.loads(ARTIFACTS[name][0].read_text())


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def test_t7b_automatic_depth_artifacts_retain_their_create_only_hashes():
    for path, expected in ARTIFACTS.values():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_t7b_automatic_depth_pre_quality_invariants_and_m5_passed():
    invariants = _load("invariants")
    m5 = _load("m5")

    assert invariants["contract_id"] == (
        "t7b-automatic-scalar-rmse-depth-v1-20260722"
    )
    assert invariants["quality_outcomes_inspected"] is False
    assert invariants["analysis"]["all_noop_cases_exact"] is True
    assert invariants["analysis"]["all_depth_branches_engaged"] is True
    assert {
        item["candidate_depth"]
        for item in invariants["analysis"]["engagement"].values()
    } == {4, 6, 8}
    assert all(
        item["prediction_exact"] and item["fitted_state_exact"]
        for item in invariants["analysis"]["comparisons"].values()
    )
    assert m5["non_ranking"] is True
    assert m5["shipping_or_default_claim_authorized"] is False
    assert m5["analysis"]["behavior_fingerprints_equal_between_arms"] is True
    assert m5["analysis"]["baseline_drift"] == []
    assert m5["analysis"]["advancement_blocked_for_drift"] is False
    assert all(
        check["passed"]
        for check in m5["analysis"]["known_floor_checks"].values()
    )


def test_t7b_automatic_depth_advance_recomputes_from_raw_rows():
    launch = _load("launch")
    result = _load("result")
    manifest = _load("manifest")
    terminal = _load("terminal")
    with ARTIFACTS["raw"][0].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    recomputed = rule.analyze_rows(rows)

    recorded = result["analysis"]
    for name in (
        "disposition",
        "gates",
        "case_count",
        "group_count",
        "worst_case",
        "worst_group",
        "worst_loo_omission",
    ):
        assert recomputed[name] == recorded[name]
    for name in (
        "geometric_mean_ratio",
        "worst_case_ratio",
        "worst_group_ratio",
        "worst_loo_ratio",
    ):
        assert recomputed[name] == pytest.approx(recorded[name], rel=1e-14)
    for name in (
        "ratios",
        "group_geometric_mean_ratio",
        "leave_one_group_out_geometric_mean_ratio",
    ):
        assert recomputed[name] == pytest.approx(recorded[name], rel=1e-14)
    assert recomputed["disposition"] == "advance"
    assert recomputed["gates"] == {
        "aggregate_not_worse": True,
        "loo_concentration_at_most_1_003": True,
        "worst_group_at_most_1_02": True,
    }

    paired = {}
    for row in rows:
        key = (
            row["dataset"],
            row["size"],
            row["seed"],
            row["weight_mode"],
        )
        paired.setdefault(key, {})[row["variant"]] = row
    classification_pairs = [
        pair
        for pair in paired.values()
        if pair["control_default"]["task"] != "regression"
    ]
    assert len(classification_pairs) == 36
    assert all(
        pair["control_default"]["prediction_sha256"]
        == pair["candidate_default"]["prediction_sha256"]
        and pair["control_default"]["probability_sha256"]
        == pair["candidate_default"]["probability_sha256"]
        for pair in classification_pairs
    )

    fit_ratios = [
        float(pair["candidate_default"]["fit_seconds"])
        / float(pair["control_default"]["fit_seconds"])
        for pair in paired.values()
    ]
    assert _geomean(fit_ratios) == pytest.approx(0.8495008317649614)

    assert result["candidate_ranking_eligible"] is True
    assert result["shipping_or_default_claim_eligible"] is False
    assert launch["inspection_spent_on_manifest_creation"] is True
    assert launch["rerun_authorized"] is False
    assert manifest["inspection_spent"] is True
    assert manifest["raw_csv"]["sha256"] == ARTIFACTS["raw"][1]
    assert terminal["disposition"] == "advance"
    assert terminal["rerun_authorized"] is False
    assert terminal["shipping_or_default_claim_eligible"] is False
    assert terminal["launch_manifest_sha256"] == ARTIFACTS["launch"][1]
    assert terminal["raw_sha256"] == ARTIFACTS["raw"][1]
    assert terminal["result_sha256"] == ARTIFACTS["result"][1]
    assert terminal["m6_manifest_sha256"] == ARTIFACTS["manifest"][1]
