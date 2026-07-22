from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import m6_quality_rule_v3 as rule


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"

ARTIFACTS = {
    "invariants": (
        BENCH / "t7b_automatic_l2_v1_invariants_20260722.json",
        "c3dee2ecb521648e2f9521e280267d41301361cf9aeccfd84ef77b817f4443f9",
    ),
    "m5": (
        BENCH / "t7b_automatic_l2_v1_m5_20260722.json",
        "3bc489a9304ccd0021ed936b8eeec3bcfb1ab6b37476b8ecc87ffb3943a3c747",
    ),
    "launch": (
        BENCH
        / "t7b_automatic_l2_v1_m6_inspection1_launch_manifest_20260722.json",
        "593d44e331b9be14f1683947315e60eaefbe947b4460e7d99354074564fc4e1f",
    ),
    "raw": (
        BENCH / "t7b_automatic_l2_v1_m6_inspection1_raw_20260722.csv",
        "dfa5560d752f1c17fa8dea0b497d90ebfaf1cb63275f2d69e7ca0afd57677a3a",
    ),
    "result": (
        BENCH / "t7b_automatic_l2_v1_m6_inspection1_result_20260722.json",
        "6fc5ececda62da257fd3e00fce7df1b8dba2978501e689d0d2a2ca678f296f26",
    ),
    "manifest": (
        BENCH
        / "t7b_automatic_l2_v1_m6_inspection1_result_20260722.json.manifest.json",
        "034bfbc47a2ef1fe872efa57cc52f3eb97d5986e269cc219e0bde802eab558d8",
    ),
    "terminal": (
        BENCH
        / "t7b_automatic_l2_v1_m6_inspection1_terminal_attestation_20260722.json",
        "6bc045080cb6db0a38f912d5d7b31d10d5483e392520dcf682d057db43d05419",
    ),
}


def _load(name: str):
    return json.loads(ARTIFACTS[name][0].read_text())


def test_t7b_automatic_l2_artifacts_retain_their_create_only_hashes():
    for path, expected in ARTIFACTS.values():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_t7b_automatic_l2_pre_quality_invariants_and_m5_passed():
    invariants = _load("invariants")
    m5 = _load("m5")

    assert invariants["contract_id"] == (
        "t7b-automatic-scalar-rmse-l2-v1-20260722"
    )
    assert invariants["quality_outcomes_inspected"] is False
    assert invariants["analysis"]["all_noop_cases_exact"] is True
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


def test_t7b_automatic_l2_terminal_kill_recomputes_from_raw_rows():
    launch = _load("launch")
    result = _load("result")
    manifest = _load("manifest")
    terminal = _load("terminal")
    with ARTIFACTS["raw"][0].open(newline="") as handle:
        recomputed = rule.analyze_rows(list(csv.DictReader(handle)))

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
    assert recomputed["disposition"] == "kill"
    assert recomputed["gates"] == {
        "aggregate_not_worse": False,
        "loo_concentration_at_most_1_003": True,
        "worst_group_at_most_1_02": True,
    }
    assert launch["inspection_spent_on_manifest_creation"] is True
    assert launch["rerun_authorized"] is False
    assert manifest["inspection_spent"] is True
    assert manifest["raw_csv"]["sha256"] == ARTIFACTS["raw"][1]
    assert terminal["disposition"] == "kill"
    assert terminal["rerun_authorized"] is False
    assert terminal["shipping_or_default_claim_eligible"] is False
    assert terminal["launch_manifest_sha256"] == ARTIFACTS["launch"][1]
    assert terminal["raw_sha256"] == ARTIFACTS["raw"][1]
    assert terminal["result_sha256"] == ARTIFACTS["result"][1]
    assert terminal["m6_manifest_sha256"] == ARTIFACTS["manifest"][1]
