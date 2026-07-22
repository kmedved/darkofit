from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
AUDIT = BENCH / "ensemble_v3_characterization_post_run_audit_20260721.json"
AUDIT_NOTE = BENCH / "ensemble_v3_characterization_post_run_audit_20260721.md"
RAW = BENCH / "ensemble_v3_characterization_raw.json"
RESULT = BENCH / "ensemble_v3_characterization_result.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_post_run_audit_preserves_and_recomputes_immutable_evidence():
    assert _sha256(AUDIT) == (
        "6fcccf098c217e07513a02f6ca588f95deb883f22d92600976077cead406fbdf"
    )
    assert _sha256(AUDIT_NOTE) == (
        "2726b699193eb669e2840bba527743c07d77505dfb8ed782a0fd8076cd94287c"
    )
    audit = json.loads(AUDIT.read_text())
    result = json.loads(RESULT.read_text())

    for record in audit["artifacts"].values():
        assert _sha256(ROOT / record["path"]) == record["sha256"]

    deltas = result["current_resources"]["process_tree_peak_minus_start_bytes"]
    case_medians = {
        case_id: statistics.median(row["paired_ratios"])
        for case_id, row in deltas.items()
    }
    aggregate = math.exp(
        sum(math.log(value) for value in case_medians.values()) / len(case_medians)
    )
    finding = audit["findings"]["peak_delta_aggregate_omission"]
    assert case_medians == finding["per_case_median_paired_ratios"]
    assert aggregate == pytest.approx(
        finding["equal_case_geometric_mean_of_case_median_paired_ratios"]
    )


def test_post_run_audit_recomputes_duration_floor_and_contract_gap():
    audit = json.loads(AUDIT.read_text())
    raw = json.loads(RAW.read_text())
    contract = json.loads(
        (BENCH / "ensemble_v3_characterization_contract.json").read_text()
    )

    timings = [
        timing
        for row in raw["rows"]
        for timing in row["predictions"].values()
    ]
    short = [timing for timing in timings if not timing["minimum_interval_met"]]
    finding = audit["findings"]["prediction_duration_floor"]
    assert len(timings) == finding["formal_interval_count"] == 144
    assert len(short) == finding["short_interval_count"] == 9
    assert min(timing["interval_seconds"] for timing in timings) == pytest.approx(
        finding["minimum_observed_interval_seconds"]
    )
    assert all(timing["rows"] == 8192 for timing in short)
    assert all(
        timing["minimum_interval_met"]
        for row in raw["rows"]
        for timing in row["predictions"].values()
        if timing["rows"] >= 65536
    )

    assert contract["schema_version"] == 1
    assert contract["outcome_blind"] is True
    assert contract["quality_uncertainty"]["leave_one_out"] is True
    assert contract["claims"]["characterization_only"] is True
    assert audit["decision"]["runner_reusable"] is False


def test_current_plans_record_completed_prediction_characterization():
    next_steps = (ROOT / "NEXT_STEPS.md").read_text()
    plan = (ROOT / "COUNTERPUNCH_PLAN.md").read_text()
    assert "dedicated prediction characterization is complete" in next_steps
    assert "current repeat-series grid" in plan
    assert "3.262867x" in next_steps
    assert "3.262867x" in plan
