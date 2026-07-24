from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import run_b3_parallel_ensemble_v2 as runner


BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
ARTIFACT_HASHES = {
    "b3_parallel_ensemble_v2_launch_20260723.json":
        "707a4fc3d7283023721ed61417b20e52254d4a7b417e35696e6fe052fbf040a3",
    "b3_parallel_ensemble_v2_raw_20260723.json":
        "1d48276bcde51e9fedd778d35ba521a954ac40f6e927626442c627e0a52b7be1",
    "b3_parallel_ensemble_v2_result_20260723.json":
        "f2e34bcb695f28ceea8309177a86e239ae170f53b8c66da7b9d29b55006f7c9c",
}


def _load(name):
    return json.loads((BENCHMARKS / name).read_text())


def _model(route):
    parallel = route == "process_parallel"
    return {
        "member_seeds": list(range(8)),
        "sampled_indices_sha256": ["a"] * 8,
        "oob_indices_sha256": ["b"] * 8,
        "best_iterations": [10] * 8,
        "fitted_thread_counts": [2 if parallel else 14] * 8,
        "prediction_thread_counts": [2 if parallel else 14] * 8,
        "schedule": (
            {
                "contract": runner.v1.CONTRACT_ID,
                "mode": "private_process_workers",
                "workers": 7,
                "member_threads": 2,
                "total_thread_budget": 14,
                "maximum_model_threads": 14,
                "result_order": "member_index",
            }
            if parallel
            else None
        ),
        "sequential": not parallel,
    }


def _rows(engaged_ratio=0.5, fallback_ratio=1.0):
    rows = []
    for case_id in runner.CASES:
        route = runner.EXPECTED_ROUTES[case_id]
        ratio = (
            engaged_ratio if route == "process_parallel" else fallback_ratio
        )
        for block in range(runner.BLOCKS):
            for arm in runner.ARMS:
                candidate = arm == "parallel_7x2"
                model_route = route if candidate else "sequential_fallback"
                records = []
                for mode in runner.MODES:
                    records.append({
                        "mode": mode,
                        "fit_seconds": 10.0 * ratio if candidate else 10.0,
                        "prediction_sha256": "prediction",
                        "probability_sha256": None,
                        "archive_bytes": 100,
                        "fit_rss": {
                            "peak_bytes": 500_000_000 if candidate else 300_000_000,
                            "errors": [],
                        },
                        "model": _model(model_route),
                    })
                rows.append({
                    "case_id": case_id,
                    "block": block,
                    "arm": arm,
                    "fingerprints": {"case": case_id},
                    "records": records,
                })
    return rows


def test_analyzer_accepts_exact_activation_gated_speedup():
    result = runner.analyze(_rows())

    assert result["checks"] == {
        "behavior_exact": True,
        "routes_match_frozen_work_rule": True,
        "resource_sampling_clean": True,
        "memory_bounded": True,
        "engaged_cold_direction_stable": True,
        "engaged_steady_direction_stable": True,
        "fallback_cold_not_materially_slower": True,
        "fallback_steady_not_materially_slower": True,
    }
    assert result["speed"]["cold_executor"]["engaged_geomean_ratio"] == (
        pytest.approx(0.5)
    )
    assert result["disposition"] == "ready_to_productize"


def test_analyzer_rejects_route_drift_and_engaged_regression():
    drifted = _rows()
    target = next(
        row
        for row in drifted
        if row["case_id"] == "general_friedman_numeric"
        and row["arm"] == "parallel_7x2"
    )
    for record in target["records"]:
        record["model"] = _model("process_parallel")
    assert runner.analyze(drifted)["checks"][
        "routes_match_frozen_work_rule"
    ] is False

    invalid_control = _rows()
    control = next(
        row
        for row in invalid_control
        if row["case_id"] == "general_numeric_binary"
        and row["arm"] == "sequential_1x14"
    )
    control["records"][0]["model"]["fitted_thread_counts"] = [2] * 8
    assert runner.analyze(invalid_control)["checks"][
        "routes_match_frozen_work_rule"
    ] is False

    slower = runner.analyze(_rows(engaged_ratio=1.01))
    assert slower["checks"]["engaged_cold_direction_stable"] is False
    assert slower["disposition"] == "needs_revision"


def test_analyzer_rejects_incomplete_or_nonexact_rows():
    with pytest.raises(RuntimeError, match="census"):
        runner.analyze(_rows()[:-1])

    changed = copy.deepcopy(_rows())
    changed[0]["records"][0]["prediction_sha256"] = "different"
    assert runner.analyze(changed)["checks"]["behavior_exact"] is False


def test_committed_b3_v2_result_recomputes_and_artifacts_are_immutable():
    for name, expected in ARTIFACT_HASHES.items():
        assert hashlib.sha256((BENCHMARKS / name).read_bytes()).hexdigest() == (
            expected
        )
    launch = _load("b3_parallel_ensemble_v2_launch_20260723.json")
    raw = _load("b3_parallel_ensemble_v2_raw_20260723.json")
    result = _load("b3_parallel_ensemble_v2_result_20260723.json")

    assert launch["source"]["head"] == (
        "b35c092bbdfef45f2ac4d5b0cc16eaaf1c89bf55"
    )
    assert launch["minimum_work"] == runner.MINIMUM_WORK
    assert runner.analyze(raw["rows"]) == result["analysis"]
    assert result["analysis"]["disposition"] == "ready_to_productize"
