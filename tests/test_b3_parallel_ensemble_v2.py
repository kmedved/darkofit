from __future__ import annotations

import copy

import pytest

from benchmarks import run_b3_parallel_ensemble_v2 as runner


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
