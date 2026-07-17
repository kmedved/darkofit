import pytest

from benchmarks import run_fused_subset_oblivious as experiment


def _result(hessian_case, lane, config, block):
    candidate = config == experiment.CANDIDATE
    behavior = f"{hessian_case}/{lane}"
    return {
        "hessian_case": hessian_case,
        "sampling_lane": lane,
        "config": config,
        "sampling": dict(experiment.SAMPLING_LANES[lane]),
        "iterations": experiment.ITERATIONS,
        "prediction_sha256": behavior,
        "model_payload_sha256": behavior,
        "behavior_fingerprint_sha256": behavior,
        "engagement_count": 6 if candidate else 0,
        "resolved_thread_count": experiment.THREADS,
        "selected_tree_mode": "catboost",
        "fit_seconds": 0.8 if candidate else 1.0,
        "tree_build_seconds": 0.7 if candidate else 1.0,
        "peak_rss_bytes": 100 if candidate else 100,
        "data": {"fingerprint": "data"},
        "support_sha256": {"darkofit/tree.py": "source"},
        "block": block,
    }


def _passing_results():
    rows = []
    for block, order in enumerate(experiment.BLOCK_ORDERS):
        for hessian_case in experiment.HESSIAN_CASES:
            for lane in experiment.SAMPLING_LANES:
                for config in order:
                    rows.append(_result(hessian_case, lane, config, block))
    return rows


def test_analyze_accepts_exact_stable_speedup():
    analysis = experiment.analyze(_passing_results())

    assert analysis["passes_all_gates"]
    assert analysis["recommendation"] == "retain_fused_subset_lanes"
    assert analysis["subset_fit_geomean_ratio"] == pytest.approx(0.8)
    assert analysis["subset_tree_build_geomean_ratio"] == pytest.approx(0.7)


def test_analyze_rejects_behavior_mismatch():
    results = _passing_results()
    results[0]["prediction_sha256"] = "changed"

    analysis = experiment.analyze(results)

    assert not analysis["passes_all_gates"]
    assert not analysis["gates"]["all_exact"]
    assert analysis["recommendation"] == "restore_full_lane_only_dispatch"


def test_analyze_rejects_missing_coordinate():
    results = _passing_results()
    results.pop()

    try:
        experiment.analyze(results)
    except RuntimeError as error:
        assert "expected 48 worker rows" in str(error)
    else:
        raise AssertionError("missing benchmark coordinate was accepted")
