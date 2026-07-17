from __future__ import annotations

from benchmarks import run_predict_throughput as baseline
from benchmarks import run_predict_throughput_integrated as integrated


def _case(seconds):
    return {
        "warm_equals_integrated": True,
        "integrated_public": {
            "seconds_per_call": seconds,
            "minimum_interval_passed": True,
        },
    }


def _worker(arm, seconds, fingerprint):
    return {
        "arm": arm,
        "behavior_fingerprint_sha256": fingerprint,
        "cases": {
            dataset: {
                str(rows): _case(seconds)
                for rows in baseline.BATCH_SIZES
            }
            for dataset in baseline.DATASETS
        },
    }


def test_integrated_analysis_closes_stable_target():
    results = []
    for _ in range(3):
        results.extend(
            [
                _worker(baseline.DARKOFIT, 0.8, "dark"),
                _worker(baseline.CHIMERABOOST, 1.0, "chimera"),
            ]
        )
    canonical = {
        baseline.DARKOFIT: results[0],
        baseline.CHIMERABOOST: results[1],
    }
    rss = {
        baseline.DARKOFIT: [100, 100, 100],
        baseline.CHIMERABOOST: [100, 100, 100],
    }

    result = integrated.analyze(canonical, results, rss)

    assert result["passes_all_gates"]
    assert result["recommendation"] == (
        "close_p2_matched_prediction_target"
    )
    assert result["stretch_public_cases_at_or_below_chimera"] == 8


def test_integrated_analysis_fails_short_interval_and_rss():
    results = []
    for _ in range(3):
        results.extend(
            [
                _worker(baseline.DARKOFIT, 0.8, "dark"),
                _worker(baseline.CHIMERABOOST, 1.0, "chimera"),
            ]
        )
    results[0]["cases"]["basketball_numeric"]["8192"]["integrated_public"][
        "minimum_interval_passed"
    ] = False
    canonical = {
        baseline.DARKOFIT: results[0],
        baseline.CHIMERABOOST: results[1],
    }
    rss = {
        baseline.DARKOFIT: [110, 110, 110],
        baseline.CHIMERABOOST: [100, 100, 100],
    }

    result = integrated.analyze(canonical, results, rss)

    assert not result["passes_all_gates"]
    assert not result["gates"]["all_intervals_at_least_0_75_seconds"]
    assert not result["gates"]["peak_rss_ratio_at_most_1_05"]


def test_integrated_repeats_are_frozen_and_seconds_scaled():
    assert integrated.INTEGRATED_REPEATS == {
        8_192: 256,
        65_536: 32,
        524_288: 4,
        2_000_000: 2,
    }
    assert set(integrated.INTEGRATED_REPEATS) == set(
        baseline.BATCH_SIZES
    )
