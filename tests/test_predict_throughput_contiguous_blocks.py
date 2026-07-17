from __future__ import annotations

from benchmarks import run_predict_throughput as baseline
from benchmarks import run_predict_throughput_contiguous_blocks as successor


def _case(public, binning=0.4, core=0.4):
    return {
        "exactness": {
            "cold_equals_warm": True,
            "public_equals_packed_core": True,
        },
        "warm_public": {"median_seconds": public},
        "binning": {"median_seconds": binning},
        "packed_core": {"median_seconds": core},
    }


def _worker(arm, public, fingerprint):
    return {
        "arm": arm,
        "behavior_fingerprint_sha256": fingerprint,
        "cases": {
            dataset: {
                str(rows): _case(public)
                for rows in baseline.BATCH_SIZES
            }
            for dataset in baseline.DATASETS
        },
    }


def test_successor_closes_stable_public_target_with_bounded_rss():
    results = []
    for _ in range(3):
        results.extend(
            [
                _worker(baseline.DARKOFIT, 0.9, "dark"),
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

    analysis = successor.analyze_successor(canonical, results, rss)

    assert analysis["meets_successor_target"]
    assert (
        analysis["successor_recommendation"]
        == "close_p2_matched_prediction_target"
    )
    assert analysis["stretch_public_cases_at_or_below_chimera"] == 8


def test_successor_retains_predecessor_failure_on_unstable_ratios():
    results = []
    for darko in (0.8, 1.4, 0.8):
        results.extend(
            [
                _worker(baseline.DARKOFIT, darko, "dark"),
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

    analysis = successor.analyze_successor(canonical, results, rss)

    assert not analysis["meets_successor_target"]
    assert not analysis["successor_gates"]["predecessor_public_target"]


def test_parse_args_uses_create_only_successor_default(tmp_path):
    args = successor.parse_args(["--data-cache", str(tmp_path / "cache.csv")])
    assert args.output == successor.DEFAULT_OUTPUT.resolve()
