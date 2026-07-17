from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks import run_predict_throughput as baseline
from benchmarks import run_predict_throughput_contiguous_blocks as successor


RECORDED_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "predict_throughput_contiguous_blocks.json"
)
EXPECTED_ARTIFACT_SHA256 = (
    "430341e101194b8bc3fbb98014b568d4bd460517686acea6188d88958619ad61"
)


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


def test_recorded_artifact_keeps_mechanism_but_does_not_close_p2():
    raw = RECORDED_ARTIFACT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_ARTIFACT_SHA256
    artifact = json.loads(raw)

    assert artifact["sources"][baseline.DARKOFIT]["clean"] is True
    assert artifact["sources"][baseline.DARKOFIT]["head"] == (
        "4caa8f2090315e5e9d655dabef6486455624a666"
    )
    assert artifact["protocol"]["sha256"] == hashlib.sha256(
        successor.PROTOCOL.read_bytes()
    ).hexdigest()
    assert artifact["protocol"]["runner_sha256"] == hashlib.sha256(
        Path(successor.__file__).read_bytes()
    ).hexdigest()
    assert artifact["analysis"]["successor_recommendation"] == (
        "start_p2_with_binning"
    )
    assert not artifact["analysis"]["meets_successor_target"]
    assert artifact["analysis"]["successor_gates"] == {
        "peak_rss_ratio_at_most_1_05": True,
        "predecessor_public_target": False,
    }
    for dataset in baseline.DATASETS:
        for rows in baseline.BATCH_SIZES:
            summary = artifact["analysis"]["paired_ratios"][dataset][str(rows)][
                "warm_public"
            ]
            assert summary["median_ratio"] <= 1.015
