from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from benchmarks import run_large_n_engine as experiment


RECORDED_ARTIFACT = (
    Path(__file__).parents[1] / "benchmarks" / "large_n_engine.json"
)
EXPECTED_ARTIFACT_SHA256 = (
    "ac9e6e9f136117b7b1db7488b38f660561195f86b29ae2f87868a5d293c62508"
)


def _result(arm, rows, fit, rmse, rss, fingerprint, *, block):
    return {
        "arm": arm,
        "train_rows": rows,
        "block": block,
        "position": experiment.BLOCK_ORDERS[block].index(arm),
        "fit_seconds": fit,
        "predict_seconds": 0.1,
        "rmse": rmse,
        "peak_rss_bytes": rss,
        "data_probe_sha256": f"data-{rows}",
        "worker_stderr": None,
        "behavior_fingerprint_sha256": fingerprint,
        "metadata": {
            "fused_engagement_count": (
                1800 if arm == experiment.DARKO else 0
            )
        },
    }


def _results(darko_ratio=0.70, rmse_ratio=1.001):
    results = []
    for block in range(3):
        for rows in experiment.TRAIN_ROWS:
            results.extend(
                (
                    _result(
                        experiment.DARKO,
                        rows,
                        darko_ratio,
                        rmse_ratio,
                        90,
                        f"darko-{rows}",
                        block=block,
                    ),
                    _result(
                        experiment.CHIMERA,
                        rows,
                        1.0,
                        1.0,
                        100,
                        f"chimera-{rows}",
                        block=block,
                    ),
                )
            )
    return results


def test_analysis_certifies_fast_quality_neutral_lane():
    analysis = experiment.analyze(_results())

    assert analysis["passes_all_gates"] is True
    assert analysis["fit_geomean_speedup"] > 1.30
    assert analysis["recommendation"] == (
        "certify_large_n_engine_advantage"
    )


def test_analysis_rejects_speed_or_quality_failure():
    slow = experiment.analyze(_results(darko_ratio=0.80))
    regret = experiment.analyze(_results(rmse_ratio=1.003))

    assert slow["gates"]["fit_geomean_at_most_1_over_1_30"] is False
    assert regret["gates"]["quality_noninferior"] is False
    assert slow["recommendation"] == (
        "do_not_claim_large_n_engine_advantage"
    )
    assert regret["recommendation"] == (
        "do_not_claim_large_n_engine_advantage"
    )


def test_analysis_pairs_reciprocal_blocks_after_result_shuffle():
    results = _results()
    random.Random(4).shuffle(results)

    analysis = experiment.analyze(results)

    assert analysis["passes_all_gates"] is True
    assert analysis["gates"]["no_worker_stderr"] is True


def test_analysis_rejects_duplicate_data_drift_and_worker_stderr():
    duplicate = _results()
    duplicate[-1] = dict(duplicate[0])
    with pytest.raises(RuntimeError, match="duplicate coordinate"):
        experiment.analyze(duplicate)

    drift = _results()
    drift[-1]["data_probe_sha256"] = "different"
    with pytest.raises(RuntimeError, match="data identity differs"):
        experiment.analyze(drift)

    noisy = _results()
    noisy[-1]["worker_stderr"] = "warning"
    analysis = experiment.analyze(noisy)
    assert analysis["gates"]["no_worker_stderr"] is False
    assert analysis["passes_all_gates"] is False

    missing = _results()
    del missing[-1]["worker_stderr"]
    with pytest.raises(RuntimeError, match="stderr record is missing"):
        experiment.analyze(missing)


def test_behavior_metadata_excludes_only_observational_timing():
    metadata = {
        "fitted_tree_count": 300,
        "final_fit": {
            "stop_reason": "iteration_limit",
            "phase_seconds": {"tree_build": 1.2},
        },
        "selection_fit": None,
    }

    behavior = experiment._behavior_metadata(metadata)

    assert behavior["final_fit"]["phase_seconds"] is None
    assert behavior["final_fit"]["stop_reason"] == "iteration_limit"
    assert metadata["final_fit"]["phase_seconds"] == {"tree_build": 1.2}
    json.dumps(behavior, allow_nan=False)


def test_recorded_artifact_closes_large_n_claim_and_survives_audit():
    raw = RECORDED_ARTIFACT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_ARTIFACT_SHA256
    artifact = json.loads(raw)

    assert artifact["sources"]["darkofit"]["head"] == (
        "d77dc8b6ede6e87c84aaebfb3c9d03447c48cbb9"
    )
    assert artifact["sources"]["chimeraboost"]["head"] == (
        experiment.EXPECTED_CHIMERA_HEAD
    )
    assert artifact["protocol"]["runner_sha256"] == (
        "3f5411b03c58c9a56cd1549510b702cadfd4b27a319010e1dabad7871363ab26"
    )
    recorded = artifact["analysis"]
    assert recorded["passes_all_gates"] is False
    assert recorded["fit_geomean_speedup"] == pytest.approx(1.2792983257)
    assert recorded["recommendation"] == (
        "do_not_claim_large_n_engine_advantage"
    )

    audited = experiment.analyze(artifact["results"])
    assert audited["gates"]["no_worker_stderr"] is True
    assert audited["fit_geomean_ratio"] == pytest.approx(
        recorded["fit_geomean_ratio"]
    )
    assert audited["recommendation"] == recorded["recommendation"]
