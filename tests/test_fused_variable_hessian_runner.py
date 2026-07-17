from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks import run_fused_variable_hessian as experiment


RECORDED_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "fused_variable_hessian.json"
)
EXPECTED_ARTIFACT_SHA256 = (
    "8af5c94a8561013c94be5d1a9997dbb7b84b9e19733a9dcd4988a1dfc4b4a6cf"
)


def _result(case, config, fit, tree, rss=100.0):
    return {
        "case": case,
        "config": config,
        "fit_seconds": fit,
        "tree_build_seconds": tree,
        "peak_rss_bytes": rss,
        "prediction_sha256": f"{case}-prediction",
        "model_payload_sha256": f"{case}-model",
        "behavior_fingerprint_sha256": f"{case}-behavior",
        "engagement_count": 0 if config == experiment.REFERENCE else 10,
    }


def _results(candidate_fit=0.9, candidate_tree=0.8, candidate_rss=100.0):
    rows = []
    for _ in range(3):
        for case in experiment.CASES:
            rows.extend(
                [
                    _result(case, experiment.REFERENCE, 1.0, 1.0),
                    _result(
                        case,
                        experiment.CANDIDATE,
                        candidate_fit,
                        candidate_tree,
                        candidate_rss,
                    ),
                ]
            )
    return rows


def test_analysis_retains_exact_stable_material_speedup():
    analysis = experiment.analyze(_results())

    assert analysis["passes_all_gates"]
    assert analysis["recommendation"] == (
        "retain_fused_variable_hessian_lane"
    )
    assert analysis["fit_geomean_ratio"] == 0.9
    assert analysis["tree_build_geomean_ratio"] == 0.8


def test_analysis_restores_reference_on_regression_or_mismatch():
    results = _results(candidate_fit=1.03)
    results[1]["model_payload_sha256"] = "different"

    analysis = experiment.analyze(results)

    assert not analysis["passes_all_gates"]
    assert not analysis["gates"]["all_exact"]
    assert not analysis["gates"]["no_fit_regression_over_2pct"]
    assert analysis["recommendation"] == (
        "restore_reference_variable_hessian_dispatch"
    )


def test_declared_orders_reverse_candidate_against_reference():
    relative = [
        order.index(experiment.CANDIDATE)
        > order.index(experiment.REFERENCE)
        for order in experiment.BLOCK_ORDERS
    ]
    assert relative == [True, False, True]


def test_model_payload_hash_ignores_only_runtime_timing(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    changed = tmp_path / "changed.npz"

    def save(path, *, timing, lr=0.1):
        header = np.asarray(
            json.dumps({"timing": timing, "lr": lr}, sort_keys=True)
        )
        np.savez(path, header=header, tree=np.arange(6, dtype=np.float64))

    save(first, timing={"tree_build": 1.0})
    save(second, timing={"tree_build": 2.0})
    save(changed, timing={"tree_build": 2.0}, lr=0.2)

    assert experiment._canonical_model_payload_sha256(
        first
    ) == experiment._canonical_model_payload_sha256(second)
    assert experiment._canonical_model_payload_sha256(
        first
    ) != experiment._canonical_model_payload_sha256(changed)


def test_recorded_artifact_retains_exact_fused_lane():
    raw = RECORDED_ARTIFACT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_ARTIFACT_SHA256
    artifact = json.loads(raw)

    assert artifact["source"]["clean"] is True
    assert artifact["source"]["head"] == (
        "1016e7e8d70c403a70feab7762de8837ea8fd09c"
    )
    assert artifact["protocol"]["sha256"] == hashlib.sha256(
        experiment.PROTOCOL.read_bytes()
    ).hexdigest()
    assert artifact["protocol"]["runner_sha256"] == hashlib.sha256(
        Path(experiment.__file__).read_bytes()
    ).hexdigest()
    assert artifact["protocol"]["lockbox_data_used"] is False
    analysis = artifact["analysis"]
    assert analysis["passes_all_gates"] is True
    assert analysis["recommendation"] == (
        "retain_fused_variable_hessian_lane"
    )
    assert analysis["fit_geomean_ratio"] < 0.79
    assert analysis["tree_build_geomean_ratio"] < 0.77
    for case in experiment.CASES:
        assert analysis["cases"][case]["exact"] is True
