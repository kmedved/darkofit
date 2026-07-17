from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks import analyze_native_ordinal_c2 as analysis
from benchmarks import run_native_ordinal_c2 as runner


def _quality_pair(task_id: int, ratio: float, fold: int = 0):
    task = {
        "task_id": task_id,
        "dataset_name": f"task_{task_id}",
        "lineage_cluster": f"lineage_{task_id}",
        "ordinal_features": {"level": ["low", "high"]},
    }
    control = {
        "test": {"rmse": 1.0},
        "validation": {"rmse": 1.0},
        "fit_seconds": 1.0,
        "public_predict_timing": {"seconds_per_call": 1.0},
        "peak_rss_bytes": 100,
        "fold": fold,
    }
    candidate = {
        "test": {"rmse": ratio},
        "validation": {"rmse": ratio},
        "fit_seconds": 1.0,
        "public_predict_timing": {"seconds_per_call": 1.0},
        "peak_rss_bytes": 100,
        "fold": fold,
    }
    return {"task": task, "control": control, "candidate": candidate}


def test_development_split_is_deterministic_disjoint_and_target_free():
    outer = np.arange(100, 300, dtype=np.int64)
    first = runner.development_split(outer, task_id=123, fold=2)
    second = runner.development_split(outer, task_id=123, fold=2)
    assert first[2] == second[2]
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert len(first[0]) == 160
    assert len(first[1]) == 40
    assert not np.intersect1d(first[0], first[1]).size
    assert np.array_equal(
        np.sort(np.concatenate((first[0], first[1]))), outer
    )
    changed = runner.development_split(outer, task_id=123, fold=1)
    assert not np.array_equal(first[1], changed[1])


def test_reciprocal_order_alternates_by_coordinate():
    assert runner.reciprocal_order(0) == (
        runner.CONTROL,
        runner.CANDIDATE,
    )
    assert runner.reciprocal_order(1) == (
        runner.CANDIDATE,
        runner.CONTROL,
    )
    assert runner.reciprocal_order(22) == runner.reciprocal_order(0)


def test_behavior_fingerprint_excludes_only_measurement_and_parent_fields():
    result = {
        "fit_seconds": 1.0,
        "peak_rss_bytes": 100,
        "public_predict_timing": {
            "total_seconds": 2.0,
            "seconds_per_call": 0.1,
            "calls": 20,
            "last_prediction_sha256": "prediction",
        },
        "model": {"logical": "state"},
    }
    fingerprint = runner._json_sha256(runner._behavior_payload(result))
    decorated = {
        **result,
        "behavior_fingerprint_sha256": fingerprint,
        "worker_returncode": 0,
        "worker_stdout": None,
        "worker_stderr": None,
        "coordinate_index": 1,
        "position": 0,
    }
    assert runner._json_sha256(
        runner._behavior_payload(decorated)
    ) == fingerprint
    decorated["model"]["logical"] = "changed"
    assert runner._json_sha256(
        runner._behavior_payload(decorated)
    ) != fingerprint


def test_empty_explicit_ordinal_archive_normalizes_to_control():
    from darkofit import DarkoRegressor

    X = pd.DataFrame({
        "category": pd.Categorical(["a", "b"] * 30),
        "numeric": np.arange(60, dtype=np.float64),
    })
    y = np.sin(np.arange(60, dtype=np.float64))
    params = {"random_state": 4, "iterations": 3}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        control = DarkoRegressor(**params).fit(
            X, y, cat_features=["category"]
        )
        candidate = DarkoRegressor(**params).fit(
            X,
            y,
            cat_features=["category"],
            ordinal_features={},
        )
    assert np.array_equal(control.predict(X), candidate.predict(X))
    assert (
        runner._archive_identity(
            control, runner.CONTROL
        )["normalized_logical_sha256"]
        == runner._archive_identity(
            candidate, runner.CANDIDATE
        )["normalized_logical_sha256"]
    )


def test_candidate_telemetry_contract_is_exact():
    task = {
        "task_id": 1,
        "feature_record": {
            "categorical_indices": [1, 2],
            "ordinal_features": [{
                "index": 1,
                "feature": "level",
                "categories": ["low", "high"],
            }],
        },
    }
    result = {
        "task_id": 1,
        "fold": 0,
        "arm": runner.CANDIDATE,
        "model": {
            "ordinal_state": {
                "mode": "explicit",
                "records": [{
                    "index": 1,
                    "name": "level",
                    "categories": ["low", "high"],
                    "source": "explicit",
                }],
                "indices": [1],
                "metadata": {
                    "mode": "explicit",
                    "active": True,
                    "feature_count": 1,
                    "feature_indices": [1],
                    "feature_names": ["level"],
                    "sources": ["explicit"],
                    "nominal_categorical_count": 1,
                    "added_columns": 0,
                    "target_stat_blocks_added": 0,
                    "target_used": False,
                    "unknown_policy": "fail_closed",
                    "missing_policy": "numeric_missing_bin",
                },
            },
            "preprocessor": {
                "cat_features": [2],
                "num_features": [0, 1],
            },
        },
    }
    failures = []
    analysis._validate_telemetry(result, task, failures)
    assert failures == []
    result["model"]["ordinal_state"]["metadata"]["target_used"] = True
    analysis._validate_telemetry(result, task, failures)
    assert failures == [
        "task 1 fold 0 arm candidate: candidate ordinal telemetry changed"
    ]


def test_no_engagement_pair_requires_prediction_and_model_exactness():
    common = {
        "tier": runner.DEVELOPMENT,
        "task_id": 1,
        "dataset_id": 2,
        "dataset_name": "control",
        "lineage_cluster": "lineage",
        "role": "nominal_no_engagement_guardrail",
        "fold": 0,
        "source": {},
        "runtime": {},
        "registry_sha256": "registry",
        "authorization": None,
        "categorical_indices": [0],
        "declared_ordinal_features": {},
        "outer_split": {},
        "inner_split": {},
        "validation": {"prediction_sha256": "v", "rmse": 1.0},
        "test": {"prediction_sha256": "t", "rmse": 1.0},
        "model": {
            "preprocessor": {"cat_features": [0]},
            "archive": {"normalized_logical_sha256": "model"},
        },
    }
    control = json.loads(json.dumps(common))
    control["arm"] = runner.CONTROL
    candidate = json.loads(json.dumps(common))
    candidate["arm"] = runner.CANDIDATE
    task = {"ordinal_features": {}}
    failures = []
    analysis._validate_pair(control, candidate, task, failures)
    assert failures == []
    candidate["model"]["archive"]["normalized_logical_sha256"] = "changed"
    analysis._validate_pair(control, candidate, task, failures)
    assert failures == [
        "task 1 fold 0: no-engagement logical model state differs"
    ]


def test_development_aggregate_and_gates_apply_equal_task_weighting():
    pairs = [
        _quality_pair(task_id, ratio, fold)
        for task_id, ratio in enumerate((0.95, 0.96, 0.97, 0.98), start=1)
        for fold in range(3)
    ]
    aggregate = analysis._aggregate(runner.DEVELOPMENT, pairs)
    assert aggregate["engaged_task_count"] == 4
    assert aggregate["engaged_split_count"] == 12
    assert aggregate["task_wins"] == 4
    assert aggregate["equal_task_test_rmse_ratio"] == pytest.approx(
        float(np.exp(np.mean(np.log([0.95, 0.96, 0.97, 0.98]))))
    )
    gates = analysis._gates(runner.DEVELOPMENT, aggregate, [])
    assert all(gate["passes"] for gate in gates.values())
    failed = analysis._gates(
        runner.DEVELOPMENT, aggregate, ["integrity failure"]
    )
    assert failed["integrity"]["passes"] is False


def test_confirmation_bootstrap_is_deterministic_and_cluster_level():
    ratios = [0.95, 0.96, 0.97, 0.98, 0.99]
    first = analysis.confirmation_bootstrap_upper(ratios)
    second = analysis.confirmation_bootstrap_upper(ratios)
    assert first == second
    assert first["draws"] == 100_000
    assert first["upper_ratio"] < 1.0
    with pytest.raises(ValueError):
        analysis.confirmation_bootstrap_upper(ratios[:4])


def test_confirmation_authorization_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    registry = runner._load_registry()
    valid = {
        "decision": "authorize_native_ordinal_c2_confirmation_once",
        "passes": True,
        "confirmation_run_authorized": True,
        "registry_sha256": registry["registry_sha256"],
        "raw_sha256": "d" * 64,
        "tier": runner.DEVELOPMENT,
        "development_outcomes_inspected": True,
        "confirmation_outcomes_inspected": False,
        "lockbox_touched": False,
    }
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = (
        tmp_path
        / "benchmarks"
        / "native_ordinal_c2_development_result.json"
    )
    path.parent.mkdir()
    path.write_text(json.dumps(valid), encoding="utf-8")
    record = runner._validate_authorization(
        runner.CONFIRMATION, path, registry
    )
    assert record["decision"] == valid["decision"]
    valid["passes"] = False
    path.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not authorized"):
        runner._validate_authorization(
            runner.CONFIRMATION, path, registry
        )
    with pytest.raises(RuntimeError, match="must not use authorization"):
        runner._validate_authorization(
            runner.DEVELOPMENT, path, registry
        )


def test_analysis_outputs_are_atomic_and_create_only(tmp_path: Path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.md"
    analysis._atomic_create_many({first: b"one", second: b"two"})
    assert first.read_bytes() == b"one"
    assert second.read_bytes() == b"two"
    with pytest.raises(FileExistsError):
        analysis._atomic_create_many({first: b"changed"})
    assert first.read_bytes() == b"one"
