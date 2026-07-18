import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks import analyze_t5_composite_confirmation as analyzer
from benchmarks import run_t5_composite_confirmation as runner


def test_cross_augmentation_is_target_free_ordered_and_nonmutating():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 5.0], "c": ["x", "y"]})
    original = X.copy()
    pairs = [(0, 1, "diff"), (0, 1, "prod")]
    result = runner._augment_crosses(X, pairs)
    pd.testing.assert_frame_equal(X, original)
    assert result.columns.tolist() == [
        "a",
        "b",
        "c",
        "__darkofit_cross_0_1_diff",
        "__darkofit_cross_0_1_prod",
    ]
    np.testing.assert_array_equal(result.iloc[:, 3], [-2.0, -3.0])
    np.testing.assert_array_equal(result.iloc[:, 4], [3.0, 10.0])


def test_cross_augmentation_maps_overflow_to_missing():
    X = pd.DataFrame({"a": [1e308], "b": [1e308]})
    result = runner._augment_crosses(X, [(0, 1, "prod")])
    assert np.isnan(result.iloc[0, -1])


def test_selection_split_is_deterministic_and_target_free():
    first = runner._selection_split(10_000)
    second = runner._selection_split(10_000)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert len(first[0]) == 8_000
    assert len(first[1]) == 2_000
    assert not set(first[0]) & set(first[1])


def test_timed_prediction_requires_seconds_scale_block(monkeypatch):
    monkeypatch.setattr(runner, "PREDICTION_BLOCK_SECONDS", 0.025)
    calls = 0

    def predict():
        nonlocal calls
        calls += 1
        time.sleep(0.001)
        return np.array([1.0, 2.0])

    prediction, timing = runner._timed_predict(predict)
    np.testing.assert_array_equal(prediction, [1.0, 2.0])
    assert timing["call_count"] >= runner.PREDICTION_MIN_CALLS
    assert timing["total_seconds"] >= runner.PREDICTION_BLOCK_SECONDS
    assert calls == timing["call_count"] + 1


def test_worker_spool_round_trip_is_create_only_and_hash_bound(tmp_path):
    binding = {
        "schema_version": 1,
        "runner_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
        "registry_file_sha256": "c" * 64,
        "registry_canonical_sha256": "d" * 64,
        "darkofit_head": "e" * 40,
        "chimeraboost_head": "f" * 40,
        "configs": list(runner.CONFIGS),
        "folds": list(runner.FOLDS),
    }
    path = runner._spool_path(tmp_path, 123, runner.CONTROL)
    result = {"task_id": 123, "config": runner.CONTROL, "folds": []}
    created, created_hash = runner._create_spool(
        path, binding, 123, runner.CONTROL, result
    )
    loaded, loaded_hash = runner._load_spool(
        path, binding, 123, runner.CONTROL
    )
    assert created == loaded == result
    assert created_hash == loaded_hash

    with pytest.raises(RuntimeError, match="binding"):
        runner._load_spool(
            path,
            {**binding, "darkofit_head": "0" * 40},
            123,
            runner.CONTROL,
        )


def test_worker_spool_rejects_result_corruption(tmp_path):
    binding = {
        "schema_version": 1,
        "runner_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
        "registry_file_sha256": "c" * 64,
        "registry_canonical_sha256": "d" * 64,
        "darkofit_head": "e" * 40,
        "chimeraboost_head": "f" * 40,
        "configs": list(runner.CONFIGS),
        "folds": list(runner.FOLDS),
    }
    path = runner._spool_path(tmp_path, 123, runner.CONTROL)
    result = {"task_id": 123, "config": runner.CONTROL, "folds": []}
    runner._create_spool(path, binding, 123, runner.CONTROL, result)
    payload = json.loads(path.read_text())
    payload["result"]["task_id"] = 124
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="hash"):
        runner._load_spool(path, binding, 123, runner.CONTROL)


def _raw_fixture(ratio=0.99, *, engaged=True):
    registry, _rows = runner._registry()
    results = []
    spool_records = []
    for task_row in registry["tasks"]:
        task_id = int(task_row["task_id"])
        for config in runner.CONFIGS:
            if config == runner.CONTROL:
                rmse = 1.0
                digest = hashlib.sha256(
                    f"control-{task_id}".encode()
                ).hexdigest()
            elif config == runner.COMPOSITE:
                rmse = ratio
                digest = (
                    hashlib.sha256(
                        f"composite-{task_id}".encode()
                    ).hexdigest()
                    if engaged
                    else hashlib.sha256(
                        f"control-{task_id}".encode()
                    ).hexdigest()
                )
                if not engaged:
                    rmse = 1.0
            elif config == runner.CHIMERA:
                rmse = 1.01
                digest = hashlib.sha256(
                    f"chimera-{task_id}".encode()
                ).hexdigest()
            else:
                rmse = 1.02
                digest = hashlib.sha256(
                    f"catboost-{task_id}".encode()
                ).hexdigest()
            folds = []
            for fold in runner.FOLDS:
                split = runner._expected_split(task_row, fold)
                folds.append(
                    {
                        "fold": fold,
                        "train_rows": split["train_size"],
                        "test_rows": split["test_size"],
                        "train_index_sha256": split[
                            "train_index_sha256"
                        ],
                        "test_index_sha256": split[
                            "test_index_sha256"
                        ],
                        "rmse": rmse,
                        "fit_seconds": (
                            2.0 if config == runner.COMPOSITE else 1.0
                        ),
                        "prediction_sha256": digest,
                        "prediction_timing": {
                            "call_count": 5,
                            "total_seconds": 0.25,
                            "per_call_median_seconds": (
                                0.012
                                if config == runner.COMPOSITE
                                else 0.01
                            ),
                        },
                        "metadata": (
                            {"engaged": engaged}
                            if config == runner.COMPOSITE
                            else {}
                        ),
                    }
                )
            behavior = {
                "task_id": task_id,
                "config": config,
                "folds": [
                    {
                        "fold": fold["fold"],
                        "rmse": fold["rmse"],
                        "prediction_sha256": fold["prediction_sha256"],
                        "metadata": fold["metadata"],
                    }
                    for fold in folds
                ],
            }
            worker_result = {
                "task_id": task_id,
                "dataset_id": task_row["dataset_id"],
                "dataset_name": task_row["dataset_name"],
                "lineage_cluster": task_row["lineage_cluster"],
                "stratum": task_row["stratum"],
                "ordinal_features": task_row["ordinal_features"],
                "config": config,
                "folds": folds,
                "fold_count": len(folds),
                "warmup_seconds": 0.1,
                "summed_fit_seconds": sum(
                    fold["fit_seconds"] for fold in folds
                ),
                "peak_rss_bytes": (
                    200 if config == runner.COMPOSITE else 100
                ),
                "behavior_fingerprint_sha256": analyzer._json_sha256(
                    behavior
                ),
            }
            results.append(worker_result)
            spool_records.append(
                {
                    "task_id": task_id,
                    "config": config,
                    "filename": f"task-{task_id}--{config}.json",
                    "sha256": hashlib.sha256(
                        f"{task_id}/{config}".encode()
                    ).hexdigest(),
                    "resumed": False,
                }
            )
    binding = {
        "schema_version": 1,
        "runner_sha256": runner._sha256(Path(runner.__file__).resolve()),
        "protocol_sha256": runner._sha256(runner.PROTOCOL),
        "registry_file_sha256": runner.EXPECTED_REGISTRY_FILE_SHA256,
        "registry_canonical_sha256": (
            runner.EXPECTED_REGISTRY_CANONICAL_SHA256
        ),
        "darkofit_head": "darkofit-head",
        "chimeraboost_head": runner.EXPECTED_CHIMERA_HEAD,
        "configs": list(runner.CONFIGS),
        "folds": list(runner.FOLDS),
    }
    raw = {
        "name": "darkofit_t5_composite_confirmation_raw_v1",
        "protocol": {
            "sha256": runner._sha256(runner.PROTOCOL),
            "runner_sha256": runner._sha256(
                Path(runner.__file__).resolve()
            ),
            "registry_file_sha256": runner.EXPECTED_REGISTRY_FILE_SHA256,
            "registry_canonical_sha256": (
                runner.EXPECTED_REGISTRY_CANONICAL_SHA256
            ),
            "configs": list(runner.CONFIGS),
            "folds": list(runner.FOLDS),
            "task_count": 25,
            "coordinate_count": 75,
            "worker_count": 100,
            "lockbox_data_used": False,
            "task_drop_allowed": False,
            "task_imputation_allowed": False,
        },
        "sources": {
            "darkofit": {"head": "darkofit-head", "clean": True},
            "chimeraboost": {
                "head": runner.EXPECTED_CHIMERA_HEAD,
                "clean": True,
            },
        },
        "spool": {
            "binding": binding,
            "record_count": 100,
            "resumed_record_count": 0,
            "records": spool_records,
        },
        "results": results,
        "outcomes_scored": True,
        "analysis_performed": False,
        "default_promotion_authorized": False,
        "lockbox_data_used": False,
    }
    raw["raw_artifact_sha256"] = analyzer._json_sha256(raw)
    return raw


def test_analyzer_passes_broad_guarded_improvement():
    summary = analyzer.analyze(_raw_fixture())
    assert summary["passes_all_gates"] is True
    assert summary["decision"] == "promote_t5_composite_automatic_policy"
    assert summary["primary"]["equal_dataset_geomean_ratio"] == pytest.approx(
        0.99
    )
    assert summary["selection"]["engaged_count"] == 75
    assert summary["hierarchical_bootstrap"]["replicates"] == 100_000


def test_analyzer_accepts_exact_declines_but_null_effect_fails_quality():
    summary = analyzer.analyze(_raw_fixture(engaged=False))
    assert summary["selection"]["declined_count"] == 75
    assert summary["selection"]["exact_declines_verified"] is True
    assert summary["passes_all_gates"] is False
    assert summary["decision"] == "close_t5_composite_candidate"


def test_analyzer_rejects_nonexact_decline():
    raw = _raw_fixture(engaged=False)
    composite = next(
        row for row in raw["results"] if row["config"] == runner.COMPOSITE
    )
    composite["folds"][0]["prediction_sha256"] = "different"
    composite["behavior_fingerprint_sha256"] = analyzer._json_sha256(
        {
            "task_id": composite["task_id"],
            "config": composite["config"],
            "folds": [
                {
                    "fold": fold["fold"],
                    "rmse": fold["rmse"],
                    "prediction_sha256": fold["prediction_sha256"],
                    "metadata": fold["metadata"],
                }
                for fold in composite["folds"]
            ],
        }
    )
    raw["raw_artifact_sha256"] = analyzer._json_sha256(
        {k: v for k, v in raw.items() if k != "raw_artifact_sha256"}
    )
    with pytest.raises(RuntimeError, match="exact decline"):
        analyzer.analyze(raw)


def test_analyzer_rejects_hash_mismatch():
    raw = _raw_fixture()
    corrupted = copy.deepcopy(raw)
    corrupted["results"][0]["folds"][0]["rmse"] = 7.0
    with pytest.raises(RuntimeError, match="hash"):
        analyzer.analyze(corrupted)
