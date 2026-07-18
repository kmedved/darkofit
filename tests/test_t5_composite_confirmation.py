import copy
import hashlib
import json
import os
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


def test_analysis_pair_publish_rolls_back_first_output(
    tmp_path, monkeypatch
):
    summary = tmp_path / "summary.json"
    markdown = tmp_path / "result.md"
    original = analyzer._atomic_create

    def fail_markdown(path, text, **kwargs):
        if path == markdown:
            raise OSError("injected second-output failure")
        return original(path, text, **kwargs)

    monkeypatch.setattr(analyzer, "_atomic_create", fail_markdown)
    with pytest.raises(OSError, match="second-output"):
        analyzer._atomic_create_pair(
            summary, "{}\n", markdown, "# result\n"
        )
    assert not summary.exists()
    assert not markdown.exists()


def test_t5_analysis_cli_defaults_are_repository_anchored(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    args = analyzer.parse_args([])
    assert args.input == analyzer.DEFAULT_INPUT
    assert args.output == analyzer.DEFAULT_OUTPUT
    assert args.markdown == analyzer.DEFAULT_MARKDOWN


def test_analysis_publish_rolls_back_if_temp_cleanup_fails(
    tmp_path, monkeypatch
):
    output = tmp_path / "summary.json"
    original = Path.unlink

    def fail_temporary(path, *args, **kwargs):
        if path.suffix == ".tmp":
            raise OSError("injected temp cleanup failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_temporary)
    with pytest.raises(OSError, match="temp cleanup"):
        analyzer._atomic_create(output, "{}\n")
    assert not output.exists()


def test_analysis_publish_removes_only_new_directories_on_failure(
    tmp_path, monkeypatch
):
    created_root = tmp_path / "new" / "nested"
    output = created_root / "summary.json"

    def fail_publish(_source, _destination):
        raise OSError("injected publish failure")

    monkeypatch.setattr(analyzer.os, "link", fail_publish)
    with pytest.raises(OSError, match="publish failure"):
        analyzer._atomic_create(output, "{}\n")
    assert not (tmp_path / "new").exists()
    assert tmp_path.exists()


def test_analysis_publish_rejects_substituted_temporary_inode(
    tmp_path, monkeypatch
):
    output = tmp_path / "summary.json"
    original = analyzer.os.link
    original_fdopen = analyzer.os.fdopen
    foreign_temporaries = []
    handles = []

    def track_fdopen(*args, **kwargs):
        handle = original_fdopen(*args, **kwargs)
        handles.append(handle)
        return handle

    def substitute_temporary(source, destination):
        source = Path(source)
        source.unlink()
        source.write_text("foreign\n", encoding="utf-8")
        foreign_temporaries.append(source)
        original(source, destination)

    monkeypatch.setattr(analyzer.os, "fdopen", track_fdopen)
    monkeypatch.setattr(analyzer.os, "link", substitute_temporary)
    with pytest.raises(RuntimeError, match="publish identity changed"):
        analyzer._atomic_create(output, "ours\n")
    assert output.read_text(encoding="utf-8") == "foreign\n"
    assert len(foreign_temporaries) == 1
    assert foreign_temporaries[0].read_text(encoding="utf-8") == "foreign\n"
    assert os.path.samefile(foreign_temporaries[0], output)
    assert handles and all(handle.closed for handle in handles)


def test_analysis_pair_rollback_preserves_replacement_inode(
    tmp_path, monkeypatch
):
    summary = tmp_path / "summary.json"
    markdown = tmp_path / "result.md"
    original = analyzer._atomic_create
    original_fdopen = analyzer.os.fdopen
    handles = []

    def track_fdopen(*args, **kwargs):
        handle = original_fdopen(*args, **kwargs)
        handles.append(handle)
        return handle

    def replace_then_fail(path, text, **kwargs):
        if path == markdown:
            summary.unlink()
            summary.write_text("other writer\n", encoding="utf-8")
            raise OSError("injected second-output failure")
        return original(path, text, **kwargs)

    monkeypatch.setattr(analyzer.os, "fdopen", track_fdopen)
    monkeypatch.setattr(analyzer, "_atomic_create", replace_then_fail)
    with pytest.raises(OSError, match="second-output"):
        analyzer._atomic_create_pair(
            summary, "ours\n", markdown, "# result\n"
        )
    assert summary.read_text(encoding="utf-8") == "other writer\n"
    assert not markdown.exists()
    assert handles and all(handle.closed for handle in handles)


def _core_fit_metadata(rounds=10):
    return {
        "iterations_requested": rounds,
        "iterations_attempted": rounds,
        "rounds_completed": rounds,
        "rounds_retained": rounds,
        "stop_reason": "iteration_limit",
        "phase_seconds": {"tree_build": 0.01},
    }


def _fit_metadata():
    return {
        "best_iteration": 10,
        "fitted_tree_count": 10,
        "resolved_learning_rate": 0.1,
        "requested_tree_mode": "auto",
        "selected_tree_mode": "lightgbm",
        "selected_lane": "boosting",
        "linear_residual_active": False,
        "linear_leaves_active": False,
        "linear_leaves": {
            "requested": False,
            "active": False,
            "inactive_reason": "disabled",
            "min_samples": 1_000,
            "linear_lambda": 1.0,
            "numeric_feature_count": 0,
            "linear_tree_count": 0,
            "linear_leaf_count": 0,
        },
        "resolved_thread_count": runner.THREADS_PER_WORKER,
        "refit": False,
        "refit_strategy": None,
        "final_fit": _core_fit_metadata(),
        "selection_fit": None,
        "selection_early_stopping_rounds": None,
        "final_early_stopping_rounds": None,
    }


def _product_default_fit_metadata():
    return {
        "best_iteration": 1_000,
        "fitted_tree_count": 1_000,
        "resolved_learning_rate": 0.1,
        "requested_tree_mode": "catboost",
        "selected_tree_mode": "catboost",
        "selected_lane": "boosting",
        "linear_residual_active": False,
        "linear_leaves_active": False,
        "linear_leaves": {
            "requested": False,
            "active": False,
            "inactive_reason": "disabled",
            "min_samples": 1_000,
            "linear_lambda": 1.0,
            "numeric_feature_count": 0,
            "linear_tree_count": 0,
            "linear_leaf_count": 0,
        },
        "resolved_thread_count": runner.THREADS_PER_WORKER,
        "refit": False,
        "refit_strategy": None,
        "final_fit": _core_fit_metadata(1_000),
        "selection_fit": None,
        "selection_early_stopping_rounds": None,
        "final_early_stopping_rounds": None,
    }


def _fold_metadata(config, train_rows, engaged):
    if config == runner.CONTROL:
        return {
            "kind": runner.CONTROL,
            "engaged": False,
            "selected_configuration": "product_default",
            "final_fit": _product_default_fit_metadata(),
        }
    if config == runner.COMPOSITE and train_rows < runner.SIZE_GATE:
        return {
            "kind": runner.COMPOSITE,
            "engaged": False,
            "selected_configuration": "product_default",
            "final_fit": _product_default_fit_metadata(),
            "decline_reason": "below_size_gate",
            "size_gate": runner.SIZE_GATE,
            "total_selection_fit_seconds": 0.0,
        }
    if config == runner.COMPOSITE:
        challenger = 0.9 if engaged else 1.0
        final_fit = (
            _fit_metadata()
            if engaged
            else _product_default_fit_metadata()
        )
        if engaged:
            final_fit["requested_tree_mode"] = "lightgbm"
        selection_fits = [
            {
                "name": "control_audition",
                "validation_rmse": 1.0,
                "fit_seconds": 0.1,
                "fit_metadata": _fit_metadata(),
                "validation": {"source": "explicit_eval_set"},
            },
            {
                "name": "challenger_auto",
                "validation_rmse": challenger,
                "fit_seconds": 0.1,
                "fit_metadata": _fit_metadata(),
                "validation": {"source": "explicit_eval_set"},
            },
        ]
        return {
            "kind": runner.COMPOSITE,
            "engaged": engaged,
            "decline_reason": None if engaged else "outer_validation_guard",
            "size_gate": runner.SIZE_GATE,
            "split": runner._selection_split(train_rows)[2],
            "outer_guard_ratio": runner.OUTER_GUARD_RATIO,
            "cross_guard_ratio": runner.CROSS_GUARD_RATIO,
            "selection_rounds": runner.SELECTION_ROUNDS,
            "control_validation_rmse": 1.0,
            "challenger_validation_rmse": challenger,
            "relative_challenger_validation_ratio": challenger,
            "selected_configuration": (
                "challenger" if engaged else "product_default"
            ),
            "selected_tree_mode": "lightgbm",
            "selected_linear_leaves": False,
            "selected_crosses": False,
            "selected_cross_pairs": [],
            "selected_cross_pair_count": 0,
            "selected_best_iteration": 10,
            "selected_resolved_learning_rate": 0.1,
            "selection_fits": selection_fits,
            "total_selection_fit_seconds": 0.2,
            "final_transform_seconds": 0.0,
            "final_fit_seconds": 1.8,
            "final_fit": final_fit,
        }
    if config == runner.CHIMERA:
        return {
            "kind": runner.CHIMERA,
            "fitted_tree_count": 10,
            "resolved_learning_rate": 0.1,
            "linear_leaves_selected": False,
            "cross_features_selected": False,
            "cross_pair_count": 0,
        }
    return {
        "kind": runner.CATBOOST,
        "fitted_tree_count": 10,
        "best_iteration": -1,
    }


def test_composite_metadata_accepts_runner_cross_pair_declarations():
    row = {
        "name": "challenger_crossed",
        "validation_rmse": 0.9,
        "fit_seconds": 0.1,
        "fit_metadata": _fit_metadata(),
        "validation": {"source": "explicit_eval_set"},
        "pair_count": 1,
        "pairs": [[0, 1, "diff"]],
        "transform_seconds": 0.01,
    }
    assert analyzer._valid_selection_fit(row)
    row["pairs"] = [[0, 1]]
    assert not analyzer._valid_selection_fit(row)


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
                fold_engaged = (
                    config == runner.COMPOSITE
                    and engaged
                    and split["train_size"] >= runner.SIZE_GATE
                )
                fold_rmse = (
                    rmse
                    if config != runner.COMPOSITE or fold_engaged
                    else 1.0
                )
                fold_digest = (
                    digest
                    if config != runner.COMPOSITE or fold_engaged
                    else hashlib.sha256(
                        f"control-{task_id}".encode()
                    ).hexdigest()
                )
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
                        "rmse": fold_rmse,
                        "fit_seconds": (
                            2.0 if config == runner.COMPOSITE else 1.0
                        ),
                        "prediction_sha256": fold_digest,
                        "prediction_timing": {
                            "call_count": 5,
                            "total_seconds": 0.25,
                            "minimum_block_seconds": 0.25,
                            "per_call_min_seconds": 0.009,
                            "per_call_median_seconds": (
                                0.012
                                if config == runner.COMPOSITE
                                else 0.01
                            ),
                            "per_call_max_seconds": 0.02,
                        },
                        "metadata": _fold_metadata(
                            config,
                            split["train_size"],
                            fold_engaged,
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
                "categorical_feature_indices": [
                    index
                    for index, column in enumerate(
                        task_row["task_record"]["fingerprint"]["columns"]
                    )
                    if column["dtype_family"] != "numeric"
                ],
                "ordinal_features": task_row["ordinal_features"],
                "config": config,
                "folds": folds,
                "fold_count": len(folds),
                "warmup_seconds": 0.1,
                "wall_seconds": sum(
                    fold["fit_seconds"]
                    + fold["prediction_timing"]["total_seconds"]
                    for fold in folds
                ),
                "summed_fit_seconds": sum(
                    fold["fit_seconds"] for fold in folds
                ),
                "summed_prediction_block_seconds": sum(
                    fold["prediction_timing"]["total_seconds"]
                    for fold in folds
                ),
                "peak_rss_bytes": (
                    200 if config == runner.COMPOSITE else 100
                ),
                "worker_stdout": None,
                "worker_stderr": None,
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
        "darkofit_head": analyzer.FROZEN_DARKOFIT_HEAD,
        "chimeraboost_head": runner.EXPECTED_CHIMERA_HEAD,
        "configs": list(runner.CONFIGS),
        "folds": list(runner.FOLDS),
    }
    raw = {
        "schema_version": 1,
        "name": "darkofit_t5_composite_confirmation_raw_v1",
        "created_at": "2026-07-17T00:00:00+00:00",
        "protocol": {
            "path": str(runner.PROTOCOL.relative_to(analyzer.ROOT)),
            "sha256": runner._sha256(runner.PROTOCOL),
            "runner_path": str(
                Path(runner.__file__).resolve().relative_to(analyzer.ROOT)
            ),
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
            "threads_per_worker": runner.THREADS_PER_WORKER,
            "concurrent_workers": runner.CONCURRENT_WORKERS,
            "size_gate": runner.SIZE_GATE,
            "validation_fraction": runner.VALIDATION_FRACTION,
            "outer_guard_ratio": runner.OUTER_GUARD_RATIO,
            "cross_guard_ratio": runner.CROSS_GUARD_RATIO,
            "selection_rounds": runner.SELECTION_ROUNDS,
            "prediction_block_seconds": runner.PREDICTION_BLOCK_SECONDS,
            "lockbox_data_used": False,
            "task_drop_allowed": False,
            "task_imputation_allowed": False,
        },
        "sources": {
            "darkofit": {
                "path": str(analyzer.ROOT),
                "head": analyzer.FROZEN_DARKOFIT_HEAD,
                "branch": "fixture",
                "clean": True,
                "status": [],
                "describe": None,
                "remotes": {},
                "tracked_main_refs": {},
            },
            "chimeraboost": {
                "path": str(runner.CHIMERA_ROOT),
                "head": runner.EXPECTED_CHIMERA_HEAD,
                "branch": "fixture",
                "clean": True,
                "status": [],
                "describe": None,
                "remotes": {},
                "tracked_main_refs": {},
            },
        },
        "environment": {
            "python": "Python fixture",
            "machine": {
                "platform": "fixture",
                "machine": "fixture",
                "cpu_brand": None,
                "logical_cpu_count": 1,
                "python": "Python fixture",
                "python_executable": "/fixture/python",
            },
            "dependencies": {
                "numpy": "fixture",
                "pandas": "fixture",
                "scikit-learn": "fixture",
                "joblib": "fixture",
                "numba": "fixture",
                "catboost": "fixture",
            },
        },
        "spool": {
            "binding": binding,
            "record_count": 100,
            "resumed_record_count": 0,
            "records": spool_records,
        },
        "registry_power_probability": registry["power_analysis"][
            "pass_probability"
        ],
        "results": results,
        "outcomes_scored": True,
        "analysis_performed": False,
        "default_promotion_authorized": False,
        "lockbox_data_used": False,
    }
    by_key = {
        (row["task_id"], row["config"]): row for row in results
    }
    for record in spool_records:
        key = (record["task_id"], record["config"])
        result = by_key[key]
        payload = {
            "schema_version": 1,
            "name": "darkofit_t5_composite_worker_spool_v1",
            "binding": binding,
            "task_id": key[0],
            "config": key[1],
            "result_sha256": analyzer._json_sha256(result),
            "result": result,
        }
        record["sha256"] = analyzer._json_sha256(payload)
    raw["raw_artifact_sha256"] = analyzer._json_sha256(raw)
    return raw


def _rehash_raw(raw):
    raw["raw_artifact_sha256"] = analyzer._json_sha256(
        {
            key: value
            for key, value in raw.items()
            if key != "raw_artifact_sha256"
        }
    )


def _rebind_worker(raw, worker):
    worker["behavior_fingerprint_sha256"] = analyzer._json_sha256(
        {
            "task_id": worker["task_id"],
            "config": worker["config"],
            "folds": [
                {
                    "fold": fold["fold"],
                    "rmse": fold["rmse"],
                    "prediction_sha256": fold["prediction_sha256"],
                    "metadata": fold["metadata"],
                }
                for fold in worker["folds"]
            ],
        }
    )
    record = next(
        row
        for row in raw["spool"]["records"]
        if row["task_id"] == worker["task_id"]
        and row["config"] == worker["config"]
    )
    record["sha256"] = analyzer._json_sha256(
        {
            "schema_version": 1,
            "name": "darkofit_t5_composite_worker_spool_v1",
            "binding": raw["spool"]["binding"],
            "task_id": worker["task_id"],
            "config": worker["config"],
            "result_sha256": analyzer._json_sha256(worker),
            "result": worker,
        }
    )
    _rehash_raw(raw)


def test_analyzer_passes_broad_guarded_improvement():
    summary = analyzer.analyze(_raw_fixture())
    assert summary["passes_all_gates"] is True
    assert summary["decision"] == "promote_t5_composite_automatic_policy"
    assert summary["primary"]["equal_dataset_geomean_ratio"] == pytest.approx(
        0.99 ** (16 / 25)
    )
    assert summary["selection"]["engaged_count"] == 48
    assert summary["hierarchical_bootstrap"]["replicates"] == 100_000
    stored_order = json.loads(json.dumps(summary, sort_keys=True))
    assert analyzer._markdown(summary) == analyzer._markdown(stored_order)


def test_analyzer_accepts_exact_declines_but_null_effect_fails_quality():
    summary = analyzer.analyze(_raw_fixture(engaged=False))
    assert summary["selection"]["declined_count"] == 75
    assert summary["selection"]["exact_declines_verified"] is True
    assert summary["passes_all_gates"] is False
    assert summary["decision"] == "close_t5_composite_candidate"


def test_analyzer_metrics_are_independent_of_valid_result_row_order():
    raw = _raw_fixture()
    reordered = copy.deepcopy(raw)
    reordered["results"].reverse()
    _rehash_raw(reordered)
    baseline = analyzer.analyze(raw)
    observed = analyzer.analyze(reordered)
    for field in (
        "primary",
        "hierarchical_bootstrap",
        "leave_one_out",
        "least_favorable_leave_one_out",
        "cost",
        "competitive_comparisons",
        "selection",
        "gates",
        "passes_all_gates",
        "decision",
    ):
        assert observed[field] == baseline[field]


def test_analyzer_rejects_nonexact_decline():
    raw = _raw_fixture(engaged=False)
    composite = next(
        row for row in raw["results"] if row["config"] == runner.COMPOSITE
    )
    composite["folds"][0]["prediction_sha256"] = "0" * 64
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


def test_analyzer_requires_complete_v1_schema():
    raw = _raw_fixture()
    raw.pop("environment")
    _rehash_raw(raw)
    with pytest.raises(RuntimeError, match="artifact name"):
        analyzer.analyze(raw)


def test_analyzer_requires_complete_typed_environment_and_source_schema():
    raw = _raw_fixture()
    raw["environment"]["machine"]["logical_cpu_count"] = True
    _rehash_raw(raw)
    with pytest.raises(RuntimeError, match="execution environment"):
        analyzer.analyze(raw)

    raw = _raw_fixture()
    raw["sources"]["darkofit"]["remotes"]["origin"] = 7
    _rehash_raw(raw)
    with pytest.raises(RuntimeError, match="source state"):
        analyzer.analyze(raw)


def test_analyzer_rejects_nonfrozen_darkofit_source():
    raw = _raw_fixture()
    raw["sources"]["darkofit"]["head"] = "0" * 40
    raw["spool"]["binding"]["darkofit_head"] = "0" * 40
    raw["raw_artifact_sha256"] = analyzer._json_sha256(
        {key: value for key, value in raw.items() if key != "raw_artifact_sha256"}
    )
    with pytest.raises(RuntimeError, match="source state"):
        analyzer.analyze(raw)


def test_analyzer_rejects_nonboolean_campaign_state():
    raw = _raw_fixture()
    raw["outcomes_scored"] = 1
    _rehash_raw(raw)
    with pytest.raises(RuntimeError, match="artifact state"):
        analyzer.analyze(raw)

    raw = _raw_fixture()
    raw["protocol"]["lockbox_data_used"] = 0
    _rehash_raw(raw)
    with pytest.raises(RuntimeError, match="protocol"):
        analyzer.analyze(raw)

    raw = _raw_fixture()
    raw["sources"]["darkofit"]["clean"] = 1
    _rehash_raw(raw)
    with pytest.raises(RuntimeError, match="source state"):
        analyzer.analyze(raw)


def test_analyzer_rejects_coerced_resource_and_timing_scalars():
    raw = _raw_fixture()
    raw["results"][0]["peak_rss_bytes"] = "100"
    _rehash_raw(raw)
    with pytest.raises(RuntimeError, match="resource record"):
        analyzer.analyze(raw)

    raw = _raw_fixture()
    raw["results"][0]["folds"][0]["prediction_timing"][
        "total_seconds"
    ] = "nan"
    _rehash_raw(raw)
    with pytest.raises(RuntimeError, match="prediction block"):
        analyzer.analyze(raw)

    raw = _raw_fixture()
    raw["spool"]["records"][0]["resumed"] = 0
    _rehash_raw(raw)
    with pytest.raises(RuntimeError, match="spool matrix"):
        analyzer.analyze(raw)


def test_analyzer_rejects_integer_aliases_for_float_measurements():
    raw = _raw_fixture()
    raw["results"][0]["warmup_seconds"] = 1
    _rehash_raw(raw)
    with pytest.raises(RuntimeError, match="resource record"):
        analyzer.analyze(raw)


def test_analyzer_binds_exact_categorical_feature_indices():
    raw = _raw_fixture()
    worker = next(
        row
        for row in raw["results"]
        if len(row["categorical_feature_indices"]) >= 2
    )
    original = worker["categorical_feature_indices"]
    feature_count = next(
        row
        for row in runner._registry()[0]["tasks"]
        if row["task_id"] == worker["task_id"]
    )["task_record"]["fingerprint"]["n_features"]
    replacement = next(
        index for index in range(feature_count) if index not in original
    )
    worker["categorical_feature_indices"] = [
        *original[:-1],
        replacement,
    ]
    _rehash_raw(raw)
    with pytest.raises(RuntimeError, match="resource record"):
        analyzer.analyze(raw)


def test_analyzer_rejects_impossible_prediction_timing():
    raw = _raw_fixture()
    timing = raw["results"][0]["folds"][0]["prediction_timing"]
    timing["per_call_median_seconds"] = 0.1
    timing["per_call_max_seconds"] = 0.1
    timing["total_seconds"] = 0.25
    _rehash_raw(raw)
    with pytest.raises(RuntimeError, match="prediction block"):
        analyzer.analyze(raw)


def test_t5_raw_json_rejects_duplicate_keys_and_nonfinite_constants():
    with pytest.raises(RuntimeError, match="invalid T5 raw artifact JSON"):
        analyzer._json_loads('{"schema_version":1,"schema_version":2}', "T5 raw artifact")
    with pytest.raises(RuntimeError, match="invalid T5 raw artifact JSON"):
        analyzer._json_loads('{"rmse":NaN}', "T5 raw artifact")
    with pytest.raises(RuntimeError, match="invalid T5 raw artifact JSON"):
        analyzer._json_loads('{"rmse":1e9999}', "T5 raw artifact")
    with pytest.raises(RuntimeError, match="invalid T5 raw artifact JSON"):
        analyzer._json_loads(
            '{"task_id":9223372036854775808}',
            "T5 raw artifact",
        )
    with pytest.raises(RuntimeError, match="invalid T5 raw artifact JSON"):
        analyzer._json_loads(
            '{"schema_version":1}'.encode("utf-16"),
            "T5 raw artifact",
        )


def test_analyzer_rejects_truthy_string_engagement():
    raw = _raw_fixture()
    composite = next(
        row for row in raw["results"] if row["config"] == runner.COMPOSITE
    )
    composite["folds"][0]["metadata"]["engaged"] = "false"
    _rebind_worker(raw, composite)
    with pytest.raises(RuntimeError, match="fold metadata"):
        analyzer.analyze(raw)


def test_analyzer_binds_composite_selection_to_selected_fit():
    raw = _raw_fixture()
    composite = next(
        row for row in raw["results"] if row["config"] == runner.COMPOSITE
    )
    eligible = next(
        fold
        for fold in composite["folds"]
        if fold["train_rows"] >= runner.SIZE_GATE
    )
    eligible["metadata"]["selected_best_iteration"] -= 1
    _rebind_worker(raw, composite)
    with pytest.raises(RuntimeError, match="fold metadata"):
        analyzer.analyze(raw)


def test_analyzer_binds_engaged_final_fit_to_selected_route():
    raw = _raw_fixture()
    composite = next(
        row for row in raw["results"] if row["config"] == runner.COMPOSITE
    )
    eligible = next(
        fold
        for fold in composite["folds"]
        if fold["train_rows"] >= runner.SIZE_GATE
    )
    eligible["metadata"]["final_fit"][
        "selected_tree_mode"
    ] = "forged_incompatible_mode"
    _rebind_worker(raw, composite)
    with pytest.raises(RuntimeError, match="fold metadata"):
        analyzer.analyze(raw)


def test_analyzer_binds_declined_final_fit_to_product_default_route():
    raw = _raw_fixture(engaged=False)
    composite = next(
        row for row in raw["results"] if row["config"] == runner.COMPOSITE
    )
    eligible = next(
        fold
        for fold in composite["folds"]
        if fold["train_rows"] >= runner.SIZE_GATE
    )
    final_fit = eligible["metadata"]["final_fit"]
    final_fit["requested_tree_mode"] = "forged_mode"
    final_fit["selected_tree_mode"] = "forged_mode"
    final_fit["selected_lane"] = "linear_leaves"
    final_fit["linear_leaves_active"] = True
    _rebind_worker(raw, composite)
    with pytest.raises(RuntimeError, match="fold metadata"):
        analyzer.analyze(raw)


@pytest.mark.parametrize("below_size_gate", [False, True])
def test_analyzer_binds_decline_learning_rate_to_paired_default(
    below_size_gate,
):
    raw = _raw_fixture(engaged=False)
    composite, declined = next(
        (row, fold)
        for row in raw["results"]
        if row["config"] == runner.COMPOSITE
        for fold in row["folds"]
        if (fold["train_rows"] < runner.SIZE_GATE) is below_size_gate
    )
    declined["metadata"]["final_fit"]["resolved_learning_rate"] = 0.333
    _rebind_worker(raw, composite)
    with pytest.raises(RuntimeError, match="exact decline"):
        analyzer.analyze(raw)


def test_analyzer_rejects_forged_engaged_linear_leaf_metadata():
    raw = _raw_fixture()
    composite = next(
        row for row in raw["results"] if row["config"] == runner.COMPOSITE
    )
    engaged = next(
        fold
        for fold in composite["folds"]
        if fold["metadata"]["engaged"]
    )
    engaged["metadata"]["final_fit"]["linear_leaves"] = {"forged": True}
    _rebind_worker(raw, composite)
    with pytest.raises(RuntimeError, match="fold metadata"):
        analyzer.analyze(raw)


def test_analyzer_binds_nested_linear_leaf_activity():
    raw = _raw_fixture()
    composite = next(
        row for row in raw["results"] if row["config"] == runner.COMPOSITE
    )
    engaged = next(
        fold
        for fold in composite["folds"]
        if fold["metadata"]["engaged"]
    )
    fit = engaged["metadata"]["selection_fits"][0]["fit_metadata"]
    fit["linear_leaves"] = {
        "requested": True,
        "active": True,
        "inactive_reason": None,
        "min_samples": 1_000,
        "linear_lambda": 1.0,
        "numeric_feature_count": 1,
        "linear_tree_count": 10,
        "linear_leaf_count": 10,
    }
    _rebind_worker(raw, composite)
    with pytest.raises(RuntimeError, match="fold metadata"):
        analyzer.analyze(raw)


def test_analyzer_binds_linear_leaf_request_to_audition_route():
    raw = _raw_fixture()
    composite = next(
        row for row in raw["results"] if row["config"] == runner.COMPOSITE
    )
    engaged = next(
        fold
        for fold in composite["folds"]
        if fold["metadata"]["engaged"]
    )
    fit = engaged["metadata"]["selection_fits"][0]["fit_metadata"]
    fit["linear_leaves"] = {
        "requested": True,
        "active": False,
        "inactive_reason": "below_min_samples",
        "min_samples": 1_000,
        "linear_lambda": 1.0,
        "numeric_feature_count": 0,
        "linear_tree_count": 0,
        "linear_leaf_count": 0,
    }
    _rebind_worker(raw, composite)
    with pytest.raises(RuntimeError, match="fold metadata"):
        analyzer.analyze(raw)


def test_analyzer_rejects_out_of_range_cross_pair_matrix():
    raw = _raw_fixture()
    composite = next(
        row for row in raw["results"] if row["config"] == runner.COMPOSITE
    )
    eligible = next(
        fold
        for fold in composite["folds"]
        if fold["train_rows"] >= runner.SIZE_GATE
    )
    metadata = eligible["metadata"]
    crossed = copy.deepcopy(metadata["selection_fits"][1])
    crossed.update(
        {
            "name": "challenger_crossed",
            "validation_rmse": 0.8,
            "pair_count": 1,
            "pairs": [[999, 1000, "diff"]],
            "transform_seconds": 0.01,
        }
    )
    metadata["selection_fits"].append(crossed)
    metadata["challenger_validation_rmse"] = 0.8
    metadata["relative_challenger_validation_ratio"] = 0.8
    metadata["selected_crosses"] = True
    metadata["selected_cross_pairs"] = crossed["pairs"]
    metadata["selected_cross_pair_count"] = 1
    metadata["total_selection_fit_seconds"] = 0.3
    metadata["final_fit_seconds"] = 1.7
    _rebind_worker(raw, composite)
    with pytest.raises(RuntimeError, match="fold metadata"):
        analyzer.analyze(raw)


def test_analyzer_rejects_impossible_linear_audition_route():
    raw = _raw_fixture()
    composite = next(
        row for row in raw["results"] if row["config"] == runner.COMPOSITE
    )
    eligible = next(
        fold
        for fold in composite["folds"]
        if fold["train_rows"] >= runner.SIZE_GATE
    )
    metadata = eligible["metadata"]
    linear = copy.deepcopy(metadata["selection_fits"][1])
    linear.update(
        {
            "name": "challenger_catboost_linear",
            "validation_rmse": 0.95,
        }
    )
    metadata["selection_fits"].append(linear)
    metadata["total_selection_fit_seconds"] = 0.3
    metadata["final_fit_seconds"] = 1.7
    _rebind_worker(raw, composite)
    with pytest.raises(RuntimeError, match="fold metadata"):
        analyzer.analyze(raw)


def test_analyzer_rejects_forged_control_metadata():
    raw = _raw_fixture()
    control = next(
        row for row in raw["results"] if row["config"] == runner.CONTROL
    )
    control["folds"][0]["metadata"]["engaged"] = "false"
    control["folds"][0]["metadata"]["forged"] = 7
    _rebind_worker(raw, control)
    with pytest.raises(RuntimeError, match="fold metadata"):
        analyzer.analyze(raw)


def test_analyzer_rejects_boolean_fold_aliases():
    raw = _raw_fixture()
    raw["protocol"]["folds"] = [False, True, 2]
    raw["spool"]["binding"]["folds"] = [False, True, 2]
    _rehash_raw(raw)
    with pytest.raises(RuntimeError, match="protocol"):
        analyzer.analyze(raw)


def test_analyzer_binds_each_full_result_to_its_spool_digest():
    raw = _raw_fixture()
    raw["results"][0]["wall_seconds"] += 1.0
    _rehash_raw(raw)
    with pytest.raises(RuntimeError, match="spool result"):
        analyzer.analyze(raw)


def test_analyzer_reads_registry_from_one_content_snapshot(monkeypatch):
    raw = _raw_fixture()
    original = Path.read_bytes
    reads = 0

    def tracked(path):
        nonlocal reads
        if path == runner.REGISTRY:
            reads += 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    analyzer.analyze(raw)
    assert reads == 1


def test_analysis_publish_rejects_nested_symlink_before_mkdir(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    output = alias / "nested" / "summary.json"
    with pytest.raises(RuntimeError, match="symlink T5 output directory"):
        analyzer._atomic_create(output, "{}\n")
    assert not (real / "nested").exists()
