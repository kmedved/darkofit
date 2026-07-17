from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks import analyze_basketball_native_ordinal as analyzer
from benchmarks import run_basketball_native_ordinal as runner
from darkofit import DarkoRegressor


def _sha256_values(values) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _fit_metadata(learning_rate: float = 0.052312) -> dict:
    return {
        "best_iteration": 1_000,
        "fitted_tree_count": 1_000,
        "resolved_learning_rate": learning_rate,
        "requested_tree_mode": "catboost",
        "selected_tree_mode": "catboost",
        "selected_lane": "boosting",
        "linear_residual_active": False,
        "linear_leaves_active": False,
        "linear_leaves": {
            "requested": False,
            "active": False,
            "inactive_reason": "disabled",
        },
        "resolved_thread_count": 18,
        "refit": False,
        "refit_strategy": None,
        "final_fit": {
            "iterations_requested": 1_000,
            "iterations_attempted": 1_000,
            "rounds_completed": 1_000,
            "rounds_retained": 1_000,
            "stop_reason": "iteration_limit",
            "phase_seconds": {"boost": 1.0},
        },
        "selection_fit": None,
        "selection_early_stopping_rounds": None,
        "final_early_stopping_rounds": None,
    }


def _model(arm: str, learning_rate: float = 0.052312) -> dict:
    importance = [0.0] * 15
    offsets = [9 * index for index in range(16)]
    return {
        "fit_metadata": _fit_metadata(learning_rate),
        "ordinal_state": (
            {
                "mode": "off",
                "records": [],
                "indices": [],
                "metadata": None,
            }
            if arm == runner.CONTROL
            else {
                "mode": "auto",
                "records": [],
                "indices": [],
                "metadata": analyzer._ordinal_metadata(),
            }
        ),
        "preprocessor": {
            "n_input_features": 15,
            "num_features": list(range(15)),
            "cat_features": [],
            "feature_map": list(range(15)),
            "n_bins": [10] * 15,
            "borders_flat_sha256": "1" * 64,
            "border_offsets": offsets,
            "block_widths": [1],
            "encoder_count": 0,
            "target_stat_blocks": 0,
        },
        "feature_importance": importance,
        "feature_importance_sha256": _sha256_values(importance),
        "archive": {
            "raw_bytes": 100,
            "raw_sha256": "2" * 64,
            "normalized_sha256": "3" * 64,
            "normalized_removed_fields": (
                [
                    "auto_params.ordinal_features",
                    "auto_params.diagnostics.ordinal_features",
                    "wrapper.state.ordinal_features_mode",
                    "wrapper.state.ordinal_features",
                    "timing",
                ]
                if arm == runner.CANDIDATE
                else ["timing"]
            ),
        },
    }


def _prediction(values: np.ndarray, score: float = 0.5) -> dict:
    return {
        "r2": score,
        "prediction_sha256": _sha256_values(values),
        "predictions": np.asarray(values, dtype=np.float64).tolist(),
    }


def _source() -> dict:
    analyzer_sha = hashlib.sha256(Path(analyzer.__file__).read_bytes()).hexdigest()
    return {
        "head": "a" * 40,
        "branch": "main",
        "origin_main": "a" * 40,
        "status": "",
        "package_tree": analyzer.IMPLEMENTATION_PACKAGE_TREE,
        "protocol_sha256": analyzer.EXPECTED_PROTOCOL_SHA256,
        "runner_normalized_sha256": (
            analyzer.EXPECTED_NORMALIZED_RUNNER_SHA256
        ),
        "analyzer_sha256": analyzer_sha,
        "support_sha256": copy.deepcopy(analyzer.EXPECTED_SUPPORT_SHA256),
    }


def _runtime() -> dict:
    return {
        **copy.deepcopy(analyzer.EXPECTED_RUNTIME),
        "python_executable": "/frozen/python",
    }


def _data() -> dict:
    features = [
        "3P",
        "3PA",
        "2P",
        "2PA",
        "FT",
        "FTA",
        "ORB",
        "DRB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PF",
        "PTS",
        "Age",
    ]
    return {
        "raw": {
            "load_source": "cache",
            "bytes": analyzer.EXPECTED_DATA["raw_bytes"],
            "sha256": analyzer.EXPECTED_DATA["raw_sha256"],
        },
        "processed": {
            "train_rows": 5_241,
            "test_rows": 2_409,
            "missing_train_feature_cells": 0,
            "features": features,
            "x_train_sha256": analyzer.EXPECTED_DATA["x_sha256"],
            "y_train_sha256": analyzer.EXPECTED_DATA["y_sha256"],
        },
        "fold_sha256": analyzer.EXPECTED_DATA["fold_sha256"],
        "fold_test_sizes": [525, *([524] * 9)],
        "feature_dtypes": {
            **{name: "float64" for name in features if name != "Age"},
            "Age": "int64",
        },
        "guardrail": {
            "cold_player_mask_sha256": analyzer.EXPECTED_DATA[
                "cold_mask_sha256"
            ],
            "train_rows": 5_241,
            "holdout_rows": 2_409,
            "cold_player_holdout_rows": 585,
            "seen_player_holdout_rows": 1_824,
            "cold_players_absent_from_training": True,
            "player_identifier_used_as_model_feature": False,
        },
    }


def _environment(cache: str) -> dict:
    values = {
        key: None
        for key in (
            "NUMBA_CPU_NAME",
            "NUMBA_CPU_FEATURES",
            "NUMBA_THREADING_LAYER",
            "NUMBA_THREADING_LAYER_PRIORITY",
            "JOBLIB_MULTIPROCESSING",
            "JOBLIB_START_METHOD",
            "LOKY_MAX_CPU_COUNT",
            "LOKY_MAX_DEPTH",
            "LOKY_PICKLER",
            "JOBLIB_TEMP_FOLDER",
        )
    }
    values.update(
        {
            "DARKOFIT_WARMUP": "0",
            "CHIMERABOOST_WARMUP": "0",
            "NUMBA_DISABLE_JIT": "0",
            "ENABLE_IPC": "1",
            "PYTHONHASHSEED": "0",
            "NUMBA_CACHE_DIR": cache,
        }
    )
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TBB_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        values[key] = "18"
    return values


def _worker(
    arm: str,
    *,
    block: int,
    position: int,
    fit_seconds: float,
    held_seconds: float,
    cold_seconds: float,
    rss: float,
) -> dict:
    folds = []
    for fold, indices in enumerate(analyzer._expected_test_indices()):
        values = np.full(len(indices), fold + 0.25, dtype=np.float64)
        folds.append(
            {
                "fold": fold,
                "train_rows": 5_241 - len(indices),
                "test_rows": len(indices),
                "test_indices": indices,
                "fit_seconds": fit_seconds / 11.0,
                "prediction": _prediction(values),
                "model": _model(
                    arm, analyzer.EXPECTED_FOLD_LEARNING_RATES[fold]
                ),
            }
        )
    held = np.full(2_409, 1.25, dtype=np.float64)
    cold = np.full(585, 1.5, dtype=np.float64)
    seen = np.full(1_824, 1.75, dtype=np.float64)
    held_record = {
        **_prediction(held),
        "loop_calls": 200,
        "loop_seconds": held_seconds,
        "loop_last_prediction_sha256": _sha256_values(held),
    }
    cold_record = {
        **_prediction(cold),
        "loop_calls": 500,
        "loop_seconds": cold_seconds,
        "loop_last_prediction_sha256": _sha256_values(cold),
    }
    seen_record = _prediction(seen)
    result = {
        "ok": True,
        "arm": arm,
        "source": _source(),
        "runtime": _runtime(),
        "data": _data(),
        "import_seconds": 0.1,
        "warmup_seconds": 0.2,
        "cache": {
            "before_import": {"file_count": 0, "compiled_file_count": 0},
            "after_import": {"file_count": 0, "compiled_file_count": 0},
            "after_warmup": {"file_count": 2, "compiled_file_count": 2},
            "after_workload": {"file_count": 3, "compiled_file_count": 3},
        },
        "folds": folds,
        "mean_r2": 0.5,
        "fold_scores": [0.5] * 10,
        "guardrail": {
            "full_fit_seconds": fit_seconds / 11.0,
            "model": _model(arm, 0.053192),
            "held": held_record,
            "cold": cold_record,
            "seen": seen_record,
            "scores_from_full_prediction": {
                "overlap_exposed_team_holdout": {
                    "rows": 2_409,
                    "r2": 0.5,
                    "prediction_sha256": _sha256_values(held),
                },
                "cold_player_subset": {
                    "rows": 585,
                    "r2": 0.5,
                    "prediction_sha256": _sha256_values(cold),
                },
                "seen_player_subset": {
                    "rows": 1_824,
                    "r2": 0.5,
                    "prediction_sha256": _sha256_values(seen),
                },
            },
        },
        "total_fit_seconds": fit_seconds,
        "worker_wall_seconds": fit_seconds + held_seconds + cold_seconds,
        "peak_rss_bytes": rss,
        "warnings": [],
        "thread_environment": _environment(f"/cache/{block}/{arm}"),
    }
    result["behavior_fingerprint_sha256"] = analyzer._behavior_fingerprint(
        result
    )
    result.update(
        {
            "worker_returncode": 0,
            "worker_stdout": None,
            "worker_stderr": None,
            "block": block,
            "position": position,
        }
    )
    return result


def _raw(monkeypatch: pytest.MonkeyPatch) -> dict:
    fold_hashes = []
    for fold, indices in enumerate(analyzer._expected_test_indices()):
        fold_hashes.append(
            _sha256_values(np.full(len(indices), fold + 0.25))
        )
    monkeypatch.setattr(
        analyzer,
        "EXPECTED_PREDICTION_SHA256",
        {
            "folds": fold_hashes,
            "held": _sha256_values(np.full(2_409, 1.25)),
            "cold": _sha256_values(np.full(585, 1.5)),
            "seen": _sha256_values(np.full(1_824, 1.75)),
        },
    )
    monkeypatch.setattr(analyzer, "EXPECTED_MEAN_R2", 0.5)
    repeats = []
    for block, order in enumerate(analyzer.BLOCK_ORDERS):
        for position, arm in enumerate(order):
            candidate = arm == analyzer.CANDIDATE
            result = _worker(
                arm,
                block=block,
                position=position,
                fit_seconds=10.1 if candidate else 10.0,
                held_seconds=1.02 if candidate else 1.0,
                cold_seconds=2.04 if candidate else 2.0,
                rss=101.0 if candidate else 100.0,
            )
            repeats.append(
                {
                    "block": block,
                    "position": position,
                    "order": list(order),
                    "arm": arm,
                    "result": result,
                }
            )
    source = _source()
    analyzer_sha = source["analyzer_sha256"]
    return {
        "schema_version": 1,
        "name": "darkofit_basketball_native_ordinal_raw_v1",
        "created_at": "2026-07-17T00:00:00+00:00",
        "protocol": {
            "path": "benchmarks/basketball_native_ordinal_protocol.md",
            "commit": analyzer.PROTOCOL_COMMIT,
            "sha256": analyzer.EXPECTED_PROTOCOL_SHA256,
        },
        "runner": {
            "path": "benchmarks/run_basketball_native_ordinal.py",
            "normalized_sha256": analyzer.EXPECTED_NORMALIZED_RUNNER_SHA256,
        },
        "analyzer": {
            "path": "benchmarks/analyze_basketball_native_ordinal.py",
            "sha256": analyzer_sha,
        },
        "source": source,
        "execution": {
            "threads": 18,
            "timing_blocks": 3,
            "block_orders": [
                list(order) for order in analyzer.BLOCK_ORDERS
            ],
            "held_prediction_calls": 200,
            "cold_prediction_calls": 500,
            "worker_count": 6,
            "partial_resume_supported": False,
        },
        "repeats": repeats,
        "categorical_outcomes_inspected": False,
        "lockbox_touched": False,
    }


def _refresh_behavior(raw: dict, arm: str | None = None) -> None:
    for record in raw["repeats"]:
        if arm is None or record["arm"] == arm:
            result = record["result"]
            result["behavior_fingerprint_sha256"] = (
                analyzer._behavior_fingerprint(result)
            )


def test_protocol_and_runner_bindings_are_current():
    assert analyzer.BLOCK_ORDERS == runner.BLOCK_ORDERS
    assert analyzer.PROTOCOL_COMMIT == runner.PROTOCOL_COMMIT
    assert analyzer.EXPECTED_PROTOCOL_SHA256 == runner.EXPECTED_PROTOCOL_SHA256
    assert analyzer.EXPECTED_NORMALIZED_RUNNER_SHA256 == (
        runner.EXPECTED_NORMALIZED_RUNNER_SHA256
    )
    assert runner._normalized_runner_sha256() == (
        runner.EXPECTED_NORMALIZED_RUNNER_SHA256
    )
    assert hashlib.sha256(runner.PROTOCOL_PATH.read_bytes()).hexdigest() == (
        runner.EXPECTED_PROTOCOL_SHA256
    )


def test_normalized_archive_identity_removes_only_inactive_declaration():
    X = pd.DataFrame(
        {
            "numeric": np.arange(60, dtype=np.float64),
            "integer": np.arange(60, dtype=np.int64),
        }
    )
    y = np.sin(np.arange(60, dtype=np.float64))
    control = DarkoRegressor(
        iterations=2,
        learning_rate=0.1,
        random_state=4,
        thread_count=1,
    ).fit(X, y)
    candidate = DarkoRegressor(
        iterations=2,
        learning_rate=0.1,
        random_state=4,
        thread_count=1,
    ).fit(X, y, ordinal_features="auto")
    control_identity = runner._archive_identity(control, runner.CONTROL)
    candidate_identity = runner._archive_identity(
        candidate, runner.CANDIDATE
    )
    assert control_identity["normalized_sha256"] == (
        candidate_identity["normalized_sha256"]
    )
    assert runner._ordinal_state(control)["mode"] == "off"
    assert runner._ordinal_state(candidate) == {
        "mode": "auto",
        "records": [],
        "indices": [],
        "metadata": analyzer._ordinal_metadata(),
    }


def test_synthetic_analyzer_passes_frozen_screen(monkeypatch):
    result = analyzer.analyze(_raw(monkeypatch), "raw-sha")
    assert result["passes"] is True
    assert result["decision"] == (
        "authorize_frozen_c2_categorical_development"
    )
    assert result["c2_categorical_development_authorized"] is True
    assert result["categorical_default_authorized"] is False
    assert result["lockbox_touched"] is False


def test_synthetic_analyzer_fails_self_consistent_prediction_change(
    monkeypatch,
):
    raw = _raw(monkeypatch)
    for record in raw["repeats"]:
        if record["arm"] == analyzer.CANDIDATE:
            prediction = record["result"]["folds"][0]["prediction"]
            prediction["predictions"][0] += 1.0
            prediction["prediction_sha256"] = _sha256_values(
                prediction["predictions"]
            )
    _refresh_behavior(raw, analyzer.CANDIDATE)
    result = analyzer.analyze(raw, "raw-sha")
    assert result["passes"] is False
    assert result["gates"]["historical_predictions_reproduced"] is False
    assert result["gates"]["paired_predictions_bitwise_exact"] is False


def test_synthetic_analyzer_fails_ordinal_engagement(monkeypatch):
    raw = _raw(monkeypatch)
    for record in raw["repeats"]:
        if record["arm"] == analyzer.CANDIDATE:
            for fold in record["result"]["folds"]:
                fold["model"]["ordinal_state"]["metadata"]["active"] = True
            record["result"]["guardrail"]["model"]["ordinal_state"][
                "metadata"
            ]["active"] = True
    _refresh_behavior(raw, analyzer.CANDIDATE)
    result = analyzer.analyze(raw, "raw-sha")
    assert result["passes"] is False
    assert result["gates"]["ordinal_no_engagement_contract"] is False


def test_synthetic_analyzer_fails_runtime_budget(monkeypatch):
    raw = _raw(monkeypatch)
    for record in raw["repeats"]:
        if record["arm"] == analyzer.CANDIDATE:
            record["result"]["total_fit_seconds"] = 12.0
    result = analyzer.analyze(raw, "raw-sha")
    assert result["passes"] is False
    assert result["gates"]["median_total_fit_ratio_at_most_1_02"] is False


def test_synthetic_analyzer_fails_reused_numba_cache(monkeypatch):
    raw = _raw(monkeypatch)
    for record in raw["repeats"]:
        record["result"]["thread_environment"]["NUMBA_CACHE_DIR"] = (
            "/cache/reused"
        )
        record["result"]["behavior_fingerprint_sha256"] = (
            analyzer._behavior_fingerprint(record["result"])
        )
    result = analyzer.analyze(raw, "raw-sha")
    assert result["passes"] is False
    assert result["gates"]["cache_isolated_and_warmed"] is False


def test_synthetic_analyzer_fails_cross_worker_runtime_drift(monkeypatch):
    raw = _raw(monkeypatch)
    changed = raw["repeats"][0]["result"]
    changed["runtime"]["python_executable"] = "/different/python"
    changed["behavior_fingerprint_sha256"] = analyzer._behavior_fingerprint(
        changed
    )
    result = analyzer.analyze(raw, "raw-sha")
    assert result["passes"] is False
    assert result["gates"]["source_runtime_data_bound"] is False


def test_analyzer_records_worker_failure(monkeypatch):
    raw = _raw(monkeypatch)
    result = raw["repeats"][0]["result"]
    result.update(
        {
            "ok": False,
            "worker_returncode": 1,
            "error_type": "RuntimeError",
            "error": "boom",
        }
    )
    analyzed = analyzer.analyze(raw, "raw-sha")
    assert analyzed["passes"] is False
    assert analyzed["gates"] == {"all_workers_succeeded": False}
    assert analyzed["failures"][0]["error"] == "boom"


def test_analyzer_rejects_ambiguous_and_existing_paths(tmp_path: Path):
    raw = tmp_path / "raw.json"
    raw.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must be distinct"):
        analyzer._validate_paths(raw, raw, tmp_path / "report.md")
    output = tmp_path / "result.json"
    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing existing"):
        analyzer._validate_paths(raw, output, tmp_path / "report.md")


def test_analyzer_rejects_symlink_raw(tmp_path: Path):
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    raw = tmp_path / "raw.json"
    raw.symlink_to(target)
    with pytest.raises(RuntimeError, match="regular non-symlink"):
        analyzer._validate_paths(
            raw, tmp_path / "result.json", tmp_path / "report.md"
        )


def test_atomic_create_never_replaces_existing_file(tmp_path: Path):
    path = tmp_path / "result.json"
    runner._atomic_create(path, b"first")
    with pytest.raises(FileExistsError):
        runner._atomic_create(path, b"second")
    assert path.read_bytes() == b"first"


def test_strict_json_rejects_duplicate_and_nonfinite_values():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        analyzer._loads_strict(b'{"a": 1, "a": 2}')
    with pytest.raises(ValueError, match="non-finite JSON"):
        analyzer._loads_strict(b'{"a": NaN}')


def test_parse_args_defaults_and_overrides(tmp_path: Path):
    defaults = analyzer.parse_args([])
    assert defaults.raw == analyzer.DEFAULT_RAW
    assert defaults.output == analyzer.DEFAULT_OUTPUT
    assert defaults.report == analyzer.DEFAULT_REPORT
    custom = analyzer.parse_args(
        [
            "--raw",
            os.fspath(tmp_path / "raw"),
            "--output",
            os.fspath(tmp_path / "out"),
            "--report",
            os.fspath(tmp_path / "report"),
        ]
    )
    assert custom.raw == tmp_path / "raw"
    assert custom.output == tmp_path / "out"
    assert custom.report == tmp_path / "report"


def test_report_is_deterministic(monkeypatch):
    result = analyzer.analyze(_raw(monkeypatch), "raw-sha")
    report = analyzer.render_report(result)
    assert report == analyzer.render_report(copy.deepcopy(result))
    assert "C1 basketball fatal screen **passed**" in report
    assert "does not authorize a categorical default" in report
    json.dumps(result, allow_nan=False)
