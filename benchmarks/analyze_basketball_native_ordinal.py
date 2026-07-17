#!/usr/bin/env python3
"""Analyze the frozen basketball native-ordinal no-engagement campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import stat
import statistics
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "benchmarks" / "run_basketball_native_ordinal.py"
PROTOCOL_PATH = REPO_ROOT / "benchmarks" / "basketball_native_ordinal_protocol.md"

CONTROL = "control"
CANDIDATE = "candidate"
ARMS = (CONTROL, CANDIDATE)
TIMING_BLOCKS = 3
BLOCK_ORDERS = (
    (CONTROL, CANDIDATE),
    (CANDIDATE, CONTROL),
    (CONTROL, CANDIDATE),
)
EXPECTED_THREADS = 18
HELD_PREDICTION_CALLS = 200
COLD_PREDICTION_CALLS = 500

IMPLEMENTATION_COMMIT = "ceb96d191e316ab5f88204cfe767bc96f78239e8"
IMPLEMENTATION_PACKAGE_TREE = "d4661e0d4a919d2e0f0da4385b12034a5c853a6c"
PROTOCOL_COMMIT = "ae0e25aaf7fb44ab07d4cfd690ac33ddf47055ef"
EXPECTED_PROTOCOL_SHA256 = (
    "35f260c6fd2403195ce8dfda658bc0bb7ad90af8ee74383a539eb3c1ca491b9b"
)
EXPECTED_NORMALIZED_RUNNER_SHA256 = (
    "fa09739b220201aced753b775826a3f41e814ddeca19a4ecd2a9c3c8d2a3be75"
)
EXPECTED_SUPPORT_SHA256 = {
    "benchmarks/basketball_campaign_harness.py": (
        "14df91eb9c99912bd0fdf5bce81434934ff2ae84d588006a34fb513743283433"
    ),
    "benchmarks/basketball_harness.py": (
        "40360ca54d1954d540bd72bec47c891a179fb0f764a0cb6061d3a33b8593aba1"
    ),
    "benchmarks/basketball_guardrails.py": (
        "4492a65161b2261e5d437b9733c286368534c54f7fd23b6f6b95a804507fff52"
    ),
    "benchmarks/run_basketball_creator_benchmark.py": (
        "9a2d6b3dc8f3a8586cd4ac20bdb3607c17675cab1d89e6bbd7c438d2bba53fec"
    ),
}
EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "platform": "macOS-26.5.2-arm64-arm-64bit",
    "machine": "arm64",
    "logical_cpu_count": 18,
    "dependencies": {
        "numpy": "2.4.6",
        "pandas": "3.0.3",
        "scikit-learn": "1.9.0",
        "numba": "0.66.0",
        "llvmlite": "0.48.0",
        "joblib": "1.5.3",
        "threadpoolctl": "3.6.0",
    },
}
EXPECTED_DATA = {
    "raw_bytes": 2_549_434,
    "raw_sha256": (
        "43af3be46074da9130a757aa1db643e207e2c0ea5405d2086d698f85555fbcb2"
    ),
    "x_sha256": (
        "05efa554a983942885b72b1b3fdcc97a10ecf4ccbb4b10843ee8b970129fe71b"
    ),
    "y_sha256": (
        "7c06b45d4649c392fdb0c3dc91a857650f2f3fc77341fe06ffdbee2b3a44aecf"
    ),
    "fold_sha256": (
        "7270350a38a687b6e783d18d9c02e5a45f2c7367faa4a6955e74c710f5e8daea"
    ),
    "cold_mask_sha256": (
        "e17887c9240fd95aee21d37d7e1e8a533c38ef87f4492acd28fb3aa496a3dd19"
    ),
}
EXPECTED_PREDICTION_SHA256 = {
    "folds": [
        "6200db22da190d8c0787d7794c1fb8d859af737ab7e3026716e23aa1be95125f",
        "96ad500c63ac3701fe769b03a369d3a01ed1af9695d71c7ea68936d36479da44",
        "230b3cb530dee9ba8f5196b2b12b77f8d62751c545828ca13bad3fe04e54261b",
        "4603c6b3036bbdee060faaa92e6eee18a1f803e4abe9bc4aa7906745db5bd1c1",
        "e00b84d4aa7b8640aad72f5aed6e5e578cef2035459aa146b972145dc8d19fef",
        "12852587a9d1cd729cde1b28d714ff0c30b8051e806d6bb2f3f68088f22912d8",
        "514663b32f0adaf0fc7591def75632f5ea1103598b2d7aaeeaf37fdc2560bb04",
        "45374906a6931f90a6fff29ba0544c4d66311bb6152e3f250d54db55e0c03384",
        "32167d2ad1ba4ee34297a812be85ae67675f638383d61fe709b130bdbb3931a5",
        "f51972e8f896568291b259d698726b224a2399711f8e8cdf451e68b5090ae38d",
    ],
    "held": (
        "5d910ae8f6b0dca563b99f9f881dcb17ee092711a46b2890452eaa3b8e68367a"
    ),
    "cold": (
        "998a14f530ed284865a50726191da067f72d69da3001614d664a4b90e7aa6376"
    ),
    "seen": (
        "c9b506afbfb3eb660dd918ee9635d996c0285b0320ba250cbf39c80df9122425"
    ),
}
EXPECTED_MEAN_R2 = 0.5267495183883605
EXPECTED_FOLD_LEARNING_RATES = (0.052312, *([0.052314] * 9))
EXPECTED_GUARDRAIL_LEARNING_RATE = 0.053192
EXPECTED_TREE_COUNT = 1_000

MAX_FIT_RATIO = 1.02
MAX_HELD_PREDICT_RATIO = 1.05
MAX_COLD_PREDICT_RATIO = 1.05
MAX_RSS_RATIO = 1.05
MAX_PAIRED_RATIO_IQR_OVER_MEDIAN = 0.10

DEFAULT_RAW = REPO_ROOT / "benchmarks" / "basketball_native_ordinal_raw.json"
DEFAULT_OUTPUT = (
    REPO_ROOT / "benchmarks" / "basketball_native_ordinal_result.json"
)
DEFAULT_REPORT = (
    REPO_ROOT / "benchmarks" / "basketball_native_ordinal_result.md"
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _normalized_runner_sha256() -> str:
    payload = RUNNER_PATH.read_bytes()
    pattern = (
        rb'EXPECTED_NORMALIZED_RUNNER_SHA256 = \(\n'
        rb'    "[0-9a-f]{64}"\n'
        rb"\)"
    )
    replacement = (
        b'EXPECTED_NORMALIZED_RUNNER_SHA256 = (\n'
        b'    "' + (b"0" * 64) + b'"\n'
        b")"
    )
    normalized, count = re.subn(pattern, replacement, payload)
    if count != 1:
        raise RuntimeError("runner normalization field changed")
    return _sha256_bytes(normalized)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_git_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _prediction_sha256(prediction: Any) -> str:
    values = np.ascontiguousarray(np.asarray(prediction, dtype="<f8"))
    return _sha256_bytes(values.tobytes())


def _without_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if "_seconds" not in key
            and key not in {"worker_stdout", "worker_stderr"}
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def _model_behavior_payload(observation: dict[str, Any]) -> dict[str, Any]:
    archive = observation["archive"]
    return {
        "fit_metadata": observation["fit_metadata"],
        "ordinal_state": observation["ordinal_state"],
        "preprocessor": observation["preprocessor"],
        "feature_importance": observation["feature_importance"],
        "feature_importance_sha256": observation[
            "feature_importance_sha256"
        ],
        "archive": {
            "normalized_sha256": archive["normalized_sha256"],
            "normalized_removed_fields": archive[
                "normalized_removed_fields"
            ],
        },
    }


def _behavior_fingerprint(result: dict[str, Any]) -> str:
    guardrail = result["guardrail"]
    payload = {
        "arm": result["arm"],
        "folds": [
            {
                "fold": row["fold"],
                "test_indices": row["test_indices"],
                "prediction_sha256": row["prediction"]["prediction_sha256"],
                "predictions": row["prediction"]["predictions"],
                "model": _model_behavior_payload(row["model"]),
            }
            for row in result["folds"]
        ],
        "guardrail": {
            "model": _model_behavior_payload(guardrail["model"]),
            "held": guardrail["held"],
            "cold": guardrail["cold"],
            "seen": guardrail["seen"],
            "scores_from_full_prediction": guardrail[
                "scores_from_full_prediction"
            ],
        },
        "runtime": result["runtime"],
        "data": result["data"],
    }
    return _sha256_bytes(_canonical_json(_without_timing(payload)))


def _ordinal_metadata() -> dict[str, Any]:
    return {
        "mode": "auto",
        "active": False,
        "feature_count": 0,
        "feature_indices": [],
        "feature_names": [],
        "sources": [],
        "nominal_categorical_count": 0,
        "added_columns": 0,
        "target_stat_blocks_added": 0,
        "target_used": False,
        "unknown_policy": "fail_closed",
        "missing_policy": "numeric_missing_bin",
    }


def _expected_test_indices() -> list[list[int]]:
    sizes = [525, *([524] * 9)]
    starts = np.cumsum([0, *sizes[:-1]])
    return [
        list(range(int(start), int(start + size)))
        for start, size in zip(starts, sizes, strict=True)
    ]


def _validate_prediction_record(
    record: dict[str, Any],
    *,
    rows: int,
) -> np.ndarray:
    prediction = np.asarray(record.get("predictions"), dtype=np.float64)
    if prediction.shape != (rows,) or not np.all(np.isfinite(prediction)):
        raise RuntimeError("native-ordinal prediction payload is invalid")
    digest = _prediction_sha256(prediction)
    if record.get("prediction_sha256") != digest:
        raise RuntimeError("native-ordinal prediction hash is inconsistent")
    score = record.get("r2")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise RuntimeError("native-ordinal score is invalid")
    return prediction


def _validate_fit_metadata(
    metadata: dict[str, Any],
    expected_learning_rate: float,
) -> bool:
    try:
        final = metadata["final_fit"]
        fixed = (
            int(metadata["best_iteration"]) == EXPECTED_TREE_COUNT
            and int(metadata["fitted_tree_count"]) == EXPECTED_TREE_COUNT
            and float(metadata["resolved_learning_rate"])
            == float(expected_learning_rate)
            and metadata["requested_tree_mode"] == "catboost"
            and metadata["selected_tree_mode"] == "catboost"
            and metadata["selected_lane"] == "boosting"
            and metadata["linear_residual_active"] is False
            and metadata["linear_leaves_active"] is False
            and int(metadata["resolved_thread_count"]) == EXPECTED_THREADS
            and metadata["refit"] is False
            and metadata["refit_strategy"] is None
            and metadata["selection_fit"] is None
            and metadata["selection_early_stopping_rounds"] is None
            and metadata["final_early_stopping_rounds"] is None
            and int(final["iterations_requested"]) == EXPECTED_TREE_COUNT
            and int(final["iterations_attempted"]) == EXPECTED_TREE_COUNT
            and int(final["rounds_completed"]) == EXPECTED_TREE_COUNT
            and int(final["rounds_retained"]) == EXPECTED_TREE_COUNT
            and final["stop_reason"] == "iteration_limit"
        )
        phase_seconds = final["phase_seconds"]
        if not isinstance(phase_seconds, dict):
            return False
        return fixed and all(
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in phase_seconds.values()
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _validate_model_observation(
    observation: dict[str, Any],
    arm: str,
    *,
    expected_learning_rate: float,
) -> dict[str, Any]:
    importance = np.asarray(
        observation.get("feature_importance"), dtype=np.float64
    )
    if importance.shape != (15,) or not np.all(np.isfinite(importance)):
        raise RuntimeError("native-ordinal feature importance is invalid")
    importance_digest = _prediction_sha256(importance)
    if observation.get("feature_importance_sha256") != importance_digest:
        raise RuntimeError("native-ordinal feature-importance hash changed")

    preprocessor = observation.get("preprocessor")
    if not isinstance(preprocessor, dict):
        raise RuntimeError("native-ordinal preprocessor state is invalid")
    numeric = list(range(15))
    preprocessor_contract = (
        preprocessor.get("n_input_features") == 15
        and preprocessor.get("num_features") == numeric
        and preprocessor.get("cat_features") == []
        and preprocessor.get("feature_map") == numeric
        and isinstance(preprocessor.get("n_bins"), list)
        and len(preprocessor["n_bins"]) == 15
        and all(
            isinstance(value, int) and value >= 2
            for value in preprocessor["n_bins"]
        )
        and isinstance(preprocessor.get("border_offsets"), list)
        and len(preprocessor["border_offsets"]) == 16
        and preprocessor["border_offsets"][0] == 0
        and all(
            left <= right
            for left, right in zip(
                preprocessor["border_offsets"],
                preprocessor["border_offsets"][1:],
            )
        )
        and isinstance(preprocessor.get("block_widths"), list)
        and all(
            isinstance(value, int) and value > 0
            for value in preprocessor["block_widths"]
        )
        and preprocessor.get("encoder_count") == 0
        and preprocessor.get("target_stat_blocks") == 0
        and _is_sha256(preprocessor.get("borders_flat_sha256"))
    )

    ordinal = observation.get("ordinal_state")
    if not isinstance(ordinal, dict):
        raise RuntimeError("native-ordinal telemetry is invalid")
    if arm == CONTROL:
        ordinal_contract = ordinal == {
            "mode": "off",
            "records": [],
            "indices": [],
            "metadata": None,
        }
    else:
        ordinal_contract = ordinal == {
            "mode": "auto",
            "records": [],
            "indices": [],
            "metadata": _ordinal_metadata(),
        }

    archive = observation.get("archive")
    if not isinstance(archive, dict):
        raise RuntimeError("native-ordinal archive identity is invalid")
    expected_removed = (
        [
            "auto_params.ordinal_features",
            "auto_params.diagnostics.ordinal_features",
            "wrapper.state.ordinal_features_mode",
            "wrapper.state.ordinal_features",
            "timing",
        ]
        if arm == CANDIDATE
        else ["timing"]
    )
    archive_contract = (
        isinstance(archive.get("raw_bytes"), int)
        and archive["raw_bytes"] > 0
        and _is_sha256(archive.get("raw_sha256"))
        and _is_sha256(archive.get("normalized_sha256"))
        and archive.get("normalized_removed_fields") == expected_removed
    )
    fit_contract = _validate_fit_metadata(
        observation.get("fit_metadata", {}),
        expected_learning_rate,
    )
    return {
        "fit_contract": fit_contract,
        "ordinal_contract": ordinal_contract,
        "preprocessor_contract": preprocessor_contract,
        "archive_contract": archive_contract,
        "logical_state": {
            "fit_metadata": _without_timing(observation["fit_metadata"]),
            "preprocessor": preprocessor,
            "feature_importance": importance.tolist(),
            "feature_importance_sha256": importance_digest,
            "archive_normalized_sha256": archive.get("normalized_sha256"),
        },
    }


def _validate_data(data: dict[str, Any]) -> bool:
    try:
        raw = data["raw"]
        processed = data["processed"]
        guardrail = data["guardrail"]
        return (
            raw["bytes"] == EXPECTED_DATA["raw_bytes"]
            and raw["sha256"] == EXPECTED_DATA["raw_sha256"]
            and raw["load_source"] == "cache"
            and processed["x_train_sha256"] == EXPECTED_DATA["x_sha256"]
            and processed["y_train_sha256"] == EXPECTED_DATA["y_sha256"]
            and processed["train_rows"] == 5_241
            and processed["test_rows"] == 2_409
            and processed["missing_train_feature_cells"] == 0
            and processed["features"]
            == [
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
            and data["fold_sha256"] == EXPECTED_DATA["fold_sha256"]
            and data["fold_test_sizes"] == [525, *([524] * 9)]
            and data["feature_dtypes"]
            == {
                **{
                    name: "float64"
                    for name in processed["features"]
                    if name != "Age"
                },
                "Age": "int64",
            }
            and guardrail["cold_player_mask_sha256"]
            == EXPECTED_DATA["cold_mask_sha256"]
            and guardrail["train_rows"] == 5_241
            and guardrail["holdout_rows"] == 2_409
            and guardrail["cold_player_holdout_rows"] == 585
            and guardrail["seen_player_holdout_rows"] == 1_824
            and guardrail["cold_players_absent_from_training"] is True
            and guardrail["player_identifier_used_as_model_feature"] is False
        )
    except (KeyError, TypeError):
        return False


def _validate_runtime(runtime: dict[str, Any]) -> bool:
    return (
        isinstance(runtime, dict)
        and all(runtime.get(key) == EXPECTED_RUNTIME[key] for key in (
            "python",
            "platform",
            "machine",
            "logical_cpu_count",
            "dependencies",
        ))
        and isinstance(runtime.get("python_executable"), str)
        and bool(runtime["python_executable"])
    )


def _validate_environment(environment: dict[str, Any]) -> bool:
    thread_keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TBB_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    )
    return (
        isinstance(environment, dict)
        and environment.get("DARKOFIT_WARMUP") == "0"
        and environment.get("CHIMERABOOST_WARMUP") == "0"
        and all(environment.get(key) == "18" for key in thread_keys)
        and environment.get("NUMBA_DISABLE_JIT") == "0"
        and environment.get("ENABLE_IPC") == "1"
        and environment.get("PYTHONHASHSEED") == "0"
        and isinstance(environment.get("NUMBA_CACHE_DIR"), str)
        and bool(environment["NUMBA_CACHE_DIR"])
        and all(
            environment.get(key) is None
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
        )
    )


def _validate_cache(cache: dict[str, Any]) -> bool:
    try:
        before = cache["before_import"]
        imported = cache["after_import"]
        warmed = cache["after_warmup"]
        final = cache["after_workload"]
        return (
            before["file_count"] == 0
            and before["compiled_file_count"] == 0
            and imported["compiled_file_count"] == 0
            and warmed["compiled_file_count"] > 0
            and final["compiled_file_count"] >= warmed["compiled_file_count"]
        )
    except (KeyError, TypeError):
        return False


def _validate_result(result: dict[str, Any], arm: str) -> dict[str, Any]:
    if result.get("arm") != arm:
        raise RuntimeError("native-ordinal worker arm is inconsistent")
    if result.get("behavior_fingerprint_sha256") != _behavior_fingerprint(result):
        raise RuntimeError("native-ordinal behavior fingerprint is inconsistent")

    folds = result.get("folds")
    if not isinstance(folds, list) or len(folds) != 10:
        raise RuntimeError("native-ordinal fold result count changed")
    expected_indices = _expected_test_indices()
    historical_predictions = True
    fit_contract = True
    ordinal_contract = True
    preprocessing_contract = True
    archive_contract = True
    logical_models = []
    for fold, row in enumerate(folds):
        if (
            row.get("fold") != fold
            or row.get("test_indices") != expected_indices[fold]
            or row.get("test_rows") != len(expected_indices[fold])
            or row.get("train_rows") != 5_241 - len(expected_indices[fold])
        ):
            raise RuntimeError("native-ordinal fold boundary changed")
        prediction = _validate_prediction_record(
            row["prediction"], rows=len(expected_indices[fold])
        )
        historical_predictions &= (
            _prediction_sha256(prediction)
            == EXPECTED_PREDICTION_SHA256["folds"][fold]
        )
        model = _validate_model_observation(
            row["model"],
            arm,
            expected_learning_rate=EXPECTED_FOLD_LEARNING_RATES[fold],
        )
        fit_contract &= model["fit_contract"]
        ordinal_contract &= model["ordinal_contract"]
        preprocessing_contract &= model["preprocessor_contract"]
        archive_contract &= model["archive_contract"]
        logical_models.append(model["logical_state"])

    fold_scores = np.asarray(result.get("fold_scores"), dtype=np.float64)
    recorded_scores = np.asarray(
        [row["prediction"]["r2"] for row in folds], dtype=np.float64
    )
    if (
        fold_scores.shape != (10,)
        or not np.all(np.isfinite(fold_scores))
        or not np.array_equal(fold_scores, recorded_scores)
    ):
        raise RuntimeError("native-ordinal fold score ledger is inconsistent")
    score_contract = (
        float(result.get("mean_r2")) == float(np.mean(fold_scores))
        and float(result["mean_r2"]) == EXPECTED_MEAN_R2
    )

    guardrail = result.get("guardrail")
    if not isinstance(guardrail, dict):
        raise RuntimeError("native-ordinal guardrail result is invalid")
    held = _validate_prediction_record(guardrail["held"], rows=2_409)
    cold = _validate_prediction_record(guardrail["cold"], rows=585)
    seen = _validate_prediction_record(guardrail["seen"], rows=1_824)
    historical_predictions &= (
        _prediction_sha256(held) == EXPECTED_PREDICTION_SHA256["held"]
        and _prediction_sha256(cold) == EXPECTED_PREDICTION_SHA256["cold"]
        and _prediction_sha256(seen) == EXPECTED_PREDICTION_SHA256["seen"]
    )
    historical_predictions &= (
        guardrail["held"].get("loop_calls") == HELD_PREDICTION_CALLS
        and guardrail["held"].get("loop_last_prediction_sha256")
        == guardrail["held"]["prediction_sha256"]
        and guardrail["cold"].get("loop_calls") == COLD_PREDICTION_CALLS
        and guardrail["cold"].get("loop_last_prediction_sha256")
        == guardrail["cold"]["prediction_sha256"]
    )
    scores = guardrail.get("scores_from_full_prediction", {})
    expected_views = {
        "overlap_exposed_team_holdout": (2_409, guardrail["held"]),
        "cold_player_subset": (585, guardrail["cold"]),
        "seen_player_subset": (1_824, guardrail["seen"]),
    }
    guardrail_score_contract = set(scores) == set(expected_views)
    for view, (rows, prediction_record) in expected_views.items():
        score = scores.get(view, {})
        guardrail_score_contract &= (
            score.get("rows") == rows
            and score.get("r2") == prediction_record.get("r2")
            and score.get("prediction_sha256")
            == prediction_record.get("prediction_sha256")
        )

    guardrail_model = _validate_model_observation(
        guardrail["model"],
        arm,
        expected_learning_rate=EXPECTED_GUARDRAIL_LEARNING_RATE,
    )
    fit_contract &= guardrail_model["fit_contract"]
    ordinal_contract &= guardrail_model["ordinal_contract"]
    preprocessing_contract &= guardrail_model["preprocessor_contract"]
    archive_contract &= guardrail_model["archive_contract"]
    logical_models.append(guardrail_model["logical_state"])

    numeric_timings = (
        result.get("total_fit_seconds"),
        result.get("worker_wall_seconds"),
        result.get("peak_rss_bytes"),
        guardrail.get("full_fit_seconds"),
        guardrail["held"].get("loop_seconds"),
        guardrail["cold"].get("loop_seconds"),
    )
    timing_valid = all(
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
        for value in numeric_timings
    )
    return {
        "historical_predictions": bool(historical_predictions),
        "fit_contract": bool(fit_contract and score_contract),
        "ordinal_contract": bool(ordinal_contract),
        "preprocessing_contract": bool(preprocessing_contract),
        "archive_contract": bool(archive_contract),
        "guardrail_score_contract": bool(guardrail_score_contract),
        "source_contract": result.get("source"),
        "runtime_contract": _validate_runtime(result.get("runtime")),
        "data_contract": _validate_data(result.get("data")),
        "environment_contract": _validate_environment(
            result.get("thread_environment")
        ),
        "cache_contract": _validate_cache(result.get("cache")),
        "warning_contract": result.get("warnings") == [],
        "timing_valid": timing_valid,
        "fold_predictions": [
            np.asarray(row["prediction"]["predictions"], dtype=np.float64)
            for row in folds
        ],
        "held_prediction": held,
        "cold_prediction": cold,
        "seen_prediction": seen,
        "logical_models": logical_models,
    }


def _verify_bindings(raw: dict[str, Any]) -> None:
    if raw.get("schema_version") != 1 or raw.get("name") != (
        "darkofit_basketball_native_ordinal_raw_v1"
    ):
        raise RuntimeError("native-ordinal raw artifact has an unknown schema")
    if raw.get("protocol") != {
        "path": "benchmarks/basketball_native_ordinal_protocol.md",
        "commit": PROTOCOL_COMMIT,
        "sha256": EXPECTED_PROTOCOL_SHA256,
    }:
        raise RuntimeError("native-ordinal protocol binding changed")
    if raw.get("runner") != {
        "path": "benchmarks/run_basketball_native_ordinal.py",
        "normalized_sha256": EXPECTED_NORMALIZED_RUNNER_SHA256,
    }:
        raise RuntimeError("native-ordinal runner binding changed")
    analyzer = raw.get("analyzer")
    source = raw.get("source")
    if not isinstance(analyzer, dict) or not isinstance(source, dict):
        raise RuntimeError("native-ordinal source binding is missing")
    analyzer_sha = _sha256_file(Path(__file__).resolve())
    if analyzer != {
        "path": "benchmarks/analyze_basketball_native_ordinal.py",
        "sha256": analyzer_sha,
    }:
        raise RuntimeError("native-ordinal analyzer binding changed")
    if (
        source.get("branch") != "main"
        or source.get("head") != source.get("origin_main")
        or not _is_git_sha(source.get("head"))
        or source.get("status") != ""
        or source.get("package_tree") != IMPLEMENTATION_PACKAGE_TREE
        or source.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
        or source.get("runner_normalized_sha256")
        != EXPECTED_NORMALIZED_RUNNER_SHA256
        or source.get("analyzer_sha256") != analyzer_sha
        or source.get("support_sha256") != EXPECTED_SUPPORT_SHA256
    ):
        raise RuntimeError("native-ordinal frozen source binding changed")
    if _sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("local native-ordinal protocol changed")
    if _normalized_runner_sha256() != EXPECTED_NORMALIZED_RUNNER_SHA256:
        raise RuntimeError("local native-ordinal runner changed")
    if any(
        _sha256_file(REPO_ROOT / relative) != digest
        for relative, digest in EXPECTED_SUPPORT_SHA256.items()
    ):
        raise RuntimeError("local native-ordinal support source changed")
    execution = raw.get("execution")
    if execution != {
        "threads": EXPECTED_THREADS,
        "timing_blocks": TIMING_BLOCKS,
        "block_orders": [list(order) for order in BLOCK_ORDERS],
        "held_prediction_calls": HELD_PREDICTION_CALLS,
        "cold_prediction_calls": COLD_PREDICTION_CALLS,
        "worker_count": 6,
        "partial_resume_supported": False,
    }:
        raise RuntimeError("native-ordinal execution contract changed")
    if (
        raw.get("categorical_outcomes_inspected") is not False
        or raw.get("lockbox_touched") is not False
    ):
        raise RuntimeError("native-ordinal evidence boundary changed")


def _group_results(
    raw: dict[str, Any],
) -> tuple[dict[int, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    repeats = raw.get("repeats")
    if not isinstance(repeats, list) or len(repeats) != 6:
        raise RuntimeError("native-ordinal worker count changed")
    expected = {
        (block, position): arm
        for block, order in enumerate(BLOCK_ORDERS)
        for position, arm in enumerate(order)
    }
    observed = set()
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    failures = []
    for record in repeats:
        try:
            coordinate = (int(record["block"]), int(record["position"]))
            arm = str(record["arm"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("native-ordinal worker coordinate is invalid") from exc
        if coordinate in observed or expected.get(coordinate) != arm:
            raise RuntimeError("native-ordinal worker schedule changed")
        observed.add(coordinate)
        if record.get("order") != list(BLOCK_ORDERS[coordinate[0]]):
            raise RuntimeError("native-ordinal worker order ledger changed")
        result = record.get("result")
        if not isinstance(result, dict) or result.get("arm") != arm:
            raise RuntimeError("native-ordinal worker result is invalid")
        if (
            result.get("block") != coordinate[0]
            or result.get("position") != coordinate[1]
        ):
            raise RuntimeError("native-ordinal nested worker coordinate changed")
        grouped.setdefault(coordinate[0], {})[arm] = result
        if result.get("ok") is not True or result.get("worker_returncode") != 0:
            failures.append(
                {
                    "block": coordinate[0],
                    "position": coordinate[1],
                    "arm": arm,
                    "returncode": result.get("worker_returncode"),
                    "error_type": result.get("error_type"),
                    "error": result.get("error"),
                    "stderr": result.get("worker_stderr"),
                }
            )
        elif (
            result.get("worker_stdout") is not None
            or result.get("worker_stderr") is not None
        ):
            failures.append(
                {
                    "block": coordinate[0],
                    "position": coordinate[1],
                    "arm": arm,
                    "returncode": result.get("worker_returncode"),
                    "error_type": "UnexpectedWorkerOutput",
                    "error": "successful worker emitted stdout or stderr",
                    "stderr": result.get("worker_stderr"),
                }
            )
    if observed != set(expected):
        raise RuntimeError("native-ordinal worker schedule is incomplete")
    return grouped, failures


def _paired_ratio_summary(
    numerator: list[float],
    denominator: list[float],
    *,
    limit: float,
) -> dict[str, Any]:
    left = np.asarray(numerator, dtype=np.float64)
    right = np.asarray(denominator, dtype=np.float64)
    if (
        left.shape != (TIMING_BLOCKS,)
        or right.shape != (TIMING_BLOCKS,)
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        or np.any(left <= 0.0)
        or np.any(right <= 0.0)
    ):
        raise RuntimeError("native-ordinal paired timing is invalid")
    ratios = left / right
    median = float(np.median(ratios))
    q1, q3 = (float(value) for value in np.percentile(ratios, [25.0, 75.0]))
    relative = float((q3 - q1) / median)
    return {
        "candidate_values": left.tolist(),
        "control_values": right.tolist(),
        "paired_ratios": ratios.tolist(),
        "candidate_median": float(np.median(left)),
        "control_median": float(np.median(right)),
        "median_ratio": median,
        "q1_ratio": q1,
        "q3_ratio": q3,
        "iqr_over_median": relative,
        "ratio_limit": float(limit),
        "ratio_passed": median <= float(limit),
        "stability_limit": MAX_PAIRED_RATIO_IQR_OVER_MEDIAN,
        "stable": relative <= MAX_PAIRED_RATIO_IQR_OVER_MEDIAN,
    }


def _failed_execution_result(
    raw_sha256: str,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "darkofit_basketball_native_ordinal_result_v1",
        "raw": {"sha256": raw_sha256},
        "passes": False,
        "decision": "close_c1_native_ordinal_execution_failed",
        "gates": {"all_workers_succeeded": False},
        "failures": failures,
        "timing": None,
        "c2_categorical_development_authorized": False,
        "categorical_default_authorized": False,
        "lockbox_touched": False,
    }


def analyze(raw: dict[str, Any], raw_sha256: str) -> dict[str, Any]:
    _verify_bindings(raw)
    grouped, failures = _group_results(raw)
    if failures:
        return _failed_execution_result(raw_sha256, failures)

    validated: dict[int, dict[str, dict[str, Any]]] = {}
    for block in range(TIMING_BLOCKS):
        validated[block] = {
            arm: _validate_result(grouped[block][arm], arm) for arm in ARMS
        }

    paired_predictions_exact = True
    paired_model_state_exact = True
    for block in range(TIMING_BLOCKS):
        control = validated[block][CONTROL]
        candidate = validated[block][CANDIDATE]
        paired_predictions_exact &= all(
            np.array_equal(left, right)
            for left, right in zip(
                control["fold_predictions"],
                candidate["fold_predictions"],
                strict=True,
            )
        )
        paired_predictions_exact &= (
            np.array_equal(
                control["held_prediction"], candidate["held_prediction"]
            )
            and np.array_equal(
                control["cold_prediction"], candidate["cold_prediction"]
            )
            and np.array_equal(
                control["seen_prediction"], candidate["seen_prediction"]
            )
        )
        paired_model_state_exact &= (
            control["logical_models"] == candidate["logical_models"]
        )

    all_rows = [
        validated[block][arm]
        for block in range(TIMING_BLOCKS)
        for arm in ARMS
    ]
    raw_results = [
        grouped[block][arm]
        for block in range(TIMING_BLOCKS)
        for arm in ARMS
    ]
    source = raw["source"]
    runtime_records_identical = all(
        row["runtime"] == raw_results[0]["runtime"]
        for row in raw_results
    )
    data_records_identical = all(
        row["data"] == raw_results[0]["data"]
        for row in raw_results
    )
    source_runtime_data_bound = all(
        row["source_contract"] == source
        and row["runtime_contract"]
        and row["data_contract"]
        and row["environment_contract"]
        for row in all_rows
    ) and runtime_records_identical and data_records_identical
    cache_paths = [
        grouped[block][arm]["thread_environment"]["NUMBA_CACHE_DIR"]
        for block in range(TIMING_BLOCKS)
        for arm in ARMS
    ]
    cache_isolated_and_warmed = (
        all(row["cache_contract"] for row in all_rows)
        and len(set(cache_paths)) == TIMING_BLOCKS * len(ARMS)
    )
    no_warnings = all(row["warning_contract"] for row in all_rows)
    historical_predictions = all(
        row["historical_predictions"] for row in all_rows
    )
    historical_fit_contract = all(
        row["fit_contract"] for row in all_rows
    )
    no_engagement = all(
        row["ordinal_contract"] for row in all_rows
    )
    preprocessing_contract = all(
        row["preprocessing_contract"] for row in all_rows
    )
    archive_contract = all(
        row["archive_contract"] for row in all_rows
    )
    guardrail_score_contract = all(
        row["guardrail_score_contract"] for row in all_rows
    )
    timing_valid = all(row["timing_valid"] for row in all_rows)
    behavior_reproduced = all(
        len(
            {
                grouped[block][arm]["behavior_fingerprint_sha256"]
                for block in range(TIMING_BLOCKS)
            }
        )
        == 1
        for arm in ARMS
    )

    def values(arm: str, getter) -> list[float]:
        return [
            float(getter(grouped[block][arm]))
            for block in range(TIMING_BLOCKS)
        ]

    timing = {
        "total_fit": _paired_ratio_summary(
            values(CANDIDATE, lambda row: row["total_fit_seconds"]),
            values(CONTROL, lambda row: row["total_fit_seconds"]),
            limit=MAX_FIT_RATIO,
        ),
        "held_prediction": _paired_ratio_summary(
            values(
                CANDIDATE,
                lambda row: row["guardrail"]["held"]["loop_seconds"],
            ),
            values(
                CONTROL,
                lambda row: row["guardrail"]["held"]["loop_seconds"],
            ),
            limit=MAX_HELD_PREDICT_RATIO,
        ),
        "cold_prediction": _paired_ratio_summary(
            values(
                CANDIDATE,
                lambda row: row["guardrail"]["cold"]["loop_seconds"],
            ),
            values(
                CONTROL,
                lambda row: row["guardrail"]["cold"]["loop_seconds"],
            ),
            limit=MAX_COLD_PREDICT_RATIO,
        ),
        "peak_rss": _paired_ratio_summary(
            values(CANDIDATE, lambda row: row["peak_rss_bytes"]),
            values(CONTROL, lambda row: row["peak_rss_bytes"]),
            limit=MAX_RSS_RATIO,
        ),
        "per_arm_diagnostics": {
            arm: {
                metric: {
                    "values": metric_values,
                    "median": float(statistics.median(metric_values)),
                    "iqr_over_median": float(
                        (
                            np.percentile(metric_values, 75.0)
                            - np.percentile(metric_values, 25.0)
                        )
                        / np.median(metric_values)
                    ),
                }
                for metric, metric_values in {
                    "total_fit_seconds": values(
                        arm, lambda row: row["total_fit_seconds"]
                    ),
                    "held_prediction_seconds": values(
                        arm,
                        lambda row: row["guardrail"]["held"]["loop_seconds"],
                    ),
                    "cold_prediction_seconds": values(
                        arm,
                        lambda row: row["guardrail"]["cold"]["loop_seconds"],
                    ),
                    "peak_rss_bytes": values(
                        arm, lambda row: row["peak_rss_bytes"]
                    ),
                }.items()
            }
            for arm in ARMS
        },
    }
    gates = {
        "all_workers_succeeded": True,
        "source_runtime_data_bound": source_runtime_data_bound,
        "cache_isolated_and_warmed": cache_isolated_and_warmed,
        "no_unexpected_warnings": no_warnings,
        "historical_predictions_reproduced": historical_predictions,
        "paired_predictions_bitwise_exact": bool(paired_predictions_exact),
        "historical_fit_contract_reproduced": historical_fit_contract,
        "paired_logical_model_state_exact": bool(paired_model_state_exact),
        "preprocessing_contract_reproduced": preprocessing_contract,
        "archive_contract_reproduced": archive_contract,
        "guardrail_score_contract_reproduced": guardrail_score_contract,
        "ordinal_no_engagement_contract": no_engagement,
        "behavior_reproduced_across_blocks": behavior_reproduced,
        "timing_values_valid": timing_valid,
        "median_total_fit_ratio_at_most_1_02": timing["total_fit"][
            "ratio_passed"
        ],
        "median_held_prediction_ratio_at_most_1_05": timing[
            "held_prediction"
        ]["ratio_passed"],
        "median_cold_prediction_ratio_at_most_1_05": timing[
            "cold_prediction"
        ]["ratio_passed"],
        "fit_paired_ratio_stable": timing["total_fit"]["stable"],
        "held_prediction_paired_ratio_stable": timing[
            "held_prediction"
        ]["stable"],
        "cold_prediction_paired_ratio_stable": timing[
            "cold_prediction"
        ]["stable"],
        "median_peak_rss_ratio_at_most_1_05": timing["peak_rss"][
            "ratio_passed"
        ],
    }
    passes = all(gates.values())
    return {
        "schema_version": 1,
        "name": "darkofit_basketball_native_ordinal_result_v1",
        "raw": {
            "sha256": raw_sha256,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "runner_normalized_sha256": EXPECTED_NORMALIZED_RUNNER_SHA256,
            "analyzer_sha256": raw["analyzer"]["sha256"],
            "source_commit": raw["source"]["head"],
        },
        "passes": passes,
        "decision": (
            "authorize_frozen_c2_categorical_development"
            if passes
            else "close_c1_native_ordinal_implementation_shape"
        ),
        "gates": gates,
        "failures": [],
        "timing": timing,
        "mean_r2": EXPECTED_MEAN_R2,
        "c2_categorical_development_authorized": passes,
        "categorical_default_authorized": False,
        "categorical_quality_claim_authorized": False,
        "lockbox_touched": False,
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Basketball native-ordinal no-engagement result",
        "",
        "## Decision",
        "",
    ]
    if result["passes"]:
        lines.extend(
            [
                "The C1 basketball fatal screen **passed**. Native ordinal "
                "handling remained an exact inactive no-op on numeric basketball "
                "data and stayed within every preregistered runtime and memory gate.",
                "",
                "This authorizes only the frozen C2 categorical development tier. "
                "It does not authorize a categorical default or quality claim.",
            ]
        )
    else:
        lines.extend(
            [
                "The C1 basketball fatal screen **failed**. This native-ordinal "
                "implementation shape is closed under the frozen protocol.",
                "",
                "C2 categorical development is not authorized from this result.",
            ]
        )
    lines.extend(["", f"Decision code: `{result['decision']}`.", ""])
    if result["failures"]:
        lines.extend(["## Worker failures", ""])
        for failure in result["failures"]:
            lines.append(
                f"- Block {failure['block']} `{failure['arm']}`: "
                f"{failure.get('error_type')}: {failure.get('error')}"
            )
        lines.append("")
    lines.extend(
        [
            "## Gates",
            "",
            "| Gate | Passed |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| `{name}` | **{passed}** |"
        for name, passed in result["gates"].items()
    )
    if result["timing"] is not None:
        lines.extend(
            [
                "",
                "## Paired operating ratios",
                "",
                "| Metric | Median candidate/control | IQR/median |",
                "|---|---:|---:|",
            ]
        )
        for key, label in (
            ("total_fit", "Eleven-fit total"),
            ("held_prediction", "Held-team prediction"),
            ("cold_prediction", "Cold-player prediction"),
            ("peak_rss", "Peak RSS"),
        ):
            row = result["timing"][key]
            lines.append(
                f"| {label} | {row['median_ratio']:.4f}× | "
                f"{row['iqr_over_median']:.4f} |"
            )
        lines.extend(
            [
                "",
                "Per-arm timing dispersion is diagnostic only. Stability gates "
                "use the preregistered same-block paired ratios.",
            ]
        )
    lines.extend(
        [
            "",
            f"Raw artifact SHA-256: `{result['raw']['sha256']}`.",
            "",
            "No CTR23, TabArena, I3, fresh-confirmation, or lockbox coordinate "
            "was touched.",
            "",
        ]
    )
    return "\n".join(lines)


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _validate_paths(raw: Path, output: Path, report: Path) -> tuple[Path, Path, Path]:
    paths = tuple(_absolute_lexical(path) for path in (raw, output, report))
    if len(set(paths)) != 3:
        raise RuntimeError(
            "raw, analyzed JSON, and Markdown report paths must be distinct"
        )
    try:
        raw_stat = paths[0].lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"raw native-ordinal artifact is unavailable: {paths[0]}") from exc
    if stat.S_ISLNK(raw_stat.st_mode) or not stat.S_ISREG(raw_stat.st_mode):
        raise RuntimeError("raw native-ordinal artifact must be a regular non-symlink")
    for path in paths[1:]:
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing existing analyzer output: {path}")
    physical = [path.resolve(strict=False) for path in paths]
    if len(set(physical)) != 3:
        raise RuntimeError(
            "raw and analyzer outputs resolve to ambiguous physical paths"
        )
    return paths


def _stable_read(path: Path) -> tuple[bytes, str]:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise RuntimeError(
            "raw native-ordinal artifact must remain a regular non-symlink"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError("raw native-ordinal artifact changed before reading")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = path.lstat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(payload) != after.st_size:
        raise RuntimeError("raw native-ordinal artifact changed while reading")
    return payload, _sha256_bytes(payload)


def _atomic_create(path: Path, payload: bytes) -> None:
    path = _absolute_lexical(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _loads_strict(payload: bytes) -> dict[str, Any]:
    value = json.loads(
        payload.decode("utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(value, dict):
        raise ValueError("native-ordinal raw artifact must contain a JSON object")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw_path, output_path, report_path = _validate_paths(
        args.raw, args.output, args.report
    )
    raw_payload, raw_sha256 = _stable_read(raw_path)
    raw = _loads_strict(raw_payload)
    result = analyze(raw, raw_sha256)

    _, prepublication_sha256 = _stable_read(raw_path)
    if prepublication_sha256 != raw_sha256:
        raise RuntimeError("raw native-ordinal artifact changed before publication")
    output_payload = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    report_payload = render_report(result).encode("utf-8")
    _atomic_create(output_path, output_payload)
    _atomic_create(report_path, report_payload)
    _, postpublication_sha256 = _stable_read(raw_path)
    if postpublication_sha256 != raw_sha256:
        raise RuntimeError("raw native-ordinal artifact changed after publication")
    print(
        json.dumps(
            {
                "raw": str(raw_path),
                "raw_sha256": raw_sha256,
                "output": str(output_path),
                "output_sha256": _sha256_bytes(output_payload),
                "report": str(report_path),
                "report_sha256": _sha256_bytes(report_payload),
                "passes": result["passes"],
                "decision": result["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
