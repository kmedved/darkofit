#!/usr/bin/env python3
"""Run the frozen basketball split-conformal quantile opportunity screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import ShuffleSplit


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_TEXT = str(REPO_ROOT)
if REPO_ROOT_TEXT in sys.path:
    sys.path.remove(REPO_ROOT_TEXT)
sys.path.insert(0, REPO_ROOT_TEXT)

import darkofit as darkofit_package  # noqa: E402
from benchmarks import basketball_harness as harness  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402


ALPHAS = (0.1, 0.9)
ALPHA_KEYS = ("0.1", "0.9")
RANDOM_STATE = 4
THREADS = 18
CALIBRATION_FRACTION = 0.10
MIN_STRICT_FOLD_WINS = 6
MAX_FOLD_PINBALL_RATIO = 1.02
MAX_WIDTH_RATIO = 1.25
PRE_PROTOCOL_COMMIT = "542e28177a4f9a8ab7fe734359e4d7647dce18d9"
EXPECTED_DARKOFIT_TREE = "1a60b529c5f5d09920d81338406b491fb7275e3a"
PROTOCOL_PATH = (
    REPO_ROOT / "benchmarks" / "basketball_quantile_calibration_protocol.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "64586086c8a5dc55d903a019602c36cf4abc289593ff9c5293611d3d1b125a97"
)
EXPECTED_NORMALIZED_RUNNER_SHA256 = (
    "017ad243cce5511bb78ad8c32ef487a160f7a8d242b3f41cacd51c72f2e67ab6"
)
EXPECTED_SUPPORT_SHA256 = {
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
DEFAULT_OUTPUT = (
    REPO_ROOT / "benchmarks" / "basketball_quantile_calibration.json"
)
EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "platform": "macOS-26.5.2-arm64-arm-64bit",
    "machine": "arm64",
    "cpu_brand": "Apple M5 Max",
    "logical_cpu_count": 18,
    "dependencies": {
        "numpy": "2.4.6",
        "numba": "0.66.0",
        "llvmlite": "0.48.0",
        "pandas": "3.0.3",
        "scikit-learn": "1.9.0",
        "scipy": "1.18.0",
        "joblib": "1.5.3",
        "threadpoolctl": "3.6.0",
    },
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalized_runner_sha256() -> str:
    payload = Path(__file__).resolve().read_bytes()
    marker = b'EXPECTED_NORMALIZED_RUNNER_SHA256 = (\n    "'
    start = payload.find(marker)
    if start < 0:
        raise RuntimeError("runner self-hash marker is missing")
    value_start = start + len(marker)
    value_end = value_start + 64
    if (
        payload[value_end : value_end + 2] != b'"\n'
        or any(byte not in b"0123456789abcdef" for byte in payload[value_start:value_end])
    ):
        raise RuntimeError("runner self-hash field is malformed")
    normalized = payload[:value_start] + (b"0" * 64) + payload[value_end:]
    return _sha256_bytes(normalized)


def support_sha256() -> dict[str, str]:
    return {
        relative: _sha256_bytes((REPO_ROOT / relative).read_bytes())
        for relative in EXPECTED_SUPPORT_SHA256
    }


def array_sha256(values: Any, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return _sha256_bytes(array.tobytes())


def index_sha256(values: Any) -> str:
    return array_sha256(values, "<i8")


def pinball_loss(y_true: Any, prediction: Any, alpha: float) -> float:
    target = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    if target.ndim != 1 or predicted.shape != target.shape or target.size == 0:
        raise ValueError("pinball inputs must be equal nonempty vectors")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(predicted)):
        raise ValueError("pinball inputs must be finite")
    residual = target - predicted
    return float(
        np.mean(np.maximum(float(alpha) * residual, (float(alpha) - 1.0) * residual))
    )


def quantile_offset(
    y_calibration: Any,
    calibration_prediction: Any,
    alpha: float,
) -> tuple[float, int]:
    target = np.asarray(y_calibration, dtype=np.float64)
    predicted = np.asarray(calibration_prediction, dtype=np.float64)
    if target.ndim != 1 or predicted.shape != target.shape or target.size == 0:
        raise ValueError("calibration inputs must be equal nonempty vectors")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(predicted)):
        raise ValueError("calibration inputs must be finite")
    if not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    residual = np.sort(target - predicted)
    rank = min(int(math.ceil((len(residual) + 1) * float(alpha))), len(residual))
    return float(residual[rank - 1]), int(rank)


def calibration_split(n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    n_rows = int(n_rows)
    if n_rows < 2:
        raise ValueError("calibration split requires at least two rows")
    fit_indices, calibration_indices = next(
        ShuffleSplit(
            n_splits=1,
            test_size=CALIBRATION_FRACTION,
            random_state=RANDOM_STATE,
        ).split(np.empty((n_rows, 0)))
    )
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    calibration_indices = np.asarray(calibration_indices, dtype=np.int64)
    combined = np.concatenate((fit_indices, calibration_indices))
    if (
        len(np.intersect1d(fit_indices, calibration_indices)) != 0
        or len(combined) != n_rows
        or not np.array_equal(np.sort(combined), np.arange(n_rows))
    ):
        raise RuntimeError("internal calibration split is not a partition")
    return fit_indices, calibration_indices


def _arm_metrics(
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    tails = {}
    for alpha, key in zip(ALPHAS, ALPHA_KEYS):
        prediction = np.asarray(predictions[key], dtype=np.float64)
        if prediction.shape != target.shape or not np.all(np.isfinite(prediction)):
            raise RuntimeError("quantile prediction is invalid")
        coverage = float(np.mean(target <= prediction))
        tails[key] = {
            "alpha": float(alpha),
            "coverage": coverage,
            "absolute_coverage_error": abs(coverage - float(alpha)),
            "pinball_loss": pinball_loss(target, prediction, alpha),
            "prediction_sha256": harness.prediction_sha256(prediction),
            "predictions": [float(value) for value in prediction],
        }
    lower = predictions["0.1"]
    upper = predictions["0.9"]
    crossing = np.asarray(lower > upper, dtype=np.bool_)
    interval_coverage = float(np.mean((target >= lower) & (target <= upper)))
    return {
        "tails": tails,
        "interval": {
            "nominal_coverage": 0.8,
            "coverage": interval_coverage,
            "absolute_coverage_error": abs(interval_coverage - 0.8),
            "mean_width": float(np.mean(upper - lower)),
            "crossing_count": int(np.count_nonzero(crossing)),
            "crossing_rate": float(np.mean(crossing)),
            "summed_pinball_loss": float(
                tails["0.1"]["pinball_loss"] + tails["0.9"]["pinball_loss"]
            ),
        },
    }


def score_view(
    y_true: Any,
    raw_predictions: dict[str, Any],
    offsets: dict[str, float],
) -> dict[str, Any]:
    target = np.asarray(y_true, dtype=np.float64)
    if target.ndim != 1 or target.size == 0 or not np.all(np.isfinite(target)):
        raise ValueError("evaluation target must be a finite nonempty vector")
    control = {
        key: harness.validate_prediction(raw_predictions[key], len(target))
        for key in ALPHA_KEYS
    }
    candidate = {
        key: harness.validate_prediction(
            control[key] + float(offsets[key]), len(target)
        )
        for key in ALPHA_KEYS
    }
    return {
        "rows": int(len(target)),
        "target_sha256": array_sha256(target, "<f8"),
        "control": _arm_metrics(target, control),
        "candidate": _arm_metrics(target, candidate),
    }


def _numeric_state_is_finite(value: Any, seen: set[int] | None = None) -> bool:
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (str, bytes, bool)):
        return True
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, (int, np.integer)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fci":
            return bool(np.all(np.isfinite(value)))
        if value.dtype.kind == "O":
            return all(_numeric_state_is_finite(item, seen) for item in value.flat)
        return True
    identity = id(value)
    if identity in seen:
        return True
    seen.add(identity)
    if isinstance(value, dict):
        return all(_numeric_state_is_finite(item, seen) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return all(_numeric_state_is_finite(item, seen) for item in value)
    if value.__class__.__module__.startswith("darkofit"):
        if hasattr(value, "__dict__") and not _numeric_state_is_finite(
            vars(value), seen
        ):
            return False
        for cls in value.__class__.__mro__:
            slots = getattr(cls, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if slot in {"__dict__", "__weakref__"} or not hasattr(value, slot):
                    continue
                if not _numeric_state_is_finite(getattr(value, slot), seen):
                    return False
    return True


def _validate_fitted_model(model: Any) -> dict[str, Any]:
    metadata = harness.extract_fit_metadata(model)
    if metadata["selected_lane"] != "boosting":
        raise RuntimeError("quantile fit unexpectedly selected another lane")
    if metadata["selected_tree_mode"] != "catboost":
        raise RuntimeError("quantile fit unexpectedly changed tree mode")
    if int(metadata["resolved_thread_count"]) != THREADS:
        raise RuntimeError("quantile fit did not resolve exactly 18 threads")
    if metadata["refit"] or metadata["selection_fit"] is not None:
        raise RuntimeError("quantile fit unexpectedly selected or refit")
    if metadata["final_early_stopping_rounds"] is not None:
        raise RuntimeError("quantile fit unexpectedly enabled early stopping")
    if not _numeric_state_is_finite(model):
        raise RuntimeError("quantile model contains non-finite numeric state")
    return metadata


def _params_sha256(model: Any) -> str:
    payload = json.dumps(
        model.get_params(deep=False),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def fit_quantile_pair(
    X_train,
    y_train,
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    X_evaluation,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
    raw_predictions: dict[str, np.ndarray] = {}
    offsets: dict[str, float] = {}
    models = {}
    for alpha, key in zip(ALPHAS, ALPHA_KEYS):
        model = darkofit_package.DarkoRegressor(
            loss="Quantile",
            alpha=alpha,
            random_state=RANDOM_STATE,
            thread_count=THREADS,
            diagnostic_warnings="never",
        )
        params_sha256 = _params_sha256(model)
        started = time.perf_counter_ns()
        model.fit(X_train.iloc[fit_indices], y_train.iloc[fit_indices])
        fit_seconds = (time.perf_counter_ns() - started) / 1e9
        calibration_prediction = harness.validate_prediction(
            model.predict(X_train.iloc[calibration_indices]),
            len(calibration_indices),
        )
        raw_prediction = harness.validate_prediction(
            model.predict(X_evaluation), len(X_evaluation)
        )
        offset, rank = quantile_offset(
            y_train.iloc[calibration_indices], calibration_prediction, alpha
        )
        residual = (
            y_train.iloc[calibration_indices].to_numpy(dtype=np.float64)
            - calibration_prediction
        )
        expected_rank = min(
            int(math.ceil((len(residual) + 1) * float(alpha))), len(residual)
        )
        expected_offset = float(np.sort(residual)[expected_rank - 1])
        if rank != expected_rank or offset != expected_offset:
            raise RuntimeError("quantile offset does not match the frozen rank")
        metadata = _validate_fitted_model(model)
        raw_predictions[key] = raw_prediction
        offsets[key] = offset
        models[key] = {
            "alpha": float(alpha),
            "constructor_params_sha256": params_sha256,
            "fit_seconds_directional": float(fit_seconds),
            "fit_metadata": metadata,
            "fitted_numeric_state_finite": True,
            "calibration_rows": int(len(calibration_indices)),
            "calibration_prediction_sha256": harness.prediction_sha256(
                calibration_prediction
            ),
            "calibration_residual_sha256": array_sha256(residual, "<f8"),
            "calibration_rank": int(rank),
            "offset": float(offset),
            "offset_rank_verified_exactly": True,
            "evaluation_prediction_sha256": harness.prediction_sha256(
                raw_prediction
            ),
        }
    return raw_predictions, offsets, models


def _tail_no_worse(
    view: dict[str, Any],
    metric: str,
) -> bool:
    return all(
        float(view["candidate"]["tails"][key][metric])
        <= float(view["control"]["tails"][key][metric])
        for key in ALPHA_KEYS
    )


def analyze_results(
    pooled: dict[str, Any],
    folds: list[dict[str, Any]],
    holdout_views: dict[str, dict[str, Any]],
    *,
    all_finite: bool,
) -> dict[str, Any]:
    if len(folds) != creator.N_SPLITS:
        raise RuntimeError("quantile calibration result must contain ten folds")
    every_view = [
        pooled,
        *(fold["scores"] for fold in folds),
        *holdout_views.values(),
    ]
    fold_control = np.asarray(
        [fold["scores"]["control"]["interval"]["summed_pinball_loss"] for fold in folds],
        dtype=np.float64,
    )
    fold_candidate = np.asarray(
        [
            fold["scores"]["candidate"]["interval"]["summed_pinball_loss"]
            for fold in folds
        ],
        dtype=np.float64,
    )
    fold_ratios = fold_candidate / fold_control
    team = holdout_views["overlap_exposed_team_holdout"]
    cold = holdout_views["cold_player_subset"]
    gates = {
        "all_values_finite": bool(all_finite),
        "crossing_count_no_regression_every_view": all(
            view["candidate"]["interval"]["crossing_count"]
            <= view["control"]["interval"]["crossing_count"]
            for view in every_view
        ),
        "pooled_tail_pinball_no_regression": _tail_no_worse(
            pooled, "pinball_loss"
        ),
        "pooled_tail_coverage_no_regression": _tail_no_worse(
            pooled, "absolute_coverage_error"
        ),
        "pooled_interval_coverage_no_regression": (
            pooled["candidate"]["interval"]["absolute_coverage_error"]
            <= pooled["control"]["interval"]["absolute_coverage_error"]
        ),
        "pooled_summed_pinball_no_regression": (
            pooled["candidate"]["interval"]["summed_pinball_loss"]
            <= pooled["control"]["interval"]["summed_pinball_loss"]
        ),
        "strict_fold_win_breadth": int(np.count_nonzero(fold_candidate < fold_control))
        >= MIN_STRICT_FOLD_WINS,
        "worst_fold_pinball_ratio": float(np.max(fold_ratios))
        <= MAX_FOLD_PINBALL_RATIO,
        "team_tail_pinball_no_regression": _tail_no_worse(team, "pinball_loss"),
        "team_tail_coverage_no_regression": _tail_no_worse(
            team, "absolute_coverage_error"
        ),
        "team_summed_pinball_no_regression": (
            team["candidate"]["interval"]["summed_pinball_loss"]
            <= team["control"]["interval"]["summed_pinball_loss"]
        ),
        "team_interval_coverage_no_regression": (
            team["candidate"]["interval"]["absolute_coverage_error"]
            <= team["control"]["interval"]["absolute_coverage_error"]
        ),
        "cold_tail_pinball_no_regression": _tail_no_worse(cold, "pinball_loss"),
        "cold_tail_coverage_no_regression": _tail_no_worse(
            cold, "absolute_coverage_error"
        ),
        "cold_summed_pinball_no_regression": (
            cold["candidate"]["interval"]["summed_pinball_loss"]
            <= cold["control"]["interval"]["summed_pinball_loss"]
        ),
        "cold_interval_coverage_no_regression": (
            cold["candidate"]["interval"]["absolute_coverage_error"]
            <= cold["control"]["interval"]["absolute_coverage_error"]
        ),
        "pooled_width_within_budget": (
            pooled["candidate"]["interval"]["mean_width"]
            <= MAX_WIDTH_RATIO * pooled["control"]["interval"]["mean_width"]
        ),
        "team_width_within_budget": (
            team["candidate"]["interval"]["mean_width"]
            <= MAX_WIDTH_RATIO * team["control"]["interval"]["mean_width"]
        ),
        "cold_width_within_budget": (
            cold["candidate"]["interval"]["mean_width"]
            <= MAX_WIDTH_RATIO * cold["control"]["interval"]["mean_width"]
        ),
    }
    passes = all(gates.values())
    return {
        "strict_fold_wins": int(np.count_nonzero(fold_candidate < fold_control)),
        "strict_fold_losses": int(np.count_nonzero(fold_candidate > fold_control)),
        "fold_ties": int(np.count_nonzero(fold_candidate == fold_control)),
        "fold_candidate_over_control_summed_pinball": [
            float(value) for value in fold_ratios
        ],
        "worst_fold_candidate_over_control_summed_pinball": float(
            np.max(fold_ratios)
        ),
        "quality_gates": gates,
        "passes_all_gates": bool(passes),
        "default_promotion_authorized": False,
        "broader_panel_authorized": False,
        "recommendation": (
            "advance_to_separately_reviewed_default_off_implementation"
            if passes
            else "stop_before_product_implementation"
        ),
    }


def _git(*args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def runtime_state() -> dict[str, Any]:
    machine = creator._machine_details()
    dependencies = {
        package: importlib.metadata.version(package)
        for package in EXPECTED_RUNTIME["dependencies"]
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_brand": machine["cpu_brand"],
        "logical_cpu_count": os.cpu_count(),
        "dependencies": dependencies,
        "python_executable": sys.executable,
    }


def validate_frozen_binding(output: Path) -> dict[str, Any]:
    if output != DEFAULT_OUTPUT:
        raise RuntimeError("formal quantile screen output path is not exact")
    if output.is_symlink() or output.exists():
        raise RuntimeError(f"refusing existing benchmark output: {output}")
    if _sha256_bytes(PROTOCOL_PATH.read_bytes()) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("quantile calibration protocol changed")
    if _normalized_runner_sha256() != EXPECTED_NORMALIZED_RUNNER_SHA256:
        raise RuntimeError("quantile calibration runner changed")
    observed_support = support_sha256()
    if observed_support != EXPECTED_SUPPORT_SHA256:
        raise RuntimeError("quantile calibration support files changed")
    imported_package = Path(darkofit_package.__file__).resolve()
    if not imported_package.is_relative_to(REPO_ROOT):
        raise RuntimeError("DarkoFit was imported outside the frozen repository")
    if _git("rev-parse", "HEAD:darkofit") != EXPECTED_DARKOFIT_TREE:
        raise RuntimeError("DarkoFit package tree changed")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PRE_PROTOCOL_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode:
        raise RuntimeError("source no longer descends from the pre-protocol commit")
    source = creator.git_state(REPO_ROOT)
    if not source["clean"]:
        raise RuntimeError("formal quantile screen requires clean committed source")
    observed_runtime = runtime_state()
    for key in (
        "python",
        "platform",
        "machine",
        "cpu_brand",
        "logical_cpu_count",
    ):
        if observed_runtime[key] != EXPECTED_RUNTIME[key]:
            raise RuntimeError(f"frozen runtime mismatch: {key}")
    if observed_runtime["dependencies"] != EXPECTED_RUNTIME["dependencies"]:
        raise RuntimeError("frozen dependency versions changed")
    return {
        "source": source,
        "runtime": observed_runtime,
        "runner_normalized_sha256": _normalized_runner_sha256(),
        "support_sha256": observed_support,
        "imported_darkofit_path": str(imported_package),
    }


def _atomic_write_new_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o644)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(f"refusing to overwrite benchmark output: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _split_record(
    external_train_indices: np.ndarray,
    fit_relative: np.ndarray,
    calibration_relative: np.ndarray,
) -> dict[str, Any]:
    fit_global = external_train_indices[fit_relative]
    calibration_global = external_train_indices[calibration_relative]
    return {
        "fit_relative_indices": [int(value) for value in fit_relative],
        "fit_relative_indices_sha256": index_sha256(fit_relative),
        "calibration_relative_indices": [
            int(value) for value in calibration_relative
        ],
        "calibration_relative_indices_sha256": index_sha256(calibration_relative),
        "fit_global_indices": [int(value) for value in fit_global],
        "fit_global_indices_sha256": index_sha256(fit_global),
        "calibration_global_indices": [
            int(value) for value in calibration_global
        ],
        "calibration_global_indices_sha256": index_sha256(calibration_global),
        "disjoint": True,
        "covers_external_training_rows_exactly": True,
    }


def run_screen(output: Path, data_cache: Path) -> dict[str, Any]:
    binding = validate_frozen_binding(output)
    source_before = binding["source"]
    dataset = harness.load_basketball_dataset(data_cache)
    n_rows = len(dataset.X)
    pooled_raw = {
        key: np.full(n_rows, np.nan, dtype=np.float64) for key in ALPHA_KEYS
    }
    pooled_candidate = {
        key: np.full(n_rows, np.nan, dtype=np.float64) for key in ALPHA_KEYS
    }
    folds = []
    started = time.perf_counter_ns()
    for fold, (external_train, external_test) in enumerate(
        creator.creator_cv().split(dataset.X, dataset.y)
    ):
        external_train = np.asarray(external_train, dtype=np.int64)
        external_test = np.asarray(external_test, dtype=np.int64)
        fit_relative, calibration_relative = calibration_split(len(external_train))
        fit_global = external_train[fit_relative]
        calibration_global = external_train[calibration_relative]
        raw, offsets, models = fit_quantile_pair(
            dataset.X,
            dataset.y,
            fit_global,
            calibration_global,
            dataset.X.iloc[external_test],
        )
        for key in ALPHA_KEYS:
            pooled_raw[key][external_test] = raw[key]
            pooled_candidate[key][external_test] = raw[key] + offsets[key]
        scores = score_view(dataset.y.iloc[external_test], raw, offsets)
        folds.append(
            {
                "fold": int(fold),
                "external_train_rows": int(len(external_train)),
                "external_test_rows": int(len(external_test)),
                "external_train_indices_sha256": index_sha256(external_train),
                "external_test_indices": [int(value) for value in external_test],
                "external_test_indices_sha256": index_sha256(external_test),
                "internal_split": _split_record(
                    external_train, fit_relative, calibration_relative
                ),
                "offsets": offsets,
                "models": models,
                "scores": scores,
            }
        )
        print(
            f"fold {fold + 1}/{creator.N_SPLITS}: "
            f"control={scores['control']['interval']['summed_pinball_loss']:.6f}, "
            f"candidate={scores['candidate']['interval']['summed_pinball_loss']:.6f}",
            flush=True,
        )
    if any(
        not np.all(np.isfinite(values))
        for group in (pooled_raw, pooled_candidate)
        for values in group.values()
    ):
        raise RuntimeError("pooled creator predictions are incomplete")
    pooled_offsets = {
        key: pooled_candidate[key] - pooled_raw[key] for key in ALPHA_KEYS
    }
    pooled_scores = {
        "rows": int(n_rows),
        "target_sha256": array_sha256(dataset.y, "<f8"),
        "control": _arm_metrics(
            dataset.y.to_numpy(dtype=np.float64), pooled_raw
        ),
        "candidate": _arm_metrics(
            dataset.y.to_numpy(dtype=np.float64), pooled_candidate
        ),
        "fold_specific_offsets": {
            key: [float(value) for value in pooled_offsets[key]]
            for key in ALPHA_KEYS
        },
    }

    guardrail = dataset.player_guardrail
    held_external = np.arange(len(guardrail.X_train), dtype=np.int64)
    held_fit, held_calibration = calibration_split(len(held_external))
    held_raw, held_offsets, held_models = fit_quantile_pair(
        guardrail.X_train,
        guardrail.y_train,
        held_fit,
        held_calibration,
        guardrail.X_holdout,
    )
    cold = np.asarray(guardrail.cold_player_mask, dtype=np.bool_)
    seen = ~cold
    holdout_views = {
        "overlap_exposed_team_holdout": score_view(
            guardrail.y_holdout, held_raw, held_offsets
        ),
        "seen_player_subset": score_view(
            guardrail.y_holdout.iloc[np.flatnonzero(seen)],
            {key: held_raw[key][seen] for key in ALPHA_KEYS},
            held_offsets,
        ),
        "cold_player_subset": score_view(
            guardrail.y_holdout.iloc[np.flatnonzero(cold)],
            {key: held_raw[key][cold] for key in ALPHA_KEYS},
            held_offsets,
        ),
    }
    all_finite = all(
        model["fitted_numeric_state_finite"]
        and math.isfinite(float(model["offset"]))
        for fold in folds
        for model in fold["models"].values()
    ) and all(
        model["fitted_numeric_state_finite"]
        and math.isfinite(float(model["offset"]))
        for model in held_models.values()
    )
    decision = analyze_results(
        pooled_scores, folds, holdout_views, all_finite=all_finite
    )
    elapsed_seconds = (time.perf_counter_ns() - started) / 1e9
    source_after = creator.git_state(REPO_ROOT)
    if source_after != source_before:
        raise RuntimeError("source changed during quantile calibration screen")
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "basketball_split_conformal_quantile_opportunity_screen",
            "sha256": EXPECTED_PROTOCOL_SHA256,
            "basketball_is_primary_fatal_boundary": True,
            "product_code_changed": False,
            "single_lever": "validation_residual_constant_offset",
            "alphas": list(ALPHAS),
            "random_state": RANDOM_STATE,
            "threads_per_fit": THREADS,
            "calibration_fraction": CALIBRATION_FRACTION,
            "minimum_strict_fold_wins": MIN_STRICT_FOLD_WINS,
            "maximum_fold_pinball_ratio": MAX_FOLD_PINBALL_RATIO,
            "maximum_width_ratio": MAX_WIDTH_RATIO,
            "lockbox_data_used": False,
        },
        "source": source_before,
        "executable_source": {
            "runner_normalized_sha256": binding[
                "runner_normalized_sha256"
            ],
            "support_sha256": binding["support_sha256"],
            "imported_darkofit_path": binding["imported_darkofit_path"],
            "darkofit_package_tree": EXPECTED_DARKOFIT_TREE,
        },
        "runtime": binding["runtime"],
        "raw_data": dataset.raw_metadata,
        "processed_data": dataset.processed_metadata,
        "guardrail": guardrail.metadata,
        "creator_cv": {
            "kind": "KFold",
            "n_splits": creator.N_SPLITS,
            "shuffle": False,
            "fold_fingerprint_sha256": dataset.fold_fingerprint_sha256,
            "fold_test_sizes": dataset.fold_test_sizes,
        },
        "folds": folds,
        "pooled_creator": pooled_scores,
        "held_team": {
            "internal_split": _split_record(
                held_external, held_fit, held_calibration
            ),
            "offsets": held_offsets,
            "models": held_models,
            "views": holdout_views,
        },
        "decision": decision,
        "directional_total_elapsed_seconds": float(elapsed_seconds),
        "literal_chimeraboost_code_copied": False,
    }
    _atomic_write_new_bytes(
        output,
        (
            json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
    )
    print(f"decision: {decision['recommendation']}")
    print(f"wrote {output}")
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-cache", type=Path, default=harness.DEFAULT_CACHE)
    args = parser.parse_args(argv)
    args.output = creator._absolute_lexical_path(args.output)
    args.data_cache = creator._absolute_lexical_path(args.data_cache)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_screen(args.output, args.data_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
