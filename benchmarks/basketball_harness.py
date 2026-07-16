"""Shared, forward-only primitives for basketball candidate screens.

Historical frozen runners remain unchanged. New campaigns use this module so
the creator split, player guardrail, fitted metadata, behavior fingerprint,
thread isolation, and reciprocal timing rules cannot drift by copy/paste.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from benchmarks import basketball_guardrails as guardrails
from benchmarks import run_basketball_creator_benchmark as creator


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = creator.DEFAULT_CACHE / "basketball_reference_toy_data.csv"
DEFAULT_TIMING_REPETITIONS = 3
MAX_TIMING_SPREAD_RATIO = 1.20


@dataclass(frozen=True)
class BasketballDataset:
    frame: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    raw_metadata: dict[str, Any]
    processed_metadata: dict[str, Any]
    player_guardrail: guardrails.BasketballGuardrailData
    fold_fingerprint_sha256: str
    fold_test_sizes: list[int]


def load_basketball_dataset(cache_path: Path = DEFAULT_CACHE) -> BasketballDataset:
    """Load and cross-check the immutable creator and player-guardrail views."""
    frame, raw_metadata = creator.load_raw_data(cache_path)
    X, y, processed_metadata = creator.prepare_creator_data(frame)
    player_guardrail = guardrails.prepare_player_guardrail(frame)
    if not X.equals(player_guardrail.X_train) or not y.equals(
        player_guardrail.y_train
    ):
        raise RuntimeError("basketball guardrail changed creator training data")
    fold_digest, fold_sizes = creator.fold_fingerprint(X, y)
    return BasketballDataset(
        frame=frame,
        X=X,
        y=y,
        raw_metadata=raw_metadata,
        processed_metadata=processed_metadata,
        player_guardrail=player_guardrail,
        fold_fingerprint_sha256=fold_digest,
        fold_test_sizes=fold_sizes,
    )


def prediction_sha256(prediction: Any) -> str:
    return guardrails.prediction_sha256(prediction)


def validate_prediction(prediction: Any, expected_rows: int) -> np.ndarray:
    values = np.asarray(prediction, dtype=np.float64)
    if values.ndim != 1 or len(values) != int(expected_rows):
        raise RuntimeError("DarkoFit produced an invalid basketball shape")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("DarkoFit produced non-finite basketball predictions")
    return values


def _phase_timing(core: Any) -> dict[str, float]:
    timing = getattr(core, "timing_", {}) or {}
    result = {str(key): float(value) for key, value in timing.items()}
    if any(not math.isfinite(value) or value < 0.0 for value in result.values()):
        raise RuntimeError("fitted phase timing contains an invalid value")
    return result


def _core_metadata(core: Any) -> dict[str, Any]:
    training = dict(getattr(core, "training_metadata_", {}) or {})
    return {
        "iterations_requested": int(
            training.get("iterations_requested", getattr(core, "iterations_", 0))
        ),
        "iterations_attempted": int(
            training.get(
                "iterations_attempted", getattr(core, "iterations_attempted_", 0)
            )
        ),
        "rounds_completed": int(
            training.get("rounds_completed", getattr(core, "rounds_completed_", 0))
        ),
        "rounds_retained": int(
            training.get("rounds_retained", len(getattr(core, "trees_", ())))
        ),
        "stop_reason": str(
            training.get("stop_reason", getattr(core, "stop_reason_", "unknown"))
        ),
        "phase_seconds": _phase_timing(core),
    }


def extract_fit_metadata(model: Any) -> dict[str, Any]:
    """Extract the common fitted route, stopping, and refit observability."""
    core = model.model_
    selection_core = getattr(model, "selection_model_", None)
    linear_active = bool(getattr(model, "linear_residual_active_", False))
    linear_leaf_metadata = dict(
        getattr(core, "auto_params_", {}).get("linear_leaves", {}) or {}
    )
    linear_leaf_active = bool(linear_leaf_metadata.get("active", False))
    if linear_active:
        selected_lane = "linear_residual"
    elif linear_leaf_active:
        selected_lane = "linear_leaves"
    else:
        selected_lane = "boosting"
    return {
        "best_iteration": int(model.best_n_estimators_),
        "fitted_tree_count": int(model.n_estimators_),
        "resolved_learning_rate": float(model.learning_rate_),
        "requested_tree_mode": str(model.tree_mode),
        "selected_tree_mode": str(core.tree_mode_),
        "selected_lane": selected_lane,
        "linear_residual_active": linear_active,
        "linear_leaves_active": linear_leaf_active,
        "linear_leaves": linear_leaf_metadata,
        "resolved_thread_count": int(core.n_threads_),
        "refit": bool(getattr(model, "refit_", False)),
        "refit_strategy": getattr(model, "refit_strategy_", None),
        "final_fit": _core_metadata(core),
        "selection_fit": (
            None if selection_core is None else _core_metadata(selection_core)
        ),
        "selection_early_stopping_rounds": (
            None
            if selection_core is None
            or selection_core.early_stopping_rounds_ is None
            else int(selection_core.early_stopping_rounds_)
        ),
        "final_early_stopping_rounds": (
            None
            if core.early_stopping_rounds_ is None
            else int(core.early_stopping_rounds_)
        ),
    }


def sum_phase_times(
    fits: Iterable[dict[str, Any]], fit_name: str
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for record in fits:
        fit = record["fit_metadata"].get(fit_name)
        if fit is None:
            continue
        for phase, seconds in fit["phase_seconds"].items():
            totals[phase] = totals.get(phase, 0.0) + float(seconds)
    return dict(sorted(totals.items()))


def reciprocal_schedule(
    first: str,
    second: str,
    repetitions: int = DEFAULT_TIMING_REPETITIONS,
) -> tuple[tuple[str, str], ...]:
    if not first or not second or first == second:
        raise ValueError("reciprocal timing requires two distinct names")
    if int(repetitions) < 2:
        raise ValueError("reciprocal timing requires at least two repetitions")
    return tuple(
        (first, second) if block % 2 == 0 else (second, first)
        for block in range(int(repetitions))
    )


def timing_summary(
    values: Iterable[float],
    repetitions: int = DEFAULT_TIMING_REPETITIONS,
) -> dict[str, Any]:
    values = [float(value) for value in values]
    if len(values) != int(repetitions):
        raise RuntimeError(
            f"basketball timing requires exactly {int(repetitions)} repetitions"
        )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise RuntimeError("basketball timing values must be positive and finite")
    minimum = min(values)
    maximum = max(values)
    return {
        "repetitions": len(values),
        "values_seconds": values,
        "minimum_seconds": minimum,
        "median_seconds": float(statistics.median(values)),
        "maximum_seconds": maximum,
        "maximum_over_minimum": maximum / minimum,
        "stable": maximum / minimum <= MAX_TIMING_SPREAD_RATIO,
    }


_TIMING_SUMMARY_KEYS = frozenset(
    {
        "repetitions",
        "values_seconds",
        "minimum_seconds",
        "median_seconds",
        "maximum_seconds",
        "maximum_over_minimum",
        "stable",
    }
)


def _is_timing_summary(value: Any) -> bool:
    return isinstance(value, dict) and value.keys() == _TIMING_SUMMARY_KEYS


def _without_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if "_seconds" not in key
            and key not in {"worker_stdout", "worker_stderr"}
            and not _is_timing_summary(item)
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def behavior_fingerprint(payload: dict[str, Any]) -> str:
    """Hash behavior and fitted metadata while excluding noisy timings."""
    encoded = json.dumps(
        _without_timing(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def worker_environment(threads: int) -> dict[str, str]:
    """Return the isolated full-machine environment for a fresh worker."""
    threads = int(threads)
    if threads < 1:
        raise ValueError("basketball worker threads must be positive")
    environment = os.environ.copy()
    for key in tuple(environment):
        if key == "ENABLE_IPC" or key.startswith(creator.EXECUTION_ENV_PREFIXES):
            environment.pop(key)
    for key in creator.THREAD_LIMIT_ENV_KEYS:
        environment[key] = str(threads)
    environment.update(
        {
            "CHIMERABOOST_WARMUP": "0",
            "NUMBA_DISABLE_JIT": "0",
            "ENABLE_IPC": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    for key in (
        "NUMBA_CPU_NAME",
        "NUMBA_CPU_FEATURES",
        "NUMBA_THREADING_LAYER",
        "NUMBA_CACHE_DIR",
        "JOBLIB_START_METHOD",
        "JOBLIB_TEMP_FOLDER",
    ):
        environment.pop(key, None)
    return environment
