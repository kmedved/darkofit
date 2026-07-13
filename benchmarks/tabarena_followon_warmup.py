"""Expanded, attested warmup for the isolated TabArena follow-on screen.

The cap-horizon warmup intentionally covers only CatBoost-mode regression.
This screen also times automatic mode selection and the linear-residual lane,
so it must compile the LightGBM, hybrid, ridge, and four-permutation target-
statistic paths before ``TabArenaContext.run_jobs`` starts its measured work.
"""

from __future__ import annotations

import hashlib
import json
import operator
import time
from collections import Counter
from typing import Any

import numba
import numpy as np

from darkofit import DarkoRegressor
from darkofit.flat_model import _PARALLEL_MIN_ROWS, flat_predict_preferred

try:
    from benchmarks.tabarena_warmup import _make_warmup_data, _prediction_batch
except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
    from tabarena_warmup import _make_warmup_data, _prediction_batch


WARMUP_SCHEMA_VERSION = 1
WARMUP_KIND = "darkofit_tabarena_followon_screen_warmup"
WARMUP_STAGE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "catboost_numeric",
        "input_kind": "numeric",
        "tree_mode": "catboost",
        "linear_residual": False,
        "ts_permutations": 1,
    },
    {
        "name": "catboost_categorical",
        "input_kind": "categorical",
        "tree_mode": "catboost",
        "linear_residual": False,
        "ts_permutations": 1,
    },
    {
        "name": "lightgbm_numeric",
        "input_kind": "numeric",
        "tree_mode": "lightgbm",
        "linear_residual": False,
        "ts_permutations": 1,
    },
    {
        "name": "lightgbm_categorical",
        "input_kind": "categorical",
        "tree_mode": "lightgbm",
        "linear_residual": False,
        "ts_permutations": 1,
    },
    {
        "name": "hybrid_numeric",
        "input_kind": "numeric",
        "tree_mode": "hybrid",
        "linear_residual": False,
        "ts_permutations": 1,
    },
    {
        "name": "hybrid_categorical",
        "input_kind": "categorical",
        "tree_mode": "hybrid",
        "linear_residual": False,
        "ts_permutations": 1,
    },
    {
        "name": "linear_residual_numeric",
        "input_kind": "numeric",
        "tree_mode": "catboost",
        "linear_residual": True,
        "ts_permutations": 1,
    },
    {
        "name": "linear_residual_categorical",
        "input_kind": "categorical",
        "tree_mode": "catboost",
        "linear_residual": True,
        "ts_permutations": 1,
    },
    {
        "name": "catboost_categorical_ts4",
        "input_kind": "categorical",
        "tree_mode": "catboost",
        "linear_residual": False,
        "ts_permutations": 4,
    },
)
EXPECTED_WARMUP_COUNTS = {
    "input_kind": {"categorical": 5, "numeric": 4},
    "requested_tree_mode": {"catboost": 5, "hybrid": 2, "lightgbm": 2},
    "resolved_tree_mode": {"catboost": 5, "hybrid": 2, "lightgbm": 2},
    "selected_lane": {"boosting": 7, "linear_residual": 2},
    "ts_permutations": {"1": 8, "4": 1},
}

WARMUP_BASE_CONFIG: dict[str, Any] = {
    "iterations": 5,
    "loss": "RMSE",
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "max_bins": 128,
    "learning_rate": 0.1,
    "ordered_boosting": "auto",
    "sampling": "uniform",
    "early_stopping": False,
    "use_best_model": False,
    "diagnostic_warnings": "never",
    "random_state": 20_260_713,
}
_PREDICTION_CASES = (
    ("serial_subthreshold", _PARALLEL_MIN_ROWS - 1),
    ("parallel_at_threshold", _PARALLEL_MIN_ROWS),
)


def _elapsed_seconds(start_ns: int) -> float:
    return float(time.monotonic_ns() - start_ns) / 1_000_000_000.0


def _prediction_fingerprint(prediction: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(prediction, dtype="<f8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _counter(values) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _run_stage(
    spec: dict[str, Any],
    data: tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int] | None],
    *,
    thread_count: int,
) -> dict[str, Any]:
    _, X_train, y_train, X_validation, y_validation, cat_features = data
    config = {
        **WARMUP_BASE_CONFIG,
        "tree_mode": spec["tree_mode"],
        "linear_residual": spec["linear_residual"],
        "ts_permutations": spec["ts_permutations"],
        "thread_count": thread_count,
    }
    fit_started_ns = time.monotonic_ns()
    model = DarkoRegressor(**config)
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_validation, y_validation),
    )
    fit_seconds = _elapsed_seconds(fit_started_ns)

    fitted = model.model_
    flat = fitted._flat_ensemble()
    flat_selected = flat is not None and flat_predict_preferred(flat)
    batches = []
    for case_name, n_rows in _PREDICTION_CASES:
        prediction_input = _prediction_batch(X_validation, n_rows)
        predict_started_ns = time.monotonic_ns()
        prediction = np.asarray(model.predict(prediction_input), dtype=np.float64)
        predict_seconds = _elapsed_seconds(predict_started_ns)
        if prediction.shape != (n_rows,) or not np.all(np.isfinite(prediction)):
            raise RuntimeError(f"follow-on warmup prediction failed for {spec['name']}")
        route = "tree_loop"
        if flat_selected:
            route = "flat_parallel" if n_rows >= _PARALLEL_MIN_ROWS else "flat_serial"
        batches.append(
            {
                "name": case_name,
                "route": route,
                "input_shape": [int(value) for value in prediction_input.shape],
                "prediction_shape": [int(value) for value in prediction.shape],
                "predict_seconds": predict_seconds,
                "prediction_sha256": _prediction_fingerprint(prediction),
            }
        )

    preprocessor = fitted.prep_
    encoders = list(getattr(preprocessor, "encoders_", ()))
    linear_active = bool(getattr(model, "linear_residual_active_", False))
    selected_lane = "linear_residual" if linear_active else "boosting"
    if linear_active is not bool(spec["linear_residual"]):
        raise RuntimeError(f"follow-on warmup lane mismatch for {spec['name']}")
    return {
        "name": spec["name"],
        "input_kind": spec["input_kind"],
        "categorical_features": list(cat_features or []),
        "config": config,
        "train_rows": int(X_train.shape[0]),
        "validation_rows": int(X_validation.shape[0]),
        "fit_seconds": fit_seconds,
        "iterations_fitted": int(model.n_estimators_),
        "tree_depths": [int(tree.depth) for tree in fitted.trees_],
        "requested_tree_mode": str(spec["tree_mode"]),
        "resolved_tree_mode": str(fitted.tree_mode_),
        "selected_lane": selected_lane,
        "linear_residual_active": linear_active,
        "resolved_learning_rate": float(model.learning_rate_),
        "resolved_ordered_boosting": bool(fitted.ordered_boosting_),
        "resolved_thread_count": int(fitted.n_threads_),
        "resolved_target_encoding_mode": str(preprocessor.target_encoding_mode),
        "resolved_include_cat_codes": bool(preprocessor.include_cat_codes),
        "resolved_ts_permutations": int(preprocessor.ts_permutations),
        "encoder_modes": [str(encoder.mode) for encoder in encoders],
        "encoder_ts_permutations": [
            int(encoder.ts_permutations) for encoder in encoders
        ],
        "flat_ensemble_type": type(flat).__name__,
        "flat_prediction_router_selected": bool(flat_selected),
        "prediction_parallel_min_rows": int(_PARALLEL_MIN_ROWS),
        "prediction_batches": batches,
    }


def warmup_tabarena_followon_screen(*, thread_count: int) -> dict[str, Any]:
    """Warm and describe every compiled model path timed by the screen."""
    if isinstance(thread_count, (bool, np.bool_)):
        raise TypeError("thread_count must be a positive integer")
    try:
        thread_count = operator.index(thread_count)
    except TypeError as exc:
        raise TypeError("thread_count must be a positive integer") from exc
    if thread_count < 1:
        raise ValueError("thread_count must be at least 1")

    data_by_kind = {stage[0]: stage for stage in _make_warmup_data()}
    original_threads = numba.get_num_threads()
    started_ns = time.monotonic_ns()
    stages = []
    try:
        for spec in WARMUP_STAGE_SPECS:
            stages.append(
                _run_stage(spec, data_by_kind[spec["input_kind"]], thread_count=thread_count)
            )
    finally:
        numba.set_num_threads(original_threads)

    counts = {
        "input_kind": _counter(stage["input_kind"] for stage in stages),
        "requested_tree_mode": _counter(
            stage["requested_tree_mode"] for stage in stages
        ),
        "resolved_tree_mode": _counter(stage["resolved_tree_mode"] for stage in stages),
        "selected_lane": _counter(stage["selected_lane"] for stage in stages),
        "ts_permutations": _counter(
            stage["resolved_ts_permutations"] for stage in stages
        ),
    }
    if counts != EXPECTED_WARMUP_COUNTS:
        raise RuntimeError("follow-on warmup did not cover the frozen path counts")
    metadata = {
        "schema_version": WARMUP_SCHEMA_VERSION,
        "kind": WARMUP_KIND,
        "clock": "time.monotonic_ns",
        "duration_seconds": _elapsed_seconds(started_ns),
        "thread_count": thread_count,
        "stage_count": len(stages),
        "counts": counts,
        "stages": stages,
    }
    json.dumps(metadata, allow_nan=False)
    return metadata


__all__ = [
    "EXPECTED_WARMUP_COUNTS",
    "WARMUP_KIND",
    "WARMUP_BASE_CONFIG",
    "WARMUP_SCHEMA_VERSION",
    "WARMUP_STAGE_SPECS",
    "warmup_tabarena_followon_screen",
]
