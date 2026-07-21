"""Protocol-matched warmup for the TabArena regression campaign.

DarkoFit compiles or loads several Numba kernels on the first fit and first
large prediction in a process.  Those one-time costs must not be charged to a
single benchmark child.  :func:`warmup_tabarena_regression` runs two small,
deterministic fits that match the frozen horizon experiment's scalar-RMSE
configuration: one purely numeric lane and one mixed numeric/categorical lane.

This helper is intentionally benchmark-local.  It is not imported by the
``darkofit`` package, has no environment-variable hook, and must be called
explicitly outside the measured interval.  It covers preprocessing, an
explicit validation set, five CatBoost-mode boosting rounds, validation
prediction, and, for the campaign's multithreaded configuration, the flat
predictor's serial sub-threshold and parallel large-batch routes.  A one-thread
call instead records the public predictor's tree-loop route.  It is not a
promise to compile every optional DarkoFit mode.
"""

from __future__ import annotations

import hashlib
import json
import operator
import time
from typing import Any

import numba
import numpy as np

from darkofit import DarkoRegressor
from darkofit.flat_model import _PARALLEL_MIN_ROWS, flat_predict_preferred


_SEED = 20260713
_TRAIN_ROWS = 2048
_VALIDATION_ROWS = 512
_NUMERIC_FEATURES = 12

# Derive both route probes from the production dispatch boundary.  The
# sub-threshold case compiles the serial flat kernel; the boundary case
# compiles the parallel flat kernel when more than one Numba thread is active.
if _PARALLEL_MIN_ROWS < 2:  # pragma: no cover - production invariant guard
    raise RuntimeError("flat prediction parallel threshold must be at least 2")
_SERIAL_PREDICTION_ROWS = _PARALLEL_MIN_ROWS - 1
_PARALLEL_PREDICTION_ROWS = _PARALLEL_MIN_ROWS
_PREDICTION_CASES = (
    ("serial_subthreshold", _SERIAL_PREDICTION_ROWS),
    ("parallel_at_threshold", _PARALLEL_PREDICTION_ROWS),
)

# Keep this explicit instead of inheriting estimator defaults: warmup must stay
# aligned with the frozen 1,000-vs-10,000-round cap experiment if defaults move.
_MODEL_CONFIG: dict[str, Any] = {
    "iterations": 5,
    "loss": "RMSE",
    "tree_mode": "catboost",
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "max_bins": 128,
    "learning_rate": 0.1,
    "ts_permutations": 1,
    "ordered_boosting": "auto",
    "sampling": "uniform",
    "linear_residual": False,
    "early_stopping": False,
    "use_best_model": False,
    "diagnostic_warnings": "never",
    "random_state": _SEED,
}


def _elapsed_seconds(start_ns: int) -> float:
    return float(time.monotonic_ns() - start_ns) / 1_000_000_000.0


def _normalize_thread_count(thread_count: int | None) -> int | None:
    if thread_count is None:
        return None
    if isinstance(thread_count, (bool, np.bool_)):
        raise TypeError("thread_count must be a positive integer or None")
    try:
        value = operator.index(thread_count)
    except TypeError as exc:
        raise TypeError(
            "thread_count must be a positive integer or None"
        ) from exc
    if value < 1:
        raise ValueError("thread_count must be at least 1")
    return int(value)


def _make_warmup_data() -> list[
    tuple[
        str,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[int] | None,
    ]
]:
    """Build deterministic numeric and mixed categorical regression lanes."""
    rng = np.random.default_rng(_SEED)
    n_rows = _TRAIN_ROWS + _VALIDATION_ROWS
    numeric = rng.normal(size=(n_rows, _NUMERIC_FEATURES))
    noise = rng.normal(scale=0.05, size=n_rows)
    base_target = (
        1.8 * numeric[:, 0]
        - 0.9 * numeric[:, 1]
        + 0.4 * numeric[:, 2] * numeric[:, 3]
        + np.sin(numeric[:, 4])
        + noise
    )

    split = _TRAIN_ROWS
    numeric_lane = (
        "numeric",
        np.ascontiguousarray(numeric[:split]),
        np.ascontiguousarray(base_target[:split]),
        np.ascontiguousarray(numeric[split:]),
        np.ascontiguousarray(base_target[split:]),
        None,
    )

    category_codes = rng.integers(0, 11, size=n_rows)
    categories = np.asarray(
        [f"category-{code}" for code in category_codes], dtype=object
    )
    mixed = np.empty((n_rows, _NUMERIC_FEATURES + 1), dtype=object)
    mixed[:, :_NUMERIC_FEATURES] = numeric
    mixed[:, _NUMERIC_FEATURES] = categories
    category_effects = np.linspace(-0.75, 0.75, 11)
    categorical_target = base_target + category_effects[category_codes]
    categorical_lane = (
        "categorical",
        mixed[:split].copy(),
        np.ascontiguousarray(categorical_target[:split]),
        mixed[split:].copy(),
        np.ascontiguousarray(categorical_target[split:]),
        [_NUMERIC_FEATURES],
    )
    return [numeric_lane, categorical_lane]


def _prediction_batch(X_validation: np.ndarray, n_rows: int) -> np.ndarray:
    repeats = (n_rows + X_validation.shape[0] - 1) // X_validation.shape[0]
    return np.tile(X_validation, (repeats, 1))[:n_rows]


def _prediction_fingerprint(prediction: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(prediction, dtype="<f8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _prediction_route(*, flat_router_selected: bool, n_rows: int) -> str:
    if not flat_router_selected:
        return "tree_loop"
    if n_rows >= _PARALLEL_MIN_ROWS:
        return "flat_parallel"
    return "flat_serial"


def _flat_router_selected(flat: Any, n_rows: int, fitted_threads: int, tree_mode: str):
    """Evaluate production dispatch under the fitted model's thread mask."""
    if flat is None:
        return False
    previous_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(int(fitted_threads))
        return bool(flat_predict_preferred(flat, n_rows, tree_mode))
    finally:
        numba.set_num_threads(previous_threads)


def _run_stage(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    cat_features: list[int] | None,
    *,
    thread_count: int | None,
) -> dict[str, Any]:
    params = dict(_MODEL_CONFIG)
    params["thread_count"] = thread_count

    fit_started_ns = time.monotonic_ns()
    model = DarkoRegressor(**params)
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_validation, y_validation),
    )
    fit_seconds = _elapsed_seconds(fit_started_ns)

    fitted = model.model_
    flat = fitted._flat_ensemble()

    prediction_batches: list[dict[str, Any]] = []
    selected_routes = []
    for case_name, n_rows in _PREDICTION_CASES:
        prediction_input = _prediction_batch(X_validation, n_rows)
        router_selected = _flat_router_selected(
            flat,
            n_rows,
            fitted.n_threads_,
            fitted.tree_mode_,
        )
        predict_started_ns = time.monotonic_ns()
        prediction = model.predict(prediction_input)
        predict_seconds = _elapsed_seconds(predict_started_ns)
        selected_routes.append(router_selected)
        prediction_batches.append(
            {
                "name": case_name,
                "route": _prediction_route(
                    flat_router_selected=router_selected,
                    n_rows=n_rows,
                ),
                "input_shape": [int(value) for value in prediction_input.shape],
                "prediction_shape": [int(value) for value in prediction.shape],
                "predict_seconds": predict_seconds,
                "prediction_sha256": _prediction_fingerprint(prediction),
            }
        )

    return {
        "name": name,
        "categorical_features": list(cat_features or []),
        "train_rows": int(X_train.shape[0]),
        "validation_rows": int(X_validation.shape[0]),
        "fit_seconds": fit_seconds,
        "iterations_fitted": int(model.n_estimators_),
        "tree_depths": [int(tree.depth) for tree in fitted.trees_],
        "resolved_learning_rate": float(model.learning_rate_),
        "resolved_tree_mode": str(fitted.tree_mode_),
        "resolved_ordered_boosting": bool(fitted.ordered_boosting_),
        "resolved_thread_count": int(fitted.n_threads_),
        "flat_ensemble_type": type(flat).__name__,
        "flat_prediction_router_selected": any(selected_routes),
        "prediction_parallel_min_rows": int(_PARALLEL_MIN_ROWS),
        "prediction_batches": prediction_batches,
    }


def warmup_tabarena_regression(
    *, thread_count: int | None = None
) -> dict[str, Any]:
    """Warm the frozen TabArena scalar-regression benchmark paths.

    Call this once in each benchmark process before starting a timed child fit.
    Pass the same CPU count used by the measured models; both row-count routes
    are exercised under that thread mask.  The helper uses only a local
    ``numpy.random.Generator`` and restores the caller's Numba thread mask.

    Parameters
    ----------
    thread_count : int or None, default None
        Positive Numba thread count for the warmup fits.  Two or more resolved
        threads reach both flat-prediction routes; one thread follows the
        public tree-loop path.  ``None`` follows DarkoFit's normal
        all-detected-cores behavior.

    Returns
    -------
    dict
        JSON-serializable audit metadata containing the frozen configuration,
        monotonic-clock durations, resolved fit settings, exact prediction
        dispatch boundaries and shapes, and deterministic fingerprints for
        both routes in the numeric and categorical lanes.
    """
    thread_count = _normalize_thread_count(thread_count)
    original_threads = numba.get_num_threads()
    started_ns = time.monotonic_ns()
    stages: list[dict[str, Any]] = []
    try:
        for stage in _make_warmup_data():
            stages.append(
                _run_stage(*stage, thread_count=thread_count)
            )
    finally:
        numba.set_num_threads(original_threads)

    metadata = {
        "schema_version": 2,
        "clock": "time.monotonic_ns",
        "duration_seconds": _elapsed_seconds(started_ns),
        "config": {**_MODEL_CONFIG, "thread_count": thread_count},
        "stages": stages,
    }
    # Fail here, outside benchmark result writing, if a future edit introduces
    # NumPy scalars, NaN, or another value the campaign manifest cannot encode.
    json.dumps(metadata, allow_nan=False)
    return metadata


__all__ = ["warmup_tabarena_regression"]
