#!/usr/bin/env python3
"""Run the frozen basketball Gaussian scalar-calibration opportunity screen."""

from __future__ import annotations

import argparse
import gc
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
import tracemalloc
from copy import copy
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable

import numpy as np
from scipy.special import ndtr
from sklearn.model_selection import ShuffleSplit


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_TEXT = str(REPO_ROOT)
if REPO_ROOT_TEXT in sys.path:
    sys.path.remove(REPO_ROOT_TEXT)
sys.path.insert(0, REPO_ROOT_TEXT)

import darkofit as darkofit_package  # noqa: E402
from benchmarks import basketball_harness as harness  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402


RANDOM_STATE = 4
THREADS = 18
EXPECTED_RESOLVED_THREADS = 2
CALIBRATION_FRACTION = 0.10
INTERVAL_ALPHA = 0.20
NOMINAL_COVERAGE = 0.80
MIN_STRICT_FOLD_WINS = 6
MAX_FOLD_NLL_RATIO = 1.02
MAX_WIDTH_RATIO = 1.25
TIMING_CALLS = 50
TIMING_BLOCKS = 3
MAX_TIMING_SPREAD_RATIO = 1.20
MAX_RUNTIME_RATIO = 1.10
MEMORY_CALLS = 5
MAX_EXTRA_TRACED_BYTES = 256 * 1024
SIGMA_MIN = 1e-12
Z_GUARD = 1000.0
SCALE_BOUNDS = (1e-6, 1e6)
PRE_PROTOCOL_COMMIT = "ba4b7f98004716a62e65d8bbb29a7074d3655313"
EXPECTED_DARKOFIT_TREE = "1a60b529c5f5d09920d81338406b491fb7275e3a"
PROTOCOL_PATH = (
    REPO_ROOT / "benchmarks" / "basketball_gaussian_scalar_calibration_protocol.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "ae045fceaf0ef3f6110be156fc9cae76791e8e8e45bccf8db0edc1f21d502e2c"
)
EXPECTED_NORMALIZED_RUNNER_SHA256 = (
    "40b6315d31b19260e8a63d00bab03ca14d0bb153bd1457a539272944169b95f8"
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
EXPECTED_TARGET_SHA256 = {
    "creator_training": (
        "7c06b45d4649c392fdb0c3dc91a857650f2f3fc77341fe06ffdbee2b3a44aecf"
    ),
    "overlap_exposed_team_holdout": (
        "c051a5ae966077792a2c28757ec4d06dc1660aef2cb4064cab10acac2216d1bf"
    ),
    "cold_player_subset": (
        "cd16264232c0966c5823709c392b32638912f14596ff6ea4d5e8a6a2b5dd30e8"
    ),
    "seen_player_subset": (
        "bca52624dbd022f53365fe319f1851d350253f2f9bd03a5360db56c7dad45d8b"
    ),
}
EXPECTED_RAW_SHA256 = "43af3be46074da9130a757aa1db643e207e2c0ea5405d2086d698f85555fbcb2"
EXPECTED_X_SHA256 = "05efa554a983942885b72b1b3fdcc97a10ecf4ccbb4b10843ee8b970129fe71b"
EXPECTED_FOLD_SHA256 = (
    "7270350a38a687b6e783d18d9c02e5a45f2c7367faa4a6955e74c710f5e8daea"
)
EXPECTED_COLD_MASK_SHA256 = (
    "e17887c9240fd95aee21d37d7e1e8a533c38ef87f4492acd28fb3aa496a3dd19"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "benchmarks" / "basketball_gaussian_scalar_calibration.json"
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
    if payload[value_end : value_end + 2] != b'"\n' or any(
        byte not in b"0123456789abcdef" for byte in payload[value_start:value_end]
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


def fit_scalar_scale(y_true: Any, mu: Any, sigma: Any) -> float:
    target = np.asarray(y_true, dtype=np.float64)
    location = np.asarray(mu, dtype=np.float64)
    scale = np.asarray(sigma, dtype=np.float64)
    if (
        target.ndim != 1
        or location.shape != target.shape
        or scale.shape != target.shape
        or target.size == 0
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(location))
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
    ):
        raise ValueError("scalar calibration inputs must be finite positive vectors")
    z = np.clip(
        (target - location) / np.maximum(scale, SIGMA_MIN),
        -Z_GUARD,
        Z_GUARD,
    )
    fitted = float(np.sqrt(max(float(np.mean(z * z)), SIGMA_MIN)))
    if not math.isfinite(fitted) or not SCALE_BOUNDS[0] < fitted < SCALE_BOUNDS[1]:
        raise RuntimeError("fitted scalar calibration is outside frozen bounds")
    return fitted


def gaussian_nll(y_true: Any, mu: Any, sigma: Any) -> float:
    target = np.asarray(y_true, dtype=np.float64)
    location = np.asarray(mu, dtype=np.float64)
    scale = np.asarray(sigma, dtype=np.float64)
    if (
        target.ndim != 1
        or location.shape != target.shape
        or scale.shape != target.shape
        or target.size == 0
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(location))
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
    ):
        raise ValueError("Gaussian NLL inputs must be finite positive vectors")
    z = (target - location) / scale
    value = np.mean(np.log(scale) + 0.5 * z * z + 0.5 * math.log(2.0 * math.pi))
    return float(value)


def gaussian_crps(y_true: Any, mu: Any, sigma: Any) -> float:
    target = np.asarray(y_true, dtype=np.float64)
    location = np.asarray(mu, dtype=np.float64)
    scale = np.asarray(sigma, dtype=np.float64)
    if (
        target.ndim != 1
        or location.shape != target.shape
        or scale.shape != target.shape
        or target.size == 0
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(location))
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
    ):
        raise ValueError("Gaussian CRPS inputs must be finite positive vectors")
    z = (target - location) / scale
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    values = scale * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Gaussian CRPS produced non-finite values")
    return float(np.mean(values))


def _interval(mu: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_value = NormalDist().inv_cdf(1.0 - INTERVAL_ALPHA / 2.0)
    return mu - z_value * sigma, mu + z_value * sigma


def _arm_metrics(
    target: np.ndarray,
    raw: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, Any]:
    lower, upper = _interval(mu, sigma)
    variance = sigma * sigma
    crossings = lower > upper
    coverage = float(np.mean((target >= lower) & (target <= upper)))
    return {
        "gaussian_nll": gaussian_nll(target, mu, sigma),
        "gaussian_crps": gaussian_crps(target, mu, sigma),
        "point_rmse": float(np.sqrt(np.mean((target - mu) ** 2))),
        "mean_sigma": float(np.mean(sigma)),
        "interval": {
            "alpha": INTERVAL_ALPHA,
            "nominal_coverage": NOMINAL_COVERAGE,
            "coverage": coverage,
            "absolute_coverage_error": abs(coverage - NOMINAL_COVERAGE),
            "mean_width": float(np.mean(upper - lower)),
            "crossing_count": int(np.count_nonzero(crossings)),
        },
        "sha256": {
            "raw": array_sha256(raw, "<f8"),
            "mean": array_sha256(mu, "<f8"),
            "sigma": array_sha256(sigma, "<f8"),
            "lower": array_sha256(lower, "<f8"),
            "upper": array_sha256(upper, "<f8"),
            "variance": array_sha256(variance, "<f8"),
        },
        "mean": [float(value) for value in mu],
        "sigma": [float(value) for value in sigma],
        "lower": [float(value) for value in lower],
        "upper": [float(value) for value in upper],
        "variance": [float(value) for value in variance],
    }


def score_parameters(
    y_true: Any,
    raw: Any,
    mu: Any,
    control_sigma: Any,
    candidate_sigma: Any,
) -> dict[str, Any]:
    target = np.asarray(y_true, dtype=np.float64)
    raw_values = np.asarray(raw, dtype=np.float64)
    location = np.asarray(mu, dtype=np.float64)
    control_scale = np.asarray(control_sigma, dtype=np.float64)
    candidate_scale = np.asarray(candidate_sigma, dtype=np.float64)
    if (
        target.ndim != 1
        or raw_values.shape != (len(target), 2)
        or location.shape != target.shape
        or control_scale.shape != target.shape
        or candidate_scale.shape != target.shape
        or target.size == 0
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(raw_values))
        or not np.all(np.isfinite(location))
        or not np.all(np.isfinite(control_scale))
        or not np.all(np.isfinite(candidate_scale))
        or np.any(control_scale <= 0.0)
        or np.any(candidate_scale <= 0.0)
    ):
        raise ValueError("score inputs must be finite Gaussian vectors")
    control = _arm_metrics(target, raw_values, location, control_scale)
    candidate = _arm_metrics(target, raw_values, location.copy(), candidate_scale)
    return {
        "rows": int(len(target)),
        "target_sha256": array_sha256(target, "<f8"),
        "raw_scores_sha256": array_sha256(raw_values, "<f8"),
        "control": control,
        "candidate": candidate,
        "invariants": {
            "means_array_exact": np.array_equal(
                np.asarray(control["mean"]), np.asarray(candidate["mean"])
            ),
            "point_rmse_exact": (control["point_rmse"] == candidate["point_rmse"]),
            "raw_scores_shared": (
                control["sha256"]["raw"] == candidate["sha256"]["raw"]
            ),
            "positive_scales": bool(
                np.all(control_scale > 0.0) and np.all(candidate_scale > 0.0)
            ),
            "zero_crossings": (
                control["interval"]["crossing_count"] == 0
                and candidate["interval"]["crossing_count"] == 0
            ),
        },
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


def _params_sha256(model: Any) -> str:
    payload = json.dumps(
        model.get_params(deep=False),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _core_raw(model: Any, X: Any) -> np.ndarray:
    values = X.to_numpy(copy=False) if hasattr(X, "to_numpy") else np.asarray(X)
    raw = np.asarray(model.model_.predict_raw(values), dtype=np.float64)
    if raw.shape != (len(values), 2) or not np.all(np.isfinite(raw)):
        raise RuntimeError("DarkoFit produced invalid Gaussian raw scores")
    return raw


def _validate_fitted_model(model: Any) -> dict[str, Any]:
    metadata = harness.extract_fit_metadata(model)
    if metadata["selected_lane"] != "boosting":
        raise RuntimeError("Gaussian fit unexpectedly selected another lane")
    if metadata["selected_tree_mode"] != "lightgbm":
        raise RuntimeError("Gaussian fit unexpectedly changed tree mode")
    if int(metadata["resolved_thread_count"]) != EXPECTED_RESOLVED_THREADS:
        raise RuntimeError("Gaussian fit resolved an unexpected thread count")
    if metadata["refit"] or metadata["selection_fit"] is not None:
        raise RuntimeError("Gaussian fit unexpectedly selected or refit")
    if metadata["final_early_stopping_rounds"] is not None:
        raise RuntimeError("Gaussian fit unexpectedly enabled early stopping")
    if metadata["fitted_tree_count"] != 1000:
        raise RuntimeError("Gaussian fit did not retain exactly 1,000 trees")
    if getattr(model, "dist_calibration_", None) != "scalar":
        raise RuntimeError("Gaussian scalar calibration was not active")
    if getattr(model, "dist_scale_source_", None) != "selection_validation":
        raise RuntimeError("Gaussian scalar calibration source changed")
    if not _numeric_state_is_finite(model):
        raise RuntimeError("Gaussian model contains non-finite numeric state")
    return metadata


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
        "calibration_relative_indices": [int(value) for value in calibration_relative],
        "calibration_relative_indices_sha256": index_sha256(calibration_relative),
        "fit_global_indices": [int(value) for value in fit_global],
        "fit_global_indices_sha256": index_sha256(fit_global),
        "calibration_global_indices": [int(value) for value in calibration_global],
        "calibration_global_indices_sha256": index_sha256(calibration_global),
        "disjoint": True,
        "covers_external_training_rows_exactly": True,
    }


def fit_gaussian_scalar(
    X_train: Any,
    y_train: Any,
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    model = darkofit_package.DarkoRegressor(
        loss="Gaussian",
        tree_mode="lightgbm",
        dist_calibration="scalar",
        use_best_model=False,
        early_stopping=False,
        random_state=RANDOM_STATE,
        thread_count=THREADS,
        diagnostic_warnings="never",
    )
    params_sha256 = _params_sha256(model)
    started = time.perf_counter_ns()
    model.fit(
        X_train.iloc[fit_indices],
        y_train.iloc[fit_indices],
        eval_set=(
            X_train.iloc[calibration_indices],
            y_train.iloc[calibration_indices],
        ),
    )
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    calibration_raw = _core_raw(model, X_train.iloc[calibration_indices])
    calibration_mu, calibration_sigma = model.model_.params_from_raw(calibration_raw)
    calibration_mu = np.asarray(calibration_mu, dtype=np.float64)
    calibration_sigma = np.asarray(calibration_sigma, dtype=np.float64)
    independent_scale = fit_scalar_scale(
        y_train.iloc[calibration_indices],
        calibration_mu,
        calibration_sigma,
    )
    fitted_scale = float(model.dist_scale_)
    if fitted_scale != independent_scale:
        raise RuntimeError("fitted scalar differs from independent product rule")
    metadata = _validate_fitted_model(model)
    record = {
        "constructor_params_sha256": params_sha256,
        "fit_seconds_directional": float(fit_seconds),
        "fit_metadata": metadata,
        "fitted_numeric_state_finite": True,
        "calibration_rows": int(len(calibration_indices)),
        "calibration_target_sha256": array_sha256(
            y_train.iloc[calibration_indices], "<f8"
        ),
        "calibration_raw_sha256": array_sha256(calibration_raw, "<f8"),
        "calibration_mu_sha256": array_sha256(calibration_mu, "<f8"),
        "calibration_sigma_sha256": array_sha256(calibration_sigma, "<f8"),
        "fitted_scale": fitted_scale,
        "independent_scale": independent_scale,
        "scale_array_exact": fitted_scale == independent_scale,
        "scale_inside_frozen_bounds": (
            SCALE_BOUNDS[0] < fitted_scale < SCALE_BOUNDS[1]
        ),
        "calibration_method": model.dist_calibration_,
        "calibration_source": model.dist_scale_source_,
    }
    return model, record


def evaluate_model(
    model: Any,
    X: Any,
    y_true: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    raw = _core_raw(model, X)
    control_mu, control_sigma = model.model_.params_from_raw(raw)
    control_mu = np.asarray(control_mu, dtype=np.float64)
    control_sigma = np.asarray(control_sigma, dtype=np.float64)
    candidate_sigma = control_sigma * float(model.dist_scale_)
    candidate_lower, candidate_upper = _interval(control_mu, candidate_sigma)
    public_mu, public_sigma = model.predict_dist(X)
    public_variance = model.predict_variance(X)
    public_lower, public_upper = model.predict_interval(X, alpha=INTERVAL_ALPHA)
    public_point = model.predict(X)
    exact_checks = {
        "public_mean": np.array_equal(public_mu, control_mu),
        "public_sigma": np.array_equal(public_sigma, candidate_sigma),
        "public_variance": np.array_equal(
            public_variance, candidate_sigma * candidate_sigma
        ),
        "public_lower": np.array_equal(public_lower, candidate_lower),
        "public_upper": np.array_equal(public_upper, candidate_upper),
        "public_point": np.array_equal(public_point, control_mu),
    }
    if not all(exact_checks.values()):
        raise RuntimeError(
            "public Gaussian prediction path differs from reconstruction"
        )
    score = score_parameters(
        y_true,
        raw,
        control_mu,
        control_sigma,
        candidate_sigma,
    )
    score["public_reconstruction_array_exact"] = exact_checks
    arrays = {
        "raw": raw,
        "mu": control_mu,
        "control_sigma": control_sigma,
        "candidate_sigma": candidate_sigma,
    }
    return score, arrays


def target_views(dataset: Any) -> dict[str, Any]:
    guardrail = dataset.player_guardrail
    creator_target = dataset.y
    holdout_target = guardrail.y_holdout
    cold = np.asarray(guardrail.cold_player_mask, dtype=np.bool_)
    seen = ~cold
    hashes = {
        "creator_training": array_sha256(creator_target, "<f8"),
        "overlap_exposed_team_holdout": array_sha256(holdout_target, "<f8"),
        "cold_player_subset": array_sha256(holdout_target.to_numpy()[cold], "<f8"),
        "seen_player_subset": array_sha256(holdout_target.to_numpy()[seen], "<f8"),
    }
    if hashes != EXPECTED_TARGET_SHA256:
        raise RuntimeError("basketball target fingerprints changed")
    rows = {
        "creator_training": int(len(creator_target)),
        "overlap_exposed_team_holdout": int(len(holdout_target)),
        "cold_player_subset": int(np.count_nonzero(cold)),
        "seen_player_subset": int(np.count_nonzero(seen)),
    }
    if rows != {
        "creator_training": 5241,
        "overlap_exposed_team_holdout": 2409,
        "cold_player_subset": 585,
        "seen_player_subset": 1824,
    }:
        raise RuntimeError("basketball target view row counts changed")
    return {
        "X_train": dataset.X,
        "y_train": creator_target,
        "X_holdout": guardrail.X_holdout,
        "y_holdout": holdout_target,
        "cold_mask": cold,
        "seen_mask": seen,
        "target_sha256": hashes,
        "rows": rows,
    }


def _tuple_array_equal(
    observed: tuple[np.ndarray, ...],
    expected: tuple[np.ndarray, ...],
) -> bool:
    return len(observed) == len(expected) and all(
        np.array_equal(left, right) for left, right in zip(observed, expected)
    )


def _time_arm(
    function: Callable[[], tuple[np.ndarray, ...]],
    expected: tuple[np.ndarray, ...],
) -> float:
    started = time.perf_counter_ns()
    for _ in range(TIMING_CALLS):
        observed = function()
        if not _tuple_array_equal(observed, expected):
            raise RuntimeError("repeated runtime output changed within an arm")
    return (time.perf_counter_ns() - started) / 1e9


def _trace_arm(
    function: Callable[[], tuple[np.ndarray, ...]],
    expected: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    peaks = []
    for _ in range(MEMORY_CALLS):
        gc.collect()
        tracemalloc.start()
        try:
            observed = function()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        if not _tuple_array_equal(observed, expected):
            raise RuntimeError("traced runtime output changed within an arm")
        peaks.append(int(peak))
    return {
        "calls": MEMORY_CALLS,
        "peak_bytes_per_call": peaks,
        "maximum_peak_bytes": int(max(peaks)),
    }


def runtime_and_memory_check(
    model: Any,
    X: Any,
    independent_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    control = copy(model)
    control.dist_calibration_ = None
    if control.model_ is not model.model_:
        raise RuntimeError("runtime arms do not share the identical fitted core")
    functions = {
        "control": lambda: tuple(
            np.asarray(value, dtype=np.float64) for value in control.predict_dist(X)
        ),
        "candidate": lambda: tuple(
            np.asarray(value, dtype=np.float64) for value in model.predict_dist(X)
        ),
    }
    expected = {name: function() for name, function in functions.items()}
    reconstructed = {
        "control": (
            independent_arrays["mu"],
            independent_arrays["control_sigma"],
        ),
        "candidate": (
            independent_arrays["mu"],
            independent_arrays["candidate_sigma"],
        ),
    }
    if not all(
        _tuple_array_equal(expected[name], reconstructed[name]) for name in expected
    ):
        raise RuntimeError("public runtime arms differ from independent reconstruction")
    durations = {name: [] for name in functions}
    blocks = []
    for block, order in enumerate(
        harness.reciprocal_schedule("control", "candidate", repetitions=TIMING_BLOCKS)
    ):
        block_record = {"block": int(block), "order": list(order), "arms": {}}
        for name in order:
            seconds = _time_arm(functions[name], expected[name])
            durations[name].append(seconds)
            block_record["arms"][name] = {
                "total_seconds": float(seconds),
                "per_call_seconds": float(seconds / TIMING_CALLS),
            }
        blocks.append(block_record)
    summaries = {
        name: harness.timing_summary(values, repetitions=TIMING_BLOCKS)
        for name, values in durations.items()
    }
    ratio = (
        summaries["candidate"]["median_seconds"]
        / summaries["control"]["median_seconds"]
    )
    memory = {
        name: _trace_arm(function, expected[name])
        for name, function in functions.items()
    }
    extra = (
        memory["candidate"]["maximum_peak_bytes"]
        - memory["control"]["maximum_peak_bytes"]
    )
    return {
        "path": "public_predict_dist",
        "shared_fitted_core": True,
        "calls_per_timing_block": TIMING_CALLS,
        "timing_blocks": blocks,
        "timing_summaries": summaries,
        "candidate_over_control_median_per_call": float(ratio),
        "runtime_output_sha256": {
            name: {
                "mean": array_sha256(values[0], "<f8"),
                "sigma": array_sha256(values[1], "<f8"),
            }
            for name, values in expected.items()
        },
        "memory": memory,
        "candidate_extra_maximum_traced_bytes": int(extra),
        "gates": {
            "control_timing_stable": (
                summaries["control"]["maximum_over_minimum"] <= MAX_TIMING_SPREAD_RATIO
            ),
            "candidate_timing_stable": (
                summaries["candidate"]["maximum_over_minimum"]
                <= MAX_TIMING_SPREAD_RATIO
            ),
            "candidate_runtime_ratio": ratio <= MAX_RUNTIME_RATIO,
            "candidate_transient_memory": extra <= MAX_EXTRA_TRACED_BYTES,
        },
    }


def _metrics_no_worse(view: dict[str, Any]) -> bool:
    return (
        view["candidate"]["gaussian_nll"] <= view["control"]["gaussian_nll"]
        and view["candidate"]["gaussian_crps"] <= view["control"]["gaussian_crps"]
        and view["candidate"]["interval"]["absolute_coverage_error"]
        <= view["control"]["interval"]["absolute_coverage_error"]
    )


def _width_ratio(view: dict[str, Any]) -> float:
    return float(
        view["candidate"]["interval"]["mean_width"]
        / view["control"]["interval"]["mean_width"]
    )


def analyze_results(
    pooled: dict[str, Any],
    folds: list[dict[str, Any]],
    holdout_views: dict[str, dict[str, Any]],
    runtime: dict[str, Any],
    *,
    all_finite: bool,
    reconstruction_checks_pass: bool,
) -> dict[str, Any]:
    if len(folds) != creator.N_SPLITS:
        raise RuntimeError("Gaussian result must contain ten creator folds")
    fold_control = np.asarray(
        [fold["scores"]["control"]["gaussian_nll"] for fold in folds],
        dtype=np.float64,
    )
    fold_candidate = np.asarray(
        [fold["scores"]["candidate"]["gaussian_nll"] for fold in folds],
        dtype=np.float64,
    )
    if np.any(fold_control <= 0.0):
        raise RuntimeError("fold NLL ratio requires positive control NLL")
    fold_ratios = fold_candidate / fold_control
    strict_wins = int(np.count_nonzero(fold_candidate < fold_control))
    team = holdout_views["overlap_exposed_team_holdout"]
    cold = holdout_views["cold_player_subset"]
    invariants = [pooled, *[fold["scores"] for fold in folds]]
    invariants.extend(holdout_views.values())
    invariant_checks = all(
        all(bool(value) for value in view["invariants"].values()) for view in invariants
    )
    crossings = all(
        view[arm]["interval"]["crossing_count"] == 0
        for view in invariants
        for arm in ("control", "candidate")
    )
    width_ratios = {
        "pooled_creator": _width_ratio(pooled),
        "overlap_exposed_team_holdout": _width_ratio(team),
        "cold_player_subset": _width_ratio(cold),
    }
    gates = {
        "all_values_finite": bool(all_finite),
        "scale_and_public_reconstruction": bool(reconstruction_checks_pass),
        "point_and_raw_invariants": bool(invariant_checks),
        "pooled_nll_strictly_lower": (
            pooled["candidate"]["gaussian_nll"] < pooled["control"]["gaussian_nll"]
        ),
        "pooled_crps_no_worse": (
            pooled["candidate"]["gaussian_crps"] <= pooled["control"]["gaussian_crps"]
        ),
        "pooled_coverage_error_no_worse": (
            pooled["candidate"]["interval"]["absolute_coverage_error"]
            <= pooled["control"]["interval"]["absolute_coverage_error"]
        ),
        "strict_fold_win_breadth": strict_wins >= MIN_STRICT_FOLD_WINS,
        "worst_fold_nll_ratio": (float(np.max(fold_ratios)) <= MAX_FOLD_NLL_RATIO),
        "team_metrics_no_worse": _metrics_no_worse(team),
        "cold_metrics_no_worse": _metrics_no_worse(cold),
        "width_ratios": all(
            value <= MAX_WIDTH_RATIO for value in width_ratios.values()
        ),
        "zero_interval_crossings": bool(crossings),
        **runtime["gates"],
    }
    passes = all(gates.values())
    return {
        "strict_fold_wins": strict_wins,
        "strict_fold_losses": int(np.count_nonzero(fold_candidate > fold_control)),
        "fold_ties": int(np.count_nonzero(fold_candidate == fold_control)),
        "fold_candidate_over_control_nll": [float(value) for value in fold_ratios],
        "worst_fold_candidate_over_control_nll": float(np.max(fold_ratios)),
        "candidate_over_control_width": width_ratios,
        "fatal_gates": gates,
        "passes_all_gates": bool(passes),
        "default_promotion_authorized": False,
        "recommendation": (
            "advance_existing_explicit_scalar_mode_to_broader_validation"
            if passes
            else "stop_distributional_scalar_calibration_at_basketball"
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
        raise RuntimeError("formal Gaussian calibration output path is not exact")
    if output.is_symlink() or output.exists():
        raise RuntimeError(f"refusing existing benchmark output: {output}")
    if _sha256_bytes(PROTOCOL_PATH.read_bytes()) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Gaussian scalar-calibration protocol changed")
    if _normalized_runner_sha256() != EXPECTED_NORMALIZED_RUNNER_SHA256:
        raise RuntimeError("Gaussian scalar-calibration runner changed")
    observed_support = support_sha256()
    if observed_support != EXPECTED_SUPPORT_SHA256:
        raise RuntimeError("Gaussian scalar-calibration support files changed")
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
        raise RuntimeError("formal Gaussian screen requires clean committed source")
    if (
        source["branch"] != "main"
        or source["tracked_main_refs"].get("origin/main") != source["head"]
    ):
        raise RuntimeError("formal Gaussian screen requires main == origin/main")
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
            raise RuntimeError(
                f"refusing to overwrite benchmark output: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _validate_data_binding(dataset: Any) -> None:
    if dataset.raw_metadata.get("sha256") != EXPECTED_RAW_SHA256:
        raise RuntimeError("basketball CSV changed")
    if dataset.processed_metadata.get("x_train_sha256") != EXPECTED_X_SHA256:
        raise RuntimeError("basketball creator features changed")
    if dataset.fold_fingerprint_sha256 != EXPECTED_FOLD_SHA256:
        raise RuntimeError("basketball creator folds changed")
    if (
        dataset.player_guardrail.metadata.get("cold_player_mask_sha256")
        != EXPECTED_COLD_MASK_SHA256
    ):
        raise RuntimeError("basketball cold-player mask changed")


def run_screen(output: Path, data_cache: Path) -> dict[str, Any]:
    binding = validate_frozen_binding(output)
    source_before = binding["source"]
    dataset = harness.load_basketball_dataset(data_cache)
    _validate_data_binding(dataset)
    views = target_views(dataset)
    X_train = views["X_train"]
    y_train = views["y_train"]
    n_rows = len(X_train)
    pooled_raw = np.full((n_rows, 2), np.nan, dtype=np.float64)
    pooled_mu = np.full(n_rows, np.nan, dtype=np.float64)
    pooled_control_sigma = np.full(n_rows, np.nan, dtype=np.float64)
    pooled_candidate_sigma = np.full(n_rows, np.nan, dtype=np.float64)
    folds = []
    started = time.perf_counter_ns()
    for fold, (external_train, external_test) in enumerate(
        creator.creator_cv().split(X_train, y_train)
    ):
        external_train = np.asarray(external_train, dtype=np.int64)
        external_test = np.asarray(external_test, dtype=np.int64)
        fit_relative, calibration_relative = calibration_split(len(external_train))
        fit_global = external_train[fit_relative]
        calibration_global = external_train[calibration_relative]
        model, model_record = fit_gaussian_scalar(
            X_train,
            y_train,
            fit_global,
            calibration_global,
        )
        scores, arrays = evaluate_model(
            model,
            X_train.iloc[external_test],
            y_train.iloc[external_test],
        )
        pooled_raw[external_test] = arrays["raw"]
        pooled_mu[external_test] = arrays["mu"]
        pooled_control_sigma[external_test] = arrays["control_sigma"]
        pooled_candidate_sigma[external_test] = arrays["candidate_sigma"]
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
                "model": model_record,
                "scores": scores,
            }
        )
        print(
            f"fold {fold + 1}/{creator.N_SPLITS}: "
            f"s={model_record['fitted_scale']:.6f}, "
            f"control={scores['control']['gaussian_nll']:.6f}, "
            f"candidate={scores['candidate']['gaussian_nll']:.6f}",
            flush=True,
        )
    pooled_arrays = (
        pooled_raw,
        pooled_mu,
        pooled_control_sigma,
        pooled_candidate_sigma,
    )
    if not all(np.all(np.isfinite(values)) for values in pooled_arrays):
        raise RuntimeError("pooled creator predictions are incomplete")
    pooled_scores = score_parameters(
        y_train,
        pooled_raw,
        pooled_mu,
        pooled_control_sigma,
        pooled_candidate_sigma,
    )
    pooled_scores["candidate_scale_scope"] = "independently_fitted_per_fold"

    held_external = np.arange(n_rows, dtype=np.int64)
    held_fit, held_calibration = calibration_split(n_rows)
    held_model, held_model_record = fit_gaussian_scalar(
        X_train,
        y_train,
        held_fit,
        held_calibration,
    )
    held_scores, held_arrays = evaluate_model(
        held_model,
        views["X_holdout"],
        views["y_holdout"],
    )
    cold = views["cold_mask"]
    seen = views["seen_mask"]
    holdout_views = {
        "overlap_exposed_team_holdout": held_scores,
        "seen_player_subset": score_parameters(
            views["y_holdout"].to_numpy()[seen],
            held_arrays["raw"][seen],
            held_arrays["mu"][seen],
            held_arrays["control_sigma"][seen],
            held_arrays["candidate_sigma"][seen],
        ),
        "cold_player_subset": score_parameters(
            views["y_holdout"].to_numpy()[cold],
            held_arrays["raw"][cold],
            held_arrays["mu"][cold],
            held_arrays["control_sigma"][cold],
            held_arrays["candidate_sigma"][cold],
        ),
    }
    runtime = runtime_and_memory_check(held_model, views["X_holdout"], held_arrays)
    expected_runtime_hashes = {
        "control": {
            "mean": held_scores["control"]["sha256"]["mean"],
            "sigma": held_scores["control"]["sha256"]["sigma"],
        },
        "candidate": {
            "mean": held_scores["candidate"]["sha256"]["mean"],
            "sigma": held_scores["candidate"]["sha256"]["sigma"],
        },
    }
    if runtime["runtime_output_sha256"] != expected_runtime_hashes:
        raise RuntimeError("runtime public path differs from scored held-team path")

    model_records = [fold["model"] for fold in folds] + [held_model_record]
    reconstruction_checks_pass = (
        all(
            model["scale_array_exact"] and model["scale_inside_frozen_bounds"]
            for model in model_records
        )
        and all(
            all(fold["scores"]["public_reconstruction_array_exact"].values())
            for fold in folds
        )
        and all(held_scores["public_reconstruction_array_exact"].values())
    )
    all_finite = all(
        model["fitted_numeric_state_finite"] and _numeric_state_is_finite(model)
        for model in model_records
    ) and _numeric_state_is_finite(
        {
            "pooled": pooled_scores,
            "folds": folds,
            "holdout_views": holdout_views,
            "runtime": runtime,
            "targets": {
                "creator": y_train.to_numpy(dtype=np.float64),
                "holdout": views["y_holdout"].to_numpy(dtype=np.float64),
            },
        }
    )
    decision = analyze_results(
        pooled_scores,
        folds,
        holdout_views,
        runtime,
        all_finite=all_finite,
        reconstruction_checks_pass=reconstruction_checks_pass,
    )
    elapsed_seconds = (time.perf_counter_ns() - started) / 1e9
    source_after = creator.git_state(REPO_ROOT)
    if source_after != source_before:
        raise RuntimeError("source changed during Gaussian calibration screen")
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "basketball_gaussian_scalar_calibration_opportunity_screen",
            "sha256": EXPECTED_PROTOCOL_SHA256,
            "basketball_is_primary_fatal_boundary": True,
            "product_code_changed": False,
            "single_lever": "existing_validation_fitted_positive_sigma_scale",
            "opportunity_score": 10.0,
            "random_state": RANDOM_STATE,
            "threads_per_fit": THREADS,
            "calibration_fraction": CALIBRATION_FRACTION,
            "interval_alpha": INTERVAL_ALPHA,
            "minimum_strict_fold_wins": MIN_STRICT_FOLD_WINS,
            "maximum_fold_nll_ratio": MAX_FOLD_NLL_RATIO,
            "maximum_width_ratio": MAX_WIDTH_RATIO,
            "maximum_runtime_ratio": MAX_RUNTIME_RATIO,
            "maximum_extra_traced_bytes": MAX_EXTRA_TRACED_BYTES,
            "lockbox_data_used": False,
        },
        "source": source_before,
        "executable_source": {
            "runner_normalized_sha256": binding["runner_normalized_sha256"],
            "support_sha256": binding["support_sha256"],
            "imported_darkofit_path": binding["imported_darkofit_path"],
            "darkofit_package_tree": EXPECTED_DARKOFIT_TREE,
        },
        "runtime": binding["runtime"],
        "raw_data": dataset.raw_metadata,
        "processed_data": dataset.processed_metadata,
        "guardrail": dataset.player_guardrail.metadata,
        "target_views": {
            "target_sha256": views["target_sha256"],
            "rows": views["rows"],
        },
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
            "internal_split": _split_record(held_external, held_fit, held_calibration),
            "model": held_model_record,
            "views": holdout_views,
        },
        "runtime_and_memory": runtime,
        "decision": decision,
        "directional_total_elapsed_seconds": float(elapsed_seconds),
        "literal_chimeraboost_code_copied": False,
    }
    _atomic_write_new_bytes(
        output,
        (json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
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
