#!/usr/bin/env python3
"""Run the frozen basketball binary temperature-scaling opportunity screen."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


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
CALIBRATION_FRACTION = 0.10
TEMPERATURE_BOUNDS = (0.05, 20.0)
LOG_BOUND_MARGIN = 1e-6
MIN_STRICT_FOLD_WINS = 6
MAX_FOLD_LOGLOSS_RATIO = 1.02
TIMING_CALLS = 50
TIMING_BLOCKS = 3
MAX_TIMING_SPREAD_RATIO = 1.20
MAX_RUNTIME_RATIO = 1.10
MEMORY_CALLS = 5
MAX_EXTRA_TRACED_BYTES = 256 * 1024
PRE_PROTOCOL_COMMIT = "ccf6d592d9788cf302cf68559f8723864a533c26"
EXPECTED_DARKOFIT_TREE = "1a60b529c5f5d09920d81338406b491fb7275e3a"
PROTOCOL_PATH = (
    REPO_ROOT / "benchmarks" / "basketball_temperature_scaling_protocol.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "83e27d03502391c6d00007c2ef06aa187735db2df56741f8d1311afb9f79aaef"
)
EXPECTED_NORMALIZED_RUNNER_SHA256 = (
    "5e08c0402c9685ae664e8ae784b9e83ac1d22fa04b01951f9a8616098868f555"
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
EXPECTED_LABEL_SHA256 = {
    "creator_training": (
        "5c5215635fbf8597298be6b78bf84648fb7ef9da6b16ffb7cde95af2e52b0374"
    ),
    "overlap_exposed_team_holdout": (
        "cfd520645de52579d2fe73cd3f62ba2e9ecc1f2a437c86cd10cea0199dfe0f46"
    ),
    "cold_player_subset": (
        "79090c357e3f0e7cb454600276d26df54cedeb4398b8272c252d4e4fd626d668"
    ),
    "seen_player_subset": (
        "58d6553479b7625902abcf7aff6f667701362d450157670f6f270128520a0b1f"
    ),
}
DEFAULT_OUTPUT = (
    REPO_ROOT / "benchmarks" / "basketball_temperature_scaling.json"
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


def stable_sigmoid(logits: Any) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("sigmoid input must be a finite vector")
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_negative = np.exp(values[~positive])
    result[~positive] = exp_negative / (1.0 + exp_negative)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("sigmoid produced non-finite probabilities")
    return result


def binary_log_loss_from_logits(target: Any, logits: Any) -> float:
    y = np.asarray(target, dtype=np.float64)
    z = np.asarray(logits, dtype=np.float64)
    if y.ndim != 1 or z.shape != y.shape or y.size == 0:
        raise ValueError("log-loss inputs must be equal nonempty vectors")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(z)):
        raise ValueError("log-loss inputs must be finite")
    if np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("binary targets must be zero or one")
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def expected_calibration_error(
    target: Any,
    probability: Any,
) -> float:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or y.size == 0:
        raise ValueError("ECE inputs must be equal nonempty vectors")
    if (
        not np.all(np.isfinite(y))
        or not np.all(np.isfinite(p))
        or np.any((p < 0.0) | (p > 1.0))
    ):
        raise ValueError("ECE inputs must be finite valid probabilities")
    bins = np.minimum(np.floor(p * 10.0).astype(np.int64), 9)
    result = 0.0
    for bin_index in range(10):
        selected = bins == bin_index
        count = int(np.count_nonzero(selected))
        if count:
            result += (count / len(y)) * abs(
                float(np.mean(y[selected])) - float(np.mean(p[selected]))
            )
    return float(result)


def calibration_split(target: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(target, dtype=np.uint8)
    if y.ndim != 1 or y.size < 2 or set(np.unique(y)) != {0, 1}:
        raise ValueError("calibration split requires both binary classes")
    fit_indices, calibration_indices = next(
        StratifiedShuffleSplit(
            n_splits=1,
            test_size=CALIBRATION_FRACTION,
            random_state=RANDOM_STATE,
        ).split(np.empty((len(y), 0)), y)
    )
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    calibration_indices = np.asarray(calibration_indices, dtype=np.int64)
    combined = np.concatenate((fit_indices, calibration_indices))
    if (
        len(np.intersect1d(fit_indices, calibration_indices)) != 0
        or len(combined) != len(y)
        or not np.array_equal(np.sort(combined), np.arange(len(y)))
        or set(np.unique(y[fit_indices])) != {0, 1}
        or set(np.unique(y[calibration_indices])) != {0, 1}
    ):
        raise RuntimeError("internal stratified calibration split is invalid")
    return fit_indices, calibration_indices


def _temperature_objective(log_temperature: float, target, logits) -> float:
    temperature = math.exp(float(log_temperature))
    return binary_log_loss_from_logits(
        target, np.asarray(logits, dtype=np.float64) / temperature
    )


def fit_temperature(target: Any, logits: Any) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.float64)
    z = np.asarray(logits, dtype=np.float64)
    lower, upper = (math.log(value) for value in TEMPERATURE_BOUNDS)
    baseline = _temperature_objective(0.0, y, z)
    result = minimize_scalar(
        _temperature_objective,
        args=(y, z),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 500},
    )
    log_temperature = float(result.x)
    temperature = math.exp(log_temperature)
    objective = float(result.fun)
    inside_bounds = min(log_temperature - lower, upper - log_temperature)
    diagnostics = {
        "temperature": temperature,
        "log_temperature": log_temperature,
        "baseline_objective": baseline,
        "candidate_objective": objective,
        "objective_delta": objective - baseline,
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
        "optimizer_evaluations": int(result.nfev),
        "objective_no_higher_than_t1": objective <= baseline,
        "inside_log_bounds_by_required_margin": inside_bounds >= LOG_BOUND_MARGIN,
    }
    if (
        not all(
            math.isfinite(float(diagnostics[key]))
            for key in (
                "temperature",
                "log_temperature",
                "baseline_objective",
                "candidate_objective",
                "objective_delta",
            )
        )
        or not diagnostics["optimizer_success"]
        or not diagnostics["objective_no_higher_than_t1"]
        or not diagnostics["inside_log_bounds_by_required_margin"]
    ):
        raise RuntimeError("temperature optimizer failed the frozen checks")
    return diagnostics


def _tie_pattern(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    sorted_values = values[order]
    adjacent_ties = np.zeros(len(sorted_values), dtype=np.uint8)
    if len(sorted_values) > 1:
        adjacent_ties[1:] = sorted_values[1:] == sorted_values[:-1]
    return adjacent_ties


def _tie_sha256(values: np.ndarray, order: np.ndarray) -> str:
    return array_sha256(_tie_pattern(values, order), "|u1")


def _arm_metrics(target: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    probability = stable_sigmoid(logits)
    classes = (probability > 0.5).astype(np.uint8)
    order = np.argsort(logits, kind="stable")
    return {
        "log_loss": binary_log_loss_from_logits(target, logits),
        "brier_score": float(np.mean((probability - target) ** 2)),
        "ece_10_equal_width": expected_calibration_error(target, probability),
        "accuracy": float(np.mean(classes == target)),
        "roc_auc": float(roc_auc_score(target, logits)),
        "logit_sha256": array_sha256(logits, "<f8"),
        "probability_sha256": array_sha256(probability, "<f8"),
        "class_prediction_sha256": array_sha256(classes, "|u1"),
        "stable_order_sha256": index_sha256(order),
        "tie_pattern_sha256": _tie_sha256(logits, order),
        "probabilities": [float(value) for value in probability],
        "logits": [float(value) for value in logits],
    }


def score_view(
    target: Any,
    raw_logits: Any,
    temperature: float,
    *,
    require_monotonic_invariants: bool,
) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.uint8)
    raw = np.asarray(raw_logits, dtype=np.float64)
    if (
        y.ndim != 1
        or raw.shape != y.shape
        or y.size == 0
        or not np.all(np.isfinite(raw))
        or set(np.unique(y)) != {0, 1}
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ValueError("score inputs are not valid finite binary vectors")
    scaled = raw / float(temperature)
    control = _arm_metrics(y, raw)
    candidate = _arm_metrics(y, scaled)
    control_probability = stable_sigmoid(raw)
    candidate_probability = stable_sigmoid(scaled)
    class_identical = np.array_equal(
        control_probability > 0.5, candidate_probability > 0.5
    )
    control_order = np.argsort(raw, kind="stable")
    candidate_order = np.argsort(scaled, kind="stable")
    order_identical = np.array_equal(control_order, candidate_order)
    ties_identical = np.array_equal(
        _tie_pattern(raw, control_order),
        _tie_pattern(scaled, candidate_order),
    )
    if require_monotonic_invariants and not (
        class_identical and order_identical and ties_identical
    ):
        raise RuntimeError("temperature scaling changed a monotonic invariant")
    return {
        "rows": int(len(y)),
        "target_sha256": array_sha256(y, "|u1"),
        "temperature": float(temperature),
        "control": control,
        "candidate": candidate,
        "invariants": {
            "required": bool(require_monotonic_invariants),
            "class_predictions_identical": class_identical,
            "stable_score_order_identical": order_identical,
            "score_ties_identical": ties_identical,
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


def _validate_fitted_model(model: Any) -> dict[str, Any]:
    metadata = harness.extract_fit_metadata(model)
    if metadata["selected_lane"] != "boosting":
        raise RuntimeError("classifier fit unexpectedly selected another lane")
    if metadata["selected_tree_mode"] != "catboost":
        raise RuntimeError("classifier fit unexpectedly changed tree mode")
    if int(metadata["resolved_thread_count"]) != THREADS:
        raise RuntimeError("classifier fit did not resolve exactly 18 threads")
    if metadata["refit"] or metadata["selection_fit"] is not None:
        raise RuntimeError("classifier fit unexpectedly selected or refit")
    if metadata["final_early_stopping_rounds"] is not None:
        raise RuntimeError("classifier fit unexpectedly enabled early stopping")
    if not _numeric_state_is_finite(model):
        raise RuntimeError("classifier contains non-finite numeric state")
    return metadata


def _predict_raw(model: Any, X: Any) -> np.ndarray:
    raw = np.asarray(model.model_.predict_raw(X), dtype=np.float64)
    return harness.validate_prediction(raw, len(X))


def fit_classifier_temperature(
    X_train,
    y_train,
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    model = darkofit_package.DarkoClassifier(
        random_state=RANDOM_STATE,
        thread_count=THREADS,
        diagnostic_warnings="never",
    )
    params_sha256 = _params_sha256(model)
    started = time.perf_counter_ns()
    model.fit(X_train.iloc[fit_indices], y_train.iloc[fit_indices])
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    calibration_logits = _predict_raw(model, X_train.iloc[calibration_indices])
    diagnostics = fit_temperature(
        y_train.iloc[calibration_indices], calibration_logits
    )
    metadata = _validate_fitted_model(model)
    record = {
        "constructor_params_sha256": params_sha256,
        "fit_seconds_directional": float(fit_seconds),
        "fit_metadata": metadata,
        "fitted_numeric_state_finite": True,
        "calibration_rows": int(len(calibration_indices)),
        "calibration_target_sha256": array_sha256(
            y_train.iloc[calibration_indices], "|u1"
        ),
        "calibration_logit_sha256": array_sha256(calibration_logits, "<f8"),
        "temperature_fit": diagnostics,
    }
    return model, diagnostics, record


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


def starter_views(dataset) -> dict[str, Any]:
    frame = dataset.frame.loc[dataset.frame["MP"] > 500].copy()
    if frame["G"].isna().any() or np.any(frame["G"].to_numpy() <= 0):
        raise RuntimeError("starter target requires finite positive G")
    frame["starter"] = (
        frame["GS"].to_numpy(dtype=np.float64)
        / frame["G"].to_numpy(dtype=np.float64)
        >= 0.5
    ).astype(np.uint8)
    teams = frame["Tm"].sort_values().drop_duplicates().tolist()
    test_teams = set(teams[: len(teams) // 3])
    training = frame.loc[~frame["Tm"].isin(test_teams)]
    holdout = frame.loc[frame["Tm"].isin(test_teams)]
    X_train = training.loc[:, creator.FEATURES]
    X_holdout = holdout.loc[:, creator.FEATURES]
    if (
        not X_train.equals(dataset.X)
        or not X_train.equals(dataset.player_guardrail.X_train)
        or not X_holdout.equals(dataset.player_guardrail.X_holdout)
    ):
        raise RuntimeError("starter views changed a frozen basketball feature view")
    y_train = training.loc[:, "starter"].astype(np.uint8)
    y_holdout = holdout.loc[:, "starter"].astype(np.uint8)
    cold = np.asarray(dataset.player_guardrail.cold_player_mask, dtype=np.bool_)
    seen = ~cold
    observed_hashes = {
        "creator_training": array_sha256(y_train, "|u1"),
        "overlap_exposed_team_holdout": array_sha256(y_holdout, "|u1"),
        "cold_player_subset": array_sha256(y_holdout.to_numpy()[cold], "|u1"),
        "seen_player_subset": array_sha256(y_holdout.to_numpy()[seen], "|u1"),
    }
    if observed_hashes != EXPECTED_LABEL_SHA256:
        raise RuntimeError("starter-label fingerprints changed")
    expected_rows = {
        "creator_training": 5241,
        "overlap_exposed_team_holdout": 2409,
        "cold_player_subset": 585,
        "seen_player_subset": 1824,
    }
    observed_rows = {
        "creator_training": int(len(y_train)),
        "overlap_exposed_team_holdout": int(len(y_holdout)),
        "cold_player_subset": int(np.count_nonzero(cold)),
        "seen_player_subset": int(np.count_nonzero(seen)),
    }
    if observed_rows != expected_rows:
        raise RuntimeError("starter view row counts changed")
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_holdout": X_holdout,
        "y_holdout": y_holdout,
        "cold_mask": cold,
        "seen_mask": seen,
        "label_sha256": observed_hashes,
        "rows": observed_rows,
        "positive_counts": {
            "creator_training": int(y_train.sum()),
            "overlap_exposed_team_holdout": int(y_holdout.sum()),
            "cold_player_subset": int(y_holdout.to_numpy()[cold].sum()),
            "seen_player_subset": int(y_holdout.to_numpy()[seen].sum()),
        },
    }


def _runtime_output(
    model: Any,
    X: Any,
    temperature: float,
    *,
    candidate: bool,
) -> np.ndarray:
    raw = _predict_raw(model, X)
    if candidate:
        raw = raw / float(temperature)
    return stable_sigmoid(raw)


def _time_arm(
    function: Callable[[], np.ndarray],
    expected: np.ndarray,
) -> float:
    started = time.perf_counter_ns()
    for _ in range(TIMING_CALLS):
        observed = function()
        if not np.array_equal(observed, expected):
            raise RuntimeError("repeated runtime output changed within an arm")
    return (time.perf_counter_ns() - started) / 1e9


def _trace_arm(
    function: Callable[[], np.ndarray],
    expected: np.ndarray,
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
        if not np.array_equal(observed, expected):
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
    temperature: float,
) -> dict[str, Any]:
    functions = {
        "control": lambda: _runtime_output(
            model, X, temperature, candidate=False
        ),
        "candidate": lambda: _runtime_output(
            model, X, temperature, candidate=True
        ),
    }
    expected = {name: function() for name, function in functions.items()}
    durations = {name: [] for name in functions}
    blocks = []
    for block, order in enumerate(
        harness.reciprocal_schedule(
            "control", "candidate", repetitions=TIMING_BLOCKS
        )
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
        "calls_per_timing_block": TIMING_CALLS,
        "timing_blocks": blocks,
        "timing_summaries": summaries,
        "candidate_over_control_median_per_call": float(ratio),
        "runtime_output_sha256": {
            name: array_sha256(values, "<f8") for name, values in expected.items()
        },
        "memory": memory,
        "candidate_extra_maximum_traced_bytes": int(extra),
        "gates": {
            "control_timing_stable": (
                summaries["control"]["maximum_over_minimum"]
                <= MAX_TIMING_SPREAD_RATIO
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
    return all(
        float(view["candidate"][metric]) <= float(view["control"][metric])
        for metric in ("log_loss", "brier_score", "ece_10_equal_width")
    )


def analyze_results(
    pooled: dict[str, Any],
    folds: list[dict[str, Any]],
    holdout_views: dict[str, dict[str, Any]],
    runtime: dict[str, Any],
    *,
    all_finite: bool,
    optimizer_checks_pass: bool,
) -> dict[str, Any]:
    if len(folds) != creator.N_SPLITS:
        raise RuntimeError("temperature result must contain ten creator folds")
    fold_control = np.asarray(
        [fold["scores"]["control"]["log_loss"] for fold in folds],
        dtype=np.float64,
    )
    fold_candidate = np.asarray(
        [fold["scores"]["candidate"]["log_loss"] for fold in folds],
        dtype=np.float64,
    )
    fold_ratios = fold_candidate / fold_control
    monotonic = all(
        all(
            bool(fold["scores"]["invariants"][key])
            for key in (
                "class_predictions_identical",
                "stable_score_order_identical",
                "score_ties_identical",
            )
        )
        for fold in folds
    ) and all(
        all(
            bool(view["invariants"][key])
            for key in (
                "class_predictions_identical",
                "stable_score_order_identical",
                "score_ties_identical",
            )
        )
        for view in holdout_views.values()
    )
    strict_wins = int(np.count_nonzero(fold_candidate < fold_control))
    team = holdout_views["overlap_exposed_team_holdout"]
    cold = holdout_views["cold_player_subset"]
    gates = {
        "all_values_finite": bool(all_finite),
        "optimizer_checks": bool(optimizer_checks_pass),
        "monotonic_invariants": bool(monotonic),
        "pooled_log_loss_strictly_lower": (
            pooled["candidate"]["log_loss"] < pooled["control"]["log_loss"]
        ),
        "pooled_brier_no_worse": (
            pooled["candidate"]["brier_score"]
            <= pooled["control"]["brier_score"]
        ),
        "pooled_ece_no_worse": (
            pooled["candidate"]["ece_10_equal_width"]
            <= pooled["control"]["ece_10_equal_width"]
        ),
        "strict_fold_win_breadth": strict_wins >= MIN_STRICT_FOLD_WINS,
        "worst_fold_log_loss_ratio": (
            float(np.max(fold_ratios)) <= MAX_FOLD_LOGLOSS_RATIO
        ),
        "team_metrics_no_worse": _metrics_no_worse(team),
        "cold_metrics_no_worse": _metrics_no_worse(cold),
        **runtime["gates"],
    }
    passes = all(gates.values())
    return {
        "strict_fold_wins": strict_wins,
        "strict_fold_losses": int(np.count_nonzero(fold_candidate > fold_control)),
        "fold_ties": int(np.count_nonzero(fold_candidate == fold_control)),
        "fold_candidate_over_control_log_loss": [
            float(value) for value in fold_ratios
        ],
        "worst_fold_candidate_over_control_log_loss": float(np.max(fold_ratios)),
        "fatal_gates": gates,
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
        raise RuntimeError("formal temperature screen output path is not exact")
    if output.is_symlink() or output.exists():
        raise RuntimeError(f"refusing existing benchmark output: {output}")
    if _sha256_bytes(PROTOCOL_PATH.read_bytes()) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("temperature-scaling protocol changed")
    if _normalized_runner_sha256() != EXPECTED_NORMALIZED_RUNNER_SHA256:
        raise RuntimeError("temperature-scaling runner changed")
    observed_support = support_sha256()
    if observed_support != EXPECTED_SUPPORT_SHA256:
        raise RuntimeError("temperature-scaling support files changed")
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
        raise RuntimeError("formal temperature screen requires clean committed source")
    if (
        source["branch"] != "main"
        or source["tracked_main_refs"].get("origin/main") != source["head"]
    ):
        raise RuntimeError("formal temperature screen requires main == origin/main")
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


def run_screen(output: Path, data_cache: Path) -> dict[str, Any]:
    binding = validate_frozen_binding(output)
    source_before = binding["source"]
    dataset = harness.load_basketball_dataset(data_cache)
    views = starter_views(dataset)
    X_train = views["X_train"]
    y_train = views["y_train"]
    n_rows = len(X_train)
    pooled_raw = np.full(n_rows, np.nan, dtype=np.float64)
    pooled_scaled = np.full(n_rows, np.nan, dtype=np.float64)
    folds = []
    started = time.perf_counter_ns()
    for fold, (external_train, external_test) in enumerate(
        creator.creator_cv().split(X_train, y_train)
    ):
        external_train = np.asarray(external_train, dtype=np.int64)
        external_test = np.asarray(external_test, dtype=np.int64)
        fit_relative, calibration_relative = calibration_split(
            y_train.iloc[external_train]
        )
        fit_global = external_train[fit_relative]
        calibration_global = external_train[calibration_relative]
        model, temperature, model_record = fit_classifier_temperature(
            X_train,
            y_train,
            fit_global,
            calibration_global,
        )
        raw = _predict_raw(model, X_train.iloc[external_test])
        value = float(temperature["temperature"])
        pooled_raw[external_test] = raw
        pooled_scaled[external_test] = raw / value
        scores = score_view(
            y_train.iloc[external_test],
            raw,
            value,
            require_monotonic_invariants=True,
        )
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
            f"T={value:.6f}, "
            f"control={scores['control']['log_loss']:.6f}, "
            f"candidate={scores['candidate']['log_loss']:.6f}",
            flush=True,
        )
    if not np.all(np.isfinite(pooled_raw)) or not np.all(np.isfinite(pooled_scaled)):
        raise RuntimeError("pooled creator predictions are incomplete")
    pooled_target = y_train.to_numpy(dtype=np.uint8)
    pooled_control_probability = stable_sigmoid(pooled_raw)
    pooled_candidate_probability = stable_sigmoid(pooled_scaled)
    pooled_scores = {
        "rows": int(n_rows),
        "target_sha256": array_sha256(pooled_target, "|u1"),
        "temperature": None,
        "temperature_scope": "independently_fitted_per_fold",
        "control": _arm_metrics(pooled_target, pooled_raw),
        "candidate": _arm_metrics(pooled_target, pooled_scaled),
        "invariants": {
            "required": False,
            "reason": "independently_fitted_fold_temperatures",
            "class_predictions_identical": np.array_equal(
                pooled_control_probability > 0.5,
                pooled_candidate_probability > 0.5,
            ),
            "stable_score_order_identical": np.array_equal(
                np.argsort(pooled_raw, kind="stable"),
                np.argsort(pooled_scaled, kind="stable"),
            ),
            "score_ties_identical": np.array_equal(
                _tie_pattern(
                    pooled_raw, np.argsort(pooled_raw, kind="stable")
                ),
                _tie_pattern(
                    pooled_scaled, np.argsort(pooled_scaled, kind="stable")
                ),
            ),
        },
    }
    held_external = np.arange(len(X_train), dtype=np.int64)
    held_fit, held_calibration = calibration_split(y_train)
    held_model, held_temperature, held_model_record = fit_classifier_temperature(
        X_train,
        y_train,
        held_fit,
        held_calibration,
    )
    held_raw = _predict_raw(held_model, views["X_holdout"])
    held_t = float(held_temperature["temperature"])
    cold = views["cold_mask"]
    seen = views["seen_mask"]
    holdout_views = {
        "overlap_exposed_team_holdout": score_view(
            views["y_holdout"],
            held_raw,
            held_t,
            require_monotonic_invariants=True,
        ),
        "seen_player_subset": score_view(
            views["y_holdout"].to_numpy()[seen],
            held_raw[seen],
            held_t,
            require_monotonic_invariants=True,
        ),
        "cold_player_subset": score_view(
            views["y_holdout"].to_numpy()[cold],
            held_raw[cold],
            held_t,
            require_monotonic_invariants=True,
        ),
    }
    runtime = runtime_and_memory_check(
        held_model, views["X_holdout"], held_t
    )
    if runtime["runtime_output_sha256"] != {
        "control": holdout_views["overlap_exposed_team_holdout"]["control"][
            "probability_sha256"
        ],
        "candidate": holdout_views["overlap_exposed_team_holdout"]["candidate"][
            "probability_sha256"
        ],
    }:
        raise RuntimeError("runtime transform differs from the scored candidate")
    model_records = [fold["model"] for fold in folds] + [held_model_record]
    all_finite = all(
        model["fitted_numeric_state_finite"]
        and _numeric_state_is_finite(model)
        for model in model_records
    ) and _numeric_state_is_finite(
        {
            "pooled": pooled_scores,
            "folds": folds,
            "holdout_views": holdout_views,
            "runtime": runtime,
            "labels": {
                "creator": views["y_train"].to_numpy(dtype=np.uint8),
                "holdout": views["y_holdout"].to_numpy(dtype=np.uint8),
            },
        }
    )
    optimizer_checks_pass = all(
        model["temperature_fit"]["optimizer_success"]
        and model["temperature_fit"]["objective_no_higher_than_t1"]
        and model["temperature_fit"]["inside_log_bounds_by_required_margin"]
        for model in model_records
    )
    decision = analyze_results(
        pooled_scores,
        folds,
        holdout_views,
        runtime,
        all_finite=all_finite,
        optimizer_checks_pass=optimizer_checks_pass,
    )
    elapsed_seconds = (time.perf_counter_ns() - started) / 1e9
    source_after = creator.git_state(REPO_ROOT)
    if source_after != source_before:
        raise RuntimeError("source changed during temperature-scaling screen")
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "basketball_binary_temperature_scaling_opportunity_screen",
            "sha256": EXPECTED_PROTOCOL_SHA256,
            "basketball_is_primary_fatal_boundary": True,
            "product_code_changed": False,
            "single_lever": "validation_fitted_positive_temperature",
            "random_state": RANDOM_STATE,
            "threads_per_fit": THREADS,
            "calibration_fraction": CALIBRATION_FRACTION,
            "temperature_bounds": list(TEMPERATURE_BOUNDS),
            "minimum_strict_fold_wins": MIN_STRICT_FOLD_WINS,
            "maximum_fold_log_loss_ratio": MAX_FOLD_LOGLOSS_RATIO,
            "maximum_runtime_ratio": MAX_RUNTIME_RATIO,
            "maximum_extra_traced_bytes": MAX_EXTRA_TRACED_BYTES,
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
        "guardrail": dataset.player_guardrail.metadata,
        "starter_views": {
            key: value
            for key, value in views.items()
            if key
            not in {
                "X_train",
                "y_train",
                "X_holdout",
                "y_holdout",
                "cold_mask",
                "seen_mask",
            }
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
            "internal_split": _split_record(
                held_external, held_fit, held_calibration
            ),
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
