#!/usr/bin/env python3
"""Run the frozen Wave-1 M1 characterization or Q0 scalar profile."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import pickle
import resource
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from functools import partial, wraps
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import basketball_campaign_harness as campaign  # noqa: E402
from benchmarks import basketball_harness as basketball  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402


PROTOCOL = ROOT / "benchmarks" / "m1_q0_wave1_protocol.md"
DEFAULT_M1_OUTPUT = ROOT / "benchmarks" / "m1_wave1.json"
DEFAULT_Q0_OUTPUT = ROOT / "benchmarks" / "q0_wave1_profile.json"
DEFAULT_DARKO_SOURCE = Path(
    "/private/tmp/darkofit-wave1-source-726e5d8"
)
DEFAULT_CHIMERA_SOURCE = ROOT.parent / "chimeraboost"

DARKO_SOURCE_HEAD = "726e5d8e6131c580bce948db833a5007d0692dca"
CHIMERA_SOURCE_HEAD = "f14be606b641f1bf0dc92bb14b3951f1fe631c6b"

DARKO = "darkofit_float"
CHIMERA_QUANTIZED = "chimeraboost_quantized"
CHIMERA_FLOAT = "chimeraboost_float"
M1_ARMS = (DARKO, CHIMERA_QUANTIZED, CHIMERA_FLOAT)
M1_BLOCK_ORDERS = tuple(itertools.permutations(M1_ARMS))

Q0_PRODUCTION = "production"
Q0_REFERENCE = "unfused_reference"
Q0_MODES = (Q0_PRODUCTION, Q0_REFERENCE)
Q0_BLOCK_ORDERS = (
    (Q0_PRODUCTION, Q0_REFERENCE),
    (Q0_REFERENCE, Q0_PRODUCTION),
    (Q0_PRODUCTION, Q0_REFERENCE),
)

TRAIN_ROWS = (500_000, 1_000_000)
HOLDOUT_ROWS = 100_000
FEATURES = 24
M1_ITERATIONS = 300
Q0_ITERATIONS = 40
THREADS = 14
WARMUP_ROWS = 5_000
WARMUP_ITERATIONS = 3
DATA_SEED = 20_260_717

MAX_IQR_OVER_MEDIAN = 0.10
MATERIAL_DONOR_GEOMEAN_RATIO = 0.90
MAX_DONOR_SIZE_RATIO = 1.02
MAX_DONOR_RMSE_RATIO = 1.002
Q_MIN_END_TO_END_REDUCTION = 0.10
Q_MAX_GEOMEAN_RATIO = 1.0 - Q_MIN_END_TO_END_REDUCTION
Q_MAX_SIZE_RATIO = 1.02
Q_KERNEL_SPEEDUP_PRIOR = 1.30
Q_REQUIRED_EQUAL_SHARE = Q_MIN_END_TO_END_REDUCTION / (
    1.0 - 1.0 / Q_KERNEL_SPEEDUP_PRIOR
)

WORKER_RESULT_PREFIX = "M1_Q0_WAVE1_RESULT="

_FUSED_NAMES = (
    "_build_histograms_subset_and_best_split",
    "_build_histograms_subset_unit_hess_and_best_split",
    "_build_histograms_and_best_split",
    "_build_histograms_unit_hess_and_best_split",
)
_HISTOGRAM_NAMES = (
    "_build_histograms_into",
    "_build_histograms_selected_into",
    "_build_histograms_rows_into",
    "_build_histograms_selected_rows_into",
    "_build_histograms_unit_hess_into",
    "_build_histograms_selected_unit_hess_into",
    "_build_histograms_rows_unit_hess_into",
    "_build_histograms_selected_rows_unit_hess_into",
    "_build_histograms_into_serial",
    "_build_histograms_selected_into_serial",
    "_build_histograms_rows_into_serial",
    "_build_histograms_selected_rows_into_serial",
    "_build_histograms_unit_hess_into_serial",
    "_build_histograms_selected_unit_hess_into_serial",
    "_build_histograms_rows_unit_hess_into_serial",
    "_build_histograms_selected_rows_unit_hess_into_serial",
    "_build_histograms_rowpar_into",
    "_build_histograms_unit_hess_rowpar_into",
)
_SPLIT_NAMES = ("_best_split", "_best_split_serial")
_SIBLING_NAMES = (
    "_build_level_histograms_subtract_into",
    "_build_level_histograms_subtract_unit_hess_into",
    "_build_level_histograms_subtract_into_serial",
    "_build_level_histograms_subtract_unit_hess_into_serial",
)
_LEAF_VALUE_NAMES = (
    "_leaf_values_and_sums",
    "_leaf_values_and_sums_rows",
)
_LEAF_ROUTING_NAMES = ("_update_leaves_with_split",)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _geomean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if (
        array.ndim != 1
        or array.size == 0
        or not np.all(np.isfinite(array))
        or np.any(array <= 0.0)
    ):
        raise RuntimeError("geometric-mean inputs must be positive and finite")
    return float(np.exp(np.mean(np.log(array))))


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("Wave-1 worker peak RSS is unavailable")
    return value


def _data(train_rows: int, holdout_rows: int = HOLDOUT_ROWS):
    rows = int(train_rows) + int(holdout_rows)
    rng = np.random.default_rng(DATA_SEED)
    X = rng.normal(size=(rows, FEATURES))
    signal = (
        1.4 * X[:, 0]
        - 0.9 * X[:, 1]
        + 0.35 * X[:, 2] * X[:, 3]
        + 0.2 * X[:, 4] ** 2
    )
    y = signal + rng.normal(0.0, 0.5, rows)
    probe = np.concatenate(
        (X[:256].ravel(), X[-256:].ravel(), y[:256], y[-256:])
    )
    return (
        X[:train_rows],
        y[:train_rows],
        X[train_rows:],
        y[train_rows:],
        _array_sha256(probe),
    )


def _common_params() -> dict[str, Any]:
    return {
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 1.0,
        "max_bins": 128,
        "subsample": 1.0,
        "colsample": 1.0,
        "min_child_weight": 1.0,
        "ordered_boosting": False,
        "early_stopping": False,
        "thread_count": THREADS,
        "random_state": 4,
    }


def _path_is_under(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def _activate_source(source: Path) -> None:
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise RuntimeError(f"package source directory does not exist: {source}")
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != source]
    sys.path.insert(0, str(source))


def _assert_import_source(module: Any, source: Path, package: str) -> str:
    module_path = Path(module.__file__).resolve()
    if not _path_is_under(module_path, source):
        raise RuntimeError(
            f"{package} imported from {module_path}, not frozen source {source}"
        )
    return str(module_path)


def _build_estimator(
    arm: str,
    iterations: int,
    *,
    darkofit_source: Path,
    chimeraboost_source: Path,
):
    if arm == DARKO:
        _activate_source(darkofit_source)
        import darkofit
        from darkofit import DarkoRegressor

        import_path = _assert_import_source(
            darkofit, darkofit_source, "darkofit"
        )
        model = DarkoRegressor(
            iterations=int(iterations),
            loss="RMSE",
            min_child_samples=1,
            tree_mode="catboost",
            linear_leaves=False,
            use_best_model=False,
            eval_train_loss=False,
            diagnostic_warnings="never",
            verbose_timing=True,
            **_common_params(),
        )
        return model, import_path

    if arm in {CHIMERA_QUANTIZED, CHIMERA_FLOAT}:
        _activate_source(chimeraboost_source)
        import chimeraboost
        from chimeraboost import ChimeraBoostRegressor

        import_path = _assert_import_source(
            chimeraboost, chimeraboost_source, "chimeraboost"
        )
        model = ChimeraBoostRegressor(
            n_estimators=int(iterations),
            linear_leaves=False,
            cross_features=False,
            cat_combinations=False,
            n_ensembles=None,
            quantize_gradients=(arm == CHIMERA_QUANTIZED),
            **_common_params(),
        )
        return model, import_path
    raise ValueError(f"unknown M1 arm: {arm}")


def _darko_metadata(model: Any) -> dict[str, Any]:
    core = model.model_
    return {
        "fitted_tree_count": int(len(core.trees_)),
        "resolved_learning_rate": float(core.lr_),
        "resolved_depth": int(core.depth),
        "resolved_thread_count": int(core.n_threads_),
        "tree_mode": str(core.tree_mode_),
        "histogram_dtype": str(core.histogram_dtype_),
        "linear_leaves_active": bool(core.linear_leaves_active_),
        "bin_sample_count": int(core.prep_.binner_.sample_count),
    }


def _chimera_metadata(model: Any) -> dict[str, Any]:
    core = model.model_
    return {
        "fitted_tree_count": int(len(core.trees_)),
        "resolved_learning_rate": float(core.lr_),
        "resolved_depth": int(core.depth),
        "resolved_thread_count": int(core.n_threads_),
        "quantize_gradients": bool(core.quantize_gradients),
        "linear_leaves_selected": bool(model.linear_leaves_selected_),
        "cross_features_selected": bool(model.cross_features_selected_),
        "bin_sample_count": None,
    }


def _serialized_sizes(model: Any, arm: str) -> dict[str, Any]:
    common = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    native_bytes = None
    native_format = None
    if arm == DARKO:
        with tempfile.TemporaryDirectory(prefix="darkofit-m1-archive-") as tmp:
            path = Path(tmp) / "model.npz"
            model.save_model(path)
            native_bytes = int(path.stat().st_size)
            native_format = "darkofit_safe_npz"
    return {
        "measurement_stage": "before_first_measured_predict",
        "common_pickle_protocol": int(pickle.HIGHEST_PROTOCOL),
        "common_pickle_bytes": int(len(common)),
        "native_format": native_format,
        "native_bytes": native_bytes,
    }


def _fit_m1(
    arm: str,
    iterations: int,
    X: np.ndarray,
    y: np.ndarray,
    *,
    darkofit_source: Path,
    chimeraboost_source: Path,
):
    model, import_path = _build_estimator(
        arm,
        iterations,
        darkofit_source=darkofit_source,
        chimeraboost_source=chimeraboost_source,
    )
    counter = np.zeros(1, dtype=np.int64)
    original_builder = None
    if arm == DARKO:
        import darkofit.booster as booster_module

        original_builder = booster_module.build_oblivious_tree
        booster_module.build_oblivious_tree = partial(
            original_builder,
            fused_oblivious_counter=counter,
        )
    try:
        started = time.perf_counter_ns()
        model.fit(X, y)
        fit_seconds = (time.perf_counter_ns() - started) / 1e9
    finally:
        if arm == DARKO:
            booster_module.build_oblivious_tree = original_builder
    return model, float(fit_seconds), int(counter[0]), import_path


def run_m1_worker(
    arm: str,
    train_rows: int,
    *,
    darkofit_source: Path,
    chimeraboost_source: Path,
    iterations: int = M1_ITERATIONS,
    holdout_rows: int = HOLDOUT_ROWS,
) -> dict[str, Any]:
    X_train, y_train, X_test, y_test, data_probe = _data(
        train_rows, holdout_rows
    )
    warmup_model, _, _, _ = _fit_m1(
        arm,
        WARMUP_ITERATIONS,
        X_train[:WARMUP_ROWS],
        y_train[:WARMUP_ROWS],
        darkofit_source=darkofit_source,
        chimeraboost_source=chimeraboost_source,
    )
    warmup_prediction = np.asarray(
        warmup_model.predict(X_train[:256]), dtype=np.float64
    )
    if warmup_prediction.shape != (256,):
        raise RuntimeError("M1 prediction warmup produced the wrong shape")
    model, fit_seconds, fused_count, import_path = _fit_m1(
        arm,
        iterations,
        X_train,
        y_train,
        darkofit_source=darkofit_source,
        chimeraboost_source=chimeraboost_source,
    )
    serialization = _serialized_sizes(model, arm)
    started = time.perf_counter_ns()
    prediction = np.asarray(model.predict(X_test), dtype=np.float64)
    predict_seconds = (time.perf_counter_ns() - started) / 1e9

    if (
        prediction.shape != (holdout_rows,)
        or not np.all(np.isfinite(prediction))
        or not math.isfinite(fit_seconds)
        or fit_seconds <= 0.0
        or not math.isfinite(predict_seconds)
        or predict_seconds <= 0.0
    ):
        raise RuntimeError("M1 worker produced invalid output")

    metadata = (
        _darko_metadata(model)
        if arm == DARKO
        else _chimera_metadata(model)
    )
    metadata["fused_engagement_count"] = int(fused_count)
    if metadata["fitted_tree_count"] != int(iterations):
        raise RuntimeError("M1 worker retained the wrong tree count")
    if metadata["resolved_thread_count"] != THREADS:
        raise RuntimeError("M1 worker resolved the wrong thread count")

    rmse = float(mean_squared_error(y_test, prediction) ** 0.5)
    behavior = {
        "arm": arm,
        "train_rows": int(train_rows),
        "holdout_rows": int(holdout_rows),
        "data_probe_sha256": data_probe,
        "prediction_sha256": _array_sha256(prediction),
        "rmse": rmse,
        "metadata": metadata,
    }
    return {
        **behavior,
        "features": int(X_train.shape[1]),
        "iterations": int(iterations),
        "fit_seconds": fit_seconds,
        "predict_seconds": float(predict_seconds),
        "peak_rss_bytes": _peak_rss_bytes(),
        "serialization": serialization,
        "phase_seconds": (
            {
                key: float(value)
                for key, value in (model.timing_ or {}).items()
            }
            if arm == DARKO
            else None
        ),
        "behavior_fingerprint_sha256": _json_sha256(behavior),
        "package_import_path": import_path,
        "thread_environment": {
            key: os.environ.get(key) for key in creator.THREAD_ENV_KEYS
        },
    }


class _ComponentProfiler:
    def __init__(self, tree_module: Any):
        self.tree_module = tree_module
        self.originals: dict[str, Any] = {}
        self.values: dict[str, dict[str, float | int]] = {}

    def install(self, names: Iterable[str], category: str) -> None:
        bucket = self.values.setdefault(
            category, {"calls": 0, "seconds": 0.0}
        )
        for name in names:
            if not hasattr(self.tree_module, name):
                continue
            if name in self.originals:
                raise RuntimeError(f"profile function installed twice: {name}")
            original = getattr(self.tree_module, name)
            self.originals[name] = original

            @wraps(original)
            def measured(*args, __original=original, __bucket=bucket, **kwargs):
                started = time.perf_counter_ns()
                try:
                    return __original(*args, **kwargs)
                finally:
                    __bucket["seconds"] = float(__bucket["seconds"]) + (
                        time.perf_counter_ns() - started
                    ) / 1e9
                    __bucket["calls"] = int(__bucket["calls"]) + 1

            setattr(self.tree_module, name, measured)

    def restore(self) -> None:
        for name, original in reversed(tuple(self.originals.items())):
            setattr(self.tree_module, name, original)
        self.originals.clear()

    def result(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "calls": int(values["calls"]),
                "seconds": float(values["seconds"]),
            }
            for name, values in sorted(self.values.items())
        }


def _fit_q0(
    mode: str,
    iterations: int,
    X: np.ndarray,
    y: np.ndarray,
    *,
    darkofit_source: Path,
    instrument: bool,
):
    model, import_path = _build_estimator(
        DARKO,
        iterations,
        darkofit_source=darkofit_source,
        chimeraboost_source=DEFAULT_CHIMERA_SOURCE,
    )
    import darkofit.booster as booster_module
    import darkofit.tree as tree_module

    profiler = _ComponentProfiler(tree_module)
    if instrument:
        profiler.install(_FUSED_NAMES, "fused_histogram_split")
        profiler.install(_HISTOGRAM_NAMES, "histogram_construction")
        profiler.install(_SPLIT_NAMES, "split_search")
        profiler.install(_SIBLING_NAMES, "sibling_subtraction")
        profiler.install(_LEAF_VALUE_NAMES, "leaf_values")
        profiler.install(_LEAF_ROUTING_NAMES, "leaf_routing")

    counter = np.zeros(1, dtype=np.int64)
    original_builder = booster_module.build_oblivious_tree
    booster_module.build_oblivious_tree = partial(
        original_builder,
        fused_oblivious_kernel=(mode == Q0_PRODUCTION),
        fused_oblivious_counter=counter,
    )
    try:
        started = time.perf_counter_ns()
        model.fit(X, y)
        fit_seconds = (time.perf_counter_ns() - started) / 1e9
    finally:
        booster_module.build_oblivious_tree = original_builder
        profiler.restore()
    return (
        model,
        float(fit_seconds),
        int(counter[0]),
        profiler.result(),
        import_path,
    )


def run_q0_worker(
    mode: str,
    train_rows: int,
    *,
    darkofit_source: Path,
    iterations: int = Q0_ITERATIONS,
    holdout_rows: int = HOLDOUT_ROWS,
) -> dict[str, Any]:
    if mode not in Q0_MODES:
        raise ValueError(f"unknown Q0 profile mode: {mode}")
    X_train, y_train, X_test, y_test, data_probe = _data(
        train_rows, holdout_rows
    )
    warmup_model, _, _, _, _ = _fit_q0(
        mode,
        WARMUP_ITERATIONS,
        X_train[:WARMUP_ROWS],
        y_train[:WARMUP_ROWS],
        darkofit_source=darkofit_source,
        instrument=False,
    )
    warmup_prediction = np.asarray(
        warmup_model.predict(X_train[:256]), dtype=np.float64
    )
    if warmup_prediction.shape != (256,):
        raise RuntimeError("Q0 prediction warmup produced the wrong shape")
    model, fit_seconds, fused_count, components, import_path = _fit_q0(
        mode,
        iterations,
        X_train,
        y_train,
        darkofit_source=darkofit_source,
        instrument=True,
    )
    started = time.perf_counter_ns()
    prediction = np.asarray(model.predict(X_test), dtype=np.float64)
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    if (
        prediction.shape != (holdout_rows,)
        or not np.all(np.isfinite(prediction))
        or not math.isfinite(fit_seconds)
        or fit_seconds <= 0.0
    ):
        raise RuntimeError("Q0 worker produced invalid output")

    metadata = _darko_metadata(model)
    if metadata["fitted_tree_count"] != int(iterations):
        raise RuntimeError("Q0 worker retained the wrong tree count")
    if metadata["resolved_thread_count"] != THREADS:
        raise RuntimeError("Q0 worker resolved the wrong thread count")
    phase_seconds = {
        key: float(value) for key, value in (model.timing_ or {}).items()
    }
    tree_components = (
        float(components["fused_histogram_split"]["seconds"])
        + float(components["histogram_construction"]["seconds"])
        + float(components["split_search"]["seconds"])
        + float(components["sibling_subtraction"]["seconds"])
        + float(components["leaf_values"]["seconds"])
        + float(components["leaf_routing"]["seconds"])
    )
    rmse = float(mean_squared_error(y_test, prediction) ** 0.5)
    behavior = {
        "train_rows": int(train_rows),
        "holdout_rows": int(holdout_rows),
        "data_probe_sha256": data_probe,
        "prediction_sha256": _array_sha256(prediction),
        "rmse": rmse,
        "metadata": metadata,
    }
    return {
        "mode": mode,
        **behavior,
        "features": int(X_train.shape[1]),
        "iterations": int(iterations),
        "fit_seconds": fit_seconds,
        "predict_seconds": float(predict_seconds),
        "peak_rss_bytes": _peak_rss_bytes(),
        "phase_seconds": phase_seconds,
        "components": components,
        "profile_accounting": {
            "timed_tree_components_seconds": tree_components,
            "tree_build_seconds": float(phase_seconds["tree_build"]),
            "unattributed_tree_build_seconds": (
                float(phase_seconds["tree_build"]) - tree_components
            ),
            "production_histogram_and_split_are_fused": (
                mode == Q0_PRODUCTION
            ),
        },
        "fused_engagement_count": int(fused_count),
        "behavior_fingerprint_sha256": _json_sha256(behavior),
        "package_import_path": import_path,
        "thread_environment": {
            key: os.environ.get(key) for key in creator.THREAD_ENV_KEYS
        },
    }


def _pair_summary(
    numerator: list[dict[str, Any]],
    denominator: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    return campaign.paired_ratio_summary(
        [float(row[field]) for row in numerator],
        [float(row[field]) for row in denominator],
        repetitions=len(numerator),
        max_iqr_over_median=MAX_IQR_OVER_MEDIAN,
    )


def _m1_metadata_valid(row: dict[str, Any]) -> bool:
    metadata = row["metadata"]
    arm = row["arm"]
    base = (
        int(metadata["fitted_tree_count"]) == M1_ITERATIONS
        and int(metadata["resolved_thread_count"]) == THREADS
        and int(metadata["resolved_depth"]) == 6
        and float(metadata["resolved_learning_rate"]) == 0.1
    )
    if arm == DARKO:
        return (
            base
            and metadata["tree_mode"] == "catboost"
            and metadata["histogram_dtype"] == "float64"
            and metadata["linear_leaves_active"] is False
            and int(metadata["bin_sample_count"]) == 200_000
        )
    expected_quantized = arm == CHIMERA_QUANTIZED
    return (
        base
        and metadata["quantize_gradients"] is expected_quantized
        and metadata["linear_leaves_selected"] is False
        and metadata["cross_features_selected"] is False
    )


def _validate_m1_results(
    results: list[dict[str, Any]],
) -> dict[tuple[int, int, str], dict[str, Any]]:
    expected = len(TRAIN_ROWS) * len(M1_ARMS) * len(M1_BLOCK_ORDERS)
    if len(results) != expected:
        raise RuntimeError(f"M1 requires exactly {expected} workers")
    coordinates = {
        (block, rows, arm)
        for block in range(len(M1_BLOCK_ORDERS))
        for rows in TRAIN_ROWS
        for arm in M1_ARMS
    }
    by_coordinate: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in results:
        coordinate = (
            int(row["block"]),
            int(row["train_rows"]),
            str(row["arm"]),
        )
        if coordinate not in coordinates:
            raise RuntimeError(f"M1 has an unexpected coordinate: {coordinate}")
        if coordinate in by_coordinate:
            raise RuntimeError(f"M1 has a duplicate coordinate: {coordinate}")
        expected_position = M1_BLOCK_ORDERS[coordinate[0]].index(
            coordinate[2]
        )
        if int(row["position"]) != expected_position:
            raise RuntimeError(f"M1 arm order changed at {coordinate}")
        if "worker_stderr" not in row:
            raise RuntimeError(f"M1 stderr record is missing at {coordinate}")
        by_coordinate[coordinate] = row
    if set(by_coordinate) != coordinates:
        raise RuntimeError("M1 is missing a worker coordinate")
    return by_coordinate


def analyze_m1(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_coordinate = _validate_m1_results(results)
    sizes: dict[str, Any] = {}
    contrast_names = {
        "darkofit_over_chimeraboost_quantized": (DARKO, CHIMERA_QUANTIZED),
        "darkofit_over_chimeraboost_float": (DARKO, CHIMERA_FLOAT),
        "chimeraboost_quantized_over_float": (
            CHIMERA_QUANTIZED,
            CHIMERA_FLOAT,
        ),
    }
    for rows in TRAIN_ROWS:
        by_arm = {
            arm: [
                by_coordinate[(block, rows, arm)]
                for block in range(len(M1_BLOCK_ORDERS))
            ]
            for arm in M1_ARMS
        }
        if len(
            {
                row["data_probe_sha256"]
                for arm_rows in by_arm.values()
                for row in arm_rows
            }
        ) != 1:
            raise RuntimeError(f"M1 data identity differs at {rows} rows")

        contrasts = {}
        for name, (numerator_arm, denominator_arm) in contrast_names.items():
            numerator = by_arm[numerator_arm]
            denominator = by_arm[denominator_arm]
            contrasts[name] = {
                "fit_ratio": _pair_summary(
                    numerator, denominator, "fit_seconds"
                ),
                "rss_ratio": _pair_summary(
                    numerator, denominator, "peak_rss_bytes"
                ),
                "rmse_ratio": float(
                    numerator[0]["rmse"] / denominator[0]["rmse"]
                ),
            }
        sizes[str(rows)] = {
            "arms": {
                arm: {
                    "rmse": float(by_arm[arm][0]["rmse"]),
                    "fit_seconds": [
                        float(row["fit_seconds"]) for row in by_arm[arm]
                    ],
                    "predict_seconds_median": float(
                        np.median(
                            [row["predict_seconds"] for row in by_arm[arm]]
                        )
                    ),
                    "peak_rss_bytes_median": float(
                        np.median(
                            [row["peak_rss_bytes"] for row in by_arm[arm]]
                        )
                    ),
                    "common_pickle_bytes": int(
                        np.median(
                            [
                                row["serialization"]["common_pickle_bytes"]
                                for row in by_arm[arm]
                            ]
                        )
                    ),
                    "behavior_stable": len(
                        {
                            row["behavior_fingerprint_sha256"]
                            for row in by_arm[arm]
                        }
                    )
                    == 1,
                }
                for arm in M1_ARMS
            },
            "contrasts": contrasts,
        }

    fit_geomeans = {}
    for name in contrast_names:
        fit_geomeans[name] = _geomean(
            sizes[str(rows)]["contrasts"][name]["fit_ratio"]["median_ratio"]
            for rows in TRAIN_ROWS
        )

    all_behavior_stable = all(
        arm_summary["behavior_stable"]
        for size in sizes.values()
        for arm_summary in size["arms"].values()
    )
    no_worker_stderr = all(not row.get("worker_stderr") for row in results)
    metadata_valid = all(_m1_metadata_valid(row) for row in results)
    darko_fused_engaged = all(
        int(row["metadata"]["fused_engagement_count"]) > 0
        for row in results
        if row["arm"] == DARKO
    )
    quant_contrast = "chimeraboost_quantized_over_float"
    quant_timing_stable = all(
        sizes[str(rows)]["contrasts"][quant_contrast]["fit_ratio"]["stable"]
        for rows in TRAIN_ROWS
    )
    quant_quality_neutral = all(
        sizes[str(rows)]["contrasts"][quant_contrast]["rmse_ratio"]
        <= MAX_DONOR_RMSE_RATIO
        for rows in TRAIN_ROWS
    )
    donor_material = (
        all_behavior_stable
        and no_worker_stderr
        and metadata_valid
        and darko_fused_engaged
        and quant_timing_stable
        and quant_quality_neutral
        and fit_geomeans[quant_contrast] <= MATERIAL_DONOR_GEOMEAN_RATIO
        and max(
            sizes[str(rows)]["contrasts"][quant_contrast]["fit_ratio"][
                "median_ratio"
            ]
            for rows in TRAIN_ROWS
        )
        <= MAX_DONOR_SIZE_RATIO
    )
    darko_contrast = "darkofit_over_chimeraboost_quantized"
    darkofit_currently_faster = (
        all_behavior_stable
        and no_worker_stderr
        and metadata_valid
        and darko_fused_engaged
        and all(
            sizes[str(rows)]["contrasts"][darko_contrast]["fit_ratio"]["stable"]
            for rows in TRAIN_ROWS
        )
        and fit_geomeans[darko_contrast] < 1.0
        and max(
            sizes[str(rows)]["contrasts"][darko_contrast]["fit_ratio"][
                "median_ratio"
            ]
            for rows in TRAIN_ROWS
        )
        < 1.0
    )
    return {
        "sizes": sizes,
        "equal_size_fit_geomean_ratios": fit_geomeans,
        "integrity": {
            "all_behavior_stable": all_behavior_stable,
            "no_worker_stderr": no_worker_stderr,
            "metadata_valid": metadata_valid,
            "darkofit_fused_engaged": darko_fused_engaged,
        },
        "descriptive_verdicts": {
            "darkofit_faster_than_current_quantized_chimera": (
                darkofit_currently_faster
            ),
            "quantized_float_timing_stable": quant_timing_stable,
            "quantized_float_quality_neutral": quant_quality_neutral,
            "material_quantization_donor_signal": donor_material,
        },
        "g_m_input": (
            "material_quantization_donor_signal"
            if donor_material
            else "no_material_quantization_donor_signal"
        ),
        "certification_or_default_change_authorized": False,
    }


def _validate_q0_results(
    results: list[dict[str, Any]],
) -> dict[tuple[int, int, str], dict[str, Any]]:
    expected = len(TRAIN_ROWS) * len(Q0_MODES) * len(Q0_BLOCK_ORDERS)
    if len(results) != expected:
        raise RuntimeError(f"Q0 requires exactly {expected} workers")
    coordinates = {
        (block, rows, mode)
        for block in range(len(Q0_BLOCK_ORDERS))
        for rows in TRAIN_ROWS
        for mode in Q0_MODES
    }
    by_coordinate: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in results:
        coordinate = (
            int(row["block"]),
            int(row["train_rows"]),
            str(row["mode"]),
        )
        if coordinate not in coordinates:
            raise RuntimeError(f"Q0 has an unexpected coordinate: {coordinate}")
        if coordinate in by_coordinate:
            raise RuntimeError(f"Q0 has a duplicate coordinate: {coordinate}")
        expected_position = Q0_BLOCK_ORDERS[coordinate[0]].index(
            coordinate[2]
        )
        if int(row["position"]) != expected_position:
            raise RuntimeError(f"Q0 mode order changed at {coordinate}")
        if "worker_stderr" not in row:
            raise RuntimeError(f"Q0 stderr record is missing at {coordinate}")
        by_coordinate[coordinate] = row
    if set(by_coordinate) != coordinates:
        raise RuntimeError("Q0 is missing a worker coordinate")
    return by_coordinate


def analyze_q0(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_coordinate = _validate_q0_results(results)
    sizes: dict[str, Any] = {}
    behavior_exact = True
    production_engaged = True
    reference_decomposed = True
    sibling_inactive = True
    component_accounting_valid = True
    for rows in TRAIN_ROWS:
        production = [
            by_coordinate[(block, rows, Q0_PRODUCTION)]
            for block in range(len(Q0_BLOCK_ORDERS))
        ]
        reference = [
            by_coordinate[(block, rows, Q0_REFERENCE)]
            for block in range(len(Q0_BLOCK_ORDERS))
        ]
        if len(
            {
                row["data_probe_sha256"]
                for row in (*production, *reference)
            }
        ) != 1:
            raise RuntimeError(f"Q0 data identity differs at {rows} rows")
        fingerprints = {
            row["behavior_fingerprint_sha256"]
            for row in (*production, *reference)
        }
        behavior_exact = behavior_exact and len(fingerprints) == 1
        production_engaged = production_engaged and all(
            int(row["fused_engagement_count"]) > 0
            and int(
                row["components"]["fused_histogram_split"]["calls"]
            )
            > 0
            and int(row["components"]["histogram_construction"]["calls"]) == 0
            and int(row["components"]["split_search"]["calls"]) == 0
            for row in production
        )
        reference_decomposed = reference_decomposed and all(
            int(row["fused_engagement_count"]) == 0
            and int(
                row["components"]["fused_histogram_split"]["calls"]
            )
            == 0
            and int(row["components"]["histogram_construction"]["calls"]) > 0
            and int(row["components"]["split_search"]["calls"]) > 0
            for row in reference
        )
        sibling_inactive = sibling_inactive and all(
            int(row["components"]["sibling_subtraction"]["calls"]) == 0
            for row in (*production, *reference)
        )
        eligible_share_values = [
            float(row["components"]["fused_histogram_split"]["seconds"])
            / float(row["fit_seconds"])
            for row in production
        ]
        component_accounting_valid = component_accounting_valid and all(
            0.0 < share <= 1.0 for share in eligible_share_values
        )
        component_accounting_valid = component_accounting_valid and all(
            float(row["profile_accounting"]["timed_tree_components_seconds"])
            <= 1.05
            * float(row["profile_accounting"]["tree_build_seconds"])
            for row in (*production, *reference)
        )
        eligible_share = float(np.median(eligible_share_values))
        projected_ratio = (1.0 - eligible_share) + (
            eligible_share / Q_KERNEL_SPEEDUP_PRIOR
        )
        infinite_speed_ratio = 1.0 - eligible_share
        sizes[str(rows)] = {
            "production_fit_seconds": [
                float(row["fit_seconds"]) for row in production
            ],
            "reference_fit_seconds": [
                float(row["fit_seconds"]) for row in reference
            ],
            "reference_over_production_fit_ratio": _pair_summary(
                reference, production, "fit_seconds"
            ),
            "eligible_fused_share_values": eligible_share_values,
            "eligible_fused_share_median": eligible_share,
            "projected_ratio_at_1_30x_kernel": projected_ratio,
            "infinite_kernel_speed_ratio": infinite_speed_ratio,
            "production_phase_seconds_median": {
                phase: float(
                    np.median(
                        [row["phase_seconds"][phase] for row in production]
                    )
                )
                for phase in production[0]["phase_seconds"]
            },
            "production_component_seconds_median": {
                component: float(
                    np.median(
                        [
                            row["components"][component]["seconds"]
                            for row in production
                        ]
                    )
                )
                for component in production[0]["components"]
            },
            "reference_component_seconds_median": {
                component: float(
                    np.median(
                        [
                            row["components"][component]["seconds"]
                            for row in reference
                        ]
                    )
                )
                for component in reference[0]["components"]
            },
            "behavior_exact_across_modes": len(fingerprints) == 1,
        }

    projected_geomean_ratio = _geomean(
        sizes[str(rows)]["projected_ratio_at_1_30x_kernel"]
        for rows in TRAIN_ROWS
    )
    infinite_geomean_ratio = _geomean(
        sizes[str(rows)]["infinite_kernel_speed_ratio"]
        for rows in TRAIN_ROWS
    )
    no_worker_stderr = all(not row.get("worker_stderr") for row in results)
    metadata_valid = all(
        int(row["metadata"]["fitted_tree_count"]) == Q0_ITERATIONS
        and int(row["metadata"]["resolved_thread_count"]) == THREADS
        and int(row["metadata"]["resolved_depth"]) == 6
        and float(row["metadata"]["resolved_learning_rate"]) == 0.1
        and row["metadata"]["tree_mode"] == "catboost"
        and row["metadata"]["histogram_dtype"] == "float64"
        and row["metadata"]["linear_leaves_active"] is False
        and int(row["metadata"]["bin_sample_count"]) == 200_000
        for row in results
    )
    integrity_passed = (
        behavior_exact
        and production_engaged
        and reference_decomposed
        and sibling_inactive
        and component_accounting_valid
        and no_worker_stderr
        and metadata_valid
    )
    conservative_projection_clears_budget = (
        projected_geomean_ratio <= Q_MAX_GEOMEAN_RATIO
    )
    profile_supports_q_funding = (
        integrity_passed and conservative_projection_clears_budget
    )
    if not integrity_passed:
        disposition = "inconclusive_profile_integrity_failure"
    elif profile_supports_q_funding:
        disposition = "eligible_for_g_m_quantization_funding_decision"
    else:
        disposition = "close_quantization_before_prototype"
    return {
        "sizes": sizes,
        "speed_budget": {
            "minimum_end_to_end_reduction": Q_MIN_END_TO_END_REDUCTION,
            "maximum_geomean_candidate_control_ratio": Q_MAX_GEOMEAN_RATIO,
            "maximum_single_size_ratio": Q_MAX_SIZE_RATIO,
            "kernel_speedup_prior": Q_KERNEL_SPEEDUP_PRIOR,
            "equal_share_required_at_prior": Q_REQUIRED_EQUAL_SHARE,
        },
        "projection": {
            "geomean_ratio_at_1_30x_kernel": projected_geomean_ratio,
            "end_to_end_reduction_at_1_30x_kernel": (
                1.0 - projected_geomean_ratio
            ),
            "infinite_kernel_speed_geomean_ratio": infinite_geomean_ratio,
            "infinite_kernel_speed_reduction": 1.0
            - infinite_geomean_ratio,
            "conservative_projection_clears_budget": (
                conservative_projection_clears_budget
            ),
        },
        "integrity": {
            "behavior_exact_across_modes": behavior_exact,
            "production_fused_engaged": production_engaged,
            "reference_histogram_and_split_decomposed": reference_decomposed,
            "sibling_subtraction_inactive": sibling_inactive,
            "component_accounting_valid": component_accounting_valid,
            "no_worker_stderr": no_worker_stderr,
            "metadata_valid": metadata_valid,
            "passed": integrity_passed,
        },
        "profile_supports_q_funding": profile_supports_q_funding,
        "disposition": disposition,
        "prototype_or_public_change_authorized": False,
    }


def _worker_environment() -> dict[str, str]:
    environment = basketball.worker_environment(THREADS)
    environment.update(
        {
            "DARKOFIT_WARMUP": "0",
            "CHIMERABOOST_WARMUP": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(ROOT),
            "NUMBA_CACHE_DIR": (
                "/private/tmp/darkofit-wave1-numba-cache"
            ),
            "MPLCONFIGDIR": "/private/tmp/darkofit-wave1-mpl-cache",
        }
    )
    return environment


def _run_worker(
    *,
    campaign_name: str,
    train_rows: int,
    darkofit_source: Path,
    chimeraboost_source: Path,
    arm: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-campaign",
        campaign_name,
        "--worker-rows",
        str(train_rows),
        "--darkofit-source",
        str(darkofit_source),
        "--chimeraboost-source",
        str(chimeraboost_source),
    ]
    if arm is not None:
        command.extend(("--worker-arm", arm))
    if mode is not None:
        command.extend(("--worker-profile-mode", mode))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_worker_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(WORKER_RESULT_PREFIX)
    ]
    label = arm if arm is not None else mode
    if completed.returncode or len(lines) != 1:
        raise RuntimeError(
            f"{campaign_name} worker {label}/{train_rows} failed with "
            f"{completed.returncode}\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    result = json.loads(lines[0][len(WORKER_RESULT_PREFIX) :])
    result["worker_stdout"] = (
        "\n".join(
            line
            for line in completed.stdout.splitlines()
            if not line.startswith(WORKER_RESULT_PREFIX)
        ).strip()
        or None
    )
    result["worker_stderr"] = completed.stderr.strip() or None
    return result


def _source_states(
    darkofit_source: Path,
    chimeraboost_source: Path,
) -> dict[str, dict[str, Any]]:
    states = {
        "harness": creator.git_state(ROOT),
        "darkofit": creator.git_state(darkofit_source),
        "chimeraboost": creator.git_state(chimeraboost_source),
    }
    if not all(state["clean"] for state in states.values()):
        raise RuntimeError("M1/Q0 requires clean harness and source trees")
    if states["darkofit"]["head"] != DARKO_SOURCE_HEAD:
        raise RuntimeError("M1/Q0 DarkoFit source head changed")
    if states["chimeraboost"]["head"] != CHIMERA_SOURCE_HEAD:
        raise RuntimeError("M1/Q0 ChimeraBoost source head changed")
    return states


def _assert_source_states(
    expected: dict[str, dict[str, Any]],
    darkofit_source: Path,
    chimeraboost_source: Path,
) -> None:
    actual = _source_states(darkofit_source, chimeraboost_source)
    if actual != expected:
        raise RuntimeError("M1/Q0 source state changed during execution")


def _artifact_base(
    campaign_name: str,
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign": campaign_name,
        "protocol": {
            "name": "wave1_m1_q0_20260720",
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": _sha256(PROTOCOL),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "train_rows": list(TRAIN_ROWS),
            "holdout_rows": HOLDOUT_ROWS,
            "features": FEATURES,
            "threads": THREADS,
            "warmup_rows": WARMUP_ROWS,
            "warmup_iterations": WARMUP_ITERATIONS,
            "lockbox_data_used": False,
            "default_change_authorized": False,
        },
        "sources": sources,
        "environment": {
            "machine": creator._machine_details(),
            "dependencies": creator._dependency_versions(),
        },
    }


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    sources = _source_states(
        args.darkofit_source, args.chimeraboost_source
    )
    results: list[dict[str, Any]] = []
    if args.campaign == "m1":
        for block, order in enumerate(M1_BLOCK_ORDERS):
            for train_rows in TRAIN_ROWS:
                for position, arm in enumerate(order):
                    _assert_source_states(
                        sources,
                        args.darkofit_source,
                        args.chimeraboost_source,
                    )
                    print(
                        f"M1 block {block + 1}/{len(M1_BLOCK_ORDERS)} "
                        f"rows={train_rows} position={position + 1} arm={arm}",
                        flush=True,
                    )
                    result = _run_worker(
                        campaign_name="m1",
                        train_rows=train_rows,
                        arm=arm,
                        darkofit_source=args.darkofit_source,
                        chimeraboost_source=args.chimeraboost_source,
                    )
                    result["block"] = int(block)
                    result["position"] = int(position)
                    results.append(result)
        analysis = analyze_m1(results)
        campaign_protocol = {
            "iterations": M1_ITERATIONS,
            "arms": list(M1_ARMS),
            "block_orders": [list(order) for order in M1_BLOCK_ORDERS],
            "material_donor_geomean_ratio": (
                MATERIAL_DONOR_GEOMEAN_RATIO
            ),
            "max_donor_size_ratio": MAX_DONOR_SIZE_RATIO,
            "max_donor_rmse_ratio": MAX_DONOR_RMSE_RATIO,
            "max_iqr_over_median": MAX_IQR_OVER_MEDIAN,
        }
    else:
        for block, order in enumerate(Q0_BLOCK_ORDERS):
            for train_rows in TRAIN_ROWS:
                for position, mode in enumerate(order):
                    _assert_source_states(
                        sources,
                        args.darkofit_source,
                        args.chimeraboost_source,
                    )
                    print(
                        f"Q0 block {block + 1}/{len(Q0_BLOCK_ORDERS)} "
                        f"rows={train_rows} position={position + 1} mode={mode}",
                        flush=True,
                    )
                    result = _run_worker(
                        campaign_name="q0",
                        train_rows=train_rows,
                        mode=mode,
                        darkofit_source=args.darkofit_source,
                        chimeraboost_source=args.chimeraboost_source,
                    )
                    result["block"] = int(block)
                    result["position"] = int(position)
                    results.append(result)
        analysis = analyze_q0(results)
        campaign_protocol = {
            "iterations": Q0_ITERATIONS,
            "modes": list(Q0_MODES),
            "block_orders": [list(order) for order in Q0_BLOCK_ORDERS],
            "minimum_end_to_end_reduction": (
                Q_MIN_END_TO_END_REDUCTION
            ),
            "max_geomean_ratio": Q_MAX_GEOMEAN_RATIO,
            "max_size_ratio": Q_MAX_SIZE_RATIO,
            "kernel_speedup_prior": Q_KERNEL_SPEEDUP_PRIOR,
            "required_equal_share": Q_REQUIRED_EQUAL_SHARE,
        }
    _assert_source_states(
        sources, args.darkofit_source, args.chimeraboost_source
    )
    artifact = {
        **_artifact_base(args.campaign, sources),
        "campaign_protocol": campaign_protocol,
        "results": results,
        "analysis": analysis,
    }
    creator._atomic_write_bytes(
        args.output,
        (
            json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8"),
    )
    print(f"disposition: {analysis.get('disposition', analysis['g_m_input'])}")
    print(f"wrote {args.output}")
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=("m1", "q0"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--darkofit-source", type=Path, default=DEFAULT_DARKO_SOURCE
    )
    parser.add_argument(
        "--chimeraboost-source", type=Path, default=DEFAULT_CHIMERA_SOURCE
    )
    parser.add_argument("--worker-campaign", choices=("m1", "q0"))
    parser.add_argument("--worker-arm", choices=M1_ARMS)
    parser.add_argument("--worker-profile-mode", choices=Q0_MODES)
    parser.add_argument("--worker-rows", type=int)
    parser.add_argument("--worker-iterations", type=int)
    parser.add_argument("--worker-holdout-rows", type=int, default=HOLDOUT_ROWS)
    args = parser.parse_args(argv)

    args.darkofit_source = creator._absolute_lexical_path(
        args.darkofit_source
    )
    args.chimeraboost_source = creator._absolute_lexical_path(
        args.chimeraboost_source
    )
    worker = args.worker_campaign is not None
    if worker:
        if args.campaign is not None or args.output is not None:
            parser.error("worker mode does not accept --campaign/--output")
        if args.worker_rows is None or args.worker_rows <= 0:
            parser.error("worker mode requires positive --worker-rows")
        if args.worker_holdout_rows <= 0:
            parser.error("--worker-holdout-rows must be positive")
        if args.worker_campaign == "m1":
            if args.worker_arm is None or args.worker_profile_mode is not None:
                parser.error("M1 worker requires only --worker-arm")
        else:
            if args.worker_profile_mode is None or args.worker_arm is not None:
                parser.error(
                    "Q0 worker requires only --worker-profile-mode"
                )
    else:
        if args.campaign is None:
            parser.error("parent mode requires --campaign")
        if any(
            value is not None
            for value in (
                args.worker_arm,
                args.worker_profile_mode,
                args.worker_rows,
                args.worker_iterations,
            )
        ):
            parser.error("parent mode does not accept worker arguments")
        if args.output is None:
            args.output = (
                DEFAULT_M1_OUTPUT
                if args.campaign == "m1"
                else DEFAULT_Q0_OUTPUT
            )
        args.output = creator._absolute_lexical_path(args.output)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_campaign == "m1":
        result = run_m1_worker(
            args.worker_arm,
            args.worker_rows,
            darkofit_source=args.darkofit_source,
            chimeraboost_source=args.chimeraboost_source,
            iterations=(
                M1_ITERATIONS
                if args.worker_iterations is None
                else args.worker_iterations
            ),
            holdout_rows=args.worker_holdout_rows,
        )
        print(
            WORKER_RESULT_PREFIX
            + json.dumps(result, sort_keys=True, allow_nan=False)
        )
        return 0
    if args.worker_campaign == "q0":
        result = run_q0_worker(
            args.worker_profile_mode,
            args.worker_rows,
            darkofit_source=args.darkofit_source,
            iterations=(
                Q0_ITERATIONS
                if args.worker_iterations is None
                else args.worker_iterations
            ),
            holdout_rows=args.worker_holdout_rows,
        )
        print(
            WORKER_RESULT_PREFIX
            + json.dumps(result, sort_keys=True, allow_nan=False)
        )
        return 0
    run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
