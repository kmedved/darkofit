#!/usr/bin/env python3
"""Run the frozen 25-lineage T5 composite confirmation campaign."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
CHIMERA_ROOT = ROOT.parent / "chimeraboost"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import basketball_harness as basketball  # noqa: E402
from benchmarks import build_ctr23_contamination_registry as fingerprints  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402
from benchmarks.run_smooth_cross_features import candidate_pairs  # noqa: E402


CONTROL = "darko_product_default"
COMPOSITE = "darko_t5_composite"
CHIMERA = "chimeraboost_0_15_0"
CATBOOST = "catboost_1_2_10"
CONFIGS = (CONTROL, COMPOSITE, CHIMERA, CATBOOST)
FOLDS = (0, 1, 2)
REGISTRY = ROOT / "benchmarks" / "t5_composite_registry.json"
PROTOCOL = ROOT / "benchmarks" / "t5_composite_registry_protocol.md"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "t5_composite_confirmation_raw.json"
DEFAULT_SPOOL_DIRECTORY = (
    ROOT / ".cache" / "t5-composite-confirmation-spool-v1"
)
EXPECTED_REGISTRY_FILE_SHA256 = (
    "0cacff16214731dc89292abd31cc6abbd1ea2fc244a6c26b681e1206cbe301bd"
)
EXPECTED_REGISTRY_CANONICAL_SHA256 = (
    "683cdb780e8e9eefbe0aeb4b3bd6f8f95ccf4b6e62397a117beb15f068db77ab"
)
EXPECTED_CHIMERA_HEAD = "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d"
RANDOM_STATE = 4
THREADS_PER_WORKER = 6
CONCURRENT_WORKERS = 3
SIZE_GATE = 2_000
VALIDATION_FRACTION = 0.20
OUTER_GUARD_RATIO = 0.995
CROSS_GUARD_RATIO = 0.95
SELECTION_ROUNDS = 100
PREDICTION_BLOCK_SECONDS = 0.25
PREDICTION_MIN_CALLS = 5
PREDICTION_MAX_CALLS = 20_000
WORKER_PREFIX = "T5_COMPOSITE_WORKER_RESULT="


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value, dtype="<f8") -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("worker peak RSS is unavailable")
    return value


def _registry():
    if _sha256(REGISTRY) != EXPECTED_REGISTRY_FILE_SHA256:
        raise RuntimeError("T5 registry file identity changed")
    payload = json.loads(REGISTRY.read_text())
    if payload["registry_sha256"] != EXPECTED_REGISTRY_CANONICAL_SHA256:
        raise RuntimeError("T5 registry canonical identity changed")
    if (
        payload["confirmation_outcomes_inspected"]
        or not payload["confirmation_run_authorized"]
        or payload["default_promotion_authorized"]
        or payload["lockbox_data_used"]
    ):
        raise RuntimeError("T5 registry authorization state is invalid")
    if (
        payload["task_count"] != 25
        or payload["coordinate_count"] != 75
        or not payload["power_analysis"]["passes"]
    ):
        raise RuntimeError("T5 registry design changed")
    return payload, {int(row["task_id"]): row for row in payload["tasks"]}


def _source_state():
    darko = creator.git_state(ROOT)
    chimera = creator.git_state(CHIMERA_ROOT)
    if not darko["clean"] or not chimera["clean"]:
        raise RuntimeError("T5 execution requires clean source trees")
    if chimera["head"] != EXPECTED_CHIMERA_HEAD:
        raise RuntimeError("T5 ChimeraBoost source changed")
    return darko, chimera


def _load_task(task_id: int, row: dict[str, Any]):
    import openml

    task = openml.tasks.get_task(task_id, download_splits=True)
    dataset = task.get_dataset()
    X, y, categorical, names = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )
    expected = row["task_record"]
    if (
        int(dataset.dataset_id) != int(row["dataset_id"])
        or str(dataset.name) != str(expected["dataset_name"])
        or list(X.columns) != list(names)
        or len(X) != int(expected["fingerprint"]["n_rows"])
        or X.shape[1] != int(expected["fingerprint"]["n_features"])
        or str(dataset.md5_checksum) != str(expected["openml_declared_md5"])
    ):
        raise RuntimeError(f"T5 task {task_id} metadata changed")
    observed_fingerprint = fingerprints.dataset_fingerprint(X, y)
    if observed_fingerprint != expected["fingerprint"]:
        raise RuntimeError(f"T5 task {task_id} data fingerprint changed")
    y = pd.to_numeric(y, errors="raise").astype(np.float64)
    if not np.all(np.isfinite(y.to_numpy())):
        raise RuntimeError(f"T5 task {task_id} target is nonfinite")
    categorical_indices = [
        index
        for index, (declared, dtype) in enumerate(zip(categorical, X.dtypes))
        if bool(declared) or not is_numeric_dtype(dtype)
    ]
    return task, X, y, categorical_indices


def _expected_split(row, fold):
    matches = [
        coordinate
        for coordinate in row["task_record"]["official_splits"]["coordinates"]
        if (
            int(coordinate["repeat"]) == 0
            and int(coordinate["fold"]) == int(fold)
            and int(coordinate["sample"]) == 0
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(f"T5 task {row['task_id']} fold {fold} is not frozen")
    return matches[0]


def _verify_split(row, fold, train, test):
    expected = _expected_split(row, fold)
    observed = {
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "train_index_sha256": _array_sha256(train, dtype="<i8"),
        "test_index_sha256": _array_sha256(test, dtype="<i8"),
    }
    for key, value in observed.items():
        if value != expected[key]:
            raise RuntimeError(
                f"T5 task {row['task_id']} fold {fold} {key} changed"
            )
    return observed


def _take(frame, indices):
    return frame.iloc[np.asarray(indices, dtype=np.int64)]


def _selection_split(n_rows: int):
    splitter = ShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_FRACTION,
        random_state=RANDOM_STATE,
    )
    train, validation = next(splitter.split(np.arange(n_rows)))
    return train, validation, {
        "policy": "ShuffleSplit",
        "random_state": RANDOM_STATE,
        "validation_fraction": VALIDATION_FRACTION,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_positions_sha256": _array_sha256(train, dtype="<i8"),
        "validation_positions_sha256": _array_sha256(
            validation, dtype="<i8"
        ),
    }


def _fit(model, X, y, cat, *, eval_set=None, ordinal_features=None):
    started = time.perf_counter_ns()
    model.fit(
        X,
        y,
        cat_features=cat or None,
        eval_set=eval_set,
        ordinal_features=ordinal_features or None,
    )
    seconds = (time.perf_counter_ns() - started) / 1e9
    return model, float(seconds)


def _selection_record(name, model, seconds):
    score = float(model.best_score_)
    if not math.isfinite(score) or score <= 0:
        raise RuntimeError(f"T5 selection score is invalid for {name}")
    validation = dict(model.model_.auto_params_.get("validation_split", {}))
    if validation.get("source") != "explicit_eval_set":
        raise RuntimeError(f"T5 selection fit {name} missed explicit eval set")
    metadata = basketball.extract_fit_metadata(model)
    if metadata["final_fit"]["stop_reason"] not in {
        "early_stopping",
        "iteration_limit",
    }:
        raise RuntimeError(f"T5 selection fit {name} stopped unexpectedly")
    result = {
        "name": name,
        "validation_rmse": score,
        "fit_seconds": float(seconds),
        "fit_metadata": metadata,
        "validation": validation,
    }
    tree_selection = getattr(model, "tree_mode_selection_", None)
    if tree_selection is not None:
        result["tree_mode_selection"] = tree_selection
    return result


def _default_model():
    from darkofit import DarkoRegressor

    return DarkoRegressor(
        random_state=RANDOM_STATE,
        thread_count=THREADS_PER_WORKER,
        diagnostic_warnings="never",
    )


def _default_audition_model():
    from darkofit import DarkoRegressor

    return DarkoRegressor(
        random_state=RANDOM_STATE,
        thread_count=THREADS_PER_WORKER,
        diagnostic_warnings="never",
        early_stopping=False,
        use_best_model=True,
        refit=False,
    )


def _challenger_model(*, tree_mode="auto", linear_leaves=False):
    from darkofit import DarkoRegressor

    return DarkoRegressor(
        iterations=10_000,
        learning_rate=0.1,
        l2_leaf_reg=3.0,
        max_bins=128,
        ts_permutations=1,
        tree_mode=tree_mode,
        selection_rounds=(SELECTION_ROUNDS if tree_mode == "auto" else None),
        linear_leaves=bool(linear_leaves),
        early_stopping=True,
        use_best_model=True,
        refit=False,
        random_state=RANDOM_STATE,
        thread_count=THREADS_PER_WORKER,
        diagnostic_warnings="never",
    )


def _final_challenger_model(*, tree_mode, linear_leaves, iterations):
    from darkofit import DarkoRegressor

    return DarkoRegressor(
        iterations=int(iterations),
        learning_rate=0.1,
        l2_leaf_reg=3.0,
        max_bins=128,
        ts_permutations=1,
        tree_mode=str(tree_mode),
        selection_rounds=None,
        linear_leaves=bool(linear_leaves),
        early_stopping=False,
        use_best_model=False,
        refit=False,
        random_state=RANDOM_STATE,
        thread_count=THREADS_PER_WORKER,
        diagnostic_warnings="never",
    )


def _cross_pairs(model, X, categorical_indices):
    importances = np.asarray(model.feature_importances_, dtype=np.float64)
    return candidate_pairs(importances, categorical_indices, X.shape[1])


def _augment_crosses(X: pd.DataFrame, pairs):
    if not pairs:
        return X.copy()
    result = X.copy()
    for left, right, operation in pairs:
        left_values = pd.to_numeric(X.iloc[:, left], errors="raise").to_numpy(
            dtype=np.float64
        )
        right_values = pd.to_numeric(
            X.iloc[:, right], errors="raise"
        ).to_numpy(dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            values = (
                left_values - right_values
                if operation == "diff"
                else left_values * right_values
            )
        values = np.asarray(values, dtype=np.float64)
        values[~np.isfinite(values)] = np.nan
        name = f"__darkofit_cross_{left}_{right}_{operation}"
        if name in result.columns:
            raise RuntimeError(f"T5 cross column collision: {name}")
        result[name] = values
    return result


def _timed_predict(predict: Callable[[], np.ndarray]):
    prediction = np.asarray(predict(), dtype=np.float64)
    if prediction.ndim != 1 or not np.all(np.isfinite(prediction)):
        raise RuntimeError("T5 prediction is invalid")
    durations = []
    total = 0.0
    last = None
    while (
        len(durations) < PREDICTION_MIN_CALLS
        or total < PREDICTION_BLOCK_SECONDS
    ):
        if len(durations) >= PREDICTION_MAX_CALLS:
            raise RuntimeError("T5 prediction timing did not reach block length")
        started = time.perf_counter_ns()
        last = np.asarray(predict(), dtype=np.float64)
        elapsed = (time.perf_counter_ns() - started) / 1e9
        durations.append(float(elapsed))
        total += elapsed
    if not np.array_equal(prediction, last):
        raise RuntimeError("T5 repeated prediction changed")
    return prediction, {
        "per_call_median_seconds": float(np.median(durations)),
        "per_call_min_seconds": float(np.min(durations)),
        "per_call_max_seconds": float(np.max(durations)),
        "total_seconds": float(total),
        "call_count": len(durations),
        "minimum_block_seconds": PREDICTION_BLOCK_SECONDS,
    }


def _fit_control(X_train, y_train, cat, X_test):
    model, fit_seconds = _fit(
        _default_model(),
        X_train,
        y_train,
        cat,
    )
    prediction, timing = _timed_predict(
        lambda: model.predict(X_test)
    )
    return prediction, fit_seconds, timing, {
        "kind": CONTROL,
        "engaged": False,
        "selected_configuration": "product_default",
        "final_fit": basketball.extract_fit_metadata(model),
    }


def _fit_composite(X_train, y_train, cat, X_test, ordinal_features):
    if len(X_train) < SIZE_GATE:
        prediction, fit_seconds, timing, metadata = _fit_control(
            X_train, y_train, cat, X_test
        )
        metadata.update(
            {
                "kind": COMPOSITE,
                "decline_reason": "below_size_gate",
                "size_gate": SIZE_GATE,
                "total_selection_fit_seconds": 0.0,
            }
        )
        return prediction, fit_seconds, timing, metadata

    train, validation, split = _selection_split(len(X_train))
    X_select = _take(X_train, train)
    y_select = _take(y_train, train)
    X_validation = _take(X_train, validation)
    y_validation = _take(y_train, validation)
    eval_set = (X_validation, y_validation)
    records = []

    control, seconds = _fit(
        _default_audition_model(),
        X_select,
        y_select,
        cat,
        eval_set=eval_set,
    )
    control_record = _selection_record("control_audition", control, seconds)
    records.append(control_record)

    uncrossed, seconds = _fit(
        _challenger_model(),
        X_select,
        y_select,
        cat,
        eval_set=eval_set,
        ordinal_features=ordinal_features,
    )
    auto_record = _selection_record("challenger_auto", uncrossed, seconds)
    records.append(auto_record)
    selected_tree_mode = str(uncrossed.model_.tree_mode_)
    selected_linear = False

    if selected_tree_mode == "catboost":
        linear, seconds = _fit(
            _challenger_model(tree_mode="catboost", linear_leaves=True),
            X_select,
            y_select,
            cat,
            eval_set=eval_set,
            ordinal_features=ordinal_features,
        )
        linear_record = _selection_record(
            "challenger_catboost_linear", linear, seconds
        )
        records.append(linear_record)
        if linear_record["validation_rmse"] < auto_record["validation_rmse"]:
            uncrossed = linear
            selected_linear = True

    uncrossed_score = float(uncrossed.best_score_)
    pairs = _cross_pairs(uncrossed, X_select, cat)
    selected_crosses = False
    crossed = None
    if pairs:
        cross_started = time.perf_counter_ns()
        X_select_cross = _augment_crosses(X_select, pairs)
        X_validation_cross = _augment_crosses(X_validation, pairs)
        transform_seconds = (time.perf_counter_ns() - cross_started) / 1e9
        crossed, seconds = _fit(
            _challenger_model(
                tree_mode=selected_tree_mode,
                linear_leaves=selected_linear,
            ),
            X_select_cross,
            y_select,
            cat,
            eval_set=(X_validation_cross, y_validation),
            ordinal_features=ordinal_features,
        )
        seconds += transform_seconds
        cross_record = _selection_record(
            "challenger_crossed", crossed, seconds
        )
        cross_record["pair_count"] = len(pairs)
        cross_record["pairs"] = [list(pair) for pair in pairs]
        cross_record["transform_seconds"] = float(transform_seconds)
        records.append(cross_record)
        if cross_record["validation_rmse"] <= (
            CROSS_GUARD_RATIO * uncrossed_score
        ):
            selected_crosses = True

    selected_model = crossed if selected_crosses else uncrossed
    challenger_score = float(selected_model.best_score_)
    engaged = challenger_score <= (
        OUTER_GUARD_RATIO * control_record["validation_rmse"]
    )

    final_pairs = pairs if selected_crosses else []
    final_rounds = int(selected_model.best_n_estimators_)
    if engaged:
        if final_pairs:
            transform_started = time.perf_counter_ns()
            X_final = _augment_crosses(X_train, final_pairs)
            final_transform_seconds = (
                time.perf_counter_ns() - transform_started
            ) / 1e9
        else:
            X_final = X_train
            final_transform_seconds = 0.0
        final_model, final_fit_seconds = _fit(
            _final_challenger_model(
                tree_mode=selected_tree_mode,
                linear_leaves=selected_linear,
                iterations=final_rounds,
            ),
            X_final,
            y_train,
            cat,
            ordinal_features=ordinal_features,
        )
        final_fit_seconds += final_transform_seconds

        def predict():
            test = (
                _augment_crosses(X_test, final_pairs)
                if final_pairs
                else X_test
            )
            return final_model.predict(test)

        selected_configuration = "challenger"
        decline_reason = None
    else:
        final_model, final_fit_seconds = _fit(
            _default_model(),
            X_train,
            y_train,
            cat,
        )
        final_transform_seconds = 0.0

        def predict():
            return final_model.predict(X_test)

        selected_configuration = "product_default"
        decline_reason = "outer_validation_guard"

    prediction, timing = _timed_predict(predict)
    selection_seconds = float(sum(row["fit_seconds"] for row in records))
    total_seconds = selection_seconds + float(final_fit_seconds)
    metadata = {
        "kind": COMPOSITE,
        "engaged": bool(engaged),
        "decline_reason": decline_reason,
        "size_gate": SIZE_GATE,
        "split": split,
        "outer_guard_ratio": OUTER_GUARD_RATIO,
        "cross_guard_ratio": CROSS_GUARD_RATIO,
        "selection_rounds": SELECTION_ROUNDS,
        "control_validation_rmse": control_record["validation_rmse"],
        "challenger_validation_rmse": challenger_score,
        "relative_challenger_validation_ratio": float(
            challenger_score / control_record["validation_rmse"]
        ),
        "selected_configuration": selected_configuration,
        "selected_tree_mode": selected_tree_mode,
        "selected_linear_leaves": bool(selected_linear),
        "selected_crosses": bool(selected_crosses),
        "selected_cross_pairs": [list(pair) for pair in final_pairs],
        "selected_cross_pair_count": len(final_pairs),
        "selected_best_iteration": final_rounds,
        "selected_resolved_learning_rate": float(
            selected_model.learning_rate_
        ),
        "selection_fits": records,
        "total_selection_fit_seconds": selection_seconds,
        "final_transform_seconds": float(final_transform_seconds),
        "final_fit_seconds": float(final_fit_seconds),
        "final_fit": basketball.extract_fit_metadata(final_model),
    }
    return prediction, total_seconds, timing, metadata


def _fit_chimera(X_train, y_train, cat, X_test):
    if str(CHIMERA_ROOT) not in sys.path:
        sys.path.insert(0, str(CHIMERA_ROOT))
    from chimeraboost import ChimeraBoostRegressor

    model = ChimeraBoostRegressor(
        random_state=RANDOM_STATE,
        thread_count=THREADS_PER_WORKER,
    )
    started = time.perf_counter_ns()
    model.fit(X_train, y_train, cat_features=cat or None)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    prediction, timing = _timed_predict(lambda: model.predict(X_test))
    return prediction, float(fit_seconds), timing, {
        "kind": CHIMERA,
        "fitted_tree_count": int(len(model.model_.trees_)),
        "resolved_learning_rate": float(model.model_.lr_),
        "linear_leaves_selected": bool(model.linear_leaves_selected_),
        "cross_features_selected": bool(model.cross_features_selected_),
        "cross_pair_count": int(len(model.cross_pairs_ or ())),
    }


def _catboost_frame(X, categorical_indices):
    result = X.copy()
    for index in categorical_indices:
        column = result.columns[index]
        values = result.iloc[:, index].astype(object)
        result[column] = values.map(
            lambda value: (
                "__DARKOFIT_MISSING_CATEGORY__"
                if pd.isna(value)
                else f"{type(value).__name__}:{value}"
            )
        )
    return result


def _fit_catboost(X_train, y_train, cat, X_test):
    from catboost import CatBoostRegressor

    train = _catboost_frame(X_train, cat)
    test = _catboost_frame(X_test, cat)
    model = CatBoostRegressor(
        random_seed=RANDOM_STATE,
        thread_count=THREADS_PER_WORKER,
        verbose=False,
        allow_writing_files=False,
    )
    started = time.perf_counter_ns()
    model.fit(train, y_train, cat_features=cat or None)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    prediction, timing = _timed_predict(lambda: model.predict(test))
    return prediction, float(fit_seconds), timing, {
        "kind": CATBOOST,
        "fitted_tree_count": int(model.tree_count_),
        "best_iteration": int(model.get_best_iteration()),
    }


def _warmup(config):
    started = time.perf_counter_ns()
    if config in {CONTROL, COMPOSITE}:
        from darkofit.warmup import warmup

        warmup()
        if config == COMPOSITE:
            rng = np.random.default_rng(9)
            X = rng.normal(size=(1152, 5))
            y = X[:, 0] - X[:, 1] + rng.normal(scale=0.1, size=len(X))
            for tree_mode, linear in (
                ("lightgbm", False),
                ("hybrid", False),
                ("catboost", True),
            ):
                model = _final_challenger_model(
                    tree_mode=tree_mode,
                    linear_leaves=linear,
                    iterations=2,
                )
                model.fit(X, y)
                model.predict(X[:16])
    elif config == CHIMERA:
        if str(CHIMERA_ROOT) not in sys.path:
            sys.path.insert(0, str(CHIMERA_ROOT))
        from chimeraboost.warmup import warmup

        warmup()
    elif config == CATBOOST:
        from catboost import CatBoostRegressor

        rng = np.random.default_rng(9)
        X = rng.normal(size=(320, 4))
        y = X[:, 0] + rng.normal(scale=0.1, size=len(X))
        model = CatBoostRegressor(
            iterations=2,
            random_seed=RANDOM_STATE,
            thread_count=THREADS_PER_WORKER,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(X, y)
        model.predict(X[:16])
    else:
        raise ValueError(f"unknown T5 config: {config}")
    return float((time.perf_counter_ns() - started) / 1e9)


def _evaluate_fold(task, row, X, y, cat, config, fold):
    train, test = task.get_train_test_split_indices(
        repeat=0, fold=int(fold), sample=0
    )
    split = _verify_split(row, fold, train, test)
    X_train = _take(X, train)
    y_train = _take(y, train)
    X_test = _take(X, test)
    ordinal_features = row["ordinal_features"]
    if config == CONTROL:
        fit = _fit_control
        prediction, fit_seconds, timing, metadata = fit(
            X_train, y_train, cat, X_test
        )
    elif config == COMPOSITE:
        prediction, fit_seconds, timing, metadata = _fit_composite(
            X_train, y_train, cat, X_test, ordinal_features
        )
    elif config == CHIMERA:
        prediction, fit_seconds, timing, metadata = _fit_chimera(
            X_train, y_train, cat, X_test
        )
    elif config == CATBOOST:
        prediction, fit_seconds, timing, metadata = _fit_catboost(
            X_train, y_train, cat, X_test
        )
    else:
        raise ValueError(f"unknown T5 config: {config}")
    if prediction.shape != (len(test),):
        raise RuntimeError("T5 prediction shape changed")
    target = y.iloc[test].to_numpy(dtype=np.float64)
    rmse = float(mean_squared_error(target, prediction) ** 0.5)
    if not math.isfinite(rmse) or rmse <= 0:
        raise RuntimeError("T5 RMSE is invalid")
    return {
        "fold": int(fold),
        "train_rows": split["train_size"],
        "test_rows": split["test_size"],
        "train_index_sha256": split["train_index_sha256"],
        "test_index_sha256": split["test_index_sha256"],
        "rmse": rmse,
        "fit_seconds": float(fit_seconds),
        "prediction_timing": timing,
        "prediction_sha256": _array_sha256(prediction),
        "metadata": metadata,
    }


def run_worker(task_id: int, config: str):
    _payload, rows = _registry()
    row = rows[task_id]
    task, X, y, cat = _load_task(task_id, row)
    warmup_seconds = _warmup(config)
    folds = []
    started = time.perf_counter_ns()
    for fold in FOLDS:
        folds.append(_evaluate_fold(task, row, X, y, cat, config, fold))
        gc.collect()
    wall_seconds = (time.perf_counter_ns() - started) / 1e9
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
    return {
        "task_id": task_id,
        "dataset_id": int(row["dataset_id"]),
        "dataset_name": row["dataset_name"],
        "lineage_cluster": row["lineage_cluster"],
        "stratum": row["stratum"],
        "categorical_feature_indices": cat,
        "ordinal_features": row["ordinal_features"],
        "config": config,
        "folds": folds,
        "fold_count": len(folds),
        "warmup_seconds": warmup_seconds,
        "wall_seconds": float(wall_seconds),
        "summed_fit_seconds": float(
            sum(fold["fit_seconds"] for fold in folds)
        ),
        "summed_prediction_block_seconds": float(
            sum(fold["prediction_timing"]["total_seconds"] for fold in folds)
        ),
        "peak_rss_bytes": _peak_rss_bytes(),
        "behavior_fingerprint_sha256": _json_sha256(behavior),
    }


def _worker_environment():
    environment = basketball.worker_environment(THREADS_PER_WORKER)
    environment.update(
        {
            "DARKOFIT_WARMUP": "0",
            "CHIMERABOOST_WARMUP": "0",
            "OPENML_CACHE_DIRECTORY": str(Path.home() / ".cache" / "openml"),
            "PYTHONPATH": os.pathsep.join(
                [
                    str(ROOT),
                    str(CHIMERA_ROOT),
                    environment.get("PYTHONPATH", ""),
                ]
            ),
        }
    )
    return environment


def _worker_command(task_id, config):
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-task",
        str(task_id),
        "--worker-config",
        config,
    ]


def _spool_binding(darko_source, chimera_source):
    return {
        "schema_version": 1,
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "protocol_sha256": _sha256(PROTOCOL),
        "registry_file_sha256": EXPECTED_REGISTRY_FILE_SHA256,
        "registry_canonical_sha256": EXPECTED_REGISTRY_CANONICAL_SHA256,
        "darkofit_head": darko_source["head"],
        "chimeraboost_head": chimera_source["head"],
        "configs": list(CONFIGS),
        "folds": list(FOLDS),
    }


def _spool_path(spool_directory, task_id, config):
    return spool_directory / f"task-{int(task_id)}--{config}.json"


def _load_spool(path, binding, task_id, config):
    if path.is_symlink():
        raise RuntimeError(f"refusing symlink T5 spool record: {path}")
    payload = json.loads(path.read_text())
    expected_hash = payload.get("spool_record_sha256")
    unhashed = dict(payload)
    unhashed.pop("spool_record_sha256", None)
    if expected_hash != _json_sha256(unhashed):
        raise RuntimeError(f"T5 spool record hash is invalid: {path}")
    if payload.get("binding") != binding:
        raise RuntimeError(f"T5 spool binding changed: {path}")
    if (
        int(payload.get("task_id", -1)) != int(task_id)
        or payload.get("config") != config
    ):
        raise RuntimeError(f"T5 spool identity changed: {path}")
    result = payload.get("result")
    if (
        not isinstance(result, dict)
        or int(result.get("task_id", -1)) != int(task_id)
        or result.get("config") != config
        or payload.get("result_sha256") != _json_sha256(result)
    ):
        raise RuntimeError(f"T5 spool result changed: {path}")
    return result, expected_hash


def _create_spool(path, binding, task_id, config, result):
    if path.is_symlink():
        raise RuntimeError(f"refusing symlink T5 spool record: {path}")
    payload = {
        "schema_version": 1,
        "name": "darkofit_t5_composite_worker_spool_v1",
        "binding": binding,
        "task_id": int(task_id),
        "config": config,
        "result_sha256": _json_sha256(result),
        "result": result,
    }
    payload["spool_record_sha256"] = _json_sha256(payload)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        return _load_spool(path, binding, task_id, config)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return result, payload["spool_record_sha256"]


def _run_one(task_id, config, spool_directory, binding):
    path = _spool_path(spool_directory, task_id, config)
    if path.exists() or path.is_symlink():
        result, spool_hash = _load_spool(
            path, binding, task_id, config
        )
        return result, spool_hash, True
    completed = subprocess.run(
        _worker_command(task_id, config),
        cwd=ROOT,
        env=_worker_environment(),
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"T5 worker {config}/{task_id} failed with "
            f"{completed.returncode}\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if len(lines) != 1:
        raise RuntimeError(
            f"T5 worker {config}/{task_id} emitted {len(lines)} results\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    result = json.loads(lines[0][len(WORKER_PREFIX) :])
    if result.get("task_id") != task_id or result.get("config") != config:
        raise RuntimeError(f"T5 worker {config}/{task_id} identity changed")
    result["worker_stdout"] = (
        "\n".join(
            line
            for line in completed.stdout.splitlines()
            if not line.startswith(WORKER_PREFIX)
        ).strip()
        or None
    )
    result["worker_stderr"] = completed.stderr.strip() or None
    result, spool_hash = _create_spool(
        path, binding, task_id, config, result
    )
    return result, spool_hash, False


def _run_wave(task_ids, config, spool_directory, binding):
    results = []
    spool_records = []
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        futures = {
            executor.submit(
                _run_one,
                task_id,
                config,
                spool_directory,
                binding,
            ): task_id
            for task_id in task_ids
        }
        completed = 0
        for future in as_completed(futures):
            task_id = futures[future]
            result, spool_hash, resumed = future.result()
            results.append(result)
            spool_records.append(
                {
                    "task_id": int(task_id),
                    "config": config,
                    "filename": _spool_path(
                        spool_directory, task_id, config
                    ).name,
                    "sha256": spool_hash,
                    "resumed": bool(resumed),
                }
            )
            completed += 1
            print(
                f"{config}: {completed}/{len(task_ids)} "
                f"(task {task_id}, "
                f"{'resumed' if resumed else 'fresh'})",
                flush=True,
            )
    return (
        sorted(results, key=lambda row: int(row["task_id"])),
        sorted(spool_records, key=lambda row: int(row["task_id"])),
    )


def _atomic_create(path: Path, payload: dict[str, Any]):
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing existing output: {path}")
    creator._atomic_write_bytes(
        path,
        (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode(),
    )


def run_parent(args):
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    if args.spool_directory.is_symlink():
        raise RuntimeError(
            f"refusing symlink T5 spool directory: {args.spool_directory}"
        )
    registry, rows = _registry()
    darko_source, chimera_source = _source_state()
    binding = _spool_binding(darko_source, chimera_source)
    task_ids = [int(row["task_id"]) for row in registry["tasks"]]
    results = []
    spool_records = []
    for index, config in enumerate(CONFIGS, start=1):
        if _source_state() != (darko_source, chimera_source):
            raise RuntimeError("T5 source changed during execution")
        print(f"wave {index}/{len(CONFIGS)}: {config}", flush=True)
        wave_results, wave_spool = _run_wave(
            task_ids,
            config,
            args.spool_directory,
            binding,
        )
        results.extend(wave_results)
        spool_records.extend(wave_spool)
    if _source_state() != (darko_source, chimera_source):
        raise RuntimeError("T5 source changed during execution")
    expected = {(task_id, config) for task_id in task_ids for config in CONFIGS}
    observed = {(row["task_id"], row["config"]) for row in results}
    if observed != expected or len(results) != len(expected):
        raise RuntimeError("T5 raw execution is incomplete")
    artifact = {
        "schema_version": 1,
        "name": "darkofit_t5_composite_confirmation_raw_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": _sha256(PROTOCOL),
            "runner_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "registry_file_sha256": EXPECTED_REGISTRY_FILE_SHA256,
            "registry_canonical_sha256": EXPECTED_REGISTRY_CANONICAL_SHA256,
            "configs": list(CONFIGS),
            "folds": list(FOLDS),
            "task_count": len(task_ids),
            "coordinate_count": len(task_ids) * len(FOLDS),
            "worker_count": len(results),
            "threads_per_worker": THREADS_PER_WORKER,
            "concurrent_workers": CONCURRENT_WORKERS,
            "size_gate": SIZE_GATE,
            "validation_fraction": VALIDATION_FRACTION,
            "outer_guard_ratio": OUTER_GUARD_RATIO,
            "cross_guard_ratio": CROSS_GUARD_RATIO,
            "selection_rounds": SELECTION_ROUNDS,
            "prediction_block_seconds": PREDICTION_BLOCK_SECONDS,
            "lockbox_data_used": False,
            "task_drop_allowed": False,
            "task_imputation_allowed": False,
        },
        "sources": {
            "darkofit": darko_source,
            "chimeraboost": chimera_source,
        },
        "environment": {
            "python": sys.version,
            "machine": creator._machine_details(),
            "dependencies": creator._dependency_versions(),
        },
        "registry_power_probability": registry["power_analysis"][
            "pass_probability"
        ],
        "spool": {
            "binding": binding,
            "record_count": len(spool_records),
            "resumed_record_count": sum(
                bool(row["resumed"]) for row in spool_records
            ),
            "records": spool_records,
        },
        "results": results,
        "outcomes_scored": True,
        "analysis_performed": False,
        "default_promotion_authorized": False,
        "lockbox_data_used": False,
    }
    artifact["raw_artifact_sha256"] = _json_sha256(artifact)
    _atomic_create(args.output, artifact)
    print(f"wrote {args.output}", flush=True)
    return artifact


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--spool-directory",
        type=Path,
        default=DEFAULT_SPOOL_DIRECTORY,
    )
    parser.add_argument("--worker-task", type=int)
    parser.add_argument("--worker-config", choices=CONFIGS)
    args = parser.parse_args(argv)
    args.output = Path(os.path.abspath(os.path.expanduser(args.output)))
    args.spool_directory = Path(
        os.path.abspath(os.path.expanduser(args.spool_directory))
    )
    if bool(args.worker_task is not None) != bool(args.worker_config):
        parser.error("--worker-task and --worker-config must be paired")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.worker_config:
        _payload, rows = _registry()
        if args.worker_task not in rows:
            raise RuntimeError("T5 worker task is outside the registry")
        result = run_worker(args.worker_task, args.worker_config)
        print(
            WORKER_PREFIX
            + json.dumps(result, sort_keys=True, allow_nan=False),
            flush=True,
        )
        return 0
    run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
