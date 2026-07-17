#!/usr/bin/env python3
"""Run the frozen fresh smooth/process selector confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
CHIMERA_ROOT = ROOT.parent / "chimeraboost"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import basketball_harness as basketball  # noqa: E402
from benchmarks import build_ctr23_contamination_registry as registry  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402
from benchmarks import run_smooth_linear_leaves_development as smooth  # noqa: E402


CONTROL = "darko_default"
SELECTOR = "smooth_margin_selector"
FIXED = "darko_linear_fixed"
CHIMERA = "chimera_product"
CATBOOST = "catboost_product"
CONFIGS = (CONTROL, SELECTOR, FIXED, CHIMERA, CATBOOST)
REGISTRY_V1 = ROOT / "benchmarks" / "fresh_confirmation_registry.json"
REGISTRY_V2 = ROOT / "benchmarks" / "fresh_confirmation_registry_v2.json"
PROTOCOL = ROOT / "benchmarks" / "fresh_selector_confirmation_protocol.md"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "fresh_selector_confirmation.json"
EXPECTED_REGISTRY_V1_SHA256 = (
    "37799ed0b788af3c1d69c8f0f7cf37a656fde998ce2d54b5c4a2196c369df4c3"
)
EXPECTED_REGISTRY_V2_SHA256 = (
    "0d878d690e32f6781a170fa3e5c232eef13d20d51d25b352c96a20ddc87e3970"
)
EXPECTED_CHIMERA_HEAD = "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d"
FOLDS = (0, 1, 2)
THREADS_PER_WORKER = 6
CONCURRENT_TASK_WORKERS = 3
VALIDATION_FRACTION = 0.2
MIN_RELATIVE_IMPROVEMENT = 0.03
PRIMARY_MAX_RATIO = 0.98
PRIMARY_MIN_WINS = 9
MAX_LINEAGE_RATIO = 1.02
WORKER_RESULT_PREFIX = "FRESH_SELECTOR_CONFIRMATION_RESULT="


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value, dtype="<f8") -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("worker peak RSS is unavailable")
    return value


def _optional_int(value):
    return None if value is None else int(value)


def _behavior_metadata(value):
    if isinstance(value, dict):
        return {
            key: (
                None
                if key in {"fit_seconds", "final_fit_seconds", "phase_seconds"}
                else _behavior_metadata(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_behavior_metadata(item) for item in value]
    return value


def _registries():
    if _sha256(REGISTRY_V1) != EXPECTED_REGISTRY_V1_SHA256:
        raise RuntimeError("fresh registry v1 identity changed")
    if _sha256(REGISTRY_V2) != EXPECTED_REGISTRY_V2_SHA256:
        raise RuntimeError("fresh registry v2 identity changed")
    v1 = json.loads(REGISTRY_V1.read_text())
    v2 = json.loads(REGISTRY_V2.read_text())
    if v1["confirmation_data_scored"] or v2["confirmation_data_scored"]:
        raise RuntimeError("fresh registry already marked scored")
    by_task = {int(row["task_id"]): row for row in v1["tasks"]}
    amendments = {int(row["task_id"]): row for row in v2["tasks"]}
    if set(by_task) != set(amendments):
        raise RuntimeError("fresh registry v1/v2 task membership differs")
    return v1, v2, by_task, amendments


def _load_task(task_id: int, registry_row: dict[str, Any]):
    import openml

    task = openml.tasks.get_task(task_id, download_splits=True)
    dataset = task.get_dataset()
    X, y, categorical, _names = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )
    if int(dataset.dataset_id) != int(registry_row["dataset_id"]):
        raise RuntimeError(f"fresh task {task_id} dataset identity changed")
    expected = registry_row["task_record"]
    if (
        str(dataset.name) != str(expected["dataset_name"])
        or len(X) != int(expected["fingerprint"]["n_rows"])
        or X.shape[1] != int(expected["fingerprint"]["n_features"])
        or str(dataset.md5_checksum) != str(expected["openml_declared_md5"])
    ):
        raise RuntimeError(f"fresh task {task_id} metadata changed")
    if registry.dataset_fingerprint(X, y) != expected["fingerprint"]:
        raise RuntimeError(f"fresh task {task_id} data fingerprint changed")
    y = pd.to_numeric(y, errors="raise").astype(np.float64)
    if not np.all(np.isfinite(y.to_numpy())):
        raise RuntimeError(f"fresh task {task_id} target is nonfinite")
    categorical_indices = [
        index
        for index, (declared, dtype) in enumerate(zip(categorical, X.dtypes))
        if bool(declared) or not is_numeric_dtype(dtype)
    ]
    return task, X, y, categorical_indices


def _expected_split(registry_row, fold):
    matches = [
        coordinate
        for coordinate in registry_row["task_record"]["official_splits"][
            "coordinates"
        ]
        if (
            int(coordinate["repeat"]) == 0
            and int(coordinate["fold"]) == int(fold)
            and int(coordinate["sample"]) == 0
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"fresh task {registry_row['task_id']} fold {fold} is not "
            "uniquely frozen"
        )
    return matches[0]


def _verify_split(registry_row, fold, train, test):
    expected = _expected_split(registry_row, fold)
    observed = {
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "train_index_sha256": _array_sha256(train, dtype="<i8"),
        "test_index_sha256": _array_sha256(test, dtype="<i8"),
    }
    for key, value in observed.items():
        if value != expected[key]:
            raise RuntimeError(
                f"fresh task {registry_row['task_id']} fold {fold} "
                f"{key} changed"
            )
    return observed


def selection_split(X, y):
    from darkofit.sklearn_api import _make_eval_split

    train, validation, policy = _make_eval_split(
        X,
        y,
        VALIDATION_FRACTION,
        creator.RANDOM_STATE,
        validation_strategy="weighted_stratified",
    )
    return train, validation, {
        "policy": policy,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_positions_sha256": _array_sha256(train, dtype="<i8"),
        "validation_positions_sha256": _array_sha256(
            validation, dtype="<i8"
        ),
    }


def _fit_darko(
    X,
    y,
    *,
    categorical_indices,
    linear_leaves: bool,
    threads: int,
    eval_set=None,
):
    from darkofit import DarkoRegressor

    params = {
        "random_state": creator.RANDOM_STATE,
        "thread_count": int(threads),
        "linear_leaves": bool(linear_leaves),
        "verbose_timing": True,
    }
    if eval_set is not None:
        params.update(
            {
                "early_stopping": True,
                "early_stopping_rounds": None,
                "use_best_model": True,
                "refit": False,
            }
        )
    model = DarkoRegressor(**params)
    started = time.perf_counter_ns()
    model.fit(
        X,
        y,
        cat_features=categorical_indices,
        eval_set=eval_set,
    )
    return model, float((time.perf_counter_ns() - started) / 1e9)


def _selection_record(name, model, seconds):
    metadata = basketball.extract_fit_metadata(model)
    validation = dict(model.model_.auto_params_.get("validation_split", {}))
    score = float(model.best_score_)
    if not math.isfinite(score) or score <= 0.0:
        raise RuntimeError("fresh selector validation score is invalid")
    if validation.get("source") != "explicit_eval_set":
        raise RuntimeError("fresh selector did not use explicit validation")
    if metadata["final_fit"]["stop_reason"] not in {
        "early_stopping",
        "iteration_limit",
    }:
        raise RuntimeError("fresh selector stopped unexpectedly")
    return {
        "name": name,
        "linear_leaves": name == "linear",
        "validation_rmse": score,
        "fit_seconds": float(seconds),
        "validation": validation,
        "fit_metadata": metadata,
    }


def _fit_selector(X_train, y_train, cat, X_test, threads):
    train, validation, split = selection_split(X_train, y_train)
    eval_set = (X_train.iloc[validation], y_train.iloc[validation])
    records = []
    for name, linear in (("constant", False), ("linear", True)):
        model, seconds = _fit_darko(
            X_train.iloc[train],
            y_train.iloc[train],
            categorical_indices=cat,
            linear_leaves=linear,
            threads=threads,
            eval_set=eval_set,
        )
        records.append(_selection_record(name, model, seconds))
    constant = records[0]["validation_rmse"]
    linear = records[1]["validation_rmse"]
    margin = float((constant - linear) / constant)
    selected = margin >= MIN_RELATIVE_IMPROVEMENT
    final, final_seconds = _fit_darko(
        X_train,
        y_train,
        categorical_indices=cat,
        linear_leaves=selected,
        threads=threads,
    )
    started = time.perf_counter_ns()
    prediction = np.asarray(final.predict(X_test), dtype=np.float64)
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    metadata = {
        "kind": SELECTOR,
        "split": split,
        "minimum_relative_improvement": MIN_RELATIVE_IMPROVEMENT,
        "relative_validation_improvement": margin,
        "selected_linear_leaves": bool(selected),
        "selection_fits": records,
        "final_fit_seconds": float(final_seconds),
        "final_fit": basketball.extract_fit_metadata(final),
    }
    total_fit = final_seconds + sum(row["fit_seconds"] for row in records)
    return prediction, float(total_fit), float(predict_seconds), metadata


def _fit_single_darko(
    X_train, y_train, cat, X_test, threads, *, linear_leaves
):
    model, fit_seconds = _fit_darko(
        X_train,
        y_train,
        categorical_indices=cat,
        linear_leaves=linear_leaves,
        threads=threads,
    )
    started = time.perf_counter_ns()
    prediction = np.asarray(model.predict(X_test), dtype=np.float64)
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    return prediction, fit_seconds, float(predict_seconds), {
        "kind": FIXED if linear_leaves else CONTROL,
        "selected_linear_leaves": bool(linear_leaves),
        "fit_metadata": basketball.extract_fit_metadata(model),
    }


def _fit_chimera(X_train, y_train, cat, X_test, threads):
    if str(CHIMERA_ROOT) not in sys.path:
        sys.path.insert(0, str(CHIMERA_ROOT))
    from chimeraboost import ChimeraBoostRegressor

    model = ChimeraBoostRegressor(
        random_state=creator.RANDOM_STATE,
        thread_count=int(threads),
    )
    started = time.perf_counter_ns()
    model.fit(X_train, y_train, cat_features=cat)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    started = time.perf_counter_ns()
    prediction = np.asarray(model.predict(X_test), dtype=np.float64)
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    core = model.model_
    return prediction, float(fit_seconds), float(predict_seconds), {
        "kind": CHIMERA,
        "fitted_tree_count": int(len(core.trees_)),
        "resolved_learning_rate": float(core.lr_),
        "linear_leaves_selected": bool(model.linear_leaves_selected_),
        "cross_features_selected": bool(model.cross_features_selected_),
        "cross_pair_count": int(len(model.cross_pairs_ or ())),
    }


def _catboost_frames(X_train, X_test, categorical_indices):
    train = X_train.copy()
    test = X_test.copy()
    for index in categorical_indices:
        combined = pd.concat(
            (train.iloc[:, index], test.iloc[:, index]),
            ignore_index=True,
        )
        codes, _uniques = pd.factorize(combined, sort=False)
        tokens = np.asarray(
            [
                "__DARKOFIT_MISSING_CATEGORY__"
                if code < 0
                else f"__DARKOFIT_CATEGORY_{code}__"
                for code in codes
            ],
            dtype=object,
        )
        train[train.columns[index]] = tokens[: len(train)]
        test[test.columns[index]] = tokens[len(train) :]
    return train, test


def _fit_catboost(X_train, y_train, cat, X_test, threads):
    from catboost import CatBoostRegressor

    train, test = _catboost_frames(X_train, X_test, cat)
    model = CatBoostRegressor(
        random_seed=creator.RANDOM_STATE,
        thread_count=int(threads),
        verbose=False,
        allow_writing_files=False,
    )
    started = time.perf_counter_ns()
    model.fit(train, y_train, cat_features=cat)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    started = time.perf_counter_ns()
    prediction = np.asarray(model.predict(test), dtype=np.float64)
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    return prediction, float(fit_seconds), float(predict_seconds), {
        "kind": CATBOOST,
        "fitted_tree_count": int(model.tree_count_),
        "best_iteration": _optional_int(model.get_best_iteration()),
    }


def fit_and_predict(config, X_train, y_train, cat, X_test, threads):
    if config == CONTROL:
        return _fit_single_darko(
            X_train, y_train, cat, X_test, threads, linear_leaves=False
        )
    if config == SELECTOR:
        return _fit_selector(X_train, y_train, cat, X_test, threads)
    if config == FIXED:
        return _fit_single_darko(
            X_train, y_train, cat, X_test, threads, linear_leaves=True
        )
    if config == CHIMERA:
        return _fit_chimera(X_train, y_train, cat, X_test, threads)
    if config == CATBOOST:
        return _fit_catboost(X_train, y_train, cat, X_test, threads)
    raise ValueError(f"unknown fresh confirmation config: {config}")


def _evaluate_fold(
    task, registry_row, X, y, cat, config, fold, threads
):
    train, test = task.get_train_test_split_indices(
        repeat=0, fold=int(fold), sample=0
    )
    split_identity = _verify_split(registry_row, fold, train, test)
    prediction, fit_seconds, predict_seconds, metadata = fit_and_predict(
        config,
        X.iloc[train],
        y.iloc[train],
        cat,
        X.iloc[test],
        threads,
    )
    if prediction.shape != (len(test),) or not np.all(np.isfinite(prediction)):
        raise RuntimeError("fresh confirmation produced invalid predictions")
    rmse = float(
        mean_squared_error(y.iloc[test].to_numpy(), prediction) ** 0.5
    )
    return {
        "fold": int(fold),
        "train_rows": split_identity["train_size"],
        "test_rows": split_identity["test_size"],
        "train_index_sha256": split_identity["train_index_sha256"],
        "test_index_sha256": split_identity["test_index_sha256"],
        "rmse": rmse,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "prediction_sha256": _array_sha256(prediction),
        "metadata": metadata,
    }


def run_worker(task_id: int, config: str, threads: int):
    _v1, _v2, by_task, amendments = _registries()
    row = by_task[task_id]
    task, X, y, cat = _load_task(task_id, row)
    _evaluate_fold(task, row, X, y, cat, config, FOLDS[0], threads)
    started = time.perf_counter_ns()
    folds = [
        _evaluate_fold(task, row, X, y, cat, config, fold, threads)
        for fold in FOLDS
    ]
    wall_seconds = (time.perf_counter_ns() - started) / 1e9
    behavior = {
        "task_id": task_id,
        "config": config,
        "folds": [
            {
                "fold": row["fold"],
                "rmse": row["rmse"],
                "prediction_sha256": row["prediction_sha256"],
                "metadata": _behavior_metadata(row["metadata"]),
            }
            for row in folds
        ],
    }
    return {
        "task_id": task_id,
        "dataset_id": int(row["dataset_id"]),
        "dataset_name": row["dataset_name"],
        "lineage_cluster": row["lineage_cluster"],
        "stratum": amendments[task_id]["stratum"],
        "feature_profile": amendments[task_id]["feature_profile"],
        "categorical_feature_indices": cat,
        "config": config,
        "folds": folds,
        "fold_count": len(folds),
        "geomean_rmse": float(
            np.exp(np.mean(np.log([fold["rmse"] for fold in folds])))
        ),
        "wall_seconds": float(wall_seconds),
        "summed_fit_seconds": float(
            sum(fold["fit_seconds"] for fold in folds)
        ),
        "summed_predict_seconds": float(
            sum(fold["predict_seconds"] for fold in folds)
        ),
        "peak_rss_bytes": _peak_rss_bytes(),
        "behavior_fingerprint_sha256": hashlib.sha256(
            json.dumps(
                behavior,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
    }


def _worker_environment():
    environment = basketball.worker_environment(THREADS_PER_WORKER)
    environment.update(
        {
            "CHIMERABOOST_WARMUP": "0",
            "OPENML_CACHE_DIRECTORY": str(Path.home() / ".cache" / "openml"),
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT), str(CHIMERA_ROOT), environment.get("PYTHONPATH", "")]
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
        "--worker-threads",
        str(THREADS_PER_WORKER),
    ]


def _stop_workers(processes):
    for _task_id, process in processes:
        if process.poll() is None:
            process.terminate()
    for _task_id, process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _run_batch(task_ids, config):
    processes = []
    try:
        for task_id in task_ids:
            process = subprocess.Popen(
                _worker_command(task_id, config),
                cwd=ROOT,
                env=_worker_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes.append((task_id, process))
    except BaseException:
        _stop_workers(processes)
        raise

    completed = []
    try:
        for task_id, process in processes:
            stdout, stderr = process.communicate()
            completed.append((task_id, process.returncode, stdout, stderr))
    except BaseException:
        _stop_workers(processes)
        raise

    failures = [
        (task_id, returncode, stdout, stderr)
        for task_id, returncode, stdout, stderr in completed
        if returncode
    ]
    if failures:
        task_id, returncode, stdout, stderr = failures[0]
        raise RuntimeError(
            f"fresh worker {config}/{task_id} failed with "
            f"{returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )

    results = []
    for task_id, _returncode, stdout, stderr in completed:
        lines = [
            line
            for line in stdout.splitlines()
            if line.startswith(WORKER_RESULT_PREFIX)
        ]
        if len(lines) != 1:
            raise RuntimeError(
                f"fresh worker {config}/{task_id} failed with "
                f"invalid result count {len(lines)}\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        try:
            result = json.loads(lines[0][len(WORKER_RESULT_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"fresh worker {config}/{task_id} returned invalid JSON"
            ) from exc
        if (
            int(result.get("task_id", -1)) != int(task_id)
            or result.get("config") != config
        ):
            raise RuntimeError(
                f"fresh worker {config}/{task_id} returned the wrong identity"
            )
        result["worker_stdout"] = (
            "\n".join(
                line
                for line in stdout.splitlines()
                if not line.startswith(WORKER_RESULT_PREFIX)
            ).strip()
            or None
        )
        result["worker_stderr"] = stderr.strip() or None
        results.append(result)
    return results


def _run_wave(task_ids, config):
    results = []
    for start in range(0, len(task_ids), CONCURRENT_TASK_WORKERS):
        batch = task_ids[start : start + CONCURRENT_TASK_WORKERS]
        print(
            f"{config}: tasks {start + 1}-{start + len(batch)}/"
            f"{len(task_ids)}",
            flush=True,
        )
        results.extend(_run_batch(batch, config))
    return results


def _geomean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.exp(np.mean(np.log(values))))


def _contrast(results, numerator, denominator, task_ids):
    per_lineage = {}
    split_ratios = []
    for task_id in task_ids:
        top = next(
            row
            for row in results
            if row["task_id"] == task_id and row["config"] == numerator
        )
        bottom = next(
            row
            for row in results
            if row["task_id"] == task_id and row["config"] == denominator
        )
        ratios = [
            float(a["rmse"] / b["rmse"])
            for a, b in zip(top["folds"], bottom["folds"])
        ]
        split_ratios.extend(ratios)
        ratio = _geomean(ratios)
        per_lineage[top["lineage_cluster"]] = {
            "task_id": task_id,
            "dataset_name": top["dataset_name"],
            "ratio": ratio,
            "split_ratios": ratios,
            "split_wins": int(np.count_nonzero(np.asarray(ratios) < 1.0)),
            "split_losses": int(np.count_nonzero(np.asarray(ratios) > 1.0)),
        }
    ratios = [row["ratio"] for row in per_lineage.values()]
    return {
        "numerator": numerator,
        "denominator": denominator,
        "lineage_count": len(ratios),
        "equal_lineage_geomean_ratio": _geomean(ratios),
        "equal_lineage_pct": (_geomean(ratios) - 1.0) * 100.0,
        "lineage_wins": int(np.count_nonzero(np.asarray(ratios) < 1.0)),
        "lineage_losses": int(np.count_nonzero(np.asarray(ratios) > 1.0)),
        "split_wins": int(np.count_nonzero(np.asarray(split_ratios) < 1.0)),
        "split_losses": int(np.count_nonzero(np.asarray(split_ratios) > 1.0)),
        "worst_lineage_ratio": float(max(ratios)),
        "worst_split_ratio": float(max(split_ratios)),
        "per_lineage": per_lineage,
    }


def analyze(results, task_strata):
    expected = len(task_strata) * len(CONFIGS)
    if len(results) != expected:
        raise RuntimeError(f"fresh confirmation requires {expected} workers")
    expected_coordinates = {
        (task_id, config) for task_id in task_strata for config in CONFIGS
    }
    observed_coordinates = set()
    task_identity = {}
    for row in results:
        coordinate = (int(row["task_id"]), str(row["config"]))
        if coordinate not in expected_coordinates:
            raise RuntimeError(
                f"fresh confirmation has an unexpected worker: {coordinate}"
            )
        if coordinate in observed_coordinates:
            raise RuntimeError(
                f"fresh confirmation has a duplicate worker: {coordinate}"
            )
        observed_coordinates.add(coordinate)
        if tuple(int(fold["fold"]) for fold in row["folds"]) != FOLDS:
            raise RuntimeError(
                f"fresh confirmation fold order changed at {coordinate}"
            )
        if row["stratum"] != task_strata[coordinate[0]]:
            raise RuntimeError(
                f"fresh confirmation stratum changed at {coordinate}"
            )
        identity = (str(row["lineage_cluster"]), str(row["dataset_name"]))
        previous_identity = task_identity.setdefault(coordinate[0], identity)
        if identity != previous_identity:
            raise RuntimeError(
                f"fresh confirmation task identity changed at {coordinate}"
            )
        for fold in row["folds"]:
            rmse = float(fold["rmse"])
            if not math.isfinite(rmse) or rmse <= 0.0:
                raise RuntimeError(
                    f"fresh confirmation RMSE is invalid at {coordinate}"
                )
    if observed_coordinates != expected_coordinates:
        raise RuntimeError("fresh confirmation is missing a worker")
    lineages = [identity[0] for identity in task_identity.values()]
    if len(set(lineages)) != len(lineages):
        raise RuntimeError("fresh confirmation lineage identity is not unique")
    ids = {
        stratum: [
            task_id
            for task_id, assigned in task_strata.items()
            if assigned == stratum
        ]
        for stratum in ("smooth_process", "categorical", "noisy_tabular")
    }
    contrasts = {}
    for stratum, task_ids in ids.items():
        for name, numerator, denominator in (
            ("selector_over_default", SELECTOR, CONTROL),
            ("fixed_over_default", FIXED, CONTROL),
            ("selector_over_fixed", SELECTOR, FIXED),
            ("selector_over_chimera", SELECTOR, CHIMERA),
            ("selector_over_catboost", SELECTOR, CATBOOST),
        ):
            contrasts[f"{stratum}_{name}"] = _contrast(
                results, numerator, denominator, task_ids
            )
    primary = contrasts["smooth_process_selector_over_default"]
    primary_chimera = contrasts["smooth_process_selector_over_chimera"]
    categorical = contrasts["categorical_selector_over_default"]
    noisy = contrasts["noisy_tabular_selector_over_default"]
    gates = {
        "primary_ratio_at_most_0_98": (
            primary["equal_lineage_geomean_ratio"] <= PRIMARY_MAX_RATIO
        ),
        "primary_at_least_9_lineage_wins": (
            primary["lineage_wins"] >= PRIMARY_MIN_WINS
        ),
        "primary_no_lineage_over_1_02": (
            primary["worst_lineage_ratio"] <= MAX_LINEAGE_RATIO
        ),
        "primary_beats_chimeraboost_product": (
            primary_chimera["equal_lineage_geomean_ratio"] <= 1.0
        ),
        "categorical_aggregate_nonregression": (
            categorical["equal_lineage_geomean_ratio"] <= 1.0
        ),
        "categorical_no_lineage_over_1_02": (
            categorical["worst_lineage_ratio"] <= MAX_LINEAGE_RATIO
        ),
        "noisy_aggregate_nonregression": (
            noisy["equal_lineage_geomean_ratio"] <= 1.0
        ),
        "noisy_no_lineage_over_1_02": (
            noisy["worst_lineage_ratio"] <= MAX_LINEAGE_RATIO
        ),
    }
    selector_rows = [
        fold
        for result in results
        if result["config"] == SELECTOR
        for fold in result["folds"]
    ]
    selection_count = sum(
        bool(row["metadata"]["selected_linear_leaves"])
        for row in selector_rows
    )
    passes = all(gates.values())
    return {
        "contrasts": contrasts,
        "selector_selection_count": selection_count,
        "selector_decline_count": len(selector_rows) - selection_count,
        "selector_coordinate_count": len(selector_rows),
        "gates": gates,
        "passes_all_gates": passes,
        "recommendation": (
            "advance_to_lockbox_power_freeze"
            if passes
            else "close_fresh_smooth_margin_selector"
        ),
    }


def run_parent(args):
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    v1, v2, by_task, amendments = _registries()
    source = creator.git_state(ROOT)
    chimera_source = creator.git_state(CHIMERA_ROOT)
    if not source["clean"] or not chimera_source["clean"]:
        raise RuntimeError("fresh confirmation requires clean source trees")
    if chimera_source["head"] != EXPECTED_CHIMERA_HEAD:
        raise RuntimeError("fresh confirmation ChimeraBoost head changed")
    task_ids = [int(row["task_id"]) for row in v1["tasks"]]
    task_strata = {
        task_id: amendments[task_id]["stratum"] for task_id in task_ids
    }
    results = []
    for wave, config in enumerate(CONFIGS):
        if creator.git_state(ROOT) != source:
            raise RuntimeError("DarkoFit changed during fresh confirmation")
        if creator.git_state(CHIMERA_ROOT) != chimera_source:
            raise RuntimeError("ChimeraBoost changed during confirmation")
        print(f"wave {wave + 1}/{len(CONFIGS)}: {config}", flush=True)
        results.extend(_run_wave(task_ids, config))
    if creator.git_state(ROOT) != source:
        raise RuntimeError("DarkoFit changed during fresh confirmation")
    if creator.git_state(CHIMERA_ROOT) != chimera_source:
        raise RuntimeError("ChimeraBoost changed during confirmation")
    analysis = analyze(results, task_strata)
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "fresh_selector_confirmation",
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": _sha256(PROTOCOL),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "registry_v1_file_sha256": EXPECTED_REGISTRY_V1_SHA256,
            "registry_v2_file_sha256": EXPECTED_REGISTRY_V2_SHA256,
            "configs": list(CONFIGS),
            "task_count": len(task_ids),
            "coordinate_count": len(task_ids) * len(FOLDS),
            "folds": list(FOLDS),
            "threads_per_worker": THREADS_PER_WORKER,
            "concurrent_task_workers": CONCURRENT_TASK_WORKERS,
            "lockbox_data_used": False,
            "task_imputation_allowed": False,
            "task_drop_allowed": False,
            "default_promotion_authorized": False,
        },
        "sources": {"darkofit": source, "chimeraboost": chimera_source},
        "registry_v1_canonical_sha256": v1["registry_sha256"],
        "registry_v2_canonical_sha256": v2["registry_v2_sha256"],
        "environment": {
            "machine": creator._machine_details(),
            "dependencies": creator._dependency_versions(),
            "python": sys.version,
        },
        "task_strata": {str(key): value for key, value in task_strata.items()},
        "results": results,
        "analysis": analysis,
    }
    creator._atomic_write_bytes(
        args.output,
        (
            json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode()
    )
    print(f"decision: {analysis['recommendation']}")
    print(f"wrote {args.output}")
    return artifact


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker-task", type=int)
    parser.add_argument("--worker-config", choices=CONFIGS)
    parser.add_argument(
        "--worker-threads", type=int, default=THREADS_PER_WORKER
    )
    args = parser.parse_args(argv)
    args.output = creator._absolute_lexical_path(args.output)
    if bool(args.worker_task is not None) != bool(args.worker_config):
        parser.error("--worker-task and --worker-config must be used together")
    if args.worker_threads < 1:
        parser.error("--worker-threads must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.worker_config:
        _v1, _v2, by_task, _amendments = _registries()
        if args.worker_task not in by_task:
            raise RuntimeError("worker task is outside fresh registry")
        result = run_worker(
            args.worker_task, args.worker_config, args.worker_threads
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
