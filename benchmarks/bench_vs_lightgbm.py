"""One-off ChimeraBoost vs LightGBM benchmark.

This script focuses only on ChimeraBoost and LightGBM, and reports both
accuracy and speed over multiple regression/classification datasets and size
tiers. It uses offline scikit-learn/synthetic data by default so results are
reproducible without network access.

Examples
--------
    python benchmarks/bench_vs_lightgbm.py
    python benchmarks/bench_vs_lightgbm.py --sizes tiny small --seeds 1
    python benchmarks/bench_vs_lightgbm.py --threads 8 --csv /tmp/cb_lgbm.csv

The benchmark deliberately runs every ChimeraBoost fit before importing
LightGBM. In some conda environments, importing LightGBM first and then running
numba/OpenMP kernels can abort the process when multiple threads are used.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_wine,
    make_classification,
    make_friedman1,
    make_regression,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


SIZE_SAMPLES = {
    "tiny": 750,
    "small": 2_500,
    "medium": 10_000,
    "large": 50_000,
    "xlarge": 500_000,
}

DEFAULT_SIZES = ("tiny", "small", "medium")
MAX_ITERS = 1_500
PATIENCE = 50


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task: str
    builder: Callable[[int, np.random.Generator], tuple[np.ndarray, np.ndarray, list[int] | None]]


@dataclass
class Result:
    dataset: str
    task: str
    size: str
    seed: int
    model: str
    n_train: int
    n_test: int
    n_features: int
    fit_seconds: float
    predict_seconds: float
    best_iteration: int | None
    chimera_effective_num_leaves: int | None
    lightgbm_num_leaves: int | None
    primary_metric: str
    primary_value: float
    sampling: str = ""
    top_rate: float | None = None
    other_rate: float | None = None
    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None
    accuracy: float | None = None
    f1_macro: float | None = None
    log_loss: float | None = None


def _resample_rows(X, y, n, rng, stratify=False):
    n = min(n, len(y))
    if len(y) == n:
        return X, y
    if stratify:
        _, y_codes = np.unique(y, return_inverse=True)
        per_class = np.bincount(y_codes)
        weights = np.zeros(len(y), dtype=float)
        for code, count in enumerate(per_class):
            weights[y_codes == code] = 1.0 / max(count, 1)
        weights /= weights.sum()
        idx = rng.choice(len(y), size=n, replace=False, p=weights)
    else:
        idx = rng.choice(len(y), size=n, replace=False)
    return X[idx], y[idx]


def _diabetes(n, rng):
    X, y = load_diabetes(return_X_y=True)
    return (*_resample_rows(X, y, n, rng), None)


def _breast_cancer(n, rng):
    X, y = load_breast_cancer(return_X_y=True)
    return (*_resample_rows(X, y, n, rng, stratify=True), None)


def _wine_multiclass(n, rng):
    X, y = load_wine(return_X_y=True)
    return (*_resample_rows(X, y, n, rng, stratify=True), None)


def _friedman(n, rng):
    X, y = make_friedman1(
        n_samples=n,
        n_features=20,
        noise=1.0,
        random_state=int(rng.integers(1_000_000_000)),
    )
    return X, y, None


def _wide_regression(n, rng):
    X, y = make_regression(
        n_samples=n,
        n_features=80,
        n_informative=20,
        noise=25.0,
        random_state=int(rng.integers(1_000_000_000)),
    )
    return X, y, None


def _categorical_regression(n, rng):
    store = rng.integers(0, 250, size=n)
    market = rng.integers(0, 12, size=n)
    num = rng.normal(size=(n, 8))
    store_effect = rng.normal(0.0, 3.5, size=250)[store]
    market_effect = np.linspace(-2.0, 2.0, 12)[market]
    y = (
        8.0 * np.sin(num[:, 0])
        + 3.5 * num[:, 1]
        - 2.0 * num[:, 2] * num[:, 3]
        + store_effect
        + market_effect
        + rng.normal(0.0, 2.0, size=n)
    )
    X = np.empty((n, 10), dtype=object)
    X[:, 0] = np.array([f"store_{v}" for v in store], dtype=object)
    X[:, 1] = np.array([f"market_{v}" for v in market], dtype=object)
    X[:, 2:] = num
    return X, y, [0, 1]


def _binary_classification(n, rng):
    X, y = make_classification(
        n_samples=n,
        n_features=40,
        n_informative=15,
        n_redundant=8,
        n_clusters_per_class=3,
        class_sep=1.0,
        flip_y=0.03,
        random_state=int(rng.integers(1_000_000_000)),
    )
    return X, y, None


def _multiclass_classification(n, rng):
    X, y = make_classification(
        n_samples=n,
        n_features=45,
        n_informative=22,
        n_redundant=6,
        n_classes=4,
        n_clusters_per_class=2,
        class_sep=1.1,
        flip_y=0.04,
        random_state=int(rng.integers(1_000_000_000)),
    )
    return X, y, None


def _categorical_binary(n, rng):
    region = rng.integers(0, 8, size=n)
    segment = rng.integers(0, 120, size=n)
    num = rng.normal(size=(n, 7))
    segment_effect = rng.normal(0.0, 1.5, size=120)[segment]
    region_effect = np.array([-1.6, -1.0, -0.5, -0.1, 0.3, 0.8, 1.2, 1.8])[region]
    logit = (
        segment_effect
        + region_effect
        + 0.9 * num[:, 0]
        - 0.7 * num[:, 1]
        + 0.3 * num[:, 2] * num[:, 3]
        + rng.normal(0.0, 0.8, size=n)
    )
    threshold = np.quantile(logit, 0.58)
    y = (logit > threshold).astype(int)
    X = np.empty((n, 9), dtype=object)
    X[:, 0] = np.array([f"region_{v}" for v in region], dtype=object)
    X[:, 1] = np.array([f"segment_{v}" for v in segment], dtype=object)
    X[:, 2:] = num
    return X, y, [0, 1]


def _categorical_multiclass(n, rng):
    channel = rng.integers(0, 6, size=n)
    sku = rng.integers(0, 160, size=n)
    num = rng.normal(size=(n, 7))
    sku_effect = rng.normal(0.0, 1.2, size=160)[sku]
    channel_effect = np.array([-1.2, -0.5, 0.0, 0.4, 0.9, 1.4])[channel]
    score = (
        sku_effect
        + channel_effect
        + 0.8 * num[:, 0]
        - 0.4 * num[:, 1]
        + 0.5 * np.sin(num[:, 2])
        + rng.normal(0.0, 0.7, size=n)
    )
    y = np.digitize(score, np.quantile(score, [0.30, 0.62]))
    X = np.empty((n, 9), dtype=object)
    X[:, 0] = np.array([f"channel_{v}" for v in channel], dtype=object)
    X[:, 1] = np.array([f"sku_{v}" for v in sku], dtype=object)
    X[:, 2:] = num
    return X, y, [0, 1]


DATASETS = (
    DatasetSpec("diabetes_resampled", "regression", _diabetes),
    DatasetSpec("friedman_numeric", "regression", _friedman),
    DatasetSpec("wide_numeric_reg", "regression", _wide_regression),
    DatasetSpec("categorical_reg", "regression", _categorical_regression),
    DatasetSpec("breast_cancer_resampled", "binary", _breast_cancer),
    DatasetSpec("numeric_binary", "binary", _binary_classification),
    DatasetSpec("wine_resampled", "multiclass", _wine_multiclass),
    DatasetSpec("numeric_multiclass", "multiclass", _multiclass_classification),
    DatasetSpec("categorical_binary", "binary", _categorical_binary),
    DatasetSpec("categorical_multiclass", "multiclass", _categorical_multiclass),
)


def _split_for_task(X, y, task, seed):
    stratify = y if task != "regression" else None
    return train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=stratify,
    )


def _validation_split(X_train, y_train, task, seed):
    stratify = y_train if task != "regression" else None
    return train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=10_000 + seed,
        stratify=stratify,
    )


def _encode_lightgbm(X_fit, X_val, X_test, cat_features):
    if not cat_features:
        return (
            np.asarray(X_fit, dtype=np.float64),
            np.asarray(X_val, dtype=np.float64),
            np.asarray(X_test, dtype=np.float64),
            None,
        )

    cat_features = list(cat_features)
    X_fit_lgb = np.asarray(X_fit, dtype=object).copy()
    X_val_lgb = np.asarray(X_val, dtype=object).copy()
    X_test_lgb = np.asarray(X_test, dtype=object).copy()

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )
    encoder.fit(X_fit_lgb[:, cat_features])
    for arr in (X_fit_lgb, X_val_lgb, X_test_lgb):
        arr[:, cat_features] = encoder.transform(arr[:, cat_features])
    return (
        X_fit_lgb.astype(np.float64),
        X_val_lgb.astype(np.float64),
        X_test_lgb.astype(np.float64),
        cat_features,
    )


def _classification_metrics(y_true, pred, proba):
    labels = np.unique(y_true)
    eps = 1e-15
    clipped = np.clip(proba, eps, 1.0 - eps)
    clipped /= clipped.sum(axis=1, keepdims=True)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1_macro": f1_score(y_true, pred, average="macro"),
        "log_loss": log_loss(y_true, clipped, labels=labels),
    }


def _regression_metrics(y_true, pred):
    return {
        "rmse": math.sqrt(mean_squared_error(y_true, pred)),
        "mae": mean_absolute_error(y_true, pred),
        "r2": r2_score(y_true, pred),
    }


def _run_chimera(spec, X_train, y_train, X_test, y_test, cat_features, args, seed):
    X_fit, X_val, y_fit, y_val = _validation_split(X_train, y_train, spec.task, seed)
    estimator_cls = (
        ChimeraBoostRegressor if spec.task == "regression" else ChimeraBoostClassifier
    )

    def fit_once():
        sampling = args.chimera_sampling
        if spec.task == "multiclass" and sampling == "goss":
            sampling = "uniform"
        model_kwargs = dict(
            iterations=args.iterations,
            early_stopping_rounds=args.patience,
            learning_rate=args.learning_rate,
            depth=args.depth,
            l2_leaf_reg=args.chimera_l2_leaf_reg,
            max_bins=args.chimera_max_bins,
            num_leaves=args.chimera_num_leaves,
            subsample=args.chimera_subsample,
            colsample=args.chimera_colsample,
            min_child_samples=args.chimera_min_child_samples,
            min_gain_to_split=args.chimera_min_gain_to_split,
            min_child_weight=args.chimera_min_child_weight,
            thread_count=args.threads,
            random_state=seed,
            ordered_boosting=False if args.no_ordered_boosting else "auto",
            tree_mode=args.tree_mode,
            sampling=sampling,
            top_rate=args.chimera_top_rate,
            other_rate=args.chimera_other_rate,
        )
        if spec.task != "regression":
            model_kwargs["multiclass_tree_strategy"] = (
                args.chimera_multiclass_tree_strategy
            )
        model = estimator_cls(**model_kwargs)
        start = time.perf_counter()
        model.fit(X_fit, y_fit, cat_features=cat_features, eval_set=(X_val, y_val))
        return model, time.perf_counter() - start

    model, fit_seconds = fit_once()
    for _ in range(max(0, args.repeat - 1)):
        candidate, elapsed = fit_once()
        if elapsed < fit_seconds:
            model, fit_seconds = candidate, elapsed

    def predict_once():
        start = time.perf_counter()
        if spec.task == "regression":
            pred = model.predict(X_test)
            proba = None
        else:
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)
        return pred, proba, time.perf_counter() - start

    pred, proba, predict_seconds = predict_once()
    for _ in range(max(0, args.repeat - 1)):
        candidate_pred, candidate_proba, elapsed = predict_once()
        if elapsed < predict_seconds:
            pred, proba, predict_seconds = candidate_pred, candidate_proba, elapsed

    return model, pred, proba, fit_seconds, predict_seconds


def _run_lightgbm(spec, X_train, y_train, X_test, y_test, cat_features, args, seed):
    import lightgbm as lgb

    X_fit, X_val, y_fit, y_val = _validation_split(X_train, y_train, spec.task, seed)
    X_fit, X_val, X_test, lgb_cat = _encode_lightgbm(
        X_fit,
        X_val,
        X_test,
        cat_features,
    )
    estimator_cls = lgb.LGBMRegressor if spec.task == "regression" else lgb.LGBMClassifier
    objective = None
    if spec.task == "binary":
        objective = "binary"
    elif spec.task == "multiclass":
        objective = "multiclass"

    def fit_once():
        model = estimator_cls(
            n_estimators=args.iterations,
            learning_rate=args.lightgbm_learning_rate,
            num_leaves=args.lightgbm_num_leaves,
            min_child_samples=args.lightgbm_min_child_samples,
            min_sum_hessian_in_leaf=args.lightgbm_min_sum_hessian_in_leaf,
            min_gain_to_split=args.lightgbm_min_gain_to_split,
            objective=objective,
            n_jobs=args.threads or -1,
            random_state=seed,
            verbosity=-1,
        )
        start = time.perf_counter()
        model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_val, y_val)],
            categorical_feature=lgb_cat,
            callbacks=[lgb.early_stopping(args.patience, verbose=False)],
        )
        return model, time.perf_counter() - start

    model, fit_seconds = fit_once()
    for _ in range(max(0, args.repeat - 1)):
        candidate, elapsed = fit_once()
        if elapsed < fit_seconds:
            model, fit_seconds = candidate, elapsed

    def predict_once():
        start = time.perf_counter()
        if spec.task == "regression":
            pred = model.predict(X_test)
            proba = None
        else:
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)
        return pred, proba, time.perf_counter() - start

    pred, proba, predict_seconds = predict_once()
    for _ in range(max(0, args.repeat - 1)):
        candidate_pred, candidate_proba, elapsed = predict_once()
        if elapsed < predict_seconds:
            pred, proba, predict_seconds = candidate_pred, candidate_proba, elapsed

    return model, pred, proba, fit_seconds, predict_seconds


def _result_from_prediction(
    spec,
    size_name,
    seed,
    model_name,
    model,
    y_test,
    pred,
    proba,
    fit_seconds,
    predict_seconds,
    n_train,
    n_test,
    n_features,
    chimera_effective_num_leaves,
    lightgbm_num_leaves,
):
    best_iter = getattr(model, "best_iteration_", None)
    if best_iter is None:
        best_iter = getattr(model, "best_iteration", None)
    if callable(best_iter):
        best_iter = best_iter()
    if best_iter is not None:
        best_iter = int(best_iter)
    fitted_core = getattr(model, "model_", model)

    common = dict(
        dataset=spec.name,
        task=spec.task,
        size=size_name,
        seed=seed,
        model=model_name,
        sampling=getattr(fitted_core, "sampling_", ""),
        top_rate=getattr(fitted_core, "top_rate", None),
        other_rate=getattr(fitted_core, "other_rate", None),
        n_train=n_train,
        n_test=n_test,
        n_features=n_features,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        best_iteration=best_iter,
        chimera_effective_num_leaves=chimera_effective_num_leaves,
        lightgbm_num_leaves=lightgbm_num_leaves,
    )
    if spec.task == "regression":
        metrics = _regression_metrics(y_test, pred)
        return Result(
            **common,
            primary_metric="rmse",
            primary_value=metrics["rmse"],
            rmse=metrics["rmse"],
            mae=metrics["mae"],
            r2=metrics["r2"],
        )

    metrics = _classification_metrics(y_test, pred, proba)
    return Result(
        **common,
        primary_metric="f1_macro",
        primary_value=metrics["f1_macro"],
        accuracy=metrics["accuracy"],
        f1_macro=metrics["f1_macro"],
        log_loss=metrics["log_loss"],
    )


def _warm_up(args):
    # Compile (and disk-cache) every numba specialization the timed fits hit.
    # Must match the real depth and pass an eval_set: the depth-6 parallel
    # histogram/split, multiclass-softmax, categorical-TS, and early-stopping
    # kernels are distinct specializations from a shallow no-eval fit, so a
    # cheap depth-2 warmup leaves the first real multiclass fit paying compile.
    depth = getattr(args, "depth", 6)
    tree_mode = getattr(args, "tree_mode", "catboost")
    rng = np.random.default_rng(123)

    Xr, yr = make_regression(n_samples=400, n_features=12, noise=0.5, random_state=123)
    Xnb, ynb, _ = _binary_classification(400, rng)
    Xnm, ynm, _ = _multiclass_classification(400, rng)
    Xcb, ycb, cb_cat = _categorical_binary(400, rng)
    Xcm, ycm, cm_cat = _categorical_multiclass(400, rng)

    families = (
        (ChimeraBoostRegressor, Xr, yr, None, "regression"),
        (ChimeraBoostClassifier, Xnb, ynb, None, "binary"),
        (ChimeraBoostClassifier, Xnm, ynm, None, "multiclass"),
        (ChimeraBoostClassifier, Xcb, ycb, cb_cat, "binary"),
        (ChimeraBoostClassifier, Xcm, ycm, cm_cat, "multiclass"),
    )
    for estimator_cls, X, y, cat_features, task in families:
        X_fit, X_val, y_fit, y_val = _validation_split(X, y, task, 0)
        sampling = getattr(args, "chimera_sampling", "uniform")
        if task == "multiclass" and sampling == "goss":
            sampling = "uniform"
        model = estimator_cls(
            iterations=5,
            early_stopping_rounds=3,
            depth=depth,
            max_bins=getattr(args, "chimera_max_bins", 128),
            num_leaves=getattr(args, "chimera_num_leaves", None),
            subsample=getattr(args, "chimera_subsample", 1.0),
            colsample=getattr(args, "chimera_colsample", 1.0),
            min_child_samples=getattr(args, "chimera_min_child_samples", 20),
            min_gain_to_split=getattr(args, "chimera_min_gain_to_split", 0.0),
            min_child_weight=getattr(args, "chimera_min_child_weight", 1.0),
            thread_count=args.threads,
            random_state=123,
            tree_mode=tree_mode,
            sampling=sampling,
            top_rate=getattr(args, "chimera_top_rate", 0.2),
            other_rate=getattr(args, "chimera_other_rate", 0.1),
        )
        model.fit(X_fit, y_fit, cat_features=cat_features, eval_set=(X_val, y_val))
        model.predict(X_val[:16])
        if task != "regression":
            model.predict_proba(X_val[:16])


def _mean(values):
    return statistics.mean(values) if values else float("nan")


def _stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _summarize_pair(chimera_results, lightgbm_results):
    primary = chimera_results[0].primary_metric
    chimera_primary = [r.primary_value for r in chimera_results]
    lightgbm_primary = [r.primary_value for r in lightgbm_results]
    chimera_fit = [r.fit_seconds for r in chimera_results]
    lightgbm_fit = [r.fit_seconds for r in lightgbm_results]
    chimera_pred = [r.predict_seconds for r in chimera_results]
    lightgbm_pred = [r.predict_seconds for r in lightgbm_results]

    if primary == "rmse":
        accuracy_delta = 100.0 * (_mean(lightgbm_primary) - _mean(chimera_primary)) / _mean(
            lightgbm_primary
        )
    else:
        accuracy_delta = 100.0 * (_mean(chimera_primary) - _mean(lightgbm_primary)) / _mean(
            lightgbm_primary
        )
    fit_speed_ratio = _mean(lightgbm_fit) / max(_mean(chimera_fit), 1e-12)
    predict_speed_ratio = _mean(lightgbm_pred) / max(_mean(chimera_pred), 1e-12)
    return {
        "primary": primary,
        "n_train": int(_mean([r.n_train for r in chimera_results])),
        "n_test": int(_mean([r.n_test for r in chimera_results])),
        "chimera_primary": _mean(chimera_primary),
        "chimera_primary_sd": _stdev(chimera_primary),
        "lightgbm_primary": _mean(lightgbm_primary),
        "lightgbm_primary_sd": _stdev(lightgbm_primary),
        "accuracy_delta": accuracy_delta,
        "chimera_fit": _mean(chimera_fit),
        "lightgbm_fit": _mean(lightgbm_fit),
        "fit_speed_ratio": fit_speed_ratio,
        "chimera_predict": _mean(chimera_pred),
        "lightgbm_predict": _mean(lightgbm_pred),
        "predict_speed_ratio": predict_speed_ratio,
        "chimera_best_iter": _mean([r.best_iteration for r in chimera_results if r.best_iteration]),
        "lightgbm_best_iter": _mean([r.best_iteration for r in lightgbm_results if r.best_iteration]),
    }


def _print_summary(results):
    by_case = {}
    for row in results:
        by_case.setdefault((row.dataset, row.task, row.size), {}).setdefault(row.model, []).append(row)

    print()
    print("CASE SUMMARY")
    print(
        "dataset                 task        size        n metric       "
        "ChimeraBoost        LightGBM            acc_delta   fit_speed  pred_speed"
    )
    print("-" * 126)
    for key in sorted(by_case):
        models = by_case[key]
        if "ChimeraBoost" not in models or "LightGBM" not in models:
            continue
        summary = _summarize_pair(models["ChimeraBoost"], models["LightGBM"])
        better_suffix = "lower" if summary["primary"] == "rmse" else "higher"
        print(
            f"{key[0]:23s} {key[1]:11s} {key[2]:9s} "
            f"{summary['n_train'] + summary['n_test']:6d} "
            f"{summary['primary']:8s} "
            f"{summary['chimera_primary']:9.4f} +/- {summary['chimera_primary_sd']:<7.4f} "
            f"{summary['lightgbm_primary']:9.4f} +/- {summary['lightgbm_primary_sd']:<7.4f} "
            f"{summary['accuracy_delta']:+8.2f}% "
            f"x{summary['fit_speed_ratio']:<8.2f} "
            f"x{summary['predict_speed_ratio']:<8.2f} "
            f"({better_suffix} is better)"
        )

    print()
    print("A positive acc_delta means ChimeraBoost is more accurate.")
    print("Speed ratios are LightGBM seconds / ChimeraBoost seconds; x>1 means ChimeraBoost is faster.")


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", choices=SIZE_SAMPLES, default=list(DEFAULT_SIZES))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="worker threads for both models; default uses each library's all-core setting",
    )
    parser.add_argument("--iterations", type=int, default=MAX_ITERS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help=(
            "ChimeraBoost max tree depth. Default is 6 for CatBoost/depthwise "
            "modes and -1 (unlimited) for LightGBM mode to match the LightGBM "
            "baseline's uncapped leaf-wise growth."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lightgbm-learning-rate", type=float, default=0.1)
    parser.add_argument("--lightgbm-num-leaves", type=int, default=64)
    parser.add_argument("--lightgbm-min-child-samples", type=int, default=20)
    parser.add_argument("--lightgbm-min-sum-hessian-in-leaf", type=float, default=1e-3)
    parser.add_argument("--lightgbm-min-gain-to-split", type=float, default=0.0)
    parser.add_argument("--chimera-l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--chimera-max-bins", type=int, default=128)
    parser.add_argument("--chimera-num-leaves", type=int, default=None)
    parser.add_argument(
        "--match-lightgbm-leaves",
        dest="match_lightgbm_leaves",
        action="store_true",
        default=True,
        help=(
            "For tree_mode=lightgbm, default an unspecified ChimeraBoost "
            "num_leaves to --lightgbm-num-leaves so benchmark comparisons use "
            "matched leaf capacity."
        ),
    )
    parser.add_argument(
        "--no-match-lightgbm-leaves",
        dest="match_lightgbm_leaves",
        action="store_false",
        help=(
            "Leave ChimeraBoost num_leaves unset when --chimera-num-leaves is "
            "omitted, preserving the estimator's native LightGBM-mode default."
        ),
    )
    parser.add_argument("--chimera-subsample", type=float, default=1.0)
    parser.add_argument("--chimera-colsample", type=float, default=1.0)
    parser.add_argument("--chimera-min-child-samples", type=int, default=20)
    parser.add_argument("--chimera-min-child-weight", type=float, default=1.0)
    parser.add_argument("--chimera-min-gain-to-split", type=float, default=0.0)
    parser.add_argument(
        "--chimera-sampling",
        choices=["uniform", "goss"],
        default="uniform",
        help=(
            "ChimeraBoost row sampling policy. Experimental 'goss' applies "
            "to scalar regression/binary rows; multiclass rows stay uniform."
        ),
    )
    parser.add_argument("--chimera-top-rate", type=float, default=0.2)
    parser.add_argument("--chimera-other-rate", type=float, default=0.1)
    parser.add_argument(
        "--chimera-multiclass-tree-strategy",
        choices=["auto", "per_class", "shared_vector"],
        default="auto",
        help=(
            "ChimeraBoost multiclass tree strategy. 'auto' preserves estimator "
            "defaults; 'shared_vector' forces one vector-valued LightGBM-mode "
            "tree per multiclass boosting round when supported."
        ),
    )
    parser.add_argument(
        "--tree-mode",
        choices=["catboost", "oblivious", "lightgbm", "depthwise", "levelwise"],
        default="catboost",
        help="ChimeraBoost tree builder: symmetric CatBoost-like or leaf-wise LightGBM-like.",
    )
    parser.add_argument("--no-ordered-boosting", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="re-time each fit/predict this many times and keep the fastest "
        "(warm timing; strips residual JIT/noise).",
    )
    parser.add_argument("--skip-lightgbm", action="store_true")
    parser.add_argument("--skip-chimera", action="store_true")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional subset by dataset name.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("benchmarks/lightgbm_oneoff_results.csv"),
        help="CSV path for per-seed raw results.",
    )
    return parser.parse_args(argv)


def _resolve_default_depth(args):
    if args.depth is None:
        args.depth = -1 if args.tree_mode == "lightgbm" else 6
    return args


def _chimera_effective_num_leaves(args):
    if args.tree_mode != "lightgbm":
        if args.depth is None or args.depth < 1:
            return None
        return 1 << int(args.depth)
    if args.chimera_num_leaves is None:
        if args.depth is None or args.depth < 0:
            return 31
        return min(31, 1 << int(args.depth))
    max_leaves = int(args.chimera_num_leaves)
    if args.depth is not None and args.depth > 0:
        max_leaves = min(max_leaves, 1 << int(args.depth))
    return max_leaves


def _resolve_benchmark_capacity(args):
    if (
        args.tree_mode == "lightgbm"
        and args.match_lightgbm_leaves
        and args.chimera_num_leaves is None
    ):
        args.chimera_num_leaves = args.lightgbm_num_leaves
    args.chimera_effective_num_leaves = _chimera_effective_num_leaves(args)
    return args


def main(argv=None):
    args = _resolve_benchmark_capacity(
        _resolve_default_depth(parse_args(argv or sys.argv[1:]))
    )
    selected = list(DATASETS)
    if args.datasets:
        requested = set(args.datasets)
        known = {spec.name for spec in DATASETS}
        unknown = requested - known
        if unknown:
            raise SystemExit(f"Unknown dataset(s): {sorted(unknown)}. Known: {sorted(known)}")
        selected = [spec for spec in selected if spec.name in requested]

    print("ChimeraBoost vs LightGBM one-off benchmark")
    print(
        f"sizes={args.sizes} seeds={args.seeds} threads={args.threads or 'all'} "
        f"iterations={args.iterations} patience={args.patience} "
        f"tree_mode={args.tree_mode} depth={args.depth}"
    )
    print(f"writing raw rows to {args.csv}")
    if not args.no_warmup and not args.skip_chimera:
        print("warming up ChimeraBoost numba kernels...")
        _warm_up(args)

    cases = []
    for size_name in args.sizes:
        for spec in selected:
            for seed in range(args.seeds):
                cases.append((size_name, spec, seed))

    model_runners = []
    if not args.skip_chimera:
        model_runners.append(("ChimeraBoost", _run_chimera))
    if not args.skip_lightgbm:
        model_runners.append(("LightGBM", _run_lightgbm))
    if not model_runners:
        raise SystemExit("Nothing to run: both models are skipped.")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(Result.__dataclass_fields__)
    results = []
    with args.csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        csv_file.flush()
        for model_name, runner in model_runners:
            for size_name, spec, seed in cases:
                n = SIZE_SAMPLES[size_name]
                rng = np.random.default_rng(20_260_527 + seed)
                X, y, cat_features = spec.builder(n, rng)
                X_train, X_test, y_train, y_test = _split_for_task(X, y, spec.task, seed)
                n_train = len(y_train)
                n_test = len(y_test)
                n_features = np.asarray(X).shape[1]
                try:
                    model, pred, proba, fit_seconds, predict_seconds = runner(
                        spec,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        cat_features,
                        args,
                        seed,
                    )
                except ImportError as exc:
                    raise SystemExit(
                        "LightGBM is not importable. Install it first, e.g. "
                        "`python -m pip install lightgbm`.\n"
                        f"Original import error: {exc}"
                    ) from exc
                result = _result_from_prediction(
                    spec,
                    size_name,
                    seed,
                    model_name,
                    model,
                    y_test,
                    pred,
                    proba,
                    fit_seconds,
                    predict_seconds,
                    n_train,
                    n_test,
                    n_features,
                    args.chimera_effective_num_leaves,
                    args.lightgbm_num_leaves,
                )
                results.append(result)
                writer.writerow({field: getattr(result, field) for field in fields})
                csv_file.flush()
                print(
                    f"done {model_name:12s} {spec.name:23s} "
                    f"{spec.task:11s} {size_name:6s} seed={seed}",
                    flush=True,
                )

    _print_summary(results)


if __name__ == "__main__":
    main()
