"""Adapters for comparing divergent ChimeraBoost revisions.

This module deliberately avoids importing ``chimeraboost`` at module import time.
The revision benchmark runs each candidate in a subprocess with that revision's
path at the front of ``sys.path``; importing the package here would defeat that
isolation.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_wine,
    make_classification,
    make_friedman1,
    make_regression,
)
from sklearn.model_selection import train_test_split


SIZE_SAMPLES = {
    "tiny": 750,
    "small": 2_500,
    "medium": 10_000,
    "large": 50_000,
    "xlarge": 500_000,
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task: str
    builder: Callable[[int, np.random.Generator], tuple]


@dataclass(frozen=True)
class RevisionSpec:
    label: str
    path: str
    tree_mode: Optional[str] = None
    use_defaults: bool = False


@dataclass(frozen=True)
class FitConfig:
    iterations: int = 1_500
    patience: int = 50
    depth: int = 6
    max_bins: int = 254
    num_leaves: Optional[int] = None
    learning_rate: Optional[float] = None
    threads: Optional[int] = None
    ordered_boosting: bool = False
    verbose_timing: bool = True
    min_child_samples: int = 20
    min_gain_to_split: float = 0.0
    sampling: str = "uniform"
    top_rate: float = 0.2
    other_rate: float = 0.1


def _resample_rows(X, y, n, rng, stratify=False):
    n = min(n, len(y))
    if len(y) == n:
        return X, y
    if stratify:
        _, y_codes = np.unique(y, return_inverse=True)
        counts = np.bincount(y_codes)
        probs = np.zeros(len(y), dtype=np.float64)
        for code, count in enumerate(counts):
            probs[y_codes == code] = 1.0 / max(count, 1)
        probs /= probs.sum()
        idx = rng.choice(len(y), size=n, replace=False, p=probs)
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
    y = (logit > np.quantile(logit, 0.58)).astype(int)
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


DATASETS = {
    spec.name: spec
    for spec in (
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
}


def build_dataset(name: str, size: str, seed: int):
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}")
    if size not in SIZE_SAMPLES:
        raise KeyError(f"unknown size {size!r}")
    spec = DATASETS[name]
    rng = np.random.default_rng(20_260_605 + int(seed))
    X, y, cat_features = spec.builder(SIZE_SAMPLES[size], rng)
    return spec, X, y, cat_features


def make_sample_weight(y, task: str, mode: str):
    """Build deterministic sample weights for benchmark stress modes."""
    y = np.asarray(y)
    if mode == "none":
        return None
    if mode == "uniform":
        return np.ones(y.shape[0], dtype=np.float64)
    if mode != "stress":
        raise ValueError(f"unknown weight mode {mode!r}")

    if task == "regression":
        order = np.argsort(np.argsort(y))
        pct = order / max(len(y) - 1, 1)
        w = 0.5 + 3.0 * pct
    else:
        _, codes = np.unique(y, return_inverse=True)
        counts = np.bincount(codes)
        w = np.array([1.0 / max(counts[c], 1) for c in codes], dtype=np.float64)
    return w * (len(w) / w.sum())


def split_case(X, y, task: str, seed: int, sample_weight=None):
    """Return deterministic train/validation/test arrays for one case."""
    strat = y if task != "regression" else None
    if sample_weight is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=strat
        )
        w_train = w_test = None
    else:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X,
            y,
            sample_weight,
            test_size=0.25,
            random_state=seed,
            stratify=strat,
        )

    strat_fit = y_train if task != "regression" else None
    if w_train is None:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.20,
            random_state=10_000 + seed,
            stratify=strat_fit,
        )
        w_fit = w_val = None
    else:
        X_fit, X_val, y_fit, y_val, w_fit, w_val = train_test_split(
            X_train,
            y_train,
            w_train,
            test_size=0.20,
            random_state=10_000 + seed,
            stratify=strat_fit,
        )

    return {
        "X_fit": X_fit,
        "X_val": X_val,
        "X_test": X_test,
        "y_fit": y_fit,
        "y_val": y_val,
        "y_test": y_test,
        "w_fit": w_fit,
        "w_val": w_val,
        "w_test": w_test,
        "n_train": len(y_fit),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "n_features": np.asarray(X).shape[1],
    }


def _signature_params(cls):
    return set(inspect.signature(cls.__init__).parameters)


def estimator_kwargs(estimator_cls, config: FitConfig, variant: RevisionSpec, seed: int):
    """Map shared benchmark settings onto one revision's estimator signature."""
    accepted = _signature_params(estimator_cls)
    kwargs = {}

    def set_if(name, value):
        if name in accepted:
            kwargs[name] = value

    if variant.use_defaults:
        set_if("thread_count", config.threads)
        set_if("random_state", seed)
        return kwargs

    if "n_estimators" in accepted:
        kwargs["n_estimators"] = config.iterations
    elif "iterations" in accepted:
        kwargs["iterations"] = config.iterations
    else:
        raise TypeError("estimator accepts neither n_estimators nor iterations")

    set_if("early_stopping", True)
    set_if("early_stopping_rounds", config.patience)
    set_if("depth", config.depth)
    set_if("max_bins", config.max_bins)
    if variant.tree_mode == "lightgbm":
        set_if("num_leaves", config.num_leaves)
    set_if("learning_rate", config.learning_rate)
    set_if("thread_count", config.threads)
    set_if("random_state", seed)
    ordered_boosting = (
        False if variant.tree_mode == "lightgbm" else config.ordered_boosting
    )
    set_if("ordered_boosting", ordered_boosting)
    set_if("verbose_timing", config.verbose_timing)
    set_if("min_child_samples", config.min_child_samples)
    set_if("min_gain_to_split", config.min_gain_to_split)
    set_if("sampling", config.sampling)
    set_if("top_rate", config.top_rate)
    set_if("other_rate", config.other_rate)

    if variant.tree_mode is not None:
        if "tree_mode" not in accepted:
            raise TypeError(
                f"{estimator_cls.__name__} does not support tree_mode="
                f"{variant.tree_mode!r}"
            )
        kwargs["tree_mode"] = variant.tree_mode
    return kwargs


def default_revision_specs(upstream=None, fork=None, candidate=None):
    specs: list[RevisionSpec] = []
    if upstream:
        specs.append(RevisionSpec("upstream_default", upstream, use_defaults=True))
        specs.append(RevisionSpec("upstream_matched", upstream))
    if fork:
        specs.append(RevisionSpec("fork_catboost_matched", fork, tree_mode="catboost"))
        specs.append(RevisionSpec("fork_lightgbm_leafwise_matched", fork, tree_mode="lightgbm"))
    if candidate:
        specs.append(RevisionSpec("candidate_catboost", candidate, tree_mode="catboost"))
        specs.append(RevisionSpec("candidate_lightgbm_leafwise", candidate, tree_mode="lightgbm"))
    return specs
