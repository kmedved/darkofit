"""Adapters for comparing divergent ChimeraBoost revisions.

This module deliberately avoids importing ``chimeraboost`` at module import time.
The revision benchmark runs each candidate in a subprocess with that revision's
path at the front of ``sys.path``; importing the package here would defeat that
isolation.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
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
from sklearn.model_selection import GroupShuffleSplit, train_test_split


SIZE_SAMPLES = {
    "tiny": 750,
    "small": 2_500,
    "medium": 10_000,
    "large": 50_000,
    "xlarge": 500_000,
}


OPENML_SUITE = {
    "credit-g": dict(data_id=31, task="binary", cats="auto"),
    "adult": dict(data_id=1590, task="binary", cats="auto"),
    "bank-marketing": dict(data_id=1461, task="binary", cats="auto"),
    "kc1": dict(data_id=1067, task="binary", cats=None),
    "phoneme": dict(data_id=1489, task="binary", cats=None),
    "electricity": dict(data_id=151, task="binary", cats=None),
    "magic": dict(data_id=1120, task="binary", cats=None),
    "spambase": dict(data_id=44, task="binary", cats=None),
    "kc2": dict(data_id=1063, task="binary", cats=None),
    "sick": dict(data_id=38, task="binary", cats="auto"),
    "mushroom": dict(data_id=24, task="binary", cats="auto"),
    "kr-vs-kp": dict(data_id=3, task="binary", cats="auto"),
    "vehicle": dict(data_id=54, task="multiclass", cats=None),
    "segment": dict(data_id=40984, task="multiclass", cats=None),
    "optdigits": dict(data_id=28, task="multiclass", cats=None),
    "car": dict(data_id=40975, task="multiclass", cats="auto"),
    "splice": dict(data_id=46, task="multiclass", cats="auto"),
    "satimage": dict(data_id=182, task="multiclass", cats=None),
    "pendigits": dict(data_id=32, task="multiclass", cats=None),
    "letter": dict(data_id=6, task="multiclass", cats=None),
    "cpu_act": dict(data_id=197, task="regression", cats=None),
    "wine_quality": dict(data_id=287, task="regression", cats=None),
    "boston": dict(data_id=531, task="regression", cats=None),
    "elevators": dict(data_id=216, task="regression", cats=None),
    "ailerons": dict(data_id=296, task="regression", cats=None),
    "abalone": dict(data_id=183, task="regression", cats="auto"),
    "house_16H": dict(data_id=574, task="regression", cats=None),
}


GRINSZTAJN_HF = (
    "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main"
)
GRINSZTAJN_FOLDERS = {
    "clf_num": ("binary", False),
    "clf_cat": ("binary", True),
    "reg_num": ("regression", False),
    "reg_cat": ("regression", True),
}
GRINSZTAJN_DATASETS = {
    "clf_num": [
        "Bioresponse", "Diabetes130US", "Higgs", "MagicTelescope",
        "MiniBooNE", "bank-marketing", "california", "covertype",
        "credit", "default-of-credit-card-clients", "electricity",
        "eye_movements", "heloc", "house_16H", "jannis", "pol",
    ],
    "clf_cat": [
        "albert", "compas-two-years", "covertype",
        "default-of-credit-card-clients", "electricity", "eye_movements",
        "road-safety",
    ],
    "reg_num": [
        "Ailerons", "Bike_Sharing_Demand", "Brazilian_houses",
        "MiamiHousing2016", "abalone", "cpu_act",
        "delays_zurich_transport", "diamonds", "elevators", "house_16H",
        "house_sales", "houses", "medical_charges",
        "nyc-taxi-green-dec-2016", "pol", "sulfur", "superconduct",
        "wine_quality", "yprop_4_1",
    ],
    "reg_cat": [
        "Airlines_DepDelay_1M", "Allstate_Claims_Severity",
        "Bike_Sharing_Demand", "Brazilian_houses",
        "Mercedes_Benz_Greener_Manufacturing",
        "SGEMM_GPU_kernel_performance", "abalone", "analcatdata_supreme",
        "delays_zurich_transport", "diamonds", "house_sales",
        "medical_charges", "nyc-taxi-green-dec-2016",
        "particulate-matter-ukair-2017", "seattlecrime6", "topo_2_1",
        "visualizing_soil",
    ],
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task: str
    builder: Callable[[int, np.random.Generator], tuple]
    loss: Optional[str] = None
    alpha: Optional[float] = None


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
    learning_rate: Optional[float] = None
    n_ensembles: Optional[int] = None
    ensemble_n_jobs: int = 1
    max_bins_ts: Optional[int] = None
    weighted_target_stats: bool = False
    threads: Optional[int] = None
    ordered_boosting: bool = False
    verbose_timing: bool = False
    validation_weight_policy: str = "product"


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


def _quantile_regression(n, rng):
    X = rng.normal(size=(n, 20))
    signal = (
        3.0 * np.sin(X[:, 0])
        + 2.0 * X[:, 1]
        - 1.5 * X[:, 2] * X[:, 3]
        + 0.7 * X[:, 4] ** 2
    )
    scale = 0.4 + 1.8 / (1.0 + np.exp(-X[:, 5]))
    y = signal + rng.normal(0.0, scale, size=n)
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
        DatasetSpec(
            "quantile_reg_10", "quantile", _quantile_regression,
            loss="Quantile", alpha=0.1,
        ),
        DatasetSpec(
            "quantile_reg_50", "quantile", _quantile_regression,
            loss="Quantile", alpha=0.5,
        ),
        DatasetSpec(
            "quantile_reg_90", "quantile", _quantile_regression,
            loss="Quantile", alpha=0.9,
        ),
        DatasetSpec("categorical_reg", "regression", _categorical_regression),
        DatasetSpec("breast_cancer_resampled", "binary", _breast_cancer),
        DatasetSpec("numeric_binary", "binary", _binary_classification),
        DatasetSpec("wine_resampled", "multiclass", _wine_multiclass),
        DatasetSpec("numeric_multiclass", "multiclass", _multiclass_classification),
        DatasetSpec("categorical_binary", "binary", _categorical_binary),
        DatasetSpec("categorical_multiclass", "multiclass", _categorical_multiclass),
    )
}


def _frame_to_dataset(X_df, y, cats, task):
    """Convert a pandas-like tabular frame to benchmark arrays.

    ``cats`` is either ``"auto"`` for object/category/string columns, an
    explicit integer-index list, or ``None``. Categorical missing values become
    a stable string token so all compared revisions see the same category.
    """
    if cats == "auto":
        def is_cat(dtype):
            text = str(dtype).lower()
            return text in ("category", "object") or text.startswith("string")
        cat_idx = [i for i, col in enumerate(X_df.columns) if is_cat(X_df[col].dtype)]
    else:
        cat_idx = cats

    if task in ("regression", "quantile"):
        y_arr = y.astype(float).to_numpy()
    else:
        y_arr = y.astype("category").cat.codes.to_numpy()

    if cat_idx:
        import pandas as pd
        cat_cols = set(cat_idx)
        cols = []
        for i, col in enumerate(X_df.columns):
            series = X_df[col]
            if i in cat_cols:
                cols.append(series.astype(object).where(series.notna(), "__nan__"))
            else:
                cols.append(pd.to_numeric(series, errors="coerce").astype(float))
        X_arr = pd.concat(cols, axis=1).to_numpy(dtype=object)
    else:
        X_arr = X_df.astype(float).to_numpy()
    return X_arr, y_arr, (cat_idx or None)


def _make_openml_builder(spec):
    def builder(n, rng):
        from sklearn.datasets import fetch_openml
        dataset = fetch_openml(data_id=spec["data_id"], as_frame=True)
        X, y, cat_features = _frame_to_dataset(
            dataset.data, dataset.target, spec["cats"], spec["task"])
        stratify = spec["task"] not in ("regression", "quantile")
        X, y = _resample_rows(X, y, n, rng, stratify=stratify)
        return X, y, cat_features
    return builder


def _make_grinsztajn_builder(folder, name, task, has_cats):
    def builder(n, rng):
        import pandas as pd
        url = f"{GRINSZTAJN_HF}/{folder}/{name}.csv"
        frame = pd.read_csv(url)
        X, y, cat_features = _frame_to_dataset(
            frame.iloc[:, :-1],
            frame.iloc[:, -1],
            "auto" if has_cats else None,
            task,
        )
        stratify = task not in ("regression", "quantile")
        X, y = _resample_rows(X, y, n, rng, stratify=stratify)
        return X, y, cat_features
    return builder


def register_openml_datasets(names=None):
    """Register opt-in OpenML datasets without fetching them yet."""
    wanted = list(OPENML_SUITE) if names is None else list(names)
    unknown = sorted(set(wanted) - set(OPENML_SUITE))
    if unknown:
        raise KeyError(f"unknown OpenML dataset(s): {unknown}")
    for name in wanted:
        spec = OPENML_SUITE[name]
        key = f"oml:{name}"
        DATASETS[key] = DatasetSpec(
            key,
            spec["task"],
            _make_openml_builder(spec),
        )


def register_grinsztajn_datasets(names=None):
    """Register opt-in Grinsztajn datasets without fetching them yet."""
    all_names = [
        f"{folder}/{name}"
        for folder, dataset_names in GRINSZTAJN_DATASETS.items()
        for name in dataset_names
    ]
    wanted = all_names if names is None else list(names)
    unknown = sorted(set(wanted) - set(all_names))
    if unknown:
        raise KeyError(f"unknown Grinsztajn dataset(s): {unknown}")
    folder_meta = dict(GRINSZTAJN_FOLDERS)
    for item in wanted:
        folder, name = item.split("/", 1)
        task, has_cats = folder_meta[folder]
        key = f"gr:{folder}/{name}"
        DATASETS[key] = DatasetSpec(
            key,
            task,
            _make_grinsztajn_builder(folder, name, task, has_cats),
        )


def register_external_datasets(dataset_names=(), *, include_openml=False,
                               include_grinsztajn=False):
    """Register requested external dataset namespaces without fetching rows."""
    names = list(dataset_names or ())
    if include_openml:
        register_openml_datasets()
    else:
        requested_openml = [name[4:] for name in names if name.startswith("oml:")]
        if requested_openml:
            register_openml_datasets(requested_openml)

    if include_grinsztajn:
        register_grinsztajn_datasets()
    else:
        requested_grinsztajn = [
            name[3:] for name in names if name.startswith("gr:")
        ]
        if requested_grinsztajn:
            register_grinsztajn_datasets(requested_grinsztajn)


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

    if task in ("regression", "quantile"):
        order = np.argsort(np.argsort(y))
        pct = order / max(len(y) - 1, 1)
        w = 0.5 + 3.0 * pct
    else:
        _, codes = np.unique(y, return_inverse=True)
        counts = np.bincount(codes)
        w = np.array([1.0 / max(counts[c], 1) for c in codes], dtype=np.float64)
    return w * (len(w) / w.sum())


def make_groups(n_samples: int, seed: int):
    """Deterministic synthetic groups for grouped benchmark split modes."""
    n_samples = int(n_samples)
    n_groups = max(10, min(500, max(1, n_samples // 20)))
    groups = np.arange(n_samples, dtype=np.int64) % n_groups
    rng = np.random.default_rng(30_311 + int(seed))
    rng.shuffle(groups)
    return groups


def _group_train_test_split(X, y, groups, sample_weight, test_size, seed):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    if sample_weight is None:
        return (
            X[train_idx],
            X[test_idx],
            y[train_idx],
            y[test_idx],
            None,
            None,
            groups[train_idx],
            groups[test_idx],
        )
    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
        sample_weight[train_idx],
        sample_weight[test_idx],
        groups[train_idx],
        groups[test_idx],
    )


def split_case(X, y, task: str, seed: int, sample_weight=None, groups=None):
    """Return deterministic train/validation/test arrays for one case."""
    sample_weight = (
        None if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    groups = None if groups is None else np.asarray(groups)
    if groups is not None and groups.shape[0] != len(y):
        raise ValueError(
            f"groups must have length {len(y)}; got {groups.shape[0]}.")
    strat = y if task not in ("regression", "quantile") else None
    if groups is not None:
        X_train, X_test, y_train, y_test, w_train, w_test, g_train, g_test = (
            _group_train_test_split(
                X, y, groups, sample_weight, 0.25, seed)
        )
    elif sample_weight is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=strat
        )
        w_train = w_test = None
        g_train = g_test = None
    else:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X,
            y,
            sample_weight,
            test_size=0.25,
            random_state=seed,
            stratify=strat,
        )
        g_train = g_test = None

    strat_fit = y_train if task not in ("regression", "quantile") else None
    if groups is not None:
        X_fit, X_val, y_fit, y_val, w_fit, w_val, g_fit, g_val = (
            _group_train_test_split(
                X_train, y_train, g_train, w_train, 0.20, 10_000 + seed)
        )
    elif w_train is None:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.20,
            random_state=10_000 + seed,
            stratify=strat_fit,
        )
        w_fit = w_val = None
        g_fit = g_val = None
    else:
        X_fit, X_val, y_fit, y_val, w_fit, w_val = train_test_split(
            X_train,
            y_train,
            w_train,
            test_size=0.20,
            random_state=10_000 + seed,
            stratify=strat_fit,
        )
        g_fit = g_val = None

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
        "groups_fit": g_fit,
        "groups_val": g_val,
        "groups_test": g_test,
        "n_train": len(y_fit),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "n_features": np.asarray(X).shape[1],
        "n_groups_train": "" if g_fit is None else len(np.unique(g_fit)),
        "n_groups_val": "" if g_val is None else len(np.unique(g_val)),
        "n_groups_test": "" if g_test is None else len(np.unique(g_test)),
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
    set_if("learning_rate", config.learning_rate)
    set_if("max_bins_ts", config.max_bins_ts)
    set_if("weighted_target_stats", config.weighted_target_stats)
    set_if("thread_count", config.threads)
    set_if("random_state", seed)
    set_if("ordered_boosting", config.ordered_boosting)
    set_if("verbose_timing", config.verbose_timing)

    if config.n_ensembles is not None and config.n_ensembles > 1:
        if "n_ensembles" not in accepted:
            raise TypeError(
                f"{estimator_cls.__name__} does not support n_ensembles="
                f"{config.n_ensembles!r}"
            )
        kwargs["n_ensembles"] = config.n_ensembles
        if "ensemble_n_jobs" in accepted:
            kwargs["ensemble_n_jobs"] = config.ensemble_n_jobs
        elif config.ensemble_n_jobs != 1:
            raise TypeError(
                f"{estimator_cls.__name__} does not support ensemble_n_jobs="
                f"{config.ensemble_n_jobs!r}"
            )

    if variant.tree_mode is not None:
        if "tree_mode" not in accepted:
            raise TypeError(
                f"{estimator_cls.__name__} does not support tree_mode="
                f"{variant.tree_mode!r}"
            )
        kwargs["tree_mode"] = variant.tree_mode
    return kwargs


def _revision_supports_tree_mode(path):
    """Cheap source-level capability check used before subprocess import."""
    root = Path(path)
    for rel in ("chimeraboost/sklearn_api.py", "chimeraboost/booster.py"):
        try:
            if "tree_mode" in (root / rel).read_text():
                return True
        except OSError:
            continue
    return False


def default_revision_specs(upstream=None, fork=None, candidate=None):
    specs: list[RevisionSpec] = []
    if upstream:
        specs.append(RevisionSpec("upstream_default", upstream, use_defaults=True))
        specs.append(RevisionSpec("upstream_matched", upstream))
    if fork:
        if _revision_supports_tree_mode(fork):
            specs.append(RevisionSpec("fork_catboost_matched", fork, tree_mode="catboost"))
            specs.append(RevisionSpec("fork_lightgbm_matched", fork, tree_mode="lightgbm"))
        else:
            specs.append(RevisionSpec("fork_matched", fork))
    if candidate:
        if _revision_supports_tree_mode(candidate):
            specs.append(RevisionSpec("candidate_catboost", candidate, tree_mode="catboost"))
            specs.append(RevisionSpec("candidate_lightgbm", candidate, tree_mode="lightgbm"))
        else:
            specs.append(RevisionSpec("candidate_matched", candidate))
    return specs
