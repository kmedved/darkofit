"""Benchmark opt-in ChimeraBoost feature modes against current defaults.

The harness is intentionally ChimeraBoost-only.  It isolates each measured
case/config in a child process so process peak RSS is at least comparable
across configs, and reports prediction/train-history deltas against the default
config for the same seeded case.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import f1_score, log_loss, mean_squared_error
from sklearn.model_selection import train_test_split

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
import chimeraboost.booster as booster_mod

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass(frozen=True)
class Case:
    name: str
    task: str
    n_samples: int
    n_features: int
    tree_mode: str
    multiclass_strategy: str | None = None
    categorical: bool = False


CASES = (
    Case("scalar_lightgbm_wide_200k", "regression", 200_000, 80, "lightgbm"),
    Case("scalar_hybrid_cat_100k", "regression", 100_000, 12, "hybrid", categorical=True),
    Case(
        "multiclass_shared_vector_60k",
        "multiclass",
        60_000,
        45,
        "lightgbm",
        multiclass_strategy="shared_vector",
    ),
)

FEATURE_CONFIGS = {
    "default": {},
    "hist_float32": {"histogram_dtype": "float32"},
    "leaf_uint32": {"leaf_dtype": "uint32"},
    "hist_float32_leaf_uint32": {
        "histogram_dtype": "float32",
        "leaf_dtype": "uint32",
    },
}


def _rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return peak / (1024 * 1024)
    return peak / 1024


def _pack_array(arr) -> dict:
    dense = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    return {
        "shape": dense.shape,
        "sha256": hashlib.sha256(dense.view(np.uint8)).hexdigest(),
        "data": base64.b64encode(dense.view(np.uint8)).decode("ascii"),
    }


def _unpack_array(payload: dict) -> np.ndarray:
    raw = base64.b64decode(payload["data"].encode("ascii"))
    return np.frombuffer(raw, dtype=np.float64).reshape(tuple(payload["shape"]))


def _make_case(case: Case, seed: int):
    rng = np.random.default_rng(seed)
    if case.categorical:
        store = rng.integers(0, 2_000, size=case.n_samples)
        region = rng.integers(0, 80, size=case.n_samples)
        num = rng.normal(size=(case.n_samples, case.n_features - 2))
        store_effect = rng.normal(0.0, 2.0, size=2_000)[store]
        region_effect = np.sin(region / 7.0)
        y = (
            3.0 * num[:, 0]
            - 2.0 * num[:, 1]
            + 0.5 * num[:, 2] * num[:, 3]
            + store_effect
            + region_effect
            + rng.normal(0.0, 1.0, size=case.n_samples)
        )
        X = np.empty((case.n_samples, case.n_features), dtype=object)
        X[:, 0] = np.array([f"store_{v}" for v in store], dtype=object)
        X[:, 1] = np.array([f"region_{v}" for v in region], dtype=object)
        X[:, 2:] = num
        return X, y, [0, 1]

    if case.task == "multiclass":
        X, y = make_classification(
            n_samples=case.n_samples,
            n_features=case.n_features,
            n_informative=22,
            n_redundant=6,
            n_classes=4,
            n_clusters_per_class=2,
            class_sep=1.1,
            flip_y=0.04,
            random_state=seed,
        )
        return X, y, None

    X, y = make_regression(
        n_samples=case.n_samples,
        n_features=case.n_features,
        n_informative=min(20, case.n_features // 2),
        noise=25.0,
        random_state=seed,
    )
    return X, y, None


def _split_case(case: Case, seed: int):
    X, y, cat_features = _make_case(case, seed)
    stratify = y if case.task != "regression" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed + 17, stratify=stratify
    )
    return X_train, X_test, y_train, y_test, cat_features


def _metric(case: Case, y_true, pred, proba):
    if case.task == "regression":
        return "rmse", math.sqrt(mean_squared_error(y_true, pred))
    labels = np.unique(y_true)
    clipped = np.clip(proba, 1e-15, 1.0 - 1e-15)
    clipped /= clipped.sum(axis=1, keepdims=True)
    return (
        "log_loss",
        log_loss(y_true, clipped, labels=labels),
    )


def _fit_feature_once(case: Case, config_name: str, seed: int, threads: int):
    X_train, X_test, y_train, y_test, cat_features = _split_case(case, seed)
    kwargs = dict(
        iterations=60,
        learning_rate=0.08,
        tree_mode=case.tree_mode,
        num_leaves=31,
        min_child_samples=20,
        thread_count=threads,
        random_state=seed,
        early_stopping=False,
        eval_train_loss=False,
        verbose_timing=True,
    )
    kwargs.update(FEATURE_CONFIGS[config_name])
    if case.task == "multiclass":
        kwargs["multiclass_tree_strategy"] = case.multiclass_strategy
        model = ChimeraBoostClassifier(**kwargs)
    else:
        model = ChimeraBoostRegressor(**kwargs)

    start = time.perf_counter()
    model.fit(X_train, y_train, cat_features=cat_features)
    fit_seconds = time.perf_counter() - start
    if case.task == "regression":
        pred = model.predict(X_test[:2_000])
        proba = None
        compare = pred
    else:
        pred = model.predict(X_test[:2_000])
        proba = model.predict_proba(X_test[:2_000])
        compare = proba
    metric_name, metric_value = _metric(case, y_test[:2_000], pred, proba)
    return {
        "fit_seconds": fit_seconds,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "best_iteration": int(getattr(model.model_, "best_iteration_", 0)),
        "timing": dict(getattr(model.model_, "timing_", {}) or {}),
        "prediction": _pack_array(compare),
        "history": _pack_array(getattr(model.model_, "train_history_", [])),
    }


def _run_feature_child(args):
    case = next(c for c in CASES if c.name == args.case)
    for _ in range(args.warmups):
        _fit_feature_once(case, args.config, args.seed, args.threads)
    runs = [
        _fit_feature_once(case, args.config, args.seed, args.threads)
        for _ in range(args.repeats)
    ]
    best = min(runs, key=lambda row: row["fit_seconds"])
    fit_times = [row["fit_seconds"] for row in runs]
    payload = {
        "kind": "feature",
        "case": case.name,
        "task": case.task,
        "config": args.config,
        "resolved_config": FEATURE_CONFIGS[args.config],
        "n_samples": case.n_samples,
        "n_features": case.n_features,
        "tree_mode": case.tree_mode,
        "multiclass_strategy": case.multiclass_strategy,
        "threads": args.threads,
        "iterations": 60,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "fit_seconds_best": best["fit_seconds"],
        "fit_seconds_mean": statistics.fmean(fit_times),
        "fit_seconds_all": fit_times,
        "peak_rss_mb": _rss_mb(),
        "metric_name": best["metric_name"],
        "metric_value": best["metric_value"],
        "best_iteration": best["best_iteration"],
        "timing": best["timing"],
        "prediction": best["prediction"],
        "history": best["history"],
    }
    print(json.dumps(payload, sort_keys=True))


def _run_cache_fit(cache_enabled: bool, seed: int, threads: int):
    import chimeraboost.sklearn_api as sklearn_api

    calls = 0
    original_fit_transform = booster_mod.FeaturePreprocessor.fit_transform
    original_cached = booster_mod._BaseBooster._fit_transform_preprocessor

    def counted_fit_transform(self, X, encode_targets, cat_features, sample_weight=None):
        nonlocal calls
        calls += 1
        return original_fit_transform(
            self, X, encode_targets, cat_features, sample_weight=sample_weight
        )

    def uncached(self, X, encode_targets, cat_features, sample_weight, **kwargs):
        sentinel = object()
        saved = getattr(self, "_preprocessing_cache", sentinel)
        if saved is not sentinel:
            delattr(self, "_preprocessing_cache")
        try:
            return original_cached(
                self, X, encode_targets, cat_features, sample_weight, **kwargs
            )
        finally:
            if saved is not sentinel:
                self._preprocessing_cache = saved

    booster_mod.FeaturePreprocessor.fit_transform = counted_fit_transform
    if not cache_enabled:
        booster_mod._BaseBooster._fit_transform_preprocessor = uncached
    try:
        case = Case("cache_auto_probe_cat_60k", "regression", 60_000, 12, "auto", categorical=True)
        X, y, cat_features = _make_case(case, seed)
        start = time.perf_counter()
        model = ChimeraBoostRegressor(
            iterations=18,
            learning_rate=0.08,
            tree_mode="auto",
            auto_learning_rate_probe=True,
            auto_learning_rate_probe_values=[0.04, 0.08, 0.16],
            auto_learning_rate_probe_iterations=4,
            early_stopping=True,
            validation_fraction=0.20,
            thread_count=threads,
            random_state=seed,
            eval_train_loss=False,
        )
        model.fit(X, y, cat_features=cat_features)
        elapsed = time.perf_counter() - start
        pred = model.predict(X[:2_000])
        return {
            "fit_seconds": elapsed,
            "fit_transform_calls": calls,
            "selected_tree_mode": getattr(model.model_, "tree_mode_", ""),
            "prediction": _pack_array(pred),
        }
    finally:
        booster_mod.FeaturePreprocessor.fit_transform = original_fit_transform
        booster_mod._BaseBooster._fit_transform_preprocessor = original_cached


def _run_cache_child(args):
    cache_enabled = args.config == "cache_on"
    for _ in range(args.warmups):
        _run_cache_fit(cache_enabled, args.seed, args.threads)
    runs = [
        _run_cache_fit(cache_enabled, args.seed, args.threads)
        for _ in range(args.repeats)
    ]
    best = min(runs, key=lambda row: row["fit_seconds"])
    fit_times = [row["fit_seconds"] for row in runs]
    payload = {
        "kind": "cache",
        "case": "cache_auto_probe_cat_60k",
        "config": args.config,
        "resolved_config": {
            "tree_mode": "auto",
            "auto_learning_rate_probe": True,
            "auto_learning_rate_probe_values": [0.04, 0.08, 0.16],
            "auto_learning_rate_probe_iterations": 4,
        },
        "n_samples": 60_000,
        "n_features": 12,
        "threads": args.threads,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "fit_seconds_best": best["fit_seconds"],
        "fit_seconds_mean": statistics.fmean(fit_times),
        "fit_seconds_all": fit_times,
        "fit_transform_calls": best["fit_transform_calls"],
        "selected_tree_mode": best["selected_tree_mode"],
        "peak_rss_mb": _rss_mb(),
        "prediction": best["prediction"],
    }
    print(json.dumps(payload, sort_keys=True))


def _child_main(args):
    if args.kind == "feature":
        _run_feature_child(args)
    else:
        _run_cache_child(args)


def _run_child(kind: str, case: str, config: str, args) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--kind",
        kind,
        "--case",
        case,
        "--config",
        config,
        "--seed",
        str(args.seed),
        "--threads",
        str(args.threads),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
    ]
    proc = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.stderr.strip():
        print(proc.stderr, file=sys.stderr, end="")
    return json.loads(proc.stdout.splitlines()[-1])


def _add_deltas(rows):
    by_case = {}
    for row in rows:
        by_case.setdefault((row["kind"], row["case"]), {})[row["config"]] = row

    for configs in by_case.values():
        base = configs.get("default") or configs.get("cache_off")
        if base is None:
            continue
        base_pred = _unpack_array(base["prediction"])
        base_hist = _unpack_array(base["history"]) if "history" in base else None
        base_time = float(base["fit_seconds_best"])
        base_metric = base.get("metric_value")
        for row in configs.values():
            pred = _unpack_array(row["prediction"])
            row["speedup_vs_base_best"] = base_time / float(row["fit_seconds_best"])
            row["prediction_equal_base"] = bool(np.array_equal(base_pred, pred))
            row["prediction_max_abs_diff_vs_base"] = float(
                np.max(np.abs(base_pred - pred)) if base_pred.size else 0.0
            )
            if base_metric is not None and "metric_value" in row:
                row["metric_delta_vs_base"] = float(row["metric_value"] - base_metric)
            if base_hist is not None and "history" in row:
                hist = _unpack_array(row["history"])
                same_shape = hist.shape == base_hist.shape
                row["history_equal_base"] = bool(same_shape and np.array_equal(base_hist, hist))
                if same_shape and hist.size:
                    row["history_max_abs_diff_vs_base"] = float(np.max(np.abs(base_hist - hist)))
                else:
                    row["history_max_abs_diff_vs_base"] = None


def _environment():
    import numba
    import sklearn
    import chimeraboost

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "sklearn": sklearn.__version__,
        "chimeraboost": getattr(chimeraboost, "__version__", "unknown"),
        "pid": os.getpid(),
    }


def _parent_main(args):
    rows = []
    feature_plan = []
    for case in CASES:
        configs = ["default", "leaf_uint32"]
        if case.task == "regression":
            configs = [
                "default",
                "hist_float32",
                "leaf_uint32",
                "hist_float32_leaf_uint32",
            ]
        for config in configs:
            feature_plan.append((case.name, config))

    for case, config in feature_plan:
        print(f"running feature case={case} config={config}", file=sys.stderr)
        rows.append(_run_child("feature", case, config, args))

    for config in ("cache_off", "cache_on"):
        print(f"running cache config={config}", file=sys.stderr)
        rows.append(_run_child("cache", "cache_auto_probe_cat_60k", config, args))

    _add_deltas(rows)
    payload = {
        "environment": _environment(),
        "seed": args.seed,
        "threads": args.threads,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "rows": rows,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"wrote {args.json} with {len(rows)} rows")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--json", type=str, default="")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--kind", choices=["feature", "cache"], default="feature")
    parser.add_argument("--case", default=CASES[0].name)
    parser.add_argument("--config", default="default")
    return parser


def main():
    args = _parser().parse_args()
    if args.child:
        _child_main(args)
    else:
        _parent_main(args)


if __name__ == "__main__":
    main()
