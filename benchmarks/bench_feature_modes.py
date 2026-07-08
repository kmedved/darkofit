"""Benchmark opt-in ChimeraBoost feature modes against current defaults.

The default campaign is intentionally narrow: it tests whether
``leaf_dtype="uint32"`` is safe and fast enough to become a default.  Float32
histograms remain opt-in/experimental and are only included with
``--include-hist`` or an explicit ``--configs`` list.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import platform
import random
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
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)


DEFAULT_SEEDS = (20260707, 20260708, 20260709, 20260710, 20260711)
DEFAULT_CONFIGS = ("default", "leaf_uint32")


@dataclass(frozen=True)
class Case:
    name: str
    task: str
    n_samples: int
    n_features: int
    tree_mode: str
    multiclass_strategy: str | None = None
    categorical: bool = False
    weighted: bool = False


CASES = (
    Case("scalar_catboost_numeric_120k", "regression", 120_000, 40, "catboost"),
    Case(
        "scalar_catboost_categorical_100k",
        "regression",
        100_000,
        12,
        "catboost",
        categorical=True,
    ),
    Case("scalar_lightgbm_wide_500k", "regression", 500_000, 80, "lightgbm"),
    Case("scalar_lightgbm_wide_200k", "regression", 200_000, 80, "lightgbm"),
    Case("scalar_hybrid_cat_100k", "regression", 100_000, 12, "hybrid", categorical=True),
    Case("scalar_hybrid_cat_200k", "regression", 200_000, 12, "hybrid", categorical=True),
    Case(
        "multiclass_shared_vector_60k",
        "multiclass",
        60_000,
        45,
        "lightgbm",
        multiclass_strategy="shared_vector",
    ),
    Case(
        "multiclass_shared_vector_120k",
        "multiclass",
        120_000,
        45,
        "lightgbm",
        multiclass_strategy="shared_vector",
    ),
    Case(
        "scalar_catboost_weighted_120k",
        "regression",
        120_000,
        40,
        "catboost",
        weighted=True,
    ),
)

FEATURE_CONFIGS = {
    "default": {},
    "leaf_uint32": {"leaf_dtype": "uint32"},
    "hist_float32": {"histogram_dtype": "float32"},
    "hist_float32_leaf_uint32": {
        "histogram_dtype": "float32",
        "leaf_dtype": "uint32",
    },
}


def _parse_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _parse_names(raw: str, allowed: set[str], name: str) -> tuple[str, ...]:
    if not raw:
        return ()
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise ValueError(f"unknown {name}: {', '.join(unknown)}")
    return values


def _validate_args(args):
    if args.warmups < 0:
        raise ValueError("--warmups must be nonnegative")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")


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


def _weighted_sample(case: Case, rng: np.random.Generator):
    if not case.weighted:
        return None
    weights = rng.lognormal(mean=0.0, sigma=2.0, size=case.n_samples)
    return weights.astype(np.float64, copy=False)


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
        return X, y, [0, 1], _weighted_sample(case, rng)

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
        return X, y, None, None

    X, y = make_regression(
        n_samples=case.n_samples,
        n_features=case.n_features,
        n_informative=min(20, max(2, case.n_features // 2)),
        noise=25.0,
        random_state=seed,
    )
    return X, y, None, _weighted_sample(case, rng)


def _split_case(case: Case, seed: int):
    X, y, cat_features, sample_weight = _make_case(case, seed)
    stratify = y if case.task != "regression" else None
    if sample_weight is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=seed + 17, stratify=stratify
        )
        return X_train, X_test, y_train, y_test, cat_features, None

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X,
        y,
        sample_weight,
        test_size=0.20,
        random_state=seed + 17,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test, cat_features, w_train


def _metric(case: Case, y_true, pred, proba):
    if case.task == "regression":
        return "rmse", math.sqrt(mean_squared_error(y_true, pred))
    labels = np.unique(y_true)
    clipped = np.clip(proba, 1e-15, 1.0 - 1e-15)
    clipped /= clipped.sum(axis=1, keepdims=True)
    return "log_loss", log_loss(y_true, clipped, labels=labels)


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _median_timing(runs: list[dict]) -> dict[str, float]:
    keys = sorted({key for row in runs for key in row["timing"]})
    return {
        key: _median([float(row["timing"].get(key, 0.0)) for row in runs])
        for key in keys
    }


def _fit_feature_once(
    case: Case,
    config_name: str,
    seed: int,
    threads: int,
    iterations: int,
):
    X_train, X_test, y_train, y_test, cat_features, sample_weight = _split_case(
        case, seed
    )
    kwargs = dict(
        iterations=iterations,
        learning_rate=0.08,
        tree_mode=case.tree_mode,
        min_child_samples=20,
        thread_count=threads,
        random_state=seed,
        early_stopping=False,
        eval_train_loss=False,
        verbose_timing=True,
    )
    if case.tree_mode in {"lightgbm", "hybrid"}:
        kwargs["num_leaves"] = 31
    kwargs.update(FEATURE_CONFIGS[config_name])
    if case.task == "multiclass":
        kwargs["multiclass_tree_strategy"] = case.multiclass_strategy
        model = ChimeraBoostClassifier(**kwargs)
    else:
        model = ChimeraBoostRegressor(**kwargs)

    start = time.perf_counter()
    model.fit(X_train, y_train, cat_features=cat_features, sample_weight=sample_weight)
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
    _validate_args(args)
    case = next(c for c in CASES if c.name == args.case)
    threads = int(args.threads)
    if threads < 1:
        raise ValueError("--threads must be at least 1")
    for _ in range(args.warmups):
        _fit_feature_once(case, args.config, args.seed, threads, args.iterations)
    runs = [
        _fit_feature_once(case, args.config, args.seed, threads, args.iterations)
        for _ in range(args.repeats)
    ]
    best = min(runs, key=lambda row: row["fit_seconds"])
    fit_times = [float(row["fit_seconds"]) for row in runs]
    tree_build_times = [
        float(row["timing"].get("tree_build", 0.0)) for row in runs
    ]
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
        "categorical": case.categorical,
        "weighted": case.weighted,
        "seed": int(args.seed),
        "threads": threads,
        "iterations": int(args.iterations),
        "warmups": int(args.warmups),
        "repeats": int(args.repeats),
        "fit_seconds_best": float(best["fit_seconds"]),
        "fit_seconds_median": _median(fit_times),
        "fit_seconds_mean": _mean(fit_times),
        "fit_seconds_all": fit_times,
        "tree_build_seconds_best": min(tree_build_times),
        "tree_build_seconds_median": _median(tree_build_times),
        "tree_build_seconds_mean": _mean(tree_build_times),
        "tree_build_seconds_all": tree_build_times,
        "peak_rss_mb": _rss_mb(),
        "metric_name": best["metric_name"],
        "metric_value": best["metric_value"],
        "best_iteration": best["best_iteration"],
        "timing": _median_timing(runs),
        "timing_best": best["timing"],
        "prediction": best["prediction"],
        "history": best["history"],
    }
    print(json.dumps(payload, sort_keys=True))


def _child_main(args):
    _run_feature_child(args)


def _run_child(case: str, config: str, seed: int, threads: int, args) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--case",
        case,
        "--config",
        config,
        "--seed",
        str(seed),
        "--threads",
        str(threads),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--iterations",
        str(args.iterations),
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


def _add_deltas(rows: list[dict]):
    by_group: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (row["case"], row["seed"], row["threads"])
        by_group.setdefault(key, {})[row["config"]] = row

    for configs in by_group.values():
        base = configs.get("default")
        if base is None:
            continue
        base_pred = _unpack_array(base["prediction"])
        base_hist = _unpack_array(base["history"])
        base_metric = base.get("metric_value")
        for row in configs.values():
            pred = _unpack_array(row["prediction"])
            row["speedup_vs_base_best"] = (
                float(base["fit_seconds_best"]) / float(row["fit_seconds_best"])
            )
            row["speedup_vs_base_median"] = (
                float(base["fit_seconds_median"]) / float(row["fit_seconds_median"])
            )
            row["tree_build_speedup_vs_base_best"] = (
                float(base["tree_build_seconds_best"])
                / float(row["tree_build_seconds_best"])
            )
            row["tree_build_speedup_vs_base_median"] = (
                float(base["tree_build_seconds_median"])
                / float(row["tree_build_seconds_median"])
            )
            row["prediction_equal_base"] = bool(np.array_equal(base_pred, pred))
            row["prediction_max_abs_diff_vs_base"] = float(
                np.max(np.abs(base_pred - pred)) if base_pred.size else 0.0
            )
            if base_metric is not None and "metric_value" in row:
                row["metric_delta_vs_base"] = float(row["metric_value"] - base_metric)
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


def _feature_configs_for_case(case: Case, selected_configs: tuple[str, ...]):
    configs = list(selected_configs)
    if case.task != "regression":
        configs = [
            name for name in configs
            if "histogram_dtype" not in FEATURE_CONFIGS[name]
        ]
    return tuple(configs)


def _leaf_gate_summary(rows: list[dict]) -> dict:
    leaf_rows = [
        row for row in rows
        if row["kind"] == "feature" and row["config"] == "leaf_uint32"
    ]
    per_case = {}
    for row in leaf_rows:
        key = (row["case"], row["threads"])
        per_case.setdefault(key, []).append(row)

    case_summaries = []
    for (case_name, threads), group in sorted(per_case.items()):
        wall = [float(row["speedup_vs_base_median"]) for row in group]
        tree = [float(row["tree_build_speedup_vs_base_median"]) for row in group]
        parity = all(
            row.get("prediction_equal_base")
            and row.get("history_equal_base")
            and row.get("prediction_max_abs_diff_vs_base") == 0.0
            and (row.get("history_max_abs_diff_vs_base") in (0.0, None))
            and row.get("metric_delta_vs_base", 0.0) == 0.0
            for row in group
        )
        case_summaries.append(
            {
                "case": case_name,
                "threads": threads,
                "n": len(group),
                "wall_median": _median(wall),
                "wall_min": min(wall),
                "tree_build_median": _median(tree),
                "tree_build_min": min(tree),
                "parity": parity,
            }
        )

    primary = [row for row in case_summaries if row["threads"] == 4]
    catboost_scalar = [
        row for row in primary
        if row["case"] in {
            "scalar_catboost_numeric_120k",
            "scalar_catboost_categorical_100k",
        }
    ]
    global_gate = bool(primary) and all(row["parity"] for row in primary)
    global_gate = global_gate and all(row["wall_min"] >= 0.99 for row in primary)
    global_gate = global_gate and all(row["tree_build_min"] >= 0.99 for row in primary)
    global_gate = global_gate and len(catboost_scalar) == 2
    global_gate = global_gate and all(
        row["tree_build_median"] >= 1.03 for row in catboost_scalar
    )
    if primary:
        pooled_wall = _median([row["wall_median"] for row in primary])
        global_gate = global_gate and pooled_wall >= 1.0
    else:
        pooled_wall = None

    catboost_auto_gate = (
        len(catboost_scalar) == 2
        and all(row["parity"] for row in catboost_scalar)
        and all(row["wall_min"] >= 0.99 for row in catboost_scalar)
        and all(row["tree_build_median"] >= 1.03 for row in catboost_scalar)
    )
    return {
        "case_summaries": case_summaries,
        "pooled_wall_median_threads4": pooled_wall,
        "global_leaf_uint32_gate": bool(global_gate),
        "catboost_auto_gate": bool(catboost_auto_gate),
    }


def _parent_main(args):
    _validate_args(args)
    case_names = _parse_names(args.cases, {case.name for case in CASES}, "case")
    cases = [case for case in CASES if not case_names or case.name in case_names]
    if args.configs:
        parsed_configs = _parse_names(
            args.configs, set(FEATURE_CONFIGS), "config"
        )
        if "default" not in parsed_configs and any(
            config != "default" for config in parsed_configs
        ):
            selected_configs = ("default",) + parsed_configs
        else:
            selected_configs = parsed_configs
    else:
        selected_configs = DEFAULT_CONFIGS
        if args.include_hist:
            selected_configs = selected_configs + (
                "hist_float32",
                "hist_float32_leaf_uint32",
            )
    seeds = _parse_ints(args.seeds)
    threads_list = _parse_ints(args.threads)
    if any(thread < 1 for thread in threads_list):
        raise ValueError("--threads values must be at least 1")

    rows = []
    plan = []
    for threads in threads_list:
        for seed in seeds:
            for case in cases:
                configs = list(_feature_configs_for_case(case, selected_configs))
                random.Random(f"{seed}:{threads}:{case.name}").shuffle(configs)
                for config in configs:
                    plan.append((case.name, config, seed, threads))

    for idx, (case, config, seed, threads) in enumerate(plan, start=1):
        print(
            f"[{idx}/{len(plan)}] running case={case} config={config} "
            f"seed={seed} threads={threads}",
            file=sys.stderr,
        )
        rows.append(_run_child(case, config, seed, threads, args))

    _add_deltas(rows)
    payload = {
        "environment": _environment(),
        "seeds": seeds,
        "threads": threads_list,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "iterations": args.iterations,
        "configs": selected_configs,
        "rows": rows,
        "leaf_gate": _leaf_gate_summary(rows),
    }
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"wrote {args.json} with {len(rows)} rows")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--threads", default="4")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--cases", default="")
    parser.add_argument("--configs", default="")
    parser.add_argument("--include-hist", action="store_true")
    parser.add_argument("--json", type=str, default="")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", default=CASES[0].name, help=argparse.SUPPRESS)
    parser.add_argument("--config", default="default", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEEDS[0], help=argparse.SUPPRESS)
    return parser


def main():
    args = _parser().parse_args()
    if args.child:
        _child_main(args)
    else:
        _parent_main(args)


if __name__ == "__main__":
    main()
