"""Tune the opt-in level-wise tree mode against the shared benchmark datasets.

This script is intentionally local-candidate only. The revision harness answers
"does this integration beat upstream/fork?"; this script answers "which
levelwise-specific knobs are worth re-testing in the revision harness?".
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from benchmark_adapters import (
        DATASETS,
        SIZE_SAMPLES,
        build_dataset,
        make_groups,
        make_sample_weight,
        split_case,
    )
    from weighted_metrics import metric_bundle
except ImportError:  # pragma: no cover - supports `python -m benchmarks...`
    from benchmarks.benchmark_adapters import (
        DATASETS,
        SIZE_SAMPLES,
        build_dataset,
        make_groups,
        make_sample_weight,
        split_case,
    )
    from benchmarks.weighted_metrics import metric_bundle

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor


CSV_FIELDS = [
    "status",
    "error",
    "mode",
    "dataset",
    "task",
    "loss",
    "alpha",
    "size",
    "seed",
    "split_mode",
    "weight_mode",
    "n_train",
    "n_val",
    "n_test",
    "n_features",
    "n_groups_train",
    "n_groups_val",
    "n_groups_test",
    "n_estimators",
    "depth",
    "learning_rate",
    "l2_leaf_reg",
    "min_child_weight",
    "leaf_estimation_iterations",
    "fit_seconds",
    "predict_seconds",
    "best_iteration",
    "primary_metric",
    "primary_value",
    "rmse",
    "mae",
    "r2",
    "pinball",
    "coverage",
    "accuracy",
    "f1_macro",
    "log_loss",
    "brier",
    "weighted_rmse",
    "weighted_mae",
    "weighted_r2",
    "weighted_pinball",
    "weighted_coverage",
    "weighted_accuracy",
    "weighted_f1_macro",
    "weighted_log_loss",
    "weighted_brier",
    "timing_preprocess",
    "timing_grad_hess",
    "timing_tree_build",
    "timing_train_update",
    "timing_validation_predict",
    "timing_loss_eval",
]


def _parse_float_list(text):
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one float")
    return out


def _parse_int_list(text):
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one int")
    return out


def _complete_row(row):
    return {field: row.get(field, "") for field in CSV_FIELDS}


def _timing_fields(model):
    timing = getattr(getattr(model, "model_", None), "timing_", None)
    out = {}
    for key in (
        "preprocess",
        "grad_hess",
        "tree_build",
        "train_update",
        "validation_predict",
        "loss_eval",
    ):
        out[f"timing_{key}"] = "" if not timing else float(timing.get(key, 0.0))
    return out


def _fit_one(mode, spec, split, cat_features, params, repeat):
    task = spec.task
    estimator_cls = (
        ChimeraBoostRegressor
        if task in ("regression", "quantile")
        else ChimeraBoostClassifier
    )
    kwargs = {
        "n_estimators": params["n_estimators"],
        "depth": params["depth"],
        "learning_rate": params["learning_rate"],
        "l2_leaf_reg": params["l2_leaf_reg"],
        "min_child_weight": params["min_child_weight"],
        "leaf_estimation_iterations": params["leaf_estimation_iterations"],
        "early_stopping": True,
        "early_stopping_rounds": params["patience"],
        "thread_count": params["threads"],
        "random_state": params["seed"],
        "tree_mode": mode,
        "verbose_timing": True,
    }
    if spec.loss is not None:
        kwargs["loss"] = spec.loss
    if spec.alpha is not None:
        kwargs["alpha"] = spec.alpha
    fit_kwargs = {
        "cat_features": cat_features,
        "eval_set": (split["X_val"], split["y_val"], split["w_val"]),
    }
    if split["w_fit"] is not None:
        fit_kwargs["sample_weight"] = split["w_fit"]

    best_model = None
    fit_seconds = math.inf
    for _ in range(max(1, repeat)):
        model = estimator_cls(**kwargs)
        start = time.perf_counter()
        model.fit(split["X_fit"], split["y_fit"], **fit_kwargs)
        elapsed = time.perf_counter() - start
        if elapsed < fit_seconds:
            fit_seconds = elapsed
            best_model = model

    pred = proba = None
    predict_seconds = math.inf
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        cand_pred = best_model.predict(split["X_test"])
        cand_proba = None
        if task not in ("regression", "quantile"):
            cand_proba = best_model.predict_proba(split["X_test"])
        elapsed = time.perf_counter() - start
        if elapsed < predict_seconds:
            predict_seconds = elapsed
            pred = cand_pred
            proba = cand_proba

    metrics = metric_bundle(
        task,
        split["y_test"],
        pred,
        proba=proba,
        labels=getattr(best_model, "classes_", None),
        sample_weight=split["w_test"],
        alpha=spec.alpha,
    )
    return {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "best_iteration": getattr(best_model, "best_iteration_", ""),
        **metrics,
        **_timing_fields(best_model),
    }


def _run_case(writer, fh, mode, spec, size, seed, weight_mode, split, cat_features,
              params, repeat, split_mode):
    row = {
        "mode": mode,
        "dataset": spec.name,
        "task": spec.task,
        "loss": spec.loss or "",
        "alpha": "" if spec.alpha is None else spec.alpha,
        "size": size,
        "seed": seed,
        "split_mode": split_mode,
        "weight_mode": weight_mode,
        "n_train": split["n_train"],
        "n_val": split["n_val"],
        "n_test": split["n_test"],
        "n_features": split["n_features"],
        "n_groups_train": split.get("n_groups_train", ""),
        "n_groups_val": split.get("n_groups_val", ""),
        "n_groups_test": split.get("n_groups_test", ""),
        "n_estimators": params["n_estimators"],
        "depth": params["depth"],
        "learning_rate": "" if params["learning_rate"] is None else params["learning_rate"],
        "l2_leaf_reg": params["l2_leaf_reg"],
        "min_child_weight": params["min_child_weight"],
        "leaf_estimation_iterations": params["leaf_estimation_iterations"],
    }
    try:
        row.update(_fit_one(mode, spec, split, cat_features, params, repeat))
        row["status"] = "ok"
        row["error"] = ""
    except Exception as exc:  # pragma: no cover - benchmark diagnostics
        row["status"] = "error"
        row["error"] = str(exc)
    writer.writerow(_complete_row(row))
    fh.flush()
    print(
        f"{row['status']:<5s} {mode:<8s} {spec.name:<24s} {size:<6s} "
        f"seed={seed} weights={weight_mode} depth={params['depth']} "
        f"lr={params['learning_rate']} l2={params['l2_leaf_reg']} "
        f"mcw={params['min_child_weight']} "
        f"leaf_iters={params['leaf_estimation_iterations']}"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["numeric_binary"])
    parser.add_argument("--sizes", nargs="+", choices=SIZE_SAMPLES, default=["medium"])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--modes", nargs="+", choices=["catboost", "lightgbm"],
                        default=["lightgbm"])
    parser.add_argument("--depths", type=_parse_int_list, default=[6])
    parser.add_argument("--learning-rates", type=_parse_float_list, default=[0.05, 0.1])
    parser.add_argument("--l2-values", type=_parse_float_list, default=[1.0, 3.0, 10.0])
    parser.add_argument("--min-child-weights", type=_parse_float_list,
                        default=[1.0, 5.0, 20.0])
    parser.add_argument("--leaf-estimation-iterations", type=_parse_int_list,
                        default=[1])
    parser.add_argument("--weight-modes", nargs="+",
                        choices=["none", "uniform", "stress"],
                        default=["none"])
    parser.add_argument("--split-modes", nargs="+", choices=["row", "group"],
                        default=["row"])
    parser.add_argument("--csv", type=Path,
                        default=Path("benchmarks/levelwise_tuning.csv"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    datasets = list(DATASETS) if args.datasets == ["all"] else list(args.datasets)
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise SystemExit(f"unknown dataset(s): {unknown}; known: {sorted(DATASETS)}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        fh.flush()
        for size in args.sizes:
            for dataset in datasets:
                for seed in range(args.seeds):
                    spec, X, y, cat_features = build_dataset(dataset, size, seed)
                    for split_mode in args.split_modes:
                        groups = (
                            make_groups(len(y), seed)
                            if split_mode == "group"
                            else None
                        )
                        for weight_mode in args.weight_modes:
                            weights = make_sample_weight(y, spec.task, weight_mode)
                            split = split_case(
                                X, y, spec.task, seed, weights, groups=groups)
                            for mode in args.modes:
                                for depth in args.depths:
                                    for lr in args.learning_rates:
                                        for l2 in args.l2_values:
                                            for mcw in args.min_child_weights:
                                                for leaf_iters in args.leaf_estimation_iterations:
                                                    params = {
                                                        "n_estimators": args.n_estimators,
                                                        "patience": args.patience,
                                                        "threads": args.threads,
                                                        "seed": seed,
                                                        "depth": depth,
                                                        "learning_rate": lr,
                                                        "l2_leaf_reg": l2,
                                                        "min_child_weight": mcw,
                                                        "leaf_estimation_iterations": leaf_iters,
                                                    }
                                                    _run_case(
                                                        writer, fh, mode, spec, size,
                                                        seed, weight_mode, split,
                                                        cat_features, params,
                                                        args.repeat, split_mode,
                                                    )
    print(f"wrote rows to {args.csv}")


if __name__ == "__main__":
    main()
