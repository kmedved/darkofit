"""Profile the internal kernels used by LightGBM-mode leaf-wise trees.

This diagnostic wraps selected functions in ``chimeraboost.tree`` while running
short warmed fits. It complements ``profile_chimera_phases.py``: that script
shows whether ``tree_build`` dominates, while this one breaks tree building
into histogram construction/refill, split scoring, row partitioning, leaf-value
aggregation, and prediction-update helpers.

Example:
    python benchmarks/profile_leafwise_tree_phases.py \
        --datasets numeric_binary categorical_binary friedman_numeric \
        --size medium --threads 4 --iterations 120 --repeat 2
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
import chimeraboost.tree as tree

import bench_vs_lightgbm as bench


GROUPS = {
    "hist_build": (
        "_build_histograms_counts_into",
        "_build_histograms_counts_positive_into",
        "_build_histograms_counts_into_serial",
        "_build_histograms_counts_selected_into",
        "_build_histograms_counts_selected_into_serial",
        "_build_histograms_counts_rows_into",
        "_build_histograms_counts_rows_into_serial",
        "_build_histograms_counts_selected_rows_into",
        "_build_histograms_counts_selected_rows_into_serial",
        "_build_histograms_unit_hess_into",
        "_build_histograms_unit_hess_into_serial",
        "_build_histograms_selected_unit_hess_into",
        "_build_histograms_selected_unit_hess_into_serial",
        "_build_histograms_rows_unit_hess_into",
        "_build_histograms_rows_unit_hess_into_serial",
        "_build_histograms_selected_rows_unit_hess_into",
        "_build_histograms_selected_rows_unit_hess_into_serial",
        "_build_multiclass_histograms_counts_into",
    ),
    "hist_refill": (
        "_refill_leaf_segment_histograms_counts_into",
        "_refill_leaf_segment_histograms_counts_positive_into",
        "_refill_leaf_segment_histograms_counts_into_serial",
        "_refill_leaf_segment_histograms_counts_selected_into",
        "_refill_leaf_segment_histograms_counts_selected_into_serial",
        "_refill_leaf_segment_histograms_unit_hess_into",
        "_refill_leaf_segment_histograms_unit_hess_into_serial",
        "_refill_leaf_segment_histograms_unit_hess_selected_into",
        "_refill_leaf_segment_histograms_unit_hess_selected_into_serial",
        "_refill_multiclass_leaf_segment_histograms_counts_into",
        "_refill_right_subtract_left_counts_into",
        "_refill_right_subtract_left_counts_positive_into",
        "_refill_right_subtract_left_counts_selected_into",
        "_refill_right_subtract_left_unit_hess_into",
        "_refill_right_subtract_left_unit_hess_selected_into",
    ),
    "hist_subtract": (
        "_subtract_right_child_histograms_into_left",
        "_subtract_right_child_histograms_into_left_serial",
        "_subtract_right_child_histograms_selected_into_left",
        "_subtract_right_child_histograms_selected_into_left_serial",
        "_subtract_right_child_unit_hess_histograms_into_left",
        "_subtract_right_child_unit_hess_histograms_into_left_serial",
        "_subtract_right_child_unit_hess_histograms_selected_into_left",
        "_subtract_right_child_unit_hess_histograms_selected_into_left_serial",
    ),
    "split_score": (
        "_best_splits_by_leaf_counts",
        "_best_splits_by_leaf_counts_full_features",
        "_best_splits_for_leaf_ids_counts",
        "_best_splits_for_leaf_ids_counts_full_features",
        "_best_splits_for_leaf_ids_counts_feature_parallel",
        "_best_multiclass_splits_for_leaf_ids_counts",
    ),
    "partition": (
        "_partition_leaf_rows",
        "_update_leafwise_leaves_with_split",
    ),
    "leaf_values": (
        "_leaf_values_and_sums",
        "_leaf_values_and_sums_rows",
        "_multiclass_leaf_values_and_sums",
    ),
}


@contextmanager
def _timed_tree_kernels(stats):
    originals = {}
    name_to_group = {
        name: group for group, names in GROUPS.items() for name in names
    }

    def make_wrapper(name, fn):
        group = name_to_group[name]

        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                stats[group]["seconds"] += elapsed
                stats[group]["calls"] += 1
                stats[f"{group}:{name}"]["seconds"] += elapsed
                stats[f"{group}:{name}"]["calls"] += 1

        return wrapper

    try:
        for name in name_to_group:
            if hasattr(tree, name):
                originals[name] = getattr(tree, name)
                setattr(tree, name, make_wrapper(name, originals[name]))
        yield
    finally:
        for name, fn in originals.items():
            setattr(tree, name, fn)


def _run_fit(spec, size_name, args, seed):
    n = bench.SIZE_SAMPLES[size_name]
    rng = np.random.default_rng(args.data_seed + seed)
    X, y, cat_features = spec.builder(n, rng)
    X_train, _, y_train, _ = bench._split_for_task(X, y, spec.task, seed)
    X_fit, X_val, y_fit, y_val = bench._validation_split(
        X_train, y_train, spec.task, seed
    )
    estimator_cls = (
        ChimeraBoostRegressor if spec.task == "regression" else ChimeraBoostClassifier
    )

    def fit_once():
        model = estimator_cls(
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            num_leaves=args.num_leaves,
            max_bins=args.max_bins,
            min_child_samples=args.min_child_samples,
            min_gain_to_split=args.min_gain_to_split,
            early_stopping_rounds=None,
            thread_count=args.threads,
            ordered_boosting=False,
            random_state=seed,
            verbose_timing=True,
            tree_mode="lightgbm",
        )
        stats = defaultdict(lambda: {"seconds": 0.0, "calls": 0})
        start = time.perf_counter()
        with _timed_tree_kernels(stats):
            model.fit(X_fit, y_fit, cat_features=cat_features, eval_set=(X_val, y_val))
        fit_seconds = time.perf_counter() - start
        return model, stats, fit_seconds

    best = fit_once()
    for _ in range(max(1, args.repeat) - 1):
        candidate = fit_once()
        if candidate[2] < best[2]:
            best = candidate
    model, stats, fit_seconds = best
    rounds = max(1, int(model.best_iteration_))
    timing = model.timing_ or {}
    tree_seconds = float(timing.get("tree_build", 0.0))
    row = {
        "dataset": spec.name,
        "task": spec.task,
        "size": size_name,
        "seed": seed,
        "threads": args.threads,
        "rounds": rounds,
        "fit_seconds": fit_seconds,
        "tree_seconds": tree_seconds,
    }
    for group in GROUPS:
        seconds = float(stats[group]["seconds"])
        row[f"{group}_seconds"] = seconds
        row[f"{group}_calls"] = int(stats[group]["calls"])
        row[f"{group}_share"] = seconds / tree_seconds if tree_seconds > 0 else 0.0
        row[f"{group}_ms_per_round"] = 1000.0 * seconds / rounds
    return row


def _print_rows(rows):
    print(
        "dataset                 task        rounds fit_s  tree_s "
        "hist% refill% split% part% leaf% hist_ms split_ms"
    )
    print("-" * 105)
    for row in rows:
        print(
            f"{row['dataset']:23s} {row['task']:11s} "
            f"{row['rounds']:6d} {row['fit_seconds']:5.2f} "
            f"{row['tree_seconds']:6.3f} "
            f"{100*row['hist_build_share']:5.1f} "
            f"{100*row['hist_refill_share']:7.1f} "
            f"{100*row['split_score_share']:6.1f} "
            f"{100*row['partition_share']:5.1f} "
            f"{100*row['leaf_values_share']:5.1f} "
            f"{row['hist_build_ms_per_round'] + row['hist_refill_ms_per_round']:7.3f} "
            f"{row['split_score_ms_per_round']:8.3f}"
        )


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=bench.SIZE_SAMPLES, default="medium")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["numeric_binary", "categorical_binary", "friedman_numeric"],
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--num-leaves", type=int, default=64)
    parser.add_argument("--depth", type=int, default=-1)
    parser.add_argument("--max-bins", type=int, default=128)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--min-gain-to-split", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--data-seed", type=int, default=20_260_527)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    requested = set(args.datasets)
    specs = [spec for spec in bench.DATASETS if spec.name in requested]
    missing = requested - {spec.name for spec in specs}
    if missing:
        raise SystemExit(f"unknown dataset(s): {sorted(missing)}")

    print("warming up ChimeraBoost numba kernels...")
    warm_args = argparse.Namespace(
        depth=args.depth,
        tree_mode="lightgbm",
        chimera_num_leaves=args.num_leaves,
        chimera_max_bins=args.max_bins,
        chimera_min_child_samples=args.min_child_samples,
        chimera_min_gain_to_split=args.min_gain_to_split,
        chimera_min_child_weight=1.0,
        threads=args.threads,
        chimera_sampling="uniform",
        chimera_top_rate=0.2,
        chimera_other_rate=0.1,
    )
    bench._warm_up(warm_args)

    rows = [_run_fit(spec, args.size, args, args.seed) for spec in specs]
    _print_rows(rows)


if __name__ == "__main__":
    main()
