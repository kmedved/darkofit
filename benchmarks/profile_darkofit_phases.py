"""Profile DarkoFit fit phases across dataset size and thread count.

This is a diagnostic companion to ``bench_vs_lightgbm.py``. It does not compare
against LightGBM; it runs short fixed-iteration DarkoFit fits with
``verbose_timing=True`` and writes per-phase timings so we can tell whether
large-data slowdowns come from tree building, loss/gradient work, ordered
updates, validation prediction, or preprocessing.

Example:
    python benchmarks/profile_darkofit_phases.py \
        --size xlarge \
        --datasets numeric_binary numeric_multiclass categorical_binary wide_numeric_reg \
        --threads 1 2 4 8 \
        --iterations 50 \
        --repeat 2
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from darkofit import DarkoClassifier, DarkoRegressor

import bench_vs_lightgbm as bench


PHASES = (
    "preprocess",
    "grad_hess",
    "tree_build",
    "train_update",
    "validation_predict",
    "loss_eval",
)


def _run_fit(spec, size_name, thread_count, args, seed):
    n = bench.SIZE_SAMPLES[size_name]
    rng = np.random.default_rng(args.data_seed + seed)
    X, y, cat_features = spec.builder(n, rng)
    X_train, X_test, y_train, y_test = bench._split_for_task(X, y, spec.task, seed)
    X_fit, X_val, y_fit, y_val = bench._validation_split(
        X_train, y_train, spec.task, seed
    )
    estimator_cls = (
        DarkoRegressor if spec.task == "regression" else DarkoClassifier
    )

    def fit_once():
        model = estimator_cls(
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            num_leaves=args.num_leaves,
            min_child_samples=args.min_child_samples,
            min_gain_to_split=args.min_gain_to_split,
            early_stopping_rounds=None,
            thread_count=thread_count,
            ordered_boosting=False if args.no_ordered_boosting else "auto",
            random_state=seed,
            verbose_timing=True,
            tree_mode=args.tree_mode,
            use_best_model=False,
        )
        start = time.perf_counter()
        model.fit(X_fit, y_fit, cat_features=cat_features, eval_set=(X_val, y_val))
        return model, time.perf_counter() - start

    best_model, best_total = fit_once()
    repeats = max(1, args.repeat)
    for _ in range(repeats - 1):
        model, elapsed = fit_once()
        if elapsed < best_total:
            best_model, best_total = model, elapsed

    timing = best_model.timing_ or {}
    phase_total = sum(float(timing.get(key, 0.0)) for key in PHASES)
    other = max(0.0, best_total - phase_total)
    rounds = int(best_model.best_iteration_)
    row = {
        "dataset": spec.name,
        "task": spec.task,
        "size": size_name,
        "seed": seed,
        "threads": thread_count,
        "repeat": repeats,
        "n_total": len(y),
        "n_fit": len(y_fit),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "n_features": np.asarray(X).shape[1],
        "iterations_requested": args.iterations,
        "iterations_run": rounds,
        "fit_seconds": best_total,
        "seconds_per_round": best_total / max(rounds, 1),
        "phase_seconds_sum": phase_total,
        "other_seconds": other,
    }
    for key in PHASES:
        value = float(timing.get(key, 0.0))
        row[f"{key}_seconds"] = value
        row[f"{key}_per_round"] = value / max(rounds, 1)
        row[f"{key}_share"] = value / best_total if best_total > 0 else 0.0
    return row


def _print_summary(rows):
    print()
    print(
        "dataset                 task        size     thr  rounds  fit_s   "
        "ms/round  tree% grad% update% val% loss%"
    )
    print("-" * 100)
    for row in rows:
        print(
            f"{row['dataset']:23s} {row['task']:11s} {row['size']:8s} "
            f"{row['threads']:3d} {row['iterations_run']:7d} "
            f"{row['fit_seconds']:7.2f} "
            f"{1000.0 * row['seconds_per_round']:8.2f} "
            f"{100.0 * row['tree_build_share']:5.1f} "
            f"{100.0 * row['grad_hess_share']:5.1f} "
            f"{100.0 * row['train_update_share']:7.1f} "
            f"{100.0 * row['validation_predict_share']:5.1f} "
            f"{100.0 * row['loss_eval_share']:5.1f}"
        )


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=bench.SIZE_SAMPLES, default="xlarge")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "numeric_binary",
            "numeric_multiclass",
            "categorical_binary",
            "wide_numeric_reg",
        ],
    )
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-leaves", type=int, default=None)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--min-gain-to-split", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument(
        "--tree-mode",
        choices=[
            "catboost", "oblivious", "lightgbm", "hybrid", "depthwise",
            "levelwise",
        ],
        default="catboost",
    )
    parser.add_argument("--no-ordered-boosting", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--data-seed", type=int, default=20_260_527)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("benchmarks/darkofit_phase_profile.csv"),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    known = {spec.name: spec for spec in bench.DATASETS}
    unknown = set(args.datasets) - set(known)
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {sorted(unknown)}")
    selected = [known[name] for name in args.datasets]

    if not args.no_warmup:
        print("warming up DarkoFit numba kernels...")
        bench._warm_up(
            SimpleNamespace(
                depth=args.depth,
                threads=max(args.threads),
                tree_mode=args.tree_mode,
                darkofit_num_leaves=args.num_leaves,
                darkofit_min_child_samples=args.min_child_samples,
                darkofit_min_gain_to_split=args.min_gain_to_split,
                darkofit_min_child_weight=1.0,
            )
        )

    fields = [
        "dataset",
        "task",
        "size",
        "seed",
        "threads",
        "repeat",
        "n_total",
        "n_fit",
        "n_val",
        "n_test",
        "n_features",
        "iterations_requested",
        "iterations_run",
        "fit_seconds",
        "seconds_per_round",
        "phase_seconds_sum",
        "other_seconds",
    ]
    for key in PHASES:
        fields.extend([f"{key}_seconds", f"{key}_per_round", f"{key}_share"])

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        f.flush()
        for spec in selected:
            for thread_count in args.threads:
                for seed in range(args.seeds):
                    row = _run_fit(spec, args.size, thread_count, args, seed)
                    rows.append(row)
                    writer.writerow(row)
                    f.flush()
                    print(
                        f"done {spec.name:23s} {args.size:8s} "
                        f"threads={thread_count} seed={seed} "
                        f"fit={row['fit_seconds']:.2f}s",
                        flush=True,
                    )

    _print_summary(rows)


if __name__ == "__main__":
    main()
