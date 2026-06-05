"""Microbenchmark the experimental histogram-subtraction oblivious grower."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chimeraboost.tree import (
    build_oblivious_tree,
    build_oblivious_tree_hist_subtract,
)


def _case(n_rows, n_features, n_bins, seed):
    rng = np.random.default_rng(seed)
    Xb = rng.integers(
        0, n_bins, size=(n_features, n_rows), dtype=np.uint8 if n_bins <= 256 else np.uint16
    )
    signal = Xb[0].astype(np.float64) / n_bins
    if n_features > 3:
        signal += 0.5 * Xb[3].astype(np.float64) / n_bins
    grad = signal - signal.mean() + rng.normal(0.0, 0.1, size=n_rows)
    hess = rng.uniform(0.5, 2.0, size=n_rows)
    n_bins_per_feature = np.full(n_features, n_bins, dtype=np.int64)
    return Xb, grad, hess, n_bins_per_feature


def _time_call(fn, *args, repeat):
    best = None
    last = None
    for _ in range(repeat):
        start = time.perf_counter()
        last = fn(*args)
        elapsed = time.perf_counter() - start
        if best is None or elapsed < best:
            best = elapsed
    return best, last


def run_case(n_rows, n_features, n_bins, depth, repeat, seed):
    Xb, grad, hess, n_bins_per_feature = _case(n_rows, n_features, n_bins, seed)
    args = (Xb, grad, hess, n_bins_per_feature, depth, 1.0, 0.1)

    direct_time, direct = _time_call(build_oblivious_tree, *args, repeat=repeat)
    subtract_time, subtract = _time_call(
        build_oblivious_tree_hist_subtract, *args, repeat=repeat)

    direct_tree, direct_leaf = direct
    subtract_tree, subtract_leaf = subtract
    same = (
        np.array_equal(direct_tree.splits_feat, subtract_tree.splits_feat)
        and np.array_equal(direct_tree.splits_thr, subtract_tree.splits_thr)
        and np.array_equal(direct_leaf, subtract_leaf)
        and np.allclose(direct_tree.values, subtract_tree.values)
    )
    if direct_tree.depth == subtract_tree.depth:
        max_pred_diff = float(
            np.max(np.abs(direct_tree.predict(Xb) - subtract_tree.predict(Xb)))
        )
    else:
        max_pred_diff = float("nan")
    return {
        "n_rows": n_rows,
        "n_features": n_features,
        "n_bins": n_bins,
        "depth": depth,
        "repeat": repeat,
        "direct_seconds": direct_time,
        "subtract_seconds": subtract_time,
        "subtract_vs_direct": subtract_time / direct_time,
        "same_tree": same,
        "max_prediction_diff": max_pred_diff,
        "direct_depth": direct_tree.depth,
        "subtract_depth": subtract_tree.depth,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", type=int, default=[50_000, 250_000])
    parser.add_argument("--features", nargs="+", type=int, default=[12, 40])
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = [
        run_case(n_rows, n_features, args.bins, args.depth, args.repeat, args.seed)
        for n_rows in args.rows
        for n_features in args.features
    ]

    fields = list(rows[0])
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    print(",".join(fields))
    for row in rows:
        print(",".join(str(row[f]) for f in fields))


if __name__ == "__main__":
    main()
