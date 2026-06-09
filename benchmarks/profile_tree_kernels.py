"""Microbenchmark ChimeraBoost's low-level oblivious-tree kernels.

This benchmark deliberately bypasses sklearn wrappers and preprocessing. It
measures the kernels that dominate warmed training and prediction time:
histogram construction, split search, leaf routing, leaf-value aggregation, and
in-place prediction.

Examples:
    python benchmarks/profile_tree_kernels.py --quick --threads 1 4
    python benchmarks/profile_tree_kernels.py --samples 10000 100000 --features 50
"""

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from chimeraboost.tree import (  # noqa: E402
    _best_split,
    _best_split_serial,
    _build_histograms_into,
    _build_histograms_into_serial,
    _build_histograms_rows_into,
    _build_histograms_rows_into_serial,
    _build_histograms_rows_unit_hess_into,
    _build_histograms_rows_unit_hess_into_serial,
    _build_histograms_selected_into,
    _build_histograms_selected_into_serial,
    _build_histograms_selected_rows_into,
    _build_histograms_selected_rows_into_serial,
    _build_histograms_selected_rows_unit_hess_into,
    _build_histograms_selected_rows_unit_hess_into_serial,
    _build_histograms_selected_unit_hess_into,
    _build_histograms_selected_unit_hess_into_serial,
    _build_histograms_unit_hess_into,
    _build_histograms_unit_hess_into_serial,
    _leaf_values_and_sums,
    _predict_tree_add,
    _refill_left_subtract_right_counts_into,
    _refill_left_subtract_right_counts_positive_into,
    _refill_left_subtract_right_unit_hess_into,
    _refill_leaf_segment_histograms_counts_into,
    _refill_leaf_segment_histograms_counts_into_serial,
    _refill_leaf_segment_histograms_counts_positive_into,
    _refill_leaf_segment_histograms_unit_hess_into,
    _refill_leaf_segment_histograms_unit_hess_into_serial,
    _refill_right_subtract_left_counts_into,
    _refill_right_subtract_left_counts_positive_into,
    _refill_right_subtract_left_unit_hess_into,
    _update_leaves_with_split,
)


def _parse_ints(values):
    out = []
    for value in values:
        out.extend(int(part) for part in str(value).split(",") if part)
    return out


def _bin_dtype(max_bins):
    return np.uint8 if max_bins <= np.iinfo(np.uint8).max + 1 else np.uint16


def _make_case(n_samples, n_features, max_bins, depth, layout, rng,
               changed_fraction=0.0):
    if max_bins < 2:
        raise ValueError("max_bins must be at least 2")
    dtype = _bin_dtype(max_bins)
    X = rng.integers(0, max_bins, size=(n_samples, n_features)).astype(dtype)
    if layout == "F":
        X = np.asfortranarray(X)
    else:
        X = np.ascontiguousarray(X)
    grad = rng.normal(size=n_samples)
    hess = rng.uniform(0.5, 2.0, size=n_samples)
    n_leaves = 1 << depth
    leaf = rng.integers(0, n_leaves, size=n_samples, dtype=np.int64)
    if changed_fraction > 0.0:
        changed_count = min(
            n_samples,
            max(2, int(round(float(changed_fraction) * n_samples))),
        )
        left_count = changed_count // 2
        order = rng.permutation(n_samples)
        leaf[order[:left_count]] = 0
        leaf[order[left_count:changed_count]] = 1
    hg = np.zeros((n_features, n_leaves, max_bins), dtype=np.float64)
    hh = np.zeros_like(hg)
    hc = np.zeros_like(hg)
    n_bins = np.full(n_features, max_bins, dtype=np.int64)
    split_scratch = tuple(np.empty((n_features, n_leaves)) for _ in range(5))
    row_count = max(1, n_samples // 2)
    row_indices = np.sort(
        rng.choice(n_samples, size=row_count, replace=False)
    ).astype(np.int64)
    feature_count = max(1, n_features // 2)
    feature_indices = np.sort(
        rng.choice(n_features, size=feature_count, replace=False)
    ).astype(np.int64)
    feature_mask = np.zeros(n_features, dtype=np.int64)
    feature_mask[feature_indices] = 1
    splits_feat = rng.integers(0, n_features, size=depth, dtype=np.int64)
    splits_thr = rng.integers(0, max_bins - 1, size=depth, dtype=np.int64)
    values = rng.normal(size=n_leaves)
    out = np.zeros(n_samples, dtype=np.float64)
    order = np.argsort(leaf, kind="stable").astype(np.int64)
    counts = np.bincount(leaf, minlength=n_leaves).astype(np.int64)
    leaf_start = np.zeros(n_leaves + 1, dtype=np.int64)
    leaf_start[1:] = np.cumsum(counts)
    changed_leaves = np.array([0, 1], dtype=np.int64)
    return {
        "X": X,
        "grad": grad,
        "hess": hess,
        "leaf": leaf,
        "hg": hg,
        "hh": hh,
        "hc": hc,
        "n_bins": n_bins,
        "split_scratch": split_scratch,
        "row_indices": row_indices,
        "feature_indices": feature_indices,
        "feature_mask": feature_mask,
        "row_order": order,
        "leaf_start": leaf_start,
        "changed_leaves": changed_leaves,
        "splits_feat": splits_feat,
        "splits_thr": splits_thr,
        "values": values,
        "out": out,
        "n_leaves": n_leaves,
    }


def _time_call(fn, repeats, setup=None):
    times = []
    for _ in range(repeats):
        if setup is not None:
            setup()
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    arr = np.asarray(times)
    median_ms = float(np.median(arr) * 1000.0)
    iqr_ms = float(np.subtract(*np.percentile(arr, [75, 25])) * 1000.0)
    return median_ms, iqr_ms


def _kernel_calls(case, threaded):
    X = case["X"]
    grad = case["grad"]
    hess = case["hess"]
    leaf = case["leaf"]
    hg = case["hg"]
    hh = case["hh"]
    hc = case["hc"]
    n_leaves = case["n_leaves"]
    row_order = case["row_order"]
    leaf_start = case["leaf_start"]
    changed_leaves = case["changed_leaves"]
    row_indices = case["row_indices"]
    feature_indices = case["feature_indices"]
    feature_mask = case["feature_mask"]
    n_bins = case["n_bins"]
    scratch = case["split_scratch"]

    if threaded:
        hist = lambda: _build_histograms_into(X, grad, hess, leaf, n_leaves, hg, hh)
        hist_unit = lambda: _build_histograms_unit_hess_into(
            X, grad, leaf, n_leaves, hg, hh
        )
        hist_rows = lambda: _build_histograms_rows_into(
            X, grad, hess, leaf, n_leaves, hg, hh, row_indices
        )
        hist_rows_unit = lambda: _build_histograms_rows_unit_hess_into(
            X, grad, leaf, n_leaves, hg, hh, row_indices
        )
        hist_cols = lambda: _build_histograms_selected_into(
            X, grad, hess, leaf, n_leaves, hg, hh, feature_indices
        )
        hist_cols_unit = lambda: _build_histograms_selected_unit_hess_into(
            X, grad, leaf, n_leaves, hg, hh, feature_indices
        )
        hist_rows_cols = lambda: _build_histograms_selected_rows_into(
            X, grad, hess, leaf, n_leaves, hg, hh, feature_indices, row_indices
        )
        hist_rows_cols_unit = lambda: _build_histograms_selected_rows_unit_hess_into(
            X, grad, leaf, n_leaves, hg, hh, feature_indices, row_indices
        )
        refill_counts = lambda: _refill_leaf_segment_histograms_counts_into(
            X, grad, hess, row_order, leaf_start, changed_leaves, 2, hg, hh, hc
        )
        refill_counts_positive = lambda: _refill_leaf_segment_histograms_counts_positive_into(
            X, grad, hess, row_order, leaf_start, changed_leaves, 2, hg, hh, hc
        )
        refill_unit = lambda: _refill_leaf_segment_histograms_unit_hess_into(
            X, grad, row_order, leaf_start, changed_leaves, 2, hg, hh
        )
        refill_left_sub_counts = lambda: _refill_left_subtract_right_counts_into(
            X, grad, hess, row_order, leaf_start, 0, 1, hg, hh, hc
        )
        refill_left_sub_counts_positive = lambda: _refill_left_subtract_right_counts_positive_into(
            X, grad, hess, row_order, leaf_start, 0, 1, hg, hh, hc
        )
        refill_left_sub_unit = lambda: _refill_left_subtract_right_unit_hess_into(
            X, grad, row_order, leaf_start, 0, 1, hg, hh
        )
        refill_right_sub_counts = lambda: _refill_right_subtract_left_counts_into(
            X, grad, hess, row_order, leaf_start, 0, 1, hg, hh, hc
        )
        refill_right_sub_counts_positive = lambda: _refill_right_subtract_left_counts_positive_into(
            X, grad, hess, row_order, leaf_start, 0, 1, hg, hh, hc
        )
        refill_right_sub_unit = lambda: _refill_right_subtract_left_unit_hess_into(
            X, grad, row_order, leaf_start, 0, 1, hg, hh
        )
        split = lambda: _best_split(
            hg, hh, n_bins, 3.0, feature_mask, 1.0, n_leaves,
            scratch[0], scratch[1], scratch[2], scratch[3], scratch[4],
        )
    else:
        hist = lambda: _build_histograms_into_serial(
            X, grad, hess, leaf, n_leaves, hg, hh
        )
        hist_unit = lambda: _build_histograms_unit_hess_into_serial(
            X, grad, leaf, n_leaves, hg, hh
        )
        hist_rows = lambda: _build_histograms_rows_into_serial(
            X, grad, hess, leaf, n_leaves, hg, hh, row_indices
        )
        hist_rows_unit = lambda: _build_histograms_rows_unit_hess_into_serial(
            X, grad, leaf, n_leaves, hg, hh, row_indices
        )
        hist_cols = lambda: _build_histograms_selected_into_serial(
            X, grad, hess, leaf, n_leaves, hg, hh, feature_indices
        )
        hist_cols_unit = lambda: _build_histograms_selected_unit_hess_into_serial(
            X, grad, leaf, n_leaves, hg, hh, feature_indices
        )
        hist_rows_cols = lambda: _build_histograms_selected_rows_into_serial(
            X, grad, hess, leaf, n_leaves, hg, hh, feature_indices, row_indices
        )
        hist_rows_cols_unit = lambda: _build_histograms_selected_rows_unit_hess_into_serial(
            X, grad, leaf, n_leaves, hg, hh, feature_indices, row_indices
        )
        refill_counts = lambda: _refill_leaf_segment_histograms_counts_into_serial(
            X, grad, hess, row_order, leaf_start, changed_leaves, 2, hg, hh, hc
        )
        refill_counts_positive = refill_counts
        refill_unit = lambda: _refill_leaf_segment_histograms_unit_hess_into_serial(
            X, grad, row_order, leaf_start, changed_leaves, 2, hg, hh
        )
        refill_left_sub_counts = None
        refill_left_sub_counts_positive = None
        refill_left_sub_unit = None
        refill_right_sub_counts = None
        refill_right_sub_counts_positive = None
        refill_right_sub_unit = None
        split = lambda: _best_split_serial(
            hg, hh, n_bins, 3.0, feature_mask, 1.0, n_leaves
        )

    update_leaf = case["leaf"].copy()
    out = case["out"]
    calls = {
        "hist": (hist, None),
        "hist_unit_hess": (hist_unit, None),
        "hist_rows_50pct": (hist_rows, None),
        "hist_rows_50pct_unit": (hist_rows_unit, None),
        "hist_cols_50pct": (hist_cols, None),
        "hist_cols_50pct_unit": (hist_cols_unit, None),
        "hist_rows_cols_50pct": (hist_rows_cols, None),
        "hist_rows_cols_50pct_unit": (hist_rows_cols_unit, None),
        "refill_counts_2leaves": (refill_counts, None),
        "refill_counts_positive_2leaves": (refill_counts_positive, None),
        "refill_unit_2leaves": (refill_unit, None),
        "best_split": (split, hist),
        "leaf_values": (
            lambda: _leaf_values_and_sums(
                leaf, grad, hess, n_leaves, 3.0, 0.1
            ),
            None,
        ),
        "update_leaves": (
            lambda: _update_leaves_with_split(
                case["X"], update_leaf, case["splits_feat"][0], case["splits_thr"][0]
            ),
            lambda: update_leaf.__setitem__(slice(None), leaf),
        ),
        "predict_add": (
            lambda: _predict_tree_add(
                case["X"], case["splits_feat"], case["splits_thr"],
                case["values"], out
            ),
            lambda: out.fill(0.0),
        ),
    }
    if refill_right_sub_counts is not None:
        calls.update({
            "refill_left_sub_counts": (refill_left_sub_counts, hist),
            "refill_left_sub_counts_positive": (
                refill_left_sub_counts_positive, hist
            ),
            "refill_left_sub_unit": (refill_left_sub_unit, hist_unit),
            "refill_right_sub_counts": (refill_right_sub_counts, hist),
            "refill_right_sub_counts_positive": (
                refill_right_sub_counts_positive, hist
            ),
            "refill_right_sub_unit": (refill_right_sub_unit, hist_unit),
        })
    return calls


def _run_case(n_samples, n_features, max_bins, depth, threads, layout, repeats,
              changed_fraction):
    import numba

    old_threads = numba.get_num_threads()
    rng = np.random.default_rng(0)
    try:
        threads = max(1, min(int(threads), numba.config.NUMBA_NUM_THREADS))
        numba.set_num_threads(threads)
        case = _make_case(
            n_samples, n_features, max_bins, depth, layout, rng,
            changed_fraction=changed_fraction,
        )
        threaded = threads > 1
        calls = _kernel_calls(case, threaded)

        # Compile and prime histograms before measuring.
        for fn, setup in calls.values():
            if setup is not None:
                setup()
            fn()

        rows = []
        for name, (fn, setup) in calls.items():
            median_ms, iqr_ms = _time_call(fn, repeats, setup)
            rows.append((name, median_ms, iqr_ms))
        return rows, case["X"].dtype, threads
    finally:
        numba.set_num_threads(old_threads)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", default=["1000", "10000"])
    parser.add_argument("--features", nargs="+", default=["10", "50"])
    parser.add_argument("--bins", nargs="+", default=["32", "128"])
    parser.add_argument("--depths", nargs="+", default=["4", "6"])
    parser.add_argument("--threads", nargs="+", default=["1"])
    parser.add_argument("--layouts", nargs="+", choices=["C", "F"], default=["C"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--changed-fraction",
        type=float,
        default=0.0,
        help=(
            "fraction of rows assigned to the two changed leaves used by "
            "leafwise refill microbenchmarks"
        ),
    )
    parser.add_argument("--quick", action="store_true",
                        help="run one small case; useful for smoke tests")
    args = parser.parse_args()

    if args.quick:
        args.samples = [1000]
        args.features = [10]
        args.bins = [32]
        args.depths = [4]
        args.repeats = min(args.repeats, 3)

    print(
        "n_samples,n_features,max_bins,depth,threads,layout,dtype,kernel,"
        "median_ms,iqr_ms"
    )
    for n_samples in _parse_ints(args.samples):
        for n_features in _parse_ints(args.features):
            for max_bins in _parse_ints(args.bins):
                for depth in _parse_ints(args.depths):
                    for threads in _parse_ints(args.threads):
                        for layout in args.layouts:
                            rows, dtype, effective_threads = _run_case(
                                n_samples, n_features, max_bins, depth,
                                threads, layout, args.repeats,
                                args.changed_fraction,
                            )
                            for kernel, median_ms, iqr_ms in rows:
                                print(
                                    f"{n_samples},{n_features},{max_bins},"
                                    f"{depth},{effective_threads},{layout},{dtype},"
                                    f"{kernel},{median_ms:.6f},{iqr_ms:.6f}"
                                )


if __name__ == "__main__":
    main()
