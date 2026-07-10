"""Oblivious (symmetric) decision trees, numba-accelerated.

An oblivious tree of depth D uses the SAME (feature, bin-threshold) split at
every node of a given level. A row's leaf is therefore just a D-bit number, one
bit per level: bit_d = 1 if X[feature_d] > threshold_d else 0. This makes:

  * prediction a handful of comparisons + an array lookup (very fast), and
  * the model strongly regularized (only D splits per tree, shared across the
    whole level), which is a big part of why the defaults don't overfit.

We grow level by level. At each level we build per-(feature, current-leaf, bin)
gradient/hessian histograms and pick the single split that maximizes the summed
XGBoost-style gain over all current leaves.
"""

import numpy as np
from numba import get_num_threads, njit, prange


@njit(cache=True, parallel=True)
def _build_histograms_into(X_binned, grad, hess, leaf, n_leaves, hg, hh):
    """Fill per-feature gradient/hessian histograms into pre-allocated buffers.

    hg, hh are caller-owned arrays of shape (n_features, max_leaves, max_bins),
    reused across every tree and level to avoid reallocating gigabytes over a
    long boosting run. We zero only the (n_leaves) slice we are about to write,
    then accumulate. Parallelized over features so each thread owns a disjoint
    slice -- no write races, no locks.
    """
    n_samples, n_features = X_binned.shape
    max_bins = hg.shape[2]
    for f in prange(n_features):
        # Zero this feature's active region (only the leaves/bins we will use).
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
        for i in range(n_samples):
            l = leaf[i]
            b = X_binned[i, f]
            hg[f, l, b] += grad[i]
            hh[f, l, b] += hess[i]


@njit(cache=True, parallel=True)
def _build_histograms_selected_into(X_binned, grad, hess, leaf, n_leaves,
                                    hg, hh, feature_indices):
    """Fill histograms only for selected feature columns."""
    n_samples = X_binned.shape[0]
    max_bins = hg.shape[2]
    for jj in prange(feature_indices.shape[0]):
        f = feature_indices[jj]
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
        for i in range(n_samples):
            l = leaf[i]
            b = X_binned[i, f]
            hg[f, l, b] += grad[i]
            hh[f, l, b] += hess[i]


@njit(cache=True, parallel=True)
def _build_histograms_rows_into(X_binned, grad, hess, leaf, n_leaves, hg, hh,
                                row_indices):
    """Fill histograms from selected rows only, for all feature columns."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
        for p in range(row_indices.shape[0]):
            i = row_indices[p]
            l = leaf[i]
            b = X_binned[i, f]
            hg[f, l, b] += grad[i]
            hh[f, l, b] += hess[i]


@njit(cache=True, parallel=True)
def _build_histograms_selected_rows_into(X_binned, grad, hess, leaf, n_leaves,
                                         hg, hh, feature_indices, row_indices):
    """Fill histograms from selected rows and selected feature columns."""
    max_bins = hg.shape[2]
    for jj in prange(feature_indices.shape[0]):
        f = feature_indices[jj]
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
        for p in range(row_indices.shape[0]):
            i = row_indices[p]
            l = leaf[i]
            b = X_binned[i, f]
            hg[f, l, b] += grad[i]
            hh[f, l, b] += hess[i]


@njit(cache=True, parallel=True)
def _build_histograms_unit_hess_into(X_binned, grad, leaf, n_leaves, hg, hh):
    """Fill histograms for unit-Hessian losses without loading hess[i]."""
    n_samples, n_features = X_binned.shape
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
        for i in range(n_samples):
            l = leaf[i]
            b = X_binned[i, f]
            hg[f, l, b] += grad[i]
            hh[f, l, b] += 1.0


@njit(cache=True, parallel=True)
def _build_histograms_selected_unit_hess_into(X_binned, grad, leaf, n_leaves,
                                              hg, hh, feature_indices):
    """Fill selected-feature histograms for unit-Hessian losses."""
    n_samples = X_binned.shape[0]
    max_bins = hg.shape[2]
    for jj in prange(feature_indices.shape[0]):
        f = feature_indices[jj]
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
        for i in range(n_samples):
            l = leaf[i]
            b = X_binned[i, f]
            hg[f, l, b] += grad[i]
            hh[f, l, b] += 1.0


@njit(cache=True, parallel=True)
def _build_histograms_rows_unit_hess_into(X_binned, grad, leaf, n_leaves, hg,
                                          hh, row_indices):
    """Fill selected-row histograms for unit-Hessian losses."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
        for p in range(row_indices.shape[0]):
            i = row_indices[p]
            l = leaf[i]
            b = X_binned[i, f]
            hg[f, l, b] += grad[i]
            hh[f, l, b] += 1.0


@njit(cache=True, parallel=True)
def _build_histograms_selected_rows_unit_hess_into(X_binned, grad, leaf,
                                                   n_leaves, hg, hh,
                                                   feature_indices,
                                                   row_indices):
    """Fill selected-row/feature histograms for unit-Hessian losses."""
    max_bins = hg.shape[2]
    for jj in prange(feature_indices.shape[0]):
        f = feature_indices[jj]
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
        for p in range(row_indices.shape[0]):
            i = row_indices[p]
            l = leaf[i]
            b = X_binned[i, f]
            hg[f, l, b] += grad[i]
            hh[f, l, b] += 1.0


@njit(cache=True)
def _build_histograms_into_serial(X_binned, grad, hess, leaf, n_leaves, hg, hh):
    """Single-thread histogram fill with row-contiguous feature access.

    For each histogram cell, rows are accumulated in the same increasing-index
    order as the feature-parallel kernel, preserving floating-point results.
    """
    n_samples, n_features = X_binned.shape
    max_bins = hg.shape[2]
    for f in range(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0

    for i in range(n_samples):
        l = leaf[i]
        gi = grad[i]
        hi = hess[i]
        for f in range(n_features):
            b = X_binned[i, f]
            hg[f, l, b] += gi
            hh[f, l, b] += hi


@njit(cache=True)
def _build_histograms_rows_into_serial(X_binned, grad, hess, leaf, n_leaves,
                                       hg, hh, row_indices):
    """Single-thread histogram fill from selected rows only."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in range(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0

    for p in range(row_indices.shape[0]):
        i = row_indices[p]
        l = leaf[i]
        gi = grad[i]
        hi = hess[i]
        for f in range(n_features):
            b = X_binned[i, f]
            hg[f, l, b] += gi
            hh[f, l, b] += hi


@njit(cache=True)
def _build_histograms_selected_into_serial(X_binned, grad, hess, leaf,
                                           n_leaves, hg, hh, feature_indices):
    """Single-thread histogram fill for selected columns only."""
    n_samples = X_binned.shape[0]
    max_bins = hg.shape[2]
    for jj in range(feature_indices.shape[0]):
        f = feature_indices[jj]
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0

    for i in range(n_samples):
        l = leaf[i]
        gi = grad[i]
        hi = hess[i]
        for jj in range(feature_indices.shape[0]):
            f = feature_indices[jj]
            b = X_binned[i, f]
            hg[f, l, b] += gi
            hh[f, l, b] += hi


@njit(cache=True)
def _build_histograms_selected_rows_into_serial(X_binned, grad, hess, leaf,
                                                n_leaves, hg, hh,
                                                feature_indices, row_indices):
    """Single-thread histogram fill for selected rows and columns."""
    max_bins = hg.shape[2]
    for jj in range(feature_indices.shape[0]):
        f = feature_indices[jj]
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0

    for p in range(row_indices.shape[0]):
        i = row_indices[p]
        l = leaf[i]
        gi = grad[i]
        hi = hess[i]
        for jj in range(feature_indices.shape[0]):
            f = feature_indices[jj]
            b = X_binned[i, f]
            hg[f, l, b] += gi
            hh[f, l, b] += hi


@njit(cache=True)
def _build_histograms_unit_hess_into_serial(X_binned, grad, leaf, n_leaves,
                                            hg, hh):
    """Single-thread histogram fill for unit-Hessian losses."""
    n_samples, n_features = X_binned.shape
    max_bins = hg.shape[2]
    for f in range(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0

    for i in range(n_samples):
        l = leaf[i]
        gi = grad[i]
        for f in range(n_features):
            b = X_binned[i, f]
            hg[f, l, b] += gi
            hh[f, l, b] += 1.0


@njit(cache=True)
def _build_histograms_rows_unit_hess_into_serial(X_binned, grad, leaf,
                                                 n_leaves, hg, hh,
                                                 row_indices):
    """Single-thread selected-row histogram fill for unit-Hessian losses."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in range(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0

    for p in range(row_indices.shape[0]):
        i = row_indices[p]
        l = leaf[i]
        gi = grad[i]
        for f in range(n_features):
            b = X_binned[i, f]
            hg[f, l, b] += gi
            hh[f, l, b] += 1.0


@njit(cache=True)
def _build_histograms_selected_unit_hess_into_serial(X_binned, grad, leaf,
                                                     n_leaves, hg, hh,
                                                     feature_indices):
    """Single-thread selected-feature histogram fill for unit-Hessian losses."""
    n_samples = X_binned.shape[0]
    max_bins = hg.shape[2]
    for jj in range(feature_indices.shape[0]):
        f = feature_indices[jj]
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0

    for i in range(n_samples):
        l = leaf[i]
        gi = grad[i]
        for jj in range(feature_indices.shape[0]):
            f = feature_indices[jj]
            b = X_binned[i, f]
            hg[f, l, b] += gi
            hh[f, l, b] += 1.0


@njit(cache=True)
def _build_histograms_selected_rows_unit_hess_into_serial(
    X_binned, grad, leaf, n_leaves, hg, hh, feature_indices, row_indices
):
    """Single-thread selected-row/feature hist fill for unit-Hessian losses."""
    max_bins = hg.shape[2]
    for jj in range(feature_indices.shape[0]):
        f = feature_indices[jj]
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0

    for p in range(row_indices.shape[0]):
        i = row_indices[p]
        l = leaf[i]
        gi = grad[i]
        for jj in range(feature_indices.shape[0]):
            f = feature_indices[jj]
            b = X_binned[i, f]
            hg[f, l, b] += gi
            hh[f, l, b] += 1.0


@njit(cache=True, parallel=True)
def _build_counts_into(X_binned, hess, leaf, n_leaves, hc):
    """Fill per-feature positive-weight row-count histograms."""
    n_samples, n_features = X_binned.shape
    max_bins = hc.shape[2]
    for f in prange(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                hc[f, l, b] = 0.0
        for i in range(n_samples):
            if hess[i] > 0.0:
                l = leaf[i]
                b = X_binned[i, f]
                hc[f, l, b] += 1.0


@njit(cache=True, parallel=True)
def _build_counts_rows_into(X_binned, hess, leaf, n_leaves, hc, row_indices):
    """Fill positive-weight row-count histograms from selected rows."""
    n_features = X_binned.shape[1]
    max_bins = hc.shape[2]
    for f in prange(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                hc[f, l, b] = 0.0
        for p in range(row_indices.shape[0]):
            i = row_indices[p]
            if hess[i] > 0.0:
                l = leaf[i]
                b = X_binned[i, f]
                hc[f, l, b] += 1.0


@njit(cache=True, parallel=True)
def _build_histograms_counts_into(X_binned, grad, hess, leaf, n_leaves, hg, hh, hc):
    """Fill gradient, hessian, and positive-row-count histograms in one pass."""
    n_samples, n_features = X_binned.shape
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
                hc[f, l, b] = 0.0
        for i in range(n_samples):
            l = leaf[i]
            b = X_binned[i, f]
            hg[f, l, b] += grad[i]
            hi = hess[i]
            hh[f, l, b] += hi
            if hi > 0.0:
                hc[f, l, b] += 1.0


@njit(cache=True, parallel=True)
def _build_histograms_counts_positive_into(
    X_binned, grad, hess, leaf, n_leaves, hg, hh, hc
):
    """Fill histograms when every scanned row has positive Hessian."""
    n_samples, n_features = X_binned.shape
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
                hc[f, l, b] = 0.0
        for i in range(n_samples):
            l = leaf[i]
            b = X_binned[i, f]
            hg[f, l, b] += grad[i]
            hh[f, l, b] += hess[i]
            hc[f, l, b] += 1.0


@njit(cache=True, parallel=True)
def _build_histograms_counts_selected_into(X_binned, grad, hess, leaf,
                                           n_leaves, hg, hh, hc,
                                           feature_indices):
    """Fill selected-feature gradient/hessian/count histograms in one pass."""
    n_samples = X_binned.shape[0]
    max_bins = hg.shape[2]
    for jj in prange(feature_indices.shape[0]):
        f = feature_indices[jj]
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
                hc[f, l, b] = 0.0
        for i in range(n_samples):
            l = leaf[i]
            b = X_binned[i, f]
            hg[f, l, b] += grad[i]
            hi = hess[i]
            hh[f, l, b] += hi
            if hi > 0.0:
                hc[f, l, b] += 1.0


@njit(cache=True)
def _count_leaf_rows(leaf, row_order, n_leaves, counts):
    """Count rows per leaf for either int64 or uint32 leaf-id streams."""
    for l in range(n_leaves):
        counts[l] = 0
    for p in range(row_order.shape[0]):
        counts[int(leaf[row_order[p]])] += 1


@njit(cache=True, parallel=True)
def _build_histograms_counts_rows_into(X_binned, grad, hess, leaf,
                                       n_leaves, hg, hh, hc, row_indices):
    """Fill selected-row gradient/hessian/count histograms in one pass."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
                hc[f, l, b] = 0.0
        for p in range(row_indices.shape[0]):
            i = row_indices[p]
            l = leaf[i]
            b = X_binned[i, f]
            hg[f, l, b] += grad[i]
            hi = hess[i]
            hh[f, l, b] += hi
            if hi > 0.0:
                hc[f, l, b] += 1.0


@njit(cache=True, parallel=True)
def _build_histograms_counts_selected_rows_into(X_binned, grad, hess, leaf,
                                                n_leaves, hg, hh, hc,
                                                feature_indices, row_indices):
    """Fill selected-row/feature grad/hess/count histograms in one pass."""
    max_bins = hg.shape[2]
    for jj in prange(feature_indices.shape[0]):
        f = feature_indices[jj]
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
                hc[f, l, b] = 0.0
        for p in range(row_indices.shape[0]):
            i = row_indices[p]
            l = leaf[i]
            b = X_binned[i, f]
            hg[f, l, b] += grad[i]
            hi = hess[i]
            hh[f, l, b] += hi
            if hi > 0.0:
                hc[f, l, b] += 1.0


# --------------------------------------------------------------------------
# Row-parallel histogram kernels.
#
# The feature-parallel kernels above re-read grad/hess/leaf once per feature
# (24 bytes x rows x features of redundant gather traffic), which is why their
# thread scaling saturates on large fits. These kernels instead split the rows
# into chunks, accumulate each chunk row-major into a thread-local buffer of
# shape (n_chunks, n_features, leaf_slots, max_bins), and then reduce the
# chunks feature-parallel. grad/hess/leaf are read exactly once and X is read
# row-contiguously, so callers should pass the C-order matrix here (not the
# Fortran copy used by the feature-parallel kernels).
#
# Summation order differs from the serial/feature-parallel kernels (per-chunk
# partials combined in chunk order), so results match to float64 rounding,
# not bitwise; for a fixed local-buffer shape they are deterministic.
# --------------------------------------------------------------------------

# Modes for the leaf-wise segment kernels.
_ROWPAR_ROOT = 0        # write the scanned sums into leaf 0, no sibling
_ROWPAR_SCAN_RIGHT = 1  # scanned = right child; left holds parent, subtract
_ROWPAR_SCAN_LEFT = 2   # scanned = left child; derive right = parent - left


@njit(cache=True, parallel=True)
def _build_histograms_rowpar_into(X_binned, grad, hess, leaf, n_leaves,
                                  hg, hh, lg, lh):
    """Row-parallel gradient/hessian histogram fill for all leaves."""
    n_samples, n_features = X_binned.shape
    max_bins = hg.shape[2]
    n_chunks = lg.shape[0]
    chunk = (n_samples + n_chunks - 1) // n_chunks
    for t in prange(n_chunks):
        for f in range(n_features):
            for l in range(n_leaves):
                for b in range(max_bins):
                    lg[t, f, l, b] = 0.0
                    lh[t, f, l, b] = 0.0
        start = t * chunk
        end = min(n_samples, start + chunk)
        for i in range(start, end):
            l = leaf[i]
            gi = grad[i]
            hi = hess[i]
            for f in range(n_features):
                b = X_binned[i, f]
                lg[t, f, l, b] += gi
                lh[t, f, l, b] += hi
    for f in prange(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                g = 0.0
                h = 0.0
                for t in range(n_chunks):
                    g += lg[t, f, l, b]
                    h += lh[t, f, l, b]
                hg[f, l, b] = g
                hh[f, l, b] = h


@njit(cache=True, parallel=True)
def _build_histograms_unit_hess_rowpar_into(X_binned, grad, leaf, n_leaves,
                                            hg, hh, lg, lh):
    """Row-parallel unit-Hessian histogram fill for all leaves."""
    n_samples, n_features = X_binned.shape
    max_bins = hg.shape[2]
    n_chunks = lg.shape[0]
    chunk = (n_samples + n_chunks - 1) // n_chunks
    for t in prange(n_chunks):
        for f in range(n_features):
            for l in range(n_leaves):
                for b in range(max_bins):
                    lg[t, f, l, b] = 0.0
                    lh[t, f, l, b] = 0.0
        start = t * chunk
        end = min(n_samples, start + chunk)
        for i in range(start, end):
            l = leaf[i]
            gi = grad[i]
            for f in range(n_features):
                b = X_binned[i, f]
                lg[t, f, l, b] += gi
                lh[t, f, l, b] += 1.0
    for f in prange(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                g = 0.0
                h = 0.0
                for t in range(n_chunks):
                    g += lg[t, f, l, b]
                    h += lh[t, f, l, b]
                hg[f, l, b] = g
                hh[f, l, b] = h


@njit(cache=True, parallel=True)
def _leafwise_segment_hist_rowpar_counts(X_binned, grad, hess, row_order,
                                         seg_start, seg_end, left_leaf,
                                         right_leaf, mode, hg, hh, hc,
                                         lg, lh, lc):
    """Row-parallel grad/hess/count fill of one leaf-wise row segment.

    mode selects what happens at merge time:
      _ROWPAR_ROOT       write sums into leaf 0 (initial full build)
      _ROWPAR_SCAN_RIGHT scanned segment is the right child; the parent
                         histogram sits in left_leaf and right is subtracted
                         from it
      _ROWPAR_SCAN_LEFT  scanned segment is the left child; the parent sits in
                         left_leaf and the right child is derived as
                         parent - left
    """
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    n_chunks = lg.shape[0]
    seg_len = seg_end - seg_start
    chunk = (seg_len + n_chunks - 1) // n_chunks
    for t in prange(n_chunks):
        for f in range(n_features):
            for b in range(max_bins):
                lg[t, f, 0, b] = 0.0
                lh[t, f, 0, b] = 0.0
                lc[t, f, 0, b] = 0.0
        start = seg_start + t * chunk
        end = min(seg_end, start + chunk)
        for p in range(start, end):
            i = row_order[p]
            gi = grad[i]
            hi = hess[i]
            for f in range(n_features):
                b = X_binned[i, f]
                lg[t, f, 0, b] += gi
                lh[t, f, 0, b] += hi
                if hi > 0.0:
                    lc[t, f, 0, b] += 1.0
    for f in prange(n_features):
        for b in range(max_bins):
            g = 0.0
            h = 0.0
            c = 0.0
            for t in range(n_chunks):
                g += lg[t, f, 0, b]
                h += lh[t, f, 0, b]
                c += lc[t, f, 0, b]
            if mode == _ROWPAR_ROOT:
                hg[f, 0, b] = g
                hh[f, 0, b] = h
                hc[f, 0, b] = c
            elif mode == _ROWPAR_SCAN_RIGHT:
                hg[f, right_leaf, b] = g
                hh[f, right_leaf, b] = h
                hc[f, right_leaf, b] = c
                hg[f, left_leaf, b] -= g
                hh[f, left_leaf, b] -= h
                hc[f, left_leaf, b] -= c
            else:
                pg = hg[f, left_leaf, b]
                ph = hh[f, left_leaf, b]
                pc = hc[f, left_leaf, b]
                hg[f, left_leaf, b] = g
                hh[f, left_leaf, b] = h
                hc[f, left_leaf, b] = c
                hg[f, right_leaf, b] = pg - g
                hh[f, right_leaf, b] = ph - h
                hc[f, right_leaf, b] = pc - c


@njit(cache=True, parallel=True)
def _leafwise_segment_hist_rowpar_unit_hess(X_binned, grad, row_order,
                                            seg_start, seg_end, left_leaf,
                                            right_leaf, mode, hg, hh, lg, lh):
    """Row-parallel unit-Hessian fill of one leaf-wise row segment."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    n_chunks = lg.shape[0]
    seg_len = seg_end - seg_start
    chunk = (seg_len + n_chunks - 1) // n_chunks
    for t in prange(n_chunks):
        for f in range(n_features):
            for b in range(max_bins):
                lg[t, f, 0, b] = 0.0
                lh[t, f, 0, b] = 0.0
        start = seg_start + t * chunk
        end = min(seg_end, start + chunk)
        for p in range(start, end):
            i = row_order[p]
            gi = grad[i]
            for f in range(n_features):
                b = X_binned[i, f]
                lg[t, f, 0, b] += gi
                lh[t, f, 0, b] += 1.0
    for f in prange(n_features):
        for b in range(max_bins):
            g = 0.0
            h = 0.0
            for t in range(n_chunks):
                g += lg[t, f, 0, b]
                h += lh[t, f, 0, b]
            if mode == _ROWPAR_ROOT:
                hg[f, 0, b] = g
                hh[f, 0, b] = h
            elif mode == _ROWPAR_SCAN_RIGHT:
                hg[f, right_leaf, b] = g
                hh[f, right_leaf, b] = h
                hg[f, left_leaf, b] -= g
                hh[f, left_leaf, b] -= h
            else:
                pg = hg[f, left_leaf, b]
                ph = hh[f, left_leaf, b]
                hg[f, left_leaf, b] = g
                hh[f, left_leaf, b] = h
                hg[f, right_leaf, b] = pg - g
                hh[f, right_leaf, b] = ph - h


def _rowpar_eligible(n_scanned, rowpar_buffers, n_leaves, max_bins):
    """Heuristic: row-parallel wins once the rows scanned dwarf the
    per-chunk buffer traffic (zero + merge of n_chunks*n_leaves*max_bins
    cells per feature)."""
    if rowpar_buffers is None:
        return False
    lg = rowpar_buffers[0]
    if lg.shape[2] < n_leaves:
        return False
    return n_scanned >= 4 * lg.shape[0] * n_leaves * max_bins


def _resolve_leafwise_row_layout(
    requested,
    n_samples,
    n_features,
    row_indices,
    feature_indices,
    feature_mask,
    reuse_leaf_histograms,
    max_leaves,
    fast_lane_eligible,
):
    """Resolve the leafwise row layout from request and lane guards.

    Explicit ``"segmented"`` remains available for direct benchmarks and
    equivalence tests. Automatic selection is conservative until
    profile-labeled benchmarks justify a size threshold, and it must not
    disable faster prefix lanes that currently require ``not use_segmented_rows``.
    """
    if requested not in {"auto", "prefix", "segmented"}:
        raise ValueError("leafwise_row_layout must be 'auto', 'prefix', or 'segmented'")

    full_rows_features = (
        row_indices is None
        and feature_indices is None
        and bool(np.all(feature_mask != 0))
    )
    can_use_segmented_rows = reuse_leaf_histograms and full_rows_features
    if requested == "segmented":
        if not can_use_segmented_rows:
            raise ValueError(
                "leafwise_row_layout='segmented' requires full rows, full features, "
                "and reuse_leaf_histograms=True"
            )
        return "segmented"

    if requested == "auto":
        _ = (n_samples, n_features, max_leaves)
        if not can_use_segmented_rows or fast_lane_eligible:
            return "prefix"
        # No benchmark-backed threshold has landed yet. Keep the old default.
        return "prefix"

    return "prefix"


# --------------------------------------------------------------------------
# Level-wise sibling subtraction.
#
# At level d >= 1 the histogram buffer still holds every level-(d-1) parent
# histogram, so each parent's smaller child is rebuilt from its rows and the
# larger child is derived as parent - smaller. Scanning the smaller side
# bounds the scan at half the rows and makes empty children exact (the
# scanned side has no rows, so the derived sibling equals the parent
# bit-for-bit), which keeps the min_child_weight legality checks clean.
#
# The expand phase walks parents in descending order: at step p only slots
# 2p and 2p+1 are written and only slot p is read, and 2p' >= p' + 1 > any
# future parent index, so no parent is clobbered before it is consumed.
# --------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def _build_level_histograms_subtract_into(X_binned, grad, hess, leaf,
                                          scan_idx, scan_side, n_parents,
                                          hg, hh):
    """Derive one level's histograms from parents plus a smaller-side scan."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for p in range(n_parents - 1, -1, -1):
            s = scan_side[p]
            derived = 2 * p + (1 - s)
            scanned = 2 * p + s
            if derived != p:
                for b in range(max_bins):
                    hg[f, derived, b] = hg[f, p, b]
                    hh[f, derived, b] = hh[f, p, b]
            for b in range(max_bins):
                hg[f, scanned, b] = 0.0
                hh[f, scanned, b] = 0.0
        for pp in range(scan_idx.shape[0]):
            i = scan_idx[pp]
            l = leaf[i]
            b = X_binned[i, f]
            hg[f, l, b] += grad[i]
            hh[f, l, b] += hess[i]
        for p in range(n_parents):
            s = scan_side[p]
            derived = 2 * p + (1 - s)
            scanned = 2 * p + s
            for b in range(max_bins):
                hg[f, derived, b] -= hg[f, scanned, b]
                hh[f, derived, b] -= hh[f, scanned, b]


@njit(cache=True, parallel=True)
def _build_level_histograms_subtract_unit_hess_into(X_binned, grad, leaf,
                                                    scan_idx, scan_side,
                                                    n_parents, hg, hh):
    """Unit-Hessian variant of the level subtraction kernel."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for p in range(n_parents - 1, -1, -1):
            s = scan_side[p]
            derived = 2 * p + (1 - s)
            scanned = 2 * p + s
            if derived != p:
                for b in range(max_bins):
                    hg[f, derived, b] = hg[f, p, b]
                    hh[f, derived, b] = hh[f, p, b]
            for b in range(max_bins):
                hg[f, scanned, b] = 0.0
                hh[f, scanned, b] = 0.0
        for pp in range(scan_idx.shape[0]):
            i = scan_idx[pp]
            l = leaf[i]
            b = X_binned[i, f]
            hg[f, l, b] += grad[i]
            hh[f, l, b] += 1.0
        for p in range(n_parents):
            s = scan_side[p]
            derived = 2 * p + (1 - s)
            scanned = 2 * p + s
            for b in range(max_bins):
                hg[f, derived, b] -= hg[f, scanned, b]
                hh[f, derived, b] -= hh[f, scanned, b]


@njit(cache=True)
def _build_level_histograms_subtract_into_serial(X_binned, grad, hess, leaf,
                                                 scan_idx, scan_side,
                                                 n_parents, hg, hh):
    """Single-thread level subtraction with row-contiguous scanning."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for p in range(n_parents - 1, -1, -1):
        s = scan_side[p]
        derived = 2 * p + (1 - s)
        scanned = 2 * p + s
        for f in range(n_features):
            if derived != p:
                for b in range(max_bins):
                    hg[f, derived, b] = hg[f, p, b]
                    hh[f, derived, b] = hh[f, p, b]
            for b in range(max_bins):
                hg[f, scanned, b] = 0.0
                hh[f, scanned, b] = 0.0
    for pp in range(scan_idx.shape[0]):
        i = scan_idx[pp]
        l = leaf[i]
        gi = grad[i]
        hi = hess[i]
        for f in range(n_features):
            b = X_binned[i, f]
            hg[f, l, b] += gi
            hh[f, l, b] += hi
    for p in range(n_parents):
        s = scan_side[p]
        derived = 2 * p + (1 - s)
        scanned = 2 * p + s
        for f in range(n_features):
            for b in range(max_bins):
                hg[f, derived, b] -= hg[f, scanned, b]
                hh[f, derived, b] -= hh[f, scanned, b]


@njit(cache=True)
def _build_level_histograms_subtract_unit_hess_into_serial(
    X_binned, grad, leaf, scan_idx, scan_side, n_parents, hg, hh
):
    """Single-thread unit-Hessian level subtraction."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for p in range(n_parents - 1, -1, -1):
        s = scan_side[p]
        derived = 2 * p + (1 - s)
        scanned = 2 * p + s
        for f in range(n_features):
            if derived != p:
                for b in range(max_bins):
                    hg[f, derived, b] = hg[f, p, b]
                    hh[f, derived, b] = hh[f, p, b]
            for b in range(max_bins):
                hg[f, scanned, b] = 0.0
                hh[f, scanned, b] = 0.0
    for pp in range(scan_idx.shape[0]):
        i = scan_idx[pp]
        l = leaf[i]
        gi = grad[i]
        for f in range(n_features):
            b = X_binned[i, f]
            hg[f, l, b] += gi
            hh[f, l, b] += 1.0
    for p in range(n_parents):
        s = scan_side[p]
        derived = 2 * p + (1 - s)
        scanned = 2 * p + s
        for f in range(n_features):
            for b in range(max_bins):
                hg[f, derived, b] -= hg[f, scanned, b]
                hh[f, derived, b] -= hh[f, scanned, b]


def _level_scan_plan(leaf, n_parents):
    """Choose each parent's smaller child and list the rows to scan."""
    counts = np.bincount(leaf, minlength=2 * n_parents)
    scan_side = (counts[1::2] < counts[0::2]).astype(np.int64)
    scan_idx = np.flatnonzero(
        (leaf & 1) == scan_side[leaf >> 1]
    ).astype(np.int64)
    return scan_idx, scan_side


# Subtraction halves the per-row work, which wins while a fit is compute
# bound (1-2 threads); at higher thread counts the level build is bandwidth
# bound, the sparse ascending row gather touches the same cache lines as a
# full scan, and the extra expand/subtract passes made it measurably slower.
_LEVEL_SUBTRACTION_MAX_THREADS = 2


def _resolve_level_subtraction(level_histogram_subtraction):
    if level_histogram_subtraction == "auto":
        return get_num_threads() <= _LEVEL_SUBTRACTION_MAX_THREADS
    return bool(level_histogram_subtraction)


def _normalize_leaf_dtype(leaf_dtype):
    try:
        dtype = np.dtype(leaf_dtype).name
    except (TypeError, ValueError):
        dtype = str(leaf_dtype).lower()
    if dtype == "int64":
        return np.int64
    if dtype == "uint32":
        return np.uint32
    raise ValueError("leaf_dtype must be 'int64' or 'uint32'")


@njit(cache=True)
def _partition_leaf_rows(X_binned, row_order, row_scratch, leaf, leaf_start,
                         n_leaves, split_leaf, new_leaf, feature, threshold):
    """Stable-partition one leaf's rows and move the new leaf to the end."""
    old_start = leaf_start[split_leaf]
    old_end = leaf_start[split_leaf + 1]
    old_total = leaf_start[n_leaves]

    if split_leaf == n_leaves - 1:
        write = old_start
        left_count = 0
        right_count = 0
        for p in range(old_start, old_end):
            i = row_order[p]
            if X_binned[i, feature] <= threshold:
                row_order[write] = i
                leaf[i] = split_leaf
                write += 1
                left_count += 1
            else:
                row_scratch[right_count] = i
                leaf[i] = new_leaf
                right_count += 1
        for q in range(right_count):
            row_order[write + q] = row_scratch[q]
        leaf_start[new_leaf] = old_start + left_count
        leaf_start[new_leaf + 1] = old_end
        return

    write = old_start
    left_count = 0
    right_count = 0
    for p in range(old_start, old_end):
        i = row_order[p]
        if X_binned[i, feature] <= threshold:
            row_order[write] = i
            leaf[i] = split_leaf
            write += 1
            left_count += 1
        else:
            row_scratch[right_count] = i
            leaf[i] = new_leaf
            right_count += 1

    for p in range(old_end, old_total):
        row_order[write] = row_order[p]
        write += 1

    for q in range(right_count):
        row_order[write + q] = row_scratch[q]

    right_start = old_total - right_count
    for l in range(n_leaves - 1, split_leaf, -1):
        leaf_start[l] -= right_count
    leaf_start[split_leaf + 1] = old_start + left_count
    leaf_start[new_leaf] = right_start
    leaf_start[new_leaf + 1] = old_total


@njit(cache=True)
def _partition_leaf_segment_rows(X_binned, row_order, row_scratch, leaf,
                                 leaf_start, leaf_count, split_leaf,
                                 new_leaf, feature, threshold):
    """Stable-partition one leaf's row segment without shifting other leaves."""
    old_start = leaf_start[split_leaf]
    old_count = leaf_count[split_leaf]
    old_end = old_start + old_count

    write = old_start
    left_count = 0
    right_count = 0
    for p in range(old_start, old_end):
        i = row_order[p]
        if X_binned[i, feature] <= threshold:
            row_order[write] = i
            leaf[i] = split_leaf
            write += 1
            left_count += 1
        else:
            row_scratch[right_count] = i
            leaf[i] = new_leaf
            right_count += 1

    for q in range(right_count):
        row_order[write + q] = row_scratch[q]

    leaf_start[split_leaf] = old_start
    leaf_count[split_leaf] = left_count
    leaf_start[new_leaf] = old_start + left_count
    leaf_count[new_leaf] = right_count


@njit(cache=True, parallel=True)
def _refill_leaf_segment_histograms_counts_into(
    X_binned, grad, hess, row_order, leaf_start, leaf_ids, n_leaf_ids,
    hg, hh, hc
):
    """Refill changed-leaf histograms from leaf-contiguous row segments."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for idx in range(n_leaf_ids):
            l = leaf_ids[idx]
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
                hc[f, l, b] = 0.0
            for p in range(leaf_start[l], leaf_start[l + 1]):
                i = row_order[p]
                b = X_binned[i, f]
                hg[f, l, b] += grad[i]
                hi = hess[i]
                hh[f, l, b] += hi
                if hi > 0.0:
                    hc[f, l, b] += 1.0


@njit(cache=True, parallel=True)
def _refill_leaf_segment_histograms_counts_positive_into(
    X_binned, grad, hess, row_order, leaf_start, leaf_ids, n_leaf_ids,
    hg, hh, hc
):
    """Refill changed-leaf histograms when scanned Hessians are positive."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for idx in range(n_leaf_ids):
            l = leaf_ids[idx]
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
                hc[f, l, b] = 0.0
            for p in range(leaf_start[l], leaf_start[l + 1]):
                i = row_order[p]
                b = X_binned[i, f]
                hg[f, l, b] += grad[i]
                hh[f, l, b] += hess[i]
                hc[f, l, b] += 1.0


@njit(cache=True, parallel=True)
def _refill_leaf_segment_histograms_unit_hess_into(
    X_binned, grad, row_order, leaf_start, leaf_ids, n_leaf_ids, hg, hh
):
    """Refill changed unit-Hessian histograms from leaf row segments."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for idx in range(n_leaf_ids):
            l = leaf_ids[idx]
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
            for p in range(leaf_start[l], leaf_start[l + 1]):
                i = row_order[p]
                b = X_binned[i, f]
                hg[f, l, b] += grad[i]
                hh[f, l, b] += 1.0


@njit(cache=True, parallel=True)
def _refill_leaf_segment_histograms_counts_selected_into(
    X_binned, grad, hess, row_order, leaf_start, leaf_ids, n_leaf_ids,
    hg, hh, hc, feature_indices
):
    """Refill changed-leaf histograms for selected feature columns."""
    max_bins = hg.shape[2]
    for jj in prange(feature_indices.shape[0]):
        f = feature_indices[jj]
        for idx in range(n_leaf_ids):
            l = leaf_ids[idx]
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
                hc[f, l, b] = 0.0
            for p in range(leaf_start[l], leaf_start[l + 1]):
                i = row_order[p]
                b = X_binned[i, f]
                hg[f, l, b] += grad[i]
                hi = hess[i]
                hh[f, l, b] += hi
                if hi > 0.0:
                    hc[f, l, b] += 1.0


@njit(cache=True, parallel=True)
def _refill_leaf_segment_histograms_unit_hess_selected_into(
    X_binned, grad, row_order, leaf_start, leaf_ids, n_leaf_ids,
    hg, hh, feature_indices
):
    """Refill changed unit-Hessian histograms for selected feature columns."""
    max_bins = hg.shape[2]
    for jj in prange(feature_indices.shape[0]):
        f = feature_indices[jj]
        for idx in range(n_leaf_ids):
            l = leaf_ids[idx]
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
            for p in range(leaf_start[l], leaf_start[l + 1]):
                i = row_order[p]
                b = X_binned[i, f]
                hg[f, l, b] += grad[i]
                hh[f, l, b] += 1.0


@njit(cache=True, parallel=True)
def _refill_left_subtract_right_unit_hess_into(
    X_binned, grad, row_order, leaf_start, left_leaf, right_leaf, hg, hh
):
    """Refill the left child and derive the right child from parent histograms."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for b in range(max_bins):
            hg[f, right_leaf, b] = hg[f, left_leaf, b]
            hh[f, right_leaf, b] = hh[f, left_leaf, b]
            hg[f, left_leaf, b] = 0.0
            hh[f, left_leaf, b] = 0.0
        for p in range(leaf_start[left_leaf], leaf_start[left_leaf + 1]):
            i = row_order[p]
            b = X_binned[i, f]
            hg[f, left_leaf, b] += grad[i]
            hh[f, left_leaf, b] += 1.0
        for b in range(max_bins):
            hg[f, right_leaf, b] -= hg[f, left_leaf, b]
            hh[f, right_leaf, b] -= hh[f, left_leaf, b]


@njit(cache=True, parallel=True)
def _refill_left_subtract_right_unit_hess_selected_into(
    X_binned, grad, row_order, leaf_start, left_leaf, right_leaf,
    feature_indices, hg, hh
):
    """Refill selected left-child histograms and derive the right child."""
    max_bins = hg.shape[2]
    for jj in prange(feature_indices.shape[0]):
        f = feature_indices[jj]
        for b in range(max_bins):
            hg[f, right_leaf, b] = hg[f, left_leaf, b]
            hh[f, right_leaf, b] = hh[f, left_leaf, b]
            hg[f, left_leaf, b] = 0.0
            hh[f, left_leaf, b] = 0.0
        for p in range(leaf_start[left_leaf], leaf_start[left_leaf + 1]):
            i = row_order[p]
            b = X_binned[i, f]
            hg[f, left_leaf, b] += grad[i]
            hh[f, left_leaf, b] += 1.0
        for b in range(max_bins):
            hg[f, right_leaf, b] -= hg[f, left_leaf, b]
            hh[f, right_leaf, b] -= hh[f, left_leaf, b]


@njit(cache=True, parallel=True)
def _refill_left_subtract_right_counts_into(
    X_binned, grad, hess, row_order, leaf_start, left_leaf, right_leaf,
    hg, hh, hc
):
    """Refill the left child and derive the right child from parent histograms."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for b in range(max_bins):
            hg[f, right_leaf, b] = hg[f, left_leaf, b]
            hh[f, right_leaf, b] = hh[f, left_leaf, b]
            hc[f, right_leaf, b] = hc[f, left_leaf, b]
            hg[f, left_leaf, b] = 0.0
            hh[f, left_leaf, b] = 0.0
            hc[f, left_leaf, b] = 0.0
        for p in range(leaf_start[left_leaf], leaf_start[left_leaf + 1]):
            i = row_order[p]
            b = X_binned[i, f]
            hg[f, left_leaf, b] += grad[i]
            hi = hess[i]
            hh[f, left_leaf, b] += hi
            if hi > 0.0:
                hc[f, left_leaf, b] += 1.0
        for b in range(max_bins):
            hg[f, right_leaf, b] -= hg[f, left_leaf, b]
            hh[f, right_leaf, b] -= hh[f, left_leaf, b]
            hc[f, right_leaf, b] -= hc[f, left_leaf, b]


@njit(cache=True, parallel=True)
def _refill_left_subtract_right_counts_positive_into(
    X_binned, grad, hess, row_order, leaf_start, left_leaf, right_leaf,
    hg, hh, hc
):
    """Refill left child and subtract it when Hessians are positive."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for b in range(max_bins):
            hg[f, right_leaf, b] = hg[f, left_leaf, b]
            hh[f, right_leaf, b] = hh[f, left_leaf, b]
            hc[f, right_leaf, b] = hc[f, left_leaf, b]
            hg[f, left_leaf, b] = 0.0
            hh[f, left_leaf, b] = 0.0
            hc[f, left_leaf, b] = 0.0
        for p in range(leaf_start[left_leaf], leaf_start[left_leaf + 1]):
            i = row_order[p]
            b = X_binned[i, f]
            hg[f, left_leaf, b] += grad[i]
            hh[f, left_leaf, b] += hess[i]
            hc[f, left_leaf, b] += 1.0
        for b in range(max_bins):
            hg[f, right_leaf, b] -= hg[f, left_leaf, b]
            hh[f, right_leaf, b] -= hh[f, left_leaf, b]
            hc[f, right_leaf, b] -= hc[f, left_leaf, b]


@njit(cache=True, parallel=True)
def _refill_left_subtract_right_counts_selected_into(
    X_binned, grad, hess, row_order, leaf_start, left_leaf, right_leaf,
    feature_indices, hg, hh, hc
):
    """Refill selected left-child histograms and derive the right child."""
    max_bins = hg.shape[2]
    for jj in prange(feature_indices.shape[0]):
        f = feature_indices[jj]
        for b in range(max_bins):
            hg[f, right_leaf, b] = hg[f, left_leaf, b]
            hh[f, right_leaf, b] = hh[f, left_leaf, b]
            hc[f, right_leaf, b] = hc[f, left_leaf, b]
            hg[f, left_leaf, b] = 0.0
            hh[f, left_leaf, b] = 0.0
            hc[f, left_leaf, b] = 0.0
        for p in range(leaf_start[left_leaf], leaf_start[left_leaf + 1]):
            i = row_order[p]
            b = X_binned[i, f]
            hg[f, left_leaf, b] += grad[i]
            hi = hess[i]
            hh[f, left_leaf, b] += hi
            if hi > 0.0:
                hc[f, left_leaf, b] += 1.0
        for b in range(max_bins):
            hg[f, right_leaf, b] -= hg[f, left_leaf, b]
            hh[f, right_leaf, b] -= hh[f, left_leaf, b]
            hc[f, right_leaf, b] -= hc[f, left_leaf, b]


@njit(cache=True, parallel=True)
def _refill_right_subtract_left_unit_hess_into(
    X_binned, grad, row_order, leaf_start, left_leaf, right_leaf, hg, hh
):
    """Refill the right child and subtract it from cached parent histograms."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for b in range(max_bins):
            hg[f, right_leaf, b] = 0.0
            hh[f, right_leaf, b] = 0.0
        for p in range(leaf_start[right_leaf], leaf_start[right_leaf + 1]):
            i = row_order[p]
            b = X_binned[i, f]
            hg[f, right_leaf, b] += grad[i]
            hh[f, right_leaf, b] += 1.0
        for b in range(max_bins):
            hg[f, left_leaf, b] -= hg[f, right_leaf, b]
            hh[f, left_leaf, b] -= hh[f, right_leaf, b]


@njit(cache=True, parallel=True)
def _refill_right_subtract_left_unit_hess_selected_into(
    X_binned, grad, row_order, leaf_start, left_leaf, right_leaf,
    feature_indices, hg, hh
):
    """Refill and subtract selected unit-Hessian feature histograms."""
    max_bins = hg.shape[2]
    for jj in prange(feature_indices.shape[0]):
        f = feature_indices[jj]
        for b in range(max_bins):
            hg[f, right_leaf, b] = 0.0
            hh[f, right_leaf, b] = 0.0
        for p in range(leaf_start[right_leaf], leaf_start[right_leaf + 1]):
            i = row_order[p]
            b = X_binned[i, f]
            hg[f, right_leaf, b] += grad[i]
            hh[f, right_leaf, b] += 1.0
        for b in range(max_bins):
            hg[f, left_leaf, b] -= hg[f, right_leaf, b]
            hh[f, left_leaf, b] -= hh[f, right_leaf, b]


@njit(cache=True, parallel=True)
def _refill_right_subtract_left_counts_into(
    X_binned, grad, hess, row_order, leaf_start, left_leaf, right_leaf,
    hg, hh, hc
):
    """Refill the right child and subtract it from cached parent histograms."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for b in range(max_bins):
            hg[f, right_leaf, b] = 0.0
            hh[f, right_leaf, b] = 0.0
            hc[f, right_leaf, b] = 0.0
        for p in range(leaf_start[right_leaf], leaf_start[right_leaf + 1]):
            i = row_order[p]
            b = X_binned[i, f]
            hg[f, right_leaf, b] += grad[i]
            hi = hess[i]
            hh[f, right_leaf, b] += hi
            if hi > 0.0:
                hc[f, right_leaf, b] += 1.0
        for b in range(max_bins):
            hg[f, left_leaf, b] -= hg[f, right_leaf, b]
            hh[f, left_leaf, b] -= hh[f, right_leaf, b]
            hc[f, left_leaf, b] -= hc[f, right_leaf, b]


@njit(cache=True, parallel=True)
def _refill_right_subtract_left_counts_positive_into(
    X_binned, grad, hess, row_order, leaf_start, left_leaf, right_leaf,
    hg, hh, hc
):
    """Refill right child and subtract it when Hessians are positive."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for b in range(max_bins):
            hg[f, right_leaf, b] = 0.0
            hh[f, right_leaf, b] = 0.0
            hc[f, right_leaf, b] = 0.0
        for p in range(leaf_start[right_leaf], leaf_start[right_leaf + 1]):
            i = row_order[p]
            b = X_binned[i, f]
            hg[f, right_leaf, b] += grad[i]
            hh[f, right_leaf, b] += hess[i]
            hc[f, right_leaf, b] += 1.0
        for b in range(max_bins):
            hg[f, left_leaf, b] -= hg[f, right_leaf, b]
            hh[f, left_leaf, b] -= hh[f, right_leaf, b]
            hc[f, left_leaf, b] -= hc[f, right_leaf, b]


@njit(cache=True, parallel=True)
def _refill_right_subtract_left_counts_selected_into(
    X_binned, grad, hess, row_order, leaf_start, left_leaf, right_leaf,
    feature_indices, hg, hh, hc
):
    """Refill and subtract selected nonconstant-Hessian feature histograms."""
    max_bins = hg.shape[2]
    for jj in prange(feature_indices.shape[0]):
        f = feature_indices[jj]
        for b in range(max_bins):
            hg[f, right_leaf, b] = 0.0
            hh[f, right_leaf, b] = 0.0
            hc[f, right_leaf, b] = 0.0
        for p in range(leaf_start[right_leaf], leaf_start[right_leaf + 1]):
            i = row_order[p]
            b = X_binned[i, f]
            hg[f, right_leaf, b] += grad[i]
            hi = hess[i]
            hh[f, right_leaf, b] += hi
            if hi > 0.0:
                hc[f, right_leaf, b] += 1.0
        for b in range(max_bins):
            hg[f, left_leaf, b] -= hg[f, right_leaf, b]
            hh[f, left_leaf, b] -= hh[f, right_leaf, b]
            hc[f, left_leaf, b] -= hc[f, right_leaf, b]


@njit(cache=True, parallel=True)
def _refill_right_subtract_left_unit_hess_by_count_into(
    X_binned, grad, row_order, leaf_start, leaf_count,
    left_leaf, right_leaf, hg, hh
):
    """Refill right child from a segmented row layout and subtract from parent."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    right_start = leaf_start[right_leaf]
    right_end = right_start + leaf_count[right_leaf]
    for f in prange(n_features):
        for b in range(max_bins):
            hg[f, right_leaf, b] = 0.0
            hh[f, right_leaf, b] = 0.0
        for p in range(right_start, right_end):
            i = row_order[p]
            b = X_binned[i, f]
            hg[f, right_leaf, b] += grad[i]
            hh[f, right_leaf, b] += 1.0
        for b in range(max_bins):
            hg[f, left_leaf, b] -= hg[f, right_leaf, b]
            hh[f, left_leaf, b] -= hh[f, right_leaf, b]


@njit(cache=True, parallel=True)
def _refill_left_subtract_right_unit_hess_by_count_into(
    X_binned, grad, row_order, leaf_start, leaf_count,
    left_leaf, right_leaf, hg, hh
):
    """Refill left child from segmented rows and derive right child."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    left_start = leaf_start[left_leaf]
    left_end = left_start + leaf_count[left_leaf]
    for f in prange(n_features):
        for b in range(max_bins):
            hg[f, right_leaf, b] = hg[f, left_leaf, b]
            hh[f, right_leaf, b] = hh[f, left_leaf, b]
            hg[f, left_leaf, b] = 0.0
            hh[f, left_leaf, b] = 0.0
        for p in range(left_start, left_end):
            i = row_order[p]
            b = X_binned[i, f]
            hg[f, left_leaf, b] += grad[i]
            hh[f, left_leaf, b] += 1.0
        for b in range(max_bins):
            hg[f, right_leaf, b] -= hg[f, left_leaf, b]
            hh[f, right_leaf, b] -= hh[f, left_leaf, b]


@njit(cache=True, parallel=True)
def _refill_right_subtract_left_counts_by_count_into(
    X_binned, grad, hess, row_order, leaf_start, leaf_count,
    left_leaf, right_leaf, hg, hh, hc
):
    """Refill right child from segmented rows and subtract from parent."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    right_start = leaf_start[right_leaf]
    right_end = right_start + leaf_count[right_leaf]
    for f in prange(n_features):
        for b in range(max_bins):
            hg[f, right_leaf, b] = 0.0
            hh[f, right_leaf, b] = 0.0
            hc[f, right_leaf, b] = 0.0
        for p in range(right_start, right_end):
            i = row_order[p]
            b = X_binned[i, f]
            hg[f, right_leaf, b] += grad[i]
            hi = hess[i]
            hh[f, right_leaf, b] += hi
            if hi > 0.0:
                hc[f, right_leaf, b] += 1.0
        for b in range(max_bins):
            hg[f, left_leaf, b] -= hg[f, right_leaf, b]
            hh[f, left_leaf, b] -= hh[f, right_leaf, b]
            hc[f, left_leaf, b] -= hc[f, right_leaf, b]


@njit(cache=True, parallel=True)
def _refill_right_subtract_left_counts_positive_by_count_into(
    X_binned, grad, hess, row_order, leaf_start, leaf_count,
    left_leaf, right_leaf, hg, hh, hc
):
    """Refill right child from segmented rows when Hessians are positive."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    right_start = leaf_start[right_leaf]
    right_end = right_start + leaf_count[right_leaf]
    for f in prange(n_features):
        for b in range(max_bins):
            hg[f, right_leaf, b] = 0.0
            hh[f, right_leaf, b] = 0.0
            hc[f, right_leaf, b] = 0.0
        for p in range(right_start, right_end):
            i = row_order[p]
            b = X_binned[i, f]
            hg[f, right_leaf, b] += grad[i]
            hh[f, right_leaf, b] += hess[i]
            hc[f, right_leaf, b] += 1.0
        for b in range(max_bins):
            hg[f, left_leaf, b] -= hg[f, right_leaf, b]
            hh[f, left_leaf, b] -= hh[f, right_leaf, b]
            hc[f, left_leaf, b] -= hc[f, right_leaf, b]


@njit(cache=True, parallel=True)
def _refill_left_subtract_right_counts_by_count_into(
    X_binned, grad, hess, row_order, leaf_start, leaf_count,
    left_leaf, right_leaf, hg, hh, hc
):
    """Refill left child from segmented rows and derive right child."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    left_start = leaf_start[left_leaf]
    left_end = left_start + leaf_count[left_leaf]
    for f in prange(n_features):
        for b in range(max_bins):
            hg[f, right_leaf, b] = hg[f, left_leaf, b]
            hh[f, right_leaf, b] = hh[f, left_leaf, b]
            hc[f, right_leaf, b] = hc[f, left_leaf, b]
            hg[f, left_leaf, b] = 0.0
            hh[f, left_leaf, b] = 0.0
            hc[f, left_leaf, b] = 0.0
        for p in range(left_start, left_end):
            i = row_order[p]
            b = X_binned[i, f]
            hg[f, left_leaf, b] += grad[i]
            hi = hess[i]
            hh[f, left_leaf, b] += hi
            if hi > 0.0:
                hc[f, left_leaf, b] += 1.0
        for b in range(max_bins):
            hg[f, right_leaf, b] -= hg[f, left_leaf, b]
            hh[f, right_leaf, b] -= hh[f, left_leaf, b]
            hc[f, right_leaf, b] -= hc[f, left_leaf, b]


@njit(cache=True, parallel=True)
def _refill_left_subtract_right_counts_positive_by_count_into(
    X_binned, grad, hess, row_order, leaf_start, leaf_count,
    left_leaf, right_leaf, hg, hh, hc
):
    """Refill left child from segmented rows when Hessians are positive."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    left_start = leaf_start[left_leaf]
    left_end = left_start + leaf_count[left_leaf]
    for f in prange(n_features):
        for b in range(max_bins):
            hg[f, right_leaf, b] = hg[f, left_leaf, b]
            hh[f, right_leaf, b] = hh[f, left_leaf, b]
            hc[f, right_leaf, b] = hc[f, left_leaf, b]
            hg[f, left_leaf, b] = 0.0
            hh[f, left_leaf, b] = 0.0
            hc[f, left_leaf, b] = 0.0
        for p in range(left_start, left_end):
            i = row_order[p]
            b = X_binned[i, f]
            hg[f, left_leaf, b] += grad[i]
            hh[f, left_leaf, b] += hess[i]
            hc[f, left_leaf, b] += 1.0
        for b in range(max_bins):
            hg[f, right_leaf, b] -= hg[f, left_leaf, b]
            hh[f, right_leaf, b] -= hh[f, left_leaf, b]
            hc[f, right_leaf, b] -= hc[f, left_leaf, b]


@njit(cache=True, parallel=True)
def _refill_right_subtract_left_counts_positive_score_full_features_into(
    X_binned, grad, hess, row_order, leaf_start, left_leaf, right_leaf,
    hg, hh, hc, n_bins_per_feature, l2, min_child_weight, min_child_samples,
    feature_gain, feature_thr
):
    """Refill right child, subtract left, and score both changed leaves.

    Narrow full-row/full-feature positive-Hessian lane for scalar leafwise
    trees. It preserves the existing parent-histogram contract: the parent
    histogram starts in ``left_leaf`` and the right child is rebuilt from its
    row segment before the left child is derived by subtraction.
    """
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    right_start = leaf_start[right_leaf]
    right_end = leaf_start[right_leaf + 1]
    left_count = float(leaf_start[left_leaf + 1] - leaf_start[left_leaf])
    right_count = float(right_end - right_start)

    for f in prange(n_features):
        for b in range(max_bins):
            hg[f, right_leaf, b] = 0.0
            hh[f, right_leaf, b] = 0.0
            hc[f, right_leaf, b] = 0.0
        for p in range(right_start, right_end):
            i = row_order[p]
            b = X_binned[i, f]
            hg[f, right_leaf, b] += grad[i]
            hh[f, right_leaf, b] += hess[i]
            hc[f, right_leaf, b] += 1.0
        for b in range(max_bins):
            hg[f, left_leaf, b] -= hg[f, right_leaf, b]
            hh[f, left_leaf, b] -= hh[f, right_leaf, b]
            hc[f, left_leaf, b] -= hc[f, right_leaf, b]

        nb = n_bins_per_feature[f]
        for leaf_pos in range(2):
            if leaf_pos == 0:
                l = left_leaf
                Ct = left_count
            else:
                l = right_leaf
                Ct = right_count
            best_t = -1
            best_gain = -np.inf
            if Ct > 0.0:
                Gt = 0.0
                Ht = 0.0
                for b in range(nb):
                    Gt += hg[f, l, b]
                    Ht += hh[f, l, b]
                parent_denom = Ht + l2
                if Ht > 0.0 and parent_denom > 0.0:
                    parent_gain = Gt * Gt / parent_denom
                    GL = 0.0
                    HL = 0.0
                    CL = 0.0
                    for t in range(nb - 1):
                        GL += hg[f, l, t]
                        HL += hh[f, l, t]
                        CL += hc[f, l, t]
                        HR = Ht - HL
                        CR = Ct - CL
                        if (
                            HL < min_child_weight
                            or HR < min_child_weight
                            or CL < min_child_samples
                            or CR < min_child_samples
                        ):
                            continue
                        left_denom = HL + l2
                        right_denom = HR + l2
                        if left_denom <= 0.0 or right_denom <= 0.0:
                            continue
                        GR = Gt - GL
                        gain = (
                            GL * GL / left_denom
                            + GR * GR / right_denom
                            - parent_gain
                        )
                        if gain > best_gain:
                            best_gain = gain
                            best_t = t
            feature_gain[f, l] = best_gain
            feature_thr[f, l] = best_t


@njit(cache=True)
def _reduce_feature_splits_for_leaf_ids(
    feature_gain, feature_thr, leaf_ids, n_leaf_ids, out_feat, out_thr, out_gain
):
    """Reduce per-feature split candidates into one best split per leaf."""
    n_features = feature_gain.shape[0]
    for idx in range(n_leaf_ids):
        l = leaf_ids[idx]
        best_f = -1
        best_t = -1
        best_gain = -np.inf
        for f in range(n_features):
            gain = feature_gain[f, l]
            if gain > best_gain:
                best_gain = gain
                best_f = f
                best_t = int(feature_thr[f, l])
        out_feat[l] = best_f
        out_thr[l] = best_t
        out_gain[l] = best_gain


@njit(cache=True)
def _subtract_right_child_histograms_into_left_serial(
    left_leaf, right_leaf, hg, hh, hc
):
    """Replace parent histograms in left_leaf with parent - right child."""
    n_features = hg.shape[0]
    max_bins = hg.shape[2]
    for f in range(n_features):
        for b in range(max_bins):
            hg[f, left_leaf, b] -= hg[f, right_leaf, b]
            hh[f, left_leaf, b] -= hh[f, right_leaf, b]
            hc[f, left_leaf, b] -= hc[f, right_leaf, b]


@njit(cache=True)
def _subtract_right_child_histograms_selected_into_left_serial(
    left_leaf, right_leaf, feature_indices, hg, hh, hc
):
    """Subtract right-child histograms from parent for selected features."""
    max_bins = hg.shape[2]
    for jj in range(feature_indices.shape[0]):
        f = feature_indices[jj]
        for b in range(max_bins):
            hg[f, left_leaf, b] -= hg[f, right_leaf, b]
            hh[f, left_leaf, b] -= hh[f, right_leaf, b]
            hc[f, left_leaf, b] -= hc[f, right_leaf, b]


@njit(cache=True)
def _subtract_right_child_unit_hess_histograms_into_left_serial(
    left_leaf, right_leaf, hg, hh
):
    """Subtract right-child unit-Hessian histograms from parent."""
    n_features = hg.shape[0]
    max_bins = hg.shape[2]
    for f in range(n_features):
        for b in range(max_bins):
            hg[f, left_leaf, b] -= hg[f, right_leaf, b]
            hh[f, left_leaf, b] -= hh[f, right_leaf, b]


@njit(cache=True)
def _subtract_right_child_unit_hess_histograms_selected_into_left_serial(
    left_leaf, right_leaf, feature_indices, hg, hh
):
    """Subtract selected right-child unit-Hessian histograms from parent."""
    max_bins = hg.shape[2]
    for jj in range(feature_indices.shape[0]):
        f = feature_indices[jj]
        for b in range(max_bins):
            hg[f, left_leaf, b] -= hg[f, right_leaf, b]
            hh[f, left_leaf, b] -= hh[f, right_leaf, b]


@njit(cache=True, parallel=True)
def _subtract_right_child_histograms_into_left(left_leaf, right_leaf, hg, hh, hc):
    """Replace parent histograms in left_leaf with parent - right child."""
    n_features = hg.shape[0]
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for b in range(max_bins):
            hg[f, left_leaf, b] -= hg[f, right_leaf, b]
            hh[f, left_leaf, b] -= hh[f, right_leaf, b]
            hc[f, left_leaf, b] -= hc[f, right_leaf, b]


@njit(cache=True, parallel=True)
def _subtract_right_child_unit_hess_histograms_into_left(
    left_leaf, right_leaf, hg, hh
):
    """Replace parent unit-Hessian histograms with parent - right child."""
    n_features = hg.shape[0]
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for b in range(max_bins):
            hg[f, left_leaf, b] -= hg[f, right_leaf, b]
            hh[f, left_leaf, b] -= hh[f, right_leaf, b]


@njit(cache=True, parallel=True)
def _subtract_right_child_unit_hess_histograms_selected_into_left(
    left_leaf, right_leaf, feature_indices, hg, hh
):
    """Subtract selected unit-Hessian right-child histograms from parent."""
    max_bins = hg.shape[2]
    for jj in prange(feature_indices.shape[0]):
        f = feature_indices[jj]
        for b in range(max_bins):
            hg[f, left_leaf, b] -= hg[f, right_leaf, b]
            hh[f, left_leaf, b] -= hh[f, right_leaf, b]


@njit(cache=True, parallel=True)
def _subtract_right_child_histograms_selected_into_left(
    left_leaf, right_leaf, feature_indices, hg, hh, hc
):
    """Subtract right-child histograms from parent for selected features."""
    max_bins = hg.shape[2]
    for jj in prange(feature_indices.shape[0]):
        f = feature_indices[jj]
        for b in range(max_bins):
            hg[f, left_leaf, b] -= hg[f, right_leaf, b]
            hh[f, left_leaf, b] -= hh[f, right_leaf, b]
            hc[f, left_leaf, b] -= hc[f, right_leaf, b]


@njit(cache=True)
def _build_counts_into_serial(X_binned, hess, leaf, n_leaves, hc):
    """Single-thread positive-weight row-count histograms."""
    n_samples, n_features = X_binned.shape
    max_bins = hc.shape[2]
    for f in range(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                hc[f, l, b] = 0.0

    for i in range(n_samples):
        if hess[i] > 0.0:
            l = leaf[i]
            for f in range(n_features):
                b = X_binned[i, f]
                hc[f, l, b] += 1.0


@njit(cache=True)
def _build_histograms_counts_into_serial(X_binned, grad, hess, leaf,
                                         n_leaves, hg, hh, hc):
    """Single-thread gradient/hessian/count histogram fill."""
    n_samples, n_features = X_binned.shape
    max_bins = hg.shape[2]
    for f in range(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
                hc[f, l, b] = 0.0

    for i in range(n_samples):
        l = leaf[i]
        gi = grad[i]
        hi = hess[i]
        positive = hi > 0.0
        for f in range(n_features):
            b = X_binned[i, f]
            hg[f, l, b] += gi
            hh[f, l, b] += hi
            if positive:
                hc[f, l, b] += 1.0


@njit(cache=True)
def _build_histograms_counts_rows_into_serial(X_binned, grad, hess, leaf,
                                              n_leaves, hg, hh, hc,
                                              row_indices):
    """Single-thread selected-row gradient/hessian/count histogram fill."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in range(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
                hc[f, l, b] = 0.0

    for p in range(row_indices.shape[0]):
        i = row_indices[p]
        l = leaf[i]
        gi = grad[i]
        hi = hess[i]
        positive = hi > 0.0
        for f in range(n_features):
            b = X_binned[i, f]
            hg[f, l, b] += gi
            hh[f, l, b] += hi
            if positive:
                hc[f, l, b] += 1.0


@njit(cache=True)
def _build_histograms_counts_selected_into_serial(X_binned, grad, hess, leaf,
                                                  n_leaves, hg, hh, hc,
                                                  feature_indices):
    """Single-thread selected-feature gradient/hessian/count histogram fill."""
    n_samples = X_binned.shape[0]
    max_bins = hg.shape[2]
    for jj in range(feature_indices.shape[0]):
        f = feature_indices[jj]
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
                hc[f, l, b] = 0.0

    for i in range(n_samples):
        l = leaf[i]
        gi = grad[i]
        hi = hess[i]
        positive = hi > 0.0
        for jj in range(feature_indices.shape[0]):
            f = feature_indices[jj]
            b = X_binned[i, f]
            hg[f, l, b] += gi
            hh[f, l, b] += hi
            if positive:
                hc[f, l, b] += 1.0


@njit(cache=True)
def _build_histograms_counts_selected_rows_into_serial(
    X_binned, grad, hess, leaf, n_leaves, hg, hh, hc,
    feature_indices, row_indices
):
    """Single-thread selected-row/feature grad/hess/count histogram fill."""
    max_bins = hg.shape[2]
    for jj in range(feature_indices.shape[0]):
        f = feature_indices[jj]
        for l in range(n_leaves):
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
                hc[f, l, b] = 0.0

    for p in range(row_indices.shape[0]):
        i = row_indices[p]
        l = leaf[i]
        gi = grad[i]
        hi = hess[i]
        positive = hi > 0.0
        for jj in range(feature_indices.shape[0]):
            f = feature_indices[jj]
            b = X_binned[i, f]
            hg[f, l, b] += gi
            hh[f, l, b] += hi
            if positive:
                hc[f, l, b] += 1.0


@njit(cache=True)
def _refill_leaf_segment_histograms_counts_into_serial(
    X_binned, grad, hess, row_order, leaf_start, leaf_ids, n_leaf_ids,
    hg, hh, hc
):
    """Single-thread changed-leaf histogram refill from row segments."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in range(n_features):
        for idx in range(n_leaf_ids):
            l = leaf_ids[idx]
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
                hc[f, l, b] = 0.0

    for idx in range(n_leaf_ids):
        l = leaf_ids[idx]
        for p in range(leaf_start[l], leaf_start[l + 1]):
            i = row_order[p]
            gi = grad[i]
            hi = hess[i]
            positive = hi > 0.0
            for f in range(n_features):
                b = X_binned[i, f]
                hg[f, l, b] += gi
                hh[f, l, b] += hi
                if positive:
                    hc[f, l, b] += 1.0


@njit(cache=True)
def _refill_leaf_segment_histograms_unit_hess_into_serial(
    X_binned, grad, row_order, leaf_start, leaf_ids, n_leaf_ids, hg, hh
):
    """Single-thread unit-Hessian histogram refill from row segments."""
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in range(n_features):
        for idx in range(n_leaf_ids):
            l = leaf_ids[idx]
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0

    for idx in range(n_leaf_ids):
        l = leaf_ids[idx]
        for p in range(leaf_start[l], leaf_start[l + 1]):
            i = row_order[p]
            gi = grad[i]
            for f in range(n_features):
                b = X_binned[i, f]
                hg[f, l, b] += gi
                hh[f, l, b] += 1.0


@njit(cache=True)
def _refill_leaf_segment_histograms_counts_selected_into_serial(
    X_binned, grad, hess, row_order, leaf_start, leaf_ids, n_leaf_ids,
    hg, hh, hc, feature_indices
):
    """Single-thread changed-leaf histogram refill for selected columns."""
    max_bins = hg.shape[2]
    for jj in range(feature_indices.shape[0]):
        f = feature_indices[jj]
        for idx in range(n_leaf_ids):
            l = leaf_ids[idx]
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0
                hc[f, l, b] = 0.0

    for idx in range(n_leaf_ids):
        l = leaf_ids[idx]
        for p in range(leaf_start[l], leaf_start[l + 1]):
            i = row_order[p]
            gi = grad[i]
            hi = hess[i]
            positive = hi > 0.0
            for jj in range(feature_indices.shape[0]):
                f = feature_indices[jj]
                b = X_binned[i, f]
                hg[f, l, b] += gi
                hh[f, l, b] += hi
                if positive:
                    hc[f, l, b] += 1.0


@njit(cache=True)
def _refill_leaf_segment_histograms_unit_hess_selected_into_serial(
    X_binned, grad, row_order, leaf_start, leaf_ids, n_leaf_ids,
    hg, hh, feature_indices
):
    """Single-thread unit-Hessian refill for selected feature columns."""
    max_bins = hg.shape[2]
    for jj in range(feature_indices.shape[0]):
        f = feature_indices[jj]
        for idx in range(n_leaf_ids):
            l = leaf_ids[idx]
            for b in range(max_bins):
                hg[f, l, b] = 0.0
                hh[f, l, b] = 0.0

    for idx in range(n_leaf_ids):
        l = leaf_ids[idx]
        for p in range(leaf_start[l], leaf_start[l + 1]):
            i = row_order[p]
            gi = grad[i]
            for jj in range(feature_indices.shape[0]):
                f = feature_indices[jj]
                b = X_binned[i, f]
                hg[f, l, b] += gi
                hh[f, l, b] += 1.0


@njit(cache=True)
def _build_counts_rows_into_serial(X_binned, hess, leaf, n_leaves, hc,
                                   row_indices):
    """Single-thread selected-row positive-weight row-count histograms."""
    n_features = X_binned.shape[1]
    max_bins = hc.shape[2]
    for f in range(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                hc[f, l, b] = 0.0

    for p in range(row_indices.shape[0]):
        i = row_indices[p]
        if hess[i] > 0.0:
            l = leaf[i]
            for f in range(n_features):
                b = X_binned[i, f]
                hc[f, l, b] += 1.0


@njit(cache=True, parallel=True)
def _best_split(hg, hh, n_bins_per_feature, l2, feat_mask, min_child_weight,
                n_leaves, scratch_Gt, scratch_Ht, scratch_GL, scratch_HL,
                scratch_parent):
    """Find the (feature, threshold) with the highest total gain.

    hg/hh may be oversized buffers (shape max_leaves); `n_leaves` says how many
    leaf rows are actually active at this level, so we only read those.

    For a candidate threshold t, bins <= t go left and bins > t go right, the
    same way in every current leaf. Gain is summed across leaves. Features with
    feat_mask[f] == 0 are skipped (column subsampling).
    
    Min-child-weight legality: because an oblivious split is applied to every
    active leaf, a threshold is legal only if it leaves at least
    `min_child_weight` hessian mass on both sides of every non-empty leaf.
    """
    n_features = n_bins_per_feature.shape[0]
    max_bins = hg.shape[2]
    feat_gain = np.full(n_features, -np.inf)
    feat_thr = np.full(n_features, -1, dtype=np.int64)

    for f in prange(n_features):
        if feat_mask[f] == 0:
            continue
        nb = n_bins_per_feature[f]
        # Totals per leaf for this feature (same regardless of threshold).
        for l in range(n_leaves):
            scratch_Gt[f, l] = 0.0
            scratch_Ht[f, l] = 0.0
            for b in range(nb):
                scratch_Gt[f, l] += hg[f, l, b]
                scratch_Ht[f, l] += hh[f, l, b]
        for l in range(n_leaves):
            scratch_GL[f, l] = 0.0
            scratch_HL[f, l] = 0.0
            parent_denom = scratch_Ht[f, l] + l2
            if scratch_Ht[f, l] > 0.0 and parent_denom > 0.0:
                scratch_parent[f, l] = (
                    scratch_Gt[f, l] * scratch_Gt[f, l] / parent_denom
                )
            else:
                scratch_parent[f, l] = 0.0

        best_g = -np.inf
        best_t = -1
        # Threshold t means "left = bins [0..t]". Last bin can't be a threshold.
        for t in range(nb - 1):
            gain = 0.0
            legal = True
            any_nonempty = False
            
            for l in range(n_leaves):
                # Pass 1: Advance the running prefix sums for every leaf regardless
                # of legality, so GL/HL carry correctly into the next threshold.
                scratch_GL[f, l] += hg[f, l, t]
                scratch_HL[f, l] += hh[f, l, t]
                
                if scratch_Ht[f, l] > 0.0:
                    any_nonempty = True
                    hl = scratch_HL[f, l]
                    hr = scratch_Ht[f, l] - hl
                    left_denom = hl + l2
                    right_denom = hr + l2
                    parent_denom = scratch_Ht[f, l] + l2
                    if (
                        hl < min_child_weight
                        or hr < min_child_weight
                        or left_denom <= 0.0
                        or right_denom <= 0.0
                        or parent_denom <= 0.0
                    ):
                        legal = False
                    else:
                        gl = scratch_GL[f, l]
                        gr = scratch_Gt[f, l] - gl
                        gain += (
                            gl * gl / left_denom
                            + gr * gr / right_denom
                            - scratch_parent[f, l]
                        )
            
            if legal and any_nonempty and gain > best_g:
                best_g = gain
                best_t = t
                
        feat_gain[f] = best_g
        feat_thr[f] = best_t

    best_f = 0
    best_gain = -np.inf
    for f in range(n_features):
        if feat_gain[f] > best_gain:
            best_gain = feat_gain[f]
            best_f = f
    return best_f, feat_thr[best_f], best_gain


@njit(cache=True)
def _best_split_serial(hg, hh, n_bins_per_feature, l2, feat_mask,
                       min_child_weight, n_leaves):
    """Single-thread split search without per-feature temporary allocations."""
    n_features = n_bins_per_feature.shape[0]
    Gt = np.empty(n_leaves)
    Ht = np.empty(n_leaves)
    GL = np.empty(n_leaves)
    HL = np.empty(n_leaves)
    parent_gain = np.empty(n_leaves)

    best_f = 0
    best_t = -1
    best_gain = -np.inf

    for f in range(n_features):
        if feat_mask[f] == 0:
            continue
        nb = n_bins_per_feature[f]
        for l in range(n_leaves):
            gt = 0.0
            ht = 0.0
            for b in range(nb):
                gt += hg[f, l, b]
                ht += hh[f, l, b]
            Gt[l] = gt
            Ht[l] = ht
            GL[l] = 0.0
            HL[l] = 0.0
            parent_denom = ht + l2
            if ht > 0.0 and parent_denom > 0.0:
                parent_gain[l] = gt * gt / parent_denom
            else:
                parent_gain[l] = 0.0

        feat_best_gain = -np.inf
        feat_best_t = -1
        for t in range(nb - 1):
            gain = 0.0
            legal = True
            any_nonempty = False
            for l in range(n_leaves):
                GL[l] += hg[f, l, t]
                HL[l] += hh[f, l, t]

                if Ht[l] > 0.0:
                    any_nonempty = True
                    hl = HL[l]
                    hr = Ht[l] - hl
                    left_denom = hl + l2
                    right_denom = hr + l2
                    parent_denom = Ht[l] + l2
                    if (
                        hl < min_child_weight
                        or hr < min_child_weight
                        or left_denom <= 0.0
                        or right_denom <= 0.0
                        or parent_denom <= 0.0
                    ):
                        legal = False
                    else:
                        gl = GL[l]
                        gr = Gt[l] - gl
                        gain += (
                            gl * gl / left_denom
                            + gr * gr / right_denom
                            - parent_gain[l]
                        )
            if legal and any_nonempty and gain > feat_best_gain:
                feat_best_gain = gain
                feat_best_t = t

        if feat_best_gain > best_gain:
            best_gain = feat_best_gain
            best_f = f
            best_t = feat_best_t

    return best_f, best_t, best_gain


@njit(cache=True)
def _best_shared_split_counts(hg, hh, hc, n_bins_per_feature, l2, feat_mask,
                              min_child_weight, min_child_samples, n_leaves):
    """Best shared split across active leaves with Hessian and count legality."""
    n_features = n_bins_per_feature.shape[0]
    max_bins = hg.shape[2]
    Gt = np.empty(n_leaves)
    Ht = np.empty(n_leaves)
    Ct = np.empty(n_leaves)
    GL = np.empty(n_leaves)
    HL = np.empty(n_leaves)
    CL = np.empty(n_leaves)
    parent_gain = np.empty(n_leaves)

    best_f = 0
    best_t = -1
    best_gain = -np.inf

    for f in range(n_features):
        if feat_mask[f] == 0:
            continue
        nb = n_bins_per_feature[f]
        for l in range(n_leaves):
            gt = 0.0
            ht = 0.0
            ct = 0.0
            for b in range(nb):
                gt += hg[f, l, b]
                ht += hh[f, l, b]
                ct += hc[f, l, b]
            Gt[l] = gt
            Ht[l] = ht
            Ct[l] = ct
            GL[l] = 0.0
            HL[l] = 0.0
            CL[l] = 0.0
            parent_denom = ht + l2
            if ht > 0.0 and parent_denom > 0.0:
                parent_gain[l] = gt * gt / parent_denom
            else:
                parent_gain[l] = 0.0

        feat_best_gain = -np.inf
        feat_best_t = -1
        for t in range(nb - 1):
            gain = 0.0
            legal = True
            any_nonempty = False
            for l in range(n_leaves):
                GL[l] += hg[f, l, t]
                HL[l] += hh[f, l, t]
                CL[l] += hc[f, l, t]

                if Ct[l] > 0.0:
                    any_nonempty = True
                    hl = HL[l]
                    hr = Ht[l] - hl
                    cl = CL[l]
                    cr = Ct[l] - cl
                    left_denom = hl + l2
                    right_denom = hr + l2
                    parent_denom = Ht[l] + l2
                    if (
                        hl < min_child_weight
                        or hr < min_child_weight
                        or cl < min_child_samples
                        or cr < min_child_samples
                        or left_denom <= 0.0
                        or right_denom <= 0.0
                        or parent_denom <= 0.0
                    ):
                        legal = False
                    else:
                        gl = GL[l]
                        gr = Gt[l] - gl
                        gain += (
                            gl * gl / left_denom
                            + gr * gr / right_denom
                            - parent_gain[l]
                        )
            if legal and any_nonempty and gain > feat_best_gain:
                feat_best_gain = gain
                feat_best_t = t

        if feat_best_gain > best_gain:
            best_gain = feat_best_gain
            best_f = f
            best_t = feat_best_t

    return best_f, best_t, best_gain


@njit(cache=True)
def _shared_split_leaf_gains(hg, hh, n_bins_per_feature, l2, feature,
                             threshold, n_leaves, out_gain):
    """Fill true per-leaf gains for a selected shared split."""
    nb = n_bins_per_feature[feature]
    for l in range(n_leaves):
        gt = 0.0
        ht = 0.0
        gl = 0.0
        hl = 0.0
        for b in range(nb):
            gb = hg[feature, l, b]
            hb = hh[feature, l, b]
            gt += gb
            ht += hb
            if b <= threshold:
                gl += gb
                hl += hb
        hr = ht - hl
        parent_denom = ht + l2
        left_denom = hl + l2
        right_denom = hr + l2
        if (
            ht > 0.0
            and parent_denom > 0.0
            and left_denom > 0.0
            and right_denom > 0.0
        ):
            gr = gt - gl
            out_gain[l] = (
                gl * gl / left_denom
                + gr * gr / right_denom
                - gt * gt / parent_denom
            )
        else:
            out_gain[l] = 0.0


@njit(cache=True)
def _assign_leaves(X_binned, splits_feat, splits_thr):
    """Map every row to its leaf index given the chosen splits."""
    n = X_binned.shape[0]
    depth = splits_feat.shape[0]
    leaf = np.zeros(n, dtype=np.int64)
    for i in range(n):
        idx = 0
        for d in range(depth):
            f = splits_feat[d]
            t = splits_thr[d]
            bit = 1 if X_binned[i, f] > t else 0
            idx = idx * 2 + bit
        leaf[i] = idx
    return leaf


@njit(cache=True, parallel=True)
def _descend_leaves(leaf, Xf, split_thr):
    """Append one oblivious split bit to existing leaf ids in place."""
    for i in prange(leaf.shape[0]):
        leaf[i] = (leaf[i] << 1) + (1 if Xf[i] > split_thr else 0)


@njit(cache=True, parallel=True)
def _update_leaves_with_split(X_binned, leaf, split_feat, split_thr):
    """Append one split bit to existing leaf ids in place."""
    for i in prange(leaf.shape[0]):
        leaf[i] = (leaf[i] << 1) + (1 if X_binned[i, split_feat] > split_thr else 0)


@njit(cache=True)
def _update_leaves_with_level_splits(X_binned, leaf, level_features,
                                     level_thresholds):
    """Append one leaf-local split bit to existing leaf ids in place."""
    for i in range(leaf.shape[0]):
        l = leaf[i]
        f = level_features[l]
        bit = 0
        if f >= 0:
            bit = 1 if X_binned[i, f] > level_thresholds[l] else 0
        leaf[i] = l * 2 + bit


@njit(cache=True)
def _update_leafwise_leaves_with_split(X_binned, leaf, split_leaf, new_leaf,
                                       split_feat, split_thr):
    """Split one current leaf; left rows keep split_leaf, right rows get new_leaf."""
    for i in range(leaf.shape[0]):
        if leaf[i] == split_leaf and X_binned[i, split_feat] > split_thr:
            leaf[i] = new_leaf


@njit(cache=True)
def _leaf_values(leaf, grad, hess, n_leaves, l2, lr):
    """Newton leaf values: value = -G / (H + l2), scaled by learning rate."""
    G = np.zeros(n_leaves)
    H = np.zeros(n_leaves)
    for i in range(leaf.shape[0]):
        G[leaf[i]] += grad[i]
        H[leaf[i]] += hess[i]
    values = np.zeros(n_leaves)
    for l in range(n_leaves):
        if H[l] > 0.0:
            values[l] = -lr * G[l] / (H[l] + l2)
    return values


@njit(cache=True)
def _leaf_values_and_sums(leaf, grad, hess, n_leaves, l2, lr):
    """Return Newton leaf values plus the leaf gradient/hessian totals."""
    G = np.zeros(n_leaves)
    H = np.zeros(n_leaves)
    for i in range(leaf.shape[0]):
        G[leaf[i]] += grad[i]
        H[leaf[i]] += hess[i]
    values = np.zeros(n_leaves)
    for l in range(n_leaves):
        if H[l] > 0.0:
            values[l] = -lr * G[l] / (H[l] + l2)
    return values, G, H


@njit(cache=True)
def _leaf_values_and_sums_rows(leaf, grad, hess, row_indices, n_leaves, l2, lr):
    """Return leaf values/sums using only selected training rows."""
    G = np.zeros(n_leaves)
    H = np.zeros(n_leaves)
    for p in range(row_indices.shape[0]):
        i = row_indices[p]
        G[leaf[i]] += grad[i]
        H[leaf[i]] += hess[i]
    values = np.zeros(n_leaves)
    for l in range(n_leaves):
        if H[l] > 0.0:
            values[l] = -lr * G[l] / (H[l] + l2)
    return values, G, H


@njit(cache=True)
def _multiclass_leaf_values_and_sums(leaf, grad, hess, n_leaves, l2, lr):
    """Return vector leaf values plus class-major G/H totals."""
    K = grad.shape[0]
    G = np.zeros((K, n_leaves))
    H = np.zeros((K, n_leaves))
    n = leaf.shape[0]
    for k in range(K):
        for i in range(n):
            l = leaf[i]
            G[k, l] += grad[k, i]
            H[k, l] += hess[k, i]
    values = np.zeros((n_leaves, K))
    for l in range(n_leaves):
        for k in range(K):
            if H[k, l] > 0.0:
                values[l, k] = -lr * G[k, l] / (H[k, l] + l2)
    return values, G, H


@njit(cache=True)
def _multiclass_leaf_values_and_sums_l2_by_class(
    leaf, grad, hess, n_leaves, l2_by_class, lr
):
    """Vector leaf values with an independent L2 value per output head."""
    K = grad.shape[0]
    G = np.zeros((K, n_leaves))
    H = np.zeros((K, n_leaves))
    n = leaf.shape[0]
    for k in range(K):
        for i in range(n):
            l = leaf[i]
            G[k, l] += grad[k, i]
            H[k, l] += hess[k, i]
    values = np.zeros((n_leaves, K))
    for l in range(n_leaves):
        for k in range(K):
            if H[k, l] > 0.0:
                values[l, k] = -lr * G[k, l] / (H[k, l] + l2_by_class[k])
    return values, G, H


@njit(cache=True)
def add_leaf_values_inplace(leaf, values, out):
    """Add precomputed scalar leaf values for already-routed training rows."""
    for i in range(leaf.shape[0]):
        out[i] += values[leaf[i]]


@njit(cache=True, parallel=True)
def ordered_leaf_update_inplace(leaf, leaf_G, leaf_H, grad, hess, lr, l2, F):
    """Leave-one-out Newton step added directly into the training margin.

    Each row's update uses its leaf's gradient/hessian totals with that row's
    own contribution removed. Rows whose leave-one-out denominator is not
    positive (e.g. singleton leaves with l2=0) are left unchanged, matching
    the zero-update fallback of the numpy formulation this replaces.
    """
    for i in prange(F.shape[0]):
        l = leaf[i]
        denom = leaf_H[l] - hess[i]
        if denom < 0.0:
            denom = 0.0
        denom += l2
        if denom > 0.0:
            F[i] += -lr * (leaf_G[l] - grad[i]) / denom


@njit(cache=True)
def add_multiclass_leaf_values_inplace(leaf, values, out):
    """Add vector leaf values into a class-major margin matrix."""
    K = out.shape[0]
    for i in range(leaf.shape[0]):
        l = leaf[i]
        for k in range(K):
            out[k, i] += values[l, k]


@njit(cache=True)
def _predict_tree(X_binned, splits_feat, splits_thr, values):
    """Route each row to its leaf and return the leaf value for each row."""
    leaf = _assign_leaves(X_binned, splits_feat, splits_thr)
    out = np.empty(X_binned.shape[0], dtype=np.float64)
    for i in range(leaf.shape[0]):
        out[i] = values[leaf[i]]
    return out


@njit(cache=True)
def _predict_tree_add(X_binned, splits_feat, splits_thr, values, out):
    """Route each row and add the leaf value into an existing output array."""
    n = X_binned.shape[0]
    depth = splits_feat.shape[0]
    for i in range(n):
        idx = 0
        for d in range(depth):
            f = splits_feat[d]
            t = splits_thr[d]
            bit = 1 if X_binned[i, f] > t else 0
            idx = idx * 2 + bit
        out[i] += values[idx]


@njit(cache=True)
def _assign_levelwise_leaves(X_binned, node_features, node_thresholds):
    """Map rows through a level-wise tree with one split per node."""
    n = X_binned.shape[0]
    depth = node_features.shape[0]
    leaf = np.zeros(n, dtype=np.int64)
    for i in range(n):
        idx = 0
        for d in range(depth):
            f = node_features[d, idx]
            bit = 0
            if f >= 0:
                bit = 1 if X_binned[i, f] > node_thresholds[d, idx] else 0
            idx = idx * 2 + bit
        leaf[i] = idx
    return leaf


@njit(cache=True)
def _predict_levelwise_tree(X_binned, node_features, node_thresholds, values):
    leaf = _assign_levelwise_leaves(X_binned, node_features, node_thresholds)
    out = np.empty(X_binned.shape[0], dtype=np.float64)
    for i in range(leaf.shape[0]):
        out[i] = values[leaf[i]]
    return out


@njit(cache=True)
def _predict_levelwise_tree_add(X_binned, node_features, node_thresholds,
                                values, out):
    n = X_binned.shape[0]
    depth = node_features.shape[0]
    for i in range(n):
        idx = 0
        for d in range(depth):
            f = node_features[d, idx]
            bit = 0
            if f >= 0:
                bit = 1 if X_binned[i, f] > node_thresholds[d, idx] else 0
            idx = idx * 2 + bit
        out[i] += values[idx]


@njit(cache=True)
def _assign_non_oblivious_leaves(X_binned, features, thresholds, left_child,
                                 right_child, leaf_index):
    """Map rows through an explicit-node binary decision tree."""
    n = X_binned.shape[0]
    leaf = np.zeros(n, dtype=np.int64)
    for i in range(n):
        node = 0
        while left_child[node] >= 0:
            f = features[node]
            t = thresholds[node]
            if X_binned[i, f] > t:
                node = right_child[node]
            else:
                node = left_child[node]
        leaf[i] = leaf_index[node]
    return leaf


@njit(cache=True)
def _predict_non_oblivious_tree(X_binned, features, thresholds, left_child,
                                right_child, leaf_index, values):
    leaf = _assign_non_oblivious_leaves(
        X_binned, features, thresholds, left_child, right_child, leaf_index
    )
    out = np.empty(X_binned.shape[0], dtype=np.float64)
    for i in range(leaf.shape[0]):
        out[i] = values[leaf[i]]
    return out


@njit(cache=True)
def _predict_non_oblivious_tree_add(X_binned, features, thresholds, left_child,
                                    right_child, leaf_index, values, out):
    n = X_binned.shape[0]
    for i in range(n):
        node = 0
        while left_child[node] >= 0:
            f = features[node]
            t = thresholds[node]
            if X_binned[i, f] > t:
                node = right_child[node]
            else:
                node = left_child[node]
        out[i] += values[leaf_index[node]]


@njit(cache=True, parallel=True)
def _predict_non_oblivious_tree_add_parallel(
    X_binned, features, thresholds, left_child, right_child, leaf_index,
    values, out
):
    n = X_binned.shape[0]
    for i in prange(n):
        node = 0
        while left_child[node] >= 0:
            f = features[node]
            t = thresholds[node]
            if X_binned[i, f] > t:
                node = right_child[node]
            else:
                node = left_child[node]
        out[i] += values[leaf_index[node]]


@njit(cache=True)
def _predict_non_oblivious_multiclass_tree_add(
    X_binned, features, thresholds, left_child, right_child, leaf_index,
    values, out
):
    n = X_binned.shape[0]
    K = values.shape[1]
    for i in range(n):
        node = 0
        while left_child[node] >= 0:
            f = features[node]
            t = thresholds[node]
            if X_binned[i, f] > t:
                node = right_child[node]
            else:
                node = left_child[node]
        l = leaf_index[node]
        for k in range(K):
            out[k, i] += values[l, k]


@njit(cache=True, parallel=True)
def _predict_non_oblivious_multiclass_tree_add_parallel(
    X_binned, features, thresholds, left_child, right_child, leaf_index,
    values, out
):
    n = X_binned.shape[0]
    K = values.shape[1]
    for i in prange(n):
        node = 0
        while left_child[node] >= 0:
            f = features[node]
            t = thresholds[node]
            if X_binned[i, f] > t:
                node = right_child[node]
            else:
                node = left_child[node]
        l = leaf_index[node]
        for k in range(K):
            out[k, i] += values[l, k]


@njit(cache=True, parallel=True)
def _best_splits_by_leaf(hg, hh, n_bins_per_feature, l2, feat_mask,
                         min_child_weight, n_leaves, out_feat, out_thr,
                         out_gain):
    """Find the best split independently for every active leaf."""
    n_features = n_bins_per_feature.shape[0]
    for l in prange(n_leaves):
        best_f = -1
        best_t = -1
        best_gain = -np.inf

        for f in range(n_features):
            if feat_mask[f] == 0:
                continue
            nb = n_bins_per_feature[f]
            Gt = 0.0
            Ht = 0.0
            for b in range(nb):
                Gt += hg[f, l, b]
                Ht += hh[f, l, b]
            parent_denom = Ht + l2
            if Ht <= 0.0 or parent_denom <= 0.0:
                continue
            parent_gain = Gt * Gt / parent_denom

            GL = 0.0
            HL = 0.0
            for t in range(nb - 1):
                GL += hg[f, l, t]
                HL += hh[f, l, t]
                HR = Ht - HL
                if HL < min_child_weight or HR < min_child_weight:
                    continue
                left_denom = HL + l2
                right_denom = HR + l2
                if left_denom <= 0.0 or right_denom <= 0.0:
                    continue
                GR = Gt - GL
                gain = (
                    GL * GL / left_denom
                    + GR * GR / right_denom
                    - parent_gain
                )
                if gain > best_gain:
                    best_gain = gain
                    best_f = f
                    best_t = t

        out_feat[l] = best_f
        out_thr[l] = best_t
        out_gain[l] = best_gain


@njit(cache=True, parallel=True)
def _best_splits_by_leaf_counts(hg, hh, hc, n_bins_per_feature, l2, feat_mask,
                                min_child_weight, min_child_samples, n_leaves,
                                out_feat, out_thr, out_gain):
    """Best split per active leaf with Hessian and positive-weight row limits."""
    n_features = n_bins_per_feature.shape[0]
    for l in prange(n_leaves):
        best_f = -1
        best_t = -1
        best_gain = -np.inf

        for f in range(n_features):
            if feat_mask[f] == 0:
                continue
            nb = n_bins_per_feature[f]
            Gt = 0.0
            Ht = 0.0
            Ct = 0.0
            for b in range(nb):
                Gt += hg[f, l, b]
                Ht += hh[f, l, b]
                Ct += hc[f, l, b]
            parent_denom = Ht + l2
            if Ht <= 0.0 or Ct <= 0.0 or parent_denom <= 0.0:
                continue
            parent_gain = Gt * Gt / parent_denom

            GL = 0.0
            HL = 0.0
            CL = 0.0
            for t in range(nb - 1):
                GL += hg[f, l, t]
                HL += hh[f, l, t]
                CL += hc[f, l, t]
                HR = Ht - HL
                CR = Ct - CL
                if (
                    HL < min_child_weight
                    or HR < min_child_weight
                    or CL < min_child_samples
                    or CR < min_child_samples
                ):
                    continue
                left_denom = HL + l2
                right_denom = HR + l2
                if left_denom <= 0.0 or right_denom <= 0.0:
                    continue
                GR = Gt - GL
                gain = (
                    GL * GL / left_denom
                    + GR * GR / right_denom
                    - parent_gain
                )
                if gain > best_gain:
                    best_gain = gain
                    best_f = f
                    best_t = t

        out_feat[l] = best_f
        out_thr[l] = best_t
        out_gain[l] = best_gain


@njit(cache=True, parallel=True)
def _best_splits_for_leaf_ids_counts(hg, hh, hc, n_bins_per_feature, l2,
                                     feat_mask, min_child_weight,
                                     min_child_samples, leaf_ids, n_leaf_ids,
                                     out_feat, out_thr, out_gain):
    """Best split for a small set of changed leaves."""
    n_features = n_bins_per_feature.shape[0]
    for idx in prange(n_leaf_ids):
        l = leaf_ids[idx]
        best_f = -1
        best_t = -1
        best_gain = -np.inf

        for f in range(n_features):
            if feat_mask[f] == 0:
                continue
            nb = n_bins_per_feature[f]
            Gt = 0.0
            Ht = 0.0
            Ct = 0.0
            for b in range(nb):
                Gt += hg[f, l, b]
                Ht += hh[f, l, b]
                Ct += hc[f, l, b]
            parent_denom = Ht + l2
            if Ht <= 0.0 or Ct <= 0.0 or parent_denom <= 0.0:
                continue
            parent_gain = Gt * Gt / parent_denom

            GL = 0.0
            HL = 0.0
            CL = 0.0
            for t in range(nb - 1):
                GL += hg[f, l, t]
                HL += hh[f, l, t]
                CL += hc[f, l, t]
                HR = Ht - HL
                CR = Ct - CL
                if (
                    HL < min_child_weight
                    or HR < min_child_weight
                    or CL < min_child_samples
                    or CR < min_child_samples
                ):
                    continue
                left_denom = HL + l2
                right_denom = HR + l2
                if left_denom <= 0.0 or right_denom <= 0.0:
                    continue
                GR = Gt - GL
                gain = (
                    GL * GL / left_denom
                    + GR * GR / right_denom
                    - parent_gain
                )
                if gain > best_gain:
                    best_gain = gain
                    best_f = f
                    best_t = t

        out_feat[l] = best_f
        out_thr[l] = best_t
        out_gain[l] = best_gain


@njit(cache=True, parallel=True)
def _best_splits_for_leaf_ids_counts_feature_parallel(
    hg, hh, hc, n_bins_per_feature, l2, feat_mask, min_child_weight,
    min_child_samples, leaf_ids, n_leaf_ids, feature_gain, feature_thr,
    out_feat, out_thr, out_gain
):
    """Best split for changed leaves, parallelized over leaf-feature pairs."""
    n_features = n_bins_per_feature.shape[0]
    for flat in prange(n_leaf_ids * n_features):
        idx = flat // n_features
        f = flat - idx * n_features
        l = leaf_ids[idx]
        best_t = -1
        best_gain = -np.inf

        if feat_mask[f] != 0:
            nb = n_bins_per_feature[f]
            Gt = 0.0
            Ht = 0.0
            Ct = 0.0
            for b in range(nb):
                Gt += hg[f, l, b]
                Ht += hh[f, l, b]
                Ct += hc[f, l, b]
            parent_denom = Ht + l2
            if Ht > 0.0 and Ct > 0.0 and parent_denom > 0.0:
                parent_gain = Gt * Gt / parent_denom

                GL = 0.0
                HL = 0.0
                CL = 0.0
                for t in range(nb - 1):
                    GL += hg[f, l, t]
                    HL += hh[f, l, t]
                    CL += hc[f, l, t]
                    HR = Ht - HL
                    CR = Ct - CL
                    if (
                        HL < min_child_weight
                        or HR < min_child_weight
                        or CL < min_child_samples
                        or CR < min_child_samples
                    ):
                        continue
                    left_denom = HL + l2
                    right_denom = HR + l2
                    if left_denom <= 0.0 or right_denom <= 0.0:
                        continue
                    GR = Gt - GL
                    gain = (
                        GL * GL / left_denom
                        + GR * GR / right_denom
                        - parent_gain
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_t = t

        feature_gain[f, l] = best_gain
        feature_thr[f, l] = best_t

    for idx in range(n_leaf_ids):
        l = leaf_ids[idx]
        best_f = -1
        best_t = -1
        best_gain = -np.inf
        for f in range(n_features):
            gain = feature_gain[f, l]
            if gain > best_gain:
                best_gain = gain
                best_f = f
                best_t = int(feature_thr[f, l])

        out_feat[l] = best_f
        out_thr[l] = best_t
        out_gain[l] = best_gain


@njit(cache=True, parallel=True)
def _best_splits_by_leaf_counts_full_features(
    hg, hh, hc, n_bins_per_feature, l2, min_child_weight,
    min_child_samples, n_leaves, leaf_start, out_feat, out_thr, out_gain
):
    """Best split per leaf when every row and every feature is active.

    In this lane each non-missing row contributes exactly once to every
    feature histogram, so leaf totals are invariant across features. Split
    legality still uses per-threshold histogram counts, but parent totals do
    not need to be recomputed for each feature.
    """
    n_features = n_bins_per_feature.shape[0]
    for l in prange(n_leaves):
        best_f = -1
        best_t = -1
        best_gain = -np.inf
        Ct = float(leaf_start[l + 1] - leaf_start[l])

        if Ct <= 0.0:
            out_feat[l] = best_f
            out_thr[l] = best_t
            out_gain[l] = best_gain
            continue

        nb0 = n_bins_per_feature[0]
        Gt = 0.0
        Ht = 0.0
        for b in range(nb0):
            Gt += hg[0, l, b]
            Ht += hh[0, l, b]
        parent_denom = Ht + l2
        if Ht <= 0.0 or parent_denom <= 0.0:
            out_feat[l] = best_f
            out_thr[l] = best_t
            out_gain[l] = best_gain
            continue
        parent_gain = Gt * Gt / parent_denom

        for f in range(n_features):
            nb = n_bins_per_feature[f]
            GL = 0.0
            HL = 0.0
            CL = 0.0
            for t in range(nb - 1):
                GL += hg[f, l, t]
                HL += hh[f, l, t]
                CL += hc[f, l, t]
                HR = Ht - HL
                CR = Ct - CL
                if (
                    HL < min_child_weight
                    or HR < min_child_weight
                    or CL < min_child_samples
                    or CR < min_child_samples
                ):
                    continue
                left_denom = HL + l2
                right_denom = HR + l2
                if left_denom <= 0.0 or right_denom <= 0.0:
                    continue
                GR = Gt - GL
                gain = (
                    GL * GL / left_denom
                    + GR * GR / right_denom
                    - parent_gain
                )
                if gain > best_gain:
                    best_gain = gain
                    best_f = f
                    best_t = t

        out_feat[l] = best_f
        out_thr[l] = best_t
        out_gain[l] = best_gain


@njit(cache=True, parallel=True)
def _best_splits_for_leaf_ids_counts_full_features(
    hg, hh, hc, n_bins_per_feature, l2, min_child_weight,
    min_child_samples, leaf_ids, n_leaf_ids, leaf_start,
    out_feat, out_thr, out_gain
):
    """Best split for changed leaves when every row/feature is active."""
    n_features = n_bins_per_feature.shape[0]
    for idx in prange(n_leaf_ids):
        l = leaf_ids[idx]
        best_f = -1
        best_t = -1
        best_gain = -np.inf
        Ct = float(leaf_start[l + 1] - leaf_start[l])

        if Ct <= 0.0:
            out_feat[l] = best_f
            out_thr[l] = best_t
            out_gain[l] = best_gain
            continue

        nb0 = n_bins_per_feature[0]
        Gt = 0.0
        Ht = 0.0
        for b in range(nb0):
            Gt += hg[0, l, b]
            Ht += hh[0, l, b]
        parent_denom = Ht + l2
        if Ht <= 0.0 or parent_denom <= 0.0:
            out_feat[l] = best_f
            out_thr[l] = best_t
            out_gain[l] = best_gain
            continue
        parent_gain = Gt * Gt / parent_denom

        for f in range(n_features):
            nb = n_bins_per_feature[f]
            GL = 0.0
            HL = 0.0
            CL = 0.0
            for t in range(nb - 1):
                GL += hg[f, l, t]
                HL += hh[f, l, t]
                CL += hc[f, l, t]
                HR = Ht - HL
                CR = Ct - CL
                if (
                    HL < min_child_weight
                    or HR < min_child_weight
                    or CL < min_child_samples
                    or CR < min_child_samples
                ):
                    continue
                left_denom = HL + l2
                right_denom = HR + l2
                if left_denom <= 0.0 or right_denom <= 0.0:
                    continue
                GR = Gt - GL
                gain = (
                    GL * GL / left_denom
                    + GR * GR / right_denom
                    - parent_gain
                )
                if gain > best_gain:
                    best_gain = gain
                    best_f = f
                    best_t = t

        out_feat[l] = best_f
        out_thr[l] = best_t
        out_gain[l] = best_gain


@njit(cache=True, parallel=True)
def _best_splits_for_leaf_ids_counts_full_features_by_count(
    hg, hh, hc, n_bins_per_feature, l2, min_child_weight,
    min_child_samples, leaf_ids, n_leaf_ids, leaf_count,
    out_feat, out_thr, out_gain
):
    """Best split for full-feature segmented leaves using explicit row counts."""
    n_features = n_bins_per_feature.shape[0]
    for idx in prange(n_leaf_ids):
        l = leaf_ids[idx]
        best_f = -1
        best_t = -1
        best_gain = -np.inf
        Ct = float(leaf_count[l])

        if Ct <= 0.0:
            out_feat[l] = best_f
            out_thr[l] = best_t
            out_gain[l] = best_gain
            continue

        nb0 = n_bins_per_feature[0]
        Gt = 0.0
        Ht = 0.0
        for b in range(nb0):
            Gt += hg[0, l, b]
            Ht += hh[0, l, b]
        parent_denom = Ht + l2
        if Ht <= 0.0 or parent_denom <= 0.0:
            out_feat[l] = best_f
            out_thr[l] = best_t
            out_gain[l] = best_gain
            continue
        parent_gain = Gt * Gt / parent_denom

        for f in range(n_features):
            nb = n_bins_per_feature[f]
            GL = 0.0
            HL = 0.0
            CL = 0.0
            for t in range(nb - 1):
                GL += hg[f, l, t]
                HL += hh[f, l, t]
                CL += hc[f, l, t]
                HR = Ht - HL
                CR = Ct - CL
                if (
                    HL < min_child_weight
                    or HR < min_child_weight
                    or CL < min_child_samples
                    or CR < min_child_samples
                ):
                    continue
                left_denom = HL + l2
                right_denom = HR + l2
                if left_denom <= 0.0 or right_denom <= 0.0:
                    continue
                GR = Gt - GL
                gain = (
                    GL * GL / left_denom
                    + GR * GR / right_denom
                    - parent_gain
                )
                if gain > best_gain:
                    best_gain = gain
                    best_f = f
                    best_t = t

        out_feat[l] = best_f
        out_thr[l] = best_t
        out_gain[l] = best_gain


@njit(cache=True)
def _selected_split_gain_full_rows(
    X_binned, grad, hess, leaf, split_leaf, split_feature, split_threshold,
    l2, min_child_weight, min_child_samples, constant_hessian
):
    """Canonical selected-split gain from original row order."""
    Gt = 0.0
    Ht = 0.0
    Ct = 0.0
    GL = 0.0
    HL = 0.0
    CL = 0.0
    n_samples = X_binned.shape[0]

    for i in range(n_samples):
        if leaf[i] != split_leaf:
            continue
        gi = grad[i]
        hi = 1.0 if constant_hessian else hess[i]
        Gt += gi
        Ht += hi
        positive = constant_hessian or hi > 0.0
        if positive:
            Ct += 1.0
        if X_binned[i, split_feature] <= split_threshold:
            GL += gi
            HL += hi
            if positive:
                CL += 1.0

    HR = Ht - HL
    CR = Ct - CL
    if (
        Ht <= 0.0
        or Ct <= 0.0
        or HL < min_child_weight
        or HR < min_child_weight
        or CL < min_child_samples
        or CR < min_child_samples
    ):
        return -np.inf
    parent_denom = Ht + l2
    left_denom = HL + l2
    right_denom = HR + l2
    if parent_denom <= 0.0 or left_denom <= 0.0 or right_denom <= 0.0:
        return -np.inf
    GR = Gt - GL
    return GL * GL / left_denom + GR * GR / right_denom - Gt * Gt / parent_denom


@njit(cache=True, parallel=True)
def _build_multiclass_histograms_counts_into(
    X_binned, grad, hess, leaf, n_leaves, hg, hh, hc
):
    """Fill class-major grad/hess histograms plus shared positive-row counts."""
    K, n_samples = grad.shape
    n_features = X_binned.shape[1]
    max_bins = hg.shape[3]
    for f in prange(n_features):
        for k in range(K):
            for l in range(n_leaves):
                for b in range(max_bins):
                    hg[k, f, l, b] = 0.0
                    hh[k, f, l, b] = 0.0
        for l in range(n_leaves):
            for b in range(max_bins):
                hc[f, l, b] = 0.0

        for i in range(n_samples):
            l = leaf[i]
            b = X_binned[i, f]
            h_sum = 0.0
            for k in range(K):
                h_sum += hess[k, i]
            if h_sum > 0.0:
                hc[f, l, b] += 1.0
            for k in range(K):
                hg[k, f, l, b] += grad[k, i]
                hh[k, f, l, b] += hess[k, i]


def _class_major_views_from_class_minor_histograms(hg, hh):
    """Return ``(K, f, leaf, bin)`` views over ``(f, leaf, bin, K)`` buffers."""
    return np.moveaxis(hg, 3, 0), np.moveaxis(hh, 3, 0)


@njit(cache=True, parallel=True)
def _build_multiclass_histograms_counts_class_minor_into(
    X_binned, grad, hess, leaf, n_leaves, hg, hh, hc
):
    """Fill class-minor grad/hess histograms plus shared positive-row counts."""
    n_samples, K = grad.shape
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                hc[f, l, b] = 0.0
                for k in range(K):
                    hg[f, l, b, k] = 0.0
                    hh[f, l, b, k] = 0.0

        for i in range(n_samples):
            l = leaf[i]
            b = X_binned[i, f]
            h_sum = 0.0
            for k in range(K):
                h_sum += hess[i, k]
            if h_sum > 0.0:
                hc[f, l, b] += 1.0
            for k in range(K):
                hg[f, l, b, k] += grad[i, k]
                hh[f, l, b, k] += hess[i, k]


@njit(cache=True, parallel=True)
def _refill_multiclass_leaf_segment_histograms_counts_into(
    X_binned, grad, hess, row_order, leaf_start, leaf_ids, n_leaf_ids,
    hg, hh, hc
):
    """Refill changed-leaf multiclass histograms from row segments."""
    K = grad.shape[0]
    n_features = X_binned.shape[1]
    max_bins = hg.shape[3]
    for f in prange(n_features):
        for idx in range(n_leaf_ids):
            l = leaf_ids[idx]
            for k in range(K):
                for b in range(max_bins):
                    hg[k, f, l, b] = 0.0
                    hh[k, f, l, b] = 0.0
            for b in range(max_bins):
                hc[f, l, b] = 0.0

            for p in range(leaf_start[l], leaf_start[l + 1]):
                i = row_order[p]
                b = X_binned[i, f]
                h_sum = 0.0
                for k in range(K):
                    h_sum += hess[k, i]
                if h_sum > 0.0:
                    hc[f, l, b] += 1.0
                for k in range(K):
                    hg[k, f, l, b] += grad[k, i]
                    hh[k, f, l, b] += hess[k, i]


@njit(cache=True, parallel=True)
def _refill_multiclass_leaf_segment_histograms_counts_class_minor_into(
    X_binned, grad, hess, row_order, leaf_start, leaf_ids, n_leaf_ids,
    hg, hh, hc
):
    """Refill changed-leaf class-minor multiclass histograms from segments."""
    K = grad.shape[1]
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for idx in range(n_leaf_ids):
            l = leaf_ids[idx]
            for b in range(max_bins):
                hc[f, l, b] = 0.0
                for k in range(K):
                    hg[f, l, b, k] = 0.0
                    hh[f, l, b, k] = 0.0

            for p in range(leaf_start[l], leaf_start[l + 1]):
                i = row_order[p]
                b = X_binned[i, f]
                h_sum = 0.0
                for k in range(K):
                    h_sum += hess[i, k]
                if h_sum > 0.0:
                    hc[f, l, b] += 1.0
                for k in range(K):
                    hg[f, l, b, k] += grad[i, k]
                    hh[f, l, b, k] += hess[i, k]


@njit(cache=True, parallel=True)
def _refill_multiclass_right_subtract_left_counts_into(
    X_binned, grad, hess, row_order, leaf_start, left_leaf, right_leaf,
    hg, hh, hc
):
    """Refill the right multiclass child and derive the left child."""
    K = grad.shape[0]
    n_features = X_binned.shape[1]
    max_bins = hg.shape[3]
    for f in prange(n_features):
        for k in range(K):
            for b in range(max_bins):
                hg[k, f, right_leaf, b] = 0.0
                hh[k, f, right_leaf, b] = 0.0
        for b in range(max_bins):
            hc[f, right_leaf, b] = 0.0

        for p in range(leaf_start[right_leaf], leaf_start[right_leaf + 1]):
            i = row_order[p]
            b = X_binned[i, f]
            h_sum = 0.0
            for k in range(K):
                h_sum += hess[k, i]
            if h_sum > 0.0:
                hc[f, right_leaf, b] += 1.0
            for k in range(K):
                hg[k, f, right_leaf, b] += grad[k, i]
                hh[k, f, right_leaf, b] += hess[k, i]

        for k in range(K):
            for b in range(max_bins):
                hg[k, f, left_leaf, b] -= hg[k, f, right_leaf, b]
                hh[k, f, left_leaf, b] -= hh[k, f, right_leaf, b]
        for b in range(max_bins):
            hc[f, left_leaf, b] -= hc[f, right_leaf, b]


@njit(cache=True, parallel=True)
def _refill_multiclass_right_subtract_left_counts_class_minor_into(
    X_binned, grad, hess, row_order, leaf_start, left_leaf, right_leaf,
    hg, hh, hc
):
    """Refill the right class-minor child and derive the left child."""
    K = grad.shape[1]
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for b in range(max_bins):
            hc[f, right_leaf, b] = 0.0
            for k in range(K):
                hg[f, right_leaf, b, k] = 0.0
                hh[f, right_leaf, b, k] = 0.0

        for p in range(leaf_start[right_leaf], leaf_start[right_leaf + 1]):
            i = row_order[p]
            b = X_binned[i, f]
            h_sum = 0.0
            for k in range(K):
                h_sum += hess[i, k]
            if h_sum > 0.0:
                hc[f, right_leaf, b] += 1.0
            for k in range(K):
                hg[f, right_leaf, b, k] += grad[i, k]
                hh[f, right_leaf, b, k] += hess[i, k]

        for b in range(max_bins):
            hc[f, left_leaf, b] -= hc[f, right_leaf, b]
            for k in range(K):
                hg[f, left_leaf, b, k] -= hg[f, right_leaf, b, k]
                hh[f, left_leaf, b, k] -= hh[f, right_leaf, b, k]


@njit(cache=True, parallel=True)
def _refill_multiclass_left_subtract_right_counts_into(
    X_binned, grad, hess, row_order, leaf_start, left_leaf, right_leaf,
    hg, hh, hc
):
    """Refill the left multiclass child and derive the right child."""
    K = grad.shape[0]
    n_features = X_binned.shape[1]
    max_bins = hg.shape[3]
    for f in prange(n_features):
        for k in range(K):
            for b in range(max_bins):
                hg[k, f, right_leaf, b] = hg[k, f, left_leaf, b]
                hh[k, f, right_leaf, b] = hh[k, f, left_leaf, b]
                hg[k, f, left_leaf, b] = 0.0
                hh[k, f, left_leaf, b] = 0.0
        for b in range(max_bins):
            hc[f, right_leaf, b] = hc[f, left_leaf, b]
            hc[f, left_leaf, b] = 0.0

        for p in range(leaf_start[left_leaf], leaf_start[left_leaf + 1]):
            i = row_order[p]
            b = X_binned[i, f]
            h_sum = 0.0
            for k in range(K):
                h_sum += hess[k, i]
            if h_sum > 0.0:
                hc[f, left_leaf, b] += 1.0
            for k in range(K):
                hg[k, f, left_leaf, b] += grad[k, i]
                hh[k, f, left_leaf, b] += hess[k, i]

        for k in range(K):
            for b in range(max_bins):
                hg[k, f, right_leaf, b] -= hg[k, f, left_leaf, b]
                hh[k, f, right_leaf, b] -= hh[k, f, left_leaf, b]
        for b in range(max_bins):
            hc[f, right_leaf, b] -= hc[f, left_leaf, b]


@njit(cache=True, parallel=True)
def _refill_multiclass_left_subtract_right_counts_class_minor_into(
    X_binned, grad, hess, row_order, leaf_start, left_leaf, right_leaf,
    hg, hh, hc
):
    """Refill the left class-minor child and derive the right child."""
    K = grad.shape[1]
    n_features = X_binned.shape[1]
    max_bins = hg.shape[2]
    for f in prange(n_features):
        for b in range(max_bins):
            hc[f, right_leaf, b] = hc[f, left_leaf, b]
            hc[f, left_leaf, b] = 0.0
            for k in range(K):
                hg[f, right_leaf, b, k] = hg[f, left_leaf, b, k]
                hh[f, right_leaf, b, k] = hh[f, left_leaf, b, k]
                hg[f, left_leaf, b, k] = 0.0
                hh[f, left_leaf, b, k] = 0.0

        for p in range(leaf_start[left_leaf], leaf_start[left_leaf + 1]):
            i = row_order[p]
            b = X_binned[i, f]
            h_sum = 0.0
            for k in range(K):
                h_sum += hess[i, k]
            if h_sum > 0.0:
                hc[f, left_leaf, b] += 1.0
            for k in range(K):
                hg[f, left_leaf, b, k] += grad[i, k]
                hh[f, left_leaf, b, k] += hess[i, k]

        for b in range(max_bins):
            hc[f, right_leaf, b] -= hc[f, left_leaf, b]
            for k in range(K):
                hg[f, right_leaf, b, k] -= hg[f, left_leaf, b, k]
                hh[f, right_leaf, b, k] -= hh[f, left_leaf, b, k]


@njit(cache=True, parallel=True)
def _best_multiclass_splits_for_leaf_ids_counts(
    hg, hh, hc, n_bins_per_feature, l2, feat_mask, min_child_weight,
    min_child_samples, leaf_ids, n_leaf_ids, out_feat, out_thr, out_gain
):
    """Best split per changed leaf by summed gain across classes."""
    K = hg.shape[0]
    n_features = hg.shape[1]
    for idx in prange(n_leaf_ids):
        l = leaf_ids[idx]
        best_f = -1
        best_t = -1
        best_gain = -np.inf

        for f in range(n_features):
            if feat_mask[f] == 0:
                continue
            nb = n_bins_per_feature[f]
            Ct = 0.0
            Gt = np.zeros(K)
            Ht = np.zeros(K)
            GL = np.zeros(K)
            HL = np.zeros(K)
            total_H = 0.0
            for b in range(nb):
                Ct += hc[f, l, b]
                for k in range(K):
                    Gt[k] += hg[k, f, l, b]
                    Ht[k] += hh[k, f, l, b]
            for k in range(K):
                total_H += Ht[k]
            if Ct <= 0.0 or total_H <= 0.0:
                continue

            CL = 0.0
            for t in range(nb - 1):
                CL += hc[f, l, t]
                sum_HL = 0.0
                for k in range(K):
                    GL[k] += hg[k, f, l, t]
                    HL[k] += hh[k, f, l, t]
                    sum_HL += HL[k]
                CR = Ct - CL
                if (
                    CL < min_child_samples
                    or CR < min_child_samples
                    or sum_HL < min_child_weight
                    or total_H - sum_HL < min_child_weight
                ):
                    continue

                split_gain = 0.0
                for k in range(K):
                    if Ht[k] <= 0.0:
                        continue
                    HR = Ht[k] - HL[k]
                    l2k = l2[k]
                    parent_denom = Ht[k] + l2k
                    left_denom = HL[k] + l2k
                    right_denom = HR + l2k
                    if (
                        parent_denom <= 0.0
                        or left_denom <= 0.0
                        or right_denom <= 0.0
                    ):
                        continue
                    GR = Gt[k] - GL[k]
                    split_gain += (
                        GL[k] * GL[k] / left_denom
                        + GR * GR / right_denom
                        - Gt[k] * Gt[k] / parent_denom
                    )

                if split_gain > best_gain:
                    best_gain = split_gain
                    best_f = f
                    best_t = t

        out_feat[l] = best_f
        out_thr[l] = best_t
        out_gain[l] = best_gain


@njit(cache=True, parallel=True)
def _best_multiclass_splits_for_leaf_ids_counts_class_minor(
    hg, hh, hc, n_bins_per_feature, l2, feat_mask, min_child_weight,
    min_child_samples, leaf_ids, n_leaf_ids, out_feat, out_thr, out_gain
):
    """Best split per changed leaf by summed gain across classes."""
    n_features = n_bins_per_feature.shape[0]
    K = hg.shape[3]
    for idx in prange(n_leaf_ids):
        l = leaf_ids[idx]
        best_f = -1
        best_t = -1
        best_gain = -np.inf

        for f in range(n_features):
            if feat_mask[f] == 0:
                continue
            nb = n_bins_per_feature[f]
            Ct = 0.0
            Gt = np.zeros(K)
            Ht = np.zeros(K)
            GL = np.zeros(K)
            HL = np.zeros(K)
            total_H = 0.0
            for b in range(nb):
                Ct += hc[f, l, b]
                for k in range(K):
                    Gt[k] += hg[f, l, b, k]
                    Ht[k] += hh[f, l, b, k]
            for k in range(K):
                total_H += Ht[k]
            if Ct <= 0.0 or total_H <= 0.0:
                continue

            CL = 0.0
            for t in range(nb - 1):
                CL += hc[f, l, t]
                sum_HL = 0.0
                for k in range(K):
                    GL[k] += hg[f, l, t, k]
                    HL[k] += hh[f, l, t, k]
                    sum_HL += HL[k]
                CR = Ct - CL
                if (
                    CL < min_child_samples
                    or CR < min_child_samples
                    or sum_HL < min_child_weight
                    or total_H - sum_HL < min_child_weight
                ):
                    continue

                split_gain = 0.0
                for k in range(K):
                    if Ht[k] <= 0.0:
                        continue
                    HR = Ht[k] - HL[k]
                    l2k = l2[k]
                    parent_denom = Ht[k] + l2k
                    left_denom = HL[k] + l2k
                    right_denom = HR + l2k
                    if (
                        parent_denom <= 0.0
                        or left_denom <= 0.0
                        or right_denom <= 0.0
                    ):
                        continue
                    GR = Gt[k] - GL[k]
                    split_gain += (
                        GL[k] * GL[k] / left_denom
                        + GR * GR / right_denom
                        - Gt[k] * Gt[k] / parent_denom
                    )

                if split_gain > best_gain:
                    best_gain = split_gain
                    best_f = f
                    best_t = t

        out_feat[l] = best_f
        out_thr[l] = best_t
        out_gain[l] = best_gain


@njit(cache=True)
def _split_noise(random_strength, split_seed, tree_iteration, step, leaf, feature,
                 threshold):
    if random_strength <= 0.0:
        return 0.0
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    x = np.uint64(split_seed)
    x ^= (np.uint64(tree_iteration) + np.uint64(0x9E3779B97F4A7C15)) & mask
    x ^= (np.uint64(step) + np.uint64(0xBF58476D1CE4E5B9)) & mask
    x ^= (np.uint64(leaf) + np.uint64(0x94D049BB133111EB)) & mask
    x ^= (np.uint64(feature) * np.uint64(0x2545F4914F6CDD1D)) & mask
    x ^= (np.uint64(threshold) * np.uint64(0xD6E8FEB86659FD93)) & mask
    x &= mask
    x ^= x >> np.uint64(30)
    x = (x * np.uint64(0xBF58476D1CE4E5B9)) & mask
    x ^= x >> np.uint64(27)
    x = (x * np.uint64(0x94D049BB133111EB)) & mask
    x ^= x >> np.uint64(31)
    unit = float(x >> np.uint64(11)) * (1.0 / float(1 << 53))
    return random_strength * (2.0 * unit - 1.0)


@njit(cache=True)
def _noisy_score(gain, random_strength, split_seed, tree_iteration, step, leaf,
                 feature, threshold):
    if not np.isfinite(gain):
        return gain
    return gain + _split_noise(
        random_strength, split_seed, tree_iteration, step, leaf, feature,
        threshold
    )


@njit(cache=True)
def _best_split_with_noise_py(hg, hh, n_bins_per_feature, l2, feat_mask,
                              min_child_weight, n_leaves, random_strength,
                              split_seed, tree_iteration, step, min_gain):
    n_features = n_bins_per_feature.shape[0]
    best_f = 0
    best_t = -1
    best_gain = -np.inf
    best_score = -np.inf
    for f in range(n_features):
        if feat_mask[f] == 0:
            continue
        nb = int(n_bins_per_feature[f])
        Gt = hg[f, :n_leaves, :nb].sum(axis=1)
        Ht = hh[f, :n_leaves, :nb].sum(axis=1)
        GL = np.zeros(n_leaves, dtype=np.float64)
        HL = np.zeros(n_leaves, dtype=np.float64)
        parent = np.zeros(n_leaves, dtype=np.float64)
        ok_parent = (Ht > 0.0) & ((Ht + l2) > 0.0)
        parent[ok_parent] = Gt[ok_parent] * Gt[ok_parent] / (Ht[ok_parent] + l2)
        for t in range(nb - 1):
            GL += hg[f, :n_leaves, t]
            HL += hh[f, :n_leaves, t]
            gain = 0.0
            legal = True
            any_nonempty = False
            for l in range(n_leaves):
                if Ht[l] <= 0.0:
                    continue
                any_nonempty = True
                hr = Ht[l] - HL[l]
                left_denom = HL[l] + l2
                right_denom = hr + l2
                parent_denom = Ht[l] + l2
                if (
                    HL[l] < min_child_weight
                    or hr < min_child_weight
                    or left_denom <= 0.0
                    or right_denom <= 0.0
                    or parent_denom <= 0.0
                ):
                    legal = False
                    break
                gr = Gt[l] - GL[l]
                gain += GL[l] * GL[l] / left_denom + gr * gr / right_denom - parent[l]
            if legal and any_nonempty and gain > min_gain:
                score = _noisy_score(
                    gain, random_strength, split_seed, tree_iteration, step,
                    -1, f, t
                )
                if score > best_score:
                    best_score = score
                    best_gain = gain
                    best_f = f
                    best_t = t
    return best_f, best_t, best_gain


@njit(cache=True)
def _best_shared_split_counts_with_noise_py(
    hg, hh, hc, n_bins_per_feature, l2, feat_mask, min_child_weight,
    min_child_samples, n_leaves, random_strength, split_seed, tree_iteration,
    step, min_gain
):
    n_features = n_bins_per_feature.shape[0]
    best_f = 0
    best_t = -1
    best_gain = -np.inf
    best_score = -np.inf
    for f in range(n_features):
        if feat_mask[f] == 0:
            continue
        nb = int(n_bins_per_feature[f])
        gt = hg[f, :n_leaves, :nb].sum(axis=1)
        ht = hh[f, :n_leaves, :nb].sum(axis=1)
        ct = hc[f, :n_leaves, :nb].sum(axis=1)
        gl = np.zeros(n_leaves, dtype=np.float64)
        hl = np.zeros(n_leaves, dtype=np.float64)
        cl = np.zeros(n_leaves, dtype=np.float64)
        parent = np.zeros(n_leaves, dtype=np.float64)
        ok_parent = (ht > 0.0) & ((ht + l2) > 0.0)
        parent[ok_parent] = gt[ok_parent] * gt[ok_parent] / (ht[ok_parent] + l2)
        for t in range(nb - 1):
            gl += hg[f, :n_leaves, t]
            hl += hh[f, :n_leaves, t]
            cl += hc[f, :n_leaves, t]
            gain = 0.0
            legal = True
            any_nonempty = False
            for l in range(n_leaves):
                if ct[l] <= 0.0:
                    continue
                any_nonempty = True
                hr = ht[l] - hl[l]
                cr = ct[l] - cl[l]
                left_denom = hl[l] + l2
                right_denom = hr + l2
                parent_denom = ht[l] + l2
                if (
                    hl[l] < min_child_weight
                    or hr < min_child_weight
                    or cl[l] < min_child_samples
                    or cr < min_child_samples
                    or left_denom <= 0.0
                    or right_denom <= 0.0
                    or parent_denom <= 0.0
                ):
                    legal = False
                    break
                gr = gt[l] - gl[l]
                gain += gl[l] * gl[l] / left_denom + gr * gr / right_denom - parent[l]
            if legal and any_nonempty and gain > min_gain:
                score = _noisy_score(
                    gain, random_strength, split_seed, tree_iteration, step,
                    -1, f, t
                )
                if score > best_score:
                    best_score = score
                    best_gain = gain
                    best_f = f
                    best_t = t
    return best_f, best_t, best_gain


@njit(cache=True)
def _best_splits_by_leaf_with_noise_py(hg, hh, n_bins_per_feature, l2,
                                       feat_mask, min_child_weight, n_leaves,
                                       out_feat, out_thr, out_gain,
                                       random_strength, split_seed,
                                       tree_iteration, step, min_gain):
    n_features = n_bins_per_feature.shape[0]
    for l in range(n_leaves):
        best_f = -1
        best_t = -1
        best_gain = -np.inf
        best_score = -np.inf
        for f in range(n_features):
            if feat_mask[f] == 0:
                continue
            nb = int(n_bins_per_feature[f])
            Gt = float(np.sum(hg[f, l, :nb]))
            Ht = float(np.sum(hh[f, l, :nb]))
            parent_denom = Ht + l2
            if Ht <= 0.0 or parent_denom <= 0.0:
                continue
            parent_gain = Gt * Gt / parent_denom
            GL = 0.0
            HL = 0.0
            for t in range(nb - 1):
                GL += hg[f, l, t]
                HL += hh[f, l, t]
                HR = Ht - HL
                if HL < min_child_weight or HR < min_child_weight:
                    continue
                left_denom = HL + l2
                right_denom = HR + l2
                if left_denom <= 0.0 or right_denom <= 0.0:
                    continue
                GR = Gt - GL
                gain = GL * GL / left_denom + GR * GR / right_denom - parent_gain
                if gain <= min_gain:
                    continue
                score = _noisy_score(
                    gain, random_strength, split_seed, tree_iteration, step,
                    l, f, t
                )
                if score > best_score:
                    best_score = score
                    best_gain = gain
                    best_f = f
                    best_t = t
        out_feat[l] = best_f
        out_thr[l] = best_t
        out_gain[l] = best_gain


@njit(cache=True)
def _best_splits_counts_for_leaf_ids_with_noise_py(
    hg, hh, hc, n_bins_per_feature, l2, feat_mask, min_child_weight,
    min_child_samples, leaf_ids, n_leaf_ids, out_feat, out_thr, out_gain,
    random_strength, split_seed, tree_iteration, step, min_gain
):
    n_features = n_bins_per_feature.shape[0]
    for idx in range(n_leaf_ids):
        l = int(leaf_ids[idx])
        best_f = -1
        best_t = -1
        best_gain = -np.inf
        best_score = -np.inf
        for f in range(n_features):
            if feat_mask[f] == 0:
                continue
            nb = int(n_bins_per_feature[f])
            Gt = float(np.sum(hg[f, l, :nb]))
            Ht = float(np.sum(hh[f, l, :nb]))
            Ct = float(np.sum(hc[f, l, :nb]))
            parent_denom = Ht + l2
            if Ht <= 0.0 or Ct <= 0.0 or parent_denom <= 0.0:
                continue
            parent_gain = Gt * Gt / parent_denom
            GL = 0.0
            HL = 0.0
            CL = 0.0
            for t in range(nb - 1):
                GL += hg[f, l, t]
                HL += hh[f, l, t]
                CL += hc[f, l, t]
                HR = Ht - HL
                CR = Ct - CL
                if (
                    HL < min_child_weight
                    or HR < min_child_weight
                    or CL < min_child_samples
                    or CR < min_child_samples
                ):
                    continue
                left_denom = HL + l2
                right_denom = HR + l2
                if left_denom <= 0.0 or right_denom <= 0.0:
                    continue
                GR = Gt - GL
                gain = GL * GL / left_denom + GR * GR / right_denom - parent_gain
                if gain <= min_gain:
                    continue
                score = _noisy_score(
                    gain, random_strength, split_seed, tree_iteration, step,
                    l, f, t
                )
                if score > best_score:
                    best_score = score
                    best_gain = gain
                    best_f = f
                    best_t = t
        out_feat[l] = best_f
        out_thr[l] = best_t
        out_gain[l] = best_gain


@njit(cache=True)
def _best_multiclass_splits_counts_for_leaf_ids_with_noise_py(
    hg, hh, hc, n_bins_per_feature, l2, feat_mask, min_child_weight,
    min_child_samples, leaf_ids, n_leaf_ids, out_feat, out_thr, out_gain,
    random_strength, split_seed, tree_iteration, step, min_gain
):
    K = hg.shape[0]
    n_features = hg.shape[1]
    for idx in range(n_leaf_ids):
        l = int(leaf_ids[idx])
        best_f = -1
        best_t = -1
        best_gain = -np.inf
        best_score = -np.inf
        for f in range(n_features):
            if feat_mask[f] == 0:
                continue
            nb = int(n_bins_per_feature[f])
            Ct = float(np.sum(hc[f, l, :nb]))
            if Ct <= 0.0:
                continue
            Gt = np.sum(hg[:K, f, l, :nb], axis=1)
            Ht = np.sum(hh[:K, f, l, :nb], axis=1)
            total_H = float(np.sum(Ht))
            if total_H <= 0.0:
                continue
            GL = np.zeros(K, dtype=np.float64)
            HL = np.zeros(K, dtype=np.float64)
            CL = 0.0
            for t in range(nb - 1):
                CL += hc[f, l, t]
                GL += hg[:K, f, l, t]
                HL += hh[:K, f, l, t]
                CR = Ct - CL
                sum_HL = float(np.sum(HL))
                if (
                    CL < min_child_samples
                    or CR < min_child_samples
                    or sum_HL < min_child_weight
                    or total_H - sum_HL < min_child_weight
                ):
                    continue
                split_gain = 0.0
                for k in range(K):
                    if Ht[k] <= 0.0:
                        continue
                    HR = Ht[k] - HL[k]
                    l2k = l2[k]
                    parent_denom = Ht[k] + l2k
                    left_denom = HL[k] + l2k
                    right_denom = HR + l2k
                    if (
                        parent_denom <= 0.0
                        or left_denom <= 0.0
                        or right_denom <= 0.0
                    ):
                        continue
                    GR = Gt[k] - GL[k]
                    split_gain += (
                        GL[k] * GL[k] / left_denom
                        + GR * GR / right_denom
                        - Gt[k] * Gt[k] / parent_denom
                    )
                if split_gain > min_gain:
                    score = _noisy_score(
                        split_gain, random_strength, split_seed,
                        tree_iteration, step, l, f, t
                    )
                    if score > best_score:
                        best_score = score
                        best_gain = split_gain
                        best_f = f
                        best_t = t
        out_feat[l] = best_f
        out_thr[l] = best_t
        out_gain[l] = best_gain


@njit(cache=True)
def _best_multiclass_splits_counts_for_leaf_ids_with_noise_class_minor_py(
    hg, hh, hc, n_bins_per_feature, l2, feat_mask, min_child_weight,
    min_child_samples, leaf_ids, n_leaf_ids, out_feat, out_thr, out_gain,
    random_strength, split_seed, tree_iteration, step, min_gain
):
    n_features = n_bins_per_feature.shape[0]
    K = hg.shape[3]
    for idx in range(n_leaf_ids):
        l = int(leaf_ids[idx])
        best_f = -1
        best_t = -1
        best_gain = -np.inf
        best_score = -np.inf
        for f in range(n_features):
            if feat_mask[f] == 0:
                continue
            nb = int(n_bins_per_feature[f])
            Ct = 0.0
            Gt = np.zeros(K, dtype=np.float64)
            Ht = np.zeros(K, dtype=np.float64)
            for b in range(nb):
                Ct += hc[f, l, b]
                for k in range(K):
                    Gt[k] += hg[f, l, b, k]
                    Ht[k] += hh[f, l, b, k]
            if Ct <= 0.0:
                continue
            total_H = 0.0
            for k in range(K):
                total_H += Ht[k]
            if total_H <= 0.0:
                continue
            GL = np.zeros(K, dtype=np.float64)
            HL = np.zeros(K, dtype=np.float64)
            CL = 0.0
            for t in range(nb - 1):
                CL += hc[f, l, t]
                sum_HL = 0.0
                for k in range(K):
                    GL[k] += hg[f, l, t, k]
                    HL[k] += hh[f, l, t, k]
                    sum_HL += HL[k]
                CR = Ct - CL
                if (
                    CL < min_child_samples
                    or CR < min_child_samples
                    or sum_HL < min_child_weight
                    or total_H - sum_HL < min_child_weight
                ):
                    continue
                split_gain = 0.0
                for k in range(K):
                    if Ht[k] <= 0.0:
                        continue
                    HR = Ht[k] - HL[k]
                    l2k = l2[k]
                    parent_denom = Ht[k] + l2k
                    left_denom = HL[k] + l2k
                    right_denom = HR + l2k
                    if (
                        parent_denom <= 0.0
                        or left_denom <= 0.0
                        or right_denom <= 0.0
                    ):
                        continue
                    GR = Gt[k] - GL[k]
                    split_gain += (
                        GL[k] * GL[k] / left_denom
                        + GR * GR / right_denom
                        - Gt[k] * Gt[k] / parent_denom
                    )
                if split_gain > min_gain:
                    score = _noisy_score(
                        split_gain, random_strength, split_seed,
                        tree_iteration, step, l, f, t
                    )
                    if score > best_score:
                        best_score = score
                        best_gain = split_gain
                        best_f = f
                        best_t = t
        out_feat[l] = best_f
        out_thr[l] = best_t
        out_gain[l] = best_gain


class ObliviousTree:
    """A single symmetric tree. Stores its splits and leaf values."""

    __slots__ = ("splits_feat", "splits_thr", "values", "gains", "depth")

    def __init__(self, splits_feat, splits_thr, values, gains=None):
        self.splits_feat = splits_feat
        self.splits_thr = splits_thr
        self.values = values
        self.gains = gains if gains is not None else np.zeros(len(splits_feat))
        self.depth = len(splits_feat)

    def apply(self, X_binned):
        """Return the leaf index of each row."""
        if self.depth == 0:
            return np.zeros(X_binned.shape[0], dtype=np.int64)
        return _assign_leaves(X_binned, self.splits_feat, self.splits_thr)

    def predict(self, X_binned):
        if self.depth == 0:
            return np.zeros(X_binned.shape[0], dtype=np.float64)
        return _predict_tree(X_binned, self.splits_feat, self.splits_thr, self.values)

    def add_predict(self, X_binned, out):
        """Add this tree's prediction into an existing output vector."""
        if self.depth > 0:
            _predict_tree_add(
                X_binned, self.splits_feat, self.splits_thr, self.values, out
            )


class LevelwiseTree:
    """A level-wise tree with one feature/threshold decision per active node."""

    __slots__ = (
        "node_features", "node_thresholds", "values", "splits_feat",
        "splits_thr", "gains", "depth",
    )

    def __init__(self, node_features, node_thresholds, values, splits_feat,
                 splits_thr, gains):
        self.node_features = node_features
        self.node_thresholds = node_thresholds
        self.values = values
        self.splits_feat = splits_feat
        self.splits_thr = splits_thr
        self.gains = gains
        self.depth = node_features.shape[0]

    def apply(self, X_binned):
        if self.depth == 0:
            return np.zeros(X_binned.shape[0], dtype=np.int64)
        return _assign_levelwise_leaves(
            X_binned, self.node_features, self.node_thresholds
        )

    def predict(self, X_binned):
        if self.depth == 0:
            return np.zeros(X_binned.shape[0], dtype=np.float64)
        return _predict_levelwise_tree(
            X_binned, self.node_features, self.node_thresholds, self.values
        )

    def add_predict(self, X_binned, out):
        if self.depth > 0:
            _predict_levelwise_tree_add(
                X_binned, self.node_features, self.node_thresholds,
                self.values, out
            )


class NonObliviousTree:
    """A CART-style tree with explicit nodes and one split per internal node."""

    __slots__ = (
        "features", "thresholds", "left_child", "right_child", "leaf_index",
        "values", "splits_feat", "splits_thr", "gains", "depth",
        "n_leaves", "n_splits",
    )

    def __init__(self, features, thresholds, left_child, right_child,
                 leaf_index, values, splits_feat, splits_thr, gains, depth,
                 n_leaves):
        self.features = features
        self.thresholds = thresholds
        self.left_child = left_child
        self.right_child = right_child
        self.leaf_index = leaf_index
        self.values = values
        self.splits_feat = splits_feat
        self.splits_thr = splits_thr
        self.gains = gains
        self.depth = int(depth)
        self.n_leaves = int(n_leaves)
        self.n_splits = int(len(splits_feat))

    def apply(self, X_binned):
        if self.n_splits == 0:
            return np.zeros(X_binned.shape[0], dtype=np.int64)
        return _assign_non_oblivious_leaves(
            X_binned, self.features, self.thresholds, self.left_child,
            self.right_child, self.leaf_index
        )

    def predict(self, X_binned):
        if self.n_splits == 0:
            return np.full(X_binned.shape[0], self.values[0], dtype=np.float64)
        return _predict_non_oblivious_tree(
            X_binned, self.features, self.thresholds, self.left_child,
            self.right_child, self.leaf_index, self.values
        )

    def add_predict(self, X_binned, out):
        if self.n_splits == 0:
            out += self.values[0]
        elif get_num_threads() > 1 and X_binned.shape[0] >= 8192:
            _predict_non_oblivious_tree_add_parallel(
                X_binned, self.features, self.thresholds, self.left_child,
                self.right_child, self.leaf_index, self.values, out
            )
        else:
            _predict_non_oblivious_tree_add(
                X_binned, self.features, self.thresholds, self.left_child,
                self.right_child, self.leaf_index, self.values, out
            )


class MultiNonObliviousTree:
    """A shared-structure tree with one vector leaf value per class."""

    __slots__ = (
        "features", "thresholds", "left_child", "right_child", "leaf_index",
        "values", "splits_feat", "splits_thr", "gains", "depth",
        "n_leaves", "n_splits",
    )

    def __init__(self, features, thresholds, left_child, right_child,
                 leaf_index, values, splits_feat, splits_thr, gains, depth,
                 n_leaves):
        self.features = features
        self.thresholds = thresholds
        self.left_child = left_child
        self.right_child = right_child
        self.leaf_index = leaf_index
        self.values = values
        self.splits_feat = splits_feat
        self.splits_thr = splits_thr
        self.gains = gains
        self.depth = int(depth)
        self.n_leaves = int(n_leaves)
        self.n_splits = int(len(splits_feat))

    def apply(self, X_binned):
        if self.n_splits == 0:
            return np.zeros(X_binned.shape[0], dtype=np.int64)
        return _assign_non_oblivious_leaves(
            X_binned, self.features, self.thresholds, self.left_child,
            self.right_child, self.leaf_index
        )

    def add_predict_class_major(self, X_binned, out):
        if self.n_splits == 0:
            out += self.values[0][:, None]
        elif get_num_threads() > 1 and X_binned.shape[0] >= 8192:
            _predict_non_oblivious_multiclass_tree_add_parallel(
                X_binned, self.features, self.thresholds, self.left_child,
                self.right_child, self.leaf_index, self.values, out
            )
        else:
            _predict_non_oblivious_multiclass_tree_add(
                X_binned, self.features, self.thresholds, self.left_child,
                self.right_child, self.leaf_index, self.values, out
            )


def build_oblivious_tree(X_binned, grad, hess, n_bins_per_feature,
                         max_depth, l2, lr, min_gain=1e-8, feature_mask=None,
                         min_child_weight=1.0, hist_buffers=None,
                         split_buffers=None,
                         return_training_state=False, X_hist_binned=None,
                         X_route_binned=None,
                         feature_indices=None, row_indices=None,
                         constant_hessian=False, rowpar_buffers=None,
                         level_histogram_subtraction="auto",
                         root_histograms=None, random_strength=0.0,
                         split_seed=0, tree_iteration=0,
                         leaf_dtype="int64"):
    """Grow one oblivious tree level by level and return an ObliviousTree.

    X_hist_binned: optional feature-contiguous view/copy of X_binned used only
    by histogram builders.
    X_route_binned: optional value-identical view/copy of X_binned used only by
    fixed-column leaf routing. Direct callers may leave this unset; callers
    that provide a different memory layout must guarantee elementwise equality.
    feature_mask: optional 0/1 array over features; 0 disables a feature for
    this tree (column subsampling). None means all features are eligible.
    feature_indices: optional selected column indices matching feature_mask;
    when supplied, histogram building zeroes and fills only those columns.
    row_indices: optional selected row indices for stochastic subsampling;
    histograms scan only these rows, while leaf routing still updates all rows.
    constant_hessian: if True, histogram building treats every scanned row's
    hessian as 1.0 and skips hess[i] loads. The caller must still pass a hess
    vector matching the same unit-Hessian semantics for final leaf values.
    min_child_weight: minimum hessian mass each side of a split must retain in
    every non-empty leaf. Stops the tree growing once no legal split remains,
    which prevents sparse-leaf overfitting at higher depth.
    hist_buffers: optional (hg, hh) arrays of shape (n_features, 2**max_depth,
    max_bins) reused across trees to avoid per-level allocation. If None, they
    are allocated here (convenient for one-off calls and tests).
    split_buffers: optional five-array tuple of shape
    (n_features, 2**max_depth) reused by the threaded split search.
    rowpar_buffers: optional (lg, lh) thread-local accumulators of shape
    (n_chunks, n_features, leaf_slots, max_bins). When supplied (and the
    full-row/full-feature lane is active with enough rows per level), the
    histogram fill switches to the row-parallel kernels, which read
    grad/hess/leaf once instead of once per feature.
    level_histogram_subtraction: derive each level >= 1 from the cached
    parent histograms by scanning only every parent's smaller child and
    subtracting (full-row/full-feature lane only). Histogram contents match
    the full rebuild to float64 rounding; empty children are exact. "auto"
    enables it at <= 2 threads, where it measured 13-45% faster; True/False
    force it on/off.
    root_histograms: optional (grad, hess[, ...]) (n_features, max_bins)
    arrays holding the precomputed root histograms (e.g. from one fused
    class-major pass in the multiclass booster). When supplied in the
    full-row/full-feature lane, level 0 copies them instead of scanning.
    """
    if X_hist_binned is None:
        X_hist_binned = X_binned
    elif X_hist_binned.shape != X_binned.shape:
        raise ValueError("X_hist_binned must have the same shape as X_binned")
    if X_route_binned is None:
        X_route_binned = X_binned
    elif X_route_binned.shape != X_binned.shape:
        raise ValueError("X_route_binned must have the same shape as X_binned")
    leaf_dtype = _normalize_leaf_dtype(leaf_dtype)
    n_samples = X_binned.shape[0]
    n_features = X_binned.shape[1]
    max_bins = n_features and int(n_bins_per_feature.max())
    if row_indices is not None:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if row_indices.ndim != 1:
            raise ValueError("row_indices must be a 1-D array")
        if np.any((row_indices < 0) | (row_indices >= n_samples)):
            raise ValueError("row_indices contains out-of-range rows")
    if feature_indices is not None:
        feature_indices = np.asarray(feature_indices, dtype=np.int64)
        if feature_indices.ndim != 1:
            raise ValueError("feature_indices must be a 1-D array")
        if np.any((feature_indices < 0) | (feature_indices >= n_features)):
            raise ValueError("feature_indices contains out-of-range columns")
        if np.unique(feature_indices).shape[0] != feature_indices.shape[0]:
            raise ValueError("feature_indices must be unique")

        selected_mask = np.zeros(n_features, dtype=np.int64)
        selected_mask[feature_indices] = 1
        if feature_mask is None:
            feature_mask = selected_mask
        else:
            feature_mask = np.asarray(feature_mask, dtype=np.int64)
            if feature_mask.shape != selected_mask.shape:
                raise ValueError("feature_mask must have one entry per feature")
            if not np.array_equal(feature_mask, selected_mask):
                raise ValueError("feature_indices must match feature_mask")
    elif feature_mask is None:
        feature_mask = np.ones(n_features, dtype=np.int64)
    else:
        feature_mask = np.asarray(feature_mask, dtype=np.int64)
        if feature_mask.shape != (n_features,):
            raise ValueError("feature_mask must have one entry per feature")
    max_leaves = 1 << max_depth
    if hist_buffers is None:
        hg = np.zeros((n_features, max_leaves, max_bins))
        hh = np.zeros((n_features, max_leaves, max_bins))
    else:
        if len(hist_buffers) != 2:
            raise ValueError("hist_buffers must contain gradient and hessian arrays")
        hg, hh = hist_buffers
        if hg.ndim != 3 or hh.ndim != 3 or hg.shape != hh.shape:
            raise ValueError("hist_buffers must be matching 3-D arrays")
        if (
            hg.shape[0] < n_features
            or hg.shape[1] < max_leaves
            or hg.shape[2] < max_bins
        ):
            raise ValueError("hist_buffers are too small")
    splits_feat = []
    splits_thr = []
    splits_gain = []
    leaf = np.zeros(X_binned.shape[0], dtype=leaf_dtype)
    use_serial_kernels = get_num_threads() == 1
    if rowpar_buffers is not None:
        if len(rowpar_buffers) < 2:
            raise ValueError("rowpar_buffers must contain at least two arrays")
        lg_rp, lh_rp = rowpar_buffers[0], rowpar_buffers[1]
        if (
            lg_rp.ndim != 4
            or lg_rp.shape != lh_rp.shape
            or lg_rp.shape[1] < n_features
            or lg_rp.shape[3] < max_bins
        ):
            raise ValueError("rowpar_buffers are too small")
    rowpar_full_lane = (
        rowpar_buffers is not None
        and not use_serial_kernels
        and row_indices is None
        and feature_indices is None
    )
    if use_serial_kernels:
        split_scratch = None
    elif split_buffers is None:
        split_scratch = (
            np.empty((n_features, max_leaves)),
            np.empty((n_features, max_leaves)),
            np.empty((n_features, max_leaves)),
            np.empty((n_features, max_leaves)),
            np.empty((n_features, max_leaves)),
        )
    else:
        if len(split_buffers) != 5:
            raise ValueError("split_buffers must contain five scratch arrays")
        for buf in split_buffers:
            if buf.shape[0] < n_features or buf.shape[1] < max_leaves:
                raise ValueError("split_buffers are too small")
        split_scratch = split_buffers

    subtract_lane = (
        _resolve_level_subtraction(level_histogram_subtraction)
        and row_indices is None
        and feature_indices is None
    )
    root_copy_lane = (
        root_histograms is not None
        and row_indices is None
        and feature_indices is None
    )

    for d in range(max_depth):
        n_leaves = 1 << d
        subtract_level = subtract_lane and d >= 1
        root_copy = root_copy_lane and d == 0
        if root_copy:
            hg[:n_features, 0, :max_bins] = root_histograms[0]
            hh[:n_features, 0, :max_bins] = root_histograms[1]
        if subtract_level:
            scan_idx, scan_side = _level_scan_plan(leaf, n_leaves >> 1)
        if use_serial_kernels:
            if root_copy:
                pass
            elif subtract_level:
                if constant_hessian:
                    _build_level_histograms_subtract_unit_hess_into_serial(
                        X_binned, grad, leaf, scan_idx, scan_side,
                        n_leaves >> 1, hg, hh
                    )
                else:
                    _build_level_histograms_subtract_into_serial(
                        X_binned, grad, hess, leaf, scan_idx, scan_side,
                        n_leaves >> 1, hg, hh
                    )
            elif constant_hessian and row_indices is None and feature_indices is None:
                _build_histograms_unit_hess_into_serial(
                    X_binned, grad, leaf, n_leaves, hg, hh
                )
            elif constant_hessian and row_indices is None:
                _build_histograms_selected_unit_hess_into_serial(
                    X_binned, grad, leaf, n_leaves, hg, hh,
                    feature_indices
                )
            elif constant_hessian and feature_indices is None:
                _build_histograms_rows_unit_hess_into_serial(
                    X_binned, grad, leaf, n_leaves, hg, hh,
                    row_indices
                )
            elif constant_hessian:
                _build_histograms_selected_rows_unit_hess_into_serial(
                    X_binned, grad, leaf, n_leaves, hg, hh,
                    feature_indices, row_indices
                )
            elif row_indices is None and feature_indices is None:
                _build_histograms_into_serial(
                    X_binned, grad, hess, leaf, n_leaves, hg, hh
                )
            elif row_indices is None:
                _build_histograms_selected_into_serial(
                    X_binned, grad, hess, leaf, n_leaves, hg, hh,
                    feature_indices
                )
            elif feature_indices is None:
                _build_histograms_rows_into_serial(
                    X_binned, grad, hess, leaf, n_leaves, hg, hh,
                    row_indices
                )
            else:
                _build_histograms_selected_rows_into_serial(
                    X_binned, grad, hess, leaf, n_leaves, hg, hh,
                    feature_indices, row_indices
                )
            if random_strength > 0.0:
                f, t, gain = _best_split_with_noise_py(
                    hg, hh, n_bins_per_feature, l2, feature_mask,
                    min_child_weight, n_leaves, random_strength, split_seed,
                    tree_iteration, d, min_gain
                )
            else:
                f, t, gain = _best_split_serial(
                    hg, hh, n_bins_per_feature, l2, feature_mask,
                    min_child_weight, n_leaves
                )
        else:
            if root_copy:
                pass
            elif subtract_level:
                if constant_hessian:
                    _build_level_histograms_subtract_unit_hess_into(
                        X_hist_binned, grad, leaf, scan_idx, scan_side,
                        n_leaves >> 1, hg, hh
                    )
                else:
                    _build_level_histograms_subtract_into(
                        X_hist_binned, grad, hess, leaf, scan_idx, scan_side,
                        n_leaves >> 1, hg, hh
                    )
            elif rowpar_full_lane and _rowpar_eligible(
                n_samples, rowpar_buffers, n_leaves, max_bins
            ):
                if constant_hessian:
                    _build_histograms_unit_hess_rowpar_into(
                        X_binned, grad, leaf, n_leaves, hg, hh,
                        rowpar_buffers[0], rowpar_buffers[1]
                    )
                else:
                    _build_histograms_rowpar_into(
                        X_binned, grad, hess, leaf, n_leaves, hg, hh,
                        rowpar_buffers[0], rowpar_buffers[1]
                    )
            elif constant_hessian and row_indices is None and feature_indices is None:
                _build_histograms_unit_hess_into(
                    X_hist_binned, grad, leaf, n_leaves, hg, hh
                )
            elif constant_hessian and row_indices is None:
                _build_histograms_selected_unit_hess_into(
                    X_hist_binned, grad, leaf, n_leaves, hg, hh,
                    feature_indices
                )
            elif constant_hessian and feature_indices is None:
                _build_histograms_rows_unit_hess_into(
                    X_hist_binned, grad, leaf, n_leaves, hg, hh,
                    row_indices
                )
            elif constant_hessian:
                _build_histograms_selected_rows_unit_hess_into(
                    X_hist_binned, grad, leaf, n_leaves, hg, hh,
                    feature_indices, row_indices
                )
            elif row_indices is None and feature_indices is None:
                _build_histograms_into(
                    X_hist_binned, grad, hess, leaf, n_leaves, hg, hh
                )
            elif row_indices is None:
                _build_histograms_selected_into(
                    X_hist_binned, grad, hess, leaf, n_leaves, hg, hh,
                    feature_indices
                )
            elif feature_indices is None:
                _build_histograms_rows_into(
                    X_hist_binned, grad, hess, leaf, n_leaves, hg, hh,
                    row_indices
                )
            else:
                _build_histograms_selected_rows_into(
                    X_hist_binned, grad, hess, leaf, n_leaves, hg, hh,
                    feature_indices, row_indices
                )
            if random_strength > 0.0:
                f, t, gain = _best_split_with_noise_py(
                    hg, hh, n_bins_per_feature, l2, feature_mask,
                    min_child_weight, n_leaves, random_strength, split_seed,
                    tree_iteration, d, min_gain
                )
            else:
                f, t, gain = _best_split(
                    hg, hh, n_bins_per_feature, l2, feature_mask,
                    min_child_weight, n_leaves, split_scratch[0],
                    split_scratch[1], split_scratch[2], split_scratch[3],
                    split_scratch[4]
                )
        if gain <= min_gain or t < 0:
            break
        splits_feat.append(f)
        splits_thr.append(t)
        splits_gain.append(gain)
        _update_leaves_with_split(X_route_binned, leaf, f, t)

    sf = np.array(splits_feat, dtype=np.int64)
    st = np.array(splits_thr, dtype=np.int64)
    n_leaves = 1 << len(splits_feat)
    if row_indices is None:
        values, leaf_G, leaf_H = _leaf_values_and_sums(
            leaf, grad, hess, n_leaves, l2, lr
        )
    else:
        values, leaf_G, leaf_H = _leaf_values_and_sums_rows(
            leaf, grad, hess, row_indices, n_leaves, l2, lr
        )
    tree = ObliviousTree(sf, st, values, np.array(splits_gain, dtype=np.float64))
    if return_training_state:
        return tree, leaf, leaf_G, leaf_H
    return tree


def build_levelwise_tree(X_binned, grad, hess, n_bins_per_feature,
                         max_depth, l2, lr, min_gain=1e-8, feature_mask=None,
                         min_child_weight=1.0, hist_buffers=None,
                         split_buffers=None,
                         return_training_state=False, X_hist_binned=None,
                         X_route_binned=None,
                         feature_indices=None, row_indices=None,
                         constant_hessian=False, rowpar_buffers=None,
                         level_histogram_subtraction="auto",
                         random_strength=0.0, split_seed=0,
                         tree_iteration=0, leaf_dtype="int64"):
    """Grow a level-wise, non-oblivious tree.

    Unlike ``build_oblivious_tree``, this builder chooses one best split per
    active leaf at each depth. It intentionally accepts the same call surface as
    the oblivious builder so it can remain available as an experimental
    depth-wise mode.
    """
    del split_buffers
    if X_hist_binned is None:
        X_hist_binned = X_binned
    elif X_hist_binned.shape != X_binned.shape:
        raise ValueError("X_hist_binned must have the same shape as X_binned")
    if X_route_binned is None:
        X_route_binned = X_binned
    elif X_route_binned.shape != X_binned.shape:
        raise ValueError("X_route_binned must have the same shape as X_binned")
    leaf_dtype = _normalize_leaf_dtype(leaf_dtype)
    n_samples = X_binned.shape[0]
    n_features = X_binned.shape[1]
    max_bins = n_features and int(n_bins_per_feature.max())
    if row_indices is not None:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if row_indices.ndim != 1:
            raise ValueError("row_indices must be a 1-D array")
        if np.any((row_indices < 0) | (row_indices >= n_samples)):
            raise ValueError("row_indices contains out-of-range rows")
    if feature_indices is not None:
        feature_indices = np.asarray(feature_indices, dtype=np.int64)
        if feature_indices.ndim != 1:
            raise ValueError("feature_indices must be a 1-D array")
        if np.any((feature_indices < 0) | (feature_indices >= n_features)):
            raise ValueError("feature_indices contains out-of-range columns")
        if np.unique(feature_indices).shape[0] != feature_indices.shape[0]:
            raise ValueError("feature_indices must be unique")

        selected_mask = np.zeros(n_features, dtype=np.int64)
        selected_mask[feature_indices] = 1
        if feature_mask is None:
            feature_mask = selected_mask
        else:
            feature_mask = np.asarray(feature_mask, dtype=np.int64)
            if feature_mask.shape != selected_mask.shape:
                raise ValueError("feature_mask must have one entry per feature")
            if not np.array_equal(feature_mask, selected_mask):
                raise ValueError("feature_indices must match feature_mask")
    elif feature_mask is None:
        feature_mask = np.ones(n_features, dtype=np.int64)
    else:
        feature_mask = np.asarray(feature_mask, dtype=np.int64)
        if feature_mask.shape != (n_features,):
            raise ValueError("feature_mask must have one entry per feature")

    max_leaves = 1 << max_depth
    if hist_buffers is None:
        hg = np.zeros((n_features, max_leaves, max_bins))
        hh = np.zeros((n_features, max_leaves, max_bins))
    else:
        if len(hist_buffers) != 2:
            raise ValueError("hist_buffers must contain gradient and hessian arrays")
        hg, hh = hist_buffers
        if hg.ndim != 3 or hh.ndim != 3 or hg.shape != hh.shape:
            raise ValueError("hist_buffers must be matching 3-D arrays")
        if (
            hg.shape[0] < n_features
            or hg.shape[1] < max_leaves
            or hg.shape[2] < max_bins
        ):
            raise ValueError("hist_buffers are too small")

    node_features = np.full((max_depth, max_leaves), -1, dtype=np.int64)
    node_thresholds = np.full((max_depth, max_leaves), -1, dtype=np.int64)
    leaf_best_feat = np.empty(max_leaves, dtype=np.int64)
    leaf_best_thr = np.empty(max_leaves, dtype=np.int64)
    leaf_best_gain = np.empty(max_leaves, dtype=np.float64)
    flat_features = []
    flat_thresholds = []
    flat_gains = []
    leaf = np.zeros(X_binned.shape[0], dtype=leaf_dtype)
    use_serial_kernels = get_num_threads() == 1
    rowpar_full_lane = (
        rowpar_buffers is not None
        and not use_serial_kernels
        and row_indices is None
        and feature_indices is None
    )
    subtract_lane = (
        _resolve_level_subtraction(level_histogram_subtraction)
        and row_indices is None
        and feature_indices is None
    )
    actual_depth = 0

    for d in range(max_depth):
        n_leaves = 1 << d
        subtract_level = subtract_lane and d >= 1
        if subtract_level:
            scan_idx, scan_side = _level_scan_plan(leaf, n_leaves >> 1)
        if use_serial_kernels:
            if subtract_level:
                if constant_hessian:
                    _build_level_histograms_subtract_unit_hess_into_serial(
                        X_binned, grad, leaf, scan_idx, scan_side,
                        n_leaves >> 1, hg, hh
                    )
                else:
                    _build_level_histograms_subtract_into_serial(
                        X_binned, grad, hess, leaf, scan_idx, scan_side,
                        n_leaves >> 1, hg, hh
                    )
            elif constant_hessian and row_indices is None and feature_indices is None:
                _build_histograms_unit_hess_into_serial(
                    X_binned, grad, leaf, n_leaves, hg, hh
                )
            elif constant_hessian and row_indices is None:
                _build_histograms_selected_unit_hess_into_serial(
                    X_binned, grad, leaf, n_leaves, hg, hh,
                    feature_indices
                )
            elif constant_hessian and feature_indices is None:
                _build_histograms_rows_unit_hess_into_serial(
                    X_binned, grad, leaf, n_leaves, hg, hh,
                    row_indices
                )
            elif constant_hessian:
                _build_histograms_selected_rows_unit_hess_into_serial(
                    X_binned, grad, leaf, n_leaves, hg, hh,
                    feature_indices, row_indices
                )
            elif row_indices is None and feature_indices is None:
                _build_histograms_into_serial(
                    X_binned, grad, hess, leaf, n_leaves, hg, hh
                )
            elif row_indices is None:
                _build_histograms_selected_into_serial(
                    X_binned, grad, hess, leaf, n_leaves, hg, hh,
                    feature_indices
                )
            elif feature_indices is None:
                _build_histograms_rows_into_serial(
                    X_binned, grad, hess, leaf, n_leaves, hg, hh,
                    row_indices
                )
            else:
                _build_histograms_selected_rows_into_serial(
                    X_binned, grad, hess, leaf, n_leaves, hg, hh,
                    feature_indices, row_indices
                )
        else:
            if subtract_level:
                if constant_hessian:
                    _build_level_histograms_subtract_unit_hess_into(
                        X_hist_binned, grad, leaf, scan_idx, scan_side,
                        n_leaves >> 1, hg, hh
                    )
                else:
                    _build_level_histograms_subtract_into(
                        X_hist_binned, grad, hess, leaf, scan_idx, scan_side,
                        n_leaves >> 1, hg, hh
                    )
            elif rowpar_full_lane and _rowpar_eligible(
                n_samples, rowpar_buffers, n_leaves, max_bins
            ):
                if constant_hessian:
                    _build_histograms_unit_hess_rowpar_into(
                        X_binned, grad, leaf, n_leaves, hg, hh,
                        rowpar_buffers[0], rowpar_buffers[1]
                    )
                else:
                    _build_histograms_rowpar_into(
                        X_binned, grad, hess, leaf, n_leaves, hg, hh,
                        rowpar_buffers[0], rowpar_buffers[1]
                    )
            elif constant_hessian and row_indices is None and feature_indices is None:
                _build_histograms_unit_hess_into(
                    X_hist_binned, grad, leaf, n_leaves, hg, hh
                )
            elif constant_hessian and row_indices is None:
                _build_histograms_selected_unit_hess_into(
                    X_hist_binned, grad, leaf, n_leaves, hg, hh,
                    feature_indices
                )
            elif constant_hessian and feature_indices is None:
                _build_histograms_rows_unit_hess_into(
                    X_hist_binned, grad, leaf, n_leaves, hg, hh,
                    row_indices
                )
            elif constant_hessian:
                _build_histograms_selected_rows_unit_hess_into(
                    X_hist_binned, grad, leaf, n_leaves, hg, hh,
                    feature_indices, row_indices
                )
            elif row_indices is None and feature_indices is None:
                _build_histograms_into(
                    X_hist_binned, grad, hess, leaf, n_leaves, hg, hh
                )
            elif row_indices is None:
                _build_histograms_selected_into(
                    X_hist_binned, grad, hess, leaf, n_leaves, hg, hh,
                    feature_indices
                )
            elif feature_indices is None:
                _build_histograms_rows_into(
                    X_hist_binned, grad, hess, leaf, n_leaves, hg, hh,
                    row_indices
                )
            else:
                _build_histograms_selected_rows_into(
                    X_hist_binned, grad, hess, leaf, n_leaves, hg, hh,
                    feature_indices, row_indices
                )

        if random_strength > 0.0:
            _best_splits_by_leaf_with_noise_py(
                hg, hh, n_bins_per_feature, l2, feature_mask,
                min_child_weight, n_leaves, leaf_best_feat, leaf_best_thr,
                leaf_best_gain, random_strength, split_seed, tree_iteration, d,
                min_gain
            )
        else:
            _best_splits_by_leaf(
                hg, hh, n_bins_per_feature, l2, feature_mask, min_child_weight,
                n_leaves, leaf_best_feat, leaf_best_thr, leaf_best_gain
            )

        any_split = False
        for l in range(n_leaves):
            if leaf_best_thr[l] >= 0 and leaf_best_gain[l] > min_gain:
                node_features[d, l] = leaf_best_feat[l]
                node_thresholds[d, l] = leaf_best_thr[l]
                flat_features.append(leaf_best_feat[l])
                flat_thresholds.append(leaf_best_thr[l])
                flat_gains.append(leaf_best_gain[l])
                any_split = True
        if not any_split:
            break
        actual_depth = d + 1
        _update_leaves_with_level_splits(
            X_route_binned, leaf, node_features[d], node_thresholds[d]
        )

    node_features = node_features[:actual_depth].copy()
    node_thresholds = node_thresholds[:actual_depth].copy()
    n_leaves = 1 << actual_depth
    if row_indices is None:
        values, leaf_G, leaf_H = _leaf_values_and_sums(
            leaf, grad, hess, n_leaves, l2, lr
        )
    else:
        values, leaf_G, leaf_H = _leaf_values_and_sums_rows(
            leaf, grad, hess, row_indices, n_leaves, l2, lr
        )
    tree = LevelwiseTree(
        node_features,
        node_thresholds,
        values,
        np.array(flat_features, dtype=np.int64),
        np.array(flat_thresholds, dtype=np.int64),
        np.array(flat_gains, dtype=np.float64),
    )
    if return_training_state:
        return tree, leaf, leaf_G, leaf_H
    return tree


def build_leafwise_tree(X_binned, grad, hess, n_bins_per_feature,
                        max_depth, l2, lr, min_gain=1e-8, feature_mask=None,
                        min_child_weight=1.0, hist_buffers=None,
                        split_buffers=None, return_training_state=False,
                        X_hist_binned=None, X_route_binned=None,
                        feature_indices=None,
                        row_indices=None, constant_hessian=False,
                        max_leaves=None, min_gain_to_split=None,
                        min_child_samples=20,
                        recompute_all_leaf_splits=False,
                        reuse_leaf_histograms=True,
                        hessian_always_positive=False,
                        leafwise_row_layout="auto",
                        fused_changed_leaf_scoring=False,
                        rowpar_buffers=None, root_histograms=None,
                        random_strength=0.0, split_seed=0,
                        tree_iteration=0, shared_trunk_depth=0,
                        leaf_dtype="int64"):
    """Grow a LightGBM-like leaf-wise, best-first non-oblivious tree.

    This builder chooses the best legal split for each current leaf, then splits
    only the leaf with the largest gain. It intentionally favors correctness and
    a clean tree contract over histogram reuse; speed optimizations belong after
    the public semantics are settled.
    """
    if min_gain_to_split is None:
        min_gain_to_split = min_gain
    if X_hist_binned is None:
        X_hist_binned = X_binned
    elif X_hist_binned.shape != X_binned.shape:
        raise ValueError("X_hist_binned must have the same shape as X_binned")
    if X_route_binned is None:
        X_route_binned = X_binned
    elif X_route_binned.shape != X_binned.shape:
        raise ValueError("X_route_binned must have the same shape as X_binned")
    leaf_dtype = _normalize_leaf_dtype(leaf_dtype)

    n_samples = X_binned.shape[0]
    n_features = X_binned.shape[1]
    max_bins = n_features and int(n_bins_per_feature.max())
    if max_leaves is None:
        if max_depth is None or max_depth < 0:
            max_leaves = 31
        else:
            max_leaves = min(31, 1 << int(max_depth))
    max_leaves = int(max_leaves)
    if max_leaves < 1:
        raise ValueError("max_leaves must be at least 1")
    max_depth_cap = -1 if max_depth is None else int(max_depth)
    if max_depth_cap == 0 and max_leaves > 1:
        raise ValueError("max_depth must be positive, None, or -1")
    min_child_samples = int(min_child_samples)
    if min_child_samples < 1:
        raise ValueError("min_child_samples must be at least 1")
    if leafwise_row_layout not in {"auto", "prefix", "segmented"}:
        raise ValueError("leafwise_row_layout must be 'auto', 'prefix', or 'segmented'")

    if row_indices is not None:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if row_indices.ndim != 1:
            raise ValueError("row_indices must be a 1-D array")
        if np.any((row_indices < 0) | (row_indices >= n_samples)):
            raise ValueError("row_indices contains out-of-range rows")
    if feature_indices is not None:
        feature_indices = np.asarray(feature_indices, dtype=np.int64)
        if feature_indices.ndim != 1:
            raise ValueError("feature_indices must be a 1-D array")
        if np.any((feature_indices < 0) | (feature_indices >= n_features)):
            raise ValueError("feature_indices contains out-of-range columns")
        if np.unique(feature_indices).shape[0] != feature_indices.shape[0]:
            raise ValueError("feature_indices must be unique")

        selected_mask = np.zeros(n_features, dtype=np.int64)
        selected_mask[feature_indices] = 1
        if feature_mask is None:
            feature_mask = selected_mask
        else:
            feature_mask = np.asarray(feature_mask, dtype=np.int64)
            if feature_mask.shape != selected_mask.shape:
                raise ValueError("feature_mask must have one entry per feature")
            if not np.array_equal(feature_mask, selected_mask):
                raise ValueError("feature_indices must match feature_mask")
    elif feature_mask is None:
        feature_mask = np.ones(n_features, dtype=np.int64)
    else:
        feature_mask = np.asarray(feature_mask, dtype=np.int64)
        if feature_mask.shape != (n_features,):
            raise ValueError("feature_mask must have one entry per feature")

    if hist_buffers is None:
        hg = np.zeros((n_features, max_leaves, max_bins))
        hh = np.zeros((n_features, max_leaves, max_bins))
        hc = np.zeros((n_features, max_leaves, max_bins))
    else:
        if len(hist_buffers) not in (2, 3):
            raise ValueError(
                "hist_buffers must contain gradient/hessian arrays, "
                "optionally plus counts"
            )
        hg, hh = hist_buffers[:2]
        if hg.ndim != 3 or hh.ndim != 3 or hg.shape != hh.shape:
            raise ValueError("hist_buffers must be matching 3-D arrays")
        if (
            hg.shape[0] < n_features
            or hg.shape[1] < max_leaves
            or hg.shape[2] < max_bins
        ):
            raise ValueError("hist_buffers are too small")
        if len(hist_buffers) == 3:
            hc = hist_buffers[2]
            if hc.ndim != 3 or hc.shape != hg.shape:
                raise ValueError("count hist_buffer must match gradient buffer")
        else:
            hc = np.zeros_like(hg)

    max_nodes = 2 * max_leaves - 1
    features = np.full(max_nodes, -1, dtype=np.int64)
    thresholds = np.full(max_nodes, -1, dtype=np.int64)
    left_child = np.full(max_nodes, -1, dtype=np.int64)
    right_child = np.full(max_nodes, -1, dtype=np.int64)
    leaf_index = np.full(max_nodes, -1, dtype=np.int64)
    leaf_node = np.empty(max_leaves, dtype=np.int64)
    leaf_depth = np.zeros(max_leaves, dtype=np.int64)
    leaf_node[0] = 0
    leaf_index[0] = 0

    best_feat = np.empty(max_leaves, dtype=np.int64)
    best_thr = np.empty(max_leaves, dtype=np.int64)
    best_gain = np.empty(max_leaves, dtype=np.float64)
    changed_leaves = np.empty(2, dtype=np.int64)
    changed_leaves[0] = 0
    n_changed_leaves = 1
    use_serial_kernels = get_num_threads() == 1
    if use_serial_kernels:
        split_scratch = None
    elif split_buffers is None:
        split_scratch = (
            np.empty((n_features, max_leaves)),
            np.empty((n_features, max_leaves)),
        )
    else:
        if len(split_buffers) < 2:
            raise ValueError("split_buffers must contain at least two scratch arrays")
        for buf in split_buffers[:2]:
            if buf.shape[0] < n_features or buf.shape[1] < max_leaves:
                raise ValueError("split_buffers are too small")
        split_scratch = split_buffers[:2]
    if row_indices is None:
        row_order = np.arange(n_samples, dtype=np.int64)
    else:
        row_order = row_indices.copy()
    row_scratch = np.empty(row_order.shape[0], dtype=np.int64)
    leaf_start = np.zeros(max_leaves + 1, dtype=np.int64)
    leaf_start[1] = row_order.shape[0]
    leaf_count = np.zeros(max_leaves, dtype=np.int64)
    leaf_count[0] = row_order.shape[0]
    if rowpar_buffers is not None:
        if len(rowpar_buffers) < 2 or (
            not constant_hessian and len(rowpar_buffers) < 3
        ):
            raise ValueError("rowpar_buffers are too small")
        lg_rp = rowpar_buffers[0]
        if (
            lg_rp.ndim != 4
            or lg_rp.shape[1] < n_features
            or lg_rp.shape[3] < max_bins
        ):
            raise ValueError("rowpar_buffers are too small")
    full_feature_positive_split_candidate = (
        hessian_always_positive
        and reuse_leaf_histograms
        and row_indices is None
        and feature_indices is None
        and bool(np.all(feature_mask != 0))
    )
    # Segment scans (the root build and child refills) read each row once, so
    # the row-parallel kernels apply whenever the full feature set is active;
    # subsampled rows are fine because the row_order segments already contain
    # only the scanned rows.
    rowpar_segment_lane_candidate = (
        rowpar_buffers is not None
        and not use_serial_kernels
        and feature_indices is None
    )
    fused_changed_leaf_scoring_candidate = (
        fused_changed_leaf_scoring
        and hessian_always_positive
        and not constant_hessian
        and row_indices is None
        and feature_indices is None
        and bool(np.all(feature_mask != 0))
        and split_scratch is not None
        and get_num_threads() > 2
    )
    resolved_leafwise_row_layout = _resolve_leafwise_row_layout(
        leafwise_row_layout,
        n_samples,
        n_features,
        row_indices,
        feature_indices,
        feature_mask,
        reuse_leaf_histograms,
        max_leaves,
        fast_lane_eligible=(
            full_feature_positive_split_candidate
            or rowpar_segment_lane_candidate
            or fused_changed_leaf_scoring_candidate
        ),
    )
    use_segmented_rows = resolved_leafwise_row_layout == "segmented"
    full_feature_positive_split = (
        full_feature_positive_split_candidate and not use_segmented_rows
    )
    rowpar_segment_lane = (
        rowpar_segment_lane_candidate and not use_segmented_rows
    )
    canonicalize_explicit_layout_gains = (
        leafwise_row_layout in {"prefix", "segmented"}
        and row_indices is None
        and feature_indices is None
        and random_strength == 0.0
    )
    split_features = []
    split_thresholds = []
    split_gains = []

    leaf = np.zeros(n_samples, dtype=leaf_dtype)
    n_nodes = 1
    n_leaves = 1
    actual_depth = 0
    can_reuse_leaf_histograms = reuse_leaf_histograms
    histograms_initialized = False
    shared_trunk_depth = int(shared_trunk_depth)
    if shared_trunk_depth < 0:
        raise ValueError("shared_trunk_depth must be nonnegative")

    if shared_trunk_depth and max_leaves > 1:
        trunk_levels = shared_trunk_depth
        if max_depth_cap >= 0:
            trunk_levels = min(trunk_levels, max_depth_cap)
        while actual_depth < trunk_levels and n_leaves * 2 <= max_leaves:
            if use_serial_kernels:
                if constant_hessian and row_indices is None and feature_indices is None:
                    _build_histograms_unit_hess_into_serial(
                        X_binned, grad, leaf, n_leaves, hg, hh
                    )
                elif constant_hessian and row_indices is None:
                    _build_histograms_selected_unit_hess_into_serial(
                        X_binned, grad, leaf, n_leaves, hg, hh,
                        feature_indices
                    )
                elif constant_hessian and feature_indices is None:
                    _build_histograms_rows_unit_hess_into_serial(
                        X_binned, grad, leaf, n_leaves, hg, hh,
                        row_indices
                    )
                elif constant_hessian:
                    _build_histograms_selected_rows_unit_hess_into_serial(
                        X_binned, grad, leaf, n_leaves, hg, hh,
                        feature_indices, row_indices
                    )
                elif row_indices is None and feature_indices is None:
                    _build_histograms_counts_into_serial(
                        X_binned, grad, hess, leaf, n_leaves, hg, hh, hc
                    )
                elif row_indices is None:
                    _build_histograms_counts_selected_into_serial(
                        X_binned, grad, hess, leaf, n_leaves, hg, hh, hc,
                        feature_indices
                    )
                elif feature_indices is None:
                    _build_histograms_counts_rows_into_serial(
                        X_binned, grad, hess, leaf, n_leaves, hg, hh, hc,
                        row_indices
                    )
                else:
                    _build_histograms_counts_selected_rows_into_serial(
                        X_binned, grad, hess, leaf, n_leaves, hg, hh, hc,
                        feature_indices, row_indices
                    )
            else:
                if constant_hessian and row_indices is None and feature_indices is None:
                    _build_histograms_unit_hess_into(
                        X_hist_binned, grad, leaf, n_leaves, hg, hh
                    )
                elif constant_hessian and row_indices is None:
                    _build_histograms_selected_unit_hess_into(
                        X_hist_binned, grad, leaf, n_leaves, hg, hh,
                        feature_indices
                    )
                elif constant_hessian and feature_indices is None:
                    _build_histograms_rows_unit_hess_into(
                        X_hist_binned, grad, leaf, n_leaves, hg, hh,
                        row_indices
                    )
                elif constant_hessian:
                    _build_histograms_selected_rows_unit_hess_into(
                        X_hist_binned, grad, leaf, n_leaves, hg, hh,
                        feature_indices, row_indices
                    )
                elif row_indices is None and feature_indices is None:
                    _build_histograms_counts_into(
                        X_hist_binned, grad, hess, leaf, n_leaves, hg, hh, hc
                    )
                elif row_indices is None:
                    _build_histograms_counts_selected_into(
                        X_hist_binned, grad, hess, leaf, n_leaves, hg, hh, hc,
                        feature_indices
                    )
                elif feature_indices is None:
                    _build_histograms_counts_rows_into(
                        X_hist_binned, grad, hess, leaf, n_leaves, hg, hh, hc,
                        row_indices
                    )
                else:
                    _build_histograms_counts_selected_rows_into(
                        X_hist_binned, grad, hess, leaf, n_leaves, hg, hh, hc,
                        feature_indices, row_indices
                    )

            count_hist = hh if constant_hessian else hc
            if random_strength > 0.0:
                f, t, gain = _best_shared_split_counts_with_noise_py(
                    hg, hh, count_hist, n_bins_per_feature, l2, feature_mask,
                    min_child_weight, min_child_samples, n_leaves,
                    random_strength, split_seed, tree_iteration, actual_depth,
                    min_gain_to_split
                )
            else:
                f, t, gain = _best_shared_split_counts(
                    hg, hh, count_hist, n_bins_per_feature, l2, feature_mask,
                    min_child_weight, min_child_samples, n_leaves
                )
            if t < 0 or gain <= min_gain_to_split:
                break

            old_n_leaves = n_leaves
            shared_leaf_gains = np.empty(old_n_leaves, dtype=np.float64)
            _shared_split_leaf_gains(
                hg, hh, n_bins_per_feature, l2, int(f), int(t),
                old_n_leaves, shared_leaf_gains
            )
            next_leaf_node = np.empty(max_leaves, dtype=np.int64)
            next_leaf_depth = np.zeros(max_leaves, dtype=np.int64)
            for l in range(old_n_leaves):
                node = leaf_node[l]
                left = n_nodes
                right = n_nodes + 1
                n_nodes += 2
                features[node] = int(f)
                thresholds[node] = int(t)
                left_child[node] = left
                right_child[node] = right
                leaf_index[node] = -1
                left_leaf = 2 * l
                right_leaf = left_leaf + 1
                next_leaf_node[left_leaf] = left
                next_leaf_node[right_leaf] = right
                next_leaf_depth[left_leaf] = leaf_depth[l] + 1
                next_leaf_depth[right_leaf] = leaf_depth[l] + 1
                leaf_index[left] = left_leaf
                leaf_index[right] = right_leaf
                split_features.append(int(f))
                split_thresholds.append(int(t))
                split_gains.append(float(shared_leaf_gains[l]))
            leaf_node[:2 * old_n_leaves] = next_leaf_node[:2 * old_n_leaves]
            leaf_depth[:2 * old_n_leaves] = next_leaf_depth[:2 * old_n_leaves]
            _update_leaves_with_split(X_route_binned, leaf, int(f), int(t))
            n_leaves = 2 * old_n_leaves
            actual_depth += 1

        if n_leaves > 1:
            if row_indices is None:
                row_order = np.argsort(leaf, kind="stable").astype(np.int64)
            else:
                row_order = row_order[np.argsort(leaf[row_order], kind="stable")]
            _count_leaf_rows(leaf, row_order, n_leaves, leaf_count)
            leaf_start.fill(0)
            np.cumsum(leaf_count[:n_leaves], out=leaf_start[1:n_leaves + 1])
            n_changed_leaves = n_leaves

    while n_leaves < max_leaves:
        refill_changed_histograms = (
            can_reuse_leaf_histograms and histograms_initialized
        )
        changed_leaves_scored = False
        if refill_changed_histograms:
            # The previous iteration split changed_leaves[0] into left
            # changed_leaves[0] and right changed_leaves[1]. The parent
            # histogram is still cached in the left slot. Rebuild only the
            # cheaper child and derive its sibling by subtraction.
            left_child_leaf = changed_leaves[0]
            right_child_leaf = changed_leaves[1]
            if use_segmented_rows:
                left_count = leaf_count[left_child_leaf]
                right_count = leaf_count[right_child_leaf]
            else:
                left_count = leaf_start[left_child_leaf + 1] - leaf_start[left_child_leaf]
                right_count = leaf_start[right_child_leaf + 1] - leaf_start[right_child_leaf]
            if right_count <= left_count:
                if (
                    fused_changed_leaf_scoring
                    and hessian_always_positive
                    and not constant_hessian
                    and row_indices is None
                    and feature_indices is None
                    and bool(np.all(feature_mask != 0))
                    and not use_segmented_rows
                    and split_scratch is not None
                    and get_num_threads() > 2
                ):
                    _refill_right_subtract_left_counts_positive_score_full_features_into(
                        X_hist_binned, grad, hess, row_order, leaf_start,
                        left_child_leaf, right_child_leaf, hg, hh, hc,
                        n_bins_per_feature, l2, min_child_weight,
                        min_child_samples, split_scratch[0], split_scratch[1]
                    )
                    _reduce_feature_splits_for_leaf_ids(
                        split_scratch[0], split_scratch[1], changed_leaves,
                        n_changed_leaves, best_feat, best_thr, best_gain
                    )
                    changed_leaves_scored = True
                elif rowpar_segment_lane and _rowpar_eligible(
                    right_count, rowpar_buffers, 1, max_bins
                ):
                    if constant_hessian:
                        _leafwise_segment_hist_rowpar_unit_hess(
                            X_binned, grad, row_order,
                            leaf_start[right_child_leaf],
                            leaf_start[right_child_leaf + 1],
                            left_child_leaf, right_child_leaf,
                            _ROWPAR_SCAN_RIGHT, hg, hh,
                            rowpar_buffers[0], rowpar_buffers[1]
                        )
                    else:
                        _leafwise_segment_hist_rowpar_counts(
                            X_binned, grad, hess, row_order,
                            leaf_start[right_child_leaf],
                            leaf_start[right_child_leaf + 1],
                            left_child_leaf, right_child_leaf,
                            _ROWPAR_SCAN_RIGHT, hg, hh, hc,
                            rowpar_buffers[0], rowpar_buffers[1],
                            rowpar_buffers[2]
                        )
                elif use_segmented_rows:
                    if constant_hessian:
                        _refill_right_subtract_left_unit_hess_by_count_into(
                            X_hist_binned, grad, row_order, leaf_start,
                            leaf_count, left_child_leaf, right_child_leaf,
                            hg, hh
                        )
                    else:
                        _refill_right_subtract_left_counts_by_count_into(
                            X_hist_binned, grad, hess, row_order, leaf_start,
                            leaf_count, left_child_leaf, right_child_leaf,
                            hg, hh, hc
                        )
                elif use_serial_kernels:
                    if constant_hessian and feature_indices is None:
                        _refill_leaf_segment_histograms_unit_hess_into_serial(
                            X_binned, grad, row_order, leaf_start,
                            changed_leaves[1:], 1, hg, hh
                        )
                    elif constant_hessian:
                        _refill_leaf_segment_histograms_unit_hess_selected_into_serial(
                            X_binned, grad, row_order, leaf_start,
                            changed_leaves[1:], 1, hg, hh, feature_indices
                        )
                    elif feature_indices is None:
                        _refill_leaf_segment_histograms_counts_into_serial(
                            X_binned, grad, hess, row_order, leaf_start,
                            changed_leaves[1:], 1, hg, hh, hc
                        )
                    else:
                        _refill_leaf_segment_histograms_counts_selected_into_serial(
                            X_binned, grad, hess, row_order, leaf_start,
                            changed_leaves[1:], 1, hg, hh, hc,
                            feature_indices
                        )
                    if feature_indices is None:
                        if constant_hessian:
                            _subtract_right_child_unit_hess_histograms_into_left_serial(
                                left_child_leaf, right_child_leaf, hg, hh
                            )
                        else:
                            _subtract_right_child_histograms_into_left_serial(
                                left_child_leaf, right_child_leaf, hg, hh, hc
                            )
                    else:
                        if constant_hessian:
                            _subtract_right_child_unit_hess_histograms_selected_into_left_serial(
                                left_child_leaf, right_child_leaf,
                                feature_indices, hg, hh
                            )
                        else:
                            _subtract_right_child_histograms_selected_into_left_serial(
                                left_child_leaf, right_child_leaf,
                                feature_indices, hg, hh, hc
                            )
                elif constant_hessian and feature_indices is None:
                    _refill_right_subtract_left_unit_hess_into(
                        X_hist_binned, grad, row_order, leaf_start,
                        left_child_leaf, right_child_leaf, hg, hh
                    )
                elif constant_hessian:
                    _refill_right_subtract_left_unit_hess_selected_into(
                        X_hist_binned, grad, row_order, leaf_start,
                        left_child_leaf, right_child_leaf, feature_indices,
                        hg, hh
                    )
                elif hessian_always_positive and feature_indices is None:
                    _refill_right_subtract_left_counts_positive_into(
                        X_hist_binned, grad, hess, row_order, leaf_start,
                        left_child_leaf, right_child_leaf, hg, hh, hc
                    )
                elif feature_indices is None:
                    _refill_right_subtract_left_counts_into(
                        X_hist_binned, grad, hess, row_order, leaf_start,
                        left_child_leaf, right_child_leaf, hg, hh, hc
                    )
                else:
                    _refill_right_subtract_left_counts_selected_into(
                        X_hist_binned, grad, hess, row_order, leaf_start,
                        left_child_leaf, right_child_leaf, feature_indices,
                        hg, hh, hc
                    )
            else:
                if rowpar_segment_lane and _rowpar_eligible(
                    left_count, rowpar_buffers, 1, max_bins
                ):
                    if constant_hessian:
                        _leafwise_segment_hist_rowpar_unit_hess(
                            X_binned, grad, row_order,
                            leaf_start[left_child_leaf],
                            leaf_start[left_child_leaf + 1],
                            left_child_leaf, right_child_leaf,
                            _ROWPAR_SCAN_LEFT, hg, hh,
                            rowpar_buffers[0], rowpar_buffers[1]
                        )
                    else:
                        _leafwise_segment_hist_rowpar_counts(
                            X_binned, grad, hess, row_order,
                            leaf_start[left_child_leaf],
                            leaf_start[left_child_leaf + 1],
                            left_child_leaf, right_child_leaf,
                            _ROWPAR_SCAN_LEFT, hg, hh, hc,
                            rowpar_buffers[0], rowpar_buffers[1],
                            rowpar_buffers[2]
                        )
                elif use_segmented_rows:
                    if constant_hessian:
                        _refill_left_subtract_right_unit_hess_by_count_into(
                            X_hist_binned, grad, row_order, leaf_start,
                            leaf_count, left_child_leaf, right_child_leaf,
                            hg, hh
                        )
                    else:
                        _refill_left_subtract_right_counts_by_count_into(
                            X_hist_binned, grad, hess, row_order, leaf_start,
                            leaf_count, left_child_leaf, right_child_leaf,
                            hg, hh, hc
                        )
                elif use_serial_kernels:
                    if constant_hessian and feature_indices is None:
                        _refill_leaf_segment_histograms_unit_hess_into_serial(
                            X_binned, grad, row_order, leaf_start,
                            changed_leaves, n_changed_leaves, hg, hh
                        )
                    elif constant_hessian:
                        _refill_leaf_segment_histograms_unit_hess_selected_into_serial(
                            X_binned, grad, row_order, leaf_start,
                            changed_leaves, n_changed_leaves, hg, hh,
                            feature_indices
                        )
                    elif feature_indices is None:
                        _refill_leaf_segment_histograms_counts_into_serial(
                            X_binned, grad, hess, row_order, leaf_start,
                            changed_leaves, n_changed_leaves, hg, hh, hc
                        )
                    else:
                        _refill_leaf_segment_histograms_counts_selected_into_serial(
                            X_binned, grad, hess, row_order, leaf_start,
                            changed_leaves, n_changed_leaves, hg, hh, hc,
                            feature_indices
                        )
                elif constant_hessian and feature_indices is None:
                    _refill_left_subtract_right_unit_hess_into(
                        X_hist_binned, grad, row_order, leaf_start,
                        left_child_leaf, right_child_leaf, hg, hh
                    )
                elif constant_hessian:
                    _refill_left_subtract_right_unit_hess_selected_into(
                        X_hist_binned, grad, row_order, leaf_start,
                        left_child_leaf, right_child_leaf, feature_indices,
                        hg, hh
                    )
                elif hessian_always_positive and feature_indices is None:
                    _refill_left_subtract_right_counts_positive_into(
                        X_hist_binned, grad, hess, row_order, leaf_start,
                        left_child_leaf, right_child_leaf, hg, hh, hc
                    )
                elif feature_indices is None:
                    _refill_left_subtract_right_counts_into(
                        X_hist_binned, grad, hess, row_order, leaf_start,
                        left_child_leaf, right_child_leaf, hg, hh, hc
                    )
                else:
                    _refill_left_subtract_right_counts_selected_into(
                        X_hist_binned, grad, hess, row_order, leaf_start,
                        left_child_leaf, right_child_leaf, feature_indices,
                        hg, hh, hc
                    )
        elif (
            n_leaves == 1
            and root_histograms is not None
            and row_indices is None
            and feature_indices is None
        ):
            # Root histograms precomputed by the caller (one fused class-major
            # pass in the multiclass booster); copy instead of scanning.
            hg[:n_features, 0, :max_bins] = root_histograms[0]
            hh[:n_features, 0, :max_bins] = root_histograms[1]
            if not constant_hessian:
                hc[:n_features, 0, :max_bins] = root_histograms[2]
        elif use_serial_kernels:
            if constant_hessian and row_indices is None and feature_indices is None:
                _build_histograms_unit_hess_into_serial(
                    X_binned, grad, leaf, n_leaves, hg, hh
                )
            elif constant_hessian and row_indices is None:
                _build_histograms_selected_unit_hess_into_serial(
                    X_binned, grad, leaf, n_leaves, hg, hh,
                    feature_indices
                )
            elif constant_hessian and feature_indices is None:
                _build_histograms_rows_unit_hess_into_serial(
                    X_binned, grad, leaf, n_leaves, hg, hh,
                    row_indices
                )
            elif constant_hessian:
                _build_histograms_selected_rows_unit_hess_into_serial(
                    X_binned, grad, leaf, n_leaves, hg, hh,
                    feature_indices, row_indices
                )
            elif row_indices is None and feature_indices is None:
                _build_histograms_counts_into_serial(
                    X_binned, grad, hess, leaf, n_leaves, hg, hh, hc
                )
            elif row_indices is None:
                _build_histograms_counts_selected_into_serial(
                    X_binned, grad, hess, leaf, n_leaves, hg, hh, hc,
                    feature_indices
                )
            elif feature_indices is None:
                _build_histograms_counts_rows_into_serial(
                    X_binned, grad, hess, leaf, n_leaves, hg, hh, hc,
                    row_indices
                )
            else:
                _build_histograms_counts_selected_rows_into_serial(
                    X_binned, grad, hess, leaf, n_leaves, hg, hh, hc,
                    feature_indices, row_indices
                )
        else:
            if (
                rowpar_segment_lane
                and n_leaves == 1
                and _rowpar_eligible(
                    row_order.shape[0], rowpar_buffers, 1, max_bins
                )
            ):
                # Root build: leaf 0's segment is the whole (possibly
                # subsampled) row_order, untouched because no split happened.
                if constant_hessian:
                    _leafwise_segment_hist_rowpar_unit_hess(
                        X_binned, grad, row_order, 0, row_order.shape[0],
                        0, 0, _ROWPAR_ROOT, hg, hh,
                        rowpar_buffers[0], rowpar_buffers[1]
                    )
                else:
                    _leafwise_segment_hist_rowpar_counts(
                        X_binned, grad, hess, row_order, 0,
                        row_order.shape[0], 0, 0, _ROWPAR_ROOT, hg, hh, hc,
                        rowpar_buffers[0], rowpar_buffers[1],
                        rowpar_buffers[2]
                    )
            elif constant_hessian and row_indices is None and feature_indices is None:
                _build_histograms_unit_hess_into(
                    X_hist_binned, grad, leaf, n_leaves, hg, hh
                )
            elif constant_hessian and row_indices is None:
                _build_histograms_selected_unit_hess_into(
                    X_hist_binned, grad, leaf, n_leaves, hg, hh,
                    feature_indices
                )
            elif constant_hessian and feature_indices is None:
                _build_histograms_rows_unit_hess_into(
                    X_hist_binned, grad, leaf, n_leaves, hg, hh,
                    row_indices
                )
            elif constant_hessian:
                _build_histograms_selected_rows_unit_hess_into(
                    X_hist_binned, grad, leaf, n_leaves, hg, hh,
                    feature_indices, row_indices
                )
            elif (
                hessian_always_positive
                and row_indices is None
                and feature_indices is None
            ):
                _build_histograms_counts_positive_into(
                    X_hist_binned, grad, hess, leaf, n_leaves, hg, hh, hc
                )
            elif row_indices is None and feature_indices is None:
                _build_histograms_counts_into(
                    X_hist_binned, grad, hess, leaf, n_leaves, hg, hh, hc
                )
            elif row_indices is None:
                _build_histograms_counts_selected_into(
                    X_hist_binned, grad, hess, leaf, n_leaves, hg, hh, hc,
                    feature_indices
                )
            elif feature_indices is None:
                _build_histograms_counts_rows_into(
                    X_hist_binned, grad, hess, leaf, n_leaves, hg, hh, hc,
                    row_indices
                )
            else:
                _build_histograms_counts_selected_rows_into(
                    X_hist_binned, grad, hess, leaf, n_leaves, hg, hh, hc,
                    feature_indices, row_indices
                )

        histograms_initialized = True
        count_hist = hh if constant_hessian else hc
        if random_strength > 0.0:
            noise_leaf_ids = np.arange(n_leaves, dtype=np.int64)
            noise_n_leaf_ids = n_leaves
            _best_splits_counts_for_leaf_ids_with_noise_py(
                hg, hh, count_hist, n_bins_per_feature, l2, feature_mask,
                min_child_weight, min_child_samples, noise_leaf_ids,
                noise_n_leaf_ids, best_feat, best_thr, best_gain,
                random_strength, split_seed, tree_iteration, n_nodes,
                min_gain_to_split
            )
        elif changed_leaves_scored:
            pass
        elif recompute_all_leaf_splits or n_changed_leaves >= n_leaves:
            if full_feature_positive_split:
                _best_splits_by_leaf_counts_full_features(
                    hg, hh, count_hist, n_bins_per_feature, l2,
                    min_child_weight, min_child_samples, n_leaves,
                    leaf_start, best_feat, best_thr, best_gain
                )
            else:
                _best_splits_by_leaf_counts(
                    hg, hh, count_hist, n_bins_per_feature, l2, feature_mask,
                    min_child_weight, min_child_samples, n_leaves,
                    best_feat, best_thr, best_gain
                )
        elif split_scratch is not None and get_num_threads() > 2:
            _best_splits_for_leaf_ids_counts_feature_parallel(
                hg, hh, count_hist, n_bins_per_feature, l2, feature_mask,
                min_child_weight, min_child_samples, changed_leaves,
                n_changed_leaves, split_scratch[0], split_scratch[1],
                best_feat, best_thr, best_gain
            )
        elif full_feature_positive_split:
            _best_splits_for_leaf_ids_counts_full_features(
                hg, hh, count_hist, n_bins_per_feature, l2,
                min_child_weight, min_child_samples, changed_leaves,
                n_changed_leaves, leaf_start, best_feat, best_thr, best_gain
            )
        elif (
            use_segmented_rows
            and hessian_always_positive
            and not constant_hessian
            and feature_indices is None
        ):
            _best_splits_for_leaf_ids_counts_full_features_by_count(
                hg, hh, count_hist, n_bins_per_feature, l2,
                min_child_weight, min_child_samples, changed_leaves,
                n_changed_leaves, leaf_count, best_feat, best_thr, best_gain
            )
        else:
            _best_splits_for_leaf_ids_counts(
                hg, hh, count_hist, n_bins_per_feature, l2, feature_mask,
                min_child_weight, min_child_samples, changed_leaves,
                n_changed_leaves, best_feat, best_thr, best_gain
            )

        split_leaf = -1
        split_gain = -np.inf
        split_score = -np.inf
        for l in range(n_leaves):
            if max_depth_cap >= 0 and leaf_depth[l] >= max_depth_cap:
                continue
            if best_thr[l] < 0 or best_gain[l] <= min_gain_to_split:
                continue
            score = _noisy_score(
                best_gain[l], random_strength, split_seed, tree_iteration,
                n_nodes, l, best_feat[l], best_thr[l]
            )
            if score > split_score:
                split_leaf = l
                split_gain = best_gain[l]
                split_score = score
        if split_leaf < 0:
            break

        node = leaf_node[split_leaf]
        f = int(best_feat[split_leaf])
        t = int(best_thr[split_leaf])
        left = n_nodes
        right = n_nodes + 1
        n_nodes += 2

        features[node] = f
        thresholds[node] = t
        left_child[node] = left
        right_child[node] = right
        leaf_index[node] = -1
        if canonicalize_explicit_layout_gains:
            split_gain = _selected_split_gain_full_rows(
                X_route_binned, grad, hess, leaf, split_leaf, f, t, l2,
                min_child_weight, min_child_samples, constant_hessian
            )

        old_depth = leaf_depth[split_leaf]
        new_leaf = n_leaves
        leaf_node[split_leaf] = left
        leaf_node[new_leaf] = right
        leaf_depth[split_leaf] = old_depth + 1
        leaf_depth[new_leaf] = old_depth + 1
        leaf_index[left] = split_leaf
        leaf_index[right] = new_leaf
        actual_depth = max(actual_depth, int(old_depth + 1))

        split_features.append(f)
        split_thresholds.append(t)
        split_gains.append(split_gain)
        if can_reuse_leaf_histograms:
            if use_segmented_rows:
                _partition_leaf_segment_rows(
                    X_route_binned, row_order, row_scratch, leaf, leaf_start,
                    leaf_count, split_leaf, new_leaf, f, t
                )
            else:
                _partition_leaf_rows(
                    X_route_binned, row_order, row_scratch, leaf, leaf_start,
                    n_leaves, split_leaf, new_leaf, f, t
                )
            if row_indices is not None:
                _update_leafwise_leaves_with_split(
                    X_route_binned, leaf, split_leaf, new_leaf, f, t
                )
        else:
            _update_leafwise_leaves_with_split(
                X_route_binned, leaf, split_leaf, new_leaf, f, t
            )
        n_leaves += 1
        changed_leaves[0] = split_leaf
        changed_leaves[1] = new_leaf
        n_changed_leaves = 2

    if row_indices is None:
        values, leaf_G, leaf_H = _leaf_values_and_sums(
            leaf, grad, hess, n_leaves, l2, lr
        )
    else:
        values, leaf_G, leaf_H = _leaf_values_and_sums_rows(
            leaf, grad, hess, row_indices, n_leaves, l2, lr
        )

    tree = NonObliviousTree(
        features[:n_nodes].copy(),
        thresholds[:n_nodes].copy(),
        left_child[:n_nodes].copy(),
        right_child[:n_nodes].copy(),
        leaf_index[:n_nodes].copy(),
        values,
        np.array(split_features, dtype=np.int64),
        np.array(split_thresholds, dtype=np.int64),
        np.array(split_gains, dtype=np.float64),
        actual_depth,
        n_leaves,
    )
    if return_training_state:
        return tree, leaf, leaf_G, leaf_H
    return tree


def build_hybrid_tree(X_binned, grad, hess, n_bins_per_feature,
                      max_depth, l2, lr, min_gain=1e-8, feature_mask=None,
                      min_child_weight=1.0, hist_buffers=None,
                      split_buffers=None, return_training_state=False,
                      X_hist_binned=None, X_route_binned=None,
                      feature_indices=None,
                      row_indices=None, constant_hessian=False,
                      max_leaves=None, min_gain_to_split=None,
                      min_child_samples=20,
                      recompute_all_leaf_splits=False,
                      reuse_leaf_histograms=True,
                      hessian_always_positive=False,
                      leafwise_row_layout="auto",
                      fused_changed_leaf_scoring=False,
                      rowpar_buffers=None, root_histograms=None,
                      random_strength=0.0, split_seed=0,
                      tree_iteration=0, shared_trunk_depth=2,
                      leaf_dtype="int64"):
    """Grow a shallow shared-prefix tree followed by leaf-wise expansion."""
    return build_leafwise_tree(
        X_binned, grad, hess, n_bins_per_feature, max_depth, l2, lr,
        min_gain=min_gain,
        feature_mask=feature_mask,
        min_child_weight=min_child_weight,
        hist_buffers=hist_buffers,
        split_buffers=split_buffers,
        return_training_state=return_training_state,
        X_hist_binned=X_hist_binned,
        X_route_binned=X_route_binned,
        feature_indices=feature_indices,
        row_indices=row_indices,
        constant_hessian=constant_hessian,
        max_leaves=max_leaves,
        min_gain_to_split=min_gain_to_split,
        min_child_samples=min_child_samples,
        recompute_all_leaf_splits=recompute_all_leaf_splits,
        reuse_leaf_histograms=reuse_leaf_histograms,
        hessian_always_positive=hessian_always_positive,
        leafwise_row_layout=leafwise_row_layout,
        fused_changed_leaf_scoring=fused_changed_leaf_scoring,
        rowpar_buffers=rowpar_buffers,
        root_histograms=root_histograms,
        random_strength=random_strength,
        split_seed=split_seed,
        tree_iteration=tree_iteration,
        shared_trunk_depth=shared_trunk_depth,
        leaf_dtype=leaf_dtype,
    )


def build_leafwise_multiclass_tree(
    X_binned, grad, hess, n_bins_per_feature, max_depth, l2, lr,
    min_gain=1e-8, feature_mask=None, min_child_weight=1.0,
    hist_buffers=None, return_training_state=False, X_hist_binned=None,
    X_route_binned=None,
    max_leaves=None, min_gain_to_split=None, min_child_samples=20,
    reuse_leaf_histograms=True,
    random_strength=0.0, split_seed=0, tree_iteration=0,
    grad_row_major=None, hess_row_major=None,
    leaf_dtype="int64",
):
    """Grow one shared-structure leaf-wise tree with vector leaf values."""
    if min_gain_to_split is None:
        min_gain_to_split = min_gain
    if X_hist_binned is None:
        X_hist_binned = X_binned
    elif X_hist_binned.shape != X_binned.shape:
        raise ValueError("X_hist_binned must have the same shape as X_binned")
    if X_route_binned is None:
        X_route_binned = X_binned
    elif X_route_binned.shape != X_binned.shape:
        raise ValueError("X_route_binned must have the same shape as X_binned")
    leaf_dtype = _normalize_leaf_dtype(leaf_dtype)

    grad = np.asarray(grad, dtype=np.float64)
    hess = np.asarray(hess, dtype=np.float64)
    if grad.ndim != 2 or hess.ndim != 2 or grad.shape != hess.shape:
        raise ValueError("grad and hess must be matching class-major arrays")

    K, n_samples = grad.shape
    l2_arr = np.asarray(l2, dtype=np.float64)
    if l2_arr.ndim == 0:
        l2_by_class = np.full(K, float(l2_arr), dtype=np.float64)
    elif l2_arr.shape == (K,):
        l2_by_class = np.ascontiguousarray(l2_arr)
    else:
        raise ValueError("l2 must be a scalar or have shape (K,)")
    if not np.all(np.isfinite(l2_by_class)) or np.any(l2_by_class < 0.0):
        raise ValueError("l2 must contain finite nonnegative values")
    if X_binned.shape[0] != n_samples:
        raise ValueError("X_binned row count must match grad/hess")
    if grad_row_major is None:
        grad_hist = np.ascontiguousarray(grad.T)
    else:
        grad_hist = np.asarray(grad_row_major, dtype=np.float64)
    if hess_row_major is None:
        hess_hist = np.ascontiguousarray(hess.T)
    else:
        hess_hist = np.asarray(hess_row_major, dtype=np.float64)
    if grad_hist.shape != (n_samples, K) or hess_hist.shape != (n_samples, K):
        raise ValueError(
            "grad_row_major and hess_row_major must have shape (n_samples, K)"
        )
    grad_hist = np.ascontiguousarray(grad_hist)
    hess_hist = np.ascontiguousarray(hess_hist)
    n_features = X_binned.shape[1]
    max_bins = n_features and int(n_bins_per_feature.max())
    if max_leaves is None:
        if max_depth is None or max_depth < 0:
            max_leaves = 31
        else:
            max_leaves = min(31, 1 << int(max_depth))
    max_leaves = int(max_leaves)
    if max_leaves < 1:
        raise ValueError("max_leaves must be at least 1")
    max_depth_cap = -1 if max_depth is None else int(max_depth)
    if max_depth_cap == 0 and max_leaves > 1:
        raise ValueError("max_depth must be positive, None, or -1")
    min_child_samples = int(min_child_samples)
    if min_child_samples < 1:
        raise ValueError("min_child_samples must be at least 1")

    if feature_mask is None:
        feature_mask = np.ones(n_features, dtype=np.int64)
    else:
        feature_mask = np.asarray(feature_mask, dtype=np.int64)
        if feature_mask.shape != (n_features,):
            raise ValueError("feature_mask must have one entry per feature")

    if hist_buffers is None:
        hg_class_minor = np.zeros((n_features, max_leaves, max_bins, K))
        hh_class_minor = np.zeros((n_features, max_leaves, max_bins, K))
        hc = np.zeros((n_features, max_leaves, max_bins))
    else:
        if len(hist_buffers) != 3:
            raise ValueError("hist_buffers must contain multiclass G/H/count arrays")
        hg_class_minor, hh_class_minor, hc = hist_buffers
        if (
            hg_class_minor.ndim != 4
            or hh_class_minor.ndim != 4
            or hg_class_minor.shape != hh_class_minor.shape
            or hg_class_minor.shape[0] < n_features
            or hg_class_minor.shape[1] < max_leaves
            or hg_class_minor.shape[2] < max_bins
        ):
            raise ValueError("multiclass histogram buffers are too small")
        # The class dimension must match exactly: the histogram builders only
        # zero/accumulate the K lanes derived from grad, while the split
        # scorers score every lane in the buffer, so an oversized class
        # dimension would leak stale lanes from a previous fit into gains.
        if hg_class_minor.shape[3] != K:
            raise ValueError(
                "multiclass histogram class dimension must match the class "
                "count exactly"
            )
        if (
            hc.ndim != 3
            or hc.shape[0] < n_features
            or hc.shape[1] < max_leaves
            or hc.shape[2] < max_bins
        ):
            raise ValueError("multiclass count histogram buffer is too small")
    hg, hh = hg_class_minor, hh_class_minor

    max_nodes = 2 * max_leaves - 1
    features = np.full(max_nodes, -1, dtype=np.int64)
    thresholds = np.full(max_nodes, -1, dtype=np.int64)
    left_child = np.full(max_nodes, -1, dtype=np.int64)
    right_child = np.full(max_nodes, -1, dtype=np.int64)
    leaf_index = np.full(max_nodes, -1, dtype=np.int64)
    leaf_node = np.empty(max_leaves, dtype=np.int64)
    leaf_depth = np.zeros(max_leaves, dtype=np.int64)
    leaf_node[0] = 0
    leaf_index[0] = 0

    best_feat = np.empty(max_leaves, dtype=np.int64)
    best_thr = np.empty(max_leaves, dtype=np.int64)
    best_gain = np.empty(max_leaves, dtype=np.float64)
    changed_leaves = np.empty(2, dtype=np.int64)
    changed_leaves[0] = 0
    n_changed_leaves = 1
    row_order = np.arange(n_samples, dtype=np.int64)
    row_scratch = np.empty(n_samples, dtype=np.int64)
    leaf_start = np.zeros(max_leaves + 1, dtype=np.int64)
    leaf_start[1] = n_samples
    split_features = []
    split_thresholds = []
    split_gains = []

    leaf = np.zeros(n_samples, dtype=leaf_dtype)
    n_nodes = 1
    n_leaves = 1
    actual_depth = 0
    histograms_initialized = False

    while n_leaves < max_leaves:
        if reuse_leaf_histograms and histograms_initialized:
            left_child_leaf = changed_leaves[0]
            right_child_leaf = changed_leaves[1]
            left_count = leaf_start[left_child_leaf + 1] - leaf_start[left_child_leaf]
            right_count = leaf_start[right_child_leaf + 1] - leaf_start[right_child_leaf]
            if right_count <= left_count:
                _refill_multiclass_right_subtract_left_counts_class_minor_into(
                    X_hist_binned, grad_hist, hess_hist,
                    row_order, leaf_start,
                    left_child_leaf, right_child_leaf, hg, hh, hc
                )
            else:
                _refill_multiclass_left_subtract_right_counts_class_minor_into(
                    X_hist_binned, grad_hist, hess_hist,
                    row_order, leaf_start,
                    left_child_leaf, right_child_leaf, hg, hh, hc
                )
        else:
            _build_multiclass_histograms_counts_class_minor_into(
                X_hist_binned, grad_hist, hess_hist,
                leaf, n_leaves, hg, hh, hc
            )
        histograms_initialized = True

        if n_changed_leaves >= n_leaves:
            changed_leaves[0] = 0
            n_changed_leaves = n_leaves
            # The first iteration has a single leaf; later code never requests
            # a full multiclass rescore, so the fixed two-slot buffer is enough.
        if random_strength > 0.0:
            noise_leaf_ids = np.arange(n_leaves, dtype=np.int64)
            _best_multiclass_splits_counts_for_leaf_ids_with_noise_class_minor_py(
                hg, hh, hc, n_bins_per_feature, l2_by_class, feature_mask,
                min_child_weight, min_child_samples, noise_leaf_ids,
                n_leaves, best_feat, best_thr, best_gain,
                random_strength, split_seed, tree_iteration, n_nodes,
                min_gain_to_split
            )
        else:
            _best_multiclass_splits_for_leaf_ids_counts_class_minor(
                hg, hh, hc, n_bins_per_feature, l2_by_class, feature_mask,
                min_child_weight, min_child_samples, changed_leaves,
                n_changed_leaves, best_feat, best_thr, best_gain
            )

        split_leaf = -1
        split_gain = -np.inf
        split_score = -np.inf
        for l in range(n_leaves):
            if max_depth_cap >= 0 and leaf_depth[l] >= max_depth_cap:
                continue
            if best_thr[l] < 0 or best_gain[l] <= min_gain_to_split:
                continue
            score = _noisy_score(
                best_gain[l], random_strength, split_seed, tree_iteration,
                n_nodes, l, best_feat[l], best_thr[l]
            )
            if score > split_score:
                split_leaf = l
                split_gain = best_gain[l]
                split_score = score
        if split_leaf < 0:
            break

        node = leaf_node[split_leaf]
        f = int(best_feat[split_leaf])
        t = int(best_thr[split_leaf])
        left = n_nodes
        right = n_nodes + 1
        n_nodes += 2

        features[node] = f
        thresholds[node] = t
        left_child[node] = left
        right_child[node] = right
        leaf_index[node] = -1

        old_depth = leaf_depth[split_leaf]
        new_leaf = n_leaves
        leaf_node[split_leaf] = left
        leaf_node[new_leaf] = right
        leaf_depth[split_leaf] = old_depth + 1
        leaf_depth[new_leaf] = old_depth + 1
        leaf_index[left] = split_leaf
        leaf_index[right] = new_leaf
        actual_depth = max(actual_depth, int(old_depth + 1))

        split_features.append(f)
        split_thresholds.append(t)
        split_gains.append(split_gain)
        _partition_leaf_rows(
            X_route_binned, row_order, row_scratch, leaf, leaf_start,
            n_leaves, split_leaf, new_leaf, f, t
        )
        n_leaves += 1
        changed_leaves[0] = split_leaf
        changed_leaves[1] = new_leaf
        n_changed_leaves = 2

    values, leaf_G, leaf_H = _multiclass_leaf_values_and_sums_l2_by_class(
        leaf, grad, hess, n_leaves, l2_by_class, lr
    )
    if len(split_features) == 0:
        values[0, :] = 0.0

    tree = MultiNonObliviousTree(
        features[:n_nodes].copy(),
        thresholds[:n_nodes].copy(),
        left_child[:n_nodes].copy(),
        right_child[:n_nodes].copy(),
        leaf_index[:n_nodes].copy(),
        values,
        np.array(split_features, dtype=np.int64),
        np.array(split_thresholds, dtype=np.int64),
        np.array(split_gains, dtype=np.float64),
        actual_depth,
        n_leaves,
    )
    if return_training_state:
        return tree, leaf, leaf_G, leaf_H
    return tree
