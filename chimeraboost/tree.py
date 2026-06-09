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
    n_features = hg.shape[0]
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
    n_features = hg.shape[0]
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


@njit(cache=True)
def _update_leaves_with_split(X_binned, leaf, split_feat, split_thr):
    """Append one split bit to existing leaf ids in place."""
    for i in range(leaf.shape[0]):
        bit = 1 if X_binned[i, split_feat] > split_thr else 0
        leaf[i] = leaf[i] * 2 + bit


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
def _best_splits_by_leaf(hg, hh, n_bins_per_feature, l2, feat_mask,
                         min_child_weight, n_leaves, out_feat, out_thr,
                         out_gain):
    """Find the best split independently for every active leaf."""
    n_features = hg.shape[0]
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
    n_features = hg.shape[0]
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
        else:
            _predict_non_oblivious_tree_add(
                X_binned, self.features, self.thresholds, self.left_child,
                self.right_child, self.leaf_index, self.values, out
            )


def build_oblivious_tree(X_binned, grad, hess, n_bins_per_feature,
                         max_depth, l2, lr, min_gain=1e-8, feature_mask=None,
                         min_child_weight=1.0, hist_buffers=None,
                         split_buffers=None,
                         return_training_state=False, X_hist_binned=None,
                         feature_indices=None, row_indices=None,
                         constant_hessian=False):
    """Grow one oblivious tree level by level and return an ObliviousTree.

    X_hist_binned: optional feature-contiguous view/copy of X_binned used only
    by the multithreaded histogram builder. Leaf routing and returned training
    leaves still use X_binned, preserving row-wise locality for those paths.
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
    """
    if X_hist_binned is None:
        X_hist_binned = X_binned
    elif X_hist_binned.shape != X_binned.shape:
        raise ValueError("X_hist_binned must have the same shape as X_binned")
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
    leaf = np.zeros(X_binned.shape[0], dtype=np.int64)
    use_serial_kernels = get_num_threads() == 1
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

    for d in range(max_depth):
        n_leaves = 1 << d
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
            f, t, gain = _best_split_serial(
                hg, hh, n_bins_per_feature, l2, feature_mask,
                min_child_weight, n_leaves
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
        _update_leaves_with_split(X_binned, leaf, f, t)

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
                         feature_indices=None, row_indices=None,
                         constant_hessian=False):
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
    leaf = np.zeros(X_binned.shape[0], dtype=np.int64)
    use_serial_kernels = get_num_threads() == 1
    actual_depth = 0

    for d in range(max_depth):
        n_leaves = 1 << d
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
            X_binned, leaf, node_features[d], node_thresholds[d]
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
                        X_hist_binned=None, feature_indices=None,
                        row_indices=None, constant_hessian=False,
                        max_leaves=None, min_gain_to_split=None,
                        min_child_samples=20):
    """Grow a LightGBM-like leaf-wise, best-first non-oblivious tree.

    This builder chooses the best legal split for each current leaf, then splits
    only the leaf with the largest gain. It intentionally favors correctness and
    a clean tree contract over histogram reuse; speed optimizations belong after
    the public semantics are settled.
    """
    del split_buffers
    if min_gain_to_split is None:
        min_gain_to_split = min_gain
    if X_hist_binned is None:
        X_hist_binned = X_binned
    elif X_hist_binned.shape != X_binned.shape:
        raise ValueError("X_hist_binned must have the same shape as X_binned")

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
    split_features = []
    split_thresholds = []
    split_gains = []

    leaf = np.zeros(n_samples, dtype=np.int64)
    n_nodes = 1
    n_leaves = 1
    actual_depth = 0
    use_serial_kernels = get_num_threads() == 1

    while n_leaves < max_leaves:
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
            if row_indices is None:
                _build_counts_into_serial(X_binned, hess, leaf, n_leaves, hc)
            else:
                _build_counts_rows_into_serial(
                    X_binned, hess, leaf, n_leaves, hc, row_indices
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
            if row_indices is None:
                _build_counts_into(X_hist_binned, hess, leaf, n_leaves, hc)
            else:
                _build_counts_rows_into(
                    X_hist_binned, hess, leaf, n_leaves, hc, row_indices
                )

        _best_splits_by_leaf_counts(
            hg, hh, hc, n_bins_per_feature, l2, feature_mask,
            min_child_weight, min_child_samples, n_leaves,
            best_feat, best_thr, best_gain
        )

        split_leaf = -1
        split_gain = -np.inf
        for l in range(n_leaves):
            if max_depth_cap >= 0 and leaf_depth[l] >= max_depth_cap:
                continue
            if best_thr[l] >= 0 and best_gain[l] > split_gain:
                split_leaf = l
                split_gain = best_gain[l]
        if split_leaf < 0 or split_gain <= min_gain_to_split:
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
        _update_leafwise_leaves_with_split(
            X_binned, leaf, split_leaf, new_leaf, f, t
        )
        n_leaves += 1

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
