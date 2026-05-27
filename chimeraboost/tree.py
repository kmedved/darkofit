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


@njit(cache=True, parallel=True)
def _best_split(hg, hh, n_bins_per_feature, l2, feat_mask, min_child_weight,
                n_leaves):
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
    feat_thr = np.zeros(n_features, dtype=np.int64)

    for f in prange(n_features):
        if feat_mask[f] == 0:
            continue
        nb = n_bins_per_feature[f]
        # Totals per leaf for this feature (same regardless of threshold).
        Gt = np.zeros(n_leaves)
        Ht = np.zeros(n_leaves)
        for l in range(n_leaves):
            for b in range(nb):
                Gt[l] += hg[f, l, b]
                Ht[l] += hh[f, l, b]

        GL = np.zeros(n_leaves)
        HL = np.zeros(n_leaves)
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
                GL[l] += hg[f, l, t]
                HL[l] += hh[f, l, t]
                
                if Ht[l] > 0.0:
                    any_nonempty = True
                    hl = HL[l]
                    hr = Ht[l] - hl
                    if hl < min_child_weight or hr < min_child_weight:
                        legal = False
                    gl = GL[l]
                    gr = Gt[l] - gl
                    gain += (
                        gl * gl / (hl + l2)
                        + gr * gr / (hr + l2)
                        - Gt[l] * Gt[l] / (Ht[l] + l2)
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

    best_f = 0
    best_t = 0
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
                    if hl < min_child_weight or hr < min_child_weight:
                        legal = False
                    gl = GL[l]
                    gr = Gt[l] - gl
                    gain += (
                        gl * gl / (hl + l2)
                        + gr * gr / (hr + l2)
                        - Gt[l] * Gt[l] / (Ht[l] + l2)
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


def build_oblivious_tree(X_binned, grad, hess, n_bins_per_feature,
                         max_depth, l2, lr, min_gain=1e-8, feature_mask=None,
                         min_child_weight=1.0, hist_buffers=None,
                         return_training_state=False, X_hist_binned=None,
                         feature_indices=None):
    """Grow one oblivious tree level by level and return an ObliviousTree.

    X_hist_binned: optional feature-contiguous view/copy of X_binned used only
    by the multithreaded histogram builder. Leaf routing and returned training
    leaves still use X_binned, preserving row-wise locality for those paths.
    feature_mask: optional 0/1 array over features; 0 disables a feature for
    this tree (column subsampling). None means all features are eligible.
    feature_indices: optional selected column indices matching feature_mask;
    when supplied, histogram building zeroes and fills only those columns.
    min_child_weight: minimum hessian mass each side of a split must retain in
    every non-empty leaf. Stops the tree growing once no legal split remains,
    which prevents sparse-leaf overfitting at higher depth.
    hist_buffers: optional (hg, hh) arrays of shape (n_features, 2**max_depth,
    max_bins) reused across trees to avoid per-level allocation. If None, they
    are allocated here (convenient for one-off calls and tests).
    """
    if X_hist_binned is None:
        X_hist_binned = X_binned
    n_features = X_binned.shape[1]
    max_bins = n_features and int(n_bins_per_feature.max())
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
        max_leaves = 1 << max_depth
        hg = np.zeros((n_features, max_leaves, max_bins))
        hh = np.zeros((n_features, max_leaves, max_bins))
    else:
        hg, hh = hist_buffers
    splits_feat = []
    splits_thr = []
    splits_gain = []
    leaf = np.zeros(X_binned.shape[0], dtype=np.int64)
    use_serial_kernels = get_num_threads() == 1

    for d in range(max_depth):
        n_leaves = 1 << d
        if use_serial_kernels:
            if feature_indices is None:
                _build_histograms_into_serial(
                    X_binned, grad, hess, leaf, n_leaves, hg, hh
                )
            else:
                _build_histograms_selected_into_serial(
                    X_binned, grad, hess, leaf, n_leaves, hg, hh,
                    feature_indices
                )
            f, t, gain = _best_split_serial(
                hg, hh, n_bins_per_feature, l2, feature_mask,
                min_child_weight, n_leaves
            )
        else:
            if feature_indices is None:
                _build_histograms_into(
                    X_hist_binned, grad, hess, leaf, n_leaves, hg, hh
                )
            else:
                _build_histograms_selected_into(
                    X_hist_binned, grad, hess, leaf, n_leaves, hg, hh,
                    feature_indices
                )
            f, t, gain = _best_split(
                hg, hh, n_bins_per_feature, l2, feature_mask,
                min_child_weight, n_leaves
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
    values, leaf_G, leaf_H = _leaf_values_and_sums(
        leaf, grad, hess, n_leaves, l2, lr
    )
    tree = ObliviousTree(sf, st, values, np.array(splits_gain, dtype=np.float64))
    if return_training_state:
        return tree, leaf, leaf_G, leaf_H
    return tree
