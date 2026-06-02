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
from numba import njit, prange


@njit(cache=True, parallel=True)
def _build_histograms_into(Xb, grad, hess, leaf, n_leaves, hist):
    """Fill per-feature gradient/hessian histograms into a pre-allocated buffer.

    `Xb` is feature-major (n_features, n_samples), so `Xb[f]` is a contiguous
    row and the inner sample loop reads bins, grads, and hessians sequentially.

    `hist` has shape (n_features, max_leaves, max_bins, 2): grad and hess for a
    bin are interleaved on the last axis so each scatter write touches a single
    cache line instead of two separate arrays. Reused across every tree and
    level; we zero only the (n_leaves) slice we are about to write. Parallelized
    over features so each thread owns a disjoint slice -- no write races.
    """
    n_features, n_samples = Xb.shape
    max_bins = hist.shape[2]
    for f in prange(n_features):
        for l in range(n_leaves):
            for b in range(max_bins):
                hist[f, l, b, 0] = 0.0
                hist[f, l, b, 1] = 0.0
        Xf = Xb[f]
        for i in range(n_samples):
            l = leaf[i]
            b = Xf[i]
            hist[f, l, b, 0] += grad[i]
            hist[f, l, b, 1] += hess[i]


@njit(cache=True, parallel=True)
def _best_split(hist, n_bins_per_feature, l2, feat_mask, min_child_weight,
                n_leaves):
    """Find the (feature, threshold) with the highest total gain.

    `hist` is the interleaved (n_features, max_leaves, max_bins, 2) buffer:
    [..., 0] is grad, [..., 1] is hess. `n_leaves` says how many leaf rows are
    actually active at this level, so we only read those.

    For a candidate threshold t, bins <= t go left and bins > t go right, the
    same way in every current leaf. Gain is summed across leaves. Features with
    feat_mask[f] == 0 are skipped (column subsampling).

    A threshold is legal unless some leaf would gain a *sparse non-empty* child
    (0 < hessian mass < min_child_weight) -- that is the sparse-leaf overfit risk,
    and since the split is shared it is rejected for the whole level. Children
    that come out EMPTY (a leaf whose samples all go one way) are exempt: pure
    leaves are normal in an oblivious tree and must not block the shared split,
    or effective depth caps far below what the data supports.
    """
    n_features = hist.shape[0]
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
                Gt[l] += hist[f, l, b, 0]
                Ht[l] += hist[f, l, b, 1]

        GL = np.zeros(n_leaves)
        HL = np.zeros(n_leaves)
        best_g = -np.inf
        best_t = -1
        # Threshold t means "left = bins [0..t]". Last bin can't be a threshold.
        for t in range(nb - 1):
            # Pass 1: Advance running prefix sums for all leaves unconditionally
            # so GL/HL carry correctly into the next threshold.
            for l in range(n_leaves):
                GL[l] += hist[f, l, t, 0]
                HL[l] += hist[f, l, t, 1]

            # Pass 2: gain of this threshold, and its legality (see docstring:
            # only a sparse non-empty child vetoes the shared split).
            gain = 0.0
            legal = True
            for l in range(n_leaves):
                if Ht[l] > 0.0:
                    hl = HL[l]
                    hr = Ht[l] - hl
                    # Empty child (hl==0 or hr==0) is exempt; only 0 < mass <
                    # min_child_weight is illegal.
                    if (hl > 0.0 and hl < min_child_weight) or \
                       (hr > 0.0 and hr < min_child_weight):
                        legal = False
                        break
                    gl = GL[l]
                    gr = Gt[l] - gl
                    gain += (
                        gl * gl / (hl + l2)
                        + gr * gr / (hr + l2)
                        - Gt[l] * Gt[l] / (Ht[l] + l2)
                    )

            if legal and gain > best_g:
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
def _assign_leaves(Xb, splits_feat, splits_thr):
    """Leaf index of every sample given the splits. `Xb` is feature-major, so
    each level reads one contiguous feature row."""
    depth = splits_feat.shape[0]
    n = Xb.shape[1]
    leaf = np.zeros(n, dtype=np.int64)
    for d in range(depth):
        Xf = Xb[splits_feat[d]]
        t = splits_thr[d]
        for i in range(n):
            leaf[i] = leaf[i] * 2 + (1 if Xf[i] > t else 0)
    return leaf


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
def _loo_leaf_step(leaf, grad, hess, n_leaves, l2, lr):
    """Leave-one-out training step for every row, fused into two passes.

    First pass scatters per-leaf grad/hess totals; second pass gathers each
    row's totals, removes the row's own contribution, and forms the shrunk
    Newton step. Replaces two np.bincount calls plus several NumPy temporaries
    with one scatter and one compute loop over `leaf`."""
    G = np.zeros(n_leaves)
    H = np.zeros(n_leaves)
    n = leaf.shape[0]
    for i in range(n):
        l = leaf[i]
        G[l] += grad[i]
        H[l] += hess[i]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        l = leaf[i]
        denom = H[l] - hess[i]
        if denom < 0.0:
            denom = 0.0
        out[i] = -lr * (G[l] - grad[i]) / (denom + l2)
    return out


@njit(cache=True)
def _predict_tree(Xb, splits_feat, splits_thr, values):
    """Route each sample to its leaf and return that leaf's value."""
    leaf = _assign_leaves(Xb, splits_feat, splits_thr)
    out = np.empty(Xb.shape[1], dtype=np.float64)
    for i in range(leaf.shape[0]):
        out[i] = values[leaf[i]]
    return out


@njit(cache=True, parallel=True)
def _predict_forest(Xb, feats, thrs, depths, vals, voff, init):
    """Sum a whole ensemble of oblivious trees in one parallel pass over samples.

    Parameters are the trees packed into flat arrays (see `pack_forest`):
    `feats`/`thrs` are (n_trees, max_depth) split tables, `depths[t]` the real
    depth of tree t, and `vals`/`voff` a ragged leaf-value table (tree t's leaf
    values live at vals[voff[t] : voff[t+1]]).

    Parallelizing over samples (not trees) means each sample loads its handful
    of feature bins once and keeps them hot in cache while walking every tree.
    The per-sample accumulation runs init + tree0 + tree1 + ... in tree order,
    matching the serial `F += tree.predict(Xb)` loop bit-for-bit."""
    n = Xb.shape[1]
    n_trees = feats.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        acc = init
        for t in range(n_trees):
            # A depth-0 tree found no legal split; like ObliviousTree.predict it
            # contributes nothing (its lone leaf value is never applied).
            if depths[t] == 0:
                continue
            leaf = 0
            for d in range(depths[t]):
                if Xb[feats[t, d], i] > thrs[t, d]:
                    leaf = leaf * 2 + 1
                else:
                    leaf = leaf * 2
            acc += vals[voff[t] + leaf]
        out[i] = acc
    return out


def pack_forest(trees, max_depth):
    """Flatten a list of ObliviousTrees into the arrays `_predict_forest` wants.

    Returns (feats, thrs, depths, vals, voff). Cached by the booster after fit
    so repeated predict calls skip the rebuild."""
    n_trees = len(trees)
    feats = np.zeros((n_trees, max_depth), dtype=np.int64)
    thrs = np.zeros((n_trees, max_depth), dtype=np.int64)
    depths = np.empty(n_trees, dtype=np.int64)
    voff = np.empty(n_trees + 1, dtype=np.int64)
    voff[0] = 0
    for t, tree in enumerate(trees):
        d = tree.depth
        depths[t] = d
        feats[t, :d] = tree.splits_feat
        thrs[t, :d] = tree.splits_thr
        voff[t + 1] = voff[t] + tree.values.shape[0]
    vals = np.empty(voff[-1], dtype=np.float64)
    for t, tree in enumerate(trees):
        vals[voff[t]:voff[t + 1]] = tree.values
    return feats, thrs, depths, vals, voff


class ObliviousTree:
    """A single symmetric tree. Stores its splits and leaf values.

    Its `apply`/`predict` take a feature-major binned matrix (n_features,
    n_samples) -- the same layout the builder consumes."""

    __slots__ = ("splits_feat", "splits_thr", "values", "gains", "depth")

    def __init__(self, splits_feat, splits_thr, values, gains=None):
        self.splits_feat = splits_feat
        self.splits_thr = splits_thr
        self.values = values
        self.gains = gains if gains is not None else np.zeros(len(splits_feat))
        self.depth = len(splits_feat)

    def apply(self, Xb):
        """Return the leaf index of each sample."""
        if self.depth == 0:
            return np.zeros(Xb.shape[1], dtype=np.int64)
        return _assign_leaves(Xb, self.splits_feat, self.splits_thr)

    def predict(self, Xb):
        if self.depth == 0:
            return np.zeros(Xb.shape[1], dtype=np.float64)
        return _predict_tree(Xb, self.splits_feat, self.splits_thr, self.values)


def build_oblivious_tree(Xb, grad, hess, n_bins_per_feature,
                         max_depth, l2, lr, min_gain=1e-8, feature_mask=None,
                         min_child_weight=1.0, hist_buffers=None):
    """Grow one oblivious tree level by level. Returns (tree, train_leaf), where
    train_leaf is the tree's leaf index for every training sample.

    Xb: feature-major binned matrix (n_features, n_samples).
    feature_mask: optional 0/1 array over features; 0 disables a feature for
    this tree (column subsampling). None means all features are eligible.
    min_child_weight: minimum hessian mass each side of a split must retain in
    every non-empty leaf. Stops the tree growing once no legal split remains,
    which prevents sparse-leaf overfitting at higher depth.
    hist_buffers: optional interleaved buffer of shape (n_features,
    2**max_depth, max_bins, 2) reused across trees to avoid per-level
    allocation. If None, it is allocated here (for one-off calls and tests).
    """
    n_features, n_samples = Xb.shape
    max_bins = n_features and int(n_bins_per_feature.max())
    if feature_mask is None:
        feature_mask = np.ones(n_features, dtype=np.int64)
    if hist_buffers is None:
        hist = np.zeros((n_features, 1 << max_depth, max_bins, 2))
    else:
        hist = hist_buffers
    splits_feat = []
    splits_thr = []
    splits_gain = []
    leaf = np.zeros(n_samples, dtype=np.int64)

    for d in range(max_depth):
        n_leaves = 1 << d
        _build_histograms_into(Xb, grad, hess, leaf, n_leaves, hist)
        f, t, gain = _best_split(hist, n_bins_per_feature, l2, feature_mask,
                                 min_child_weight, n_leaves)
        if gain <= min_gain or t < 0:
            break
        splits_feat.append(f)
        splits_thr.append(t)
        splits_gain.append(gain)
        # Push each sample one bit deeper from the just-chosen split. Xb[f] is
        # a contiguous row, so this re-bucketing reads sequentially.
        leaf = (leaf << 1) + (Xb[f] > t).astype(np.int64)

    sf = np.array(splits_feat, dtype=np.int64)
    st = np.array(splits_thr, dtype=np.int64)
    n_leaves = 1 << len(splits_feat)
    values = _leaf_values(leaf, grad, hess, n_leaves, l2, lr)
    tree = ObliviousTree(sf, st, values, np.array(splits_gain, dtype=np.float64))
    # `leaf` is the training-set assignment, returned so callers (LOO update,
    # leaf correction) reuse it instead of recomputing tree.apply(Xb).
    return tree, leaf