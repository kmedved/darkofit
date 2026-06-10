"""Flattened ensembles for fast batch prediction.

``predict_raw`` historically looped over trees in Python, launching one numba
kernel per tree and streaming the binned matrix (and the output vector) from
memory once per tree. These helpers pack a fitted ensemble's trees into a few
contiguous arrays once, then a single row-parallel kernel walks every tree for
a row while that row is hot in cache.

Per-row accumulation runs in the same tree order as the original loop and
starts from the existing output value, so predictions are bitwise identical
to per-tree ``add_predict`` calls.

Unsupported tree types (the experimental level-wise trees) return None from
the builders; callers fall back to the per-tree loop.
"""

import numpy as np
from numba import get_num_threads, njit, prange

from .tree import MultiNonObliviousTree, NonObliviousTree, ObliviousTree

_PARALLEL_MIN_ROWS = 8192


@njit(cache=True)
def _flat_oblivious_add(X_binned, depths, feats, thrs, value_offsets, values,
                        out):
    n = X_binned.shape[0]
    n_trees = depths.shape[0]
    for i in range(n):
        acc = out[i]
        for t in range(n_trees):
            idx = 0
            for d in range(depths[t]):
                f = feats[t, d]
                bit = 1 if X_binned[i, f] > thrs[t, d] else 0
                idx = idx * 2 + bit
            acc += values[value_offsets[t] + idx]
        out[i] = acc


@njit(cache=True, parallel=True)
def _flat_oblivious_add_parallel(X_binned, depths, feats, thrs, value_offsets,
                                 values, out):
    n = X_binned.shape[0]
    n_trees = depths.shape[0]
    for i in prange(n):
        acc = out[i]
        for t in range(n_trees):
            idx = 0
            for d in range(depths[t]):
                f = feats[t, d]
                bit = 1 if X_binned[i, f] > thrs[t, d] else 0
                idx = idx * 2 + bit
            acc += values[value_offsets[t] + idx]
        out[i] = acc


# Non-oblivious walks are branchy, so the kernels process rows in blocks and
# iterate trees outermost within each block: one tree's node path stays in L1
# while a block of rows walks it (the access pattern of the fast per-tree
# loop), and the block's X rows and outputs stay hot across all trees. Adds
# into each out[i] still happen in ascending tree order, so results remain
# bitwise identical to per-tree add_predict calls.
_ROW_BLOCK = 256


@njit(cache=True)
def _flat_nonoblivious_add(X_binned, node_offsets, features, thresholds,
                           left_child, right_child, leaf_index, value_offsets,
                           values, out):
    n = X_binned.shape[0]
    n_trees = node_offsets.shape[0] - 1
    for start in range(0, n, _ROW_BLOCK):
        end = min(n, start + _ROW_BLOCK)
        for t in range(n_trees):
            base = node_offsets[t]
            voff = value_offsets[t]
            for i in range(start, end):
                node = 0
                while left_child[base + node] >= 0:
                    if X_binned[i, features[base + node]] > thresholds[base + node]:
                        node = right_child[base + node]
                    else:
                        node = left_child[base + node]
                out[i] += values[voff + leaf_index[base + node]]


@njit(cache=True, parallel=True)
def _flat_nonoblivious_add_parallel(X_binned, node_offsets, features,
                                    thresholds, left_child, right_child,
                                    leaf_index, value_offsets, values, out):
    n = X_binned.shape[0]
    n_trees = node_offsets.shape[0] - 1
    n_blocks = (n + _ROW_BLOCK - 1) // _ROW_BLOCK
    for blk in prange(n_blocks):
        start = blk * _ROW_BLOCK
        end = min(n, start + _ROW_BLOCK)
        for t in range(n_trees):
            base = node_offsets[t]
            voff = value_offsets[t]
            for i in range(start, end):
                node = 0
                while left_child[base + node] >= 0:
                    if X_binned[i, features[base + node]] > thresholds[base + node]:
                        node = right_child[base + node]
                    else:
                        node = left_child[base + node]
                out[i] += values[voff + leaf_index[base + node]]


@njit(cache=True, parallel=True)
def _flat_oblivious_class_add_parallel(X_binned, depths, feats, thrs,
                                       value_offsets, values, class_ids, out):
    """Scalar per-class trees adding into a class-major (K, n) margin."""
    n = X_binned.shape[0]
    n_trees = depths.shape[0]
    for i in prange(n):
        for t in range(n_trees):
            idx = 0
            for d in range(depths[t]):
                f = feats[t, d]
                bit = 1 if X_binned[i, f] > thrs[t, d] else 0
                idx = idx * 2 + bit
            out[class_ids[t], i] += values[value_offsets[t] + idx]


@njit(cache=True, parallel=True)
def _flat_nonoblivious_class_add_parallel(X_binned, node_offsets, features,
                                          thresholds, left_child, right_child,
                                          leaf_index, value_offsets, values,
                                          class_ids, out):
    n = X_binned.shape[0]
    n_trees = node_offsets.shape[0] - 1
    n_blocks = (n + _ROW_BLOCK - 1) // _ROW_BLOCK
    for blk in prange(n_blocks):
        start = blk * _ROW_BLOCK
        end = min(n, start + _ROW_BLOCK)
        for t in range(n_trees):
            base = node_offsets[t]
            voff = value_offsets[t]
            k = class_ids[t]
            for i in range(start, end):
                node = 0
                while left_child[base + node] >= 0:
                    if X_binned[i, features[base + node]] > thresholds[base + node]:
                        node = right_child[base + node]
                    else:
                        node = left_child[base + node]
                out[k, i] += values[voff + leaf_index[base + node]]


@njit(cache=True, parallel=True)
def _flat_multi_add_parallel(X_binned, node_offsets, features, thresholds,
                             left_child, right_child, leaf_index,
                             value_offsets, values, out):
    """Shared-structure trees with (leaf, K) vector values; out is (K, n)."""
    n = X_binned.shape[0]
    n_trees = node_offsets.shape[0] - 1
    n_classes = out.shape[0]
    n_blocks = (n + _ROW_BLOCK - 1) // _ROW_BLOCK
    for blk in prange(n_blocks):
        start = blk * _ROW_BLOCK
        end = min(n, start + _ROW_BLOCK)
        for t in range(n_trees):
            base = node_offsets[t]
            voff = value_offsets[t]
            for i in range(start, end):
                node = 0
                while left_child[base + node] >= 0:
                    if X_binned[i, features[base + node]] > thresholds[base + node]:
                        node = right_child[base + node]
                    else:
                        node = left_child[base + node]
                l = voff + leaf_index[base + node]
                for k in range(n_classes):
                    out[k, i] += values[l, k]


class FlatObliviousEnsemble:
    """Padded (n_trees, max_depth) split matrices plus concatenated values."""

    __slots__ = ("depths", "feats", "thrs", "value_offsets", "values",
                 "class_ids")

    def __init__(self, trees, class_ids=None):
        n_trees = len(trees)
        max_depth = max((t.depth for t in trees), default=0)
        self.depths = np.array([t.depth for t in trees], dtype=np.int64)
        self.feats = np.zeros((n_trees, max(max_depth, 1)), dtype=np.int64)
        self.thrs = np.zeros((n_trees, max(max_depth, 1)), dtype=np.int64)
        self.value_offsets = np.zeros(n_trees, dtype=np.int64)
        offset = 0
        chunks = []
        for t, tree in enumerate(trees):
            d = tree.depth
            if d:
                self.feats[t, :d] = tree.splits_feat
                self.thrs[t, :d] = tree.splits_thr
            self.value_offsets[t] = offset
            chunks.append(tree.values)
            offset += tree.values.shape[0]
        self.values = (np.concatenate(chunks) if chunks
                       else np.empty(0, dtype=np.float64))
        self.class_ids = class_ids

    def add_predict(self, X_binned, out):
        if get_num_threads() > 1 and X_binned.shape[0] >= _PARALLEL_MIN_ROWS:
            _flat_oblivious_add_parallel(
                X_binned, self.depths, self.feats, self.thrs,
                self.value_offsets, self.values, out
            )
        else:
            _flat_oblivious_add(
                X_binned, self.depths, self.feats, self.thrs,
                self.value_offsets, self.values, out
            )

    def add_predict_class_major(self, X_binned, out):
        _flat_oblivious_class_add_parallel(
            X_binned, self.depths, self.feats, self.thrs,
            self.value_offsets, self.values, self.class_ids, out
        )


class FlatNonObliviousEnsemble:
    """CSR-style concatenated node arrays for explicit-node trees."""

    __slots__ = ("node_offsets", "features", "thresholds", "left_child",
                 "right_child", "leaf_index", "value_offsets", "values",
                 "class_ids")

    def __init__(self, trees, class_ids=None, vector_values=False):
        n_trees = len(trees)
        self.node_offsets = np.zeros(n_trees + 1, dtype=np.int64)
        self.value_offsets = np.zeros(n_trees, dtype=np.int64)
        feat_chunks, thr_chunks = [], []
        left_chunks, right_chunks, leaf_chunks = [], [], []
        value_chunks = []
        n_offset = 0
        v_offset = 0
        for t, tree in enumerate(trees):
            self.node_offsets[t] = n_offset
            self.value_offsets[t] = v_offset
            feat_chunks.append(tree.features)
            thr_chunks.append(tree.thresholds)
            left_chunks.append(tree.left_child)
            right_chunks.append(tree.right_child)
            leaf_chunks.append(tree.leaf_index)
            value_chunks.append(tree.values)
            n_offset += tree.features.shape[0]
            v_offset += tree.values.shape[0]
        self.node_offsets[n_trees] = n_offset
        self.features = np.concatenate(feat_chunks)
        self.thresholds = np.concatenate(thr_chunks)
        self.left_child = np.concatenate(left_chunks)
        self.right_child = np.concatenate(right_chunks)
        self.leaf_index = np.concatenate(leaf_chunks)
        if vector_values:
            self.values = np.vstack(value_chunks)
        else:
            self.values = np.concatenate(value_chunks)
        self.class_ids = class_ids

    def add_predict(self, X_binned, out):
        if get_num_threads() > 1 and X_binned.shape[0] >= _PARALLEL_MIN_ROWS:
            _flat_nonoblivious_add_parallel(
                X_binned, self.node_offsets, self.features, self.thresholds,
                self.left_child, self.right_child, self.leaf_index,
                self.value_offsets, self.values, out
            )
        else:
            _flat_nonoblivious_add(
                X_binned, self.node_offsets, self.features, self.thresholds,
                self.left_child, self.right_child, self.leaf_index,
                self.value_offsets, self.values, out
            )

    def add_predict_class_major(self, X_binned, out):
        if self.class_ids is not None:
            _flat_nonoblivious_class_add_parallel(
                X_binned, self.node_offsets, self.features, self.thresholds,
                self.left_child, self.right_child, self.leaf_index,
                self.value_offsets, self.values, self.class_ids, out
            )
        else:
            _flat_multi_add_parallel(
                X_binned, self.node_offsets, self.features, self.thresholds,
                self.left_child, self.right_child, self.leaf_index,
                self.value_offsets, self.values, out
            )


def flat_predict_preferred(flat):
    """Empirical routing for batch prediction.

    The fused kernel is a clear win for oblivious ensembles with threads
    (branch-free fixed-depth walks; the per-tree oblivious kernel is serial).
    For explicit-node trees the per-tree parallel loop measured at parity or
    better at every thread count tried, and single-threaded the per-tree
    loop won for every family, so those keep the loop.
    """
    return (
        isinstance(flat, FlatObliviousEnsemble) and get_num_threads() > 1
    )


def build_flat_ensemble(trees):
    """Flatten a scalar-booster tree list, or None if a type is unsupported."""
    if not trees:
        return None
    if all(type(t) is ObliviousTree for t in trees):
        return FlatObliviousEnsemble(trees)
    if all(type(t) is NonObliviousTree for t in trees):
        return FlatNonObliviousEnsemble(trees)
    return None


def build_flat_multiclass_ensemble(rounds, n_classes):
    """Flatten multiclass rounds (per-class scalar trees or shared-vector
    trees) into one ensemble, or None if unsupported."""
    if not rounds:
        return None
    if all(type(r) is MultiNonObliviousTree for r in rounds):
        return FlatNonObliviousEnsemble(rounds, vector_values=True)
    flat_trees = []
    class_ids = []
    for round_trees in rounds:
        if isinstance(round_trees, (list, tuple)):
            if len(round_trees) != n_classes:
                return None
            for k, tree in enumerate(round_trees):
                flat_trees.append(tree)
                class_ids.append(k)
        else:
            return None
    class_ids = np.array(class_ids, dtype=np.int64)
    if all(type(t) is ObliviousTree for t in flat_trees):
        return FlatObliviousEnsemble(flat_trees, class_ids=class_ids)
    if all(type(t) is NonObliviousTree for t in flat_trees):
        return FlatNonObliviousEnsemble(flat_trees, class_ids=class_ids)
    return None
