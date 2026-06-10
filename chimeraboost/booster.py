"""The gradient boosting core: builds the full additive model.

Two boosters share the same machinery (FeaturePreprocessor, oblivious trees):
  * GradientBoosting     -> scalar output (regression, binary classification)
  * MulticlassBoosting   -> K simultaneous outputs (softmax multiclass)
"""

import time
import numpy as np

from .binning import DEFAULT_BIN_SAMPLE_COUNT
from .flat_model import (
    build_flat_ensemble,
    build_flat_multiclass_ensemble,
    flat_predict_preferred,
)
from .losses import LOSSES, MultiSoftmax
from .preprocessing import FeaturePreprocessor
from .tree import (
    _build_multiclass_histograms_counts_into,
    add_leaf_values_inplace,
    add_multiclass_leaf_values_inplace,
    build_leafwise_multiclass_tree,
    build_leafwise_tree,
    build_levelwise_tree,
    build_oblivious_tree,
    ordered_leaf_update_inplace,
)

_LEAF_CORRECTION_SORT_MIN_LEAVES = 16


def _apply_thread_count(thread_count):
    """Set numba's thread pool size. None / -1 means use all detected cores.

    Returns the effective thread count so callers can record it.
    """
    import numba
    max_threads = numba.config.NUMBA_NUM_THREADS
    if thread_count is None or thread_count < 0:
        n = max_threads
    else:
        n = max(1, min(int(thread_count), max_threads))
    numba.set_num_threads(n)
    return n


def _fit_thread_count(thread_count, tree_mode_, n_samples):
    """Choose the effective Numba thread count for one fit.

    The leaf-wise builder's medium/large tree kernels are memory-traffic heavy;
    on these row counts, extra threads routinely add scheduling/cache overhead.
    Treat the public thread_count as a maximum and cap smaller LightGBM-mode
    fits at two threads. Larger fits can still use the caller's requested count.
    """
    if tree_mode_ == "lightgbm" and n_samples <= 50_000:
        if thread_count is None or thread_count < 0:
            return _apply_thread_count(2)
        return _apply_thread_count(min(int(thread_count), 2))
    return _apply_thread_count(thread_count)


def _auto_learning_rate(n_samples, iterations, early_stopping):
    """Pick a default learning rate when the user did not specify one.

    With early stopping, we default to 0.1. Without early stopping, the rate
    scales inversely with the iteration budget.
    """
    if early_stopping:
        return 0.1
    lr = 20.0 / max(iterations, 1)
    return float(np.clip(lr, 0.03, 0.2))


def _one_hot_class_major(y_idx, n_classes):
    """Build one-hot targets directly as class-major rows."""
    y_idx = np.asarray(y_idx, dtype=np.int64)
    out = np.zeros((int(n_classes), y_idx.shape[0]), dtype=np.float64)
    out[y_idx, np.arange(y_idx.shape[0])] = 1.0
    return out


def _new_timing(enabled):
    if not enabled:
        return None
    return {
        "preprocess": 0.0,
        "grad_hess": 0.0,
        "tree_build": 0.0,
        "train_update": 0.0,
        "validation_predict": 0.0,
        "loss_eval": 0.0,
    }


def _add_timing(timing, key, start):
    if timing is not None:
        timing[key] += time.perf_counter() - start


def _start_timing(timing):
    return time.perf_counter() if timing is not None else 0.0


def _normalize_tree_mode(tree_mode):
    """Map public tree-mode aliases to the internal builder family."""
    if tree_mode is None:
        tree_mode = "catboost"
    mode = str(tree_mode).lower().replace("-", "_")
    if mode in {"catboost", "oblivious", "symmetric"}:
        return "catboost"
    if mode in {"lightgbm", "leafwise", "leaf_wise"}:
        return "lightgbm"
    if mode in {"levelwise", "level_wise", "depthwise", "depth_wise",
                "non_oblivious"}:
        return "depthwise"
    raise ValueError(
        "tree_mode must be one of 'catboost', 'oblivious', "
        "'lightgbm', or experimental 'depthwise'"
    )


def _normalize_sampling(sampling):
    if sampling is None:
        sampling = "uniform"
    mode = str(sampling).lower().replace("-", "_")
    if mode in {"uniform", "random"}:
        return "uniform"
    if mode == "goss":
        return "goss"
    raise ValueError("sampling must be one of 'uniform' or experimental 'goss'")


def _validate_sample_weight(sample_weight, n_samples, name="sample_weight"):
    """Return a mean-one validated weight vector, or None."""
    if sample_weight is None:
        return None
    w = np.asarray(sample_weight, dtype=np.float64)
    if w.shape != (n_samples,):
        raise ValueError(f"{name} must have shape ({n_samples},)")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(w < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    total = w.sum()
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total weight")
    return w * (n_samples / total)


def _resolve_default_depth(depth, tree_mode_):
    if depth is not None:
        return int(depth)
    if tree_mode_ == "lightgbm":
        return -1
    return 6


class _BaseBooster:
    """Shared machinery for the scalar and multiclass boosters.

    Holds the common hyperparameters and the helpers both subclasses use:
    histogram-buffer allocation, column subsampling, row subsampling, feature
    preprocessing, and split-gain feature importances. Subclasses implement
    `fit` and `predict_raw`.
    """

    def __init__(self, iterations=500, learning_rate=None, depth=None,
                 l2_leaf_reg=3.0, max_bins=128, subsample=1.0,
                 colsample=1.0, cat_smoothing=1.0, early_stopping_rounds=None,
                 min_child_weight=1.0, min_child_samples=20,
                 min_gain_to_split=0.0, num_leaves=None, thread_count=None,
                 random_state=None, verbose=False, ordered_boosting="auto",
                 verbose_timing=False, tree_mode="catboost",
                 sampling="uniform", top_rate=0.2, other_rate=0.1,
                 multiclass_tree_strategy="auto", eval_train_loss=True,
                 bin_sample_count=DEFAULT_BIN_SAMPLE_COUNT,
                 histogram_parallelism="auto"):
        self.iterations = int(iterations)
        self.learning_rate = learning_rate
        self.l2_leaf_reg = float(l2_leaf_reg)
        self.max_bins = int(max_bins)
        self.subsample = float(subsample)
        self.colsample = float(colsample)
        self.cat_smoothing = float(cat_smoothing)
        if self.cat_smoothing <= 0.0:
            raise ValueError("cat_smoothing must be positive")
        self.early_stopping_rounds = early_stopping_rounds
        self.min_child_weight = float(min_child_weight)
        self.min_child_samples = int(min_child_samples)
        self.min_gain_to_split = float(min_gain_to_split)
        self.num_leaves = num_leaves
        self.thread_count = thread_count
        self.random_state = random_state
        self.verbose = verbose
        self.ordered_boosting = ordered_boosting
        self.verbose_timing = bool(verbose_timing)
        self.tree_mode = tree_mode
        self.tree_mode_ = _normalize_tree_mode(tree_mode)
        self.depth = _resolve_default_depth(depth, self.tree_mode_)
        self.sampling = sampling
        self.top_rate = float(top_rate)
        self.other_rate = float(other_rate)
        self.multiclass_tree_strategy = multiclass_tree_strategy
        self.eval_train_loss = bool(eval_train_loss)
        self.bin_sample_count = bin_sample_count
        self.histogram_parallelism = histogram_parallelism
        if histogram_parallelism not in {"auto", "feature", "row"}:
            raise ValueError(
                "histogram_parallelism must be 'auto', 'feature', or 'row'"
            )
        self._validate_sampling_config()

    def _validate_sampling_config(self):
        self.sampling_ = _normalize_sampling(self.sampling)
        self.top_rate = float(self.top_rate)
        self.other_rate = float(self.other_rate)
        if self.sampling_ == "goss":
            if not (0.0 < self.top_rate < 1.0):
                raise ValueError("top_rate must be in (0, 1) for sampling='goss'")
            if not (0.0 < self.other_rate < 1.0):
                raise ValueError("other_rate must be in (0, 1) for sampling='goss'")
            if self.top_rate + self.other_rate > 1.0:
                raise ValueError(
                    "top_rate + other_rate must be <= 1 for sampling='goss'"
                )

    def _catboost_depth(self):
        if self.depth is None or self.depth < 1:
            raise ValueError("depth must be positive for tree_mode='catboost'")
        return int(self.depth)

    def _max_tree_leaves(self):
        if self.tree_mode_ in {"catboost", "depthwise"}:
            if self.num_leaves is not None:
                raise ValueError("num_leaves is only supported with tree_mode='lightgbm'")
            return 1 << self._catboost_depth()
        if self.depth == 0 or (self.depth is not None and self.depth < -1):
            raise ValueError("depth must be positive, None, or -1 for tree_mode='lightgbm'")
        if self.num_leaves is None:
            if self.depth is None or self.depth < 0:
                return 31
            return min(31, 1 << int(self.depth))
        max_leaves = int(self.num_leaves)
        if max_leaves < 1:
            raise ValueError("num_leaves must be at least 1")
        if self.depth is not None and self.depth > 0:
            max_leaves = min(max_leaves, 1 << int(self.depth))
        return max_leaves

    def _max_tree_depth(self):
        if self.tree_mode_ in {"catboost", "depthwise"}:
            return self._catboost_depth()
        if self.depth is None:
            return -1
        return int(self.depth)

    def _resolve_ordered_boosting(self):
        if self.ordered_boosting == "auto":
            return self.tree_mode_ in {"catboost", "depthwise"}
        resolved = bool(self.ordered_boosting)
        if self.tree_mode_ == "lightgbm" and resolved:
            raise ValueError(
                "ordered_boosting=True is only supported with tree_mode='catboost'"
            )
        return resolved

    # Interleaving wins by touching one cache line per (row, feature) instead
    # of two or three, which pays off while per-thread bandwidth is the
    # constraint; at higher thread counts the shared memory bus saturates
    # either way and the measured effect is neutral-to-negative.
    _HIST_INTERLEAVE_MAX_THREADS = 4

    def _alloc_hist_buffers(self, n_features, n_bins):
        """Allocate reusable histogram buffers once per fit.

        Shape (n_features, max_tree_leaves, max_bins). Reused for every tree and level
        via _build_histograms_into, which zeroes the active slice each call.
        This avoids reallocating these (potentially large) arrays thousands of
        times over a long boosting run.

        At low thread counts the gradient/hessian(/count) buffers are lane
        views into one interleaved (n_features, leaves, bins, n_arrays) base
        array, so each bin's statistics share a cache line. The kernels are
        layout-agnostic and the per-bin summation order is unchanged, so
        results are bitwise identical to separate buffers.
        """
        max_leaves = self._max_tree_leaves()
        max_bins = int(n_bins.max()) if len(n_bins) else 1
        n_arrays = 3 if self.tree_mode_ == "lightgbm" else 2
        n_threads = getattr(self, "n_threads_", 1)
        if n_threads <= self._HIST_INTERLEAVE_MAX_THREADS:
            base = np.zeros((n_features, max_leaves, max_bins, n_arrays))
            return tuple(base[..., k] for k in range(n_arrays))
        return tuple(
            np.zeros((n_features, max_leaves, max_bins))
            for _ in range(n_arrays)
        )

    def _alloc_split_buffers(self, n_features):
        """Allocate reusable per-feature split-search scratch buffers."""
        max_leaves = self._max_tree_leaves()
        return tuple(np.empty((n_features, max_leaves)) for _ in range(5))

    _ROWPAR_MAX_BYTES = 512 * 1024 * 1024

    def _alloc_rowpar_buffers(self, n_features, n_bins, n_samples):
        """Thread-local accumulators for the row-parallel histogram kernels.

        Opt-in via histogram_parallelism='row'. The row-parallel kernels read
        grad/hess/leaf once instead of once per feature, which targets
        machines where those streams fall out of cache; on the Apple-silicon
        dev box they measured slower than the feature-parallel kernels at
        every size tried, so 'auto' currently means 'feature'. Returns None
        for single-threaded fits, fits too small for the eligibility rule to
        ever fire, or local buffers beyond the memory budget.
        """
        if self.histogram_parallelism != "row":
            return None
        if self.n_threads_ <= 1 or self.tree_mode_ == "depthwise":
            return None
        max_bins = int(n_bins.max()) if len(n_bins) else 1
        n_chunks = self.n_threads_
        if n_samples < 4 * n_chunks * max_bins:
            return None
        if self.tree_mode_ == "lightgbm":
            leaf_slots, n_arrays = 1, 3
        else:
            leaf_slots, n_arrays = self._max_tree_leaves(), 2
        n_bytes = 8 * n_arrays * n_chunks * n_features * leaf_slots * max_bins
        if n_bytes > self._ROWPAR_MAX_BYTES:
            return None
        shape = (n_chunks, n_features, leaf_slots, max_bins)
        return tuple(np.zeros(shape) for _ in range(n_arrays))

    def _alloc_multiclass_hist_buffers(self, n_classes, n_features, n_bins):
        """Allocate reusable class-major LightGBM-mode histogram buffers."""
        max_leaves = self._max_tree_leaves()
        max_bins = int(n_bins.max()) if len(n_bins) else 1
        hg = np.zeros((n_classes, n_features, max_leaves, max_bins))
        hh = np.zeros((n_classes, n_features, max_leaves, max_bins))
        hc = np.zeros((n_features, max_leaves, max_bins))
        return hg, hh, hc

    def _feature_selection(self, n_cols, rng):
        """Return a 0/1 mask and selected column indices for one tree."""
        if self.colsample >= 1.0:
            return None, None
        k = max(1, int(round(self.colsample * n_cols)))
        mask = np.zeros(n_cols, dtype=np.int64)
        idx = np.sort(rng.choice(n_cols, size=k, replace=False)).astype(np.int64)
        mask[idx] = 1
        return mask, idx

    def _new_preprocessor(self):
        """Build a FeaturePreprocessor configured from this booster's params."""
        cat_smoothing = self.cat_smoothing
        if (
            self.tree_mode_ == "lightgbm"
            and getattr(self, "loss_name", None) == "RMSE"
            and cat_smoothing == 1.0
            and self._include_cat_codes()
        ):
            # LightGBM-mode regression uses both K-fold target statistics and
            # raw category-code features. A little extra smoothing keeps the
            # target-stat columns from over-specializing before the code columns
            # have a chance to split.
            cat_smoothing = 3.0
        return FeaturePreprocessor(self.max_bins, cat_smoothing,
                                   self.random_state,
                                   include_cat_codes=self._include_cat_codes(),
                                   target_encoding_mode=(
                                       "kfold"
                                       if self.tree_mode_ == "lightgbm"
                                       else "ordered"
                                   ),
                                   bin_sample_count=self.bin_sample_count)

    def _include_cat_codes(self):
        return self.tree_mode_ == "lightgbm"

    def _tree_builder(self):
        if self.tree_mode_ == "lightgbm":
            return build_leafwise_tree
        if self.tree_mode_ == "depthwise":
            return build_levelwise_tree
        return build_oblivious_tree

    def _maybe_subsample(self, grad, hess, rng):
        """Return zeroed gradients plus sampled row indices for one tree.

        Unsampled rows keep zero grad/hess so ordered boosting updates preserve
        the old fallback behavior, while histogram builders can skip them.
        """
        if self.sampling_ == "goss":
            return self._goss_subsample(grad, hess, rng)
        if self.subsample >= 1.0:
            return grad, hess, None
        mask = rng.random(grad.shape[0]) < self.subsample
        row_indices = np.flatnonzero(mask).astype(np.int64)
        return np.where(mask, grad, 0.0), np.where(mask, hess, 0.0), row_indices

    def _goss_subsample(self, grad, hess, rng):
        """Gradient-based one-side sampling for scalar boosters.

        Keeps all large-gradient rows and samples a fraction of the remaining
        rows, scaling the sampled small-gradient rows to preserve total mass.
        """
        n_samples = grad.shape[0]
        if n_samples <= 1:
            return grad, hess, None
        top_count = min(n_samples, max(1, int(round(self.top_rate * n_samples))))
        remaining_count = n_samples - top_count
        if remaining_count <= 0:
            return grad, hess, None
        other_count = min(
            remaining_count, max(1, int(round(self.other_rate * n_samples)))
        )

        abs_grad = np.abs(grad)
        top_idx = np.argpartition(abs_grad, n_samples - top_count)[-top_count:]
        remaining_mask = np.ones(n_samples, dtype=bool)
        remaining_mask[top_idx] = False
        remaining_idx = np.flatnonzero(remaining_mask)
        if other_count >= remaining_idx.shape[0]:
            other_idx = remaining_idx
        else:
            other_idx = rng.choice(remaining_idx, size=other_count, replace=False)

        if top_idx.shape[0] + other_idx.shape[0] == n_samples:
            return grad, hess, None

        g = np.zeros_like(grad)
        h = np.zeros_like(hess)
        g[top_idx] = grad[top_idx]
        h[top_idx] = hess[top_idx]
        scale = remaining_count / max(other_idx.shape[0], 1)
        g[other_idx] = grad[other_idx] * scale
        h[other_idx] = hess[other_idx] * scale
        row_indices = np.sort(
            np.concatenate((top_idx.astype(np.int64), other_idx.astype(np.int64)))
        )
        return g, h, row_indices

    def _goss_subsample_multiclass(self, grad, hess, rng):
        """Gradient-based one-side sampling for class-major (K, n) gradients.

        Rows are ranked by the L1 norm of their per-class gradients; the kept
        row set and the small-gradient scaling are shared across classes so
        every class tree in the round sees the same sample.
        """
        n_samples = grad.shape[1]
        if n_samples <= 1:
            return grad, hess, None
        top_count = min(n_samples, max(1, int(round(self.top_rate * n_samples))))
        remaining_count = n_samples - top_count
        if remaining_count <= 0:
            return grad, hess, None
        other_count = min(
            remaining_count, max(1, int(round(self.other_rate * n_samples)))
        )

        abs_grad = np.abs(grad).sum(axis=0)
        top_idx = np.argpartition(abs_grad, n_samples - top_count)[-top_count:]
        remaining_mask = np.ones(n_samples, dtype=bool)
        remaining_mask[top_idx] = False
        remaining_idx = np.flatnonzero(remaining_mask)
        if other_count >= remaining_idx.shape[0]:
            other_idx = remaining_idx
        else:
            other_idx = rng.choice(remaining_idx, size=other_count, replace=False)

        if top_idx.shape[0] + other_idx.shape[0] == n_samples:
            return grad, hess, None

        g = np.zeros_like(grad)
        h = np.zeros_like(hess)
        g[:, top_idx] = grad[:, top_idx]
        h[:, top_idx] = hess[:, top_idx]
        scale = remaining_count / max(other_idx.shape[0], 1)
        g[:, other_idx] = grad[:, other_idx] * scale
        h[:, other_idx] = hess[:, other_idx] * scale
        row_indices = np.sort(
            np.concatenate((top_idx.astype(np.int64), other_idx.astype(np.int64)))
        )
        return g, h, row_indices

    def _builder_kwargs(self, fmask, findices, row_indices,
                        hist_buffers, split_buffers, X_hist_binned,
                        use_constant_hessian, hessian_always_positive=False,
                        rowpar_buffers=None):
        kwargs = {
            "feature_mask": fmask,
            "min_child_weight": self.min_child_weight,
            "hist_buffers": hist_buffers,
            "split_buffers": split_buffers,
            "return_training_state": True,
            "X_hist_binned": X_hist_binned,
            "feature_indices": findices,
            "row_indices": row_indices,
            "constant_hessian": use_constant_hessian,
            "rowpar_buffers": rowpar_buffers,
        }
        if self.tree_mode_ == "lightgbm":
            kwargs.update(
                max_leaves=self._max_tree_leaves(),
                min_child_samples=self.min_child_samples,
                min_gain_to_split=self.min_gain_to_split,
                hessian_always_positive=hessian_always_positive,
            )
        return kwargs

    def _accumulate_importance(self, tree):
        """Add this tree's per-split gains to the running importance totals,
        mapped from internal columns back to original input features."""
        for f, g in zip(tree.splits_feat, tree.gains):
            orig = self.prep_.feature_map_[f]
            self._importance[orig] += g

    def _flat_ensemble(self):
        """Build (lazily, once per fitted tree list) the flattened ensemble
        used for batch prediction. Returns None for unsupported tree types,
        in which case callers fall back to the per-tree loop. The cache is
        keyed on the trees_ list identity, so a refit (which rebinds trees_)
        invalidates it exactly."""
        cache = getattr(self, "_flat_cache_", None)
        if cache is not None and cache[0] is self.trees_:
            return cache[1]
        flat = self._build_flat_ensemble()
        self._flat_cache_ = (self.trees_, flat)
        return flat

    @property
    def feature_importances_(self):
        """Total split gain per ORIGINAL input column, normalized to sum 1."""
        imp = self._importance.copy()
        s = imp.sum()
        return imp / s if s > 0 else imp

    def save_model(self, path):
        """Serialize this fitted booster to a single ``.npz`` file."""
        from .serialization import save_booster
        save_booster(self, path)

    @classmethod
    def load_model(cls, path):
        """Load a booster saved with :meth:`save_model`."""
        from .serialization import load_booster
        booster = load_booster(path)
        if cls is not _BaseBooster and not isinstance(booster, cls):
            raise TypeError(
                f"{path!r} contains a {type(booster).__name__}, "
                f"not a {cls.__name__}"
            )
        return booster


class GradientBoosting(_BaseBooster):
    """Scalar booster: regression and binary classification."""

    def __init__(self, loss="RMSE", loss_kwargs=None, **kw):
        super().__init__(**kw)
        self.loss_name = loss
        self.loss_kwargs = loss_kwargs or {}

    def _include_cat_codes(self):
        return self.tree_mode_ == "lightgbm" and self.loss_name in {"Logloss", "RMSE"}

    def fit(self, X, y, cat_features=None, eval_set=None, sample_weight=None,
            eval_sample_weight=None):
        """Fit the additive model. Optionally pass `cat_features` (column indices
        to target-encode) and `eval_set=(X_val, y_val)` for early stopping.
        `sample_weight` is a 1-D array of per-sample weights; None means uniform.
        Weights are normalized to mean 1 internally so the gradient scale stays
        comparable to the no-weight case."""
        X = (np.asarray(X, dtype=object) if cat_features
             else np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64)
        n_samples = X.shape[0]

        # Normalize weights to mean=1. np.ones(n) stays np.ones(n), so
        # sample_weight=np.ones(n) is bitwise-equivalent to sample_weight=None
        # for all losses except MAE/Quantile (which use a different quantile
        # algorithm when weights are present).
        self.tree_mode_ = _normalize_tree_mode(self.tree_mode)
        self.n_threads_ = _fit_thread_count(
            self.thread_count, self.tree_mode_, n_samples
        )
        self._validate_sampling_config()
        if self.sampling_ == "goss" and self.subsample < 1.0:
            raise ValueError(
                "sampling='goss' controls row sampling; use subsample=1.0"
            )
        self.ordered_boosting_ = self._resolve_ordered_boosting()
        w = _validate_sample_weight(sample_weight, n_samples)
        self.loss_ = LOSSES[self.loss_name](**self.loss_kwargs)
        use_constant_hessian = (
            getattr(self.loss_, "constant_hessian", False)
            and w is None
            and self.sampling_ != "goss"
        )
        hessian_always_positive = (
            self.tree_mode_ == "lightgbm"
            and self.loss_name == "Logloss"
            and w is None
        )
        _es = self.early_stopping_rounds is not None and eval_set is not None
        self.lr_ = (self.learning_rate if self.learning_rate is not None
                    else _auto_learning_rate(n_samples, self.iterations, _es))

        timing = _new_timing(self.verbose_timing)
        self.timing_ = timing
        phase = _start_timing(timing)
        self.prep_ = self._new_preprocessor()
        X_binned = self.prep_.fit_transform(
            X, [y], cat_features, sample_weight=w
        )
        X_hist_binned = (
            np.asfortranarray(X_binned) if self.n_threads_ > 1 else X_binned
        )
        n_bins = self.prep_.n_bins_
        hist_buffers = self._alloc_hist_buffers(X_binned.shape[1], n_bins)
        split_buffers = (
            self._alloc_split_buffers(X_binned.shape[1])
            if self.n_threads_ > 1 else None
        )
        rowpar_buffers = self._alloc_rowpar_buffers(
            X_binned.shape[1], n_bins, n_samples
        )
        self._importance = np.zeros(self.prep_.n_input_features_)

        Xv_binned = yv = Fv = wv = None
        if eval_set is not None:
            Xv, yv = eval_set
            Xv = (np.asarray(Xv, dtype=object) if cat_features
                  else np.asarray(Xv, dtype=np.float64))
            yv = np.asarray(yv, dtype=np.float64)
            wv = _validate_sample_weight(
                eval_sample_weight, len(yv), name="eval_sample_weight"
            )
            Xv_binned = self.prep_.transform(Xv)
        _add_timing(timing, "preprocess", phase)

        self.init_ = self.loss_.init(y, w)
        F = np.full(n_samples, self.init_, dtype=np.float64)
        grad_buffer = np.empty(n_samples, dtype=np.float64)
        hess_buffer = np.empty(n_samples, dtype=np.float64)
        if yv is not None:
            Fv = np.full(len(yv), self.init_)

        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        self._flat_cache_ = None  # drop any previous fit's flattened ensemble
        self.train_history_, self.valid_history_ = [], []
        # Train loss is diagnostic only (early stopping watches the eval set),
        # so skipping it when nobody will read it saves one full O(n) pass per
        # round. verbose forces it back on because the progress log prints it.
        eval_train = self.eval_train_loss or bool(self.verbose)
        best_score, best_iter = np.inf, 0
        t0 = time.time()

        for m in range(self.iterations):
            phase = _start_timing(timing)
            if hasattr(self.loss_, "grad_hess_into"):
                self.loss_.grad_hess_into(y, F, w, grad_buffer, hess_buffer)
                grad, hess = grad_buffer, hess_buffer
            else:
                grad, hess = self.loss_.grad_hess(y, F)
                if w is not None:
                    grad = grad * w
                    hess = hess * w
            g, h, row_indices = self._maybe_subsample(grad, hess, rng)
            fmask, findices = self._feature_selection(X_binned.shape[1], rng)
            _add_timing(timing, "grad_hess", phase)

            phase = _start_timing(timing)
            tree, leaf, leaf_G, leaf_H = self._tree_builder()(
                X_binned, g, h, n_bins, self._max_tree_depth(),
                self.l2_leaf_reg, self.lr_,
                **self._builder_kwargs(
                    fmask, findices, row_indices, hist_buffers, split_buffers,
                    X_hist_binned, use_constant_hessian,
                    hessian_always_positive=hessian_always_positive,
                    rowpar_buffers=rowpar_buffers
                ),
            )
            _add_timing(timing, "tree_build", phase)
            # A depth-0 tree found no legal split; subsequent rounds on the same
            # gradients would too, so stop rather than append empty trees.
            if tree.depth == 0:
                if self.verbose:
                    print(f"No further splits at iteration {m}; stopping.")
                break
            phase = _start_timing(timing)
            if getattr(self.loss_, "adjusts_leaves", False):
                self._correct_leaves(tree, X_binned, y - F, w, leaf=leaf)
            self.trees_.append(tree)
            self._accumulate_importance(tree)
            if self.ordered_boosting_ and not getattr(self.loss_, "adjusts_leaves", False):
                # Leave-one-out leaf step: each row's update uses its leaf's
                # gradient/hessian totals with that row's own contribution
                # removed, reducing the self-reinforcement of plain boosting.
                # tree.values keeps the standard Newton values for inference;
                # only the training F uses this corrected update. Subsampled-out
                # rows (g=h=0) fall back to the standard leaf value.
                ordered_leaf_update_inplace(
                    leaf, leaf_G, leaf_H, g, h, self.lr_, self.l2_leaf_reg, F
                )
            elif self.tree_mode_ == "lightgbm":
                add_leaf_values_inplace(leaf, tree.values, F)
            else:
                tree.add_predict(X_binned, F)
            _add_timing(timing, "train_update", phase)

            if eval_train:
                phase = _start_timing(timing)
                self.train_history_.append(self.loss_.eval(y, F, w))
                _add_timing(timing, "loss_eval", phase)

            if Fv is not None:
                phase = _start_timing(timing)
                tree.add_predict(Xv_binned, Fv)
                _add_timing(timing, "validation_predict", phase)
                phase = _start_timing(timing)
                val = self.loss_.eval(yv, Fv, wv)
                _add_timing(timing, "loss_eval", phase)
                self.valid_history_.append(val)
                if val < best_score - 1e-9:
                    best_score, best_iter = val, m
                elif (self.early_stopping_rounds and
                      m - best_iter >= self.early_stopping_rounds):
                    if self.verbose:
                        print(f"Early stop at {m} (best {best_iter}, "
                              f"val {best_score:.5f})")
                    self.trees_ = self.trees_[: best_iter + 1]
                    break

            if self.verbose and (m % max(1, self.iterations // 10) == 0):
                msg = f"[{m}] train {self.train_history_[-1]:.5f}"
                if Fv is not None:
                    msg += f"  val {self.valid_history_[-1]:.5f}"
                print(msg)

        self.fit_time_ = time.time() - t0
        self.best_iteration_ = len(self.trees_)
        if self.valid_history_:
            self.best_score_ = best_score
        elif Fv is not None:
            self.best_score_ = self.loss_.eval(yv, Fv, wv)
        elif self.train_history_:
            self.best_score_ = self.train_history_[-1]
        else:
            self.best_score_ = self.loss_.eval(y, F, w)
        return self

    def _correct_leaves(self, tree, X_binned, residuals, sample_weight=None, leaf=None):
        """Override Newton leaf values with the loss-appropriate residual
        statistic (median for MAE, alpha-quantile for Quantile). The tree
        structure was chosen by the gradient; this fixes the step size."""
        if leaf is None:
            leaf = tree.apply(X_binned)
        n_leaves = tree.values.shape[0]
        if n_leaves < _LEAF_CORRECTION_SORT_MIN_LEAVES:
            for l in range(n_leaves):
                mask = leaf == l
                r = residuals[mask]
                w = sample_weight[mask] if sample_weight is not None else None
                tree.values[l] = self.lr_ * self.loss_.leaf_value(r, w)
            return

        order = np.argsort(leaf)
        residuals_sorted = residuals[order]
        weights_sorted = sample_weight[order] if sample_weight is not None else None
        counts = np.bincount(leaf, minlength=n_leaves)

        start = 0
        for l in range(n_leaves):
            end = start + counts[l]
            r = residuals_sorted[start:end]
            w = weights_sorted[start:end] if weights_sorted is not None else None
            tree.values[l] = self.lr_ * self.loss_.leaf_value(r, w)
            start = end

    def _build_flat_ensemble(self):
        return build_flat_ensemble(self.trees_)

    def predict_raw(self, X):
        """Return raw additive scores (pre-link): the regression prediction, or
        the log-odds for binary classification."""
        X = (np.asarray(X, dtype=object) if self.prep_.cat_features_
             else np.asarray(X, dtype=np.float64))
        X_binned = self.prep_.transform(X)
        F = np.full(X_binned.shape[0], self.init_, dtype=np.float64)
        flat = self._flat_ensemble()
        if flat is not None and flat_predict_preferred(flat):
            flat.add_predict(X_binned, F)
        else:
            for tree in self.trees_:
                tree.add_predict(X_binned, F)
        return F

    def staged_predict_raw(self, X):
        """Yield the cumulative raw prediction after each tree (1..n_trees)."""
        X = (np.asarray(X, dtype=object) if self.prep_.cat_features_
             else np.asarray(X, dtype=np.float64))
        X_binned = self.prep_.transform(X)
        F = np.full(X_binned.shape[0], self.init_, dtype=np.float64)
        for tree in self.trees_:
            tree.add_predict(X_binned, F)
            yield F.copy()


class MulticlassBoosting(_BaseBooster):
    """Softmax multiclass booster: fits K trees per round (one per class)."""

    def fit(self, X, y, cat_features=None, eval_set=None, sample_weight=None,
            eval_sample_weight=None):
        """Fit K trees per boosting round (one per class) under softmax loss.
        Same `cat_features` / `eval_set` / `sample_weight` semantics as the
        scalar booster."""
        X = (np.asarray(X, dtype=object) if cat_features
             else np.asarray(X, dtype=np.float64))
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        K = self.classes_.size
        self.n_classes_ = K
        y_idx = np.searchsorted(self.classes_, y)
        Y_class = _one_hot_class_major(y_idx, K)  # class-major (K, n)
        n_samples = X.shape[0]

        self.tree_mode_ = _normalize_tree_mode(self.tree_mode)
        self.n_threads_ = _fit_thread_count(
            self.thread_count, self.tree_mode_, n_samples
        )
        self._validate_sampling_config()
        if self.sampling_ == "goss" and self.subsample < 1.0:
            raise ValueError(
                "sampling='goss' controls row sampling; use subsample=1.0"
            )
        self.ordered_boosting_ = self._resolve_ordered_boosting()
        w = _validate_sample_weight(sample_weight, n_samples)
        self.loss_ = MultiSoftmax(K)
        _es = self.early_stopping_rounds is not None and eval_set is not None
        self.lr_ = (self.learning_rate if self.learning_rate is not None
                    else _auto_learning_rate(n_samples, self.iterations, _es))
        self.l2_leaf_reg_ = self.l2_leaf_reg
        if self.tree_mode_ == "lightgbm" and self.l2_leaf_reg == 3.0:
            self.l2_leaf_reg_ = 1.0

        timing = _new_timing(self.verbose_timing)
        self.timing_ = timing
        phase = _start_timing(timing)
        # One ordered-TS target per class (CatBoost-style per-class statistics).
        self.prep_ = self._new_preprocessor()
        X_binned = self.prep_.fit_transform(X, [Y_class[k] for k in range(K)],
                                            cat_features, sample_weight=w)
        X_hist_binned = (
            np.asfortranarray(X_binned) if self.n_threads_ > 1 else X_binned
        )
        n_bins = self.prep_.n_bins_
        hist_buffers = self._alloc_hist_buffers(X_binned.shape[1], n_bins)
        split_buffers = (
            self._alloc_split_buffers(X_binned.shape[1])
            if self.n_threads_ > 1 else None
        )
        rowpar_buffers = self._alloc_rowpar_buffers(
            X_binned.shape[1], n_bins, n_samples
        )
        strategy = self.multiclass_tree_strategy
        if strategy not in {"auto", "per_class", "shared_vector"}:
            raise ValueError(
                "multiclass_tree_strategy must be 'auto', 'per_class', "
                "or 'shared_vector'"
            )
        can_use_shared_lightgbm_multiclass = (
            self.tree_mode_ == "lightgbm"
            and self.sampling_ == "uniform"
            and self.subsample >= 1.0
            and self.colsample >= 1.0
            and not self.ordered_boosting_
        )
        if strategy == "shared_vector" and not can_use_shared_lightgbm_multiclass:
            raise ValueError(
                "multiclass_tree_strategy='shared_vector' requires "
                "tree_mode='lightgbm', no ordered boosting, and full row/column sampling"
            )
        use_shared_lightgbm_multiclass = (
            can_use_shared_lightgbm_multiclass
            and (
                strategy == "shared_vector"
                or (strategy == "auto" and bool(cat_features))
            )
        )
        self.multiclass_tree_strategy_ = (
            "shared_vector" if use_shared_lightgbm_multiclass else "per_class"
        )
        multiclass_hist_buffers = (
            self._alloc_multiclass_hist_buffers(K, X_binned.shape[1], n_bins)
            if use_shared_lightgbm_multiclass else None
        )
        # One fused class-major pass per round builds every class's root
        # histogram in a single scan of X; per-class builders then copy
        # their slice instead of re-scanning all rows K times.
        use_fused_root = (
            not use_shared_lightgbm_multiclass
            and self.tree_mode_ in {"catboost", "lightgbm"}
            and self.sampling_ == "uniform"
            and self.subsample >= 1.0
            and self.colsample >= 1.0
        )
        if use_fused_root:
            max_bins_ = int(n_bins.max()) if len(n_bins) else 1
            root_g = np.zeros((K, X_binned.shape[1], 1, max_bins_))
            root_h = np.zeros((K, X_binned.shape[1], 1, max_bins_))
            root_c = np.zeros((X_binned.shape[1], 1, max_bins_))
            root_leaf = np.zeros(n_samples, dtype=np.int64)
        self._importance = np.zeros(self.prep_.n_input_features_)

        Xv_binned = Yv_class = Fv = yv_idx = wv = None
        if eval_set is not None:
            Xv, yv = eval_set
            Xv = (np.asarray(Xv, dtype=object) if cat_features
                  else np.asarray(Xv, dtype=np.float64))
            yv_arr = np.asarray(yv)
            if np.any(~np.isin(yv_arr, self.classes_)):
                raise ValueError("eval_set contains labels not present in training data")
            yv_idx = np.searchsorted(self.classes_, yv_arr)
            Yv_class = _one_hot_class_major(yv_idx, K)
            wv = _validate_sample_weight(
                eval_sample_weight, len(yv_idx), name="eval_sample_weight"
            )
            Xv_binned = self.prep_.transform(Xv)
        _add_timing(timing, "preprocess", phase)

        self.init_ = self.loss_.init_class_major(Y_class, w)  # (K,)
        F = np.tile(self.init_[:, None], (1, n_samples))  # class-major (K, n)
        grad_buffer = np.empty_like(F)
        hess_buffer = np.empty_like(F)
        if Yv_class is not None:
            Fv = np.tile(self.init_[:, None], (1, len(yv_idx)))

        rng = np.random.default_rng(self.random_state)
        self.trees_ = []                           # list of rounds; each = K trees
        self._flat_cache_ = None  # drop any previous fit's flattened ensemble
        self.train_history_, self.valid_history_ = [], []
        eval_train = self.eval_train_loss or bool(self.verbose)
        best_score, best_iter = np.inf, 0
        t0 = time.time()

        for m in range(self.iterations):
            phase = _start_timing(timing)
            if hasattr(self.loss_, "grad_hess_class_major_into"):
                self.loss_.grad_hess_class_major_into(
                    Y_class, F, w, grad_buffer, hess_buffer
                )
                grad, hess = grad_buffer, hess_buffer
            else:
                grad, hess = self.loss_.grad_hess_class_major(Y_class, F)
                if w is not None:
                    grad = grad * w[None, :]
                    hess = hess * w[None, :]
            fmask, findices = self._feature_selection(X_binned.shape[1], rng)
            if self.sampling_ == "goss":
                grad, hess, row_indices_round = self._goss_subsample_multiclass(
                    grad, hess, rng
                )
            elif self.subsample >= 1.0:
                row_indices_round = None
            else:
                mask = rng.random(n_samples) < self.subsample
                row_indices_round = np.flatnonzero(mask).astype(np.int64)
            _add_timing(timing, "grad_hess", phase)
            if use_shared_lightgbm_multiclass:
                phase = _start_timing(timing)
                tree, leaf, leaf_G, leaf_H = build_leafwise_multiclass_tree(
                    X_binned, grad, hess, n_bins, self._max_tree_depth(),
                    self.l2_leaf_reg_, self.lr_,
                    feature_mask=fmask,
                    min_child_weight=self.min_child_weight,
                    hist_buffers=multiclass_hist_buffers,
                    return_training_state=True,
                    X_hist_binned=X_hist_binned,
                    max_leaves=self._max_tree_leaves(),
                    min_child_samples=self.min_child_samples,
                    min_gain_to_split=self.min_gain_to_split,
                )
                _add_timing(timing, "tree_build", phase)
                if tree.depth == 0:
                    if self.verbose:
                        print(f"No further splits at iteration {m}; stopping.")
                    break

                phase = _start_timing(timing)
                self.trees_.append(tree)
                self._accumulate_importance(tree)
                add_multiclass_leaf_values_inplace(leaf, tree.values, F)
                _add_timing(timing, "train_update", phase)

                if eval_train:
                    phase = _start_timing(timing)
                    self.train_history_.append(
                        self.loss_.eval_class_major_labels(y_idx, F, w)
                    )
                    _add_timing(timing, "loss_eval", phase)

                if Fv is not None:
                    phase = _start_timing(timing)
                    tree.add_predict_class_major(Xv_binned, Fv)
                    _add_timing(timing, "validation_predict", phase)
                    phase = _start_timing(timing)
                    val = self.loss_.eval_class_major_labels(yv_idx, Fv, wv)
                    _add_timing(timing, "loss_eval", phase)
                    self.valid_history_.append(val)
                    if val < best_score - 1e-9:
                        best_score, best_iter = val, m
                    elif (self.early_stopping_rounds and
                          m - best_iter >= self.early_stopping_rounds):
                        if self.verbose:
                            print(f"Early stop at {m} (best {best_iter})")
                        self.trees_ = self.trees_[: best_iter + 1]
                        break

                if self.verbose and (m % max(1, self.iterations // 10) == 0):
                    msg = f"[{m}] train {self.train_history_[-1]:.5f}"
                    if Fv is not None:
                        msg += f"  val {self.valid_history_[-1]:.5f}"
                    print(msg)
                continue

            phase = _start_timing(timing)
            fuse_root_this_round = use_fused_root and fmask is None
            if fuse_root_this_round:
                _build_multiclass_histograms_counts_into(
                    X_hist_binned, grad, hess, root_leaf, 1,
                    root_g, root_h, root_c
                )
            _add_timing(timing, "tree_build", phase)

            round_trees = []
            for k in range(K):
                phase = _start_timing(timing)
                if row_indices_round is None or self.sampling_ == "goss":
                    # GOSS gradients are already zeroed and scaled in place.
                    g, h = grad[k], hess[k]
                else:
                    row_mask = np.zeros(n_samples, dtype=bool)
                    row_mask[row_indices_round] = True
                    g = np.where(row_mask, grad[k], 0.0)
                    h = np.where(row_mask, hess[k], 0.0)
                _add_timing(timing, "grad_hess", phase)
                phase = _start_timing(timing)
                builder_kwargs = self._builder_kwargs(
                    fmask, findices, row_indices_round, hist_buffers,
                    split_buffers, X_hist_binned, False,
                    hessian_always_positive=(
                        self.tree_mode_ == "lightgbm"
                        and w is None
                        and row_indices_round is None
                    ),
                    rowpar_buffers=rowpar_buffers
                )
                if fuse_root_this_round:
                    builder_kwargs["root_histograms"] = (
                        root_g[k, :, 0, :], root_h[k, :, 0, :],
                        root_c[:, 0, :],
                    )
                tree, leaf, leaf_G, leaf_H = self._tree_builder()(
                    X_binned, g, h, n_bins, self._max_tree_depth(),
                    self.l2_leaf_reg_, self.lr_,
                    **builder_kwargs,
                )
                _add_timing(timing, "tree_build", phase)
                phase = _start_timing(timing)
                if tree.depth == 0:
                    # A no-split class tree is not a productive weak learner in
                    # the boosting loop. Keep the round shape K-wide, but make
                    # this tree a prediction no-op if another class did split.
                    tree.values[0] = 0.0
                round_trees.append(tree)
                if tree.depth > 0:
                    self._accumulate_importance(tree)
                if self.ordered_boosting_ and tree.depth > 0:
                    ordered_leaf_update_inplace(
                        leaf, leaf_G, leaf_H, g, h, self.lr_,
                        self.l2_leaf_reg_, F[k]
                    )
                elif self.tree_mode_ == "lightgbm" and tree.depth > 0:
                    add_leaf_values_inplace(leaf, tree.values, F[k])
                elif tree.depth > 0:
                    tree.add_predict(X_binned, F[k])
                _add_timing(timing, "train_update", phase)
            # Stop only if EVERY class exhausted its splits this round; if even
            # one class is still learning, the round was productive.
            if all(t.depth == 0 for t in round_trees):
                if self.verbose:
                    print(f"No further splits for any class at iteration {m}; "
                          f"stopping.")
                break
            self.trees_.append(round_trees)
            if eval_train:
                phase = _start_timing(timing)
                self.train_history_.append(
                    self.loss_.eval_class_major_labels(y_idx, F, w)
                )
                _add_timing(timing, "loss_eval", phase)

            if Fv is not None:
                phase = _start_timing(timing)
                for k in range(K):
                    round_trees[k].add_predict(Xv_binned, Fv[k])
                _add_timing(timing, "validation_predict", phase)
                phase = _start_timing(timing)
                val = self.loss_.eval_class_major_labels(yv_idx, Fv, wv)
                _add_timing(timing, "loss_eval", phase)
                self.valid_history_.append(val)
                if val < best_score - 1e-9:
                    best_score, best_iter = val, m
                elif (self.early_stopping_rounds and
                      m - best_iter >= self.early_stopping_rounds):
                    if self.verbose:
                        print(f"Early stop at {m} (best {best_iter})")
                    self.trees_ = self.trees_[: best_iter + 1]
                    break

            if self.verbose and (m % max(1, self.iterations // 10) == 0):
                msg = f"[{m}] train {self.train_history_[-1]:.5f}"
                if Fv is not None:
                    msg += f"  val {self.valid_history_[-1]:.5f}"
                print(msg)

        self.fit_time_ = time.time() - t0
        self.best_iteration_ = len(self.trees_)
        if self.valid_history_:
            self.best_score_ = best_score
        elif Fv is not None:
            self.best_score_ = self.loss_.eval_class_major_labels(yv_idx, Fv, wv)
        elif self.train_history_:
            self.best_score_ = self.train_history_[-1]
        else:
            self.best_score_ = self.loss_.eval_class_major_labels(y_idx, F, w)
        return self

    def _build_flat_ensemble(self):
        return build_flat_multiclass_ensemble(self.trees_, self.n_classes_)

    def predict_raw(self, X):
        """Return the (n_samples, n_classes) matrix of raw per-class scores
        (pre-softmax)."""
        X = (np.asarray(X, dtype=object) if self.prep_.cat_features_
             else np.asarray(X, dtype=np.float64))
        X_binned = self.prep_.transform(X)
        F = np.tile(self.init_[:, None], (1, X_binned.shape[0]))
        flat = self._flat_ensemble()
        if flat is not None and flat_predict_preferred(flat):
            flat.add_predict_class_major(X_binned, F)
        else:
            for round_trees in self.trees_:
                if hasattr(round_trees, "add_predict_class_major"):
                    round_trees.add_predict_class_major(X_binned, F)
                else:
                    for k in range(self.n_classes_):
                        round_trees[k].add_predict(X_binned, F[k])
        return F.T

    def staged_predict_raw(self, X):
        """Yield raw scores after each complete multiclass boosting round."""
        X = (np.asarray(X, dtype=object) if self.prep_.cat_features_
             else np.asarray(X, dtype=np.float64))
        X_binned = self.prep_.transform(X)
        F = np.tile(self.init_[:, None], (1, X_binned.shape[0]))
        for round_trees in self.trees_:
            if hasattr(round_trees, "add_predict_class_major"):
                round_trees.add_predict_class_major(X_binned, F)
            else:
                for k in range(self.n_classes_):
                    round_trees[k].add_predict(X_binned, F[k])
            yield F.T.copy()
