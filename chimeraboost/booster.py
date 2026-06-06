"""The gradient boosting core: builds the full additive model.

Two boosters share the same machinery (FeaturePreprocessor, oblivious trees):
  * GradientBoosting     -> scalar output (regression, binary classification)
  * MulticlassBoosting   -> K simultaneous outputs (softmax multiclass)
"""

import time
import numpy as np

from .losses import LOSSES, MultiSoftmax
from .preprocessing import FeaturePreprocessor
from .tree import (build_levelwise_multiclass_tree, build_levelwise_tree,
                   build_oblivious_tree, _loo_leaf_step, _leaf_values,
                   _leaf_values_hs, _linear_predict, _predict_forest,
                   pack_forest, _predict_forest_linear, pack_forest_linear,
                   _shap_forest_linear)


def _factorials(n):
    """Factorials 0!..n! as a float array (Shapley coalition weights)."""
    f = np.empty(n + 1)
    f[0] = 1.0
    for i in range(1, n + 1):
        f[i] = f[i - 1] * i
    return f


# Below this many training rows, per-leaf linear models overfit (noisy small
# data has too little signal per leaf to support a stable slope), so linear
# leaves silently fall back to constant leaves. Matches the codebase's recurring
# "sub-~1k rows is the small-data danger zone" boundary (cf. _auto_min_child_weight,
# the max_bins sub-1k overfit). Validated: protects kc2 (~313 train) from a -4.6%
# Brier loss while keeping the wins on larger sets (sick/spambase/electricity).
LINEAR_LEAVES_MIN_SAMPLES = 1000

# Default number of training rows retained as the SHAP background distribution.
SHAP_BACKGROUND_SIZE = 200

TIMING_KEYS = (
    "preprocess",
    "grad_hess",
    "tree_build",
    "train_update",
    "validation_predict",
    "loss_eval",
)


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


def _auto_learning_rate(n_samples, n_estimators, early_stopping):
    """Default learning rate when the user did not specify one.

    With early stopping, 0.1 (the field-standard default) lets early stopping
    pick the ensemble size; it converges in ~half the trees of a smaller rate
    with no measured accuracy cost, which speeds up both fit and predict.
    Otherwise the rate scales inversely with the iteration budget so short runs
    still cover enough ground.
    """
    if early_stopping:
        return 0.1
    return float(np.clip(20.0 / max(n_estimators, 1), 0.03, 0.2))


_TREE_MODE_ALIASES = {
    "catboost": "catboost",
    "oblivious": "catboost",
    "symmetric": "catboost",
    "lightgbm": "lightgbm",
    "levelwise": "lightgbm",
    "level_wise": "lightgbm",
    "non_oblivious": "lightgbm",
    "non-oblivious": "lightgbm",
}


def _normalize_tree_mode(tree_mode):
    """Return the canonical tree mode name or raise a clear ValueError."""
    if not isinstance(tree_mode, str):
        raise ValueError(
            "tree_mode must be a string alias in {'catboost', 'oblivious', "
            "'symmetric', 'lightgbm', 'levelwise'}; "
            f"got {tree_mode!r}.")
    key = tree_mode.lower()
    try:
        return _TREE_MODE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            "tree_mode must be one of {'catboost', 'oblivious', 'symmetric', "
            "'lightgbm', 'levelwise'}; "
            f"got {tree_mode!r}.") from exc


def _unpack_eval_set(eval_set):
    if eval_set is None:
        return None, None, None
    if len(eval_set) == 2:
        return eval_set[0], eval_set[1], None
    return eval_set[0], eval_set[1], eval_set[2]


def _one_hot_class_major(y_idx, n_classes):
    """Build one-hot targets directly as class-major rows."""
    y_idx = np.asarray(y_idx, dtype=np.int64)
    out = np.zeros((int(n_classes), y_idx.shape[0]), dtype=np.float64)
    out[y_idx, np.arange(y_idx.shape[0])] = 1.0
    return out


class _EarlyStopper:
    """Tracks the best validation score and signals when patience runs out."""

    def __init__(self, patience):
        self.patience = patience
        self.best_score = np.inf
        self.best_iter = 0

    def step(self, score, m):
        """Record the round-*m* score; return True if training should stop."""
        if score < self.best_score - 1e-9:
            self.best_score, self.best_iter = score, m
            return False
        return bool(self.patience) and (m - self.best_iter >= self.patience)


class _BaseBooster:
    """Shared machinery for the scalar and multiclass boosters.

    Holds the common hyperparameters and the helpers both subclasses use:
    histogram-buffer allocation, column subsampling, row subsampling, feature
    preprocessing, and split-gain feature importances. Subclasses implement
    `fit` and `predict_raw`.
    """

    def __init__(self, n_estimators=500, learning_rate=None, depth=6,
                 l2_leaf_reg=1.0, max_bins=128, max_bins_ts=None, subsample=1.0,
                 colsample=1.0, cat_smoothing=1.0, cat_n_permutations=4,
                 early_stopping_rounds=None, min_child_weight=1.0,
                 thread_count=None, random_state=None, verbose=False,
                 ordered_boosting=True, cat_combinations=False,
                 leaf_estimation_iterations=1, hs_lambda=0.0,
                 linear_leaves=False, linear_lambda=1.0,
                 tree_mode="catboost", verbose_timing=False,
                 weighted_target_stats=False):
        self.n_estimators = int(n_estimators)
        self.learning_rate = learning_rate
        self.depth = int(depth)
        self.l2_leaf_reg = float(l2_leaf_reg)
        self.max_bins = int(max_bins)
        self.max_bins_ts = None if max_bins_ts is None else int(max_bins_ts)
        self.subsample = float(subsample)
        self.colsample = float(colsample)
        self.cat_smoothing = float(cat_smoothing)
        self.cat_n_permutations = int(cat_n_permutations)
        self.early_stopping_rounds = early_stopping_rounds
        self.min_child_weight = float(min_child_weight)
        self.thread_count = thread_count
        self.random_state = random_state
        self.verbose = verbose
        self.verbose_timing = bool(verbose_timing)
        self.ordered_boosting = bool(ordered_boosting)
        self.cat_combinations = bool(cat_combinations)
        self.leaf_estimation_iterations = int(leaf_estimation_iterations)
        self.hs_lambda = float(hs_lambda)
        self.linear_leaves = bool(linear_leaves)
        self.linear_lambda = float(linear_lambda)
        self.tree_mode = tree_mode
        self.tree_mode_ = _normalize_tree_mode(tree_mode)
        self.weighted_target_stats = bool(weighted_target_stats)
        self.supports_exact_shap = self.tree_mode_ == "catboost"
        self.supports_linear_leaves = self.tree_mode_ == "catboost"
        if self.tree_mode_ != "catboost" and self.hs_lambda > 0.0:
            raise NotImplementedError(
                "hs_lambda is not supported for tree_mode='lightgbm'.")

    def _reset_timing(self):
        self.timing_ = {key: 0.0 for key in TIMING_KEYS}

    def _tree_builder(self):
        if self.tree_mode_ == "lightgbm":
            return build_levelwise_tree
        return build_oblivious_tree

    def _alloc_hist_buffers(self, n_features, n_bins):
        """Allocate the reusable histogram buffer once per fit.

        Shape (n_features, 2**depth, max_bins, 2); the last axis interleaves
        grad and hess so each scatter write hits one cache line. Reused for
        every tree and level (the kernel zeroes the active slice each call),
        avoiding thousands of reallocations over a long boosting run.
        """
        max_leaves = 1 << self.depth
        max_bins = int(n_bins.max()) if len(n_bins) else 1
        return np.zeros((n_features, max_leaves, max_bins, 2))

    def _build_centers_std(self, Xb, n_bins):
        """Per-(feature, bin) table of STANDARDIZED bin-center values for the
        optional linear-leaf models. Standardizing per feature over the training
        distribution makes the linear ridge penalty scale-fair across features.
        Non-numeric (target-encoded) columns get zeros (never used as linear
        terms); the NaN/missing bin keeps NaN (treated as 0 = mean downstream)."""
        n_features = Xb.shape[0]
        max_bins = int(n_bins.max()) if len(n_bins) else 1
        centers_std = np.zeros((n_features, max_bins))
        is_num = self.prep_.is_numeric_binned_
        bc = self.prep_.binner_.bin_centers_
        for f in range(n_features):
            if not is_num[f]:
                continue
            c = bc[f]
            per_sample = c[Xb[f]]
            finite = per_sample[np.isfinite(per_sample)]
            if finite.size == 0:
                continue
            mu = float(finite.mean())
            sd = float(finite.std())
            if sd <= 0.0:
                sd = 1.0
            centers_std[f, :c.shape[0]] = (c - mu) / sd
        return centers_std

    def _feature_mask(self, n_cols, rng):
        """0/1 mask selecting a random subset of columns for one tree."""
        if self.colsample >= 1.0:
            return None
        k = max(1, int(round(self.colsample * n_cols)))
        mask = np.zeros(n_cols, dtype=np.int64)
        mask[rng.choice(n_cols, size=k, replace=False)] = 1
        return mask

    @staticmethod
    def _feature_indices(feature_mask):
        if feature_mask is None:
            return None
        return np.flatnonzero(feature_mask).astype(np.int64)

    def _new_preprocessor(self):
        """Build a FeaturePreprocessor configured from this booster's params."""
        return FeaturePreprocessor(self.max_bins, self.cat_smoothing,
                                   self.random_state, self.cat_n_permutations,
                                   self.cat_combinations, self.max_bins_ts)

    @staticmethod
    def _normalize_weights(sample_weight, n_samples):
        """Scale weights to mean 1 so the gradient magnitude matches the
        unweighted case. None passes through unchanged."""
        if sample_weight is None:
            return None
        w = np.asarray(sample_weight, dtype=np.float64)
        return w * (n_samples / w.sum())

    def _resolve_lr(self, n_samples, eval_set):
        if self.learning_rate is not None:
            return float(self.learning_rate)
        es = self.early_stopping_rounds is not None and eval_set is not None
        return _auto_learning_rate(n_samples, self.n_estimators, es)

    def _loo_update(self, tree, leaf, g, h):
        """Leave-one-out leaf step: each row's training update uses its leaf's
        grad/hess totals with its own contribution removed, which curbs the
        self-reinforcement of plain boosting. ``tree.values`` keeps the standard
        Newton values for inference; only the training scores use this. Rows
        subsampled out (g=h=0) reduce to the standard leaf value. `leaf` is the
        training assignment returned by build_oblivious_tree."""
        return _loo_leaf_step(leaf, g, h, tree.values.shape[0],
                              self.l2_leaf_reg, self.lr_)

    def _mvs_threshold(self, abs_g, target):
        """MVS: find threshold λ s.t. sum(min(|g_i|/λ, 1)) = target.

        Vectorized: sort once, then find the cutoff k (first row with p<1) via
        a single boolean scan. O(n log n) sort + O(n) NumPy, no Python loop.
        Returns λ=0 to signal "use uniform fallback" (degenerate cases).
        """
        n = len(abs_g)
        if target >= n:
            return 0.0
        sorted_g = np.sort(abs_g)[::-1]  # descending
        total = sorted_g.sum()
        if total < 1e-12:
            return 0.0
        # prefix[k] = sum(sorted_g[:k]); suffix[k] = sum(sorted_g[k:])
        prefix = np.empty(n)
        prefix[0] = 0.0
        prefix[1:] = np.cumsum(sorted_g[:-1])
        suffix = total - prefix
        remaining = target - np.arange(n, dtype=np.float64)
        # Stop at first k where sorted_g[k] * remaining[k] <= suffix[k]
        # (equivalent to sorted_g[k] <= λ_k = suffix[k]/remaining[k])
        cond = (remaining > 0) & (sorted_g * remaining <= suffix)
        if not cond.any():
            return 0.0  # all rows forced
        k = int(np.argmax(cond))
        return suffix[k] / remaining[k]

    def _maybe_subsample(self, grad, hess, rng):
        """MVS (Minimum Variance Sampling): gradient-weighted row subsampling.

        Rows with larger |grad| are sampled with higher probability and
        reweighted by 1/p to keep the leaf gradient sum unbiased. Reduces
        tree-to-tree correlation while concentrating capacity on uncertain
        samples — CatBoost's approach. Falls back to uniform when subsample=1.
        """
        if self.subsample >= 1.0:
            return grad, hess, None
        n = grad.shape[0]
        target = self.subsample * n
        abs_g = np.abs(grad)
        lam = self._mvs_threshold(abs_g, target)
        if lam == 0.0:
            # degenerate or all rows selected: uniform fallback
            mask = rng.random(n) < self.subsample
            row_indices = np.flatnonzero(mask).astype(np.int64)
            return (np.where(mask, grad, 0.0),
                    np.where(mask, hess, 0.0),
                    row_indices)
        prob = np.minimum(abs_g / lam, 1.0)
        mask = rng.random(n) < prob
        # importance weight = 1/p; capped at 1/subsample to avoid blowup on
        # near-zero-gradient rows (whose effective contribution g_i/p_i = λ)
        max_w = 1.0 / max(self.subsample, 1e-3)
        w = np.where(mask, np.minimum(1.0 / np.maximum(prob, 1e-10), max_w), 0.0)
        row_indices = np.flatnonzero(mask).astype(np.int64)
        return grad * w, hess * w, row_indices

    def _accumulate_importance(self, tree):
        """Add this tree's per-split gains to the running importance totals,
        mapped from internal columns back to original input features."""
        for f, g in zip(tree.splits_feat, tree.gains):
            orig = self.prep_.feature_map_[f]
            self._importance[orig] += g

    @property
    def feature_importances_(self):
        """Total split gain per ORIGINAL input column, normalized to sum 1."""
        imp = self._importance.copy()
        s = imp.sum()
        return imp / s if s > 0 else imp


class GradientBoosting(_BaseBooster):
    """Scalar booster: regression and binary classification."""

    def __init__(self, loss="RMSE", loss_kwargs=None, **kw):
        super().__init__(**kw)
        self.loss_name = loss
        self.loss_kwargs = loss_kwargs or {}

    def fit(self, X, y, cat_features=None, eval_set=None, sample_weight=None):
        """Fit the additive model. Optionally pass `cat_features` (column indices
        to target-encode) and `eval_set=(X_val, y_val[, sample_weight_val])`
        for early stopping.
        `sample_weight` is a 1-D array of per-sample weights; None means uniform.
        Weights are normalized to mean 1 internally so the gradient scale stays
        comparable to the no-weight case."""
        X = (np.asarray(X, dtype=object) if cat_features
             else np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64)
        n_samples = X.shape[0]
        w = self._normalize_weights(sample_weight, n_samples)
        self.train_weight_sum_ = None if w is None else float(w.sum())

        self.n_threads_ = _apply_thread_count(self.thread_count)
        self.loss_ = LOSSES[self.loss_name](**self.loss_kwargs)
        self.lr_ = self._resolve_lr(n_samples, eval_set)
        self._reset_timing()
        timing_enabled = self.verbose_timing
        tree_builder = self._tree_builder()
        if self.tree_mode_ != "catboost" and self.linear_leaves:
            raise NotImplementedError(
                "linear_leaves is not supported for tree_mode='lightgbm'.")

        if timing_enabled:
            t_phase = time.perf_counter()
        self.prep_ = self._new_preprocessor()
        # Tree kernels consume a feature-major matrix; transpose once here.
        ts_weight = w if self.weighted_target_stats else None
        Xb = np.ascontiguousarray(
            self.prep_.fit_transform(
                X, [y], cat_features, sample_weight=ts_weight).T)
        n_bins = self.prep_.n_bins_
        hist_buffers = self._alloc_hist_buffers(Xb.shape[0], n_bins)
        self._importance = np.zeros(self.prep_.n_input_features_)
        # Keep a small sample of the (binned) training rows as the default SHAP
        # background -- the reference distribution interventional TreeSHAP
        # integrates over. Capped so it never bloats the pickled model.
        bg_n = min(n_samples, SHAP_BACKGROUND_SIZE)
        bg_idx = np.random.default_rng(self.random_state).choice(
            n_samples, bg_n, replace=False)
        self._shap_background_ = np.ascontiguousarray(Xb[:, bg_idx])

        Xvb = yv = Fv = wv = None
        if eval_set is not None:
            Xv, yv, wv = _unpack_eval_set(eval_set)
            Xv = (np.asarray(Xv, dtype=object) if cat_features
                  else np.asarray(Xv, dtype=np.float64))
            yv = np.asarray(yv, dtype=np.float64)
            wv = None if wv is None else np.asarray(wv, dtype=np.float64)
            Xvb = np.ascontiguousarray(self.prep_.transform(Xv).T)
        self.validation_weight_sum_ = None if wv is None else float(wv.sum())
        if timing_enabled:
            self.timing_["preprocess"] += time.perf_counter() - t_phase

        self.init_ = self.loss_.init(y, w)
        F = np.full(n_samples, self.init_, dtype=np.float64)
        if yv is not None:
            Fv = np.full(len(yv), self.init_)

        adjusts_leaves = getattr(self.loss_, "adjusts_leaves", False)
        use_constant_hessian = (
            getattr(self.loss_, "constant_hessian", False)
            and w is None
            and self.subsample >= 1.0
        )
        # Linear leaves are incompatible with the median/quantile leaf override
        # (adjusts_leaves) and with ordered boosting (a linear LOO step is not
        # implemented); they take over the leaf value otherwise. Below
        # LINEAR_LEAVES_MIN_SAMPLES rows they overfit, so fall back to constant.
        ll_active = (self.linear_leaves and self.supports_linear_leaves
                     and not adjusts_leaves
                     and n_samples >= LINEAR_LEAVES_MIN_SAMPLES)
        self._centers_std_ = (self._build_centers_std(Xb, n_bins)
                              if ll_active else None)
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        self._forest_ = None   # packed-forest cache; built lazily on predict
        self.train_history_, self.valid_history_ = [], []
        stopper = _EarlyStopper(self.early_stopping_rounds)
        t0 = time.time()

        for m in range(self.n_estimators):
            if timing_enabled:
                t_phase = time.perf_counter()
            grad, hess = self.loss_.grad_hess(y, F)
            if w is not None:
                grad, hess = grad * w, hess * w
            if self.subsample >= 1.0:
                g, h, row_indices = grad, hess, None
            else:
                g, h, row_indices = self._maybe_subsample(grad, hess, rng)
            if self.colsample >= 1.0:
                fmask = findices = None
            else:
                fmask = self._feature_mask(Xb.shape[0], rng)
                findices = self._feature_indices(fmask)
            if timing_enabled:
                self.timing_["grad_hess"] += time.perf_counter() - t_phase
                t_phase = time.perf_counter()
            tree, leaf = tree_builder(
                Xb, g, h, n_bins, self.depth, self.l2_leaf_reg, self.lr_,
                feature_mask=fmask, min_child_weight=self.min_child_weight,
                hist_buffers=hist_buffers, hs_lambda=self.hs_lambda,
                linear_leaves=ll_active, centers_std=self._centers_std_,
                is_numeric=self.prep_.is_numeric_binned_,
                linear_lambda=self.linear_lambda,
                constant_hessian=use_constant_hessian,
                feature_indices=findices, row_indices=row_indices)
            if timing_enabled:
                self.timing_["tree_build"] += time.perf_counter() - t_phase
            # A depth-0 tree found no legal split; the next round on the same
            # gradients would too, so stop rather than bank empty trees.
            if tree.depth == 0:
                break
            if timing_enabled:
                t_phase = time.perf_counter()
            if adjusts_leaves:
                self._correct_leaves(tree, leaf, y - F, w)
            self.trees_.append(tree)
            self._accumulate_importance(tree)
            # Ordered boosting and leaf adjustment are mutually exclusive: the
            # former rewrites the training step, the latter the leaf value.
            if ll_active and tree.lin_coef is not None:
                # Linear-leaf path: training update is the leaf's local linear
                # model (no ordered boosting / no leaf_estimation refinement).
                F += _linear_predict(leaf, tree.lin_feats, tree.lin_coef,
                                     self._centers_std_, Xb)
            elif self.ordered_boosting and not adjusts_leaves:
                F += self._loo_update(tree, leaf, g, h)
            else:
                # Additional Newton steps refine the leaf values using the updated
                # residuals after each step. For constant-hessian losses (RMSE)
                # this converges in a few steps; for Logloss it reaches a better
                # per-leaf approximation than the single first-order step.
                for _ in range(self.leaf_estimation_iterations - 1):
                    F_tmp = F + tree.values[leaf]
                    g2, h2 = self.loss_.grad_hess(y, F_tmp)
                    if w is not None:
                        g2, h2 = g2 * w, h2 * w
                    n_lv = tree.values.shape[0]
                    if self.hs_lambda > 0.0:
                        tree.values += _leaf_values_hs(leaf, g2, h2, n_lv,
                                                       self.l2_leaf_reg, self.lr_,
                                                       self.hs_lambda)
                    else:
                        tree.values += _leaf_values(leaf, g2, h2, n_lv,
                                                    self.l2_leaf_reg, self.lr_)
                F += tree.values[leaf]
            if timing_enabled:
                self.timing_["train_update"] += time.perf_counter() - t_phase
            if self.verbose:
                if timing_enabled:
                    t_phase = time.perf_counter()
                self.train_history_.append(self.loss_.eval(y, F, w))
                if timing_enabled:
                    self.timing_["loss_eval"] += time.perf_counter() - t_phase

            if Fv is not None:
                if timing_enabled:
                    t_phase = time.perf_counter()
                Fv += tree.predict(Xvb)
                if timing_enabled:
                    self.timing_["validation_predict"] += time.perf_counter() - t_phase
                    t_phase = time.perf_counter()
                val = self.loss_.eval(yv, Fv, wv)
                self.valid_history_.append(val)
                if timing_enabled:
                    self.timing_["loss_eval"] += time.perf_counter() - t_phase
                if stopper.step(val, m):
                    if self.verbose:
                        print(f"Early stop at {m} (best {stopper.best_iter})")
                    self.trees_ = self.trees_[: stopper.best_iter + 1]
                    break

            if self.verbose and (m % max(1, self.n_estimators // 10) == 0):
                msg = f"[{m}] train {self.train_history_[-1]:.5f}"
                if Fv is not None:
                    msg += f"  val {self.valid_history_[-1]:.5f}"
                print(msg)

        self.fit_time_ = time.time() - t0
        self.best_iteration_ = len(self.trees_)
        return self

    def _correct_leaves(self, tree, leaf, residuals, sample_weight=None):
        """Override Newton leaf values with the loss-appropriate residual
        statistic (median for MAE, alpha-quantile for Quantile). The tree
        structure was chosen by the gradient; this fixes the step size.
        `leaf` is the training assignment from build_oblivious_tree."""
        n_leaves = tree.values.shape[0]
        if sample_weight is None:
            for l in range(n_leaves):
                mask = leaf == l
                r = residuals[mask]
                tree.values[l] = self.lr_ * self.loss_.leaf_value(r, None)
            return

        order = np.argsort(leaf)
        leaf_sorted = leaf[order]
        residuals_sorted = residuals[order]
        weights_sorted = sample_weight[order]
        counts = np.bincount(leaf_sorted, minlength=n_leaves)
        start = 0
        for l in range(n_leaves):
            end = start + counts[l]
            r = residuals_sorted[start:end]
            w = weights_sorted[start:end]
            tree.values[l] = self.lr_ * self.loss_.leaf_value(r, w)
            start = end

    def predict_raw(self, X):
        """Return raw additive scores (pre-link): the regression prediction, or
        the log-odds for binary classification."""
        X = (np.asarray(X, dtype=object) if self.prep_.cat_features_
             else np.asarray(X, dtype=np.float64))
        Xb = np.ascontiguousarray(self.prep_.transform(X).T)
        if not self.trees_:
            return np.full(Xb.shape[1], self.init_, dtype=np.float64)
        if self.tree_mode_ != "catboost":
            F = np.full(Xb.shape[1], self.init_, dtype=np.float64)
            for tree in self.trees_:
                F += tree.predict(Xb)
            return F
        if getattr(self, "_centers_std_", None) is not None:
            # Linear-leaf path: a dedicated fused kernel walks the whole forest
            # in one parallel pass (constant trees ride along as k=0).
            if self._forest_ is None:
                self._forest_ = pack_forest_linear(self.trees_, self.depth)
            feats, thrs, depths, lin_k, foff, lidx, coff, coef = self._forest_
            return _predict_forest_linear(Xb, feats, thrs, depths, lin_k, foff,
                                          lidx, coff, coef, self._centers_std_,
                                          self.init_)
        if self._forest_ is None:
            self._forest_ = pack_forest(self.trees_, self.depth)
        feats, thrs, depths, vals, voff = self._forest_
        return _predict_forest(Xb, feats, thrs, depths, vals, voff, self.init_)

    def staged_predict_raw(self, X):
        """Yield the cumulative raw prediction after each tree (1..n_trees)."""
        X = (np.asarray(X, dtype=object) if self.prep_.cat_features_
             else np.asarray(X, dtype=np.float64))
        Xb = np.ascontiguousarray(self.prep_.transform(X).T)
        F = np.full(Xb.shape[1], self.init_, dtype=np.float64)
        for tree in self.trees_:
            F += tree.predict(Xb)
            yield F.copy()

    def shap_values(self, X, background=None, max_background=SHAP_BACKGROUND_SIZE,
                    random_state=0):
        """Exact interventional TreeSHAP, in raw-score (margin) space.

        Returns ``(phi, expected_value)`` where ``phi`` has shape
        ``(n_samples, n_input_features)`` and, for every row,
        ``phi.sum(axis=1) + expected_value == predict_raw(X)`` to floating-point
        tolerance (Shapley efficiency). Each ``phi[i, f]`` is feature f's signed
        additive contribution to the raw score of row i -- the regression target,
        or the binary log-odds. Linear-leaf slope terms are included exactly.

        ``background`` is the reference distribution SHAP integrates over
        (defaults to the training-data sample captured at fit); ``max_background``
        subsamples it for speed (cost is linear in the background size)."""
        if not self.supports_exact_shap:
            raise NotImplementedError(
                "shap_values is not supported for tree_mode='lightgbm'.")
        X = (np.asarray(X, dtype=object) if self.prep_.cat_features_
             else np.asarray(X, dtype=np.float64))
        Xb = np.ascontiguousarray(self.prep_.transform(X).T)
        n_orig = self.prep_.n_input_features_
        if not self.trees_:
            return np.zeros((Xb.shape[1], n_orig)), float(self.init_)
        if background is None:
            Rb = self._shap_background_
        else:
            bg = (np.asarray(background, dtype=object) if self.prep_.cat_features_
                  else np.asarray(background, dtype=np.float64))
            Rb = np.ascontiguousarray(self.prep_.transform(bg).T)
        if Rb.shape[1] > max_background:
            sel = np.random.default_rng(random_state).choice(
                Rb.shape[1], max_background, replace=False)
            Rb = np.ascontiguousarray(Rb[:, sel])
        feats, thrs, depths, lin_k, foff, lidx, coff, coef = \
            pack_forest_linear(self.trees_, self.depth)
        cs = getattr(self, "_centers_std_", None)
        if cs is None:
            cs = np.zeros((1, 1))   # unused: every tree is constant (k=0)
        fact = _factorials(self.depth)
        phi = _shap_forest_linear(Xb, Rb, feats, thrs, depths, lin_k, foff, lidx,
                                  coff, coef, cs, self.prep_.feature_map_, n_orig,
                                  fact)
        base = _predict_forest_linear(Rb, feats, thrs, depths, lin_k, foff, lidx,
                                      coff, coef, cs, self.init_)
        return phi, float(base.mean())


class MulticlassBoosting(_BaseBooster):
    """Softmax multiclass booster: fits K trees per round (one per class)."""

    def fit(self, X, y, cat_features=None, eval_set=None, sample_weight=None):
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
        Y_class = _one_hot_class_major(y_idx, K)  # one-hot (K, n)
        n_samples = X.shape[0]
        w = self._normalize_weights(sample_weight, n_samples)
        self.train_weight_sum_ = None if w is None else float(w.sum())

        self.n_threads_ = _apply_thread_count(self.thread_count)
        self.loss_ = MultiSoftmax(K)
        self.lr_ = self._resolve_lr(n_samples, eval_set)
        self._reset_timing()
        timing_enabled = self.verbose_timing
        tree_builder = self._tree_builder()

        # One ordered-TS target per class (CatBoost-style per-class statistics).
        if timing_enabled:
            t_phase = time.perf_counter()
        self.prep_ = self._new_preprocessor()
        Xb = np.ascontiguousarray(
            self.prep_.fit_transform(X, [Y_class[k] for k in range(K)],
                                     cat_features,
                                     sample_weight=(w if self.weighted_target_stats
                                                    else None)).T)
        n_bins = self.prep_.n_bins_
        hist_buffers = self._alloc_hist_buffers(Xb.shape[0], n_bins)
        self._importance = np.zeros(self.prep_.n_input_features_)

        Xvb = Yv_class = Fv = yv_idx = wv = None
        if eval_set is not None:
            Xv, yv, wv = _unpack_eval_set(eval_set)
            Xv = (np.asarray(Xv, dtype=object) if cat_features
                  else np.asarray(Xv, dtype=np.float64))
            yv_idx = np.searchsorted(self.classes_, np.asarray(yv))
            Yv_class = _one_hot_class_major(yv_idx, K)
            wv = None if wv is None else np.asarray(wv, dtype=np.float64)
            Xvb = np.ascontiguousarray(self.prep_.transform(Xv).T)
        self.validation_weight_sum_ = None if wv is None else float(wv.sum())
        if timing_enabled:
            self.timing_["preprocess"] += time.perf_counter() - t_phase

        self.init_ = self.loss_.init_class_major(Y_class, w)  # (K,)
        F = np.tile(self.init_[:, None], (1, n_samples))      # (K, n)
        if Yv_class is not None:
            Fv = np.tile(self.init_[:, None], (1, len(yv_idx)))

        coupling = (K - 1) / K   # softmax Hessian has rank K-1, not K
        rng = np.random.default_rng(self.random_state)
        use_shared_levelwise = (
            self.tree_mode_ == "lightgbm"
            and self.subsample >= 1.0
            and not self.ordered_boosting
        )
        self._multiclass_shared_trees_ = use_shared_levelwise
        if use_shared_levelwise:
            max_bins = int(n_bins.max()) if len(n_bins) else 1
            hist_buffers_multi = np.zeros(
                (Xb.shape[0], 1 << self.depth, max_bins, K, 2))
        else:
            hist_buffers_multi = None
        self.trees_ = []                           # rounds: K trees or one vector tree
        self._forests_ = None   # per-class packed-forest cache (lazy on predict)
        self.train_history_, self.valid_history_ = [], []
        stopper = _EarlyStopper(self.early_stopping_rounds)
        t0 = time.time()

        for m in range(self.n_estimators):
            if timing_enabled:
                t_phase = time.perf_counter()
            grad, hess = self.loss_.grad_hess_class_major(Y_class, F)
            if w is not None:
                grad, hess = grad * w[None, :], hess * w[None, :]
            if self.colsample >= 1.0:
                fmask = findices = None
            else:
                fmask = self._feature_mask(Xb.shape[0], rng)
                findices = self._feature_indices(fmask)
            if timing_enabled:
                self.timing_["grad_hess"] += time.perf_counter() - t_phase
            if use_shared_levelwise:
                if timing_enabled:
                    t_phase = time.perf_counter()
                tree, leaf = build_levelwise_multiclass_tree(
                    Xb, grad, hess * coupling, n_bins, self.depth,
                    self.l2_leaf_reg, self.lr_,
                    feature_mask=fmask,
                    min_child_weight=self.min_child_weight,
                    hist_buffers=hist_buffers_multi)
                if timing_enabled:
                    self.timing_["tree_build"] += time.perf_counter() - t_phase
                if tree.depth == 0:
                    break
                if timing_enabled:
                    t_phase = time.perf_counter()
                self.trees_.append(tree)
                self._accumulate_importance(tree)
                F += tree.values[leaf].T
                if timing_enabled:
                    self.timing_["train_update"] += time.perf_counter() - t_phase

                if self.verbose:
                    if timing_enabled:
                        t_phase = time.perf_counter()
                    self.train_history_.append(
                        self.loss_.eval_class_major(Y_class, F, w))
                    if timing_enabled:
                        self.timing_["loss_eval"] += time.perf_counter() - t_phase

                if Fv is not None:
                    if timing_enabled:
                        t_phase = time.perf_counter()
                    Fv += tree.predict(Xvb).T
                    if timing_enabled:
                        self.timing_["validation_predict"] += time.perf_counter() - t_phase
                        t_phase = time.perf_counter()
                    val = self.loss_.eval_class_major(Yv_class, Fv, wv)
                    self.valid_history_.append(val)
                    if timing_enabled:
                        self.timing_["loss_eval"] += time.perf_counter() - t_phase
                    if stopper.step(val, m):
                        if self.verbose:
                            print(f"Early stop at {m} (best {stopper.best_iter})")
                        self.trees_ = self.trees_[: stopper.best_iter + 1]
                        break

                if self.verbose and (m % max(1, self.n_estimators // 10) == 0):
                    msg = f"[{m}] train {self.train_history_[-1]:.5f}"
                    if Fv is not None:
                        msg += f"  val {self.valid_history_[-1]:.5f}"
                    print(msg)
                continue

            round_trees = []
            for k in range(K):
                if timing_enabled:
                    t_phase = time.perf_counter()
                hk = hess[k] * coupling
                if self.subsample >= 1.0:
                    g, h, row_indices = grad[k], hk, None
                else:
                    g, h, row_indices = self._maybe_subsample(
                        grad[k], hk, rng)
                tree, leaf = tree_builder(
                    Xb, g, h, n_bins, self.depth, self.l2_leaf_reg, self.lr_,
                    feature_mask=fmask,
                    min_child_weight=self.min_child_weight,
                    hist_buffers=hist_buffers,
                    hs_lambda=self.hs_lambda,
                    feature_indices=findices,
                    row_indices=row_indices)
                if timing_enabled:
                    self.timing_["tree_build"] += time.perf_counter() - t_phase
                    t_phase = time.perf_counter()
                round_trees.append(tree)
                self._accumulate_importance(tree)
                if self.ordered_boosting and tree.depth > 0:
                    F[k] += self._loo_update(tree, leaf, g, h)
                elif tree.depth > 0:
                    F[k] += tree.values[leaf]
                # depth-0 trees contribute nothing (predict would be zeros).
                if timing_enabled:
                    self.timing_["train_update"] += time.perf_counter() - t_phase
            # Stop only once EVERY class has exhausted its splits; a single class
            # still learning makes the round productive.
            if all(t.depth == 0 for t in round_trees):
                break
            self.trees_.append(round_trees)
            if self.verbose:
                if timing_enabled:
                    t_phase = time.perf_counter()
                self.train_history_.append(
                    self.loss_.eval_class_major(Y_class, F, w))
                if timing_enabled:
                    self.timing_["loss_eval"] += time.perf_counter() - t_phase

            if Fv is not None:
                if timing_enabled:
                    t_phase = time.perf_counter()
                for k in range(K):
                    Fv[k] += round_trees[k].predict(Xvb)
                if timing_enabled:
                    self.timing_["validation_predict"] += time.perf_counter() - t_phase
                    t_phase = time.perf_counter()
                val = self.loss_.eval_class_major(Yv_class, Fv, wv)
                self.valid_history_.append(val)
                if timing_enabled:
                    self.timing_["loss_eval"] += time.perf_counter() - t_phase
                if stopper.step(val, m):
                    if self.verbose:
                        print(f"Early stop at {m} (best {stopper.best_iter})")
                    self.trees_ = self.trees_[: stopper.best_iter + 1]
                    break

            if self.verbose and (m % max(1, self.n_estimators // 10) == 0):
                msg = f"[{m}] train {self.train_history_[-1]:.5f}"
                if Fv is not None:
                    msg += f"  val {self.valid_history_[-1]:.5f}"
                print(msg)

        self.fit_time_ = time.time() - t0
        self.best_iteration_ = len(self.trees_)
        return self

    def predict_raw(self, X):
        """Return the (n_samples, n_classes) matrix of raw per-class scores
        (pre-softmax)."""
        X = (np.asarray(X, dtype=object) if self.prep_.cat_features_
             else np.asarray(X, dtype=np.float64))
        Xb = np.ascontiguousarray(self.prep_.transform(X).T)
        F = np.tile(self.init_, (Xb.shape[1], 1))
        if not self.trees_:
            return F
        if getattr(self, "_multiclass_shared_trees_", False):
            for tree in self.trees_:
                F += tree.predict(Xb)
            return F
        if self.tree_mode_ != "catboost":
            for round_trees in self.trees_:
                for k in range(self.n_classes_):
                    F[:, k] += round_trees[k].predict(Xb)
            return F
        if self._forests_ is None:
            # One packed forest per class: class k's trees are round_trees[k]
            # across every round.
            self._forests_ = [
                pack_forest([rt[k] for rt in self.trees_], self.depth)
                for k in range(self.n_classes_)]
        for k in range(self.n_classes_):
            feats, thrs, depths, vals, voff = self._forests_[k]
            F[:, k] = _predict_forest(Xb, feats, thrs, depths, vals, voff,
                                      self.init_[k])
        return F
