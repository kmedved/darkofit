"""The gradient boosting core: builds the full additive model.

Two boosters share the same machinery (FeaturePreprocessor, oblivious trees):
  * GradientBoosting     -> scalar output (regression, binary classification)
  * MulticlassBoosting   -> K simultaneous outputs (softmax multiclass)
"""

import time
import numpy as np

from .losses import LOSSES, MultiSoftmax
from .preprocessing import FeaturePreprocessor
from .tree import build_oblivious_tree


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


def _auto_learning_rate(n_samples, iterations, early_stopping):
    """Pick a sensible default learning rate.

    Two regimes:
      * No early stopping: `iterations` IS the budget, so scale lr inversely with
        it (fewer trees -> bigger steps) to spend a fixed budget well.
      * Early stopping on: `iterations` is only a high ceiling, NOT the intended
        budget. Deriving lr from the ceiling is a bug -- a 2000-tree safety
        ceiling would force lr=0.01 and never converge before patience runs out.
        In this regime we defer to the ecosystem-standard default of 0.1
        (XGBoost / LightGBM / sklearn HGB all use ~0.1 here) and let early
        stopping decide the tree count. We use the field's well-validated
        constant rather than a value tuned on our own small benchmark suite.

    Oblivious-tree note: because the same feature/threshold is shared across all
    nodes of a level, each oblivious tree carries less information than a full
    CART tree of the same depth. This means the model may genuinely benefit from
    more trees at a smaller step than standard GBDT. If you observe that tree
    count consistently correlates with accuracy on diverse (e.g. OpenML)
    datasets, try --lr 0.05 in the benchmark harness. Promote a lower default
    only after verifying on OpenML, not on the synthetic suite.
    Override with an explicit learning_rate any time.
    """
    if early_stopping:
        return 0.1
    lr = 20.0 / max(iterations, 1)
    return float(np.clip(lr, 0.03, 0.2))


class _BaseBooster:
    def __init__(self, iterations=500, learning_rate=None, depth=6,
                 l2_leaf_reg=3.0, max_bins=128, subsample=1.0,
                 colsample=1.0, cat_smoothing=1.0, early_stopping_rounds=None,
                 min_child_weight=1.0, thread_count=None, random_state=None,
                 verbose=False, ordered_boosting=True):
        self.iterations = int(iterations)
        self.learning_rate = learning_rate
        self.depth = int(depth)
        self.l2_leaf_reg = float(l2_leaf_reg)
        self.max_bins = int(max_bins)
        self.subsample = float(subsample)
        self.colsample = float(colsample)
        self.cat_smoothing = float(cat_smoothing)
        self.early_stopping_rounds = early_stopping_rounds
        self.min_child_weight = float(min_child_weight)
        self.thread_count = thread_count
        self.random_state = random_state
        self.verbose = verbose
        self.ordered_boosting = bool(ordered_boosting)

    def _alloc_hist_buffers(self, n_features, n_bins):
        """Allocate reusable histogram buffers once per fit.

        Shape (n_features, 2**depth, max_bins). Reused for every tree and level
        via _build_histograms_into, which zeroes the active slice each call.
        This avoids reallocating these (potentially large) arrays thousands of
        times over a long boosting run.
        """
        max_leaves = 1 << self.depth
        max_bins = int(n_bins.max()) if len(n_bins) else 1
        hg = np.zeros((n_features, max_leaves, max_bins))
        hh = np.zeros((n_features, max_leaves, max_bins))
        return (hg, hh)

    def _feature_mask(self, n_cols, rng):
        """0/1 mask selecting a random subset of columns for one tree."""
        if self.colsample >= 1.0:
            return None
        k = max(1, int(round(self.colsample * n_cols)))
        mask = np.zeros(n_cols, dtype=np.int64)
        mask[rng.choice(n_cols, size=k, replace=False)] = 1
        return mask

    def _new_preprocessor(self):
        return FeaturePreprocessor(self.max_bins, self.cat_smoothing,
                                   self.random_state)

    def _maybe_subsample(self, grad, hess, rng):
        if self.subsample >= 1.0:
            return grad, hess
        mask = rng.random(grad.shape[0]) < self.subsample
        return np.where(mask, grad, 0.0), np.where(mask, hess, 0.0)

    def _accumulate_importance(self, tree):
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

    def fit(self, X, y, cat_features=None, eval_set=None):
        X = (np.asarray(X, dtype=object) if cat_features
             else np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64)
        n_samples = X.shape[0]

        self.n_threads_ = _apply_thread_count(self.thread_count)
        self.loss_ = LOSSES[self.loss_name](**self.loss_kwargs)
        _es = self.early_stopping_rounds is not None and eval_set is not None
        self.lr_ = (self.learning_rate if self.learning_rate is not None
                    else _auto_learning_rate(n_samples, self.iterations, _es))

        self.prep_ = self._new_preprocessor()
        X_binned = self.prep_.fit_transform(X, [y], cat_features)
        n_bins = self.prep_.n_bins_
        hist_buffers = self._alloc_hist_buffers(X_binned.shape[1], n_bins)
        self._importance = np.zeros(self.prep_.n_input_features_)

        Xv_binned = yv = Fv = None
        if eval_set is not None:
            Xv, yv = eval_set
            Xv = (np.asarray(Xv, dtype=object) if cat_features
                  else np.asarray(Xv, dtype=np.float64))
            yv = np.asarray(yv, dtype=np.float64)
            Xv_binned = self.prep_.transform(Xv)

        self.init_ = self.loss_.init(y)
        F = np.full(n_samples, self.init_, dtype=np.float64)
        if yv is not None:
            Fv = np.full(len(yv), self.init_)

        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        self.train_history_, self.valid_history_ = [], []
        best_score, best_iter = np.inf, 0
        t0 = time.time()

        for m in range(self.iterations):
            grad, hess = self.loss_.grad_hess(y, F)
            g, h = self._maybe_subsample(grad, hess, rng)
            fmask = self._feature_mask(X_binned.shape[1], rng)
            tree = build_oblivious_tree(X_binned, g, h, n_bins, self.depth,
                                        self.l2_leaf_reg, self.lr_,
                                        feature_mask=fmask,
                                        min_child_weight=self.min_child_weight,
                                        hist_buffers=hist_buffers)
            # A depth-0 tree found no legal split: it predicts 0 and adds nothing.
            # Further rounds on the same gradients would do the same, so stop now
            # rather than bank useless trees while patience ticks down.
            if tree.depth == 0:
                if self.verbose:
                    print(f"No further splits at iteration {m}; stopping.")
                break
            if getattr(self.loss_, "adjusts_leaves", False):
                self._correct_leaves(tree, X_binned, y - F)
            self.trees_.append(tree)
            self._accumulate_importance(tree)
            if self.ordered_boosting and not getattr(self.loss_, "adjusts_leaves", False):
                # LOO leaf correction: remove each sample's self-influence from
                # its leaf step.  tree.values keeps normal Newton values for
                # prediction on new data; only the training F uses the corrected
                # update.  For OOB samples (g_i = h_i = 0 from _maybe_subsample)
                # the correction naturally reduces to the normal leaf value.
                leaf = tree.apply(X_binned)
                n_lv = tree.values.shape[0]
                leaf_G = np.bincount(leaf, weights=g, minlength=n_lv)
                leaf_H = np.bincount(leaf, weights=h, minlength=n_lv)
                F += -self.lr_ * (leaf_G[leaf] - g) / (
                    np.maximum(leaf_H[leaf] - h, 0.0) + self.l2_leaf_reg)
            else:
                F += tree.predict(X_binned)
            self.train_history_.append(self.loss_.eval(y, F))

            if Fv is not None:
                Fv += tree.predict(Xv_binned)
                val = self.loss_.eval(yv, Fv)
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
        return self

    def _correct_leaves(self, tree, X_binned, residuals):
        """Override Newton leaf values with the loss-appropriate residual
        statistic (median for MAE, alpha-quantile for Quantile). The tree
        structure was chosen by the gradient; this fixes the step size."""
        leaf = tree.apply(X_binned)
        n_leaves = tree.values.shape[0]
        for l in range(n_leaves):
            r = residuals[leaf == l]
            tree.values[l] = self.lr_ * self.loss_.leaf_value(r)

    def predict_raw(self, X):
        X = (np.asarray(X, dtype=object) if self.prep_.cat_features_
             else np.asarray(X, dtype=np.float64))
        X_binned = self.prep_.transform(X)
        F = np.full(X_binned.shape[0], self.init_, dtype=np.float64)
        for tree in self.trees_:
            F += tree.predict(X_binned)
        return F

    def staged_predict_raw(self, X):
        """Yield the cumulative raw prediction after each tree (1..n_trees)."""
        X = (np.asarray(X, dtype=object) if self.prep_.cat_features_
             else np.asarray(X, dtype=np.float64))
        X_binned = self.prep_.transform(X)
        F = np.full(X_binned.shape[0], self.init_, dtype=np.float64)
        for tree in self.trees_:
            F += tree.predict(X_binned)
            yield F.copy()


class MulticlassBoosting(_BaseBooster):
    """Softmax multiclass booster: fits K trees per round (one per class)."""

    def fit(self, X, y, cat_features=None, eval_set=None):
        X = (np.asarray(X, dtype=object) if cat_features
             else np.asarray(X, dtype=np.float64))
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        K = self.classes_.size
        self.n_classes_ = K
        y_idx = np.searchsorted(self.classes_, y)
        Y = np.eye(K)[y_idx]                      # one-hot (n, K)
        n_samples = X.shape[0]

        self.n_threads_ = _apply_thread_count(self.thread_count)
        self.loss_ = MultiSoftmax(K)
        _es = self.early_stopping_rounds is not None and eval_set is not None
        self.lr_ = (self.learning_rate if self.learning_rate is not None
                    else _auto_learning_rate(n_samples, self.iterations, _es))

        # One ordered-TS target per class (CatBoost-style per-class statistics).
        self.prep_ = self._new_preprocessor()
        X_binned = self.prep_.fit_transform(X, [Y[:, k] for k in range(K)],
                                            cat_features)
        n_bins = self.prep_.n_bins_
        hist_buffers = self._alloc_hist_buffers(X_binned.shape[1], n_bins)
        self._importance = np.zeros(self.prep_.n_input_features_)

        Xv_binned = Yv = Fv = yv_idx = None
        if eval_set is not None:
            Xv, yv = eval_set
            Xv = (np.asarray(Xv, dtype=object) if cat_features
                  else np.asarray(Xv, dtype=np.float64))
            yv_idx = np.searchsorted(self.classes_, np.asarray(yv))
            Yv = np.eye(K)[yv_idx]
            Xv_binned = self.prep_.transform(Xv)

        self.init_ = self.loss_.init(Y)            # (K,)
        F = np.tile(self.init_, (n_samples, 1))    # (n, K)
        if Yv is not None:
            Fv = np.tile(self.init_, (len(yv_idx), 1))

        rng = np.random.default_rng(self.random_state)
        self.trees_ = []                           # list of rounds; each = K trees
        self.train_history_, self.valid_history_ = [], []
        best_score, best_iter = np.inf, 0
        t0 = time.time()

        for m in range(self.iterations):
            grad, hess = self.loss_.grad_hess(Y, F)   # (n, K) each
            fmask = self._feature_mask(X_binned.shape[1], rng)
            round_trees = []
            for k in range(K):
                g, h = self._maybe_subsample(np.ascontiguousarray(grad[:, k]),
                                             np.ascontiguousarray(hess[:, k]), rng)
                tree = build_oblivious_tree(X_binned, g, h, n_bins, self.depth,
                                            self.l2_leaf_reg, self.lr_,
                                            feature_mask=fmask,
                                            min_child_weight=self.min_child_weight,
                                            hist_buffers=hist_buffers)
                round_trees.append(tree)
                self._accumulate_importance(tree)
                if self.ordered_boosting and tree.depth > 0:
                    leaf = tree.apply(X_binned)
                    n_lv = tree.values.shape[0]
                    leaf_G = np.bincount(leaf, weights=g, minlength=n_lv)
                    leaf_H = np.bincount(leaf, weights=h, minlength=n_lv)
                    F[:, k] += -self.lr_ * (leaf_G[leaf] - g) / (
                        np.maximum(leaf_H[leaf] - h, 0.0) + self.l2_leaf_reg)
                else:
                    F[:, k] += tree.predict(X_binned)
            # Stop only if EVERY class exhausted its splits this round; if even
            # one class is still learning, the round was productive.
            if all(t.depth == 0 for t in round_trees):
                if self.verbose:
                    print(f"No further splits for any class at iteration {m}; "
                          f"stopping.")
                break
            self.trees_.append(round_trees)
            self.train_history_.append(self.loss_.eval(Y, F))

            if Fv is not None:
                for k in range(K):
                    Fv[:, k] += round_trees[k].predict(Xv_binned)
                val = self.loss_.eval(Yv, Fv)
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
        return self

    def predict_raw(self, X):
        X = (np.asarray(X, dtype=object) if self.prep_.cat_features_
             else np.asarray(X, dtype=np.float64))
        X_binned = self.prep_.transform(X)
        F = np.tile(self.init_, (X_binned.shape[0], 1))
        for round_trees in self.trees_:
            for k in range(self.n_classes_):
                F[:, k] += round_trees[k].predict(X_binned)
        return F
