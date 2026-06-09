"""Scikit-learn flavored estimators: fit / predict / predict_proba."""

import numpy as np
from .booster import GradientBoosting, MulticlassBoosting
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

# Parameters that exist only on the sklearn wrappers, not on the core boosters.
_SKLEARN_ONLY = frozenset({"early_stopping", "validation_fraction"})


def _should_early_stop(setting):
    """Resolve early_stopping to a bool."""
    return bool(setting)


def _ensure_dense(X):
    """Reject sparse inputs with a clear public API error."""
    if hasattr(X, "tocoo") and hasattr(X, "format"):
        raise ValueError("sparse matrices are not supported; pass a dense array")
    return X


def _ensure_dense_eval_set(eval_set):
    if eval_set is None:
        return None
    Xv, yv = eval_set
    return (_ensure_dense(Xv), yv)


def _make_eval_split(X, y, validation_fraction, random_state,
                     groups=None, stratify=None):
    """Return (train_idx, val_idx) for automatic early-stopping splits.

    Parameters
    ----------
    stratify : array-like or None
        Class labels for stratified splitting (pass for classification tasks).
    groups : array-like or None
        Group membership array (e.g. ``df['subject_id']``).  When supplied,
        groups are kept intact across the split boundary.  For classification,
        ``StratifiedGroupKFold`` is used so class proportions are preserved;
        for regression ``GroupShuffleSplit`` is used.
    """
    from sklearn.model_selection import (
        ShuffleSplit,
        StratifiedShuffleSplit,
        GroupShuffleSplit,
        StratifiedGroupKFold,
    )

    if groups is not None:
        groups = np.asarray(groups)
        if stratify is not None:
            # StratifiedGroupKFold approximates the desired val fraction via
            # n_splits = round(1 / validation_fraction).
            n_splits = max(2, round(1.0 / validation_fraction))
            splitter = StratifiedGroupKFold(n_splits=n_splits)
            train_idx, val_idx = next(
                splitter.split(X, stratify, groups=groups)
            )
        else:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=validation_fraction,
                random_state=random_state,
            )
            train_idx, val_idx = next(splitter.split(X, y, groups=groups))
    elif stratify is not None:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=random_state,
        )
        train_idx, val_idx = next(splitter.split(X, stratify))
    else:
        splitter = ShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=random_state,
        )
        train_idx, val_idx = next(splitter.split(X))

    return train_idx, val_idx

class ChimeraBoostRegressor(BaseEstimator, RegressorMixin):
    """Gradient boosted oblivious trees for regression.

    loss: "RMSE" (default), "MAE", or "Quantile". For "Quantile" pass the level
    via `alpha` (e.g. alpha=0.9 for the 90th-percentile predictor).

    early_stopping : bool, default False
        Whether to use early stopping to terminate training when the validation
        score stops improving.  Requires ``early_stopping_rounds`` (defaults
        to 10 when early stopping is active but the param is None).
    validation_fraction : float, default 0.1
        Fraction of training data to hold out as a validation set when
        *early_stopping* is active and no explicit *eval_set* is passed.
        Ignored when an explicit *eval_set* is given to ``fit``.
    """

    def __init__(self, iterations=500, learning_rate=None, depth=6,
                 l2_leaf_reg=3.0, max_bins=128, subsample=1.0, colsample=1.0,
                 cat_smoothing=1.0, early_stopping_rounds=None,
                 loss="RMSE", alpha=0.5, min_child_weight=1.0,
                 min_child_samples=20, min_gain_to_split=0.0, num_leaves=None,
                 thread_count=None, random_state=None, verbose=False,
                 ordered_boosting="auto",
                 early_stopping=False, validation_fraction=0.1,
                 verbose_timing=False, tree_mode="catboost"):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.max_bins = max_bins
        self.subsample = subsample
        self.colsample = colsample
        self.cat_smoothing = cat_smoothing
        self.early_stopping_rounds = early_stopping_rounds
        self.loss = loss
        self.alpha = alpha
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.min_gain_to_split = min_gain_to_split
        self.num_leaves = num_leaves
        self.thread_count = thread_count
        self.random_state = random_state
        self.verbose = verbose
        self.ordered_boosting = ordered_boosting
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.verbose_timing = verbose_timing
        self.tree_mode = tree_mode

    def fit(self, X, y, cat_features=None, eval_set=None, groups=None,
            sample_weight=None, eval_sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X, y : array-like
            Training data.
        cat_features : list of int or None
            Column indices to treat as categoricals.
        eval_set : (X_val, y_val) tuple or None
            Explicit validation set.  When provided, automatic splitting is
            skipped regardless of the *early_stopping* setting.
        groups : array-like of shape (n_samples,) or None
            Group labels for the samples (e.g. ``df['subject_id']``).  When
            supplied and *early_stopping* triggers an automatic split, groups
            are kept intact across the train/validation boundary using
            ``GroupShuffleSplit``.
        sample_weight : array-like of shape (n_samples,) or None
            Per-sample weights.  Normalized to mean 1 internally.
        eval_sample_weight : array-like of shape (n_validation_samples,) or None
            Validation weights used when evaluating early stopping.
        """
        X = _ensure_dense(X)
        eval_set = _ensure_dense_eval_set(eval_set)
        X = (np.asarray(X, dtype=object) if cat_features
             else np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        es_active = _should_early_stop(self.early_stopping)
        if es_active and eval_set is None:
            train_idx, val_idx = _make_eval_split(
                X, y, self.validation_fraction, self.random_state,
                groups=groups, stratify=None,
            )
            eval_set = (X[val_idx], y[val_idx])
            if sample_weight is not None:
                eval_sample_weight = sample_weight[val_idx]
            X, y = X[train_idx], y[train_idx]
            if sample_weight is not None:
                sample_weight = sample_weight[train_idx]

        # If early stopping is active but patience not explicitly set, use 10.
        es_rounds = self.early_stopping_rounds
        if es_active and es_rounds is None:
            es_rounds = 10

        loss_kwargs = {"alpha": self.alpha} if self.loss == "Quantile" else {}
        kw = {k: v for k, v in self.get_params().items()
              if k not in {"loss", "alpha"} | _SKLEARN_ONLY}
        kw["early_stopping_rounds"] = es_rounds
        self.model_ = GradientBoosting(loss=self.loss, loss_kwargs=loss_kwargs,
                                       **kw)
        self.model_.fit(X, y, cat_features=cat_features, eval_set=eval_set,
                        sample_weight=sample_weight,
                        eval_sample_weight=eval_sample_weight)
        return self

    def predict(self, X):
        X = _ensure_dense(X)
        return self.model_.predict_raw(X)

    def staged_predict(self, X):
        """Yield the prediction after each successive tree."""
        X = _ensure_dense(X)
        yield from self.model_.staged_predict_raw(X)

    @property
    def best_iteration_(self):
        return self.model_.best_iteration_

    @property
    def best_score_(self):
        return self.model_.best_score_

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_

    @property
    def timing_(self):
        return self.model_.timing_


class ChimeraBoostClassifier(BaseEstimator, ClassifierMixin):
    """Gradient boosted oblivious trees for classification.

    Automatically uses binary logloss for 2 classes and softmax multiclass for
    3+. `classes_` preserves the original label values.

    early_stopping : bool, default False
        Whether to use early stopping.  The validation split is always
        stratified to preserve class proportions; when *groups* is passed,
        ``StratifiedGroupKFold`` is used instead.
    validation_fraction : float, default 0.1
        Fraction of training data held out for the automatic validation set.
        Ignored when an explicit *eval_set* is given to ``fit``.
    """

    def __init__(self, iterations=500, learning_rate=None, depth=6,
                 l2_leaf_reg=3.0, max_bins=128, subsample=1.0, colsample=1.0,
                 cat_smoothing=1.0, early_stopping_rounds=None,
                 min_child_weight=1.0, min_child_samples=20,
                 min_gain_to_split=0.0, num_leaves=None, thread_count=None,
                 random_state=None, verbose=False, ordered_boosting="auto",
                 early_stopping=False, validation_fraction=0.1,
                 verbose_timing=False, tree_mode="catboost"):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.max_bins = max_bins
        self.subsample = subsample
        self.colsample = colsample
        self.cat_smoothing = cat_smoothing
        self.early_stopping_rounds = early_stopping_rounds
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.min_gain_to_split = min_gain_to_split
        self.num_leaves = num_leaves
        self.thread_count = thread_count
        self.random_state = random_state
        self.verbose = verbose
        self.ordered_boosting = ordered_boosting
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.verbose_timing = verbose_timing
        self.tree_mode = tree_mode

    def fit(self, X, y, cat_features=None, eval_set=None, groups=None,
            sample_weight=None, eval_sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X, y : array-like
            Training data.
        cat_features : list of int or None
            Column indices to treat as categoricals.
        eval_set : (X_val, y_val) tuple or None
            Explicit validation set with original class labels.  When provided,
            automatic splitting is skipped.
        groups : array-like of shape (n_samples,) or None
            Group labels (e.g. ``df['subject_id']``).  When supplied and early
            stopping triggers an automatic split, ``StratifiedGroupKFold`` keeps
            groups intact and class proportions balanced across the split.
        sample_weight : array-like of shape (n_samples,) or None
            Per-sample weights.  Normalized to mean 1 internally.
        eval_sample_weight : array-like of shape (n_validation_samples,) or None
            Validation weights used when evaluating early stopping.
        """
        X = _ensure_dense(X)
        eval_set = _ensure_dense_eval_set(eval_set)
        X = (np.asarray(X, dtype=object) if cat_features
             else np.asarray(X, dtype=np.float64))
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size
        if self.n_classes_ < 2:
            raise ValueError("Need at least 2 classes.")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        es_active = _should_early_stop(self.early_stopping)
        if es_active and eval_set is None:
            train_idx, val_idx = _make_eval_split(
                X, y, self.validation_fraction, self.random_state,
                groups=groups, stratify=y,  # always stratify for classification
            )
            eval_set = (X[val_idx], y[val_idx])
            if sample_weight is not None:
                eval_sample_weight = sample_weight[val_idx]
            X, y = X[train_idx], y[train_idx]
            if sample_weight is not None:
                sample_weight = sample_weight[train_idx]
            train_classes = np.unique(y)
            if train_classes.size != self.classes_.size:
                raise ValueError("automatic validation split removed a class from training data")

        es_rounds = self.early_stopping_rounds
        if es_active and es_rounds is None:
            es_rounds = 10

        kw = {k: v for k, v in self.get_params().items()
              if k not in _SKLEARN_ONLY}
        kw["early_stopping_rounds"] = es_rounds

        if self.n_classes_ == 2:
            self._multiclass = False
            y01 = (y == self.classes_[1]).astype(np.float64)
            if eval_set is not None:
                Xv, yv = eval_set
                if np.any(~np.isin(np.asarray(yv), self.classes_)):
                    raise ValueError("eval_set contains labels not present in training data")
                eval_set = (Xv, (np.asarray(yv) == self.classes_[1]).astype(np.float64))
            self.model_ = GradientBoosting(loss="Logloss", **kw)
            self.model_.fit(X, y01, cat_features=cat_features, eval_set=eval_set,
                            sample_weight=sample_weight,
                            eval_sample_weight=eval_sample_weight)
        else:
            self._multiclass = True
            self.model_ = MulticlassBoosting(**kw)
            self.model_.fit(X, y, cat_features=cat_features, eval_set=eval_set,
                            sample_weight=sample_weight,
                            eval_sample_weight=eval_sample_weight)
            self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        X = _ensure_dense(X)
        raw = self.model_.predict_raw(X)
        if self._multiclass:
            return self.model_.loss_.transform(raw)            # (n, K)
        p1 = self.model_.loss_.transform(raw)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def staged_predict_proba(self, X):
        """Yield class probabilities after each successive boosting round."""
        X = _ensure_dense(X)
        for raw in self.model_.staged_predict_raw(X):
            if self._multiclass:
                yield self.model_.loss_.transform(raw)
            else:
                p1 = self.model_.loss_.transform(raw)
                yield np.column_stack([1.0 - p1, p1])

    def staged_predict(self, X):
        """Yield class labels after each successive boosting round."""
        for proba in self.staged_predict_proba(X):
            yield self.classes_[np.argmax(proba, axis=1)]

    def staged_predict_raw(self, X):
        """Yield raw margins after each successive boosting round."""
        X = _ensure_dense(X)
        yield from self.model_.staged_predict_raw(X)

    @property
    def best_iteration_(self):
        return self.model_.best_iteration_

    @property
    def best_score_(self):
        return self.model_.best_score_

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_

    @property
    def timing_(self):
        return self.model_.timing_
