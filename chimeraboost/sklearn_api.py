"""Scikit-learn flavored estimators: fit / predict / predict_proba."""

import numpy as np
from ._validation import n_features_from_array_like, normalize_cat_features
from .booster import GradientBoosting, MulticlassBoosting
from .auto_params import (
    effective_sample_size,
    is_auto_learning_rate,
    resolve_learning_rate_details,
)
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

# Parameters that exist only on the sklearn wrappers, not on the core boosters.
_SKLEARN_ONLY = frozenset({
    "early_stopping", "validation_fraction", "validation_strategy", "refit",
    "refit_strategy", "auto_learning_rate_probe",
    "auto_learning_rate_probe_values", "auto_learning_rate_probe_iterations",
})

_REFIT_STRATEGY_EXPONENT = {
    "best": 0.0,
    "exact": 0.0,
    "sqrt": 0.5,
    "linear": 1.0,
    "scaled": 1.0,
}


def _should_early_stop(setting):
    """Resolve early_stopping to a bool."""
    if not isinstance(setting, (bool, np.bool_)):
        raise ValueError("early_stopping must be a bool")
    return bool(setting)


def _normalize_validation_strategy(strategy):
    mode = str(strategy).lower().replace("-", "_")
    if mode in {"random", "weighted_stratified"}:
        return mode
    raise ValueError(
        "validation_strategy must be 'random' or 'weighted_stratified'"
    )


def _resolve_validation_fraction(validation_fraction, sample_weight, n_samples):
    if validation_fraction == "auto":
        n_eff = effective_sample_size(sample_weight, n_samples)
        return float(np.clip(max(0.10, 200.0 / max(n_eff, 1.0)), 0.10, 0.25))
    fraction = float(validation_fraction)
    if not (0.0 < fraction < 1.0):
        raise ValueError("validation_fraction must be in (0, 1) or 'auto'")
    return fraction


def _validate_wrapper_sample_weight(sample_weight, n_samples, name="sample_weight"):
    if sample_weight is None:
        return None
    w = np.asarray(sample_weight, dtype=np.float64)
    if w.shape != (n_samples,):
        raise ValueError(f"{name} must have shape ({n_samples},)")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(w < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    if float(np.sum(w)) <= 0.0:
        raise ValueError(f"{name} must have positive total weight")
    return w


def _weighted_quantile(values, sample_weight, qs):
    values = np.asarray(values, dtype=np.float64)
    if sample_weight is None:
        return np.quantile(values, qs)
    original_values = values
    w = np.asarray(sample_weight, dtype=np.float64)
    order = np.argsort(values)
    values = values[order]
    w = w[order]
    positive = w > 0.0
    values = values[positive]
    w = w[positive]
    if values.size == 0:
        return np.quantile(original_values, qs)
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return np.interp(qs, cw, values)


def _regression_validation_strata(
    y, sample_weight=None, validation_fraction=0.1, max_bins=10
):
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[0]
    if n < 4:
        return None
    n_val = int(np.ceil(float(validation_fraction) * n))
    n_train = n - n_val
    max_feasible_strata = min(int(max_bins), n_val, n_train)
    if max_feasible_strata < 2:
        return None
    n_bins = int(min(max_feasible_strata, max(2, n // 4)))
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    if qs.size == 0:
        return None
    try:
        edges = np.unique(_weighted_quantile(y, sample_weight, qs))
    except ValueError:
        return None
    if edges.size == 0:
        return None
    strata = np.searchsorted(edges, y, side="right")
    _, counts = np.unique(strata, return_counts=True)
    if np.min(counts) < 2:
        return None
    return strata


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


def _feature_names_from_input(X):
    columns = getattr(X, "columns", None)
    if columns is None:
        return None
    names = np.asarray(columns, dtype=object)
    if names.ndim == 1 and all(isinstance(name, str) for name in names):
        return names
    return None


def _coerce_fit_X(X, cat_features):
    X = _ensure_dense(X)
    n_features = n_features_from_array_like(X)
    cat_features = normalize_cat_features(cat_features, n_features)
    X_arr = (np.asarray(X, dtype=object) if cat_features
             else np.asarray(X, dtype=np.float64))
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    return X_arr, cat_features, n_features


def _validate_eval_set_features(eval_set, n_features):
    if eval_set is None:
        return None
    Xv, yv = eval_set
    actual = n_features_from_array_like(Xv, name="eval_set[0]")
    if actual != int(n_features):
        raise ValueError(
            f"eval_set[0] has {actual} features, but X has "
            f"{int(n_features)} features"
        )
    return eval_set


def _infer_model_n_features(model):
    prep = getattr(model, "prep_", None)
    if prep is None:
        return None
    n_features = getattr(prep, "n_input_features_", None)
    return None if n_features is None else int(n_features)


def _check_predict_input(estimator, X):
    check_is_fitted(estimator, "model_")
    X = _ensure_dense(X)
    actual = n_features_from_array_like(X)
    expected = getattr(estimator, "n_features_in_", None)
    if expected is None:
        expected = _infer_model_n_features(estimator.model_)
        if expected is not None:
            estimator.n_features_in_ = int(expected)
    if expected is not None and actual != int(expected):
        raise ValueError(
            f"X has {actual} features, but {type(estimator).__name__} "
            f"is expecting {int(expected)} features as input"
        )
    return X


def _make_eval_split(X, y, validation_fraction, random_state,
                     groups=None, stratify=None, sample_weight=None,
                     validation_strategy="random"):
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

    validation_strategy = _normalize_validation_strategy(validation_strategy)
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
            realized_policy = "class_stratified_group"
        else:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=validation_fraction,
                random_state=random_state,
            )
            train_idx, val_idx = next(splitter.split(X, y, groups=groups))
            realized_policy = "group_shuffle"
    elif stratify is not None:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=random_state,
        )
        train_idx, val_idx = next(splitter.split(X, stratify))
        realized_policy = "class_stratified"
    elif validation_strategy == "weighted_stratified":
        regression_strata = _regression_validation_strata(
            y, sample_weight, validation_fraction
        )
        if regression_strata is not None:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=validation_fraction,
                random_state=random_state,
            )
            train_idx, val_idx = next(splitter.split(X, regression_strata))
            realized_policy = "weighted_target_stratified"
        else:
            splitter = ShuffleSplit(
                n_splits=1,
                test_size=validation_fraction,
                random_state=random_state,
            )
            train_idx, val_idx = next(splitter.split(X))
            realized_policy = "random_fallback"
    else:
        splitter = ShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=random_state,
        )
        train_idx, val_idx = next(splitter.split(X))
        realized_policy = "random"

    return train_idx, val_idx, realized_policy


class _RefitParamsMixin:
    """Shared fitted-model metadata and full-data refit helpers."""

    def _clear_refit_selection_metadata(self):
        for name in (
            "_selection_n_total_", "_selection_n_train_",
            "_best_n_estimators_", "_best_score_", "_learning_rate_",
            "selection_model_", "refit_", "refit_n_estimators_",
            "refit_strategy_",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def _record_input_feature_metadata(self, X, n_features):
        self.n_features_in_ = int(n_features)
        feature_names = _feature_names_from_input(X)
        if feature_names is not None:
            self.feature_names_in_ = feature_names
        elif hasattr(self, "feature_names_in_"):
            delattr(self, "feature_names_in_")

    def _restore_n_features_from_model(self):
        n_features = _infer_model_n_features(getattr(self, "model_", None))
        if n_features is not None:
            self.n_features_in_ = int(n_features)

    def _record_refit_selection_metadata(self, n_total, train_idx):
        self._selection_n_total_ = int(n_total)
        self._selection_n_train_ = int(len(train_idx))

    def _record_selection_result(self, model):
        self._best_n_estimators_ = int(model.best_iteration_)
        self._best_score_ = model.best_score_
        self._learning_rate_ = model.lr_
        self.refit_ = False
        self.refit_n_estimators_ = None
        self.refit_strategy_ = None

    def _record_refit_result(self, selection_model, strategy):
        self.selection_model_ = selection_model
        self.refit_ = True
        self.refit_n_estimators_ = len(self.model_.trees_)
        self.refit_strategy_ = strategy

    def _attach_validation_metadata(self, metadata):
        model = getattr(self, "model_", None)
        auto_params = getattr(model, "auto_params_", None)
        if auto_params is not None:
            auto_params["validation_split"] = metadata
            auto_params.setdefault("diagnostics", {})
            auto_params["diagnostics"]["validation_split"] = metadata

    def _attach_learning_rate_probe_metadata(self, metadata):
        if metadata is None:
            return
        model = getattr(self, "model_", None)
        auto_params = getattr(model, "auto_params_", None)
        if auto_params is not None:
            auto_params["learning_rate_probe"] = metadata
            auto_params.setdefault("diagnostics", {})
            auto_params["diagnostics"]["learning_rate_probe"] = metadata

    def _attach_selection_validation_metadata(self, metadata):
        model = getattr(self, "model_", None)
        auto_params = getattr(model, "auto_params_", None)
        if auto_params is not None:
            auto_params["selection_validation_split"] = metadata
            auto_params.setdefault("diagnostics", {})
            auto_params["diagnostics"]["selection_validation_split"] = metadata

    def _wrapper_state_header(self):
        if not hasattr(self, "model_"):
            return {}
        state = {
            "best_n_estimators": self.best_n_estimators_,
            "best_score": self.best_score_,
            "learning_rate": self.learning_rate_,
            "refit": getattr(self, "refit_", False),
            "refit_n_estimators": getattr(self, "refit_n_estimators_", None),
            "refit_strategy": getattr(self, "refit_strategy_", None),
        }
        if hasattr(self, "n_features_in_"):
            state["n_features_in"] = int(self.n_features_in_)
        if hasattr(self, "feature_names_in_"):
            state["feature_names_in"] = self.feature_names_in_.tolist()
        if getattr(self, "refit_", False):
            state["selection_model_persisted"] = False
        if hasattr(self, "_selection_n_total_"):
            state["selection_n_total"] = self._selection_n_total_
        if hasattr(self, "_selection_n_train_"):
            state["selection_n_train"] = self._selection_n_train_
        return state

    def _restore_wrapper_state(self, state):
        state = state or {}
        if "best_n_estimators" in state:
            self._best_n_estimators_ = int(state["best_n_estimators"])
        if "best_score" in state:
            self._best_score_ = state["best_score"]
        if "learning_rate" in state:
            self._learning_rate_ = state["learning_rate"]
        if "n_features_in" in state:
            self.n_features_in_ = int(state["n_features_in"])
        else:
            self._restore_n_features_from_model()
        if "feature_names_in" in state:
            self.feature_names_in_ = np.asarray(
                state["feature_names_in"], dtype=object
            )
        self.refit_ = bool(state.get("refit", False))
        self.refit_n_estimators_ = state.get("refit_n_estimators")
        self.refit_strategy_ = state.get("refit_strategy")
        if self.refit_ and state.get("selection_model_persisted") is False:
            self.selection_model_ = None
            self.selection_model_persisted_ = False
        if "selection_n_total" in state:
            self._selection_n_total_ = int(state["selection_n_total"])
        if "selection_n_train" in state:
            self._selection_n_train_ = int(state["selection_n_train"])

    def _refit_params_for_booster(self, strategy):
        params = self.get_refit_params(strategy=strategy)
        return {
            k: v for k, v in params.items()
            if k not in {"loss", "alpha"} | _SKLEARN_ONLY
        }

    def _refit_strategy_exponent(self, strategy):
        try:
            return _REFIT_STRATEGY_EXPONENT[strategy]
        except KeyError as exc:
            valid = ", ".join(sorted(_REFIT_STRATEGY_EXPONENT))
            raise ValueError(
                f"unknown refit strategy {strategy!r}; expected one of {valid}"
            ) from exc

    def _validate_refit_strategy_for_fit(self, strategy):
        exponent = self._refit_strategy_exponent(strategy)
        if exponent and not (hasattr(self, "_selection_n_total_") and
                             hasattr(self, "_selection_n_train_")):
            raise ValueError(
                f"strategy={strategy!r} requires an automatic validation "
                "split from fit; use strategy='exact' or set iterations "
                "manually when fit used an explicit eval_set"
            )

    def _learning_rate_probe_candidates(self, base_lr):
        values = self.auto_learning_rate_probe_values
        if values is None:
            raw = [0.5 * base_lr, 0.75 * base_lr, base_lr,
                   1.25 * base_lr, 1.5 * base_lr]
        else:
            raw = [float(v) for v in values]
        candidates = []
        for lr in raw:
            if lr <= 0.0 or not np.isfinite(lr):
                raise ValueError(
                    "auto_learning_rate_probe_values must contain positive "
                    "finite learning rates"
                )
            if not any(abs(lr - prev) <= 1e-15 for prev in candidates):
                candidates.append(float(lr))
        if not any(abs(base_lr - prev) <= 1e-15 for prev in candidates):
            candidates.append(float(base_lr))
        return candidates

    def _run_learning_rate_probe(
        self, make_model, X, y, *, cat_features, eval_set,
        sample_weight, eval_sample_weight, fit_kwargs
    ):
        if not self.auto_learning_rate_probe:
            return None, {"enabled": False, "reason": "disabled"}
        if eval_set is None:
            return None, {"enabled": False, "reason": "no_eval_set"}
        if not is_auto_learning_rate(self.learning_rate):
            return None, {"enabled": False, "reason": "learning_rate_explicit"}
        probe_iterations = int(
            min(max(1, int(self.auto_learning_rate_probe_iterations)),
                int(self.iterations))
        )
        final_iterations = int(fit_kwargs.get("iterations", self.iterations))
        context_kwargs = dict(fit_kwargs)
        context_kwargs["iterations"] = 0
        context_kwargs["diagnostic_warnings"] = "never"
        context_model = make_model(context_kwargs)
        context_model.fit(
            X, y, cat_features=cat_features, eval_set=eval_set,
            sample_weight=sample_weight,
            eval_sample_weight=eval_sample_weight,
        )
        context_auto = getattr(context_model, "auto_params_", {})
        p_model = (
            context_auto.get("learning_rate", {}).get("p_model")
            or context_auto.get("features", {}).get("model_feature_count")
        )
        n_eff = effective_sample_size(sample_weight, X.shape[0])
        n_eff_fraction = n_eff / float(X.shape[0]) if X.shape[0] else 0.0
        use_best_model = bool(
            eval_set is not None and getattr(context_model, "use_best_model_", False)
        )
        loss_name = getattr(context_model, "loss_name", "RMSE")
        max_leaves = context_model._max_tree_leaves()
        base_details = resolve_learning_rate_details(
            self.learning_rate,
            loss_name=loss_name,
            n_eff=n_eff,
            iterations=final_iterations,
            use_best_model=use_best_model,
            tree_mode=context_model.tree_mode_,
            max_leaves=max_leaves,
            n_eff_fraction=n_eff_fraction,
            p_model=p_model,
        )
        short_budget_details = resolve_learning_rate_details(
            self.learning_rate,
            loss_name=loss_name,
            n_eff=n_eff,
            iterations=probe_iterations,
            use_best_model=use_best_model,
            tree_mode=context_model.tree_mode_,
            max_leaves=max_leaves,
            n_eff_fraction=n_eff_fraction,
            p_model=p_model,
        )
        base_lr = float(base_details["resolved"])
        candidates = self._learning_rate_probe_candidates(base_lr)
        results = []
        best_lr = None
        best_score = np.inf
        for lr in candidates:
            probe_kwargs = dict(fit_kwargs)
            probe_kwargs["iterations"] = probe_iterations
            probe_kwargs["learning_rate"] = float(lr)
            model = make_model(probe_kwargs)
            model.fit(
                X, y, cat_features=cat_features, eval_set=eval_set,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight,
            )
            score = float(model.best_score_)
            results.append({
                "learning_rate": float(lr),
                "score": score,
                "best_iteration": int(model.best_iteration_),
                "source": (
                    "auto_base" if abs(lr - base_lr) <= 1e-15 else "candidate"
                ),
            })
            if score < best_score:
                best_score = score
                best_lr = float(lr)
        if best_lr is None:
            best_lr = base_lr
            best_score = float("nan")
        return best_lr, {
            "enabled": True,
            "probe_iterations": probe_iterations,
            "final_iterations": final_iterations,
            "base_learning_rate": base_lr,
            "base_learning_rate_full_iterations": base_lr,
            "base_learning_rate_short_iterations": float(
                short_budget_details["resolved"]
            ),
            "base_learning_rate_details": base_details,
            "selected_learning_rate": float(best_lr),
            "selected_score": float(best_score),
            "candidates": results,
        }

    def get_refit_params(self, strategy="exact"):
        """Return parameters for a fresh full-data refit.

        The returned params disable early stopping, set ``iterations`` to the
        fitted round count (or an explicit scaling of it), and freeze the
        resolved learning rate from the selection fit. Freezing the learning
        rate avoids changing the boosting path when ``learning_rate=None`` was
        used with early stopping.

        Parameters
        ----------
        strategy : {"exact", "best", "sqrt", "linear", "scaled"}
            ``"exact"`` and ``"best"`` use the fitted number of boosting
            rounds. ``"sqrt"`` and ``"linear"`` scale that count by the
            empirical automatic validation split ratio. ``"scaled"`` is an
            alias for ``"linear"``.
        """
        if not hasattr(self, "model_"):
            raise ValueError("model must be fitted before calling get_refit_params")

        exponent = self._refit_strategy_exponent(strategy)

        rounds = int(self.best_n_estimators_)
        if exponent:
            if not (hasattr(self, "_selection_n_total_") and
                    hasattr(self, "_selection_n_train_")):
                raise ValueError(
                    f"strategy={strategy!r} requires an automatic validation "
                    "split from fit; use strategy='exact' or set iterations "
                    "manually when fit used an explicit eval_set"
                )
            scale = self._selection_n_total_ / max(1, self._selection_n_train_)
            rounds = int(np.ceil(rounds * (scale ** exponent)))

        params = self.get_params()
        params["iterations"] = max(0, rounds)
        params["learning_rate"] = self.learning_rate_
        params["early_stopping"] = False
        params["early_stopping_rounds"] = None
        auto = getattr(self.model_, "auto_params_", {})
        resolved = auto.get("auto_structure", {}).get("resolved", {})
        for name in (
            "depth", "num_leaves", "l2_leaf_reg", "min_child_samples",
            "min_child_weight", "cat_smoothing",
        ):
            if name in resolved:
                params[name] = resolved[name]["resolved"]
        if "cat_smoothing" not in resolved and "binning" in auto:
            params["cat_smoothing"] = auto["binning"].get(
                "cat_smoothing_resolved", params.get("cat_smoothing")
            )
        if "refit" in params:
            params["refit"] = False
        return params

    @property
    def best_n_estimators_(self):
        """Number of boosting rounds selected/retained by the fitted model."""
        check_is_fitted(self, "model_")
        return getattr(self, "_best_n_estimators_", self.model_.best_iteration_)

    @property
    def n_estimators_(self):
        """Number of boosting rounds present in the fitted model."""
        check_is_fitted(self, "model_")
        return len(self.model_.trees_)

    @property
    def learning_rate_(self):
        """Resolved learning rate used by the fitted booster."""
        check_is_fitted(self, "model_")
        return getattr(self, "_learning_rate_", self.model_.lr_)


class ChimeraBoostRegressor(RegressorMixin, _RefitParamsMixin, BaseEstimator):
    """Gradient boosted oblivious trees for regression.

    loss: "RMSE" (default), "MAE", or "Quantile". For "Quantile" pass the level
    via `alpha` (e.g. alpha=0.9 for the 90th-percentile predictor).

    early_stopping : bool, default False
        Whether to use early stopping to terminate training when the validation
        score stops improving.  Requires ``early_stopping_rounds`` (resolved
        automatically from the learning rate when early stopping is active but
        the param is None).
    validation_fraction : float, default 0.1
        Fraction of training data to hold out as a validation set when
        *early_stopping* is active and no explicit *eval_set* is passed.
        Ignored when an explicit *eval_set* is given to ``fit``.
    """

    def __init__(self, iterations=1000, learning_rate=None, depth=None,
                 l2_leaf_reg="auto", max_bins=254, subsample=1.0, colsample=1.0,
                 cat_smoothing=1.0, early_stopping_rounds=None,
                 early_stopping_min_delta=None,
                 loss="RMSE", alpha=0.5, min_child_weight=1.0,
                 min_child_samples=20, min_gain_to_split=0.0, num_leaves=None,
                 thread_count=None, random_state=None, verbose=False,
                 ordered_boosting="auto",
                 early_stopping=False, validation_fraction=0.1,
                 validation_strategy="random",
                 refit=False, refit_strategy="exact",
                 verbose_timing=False, tree_mode="catboost",
                 sampling="uniform", top_rate=0.2, other_rate=0.1,
                 eval_train_loss=True, bin_sample_count=200_000,
                 histogram_parallelism="auto", use_best_model=True,
                 bootstrap_type="none", bagging_temperature=0.0,
                 mvs_reg=1.0, random_strength=0.0,
                 diagnostic_warnings="once", auto_learning_rate_probe=False,
                 auto_learning_rate_probe_values=None,
                 auto_learning_rate_probe_iterations=80):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.max_bins = max_bins
        self.subsample = subsample
        self.colsample = colsample
        self.cat_smoothing = cat_smoothing
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_min_delta = early_stopping_min_delta
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
        self.validation_strategy = validation_strategy
        self.refit = refit
        self.refit_strategy = refit_strategy
        self.verbose_timing = verbose_timing
        self.tree_mode = tree_mode
        self.sampling = sampling
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.eval_train_loss = eval_train_loss
        self.bin_sample_count = bin_sample_count
        self.histogram_parallelism = histogram_parallelism
        self.use_best_model = use_best_model
        self.bootstrap_type = bootstrap_type
        self.bagging_temperature = bagging_temperature
        self.mvs_reg = mvs_reg
        self.random_strength = random_strength
        self.diagnostic_warnings = diagnostic_warnings
        self.auto_learning_rate_probe = auto_learning_rate_probe
        self.auto_learning_rate_probe_values = auto_learning_rate_probe_values
        self.auto_learning_rate_probe_iterations = auto_learning_rate_probe_iterations

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
        X_input = X
        X, cat_features, n_features = _coerce_fit_X(X, cat_features)
        eval_set = _ensure_dense_eval_set(eval_set)
        eval_set = _validate_eval_set_features(eval_set, n_features)
        y = np.asarray(y, dtype=np.float64)
        sample_weight = _validate_wrapper_sample_weight(
            sample_weight, X.shape[0]
        )
        X_full, y_full = X, y
        sample_weight_full = sample_weight
        explicit_eval_set = eval_set is not None
        validation_strategy_ = _normalize_validation_strategy(
            self.validation_strategy
        )
        validation_fraction_resolved = None
        realized_validation_policy = "none"
        split_train_n = X.shape[0]
        split_eval_n = None

        self._clear_refit_selection_metadata()
        if self.refit:
            self._refit_strategy_exponent(self.refit_strategy)
        es_active = _should_early_stop(self.early_stopping)
        if (
            es_active
            and eval_set is None
            and groups is not None
            and validation_strategy_ == "weighted_stratified"
        ):
            raise ValueError(
                "validation_strategy='weighted_stratified' is only supported "
                "for ungrouped regression automatic validation splits"
            )
        if es_active and eval_set is None:
            n_total = X.shape[0]
            validation_fraction_resolved = _resolve_validation_fraction(
                self.validation_fraction, sample_weight, n_total
            )
            train_idx, val_idx, realized_validation_policy = _make_eval_split(
                X, y, validation_fraction_resolved, self.random_state,
                groups=groups, stratify=None, sample_weight=sample_weight,
                validation_strategy=validation_strategy_,
            )
            self._record_refit_selection_metadata(n_total, train_idx)
            eval_set = (X[val_idx], y[val_idx])
            split_eval_n = len(val_idx)
            if sample_weight is not None:
                eval_sample_weight = sample_weight[val_idx]
            X, y = X[train_idx], y[train_idx]
            if sample_weight is not None:
                sample_weight = sample_weight[train_idx]

        es_rounds = self.early_stopping_rounds
        if es_active and es_rounds is None:
            es_rounds = "auto"
        selection_active = (
            eval_set is not None and (es_rounds is not None or self.use_best_model)
        )
        if self.refit and selection_active:
            self._validate_refit_strategy_for_fit(self.refit_strategy)

        loss_kwargs = {"alpha": self.alpha} if self.loss == "Quantile" else {}
        kw = {k: v for k, v in self.get_params().items()
              if k not in {"loss", "alpha"} | _SKLEARN_ONLY}
        kw["early_stopping_rounds"] = es_rounds
        probe_lr, probe_metadata = self._run_learning_rate_probe(
            lambda probe_kw: GradientBoosting(
                loss=self.loss, loss_kwargs=loss_kwargs, **probe_kw
            ),
            X, y,
            cat_features=cat_features,
            eval_set=eval_set,
            sample_weight=sample_weight,
            eval_sample_weight=eval_sample_weight,
            fit_kwargs=kw,
        )
        if probe_lr is not None:
            kw["learning_rate"] = probe_lr
        model = GradientBoosting(loss=self.loss, loss_kwargs=loss_kwargs, **kw)
        model.fit(
            X, y, cat_features=cat_features, eval_set=eval_set,
            sample_weight=sample_weight,
            eval_sample_weight=eval_sample_weight,
        )
        self.model_ = model
        self._record_input_feature_metadata(X_input, n_features)
        if explicit_eval_set:
            split_source = "explicit_eval_set"
            realized_validation_policy = "explicit_eval_set"
            split_eval_n = len(eval_set[1]) if eval_set is not None else None
        elif es_active:
            split_source = "automatic"
        else:
            split_source = "none"
        validation_metadata = {
            "source": split_source,
            "validation_fraction_input": self.validation_fraction,
            "validation_fraction_resolved": validation_fraction_resolved,
            "validation_strategy": validation_strategy_,
            "realized_validation_strategy": realized_validation_policy,
            "groups_provided": groups is not None,
            "sample_weight_provided": sample_weight_full is not None,
            "train_n_samples": int(X.shape[0]),
            "eval_n_samples": None if split_eval_n is None else int(split_eval_n),
            "original_n_samples": int(split_train_n),
            "refit": bool(self.refit and selection_active),
        }
        self._attach_validation_metadata(validation_metadata)
        self._attach_learning_rate_probe_metadata(probe_metadata)
        selection_model = self.model_
        self._record_selection_result(selection_model)

        if self.refit and selection_active:
            refit_kw = self._refit_params_for_booster(self.refit_strategy)
            refit_model = GradientBoosting(
                loss=self.loss, loss_kwargs=loss_kwargs, **refit_kw
            )
            refit_model.fit(
                X_full, y_full, cat_features=cat_features,
                sample_weight=sample_weight_full,
            )
            self.model_ = refit_model
            self._record_refit_result(selection_model, self.refit_strategy)
            refit_validation_metadata = {
                "source": "refit_full_data",
                "selection_source": validation_metadata["source"],
                "validation_fraction_input": self.validation_fraction,
                "validation_fraction_resolved": None,
                "validation_strategy": validation_strategy_,
                "realized_validation_strategy": "refit_full_data",
                "groups_provided": groups is not None,
                "sample_weight_provided": sample_weight_full is not None,
                "train_n_samples": int(X_full.shape[0]),
                "eval_n_samples": None,
                "original_n_samples": int(X_full.shape[0]),
                "refit": True,
            }
            self._attach_validation_metadata(refit_validation_metadata)
            self._attach_selection_validation_metadata(validation_metadata)
            self._attach_learning_rate_probe_metadata(probe_metadata)
        return self

    def predict(self, X):
        X = _check_predict_input(self, X)
        return self.model_.predict_raw(X)

    def staged_predict(self, X):
        """Yield the prediction after each successive tree."""
        X = _check_predict_input(self, X)
        yield from self.model_.staged_predict_raw(X)

    def save_model(self, path):
        """Serialize the fitted model to a single ``.npz`` file."""
        check_is_fitted(self, "model_")
        from .serialization import save_booster
        save_booster(
            self.model_, path,
            wrapper_header={"wrapper_class": type(self).__name__,
                            "params": self.get_params(),
                            "state": self._wrapper_state_header()},
        )

    @classmethod
    def load_model(cls, path):
        """Load a model saved with :meth:`save_model`."""
        from .serialization import load_booster
        booster, wrapper_header, _ = load_booster(
            path, return_wrapper_payload=True
        )
        saved_class = wrapper_header.get("wrapper_class")
        if saved_class is not None and saved_class != cls.__name__:
            raise TypeError(
                f"{path!r} was saved by {saved_class}, not {cls.__name__}"
            )
        if isinstance(booster, MulticlassBoosting):
            raise TypeError(
                f"{path!r} contains a multiclass model; "
                "use ChimeraBoostClassifier.load_model"
            )
        est = cls()
        params = wrapper_header.get("params") or {}
        known = est.get_params()
        est.set_params(**{k: v for k, v in params.items() if k in known})
        est.model_ = booster
        est._restore_wrapper_state(wrapper_header.get("state", {}))
        return est

    @property
    def best_iteration_(self):
        return self.best_n_estimators_

    @property
    def best_score_(self):
        check_is_fitted(self, "model_")
        return getattr(self, "_best_score_", self.model_.best_score_)

    @property
    def feature_importances_(self):
        check_is_fitted(self, "model_")
        return self.model_.feature_importances_

    @property
    def timing_(self):
        check_is_fitted(self, "model_")
        return self.model_.timing_


class ChimeraBoostClassifier(ClassifierMixin, _RefitParamsMixin, BaseEstimator):
    """Gradient boosted oblivious trees for classification.

    Automatically uses binary logloss for 2 classes and softmax multiclass for
    3+. `classes_` preserves the original label values.

    early_stopping : bool, default False
        Whether to use early stopping.  Patience is resolved automatically from
        the learning rate when ``early_stopping_rounds`` is None. The
        validation split is always stratified to preserve class proportions;
        when *groups* is passed, ``StratifiedGroupKFold`` is used instead.
    validation_fraction : float, default 0.1
        Fraction of training data held out for the automatic validation set.
        Ignored when an explicit *eval_set* is given to ``fit``.
    """

    def __init__(self, iterations=1000, learning_rate=None, depth=None,
                 l2_leaf_reg="auto", max_bins=254, subsample=1.0, colsample=1.0,
                 cat_smoothing=1.0, early_stopping_rounds=None,
                 early_stopping_min_delta=None,
                 min_child_weight=1.0, min_child_samples=20,
                 min_gain_to_split=0.0, num_leaves=None, thread_count=None,
                 random_state=None, verbose=False, ordered_boosting="auto",
                 early_stopping=False, validation_fraction=0.1,
                 validation_strategy="random",
                 refit=False, refit_strategy="exact",
                 verbose_timing=False, tree_mode="catboost",
                 sampling="uniform", top_rate=0.2, other_rate=0.1,
                 multiclass_tree_strategy="auto", eval_train_loss=True,
                 bin_sample_count=200_000, histogram_parallelism="auto",
                 use_best_model=True, bootstrap_type="none",
                 bagging_temperature=0.0, mvs_reg=1.0,
                 random_strength=0.0, diagnostic_warnings="once",
                 auto_learning_rate_probe=False,
                 auto_learning_rate_probe_values=None,
                 auto_learning_rate_probe_iterations=80):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.max_bins = max_bins
        self.subsample = subsample
        self.colsample = colsample
        self.cat_smoothing = cat_smoothing
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_min_delta = early_stopping_min_delta
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
        self.validation_strategy = validation_strategy
        self.refit = refit
        self.refit_strategy = refit_strategy
        self.verbose_timing = verbose_timing
        self.tree_mode = tree_mode
        self.sampling = sampling
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.multiclass_tree_strategy = multiclass_tree_strategy
        self.eval_train_loss = eval_train_loss
        self.bin_sample_count = bin_sample_count
        self.histogram_parallelism = histogram_parallelism
        self.use_best_model = use_best_model
        self.bootstrap_type = bootstrap_type
        self.bagging_temperature = bagging_temperature
        self.mvs_reg = mvs_reg
        self.random_strength = random_strength
        self.diagnostic_warnings = diagnostic_warnings
        self.auto_learning_rate_probe = auto_learning_rate_probe
        self.auto_learning_rate_probe_values = auto_learning_rate_probe_values
        self.auto_learning_rate_probe_iterations = auto_learning_rate_probe_iterations

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
        X_input = X
        X, cat_features, n_features = _coerce_fit_X(X, cat_features)
        eval_set = _ensure_dense_eval_set(eval_set)
        eval_set = _validate_eval_set_features(eval_set, n_features)
        y = np.asarray(y)
        classes = np.unique(y)
        n_classes = classes.size
        if n_classes < 2:
            raise ValueError("Need at least 2 classes.")
        sample_weight = _validate_wrapper_sample_weight(
            sample_weight, X.shape[0]
        )
        X_full, y_full = X, y
        sample_weight_full = sample_weight
        explicit_eval_set = eval_set is not None
        validation_strategy_ = _normalize_validation_strategy(
            self.validation_strategy
        )
        validation_fraction_resolved = None
        realized_validation_policy = "none"
        split_train_n = X.shape[0]
        split_eval_n = None

        self._clear_refit_selection_metadata()
        if self.refit:
            self._refit_strategy_exponent(self.refit_strategy)
        es_active = _should_early_stop(self.early_stopping)
        if (
            es_active
            and eval_set is None
            and validation_strategy_ == "weighted_stratified"
        ):
            raise ValueError(
                "validation_strategy='weighted_stratified' is only supported "
                "for regression automatic validation splits"
            )
        if es_active and eval_set is None:
            n_total = X.shape[0]
            validation_fraction_resolved = _resolve_validation_fraction(
                self.validation_fraction, sample_weight, n_total
            )
            train_idx, val_idx, realized_validation_policy = _make_eval_split(
                X, y, validation_fraction_resolved, self.random_state,
                groups=groups, stratify=y,  # always stratify for classification
                sample_weight=sample_weight,
                validation_strategy=validation_strategy_,
            )
            self._record_refit_selection_metadata(n_total, train_idx)
            eval_set = (X[val_idx], y[val_idx])
            split_eval_n = len(val_idx)
            if sample_weight is not None:
                eval_sample_weight = sample_weight[val_idx]
            X, y = X[train_idx], y[train_idx]
            if sample_weight is not None:
                sample_weight = sample_weight[train_idx]
            train_classes = np.unique(y)
            if train_classes.size != n_classes:
                raise ValueError("automatic validation split removed a class from training data")

        es_rounds = self.early_stopping_rounds
        if es_active and es_rounds is None:
            es_rounds = "auto"
        selection_active = (
            eval_set is not None and (es_rounds is not None or self.use_best_model)
        )
        if self.refit and selection_active:
            self._validate_refit_strategy_for_fit(self.refit_strategy)

        kw = {k: v for k, v in self.get_params().items()
              if k not in _SKLEARN_ONLY}
        kw["early_stopping_rounds"] = es_rounds
        probe_metadata = None

        if n_classes == 2:
            multiclass = False
            y01 = (y == classes[1]).astype(np.float64)
            if eval_set is not None:
                Xv, yv = eval_set
                if np.any(~np.isin(np.asarray(yv), classes)):
                    raise ValueError("eval_set contains labels not present in training data")
                eval_set = (Xv, (np.asarray(yv) == classes[1]).astype(np.float64))
            probe_lr, probe_metadata = self._run_learning_rate_probe(
                lambda probe_kw: GradientBoosting(loss="Logloss", **probe_kw),
                X, y01,
                cat_features=cat_features,
                eval_set=eval_set,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight,
                fit_kwargs=kw,
            )
            if probe_lr is not None:
                kw["learning_rate"] = probe_lr
            model = GradientBoosting(loss="Logloss", **kw)
            model.fit(
                X, y01, cat_features=cat_features, eval_set=eval_set,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight,
            )
        else:
            multiclass = True
            probe_lr, probe_metadata = self._run_learning_rate_probe(
                lambda probe_kw: MulticlassBoosting(**probe_kw),
                X, y,
                cat_features=cat_features,
                eval_set=eval_set,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight,
                fit_kwargs=kw,
            )
            if probe_lr is not None:
                kw["learning_rate"] = probe_lr
            model = MulticlassBoosting(**kw)
            model.fit(
                X, y, cat_features=cat_features, eval_set=eval_set,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight,
            )
            classes = model.classes_
        self.model_ = model
        self._multiclass = multiclass
        self.classes_ = classes
        self.n_classes_ = len(classes)
        self._record_input_feature_metadata(X_input, n_features)
        if explicit_eval_set:
            split_source = "explicit_eval_set"
            realized_validation_policy = "explicit_eval_set"
            split_eval_n = len(eval_set[1]) if eval_set is not None else None
        elif es_active:
            split_source = "automatic"
        else:
            split_source = "none"
        validation_metadata = {
            "source": split_source,
            "validation_fraction_input": self.validation_fraction,
            "validation_fraction_resolved": validation_fraction_resolved,
            "validation_strategy": validation_strategy_,
            "realized_validation_strategy": realized_validation_policy,
            "groups_provided": groups is not None,
            "sample_weight_provided": sample_weight_full is not None,
            "train_n_samples": int(X.shape[0]),
            "eval_n_samples": None if split_eval_n is None else int(split_eval_n),
            "original_n_samples": int(split_train_n),
            "refit": bool(self.refit and selection_active),
        }
        self._attach_validation_metadata(validation_metadata)
        self._attach_learning_rate_probe_metadata(probe_metadata)
        selection_model = self.model_
        self._record_selection_result(selection_model)

        if self.refit and selection_active:
            refit_kw = self._refit_params_for_booster(self.refit_strategy)
            if multiclass:
                refit_model = MulticlassBoosting(**refit_kw)
                refit_model.fit(
                    X_full, y_full, cat_features=cat_features,
                    sample_weight=sample_weight_full,
                )
                classes = refit_model.classes_
            else:
                y01_full = (y_full == classes[1]).astype(np.float64)
                refit_model = GradientBoosting(loss="Logloss", **refit_kw)
                refit_model.fit(
                    X_full, y01_full, cat_features=cat_features,
                    sample_weight=sample_weight_full,
                )
            self.model_ = refit_model
            self._multiclass = multiclass
            self.classes_ = classes
            self.n_classes_ = len(classes)
            self._record_refit_result(selection_model, self.refit_strategy)
            refit_validation_metadata = {
                "source": "refit_full_data",
                "selection_source": validation_metadata["source"],
                "validation_fraction_input": self.validation_fraction,
                "validation_fraction_resolved": None,
                "validation_strategy": validation_strategy_,
                "realized_validation_strategy": "refit_full_data",
                "groups_provided": groups is not None,
                "sample_weight_provided": sample_weight_full is not None,
                "train_n_samples": int(X_full.shape[0]),
                "eval_n_samples": None,
                "original_n_samples": int(X_full.shape[0]),
                "refit": True,
            }
            self._attach_validation_metadata(refit_validation_metadata)
            self._attach_selection_validation_metadata(validation_metadata)
            self._attach_learning_rate_probe_metadata(probe_metadata)
        return self

    def predict_proba(self, X):
        X = _check_predict_input(self, X)
        raw = self.model_.predict_raw(X)
        if self._multiclass:
            return self.model_.loss_.transform(raw)            # (n, K)
        p1 = self.model_.loss_.transform(raw)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        X = _check_predict_input(self, X)
        raw = self.model_.predict_raw(X)
        if self._multiclass:
            return self.classes_[np.argmax(raw, axis=1)]
        p1 = self.model_.loss_.transform(raw)
        return self.classes_[(p1 > 0.5).astype(np.int64)]

    def staged_predict_proba(self, X):
        """Yield class probabilities after each successive boosting round."""
        X = _check_predict_input(self, X)
        for raw in self.model_.staged_predict_raw(X):
            if self._multiclass:
                yield self.model_.loss_.transform(raw)
            else:
                p1 = self.model_.loss_.transform(raw)
                yield np.column_stack([1.0 - p1, p1])

    def staged_predict(self, X):
        """Yield class labels after each successive boosting round."""
        X = _check_predict_input(self, X)
        for raw in self.model_.staged_predict_raw(X):
            if self._multiclass:
                yield self.classes_[np.argmax(raw, axis=1)]
            else:
                p1 = self.model_.loss_.transform(raw)
                yield self.classes_[(p1 > 0.5).astype(np.int64)]

    def staged_predict_raw(self, X):
        """Yield raw margins after each successive boosting round."""
        X = _check_predict_input(self, X)
        yield from self.model_.staged_predict_raw(X)

    def save_model(self, path):
        """Serialize the fitted model to a single ``.npz`` file."""
        check_is_fitted(self, "model_")
        from .serialization import _encode_categories, save_booster

        cls_arr = np.asarray(self.classes_)
        if cls_arr.dtype == object:
            values, kinds = _encode_categories(self.classes_)
            wrapper_arrays = {"classes": values, "classes_kinds": kinds}
        else:
            wrapper_arrays = {"classes": cls_arr}
        save_booster(
            self.model_, path,
            wrapper_header={"wrapper_class": type(self).__name__,
                            "params": self.get_params(),
                            "state": self._wrapper_state_header()},
            wrapper_arrays=wrapper_arrays,
        )

    @classmethod
    def load_model(cls, path):
        """Load a model saved with :meth:`save_model`."""
        from .serialization import _decode_categories, load_booster

        booster, wrapper_header, wrapper_arrays = load_booster(
            path, return_wrapper_payload=True
        )
        saved_class = wrapper_header.get("wrapper_class")
        if saved_class is not None and saved_class != cls.__name__:
            raise TypeError(
                f"{path!r} was saved by {saved_class}, not {cls.__name__}"
            )
        est = cls()
        params = wrapper_header.get("params") or {}
        known = est.get_params()
        est.set_params(**{k: v for k, v in params.items() if k in known})
        est.model_ = booster
        est._restore_wrapper_state(wrapper_header.get("state", {}))
        est._multiclass = isinstance(booster, MulticlassBoosting)
        if "classes" in wrapper_arrays:
            classes = wrapper_arrays["classes"]
            if "classes_kinds" in wrapper_arrays:
                classes = _decode_categories(
                    classes, wrapper_arrays["classes_kinds"]
                )
        elif est._multiclass:
            classes = booster.classes_  # booster-level multiclass save
        else:
            raise ValueError(
                f"{path!r} has no class labels; binary classifiers must be "
                "saved with ChimeraBoostClassifier.save_model"
            )
        est.classes_ = classes
        est.n_classes_ = len(classes)
        return est

    @property
    def best_iteration_(self):
        return self.best_n_estimators_

    @property
    def best_score_(self):
        check_is_fitted(self, "model_")
        return getattr(self, "_best_score_", self.model_.best_score_)

    @property
    def feature_importances_(self):
        check_is_fitted(self, "model_")
        return self.model_.feature_importances_

    @property
    def timing_(self):
        check_is_fitted(self, "model_")
        return self.model_.timing_
