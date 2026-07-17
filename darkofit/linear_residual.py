"""Linear residual trend helpers for the sklearn regression wrapper.

The public estimator owns when this trend is fitted and how predictions are
combined.  This module only selects numeric raw input columns and fits a small
weighted ridge model whose state can be serialized as JSON metadata plus plain
numpy arrays.
"""

import operator

import numpy as np

from ._validation import (
    array_like_to_numpy,
    n_features_from_array_like,
    n_samples_from_array_like,
)


SUPPORTED_LINEAR_RESIDUAL_LOSSES = frozenset({
    "RMSE",
    "MAE",
    "Quantile",
    "Gaussian",
    "StudentT",
})
UNSUPPORTED_LINEAR_RESIDUAL_V1_LOSSES = frozenset({
    "LogNormal",
    "Poisson",
    "NegativeBinomial",
})

_MISSING_REASON = "all_missing"
_NON_NUMERIC_REASON = "non_numeric"
_CONSTANT_REASON = "constant"
_SCALE_FLOOR = np.sqrt(np.finfo(np.float64).eps)


def validate_linear_residual_loss(loss):
    """Raise when the v1 additive-location protocol is invalid for *loss*."""
    if loss in SUPPORTED_LINEAR_RESIDUAL_LOSSES:
        return
    if loss in UNSUPPORTED_LINEAR_RESIDUAL_V1_LOSSES:
        raise ValueError(
            "linear_residual=True is not supported for loss="
            f"{loss!r} in v1. LogNormal, Poisson, and NegativeBinomial need "
            "a distribution-specific offset protocol; disable "
            "linear_residual or use RMSE, MAE, Quantile, Gaussian, or "
            "StudentT."
        )
    raise ValueError(
        "linear_residual=True is only supported for loss='RMSE', 'MAE', "
        "'Quantile', 'Gaussian', or 'StudentT'"
    )


def _as_2d_array(X, *, dtype=object):
    arr = array_like_to_numpy(X, dtype)
    if arr.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    return arr


def _numeric_matrix_or_none(X):
    try:
        arr = array_like_to_numpy(X, np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2:
        return None
    return np.asarray(arr, dtype=np.float64)


def _feature_name(feature_names, index):
    if feature_names is None:
        return None
    return str(feature_names[int(index)])


def _column_values(X, index, *, numeric_matrix=None):
    if numeric_matrix is not None:
        return numeric_matrix[:, int(index)]
    iloc = getattr(X, "iloc", None)
    if iloc is not None:
        return np.asarray(iloc[:, int(index)], dtype=object)
    arr = _as_2d_array(X, dtype=object)
    return np.asarray(arr[:, int(index)], dtype=object)


def _coerce_numeric_column(X, index, *, numeric_matrix=None):
    values = _column_values(X, index, numeric_matrix=numeric_matrix)
    try:
        col = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(_NON_NUMERIC_REASON) from exc
    if col.ndim != 1:
        raise ValueError(_NON_NUMERIC_REASON)
    return col


def _weighted_mean(values, weights):
    if weights is None:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


def _weighted_variance(values, weights, mean):
    centered = values - float(mean)
    if weights is None:
        return float(np.mean(centered * centered))
    return float(np.average(centered * centered, weights=weights))


def _normalize_sample_weight(sample_weight, n_samples):
    if sample_weight is None:
        raw = np.ones(int(n_samples), dtype=np.float64)
    else:
        raw = np.asarray(sample_weight, dtype=np.float64)
        if raw.shape != (int(n_samples),):
            raise ValueError(f"sample_weight must have shape ({int(n_samples)},)")
        if not np.all(np.isfinite(raw)):
            raise ValueError("sample_weight must contain only finite values")
        if np.any(raw < 0.0):
            raise ValueError("sample_weight must be nonnegative")
    positive = raw > 0.0
    if not np.any(positive):
        raise ValueError("sample_weight must have positive total weight")
    weight_sum = float(np.sum(raw[positive]))
    positive_n = int(np.sum(positive))
    normalized = np.zeros_like(raw, dtype=np.float64)
    normalized[positive] = raw[positive] * (float(positive_n) / weight_sum)
    effective_n = float(weight_sum * weight_sum / np.sum(raw[positive] ** 2))
    return raw, normalized, positive, weight_sum, positive_n, effective_n


def _normalize_feature_names(feature_names, n_features):
    if feature_names is None:
        return None
    names = np.asarray(feature_names, dtype=object)
    if names.ndim != 1 or names.shape[0] != int(n_features):
        raise ValueError("feature_names must have shape (n_features,)")
    return names


def _normalize_feature_selector(features, *, feature_names, n_features, cat_features):
    explicit = not (
        features is None
        or (isinstance(features, str) and features == "auto")
    )
    cat_set = {int(idx) for idx in (cat_features or ())}
    if not explicit:
        return [
            int(idx) for idx in range(int(n_features)) if int(idx) not in cat_set
        ], False

    if isinstance(features, (str, bytes)):
        raw_values = [features]
    else:
        arr = np.asarray(features)
        if arr.ndim == 0:
            raw_values = [arr.item()]
        elif arr.dtype == np.bool_:
            if arr.shape != (int(n_features),):
                raise ValueError(
                    "linear_residual_features boolean mask must have shape "
                    "(n_features,)"
                )
            raw_values = np.flatnonzero(arr).tolist()
        elif isinstance(features, np.ndarray):
            raw_values = arr.ravel().tolist()
        else:
            raw_values = list(features)

    if not raw_values:
        raise ValueError("linear_residual_features must select at least one column")

    all_str = all(isinstance(value, str) for value in raw_values)
    all_int = all(
        not isinstance(value, (bool, np.bool_))
        and not isinstance(value, str)
        for value in raw_values
    )
    if all_str:
        if feature_names is None:
            raise ValueError(
                "string linear_residual_features require named input columns"
            )
        indices = []
        names = np.asarray(feature_names, dtype=object)
        for value in raw_values:
            matches = np.flatnonzero(names == value)
            if matches.size != 1:
                raise ValueError(
                    f"linear_residual_features={value!r} was not found "
                    "uniquely in the fit feature names"
                )
            indices.append(int(matches[0]))
    elif all_int:
        indices = []
        for value in raw_values:
            try:
                idx = operator.index(value)
            except TypeError as exc:
                raise ValueError(
                    "linear_residual_features must be column names, integer "
                    "indices, a boolean mask, 'auto', or None"
                ) from exc
            if idx < 0 or idx >= int(n_features):
                raise ValueError(
                    f"linear_residual_features index {idx} is out of bounds "
                    f"for {int(n_features)} input features"
                )
            indices.append(int(idx))
    else:
        raise ValueError(
            "linear_residual_features must not mix column names with indices"
        )

    if len(set(indices)) != len(indices):
        raise ValueError("linear_residual_features contains duplicate columns")
    cat_selected = sorted(set(indices) & cat_set)
    if cat_selected:
        raise ValueError(
            "linear_residual_features cannot include categorical columns; "
            f"got indices {cat_selected}"
        )
    return indices, True


def _drop_record(index, feature_names, reason, *, explicit):
    record = {
        "index": int(index),
        "reason": reason,
        "source": "explicit" if explicit else "auto",
    }
    name = _feature_name(feature_names, index)
    if name is not None:
        record["name"] = name
    return record


def _as_float_array(name, value, *, ndim=1, length=None, positive=False):
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != int(ndim):
        raise ValueError(f"invalid DarkoFit model: {name} must be {ndim}D")
    if length is not None and arr.shape[0] != int(length):
        raise ValueError(
            f"invalid DarkoFit model: {name} length does not match "
            "linear residual feature count"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"invalid DarkoFit model: {name} must contain finite values"
        )
    if positive and np.any(arr <= 0.0):
        raise ValueError(
            f"invalid DarkoFit model: {name} must contain positive values"
        )
    return arr.astype(np.float64, copy=True)


class WeightedRidgeTrend:
    """Weighted ridge trend over selected numeric raw input columns."""

    def __init__(
        self,
        *,
        alpha=1.0,
        features="auto",
        fit_intercept=True,
        standardize=True,
    ):
        self.alpha = self._validate_alpha(alpha)
        self.features = features
        self.fit_intercept = bool(fit_intercept)
        self.standardize = bool(standardize)
        self.active_ = False
        self.inactive_reason_ = None

    @staticmethod
    def _validate_alpha(alpha):
        value = float(alpha)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("linear_residual_alpha must be a finite nonnegative number")
        return value

    def fit(self, X, y, *, sample_weight=None, cat_features=None, feature_names=None):
        n_features = n_features_from_array_like(X)
        n_samples = n_samples_from_array_like(X)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1 or y.shape[0] != n_samples:
            raise ValueError("y must be a 1-dimensional array aligned with X")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values")
        feature_names = _normalize_feature_names(feature_names, n_features)
        numeric_matrix = _numeric_matrix_or_none(X)
        raw_w, w, positive, weight_sum, positive_n, effective_n = (
            _normalize_sample_weight(sample_weight, y.shape[0])
        )
        candidate_indices, explicit = _normalize_feature_selector(
            self.features,
            feature_names=feature_names,
            n_features=n_features,
            cat_features=cat_features,
        )
        columns = []
        selected = []
        centers = []
        scales = []
        imputes = []
        dropped = []
        w_pos = w[positive]

        for index in candidate_indices:
            try:
                raw_col = _coerce_numeric_column(
                    X, index, numeric_matrix=numeric_matrix
                )
            except ValueError:
                if explicit:
                    raise ValueError(
                        "linear_residual_features selected non-numeric "
                        f"column {index}"
                    )
                dropped.append(
                    _drop_record(
                        index, feature_names, _NON_NUMERIC_REASON,
                        explicit=explicit,
                    )
                )
                continue
            finite = np.isfinite(raw_col)
            fit_finite = finite & positive
            if not np.any(fit_finite):
                if explicit:
                    raise ValueError(
                        "linear_residual_features selected a column with no "
                        f"finite positive-weight values: {index}"
                    )
                dropped.append(
                    _drop_record(
                        index, feature_names, _MISSING_REASON,
                        explicit=explicit,
                    )
                )
                continue
            impute = _weighted_mean(raw_col[fit_finite], w[fit_finite])
            imputed = np.where(finite, raw_col, impute)
            finite_center = _weighted_mean(imputed[positive], w_pos)
            variance = _weighted_variance(imputed[positive], w_pos, finite_center)
            if variance <= np.finfo(np.float64).eps:
                dropped.append(
                    _drop_record(
                        index, feature_names, _CONSTANT_REASON,
                        explicit=explicit,
                    )
                )
                continue
            center = finite_center if self.fit_intercept else 0.0
            scale = float(np.sqrt(max(variance, _SCALE_FLOOR ** 2)))
            if not self.standardize:
                scale = 1.0
            transformed = (imputed - center) / scale
            columns.append(transformed.astype(np.float64, copy=False))
            selected.append(int(index))
            centers.append(float(center))
            scales.append(float(scale))
            imputes.append(float(impute))

        if not selected:
            if explicit:
                raise ValueError(
                    "linear_residual_features did not leave any usable "
                    "numeric, non-constant columns"
                )
            self._set_inactive(
                reason="no_usable_auto_features",
                n_samples=y.shape[0],
                n_features=n_features,
                dropped_features=dropped,
                weight_sum=weight_sum,
                positive_n=positive_n,
                effective_n=effective_n,
            )
            return self

        Z = np.column_stack(columns)
        target_mean = _weighted_mean(y[positive], w_pos) if self.fit_intercept else 0.0
        y_centered = y - target_mean
        Zp = Z[positive]
        yp = y_centered[positive]
        sqrt_w = np.sqrt(w_pos)
        Zw = Zp * sqrt_w[:, None]
        yw = yp * sqrt_w
        U, singular_values, Vt = np.linalg.svd(Zw, full_matrices=False)
        rank = self._svd_rank(singular_values, positive_n, len(selected))
        rhs = U.T @ yw
        if self.alpha == 0.0:
            factors = np.zeros_like(singular_values)
            cutoff = self._svd_cutoff(singular_values, positive_n, len(selected))
            keep = singular_values > cutoff
            factors[keep] = 1.0 / singular_values[keep]
        else:
            denom = singular_values * singular_values + self.alpha * float(positive_n)
            factors = singular_values / denom
        transformed_coef = Vt.T @ (factors * rhs)
        centers = np.asarray(centers, dtype=np.float64)
        scales = np.asarray(scales, dtype=np.float64)
        coef = transformed_coef / scales
        intercept = (
            float(target_mean - np.dot(centers, coef))
            if self.fit_intercept
            else 0.0
        )

        self.active_ = True
        self.inactive_reason_ = None
        self.alpha_ = float(self.alpha)
        self.fit_intercept_ = bool(self.fit_intercept)
        self.standardize_ = bool(self.standardize)
        self.feature_indices_ = np.asarray(selected, dtype=np.int64)
        self.feature_names_ = (
            None if feature_names is None
            else np.asarray([str(feature_names[idx]) for idx in selected], dtype=object)
        )
        self.dropped_features_ = list(dropped)
        self.intercept_ = intercept
        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.transformed_coef_ = np.asarray(transformed_coef, dtype=np.float64)
        self.center_ = centers
        self.scale_ = scales
        self.impute_values_ = np.asarray(imputes, dtype=np.float64)
        self.rank_ = int(rank)
        self.singular_values_ = np.asarray(singular_values, dtype=np.float64)
        self.weight_sum_ = float(weight_sum)
        self.positive_weight_n_ = int(positive_n)
        self.effective_n_ = float(effective_n)
        self.target_mean_ = float(target_mean)
        trend = self.predict(X)
        residual = y - trend
        self.trend_train_mean_ = _weighted_mean(trend[positive], w_pos)
        self.residual_stats_ = self._residual_stats(residual, w, positive)
        return self

    def _set_inactive(
        self,
        *,
        reason,
        n_samples,
        n_features,
        dropped_features,
        weight_sum,
        positive_n,
        effective_n,
    ):
        self.active_ = False
        self.inactive_reason_ = reason
        self.alpha_ = float(self.alpha)
        self.fit_intercept_ = bool(self.fit_intercept)
        self.standardize_ = bool(self.standardize)
        self.feature_indices_ = np.empty(0, dtype=np.int64)
        self.feature_names_ = None
        self.dropped_features_ = list(dropped_features)
        self.intercept_ = 0.0
        self.coef_ = np.empty(0, dtype=np.float64)
        self.transformed_coef_ = np.empty(0, dtype=np.float64)
        self.center_ = np.empty(0, dtype=np.float64)
        self.scale_ = np.empty(0, dtype=np.float64)
        self.impute_values_ = np.empty(0, dtype=np.float64)
        self.rank_ = 0
        self.singular_values_ = np.empty(0, dtype=np.float64)
        self.weight_sum_ = float(weight_sum)
        self.positive_weight_n_ = int(positive_n)
        self.effective_n_ = float(effective_n)
        self.target_mean_ = 0.0
        self.trend_train_mean_ = 0.0
        self.residual_stats_ = {
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "positive_weight_n": int(positive_n),
        }

    @staticmethod
    def _svd_cutoff(singular_values, n_positive, p):
        if singular_values.size == 0:
            return 0.0
        return (
            max(int(n_positive), int(p))
            * np.finfo(np.float64).eps
            * float(np.max(singular_values))
        )

    @classmethod
    def _svd_rank(cls, singular_values, n_positive, p):
        return int(
            np.sum(singular_values > cls._svd_cutoff(singular_values, n_positive, p))
        )

    @staticmethod
    def _residual_stats(residual, weights, positive):
        residual = np.asarray(residual, dtype=np.float64)
        w_pos = weights[positive]
        r_pos = residual[positive]
        weighted_mean = _weighted_mean(r_pos, w_pos)
        weighted_rmse = float(np.sqrt(_weighted_mean(r_pos * r_pos, w_pos)))
        return {
            "n_samples": int(residual.shape[0]),
            "positive_weight_n": int(np.sum(positive)),
            "weighted_mean": float(weighted_mean),
            "weighted_rmse": weighted_rmse,
            "unweighted_mean": float(np.mean(residual)),
            "unweighted_rmse": float(np.sqrt(np.mean(residual * residual))),
        }

    def _selected_matrix(self, X):
        cols = []
        n_samples = n_samples_from_array_like(X, allow_empty=True)
        numeric_matrix = _numeric_matrix_or_none(X)
        for offset, index in enumerate(self.feature_indices_):
            try:
                raw_col = _coerce_numeric_column(
                    X, int(index), numeric_matrix=numeric_matrix
                )
            except ValueError as exc:
                raise ValueError(
                    "linear residual feature "
                    f"{int(index)} cannot be converted to float at predict time"
                ) from exc
            imputed = np.where(
                np.isfinite(raw_col), raw_col, self.impute_values_[offset]
            )
            cols.append(imputed.astype(np.float64, copy=False))
        return np.column_stack(cols) if cols else np.empty((n_samples, 0))

    def predict(self, X):
        if not getattr(self, "active_", False):
            return np.zeros(
                n_samples_from_array_like(X, allow_empty=True),
                dtype=np.float64,
            )
        matrix = self._selected_matrix(X)
        return self.intercept_ + matrix @ self.coef_

    def residualize(self, X, y):
        return np.asarray(y, dtype=np.float64) - self.predict(X)

    def summary(self):
        return {
            "version": 1,
            "active": bool(getattr(self, "active_", False)),
            "inactive_reason": getattr(self, "inactive_reason_", None),
            "alpha": float(getattr(self, "alpha_", self.alpha)),
            "fit_intercept": bool(getattr(self, "fit_intercept_", self.fit_intercept)),
            "standardize": bool(getattr(self, "standardize_", self.standardize)),
            "n_features": int(len(getattr(self, "feature_indices_", []))),
            "feature_indices": [
                int(idx) for idx in getattr(self, "feature_indices_", [])
            ],
            "feature_names": (
                None if getattr(self, "feature_names_", None) is None
                else [str(name) for name in self.feature_names_]
            ),
            "dropped_features": list(getattr(self, "dropped_features_", [])),
            "rank": int(getattr(self, "rank_", 0)),
            "weight_sum": float(getattr(self, "weight_sum_", 0.0)),
            "positive_weight_n": int(getattr(self, "positive_weight_n_", 0)),
            "effective_n": float(getattr(self, "effective_n_", 0.0)),
            "target_mean": float(getattr(self, "target_mean_", 0.0)),
            "trend_train_mean": float(getattr(self, "trend_train_mean_", 0.0)),
            "residual_stats": dict(getattr(self, "residual_stats_", {})),
        }

    def state_header(self):
        state = {
            "linear_residual_version": 1,
            "linear_residual_active": bool(getattr(self, "active_", False)),
            "linear_residual_alpha": float(getattr(self, "alpha_", self.alpha)),
            "linear_residual_fit_intercept": bool(
                getattr(self, "fit_intercept_", self.fit_intercept)
            ),
            "linear_residual_standardize": bool(
                getattr(self, "standardize_", self.standardize)
            ),
            "linear_residual_inactive_reason": getattr(
                self, "inactive_reason_", None
            ),
            "linear_residual_feature_names": (
                None if getattr(self, "feature_names_", None) is None
                else [str(name) for name in self.feature_names_]
            ),
            "linear_residual_dropped_features": list(
                getattr(self, "dropped_features_", [])
            ),
            "linear_residual_intercept": float(getattr(self, "intercept_", 0.0)),
            "linear_residual_rank": int(getattr(self, "rank_", 0)),
            "linear_residual_weight_sum": float(getattr(self, "weight_sum_", 0.0)),
            "linear_residual_positive_weight_n": int(
                getattr(self, "positive_weight_n_", 0)
            ),
            "linear_residual_effective_n": float(
                getattr(self, "effective_n_", 0.0)
            ),
            "linear_residual_target_mean": float(
                getattr(self, "target_mean_", 0.0)
            ),
            "linear_residual_trend_train_mean": float(
                getattr(self, "trend_train_mean_", 0.0)
            ),
            "linear_residual_residual_stats": dict(
                getattr(self, "residual_stats_", {})
            ),
            "linear_residual_prediction_mode": "additive_location",
            "linear_residual_beta_uncertainty_included": False,
        }
        return state

    def state_arrays(self):
        if not getattr(self, "active_", False):
            return {}
        return {
            "linear_residual_feature_indices": self.feature_indices_,
            "linear_residual_coef": self.coef_,
            "linear_residual_transformed_coef": self.transformed_coef_,
            "linear_residual_center": self.center_,
            "linear_residual_scale": self.scale_,
            "linear_residual_impute_values": self.impute_values_,
            "linear_residual_singular_values": self.singular_values_,
        }

    @classmethod
    def from_payload(cls, state, arrays, *, n_features=None):
        state = state or {}
        enabled = bool(state.get("linear_residual_enabled", False))
        active = bool(state.get("linear_residual_active", False))
        trend = cls(
            alpha=state.get("linear_residual_alpha", 1.0),
            fit_intercept=state.get("linear_residual_fit_intercept", True),
            standardize=state.get("linear_residual_standardize", True),
        )
        trend.enabled_ = enabled
        trend.active_ = active
        trend.inactive_reason_ = state.get("linear_residual_inactive_reason")
        trend.alpha_ = float(state.get("linear_residual_alpha", trend.alpha))
        trend.fit_intercept_ = bool(
            state.get("linear_residual_fit_intercept", trend.fit_intercept)
        )
        trend.standardize_ = bool(
            state.get("linear_residual_standardize", trend.standardize)
        )
        if not active:
            trend.feature_indices_ = np.empty(0, dtype=np.int64)
            trend.feature_names_ = None
            trend.dropped_features_ = list(
                state.get("linear_residual_dropped_features", [])
            )
            trend.intercept_ = 0.0
            trend.coef_ = np.empty(0, dtype=np.float64)
            trend.transformed_coef_ = np.empty(0, dtype=np.float64)
            trend.center_ = np.empty(0, dtype=np.float64)
            trend.scale_ = np.empty(0, dtype=np.float64)
            trend.impute_values_ = np.empty(0, dtype=np.float64)
            trend.rank_ = 0
            trend.singular_values_ = np.empty(0, dtype=np.float64)
            trend.weight_sum_ = float(state.get("linear_residual_weight_sum", 0.0))
            trend.positive_weight_n_ = int(
                state.get("linear_residual_positive_weight_n", 0)
            )
            trend.effective_n_ = float(
                state.get("linear_residual_effective_n", 0.0)
            )
            trend.target_mean_ = float(
                state.get("linear_residual_target_mean", 0.0)
            )
            trend.trend_train_mean_ = float(
                state.get("linear_residual_trend_train_mean", 0.0)
            )
            trend.residual_stats_ = dict(
                state.get("linear_residual_residual_stats", {})
            )
            return trend

        version = int(state.get("linear_residual_version", 0))
        if version != 1:
            raise ValueError(
                "invalid DarkoFit model: unsupported linear residual "
                f"version {version}"
            )
        required = (
            "linear_residual_feature_indices",
            "linear_residual_coef",
            "linear_residual_transformed_coef",
            "linear_residual_center",
            "linear_residual_scale",
            "linear_residual_impute_values",
            "linear_residual_singular_values",
        )
        missing = [name for name in required if name not in arrays]
        if missing:
            raise ValueError(
                "invalid DarkoFit model: missing linear residual arrays "
                + ", ".join(missing)
            )
        feature_indices = np.asarray(
            arrays["linear_residual_feature_indices"]
        )
        if feature_indices.ndim != 1 or not np.issubdtype(
            feature_indices.dtype, np.integer
        ):
            raise ValueError(
                "invalid DarkoFit model: linear residual feature indices "
                "must be a 1D integer array"
            )
        feature_indices = feature_indices.astype(np.int64, copy=True)
        if np.unique(feature_indices).size != feature_indices.size:
            raise ValueError(
                "invalid DarkoFit model: linear residual feature indices "
                "contain duplicates"
            )
        if n_features is not None and feature_indices.size:
            if (
                np.any(feature_indices < 0)
                or np.any(feature_indices >= int(n_features))
            ):
                raise ValueError(
                    "invalid DarkoFit model: linear residual feature "
                    "indices are out of range"
                )
        p = int(feature_indices.size)
        if p == 0:
            raise ValueError(
                "invalid DarkoFit model: active linear residual has no "
                "features"
            )
        trend.feature_indices_ = feature_indices
        trend.coef_ = _as_float_array(
            "linear residual coef", arrays["linear_residual_coef"], length=p
        )
        trend.transformed_coef_ = _as_float_array(
            "linear residual transformed coef",
            arrays["linear_residual_transformed_coef"],
            length=p,
        )
        trend.center_ = _as_float_array(
            "linear residual center", arrays["linear_residual_center"], length=p
        )
        trend.scale_ = _as_float_array(
            "linear residual scale",
            arrays["linear_residual_scale"],
            length=p,
            positive=True,
        )
        trend.impute_values_ = _as_float_array(
            "linear residual impute values",
            arrays["linear_residual_impute_values"],
            length=p,
        )
        trend.singular_values_ = _as_float_array(
            "linear residual singular values",
            arrays["linear_residual_singular_values"],
        )
        if np.any(trend.singular_values_ < 0.0):
            raise ValueError(
                "invalid DarkoFit model: linear residual singular values "
                "must be nonnegative"
            )
        feature_names = state.get("linear_residual_feature_names")
        trend.feature_names_ = (
            None if feature_names is None
            else np.asarray([str(name) for name in feature_names], dtype=object)
        )
        if trend.feature_names_ is not None and trend.feature_names_.shape != (p,):
            raise ValueError(
                "invalid DarkoFit model: linear residual feature names "
                "length does not match feature indices"
            )
        trend.dropped_features_ = list(
            state.get("linear_residual_dropped_features", [])
        )
        trend.intercept_ = float(state.get("linear_residual_intercept", 0.0))
        if not np.isfinite(trend.intercept_):
            raise ValueError(
                "invalid DarkoFit model: linear residual intercept must be finite"
            )
        trend.rank_ = int(state.get("linear_residual_rank", 0))
        if trend.rank_ < 0 or trend.rank_ > p:
            raise ValueError(
                "invalid DarkoFit model: linear residual rank is invalid"
            )
        trend.weight_sum_ = float(state.get("linear_residual_weight_sum", 0.0))
        trend.positive_weight_n_ = int(
            state.get("linear_residual_positive_weight_n", 0)
        )
        trend.effective_n_ = float(state.get("linear_residual_effective_n", 0.0))
        trend.target_mean_ = float(state.get("linear_residual_target_mean", 0.0))
        trend.trend_train_mean_ = float(
            state.get("linear_residual_trend_train_mean", 0.0)
        )
        trend.residual_stats_ = dict(
            state.get("linear_residual_residual_stats", {})
        )
        return trend
