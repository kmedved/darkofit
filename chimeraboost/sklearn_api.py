"""Scikit-learn flavored estimators: fit / predict / predict_proba."""

import math
import warnings

import numpy as np
from ._validation import (
    array_like_to_numpy,
    n_features_from_array_like,
    n_samples_from_array_like,
    normalize_random_state_seed,
    normalize_cat_features,
    validate_target_vector,
)
from .booster import (
    DistributionalBoosting,
    GradientBoosting,
    MulticlassBoosting,
    _EMITTED_DIAGNOSTIC_WARNING_CODES,
    _apply_thread_count,
    _normalize_diagnostic_warnings,
    _normalize_tree_mode,
)
from .auto_params import (
    effective_sample_size,
    is_auto_learning_rate,
    resolve_learning_rate_details,
)
from .losses import VECTOR_LOSSES
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

# Parameters that exist only on the sklearn wrappers, not on the core boosters.
_SKLEARN_ONLY = frozenset({
    "early_stopping", "validation_fraction", "validation_strategy", "refit",
    "refit_strategy", "auto_learning_rate_probe",
    "auto_learning_rate_probe_values", "auto_learning_rate_probe_iterations",
    "dist_calibration", "dist_params", "sigma_calibration",
})

_REFIT_STRATEGY_EXPONENT = {
    "best": 0.0,
    "exact": 0.0,
    "sqrt": 0.5,
    "linear": 1.0,
    "scaled": 1.0,
}

_AUTO_TREE_MODE_CANDIDATES = ("catboost", "lightgbm", "hybrid")
_SIGMA_CALIBRATION_MIN_EFFECTIVE_N = 200.0
_SIGMA_AFFINE_BOUNDS = (0.5, 2.0)
_SIGMA_CALIBRATION_Z_GUARD = 1000.0
_SIGMA_CALIBRATION_INFLUENCE_TOP_K = 5
_SIGMA_CALIBRATION_INFLUENCE_THRESHOLD = 0.5
_SIGMA_MIN = 1e-12


def _normalize_tree_mode_token(tree_mode):
    if tree_mode is None:
        return "catboost"
    return str(tree_mode).lower().replace("-", "_")


def _is_auto_tree_mode(tree_mode):
    return _normalize_tree_mode_token(tree_mode) == "auto"


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


def _normalize_sigma_calibration(calibration):
    if calibration is None or calibration is False:
        return None
    if calibration is True:
        return "scalar"
    mode = str(calibration).lower().replace("-", "_")
    if mode in {"none", "off", "false", "no"}:
        return None
    if mode in {"scalar", "scale", "sigma_scale"}:
        return "scalar"
    if mode in {"affine", "log_affine", "log_sigma_affine"}:
        return "affine"
    raise ValueError(
        "sigma_calibration must be None, False, True, 'scalar', or 'affine'"
    )


def _is_distributional_loss(loss):
    return loss in VECTOR_LOSSES


def _normalize_dist_calibration(
    dist_calibration, sigma_calibration=None, *, warn_legacy=False
):
    try:
        dist_mode = _normalize_sigma_calibration(dist_calibration)
    except ValueError:
        mode = str(dist_calibration).lower().replace("-", "_")
        if mode in {"dispersion", "alpha", "dispersion_scale"}:
            dist_mode = "dispersion"
        else:
            raise ValueError(
                "dist_calibration must be None, False, True, 'scalar', "
                "'affine', or 'dispersion'"
            )
    sigma_mode = _normalize_sigma_calibration(sigma_calibration)
    if dist_mode is not None and sigma_mode is not None and dist_mode != sigma_mode:
        raise ValueError(
            "dist_calibration and deprecated sigma_calibration specify "
            "different modes"
        )
    if sigma_mode is not None:
        if warn_legacy:
            warnings.warn(
                "sigma_calibration is deprecated; use dist_calibration instead",
                DeprecationWarning,
                stacklevel=3,
            )
        return sigma_mode
    return dist_mode


def _sigma_calibration_arrays(model, X_val, y_val, sample_weight=None):
    params = model.predict_dist(X_val)
    loss = getattr(model, "loss_", None)
    y_val = np.asarray(y_val, dtype=np.float64)
    if hasattr(loss, "scale_calibration_arrays"):
        target, mu, sigma = loss.scale_calibration_arrays(y_val, params)
    else:
        mu, sigma = params[:2]
        target = y_val
    target = np.asarray(target, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), _SIGMA_MIN)
    if sample_weight is None:
        return target, mu, sigma, None
    w = np.asarray(sample_weight, dtype=np.float64)
    positive = w > 0.0
    if not np.any(positive):
        raise ValueError("eval_sample_weight must have positive total weight")
    return target[positive], mu[positive], sigma[positive], w[positive]


def _weighted_average(values, weights=None):
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


def _weighted_sum(values, weights=None):
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        return float(np.sum(values))
    return float(np.sum(values * weights))


def _fit_scalar_sigma_scale_from_arrays(y_val, mu, sigma, sample_weight=None):
    z = np.clip(
        (y_val - mu) / sigma,
        -_SIGMA_CALIBRATION_Z_GUARD,
        _SIGMA_CALIBRATION_Z_GUARD,
    )
    z2 = z * z
    scale2 = _weighted_average(z2, sample_weight)
    return float(np.sqrt(max(scale2, _SIGMA_MIN)))


def _fit_scalar_sigma_scale(model, X_val, y_val, sample_weight=None):
    y_val, mu, sigma, w = _sigma_calibration_arrays(
        model, X_val, y_val, sample_weight
    )
    loss = getattr(model, "loss_", None)
    if getattr(loss, "name", None) == "StudentT":
        return _fit_scalar_student_t_scale_from_arrays(
            y_val, mu, sigma, loss.nu, w
        )
    return _fit_scalar_sigma_scale_from_arrays(y_val, mu, sigma, w)


def _student_t_scale_objective(y_val, mu, scale, nu, log_scale, sample_weight=None):
    scale_multiplier = float(np.exp(np.clip(log_scale, -50.0, 50.0)))
    calibrated = np.maximum(scale * scale_multiplier, _SIGMA_MIN)
    z = (y_val - mu) / calibrated
    z = np.clip(z, -1e150, 1e150)
    const = (
        0.5 * np.log(nu * np.pi)
        + math.lgamma(nu / 2.0)
        - math.lgamma((nu + 1.0) / 2.0)
    )
    nll = np.log(calibrated) + const + 0.5 * (nu + 1.0) * np.log1p(z * z / nu)
    return float(np.average(nll, weights=sample_weight))


def _fit_scalar_student_t_scale_from_arrays(y_val, mu, scale, nu, sample_weight=None):
    def objective(log_s):
        return _student_t_scale_objective(y_val, mu, scale, nu, log_s, sample_weight)

    best_log_s, _ = _golden_section_minimize(objective, -3.0, 3.0)
    return float(np.exp(np.clip(best_log_s, -50.0, 50.0)))


def _mean_calibration_arrays(model, X_val, y_val, sample_weight=None):
    params = model.predict_dist(X_val)
    mean = np.maximum(np.asarray(params[0], dtype=np.float64), _SIGMA_MIN)
    y_val = np.asarray(y_val, dtype=np.float64)
    if sample_weight is None:
        return y_val, mean, None
    w = np.asarray(sample_weight, dtype=np.float64)
    positive = w > 0.0
    if not np.any(positive):
        raise ValueError("eval_sample_weight must have positive total weight")
    return y_val[positive], mean[positive], w[positive]


def _fit_scalar_mean_calibration(model, X_val, y_val, sample_weight=None):
    loss = getattr(model, "loss_", None)
    params = model.predict_dist(X_val)
    if getattr(loss, "name", None) == "NegativeBinomial":
        y_val = np.asarray(y_val, dtype=np.float64)
        mu = np.maximum(np.asarray(params[0], dtype=np.float64), _SIGMA_MIN)
        alpha = np.asarray(params[1], dtype=np.float64)
        w = None
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            positive = sample_weight > 0.0
            if not np.any(positive):
                raise ValueError("eval_sample_weight must have positive total weight")
            y_val = y_val[positive]
            mu = mu[positive]
            alpha = alpha[positive]
            w = sample_weight[positive]

        def objective(log_scale):
            scale = float(np.exp(np.clip(log_scale, -10.0, 10.0)))
            return _negative_binomial_nll_from_params(y_val, mu * scale, alpha, w)

        best_log_scale, _ = _golden_section_minimize(objective, -5.0, 5.0)
        return {
            "scale": float(np.exp(np.clip(best_log_scale, -10.0, 10.0))),
            "mean_calibration_objective": "negative_binomial_nll",
        }

    y_val, mean, w = _mean_calibration_arrays(model, X_val, y_val, sample_weight)
    numerator = _weighted_sum(y_val, w)
    denominator = _weighted_sum(mean, w)
    scale = numerator / max(denominator, _SIGMA_MIN)
    return {
        "scale": float(max(scale, _SIGMA_MIN)),
        "numerator": float(numerator),
        "denominator": float(denominator),
        "mean_calibration_objective": "poisson_closed_form",
    }


def _negative_binomial_nll_from_params(y_val, mu, alpha, sample_weight=None):
    y_val = np.asarray(y_val, dtype=np.float64)
    mu = np.maximum(np.asarray(mu, dtype=np.float64), _SIGMA_MIN)
    alpha = np.maximum(np.asarray(alpha, dtype=np.float64), 1e-12)
    r = 1.0 / alpha
    log_r = np.log(r)
    log_r_mu = np.log(r + mu)
    lgamma_y_r = np.fromiter(
        (math.lgamma(float(yi + ri)) for yi, ri in zip(y_val, r)),
        dtype=np.float64,
        count=y_val.size,
    )
    lgamma_r = np.fromiter(
        (math.lgamma(float(ri)) for ri in r),
        dtype=np.float64,
        count=y_val.size,
    )
    lgamma_y = np.fromiter(
        (math.lgamma(float(yi) + 1.0) for yi in y_val),
        dtype=np.float64,
        count=y_val.size,
    )
    nll = (
        -lgamma_y_r
        + lgamma_r
        + lgamma_y
        - r * (log_r - log_r_mu)
        - y_val * (np.log(mu) - log_r_mu)
    )
    return float(np.average(nll, weights=sample_weight))


def _fit_dispersion_calibration(model, X_val, y_val, sample_weight=None):
    params = model.predict_dist(X_val)
    if len(params) < 2:
        raise ValueError("dispersion calibration requires a dispersion parameter")
    mu = np.asarray(params[0], dtype=np.float64)
    alpha = np.asarray(params[1], dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.float64)
    w = None
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        positive = sample_weight > 0.0
        if not np.any(positive):
            raise ValueError("eval_sample_weight must have positive total weight")
        y_val = y_val[positive]
        mu = mu[positive]
        alpha = alpha[positive]
        w = sample_weight[positive]

    def objective(log_scale):
        scale = float(np.exp(np.clip(log_scale, -10.0, 10.0)))
        return _negative_binomial_nll_from_params(y_val, mu, alpha * scale, w)

    best_log_scale, _ = _golden_section_minimize(objective, -5.0, 5.0)
    return float(np.exp(np.clip(best_log_scale, -10.0, 10.0)))


def _sigma_calibration_influence_stats(model, X_val, y_val, sample_weight=None):
    y_val, mu, sigma, w = _sigma_calibration_arrays(
        model, X_val, y_val, sample_weight
    )
    z = np.clip(
        (y_val - mu) / sigma,
        -_SIGMA_CALIBRATION_Z_GUARD,
        _SIGMA_CALIBRATION_Z_GUARD,
    )
    contribution = z * z
    if w is not None:
        contribution = contribution * w
    total = float(np.sum(contribution))
    top_k = min(_SIGMA_CALIBRATION_INFLUENCE_TOP_K, contribution.size)
    if total <= 0.0 or top_k == 0:
        fraction = 0.0
    else:
        top = np.partition(contribution, contribution.size - top_k)[-top_k:]
        fraction = float(np.sum(top) / total)
    return {
        "top_residual_count": int(top_k),
        "top_residual_contribution_fraction": fraction,
        "high_influence_warning": (
            fraction > _SIGMA_CALIBRATION_INFLUENCE_THRESHOLD
        ),
        "high_influence_threshold": _SIGMA_CALIBRATION_INFLUENCE_THRESHOLD,
        "residual_guard": _SIGMA_CALIBRATION_Z_GUARD,
    }


def _profile_affine_sigma_calibration(residual2, log_sigma, b, sample_weight=None):
    """Return profiled ``(a, objective)`` for ``exp(a + b * log_sigma)``."""
    residual2 = np.asarray(residual2, dtype=np.float64)
    log_sigma = np.asarray(log_sigma, dtype=np.float64)
    b = float(b)
    exponent = np.clip(-2.0 * b * log_sigma, -700.0, 700.0)
    scaled_residual2 = residual2 * np.exp(exponent)
    scaled_residual2 = np.minimum(
        scaled_residual2,
        _SIGMA_CALIBRATION_Z_GUARD * _SIGMA_CALIBRATION_Z_GUARD,
    )
    mean_scaled_residual2 = _weighted_average(scaled_residual2, sample_weight)
    if not np.isfinite(mean_scaled_residual2) or mean_scaled_residual2 <= 0.0:
        return float("nan"), float("inf")
    a = 0.5 * float(np.log(max(mean_scaled_residual2, _SIGMA_MIN)))
    objective = a + b * _weighted_average(log_sigma, sample_weight) + 0.5
    if not np.isfinite(objective):
        return a, float("inf")
    return a, float(objective)


def _golden_section_minimize(func, lower, upper, *, iterations=64, tol=1e-5):
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    inv_phi2 = (3.0 - np.sqrt(5.0)) / 2.0
    a = float(lower)
    b = float(upper)
    h = b - a
    if h <= tol:
        x = 0.5 * (a + b)
        return x, float(func(x))

    c = a + inv_phi2 * h
    d = a + inv_phi * h
    yc = float(func(c))
    yd = float(func(d))
    for _ in range(int(iterations)):
        if h <= tol:
            break
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = inv_phi * h
            c = a + inv_phi2 * h
            yc = float(func(c))
        else:
            a = c
            c = d
            yc = yd
            h = inv_phi * h
            d = a + inv_phi * h
            yd = float(func(d))
    x = 0.5 * (a + b)
    return x, float(func(x))


def _fit_affine_sigma_calibration(
    model, X_val, y_val, sample_weight=None, *, fold_stats=None
):
    y_val, mu, sigma, w = _sigma_calibration_arrays(
        model, X_val, y_val, sample_weight
    )
    loss = getattr(model, "loss_", None)
    if getattr(loss, "name", None) == "StudentT":
        return _fit_affine_student_t_scale_calibration_from_arrays(
            y_val, mu, sigma, loss.nu, w, fold_stats=fold_stats
        )
    scalar_scale = _fit_scalar_sigma_scale_from_arrays(y_val, mu, sigma, w)
    fallback = None
    if fold_stats is not None:
        effective_n = float(fold_stats.get("validation_effective_n", 0.0))
    else:
        effective_n = float(y_val.shape[0]) if w is None else (
            _weighted_sum(w, None) ** 2 / max(_weighted_sum(w * w, None), _SIGMA_MIN)
        )
    a = float(np.log(max(scalar_scale, _SIGMA_MIN)))
    b = 1.0
    if effective_n < _SIGMA_CALIBRATION_MIN_EFFECTIVE_N:
        fallback = "small_fold"
    else:
        residual2 = np.maximum((y_val - mu) ** 2, 0.0)
        log_sigma = np.log(np.maximum(sigma, _SIGMA_MIN))

        def objective(candidate_b):
            _, value = _profile_affine_sigma_calibration(
                residual2, log_sigma, candidate_b, w
            )
            return value

        lower, upper = _SIGMA_AFFINE_BOUNDS
        best_b, best_value = _golden_section_minimize(objective, lower, upper)
        boundary_eps = 1e-3
        if (
            not np.isfinite(best_value)
            or best_b <= lower + boundary_eps
            or best_b >= upper - boundary_eps
        ):
            fallback = "slope_bound"
        else:
            best_a, _ = _profile_affine_sigma_calibration(
                residual2, log_sigma, best_b, w
            )
            if np.isfinite(best_a):
                a = float(best_a)
                b = float(best_b)
            else:
                fallback = "non_finite_profile"
    sigma_scale = float(np.exp(np.clip(a, -700.0, 700.0)))
    return {
        "sigma_scale": sigma_scale,
        "sigma_affine_a": a,
        "sigma_affine_b": b,
        "fallback_reason": fallback,
    }


def _fit_affine_student_t_scale_calibration_from_arrays(
    y_val, mu, scale, nu, sample_weight=None, *, fold_stats=None
):
    if fold_stats is not None:
        effective_n = float(fold_stats.get("validation_effective_n", 0.0))
    else:
        effective_n = float(y_val.shape[0]) if sample_weight is None else (
            _weighted_sum(sample_weight, None) ** 2
            / max(_weighted_sum(sample_weight * sample_weight, None), _SIGMA_MIN)
        )
    scalar = _fit_scalar_student_t_scale_from_arrays(
        y_val, mu, scale, nu, sample_weight
    )
    a = float(np.log(max(scalar, _SIGMA_MIN)))
    b = 1.0
    fallback = "small_fold" if effective_n < _SIGMA_CALIBRATION_MIN_EFFECTIVE_N else None
    if fallback is None:
        log_scale = np.log(np.maximum(scale, _SIGMA_MIN))

        def objective_ab(candidate_a, candidate_b):
            calibrated_log_scale = candidate_a + candidate_b * log_scale
            calibrated = np.exp(np.clip(calibrated_log_scale, -50.0, 50.0))
            z = (y_val - mu) / np.maximum(calibrated, _SIGMA_MIN)
            z = np.clip(z, -1e150, 1e150)
            const = (
                0.5 * np.log(nu * np.pi)
                + math.lgamma(nu / 2.0)
                - math.lgamma((nu + 1.0) / 2.0)
            )
            nll = (
                calibrated_log_scale
                + const
                + 0.5 * (nu + 1.0) * np.log1p(z * z / nu)
            )
            return float(np.average(nll, weights=sample_weight))

        def objective_b(candidate_b):
            def objective_a(candidate_a):
                return objective_ab(candidate_a, candidate_b)

            best_a, value = _golden_section_minimize(objective_a, -5.0, 5.0)
            return value

        lower, upper = _SIGMA_AFFINE_BOUNDS
        best_b, best_value = _golden_section_minimize(objective_b, lower, upper)
        boundary_eps = 1e-3
        if (
            not np.isfinite(best_value)
            or best_b <= lower + boundary_eps
            or best_b >= upper - boundary_eps
        ):
            fallback = "slope_bound"
        else:
            def objective_a(candidate_a):
                return objective_ab(candidate_a, best_b)

            best_a, best_value = _golden_section_minimize(objective_a, -5.0, 5.0)
            if np.isfinite(best_value):
                a = float(best_a)
                b = float(best_b)
            else:
                fallback = "non_finite_profile"
    sigma_scale = float(np.exp(np.clip(a, -700.0, 700.0)))
    return {
        "sigma_scale": sigma_scale,
        "sigma_affine_a": a,
        "sigma_affine_b": b,
        "fallback_reason": fallback,
    }


def _sigma_calibration_fold_stats(n_samples, sample_weight=None):
    n_samples = int(n_samples)
    if sample_weight is None:
        return {
            "validation_n_samples": n_samples,
            "validation_positive_weight_n": n_samples,
            "validation_effective_n": float(n_samples),
            "small_fold_threshold": _SIGMA_CALIBRATION_MIN_EFFECTIVE_N,
            "small_fold_warning": (
                float(n_samples) < _SIGMA_CALIBRATION_MIN_EFFECTIVE_N
            ),
        }
    w = np.asarray(sample_weight, dtype=np.float64)
    positive = w > 0.0
    w_pos = w[positive]
    if w_pos.size == 0:
        n_eff = 0.0
    else:
        n_eff = float((np.sum(w_pos) ** 2) / np.sum(w_pos * w_pos))
    return {
        "validation_n_samples": n_samples,
        "validation_positive_weight_n": int(w_pos.size),
        "validation_effective_n": n_eff,
        "small_fold_threshold": _SIGMA_CALIBRATION_MIN_EFFECTIVE_N,
        "small_fold_warning": n_eff < _SIGMA_CALIBRATION_MIN_EFFECTIVE_N,
    }


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


def _reject_eval_sample_weight_without_eval_set(eval_sample_weight, eval_set):
    if eval_sample_weight is not None and eval_set is None:
        raise ValueError(
            "eval_sample_weight requires an explicit eval_set; automatic "
            "validation splits derive validation weights from sample_weight"
        )


def _validate_split_sample_weight_mass(
    sample_weight, train_idx, val_idx, stratify=None
):
    if sample_weight is None:
        return
    w = np.asarray(sample_weight, dtype=np.float64)
    train_mass = float(np.sum(w[train_idx]))
    val_mass = float(np.sum(w[val_idx]))
    if train_mass <= 0.0 or val_mass <= 0.0:
        raise ValueError(
            "automatic validation split must assign positive sample_weight "
            "mass to both training and validation sets"
        )
    if stratify is None:
        return
    labels = np.asarray(stratify)
    for cls in np.unique(labels):
        if float(np.sum(w[labels == cls])) <= 0.0:
            continue
        train_class_mass = float(np.sum(w[train_idx][labels[train_idx] == cls]))
        val_class_mass = float(np.sum(w[val_idx][labels[val_idx] == cls]))
        if train_class_mass <= 0.0 or val_class_mass <= 0.0:
            raise ValueError(
                "automatic validation split must assign positive sample_weight "
                "mass for each positive-mass class to both training and "
                "validation sets"
            )


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
    if isinstance(eval_set, (list, tuple)):
        if (
            len(eval_set) == 1
            and isinstance(eval_set[0], (list, tuple))
            and len(eval_set[0]) == 2
        ):
            eval_set = eval_set[0]
        if len(eval_set) != 2:
            raise ValueError(
                "eval_set must be a (X_val, y_val) tuple or a one-element "
                "list containing that tuple"
            )
    else:
        raise ValueError(
            "eval_set must be a (X_val, y_val) tuple or a one-element list "
            "containing that tuple"
        )
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
    X_arr = (array_like_to_numpy(X, object) if cat_features
             else array_like_to_numpy(X, np.float64))
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    return X_arr, cat_features, n_features


def _validate_feature_names(expected_names, X, *, name="X"):
    if expected_names is None:
        return
    actual_names = _feature_names_from_input(X)
    if actual_names is None:
        return
    if not np.array_equal(actual_names, expected_names):
        raise ValueError(
            f"{name} feature names must match fit feature names in the same order"
        )


def _validate_eval_set_features(eval_set, n_features, expected_feature_names=None):
    if eval_set is None:
        return None
    Xv, yv = eval_set
    actual = n_features_from_array_like(Xv, name="eval_set[0]")
    if actual != int(n_features):
        raise ValueError(
            f"eval_set[0] has {actual} features, but X has "
            f"{int(n_features)} features"
        )
    _validate_feature_names(
        expected_feature_names, Xv, name="eval_set[0]"
    )
    yv = validate_target_vector(
        yv, n_samples_from_array_like(Xv, name="eval_set[0]"),
        name="eval_set[1]",
    )
    return (Xv, yv)


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
    _validate_feature_names(
        getattr(estimator, "feature_names_in_", None),
        X,
        name="X",
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
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
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

    _validate_split_sample_weight_mass(
        sample_weight, train_idx, val_idx, stratify=stratify
    )
    return train_idx, val_idx, realized_policy


class _RefitParamsMixin:
    """Shared fitted-model metadata and full-data refit helpers."""

    def _clear_refit_selection_metadata(self):
        for name in (
            "_selection_n_total_", "_selection_n_train_",
            "_best_n_estimators_", "_best_score_", "_learning_rate_",
            "selection_model_", "refit_", "refit_n_estimators_",
            "refit_strategy_", "tree_mode_selection_",
            "selection_model_persisted_", "dist_calibration_",
            "dist_scale_", "dist_scale_source_",
            "dist_calibration_fold_stats_",
            "dist_affine_a_", "dist_affine_b_",
            "dist_calibration_fallback_reason_",
            "dist_calibration_influence_stats_",
            "dist_calibration_pooling_",
            "dist_mean_calibration_numerator_",
            "dist_mean_calibration_denominator_",
            "dist_mean_calibration_objective_",
            "selection_model_persisted_", "sigma_calibration_",
            "sigma_scale_", "sigma_scale_source_",
            "sigma_calibration_fold_stats_",
            "sigma_affine_a_", "sigma_affine_b_",
            "sigma_calibration_fallback_reason_",
            "sigma_calibration_influence_stats_",
            "sigma_calibration_pooling_",
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
        self.selection_model_persisted_ = True
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

    def _attach_tree_mode_selection_metadata(self, metadata):
        if metadata is None:
            return
        self.tree_mode_selection_ = metadata
        model = getattr(self, "model_", None)
        auto_params = getattr(model, "auto_params_", None)
        if auto_params is not None:
            auto_params["tree_mode_selection"] = metadata
            auto_params.setdefault("diagnostics", {})
            auto_params["diagnostics"]["tree_mode_selection"] = metadata

    def _validate_tree_mode_selection_request(self):
        if not _is_auto_tree_mode(self.tree_mode):
            return
        if self.ordered_boosting == "auto":
            return
        if bool(self.ordered_boosting):
            raise ValueError(
                "ordered_boosting=True cannot be combined with "
                "tree_mode='auto'; use tree_mode='catboost' for ordered-only "
                "fits or leave ordered_boosting='auto'"
            )

    def _tree_mode_candidate_kwargs(self, fit_kwargs, tree_mode):
        candidate_kwargs = dict(fit_kwargs)
        candidate_kwargs["tree_mode"] = tree_mode
        if tree_mode not in {"lightgbm", "hybrid"}:
            candidate_kwargs["num_leaves"] = None
        return candidate_kwargs

    def _tree_mode_selection_score(self, model):
        valid_history = getattr(model, "valid_history_", None)
        if valid_history is not None and len(valid_history) > 0:
            if getattr(model, "use_best_model_", False):
                return float(model.best_score_)
            return float(valid_history[-1])
        return float(model.best_score_)

    def _fit_tree_mode_auto(
        self, make_model, fit_kwargs, X, y, *, cat_features, eval_set,
        sample_weight, eval_sample_weight
    ):
        results = []
        best_model = None
        best_score = np.inf
        best_probe_metadata = None

        for tree_mode in _AUTO_TREE_MODE_CANDIDATES:
            candidate_kwargs = self._tree_mode_candidate_kwargs(
                fit_kwargs, tree_mode
            )
            probe_lr, probe_metadata = self._run_learning_rate_probe(
                make_model,
                X, y,
                cat_features=cat_features,
                eval_set=eval_set,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight,
                fit_kwargs=candidate_kwargs,
            )
            if probe_lr is not None:
                candidate_kwargs["learning_rate"] = probe_lr
            model = make_model(candidate_kwargs)
            model.fit(
                X, y, cat_features=cat_features, eval_set=eval_set,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight,
            )
            score = self._tree_mode_selection_score(model)
            results.append({
                "tree_mode": tree_mode,
                "score": score,
                "best_iteration": int(model.best_iteration_),
                "n_estimators": len(model.trees_),
                "learning_rate": float(model.lr_),
                "probe": probe_metadata,
            })
            if score < best_score:
                best_score = score
                best_model = model
                best_probe_metadata = probe_metadata

        if best_model is None:
            raise ValueError(
                "tree_mode='auto' could not select a model because all "
                "candidate scores were non-finite"
            )
        selected = getattr(best_model, "tree_mode_", None)
        metadata = {
            "enabled": True,
            "input": self.tree_mode,
            "candidates": results,
            "selected_tree_mode": selected,
            "selected_score": float(best_score),
        }
        if hasattr(best_model, "n_threads_"):
            _apply_thread_count(best_model.n_threads_)
        return best_model, best_probe_metadata, metadata

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
        if hasattr(self, "tree_mode_selection_"):
            state["tree_mode_selection"] = self.tree_mode_selection_
        if hasattr(self, "_selection_n_total_"):
            state["selection_n_total"] = self._selection_n_total_
        if hasattr(self, "_selection_n_train_"):
            state["selection_n_train"] = self._selection_n_train_
        calibration_method = getattr(
            self, "dist_calibration_",
            getattr(self, "sigma_calibration_", None),
        )
        if hasattr(self, "dist_scale_") and calibration_method is not None:
            state["dist_scale"] = float(self.dist_scale_)
            state["dist_calibration"] = calibration_method
            state["dist_scale_source"] = getattr(
                self, "dist_scale_source_", None
            )
            if hasattr(self, "dist_affine_a_"):
                state["dist_affine_a"] = float(self.dist_affine_a_)
            if hasattr(self, "dist_affine_b_"):
                state["dist_affine_b"] = float(self.dist_affine_b_)
            fallback_reason = getattr(
                self, "dist_calibration_fallback_reason_", None
            )
            if fallback_reason is not None:
                state["dist_calibration_fallback_reason"] = fallback_reason
            if hasattr(self, "dist_mean_calibration_numerator_"):
                state["dist_mean_calibration_numerator"] = float(
                    self.dist_mean_calibration_numerator_
                )
            if hasattr(self, "dist_mean_calibration_denominator_"):
                state["dist_mean_calibration_denominator"] = float(
                    self.dist_mean_calibration_denominator_
                )
            if hasattr(self, "dist_mean_calibration_objective_"):
                state["dist_mean_calibration_objective"] = (
                    self.dist_mean_calibration_objective_
                )
            # Backward-compatible Gaussian aliases for one release.
            state["sigma_scale"] = float(self.dist_scale_)
            state["sigma_calibration"] = calibration_method
            state["sigma_scale_source"] = getattr(
                self, "dist_scale_source_", None
            )
            if hasattr(self, "dist_affine_a_"):
                state["sigma_affine_a"] = float(self.dist_affine_a_)
            if hasattr(self, "dist_affine_b_"):
                state["sigma_affine_b"] = float(self.dist_affine_b_)
            if fallback_reason is not None:
                state["sigma_calibration_fallback_reason"] = fallback_reason
        return state

    def _wrapper_params_header(self):
        params = self.get_params()
        random_state = params.get("random_state")
        if random_state is not None:
            fit_seed = getattr(
                getattr(self, "model_", None), "_fit_random_state_seed_", None
            )
            params["random_state"] = (
                int(fit_seed)
                if fit_seed is not None
                else normalize_random_state_seed(random_state)
            )
        return params

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
        if "tree_mode_selection" in state:
            self.tree_mode_selection_ = state["tree_mode_selection"]
        if self.refit_ and state.get("selection_model_persisted") is False:
            self.selection_model_ = None
            self.selection_model_persisted_ = False
        if "selection_n_total" in state:
            self._selection_n_total_ = int(state["selection_n_total"])
        if "selection_n_train" in state:
            self._selection_n_train_ = int(state["selection_n_train"])
        if "dist_scale" in state or "sigma_scale" in state:
            scale_key = "dist_scale" if "dist_scale" in state else "sigma_scale"
            calibration = state.get(
                "dist_calibration", state.get("sigma_calibration")
            )
            source = state.get(
                "dist_scale_source", state.get("sigma_scale_source")
            )
            self.dist_scale_ = float(state[scale_key])
            self.dist_calibration_ = calibration
            self.dist_scale_source_ = source
            self.sigma_scale_ = self.dist_scale_
            self.sigma_calibration_ = calibration
            self.sigma_scale_source_ = source
            if "dist_affine_a" in state or "sigma_affine_a" in state:
                self.dist_affine_a_ = float(
                    state.get("dist_affine_a", state.get("sigma_affine_a"))
                )
                self.sigma_affine_a_ = self.dist_affine_a_
            if "dist_affine_b" in state or "sigma_affine_b" in state:
                self.dist_affine_b_ = float(
                    state.get("dist_affine_b", state.get("sigma_affine_b"))
                )
                self.sigma_affine_b_ = self.dist_affine_b_
            fallback_reason = state.get(
                "dist_calibration_fallback_reason",
                state.get("sigma_calibration_fallback_reason"),
            )
            if fallback_reason is not None:
                self.dist_calibration_fallback_reason_ = fallback_reason
                self.sigma_calibration_fallback_reason_ = fallback_reason
            if "dist_mean_calibration_numerator" in state:
                self.dist_mean_calibration_numerator_ = float(
                    state["dist_mean_calibration_numerator"]
                )
            if "dist_mean_calibration_denominator" in state:
                self.dist_mean_calibration_denominator_ = float(
                    state["dist_mean_calibration_denominator"]
                )
            if "dist_mean_calibration_objective" in state:
                self.dist_mean_calibration_objective_ = str(
                    state["dist_mean_calibration_objective"]
                )

    def _attach_dist_calibration_metadata(self, *, emit_warning=True):
        if not hasattr(self, "dist_scale_"):
            return
        model = getattr(self, "model_", None)
        auto_params = getattr(model, "auto_params_", None)
        if auto_params is None:
            return
        metadata = {
            "method": getattr(self, "dist_calibration_", None),
            "dist_scale": float(self.dist_scale_),
            "sigma_scale": float(self.dist_scale_),
            "source": getattr(self, "dist_scale_source_", None),
        }
        if hasattr(self, "dist_affine_a_"):
            metadata["dist_affine_a"] = float(self.dist_affine_a_)
            metadata["sigma_affine_a"] = float(self.dist_affine_a_)
        if hasattr(self, "dist_affine_b_"):
            metadata["dist_affine_b"] = float(self.dist_affine_b_)
            metadata["sigma_affine_b"] = float(self.dist_affine_b_)
        if hasattr(self, "dist_mean_calibration_numerator_"):
            metadata["mean_calibration_numerator"] = float(
                self.dist_mean_calibration_numerator_
            )
        if hasattr(self, "dist_mean_calibration_denominator_"):
            metadata["mean_calibration_denominator"] = float(
                self.dist_mean_calibration_denominator_
            )
        if hasattr(self, "dist_mean_calibration_objective_"):
            metadata["mean_calibration_objective"] = (
                self.dist_mean_calibration_objective_
            )
        fallback_reason = getattr(
            self, "dist_calibration_fallback_reason_", None
        )
        if fallback_reason is not None:
            metadata["fallback_reason"] = fallback_reason
        pooling = getattr(self, "dist_calibration_pooling_", None)
        if pooling is not None:
            metadata["pooling"] = pooling
        fold_stats = getattr(self, "dist_calibration_fold_stats_", None)
        if fold_stats is not None:
            metadata.update(fold_stats)
        influence_stats = getattr(
            self, "dist_calibration_influence_stats_", None
        )
        if influence_stats is not None:
            metadata.update(influence_stats)
        auto_params["dist_calibration"] = metadata
        auto_params["sigma_calibration"] = metadata
        auto_params.setdefault("diagnostics", {})
        diagnostics = auto_params["diagnostics"]
        diagnostics["dist_calibration"] = metadata
        diagnostics["sigma_calibration"] = metadata
        warnings_list = diagnostics.setdefault("warnings", [])
        warning_codes = {
            warning.get("code")
            for warning in warnings_list
        }
        if metadata.get("small_fold_warning"):
            warning_record = {
                "code": "small_sigma_calibration_fold",
                "message": (
                    "ChimeraBoost "
                    f"sigma_calibration={metadata.get('method')!r} was "
                    "estimated "
                    "from a small validation fold "
                    f"(effective n={metadata['validation_effective_n']:.1f} "
                    f"< {_SIGMA_CALIBRATION_MIN_EFFECTIVE_N:.0f}); the sigma "
                    "scale may be noisy."
                ),
            }
            if warning_record["code"] not in warning_codes:
                warnings_list.append(warning_record)
                warning_codes.add(warning_record["code"])
            if emit_warning:
                self._emit_wrapper_diagnostic_warning(
                    diagnostics, warning_record
                )
        if metadata.get("high_influence_warning"):
            warning_record = {
                "code": "high_influence_sigma_calibration_fold",
                "message": (
                    "ChimeraBoost sigma calibration is dominated by the "
                    f"largest {metadata['top_residual_count']} validation "
                    "residual contributions "
                    f"({metadata['top_residual_contribution_fraction']:.1%} "
                    "of weighted z^2); inspect outliers before using sigma "
                    "as calibrated observation noise."
                ),
            }
            if warning_record["code"] not in warning_codes:
                warnings_list.append(warning_record)
                warning_codes.add(warning_record["code"])
            if emit_warning:
                self._emit_wrapper_diagnostic_warning(
                    diagnostics, warning_record
                )

    def _attach_sigma_calibration_metadata(self, *, emit_warning=True):
        self._attach_dist_calibration_metadata(emit_warning=emit_warning)

    def _emit_wrapper_diagnostic_warning(self, diagnostics, warning_record):
        policy = _normalize_diagnostic_warnings(
            getattr(self, "diagnostic_warnings", "once")
        )
        diagnostics["runtime_warning_policy"] = policy
        emitted = diagnostics.setdefault("runtime_warnings_emitted", [])
        if policy == "never":
            return
        code = warning_record.get("code")
        if (
            policy == "once"
            and code in _EMITTED_DIAGNOSTIC_WARNING_CODES
        ):
            return
        warnings.warn(warning_record["message"], RuntimeWarning, stacklevel=3)
        if code is not None:
            emitted.append(code)
            if policy == "once":
                _EMITTED_DIAGNOSTIC_WARNING_CODES.add(code)

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
        loss_name = getattr(context_model, "loss_name", None)
        if loss_name is None and hasattr(context_model, "loss_"):
            loss_name = getattr(context_model.loss_, "name", None)
        if loss_name is None:
            loss_name = "RMSE"
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
            score = self._tree_mode_selection_score(model)
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
        used with early stopping.  For distributional calibration, the returned
        params disable ``dist_calibration``/``sigma_calibration`` because the frozen map is
        fitted-state metadata, not a constructor parameter that can be
        recomputed during a validation-free full-data refit.

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
        selected_tree_mode = getattr(self.model_, "tree_mode_", None)
        if selected_tree_mode is not None:
            params["tree_mode"] = selected_tree_mode
        params["early_stopping"] = False
        params["early_stopping_rounds"] = None
        if (
            params.get("loss") in VECTOR_LOSSES
            and _normalize_dist_calibration(
                params.get("dist_calibration"),
                params.get("sigma_calibration"),
            ) is not None
        ):
            params["dist_calibration"] = None
            params["sigma_calibration"] = None
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

    loss: "RMSE" (default), "MAE", "Quantile", "Gaussian", "LogNormal",
    "StudentT", "Poisson", or "NegativeBinomial". For "Quantile" pass the
    level via `alpha` (e.g. alpha=0.9 for the 90th-percentile predictor).
    Distributional losses use shared vector-valued leaf-wise trees.

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
                 eval_metric=None, loss="RMSE", alpha=0.5, min_child_weight=1.0,
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
                 dist_calibration=None, dist_params=None,
                 sigma_calibration=None,
                 diagnostic_warnings="once",
                 auto_learning_rate_probe=False,
                 auto_learning_rate_probe_values=None,
                 auto_learning_rate_probe_iterations=80,
                 histogram_dtype="float64",
                 leaf_dtype="int64",
                 ts_permutations=1,
                 target_ordered_cat_codes="off",
                 rho_learning_rate_multiplier=1.0,
                 rho_l2_leaf_reg_multiplier=1.0):
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
        self.eval_metric = eval_metric
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
        self.dist_calibration = dist_calibration
        self.dist_params = dist_params
        self.sigma_calibration = sigma_calibration
        self.diagnostic_warnings = diagnostic_warnings
        self.histogram_dtype = histogram_dtype
        self.leaf_dtype = leaf_dtype
        self.ts_permutations = ts_permutations
        self.target_ordered_cat_codes = target_ordered_cat_codes
        self.rho_learning_rate_multiplier = rho_learning_rate_multiplier
        self.rho_l2_leaf_reg_multiplier = rho_l2_leaf_reg_multiplier
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
        eval_set = _validate_eval_set_features(
            eval_set, n_features,
            expected_feature_names=_feature_names_from_input(X_input),
        )
        y = validate_target_vector(y, X.shape[0], dtype=np.float64)
        sample_weight = _validate_wrapper_sample_weight(
            sample_weight, X.shape[0]
        )
        _reject_eval_sample_weight_without_eval_set(eval_sample_weight, eval_set)
        X_full, y_full = X, y
        sample_weight_full = sample_weight
        explicit_eval_set = eval_set is not None
        validation_strategy_ = _normalize_validation_strategy(
            self.validation_strategy
        )
        fit_random_state = normalize_random_state_seed(self.random_state)
        validation_fraction_resolved = None
        realized_validation_policy = "none"
        split_train_n = X.shape[0]
        split_eval_n = None

        self._clear_refit_selection_metadata()
        tree_mode_auto = _is_auto_tree_mode(self.tree_mode)
        distributional_loss = _is_distributional_loss(self.loss)
        dist_calibration_ = _normalize_dist_calibration(
            self.dist_calibration,
            self.sigma_calibration,
            warn_legacy=self.sigma_calibration is not None,
        )
        if (
            not distributional_loss
            and self.eval_metric not in {None, "auto", "loss"}
        ):
            raise ValueError(
                "eval_metric is only configurable for distributional losses"
            )
        if not distributional_loss and dist_calibration_ is not None:
            calibration_name = (
                "sigma_calibration"
                if self.sigma_calibration is not None and self.dist_calibration is None
                else "dist_calibration"
            )
            if calibration_name == "sigma_calibration":
                raise ValueError(
                    "sigma_calibration is only supported for loss='Gaussian' "
                    "or other distributional losses"
                )
            raise ValueError(
                f"{calibration_name} is only supported for distributional losses"
            )
        if distributional_loss:
            if self.alpha != 0.5:
                raise ValueError(
                    "alpha is only used with loss='Quantile'; leave alpha=0.5 "
                    f"for loss={self.loss!r}"
                )
            if tree_mode_auto:
                raise ValueError(
                    f"loss={self.loss!r} requires tree_mode='lightgbm'; "
                    "tree_mode='auto' is not supported in v1"
                )
            loss_cls = VECTOR_LOSSES[self.loss]
            targets = tuple(getattr(loss_cls, "calibration_targets", ()))
            if dist_calibration_ == "affine" and "scale" not in targets:
                raise ValueError(
                    f"dist_calibration='affine' is not supported for "
                    f"loss={self.loss!r}"
                )
            if dist_calibration_ == "scalar" and not targets:
                raise ValueError(
                    f"dist_calibration='scalar' is not supported for "
                    f"loss={self.loss!r}"
                )
            if dist_calibration_ == "dispersion" and "dispersion" not in targets:
                raise ValueError(
                    f"dist_calibration='dispersion' is not supported for "
                    f"loss={self.loss!r}"
                )
        elif self.dist_params not in (None, {}):
            raise ValueError("dist_params is only supported for distributional losses")
        self._validate_tree_mode_selection_request()
        if self.refit:
            self._refit_strategy_exponent(self.refit_strategy)
        es_active = _should_early_stop(self.early_stopping)
        if (
            (es_active or tree_mode_auto)
            and eval_set is None
            and groups is not None
            and validation_strategy_ == "weighted_stratified"
        ):
            raise ValueError(
                "validation_strategy='weighted_stratified' is only supported "
                "for ungrouped regression automatic validation splits"
            )
        if (es_active or tree_mode_auto) and eval_set is None:
            if eval_sample_weight is not None:
                raise ValueError(
                    "eval_sample_weight requires an explicit eval_set; "
                    "automatic validation splits derive validation weights "
                    "from sample_weight"
                )
            n_total = X.shape[0]
            validation_fraction_resolved = _resolve_validation_fraction(
                self.validation_fraction, sample_weight, n_total
            )
            train_idx, val_idx, realized_validation_policy = _make_eval_split(
                X, y, validation_fraction_resolved, fit_random_state,
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

        if dist_calibration_ is not None and eval_set is None:
            calibration_name = (
                "sigma_calibration"
                if self.sigma_calibration is not None and self.dist_calibration is None
                else "dist_calibration"
            )
            raise ValueError(
                f"{calibration_name}={dist_calibration_!r} requires a "
                "validation set; pass eval_set or set early_stopping=True to "
                "create an automatic validation split"
            )

        es_rounds = self.early_stopping_rounds
        if es_active and es_rounds is None:
            es_rounds = "auto"
        selection_active = (
            eval_set is not None
            and (es_rounds is not None or self.use_best_model or tree_mode_auto)
        )
        if self.refit and selection_active:
            self._validate_refit_strategy_for_fit(self.refit_strategy)

        loss_kwargs = {"alpha": self.alpha} if self.loss == "Quantile" else {}
        dist_kwargs = dict(self.dist_params or {})
        kw = {k: v for k, v in self.get_params().items()
              if k not in {"loss", "alpha"} | _SKLEARN_ONLY}
        kw["early_stopping_rounds"] = es_rounds
        kw["random_state"] = fit_random_state
        preprocessing_cache = (
            {} if (tree_mode_auto or self.auto_learning_rate_probe) else None
        )

        def make_model(model_kw):
            if distributional_loss:
                tree_mode = model_kw.get("tree_mode", self.tree_mode)
                if _normalize_tree_mode(tree_mode) != "lightgbm":
                    raise ValueError(
                        f"loss={self.loss!r} requires tree_mode='lightgbm'; got "
                        f"tree_mode={tree_mode!r}. Distributional regression "
                        "uses shared vector-valued leaf-wise trees."
                    )
                model = DistributionalBoosting(
                    loss=self.loss, loss_kwargs=dist_kwargs, **model_kw
                )
            else:
                model = GradientBoosting(
                    loss=self.loss, loss_kwargs=loss_kwargs, **model_kw
                )
            if preprocessing_cache is not None:
                model._preprocessing_cache = preprocessing_cache
            return model

        tree_mode_selection_metadata = None
        if tree_mode_auto:
            model, probe_metadata, tree_mode_selection_metadata = (
                self._fit_tree_mode_auto(
                    make_model, kw, X, y,
                    cat_features=cat_features,
                    eval_set=eval_set,
                    sample_weight=sample_weight,
                    eval_sample_weight=eval_sample_weight,
                )
            )
        else:
            probe_lr, probe_metadata = self._run_learning_rate_probe(
                make_model,
                X, y,
                cat_features=cat_features,
                eval_set=eval_set,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight,
                fit_kwargs=kw,
            )
            if probe_lr is not None:
                kw["learning_rate"] = probe_lr
            model = make_model(kw)
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
        elif tree_mode_auto and not es_active:
            split_source = "automatic_tree_mode_selection"
        elif es_active:
            split_source = "automatic"
        else:
            split_source = "none"
        validation_metadata = {
            "source": split_source,
            "validation_fraction_input": self.validation_fraction,
            "validation_fraction_resolved": validation_fraction_resolved,
            "validation_fraction_realized": (
                None if split_eval_n is None
                else float(split_eval_n) / max(1, int(split_train_n))
            ),
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
        self._attach_tree_mode_selection_metadata(tree_mode_selection_metadata)
        selection_model = self.model_
        self._record_selection_result(selection_model)
        if distributional_loss:
            self.dist_calibration_ = dist_calibration_
            self.sigma_calibration_ = dist_calibration_
            self.dist_scale_ = 1.0
            self.sigma_scale_ = 1.0
            self.dist_scale_source_ = "none"
            self.sigma_scale_source_ = "none"
            if dist_calibration_ is not None:
                X_cal, y_cal = eval_set
                self.dist_scale_source_ = "selection_validation"
                self.sigma_scale_source_ = "selection_validation"
                fold_stats = (
                    _sigma_calibration_fold_stats(
                        len(y_cal), eval_sample_weight
                    )
                )
                self.dist_calibration_fold_stats_ = fold_stats
                self.sigma_calibration_fold_stats_ = fold_stats
                selection_targets = tuple(
                    getattr(selection_model.loss_, "calibration_targets", ())
                )
                if "scale" in selection_targets:
                    influence_stats = (
                        _sigma_calibration_influence_stats(
                            selection_model, X_cal, y_cal, eval_sample_weight
                        )
                    )
                    self.dist_calibration_influence_stats_ = influence_stats
                    self.sigma_calibration_influence_stats_ = influence_stats
                    if dist_calibration_ == "scalar":
                        self.dist_scale_ = _fit_scalar_sigma_scale(
                            selection_model, X_cal, y_cal, eval_sample_weight
                        )
                        self.sigma_scale_ = self.dist_scale_
                    elif dist_calibration_ == "affine":
                        calibration = _fit_affine_sigma_calibration(
                            selection_model, X_cal, y_cal, eval_sample_weight,
                            fold_stats=self.sigma_calibration_fold_stats_,
                        )
                        self.dist_scale_ = calibration["sigma_scale"]
                        self.sigma_scale_ = self.dist_scale_
                        self.dist_affine_a_ = calibration["sigma_affine_a"]
                        self.dist_affine_b_ = calibration["sigma_affine_b"]
                        self.sigma_affine_a_ = self.dist_affine_a_
                        self.sigma_affine_b_ = self.dist_affine_b_
                        fallback_reason = calibration.get("fallback_reason")
                        if fallback_reason is not None:
                            self.dist_calibration_fallback_reason_ = (
                                fallback_reason
                            )
                            self.sigma_calibration_fallback_reason_ = (
                                fallback_reason
                            )
                elif (
                    dist_calibration_ == "dispersion"
                    and "dispersion" in selection_targets
                ):
                    self.dist_scale_ = _fit_dispersion_calibration(
                        selection_model, X_cal, y_cal, eval_sample_weight
                    )
                    self.sigma_scale_ = self.dist_scale_
                elif "mean" in selection_targets:
                    if dist_calibration_ != "scalar":
                        raise ValueError(
                            f"dist_calibration={dist_calibration_!r} is not "
                            f"supported for loss={self.loss!r}"
                        )
                    calibration = _fit_scalar_mean_calibration(
                        selection_model, X_cal, y_cal, eval_sample_weight
                    )
                    self.dist_scale_ = calibration["scale"]
                    self.sigma_scale_ = self.dist_scale_
                    if "numerator" in calibration:
                        self.dist_mean_calibration_numerator_ = calibration[
                            "numerator"
                        ]
                    if "denominator" in calibration:
                        self.dist_mean_calibration_denominator_ = calibration[
                            "denominator"
                        ]
                    self.dist_mean_calibration_objective_ = calibration[
                        "mean_calibration_objective"
                    ]
            self._attach_dist_calibration_metadata(
                emit_warning=not (self.refit and selection_active)
            )

        if self.refit and selection_active:
            refit_kw = self._refit_params_for_booster(self.refit_strategy)
            refit_model = make_model(refit_kw)
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
            self._attach_tree_mode_selection_metadata(tree_mode_selection_metadata)
            self._attach_dist_calibration_metadata()
        return self

    def predict(self, X):
        X = _check_predict_input(self, X)
        raw = self.model_.predict_raw(X)
        if _is_distributional_loss(self.loss):
            loss = self.model_.loss_
            if self._active_dist_calibration() is not None and hasattr(
                loss, "mean_from_params"
            ):
                return loss.mean_from_params(
                    *self._calibrated_params_from_raw(raw)
                )
            return loss.mean_from_raw(raw)
        return raw

    def staged_predict(self, X):
        """Yield the prediction after each successive tree."""
        X = _check_predict_input(self, X)
        if _is_distributional_loss(self.loss):
            loss = self.model_.loss_
            for raw in self.model_.staged_predict_raw(X):
                if self._active_dist_calibration() is not None and hasattr(
                    loss, "mean_from_params"
                ):
                    yield loss.mean_from_params(
                        *self._calibrated_params_from_raw(raw)
                    )
                else:
                    yield loss.mean_from_raw(raw)
        else:
            yield from self.model_.staged_predict_raw(X)

    def _require_distributional(self, method_name, capability=None):
        if not _is_distributional_loss(self.loss):
            raise ValueError(
                f"{method_name}() requires a distributional loss; this model was "
                f"fit with loss={self.loss!r}"
            )
        loss = getattr(self.model_, "loss_", None)
        if capability == "interval" and not getattr(loss, "interval_support", False):
            raise NotImplementedError(
                f"{method_name}() is not implemented for loss={self.loss!r}"
            )
        if capability == "sample" and not getattr(loss, "sample_support", False):
            raise NotImplementedError(
                f"{method_name}() is not implemented for loss={self.loss!r}"
            )
        return loss

    def _require_gaussian(self, method_name):
        if self.loss != "Gaussian":
            raise ValueError(
                f"{method_name}() requires loss='Gaussian'; this model was "
                f"fit with loss={self.loss!r}"
            )

    def _calibrated_params_from_raw(self, raw):
        loss = self.model_.loss_
        params = list(loss.params_from_raw(raw))
        method = self._active_dist_calibration()
        if method is None:
            return tuple(params)

        targets = tuple(getattr(loss, "calibration_targets", ()))
        if "scale" in targets:
            idx = int(getattr(loss, "scale_param_index", 1))
            scale_values = np.asarray(params[idx], dtype=np.float64)
            if method == "affine":
                a = getattr(
                    self, "dist_affine_a_",
                    getattr(self, "sigma_affine_a_", None),
                )
                b = getattr(
                    self, "dist_affine_b_",
                    getattr(self, "sigma_affine_b_", None),
                )
                if a is not None and b is not None:
                    log_scale = np.log(np.maximum(scale_values, _SIGMA_MIN))
                    params[idx] = np.exp(
                        np.clip(float(a) + float(b) * log_scale, -700.0, 700.0)
                    )
            else:
                scale = float(
                    getattr(self, "dist_scale_", getattr(self, "sigma_scale_", 1.0))
                )
                if scale != 1.0:
                    params[idx] = scale_values * scale
        elif "mean" in targets and method == "scalar":
            idx = int(getattr(loss, "mean_param_index", 0))
            scale = float(getattr(self, "dist_scale_", 1.0))
            if scale != 1.0:
                params[idx] = np.asarray(params[idx], dtype=np.float64) * scale
        elif "dispersion" in targets and method == "dispersion":
            idx = int(getattr(loss, "dispersion_param_index", 1))
            scale = float(getattr(self, "dist_scale_", 1.0))
            if scale != 1.0:
                params[idx] = np.asarray(params[idx], dtype=np.float64) * scale
        return tuple(params)

    def _active_dist_calibration(self):
        return getattr(
            self, "dist_calibration_",
            getattr(self, "sigma_calibration_", None),
        )

    def _predict_dist_checked(self, X, method_name):
        X = _check_predict_input(self, X)
        self._require_distributional(method_name)
        raw = self.model_.predict_raw(X)
        return self._calibrated_params_from_raw(raw)

    def predict_dist(self, X):
        """Return distribution parameters for a fitted distributional model."""
        return self._predict_dist_checked(X, "predict_dist")

    def predict_variance(self, X):
        """Return predictive variance for a fitted distributional model."""
        X = _check_predict_input(self, X)
        loss = self._require_distributional("predict_variance")
        raw = self.model_.predict_raw(X)
        if self._active_dist_calibration() is None:
            return loss.variance_from_raw(raw)
        params = self._calibrated_params_from_raw(raw)
        if hasattr(loss, "variance_from_params"):
            return loss.variance_from_params(*params)
        if self.loss == "Gaussian":
            return params[1] * params[1]
        raise NotImplementedError(
            f"predict_variance() with calibration is not implemented for "
            f"loss={self.loss!r}"
        )

    def predict_interval(self, X, alpha=0.1):
        """Return central prediction interval bounds."""
        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        X = _check_predict_input(self, X)
        loss = self._require_distributional(
            "predict_interval", capability="interval"
        )
        raw = self.model_.predict_raw(X)
        if self._active_dist_calibration() is None:
            return loss.interval_from_raw(raw, alpha)
        params = self._calibrated_params_from_raw(raw)
        if hasattr(loss, "interval_from_params"):
            return loss.interval_from_params(*params, alpha)
        raise NotImplementedError(
            f"predict_interval() with calibration is not implemented for "
            f"loss={self.loss!r}"
        )

    def sample(self, X, n_samples=1, random_state=None):
        """Draw samples from the fitted predictive distribution."""
        n_samples = int(n_samples)
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        X = _check_predict_input(self, X)
        loss = self._require_distributional("sample", capability="sample")
        raw = self.model_.predict_raw(X)
        rng = np.random.default_rng(random_state)
        if self._active_dist_calibration() is None:
            return loss.sample_from_raw(raw, rng, n_samples)
        params = self._calibrated_params_from_raw(raw)
        if hasattr(loss, "sample_from_params"):
            return loss.sample_from_params(*params, rng, n_samples)
        raise NotImplementedError(
            f"sample() with calibration is not implemented for loss={self.loss!r}"
        )

    def save_model(self, path):
        """Serialize the fitted model to a single ``.npz`` file."""
        check_is_fitted(self, "model_")
        from .serialization import save_booster
        save_booster(
            self.model_, path,
            wrapper_header={"wrapper_class": type(self).__name__,
                            "params": self._wrapper_params_header(),
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
        if isinstance(booster, DistributionalBoosting):
            saved_loss = params.get("loss")
            if saved_loss is not None and saved_loss != booster.loss_name:
                raise ValueError(
                    "invalid ChimeraBoost model: wrapper loss does not match "
                    "the loaded distributional booster"
                )
        known = est.get_params()
        est.set_params(**{k: v for k, v in params.items() if k in known})
        est.model_ = booster
        if isinstance(booster, DistributionalBoosting):
            est.loss = booster.loss_name
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
                 auto_learning_rate_probe_iterations=80,
                 histogram_dtype="float64",
                 leaf_dtype="int64",
                 ts_permutations=1,
                 target_ordered_cat_codes="off"):
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
        self.histogram_dtype = histogram_dtype
        self.leaf_dtype = leaf_dtype
        self.ts_permutations = ts_permutations
        self.target_ordered_cat_codes = target_ordered_cat_codes
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
        eval_set = _validate_eval_set_features(
            eval_set, n_features,
            expected_feature_names=_feature_names_from_input(X_input),
        )
        y = validate_target_vector(y, X.shape[0])
        target_type = type_of_target(y)
        if target_type not in {"binary", "multiclass"}:
            raise ValueError(f"Unknown label type: {target_type}")
        classes = np.unique(y)
        n_classes = classes.size
        if n_classes < 2:
            raise ValueError("Need at least 2 classes.")
        sample_weight = _validate_wrapper_sample_weight(
            sample_weight, X.shape[0]
        )
        _reject_eval_sample_weight_without_eval_set(eval_sample_weight, eval_set)
        X_full, y_full = X, y
        sample_weight_full = sample_weight
        explicit_eval_set = eval_set is not None
        validation_strategy_ = _normalize_validation_strategy(
            self.validation_strategy
        )
        fit_random_state = normalize_random_state_seed(self.random_state)
        validation_fraction_resolved = None
        realized_validation_policy = "none"
        split_train_n = X.shape[0]
        split_eval_n = None

        self._clear_refit_selection_metadata()
        tree_mode_auto = _is_auto_tree_mode(self.tree_mode)
        self._validate_tree_mode_selection_request()
        if self.refit:
            self._refit_strategy_exponent(self.refit_strategy)
        es_active = _should_early_stop(self.early_stopping)
        if (
            (es_active or tree_mode_auto)
            and eval_set is None
            and validation_strategy_ == "weighted_stratified"
        ):
            raise ValueError(
                "validation_strategy='weighted_stratified' is only supported "
                "for regression automatic validation splits"
            )
        if (es_active or tree_mode_auto) and eval_set is None:
            if eval_sample_weight is not None:
                raise ValueError(
                    "eval_sample_weight requires an explicit eval_set; "
                    "automatic validation splits derive validation weights "
                    "from sample_weight"
                )
            n_total = X.shape[0]
            validation_fraction_resolved = _resolve_validation_fraction(
                self.validation_fraction, sample_weight, n_total
            )
            train_idx, val_idx, realized_validation_policy = _make_eval_split(
                X, y, validation_fraction_resolved, fit_random_state,
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
            eval_set is not None
            and (es_rounds is not None or self.use_best_model or tree_mode_auto)
        )
        if self.refit and selection_active:
            self._validate_refit_strategy_for_fit(self.refit_strategy)

        kw = {k: v for k, v in self.get_params().items()
              if k not in _SKLEARN_ONLY}
        kw["early_stopping_rounds"] = es_rounds
        kw["random_state"] = fit_random_state
        probe_metadata = None
        tree_mode_selection_metadata = None

        if n_classes == 2:
            multiclass = False
            y01 = (y == classes[1]).astype(np.float64)
            if eval_set is not None:
                Xv, yv = eval_set
                if np.any(~np.isin(np.asarray(yv), classes)):
                    raise ValueError("eval_set contains labels not present in training data")
                eval_set = (Xv, (np.asarray(yv) == classes[1]).astype(np.float64))
            preprocessing_cache = (
                {} if (tree_mode_auto or self.auto_learning_rate_probe) else None
            )

            def make_model(model_kw):
                model = GradientBoosting(loss="Logloss", **model_kw)
                if preprocessing_cache is not None:
                    model._preprocessing_cache = preprocessing_cache
                return model

            if tree_mode_auto:
                model, probe_metadata, tree_mode_selection_metadata = (
                    self._fit_tree_mode_auto(
                        make_model, kw, X, y01,
                        cat_features=cat_features,
                        eval_set=eval_set,
                        sample_weight=sample_weight,
                        eval_sample_weight=eval_sample_weight,
                    )
                )
            else:
                probe_lr, probe_metadata = self._run_learning_rate_probe(
                    make_model,
                    X, y01,
                    cat_features=cat_features,
                    eval_set=eval_set,
                    sample_weight=sample_weight,
                    eval_sample_weight=eval_sample_weight,
                    fit_kwargs=kw,
                )
                if probe_lr is not None:
                    kw["learning_rate"] = probe_lr
                model = make_model(kw)
                model.fit(
                    X, y01, cat_features=cat_features, eval_set=eval_set,
                    sample_weight=sample_weight,
                    eval_sample_weight=eval_sample_weight,
                )
        else:
            multiclass = True
            preprocessing_cache = (
                {} if (tree_mode_auto or self.auto_learning_rate_probe) else None
            )

            def make_model(model_kw):
                model = MulticlassBoosting(**model_kw)
                if preprocessing_cache is not None:
                    model._preprocessing_cache = preprocessing_cache
                return model

            if tree_mode_auto:
                model, probe_metadata, tree_mode_selection_metadata = (
                    self._fit_tree_mode_auto(
                        make_model, kw, X, y,
                        cat_features=cat_features,
                        eval_set=eval_set,
                        sample_weight=sample_weight,
                        eval_sample_weight=eval_sample_weight,
                    )
                )
            else:
                probe_lr, probe_metadata = self._run_learning_rate_probe(
                    make_model,
                    X, y,
                    cat_features=cat_features,
                    eval_set=eval_set,
                    sample_weight=sample_weight,
                    eval_sample_weight=eval_sample_weight,
                    fit_kwargs=kw,
                )
                if probe_lr is not None:
                    kw["learning_rate"] = probe_lr
                model = make_model(kw)
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
        elif tree_mode_auto and not es_active:
            split_source = "automatic_tree_mode_selection"
        elif es_active:
            split_source = "automatic"
        else:
            split_source = "none"
        validation_metadata = {
            "source": split_source,
            "validation_fraction_input": self.validation_fraction,
            "validation_fraction_resolved": validation_fraction_resolved,
            "validation_fraction_realized": (
                None if split_eval_n is None
                else float(split_eval_n) / max(1, int(split_train_n))
            ),
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
        self._attach_tree_mode_selection_metadata(tree_mode_selection_metadata)
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
            self._attach_tree_mode_selection_metadata(tree_mode_selection_metadata)
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
                            "params": self._wrapper_params_header(),
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
                    classes,
                    wrapper_arrays["classes_kinds"],
                    name="wrapper classes",
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
