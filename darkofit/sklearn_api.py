"""Scikit-learn flavored estimators: fit / predict / predict_proba."""

import hashlib
import math
import warnings
from collections.abc import Mapping

import numpy as np
from ._validation import (
    array_like_to_numpy,
    coerce_feature_matrix,
    feature_names_from_input,
    n_features_from_array_like,
    n_samples_from_array_like,
    normalize_random_state_seed,
    resolve_cat_features,
    sklearn_assume_finite,
    validate_feature_names,
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
from .callbacks import WallClockStopper, _normalize_callbacks
from .losses import VECTOR_LOSSES
from .linear_residual import WeightedRidgeTrend, validate_linear_residual_loss
from .target_encoding import _is_missing_value
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

# Parameters that exist only on the sklearn wrappers, not on the core boosters.
_SKLEARN_ONLY = frozenset({
    "early_stopping", "validation_fraction", "validation_strategy", "refit",
    "refit_strategy", "auto_learning_rate_probe",
    "auto_learning_rate_probe_values", "auto_learning_rate_probe_iterations",
    "dist_calibration", "dist_calibration_feature", "dist_params",
    "sigma_calibration", "linear_residual", "linear_residual_alpha",
    "linear_residual_features", "linear_residual_fit_intercept",
    "linear_residual_standardize", "preset", "selection_rounds",
    "n_ensembles", "ensemble_bootstrap", "ensemble_shared_preprocessing",
})

_REFIT_STRATEGY_EXPONENT = {
    "best": 0.0,
    "exact": 0.0,
    "sqrt": 0.5,
    "linear": 1.0,
    "scaled": 1.0,
}

_AUTO_TREE_MODE_CANDIDATES = ("catboost", "lightgbm", "hybrid")
_ACCURACY_PRESET_PARAMS = {
    "iterations": 10_000,
    "tree_mode": "auto",
    "l2_leaf_reg": 3.0,
    "max_bins": 128,
    "learning_rate": 0.1,
    "ts_permutations": 1,
    "linear_residual": False,
    "early_stopping": True,
    "use_best_model": True,
}
_SIGMA_CALIBRATION_MIN_EFFECTIVE_N = 200.0
_SIGMA_AFFINE_BOUNDS = (0.5, 2.0)
_SIGMA_CALIBRATION_Z_GUARD = 1000.0
_SIGMA_CALIBRATION_INFLUENCE_TOP_K = 5
_SIGMA_CALIBRATION_INFLUENCE_THRESHOLD = 0.5
_SIGMA_MIN = 1e-12
_GROUP_AFFINE_DEFAULT_FEATURE = "metric_code"


def _normalize_tree_mode_token(tree_mode):
    if tree_mode is None:
        return "catboost"
    return str(tree_mode).lower().replace("-", "_")


def _is_auto_tree_mode(tree_mode):
    return _normalize_tree_mode_token(tree_mode) == "auto"


def _normalize_regression_preset(preset):
    if preset is None:
        return None
    token = str(preset).strip().lower().replace("-", "_")
    if token in {"", "none", "off", "false", "no"}:
        return None
    if token == "accuracy":
        return token
    raise ValueError("preset must be None or 'accuracy'")


def _normalize_selection_rounds(selection_rounds):
    if selection_rounds is None:
        return None
    if isinstance(selection_rounds, (bool, np.bool_)) or not isinstance(
        selection_rounds, (int, np.integer)
    ):
        raise TypeError("selection_rounds must be a positive integer or None")
    selection_rounds = int(selection_rounds)
    if selection_rounds < 1:
        raise ValueError("selection_rounds must be at least 1")
    return selection_rounds


def _normalize_n_ensembles(n_ensembles):
    if isinstance(n_ensembles, (bool, np.bool_)) or not isinstance(
        n_ensembles, (int, np.integer)
    ):
        raise TypeError("n_ensembles must be a positive integer")
    n_ensembles = int(n_ensembles)
    if n_ensembles < 1:
        raise ValueError("n_ensembles must be at least 1")
    return n_ensembles


def _normalize_ensemble_bootstrap(bootstrap):
    token = str(bootstrap).strip().lower().replace("-", "_")
    if token not in {"rows", "groups"}:
        raise ValueError("ensemble_bootstrap must be 'rows' or 'groups'")
    return token


def _take_rows(values, indices):
    iloc = getattr(values, "iloc", None)
    if iloc is not None:
        return iloc[indices]
    return np.asarray(values)[indices]


def _index_sha256(indices):
    values = np.ascontiguousarray(np.asarray(indices, dtype="<i8"))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _ensemble_bootstrap_plan(
    n_rows,
    seed,
    *,
    bootstrap,
    groups=None,
    y=None,
    required_class_count=None,
    sample_weight=None,
    max_attempts=128,
):
    """Return a deterministic bootstrap/OOB plan with usable validation."""
    n_rows = int(n_rows)
    if n_rows < 2:
        raise ValueError("an ensemble requires at least two training rows")
    rng = np.random.default_rng(int(seed))
    group_codes = None
    unique_group_count = None
    if bootstrap == "groups":
        group_values = np.asarray(groups)
        if group_values.ndim != 1 or len(group_values) != n_rows:
            raise ValueError(
                "groups must be one-dimensional with one value per training row"
            )
        try:
            _, group_codes = np.unique(group_values, return_inverse=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "groups must contain consistently comparable scalar values"
            ) from exc
        group_codes = np.asarray(group_codes, dtype=np.int64)
        unique_group_count = int(group_codes.max()) + 1
        if unique_group_count < 2:
            raise ValueError(
                "ensemble_bootstrap='groups' requires at least two groups"
            )

    weights = (
        None
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    labels = None if y is None else np.asarray(y)
    for attempt in range(1, int(max_attempts) + 1):
        if bootstrap == "rows":
            sampled = rng.integers(
                0, n_rows, size=n_rows, dtype=np.int64
            )
            selected_groups = None
        else:
            selected_groups = rng.integers(
                0, unique_group_count,
                size=unique_group_count,
                dtype=np.int64,
            )
            sampled = np.concatenate(
                [
                    np.flatnonzero(group_codes == group)
                    for group in selected_groups
                ]
            ).astype(np.int64, copy=False)
        oob_mask = np.ones(n_rows, dtype=np.bool_)
        if bootstrap == "groups":
            oob_mask[np.isin(group_codes, np.unique(selected_groups))] = False
        else:
            oob_mask[sampled] = False
        oob = np.flatnonzero(oob_mask).astype(np.int64, copy=False)
        if not len(oob):
            continue
        if weights is not None and (
            float(np.sum(weights[sampled])) <= 0.0
            or float(np.sum(weights[oob])) <= 0.0
        ):
            continue
        if (
            required_class_count is not None
            and np.unique(labels[sampled]).size != int(required_class_count)
        ):
            continue
        return {
            "sampled": sampled,
            "oob": oob,
            "attempts": attempt,
            "sampled_group_draws": (
                None if selected_groups is None else len(selected_groups)
            ),
            "sampled_unique_groups": (
                None
                if selected_groups is None
                else np.unique(selected_groups).size
            ),
            "oob_groups": (
                None
                if group_codes is None
                else np.unique(group_codes[oob]).size
            ),
        }
    raise RuntimeError(
        "could not construct a bootstrap sample with a usable, class-safe "
        "out-of-bag validation set"
    )


def _validate_loaded_ensemble_metadata(metadata, members, *, classification):
    """Fail closed on contradictory fitted-ensemble provenance."""
    if (
        metadata.get("version") != 1
        or metadata.get("claim_tier") != "E"
        or metadata.get("default_changed") is not False
        or metadata.get("oob_early_stopping") is not True
    ):
        raise ValueError(
            "invalid DarkoFit model: ensemble provenance is invalid"
        )
    member_count = len(members)
    if metadata.get("member_count") != member_count:
        raise ValueError(
            "invalid DarkoFit model: ensemble metadata member count does not "
            "match its payload"
        )
    bootstrap = metadata.get("bootstrap")
    if bootstrap not in {"rows", "groups"}:
        raise ValueError(
            "invalid DarkoFit model: ensemble bootstrap mode is invalid"
        )
    expected_aggregation = "soft_vote" if classification else "mean"
    if metadata.get("aggregation") != expected_aggregation:
        raise ValueError(
            "invalid DarkoFit model: ensemble aggregation is invalid"
        )
    shared = metadata.get("shared_preprocessing")
    if shared not in {"numeric_target_free", "member_local"}:
        raise ValueError(
            "invalid DarkoFit model: ensemble preprocessing mode is invalid"
        )
    seeds = metadata.get("member_seeds")
    records = metadata.get("members")
    if (
        not isinstance(seeds, list)
        or not isinstance(records, list)
        or len(seeds) != member_count
        or len(records) != member_count
    ):
        raise ValueError(
            "invalid DarkoFit model: ensemble member provenance is invalid"
        )
    for index, (seed, record, member) in enumerate(
        zip(seeds, records, members)
    ):
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(record, Mapping)
            or record.get("member") != index
            or record.get("seed") != seed
            or member.get_params().get("random_state") != seed
            or record.get("validation_source") != "explicit_eval_set"
        ):
            raise ValueError(
                "invalid DarkoFit model: ensemble member provenance does not "
                "match its payload"
            )
        for name in ("bootstrap_indices_sha256", "oob_indices_sha256"):
            digest = record.get(name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError(
                    "invalid DarkoFit model: ensemble index digest is invalid"
                )
        for name in (
            "bootstrap_attempts",
            "bootstrap_rows",
            "bootstrap_unique_rows",
            "oob_rows",
            "best_iteration",
        ):
            value = record.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if name == "best_iteration" else 1)
            ):
                raise ValueError(
                    f"invalid DarkoFit model: ensemble member {name} is invalid"
                )
        if bootstrap == "groups" and record.get("group_disjoint") is not True:
            raise ValueError(
                "invalid DarkoFit model: group ensemble is not disjoint"
            )


class _SelectionRoundStopper:
    """Private callback that caps an automatic-mode audition."""

    stop_reason = "selection_round_limit"

    def __init__(self, rounds):
        self.rounds = int(rounds)

    def __call__(self, progress):
        return progress.rounds_completed >= self.rounds


def _wall_clock_callback_state(callbacks, *, refresh_deadline=False):
    """Snapshot aggregate state from shared monotonic deadline callbacks."""
    stoppers = tuple(
        callback
        for callback in callbacks
        if isinstance(callback, WallClockStopper)
    )
    if not stoppers:
        return {
            "wall_clock_stopper_count": 0,
            "wall_clock_elapsed_seconds": None,
            "deadline_hit": False,
        }
    if refresh_deadline:
        for stopper in stoppers:
            stopper.check_deadline()
    return {
        "wall_clock_stopper_count": len(stoppers),
        "wall_clock_elapsed_seconds": max(
            float(stopper.elapsed_seconds) for stopper in stoppers
        ),
        "deadline_hit": any(stopper.deadline_hit for stopper in stoppers),
    }


def _should_early_stop(setting):
    """Resolve early_stopping to a bool."""
    if not isinstance(setting, (bool, np.bool_)):
        raise ValueError("early_stopping must be a bool")
    return bool(setting)


def _normalize_validation_strategy(strategy):
    mode = str(strategy).lower().replace("-", "_")
    if mode in {"random", "weighted_stratified", "group"}:
        return mode
    raise ValueError(
        "validation_strategy must be 'random', 'weighted_stratified', or 'group'"
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
    if mode in {
        "per_metric_affine",
        "metric_affine",
        "affine_by_metric",
        "per_group_affine",
        "group_affine",
        "affine_by_group",
    }:
        return "per_metric_affine"
    raise ValueError(
        "sigma_calibration must be None, False, True, 'scalar', 'affine', "
        "or 'per_metric_affine'"
    )


def _is_distributional_loss(loss):
    return loss in VECTOR_LOSSES


def _should_use_linear_residual(setting):
    if not isinstance(setting, (bool, np.bool_)):
        raise ValueError("linear_residual must be a bool")
    return bool(setting)


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
                "'affine', 'per_metric_affine', or 'dispersion'"
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


def _sigma_calibration_base_arrays(model, X_val, y_val):
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
    return target, mu, sigma


def _sigma_calibration_arrays(model, X_val, y_val, sample_weight=None):
    target, mu, sigma = _sigma_calibration_base_arrays(model, X_val, y_val)
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


def _fit_affine_sigma_calibration_from_arrays(
    y_val, mu, sigma, sample_weight=None, *, loss=None, fold_stats=None
):
    if getattr(loss, "name", None) == "StudentT":
        return _fit_affine_student_t_scale_calibration_from_arrays(
            y_val, mu, sigma, loss.nu, sample_weight, fold_stats=fold_stats
        )
    scalar_scale = _fit_scalar_sigma_scale_from_arrays(
        y_val, mu, sigma, sample_weight
    )
    fallback = None
    if fold_stats is not None:
        effective_n = float(fold_stats.get("validation_effective_n", 0.0))
    else:
        effective_n = float(y_val.shape[0]) if sample_weight is None else (
            _weighted_sum(sample_weight, None) ** 2
            / max(_weighted_sum(sample_weight * sample_weight, None), _SIGMA_MIN)
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
                residual2, log_sigma, candidate_b, sample_weight
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
                residual2, log_sigma, best_b, sample_weight
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
    return _fit_affine_sigma_calibration_from_arrays(
        y_val, mu, sigma, w, loss=loss, fold_stats=fold_stats
    )


def _calibration_group_key(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("dist_calibration_feature contains non-finite values")
        return value
    return str(value)


def _normalize_calibration_groups(values):
    values = np.asarray(values, dtype=object)
    if values.ndim != 1:
        raise ValueError("dist_calibration_feature must resolve to one column")
    return np.asarray([_calibration_group_key(v) for v in values], dtype=object)


def _jsonable_group_value(value):
    value = _calibration_group_key(value)
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _resolve_dist_calibration_feature(feature, feature_names, n_features):
    if feature is None:
        feature = _GROUP_AFFINE_DEFAULT_FEATURE
    if isinstance(feature, (int, np.integer)):
        index = int(feature)
        if index < 0:
            index += int(n_features)
        if index < 0 or index >= int(n_features):
            raise ValueError(
                "dist_calibration_feature index is out of bounds for X"
            )
        name = None
        if feature_names is not None:
            name = str(feature_names[index])
        return index, name, feature
    if isinstance(feature, str):
        if feature_names is None:
            raise ValueError(
                "dist_calibration='per_metric_affine' with a string "
                "dist_calibration_feature requires named input columns; pass "
                "an integer column index for numpy arrays"
            )
        matches = np.flatnonzero(np.asarray(feature_names, dtype=object) == feature)
        if matches.size != 1:
            raise ValueError(
                f"dist_calibration_feature={feature!r} was not found uniquely "
                "in the fit feature names"
            )
        return int(matches[0]), feature, feature
    raise ValueError("dist_calibration_feature must be a column name or index")


def _extract_feature_column_by_index(X, index):
    iloc = getattr(X, "iloc", None)
    if iloc is not None:
        return np.asarray(iloc[:, int(index)], dtype=object)
    if isinstance(X, np.ndarray):
        # Slice the one calibration column instead of copying the whole
        # matrix to object dtype on every predict call. Coerce first so ndarray
        # subclasses such as np.matrix cannot preserve a two-dimensional
        # (n, 1) slice here.
        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array")
        return np.asarray(X)[:, int(index)].astype(object, copy=False)
    arr = array_like_to_numpy(X, object)
    if arr.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    return np.asarray(arr[:, int(index)], dtype=object)


def _fit_grouped_affine_sigma_calibration(
    model, X_val, y_val, groups, sample_weight=None, *, fold_stats=None
):
    target, mu, sigma = _sigma_calibration_base_arrays(model, X_val, y_val)
    groups = _normalize_calibration_groups(groups)
    if groups.shape[0] != target.shape[0]:
        raise ValueError(
            "dist_calibration_feature column length must match validation rows"
        )
    w = None
    if sample_weight is not None:
        w_all = np.asarray(sample_weight, dtype=np.float64)
        positive = w_all > 0.0
        if not np.any(positive):
            raise ValueError("eval_sample_weight must have positive total weight")
        target = target[positive]
        mu = mu[positive]
        sigma = sigma[positive]
        groups = groups[positive]
        w = w_all[positive]

    loss = getattr(model, "loss_", None)
    global_calibration = _fit_affine_sigma_calibration_from_arrays(
        target, mu, sigma, w, loss=loss, fold_stats=fold_stats
    )
    records = []
    seen = []
    for group in groups:
        if not any(group == existing for existing in seen):
            seen.append(group)
    for group in seen:
        mask = groups == group
        group_w = None if w is None else w[mask]
        group_stats = _sigma_calibration_fold_stats(int(np.sum(mask)), group_w)
        mass = (
            float(np.sum(group_w))
            if group_w is not None
            else float(np.sum(mask))
        )
        fallback_reason = None
        if (
            group_stats["validation_effective_n"]
            < _SIGMA_CALIBRATION_MIN_EFFECTIVE_N
        ):
            calibration = global_calibration
            fallback_reason = "small_group"
        else:
            calibration = _fit_affine_sigma_calibration_from_arrays(
                target[mask], mu[mask], sigma[mask], group_w,
                loss=loss, fold_stats=group_stats,
            )
            fallback_reason = calibration.get("fallback_reason")
        records.append({
            "group": _jsonable_group_value(group),
            "sigma_scale": float(calibration["sigma_scale"]),
            "sigma_affine_a": float(calibration["sigma_affine_a"]),
            "sigma_affine_b": float(calibration["sigma_affine_b"]),
            "validation_n_samples": int(group_stats["validation_n_samples"]),
            "validation_positive_weight_n": int(
                group_stats["validation_positive_weight_n"]
            ),
            "validation_effective_n": float(
                group_stats["validation_effective_n"]
            ),
            "validation_weight_sum": mass,
            "fallback_reason": fallback_reason,
        })
    global_calibration["group_affine"] = records
    global_calibration["group_count"] = len(records)
    global_calibration["group_fallback_count"] = sum(
        1 for record in records if record.get("fallback_reason") is not None
    )
    return global_calibration


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
        raise ValueError(
            f"{name} sums to zero; at least one weight must be positive"
        )
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


def _ordinal_category_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "ordinal feature byte categories must be valid UTF-8"
            ) from exc
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("ordinal feature categories must be finite")
        return value
    raise TypeError(
        "ordinal feature categories must contain only strings, booleans, "
        "integers, or finite floats"
    )


def _normalize_ordinal_categories(
    categories, *, feature, min_categories=2
):
    if isinstance(categories, (str, bytes, set, frozenset, Mapping)):
        raise TypeError(
            f"ordinal_features[{feature!r}] must be an ordered category sequence"
        )
    try:
        values = list(categories)
    except TypeError as exc:
        raise TypeError(
            f"ordinal_features[{feature!r}] must be an ordered category sequence"
        ) from exc
    normalized = [_ordinal_category_scalar(value) for value in values]
    if len(normalized) < int(min_categories):
        minimum = (
            "two" if int(min_categories) == 2 else str(int(min_categories))
        )
        raise ValueError(
            f"ordinal_features[{feature!r}] must declare at least "
            f"{minimum} categories"
        )
    for index, value in enumerate(normalized):
        if any(value == earlier for earlier in normalized[:index]):
            raise ValueError(
                f"ordinal_features[{feature!r}] contains duplicate category "
                f"{value!r}"
            )
    return normalized


class _FrozenAutoOrdinalFeatures(dict):
    """Internal full-data auto resolution passed unchanged to CV folds."""


def _ordinal_column_values(X, index):
    if hasattr(X, "iloc"):
        return np.asarray(X.iloc[:, int(index)], dtype=object)
    return array_like_to_numpy(X, object)[:, int(index)]


def _auto_ordinal_categories(X, index, *, allow_integer_codes):
    if hasattr(X, "dtypes") and hasattr(X, "iloc"):
        try:
            import pandas as pd

            dtype = X.dtypes.iloc[int(index)]
            if isinstance(dtype, pd.CategoricalDtype) and bool(dtype.ordered):
                categories = _normalize_ordinal_categories(
                    dtype.categories,
                    feature=int(index),
                    min_categories=0,
                )
                return categories, "auto_ordered_categorical"
        except (AttributeError, ImportError):
            pass

    if not allow_integer_codes:
        return None

    values = [
        value.item() if isinstance(value, np.generic) else value
        for value in _ordinal_column_values(X, index)
        if not _is_missing_value(value)
    ]
    if values and all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in values
    ):
        categories = sorted(set(int(value) for value in values))
        if len(categories) >= 2:
            return categories, "auto_integer_codes"
    return None


def _resolve_ordinal_features(X, cat_features, ordinal_features):
    if ordinal_features is None or ordinal_features is False:
        return cat_features, "off", []
    X = _ensure_dense(X)
    n_features = n_features_from_array_like(X)
    names = feature_names_from_input(X)
    nominal = list(resolve_cat_features(cat_features, X, n_features))

    records = []
    if isinstance(ordinal_features, str):
        if ordinal_features.lower().replace("-", "_") != "auto":
            raise ValueError("ordinal_features must be a mapping, 'auto', or None")
        mode = "auto"
        nominal_set = set(nominal)
        for index in range(n_features):
            detected = _auto_ordinal_categories(
                X,
                index,
                allow_integer_codes=index in nominal_set,
            )
            if detected is None:
                continue
            categories, source = detected
            records.append({
                "index": int(index),
                "name": None if names is None else str(names[index]),
                "categories": categories,
                "source": source,
            })
    elif isinstance(ordinal_features, Mapping):
        mode = "explicit"
        for feature, categories in ordinal_features.items():
            resolved = resolve_cat_features([feature], X, n_features)
            index = int(resolved[0])
            records.append({
                "index": index,
                "name": None if names is None else str(names[index]),
                "categories": _normalize_ordinal_categories(
                    categories,
                    feature=feature,
                    min_categories=(
                        0
                        if isinstance(
                            ordinal_features, _FrozenAutoOrdinalFeatures
                        )
                        else 2
                    ),
                ),
                "source": "explicit",
            })
    else:
        raise TypeError("ordinal_features must be a mapping, 'auto', or None")

    records.sort(key=lambda record: record["index"])
    indices = [record["index"] for record in records]
    if len(indices) != len(set(indices)):
        raise ValueError("ordinal_features contains duplicate feature references")
    ordinal_set = set(indices)
    nominal = [index for index in nominal if index not in ordinal_set]
    return nominal, mode, records


def _ordinal_codes(values, categories, *, feature, name):
    values = np.asarray(
        [
            (
                value.item()
                if isinstance(value, np.generic)
                else value
            )
            for value in np.asarray(values, dtype=object)
        ],
        dtype=object,
    )
    for index, value in enumerate(values):
        if isinstance(value, bytes):
            try:
                values[index] = value.decode("utf-8")
            except UnicodeDecodeError:
                # A declared byte category must have been normalized from
                # valid UTF-8. Leave invalid observed bytes unmatched so the
                # documented fail-closed unknown-category error is raised.
                pass
    try:
        import pandas as pd

        series = pd.Series(values, dtype=object)
        missing = pd.isna(series).to_numpy()
        try:
            codes = pd.Index(categories).get_indexer(series).astype(np.float64)
        except TypeError:
            codes = _ordinal_codes_elementwise(values, categories, missing)
    except ImportError:
        missing = np.fromiter(
            (_is_missing_value(value) for value in values),
            dtype=bool,
            count=len(values),
        )
        category_map = {value: index for index, value in enumerate(categories)}
        codes = np.fromiter(
            (
                (
                    -1.0
                    if is_missing
                    else _ordinal_hash_lookup(category_map, value)
                )
                for value, is_missing in zip(values, missing)
            ),
            dtype=np.float64,
            count=len(values),
        )
    unknown = (codes < 0.0) & ~missing
    if np.any(unknown):
        first = values[int(np.flatnonzero(unknown)[0])]
        label = (
            f"column {feature}"
            if name is None
            else f"column {name!r} (index {feature})"
        )
        raise ValueError(
            f"{label} contains unknown ordinal category {first!r}; "
            "declare the complete ordered category list"
        )
    codes[missing] = np.nan
    return codes


def _ordinal_hash_lookup(category_map, value):
    try:
        return float(category_map.get(value, -1))
    except TypeError:
        return -1.0


def _ordinal_codes_elementwise(values, categories, missing):
    codes = np.full(len(values), -1.0, dtype=np.float64)
    for row, (value, is_missing) in enumerate(zip(values, missing)):
        if is_missing:
            continue
        for code, category in enumerate(categories):
            try:
                equal = value == category
            except (TypeError, ValueError):
                continue
            if isinstance(equal, (bool, np.bool_)) and bool(equal):
                codes[row] = float(code)
                break
    return codes


def _transform_ordinal_features(X, records):
    if not records:
        return X
    if hasattr(X, "iloc") and hasattr(X, "copy"):
        transformed = X.copy()
        for record in records:
            index = record["index"]
            codes = _ordinal_codes(
                _ordinal_column_values(transformed, index),
                record["categories"],
                feature=index,
                name=record.get("name"),
            )
            if hasattr(transformed, "isetitem"):
                transformed.isetitem(index, codes)
            else:
                transformed.iloc[:, index] = codes
        return transformed

    transformed = array_like_to_numpy(X, object).copy()
    for record in records:
        index = record["index"]
        transformed[:, index] = _ordinal_codes(
            transformed[:, index],
            record["categories"],
            feature=index,
            name=record.get("name"),
        )
    return transformed


def _restore_ordinal_records(records, *, n_features, feature_names=None):
    if not isinstance(records, list):
        raise ValueError(
            "invalid DarkoFit model: ordinal feature state must be a list"
        )
    if records and feature_names is not None:
        feature_names = np.asarray(feature_names, dtype=object)
        if (
            feature_names.ndim != 1
            or feature_names.shape[0] != int(n_features)
            or not all(isinstance(name, str) for name in feature_names)
        ):
            raise ValueError(
                "invalid DarkoFit model: feature name state does not match "
                "the ordinal feature state"
            )
    restored = []
    seen = set()
    allowed_sources = {
        "explicit",
        "auto_integer_codes",
        "auto_ordered_categorical",
    }
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(
                "invalid DarkoFit model: ordinal feature record must be an object"
            )
        index = record.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(
                "invalid DarkoFit model: ordinal feature index must be an integer"
            )
        if index < 0 or index >= int(n_features) or index in seen:
            raise ValueError(
                "invalid DarkoFit model: ordinal feature index is invalid"
            )
        seen.add(index)
        source = record.get("source")
        if not isinstance(source, str) or source not in allowed_sources:
            raise ValueError(
                "invalid DarkoFit model: ordinal feature source is invalid"
            )
        try:
            categories = _normalize_ordinal_categories(
                record.get("categories"),
                feature=index,
                min_categories=(
                    0 if source == "auto_ordered_categorical" else 2
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid DarkoFit model: {exc}") from exc
        name = record.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError(
                "invalid DarkoFit model: ordinal feature name must be a string"
            )
        if feature_names is not None:
            if name is None or str(feature_names[index]) != name:
                raise ValueError(
                    "invalid DarkoFit model: ordinal feature name does not "
                    "match its index"
                )
        elif name is not None:
            raise ValueError(
                "invalid DarkoFit model: unnamed input cannot have an "
                "ordinal feature name"
            )
        restored.append({
            "index": index,
            "name": name,
            "categories": categories,
            "source": source,
        })
    return restored


def _feature_names_from_input(X):
    return feature_names_from_input(X)


def _coerce_fit_X(X, cat_features):
    X = _ensure_dense(X)
    return coerce_feature_matrix(
        X,
        cat_features,
        name="X",
        resolve_names=True,
        check_infinite=not sklearn_assume_finite(),
    )


def _validate_feature_names(expected_names, X, *, name="X"):
    validate_feature_names(
        expected_names,
        X,
        name=name,
        fitted_name="DarkoFit",
    )


def _validate_eval_set_features(
    eval_set,
    n_features,
    expected_feature_names=None,
    cat_features=(),
    feature_names_validated=False,
):
    if eval_set is None:
        return None
    Xv, yv = eval_set
    _validate_eval_set_feature_schema(
        Xv,
        n_features,
        expected_feature_names=expected_feature_names,
        feature_names_validated=feature_names_validated,
    )
    Xv, _, _ = coerce_feature_matrix(
        Xv,
        cat_features,
        name="eval_set[0]",
        check_infinite=not sklearn_assume_finite(),
    )
    yv = validate_target_vector(
        yv, n_samples_from_array_like(Xv, name="eval_set[0]"),
        name="eval_set[1]",
    )
    return (Xv, yv)


def _validate_eval_set_feature_schema(
    Xv,
    n_features,
    expected_feature_names=None,
    feature_names_validated=False,
):
    actual = n_features_from_array_like(Xv, name="eval_set[0]")
    if actual != int(n_features):
        raise ValueError(
            f"eval_set[0] has {actual} features, but X has "
            f"{int(n_features)} features"
        )
    if not feature_names_validated:
        _validate_feature_names(
            expected_feature_names, Xv, name="eval_set[0]"
        )


def _infer_model_n_features(model):
    prep = getattr(model, "prep_", None)
    if prep is None:
        return None
    n_features = getattr(prep, "n_input_features_", None)
    return None if n_features is None else int(n_features)


def _check_predict_input(estimator, X):
    check_is_fitted(estimator, "model_")
    X = _ensure_dense(X)
    actual = n_features_from_array_like(X, allow_empty=True)
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
    X = _transform_ordinal_features(
        X, getattr(estimator, "ordinal_features_", ())
    )
    prep = getattr(estimator.model_, "prep_", None)
    cat_features = getattr(prep, "cat_features_", ())
    X, _, _ = coerce_feature_matrix(
        X,
        cat_features,
        name="X",
        check_infinite=not sklearn_assume_finite(),
        allow_empty=True,
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
    if validation_strategy == "group" and groups is None:
        raise ValueError("validation_strategy='group' requires groups")
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

    def _clear_ensemble_state(self):
        for name in (
            "estimators_",
            "ensemble_metadata_",
            "ensemble_best_iterations_",
            "ensemble_learning_rates_",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def _configure_model_preprocessing(self, model):
        payload = getattr(self, "_shared_numeric_preprocessing_", None)
        if payload is not None:
            model._shared_preprocessing_payload = payload
        return model

    def _make_shared_numeric_preprocessing(
        self, X, y, sample_weight, seed
    ):
        """Fit one target-free numeric preprocessor for all bag members."""
        from .preprocessing import FeaturePreprocessor

        prep = FeaturePreprocessor(
            self.max_bins,
            self.cat_smoothing,
            None if seed is None else int(seed),
            include_cat_codes=False,
            target_encoding_mode="ordered",
            ts_permutations=self.ts_permutations,
            target_ordered_cat_codes="off",
            bin_sample_count=self.bin_sample_count,
        )
        placeholder_target = np.zeros(len(X), dtype=np.float64)
        binned = prep.fit_transform(
            X,
            [placeholder_target],
            cat_features=None,
            sample_weight=sample_weight,
        )
        return prep, np.asarray(binned)

    def _adopt_ensemble(self, estimators, metadata):
        if not estimators:
            raise ValueError("an ensemble must contain at least one estimator")
        first = estimators[0]
        self.estimators_ = tuple(estimators)
        self.model_ = first.model_
        for name in (
            "n_features_in_",
            "feature_names_in_",
            "ordinal_features_mode_",
            "ordinal_features_",
            "ordinal_feature_indices_",
            "_ordinal_nominal_cat_count_",
            "classes_",
            "n_classes_",
            "_multiclass",
            "preset_",
            "preset_params_",
        ):
            if hasattr(first, name):
                setattr(self, name, getattr(first, name))
        best_iterations = tuple(
            int(member.best_n_estimators_) for member in estimators
        )
        learning_rates = tuple(
            float(member.learning_rate_) for member in estimators
        )
        best_scores = np.asarray(
            [float(member.best_score_) for member in estimators],
            dtype=np.float64,
        )
        self.ensemble_best_iterations_ = best_iterations
        self.ensemble_learning_rates_ = learning_rates
        self._best_n_estimators_ = int(np.median(best_iterations))
        self._learning_rate_ = float(np.median(learning_rates))
        self._best_score_ = (
            float(np.mean(best_scores[np.isfinite(best_scores)]))
            if np.any(np.isfinite(best_scores))
            else float("nan")
        )
        self.ensemble_metadata_ = metadata
        return self

    def _ensemble_params_header(self):
        params = self.get_params()
        if params.get("random_state") is not None:
            params["random_state"] = int(
                self.ensemble_metadata_["fit_random_state_seed"]
            )
        return params

    @classmethod
    def _load_ensemble_model(cls, path):
        import io
        from .serialization import load_ensemble

        archive = load_ensemble(path)
        if archive is None:
            return None
        header, payloads = archive
        saved_class = header["wrapper_class"]
        if saved_class != cls.__name__:
            raise TypeError(
                f"{path!r} was saved by {saved_class}, not {cls.__name__}"
            )
        est = cls()
        params = header["params"]
        known = est.get_params()
        est.set_params(**{k: v for k, v in params.items() if k in known})
        expected_members = _normalize_n_ensembles(est.n_ensembles)
        if expected_members != len(payloads):
            raise ValueError(
                "invalid DarkoFit model: ensemble params do not match its "
                "member count"
            )
        members = [
            cls.load_model(io.BytesIO(payload)) for payload in payloads
        ]
        for member in members:
            if _normalize_n_ensembles(member.n_ensembles) != 1:
                raise ValueError(
                    "invalid DarkoFit model: nested ensemble member detected"
                )
        metadata = header["metadata"]
        _validate_loaded_ensemble_metadata(
            metadata,
            members,
            classification=hasattr(members[0], "classes_"),
        )
        feature_counts = {
            int(member.n_features_in_) for member in members
        }
        if len(feature_counts) != 1:
            raise ValueError(
                "invalid DarkoFit model: ensemble members disagree on input "
                "feature count"
            )
        if hasattr(members[0], "classes_"):
            reference = np.asarray(members[0].classes_)
            if any(
                not np.array_equal(reference, np.asarray(member.classes_))
                for member in members[1:]
            ):
                raise ValueError(
                    "invalid DarkoFit model: ensemble members disagree on "
                    "class labels"
                )
        return est._adopt_ensemble(members, metadata)

    def _fit_ensemble(
        self,
        X,
        y,
        *,
        cat_features,
        eval_set,
        groups,
        sample_weight,
        eval_sample_weight,
        callbacks,
        ordinal_features,
        classification,
    ):
        n_members = _normalize_n_ensembles(self.n_ensembles)
        if n_members == 1:
            return None
        bootstrap = _normalize_ensemble_bootstrap(self.ensemble_bootstrap)
        if eval_set is not None or eval_sample_weight is not None:
            raise ValueError(
                "ensemble fits use each member's out-of-bag rows for "
                "validation; eval_set and eval_sample_weight are not supported"
            )
        if callbacks:
            raise ValueError(
                "callbacks are not supported with n_ensembles > 1 because "
                "each member owns an independent boosting lifecycle"
            )
        if self.refit:
            raise ValueError(
                "refit=True is not supported with n_ensembles > 1; OOB "
                "members intentionally retain their bootstrap training rows"
            )
        if isinstance(ordinal_features, str) and (
            ordinal_features.strip().lower() == "auto"
        ):
            raise ValueError(
                "ordinal_features='auto' is not supported with ensembles; "
                "declare complete ordinal category orders explicitly"
            )
        if not isinstance(
            self.ensemble_shared_preprocessing, (bool, np.bool_)
        ):
            raise TypeError("ensemble_shared_preprocessing must be a bool")
        if not classification and _is_distributional_loss(self.loss):
            raise ValueError(
                "n_ensembles currently supports scalar regression losses only; "
                "distributional parameter aggregation is not yet defined"
            )

        X_checked, resolved_cat_features, n_features = _coerce_fit_X(
            X, cat_features
        )
        y_checked = validate_target_vector(
            y,
            X_checked.shape[0],
            dtype=None if classification else np.float64,
        )
        if classification:
            target_type = type_of_target(y_checked)
            if target_type not in {"binary", "multiclass"}:
                raise ValueError(f"Unknown label type: {target_type}")
            required_class_count = int(np.unique(y_checked).size)
            if required_class_count < 2:
                raise ValueError(
                    f"Need at least 2 classes; got {required_class_count} class."
                )
        else:
            required_class_count = None
        sample_weight_checked = _validate_wrapper_sample_weight(
            sample_weight, X_checked.shape[0]
        )
        group_values = None
        if bootstrap == "groups":
            if groups is None:
                raise ValueError(
                    "ensemble_bootstrap='groups' requires groups in fit"
                )
            group_values = np.asarray(groups)
            if group_values.ndim != 1 or len(group_values) != len(X_checked):
                raise ValueError(
                    "groups must be one-dimensional with one value per "
                    "training row"
                )

        fit_seed = normalize_random_state_seed(self.random_state)
        member_seeds = tuple(
            int(value)
            for value in np.random.default_rng(fit_seed).integers(
                0, 2**31 - 1, size=n_members
            )
        )
        shared_requested = bool(self.ensemble_shared_preprocessing)
        shared_eligible = (
            shared_requested
            and not resolved_cat_features
            and (
                ordinal_features is None
                or (
                    isinstance(ordinal_features, Mapping)
                    and not ordinal_features
                )
            )
            and np.asarray(X_checked).dtype.kind in "biuf"
        )
        shared_prep = None
        full_binned = None
        if shared_eligible:
            shared_prep, full_binned = self._make_shared_numeric_preprocessing(
                X_checked, y_checked, sample_weight_checked, fit_seed
            )

        estimators = []
        member_metadata = []
        for member_index, member_seed in enumerate(member_seeds):
            plan = _ensemble_bootstrap_plan(
                len(X_checked),
                member_seed,
                bootstrap=bootstrap,
                groups=group_values,
                y=y_checked,
                required_class_count=required_class_count,
                sample_weight=sample_weight_checked,
            )
            sampled = plan["sampled"]
            oob = plan["oob"]
            member = clone(self)
            member.set_params(
                n_ensembles=1,
                random_state=member_seed,
                early_stopping=True,
                use_best_model=True,
                refit=False,
            )
            member._suppress_wrapper_deprecation_warning = True
            if shared_eligible:
                member._shared_numeric_preprocessing_ = {
                    "prep": shared_prep,
                    "X_binned": np.asarray(full_binned[sampled]),
                }
            member_sample_weight = (
                None
                if sample_weight_checked is None
                else sample_weight_checked[sampled]
            )
            member_eval_weight = (
                None
                if sample_weight_checked is None
                else sample_weight_checked[oob]
            )
            try:
                member.fit(
                    _take_rows(X, sampled),
                    _take_rows(y, sampled),
                    cat_features=cat_features,
                    eval_set=(
                        _take_rows(X, oob),
                        _take_rows(y, oob),
                    ),
                    sample_weight=member_sample_weight,
                    eval_sample_weight=member_eval_weight,
                    ordinal_features=ordinal_features,
                )
            finally:
                if hasattr(member, "_shared_numeric_preprocessing_"):
                    del member._shared_numeric_preprocessing_
                if hasattr(member, "_suppress_wrapper_deprecation_warning"):
                    del member._suppress_wrapper_deprecation_warning
            validation = dict(
                member.model_.auto_params_.get("validation_split", {})
            )
            if (
                validation.get("source") != "explicit_eval_set"
                or int(validation.get("eval_n_samples", -1)) != len(oob)
            ):
                raise RuntimeError(
                    "ensemble member did not bind early stopping to its OOB rows"
                )
            estimators.append(member)
            member_metadata.append({
                "member": member_index,
                "seed": member_seed,
                "bootstrap_attempts": int(plan["attempts"]),
                "bootstrap_rows": int(len(sampled)),
                "bootstrap_unique_rows": int(np.unique(sampled).size),
                "bootstrap_indices_sha256": _index_sha256(sampled),
                "oob_rows": int(len(oob)),
                "oob_indices_sha256": _index_sha256(oob),
                "sampled_group_draws": plan["sampled_group_draws"],
                "sampled_unique_groups": plan["sampled_unique_groups"],
                "oob_groups": plan["oob_groups"],
                "group_disjoint": (
                    None if bootstrap == "rows" else True
                ),
                "best_iteration": int(member.best_n_estimators_),
                "learning_rate": float(member.learning_rate_),
                "stop_reason": str(
                    getattr(member.model_, "stop_reason_", "unknown")
                ),
                "validation_source": validation["source"],
            })

        metadata = {
            "version": 1,
            "claim_tier": "E",
            "default_changed": False,
            "member_count": n_members,
            "member_seeds": list(member_seeds),
            "fit_random_state_seed": fit_seed,
            "bootstrap": bootstrap,
            "aggregation": (
                "soft_vote" if classification else "mean"
            ),
            "oob_early_stopping": True,
            "shared_preprocessing_requested": shared_requested,
            "shared_preprocessing": (
                "numeric_target_free"
                if shared_eligible
                else "member_local"
            ),
            "shared_preprocessing_fallback_reason": (
                None
                if shared_eligible or not shared_requested
                else (
                    "categorical_or_ordinal_features"
                    if resolved_cat_features or ordinal_features is not None
                    else "non_numeric_dtype"
                )
            ),
            "input_feature_count": int(n_features),
            "members": member_metadata,
        }
        return self._adopt_ensemble(estimators, metadata)

    def _prepare_ordinal_fit_input(self, X, cat_features, ordinal_features):
        nominal, mode, records = _resolve_ordinal_features(
            X, cat_features, ordinal_features
        )
        return _transform_ordinal_features(X, records), nominal, mode, records

    def _set_ordinal_fit_state(self, mode, records, nominal_cat_count):
        self.ordinal_features_mode_ = mode
        self.ordinal_features_ = records
        self.ordinal_feature_indices_ = np.asarray(
            [record["index"] for record in records], dtype=np.int64
        )
        self._ordinal_nominal_cat_count_ = int(nominal_cat_count)

    def _attach_ordinal_metadata(self):
        records = list(getattr(self, "ordinal_features_", ()))
        if getattr(self, "ordinal_features_mode_", "off") == "off":
            return
        metadata = {
            "mode": getattr(self, "ordinal_features_mode_", "off"),
            "active": bool(records),
            "feature_count": len(records),
            "feature_indices": [int(record["index"]) for record in records],
            "feature_names": [record.get("name") for record in records],
            "sources": [record["source"] for record in records],
            "nominal_categorical_count": int(
                getattr(self, "_ordinal_nominal_cat_count_", 0)
            ),
            "added_columns": 0,
            "target_stat_blocks_added": 0,
            "target_used": False,
            "unknown_policy": "fail_closed",
            "missing_policy": "numeric_missing_bin",
        }
        model = getattr(self, "model_", None)
        if model is not None:
            model.auto_params_["ordinal_features"] = metadata
            model.auto_params_.setdefault("diagnostics", {})
            model.auto_params_["diagnostics"]["ordinal_features"] = metadata

    def _restore_ordinal_state(self, state):
        mode = state.get("ordinal_features_mode", "off")
        if not isinstance(mode, str) or mode not in {"off", "explicit", "auto"}:
            raise ValueError(
                "invalid DarkoFit model: ordinal feature mode is invalid"
            )
        records = _restore_ordinal_records(
            state.get("ordinal_features", []),
            n_features=getattr(self, "n_features_in_", 0),
            feature_names=getattr(self, "feature_names_in_", None),
        )
        if mode == "off" and records:
            raise ValueError(
                "invalid DarkoFit model: off ordinal mode has active features"
            )
        if mode == "explicit" and any(
            record["source"] != "explicit" for record in records
        ):
            raise ValueError(
                "invalid DarkoFit model: ordinal feature mode and source "
                "do not match"
            )
        if mode == "auto" and any(
            record["source"] == "explicit" for record in records
        ):
            raise ValueError(
                "invalid DarkoFit model: ordinal feature mode and source "
                "do not match"
            )
        prep = getattr(getattr(self, "model_", None), "prep_", None)
        if prep is not None:
            nominal = set(int(index) for index in prep.cat_features_)
            numeric = set(int(index) for index in prep.num_features_)
            if any(
                record["index"] in nominal
                or record["index"] not in numeric
                for record in records
            ):
                raise ValueError(
                    "invalid DarkoFit model: ordinal feature index does not "
                    "match fitted preprocessing"
                )
        self.ordinal_features_mode_ = mode
        self.ordinal_features_ = records
        self.ordinal_feature_indices_ = np.asarray(
            [record["index"] for record in records], dtype=np.int64
        )
        self._ordinal_nominal_cat_count_ = len(
            getattr(prep, "cat_features_", ())
        )

    def _warn_wrapper_deprecated_options(self, *, stacklevel=3):
        if (
            self.auto_learning_rate_probe
            or self.auto_learning_rate_probe_values is not None
            or self.auto_learning_rate_probe_iterations != 80
        ):
            warnings.warn(
                "auto_learning_rate_probe, "
                "auto_learning_rate_probe_values, and "
                "auto_learning_rate_probe_iterations are deprecated and will "
                "be removed in DarkoFit 1.0; use an explicit "
                "validation-backed learning-rate search instead",
                FutureWarning,
                stacklevel=stacklevel,
            )
        if (
            bool(getattr(self, "linear_residual", False))
            or float(getattr(self, "linear_residual_alpha", 1.0)) != 1.0
            or getattr(self, "linear_residual_features", "auto") != "auto"
            or not bool(
                getattr(self, "linear_residual_fit_intercept", True)
            )
            or not bool(
                getattr(self, "linear_residual_standardize", True)
            )
        ):
            warnings.warn(
                "linear_residual, linear_residual_alpha, "
                "linear_residual_features, linear_residual_fit_intercept, "
                "and linear_residual_standardize are deprecated and will be "
                "removed in DarkoFit 1.0; use local linear_leaves=True for "
                "scalar RMSE or detrend explicitly before fitting other "
                "losses",
                FutureWarning,
                stacklevel=stacklevel,
            )

    def __sklearn_is_fitted__(self):
        return hasattr(self, "model_")

    def __sklearn_tags__(self):
        parent = getattr(super(), "__sklearn_tags__", None)
        if parent is None:
            return self._more_tags()
        tags = parent()
        tags.input_tags.allow_nan = True
        tags.input_tags.sparse = False
        return tags

    def _more_tags(self):
        return {"allow_nan": True, "X_types": ["2darray"]}

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
            "dist_group_affine_groups_", "dist_group_affine_a_",
            "dist_group_affine_b_", "dist_group_affine_metadata_",
            "dist_calibration_feature_", "dist_calibration_feature_index_",
            "dist_calibration_feature_name_",
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
            "sigma_group_affine_groups_", "sigma_group_affine_a_",
            "sigma_group_affine_b_",
            "sigma_calibration_fallback_reason_",
            "sigma_calibration_influence_stats_",
            "sigma_calibration_pooling_",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def _clear_preset_state(self):
        for name in ("preset_", "preset_params_"):
            if hasattr(self, name):
                delattr(self, name)

    def _attach_preset_metadata(self):
        preset = getattr(self, "preset_", None)
        if preset is None:
            return
        metadata = {
            "name": preset,
            "claim_tier": "E",
            "default_changed": False,
            "resolved": dict(self.preset_params_),
            "evidence_scope": "spent_development_panel",
        }
        model = getattr(self, "model_", None)
        auto_params = getattr(model, "auto_params_", None)
        if auto_params is not None:
            auto_params["preset"] = metadata
            auto_params.setdefault("diagnostics", {})
            auto_params["diagnostics"]["preset"] = metadata

    def _clear_linear_residual_state(self):
        for name in (
            "linear_residual_enabled_", "linear_residual_active_",
            "linear_residual_inactive_reason_", "linear_residual_trend_",
            "linear_residual_alpha_", "linear_residual_fit_intercept_",
            "linear_residual_standardize_", "linear_residual_feature_indices_",
            "linear_residual_feature_names_", "linear_residual_dropped_features_",
            "linear_residual_intercept_", "linear_residual_coef_",
            "linear_residual_transformed_coef_", "linear_residual_center_",
            "linear_residual_scale_", "linear_residual_impute_values_",
            "linear_residual_rank_", "linear_residual_singular_values_",
            "linear_residual_weight_sum_", "linear_residual_positive_weight_n_",
            "linear_residual_effective_n_", "linear_residual_target_mean_",
            "linear_residual_trend_train_mean_",
            "linear_residual_residual_stats_",
            "selection_linear_residual_summary_",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def _set_linear_residual_disabled_state(self):
        self.linear_residual_enabled_ = False
        self.linear_residual_active_ = False
        self.linear_residual_inactive_reason_ = "disabled"
        self.linear_residual_trend_ = None
        self.linear_residual_alpha_ = float(
            getattr(self, "linear_residual_alpha", 1.0)
        )
        self.linear_residual_fit_intercept_ = bool(
            getattr(self, "linear_residual_fit_intercept", True)
        )
        self.linear_residual_standardize_ = bool(
            getattr(self, "linear_residual_standardize", True)
        )
        self.linear_residual_feature_indices_ = np.empty(0, dtype=np.int64)
        self.linear_residual_feature_names_ = None
        self.linear_residual_dropped_features_ = []
        self.linear_residual_intercept_ = 0.0
        self.linear_residual_coef_ = np.empty(0, dtype=np.float64)
        self.linear_residual_transformed_coef_ = np.empty(0, dtype=np.float64)
        self.linear_residual_center_ = np.empty(0, dtype=np.float64)
        self.linear_residual_scale_ = np.empty(0, dtype=np.float64)
        self.linear_residual_impute_values_ = np.empty(0, dtype=np.float64)
        self.linear_residual_rank_ = 0
        self.linear_residual_singular_values_ = np.empty(0, dtype=np.float64)
        self.linear_residual_weight_sum_ = 0.0
        self.linear_residual_positive_weight_n_ = 0
        self.linear_residual_effective_n_ = 0.0
        self.linear_residual_target_mean_ = 0.0
        self.linear_residual_trend_train_mean_ = 0.0
        self.linear_residual_residual_stats_ = {}

    def _sync_linear_residual_state(self, trend, *, enabled):
        self.linear_residual_enabled_ = bool(enabled)
        self.linear_residual_trend_ = trend
        self.linear_residual_active_ = bool(getattr(trend, "active_", False))
        self.linear_residual_inactive_reason_ = getattr(
            trend, "inactive_reason_", None
        )
        self.linear_residual_alpha_ = float(getattr(trend, "alpha_", trend.alpha))
        self.linear_residual_fit_intercept_ = bool(
            getattr(trend, "fit_intercept_", trend.fit_intercept)
        )
        self.linear_residual_standardize_ = bool(
            getattr(trend, "standardize_", trend.standardize)
        )
        self.linear_residual_feature_indices_ = np.asarray(
            getattr(trend, "feature_indices_", np.empty(0, dtype=np.int64)),
            dtype=np.int64,
        )
        feature_names = getattr(trend, "feature_names_", None)
        self.linear_residual_feature_names_ = (
            None if feature_names is None
            else np.asarray(feature_names, dtype=object)
        )
        self.linear_residual_dropped_features_ = list(
            getattr(trend, "dropped_features_", [])
        )
        self.linear_residual_intercept_ = float(
            getattr(trend, "intercept_", 0.0)
        )
        self.linear_residual_coef_ = np.asarray(
            getattr(trend, "coef_", np.empty(0, dtype=np.float64)),
            dtype=np.float64,
        )
        self.linear_residual_transformed_coef_ = np.asarray(
            getattr(trend, "transformed_coef_", np.empty(0, dtype=np.float64)),
            dtype=np.float64,
        )
        self.linear_residual_center_ = np.asarray(
            getattr(trend, "center_", np.empty(0, dtype=np.float64)),
            dtype=np.float64,
        )
        self.linear_residual_scale_ = np.asarray(
            getattr(trend, "scale_", np.empty(0, dtype=np.float64)),
            dtype=np.float64,
        )
        self.linear_residual_impute_values_ = np.asarray(
            getattr(trend, "impute_values_", np.empty(0, dtype=np.float64)),
            dtype=np.float64,
        )
        self.linear_residual_rank_ = int(getattr(trend, "rank_", 0))
        self.linear_residual_singular_values_ = np.asarray(
            getattr(trend, "singular_values_", np.empty(0, dtype=np.float64)),
            dtype=np.float64,
        )
        self.linear_residual_weight_sum_ = float(
            getattr(trend, "weight_sum_", 0.0)
        )
        self.linear_residual_positive_weight_n_ = int(
            getattr(trend, "positive_weight_n_", 0)
        )
        self.linear_residual_effective_n_ = float(
            getattr(trend, "effective_n_", 0.0)
        )
        self.linear_residual_target_mean_ = float(
            getattr(trend, "target_mean_", 0.0)
        )
        self.linear_residual_trend_train_mean_ = float(
            getattr(trend, "trend_train_mean_", 0.0)
        )
        self.linear_residual_residual_stats_ = dict(
            getattr(trend, "residual_stats_", {})
        )

    def _fit_linear_residual_trend(
        self, X, y, sample_weight, cat_features, feature_names
    ):
        if not _should_use_linear_residual(
            getattr(self, "linear_residual", False)
        ):
            self._set_linear_residual_disabled_state()
            return y
        validate_linear_residual_loss(self.loss)
        trend = WeightedRidgeTrend(
            alpha=getattr(self, "linear_residual_alpha", 1.0),
            features=getattr(self, "linear_residual_features", "auto"),
            fit_intercept=getattr(self, "linear_residual_fit_intercept", True),
            standardize=getattr(self, "linear_residual_standardize", True),
        )
        trend.fit(
            X, y,
            sample_weight=sample_weight,
            cat_features=cat_features,
            feature_names=feature_names,
        )
        self._sync_linear_residual_state(trend, enabled=True)
        if not getattr(trend, "active_", False):
            return y
        return trend.residualize(X, y)

    def _linear_residual_metadata(self):
        enabled = bool(getattr(self, "linear_residual_enabled_", False))
        if not enabled:
            return {
                "version": 1,
                "enabled": False,
                "active": False,
                "inactive_reason": getattr(
                    self, "linear_residual_inactive_reason_", "disabled"
                ),
                "predictive_variance_policy": "residual_only",
                "beta_uncertainty_included": False,
            }
        trend = getattr(self, "linear_residual_trend_", None)
        if trend is not None:
            metadata = trend.summary()
        else:
            metadata = {
                "version": 1,
                "active": bool(getattr(self, "linear_residual_active_", False)),
                "inactive_reason": getattr(
                    self, "linear_residual_inactive_reason_", None
                ),
                "alpha": float(getattr(self, "linear_residual_alpha_", 1.0)),
                "fit_intercept": bool(
                    getattr(self, "linear_residual_fit_intercept_", True)
                ),
                "standardize": bool(
                    getattr(self, "linear_residual_standardize_", True)
                ),
                "n_features": int(
                    len(getattr(self, "linear_residual_feature_indices_", []))
                ),
                "feature_indices": [
                    int(idx)
                    for idx in getattr(
                        self, "linear_residual_feature_indices_", []
                    )
                ],
                "feature_names": (
                    None
                    if getattr(self, "linear_residual_feature_names_", None)
                    is None
                    else [
                        str(name)
                        for name in self.linear_residual_feature_names_
                    ]
                ),
                "dropped_features": list(
                    getattr(self, "linear_residual_dropped_features_", [])
                ),
                "rank": int(getattr(self, "linear_residual_rank_", 0)),
                "weight_sum": float(
                    getattr(self, "linear_residual_weight_sum_", 0.0)
                ),
                "positive_weight_n": int(
                    getattr(self, "linear_residual_positive_weight_n_", 0)
                ),
                "effective_n": float(
                    getattr(self, "linear_residual_effective_n_", 0.0)
                ),
                "target_mean": float(
                    getattr(self, "linear_residual_target_mean_", 0.0)
                ),
                "trend_train_mean": float(
                    getattr(self, "linear_residual_trend_train_mean_", 0.0)
                ),
                "residual_stats": dict(
                    getattr(self, "linear_residual_residual_stats_", {})
                ),
            }
        metadata["enabled"] = True
        metadata["predictive_variance_policy"] = "residual_only"
        metadata["beta_uncertainty_included"] = False
        return metadata

    def _attach_linear_residual_metadata(self):
        model = getattr(self, "model_", None)
        auto_params = getattr(model, "auto_params_", None)
        if auto_params is None:
            return
        metadata = self._linear_residual_metadata()
        selection_summary = getattr(
            self, "selection_linear_residual_summary_", None
        )
        if selection_summary is not None:
            metadata = dict(metadata)
            metadata["selection_summary"] = selection_summary
        auto_params["linear_residual"] = metadata
        auto_params.setdefault("diagnostics", {})
        auto_params["diagnostics"]["linear_residual"] = metadata

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
        selection_rounds = _normalize_selection_rounds(
            getattr(self, "selection_rounds", None)
        )
        if selection_rounds is not None and not _is_auto_tree_mode(
            self.tree_mode
        ):
            raise ValueError(
                "selection_rounds currently requires tree_mode='auto'"
            )
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
        sample_weight, eval_sample_weight, callbacks=()
    ):
        selection_rounds = _normalize_selection_rounds(
            getattr(self, "selection_rounds", None)
        )
        requested_iterations = fit_kwargs.get("iterations", self.iterations)
        cap_active = (
            selection_rounds is not None
            and selection_rounds < requested_iterations
        )
        results = []
        best_model = None
        best_score = np.inf
        best_probe_metadata = None
        best_candidate_index = None
        selected_lane = (
            "linear_residual"
            if bool(getattr(self, "linear_residual_active_", False))
            else "boosting"
        )

        for tree_mode in _AUTO_TREE_MODE_CANDIDATES:
            deadline_before = _wall_clock_callback_state(
                callbacks, refresh_deadline=bool(results)
            )
            if results and deadline_before["deadline_hit"]:
                elapsed = deadline_before["wall_clock_elapsed_seconds"]
                results.append({
                    "tree_mode": tree_mode,
                    "fit_status": "skipped_deadline",
                    "score": None,
                    "validation_score": None,
                    "selected": False,
                    "lane": selected_lane,
                    "iterations_requested": int(
                        fit_kwargs.get("iterations", self.iterations)
                    ),
                    "iterations_attempted": 0,
                    "rounds_completed": 0,
                    "rounds_retained": 0,
                    "best_iteration": None,
                    "best_prefix_round": None,
                    "n_estimators": 0,
                    "learning_rate": None,
                    "resolved_learning_rate": None,
                    "stop_reason": "time_limit",
                    "wall_clock_stopper_count": int(
                        deadline_before["wall_clock_stopper_count"]
                    ),
                    "wall_clock_elapsed_seconds_start": elapsed,
                    "wall_clock_elapsed_seconds_end": elapsed,
                    "wall_clock_elapsed_seconds": elapsed,
                    "deadline_hit_start": True,
                    "deadline_hit_end": True,
                    "deadline_hit": True,
                    "probe": {
                        "enabled": False,
                        "reason": "skipped_deadline",
                    },
                })
                continue
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
            candidate_callbacks = callbacks
            if cap_active:
                candidate_callbacks = (
                    *callbacks,
                    _SelectionRoundStopper(selection_rounds),
                )
            model.fit(
                X, y, cat_features=cat_features, eval_set=eval_set,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight,
                callbacks=candidate_callbacks,
            )
            deadline_after = _wall_clock_callback_state(callbacks)
            score = self._tree_mode_selection_score(model)
            training = getattr(model, "training_metadata_", {})
            retained = int(
                training.get("rounds_retained", len(model.trees_))
            )
            completed = int(
                training.get(
                    "rounds_completed",
                    max(retained, len(getattr(model, "valid_history_", ()))),
                )
            )
            results.append({
                "tree_mode": tree_mode,
                "fit_status": "fitted",
                "score": score,
                "validation_score": score,
                "selected": False,
                "lane": selected_lane,
                "iterations_requested": int(
                    training.get(
                        "iterations_requested",
                        candidate_kwargs.get("iterations", self.iterations),
                    )
                ),
                "iterations_attempted": int(
                    training.get(
                        "iterations_attempted",
                        getattr(model, "iterations_attempted_", completed),
                    )
                ),
                "rounds_completed": completed,
                "rounds_retained": retained,
                "best_iteration": int(model.best_iteration_),
                "best_prefix_round": training.get("best_prefix_round"),
                "n_estimators": len(model.trees_),
                "learning_rate": float(model.lr_),
                "resolved_learning_rate": float(model.lr_),
                "stop_reason": str(
                    training.get(
                        "stop_reason", getattr(model, "stop_reason_", "unknown")
                    )
                ),
                "wall_clock_stopper_count": int(
                    deadline_after["wall_clock_stopper_count"]
                ),
                "wall_clock_elapsed_seconds_start": deadline_before[
                    "wall_clock_elapsed_seconds"
                ],
                "wall_clock_elapsed_seconds_end": deadline_after[
                    "wall_clock_elapsed_seconds"
                ],
                "wall_clock_elapsed_seconds": deadline_after[
                    "wall_clock_elapsed_seconds"
                ],
                "deadline_hit_start": bool(deadline_before["deadline_hit"]),
                "deadline_hit_end": bool(deadline_after["deadline_hit"]),
                "deadline_hit": bool(deadline_after["deadline_hit"]),
                "probe": probe_metadata,
            })
            if score < best_score:
                best_score = score
                best_model = model
                best_probe_metadata = probe_metadata
                best_candidate_index = len(results) - 1

        if best_model is None:
            raise ValueError(
                "tree_mode='auto' could not select a model because all "
                "candidate scores were non-finite"
            )
        selected = getattr(best_model, "tree_mode_", None)
        results[best_candidate_index]["selected"] = True
        audition_model = best_model
        final_refit_performed = False
        final_refit_status = "not_requested"
        if cap_active:
            deadline_before_refit = _wall_clock_callback_state(
                callbacks, refresh_deadline=True
            )
            if deadline_before_refit["deadline_hit"]:
                final_refit_status = "skipped_deadline"
            else:
                final_kwargs = self._tree_mode_candidate_kwargs(
                    fit_kwargs, selected
                )
                selected_probe_lr = (
                    None
                    if best_probe_metadata is None
                    else best_probe_metadata.get("selected_learning_rate")
                )
                if selected_probe_lr is not None:
                    final_kwargs["learning_rate"] = selected_probe_lr
                best_model = make_model(final_kwargs)
                best_model.fit(
                    X, y, cat_features=cat_features, eval_set=eval_set,
                    sample_weight=sample_weight,
                    eval_sample_weight=eval_sample_weight,
                    callbacks=callbacks,
                )
                final_refit_performed = True
                final_refit_status = "fitted"
        deadline_final = _wall_clock_callback_state(callbacks)
        fit_status_counts = {
            status: sum(
                result["fit_status"] == status for result in results
            )
            for status in ("fitted", "skipped_deadline")
        }
        metadata = {
            "enabled": True,
            "input": self.tree_mode,
            "candidates": results,
            "selected_tree_mode": selected,
            "selected_lane": selected_lane,
            "selected_candidate_index": int(best_candidate_index),
            "selected_score": float(best_score),
            "candidate_count": len(results),
            "fitted_candidate_count": fit_status_counts["fitted"],
            "skipped_deadline_candidate_count": fit_status_counts[
                "skipped_deadline"
            ],
            "candidate_fit_status_counts": fit_status_counts,
            "wall_clock_stopper_count": int(
                deadline_final["wall_clock_stopper_count"]
            ),
            "wall_clock_elapsed_seconds": deadline_final[
                "wall_clock_elapsed_seconds"
            ],
            "deadline_hit": bool(deadline_final["deadline_hit"]),
        }
        if selection_rounds is not None:
            final_training = getattr(best_model, "training_metadata_", {})
            metadata.update({
                "selection_rounds": selection_rounds,
                "selection_cap_active": cap_active,
                "final_refit_performed": final_refit_performed,
                "final_refit_status": final_refit_status,
                "final_iterations_requested": int(requested_iterations),
                "audition_selected_rounds_retained": len(
                    audition_model.trees_
                ),
                "final_rounds_retained": len(best_model.trees_),
                "final_stop_reason": str(
                    final_training.get(
                        "stop_reason",
                        getattr(best_model, "stop_reason_", "unknown"),
                    )
                ),
            })
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
        ordinal_mode = getattr(self, "ordinal_features_mode_", "off")
        if ordinal_mode != "off":
            state["ordinal_features_mode"] = ordinal_mode
            state["ordinal_features"] = list(self.ordinal_features_)
        if getattr(self, "refit_", False):
            state["selection_model_persisted"] = False
        if hasattr(self, "tree_mode_selection_"):
            state["tree_mode_selection"] = self.tree_mode_selection_
        if hasattr(self, "preset_"):
            state["preset"] = self.preset_
            state["preset_params"] = dict(self.preset_params_)
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
            if hasattr(self, "dist_group_affine_metadata_"):
                state["dist_group_affine"] = self.dist_group_affine_metadata_
                state["dist_calibration_feature"] = getattr(
                    self, "dist_calibration_feature_", None
                )
                state["dist_calibration_feature_index"] = int(
                    self.dist_calibration_feature_index_
                )
                state["dist_calibration_feature_name"] = getattr(
                    self, "dist_calibration_feature_name_", None
                )
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
            if hasattr(self, "dist_group_affine_metadata_"):
                state["sigma_group_affine"] = self.dist_group_affine_metadata_
            if fallback_reason is not None:
                state["sigma_calibration_fallback_reason"] = fallback_reason
        if hasattr(self, "linear_residual_enabled_"):
            state["linear_residual_enabled"] = bool(
                self.linear_residual_enabled_
            )
            trend = getattr(self, "linear_residual_trend_", None)
            if trend is not None:
                state.update(trend.state_header())
            else:
                state.update({
                    "linear_residual_version": 1,
                    "linear_residual_active": bool(
                        getattr(self, "linear_residual_active_", False)
                    ),
                    "linear_residual_alpha": float(
                        getattr(self, "linear_residual_alpha_", 1.0)
                    ),
                    "linear_residual_fit_intercept": bool(
                        getattr(self, "linear_residual_fit_intercept_", True)
                    ),
                    "linear_residual_standardize": bool(
                        getattr(self, "linear_residual_standardize_", True)
                    ),
                    "linear_residual_inactive_reason": getattr(
                        self, "linear_residual_inactive_reason_", None
                    ),
                    "linear_residual_feature_names": (
                        None
                        if getattr(self, "linear_residual_feature_names_", None)
                        is None
                        else [
                            str(name)
                            for name in self.linear_residual_feature_names_
                        ]
                    ),
                    "linear_residual_dropped_features": list(
                        getattr(self, "linear_residual_dropped_features_", [])
                    ),
                    "linear_residual_intercept": float(
                        getattr(self, "linear_residual_intercept_", 0.0)
                    ),
                    "linear_residual_rank": int(
                        getattr(self, "linear_residual_rank_", 0)
                    ),
                    "linear_residual_weight_sum": float(
                        getattr(self, "linear_residual_weight_sum_", 0.0)
                    ),
                    "linear_residual_positive_weight_n": int(
                        getattr(self, "linear_residual_positive_weight_n_", 0)
                    ),
                    "linear_residual_effective_n": float(
                        getattr(self, "linear_residual_effective_n_", 0.0)
                    ),
                    "linear_residual_target_mean": float(
                        getattr(self, "linear_residual_target_mean_", 0.0)
                    ),
                    "linear_residual_trend_train_mean": float(
                        getattr(self, "linear_residual_trend_train_mean_", 0.0)
                    ),
                    "linear_residual_residual_stats": dict(
                        getattr(self, "linear_residual_residual_stats_", {})
                    ),
                    "linear_residual_prediction_mode": "additive_location",
                    "linear_residual_beta_uncertainty_included": False,
                })
            selection_summary = getattr(
                self, "selection_linear_residual_summary_", None
            )
            if selection_summary is not None:
                state["selection_linear_residual_summary"] = selection_summary
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

    def _wrapper_arrays(self):
        arrays = {}
        if not getattr(self, "linear_residual_active_", False):
            return arrays
        trend = getattr(self, "linear_residual_trend_", None)
        if trend is not None:
            arrays.update(trend.state_arrays())
            return arrays
        arrays["linear_residual_feature_indices"] = np.asarray(
            self.linear_residual_feature_indices_, dtype=np.int64
        )
        arrays["linear_residual_coef"] = np.asarray(
            self.linear_residual_coef_, dtype=np.float64
        )
        arrays["linear_residual_transformed_coef"] = np.asarray(
            self.linear_residual_transformed_coef_, dtype=np.float64
        )
        arrays["linear_residual_center"] = np.asarray(
            self.linear_residual_center_, dtype=np.float64
        )
        arrays["linear_residual_scale"] = np.asarray(
            self.linear_residual_scale_, dtype=np.float64
        )
        arrays["linear_residual_impute_values"] = np.asarray(
            self.linear_residual_impute_values_, dtype=np.float64
        )
        arrays["linear_residual_singular_values"] = np.asarray(
            self.linear_residual_singular_values_, dtype=np.float64
        )
        return arrays

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
        self._restore_ordinal_state(state)
        self.refit_ = bool(state.get("refit", False))
        self.refit_n_estimators_ = state.get("refit_n_estimators")
        self.refit_strategy_ = state.get("refit_strategy")
        if "tree_mode_selection" in state:
            self.tree_mode_selection_ = state["tree_mode_selection"]
        if "preset" in state:
            preset = _normalize_regression_preset(state["preset"])
            preset_params = state.get("preset_params")
            if preset is None or not isinstance(preset_params, Mapping):
                raise ValueError(
                    "invalid DarkoFit model: preset state is invalid"
                )
            expected = _ACCURACY_PRESET_PARAMS if preset == "accuracy" else {}
            if dict(preset_params) != expected:
                raise ValueError(
                    "invalid DarkoFit model: preset parameters do not match "
                    "the saved preset"
                )
            self.preset_ = preset
            self.preset_params_ = dict(preset_params)
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
            group_records = state.get(
                "dist_group_affine", state.get("sigma_group_affine")
            )
            if group_records is not None:
                self.dist_group_affine_metadata_ = list(group_records)
                self.dist_group_affine_groups_ = np.asarray(
                    [
                        _calibration_group_key(record["group"])
                        for record in self.dist_group_affine_metadata_
                    ],
                    dtype=object,
                )
                self.dist_group_affine_a_ = np.asarray(
                    [
                        float(record["sigma_affine_a"])
                        for record in self.dist_group_affine_metadata_
                    ],
                    dtype=np.float64,
                )
                self.dist_group_affine_b_ = np.asarray(
                    [
                        float(record["sigma_affine_b"])
                        for record in self.dist_group_affine_metadata_
                    ],
                    dtype=np.float64,
                )
                self.sigma_group_affine_groups_ = self.dist_group_affine_groups_
                self.sigma_group_affine_a_ = self.dist_group_affine_a_
                self.sigma_group_affine_b_ = self.dist_group_affine_b_
                self.dist_calibration_feature_ = state.get(
                    "dist_calibration_feature", _GROUP_AFFINE_DEFAULT_FEATURE
                )
                self.dist_calibration_feature_index_ = int(
                    state.get("dist_calibration_feature_index", 0)
                )
                feature_name = state.get("dist_calibration_feature_name")
                if feature_name is not None:
                    self.dist_calibration_feature_name_ = feature_name
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

    def _restore_linear_residual_state(self, state, wrapper_arrays):
        state = dict(state or {})
        if (
            "linear_residual_enabled" not in state
            and "linear_residual_active" not in state
        ):
            self._set_linear_residual_disabled_state()
            return
        enabled = bool(
            state.get(
                "linear_residual_enabled",
                state.get("linear_residual_active", False),
            )
        )
        state["linear_residual_enabled"] = enabled
        active = bool(state.get("linear_residual_active", False))
        if active and not enabled:
            raise ValueError(
                "invalid DarkoFit model: active linear residual state "
                "cannot be disabled"
            )
        if active:
            validate_linear_residual_loss(getattr(self, "loss", "RMSE"))
        trend = WeightedRidgeTrend.from_payload(
            state,
            wrapper_arrays or {},
            n_features=getattr(self, "n_features_in_", None),
        )
        if (
            active
            and getattr(trend, "feature_names_", None) is not None
            and hasattr(self, "feature_names_in_")
        ):
            expected = np.asarray(self.feature_names_in_, dtype=object)[
                trend.feature_indices_
            ]
            if not np.array_equal(
                np.asarray([str(name) for name in expected], dtype=object),
                trend.feature_names_,
            ):
                raise ValueError(
                    "invalid DarkoFit model: linear residual feature names "
                    "do not match saved feature indices"
                )
        self._sync_linear_residual_state(trend, enabled=enabled)
        if "selection_linear_residual_summary" in state:
            self.selection_linear_residual_summary_ = state[
                "selection_linear_residual_summary"
            ]
        self._attach_linear_residual_metadata()

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
        if hasattr(self, "dist_group_affine_metadata_"):
            metadata["dist_group_affine"] = self.dist_group_affine_metadata_
            metadata["sigma_group_affine"] = self.dist_group_affine_metadata_
            metadata["group_count"] = len(self.dist_group_affine_metadata_)
            metadata["group_fallback_count"] = sum(
                1
                for record in self.dist_group_affine_metadata_
                if record.get("fallback_reason") is not None
            )
            metadata["dist_calibration_feature"] = getattr(
                self, "dist_calibration_feature_", None
            )
            metadata["dist_calibration_feature_index"] = int(
                self.dist_calibration_feature_index_
            )
            feature_name = getattr(self, "dist_calibration_feature_name_", None)
            if feature_name is not None:
                metadata["dist_calibration_feature_name"] = feature_name
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
                    "DarkoFit "
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
                    "DarkoFit sigma calibration is dominated by the "
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
        # Known caveat: candidates are scored at the short probe budget
        # (default 80 rounds) but deployed at the full iteration budget, which
        # biases selection toward larger rates that lead early. Treat the
        # probe as a coarse sanity check, not a tuned-rate substitute.
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
        if hasattr(self, "estimators_"):
            raise ValueError(
                "get_refit_params() is not defined for OOB ensembles; each "
                "member selected its own boosting horizon"
            )

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
        if hasattr(self, "preset_params_"):
            params.update(self.preset_params_)
            params["preset"] = None
        params["iterations"] = max(0, rounds)
        params["learning_rate"] = self.learning_rate_
        selected_tree_mode = getattr(self.model_, "tree_mode_", None)
        if selected_tree_mode is not None:
            params["tree_mode"] = selected_tree_mode
            if "selection_rounds" in params:
                params["selection_rounds"] = None
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
        if bool(params.get("linear_leaves", False)):
            linear_metadata = auto.get("linear_leaves", {})
            if not (
                isinstance(linear_metadata, dict)
                and bool(linear_metadata.get("active", False))
            ):
                # A full-data refit must not activate a model family that the
                # validation/selection fit only evaluated through its exact
                # constant-leaf fallback (most notably around the 1,000-row
                # eligibility boundary).
                params["linear_leaves"] = False
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
        if hasattr(self, "estimators_"):
            return int(np.median([
                len(member.model_.trees_) for member in self.estimators_
            ]))
        return len(self.model_.trees_)

    @property
    def learning_rate_(self):
        """Resolved learning rate used by the fitted booster."""
        check_is_fitted(self, "model_")
        return getattr(self, "_learning_rate_", self.model_.lr_)


class DarkoRegressor(RegressorMixin, _RefitParamsMixin, BaseEstimator):
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
    linear_leaves : bool, default False
        Experimental local-linear leaves for scalar RMSE CatBoost-mode fits.
        The option is never selected automatically; small and all-categorical
        fits retain exact constant leaves and record the fallback reason.
    linear_lambda : float, default 1.0
        Nonnegative ridge penalty applied to local-linear slopes. The regular
        leaf ``l2_leaf_reg`` remains the intercept penalty.
    preset : {None, "accuracy"}, default None
        Optional profile. ``"accuracy"`` applies the frozen A10 development
        configuration during fit without changing the conservative default.
        Explicit parameters outside the managed A10 fields remain in effect.
    selection_rounds : int or None, default None
        Optional cap for each ``tree_mode="auto"`` audition. The selected
        mode is then fit from scratch with the full requested round budget.
    n_ensembles : int, default 1
        Number of OOB-selected bootstrap members. Values above one opt into
        mean aggregation. ``ensemble_bootstrap="groups"`` requires ``groups``
        in :meth:`fit`; numeric-only members may safely share target-free
        preprocessing.
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
                 eval_train_loss=False, bin_sample_count=200_000,
                 histogram_parallelism="auto", use_best_model=True,
                 bootstrap_type="none", bagging_temperature=0.0,
                 mvs_reg=1.0, random_strength=0.0,
                 dist_calibration=None,
                 dist_calibration_feature=_GROUP_AFFINE_DEFAULT_FEATURE,
                 dist_params=None,
                 sigma_calibration=None,
                 linear_residual=False,
                 linear_residual_alpha=1.0,
                 linear_residual_features="auto",
                 linear_residual_fit_intercept=True,
                 linear_residual_standardize=True,
                 diagnostic_warnings="once",
                 auto_learning_rate_probe=False,
                 auto_learning_rate_probe_values=None,
                 auto_learning_rate_probe_iterations=80,
                 histogram_dtype="float64",
                 leaf_dtype="int64",
                 ts_permutations=1,
                 target_ordered_cat_codes="off",
                 rho_learning_rate_multiplier=1.0,
                 rho_l2_leaf_reg_multiplier=1.0,
                 linear_leaves=False, linear_lambda=1.0,
                 preset=None, selection_rounds=None, n_ensembles=1,
                 ensemble_bootstrap="rows",
                 ensemble_shared_preprocessing=True):
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
        self.dist_calibration_feature = dist_calibration_feature
        self.dist_params = dist_params
        self.sigma_calibration = sigma_calibration
        self.linear_residual = linear_residual
        self.linear_residual_alpha = linear_residual_alpha
        self.linear_residual_features = linear_residual_features
        self.linear_residual_fit_intercept = linear_residual_fit_intercept
        self.linear_residual_standardize = linear_residual_standardize
        self.diagnostic_warnings = diagnostic_warnings
        self.histogram_dtype = histogram_dtype
        self.leaf_dtype = leaf_dtype
        self.ts_permutations = ts_permutations
        self.target_ordered_cat_codes = target_ordered_cat_codes
        self.rho_learning_rate_multiplier = rho_learning_rate_multiplier
        self.rho_l2_leaf_reg_multiplier = rho_l2_leaf_reg_multiplier
        self.linear_leaves = linear_leaves
        self.linear_lambda = linear_lambda
        self.preset = preset
        self.selection_rounds = selection_rounds
        self.n_ensembles = n_ensembles
        self.ensemble_bootstrap = ensemble_bootstrap
        self.ensemble_shared_preprocessing = ensemble_shared_preprocessing
        self.auto_learning_rate_probe = auto_learning_rate_probe
        self.auto_learning_rate_probe_values = auto_learning_rate_probe_values
        self.auto_learning_rate_probe_iterations = auto_learning_rate_probe_iterations

    def fit(self, X, y, cat_features=None, eval_set=None, groups=None,
            sample_weight=None, eval_sample_weight=None, callbacks=None,
            ordinal_features=None):
        """Fit the model, resolving any opt-in product preset."""
        self._clear_ensemble_state()
        if not getattr(self, "_suppress_wrapper_deprecation_warning", False):
            self._warn_wrapper_deprecated_options()
        preset = _normalize_regression_preset(self.preset)
        self._clear_preset_state()
        if preset is None:
            return self._fit_resolved(
                X, y, cat_features=cat_features, eval_set=eval_set,
                groups=groups, sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight, callbacks=callbacks,
                ordinal_features=ordinal_features,
            )
        if self.loss != "RMSE":
            raise ValueError(
                "preset='accuracy' currently requires loss='RMSE'"
            )
        original = {
            name: getattr(self, name) for name in _ACCURACY_PRESET_PARAMS
        }
        try:
            for name, value in _ACCURACY_PRESET_PARAMS.items():
                setattr(self, name, value)
            fitted = self._fit_resolved(
                X, y, cat_features=cat_features, eval_set=eval_set,
                groups=groups, sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight, callbacks=callbacks,
                ordinal_features=ordinal_features,
            )
        finally:
            for name, value in original.items():
                setattr(self, name, value)
        self.preset_ = preset
        self.preset_params_ = dict(_ACCURACY_PRESET_PARAMS)
        self._attach_preset_metadata()
        return fitted

    def _fit_resolved(self, X, y, cat_features=None, eval_set=None,
                      groups=None, sample_weight=None,
                      eval_sample_weight=None, callbacks=None,
                      ordinal_features=None):
        """Fit the model.

        Parameters
        ----------
        X, y : array-like
            Training data.
        cat_features : list of int or str, or None
            Column indices or, for named input, column names to treat as
            categoricals.
        ordinal_features : mapping, {"auto"}, or None
            Ordered categorical features to encode as rank-valued numeric
            columns before binning. A mapping declares each feature's complete
            ordered category sequence. ``"auto"`` recognizes ordered pandas
            categoricals and integer-coded columns listed in *cat_features*.
            Unknown categories at prediction time are rejected.
        eval_set : (X_val, y_val) tuple or None
            Explicit validation set.  When provided, automatic splitting is
            skipped regardless of the *early_stopping* setting.
        groups : array-like of shape (n_samples,) or None
            Group labels for the samples (e.g. ``df['subject_id']``).  When
            supplied and *early_stopping* triggers an automatic split, groups
            are kept intact across the train/validation boundary using
            ``GroupShuffleSplit``. Set ``validation_strategy='group'`` to
            require this behavior explicitly.
        sample_weight : array-like of shape (n_samples,) or None
            Per-sample weights.  Normalized to mean 1 internally.
        eval_sample_weight : array-like of shape (n_validation_samples,) or None
            Validation weights used when evaluating early stopping.
        callbacks : callable or iterable of callables, or None
            Fit-time boosting callbacks. Each callback receives a
            :class:`darkofit.callbacks.BoostingProgress` snapshot before the
            next boosting round and may return ``True`` to stop. Automatic
            tree-mode selection shares the same callback objects across its
            candidate fits. Callbacks are not supported with automatic
            learning-rate probes or refitting.
        """
        ensemble = self._fit_ensemble(
            X,
            y,
            cat_features=cat_features,
            eval_set=eval_set,
            groups=groups,
            sample_weight=sample_weight,
            eval_sample_weight=eval_sample_weight,
            callbacks=callbacks,
            ordinal_features=ordinal_features,
            classification=False,
        )
        if ensemble is not None:
            return ensemble
        callbacks = _normalize_callbacks(callbacks)
        X_input = X
        feature_names = _feature_names_from_input(X_input)
        X, cat_features, ordinal_mode, ordinal_records = (
            self._prepare_ordinal_fit_input(
            X, cat_features, ordinal_features
            )
        )
        X, cat_features, n_features = _coerce_fit_X(X, cat_features)
        ordinal_nominal_cat_count = len(cat_features)
        eval_set = _ensure_dense_eval_set(eval_set)
        if eval_set is not None:
            _validate_eval_set_feature_schema(
                eval_set[0],
                n_features,
                expected_feature_names=feature_names,
            )
            eval_set = (
                _transform_ordinal_features(eval_set[0], ordinal_records),
                eval_set[1],
            )
        eval_set = _validate_eval_set_features(
            eval_set, n_features,
            expected_feature_names=feature_names,
            cat_features=cat_features,
            feature_names_validated=True,
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
        self._clear_linear_residual_state()
        tree_mode_auto = _is_auto_tree_mode(self.tree_mode)
        distributional_loss = _is_distributional_loss(self.loss)
        if callbacks:
            if self.refit:
                raise ValueError("callbacks are not supported with refit=True")
            if self.auto_learning_rate_probe:
                raise ValueError(
                    "callbacks are not supported with "
                    "auto_learning_rate_probe=True"
                )
        linear_residual_enabled = _should_use_linear_residual(
            self.linear_residual
        )
        if not isinstance(self.linear_leaves, (bool, np.bool_)):
            raise TypeError("linear_leaves must be a bool")
        if self.linear_leaves:
            if self.loss != "RMSE":
                raise ValueError(
                    "linear_leaves=True is currently supported only for "
                    "loss='RMSE'"
                )
            if tree_mode_auto or _normalize_tree_mode(self.tree_mode) != "catboost":
                raise ValueError(
                    "linear_leaves=True currently requires "
                    "tree_mode='catboost'"
                )
        if linear_residual_enabled:
            validate_linear_residual_loss(self.loss)
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
            if (
                dist_calibration_ in {"affine", "per_metric_affine"}
                and "scale" not in targets
            ):
                raise ValueError(
                    f"dist_calibration={dist_calibration_!r} is not supported for "
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
        if validation_strategy_ == "group" and groups is None:
            raise ValueError("validation_strategy='group' requires groups")
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

        y = self._fit_linear_residual_trend(
            X, y, sample_weight, cat_features, feature_names
        )
        if eval_set is not None and self.linear_residual_active_:
            X_eval, y_eval = eval_set
            eval_set = (
                X_eval,
                self.linear_residual_trend_.residualize(X_eval, y_eval),
            )

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
        # Refit can only reuse the selection fit's preprocessing when both
        # train on the same rows, i.e. with an explicit eval set; automatic
        # splits refit on different (full) data, where caching would only
        # add copies.
        refit_can_reuse_preprocessing = (
            bool(self.refit) and selection_active and explicit_eval_set
        )
        preprocessing_cache = (
            {}
            if (
                tree_mode_auto
                or self.auto_learning_rate_probe
                or refit_can_reuse_preprocessing
            )
            else None
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
            return self._configure_model_preprocessing(model)

        tree_mode_selection_metadata = None
        if tree_mode_auto:
            model, probe_metadata, tree_mode_selection_metadata = (
                self._fit_tree_mode_auto(
                    make_model, kw, X, y,
                    cat_features=cat_features,
                    eval_set=eval_set,
                    sample_weight=sample_weight,
                    eval_sample_weight=eval_sample_weight,
                    callbacks=callbacks,
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
                callbacks=callbacks,
            )
        self.model_ = model
        self._set_ordinal_fit_state(
            ordinal_mode, ordinal_records, ordinal_nominal_cat_count
        )
        self._attach_ordinal_metadata()
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
        self._attach_linear_residual_metadata()
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
                    elif dist_calibration_ == "per_metric_affine":
                        feature_names = _feature_names_from_input(X_input)
                        feature_index, feature_name, feature_spec = (
                            _resolve_dist_calibration_feature(
                                self.dist_calibration_feature,
                                feature_names,
                                n_features,
                            )
                        )
                        groups = _extract_feature_column_by_index(
                            X_cal, feature_index
                        )
                        calibration = _fit_grouped_affine_sigma_calibration(
                            selection_model, X_cal, y_cal, groups,
                            eval_sample_weight,
                            fold_stats=self.sigma_calibration_fold_stats_,
                        )
                        self.dist_scale_ = calibration["sigma_scale"]
                        self.sigma_scale_ = self.dist_scale_
                        self.dist_affine_a_ = calibration["sigma_affine_a"]
                        self.dist_affine_b_ = calibration["sigma_affine_b"]
                        self.sigma_affine_a_ = self.dist_affine_a_
                        self.sigma_affine_b_ = self.dist_affine_b_
                        records = list(calibration["group_affine"])
                        self.dist_group_affine_metadata_ = records
                        self.dist_group_affine_groups_ = np.asarray(
                            [
                                _calibration_group_key(record["group"])
                                for record in records
                            ],
                            dtype=object,
                        )
                        self.dist_group_affine_a_ = np.asarray(
                            [
                                float(record["sigma_affine_a"])
                                for record in records
                            ],
                            dtype=np.float64,
                        )
                        self.dist_group_affine_b_ = np.asarray(
                            [
                                float(record["sigma_affine_b"])
                                for record in records
                            ],
                            dtype=np.float64,
                        )
                        self.sigma_group_affine_groups_ = (
                            self.dist_group_affine_groups_
                        )
                        self.sigma_group_affine_a_ = self.dist_group_affine_a_
                        self.sigma_group_affine_b_ = self.dist_group_affine_b_
                        self.dist_calibration_feature_ = feature_spec
                        self.dist_calibration_feature_index_ = int(feature_index)
                        if feature_name is not None:
                            self.dist_calibration_feature_name_ = feature_name
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
            selection_linear_residual_summary = (
                self._linear_residual_metadata()
                if linear_residual_enabled else None
            )
            selection_linear_leaf_metadata = dict(
                selection_model.auto_params_.get("linear_leaves", {}) or {}
            )
            refit_kw = self._refit_params_for_booster(self.refit_strategy)
            refit_model = make_model(refit_kw)
            y_full_refit = self._fit_linear_residual_trend(
                X_full, y_full, sample_weight_full, cat_features,
                feature_names,
            )
            refit_model.fit(
                X_full, y_full_refit, cat_features=cat_features,
                sample_weight=sample_weight_full,
            )
            self.model_ = refit_model
            if selection_linear_residual_summary is not None:
                self.selection_linear_residual_summary_ = (
                    selection_linear_residual_summary
                )
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
            self._attach_linear_residual_metadata()
            self._attach_dist_calibration_metadata()
            if selection_linear_leaf_metadata:
                self.model_.auto_params_["selection_linear_leaves"] = (
                    selection_linear_leaf_metadata
                )
                self.model_.auto_params_.setdefault("diagnostics", {})
                self.model_.auto_params_["diagnostics"][
                    "selection_linear_leaves"
                ] = selection_linear_leaf_metadata
        if preprocessing_cache is not None:
            # Free the binned-matrix copies; fitted models keep a reference
            # to this dict, so an unemptied cache would pin them in memory.
            preprocessing_cache.clear()
        self._attach_ordinal_metadata()
        return self

    def _linear_residual_trend(self, X):
        if not getattr(self, "linear_residual_active_", False):
            return None
        return self.linear_residual_trend_.predict(X)

    def _check_fitted_loss_matches_params(self, method_name):
        model = getattr(self, "model_", None)
        fitted_loss = getattr(model, "loss_name", None)
        if fitted_loss is None:
            return
        current_loss = getattr(self, "loss", fitted_loss)
        if current_loss != fitted_loss:
            raise ValueError(
                f"{method_name}() cannot be used after changing loss from "
                f"the fitted loss {fitted_loss!r} to {current_loss!r}; "
                "refit the estimator after set_params(loss=...)."
            )

    def _fitted_loss_name(self):
        return getattr(getattr(self, "model_", None), "loss_name", self.loss)

    def _fitted_distributional(self):
        return isinstance(getattr(self, "model_", None), DistributionalBoosting)

    def _linear_residual_shift_params(self, params, trend):
        if trend is None:
            return params
        if self._fitted_loss_name() not in {"Gaussian", "StudentT"}:
            return params
        shifted = list(params)
        shifted[0] = np.asarray(shifted[0], dtype=np.float64) + trend
        return tuple(shifted)

    @staticmethod
    def _linear_residual_shift_interval(interval, trend):
        if trend is None:
            return interval
        lo, hi = interval
        return (
            np.asarray(lo, dtype=np.float64) + trend,
            np.asarray(hi, dtype=np.float64) + trend,
        )

    @staticmethod
    def _linear_residual_shift_samples(samples, trend):
        if trend is None:
            return samples
        return np.asarray(samples, dtype=np.float64) + trend[:, None]

    def predict(self, X):
        X = _check_predict_input(self, X)
        self._check_fitted_loss_matches_params("predict")
        if hasattr(self, "estimators_"):
            predictions = []
            for member in self.estimators_:
                trend = member._linear_residual_trend(X)
                raw = member.model_.predict_raw(X, _validated=True)
                predictions.append(raw if trend is None else raw + trend)
            return np.mean(np.stack(predictions, axis=0), axis=0)
        trend = self._linear_residual_trend(X)
        raw = self.model_.predict_raw(X, _validated=True)
        if self._fitted_distributional():
            loss = self.model_.loss_
            if hasattr(loss, "mean_from_params"):
                params = self._linear_residual_shift_params(
                    self._calibrated_params_from_raw(raw, X), trend
                )
                return loss.mean_from_params(*params)
            return loss.mean_from_raw(raw)
        return raw if trend is None else raw + trend

    def shap_values(self, X, X_background=None):
        """Return exact interventional SHAP contributions to predictions.

        The returned array has shape ``(n_samples, n_features)``. Its rows sum
        to ``predict(X) - expected_value_``. Constant and local-linear
        oblivious leaves are supported; distributional heads and active global
        linear residuals are intentionally outside this scalar explanation.
        """
        X = _check_predict_input(self, X)
        self._check_fitted_loss_matches_params("shap_values")
        if self._fitted_distributional():
            raise NotImplementedError(
                "shap_values() is not implemented for distributional losses"
            )
        if getattr(self, "linear_residual_active_", False):
            raise NotImplementedError(
                "shap_values() is not implemented with an active "
                "linear_residual"
            )
        if X_background is not None:
            X_background = _check_predict_input(self, X_background)
        if hasattr(self, "estimators_"):
            values = []
            expected_values = []
            for member in self.estimators_:
                if getattr(member, "linear_residual_active_", False):
                    raise NotImplementedError(
                        "shap_values() is not implemented when an ensemble "
                        "member has an active linear_residual"
                    )
                contributions, expected_value = member.model_.shap_values(
                    X,
                    background=X_background,
                    _validated=True,
                    _background_validated=X_background is not None,
                )
                values.append(contributions)
                expected_values.append(float(expected_value))
            contributions = np.mean(np.stack(values, axis=0), axis=0)
            expected_value = float(np.mean(expected_values))
        else:
            contributions, expected_value = self.model_.shap_values(
                X,
                background=X_background,
                _validated=True,
                _background_validated=X_background is not None,
            )
        self.expected_value_ = expected_value
        return contributions

    def staged_predict(self, X):
        """Yield the prediction after each successive tree."""
        X = _check_predict_input(self, X)
        self._check_fitted_loss_matches_params("staged_predict")
        if hasattr(self, "estimators_"):
            trends = [
                member._linear_residual_trend(X)
                for member in self.estimators_
            ]
            generators = [
                member.model_.staged_predict_raw(X, _validated=True)
                for member in self.estimators_
            ]
            for raw_values in zip(*generators):
                predictions = [
                    raw if trend is None else raw + trend
                    for raw, trend in zip(raw_values, trends)
                ]
                yield np.mean(np.stack(predictions, axis=0), axis=0)
            return
        trend = self._linear_residual_trend(X)
        if self._fitted_distributional():
            loss = self.model_.loss_
            for raw in self.model_.staged_predict_raw(X, _validated=True):
                if hasattr(loss, "mean_from_params"):
                    params = self._linear_residual_shift_params(
                        self._calibrated_params_from_raw(raw, X), trend
                    )
                    yield loss.mean_from_params(*params)
                else:
                    yield loss.mean_from_raw(raw)
        else:
            for raw in self.model_.staged_predict_raw(X, _validated=True):
                yield raw if trend is None else raw + trend

    def _require_distributional(self, method_name, capability=None):
        self._check_fitted_loss_matches_params(method_name)
        if not self._fitted_distributional():
            raise ValueError(
                f"{method_name}() requires a distributional loss; this model was "
                f"fit with loss={self._fitted_loss_name()!r}"
            )
        loss = getattr(self.model_, "loss_", None)
        if capability == "interval" and not getattr(loss, "interval_support", False):
            raise NotImplementedError(
                f"{method_name}() is not implemented for "
                f"loss={self._fitted_loss_name()!r}"
            )
        if capability == "sample" and not getattr(loss, "sample_support", False):
            raise NotImplementedError(
                f"{method_name}() is not implemented for "
                f"loss={self._fitted_loss_name()!r}"
            )
        return loss

    def _require_gaussian(self, method_name):
        self._check_fitted_loss_matches_params(method_name)
        if not self._fitted_distributional() or self._fitted_loss_name() != "Gaussian":
            raise ValueError(
                f"{method_name}() requires loss='Gaussian'; this model was "
                f"fit with loss={self._fitted_loss_name()!r}"
            )

    def _group_affine_scale_values(self, scale_values, X):
        if X is None:
            raise ValueError(
                "dist_calibration='per_metric_affine' requires X so the "
                "calibration feature can be read"
            )
        groups = _normalize_calibration_groups(
            _extract_feature_column_by_index(
                X, int(self.dist_calibration_feature_index_)
            )
        )
        if groups.shape[0] != scale_values.shape[0]:
            raise ValueError(
                "dist_calibration_feature column length must match prediction rows"
            )
        a = float(getattr(self, "dist_affine_a_", 0.0))
        b = float(getattr(self, "dist_affine_b_", 1.0))
        log_scale = np.log(np.maximum(scale_values, _SIGMA_MIN))
        calibrated = np.exp(np.clip(a + b * log_scale, -700.0, 700.0))
        group_keys = getattr(self, "dist_group_affine_groups_", None)
        if group_keys is None:
            return calibrated
        group_a = np.asarray(self.dist_group_affine_a_, dtype=np.float64)
        group_b = np.asarray(self.dist_group_affine_b_, dtype=np.float64)
        for key, key_a, key_b in zip(group_keys, group_a, group_b):
            mask = groups == key
            if np.any(mask):
                calibrated[mask] = np.exp(
                    np.clip(key_a + key_b * log_scale[mask], -700.0, 700.0)
                )
        return calibrated

    def _calibrated_params_from_raw(self, raw, X=None):
        loss = self.model_.loss_
        if hasattr(self.model_, "params_from_raw"):
            params = list(self.model_.params_from_raw(raw))
        else:
            params = list(loss.params_from_raw(raw))
        method = self._active_dist_calibration()
        if method is None:
            return tuple(params)

        targets = tuple(getattr(loss, "calibration_targets", ()))
        if "scale" in targets:
            idx = int(getattr(loss, "scale_param_index", 1))
            scale_values = np.asarray(params[idx], dtype=np.float64)
            if method == "per_metric_affine":
                params[idx] = self._group_affine_scale_values(scale_values, X)
            elif method == "affine":
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
        raw = self.model_.predict_raw(X, _validated=True)
        params = self._calibrated_params_from_raw(raw, X)
        return self._linear_residual_shift_params(
            params, self._linear_residual_trend(X)
        )

    def predict_dist(self, X):
        """Return distribution parameters for a fitted distributional model."""
        return self._predict_dist_checked(X, "predict_dist")

    def predict_variance(self, X):
        """Return predictive variance for a fitted distributional model."""
        X = _check_predict_input(self, X)
        loss = self._require_distributional("predict_variance")
        raw = self.model_.predict_raw(X, _validated=True)
        if self._active_dist_calibration() is None:
            if hasattr(self.model_, "variance_from_raw"):
                return self.model_.variance_from_raw(raw)
            return loss.variance_from_raw(raw)
        params = self._calibrated_params_from_raw(raw, X)
        if hasattr(loss, "variance_from_params"):
            return loss.variance_from_params(*params)
        if self._fitted_loss_name() == "Gaussian":
            return params[1] * params[1]
        raise NotImplementedError(
            f"predict_variance() with calibration is not implemented for "
            f"loss={self._fitted_loss_name()!r}"
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
        trend = self._linear_residual_trend(X)
        raw = self.model_.predict_raw(X, _validated=True)
        params = self._calibrated_params_from_raw(raw, X)
        if hasattr(loss, "interval_from_params"):
            return self._linear_residual_shift_interval(
                loss.interval_from_params(*params, alpha), trend
            )
        raise NotImplementedError(
            f"predict_interval() with calibration is not implemented for "
            f"loss={self._fitted_loss_name()!r}"
        )

    def sample(self, X, n_samples=1, random_state=None):
        """Draw samples from the fitted predictive distribution."""
        n_samples = int(n_samples)
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        X = _check_predict_input(self, X)
        loss = self._require_distributional("sample", capability="sample")
        trend = self._linear_residual_trend(X)
        raw = self.model_.predict_raw(X, _validated=True)
        rng = np.random.default_rng(random_state)
        params = self._calibrated_params_from_raw(raw, X)
        if hasattr(loss, "sample_from_params"):
            return self._linear_residual_shift_samples(
                loss.sample_from_params(*params, rng, n_samples), trend
            )
        raise NotImplementedError(
            "sample() with calibration is not implemented for "
            f"loss={self._fitted_loss_name()!r}"
        )

    def save_model(self, path):
        """Serialize the fitted model to a single ``.npz`` file."""
        check_is_fitted(self, "model_")
        if hasattr(self, "estimators_"):
            from .serialization import save_ensemble

            save_ensemble(
                self.estimators_,
                path,
                wrapper_class=type(self).__name__,
                params=self._ensemble_params_header(),
                metadata=self.ensemble_metadata_,
            )
            return
        from .serialization import save_booster
        save_booster(
            self.model_, path,
            wrapper_header={"wrapper_class": type(self).__name__,
                            "params": self._wrapper_params_header(),
                            "state": self._wrapper_state_header()},
            wrapper_arrays=self._wrapper_arrays(),
        )

    @classmethod
    def load_model(cls, path):
        """Load a model saved with :meth:`save_model`."""
        ensemble = cls._load_ensemble_model(path)
        if ensemble is not None:
            return ensemble
        from .serialization import load_booster
        booster, wrapper_header, wrapper_arrays = load_booster(
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
                "use DarkoClassifier.load_model"
            )
        est = cls()
        params = wrapper_header.get("params") or {}
        if isinstance(booster, DistributionalBoosting):
            saved_loss = params.get("loss")
            if saved_loss is not None and saved_loss != booster.loss_name:
                raise ValueError(
                    "invalid DarkoFit model: wrapper loss does not match "
                    "the loaded distributional booster"
                )
        known = est.get_params()
        est.set_params(**{k: v for k, v in params.items() if k in known})
        est.model_ = booster
        if isinstance(booster, DistributionalBoosting):
            est.loss = booster.loss_name
        elif isinstance(booster, GradientBoosting):
            est.loss = booster.loss_name
        state = wrapper_header.get("state", {})
        est._restore_wrapper_state(state)
        est._restore_linear_residual_state(state, wrapper_arrays)
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
        if hasattr(self, "estimators_"):
            return np.mean(
                np.stack(
                    [
                        member.model_.feature_importances_
                        for member in self.estimators_
                    ],
                    axis=0,
                ),
                axis=0,
            )
        return self.model_.feature_importances_

    @property
    def timing_(self):
        check_is_fitted(self, "model_")
        if hasattr(self, "estimators_"):
            return {
                "ensemble_member_count": len(self.estimators_),
                "members": [
                    member.model_.timing_ for member in self.estimators_
                ],
            }
        return self.model_.timing_


class DarkoClassifier(ClassifierMixin, _RefitParamsMixin, BaseEstimator):
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
    n_ensembles : int, default 1
        Number of OOB-selected bootstrap members. Values above one opt into
        soft-vote aggregation. ``ensemble_bootstrap="groups"`` requires
        ``groups`` in :meth:`fit`.
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
                 multiclass_tree_strategy="auto", eval_train_loss=False,
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
                 target_ordered_cat_codes="off",
                 selection_rounds=None, n_ensembles=1,
                 ensemble_bootstrap="rows",
                 ensemble_shared_preprocessing=True):
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
        self.selection_rounds = selection_rounds
        self.n_ensembles = n_ensembles
        self.ensemble_bootstrap = ensemble_bootstrap
        self.ensemble_shared_preprocessing = ensemble_shared_preprocessing
        self.auto_learning_rate_probe = auto_learning_rate_probe
        self.auto_learning_rate_probe_values = auto_learning_rate_probe_values
        self.auto_learning_rate_probe_iterations = auto_learning_rate_probe_iterations

    def fit(self, X, y, cat_features=None, eval_set=None, groups=None,
            sample_weight=None, eval_sample_weight=None, callbacks=None,
            ordinal_features=None):
        """Fit the model.

        Parameters
        ----------
        X, y : array-like
            Training data.
        cat_features : list of int or str, or None
            Column indices or, for named input, column names to treat as
            categoricals.
        ordinal_features : mapping, {"auto"}, or None
            Ordered categorical features to encode as rank-valued numeric
            columns before binning. A mapping declares each feature's complete
            ordered category sequence. ``"auto"`` recognizes ordered pandas
            categoricals and integer-coded columns listed in *cat_features*.
            Unknown categories at prediction time are rejected.
        eval_set : (X_val, y_val) tuple or None
            Explicit validation set with original class labels.  When provided,
            automatic splitting is skipped.
        groups : array-like of shape (n_samples,) or None
            Group labels (e.g. ``df['subject_id']``).  When supplied and early
            stopping triggers an automatic split, ``StratifiedGroupKFold`` keeps
            groups intact and class proportions balanced across the split.
            Set ``validation_strategy='group'`` to require this behavior
            explicitly.
        sample_weight : array-like of shape (n_samples,) or None
            Per-sample weights.  Normalized to mean 1 internally.
        eval_sample_weight : array-like of shape (n_validation_samples,) or None
            Validation weights used when evaluating early stopping.
        callbacks : callable or iterable of callables, or None
            Fit-time boosting callbacks. Each callback receives a
            :class:`darkofit.callbacks.BoostingProgress` snapshot before the
            next boosting round and may return ``True`` to stop. Automatic
            tree-mode selection shares the same callback objects across its
            candidate fits. Callbacks are not supported with automatic
            learning-rate probes or refitting.
        """
        self._clear_ensemble_state()
        if not getattr(self, "_suppress_wrapper_deprecation_warning", False):
            self._warn_wrapper_deprecated_options()
        ensemble = self._fit_ensemble(
            X,
            y,
            cat_features=cat_features,
            eval_set=eval_set,
            groups=groups,
            sample_weight=sample_weight,
            eval_sample_weight=eval_sample_weight,
            callbacks=callbacks,
            ordinal_features=ordinal_features,
            classification=True,
        )
        if ensemble is not None:
            return ensemble
        callbacks = _normalize_callbacks(callbacks)
        X_input = X
        feature_names = _feature_names_from_input(X_input)
        X, cat_features, ordinal_mode, ordinal_records = (
            self._prepare_ordinal_fit_input(
            X, cat_features, ordinal_features
            )
        )
        X, cat_features, n_features = _coerce_fit_X(X, cat_features)
        ordinal_nominal_cat_count = len(cat_features)
        eval_set = _ensure_dense_eval_set(eval_set)
        if eval_set is not None:
            _validate_eval_set_feature_schema(
                eval_set[0],
                n_features,
                expected_feature_names=feature_names,
            )
            eval_set = (
                _transform_ordinal_features(eval_set[0], ordinal_records),
                eval_set[1],
            )
        eval_set = _validate_eval_set_features(
            eval_set, n_features,
            expected_feature_names=feature_names,
            cat_features=cat_features,
            feature_names_validated=True,
        )
        y = validate_target_vector(y, X.shape[0])
        target_type = type_of_target(y)
        if target_type not in {"binary", "multiclass"}:
            raise ValueError(f"Unknown label type: {target_type}")
        classes = np.unique(y)
        n_classes = classes.size
        if n_classes < 2:
            raise ValueError(
                f"Need at least 2 classes; got {int(n_classes)} class."
            )
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
        if callbacks:
            if self.refit:
                raise ValueError("callbacks are not supported with refit=True")
            if self.auto_learning_rate_probe:
                raise ValueError(
                    "callbacks are not supported with "
                    "auto_learning_rate_probe=True"
                )
        self._validate_tree_mode_selection_request()
        if self.refit:
            self._refit_strategy_exponent(self.refit_strategy)
        es_active = _should_early_stop(self.early_stopping)
        if validation_strategy_ == "group" and groups is None:
            raise ValueError("validation_strategy='group' requires groups")
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
        # Refit can only reuse the selection fit's preprocessing when both
        # train on the same rows, i.e. with an explicit eval set; automatic
        # splits refit on different (full) data, where caching would only
        # add copies.
        refit_can_reuse_preprocessing = (
            bool(self.refit) and selection_active and explicit_eval_set
        )

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
                {}
                if (
                    tree_mode_auto
                    or self.auto_learning_rate_probe
                    or refit_can_reuse_preprocessing
                )
                else None
            )

            def make_model(model_kw):
                model = GradientBoosting(loss="Logloss", **model_kw)
                if preprocessing_cache is not None:
                    model._preprocessing_cache = preprocessing_cache
                return self._configure_model_preprocessing(model)

            if tree_mode_auto:
                model, probe_metadata, tree_mode_selection_metadata = (
                    self._fit_tree_mode_auto(
                        make_model, kw, X, y01,
                        cat_features=cat_features,
                        eval_set=eval_set,
                        sample_weight=sample_weight,
                        eval_sample_weight=eval_sample_weight,
                        callbacks=callbacks,
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
                    callbacks=callbacks,
                )
        else:
            multiclass = True
            preprocessing_cache = (
                {}
                if (
                    tree_mode_auto
                    or self.auto_learning_rate_probe
                    or refit_can_reuse_preprocessing
                )
                else None
            )

            def make_model(model_kw):
                model = MulticlassBoosting(**model_kw)
                if preprocessing_cache is not None:
                    model._preprocessing_cache = preprocessing_cache
                return self._configure_model_preprocessing(model)

            if tree_mode_auto:
                model, probe_metadata, tree_mode_selection_metadata = (
                    self._fit_tree_mode_auto(
                        make_model, kw, X, y,
                        cat_features=cat_features,
                        eval_set=eval_set,
                        sample_weight=sample_weight,
                        eval_sample_weight=eval_sample_weight,
                        callbacks=callbacks,
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
                    callbacks=callbacks,
                )
            classes = model.classes_
        self.model_ = model
        self._multiclass = multiclass
        self.classes_ = classes
        self.n_classes_ = len(classes)
        self._set_ordinal_fit_state(
            ordinal_mode, ordinal_records, ordinal_nominal_cat_count
        )
        self._attach_ordinal_metadata()
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
                if preprocessing_cache is not None:
                    refit_model._preprocessing_cache = preprocessing_cache
                refit_model.fit(
                    X_full, y_full, cat_features=cat_features,
                    sample_weight=sample_weight_full,
                )
                classes = refit_model.classes_
            else:
                y01_full = (y_full == classes[1]).astype(np.float64)
                refit_model = GradientBoosting(loss="Logloss", **refit_kw)
                if preprocessing_cache is not None:
                    refit_model._preprocessing_cache = preprocessing_cache
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
        if preprocessing_cache is not None:
            # Free the binned-matrix copies; fitted models keep a reference
            # to this dict, so an unemptied cache would pin them in memory.
            preprocessing_cache.clear()
        self._attach_ordinal_metadata()
        return self

    def predict_proba(self, X):
        X = _check_predict_input(self, X)
        if hasattr(self, "estimators_"):
            probabilities = []
            for member in self.estimators_:
                raw = member.model_.predict_raw(X, _validated=True)
                if member._multiclass:
                    probabilities.append(member.model_.loss_.transform(raw))
                else:
                    p1 = member.model_.loss_.transform(raw)
                    probabilities.append(np.column_stack([1.0 - p1, p1]))
            return np.mean(np.stack(probabilities, axis=0), axis=0)
        raw = self.model_.predict_raw(X, _validated=True)
        if self._multiclass:
            return self.model_.loss_.transform(raw)            # (n, K)
        p1 = self.model_.loss_.transform(raw)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        if hasattr(self, "estimators_"):
            probabilities = self.predict_proba(X)
            return self.classes_[np.argmax(probabilities, axis=1)]
        X = _check_predict_input(self, X)
        raw = self.model_.predict_raw(X, _validated=True)
        if self._multiclass:
            return self.classes_[np.argmax(raw, axis=1)]
        p1 = self.model_.loss_.transform(raw)
        return self.classes_[(p1 > 0.5).astype(np.int64)]

    def shap_values(self, X, X_background=None):
        """Return exact interventional SHAP contributions in log-odds space."""
        X = _check_predict_input(self, X)
        if self._multiclass:
            raise NotImplementedError(
                "shap_values() currently supports only binary classifiers"
            )
        if X_background is not None:
            X_background = _check_predict_input(self, X_background)
        if hasattr(self, "estimators_"):
            values = []
            expected_values = []
            for member in self.estimators_:
                contributions, expected_value = member.model_.shap_values(
                    X,
                    background=X_background,
                    _validated=True,
                    _background_validated=X_background is not None,
                )
                values.append(contributions)
                expected_values.append(float(expected_value))
            contributions = np.mean(np.stack(values, axis=0), axis=0)
            expected_value = float(np.mean(expected_values))
        else:
            contributions, expected_value = self.model_.shap_values(
                X,
                background=X_background,
                _validated=True,
                _background_validated=X_background is not None,
            )
        self.expected_value_ = expected_value
        return contributions

    def staged_predict_proba(self, X):
        """Yield class probabilities after each successive boosting round."""
        X = _check_predict_input(self, X)
        if hasattr(self, "estimators_"):
            generators = [
                member.model_.staged_predict_raw(X, _validated=True)
                for member in self.estimators_
            ]
            for raw_values in zip(*generators):
                probabilities = []
                for member, raw in zip(self.estimators_, raw_values):
                    if member._multiclass:
                        probabilities.append(
                            member.model_.loss_.transform(raw)
                        )
                    else:
                        p1 = member.model_.loss_.transform(raw)
                        probabilities.append(
                            np.column_stack([1.0 - p1, p1])
                        )
                yield np.mean(np.stack(probabilities, axis=0), axis=0)
            return
        for raw in self.model_.staged_predict_raw(X, _validated=True):
            if self._multiclass:
                yield self.model_.loss_.transform(raw)
            else:
                p1 = self.model_.loss_.transform(raw)
                yield np.column_stack([1.0 - p1, p1])

    def staged_predict(self, X):
        """Yield class labels after each successive boosting round."""
        if hasattr(self, "estimators_"):
            for probabilities in self.staged_predict_proba(X):
                yield self.classes_[np.argmax(probabilities, axis=1)]
            return
        X = _check_predict_input(self, X)
        for raw in self.model_.staged_predict_raw(X, _validated=True):
            if self._multiclass:
                yield self.classes_[np.argmax(raw, axis=1)]
            else:
                p1 = self.model_.loss_.transform(raw)
                yield self.classes_[(p1 > 0.5).astype(np.int64)]

    def staged_predict_raw(self, X):
        """Yield raw margins after each successive boosting round."""
        X = _check_predict_input(self, X)
        if hasattr(self, "estimators_"):
            generators = [
                member.model_.staged_predict_raw(X, _validated=True)
                for member in self.estimators_
            ]
            for raw_values in zip(*generators):
                yield np.mean(np.stack(raw_values, axis=0), axis=0)
            return
        yield from self.model_.staged_predict_raw(X, _validated=True)

    def save_model(self, path):
        """Serialize the fitted model to a single ``.npz`` file."""
        check_is_fitted(self, "model_")
        if hasattr(self, "estimators_"):
            from .serialization import save_ensemble

            save_ensemble(
                self.estimators_,
                path,
                wrapper_class=type(self).__name__,
                params=self._ensemble_params_header(),
                metadata=self.ensemble_metadata_,
            )
            return
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
        ensemble = cls._load_ensemble_model(path)
        if ensemble is not None:
            return ensemble
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
                "saved with DarkoClassifier.save_model"
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
        if hasattr(self, "estimators_"):
            return np.mean(
                np.stack(
                    [
                        member.model_.feature_importances_
                        for member in self.estimators_
                    ],
                    axis=0,
                ),
                axis=0,
            )
        return self.model_.feature_importances_

    @property
    def timing_(self):
        check_is_fitted(self, "model_")
        if hasattr(self, "estimators_"):
            return {
                "ensemble_member_count": len(self.estimators_),
                "members": [
                    member.model_.timing_ for member in self.estimators_
                ],
            }
        return self.model_.timing_
