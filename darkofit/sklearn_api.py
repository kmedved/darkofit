"""Scikit-learn flavored estimators: fit / predict / predict_proba."""

import hashlib
import json
import math
import time
import warnings
from collections.abc import Mapping
from functools import wraps

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
    _normalize_diagnostic_warnings,
    _normalize_oblivious_kernel,
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
from .serialization import (
    MAX_ENSEMBLE_MEMBERS,
    _booster_constructor_params,
    _jsonify,
)
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
    "interval_calibration",
    "sigma_calibration", "linear_residual", "linear_residual_alpha",
    "linear_residual_features", "linear_residual_fit_intercept",
    "linear_residual_standardize", "preset", "selection_rounds",
    "n_ensembles", "ensemble_bootstrap", "ensemble_shared_preprocessing",
    "ensemble_mode", "ensemble_member_learning_rate",
    "ensemble_member_colsample", "categorical_crosses",
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
_SCALAR_REGRESSION_LOSSES = frozenset({"RMSE", "MAE", "Quantile"})
_ORDINAL_STATE_VERSION = 1
_GROUP_CENTERED_CROSSES_VERSION = 1
_GROUP_CENTERED_CROSSES_MIN_SELECTION_ROWS = 2_000
_GROUP_CENTERED_CROSSES_VALIDATION_FRACTION = 0.15
_GROUP_CENTERED_CROSSES_TOP_NUMERIC = 4
_GROUP_CENTERED_CROSSES_TOP_CATEGORICAL = 3
_GROUP_CENTERED_CROSSES_DATA_FALLBACK_REASONS = frozenset({
    "no_categorical_features",
    "no_numeric_features",
    "below_min_samples",
})


def _group_centered_candidate_pairs(importances, cat_features, n_features):
    """Return deterministic top-importance numeric/category pairs."""
    cat = sorted({int(index) for index in (cat_features or ())})
    cat_set = set(cat)
    numeric = [index for index in range(int(n_features)) if index not in cat_set]
    values = np.asarray(importances, dtype=np.float64)
    if values.shape != (int(n_features),) or not np.all(np.isfinite(values)):
        raise RuntimeError("group-centered candidate importances are invalid")
    numeric = sorted(numeric, key=lambda index: (-values[index], index))[
        :_GROUP_CENTERED_CROSSES_TOP_NUMERIC
    ]
    cat = sorted(cat, key=lambda index: (-values[index], index))[
        :_GROUP_CENTERED_CROSSES_TOP_CATEGORICAL
    ]
    return [(numeric_index, cat_index) for numeric_index in numeric for cat_index in cat]


def _group_centered_preprocessing_record(prep):
    """Return compact, deterministic provenance for fitted centered columns."""
    pairs = [
        [int(numeric), int(categorical)]
        for numeric, categorical in getattr(prep, "group_centered_pairs_", ())
    ]
    means = list(getattr(prep, "group_centered_means_", ()))
    globals_ = np.asarray(
        getattr(prep, "group_centered_global_means_", ()), dtype="<f8"
    )
    if len(means) != len(pairs) or len(globals_) != len(pairs):
        raise RuntimeError("group-centered fitted preprocessing is inconsistent")
    digest = hashlib.sha256()
    digest.update(np.asarray(pairs, dtype="<i8").reshape(-1, 2).tobytes())
    category_counts = []
    for values in means:
        values = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise RuntimeError("group-centered fitted means are invalid")
        category_counts.append(int(len(values)))
        digest.update(values.tobytes())
    if not np.all(np.isfinite(globals_)):
        raise RuntimeError("group-centered fitted global means are invalid")
    digest.update(np.ascontiguousarray(globals_).tobytes())
    return {
        "pair_count": len(pairs),
        "pairs": pairs,
        "category_counts": category_counts,
        "global_means": globals_.tolist(),
        "means_sha256": digest.hexdigest(),
    }


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


def _normalize_categorical_crosses(setting):
    if not isinstance(setting, (bool, np.bool_)):
        raise TypeError("categorical_crosses must be a bool")
    return bool(setting)


def _preset_metadata_payload(preset, preset_params):
    return {
        "name": preset,
        "claim_tier": "E",
        "default_changed": False,
        "resolved": dict(preset_params),
        "evidence_scope": "spent_development_panel",
    }


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
    if n_ensembles > MAX_ENSEMBLE_MEMBERS:
        raise ValueError(
            f"n_ensembles must be at most {MAX_ENSEMBLE_MEMBERS}"
        )
    return n_ensembles


def _normalize_ensemble_bootstrap(bootstrap):
    token = str(bootstrap).strip().lower().replace("-", "_")
    if token not in {"rows", "groups"}:
        raise ValueError("ensemble_bootstrap must be 'rows' or 'groups'")
    return token


def _normalize_ensemble_mode(mode):
    token = str(mode).strip().lower().replace("-", "_")
    if token not in {"bootstrap", "v3"}:
        raise ValueError("ensemble_mode must be 'bootstrap' or 'v3'")
    return token


_PRIVATE_ENSEMBLE_V3_POLICY_FIELDS = ("learning_rate", "colsample")
_PRIVATE_ENSEMBLE_V3_POLICY_VALUES = {
    "learning_rate": 0.15,
    "colsample": 0.85,
}
_PRIVATE_ENSEMBLE_V3_PROTOTYPE = "ensemble_v3_b1_b2"
_PRIVATE_ENSEMBLE_V3_METADATA_VERSION = 4
_ENSEMBLE_V3_PUBLIC_CONTRACT = "ensemble-v3-public-contract-v1"
_ENSEMBLE_V3_POLICY_SENTINEL = "policy"
_ENSEMBLE_V3_RELEASE_CANDIDATE = "ensemble_v3_release_candidate"
_ENSEMBLE_V3_RELEASE_CANDIDATE_METADATA_VERSION = 5
_ENSEMBLE_V3_RELEASE_CANDIDATE_MEMBERS = 8
_ENSEMBLE_V3_PUBLIC_METADATA_VERSION = 1
_ENSEMBLE_V3_PUBLIC_PARAM_DEFAULTS = {
    "ensemble_mode": "bootstrap",
    "ensemble_member_learning_rate": _ENSEMBLE_V3_POLICY_SENTINEL,
    "ensemble_member_colsample": _ENSEMBLE_V3_POLICY_SENTINEL,
}
_PRIVATE_ENSEMBLE_V3_PROTOTYPES = frozenset({
    _PRIVATE_ENSEMBLE_V3_PROTOTYPE,
    _ENSEMBLE_V3_RELEASE_CANDIDATE,
})
_PRIVATE_ENSEMBLE_V3_MEMBER_OVERRIDES = frozenset({
    "n_ensembles",
    "random_state",
    "early_stopping",
    "use_best_model",
    "refit",
    "ensemble_mode",
    "ensemble_member_learning_rate",
    "ensemble_member_colsample",
    "thread_count",
    *_PRIVATE_ENSEMBLE_V3_POLICY_FIELDS,
})
_B3_PARALLEL_ENSEMBLE_CONTRACT = (
    "b3-parallel-ensemble-members-v1-20260723"
)


def _is_private_ensemble_v3_metadata(metadata):
    return (
        isinstance(metadata, Mapping)
        and metadata.get("private_prototype")
        in _PRIVATE_ENSEMBLE_V3_PROTOTYPES
    )


def _is_ensemble_v3_release_candidate_metadata(metadata):
    return (
        isinstance(metadata, Mapping)
        and metadata.get("private_prototype")
        == _ENSEMBLE_V3_RELEASE_CANDIDATE
    )


def _is_public_ensemble_v3_metadata(metadata):
    return (
        isinstance(metadata, Mapping)
        and metadata.get("ensemble_mode") == "v3"
        and metadata.get("recipe_contract") == _ENSEMBLE_V3_PUBLIC_CONTRACT
        and metadata.get("public_fit_surface") is True
        and "private_prototype" not in metadata
    )


def _private_ensemble_v3_json_token(value):
    try:
        return json.dumps(
            _jsonify(value),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "private ensemble-v3 constructor parameters must be JSON-safe"
        ) from exc


def _private_ensemble_v3_values_equal(left, right):
    try:
        return (
            _private_ensemble_v3_json_token(left)
            == _private_ensemble_v3_json_token(right)
        )
    except ValueError:
        return False


def _canonicalize_legacy_ensemble_v3_params(params, *, allowed):
    """Add post-v0.10 public defaults to historical private-v3 schemas."""
    if not isinstance(params, Mapping):
        return params
    values = dict(params)
    missing = set(_ENSEMBLE_V3_PUBLIC_PARAM_DEFAULTS).difference(values)
    if allowed and missing == set(_ENSEMBLE_V3_PUBLIC_PARAM_DEFAULTS):
        values.update(_ENSEMBLE_V3_PUBLIC_PARAM_DEFAULTS)
    return values


def _normalize_private_ensemble_v3_sampling(sampling):
    token = str(sampling).strip().lower().replace("-", "_")
    if token not in {"bootstrap", "without_replacement"}:
        raise ValueError(
            "private ensemble-v3 sampling must be 'bootstrap' or "
            "'without_replacement'"
        )
    return token


def _normalize_private_ensemble_v3_policy(policy):
    token = str(policy).strip().lower().replace("-", "_")
    if token not in {"none", "donor_balanced_v1"}:
        raise ValueError(
            "private ensemble-v3 member_policy must be 'none' or "
            "'donor_balanced_v1'"
        )
    return token


def _normalize_private_explicit_user_params(explicit_user_params):
    if explicit_user_params is None:
        return ()
    if isinstance(explicit_user_params, (str, bytes)):
        values = [explicit_user_params]
    else:
        try:
            values = list(explicit_user_params)
        except TypeError as exc:
            raise TypeError(
                "explicit_user_params must be an iterable of parameter names"
            ) from exc
    if any(not isinstance(value, str) for value in values):
        raise TypeError("explicit_user_params must contain only strings")
    if len(values) != len(set(values)):
        raise ValueError("explicit_user_params must not contain duplicates")
    unknown = sorted(
        set(values).difference(_PRIVATE_ENSEMBLE_V3_POLICY_FIELDS)
    )
    if unknown:
        raise ValueError(
            "explicit_user_params contains unsupported fields: "
            + ", ".join(unknown)
        )
    return tuple(
        name
        for name in _PRIVATE_ENSEMBLE_V3_POLICY_FIELDS
        if name in values
    )


def _resolve_private_ensemble_v3_policy(
    estimator, policy, explicit_user_params, explicit_user_values=None
):
    policy = _normalize_private_ensemble_v3_policy(policy)
    explicit = frozenset(
        _normalize_private_explicit_user_params(explicit_user_params)
    )
    if explicit_user_values is None:
        explicit_values = {name: getattr(estimator, name) for name in explicit}
    elif not isinstance(explicit_user_values, Mapping):
        raise TypeError("explicit_user_values must be a mapping or None")
    else:
        explicit_values = dict(explicit_user_values)
    if set(explicit_values) != set(explicit):
        raise ValueError(
            "explicit_user_values must contain exactly explicit_user_params"
        )
    resolutions = {}
    member_params = {}
    for name in _PRIVATE_ENSEMBLE_V3_POLICY_FIELDS:
        base_value = getattr(estimator, name)
        if name in explicit:
            resolved = explicit_values.get(name, base_value)
            source = "explicit_user"
        elif policy == "donor_balanced_v1":
            resolved = _PRIVATE_ENSEMBLE_V3_POLICY_VALUES[name]
            source = "member_policy"
        else:
            resolved = base_value
            source = "base"
        resolutions[name] = {
            "base": base_value,
            "resolved": resolved,
            "source": source,
        }
        member_params[name] = resolved
    return policy, tuple(name for name in _PRIVATE_ENSEMBLE_V3_POLICY_FIELDS
                         if name in explicit), resolutions, member_params


def _normalize_ensemble_v3_release_candidate_overrides(
    member_learning_rate=_ENSEMBLE_V3_POLICY_SENTINEL,
    member_colsample=_ENSEMBLE_V3_POLICY_SENTINEL,
):
    """Resolve the future clone-safe sentinel surface without exposing it."""
    values = {
        "learning_rate": member_learning_rate,
        "colsample": member_colsample,
    }
    explicit = {}
    for name, value in values.items():
        if isinstance(value, str) and value == _ENSEMBLE_V3_POLICY_SENTINEL:
            continue
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"ensemble_member_{name} must not be a bool")
        if name == "learning_rate" and value is None:
            explicit[name] = None
            continue
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"ensemble_member_{name} must be 'policy' or numeric"
            )
        numeric = float(value)
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"ensemble_member_{name} must be positive and finite")
        if name == "colsample" and numeric > 1.0:
            raise ValueError("ensemble_member_colsample must be at most 1")
        explicit[name] = value.item() if isinstance(value, np.generic) else value
    return values, explicit


def _resolve_public_ensemble_surface(estimator):
    """Validate the additive public ensemble controls before fitting."""
    mode = _normalize_ensemble_mode(estimator.ensemble_mode)
    member_learning_rate = estimator.ensemble_member_learning_rate
    member_colsample = estimator.ensemble_member_colsample
    if mode == "bootstrap":
        if (
            not (
                isinstance(member_learning_rate, str)
                and member_learning_rate == _ENSEMBLE_V3_POLICY_SENTINEL
            )
            or not (
                isinstance(member_colsample, str)
                and member_colsample == _ENSEMBLE_V3_POLICY_SENTINEL
            )
        ):
            raise ValueError(
                "ensemble_member_learning_rate and "
                "ensemble_member_colsample must remain 'policy' when "
                "ensemble_mode='bootstrap'"
            )
        return mode, None, None
    future_values, explicit_values = (
        _normalize_ensemble_v3_release_candidate_overrides(
            member_learning_rate,
            member_colsample,
        )
    )
    return mode, future_values, explicit_values


class _NamedRowSubset:
    """Array-backed row subset that preserves frame-like feature names."""

    def __init__(self, values, names):
        self._values = np.asarray(values)
        self.columns = [str(name) for name in names]
        self.shape = self._values.shape

    def to_numpy(self, dtype=None, na_value=None):
        del na_value
        return np.asarray(self._values, dtype=dtype)

    def __array__(self, dtype=None, copy=None):
        values = np.asarray(self._values, dtype=dtype)
        return values.copy() if copy else values


def _take_rows(values, indices):
    iloc = getattr(values, "iloc", None)
    if iloc is not None:
        return iloc[indices]
    gather = getattr(values, "gather", None)
    if callable(gather):
        return gather(indices)
    take = getattr(values, "take", None)
    if getattr(values, "column_names", None) is not None and callable(take):
        return take(indices)
    names = feature_names_from_input(values)
    subset = array_like_to_numpy(values)[indices]
    if names is not None:
        return _NamedRowSubset(subset, names)
    return subset


def _index_sha256(indices):
    values = np.ascontiguousarray(np.asarray(indices, dtype="<i8"))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _ensemble_class_partitions_are_usable(
    class_codes,
    sampled,
    oob,
    required_class_count,
    weights,
):
    """Return whether both partitions retain each class and its usable mass."""
    if required_class_count is None:
        return True
    class_count = int(required_class_count)
    positive_mass_classes = None
    if weights is not None:
        full_class_weight = np.bincount(
            class_codes,
            weights=weights,
            minlength=class_count,
        )
        positive_mass_classes = full_class_weight > 0.0
    for indices in (sampled, oob):
        partition_codes = class_codes[indices]
        observations = np.bincount(
            partition_codes,
            minlength=class_count,
        )
        if np.count_nonzero(observations) != class_count:
            return False
        if weights is not None:
            class_weight = np.bincount(
                partition_codes,
                weights=weights[indices],
                minlength=class_count,
            )
            if np.any(class_weight[positive_mass_classes] <= 0.0):
                return False
    return True


def _normalize_ensemble_group_codes(groups, n_rows, *, context):
    group_values = np.asarray(groups)
    if group_values.ndim != 1 or len(group_values) != int(n_rows):
        raise ValueError(
            f"{context} groups must be one-dimensional with one value per "
            "training row"
        )
    try:
        _, group_codes = np.unique(group_values, return_inverse=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} groups must contain consistently comparable scalar values"
        ) from exc
    group_codes = np.ascontiguousarray(group_codes, dtype="<i8")
    unique_group_count = int(group_codes.max()) + 1
    if unique_group_count < 2:
        raise ValueError(f"{context} group sampling requires at least two groups")
    return group_codes, unique_group_count


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
        group_codes, unique_group_count = _normalize_ensemble_group_codes(
            groups, n_rows, context="ensemble bootstrap"
        )

    weights = (
        None
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    labels = None if y is None else np.asarray(y)
    class_codes = None
    if required_class_count is not None:
        _, class_codes = np.unique(labels, return_inverse=True)
        class_codes = np.asarray(class_codes, dtype=np.int64)
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
        if not _ensemble_class_partitions_are_usable(
            class_codes,
            sampled,
            oob,
            required_class_count,
            weights,
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


def _ensemble_without_replacement_plan(
    n_rows,
    seed,
    *,
    sampling_unit,
    sample_fraction,
    groups=None,
    y=None,
    required_class_count=None,
    sample_weight=None,
    max_attempts=128,
):
    """Return the deterministic private B1 sample/OOB plan."""
    n_rows = int(n_rows)
    if n_rows < 2:
        raise ValueError("an ensemble requires at least two training rows")
    sampling_unit = _normalize_ensemble_bootstrap(sampling_unit)
    try:
        fraction = float(sample_fraction)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("sample_fraction must be the float 0.8") from exc
    if (
        isinstance(sample_fraction, (bool, np.bool_))
        or not np.isfinite(fraction)
        or fraction != 0.8
    ):
        raise ValueError("the funded private sample_fraction is exactly 0.8")

    group_codes = None
    unique_group_count = None
    if sampling_unit == "groups":
        group_codes, unique_group_count = _normalize_ensemble_group_codes(
            groups, n_rows, context="private"
        )

    weights = (
        None
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    labels = None if y is None else np.asarray(y)
    class_codes = None
    if required_class_count is not None:
        _, class_codes = np.unique(labels, return_inverse=True)
        class_codes = np.asarray(class_codes, dtype=np.int64)
    unit_count = n_rows if sampling_unit == "rows" else unique_group_count
    sample_count = min(
        unit_count - 1,
        max(1, int(math.floor(fraction * unit_count + 0.5))),
    )
    rng = np.random.default_rng(int(seed))
    for attempt in range(1, int(max_attempts) + 1):
        selected_units = np.asarray(
            rng.choice(unit_count, size=sample_count, replace=False),
            dtype=np.int64,
        )
        if sampling_unit == "rows":
            sampled = selected_units
        else:
            sampled = np.concatenate(
                [
                    np.flatnonzero(group_codes == group)
                    for group in selected_units
                ]
            ).astype(np.int64, copy=False)
        oob_mask = np.ones(n_rows, dtype=np.bool_)
        oob_mask[sampled] = False
        oob = np.flatnonzero(oob_mask).astype(np.int64, copy=False)
        if not len(sampled) or not len(oob):
            continue
        if weights is not None and (
            float(np.sum(weights[sampled])) <= 0.0
            or float(np.sum(weights[oob])) <= 0.0
        ):
            continue
        if not _ensemble_class_partitions_are_usable(
            class_codes,
            sampled,
            oob,
            required_class_count,
            weights,
        ):
            continue
        return {
            "sampled": sampled,
            "oob": oob,
            "attempts": attempt,
            "sampled_group_draws": (
                None if sampling_unit == "rows" else len(selected_units)
            ),
            "sampled_unique_groups": (
                None if sampling_unit == "rows" else len(selected_units)
            ),
            "oob_groups": (
                None
                if sampling_unit == "rows"
                else np.unique(group_codes[oob]).size
            ),
        }
    raise RuntimeError(
        "could not construct a without-replacement sample with a usable, "
        "class-safe out-of-bag validation set"
    )


def _validate_loaded_private_ensemble_v3_metadata(
    metadata,
    members,
    *,
    classification,
    index_provenance,
    group_codes,
    base_constructor_params,
    allow_legacy_param_schema=True,
):
    """Fail closed on private B1/B2 fitted and persistence provenance."""
    member_count = len(members)
    b3_schedule = metadata.get("private_b3_schedule")
    if b3_schedule is None:
        scheduling_valid = metadata.get("sequential") is True
        expected_member_thread_count = None
    else:
        scheduling_valid = (
            metadata.get("sequential") is False
            and isinstance(b3_schedule, Mapping)
            and set(b3_schedule)
            == {
                "contract",
                "mode",
                "workers",
                "member_threads",
                "total_thread_budget",
                "maximum_model_threads",
                "result_order",
            }
            and b3_schedule.get("contract")
            == _B3_PARALLEL_ENSEMBLE_CONTRACT
            and b3_schedule.get("mode") == "private_process_workers"
            and b3_schedule.get("result_order") == "member_index"
        )
        if scheduling_valid:
            for name in (
                "workers",
                "member_threads",
                "total_thread_budget",
                "maximum_model_threads",
            ):
                value = b3_schedule.get(name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 1
                ):
                    scheduling_valid = False
                    break
        if scheduling_valid:
            expected_topology = _resolve_b3_parallel_topology(
                member_count, b3_schedule["total_thread_budget"]
            )
            scheduling_valid = (
                expected_topology
                == (b3_schedule["workers"], b3_schedule["member_threads"])
                and b3_schedule["maximum_model_threads"]
                == b3_schedule["workers"] * b3_schedule["member_threads"]
                and b3_schedule["maximum_model_threads"]
                <= b3_schedule["total_thread_budget"]
            )
        expected_member_thread_count = (
            b3_schedule.get("member_threads") if scheduling_valid else None
        )
    release_candidate = _is_ensemble_v3_release_candidate_metadata(metadata)
    expected_version = (
        _ENSEMBLE_V3_RELEASE_CANDIDATE_METADATA_VERSION
        if release_candidate
        else _PRIVATE_ENSEMBLE_V3_METADATA_VERSION
    )
    expected_prototype = (
        _ENSEMBLE_V3_RELEASE_CANDIDATE
        if release_candidate
        else _PRIVATE_ENSEMBLE_V3_PROTOTYPE
    )
    if (
        metadata.get("version") != expected_version
        or metadata.get("private_prototype") != expected_prototype
        or metadata.get("claim_tier") != "E"
        or metadata.get("default_changed") is not False
        or metadata.get("public_fit_surface") is not False
        or not scheduling_valid
        or metadata.get("oob_early_stopping") is not True
    ):
        raise ValueError(
            "invalid DarkoFit model: private ensemble-v3 provenance is invalid"
        )
    if (
        isinstance(metadata.get("member_count"), bool)
        or not isinstance(metadata.get("member_count"), int)
        or metadata["member_count"] != member_count
    ):
        raise ValueError(
            "invalid DarkoFit model: private ensemble-v3 member count is invalid"
        )
    saved_base_params = _canonicalize_legacy_ensemble_v3_params(
        metadata.get("base_constructor_params"),
        allowed=allow_legacy_param_schema,
    )
    base_constructor_params = _canonicalize_legacy_ensemble_v3_params(
        base_constructor_params,
        allowed=allow_legacy_param_schema,
    )
    expected_param_names = set(members[0].get_params(deep=False))
    if (
        not isinstance(saved_base_params, Mapping)
        or not isinstance(base_constructor_params, Mapping)
        or set(saved_base_params) != expected_param_names
        or set(base_constructor_params) != expected_param_names
        or not _private_ensemble_v3_values_equal(
            saved_base_params, base_constructor_params
        )
        or saved_base_params.get("n_ensembles") != member_count
        or isinstance(saved_base_params.get("n_ensembles"), bool)
        or saved_base_params.get("refit") is not False
        or (
            "preset" in saved_base_params
            and saved_base_params.get("preset") is not None
        )
        or _is_auto_tree_mode(saved_base_params.get("tree_mode"))
        or saved_base_params.get("auto_learning_rate_probe") is not False
    ):
        raise ValueError(
            "invalid DarkoFit model: private ensemble-v3 base constructor "
            "parameters are invalid"
        )
    fit_seed = metadata.get("fit_random_state_seed")
    if (
        (
            fit_seed is not None
            and (
                isinstance(fit_seed, bool)
                or not isinstance(fit_seed, int)
                or fit_seed < 0
            )
        )
        or saved_base_params.get("random_state") != fit_seed
        or type(saved_base_params.get("random_state")) is not type(fit_seed)
    ):
        raise ValueError(
            "invalid DarkoFit model: private ensemble-v3 base random state "
            "is invalid"
        )
    sampling = metadata.get("sampling")
    sampling_unit = metadata.get("sampling_unit")
    fraction = metadata.get("sample_fraction")
    if (
        sampling not in {"bootstrap", "without_replacement"}
        or sampling_unit not in {"rows", "groups"}
        or metadata.get("bootstrap") != sampling_unit
        or (
            sampling == "bootstrap"
            and fraction is not None
        )
        or (
            sampling == "without_replacement"
            and (
                isinstance(fraction, bool)
                or not isinstance(fraction, (int, float))
                or float(fraction) != 0.8
            )
        )
    ):
        raise ValueError(
            "invalid DarkoFit model: private ensemble-v3 sampling is invalid"
        )
    policy = metadata.get("member_policy")
    explicit = metadata.get("explicit_user_params")
    resolutions = metadata.get("policy_resolutions")
    explicit_values = None
    if release_candidate:
        future_params = metadata.get("future_constructor_params")
        if (
            metadata.get("recipe_contract") != _ENSEMBLE_V3_PUBLIC_CONTRACT
            or metadata.get("recipe_version") != 1
            or not isinstance(future_params, Mapping)
            or set(future_params)
            != {
                "ensemble_mode",
                "ensemble_member_learning_rate",
                "ensemble_member_colsample",
            }
            or future_params.get("ensemble_mode") != "v3"
            or member_count != _ENSEMBLE_V3_RELEASE_CANDIDATE_MEMBERS
        ):
            raise ValueError(
                "invalid DarkoFit model: ensemble-v3 release-candidate "
                "contract is invalid"
            )
        try:
            future_values, explicit_values = (
                _normalize_ensemble_v3_release_candidate_overrides(
                    future_params["ensemble_member_learning_rate"],
                    future_params["ensemble_member_colsample"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "invalid DarkoFit model: ensemble-v3 release-candidate "
                "overrides are invalid"
            ) from exc
        if not _private_ensemble_v3_values_equal(
            future_values,
            {
                "learning_rate": future_params[
                    "ensemble_member_learning_rate"
                ],
                "colsample": future_params["ensemble_member_colsample"],
            },
        ):
            raise ValueError(
                "invalid DarkoFit model: ensemble-v3 release-candidate "
                "overrides are contradictory"
            )
    explicit_is_valid = (
        isinstance(explicit, list)
        and all(isinstance(name, str) for name in explicit)
    )
    if explicit_is_valid:
        explicit_is_valid = (
            len(explicit) == len(set(explicit))
            and all(
                name in _PRIVATE_ENSEMBLE_V3_POLICY_FIELDS
                for name in explicit
            )
        )
    if (
        policy not in {"none", "donor_balanced_v1"}
        or not explicit_is_valid
        or not isinstance(resolutions, Mapping)
        or set(resolutions) != set(_PRIVATE_ENSEMBLE_V3_POLICY_FIELDS)
    ):
        raise ValueError(
            "invalid DarkoFit model: private ensemble-v3 policy is invalid"
        )
    if release_candidate and (
        policy != "donor_balanced_v1"
        or sampling != "without_replacement"
        or sampling_unit != saved_base_params.get("ensemble_bootstrap")
        or set(explicit) != set(explicit_values)
    ):
        raise ValueError(
            "invalid DarkoFit model: ensemble-v3 release-candidate recipe "
            "is contradictory"
        )
    for name in _PRIVATE_ENSEMBLE_V3_POLICY_FIELDS:
        resolution = resolutions[name]
        if (
            not isinstance(resolution, Mapping)
            or set(resolution) != {"base", "resolved", "source"}
        ):
            raise ValueError(
                "invalid DarkoFit model: private ensemble-v3 policy "
                "resolution is invalid"
            )
        expected_source = (
            "explicit_user"
            if name in explicit
            else "member_policy"
            if policy == "donor_balanced_v1"
            else "base"
        )
        expected_value = (
            explicit_values[name]
            if expected_source == "explicit_user" and release_candidate
            else _PRIVATE_ENSEMBLE_V3_POLICY_VALUES[name]
            if expected_source == "member_policy"
            else resolution["base"]
        )
        if (
            resolution["source"] != expected_source
            or not _private_ensemble_v3_values_equal(
                resolution["base"], saved_base_params[name]
            )
            or type(resolution["resolved"]) is not type(expected_value)
            or resolution["resolved"] != expected_value
        ):
            raise ValueError(
                "invalid DarkoFit model: private ensemble-v3 policy "
                "resolution is contradictory"
            )
    expected_aggregation = "soft_vote" if classification else "mean"
    if metadata.get("aggregation") != expected_aggregation:
        raise ValueError(
            "invalid DarkoFit model: private ensemble-v3 aggregation is invalid"
        )
    input_rows = metadata.get("input_row_count")
    input_features = metadata.get("input_feature_count")
    if (
        isinstance(input_rows, bool)
        or not isinstance(input_rows, int)
        or input_rows < 2
        or isinstance(input_features, bool)
        or not isinstance(input_features, int)
        or input_features < 1
    ):
        raise ValueError(
            "invalid DarkoFit model: private ensemble-v3 input shape is invalid"
        )
    input_group_count = metadata.get("input_group_count")
    group_codes_sha256 = metadata.get("group_codes_sha256")
    if sampling_unit == "rows":
        if (
            group_codes is not None
            or input_group_count is not None
            or group_codes_sha256 is not None
        ):
            raise ValueError(
                "invalid DarkoFit model: private row-sampling group provenance "
                "is invalid"
            )
    else:
        group_codes = np.asarray(group_codes)
        if (
            group_codes.ndim != 1
            or group_codes.dtype != np.dtype("<i8")
            or group_codes.size != input_rows
            or np.any(group_codes < 0)
            or isinstance(input_group_count, bool)
            or not isinstance(input_group_count, int)
            or input_group_count < 2
            or not np.array_equal(
                np.unique(group_codes),
                np.arange(input_group_count, dtype=np.int64),
            )
            or not isinstance(group_codes_sha256, str)
            or group_codes_sha256 != _index_sha256(group_codes)
        ):
            raise ValueError(
                "invalid DarkoFit model: private group-code provenance is invalid"
            )
    shared = metadata.get("shared_preprocessing")
    shared_requested = metadata.get("shared_preprocessing_requested")
    fallback_reason = metadata.get("shared_preprocessing_fallback_reason")
    if (
        shared not in {"numeric_target_free", "member_local"}
        or not isinstance(shared_requested, bool)
        or (
            shared == "numeric_target_free"
            and (
                not shared_requested
                or fallback_reason is not None
            )
        )
        or (
            shared == "member_local"
            and (
                (
                    shared_requested
                    and fallback_reason
                    not in {
                        "categorical_or_ordinal_features",
                        "non_numeric_dtype",
                    }
                )
                or (not shared_requested and fallback_reason is not None)
            )
        )
    ):
        raise ValueError(
            "invalid DarkoFit model: private ensemble-v3 preprocessing is invalid"
        )
    seeds = metadata.get("member_seeds")
    records = metadata.get("members")
    if (
        not isinstance(seeds, list)
        or not isinstance(records, list)
        or len(seeds) != member_count
        or len(records) != member_count
        or not isinstance(index_provenance, list)
        or len(index_provenance) != member_count
    ):
        raise ValueError(
            "invalid DarkoFit model: private ensemble-v3 member provenance is invalid"
        )

    for index, (seed, record, member, provenance) in enumerate(
        zip(seeds, records, members, index_provenance)
    ):
        model = getattr(member, "model_", None)
        if classification:
            multiclass = getattr(member, "_multiclass", None)
            valid_model = (
                multiclass is True
                and isinstance(model, MulticlassBoosting)
            ) or (
                multiclass is False
                and isinstance(model, GradientBoosting)
                and getattr(model, "loss_name", None) == "Logloss"
            )
        else:
            valid_model = (
                isinstance(model, GradientBoosting)
                and getattr(model, "loss_name", None)
                in {"RMSE", "MAE", "Quantile"}
            )
        if (
            not valid_model
            or getattr(member, "early_stopping", None) is not True
            or getattr(member, "use_best_model", None) is not True
            or getattr(member, "refit", None) is not False
        ):
            raise ValueError(
                "invalid DarkoFit model: private ensemble-v3 member model is invalid"
            )
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or seed >= 2**31 - 1
            or not isinstance(record, Mapping)
            or record.get("member") != index
            or record.get("seed") != seed
            or isinstance(record.get("member"), bool)
            or isinstance(record.get("seed"), bool)
            or member.get_params().get("random_state") != seed
            or getattr(model, "random_state", None) != seed
            or record.get("validation_source") != "explicit_eval_set"
        ):
            raise ValueError(
                "invalid DarkoFit model: private ensemble-v3 member identity is invalid"
            )
        saved_member_params = _canonicalize_legacy_ensemble_v3_params(
            record.get("member_constructor_params"),
            allowed=allow_legacy_param_schema,
        )
        saved_booster_params = record.get("booster_constructor_params")
        try:
            actual_member_params = (
                _private_ensemble_v3_wrapper_constructor_params(member)
            )
            expected_member_params = (
                _private_ensemble_v3_expected_member_params(
                    saved_base_params,
                    seed=seed,
                    policy_resolutions=resolutions,
                    thread_count=expected_member_thread_count,
                )
            )
            actual_booster_params = _booster_constructor_params(
                model, include_linear=True
            )
            expected_booster_params = (
                _private_ensemble_v3_expected_booster_params(
                    member, actual_member_params
                )
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "invalid DarkoFit model: private ensemble-v3 constructor "
                "metadata is invalid"
            ) from exc
        if (
            not isinstance(saved_member_params, Mapping)
            or not isinstance(saved_booster_params, Mapping)
            or set(saved_member_params) != set(actual_member_params)
            or set(saved_booster_params) != set(actual_booster_params)
            or not _private_ensemble_v3_values_equal(
                saved_member_params, actual_member_params
            )
            or not _private_ensemble_v3_values_equal(
                actual_member_params, expected_member_params
            )
            or not _private_ensemble_v3_values_equal(
                saved_booster_params, actual_booster_params
            )
            or not _private_ensemble_v3_values_equal(
                actual_booster_params, expected_booster_params
            )
        ):
            raise ValueError(
                "invalid DarkoFit model: private ensemble-v3 wrapper/booster "
                "constructor metadata differs"
            )
        if not isinstance(provenance, Mapping) or set(provenance) != {
            "sampled",
            "oob",
        }:
            raise ValueError(
                "invalid DarkoFit model: private ensemble-v3 index provenance "
                "is invalid"
            )
        sampled_indices = np.asarray(provenance["sampled"])
        oob_indices = np.asarray(provenance["oob"])
        if (
            sampled_indices.ndim != 1
            or oob_indices.ndim != 1
            or sampled_indices.dtype != np.dtype("<i8")
            or oob_indices.dtype != np.dtype("<i8")
            or sampled_indices.size == 0
            or oob_indices.size == 0
            or np.any(sampled_indices < 0)
            or np.any(sampled_indices >= input_rows)
            or np.any(oob_indices < 0)
            or np.any(oob_indices >= input_rows)
        ):
            raise ValueError(
                "invalid DarkoFit model: private ensemble-v3 index provenance "
                "is invalid"
            )
        sampled_unique = np.unique(sampled_indices)
        expected_oob = np.setdiff1d(
            np.arange(input_rows, dtype=np.int64),
            sampled_unique,
            assume_unique=True,
        )
        if not np.array_equal(oob_indices, expected_oob):
            raise ValueError(
                "invalid DarkoFit model: private ensemble-v3 index provenance "
                "is contradictory"
            )
        for name, values in (
            ("sampled_indices_sha256", sampled_indices),
            ("oob_indices_sha256", oob_indices),
        ):
            digest = record.get(name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or digest != _index_sha256(values)
            ):
                raise ValueError(
                    "invalid DarkoFit model: private ensemble-v3 index digest "
                    "is invalid"
                )
        for name in (
            "sampling_attempts",
            "sampled_rows",
            "sampled_unique_rows",
            "oob_rows",
            "fitted_thread_count",
            "best_iteration",
        ):
            value = record.get(name)
            minimum = 0 if name == "best_iteration" else 1
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(
                    f"invalid DarkoFit model: private ensemble-v3 {name} is invalid"
                )
        if (
            record["sampling_attempts"] > 128
            or record["sampled_rows"] != sampled_indices.size
            or record["sampled_unique_rows"] != sampled_unique.size
            or record["oob_rows"] != oob_indices.size
            or record["sampled_unique_rows"] > record["sampled_rows"]
            or record["sampled_unique_rows"] + record["oob_rows"]
            != input_rows
            or record.get("requested_sample_fraction") != fraction
            or record.get("realized_row_fraction")
            != record["sampled_unique_rows"] / float(input_rows)
        ):
            raise ValueError(
                "invalid DarkoFit model: private ensemble-v3 sample counts are invalid"
            )
        if sampling == "without_replacement" and (
            record["sampled_rows"] != record["sampled_unique_rows"]
        ):
            raise ValueError(
                "invalid DarkoFit model: private ensemble-v3 sample is not unique"
            )
        if (
            sampling == "bootstrap"
            and sampling_unit == "rows"
            and record["sampled_rows"] != input_rows
        ):
            raise ValueError(
                "invalid DarkoFit model: private ensemble-v3 bootstrap size is invalid"
            )
        group_fields = (
            "sampled_group_draws",
            "sampled_unique_groups",
            "oob_groups",
        )
        if sampling_unit == "rows":
            if (
                any(record.get(name) is not None for name in group_fields)
                or record.get("group_disjoint") is not None
                or (
                    sampling == "without_replacement"
                    and record["sampled_unique_rows"]
                    != min(
                        input_rows - 1,
                        max(
                            1,
                            int(math.floor(0.8 * input_rows + 0.5)),
                        ),
                    )
                )
            ):
                raise ValueError(
                    "invalid DarkoFit model: private row-sampling metadata is invalid"
                )
        else:
            invalid_group_fields = any(
                isinstance(record.get(name), bool)
                or not isinstance(record.get(name), int)
                or record[name] < 1
                for name in group_fields
            )
            row_multiplicities = np.bincount(
                sampled_indices, minlength=input_rows
            )
            group_multiplicities = np.empty(
                input_group_count, dtype=np.int64
            )
            complete_group_sampling = True
            for group in range(input_group_count):
                counts = row_multiplicities[group_codes == group]
                if counts.size == 0 or np.any(counts != counts[0]):
                    complete_group_sampling = False
                    break
                group_multiplicities[group] = counts[0]
            if complete_group_sampling:
                expected_group_draws = int(group_multiplicities.sum())
                expected_unique_groups = int(
                    np.count_nonzero(group_multiplicities)
                )
                expected_oob_groups = int(
                    input_group_count - expected_unique_groups
                )
                sampled_group_codes = np.unique(group_codes[sampled_unique])
                oob_group_codes = np.unique(group_codes[oob_indices])
                group_disjoint = not np.intersect1d(
                    sampled_group_codes, oob_group_codes
                ).size
            else:
                expected_group_draws = -1
                expected_unique_groups = -1
                expected_oob_groups = -1
                group_disjoint = False
            expected_without_replacement_groups = min(
                input_group_count - 1,
                max(
                    1,
                    int(math.floor(0.8 * input_group_count + 0.5)),
                ),
            )
            if (
                invalid_group_fields
                or not complete_group_sampling
                or record["sampled_group_draws"] != expected_group_draws
                or record["sampled_unique_groups"] != expected_unique_groups
                or record["oob_groups"] != expected_oob_groups
                or (
                    sampling == "without_replacement"
                    and (
                        expected_group_draws != expected_unique_groups
                        or expected_unique_groups
                        != expected_without_replacement_groups
                        or np.any(group_multiplicities > 1)
                    )
                )
                or (
                    sampling == "bootstrap"
                    and expected_group_draws != input_group_count
                )
                or not group_disjoint
                or record.get("group_disjoint") is not True
            ):
                raise ValueError(
                    "invalid DarkoFit model: private group-sampling metadata is invalid"
                )
        if record.get("policy_resolutions") != resolutions:
            raise ValueError(
                "invalid DarkoFit model: private ensemble-v3 member policy differs"
            )
        expected_constructor_lr = resolutions["learning_rate"]["resolved"]
        expected_constructor_colsample = resolutions["colsample"]["resolved"]
        if (
            type(record.get("constructor_learning_rate"))
            is not type(expected_constructor_lr)
            or record.get("constructor_learning_rate")
            != expected_constructor_lr
            or type(member.learning_rate) is not type(expected_constructor_lr)
            or member.learning_rate != expected_constructor_lr
            or type(record.get("constructor_colsample"))
            is not type(expected_constructor_colsample)
            or record.get("constructor_colsample")
            != expected_constructor_colsample
            or type(member.colsample) is not type(expected_constructor_colsample)
            or member.colsample != expected_constructor_colsample
            or record["fitted_thread_count"] != int(model.n_threads_)
            or record["best_iteration"] != int(member.best_n_estimators_)
            or record["best_iteration"] != int(model.best_iteration_)
            or record.get("resolved_learning_rate")
            != float(member.learning_rate_)
            or record.get("resolved_learning_rate") != float(model.lr_)
            or record.get("stop_reason")
            != str(getattr(model, "stop_reason_", "unknown"))
        ):
            raise ValueError(
                "invalid DarkoFit model: private ensemble-v3 fitted metadata differs"
            )
        if b3_schedule is None:
            if "prediction_thread_count" in record:
                raise ValueError(
                    "invalid DarkoFit model: sequential ensemble carries B3 "
                    "thread provenance"
                )
        elif (
            record.get("fitted_thread_count")
            != expected_member_thread_count
            or record.get("prediction_thread_count")
            != int(model.n_threads_)
        ):
            raise ValueError(
                "invalid DarkoFit model: B3 member thread provenance differs"
            )
        validation = getattr(model, "auto_params_", {}).get(
            "validation_split"
        )
        if (
            not isinstance(validation, Mapping)
            or validation.get("source") != "explicit_eval_set"
            or validation.get("train_n_samples") != record["sampled_rows"]
            or validation.get("eval_n_samples") != record["oob_rows"]
        ):
            raise ValueError(
                "invalid DarkoFit model: private ensemble-v3 OOB metadata differs"
            )


def _validate_loaded_public_ensemble_v3_metadata(
    metadata,
    members,
    *,
    classification,
    index_provenance,
    group_codes,
    base_constructor_params,
):
    """Validate the public v4 mapping through the frozen v3 recipe."""
    saved_base_params = metadata.get("base_constructor_params")
    if (
        metadata.get("version") != _ENSEMBLE_V3_PUBLIC_METADATA_VERSION
        or metadata.get("ensemble_mode") != "v3"
        or metadata.get("recipe_contract") != _ENSEMBLE_V3_PUBLIC_CONTRACT
        or metadata.get("recipe_version") != 1
        or metadata.get("public_fit_surface") is not True
        or "private_prototype" in metadata
        or "future_constructor_params" in metadata
        or not isinstance(saved_base_params, Mapping)
        or len(members) != _ENSEMBLE_V3_RELEASE_CANDIDATE_MEMBERS
    ):
        raise ValueError(
            "invalid DarkoFit model: public ensemble-v3 contract is invalid"
        )
    try:
        if _normalize_ensemble_mode(saved_base_params["ensemble_mode"]) != "v3":
            raise ValueError
        future_values, _ = _normalize_ensemble_v3_release_candidate_overrides(
            saved_base_params["ensemble_member_learning_rate"],
            saved_base_params["ensemble_member_colsample"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "invalid DarkoFit model: public ensemble-v3 constructor "
            "parameters are invalid"
        ) from exc
    proxy = dict(metadata)
    proxy.update({
        "version": _ENSEMBLE_V3_RELEASE_CANDIDATE_METADATA_VERSION,
        "private_prototype": _ENSEMBLE_V3_RELEASE_CANDIDATE,
        "public_fit_surface": False,
        "future_constructor_params": {
            "ensemble_mode": "v3",
            "ensemble_member_learning_rate": future_values["learning_rate"],
            "ensemble_member_colsample": future_values["colsample"],
        },
    })
    _validate_loaded_private_ensemble_v3_metadata(
        proxy,
        members,
        classification=classification,
        index_provenance=index_provenance,
        group_codes=group_codes,
        base_constructor_params=base_constructor_params,
        allow_legacy_param_schema=False,
    )


def _validate_loaded_ensemble_metadata(
    metadata,
    members,
    *,
    classification,
    index_provenance=None,
    group_codes=None,
    base_constructor_params=None,
):
    """Fail closed on contradictory fitted-ensemble provenance."""
    version = metadata.get("version")
    if _is_private_ensemble_v3_metadata(metadata):
        return _validate_loaded_private_ensemble_v3_metadata(
            metadata,
            members,
            classification=classification,
            index_provenance=index_provenance,
            group_codes=group_codes,
            base_constructor_params=base_constructor_params,
        )
    if _is_public_ensemble_v3_metadata(metadata):
        return _validate_loaded_public_ensemble_v3_metadata(
            metadata,
            members,
            classification=classification,
            index_provenance=index_provenance,
            group_codes=group_codes,
            base_constructor_params=base_constructor_params,
        )
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != 1
        or metadata.get("claim_tier") != "E"
        or metadata.get("default_changed") is not False
        or metadata.get("oob_early_stopping") is not True
    ):
        raise ValueError(
            "invalid DarkoFit model: ensemble provenance is invalid"
        )
    member_count = len(members)
    saved_member_count = metadata.get("member_count")
    if (
        isinstance(saved_member_count, bool)
        or not isinstance(saved_member_count, int)
        or saved_member_count != member_count
    ):
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
    shared_requested = metadata.get("shared_preprocessing_requested")
    if not isinstance(shared_requested, bool):
        raise ValueError(
            "invalid DarkoFit model: ensemble preprocessing request is invalid"
        )
    fallback_reason = metadata.get("shared_preprocessing_fallback_reason")
    if (
        shared == "numeric_target_free"
        and (not shared_requested or fallback_reason is not None)
    ):
        raise ValueError(
            "invalid DarkoFit model: shared ensemble preprocessing provenance "
            "is inconsistent"
        )
    if shared == "member_local" and (
        (
            shared_requested
            and fallback_reason
            not in {"categorical_or_ordinal_features", "non_numeric_dtype"}
        )
        or (not shared_requested and fallback_reason is not None)
    ):
        raise ValueError(
            "invalid DarkoFit model: member-local ensemble preprocessing "
            "provenance is inconsistent"
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
    for member in members:
        model = getattr(member, "model_", None)
        if classification:
            multiclass = getattr(member, "_multiclass", None)
            valid_model = (
                multiclass is True
                and isinstance(model, MulticlassBoosting)
            ) or (
                multiclass is False
                and isinstance(model, GradientBoosting)
                and getattr(model, "loss_name", None) == "Logloss"
            )
        else:
            valid_model = (
                isinstance(model, GradientBoosting)
                and getattr(model, "loss_name", None)
                in {"RMSE", "MAE", "Quantile"}
            )
        if not valid_model:
            raise ValueError(
                "invalid DarkoFit model: ensemble member model family is "
                "invalid"
            )
        if (
            getattr(member, "early_stopping", None) is not True
            or getattr(member, "use_best_model", None) is not True
            or getattr(member, "refit", None) is not False
        ):
            raise ValueError(
                "invalid DarkoFit model: ensemble member selection params do "
                "not match fitted OOB provenance"
            )
    for index, (seed, record, member) in enumerate(
        zip(seeds, records, members)
    ):
        member_index = (
            record.get("member") if isinstance(record, Mapping) else None
        )
        record_seed = (
            record.get("seed") if isinstance(record, Mapping) else None
        )
        member_random_state = member.get_params().get("random_state")
        core_random_state = getattr(member.model_, "random_state", None)
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or seed >= 2**31 - 1
            or not isinstance(record, Mapping)
            or isinstance(member_index, bool)
            or not isinstance(member_index, int)
            or member_index != index
            or isinstance(record_seed, bool)
            or not isinstance(record_seed, int)
            or record_seed < 0
            or record_seed >= 2**31 - 1
            or record_seed != seed
            or isinstance(member_random_state, bool)
            or not isinstance(member_random_state, int)
            or member_random_state != seed
            or isinstance(core_random_state, bool)
            or not isinstance(core_random_state, int)
            or core_random_state != seed
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
        validation = getattr(member.model_, "auto_params_", {}).get(
            "validation_split"
        )
        if (
            not isinstance(validation, Mapping)
            or validation.get("source") != "explicit_eval_set"
            or isinstance(validation.get("train_n_samples"), bool)
            or not isinstance(validation.get("train_n_samples"), int)
            or validation["train_n_samples"] != record["bootstrap_rows"]
            or isinstance(validation.get("eval_n_samples"), bool)
            or not isinstance(validation.get("eval_n_samples"), int)
            or validation["eval_n_samples"] != record["oob_rows"]
            or record["bootstrap_unique_rows"] > record["bootstrap_rows"]
        ):
            raise ValueError(
                "invalid DarkoFit model: ensemble member sampling metadata "
                "does not match its payload"
            )
        group_fields = (
            "sampled_group_draws",
            "sampled_unique_groups",
            "oob_groups",
        )
        if bootstrap == "rows":
            if (
                any(record.get(name) is not None for name in group_fields)
                or record.get("group_disjoint") is not None
                or record["bootstrap_rows"]
                != record["bootstrap_unique_rows"] + record["oob_rows"]
            ):
                raise ValueError(
                    "invalid DarkoFit model: row ensemble sampling metadata "
                    "is invalid"
                )
        elif (
            any(
                isinstance(record.get(name), bool)
                or not isinstance(record.get(name), int)
                or record[name] < 1
                for name in group_fields
            )
            or record["sampled_unique_groups"]
            > record["sampled_group_draws"]
            or record["sampled_unique_groups"] + record["oob_groups"]
            != record["sampled_group_draws"]
            or record["sampled_group_draws"] > record["bootstrap_rows"]
            or record["sampled_unique_groups"]
            > record["bootstrap_unique_rows"]
            or record["oob_groups"] > record["oob_rows"]
            or record.get("group_disjoint") is not True
        ):
            raise ValueError(
                "invalid DarkoFit model: group ensemble metadata is invalid"
            )
        if record["bootstrap_attempts"] > 128:
            raise ValueError(
                "invalid DarkoFit model: ensemble bootstrap attempt count is "
                "invalid"
            )
        learning_rate = record.get("learning_rate")
        stop_reason = record.get("stop_reason")
        try:
            learning_rate_value = float(learning_rate)
        except (TypeError, ValueError, OverflowError):
            learning_rate_value = float("nan")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not np.isfinite(learning_rate_value)
            or learning_rate_value != float(member.learning_rate_)
            or learning_rate_value != float(member.model_.lr_)
            or record["best_iteration"] != int(member.best_n_estimators_)
            or record["best_iteration"] != int(member.model_.best_iteration_)
            or not isinstance(stop_reason, str)
            or stop_reason != str(
                getattr(member.model_, "stop_reason_", "unknown")
            )
        ):
            raise ValueError(
                "invalid DarkoFit model: ensemble member fitted metadata does "
                "not match its payload"
            )


def _validate_loaded_wrapper_fitted_params(params, booster):
    """Reject wrapper constructor state that contradicts its fitted booster."""
    if "categorical_crosses" in params:
        try:
            _normalize_categorical_crosses(params["categorical_crosses"])
        except TypeError as exc:
            raise ValueError(
                "invalid DarkoFit model: wrapper categorical_crosses "
                "parameter is invalid"
            ) from exc
    fitted_loss = getattr(booster, "loss_name", None)
    if (
        "loss" in params
        and fitted_loss is not None
        and params["loss"] != fitted_loss
    ):
        raise ValueError(
            "invalid DarkoFit model: wrapper loss does not match the loaded "
            "booster"
        )
    fitted_oblivious_kernel = getattr(booster, "oblivious_kernel", None)
    if "oblivious_kernel" in params and fitted_oblivious_kernel is not None:
        try:
            saved_oblivious_kernel = _normalize_oblivious_kernel(
                params["oblivious_kernel"]
            )
            fitted_oblivious_kernel = _normalize_oblivious_kernel(
                fitted_oblivious_kernel
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "invalid DarkoFit model: wrapper oblivious kernel is invalid"
            ) from exc
        if saved_oblivious_kernel != fitted_oblivious_kernel:
            raise ValueError(
                "invalid DarkoFit model: wrapper oblivious kernel does not "
                "match the loaded booster"
            )
    if fitted_loss == "Quantile" and "alpha" in params:
        saved_alpha = params["alpha"]
        fitted_alpha = getattr(booster, "loss_kwargs", {}).get("alpha")
        try:
            saved_alpha_value = float(saved_alpha)
            fitted_alpha_value = float(fitted_alpha)
        except (TypeError, ValueError, OverflowError):
            saved_alpha_value = float("nan")
            fitted_alpha_value = float("nan")
        if (
            isinstance(saved_alpha, bool)
            or not isinstance(saved_alpha, (int, float))
            or not np.isfinite(saved_alpha_value)
            or not np.isfinite(fitted_alpha_value)
            or saved_alpha_value != fitted_alpha_value
        ):
            raise ValueError(
                "invalid DarkoFit model: wrapper quantile alpha does not "
                "match the loaded booster"
            )
    fitted_random_state = getattr(booster, "random_state", None)
    if isinstance(fitted_random_state, (bool, np.bool_)):
        raise ValueError(
            "invalid DarkoFit model: wrapper random state is invalid"
        )
    try:
        fitted_seed = normalize_random_state_seed(fitted_random_state)
    except ValueError as exc:
        raise ValueError(
            "invalid DarkoFit model: wrapper random state is invalid"
        ) from exc
    if fitted_seed is not None and fitted_seed < 0:
        raise ValueError(
            "invalid DarkoFit model: wrapper random state is invalid"
        )
    if "random_state" in params:
        if isinstance(params["random_state"], (bool, np.bool_)):
            raise ValueError(
                "invalid DarkoFit model: wrapper random state is invalid"
            )
        try:
            saved_seed = normalize_random_state_seed(params["random_state"])
        except ValueError as exc:
            raise ValueError(
                "invalid DarkoFit model: wrapper random state is invalid"
            ) from exc
        if (
            saved_seed is not None
            and saved_seed < 0
        ) or saved_seed != fitted_seed:
            raise ValueError(
                "invalid DarkoFit model: wrapper random state does not match "
                "the loaded booster"
            )


def _validate_loaded_wrapper_fitted_state(state, booster):
    """Bind wrapper fitted summaries and input width to the decoded booster."""
    refit = state.get("refit", False)

    if "learning_rate" in state:
        learning_rate = state["learning_rate"]
        fitted_learning_rate = getattr(booster, "lr_", None)
        try:
            learning_rate_value = float(learning_rate)
            fitted_learning_rate_value = float(fitted_learning_rate)
        except (TypeError, ValueError, OverflowError):
            learning_rate_value = float("nan")
            fitted_learning_rate_value = float("nan")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not np.isfinite(learning_rate_value)
            or not np.isfinite(fitted_learning_rate_value)
            or learning_rate_value != fitted_learning_rate_value
        ):
            raise ValueError(
                "invalid DarkoFit model: wrapper fitted learning rate does "
                "not match the loaded booster"
            )

    has_best_n = "best_n_estimators" in state
    best_n = state.get("best_n_estimators")
    if has_best_n:
        if (
            isinstance(best_n, bool)
            or not isinstance(best_n, int)
            or best_n < 0
        ):
            raise ValueError(
                "invalid DarkoFit model: wrapper fitted estimator count is "
                "invalid"
            )
        if not refit:
            expected_best_n = getattr(booster, "best_iteration_", None)
            if (
                isinstance(expected_best_n, bool)
                or not isinstance(expected_best_n, int)
                or best_n != expected_best_n
            ):
                raise ValueError(
                    "invalid DarkoFit model: wrapper fitted estimator count "
                    "does not match the loaded booster"
                )
        else:
            strategy = state.get("refit_strategy")
            exponent = _REFIT_STRATEGY_EXPONENT[strategy]
            expected_refit_n = best_n
            if exponent:
                if (
                    "selection_n_total" not in state
                    or "selection_n_train" not in state
                ):
                    raise ValueError(
                        "invalid DarkoFit model: scaled refit strategy has no "
                        "selection sample counts"
                    )
                total = state["selection_n_total"]
                train = state["selection_n_train"]
                try:
                    expected_refit_n = int(
                        np.ceil(best_n * ((total / train) ** exponent))
                    )
                except (OverflowError, ValueError) as exc:
                    raise ValueError(
                        "invalid DarkoFit model: wrapper selection and refit "
                        "estimator counts are inconsistent"
                    ) from exc
            if state.get("refit_n_estimators") != expected_refit_n:
                raise ValueError(
                    "invalid DarkoFit model: wrapper selection and refit "
                    "estimator counts are inconsistent"
                )
    elif refit:
        raise ValueError(
            "invalid DarkoFit model: refit fitted state has no selection "
            "estimator count"
        )

    if "best_score" in state:
        best_score = state["best_score"]
        try:
            best_score_value = float(best_score)
        except (TypeError, ValueError, OverflowError):
            best_score_value = float("nan")
        if (
            isinstance(best_score, bool)
            or not isinstance(best_score, (int, float))
            or not np.isfinite(best_score_value)
        ):
            raise ValueError(
                "invalid DarkoFit model: wrapper fitted score is invalid"
            )
        if not refit:
            fitted_score = getattr(booster, "best_score_", None)
            try:
                score_matches = (
                    best_score_value == float(fitted_score)
                )
            except (TypeError, ValueError, OverflowError):
                score_matches = False
            if not score_matches:
                raise ValueError(
                    "invalid DarkoFit model: wrapper fitted score does not "
                    "match the loaded booster"
                )

    fitted_n_features = _infer_model_n_features(booster)
    if "n_features_in" in state:
        n_features = state["n_features_in"]
        if (
            isinstance(n_features, bool)
            or not isinstance(n_features, int)
            or n_features < 1
            or fitted_n_features is None
            or n_features != fitted_n_features
        ):
            raise ValueError(
                "invalid DarkoFit model: wrapper input feature count does "
                "not match the loaded booster"
            )
    if "feature_names_in" in state:
        feature_names = state["feature_names_in"]
        if (
            not isinstance(feature_names, list)
            or fitted_n_features is None
            or len(feature_names) != fitted_n_features
            or not all(isinstance(name, str) for name in feature_names)
        ):
            raise ValueError(
                "invalid DarkoFit model: feature name state does not match "
                "the loaded booster input width"
            )


def _validate_loaded_linear_residual_params(params, state):
    """Reject constructor/state contradictions in linear-residual archives."""
    if (
        "linear_residual_enabled" not in state
        and "linear_residual_active" not in state
    ):
        return
    enabled = state.get(
        "linear_residual_enabled",
        state.get("linear_residual_active", False),
    )
    active = state.get("linear_residual_active", False)
    if not isinstance(enabled, bool) or not isinstance(active, bool):
        raise ValueError(
            "invalid DarkoFit model: linear residual flags must be booleans"
        )
    if active and not enabled:
        raise ValueError(
            "invalid DarkoFit model: active linear residual state cannot be "
            "disabled"
        )
    if "linear_residual_version" in state:
        version = state["linear_residual_version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != 1
        ):
            raise ValueError(
                "invalid DarkoFit model: linear residual version is invalid"
            )
    if "linear_residual" in params and (
        not isinstance(params["linear_residual"], bool)
        or params["linear_residual"] is not enabled
    ):
        raise ValueError(
            "invalid DarkoFit model: linear residual parameter does not "
            "match its fitted state"
        )
    comparisons = (
        (
            "linear_residual_fit_intercept",
            "linear_residual_fit_intercept",
            bool,
        ),
        (
            "linear_residual_standardize",
            "linear_residual_standardize",
            bool,
        ),
    )
    for param_name, state_name, expected_type in comparisons:
        if param_name in params and not isinstance(
            params[param_name], expected_type
        ):
            raise ValueError(
                f"invalid DarkoFit model: {param_name} is invalid"
            )
        if state_name in state and not isinstance(
            state[state_name], expected_type
        ):
            raise ValueError(
                f"invalid DarkoFit model: {state_name} is invalid"
            )
        if (
            param_name in params
            and state_name in state
            and params[param_name] is not state[state_name]
        ):
            raise ValueError(
                f"invalid DarkoFit model: {param_name} does not match its "
                "fitted state"
            )
    alpha_values = {}
    for section_name, section in (("parameter", params), ("state", state)):
        if "linear_residual_alpha" not in section:
            continue
        alpha = section["linear_residual_alpha"]
        try:
            alpha_value = float(alpha)
        except (TypeError, ValueError, OverflowError):
            alpha_value = float("nan")
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not np.isfinite(alpha_value)
            or alpha_value < 0.0
        ):
            raise ValueError(
                "invalid DarkoFit model: linear_residual_alpha "
                f"{section_name} is invalid"
            )
        alpha_values[section_name] = alpha_value
    if (
        len(alpha_values) == 2
        and alpha_values["parameter"] != alpha_values["state"]
    ):
        raise ValueError(
            "invalid DarkoFit model: linear_residual_alpha does not match "
            "its fitted state"
        )


def _validate_loaded_refit_state(state, booster):
    """Reject wrapper selection/refit state that contradicts its booster."""
    refit = state.get("refit", False)
    if not isinstance(refit, bool):
        raise ValueError(
            "invalid DarkoFit model: refit fitted state must be a boolean"
        )
    auto_params = getattr(booster, "auto_params_", {})
    validation = auto_params.get("validation_split")
    if isinstance(validation, Mapping):
        validation_refit = validation.get("refit")
        if (
            not isinstance(validation_refit, bool)
            or validation_refit is not refit
        ):
            raise ValueError(
                "invalid DarkoFit model: refit wrapper and booster "
                "provenance disagree"
            )
    elif refit:
        raise ValueError(
            "invalid DarkoFit model: refit wrapper state has no booster "
            "provenance"
        )

    if refit:
        refit_n = state.get("refit_n_estimators")
        strategy = state.get("refit_strategy")
        selection_validation = auto_params.get("selection_validation_split")
        if (
            isinstance(refit_n, bool)
            or not isinstance(refit_n, int)
            or refit_n != len(getattr(booster, "trees_", ()))
            or not isinstance(strategy, str)
            or strategy not in _REFIT_STRATEGY_EXPONENT
            or state.get("selection_model_persisted") is not False
            or not isinstance(selection_validation, Mapping)
            or validation.get("source") != "refit_full_data"
        ):
            raise ValueError(
                "invalid DarkoFit model: refit fitted state is inconsistent"
            )
    elif (
        state.get("refit_n_estimators") is not None
        or state.get("refit_strategy") is not None
        or "selection_model_persisted" in state
    ):
        raise ValueError(
            "invalid DarkoFit model: non-refit model has refit fitted state"
        )

    has_total = "selection_n_total" in state
    has_train = "selection_n_train" in state
    if has_total != has_train:
        raise ValueError(
            "invalid DarkoFit model: selection sample counts are incomplete"
        )
    selection_validation = (
        auto_params.get("selection_validation_split")
        if refit else validation
    )
    if has_total:
        total = state["selection_n_total"]
        train = state["selection_n_train"]
        selection_source = (
            selection_validation.get("source")
            if isinstance(selection_validation, Mapping)
            else None
        )
        if (
            isinstance(total, bool)
            or not isinstance(total, int)
            or total < 2
            or isinstance(train, bool)
            or not isinstance(train, int)
            or train < 1
            or train >= total
            or not isinstance(selection_validation, Mapping)
            or not isinstance(selection_source, str)
            or not selection_source.startswith("automatic")
            or selection_validation.get("original_n_samples") != total
            or selection_validation.get("train_n_samples") != train
        ):
            raise ValueError(
                "invalid DarkoFit model: selection sample counts do not "
                "match booster provenance"
            )
        if (
            refit
            and (
                not isinstance(validation, Mapping)
                or validation.get("original_n_samples") != total
                or validation.get("train_n_samples") != total
            )
        ):
            raise ValueError(
                "invalid DarkoFit model: selection and refit sample counts "
                "do not match booster provenance"
            )
    elif (
        isinstance(selection_validation, Mapping)
        and str(selection_validation.get("source", "")).startswith("automatic")
    ):
        raise ValueError(
            "invalid DarkoFit model: automatic selection provenance has no "
            "wrapper sample counts"
        )


def _validate_loaded_dist_calibration_state(state, booster, params=None):
    """Bind prediction-changing calibration state to booster provenance."""
    auto_params = getattr(booster, "auto_params_", {})
    fitted = auto_params.get("dist_calibration")
    fitted_sigma = auto_params.get("sigma_calibration")
    diagnostics = auto_params.get("diagnostics", {})
    diagnostic_dist = (
        diagnostics.get("dist_calibration")
        if isinstance(diagnostics, Mapping)
        else None
    )
    diagnostic_sigma = (
        diagnostics.get("sigma_calibration")
        if isinstance(diagnostics, Mapping)
        else None
    )
    calibration_metadata = (
        fitted,
        fitted_sigma,
        diagnostic_dist,
        diagnostic_sigma,
    )
    if any(value is not None for value in calibration_metadata) and (
        not all(
            isinstance(value, Mapping)
            for value in calibration_metadata
        )
        or any(
            dict(value) != dict(fitted)
            for value in calibration_metadata[1:]
        )
    ):
        raise ValueError(
            "invalid DarkoFit model: distribution calibration booster "
            "provenance is inconsistent"
        )

    def finite_number(value, *, positive=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(result) or (positive and result <= 0.0):
            return None
        return result

    method = state.get(
        "dist_calibration", state.get("sigma_calibration")
    )
    fitted_method = (
        fitted.get("method") if isinstance(fitted, Mapping) else None
    )
    param_method = None
    has_calibration_param = (
        isinstance(params, Mapping)
        and (
            "dist_calibration" in params
            or "sigma_calibration" in params
        )
    )
    if has_calibration_param:
        try:
            param_method = _normalize_dist_calibration(
                params.get("dist_calibration"),
                params.get("sigma_calibration"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "invalid DarkoFit model: distribution calibration parameter "
                "is invalid"
            ) from exc
    if method is None:
        if fitted_method is not None or param_method is not None:
            raise ValueError(
                "invalid DarkoFit model: distribution calibration parameter, "
                "wrapper state, and booster provenance disagree"
            )
        if isinstance(fitted, Mapping):
            dist_scale = finite_number(
                fitted.get("dist_scale"), positive=True
            )
            sigma_scale = finite_number(
                fitted.get("sigma_scale"), positive=True
            )
            if (
                dist_scale != 1.0
                or sigma_scale != 1.0
                or fitted.get("source") != "none"
            ):
                raise ValueError(
                    "invalid DarkoFit model: uncalibrated distribution "
                    "provenance is invalid"
                )
        if any(
            key in state
            for key in (
                "dist_scale",
                "sigma_scale",
                "dist_affine_a",
                "sigma_affine_a",
                "dist_group_affine",
                "sigma_group_affine",
            )
        ):
            raise ValueError(
                "invalid DarkoFit model: uncalibrated distribution wrapper "
                "contains fitted calibration state"
            )
        return
    try:
        method = _normalize_dist_calibration(method)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "invalid DarkoFit model: distribution calibration method is "
            "invalid"
        ) from exc
    if not isinstance(fitted, Mapping) or fitted_method != method:
        raise ValueError(
            "invalid DarkoFit model: distribution calibration wrapper and "
            "booster provenance disagree"
        )
    if has_calibration_param and param_method != method:
        raise ValueError(
            "invalid DarkoFit model: distribution calibration parameter does "
            "not match its fitted state"
        )
    if state.get("sigma_calibration") != method:
        raise ValueError(
            "invalid DarkoFit model: distribution calibration wrapper and "
            "booster provenance disagree"
        )

    def require_number(state_key, metadata_key, *, positive=False):
        value = state.get(state_key)
        metadata_value = fitted.get(metadata_key)
        value_float = finite_number(value, positive=positive)
        metadata_float = finite_number(metadata_value, positive=positive)
        if (
            value_float is None
            or metadata_float is None
            or value_float != metadata_float
        ):
            raise ValueError(
                "invalid DarkoFit model: distribution calibration wrapper "
                "and booster provenance disagree"
            )
        return value_float

    dist_scale = require_number(
        "dist_scale", "dist_scale", positive=True
    )
    sigma_scale = require_number(
        "sigma_scale", "sigma_scale", positive=True
    )
    if dist_scale != sigma_scale:
        raise ValueError(
            "invalid DarkoFit model: distribution calibration aliases "
            "disagree"
        )
    if (
        state.get("dist_scale_source") != fitted.get("source")
        or state.get("sigma_scale_source") != fitted.get("source")
        or not isinstance(fitted.get("source"), str)
        or not fitted.get("source")
        or fitted.get("source") == "none"
    ):
        raise ValueError(
            "invalid DarkoFit model: distribution calibration wrapper and "
            "booster provenance disagree"
        )
    for state_key, metadata_key in (
        ("dist_affine_a", "dist_affine_a"),
        ("dist_affine_b", "dist_affine_b"),
        ("sigma_affine_a", "sigma_affine_a"),
        ("sigma_affine_b", "sigma_affine_b"),
    ):
        if state_key in state or metadata_key in fitted:
            require_number(state_key, metadata_key)
    if method in {"affine", "per_metric_affine"}:
        affine_pairs = (
            ("dist_affine_a", "sigma_affine_a"),
            ("dist_affine_b", "sigma_affine_b"),
        )
        for dist_key, sigma_key in affine_pairs:
            if dist_key not in state or sigma_key not in state:
                raise ValueError(
                    "invalid DarkoFit model: affine distribution calibration "
                    "state is incomplete"
                )
            if finite_number(state[dist_key]) != finite_number(
                state[sigma_key]
            ):
                raise ValueError(
                    "invalid DarkoFit model: distribution calibration aliases "
                    "disagree"
                )
    for state_key, metadata_key in (
        ("dist_group_affine", "dist_group_affine"),
        ("sigma_group_affine", "sigma_group_affine"),
        ("dist_calibration_feature", "dist_calibration_feature"),
        (
            "dist_calibration_feature_index",
            "dist_calibration_feature_index",
        ),
        ("dist_calibration_feature_name", "dist_calibration_feature_name"),
    ):
        if state_key in state or metadata_key in fitted:
            if state.get(state_key) != fitted.get(metadata_key):
                raise ValueError(
                    "invalid DarkoFit model: distribution calibration wrapper "
                    "and booster provenance disagree"
                )
    for state_key, metadata_key in (
        (
            "dist_mean_calibration_numerator",
            "mean_calibration_numerator",
        ),
        (
            "dist_mean_calibration_denominator",
            "mean_calibration_denominator",
        ),
    ):
        if state_key in state or metadata_key in fitted:
            require_number(state_key, metadata_key)
    if (
        state.get("dist_calibration_fallback_reason")
        != fitted.get("fallback_reason")
        or state.get("sigma_calibration_fallback_reason")
        != fitted.get("fallback_reason")
        or state.get("dist_mean_calibration_objective")
        != fitted.get("mean_calibration_objective")
    ):
        raise ValueError(
            "invalid DarkoFit model: distribution calibration wrapper and "
            "booster provenance disagree"
        )
    if (
        method == "per_metric_affine"
        and isinstance(params, Mapping)
        and "dist_calibration_feature" in params
        and (
            isinstance(params["dist_calibration_feature"], bool)
            or isinstance(state.get("dist_calibration_feature"), bool)
            or type(params["dist_calibration_feature"])
            is not type(state.get("dist_calibration_feature"))
            or params["dist_calibration_feature"]
            != state.get("dist_calibration_feature")
        )
    ):
        raise ValueError(
            "invalid DarkoFit model: distribution calibration feature "
            "parameter does not match its fitted state"
        )
    if method == "per_metric_affine":
        records = state.get("dist_group_affine")
        feature = state.get("dist_calibration_feature")
        feature_index = state.get("dist_calibration_feature_index")
        feature_name = state.get("dist_calibration_feature_name")
        group_count = fitted.get("group_count")
        group_fallback_count = fitted.get("group_fallback_count")
        saved_feature_names = getattr(booster, "feature_names_in_", None)
        if (
            not isinstance(records, list)
            or not records
            or records != state.get("sigma_group_affine")
            or isinstance(feature, bool)
            or not isinstance(feature, (int, str))
            or isinstance(feature_index, bool)
            or not isinstance(feature_index, int)
            or feature_index < 0
            or feature_index >= int(getattr(booster, "n_features_in_", 0))
            or (
                feature_name is not None
                and not isinstance(feature_name, str)
            )
            or isinstance(group_count, bool)
            or not isinstance(group_count, int)
            or group_count != len(records)
            or isinstance(group_fallback_count, bool)
            or not isinstance(group_fallback_count, int)
            or (
                saved_feature_names is None
                and feature_name is not None
            )
            or (
                saved_feature_names is not None
                and feature_name
                != str(np.asarray(saved_feature_names)[feature_index])
            )
            or (
                isinstance(feature, str)
                and feature_name != feature
            )
        ):
            raise ValueError(
                "invalid DarkoFit model: grouped distribution calibration "
                "state is invalid"
            )
        seen_groups = []
        fallback_count = 0
        validation_count = 0
        positive_weight_count = 0
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(
                    "invalid DarkoFit model: grouped distribution calibration "
                    "record is invalid"
                )
            group = record.get("group")
            record_validation_count = record.get("validation_n_samples")
            record_positive_weight_count = record.get(
                "validation_positive_weight_n"
            )
            record_effective_n = finite_number(
                record.get("validation_effective_n"), positive=True
            )
            record_weight_sum = finite_number(
                record.get("validation_weight_sum"), positive=True
            )
            record_fallback = record.get("fallback_reason")
            if (
                not isinstance(group, (str, bool, int, float))
                or (
                    isinstance(group, float)
                    and not np.isfinite(group)
                )
                or finite_number(
                    record.get("sigma_scale"), positive=True
                ) is None
                or finite_number(record.get("sigma_affine_a")) is None
                or finite_number(record.get("sigma_affine_b")) is None
                or isinstance(record_validation_count, bool)
                or not isinstance(record_validation_count, int)
                or record_validation_count < 1
                or isinstance(record_positive_weight_count, bool)
                or not isinstance(record_positive_weight_count, int)
                or record_positive_weight_count < 1
                or record_positive_weight_count != record_validation_count
                or record_effective_n is None
                or record_effective_n - 1e-9
                > record_positive_weight_count
                or record_weight_sum is None
                or (
                    record_fallback is not None
                    and record_fallback
                    not in {
                        "small_group",
                        "slope_bound",
                        "non_finite_profile",
                    }
                )
                or any(group == prior for prior in seen_groups)
            ):
                raise ValueError(
                    "invalid DarkoFit model: grouped distribution calibration "
                    "record is invalid"
                )
            seen_groups.append(group)
            validation_count += record_validation_count
            positive_weight_count += record_positive_weight_count
            fallback_count += record_fallback is not None
        fitted_validation_count = fitted.get("validation_n_samples")
        fitted_positive_weight_count = fitted.get(
            "validation_positive_weight_n"
        )
        if (
            group_fallback_count != fallback_count
            or isinstance(fitted_validation_count, bool)
            or not isinstance(fitted_validation_count, int)
            or fitted_validation_count < validation_count
            or isinstance(fitted_positive_weight_count, bool)
            or not isinstance(fitted_positive_weight_count, int)
            or fitted_positive_weight_count != validation_count
            or fitted_positive_weight_count != positive_weight_count
        ):
            raise ValueError(
                "invalid DarkoFit model: grouped distribution calibration "
                "aggregate provenance is invalid"
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


def _normalize_interval_calibration(calibration):
    if calibration is None or calibration is False:
        return None
    mode = str(calibration).lower().replace("-", "_")
    if mode in {"none", "off", "false", "no"}:
        return None
    if mode in {"conformal", "split_conformal"}:
        return "conformal"
    raise ValueError(
        "interval_calibration must be None, False, or 'conformal'"
    )


def _conformal_order_statistic(scores, alpha):
    values = np.asarray(scores, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
    ):
        raise ValueError("conformal calibration scores are invalid")
    rank = int(math.ceil((values.size + 1) * (1.0 - float(alpha))))
    if rank > values.size:
        raise ValueError(
            "the conformal calibration set is too small for the requested "
            f"alpha={float(alpha):g}; use at least "
            f"{int(math.ceil(1.0 / float(alpha) - 1.0))} calibration rows "
            "or request a wider miscoverage level"
        )
    index = max(rank - 1, 0)
    return float(np.partition(values, index)[index]), rank


def _reserve_conformal_holdout(eval_set, random_state, *, selection_needed):
    """Keep conformal rows untouched by fitting, selection, and calibration."""
    X_eval, y_eval = eval_set
    n_eval = int(len(y_eval))
    if not selection_needed:
        return None, eval_set, {
            "selection_n_samples": 0,
            "calibration_n_samples": n_eval,
            "holdout_fraction": 1.0,
        }
    if n_eval < 4:
        raise ValueError(
            "interval_calibration='conformal' needs at least 4 validation "
            "rows when the validation set is also used for model selection "
            "or distribution calibration"
        )
    seed = normalize_random_state_seed(random_state)
    rng = np.random.default_rng(
        None if seed is None else int(seed) ^ 0x434F4E46
    )
    order = rng.permutation(n_eval)
    calibration_n = max(1, n_eval // 2)
    calibration_idx = order[:calibration_n]
    selection_idx = order[calibration_n:]
    selection_set = (
        X_eval[selection_idx],
        np.asarray(y_eval)[selection_idx],
    )
    calibration_set = (
        X_eval[calibration_idx],
        np.asarray(y_eval)[calibration_idx],
    )
    return selection_set, calibration_set, {
        "selection_n_samples": int(selection_idx.size),
        "calibration_n_samples": int(calibration_idx.size),
        "holdout_fraction": float(calibration_idx.size / n_eval),
    }


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
    if isinstance(feature, (bool, np.bool_)):
        raise ValueError("dist_calibration_feature must be a column name or index")
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


def _ordinal_category_order_sha256(records):
    """Bind ordered category values and scalar types to fitted provenance."""
    encoded_records = []
    for record in records:
        encoded_categories = []
        for value in record["categories"]:
            if isinstance(value, bool):
                encoded = ("bool", "1" if value else "0")
            elif isinstance(value, int):
                encoded = ("int", str(value))
            elif isinstance(value, float):
                encoded = ("float", value.hex())
            else:
                encoded = ("str", value)
            encoded_categories.append(encoded)
        encoded_records.append((int(record["index"]), encoded_categories))
    payload = json.dumps(
        encoded_records,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordinal_metadata_payload(mode, records, nominal_cat_count):
    metadata = {}
    if records:
        metadata["state_version"] = _ORDINAL_STATE_VERSION
    metadata.update({
        "mode": mode,
        "active": bool(records),
        "feature_count": len(records),
        "feature_indices": [int(record["index"]) for record in records],
        "feature_names": [record.get("name") for record in records],
        "sources": [record["source"] for record in records],
    })
    if records:
        metadata["category_order_sha256"] = (
            _ordinal_category_order_sha256(records)
        )
    metadata.update({
        "nominal_categorical_count": int(nominal_cat_count),
        "added_columns": 0,
        "target_stat_blocks_added": 0,
        "target_used": False,
        "unknown_policy": "fail_closed",
        "missing_policy": "numeric_missing_bin",
    })
    return metadata


def _ordinal_metadata_matches(value, expected):
    if not isinstance(value, Mapping):
        return False
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_encoded = json.dumps(
        expected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return encoded == expected_encoded


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


def _restore_fitted_state_on_fit_failure(method):
    """Keep an already-fitted wrapper transactional across failed refits."""

    @wraps(method)
    def transactional_fit(self, *args, **kwargs):
        previous_state = dict(self.__dict__)
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            self.__dict__.clear()
            self.__dict__.update(previous_state)
            raise

    return transactional_fit


class _RefitParamsMixin:
    """Shared fitted-model metadata and full-data refit helpers."""

    def _clear_ensemble_state(self):
        for name in (
            "estimators_",
            "ensemble_metadata_",
            "_ensemble_group_codes_",
            "ensemble_best_iterations_",
            "ensemble_learning_rates_",
            "expected_value_",
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
        self._clear_refit_selection_metadata()
        self._clear_linear_residual_state()
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
            elif hasattr(self, name):
                delattr(self, name)
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
        if (
            _is_private_ensemble_v3_metadata(self.ensemble_metadata_)
            or _is_public_ensemble_v3_metadata(self.ensemble_metadata_)
        ):
            base_params = self.ensemble_metadata_.get(
                "base_constructor_params"
            )
            if not isinstance(base_params, Mapping):
                raise ValueError(
                    "cannot save ensemble-v3 without canonical base "
                    "constructor parameters"
                )
            return dict(base_params)
        params = self.get_params()
        first = self.estimators_[0]
        fitted_member_params = first._wrapper_params_header()
        fitted_loss = getattr(first.model_, "loss_name", None)
        if "loss" in params and fitted_loss is not None:
            params["loss"] = fitted_loss
        if "alpha" in params and fitted_loss == "Quantile":
            params["alpha"] = float(first.model_.loss_kwargs["alpha"])
        if "oblivious_kernel" in params:
            fitted_kernels = {
                _normalize_oblivious_kernel(member.model_.oblivious_kernel)
                for member in self.estimators_
            }
            member_kernels = {
                _normalize_oblivious_kernel(
                    member._wrapper_params_header()["oblivious_kernel"]
                )
                for member in self.estimators_
            }
            if len(fitted_kernels) != 1 or member_kernels != fitted_kernels:
                raise ValueError(
                    "cannot save ensemble whose oblivious kernel differs "
                    "across member payloads"
                )
            params["oblivious_kernel"] = next(iter(fitted_kernels))
        for name in (
            "early_stopping",
            "use_best_model",
            "refit_strategy",
            "preset",
            "selection_rounds",
            "tree_mode",
            "interval_calibration",
            "dist_calibration",
            "sigma_calibration",
            "linear_residual",
            "linear_residual_alpha",
            "linear_residual_features",
            "linear_residual_fit_intercept",
            "linear_residual_standardize",
        ):
            if name in params and name in fitted_member_params:
                params[name] = fitted_member_params[name]
        params["n_ensembles"] = len(self.estimators_)
        params["ensemble_bootstrap"] = self.ensemble_metadata_["bootstrap"]
        params["ensemble_shared_preprocessing"] = bool(
            self.ensemble_metadata_["shared_preprocessing_requested"]
        )
        params["refit"] = False
        fit_seed = self.ensemble_metadata_["fit_random_state_seed"]
        params["random_state"] = (
            None if fit_seed is None else int(fit_seed)
        )
        return params

    @classmethod
    def _load_ensemble_model(cls, path):
        import io
        from .serialization import load_ensemble

        archive = load_ensemble(path)
        if archive is None:
            return None
        header, payloads, index_provenance, group_codes = archive
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
        for payload in payloads:
            if load_ensemble(io.BytesIO(payload)) is not None:
                raise ValueError(
                    "invalid DarkoFit model: nested ensemble member detected"
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
        private_provenance = _is_private_ensemble_v3_metadata(metadata)
        public_v3 = _is_public_ensemble_v3_metadata(metadata)
        v3_provenance = private_provenance or public_v3
        classification = issubclass(cls, ClassifierMixin)
        _validate_loaded_ensemble_metadata(
            metadata,
            members,
            classification=classification,
            index_provenance=index_provenance,
            group_codes=group_codes,
            base_constructor_params=params,
        )
        if (
            (
                not private_provenance
                and _normalize_ensemble_bootstrap(est.ensemble_bootstrap)
                != metadata["bootstrap"]
            )
            or not isinstance(
                est.ensemble_shared_preprocessing, (bool, np.bool_)
            )
            or bool(est.ensemble_shared_preprocessing)
            != metadata["shared_preprocessing_requested"]
            or est.refit is not False
        ):
            raise ValueError(
                "invalid DarkoFit model: ensemble params do not match fitted "
                "provenance"
            )
        if public_v3:
            try:
                mode, future_values, explicit_values = (
                    _resolve_public_ensemble_surface(est)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "invalid DarkoFit model: public ensemble-v3 parameters "
                    "are invalid"
                ) from exc
            if (
                mode != "v3"
                or len(members) != _ENSEMBLE_V3_RELEASE_CANDIDATE_MEMBERS
                or set(explicit_values)
                != set(metadata.get("explicit_user_params", ()))
                or not _private_ensemble_v3_values_equal(
                    future_values,
                    {
                        "learning_rate": est.ensemble_member_learning_rate,
                        "colsample": est.ensemble_member_colsample,
                    },
                )
            ):
                raise ValueError(
                    "invalid DarkoFit model: public ensemble-v3 parameters "
                    "do not match fitted provenance"
                )
        fit_seed = metadata.get("fit_random_state_seed")
        outer_random_state = est.random_state
        if (
            (
                fit_seed is not None
                and (
                    isinstance(fit_seed, bool)
                    or not isinstance(fit_seed, int)
                    or fit_seed < 0
                )
            )
            or (
                fit_seed is None
                and outer_random_state is not None
            )
            or (
                fit_seed is not None
                and (
                    isinstance(outer_random_state, bool)
                    or not isinstance(outer_random_state, int)
                    or outer_random_state != fit_seed
                )
            )
        ):
            raise ValueError(
                "invalid DarkoFit model: ensemble random state does not match "
                "fitted provenance"
            )
        if hasattr(est, "loss"):
            fitted_losses = {
                member._fitted_loss_name() for member in members
            }
            if len(fitted_losses) != 1 or est.loss != next(
                iter(fitted_losses)
            ):
                raise ValueError(
                    "invalid DarkoFit model: ensemble loss does not match its "
                    "member payloads"
                )
            if est.loss == "Quantile":
                try:
                    outer_alpha_param = est.alpha
                    outer_alpha = float(est.alpha)
                    fitted_alphas = {
                        float(member.model_.loss_kwargs["alpha"])
                        for member in members
                    }
                    member_alphas = {
                        float(member.alpha) for member in members
                    }
                except (KeyError, TypeError, ValueError, OverflowError):
                    outer_alpha = float("nan")
                    fitted_alphas = set()
                    member_alphas = set()
                if (
                    isinstance(outer_alpha_param, bool)
                    or not isinstance(outer_alpha_param, (int, float))
                    or not np.isfinite(outer_alpha)
                    or len(fitted_alphas) != 1
                    or member_alphas != fitted_alphas
                    or outer_alpha != next(iter(fitted_alphas))
                ):
                    raise ValueError(
                        "invalid DarkoFit model: ensemble quantile does not "
                        "match its member payloads"
                    )
            linear_bool_params = (
                "linear_residual",
                "linear_residual_fit_intercept",
                "linear_residual_standardize",
            )
            if any(
                not isinstance(getattr(est, name), bool)
                for name in linear_bool_params
            ):
                raise ValueError(
                    "invalid DarkoFit model: ensemble linear residual "
                    "parameters are invalid"
                )
            linear_alpha = est.linear_residual_alpha
            try:
                linear_alpha_value = float(linear_alpha)
            except (TypeError, ValueError, OverflowError):
                linear_alpha_value = float("nan")
            if (
                isinstance(linear_alpha, bool)
                or not isinstance(linear_alpha, (int, float))
                or not np.isfinite(linear_alpha_value)
            ):
                raise ValueError(
                    "invalid DarkoFit model: ensemble linear residual "
                    "parameters are invalid"
                )
            for name in (
                *linear_bool_params,
                "linear_residual_alpha",
                "linear_residual_features",
            ):
                outer_value = getattr(est, name)
                if any(
                    not np.array_equal(
                        np.asarray(outer_value, dtype=object),
                        np.asarray(getattr(member, name), dtype=object),
                    )
                    for member in members
                ):
                    raise ValueError(
                        "invalid DarkoFit model: ensemble linear residual "
                        "parameters do not match member payloads"
                    )
        if not v3_provenance:
            for name in (
                "early_stopping",
                "use_best_model",
                "refit_strategy",
                "preset",
                "selection_rounds",
                "tree_mode",
                "interval_calibration",
                "dist_calibration",
                "sigma_calibration",
                "oblivious_kernel",
            ):
                if not hasattr(est, name):
                    continue
                outer_value = getattr(est, name)
                member_values = [
                    getattr(member, name) for member in members
                ]
                if (
                    any(type(value) is not type(member_values[0])
                        for value in member_values[1:])
                    or type(outer_value) is not type(member_values[0])
                    or outer_value != member_values[0]
                    or any(
                        value != member_values[0]
                        for value in member_values[1:]
                    )
                ):
                    raise ValueError(
                        f"invalid DarkoFit model: ensemble {name} parameter "
                        "does not match member payloads"
                    )
        if fit_seed is not None:
            expected_seeds = [
                int(value)
                for value in np.random.default_rng(fit_seed).integers(
                    0, 2**31 - 1, size=len(members)
                )
            ]
            if metadata["member_seeds"] != expected_seeds:
                raise ValueError(
                    "invalid DarkoFit model: ensemble member seeds do not match "
                    "the fitted random state"
                )
        feature_counts = {
            int(member.n_features_in_) for member in members
        }
        if len(feature_counts) != 1:
            raise ValueError(
                "invalid DarkoFit model: ensemble members disagree on input "
                "feature count"
            )
        input_feature_count = metadata.get("input_feature_count")
        if (
            isinstance(input_feature_count, bool)
            or not isinstance(input_feature_count, int)
            or input_feature_count != next(iter(feature_counts))
        ):
            raise ValueError(
                "invalid DarkoFit model: ensemble input feature count does not "
                "match its payload"
            )
        reference_names = getattr(members[0], "feature_names_in_", None)
        reference_ordinals = getattr(members[0], "ordinal_features_", ())
        reference_cat_features = tuple(
            int(index)
            for index in members[0].model_.prep_.cat_features_
        )
        if any(
            (
                (reference_names is None)
                != (getattr(member, "feature_names_in_", None) is None)
            )
            or (
                reference_names is not None
                and not np.array_equal(
                    np.asarray(reference_names),
                    np.asarray(member.feature_names_in_),
                )
            )
            or getattr(member, "ordinal_features_", ()) != reference_ordinals
            or tuple(
                int(index) for index in member.model_.prep_.cat_features_
            )
            != reference_cat_features
            for member in members[1:]
        ):
            raise ValueError(
                "invalid DarkoFit model: ensemble members disagree on input "
                "feature schema"
            )
        if metadata["shared_preprocessing"] == "numeric_target_free":
            if reference_ordinals or reference_cat_features:
                raise ValueError(
                    "invalid DarkoFit model: shared preprocessing provenance "
                    "contradicts the fitted feature schema"
                )
            reference_prep = members[0].model_.prep_
            reference_binner = reference_prep.binner_
            if any(
                not np.array_equal(
                    np.asarray(reference_prep.num_features_),
                    np.asarray(member.model_.prep_.num_features_),
                )
                or not np.array_equal(
                    np.asarray(reference_prep.feature_map_),
                    np.asarray(member.model_.prep_.feature_map_),
                )
                or not np.array_equal(
                    np.asarray(reference_binner._borders_flat_),
                    np.asarray(
                        member.model_.prep_.binner_._borders_flat_
                    ),
                )
                or not np.array_equal(
                    np.asarray(reference_binner._border_offsets_),
                    np.asarray(
                        member.model_.prep_.binner_._border_offsets_
                    ),
                )
                or not np.array_equal(
                    np.asarray(reference_binner.n_bins_),
                    np.asarray(member.model_.prep_.binner_.n_bins_),
                )
                for member in members[1:]
            ):
                raise ValueError(
                    "invalid DarkoFit model: shared ensemble preprocessing "
                    "payloads disagree"
                )
        if classification:
            reference = np.asarray(members[0].classes_)
            if any(
                member._multiclass != members[0]._multiclass
                or not np.array_equal(
                    reference, np.asarray(member.classes_)
                )
                for member in members[1:]
            ):
                raise ValueError(
                    "invalid DarkoFit model: ensemble members disagree on "
                    "class labels"
                )
        if index_provenance is not None:
            for member, provenance in zip(members, index_provenance):
                member._ensemble_sampled_indices_ = provenance[
                    "sampled"
                ].copy()
                member._ensemble_oob_indices_ = provenance["oob"].copy()
        result = est._adopt_ensemble(members, metadata)
        if group_codes is not None:
            result._ensemble_group_codes_ = group_codes.copy()
        return result

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
        callbacks = _normalize_callbacks(callbacks)
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

        (
            X_for_validation,
            nominal_cat_features,
            _ordinal_mode,
            ordinal_records,
        ) = self._prepare_ordinal_fit_input(
            X, cat_features, ordinal_features
        )
        X_checked, resolved_cat_features, n_features = _coerce_fit_X(
            X_for_validation, nominal_cat_features
        )
        if ordinal_records:
            frozen_ordinal_records = {
                int(record["index"]): tuple(record["categories"])
                for record in ordinal_records
            }
            member_ordinal_features = (
                _FrozenAutoOrdinalFeatures(frozen_ordinal_records)
                if isinstance(
                    ordinal_features, _FrozenAutoOrdinalFeatures
                )
                else frozen_ordinal_records
            )
        else:
            member_ordinal_features = ordinal_features
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
        if bootstrap == "rows" and groups is not None:
            raise ValueError(
                "groups cannot be used with ensemble_bootstrap='rows'; set "
                "ensemble_bootstrap='groups' to keep entities intact across "
                "bootstrap and out-of-bag rows"
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
            and not ordinal_records
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
            if hasattr(self, "_group_centered_crosses_private_mode"):
                member._group_centered_crosses_private_mode = "off"
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
                    groups=(
                        None
                        if group_values is None
                        else group_values[sampled]
                    ),
                    sample_weight=member_sample_weight,
                    eval_sample_weight=member_eval_weight,
                    ordinal_features=member_ordinal_features,
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
                    if resolved_cat_features or ordinal_records
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
        mode = getattr(self, "ordinal_features_mode_", "off")
        if mode == "off":
            return
        metadata = _ordinal_metadata_payload(
            mode,
            records,
            getattr(self, "_ordinal_nominal_cat_count_", 0),
        )
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
        has_state_version = "ordinal_features_version" in state
        state_version = state.get("ordinal_features_version")
        auto_params = getattr(
            getattr(self, "model_", None), "auto_params_", {}
        )
        fitted_metadata = auto_params.get("ordinal_features")
        fitted_version = (
            fitted_metadata.get("state_version")
            if isinstance(fitted_metadata, Mapping)
            else None
        )
        expected_metadata = _ordinal_metadata_payload(
            mode,
            records,
            self._ordinal_nominal_cat_count_,
        )
        diagnostics = auto_params.get("diagnostics")
        diagnostics_metadata = (
            diagnostics.get("ordinal_features")
            if isinstance(diagnostics, Mapping)
            else None
        )
        if not has_state_version:
            if fitted_version is None:
                if mode == "off":
                    if (
                        fitted_metadata is None
                        and diagnostics_metadata is None
                    ):
                        return
                else:
                    legacy_expected = dict(expected_metadata)
                    legacy_expected.pop("state_version", None)
                    legacy_expected.pop("category_order_sha256", None)
                    if (
                        _ordinal_metadata_matches(
                            fitted_metadata, legacy_expected
                        )
                        and _ordinal_metadata_matches(
                            diagnostics_metadata, legacy_expected
                        )
                    ):
                        return
                raise ValueError(
                    "invalid DarkoFit model: ordinal wrapper state does not "
                    "match fitted provenance"
                )
            raise ValueError(
                "invalid DarkoFit model: ordinal wrapper state is missing "
                "fitted provenance"
            )
        elif (
            isinstance(state_version, bool)
            or not isinstance(state_version, int)
            or state_version != _ORDINAL_STATE_VERSION
            or not records
        ):
            raise ValueError(
                "invalid DarkoFit model: ordinal feature state version is "
                "invalid"
            )
        if (
            not _ordinal_metadata_matches(
                fitted_metadata, expected_metadata
            )
            or not _ordinal_metadata_matches(
                diagnostics_metadata, expected_metadata
            )
        ):
            raise ValueError(
                "invalid DarkoFit model: ordinal wrapper state does not match "
                "fitted provenance"
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
            return {
                **_RefitParamsMixin._more_tags(self),
                "requires_y": True,
            }
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
            "interval_calibration_", "interval_calibration_source_",
            "interval_calibration_split_", "conformal_scores_",
            "conformal_score_count_",
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
        metadata = _preset_metadata_payload(preset, self.preset_params_)
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
            ordinal_metadata = getattr(
                getattr(self, "model_", None), "auto_params_", {}
            ).get("ordinal_features")
            if (
                self.ordinal_features_
                and isinstance(ordinal_metadata, Mapping)
                and ordinal_metadata.get("state_version")
                == _ORDINAL_STATE_VERSION
            ):
                state["ordinal_features_version"] = _ORDINAL_STATE_VERSION
            state["ordinal_features_mode"] = ordinal_mode
            state["ordinal_features"] = list(self.ordinal_features_)
        if getattr(self, "refit_", False):
            state["selection_model_persisted"] = False
        if hasattr(self, "tree_mode_selection_"):
            state["tree_mode_selection"] = self.tree_mode_selection_
        if hasattr(self, "preset_"):
            state["preset"] = self.preset_
            state["preset_params"] = dict(self.preset_params_)
        if (
            hasattr(self, "group_centered_categorical_crosses_")
            and getattr(
                self, "_group_centered_cross_metadata_persisted_", False
            )
        ):
            state["group_centered_categorical_crosses"] = dict(
                self.group_centered_categorical_crosses_
            )
        if hasattr(self, "_selection_n_total_"):
            state["selection_n_total"] = self._selection_n_total_
        if hasattr(self, "_selection_n_train_"):
            state["selection_n_train"] = self._selection_n_train_
        if hasattr(self, "interval_calibration_"):
            state["interval_calibration"] = self.interval_calibration_
            state["interval_calibration_source"] = (
                self.interval_calibration_source_
            )
            state["interval_calibration_split"] = dict(
                self.interval_calibration_split_
            )
            state["conformal_score_count"] = int(
                self.conformal_score_count_
            )
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
        model = getattr(self, "model_", None)
        fitted_loss = getattr(model, "loss_name", None)
        if "loss" in params and fitted_loss is not None:
            params["loss"] = fitted_loss
        if "alpha" in params and fitted_loss == "Quantile":
            params["alpha"] = float(model.loss_kwargs["alpha"])
        if "oblivious_kernel" in params and model is not None:
            params["oblivious_kernel"] = _normalize_oblivious_kernel(
                model.oblivious_kernel
            )
        if "preset" in params:
            params["preset"] = getattr(self, "preset_", None)
        if "interval_calibration" in params:
            params["interval_calibration"] = getattr(
                self, "interval_calibration_", None
            )
        fitted_dist_calibration = getattr(
            self,
            "dist_calibration_",
            getattr(self, "sigma_calibration_", None),
        )
        if "dist_calibration" in params:
            params["dist_calibration"] = fitted_dist_calibration
        if "sigma_calibration" in params:
            params["sigma_calibration"] = None
        if (
            fitted_dist_calibration == "per_metric_affine"
            and "dist_calibration_feature" in params
        ):
            params["dist_calibration_feature"] = getattr(
                self,
                "dist_calibration_feature_",
                _GROUP_AFFINE_DEFAULT_FEATURE,
            )
        if hasattr(self, "linear_residual_enabled_"):
            params["linear_residual"] = bool(
                self.linear_residual_enabled_
            )
            params["linear_residual_alpha"] = float(
                self.linear_residual_alpha_
            )
            params["linear_residual_fit_intercept"] = bool(
                self.linear_residual_fit_intercept_
            )
            params["linear_residual_standardize"] = bool(
                self.linear_residual_standardize_
            )
            trend = getattr(self, "linear_residual_trend_", None)
            if trend is not None:
                selector = getattr(trend, "features", "auto")
                if (
                    selector is None
                    or isinstance(selector, (str, list, tuple, np.ndarray))
                ):
                    params["linear_residual_features"] = selector
                else:
                    params["linear_residual_features"] = [
                        int(index)
                        for index in self.linear_residual_feature_indices_
                    ]
        if "random_state" in params:
            fit_seed = (
                model._fit_random_state_seed_
                if hasattr(model, "_fit_random_state_seed_")
                else getattr(model, "random_state", None)
            )
            params["random_state"] = (
                None if fit_seed is None else int(fit_seed)
            )
        return params

    def _wrapper_arrays(self):
        arrays = {}
        if hasattr(self, "conformal_scores_"):
            arrays["conformal_scores"] = np.asarray(
                self.conformal_scores_, dtype=np.float64
            )
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

    def _restore_wrapper_state(self, state, wrapper_params=None):
        state = state or {}
        _validate_loaded_refit_state(state, self.model_)
        _validate_loaded_wrapper_fitted_state(state, self.model_)
        _validate_loaded_dist_calibration_state(
            state, self.model_, wrapper_params
        )
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
            saved_selection = state["tree_mode_selection"]
            auto_params = getattr(self.model_, "auto_params_", {})
            fitted_selection = auto_params.get("tree_mode_selection")
            diagnostics = auto_params.get("diagnostics", {})
            diagnostic_selection = (
                diagnostics.get("tree_mode_selection")
                if isinstance(diagnostics, Mapping)
                else None
            )
            if (
                not isinstance(saved_selection, Mapping)
                or not isinstance(fitted_selection, Mapping)
                or dict(saved_selection) != dict(fitted_selection)
                or not isinstance(diagnostic_selection, Mapping)
                or dict(saved_selection) != dict(diagnostic_selection)
            ):
                raise ValueError(
                    "invalid DarkoFit model: tree mode selection wrapper and "
                    "booster provenance disagree"
                )
            self.tree_mode_selection_ = dict(saved_selection)
        elif getattr(self.model_, "auto_params_", {}).get(
            "tree_mode_selection"
        ) is not None:
            raise ValueError(
                "invalid DarkoFit model: booster tree mode selection "
                "provenance has no wrapper fitted state"
            )
        cross_state = state.get("group_centered_categorical_crosses")
        fitted_cross_state = getattr(self.model_, "auto_params_", {}).get(
            "group_centered_categorical_crosses"
        )
        saved_crosses_enabled = (
            _normalize_categorical_crosses(
                wrapper_params.get("categorical_crosses", False)
            )
            if isinstance(wrapper_params, Mapping)
            else False
        )
        if cross_state is not None:
            if not saved_crosses_enabled:
                raise ValueError(
                    "invalid DarkoFit model: group-centered cross fitted "
                    "state requires categorical_crosses=True"
                )
            if (
                not isinstance(cross_state, Mapping)
                or not isinstance(fitted_cross_state, Mapping)
                or dict(cross_state) != dict(fitted_cross_state)
            ):
                raise ValueError(
                    "invalid DarkoFit model: group-centered cross wrapper and "
                    "booster provenance disagree"
                )
            selected = cross_state.get("selected")
            final_pairs = cross_state.get("final_pairs")
            final_preprocessing = cross_state.get("final_preprocessing")
            fitted_pairs = [
                list(pair)
                for pair in getattr(
                    self.model_.prep_, "group_centered_pairs_", ()
                )
            ]
            if (
                not isinstance(selected, bool)
                or not isinstance(final_pairs, list)
                or final_pairs != fitted_pairs
                or bool(fitted_pairs) != selected
                or not isinstance(final_preprocessing, Mapping)
                or dict(final_preprocessing)
                != _group_centered_preprocessing_record(self.model_.prep_)
            ):
                raise ValueError(
                    "invalid DarkoFit model: group-centered cross fitted "
                    "state disagrees with preprocessing payload"
                )
            self.group_centered_categorical_crosses_ = dict(cross_state)
            self._group_centered_cross_metadata_persisted_ = True
        elif saved_crosses_enabled:
            raise ValueError(
                "invalid DarkoFit model: categorical_crosses=True has no "
                "group-centered cross fitted state"
            )
        elif fitted_cross_state is not None or getattr(
            self.model_.prep_, "group_centered_pairs_", ()
        ):
            raise ValueError(
                "invalid DarkoFit model: group-centered cross booster "
                "provenance has no wrapper fitted state"
            )
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
            auto_params = getattr(self.model_, "auto_params_", {})
            fitted_preset = auto_params.get("preset")
            diagnostics = auto_params.get("diagnostics", {})
            diagnostic_preset = (
                diagnostics.get("preset")
                if isinstance(diagnostics, Mapping)
                else None
            )
            expected_metadata = _preset_metadata_payload(
                preset, preset_params
            )
            if (
                _normalize_regression_preset(
                    getattr(self, "preset", None)
                )
                != preset
                or not isinstance(fitted_preset, Mapping)
                or dict(fitted_preset) != expected_metadata
                or not isinstance(diagnostic_preset, Mapping)
                or dict(diagnostic_preset) != expected_metadata
            ):
                raise ValueError(
                    "invalid DarkoFit model: preset wrapper and booster "
                    "provenance disagree"
                )
            self.preset_ = preset
            self.preset_params_ = dict(preset_params)
        elif getattr(self.model_, "auto_params_", {}).get("preset") is not None:
            raise ValueError(
                "invalid DarkoFit model: booster preset provenance has no "
                "wrapper fitted state"
            )
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
        elif isinstance(self.model_, DistributionalBoosting):
            self.dist_calibration_ = None
            self.sigma_calibration_ = None
            self.dist_scale_ = 1.0
            self.sigma_scale_ = 1.0
            self.dist_scale_source_ = "none"
            self.sigma_scale_source_ = "none"

    def _restore_interval_calibration_state(self, state, wrapper_arrays):
        state = state or {}
        method = state.get("interval_calibration")
        scores = (wrapper_arrays or {}).get("conformal_scores")
        fitted_metadata = getattr(self.model_, "auto_params_", {}).get(
            "interval_calibration"
        )
        if method is None:
            if _normalize_interval_calibration(
                getattr(self, "interval_calibration", None)
            ) is not None:
                raise ValueError(
                    "invalid DarkoFit model: interval calibration parameter "
                    "has no fitted calibration state"
                )
            if scores is not None:
                raise ValueError(
                    "invalid DarkoFit model: conformal scores have no "
                    "interval calibration state"
                )
            if fitted_metadata is not None:
                raise ValueError(
                    "invalid DarkoFit model: booster conformal provenance has "
                    "no wrapper calibration state"
                )
            return
        method = _normalize_interval_calibration(method)
        if method != "conformal":
            raise ValueError(
                "invalid DarkoFit model: unsupported interval calibration"
            )
        if (
            _normalize_interval_calibration(
                getattr(self, "interval_calibration", None)
            )
            != method
        ):
            raise ValueError(
                "invalid DarkoFit model: conformal interval calibration "
                "parameter does not match its fitted state"
            )
        if self._fitted_loss_name() != "Gaussian":
            raise ValueError(
                "invalid DarkoFit model: conformal interval calibration "
                "requires a Gaussian model"
            )
        values = (
            np.asarray(scores, dtype=np.float64)
            if scores is not None
            else None
        )
        expected_count = state.get("conformal_score_count")
        if (
            values is None
            or values.ndim != 1
            or values.size == 0
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count != values.size
        ):
            raise ValueError(
                "invalid DarkoFit model: conformal calibration scores are "
                "missing or invalid"
            )
        source = state.get("interval_calibration_source")
        split = state.get("interval_calibration_split")
        if source != "held_out_validation" or not isinstance(split, Mapping):
            raise ValueError(
                "invalid DarkoFit model: conformal calibration provenance "
                "is missing or invalid"
            )
        selection_count = split.get("selection_n_samples")
        calibration_count = split.get("calibration_n_samples")
        holdout_fraction = split.get("holdout_fraction")
        try:
            holdout_fraction_value = float(holdout_fraction)
        except (TypeError, ValueError, OverflowError):
            holdout_fraction_value = float("nan")
        if (
            isinstance(selection_count, bool)
            or not isinstance(selection_count, int)
            or selection_count < 0
            or isinstance(calibration_count, bool)
            or not isinstance(calibration_count, int)
            or calibration_count != values.size
            or isinstance(holdout_fraction, bool)
            or not isinstance(holdout_fraction, (int, float))
            or not np.isfinite(holdout_fraction_value)
            or not math.isclose(
                holdout_fraction_value,
                calibration_count / (selection_count + calibration_count),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or split.get("selection_source")
            not in {"explicit_eval_set", "automatic_validation_split"}
            or split.get("calibration_rows_used_for_fit") is not False
            or split.get("calibration_rows_used_for_selection") is not False
            or split.get(
                "calibration_rows_used_for_dist_calibration"
            ) is not False
        ):
            raise ValueError(
                "invalid DarkoFit model: conformal holdout metadata is "
                "inconsistent"
            )
        if (
            not isinstance(fitted_metadata, Mapping)
            or fitted_metadata.get("method") != method
            or fitted_metadata.get("source") != source
            or isinstance(fitted_metadata.get("score_count"), bool)
            or not isinstance(fitted_metadata.get("score_count"), int)
            or fitted_metadata["score_count"] != values.size
            or fitted_metadata.get("weighted") is not False
            or not isinstance(fitted_metadata.get("split"), Mapping)
            or dict(fitted_metadata["split"]) != dict(split)
        ):
            raise ValueError(
                "invalid DarkoFit model: conformal wrapper and booster "
                "provenance disagree"
            )
        self.interval_calibration_ = method
        self.interval_calibration_source_ = source
        self.interval_calibration_split_ = dict(split)
        self.conformal_scores_ = np.sort(values.copy())
        self.conformal_score_count_ = int(values.size)
        self.model_.auto_params_["interval_calibration"] = {
            "method": method,
            "source": source,
            "score_count": int(values.size),
            "weighted": False,
            "split": dict(split),
        }

    def _restore_linear_residual_state(self, state, wrapper_arrays):
        state = dict(state or {})
        if (
            "linear_residual_enabled" not in state
            and "linear_residual_active" not in state
        ):
            fitted_metadata = getattr(
                self.model_, "auto_params_", {}
            ).get("linear_residual")
            has_linear_arrays = any(
                str(name).startswith("linear_residual_")
                for name in (wrapper_arrays or {})
            )
            if has_linear_arrays:
                raise ValueError(
                    "invalid DarkoFit model: linear residual arrays have no "
                    "wrapper fitted state"
                )
            if fitted_metadata is not None:
                if not isinstance(fitted_metadata, Mapping):
                    raise ValueError(
                        "invalid DarkoFit model: linear residual booster "
                        "provenance is invalid"
                    )
                enabled = fitted_metadata.get("enabled")
                active = fitted_metadata.get("active")
                if (
                    not isinstance(enabled, bool)
                    or not isinstance(active, bool)
                    or enabled
                    or active
                    or not isinstance(self.linear_residual, bool)
                    or self.linear_residual is not enabled
                ):
                    raise ValueError(
                        "invalid DarkoFit model: booster linear residual "
                        "provenance has no matching wrapper fitted state"
                    )
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
        trend.features = getattr(
            self, "linear_residual_features", "auto"
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
        restored_metadata = self._linear_residual_metadata()
        if hasattr(self, "selection_linear_residual_summary_"):
            restored_metadata = dict(restored_metadata)
            restored_metadata["selection_summary"] = (
                self.selection_linear_residual_summary_
            )
        fitted_metadata = getattr(self.model_, "auto_params_", {}).get(
            "linear_residual"
        )
        if (
            not isinstance(fitted_metadata, Mapping)
            or dict(fitted_metadata) != restored_metadata
        ):
            raise ValueError(
                "invalid DarkoFit model: linear residual wrapper and booster "
                "provenance disagree"
            )
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


def _private_ensemble_v3_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _private_ensemble_v3_wrapper_constructor_params(estimator):
    params = _jsonify(estimator._wrapper_params_header())
    if not isinstance(params, dict):
        raise ValueError(
            "private ensemble-v3 wrapper constructor parameters are invalid"
        )
    _private_ensemble_v3_json_token(params)
    return params


def _private_ensemble_v3_base_constructor_params(
    estimator,
    member_params,
    *,
    fit_seed,
    n_members,
    policy_resolutions,
):
    raw_base = _jsonify(estimator.get_params(deep=False))
    if set(raw_base) != set(member_params):
        raise RuntimeError(
            "private ensemble-v3 wrapper parameter schema changed during fit"
        )
    base = dict(member_params)
    for name in _PRIVATE_ENSEMBLE_V3_MEMBER_OVERRIDES:
        base[name] = raw_base[name]
    base["n_ensembles"] = int(n_members)
    base["random_state"] = None if fit_seed is None else int(fit_seed)
    base["refit"] = False
    for name in _PRIVATE_ENSEMBLE_V3_POLICY_FIELDS:
        base[name] = policy_resolutions[name]["base"]
    _private_ensemble_v3_json_token(base)
    return base


def _private_ensemble_v3_expected_member_params(
    base_params, *, seed, policy_resolutions, thread_count=None
):
    expected = dict(base_params)
    expected.update({
        "n_ensembles": 1,
        "random_state": int(seed),
        "early_stopping": True,
        "use_best_model": True,
        "refit": False,
        "ensemble_mode": "bootstrap",
        "ensemble_member_learning_rate": _ENSEMBLE_V3_POLICY_SENTINEL,
        "ensemble_member_colsample": _ENSEMBLE_V3_POLICY_SENTINEL,
    })
    for name in _PRIVATE_ENSEMBLE_V3_POLICY_FIELDS:
        expected[name] = policy_resolutions[name]["resolved"]
    if thread_count is not None:
        expected["thread_count"] = int(thread_count)
    return expected


def _resolve_b3_parallel_topology(n_members, total_thread_budget):
    """Resolve the contract-frozen private B3 worker topology."""
    if isinstance(n_members, (bool, np.bool_)) or not isinstance(
        n_members, (int, np.integer)
    ):
        raise TypeError("n_members must be a positive integer")
    if isinstance(total_thread_budget, (bool, np.bool_)) or not isinstance(
        total_thread_budget, (int, np.integer)
    ):
        raise TypeError("total_thread_budget must be a positive integer")
    n_members = int(n_members)
    total_thread_budget = int(total_thread_budget)
    if n_members < 1:
        raise ValueError("n_members must be positive")
    if total_thread_budget < 1:
        raise ValueError("total_thread_budget must be positive")
    workers = min(n_members, max(1, total_thread_budget // 2))
    member_threads = max(1, total_thread_budget // workers)
    if workers * member_threads > total_thread_budget:
        raise RuntimeError("B3 topology exceeds the total thread budget")
    return workers, member_threads


def _fit_private_ensemble_v3_member(payload):
    """Fit one preplanned v3 member; suitable for a process worker."""
    estimator = payload["estimator"]
    member_index = int(payload["member_index"])
    member_seed = int(payload["member_seed"])
    plan = payload["plan"]
    sampled = plan["sampled"]
    oob = plan["oob"]
    member = clone(estimator)
    overrides = {
        "n_ensembles": 1,
        "random_state": member_seed,
        "early_stopping": True,
        "use_best_model": True,
        "refit": False,
        "ensemble_mode": "bootstrap",
        "ensemble_member_learning_rate": _ENSEMBLE_V3_POLICY_SENTINEL,
        "ensemble_member_colsample": _ENSEMBLE_V3_POLICY_SENTINEL,
        **payload["policy_member_params"],
    }
    if payload["member_thread_count"] is not None:
        overrides["thread_count"] = int(payload["member_thread_count"])
    member.set_params(**overrides)
    if hasattr(estimator, "_group_centered_crosses_private_mode"):
        member._group_centered_crosses_private_mode = "off"
    member._suppress_wrapper_deprecation_warning = True
    if payload["shared_eligible"]:
        member._shared_numeric_preprocessing_ = {
            "prep": payload["shared_prep"],
            "X_binned": np.asarray(payload["full_binned"][sampled]),
        }
    sample_weight = payload["sample_weight_checked"]
    member_sample_weight = (
        None if sample_weight is None else sample_weight[sampled]
    )
    member_eval_weight = None if sample_weight is None else sample_weight[oob]
    group_values = payload["group_values"]
    try:
        member.fit(
            _take_rows(payload["X"], sampled),
            _take_rows(payload["y"], sampled),
            cat_features=payload["cat_features"],
            eval_set=(
                _take_rows(payload["X"], oob),
                _take_rows(payload["y"], oob),
            ),
            groups=(
                None if group_values is None else group_values[sampled]
            ),
            sample_weight=member_sample_weight,
            eval_sample_weight=member_eval_weight,
            ordinal_features=payload["member_ordinal_features"],
        )
    finally:
        if hasattr(member, "_shared_numeric_preprocessing_"):
            del member._shared_numeric_preprocessing_
        if hasattr(member, "_suppress_wrapper_deprecation_warning"):
            del member._suppress_wrapper_deprecation_warning
    validation = dict(member.model_.auto_params_.get("validation_split", {}))
    if (
        validation.get("source") != "explicit_eval_set"
        or int(validation.get("train_n_samples", -1)) != len(sampled)
        or int(validation.get("eval_n_samples", -1)) != len(oob)
    ):
        raise RuntimeError(
            "private ensemble-v3 member did not bind training/OOB rows"
        )
    return {
        "member_index": member_index,
        "member_seed": member_seed,
        "member": member,
        "plan": plan,
        "validation": validation,
    }


def _private_ensemble_v3_expected_booster_params(member, wrapper_params):
    core_params = {
        name: value
        for name, value in wrapper_params.items()
        if name not in _SKLEARN_ONLY and name not in {"loss", "alpha"}
    }
    if (
        wrapper_params.get("early_stopping") is True
        and core_params.get("early_stopping_rounds") is None
    ):
        core_params["early_stopping_rounds"] = "auto"
    if isinstance(member, ClassifierMixin):
        if getattr(member, "_multiclass", None) is True:
            expected = MulticlassBoosting(**core_params)
        else:
            expected = GradientBoosting(loss="Logloss", **core_params)
    else:
        loss = wrapper_params.get("loss")
        loss_kwargs = (
            {"alpha": wrapper_params.get("alpha")}
            if loss == "Quantile"
            else {}
        )
        expected = GradientBoosting(
            loss=loss,
            loss_kwargs=loss_kwargs,
            **core_params,
        )
    return _booster_constructor_params(expected, include_linear=True)


def _fit_private_ensemble_v3(
    estimator,
    X,
    y,
    *,
    sampling,
    sampling_unit,
    sample_fraction=None,
    member_policy="none",
    explicit_user_params=(),
    explicit_user_values=None,
    release_candidate_params=None,
    public_fit_surface=False,
    cat_features=None,
    eval_set=None,
    groups=None,
    sample_weight=None,
    eval_sample_weight=None,
    callbacks=None,
    ordinal_features=None,
    b3_parallel=False,
    b3_total_thread_budget=None,
):
    """Fit the contract-frozen private sequential B1/B2 prototype.

    This deliberately bypasses the public ``fit`` routing without adding a
    constructor surface. It is prediction/serialization capable, but only the
    benchmark and invariant tests should call it. ``b3_parallel`` is the
    contract-frozen private scheduling experiment; public callers never set it.
    """
    previous_state = dict(estimator.__dict__)
    try:
        estimator._clear_ensemble_state()
        n_members = _normalize_n_ensembles(estimator.n_ensembles)
        if n_members < 2:
            raise ValueError(
                "the private ensemble-v3 prototype requires n_ensembles > 1"
            )
        if not isinstance(b3_parallel, (bool, np.bool_)):
            raise TypeError("b3_parallel must be a bool")
        b3_parallel = bool(b3_parallel)
        if b3_parallel:
            b3_workers, b3_member_threads = _resolve_b3_parallel_topology(
                n_members, b3_total_thread_budget
            )
            b3_schedule = {
                "contract": _B3_PARALLEL_ENSEMBLE_CONTRACT,
                "mode": "private_process_workers",
                "workers": b3_workers,
                "member_threads": b3_member_threads,
                "total_thread_budget": int(b3_total_thread_budget),
                "maximum_model_threads": b3_workers * b3_member_threads,
                "result_order": "member_index",
            }
        else:
            if b3_total_thread_budget is not None:
                raise ValueError(
                    "b3_total_thread_budget requires b3_parallel=True"
                )
            b3_workers = 1
            b3_member_threads = None
            b3_schedule = None
        if (
            hasattr(estimator, "preset")
            and _normalize_regression_preset(estimator.preset) is not None
        ):
            raise ValueError(
                "preset is not supported by the constructor-bound private "
                "ensemble-v3 prototype"
            )
        if _is_auto_tree_mode(estimator.tree_mode):
            raise ValueError(
                "tree_mode='auto' is not supported by the constructor-bound "
                "private ensemble-v3 prototype"
            )
        if estimator.auto_learning_rate_probe:
            raise ValueError(
                "auto_learning_rate_probe=True is not supported by the "
                "constructor-bound private ensemble-v3 prototype"
            )
        sampling = _normalize_private_ensemble_v3_sampling(sampling)
        sampling_unit = _normalize_ensemble_bootstrap(sampling_unit)
        if sampling == "bootstrap":
            if sample_fraction is not None:
                raise ValueError(
                    "sample_fraction must be None for bootstrap controls"
                )
            normalized_fraction = None
        else:
            try:
                normalized_fraction = float(sample_fraction)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError(
                    "sample_fraction must be the float 0.8"
                ) from exc
            if (
                isinstance(sample_fraction, (bool, np.bool_))
                or not np.isfinite(normalized_fraction)
                or normalized_fraction != 0.8
            ):
                raise ValueError(
                    "the funded private sample_fraction is exactly 0.8"
                )
        (
            member_policy,
            explicit_user_params,
            policy_resolutions,
            policy_member_params,
        ) = _resolve_private_ensemble_v3_policy(
            estimator,
            member_policy,
            explicit_user_params,
            explicit_user_values,
        )
        if public_fit_surface and release_candidate_params is None:
            raise ValueError(
                "public ensemble-v3 fits require constructor-bound recipe "
                "parameters"
            )
        if release_candidate_params is not None:
            if (
                not isinstance(release_candidate_params, Mapping)
                or set(release_candidate_params)
                != {
                    "ensemble_mode",
                    "ensemble_member_learning_rate",
                    "ensemble_member_colsample",
                }
                or release_candidate_params.get("ensemble_mode") != "v3"
            ):
                raise ValueError(
                    "release_candidate_params do not match the frozen public "
                    "contract"
                )
            candidate_values, candidate_explicit = (
                _normalize_ensemble_v3_release_candidate_overrides(
                    release_candidate_params[
                        "ensemble_member_learning_rate"
                    ],
                    release_candidate_params["ensemble_member_colsample"],
                )
            )
            candidate_explicit = {
                name: _private_ensemble_v3_scalar(value)
                for name, value in candidate_explicit.items()
            }
            resolved_explicit = {
                name: _private_ensemble_v3_scalar(
                    policy_resolutions[name]["resolved"]
                )
                for name in explicit_user_params
            }
            if (
                n_members != _ENSEMBLE_V3_RELEASE_CANDIDATE_MEMBERS
                or sampling != "without_replacement"
                or normalized_fraction != 0.8
                or sampling_unit
                != _normalize_ensemble_bootstrap(estimator.ensemble_bootstrap)
                or member_policy != "donor_balanced_v1"
                or set(candidate_explicit) != set(explicit_user_params)
                or not _private_ensemble_v3_values_equal(
                    candidate_explicit, resolved_explicit
                )
                or not _private_ensemble_v3_values_equal(
                    candidate_values,
                    {
                        "learning_rate": release_candidate_params[
                            "ensemble_member_learning_rate"
                        ],
                        "colsample": release_candidate_params[
                            "ensemble_member_colsample"
                        ],
                    },
                )
            ):
                raise ValueError(
                    "release candidate controls contradict the frozen public "
                    "contract"
                )

        callbacks = _normalize_callbacks(callbacks)
        if eval_set is not None or eval_sample_weight is not None:
            raise ValueError(
                "private ensemble-v3 fits use each member's out-of-bag rows "
                "for validation; eval_set and eval_sample_weight are not "
                "supported"
            )
        if callbacks:
            raise ValueError(
                "callbacks are not supported by the private sequential "
                "ensemble-v3 prototype"
            )
        if estimator.refit is not False:
            raise ValueError(
                "refit=True is not supported by the private ensemble-v3 "
                "prototype"
            )
        if isinstance(ordinal_features, str) and (
            ordinal_features.strip().lower() == "auto"
        ):
            raise ValueError(
                "ordinal_features='auto' is not supported with ensembles; "
                "declare complete ordinal category orders explicitly"
            )
        if not isinstance(
            estimator.ensemble_shared_preprocessing, (bool, np.bool_)
        ):
            raise TypeError("ensemble_shared_preprocessing must be a bool")
        classification = isinstance(estimator, ClassifierMixin)
        if (
            not classification
            and _is_distributional_loss(estimator.loss)
        ):
            raise ValueError(
                "the private ensemble-v3 prototype supports scalar regression "
                "losses only"
            )

        (
            X_for_validation,
            nominal_cat_features,
            _ordinal_mode,
            ordinal_records,
        ) = estimator._prepare_ordinal_fit_input(
            X, cat_features, ordinal_features
        )
        X_checked, resolved_cat_features, n_features = _coerce_fit_X(
            X_for_validation, nominal_cat_features
        )
        if ordinal_records:
            frozen_ordinal_records = {
                int(record["index"]): tuple(record["categories"])
                for record in ordinal_records
            }
            member_ordinal_features = (
                _FrozenAutoOrdinalFeatures(frozen_ordinal_records)
                if isinstance(
                    ordinal_features, _FrozenAutoOrdinalFeatures
                )
                else frozen_ordinal_records
            )
        else:
            member_ordinal_features = ordinal_features
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
        if sampling_unit == "rows" and groups is not None:
            raise ValueError(
                "groups cannot be used with private row sampling; choose "
                "sampling_unit='groups' to keep entities intact"
            )
        group_values = None
        group_codes = None
        input_group_count = None
        if sampling_unit == "groups":
            if groups is None:
                raise ValueError(
                    "private group sampling requires groups in fit"
                )
            group_values = np.asarray(groups)
            group_codes, input_group_count = _normalize_ensemble_group_codes(
                group_values, len(X_checked), context="private"
            )

        fit_seed = normalize_random_state_seed(estimator.random_state)
        member_seeds = tuple(
            int(value)
            for value in np.random.default_rng(fit_seed).integers(
                0, 2**31 - 1, size=n_members
            )
        )
        shared_requested = bool(estimator.ensemble_shared_preprocessing)
        shared_eligible = (
            shared_requested
            and not resolved_cat_features
            and not ordinal_records
            and np.asarray(X_checked).dtype.kind in "biuf"
        )
        shared_prep = None
        full_binned = None
        if shared_eligible:
            shared_prep, full_binned = (
                estimator._make_shared_numeric_preprocessing(
                    X_checked,
                    y_checked,
                    sample_weight_checked,
                    fit_seed,
                )
            )

        serializable_policy_resolutions = {
            name: {
                key: _private_ensemble_v3_scalar(value)
                for key, value in record.items()
            }
            for name, record in policy_resolutions.items()
        }
        estimators = []
        member_metadata = []
        base_constructor_params = None
        member_tasks = []
        for member_index, member_seed in enumerate(member_seeds):
            if sampling == "bootstrap":
                plan = _ensemble_bootstrap_plan(
                    len(X_checked),
                    member_seed,
                    bootstrap=sampling_unit,
                    groups=group_values,
                    y=y_checked,
                    required_class_count=required_class_count,
                    sample_weight=sample_weight_checked,
                )
            else:
                plan = _ensemble_without_replacement_plan(
                    len(X_checked),
                    member_seed,
                    sampling_unit=sampling_unit,
                    sample_fraction=normalized_fraction,
                    groups=group_values,
                    y=y_checked,
                    required_class_count=required_class_count,
                    sample_weight=sample_weight_checked,
                )
            member_tasks.append({
                "estimator": estimator,
                "member_index": member_index,
                "member_seed": member_seed,
                "plan": plan,
                "X": X,
                "y": y,
                "cat_features": cat_features,
                "group_values": group_values,
                "sample_weight_checked": sample_weight_checked,
                "member_ordinal_features": member_ordinal_features,
                "policy_member_params": policy_member_params,
                "shared_eligible": shared_eligible,
                "shared_prep": shared_prep,
                "full_binned": full_binned,
                "member_thread_count": b3_member_threads,
            })
        if b3_parallel:
            from joblib import Parallel, delayed, parallel_config

            with parallel_config(
                backend="loky", inner_max_num_threads=b3_member_threads
            ):
                member_outcomes = Parallel(n_jobs=b3_workers)(
                    delayed(_fit_private_ensemble_v3_member)(task)
                    for task in member_tasks
                )
        else:
            member_outcomes = [
                _fit_private_ensemble_v3_member(task)
                for task in member_tasks
            ]
        if (
            len(member_outcomes) != n_members
            or {outcome.get("member_index") for outcome in member_outcomes}
            != set(range(n_members))
        ):
            raise RuntimeError("private ensemble-v3 member results are invalid")
        member_outcomes.sort(key=lambda outcome: outcome["member_index"])

        for outcome in member_outcomes:
            member_index = int(outcome["member_index"])
            member_seed = int(outcome["member_seed"])
            if member_seed != member_seeds[member_index]:
                raise RuntimeError("private ensemble-v3 member seed changed")
            member = outcome["member"]
            plan = outcome["plan"]
            validation = outcome["validation"]
            sampled = plan["sampled"]
            oob = plan["oob"]
            member_constructor_params = (
                _private_ensemble_v3_wrapper_constructor_params(member)
            )
            if base_constructor_params is None:
                base_constructor_params = (
                    _private_ensemble_v3_base_constructor_params(
                        estimator,
                        member_constructor_params,
                        fit_seed=fit_seed,
                        n_members=n_members,
                        policy_resolutions=serializable_policy_resolutions,
                    )
                )
            expected_member_params = (
                _private_ensemble_v3_expected_member_params(
                    base_constructor_params,
                    seed=member_seed,
                    policy_resolutions=serializable_policy_resolutions,
                    thread_count=b3_member_threads,
                )
            )
            if not _private_ensemble_v3_values_equal(
                member_constructor_params, expected_member_params
            ):
                raise RuntimeError(
                    "private ensemble-v3 member constructor changed outside "
                    "the frozen override set"
                )
            booster_constructor_params = _booster_constructor_params(
                member.model_, include_linear=True
            )
            try:
                expected_booster_params = (
                    _private_ensemble_v3_expected_booster_params(
                        member, member_constructor_params
                    )
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    "private ensemble-v3 member constructor could not be "
                    "bound to its booster"
                ) from exc
            if not _private_ensemble_v3_values_equal(
                booster_constructor_params, expected_booster_params
            ):
                raise RuntimeError(
                    "private ensemble-v3 member booster constructor differs "
                    "from its wrapper constructor"
                )
            sampled_unique_rows = int(np.unique(sampled).size)
            member._ensemble_sampled_indices_ = np.ascontiguousarray(
                sampled, dtype="<i8"
            )
            member._ensemble_oob_indices_ = np.ascontiguousarray(
                oob, dtype="<i8"
            )
            member_metadata.append({
                "member": member_index,
                "seed": member_seed,
                "sampling_attempts": int(plan["attempts"]),
                "sampled_rows": int(len(sampled)),
                "sampled_unique_rows": sampled_unique_rows,
                "sampled_indices_sha256": _index_sha256(sampled),
                "oob_rows": int(len(oob)),
                "oob_indices_sha256": _index_sha256(oob),
                "sampled_group_draws": (
                    None
                    if plan["sampled_group_draws"] is None
                    else int(plan["sampled_group_draws"])
                ),
                "sampled_unique_groups": (
                    None
                    if plan["sampled_unique_groups"] is None
                    else int(plan["sampled_unique_groups"])
                ),
                "oob_groups": (
                    None
                    if plan["oob_groups"] is None
                    else int(plan["oob_groups"])
                ),
                "group_disjoint": (
                    None if sampling_unit == "rows" else True
                ),
                "requested_sample_fraction": normalized_fraction,
                "realized_row_fraction": (
                    sampled_unique_rows / float(len(X_checked))
                ),
                "policy_resolutions": {
                    name: dict(record)
                    for name, record in serializable_policy_resolutions.items()
                },
                "member_constructor_params": dict(
                    member_constructor_params
                ),
                "booster_constructor_params": dict(
                    booster_constructor_params
                ),
                "constructor_learning_rate": (
                    _private_ensemble_v3_scalar(member.learning_rate)
                ),
                "constructor_colsample": (
                    _private_ensemble_v3_scalar(member.colsample)
                ),
                "fitted_thread_count": int(member.model_.n_threads_),
                **(
                    {
                        "prediction_thread_count": int(
                            member.model_.n_threads_
                        )
                    }
                    if b3_parallel
                    else {}
                ),
                "best_iteration": int(member.best_n_estimators_),
                "resolved_learning_rate": float(member.learning_rate_),
                "stop_reason": str(
                    getattr(member.model_, "stop_reason_", "unknown")
                ),
                "validation_source": validation["source"],
            })
            estimators.append(member)

        release_candidate = (
            release_candidate_params is not None and not public_fit_surface
        )
        public_v3 = bool(public_fit_surface)
        metadata = {
            "version": (
                _ENSEMBLE_V3_PUBLIC_METADATA_VERSION
                if public_v3
                else _ENSEMBLE_V3_RELEASE_CANDIDATE_METADATA_VERSION
                if release_candidate
                else _PRIVATE_ENSEMBLE_V3_METADATA_VERSION
            ),
            "claim_tier": "E",
            "default_changed": False,
            "public_fit_surface": public_v3,
            "sequential": not b3_parallel,
            "member_count": n_members,
            "member_seeds": list(member_seeds),
            "fit_random_state_seed": fit_seed,
            "sampling": sampling,
            "sampling_unit": sampling_unit,
            "sample_fraction": normalized_fraction,
            "bootstrap": sampling_unit,
            "member_policy": member_policy,
            "explicit_user_params": list(explicit_user_params),
            "policy_resolutions": serializable_policy_resolutions,
            "base_constructor_params": dict(base_constructor_params),
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
                    if resolved_cat_features or ordinal_records
                    else "non_numeric_dtype"
                )
            ),
            "input_row_count": int(len(X_checked)),
            "input_feature_count": int(n_features),
            "input_group_count": input_group_count,
            "group_codes_sha256": (
                None if group_codes is None else _index_sha256(group_codes)
            ),
            "members": member_metadata,
        }
        if b3_schedule is not None:
            metadata["private_b3_schedule"] = dict(b3_schedule)
        if public_v3:
            metadata.update({
                "ensemble_mode": "v3",
                "recipe_contract": _ENSEMBLE_V3_PUBLIC_CONTRACT,
                "recipe_version": 1,
            })
        else:
            metadata["private_prototype"] = (
                _ENSEMBLE_V3_RELEASE_CANDIDATE
                if release_candidate
                else _PRIVATE_ENSEMBLE_V3_PROTOTYPE
            )
        if release_candidate:
            metadata.update({
                "recipe_contract": _ENSEMBLE_V3_PUBLIC_CONTRACT,
                "recipe_version": 1,
                "future_constructor_params": {
                    name: _private_ensemble_v3_scalar(value)
                    for name, value in release_candidate_params.items()
                },
            })
        result = estimator._adopt_ensemble(estimators, metadata)
        if group_codes is not None:
            result._ensemble_group_codes_ = group_codes.copy()
        return result
    except BaseException:
        estimator.__dict__.clear()
        estimator.__dict__.update(previous_state)
        raise


def _fit_ensemble_v3_release_candidate(
    estimator,
    X,
    y,
    *,
    member_learning_rate=_ENSEMBLE_V3_POLICY_SENTINEL,
    member_colsample=_ENSEMBLE_V3_POLICY_SENTINEL,
    cat_features=None,
    eval_set=None,
    groups=None,
    sample_weight=None,
    eval_sample_weight=None,
    callbacks=None,
    ordinal_features=None,
):
    """Fit the non-exported candidate for the frozen future v3 contract."""
    n_members = _normalize_n_ensembles(estimator.n_ensembles)
    if n_members != _ENSEMBLE_V3_RELEASE_CANDIDATE_MEMBERS:
        raise ValueError(
            "the ensemble-v3 release candidate requires n_ensembles=8; "
            "eight is the only evaluated recipe"
        )
    future_values, explicit_values = (
        _normalize_ensemble_v3_release_candidate_overrides(
            member_learning_rate,
            member_colsample,
        )
    )
    release_candidate_params = {
        "ensemble_mode": "v3",
        "ensemble_member_learning_rate": future_values["learning_rate"],
        "ensemble_member_colsample": future_values["colsample"],
    }
    return _fit_private_ensemble_v3(
        estimator,
        X,
        y,
        sampling="without_replacement",
        sampling_unit=estimator.ensemble_bootstrap,
        sample_fraction=0.8,
        member_policy="donor_balanced_v1",
        explicit_user_params=tuple(explicit_values),
        explicit_user_values=explicit_values,
        release_candidate_params=release_candidate_params,
        cat_features=cat_features,
        eval_set=eval_set,
        groups=groups,
        sample_weight=sample_weight,
        eval_sample_weight=eval_sample_weight,
        callbacks=callbacks,
        ordinal_features=ordinal_features,
    )


def _fit_public_ensemble_v3(
    estimator,
    X,
    y,
    *,
    future_values,
    explicit_values,
    cat_features=None,
    eval_set=None,
    groups=None,
    sample_weight=None,
    eval_sample_weight=None,
    callbacks=None,
    ordinal_features=None,
):
    """Fit the public, contract-bound ensemble-v3 recipe."""
    n_members = _normalize_n_ensembles(estimator.n_ensembles)
    if n_members != _ENSEMBLE_V3_RELEASE_CANDIDATE_MEMBERS:
        raise ValueError(
            "ensemble_mode='v3' requires n_ensembles=8; eight is the only "
            "evaluated recipe"
        )
    release_candidate_params = {
        "ensemble_mode": "v3",
        "ensemble_member_learning_rate": future_values["learning_rate"],
        "ensemble_member_colsample": future_values["colsample"],
    }
    return _fit_private_ensemble_v3(
        estimator,
        X,
        y,
        sampling="without_replacement",
        sampling_unit=estimator.ensemble_bootstrap,
        sample_fraction=0.8,
        member_policy="donor_balanced_v1",
        explicit_user_params=tuple(explicit_values),
        explicit_user_values=explicit_values,
        release_candidate_params=release_candidate_params,
        public_fit_surface=True,
        cat_features=cat_features,
        eval_set=eval_set,
        groups=groups,
        sample_weight=sample_weight,
        eval_sample_weight=eval_sample_weight,
        callbacks=callbacks,
        ordinal_features=ordinal_features,
    )


def _fit_public_ensemble_v3_parallel_candidate(
    estimator,
    X,
    y,
    *,
    total_thread_budget,
    cat_features=None,
    eval_set=None,
    groups=None,
    sample_weight=None,
    eval_sample_weight=None,
    callbacks=None,
    ordinal_features=None,
):
    """Fit the private B3 process-scheduled public-v3 candidate."""
    ensemble_mode, future_values, explicit_values = (
        _resolve_public_ensemble_surface(estimator)
    )
    if ensemble_mode != "v3":
        raise ValueError("the private B3 candidate requires ensemble_mode='v3'")
    n_members = _normalize_n_ensembles(estimator.n_ensembles)
    if n_members != _ENSEMBLE_V3_RELEASE_CANDIDATE_MEMBERS:
        raise ValueError(
            "the private B3 candidate requires n_ensembles=8"
        )
    release_candidate_params = {
        "ensemble_mode": "v3",
        "ensemble_member_learning_rate": future_values["learning_rate"],
        "ensemble_member_colsample": future_values["colsample"],
    }
    return _fit_private_ensemble_v3(
        estimator,
        X,
        y,
        sampling="without_replacement",
        sampling_unit=estimator.ensemble_bootstrap,
        sample_fraction=0.8,
        member_policy="donor_balanced_v1",
        explicit_user_params=tuple(explicit_values),
        explicit_user_values=explicit_values,
        release_candidate_params=release_candidate_params,
        public_fit_surface=True,
        cat_features=cat_features,
        eval_set=eval_set,
        groups=groups,
        sample_weight=sample_weight,
        eval_sample_weight=eval_sample_weight,
        callbacks=callbacks,
        ordinal_features=ordinal_features,
        b3_parallel=True,
        b3_total_thread_budget=total_thread_budget,
    )


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
    oblivious_kernel : {"auto", "fused", "unfused"}, default "auto"
        Static fused-kernel dispatch for eligible scalar CatBoost-mode fits.
        On macOS arm64 within the measured shape envelope, ``"auto"`` uses a
        fixed scan-work threshold to select the fused or unfused lane. Explicit
        modes remain an observability and escape-hatch surface.
    preset : {None, "accuracy"}, default None
        Optional profile. ``"accuracy"`` applies the frozen A10 development
        configuration during fit without changing the conservative default.
        Explicit parameters outside the managed A10 fields remain in effect.
    selection_rounds : int or None, default None
        Optional cap for each ``tree_mode="auto"`` audition. The selected
        mode is then fit from scratch with the full requested round budget,
        unless a shared wall-clock deadline expires before the refit starts.
    interval_calibration : {None, "conformal"}, default None
        Opt into split-conformal Gaussian intervals using standardized
        residual scores from the explicit or automatic validation set.
    n_ensembles : int, default 1
        Number of OOB-selected bootstrap members, from 1 through 256. Values
        above one opt into mean aggregation. ``ensemble_bootstrap="groups"``
        requires ``groups`` in :meth:`fit`; numeric-only members may safely
        share target-free preprocessing.
    ensemble_mode : {"bootstrap", "v3"}, default "bootstrap"
        Keep legacy bootstrap sampling or select the fixed public v3 recipe.
        V3 requires eight members and uses deterministic 80%
        without-replacement samples with donor-balanced member settings.
    categorical_crosses : bool, default False
        Run a held-out automatic audition of group-centered numeric-by-category
        features for eligible scalar-RMSE CatBoost fits. Data that is too
        small or lacks either feature type falls back exactly and records why.
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
                 interval_calibration=None,
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
                 ensemble_shared_preprocessing=True,
                 ensemble_mode="bootstrap",
                 ensemble_member_learning_rate="policy",
                 ensemble_member_colsample="policy",
                 categorical_crosses=False,
                 oblivious_kernel="auto"):
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
        self.interval_calibration = interval_calibration
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
        self.ensemble_mode = ensemble_mode
        self.ensemble_member_learning_rate = ensemble_member_learning_rate
        self.ensemble_member_colsample = ensemble_member_colsample
        self.categorical_crosses = categorical_crosses
        self.oblivious_kernel = oblivious_kernel
        self.auto_learning_rate_probe = auto_learning_rate_probe
        self.auto_learning_rate_probe_values = auto_learning_rate_probe_values
        self.auto_learning_rate_probe_iterations = auto_learning_rate_probe_iterations

    def _clear_group_centered_cross_state(self):
        if hasattr(self, "group_centered_categorical_crosses_"):
            del self.group_centered_categorical_crosses_
        if hasattr(self, "_group_centered_cross_metadata_persisted_"):
            del self._group_centered_cross_metadata_persisted_

    def _attach_group_centered_cross_metadata(self, metadata, *, persist=True):
        metadata = dict(metadata)
        self.group_centered_categorical_crosses_ = metadata
        self._group_centered_cross_metadata_persisted_ = bool(persist)
        model = getattr(self, "model_", None)
        if persist and model is not None and not hasattr(self, "estimators_"):
            model.auto_params_["group_centered_categorical_crosses"] = metadata
            model.auto_params_.setdefault("diagnostics", {})[
                "group_centered_categorical_crosses"
            ] = metadata

    def _group_centered_cross_ineligible_reason(
        self, X, cat_features, *, eval_set, callbacks, ordinal_features
    ):
        if _normalize_n_ensembles(self.n_ensembles) != 1 or (
            str(self.ensemble_mode).strip().lower().replace("-", "_")
            != "bootstrap"
        ):
            return "ensemble"
        if _normalize_regression_preset(self.preset) is not None:
            return "preset"
        if self.loss != "RMSE":
            return "non_rmse_loss"
        if _is_auto_tree_mode(self.tree_mode):
            return "automatic_tree_mode"
        if _normalize_tree_mode(self.tree_mode) != "catboost":
            return "non_catboost_tree_mode"
        if self.auto_learning_rate_probe:
            return "automatic_learning_rate_probe"
        if _should_use_linear_residual(self.linear_residual):
            return "linear_residual"
        if self.linear_leaves:
            return "linear_leaves"
        if self.refit:
            return "refit"
        if _normalize_dist_calibration(
            self.dist_calibration, self.sigma_calibration
        ) is not None:
            return "distributional_calibration"
        if _normalize_interval_calibration(self.interval_calibration) is not None:
            return "interval_calibration"
        if callbacks:
            return "callbacks"
        if ordinal_features is not None:
            return "ordinal_features"
        if self.ordered_boosting is True or (
            isinstance(self.ordered_boosting, np.bool_)
            and bool(self.ordered_boosting)
        ):
            return "ordered_boosting"
        X_checked, normalized_cats, n_features = _coerce_fit_X(X, cat_features)
        n_cats = len(normalized_cats)
        if n_cats == 0:
            return "no_categorical_features"
        if n_cats == n_features:
            return "no_numeric_features"
        minimum_rows = (
            _GROUP_CENTERED_CROSSES_MIN_SELECTION_ROWS
            if eval_set is not None
            else math.ceil(
                _GROUP_CENTERED_CROSSES_MIN_SELECTION_ROWS
                / (1.0 - _GROUP_CENTERED_CROSSES_VALIDATION_FRACTION)
            )
        )
        if X_checked.shape[0] < minimum_rows:
            return "below_min_samples"
        return None

    def _fit_group_centered_cross_fallback(
        self,
        reason,
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
        persist_metadata,
    ):
        had_previous_mode = hasattr(
            self, "_group_centered_crosses_private_mode"
        )
        previous_mode = getattr(
            self, "_group_centered_crosses_private_mode", None
        )
        previous_fallback_active = getattr(
            self, "_group_centered_cross_fallback_active", None
        )
        try:
            self._group_centered_crosses_private_mode = "off"
            self._group_centered_cross_fallback_active = True
            fitted = self.fit(
                X,
                y,
                cat_features=cat_features,
                eval_set=eval_set,
                groups=groups,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight,
                callbacks=callbacks,
                ordinal_features=ordinal_features,
            )
        finally:
            if had_previous_mode:
                self._group_centered_crosses_private_mode = previous_mode
            elif hasattr(self, "_group_centered_crosses_private_mode"):
                del self._group_centered_crosses_private_mode
            if previous_fallback_active is None:
                if hasattr(self, "_group_centered_cross_fallback_active"):
                    del self._group_centered_cross_fallback_active
            else:
                self._group_centered_cross_fallback_active = (
                    previous_fallback_active
                )
        self._attach_group_centered_cross_metadata(
            {
                "version": _GROUP_CENTERED_CROSSES_VERSION,
                "eligible": False,
                "reason": str(reason),
                "selected": False,
                "pairs": [],
                "split": {"source": "none"},
                "control_validation_rmse": None,
                "augmented_validation_rmse": None,
                "relative_validation_improvement": None,
                "selection_total_seconds": 0.0,
                "final_pairs": [],
                "final_preprocessing": _group_centered_preprocessing_record(
                    getattr(getattr(self, "model_", None), "prep_", None)
                ),
            },
            persist=persist_metadata,
        )
        return fitted

    @staticmethod
    def _group_centered_cross_fit_record(name, model, seconds):
        score = float(model.best_score_)
        if not np.isfinite(score) or score < 0.0:
            raise RuntimeError(
                "group-centered cross selector produced invalid validation RMSE"
            )
        return {
            "name": name,
            "validation_rmse": score,
            "best_n_estimators": int(model.best_n_estimators_),
            "fit_seconds": float(seconds),
        }

    def _fit_group_centered_cross_selector(
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
        persist_fallback_metadata=True,
    ):
        self._clear_group_centered_cross_state()
        reason = self._group_centered_cross_ineligible_reason(
            X,
            cat_features,
            eval_set=eval_set,
            callbacks=callbacks,
            ordinal_features=ordinal_features,
        )
        if reason is not None:
            return self._fit_group_centered_cross_fallback(
                reason,
                X,
                y,
                cat_features=cat_features,
                eval_set=eval_set,
                groups=groups,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight,
                callbacks=callbacks,
                ordinal_features=ordinal_features,
                persist_metadata=persist_fallback_metadata,
            )

        _ensure_dense(X)
        X_checked, normalized_cats, n_features = _coerce_fit_X(X, cat_features)
        n_rows = X_checked.shape[0]
        y_checked = validate_target_vector(y, n_rows, dtype=np.float64)
        weights = _validate_wrapper_sample_weight(sample_weight, n_rows)
        requested_random_state = self.random_state
        fit_seed = normalize_random_state_seed(requested_random_state)
        if fit_seed is None:
            fit_seed = 0
        if eval_set is None:
            if eval_sample_weight is not None:
                raise ValueError("eval_sample_weight requires an explicit eval_set")
            group_values = None if groups is None else np.asarray(groups)
            if group_values is not None and group_values.shape != (n_rows,):
                raise ValueError(
                    "groups must be one-dimensional with one value per training row"
                )
            train_idx, validation_idx, policy = _make_eval_split(
                X_checked,
                y_checked,
                _GROUP_CENTERED_CROSSES_VALIDATION_FRACTION,
                fit_seed,
                groups=group_values,
                sample_weight=weights,
                validation_strategy="weighted_stratified",
            )
            selection_X = _take_rows(X, train_idx)
            selection_y = y_checked[train_idx]
            selection_eval_set = (
                _take_rows(X, validation_idx),
                y_checked[validation_idx],
            )
            selection_groups = (
                None if group_values is None else group_values[train_idx]
            )
            selection_weight = None if weights is None else weights[train_idx]
            selection_eval_weight = (
                None if weights is None else weights[validation_idx]
            )
            if np.intersect1d(train_idx, validation_idx).size:
                raise RuntimeError("group-centered selection rows overlap")
            group_disjoint = None
            if group_values is not None:
                group_disjoint = not bool(
                    np.intersect1d(
                        np.unique(group_values[train_idx]),
                        np.unique(group_values[validation_idx]),
                    ).size
                )
                if not group_disjoint:
                    raise RuntimeError("group-centered selection groups overlap")
            split = {
                "source": "automatic_holdout",
                "policy": policy,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(validation_idx)),
                "train_positions_sha256": _index_sha256(train_idx),
                "validation_positions_sha256": _index_sha256(validation_idx),
                "rows_disjoint": True,
                "group_disjoint": group_disjoint,
                "sample_weight_provided": weights is not None,
            }
        else:
            normalized_eval_set = _ensure_dense_eval_set(eval_set)
            selection_X = X
            selection_y = y_checked
            selection_eval_set = normalized_eval_set
            selection_groups = groups
            selection_weight = weights
            eval_rows = n_samples_from_array_like(normalized_eval_set[0])
            selection_eval_weight = _validate_wrapper_sample_weight(
                eval_sample_weight,
                eval_rows,
                name="eval_sample_weight",
            )
            split = {
                "source": "explicit_eval_set",
                "policy": "explicit_eval_set",
                "train_rows": int(n_rows),
                "validation_rows": int(eval_rows),
                "train_positions_sha256": _index_sha256(np.arange(n_rows)),
                "validation_positions_sha256": None,
                "rows_disjoint": None,
                "group_disjoint": None,
                "sample_weight_provided": (
                    weights is not None or selection_eval_weight is not None
                ),
            }

        def fit_audition(name, pairs):
            candidate = clone(self).set_params(
                random_state=fit_seed,
                early_stopping=True,
                early_stopping_rounds=None,
                use_best_model=True,
                refit=False,
                diagnostic_warnings="never",
            )
            candidate._group_centered_crosses_private_mode = "forced"
            candidate._group_centered_pairs_override = list(pairs)
            started = time.perf_counter_ns()
            candidate.fit(
                selection_X,
                selection_y,
                cat_features=cat_features,
                eval_set=selection_eval_set,
                groups=selection_groups,
                sample_weight=selection_weight,
                eval_sample_weight=selection_eval_weight,
                ordinal_features=ordinal_features,
            )
            seconds = (time.perf_counter_ns() - started) / 1e9
            return candidate, self._group_centered_cross_fit_record(
                name, candidate, seconds
            )

        selection_started = time.perf_counter_ns()
        control, control_record = fit_audition("control", ())
        pairs = _group_centered_candidate_pairs(
            control.feature_importances_, normalized_cats, n_features
        )
        if not pairs or len(pairs) > 12:
            raise RuntimeError("group-centered candidate pair budget is invalid")
        augmented, augmented_record = fit_audition("augmented", pairs)
        selection_seconds = (time.perf_counter_ns() - selection_started) / 1e9
        control_score = control_record["validation_rmse"]
        augmented_score = augmented_record["validation_rmse"]
        selected = augmented_score < control_score
        margin = (
            0.0
            if control_score == 0.0 and augmented_score == 0.0
            else -1.0
            if control_score == 0.0
            else (control_score - augmented_score) / control_score
        )
        del control, augmented
        metadata = {
            "version": _GROUP_CENTERED_CROSSES_VERSION,
            "eligible": True,
            "reason": "selected_augmented" if selected else "control_won",
            "selected": bool(selected),
            "fit_random_state_seed": int(fit_seed),
            "pairs": [list(pair) for pair in pairs],
            "split": split,
            "control_validation_rmse": float(control_score),
            "augmented_validation_rmse": float(augmented_score),
            "relative_validation_improvement": float(margin),
            "selection_fits": [control_record, augmented_record],
            "selection_total_seconds": float(selection_seconds),
        }
        had_previous_mode = hasattr(
            self, "_group_centered_crosses_private_mode"
        )
        previous_mode = getattr(
            self, "_group_centered_crosses_private_mode", None
        )
        previous_pairs = getattr(self, "_group_centered_pairs_override", None)
        try:
            self._group_centered_crosses_private_mode = "forced"
            self._group_centered_pairs_override = pairs if selected else []
            self.random_state = fit_seed
            fitted = self.fit(
                X,
                y,
                cat_features=cat_features,
                eval_set=eval_set,
                groups=groups,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight,
                callbacks=callbacks,
                ordinal_features=ordinal_features,
            )
        finally:
            if had_previous_mode:
                self._group_centered_crosses_private_mode = previous_mode
            elif hasattr(self, "_group_centered_crosses_private_mode"):
                del self._group_centered_crosses_private_mode
            self.random_state = requested_random_state
            if previous_pairs is None:
                if hasattr(self, "_group_centered_pairs_override"):
                    del self._group_centered_pairs_override
            else:
                self._group_centered_pairs_override = previous_pairs
        metadata["final_pairs"] = [
            list(pair)
            for pair in getattr(self.model_.prep_, "group_centered_pairs_", ())
        ]
        metadata["final_preprocessing"] = _group_centered_preprocessing_record(
            self.model_.prep_
        )
        self._attach_group_centered_cross_metadata(metadata)
        return fitted

    @_restore_fitted_state_on_fit_failure
    def fit(self, X, y, cat_features=None, eval_set=None, groups=None,
            sample_weight=None, eval_sample_weight=None, callbacks=None,
            ordinal_features=None):
        """Fit the model, resolving any opt-in product preset."""
        private_cross_mode = getattr(
            self, "_group_centered_crosses_private_mode", None
        )
        if private_cross_mode is None:
            if _normalize_categorical_crosses(self.categorical_crosses):
                normalized_callbacks = _normalize_callbacks(callbacks)
                reason = self._group_centered_cross_ineligible_reason(
                    X,
                    cat_features,
                    eval_set=eval_set,
                    callbacks=normalized_callbacks,
                    ordinal_features=ordinal_features,
                )
                if (
                    reason is not None
                    and reason
                    not in _GROUP_CENTERED_CROSSES_DATA_FALLBACK_REASONS
                ):
                    raise ValueError(
                        "categorical_crosses=True is incompatible with the "
                        f"requested fit ({reason})"
                    )
                return self._fit_group_centered_cross_selector(
                    X,
                    y,
                    cat_features=cat_features,
                    eval_set=eval_set,
                    groups=groups,
                    sample_weight=sample_weight,
                    eval_sample_weight=eval_sample_weight,
                    callbacks=normalized_callbacks,
                    ordinal_features=ordinal_features,
                    persist_fallback_metadata=True,
                )
            private_cross_mode = "off"
        if private_cross_mode == "auto":
            return self._fit_group_centered_cross_selector(
                X,
                y,
                cat_features=cat_features,
                eval_set=eval_set,
                groups=groups,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight,
                callbacks=_normalize_callbacks(callbacks),
                ordinal_features=ordinal_features,
                persist_fallback_metadata=False,
            )
        if private_cross_mode not in {"off", "forced"}:
            raise RuntimeError("invalid private group-centered cross mode")
        self._clear_group_centered_cross_state()
        oblivious_kernel = _normalize_oblivious_kernel(self.oblivious_kernel)
        if oblivious_kernel != "auto" and _is_auto_tree_mode(self.tree_mode):
            raise ValueError(
                "an explicit oblivious_kernel requires an explicit "
                "tree_mode='catboost'; tree_mode='auto' auditions "
                "non-oblivious modes"
            )
        ensemble_mode, future_values, explicit_values = (
            _resolve_public_ensemble_surface(self)
        )
        self._clear_ensemble_state()
        if not getattr(self, "_suppress_wrapper_deprecation_warning", False):
            self._warn_wrapper_deprecated_options(
                stacklevel=(
                    8
                    if getattr(
                        self, "_group_centered_cross_fallback_active", False
                    )
                    else 4
                )
            )
        if ensemble_mode == "v3":
            return _fit_public_ensemble_v3(
                self,
                X,
                y,
                future_values=future_values,
                explicit_values=explicit_values,
                cat_features=cat_features,
                eval_set=eval_set,
                groups=groups,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight,
                callbacks=callbacks,
                ordinal_features=ordinal_features,
            )
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
        oblivious_kernel = _normalize_oblivious_kernel(self.oblivious_kernel)
        if oblivious_kernel != "auto" and _is_auto_tree_mode(self.tree_mode):
            raise ValueError(
                "an explicit oblivious_kernel requires an explicit "
                "tree_mode='catboost'; tree_mode='auto' auditions "
                "non-oblivious modes"
            )
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
        interval_calibration_ = _normalize_interval_calibration(
            self.interval_calibration
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
        if not distributional_loss and interval_calibration_ is not None:
            raise ValueError(
                "interval_calibration is only supported for "
                "distributional losses"
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
            if (
                interval_calibration_ == "conformal"
                and self.loss != "Gaussian"
            ):
                raise ValueError(
                    "interval_calibration='conformal' is currently "
                    "supported only for loss='Gaussian'"
                )
        elif self.dist_params not in (None, {}):
            raise ValueError("dist_params is only supported for distributional losses")
        self._validate_tree_mode_selection_request()
        if self.refit:
            self._refit_strategy_exponent(self.refit_strategy)
        es_active = _should_early_stop(self.early_stopping)
        if interval_calibration_ is not None and self.refit:
            raise ValueError(
                "interval_calibration='conformal' is incompatible with "
                "refit=True because refitting on the held-out calibration "
                "rows would invalidate split-conformal coverage"
            )
        if (
            interval_calibration_ == "conformal"
            and (
                sample_weight is not None
                or eval_sample_weight is not None
            )
        ):
            raise ValueError(
                "interval_calibration='conformal' does not yet support "
                "sample weights"
            )
        if validation_strategy_ == "group" and groups is None:
            raise ValueError("validation_strategy='group' requires groups")
        if (
            (es_active or tree_mode_auto or interval_calibration_ is not None)
            and eval_set is None
            and groups is not None
            and validation_strategy_ == "weighted_stratified"
        ):
            raise ValueError(
                "validation_strategy='weighted_stratified' is only supported "
                "for ungrouped regression automatic validation splits"
            )
        if (
            es_active
            or tree_mode_auto
            or interval_calibration_ is not None
        ) and eval_set is None:
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

        conformal_eval_set = None
        conformal_split_metadata = None
        if interval_calibration_ == "conformal":
            selection_needed = bool(
                es_active
                or tree_mode_auto
                or dist_calibration_ is not None
                or self.use_best_model
                or self.auto_learning_rate_probe
            )
            eval_set, conformal_eval_set, conformal_split_metadata = (
                _reserve_conformal_holdout(
                    eval_set,
                    fit_random_state,
                    selection_needed=selection_needed,
                )
            )
            split_eval_n = None if eval_set is None else len(eval_set[1])
            eval_sample_weight = None

        y = self._fit_linear_residual_trend(
            X, y, sample_weight, cat_features, feature_names
        )
        if eval_set is not None and self.linear_residual_active_:
            X_eval, y_eval = eval_set
            eval_set = (
                X_eval,
                self.linear_residual_trend_.residualize(X_eval, y_eval),
            )
        if conformal_eval_set is not None and self.linear_residual_active_:
            X_conformal, y_conformal = conformal_eval_set
            conformal_eval_set = (
                X_conformal,
                self.linear_residual_trend_.residualize(
                    X_conformal, y_conformal
                ),
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
        if interval_calibration_ is not None and eval_set is None:
            if conformal_eval_set is None:
                raise ValueError(
                    "interval_calibration='conformal' requires a validation "
                    "set; pass eval_set or allow the automatic validation split"
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
            model._group_centered_pairs = list(
                getattr(self, "_group_centered_pairs_override", ())
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
        elif interval_calibration_ is not None:
            split_source = "automatic_interval_calibration"
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
                        calibration_groups = _extract_feature_column_by_index(
                            X_cal, feature_index
                        )
                        calibration = _fit_grouped_affine_sigma_calibration(
                            selection_model, X_cal, y_cal, calibration_groups,
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

        if distributional_loss and interval_calibration_ is not None:
            self.interval_calibration_ = interval_calibration_
            self.interval_calibration_source_ = "held_out_validation"
            if interval_calibration_ == "conformal":
                X_cal, y_cal = conformal_eval_set
                raw_cal = selection_model.predict_raw(X_cal)
                params_cal = self._calibrated_params_from_raw(raw_cal, X_cal)
                mu_cal = np.asarray(params_cal[0], dtype=np.float64)
                sigma_cal = np.maximum(
                    np.asarray(params_cal[1], dtype=np.float64),
                    _SIGMA_MIN,
                )
                scores = np.abs(
                    (np.asarray(y_cal, dtype=np.float64) - mu_cal) / sigma_cal
                )
                if (
                    scores.ndim != 1
                    or scores.size == 0
                    or not np.all(np.isfinite(scores))
                ):
                    raise RuntimeError(
                        "conformal interval calibration produced invalid "
                        "scores"
                    )
                self.conformal_scores_ = np.sort(scores)
                self.conformal_score_count_ = int(scores.size)
                self.interval_calibration_split_ = {
                    **conformal_split_metadata,
                    "selection_source": (
                        "explicit_eval_set"
                        if explicit_eval_set
                        else "automatic_validation_split"
                    ),
                    "calibration_rows_used_for_fit": False,
                    "calibration_rows_used_for_selection": False,
                    "calibration_rows_used_for_dist_calibration": False,
                }

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
        if distributional_loss and interval_calibration_ is not None:
            self.model_.auto_params_["interval_calibration"] = {
                "method": self.interval_calibration_,
                "source": self.interval_calibration_source_,
                "score_count": int(
                    getattr(self, "conformal_score_count_", 0)
                ),
                "weighted": False,
                "split": dict(self.interval_calibration_split_),
            }
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
        cross_metadata = getattr(
            self, "group_centered_categorical_crosses_", None
        )
        if isinstance(cross_metadata, Mapping) and cross_metadata.get("selected"):
            raise NotImplementedError(
                "shap_values() is not implemented with active group-centered "
                "categorical crosses"
            )
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

    def predict_interval(self, X, alpha=0.1, calibrate=None):
        """Return central prediction interval bounds.

        Pass ``calibrate="conformal"`` to use held-out standardized
        residual scores collected by a fit with
        ``interval_calibration="conformal"``. The default remains the
        fitted distribution's parametric interval.
        """
        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        calibration = _normalize_interval_calibration(calibrate)
        X = _check_predict_input(self, X)
        loss = self._require_distributional(
            "predict_interval", capability="interval"
        )
        trend = self._linear_residual_trend(X)
        raw = self.model_.predict_raw(X, _validated=True)
        params = self._calibrated_params_from_raw(raw, X)
        if calibration == "conformal":
            if getattr(self, "interval_calibration_", None) != "conformal":
                raise ValueError(
                    "calibrate='conformal' requires fitting with "
                    "interval_calibration='conformal'"
                )
            if self._fitted_loss_name() != "Gaussian":
                raise ValueError(
                    "conformal intervals are currently supported only for "
                    "loss='Gaussian'"
                )
            quantile, _ = _conformal_order_statistic(
                self.conformal_scores_, alpha
            )
            mu = np.asarray(params[0], dtype=np.float64)
            sigma = np.maximum(
                np.asarray(params[1], dtype=np.float64), _SIGMA_MIN
            )
            return self._linear_residual_shift_interval(
                (mu - quantile * sigma, mu + quantile * sigma),
                trend,
            )
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
                group_codes=getattr(self, "_ensemble_group_codes_", None),
            )
            return
        from .serialization import save_booster
        params = self._wrapper_params_header()
        _validate_loaded_wrapper_fitted_params(params, self.model_)
        save_booster(
            self.model_, path,
            wrapper_header={"wrapper_class": type(self).__name__,
                            "params": params,
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
        if not isinstance(wrapper_header, Mapping):
            raise ValueError(
                "invalid DarkoFit model: wrapper header must be an object"
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
        if not (
            isinstance(booster, DistributionalBoosting)
            or (
                isinstance(booster, GradientBoosting)
                and getattr(booster, "loss_name", None)
                in _SCALAR_REGRESSION_LOSSES
            )
        ):
            raise ValueError(
                "invalid DarkoFit model: regressor booster family is invalid"
            )
        est = cls()
        params = wrapper_header.get("params", {})
        if not isinstance(params, Mapping):
            raise ValueError(
                "invalid DarkoFit model: wrapper params must be an object"
            )
        _validate_loaded_wrapper_fitted_params(params, booster)
        known = est.get_params()
        est.set_params(**{k: v for k, v in params.items() if k in known})
        est.model_ = booster
        if isinstance(booster, DistributionalBoosting):
            est.loss = booster.loss_name
        elif isinstance(booster, GradientBoosting):
            est.loss = booster.loss_name
            if booster.loss_name == "Quantile":
                est.alpha = float(booster.loss_kwargs["alpha"])
        if "random_state" not in params:
            est.random_state = getattr(booster, "random_state", None)
        if "oblivious_kernel" not in params:
            est.oblivious_kernel = getattr(booster, "oblivious_kernel", "auto")
        state = wrapper_header.get("state", {})
        if not isinstance(state, Mapping):
            raise ValueError(
                "invalid DarkoFit model: wrapper state must be an object"
            )
        _validate_loaded_linear_residual_params(params, state)
        est._restore_wrapper_state(state, params)
        est._restore_interval_calibration_state(state, wrapper_arrays)
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
        Number of OOB-selected bootstrap members, from 1 through 256. Values
        above one opt into soft-vote aggregation.
        ``ensemble_bootstrap="groups"`` requires ``groups`` in :meth:`fit`.
    ensemble_mode : {"bootstrap", "v3"}, default "bootstrap"
        Keep legacy bootstrap sampling or select the fixed public v3 recipe.
        V3 requires eight members and uses deterministic 80%
        without-replacement samples with donor-balanced member settings.
    oblivious_kernel : {"auto", "fused", "unfused"}, default "auto"
        Static fused-kernel dispatch for eligible binary CatBoost-mode fits.
        On macOS arm64 within the measured shape envelope, ``"auto"`` uses a
        fixed scan-work threshold to select the fused or unfused lane.
        Multiclass and non-oblivious fits retain their current kernels under
        ``"auto"`` and reject explicit modes.
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
                 ensemble_shared_preprocessing=True,
                 ensemble_mode="bootstrap",
                 ensemble_member_learning_rate="policy",
                 ensemble_member_colsample="policy",
                 oblivious_kernel="auto"):
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
        self.ensemble_mode = ensemble_mode
        self.ensemble_member_learning_rate = ensemble_member_learning_rate
        self.ensemble_member_colsample = ensemble_member_colsample
        self.oblivious_kernel = oblivious_kernel
        self.auto_learning_rate_probe = auto_learning_rate_probe
        self.auto_learning_rate_probe_values = auto_learning_rate_probe_values
        self.auto_learning_rate_probe_iterations = auto_learning_rate_probe_iterations

    @_restore_fitted_state_on_fit_failure
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
        oblivious_kernel = _normalize_oblivious_kernel(self.oblivious_kernel)
        if oblivious_kernel != "auto" and _is_auto_tree_mode(self.tree_mode):
            raise ValueError(
                "an explicit oblivious_kernel requires an explicit "
                "tree_mode='catboost'; tree_mode='auto' auditions "
                "non-oblivious modes"
            )
        ensemble_mode, future_values, explicit_values = (
            _resolve_public_ensemble_surface(self)
        )
        self._clear_ensemble_state()
        if not getattr(self, "_suppress_wrapper_deprecation_warning", False):
            self._warn_wrapper_deprecated_options(stacklevel=4)
        if ensemble_mode == "v3":
            return _fit_public_ensemble_v3(
                self,
                X,
                y,
                future_values=future_values,
                explicit_values=explicit_values,
                cat_features=cat_features,
                eval_set=eval_set,
                groups=groups,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight,
                callbacks=callbacks,
                ordinal_features=ordinal_features,
            )
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
                group_codes=getattr(self, "_ensemble_group_codes_", None),
            )
            return
        from .serialization import _encode_categories, save_booster

        cls_arr = np.asarray(self.classes_)
        if cls_arr.dtype == object:
            values, kinds = _encode_categories(self.classes_)
            wrapper_arrays = {"classes": values, "classes_kinds": kinds}
        else:
            wrapper_arrays = {"classes": cls_arr}
        params = self._wrapper_params_header()
        _validate_loaded_wrapper_fitted_params(params, self.model_)
        save_booster(
            self.model_, path,
            wrapper_header={"wrapper_class": type(self).__name__,
                            "params": params,
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
        if not isinstance(wrapper_header, Mapping):
            raise ValueError(
                "invalid DarkoFit model: wrapper header must be an object"
            )
        saved_class = wrapper_header.get("wrapper_class")
        if saved_class is not None and saved_class != cls.__name__:
            raise TypeError(
                f"{path!r} was saved by {saved_class}, not {cls.__name__}"
            )
        if not (
            isinstance(booster, MulticlassBoosting)
            or (
                isinstance(booster, GradientBoosting)
                and getattr(booster, "loss_name", None) == "Logloss"
            )
        ):
            raise ValueError(
                "invalid DarkoFit model: classifier booster family is invalid"
            )
        est = cls()
        params = wrapper_header.get("params", {})
        if not isinstance(params, Mapping):
            raise ValueError(
                "invalid DarkoFit model: wrapper params must be an object"
            )
        _validate_loaded_wrapper_fitted_params(params, booster)
        known = est.get_params()
        est.set_params(**{k: v for k, v in params.items() if k in known})
        est.model_ = booster
        if "random_state" not in params:
            est.random_state = getattr(booster, "random_state", None)
        if "oblivious_kernel" not in params:
            est.oblivious_kernel = getattr(booster, "oblivious_kernel", "auto")
        state = wrapper_header.get("state", {})
        if not isinstance(state, Mapping):
            raise ValueError(
                "invalid DarkoFit model: wrapper state must be an object"
            )
        est._restore_wrapper_state(state, params)
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
        classes = np.asarray(classes)
        if classes.ndim != 1 or classes.size < 2:
            raise ValueError(
                "invalid DarkoFit model: classifier class labels are invalid"
            )
        if any(
            _is_missing_value(value)
            or (
                isinstance(value, (float, np.floating))
                and not np.isfinite(value)
            )
            for value in classes
        ):
            raise ValueError(
                "invalid DarkoFit model: classifier class labels are invalid"
            )
        try:
            target_type = type_of_target(classes)
            unique_class_count = int(np.unique(classes).size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "invalid DarkoFit model: classifier class labels are invalid"
            ) from exc
        if (
            target_type not in {"binary", "multiclass"}
            or unique_class_count != classes.size
        ):
            raise ValueError(
                "invalid DarkoFit model: classifier class labels are invalid"
            )
        if est._multiclass:
            reference_classes = np.asarray(booster.classes_)
            if (
                classes.size != int(booster.n_classes_)
                or not np.array_equal(classes, reference_classes)
            ):
                raise ValueError(
                    "invalid DarkoFit model: wrapper class labels do not "
                    "match the multiclass booster"
                )
        elif classes.size != 2 or bool(classes[0] == classes[1]):
            raise ValueError(
                "invalid DarkoFit model: binary classifier must contain two "
                "distinct class labels"
            )
        est.classes_ = classes
        est.n_classes_ = int(classes.size)
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
