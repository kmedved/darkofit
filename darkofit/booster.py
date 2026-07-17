"""The gradient boosting core: builds the full additive model.

Boosters share the same machinery (FeaturePreprocessor, oblivious trees):
  * GradientBoosting     -> scalar output (regression, binary classification)
  * MulticlassBoosting   -> K simultaneous outputs (softmax multiclass)
  * DistributionalBoosting -> vector regression heads (Gaussian NLL)
"""

import copy
import ctypes
import hashlib
import operator
import time
import warnings
import numpy as np

from ._numba_runtime import numba_thread_setup
from ._validation import (
    coerce_feature_matrix,
    feature_names_from_input,
    n_features_from_array_like,
    normalize_random_state_seed,
    sklearn_assume_finite,
    validate_feature_names,
    validate_target_vector,
)
from .auto_params import (
    AUTO_LR_RULE,
    AUTO_LR_MIN,
    AUTO_LR_MAX,
    AUTO_LR_FEATURE_MULTIPLIER_MIN,
    AUTO_LR_FEATURE_RATIO_REFERENCE,
    CATBOOST_WEIGHTED_RMSE_LR_MULTIPLIER,
    LIGHTGBM_UNWEIGHTED_LR_MULTIPLIERS,
    effective_sample_size,
    is_auto_learning_rate,
    resolve_learning_rate_details,
)
from .binning import DEFAULT_BIN_SAMPLE_COUNT
from .callbacks import (
    BoostingProgress,
    _callback_stop_reason,
    _normalize_callbacks,
)
from .flat_model import (
    FlatNonObliviousEnsemble,
    build_flat_ensemble,
    build_flat_multiclass_ensemble,
    flat_predict_preferred,
)
from .losses import LOSSES, MultiSoftmax, VECTOR_LOSSES
from .preprocessing import FeaturePreprocessor, _normalize_target_ordered_cat_codes
from .shap import (
    SHAP_BACKGROUND_SIZE,
    SHAP_MAX_PLAYERS,
    factorials,
    max_original_players,
    normalize_max_background,
    pack_oblivious_shap_forest,
    shap_forest_linear,
)
from .tree import (
    _build_multiclass_histograms_counts_into,
    _leaf_values_and_sums,
    _leaf_values_and_sums_rows,
    add_linear_leaf_values_inplace,
    add_leaf_values_inplace,
    add_multiclass_leaf_values_inplace,
    build_hybrid_tree,
    build_leafwise_multiclass_tree,
    build_leafwise_tree,
    build_levelwise_tree,
    build_oblivious_tree,
    attach_oblivious_linear_leaves,
    ordered_leaf_update_inplace,
)

_LEAF_CORRECTION_SORT_MIN_LEAVES = 16
_LOW_EFFECTIVE_SAMPLE_FRACTION = 0.3
_MAX_CONSECUTIVE_SAMPLED_DEPTH0_RETRIES = 10
LINEAR_LEAVES_MIN_SAMPLES = 1000
_EMITTED_DIAGNOSTIC_WARNING_CODES = set()


def reset_diagnostic_warning_registry():
    """Clear process-level diagnostic warning throttling state."""
    _EMITTED_DIAGNOSTIC_WARNING_CODES.clear()


def _apply_thread_count(thread_count):
    """Set numba's thread pool size. None / -1 means use all detected cores.

    Returns the effective thread count so callers can record it.
    """
    with numba_thread_setup():
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
    if tree_mode_ in {"lightgbm", "hybrid"} and n_samples <= 50_000:
        if thread_count is None or thread_count < 0:
            return _apply_thread_count(2)
        return _apply_thread_count(min(int(thread_count), 2))
    return _apply_thread_count(thread_count)


def _reject_eval_sample_weight_without_eval_set(eval_sample_weight, eval_set):
    if eval_sample_weight is not None and eval_set is None:
        raise ValueError("eval_sample_weight requires an explicit eval_set")


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
    if mode in {"hybrid", "hybrid_leafwise", "shared_prefix"}:
        return "hybrid"
    if mode in {"levelwise", "level_wise", "depthwise", "depth_wise",
                "non_oblivious"}:
        return "depthwise"
    raise ValueError(
        "tree_mode must be one of 'catboost', 'oblivious', "
        "'lightgbm', experimental 'hybrid', or experimental 'depthwise'"
    )


def _normalize_eval_metric(eval_metric, loss_name):
    if loss_name in VECTOR_LOSSES:
        loss_cls = VECTOR_LOSSES[loss_name]
        default = getattr(loss_cls, "default_eval_metric", "nll")
        supported = tuple(getattr(loss_cls, "supported_eval_metrics", ("nll",)))
        if default not in supported:
            raise ValueError(
                f"loss={loss_name!r} default_eval_metric={default!r} is not "
                "listed in supported_eval_metrics"
            )
        if eval_metric is None or eval_metric == "auto":
            return default
        metric = str(eval_metric).lower().replace("-", "_")
        aliases = {
            "loss": default,
            "negative_log_likelihood": "nll",
            f"{str(loss_name).lower()}_nll": "nll",
        }
        distribution_name = getattr(loss_cls, "distribution_name", None)
        if distribution_name:
            aliases[f"{str(distribution_name).lower()}_nll"] = "nll"
        metric = aliases.get(metric, metric)
        if metric in supported:
            if metric == "crps" and not hasattr(loss_cls, "crps_class_major"):
                raise ValueError(
                    f"loss={loss_name!r} lists eval_metric='crps' but does "
                    "not implement crps_class_major"
                )
            return metric
        allowed = ", ".join(repr(v) for v in supported)
        raise ValueError(
            f"eval_metric for loss={loss_name!r} must be one of None, "
            f"'auto', 'loss', or {allowed}"
        )
    if eval_metric is None or eval_metric == "auto":
        return "loss"
    metric = str(eval_metric).lower().replace("-", "_")
    if metric != "loss":
        raise ValueError(
            "eval_metric is only configurable for distributional losses"
        )
    return "loss"


def _normalize_sampling(sampling):
    if sampling is None:
        sampling = "uniform"
    mode = str(sampling).lower().replace("-", "_")
    if mode in {"uniform", "random"}:
        return "uniform"
    if mode == "goss":
        return "goss"
    if mode in {"weighted_goss", "weightedgoss", "goss_weighted"}:
        return "weighted_goss"
    if mode == "mvs":
        return "mvs"
    raise ValueError(
        "sampling must be one of 'uniform', experimental 'goss', "
        "experimental 'weighted_goss', or experimental 'mvs'"
    )


def _exact_mvs_probabilities(importance, target):
    importance = np.asarray(importance, dtype=np.float64)
    n_samples = importance.shape[0]
    probs = np.zeros(n_samples, dtype=np.float64)
    if target <= 0.0 or n_samples == 0:
        return probs

    positive = np.maximum(importance, 0.0)
    active = positive > 0.0
    if not np.any(active):
        return np.full(n_samples, target / n_samples, dtype=np.float64)

    values = np.sort(positive[active])[::-1]
    total = float(np.sum(values))
    active_count = values.shape[0]
    if target >= active_count:
        probs[active] = 1.0
        return probs

    prefix = np.concatenate(([0.0], np.cumsum(values)))
    theta = None
    max_saturated = min(active_count - 1, int(np.floor(target)))
    for saturated in range(max_saturated + 1):
        denom = target - saturated
        if denom <= 0.0:
            break
        remaining_sum = total - float(prefix[saturated])
        candidate = remaining_sum / denom
        if candidate <= 0.0 or not np.isfinite(candidate):
            continue
        # Tolerant boundary checks: candidate comes from floating-point
        # division, so an exact piecewise boundary can land 1 ulp on the
        # wrong side of a strict comparison and skip the valid segment,
        # silently falling through to the inexact no-saturation fallback.
        left_ok = (
            saturated == 0
            or values[saturated - 1] >= candidate * (1.0 - 1e-11)
        )
        right_ok = (
            saturated == active_count
            or candidate >= values[saturated] * (1.0 - 1e-11)
        )
        if left_ok and right_ok:
            theta = candidate
            break

    if theta is None:
        theta = total / max(target, 1e-300)
    probs[active] = np.minimum(1.0, positive[active] / max(theta, 1e-300))
    prob_sum = float(np.sum(probs))
    if prob_sum > 0.0:
        probs *= target / max(prob_sum, 1e-300)
    return np.minimum(1.0, probs)


def _exact_weighted_goss_probabilities(mass, target_mass):
    mass = np.asarray(mass, dtype=np.float64)
    probs = np.zeros(mass.shape[0], dtype=np.float64)
    if mass.size == 0 or target_mass <= 0.0:
        return probs

    positive = np.maximum(mass, 0.0)
    active = positive > 0.0
    if not np.any(active):
        return probs

    total_mass = float(np.sum(positive))
    target_mass = min(float(target_mass), total_mass)
    if target_mass >= total_mass:
        probs[active] = 1.0
        return probs

    values = np.sort(positive[active])[::-1]
    prefix_mass = np.concatenate(([0.0], np.cumsum(values)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(values * values)))
    total_sq = float(prefix_sq[-1])
    active_count = values.shape[0]
    alpha = None

    for saturated in range(active_count + 1):
        saturated_mass = float(prefix_mass[saturated])
        needed = target_mass - saturated_mass
        if needed <= 0.0:
            break
        remaining_sq = total_sq - float(prefix_sq[saturated])
        if remaining_sq <= 0.0:
            break
        candidate = needed / remaining_sq
        if candidate <= 0.0 or not np.isfinite(candidate):
            continue
        # Tolerant boundary checks: see _exact_mvs_probabilities. This path
        # has no post-rescale, so a strict-comparison miss would materially
        # undershoot the target mass via the no-saturation fallback.
        left_ok = (
            saturated == 0
            or candidate * values[saturated - 1] >= 1.0 - 1e-11
        )
        right_ok = (
            saturated == active_count
            or candidate * values[saturated] <= 1.0 + 1e-11
        )
        if left_ok and right_ok:
            alpha = candidate
            break

    if alpha is None:
        alpha = target_mass / max(total_sq, 1e-300)
    probs[active] = np.minimum(1.0, alpha * positive[active])
    return probs


def _array_content_signature(value):
    if value is None:
        return None
    arr = np.asarray(value)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    if arr.dtype == object:
        # Factorization groups object categories by Python equality/hash
        # behavior. Repr/hash fingerprints can legally collide for distinct
        # objects, so hash the raw PyObject* pointer table. This cache is
        # fit-local and only needs to reuse the same object graph safely; using
        # the contiguous pointer bytes keeps large object-array keys cheap.
        obj = np.ascontiguousarray(arr)
        ptr = int(obj.__array_interface__["data"][0])
        h.update(ctypes.string_at(ptr, obj.size * obj.dtype.itemsize))
    else:
        h.update(np.ascontiguousarray(arr).view(np.uint8))
    return (tuple(arr.shape), str(arr.dtype), h.hexdigest())


def _preprocessing_cache_key(
    booster, prep, X, encode_targets, cat_features, sample_weight
):
    # The cached artifacts (fitted preprocessor + binned train matrix) never
    # see the eval set, so the key deliberately excludes it; a selection fit
    # and a same-data refit without the eval set share one entry.
    return (
        booster.__class__.__name__,
        getattr(booster, "loss_name", booster.__class__.__name__),
        int(prep.max_bins),
        float(prep.cat_smoothing),
        prep.random_state,
        bool(prep.include_cat_codes),
        prep.target_encoding_mode,
        int(prep.target_encoding_folds),
        int(prep.ts_permutations),
        prep.target_ordered_cat_codes,
        None if prep.bin_sample_count is None else int(prep.bin_sample_count),
        tuple([] if cat_features is None else [int(c) for c in cat_features]),
        int(X.shape[1]),
        _array_content_signature(X),
        tuple(_array_content_signature(target) for target in encode_targets),
        _array_content_signature(sample_weight),
    )


def _normalize_bootstrap_type(bootstrap_type):
    if bootstrap_type is None:
        bootstrap_type = "none"
    mode = str(bootstrap_type).lower().replace("-", "_")
    if mode in {"none", "no", "off"}:
        return "none"
    if mode in {"bayesian", "bayes"}:
        return "bayesian"
    raise ValueError("bootstrap_type must be 'none' or experimental 'bayesian'")


def _normalize_diagnostic_warnings(value):
    if isinstance(value, bool):
        return "always" if value else "never"
    mode = str(value).lower().replace("-", "_")
    if mode in {"once", "always", "never"}:
        return mode
    raise ValueError(
        "diagnostic_warnings must be 'once', 'always', 'never', or a bool"
    )


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
        raise ValueError(
            f"{name} sums to zero; at least one weight must be positive"
        )
    return w * (n_samples / total)


def _resolve_default_depth(depth, tree_mode_):
    if depth is not None:
        return int(depth)
    if tree_mode_ in {"lightgbm", "hybrid"}:
        return -1
    return 6


def _is_auto_param(value):
    return isinstance(value, str) and value.lower().replace("-", "_") == "auto"


def _normalize_histogram_dtype(histogram_dtype):
    try:
        dtype = np.dtype(histogram_dtype).name
    except (TypeError, ValueError):
        dtype = str(histogram_dtype).lower()
    if dtype not in {"float64", "float32"}:
        raise ValueError("histogram_dtype must be 'float64' or 'float32'")
    return dtype


def _normalize_leaf_dtype_name(leaf_dtype):
    try:
        dtype = np.dtype(leaf_dtype).name
    except (TypeError, ValueError):
        dtype = str(leaf_dtype).lower()
    if dtype not in {"int64", "uint32"}:
        raise ValueError("leaf_dtype must be 'int64' or 'uint32'")
    return dtype


def _normalize_iterations(iterations):
    """Return a nonnegative Python integer without truncating user input."""
    if isinstance(iterations, (bool, np.bool_)):
        raise TypeError("iterations must be a nonnegative integer")
    try:
        value = operator.index(iterations)
    except TypeError as exc:
        raise TypeError("iterations must be a nonnegative integer") from exc
    if value < 0:
        raise ValueError("iterations must be nonnegative")
    return int(value)


class _BaseBooster:
    """Shared machinery for the scalar and multiclass boosters.

    Holds the common hyperparameters and the helpers both subclasses use:
    histogram-buffer allocation, column subsampling, row subsampling, feature
    preprocessing, and split-gain feature importances. Subclasses implement
    `fit` and `predict_raw`.
    """

    def __init__(self, iterations=1000, learning_rate=None, depth=None,
                 l2_leaf_reg=3.0, max_bins=254, subsample=1.0,
                 colsample=1.0, cat_smoothing=1.0, early_stopping_rounds=None,
                 early_stopping_min_delta=None,
                 min_child_weight=1.0, min_child_samples=20,
                 min_gain_to_split=0.0, num_leaves=None, thread_count=None,
                 random_state=None, verbose=False, ordered_boosting="auto",
                 verbose_timing=False, tree_mode="catboost",
                 sampling="uniform", top_rate=0.2, other_rate=0.1,
                 multiclass_tree_strategy="auto", eval_train_loss=False,
                 bin_sample_count=DEFAULT_BIN_SAMPLE_COUNT,
                 histogram_parallelism="auto", use_best_model=True,
                 bootstrap_type="none", bagging_temperature=0.0,
                 mvs_reg=1.0, random_strength=0.0,
                 diagnostic_warnings="once", histogram_dtype="float64",
                 leaf_dtype="int64", ts_permutations=1,
                 target_ordered_cat_codes="off", eval_metric=None,
                 rho_learning_rate_multiplier=1.0,
                 rho_l2_leaf_reg_multiplier=1.0,
                 linear_leaves=False, linear_lambda=1.0):
        self.iterations = _normalize_iterations(iterations)
        self.learning_rate = learning_rate
        self._depth_input = depth
        self._num_leaves_input = num_leaves
        self._l2_leaf_reg_input = l2_leaf_reg
        self._min_child_samples_input = min_child_samples
        self._min_child_weight_input = min_child_weight
        self._cat_smoothing_input = cat_smoothing
        self.l2_leaf_reg = l2_leaf_reg if _is_auto_param(l2_leaf_reg) else float(l2_leaf_reg)
        self.max_bins = int(max_bins)
        self.subsample = float(subsample)
        self.colsample = float(colsample)
        self.cat_smoothing = (
            cat_smoothing if _is_auto_param(cat_smoothing) else float(cat_smoothing)
        )
        if not _is_auto_param(self.cat_smoothing) and self.cat_smoothing <= 0.0:
            raise ValueError("cat_smoothing must be positive")
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_min_delta = early_stopping_min_delta
        self.min_child_weight = (
            min_child_weight
            if _is_auto_param(min_child_weight)
            else float(min_child_weight)
        )
        self.min_child_samples = (
            min_child_samples
            if _is_auto_param(min_child_samples)
            else int(min_child_samples)
        )
        self.min_gain_to_split = float(min_gain_to_split)
        self.num_leaves = num_leaves
        self.thread_count = thread_count
        self.random_state = random_state
        self.verbose = verbose
        self.ordered_boosting = ordered_boosting
        self.verbose_timing = bool(verbose_timing)
        self.tree_mode = tree_mode
        self.tree_mode_ = _normalize_tree_mode(tree_mode)
        self.depth = (
            "auto"
            if _is_auto_param(depth)
            else _resolve_default_depth(depth, self.tree_mode_)
        )
        self.sampling = sampling
        self.top_rate = float(top_rate)
        self.other_rate = float(other_rate)
        self.multiclass_tree_strategy = multiclass_tree_strategy
        self.eval_train_loss = bool(eval_train_loss)
        self.bin_sample_count = bin_sample_count
        self.histogram_parallelism = histogram_parallelism
        self.use_best_model = bool(use_best_model)
        self.bootstrap_type = bootstrap_type
        self.bootstrap_type_ = _normalize_bootstrap_type(bootstrap_type)
        self.bagging_temperature = float(bagging_temperature)
        self.mvs_reg = float(mvs_reg)
        self.random_strength = float(random_strength)
        self.diagnostic_warnings = diagnostic_warnings
        self.diagnostic_warnings_ = _normalize_diagnostic_warnings(
            diagnostic_warnings
        )
        self.histogram_dtype = _normalize_histogram_dtype(histogram_dtype)
        self.histogram_dtype_ = self.histogram_dtype
        self.leaf_dtype = _normalize_leaf_dtype_name(leaf_dtype)
        self.leaf_dtype_ = self.leaf_dtype
        self.ts_permutations = int(ts_permutations)
        if self.ts_permutations < 1:
            raise ValueError("ts_permutations must be at least 1")
        self.target_ordered_cat_codes = _normalize_target_ordered_cat_codes(
            target_ordered_cat_codes
        )
        self.eval_metric = eval_metric
        self.eval_metric_ = None
        self.rho_learning_rate_multiplier = float(rho_learning_rate_multiplier)
        self.rho_l2_leaf_reg_multiplier = float(rho_l2_leaf_reg_multiplier)
        if not isinstance(linear_leaves, (bool, np.bool_)):
            raise TypeError("linear_leaves must be a bool")
        self.linear_leaves = bool(linear_leaves)
        self.linear_lambda = float(linear_lambda)
        if self.bagging_temperature < 0.0:
            raise ValueError("bagging_temperature must be nonnegative")
        if self.mvs_reg < 0.0:
            raise ValueError("mvs_reg must be nonnegative")
        if self.random_strength < 0.0:
            raise ValueError("random_strength must be nonnegative")
        if not np.isfinite(self.linear_lambda) or self.linear_lambda < 0.0:
            raise ValueError("linear_lambda must be nonnegative and finite")
        if (
            not np.isfinite(self.rho_learning_rate_multiplier)
            or self.rho_learning_rate_multiplier <= 0.0
        ):
            raise ValueError(
                "rho_learning_rate_multiplier must be a positive finite number"
            )
        if (
            not np.isfinite(self.rho_l2_leaf_reg_multiplier)
            or self.rho_l2_leaf_reg_multiplier <= 0.0
        ):
            raise ValueError(
                "rho_l2_leaf_reg_multiplier must be a positive finite number"
            )
        if histogram_parallelism not in {"auto", "feature", "row"}:
            raise ValueError(
                "histogram_parallelism must be 'auto', 'feature', or 'row'"
            )
        self._validate_sampling_config()

    def _resolve_auto_structure_params(
        self, *, loss_name, n_samples, sample_weight, X=None, cat_features=None
    ):
        n_eff = effective_sample_size(sample_weight, n_samples)
        n_eff_fraction = n_eff / float(n_samples) if n_samples else 0.0
        resolved = {}
        candidates = {}

        depth_input = self._depth_input
        if _is_auto_param(depth_input):
            if self.tree_mode_ in {"lightgbm", "hybrid"}:
                depth = -1
                candidates["depth"] = {"rule": f"{self.tree_mode_}_unlimited"}
            else:
                if n_eff < 512:
                    depth = 4
                elif n_eff < 5_000:
                    depth = 5
                elif n_eff < 50_000:
                    depth = 6
                else:
                    depth = 7
                candidates["depth"] = {"rule": "n_eff_buckets_4_7"}
            self.depth = int(depth)
            depth_source = "auto"
        else:
            if self.depth == "auto":
                self.depth = _resolve_default_depth(depth_input, self.tree_mode_)
            if (
                depth_input is None
                and self.tree_mode_ == "depthwise"
                and loss_name == "RMSE"
            ):
                self.depth = 2
                candidates["depth"] = {
                    "rule": "depthwise_rmse_shallow_default"
                }
            depth_source = "default" if depth_input is None else "explicit"
        resolved["depth"] = {
            "input": depth_input,
            "resolved": self.depth,
            "source": depth_source,
        }

        num_leaves_input = self._num_leaves_input
        if _is_auto_param(num_leaves_input):
            if self.tree_mode_ in {"lightgbm", "hybrid"}:
                leaves = int(np.clip(round(np.sqrt(max(n_eff, 1.0))), 7, 63))
                if self.depth is not None and self.depth > 0:
                    leaves = min(leaves, 1 << int(self.depth))
                self.num_leaves = max(1, leaves)
                source = "auto"
                candidates["num_leaves"] = {"rule": "sqrt_n_eff_clipped_7_63"}
            else:
                self.num_leaves = None
                source = "auto_disabled_for_symmetric_trees"
                candidates["num_leaves"] = {"rule": "not_applicable"}
        else:
            source = "default" if num_leaves_input is None else "explicit"
        resolved["num_leaves"] = {
            "input": num_leaves_input,
            "resolved": self.num_leaves,
            "source": source,
        }

        l2_input = self._l2_leaf_reg_input
        if _is_auto_param(l2_input):
            if self.tree_mode_ == "lightgbm":
                base = 1.0
            elif self.tree_mode_ == "hybrid":
                base = 2.0
            else:
                base = 3.0
            concentration = (
                np.sqrt(1.0 / max(n_eff_fraction, 1e-12))
                if n_eff_fraction < 1.0 else 1.0
            )
            self.l2_leaf_reg = float(np.clip(base * concentration, base, 20.0))
            source = "auto"
            candidates["l2_leaf_reg"] = {
                "rule": "base_by_tree_mode_times_weight_concentration",
                "base": base,
            }
        else:
            self.l2_leaf_reg = float(self.l2_leaf_reg)
            source = "default" if l2_input == 3.0 else "explicit"
        resolved["l2_leaf_reg"] = {
            "input": l2_input,
            "resolved": float(self.l2_leaf_reg),
            "source": source,
        }

        samples_input = self._min_child_samples_input
        if _is_auto_param(samples_input):
            self.min_child_samples = int(np.clip(np.ceil(0.01 * n_eff), 5, 100))
            source = "auto"
            candidates["min_child_samples"] = {"rule": "ceil_1pct_n_eff_clipped_5_100"}
        else:
            self.min_child_samples = int(self.min_child_samples)
            source = "default" if samples_input == 20 else "explicit"
        resolved["min_child_samples"] = {
            "input": samples_input,
            "resolved": int(self.min_child_samples),
            "source": source,
        }

        weight_input = self._min_child_weight_input
        if _is_auto_param(weight_input):
            self.min_child_weight = float(np.clip(0.001 * n_eff, 1.0, 50.0))
            source = "auto"
            candidates["min_child_weight"] = {"rule": "0.1pct_n_eff_clipped_1_50"}
        else:
            self.min_child_weight = float(self.min_child_weight)
            source = "default" if weight_input == 1.0 else "explicit"
        resolved["min_child_weight"] = {
            "input": weight_input,
            "resolved": float(self.min_child_weight),
            "source": source,
        }

        cat_cardinalities = []
        smoothing_input = self._cat_smoothing_input
        if _is_auto_param(smoothing_input):
            if cat_features and X is not None:
                X_arr = np.asarray(X, dtype=object)
                for j in cat_features:
                    col = X_arr[:, int(j)]
                    cat_cardinalities.append(
                        len({(type(v).__name__, repr(v)) for v in col})
                    )
            if cat_cardinalities:
                median_cardinality = float(np.median(cat_cardinalities))
                smoothing = np.sqrt(max(median_cardinality, 1.0))
                if n_eff_fraction < 1.0:
                    smoothing *= np.sqrt(1.0 / max(n_eff_fraction, 1e-12))
                if (
                    self.tree_mode_ in {"lightgbm", "hybrid"}
                    and (loss_name == "RMSE" or loss_name in VECTOR_LOSSES)
                    and self._include_cat_codes()
                ):
                    smoothing = max(smoothing, 3.0)
                self.cat_smoothing = float(np.clip(smoothing, 1.0, 50.0))
                rule = "sqrt_median_cardinality_weight_adjusted"
            else:
                self.cat_smoothing = 1.0
                rule = "no_categoricals"
            source = "auto"
            candidates["cat_smoothing"] = {"rule": rule}
        else:
            self.cat_smoothing = float(self.cat_smoothing)
            if self.cat_smoothing <= 0.0:
                raise ValueError("cat_smoothing must be positive")
            source = "default" if smoothing_input == 1.0 else "explicit"
        resolved["cat_smoothing"] = {
            "input": smoothing_input,
            "resolved": float(self.cat_smoothing),
            "source": source,
            "cat_cardinalities": cat_cardinalities,
        }

        self._auto_structure_params_ = {
            "n_eff": float(n_eff),
            "n_eff_fraction": float(n_eff_fraction),
            "resolved": resolved,
            "candidates": candidates,
        }

    def _validate_sampling_config(self):
        self.sampling_ = _normalize_sampling(self.sampling)
        self.bootstrap_type_ = _normalize_bootstrap_type(self.bootstrap_type)
        self.top_rate = float(self.top_rate)
        self.other_rate = float(self.other_rate)
        if not np.isfinite(self.subsample):
            raise ValueError("subsample must be finite")
        if (
            self.sampling_ in {"uniform", "mvs"}
            and not (0.0 < self.subsample <= 1.0)
        ):
            raise ValueError(
                "subsample must be in (0, 1] for sampling='uniform' or sampling='mvs'"
            )
        if self.sampling_ in {"goss", "weighted_goss"}:
            if self.subsample != 1.0:
                raise ValueError(
                    "subsample must be 1.0 for sampling='goss' or "
                    "sampling='weighted_goss'"
                )
            if not (0.0 < self.top_rate < 1.0):
                raise ValueError(
                    "top_rate must be in (0, 1) for sampling='goss' or "
                    "sampling='weighted_goss'"
                )
            if not (0.0 < self.other_rate < 1.0):
                raise ValueError(
                    "other_rate must be in (0, 1) for sampling='goss' or "
                    "sampling='weighted_goss'"
                )
            if self.top_rate + self.other_rate > 1.0:
                raise ValueError(
                    "top_rate + other_rate must be <= 1 for sampling='goss' "
                    "or sampling='weighted_goss'"
                )

    def _bayesian_bootstrap_active(self):
        return self.bootstrap_type_ == "bayesian" and self.bagging_temperature > 0.0

    def _mvs_active(self):
        return self.sampling_ == "mvs" and self.subsample < 1.0

    def _row_sampling_active(self):
        return (
            self.sampling_ in {"goss", "weighted_goss"}
            or self._mvs_active()
            or (self.sampling_ == "uniform" and self.subsample < 1.0)
        )

    def _reset_stochastic_diagnostics(self):
        self._stochastic_diagnostics_ = {
            "sampled_rows": 0,
            "sampling_rounds": 0,
            "sampling_fraction_sum": 0.0,
            "bootstrap_rounds": 0,
            "random_strength_seed_policy": (
                "per_tree_deterministic_hash"
                if self.random_strength > 0.0
                else "disabled"
            ),
        }

    def _record_sampling_diagnostic(self, row_indices, n_samples):
        diag = getattr(self, "_stochastic_diagnostics_", None)
        if diag is None:
            return
        if row_indices is None:
            sampled = int(n_samples)
        else:
            sampled = int(row_indices.shape[0])
        diag["sampled_rows"] += sampled
        diag["sampling_rounds"] += 1
        diag["sampling_fraction_sum"] += (
            sampled / float(n_samples) if n_samples else 0.0
        )

    def _record_bootstrap_diagnostic(self, factors):
        diag = getattr(self, "_stochastic_diagnostics_", None)
        if diag is None or factors is None:
            return
        diag["bootstrap_rounds"] += 1

    def _initialize_split_seed(self, rng, random_state_seed):
        if self.random_strength <= 0.0:
            self._split_seed_ = (
                0 if random_state_seed is None else int(random_state_seed)
            )
            return
        if random_state_seed is None:
            self._split_seed_ = int(rng.integers(0, np.iinfo(np.int32).max))
        else:
            self._split_seed_ = int(random_state_seed)

    def _bayesian_bootstrap_factors(self, n_samples, rng):
        if (
            not self._bayesian_bootstrap_active()
            or n_samples <= 0
        ):
            return None
        factors = rng.exponential(scale=1.0, size=n_samples)
        if self.bagging_temperature != 1.0:
            factors = factors ** self.bagging_temperature
        mean = float(np.mean(factors))
        if mean > 0.0 and np.isfinite(mean):
            factors = factors / mean
        self._record_bootstrap_diagnostic(factors)
        return factors.astype(np.float64, copy=False)

    def _apply_bootstrap(self, grad, hess, factors):
        if factors is None:
            return grad, hess
        return grad * factors, hess * factors

    def _apply_bootstrap_multiclass(self, grad, hess, factors):
        if factors is None:
            return grad, hess
        return grad * factors[None, :], hess * factors[None, :]

    def _resolve_fit_auto_params(self, *, loss_name, n_samples, sample_weight,
                                 eval_set_present, p_model=None):
        self.iterations_ = int(self.iterations)
        self.use_best_model_ = bool(eval_set_present and self.use_best_model)
        n_eff = effective_sample_size(sample_weight, n_samples)
        n_eff_fraction = n_eff / float(n_samples) if n_samples else 0.0
        self._learning_rate_details_ = resolve_learning_rate_details(
            self.learning_rate,
            loss_name=loss_name,
            n_eff=n_eff,
            iterations=self.iterations_,
            use_best_model=self.use_best_model_,
            tree_mode=self.tree_mode_,
            max_leaves=self._max_tree_leaves(),
            n_eff_fraction=n_eff_fraction,
            p_model=p_model,
        )
        self.lr_ = self._learning_rate_details_["resolved"]
        if self.early_stopping_rounds is None:
            self.early_stopping_rounds_ = None
            self.early_stopping_rounds_rule_ = "none"
        elif self.early_stopping_rounds == "auto":
            self.early_stopping_rounds_ = int(
                np.clip(np.ceil(5.0 / max(self.lr_, 1e-12)), 20, 200)
            )
            self.early_stopping_rounds_rule_ = "ceil(5/lr)_clipped_20_200"
        else:
            self.early_stopping_rounds_ = int(self.early_stopping_rounds)
            self.early_stopping_rounds_rule_ = "explicit"
        if self.early_stopping_min_delta is None:
            self.early_stopping_min_delta_ = 1e-9
            self.early_stopping_min_delta_rule_ = "legacy_1e-9"
        elif self.early_stopping_min_delta == "auto":
            self.early_stopping_min_delta_ = None
            self.early_stopping_min_delta_rule_ = "auto"
        else:
            self.early_stopping_min_delta_ = float(self.early_stopping_min_delta)
            if self.early_stopping_min_delta_ < 0.0:
                raise ValueError("early_stopping_min_delta must be nonnegative")
            self.early_stopping_min_delta_rule_ = "explicit"
        if self.verbose and is_auto_learning_rate(self.learning_rate):
            print(f"Learning rate set to {self.lr_}")

    def _finalize_early_stopping_min_delta(self, baseline_loss, loss_name):
        if self.early_stopping_min_delta_ is not None:
            return
        delta = max(1e-12, 1e-4 * abs(float(baseline_loss)))
        self.early_stopping_min_delta_ = float(delta)

    def _record_scalar_target_stats(self, y, sample_weight):
        y = np.asarray(y, dtype=np.float64)
        stats = {
            "n_samples": int(y.shape[0]),
            "mean": float(np.mean(y)) if y.size else None,
            "std": float(np.std(y)) if y.size else None,
            "min": float(np.min(y)) if y.size else None,
            "max": float(np.max(y)) if y.size else None,
        }
        if sample_weight is not None and y.size:
            w = np.asarray(sample_weight, dtype=np.float64)
            mean = float(np.average(y, weights=w))
            variance = float(np.average((y - mean) ** 2, weights=w))
            stats.update({
                "weighted_mean": mean,
                "weighted_std": float(np.sqrt(max(variance, 0.0))),
            })
        self._target_stats_ = stats

    def _record_classification_target_stats(self, y_idx, classes, sample_weight):
        y_idx = np.asarray(y_idx, dtype=np.int64)
        counts = np.bincount(y_idx, minlength=len(classes)).astype(np.float64)
        if sample_weight is None:
            weighted_counts = counts.copy()
        else:
            weighted_counts = np.bincount(
                y_idx, weights=np.asarray(sample_weight, dtype=np.float64),
                minlength=len(classes),
            ).astype(np.float64)
        total = float(np.sum(counts))
        weighted_total = float(np.sum(weighted_counts))
        self._target_stats_ = {
            "n_samples": int(y_idx.shape[0]),
            "n_classes": int(len(classes)),
            "class_counts": counts.astype(int).tolist(),
            "class_weighted_counts": weighted_counts.tolist(),
            "class_frequencies": (
                (counts / total).tolist() if total > 0.0 else []
            ),
            "class_weighted_frequencies": (
                (weighted_counts / weighted_total).tolist()
                if weighted_total > 0.0 else []
            ),
        }

    def _diagnostic_warnings(self, *, n_eff_fraction, sample_weight_provided):
        warning_records = []
        if (
            self._learning_rate_details_.get("clipped")
            and is_auto_learning_rate(self.learning_rate)
        ):
            bound = self._learning_rate_details_["clip_bound"]
            limit = AUTO_LR_MIN if bound == "min" else AUTO_LR_MAX
            raw = self._learning_rate_details_["raw_auto"]
            warning_records.append({
                "code": f"learning_rate_clipped_{bound}",
                "message": (
                    "DarkoFit automatic learning rate clipped to "
                    f"{bound} {limit:g} (raw={raw:.6g})."
                ),
            })
        if (
            sample_weight_provided
            and
            n_eff_fraction < _LOW_EFFECTIVE_SAMPLE_FRACTION
            and n_eff_fraction < 1.0
        ):
            warning_records.append({
                "code": "low_effective_sample_size_fraction",
                "message": (
                    "DarkoFit effective sample size is low "
                    f"(n_eff/n={n_eff_fraction:.3f} < "
                    f"{_LOW_EFFECTIVE_SAMPLE_FRACTION:.2f}); sample weights "
                    "are highly concentrated."
                ),
            })
        return warning_records

    def _emit_auto_param_warnings(self):
        policy = getattr(self, "diagnostic_warnings_", "once")
        diagnostics = getattr(self, "auto_params_", {}).get("diagnostics", {})
        emitted = []
        if policy == "never":
            diagnostics["runtime_warning_policy"] = policy
            diagnostics["runtime_warnings_emitted"] = emitted
            return
        for warning_record in diagnostics.get("warnings", []):
            code = warning_record.get("code")
            if policy == "once" and code in _EMITTED_DIAGNOSTIC_WARNING_CODES:
                continue
            warnings.warn(
                warning_record["message"],
                RuntimeWarning,
                stacklevel=3,
            )
            if code is not None:
                emitted.append(code)
                if policy == "once":
                    _EMITTED_DIAGNOSTIC_WARNING_CODES.add(code)
        diagnostics["runtime_warning_policy"] = policy
        diagnostics["runtime_warnings_emitted"] = emitted

    def _resolved_auto_params(
        self,
        *,
        n_samples,
        n_raw_features,
        X_binned,
        n_bins,
        sample_weight,
        eval_set_present,
        eval_n_samples=0,
        eval_sample_weight=None,
        rowpar_buffers=None,
        extra=None,
    ):
        """Expose resolved fit-time defaults and data-dependent fit context.

        The dictionary is deliberately observational: it records what this fit
        used without changing any model behavior. Future data-aware defaults can
        extend it while preserving a single debugging surface.
        """
        n_eff = effective_sample_size(sample_weight, n_samples)
        n_eff_fraction = n_eff / float(n_samples) if n_samples else 0.0
        max_bins_observed = int(np.max(n_bins)) if len(n_bins) else 0
        observed_total_bins = int(np.sum(n_bins)) if len(n_bins) else 0
        raw_feature_count = int(n_raw_features)
        model_feature_count = int(X_binned.shape[1])
        feature_expansion_factor = (
            model_feature_count / float(raw_feature_count)
            if raw_feature_count else 0.0
        )
        binner = self.prep_.binner_
        weighted_binning_active = bool(getattr(binner, "weighted_", False))
        best_prefix_policy = (
            "validation_best_prefix"
            if eval_set_present and getattr(self, "use_best_model_", False)
            else "disabled"
        )
        lightgbm_unweighted_damping = (
            self.tree_mode_ == "lightgbm"
            and is_auto_learning_rate(self.learning_rate)
            and n_eff_fraction >= 0.99
        )
        lightgbm_multiplier = None
        if lightgbm_unweighted_damping:
            base_loss = getattr(self, "loss_name", "MultiClass")
            if base_loss in {"MAE", "Quantile"} or base_loss in VECTOR_LOSSES:
                base_loss = "RMSE"
            lightgbm_multiplier = LIGHTGBM_UNWEIGHTED_LR_MULTIPLIERS.get(base_loss, 0.4)
        weighted_rmse_catboost_uplift = (
            self.tree_mode_ in {"catboost", "oblivious"}
            and getattr(self, "loss_name", "MultiClass") == "RMSE"
            and is_auto_learning_rate(self.learning_rate)
            and n_eff_fraction < 0.99
        )
        diagnostic_warnings = self._diagnostic_warnings(
            n_eff_fraction=n_eff_fraction,
            sample_weight_provided=sample_weight is not None,
        )
        learning_rate_details = getattr(self, "_learning_rate_details_", {})
        params = {
            "loss": getattr(self, "loss_name", "MultiClass"),
            "iterations": int(self.iterations_),
            "iterations_input": (
                None if self.iterations is None else int(self.iterations)
            ),
            "auto_policy": {
                "learning_rate_rule": (
                    AUTO_LR_RULE
                    if is_auto_learning_rate(self.learning_rate)
                    else "explicit"
                ),
                "lightgbm_unweighted_lr_multiplier": (
                    lightgbm_multiplier
                ),
                "catboost_weighted_rmse_lr_multiplier": (
                    CATBOOST_WEIGHTED_RMSE_LR_MULTIPLIER
                    if weighted_rmse_catboost_uplift
                    else None
                ),
                "feature_lr_reference_ratio": AUTO_LR_FEATURE_RATIO_REFERENCE,
                "feature_lr_min_multiplier": AUTO_LR_FEATURE_MULTIPLIER_MIN,
            },
            "learning_rate": {
                "resolved": float(self.lr_),
                "source": learning_rate_details.get("source", "explicit"),
                "input": learning_rate_details.get("input"),
                "rule": learning_rate_details.get("rule", "explicit"),
                "loss_coefficient_source": learning_rate_details.get(
                    "loss_coefficient_source"
                ),
                "raw_auto": learning_rate_details.get("raw_auto"),
                "p_model": learning_rate_details.get("p_model"),
                "feature_ratio": learning_rate_details.get("feature_ratio"),
                "feature_multiplier": learning_rate_details.get(
                    "feature_multiplier", 1.0
                ),
                "feature_shrinkage_active": bool(
                    learning_rate_details.get("feature_shrinkage_active", False)
                ),
                "clipped": bool(learning_rate_details.get("clipped", False)),
                "clip_bound": learning_rate_details.get("clip_bound"),
                "clip_min": float(AUTO_LR_MIN),
                "clip_max": float(AUTO_LR_MAX),
            },
            "sample_weight": {
                "provided": sample_weight is not None,
                "effective_sample_size": n_eff,
                "effective_sample_size_fraction": (
                    n_eff_fraction
                ),
                "normalized_sum": (
                    None if sample_weight is None else float(np.sum(sample_weight))
                ),
            },
            "target": getattr(self, "_target_stats_", {}),
            "features": {
                "raw_feature_count": raw_feature_count,
                "model_feature_count": model_feature_count,
                "input_feature_count": int(self.prep_.n_input_features_),
                "feature_expansion_factor": feature_expansion_factor,
            },
            "tree": {
                "tree_mode": self.tree_mode_,
                "ordered_boosting": bool(self.ordered_boosting_),
                "ordered_boosting_input": self.ordered_boosting,
                "ordered_boosting_rule": getattr(
                    self, "ordered_boosting_rule_", None
                ),
                "depth": int(self.depth) if self.depth is not None else None,
                "max_tree_depth": int(self._max_tree_depth()),
                "max_leaves": int(self._max_tree_leaves()),
                "num_leaves": None if self.num_leaves is None else int(self.num_leaves),
                "l2_leaf_reg": float(getattr(self, "l2_leaf_reg_", self.l2_leaf_reg)),
                "min_child_samples": int(self.min_child_samples),
                "min_child_weight": float(self.min_child_weight),
                "min_gain_to_split": float(self.min_gain_to_split),
                "auto_structure": getattr(self, "_auto_structure_params_", {}),
            },
            "auto_structure": getattr(self, "_auto_structure_params_", {}),
            "binning": {
                "max_bins": int(self.max_bins),
                "bin_sample_count": (
                    None if self.bin_sample_count is None else int(self.bin_sample_count)
                ),
                "cat_smoothing_input": self._cat_smoothing_input,
                "cat_smoothing_resolved": float(self.prep_.cat_smoothing),
                "numeric_binning_weighted": weighted_binning_active,
                "weighted_sampling": bool(getattr(binner, "weighted_sampling_", False)),
                "weighted_sample_count": getattr(binner, "weighted_sample_count_", None),
                "observed_max_bins": max_bins_observed,
                "observed_total_bins": observed_total_bins,
            },
            "early_stopping": {
                "enabled": bool(self.early_stopping_rounds_ is not None and eval_set_present),
                "rounds": (
                    None
                    if self.early_stopping_rounds_ is None
                    else int(self.early_stopping_rounds_)
                ),
                "rounds_input": self.early_stopping_rounds,
                "rounds_rule": self.early_stopping_rounds_rule_,
                "min_delta": float(self.early_stopping_min_delta_),
                "min_delta_input": self.early_stopping_min_delta,
                "min_delta_rule": self.early_stopping_min_delta_rule_,
                "eval_set_provided": bool(eval_set_present),
                "eval_n_samples": int(eval_n_samples) if eval_set_present else None,
                "eval_sample_weight_provided": eval_sample_weight is not None,
                "use_best_model": bool(
                    eval_set_present and getattr(self, "use_best_model_", False)
                ),
                "use_best_model_input": bool(self.use_best_model),
                "best_prefix_policy": best_prefix_policy,
                "eval_effective_sample_size": (
                    None
                    if not eval_set_present
                    else effective_sample_size(eval_sample_weight, eval_n_samples)
                ),
                "improvement_tolerance": float(self.early_stopping_min_delta_),
            },
            "sampling": {
                "sampling": self.sampling_,
                "weighted_goss_active": bool(self.sampling_ == "weighted_goss"),
                "subsample": float(self.subsample),
                "colsample": float(self.colsample),
                "top_rate": float(self.top_rate),
                "other_rate": float(self.other_rate),
                "mvs_reg": float(self.mvs_reg),
            },
            "stochastic_regularization": self._resolved_stochastic_params(
                n_samples
            ),
            "threading": {
                "thread_count_input": self.thread_count,
                "thread_count_resolved": int(self.n_threads_),
                "histogram_parallelism": self.histogram_parallelism,
                "row_parallel_histograms_active": rowpar_buffers is not None,
            },
            "diagnostics": {
                "warnings": diagnostic_warnings,
                "low_effective_sample_size_fraction_threshold": (
                    _LOW_EFFECTIVE_SAMPLE_FRACTION
                ),
                "effective_sample_size_fraction": n_eff_fraction,
                "learning_rate_clipped": bool(
                    learning_rate_details.get("clipped", False)
                ),
                "learning_rate_clip_bound": learning_rate_details.get("clip_bound"),
                "weighted_binning_active": weighted_binning_active,
                "observed_max_bins": max_bins_observed,
                "observed_total_bins": observed_total_bins,
                "feature_expansion_factor": feature_expansion_factor,
                "learning_rate_feature_multiplier": learning_rate_details.get(
                    "feature_multiplier", 1.0
                ),
                "learning_rate_feature_shrinkage_active": bool(
                    learning_rate_details.get("feature_shrinkage_active", False)
                ),
                "best_prefix_policy": best_prefix_policy,
                "use_best_model": bool(
                    eval_set_present and getattr(self, "use_best_model_", False)
                ),
                "runtime_warning_policy": self.diagnostic_warnings_,
                "runtime_warnings_emitted": [],
            },
        }
        if extra:
            params.update(extra)
        return params

    def _resolved_stochastic_params(self, n_samples):
        diag = getattr(self, "_stochastic_diagnostics_", {})
        rounds = int(diag.get("sampling_rounds", 0))
        sampled_rows = int(diag.get("sampled_rows", 0))
        avg_fraction = (
            float(diag.get("sampling_fraction_sum", 0.0)) / rounds
            if rounds else None
        )
        return {
            "bootstrap_type": self.bootstrap_type_,
            "bagging_temperature": float(self.bagging_temperature),
            "bayesian_bootstrap_active": bool(self._bayesian_bootstrap_active()),
            "bayesian_bootstrap_rounds": int(diag.get("bootstrap_rounds", 0)),
            "sampling": self.sampling_,
            "subsample": float(self.subsample),
            "mvs_reg": float(self.mvs_reg),
            "mvs_active": bool(self._mvs_active()),
            "weighted_goss_active": bool(self.sampling_ == "weighted_goss"),
            "row_sampling_active": bool(self._row_sampling_active()),
            "sampled_rows_total": sampled_rows,
            "sampling_rounds": rounds,
            "average_sampled_row_fraction": avg_fraction,
            "n_samples": int(n_samples),
            "random_strength": float(self.random_strength),
            "random_strength_active": bool(self.random_strength > 0.0),
            "random_strength_seed_policy": diag.get(
                "random_strength_seed_policy", "disabled"
            ),
        }

    def _refresh_stochastic_auto_params(self, n_samples):
        if hasattr(self, "auto_params_"):
            stochastic = self._resolved_stochastic_params(n_samples)
            self.auto_params_["stochastic_regularization"] = stochastic
            self.auto_params_.setdefault("diagnostics", {})
            self.auto_params_["diagnostics"]["stochastic_regularization"] = stochastic

    def _truncate_to_best_model(self, best_iter, valid_history):
        if not (getattr(self, "use_best_model_", False) and valid_history):
            return
        keep = int(best_iter) + 1
        if keep < len(self.trees_):
            self.trees_ = self.trees_[:keep]
            self._flat_cache_ = None
            self._rebuild_importance_from_trees()

    def _callback_stop_reason(self, callbacks, next_iteration,
                              iterations_attempted):
        if not callbacks:
            return None
        progress = BoostingProgress(
            next_iteration=int(next_iteration),
            iterations_attempted=int(iterations_attempted),
            rounds_completed=len(self.trees_),
            last_train_score=(
                None if not self.train_history_ else float(self.train_history_[-1])
            ),
            last_validation_score=(
                None if not self.valid_history_ else float(self.valid_history_[-1])
            ),
        )
        return _callback_stop_reason(callbacks, progress)

    def _finalize_training_metadata(self, *, stop_reason,
                                    iterations_attempted,
                                    rounds_completed,
                                    best_prefix_iter, callbacks):
        retained = len(self.trees_)
        metadata = {
            "stop_reason": str(stop_reason),
            "iterations_requested": int(self.iterations_),
            "iterations_attempted": int(iterations_attempted),
            "rounds_completed": int(rounds_completed),
            "rounds_retained": int(retained),
            "best_prefix_round": (
                int(best_prefix_iter) + 1 if self.valid_history_ else None
            ),
            "best_model_truncated": bool(retained != rounds_completed),
            "stop_check_policy": "before_iteration" if callbacks else "none",
            "time_limit_is_soft": bool(stop_reason == "time_limit"),
        }
        self.stop_reason_ = str(stop_reason)
        self.iterations_attempted_ = int(iterations_attempted)
        self.rounds_completed_ = int(rounds_completed)
        self.training_metadata_ = metadata
        self.auto_params_["training"] = dict(metadata)

    def _catboost_depth(self):
        if self.depth is None or self.depth < 1:
            raise ValueError("depth must be positive for tree_mode='catboost'")
        return int(self.depth)

    def _max_tree_leaves(self):
        if self.tree_mode_ in {"catboost", "depthwise"}:
            if self.num_leaves is not None:
                raise ValueError(
                    "num_leaves is only supported with tree_mode='lightgbm' "
                    "or tree_mode='hybrid'"
                )
            return 1 << self._catboost_depth()
        if self.depth == 0 or (self.depth is not None and self.depth < -1):
            raise ValueError(
                "depth must be positive, None, or -1 for tree_mode='lightgbm' "
                "or tree_mode='hybrid'"
            )
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

    # Scalar regression losses where the separate ordered leave-one-out leaf
    # update creates a train/inference gap. Categorical preprocessing already
    # uses ordered target statistics to control leakage; real-data guardrails
    # found the additional leaf update harmful for both numeric and
    # categorical regression, so "auto" uses plain boosting for all three.
    _ORDERED_AUTO_OFF_REGRESSION_LOSSES = frozenset(
        {"RMSE", "MAE", "Quantile"}
    )
    # Losses whose leaf values are overwritten by a residual statistic
    # (adjusts_leaves): the boosting loop never applies the ordered update
    # for them, so "auto" must resolve off and explicit True is rejected
    # rather than silently ignored.
    _ADJUSTED_LEAF_LOSSES = frozenset({"MAE", "Quantile"})

    def _resolve_ordered_boosting(self, loss_name=None):
        adjusted_leaf_loss = loss_name in self._ADJUSTED_LEAF_LOSSES
        if self.ordered_boosting == "auto":
            if self.tree_mode_ not in {"catboost", "depthwise"}:
                self.ordered_boosting_rule_ = "auto_off_leafwise_mode"
                return False
            if adjusted_leaf_loss:
                self.ordered_boosting_rule_ = "auto_off_adjusted_leaf_loss"
                return False
            if loss_name in self._ORDERED_AUTO_OFF_REGRESSION_LOSSES:
                self.ordered_boosting_rule_ = "auto_off_scalar_regression"
                return False
            self.ordered_boosting_rule_ = "auto_on_symmetric_mode"
            return True
        resolved = bool(self.ordered_boosting)
        if self.tree_mode_ in {"lightgbm", "hybrid"} and resolved:
            raise ValueError(
                "ordered_boosting=True is only supported with tree_mode='catboost'"
            )
        if resolved and adjusted_leaf_loss:
            raise ValueError(
                f"ordered_boosting=True is not supported for loss="
                f"{loss_name!r}: leaf-adjusted losses recompute leaf values "
                "from residual statistics, so the ordered leave-one-out "
                "update never applies"
            )
        self.ordered_boosting_rule_ = "explicit"
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
        n_arrays = 3 if self.tree_mode_ in {"lightgbm", "hybrid"} else 2
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
        return (
            *(np.empty((n_features, max_leaves)) for _ in range(5)),
            np.empty((n_features, max_leaves), dtype=np.int64),
        )

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
        if self.tree_mode_ in {"lightgbm", "hybrid"}:
            leaf_slots, n_arrays = 1, 3
        else:
            leaf_slots, n_arrays = self._max_tree_leaves(), 2
        n_bytes = 8 * n_arrays * n_chunks * n_features * leaf_slots * max_bins
        if n_bytes > self._ROWPAR_MAX_BYTES:
            return None
        shape = (n_chunks, n_features, leaf_slots, max_bins)
        return tuple(np.zeros(shape) for _ in range(n_arrays))

    def _coerce_predict_X(self, X):
        validate_feature_names(
            getattr(self, "feature_names_in_", None),
            X,
            name="X",
            fitted_name=type(self).__name__,
        )
        actual = n_features_from_array_like(X, allow_empty=True)
        expected = getattr(self, "n_features_in_", None)
        if expected is None:
            expected = getattr(self.prep_, "n_input_features_", None)
        if expected is not None and int(actual) != int(expected):
            raise ValueError(
                f"X has {actual} features, but {type(self).__name__} "
                f"is expecting {int(expected)} features as input"
            )
        X, _, _ = coerce_feature_matrix(
            X,
            self.prep_.cat_features_,
            name="X",
            check_infinite=not sklearn_assume_finite(),
            allow_empty=True,
        )
        return X

    def _coerce_fit_X(self, X, cat_features):
        feature_names = feature_names_from_input(X)
        X, cat_features, n_features = coerce_feature_matrix(
            X,
            cat_features,
            name="X",
            resolve_names=True,
            check_infinite=not sklearn_assume_finite(),
        )
        return X, cat_features, int(n_features), feature_names

    def _record_input_feature_metadata(self, n_features, feature_names):
        self.n_features_in_ = int(n_features)
        if feature_names is not None:
            self.feature_names_in_ = feature_names
        elif hasattr(self, "feature_names_in_"):
            delattr(self, "feature_names_in_")

    def _coerce_eval_X(
        self,
        X,
        cat_features,
        *,
        expected_n_features,
        expected_feature_names,
    ):
        validate_feature_names(
            expected_feature_names,
            X,
            name="eval_set[0]",
            fitted_name=type(self).__name__,
        )
        X, _, n_features = coerce_feature_matrix(
            X,
            cat_features,
            name="eval_set[0]",
            check_infinite=not sklearn_assume_finite(),
        )
        if int(n_features) != int(expected_n_features):
            raise ValueError(
                f"eval_set[0] has {n_features} features, but X has "
                f"{expected_n_features} features"
            )
        return X

    def _restore_thread_count(self):
        """Restore this model's fitted Numba thread mask."""
        fitted_threads = getattr(self, "n_threads_", None)
        if fitted_threads is not None:
            _apply_thread_count(fitted_threads)

    def _prepare_predict_X(self, X, *, validated=False):
        """Restore fitted threading and validate prediction input."""
        self._restore_thread_count()
        return X if validated else self._coerce_predict_X(X)

    def _alloc_multiclass_hist_buffers(self, n_classes, n_features, n_bins):
        """Allocate reusable class-minor LightGBM-mode histogram buffers."""
        max_leaves = self._max_tree_leaves()
        max_bins = int(n_bins.max()) if len(n_bins) else 1
        hg = np.zeros((n_features, max_leaves, max_bins, n_classes))
        hh = np.zeros((n_features, max_leaves, max_bins, n_classes))
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
            self.tree_mode_ in {"lightgbm", "hybrid"}
            and (
                getattr(self, "loss_name", None) == "RMSE"
                or getattr(self, "loss_name", None) in VECTOR_LOSSES
            )
            and cat_smoothing == 1.0
            and self._include_cat_codes()
        ):
            # LightGBM-mode regression uses both K-fold target statistics and
            # raw category-code features. A little extra smoothing keeps the
            # target-stat columns from over-specializing before the code columns
            # have a chance to split.
            cat_smoothing = 3.0
        random_state = getattr(
            self, "_fit_random_state_seed_", self.random_state
        )
        return FeaturePreprocessor(
            self.max_bins, cat_smoothing, random_state,
            include_cat_codes=self._include_cat_codes(),
            target_encoding_mode=(
                "kfold"
                if self.tree_mode_ in {"lightgbm", "hybrid"}
                else "ordered"
            ),
            ts_permutations=self.ts_permutations,
            target_ordered_cat_codes=self.target_ordered_cat_codes,
            bin_sample_count=self.bin_sample_count,
        )

    def _fit_transform_preprocessor(
        self, X, encode_targets, cat_features, sample_weight
    ):
        prep = self._new_preprocessor()
        cache = getattr(self, "_preprocessing_cache", None)
        if cache is None:
            self.prep_ = prep
            return prep.fit_transform(
                X, encode_targets, cat_features, sample_weight=sample_weight
            )

        key = _preprocessing_cache_key(
            self, prep, X, encode_targets, cat_features, sample_weight
        )
        cached = cache.get(key)
        if cached is not None:
            cached_prep, cached_X_binned = cached
            self.prep_ = copy.deepcopy(cached_prep)
            X_binned = np.asarray(cached_X_binned).copy()
            if X_binned.shape[0] != X.shape[0]:
                raise ValueError("cached preprocessing row count mismatch")
            if X_binned.shape[1] != self.prep_.n_bins_.shape[0]:
                raise ValueError("cached preprocessing feature count mismatch")
            if not np.array_equal(self.prep_.n_bins_, cached_prep.n_bins_):
                raise ValueError("cached preprocessing n_bins mismatch")
            return X_binned

        self.prep_ = prep
        X_binned = prep.fit_transform(
            X, encode_targets, cat_features, sample_weight=sample_weight
        )
        cache[key] = (copy.deepcopy(prep), np.asarray(X_binned).copy())
        return X_binned

    def _prepare_linear_leaf_state(
        self, X_binned, loss_name, n_samples, sample_weight=None
    ):
        """Resolve and, when eligible, build standardized numeric-bin values."""
        self.linear_leaves_active_ = False
        self.linear_leaves_inactive_reason_ = "disabled"
        self.linear_bin_values_ = None
        self.linear_tree_count_ = 0
        self.linear_leaf_count_ = 0
        self.linear_numeric_features_ = np.zeros(
            X_binned.shape[1], dtype=np.bool_
        )
        if not self.linear_leaves:
            return
        if loss_name != "RMSE":
            raise ValueError(
                "linear_leaves=True is currently supported only for "
                "loss='RMSE'"
            )
        if self.tree_mode_ != "catboost":
            raise ValueError(
                "linear_leaves=True currently requires tree_mode='catboost'"
            )
        if self.ordered_boosting_:
            raise ValueError(
                "linear_leaves=True is incompatible with ordered_boosting=True"
            )
        if n_samples < LINEAR_LEAVES_MIN_SAMPLES:
            self.linear_leaves_inactive_reason_ = "below_min_samples"
            return
        n_numeric = len(self.prep_.num_features_)
        if n_numeric == 0:
            self.linear_leaves_inactive_reason_ = "no_numeric_features"
            return

        self.linear_numeric_features_[:n_numeric] = True
        max_bins = int(self.prep_.n_bins_.max()) if X_binned.shape[1] else 1
        values = np.zeros((X_binned.shape[1], max_bins), dtype=np.float64)
        linear_weight = (
            None
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        for feature in range(n_numeric):
            centers = self.prep_.binner_._centers_for(
                self.prep_.binner_.borders_[feature]
            )
            values[feature, : len(centers)] = centers
            sample_values = centers[X_binned[:, feature]]
            fit_mask = np.isfinite(sample_values)
            if linear_weight is not None:
                fit_mask &= linear_weight > 0.0
            finite = sample_values[fit_mask]
            if not finite.size:
                continue
            if linear_weight is None:
                mean = float(np.mean(finite))
                variance = float(np.mean((finite - mean) ** 2))
            else:
                finite_weight = linear_weight[fit_mask]
                mean = float(np.average(finite, weights=finite_weight))
                variance = float(
                    np.average(
                        (finite - mean) ** 2, weights=finite_weight
                    )
                )
            scale = float(np.sqrt(max(variance, 0.0)))
            if not np.isfinite(scale) or scale <= 0.0:
                scale = 1.0
            values[feature, : len(centers)] = (centers - mean) / scale
        self.linear_bin_values_ = values
        self.linear_leaves_active_ = True
        self.linear_leaves_inactive_reason_ = None

    def _linear_leaf_metadata(self):
        return {
            "requested": bool(self.linear_leaves),
            "active": bool(getattr(self, "linear_leaves_active_", False)),
            "inactive_reason": getattr(
                self, "linear_leaves_inactive_reason_", "disabled"
            ),
            "min_samples": LINEAR_LEAVES_MIN_SAMPLES,
            "linear_lambda": float(self.linear_lambda),
            "numeric_feature_count": int(
                np.count_nonzero(
                    getattr(self, "linear_numeric_features_", np.empty(0))
                )
            ),
            "linear_tree_count": int(
                getattr(self, "linear_tree_count_", 0)
            ),
            "linear_leaf_count": int(
                getattr(self, "linear_leaf_count_", 0)
            ),
        }

    def _record_linear_leaf_metadata(self):
        if hasattr(self, "auto_params_"):
            metadata = self._linear_leaf_metadata()
            self.auto_params_["linear_leaves"] = metadata
            self.auto_params_.setdefault("diagnostics", {})
            self.auto_params_["diagnostics"]["linear_leaves"] = metadata

    def _finalize_linear_leaf_metadata(self):
        linear_trees = [
            tree
            for tree in self._iter_tree_objects()
            if getattr(tree, "linear_coefficients", None) is not None
        ]
        self.linear_tree_count_ = len(linear_trees)
        self.linear_leaf_count_ = int(
            sum(tree.linear_coefficients.shape[0] for tree in linear_trees)
        )
        if self.linear_leaves_active_ and not linear_trees:
            self.linear_leaves_active_ = False
            self.linear_leaves_inactive_reason_ = "no_retained_linear_trees"
        self._record_linear_leaf_metadata()

    def _include_cat_codes(self):
        return self.tree_mode_ in {"lightgbm", "hybrid"}

    def _tree_builder(self):
        if self.tree_mode_ == "lightgbm":
            return build_leafwise_tree
        if self.tree_mode_ == "hybrid":
            return build_hybrid_tree
        if self.tree_mode_ == "depthwise":
            return build_levelwise_tree
        return build_oblivious_tree

    def _maybe_subsample(self, grad, hess, rng):
        """Return zeroed gradients plus sampled row indices for one tree.

        Unsampled rows keep zero grad/hess so ordered boosting updates preserve
        the old fallback behavior, while histogram builders can skip them.
        """
        if self.sampling_ == "goss":
            g, h, row_indices = self._goss_subsample(grad, hess, rng)
            self._record_sampling_diagnostic(row_indices, grad.shape[0])
            return g, h, row_indices
        if self.sampling_ == "weighted_goss":
            g, h, row_indices = self._weighted_goss_subsample(grad, hess, rng)
            self._record_sampling_diagnostic(row_indices, grad.shape[0])
            return g, h, row_indices
        if self.sampling_ == "mvs":
            g, h, row_indices = self._mvs_subsample(grad, hess, rng)
            self._record_sampling_diagnostic(row_indices, grad.shape[0])
            return g, h, row_indices
        if self.subsample >= 1.0:
            self._record_sampling_diagnostic(None, grad.shape[0])
            return grad, hess, None
        mask = rng.random(grad.shape[0]) < self.subsample
        if not np.any(mask):
            importance = np.abs(grad) + np.maximum(hess, 0.0)
            if (
                not np.any(np.isfinite(importance))
                or float(np.sum(importance)) <= 0.0
            ):
                chosen = int(rng.integers(0, grad.shape[0]))
            else:
                chosen = int(np.nanargmax(importance))
            mask[chosen] = True
        row_indices = np.flatnonzero(mask).astype(np.int64)
        self._record_sampling_diagnostic(row_indices, grad.shape[0])
        return np.where(mask, grad, 0.0), np.where(mask, hess, 0.0), row_indices

    def _mvs_probabilities(self, importance):
        n_samples = importance.shape[0]
        if n_samples <= 1 or self.subsample >= 1.0:
            return None
        target = float(np.clip(self.subsample, 0.0, 1.0)) * n_samples
        if target <= 0.0:
            return np.zeros(n_samples, dtype=np.float64)
        if target >= n_samples:
            return None
        importance = np.asarray(importance, dtype=np.float64)
        if (
            not np.all(np.isfinite(importance))
            or float(np.sum(importance)) <= 0.0
            or np.all(importance <= 0.0)
        ):
            return np.full(n_samples, target / n_samples, dtype=np.float64)
        return _exact_mvs_probabilities(importance, target)

    def _mvs_subsample(self, grad, hess, rng):
        n_samples = grad.shape[0]
        probs = self._mvs_probabilities(
            np.sqrt(grad * grad + self.mvs_reg * hess * hess)
        )
        if probs is None:
            return grad, hess, None
        mask = rng.random(n_samples) < probs
        if not np.any(mask):
            mask[int(np.argmax(probs))] = True
        row_indices = np.flatnonzero(mask).astype(np.int64)
        scale = np.zeros(n_samples, dtype=np.float64)
        scale[row_indices] = 1.0 / np.maximum(probs[row_indices], 1e-300)
        return grad * scale, hess * scale, row_indices

    def _mvs_subsample_multiclass(self, grad, hess, rng):
        n_samples = grad.shape[1]
        importance = np.sqrt(
            np.sum(grad * grad, axis=0) + self.mvs_reg * np.sum(hess * hess, axis=0)
        )
        probs = self._mvs_probabilities(importance)
        if probs is None:
            return grad, hess, None
        mask = rng.random(n_samples) < probs
        if not np.any(mask):
            mask[int(np.argmax(probs))] = True
        row_indices = np.flatnonzero(mask).astype(np.int64)
        scale = np.zeros(n_samples, dtype=np.float64)
        scale[row_indices] = 1.0 / np.maximum(probs[row_indices], 1e-300)
        return grad * scale[None, :], hess * scale[None, :], row_indices

    def _weighted_goss_probabilities(self, mass, target_mass):
        mass = np.asarray(mass, dtype=np.float64)
        if (
            mass.size == 0
            or target_mass <= 0.0
            or not np.all(np.isfinite(mass))
            or float(np.sum(mass)) <= 0.0
        ):
            return None
        target_mass = min(float(target_mass), float(np.sum(mass)))
        return _exact_weighted_goss_probabilities(mass, target_mass)

    def _weighted_goss_subsample_from_score(self, grad, hess, score, mass, rng):
        n_samples = score.shape[0]
        if n_samples <= 1:
            return grad, hess, None
        mass = np.asarray(mass, dtype=np.float64)
        if (
            not np.all(np.isfinite(mass))
            or float(np.sum(mass)) <= 0.0
            or np.all(mass <= 0.0)
        ):
            mass = np.ones(n_samples, dtype=np.float64)
        else:
            mass = np.maximum(mass, 0.0)
        total_mass = float(np.sum(mass))
        if mass[0] > 0.0 and np.all(mass == mass[0]):
            return self._weighted_goss_uniform_mass_subsample(
                grad, hess, score, rng
            )
        top_target = self.top_rate * total_mass
        top_idx = self._weighted_goss_top_indices(score, mass, top_target)
        remaining_mask = np.ones(n_samples, dtype=bool)
        remaining_mask[top_idx] = False
        remaining_idx = np.flatnonzero(remaining_mask)
        if remaining_idx.size == 0:
            return grad, hess, None

        target_other_mass = self.other_rate * total_mass
        probs = self._weighted_goss_probabilities(
            mass[remaining_idx], target_other_mass
        )
        if probs is None:
            probs = np.full(
                remaining_idx.shape[0],
                min(1.0, self.other_rate * n_samples / remaining_idx.shape[0]),
                dtype=np.float64,
            )
        other_mask = rng.random(remaining_idx.shape[0]) < probs
        other_idx = remaining_idx[other_mask]

        if top_idx.shape[0] + other_idx.shape[0] == n_samples:
            return grad, hess, None

        scale = np.zeros(n_samples, dtype=np.float64)
        scale[top_idx] = 1.0
        scale[other_idx] = 1.0 / np.maximum(probs[other_mask], 1e-300)
        row_indices = np.sort(
            np.concatenate((top_idx.astype(np.int64), other_idx.astype(np.int64)))
        )
        if grad.ndim == 1:
            return grad * scale, hess * scale, row_indices
        return grad * scale[None, :], hess * scale[None, :], row_indices

    def _weighted_goss_top_indices(self, score, mass, target_mass):
        """Return highest-score rows whose cumulative sample mass hits target."""
        n_samples = score.shape[0]
        if n_samples <= 1:
            return np.arange(n_samples, dtype=np.int64)
        target_mass = min(max(float(target_mass), 0.0), float(np.sum(mass)))
        mean_mass = float(np.mean(mass))
        if target_mass <= 0.0 or mean_mass <= 0.0:
            top_count = 1
        else:
            top_count = int(np.ceil(target_mass / mean_mass))
            top_count = min(n_samples, max(1, top_count))

        while top_count < n_samples:
            candidate = np.argpartition(score, n_samples - top_count)[
                n_samples - top_count:
            ]
            if float(np.sum(mass[candidate])) >= target_mass:
                break
            top_count = min(n_samples, max(top_count + 1, top_count * 2))
        else:
            candidate = np.arange(n_samples, dtype=np.int64)

        candidate_order = np.argsort(score[candidate])[::-1]
        ordered = candidate[candidate_order]
        cum_mass = np.cumsum(mass[ordered])
        selected_count = min(
            ordered.shape[0],
            max(1, int(np.searchsorted(cum_mass, target_mass, side="left") + 1)),
        )
        return ordered[:selected_count].astype(np.int64, copy=False)

    def _weighted_goss_uniform_mass_subsample(self, grad, hess, score, rng):
        n_samples = score.shape[0]
        top_count = min(
            n_samples,
            max(1, int(np.ceil(self.top_rate * n_samples))),
        )
        remaining_count = n_samples - top_count
        if remaining_count <= 0:
            return grad, hess, None

        top_idx = np.argpartition(score, n_samples - top_count)[-top_count:]
        remaining_mask = np.ones(n_samples, dtype=bool)
        remaining_mask[top_idx] = False
        remaining_idx = np.flatnonzero(remaining_mask)
        prob = min(1.0, self.other_rate * n_samples / remaining_idx.shape[0])
        if prob >= 1.0:
            other_idx = remaining_idx
        else:
            other_mask = rng.random(remaining_idx.shape[0]) < prob
            other_idx = remaining_idx[other_mask]

        if top_idx.shape[0] + other_idx.shape[0] == n_samples:
            return grad, hess, None

        scale = np.zeros(n_samples, dtype=np.float64)
        scale[top_idx] = 1.0
        scale[other_idx] = 1.0 / max(prob, 1e-300)
        row_indices = np.sort(
            np.concatenate((top_idx.astype(np.int64), other_idx.astype(np.int64)))
        )
        if grad.ndim == 1:
            return grad * scale, hess * scale, row_indices
        return grad * scale[None, :], hess * scale[None, :], row_indices

    def _weighted_goss_subsample(self, grad, hess, rng):
        score = np.abs(grad)
        mass = np.maximum(hess, 0.0)
        return self._weighted_goss_subsample_from_score(
            grad, hess, score, mass, rng
        )

    def _weighted_goss_subsample_multiclass(self, grad, hess, rng):
        score = np.abs(grad).sum(axis=0)
        mass = np.maximum(hess, 0.0).sum(axis=0)
        return self._weighted_goss_subsample_from_score(
            grad, hess, score, mass, rng
        )

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
                        X_route_binned,
                        use_constant_hessian, hessian_always_positive=False,
                        rowpar_buffers=None, tree_iteration=0):
        kwargs = {
            "feature_mask": fmask,
            "min_child_weight": self.min_child_weight,
            "hist_buffers": hist_buffers,
            "split_buffers": split_buffers,
            "return_training_state": True,
            "X_hist_binned": X_hist_binned,
            "X_route_binned": X_route_binned,
            "feature_indices": findices,
            "row_indices": row_indices,
            "constant_hessian": use_constant_hessian,
            "rowpar_buffers": rowpar_buffers,
            "random_strength": self.random_strength,
            "split_seed": int(getattr(self, "_split_seed_", 0)),
            "tree_iteration": int(tree_iteration),
            "leaf_dtype": self.leaf_dtype_,
        }
        if self.tree_mode_ in {"lightgbm", "hybrid"}:
            kwargs.update(
                max_leaves=self._max_tree_leaves(),
                min_child_samples=self.min_child_samples,
                min_gain_to_split=self.min_gain_to_split,
                hessian_always_positive=hessian_always_positive,
                fused_changed_leaf_scoring=(
                    hessian_always_positive
                    and not use_constant_hessian
                    and self.random_strength <= 0.0
                    and fmask is None
                    and row_indices is None
                    and findices is None
                    and rowpar_buffers is None
                    and getattr(self, "n_threads_", 1) > 2
                ),
            )
        return kwargs

    def _scalar_histogram_streams(
        self, grad, hess, grad_stream_buffer, hess_stream_buffer
    ):
        if self.histogram_dtype_ == "float64":
            return grad, hess
        grad_stream_buffer[:] = grad
        hess_stream_buffer[:] = hess
        return grad_stream_buffer, hess_stream_buffer

    def _refresh_scalar_leaf_values_float64(self, tree, leaf, grad, hess,
                                            row_indices):
        n_leaves = tree.values.shape[0]
        if row_indices is None:
            values, leaf_G, leaf_H = _leaf_values_and_sums(
                leaf, grad, hess, n_leaves, self.l2_leaf_reg, self.lr_
            )
        else:
            values, leaf_G, leaf_H = _leaf_values_and_sums_rows(
                leaf, grad, hess, row_indices, n_leaves, self.l2_leaf_reg,
                self.lr_
            )
        tree.values = values
        return leaf_G, leaf_H

    def _accumulate_importance(self, tree):
        """Add this tree's per-split gains to the running importance totals,
        mapped from internal columns back to original input features."""
        for f, g in zip(tree.splits_feat, tree.gains):
            orig = self.prep_.feature_map_[f]
            self._importance[orig] += g

    def _iter_tree_objects(self):
        for item in self.trees_:
            if isinstance(item, (list, tuple)):
                yield from item
            else:
                yield item

    def _rebuild_importance_from_trees(self):
        self._importance = np.zeros(self.prep_.n_input_features_)
        for tree in self._iter_tree_objects():
            if getattr(tree, "depth", 0) > 0:
                self._accumulate_importance(tree)

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
        return (
            self.tree_mode_ in {"lightgbm", "hybrid"}
            and self.loss_name in {"Logloss", "RMSE"}
        )

    def fit(self, X, y, cat_features=None, eval_set=None, sample_weight=None,
            eval_sample_weight=None, callbacks=None):
        """Fit the additive model. Optionally pass `cat_features` (column indices
        to target-encode) and `eval_set=(X_val, y_val)` for early stopping.
        `sample_weight` is a 1-D array of per-sample weights; None means uniform.
        Weights are normalized to mean 1 internally so the gradient scale stays
        comparable to the no-weight case."""
        callbacks = _normalize_callbacks(callbacks)
        X, cat_features, n_features, feature_names = self._coerce_fit_X(
            X, cat_features
        )
        n_samples = X.shape[0]
        y = validate_target_vector(
            y, n_samples, dtype=np.float64
        )
        _reject_eval_sample_weight_without_eval_set(eval_sample_weight, eval_set)
        Xv = yv = wv = None
        if eval_set is not None:
            Xv, yv = eval_set
            Xv = self._coerce_eval_X(
                Xv,
                cat_features,
                expected_n_features=n_features,
                expected_feature_names=feature_names,
            )
            yv = validate_target_vector(
                yv, Xv.shape[0], name="eval_set[1]", dtype=np.float64
            )
            wv = _validate_sample_weight(
                eval_sample_weight, len(yv), name="eval_sample_weight"
            )

        # Normalize weights to mean=1. np.ones(n) stays np.ones(n), so
        # sample_weight=np.ones(n) is bitwise-equivalent to sample_weight=None
        # for all losses except MAE/Quantile (which use a different quantile
        # algorithm when weights are present).
        self.tree_mode_ = _normalize_tree_mode(self.tree_mode)
        fit_random_state = normalize_random_state_seed(self.random_state)
        self._fit_random_state_seed_ = fit_random_state
        self.n_threads_ = _fit_thread_count(
            self.thread_count, self.tree_mode_, n_samples
        )
        self.histogram_dtype_ = _normalize_histogram_dtype(self.histogram_dtype)
        self.leaf_dtype_ = _normalize_leaf_dtype_name(self.leaf_dtype)
        self._validate_sampling_config()
        self.ordered_boosting_ = self._resolve_ordered_boosting(
            loss_name=self.loss_name,
        )
        w = _validate_sample_weight(sample_weight, n_samples)
        self.loss_ = LOSSES[self.loss_name](**self.loss_kwargs)
        self._resolve_auto_structure_params(
            loss_name=self.loss_name,
            n_samples=n_samples,
            sample_weight=w,
            X=X,
            cat_features=cat_features,
        )
        use_constant_hessian = (
            getattr(self.loss_, "constant_hessian", False)
            and w is None
            and self.sampling_ not in {"goss", "weighted_goss"}
            and not self._mvs_active()
            and not self._bayesian_bootstrap_active()
        )
        hessian_always_positive = (
            self.tree_mode_ in {"lightgbm", "hybrid"}
            and self.loss_name == "Logloss"
            and w is None
            and self.sampling_ not in {"goss", "weighted_goss"}
            and not self._mvs_active()
            and not self._bayesian_bootstrap_active()
        )
        timing = _new_timing(self.verbose_timing)
        self.timing_ = timing
        phase = _start_timing(timing)
        preprocessing_target = y
        if hasattr(self.loss_, "preprocessing_target"):
            preprocessing_target = self.loss_.preprocessing_target(y)
        X_binned = self._fit_transform_preprocessor(
            X, [preprocessing_target], cat_features, w,
        )
        self._shap_background_ = None
        if self.tree_mode_ == "catboost":
            shap_background_size = min(n_samples, SHAP_BACKGROUND_SIZE)
            shap_seed = 0 if fit_random_state is None else fit_random_state
            shap_indices = np.random.default_rng(shap_seed).choice(
                n_samples, shap_background_size, replace=False
            )
            self._shap_background_ = np.ascontiguousarray(
                X_binned[shap_indices]
            )
        X_route_binned = np.asfortranarray(X_binned)
        X_hist_binned = (
            X_route_binned if self.n_threads_ > 1 else X_binned
        )
        n_bins = self.prep_.n_bins_
        self._prepare_linear_leaf_state(
            X_binned, self.loss_name, n_samples, w
        )
        self._resolve_fit_auto_params(
            loss_name=self.loss_name,
            n_samples=n_samples,
            sample_weight=w,
            eval_set_present=eval_set is not None,
            p_model=X_binned.shape[1],
        )
        hist_buffers = self._alloc_hist_buffers(X_binned.shape[1], n_bins)
        split_buffers = (
            self._alloc_split_buffers(X_binned.shape[1])
            if self.n_threads_ > 1 else None
        )
        rowpar_buffers = self._alloc_rowpar_buffers(
            X_binned.shape[1], n_bins, n_samples
        )
        self._importance = np.zeros(self.prep_.n_input_features_)

        Xv_binned = Fv = None
        if Xv is not None:
            Xv_binned = self.prep_.transform(Xv)
        _add_timing(timing, "preprocess", phase)

        self.init_ = self.loss_.init(y, w)
        self._record_scalar_target_stats(y, w)
        F = np.full(n_samples, self.init_, dtype=np.float64)
        grad_buffer = np.empty(n_samples, dtype=np.float64)
        hess_buffer = np.empty(n_samples, dtype=np.float64)
        grad_stream_buffer = hess_stream_buffer = None
        if self.histogram_dtype_ == "float32":
            grad_stream_buffer = np.empty(n_samples, dtype=np.float32)
            hess_stream_buffer = np.empty(n_samples, dtype=np.float32)
        if yv is not None:
            Fv = np.full(len(yv), self.init_)
        baseline_loss = (
            self.loss_.eval(yv, Fv, wv)
            if Fv is not None
            else self.loss_.eval(y, F, w)
        )
        self._finalize_early_stopping_min_delta(baseline_loss, self.loss_name)
        self.auto_params_ = self._resolved_auto_params(
            n_samples=n_samples,
            n_raw_features=X.shape[1],
            X_binned=X_binned,
            n_bins=n_bins,
            sample_weight=w,
            eval_set_present=eval_set is not None,
            eval_n_samples=0 if yv is None else len(yv),
            eval_sample_weight=wv,
            rowpar_buffers=rowpar_buffers,
        )
        self._record_linear_leaf_metadata()
        self._emit_auto_param_warnings()
        self._reset_stochastic_diagnostics()

        rng = np.random.default_rng(fit_random_state)
        self._initialize_split_seed(rng, fit_random_state)
        self.trees_ = []
        self._flat_cache_ = None  # drop any previous fit's flattened ensemble
        self.train_history_, self.valid_history_ = [], []
        # Train loss is diagnostic only (early stopping watches the eval set),
        # so skipping it when nobody will read it saves one full O(n) pass per
        # round. verbose forces it back on because the progress log prints it.
        eval_train = self.eval_train_loss or bool(self.verbose)
        patience_score, patience_iter = np.inf, 0
        best_prefix_score, best_prefix_iter = np.inf, 0
        sampled_depth0_retries = 0
        iterations_attempted = 0
        stop_reason = "iteration_limit"
        t0 = time.perf_counter()

        for m in range(self.iterations_):
            callback_reason = self._callback_stop_reason(
                callbacks, m, iterations_attempted
            )
            if callback_reason is not None:
                stop_reason = callback_reason
                break
            iterations_attempted = m + 1
            phase = _start_timing(timing)
            if hasattr(self.loss_, "grad_hess_into"):
                self.loss_.grad_hess_into(y, F, w, grad_buffer, hess_buffer)
                grad, hess = grad_buffer, hess_buffer
            else:
                grad, hess = self.loss_.grad_hess(y, F)
                if w is not None:
                    grad = grad * w
                    hess = hess * w
            bootstrap_factors = self._bayesian_bootstrap_factors(n_samples, rng)
            grad, hess = self._apply_bootstrap(grad, hess, bootstrap_factors)
            g, h, row_indices = self._maybe_subsample(grad, hess, rng)
            fmask, findices = self._feature_selection(X_binned.shape[1], rng)
            _add_timing(timing, "grad_hess", phase)

            phase = _start_timing(timing)
            build_g, build_h = self._scalar_histogram_streams(
                g, h, grad_stream_buffer, hess_stream_buffer
            )
            tree, leaf, leaf_G, leaf_H = self._tree_builder()(
                X_binned, build_g, build_h, n_bins, self._max_tree_depth(),
                self.l2_leaf_reg, self.lr_,
                **self._builder_kwargs(
                    fmask, findices, row_indices, hist_buffers, split_buffers,
                    X_hist_binned, X_route_binned, use_constant_hessian,
                    hessian_always_positive=hessian_always_positive,
                    rowpar_buffers=rowpar_buffers,
                    tree_iteration=m,
                ),
            )
            # A depth-0 tree found no legal split; subsequent rounds on the same
            # gradients would too, so stop rather than append empty trees.
            if tree.depth == 0:
                _add_timing(timing, "tree_build", phase)
                if (
                    row_indices is not None
                    and m + 1 < self.iterations_
                    and sampled_depth0_retries
                    < _MAX_CONSECUTIVE_SAMPLED_DEPTH0_RETRIES
                ):
                    sampled_depth0_retries += 1
                    if self.verbose:
                        print(
                            f"No split at sampled iteration {m}; retrying "
                            "with a new row sample."
                        )
                    continue
                if self.verbose:
                    print(f"No further splits at iteration {m}; stopping.")
                stop_reason = "no_split"
                break
            sampled_depth0_retries = 0
            if self.histogram_dtype_ == "float32":
                leaf_G, leaf_H = self._refresh_scalar_leaf_values_float64(
                    tree, leaf, g, h, row_indices
                )
            if self.linear_leaves_active_:
                attach_oblivious_linear_leaves(
                    tree,
                    leaf,
                    g,
                    h,
                    X_binned,
                    self.linear_bin_values_,
                    self.linear_numeric_features_,
                    self.l2_leaf_reg,
                    self.linear_lambda,
                    self.lr_,
                    row_indices=row_indices,
                )
            _add_timing(timing, "tree_build", phase)
            phase = _start_timing(timing)
            if getattr(self.loss_, "adjusts_leaves", False):
                correction_weight = self._correction_weight(w, bootstrap_factors)
                self._correct_leaves(
                    tree, X_binned, y - F, correction_weight, leaf=leaf
                )
            self.trees_.append(tree)
            self._accumulate_importance(tree)
            if getattr(tree, "linear_coefficients", None) is not None:
                add_linear_leaf_values_inplace(
                    leaf,
                    tree.linear_features,
                    tree.linear_coefficients,
                    tree.linear_bin_values,
                    X_binned,
                    F,
                )
            elif self.ordered_boosting_ and not getattr(self.loss_, "adjusts_leaves", False):
                # Leave-one-out leaf step: each row's update uses its leaf's
                # gradient/hessian totals with that row's own contribution
                # removed, reducing the self-reinforcement of plain boosting.
                # tree.values keeps the standard Newton values for inference;
                # only the training F uses this corrected update. Subsampled-out
                # rows (g=h=0) fall back to the standard leaf value.
                ordered_leaf_update_inplace(
                    leaf, leaf_G, leaf_H, g, h, self.lr_, self.l2_leaf_reg, F
                )
            else:
                add_leaf_values_inplace(leaf, tree.values, F)
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
                successful_iter = len(self.trees_) - 1
                if val < best_prefix_score:
                    best_prefix_score, best_prefix_iter = val, successful_iter
                if patience_score - val > self.early_stopping_min_delta_:
                    patience_score, patience_iter = val, successful_iter
                elif (self.early_stopping_rounds_ and
                      successful_iter - patience_iter >= self.early_stopping_rounds_):
                    if self.verbose:
                        print(f"Early stop at {m} (best {best_prefix_iter}, "
                              f"val {best_prefix_score:.5f})")
                    stop_reason = "early_stopping"
                    break

            if self.verbose and (m % max(1, self.iterations_ // 10) == 0):
                msg = f"[{m}] train {self.train_history_[-1]:.5f}"
                if Fv is not None:
                    msg += f"  val {self.valid_history_[-1]:.5f}"
                print(msg)

        rounds_completed = len(self.trees_)
        self.fit_time_ = time.perf_counter() - t0
        self._truncate_to_best_model(best_prefix_iter, self.valid_history_)
        self._finalize_linear_leaf_metadata()
        self._refresh_stochastic_auto_params(n_samples)
        self.best_iteration_ = len(self.trees_)
        self._finalize_training_metadata(
            stop_reason=stop_reason,
            iterations_attempted=iterations_attempted,
            rounds_completed=rounds_completed,
            best_prefix_iter=best_prefix_iter,
            callbacks=callbacks,
        )
        if self.valid_history_:
            self.best_score_ = best_prefix_score
        elif Fv is not None:
            self.best_score_ = self.loss_.eval(yv, Fv, wv)
        elif self.train_history_:
            self.best_score_ = self.train_history_[-1]
        else:
            self.best_score_ = self.loss_.eval(y, F, w)
        self._record_input_feature_metadata(n_features, feature_names)
        return self

    def _correction_weight(self, sample_weight, bootstrap_factors):
        if bootstrap_factors is None:
            return sample_weight
        if sample_weight is None:
            return bootstrap_factors
        return sample_weight * bootstrap_factors

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

    def predict_raw(self, X, *, _validated=False):
        """Return raw additive scores (pre-link): the regression prediction, or
        the log-odds for binary classification."""
        X = self._prepare_predict_X(X, validated=_validated)
        X_binned = self.prep_.transform(X)
        F = np.full(X_binned.shape[0], self.init_, dtype=np.float64)
        flat = self._flat_ensemble()
        if flat is not None and flat_predict_preferred(
            flat, X_binned.shape[0], self.tree_mode_
        ):
            if isinstance(flat, FlatNonObliviousEnsemble):
                flat.add_predict_scalar_packed(X_binned, F)
            else:
                flat.add_predict(X_binned, F)
        else:
            for tree in self.trees_:
                tree.add_predict(X_binned, F)
        return F

    def staged_predict_raw(self, X, *, _validated=False):
        """Yield the cumulative raw prediction after each tree (1..n_trees)."""
        X = self._prepare_predict_X(X, validated=_validated)
        X_binned = self.prep_.transform(X)
        F = np.full(X_binned.shape[0], self.init_, dtype=np.float64)
        for stage, tree in enumerate(self.trees_):
            if stage:
                self._restore_thread_count()
            tree.add_predict(X_binned, F)
            yield F.copy()

    def shap_values(
        self,
        X,
        background=None,
        max_background=SHAP_BACKGROUND_SIZE,
        random_state=0,
        _validated=False,
        _background_validated=False,
    ):
        """Return exact interventional TreeSHAP in raw-score space.

        Returns ``(contributions, expected_value)``. Contributions are in the
        original input-feature space and include local-linear leaf slopes.
        Their row sums plus ``expected_value`` equal :meth:`predict_raw` to
        floating-point tolerance.
        """
        max_background = normalize_max_background(max_background)
        X = self._prepare_predict_X(X, validated=_validated)
        X_binned = self.prep_.transform(X)
        if self.tree_mode_ != "catboost":
            raise NotImplementedError(
                "TreeSHAP currently supports only oblivious trees"
            )
        n_original_features = int(self.prep_.n_input_features_)
        packed = None
        if self.trees_:
            packed = pack_oblivious_shap_forest(self.trees_)
            player_count = max_original_players(
                self.trees_, self.prep_.feature_map_
            )
            if player_count > SHAP_MAX_PLAYERS:
                raise NotImplementedError(
                    "TreeSHAP supports at most "
                    f"{SHAP_MAX_PLAYERS} distinct features per tree"
                )
        if background is None:
            background_binned = getattr(self, "_shap_background_", None)
            if background_binned is None:
                raise ValueError(
                    "this model has no stored SHAP background; pass "
                    "background explicitly"
                )
        else:
            background = self._prepare_predict_X(
                background, validated=_background_validated
            )
            background_binned = self.prep_.transform(background)
        if background_binned.shape[0] == 0:
            raise ValueError("SHAP background must contain at least one row")
        if background_binned.shape[0] > max_background:
            seed = normalize_random_state_seed(
                random_state, name="random_state"
            )
            selected = np.random.default_rng(seed).choice(
                background_binned.shape[0],
                max_background,
                replace=False,
            )
            background_binned = np.ascontiguousarray(
                background_binned[selected]
            )
        else:
            background_binned = np.ascontiguousarray(background_binned)

        if not self.trees_:
            return (
                np.zeros(
                    (X_binned.shape[0], n_original_features),
                    dtype=np.float64,
                ),
                float(self.init_),
            )
        contributions = shap_forest_linear(
            np.ascontiguousarray(X_binned),
            background_binned,
            *packed,
            np.asarray(self.prep_.feature_map_, dtype=np.int64),
            n_original_features,
            factorials(player_count),
        )
        background_prediction = np.full(
            background_binned.shape[0], self.init_, dtype=np.float64
        )
        flat = self._flat_ensemble()
        if flat is None:
            raise RuntimeError("failed to pack the fitted SHAP forest")
        flat.add_predict(background_binned, background_prediction)
        return contributions, float(background_prediction.mean())


class MulticlassBoosting(_BaseBooster):
    """Softmax multiclass booster: fits K trees per round (one per class)."""

    def fit(self, X, y, cat_features=None, eval_set=None, sample_weight=None,
            eval_sample_weight=None, callbacks=None):
        """Fit K trees per boosting round (one per class) under softmax loss.
        Same `cat_features` / `eval_set` / `sample_weight` semantics as the
        scalar booster."""
        if self.linear_leaves:
            raise ValueError(
                "linear_leaves=True is currently supported only for "
                "scalar RMSE regression"
            )
        callbacks = _normalize_callbacks(callbacks)
        X, cat_features, n_features, feature_names = self._coerce_fit_X(
            X, cat_features
        )
        y = validate_target_vector(y, X.shape[0])
        classes = np.unique(y)
        K = classes.size
        self.histogram_dtype_ = _normalize_histogram_dtype(self.histogram_dtype)
        self.leaf_dtype_ = _normalize_leaf_dtype_name(self.leaf_dtype)
        if self.histogram_dtype_ != "float64":
            raise ValueError(
                "histogram_dtype='float32' is currently only supported for "
                "scalar GradientBoosting fits; multiclass support lands with "
                "the R6 shared-vector layout"
            )
        y_idx = np.searchsorted(classes, y)
        Y_class = _one_hot_class_major(y_idx, K)  # class-major (K, n)
        n_samples = X.shape[0]
        _reject_eval_sample_weight_without_eval_set(eval_sample_weight, eval_set)
        Xv = yv_idx = wv = None
        if eval_set is not None:
            Xv, yv = eval_set
            Xv = self._coerce_eval_X(
                Xv,
                cat_features,
                expected_n_features=n_features,
                expected_feature_names=feature_names,
            )
            yv_arr = validate_target_vector(
                yv, Xv.shape[0], name="eval_set[1]"
            )
            if np.any(~np.isin(yv_arr, classes)):
                raise ValueError(
                    "eval_set contains labels not present in training data"
                )
            yv_idx = np.searchsorted(classes, yv_arr)
            wv = _validate_sample_weight(
                eval_sample_weight, len(yv_idx), name="eval_sample_weight"
            )

        self.tree_mode_ = _normalize_tree_mode(self.tree_mode)
        fit_random_state = normalize_random_state_seed(self.random_state)
        self._fit_random_state_seed_ = fit_random_state
        self.n_threads_ = _fit_thread_count(
            self.thread_count, self.tree_mode_, n_samples
        )
        self._validate_sampling_config()
        self.ordered_boosting_ = self._resolve_ordered_boosting(
            loss_name="MultiClass",
        )
        w = _validate_sample_weight(sample_weight, n_samples)
        self.classes_ = classes
        self.n_classes_ = K
        self.loss_ = MultiSoftmax(K)
        self._resolve_auto_structure_params(
            loss_name="MultiClass",
            n_samples=n_samples,
            sample_weight=w,
            X=X,
            cat_features=cat_features,
        )
        self.l2_leaf_reg_ = self.l2_leaf_reg
        strategy = self.multiclass_tree_strategy
        if strategy not in {"auto", "per_class", "shared_vector"}:
            raise ValueError(
                "multiclass_tree_strategy must be 'auto', 'per_class', "
                "or 'shared_vector'"
            )
        can_use_shared_lightgbm_multiclass = (
            self.tree_mode_ == "lightgbm"
            and not self._row_sampling_active()
            and not self._bayesian_bootstrap_active()
            and self.colsample >= 1.0
            and not self.ordered_boosting_
        )
        if strategy == "shared_vector" and not can_use_shared_lightgbm_multiclass:
            raise ValueError(
                "multiclass_tree_strategy='shared_vector' requires "
                "tree_mode='lightgbm', no ordered boosting, and full row/column sampling"
            )

        timing = _new_timing(self.verbose_timing)
        self.timing_ = timing
        phase = _start_timing(timing)
        # One ordered-TS target per class (CatBoost-style per-class statistics).
        X_binned = self._fit_transform_preprocessor(
            X, [Y_class[k] for k in range(K)], cat_features, w,
        )
        X_route_binned = np.asfortranarray(X_binned)
        X_hist_binned = (
            X_route_binned if self.n_threads_ > 1 else X_binned
        )
        n_bins = self.prep_.n_bins_
        self._resolve_fit_auto_params(
            loss_name="MultiClass",
            n_samples=n_samples,
            sample_weight=w,
            eval_set_present=eval_set is not None,
            p_model=X_binned.shape[1],
        )
        hist_buffers = self._alloc_hist_buffers(X_binned.shape[1], n_bins)
        split_buffers = (
            self._alloc_split_buffers(X_binned.shape[1])
            if self.n_threads_ > 1 else None
        )
        rowpar_buffers = self._alloc_rowpar_buffers(
            X_binned.shape[1], n_bins, n_samples
        )
        use_shared_lightgbm_multiclass = (
            can_use_shared_lightgbm_multiclass
            and strategy in {"auto", "shared_vector"}
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
            and not self._row_sampling_active()
            and not self._bayesian_bootstrap_active()
            and self.colsample >= 1.0
        )
        if use_fused_root:
            max_bins_ = int(n_bins.max()) if len(n_bins) else 1
            root_g = np.zeros((K, X_binned.shape[1], 1, max_bins_))
            root_h = np.zeros((K, X_binned.shape[1], 1, max_bins_))
            root_c = np.zeros((X_binned.shape[1], 1, max_bins_))
            root_leaf = np.zeros(n_samples, dtype=np.int64)
        self._importance = np.zeros(self.prep_.n_input_features_)

        Xv_binned = Fv = None
        if Xv is not None:
            Xv_binned = self.prep_.transform(Xv)
        _add_timing(timing, "preprocess", phase)

        self.init_ = self.loss_.init_class_major(Y_class, w)  # (K,)
        self._record_classification_target_stats(y_idx, self.classes_, w)
        F = np.tile(self.init_[:, None], (1, n_samples))  # class-major (K, n)
        grad_buffer = np.empty_like(F)
        hess_buffer = np.empty_like(F)
        grad_shared_row_major = hess_shared_row_major = None
        if use_shared_lightgbm_multiclass:
            grad_shared_row_major = np.empty((n_samples, K), dtype=np.float64)
            hess_shared_row_major = np.empty((n_samples, K), dtype=np.float64)
        if yv_idx is not None:
            Fv = np.tile(self.init_[:, None], (1, len(yv_idx)))
        baseline_loss = (
            self.loss_.eval_class_major_labels(yv_idx, Fv, wv)
            if Fv is not None
            else self.loss_.eval_class_major_labels(y_idx, F, w)
        )
        self._finalize_early_stopping_min_delta(baseline_loss, "MultiClass")
        self.auto_params_ = self._resolved_auto_params(
            n_samples=n_samples,
            n_raw_features=X.shape[1],
            X_binned=X_binned,
            n_bins=n_bins,
            sample_weight=w,
            eval_set_present=eval_set is not None,
            eval_n_samples=0 if yv_idx is None else len(yv_idx),
            eval_sample_weight=wv,
            rowpar_buffers=rowpar_buffers,
            extra={
                "multiclass": {
                    "n_classes": int(K),
                    "tree_strategy": self.multiclass_tree_strategy_,
                    "shared_lightgbm_multiclass": bool(use_shared_lightgbm_multiclass),
                    "fused_root_histograms": bool(use_fused_root),
                }
            },
        )
        self._emit_auto_param_warnings()
        self._reset_stochastic_diagnostics()

        rng = np.random.default_rng(fit_random_state)
        self._initialize_split_seed(rng, fit_random_state)
        self.trees_ = []                           # list of rounds; each = K trees
        self._flat_cache_ = None  # drop any previous fit's flattened ensemble
        self.train_history_, self.valid_history_ = [], []
        eval_train = self.eval_train_loss or bool(self.verbose)
        patience_score, patience_iter = np.inf, 0
        best_prefix_score, best_prefix_iter = np.inf, 0
        sampled_depth0_retries = 0
        iterations_attempted = 0
        stop_reason = "iteration_limit"
        t0 = time.perf_counter()

        for m in range(self.iterations_):
            callback_reason = self._callback_stop_reason(
                callbacks, m, iterations_attempted
            )
            if callback_reason is not None:
                stop_reason = callback_reason
                break
            iterations_attempted = m + 1
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
            bootstrap_factors = self._bayesian_bootstrap_factors(n_samples, rng)
            grad, hess = self._apply_bootstrap_multiclass(
                grad, hess, bootstrap_factors
            )
            fmask, findices = self._feature_selection(X_binned.shape[1], rng)
            if self.sampling_ == "goss":
                grad, hess, row_indices_round = self._goss_subsample_multiclass(
                    grad, hess, rng
                )
                self._record_sampling_diagnostic(row_indices_round, n_samples)
            elif self.sampling_ == "weighted_goss":
                grad, hess, row_indices_round = self._weighted_goss_subsample_multiclass(
                    grad, hess, rng
                )
                self._record_sampling_diagnostic(row_indices_round, n_samples)
            elif self.sampling_ == "mvs":
                grad, hess, row_indices_round = self._mvs_subsample_multiclass(
                    grad, hess, rng
                )
                self._record_sampling_diagnostic(row_indices_round, n_samples)
            elif self.subsample >= 1.0:
                row_indices_round = None
                self._record_sampling_diagnostic(None, n_samples)
            else:
                mask = rng.random(n_samples) < self.subsample
                if not np.any(mask):
                    importance = (
                        np.sum(np.abs(grad), axis=0)
                        + np.sum(np.maximum(hess, 0.0), axis=0)
                    )
                    if (
                        not np.any(np.isfinite(importance))
                        or float(np.sum(importance)) <= 0.0
                    ):
                        chosen = int(rng.integers(0, n_samples))
                    else:
                        chosen = int(np.nanargmax(importance))
                    mask[chosen] = True
                row_indices_round = np.flatnonzero(mask).astype(np.int64)
                self._record_sampling_diagnostic(row_indices_round, n_samples)
            _add_timing(timing, "grad_hess", phase)

            grad_for_round, hess_for_round = grad, hess
            if row_indices_round is not None and self.sampling_ == "uniform":
                row_mask = np.zeros(n_samples, dtype=bool)
                row_mask[row_indices_round] = True
                grad_for_round = np.where(row_mask[None, :], grad, 0.0)
                hess_for_round = np.where(row_mask[None, :], hess, 0.0)

            if use_shared_lightgbm_multiclass:
                phase = _start_timing(timing)
                grad_shared_row_major[:, :] = grad_for_round.T
                hess_shared_row_major[:, :] = hess_for_round.T
                tree, leaf, leaf_G, leaf_H = build_leafwise_multiclass_tree(
                    X_binned, grad_for_round, hess_for_round, n_bins,
                    self._max_tree_depth(),
                    self.l2_leaf_reg_, self.lr_,
                    feature_mask=fmask,
                    min_child_weight=self.min_child_weight,
                    hist_buffers=multiclass_hist_buffers,
                    return_training_state=True,
                    X_hist_binned=X_hist_binned,
                    X_route_binned=X_route_binned,
                    max_leaves=self._max_tree_leaves(),
                    min_child_samples=self.min_child_samples,
                    min_gain_to_split=self.min_gain_to_split,
                    random_strength=self.random_strength,
                    split_seed=int(getattr(self, "_split_seed_", 0)),
                    tree_iteration=m,
                    grad_row_major=grad_shared_row_major,
                    hess_row_major=hess_shared_row_major,
                    leaf_dtype=self.leaf_dtype_,
                )
                _add_timing(timing, "tree_build", phase)
                if tree.depth == 0:
                    if (
                        row_indices_round is not None
                        and m + 1 < self.iterations_
                        and sampled_depth0_retries
                        < _MAX_CONSECUTIVE_SAMPLED_DEPTH0_RETRIES
                    ):
                        sampled_depth0_retries += 1
                        if self.verbose:
                            print(
                                f"No split at sampled iteration {m}; retrying "
                                "with a new row sample."
                            )
                        continue
                    if self.verbose:
                        print(f"No further splits at iteration {m}; stopping.")
                    stop_reason = "no_split"
                    break

                sampled_depth0_retries = 0
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
                    successful_iter = len(self.trees_) - 1
                    if val < best_prefix_score:
                        best_prefix_score, best_prefix_iter = val, successful_iter
                    if patience_score - val > self.early_stopping_min_delta_:
                        patience_score, patience_iter = val, successful_iter
                    elif (self.early_stopping_rounds_ and
                          successful_iter - patience_iter >= self.early_stopping_rounds_):
                        if self.verbose:
                            print(f"Early stop at {m} (best {best_prefix_iter})")
                        stop_reason = "early_stopping"
                        break

                if self.verbose and (m % max(1, self.iterations_ // 10) == 0):
                    msg = f"[{m}] train {self.train_history_[-1]:.5f}"
                    if Fv is not None:
                        msg += f"  val {self.valid_history_[-1]:.5f}"
                    print(msg)
                continue

            phase = _start_timing(timing)
            fuse_root_this_round = use_fused_root and fmask is None
            if fuse_root_this_round:
                _build_multiclass_histograms_counts_into(
                    X_hist_binned, grad_for_round, hess_for_round, root_leaf, 1,
                    root_g, root_h, root_c
                )
            _add_timing(timing, "tree_build", phase)

            phase = _start_timing(timing)
            grad_for_tree, hess_for_tree = grad_for_round, hess_for_round
            _add_timing(timing, "grad_hess", phase)

            round_trees = []
            for k in range(K):
                g, h = grad_for_tree[k], hess_for_tree[k]
                phase = _start_timing(timing)
                builder_kwargs = self._builder_kwargs(
                    fmask, findices, row_indices_round, hist_buffers,
                    split_buffers, X_hist_binned, X_route_binned, False,
                    hessian_always_positive=(
                        self.tree_mode_ in {"lightgbm", "hybrid"}
                        and w is None
                        and row_indices_round is None
                    ),
                    rowpar_buffers=rowpar_buffers,
                    tree_iteration=(m * K + k),
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
                elif tree.depth > 0:
                    add_leaf_values_inplace(leaf, tree.values, F[k])
                _add_timing(timing, "train_update", phase)
            # Stop only if EVERY class exhausted its splits this round; if even
            # one class is still learning, the round was productive.
            if all(t.depth == 0 for t in round_trees):
                if (
                    row_indices_round is not None
                    and m + 1 < self.iterations_
                    and sampled_depth0_retries
                    < _MAX_CONSECUTIVE_SAMPLED_DEPTH0_RETRIES
                ):
                    sampled_depth0_retries += 1
                    if self.verbose:
                        print(
                            f"No class split at sampled iteration {m}; "
                            "retrying with a new row sample."
                        )
                    continue
                if self.verbose:
                    print(f"No further splits for any class at iteration {m}; "
                          f"stopping.")
                stop_reason = "no_split"
                break
            sampled_depth0_retries = 0
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
                successful_iter = len(self.trees_) - 1
                if val < best_prefix_score:
                    best_prefix_score, best_prefix_iter = val, successful_iter
                if patience_score - val > self.early_stopping_min_delta_:
                    patience_score, patience_iter = val, successful_iter
                elif (self.early_stopping_rounds_ and
                      successful_iter - patience_iter >= self.early_stopping_rounds_):
                    if self.verbose:
                        print(f"Early stop at {m} (best {best_prefix_iter})")
                    stop_reason = "early_stopping"
                    break

            if self.verbose and (m % max(1, self.iterations_ // 10) == 0):
                msg = f"[{m}] train {self.train_history_[-1]:.5f}"
                if Fv is not None:
                    msg += f"  val {self.valid_history_[-1]:.5f}"
                print(msg)

        rounds_completed = len(self.trees_)
        self.fit_time_ = time.perf_counter() - t0
        self._truncate_to_best_model(best_prefix_iter, self.valid_history_)
        self._refresh_stochastic_auto_params(n_samples)
        self.best_iteration_ = len(self.trees_)
        self._finalize_training_metadata(
            stop_reason=stop_reason,
            iterations_attempted=iterations_attempted,
            rounds_completed=rounds_completed,
            best_prefix_iter=best_prefix_iter,
            callbacks=callbacks,
        )
        if self.valid_history_:
            self.best_score_ = best_prefix_score
        elif Fv is not None:
            self.best_score_ = self.loss_.eval_class_major_labels(yv_idx, Fv, wv)
        elif self.train_history_:
            self.best_score_ = self.train_history_[-1]
        else:
            self.best_score_ = self.loss_.eval_class_major_labels(y_idx, F, w)
        self._record_input_feature_metadata(n_features, feature_names)
        return self

    def _build_flat_ensemble(self):
        return build_flat_multiclass_ensemble(self.trees_, self.n_classes_)

    def predict_raw(self, X, *, _validated=False):
        """Return the (n_samples, n_classes) matrix of raw per-class scores
        (pre-softmax)."""
        X = self._prepare_predict_X(X, validated=_validated)
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

    def staged_predict_raw(self, X, *, _validated=False):
        """Yield raw scores after each complete multiclass boosting round."""
        X = self._prepare_predict_X(X, validated=_validated)
        X_binned = self.prep_.transform(X)
        F = np.tile(self.init_[:, None], (1, X_binned.shape[0]))
        for stage, round_trees in enumerate(self.trees_):
            if stage:
                self._restore_thread_count()
            if hasattr(round_trees, "add_predict_class_major"):
                round_trees.add_predict_class_major(X_binned, F)
            else:
                for k in range(self.n_classes_):
                    round_trees[k].add_predict(X_binned, F[k])
            yield F.T.copy()


class DistributionalBoosting(_BaseBooster):
    """Vector-output regression booster for distributional losses."""

    def __init__(self, loss="Gaussian", loss_kwargs=None, **kw):
        super().__init__(**kw)
        self.loss_name = loss
        self.loss_kwargs = dict(loss_kwargs or {})

    @staticmethod
    def _disabled_target_transform():
        return {
            "enabled": False,
            "mean": 0.0,
            "scale": 1.0,
            "basis": "target",
        }

    def _standardization_target(self, y):
        if hasattr(self.loss_, "standardization_target"):
            values = self.loss_.standardization_target(y)
        else:
            values = y
        values = np.asarray(values, dtype=np.float64)
        if values.shape != np.asarray(y).shape:
            raise ValueError(
                f"loss={self.loss_name!r} standardization target shape "
                "does not match y"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"loss={self.loss_name!r} standardization target must be finite"
            )
        return values

    def _fit_target_transform(self, y, sample_weight):
        if not bool(getattr(self.loss_, "target_standardization", False)):
            self.target_transform_ = self._disabled_target_transform()
            return np.asarray(y, dtype=np.float64)

        values = self._standardization_target(y)
        if sample_weight is None:
            mean = float(np.mean(values))
            variance = float(np.mean((values - mean) ** 2))
        else:
            w = np.asarray(sample_weight, dtype=np.float64)
            positive = w > 0.0
            if not np.any(positive):
                raise ValueError("sample_weight must have positive total weight")
            values_fit = values[positive]
            weights_fit = w[positive]
            mean = float(np.average(values_fit, weights=weights_fit))
            variance = float(
                np.average((values_fit - mean) ** 2, weights=weights_fit)
            )
        scale = float(np.sqrt(max(variance, 1e-12)))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        self.target_transform_ = {
            "enabled": True,
            "mean": mean,
            "scale": scale,
            "basis": str(
                getattr(self.loss_, "target_standardization_basis", "target")
            ),
        }
        return self._apply_target_transform(y)

    def _apply_target_transform(self, y):
        transform = getattr(
            self, "target_transform_", self._disabled_target_transform()
        )
        if not bool(transform.get("enabled", False)):
            return np.asarray(y, dtype=np.float64)
        mean = float(transform.get("mean", 0.0))
        scale = float(transform.get("scale", 1.0))
        if hasattr(self.loss_, "transform_target"):
            return np.asarray(
                self.loss_.transform_target(y, mean, scale),
                dtype=np.float64,
            )
        return (np.asarray(y, dtype=np.float64) - mean) / scale

    def _raw_to_target_scale(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        transform = getattr(
            self, "target_transform_", self._disabled_target_transform()
        )
        if not bool(transform.get("enabled", False)):
            return raw
        scale = float(transform.get("scale", 1.0))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        out = raw.copy()
        if out.shape[1] >= 1:
            out[:, 0] = float(transform.get("mean", 0.0)) + scale * out[:, 0]
        if out.shape[1] >= 2:
            out[:, 1] = out[:, 1] + np.log(scale)
        return out

    def _raw_to_internal_scale(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        transform = getattr(
            self, "target_transform_", self._disabled_target_transform()
        )
        if not bool(transform.get("enabled", False)):
            return raw
        scale = float(transform.get("scale", 1.0))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        out = raw.copy()
        if out.shape[1] >= 1:
            out[:, 0] = (out[:, 0] - float(transform.get("mean", 0.0))) / scale
        if out.shape[1] >= 2:
            out[:, 1] = out[:, 1] - np.log(scale)
        return out

    def _params_to_target_scale(self, params):
        transform = getattr(
            self, "target_transform_", self._disabled_target_transform()
        )
        if not bool(transform.get("enabled", False)):
            return tuple(params)
        scale = float(transform.get("scale", 1.0))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        out = list(params)
        if out:
            out[0] = (
                float(transform.get("mean", 0.0))
                + scale * np.asarray(out[0], dtype=np.float64)
            )
        scale_idx = int(getattr(self.loss_, "scale_param_index", 1))
        if 0 <= scale_idx < len(out):
            out[scale_idx] = scale * np.asarray(out[scale_idx], dtype=np.float64)
        return tuple(out)

    def params_from_raw(self, raw):
        internal_raw = self._raw_to_internal_scale(raw)
        return self._params_to_target_scale(
            self.loss_.params_from_raw(internal_raw)
        )

    def variance_from_raw(self, raw):
        params = self.params_from_raw(raw)
        if hasattr(self.loss_, "variance_from_params"):
            return self.loss_.variance_from_params(*params)
        return self.loss_.variance_from_raw(self._raw_to_internal_scale(raw))

    def fit(self, X, y, cat_features=None, eval_set=None, sample_weight=None,
            eval_sample_weight=None, callbacks=None):
        """Fit a shared-vector distributional regression model."""
        if self.linear_leaves:
            raise ValueError(
                "linear_leaves=True is currently supported only for "
                "scalar RMSE regression"
            )
        callbacks = _normalize_callbacks(callbacks)
        X, cat_features, n_features, feature_names = self._coerce_fit_X(
            X, cat_features
        )
        n_samples = X.shape[0]
        y = validate_target_vector(y, n_samples, dtype=np.float64)
        _reject_eval_sample_weight_without_eval_set(eval_sample_weight, eval_set)
        try:
            self.loss_ = VECTOR_LOSSES[self.loss_name](**self.loss_kwargs)
        except KeyError as exc:
            valid = ", ".join(sorted(VECTOR_LOSSES))
            raise ValueError(
                f"unknown distributional loss {self.loss_name!r}; valid "
                f"losses are: {valid}"
            ) from exc
        self.loss_.validate_target(y)
        Xv = yv = yv_fit = wv = None
        if eval_set is not None:
            Xv, yv = eval_set
            Xv = self._coerce_eval_X(
                Xv,
                cat_features,
                expected_n_features=n_features,
                expected_feature_names=feature_names,
            )
            yv = validate_target_vector(
                yv, Xv.shape[0], name="eval_set[1]", dtype=np.float64
            )
            self.loss_.validate_target(yv)
            wv = _validate_sample_weight(
                eval_sample_weight, len(yv), name="eval_sample_weight"
            )

        self.tree_mode_ = _normalize_tree_mode(self.tree_mode)
        fit_random_state = normalize_random_state_seed(self.random_state)
        self._fit_random_state_seed_ = fit_random_state
        self.n_threads_ = _fit_thread_count(
            self.thread_count, self.tree_mode_, n_samples
        )
        self.histogram_dtype_ = _normalize_histogram_dtype(self.histogram_dtype)
        self.leaf_dtype_ = _normalize_leaf_dtype_name(self.leaf_dtype)
        self._validate_sampling_config()
        self.ordered_boosting_ = self._resolve_ordered_boosting(
            loss_name=self.loss_name,
        )
        w = _validate_sample_weight(sample_weight, n_samples)
        y_fit = self._fit_target_transform(y, w)
        self.eval_metric_ = _normalize_eval_metric(
            self.eval_metric, self.loss_name
        )
        self._resolve_auto_structure_params(
            loss_name=self.loss_name,
            n_samples=n_samples,
            sample_weight=w,
            X=X,
            cat_features=cat_features,
        )
        self.l2_leaf_reg_ = float(self.l2_leaf_reg)

        if self.tree_mode_ != "lightgbm":
            raise ValueError(
                f"loss={self.loss_name!r} requires tree_mode='lightgbm' "
                f"(shared vector trees); got {self.tree_mode!r}"
            )
        if self.sampling_ in {"goss", "weighted_goss"} or self._mvs_active():
            raise ValueError(
                f"loss={self.loss_name!r} supports only sampling='uniform' "
                "in v1.1; "
                "GOSS and MVS sampling are not supported yet"
            )
        if self._bayesian_bootstrap_active():
            raise ValueError(
                f"loss={self.loss_name!r} does not support Bayesian "
                "bootstrap in v1; "
                "leave bootstrap_type='none' or bagging_temperature=0.0"
            )
        if self.ordered_boosting_:
            raise ValueError(
                f"loss={self.loss_name!r} does not support "
                "ordered_boosting=True in v1"
            )
        if self.histogram_dtype_ != "float64":
            raise ValueError(
                "histogram_dtype='float32' is not supported for "
                f"loss={self.loss_name!r}; shared vector trees are "
                "float64-only"
            )

        timing = _new_timing(self.verbose_timing)
        self.timing_ = timing
        phase = _start_timing(timing)
        preprocessing_target = y_fit
        if hasattr(self.loss_, "preprocessing_target"):
            preprocessing_target = self.loss_.preprocessing_target(y_fit)
        X_binned = self._fit_transform_preprocessor(
            X, [preprocessing_target], cat_features, w,
        )
        X_route_binned = np.asfortranarray(X_binned)
        X_hist_binned = (
            X_route_binned if self.n_threads_ > 1 else X_binned
        )
        n_bins = self.prep_.n_bins_
        self._resolve_fit_auto_params(
            loss_name=self.loss_name,
            n_samples=n_samples,
            sample_weight=w,
            eval_set_present=eval_set is not None,
            p_model=X_binned.shape[1],
        )
        K = int(self.loss_.n_outputs)
        self.n_outputs_ = K
        self.l2_leaf_reg_by_output_ = np.full(
            K, self.l2_leaf_reg_, dtype=np.float64
        )
        if K >= 2:
            self.l2_leaf_reg_by_output_[1] = (
                self.l2_leaf_reg_ * self.rho_l2_leaf_reg_multiplier
            )
        hist_buffers = self._alloc_multiclass_hist_buffers(
            K, X_binned.shape[1], n_bins
        )
        self._importance = np.zeros(self.prep_.n_input_features_)

        Xv_binned = Fv = None
        if Xv is not None:
            yv_fit = self._apply_target_transform(yv)
            Xv_binned = self.prep_.transform(Xv)
        _add_timing(timing, "preprocess", phase)

        self._record_scalar_target_stats(y, w)
        self.init_ = self.loss_.init_class_major(y_fit, w)
        F = np.tile(self.init_[:, None], (1, n_samples))
        grad_buffer = np.empty_like(F)
        hess_buffer = np.empty_like(F)
        grad_row_major = np.empty((n_samples, K), dtype=np.float64)
        hess_row_major = np.empty((n_samples, K), dtype=np.float64)
        if yv is not None:
            Fv = np.tile(self.init_[:, None], (1, len(yv)))
        baseline_loss = (
            self._eval_metric_class_major(yv_fit, Fv, wv)
            if Fv is not None
            else self._eval_metric_class_major(y_fit, F, w)
        )
        self._finalize_early_stopping_min_delta(baseline_loss, self.loss_name)
        self.auto_params_ = self._resolved_auto_params(
            n_samples=n_samples,
            n_raw_features=X.shape[1],
            X_binned=X_binned,
            n_bins=n_bins,
            sample_weight=w,
            eval_set_present=eval_set is not None,
            eval_n_samples=0 if yv is None else len(yv),
            eval_sample_weight=wv,
            rowpar_buffers=None,
            extra={
                "distributional": {
                    "loss_name": self.loss_name,
                    "distribution_name": getattr(
                        self.loss_, "distribution_name", self.loss_name
                    ),
                    "n_outputs": int(K),
                    "hessian_mode": getattr(self.loss_, "hessian_mode", None),
                    "eval_metric": self.eval_metric_,
                    "rho_learning_rate_multiplier": (
                        float(self.rho_learning_rate_multiplier)
                    ),
                    "rho_l2_leaf_reg_multiplier": (
                        float(self.rho_l2_leaf_reg_multiplier)
                    ),
                    "l2_leaf_reg_by_output": [
                        float(v) for v in self.l2_leaf_reg_by_output_
                    ],
                    "target_transform": dict(self.target_transform_),
                }
            },
        )
        self._emit_auto_param_warnings()
        self._reset_stochastic_diagnostics()

        rng = np.random.default_rng(fit_random_state)
        self._initialize_split_seed(rng, fit_random_state)
        self.trees_ = []
        self._flat_cache_ = None
        self.train_history_, self.valid_history_ = [], []
        eval_train = self.eval_train_loss or bool(self.verbose)
        patience_score, patience_iter = np.inf, 0
        best_prefix_score, best_prefix_iter = np.inf, 0
        sampled_depth0_retries = 0
        iterations_attempted = 0
        stop_reason = "iteration_limit"
        t0 = time.perf_counter()

        for m in range(self.iterations_):
            callback_reason = self._callback_stop_reason(
                callbacks, m, iterations_attempted
            )
            if callback_reason is not None:
                stop_reason = callback_reason
                break
            iterations_attempted = m + 1
            phase = _start_timing(timing)
            self.loss_.grad_hess_class_major_into(
                y_fit, F, w, grad_buffer, hess_buffer
            )
            fmask, findices = self._feature_selection(X_binned.shape[1], rng)
            if self.subsample >= 1.0:
                row_indices_round = None
                grad_for_round = grad_buffer
                hess_for_round = hess_buffer
                self._record_sampling_diagnostic(None, n_samples)
            else:
                mask = rng.random(n_samples) < self.subsample
                if not np.any(mask):
                    importance = (
                        np.sum(np.abs(grad_buffer), axis=0)
                        + np.sum(np.maximum(hess_buffer, 0.0), axis=0)
                    )
                    if (
                        not np.any(np.isfinite(importance))
                        or float(np.sum(importance)) <= 0.0
                    ):
                        chosen = int(rng.integers(0, n_samples))
                    else:
                        chosen = int(np.nanargmax(importance))
                    mask[chosen] = True
                row_indices_round = np.flatnonzero(mask).astype(np.int64)
                self._record_sampling_diagnostic(row_indices_round, n_samples)
                row_mask = np.zeros(n_samples, dtype=bool)
                row_mask[row_indices_round] = True
                grad_for_round = np.where(row_mask[None, :], grad_buffer, 0.0)
                hess_for_round = np.where(row_mask[None, :], hess_buffer, 0.0)
            grad_row_major[:, :] = grad_for_round.T
            hess_row_major[:, :] = hess_for_round.T
            _add_timing(timing, "grad_hess", phase)

            phase = _start_timing(timing)
            tree, leaf, leaf_G, leaf_H = build_leafwise_multiclass_tree(
                X_binned, grad_for_round, hess_for_round, n_bins,
                self._max_tree_depth(), self.l2_leaf_reg_by_output_, self.lr_,
                feature_mask=fmask,
                min_child_weight=self.min_child_weight,
                hist_buffers=hist_buffers,
                return_training_state=True,
                X_hist_binned=X_hist_binned,
                X_route_binned=X_route_binned,
                max_leaves=self._max_tree_leaves(),
                min_child_samples=self.min_child_samples,
                min_gain_to_split=self.min_gain_to_split,
                random_strength=self.random_strength,
                split_seed=int(getattr(self, "_split_seed_", 0)),
                tree_iteration=m,
                grad_row_major=grad_row_major,
                hess_row_major=hess_row_major,
                leaf_dtype=self.leaf_dtype_,
            )
            _add_timing(timing, "tree_build", phase)
            if tree.depth == 0:
                if (
                    (row_indices_round is not None or fmask is not None)
                    and m + 1 < self.iterations_
                    and sampled_depth0_retries
                    < _MAX_CONSECUTIVE_SAMPLED_DEPTH0_RETRIES
                ):
                    sampled_depth0_retries += 1
                    if self.verbose:
                        print(
                            f"No split at sampled iteration {m}; retrying "
                            "with a new sample."
                        )
                    continue
                if self.verbose:
                    print(f"No further splits at iteration {m}; stopping.")
                stop_reason = "no_split"
                break
            sampled_depth0_retries = 0

            phase = _start_timing(timing)
            if self.rho_learning_rate_multiplier != 1.0 and K >= 2:
                tree.values[:, 1] *= self.rho_learning_rate_multiplier
            self.trees_.append(tree)
            self._accumulate_importance(tree)
            add_multiclass_leaf_values_inplace(leaf, tree.values, F)
            state_refreshed = False
            if hasattr(self.loss_, "refresh_state"):
                state_refreshed = bool(
                    self.loss_.refresh_state(y_fit, F, w, len(self.trees_))
                )
            _add_timing(timing, "train_update", phase)

            if eval_train:
                phase = _start_timing(timing)
                self.train_history_.append(
                    self._eval_metric_class_major(y_fit, F, w)
                )
                _add_timing(timing, "loss_eval", phase)

            if Fv is not None:
                phase = _start_timing(timing)
                tree.add_predict_class_major(Xv_binned, Fv)
                _add_timing(timing, "validation_predict", phase)
                phase = _start_timing(timing)
                val = self._eval_metric_class_major(yv_fit, Fv, wv)
                _add_timing(timing, "loss_eval", phase)
                self.valid_history_.append(val)
                successful_iter = len(self.trees_) - 1
                if state_refreshed:
                    self.valid_history_ = (
                        self._rescore_class_major_prefix_history(
                            Xv_binned, yv_fit, wv
                        )
                    )
                    best_prefix_score, best_prefix_iter = (
                        self._best_prefix_from_history(self.valid_history_)
                    )
                    patience_score, patience_iter = (
                        self._patience_prefix_from_history(self.valid_history_)
                    )
                    val = self.valid_history_[-1]
                elif val < best_prefix_score:
                    best_prefix_score, best_prefix_iter = val, successful_iter
                if (
                    not state_refreshed
                    and patience_score - val > self.early_stopping_min_delta_
                ):
                    patience_score, patience_iter = val, successful_iter
                elif (self.early_stopping_rounds_ and
                      successful_iter - patience_iter >= self.early_stopping_rounds_):
                    if self.verbose:
                        print(f"Early stop at {m} (best {best_prefix_iter})")
                    stop_reason = "early_stopping"
                    break

            if self.verbose and (m % max(1, self.iterations_ // 10) == 0):
                msg = f"[{m}] train {self.train_history_[-1]:.5f}"
                if Fv is not None:
                    msg += f"  val {self.valid_history_[-1]:.5f}"
                print(msg)

        rounds_completed = len(self.trees_)
        self.fit_time_ = time.perf_counter() - t0
        stateful_loss = hasattr(self.loss_, "refresh_state")
        if stateful_loss:
            F = np.tile(self.init_[:, None], (1, n_samples))
            for tree in self.trees_:
                tree.add_predict_class_major(X_binned, F)
            self.loss_.refresh_state(y_fit, F, w, len(self.trees_), force=True)
            if Xv_binned is not None and self.valid_history_:
                self.valid_history_ = self._rescore_class_major_prefix_history(
                    Xv_binned, yv_fit, wv
                )
                best_prefix_score, best_prefix_iter = (
                    self._best_prefix_from_history(self.valid_history_)
                )
            elif Xv_binned is None:
                best_prefix_score = self._eval_metric_class_major(y_fit, F, w)
                if self.train_history_:
                    self.train_history_[-1] = best_prefix_score

        while True:
            tree_count_before_truncate = len(self.trees_)
            self._truncate_to_best_model(best_prefix_iter, self.valid_history_)
            truncated = len(self.trees_) != tree_count_before_truncate
            if not (stateful_loss and truncated):
                break

            F = np.tile(self.init_[:, None], (1, n_samples))
            for tree in self.trees_:
                tree.add_predict_class_major(X_binned, F)
            self.loss_.refresh_state(y_fit, F, w, len(self.trees_), force=True)
            if Xv_binned is not None and self.valid_history_:
                self.valid_history_ = self._rescore_class_major_prefix_history(
                    Xv_binned, yv_fit, wv
                )
                best_prefix_score, best_prefix_iter = (
                    self._best_prefix_from_history(self.valid_history_)
                )
            else:
                best_prefix_score = self._eval_metric_class_major(y_fit, F, w)
                break
        self._refresh_stochastic_auto_params(n_samples)
        self.best_iteration_ = len(self.trees_)
        self._finalize_training_metadata(
            stop_reason=stop_reason,
            iterations_attempted=iterations_attempted,
            rounds_completed=rounds_completed,
            best_prefix_iter=best_prefix_iter,
            callbacks=callbacks,
        )
        if self.valid_history_:
            self.best_score_ = best_prefix_score
        elif stateful_loss and Xv_binned is None:
            self.best_score_ = best_prefix_score
        elif Fv is not None:
            self.best_score_ = self._eval_metric_class_major(yv_fit, Fv, wv)
        elif self.train_history_:
            self.best_score_ = self.train_history_[-1]
        else:
            self.best_score_ = self._eval_metric_class_major(y_fit, F, w)
        self._record_input_feature_metadata(n_features, feature_names)
        return self

    def _rescore_class_major_prefix_history(self, X_binned, y, sample_weight):
        F = np.tile(self.init_[:, None], (1, len(y)))
        history = []
        for tree in self.trees_:
            tree.add_predict_class_major(X_binned, F)
            history.append(self._eval_metric_class_major(y, F, sample_weight))
        return history

    @staticmethod
    def _best_prefix_from_history(history):
        best_score, best_iter = np.inf, 0
        for i, val in enumerate(history):
            if val < best_score:
                best_score, best_iter = float(val), int(i)
        return best_score, best_iter

    def _patience_prefix_from_history(self, history):
        patience_score, patience_iter = np.inf, 0
        for i, val in enumerate(history):
            if patience_score - val > self.early_stopping_min_delta_:
                patience_score, patience_iter = float(val), int(i)
        return patience_score, patience_iter

    def _eval_metric_class_major(self, y, F, sample_weight=None):
        if getattr(self, "eval_metric_", None) == "crps":
            return self.loss_.crps_class_major(y, F, sample_weight)
        return self.loss_.eval_class_major(y, F, sample_weight)

    def _build_flat_ensemble(self):
        return build_flat_multiclass_ensemble(self.trees_, self.n_outputs_)

    def predict_raw(self, X, *, _validated=False):
        """Return sample-major raw scores for the fitted distribution head."""
        X = self._prepare_predict_X(X, validated=_validated)
        X_binned = self.prep_.transform(X)
        F = np.tile(self.init_[:, None], (1, X_binned.shape[0]))
        flat = self._flat_ensemble()
        if flat is not None and flat_predict_preferred(flat):
            flat.add_predict_class_major(X_binned, F)
        else:
            for tree in self.trees_:
                tree.add_predict_class_major(X_binned, F)
        return self._raw_to_target_scale(F.T)

    def predict_dist(self, X):
        raw = self.predict_raw(X)
        return self.params_from_raw(raw)

    def predict_variance(self, X):
        raw = self.predict_raw(X)
        return self.variance_from_raw(raw)

    def staged_predict_raw(self, X, *, _validated=False):
        """Yield sample-major raw scores after each vector-tree round."""
        X = self._prepare_predict_X(X, validated=_validated)
        X_binned = self.prep_.transform(X)
        F = np.tile(self.init_[:, None], (1, X_binned.shape[0]))
        for stage, tree in enumerate(self.trees_):
            if stage:
                self._restore_thread_count()
            tree.add_predict_class_major(X_binned, F)
            yield self._raw_to_target_scale(F.T).copy()
