"""Model persistence: save fitted boosters to a single ``.npz`` file.

The format is a compressed numpy archive holding only plain (non-object)
arrays plus one JSON header string, so it loads with ``allow_pickle=False``
and is robust to library-version drift in a way pickled objects are not.

Layout (format_version 2):
  header                 JSON: format/library versions, model class, params,
                         loss, fitted scalars, preprocessor settings
  classes                class labels (numeric or unicode), multiclass only
  importance             per-input-feature split-gain totals
  prep__* / bin__*       preprocessor and binner arrays
  cat{j}__values/kinds   per-categorical-column category values, stringified,
                         with a parallel kind code (0=str, 1=float, 2=int) so
                         exact key types are rebuilt for dict lookups
  enc{t}__*              per-target encoder category sums/counts
  trees__*               concatenated per-tree arrays with offsets

Categories must be str, float, int, or bool; anything else raises at save
time. The experimental level-wise tree mode is not serializable.
"""

import json
from pathlib import Path

import numpy as np

from .binning import Binner
from .losses import LOSSES, MultiSoftmax
from .preprocessing import FeaturePreprocessor
from .target_encoding import OrderedTargetEncoder, _MISSING_CATEGORY
from .tree import MultiNonObliviousTree, NonObliviousTree, ObliviousTree

FORMAT_VERSION = 2

_KIND_STR = 0
_KIND_FLOAT = 1
_KIND_INT = 2
_KIND_BOOL = 3
_KIND_MISSING = 4


def _jsonify(value):
    """Make constructor params / fitted scalars JSON-safe."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _validate_plain_arrays(arrays):
    for key, value in list(arrays.items()):
        arr = np.asarray(value)
        if arr.dtype.hasobject:
            raise ValueError(
                f"cannot save object-dtype array {key!r}; ChimeraBoost model "
                "archives are loaded with allow_pickle=False"
            )
        arrays[key] = arr


def _load_path(path):
    candidate = Path(path)
    if candidate.exists():
        return candidate
    if candidate.suffix != ".npz":
        with_suffix = candidate.with_suffix(candidate.suffix + ".npz")
        if with_suffix.exists():
            return with_suffix
    return candidate


def _require_offsets(name, offsets, total_size=None, expected_count=None):
    offsets = np.asarray(offsets)
    if offsets.ndim != 1 or offsets.size == 0:
        raise ValueError(f"invalid ChimeraBoost model: {name} offsets are empty")
    if not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError(
            f"invalid ChimeraBoost model: {name} offsets must contain integer values"
        )
    offsets = offsets.astype(np.int64, copy=False)
    if expected_count is not None and offsets.size != int(expected_count):
        raise ValueError(
            f"invalid ChimeraBoost model: {name} offsets length is "
            f"{offsets.size}, expected {int(expected_count)}"
        )
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError(
            f"invalid ChimeraBoost model: {name} offsets are not monotonic"
        )
    if total_size is not None and offsets[-1] != int(total_size):
        raise ValueError(
            f"invalid ChimeraBoost model: {name} offsets do not match array length"
        )
    return offsets


def _require_same_offsets(name, offsets, arrays, expected_count=None):
    """Validate one offset vector against every coupled flat array."""
    checked = None
    for array_name, array in arrays:
        checked = _require_offsets(
            f"{name} {array_name}", offsets, len(array),
            expected_count=expected_count,
        )
    return checked


def _invalid_model(message):
    raise ValueError(f"invalid ChimeraBoost model: {message}")


def _require_array_ndim(name, array, ndim):
    array = np.asarray(array)
    if array.ndim != int(ndim):
        _invalid_model(f"{name} must be {int(ndim)}-dimensional")
    return array


def _require_integer_array(name, array, ndim=1):
    array = _require_array_ndim(name, array, ndim)
    if not np.issubdtype(array.dtype, np.integer):
        _invalid_model(f"{name} must contain integer values")
    return array.astype(np.int64, copy=False)


def _encode_categories(cats):
    """Stringify one column's category values with per-value kind codes."""
    values = []
    kinds = []
    for v in cats:
        if v is _MISSING_CATEGORY:
            values.append("")
            kinds.append(_KIND_MISSING)
        elif isinstance(v, str):
            values.append(v)
            kinds.append(_KIND_STR)
        elif isinstance(v, (bool, np.bool_)):
            values.append(str(bool(v)))
            kinds.append(_KIND_BOOL)
        elif isinstance(v, (int, np.integer)):
            values.append(repr(int(v)))
            kinds.append(_KIND_INT)
        elif isinstance(v, (float, np.floating)):
            values.append(repr(float(v)))
            kinds.append(_KIND_FLOAT)
        else:
            raise ValueError(
                "categorical values must be str, int, float, or bool to "
                f"save; got {type(v).__name__}"
            )
    return (np.array(values, dtype=np.str_),
            np.array(kinds, dtype=np.int8))


def _decode_categories(values, kinds, *, legacy_missing_sentinel=False):
    out = np.empty(len(values), dtype=object)
    for i, (s, k) in enumerate(zip(values, kinds)):
        if k == _KIND_STR:
            value = str(s)
            out[i] = (
                _MISSING_CATEGORY
                if legacy_missing_sentinel and value == "__nan__"
                else value
            )
        elif k == _KIND_FLOAT:
            out[i] = float(s)
        elif k == _KIND_INT:
            out[i] = int(s)
        elif k == _KIND_BOOL:
            out[i] = s == "True"
        elif k == _KIND_MISSING:
            out[i] = _MISSING_CATEGORY
        else:
            raise ValueError(
                f"invalid ChimeraBoost model: unknown category kind {int(k)}"
            )
    return out


def _concat_with_offsets(arrays, dtype=np.float64):
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum([len(a) for a in arrays], out=offsets[1:])
    if arrays:
        flat = np.concatenate(
            [np.asarray(a, dtype=dtype) for a in arrays]
        ) if offsets[-1] else np.empty(0, dtype=dtype)
    else:
        flat = np.empty(0, dtype=dtype)
    return flat, offsets


def _tree_kind(trees_):
    """Classify the fitted tree list; raise for unserializable kinds."""
    if not trees_:
        return "empty"
    first = trees_[0]
    if isinstance(first, (list, tuple)):  # multiclass per-class rounds
        inner = first[0]
        if type(inner) is ObliviousTree:
            return "oblivious_per_class"
        if type(inner) is NonObliviousTree:
            return "nonoblivious_per_class"
        first = inner  # report the per-class tree type in the error
    elif type(first) is ObliviousTree:
        return "oblivious"
    elif type(first) is NonObliviousTree:
        return "nonoblivious"
    elif type(first) is MultiNonObliviousTree:
        return "multi"
    raise ValueError(
        f"cannot serialize trees of type {type(first).__name__}; the "
        "experimental level-wise mode has no save format"
    )


def _pack_oblivious(trees, arrays):
    arrays["trees__depths"] = np.array([t.depth for t in trees],
                                       dtype=np.int64)
    for name, key in (("splits_feat", "feats"), ("splits_thr", "thrs")):
        flat, offsets = _concat_with_offsets(
            [getattr(t, name) for t in trees], dtype=np.int64
        )
        arrays[f"trees__{key}_flat"] = flat
    arrays["trees__gains_flat"], arrays["trees__split_offsets"] = (
        _concat_with_offsets([t.gains for t in trees])
    )
    arrays["trees__values_flat"], arrays["trees__value_offsets"] = (
        _concat_with_offsets([t.values for t in trees])
    )


def _unpack_oblivious(data):
    depths = _require_integer_array("oblivious depths", data["trees__depths"])
    feats = _require_integer_array("oblivious features", data["trees__feats_flat"])
    thrs = _require_integer_array("oblivious thresholds", data["trees__thrs_flat"])
    gains = _require_array_ndim("oblivious gains", data["trees__gains_flat"], 1)
    values = _require_array_ndim("oblivious values", data["trees__values_flat"], 1)
    so = _require_same_offsets(
        "oblivious split",
        data["trees__split_offsets"],
        (("features", feats), ("thresholds", thrs), ("gains", gains)),
        expected_count=len(depths) + 1,
    )
    vo = _require_offsets(
        "oblivious value", data["trees__value_offsets"],
        len(values), expected_count=len(depths) + 1,
    )
    trees = []
    for t in range(len(depths)):
        s0, s1 = so[t], so[t + 1]
        depth = int(depths[t])
        split_len = int(s1 - s0)
        value_len = int(vo[t + 1] - vo[t])
        if split_len != depth:
            _invalid_model("oblivious split payload does not match depth")
        if value_len != (1 << depth):
            _invalid_model("oblivious value payload does not match depth")
        trees.append(ObliviousTree(
            feats[s0:s1].copy(), thrs[s0:s1].copy(),
            values[vo[t]:vo[t + 1]].copy(), gains[s0:s1].copy()
        ))
    return trees


def _pack_nonoblivious(trees, arrays, vector_values=False):
    for name in ("features", "thresholds", "left_child", "right_child",
                 "leaf_index"):
        flat, offsets = _concat_with_offsets(
            [getattr(t, name) for t in trees], dtype=np.int64
        )
        arrays[f"trees__{name}_flat"] = flat
        arrays["trees__node_offsets"] = offsets
    for name in ("splits_feat", "splits_thr"):
        flat, offsets = _concat_with_offsets(
            [getattr(t, name) for t in trees], dtype=np.int64
        )
        arrays[f"trees__{name}_flat"] = flat
        arrays["trees__split_offsets"] = offsets
    arrays["trees__gains_flat"], _ = _concat_with_offsets(
        [t.gains for t in trees]
    )
    if vector_values:
        arrays["trees__values_flat"] = np.vstack([t.values for t in trees])
        leaf_counts = [t.values.shape[0] for t in trees]
        offsets = np.zeros(len(trees) + 1, dtype=np.int64)
        np.cumsum(leaf_counts, out=offsets[1:])
        arrays["trees__value_offsets"] = offsets
    else:
        arrays["trees__values_flat"], arrays["trees__value_offsets"] = (
            _concat_with_offsets([t.values for t in trees])
        )
    arrays["trees__depths"] = np.array([t.depth for t in trees],
                                       dtype=np.int64)
    arrays["trees__n_leaves"] = np.array([t.n_leaves for t in trees],
                                         dtype=np.int64)


def _unpack_nonoblivious(data, cls, expected_value_width=None):
    depths = _require_integer_array("nonoblivious depths", data["trees__depths"])
    n_leaves = _require_integer_array(
        "nonoblivious n_leaves", data["trees__n_leaves"]
    )
    if len(n_leaves) != len(depths):
        _invalid_model("nonoblivious n_leaves length does not match depths")
    features = _require_integer_array(
        "nonoblivious features", data["trees__features_flat"]
    )
    thresholds = _require_integer_array(
        "nonoblivious thresholds", data["trees__thresholds_flat"]
    )
    left_child = _require_integer_array(
        "nonoblivious left_child", data["trees__left_child_flat"]
    )
    right_child = _require_integer_array(
        "nonoblivious right_child", data["trees__right_child_flat"]
    )
    leaf_index = _require_integer_array(
        "nonoblivious leaf_index", data["trees__leaf_index_flat"]
    )
    splits_feat = _require_integer_array(
        "nonoblivious split features", data["trees__splits_feat_flat"]
    )
    splits_thr = _require_integer_array(
        "nonoblivious split thresholds", data["trees__splits_thr_flat"]
    )
    gains = _require_array_ndim("nonoblivious gains", data["trees__gains_flat"], 1)
    if cls is MultiNonObliviousTree:
        values = _require_array_ndim(
            "multiclass nonoblivious values", data["trees__values_flat"], 2
        )
        if (
            expected_value_width is not None
            and values.shape[1] != int(expected_value_width)
        ):
            _invalid_model(
                "multiclass nonoblivious value width does not match classes"
            )
    else:
        values = _require_array_ndim(
            "nonoblivious values", data["trees__values_flat"], 1
        )
    no = _require_same_offsets(
        "nonoblivious node",
        data["trees__node_offsets"],
        (
            ("features", features),
            ("thresholds", thresholds),
            ("left_child", left_child),
            ("right_child", right_child),
            ("leaf_index", leaf_index),
        ),
        expected_count=len(depths) + 1,
    )
    so = _require_same_offsets(
        "nonoblivious split",
        data["trees__split_offsets"],
        (("features", splits_feat), ("thresholds", splits_thr), ("gains", gains)),
        expected_count=len(depths) + 1,
    )
    vo = _require_offsets(
        "nonoblivious value", data["trees__value_offsets"],
        len(values), expected_count=len(depths) + 1,
    )
    trees = []
    for t in range(len(depths)):
        n0, n1 = no[t], no[t + 1]
        s0, s1 = so[t], so[t + 1]
        node_len = int(n1 - n0)
        split_len = int(s1 - s0)
        leaf_count = int(n_leaves[t])
        value_len = int(vo[t + 1] - vo[t])
        if leaf_count < 1:
            _invalid_model("nonoblivious tree has no leaves")
        if split_len != leaf_count - 1:
            _invalid_model("nonoblivious split payload does not match leaves")
        if node_len != 2 * leaf_count - 1:
            _invalid_model("nonoblivious node payload does not match leaves")
        if value_len != leaf_count:
            _invalid_model("nonoblivious value payload does not match leaves")
        leaf_slice = leaf_index[n0:n1]
        if np.any((leaf_slice >= leaf_count) | (leaf_slice < -1)):
            _invalid_model("nonoblivious leaf indexes are out of range")
        child_slice = np.concatenate((left_child[n0:n1], right_child[n0:n1]))
        if np.any((child_slice >= node_len) | (child_slice < -1)):
            _invalid_model("nonoblivious child indexes are out of range")
        trees.append(cls(
            features[n0:n1].copy(),
            thresholds[n0:n1].copy(),
            left_child[n0:n1].copy(),
            right_child[n0:n1].copy(),
            leaf_index[n0:n1].copy(),
            values[vo[t]:vo[t + 1]].copy(),
            splits_feat[s0:s1].copy(),
            splits_thr[s0:s1].copy(),
            gains[s0:s1].copy(),
            int(depths[t]),
            leaf_count,
        ))
    return trees


def save_booster(booster, path, wrapper_header=None, wrapper_arrays=None):
    """Serialize a fitted GradientBoosting / MulticlassBoosting to ``path``.

    ``wrapper_header`` / ``wrapper_arrays`` let the sklearn wrappers attach
    their own state (e.g. the binary classifier's original class labels)
    under the ``wrapper`` header key and ``wrapper__*`` array keys.
    """
    from . import __version__
    from .booster import GradientBoosting, MulticlassBoosting

    if not hasattr(booster, "trees_"):
        raise ValueError("cannot save an unfitted model")
    prep = booster.prep_
    arrays = {}
    header = {
        "format_version": FORMAT_VERSION,
        "library_version": __version__,
        "model_class": type(booster).__name__,
        "lr": float(booster.lr_),
        "best_iteration": int(booster.best_iteration_),
        "best_score": float(booster.best_score_),
        "auto_params": _jsonify(getattr(booster, "auto_params_", {})),
        "timing": _jsonify(getattr(booster, "timing_", None)),
        "train_history": _jsonify(getattr(booster, "train_history_", [])),
        "valid_history": _jsonify(getattr(booster, "valid_history_", [])),
        "n_threads": _jsonify(getattr(booster, "n_threads_", None)),
        "n_input_features": int(prep.n_input_features_),
        "prep": {
            "max_bins": prep.max_bins,
            "cat_smoothing": prep.cat_smoothing,
            "include_cat_codes": prep.include_cat_codes,
            "target_encoding_mode": prep.target_encoding_mode,
            "target_encoding_folds": prep.target_encoding_folds,
            "n_encoders": len(getattr(prep, "encoders_", [])),
            "encoder_priors": [
                float(e.prior_) for e in getattr(prep, "encoders_", [])
            ],
            "encoder_smoothings": [
                float(e.smoothing) for e in getattr(prep, "encoders_", [])
            ],
            "encoder_modes": [
                e.mode for e in getattr(prep, "encoders_", [])
            ],
            "legacy_missing_aliases": [
                (
                    "__nan__" in cat_map
                    and _MISSING_CATEGORY in cat_map
                    and cat_map["__nan__"] == cat_map[_MISSING_CATEGORY]
                )
                for cat_map in getattr(prep, "cat_maps_", [])
            ],
        },
    }

    if isinstance(booster, MulticlassBoosting):
        header["init"] = [float(v) for v in booster.init_]
        header["n_classes"] = int(booster.n_classes_)
        header["loss_name"] = "MultiSoftmax"
        header["loss_kwargs"] = {}
        arrays["classes"] = np.asarray(booster.classes_)
        if arrays["classes"].dtype == object:
            vals, kinds = _encode_categories(booster.classes_)
            arrays["classes"] = vals
            arrays["classes_kinds"] = kinds
    elif isinstance(booster, GradientBoosting):
        header["init"] = float(booster.init_)
        header["loss_name"] = booster.loss_name
        header["loss_kwargs"] = _jsonify(booster.loss_kwargs)
    else:
        raise TypeError(f"unsupported booster type {type(booster).__name__}")

    param_names = (
        "iterations", "learning_rate", "depth", "l2_leaf_reg",
        "max_bins", "subsample", "colsample", "cat_smoothing",
        "early_stopping_rounds", "early_stopping_min_delta",
        "min_child_weight", "min_child_samples", "min_gain_to_split",
        "num_leaves", "thread_count", "random_state", "ordered_boosting",
        "tree_mode", "sampling", "top_rate", "other_rate",
        "multiclass_tree_strategy", "eval_train_loss", "bin_sample_count",
        "histogram_parallelism", "use_best_model", "bootstrap_type",
        "bagging_temperature", "mvs_reg", "random_strength",
        "diagnostic_warnings",
    )
    constructor_inputs = {
        "depth": "_depth_input",
        "num_leaves": "_num_leaves_input",
        "l2_leaf_reg": "_l2_leaf_reg_input",
        "min_child_samples": "_min_child_samples_input",
        "min_child_weight": "_min_child_weight_input",
        "cat_smoothing": "_cat_smoothing_input",
    }
    params = {}
    for name in param_names:
        if name == "random_state" and hasattr(booster, "_fit_random_state_seed_"):
            value = booster._fit_random_state_seed_
        else:
            attr = constructor_inputs.get(name)
            value = (
                getattr(booster, attr)
                if attr and hasattr(booster, attr)
                else getattr(booster, name)
            )
        params[name] = _jsonify(value)
    header["params"] = params

    # ---- trees -------------------------------------------------------------
    kind = _tree_kind(booster.trees_)
    header["tree_kind"] = kind
    if kind in {"oblivious_per_class", "nonoblivious_per_class"}:
        header["n_rounds"] = len(booster.trees_)
        flat_trees = [t for round_trees in booster.trees_
                      for t in round_trees]
    else:
        flat_trees = list(booster.trees_)
    if kind.startswith("oblivious"):
        _pack_oblivious(flat_trees, arrays)
    elif kind != "empty":
        _pack_nonoblivious(flat_trees, arrays, vector_values=(kind == "multi"))

    # ---- preprocessor --------------------------------------------------------
    arrays["importance"] = booster._importance
    arrays["prep__num_features"] = np.asarray(prep.num_features_,
                                              dtype=np.int64)
    arrays["prep__cat_features"] = np.asarray(prep.cat_features_,
                                              dtype=np.int64)
    arrays["prep__feature_map"] = prep.feature_map_
    binner = prep.binner_
    arrays["bin__borders_flat"] = binner._borders_flat_
    arrays["bin__border_offsets"] = binner._border_offsets_
    arrays["bin__n_bins"] = binner.n_bins_
    arrays["bin__block_widths"] = np.asarray(binner._block_widths_,
                                             dtype=np.int64)
    for j, cats in enumerate(getattr(prep, "cat_categories_", [])):
        vals, kinds = _encode_categories(cats)
        arrays[f"cat{j}__values"] = vals
        arrays[f"cat{j}__kinds"] = kinds
    for t, enc in enumerate(getattr(prep, "encoders_", [])):
        sums_flat, offsets = _concat_with_offsets(enc.sums_)
        counts_flat, _ = _concat_with_offsets(enc.counts_)
        arrays[f"enc{t}__sums_flat"] = sums_flat
        arrays[f"enc{t}__counts_flat"] = counts_flat
        arrays[f"enc{t}__offsets"] = offsets

    if wrapper_header:
        header["wrapper"] = _jsonify(wrapper_header)
    if wrapper_arrays:
        for key, value in wrapper_arrays.items():
            arrays[f"wrapper__{key}"] = value

    arrays["header"] = np.array(json.dumps(header))
    _validate_plain_arrays(arrays)
    np.savez_compressed(path, **arrays)


def load_booster(path, return_wrapper_payload=False):
    """Load a booster saved by :func:`save_booster`.

    With ``return_wrapper_payload=True``, returns
    ``(booster, wrapper_header, wrapper_arrays)`` so the sklearn wrappers can
    restore their own state.
    """
    from .booster import GradientBoosting, MulticlassBoosting

    wrapper_header = {}
    wrapper_arrays = {}
    try:
        archive = np.load(_load_path(path), allow_pickle=False)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"{path!r} is not a ChimeraBoost model archive") from exc
    with archive as data:
        try:
            header = json.loads(str(data["header"]))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{path!r} is not a ChimeraBoost model archive"
            ) from exc
        format_version = int(header["format_version"])
        if format_version > FORMAT_VERSION:
            raise ValueError(
                f"model format {format_version} is newer than this "
                f"library understands ({FORMAT_VERSION})"
            )
        model_class = header["model_class"]
        params = header["params"]
        if model_class == "GradientBoosting":
            booster = GradientBoosting(
                loss=header["loss_name"], loss_kwargs=header["loss_kwargs"],
                **params
            )
            booster.init_ = header["init"]
            booster.loss_ = LOSSES[header["loss_name"]](
                **header["loss_kwargs"]
            )
        elif model_class == "MulticlassBoosting":
            booster = MulticlassBoosting(**params)
            booster.init_ = np.array(header["init"], dtype=np.float64)
            booster.n_classes_ = header["n_classes"]
            booster.loss_ = MultiSoftmax(booster.n_classes_)
            classes = data["classes"]
            if "classes_kinds" in data:
                classes = _decode_categories(classes, data["classes_kinds"])
            booster.classes_ = classes
        else:
            raise ValueError(f"unknown model class {model_class!r}")

        booster.lr_ = header["lr"]
        booster.best_iteration_ = header["best_iteration"]
        booster.best_score_ = header["best_score"]
        booster.auto_params_ = header.get("auto_params", {})
        booster.timing_ = header.get("timing")
        booster.train_history_ = list(header.get("train_history", []))
        booster.valid_history_ = list(header.get("valid_history", []))
        saved_n_threads = header.get("n_threads")
        if saved_n_threads is not None:
            booster.n_threads_ = int(saved_n_threads)
        else:
            import numba
            requested_threads = params.get("thread_count")
            max_threads = numba.config.NUMBA_NUM_THREADS
            if requested_threads is None or requested_threads < 0:
                booster.n_threads_ = int(max_threads)
            else:
                booster.n_threads_ = max(1, min(int(requested_threads), max_threads))
        booster._importance = data["importance"]

        # ---- trees ----------------------------------------------------------
        kind = header["tree_kind"]
        if kind not in {
            "empty",
            "oblivious",
            "nonoblivious",
            "multi",
            "oblivious_per_class",
            "nonoblivious_per_class",
        }:
            _invalid_model(f"unknown tree kind {kind!r}")
        if kind == "empty":
            trees = []
        elif kind.startswith("oblivious"):
            trees = _unpack_oblivious(data)
        elif kind == "multi":
            trees = _unpack_nonoblivious(
                data, MultiNonObliviousTree,
                expected_value_width=header["n_classes"],
            )
        else:
            trees = _unpack_nonoblivious(data, NonObliviousTree)
        if kind in {"oblivious_per_class", "nonoblivious_per_class"}:
            K = header["n_classes"]
            n_rounds = int(header["n_rounds"])
            expected_tree_count = n_rounds * int(K)
            if len(trees) != expected_tree_count:
                _invalid_model(
                    "per-class tree count does not match rounds and classes"
                )
            booster.trees_ = [
                trees[r * K:(r + 1) * K]
                for r in range(n_rounds)
            ]
        else:
            booster.trees_ = trees

        # ---- preprocessor ----------------------------------------------------
        prep_cfg = header["prep"]
        prep = FeaturePreprocessor(
            prep_cfg["max_bins"], prep_cfg["cat_smoothing"],
            params.get("random_state"),
            include_cat_codes=prep_cfg["include_cat_codes"],
            target_encoding_mode=prep_cfg["target_encoding_mode"],
            target_encoding_folds=prep_cfg["target_encoding_folds"],
        )
        prep.num_features_ = data["prep__num_features"].tolist()
        prep.cat_features_ = data["prep__cat_features"].tolist()
        prep.feature_map_ = data["prep__feature_map"]
        prep.n_input_features_ = header["n_input_features"]
        prep._cat_indexes_ = {}
        prep.cat_categories_ = []
        prep.cat_maps_ = []
        legacy_aliases = prep_cfg.get("legacy_missing_aliases", [])
        for j in range(len(prep.cat_features_)):
            cats = _decode_categories(
                data[f"cat{j}__values"], data[f"cat{j}__kinds"],
                legacy_missing_sentinel=(format_version <= 1),
            )
            prep.cat_categories_.append(cats)
            cat_map = {v: i for i, v in enumerate(cats)}
            has_saved_legacy_alias = (
                j < len(legacy_aliases) and bool(legacy_aliases[j])
            )
            if (
                (format_version <= 1 or has_saved_legacy_alias)
                and _MISSING_CATEGORY in cat_map
            ):
                cat_map.setdefault("__nan__", cat_map[_MISSING_CATEGORY])
            prep.cat_maps_.append(cat_map)
        prep.encoders_ = []
        for t in range(prep_cfg["n_encoders"]):
            enc = OrderedTargetEncoder(
                prep_cfg["encoder_smoothings"][t],
                mode=prep_cfg["encoder_modes"][t],
            )
            enc.prior_ = prep_cfg["encoder_priors"][t]
            sums_flat = data[f"enc{t}__sums_flat"]
            counts_flat = data[f"enc{t}__counts_flat"]
            offsets = _require_same_offsets(
                f"encoder {t}",
                data[f"enc{t}__offsets"],
                (("sums", sums_flat), ("counts", counts_flat)),
                expected_count=len(prep.cat_features_) + 1,
            )
            enc.sums_ = [
                sums_flat[offsets[j]:offsets[j + 1]].copy()
                for j in range(len(offsets) - 1)
            ]
            enc.counts_ = [
                counts_flat[offsets[j]:offsets[j + 1]].copy()
                for j in range(len(offsets) - 1)
            ]
            enc.n_cat_ = [len(s) for s in enc.sums_]
            prep.encoders_.append(enc)

        binner = Binner(prep_cfg["max_bins"])
        binner._borders_flat_ = data["bin__borders_flat"]
        binner._border_offsets_ = data["bin__border_offsets"]
        binner.n_bins_ = data["bin__n_bins"]
        binner._block_widths_ = data["bin__block_widths"].tolist()
        binner.borders_ = [
            binner._borders_flat_[
                binner._border_offsets_[f]:binner._border_offsets_[f + 1]
            ]
            for f in range(len(binner.n_bins_))
        ]
        prep.binner_ = binner
        prep.n_bins_ = binner.n_bins_
        booster.prep_ = prep

        wrapper_header = header.get("wrapper", {})
        for key in data.files:
            if key.startswith("wrapper__"):
                wrapper_arrays[key[len("wrapper__"):]] = data[key]

    if return_wrapper_payload:
        return booster, wrapper_header, wrapper_arrays
    return booster
