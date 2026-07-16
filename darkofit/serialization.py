"""Model persistence: save fitted boosters to a single ``.npz`` file.

The format is a compressed numpy archive holding only plain (non-object)
arrays plus one JSON header string, so it loads with ``allow_pickle=False``
and is robust to library-version drift in a way pickled objects are not.

Layout (format_version 2/3/4):
  header                 JSON: format/library versions, model class, params,
                         loss, fitted scalars, preprocessor settings
  classes                class labels (numeric or unicode), multiclass only
  importance             per-input-feature split-gain totals
  prep__* / bin__*       preprocessor and binner arrays
  cat{j}__values/kinds   per-categorical-column category values, stringified,
                         with a parallel kind code (0=str, 1=float, 2=int) so
                         exact key types are rebuilt for dict lookups
  cat{j}__code_remap     format v3 only, optional raw-code target-order remap
  linear__* / trees__linear_*
                         format v4 only, optional local-linear leaf payload
  enc{t}__*              per-target encoder category sums/counts
  trees__*               concatenated per-tree arrays with offsets

Categories must be str, float, int, or bool; anything else raises at save
time.
"""

import json
from pathlib import Path

import numpy as np

from .binning import Binner, DEFAULT_BIN_SAMPLE_COUNT
from .losses import LOSSES, MultiSoftmax, VECTOR_LOSSES
from .preprocessing import FeaturePreprocessor
from .target_encoding import OrderedTargetEncoder, _MISSING_CATEGORY
from .tree import (
    LevelwiseTree,
    MultiNonObliviousTree,
    NonObliviousTree,
    ObliviousTree,
)

FORMAT_VERSION = 4
BASE_FORMAT_VERSION = 2

_KIND_STR = 0
_KIND_FLOAT = 1
_KIND_INT = 2
_KIND_BOOL = 3
_KIND_MISSING = 4
_KNOWN_CATEGORY_KINDS = frozenset({
    _KIND_STR,
    _KIND_FLOAT,
    _KIND_INT,
    _KIND_BOOL,
    _KIND_MISSING,
})
_LINEAR_PAYLOAD_KEYS = (
    "trees__linear_counts",
    "trees__linear_features_flat",
    "trees__linear_feature_offsets",
    "trees__linear_coefficients_flat",
    "trees__linear_coefficient_offsets",
    "linear__bin_values",
)


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


def _archive_format_version(prep, booster, wrapper_header=None):
    wrapper_params = (
        wrapper_header.get("params", {})
        if isinstance(wrapper_header, dict)
        else {}
    )
    wrapper_uses_linear_params = (
        isinstance(wrapper_params, dict)
        and (
            bool(wrapper_params.get("linear_leaves", False))
            or float(wrapper_params.get("linear_lambda", 1.0)) != 1.0
        )
    )
    if (
        wrapper_uses_linear_params
        or bool(getattr(booster, "linear_leaves", False))
        or float(getattr(booster, "linear_lambda", 1.0)) != 1.0
        or any(
            getattr(tree, "linear_coefficients", None) is not None
            for tree in booster._iter_tree_objects()
        )
    ):
        return 4
    if (
        getattr(prep, "target_ordered_cat_codes", "off") == "leaky_full"
        and getattr(prep, "include_cat_codes", False)
        and len(getattr(prep, "cat_code_remaps_", [])) > 0
    ):
        return 3
    return BASE_FORMAT_VERSION


def _validate_plain_arrays(arrays):
    for key, value in list(arrays.items()):
        arr = np.asarray(value)
        if arr.dtype.hasobject:
            raise ValueError(
                f"cannot save object-dtype array {key!r}; DarkoFit model "
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
        raise ValueError(f"invalid DarkoFit model: {name} offsets are empty")
    if not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError(
            f"invalid DarkoFit model: {name} offsets must contain integer values"
        )
    offsets = offsets.astype(np.int64, copy=False)
    if expected_count is not None and offsets.size != int(expected_count):
        raise ValueError(
            f"invalid DarkoFit model: {name} offsets length is "
            f"{offsets.size}, expected {int(expected_count)}"
        )
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError(
            f"invalid DarkoFit model: {name} offsets are not monotonic"
        )
    if total_size is not None and offsets[-1] != int(total_size):
        raise ValueError(
            f"invalid DarkoFit model: {name} offsets do not match array length"
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
    raise ValueError(f"invalid DarkoFit model: {message}")


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


def _require_unique_feature_indices(name, values, n_input_features):
    if len(values) and (
        np.any(values < 0) or np.any(values >= int(n_input_features))
    ):
        _invalid_model(f"{name} contains out-of-range feature indices")
    if np.unique(values).size != values.size:
        _invalid_model(f"{name} contains duplicate feature indices")


def _require_category_payload(name, values, kinds):
    values = _require_array_ndim(f"{name} values", values, 1)
    kinds = _require_integer_array(f"{name} kinds", kinds)
    if len(values) != len(kinds):
        _invalid_model(f"{name} values and kinds length mismatch")
    if len(kinds):
        valid = np.isin(kinds, list(_KNOWN_CATEGORY_KINDS))
        if not np.all(valid):
            bad = int(kinds[np.flatnonzero(~valid)[0]])
            _invalid_model(f"unknown category kind {bad}")
    return values, kinds


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


def _decode_categories(
    values,
    kinds,
    *,
    legacy_missing_sentinel=False,
    name="category payload",
):
    values, kinds = _require_category_payload(name, values, kinds)
    out = np.empty(len(values), dtype=object)
    for i in range(len(values)):
        s = values[i]
        k = int(kinds[i])
        if k == _KIND_STR:
            value = str(s)
            out[i] = (
                _MISSING_CATEGORY
                if legacy_missing_sentinel and value == "__nan__"
                else value
            )
        elif k == _KIND_FLOAT:
            try:
                out[i] = float(s)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid DarkoFit model: {name} float payload is invalid"
                ) from exc
        elif k == _KIND_INT:
            try:
                out[i] = int(s)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid DarkoFit model: {name} int payload is invalid"
                ) from exc
        elif k == _KIND_BOOL:
            value = str(s)
            if value not in {"True", "False"}:
                _invalid_model(
                    f"{name} bool payload must be exactly 'True' or 'False'"
                )
            out[i] = value == "True"
        elif k == _KIND_MISSING:
            out[i] = _MISSING_CATEGORY
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
        if type(inner) is LevelwiseTree:
            return "levelwise_per_class"
        first = inner  # report the per-class tree type in the error
    elif type(first) is ObliviousTree:
        return "oblivious"
    elif type(first) is NonObliviousTree:
        return "nonoblivious"
    elif type(first) is LevelwiseTree:
        return "levelwise"
    elif type(first) is MultiNonObliviousTree:
        return "multi"
    raise ValueError(
        f"cannot serialize trees of type {type(first).__name__}"
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
    linear_trees = [
        tree for tree in trees if tree.linear_coefficients is not None
    ]
    if not linear_trees:
        return
    shared_values = linear_trees[0].linear_bin_values
    for tree in linear_trees[1:]:
        if (
            tree.linear_bin_values is not shared_values
            and not np.array_equal(
                tree.linear_bin_values, shared_values, equal_nan=True
            )
        ):
            raise ValueError("linear trees do not share fitted bin values")
    counts = np.array(
        [
            0 if tree.linear_coefficients is None else len(tree.linear_features)
            for tree in trees
        ],
        dtype=np.int64,
    )
    features, feature_offsets = _concat_with_offsets(
        [
            np.empty(0, dtype=np.int64)
            if tree.linear_coefficients is None
            else tree.linear_features
            for tree in trees
        ],
        dtype=np.int64,
    )
    coefficients, coefficient_offsets = _concat_with_offsets(
        [
            np.empty(0, dtype=np.float64)
            if tree.linear_coefficients is None
            else tree.linear_coefficients.reshape(-1)
            for tree in trees
        ]
    )
    arrays["trees__linear_counts"] = counts
    arrays["trees__linear_features_flat"] = features
    arrays["trees__linear_feature_offsets"] = feature_offsets
    arrays["trees__linear_coefficients_flat"] = coefficients
    arrays["trees__linear_coefficient_offsets"] = coefficient_offsets
    arrays["linear__bin_values"] = np.asarray(shared_values, dtype=np.float64)


def _require_split_bounds(name, feats, thrs, n_bins, allow_inactive=False):
    """Reject split feature/threshold payloads outside the fitted bin space.

    Prediction kernels index ``X_binned[i, f]`` without bounds checks, and a
    legal split threshold for feature ``f`` lies in ``[0, n_bins[f] - 2]``
    (the top bin holds NaN/unseen values and always routes right), so any
    payload outside those ranges is corruption, not a valid model.
    """
    feats = np.asarray(feats)
    thrs = np.asarray(thrs)
    inactive = feats == -1
    if not allow_inactive and bool(np.any(inactive)):
        _invalid_model(f"{name} split features must be nonnegative")
    active = ~inactive
    active_feats = feats[active]
    if active_feats.size:
        if np.any((active_feats < 0) | (active_feats >= len(n_bins))):
            _invalid_model(f"{name} split features are out of range")
        active_thrs = thrs[active]
        if np.any(
            (active_thrs < 0) | (active_thrs > n_bins[active_feats] - 2)
        ):
            _invalid_model(f"{name} split thresholds are out of range")


def _require_nonoblivious_structure(features, thresholds, left, right,
                                    leaf_index, leaf_count, n_bins):
    """Reject node payloads that are not a single well-formed binary tree.

    The prediction walk (`while left_child[node] >= 0`) terminates and lands
    on a valid leaf only when every node is strictly internal or terminal,
    children descend forward from their parents (which rules out cycles and
    self-loops), every non-root node is referenced exactly once, and the leaf
    indexes are a permutation of ``range(leaf_count)``. Anything looser can
    hang prediction, wrap negative indexes, or silently misroute rows.
    """
    node_len = features.shape[0]
    internal = left >= 0
    if not np.array_equal(internal, right >= 0):
        _invalid_model(
            "nonoblivious nodes must have both children or neither"
        )
    if np.any(internal & (leaf_index != -1)):
        _invalid_model(
            "nonoblivious internal nodes must not carry leaf indexes"
        )
    leaf_mask = ~internal
    if int(np.count_nonzero(leaf_mask)) != int(leaf_count):
        _invalid_model(
            "nonoblivious terminal node count does not match leaves"
        )
    if not np.array_equal(
        np.sort(leaf_index[leaf_mask]),
        np.arange(leaf_count, dtype=np.int64),
    ):
        _invalid_model(
            "nonoblivious leaf indexes must be a permutation of the leaves"
        )
    parents = np.arange(node_len, dtype=np.int64)
    if np.any(internal & ((left <= parents) | (right <= parents))):
        _invalid_model(
            "nonoblivious child nodes must descend forward from their parent"
        )
    children = np.concatenate((left[internal], right[internal]))
    if not np.array_equal(
        np.sort(children), np.arange(1, node_len, dtype=np.int64)
    ):
        _invalid_model(
            "nonoblivious nodes must form a single tree from the root"
        )
    if np.any(features[leaf_mask] != -1):
        _invalid_model(
            "nonoblivious terminal nodes must not carry split features"
        )
    _require_split_bounds(
        "nonoblivious node", features[internal], thresholds[internal], n_bins
    )


def _unpack_oblivious(data, n_bins, format_version, n_numeric_features):
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
    present = [key in data.files for key in _LINEAR_PAYLOAD_KEYS]
    if any(present) and not all(present):
        _invalid_model("linear leaf payload is incomplete")
    linear_payload = all(present)
    if linear_payload and format_version < 4:
        _invalid_model("linear leaf payload requires format version 4")
    if linear_payload:
        linear_counts = _require_integer_array(
            "linear leaf counts", data["trees__linear_counts"]
        )
        if len(linear_counts) != len(depths) or np.any(linear_counts < 0):
            _invalid_model("linear leaf counts do not match tree count")
        linear_features = _require_integer_array(
            "linear leaf features", data["trees__linear_features_flat"]
        )
        linear_feature_offsets = _require_offsets(
            "linear leaf feature",
            data["trees__linear_feature_offsets"],
            len(linear_features),
            expected_count=len(depths) + 1,
        )
        if not np.array_equal(np.diff(linear_feature_offsets), linear_counts):
            _invalid_model("linear leaf feature offsets do not match counts")
        linear_coefficients = _require_array_ndim(
            "linear leaf coefficients",
            data["trees__linear_coefficients_flat"],
            1,
        )
        if not (
            np.issubdtype(linear_coefficients.dtype, np.integer)
            or np.issubdtype(linear_coefficients.dtype, np.floating)
        ):
            _invalid_model("linear leaf coefficients must be numeric")
        linear_coefficients = linear_coefficients.astype(
            np.float64, copy=False
        )
        if not np.all(np.isfinite(linear_coefficients)):
            _invalid_model("linear leaf coefficients must be finite")
        linear_coefficient_offsets = _require_offsets(
            "linear leaf coefficient",
            data["trees__linear_coefficient_offsets"],
            len(linear_coefficients),
            expected_count=len(depths) + 1,
        )
        linear_bin_values = _require_array_ndim(
            "linear bin values", data["linear__bin_values"], 2
        )
        if not (
            np.issubdtype(linear_bin_values.dtype, np.integer)
            or np.issubdtype(linear_bin_values.dtype, np.floating)
        ):
            _invalid_model("linear bin values must be numeric")
        linear_bin_values = linear_bin_values.astype(np.float64, copy=False)
        expected_shape = (
            len(n_bins), int(n_bins.max()) if len(n_bins) else 1
        )
        if linear_bin_values.shape != expected_shape:
            _invalid_model("linear bin values do not match fitted bin space")
        if np.any(np.isinf(linear_bin_values)):
            _invalid_model("linear bin values must not contain infinity")
        if n_numeric_features < len(n_bins) and not np.array_equal(
            linear_bin_values[n_numeric_features:],
            np.zeros_like(linear_bin_values[n_numeric_features:]),
        ):
            _invalid_model("non-numeric linear bin values must be zero")
        for feature in range(n_numeric_features):
            missing_bin = int(n_bins[feature]) - 1
            if not np.isnan(linear_bin_values[feature, missing_bin]):
                _invalid_model("numeric missing-bin linear value must be NaN")
            if not np.all(np.isfinite(linear_bin_values[feature, :missing_bin])):
                _invalid_model("numeric linear bin values must be finite")
    else:
        linear_counts = np.zeros(len(depths), dtype=np.int64)
        linear_features = np.empty(0, dtype=np.int64)
        linear_feature_offsets = np.zeros(len(depths) + 1, dtype=np.int64)
        linear_coefficients = np.empty(0, dtype=np.float64)
        linear_coefficient_offsets = np.zeros(len(depths) + 1, dtype=np.int64)
        linear_bin_values = None
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
        _require_split_bounds("oblivious", feats[s0:s1], thrs[s0:s1], n_bins)
        linear_count = int(linear_counts[t])
        feature_start = int(linear_feature_offsets[t])
        feature_stop = int(linear_feature_offsets[t + 1])
        coefficient_start = int(linear_coefficient_offsets[t])
        coefficient_stop = int(linear_coefficient_offsets[t + 1])
        expected_coefficients = (
            0 if linear_count == 0 else (1 << depth) * (1 + linear_count)
        )
        if coefficient_stop - coefficient_start != expected_coefficients:
            _invalid_model(
                "linear leaf coefficient payload does not match tree shape"
            )
        tree_linear_features = None
        tree_linear_coefficients = None
        if linear_count:
            if depth == 0:
                _invalid_model("depth-zero tree cannot have linear leaves")
            tree_linear_features = linear_features[
                feature_start:feature_stop
            ].copy()
            if len(np.unique(tree_linear_features)) != linear_count:
                _invalid_model("linear leaf features must be unique")
            if np.any(
                (tree_linear_features < 0)
                | (tree_linear_features >= int(n_numeric_features))
            ):
                _invalid_model("linear leaf features must be numeric")
            if not set(tree_linear_features.tolist()).issubset(
                set(feats[s0:s1].tolist())
            ):
                _invalid_model("linear leaf feature was not used by its tree")
            tree_linear_coefficients = linear_coefficients[
                coefficient_start:coefficient_stop
            ].reshape(1 << depth, 1 + linear_count).copy()
        trees.append(ObliviousTree(
            feats[s0:s1].copy(), thrs[s0:s1].copy(),
            values[vo[t]:vo[t + 1]].copy(), gains[s0:s1].copy(),
            linear_features=tree_linear_features,
            linear_coefficients=tree_linear_coefficients,
            linear_bin_values=(
                None if tree_linear_coefficients is None else linear_bin_values
            ),
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


def _unpack_nonoblivious(data, cls, n_bins, expected_value_width=None):
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
        _require_nonoblivious_structure(
            features[n0:n1], thresholds[n0:n1], left_child[n0:n1],
            right_child[n0:n1], leaf_slice, leaf_count, n_bins,
        )
        _require_split_bounds(
            "nonoblivious", splits_feat[s0:s1], splits_thr[s0:s1], n_bins
        )
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


def _pack_levelwise(trees, arrays):
    n_trees = len(trees)
    depths = np.empty(n_trees, dtype=np.int64)
    node_widths = np.empty(n_trees, dtype=np.int64)
    node_offsets = np.zeros(n_trees + 1, dtype=np.int64)
    node_features = []
    node_thresholds = []
    total_nodes = 0

    for i, tree in enumerate(trees):
        features = np.asarray(tree.node_features, dtype=np.int64)
        thresholds = np.asarray(tree.node_thresholds, dtype=np.int64)
        if features.ndim != 2 or thresholds.ndim != 2:
            raise ValueError("levelwise node arrays must be 2-dimensional")
        if features.shape != thresholds.shape:
            raise ValueError("levelwise node feature/threshold shape mismatch")
        depth = int(tree.depth)
        if depth != features.shape[0]:
            raise ValueError("levelwise node rows do not match tree depth")
        values = np.asarray(tree.values)
        splits_feat = np.asarray(tree.splits_feat)
        splits_thr = np.asarray(tree.splits_thr)
        gains = np.asarray(tree.gains)
        if values.ndim != 1:
            raise ValueError("levelwise values must be 1-dimensional")
        if values.shape[0] != (1 << depth):
            raise ValueError("levelwise values do not match tree depth")
        if not (
            splits_feat.ndim == splits_thr.ndim == gains.ndim == 1
            and len(splits_feat) == len(splits_thr) == len(gains)
        ):
            raise ValueError("levelwise split arrays must have matching length")
        if len(splits_feat) > ((1 << depth) - 1):
            raise ValueError("levelwise split arrays exceed tree depth")
        depths[i] = depth
        node_widths[i] = features.shape[1]
        total_nodes += features.size
        node_offsets[i + 1] = total_nodes
        node_features.append(features.ravel())
        node_thresholds.append(thresholds.ravel())

    arrays["trees__depths"] = depths
    arrays["trees__node_widths"] = node_widths
    arrays["trees__node_offsets"] = node_offsets
    arrays["trees__node_features_flat"] = (
        np.concatenate(node_features) if total_nodes
        else np.empty(0, dtype=np.int64)
    )
    arrays["trees__node_thresholds_flat"] = (
        np.concatenate(node_thresholds) if total_nodes
        else np.empty(0, dtype=np.int64)
    )
    arrays["trees__values_flat"], arrays["trees__value_offsets"] = (
        _concat_with_offsets([t.values for t in trees])
    )
    arrays["trees__splits_feat_flat"], arrays["trees__split_offsets"] = (
        _concat_with_offsets([t.splits_feat for t in trees], dtype=np.int64)
    )
    arrays["trees__splits_thr_flat"], _ = _concat_with_offsets(
        [t.splits_thr for t in trees], dtype=np.int64
    )
    arrays["trees__gains_flat"], _ = _concat_with_offsets(
        [t.gains for t in trees]
    )


def _unpack_levelwise(data, n_bins):
    depths = _require_integer_array("levelwise depths", data["trees__depths"])
    node_widths = _require_integer_array(
        "levelwise node_widths", data["trees__node_widths"]
    )
    if len(node_widths) != len(depths):
        _invalid_model("levelwise node_widths length does not match depths")
    node_features = _require_integer_array(
        "levelwise node features", data["trees__node_features_flat"]
    )
    node_thresholds = _require_integer_array(
        "levelwise node thresholds", data["trees__node_thresholds_flat"]
    )
    values = _require_array_ndim("levelwise values", data["trees__values_flat"], 1)
    splits_feat = _require_integer_array(
        "levelwise split features", data["trees__splits_feat_flat"]
    )
    splits_thr = _require_integer_array(
        "levelwise split thresholds", data["trees__splits_thr_flat"]
    )
    gains = _require_array_ndim("levelwise gains", data["trees__gains_flat"], 1)
    no = _require_same_offsets(
        "levelwise node",
        data["trees__node_offsets"],
        (("features", node_features), ("thresholds", node_thresholds)),
        expected_count=len(depths) + 1,
    )
    vo = _require_offsets(
        "levelwise value",
        data["trees__value_offsets"],
        len(values),
        expected_count=len(depths) + 1,
    )
    so = _require_same_offsets(
        "levelwise split",
        data["trees__split_offsets"],
        (("features", splits_feat), ("thresholds", splits_thr), ("gains", gains)),
        expected_count=len(depths) + 1,
    )

    trees = []
    for t in range(len(depths)):
        depth = int(depths[t])
        width = int(node_widths[t])
        if depth < 0:
            _invalid_model("levelwise tree depth is negative")
        if width < 0:
            _invalid_model("levelwise node width is negative")
        if depth > 0 and width < (1 << (depth - 1)):
            _invalid_model("levelwise node width does not match depth")
        n0, n1 = no[t], no[t + 1]
        s0, s1 = so[t], so[t + 1]
        node_len = int(n1 - n0)
        split_len = int(s1 - s0)
        value_len = int(vo[t + 1] - vo[t])
        if node_len != depth * width:
            _invalid_model("levelwise node payload does not match depth/width")
        if value_len != (1 << depth):
            _invalid_model("levelwise value payload does not match depth")
        if split_len > ((1 << depth) - 1):
            _invalid_model("levelwise split payload exceeds depth")
        if depth == 0 and split_len:
            _invalid_model("levelwise depth-zero tree has split payload")
        features = node_features[n0:n1].reshape(depth, width).copy()
        thresholds = node_thresholds[n0:n1].reshape(depth, width).copy()
        # Inactive node-table slots hold -1 and are skipped by the predict
        # kernels; active slots must index real binned features/thresholds.
        _require_split_bounds(
            "levelwise node", features.ravel(), thresholds.ravel(), n_bins,
            allow_inactive=True,
        )
        _require_split_bounds(
            "levelwise", splits_feat[s0:s1], splits_thr[s0:s1], n_bins
        )
        trees.append(LevelwiseTree(
            features,
            thresholds,
            values[vo[t]:vo[t + 1]].copy(),
            splits_feat[s0:s1].copy(),
            splits_thr[s0:s1].copy(),
            gains[s0:s1].copy(),
        ))
    return trees


def _validate_boosting_round_count(kind, trees, header):
    best_iteration = int(header["best_iteration"])
    if best_iteration < 0:
        _invalid_model("best_iteration must be nonnegative")
    if kind == "empty":
        if best_iteration != 0:
            _invalid_model("empty tree kind does not match best_iteration")
        return
    if kind in {
        "oblivious_per_class",
        "nonoblivious_per_class",
        "levelwise_per_class",
    }:
        n_rounds = int(header["n_rounds"])
        if n_rounds != best_iteration:
            _invalid_model(
                "per-class tree count n_rounds does not match best_iteration"
            )
        loaded_rounds = n_rounds
    else:
        loaded_rounds = len(trees)
    if loaded_rounds != best_iteration:
        _invalid_model("tree count does not match best_iteration")


def _restore_training_metadata(booster):
    """Restore fit diagnostics without serializing callbacks or deadlines."""
    if not isinstance(booster.auto_params_, dict):
        _invalid_model("auto_params must be an object")
    training = booster.auto_params_.get("training")
    if training is None:
        completed = max(
            len(booster.trees_),
            len(getattr(booster, "train_history_", [])),
            len(getattr(booster, "valid_history_", [])),
        )
        booster.stop_reason_ = "legacy_unknown"
        booster.iterations_attempted_ = None
        booster.rounds_completed_ = int(completed)
        booster.training_metadata_ = {
            "stop_reason": "legacy_unknown",
            "iterations_attempted": None,
            "rounds_completed": int(completed),
            "rounds_retained": int(len(booster.trees_)),
        }
        return
    if not isinstance(training, dict):
        _invalid_model("training metadata must be an object")

    def nonnegative_int(name):
        value = training.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _invalid_model(
                f"training metadata {name} must be a nonnegative integer"
            )
        return int(value)

    requested = nonnegative_int("iterations_requested")
    attempted = nonnegative_int("iterations_attempted")
    completed = nonnegative_int("rounds_completed")
    retained = nonnegative_int("rounds_retained")
    if requested != int(booster.iterations):
        _invalid_model(
            "training metadata requested rounds do not match booster params"
        )
    if attempted > requested:
        _invalid_model("training metadata attempted rounds exceed requested rounds")
    if completed > attempted:
        _invalid_model("training metadata completed rounds exceed attempted rounds")
    if retained > completed:
        _invalid_model("training metadata retained rounds exceed completed rounds")
    if retained != len(booster.trees_):
        _invalid_model("training metadata retained rounds do not match tree payload")

    reason = training.get("stop_reason")
    if not isinstance(reason, str) or not reason:
        _invalid_model("training metadata stop_reason must be a nonempty string")
    best_prefix = training.get("best_prefix_round")
    if best_prefix is not None and (
        isinstance(best_prefix, bool)
        or not isinstance(best_prefix, int)
        or best_prefix < 1
        or best_prefix > completed
    ):
        _invalid_model(
            "training metadata best_prefix_round is outside completed rounds"
        )
    for name in ("best_model_truncated", "time_limit_is_soft"):
        if not isinstance(training.get(name), bool):
            _invalid_model(f"training metadata {name} must be a boolean")
    if training["best_model_truncated"] != (retained != completed):
        _invalid_model("training metadata truncation flag is inconsistent")
    if training["time_limit_is_soft"] != (reason == "time_limit"):
        _invalid_model("training metadata time-limit flag is inconsistent")
    stop_check_policy = training.get("stop_check_policy")
    if stop_check_policy not in {"none", "before_iteration"}:
        _invalid_model(
            "training metadata stop_check_policy must be 'none' or "
            "'before_iteration'"
        )
    if reason == "iteration_limit" and attempted != requested:
        _invalid_model(
            "training metadata iteration_limit requires all requested rounds "
            "to be attempted"
        )
    if reason == "time_limit" and attempted >= requested:
        _invalid_model(
            "training metadata time_limit requires stopping before the "
            "iteration limit"
        )
    if reason == "time_limit" and stop_check_policy != "before_iteration":
        _invalid_model(
            "training metadata time_limit requires before_iteration stop checks"
        )
    if reason == "no_split" and attempted <= completed:
        _invalid_model(
            "training metadata no_split requires a failed attempted round"
        )
    if reason == "early_stopping" and completed == 0:
        _invalid_model(
            "training metadata early_stopping requires a completed round"
        )

    booster.stop_reason_ = reason
    booster.iterations_attempted_ = attempted
    booster.rounds_completed_ = completed
    booster.training_metadata_ = dict(training)


def _validate_preprocessor_payload(data, prep_cfg, n_input_features):
    n_input_features = int(n_input_features)
    if n_input_features <= 0:
        _invalid_model("n_input_features must be positive")
    num_features = _require_integer_array(
        "preprocessor numeric features", data["prep__num_features"]
    )
    cat_features = _require_integer_array(
        "preprocessor categorical features", data["prep__cat_features"]
    )
    feature_map = _require_integer_array(
        "preprocessor feature_map", data["prep__feature_map"]
    )
    _require_unique_feature_indices(
        "preprocessor numeric features", num_features, n_input_features
    )
    _require_unique_feature_indices(
        "preprocessor categorical features", cat_features, n_input_features
    )
    if np.intersect1d(num_features, cat_features).size:
        _invalid_model("preprocessor numeric and categorical features overlap")
    combined = np.sort(np.concatenate((num_features, cat_features)))
    if not np.array_equal(combined, np.arange(n_input_features, dtype=np.int64)):
        _invalid_model("preprocessor feature lists do not cover input features")
    if len(feature_map) and (
        np.any(feature_map < 0) or np.any(feature_map >= n_input_features)
    ):
        _invalid_model("preprocessor feature_map contains out-of-range features")

    n_encoders = int(prep_cfg["n_encoders"])
    if n_encoders < 0:
        _invalid_model("preprocessor encoder count must be nonnegative")
    for key in ("encoder_priors", "encoder_smoothings", "encoder_modes"):
        if len(prep_cfg[key]) != n_encoders:
            _invalid_model(f"preprocessor {key} length does not match n_encoders")
    if "encoder_ts_permutations" in prep_cfg:
        if len(prep_cfg["encoder_ts_permutations"]) != n_encoders:
            _invalid_model(
                "preprocessor encoder_ts_permutations length does not "
                "match n_encoders"
            )
    legacy_aliases = prep_cfg.get("legacy_missing_aliases")
    if legacy_aliases is not None and len(legacy_aliases) != len(cat_features):
        _invalid_model(
            "preprocessor legacy_missing_aliases length does not match categories"
        )
    if len(cat_features) == 0 and n_encoders != 0:
        _invalid_model("preprocessor encoders require categorical features")

    borders = _require_array_ndim("binner borders", data["bin__borders_flat"], 1)
    if not np.issubdtype(borders.dtype, np.number):
        _invalid_model("binner borders must be numeric")
    borders = borders.astype(np.float64, copy=False)
    if not np.all(np.isfinite(borders)):
        _invalid_model("binner borders must be finite")
    n_bins = _require_integer_array("binner n_bins", data["bin__n_bins"])
    block_widths = _require_integer_array(
        "binner block_widths", data["bin__block_widths"]
    )
    if len(n_bins) != len(feature_map):
        _invalid_model("binner n_bins length does not match feature_map")
    if np.any(n_bins < 2):
        _invalid_model("binner n_bins must be at least 2")
    offsets = _require_offsets(
        "binner border",
        data["bin__border_offsets"],
        len(borders),
        expected_count=len(n_bins) + 1,
    )
    border_lengths = np.diff(offsets)
    if not np.array_equal(n_bins, border_lengths + 2):
        _invalid_model("binner n_bins do not match border payload")
    for start, stop in zip(offsets[:-1], offsets[1:]):
        if stop - start > 1 and np.any(np.diff(borders[start:stop]) <= 0.0):
            _invalid_model("binner borders must be strictly increasing")

    expected_widths = [len(num_features)]
    if len(cat_features):
        if prep_cfg["include_cat_codes"]:
            expected_widths.append(len(cat_features))
        expected_widths.extend([len(cat_features)] * n_encoders)
    expected_widths = np.asarray(expected_widths, dtype=np.int64)
    if not np.array_equal(block_widths, expected_widths):
        _invalid_model("binner block_widths do not match preprocessor layout")
    if int(np.sum(block_widths)) != len(n_bins):
        _invalid_model("binner block_widths do not sum to n_bins")

    expected_feature_map = list(num_features)
    if len(cat_features):
        if prep_cfg["include_cat_codes"]:
            expected_feature_map.extend(cat_features)
        for _ in range(n_encoders):
            expected_feature_map.extend(cat_features)
    expected_feature_map = np.asarray(expected_feature_map, dtype=np.int64)
    if not np.array_equal(feature_map, expected_feature_map):
        _invalid_model("preprocessor feature_map does not match fitted layout")

    binner = Binner(prep_cfg["max_bins"])
    binner._borders_flat_ = borders
    binner._border_offsets_ = offsets
    binner.n_bins_ = n_bins
    binner._block_widths_ = block_widths.tolist()
    binner.borders_ = [
        borders[offsets[f]:offsets[f + 1]].copy()
        for f in range(len(n_bins))
    ]
    return num_features.tolist(), cat_features.tolist(), feature_map, binner


def save_booster(booster, path, wrapper_header=None, wrapper_arrays=None):
    """Serialize a fitted GradientBoosting / MulticlassBoosting to ``path``.

    ``wrapper_header`` / ``wrapper_arrays`` let the sklearn wrappers attach
    their own state (e.g. the binary classifier's original class labels)
    under the ``wrapper`` header key and ``wrapper__*`` array keys.
    """
    from . import __version__
    from .booster import (
        DistributionalBoosting,
        GradientBoosting,
        MulticlassBoosting,
    )

    if not hasattr(booster, "trees_"):
        raise ValueError("cannot save an unfitted model")
    prep = booster.prep_
    arrays = {}
    header = {
        "format_version": _archive_format_version(
            prep, booster, wrapper_header
        ),
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
            "ts_permutations": prep.ts_permutations,
            "target_ordered_cat_codes": prep.target_ordered_cat_codes,
            "target_ordered_cat_code_policy": (
                "full_target_smoothed_leaky_opt_in"
                if prep.target_ordered_cat_codes == "leaky_full"
                else "off"
            ),
            "bin_sample_count": prep.bin_sample_count,
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
            "encoder_ts_permutations": [
                int(e.ts_permutations) for e in getattr(prep, "encoders_", [])
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
    elif isinstance(booster, DistributionalBoosting):
        header["init"] = [float(v) for v in booster.init_]
        header["n_outputs"] = int(booster.n_outputs_)
        header["loss_name"] = booster.loss_name
        header["loss_kwargs"] = _jsonify(booster.loss_kwargs)
        header["target_transform"] = _jsonify(
            getattr(
                booster,
                "target_transform_",
                {
                    "enabled": False,
                    "mean": 0.0,
                    "scale": 1.0,
                    "basis": "target",
                },
            )
        )
        loss_state = getattr(getattr(booster, "loss_", None), "state_", None)
        if loss_state:
            header["loss_state"] = _jsonify(loss_state)
    elif isinstance(booster, GradientBoosting):
        header["init"] = float(booster.init_)
        header["loss_name"] = booster.loss_name
        header["loss_kwargs"] = _jsonify(booster.loss_kwargs)
    else:
        raise TypeError(f"unsupported booster type {type(booster).__name__}")

    param_names = (
        "iterations", "learning_rate", "depth", "l2_leaf_reg",
        "max_bins", "subsample", "colsample", "cat_smoothing",
        "early_stopping_rounds", "early_stopping_min_delta", "eval_metric",
        "min_child_weight", "min_child_samples", "min_gain_to_split",
        "num_leaves", "thread_count", "random_state", "ordered_boosting",
        "tree_mode", "sampling", "top_rate", "other_rate",
        "multiclass_tree_strategy", "eval_train_loss", "bin_sample_count",
        "histogram_parallelism", "use_best_model", "bootstrap_type",
        "bagging_temperature", "mvs_reg", "random_strength",
        "diagnostic_warnings", "histogram_dtype", "leaf_dtype",
        "ts_permutations", "target_ordered_cat_codes",
        "rho_learning_rate_multiplier", "rho_l2_leaf_reg_multiplier",
    )
    if header["format_version"] >= 4:
        param_names += ("linear_leaves", "linear_lambda")
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
    if kind in {
        "oblivious_per_class",
        "nonoblivious_per_class",
        "levelwise_per_class",
    }:
        header["n_rounds"] = len(booster.trees_)
        flat_trees = [t for round_trees in booster.trees_
                      for t in round_trees]
    else:
        flat_trees = list(booster.trees_)
    if kind.startswith("oblivious"):
        _pack_oblivious(flat_trees, arrays)
    elif kind.startswith("levelwise"):
        _pack_levelwise(flat_trees, arrays)
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
    if prep.target_ordered_cat_codes == "leaky_full" and prep.include_cat_codes:
        remaps = list(getattr(prep, "cat_code_remaps_", []))
        if len(remaps) != len(getattr(prep, "cat_categories_", [])):
            raise ValueError(
                "target-ordered raw category code remaps do not match "
                "categorical feature count"
            )
        for j, remap in enumerate(remaps):
            arrays[f"cat{j}__code_remap"] = np.asarray(remap, dtype=np.int64)
    for t, enc in enumerate(getattr(prep, "encoders_", [])):
        sums_flat, offsets = _concat_with_offsets(enc.sums_)
        counts_flat, _ = _concat_with_offsets(enc.counts_)
        arrays[f"enc{t}__sums_flat"] = sums_flat
        arrays[f"enc{t}__counts_flat"] = counts_flat
        arrays[f"enc{t}__offsets"] = offsets

    if wrapper_header:
        wrapper_header = dict(wrapper_header)
        wrapper_params = wrapper_header.get("params")
        if header["format_version"] < 4 and isinstance(wrapper_params, dict):
            wrapper_params = dict(wrapper_params)
            wrapper_params.pop("linear_leaves", None)
            wrapper_params.pop("linear_lambda", None)
            wrapper_header["params"] = wrapper_params
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
    from .booster import (
        DistributionalBoosting,
        GradientBoosting,
        MulticlassBoosting,
        _normalize_eval_metric,
        _normalize_tree_mode,
    )

    wrapper_header = {}
    wrapper_arrays = {}
    try:
        archive = np.load(_load_path(path), allow_pickle=False)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"{path!r} is not a DarkoFit model archive") from exc
    with archive as data:
        try:
            header = json.loads(str(data["header"]))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{path!r} is not a DarkoFit model archive"
            ) from exc
        format_version = int(header["format_version"])
        if format_version > FORMAT_VERSION:
            raise ValueError(
                f"model format {format_version} is newer than this "
                f"library understands ({FORMAT_VERSION})"
            )
        model_class = header.get("model_class")
        params = header.get("params")
        if not isinstance(params, dict):
            _invalid_model("params header must be an object")
        loss_name = header.get("loss_name")
        if model_class == "GradientBoosting":
            if not isinstance(loss_name, str) or loss_name not in LOSSES:
                _invalid_model(f"unknown loss {loss_name!r}")
            try:
                booster = GradientBoosting(
                    loss=loss_name, loss_kwargs=header["loss_kwargs"],
                    **params
                )
                booster.loss_ = LOSSES[loss_name](**header["loss_kwargs"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                _invalid_model(f"invalid booster params: {exc}")
            booster.init_ = header["init"]
        elif model_class == "MulticlassBoosting":
            try:
                booster = MulticlassBoosting(**params)
            except (TypeError, ValueError, OverflowError) as exc:
                _invalid_model(f"invalid booster params: {exc}")
            booster.init_ = np.array(header["init"], dtype=np.float64)
            booster.n_classes_ = header["n_classes"]
            booster.loss_ = MultiSoftmax(booster.n_classes_)
            classes = data["classes"]
            if "classes_kinds" in data:
                classes = _decode_categories(
                    classes, data["classes_kinds"],
                    name="multiclass classes",
                )
            booster.classes_ = classes
        elif model_class == "DistributionalBoosting":
            if (
                not isinstance(loss_name, str)
                or loss_name not in VECTOR_LOSSES
            ):
                _invalid_model(f"unknown distributional loss {loss_name!r}")
            try:
                booster = DistributionalBoosting(
                    loss=loss_name, loss_kwargs=header["loss_kwargs"],
                    **params
                )
                booster.loss_ = VECTOR_LOSSES[loss_name](
                    **header["loss_kwargs"]
                )
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                _invalid_model(f"invalid booster params: {exc}")
            booster.init_ = np.array(header["init"], dtype=np.float64)
            booster.n_outputs_ = int(header["n_outputs"])
            target_transform = header.get("target_transform") or {
                "enabled": False,
                "mean": 0.0,
                "scale": 1.0,
                "basis": "target",
            }
            if not isinstance(target_transform, dict):
                _invalid_model("distributional target transform must be an object")
            enabled = bool(target_transform.get("enabled", False))
            mean = float(target_transform.get("mean", 0.0))
            scale = float(target_transform.get("scale", 1.0))
            if not np.isfinite(mean):
                _invalid_model("distributional target transform mean is not finite")
            if not np.isfinite(scale) or scale <= 0.0:
                _invalid_model("distributional target transform scale must be positive")
            booster.target_transform_ = {
                "enabled": enabled,
                "mean": mean,
                "scale": scale,
                "basis": str(target_transform.get("basis", "target")),
            }
            loss_state = header.get("loss_state")
            if loss_state:
                booster.loss_.state_ = loss_state
            expected_outputs = int(getattr(booster.loss_, "n_outputs", 0))
            if booster.n_outputs_ != expected_outputs:
                _invalid_model(
                    "distributional n_outputs does not match loss "
                    f"{header['loss_name']!r}"
                )
        else:
            raise ValueError(f"unknown model class {model_class!r}")

        booster.lr_ = header["lr"]
        booster.best_iteration_ = header["best_iteration"]
        booster.best_score_ = header["best_score"]
        booster.auto_params_ = header.get("auto_params", {})
        if not isinstance(booster.auto_params_, dict):
            _invalid_model("auto_params must be an object")
        saved_linear_metadata = booster.auto_params_.get("linear_leaves")
        if saved_linear_metadata is not None:
            if not isinstance(saved_linear_metadata, dict):
                _invalid_model("linear leaf metadata must be an object")
            for name in ("requested", "active"):
                if not isinstance(saved_linear_metadata.get(name), bool):
                    _invalid_model(
                        f"linear leaf metadata {name} must be a boolean"
                    )
            for name in (
                "numeric_feature_count",
                "linear_tree_count",
                "linear_leaf_count",
            ):
                value = saved_linear_metadata.get(name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    _invalid_model(
                        f"linear leaf metadata {name} must be nonnegative"
                    )
        linear_payload_members = [
            key in data.files for key in _LINEAR_PAYLOAD_KEYS
        ]
        linear_payload_complete = all(linear_payload_members)
        saved_linear_active = (
            isinstance(saved_linear_metadata, dict)
            and saved_linear_metadata.get("active") is True
        )
        if saved_linear_active and not linear_payload_complete:
            _invalid_model(
                "active linear leaf metadata requires a complete payload"
            )
        if linear_payload_complete and not saved_linear_active:
            _invalid_model(
                "linear leaf payload requires active fitted metadata"
            )
        linear_requested = bool(getattr(booster, "linear_leaves", False))
        if (
            isinstance(saved_linear_metadata, dict)
            and saved_linear_metadata["requested"] != linear_requested
        ):
            _invalid_model(
                "linear leaf metadata requested state disagrees with params"
            )
        if linear_requested:
            if not isinstance(booster, GradientBoosting):
                _invalid_model(
                    "linear_leaves=True is only valid for scalar regression"
                )
            if getattr(booster, "loss_name", None) != "RMSE":
                _invalid_model("linear_leaves=True requires loss='RMSE'")
            try:
                loaded_tree_mode = _normalize_tree_mode(booster.tree_mode)
            except (TypeError, ValueError) as exc:
                _invalid_model(f"invalid linear leaf tree mode: {exc}")
            if loaded_tree_mode != "catboost":
                _invalid_model(
                    "linear_leaves=True requires tree_mode='catboost'"
                )
            if (
                booster.ordered_boosting != "auto"
                and bool(booster.ordered_boosting)
            ):
                _invalid_model(
                    "linear_leaves=True is incompatible with "
                    "ordered_boosting=True"
                )
            tree_metadata = booster.auto_params_.get("tree", {})
            if isinstance(tree_metadata, dict) and bool(
                tree_metadata.get("ordered_boosting", False)
            ):
                _invalid_model(
                    "linear leaf archive resolved ordered boosting on"
                )
            booster.tree_mode_ = loaded_tree_mode
            booster.ordered_boosting_ = False
        if isinstance(booster, DistributionalBoosting):
            metric = (
                booster.auto_params_
                .get("distributional", {})
                .get("eval_metric")
            )
            booster.eval_metric_ = (
                metric
                if metric is not None
                else _normalize_eval_metric(
                    params.get("eval_metric"), booster.loss_name
                )
            )
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

        # ---- preprocessor payload -------------------------------------------
        # Validated before the trees so tree split feature ids and thresholds
        # can be bounds-checked against the fitted binner; the prediction
        # kernels index X_binned without bounds checks, so out-of-range
        # payloads must be rejected here, not discovered at inference time.
        prep_cfg = header["prep"]
        (
            num_features,
            cat_features,
            feature_map,
            binner,
        ) = _validate_preprocessor_payload(
            data, prep_cfg, header["n_input_features"]
        )
        n_bins = np.asarray(binner.n_bins_, dtype=np.int64)

        # ---- trees ----------------------------------------------------------
        kind = header["tree_kind"]
        if kind not in {
            "empty",
            "oblivious",
            "nonoblivious",
            "multi",
            "oblivious_per_class",
            "nonoblivious_per_class",
            "levelwise",
            "levelwise_per_class",
        }:
            _invalid_model(f"unknown tree kind {kind!r}")
        if kind == "empty":
            if any(key.startswith("trees__") for key in data.files):
                _invalid_model("empty tree kind has tree payload")
            trees = []
        elif kind.startswith("oblivious"):
            trees = _unpack_oblivious(
                data,
                n_bins,
                format_version,
                len(num_features),
            )
        elif kind.startswith("levelwise"):
            trees = _unpack_levelwise(data, n_bins)
        elif kind == "multi":
            if "n_outputs" in header:
                value_width = int(header["n_outputs"])
            else:
                value_width = int(header["n_classes"])
            trees = _unpack_nonoblivious(
                data, MultiNonObliviousTree, n_bins,
                expected_value_width=value_width,
            )
        else:
            trees = _unpack_nonoblivious(data, NonObliviousTree, n_bins)
        _validate_boosting_round_count(kind, trees, header)
        if kind in {
            "oblivious_per_class",
            "nonoblivious_per_class",
            "levelwise_per_class",
        }:
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
        prep = FeaturePreprocessor(
            prep_cfg["max_bins"], prep_cfg["cat_smoothing"],
            params.get("random_state"),
            include_cat_codes=prep_cfg["include_cat_codes"],
            target_encoding_mode=prep_cfg["target_encoding_mode"],
            target_encoding_folds=prep_cfg["target_encoding_folds"],
            ts_permutations=prep_cfg.get("ts_permutations", 1),
            target_ordered_cat_codes=prep_cfg.get(
                "target_ordered_cat_codes", "off"
            ),
            bin_sample_count=prep_cfg.get(
                "bin_sample_count", DEFAULT_BIN_SAMPLE_COUNT
            ),
        )
        prep.num_features_ = num_features
        prep.cat_features_ = cat_features
        prep.feature_map_ = feature_map
        prep.n_input_features_ = header["n_input_features"]
        prep._cat_indexes_ = {}
        prep.cat_categories_ = []
        prep.cat_maps_ = []
        prep.cat_code_remaps_ = []
        legacy_aliases = prep_cfg.get("legacy_missing_aliases", [])
        for j in range(len(prep.cat_features_)):
            cats = _decode_categories(
                data[f"cat{j}__values"], data[f"cat{j}__kinds"],
                legacy_missing_sentinel=(format_version <= 1),
                name=f"categorical feature {j}",
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
            if (
                prep.target_ordered_cat_codes == "leaky_full"
                and prep.include_cat_codes
            ):
                if format_version < 3:
                    _invalid_model(
                        "target-ordered raw category codes require "
                        "format version 3"
                    )
                remap_key = f"cat{j}__code_remap"
                if remap_key not in data.files:
                    _invalid_model("missing categorical code remap payload")
                remap = _require_integer_array(
                    f"categorical feature {j} code remap",
                    data[remap_key],
                )
                if remap.ndim != 1 or len(remap) != len(cats):
                    _invalid_model(
                        "categorical code remap length does not match "
                        "category count"
                    )
                if not np.array_equal(
                    np.sort(remap), np.arange(len(cats), dtype=np.int64)
                ):
                    _invalid_model("categorical code remap must be a permutation")
                prep.cat_code_remaps_.append(remap.copy())
        prep.encoders_ = []
        for t in range(prep_cfg["n_encoders"]):
            enc = OrderedTargetEncoder(
                prep_cfg["encoder_smoothings"][t],
                mode=prep_cfg["encoder_modes"][t],
                ts_permutations=prep_cfg.get(
                    "encoder_ts_permutations", [1] * prep_cfg["n_encoders"]
                )[t],
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
            # A self-consistent but truncated encoder payload would otherwise
            # load cleanly and silently prior-encode known categories whose
            # codes fall at or beyond the truncated n_cat_ at predict time.
            for j in range(len(prep.cat_features_)):
                if len(enc.sums_[j]) != len(prep.cat_categories_[j]):
                    _invalid_model(
                        f"encoder {t} statistics do not match the "
                        "categorical payload"
                    )
            enc.n_cat_ = [len(s) for s in enc.sums_]
            prep.encoders_.append(enc)

        prep.binner_ = binner
        prep.n_bins_ = binner.n_bins_
        booster.prep_ = prep
        linear_trees = [
            tree
            for tree in booster._iter_tree_objects()
            if getattr(tree, "linear_coefficients", None) is not None
        ]
        if saved_linear_active != bool(linear_trees):
            _invalid_model(
                "linear leaf active metadata disagrees with decoded trees"
            )
        if linear_trees:
            if not isinstance(booster, GradientBoosting):
                _invalid_model(
                    "linear leaf payload is only valid for scalar regression"
                )
            if not bool(getattr(booster, "linear_leaves", False)):
                _invalid_model(
                    "linear leaf payload requires linear_leaves=True"
                )
            booster.linear_leaves_active_ = True
            booster.linear_leaves_inactive_reason_ = None
            booster.linear_bin_values_ = linear_trees[0].linear_bin_values
            booster.linear_numeric_features_ = np.zeros(
                len(n_bins), dtype=np.bool_
            )
            booster.linear_numeric_features_[:len(num_features)] = True
            booster.linear_tree_count_ = len(linear_trees)
            booster.linear_leaf_count_ = int(sum(
                tree.linear_coefficients.shape[0]
                for tree in linear_trees
            ))
            for name, observed in (
                ("numeric_feature_count", len(num_features)),
                ("linear_tree_count", booster.linear_tree_count_),
                ("linear_leaf_count", booster.linear_leaf_count_),
            ):
                expected = saved_linear_metadata.get(name)
                if (
                    isinstance(expected, bool)
                    or not isinstance(expected, int)
                    or int(expected) != int(observed)
                ):
                    _invalid_model(
                        f"linear leaf metadata {name} does not match payload"
                    )
        else:
            booster.linear_leaves_active_ = False
            saved_inactive_reason = (
                saved_linear_metadata.get("inactive_reason")
                if isinstance(saved_linear_metadata, dict)
                else None
            )
            booster.linear_leaves_inactive_reason_ = (
                saved_inactive_reason
                if (
                    bool(getattr(booster, "linear_leaves", False))
                    and isinstance(saved_inactive_reason, str)
                    and saved_inactive_reason
                )
                else (
                    "no_retained_linear_trees"
                    if bool(getattr(booster, "linear_leaves", False))
                    else "disabled"
                )
            )
            booster.linear_bin_values_ = None
            saved_numeric_feature_count = 0
            if isinstance(saved_linear_metadata, dict):
                saved_numeric_feature_count = int(
                    saved_linear_metadata["numeric_feature_count"]
                )
                if saved_numeric_feature_count > len(num_features):
                    _invalid_model(
                        "linear leaf numeric feature metadata exceeds the "
                        "fitted preprocessor"
                    )
            booster.linear_numeric_features_ = np.zeros(
                len(n_bins), dtype=np.bool_
            )
            booster.linear_numeric_features_[:saved_numeric_feature_count] = True
            booster.linear_tree_count_ = 0
            booster.linear_leaf_count_ = 0
            if isinstance(saved_linear_metadata, dict):
                for name in ("linear_tree_count", "linear_leaf_count"):
                    expected = saved_linear_metadata.get(name, 0)
                    if (
                        isinstance(expected, bool)
                        or not isinstance(expected, int)
                        or int(expected) != 0
                    ):
                        _invalid_model(
                            f"linear leaf metadata {name} requires payload"
                        )
        linear_metadata = booster._linear_leaf_metadata()
        booster.auto_params_["linear_leaves"] = linear_metadata
        booster.auto_params_.setdefault("diagnostics", {})
        booster.auto_params_["diagnostics"]["linear_leaves"] = (
            linear_metadata
        )
        _restore_training_metadata(booster)

        wrapper_header = header.get("wrapper", {})
        for key in data.files:
            if key.startswith("wrapper__"):
                wrapper_arrays[key[len("wrapper__"):]] = data[key]

    if return_wrapper_payload:
        return booster, wrapper_header, wrapper_arrays
    return booster
