"""Regression tests for review blockers: corrupt-archive tree semantics,
class-minor buffer contracts, exact-solver boundaries, and prep round-trip."""

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from darkofit import DarkoClassifier, DarkoRegressor
from darkofit.booster import (
    _array_content_signature,
    _exact_mvs_probabilities,
    _exact_weighted_goss_probabilities,
)
from darkofit.serialization import FORMAT_VERSION
from darkofit.tree import build_leafwise_multiclass_tree


def _make_regression(n=400, f=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    y = X[:, 0] * 2.0 + np.sin(X[:, 1]) + rng.normal(scale=0.2, size=n)
    return X, y


def _mutated_archive(src, dst, **updates):
    with np.load(src, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    for key, mutate in updates.items():
        arrays[key] = mutate(np.array(arrays[key]))
    np.savez_compressed(dst, **arrays)
    return dst


def _fit_and_save(tmp_path, tree_mode, name):
    X, y = _make_regression()
    model = DarkoRegressor(
        iterations=12, tree_mode=tree_mode, learning_rate=0.3,
        random_state=0, early_stopping=False,
    )
    model.fit(X, y)
    path = str(tmp_path / name)
    model.save_model(path)
    return X, model, path


def _assert_load_rejected(path, match):
    with pytest.raises(ValueError, match=match):
        DarkoRegressor.load_model(path)


@pytest.mark.parametrize(
    ("location", "key"),
    [
        pytest.param("header", "init", id="missing-init"),
        pytest.param("header", "prep", id="missing-prep"),
        pytest.param("array", "importance", id="missing-importance"),
        pytest.param("array", "trees__depths", id="missing-tree-depths"),
    ],
)
def test_load_normalizes_missing_payload_fields_to_value_error(
    tmp_path, location, key
):
    _, _, source = _fit_and_save(
        tmp_path, "catboost", f"{location}-{key}-source.npz"
    )
    with np.load(source, allow_pickle=False) as archive:
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in archive.files
        }
    if location == "header":
        header = json.loads(str(arrays["header"]))
        header.pop(key)
        arrays["header"] = np.asarray(json.dumps(header))
    else:
        arrays.pop(key)
    corrupt = tmp_path / f"{location}-{key}-corrupt.npz"
    np.savez_compressed(corrupt, **arrays)

    _assert_load_rejected(corrupt, "invalid DarkoFit model")


def test_load_normalizes_corrupt_compressed_member_to_value_error(tmp_path):
    _, _, source = _fit_and_save(
        tmp_path, "catboost", "compressed-member-source.npz"
    )
    payload = bytearray(Path(source).read_bytes())
    with zipfile.ZipFile(source) as archive:
        info = archive.getinfo("trees__values_flat.npy")
    data_start = (
        info.header_offset
        + 30
        + len(info.filename.encode())
        + len(info.extra)
    )
    payload[data_start + max(1, info.compress_size // 2)] ^= 0xFF
    corrupt = tmp_path / "compressed-member-corrupt.npz"
    corrupt.write_bytes(payload)

    _assert_load_rejected(corrupt, "invalid DarkoFit model")


# ---------------------------------------------------------------------------
# Archive format versions must be exact integers, while genuine v1 stays valid.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "header",
    [
        pytest.param([], id="list"),
        pytest.param(None, id="null"),
        pytest.param(True, id="bool"),
        pytest.param(1, id="number"),
        pytest.param("header", id="string"),
    ],
)
def test_load_rejects_non_object_model_headers(tmp_path, header):
    _, _, path = _fit_and_save(tmp_path, "catboost", "header-source.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]).copy() for key in data.files}
    arrays["header"] = np.asarray(json.dumps(header))
    corrupt = tmp_path / f"header-{type(header).__name__}.npz"
    np.savez_compressed(corrupt, **arrays)

    _assert_load_rejected(corrupt, "header must be an object")


@pytest.mark.parametrize(
    ("present", "version"),
    [
        pytest.param(False, None, id="missing"),
        pytest.param(True, None, id="null"),
        pytest.param(True, True, id="bool"),
        pytest.param(True, 1.0, id="float"),
        pytest.param(True, 0, id="zero"),
        pytest.param(True, -1, id="negative"),
        pytest.param(True, FORMAT_VERSION + 1, id="too-new"),
    ],
)
def test_load_rejects_invalid_model_format_versions(
    tmp_path, present, version
):
    _, _, path = _fit_and_save(tmp_path, "catboost", "version-source.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]).copy() for key in data.files}
    header = json.loads(str(arrays["header"]))
    if present:
        header["format_version"] = version
    else:
        header.pop("format_version")
    arrays["header"] = np.asarray(json.dumps(header))
    corrupt = tmp_path / f"version-{present}-{version}.npz"
    np.savez_compressed(corrupt, **arrays)

    _assert_load_rejected(corrupt, "format version|model format")


def test_integer_v1_archive_survives_load_save_load(tmp_path):
    X, model, path = _fit_and_save(
        tmp_path, "catboost", "version-v1-source.npz"
    )
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]).copy() for key in data.files}
    header = json.loads(str(arrays["header"]))
    header["format_version"] = 1
    arrays["header"] = np.asarray(json.dumps(header))
    legacy = tmp_path / "version-v1-legacy.npz"
    second = tmp_path / "version-v1-second.npz"
    np.savez_compressed(legacy, **arrays)

    loaded = DarkoRegressor.load_model(legacy)
    loaded.save_model(second)
    reloaded = DarkoRegressor.load_model(second)

    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
    np.testing.assert_array_equal(model.predict(X), reloaded.predict(X))


@pytest.mark.parametrize(
    "diagnostics",
    [
        pytest.param(None, id="null"),
        pytest.param([], id="list"),
        pytest.param(True, id="bool"),
        pytest.param(1, id="number"),
        pytest.param("diagnostics", id="string"),
    ],
)
def test_load_rejects_non_object_diagnostics_metadata(
    tmp_path, diagnostics
):
    _, _, path = _fit_and_save(
        tmp_path, "catboost", "diagnostics-source.npz"
    )
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]).copy() for key in data.files}
    header = json.loads(str(arrays["header"]))
    header["auto_params"]["diagnostics"] = diagnostics
    arrays["header"] = np.asarray(json.dumps(header))
    corrupt = tmp_path / f"diagnostics-{type(diagnostics).__name__}.npz"
    np.savez_compressed(corrupt, **arrays)

    _assert_load_rejected(corrupt, "diagnostics metadata must be an object")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        pytest.param("lr", "0.5", "lr must be numeric", id="string-lr"),
        pytest.param("lr", np.inf, "lr must be finite", id="infinite-lr"),
        pytest.param(
            "best_score",
            "1.0",
            "best_score must be numeric",
            id="string-score",
        ),
        pytest.param(
            "best_iteration",
            12.0,
            "best_iteration must be an integer",
            id="float-best-iteration",
        ),
        pytest.param(
            "best_iteration",
            True,
            "best_iteration must be an integer",
            id="bool-best-iteration",
        ),
        pytest.param(
            "n_input_features",
            6.0,
            "n_input_features must be an integer",
            id="float-input-count",
        ),
    ],
)
def test_load_rejects_untyped_or_nonfinite_core_header_scalars(
    tmp_path, field, value, message
):
    _, _, source = _fit_and_save(
        tmp_path, "catboost", f"{field}-header-source.npz"
    )
    with np.load(source, allow_pickle=False) as archive:
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in archive.files
        }
    header = json.loads(str(arrays["header"]))
    header[field] = value
    arrays["header"] = np.asarray(json.dumps(header))
    corrupt = tmp_path / f"{field}-header-corrupt.npz"
    np.savez_compressed(corrupt, **arrays)

    _assert_load_rejected(corrupt, message)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        pytest.param(
            "n_outputs",
            2.0,
            "n_outputs must be an integer",
            id="float-output-count",
        ),
        pytest.param(
            "enabled",
            1,
            "enabled must be a boolean",
            id="integer-transform-enabled",
        ),
        pytest.param(
            "mean",
            "0.0",
            "transform mean must be numeric",
            id="string-transform-mean",
        ),
        pytest.param(
            "scale",
            "1.0",
            "transform scale must be numeric",
            id="string-transform-scale",
        ),
        pytest.param(
            "scale",
            np.inf,
            "transform scale must be finite",
            id="infinite-transform-scale",
        ),
    ],
)
def test_distributional_load_rejects_untyped_header_scalars(
    tmp_path, field, value, message
):
    X, y = _make_regression(n=160, f=4, seed=23)
    model = DarkoRegressor(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=3,
        min_child_samples=3,
        num_leaves=5,
        random_state=0,
        diagnostic_warnings="never",
    ).fit(X, y)
    source = tmp_path / f"distributional-{field}-source.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in archive.files
        }
    header = json.loads(str(arrays["header"]))
    if field == "n_outputs":
        header[field] = value
    else:
        header["target_transform"][field] = value
    arrays["header"] = np.asarray(json.dumps(header))
    corrupt = tmp_path / f"distributional-{field}-corrupt.npz"
    np.savez_compressed(corrupt, **arrays)

    _assert_load_rejected(corrupt, message)


# ---------------------------------------------------------------------------
# Blocker: nonoblivious tree payloads must be structurally valid.
# ---------------------------------------------------------------------------

def _first_internal_and_leaf(arrays):
    left = arrays["trees__left_child_flat"]
    internal = int(np.flatnonzero(left >= 0)[0])
    leaf = int(np.flatnonzero(left < 0)[0])
    return internal, leaf


def test_load_rejects_nonoblivious_child_self_loop(tmp_path):
    _, _, path = _fit_and_save(tmp_path, "lightgbm", "m.npz")

    def corrupt(arr):
        arr[0] = 0  # root's left child points back at the root
        return arr

    bad = _mutated_archive(
        path, str(tmp_path / "self_loop.npz"),
        trees__left_child_flat=corrupt,
    )
    _assert_load_rejected(bad, "invalid DarkoFit model")


def test_load_rejects_nonoblivious_internal_node_with_one_child(tmp_path):
    _, _, path = _fit_and_save(tmp_path, "lightgbm", "m.npz")
    with np.load(path, allow_pickle=False) as data:
        internal, _ = _first_internal_and_leaf(
            {k: data[k] for k in data.files}
        )

    def corrupt(arr):
        arr[internal] = -1  # internal node loses its right child
        return arr

    bad = _mutated_archive(
        path, str(tmp_path / "one_child.npz"),
        trees__right_child_flat=corrupt,
    )
    _assert_load_rejected(bad, "invalid DarkoFit model")


def test_load_rejects_nonoblivious_terminal_without_leaf_index(tmp_path):
    _, _, path = _fit_and_save(tmp_path, "lightgbm", "m.npz")
    with np.load(path, allow_pickle=False) as data:
        _, leaf = _first_internal_and_leaf({k: data[k] for k in data.files})

    def corrupt(arr):
        arr[leaf] = -1  # terminal node would predict values[-1]
        return arr

    bad = _mutated_archive(
        path, str(tmp_path / "leafless.npz"),
        trees__leaf_index_flat=corrupt,
    )
    _assert_load_rejected(bad, "invalid DarkoFit model")


def test_load_rejects_nonoblivious_out_of_range_feature(tmp_path):
    X, _, path = _fit_and_save(tmp_path, "lightgbm", "m.npz")
    with np.load(path, allow_pickle=False) as data:
        internal, _ = _first_internal_and_leaf(
            {k: data[k] for k in data.files}
        )

    def corrupt(arr):
        arr[internal] = X.shape[1] + 7  # beyond the binned feature space
        return arr

    bad = _mutated_archive(
        path, str(tmp_path / "oob_feature.npz"),
        trees__features_flat=corrupt,
    )
    _assert_load_rejected(bad, "out of range")


def test_load_rejects_nonoblivious_out_of_range_threshold(tmp_path):
    _, _, path = _fit_and_save(tmp_path, "lightgbm", "m.npz")
    with np.load(path, allow_pickle=False) as data:
        internal, _ = _first_internal_and_leaf(
            {k: data[k] for k in data.files}
        )

    def corrupt(arr):
        arr[internal] = 60000  # far beyond any per-feature bin count
        return arr

    bad = _mutated_archive(
        path, str(tmp_path / "oob_threshold.npz"),
        trees__thresholds_flat=corrupt,
    )
    _assert_load_rejected(bad, "out of range")


# ---------------------------------------------------------------------------
# Blocker: oblivious and levelwise split payloads must stay in bounds.
# ---------------------------------------------------------------------------

def test_load_rejects_oblivious_out_of_range_payloads(tmp_path):
    X, _, path = _fit_and_save(tmp_path, "catboost", "m.npz")

    def bad_feature(arr):
        arr[0] = X.shape[1] + 5
        return arr

    def bad_threshold(arr):
        arr[0] = 60000
        return arr

    _assert_load_rejected(
        _mutated_archive(
            path, str(tmp_path / "obl_feat.npz"),
            trees__feats_flat=bad_feature,
        ),
        "out of range",
    )
    _assert_load_rejected(
        _mutated_archive(
            path, str(tmp_path / "obl_thr.npz"),
            trees__thrs_flat=bad_threshold,
        ),
        "out of range",
    )


def test_load_rejects_levelwise_out_of_range_node_feature(tmp_path):
    X, _, path = _fit_and_save(tmp_path, "depthwise", "m.npz")

    def corrupt(arr):
        arr[:] = X.shape[1] + 3  # every slot, so active ones are included
        return arr

    bad = _mutated_archive(
        path, str(tmp_path / "lvl_feat.npz"),
        trees__node_features_flat=corrupt,
    )
    _assert_load_rejected(bad, "out of range")


# ---------------------------------------------------------------------------
# Tree gains and scalar/vector leaf values must stay numeric and finite.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tree_mode", ["catboost", "lightgbm", "depthwise"])
def test_load_rejects_malformed_scalar_tree_numeric_payloads(
    tmp_path, tree_mode
):
    _, _, source = _fit_and_save(
        tmp_path, tree_mode, f"{tree_mode}-numeric-source.npz"
    )
    with np.load(source, allow_pickle=False) as archive:
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in archive.files
        }

    for key in ("trees__gains_flat", "trees__values_flat"):
        assert arrays[key].size
        for suffix, mutate, match in (
            ("string", lambda value: value.astype(str), "must be numeric"),
            (
                "nan",
                lambda value: np.full_like(value, np.nan),
                "must be finite",
            ),
            (
                "inf",
                lambda value: np.full_like(value, np.inf),
                "must be finite",
            ),
        ):
            corrupt = dict(arrays)
            corrupt[key] = mutate(arrays[key])
            path = tmp_path / f"{tree_mode}-{key}-{suffix}.npz"
            np.savez_compressed(path, **corrupt)
            _assert_load_rejected(path, match)


def test_load_rejects_malformed_vector_tree_values(tmp_path):
    rng = np.random.default_rng(17)
    X = rng.normal(size=(180, 4))
    y = np.digitize(X[:, 0] - 0.4 * X[:, 1], [-0.5, 0.5])
    model = DarkoClassifier(
        iterations=4,
        tree_mode="lightgbm",
        multiclass_tree_strategy="shared_vector",
        min_child_samples=3,
        num_leaves=5,
        random_state=0,
        early_stopping=False,
    ).fit(X, y)
    source = tmp_path / "vector-numeric-source.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in archive.files
        }
    values = arrays["trees__values_flat"]
    assert values.ndim == 2
    for suffix, corrupt_values, match in (
        ("string", values.astype(str), "must be numeric"),
        ("nan", np.full_like(values, np.nan), "must be finite"),
        ("inf", np.full_like(values, np.inf), "must be finite"),
    ):
        corrupt = dict(arrays)
        corrupt["trees__values_flat"] = corrupt_values
        path = tmp_path / f"vector-values-{suffix}.npz"
        np.savez_compressed(path, **corrupt)
        with pytest.raises(ValueError, match=match):
            DarkoClassifier.load_model(path)


# ---------------------------------------------------------------------------
# Blocker: encoder statistics must match the categorical payload.
# ---------------------------------------------------------------------------

def test_load_rejects_truncated_encoder_statistics(tmp_path):
    rng = np.random.default_rng(3)
    n = 300
    X = np.empty((n, 2), dtype=object)
    X[:, 0] = rng.normal(size=n)
    X[:, 1] = np.array(["a", "b", "c", "d"], dtype=object)[
        rng.integers(0, 4, size=n)
    ]
    y = rng.normal(size=n)
    model = DarkoRegressor(
        iterations=8, random_state=0, early_stopping=False
    )
    model.fit(X, y, cat_features=[1])
    path = str(tmp_path / "cats.npz")
    model.save_model(path)

    # Drop the last category's statistics while keeping the payload
    # self-consistent (offsets still match the flat array lengths).
    bad = _mutated_archive(
        path, str(tmp_path / "cats_truncated.npz"),
        enc0__sums_flat=lambda arr: arr[:-1],
        enc0__counts_flat=lambda arr: arr[:-1],
        enc0__offsets=lambda arr: np.concatenate((arr[:-1], [arr[-1] - 1])),
    )
    _assert_load_rejected(bad, "do not match the categorical payload")


# ---------------------------------------------------------------------------
# Blocker: oversized class-minor histogram buffers must be rejected.
# ---------------------------------------------------------------------------

def test_multiclass_builder_rejects_oversized_class_dimension():
    rng = np.random.default_rng(5)
    n, f, K = 200, 4, 3
    X_binned = rng.integers(0, 8, size=(n, f)).astype(np.uint8)
    grad = rng.normal(size=(K, n))
    hess = np.abs(rng.normal(size=(K, n))) + 0.1
    n_bins = np.full(f, 8, dtype=np.int64)
    max_leaves = 7
    # Stale nonzero lanes in the extra class slot must never reach scoring.
    hg = np.full((f, max_leaves, 8, K + 1), 123.0)
    hh = np.full((f, max_leaves, 8, K + 1), 123.0)
    hc = np.zeros((f, max_leaves, 8))
    with pytest.raises(ValueError, match="class dimension"):
        build_leafwise_multiclass_tree(
            X_binned, grad, hess, n_bins, 4, 3.0, 0.1,
            hist_buffers=(hg, hh, hc), max_leaves=max_leaves,
        )


# ---------------------------------------------------------------------------
# Blocker: exact samplers must hit their targets at piecewise boundaries.
# ---------------------------------------------------------------------------

def test_exact_weighted_goss_hits_target_mass_at_boundaries():
    mass = np.array([3.0, 1.0, 1.0, 1.0, 0.5])
    breakpoints = np.cumsum(np.sort(mass)[::-1])
    targets = []
    for b in breakpoints[:-1]:
        for shift in (-3, -1, 0, 1, 3):
            t = b
            for _ in range(abs(shift)):
                t = np.nextafter(t, np.inf if shift > 0 else -np.inf)
            targets.append(float(t))
    targets += list(np.linspace(0.05, float(np.sum(mass)) - 1e-9, 23))
    for target in targets:
        probs = _exact_weighted_goss_probabilities(mass, target)
        achieved = float(np.sum(probs * mass))
        assert achieved == pytest.approx(target, rel=1e-9), (
            f"target_mass={target!r} achieved={achieved!r}"
        )


def test_exact_mvs_hits_target_count_at_boundaries():
    importance = np.array([4.0, 2.0, 2.0, 1.0, 0.25, 0.25])
    targets = list(np.linspace(0.5, importance.size - 1e-9, 29))
    # Exact saturation boundaries: theta == importance[k] transitions.
    sorted_desc = np.sort(importance)[::-1]
    prefix = np.concatenate(([0.0], np.cumsum(sorted_desc)))
    for k in range(1, importance.size):
        theta = sorted_desc[k - 1]
        boundary = k - 1 + (prefix[-1] - prefix[k - 1]) / theta
        if 0.0 < boundary < importance.size:
            for shift in (-2, 0, 2):
                t = boundary
                for _ in range(abs(shift)):
                    t = np.nextafter(t, np.inf if shift > 0 else -np.inf)
                targets.append(float(t))
    for target in targets:
        probs = _exact_mvs_probabilities(importance, target)
        assert float(np.sum(probs)) == pytest.approx(target, rel=1e-9), (
            f"target={target!r} sum={float(np.sum(probs))!r}"
        )
        assert np.all(probs <= 1.0 + 1e-15)


# ---------------------------------------------------------------------------
# Concern: bin_sample_count must survive save/load.
# ---------------------------------------------------------------------------


def test_object_cache_signature_separates_repr_hash_collisions():
    class ReprHashCollision:
        def __init__(self, group):
            self.group = group

        def __repr__(self):
            return "same"

        def __hash__(self):
            return 7

        def __eq__(self, other):
            return (
                isinstance(other, ReprHashCollision)
                and self.group == other.group
            )

    a = np.array([[ReprHashCollision("left")]], dtype=object)
    b = np.array([[ReprHashCollision("right")]], dtype=object)

    assert _array_content_signature(a) != _array_content_signature(b)
    assert _array_content_signature(a) == _array_content_signature(a)


@pytest.mark.parametrize("bin_sample_count", [1234, None])
def test_bin_sample_count_round_trips(tmp_path, bin_sample_count):
    X, y = _make_regression(n=250)
    model = DarkoRegressor(
        iterations=6, random_state=0, early_stopping=False,
        bin_sample_count=bin_sample_count,
    )
    model.fit(X, y)
    path = str(tmp_path / "bsc.npz")
    model.save_model(path)
    loaded = DarkoRegressor.load_model(path)
    assert loaded.model_.prep_.bin_sample_count == bin_sample_count
    assert np.array_equal(model.predict(X), loaded.predict(X))


# ---------------------------------------------------------------------------
# Sanity: valid models across all serializable modes still load and predict
# identically after the new validation (guards against over-rejection).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tree_mode", ["catboost", "lightgbm", "hybrid",
                                       "depthwise"])
def test_hardened_load_accepts_valid_models(tmp_path, tree_mode):
    X, _, path = _fit_and_save(tmp_path, tree_mode, f"{tree_mode}.npz")
    loaded = DarkoRegressor.load_model(path)
    reference = DarkoRegressor.load_model(path)
    assert np.array_equal(loaded.predict(X), reference.predict(X))


def test_hardened_load_accepts_valid_multiclass_models(tmp_path):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(300, 5))
    y = rng.integers(0, 3, size=300)
    for strategy in ("shared_vector", "per_class"):
        model = DarkoClassifier(
            iterations=8, tree_mode="lightgbm", random_state=0,
            early_stopping=False, multiclass_tree_strategy=strategy,
        )
        model.fit(X, y)
        path = str(tmp_path / f"mc_{strategy}.npz")
        model.save_model(path)
        loaded = DarkoClassifier.load_model(path)
        assert np.array_equal(model.predict_proba(X), loaded.predict_proba(X))


# ---------------------------------------------------------------------------
# Corrupt headers must surface as "invalid DarkoFit model" ValueErrors, not
# raw KeyError/TypeError from model reconstruction.
# ---------------------------------------------------------------------------

def _mutated_header(src, dst, mutate):
    import json

    with np.load(src, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    header = json.loads(str(arrays["header"]))
    mutate(header)
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(dst, **arrays)
    return dst


def test_load_rejects_unknown_loss_name(tmp_path):
    _, _, path = _fit_and_save(tmp_path, "catboost", "loss_src.npz")
    bad = _mutated_header(
        path, str(tmp_path / "bad_loss.npz"),
        lambda header: header.update(loss_name="NotALoss"),
    )
    _assert_load_rejected(bad, "unknown loss")


def test_load_rejects_unhashable_loss_names(tmp_path):
    X, y = _make_regression(n=120)

    scalar = DarkoRegressor(iterations=2, random_state=0).fit(X, y)
    scalar_path = str(tmp_path / "scalar_loss_src.npz")
    scalar.save_model(scalar_path)
    bad_scalar = _mutated_header(
        scalar_path, str(tmp_path / "bad_scalar_loss.npz"),
        lambda header: header.update(loss_name=["RMSE"]),
    )
    _assert_load_rejected(bad_scalar, "unknown loss")

    distributional = DarkoRegressor(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=2,
        min_child_samples=3,
        num_leaves=3,
        random_state=0,
        diagnostic_warnings="never",
    ).fit(X, y)
    dist_path = str(tmp_path / "dist_loss_src.npz")
    distributional.save_model(dist_path)
    bad_dist = _mutated_header(
        dist_path, str(tmp_path / "bad_dist_loss.npz"),
        lambda header: header.update(loss_name={"Gaussian": True}),
    )
    _assert_load_rejected(bad_dist, "unknown distributional loss")


def test_load_rejects_non_object_params_header(tmp_path):
    _, _, path = _fit_and_save(tmp_path, "catboost", "params_src.npz")
    bad = _mutated_header(
        path, str(tmp_path / "bad_params.npz"),
        lambda header: header.update(params=["not", "a", "dict"]),
    )
    _assert_load_rejected(bad, "params header must be an object")


def test_load_rejects_unexpected_constructor_param(tmp_path):
    _, _, path = _fit_and_save(tmp_path, "catboost", "ctor_src.npz")
    bad = _mutated_header(
        path, str(tmp_path / "bad_ctor.npz"),
        lambda header: header["params"].update(definitely_not_a_param=1),
    )
    _assert_load_rejected(bad, "invalid booster params")

    overflow = _mutated_header(
        path, str(tmp_path / "overflow_ctor.npz"),
        lambda header: header["params"].update(iterations=float("inf")),
    )
    _assert_load_rejected(overflow, "invalid booster params")
