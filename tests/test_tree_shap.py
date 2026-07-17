import math

import numba
import numpy as np
import pytest

from darkofit import DarkoClassifier, DarkoRegressor
from darkofit.shap import (
    factorials,
    max_original_players,
    pack_oblivious_shap_forest,
    shap_forest_linear,
)
from darkofit.tree import ObliviousTree


def _base_params(**overrides):
    params = {
        "iterations": 16,
        "learning_rate": 0.1,
        "depth": 3,
        "l2_leaf_reg": 1.0,
        "max_bins": 32,
        "min_child_samples": 1,
        "ordered_boosting": False,
        "random_state": 0,
        "thread_count": 2,
        "diagnostic_warnings": "never",
    }
    params.update(overrides)
    return params


def _efficiency_error(prediction, contributions, expected_value):
    return float(
        np.max(
            np.abs(
                np.asarray(contributions).sum(axis=1)
                + expected_value
                - np.asarray(prediction)
            )
        )
    )


def _brute_force_tree_shap(tree, X_binned, background_binned, feature_map):
    """Independent pure-Python coalition oracle for one oblivious tree."""
    internal_features = list(tree.splits_feat)
    if tree.linear_coefficients is not None:
        internal_features.extend(tree.linear_features)
    players = sorted({int(feature_map[f]) for f in internal_features})
    player_slot = {feature: index for index, feature in enumerate(players)}
    count = len(players)
    output = np.zeros((len(X_binned), int(feature_map.max()) + 1))

    def coalition_value(explained, background, mask):
        leaf = 0
        for internal, threshold in zip(tree.splits_feat, tree.splits_thr):
            original = int(feature_map[internal])
            source = (
                explained
                if (mask >> player_slot[original]) & 1
                else background
            )
            leaf = leaf * 2 + int(source[internal] > threshold)
        if tree.linear_coefficients is None:
            return float(tree.values[leaf])
        value = float(tree.linear_coefficients[leaf, 0])
        for index, internal in enumerate(tree.linear_features):
            original = int(feature_map[internal])
            source = (
                explained
                if (mask >> player_slot[original]) & 1
                else background
            )
            linear_value = tree.linear_bin_values[internal, source[internal]]
            if np.isfinite(linear_value):
                value += (
                    float(tree.linear_coefficients[leaf, index + 1])
                    * linear_value
                )
        return value

    for row, explained in enumerate(X_binned):
        for background in background_binned:
            values = [
                coalition_value(explained, background, mask)
                for mask in range(1 << count)
            ]
            for player, original in enumerate(players):
                bit = 1 << player
                marginal = 0.0
                for mask in range(1 << count):
                    if mask & bit:
                        continue
                    size = mask.bit_count()
                    weight = (
                        math.factorial(size)
                        * math.factorial(count - size - 1)
                        / math.factorial(count)
                    )
                    marginal += weight * (values[mask | bit] - values[mask])
                output[row, original] += marginal / len(background_binned)
    return output


def test_tree_shap_matches_independent_coalition_oracle_and_threads():
    tree = ObliviousTree(
        np.array([0, 1], dtype=np.int64),
        np.array([0, 0], dtype=np.int64),
        np.array([1.0, 2.0, 4.0, 8.0]),
    )
    X_binned = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.uint8)
    background = np.array([[0, 0], [1, 0]], dtype=np.uint8)
    feature_map = np.array([0, 1], dtype=np.int64)
    packed = pack_oblivious_shap_forest([tree])
    expected = _brute_force_tree_shap(
        tree, X_binned, background, feature_map
    )
    previous_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        serial = shap_forest_linear(
            X_binned,
            background,
            *packed,
            feature_map,
            2,
            factorials(max_original_players([tree], feature_map)),
        )
        numba.set_num_threads(2)
        parallel = shap_forest_linear(
            X_binned,
            background,
            *packed,
            feature_map,
            2,
            factorials(max_original_players([tree], feature_map)),
        )
    finally:
        numba.set_num_threads(previous_threads)
    assert np.array_equal(serial, parallel)
    assert np.allclose(serial, expected, rtol=0.0, atol=1e-12)

    # Three split levels collapse to one original-feature player: internal
    # feature 0 repeats and feature 1 aliases it. The linear slope is a second
    # player not present in the split set.
    linear_coefficients = np.column_stack(
        [np.arange(1.0, 9.0), np.linspace(-0.8, 0.6, 8)]
    )
    linear_bin_values = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.5, 2.0],
        ]
    )
    grouped_linear_tree = ObliviousTree(
        np.array([0, 0, 1], dtype=np.int64),
        np.array([0, 1, 0], dtype=np.int64),
        linear_coefficients[:, 0].copy(),
        linear_features=np.array([2], dtype=np.int64),
        linear_coefficients=linear_coefficients,
        linear_bin_values=linear_bin_values,
    )
    grouped_X = np.array(
        [[2, 1, 2], [1, 2, 0], [0, 1, 1]], dtype=np.uint8
    )
    grouped_background = np.array(
        [[0, 0, 0], [2, 2, 1]], dtype=np.uint8
    )
    grouped_map = np.array([0, 0, 1], dtype=np.int64)
    grouped_packed = pack_oblivious_shap_forest([grouped_linear_tree])
    grouped_actual = shap_forest_linear(
        grouped_X,
        grouped_background,
        *grouped_packed,
        grouped_map,
        2,
        factorials(
            max_original_players([grouped_linear_tree], grouped_map)
        ),
    )
    grouped_expected = _brute_force_tree_shap(
        grouped_linear_tree,
        grouped_X,
        grouped_background,
        grouped_map,
    )
    assert np.allclose(
        grouped_actual, grouped_expected, rtol=0.0, atol=1e-12
    )


def test_regression_tree_shap_efficiency_background_and_determinism():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 6))
    y = 2.0 * X[:, 0] - X[:, 1] + X[:, 2] * X[:, 3] + rng.normal(
        scale=0.1, size=len(X)
    )
    model = DarkoRegressor(**_base_params()).fit(X, y)
    repeated = DarkoRegressor(**_base_params()).fit(X, y)
    background = X[100:132]

    first = model.shap_values(X[:12], X_background=background)
    first_base = model.expected_value_
    second = model.shap_values(X[:12], X_background=background)

    assert np.array_equal(first, second)
    assert np.array_equal(
        model.model_._shap_background_, repeated.model_._shap_background_
    )
    assert first.shape == (12, X.shape[1])
    assert abs(first_base - model.predict(background).mean()) < 1e-12
    assert _efficiency_error(model.predict(X[:12]), first, first_base) < 1e-9

    default_first = model.model_.shap_values(
        X[:5], max_background=16, random_state=7
    )
    default_second = model.model_.shap_values(
        X[:5], max_background=16, random_state=7
    )
    assert np.array_equal(default_first[0], default_second[0])
    assert default_first[1] == default_second[1]


def test_linear_leaf_tree_shap_efficiency():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(1200, 6))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + X[:, 2] * X[:, 3]
    y += rng.normal(scale=0.1, size=len(X))
    model = DarkoRegressor(
        **_base_params(iterations=12, linear_leaves=True)
    ).fit(X, y)
    assert model.model_.linear_leaves_active_

    background = X[100:124]
    contributions = model.shap_values(X[:8], X_background=background)
    assert _efficiency_error(
        model.predict(X[:8]), contributions, model.expected_value_
    ) < 1e-9


def test_categorical_tree_shap_maps_to_original_features():
    rng = np.random.default_rng(3)
    numeric = rng.normal(size=500)
    category = rng.choice(["a", "b", "c"], size=500).astype(object)
    X = np.column_stack([numeric, category])
    y = numeric + 1.5 * (category == "b") + rng.normal(scale=0.1, size=500)
    model = DarkoRegressor(**_base_params()).fit(X, y, cat_features=[1])

    contributions = model.shap_values(
        X[:10], X_background=X[100:132]
    )
    assert contributions.shape == (10, 2)
    assert _efficiency_error(
        model.predict(X[:10]), contributions, model.expected_value_
    ) < 1e-9


@pytest.mark.parametrize(
    ("loss", "alpha"), (("MAE", 0.5), ("Quantile", 0.8))
)
def test_scalar_alternate_loss_tree_shap_efficiency(loss, alpha):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(350, 5))
    y = X[:, 0] - X[:, 1] + rng.normal(scale=0.2, size=len(X))
    model = DarkoRegressor(
        **_base_params(iterations=10, loss=loss, alpha=alpha)
    ).fit(X, y)
    contributions = model.shap_values(
        X[:6], X_background=X[100:120]
    )
    assert _efficiency_error(
        model.predict(X[:6]), contributions, model.expected_value_
    ) < 1e-9


def test_binary_tree_shap_explains_raw_log_odds():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(500, 6))
    score = 2 * X[:, 0] - X[:, 1] + X[:, 2] * X[:, 3]
    y = (score + rng.normal(scale=0.2, size=len(X)) > 0).astype(int)
    model = DarkoClassifier(**_base_params()).fit(X, y)
    contributions = model.shap_values(
        X[:8], X_background=X[100:132]
    )
    raw = model.model_.predict_raw(X[:8])
    assert _efficiency_error(raw, contributions, model.expected_value_) < 1e-9


def test_tree_shap_rejects_unsupported_models():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(300, 5))
    y = X[:, 0] + rng.normal(size=len(X))

    leafwise = DarkoRegressor(
        **_base_params(iterations=5, tree_mode="lightgbm")
    ).fit(X, y)
    with pytest.raises(NotImplementedError, match="only oblivious"):
        leafwise.shap_values(X[:2])

    distributional = DarkoRegressor(
        **_base_params(
            iterations=3, loss="Gaussian", tree_mode="lightgbm"
        )
    ).fit(X, y)
    with pytest.raises(NotImplementedError, match="distributional"):
        distributional.shap_values(X[:2], X_background=X[:8])

    residual = DarkoRegressor(
        **_base_params(iterations=5, linear_residual=True)
    ).fit(X, y)
    assert residual.linear_residual_active_
    with pytest.raises(NotImplementedError, match="linear_residual"):
        residual.shap_values(X[:2], X_background=X[:8])

    multiclass_y = rng.integers(0, 3, size=len(X))
    multiclass = DarkoClassifier(
        **_base_params(iterations=5)
    ).fit(X, multiclass_y)
    with pytest.raises(NotImplementedError, match="binary"):
        multiclass.shap_values(X[:2], X_background=X[:8])


def test_tree_shap_validates_background_controls():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(250, 4))
    y = X[:, 0] - X[:, 1]
    model = DarkoRegressor(**_base_params(iterations=5)).fit(X, y)

    for invalid in (0, -1):
        with pytest.raises(ValueError, match="at least 1"):
            model.model_.shap_values(X[:2], max_background=invalid)
    for invalid in (True, 1.5, "2"):
        with pytest.raises(TypeError, match="positive integer"):
            model.model_.shap_values(X[:2], max_background=invalid)
    with pytest.raises(ValueError, match="at least one row"):
        model.model_.shap_values(X[:2], background=X[:0])
    model.model_._shap_background_ = None
    with pytest.raises(ValueError, match="no stored SHAP background"):
        model.shap_values(X[:2])
    custom = model.shap_values(X[:2], X_background=X[:8])
    assert custom.shape == (2, X.shape[1])


def test_tree_shap_rejects_more_than_sixteen_players_before_kernel():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(80, 17))
    y = X[:, 0] + rng.normal(scale=0.1, size=len(X))
    model = DarkoRegressor(
        **_base_params(iterations=1, depth=1)
    ).fit(X, y)
    model.model_.trees_ = [
        ObliviousTree(
            np.arange(17, dtype=np.int64),
            np.zeros(17, dtype=np.int64),
            np.zeros(1 << 17, dtype=np.float64),
        )
    ]
    model.model_._flat_cache_ = None

    with pytest.raises(NotImplementedError, match="at most 16"):
        model.shap_values(X[:1], X_background=X[1:2])


def test_tree_shap_background_serialization_and_corruption(tmp_path):
    rng = np.random.default_rng(8)
    X = rng.normal(size=(300, 5))
    y = X[:, 0] - X[:, 1] + rng.normal(scale=0.1, size=len(X))
    model = DarkoRegressor(**_base_params(iterations=8)).fit(X, y)
    before = model.shap_values(X[:5])
    before_base = model.expected_value_
    path = tmp_path / "shap.npz"
    model.save_model(path)

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    assert arrays["shap__background"].shape == (200, X.shape[1])

    loaded = DarkoRegressor.load_model(path)
    after = loaded.shap_values(X[:5])
    assert np.array_equal(before, after)
    assert before_base == loaded.expected_value_
    assert np.array_equal(
        model.model_._shap_background_, loaded.model_._shap_background_
    )

    bad_path = tmp_path / "bad-shap.npz"
    bad = arrays["shap__background"].copy()
    bad[0, 0] = np.iinfo(bad.dtype).max
    np.savez_compressed(bad_path, **{**arrays, "shap__background": bad})
    with pytest.raises(ValueError, match="out-of-range bins"):
        DarkoRegressor.load_model(bad_path)

    legacy_path = tmp_path / "legacy-without-shap.npz"
    arrays.pop("shap__background")
    np.savez_compressed(legacy_path, **arrays)
    legacy = DarkoRegressor.load_model(legacy_path)
    with pytest.raises(ValueError, match="no stored SHAP background"):
        legacy.shap_values(X[:2])
    contributions = legacy.shap_values(X[:2], X_background=X[:8])
    assert contributions.shape == (2, X.shape[1])
