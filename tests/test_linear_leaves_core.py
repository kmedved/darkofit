import json

import numpy as np
import pytest

from darkofit import DarkoRegressor
from darkofit.binning import Binner
from darkofit.booster import DistributionalBoosting, MulticlassBoosting
from darkofit.flat_model import FlatLinearObliviousEnsemble
from darkofit.tree import (
    ObliviousTree,
    _solve_small,
    attach_oblivious_linear_leaves,
)


def test_binner_centers_cover_edges_interiors_and_missing_bin():
    np.testing.assert_array_equal(
        Binner._centers_for(np.array([])),
        np.array([0.0, np.nan]),
    )
    np.testing.assert_array_equal(
        Binner._centers_for(np.array([2.0])),
        np.array([2.0, 2.0, np.nan]),
    )
    np.testing.assert_array_equal(
        Binner._centers_for(np.array([1.0, 3.0, 7.0])),
        np.array([0.0, 2.0, 5.0, 9.0, np.nan]),
    )


def test_small_solver_matches_numpy_and_signals_singular_systems():
    rng = np.random.default_rng(9)
    for dimension in range(1, 9):
        for _ in range(20):
            design = rng.normal(size=(3 * dimension, dimension))
            matrix = design.T @ design + np.eye(dimension)
            rhs = rng.normal(size=dimension)
            expected = np.linalg.solve(matrix, rhs)
            actual = _solve_small(matrix.copy(), rhs.copy())
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    singular = _solve_small(np.zeros((3, 3)), np.ones(3))
    assert np.all(np.isnan(singular))


def _readable_coefficients(
    leaf,
    grad,
    hess,
    feature_values,
    fallback,
    intercept_lambda,
    linear_lambda,
    learning_rate,
):
    result = np.zeros((len(fallback), 2), dtype=np.float64)
    result[:, 0] = fallback
    for leaf_index in range(len(fallback)):
        rows = np.flatnonzero(leaf == leaf_index)
        if len(rows) < 4:
            continue
        design = np.column_stack([np.ones(len(rows)), feature_values[rows]])
        matrix = design.T @ (hess[rows, None] * design)
        matrix += np.diag([intercept_lambda, linear_lambda])
        matrix += 1e-9 * np.eye(2)
        rhs = -(design.T @ grad[rows])
        result[leaf_index] = learning_rate * np.linalg.solve(matrix, rhs)
    return result


def test_attach_linear_leaves_matches_readable_oracle_without_changing_tree():
    X_binned = np.array(
        [
            [0, 0],
            [1, 1],
            [2, 0],
            [3, 1],
            [0, 0],
            [1, 1],
            [2, 0],
            [3, 1],
        ],
        dtype=np.uint8,
    )
    tree = ObliviousTree(
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([-0.25, 0.75]),
        np.array([3.5]),
    )
    leaf = tree.apply(X_binned)
    grad = np.array([1.2, -0.7, 0.8, -1.1, 0.3, -0.4, 1.0, -0.6])
    hess = np.array([1.0, 1.2, 0.9, 1.1, 1.3, 0.8, 1.4, 1.0])
    bin_values = np.array(
        [
            [-1.5, -0.5, 0.5, 1.5],
            [-1.0, 1.0, np.nan, np.nan],
        ]
    )
    original = (
        tree.splits_feat.copy(),
        tree.splits_thr.copy(),
        tree.values.copy(),
        tree.gains.copy(),
    )

    attached = attach_oblivious_linear_leaves(
        tree,
        leaf,
        grad,
        hess,
        X_binned,
        bin_values,
        np.array([True, False]),
        intercept_lambda=3.0,
        linear_lambda=1.5,
        learning_rate=0.1,
    )

    assert attached is True
    expected = _readable_coefficients(
        leaf,
        grad,
        hess,
        bin_values[0, X_binned[:, 0]],
        original[2],
        3.0,
        1.5,
        0.1,
    )
    np.testing.assert_allclose(
        tree.linear_coefficients, expected, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(tree.linear_features, np.array([0]))
    for actual, before in zip(
        (tree.splits_feat, tree.splits_thr, tree.values, tree.gains), original
    ):
        np.testing.assert_array_equal(actual, before)

    expected_prediction = (
        expected[leaf, 0] + expected[leaf, 1] * bin_values[0, X_binned[:, 0]]
    )
    np.testing.assert_allclose(tree.predict(X_binned), expected_prediction)
    accumulated = np.full(len(X_binned), 2.0)
    tree.add_predict(X_binned, accumulated)
    np.testing.assert_allclose(accumulated, 2.0 + expected_prediction)


def test_small_linear_leaves_and_no_numeric_split_fall_back_exactly():
    X_binned = np.array([[0], [0], [0], [1], [1], [1]], dtype=np.uint8)
    fallback = np.array([0.125, -0.375])
    tree = ObliviousTree(
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
        fallback.copy(),
    )
    leaf = tree.apply(X_binned)
    attached = attach_oblivious_linear_leaves(
        tree,
        leaf,
        np.ones(6),
        np.ones(6),
        X_binned,
        np.array([[-1.0, 1.0]]),
        np.array([True]),
        intercept_lambda=3.0,
        linear_lambda=1.0,
        learning_rate=0.1,
    )
    assert attached is True
    np.testing.assert_array_equal(tree.linear_coefficients[:, 0], fallback)
    np.testing.assert_array_equal(tree.linear_coefficients[:, 1], np.zeros(2))
    np.testing.assert_array_equal(tree.predict(X_binned), fallback[leaf])

    constant = ObliviousTree(
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
        fallback.copy(),
    )
    assert (
        attach_oblivious_linear_leaves(
            constant,
            leaf,
            np.ones(6),
            np.ones(6),
            X_binned,
            np.array([[-1.0, 1.0]]),
            np.array([False]),
            intercept_lambda=3.0,
            linear_lambda=1.0,
            learning_rate=0.1,
        )
        is False
    )
    assert constant.linear_coefficients is None
    np.testing.assert_array_equal(constant.predict(X_binned), fallback[leaf])

    zero_weight = ObliviousTree(
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
        fallback.copy(),
    )
    zero_leaf = zero_weight.apply(X_binned)
    assert attach_oblivious_linear_leaves(
        zero_weight,
        zero_leaf,
        np.zeros(6),
        np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        X_binned,
        np.array([[-1.0, 1.0]]),
        np.array([True]),
        intercept_lambda=3.0,
        linear_lambda=1.0,
        learning_rate=0.1,
    ) is True
    np.testing.assert_array_equal(
        zero_weight.linear_coefficients[:, 0], fallback
    )
    np.testing.assert_array_equal(
        zero_weight.linear_coefficients[:, 1], np.zeros(2)
    )


def test_attach_linear_leaves_validates_penalties_and_numeric_mask():
    tree = ObliviousTree(np.array([0]), np.array([0]), np.array([0.0, 0.0]))
    X_binned = np.array([[0], [1]], dtype=np.uint8)
    leaf = tree.apply(X_binned)
    common = (
        tree,
        leaf,
        np.ones(2),
        np.ones(2),
        X_binned,
        np.array([[-1.0, 1.0]]),
        np.array([True]),
    )
    with pytest.raises(ValueError, match="intercept_lambda"):
        attach_oblivious_linear_leaves(
            *common,
            intercept_lambda=-1.0,
            linear_lambda=1.0,
            learning_rate=0.1,
        )
    with pytest.raises(ValueError, match="linear_lambda"):
        attach_oblivious_linear_leaves(
            *common,
            intercept_lambda=1.0,
            linear_lambda=np.nan,
            learning_rate=0.1,
        )
    with pytest.raises(ValueError, match="mask"):
        attach_oblivious_linear_leaves(
            tree,
            leaf,
            np.ones(2),
            np.ones(2),
            X_binned,
            np.array([[-1.0, 1.0]]),
            np.array([True, False]),
            intercept_lambda=1.0,
            linear_lambda=1.0,
            learning_rate=0.1,
        )


def _smooth_regression(n_samples=1200, seed=23):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.5, 2.5, size=(n_samples, 5))
    y = (
        np.sin(1.4 * X[:, 0])
        + 0.6 * X[:, 1] * X[:, 2]
        - 0.2 * X[:, 3] ** 2
        + rng.normal(0.0, 0.04, size=n_samples)
    )
    return X, y


def _linear_model_params(**overrides):
    params = {
        "iterations": 12,
        "learning_rate": 0.1,
        "depth": 4,
        "tree_mode": "catboost",
        "ordered_boosting": False,
        "thread_count": 1,
        "random_state": 4,
    }
    params.update(overrides)
    return params


def test_disabled_and_below_threshold_paths_are_prediction_exact():
    X, y = _smooth_regression(n_samples=320)
    implicit = DarkoRegressor(**_linear_model_params()).fit(X, y)
    disabled = DarkoRegressor(**_linear_model_params(linear_leaves=False)).fit(X, y)
    fallback = DarkoRegressor(**_linear_model_params(linear_leaves=True)).fit(X, y)

    expected = implicit.predict(X)
    np.testing.assert_array_equal(disabled.predict(X), expected)
    np.testing.assert_array_equal(fallback.predict(X), expected)
    assert fallback.model_.linear_leaves_active_ is False
    assert fallback.model_.linear_leaves_inactive_reason_ == "below_min_samples"
    assert fallback.model_.auto_params_["linear_leaves"] == {
        "requested": True,
        "active": False,
        "inactive_reason": "below_min_samples",
        "min_samples": 1000,
        "linear_lambda": 1.0,
        "numeric_feature_count": 0,
        "linear_tree_count": 0,
        "linear_leaf_count": 0,
    }


def test_all_categorical_request_records_exact_constant_fallback():
    rng = np.random.default_rng(12)
    X = np.empty((1100, 2), dtype=object)
    X[:, 0] = rng.choice(["guard", "wing", "center"], size=len(X))
    X[:, 1] = rng.choice(["home", "away"], size=len(X))
    y = (
        (X[:, 0] == "guard").astype(float)
        + 0.3 * (X[:, 1] == "home").astype(float)
        + rng.normal(0.0, 0.2, size=len(X))
    )
    base = DarkoRegressor(**_linear_model_params()).fit(X, y, cat_features=[0, 1])
    fallback = DarkoRegressor(**_linear_model_params(linear_leaves=True)).fit(
        X, y, cat_features=[0, 1]
    )

    np.testing.assert_array_equal(fallback.predict(X), base.predict(X))
    assert fallback.model_.linear_leaves_active_ is False
    assert fallback.model_.linear_leaves_inactive_reason_ == "no_numeric_features"


def test_active_linear_forest_has_exact_packed_and_staged_prediction():
    X, y = _smooth_regression()
    model = DarkoRegressor(**_linear_model_params(linear_leaves=True)).fit(X, y)
    booster = model.model_

    assert booster.linear_leaves_active_ is True
    assert booster.linear_tree_count_ > 0
    assert booster.linear_leaf_count_ > 0
    assert isinstance(booster._flat_ensemble(), FlatLinearObliviousEnsemble)
    X_binned = booster.prep_.transform(X[:127])
    tree_loop = np.full(len(X_binned), booster.init_, dtype=np.float64)
    for tree in booster.trees_:
        tree.add_predict(X_binned, tree_loop)
    np.testing.assert_array_equal(booster.predict_raw(X[:127]), tree_loop)
    np.testing.assert_array_equal(
        list(booster.staged_predict_raw(X[:127]))[-1],
        tree_loop,
    )
    assert np.all(np.isfinite(tree_loop))


def test_linear_forest_serialization_round_trip_is_exact(tmp_path):
    X, y = _smooth_regression()
    model = DarkoRegressor(**_linear_model_params(linear_leaves=True)).fit(X, y)
    expected = model.predict(X[:173])
    path = tmp_path / "linear-leaves.npz"
    model.save_model(path)

    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(str(archive["header"]))
        assert header["format_version"] == 4
        assert "trees__linear_coefficients_flat" in archive.files
        assert not any(archive[key].dtype.hasobject for key in archive.files)

    loaded = DarkoRegressor.load_model(path)
    np.testing.assert_array_equal(loaded.predict(X[:173]), expected)
    assert loaded.get_params()["linear_leaves"] is True
    assert loaded.model_.linear_leaves_active_ is True
    assert loaded.model_.linear_tree_count_ == model.model_.linear_tree_count_
    assert loaded.model_.linear_leaf_count_ == model.model_.linear_leaf_count_


def test_archive_version_tracks_new_params_without_mislabeling_old_format(
    tmp_path,
):
    X, y = _smooth_regression(n_samples=320)
    default = DarkoRegressor(
        **_linear_model_params(iterations=3, linear_leaves=False)
    ).fit(X, y)
    default_path = tmp_path / "default.npz"
    default.save_model(default_path)
    with np.load(default_path, allow_pickle=False) as archive:
        header = json.loads(str(archive["header"]))
    assert header["format_version"] == 2
    assert "linear_leaves" not in header["params"]
    assert "linear_lambda" not in header["params"]
    assert "linear_leaves" not in header["wrapper"]["params"]
    assert "linear_lambda" not in header["wrapper"]["params"]
    loaded_default = DarkoRegressor.load_model(default_path)
    assert loaded_default.linear_leaves is False
    np.testing.assert_array_equal(loaded_default.predict(X), default.predict(X))

    fallback = DarkoRegressor(
        **_linear_model_params(iterations=3, linear_leaves=True)
    ).fit(X, y)
    fallback_path = tmp_path / "fallback.npz"
    fallback.save_model(fallback_path)
    with np.load(fallback_path, allow_pickle=False) as archive:
        header = json.loads(str(archive["header"]))
        assert header["format_version"] == 4
        assert header["params"]["linear_leaves"] is True
        assert "trees__linear_counts" not in archive.files
    loaded = DarkoRegressor.load_model(fallback_path)
    np.testing.assert_array_equal(loaded.predict(X), fallback.predict(X))
    assert loaded.model_.linear_leaves_inactive_reason_ == "below_min_samples"

    custom_lambda = DarkoRegressor(
        **_linear_model_params(iterations=3, linear_lambda=2.0)
    ).fit(X, y)
    custom_path = tmp_path / "custom-lambda.npz"
    custom_lambda.save_model(custom_path)
    with np.load(custom_path, allow_pickle=False) as archive:
        header = json.loads(str(archive["header"]))
    assert header["format_version"] == 4
    assert header["params"]["linear_lambda"] == 2.0


def test_eligible_fit_with_no_retained_linear_tree_reports_inactive(tmp_path):
    X = np.zeros((1200, 2), dtype=np.float64)
    y = np.ones(1200, dtype=np.float64)
    model = DarkoRegressor(
        **_linear_model_params(iterations=3, linear_leaves=True)
    ).fit(X, y)
    metadata = model.model_.auto_params_["linear_leaves"]
    assert model.model_.trees_ == []
    assert model.model_.linear_leaves_active_ is False
    assert model.model_.linear_leaves_inactive_reason_ == (
        "no_retained_linear_trees"
    )
    assert metadata["active"] is False
    assert metadata["linear_tree_count"] == 0
    assert metadata["inactive_reason"] == "no_retained_linear_trees"

    path = tmp_path / "no-retained-linear-tree.npz"
    expected = model.predict(X[:50])
    model.save_model(path)
    loaded = DarkoRegressor.load_model(path)
    np.testing.assert_array_equal(loaded.predict(X[:50]), expected)
    assert loaded.model_.auto_params_["linear_leaves"] == metadata


def test_refit_cannot_activate_linear_lane_not_used_during_selection(
    tmp_path,
):
    X, y = _smooth_regression(n_samples=1050)
    model = DarkoRegressor(
        **_linear_model_params(
            iterations=5,
            linear_leaves=True,
            early_stopping=True,
            validation_fraction=0.1,
            refit=True,
            refit_strategy="exact",
        )
    ).fit(X, y)

    selection = model.selection_model_.auto_params_["linear_leaves"]
    assert selection["requested"] is True
    assert selection["active"] is False
    assert selection["inactive_reason"] == "below_min_samples"
    assert model.model_.linear_leaves is False
    assert model.model_.linear_leaves_active_ is False
    assert not any(
        tree.linear_coefficients is not None for tree in model.model_.trees_
    )
    assert model.model_.auto_params_["selection_linear_leaves"] == selection
    assert model.get_refit_params()["linear_leaves"] is False

    path = tmp_path / "selection-fallback-refit.npz"
    expected = model.predict(X[:50])
    model.save_model(path)
    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(str(archive["header"]))
    assert header["format_version"] == 4
    assert header["params"]["linear_leaves"] is False
    assert header["wrapper"]["params"]["linear_leaves"] is True
    loaded = DarkoRegressor.load_model(path)
    assert loaded.get_params()["linear_leaves"] is True
    assert loaded.model_.linear_leaves is False
    np.testing.assert_array_equal(loaded.predict(X[:50]), expected)


def test_all_missing_numeric_feature_round_trips_with_active_linear_forest(
    tmp_path,
):
    X, y = _smooth_regression()
    X[:, -1] = np.nan
    model = DarkoRegressor(
        **_linear_model_params(iterations=5, linear_leaves=True)
    ).fit(X, y)
    missing_bin = int(model.model_.prep_.n_bins_[-1]) - 1
    assert np.isnan(model.model_.linear_bin_values_[-1, missing_bin])

    path = tmp_path / "all-missing-feature.npz"
    model.save_model(path)
    loaded = DarkoRegressor.load_model(path)
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


def test_linear_feature_standardization_uses_positive_sample_weight():
    X, y = _smooth_regression()
    weights = np.linspace(0.0, 3.0, len(y))
    model = DarkoRegressor(
        **_linear_model_params(iterations=3, linear_leaves=True)
    ).fit(X, y, sample_weight=weights)
    booster = model.model_
    X_binned = booster.prep_.transform(X)
    for feature in range(X.shape[1]):
        centers = booster.prep_.binner_._centers_for(
            booster.prep_.binner_.borders_[feature]
        )
        sample_values = centers[X_binned[:, feature]]
        mask = np.isfinite(sample_values) & (weights > 0.0)
        mean = np.average(sample_values[mask], weights=weights[mask])
        variance = np.average((sample_values[mask] - mean) ** 2, weights=weights[mask])
        scale = np.sqrt(max(float(variance), 0.0)) or 1.0
        expected = (centers - mean) / scale
        np.testing.assert_allclose(
            booster.linear_bin_values_[feature, : len(centers)],
            expected,
            rtol=1e-13,
            atol=1e-13,
        )


def _read_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def test_linear_forest_serialization_rejects_malformed_payloads(tmp_path):
    X, y = _smooth_regression()
    model = DarkoRegressor(
        **_linear_model_params(iterations=4, linear_leaves=True)
    ).fit(X, y)
    source = tmp_path / "linear-source.npz"
    model.save_model(source)

    def missing_key(arrays):
        del arrays["trees__linear_counts"]

    def missing_entire_payload(arrays):
        for key in (
            "trees__linear_counts",
            "trees__linear_features_flat",
            "trees__linear_feature_offsets",
            "trees__linear_coefficients_flat",
            "trees__linear_coefficient_offsets",
            "linear__bin_values",
        ):
            del arrays[key]

    def downgraded_format(arrays):
        header = json.loads(str(arrays["header"]))
        header["format_version"] = 3
        arrays["header"] = np.array(json.dumps(header))

    def negative_count(arrays):
        arrays["trees__linear_counts"] = arrays["trees__linear_counts"].copy()
        arrays["trees__linear_counts"][0] = -1

    def bad_feature(arrays):
        arrays["trees__linear_features_flat"] = arrays[
            "trees__linear_features_flat"
        ].copy()
        arrays["trees__linear_features_flat"][0] = len(arrays["bin__n_bins"])

    def nonfinite_coefficient(arrays):
        arrays["trees__linear_coefficients_flat"] = arrays[
            "trees__linear_coefficients_flat"
        ].copy()
        arrays["trees__linear_coefficients_flat"][0] = np.nan

    def nonfinite_bin_value(arrays):
        arrays["linear__bin_values"] = arrays["linear__bin_values"].copy()
        arrays["linear__bin_values"][0, 0] = np.inf

    def bad_coefficient_offset(arrays):
        arrays["trees__linear_coefficient_offsets"] = arrays[
            "trees__linear_coefficient_offsets"
        ].copy()
        arrays["trees__linear_coefficient_offsets"][-1] -= 1

    def wrong_loss(arrays):
        header = json.loads(str(arrays["header"]))
        header["loss_name"] = "Logloss"
        header["loss_kwargs"] = {}
        arrays["header"] = np.array(json.dumps(header))

    def wrong_tree_mode(arrays):
        header = json.loads(str(arrays["header"]))
        header["params"]["tree_mode"] = "lightgbm"
        arrays["header"] = np.array(json.dumps(header))

    def ordered_boosting(arrays):
        header = json.loads(str(arrays["header"]))
        header["params"]["ordered_boosting"] = True
        arrays["header"] = np.array(json.dumps(header))

    def truthy_ordered_boosting(arrays):
        header = json.loads(str(arrays["header"]))
        header["params"]["ordered_boosting"] = "true"
        arrays["header"] = np.array(json.dumps(header))

    def inactive_linear_metadata(arrays):
        header = json.loads(str(arrays["header"]))
        header["auto_params"]["linear_leaves"]["active"] = False
        arrays["header"] = np.array(json.dumps(header))

    def wrong_linear_tree_count(arrays):
        header = json.loads(str(arrays["header"]))
        header["auto_params"]["linear_leaves"]["linear_tree_count"] += 1
        arrays["header"] = np.array(json.dumps(header))

    def wrong_linear_leaf_count(arrays):
        header = json.loads(str(arrays["header"]))
        header["auto_params"]["linear_leaves"]["linear_leaf_count"] += 1
        arrays["header"] = np.array(json.dumps(header))

    def active_empty_linear_payload(arrays):
        tree_count = len(arrays["trees__depths"])
        arrays["trees__linear_counts"] = np.zeros(tree_count, dtype=np.int64)
        arrays["trees__linear_features_flat"] = np.empty(0, dtype=np.int64)
        arrays["trees__linear_feature_offsets"] = np.zeros(
            tree_count + 1, dtype=np.int64
        )
        arrays["trees__linear_coefficients_flat"] = np.empty(
            0, dtype=np.float64
        )
        arrays["trees__linear_coefficient_offsets"] = np.zeros(
            tree_count + 1, dtype=np.int64
        )

    corruptions = {
        "missing-key": missing_key,
        "missing-entire-payload": missing_entire_payload,
        "downgraded-format": downgraded_format,
        "negative-count": negative_count,
        "bad-feature": bad_feature,
        "nonfinite-coefficient": nonfinite_coefficient,
        "nonfinite-bin-value": nonfinite_bin_value,
        "bad-coefficient-offset": bad_coefficient_offset,
        "wrong-loss": wrong_loss,
        "wrong-tree-mode": wrong_tree_mode,
        "ordered-boosting": ordered_boosting,
        "truthy-ordered-boosting": truthy_ordered_boosting,
        "inactive-linear-metadata": inactive_linear_metadata,
        "wrong-linear-tree-count": wrong_linear_tree_count,
        "wrong-linear-leaf-count": wrong_linear_leaf_count,
        "active-empty-linear-payload": active_empty_linear_payload,
    }
    for name, corrupt in corruptions.items():
        arrays = _read_archive(source)
        corrupt(arrays)
        path = tmp_path / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        with pytest.raises(ValueError, match="invalid DarkoFit model"):
            DarkoRegressor.load_model(path)


def test_linear_leaf_input_and_model_family_validation():
    X, y = _smooth_regression(n_samples=1000)
    with pytest.raises(TypeError, match="linear_leaves must be a bool"):
        DarkoRegressor(linear_leaves="yes").fit(X, y)
    with pytest.raises(ValueError, match="linear_lambda"):
        DarkoRegressor(tree_mode="catboost", linear_lambda=np.inf).fit(X, y)
    with pytest.raises(ValueError, match="loss='RMSE'"):
        DarkoRegressor(loss="MAE", tree_mode="catboost", linear_leaves=True).fit(X, y)
    with pytest.raises(ValueError, match="tree_mode='catboost'"):
        DarkoRegressor(tree_mode="auto", linear_leaves=True).fit(X, y)
    with pytest.raises(ValueError, match="ordered_boosting"):
        DarkoRegressor(
            tree_mode="catboost",
            ordered_boosting=True,
            linear_leaves=True,
        ).fit(X, y)
    with pytest.raises(ValueError, match="scalar RMSE regression"):
        MulticlassBoosting(linear_leaves=True).fit(X, (y > 0).astype(int))
    with pytest.raises(ValueError, match="scalar RMSE regression"):
        DistributionalBoosting(linear_leaves=True).fit(X, y)
