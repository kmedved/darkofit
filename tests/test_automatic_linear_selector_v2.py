import json

import numba
import numpy as np
import pytest
from sklearn.base import clone

from darkofit import DarkoClassifier, DarkoRegressor
from darkofit.sklearn_api import _paired_mse_gain_statistics


def _smooth_data(n_samples=1400, seed=17):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, 4))
    y = (
        2.4 * X[:, 0]
        - 1.7 * X[:, 1]
        + 0.35 * X[:, 2] ** 2
        + rng.normal(0.0, 0.15, size=n_samples)
    )
    return X, y


def _params(**overrides):
    params = {
        "iterations": 6,
        "learning_rate": 0.1,
        "depth": 3,
        "tree_mode": "catboost",
        "ordered_boosting": False,
        "thread_count": 1,
        "random_state": 19,
        "diagnostic_warnings": "never",
    }
    params.update(overrides)
    return params


def _rewrite_header(source, destination, mutate):
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    mutate(header)
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(destination, **arrays)


def test_paired_gain_statistics_match_unweighted_and_weighted_definitions():
    y = np.array([0.0, 1.0, 2.0, 4.0])
    constant = np.array([0.5, 0.0, 3.0, 2.0])
    linear = np.array([0.1, 0.9, 2.2, 3.5])
    differences = (y - constant) ** 2 - (y - linear) ** 2

    gain, standard_error, z_score = _paired_mse_gain_statistics(
        y, constant, linear
    )
    assert gain == pytest.approx(np.mean(differences))
    assert standard_error == pytest.approx(
        np.std(differences, ddof=1) / np.sqrt(len(y))
    )
    assert z_score == pytest.approx(gain / standard_error)

    weights = np.array([0.5, 1.0, 1.5, 3.0])
    gain, standard_error, z_score = _paired_mse_gain_statistics(
        y, constant, linear, weights
    )
    weight_sum = np.sum(weights)
    effective_n = weight_sum**2 / np.dot(weights, weights)
    expected_gain = np.dot(weights, differences) / weight_sum
    expected_variance = (
        np.dot(weights, (differences - expected_gain) ** 2)
        / weight_sum
        * effective_n
        / (effective_n - 1.0)
    )
    assert gain == pytest.approx(expected_gain)
    assert standard_error == pytest.approx(
        np.sqrt(expected_variance / effective_n)
    )
    assert z_score == pytest.approx(gain / standard_error)


def test_default_auto_selects_linear_and_final_fit_matches_explicit_boolean():
    X, y = _smooth_data()
    previous_threads = numba.get_num_threads()
    automatic = DarkoRegressor(**_params()).fit(X, y)
    assert numba.get_num_threads() == previous_threads
    explicit = DarkoRegressor(
        **_params(linear_leaves=True)
    ).fit(X, y)

    assert automatic.get_params(deep=False)["linear_leaves"] == "auto"
    assert automatic.linear_leaves_selected_ is True
    metadata = automatic.automatic_linear_selector_
    assert metadata["reason"] == "selected_linear"
    assert metadata["fit_random_state_seed"] == 19
    assert metadata["split"]["source"] == "automatic_holdout"
    assert metadata["split"]["policy"] == "weighted_target_stratified"
    assert metadata["split"]["rows_disjoint"] is True
    assert metadata["minimum_relative_improvement"] == 0.0
    assert metadata["minimum_gain_z"] == 2.0
    assert metadata["paired_mse_gain"] > 0.0
    assert metadata["paired_mse_gain_standard_error"] >= 0.0
    assert metadata["paired_mse_gain_z"] >= metadata["minimum_gain_z"]
    np.testing.assert_array_equal(automatic.predict(X), explicit.predict(X))
    assert numba.get_num_threads() == previous_threads
    assert automatic.best_n_estimators_ == explicit.best_n_estimators_
    assert automatic.learning_rate_ == explicit.learning_rate_
    assert automatic.get_refit_params()["linear_leaves"] is True


def test_auto_decline_and_small_fallback_are_exact_constant_models():
    X, _ = _smooth_data()
    y = np.ones(len(X), dtype=np.float64)
    automatic = DarkoRegressor(**_params()).fit(X, y)
    explicit = DarkoRegressor(
        **_params(linear_leaves=False)
    ).fit(X, y)

    assert automatic.linear_leaves_selected_ is False
    assert automatic.automatic_linear_selector_["reason"] == (
        "linear_candidate_inactive"
    )
    np.testing.assert_array_equal(automatic.predict(X), explicit.predict(X))

    small_X, small_y = _smooth_data(n_samples=300)
    small_auto = DarkoRegressor(**_params()).fit(small_X, small_y)
    small_explicit = DarkoRegressor(
        **_params(linear_leaves=False)
    ).fit(small_X, small_y)
    assert small_auto.automatic_linear_selector_["eligible"] is False
    assert small_auto.automatic_linear_selector_["reason"] == "below_min_samples"
    np.testing.assert_array_equal(
        small_auto.predict(small_X), small_explicit.predict(small_X)
    )


def test_auto_split_is_deterministic_group_disjoint_and_weight_aware():
    X, y = _smooth_data(n_samples=1500)
    groups = np.repeat(np.arange(150), 10)
    weights = np.linspace(0.2, 2.0, len(y))
    first = DarkoRegressor(**_params(iterations=3)).fit(
        X, y, groups=groups, sample_weight=weights
    )
    second = DarkoRegressor(**_params(iterations=3)).fit(
        X, y, groups=groups, sample_weight=weights
    )

    first_split = first.automatic_linear_selector_["split"]
    second_split = second.automatic_linear_selector_["split"]
    assert first_split["policy"] == "group_shuffle"
    assert first_split["group_disjoint"] is True
    assert first_split["sample_weight_provided"] is True
    assert first_split["train_positions_sha256"] == second_split[
        "train_positions_sha256"
    ]
    assert first_split["validation_positions_sha256"] == second_split[
        "validation_positions_sha256"
    ]
    assert first_split["train_weight_sum"] > 0.0
    assert first_split["validation_weight_sum"] > 0.0


def test_explicit_eval_set_and_weights_are_used_for_both_auditions():
    X, y = _smooth_data(n_samples=1500)
    X_train, X_eval = X[:1300], X[1300:]
    y_train, y_eval = y[:1300], y[1300:]
    train_weight = np.linspace(0.5, 1.5, len(y_train))
    eval_weight = np.linspace(1.0, 2.0, len(y_eval))
    model = DarkoRegressor(**_params(iterations=3)).fit(
        X_train,
        y_train,
        eval_set=(X_eval, y_eval),
        sample_weight=train_weight,
        eval_sample_weight=eval_weight,
    )

    split = model.automatic_linear_selector_["split"]
    assert split["source"] == "explicit_eval_set"
    assert split["train_rows"] == 1300
    assert split["validation_rows"] == 200
    assert split["train_weight_sum"] == pytest.approx(train_weight.sum())
    assert split["validation_weight_sum"] == pytest.approx(eval_weight.sum())
    assert all(
        record["best_n_estimators"] >= 0
        for record in model.automatic_linear_selector_["selection_fits"]
    )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"loss": "MAE"}, "non_rmse_loss"),
        ({"tree_mode": "lightgbm"}, "non_catboost_tree_mode"),
        ({"linear_residual": True}, "linear_residual"),
    ],
)
def test_unsupported_auto_modes_fall_back_without_changing_behavior(
    overrides, reason
):
    X, y = _smooth_data(n_samples=260)
    automatic = DarkoRegressor(**_params(**overrides)).fit(X, y)
    explicit = DarkoRegressor(
        **_params(**overrides, linear_leaves=False)
    ).fit(X, y)

    assert automatic.automatic_linear_selector_["eligible"] is False
    assert automatic.automatic_linear_selector_["reason"] == reason
    np.testing.assert_array_equal(automatic.predict(X), explicit.predict(X))


def test_failed_selector_refit_is_transactional_and_restores_thread_mask():
    X_small, y_small = _smooth_data(n_samples=300)
    model = DarkoRegressor(
        **_params(linear_leaves=False)
    ).fit(X_small, y_small)
    expected_model = model.model_
    expected_prediction = model.predict(X_small)

    X, y = _smooth_data()
    model.set_params(linear_leaves="auto", linear_lambda=np.inf)
    previous_threads = numba.get_num_threads()
    with pytest.raises(ValueError, match="linear_lambda"):
        model.fit(X, y)
    assert numba.get_num_threads() == previous_threads
    assert model.model_ is expected_model
    np.testing.assert_array_equal(model.predict(X_small), expected_prediction)
    assert numba.get_num_threads() == previous_threads


def test_random_state_object_is_resolved_once_for_exact_final_fit():
    X, y = _smooth_data()
    automatic_rng = np.random.RandomState(91)
    explicit_rng = np.random.RandomState(91)
    automatic = DarkoRegressor(
        **_params(random_state=automatic_rng)
    ).fit(X, y)
    selected = automatic.linear_leaves_selected_
    explicit = DarkoRegressor(
        **_params(random_state=explicit_rng, linear_leaves=selected)
    ).fit(X, y)

    np.testing.assert_array_equal(automatic.predict(X), explicit.predict(X))
    assert automatic.automatic_linear_selector_["fit_random_state_seed"] == (
        explicit.model_._fit_random_state_seed_
    )


def test_auto_final_state_matches_explicit_refit_fallback_semantics(tmp_path):
    X, y = _smooth_data()
    params = _params(
        iterations=4,
        early_stopping=True,
        validation_fraction=0.4,
        refit=True,
    )
    automatic = DarkoRegressor(**params).fit(X, y)
    explicit = DarkoRegressor(
        **{**params, "linear_leaves": True}
    ).fit(X, y)

    assert automatic.linear_leaves_selected_ is True
    assert automatic.automatic_linear_selector_[
        "final_booster_linear_leaves"
    ] is False
    assert automatic.automatic_linear_selector_[
        "final_linear_leaves_active"
    ] is False
    np.testing.assert_array_equal(automatic.predict(X), explicit.predict(X))
    assert automatic.get_refit_params()["linear_leaves"] is False
    path = tmp_path / "automatic-refit-fallback.npz"
    automatic.save_model(path)
    loaded = DarkoRegressor.load_model(path)
    assert loaded.linear_leaves_selected_ is True
    np.testing.assert_array_equal(loaded.predict(X), automatic.predict(X))


def test_auto_safe_npz_round_trips_and_rejects_selector_corruption(tmp_path):
    X, y = _smooth_data()
    model = DarkoRegressor(**_params()).fit(X, y)
    expected = model.predict(X[:80])
    first_path = tmp_path / "automatic-selector.npz"
    second_path = tmp_path / "automatic-selector-second.npz"
    corrupt_path = tmp_path / "automatic-selector-corrupt.npz"
    corrupt_gain_path = tmp_path / "automatic-selector-corrupt-gain.npz"
    model.save_model(first_path)

    loaded = DarkoRegressor.load_model(first_path)
    loaded.save_model(second_path)
    reloaded = DarkoRegressor.load_model(second_path)
    assert loaded.linear_leaves == "auto"
    assert reloaded.automatic_linear_selector_ == (
        model.automatic_linear_selector_
    )
    np.testing.assert_array_equal(loaded.predict(X[:80]), expected)
    np.testing.assert_array_equal(reloaded.predict(X[:80]), expected)

    def corrupt(header):
        header["wrapper"]["state"]["automatic_linear_selector"][
            "reason"
        ] = "forged"

    _rewrite_header(first_path, corrupt_path, corrupt)
    with pytest.raises(ValueError, match="selector"):
        DarkoRegressor.load_model(corrupt_path)

    def corrupt_gain(header):
        records = (
            header["wrapper"]["state"]["automatic_linear_selector"],
            header["auto_params"]["automatic_linear_selector"],
            header["auto_params"]["diagnostics"][
                "automatic_linear_selector"
            ],
        )
        for record in records:
            record["paired_mse_gain_z"] += 0.5

    _rewrite_header(first_path, corrupt_gain_path, corrupt_gain)
    with pytest.raises(ValueError, match="selector"):
        DarkoRegressor.load_model(corrupt_gain_path)

    fallback = DarkoRegressor(**_params()).fit(X[:300], y[:300])
    fallback_path = tmp_path / "automatic-selector-fallback.npz"
    corrupt_fallback_path = tmp_path / "automatic-selector-fallback-corrupt.npz"
    fallback.save_model(fallback_path)

    def corrupt_fallback(header):
        selector_records = (
            header["wrapper"]["state"]["automatic_linear_selector"],
            header["auto_params"]["automatic_linear_selector"],
            header["auto_params"]["diagnostics"][
                "automatic_linear_selector"
            ],
        )
        for record in selector_records:
            record["reason"] = "forged"

    _rewrite_header(fallback_path, corrupt_fallback_path, corrupt_fallback)
    with pytest.raises(ValueError, match="selector"):
        DarkoRegressor.load_model(corrupt_fallback_path)


def test_auto_preserves_named_categorical_ordinal_and_empty_prediction():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(8)
    n_rows = 1300
    grade = rng.choice(["low", "mid", "high"], size=n_rows)
    nominal = rng.choice(["a", "b", "c"], size=n_rows)
    numeric = rng.normal(size=n_rows)
    frame = pd.DataFrame(
        {"numeric": numeric, "grade": grade, "nominal": nominal}
    )
    y = (
        1.8 * numeric
        + 0.5 * (grade == "high")
        + 0.2 * (nominal == "b")
        + rng.normal(0.0, 0.2, size=n_rows)
    )
    model = DarkoRegressor(**_params(iterations=3)).fit(
        frame,
        y,
        cat_features=["nominal"],
        ordinal_features={"grade": ["low", "mid", "high"]},
    )

    assert model.feature_names_in_.tolist() == list(frame.columns)
    assert model.automatic_linear_selector_["eligible"] is True
    assert model.predict(frame.iloc[:0]).shape == (0,)


def test_default_auto_ensemble_is_exact_fallback_and_round_trips(tmp_path):
    X, y = _smooth_data(n_samples=240)
    params = _params(iterations=3, n_ensembles=2)
    automatic = DarkoRegressor(**params).fit(X, y)
    explicit = DarkoRegressor(
        **{**params, "linear_leaves": False}
    ).fit(X, y)
    np.testing.assert_array_equal(automatic.predict(X), explicit.predict(X))
    assert automatic.automatic_linear_selector_["reason"] == "ensemble"
    assert all(member.linear_leaves is False for member in automatic.estimators_)

    path = tmp_path / "automatic-ensemble.npz"
    automatic.save_model(path)
    loaded = DarkoRegressor.load_model(path)
    assert loaded.linear_leaves == "auto"
    assert loaded.linear_leaves_selected_ is False
    np.testing.assert_array_equal(loaded.predict(X), automatic.predict(X))


def test_classifier_surface_and_behavior_do_not_gain_selector_state():
    X, y_regression = _smooth_data(n_samples=300)
    y = (y_regression > np.median(y_regression)).astype(np.int64)
    classifier = DarkoClassifier(
        iterations=3,
        depth=2,
        random_state=4,
        thread_count=1,
        diagnostic_warnings="never",
    )
    cloned = clone(classifier)
    model = classifier.fit(X, y)

    assert "linear_leaves" not in cloned.get_params(deep=False)
    assert not hasattr(model, "automatic_linear_selector_")
    assert model.predict(X[:7]).shape == (7,)
