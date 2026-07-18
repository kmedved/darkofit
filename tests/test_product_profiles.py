import numpy as np
import pytest

from darkofit import DarkoClassifier, DarkoRegressor


def _regression_data(seed=0, n=180):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = (
        1.8 * X[:, 0]
        - 0.7 * X[:, 1]
        + np.sin(1.3 * X[:, 2])
        + 0.05 * rng.normal(size=n)
    )
    return X, y


def _small_common(**overrides):
    params = {
        "depth": 2,
        "min_child_samples": 2,
        "thread_count": 1,
        "random_state": 7,
        "diagnostic_warnings": "never",
    }
    params.update(overrides)
    return params


def test_none_preset_preserves_the_existing_default_path_exactly():
    X, y = _regression_data(seed=1, n=120)
    control = DarkoRegressor(
        **_small_common(iterations=7, learning_rate=0.1)
    ).fit(X, y)
    explicit_none = DarkoRegressor(
        **_small_common(
            iterations=7, learning_rate=0.1, preset=None,
            selection_rounds=None,
        )
    ).fit(X, y)

    assert np.array_equal(control.predict(X), explicit_none.predict(X))
    assert control.model_.auto_params_ == explicit_none.model_.auto_params_
    assert not hasattr(explicit_none, "preset_")


def test_accuracy_preset_matches_the_frozen_a10_configuration(tmp_path):
    X, y = _regression_data(seed=2)
    X_train, X_val = X[:140], X[140:]
    y_train, y_val = y[:140], y[140:]
    common = _small_common(early_stopping_rounds=3)

    preset = DarkoRegressor(preset="accuracy", **common).fit(
        X_train, y_train, eval_set=(X_val, y_val)
    )
    explicit = DarkoRegressor(
        iterations=10_000,
        tree_mode="auto",
        l2_leaf_reg=3.0,
        max_bins=128,
        learning_rate=0.1,
        ts_permutations=1,
        linear_residual=False,
        early_stopping=True,
        use_best_model=True,
        **common,
    ).fit(X_train, y_train, eval_set=(X_val, y_val))

    assert np.array_equal(preset.predict(X), explicit.predict(X))
    assert preset.model_.tree_mode_ == explicit.model_.tree_mode_
    assert preset.best_n_estimators_ == explicit.best_n_estimators_
    assert preset.preset_ == "accuracy"
    metadata = preset.model_.auto_params_["preset"]
    assert metadata["claim_tier"] == "E"
    assert metadata["default_changed"] is False
    assert metadata["evidence_scope"] == "spent_development_panel"
    assert metadata["resolved"]["iterations"] == 10_000

    refit = preset.get_refit_params()
    assert refit["preset"] is None
    assert refit["selection_rounds"] is None
    assert refit["tree_mode"] == preset.model_.tree_mode_
    assert refit["iterations"] == preset.best_n_estimators_
    assert refit["learning_rate"] == preset.learning_rate_

    path = tmp_path / "accuracy.npz"
    preset.save_model(path)
    loaded = DarkoRegressor.load_model(path)
    assert loaded.preset_ == "accuracy"
    assert loaded.preset_params_ == preset.preset_params_
    assert np.array_equal(loaded.predict(X), preset.predict(X))


@pytest.mark.parametrize("preset", ["fast", object()])
def test_unknown_regression_preset_is_rejected(preset):
    X, y = _regression_data(seed=3, n=40)
    with pytest.raises(ValueError, match="preset must be"):
        DarkoRegressor(preset=preset).fit(X, y)


def test_accuracy_preset_rejects_non_rmse_losses():
    X, y = _regression_data(seed=4, n=40)
    with pytest.raises(ValueError, match="requires loss='RMSE'"):
        DarkoRegressor(preset="accuracy", loss="MAE").fit(X, y)


def test_selection_rounds_caps_auditions_then_refits_selected_mode_exactly():
    X, y = _regression_data(seed=5)
    X_train, X_val = X[:140], X[140:]
    y_train, y_val = y[:140], y[140:]
    params = _small_common(
        iterations=9,
        learning_rate=0.1,
        tree_mode="auto",
        selection_rounds=3,
        early_stopping=False,
        use_best_model=True,
    )
    selected = DarkoRegressor(**params).fit(
        X_train, y_train, eval_set=(X_val, y_val)
    )

    explicit_params = dict(params)
    explicit_params["tree_mode"] = selected.model_.tree_mode_
    explicit_params["selection_rounds"] = None
    explicit = DarkoRegressor(**explicit_params).fit(
        X_train, y_train, eval_set=(X_val, y_val)
    )

    assert np.array_equal(selected.predict(X), explicit.predict(X))
    assert selected.best_n_estimators_ == explicit.best_n_estimators_
    metadata = selected.tree_mode_selection_
    assert metadata["selection_rounds"] == 3
    assert metadata["selection_cap_active"] is True
    assert metadata["final_refit_performed"] is True
    assert metadata["final_refit_status"] == "fitted"
    assert metadata["final_iterations_requested"] == 9
    assert metadata["final_rounds_retained"] == selected.n_estimators_
    assert all(
        candidate["iterations_attempted"] <= 3
        for candidate in metadata["candidates"]
        if candidate["fit_status"] == "fitted"
    )


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_selection_rounds_validation(value):
    X, y = _regression_data(seed=6, n=60)
    with pytest.raises((TypeError, ValueError), match="selection_rounds"):
        DarkoRegressor(
            tree_mode="auto", selection_rounds=value
        ).fit(X, y)


@pytest.mark.parametrize("estimator_cls", [DarkoRegressor, DarkoClassifier])
def test_selection_rounds_requires_automatic_tree_mode(estimator_cls):
    X, y = _regression_data(seed=7, n=80)
    if estimator_cls is DarkoClassifier:
        y = (y > np.median(y)).astype(np.int64)
    with pytest.raises(ValueError, match="requires tree_mode='auto'"):
        estimator_cls(selection_rounds=3).fit(X, y)
