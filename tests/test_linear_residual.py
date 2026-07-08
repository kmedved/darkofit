import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge

import chimeraboost.linear_residual as linear_residual_module
from chimeraboost import ChimeraBoostRegressor
from chimeraboost.booster import GradientBoosting
from chimeraboost.linear_residual import (
    WeightedRidgeTrend,
    validate_linear_residual_loss,
)
from chimeraboost.serialization import save_booster


def test_weighted_ridge_trend_ols_recovers_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = 1.25 + 2.0 * X[:, 0] - 0.5 * X[:, 2]

    trend = WeightedRidgeTrend(alpha=0.0).fit(X, y)

    assert trend.active_
    assert trend.rank_ == 3
    np.testing.assert_allclose(trend.intercept_, 1.25, atol=1e-10)
    np.testing.assert_allclose(trend.coef_, [2.0, 0.0, -0.5], atol=1e-10)
    np.testing.assert_allclose(trend.predict(X), y, atol=1e-10)


def test_weighted_ridge_trend_matches_sklearn_ridge_alpha_convention():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 4))
    y = 0.3 + X @ np.array([1.1, -0.8, 0.0, 0.4])
    alpha = 0.7

    trend = WeightedRidgeTrend(
        alpha=alpha, standardize=False
    ).fit(X, y)
    expected = Ridge(alpha=alpha * X.shape[0], fit_intercept=True).fit(X, y)

    np.testing.assert_allclose(trend.intercept_, expected.intercept_, atol=1e-12)
    np.testing.assert_allclose(trend.coef_, expected.coef_, atol=1e-12)
    np.testing.assert_allclose(trend.predict(X[:10]), expected.predict(X[:10]))


def test_weighted_ridge_trend_numeric_ndarray_avoids_object_column_conversion(
    monkeypatch,
):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 12))
    coef = np.zeros(12)
    coef[[1, 5, 9]] = [0.7, -1.2, 0.4]
    y = 0.5 + X @ coef
    expected = WeightedRidgeTrend(features=[1, 5, 9], alpha=0.1).fit(X, y)

    def fail_object_matrix_conversion(*args, **kwargs):
        raise AssertionError("numeric ndarray used object matrix conversion")

    monkeypatch.setattr(
        linear_residual_module, "_as_2d_array", fail_object_matrix_conversion
    )

    trend = WeightedRidgeTrend(features=[1, 5, 9], alpha=0.1).fit(X, y)

    np.testing.assert_allclose(trend.coef_, expected.coef_)
    np.testing.assert_allclose(trend.intercept_, expected.intercept_)
    np.testing.assert_allclose(trend.predict(X[:10]), expected.predict(X[:10]))


def test_weighted_ridge_trend_sample_weight_scaling_and_zero_rows():
    X = np.array([
        [0.0, 1.0],
        [1.0, 2.0],
        [2.0, 3.0],
        [1000.0, -1000.0],
        [3.0, 4.0],
    ])
    y = 2.0 + 3.0 * X[:, 0]
    weights = np.array([1.0, 2.0, 3.0, 0.0, 4.0])

    base = WeightedRidgeTrend(alpha=0.2).fit(X, y, sample_weight=weights)
    scaled = WeightedRidgeTrend(alpha=0.2).fit(X, y, sample_weight=7.0 * weights)
    without_zero = WeightedRidgeTrend(alpha=0.2).fit(
        X[weights > 0.0], y[weights > 0.0], sample_weight=weights[weights > 0.0]
    )

    np.testing.assert_allclose(base.coef_, scaled.coef_)
    np.testing.assert_allclose(base.intercept_, scaled.intercept_)
    np.testing.assert_allclose(base.coef_, without_zero.coef_)
    np.testing.assert_allclose(base.intercept_, without_zero.intercept_)
    assert base.positive_weight_n_ == 4


def test_weighted_ridge_trend_auto_missing_constant_and_predict_imputation():
    X = np.array([
        [0.0, 1.0, np.nan],
        [1.0, 1.0, 2.0],
        [2.0, 1.0, np.inf],
        [3.0, 1.0, 4.0],
    ])
    y = 1.0 + 2.0 * X[:, 0]

    trend = WeightedRidgeTrend(alpha=0.0).fit(X, y)

    assert trend.active_
    np.testing.assert_array_equal(trend.feature_indices_, np.array([0, 2]))
    assert {record["reason"] for record in trend.dropped_features_} == {"constant"}
    pred = trend.predict(np.array([[4.0, 1.0, np.nan]]))
    assert np.isfinite(pred[0])


def test_weighted_ridge_trend_auto_no_usable_features_is_inactive():
    X = np.array([["a", 1.0], ["b", 1.0], ["c", 1.0]], dtype=object)
    y = np.array([1.0, 2.0, 3.0])

    trend = WeightedRidgeTrend().fit(X, y)

    assert not trend.active_
    assert trend.inactive_reason_ == "no_usable_auto_features"
    np.testing.assert_allclose(trend.predict(X), np.zeros(3))
    assert {record["reason"] for record in trend.dropped_features_} == {
        "non_numeric",
        "constant",
    }


def test_weighted_ridge_trend_explicit_feature_validation():
    X = np.array([
        [0.0, "a", np.nan, 1.0],
        [1.0, "b", np.nan, 1.0],
        [2.0, "c", np.nan, 1.0],
    ], dtype=object)
    y = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="categorical"):
        WeightedRidgeTrend(features=[1]).fit(X, y, cat_features=[1])
    with pytest.raises(ValueError, match="non-numeric"):
        WeightedRidgeTrend(features=[1]).fit(X, y)
    with pytest.raises(ValueError, match="no finite"):
        WeightedRidgeTrend(features=[2]).fit(X, y)
    with pytest.raises(ValueError, match="duplicate"):
        WeightedRidgeTrend(features=[0, 0]).fit(X, y)
    with pytest.raises(ValueError, match="must not mix"):
        WeightedRidgeTrend(features=[0, "b"]).fit(
            X, y, feature_names=np.array(["0", "b", "c", "d"], dtype=object)
        )

    trend = WeightedRidgeTrend(features=[0, 3], alpha=0.0).fit(X, y)
    np.testing.assert_array_equal(trend.feature_indices_, np.array([0]))
    assert trend.dropped_features_[0]["reason"] == "constant"


def test_weighted_ridge_trend_named_features_round_trip():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({
        "x": [0.0, 1.0, 2.0, 3.0],
        "drop": [5.0, 5.0, 5.0, 5.0],
        "z": [1.0, 2.0, 1.0, 2.0],
    })
    y = np.array([1.0, 3.0, 5.0, 7.0])

    trend = WeightedRidgeTrend(features=["x", "drop"], alpha=0.0).fit(
        X, y, feature_names=np.asarray(X.columns, dtype=object)
    )
    restored = WeightedRidgeTrend.from_payload(
        {
            **trend.state_header(),
            "linear_residual_enabled": True,
        },
        trend.state_arrays(),
        n_features=X.shape[1],
    )

    np.testing.assert_array_equal(restored.feature_indices_, np.array([0]))
    assert restored.feature_names_.tolist() == ["x"]
    np.testing.assert_allclose(restored.predict(X), trend.predict(X))


def test_weighted_ridge_trend_payload_validation_rejects_corruption():
    X = np.arange(12.0).reshape(6, 2)
    y = 1.0 + X[:, 0]
    trend = WeightedRidgeTrend(alpha=0.0).fit(X, y)
    state = {**trend.state_header(), "linear_residual_enabled": True}
    arrays = dict(trend.state_arrays())

    bad_arrays = dict(arrays)
    bad_arrays["linear_residual_feature_indices"] = np.array([0, 0])
    with pytest.raises(ValueError, match="duplicates"):
        WeightedRidgeTrend.from_payload(state, bad_arrays, n_features=2)

    bad_arrays = dict(arrays)
    bad_arrays["linear_residual_scale"] = np.zeros_like(arrays["linear_residual_scale"])
    with pytest.raises(ValueError, match="positive"):
        WeightedRidgeTrend.from_payload(state, bad_arrays, n_features=2)


def test_validate_linear_residual_loss_rejects_v2_distributions():
    validate_linear_residual_loss("RMSE")
    validate_linear_residual_loss("Gaussian")
    validate_linear_residual_loss("StudentT")
    for loss in ("LogNormal", "Poisson", "NegativeBinomial"):
        with pytest.raises(ValueError, match=loss):
            validate_linear_residual_loss(loss)


def _wrapper_params(**overrides):
    params = dict(
        iterations=8,
        learning_rate=0.1,
        depth=2,
        min_child_samples=2,
        random_state=0,
        thread_count=1,
        linear_residual=True,
        linear_residual_alpha=0.0,
    )
    params.update(overrides)
    return params


def _read_archive_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _write_archive(path, arrays):
    np.savez_compressed(path, **arrays)
    return path


def test_regressor_predict_adds_linear_trend_to_residual_booster():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 3))
    y = 1.5 + 2.0 * X[:, 0] - X[:, 1] + 0.25 * np.sin(X[:, 2])

    model = ChimeraBoostRegressor(**_wrapper_params()).fit(X, y)
    Xq = X[:12]
    trend = model._linear_residual_trend(Xq)
    raw = model.model_.predict_raw(Xq)

    assert model.linear_residual_enabled_
    assert model.linear_residual_active_
    np.testing.assert_allclose(model.predict(Xq), raw + trend)
    np.testing.assert_allclose(
        model.model_.auto_params_["linear_residual"]["feature_indices"],
        model.linear_residual_feature_indices_,
    )


def test_regressor_staged_predict_adds_same_linear_trend_each_stage():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(70, 2))
    y = 0.2 + 1.8 * X[:, 0] + np.cos(X[:, 1])

    model = ChimeraBoostRegressor(**_wrapper_params(iterations=5)).fit(X, y)
    Xq = X[:8]
    trend = model._linear_residual_trend(Xq)
    public_stages = list(model.staged_predict(Xq))
    raw_stages = list(model.model_.staged_predict_raw(Xq))

    assert len(public_stages) == len(raw_stages)
    for public, raw in zip(public_stages, raw_stages):
        np.testing.assert_allclose(public, raw + trend)
    np.testing.assert_allclose(public_stages[-1], model.predict(Xq))


def test_regressor_explicit_eval_set_does_not_fit_linear_trend():
    Xtr = np.linspace(-1.0, 1.0, 40)[:, None]
    ytr = 1.0 + 2.0 * Xtr[:, 0]
    Xv = np.linspace(10.0, 20.0, 12)[:, None]
    yv = -500.0 + 100.0 * Xv[:, 0]
    expected = WeightedRidgeTrend(alpha=0.0).fit(Xtr, ytr)

    model = ChimeraBoostRegressor(
        **_wrapper_params(
            iterations=4,
            early_stopping=True,
            early_stopping_rounds=2,
            use_best_model=True,
        )
    ).fit(Xtr, ytr, eval_set=(Xv, yv))

    np.testing.assert_allclose(model.linear_residual_coef_, expected.coef_)
    np.testing.assert_allclose(
        model.linear_residual_intercept_, expected.intercept_
    )
    assert model.model_.auto_params_["validation_split"]["source"] == (
        "explicit_eval_set"
    )


def test_regressor_rejects_v2_distributional_losses_with_linear_residual():
    X = np.linspace(0.1, 2.0, 20)[:, None]
    y = np.exp(0.2 + X[:, 0])

    for loss in ("LogNormal", "Poisson", "NegativeBinomial"):
        with pytest.raises(ValueError, match=loss):
            ChimeraBoostRegressor(
                **_wrapper_params(loss=loss, tree_mode="lightgbm")
            ).fit(X, y)


def _distribution_params(**overrides):
    params = _wrapper_params(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=6,
        num_leaves=5,
        min_child_samples=2,
    )
    params.update(overrides)
    return params


def test_gaussian_linear_residual_shifts_location_not_variance_interval_sample():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(90, 3))
    y = 0.7 + 1.4 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(scale=0.2, size=90)
    model = ChimeraBoostRegressor(**_distribution_params()).fit(X, y)
    Xq = X[:10]

    trend = model._linear_residual_trend(Xq)
    raw = model.model_.predict_raw(Xq)
    residual_mu, residual_sigma = model._calibrated_params_from_raw(raw, Xq)
    public_mu, public_sigma = model.predict_dist(Xq)

    np.testing.assert_allclose(public_mu, residual_mu + trend)
    np.testing.assert_allclose(public_sigma, residual_sigma)
    np.testing.assert_allclose(model.predict(Xq), public_mu)
    np.testing.assert_allclose(model.predict_variance(Xq), residual_sigma ** 2)

    residual_lo, residual_hi = model.model_.loss_.interval_from_raw(raw, 0.2)
    public_lo, public_hi = model.predict_interval(Xq, alpha=0.2)
    np.testing.assert_allclose(public_lo, residual_lo + trend)
    np.testing.assert_allclose(public_hi, residual_hi + trend)

    residual_draws = model.model_.loss_.sample_from_raw(
        raw, np.random.default_rng(123), 4
    )
    public_draws = model.sample(Xq, n_samples=4, random_state=123)
    np.testing.assert_allclose(public_draws, residual_draws + trend[:, None])


def test_student_t_linear_residual_shifts_only_mu():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(80, 2))
    y = -0.2 + 1.2 * X[:, 0] + rng.standard_t(5.0, size=80) * 0.15
    model = ChimeraBoostRegressor(
        **_distribution_params(loss="StudentT", dist_params={"nu": 5.0})
    ).fit(X, y)
    Xq = X[:8]

    trend = model._linear_residual_trend(Xq)
    raw = model.model_.predict_raw(Xq)
    residual_mu, residual_scale, residual_nu = model._calibrated_params_from_raw(
        raw, Xq
    )
    public_mu, public_scale, public_nu = model.predict_dist(Xq)

    np.testing.assert_allclose(public_mu, residual_mu + trend)
    np.testing.assert_allclose(public_scale, residual_scale)
    np.testing.assert_allclose(public_nu, residual_nu)
    np.testing.assert_allclose(
        model.predict_variance(Xq),
        residual_scale ** 2 * (5.0 / 3.0),
    )


def test_refit_uses_full_data_linear_residual_trend_and_keeps_selection_summary():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(100, 3))
    y = 0.5 + 0.9 * X[:, 0] - 0.4 * X[:, 2] + 0.1 * rng.normal(size=100)
    expected_full = WeightedRidgeTrend(alpha=0.0).fit(X, y)

    model = ChimeraBoostRegressor(
        **_wrapper_params(
            iterations=10,
            early_stopping=True,
            early_stopping_rounds=3,
            refit=True,
            refit_strategy="exact",
            validation_fraction=0.2,
        )
    ).fit(X, y)

    assert model.refit_
    np.testing.assert_allclose(model.linear_residual_coef_, expected_full.coef_)
    np.testing.assert_allclose(
        model.linear_residual_intercept_, expected_full.intercept_
    )
    metadata = model.model_.auto_params_["linear_residual"]
    assert metadata["active"]
    assert metadata["selection_summary"]["enabled"]


def test_linear_residual_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(14)
    X = rng.normal(size=(70, 3))
    y = -1.0 + 1.7 * X[:, 1] + 0.2 * np.sin(X[:, 0])
    model = ChimeraBoostRegressor(**_wrapper_params(iterations=5)).fit(X, y)
    path = tmp_path / "linear-residual.npz"

    model.save_model(path)
    loaded = ChimeraBoostRegressor.load_model(path)

    assert loaded.linear_residual_enabled_
    assert loaded.linear_residual_active_
    np.testing.assert_allclose(loaded.linear_residual_coef_, model.linear_residual_coef_)
    np.testing.assert_allclose(loaded.predict(X[:12]), model.predict(X[:12]))
    np.testing.assert_allclose(
        loaded.model_.auto_params_["linear_residual"]["feature_indices"],
        model.model_.auto_params_["linear_residual"]["feature_indices"],
    )


def test_regressor_loads_wrapperless_scalar_archive_with_fitted_loss(tmp_path):
    X = np.linspace(-1.0, 1.0, 40)[:, None]
    y = np.abs(X[:, 0]) + 0.1 * X[:, 0]
    core = GradientBoosting(
        loss="MAE",
        iterations=4,
        learning_rate=0.1,
        depth=2,
        random_state=0,
        thread_count=1,
    ).fit(X, y)
    path = tmp_path / "core-mae.npz"
    save_booster(core, path)

    loaded = ChimeraBoostRegressor.load_model(path)

    assert loaded.loss == "MAE"
    np.testing.assert_allclose(loaded.predict(X[:10]), core.predict_raw(X[:10]))


def test_linear_residual_archive_corruption_is_rejected(tmp_path):
    X = np.arange(30.0).reshape(15, 2)
    y = 1.0 + 0.2 * X[:, 0]
    model = ChimeraBoostRegressor(**_wrapper_params(iterations=3)).fit(X, y)
    path = tmp_path / "linear-residual-corrupt.npz"
    model.save_model(path)
    arrays = _read_archive_arrays(path)

    missing = dict(arrays)
    missing.pop("wrapper__linear_residual_coef")
    missing_path = _write_archive(tmp_path / "missing-linear-coef.npz", missing)
    with pytest.raises(ValueError, match="missing linear residual arrays"):
        ChimeraBoostRegressor.load_model(missing_path)

    bad_index = dict(arrays)
    bad_index["wrapper__linear_residual_feature_indices"] = np.array([99])
    bad_index_path = _write_archive(tmp_path / "bad-linear-index.npz", bad_index)
    with pytest.raises(ValueError, match="out of range"):
        ChimeraBoostRegressor.load_model(bad_index_path)

    disabled_active = dict(arrays)
    header = json.loads(str(disabled_active["header"]))
    header["wrapper"]["state"]["linear_residual_enabled"] = False
    header["wrapper"]["state"]["linear_residual_active"] = True
    disabled_active["header"] = np.array(json.dumps(header))
    disabled_active_path = _write_archive(
        tmp_path / "disabled-active-linear.npz", disabled_active
    )
    with pytest.raises(ValueError, match="cannot be disabled"):
        ChimeraBoostRegressor.load_model(disabled_active_path)


def test_linear_residual_archive_feature_name_mismatch_is_rejected(tmp_path):
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({
        "a": np.linspace(0.0, 1.0, 20),
        "b": np.linspace(1.0, 2.0, 20),
    })
    y = 0.3 + 2.0 * X["a"].to_numpy()
    model = ChimeraBoostRegressor(
        **_wrapper_params(iterations=3, linear_residual_features=["a"])
    ).fit(X, y)
    path = tmp_path / "linear-residual-names.npz"
    model.save_model(path)
    arrays = _read_archive_arrays(path)
    header = json.loads(str(arrays["header"]))
    header["wrapper"]["state"]["linear_residual_feature_names"] = ["b"]
    arrays["header"] = np.array(json.dumps(header))

    bad_path = _write_archive(tmp_path / "bad-linear-name.npz", arrays)
    with pytest.raises(ValueError, match="feature names"):
        ChimeraBoostRegressor.load_model(bad_path)
