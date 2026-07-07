import json
import warnings

import numpy as np
import pytest

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.booster import DistributionalBoosting, MulticlassBoosting
from chimeraboost.losses import (
    GaussianNLL,
    _GAUSS_RHO_MAX,
    _GAUSS_RHO_MIN,
    _GAUSS_Z_CLIP,
)
from chimeraboost.tuning import ChimeraBoostStepwiseSearchCV


def _make_heteroscedastic(seed=0, n=160):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    sigma = 0.25 + 0.35 * np.abs(X[:, 1])
    y = 1.1 * X[:, 0] - 0.7 * X[:, 2] + rng.normal(scale=sigma)
    return X, y


def _make_spec_heteroscedastic(seed=123, n_train=20_000, n_test=5_000):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_train + n_test, 4))
    sigma = 0.3 + np.abs(X[:, 1])
    y = np.sin(3.0 * X[:, 0]) + sigma * rng.normal(size=X.shape[0])
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:], sigma[n_train:]


def _row_nll(y, mu, rho):
    r = float(np.clip(rho, _GAUSS_RHO_MIN, _GAUSS_RHO_MAX))
    sigma = np.exp(r)
    z = float(np.clip((y - mu) / sigma, -_GAUSS_Z_CLIP, _GAUSS_Z_CLIP))
    return 0.5 * np.log(2.0 * np.pi) + r + 0.5 * z * z


def _gaussian_test_params(**overrides):
    params = dict(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=8,
        learning_rate=0.1,
        num_leaves=7,
        min_child_samples=3,
        random_state=0,
        diagnostic_warnings="never",
        thread_count=1,
    )
    params.update(overrides)
    return params


def test_gaussian_loss_math_and_zero_weight_safety():
    y = np.array([0.2, -0.8, 1.1, 0.7], dtype=np.float64)
    F = np.array([
        [0.0, -0.5, 1.0, 0.5],
        [-0.1, 0.2, -0.2, 0.1],
    ], dtype=np.float64)
    weights = np.array([1.0, 0.7, 1.5, 2.0], dtype=np.float64)
    loss = GaussianNLL()

    for sample_weight in (None, weights):
        grad = np.empty_like(F)
        hess = np.empty_like(F)
        loss.grad_hess_class_major_into(y, F, sample_weight, grad, hess)
        for k in range(F.shape[0]):
            for i in range(F.shape[1]):
                w = 1.0 if sample_weight is None else sample_weight[i]
                eps = 1e-6
                plus = F.copy()
                minus = F.copy()
                plus[k, i] += eps
                minus[k, i] -= eps
                fd = (
                    w * _row_nll(y[i], plus[0, i], plus[1, i])
                    - w * _row_nll(y[i], minus[0, i], minus[1, i])
                ) / (2.0 * eps)
                assert grad[k, i] == pytest.approx(fd, rel=1e-5, abs=1e-7)

                if k == 0:
                    center = w * _row_nll(y[i], F[0, i], F[1, i])
                    second = (
                        w * _row_nll(y[i], plus[0, i], plus[1, i])
                        - 2.0 * center
                        + w * _row_nll(y[i], minus[0, i], minus[1, i])
                    ) / (eps * eps)
                    assert hess[k, i] == pytest.approx(second, rel=1e-3, abs=1e-4)

    F_extreme = np.array([
        [0.0, 0.0],
        [1000.0, -1000.0],
    ], dtype=np.float64)
    y_extreme = np.array([1e300, -1e300], dtype=np.float64)
    w_extreme = np.array([0.0, 1.0], dtype=np.float64)
    grad = np.empty_like(F_extreme)
    hess = np.empty_like(F_extreme)
    loss.grad_hess_class_major_into(
        y_extreme, F_extreme, w_extreme, grad, hess
    )
    assert np.array_equal(grad[:, 0], np.zeros(2))
    assert np.array_equal(hess[:, 0], np.zeros(2))
    assert np.all(np.isfinite(grad[:, 1]))
    assert np.all(np.isfinite(hess[:, 1]))
    assert np.isfinite(loss.eval_class_major(y_extreme, F_extreme, w_extreme))
    assert np.isfinite(loss.crps_class_major(y_extreme, F_extreme, w_extreme))


def test_gaussian_init_constant_target_and_crps_mc():
    y = np.array([1.0, 2.0, 5.0, 7.0], dtype=np.float64)
    weights = np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float64)
    loss = GaussianNLL()
    init = loss.init_class_major(y, weights)
    mu0 = np.average(y, weights=weights)
    var0 = np.average((y - mu0) ** 2, weights=weights)
    np.testing.assert_allclose(
        init, np.array([mu0, 0.5 * np.log(var0)]), rtol=0.0, atol=1e-15
    )
    extreme_init = loss.init_class_major(
        np.array([1e300, 2.0, 4.0], dtype=np.float64),
        np.array([0.0, 1.0, 3.0], dtype=np.float64),
    )
    np.testing.assert_allclose(
        extreme_init,
        np.array([3.5, 0.5 * np.log(0.75)]),
        rtol=0.0,
        atol=1e-15,
    )

    const = np.full(60, 3.25)
    X_const = np.linspace(0.0, 1.0, const.size)[:, None]
    model = ChimeraBoostRegressor(
        **_gaussian_test_params(iterations=2, num_leaves=3, min_child_samples=2)
    ).fit(X_const, const)
    mu, sigma = model.predict_dist(X_const[:5])
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(sigma))

    y_grid = np.array([-1.0, 0.2, 2.0], dtype=np.float64)
    mu_grid = np.array([-0.8, 0.0, 1.5], dtype=np.float64)
    sigma_grid = np.array([0.4, 1.2, 2.0], dtype=np.float64)
    F = np.vstack([mu_grid, np.log(sigma_grid)])
    closed_form = loss.crps_class_major(y_grid, F)

    rng = np.random.default_rng(42)
    draws_a = rng.normal(mu_grid[:, None], sigma_grid[:, None], size=(3, 100_000))
    draws_b = rng.normal(mu_grid[:, None], sigma_grid[:, None], size=(3, 100_000))
    mc = (
        np.mean(np.abs(draws_a - y_grid[:, None]), axis=1)
        - 0.5 * np.mean(np.abs(draws_a - draws_b), axis=1)
    )
    assert closed_form == pytest.approx(float(np.mean(mc)), abs=5e-3)
    assert closed_form >= 0.0


def test_gaussian_train_nll_decreases():
    X, y = _make_heteroscedastic(seed=6, n=800)
    model = ChimeraBoostRegressor(
        **_gaussian_test_params(
            iterations=50,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=10,
            eval_train_loss=True,
        )
    ).fit(X, y)
    history = np.asarray(model.model_.train_history_)
    assert history.shape == (50,)
    assert np.all(np.diff(history) <= 1e-10)


def test_gaussian_heteroscedastic_recovery_and_point_quality():
    Xtr, Xte, ytr, yte, sigma_true = _make_spec_heteroscedastic()
    params = dict(
        tree_mode="lightgbm",
        iterations=60,
        learning_rate=0.08,
        num_leaves=31,
        min_child_samples=40,
        random_state=0,
        diagnostic_warnings="never",
        thread_count=1,
    )
    gaussian = ChimeraBoostRegressor(loss="Gaussian", **params).fit(Xtr, ytr)
    rmse = ChimeraBoostRegressor(loss="RMSE", **params).fit(Xtr, ytr)

    mu, sigma_hat = gaussian.predict_dist(Xte)
    corr = float(np.corrcoef(sigma_hat, sigma_true)[0, 1])
    lo, hi = gaussian.predict_interval(Xte, alpha=0.1)
    coverage = float(np.mean((yte >= lo) & (yte <= hi)))
    gaussian_rmse = float(np.sqrt(np.mean((yte - mu) ** 2)))
    rmse_rmse = float(np.sqrt(np.mean((yte - rmse.predict(Xte)) ** 2)))

    assert corr > 0.8
    assert 0.86 <= coverage <= 0.94
    assert gaussian_rmse <= 1.05 * rmse_rmse


def test_gaussian_homoscedastic_sigma_sanity():
    rng = np.random.default_rng(2)
    n_train, n_test = 4_000, 1_000
    X = rng.normal(size=(n_train + n_test, 4))
    y = 0.7 * X[:, 0] - 0.2 * X[:, 2] + 2.0 * rng.normal(size=X.shape[0])
    model = ChimeraBoostRegressor(
        **_gaussian_test_params(
            iterations=50,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=40,
        )
    ).fit(X[:n_train], y[:n_train])
    _, sigma_hat = model.predict_dist(X[n_train:])
    assert 1.8 <= float(np.median(sigma_hat)) <= 2.2


def test_gaussian_weight_normalization_invariant():
    X, y = _make_heteroscedastic(seed=7, n=180)
    y = y.copy()
    y[:2] = np.array([1e100, -1e100])
    weights = np.ones(X.shape[0])
    weights[:2] = 0.0

    base = ChimeraBoostRegressor(
        **_gaussian_test_params(iterations=5, min_child_samples=4)
    ).fit(X, y, sample_weight=weights)
    scaled = ChimeraBoostRegressor(
        **_gaussian_test_params(iterations=5, min_child_samples=4)
    ).fit(X, y, sample_weight=weights * 17.0)

    for left, right in zip(base.predict_dist(X[:20]), scaled.predict_dist(X[:20])):
        np.testing.assert_allclose(left, right, rtol=0.0, atol=0.0)


def test_gaussian_regressor_api_alias_and_auto_metadata():
    X, y = _make_heteroscedastic(seed=1)
    model = ChimeraBoostRegressor(
        loss="Gaussian",
        tree_mode="leafwise",
        iterations=4,
        min_child_samples=3,
        num_leaves=7,
        random_state=0,
        diagnostic_warnings="never",
    ).fit(X, y)

    assert isinstance(model.model_, DistributionalBoosting)
    mu, sigma = model.predict_dist(X[:12])
    assert np.array_equal(model.predict(X[:12]), mu)
    assert mu.shape == sigma.shape == (12,)
    assert np.all(np.isfinite(mu))
    assert np.all(sigma > 0.0)

    stages = list(model.staged_predict(X[:12]))
    assert stages[-1].shape == (12,)
    assert np.array_equal(stages[-1], model.predict(X[:12]))

    lo, hi = model.predict_interval(X[:12], alpha=0.2)
    assert lo.shape == hi.shape == (12,)
    assert np.all(lo < hi)
    draws = model.sample(X[:12], n_samples=3, random_state=0)
    assert draws.shape == (12, 3)
    assert np.all(np.isfinite(draws))

    many_draws = model.sample(X[:12], n_samples=2_000, random_state=1)
    np.testing.assert_allclose(many_draws.mean(axis=1), mu, atol=0.12)
    np.testing.assert_allclose(many_draws.std(axis=1), sigma, rtol=0.15, atol=0.05)

    lr_meta = model.model_.auto_params_["learning_rate"]
    assert lr_meta["loss_coefficient_source"] == "rmse_coefs_for_gaussian"
    assert model.model_.auto_params_["distributional"]["n_outputs"] == 2


def test_gaussian_early_stopping_eval_weights_and_refit_params():
    X, y = _make_heteroscedastic(seed=8, n=800)
    Xtr, Xv = X[:500], X[500:]
    ytr, yv = y[:500], y[500:]
    sample_weight = np.linspace(0.8, 1.4, Xtr.shape[0])
    eval_weight = np.linspace(1.3, 0.7, Xv.shape[0])

    model = ChimeraBoostRegressor(
        **_gaussian_test_params(
            iterations=80,
            learning_rate=0.1,
            num_leaves=15,
            min_child_samples=10,
            early_stopping=True,
            early_stopping_rounds=3,
            use_best_model=True,
        )
    ).fit(
        Xtr, ytr,
        eval_set=(Xv, yv),
        sample_weight=sample_weight,
        eval_sample_weight=eval_weight,
    )

    assert 0 < model.n_estimators_ < 80
    assert model.best_iteration_ == model.n_estimators_
    assert len(model.model_.valid_history_) > model.n_estimators_
    params = model.get_refit_params()
    assert params["loss"] == "Gaussian"
    assert params["early_stopping"] is False
    assert params["iterations"] == model.n_estimators_


def test_gaussian_crps_eval_metric_controls_validation_history():
    X, y = _make_heteroscedastic(seed=13, n=140)
    Xtr, Xv = X[:100], X[100:]
    ytr, yv = y[:100], y[100:]

    model = ChimeraBoostRegressor(
        **_gaussian_test_params(
            iterations=1,
            eval_metric="crps",
        )
    ).fit(Xtr, ytr, eval_set=(Xv, yv))

    raw = model.model_.predict_raw(Xv).T
    expected = model.model_.loss_.crps_class_major(yv, raw)
    assert model.model_.auto_params_["distributional"]["eval_metric"] == "crps"
    assert np.isclose(model.model_.valid_history_[-1], expected)
    assert np.isclose(model.best_score_, expected)

    with pytest.raises(ValueError, match="eval_metric"):
        ChimeraBoostRegressor(
            loss="RMSE",
            eval_metric="crps",
            iterations=1,
        ).fit(Xtr, ytr)


def test_gaussian_scalar_sigma_calibration_scales_public_distribution_only():
    X, y = _make_heteroscedastic(seed=14, n=180)
    Xtr, Xv = X[:120], X[120:]
    ytr, yv = y[:120], y[120:]
    yv_cal = yv.copy()
    yv_cal[0] = 1e100
    eval_weight = np.ones_like(yv_cal)
    eval_weight[0] = 0.0

    model = ChimeraBoostRegressor(
        **_gaussian_test_params(
            iterations=10,
            learning_rate=0.08,
            sigma_calibration="scalar",
        )
    ).fit(Xtr, ytr, eval_set=(Xv, yv_cal), eval_sample_weight=eval_weight)

    raw_mu, raw_sigma = model.model_.predict_dist(Xv)
    positive = eval_weight > 0.0
    expected_scale = float(
        np.sqrt(
            np.average(
                (
                    (yv_cal[positive] - raw_mu[positive])
                    / np.maximum(raw_sigma[positive], 1e-12)
                ) ** 2,
                weights=eval_weight[positive],
            )
        )
    )
    public_mu, public_sigma = model.predict_dist(Xv)
    np.testing.assert_allclose(model.sigma_scale_, expected_scale)
    np.testing.assert_allclose(public_mu, raw_mu, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        public_sigma, raw_sigma * model.sigma_scale_, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(model.predict(Xv), raw_mu, rtol=0.0, atol=0.0)
    calibration_meta = model.model_.auto_params_["sigma_calibration"]
    assert calibration_meta["method"] == "scalar"
    assert calibration_meta["sigma_scale"] == model.sigma_scale_
    assert calibration_meta["source"] == "selection_validation"
    assert calibration_meta["validation_n_samples"] == yv_cal.shape[0]
    assert calibration_meta["validation_positive_weight_n"] == positive.sum()
    assert calibration_meta["validation_effective_n"] == pytest.approx(
        float(positive.sum())
    )
    assert calibration_meta["small_fold_warning"] is True
    warning_codes = {
        warning["code"]
        for warning in model.model_.auto_params_["diagnostics"]["warnings"]
    }
    assert "small_sigma_calibration_fold" in warning_codes


def test_gaussian_small_sigma_calibration_warning_respects_reset():
    import chimeraboost.booster as booster_mod

    X, y = _make_heteroscedastic(seed=16, n=160)
    params = _gaussian_test_params(
        iterations=2,
        learning_rate=0.08,
        sigma_calibration="scalar",
        diagnostic_warnings="once",
    )

    def fit_and_messages():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ChimeraBoostRegressor(**params).fit(
                X[:120], y[:120], eval_set=(X[120:], y[120:])
            )
        return [str(warning.message) for warning in caught]

    booster_mod.reset_diagnostic_warning_registry()
    try:
        assert any(
            "sigma_calibration='scalar'" in msg for msg in fit_and_messages()
        )
        assert not any(
            "sigma_calibration='scalar'" in msg for msg in fit_and_messages()
        )
        booster_mod.reset_diagnostic_warning_registry()
        assert any(
            "sigma_calibration='scalar'" in msg for msg in fit_and_messages()
        )
    finally:
        booster_mod.reset_diagnostic_warning_registry()


def test_gaussian_refit_emits_small_calibration_warning_once():
    X, y = _make_heteroscedastic(seed=17, n=160)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ChimeraBoostRegressor(
            **_gaussian_test_params(
                iterations=4,
                learning_rate=0.08,
                early_stopping=True,
                early_stopping_rounds=2,
                validation_fraction=0.25,
                refit=True,
                sigma_calibration="scalar",
                diagnostic_warnings="always",
            )
        ).fit(X, y)

    sigma_messages = [
        str(warning.message)
        for warning in caught
        if "sigma_calibration='scalar'" in str(warning.message)
    ]
    assert len(sigma_messages) == 1
    emitted = model.model_.auto_params_["diagnostics"][
        "runtime_warnings_emitted"
    ]
    assert emitted.count("small_sigma_calibration_fold") == 1


def test_gaussian_sigma_calibration_requires_validation_and_survives_refit_load(
    tmp_path,
):
    X, y = _make_heteroscedastic(seed=15, n=180)

    with pytest.raises(ValueError, match="requires a validation set"):
        ChimeraBoostRegressor(
            loss="Gaussian",
            tree_mode="lightgbm",
            sigma_calibration="scalar",
            iterations=2,
            random_state=0,
            diagnostic_warnings="never",
        ).fit(X, y)
    with pytest.raises(ValueError, match="only supported for loss='Gaussian'"):
        ChimeraBoostRegressor(
            loss="RMSE",
            sigma_calibration="scalar",
            iterations=2,
        ).fit(X, y, eval_set=(X[:20], y[:20]))

    model = ChimeraBoostRegressor(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=12,
        learning_rate=0.08,
        min_child_samples=3,
        num_leaves=7,
        early_stopping=True,
        early_stopping_rounds=3,
        validation_fraction=0.25,
        refit=True,
        sigma_calibration=True,
        random_state=0,
        diagnostic_warnings="never",
        thread_count=1,
    ).fit(X, y)

    assert model.refit_ is True
    assert model.sigma_calibration_ == "scalar"
    assert model.sigma_scale_ > 0.0
    assert model.sigma_scale_source_ == "selection_validation"
    _, raw_sigma = model.model_.predict_dist(X[:12])
    _, public_sigma = model.predict_dist(X[:12])
    np.testing.assert_allclose(
        public_sigma, raw_sigma * model.sigma_scale_, rtol=0.0, atol=0.0
    )

    path = tmp_path / "calibrated_gaussian.npz"
    model.save_model(path)
    with np.load(path, allow_pickle=False) as data:
        header = json.loads(str(data["header"]))
    state = header["wrapper"]["state"]
    assert state["sigma_calibration"] == "scalar"
    assert state["sigma_scale"] == model.sigma_scale_
    loaded = ChimeraBoostRegressor.load_model(path)
    assert loaded.sigma_calibration_ == "scalar"
    assert loaded.sigma_scale_ == model.sigma_scale_
    np.testing.assert_allclose(
        loaded.predict_dist(X[:12])[1], public_sigma, rtol=0.0, atol=0.0
    )


def test_gaussian_refit_uses_distributional_booster():
    X, y = _make_heteroscedastic(seed=2, n=120)
    model = ChimeraBoostRegressor(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=8,
        learning_rate=0.15,
        min_child_samples=3,
        num_leaves=7,
        early_stopping=True,
        early_stopping_rounds=3,
        validation_fraction=0.25,
        refit=True,
        random_state=0,
    ).fit(X, y)

    assert model.refit_ is True
    assert isinstance(model.selection_model_, DistributionalBoosting)
    assert isinstance(model.model_, DistributionalBoosting)
    assert model.get_refit_params()["loss"] == "Gaussian"


def test_gaussian_auto_learning_rate_probe_runs():
    X, y = _make_heteroscedastic(seed=9, n=180)
    model = ChimeraBoostRegressor(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=5,
        learning_rate=None,
        auto_learning_rate_probe=True,
        auto_learning_rate_probe_values=[0.03, 0.06],
        auto_learning_rate_probe_iterations=2,
        min_child_samples=3,
        num_leaves=7,
        random_state=0,
        diagnostic_warnings="never",
        thread_count=1,
    ).fit(X[:120], y[:120], eval_set=(X[120:], y[120:]))

    probe = model.model_.auto_params_["learning_rate_probe"]
    assert probe["enabled"] is True
    assert probe["selected_learning_rate"] == model.learning_rate_
    assert probe["base_learning_rate_details"]["loss_coefficient_source"] == (
        "rmse_coefs_for_gaussian"
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"tree_mode": "catboost"}, "tree_mode='lightgbm'"),
        ({"tree_mode": "auto"}, "tree_mode='auto'"),
        ({"tree_mode": "hybrid"}, "tree_mode='lightgbm'"),
        ({"tree_mode": "depthwise"}, "tree_mode='lightgbm'"),
        ({"sampling": "goss"}, "GOSS and MVS"),
        ({"sampling": "mvs", "subsample": 0.5}, "GOSS and MVS"),
        (
            {"bootstrap_type": "bayesian", "bagging_temperature": 0.5},
            "Bayesian bootstrap",
        ),
        ({"ordered_boosting": True}, "ordered_boosting"),
        ({"histogram_dtype": "float32"}, "histogram_dtype='float32'"),
        ({"alpha": 0.9}, "alpha"),
        ({"tree_mode": "leaf_wise"}, None),
        ({"subsample": 0.5}, None),
        ({"colsample": 0.5}, None),
    ],
)
def test_gaussian_guards(kwargs, match):
    X, y = _make_heteroscedastic(seed=3, n=80)
    params = dict(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=1,
        learning_rate=0.1,
        min_child_samples=2,
        num_leaves=3,
        random_state=0,
    )
    params.update(kwargs)
    if match is None:
        ChimeraBoostRegressor(**params).fit(X, y)
        return
    with pytest.raises(ValueError, match=match):
        ChimeraBoostRegressor(**params).fit(X, y)


def test_gaussian_uniform_subsample_and_colsample_metadata():
    X, y = _make_heteroscedastic(seed=12, n=220)
    model = ChimeraBoostRegressor(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=6,
        learning_rate=0.08,
        min_child_samples=3,
        num_leaves=7,
        subsample=0.55,
        colsample=0.6,
        random_state=0,
        diagnostic_warnings="never",
        thread_count=1,
    ).fit(X, y)

    mu, sigma = model.predict_dist(X[:15])
    assert mu.shape == sigma.shape == (15,)
    assert np.all(np.isfinite(mu))
    assert np.all(sigma > 0.0)
    stochastic = model.model_.auto_params_["stochastic_regularization"]
    assert stochastic["row_sampling_active"] is True
    assert stochastic["average_sampled_row_fraction"] < 1.0


def test_gaussian_sampled_depth_zero_retries_are_capped():
    from chimeraboost.booster import _MAX_CONSECUTIVE_SAMPLED_DEPTH0_RETRIES

    X = np.zeros((80, 3), dtype=np.float64)
    y = np.zeros(80, dtype=np.float64)
    model = ChimeraBoostRegressor(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=1000,
        subsample=0.5,
        random_state=0,
        diagnostic_warnings="never",
    ).fit(X, y)

    meta = model.model_.auto_params_["stochastic_regularization"]
    assert model.best_iteration_ == 0
    assert meta["sampling_rounds"] == _MAX_CONSECUTIVE_SAMPLED_DEPTH0_RETRIES + 1


def test_gaussian_only_methods_and_tuner_lanes():
    X, y = _make_heteroscedastic(seed=4, n=80)
    rmse = ChimeraBoostRegressor(iterations=1, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="predict_dist"):
        rmse.predict_dist(X[:2])
    with pytest.raises(ValueError, match="predict_interval"):
        rmse.predict_interval(X[:2])
    with pytest.raises(ValueError, match="sample"):
        rmse.sample(X[:2])

    search = ChimeraBoostStepwiseSearchCV(
        ChimeraBoostRegressor(
            loss="Gaussian",
            tree_mode="lightgbm",
            iterations=4,
            learning_rate=0.1,
            min_child_samples=2,
            num_leaves=3,
            random_state=0,
            diagnostic_warnings="never",
            thread_count=1,
        ),
        phases=("sampling_regularization",),
        n_trials=2,
        cv=2,
        random_state=0,
    ).fit(X, y)

    assert search.scorer_.name == "neg_gaussian_nll"
    assert search.tree_modes_ == ("lightgbm",)
    assert search.tuning_metadata_["tree_modes_requested"] == [
        "catboost",
        "lightgbm",
    ]
    assert search.tuning_metadata_["tree_modes_resolved"] == ["lightgbm"]
    assert np.isfinite(search.best_loss_)
    completed = [
        trial for trial in search.study_.trials
        if trial.user_attrs.get("status") == "OK"
    ]
    assert completed
    assert all(
        trial.user_attrs["params_full"]["sampling"] == "uniform"
        for trial in completed
    )
    assert all(
        trial.user_attrs["params_full"]["bootstrap_type"] == "none"
        for trial in completed
    )

    with pytest.raises(ValueError, match="tree_mode='lightgbm'"):
        ChimeraBoostStepwiseSearchCV(
            ChimeraBoostRegressor(
                loss="Gaussian",
                tree_mode="lightgbm",
                iterations=2,
                random_state=0,
                diagnostic_warnings="never",
            ),
            tree_modes=("catboost",),
            n_trials=1,
        ).fit(X, y)


def test_gaussian_serialization_roundtrip(tmp_path):
    X, y = _make_heteroscedastic(seed=5, n=120)
    model = ChimeraBoostRegressor(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=4,
        learning_rate=0.12,
        min_child_samples=3,
        num_leaves=7,
        eval_metric="crps",
        random_state=0,
    ).fit(X, y)

    path = tmp_path / "gaussian.npz"
    model.save_model(path)
    loaded = ChimeraBoostRegressor.load_model(path)
    assert loaded.loss == "Gaussian"
    assert loaded.eval_metric == "crps"
    assert isinstance(loaded.model_, DistributionalBoosting)
    assert loaded.model_.eval_metric_ == "crps"
    np.testing.assert_allclose(
        loaded.predict_dist(X[:20])[0], model.predict_dist(X[:20])[0],
        rtol=0.0, atol=0.0,
    )
    np.testing.assert_allclose(
        loaded.predict_dist(X[:20])[1], model.predict_dist(X[:20])[1],
        rtol=0.0, atol=0.0,
    )

    with np.load(path, allow_pickle=False) as data:
        header = json.loads(str(data["header"]))
    assert header["model_class"] == "DistributionalBoosting"
    assert header["n_outputs"] == 2
    assert "n_classes" not in header
    assert "sigma_scale" not in header["wrapper"]["state"]
    np.testing.assert_allclose(loaded.predict(X[:20]), model.predict(X[:20]))


def test_multiclass_roundtrip_still_uses_n_classes(tmp_path):
    rng = np.random.default_rng(10)
    X = rng.normal(size=(120, 4))
    y = np.digitize(X[:, 0] - 0.4 * X[:, 1], [-0.5, 0.4])
    model = ChimeraBoostClassifier(
        iterations=4,
        tree_mode="lightgbm",
        multiclass_tree_strategy="shared_vector",
        learning_rate=0.15,
        min_child_samples=3,
        num_leaves=7,
        random_state=0,
    ).fit(X, y)
    assert isinstance(model.model_, MulticlassBoosting)

    path = tmp_path / "multi.npz"
    model.save_model(path)
    loaded = ChimeraBoostClassifier.load_model(path)
    np.testing.assert_allclose(
        loaded.predict_proba(X[:20]), model.predict_proba(X[:20])
    )
    with np.load(path, allow_pickle=False) as data:
        header = json.loads(str(data["header"]))
    assert header["model_class"] == "MulticlassBoosting"
    assert header["n_classes"] == 3
    assert "n_outputs" not in header


def test_gaussian_categorical_fit_with_eval_weights():
    rng = np.random.default_rng(11)
    n = 180
    region = rng.choice(["north", "south", "east"], size=n)
    numeric = rng.normal(size=(n, 2))
    y = (
        np.select([region == "north", region == "south"], [1.0, -0.5], 0.2)
        + 0.8 * numeric[:, 0]
        + rng.normal(scale=0.4 + 0.2 * np.abs(numeric[:, 1]), size=n)
    )
    X = np.empty((n, 3), dtype=object)
    X[:, 0] = region
    X[:, 1:] = numeric

    model = ChimeraBoostRegressor(
        **_gaussian_test_params(iterations=5, min_child_samples=3)
    ).fit(
        X[:120], y[:120],
        cat_features=[0],
        eval_set=(X[120:], y[120:]),
        sample_weight=np.linspace(0.5, 1.5, 120),
        eval_sample_weight=np.linspace(1.2, 0.8, 60),
    )
    mu, sigma = model.predict_dist(X[120:130])
    assert mu.shape == sigma.shape == (10,)
    assert np.all(np.isfinite(mu))
    assert np.all(sigma > 0.0)
