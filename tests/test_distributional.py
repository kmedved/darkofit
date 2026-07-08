import json
import warnings

import numpy as np
import pytest

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.booster import DistributionalBoosting, MulticlassBoosting
from chimeraboost.losses import (
    GaussianNLL,
    LogNormalNLL,
    NegativeBinomialNLL,
    PoissonNLL,
    StudentTNLL,
    _GAUSS_EVAL_Z_GUARD,
    _GAUSS_RHO_MAX,
    _GAUSS_RHO_MIN,
    _GAUSS_Z_CLIP,
    _T_RHO_MIN,
)
from chimeraboost.sklearn_api import _fit_affine_sigma_calibration
from chimeraboost.tuning.scoring import resolve_scorer
from chimeraboost.tuning.search import _pooled_trial_sigma_calibration
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


class _FakeDistModel:
    def __init__(self, mu, sigma):
        self._mu = np.asarray(mu, dtype=np.float64)
        self._sigma = np.asarray(sigma, dtype=np.float64)

    def predict_dist(self, X):
        n = np.asarray(X).shape[0]
        return self._mu[:n], self._sigma[:n]


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


def test_gaussian_eval_nll_uses_overflow_guard_not_training_clip():
    loss = GaussianNLL()
    y = np.array([100.0], dtype=np.float64)
    F_unit = np.array([[0.0], [0.0]], dtype=np.float64)
    F_tiny_sigma = np.array([[0.0], [-14.0]], dtype=np.float64)

    unit_nll = loss.eval_class_major(y, F_unit)
    tiny_sigma_nll = loss.eval_class_major(y, F_tiny_sigma)

    assert tiny_sigma_nll > unit_nll
    assert tiny_sigma_nll > 0.25 * _GAUSS_EVAL_Z_GUARD ** 2


def test_gaussian_crps_uses_unclipped_closed_form_tail():
    loss = GaussianNLL()
    y = np.array([100.0], dtype=np.float64)
    F = np.array([[0.0], [0.0]], dtype=np.float64)

    crps = loss.crps_class_major(y, F)
    expected = 100.0 - (1.0 / np.sqrt(np.pi))

    assert crps == pytest.approx(expected)


def test_gaussian_target_standardization_is_scale_invariant_and_roundtrips(tmp_path):
    X, y = _make_heteroscedastic(seed=120, n=180)
    params = _gaussian_test_params(
        iterations=10,
        learning_rate=0.08,
        num_leaves=7,
        min_child_samples=3,
        random_state=12,
    )
    unit = ChimeraBoostRegressor(**params).fit(X, y)
    factor = 1e8
    scaled = ChimeraBoostRegressor(**params).fit(X, factor * y)

    assert unit.model_.target_transform_["enabled"] is True
    assert scaled.model_.target_transform_["enabled"] is True
    assert scaled.model_.target_transform_["scale"] == pytest.approx(
        factor * unit.model_.target_transform_["scale"]
    )
    assert (
        scaled.model_.auto_params_["distributional"]["target_transform"]
        ["enabled"]
        is True
    )

    Xu = X[:40]
    raw_unit = unit.model_.predict_raw(Xu)
    raw_scaled = scaled.model_.predict_raw(Xu)
    np.testing.assert_allclose(raw_scaled[:, 0] / factor, raw_unit[:, 0])
    np.testing.assert_allclose(raw_scaled[:, 1] - np.log(factor), raw_unit[:, 1])

    mu_unit, sigma_unit = unit.predict_dist(Xu)
    mu_scaled, sigma_scaled = scaled.predict_dist(Xu)
    np.testing.assert_allclose(mu_scaled / factor, mu_unit)
    np.testing.assert_allclose(sigma_scaled / factor, sigma_unit)
    np.testing.assert_allclose(
        scaled.predict_variance(Xu) / (factor * factor),
        unit.predict_variance(Xu),
    )
    unit_lo, unit_hi = unit.predict_interval(Xu, alpha=0.2)
    scaled_lo, scaled_hi = scaled.predict_interval(Xu, alpha=0.2)
    np.testing.assert_allclose(scaled_lo / factor, unit_lo)
    np.testing.assert_allclose(scaled_hi / factor, unit_hi)
    np.testing.assert_allclose(
        scaled.sample(Xu, n_samples=5, random_state=123) / factor,
        unit.sample(Xu, n_samples=5, random_state=123),
    )

    path = tmp_path / "gaussian-scaled.npz"
    scaled.save_model(path)
    loaded = ChimeraBoostRegressor.load_model(path)
    assert loaded.model_.target_transform_ == scaled.model_.target_transform_
    loaded_mu, loaded_sigma = loaded.predict_dist(Xu)
    np.testing.assert_allclose(loaded_mu, mu_scaled)
    np.testing.assert_allclose(loaded_sigma, sigma_scaled)


def test_regressor_prediction_rejects_post_fit_loss_mismatch():
    X, y = _make_heteroscedastic(seed=121, n=80)
    rmse = ChimeraBoostRegressor(
        iterations=4,
        learning_rate=0.1,
        random_state=0,
        diagnostic_warnings="never",
    ).fit(X, y)
    rmse.set_params(loss="Gaussian")
    with pytest.raises(ValueError, match="fitted loss 'RMSE'.*refit"):
        rmse.predict(X[:3])

    gaussian = ChimeraBoostRegressor(**_gaussian_test_params(iterations=4)).fit(X, y)
    gaussian.set_params(loss="RMSE")
    with pytest.raises(ValueError, match="fitted loss 'Gaussian'.*refit"):
        gaussian.predict_dist(X[:3])


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

    raw_public = model.model_.predict_raw(Xv)
    transform = model.model_.target_transform_
    raw_internal = raw_public.copy()
    raw_internal[:, 0] = (
        raw_internal[:, 0] - transform["mean"]
    ) / transform["scale"]
    raw_internal[:, 1] = raw_internal[:, 1] - np.log(transform["scale"])
    yv_internal = (yv - transform["mean"]) / transform["scale"]
    expected = model.model_.loss_.crps_class_major(yv_internal, raw_internal.T)
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


def test_gaussian_affine_sigma_calibration_recovers_log_sigma_stretch():
    n = 600
    raw_sigma = np.exp(np.linspace(-1.5, 1.2, n))
    true_a = np.log(0.7)
    true_b = 1.23
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    mu = np.linspace(-0.4, 0.4, n)
    y = mu + signs * np.exp(true_a + true_b * np.log(raw_sigma))
    weights = np.ones(n, dtype=np.float64)
    y[0] = 1e100
    weights[0] = 0.0

    calibration = _fit_affine_sigma_calibration(
        _FakeDistModel(mu, raw_sigma),
        np.zeros((n, 1), dtype=np.float64),
        y,
        weights,
        fold_stats={
            "validation_effective_n": float(n - 1),
            "small_fold_warning": False,
        },
    )

    assert calibration["fallback_reason"] is None
    assert calibration["sigma_affine_a"] == pytest.approx(true_a, abs=3e-3)
    assert calibration["sigma_affine_b"] == pytest.approx(true_b, abs=3e-3)


def test_gaussian_affine_sigma_calibration_transforms_public_distribution_only(
    tmp_path,
):
    X, y = _make_heteroscedastic(seed=24, n=520)
    Xtr, Xv = X[:250], X[250:]
    ytr, yv = y[:250], y[250:]

    model = ChimeraBoostRegressor(
        **_gaussian_test_params(
            iterations=12,
            learning_rate=0.08,
            sigma_calibration="affine",
        )
    ).fit(Xtr, ytr, eval_set=(Xv, yv))

    assert model.sigma_calibration_ == "affine"
    assert np.isfinite(model.sigma_affine_a_)
    assert np.isfinite(model.sigma_affine_b_)
    raw_mu, raw_sigma = model.model_.predict_dist(Xv[:20])
    public_mu, public_sigma = model.predict_dist(Xv[:20])
    expected_sigma = np.exp(
        model.sigma_affine_a_ + model.sigma_affine_b_ * np.log(raw_sigma)
    )
    np.testing.assert_allclose(public_mu, raw_mu, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(public_sigma, expected_sigma)
    np.testing.assert_allclose(model.predict(Xv[:20]), raw_mu)

    meta = model.model_.auto_params_["sigma_calibration"]
    assert meta["method"] == "affine"
    assert meta["sigma_affine_a"] == model.sigma_affine_a_
    assert meta["sigma_affine_b"] == model.sigma_affine_b_

    path = tmp_path / "affine_calibrated_gaussian.npz"
    model.save_model(path)
    with np.load(path, allow_pickle=False) as data:
        header = json.loads(str(data["header"]))
    state = header["wrapper"]["state"]
    assert state["sigma_calibration"] == "affine"
    assert state["sigma_affine_a"] == model.sigma_affine_a_
    assert state["sigma_affine_b"] == model.sigma_affine_b_
    loaded = ChimeraBoostRegressor.load_model(path)
    assert loaded.sigma_calibration_ == "affine"
    np.testing.assert_allclose(
        loaded.predict_dist(Xv[:20])[1], public_sigma, rtol=0.0, atol=0.0
    )


def test_gaussian_per_metric_affine_calibration_applies_group_maps_and_roundtrips(
    tmp_path,
):
    rng = np.random.default_rng(43)
    n = 520
    groups = np.tile([0.0, 1.0], n // 2)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([groups, x1, x2])
    sigma = np.where(
        groups == 0.0,
        0.25 + 0.15 * np.abs(x1),
        0.7 + 0.4 * np.abs(x2),
    )
    y = 0.4 * x1 - 0.2 * x2 + rng.normal(scale=sigma)

    model = ChimeraBoostRegressor(
        **_gaussian_test_params(
            iterations=5,
            random_state=7,
            dist_calibration="per_metric_affine",
            dist_calibration_feature=0,
        )
    ).fit(
        X[:360],
        y[:360],
        cat_features=[0],
        eval_set=(X[360:], y[360:]),
    )

    assert model.dist_calibration_ == "per_metric_affine"
    assert model.dist_calibration_feature_index_ == 0
    assert sorted(record["group"] for record in model.dist_group_affine_metadata_) == [
        0.0,
        1.0,
    ]

    raw_mu, raw_sigma = model.model_.predict_dist(X[360:390])
    public_mu, public_sigma = model.predict_dist(X[360:390])
    np.testing.assert_allclose(public_mu, raw_mu)
    expected = np.empty_like(public_sigma)
    for record in model.dist_group_affine_metadata_:
        mask = X[360:390, 0] == record["group"]
        expected[mask] = np.exp(
            record["sigma_affine_a"]
            + record["sigma_affine_b"] * np.log(raw_sigma[mask])
        )
    np.testing.assert_allclose(public_sigma, expected)
    np.testing.assert_allclose(model.predict_variance(X[360:390]), expected * expected)

    path = tmp_path / "per_metric_affine_gaussian.npz"
    model.save_model(path)
    loaded = ChimeraBoostRegressor.load_model(path)
    assert loaded.dist_calibration_ == "per_metric_affine"
    assert loaded.dist_calibration_feature_index_ == 0
    np.testing.assert_allclose(
        loaded.predict_dist(X[360:390])[1],
        public_sigma,
    )


def test_gaussian_per_metric_affine_calibration_resolves_pandas_feature_name():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(44)
    n = 260
    groups = np.tile([0.0, 1.0], n // 2)
    X = pd.DataFrame({
        "metric_code": groups,
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    y = 0.3 * X["x1"].to_numpy() + rng.normal(
        scale=0.4 + 0.2 * groups, size=n
    )

    model = ChimeraBoostRegressor(
        **_gaussian_test_params(
            iterations=4,
            random_state=8,
            dist_calibration="per_metric_affine",
        )
    ).fit(
        X.iloc[:180],
        y[:180],
        cat_features=[0],
        eval_set=(X.iloc[180:], y[180:]),
    )

    assert model.dist_calibration_feature_ == "metric_code"
    assert model.dist_calibration_feature_index_ == 0
    assert model.dist_calibration_feature_name_ == "metric_code"
    public = model.predict_dist(X.iloc[180:190])[1]
    assert np.all(np.isfinite(public))
    assert np.all(public > 0.0)


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


def test_gaussian_get_refit_params_clears_scalar_sigma_calibration():
    X, y = _make_heteroscedastic(seed=19, n=160)
    model = ChimeraBoostRegressor(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=8,
        learning_rate=0.08,
        min_child_samples=3,
        num_leaves=7,
        early_stopping=True,
        early_stopping_rounds=3,
        validation_fraction=0.25,
        sigma_calibration="scalar",
        random_state=0,
        diagnostic_warnings="never",
        thread_count=1,
    ).fit(X, y)

    params = model.get_refit_params()
    assert params["sigma_calibration"] is None
    refit = ChimeraBoostRegressor(**params).fit(X, y)
    assert refit.sigma_calibration_ is None
    mu, sigma = refit.predict_dist(X[:8])
    assert mu.shape == sigma.shape == (8,)
    assert np.all(sigma > 0.0)


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
        (
            {"rho_learning_rate_multiplier": 0.0},
            "rho_learning_rate_multiplier",
        ),
        (
            {"rho_l2_leaf_reg_multiplier": 0.0},
            "rho_l2_leaf_reg_multiplier",
        ),
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


def test_gaussian_rho_learning_rate_multiplier_scales_only_sigma_head():
    X, y = _make_heteroscedastic(seed=21, n=180)
    params = _gaussian_test_params(
        iterations=1,
        learning_rate=0.12,
        min_child_samples=5,
        num_leaves=5,
        random_state=7,
    )
    base = ChimeraBoostRegressor(**params).fit(X, y)
    scaled = ChimeraBoostRegressor(
        **{**params, "rho_learning_rate_multiplier": 0.5}
    ).fit(X, y)

    base_values = base.model_.trees_[0].values
    scaled_values = scaled.model_.trees_[0].values
    assert np.allclose(scaled_values[:, 0], base_values[:, 0])
    assert np.allclose(scaled_values[:, 1], 0.5 * base_values[:, 1])
    assert np.max(np.abs(base_values[:, 1])) > 0.0
    meta = scaled.model_.auto_params_["distributional"]
    assert meta["rho_learning_rate_multiplier"] == 0.5


def test_gaussian_rho_l2_leaf_reg_multiplier_metadata_and_roundtrip(tmp_path):
    X, y = _make_heteroscedastic(seed=27, n=160)
    model = ChimeraBoostRegressor(
        **_gaussian_test_params(
            iterations=3,
            learning_rate=0.1,
            l2_leaf_reg=2.0,
            rho_l2_leaf_reg_multiplier=4.0,
        )
    ).fit(X, y)

    meta = model.model_.auto_params_["distributional"]
    assert meta["rho_l2_leaf_reg_multiplier"] == 4.0
    assert meta["l2_leaf_reg_by_output"] == [2.0, 8.0]

    path = tmp_path / "rho_l2_gaussian.npz"
    model.save_model(path)
    loaded = ChimeraBoostRegressor.load_model(path)
    assert loaded.rho_l2_leaf_reg_multiplier == 4.0
    np.testing.assert_allclose(
        loaded.predict_dist(X[:12])[1], model.predict_dist(X[:12])[1]
    )


def test_distributional_protocol_predict_variance_and_dist_calibration_alias():
    X, y = _make_heteroscedastic(seed=31, n=140)
    params = _gaussian_test_params(iterations=4, random_state=2)
    dist = ChimeraBoostRegressor(
        **params, dist_calibration="scalar"
    ).fit(X[:100], y[:100], eval_set=(X[100:], y[100:]))
    with pytest.warns(DeprecationWarning):
        legacy = ChimeraBoostRegressor(
            **params, sigma_calibration="scalar"
        ).fit(X[:100], y[:100], eval_set=(X[100:], y[100:]))

    np.testing.assert_allclose(dist.predict(X[:15]), legacy.predict(X[:15]))
    for left, right in zip(dist.predict_dist(X[:15]), legacy.predict_dist(X[:15])):
        np.testing.assert_allclose(left, right)
    mu, sigma = dist.predict_dist(X[:15])
    np.testing.assert_allclose(dist.predict_variance(X[:15]), sigma * sigma)
    assert "dist_calibration" in dist.model_.auto_params_
    assert "sigma_calibration" in dist.model_.auto_params_


def test_legacy_sigma_only_calibration_applies_to_public_distribution_methods():
    X, y = _make_heteroscedastic(seed=35, n=140)
    model = ChimeraBoostRegressor(
        **_gaussian_test_params(
            iterations=4,
            random_state=2,
            sigma_calibration="scalar",
        )
    ).fit(X[:100], y[:100], eval_set=(X[100:], y[100:]))

    for name in ("dist_calibration_", "dist_scale_", "dist_scale_source_"):
        if hasattr(model, name):
            delattr(model, name)

    raw_mu, raw_sigma = model.model_.predict_dist(X[:10])
    calibrated_sigma = raw_sigma * model.sigma_scale_
    public_mu, public_sigma = model.predict_dist(X[:10])
    np.testing.assert_allclose(public_mu, raw_mu)
    np.testing.assert_allclose(public_sigma, calibrated_sigma)
    np.testing.assert_allclose(
        model.predict_variance(X[:10]), calibrated_sigma * calibrated_sigma
    )

    lo, hi = model.predict_interval(X[:10], alpha=0.2)
    assert np.all(lo < raw_mu)
    assert np.all(hi > raw_mu)

    samples = model.sample(X[:10], n_samples=3, random_state=123)
    expected = np.random.default_rng(123).normal(
        raw_mu[:, None], calibrated_sigma[:, None], size=(10, 3)
    )
    np.testing.assert_allclose(samples, expected)


def test_lognormal_matches_gaussian_on_log_target_and_roundtrips(tmp_path):
    rng = np.random.default_rng(32)
    X = rng.normal(size=(160, 4))
    log_y = 0.6 * X[:, 0] - 0.3 * X[:, 2] + rng.normal(scale=0.25, size=160)
    y = np.exp(log_y)
    common = dict(
        tree_mode="lightgbm",
        iterations=6,
        learning_rate=0.08,
        num_leaves=7,
        min_child_samples=3,
        random_state=3,
        diagnostic_warnings="never",
        thread_count=1,
    )
    gaussian = ChimeraBoostRegressor(loss="Gaussian", **common).fit(X, log_y)
    lognormal = ChimeraBoostRegressor(loss="LogNormal", **common).fit(X, y)

    np.testing.assert_allclose(
        lognormal.model_.predict_raw(X[:30]),
        gaussian.model_.predict_raw(X[:30]),
        rtol=0.0,
        atol=1e-12,
    )
    m, s = lognormal.predict_dist(X[:20])
    np.testing.assert_allclose(lognormal.predict(X[:20]), np.exp(m + 0.5 * s * s))
    np.testing.assert_allclose(
        lognormal.predict_variance(X[:20]),
        (np.exp(s * s) - 1.0) * np.exp(2.0 * m + s * s),
    )
    extreme_raw = np.array([
        [1000.0, 15.0],
        [-1000.0, 15.0],
        [0.0, -15.0],
    ])
    extreme_variance = LogNormalNLL().variance_from_raw(extreme_raw)
    assert np.all(np.isfinite(extreme_variance))
    assert np.all(extreme_variance >= 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        zero_variance = LogNormalNLL().variance_from_params(
            np.array([0.0]), np.array([0.0])
        )
    np.testing.assert_allclose(zero_variance, np.array([0.0]))
    with pytest.raises(ValueError, match="strictly positive"):
        ChimeraBoostRegressor(loss="LogNormal", **common).fit(X, np.zeros_like(y))

    path = tmp_path / "lognormal.npz"
    lognormal.save_model(path)
    loaded = ChimeraBoostRegressor.load_model(path)
    assert loaded.loss == "LogNormal"
    np.testing.assert_allclose(loaded.predict_dist(X[:10])[0], m[:10])


def test_lognormal_preprocessor_uses_log_target_for_categorical_encodings(
    monkeypatch,
):
    import chimeraboost.booster as booster_mod

    seen_targets = []
    original = booster_mod.FeaturePreprocessor.fit_transform

    def wrapped_fit_transform(self, X, encode_targets, cat_features,
                              sample_weight=None):
        seen_targets.append(np.asarray(encode_targets[0], dtype=np.float64).copy())
        return original(
            self, X, encode_targets, cat_features, sample_weight=sample_weight
        )

    monkeypatch.setattr(
        booster_mod.FeaturePreprocessor, "fit_transform", wrapped_fit_transform
    )
    rng = np.random.default_rng(37)
    n = 80
    cats = rng.choice(np.array(["small", "large"], dtype=object), size=n)
    numeric = rng.normal(size=n)
    y = np.exp(
        np.where(cats == "large", 2.0, 0.1)
        + 0.2 * numeric
        + rng.normal(scale=0.15, size=n)
    )
    X = np.empty((n, 2), dtype=object)
    X[:, 0] = cats
    X[:, 1] = numeric

    ChimeraBoostRegressor(
        loss="LogNormal",
        tree_mode="lightgbm",
        iterations=2,
        learning_rate=0.08,
        num_leaves=5,
        min_child_samples=3,
        target_ordered_cat_codes="leaky_full",
        random_state=0,
        diagnostic_warnings="never",
        thread_count=1,
    ).fit(X, y, cat_features=[0])

    assert seen_targets
    log_y = np.log(y)
    expected = (log_y - np.mean(log_y)) / np.std(log_y)
    np.testing.assert_allclose(seen_targets[0], expected)


def test_student_t_gradient_bounds_and_distribution_api(tmp_path):
    loss = StudentTNLL(nu=4.0)
    z = np.array([-1e6, -10.0, -2.0, 0.0, 2.0, 10.0, 1e6])
    F = np.vstack([np.zeros_like(z), np.zeros_like(z)])
    y = z.copy()
    grad = np.empty_like(F)
    hess = np.empty_like(F)
    loss.grad_hess_class_major_into(y, F, None, grad, hess)
    bound = (loss.nu + 1.0) / (2.0 * np.sqrt(loss.nu))
    assert np.max(np.abs(grad[0])) <= bound + 1e-12
    assert np.min(grad[1]) >= -loss.nu - 1e-12
    assert np.max(grad[1]) <= 1.0 + 1e-12
    assert np.all(hess > 0.0)

    y_extreme = np.array([1e308, -1e308])
    F_extreme = np.vstack([
        np.zeros_like(y_extreme),
        np.full_like(y_extreme, _T_RHO_MIN),
    ])
    grad_extreme = np.empty_like(F_extreme)
    hess_extreme = np.empty_like(F_extreme)
    loss.grad_hess_class_major_into(
        y_extreme, F_extreme, None, grad_extreme, hess_extreme
    )
    assert np.all(np.isfinite(grad_extreme))
    assert np.all(np.isfinite(hess_extreme))
    assert np.max(np.abs(grad_extreme[0])) <= bound + 1e-12
    assert np.min(grad_extreme[1]) >= -loss.nu - 1e-12
    assert np.max(grad_extreme[1]) <= 1.0 + 1e-12

    with pytest.raises(ValueError, match="nu > 2"):
        StudentTNLL(nu=2.0)

    X, y_fit = _make_heteroscedastic(seed=33, n=150)
    model = ChimeraBoostRegressor(
        loss="StudentT",
        dist_params={"nu": 4.0},
        tree_mode="lightgbm",
        iterations=5,
        learning_rate=0.08,
        num_leaves=7,
        min_child_samples=3,
        random_state=4,
        diagnostic_warnings="never",
        thread_count=1,
    ).fit(X, y_fit)
    mu, scale, nu = model.predict_dist(X[:12])
    np.testing.assert_allclose(nu, np.full(12, 4.0))
    np.testing.assert_allclose(
        model.predict_variance(X[:12]), scale * scale * 4.0 / 2.0
    )
    lo, hi = model.predict_interval(X[:12], alpha=0.1)
    assert np.all(lo < mu)
    assert np.all(hi > mu)
    path = tmp_path / "student_t.npz"
    model.save_model(path)
    loaded = ChimeraBoostRegressor.load_model(path)
    assert loaded.dist_params == {"nu": 4.0}
    np.testing.assert_allclose(loaded.predict_dist(X[:12])[1], scale)


def test_poisson_and_negative_binomial_count_heads_roundtrip_and_calibrate(tmp_path):
    rng = np.random.default_rng(34)
    X = rng.normal(size=(180, 4))
    lam = np.exp(np.clip(0.4 * X[:, 0] - 0.2 * X[:, 1], -2.0, 2.0))
    y_pois = rng.poisson(lam)
    common = dict(
        tree_mode="lightgbm",
        iterations=6,
        learning_rate=0.08,
        num_leaves=7,
        min_child_samples=3,
        random_state=5,
        diagnostic_warnings="never",
        thread_count=1,
    )
    poisson = ChimeraBoostRegressor(
        loss="Poisson", dist_calibration="scalar", **common
    ).fit(X[:130], y_pois[:130], eval_set=(X[130:], y_pois[130:]))
    (lam_hat,) = poisson.predict_dist(X[:15])
    assert np.all(lam_hat > 0.0)
    np.testing.assert_allclose(poisson.predict_variance(X[:15]), lam_hat)
    poisson.dist_scale_ = 1.75
    poisson.sigma_scale_ = 1.75
    raw_lam = poisson.model_.predict_dist(X[:15])[0]
    calibrated_lam = poisson.predict_dist(X[:15])[0]
    np.testing.assert_allclose(calibrated_lam, raw_lam * 1.75)
    np.testing.assert_allclose(poisson.predict(X[:15]), calibrated_lam)
    np.testing.assert_allclose(
        list(poisson.staged_predict(X[:15]))[-1], calibrated_lam
    )
    generic_score, _ = resolve_scorer(
        poisson, "neg_distributional_nll"
    )(poisson, X[:40], y_pois[:40])
    poisson_score, _ = resolve_scorer(
        poisson, "neg_poisson_nll"
    )(poisson, X[:40], y_pois[:40])
    assert generic_score == pytest.approx(poisson_score)
    meta = poisson.model_.auto_params_["dist_calibration"]
    assert meta["mean_calibration_numerator"] >= 0.0
    assert meta["mean_calibration_denominator"] > 0.0
    assert meta["mean_calibration_objective"] == "poisson_closed_form"
    with pytest.raises(ValueError, match="integer counts"):
        ChimeraBoostRegressor(loss="Poisson", **common).fit(X, y_pois + 0.25)

    r_true = 5.0
    y_nb = rng.negative_binomial(r_true, r_true / (r_true + lam))
    nb = ChimeraBoostRegressor(
        loss="NegativeBinomial",
        tree_mode="lightgbm",
        iterations=8,
        learning_rate=0.08,
        num_leaves=7,
        min_child_samples=3,
        random_state=6,
        diagnostic_warnings="never",
        thread_count=1,
    ).fit(X, y_nb)
    mu, alpha = nb.predict_dist(X[:15])
    assert np.all(mu > 0.0)
    assert np.all(alpha > 0.0)
    np.testing.assert_allclose(nb.predict_variance(X[:15]), mu + alpha * mu * mu)
    assert nb.model_.loss_.state_["r"] > 0.0
    assert nb.model_.loss_.state_["r_path"]
    final_train_nll = nb.model_._eval_metric_class_major(
        y_nb, nb.model_.predict_raw(X).T, None
    )
    assert nb.model_.best_score_ == pytest.approx(final_train_nll)
    assert nb.model_.train_history_[-1] == pytest.approx(final_train_nll)

    path = tmp_path / "nb.npz"
    nb.save_model(path)
    loaded = ChimeraBoostRegressor.load_model(path)
    assert loaded.loss == "NegativeBinomial"
    assert loaded.model_.loss_.state_["r"] == pytest.approx(nb.model_.loss_.state_["r"])
    np.testing.assert_allclose(loaded.predict_dist(X[:15])[0], mu)
    nb.dist_calibration_ = "scalar"
    nb.sigma_calibration_ = "scalar"
    nb.dist_scale_ = 1.4
    nb.sigma_scale_ = 1.4
    raw_mu = nb.model_.predict_dist(X[:15])[0]
    calibrated_mu = nb.predict_dist(X[:15])[0]
    np.testing.assert_allclose(calibrated_mu, raw_mu * 1.4)
    np.testing.assert_allclose(nb.predict(X[:15]), calibrated_mu)

    nb_cal = ChimeraBoostRegressor(
        loss="NegativeBinomial", dist_calibration="scalar", **common
    ).fit(X[:130], y_nb[:130], eval_set=(X[130:], y_nb[130:]))
    nb_meta = nb_cal.model_.auto_params_["dist_calibration"]
    assert nb_meta["mean_calibration_objective"] == "negative_binomial_nll"
    assert "mean_calibration_numerator" not in nb_meta
    base_scale = nb_cal.dist_scale_
    nb_score, _ = resolve_scorer(nb_cal, "neg_negative_binomial_nll")(
        nb_cal, X[130:], y_nb[130:]
    )
    base_nll = -nb_score
    for factor in (np.exp(-0.25), np.exp(0.25)):
        nb_cal.dist_scale_ = base_scale * factor
        trial_score, _ = resolve_scorer(nb_cal, "neg_negative_binomial_nll")(
            nb_cal, X[130:], y_nb[130:]
        )
        assert -trial_score >= base_nll - 1e-6
    nb_cal.dist_scale_ = base_scale


def test_negative_binomial_refresh_rescores_validation_history():
    rng = np.random.default_rng(36)
    X = rng.normal(size=(180, 4))
    mu = np.exp(np.clip(0.7 * X[:, 0] - 0.3 * X[:, 1], -2.0, 2.0))
    r = 3.0
    y = rng.negative_binomial(r, r / (r + mu))
    X_train, X_val = X[:130], X[130:]
    y_train, y_val = y[:130], y[130:]

    model = ChimeraBoostRegressor(
        loss="NegativeBinomial",
        tree_mode="lightgbm",
        iterations=30,
        learning_rate=0.08,
        num_leaves=7,
        min_child_samples=3,
        early_stopping=False,
        use_best_model=True,
        random_state=7,
        diagnostic_warnings="never",
        thread_count=1,
    ).fit(X_train, y_train, eval_set=(X_val, y_val))

    booster = model.model_
    assert any(
        item["source"] == "refresh" for item in booster.loss_.state_["r_path"]
    )
    rescored = [
        booster._eval_metric_class_major(y_val, raw.T, None)
        for raw in booster.staged_predict_raw(X_val)
    ]
    np.testing.assert_allclose(
        booster.valid_history_, rescored, rtol=1e-12, atol=1e-12
    )
    assert booster.valid_history_[-1] == pytest.approx(min(rescored))
    assert booster.best_score_ == pytest.approx(booster.valid_history_[-1])


def test_searchcv_mean_calibration_pooling_keeps_positive_floor():
    class Trial:
        user_attrs = {
            "fold_weight_sums": [3.0, 4.0],
            "fold_auto_params": [
                {
                    "dist_calibration": {
                        "method": "scalar",
                        "dist_scale": 1e-12,
                        "mean_calibration_numerator": 0.0,
                        "mean_calibration_denominator": 5.0,
                    }
                },
                {
                    "dist_calibration": {
                        "method": "scalar",
                        "dist_scale": 1e-12,
                        "mean_calibration_numerator": 0.0,
                        "mean_calibration_denominator": 7.0,
                    }
                },
            ],
        }

    pooled = _pooled_trial_sigma_calibration(Trial())
    assert pooled["pooling"] == "exact_mean_sufficient_statistics"
    assert pooled["dist_scale"] == pytest.approx(1e-12)
    assert pooled["mean_calibration_numerator"] == 0.0
    assert pooled["mean_calibration_denominator"] == 12.0


def test_searchcv_dispersion_calibration_pooling_uses_positive_log_scale():
    class Trial:
        user_attrs = {
            "fold_weight_sums": [2.0, 8.0],
            "fold_auto_params": [
                {
                    "dist_calibration": {
                        "method": "dispersion",
                        "dist_scale": 0.5,
                        "validation_n_samples": 20,
                    }
                },
                {
                    "dist_calibration": {
                        "method": "dispersion",
                        "dist_scale": 2.0,
                        "validation_n_samples": 80,
                    }
                },
            ],
        }

    pooled = _pooled_trial_sigma_calibration(Trial())
    expected = np.exp(np.average(np.log([0.5, 2.0]), weights=[2.0, 8.0]))
    assert pooled["method"] == "dispersion"
    assert pooled["pooling"] == "validation_mass_weighted_log_dispersion_scale"
    assert pooled["dist_scale"] == pytest.approx(expected)
    assert pooled["sigma_scale"] == pytest.approx(expected)
    assert pooled["fold_stats"]["validation_n_samples"] == 100


def test_searchcv_pooled_calibration_warns_when_any_fold_is_small():
    class Trial:
        user_attrs = {
            "fold_weight_sums": [1.0, 1.0],
            "fold_auto_params": [
                {
                    "sigma_calibration": {
                        "method": "scalar",
                        "sigma_scale": 0.9,
                        "validation_effective_n": 20.0,
                        "small_fold_threshold": 200.0,
                        "small_fold_warning": True,
                    }
                },
                {
                    "sigma_calibration": {
                        "method": "scalar",
                        "sigma_scale": 1.1,
                        "validation_effective_n": 500.0,
                        "small_fold_threshold": 200.0,
                        "small_fold_warning": False,
                    }
                },
            ],
        }

    pooled = _pooled_trial_sigma_calibration(Trial())
    assert pooled["fold_stats"]["validation_effective_n"] == pytest.approx(520.0)
    assert pooled["fold_stats"]["small_fold_warning"] is True


def test_searchcv_negative_binomial_mean_calibration_pools_log_scale():
    class Trial:
        user_attrs = {
            "fold_weight_sums": [1.0, 3.0],
            "fold_auto_params": [
                {
                    "dist_calibration": {
                        "method": "scalar",
                        "dist_scale": 0.75,
                        "mean_calibration_objective": "negative_binomial_nll",
                    }
                },
                {
                    "dist_calibration": {
                        "method": "scalar",
                        "dist_scale": 1.5,
                        "mean_calibration_objective": "negative_binomial_nll",
                    }
                },
            ],
        }

    pooled = _pooled_trial_sigma_calibration(Trial())
    expected = np.exp(np.average(np.log([0.75, 1.5]), weights=[1.0, 3.0]))
    assert pooled["method"] == "scalar"
    assert pooled["pooling"] == "validation_mass_weighted_log_mean_scale"
    assert pooled["mean_calibration_objective"] == "negative_binomial_nll"
    assert pooled["dist_scale"] == pytest.approx(expected)


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


def test_gaussian_searchcv_refit_freezes_scalar_sigma_calibration():
    X, y = _make_heteroscedastic(seed=18, n=120)
    search = ChimeraBoostStepwiseSearchCV(
        ChimeraBoostRegressor(
            loss="Gaussian",
            tree_mode="lightgbm",
            iterations=4,
            learning_rate=0.1,
            min_child_samples=2,
            num_leaves=3,
            sigma_calibration="scalar",
            random_state=0,
            diagnostic_warnings="never",
            thread_count=1,
        ),
        phases=("probe",),
        tree_modes=("lightgbm",),
        n_trials={"probe": 1},
        cv=2,
        refit=True,
        random_state=0,
        resume=False,
        trial_thread_count=1,
    ).fit(X, y)

    final = search.best_estimator_
    assert search.refit_params_["sigma_calibration"] is None
    expected_rounds = max(
        1, int(np.ceil(np.median(search.best_trial_.user_attrs["fold_best_iterations"])))
    )
    assert search.refit_params_["iterations"] == expected_rounds
    assert (
        search.refit_iterations_source_
        == "median_fold_best_for_calibrated_distribution"
    )
    assert search.tuning_metadata_["refit_iterations"] == expected_rounds
    assert final.sigma_calibration_ == "scalar"
    assert final.sigma_scale_source_ == "search_cv_validation"
    fold_metas = search.best_trial_.user_attrs["fold_auto_params"]
    fold_masses = search.best_trial_.user_attrs["fold_weight_sums"]
    expected_scale = np.sqrt(
        np.average(
            [
                meta["sigma_calibration"]["sigma_scale"] ** 2
                for meta in fold_metas
            ],
            weights=fold_masses,
        )
    )
    assert final.sigma_scale_ == pytest.approx(expected_scale)
    _, raw_sigma = final.model_.predict_dist(X[:8])
    _, public_sigma = final.predict_dist(X[:8])
    np.testing.assert_allclose(
        public_sigma, raw_sigma * final.sigma_scale_, rtol=0.0, atol=0.0
    )


def test_gaussian_searchcv_refit_freezes_affine_sigma_calibration():
    X, y = _make_heteroscedastic(seed=25, n=140)
    search = ChimeraBoostStepwiseSearchCV(
        ChimeraBoostRegressor(
            loss="Gaussian",
            tree_mode="lightgbm",
            iterations=4,
            learning_rate=0.1,
            min_child_samples=2,
            num_leaves=3,
            sigma_calibration="affine",
            random_state=0,
            diagnostic_warnings="never",
            thread_count=1,
        ),
        phases=("probe",),
        tree_modes=("lightgbm",),
        n_trials={"probe": 1},
        cv=2,
        refit=True,
        random_state=0,
        resume=False,
        trial_thread_count=1,
    ).fit(X, y)

    final = search.best_estimator_
    assert search.refit_params_["sigma_calibration"] is None
    assert final.sigma_calibration_ == "affine"
    assert np.isfinite(final.sigma_affine_a_)
    assert np.isfinite(final.sigma_affine_b_)
    fold_metas = search.best_trial_.user_attrs["fold_auto_params"]
    fold_masses = search.best_trial_.user_attrs["fold_weight_sums"]
    expected_a = np.average(
        [
            meta["sigma_calibration"]["sigma_affine_a"]
            for meta in fold_metas
        ],
        weights=fold_masses,
    )
    expected_b = np.average(
        [
            meta["sigma_calibration"]["sigma_affine_b"]
            for meta in fold_metas
        ],
        weights=fold_masses,
    )
    assert final.sigma_affine_a_ == pytest.approx(expected_a)
    assert final.sigma_affine_b_ == pytest.approx(expected_b)
    _, raw_sigma = final.model_.predict_dist(X[:8])
    _, public_sigma = final.predict_dist(X[:8])
    np.testing.assert_allclose(
        public_sigma,
        np.exp(final.sigma_affine_a_ + final.sigma_affine_b_ * np.log(raw_sigma)),
    )


def test_gaussian_searchcv_refit_freezes_per_metric_affine_sigma_calibration():
    rng = np.random.default_rng(45)
    n = 160
    groups = np.tile([0.0, 1.0], n // 2)
    X = np.column_stack([groups, rng.normal(size=(n, 3))])
    y = 0.5 * X[:, 1] + rng.normal(scale=0.35 + 0.25 * groups, size=n)
    search = ChimeraBoostStepwiseSearchCV(
        ChimeraBoostRegressor(
            loss="Gaussian",
            tree_mode="lightgbm",
            iterations=4,
            learning_rate=0.1,
            min_child_samples=2,
            num_leaves=3,
            dist_calibration="per_metric_affine",
            dist_calibration_feature=0,
            random_state=0,
            diagnostic_warnings="never",
            thread_count=1,
        ),
        phases=("probe",),
        tree_modes=("lightgbm",),
        n_trials={"probe": 1},
        cv=2,
        refit=True,
        random_state=0,
        resume=False,
        trial_thread_count=1,
    ).fit(X, y, cat_features=[0])

    final = search.best_estimator_
    assert search.refit_params_["dist_calibration"] is None
    assert final.dist_calibration_ == "per_metric_affine"
    assert final.dist_calibration_feature_index_ == 0
    assert len(final.dist_group_affine_metadata_) == 2
    _, raw_sigma = final.model_.predict_dist(X[:8])
    _, public_sigma = final.predict_dist(X[:8])
    expected = np.empty_like(public_sigma)
    for record in final.dist_group_affine_metadata_:
        mask = X[:8, 0] == record["group"]
        expected[mask] = np.exp(
            record["sigma_affine_a"]
            + record["sigma_affine_b"] * np.log(raw_sigma[mask])
        )
    np.testing.assert_allclose(public_sigma, expected)


def test_negative_binomial_searchcv_refit_freezes_dispersion_calibration():
    rng = np.random.default_rng(35)
    X = rng.normal(size=(140, 4))
    mu = np.exp(np.clip(0.5 * X[:, 0] - 0.25 * X[:, 1], -2.0, 2.0))
    r = 4.0
    y = rng.negative_binomial(r, r / (r + mu))

    search = ChimeraBoostStepwiseSearchCV(
        ChimeraBoostRegressor(
            loss="NegativeBinomial",
            tree_mode="lightgbm",
            iterations=5,
            learning_rate=0.08,
            min_child_samples=3,
            num_leaves=7,
            dist_calibration="dispersion",
            random_state=0,
            diagnostic_warnings="never",
            thread_count=1,
        ),
        phases=("probe",),
        tree_modes=("lightgbm",),
        n_trials={"probe": 1},
        cv=2,
        refit=True,
        random_state=0,
        resume=False,
        trial_thread_count=1,
    ).fit(X, y)

    final = search.best_estimator_
    assert search.refit_params_["dist_calibration"] is None
    assert final.dist_calibration_ == "dispersion"
    assert final.dist_scale_source_ == "search_cv_validation"
    assert final.dist_calibration_pooling_ == (
        "validation_mass_weighted_log_dispersion_scale"
    )
    fold_metas = search.best_trial_.user_attrs["fold_auto_params"]
    fold_masses = search.best_trial_.user_attrs["fold_weight_sums"]
    expected_scale = np.exp(
        np.average(
            [
                np.log(meta["dist_calibration"]["dist_scale"])
                for meta in fold_metas
            ],
            weights=fold_masses,
        )
    )
    assert final.dist_scale_ == pytest.approx(expected_scale)
    raw_mu, raw_alpha = final.model_.predict_dist(X[:8])
    public_mu, public_alpha = final.predict_dist(X[:8])
    np.testing.assert_allclose(public_mu, raw_mu, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        public_alpha, raw_alpha * final.dist_scale_, rtol=0.0, atol=0.0
    )


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


def test_distributional_load_rejects_loss_output_width_mismatch(tmp_path):
    X, y = _make_heteroscedastic(seed=26, n=100)
    model = ChimeraBoostRegressor(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=3,
        learning_rate=0.12,
        min_child_samples=3,
        num_leaves=5,
        random_state=0,
    ).fit(X, y)

    path = tmp_path / "gaussian_valid.npz"
    bad_path = tmp_path / "gaussian_bad_width.npz"
    model.save_model(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    header = json.loads(str(arrays["header"]))
    header["n_outputs"] = 3
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(bad_path, **arrays)

    with pytest.raises(ValueError, match="n_outputs"):
        ChimeraBoostRegressor.load_model(bad_path)


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
