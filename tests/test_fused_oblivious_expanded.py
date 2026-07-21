import numpy as np
import pytest

import darkofit.booster as booster_module
from benchmarks import fused_lane_dispatch_campaign as dispatch_campaign
from darkofit import DarkoRegressor
from darkofit.booster import GradientBoosting


def _regression_data(seed=71, n=180):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = (
        1.4 * X[:, 0]
        - 0.8 * X[:, 1]
        + 0.3 * X[:, 2] * X[:, 3]
        + rng.normal(scale=0.15, size=n)
    )
    return X, y


def _categorical_data():
    X, y = _regression_data(seed=73)
    categories = np.asarray(["guard", "wing", "big"], dtype=object)
    categorical = categories[np.arange(len(X)) % len(categories)]
    mixed = np.empty((len(X), 3), dtype=object)
    mixed[:, 0] = X[:, 0]
    mixed[:, 1] = categorical
    mixed[:, 2] = X[:, 1]
    y = y + (categorical == "guard") * 0.4 - (categorical == "big") * 0.2
    return mixed, y


def _core_params(**overrides):
    params = {
        "iterations": 14,
        "learning_rate": 0.1,
        "depth": 3,
        "l2_leaf_reg": 3.0,
        "max_bins": 32,
        "min_child_weight": 0.0,
        "min_child_samples": 2,
        "thread_count": 4,
        "random_state": 17,
        "tree_mode": "catboost",
        "diagnostic_warnings": "never",
    }
    params.update(overrides)
    return params


def _fit_core(monkeypatch, X, y, *, fused, params=None, fit_kwargs=None):
    params = dict(params or _core_params())
    params["oblivious_kernel"] = "fused" if fused else "unfused"
    with monkeypatch.context() as context:
        # This file also exercises the proven sampled row/feature builder
        # expansion, which remains outside the narrower public dispatch scope.
        context.setattr(
            booster_module._BaseBooster,
            "_oblivious_functional_ineligibility",
            lambda *_args, **_kwargs: None,
        )
        model = GradientBoosting(**params).fit(
            X, y, **(fit_kwargs or {})
        )
    return model, int(
        model.oblivious_kernel_dispatch_["fused_level_count"]
    )


def _fit_wrapper(monkeypatch, X, y, *, fused, params):
    params = dict(params)
    params["oblivious_kernel"] = "fused" if fused else "unfused"
    with monkeypatch.context() as context:
        context.setattr(
            booster_module._BaseBooster,
            "_oblivious_functional_ineligibility",
            lambda *_args, **_kwargs: None,
        )
        model = DarkoRegressor(**params).fit(X, y)
    return model, int(
        model.model_.oblivious_kernel_dispatch_["fused_level_count"]
    )


def _assert_exact_models(
    reference, candidate, X, tmp_path, name, *, raw=True, persist=True
):
    predict = "predict_raw" if raw else "predict"
    np.testing.assert_array_equal(
        getattr(candidate, predict)(X), getattr(reference, predict)(X)
    )
    np.testing.assert_array_equal(
        candidate.feature_importances_, reference.feature_importances_
    )
    if not persist:
        return
    reference_path = tmp_path / f"{name}-reference.npz"
    candidate_path = tmp_path / f"{name}-candidate.npz"
    reference.save_model(reference_path)
    candidate.save_model(candidate_path)
    assert dispatch_campaign.canonical_archive_sha256(
        candidate_path, project_dispatch=True
    ) == dispatch_campaign.canonical_archive_sha256(
        reference_path, project_dispatch=True
    )


@pytest.mark.parametrize(
    ("loss", "loss_kwargs"),
    [("RMSE", {}), ("MAE", {}), ("Quantile", {"alpha": 0.8})],
)
def test_fused_scalar_losses_are_archive_exact(
    monkeypatch, tmp_path, loss, loss_kwargs
):
    X, y = _regression_data()
    params = _core_params(loss=loss, loss_kwargs=loss_kwargs)
    reference, reference_count = _fit_core(
        monkeypatch, X, y, fused=False, params=params
    )
    candidate, candidate_count = _fit_core(
        monkeypatch, X, y, fused=True, params=params
    )

    assert reference_count == 0
    assert candidate_count > 0
    _assert_exact_models(reference, candidate, X, tmp_path, loss.lower())


def test_fused_categorical_rmse_is_archive_exact(monkeypatch, tmp_path):
    X, y = _categorical_data()
    fit_kwargs = {"cat_features": [1]}
    reference, reference_count = _fit_core(
        monkeypatch, X, y, fused=False, fit_kwargs=fit_kwargs
    )
    candidate, candidate_count = _fit_core(
        monkeypatch, X, y, fused=True, fit_kwargs=fit_kwargs
    )

    assert reference_count == 0
    assert candidate_count > 0
    _assert_exact_models(reference, candidate, X, tmp_path, "categorical")


@pytest.mark.parametrize("lane", ["weighted_rmse", "binary_logloss"])
def test_nonconstant_hessian_lanes_are_fused_and_archive_exact(
    monkeypatch, tmp_path, lane
):
    X, y = _regression_data(seed=79)
    params = _core_params()
    fit_kwargs = {}
    if lane == "weighted_rmse":
        fit_kwargs["sample_weight"] = np.linspace(0.5, 1.5, len(y))
    else:
        y = (y > np.median(y)).astype(np.float64)
        params["loss"] = "Logloss"

    reference, reference_count = _fit_core(
        monkeypatch, X, y, fused=False, params=params, fit_kwargs=fit_kwargs
    )
    candidate, candidate_count = _fit_core(
        monkeypatch, X, y, fused=True, params=params, fit_kwargs=fit_kwargs
    )

    assert reference_count == 0
    assert candidate_count > 0
    _assert_exact_models(reference, candidate, X, tmp_path, lane)


def test_fused_callback_stop_is_archive_exact(monkeypatch, tmp_path):
    X, y = _regression_data(seed=83)

    class StopAfterThree:
        stop_reason = "expanded_test_limit"

        def __call__(self, progress):
            return progress.rounds_completed >= 3

    fit_kwargs = {"eval_set": (X, y), "callbacks": StopAfterThree()}
    reference, reference_count = _fit_core(
        monkeypatch, X, y, fused=False, fit_kwargs=fit_kwargs
    )
    fit_kwargs = {"eval_set": (X, y), "callbacks": StopAfterThree()}
    candidate, candidate_count = _fit_core(
        monkeypatch, X, y, fused=True, fit_kwargs=fit_kwargs
    )

    assert reference_count == 0
    assert candidate_count > 0
    assert reference.stop_reason_ == candidate.stop_reason_ == "expanded_test_limit"
    assert reference.best_iteration_ == candidate.best_iteration_ == 3
    _assert_exact_models(reference, candidate, X, tmp_path, "callback")


def test_fused_early_stopping_exact_refit_is_archive_exact(monkeypatch, tmp_path):
    X, y = _regression_data(seed=89, n=240)
    params = {
        "iterations": 40,
        "learning_rate": 0.1,
        "depth": 3,
        "l2_leaf_reg": 3.0,
        "max_bins": 32,
        "min_child_samples": 2,
        "early_stopping": True,
        "early_stopping_rounds": 4,
        "validation_fraction": 0.2,
        "refit": True,
        "tree_mode": "catboost",
        "thread_count": 4,
        "random_state": 23,
        "diagnostic_warnings": "never",
    }
    reference, reference_count = _fit_wrapper(
        monkeypatch, X, y, fused=False, params=params
    )
    candidate, candidate_count = _fit_wrapper(
        monkeypatch, X, y, fused=True, params=params
    )

    assert reference_count == 0
    assert candidate_count > 0
    assert reference.refit_ is candidate.refit_ is True
    assert reference.n_estimators_ == candidate.n_estimators_
    assert reference.best_n_estimators_ == candidate.best_n_estimators_
    _assert_exact_models(
        reference, candidate, X, tmp_path, "early-refit", raw=False
    )


def test_default_internal_dispatch_engages_proven_fused_lane():
    X, y = _regression_data(seed=97)

    model = GradientBoosting(**_core_params()).fit(X, y)

    assert model.oblivious_kernel_dispatch_["fused_level_count"] > 0
    assert model.oblivious_kernel_dispatch_["unfused_level_count"] == 0


@pytest.mark.parametrize(
    "sampling_params",
    [
        {"colsample": 2 / 3},
        {"subsample": 0.8},
        {"colsample": 2 / 3, "subsample": 0.8},
    ],
)
def test_subset_lanes_use_exact_fused_builder(
    monkeypatch, tmp_path, sampling_params
):
    X, y = _regression_data(seed=101, n=320)
    params = _core_params(**sampling_params)
    reference, reference_count = _fit_core(
        monkeypatch, X, y, fused=False, params=params
    )
    candidate, candidate_count = _fit_core(
        monkeypatch, X, y, fused=True, params=params
    )

    assert reference_count == 0
    assert candidate_count > 0
    name = "-".join(sorted(sampling_params))
    _assert_exact_models(
        reference, candidate, X, tmp_path, name, persist=False
    )
