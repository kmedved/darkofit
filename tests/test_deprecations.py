from __future__ import annotations

import warnings

import numpy as np
import pytest

from darkofit import DarkoClassifier, DarkoRegressor
from darkofit.booster import GradientBoosting, reset_diagnostic_warning_registry


X = np.arange(48, dtype=np.float64).reshape(24, 2)
Y = np.linspace(-1.0, 1.0, 24)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tree_mode": "depthwise"}, "tree_mode='depthwise'"),
        ({"histogram_dtype": "float32"}, "histogram_dtype"),
        ({"leaf_dtype": "uint32"}, "leaf_dtype"),
        ({"histogram_parallelism": "row"}, "histogram_parallelism"),
        ({"sampling": "weighted_goss"}, "sampling='weighted_goss'"),
        ({"bootstrap_type": "bayesian"}, "bootstrap_type='bayesian'"),
        ({"bagging_temperature": 0.5}, "bagging_temperature"),
    ],
)
def test_core_warns_when_retiring_option_is_selected(kwargs, message):
    booster = GradientBoosting(iterations=0, **kwargs)
    with pytest.warns(FutureWarning, match=message):
        booster.fit(X, Y)


def test_supported_core_options_do_not_warn():
    booster = GradientBoosting(
        iterations=0,
        random_strength=0.5,
        rho_learning_rate_multiplier=0.5,
        rho_l2_leaf_reg_multiplier=2.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        booster.fit(X, Y)


@pytest.mark.parametrize("estimator", [DarkoRegressor, DarkoClassifier])
def test_wrapper_warns_for_auto_learning_rate_probe_family(estimator):
    y = Y if estimator is DarkoRegressor else (Y > 0.0).astype(np.int64)
    model = estimator(
        iterations=1,
        learning_rate=0.1,
        auto_learning_rate_probe_values=(0.05, 0.1),
    )
    with pytest.warns(FutureWarning, match="auto_learning_rate_probe"):
        model.fit(X, y)


def test_wrapper_defaults_do_not_warn():
    model = DarkoRegressor(iterations=1, learning_rate=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model.fit(X, Y)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"linear_residual": True},
        {"linear_residual_alpha": 2.0},
        {"linear_residual_features": [0]},
        {"linear_residual_fit_intercept": False},
        {"linear_residual_standardize": False},
    ],
)
def test_wrapper_warns_for_linear_residual_family(kwargs):
    model = DarkoRegressor(
        iterations=1,
        learning_rate=0.1,
        **kwargs,
    )
    with pytest.warns(FutureWarning, match="linear_residual"):
        model.fit(X, Y)


def test_wrapper_core_warning_points_to_caller():
    model = DarkoRegressor(
        iterations=0,
        tree_mode="depthwise",
        learning_rate=0.1,
    )
    with pytest.warns(FutureWarning) as recorded:
        model.fit(X, Y)
    assert len(recorded) == 1
    assert recorded[0].filename == __file__
    assert not hasattr(model, "_suppress_wrapper_deprecation_warning")


def test_linear_residual_warning_points_to_caller():
    model = DarkoRegressor(
        iterations=1,
        learning_rate=0.1,
        linear_residual=True,
    )
    with pytest.warns(FutureWarning) as recorded:
        model.fit(X, Y)
    assert len(recorded) == 1
    assert recorded[0].filename == __file__
    assert not hasattr(model, "_suppress_wrapper_deprecation_warning")


def test_automatic_selector_emits_core_deprecation_warning_once_at_caller():
    rng = np.random.default_rng(22)
    X_large = rng.normal(size=(1300, 3))
    y_large = 1.5 * X_large[:, 0] + rng.normal(size=1300)
    model = DarkoRegressor(
        iterations=1,
        depth=2,
        learning_rate=0.1,
        histogram_dtype="float32",
        random_state=3,
    )

    with pytest.warns(FutureWarning, match="histogram_dtype") as recorded:
        model.fit(X_large, y_large)

    assert len(recorded) == 1
    assert recorded[0].filename == __file__
    assert not hasattr(model, "_suppress_core_deprecation_warning")


def test_automatic_selector_leaves_runtime_diagnostic_to_final_fit():
    rng = np.random.default_rng(23)
    X_large = rng.normal(size=(1300, 3))
    y_large = X_large[:, 0] + rng.normal(size=1300)
    model = DarkoRegressor(
        iterations=1,
        depth=2,
        learning_rate=None,
        random_state=4,
        diagnostic_warnings="once",
    )

    reset_diagnostic_warning_registry()
    try:
        with pytest.warns(RuntimeWarning, match="learning rate clipped") as recorded:
            model.fit(X_large, y_large)
        emitted = model.model_.auto_params_["diagnostics"][
            "runtime_warnings_emitted"
        ]
        assert emitted == ["learning_rate_clipped_max"]
        assert len(recorded) == 1
    finally:
        reset_diagnostic_warning_registry()
