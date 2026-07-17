import numpy as np
import pytest

import darkofit.tree as tree_module
from darkofit import DarkoRegressor
from darkofit.booster import GradientBoosting
from darkofit.tree import (
    _SMALL_LEAF_DESCENT_ROWS,
    _update_leaves_with_split,
    _update_leaves_with_split_parallel,
    _update_leaves_with_split_serial,
)


@pytest.mark.parametrize("rows", [0, 1, 257, 4_248])
@pytest.mark.parametrize("leaf_dtype", [np.int64, np.uint32])
@pytest.mark.parametrize("order", ["C", "F"])
def test_serial_leaf_descent_is_parallel_exact(rows, leaf_dtype, order):
    rng = np.random.default_rng(101 + rows)
    X = rng.integers(0, 64, size=(rows, 5), dtype=np.uint8)
    X = np.array(X, order=order, copy=True)
    initial = rng.integers(0, 16, size=rows, dtype=np.int64).astype(leaf_dtype)
    serial = initial.copy()
    parallel = initial.copy()

    _update_leaves_with_split_serial(X, serial, 2, 31)
    _update_leaves_with_split_parallel(X, parallel, 2, 31)

    np.testing.assert_array_equal(serial, parallel)


def test_leaf_descent_threshold_ties_are_exact():
    X = np.asarray([[2], [3], [4], [3]], dtype=np.uint8)
    initial = np.asarray([0, 1, 2, 3], dtype=np.int64)
    expected = np.asarray([0, 2, 5, 6], dtype=np.int64)
    serial = initial.copy()
    parallel = initial.copy()

    _update_leaves_with_split_serial(X, serial, 0, 3)
    _update_leaves_with_split_parallel(X, parallel, 0, 3)

    np.testing.assert_array_equal(serial, expected)
    np.testing.assert_array_equal(parallel, expected)


def test_leaf_descent_router_engages_exact_boundary(monkeypatch):
    calls = []

    def serial(*args):
        calls.append("serial")

    def parallel(*args):
        calls.append("parallel")

    monkeypatch.setattr(tree_module, "_update_leaves_with_split_serial", serial)
    monkeypatch.setattr(
        tree_module, "_update_leaves_with_split_parallel", parallel
    )
    small_X = np.zeros((_SMALL_LEAF_DESCENT_ROWS - 1, 1), dtype=np.uint8)
    large_X = np.zeros((_SMALL_LEAF_DESCENT_ROWS, 1), dtype=np.uint8)

    _update_leaves_with_split(
        small_X, np.zeros(len(small_X), dtype=np.int64), 0, 0
    )
    _update_leaves_with_split(
        large_X, np.zeros(len(large_X), dtype=np.int64), 0, 0
    )

    assert calls == ["serial", "parallel"]


def _regression_data(seed=107, rows=180):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, 6))
    y = (
        1.2 * X[:, 0]
        - 0.7 * X[:, 1]
        + 0.25 * X[:, 2] * X[:, 3]
        + rng.normal(scale=0.15, size=rows)
    )
    return X, y


def _categorical_data():
    X, y = _regression_data(seed=109)
    labels = np.asarray(["guard", "wing", "big"], dtype=object)
    categories = labels[np.arange(len(X)) % len(labels)]
    mixed = np.empty((len(X), 3), dtype=object)
    mixed[:, 0] = X[:, 0]
    mixed[:, 1] = categories
    mixed[:, 2] = X[:, 1]
    y = y + 0.3 * (categories == "guard") - 0.2 * (categories == "big")
    return mixed, y


def _core_params(**overrides):
    params = {
        "iterations": 12,
        "learning_rate": 0.1,
        "depth": 3,
        "l2_leaf_reg": 3.0,
        "max_bins": 32,
        "min_child_weight": 0.0,
        "min_child_samples": 2,
        "thread_count": 4,
        "random_state": 29,
        "tree_mode": "catboost",
        "diagnostic_warnings": "never",
    }
    params.update(overrides)
    return params


def _observed_descent(reference, serial_calls):
    serial = tree_module._update_leaves_with_split_serial
    parallel = tree_module._update_leaves_with_split_parallel

    if reference:
        return parallel

    def automatic(X_binned, leaf, split_feat, split_thr):
        if leaf.shape[0] < _SMALL_LEAF_DESCENT_ROWS:
            serial_calls[0] += 1
            return serial(X_binned, leaf, split_feat, split_thr)
        return parallel(X_binned, leaf, split_feat, split_thr)

    return automatic


def _fit_core(monkeypatch, X, y, *, reference, params=None, fit_kwargs=None):
    serial_calls = np.zeros(1, dtype=np.int64)
    with monkeypatch.context() as context:
        context.setattr(
            tree_module,
            "_update_leaves_with_split",
            _observed_descent(reference, serial_calls),
        )
        model = GradientBoosting(**(params or _core_params())).fit(
            X, y, **(fit_kwargs or {})
        )
    return model, int(serial_calls[0])


def _assert_archive_exact(reference, candidate, X, tmp_path, name, *, raw=True):
    method = "predict_raw" if raw else "predict"
    np.testing.assert_array_equal(
        getattr(candidate, method)(X), getattr(reference, method)(X)
    )
    np.testing.assert_array_equal(
        candidate.feature_importances_, reference.feature_importances_
    )
    reference_path = tmp_path / f"{name}-reference.npz"
    candidate_path = tmp_path / f"{name}-candidate.npz"
    reference.save_model(reference_path)
    candidate.save_model(candidate_path)
    assert candidate_path.read_bytes() == reference_path.read_bytes()


@pytest.mark.parametrize(
    ("case", "loss", "loss_kwargs"),
    [
        ("rmse", "RMSE", {}),
        ("mae", "MAE", {}),
        ("quantile", "Quantile", {"alpha": 0.8}),
        ("logloss", "Logloss", {}),
        ("weighted", "RMSE", {}),
        ("categorical", "RMSE", {}),
    ],
)
def test_small_serial_descent_scalar_lanes_are_archive_exact(
    monkeypatch, tmp_path, case, loss, loss_kwargs
):
    X, y = _categorical_data() if case == "categorical" else _regression_data()
    params = _core_params(loss=loss, loss_kwargs=loss_kwargs)
    fit_kwargs = {}
    if case == "logloss":
        y = (y > np.median(y)).astype(np.float64)
    elif case == "weighted":
        fit_kwargs["sample_weight"] = np.linspace(0.5, 1.5, len(y))
    elif case == "categorical":
        fit_kwargs["cat_features"] = [1]

    reference, reference_calls = _fit_core(
        monkeypatch,
        X,
        y,
        reference=True,
        params=params,
        fit_kwargs=fit_kwargs,
    )
    candidate, candidate_calls = _fit_core(
        monkeypatch,
        X,
        y,
        reference=False,
        params=params,
        fit_kwargs=fit_kwargs,
    )

    assert reference_calls == 0
    assert candidate_calls > 0
    _assert_archive_exact(reference, candidate, X, tmp_path, case)


def test_small_serial_descent_callback_is_archive_exact(monkeypatch, tmp_path):
    X, y = _regression_data(seed=113)

    class StopAfterThree:
        stop_reason = "serial_descent_test"

        def __call__(self, progress):
            return progress.rounds_completed >= 3

    reference, _ = _fit_core(
        monkeypatch,
        X,
        y,
        reference=True,
        fit_kwargs={"eval_set": (X, y), "callbacks": StopAfterThree()},
    )
    candidate, candidate_calls = _fit_core(
        monkeypatch,
        X,
        y,
        reference=False,
        fit_kwargs={"eval_set": (X, y), "callbacks": StopAfterThree()},
    )

    assert candidate_calls > 0
    assert reference.stop_reason_ == candidate.stop_reason_ == "serial_descent_test"
    _assert_archive_exact(reference, candidate, X, tmp_path, "callback")


def test_small_serial_descent_exact_refit_is_archive_exact(monkeypatch, tmp_path):
    X, y = _regression_data(seed=127, rows=240)
    params = {
        "iterations": 35,
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
        "random_state": 31,
        "diagnostic_warnings": "never",
    }

    def fit(reference):
        serial_calls = np.zeros(1, dtype=np.int64)
        with monkeypatch.context() as context:
            context.setattr(
                tree_module,
                "_update_leaves_with_split",
                _observed_descent(reference, serial_calls),
            )
            model = DarkoRegressor(**params).fit(X, y)
        return model, int(serial_calls[0])

    reference, reference_calls = fit(True)
    candidate, candidate_calls = fit(False)

    assert reference_calls == 0
    assert candidate_calls > 0
    assert reference.refit_ is candidate.refit_ is True
    _assert_archive_exact(
        reference, candidate, X, tmp_path, "exact-refit", raw=False
    )


def test_small_serial_descent_hybrid_shared_trunk_is_archive_exact(
    monkeypatch, tmp_path
):
    X, y = _regression_data(seed=131)
    params = _core_params(tree_mode="hybrid", iterations=8)
    reference, reference_calls = _fit_core(
        monkeypatch, X, y, reference=True, params=params
    )
    candidate, candidate_calls = _fit_core(
        monkeypatch, X, y, reference=False, params=params
    )

    assert reference_calls == 0
    assert candidate_calls > 0
    _assert_archive_exact(reference, candidate, X, tmp_path, "hybrid")
