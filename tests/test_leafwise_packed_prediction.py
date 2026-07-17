"""Exactness gates for bounded scalar leafwise packed prediction."""

import numpy as np
import pytest

from darkofit import DarkoClassifier, DarkoRegressor
from darkofit.flat_model import (
    FlatNonObliviousEnsemble,
    flat_predict_preferred,
)


def _numeric_data(n_rows=600):
    rng = np.random.default_rng(410)
    X = rng.normal(size=(n_rows, 6))
    y = (
        1.4 * X[:, 0]
        - 0.8 * X[:, 1] * X[:, 2]
        + np.sin(2.0 * X[:, 3])
        + rng.normal(scale=0.35, size=n_rows)
    )
    return X, y


def _categorical_data(n_rows=600):
    X, y = _numeric_data(n_rows)
    categories = np.asarray(["guard", "wing", "center"], dtype=object)
    labels = categories[np.arange(n_rows) % len(categories)]
    mixed = np.empty((n_rows, X.shape[1] + 1), dtype=object)
    mixed[:, 0] = X[:, 0]
    mixed[:, 1] = labels
    mixed[:, 2:] = X[:, 1:]
    y = y + np.where(labels == "guard", 0.6, -0.2)
    return mixed, y


def _model_params(**overrides):
    params = {
        "iterations": 64,
        "learning_rate": 0.1,
        "depth": 4,
        "l2_leaf_reg": 1,
        "max_bins": 32,
        "tree_mode": "lightgbm",
        "ordered_boosting": False,
        "use_best_model": False,
        "early_stopping": False,
        "min_child_samples": 5,
        "thread_count": 2,
        "random_state": 17,
        "diagnostic_warnings": "never",
    }
    params.update(overrides)
    return params


def _loop_predict_raw(core, X):
    core._restore_thread_count()
    values = (
        np.asarray(X, dtype=object)
        if core.prep_.cat_features_
        else np.asarray(X, dtype=np.float64)
    )
    X_binned = core.prep_.transform(values)
    prediction = np.full(X_binned.shape[0], core.init_, dtype=np.float64)
    for tree in core.trees_:
        tree.add_predict(X_binned, prediction)
    return prediction


def _assert_selected(core, n_rows):
    flat = core._flat_ensemble()
    assert isinstance(flat, FlatNonObliviousEnsemble)
    assert flat_predict_preferred(flat, n_rows, core.tree_mode_)
    return flat


@pytest.fixture(scope="module")
def numeric_model():
    X, y = _numeric_data()
    model = DarkoRegressor(**_model_params()).fit(X, y)
    assert model.model_.n_threads_ == 2
    assert len(model.model_.trees_) == 64
    return model, X


@pytest.mark.parametrize(
    "case",
    ["numeric_rmse", "weighted_rmse", "categorical_rmse", "mae", "quantile"],
)
def test_leafwise_packed_public_regression_parity(case):
    if case == "categorical_rmse":
        X, y = _categorical_data()
        fit_kwargs = {"cat_features": [1]}
        model_kwargs = {}
    else:
        X, y = _numeric_data()
        fit_kwargs = {}
        model_kwargs = {}
    if case == "weighted_rmse":
        fit_kwargs["sample_weight"] = np.linspace(0.5, 1.5, len(y))
    elif case == "mae":
        model_kwargs["loss"] = "MAE"
    elif case == "quantile":
        model_kwargs.update(loss="Quantile", alpha=0.3)

    model = DarkoRegressor(**_model_params(**model_kwargs)).fit(
        X, y, **fit_kwargs
    )
    _assert_selected(model.model_, len(X))
    assert np.array_equal(model.predict(X), _loop_predict_raw(model.model_, X))


def test_leafwise_packed_public_binary_parity():
    X, signal = _numeric_data()
    y = (signal > np.median(signal)).astype(np.int64)
    model = DarkoClassifier(**_model_params()).fit(X, y)
    _assert_selected(model.model_, len(X))
    assert np.array_equal(
        model.model_.predict_raw(X),
        _loop_predict_raw(model.model_, X),
    )


def test_leafwise_packed_direct_kernel_and_public_dispatch(
    numeric_model, monkeypatch
):
    import darkofit.flat_model as flat_module

    model, X = numeric_model
    core = model.model_
    flat = _assert_selected(core, len(X))
    X_binned = core.prep_.transform(np.asarray(X, dtype=np.float64))

    packed = np.zeros(len(X), dtype=np.float64)
    flat.add_predict_scalar_packed(X_binned, packed)
    loop = np.zeros(len(X), dtype=np.float64)
    for tree in core.trees_:
        tree.add_predict(X_binned, loop)
    assert np.array_equal(packed, loop)

    calls = []
    original = flat_module._flat_nonoblivious_scalar_add_parallel

    def observed(*args):
        calls.append(int(args[0].shape[0]))
        return original(*args)

    monkeypatch.setattr(
        flat_module, "_flat_nonoblivious_scalar_add_parallel", observed
    )
    assert np.array_equal(model.predict(X), _loop_predict_raw(core, X))
    assert calls == [len(X)]


def test_leafwise_packed_router_boundaries_and_mode(numeric_model):
    import numba

    model, _ = numeric_model
    trees = model.model_.trees_
    repeated = (trees * 5)[:259]
    cases = (
        (1, 32768, False),
        (5, 8192, True),
        (16, 2409, True),
        (25, 525, False),
        (62, 525, False),
        (63, 525, True),
        (258, 127, False),
        (259, 127, True),
        (259, 32768, True),
        (259, 65536, False),
    )
    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(2)
        for tree_count, row_count, expected in cases:
            flat = FlatNonObliviousEnsemble(repeated[:tree_count])
            assert (
                flat_predict_preferred(flat, row_count, "lightgbm") is expected
            )
        flat = FlatNonObliviousEnsemble(repeated)
        assert not flat_predict_preferred(flat, 600, "hybrid")
        flat.class_ids = np.zeros(len(repeated), dtype=np.int64)
        assert not flat_predict_preferred(flat, 600, "lightgbm")
    finally:
        numba.set_num_threads(previous)


@pytest.mark.parametrize("thread_count", [1, 4, 18])
def test_leafwise_packed_other_threads_keep_tree_loop(
    numeric_model, monkeypatch, thread_count
):
    import numba
    import darkofit.booster as booster_module
    import darkofit.flat_model as flat_module

    if thread_count > numba.config.NUMBA_NUM_THREADS:
        pytest.skip(f"Numba runtime exposes fewer than {thread_count} threads")
    model, X = numeric_model
    core = model.model_
    previous = core.n_threads_
    entered = []
    selections = []

    def forbidden(*args):
        entered.append(True)
        raise AssertionError("scalar packed kernel entered fallback lane")

    original_selector = booster_module.flat_predict_preferred

    def observed_selector(flat, n_rows=None, tree_mode=None):
        selected = original_selector(flat, n_rows, tree_mode)
        selections.append((int(n_rows), tree_mode, bool(selected)))
        return selected

    monkeypatch.setattr(
        flat_module, "_flat_nonoblivious_scalar_add_parallel", forbidden
    )
    monkeypatch.setattr(
        booster_module, "flat_predict_preferred", observed_selector
    )
    try:
        core.n_threads_ = thread_count
        core._restore_thread_count()
        flat = core._flat_ensemble()
        assert not flat_predict_preferred(flat, len(X), core.tree_mode_)
        assert np.array_equal(model.predict(X), _loop_predict_raw(core, X))
        assert not entered
        assert selections == [(len(X), "lightgbm", False)]
    finally:
        core.n_threads_ = previous
        core._restore_thread_count()


def test_leafwise_packed_serialization_staging_and_cache(
    numeric_model, tmp_path
):
    model, X = numeric_model
    path = tmp_path / "leafwise.npz"
    roundtrip_path = tmp_path / "leafwise-roundtrip.npz"
    model.save_model(path)
    before = path.read_bytes()

    loaded = DarkoRegressor.load_model(path)
    prediction = loaded.predict(X)
    assert np.array_equal(prediction, _loop_predict_raw(loaded.model_, X))
    assert np.array_equal(
        prediction,
        list(loaded.model_.staged_predict_raw(X))[-1],
    )

    cache = loaded.model_._flat_cache_
    flat = cache[1]
    packed_bytes = sum(
        value.nbytes
        for slot in flat.__slots__
        if isinstance((value := getattr(flat, slot)), np.ndarray)
    )
    loaded.predict(X)
    assert loaded.model_._flat_cache_ is cache
    assert sum(
        value.nbytes
        for slot in flat.__slots__
        if isinstance((value := getattr(flat, slot)), np.ndarray)
    ) == packed_bytes

    loaded.save_model(roundtrip_path)
    assert roundtrip_path.read_bytes() == before
