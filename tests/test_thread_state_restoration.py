import numba
import numpy as np
import pytest

from darkofit import DarkoRegressor


def _small_regression_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(160, 4))
    y = 1.5 * X[:, 0] - 0.4 * X[:, 1] + 0.1 * rng.normal(size=X.shape[0])
    return X, y


def _small_lightgbm_regressor(**overrides):
    params = {
        "iterations": 3,
        "learning_rate": 0.1,
        "tree_mode": "lightgbm",
        "num_leaves": 7,
        "min_child_samples": 5,
        "thread_count": None,
        "random_state": 0,
        "diagnostic_warnings": "never",
    }
    params.update(overrides)
    return DarkoRegressor(**params)


@pytest.fixture
def ambient_numba_thread_count():
    max_threads = int(numba.config.NUMBA_NUM_THREADS)
    if max_threads < 2:
        pytest.skip("requires two available Numba threads")

    original_threads = numba.get_num_threads()
    ambient_threads = 1 if max_threads == 2 else min(4, max_threads)
    numba.set_num_threads(ambient_threads)
    try:
        yield ambient_threads
    finally:
        numba.set_num_threads(original_threads)


def test_fit_and_predict_restore_callers_numba_thread_mask(
    ambient_numba_thread_count,
):
    X, y = _small_regression_data()
    model = _small_lightgbm_regressor().fit(X, y)

    assert model.model_.n_threads_ == 2
    assert numba.get_num_threads() == ambient_numba_thread_count

    prediction = model.predict(X[:12])

    assert prediction.shape == (12,)
    assert numba.get_num_threads() == ambient_numba_thread_count


def test_predict_during_fit_restores_outer_fitted_thread_mask(
    ambient_numba_thread_count,
):
    X, y = _small_regression_data()
    inner = _small_lightgbm_regressor(
        iterations=1,
        thread_count=1,
    ).fit(X, y)
    observed_thread_counts = []

    def predict_during_fit(progress):
        before_predict = numba.get_num_threads()
        inner.predict(X[:8])
        observed_thread_counts.append(
            (before_predict, numba.get_num_threads())
        )
        return False

    outer = _small_lightgbm_regressor(iterations=2).fit(
        X,
        y,
        callbacks=predict_during_fit,
    )

    assert outer.model_.n_threads_ == 2
    assert observed_thread_counts
    assert observed_thread_counts == [(2, 2)] * len(observed_thread_counts)
    assert numba.get_num_threads() == ambient_numba_thread_count


def test_fit_restores_callers_numba_thread_mask_after_callback_error(
    ambient_numba_thread_count,
):
    X, y = _small_regression_data()

    def fail_during_fit(_progress):
        assert numba.get_num_threads() == 2
        raise RuntimeError("simulated callback failure")

    with pytest.raises(RuntimeError, match="simulated callback failure"):
        _small_lightgbm_regressor(iterations=2).fit(
            X,
            y,
            callbacks=fail_during_fit,
        )

    assert numba.get_num_threads() == ambient_numba_thread_count


def test_predict_restores_callers_numba_thread_mask_after_error(
    ambient_numba_thread_count, monkeypatch
):
    X, y = _small_regression_data()
    model = _small_lightgbm_regressor().fit(X, y).model_

    def fail_during_predict(*_args, **_kwargs):
        assert numba.get_num_threads() == 2
        raise RuntimeError("simulated prediction failure")

    monkeypatch.setattr(model, "_prepare_predict_X", fail_during_predict)
    with pytest.raises(RuntimeError, match="simulated prediction failure"):
        model.predict_raw(X[:8])

    assert numba.get_num_threads() == ambient_numba_thread_count


def test_staged_predict_restores_callers_numba_thread_mask_between_steps(
    ambient_numba_thread_count,
):
    X, y = _small_regression_data()
    model = _small_lightgbm_regressor().fit(X, y).model_
    predictions = model.staged_predict_raw(X[:8])

    first = next(predictions)
    assert first.shape == (8,)
    assert numba.get_num_threads() == ambient_numba_thread_count

    predictions.close()
    assert numba.get_num_threads() == ambient_numba_thread_count
