from __future__ import annotations

import pickle

import numba
import numpy as np
import pytest
from sklearn.base import clone

from darkofit import DarkoClassifier, DarkoRegressor


RULE = "scalar_rmse_catboost_base_1_times_weight_concentration"


def _regression(seed=0, n=180, p=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (
        1.5 * np.sin(X[:, 0])
        - 0.7 * X[:, 1]
        + 0.3 * X[:, 2] * X[:, 3]
        + rng.normal(0.0, 0.15, size=n)
    )
    return X, y


def _fit_regressor(**kwargs):
    X, y = _regression()
    model = DarkoRegressor(
        iterations=5,
        learning_rate=0.1,
        depth=3,
        ordered_boosting=False,
        random_state=0,
        diagnostic_warnings="never",
        **kwargs,
    ).fit(X, y)
    return model, X, y


def _l2_metadata(model):
    return model.model_.auto_params_["auto_structure"]


def test_scalar_rmse_catboost_auto_l2_resolves_to_one_and_records_rule():
    model, _, _ = _fit_regressor()
    metadata = _l2_metadata(model)

    assert model.l2_leaf_reg == "auto"
    assert model.model_.l2_leaf_reg == 1.0
    assert metadata["resolved"]["l2_leaf_reg"] == {
        "input": "auto",
        "resolved": 1.0,
        "source": "auto",
    }
    assert metadata["candidates"]["l2_leaf_reg"] == {
        "rule": RULE,
        "base": 1.0,
    }
    assert model.model_.auto_params_["tree"]["l2_leaf_reg"] == 1.0


def test_oblivious_alias_uses_the_normalized_catboost_policy():
    model, _, _ = _fit_regressor(tree_mode="oblivious")

    assert model.model_.tree_mode_ == "catboost"
    assert model.model_.l2_leaf_reg == 1.0
    assert _l2_metadata(model)["candidates"]["l2_leaf_reg"]["rule"] == RULE


def test_scalar_rmse_catboost_auto_l2_retains_weight_concentration():
    X, y = _regression(n=120)
    weights = np.ones(len(y), dtype=np.float64)
    weights[:6] = 20.0
    model = DarkoRegressor(
        iterations=3,
        learning_rate=0.1,
        depth=3,
        ordered_boosting=False,
        random_state=0,
        diagnostic_warnings="never",
    ).fit(X, y, sample_weight=weights)
    metadata = _l2_metadata(model)
    fraction = float(metadata["n_eff_fraction"])
    expected = float(np.clip(np.sqrt(1.0 / fraction), 1.0, 20.0))

    assert fraction < 1.0
    assert model.model_.l2_leaf_reg == pytest.approx(expected, rel=0, abs=1e-15)
    assert metadata["resolved"]["l2_leaf_reg"]["resolved"] == pytest.approx(
        expected, rel=0, abs=1e-15
    )
    assert metadata["candidates"]["l2_leaf_reg"] == {
        "rule": RULE,
        "base": 1.0,
    }


@pytest.mark.parametrize("value", [0.5, 3.0, 11.0])
def test_explicit_l2_values_still_win(value):
    model, _, _ = _fit_regressor(l2_leaf_reg=value)
    metadata = _l2_metadata(model)

    assert model.model_.l2_leaf_reg == value
    assert metadata["resolved"]["l2_leaf_reg"]["resolved"] == value
    assert metadata["resolved"]["l2_leaf_reg"]["source"] in {
        "default",
        "explicit",
    }
    assert "l2_leaf_reg" not in metadata["candidates"]


def test_non_rmse_and_non_catboost_auto_l2_resolutions_are_unchanged():
    X, y = _regression()
    labels = (y > np.median(y)).astype(np.int64)
    common = {
        "iterations": 3,
        "learning_rate": 0.1,
        "depth": 3,
        "ordered_boosting": False,
        "random_state": 0,
        "diagnostic_warnings": "never",
    }
    cases = {
        "classifier": DarkoClassifier(**common).fit(X, labels),
        "mae": DarkoRegressor(**common, loss="MAE").fit(X, y),
        "lightgbm": DarkoRegressor(
            **common, tree_mode="lightgbm", num_leaves=7
        ).fit(X, y),
        "hybrid": DarkoRegressor(
            **common, tree_mode="hybrid", num_leaves=7
        ).fit(X, y),
        "depthwise": DarkoRegressor(**common, tree_mode="depthwise").fit(X, y),
    }
    expected = {
        "classifier": 3.0,
        "mae": 3.0,
        "lightgbm": 1.0,
        "hybrid": 2.0,
        "depthwise": 3.0,
    }

    for name, model in cases.items():
        metadata = _l2_metadata(model)
        core = model.model_
        resolved = float(getattr(core, "l2_leaf_reg_", core.l2_leaf_reg))
        assert resolved == expected[name]
        assert metadata["resolved"]["l2_leaf_reg"]["resolved"] == expected[name]
        assert metadata["candidates"]["l2_leaf_reg"]["rule"] == (
            "base_by_tree_mode_times_weight_concentration"
        )


def test_safe_npz_pickle_clone_and_refit_preserve_policy(tmp_path):
    model, X, _ = _fit_regressor()
    expected = model.predict(X[:24])
    path = tmp_path / "model.npz"
    model.save_model(path)
    loaded = DarkoRegressor.load_model(path)
    pickled = pickle.loads(pickle.dumps(model))

    for restored in (loaded, pickled):
        assert restored.l2_leaf_reg == "auto"
        assert _l2_metadata(restored)["resolved"]["l2_leaf_reg"] == {
            "input": "auto",
            "resolved": 1.0,
            "source": "auto",
        }
        assert restored.model_.auto_params_["tree"]["l2_leaf_reg"] == 1.0
        assert _l2_metadata(restored)["candidates"]["l2_leaf_reg"] == {
            "rule": RULE,
            "base": 1.0,
        }
        assert np.array_equal(restored.predict(X[:24]), expected)

    cloned = clone(model)
    assert cloned.l2_leaf_reg == "auto"
    assert cloned.get_params()["l2_leaf_reg"] == "auto"
    assert model.get_refit_params()["l2_leaf_reg"] == 1.0


def test_fit_predict_and_empty_predict_restore_ambient_numba_mask():
    available = int(numba.config.NUMBA_NUM_THREADS)
    ambient = min(available, 3)
    previous = int(numba.get_num_threads())
    try:
        numba.set_num_threads(ambient)
        model, X, _ = _fit_regressor(thread_count=1)
        assert numba.get_num_threads() == ambient
        model.predict(X[:12])
        assert numba.get_num_threads() == ambient
        empty = model.predict(X[:0])
        assert empty.shape == (0,)
        assert numba.get_num_threads() == ambient
    finally:
        numba.set_num_threads(previous)
