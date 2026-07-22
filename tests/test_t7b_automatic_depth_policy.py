from __future__ import annotations

import json
import pickle

import numba
import numpy as np
import pytest
from sklearn.base import clone

from darkofit import DarkoClassifier, DarkoRegressor


RULE = "scalar_rmse_catboost_n_eff_per_input_feature_4_6_8"


def _regression(*, n=200, p=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 1.5 * np.sin(X[:, 0])
    if p > 1:
        y = y - 0.7 * X[:, 1]
    if p > 3:
        y = y + 0.3 * X[:, 2] * X[:, 3]
    y = y + rng.normal(0.0, 0.15, size=n)
    return X, y


def _fit(X, y, *, sample_weight=None, cat_features=None, **kwargs):
    model = DarkoRegressor(
        iterations=4,
        learning_rate=0.1,
        early_stopping=False,
        ordered_boosting=False,
        random_state=0,
        diagnostic_warnings="never",
        **kwargs,
    )
    return model.fit(
        X,
        y,
        cat_features=cat_features,
        sample_weight=sample_weight,
    )


def _structure(model):
    return model.model_.auto_params_["auto_structure"]


@pytest.mark.parametrize(
    ("n", "p", "depth", "density", "branch"),
    [
        (200, 8, 4, 25.0, "low_density"),
        (200, 2, 6, 100.0, "middle_density"),
        (2_499, 1, 6, 2_499.0, "middle_density"),
        (2_500, 1, 8, 2_500.0, "high_density"),
    ],
)
def test_default_scalar_rmse_catboost_resolves_frozen_density_branches(
    n, p, depth, density, branch
):
    X, y = _regression(n=n, p=p)
    model = _fit(X, y)
    structure = _structure(model)
    policy = structure["candidates"]["depth"]

    assert model.depth is None
    assert model.model_._depth_input is None
    assert model.model_.depth == depth
    assert structure["resolved"]["depth"] == {
        "input": None,
        "resolved": depth,
        "source": "auto",
    }
    assert policy == {
        "rule": RULE,
        "n_eff": float(n),
        "input_feature_count": p,
        "effective_rows_per_feature": density,
        "low_threshold": 100.0,
        "high_threshold": 2_500.0,
        "branch": branch,
    }
    assert model.model_.l2_leaf_reg == 3.0


def test_stress_weights_change_depth_only_through_effective_sample_size():
    X, y = _regression(n=240, p=2)
    weights = np.ones(len(y), dtype=np.float64)
    weights[:24] = 16.0
    model = _fit(X, y, sample_weight=weights)
    structure = _structure(model)
    policy = structure["candidates"]["depth"]
    expected_n_eff = float(weights.sum() ** 2 / np.dot(weights, weights))

    assert expected_n_eff < 200.0
    assert model.model_.depth == 4
    assert policy["n_eff"] == pytest.approx(expected_n_eff, rel=0, abs=1e-12)
    assert policy["effective_rows_per_feature"] == pytest.approx(
        expected_n_eff / 2.0, rel=0, abs=1e-12
    )
    assert policy["branch"] == "low_density"
    assert model.model_.l2_leaf_reg == pytest.approx(
        3.0 * np.sqrt(len(y) / expected_n_eff), rel=0, abs=1e-12
    )


@pytest.mark.parametrize("depth", [3, 6, 9])
def test_explicit_numeric_depth_always_wins(depth):
    X, y = _regression()
    model = _fit(X, y, depth=depth)
    structure = _structure(model)

    assert model.model_.depth == depth
    assert structure["resolved"]["depth"] == {
        "input": depth,
        "resolved": depth,
        "source": "explicit",
    }
    assert "depth" not in structure["candidates"]


def test_literal_auto_keeps_existing_effective_row_buckets():
    X, y = _regression(n=200, p=8)
    model = _fit(X, y, depth="auto")
    structure = _structure(model)

    assert model.model_.depth == 4
    assert structure["resolved"]["depth"] == {
        "input": "auto",
        "resolved": 4,
        "source": "auto",
    }
    assert structure["candidates"]["depth"] == {
        "rule": "n_eff_buckets_4_7"
    }


def test_noneligible_losses_tasks_and_tree_modes_keep_control_depths():
    X, y = _regression(n=200, p=8)
    labels = (y > np.median(y)).astype(np.int64)
    common = {
        "iterations": 3,
        "learning_rate": 0.1,
        "early_stopping": False,
        "ordered_boosting": False,
        "random_state": 0,
        "diagnostic_warnings": "never",
    }
    cases = {
        "classifier": DarkoClassifier(**common).fit(X, labels),
        "mae": DarkoRegressor(**common, loss="MAE").fit(X, y),
        "quantile": DarkoRegressor(**common, loss="Quantile").fit(X, y),
        "lightgbm": DarkoRegressor(
            **common, tree_mode="lightgbm", num_leaves=15
        ).fit(X, y),
        "hybrid": DarkoRegressor(
            **common, tree_mode="hybrid", num_leaves=15
        ).fit(X, y),
        "depthwise": DarkoRegressor(**common, tree_mode="depthwise").fit(X, y),
        "gaussian": DarkoRegressor(
            **common,
            loss="Gaussian",
            tree_mode="lightgbm",
            num_leaves=15,
        ).fit(X, y),
    }
    expected = {
        "classifier": 6,
        "mae": 6,
        "quantile": 6,
        "lightgbm": -1,
        "hybrid": -1,
        "depthwise": 2,
        "gaussian": -1,
    }

    for name, model in cases.items():
        structure = _structure(model)
        assert model.model_.depth == expected[name]
        assert structure["resolved"]["depth"]["source"] == "default"
        if name == "depthwise":
            assert structure["candidates"]["depth"] == {
                "rule": "depthwise_rmse_shallow_default"
            }
        else:
            assert "depth" not in structure["candidates"]


def test_oblivious_alias_uses_normalized_catboost_policy():
    X, y = _regression(n=200, p=8)
    model = _fit(X, y, tree_mode="oblivious")

    assert model.model_.tree_mode_ == "catboost"
    assert model.model_.depth == 4
    assert _structure(model)["candidates"]["depth"]["rule"] == RULE


def test_feature_names_and_categoricals_use_input_feature_count():
    pandas = pytest.importorskip("pandas")
    n = 240
    rng = np.random.default_rng(12)
    X = pandas.DataFrame(
        {
            "numeric": rng.normal(size=n),
            "category": np.array([f"c{i % 6}" for i in range(n)]),
            "other": rng.normal(size=n),
        }
    )
    y = X["numeric"].to_numpy() + rng.normal(0.0, 0.1, size=n)
    model = _fit(X, y, cat_features=["category"])
    policy = _structure(model)["candidates"]["depth"]

    assert model.feature_names_in_.tolist() == list(X.columns)
    assert policy["input_feature_count"] == 3
    assert policy["effective_rows_per_feature"] == 80.0
    assert model.model_.depth == 4


def test_safe_npz_pickle_clone_and_refit_preserve_policy(tmp_path):
    X, y = _regression(n=200, p=8)
    model = _fit(X, y)
    expected = model.predict(X[:24])
    path = tmp_path / "model.npz"
    model.save_model(path)
    loaded = DarkoRegressor.load_model(path)
    pickled = pickle.loads(pickle.dumps(model))

    for restored in (loaded, pickled):
        assert restored.depth is None
        assert restored.model_._depth_input is None
        assert restored.model_.depth == 4
        assert _structure(restored)["candidates"]["depth"]["rule"] == RULE
        assert np.array_equal(restored.predict(X[:24]), expected)

    cloned = clone(model)
    assert cloned.depth is None
    assert cloned.get_params()["depth"] is None
    assert model.get_refit_params()["depth"] == 4


def test_safe_npz_rejects_inconsistent_automatic_depth_metadata(tmp_path):
    X, y = _regression(n=200, p=8)
    model = _fit(X, y)
    path = tmp_path / "model.npz"
    model.save_model(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    header = json.loads(str(arrays["header"]))
    depth_policy = header["auto_params"]["auto_structure"]["candidates"][
        "depth"
    ]
    depth_policy["branch"] = "high_density"
    arrays["header"] = np.array(json.dumps(header))
    corrupted = tmp_path / "corrupted.npz"
    np.savez_compressed(corrupted, **arrays)

    with pytest.raises(ValueError, match="depth branch is inconsistent"):
        DarkoRegressor.load_model(corrupted)

    header = json.loads(str(arrays["header"]))
    del header["auto_params"]["auto_structure"]["candidates"]["depth"]
    arrays["header"] = np.array(json.dumps(header))
    missing_policy = tmp_path / "missing-policy.npz"
    np.savez_compressed(missing_policy, **arrays)
    with pytest.raises(ValueError, match="depth metadata is inconsistent"):
        DarkoRegressor.load_model(missing_policy)


def test_refit_freezes_selection_depth_across_density_boundary():
    X, y = _regression(n=210, p=2)
    model = DarkoRegressor(
        iterations=6,
        learning_rate=0.1,
        early_stopping=True,
        early_stopping_rounds=2,
        validation_fraction=0.1,
        refit=True,
        ordered_boosting=False,
        random_state=0,
        diagnostic_warnings="never",
    ).fit(X, y)

    selection_policy = model.selection_model_.auto_params_[
        "auto_structure"
    ]["candidates"]["depth"]
    assert selection_policy["branch"] == "low_density"
    assert selection_policy["effective_rows_per_feature"] < 100.0
    assert model.selection_model_.depth == 4
    assert model.model_.depth == 4
    assert model.model_._depth_input == 4
    assert model.get_refit_params()["depth"] == 4


def test_fit_predict_and_empty_predict_restore_ambient_numba_mask():
    available = int(numba.config.NUMBA_NUM_THREADS)
    ambient = min(available, 3)
    previous = int(numba.get_num_threads())
    X, y = _regression()
    try:
        numba.set_num_threads(ambient)
        model = _fit(X, y, thread_count=1)
        assert numba.get_num_threads() == ambient
        model.predict(X[:12])
        assert numba.get_num_threads() == ambient
        empty = model.predict(X[:0])
        assert empty.shape == (0,)
        assert numba.get_num_threads() == ambient
    finally:
        numba.set_num_threads(previous)
