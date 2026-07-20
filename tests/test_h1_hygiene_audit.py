"""Named regression and reproducer coverage for Track H item H1."""

from __future__ import annotations

import numpy as np
import pytest

from darkofit import DarkoClassifier, DarkoRegressor


def _regression_data(n_samples=80):
    rng = np.random.default_rng(20260720)
    X = rng.normal(size=(n_samples, 5))
    y = 1.4 * X[:, 0] - 0.6 * X[:, 1] + 0.1 * rng.normal(size=n_samples)
    return X, y


def test_h1_serialization_excludes_rebuildable_flat_predictor_cache(tmp_path):
    X, y = _regression_data()
    model = DarkoRegressor(
        iterations=3,
        learning_rate=0.1,
        depth=2,
        random_state=0,
        diagnostic_warnings="never",
    ).fit(X, y)
    before_path = tmp_path / "before-predict.npz"
    after_path = tmp_path / "after-predict.npz"

    model.save_model(before_path)
    expected = model.predict(X)
    assert model.model_._flat_cache_[0] is model.model_.trees_
    model.save_model(after_path)

    assert after_path.read_bytes() == before_path.read_bytes()
    loaded = DarkoRegressor.load_model(after_path)
    assert getattr(loaded.model_, "_flat_cache_", None) is None
    np.testing.assert_array_equal(loaded.predict(X), expected)


@pytest.mark.parametrize(
    ("training_labels", "bad_label"),
    [
        (np.array([0, 1] * 20), 2),
        (np.array([0, 1, 2, 0] * 10), 3),
    ],
)
def test_h1_classifier_eval_set_rejects_unseen_labels(
    training_labels,
    bad_label,
):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(len(training_labels), 4))
    eval_labels = training_labels[:8].copy()
    eval_labels[0] = bad_label
    model = DarkoClassifier(iterations=1, random_state=0)

    with pytest.raises(ValueError, match="labels not present in training"):
        model.fit(
            X,
            training_labels,
            eval_set=(X[:8], eval_labels),
        )
    assert not hasattr(model, "model_")


@pytest.mark.parametrize(
    ("estimator", "classification"),
    [
        (DarkoRegressor, False),
        (DarkoClassifier, True),
    ],
)
def test_h1_positional_sample_weight_misuse_names_keyword(
    estimator,
    classification,
):
    X, y = _regression_data()
    if classification:
        y = (y > np.median(y)).astype(np.int64)
    weights = np.linspace(0.5, 1.5, len(y))
    model = estimator(iterations=1, random_state=0)

    with pytest.raises(ValueError, match=r"sample_weight=w"):
        model.fit(X, y, weights)
    assert not hasattr(model, "model_")


@pytest.mark.parametrize(
    ("tree_mode", "expected_depth"),
    [("catboost", 6), ("lightgbm", -1)],
)
def test_h1_classifier_depth_none_uses_documented_mode_default(
    tree_mode,
    expected_depth,
):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(40, 4))
    y = np.array([0, 1] * 20)
    model = DarkoClassifier(
        iterations=0,
        learning_rate=0.1,
        depth=None,
        tree_mode=tree_mode,
        random_state=0,
        diagnostic_warnings="never",
    ).fit(X, y)

    resolution = model.model_.auto_params_["auto_structure"]["resolved"][
        "depth"
    ]
    assert model.model_.depth == expected_depth
    assert resolution == {
        "input": None,
        "resolved": expected_depth,
        "source": "default",
    }
