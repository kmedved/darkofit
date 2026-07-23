from __future__ import annotations

import json

import numba
import numpy as np
import pytest
from sklearn.base import clone

import darkofit.sklearn_api as sklearn_api
from darkofit import DarkoClassifier, DarkoRegressor


def _centered_problem(n: int = 800, seed: int = 7):
    rng = np.random.default_rng(seed)
    category = rng.integers(0, 8, n)
    baseline = np.array([-9.0, -6.0, -3.0, -1.0, 2.0, 5.0, 8.0, 11.0])[
        category
    ]
    deviation = rng.normal(0.0, 2.0, n)
    numeric = baseline + deviation
    y = deviation + rng.normal(0.0, 0.15, n)
    X = np.empty((n, 3), dtype=object)
    X[:, 0] = category.astype(str)
    X[:, 1] = numeric
    X[:, 2] = rng.normal(size=n)
    groups = np.repeat(np.arange((n + 4) // 5), 5)[:n]
    return X, y, groups


def _estimator(**overrides):
    params = {
        "iterations": 100,
        "learning_rate": 0.08,
        "depth": 2,
        "max_bins": 64,
        "random_state": 11,
        "thread_count": 2,
        "diagnostic_warnings": "never",
    }
    params.update(overrides)
    return DarkoRegressor(**params)


@pytest.fixture(scope="module")
def selected_fit():
    patch = pytest.MonkeyPatch()
    patch.setattr(
        sklearn_api, "_GROUP_CENTERED_CROSSES_MIN_SELECTION_ROWS", 100
    )
    X, y, groups = _centered_problem()
    model = _estimator().fit(X, y, cat_features=[0], groups=groups)
    yield model, X, y, groups
    patch.undo()


def _archive_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _write_archive(path, arrays):
    np.savez_compressed(path, **arrays)


def test_automatic_selector_engages_and_records_disjoint_provenance(
    selected_fit,
) -> None:
    model, _, _, _ = selected_fit
    metadata = model.group_centered_categorical_crosses_

    assert metadata["eligible"] is True
    assert metadata["selected"] is True
    assert metadata["reason"] == "selected_augmented"
    assert metadata["augmented_validation_rmse"] < metadata[
        "control_validation_rmse"
    ]
    assert metadata["split"]["rows_disjoint"] is True
    assert metadata["split"]["group_disjoint"] is True
    assert len(metadata["pairs"]) <= 12
    assert metadata["final_pairs"] == metadata["pairs"]
    assert metadata["final_preprocessing"]["pairs"] == metadata["pairs"]
    assert len(metadata["final_preprocessing"]["means_sha256"]) == 64
    assert model.feature_importances_.shape == (3,)


def test_selected_final_fit_is_exact_to_forced_recorded_lane(selected_fit) -> None:
    model, X, y, groups = selected_fit
    pairs = [tuple(pair) for pair in model.group_centered_categorical_crosses_["pairs"]]
    forced = _estimator()
    forced._group_centered_crosses_private_mode = "forced"
    forced._group_centered_pairs_override = pairs
    forced.fit(X, y, cat_features=[0], groups=groups)

    np.testing.assert_array_equal(model.predict(X), forced.predict(X))
    np.testing.assert_array_equal(
        model.feature_importances_, forced.feature_importances_
    )
    assert model.best_n_estimators_ == forced.best_n_estimators_
    assert model.model_.prep_.group_centered_pairs_ == pairs
    for observed, expected in zip(
        model.model_.prep_.group_centered_means_,
        forced.model_.prep_.group_centered_means_,
    ):
        np.testing.assert_array_equal(observed, expected)


def test_eligible_control_win_is_exact_to_forced_control(monkeypatch) -> None:
    monkeypatch.setattr(
        sklearn_api, "_GROUP_CENTERED_CROSSES_MIN_SELECTION_ROWS", 100
    )
    X, _, _ = _centered_problem(n=300, seed=15)
    y = np.full(len(X), 2.0)
    automatic = _estimator(iterations=12).fit(X, y, cat_features=[0])
    control = _estimator(iterations=12)
    control._group_centered_crosses_private_mode = "forced"
    control._group_centered_pairs_override = []
    control.fit(X, y, cat_features=[0])

    metadata = automatic.group_centered_categorical_crosses_
    assert metadata["eligible"] is True
    assert metadata["selected"] is False
    assert metadata["reason"] == "control_won"
    assert metadata["final_pairs"] == []
    np.testing.assert_array_equal(automatic.predict(X), control.predict(X))
    np.testing.assert_array_equal(
        automatic.feature_importances_, control.feature_importances_
    )


@pytest.mark.parametrize(
    "X, cat_features, overrides, reason",
    [
        (
            np.arange(240, dtype=np.float64).reshape(120, 2),
            None,
            {},
            "no_categorical_features",
        ),
        (
            _centered_problem(n=40, seed=21)[0],
            [0],
            {},
            "below_min_samples",
        ),
        (
            _centered_problem(n=120, seed=22)[0],
            [0],
            {"tree_mode": "lightgbm"},
            "non_catboost_tree_mode",
        ),
    ],
)
def test_ineligible_fits_are_exact_control_with_stable_reason(
    X, cat_features, overrides, reason
) -> None:
    y = np.linspace(-1.0, 1.0, len(X))
    automatic = _estimator(iterations=8, **overrides).fit(
        X, y, cat_features=cat_features
    )
    control = _estimator(iterations=8, **overrides)
    control._group_centered_crosses_private_mode = "off"
    control.fit(X, y, cat_features=cat_features)

    assert automatic.group_centered_categorical_crosses_["reason"] == reason
    np.testing.assert_array_equal(automatic.predict(X), control.predict(X))
    np.testing.assert_array_equal(
        automatic.feature_importances_, control.feature_importances_
    )


def test_classifier_and_clone_surface_are_unchanged() -> None:
    X, y, _ = _centered_problem(n=120, seed=30)
    classifier = DarkoClassifier(
        iterations=8, random_state=2, diagnostic_warnings="never"
    ).fit(X, y > np.median(y), cat_features=[0])
    assert not hasattr(classifier, "group_centered_categorical_crosses_")

    regressor = _estimator(iterations=8)
    cloned = clone(regressor)
    assert cloned.get_params() == regressor.get_params()
    assert not hasattr(cloned, "_group_centered_crosses_private_mode")


def test_ineligible_ensemble_remains_operational() -> None:
    X, y, _ = _centered_problem(n=80, seed=31)
    model = _estimator(iterations=2, n_ensembles=2).fit(
        X, y, cat_features=[0]
    )
    control = _estimator(iterations=2, n_ensembles=2)
    control._group_centered_crosses_private_mode = "off"
    control.fit(X, y, cat_features=[0])

    assert model.group_centered_categorical_crosses_["reason"] == "ensemble"
    assert len(model.estimators_) == 2
    np.testing.assert_array_equal(model.predict(X), control.predict(X))
    assert all(
        "group_centered_categorical_crosses" not in member.model_.auto_params_
        for member in model.estimators_
    )


def test_selected_prediction_contract_and_thread_restoration(
    selected_fit, monkeypatch
) -> None:
    model, X, _, _ = selected_fit
    staged = list(model.staged_predict(X[:17]))
    assert staged
    np.testing.assert_array_equal(staged[-1], model.predict(X[:17]))
    assert model.predict(X[:0]).shape == (0,)
    with pytest.raises(NotImplementedError, match="group-centered"):
        model.shap_values(X[:4])

    ambient = numba.get_num_threads()
    model.predict(X[:20])
    assert numba.get_num_threads() == ambient

    monkeypatch.setattr(
        sklearn_api, "_GROUP_CENTERED_CROSSES_MIN_SELECTION_ROWS", 100
    )
    X2, y2, _ = _centered_problem(n=300, seed=33)
    _estimator(iterations=12).fit(X2, y2, cat_features=[0])
    assert numba.get_num_threads() == ambient


def test_selected_pandas_feature_names_and_missing_values(monkeypatch) -> None:
    pd = pytest.importorskip("pandas")
    monkeypatch.setattr(
        sklearn_api, "_GROUP_CENTERED_CROSSES_MIN_SELECTION_ROWS", 100
    )
    X, y, _ = _centered_problem(n=300, seed=40)
    frame = pd.DataFrame(
        {"category": X[:, 0], "signal": X[:, 1], "noise": X[:, 2]}
    )
    frame.loc[3, "category"] = None
    model = _estimator(iterations=25).fit(
        frame, y, cat_features=["category"]
    )

    assert model.feature_names_in_.tolist() == list(frame.columns)
    test = frame.iloc[:5].copy()
    test.loc[test.index[0], "category"] = "unseen"
    assert np.all(np.isfinite(model.predict(test)))


def test_selected_safe_npz_round_trip_and_corruption_rejection(
    selected_fit, tmp_path
) -> None:
    model, X, _, _ = selected_fit
    first = tmp_path / "selected.npz"
    second = tmp_path / "selected-resaved.npz"
    model.save_model(first)
    loaded = DarkoRegressor.load_model(first)
    loaded.save_model(second)

    np.testing.assert_array_equal(model.predict(X[:30]), loaded.predict(X[:30]))
    assert (
        loaded.group_centered_categorical_crosses_
        == model.group_centered_categorical_crosses_
    )
    assert first.read_bytes() == second.read_bytes()
    arrays = _archive_arrays(first)
    header = json.loads(str(arrays["header"]))
    assert header["format_version"] == 5
    assert header["prep"]["group_centered_pair_count"] > 0

    missing = dict(arrays)
    missing.pop("prep__group_centered_means_flat")
    missing_path = tmp_path / "missing-means.npz"
    _write_archive(missing_path, missing)
    with pytest.raises(ValueError, match="missing group-centered"):
        DarkoRegressor.load_model(missing_path)

    wrong_pair = dict(arrays)
    wrong_pair["prep__group_centered_pairs"] = arrays[
        "prep__group_centered_pairs"
    ].copy()
    wrong_pair["prep__group_centered_pairs"][0] = [0, 1]
    wrong_pair_path = tmp_path / "wrong-pair.npz"
    _write_archive(wrong_pair_path, wrong_pair)
    with pytest.raises(ValueError, match="one numeric and one categorical"):
        DarkoRegressor.load_model(wrong_pair_path)

    wrong_metadata = dict(arrays)
    wrong_header = json.loads(str(arrays["header"]))
    wrong_header["wrapper"]["state"][
        "group_centered_categorical_crosses"
    ]["final_pairs"] = []
    wrong_metadata["header"] = np.array(json.dumps(wrong_header))
    wrong_metadata_path = tmp_path / "wrong-metadata.npz"
    _write_archive(wrong_metadata_path, wrong_metadata)
    with pytest.raises(ValueError, match="provenance disagree"):
        DarkoRegressor.load_model(wrong_metadata_path)
