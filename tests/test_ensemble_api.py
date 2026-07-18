import json

import numpy as np
import pytest

from darkofit import DarkoClassifier, DarkoRegressor
from darkofit.sklearn_api import _ensemble_bootstrap_plan


def _regression_data(seed=41, n=220):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = 1.4 * X[:, 0] - 0.6 * X[:, 1] + rng.normal(scale=0.2, size=n)
    return X, y


def _params(**extra):
    params = {
        "iterations": 24,
        "learning_rate": 0.1,
        "depth": 4,
        "early_stopping_rounds": 5,
        "random_state": 17,
    }
    params.update(extra)
    return params


def test_single_member_default_preserves_single_estimator_behavior():
    X, y = _regression_data()
    default = DarkoRegressor(**_params()).fit(X, y)
    explicit = DarkoRegressor(**_params(n_ensembles=1)).fit(X, y)

    assert not hasattr(default, "estimators_")
    assert not hasattr(explicit, "estimators_")
    np.testing.assert_array_equal(default.predict(X), explicit.predict(X))


def test_regression_ensemble_is_deterministic_mean_with_shared_preprocessing():
    X, y = _regression_data()
    left = DarkoRegressor(**_params(n_ensembles=3)).fit(X, y)
    right = DarkoRegressor(**_params(n_ensembles=3)).fit(X, y)

    expected = np.mean(
        np.stack([member.predict(X) for member in left.estimators_]),
        axis=0,
    )
    np.testing.assert_array_equal(left.predict(X), expected)
    np.testing.assert_array_equal(left.predict(X), right.predict(X))
    assert left.ensemble_metadata_ == right.ensemble_metadata_
    assert left.ensemble_metadata_["claim_tier"] == "E"
    assert left.ensemble_metadata_["default_changed"] is False
    assert (
        left.ensemble_metadata_["shared_preprocessing"]
        == "numeric_target_free"
    )
    borders = [
        member.model_.prep_.binner_._borders_flat_
        for member in left.estimators_
    ]
    for candidate in borders[1:]:
        np.testing.assert_array_equal(candidate, borders[0])
    for member in left.ensemble_metadata_["members"]:
        assert member["oob_rows"] > 0
        assert member["validation_source"] == "explicit_eval_set"


def test_regression_ensemble_shap_averages_and_remains_additive():
    X, y = _regression_data()
    model = DarkoRegressor(**_params(n_ensembles=3)).fit(X, y)
    values = model.shap_values(X[:9])

    expected = np.mean(
        np.stack(
            [member.shap_values(X[:9]) for member in model.estimators_],
            axis=0,
        ),
        axis=0,
    )
    np.testing.assert_allclose(values, expected, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        values.sum(axis=1) + model.expected_value_,
        model.predict(X[:9]),
        rtol=0.0,
        atol=2e-13,
    )


def test_group_bootstrap_is_group_disjoint_and_public_fit_records_it():
    X, y = _regression_data(n=240)
    groups = np.repeat(np.arange(40), 6)
    plan = _ensemble_bootstrap_plan(
        len(X),
        19,
        bootstrap="groups",
        groups=groups,
    )
    assert set(groups[plan["sampled"]]).isdisjoint(groups[plan["oob"]])

    model = DarkoRegressor(
        **_params(
            n_ensembles=3,
            ensemble_bootstrap="groups",
        )
    ).fit(X, y, groups=groups)
    assert model.ensemble_metadata_["bootstrap"] == "groups"
    assert all(
        member["group_disjoint"] is True
        and member["oob_groups"] > 0
        for member in model.ensemble_metadata_["members"]
    )


def test_categorical_ensemble_uses_member_local_target_statistics():
    pd = pytest.importorskip("pandas")
    X_num, y = _regression_data(n=180)
    X = pd.DataFrame({
        "value": X_num[:, 0],
        "kind": np.where(X_num[:, 1] > 0, "up", "down"),
    })
    model = DarkoRegressor(**_params(n_ensembles=3)).fit(
        X, y, cat_features=["kind"]
    )

    assert model.ensemble_metadata_["shared_preprocessing"] == "member_local"
    assert (
        model.ensemble_metadata_["shared_preprocessing_fallback_reason"]
        == "categorical_or_ordinal_features"
    )
    expected = np.mean(
        np.stack([member.predict(X) for member in model.estimators_]),
        axis=0,
    )
    np.testing.assert_array_equal(model.predict(X), expected)


@pytest.mark.parametrize("multiclass", [False, True])
def test_classifier_ensemble_soft_votes(multiclass):
    X, y_cont = _regression_data(n=260)
    if multiclass:
        y = np.digitize(y_cont, np.quantile(y_cont, [1 / 3, 2 / 3]))
    else:
        y = (y_cont > np.median(y_cont)).astype(np.int64)
    model = DarkoClassifier(**_params(n_ensembles=3)).fit(X, y)

    expected = np.mean(
        np.stack(
            [member.predict_proba(X) for member in model.estimators_],
            axis=0,
        ),
        axis=0,
    )
    np.testing.assert_array_equal(model.predict_proba(X), expected)
    np.testing.assert_array_equal(
        model.predict(X), model.classes_[np.argmax(expected, axis=1)]
    )


@pytest.mark.parametrize(
    ("estimator", "target"),
    [
        (DarkoRegressor, "regression"),
        (DarkoClassifier, "classification"),
    ],
)
def test_ensemble_safe_roundtrip(tmp_path, estimator, target):
    X, y = _regression_data()
    if target == "classification":
        y = (y > np.median(y)).astype(np.int64)
    model = estimator(**_params(n_ensembles=3)).fit(X, y)
    path = tmp_path / "ensemble.npz"
    model.save_model(path)

    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(str(archive["header"]))
        assert header["archive_kind"] == "darkofit_ensemble"
        assert all(not archive[name].dtype.hasobject for name in archive.files)
    restored = estimator.load_model(path)
    if target == "classification":
        np.testing.assert_array_equal(
            model.predict_proba(X), restored.predict_proba(X)
        )
    else:
        np.testing.assert_array_equal(model.predict(X), restored.predict(X))
    assert restored.ensemble_metadata_ == model.ensemble_metadata_
    assert len(restored.estimators_) == 3


def test_ensemble_load_rejects_contradictory_provenance(tmp_path):
    X, y = _regression_data()
    model = DarkoRegressor(**_params(n_ensembles=2)).fit(X, y)
    source = tmp_path / "source.npz"
    corrupt = tmp_path / "corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    header["metadata"]["aggregation"] = "soft_vote"
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="aggregation"):
        DarkoRegressor.load_model(corrupt)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_ensembles": 0}, "at least 1"),
        ({"n_ensembles": True}, "positive integer"),
        (
            {"n_ensembles": 2, "ensemble_bootstrap": "groups"},
            "requires groups",
        ),
        (
            {"n_ensembles": 2, "ensemble_bootstrap": "unknown"},
            "must be 'rows' or 'groups'",
        ),
    ],
)
def test_ensemble_parameter_validation(kwargs, message):
    X, y = _regression_data()
    with pytest.raises((TypeError, ValueError), match=message):
        DarkoRegressor(**_params(**kwargs)).fit(X, y)


def test_ensemble_rejects_ambiguous_eval_and_distributional_aggregation():
    X, y = _regression_data()
    with pytest.raises(ValueError, match="out-of-bag"):
        DarkoRegressor(**_params(n_ensembles=2)).fit(
            X, y, eval_set=(X[:20], y[:20])
        )
    with pytest.raises(ValueError, match="scalar regression"):
        DarkoRegressor(
            **_params(
                n_ensembles=2,
                loss="Gaussian",
                tree_mode="lightgbm",
            )
        ).fit(X, y)
    with pytest.raises(ValueError, match="refit=True"):
        DarkoRegressor(**_params(n_ensembles=2, refit=True)).fit(X, y)


def test_empty_callback_collection_remains_a_noop_for_ensemble():
    X, y = _regression_data()
    model = DarkoRegressor(**_params(n_ensembles=2)).fit(
        X, y, callbacks=()
    )
    assert len(model.estimators_) == 2
