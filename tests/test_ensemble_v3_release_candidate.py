import json

import numba
import numpy as np
import pytest
from sklearn.base import clone

import darkofit.sklearn_api as sklearn_api
from darkofit import DarkoClassifier, DarkoRegressor
from darkofit.sklearn_api import (
    _fit_ensemble_v3_release_candidate,
    _normalize_ensemble_v3_release_candidate_overrides,
)


def _data(seed=20260721, n=120):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    signal = 1.4 * X[:, 0] - 0.5 * X[:, 1] + 0.2 * X[:, 2]
    y = signal + rng.normal(scale=0.25, size=n)
    return X, y


def _params(**extra):
    params = {
        "iterations": 4,
        "depth": 3,
        "early_stopping_rounds": 2,
        "random_state": 41,
        "n_ensembles": 8,
        "diagnostic_warnings": "never",
    }
    params.update(extra)
    return params


def _fit(estimator, X, y, **kwargs):
    return _fit_ensemble_v3_release_candidate(
        estimator,
        X,
        y,
        **kwargs,
    )


def _rewrite_header(path, output, mutate):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    mutate(header)
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(output, **arrays)


def test_release_candidate_override_sentinel_and_values_are_unambiguous():
    future, explicit = _normalize_ensemble_v3_release_candidate_overrides()
    assert future == {"learning_rate": "policy", "colsample": "policy"}
    assert explicit == {}

    future, explicit = _normalize_ensemble_v3_release_candidate_overrides(
        None,
        1.0,
    )
    assert future == {"learning_rate": None, "colsample": 1.0}
    assert explicit == {"learning_rate": None, "colsample": 1.0}

    for bad in (True, 0.0, np.inf, "auto", np.array([0.2])):
        with pytest.raises((TypeError, ValueError)):
            _normalize_ensemble_v3_release_candidate_overrides(bad, "policy")
    for bad in (False, 0.0, 1.01, np.nan, None, "auto"):
        with pytest.raises((TypeError, ValueError)):
            _normalize_ensemble_v3_release_candidate_overrides("policy", bad)


def test_release_candidate_requires_the_only_evaluated_member_count():
    X, y = _data(n=80)
    for count in (1, 2, 7, 9):
        with pytest.raises(ValueError, match="requires n_ensembles=8"):
            _fit(DarkoRegressor(**_params(n_ensembles=count)), X, y)


def test_release_candidate_is_deterministic_and_records_distinct_contract():
    X, y = _data()
    left = _fit(DarkoRegressor(**_params()), X, y)
    right = _fit(DarkoRegressor(**_params()), X, y)

    np.testing.assert_array_equal(left.predict(X), right.predict(X))
    np.testing.assert_array_equal(
        left.predict(X),
        np.mean([member.predict(X) for member in left.estimators_], axis=0),
    )
    assert left.ensemble_metadata_ == right.ensemble_metadata_
    metadata = left.ensemble_metadata_
    assert metadata["version"] == 5
    assert metadata["private_prototype"] == "ensemble_v3_release_candidate"
    assert metadata["recipe_contract"] == "ensemble-v3-public-contract-v1"
    assert metadata["recipe_version"] == 1
    assert metadata["sampling"] == "without_replacement"
    assert metadata["sample_fraction"] == 0.8
    assert metadata["member_policy"] == "donor_balanced_v1"
    assert metadata["future_constructor_params"] == {
        "ensemble_mode": "v3",
        "ensemble_member_learning_rate": "policy",
        "ensemble_member_colsample": "policy",
    }
    assert metadata["explicit_user_params"] == []
    assert len(metadata["members"]) == 8
    assert all(
        record["sampled_rows"] == record["sampled_unique_rows"] == 96
        and record["oob_rows"] == 24
        for record in metadata["members"]
    )
    staged = list(left.staged_predict(X[:12]))
    assert staged
    np.testing.assert_array_equal(staged[-1], left.predict(X[:12]))
    np.testing.assert_array_equal(
        left.shap_values(X[:8]),
        np.mean(
            [member.shap_values(X[:8]) for member in left.estimators_],
            axis=0,
        ),
    )


def test_release_candidate_explicit_overrides_clone_and_safe_roundtrip(tmp_path):
    X, y = _data(n=100)
    base = DarkoRegressor(**_params(learning_rate=0.07, colsample=0.6))
    assert clone(base).get_params(deep=False) == base.get_params(deep=False)
    model = _fit(
        base,
        X,
        y,
        member_learning_rate=None,
        member_colsample=1.0,
    )
    metadata = model.ensemble_metadata_
    assert metadata["base_constructor_params"]["learning_rate"] == 0.07
    assert metadata["base_constructor_params"]["colsample"] == 0.6
    assert metadata["explicit_user_params"] == ["learning_rate", "colsample"]
    assert metadata["policy_resolutions"]["learning_rate"] == {
        "base": 0.07,
        "resolved": None,
        "source": "explicit_user",
    }
    assert metadata["policy_resolutions"]["colsample"] == {
        "base": 0.6,
        "resolved": 1.0,
        "source": "explicit_user",
    }
    assert all(member.learning_rate is None for member in model.estimators_)
    assert all(member.colsample == 1.0 for member in model.estimators_)

    path = tmp_path / "candidate.npz"
    resaved = tmp_path / "candidate-resaved.npz"
    model.save_model(path)
    restored = DarkoRegressor.load_model(path)
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    assert restored.ensemble_metadata_ == metadata
    restored.save_model(resaved)
    assert resaved.read_bytes() == path.read_bytes()


def test_release_candidate_group_sampling_is_disjoint_and_roundtrips(tmp_path):
    group_sizes = np.arange(2, 14)
    groups = np.repeat(np.arange(len(group_sizes)), group_sizes)
    X, y = _data(n=len(groups))
    model = _fit(
        DarkoRegressor(**_params(ensemble_bootstrap="groups")),
        X,
        y,
        groups=groups,
        sample_weight=np.linspace(0.3, 1.7, len(y)),
    )
    assert all(
        record["group_disjoint"] is True
        and record["sampled_unique_groups"] == 10
        and record["oob_groups"] == 2
        for record in model.ensemble_metadata_["members"]
    )
    path = tmp_path / "candidate-groups.npz"
    model.save_model(path)
    restored = DarkoRegressor.load_model(path)
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    np.testing.assert_array_equal(
        restored._ensemble_group_codes_, model._ensemble_group_codes_
    )


def test_release_candidate_classifier_soft_vote_and_thread_state():
    X, continuous = _data(n=150)
    y = np.digitize(continuous, np.quantile(continuous, [1 / 3, 2 / 3]))
    ambient = numba.get_num_threads()
    try:
        model = _fit(
            DarkoClassifier(**_params(thread_count=2)),
            X,
            y,
            sample_weight=np.linspace(0.5, 1.5, len(y)),
        )
        expected = np.mean(
            [member.predict_proba(X) for member in model.estimators_], axis=0
        )
        np.testing.assert_array_equal(model.predict_proba(X), expected)
        assert numba.get_num_threads() == ambient
        list(model.staged_predict_proba(X[:10]))
        assert numba.get_num_threads() == ambient
    finally:
        numba.set_num_threads(ambient)


def test_release_candidate_binary_categorical_and_explicit_ordinal_paths():
    pd = pytest.importorskip("pandas")
    X_num, continuous = _data(n=120)
    levels = np.array(["low", "mid", "high"])
    X = pd.DataFrame({
        "value": X_num[:, 0],
        "kind": np.where(X_num[:, 1] >= 0.0, "up", "down"),
        "tier": levels[np.digitize(X_num[:, 2], [-0.5, 0.5])],
    })
    y = (continuous > np.median(continuous)).astype(np.int64)
    model = _fit(
        DarkoClassifier(**_params()),
        X,
        y,
        cat_features=["kind"],
        ordinal_features={"tier": ("low", "mid", "high")},
    )
    assert model.predict_proba(X).shape == (len(X), 2)
    assert model.ensemble_metadata_["shared_preprocessing"] == "member_local"


@pytest.mark.parametrize(
    ("estimator_kwargs", "fit_kwargs", "message"),
    [
        ({"preset": "accuracy"}, {}, "preset is not supported"),
        ({"tree_mode": "auto"}, {}, "tree_mode='auto'"),
        (
            {"auto_learning_rate_probe": True},
            {},
            "auto_learning_rate_probe=True",
        ),
        ({"refit": True}, {}, "refit=True"),
        ({"loss": "Gaussian"}, {}, "scalar regression losses only"),
        ({}, {"callbacks": [lambda _: False]}, "callbacks are not supported"),
        ({}, {"eval_set": (np.zeros((4, 5)), np.zeros(4))}, "eval_set"),
        ({}, {"ordinal_features": "auto"}, "ordinal_features='auto'"),
        ({}, {"groups": np.arange(80)}, "groups cannot be used"),
    ],
)
def test_release_candidate_unsupported_surfaces_fail_transactionally(
    estimator_kwargs,
    fit_kwargs,
    message,
):
    X, y = _data(n=80)
    estimator = DarkoRegressor(**_params(**estimator_kwargs))
    with pytest.raises(ValueError, match=message):
        _fit(estimator, X, y, **fit_kwargs)
    assert not hasattr(estimator, "model_")
    assert not hasattr(estimator, "estimators_")


def test_release_candidate_member_failure_restores_prior_fit(monkeypatch):
    X, y = _data(n=90)
    estimator = DarkoRegressor(**_params(n_ensembles=1, iterations=2)).fit(X, y)
    expected = estimator.predict(X)
    estimator.set_params(n_ensembles=8)
    original = DarkoRegressor.fit
    calls = 0

    def fail_second(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("candidate member failed")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DarkoRegressor, "fit", fail_second)
    with pytest.raises(RuntimeError, match="candidate member failed"):
        _fit(estimator, X, y)
    np.testing.assert_array_equal(estimator.predict(X), expected)
    assert not hasattr(estimator, "estimators_")


def test_release_candidate_safe_load_rejects_contract_forgery(tmp_path):
    X, y = _data(n=90)
    model = _fit(DarkoRegressor(**_params(iterations=2)), X, y)
    source = tmp_path / "candidate.npz"
    corrupt = tmp_path / "candidate-corrupt.npz"
    model.save_model(source)
    _rewrite_header(
        source,
        corrupt,
        lambda header: header["metadata"].__setitem__("recipe_version", 2),
    )
    with pytest.raises(ValueError, match="release-candidate contract"):
        DarkoRegressor.load_model(corrupt)


def test_release_candidate_helper_remains_private_after_public_ship():
    import darkofit

    assert not hasattr(darkofit, "fit_ensemble_v3")
    assert not hasattr(darkofit, "_fit_ensemble_v3_release_candidate")
    assert {
        "ensemble_mode",
        "ensemble_member_learning_rate",
        "ensemble_member_colsample",
    }.issubset(DarkoRegressor().get_params(deep=False))
    assert {
        "ensemble_mode",
        "ensemble_member_learning_rate",
        "ensemble_member_colsample",
    }.issubset(sklearn_api._SKLEARN_ONLY)
