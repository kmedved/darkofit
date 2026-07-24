import copy

import numba
import numpy as np
import pytest

from darkofit import DarkoClassifier, DarkoRegressor
from darkofit import sklearn_api as sklearn_api_module
from darkofit.sklearn_api import (
    _B3_V2_PARALLEL_MINIMUM_WORK,
    _b3_parallel_dispatch,
    _b3_parallel_work_estimate,
    _fit_public_ensemble_v3_parallel_candidate,
    _resolve_b3_parallel_topology,
)


def _regression_data(seed=4, n=420):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = X[:, 0] - 0.7 * X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


def _regressor(**params):
    defaults = dict(
        iterations=12,
        early_stopping_rounds=4,
        n_ensembles=8,
        ensemble_mode="v3",
        random_state=4,
        thread_count=2,
        diagnostic_warnings="never",
    )
    defaults.update(params)
    return DarkoRegressor(**defaults)


def _fit_parallel(model, X, y, **fit_params):
    model.set_params(ensemble_parallelism="parallel")
    return _fit_public_ensemble_v3_parallel_candidate(
        model,
        X,
        y,
        total_thread_budget=14,
        minimum_work=0,
        **fit_params,
    )


def _member_identity(model):
    return [
        (
            record["member"],
            record["seed"],
            record["sampled_indices_sha256"],
            record["oob_indices_sha256"],
            record["best_iteration"],
        )
        for record in model.ensemble_metadata_["members"]
    ]


@pytest.mark.parametrize(
    "members,budget,expected",
    [
        (8, 14, (7, 2)),
        (8, 1, (1, 1)),
        (8, 2, (1, 2)),
        (8, 8, (4, 2)),
        (3, 5, (2, 2)),
        (1, 14, (1, 14)),
    ],
)
def test_b3_topology_is_deterministic_and_bounded(members, budget, expected):
    assert _resolve_b3_parallel_topology(members, budget) == expected
    workers, threads = expected
    assert 1 <= workers <= members
    assert workers * threads <= budget


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "14"])
def test_b3_topology_rejects_invalid_budgets(value):
    error = TypeError if value in {True, 1.5, "14"} else ValueError
    with pytest.raises(error):
        _resolve_b3_parallel_topology(8, value)


def test_b3_v2_work_estimate_includes_output_width():
    X, y = _regression_data()
    regression = _b3_parallel_work_estimate(_regressor(), X, y)
    labels = np.digitize(y, [-0.5, 0.5])
    classifier = _b3_parallel_work_estimate(
        DarkoClassifier(iterations=12), X, labels
    )
    binary = _b3_parallel_work_estimate(
        DarkoClassifier(iterations=12), X, labels > 0
    )

    assert regression == {
        "version": 1,
        "input_rows": 420,
        "sampled_rows": 336,
        "active_features": 6,
        "planned_iterations": 12,
        "output_width": 1,
        "member_work": 24_192,
    }
    assert classifier["output_width"] == 3
    assert classifier["member_work"] == 3 * regression["member_work"]
    assert binary["output_width"] == 1
    assert binary["member_work"] == regression["member_work"]


def test_b3_v2_auto_dispatch_is_hardware_scoped_and_rollback_capable(
    monkeypatch,
):
    X, y = _regression_data()
    estimator = _regressor(thread_count=14)
    monkeypatch.setattr(sklearn_api_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sklearn_api_module.platform, "machine", lambda: "arm64")
    measured = _b3_parallel_dispatch(
        estimator,
        X,
        y,
        total_thread_budget=14,
        minimum_work=0,
        requested="auto",
    )
    assert measured["route"] == "process_parallel"
    assert measured["reason"] == "at_or_above_minimum_work"

    monkeypatch.setattr(sklearn_api_module.platform, "machine", lambda: "x86_64")
    outside = _b3_parallel_dispatch(
        estimator,
        X,
        y,
        total_thread_budget=14,
        minimum_work=0,
        requested="auto",
    )
    assert outside["route"] == "sequential_fallback"
    assert outside["reason"] == "outside_measured_envelope"
    rollback = _b3_parallel_dispatch(
        estimator,
        X,
        y,
        total_thread_budget=14,
        minimum_work=0,
        requested="sequential",
    )
    assert rollback["route"] == "sequential_fallback"
    assert rollback["reason"] == "user_forced_sequential"


def test_b3_v2_below_threshold_is_exact_sequential_fallback(tmp_path):
    X, y = _regression_data(seed=3)
    sequential = _regressor().fit(X, y)
    candidate = _regressor()
    _fit_public_ensemble_v3_parallel_candidate(
        candidate,
        X,
        y,
        total_thread_budget=14,
        minimum_work=_B3_V2_PARALLEL_MINIMUM_WORK,
    )

    assert candidate.ensemble_parallel_dispatch_["route"] == (
        "sequential_fallback"
    )
    assert candidate.ensemble_parallel_dispatch_["reason"] == (
        "below_minimum_work"
    )
    candidate_metadata = dict(candidate.ensemble_metadata_)
    sequential_metadata = dict(sequential.ensemble_metadata_)
    candidate_metadata.pop("parallel_dispatch")
    sequential_metadata.pop("parallel_dispatch")
    assert candidate_metadata == sequential_metadata
    assert np.array_equal(candidate.predict(X), sequential.predict(X))
    sequential_path = tmp_path / "sequential.npz"
    candidate_path = tmp_path / "candidate.npz"
    sequential.save_model(sequential_path)
    candidate.save_model(candidate_path)
    with np.load(sequential_path, allow_pickle=False) as sequential_archive:
        with np.load(candidate_path, allow_pickle=False) as candidate_archive:
            assert set(sequential_archive.files) == set(candidate_archive.files)
            for name in sequential_archive.files:
                if name == "header":
                    continue
                assert np.array_equal(
                    sequential_archive[name], candidate_archive[name]
                )


@pytest.mark.parametrize("value", [True, -1, 1.5, "80000000"])
def test_b3_v2_rejects_invalid_minimum_work(value):
    X, y = _regression_data(n=80)
    error = ValueError if value == -1 else TypeError
    with pytest.raises(error, match="minimum_work"):
        _fit_public_ensemble_v3_parallel_candidate(
            _regressor(),
            X,
            y,
            total_thread_budget=14,
            minimum_work=value,
        )


def test_b3_parallel_matches_same_thread_sequential_and_restores_ambient():
    X, y = _regression_data()
    ambient = numba.get_num_threads()
    sequential = _regressor().fit(X, y)
    candidate = _regressor()

    _fit_parallel(candidate, X, y)

    assert numba.get_num_threads() == ambient
    assert np.array_equal(sequential.predict(X), candidate.predict(X))
    assert _member_identity(sequential) == _member_identity(candidate)
    assert candidate.ensemble_metadata_["private_b3_schedule"] == {
        "contract": "b3-parallel-ensemble-members-v1-20260723",
        "mode": "private_process_workers",
        "workers": 7,
        "member_threads": 2,
        "total_thread_budget": 14,
        "maximum_model_threads": 14,
        "result_order": "member_index",
    }
    assert candidate.ensemble_metadata_["sequential"] is False
    assert candidate.ensemble_parallel_dispatch_["route"] == "process_parallel"
    assert candidate.ensemble_parallel_dispatch_["minimum_work"] == 0
    assert all(member.model_.n_threads_ == 2 for member in candidate.estimators_)
    assert all(
        record["prediction_thread_count"] == 2
        for record in candidate.ensemble_metadata_["members"]
    )
    assert "private_b3_schedule" not in sequential.ensemble_metadata_
    assert sequential.ensemble_metadata_["sequential"] is True


def test_b3_grouped_weighted_sampling_and_round_trip(tmp_path):
    X, y = _regression_data(seed=7, n=480)
    groups = np.repeat(np.arange(120), 4)
    weights = np.linspace(0.0, 2.0, len(y))
    candidate = _regressor(ensemble_bootstrap="groups")
    ambient = numba.get_num_threads()

    _fit_parallel(
        candidate,
        X,
        y,
        groups=groups,
        sample_weight=weights,
    )

    assert numba.get_num_threads() == ambient
    assert all(
        record["group_disjoint"] is True
        for record in candidate.ensemble_metadata_["members"]
    )
    first = tmp_path / "candidate.npz"
    second = tmp_path / "resaved.npz"
    candidate.save_model(first)
    loaded = DarkoRegressor.load_model(first)
    loaded.save_model(second)
    assert np.array_equal(candidate.predict(X), loaded.predict(X))
    assert first.read_bytes() == second.read_bytes()
    assert (
        loaded.ensemble_metadata_["private_b3_schedule"]
        == candidate.ensemble_metadata_["private_b3_schedule"]
    )


def test_b3_categorical_predictions_match_same_thread_control():
    rng = np.random.default_rng(9)
    n = 420
    X = np.empty((n, 4), dtype=object)
    X[:, 0] = rng.choice(["a", "b", "c", None], size=n)
    X[:, 1:] = rng.normal(size=(n, 3))
    y = (
        np.asarray(X[:, 1], dtype=float)
        + 0.5 * (X[:, 0] == "b")
        + 0.1 * rng.normal(size=n)
    )
    sequential = _regressor().fit(X, y, cat_features=[0])
    candidate = _regressor()

    _fit_parallel(candidate, X, y, cat_features=[0])

    assert np.array_equal(sequential.predict(X), candidate.predict(X))
    assert _member_identity(sequential) == _member_identity(candidate)


@pytest.mark.parametrize("multiclass", [False, True])
def test_b3_classifier_probabilities_match_same_thread_control(multiclass):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(450, 5))
    score = X[:, 0] - X[:, 1] + 0.25 * X[:, 2]
    y = (
        np.digitize(score, [-0.5, 0.5])
        if multiclass
        else (score > 0.0).astype(np.int64)
    )
    params = dict(
        iterations=12,
        early_stopping_rounds=4,
        n_ensembles=8,
        ensemble_mode="v3",
        random_state=4,
        thread_count=2,
        diagnostic_warnings="never",
    )
    sequential = DarkoClassifier(**params).fit(X, y)
    candidate = DarkoClassifier(**params)

    _fit_parallel(candidate, X, y)

    assert np.array_equal(
        sequential.predict_proba(X), candidate.predict_proba(X)
    )
    assert np.array_equal(sequential.classes_, candidate.classes_)
    assert _member_identity(sequential) == _member_identity(candidate)


def test_b3_failure_restores_previously_fitted_estimator():
    X, y = _regression_data(seed=15)
    candidate = _regressor().fit(X, y)
    before = candidate.predict(X)
    state = copy.deepcopy(candidate.ensemble_metadata_)
    ambient = numba.get_num_threads()

    with pytest.raises(ValueError, match="groups cannot be used"):
        _fit_parallel(
            candidate,
            X,
            y,
            groups=np.arange(3),
        )

    assert numba.get_num_threads() == ambient
    assert np.array_equal(candidate.predict(X), before)
    assert candidate.ensemble_metadata_ == state


def test_b3_worker_failure_propagates_and_restores_state(monkeypatch):
    X, y = _regression_data(seed=22)
    candidate = _regressor().fit(X, y)
    before = candidate.predict(X)
    state = copy.deepcopy(candidate.ensemble_metadata_)
    original = sklearn_api_module._fit_private_ensemble_v3_member

    def fail_one_worker(payload):
        if payload["member_index"] == 3:
            raise RuntimeError("injected B3 worker failure")
        return original(payload)

    monkeypatch.setattr(
        sklearn_api_module,
        "_fit_private_ensemble_v3_member",
        fail_one_worker,
    )
    ambient = numba.get_num_threads()

    with pytest.raises(RuntimeError, match="injected B3 worker failure"):
        _fit_parallel(candidate, X, y)

    assert numba.get_num_threads() == ambient
    assert np.array_equal(candidate.predict(X), before)
    assert candidate.ensemble_metadata_ == state
