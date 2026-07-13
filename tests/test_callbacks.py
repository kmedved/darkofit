from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

import darkofit.callbacks as callbacks_module
from darkofit import BoostingProgress as PublicBoostingProgress
from darkofit import WallClockStopper as PublicWallClockStopper
from darkofit.booster import (
    DistributionalBoosting,
    GradientBoosting,
    MulticlassBoosting,
)
from darkofit.callbacks import BoostingProgress, WallClockStopper
from darkofit.sklearn_api import DarkoClassifier, DarkoRegressor


def _regression_data(seed=0, n=120):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = 1.7 * X[:, 0] - 0.6 * X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


def _mutated_model_header(src, dst, mutate):
    with np.load(src, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    header = json.loads(str(arrays["header"]))
    mutate(header)
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(dst, **arrays)
    return dst


def _scalar_params(**overrides):
    params = {
        "iterations": 8,
        "learning_rate": 0.1,
        "depth": 2,
        "min_child_samples": 2,
        "thread_count": 1,
        "random_state": 0,
        "diagnostic_warnings": "never",
    }
    params.update(overrides)
    return params


def test_boosting_progress_is_frozen():
    assert PublicBoostingProgress is BoostingProgress
    assert PublicWallClockStopper is WallClockStopper
    progress = BoostingProgress(1, 1, 1, None, 0.5)
    with pytest.raises(FrozenInstanceError):
        progress.rounds_completed = 2


def test_wall_clock_stopper_uses_monotonic_budget_and_margin(monkeypatch):
    readings = iter([100.0, 103.0, 104.0, 105.0])
    monkeypatch.setattr(
        callbacks_module.time, "monotonic", lambda: next(readings)
    )
    stopper = WallClockStopper(5.0, safety_margin=1.0)
    progress = BoostingProgress(0, 0, 0, None, None)

    assert stopper.seconds == 5.0
    assert stopper.safety_margin == 1.0
    assert stopper.effective_seconds == 4.0
    assert stopper(progress) is False
    assert stopper.deadline_hit is False
    assert stopper(progress) is True
    assert stopper.deadline_hit is True
    assert stopper.elapsed_seconds == 5.0
    assert stopper.stop_reason == "time_limit"


@pytest.mark.parametrize(
    "seconds, margin",
    [(-1.0, 0.0), (np.inf, 0.0), (1.0, -1.0), (1.0, np.nan)],
)
def test_wall_clock_stopper_rejects_invalid_limits(seconds, margin):
    with pytest.raises(ValueError):
        WallClockStopper(seconds, safety_margin=margin)


def test_scalar_callback_observes_prior_round_state_and_can_stop():
    X, y = _regression_data()
    observed = []

    class StopAfterTwo:
        stop_reason = "test_limit"

        def __call__(self, progress):
            observed.append(progress)
            return progress.rounds_completed >= 2

    model = GradientBoosting(
        **_scalar_params(iterations=20, eval_train_loss=True, use_best_model=False)
    ).fit(X, y, eval_set=(X, y), callbacks=StopAfterTwo())

    assert [(p.next_iteration, p.iterations_attempted, p.rounds_completed)
            for p in observed] == [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    assert observed[0].last_train_score is None
    assert observed[0].last_validation_score is None
    assert observed[-1].last_train_score == model.train_history_[-1]
    assert observed[-1].last_validation_score == model.valid_history_[-1]
    assert model.stop_reason_ == "test_limit"
    assert model.iterations_attempted_ == 2
    assert model.rounds_completed_ == 2
    assert model.best_iteration_ == 2
    assert model.training_metadata_ == model.auto_params_["training"]
    assert model.training_metadata_ is not model.auto_params_["training"]


def test_only_literal_true_stops_and_callback_collection_is_snapshotted():
    X, y = _regression_data()
    calls = []

    def truthy_integer(progress):
        calls.append(progress.next_iteration)
        return 1

    model = GradientBoosting(**_scalar_params(iterations=3)).fit(
        X, y, callbacks=(callback for callback in [truthy_integer])
    )

    assert calls == [0, 1, 2]
    assert model.stop_reason_ == "iteration_limit"
    assert model.iterations_attempted_ == 3
    assert model.best_iteration_ == 3


def test_callback_absence_and_non_stopping_callback_are_bitwise_equivalent():
    X, y = _regression_data(seed=3)
    plain = GradientBoosting(**_scalar_params(iterations=7)).fit(X, y)
    observed = GradientBoosting(**_scalar_params(iterations=7)).fit(
        X, y, callbacks=lambda progress: False
    )

    assert np.array_equal(plain.predict_raw(X), observed.predict_raw(X))
    assert plain.train_history_ == observed.train_history_
    assert plain.valid_history_ == observed.valid_history_
    assert plain.stop_reason_ == observed.stop_reason_ == "iteration_limit"
    assert plain.training_metadata_["stop_check_policy"] == "none"
    assert observed.training_metadata_["stop_check_policy"] == "before_iteration"


def test_immediate_wall_clock_stop_allows_zero_round_scalar_fit():
    X, y = _regression_data()
    stopper = WallClockStopper(0.0)
    model = GradientBoosting(**_scalar_params(iterations=20)).fit(
        X, y, callbacks=stopper
    )

    assert len(model.trees_) == 0
    assert np.all(model.predict_raw(X) == model.init_)
    assert model.stop_reason_ == "time_limit"
    assert model.iterations_attempted_ == 0
    assert model.rounds_completed_ == 0
    assert model.best_iteration_ == 0
    assert model.training_metadata_["time_limit_is_soft"] is True
    assert stopper.deadline_hit is True


def test_no_split_and_early_stopping_record_causal_termination_metadata():
    constant = GradientBoosting(**_scalar_params(iterations=20)).fit(
        np.zeros((40, 2)), np.zeros(40)
    )
    assert constant.stop_reason_ == "no_split"
    assert constant.iterations_attempted_ == 1
    assert constant.rounds_completed_ == constant.best_iteration_ == 0

    rng = np.random.default_rng(11)
    X = rng.normal(size=(300, 4))
    y = 2.0 * X[:, 0] - X[:, 1]
    Xv = rng.normal(size=(150, 4))
    yv = -2.0 * Xv[:, 0] + Xv[:, 1]
    early = GradientBoosting(
        **_scalar_params(
            iterations=30,
            depth=3,
            early_stopping_rounds=1,
            use_best_model=True,
        )
    ).fit(X, y, eval_set=(Xv, yv))

    assert early.stop_reason_ == "early_stopping"
    assert early.rounds_completed_ == 2
    assert early.best_iteration_ == 1
    assert early.training_metadata_["best_prefix_round"] == 1
    assert early.training_metadata_["best_model_truncated"] is True


def test_multiclass_and_stateful_distributional_fits_can_stop_before_round_zero():
    X, y = _regression_data(seed=5, n=100)
    labels = np.argmax(
        np.column_stack((X[:, 0], X[:, 1], -X[:, 0] - X[:, 1])), axis=1
    )
    multiclass = MulticlassBoosting(**_scalar_params(iterations=10)).fit(
        X, labels, callbacks=WallClockStopper(0.0)
    )
    assert multiclass.stop_reason_ == "time_limit"
    assert multiclass.iterations_attempted_ == 0
    assert multiclass.rounds_completed_ == multiclass.best_iteration_ == 0

    counts = np.random.default_rng(7).poisson(np.exp(0.2 * X[:, 0]))
    distributional = DistributionalBoosting(
        loss="NegativeBinomial",
        tree_mode="lightgbm",
        iterations=10,
        learning_rate=0.1,
        num_leaves=5,
        min_child_samples=2,
        thread_count=1,
        random_state=0,
        diagnostic_warnings="never",
    ).fit(X, counts, callbacks=WallClockStopper(0.0))

    assert distributional.stop_reason_ == "time_limit"
    assert distributional.iterations_attempted_ == 0
    assert distributional.rounds_completed_ == distributional.best_iteration_ == 0
    assert distributional.loss_.state_["r_path"][-1]["source"] == "final_refresh"


def test_callback_validation_happens_before_fit_work():
    X, y = _regression_data()
    with pytest.raises(TypeError, match=r"callbacks\[1\] must be callable"):
        GradientBoosting(**_scalar_params()).fit(
            X, y, callbacks=[lambda progress: False, object()]
        )


@pytest.mark.parametrize("iterations", [-1, -0.9, 1.5, True])
def test_invalid_iterations_are_rejected_without_truncation(iterations):
    X, y = _regression_data()
    with pytest.raises((TypeError, ValueError), match="iterations must be"):
        GradientBoosting(**_scalar_params(iterations=iterations)).fit(X, y)


def test_sklearn_wrapper_forwards_single_fit_callback():
    X, y = _regression_data(seed=13)

    def stop_after_three(progress):
        return progress.rounds_completed >= 3

    model = DarkoRegressor(
        iterations=20,
        learning_rate=0.1,
        depth=2,
        min_child_samples=2,
        thread_count=1,
        random_state=0,
        diagnostic_warnings="never",
    ).fit(X, y, callbacks=stop_after_three)

    assert model.best_n_estimators_ == 3
    assert model.n_estimators_ == 3
    assert model.model_.stop_reason_ == "callback"
    assert model.model_.rounds_completed_ == 3


@pytest.mark.parametrize(
    "estimator",
    [
        DarkoRegressor(refit=True),
        DarkoRegressor(auto_learning_rate_probe=True),
        DarkoClassifier(refit=True),
        DarkoClassifier(auto_learning_rate_probe=True),
    ],
)
def test_sklearn_wrapper_rejects_callbacks_for_multi_fit_policies(estimator):
    X, y = _regression_data(seed=17)
    if isinstance(estimator, DarkoClassifier):
        y = (y > np.median(y)).astype(int)
    with pytest.raises(ValueError, match="callbacks are not supported"):
        estimator.fit(X, y, callbacks=lambda progress: False)


def test_tree_mode_auto_shares_wall_clock_stopper_and_audits_candidates():
    X, y = _regression_data(seed=18)
    stopper = WallClockStopper(0.0)
    model = DarkoRegressor(
        iterations=5,
        learning_rate=0.1,
        tree_mode="auto",
        depth=2,
        min_child_samples=2,
        thread_count=1,
        random_state=0,
        diagnostic_warnings="never",
    ).fit(X[:90], y[:90], eval_set=(X[90:], y[90:]), callbacks=stopper)

    selection = model.tree_mode_selection_
    candidates = selection["candidates"]
    assert [candidate["tree_mode"] for candidate in candidates] == [
        "catboost", "lightgbm", "hybrid",
    ]
    assert sum(candidate["selected"] for candidate in candidates) == 1
    selected_candidate = candidates[selection["selected_candidate_index"]]
    assert selected_candidate["selected"] is True
    assert selection["selected_tree_mode"] == model.model_.tree_mode_
    assert selection["selected_lane"] == "boosting"
    assert selection["candidate_count"] == 3
    assert selection["fitted_candidate_count"] == 1
    assert selection["skipped_deadline_candidate_count"] == 2
    assert selection["candidate_fit_status_counts"] == {
        "fitted": 1,
        "skipped_deadline": 2,
    }
    assert selection["wall_clock_stopper_count"] == 1
    assert selection["deadline_hit"] is True
    assert type(selection["wall_clock_elapsed_seconds"]) is float

    elapsed = []
    fitted, *skipped = candidates
    assert fitted["fit_status"] == "fitted"
    assert fitted["iterations_requested"] == 5
    assert fitted["iterations_attempted"] == 0
    assert fitted["rounds_completed"] == 0
    assert fitted["rounds_retained"] == 0
    assert fitted["best_iteration"] == 0
    assert fitted["best_prefix_round"] is None
    assert fitted["stop_reason"] == "time_limit"
    assert fitted["resolved_learning_rate"] == 0.1
    assert np.isfinite(fitted["validation_score"])

    for candidate in skipped:
        assert candidate["fit_status"] == "skipped_deadline"
        assert candidate["iterations_requested"] == 5
        assert candidate["iterations_attempted"] == 0
        assert candidate["rounds_completed"] == 0
        assert candidate["rounds_retained"] == 0
        assert candidate["best_iteration"] is None
        assert candidate["best_prefix_round"] is None
        assert candidate["stop_reason"] == "time_limit"
        assert candidate["resolved_learning_rate"] is None
        assert candidate["validation_score"] is None
        assert candidate["probe"] == {
            "enabled": False,
            "reason": "skipped_deadline",
        }
        assert candidate["lane"] == "boosting"
        assert candidate["wall_clock_stopper_count"] == 1
        assert candidate["deadline_hit_end"] is True
        assert type(candidate["wall_clock_elapsed_seconds_start"]) is float
        assert type(candidate["wall_clock_elapsed_seconds_end"]) is float
        assert (
            candidate["wall_clock_elapsed_seconds"]
            == candidate["wall_clock_elapsed_seconds_end"]
        )
        assert candidate["deadline_hit"] is True
        assert (
            candidate["wall_clock_elapsed_seconds_end"]
            >= candidate["wall_clock_elapsed_seconds_start"]
        )
    for candidate in candidates:
        elapsed.extend([
            candidate["wall_clock_elapsed_seconds_start"],
            candidate["wall_clock_elapsed_seconds_end"],
        ])
    assert elapsed == sorted(elapsed)
    assert model.model_.auto_params_["tree_mode_selection"] == selection


def test_tree_mode_auto_reuses_callback_objects_for_classifier_candidates():
    X, y = _regression_data(seed=19)
    labels = (y > np.median(y)).astype(int)
    observed_first_rounds = []

    class Observer:
        def __call__(self, progress):
            if progress.next_iteration == 0:
                observed_first_rounds.append(progress)
            return False

    observer = Observer()
    model = DarkoClassifier(
        iterations=2,
        learning_rate=0.1,
        tree_mode="auto",
        depth=2,
        min_child_samples=2,
        thread_count=1,
        random_state=0,
        diagnostic_warnings="never",
    ).fit(
        X[:90], labels[:90],
        eval_set=(X[90:], labels[90:]),
        callbacks=(callback for callback in [observer]),
    )

    assert len(observed_first_rounds) == 3
    assert all(
        progress.iterations_attempted == 0
        for progress in observed_first_rounds
    )
    assert all(
        progress.rounds_completed == 0
        for progress in observed_first_rounds
    )
    assert len(model.tree_mode_selection_["candidates"]) == 3
    assert model.tree_mode_selection_["candidate_fit_status_counts"] == {
        "fitted": 3,
        "skipped_deadline": 0,
    }
    assert all(
        candidate["fit_status"] == "fitted"
        and candidate["wall_clock_stopper_count"] == 0
        and candidate["wall_clock_elapsed_seconds_start"] is None
        and candidate["wall_clock_elapsed_seconds_end"] is None
        and candidate["deadline_hit_start"] is False
        and candidate["deadline_hit_end"] is False
        for candidate in model.tree_mode_selection_["candidates"]
    )


@pytest.mark.parametrize(
    "policy",
    [
        {"refit": True},
        {"tree_mode": "auto"},
        {"auto_learning_rate_probe": True},
    ],
)
def test_empty_callback_collection_is_a_noop_for_wrapper_policies(policy):
    X, y = _regression_data(seed=18)
    model = DarkoRegressor(
        iterations=2,
        learning_rate=0.1,
        depth=2,
        min_child_samples=2,
        thread_count=1,
        random_state=0,
        diagnostic_warnings="never",
        **policy,
    ).fit(X, y, callbacks=[])
    assert model.n_estimators_ >= 0


def test_training_metadata_roundtrips_without_serializing_callback(tmp_path):
    X, y = _regression_data(seed=19)

    def stop_after_two(progress):
        return progress.rounds_completed >= 2

    model = GradientBoosting(**_scalar_params(iterations=10)).fit(
        X, y, callbacks=stop_after_two
    )
    path = tmp_path / "callback_model.npz"
    model.save_model(path)
    loaded = GradientBoosting.load_model(path)

    assert np.array_equal(model.predict_raw(X), loaded.predict_raw(X))
    assert loaded.stop_reason_ == "callback"
    assert loaded.iterations_attempted_ == 2
    assert loaded.rounds_completed_ == 2
    assert loaded.training_metadata_ == model.training_metadata_
    assert loaded.auto_params_["training"] == model.training_metadata_


def test_legacy_archive_without_training_metadata_gets_explicit_fallback(tmp_path):
    X, y = _regression_data(seed=23)
    model = GradientBoosting(**_scalar_params(iterations=3)).fit(X, y)
    model.auto_params_.pop("training")
    path = tmp_path / "legacy_model.npz"
    model.save_model(path)
    loaded = GradientBoosting.load_model(path)

    assert np.array_equal(model.predict_raw(X), loaded.predict_raw(X))
    assert loaded.stop_reason_ == "legacy_unknown"
    assert loaded.iterations_attempted_ is None
    assert loaded.rounds_completed_ == 3
    assert loaded.training_metadata_["rounds_retained"] == 3


def test_distributional_archive_rejects_nonobject_auto_params(tmp_path):
    X, y = _regression_data(seed=29)
    model = DistributionalBoosting(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=2,
        learning_rate=0.1,
        num_leaves=5,
        min_child_samples=2,
        thread_count=1,
        random_state=0,
        diagnostic_warnings="never",
    ).fit(X, y)
    path = tmp_path / "distributional.npz"
    model.save_model(path)
    bad = _mutated_model_header(
        path,
        tmp_path / "bad_auto_params.npz",
        lambda header: header.update(auto_params=[]),
    )

    with pytest.raises(ValueError, match="auto_params must be an object"):
        DistributionalBoosting.load_model(bad)


def test_archive_rejects_training_horizon_mismatch(tmp_path):
    X, y = _regression_data(seed=31)
    model = GradientBoosting(**_scalar_params(iterations=3)).fit(X, y)
    path = tmp_path / "horizon.npz"
    model.save_model(path)
    bad = _mutated_model_header(
        path,
        tmp_path / "bad_horizon.npz",
        lambda header: header["auto_params"]["training"].update(
            iterations_requested=999
        ),
    )

    with pytest.raises(ValueError, match="requested rounds do not match"):
        GradientBoosting.load_model(bad)


def test_archive_rejects_builtin_stop_reason_causal_inconsistencies(tmp_path):
    X, y = _regression_data(seed=37)

    class StopAfterTwo:
        stop_reason = "custom_budget"

        def __call__(self, progress):
            return progress.rounds_completed >= 2

    model = GradientBoosting(**_scalar_params(iterations=5)).fit(
        X, y, callbacks=StopAfterTwo()
    )
    path = tmp_path / "custom_stop.npz"
    model.save_model(path)
    loaded = GradientBoosting.load_model(path)
    assert loaded.stop_reason_ == "custom_budget"

    def make_iteration_limit_inconsistent(training):
        training["stop_reason"] = "iteration_limit"

    def make_time_limit_inconsistent(training):
        training.update(
            stop_reason="time_limit",
            iterations_attempted=training["iterations_requested"],
            time_limit_is_soft=True,
        )

    def make_no_split_inconsistent(training):
        training["stop_reason"] = "no_split"

    def make_time_limit_policy_inconsistent(training):
        training.update(
            stop_reason="time_limit",
            stop_check_policy="none",
            time_limit_is_soft=True,
        )

    cases = (
        (
            make_iteration_limit_inconsistent,
            "iteration_limit requires all requested rounds",
        ),
        (make_time_limit_inconsistent, "time_limit requires stopping before"),
        (
            make_time_limit_policy_inconsistent,
            "time_limit requires before_iteration stop checks",
        ),
        (make_no_split_inconsistent, "no_split requires a failed attempted round"),
    )
    for index, (mutate, match) in enumerate(cases):
        bad = _mutated_model_header(
            path,
            tmp_path / f"bad_reason_{index}.npz",
            lambda header, mutate=mutate: mutate(
                header["auto_params"]["training"]
            ),
        )
        with pytest.raises(ValueError, match=match):
            GradientBoosting.load_model(bad)

    zero = GradientBoosting(**_scalar_params(iterations=5)).fit(
        X, y, callbacks=WallClockStopper(0.0)
    )
    zero_path = tmp_path / "zero_round.npz"
    zero.save_model(zero_path)
    loaded_zero = GradientBoosting.load_model(zero_path)
    assert loaded_zero.stop_reason_ == "time_limit"

    def make_early_stopping_inconsistent(header):
        training = header["auto_params"]["training"]
        training.update(
            stop_reason="early_stopping",
            time_limit_is_soft=False,
        )

    bad_early = _mutated_model_header(
        zero_path,
        tmp_path / "bad_early_stopping.npz",
        make_early_stopping_inconsistent,
    )
    with pytest.raises(ValueError, match="early_stopping requires a completed round"):
        GradientBoosting.load_model(bad_early)
