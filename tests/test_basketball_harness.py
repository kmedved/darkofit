from types import SimpleNamespace

import pytest

from benchmarks import basketball_harness as harness


def _core(*, phase=1.0, patience=None):
    return SimpleNamespace(
        training_metadata_={
            "iterations_requested": 100,
            "iterations_attempted": 95,
            "rounds_completed": 95,
            "rounds_retained": 80,
            "stop_reason": "early_stopping",
        },
        timing_={"tree_build": phase},
        trees_=[None] * 80,
        tree_mode_="catboost",
        n_threads_=8,
        early_stopping_rounds_=patience,
    )


def _model(*, phase=1.0, selection=True):
    selected = _core(phase=phase, patience=50) if selection else None
    final = _core(phase=phase, patience=None)
    return SimpleNamespace(
        model_=final,
        selection_model_=selected,
        linear_residual_active_=False,
        best_n_estimators_=80,
        n_estimators_=80,
        learning_rate_=0.1,
        tree_mode="catboost",
        refit_=selection,
        refit_strategy_="exact" if selection else None,
    )


def test_extract_fit_metadata_records_route_stops_and_patience():
    metadata = harness.extract_fit_metadata(_model())

    assert metadata["best_iteration"] == 80
    assert metadata["fitted_tree_count"] == 80
    assert metadata["resolved_learning_rate"] == 0.1
    assert metadata["selected_tree_mode"] == "catboost"
    assert metadata["selected_lane"] == "boosting"
    assert metadata["selection_fit"]["stop_reason"] == "early_stopping"
    assert metadata["final_fit"]["rounds_retained"] == 80
    assert metadata["selection_early_stopping_rounds"] == 50
    assert metadata["final_early_stopping_rounds"] is None


def test_reciprocal_schedule_and_timing_summary_are_fail_closed():
    assert harness.reciprocal_schedule("default", "candidate") == (
        ("default", "candidate"),
        ("candidate", "default"),
        ("default", "candidate"),
    )
    summary = harness.timing_summary([10.0, 11.0, 12.0])
    assert summary["median_seconds"] == 11.0
    assert summary["maximum_over_minimum"] == 1.2
    assert summary["stable"] is True

    with pytest.raises(ValueError, match="distinct"):
        harness.reciprocal_schedule("same", "same")
    with pytest.raises(RuntimeError, match="exactly 3"):
        harness.timing_summary([10.0, 11.0])
    with pytest.raises(RuntimeError, match="positive and finite"):
        harness.timing_summary([10.0, 0.0, 11.0])


def test_behavior_fingerprint_ignores_timing_only():
    first = {
        "prediction_sha256": "abc",
        "fit_seconds": 1.0,
        "fit_metadata": harness.extract_fit_metadata(_model(phase=1.0)),
        "timing_summary": harness.timing_summary([1.0, 1.1, 1.2]),
    }
    second = {
        "prediction_sha256": "abc",
        "fit_seconds": 99.0,
        "fit_metadata": harness.extract_fit_metadata(_model(phase=88.0)),
        "timing_summary": harness.timing_summary([7.0, 9.0, 12.0]),
    }
    changed = {**second, "prediction_sha256": "changed"}

    assert harness.behavior_fingerprint(first) == harness.behavior_fingerprint(
        second
    )
    assert harness.behavior_fingerprint(first) != harness.behavior_fingerprint(
        changed
    )


def test_validate_prediction_and_phase_totals():
    prediction = harness.validate_prediction([1.0, 2.0], 2)
    assert prediction.tolist() == [1.0, 2.0]
    with pytest.raises(RuntimeError, match="shape"):
        harness.validate_prediction([[1.0, 2.0]], 2)
    with pytest.raises(RuntimeError, match="non-finite"):
        harness.validate_prediction([1.0, float("nan")], 2)

    records = [
        {"fit_metadata": harness.extract_fit_metadata(_model(phase=1.5))},
        {"fit_metadata": harness.extract_fit_metadata(_model(phase=2.0))},
    ]
    assert harness.sum_phase_times(records, "final_fit") == {
        "tree_build": 3.5
    }


def test_worker_environment_is_explicit_and_isolated(monkeypatch):
    monkeypatch.setenv("NUMBA_CPU_NAME", "stale")
    monkeypatch.setenv("OMP_NUM_THREADS", "99")
    monkeypatch.setenv("ENABLE_IPC", "0")

    environment = harness.worker_environment(6)

    assert environment["NUMBA_NUM_THREADS"] == "6"
    assert environment["OMP_NUM_THREADS"] == "6"
    assert environment["ENABLE_IPC"] == "1"
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["PYTHONPATH"] == str(harness.REPO_ROOT)
    assert "NUMBA_CPU_NAME" not in environment
