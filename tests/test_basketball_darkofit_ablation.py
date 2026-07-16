import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmarks import run_basketball_darkofit_ablation as ablation


def test_frozen_config_matrix_keeps_levers_separate():
    assert ablation.CONFIG_ORDER == (
        "default",
        "a10_numeric",
        "a10_numeric_2000",
        "a10_early_stopping_refit",
        "linear_residual",
    )
    assert ablation.CONFIGS["default"] == {}
    assert ablation.CONFIGS["a10_numeric"] == {
        "l2_leaf_reg": 1.0,
        "max_bins": 128,
        "learning_rate": 0.1,
    }
    assert ablation.CONFIGS["a10_numeric_2000"] == {
        "iterations": 2_000,
        **ablation.CONFIGS["a10_numeric"],
    }
    assert ablation.CONFIGS["linear_residual"] == {"linear_residual": True}
    refit = ablation.CONFIGS["a10_early_stopping_refit"]
    assert refit["iterations"] == 2_000
    assert refit["early_stopping"] is True
    assert refit["early_stopping_rounds"] == "auto"
    assert refit["validation_fraction"] == 0.1
    assert refit["use_best_model"] is True
    assert refit["refit"] is True
    assert refit["refit_strategy"] == "exact"


def test_prepare_held_team_data_uses_alphabetical_first_third():
    rows = []
    for index, team in enumerate(("D", "B", "F", "A", "E", "C")):
        row = {feature: float(index) for feature in ablation.baseline.FEATURES}
        row.update({"MP": 600.0, "G": 60.0, "GS": 30.0, "Tm": team})
        rows.append(row)
    frame = pd.DataFrame(rows)

    X_train, y_train, X_test, y_test, metadata = (
        ablation.prepare_held_team_data(frame)
    )

    assert metadata["test_teams"] == ["A", "B"]
    assert metadata["train_team_count"] == 4
    assert metadata["test_team_count"] == 2
    assert len(X_train) == len(y_train) == 4
    assert len(X_test) == len(y_test) == 2
    assert y_train.tolist() == [10.0] * 4
    assert y_test.tolist() == [10.0] * 2


def test_extract_fit_metadata_records_final_and_selection_fits():
    final = SimpleNamespace(
        training_metadata_={
            "iterations_requested": 17,
            "iterations_attempted": 17,
            "rounds_completed": 17,
            "rounds_retained": 17,
            "stop_reason": "iteration_limit",
        },
        trees_=[object()] * 17,
        timing_={"preprocess": 1.0, "tree_build": 2.0},
        tree_mode_="catboost",
        n_threads_=8,
    )
    selection = SimpleNamespace(
        training_metadata_={
            "iterations_requested": 2_000,
            "iterations_attempted": 42,
            "rounds_completed": 42,
            "rounds_retained": 17,
            "stop_reason": "early_stopping",
        },
        trees_=[object()] * 17,
        timing_={"validation_predict": 0.5},
    )
    model = SimpleNamespace(
        model_=final,
        selection_model_=selection,
        best_n_estimators_=17,
        n_estimators_=17,
        learning_rate_=0.1,
        tree_mode="catboost",
        linear_residual_active_=False,
        refit_=True,
        refit_strategy_="exact",
    )

    metadata = ablation.extract_fit_metadata(model)

    assert metadata["best_iteration"] == 17
    assert metadata["fitted_tree_count"] == 17
    assert metadata["resolved_learning_rate"] == 0.1
    assert metadata["selected_lane"] == "boosting"
    assert metadata["final_fit"]["stop_reason"] == "iteration_limit"
    assert metadata["selection_fit"]["stop_reason"] == "early_stopping"
    assert metadata["selection_fit"]["iterations_attempted"] == 42


def _synthetic_result(
    name,
    scores,
    *,
    held_r2=0.4,
    runtime=20.0,
):
    return {
        "config": name,
        "fold_scores": list(scores),
        "mean_r2": float(np.mean(scores)),
        "held_team": {"r2": held_r2},
        "steady_wall_seconds": runtime,
    }


def test_analyze_results_requires_every_gate():
    default_scores = np.full(10, 0.50)
    results = [
        _synthetic_result("default", default_scores, held_r2=0.40, runtime=25.0),
        _synthetic_result(
            "a10_numeric",
            default_scores + 0.003,
            held_r2=0.401,
            runtime=20.0,
        ),
        _synthetic_result(
            "a10_numeric_2000",
            np.r_[default_scores[:9], 0.53],
            held_r2=0.401,
            runtime=20.0,
        ),
        _synthetic_result(
            "a10_early_stopping_refit",
            default_scores + 0.003,
            held_r2=0.39,
            runtime=20.0,
        ),
        _synthetic_result(
            "linear_residual",
            default_scores + 0.003,
            held_r2=0.401,
            runtime=30.0,
        ),
    ]

    decision = ablation.analyze_results(results)
    candidates = {row["config"]: row for row in decision["candidates"]}

    assert decision["advancing_candidate"] == "a10_numeric"
    assert decision["recommendation"] == "advance_a10_numeric"
    assert candidates["a10_numeric"]["passes_all_gates"] is True
    assert candidates["a10_numeric_2000"]["gates"]["fold_breadth"] is False
    assert candidates["a10_early_stopping_refit"]["gates"][
        "held_team_no_regression"
    ] is False
    assert candidates["linear_residual"]["gates"][
        "historical_runtime_cap"
    ] is False


def test_analyze_results_rejects_a_single_fold_driven_gain():
    default_scores = np.full(10, 0.5)
    one_fold = default_scores.copy()
    one_fold[-1] += 0.03
    neutral = default_scores.copy()
    results = [
        _synthetic_result("default", default_scores),
        _synthetic_result("a10_numeric", one_fold),
        _synthetic_result("a10_numeric_2000", neutral),
        _synthetic_result("a10_early_stopping_refit", neutral),
        _synthetic_result("linear_residual", neutral),
    ]

    decision = ablation.analyze_results(results)
    candidate = decision["candidates"][0]

    assert candidate["mean_r2_delta"] == pytest.approx(0.003)
    assert min(candidate["leave_one_fold_out_mean_deltas"]) == pytest.approx(0.0)
    assert candidate["gates"]["positive_leave_one_fold_out_gain"] is False
    assert candidate["passes_all_gates"] is False


def test_prediction_fingerprint_is_shape_and_value_stable():
    expected = ablation._prediction_sha256(np.array([1.0, 2.0]))
    assert expected == ablation._prediction_sha256([1.0, 2.0])
    assert expected != ablation._prediction_sha256([2.0, 1.0])


def test_invalid_phase_timing_fails_closed():
    with pytest.raises(RuntimeError, match="invalid value"):
        ablation._phase_timing(SimpleNamespace(timing_={"tree_build": -1.0}))


def test_profile_worker_process_forwards_requested_fold(monkeypatch):
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        payload = ablation.WORKER_RESULT_PREFIX + json.dumps({}) + "\n"
        return SimpleNamespace(returncode=0, stdout=payload, stderr="")

    monkeypatch.setattr(ablation.subprocess, "run", fake_run)
    args = SimpleNamespace(
        data_cache=Path("data.csv"),
        threads=2,
        profile_fold=3,
    )

    ablation._run_worker_process(args, "default", profile=True)

    index = observed["command"].index("--profile-fold")
    assert observed["command"][index + 1] == "3"


@pytest.mark.parametrize("value", ("-1", "10"))
def test_parse_args_rejects_out_of_range_profile_fold(value):
    with pytest.raises(SystemExit):
        ablation.parse_args(["--profile-fold", value])
