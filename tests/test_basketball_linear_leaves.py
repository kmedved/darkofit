from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmarks import basketball_harness as harness
from benchmarks import run_basketball_linear_leaves as experiment


def test_selection_split_is_deterministic_disjoint_and_hashed():
    X = pd.DataFrame({"value": np.arange(100)})
    y = pd.Series(np.arange(100, dtype=np.float64))
    train, validation, metadata = experiment.selection_split(X, y)
    again_train, again_validation, again_metadata = experiment.selection_split(X, y)

    assert np.array_equal(train, again_train)
    assert np.array_equal(validation, again_validation)
    assert metadata == again_metadata
    assert len(train) == 90
    assert len(validation) == 10
    assert set(train).isdisjoint(set(validation))
    assert sorted(np.concatenate((train, validation)).tolist()) == list(range(100))
    assert metadata["policy"] == "random"
    assert len(metadata["train_positions_sha256"]) == 64


def _fitted_metadata(lane="boosting"):
    return {
        "best_iteration": 1000,
        "fitted_tree_count": 1000,
        "resolved_learning_rate": 0.05,
        "requested_tree_mode": "catboost",
        "selected_tree_mode": "catboost",
        "selected_lane": lane,
        "linear_residual_active": False,
        "linear_leaves_active": lane == "linear_leaves",
        "linear_leaves": {"active": lane == "linear_leaves"},
        "resolved_thread_count": 4,
        "refit": False,
        "refit_strategy": None,
        "final_fit": {
            "iterations_requested": 1000,
            "iterations_attempted": 1000,
            "rounds_completed": 1000,
            "rounds_retained": 1000,
            "stop_reason": "iteration_limit",
            "phase_seconds": {},
        },
        "selection_fit": None,
        "selection_early_stopping_rounds": None,
        "final_early_stopping_rounds": None,
    }


def _candidate_metadata(constant_score=5.0, linear_score=4.9):
    selected = linear_score < constant_score
    records = [
        {
            "name": "constant",
            "linear_leaves": False,
            "validation_rmse": constant_score,
            "validation_rmse_history": [constant_score + 0.1, constant_score],
            "validation": {"source": "explicit_eval_set"},
            "fit_metadata": _fitted_metadata("boosting"),
        },
        {
            "name": "linear",
            "linear_leaves": True,
            "validation_rmse": linear_score,
            "validation_rmse_history": [linear_score + 0.1, linear_score],
            "validation": {"source": "explicit_eval_set"},
            "fit_metadata": _fitted_metadata("linear_leaves"),
        },
    ]
    return {
        "kind": "validation_selector",
        "selection_fits": records,
        "selected_linear_leaves": selected,
        "selected_name": "linear" if selected else "constant",
        "final_fit": _fitted_metadata("linear_leaves" if selected else "boosting"),
    }


def test_fitted_metadata_validation_enforces_selector_and_tie_policy():
    default = {
        "kind": "single",
        "selected_linear_leaves": False,
        "final_fit": _fitted_metadata(),
    }
    experiment.validate_fitted_metadata(experiment.DEFAULT_CONFIG, default)

    candidate = _candidate_metadata()
    experiment.validate_fitted_metadata(experiment.CANDIDATE_CONFIG, candidate)
    tie = _candidate_metadata(constant_score=5.0, linear_score=5.0)
    assert tie["selected_linear_leaves"] is False
    experiment.validate_fitted_metadata(experiment.CANDIDATE_CONFIG, tie)

    invalid = deepcopy(candidate)
    invalid["selected_linear_leaves"] = False
    with pytest.raises(RuntimeError, match="lane disagrees|tie/score policy"):
        experiment.validate_fitted_metadata(experiment.CANDIDATE_CONFIG, invalid)


def _scores(value):
    return {
        "overlap_exposed_team_holdout": {"r2": value},
        "cold_player_subset": {"r2": value},
        "seen_player_subset": {"r2": value},
    }


def _canonical(delta=0.001, holdout_delta=0.01):
    default_scores = np.full(
        experiment.creator.N_SPLITS,
        experiment.EXPECTED_DEFAULT_MEAN_R2,
        dtype=np.float64,
    )
    candidate_scores = default_scores + delta

    def folds(scores, model_bytes):
        return [{"r2": float(score), "model_bytes": model_bytes} for score in scores]

    return {
        experiment.DEFAULT_CONFIG: {
            "mean_r2": float(default_scores.mean()),
            "fold_scores": default_scores.tolist(),
            "folds": folds(default_scores, 100),
            "holdout": {"scores": _scores(0.50), "model_bytes": 100},
        },
        experiment.CANDIDATE_CONFIG: {
            "mean_r2": float(candidate_scores.mean()),
            "fold_scores": candidate_scores.tolist(),
            "folds": folds(candidate_scores, 250),
            "holdout": {
                "scores": _scores(0.50 + holdout_delta),
                "model_bytes": 250,
            },
        },
    }


def test_quality_analysis_requires_every_noisy_data_gate():
    passing = experiment.analyze_quality(_canonical())
    assert passing["passes_quality_gates"] is True
    assert passing["fold_wins"] == experiment.creator.N_SPLITS

    cold_failure = _canonical()
    cold_failure[experiment.CANDIDATE_CONFIG]["holdout"]["scores"][
        "cold_player_subset"
    ]["r2"] = 0.49
    failed = experiment.analyze_quality(cold_failure)
    assert failed["quality_gates"]["cold_player_no_regression"] is False
    assert failed["passes_quality_gates"] is False

    broad_failure = _canonical(delta=0.001)
    scores = np.asarray(broad_failure[experiment.CANDIDATE_CONFIG]["fold_scores"])
    scores[:5] -= 0.002
    broad_failure[experiment.CANDIDATE_CONFIG]["fold_scores"] = scores.tolist()
    broad_failure[experiment.CANDIDATE_CONFIG]["mean_r2"] = float(scores.mean())
    failed = experiment.analyze_quality(broad_failure)
    assert failed["quality_gates"]["fold_breadth"] is False


def test_runtime_analysis_checks_fit_predict_model_and_memory_budgets():
    wall = {
        experiment.DEFAULT_CONFIG: harness.timing_summary([10.0, 10.5, 11.0]),
        experiment.CANDIDATE_CONFIG: harness.timing_summary([20.0, 21.0, 22.0]),
    }
    fit = {
        experiment.DEFAULT_CONFIG: harness.timing_summary([8.0, 8.5, 9.0]),
        experiment.CANDIDATE_CONFIG: harness.timing_summary([24.0, 25.0, 26.0]),
    }
    predict = {
        experiment.DEFAULT_CONFIG: harness.timing_summary([1.0, 1.05, 1.1]),
        experiment.CANDIDATE_CONFIG: harness.timing_summary([1.1, 1.15, 1.2]),
    }
    rss = {
        experiment.DEFAULT_CONFIG: [100.0, 105.0, 110.0],
        experiment.CANDIDATE_CONFIG: [170.0, 175.0, 180.0],
    }
    result = experiment.analyze_runtime(_canonical(), wall, fit, predict, rss)
    assert result["passes_runtime_gates"] is True
    assert result["maximum_model_size_ratio"] == 2.5

    slow = deepcopy(predict)
    slow[experiment.CANDIDATE_CONFIG] = harness.timing_summary([1.4, 1.45, 1.5])
    result = experiment.analyze_runtime(_canonical(), wall, fit, slow, rss)
    assert result["runtime_gates"]["prediction_cost_within_budget"] is False
    assert result["passes_runtime_gates"] is False


@pytest.mark.parametrize("selected_linear", [False, True])
def test_candidate_warmup_compiles_the_opposite_prediction_lane(
    monkeypatch, selected_linear
):
    X = pd.DataFrame({"value": np.arange(20, dtype=np.float64)})
    y = pd.Series(np.arange(20, dtype=np.float64))
    dataset = SimpleNamespace(X=X, y=y)
    warmed = []

    monkeypatch.setattr(
        experiment,
        "fit_and_predict",
        lambda *args, **kwargs: (
            np.zeros(2),
            0.1,
            0.01,
            1,
            {"selected_linear_leaves": selected_linear},
        ),
    )

    class FakeModel:
        def predict(self, X_test):
            return np.zeros(len(X_test))

    def fake_fit_model(*args, linear_leaves, **kwargs):
        warmed.append(linear_leaves)
        return FakeModel(), 0.1

    monkeypatch.setattr(experiment, "_fit_model", fake_fit_model)
    experiment._warmup(experiment.CANDIDATE_CONFIG, dataset)
    assert warmed == [not selected_linear]


def test_parse_args_rejects_nonpositive_threads(tmp_path):
    args = experiment.parse_args(
        ["--threads", "2", "--output", str(tmp_path / "result.json")]
    )
    assert args.threads == 2
    assert args.output == tmp_path / "result.json"
    with pytest.raises(SystemExit):
        experiment.parse_args(["--threads", "0"])
