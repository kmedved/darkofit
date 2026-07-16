from copy import deepcopy

import numpy as np
import pytest

from benchmarks import run_basketball_auto_lr_refit as experiment


def test_frozen_arms_isolate_auto_lr_early_stopping_and_refit():
    assert experiment.CONFIGS["default"] == {}
    assert experiment.CONFIGS["auto_lr_early_stopping_refit"] == {
        "early_stopping": True,
        "early_stopping_rounds": None,
        "validation_fraction": 0.1,
        "use_best_model": True,
        "refit": True,
        "refit_strategy": "exact",
    }
    candidate = experiment.CONFIGS["auto_lr_early_stopping_refit"]
    assert "learning_rate" not in candidate
    assert "iterations" not in candidate
    assert experiment.TIMING_SCHEDULE == (
        ("default", "auto_lr_early_stopping_refit"),
        ("auto_lr_early_stopping_refit", "default"),
        ("default", "auto_lr_early_stopping_refit"),
    )


def _fit_metadata(*, candidate=False, phase=1.0):
    metadata = {
        "best_iteration": 100 if candidate else 1_000,
        "fitted_tree_count": 100 if candidate else 1_000,
        "resolved_learning_rate": 0.0523,
        "requested_tree_mode": "catboost",
        "selected_tree_mode": "catboost",
        "selected_lane": "boosting",
        "linear_residual_active": False,
        "resolved_thread_count": 8,
        "refit": candidate,
        "refit_strategy": "exact" if candidate else None,
        "final_fit": {
            "iterations_requested": 100 if candidate else 1_000,
            "iterations_attempted": 100 if candidate else 1_000,
            "rounds_completed": 100 if candidate else 1_000,
            "rounds_retained": 100 if candidate else 1_000,
            "stop_reason": "iteration_limit",
            "phase_seconds": {"tree_build": phase},
        },
        "selection_fit": (
            {
                "iterations_requested": 1_000,
                "iterations_attempted": 196,
                "rounds_completed": 196,
                "rounds_retained": 100,
                "stop_reason": "early_stopping",
                "phase_seconds": {"tree_build": phase},
            }
            if candidate
            else None
        ),
        "selection_early_stopping_rounds": 96 if candidate else None,
        "final_early_stopping_rounds": None,
    }
    return metadata


def test_validate_fitted_metadata_accepts_frozen_routes():
    experiment.validate_fitted_metadata("default", _fit_metadata())
    experiment.validate_fitted_metadata(
        "auto_lr_early_stopping_refit", _fit_metadata(candidate=True)
    )


def test_validate_fitted_metadata_rejects_missing_selection():
    metadata = _fit_metadata(candidate=True)
    metadata["selection_fit"] = None
    with pytest.raises(RuntimeError, match="missing selection"):
        experiment.validate_fitted_metadata(
            "auto_lr_early_stopping_refit", metadata
        )


def test_validate_fitted_metadata_rejects_inexact_final_tree_count():
    metadata = _fit_metadata(candidate=True)
    metadata["fitted_tree_count"] += 1
    with pytest.raises(RuntimeError, match="retained the wrong tree count"):
        experiment.validate_fitted_metadata(
            "auto_lr_early_stopping_refit", metadata
        )


def _behavior_result(*, phase=1.0, prediction_hash="fold-hash"):
    return {
        "config": "default",
        "folds": [
            {
                "fold": 0,
                "r2": 0.5,
                "prediction_sha256": prediction_hash,
                "fit_metadata": _fit_metadata(phase=phase),
            }
        ],
        "holdout": {
            "scores": {"cold_player_subset": {"r2": 0.4}},
            "prediction_sha256": "holdout-hash",
            "fit_metadata": _fit_metadata(phase=phase),
        },
    }


def test_behavior_fingerprint_ignores_timing_but_not_predictions():
    baseline = experiment._behavior_fingerprint(_behavior_result(phase=1.0))
    assert baseline == experiment._behavior_fingerprint(
        _behavior_result(phase=99.0)
    )
    assert baseline != experiment._behavior_fingerprint(
        _behavior_result(prediction_hash="changed")
    )


def _canonical(candidate_scores, *, cold_delta=0.01, team_delta=0.01):
    default_score = experiment.EXPECTED_DEFAULT_MEAN_R2

    def result(name, scores, cold, team):
        return {
            "config": name,
            "fold_scores": list(scores),
            "mean_r2": float(np.mean(scores)),
            "holdout": {
                "scores": {
                    "cold_player_subset": {"r2": cold},
                    "overlap_exposed_team_holdout": {"r2": team},
                }
            },
        }

    default_scores = np.full(10, default_score)
    return {
        "default": result("default", default_scores, 0.40, 0.50),
        "auto_lr_early_stopping_refit": result(
            "auto_lr_early_stopping_refit",
            np.asarray(candidate_scores),
            0.40 + cold_delta,
            0.50 + team_delta,
        ),
    }


def _timing(*, candidate=(14.0, 15.0, 16.0), default=(24.0, 25.0, 26.0)):
    return {
        "default": experiment._timing_summary(list(default)),
        "auto_lr_early_stopping_refit": experiment._timing_summary(
            list(candidate)
        ),
    }


def test_analyze_results_advances_only_broad_stable_gain():
    base = experiment.EXPECTED_DEFAULT_MEAN_R2
    decision = experiment.analyze_results(
        _canonical(np.full(10, base + 0.001)),
        _timing(),
    )

    assert decision["fold_wins"] == 10
    assert decision["passes_quality_gates"] is True
    assert decision["passes_timing_gates"] is True
    assert decision["passes_all_gates"] is True
    assert decision["kernel_profile"]["required"] is True


def test_analyze_results_rejects_cold_player_regression():
    base = experiment.EXPECTED_DEFAULT_MEAN_R2
    decision = experiment.analyze_results(
        _canonical(np.full(10, base + 0.001), cold_delta=-0.001),
        _timing(),
    )

    assert decision["quality_gates"]["cold_player_no_regression"] is False
    assert decision["passes_all_gates"] is False
    assert decision["kernel_profile"]["required"] is False


def test_analyze_results_rejects_single_fold_gain():
    base = experiment.EXPECTED_DEFAULT_MEAN_R2
    candidate = np.full(10, base)
    candidate[-1] += 0.02
    decision = experiment.analyze_results(_canonical(candidate), _timing())

    assert decision["quality_gates"]["fold_breadth"] is False
    assert decision["quality_gates"]["leave_one_fold_out_no_regression"] is True
    assert decision["passes_all_gates"] is False


def test_analyze_results_rejects_unstable_or_small_speed_gain():
    base = experiment.EXPECTED_DEFAULT_MEAN_R2
    decision = experiment.analyze_results(
        _canonical(np.full(10, base + 0.001)),
        _timing(candidate=(20.0, 21.0, 30.0)),
    )

    assert decision["timing_gates"]["candidate_timing_stable"] is False
    assert decision["timing_gates"]["material_speedup"] is False
    assert decision["passes_all_gates"] is False


def test_timing_summary_requires_three_repetitions():
    with pytest.raises(RuntimeError, match="three repetitions"):
        experiment._timing_summary([1.0, 2.0])
