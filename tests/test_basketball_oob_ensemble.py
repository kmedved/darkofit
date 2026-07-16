from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import darkofit
from benchmarks import basketball_harness as harness
from benchmarks import run_basketball_oob_ensemble as experiment


def _fit_metadata(*, stop_reason="early_stopping", patience=20):
    return {
        "best_iteration": 12,
        "fitted_tree_count": 12,
        "resolved_learning_rate": 0.1,
        "requested_tree_mode": "catboost",
        "selected_tree_mode": "catboost",
        "selected_lane": "boosting",
        "linear_residual_active": False,
        "resolved_thread_count": 4,
        "refit": False,
        "refit_strategy": None,
        "final_fit": {
            "iterations_requested": 1000,
            "iterations_attempted": 32,
            "rounds_completed": 32,
            "rounds_retained": 12,
            "stop_reason": stop_reason,
            "phase_seconds": {"tree_build": 0.5},
        },
        "selection_fit": None,
        "selection_early_stopping_rounds": None,
        "final_early_stopping_rounds": patience,
    }


def test_member_seeds_and_bootstraps_are_deterministic_and_oob_clean():
    seeds = experiment.member_seeds()
    assert seeds == experiment.member_seeds()
    assert len(seeds) == experiment.N_MEMBERS
    assert len(set(seeds)) == experiment.N_MEMBERS

    sampled, oob = experiment.bootstrap_plan(100, seeds[0])
    sampled_again, oob_again = experiment.bootstrap_plan(100, seeds[0])
    assert np.array_equal(sampled, sampled_again)
    assert np.array_equal(oob, oob_again)
    assert sampled.shape == (100,)
    assert len(oob) > 0
    assert set(oob).isdisjoint(set(sampled))

    with pytest.raises(ValueError, match="at least two members"):
        experiment.member_seeds(n_members=1)
    with pytest.raises(ValueError, match="at least two rows"):
        experiment.bootstrap_plan(1, seeds[0])


def test_oob_ensemble_fits_each_member_on_bootstrap_and_exact_oob(
    monkeypatch,
):
    seen = []

    class FakeRegressor:
        def __init__(self, **params):
            self.params = params
            self.random_state = params["random_state"]

        def fit(self, X, y, eval_set):
            X_eval, y_eval = eval_set
            assert len(X) == len(y) == 40
            assert len(X_eval) == len(y_eval)
            assert set(X_eval.index).isdisjoint(set(X.index))
            seen.append((self.random_state, X.index.tolist(), X_eval.index.tolist()))
            self.model_ = SimpleNamespace(
                auto_params_={
                    "validation_split": {
                        "source": "explicit_eval_set",
                        "eval_n_samples": len(X_eval),
                    }
                }
            )
            return self

        def predict(self, X):
            return np.full(len(X), self.random_state % 17, dtype=np.float64)

    monkeypatch.setattr(darkofit, "DarkoRegressor", FakeRegressor)
    monkeypatch.setattr(
        experiment.harness,
        "extract_fit_metadata",
        lambda model: _fit_metadata(),
    )
    X = pd.DataFrame({"x": np.arange(40)}, index=np.arange(100, 140))
    y = pd.Series(np.arange(40, dtype=np.float64), index=X.index)
    prediction, fit_seconds, predict_seconds, metadata = (
        experiment._fit_oob_ensemble(X, y, X.iloc[:3])
    )

    assert prediction.shape == (3,)
    assert fit_seconds >= 0.0
    assert predict_seconds >= 0.0
    assert len(seen) == experiment.N_MEMBERS
    assert metadata["member_count"] == experiment.N_MEMBERS
    assert tuple(metadata["member_seeds"]) == experiment.member_seeds()
    for member, (_, sampled_index, oob_index) in zip(metadata["members"], seen):
        assert member["bootstrap_rows"] == 40
        assert member["oob_rows"] == len(oob_index)
        assert set(sampled_index).isdisjoint(set(oob_index))
        assert member["validation"]["source"] == "explicit_eval_set"


def test_validate_candidate_metadata_requires_real_oob_early_stopping():
    members = []
    for member, seed in enumerate(experiment.member_seeds()):
        members.append(
            {
                "member": member,
                "seed": seed,
                "oob_rows": 10,
                "validation": {
                    "source": "explicit_eval_set",
                    "eval_n_samples": 10,
                },
                "fit_metadata": _fit_metadata(),
            }
        )
    metadata = {
        "kind": "oob_ensemble",
        "member_count": experiment.N_MEMBERS,
        "member_seeds": list(experiment.member_seeds()),
        "members": members,
    }
    experiment.validate_fitted_metadata(experiment.CANDIDATE_CONFIG, metadata)

    members[0]["fit_metadata"] = _fit_metadata(stop_reason="completed")
    with pytest.raises(RuntimeError, match="did not stop"):
        experiment.validate_fitted_metadata(
            experiment.CANDIDATE_CONFIG, metadata
        )


def _canonical_results(delta=0.001):
    default_scores = np.full(
        10, experiment.EXPECTED_DEFAULT_MEAN_R2, dtype=np.float64
    )
    candidate_scores = default_scores + delta
    score = lambda value: {  # noqa: E731
        "overlap_exposed_team_holdout": {"r2": value},
        "cold_player_subset": {"r2": value},
        "seen_player_subset": {"r2": value},
    }
    return {
        experiment.DEFAULT_CONFIG: {
            "mean_r2": float(default_scores.mean()),
            "fold_scores": default_scores.tolist(),
            "holdout": {"scores": score(0.50)},
        },
        experiment.CANDIDATE_CONFIG: {
            "mean_r2": float(candidate_scores.mean()),
            "fold_scores": candidate_scores.tolist(),
            "holdout": {"scores": score(0.51)},
        },
    }


def test_analysis_advances_only_as_opt_in_when_every_gate_passes():
    canonical = _canonical_results()
    wall = {
        experiment.DEFAULT_CONFIG: harness.timing_summary([10.0, 10.5, 11.0]),
        experiment.CANDIDATE_CONFIG: harness.timing_summary([25.0, 26.0, 27.0]),
    }
    predict = {
        experiment.DEFAULT_CONFIG: harness.timing_summary([1.0, 1.05, 1.1]),
        experiment.CANDIDATE_CONFIG: harness.timing_summary([4.8, 5.0, 5.2]),
    }

    decision = experiment.analyze_results(canonical, wall, predict)

    assert decision["passes_all_gates"] is True
    assert decision["candidate_scope"] == "opt_in_only"
    assert decision["default_promotion_authorized"] is False
    assert decision["recommendation"] == "advance_to_public_api_implementation"


def test_analysis_fails_closed_on_runtime_or_cold_player_regression():
    canonical = _canonical_results()
    canonical[experiment.CANDIDATE_CONFIG]["holdout"]["scores"][
        "cold_player_subset"
    ]["r2"] = 0.49
    wall = {
        experiment.DEFAULT_CONFIG: harness.timing_summary([10.0, 10.5, 11.0]),
        experiment.CANDIDATE_CONFIG: harness.timing_summary([45.0, 46.0, 47.0]),
    }
    predict = {
        experiment.DEFAULT_CONFIG: harness.timing_summary([1.0, 1.05, 1.1]),
        experiment.CANDIDATE_CONFIG: harness.timing_summary([4.8, 5.0, 5.2]),
    }

    decision = experiment.analyze_results(canonical, wall, predict)

    assert decision["quality_gates"]["cold_player_no_regression"] is False
    assert (
        decision["runtime_gates"]["beats_naive_fivefold_wall_scaling"]
        is False
    )
    assert decision["passes_all_gates"] is False
    assert decision["recommendation"] == "advance_none"


def test_analysis_marks_dirty_source_as_development_only():
    canonical = _canonical_results()
    wall = {
        experiment.DEFAULT_CONFIG: harness.timing_summary([10.0, 10.5, 11.0]),
        experiment.CANDIDATE_CONFIG: harness.timing_summary([25.0, 26.0, 27.0]),
    }
    predict = {
        experiment.DEFAULT_CONFIG: harness.timing_summary([1.0, 1.05, 1.1]),
        experiment.CANDIDATE_CONFIG: harness.timing_summary([4.8, 5.0, 5.2]),
    }

    decision = experiment.analyze_results(
        canonical, wall, predict, source_clean=False
    )

    assert decision["passes_quality_gates"] is True
    assert decision["passes_runtime_gates"] is True
    assert decision["evidence_eligible"] is False
    assert decision["passes_all_gates"] is False
    assert decision["recommendation"] == "development_only_dirty_source"


def test_parse_args_rejects_nonpositive_threads(tmp_path):
    args = experiment.parse_args(
        ["--threads", "3", "--output", str(tmp_path / "result.json")]
    )
    assert args.threads == 3
    assert args.output == tmp_path / "result.json"

    with pytest.raises(SystemExit):
        experiment.parse_args(["--threads", "0"])
