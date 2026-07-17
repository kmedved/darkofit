from copy import deepcopy

import numpy as np
import pytest

from benchmarks import run_basketball_quantile_calibration as experiment
from darkofit.tree import ObliviousTree


def test_quantile_offset_uses_the_exact_frozen_rank():
    y = np.arange(9, dtype=np.float64)
    prediction = np.zeros(9, dtype=np.float64)

    lower, lower_rank = experiment.quantile_offset(y, prediction, 0.1)
    upper, upper_rank = experiment.quantile_offset(y, prediction, 0.9)

    assert (lower, lower_rank) == (0.0, 1)
    assert (upper, upper_rank) == (8.0, 9)
    with pytest.raises(ValueError, match="equal nonempty"):
        experiment.quantile_offset([], [], 0.1)


def test_pinball_and_interval_metrics_are_exact():
    target = np.array([0.0, 1.0, 2.0])
    raw = {
        "0.1": np.array([-1.0, 0.0, 1.0]),
        "0.9": np.array([1.0, 2.0, 3.0]),
    }
    scored = experiment.score_view(
        target, raw, {"0.1": -1.0, "0.9": 1.0}
    )

    assert experiment.pinball_loss(target, raw["0.1"], 0.1) == pytest.approx(0.1)
    assert scored["control"]["interval"]["coverage"] == 1.0
    assert scored["control"]["interval"]["mean_width"] == 2.0
    assert scored["candidate"]["interval"]["mean_width"] == 4.0
    assert scored["control"]["interval"]["crossing_count"] == 0


def test_calibration_split_is_deterministic_disjoint_and_complete():
    fit, calibration = experiment.calibration_split(101)
    fit_again, calibration_again = experiment.calibration_split(101)

    assert np.array_equal(fit, fit_again)
    assert np.array_equal(calibration, calibration_again)
    assert len(np.intersect1d(fit, calibration)) == 0
    assert np.array_equal(
        np.sort(np.concatenate((fit, calibration))), np.arange(101)
    )


def test_finiteness_audit_traverses_slotted_tree_state():
    finite = ObliviousTree(
        np.array([0]), np.array([0]), np.array([1.0, 2.0])
    )
    nonfinite_value = ObliviousTree(
        np.array([0]), np.array([0]), np.array([1.0, np.nan])
    )
    nonfinite_gain = ObliviousTree(
        np.array([0]),
        np.array([0]),
        np.array([1.0, 2.0]),
        gains=np.array([np.inf]),
    )

    assert experiment._numeric_state_is_finite(finite) is True
    assert experiment._numeric_state_is_finite(nonfinite_value) is False
    assert experiment._numeric_state_is_finite(nonfinite_gain) is False


def test_finiteness_audit_covers_estimator_wrapper_state():
    estimator = experiment.darkofit_package.DarkoRegressor()
    estimator._learning_rate_ = np.nan

    assert experiment._numeric_state_is_finite(estimator) is False


def _tail(pinball, coverage_error):
    return {
        "pinball_loss": float(pinball),
        "absolute_coverage_error": float(coverage_error),
    }


def _view(control=1.0, candidate=0.9):
    return {
        "control": {
            "tails": {
                "0.1": _tail(control / 2, 0.1),
                "0.9": _tail(control / 2, 0.1),
            },
            "interval": {
                "summed_pinball_loss": float(control),
                "absolute_coverage_error": 0.1,
                "mean_width": 10.0,
                "crossing_count": 1,
            },
        },
        "candidate": {
            "tails": {
                "0.1": _tail(candidate / 2, 0.05),
                "0.9": _tail(candidate / 2, 0.05),
            },
            "interval": {
                "summed_pinball_loss": float(candidate),
                "absolute_coverage_error": 0.05,
                "mean_width": 11.0,
                "crossing_count": 0,
            },
        },
    }


def _passing_inputs():
    pooled = _view()
    folds = [{"scores": _view()} for _ in range(10)]
    holdout = {
        "overlap_exposed_team_holdout": _view(),
        "seen_player_subset": _view(),
        "cold_player_subset": _view(),
    }
    return pooled, folds, holdout


def test_analysis_passes_only_a_broad_strict_no_regression_result():
    pooled, folds, holdout = _passing_inputs()

    decision = experiment.analyze_results(
        pooled, folds, holdout, all_finite=True
    )

    assert decision["passes_all_gates"] is True
    assert decision["strict_fold_wins"] == 10
    assert decision["default_promotion_authorized"] is False
    assert decision["recommendation"] == (
        "advance_to_separately_reviewed_default_off_implementation"
    )


def test_analysis_fails_when_one_cold_tail_regresses():
    pooled, folds, holdout = _passing_inputs()
    holdout["cold_player_subset"]["candidate"]["tails"]["0.1"][
        "pinball_loss"
    ] = 0.51

    decision = experiment.analyze_results(
        pooled, folds, holdout, all_finite=True
    )

    assert decision["quality_gates"]["cold_tail_pinball_no_regression"] is False
    assert decision["passes_all_gates"] is False
    assert decision["recommendation"] == "stop_before_product_implementation"


def test_analysis_rejects_noop_and_added_crossings():
    pooled, folds, holdout = _passing_inputs()
    for fold in folds:
        fold["scores"] = _view(control=1.0, candidate=1.0)
    holdout_with_crossing = deepcopy(holdout)
    holdout_with_crossing["cold_player_subset"]["candidate"]["interval"][
        "crossing_count"
    ] = 2

    noop = experiment.analyze_results(
        pooled, folds, holdout, all_finite=True
    )
    crossing = experiment.analyze_results(
        pooled, _passing_inputs()[1], holdout_with_crossing, all_finite=True
    )

    assert noop["quality_gates"]["strict_fold_win_breadth"] is False
    assert crossing["quality_gates"][
        "crossing_count_no_regression_every_view"
    ] is False


def test_frozen_binding_rejects_wrong_runtime_or_package(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, "DEFAULT_OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(
        experiment,
        "_git",
        lambda *args, **kwargs: experiment.EXPECTED_DARKOFIT_TREE,
    )
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda *args, **kwargs: type("Completed", (), {"returncode": 0})(),
    )
    monkeypatch.setattr(
        experiment.creator,
        "git_state",
        lambda root: {"clean": True},
    )
    monkeypatch.setattr(
        experiment,
        "runtime_state",
        lambda: {
            **experiment.EXPECTED_RUNTIME,
            "dependencies": dict(experiment.EXPECTED_RUNTIME["dependencies"]),
            "python_executable": "/frozen/python",
        },
    )

    binding = experiment.validate_frozen_binding(experiment.DEFAULT_OUTPUT)
    assert binding["source"]["clean"] is True

    monkeypatch.setattr(
        experiment,
        "runtime_state",
        lambda: {
            **experiment.EXPECTED_RUNTIME,
            "python": "3.13.0",
            "dependencies": dict(experiment.EXPECTED_RUNTIME["dependencies"]),
            "python_executable": "/wrong/python",
        },
    )
    with pytest.raises(RuntimeError, match="python"):
        experiment.validate_frozen_binding(experiment.DEFAULT_OUTPUT)


def test_create_only_artifact_publication(tmp_path):
    output = tmp_path / "result.json"
    experiment._atomic_write_new_bytes(output, b"first\n")

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        experiment._atomic_write_new_bytes(output, b"second\n")

    assert output.read_bytes() == b"first\n"


def test_runner_and_support_sources_match_the_frozen_manifest():
    assert experiment._normalized_runner_sha256() == (
        experiment.EXPECTED_NORMALIZED_RUNNER_SHA256
    )
    assert experiment.support_sha256() == experiment.EXPECTED_SUPPORT_SHA256
    assert experiment.darkofit_package.__file__.startswith(
        str(experiment.REPO_ROOT)
    )
