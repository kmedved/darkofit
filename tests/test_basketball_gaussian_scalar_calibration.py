from copy import deepcopy

import numpy as np
import pytest

from benchmarks import run_basketball_gaussian_scalar_calibration as experiment
from darkofit.tree import ObliviousTree


def test_gaussian_metrics_match_closed_form_at_the_mean():
    target = np.array([0.0, 2.0])
    mu = target.copy()
    sigma = np.array([1.0, 2.0])

    nll = experiment.gaussian_nll(target, mu, sigma)
    crps = experiment.gaussian_crps(target, mu, sigma)

    assert nll == pytest.approx(np.mean(np.log(sigma) + 0.5 * np.log(2.0 * np.pi)))
    assert crps == pytest.approx(
        np.mean(sigma) * (np.sqrt(2.0 / np.pi) - 1.0 / np.sqrt(np.pi))
    )
    with pytest.raises(ValueError, match="positive vectors"):
        experiment.gaussian_nll(target, mu, [1.0, 0.0])


def test_scalar_scale_uses_frozen_guard_and_is_deterministic():
    target = np.array([10000.0, -10000.0, 1.0])
    mu = np.zeros(3)
    sigma = np.ones(3)

    observed = experiment.fit_scalar_scale(target, mu, sigma)

    expected_z = np.array([1000.0, -1000.0, 1.0])
    assert observed == np.sqrt(np.mean(expected_z * expected_z))
    assert observed == experiment.fit_scalar_scale(target, mu, sigma)


def test_calibration_split_is_deterministic_disjoint_and_complete():
    fit, calibration = experiment.calibration_split(100)
    fit_again, calibration_again = experiment.calibration_split(100)

    assert np.array_equal(fit, fit_again)
    assert np.array_equal(calibration, calibration_again)
    assert len(np.intersect1d(fit, calibration)) == 0
    assert np.array_equal(np.sort(np.concatenate((fit, calibration))), np.arange(100))


def test_score_parameters_preserves_point_and_raw_invariants():
    target = np.array([0.0, 1.0, 2.0])
    mu = np.array([0.1, 0.9, 2.1])
    control_sigma = np.array([0.5, 0.75, 1.0])
    candidate_sigma = control_sigma * 1.25
    raw = np.column_stack((mu, np.log(control_sigma)))

    scored = experiment.score_parameters(
        target, raw, mu, control_sigma, candidate_sigma
    )

    assert all(scored["invariants"].values())
    assert scored["control"]["point_rmse"] == scored["candidate"]["point_rmse"]
    assert scored["control"]["sha256"]["raw"] == (scored["candidate"]["sha256"]["raw"])
    assert scored["candidate"]["mean_sigma"] == pytest.approx(
        scored["control"]["mean_sigma"] * 1.25
    )


def test_finiteness_audit_traverses_slotted_tree_state():
    finite = ObliviousTree(np.array([0]), np.array([0]), np.array([1.0, 2.0]))
    nonfinite = ObliviousTree(np.array([0]), np.array([0]), np.array([1.0, np.nan]))

    assert experiment._numeric_state_is_finite(finite) is True
    assert experiment._numeric_state_is_finite(nonfinite) is False


def _arm(nll, crps=0.9, coverage_error=0.05, width=2.0):
    return {
        "gaussian_nll": float(nll),
        "gaussian_crps": float(crps),
        "point_rmse": 1.0,
        "interval": {
            "absolute_coverage_error": float(coverage_error),
            "mean_width": float(width),
            "crossing_count": 0,
        },
    }


def _view(control=1.0, candidate=0.9):
    return {
        "control": _arm(control, crps=1.0, coverage_error=0.1, width=2.0),
        "candidate": _arm(candidate, crps=0.9, coverage_error=0.05, width=2.2),
        "invariants": {
            "means_array_exact": True,
            "point_rmse_exact": True,
            "raw_scores_shared": True,
            "positive_scales": True,
            "zero_crossings": True,
        },
    }


def _runtime():
    return {
        "gates": {
            "control_timing_stable": True,
            "candidate_timing_stable": True,
            "candidate_runtime_ratio": True,
            "candidate_transient_memory": True,
        }
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


def test_analysis_passes_only_broad_explicit_mode_validation():
    pooled, folds, holdout = _passing_inputs()

    decision = experiment.analyze_results(
        pooled,
        folds,
        holdout,
        _runtime(),
        all_finite=True,
        reconstruction_checks_pass=True,
    )

    assert decision["passes_all_gates"] is True
    assert decision["strict_fold_wins"] == 10
    assert decision["default_promotion_authorized"] is False
    assert decision["recommendation"] == (
        "advance_existing_explicit_scalar_mode_to_broader_validation"
    )


def test_analysis_rejects_cold_regression_and_insufficient_fold_breadth():
    pooled, folds, holdout = _passing_inputs()
    cold_regression = deepcopy(holdout)
    cold_regression["cold_player_subset"]["candidate"]["gaussian_crps"] = 1.1
    for fold in folds[5:]:
        fold["scores"]["candidate"]["gaussian_nll"] = 1.0

    decision = experiment.analyze_results(
        pooled,
        folds,
        cold_regression,
        _runtime(),
        all_finite=True,
        reconstruction_checks_pass=True,
    )

    assert decision["fatal_gates"]["cold_metrics_no_worse"] is False
    assert decision["fatal_gates"]["strict_fold_win_breadth"] is False
    assert decision["passes_all_gates"] is False
    assert decision["recommendation"] == (
        "stop_distributional_scalar_calibration_at_basketball"
    )


def test_analysis_rejects_width_and_worst_fold_regressions():
    pooled, folds, holdout = _passing_inputs()
    pooled["candidate"]["interval"]["mean_width"] = 2.6
    folds[0]["scores"]["candidate"]["gaussian_nll"] = 1.03

    decision = experiment.analyze_results(
        pooled,
        folds,
        holdout,
        _runtime(),
        all_finite=True,
        reconstruction_checks_pass=True,
    )

    assert decision["fatal_gates"]["width_ratios"] is False
    assert decision["fatal_gates"]["worst_fold_nll_ratio"] is False
    assert decision["passes_all_gates"] is False


def test_frozen_binding_rejects_wrong_runtime_or_branch(monkeypatch, tmp_path):
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
    source = {
        "clean": True,
        "branch": "main",
        "head": "source",
        "tracked_main_refs": {"origin/main": "source"},
    }
    monkeypatch.setattr(experiment.creator, "git_state", lambda root: source)
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

    source["branch"] = "topic"
    with pytest.raises(RuntimeError, match="main == origin/main"):
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
    assert experiment.darkofit_package.__file__.startswith(str(experiment.REPO_ROOT))
