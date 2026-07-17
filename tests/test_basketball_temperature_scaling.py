from copy import deepcopy

import numpy as np
import pytest

from benchmarks import run_basketball_temperature_scaling as experiment
from darkofit.tree import ObliviousTree


def test_stable_sigmoid_and_log_loss_are_exact():
    logits = np.array([-1000.0, 0.0, 1000.0])
    target = np.array([0, 1, 1], dtype=np.uint8)

    probability = experiment.stable_sigmoid(logits)
    loss = experiment.binary_log_loss_from_logits(target, logits)

    assert np.array_equal(probability, np.array([0.0, 0.5, 1.0]))
    assert loss == pytest.approx(np.log(2.0) / 3.0)
    with pytest.raises(ValueError, match="zero or one"):
        experiment.binary_log_loss_from_logits([0, 2], [0.0, 0.0])


def test_expected_calibration_error_uses_frozen_equal_width_bins():
    target = np.array([0, 0, 1, 1], dtype=np.uint8)
    probability = np.array([0.05, 0.15, 0.85, 1.0])

    observed = experiment.expected_calibration_error(target, probability)

    assert observed == pytest.approx((0.05 + 0.15 + 0.15 + 0.0) / 4.0)
    with pytest.raises(ValueError, match="valid probabilities"):
        experiment.expected_calibration_error(target, [-0.1, 0.1, 0.9, 1.0])


def test_calibration_split_is_deterministic_stratified_and_complete():
    target = np.tile(np.array([0, 1], dtype=np.uint8), 100)

    fit, calibration = experiment.calibration_split(target)
    fit_again, calibration_again = experiment.calibration_split(target)

    assert np.array_equal(fit, fit_again)
    assert np.array_equal(calibration, calibration_again)
    assert len(np.intersect1d(fit, calibration)) == 0
    assert np.array_equal(
        np.sort(np.concatenate((fit, calibration))), np.arange(len(target))
    )
    assert set(target[fit]) == {0, 1}
    assert set(target[calibration]) == {0, 1}


def test_temperature_optimizer_uses_positive_bounded_scalar():
    target = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint8)
    logits = np.array([-8.0, -4.0, 2.0, -2.0, 4.0, 8.0])

    fitted = experiment.fit_temperature(target, logits)

    assert experiment.TEMPERATURE_BOUNDS[0] < fitted["temperature"]
    assert fitted["temperature"] < experiment.TEMPERATURE_BOUNDS[1]
    assert fitted["candidate_objective"] <= fitted["baseline_objective"]
    assert fitted["optimizer_success"] is True


def test_scoring_preserves_binary_decisions_order_and_ties():
    target = np.array([0, 0, 1, 1], dtype=np.uint8)
    logits = np.array([-2.0, -2.0, 1.0, 3.0])

    scored = experiment.score_view(
        target,
        logits,
        2.5,
        require_monotonic_invariants=True,
    )

    assert scored["invariants"] == {
        "required": True,
        "class_predictions_identical": True,
        "stable_score_order_identical": True,
        "score_ties_identical": True,
    }
    assert scored["control"]["accuracy"] == scored["candidate"]["accuracy"]
    assert scored["control"]["roc_auc"] == scored["candidate"]["roc_auc"]


def test_finiteness_audit_traverses_slotted_tree_state():
    finite = ObliviousTree(
        np.array([0]), np.array([0]), np.array([1.0, 2.0])
    )
    nonfinite = ObliviousTree(
        np.array([0]), np.array([0]), np.array([1.0, np.nan])
    )

    assert experiment._numeric_state_is_finite(finite) is True
    assert experiment._numeric_state_is_finite(nonfinite) is False


def _arm(log_loss, brier=0.2, ece=0.1):
    return {
        "log_loss": float(log_loss),
        "brier_score": float(brier),
        "ece_10_equal_width": float(ece),
    }


def _view(control=1.0, candidate=0.9):
    return {
        "control": _arm(control),
        "candidate": _arm(candidate, brier=0.19, ece=0.09),
        "invariants": {
            "required": True,
            "class_predictions_identical": True,
            "stable_score_order_identical": True,
            "score_ties_identical": True,
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


def test_analysis_passes_only_broad_no_regression_result():
    pooled, folds, holdout = _passing_inputs()

    decision = experiment.analyze_results(
        pooled,
        folds,
        holdout,
        _runtime(),
        all_finite=True,
        optimizer_checks_pass=True,
    )

    assert decision["passes_all_gates"] is True
    assert decision["strict_fold_wins"] == 10
    assert decision["default_promotion_authorized"] is False
    assert decision["recommendation"] == (
        "advance_to_separately_reviewed_default_off_implementation"
    )


def test_analysis_rejects_cold_regression_and_noop_fold_breadth():
    pooled, folds, holdout = _passing_inputs()
    cold_regression = deepcopy(holdout)
    cold_regression["cold_player_subset"]["candidate"]["ece_10_equal_width"] = 0.11
    for fold in folds:
        fold["scores"]["candidate"]["log_loss"] = 1.0

    decision = experiment.analyze_results(
        pooled,
        folds,
        cold_regression,
        _runtime(),
        all_finite=True,
        optimizer_checks_pass=True,
    )

    assert decision["fatal_gates"]["cold_metrics_no_worse"] is False
    assert decision["fatal_gates"]["strict_fold_win_breadth"] is False
    assert decision["passes_all_gates"] is False
    assert decision["recommendation"] == "stop_before_product_implementation"


def test_analysis_does_not_require_cross_fold_pooled_order():
    pooled, folds, holdout = _passing_inputs()
    pooled["invariants"]["required"] = False
    pooled["invariants"]["stable_score_order_identical"] = False

    decision = experiment.analyze_results(
        pooled,
        folds,
        holdout,
        _runtime(),
        all_finite=True,
        optimizer_checks_pass=True,
    )

    assert decision["fatal_gates"]["monotonic_invariants"] is True
    assert decision["passes_all_gates"] is True


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
    assert experiment.darkofit_package.__file__.startswith(
        str(experiment.REPO_ROOT)
    )
