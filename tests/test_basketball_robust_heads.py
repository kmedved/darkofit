import copy

import pytest

from benchmarks import run_basketball_robust_heads as experiment


def _holdout(value):
    return {
        "scores": {
            "overlap_exposed_team_holdout": {"r2": value},
            "seen_player_subset": {"r2": value},
            "cold_player_subset": {"r2": value},
        }
    }


def _arm(name, scores, holdout=0.4):
    return {
        "config": name,
        "fold_scores": list(scores),
        "mean_r2": sum(scores) / len(scores),
        "holdout": _holdout(holdout),
    }


def test_quality_gate_requires_mean_lofo_held_and_cold(monkeypatch):
    base = [0.5] * 10
    monkeypatch.setattr(experiment, "EXPECTED_CONTROL_MEAN_R2", 0.5)
    control = _arm(experiment.CONTROL, base)
    candidate = _arm(experiment.STUDENT_T, [0.503] * 10, holdout=0.401)

    result = experiment.analyze_quality(control, candidate)

    assert result["mean_r2_delta"] == pytest.approx(0.003)
    assert result["passes_quality_gates"] is True

    cold_regression = copy.deepcopy(candidate)
    cold_regression["holdout"]["scores"]["cold_player_subset"]["r2"] = 0.399
    failed = experiment.analyze_quality(control, cold_regression)
    assert failed["quality_gates"]["cold_player_no_regression"] is False
    assert failed["passes_quality_gates"] is False


def test_quality_gate_rejects_concentrated_or_subthreshold_gain(monkeypatch):
    monkeypatch.setattr(experiment, "EXPECTED_CONTROL_MEAN_R2", 0.5)
    control = _arm(experiment.CONTROL, [0.5] * 10)

    concentrated = _arm(
        experiment.MAE,
        [0.54] + [0.499] * 9,
    )
    result = experiment.analyze_quality(control, concentrated)
    assert result["mean_r2_delta"] > 0.002
    assert result["quality_gates"]["leave_one_fold_out_no_regression"] is False
    assert result["passes_quality_gates"] is False

    subthreshold = _arm(experiment.MAE, [0.501] * 10)
    failed = experiment.analyze_quality(control, subthreshold)
    assert failed["quality_gates"]["mean_r2_gain_at_least_0_002"] is False
    assert failed["passes_quality_gates"] is False


def test_declared_orders_reverse_each_candidate_against_control():
    for candidate in (experiment.STUDENT_T, experiment.MAE):
        relative = [
            order.index(candidate) > order.index(experiment.CONTROL)
            for order in experiment.BLOCK_ORDERS
        ]
        assert relative == [True, False, True]
