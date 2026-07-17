from __future__ import annotations

import copy

import numpy as np

from benchmarks import run_basketball_entity_ensemble as experiment


def _arm(mean, held=0.5, cold=0.5, seen=0.5):
    return {
        "mean_r2": mean,
        "fold_scores": [mean] * 10,
        "holdout": {
            "scores": {
                "overlap_exposed_team_holdout": {"r2": held},
                "cold_player_subset": {"r2": cold},
                "seen_player_subset": {"r2": seen},
            }
        },
    }


def test_group_bootstrap_is_deterministic_and_group_disjoint():
    groups = np.asarray(["a", "a", "b", "c", "c", "d"])
    first = experiment.group_bootstrap_plan(groups, 17)
    second = experiment.group_bootstrap_plan(groups, 17)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert len(first[2]) == len(np.unique(groups))
    assert not set(first[2]).intersection(set(groups[first[1]]))
    for group in first[2]:
        expected = np.flatnonzero(groups == group)
        assert any(
            np.array_equal(first[0][start:start + len(expected)], expected)
            for start in range(len(first[0]) - len(expected) + 1)
        )


def test_quality_gate_requires_material_broad_and_guardrail_gain(monkeypatch):
    monkeypatch.setattr(experiment, "EXPECTED_CONTROL_MEAN_R2", 0.5)
    control = _arm(0.5)
    candidate = _arm(0.505, held=0.501, cold=0.501)

    result = experiment.analyze_quality(control, candidate)

    assert result["passes_quality_gates"]
    assert result["fold_wins"] == 10

    failed = copy.deepcopy(candidate)
    failed["fold_scores"][0] = 0.45
    failed["mean_r2"] = float(np.mean(failed["fold_scores"]))
    result = experiment.analyze_quality(control, failed)
    assert not result["quality_gates"]["leave_one_fold_out_no_regression"]


def test_declared_orders_reverse_candidate_against_control():
    positions = [
        order.index(experiment.CANDIDATE) > order.index(experiment.CONTROL)
        for order in experiment.BLOCK_ORDERS
    ]
    assert positions == [True, False, True]
