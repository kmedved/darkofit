from __future__ import annotations

import copy

from benchmarks import run_basketball_group_linear_selector as experiment


def _model_hash(value="same"):
    return {"canonical_payload_sha256": value}


def _record(prediction="same", model="same", selected=False, margin=0.01):
    return {
        "prediction_sha256": prediction,
        "metadata": {
            "model_state": _model_hash(model),
            "split": {"groups_disjoint": True},
            "selected_linear_leaves": selected,
            "relative_validation_improvement": margin,
        },
    }


def _arm(config):
    folds = [_record() for _ in range(10)]
    return {
        "config": config,
        "mean_r2": experiment.EXPECTED_CONTROL_MEAN_R2,
        "fold_scores": [0.5] * 10,
        "folds": folds,
        "holdout": {
            **_record(),
            "scores": {
                "overlap_exposed_team_holdout": {"r2": 0.5},
                "seen_player_subset": {"r2": 0.5},
                "cold_player_subset": {"r2": 0.5},
            },
        },
    }


def test_exactness_requires_decline_disjoint_predictions_and_models():
    control = _arm(experiment.CONTROL)
    candidate = _arm(experiment.CANDIDATE)

    decision = experiment.analyze_exact(control, candidate)
    assert decision["passes"]
    assert decision["maximum_selection_margin"] == 0.01

    changed = copy.deepcopy(candidate)
    changed["folds"][2]["metadata"]["selected_linear_leaves"] = True
    assert not experiment.analyze_exact(control, changed)["passes"]

    changed = copy.deepcopy(candidate)
    changed["holdout"]["metadata"]["model_state"][
        "canonical_payload_sha256"
    ] = "different"
    assert not experiment.analyze_exact(control, changed)["passes"]


def test_declared_orders_reverse_candidate_against_control():
    positions = [
        order.index(experiment.CANDIDATE) > order.index(experiment.CONTROL)
        for order in experiment.BLOCK_ORDERS
    ]
    assert positions == [True, False, True]


def test_selector_threshold_is_three_percent():
    assert experiment.MIN_RELATIVE_IMPROVEMENT == 0.03


def test_behavior_fingerprint_excludes_resource_observations():
    baseline = {
        "prediction_sha256": "same",
        "peak_rss_bytes": 100,
        "fit_seconds": 1.0,
    }
    repeated = {
        "prediction_sha256": "same",
        "peak_rss_bytes": 200,
        "fit_seconds": 2.0,
    }
    changed = {
        **repeated,
        "prediction_sha256": "different",
    }

    assert experiment._behavior_fingerprint(
        baseline
    ) == experiment._behavior_fingerprint(repeated)
    assert experiment._behavior_fingerprint(
        baseline
    ) != experiment._behavior_fingerprint(changed)
