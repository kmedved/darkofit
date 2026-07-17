from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks import run_basketball_entity_ensemble as experiment


RECORDED_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "basketball_entity_ensemble.json"
)
EXPECTED_ARTIFACT_SHA256 = (
    "7088170060de9124d6508f1096df7af737332ab9e3b95ed11c1058b51e79ba35"
)


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


def test_recorded_artifact_closes_entity_ensemble_on_fatal_quality():
    raw = RECORDED_ARTIFACT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_ARTIFACT_SHA256
    artifact = json.loads(raw)

    assert artifact["source"]["clean"] is True
    assert artifact["source"]["head"] == (
        "d2c14ba3fc8131ee83d28905566efdb007f799cf"
    )
    assert artifact["protocol"]["sha256"] == hashlib.sha256(
        experiment.PROTOCOL.read_bytes()
    ).hexdigest()
    assert artifact["protocol"]["runner_sha256"] == hashlib.sha256(
        Path(experiment.__file__).read_bytes()
    ).hexdigest()
    assert artifact["protocol"]["executed_blocks"] == 1
    assert artifact["protocol"]["lockbox_data_used"] is False
    assert artifact["paired_timing"] is None
    assert artifact["passes_all_gates"] is False
    assert artifact["recommendation"] == "close_entity_ensemble_as_shaped"
    assert artifact["quality"]["mean_r2_delta"] < 0.0
    assert artifact["quality"]["cold_player_r2_delta"] > 0.0
