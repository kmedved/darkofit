import json

import pytest

from benchmarks import run_automatic_linear_selector_v2_m6_engagement as runner


def test_engagement_companion_uses_exact_m6_v3_grid():
    identities = runner.expected_identities()
    assert len(identities) == 60
    assert len(set(identities)) == 60
    assert {identity[1] for identity in identities} == {"medium"}
    assert {identity[2] for identity in identities} == {0, 1, 2}
    assert {identity[3] for identity in identities} == {"none", "stress"}


def test_engagement_companion_validates_selector_and_classification_absence():
    selector = {
        "version": 1,
        "requested": "auto",
        "fit_random_state_seed": 3,
        "eligible": True,
        "resolved_linear_leaves": False,
        "final_booster_linear_leaves": False,
        "final_linear_leaves_active": False,
        "reason": "margin_below_threshold",
        "minimum_relative_improvement": 0.03,
        "split": {},
        "constant_validation_rmse": 1.0,
        "linear_validation_rmse": 1.0,
        "relative_validation_improvement": 0.0,
        "selection_fits": [],
        "selection_total_seconds": 1.0,
    }
    row = {"model_metadata": json.dumps({
        "automatic_linear_selector": selector
    })}
    assert runner._selector_record(row, task="regression") == selector

    classification = {"model_metadata": json.dumps({
        "automatic_linear_selector": None
    })}
    assert runner._selector_record(
        classification, task="binary"
    ) == {
        "eligible": False,
        "resolved_linear_leaves": False,
        "reason": "classification_not_applicable",
    }

    with pytest.raises(RuntimeError, match="unexpectedly"):
        runner._selector_record(row, task="binary")
