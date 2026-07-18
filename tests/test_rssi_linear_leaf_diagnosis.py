import copy
import json
from pathlib import Path

import pytest

from benchmarks import run_rssi_linear_leaf_diagnosis as diagnosis


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks" / "rssi_linear_leaf_diagnosis.json"


def _row(arm, *, marker="same", best=1.0, selected=None, cross=None):
    return {
        "arm": arm,
        "borders_sha256": marker,
        "validation_history_sha256": marker,
        "model_sha256": marker,
        "prediction_sha256": marker,
        "fitted_tree_count": 10,
        "best_validation_rmse": best,
        "test_rmse": best,
        "linear_leaves_selected": selected,
        "cross_features_selected": cross,
    }


def _valid_rows():
    rows = []
    for arm in diagnosis.ARMS:
        if arm in {
            "darko_shared_constant",
            "chimera_shared_constant",
            "chimera_full_selector",
        }:
            row = _row(arm, best=0.8, selected=False)
        elif arm in {
            "darko_shared_linear",
            "darko_matched_auto20_linear",
            "chimera_shared_linear",
        }:
            row = _row(arm, best=1.0)
        elif arm == "chimera_capped_selector":
            row = _row(arm, best=1.0, selected=True)
        elif arm == "chimera_product":
            row = _row(arm, best=1.0, selected=True, cross=False)
        elif arm == "chimera_full_product":
            row = _row(arm, best=1.0, selected=False, cross=False)
        else:
            row = _row(arm, best=1.0)
        rows.append(row)
    return rows


def test_protocol_uses_only_previously_scored_coordinate():
    registry = json.loads(
        (ROOT / "benchmarks" / "fresh_confirmation_registry.json").read_text()
    )
    coordinate = {
        "task_id": diagnosis.TASK_ID,
        "repeat": diagnosis.OUTER_REPEAT,
        "fold": diagnosis.OUTER_FOLD,
        "sample": diagnosis.OUTER_SAMPLE,
    }
    assert coordinate in registry["coordinates"]

    prior = json.loads(
        (ROOT / "benchmarks" / "fresh_selector_confirmation.json").read_text()
    )
    matching = [
        row for row in prior["results"] if row["task_id"] == diagnosis.TASK_ID
    ]
    assert matching
    assert all(
        diagnosis.OUTER_FOLD in [fold["fold"] for fold in row["folds"]]
        for row in matching
    )


def test_analysis_reports_capped_misselection_without_making_shipping_claim():
    analysis = diagnosis.analyze(_valid_rows())
    assert analysis["forced_full_budget_validation_winner"] == "constant"
    assert analysis["full_selector_winner"] == "constant"
    assert analysis["capped_selector_winner"] == "linear"
    assert analysis["capped_selector_disagrees_with_full"] is True
    assert analysis["fresh_claim_eligible"] is False
    assert analysis["claim_tier"] == "development_diagnostic_only"


def test_analysis_rejects_nonexact_matched_engines():
    rows = _valid_rows()
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["arm"] == "chimera_shared_linear":
            row["prediction_sha256"] = "different"
    with pytest.raises(RuntimeError, match="prediction_sha256"):
        diagnosis.analyze(changed)


def test_analysis_rejects_full_selector_that_disagrees_with_forced_race():
    rows = _valid_rows()
    for row in rows:
        if row["arm"] == "chimera_full_selector":
            row["linear_leaves_selected"] = True
    with pytest.raises(RuntimeError, match="full selector disagrees"):
        diagnosis.analyze(rows)


def test_arm_manifest_has_no_duplicate_or_undeclared_lane():
    assert len(diagnosis.ARMS) == len(set(diagnosis.ARMS))
    assert set(diagnosis.ARMS) == {
        "darko_default",
        "darko_matched_auto10_linear",
        "darko_matched_auto20_linear",
        "darko_shared_constant",
        "darko_shared_linear",
        "chimera_shared_constant",
        "chimera_shared_linear",
        "chimera_full_selector",
        "chimera_capped_selector",
        "chimera_full_product",
        "chimera_product",
    }


def test_recorded_artifact_reproduces_binding_diagnosis(
    assert_analysis_equal,
):
    artifact = json.loads(ARTIFACT.read_text())
    analysis = diagnosis.analyze(artifact["results"])
    assert_analysis_equal(artifact["analysis"], analysis)
    assert artifact["spent_boundary"]["fresh_claim_eligible"] is False
    assert artifact["protocol"]["timing_claim_eligible"] is False
    assert analysis["capped_selector_disagrees_with_full"] is True
    assert analysis["chimera_product_cross_selected"] is False
    assert (
        analysis["test_rmse_ratios"][
            "darko_shared_constant_over_chimera_product"
        ]
        < 1.0
    )
