import copy
import json
from pathlib import Path

import pytest

from benchmarks import analyze_smooth_cross_margin as analysis


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks" / "smooth_cross_margin_analysis.json"


def _row(dataset, fold, margin, ratio, *, selected=True):
    return {
        "task_id": hash(dataset) % 10_000,
        "dataset_name": dataset,
        "fold": fold,
        "cross_selected": selected,
        "base": {
            "best_validation_rmse": 1.0,
            "test_rmse": 1.0,
        },
        "selected": {
            "best_validation_rmse": 1.0 - margin,
            "test_rmse": ratio,
        },
    }


def _source():
    return {
        "results": [
            _row("a", 0, 0.02, 1.10),
            _row("a", 1, 0.06, 0.80),
            _row("b", 0, 0.07, 0.90),
            _row("b", 1, 0.00, 1.00, selected=False),
        ]
    }


def test_margin_evaluation_declines_below_threshold_as_exact_tie():
    result = analysis.evaluate_margin(_source()["results"], 0.05)
    records = {
        (row["dataset_name"], row["fold"]): row
        for row in result["split_records"]
    }
    assert records[("a", 0)]["engaged"] is False
    assert records[("a", 0)]["test_ratio"] == 1.0
    assert records[("a", 1)]["engaged"] is True
    assert result["worst_split_ratio"] == 1.0


def test_analysis_nominates_smallest_grid_margin_with_zero_observed_harm(
    monkeypatch,
):
    monkeypatch.setattr(analysis, "MARGIN_GRID", (0.0, 0.01, 0.05, 0.10))
    result = analysis.analyze(_source())
    assert result["nominee"]["minimum_validation_improvement"] == 0.05
    assert result["nominee"]["worst_split_ratio"] == 1.0
    assert result["fresh_claim_eligible"] is False
    assert result["nominee_requires_fresh_confirmation"] is True


def test_no_nominee_when_every_grid_point_has_harm(monkeypatch):
    source = _source()
    changed = copy.deepcopy(source)
    changed["results"][0]["selected"]["best_validation_rmse"] = 0.5
    monkeypatch.setattr(analysis, "MARGIN_GRID", (0.0, 0.10))
    result = analysis.analyze(changed)
    assert result["nominee"] is None
    assert result["nominee_requires_fresh_confirmation"] is False


def test_geomean_rejects_nonpositive_values():
    with pytest.raises(RuntimeError, match="positive"):
        analysis._geomean([1.0, 0.0])


def test_recorded_analysis_nominates_five_percent_for_fresh_confirmation(
    assert_analysis_equal,
):
    artifact = json.loads(ARTIFACT.read_text())
    source = json.loads(analysis.SOURCE.read_text())
    assert_analysis_equal(
        artifact["analysis"],
        analysis.analyze(source),
    )
    nominee = artifact["analysis"]["nominee"]
    assert nominee["minimum_validation_improvement"] == 0.05
    assert nominee["worst_split_ratio"] == 1.0
    assert artifact["analysis"]["fresh_claim_eligible"] is False
