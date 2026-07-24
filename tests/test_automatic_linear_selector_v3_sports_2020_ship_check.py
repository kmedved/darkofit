from copy import deepcopy

import pytest

from benchmarks import (
    run_automatic_linear_selector_v3_sports_2020_ship_check as runner,
)


def _row(target, arm, *, exact=True, ratio=1.0):
    views = {}
    for view in ("all_held", "seen_player", "cold_player"):
        views[view] = {
            "rows": 10,
            "rmse": 2.0 * ratio if arm == "automatic" else 2.0,
            "prediction_sha256": (
                f"{target}-{view}-control"
                if arm == "control" or exact
                else f"{target}-{view}-automatic"
            ),
        }
    return {
        "target": target,
        "arm": arm,
        "integrity_passes": True,
        "panel_sha256": "panel",
        "train_rows": 300,
        "test_rows": 100,
        "fit_seconds": 2.0 if arm == "automatic" else 1.0,
        "predict_seconds": 1.0,
        "peak_process_tree_rss_bytes": 200 if arm == "automatic" else 100,
        "automatic_linear_selector": (
            None
            if arm == "control"
            else {"reason": "below_min_samples"}
        ),
        "views": views,
    }


def _rows(*, exact=True, ratio=1.0):
    rows = []
    for target in runner.TARGETS:
        rows.extend([
            _row(target, "control"),
            _row(target, "automatic", exact=exact, ratio=ratio),
        ])
    return rows


def test_analyzer_advances_exact_fallback():
    result = runner.analyze_rows(_rows())
    assert result["default_eligible"] is True
    assert result["all_prediction_vectors_exact"] is True
    assert result["aggregate_view_ratios"] == {
        "all_held": 1.0,
        "seen_player": 1.0,
        "cold_player": 1.0,
    }


def test_analyzer_rejects_harm_and_inexact_fallback():
    harmed = runner.analyze_rows(_rows(exact=False, ratio=1.01))
    assert harmed["default_eligible"] is False
    assert harmed["all_prediction_vectors_exact"] is False


def test_analyzer_accepts_nonexact_engagement_when_every_view_improves():
    improved = runner.analyze_rows(_rows(exact=False, ratio=0.99))
    assert improved["default_eligible"] is True
    assert improved["all_prediction_vectors_exact"] is False
    assert all(row["pair_safe"] for row in improved["targets"])


def test_analyzer_rejects_incomplete_or_failed_rows():
    with pytest.raises(RuntimeError, match="target census"):
        runner.analyze_rows(_rows()[:-2])

    failed = deepcopy(_rows())
    failed[0]["integrity_passes"] = False
    with pytest.raises(RuntimeError, match="integrity"):
        runner.analyze_rows(failed)
