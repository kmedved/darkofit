from copy import deepcopy

import pytest

from benchmarks import (
    analyze_automatic_linear_selector_v3_ctr23_ship_check as analyzer,
)
from benchmarks import run_automatic_linear_selector_v3_ctr23_ship_check as runner


def _row(task, fold, arm, ratio=1.0, selected=False):
    selector = None
    if arm == "automatic":
        selector = {
            "reason": "selected_linear" if selected else "gain_not_above_noise",
            "resolved_linear_leaves": selected,
        }
    rmse = 2.0 * ratio if arm == "automatic" else 2.0
    return {
        "arm": arm,
        "task_id": task,
        "dataset_id": task + 100,
        "dataset_name": f"task-{task}",
        "fold": fold,
        "train_rows": 100,
        "test_rows": 50,
        "input_features": 4,
        "train_index_sha256": f"train-{task}-{fold}",
        "test_index_sha256": f"test-{task}-{fold}",
        "integrity_passes": True,
        "rmse": rmse,
        "prediction_sha256": (
            f"linear-{task}-{fold}"
            if arm == "automatic" and selected
            else f"constant-{task}-{fold}"
        ),
        "fit_seconds": 2.0 if arm == "automatic" else 1.0,
        "predict_seconds": 1.0,
        "peak_process_tree_rss_bytes": 200 if arm == "automatic" else 100,
        "automatic_linear_selector": selector,
    }


def _raw(*, selected_task=None, selected_ratio=0.95):
    rows = []
    for task in range(9):
        for fold in range(3):
            selected = task == selected_task
            ratio = selected_ratio if selected else 1.0
            rows.extend([
                _row(task, fold, "control"),
                _row(task, fold, "automatic", ratio, selected),
            ])
    return {
        "ship_check_id": runner.SHIP_CHECK_ID,
        "complete": True,
        "rows": rows,
    }


def test_manifest_reuses_exact_ctr23_snapshot_membership():
    manifest = runner.build_manifest()
    assert manifest["ship_check_id"] == runner.SHIP_CHECK_ID
    assert manifest["task_count"] == 9
    assert len(manifest["tasks"]) == 9
    assert all(len(task["coordinates"]) == 3 for task in manifest["tasks"])
    assert manifest["holdout"] == "CTR23 observed release-validation"


def test_analyzer_advances_nonharming_selector():
    result = analyzer.analyze(_raw(selected_task=3, selected_ratio=0.95))
    assert result["default_eligible_on_ctr23"] is True
    assert result["disposition"] == (
        "ready_for_untouched_sports_season_ship_check"
    )
    assert result["quality"]["worst_task_ratio"] == 1.0
    assert result["selector"]["selected_pairs"] == 3


def test_analyzer_rejects_any_task_level_harm():
    result = analyzer.analyze(_raw(selected_task=3, selected_ratio=1.01))
    assert result["default_eligible_on_ctr23"] is False
    assert result["disposition"] == "keep_explicit_opt_in"
    assert result["quality"]["worst_task_ratio"] == pytest.approx(1.01)


def test_analyzer_rejects_duplicates_and_integrity_failures():
    duplicated = _raw()
    duplicated["rows"].append(deepcopy(duplicated["rows"][0]))
    with pytest.raises(RuntimeError, match="incomplete"):
        analyzer.analyze(duplicated)

    failed = _raw()
    failed["rows"][0]["integrity_passes"] = False
    with pytest.raises(RuntimeError, match="integrity"):
        analyzer.analyze(failed)

    inexact_fallback = _raw()
    inexact_fallback["rows"][1]["prediction_sha256"] = "different"
    with pytest.raises(RuntimeError, match="fall back exactly"):
        analyzer.analyze(inexact_fallback)
