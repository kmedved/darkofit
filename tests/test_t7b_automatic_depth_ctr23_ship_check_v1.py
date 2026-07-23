from __future__ import annotations

from benchmarks import analyze_t7b_automatic_depth_ctr23_ship_check_v1 as analyzer
from benchmarks import run_t7b_automatic_depth_ctr23_ship_check_v1 as runner


def test_manifest_uses_exactly_the_nine_sealed_ctr23_tasks():
    manifest = runner.build_manifest()

    assert manifest["status"] == "ready"
    assert manifest["task_count"] == 9
    assert [row["task_id"] for row in manifest["tasks"]] == [
        361247,
        361253,
        361254,
        361261,
        361264,
        361272,
        361616,
        361617,
        361618,
    ]
    assert all(
        [coordinate["fold"] for coordinate in row["coordinates"]] == [0, 1, 2]
        for row in manifest["tasks"]
    )
    assert all(
        len(coordinate["train_index_sha256"]) == 64
        and len(coordinate["test_index_sha256"]) == 64
        for row in manifest["tasks"]
        for coordinate in row["coordinates"]
    )


def _raw(candidate_ratio=1.0):
    rows = []
    for task_index in range(9):
        for fold in range(3):
            common = {
                "status": "ok",
                "task_id": 100 + task_index,
                "dataset_id": 200 + task_index,
                "dataset_name": f"task-{task_index}",
                "fold": fold,
                "train_rows": 80,
                "test_rows": 20,
                "input_features": 4,
                "train_index_sha256": f"train-{task_index}-{fold}",
                "test_index_sha256": f"test-{task_index}-{fold}",
                "integrity_passes": True,
                "fit_seconds": 2.0,
                "predict_seconds": 1.0,
            }
            rows.append(
                {
                    **common,
                    "arm": "control",
                    "rmse": 1.0,
                    "fitted_depth": 6,
                }
            )
            rows.append(
                {
                    **common,
                    "arm": "candidate",
                    "rmse": candidate_ratio,
                    "fit_seconds": 1.5,
                    "predict_seconds": 0.9,
                    "fitted_depth": 4,
                }
            )
    return {
        "schema_version": 1,
        "ship_check_id": runner.SHIP_CHECK_ID,
        "complete": True,
        "rows": rows,
    }


def test_analyzer_reports_holdout_quality_without_shipping_verdict():
    result = analyzer.analyze(_raw(0.99))

    assert result["quality"]["task_geomean_ratio"] == 0.99
    assert result["quality"]["task_wins"] == 9
    assert result["candidate_depth_counts"] == {"4": 27}
    assert result["integrity"]["passes"] is True
    assert "go" not in result
    assert "disposition" not in result
