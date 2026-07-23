from __future__ import annotations

import pytest

from benchmarks import analyze_t7b_automatic_depth_development_v1 as analyzer
from benchmarks import run_t7b_automatic_depth_development_v1 as runner


def _worker_row(*, depth=6, branch="middle_density"):
    return {
        "status": "integrity_failed",
        "contract_id": "historical",
        "arm": "candidate",
        "branch": "depth_8",
        "fitted_depth": depth,
        "automatic_depth_policy": {
            "branch": branch,
            "rule": "scalar_rmse_catboost_n_eff_per_input_feature_4_6_8",
        },
        "rmse": 1.0,
        "fit_seconds": 1.0,
        "predict_seconds_repeats": [0.1, 0.1, 0.1],
        "safe_npz_exact": True,
        "ambient_thread_restored": True,
    }


def _synthetic_raw():
    rows = []
    for lineage_index in range(32):
        branch = "depth_4" if lineage_index < 16 else "depth_8"
        for coordinate in range(3):
            common = {
                "status": "ok",
                "lineage_id": f"lineage_{lineage_index:02d}",
                "slot": f"slot_{lineage_index:02d}",
                "stratum": (
                    "low_density_numeric"
                    if branch == "depth_4"
                    else "high_density_numeric"
                ),
                "branch": branch,
                "coordinate": coordinate,
                "weight_mode": "nonuniform" if coordinate == 1 else "ordinary",
                "task_id": 10_000 + lineage_index,
                "dataset_id": 20_000 + lineage_index,
                "split_sha256": f"split-{lineage_index}-{coordinate}",
                "train_rows": 1000,
                "test_rows": 250,
                "input_features": 10,
                "predict_seconds_repeats": [1.0, 1.01, 0.99],
                "peak_process_tree_rss_bytes": 200_000_000,
                "integrity_passes": True,
                "ambient_thread_restored": True,
                "safe_npz_exact": True,
            }
            rows.append(
                {
                    **common,
                    "arm": "control",
                    "rmse": 1.0,
                    "fit_seconds": 2.0,
                }
            )
            rows.append(
                {
                    **common,
                    "arm": "candidate",
                    "rmse": 0.98,
                    "fit_seconds": 1.8,
                    "predict_seconds_repeats": [0.9, 0.91, 0.89],
                    "peak_process_tree_rss_bytes": 195_000_000,
                }
            )
    return {
        "schema_version": 1,
        "benchmark_id": runner.BENCHMARK_ID,
        "complete": True,
        "environment": {"physical_memory_bytes": 64 * 1024**3},
        "rows": rows,
    }


def test_worker_accepts_policy_resolved_middle_depth_despite_panel_label(monkeypatch):
    monkeypatch.setattr(runner.legacy, "run_worker", lambda *args, **kwargs: _worker_row())
    row = runner.run_worker(
        {"branch": "depth_8"},
        coordinate=0,
        arm="candidate",
        source=runner.ROOT,
    )

    assert row["status"] == "ok"
    assert row["integrity_passes"] is True
    assert row["panel_branch"] == "depth_8"
    assert row["resolved_depth"] == 6
    assert row["resolved_policy_branch"] == "middle_density"


def test_worker_rejects_inconsistent_recorded_policy(monkeypatch):
    monkeypatch.setattr(
        runner.legacy,
        "run_worker",
        lambda *args, **kwargs: _worker_row(depth=6, branch="high_density"),
    )
    row = runner.run_worker(
        {"branch": "depth_8"},
        coordinate=0,
        arm="candidate",
        source=runner.ROOT,
    )

    assert row["status"] == "integrity_failed"
    assert row["integrity_passes"] is False


def test_preflight_reuses_all_32_verified_lineages_and_drops_frozen_verdict():
    preflight = runner.build_preflight()

    assert preflight["status"] == "preflight_passed"
    assert preflight["active_lineage_count"] == 32
    assert len(preflight["active_lineages"]) == 32
    assert "contract_id" not in preflight
    assert any("development benchmark" in note for note in preflight["notes"])


def test_development_analysis_is_descriptive_and_records_actual_resolutions():
    raw = _synthetic_raw()
    for row in raw["rows"]:
        row["panel_branch"] = row["branch"]
        if row["arm"] == "candidate":
            row["resolved_depth"] = 4 if row["branch"] == "depth_4" else 6
        else:
            row["resolved_depth"] = 6

    result = analyzer.analyze(raw)

    assert "go" not in result
    assert "disposition" not in result
    assert result["integrity"]["passes"] is True
    assert result["candidate_policy_resolutions"]["all_coordinates"] == {
        "4": 48,
        "6": 48,
    }
    assert "panel_branch_geomean_ratio" in result["quality"]
    assert "branch_geomean_ratio" not in result["quality"]
    assert result["historical_reference_gate_diagnostics"]["note"].startswith(
        "Telemetry only"
    )


def test_output_paths_reject_source_tree():
    with pytest.raises(ValueError, match="outside"):
        runner.output_paths(runner.ROOT / "benchmark-output")
