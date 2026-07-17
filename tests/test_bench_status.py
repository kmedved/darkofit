import json
from pathlib import Path

import pytest

from benchmarks import bench_status


pytestmark = pytest.mark.campaign


def test_release_status_preserves_panel_boundaries_and_decisions():
    status = bench_status.build_status()

    assert status["schema_version"] == 1
    assert "no cross-panel composite" in status["evidence_policy"]
    assert status["general_pareto"]["decision"] == "descriptive_only"
    assert all(row["pareto"] for row in status["general_pareto"]["rows"])

    sports = {
        row["engine"]: row for row in status["sports_pareto"]["rows"]
    }
    assert sports["DarkoFit 0.9.0"]["equal_cell_r2"] > (
        sports["ChimeraBoost 0.15.0"]["equal_cell_r2"]
    )
    assert not sports["DarkoFit 0.9.0"]["pareto"]
    assert sports["ChimeraBoost 0.15.0"]["pareto"]
    assert sports["CatBoost 1.2.10"]["pareto"]

    assert status["large_n"]["fit_speedup"] == pytest.approx(
        1.2792983256567738
    )
    assert not status["large_n"]["certified"]
    assert status["native_ordinal_c2"]["confirmation_run"] is False


def test_prediction_status_disambiguates_legacy_counter():
    prediction = bench_status.build_status()["prediction"]

    assert prediction["case_count"] == 8
    assert prediction["median_at_or_below_chimera_count"] == 8
    assert prediction["stable_and_at_or_below_chimera_count"] == 6
    assert prediction["legacy_counter_name_is_ambiguous"] is True
    assert not prediction["certified"]


def test_committed_status_outputs_are_current():
    status = bench_status.build_status()
    expected_json, expected_markdown = bench_status._serialized(status)

    assert bench_status.OUTPUT_JSON.read_text(encoding="utf-8") == expected_json
    assert (
        bench_status.OUTPUT_MARKDOWN.read_text(encoding="utf-8")
        == expected_markdown
    )
    assert json.loads(expected_json)["sources"]


def test_pareto_rejects_strictly_dominated_row():
    rows = [
        {"engine": "a", "loss": 1.0, "time": 1.0},
        {"engine": "b", "loss": 1.1, "time": 1.1},
        {"engine": "c", "loss": 0.9, "time": 1.2},
    ]
    flags = bench_status._pareto_flags(
        rows, minimize=("loss", "time")
    )

    assert flags == {"a": True, "b": False, "c": True}


def test_source_hashes_bind_existing_repo_relative_files():
    status = bench_status.build_status()
    for source in status["sources"].values():
        path = bench_status.ROOT / Path(source["path"])
        assert path.is_file()
        assert len(source["sha256"]) == 64
