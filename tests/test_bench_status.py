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
    assert status["fused_subset"]["all_exact"] is True
    assert status["fused_subset"]["tier_e_dispatch_shipped"] is True


def test_prediction_status_disambiguates_legacy_counter():
    prediction = bench_status.build_status()["prediction"]

    assert prediction["case_count"] == 8
    assert prediction["median_at_or_below_chimera_count"] == 8
    assert prediction["stable_and_at_or_below_chimera_count"] == 6
    assert prediction["legacy_counter_name_is_ambiguous"] is True
    assert not prediction["certified"]


def test_committed_status_outputs_are_current():
    status = bench_status.build_status()
    expected_json, expected_markdown, expected_measurements = (
        bench_status._serialized(status)
    )

    assert bench_status.OUTPUT_JSON.read_text(encoding="utf-8") == expected_json
    assert (
        bench_status.OUTPUT_MARKDOWN.read_text(encoding="utf-8")
        == expected_markdown
    )
    assert (
        bench_status.OUTPUT_MEASUREMENTS.read_text(encoding="utf-8")
        == expected_measurements
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


@pytest.mark.parametrize("value", [True, "1.0"])
def test_numeric_evidence_rejects_non_numeric_domain_types(value):
    with pytest.raises(ValueError, match="finite and positive"):
        bench_status._finite_positive(value, "test value")
    with pytest.raises(ValueError, match="must be finite"):
        bench_status._finite(value, "test value")


def test_source_hashes_bind_and_fail_closed(
    tmp_path, monkeypatch
):
    status = bench_status.build_status()
    assert set(status["sources"]) == set(bench_status.EXPECTED_SOURCE_SHA256)
    for name, source in status["sources"].items():
        path = bench_status.ROOT / Path(source["path"])
        assert path.is_file()
        assert source["sha256"] == bench_status.EXPECTED_SOURCE_SHA256[name]

    changed = tmp_path / bench_status.TABARENA_SUMMARY.name
    changed.write_bytes(bench_status.TABARENA_SUMMARY.read_bytes() + b"\n")
    monkeypatch.setattr(bench_status, "TABARENA_SUMMARY", changed)

    with pytest.raises(
        ValueError, match="general_panel frozen source hash changed"
    ):
        bench_status.build_status()
