import json
from pathlib import Path

import pytest

from benchmarks.analyze_m3b_ensemble_v3_r3_vs_single import (
    QUALITY_SHA256,
    RESULT_SHA256,
    build_readout,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
QUALITY = ROOT / "benchmarks" / "m3b_ensemble_v3_r3_quality.json"
RESULT = ROOT / "benchmarks" / "m3b_ensemble_v3_r3_result.json"
READOUT_JSON = (
    ROOT / "benchmarks" / "m3b_ensemble_v3_r3_vs_single_readout_20260721.json"
)
READOUT_MD = (
    ROOT / "benchmarks" / "m3b_ensemble_v3_r3_vs_single_readout_20260721.md"
)


def test_m3b_r3_vs_single_readout_is_hash_bound_and_exact():
    readout = build_readout(QUALITY, RESULT)
    combined = readout["arms_vs_single"]["b1_b2_combined"]

    assert readout["quality_artifact"]["sha256"] == QUALITY_SHA256
    assert readout["frozen_result_artifact"]["sha256"] == RESULT_SHA256
    assert readout["amends_frozen_m3b_result"] is False
    assert readout["sports_primary_scope"].startswith("player-disjoint")
    assert combined["all_case_geometric_mean"] == pytest.approx(
        0.9655130356
    )
    assert combined["sports_geometric_mean"] == pytest.approx(0.9611, abs=5e-5)
    assert combined["wins_vs_single"] == combined["case_count"] == 13
    assert combined["worst_case_ratio"] < 1.0
    assert readout["finding"] == {
        "combined_beats_single_all_cases": True,
        "combined_case_count": 13,
        "combined_median_archive_to_single": 5.534767493867151,
        "frozen_archive_to_single_limit": 4.0,
        "combined_survived_frozen_gate": False,
        "frozen_disposition": "close_b1_b2_preserve_existing_opt_in",
        "serialization_authorized": False,
    }


def test_m3b_r3_vs_single_readout_rejects_unbound_quality_copy(tmp_path):
    payload = json.loads(QUALITY.read_text(encoding="utf-8"))
    payload["rows"][0]["primary_loss"] *= 0.5
    forged = tmp_path / "quality.json"
    forged.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hash differs"):
        build_readout(forged, RESULT)


def test_m3b_r3_vs_single_markdown_preserves_frozen_boundary():
    readout = build_readout(QUALITY, RESULT)
    rendered = render_markdown(readout)
    assert "13/13" in rendered
    assert "player-disjoint cold-player" in rendered
    assert "did not survive the prospectively frozen campaign" in rendered
    assert "No serializer" in rendered
    assert json.loads(READOUT_JSON.read_text(encoding="utf-8")) == readout
    assert READOUT_MD.read_text(encoding="utf-8") == rendered
