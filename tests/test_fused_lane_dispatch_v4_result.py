import hashlib
import json
from pathlib import Path

from benchmarks import fused_lane_dispatch_campaign as campaign


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
RAW = BENCHMARKS / "fused_lane_dispatch_calibration_raw_v4.json"
ANALYSIS = BENCHMARKS / "fused_lane_dispatch_calibration_analysis_v4.json"
CONTRACT = BENCHMARKS / "fused_lane_dispatch_calibration_contract_v4.json"
AUTHORIZATION = (
    BENCHMARKS / "fused_lane_dispatch_calibration_authorization_v4.json"
)
RAW_SHA256 = "27a94aa8b93626ec1ae5db329d281b528b52e62beaf0ba3f416d0877a203fea0"
ANALYSIS_SHA256 = (
    "c47314191eaec43e6ceb5fa7a2eca870b7af2308cc736dae23c12b9735f3bf9b"
)
CONTRACT_SHA256 = (
    "fab0784beee165b4643b817f12076b79ff832d95224469bc244cc15c839e9c7f"
)
AUTHORIZATION_SHA256 = (
    "42fb0ab01f8a7b271cda2610c59a953d5815e93657ca0a5ab3a003e38dfea775"
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_calibration_v4_result_is_hash_bound_and_recomputable():
    assert _sha256(RAW) == RAW_SHA256
    assert _sha256(ANALYSIS) == ANALYSIS_SHA256
    assert _sha256(CONTRACT) == CONTRACT_SHA256
    assert _sha256(AUTHORIZATION) == AUTHORIZATION_SHA256

    raw = json.loads(RAW.read_text(encoding="utf-8"))
    stored = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    recomputed = campaign.analyze_calibration(raw["rows"])
    recomputed.update(
        {
            "raw_sha256": RAW_SHA256,
            "source": raw["source"],
            "execution_identity": raw["execution_identity"],
            "execution_contract_sha256": raw["execution_contract_sha256"],
        }
    )

    assert stored == recomputed
    assert len(raw["rows"]) == stored["cell_count"] == 30
    assert stored["all_exact"] is True
    assert stored["all_stable"] is False
    assert sum(
        cell["iqr_over_median"] > 0.10 for cell in stored["cells"]
    ) == 6
    assert stored["selected"] == {
        "geomean_regret": 1.002833289467028,
        "selected_fused_cells": 18,
        "selected_fused_geomean_ratio": 0.9738461173976614,
        "selected_unfused_cells": 12,
        "threshold": 1048576,
        "worst_selected_fused_ratio": 1.0,
    }
    assert stored["qualifies"] is False
    assert stored["disposition"] == "close_dispatch_campaign"
    assert not (
        BENCHMARKS
        / "fused_lane_dispatch_calibration_raw_v4_terminal.json"
    ).exists()


def test_calibration_v4_closeout_docs_preserve_the_binding_disposition():
    plan = (ROOT / "COUNTERPUNCH_PLAN.md").read_text(encoding="utf-8")
    log = (BENCHMARKS / "TESTING_LOG.md").read_text(encoding="utf-8")

    for document in (plan, log):
        assert "0.973846" in document
        assert "close_dispatch_campaign" in document
        assert "no validation" in document.lower()
        assert "`auto`" in document
        assert "fused" in document
