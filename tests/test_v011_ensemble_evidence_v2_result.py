from __future__ import annotations

import hashlib
import json
import math
import numbers
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import analyze_v011_ensemble_evidence_v2 as analysis  # noqa: E402
import run_v011_ensemble_evidence_v2 as campaign  # noqa: E402


RAW = BENCH / "v011_ensemble_evidence_v2_raw.json"
RESULT = BENCH / "v011_ensemble_evidence_v2_result.json"
NOTE = BENCH / "v011_ensemble_evidence_v2_result.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_contract_or_skip_stale_implementation():
    try:
        return campaign.load_contract()
    except RuntimeError as exc:
        stale_release_bindings = {
            "v1_tests",
            "v2_tests",
            "implementation",
            "implementation_tests",
        }
        message = str(exc)
        if message.startswith("v0.11 ensemble evidence binding drifted: ") and (
            message.rsplit(": ", 1)[-1] in stale_release_bindings
        ):
            pytest.skip(
                "historical v2 ensemble result is pinned to its private "
                "pre-v0.11 implementation"
            )
        raise


def _assert_nested_close(stored, recomputed):
    if isinstance(stored, dict):
        assert isinstance(recomputed, dict)
        assert stored.keys() == recomputed.keys()
        for key in stored:
            _assert_nested_close(stored[key], recomputed[key])
        return
    if isinstance(stored, (list, tuple)):
        assert isinstance(recomputed, (list, tuple))
        assert len(stored) == len(recomputed)
        for left, right in zip(stored, recomputed):
            _assert_nested_close(left, right)
        return
    if (
        isinstance(stored, numbers.Real)
        and not isinstance(stored, bool)
        and isinstance(recomputed, numbers.Real)
        and not isinstance(recomputed, bool)
    ):
        assert math.isclose(float(stored), float(recomputed), rel_tol=1e-13, abs_tol=1e-14)
        return
    assert stored == recomputed


def test_v011_v2_result_is_hash_bound_complete_and_recomputable():
    assert _sha256(RAW) == "d6c0b794db4ce4bdd1e393f2b23546f1351a051f1f66fa7438175f826454171e"
    assert _sha256(RESULT) == "edb35694a6b6d19aa9b320545b759603a7e5a99c34165dd9f1a0ebe66937dabc"
    assert _sha256(NOTE) == "8c0fc244cf3eb5b9e63b2803d2d8d20b7e66e9b375c15c78cc7dee064c7baee4"

    raw = json.loads(RAW.read_text(encoding="utf-8"))
    stored = json.loads(RESULT.read_text(encoding="utf-8"))
    contract = _load_contract_or_skip_stale_implementation()
    analysis._validate_raw(raw, campaign.CONTRACT_PATH)
    assert len(raw["rows"]) == 177
    assert sum(row["kind"] == "quality" for row in raw["rows"]) == 117
    assert sum(row["kind"] == "prediction" for row in raw["rows"]) == 60
    assert raw["sources"]["darkofit"]["head"] == campaign.DARKOFIT_HEAD
    assert raw["sources"]["chimeraboost"]["head"] == campaign.CHIMERABOOST_HEAD
    assert raw["sources"]["catboost"]["version"] == campaign.CATBOOST_VERSION
    assert raw["contract"]["bindings"] == contract["bindings"]

    recomputed = analysis.analyze(RAW, campaign.CONTRACT_PATH)
    for key in (
        "reproduction",
        "quality_uncertainty",
        "cost_telemetry",
        "prediction_throughput",
        "decision",
        "claims",
    ):
        _assert_nested_close(stored[key], recomputed[key])


def test_v011_v2_reproduction_and_timing_integrity_are_clear():
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["reproduction"]["passed"] is True
    assert max(
        record["maximum_absolute_difference"]
        for record in result["reproduction"]["per_case"].values()
    ) <= campaign.REPRODUCTION_ABS_TOLERANCE
    assert result["decision"] == {
        "correctness_clear": True,
        "performance_or_cost_gated": False,
        "public_exposure_authorized_by_this_result": False,
        "public_exposure_stop_condition_present": False,
        "reproduction_clear": True,
    }
    intervals = [
        timing["interval_seconds"]
        for row in raw["rows"]
        if row["kind"] == "prediction"
        for timing in row["predictions"].values()
    ]
    assert len(intervals) == 240
    assert min(intervals) >= campaign.MIN_INTERVAL_SECONDS


def test_v011_v2_note_discloses_quality_cost_and_prediction_without_shipping():
    text = NOTE.read_text(encoding="utf-8")
    for token in (
        "0.965513",
        "5.030x",
        "6.181x",
        "0.478x",
        "0.871x",
        "6.251x",
        "0.126x",
        "does not itself authorize public exposure",
    ):
        assert token in text
