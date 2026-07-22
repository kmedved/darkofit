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

import analyze_ensemble_v3_characterization as analysis  # noqa: E402
import run_ensemble_v3_characterization as campaign  # noqa: E402


RAW = BENCH / "ensemble_v3_characterization_raw.json"
RESULT = BENCH / "ensemble_v3_characterization_result.json"
NOTE = BENCH / "ensemble_v3_characterization_result.md"
INTERPRETATION = BENCH / "ensemble_v3_characterization_interpretation_20260721.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_contract_or_skip_stale_implementation():
    try:
        return campaign.load_contract()
    except RuntimeError as exc:
        if str(exc) == "characterization binding drifted: implementation":
            pytest.skip(
                "historical characterization is pinned to its pre-v0.11 "
                "implementation"
            )
        raise


def _assert_recomputed_equal(stored, recomputed):
    """Compare derived evidence portably while artifact hashes stay exact."""
    if isinstance(stored, dict):
        assert isinstance(recomputed, dict)
        assert stored.keys() == recomputed.keys()
        for key in stored:
            _assert_recomputed_equal(stored[key], recomputed[key])
        return
    if isinstance(stored, (list, tuple)):
        assert isinstance(recomputed, (list, tuple))
        assert len(stored) == len(recomputed)
        for left, right in zip(stored, recomputed):
            _assert_recomputed_equal(left, right)
        return
    if (
        isinstance(stored, numbers.Real)
        and not isinstance(stored, bool)
        and isinstance(recomputed, numbers.Real)
        and not isinstance(recomputed, bool)
    ):
        assert math.isclose(float(stored), float(recomputed), rel_tol=1e-14, abs_tol=1e-15)
        return
    assert stored == recomputed


def test_characterization_result_is_hash_bound_and_recomputable():
    assert _sha256(RAW) == "005c50a89a06e100aa95cb6a776dd7f67026786de6f261470e808a39f9310a9b"
    assert _sha256(RESULT) == "5cfd7b40382187aebed43798715017e1e2867744c5c40f66a00e935f6acefeed"
    assert _sha256(NOTE) == "bef08bf9f972eba7ebfd9b2f51ce1d42828b9444c6e4697063166351ed21b0e4"

    contract = _load_contract_or_skip_stale_implementation()
    raw = json.loads(RAW.read_text())
    result = json.loads(RESULT.read_text())
    rows = analysis.validate_raw(raw, contract)
    readout = json.loads(analysis.HISTORICAL_READOUT.read_text())
    historical_result = json.loads(analysis.HISTORICAL_RESULT.read_text())
    historical_timing = json.loads(analysis.HISTORICAL_TIMING.read_text())

    _assert_recomputed_equal(result["quality"], analysis.analyze_quality(readout))
    _assert_recomputed_equal(
        result["current_resources"], analysis.analyze_resources(rows)
    )
    _assert_recomputed_equal(result["prediction"], analysis.analyze_prediction(rows))
    _assert_recomputed_equal(
        result["historical_resources"],
        analysis.analyze_historical_resources(historical_result, historical_timing),
    )
    assert result["decision"] == "report_only_await_separate_public_ship_authorization"
    assert not any(
        (
            result["claims"]["public_api_or_default_change_authorized"],
            result["claims"]["release_authorized"],
            result["claims"]["m2_or_m4"],
            result["claims"]["fresh_or_lockbox_data_used"],
            result["claims"]["prediction_certified"],
        )
    )


def test_characterization_notes_disclose_cost_and_duration_limit():
    note = NOTE.read_text()
    interpretation = INTERPRETATION.read_text()
    for token in ("6.142x", "1.136x", "8.125x", "6.208x", "Short intervals: 9"):
        assert token in note
    for token in (
        "13/13",
        "6.142053x",
        "3.013607x",
        "Nine of 144",
        "useful repeat-series characterization",
        "certificate.",
        _sha256(RAW),
        _sha256(RESULT),
    ):
        assert token in interpretation
