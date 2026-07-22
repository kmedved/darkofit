import sys
import warnings
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import run_v011_ensemble_evidence as v1  # noqa: E402
import run_v011_ensemble_evidence_v2 as v2  # noqa: E402


def test_v2_changes_only_identity_bindings_and_warmup_warning_capture():
    assert v2.CONTRACT_ID == "v011-private-ensemble-evidence-v2"
    assert v2.execution_spec() == {
        **v1.execution_spec(),
        "contract_id": v2.CONTRACT_ID,
    }
    assert v2.uncertainty_spec() == v1.uncertainty_spec()
    assert v2.claim_spec() == v1.claim_spec()
    assert v2.DARKOFIT_HEAD == v1.DARKOFIT_HEAD
    assert v2.CHIMERABOOST_HEAD == v1.CHIMERABOOST_HEAD
    assert v2.CATBOOST_VERSION == v1.CATBOOST_VERSION
    assert "v1_contract" in v2.BOUND_PATHS


def test_v2_warmup_captures_warning_without_hiding_exception(monkeypatch):
    def noisy(*args, **kwargs):
        warnings.warn("expected warmup warning", UserWarning)
        return "ok"

    monkeypatch.setattr(v2, "_v1_warmup", noisy)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert v2._warmup(None) == "ok"
    assert caught == []

    def broken(*args, **kwargs):
        warnings.warn("before failure", UserWarning)
        raise RuntimeError("warmup failed")

    monkeypatch.setattr(v2, "_v1_warmup", broken)
    with pytest.raises(RuntimeError, match="warmup failed"):
        v2._warmup(None)


def test_v2_protocol_records_pre_execution_retirement():
    text = v2.PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "retired before execution" in text
    assert "No v1 formal worker ran" in text
    assert "sole v2 amendment" in text
    assert "Warnings raised by the formal fit remain captured" in text


def test_v2_contract_loads_when_present():
    if not v2.CONTRACT_PATH.exists():
        pytest.skip("prospective v2 contract has not been frozen yet")
    contract = v2.load_contract()
    assert contract["contract_id"] == v2.CONTRACT_ID
    assert contract["attempt_lineage"] == {
        "v1_formal_workers_started": 0,
        "v1_outcomes_opened": False,
        "v1_raw_or_terminal_published": False,
        "sole_amendment": "capture and discard unmeasured warmup warnings",
    }
