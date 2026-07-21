"""Invariant tests for the zero-outcome M3b attempt-2 successor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import analyze_m3b_ensemble_v3_r2 as analyzer
from benchmarks import run_m3b_ensemble_v3 as attempt1_runner
from benchmarks import run_m3b_ensemble_v3_r2 as runner


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_m3b_r2_does_not_mutate_attempt1_module_identity():
    assert attempt1_runner.CONTRACT_NAME == "wave2_m3b_ensemble_v3_20260720"
    assert runner.CONTRACT_NAME == "wave2_m3b_ensemble_v3_r2_20260720"
    assert attempt1_runner.CONTRACT_PATH != runner.CONTRACT_PATH
    assert Path(runner._base.__file__).resolve() == Path(runner.__file__).resolve()


def test_m3b_r2_rss_sampler_avoids_process_tree_enumeration(monkeypatch):
    import psutil

    def forbidden(*_args, **_kwargs):
        raise AssertionError("attempt-2 RSS sampler enumerated child processes")

    monkeypatch.setattr(psutil.Process, "children", forbidden)
    assert runner.assert_rss_capability() > 0
    with runner.SelfWorkerRSSSampler(interval_seconds=0.001) as sampler:
        payload = bytearray(100_000)
        assert len(payload) == 100_000
    assert sampler.peak_bytes > 0
    assert sampler.samples >= 2
    assert sampler.errors == []


def test_m3b_r2_changes_only_execution_rss_contract():
    assert runner.case_specs() == attempt1_runner.case_specs()
    assert runner.quality_orders() == attempt1_runner.quality_orders()
    assert runner.decision_rules() == attempt1_runner.decision_rules()
    assert {arm: runner.arm_config(arm) for arm in runner.ARMS} == {
        arm: attempt1_runner.arm_config(arm) for arm in attempt1_runner.ARMS
    }
    execution = runner.execution_contract()
    assert execution["rss_scope"] == runner.RSS_SCOPE
    assert execution["rss_capability_preflight"] is True
    assert execution["attempt1_terminal_failure_bound"] is True


def test_m3b_attempt1_failure_record_is_bound_and_zero_outcome():
    contract_path = ROOT / "benchmarks" / "m3b_ensemble_v3_contract.json"
    record_path = ROOT / "benchmarks" / "m3b_ensemble_v3_attempt1_failure_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))

    assert record["failed_contract"]["sha256"] == _sha256(contract_path)
    assert record["terminal_artifact"]["completed_rows_discarded"] == 0
    assert record["terminal_artifact"]["rows"] is None
    assert record["failure"]["model_outcomes_opened"] is False
    assert record["disposition"]["attempt_terminal"] is True
    assert record["disposition"]["rerun_same_identity"] is False


def test_m3b_r2_analyzer_requires_self_worker_rss_scope(monkeypatch):
    artifact = {
        "rss_scope": runner.RSS_SCOPE,
        "rows": [{"rss_scope": runner.RSS_SCOPE}],
    }
    monkeypatch.setattr(
        analyzer,
        "_base_validate_artifact",
        lambda *_args, **_kwargs: artifact,
    )
    assert analyzer.validate_artifact(Path("a"), Path("b")) == artifact

    artifact["rows"][0]["rss_scope"] = "process_tree"
    with pytest.raises(RuntimeError, match="RSS scope provenance"):
        analyzer.validate_artifact(Path("a"), Path("b"))


def test_m3b_r2_frozen_contract_preserved_scientific_grid_and_source_pin():
    contract = json.loads(runner.CONTRACT_PATH.read_text(encoding="utf-8"))
    attempt1 = json.loads(
        (ROOT / "benchmarks" / "m3b_ensemble_v3_contract.json").read_text(
            encoding="utf-8"
        )
    )

    assert contract["sources"]["darkofit"] == attempt1["sources"]["darkofit"]
    assert contract["cases"] == attempt1["cases"]
    assert contract["case_manifests"] == attempt1["case_manifests"]
    assert contract["arms"] == attempt1["arms"]
    assert contract["quality_orders"] == attempt1["quality_orders"]
    assert contract["decision_rules"] == attempt1["decision_rules"]
    assert contract["rss_scope"] == runner.RSS_SCOPE
    assert contract["attempt_lineage"]["attempt1_completed_rows"] == 0
    assert contract["attempt_lineage"]["attempt1_rerun"] is False
    assert set(contract["bound_files"]) == set(runner.BOUND_PATHS)


def test_m3b_r2_contract_is_historical_after_successor_source_fix():
    contract = json.loads(runner.CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["name"] == runner.CONTRACT_NAME
    assert contract["rss_scope"] == runner.RSS_SCOPE
    assert contract["sources"]["darkofit"] == "210434ff09bf54f8356f1268e990af72dc7a5129"
    with pytest.raises(RuntimeError, match="bound file changed"):
        runner.load_contract()
