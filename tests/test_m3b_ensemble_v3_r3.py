"""Invariant tests for the corrected-source M3b attempt 3."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from benchmarks import analyze_m3b_ensemble_v3_r3 as analyzer
from benchmarks import run_m3b_ensemble_v3_r2 as attempt2_runner
from benchmarks import run_m3b_ensemble_v3_r3 as runner


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(runner.CONTRACT_PATH.read_text(encoding="utf-8"))
PINNED_SOURCE_PATHS = frozenset(
    {"darkofit/sklearn_api.py", "tests/test_private_ensemble_v3.py"}
)


@pytest.fixture(autouse=True)
def _validate_closed_campaign_from_its_historical_commits(monkeypatch):
    """Closed evidence binds Git objects, not the evolving worktree."""

    def historical_bound_file_ok(record):
        relative = record.get("path")
        commit = (
            runner.MODEL_SOURCE_HEAD
            if relative in PINNED_SOURCE_PATHS
            else CONTRACT["sources"]["harness"]
        )
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        payload = completed.stdout
        expected_bytes = record.get("bytes")
        expected_digest = record.get("sha256")
        return (
            completed.returncode == 0
            and not isinstance(expected_bytes, bool)
            and isinstance(expected_bytes, int)
            and expected_bytes == len(payload)
            and isinstance(expected_digest, str)
            and len(expected_digest) == 64
            and expected_digest == hashlib.sha256(payload).hexdigest()
        )

    monkeypatch.setattr(runner._base, "_bound_file_ok", historical_bound_file_ok)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_m3b_r3_has_distinct_identity_without_mutating_attempt2():
    assert attempt2_runner.CONTRACT_NAME == "wave2_m3b_ensemble_v3_r2_20260720"
    assert runner.CONTRACT_NAME == "wave2_m3b_ensemble_v3_r3_20260720"
    assert runner.CONTRACT_PATH != attempt2_runner.CONTRACT_PATH
    assert Path(runner._base.__file__).resolve() == Path(runner.__file__).resolve()


def test_m3b_r3_preserves_grid_rules_and_rss_scope():
    assert runner.case_specs() == attempt2_runner.case_specs()
    assert runner.quality_orders() == attempt2_runner.quality_orders()
    assert runner.decision_rules() == attempt2_runner.decision_rules()
    assert {arm: runner.arm_config(arm) for arm in runner.ARMS} == {
        arm: attempt2_runner.arm_config(arm) for arm in attempt2_runner.ARMS
    }
    assert runner.RSS_SCOPE == attempt2_runner.RSS_SCOPE
    execution = runner.execution_contract()
    assert execution["attempt2_terminal_failure_bound"] is True
    assert execution["group_bootstrap_loader_fix_bound"] is True


def test_m3b_attempt2_failure_record_is_bound_and_uninspected():
    contract_path = ROOT / "benchmarks" / "m3b_ensemble_v3_r2_contract.json"
    record_path = ROOT / "benchmarks" / "m3b_ensemble_v3_attempt2_failure_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))

    assert record["failed_contract"]["sha256"] == _sha256(contract_path)
    assert record["terminal_artifact"]["completed_rows_discarded"] == 1
    assert record["terminal_artifact"]["rows"] is None
    assert record["failure"]["completed_results_published"] is False
    assert record["failure"]["completed_results_inspected"] is False
    assert record["disposition"]["rerun_same_identity"] is False


def test_m3b_r3_uses_corrected_source_pin():
    assert runner.MODEL_SOURCE_HEAD == "6d063f98128d457f8b8bbf610c7aec46e675d844"
    for relative in PINNED_SOURCE_PATHS:
        record = next(
            value
            for value in CONTRACT["bound_files"].values()
            if value["path"] == relative
        )
        payload = subprocess.run(
            ["git", "show", f"{runner.MODEL_SOURCE_HEAD}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert len(payload) == record["bytes"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]


def test_m3b_r3_analyzer_routes_to_attempt3_runner():
    assert analyzer.runner is runner
    assert analyzer._foundation.runner is runner
    assert analyzer._foundation._base.runner is runner


def test_m3b_r3_freezer_binds_both_terminal_predecessors():
    contract = CONTRACT
    attempt2 = json.loads(
        (ROOT / "benchmarks" / "m3b_ensemble_v3_r2_contract.json").read_text(
            encoding="utf-8"
        )
    )

    assert contract["sources"]["darkofit"] == runner.MODEL_SOURCE_HEAD
    assert contract["cases"] == attempt2["cases"]
    assert contract["case_manifests"] == attempt2["case_manifests"]
    assert contract["decision_rules"] == attempt2["decision_rules"]
    assert contract["attempt_lineage"]["attempt1_completed_rows"] == 0
    assert contract["attempt_lineage"]["attempt2_completed_rows_discarded"] == 1
    assert contract["attempt_lineage"]["attempt2_results_inspected"] is False
    assert contract["attempt_lineage"]["attempt2_rerun"] is False
    assert set(contract["bound_files"]) == set(runner.BOUND_PATHS)


def test_m3b_r3_frozen_contract_loads_when_present():
    if not runner.CONTRACT_PATH.exists():
        pytest.skip("attempt-3 contract is created after the harness commit")
    contract = runner.load_contract()
    assert contract["name"] == runner.CONTRACT_NAME
    assert contract["sources"]["darkofit"] == runner.MODEL_SOURCE_HEAD
