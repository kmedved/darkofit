"""Contract tests for the prospective fused-lane calibration v3 freeze."""

import json

from benchmarks import freeze_fused_lane_dispatch_calibration_v3 as freezer_v3
from benchmarks import run_fused_lane_dispatch as runner


def test_v3_freezer_preserves_science_and_binds_gate_repairs(monkeypatch):
    source = "c" * 40
    monkeypatch.setattr(
        runner, "git_state", lambda *_args: {"head": source, "status": ""}
    )

    contract = freezer_v3.build_contract()

    assert contract["source"] == source
    assert contract["execution_identity"] == "calibration_v3"
    assert contract["supersedes"]["contract_sha256"] == (
        freezer_v3.V2_CONTRACT_SHA256
    )
    assert contract["supersedes"]["formal_worker_started"] is False
    assert contract["supersedes"]["outcomes_opened"] is False
    assert contract["supersedes"]["scientific_grid_or_gate_changed"] is False
    assert contract["execution"]["scientific_change_from_v2"] is False
    assert contract["execution"]["worker_self_authorization_required"] is True
    assert (
        contract["execution"]["exact_frozen_worker_environment_required"]
        is True
    )
    assert contract["execution"]["actual_builder_counters_required"] is True
    v2_contract = json.loads(
        (freezer_v3.ROOT / freezer_v3.V2_CONTRACT).read_text(encoding="utf-8")
    )
    assert contract["runtime"]["worker_environments"] == (
        v2_contract["runtime"]["worker_environments"]
    )
    assert contract["outputs"]["raw"].endswith("_raw_v3.json")
    assert contract["execution_authorized"] is False
    assert contract["outcomes_opened"] is False
