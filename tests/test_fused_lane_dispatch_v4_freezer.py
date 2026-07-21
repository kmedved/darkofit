"""Contract tests for the prospective fused-lane calibration v4 freeze."""

import json

from benchmarks import freeze_fused_lane_dispatch_calibration_v4 as freezer_v4
from benchmarks import run_fused_lane_dispatch as runner


def test_v4_freezer_preserves_science_and_binds_final_repairs(monkeypatch):
    source = "d" * 40
    monkeypatch.setattr(
        runner, "git_state", lambda *_args: {"head": source, "status": ""}
    )

    contract = freezer_v4.build_contract()

    assert contract["source"] == source
    assert contract["execution_identity"] == "calibration_v4"
    assert contract["supersedes"]["contract_sha256"] == (
        freezer_v4.V3_CONTRACT_SHA256
    )
    assert contract["supersedes"]["formal_worker_started"] is False
    assert contract["supersedes"]["outcomes_opened"] is False
    assert contract["supersedes"]["scientific_grid_or_gate_changed"] is False
    assert contract["execution"]["scientific_change_from_v3"] is False
    assert contract["execution"]["parent_pipe_capability_required"] is True
    assert (
        contract["execution"]["authorization_alone_rejected_by_workers"]
        is True
    )
    assert contract["execution"]["production_routing_layout_required"] is True
    assert (
        contract["execution"]["wrapper_booster_kernel_binding_required"]
        is True
    )
    assert (
        contract["execution"]["positive_mass_class_semantics_required"]
        is True
    )
    v3_contract = json.loads(
        (freezer_v4.ROOT / freezer_v4.V3_CONTRACT).read_text(encoding="utf-8")
    )
    assert contract["runtime"]["worker_environments"] == (
        v3_contract["runtime"]["worker_environments"]
    )
    assert contract["generator"] == v3_contract["generator"]
    assert contract["decision_rules"] == v3_contract["decision_rules"]
    assert contract["downstream"] == v3_contract["downstream"]
    assert contract["outputs"]["raw"].endswith("_raw_v4.json")
    assert contract["execution_authorized"] is False
    assert contract["outcomes_opened"] is False
