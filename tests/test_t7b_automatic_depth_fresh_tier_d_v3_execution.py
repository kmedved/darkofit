from __future__ import annotations

import json

import pytest

from benchmarks import analyze_t7b_automatic_depth_fresh_tier_d_v3 as analyzer
from benchmarks import run_t7b_automatic_depth_fresh_tier_d_v3 as runner


def _contract():
    return json.loads(runner.CONTRACT.read_text())


def test_v3_execution_contract_binds_fillable_registry_and_qualified_power():
    contract = _contract()
    runner.validate_contract(contract)
    preflight = runner.build_preflight()

    assert preflight["status"] == "preflight_passed"
    assert preflight["active_lineage_count"] == 32
    assert preflight["active_branch_counts"] == {"depth_4": 17, "depth_8": 15}
    assert preflight["active_group_safe_count"] == 3
    assert preflight["attestations"]["no_new_data_access"] is True
    assert preflight["attestations"]["fresh_inspection_spent"] is False
    assert [row["lineage_id"] for row in preflight["active_lineages"]] == contract[
        "verified_registry"
    ]["eligible_lineage_ids"]


def test_frozen_contract_cannot_execute_without_later_owner_record():
    contract = _contract()
    authorization = {
        "schema_version": 1,
        "authorization_id": (
            "t7b-automatic-depth-fresh-tier-d-v3-owner-run-authorization-v1"
        ),
        "contract_id": runner.CONTRACT_ID,
        "execution_contract_sha256": runner.file_sha256(runner.CONTRACT),
        "enumeration_sha256": runner.file_sha256(runner.ENUMERATION),
        "power_result_sha256": runner.file_sha256(runner.POWER_RESULT),
        "confirmation_run_authorized": True,
        "candidate_modification_authorized": False,
        "panel_change_authorized": False,
        "gate_change_authorized": False,
        "rerun_authorized": False,
        "partial_read_authorized": False,
        "tabarena_authorized": False,
        "ctr23_authorized": False,
        "lockbox_authorized": False,
        "release_publication_authorized": False,
    }
    runner.validate_owner_authorization(authorization, contract)

    authorization["confirmation_run_authorized"] = False
    with pytest.raises(RuntimeError, match="absent or changed"):
        runner.validate_owner_authorization(authorization, contract)


def test_v3_analyzer_binds_v3_contract_and_power_paths():
    assert analyzer.CONTRACT == runner.CONTRACT
    assert analyzer.POWER_CONTRACT == runner.POWER_CONTRACT


def test_v3_one_shot_is_terminally_closed_without_quality_verdict():
    authorization_path = (
        runner.ROOT
        / "benchmarks"
        / "t7b_automatic_depth_fresh_tier_d_v3_owner_run_authorization_20260723.json"
    )
    launch_path = (
        runner.ROOT
        / "benchmarks"
        / "t7b_automatic_depth_fresh_tier_d_v3_launch_manifest_20260723.json"
    )
    terminal_path = (
        runner.ROOT
        / "benchmarks"
        / "t7b_automatic_depth_fresh_tier_d_v3_terminal_failure_20260723.json"
    )
    authorization = json.loads(authorization_path.read_text())
    runner.validate_owner_authorization(authorization, _contract())
    launch = json.loads(launch_path.read_text())
    terminal = json.loads(terminal_path.read_text())

    assert launch["sole_inspection_spent"] is True
    assert launch["no_rerun"] is True
    assert launch["partial_reads_forbidden"] is True
    assert terminal["status"] == "terminal_failed_after_launch"
    assert terminal["rerun_authorized"] is False
    assert terminal["completed_rows_unpublished_and_unread"] == 1
    assert terminal["launch_sha256"] == runner.file_sha256(launch_path)
    assert "'branch': 'depth_8'" in terminal["error"]
    assert "'fitted_depth': 6" in terminal["error"]
