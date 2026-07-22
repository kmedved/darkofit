"""Contract tests for the 14-CPU v0.11 M2 successor."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import run_v011_m2_broad_panel as v1
from benchmarks import run_v011_m2_broad_panel_v2 as campaign


def test_v2_changes_only_the_common_resource_contract():
    first = v1.frozen_protocol()
    second = campaign.frozen_protocol()
    successor = second.pop("successor")
    first_pin = first.pop("darkofit_execution_source_pin")
    second_pin = second.pop("darkofit_execution_source_pin")
    first_dispatch = first["execution_dispatch"]
    second_dispatch = second["execution_dispatch"]
    resource_basis = second_dispatch.pop("resource_basis")
    first_environment = first_dispatch.pop("worker_environment")
    second_environment = second_dispatch.pop("worker_environment")
    first_cpus = first.pop("num_cpus")
    second_cpus = second.pop("num_cpus")
    assert first == second
    assert first_cpus == 18 and second_cpus == 14
    assert resource_basis == "equal_live_host_maximum_14_logical_physical_active_cpus"
    assert {value for key, value in second_environment.items() if "THREADS" in key} == {
        "14"
    }
    assert {value for key, value in first_environment.items() if "THREADS" in key} == {
        "18"
    }
    assert first_pin["only_path_added_after_harness_freeze"].endswith(
        "contract_20260722.json"
    )
    assert second_pin["only_path_added_after_harness_freeze"].endswith(
        "contract_v2_20260722.json"
    )
    assert successor["only_protocol_change"] == (
        "common_cpu_and_thread_budget_18_to_14"
    )
    assert successor["v1_fit_count"] == successor["v1_campaign_artifact_count"] == 0


def test_v2_successor_configuration_is_scoped_and_selects_its_own_worker_cli():
    base = v1._base
    before = (
        v1.EXPECTED_CHILD_CPUS,
        v1.WORKER_ENVIRONMENT,
        v1.CAMPAIGN_KIND,
        v1.__file__,
        v1._OVERRIDES,
    )
    base_before = (
        base.EXPECTED_CHILD_CPUS,
        base.CAMPAIGN_KIND,
        base.SOURCE_FILES,
        base.protocol_sha256,
    )
    with campaign.configured_successor():
        assert v1.EXPECTED_CHILD_CPUS == 14
        assert v1.WORKER_ENVIRONMENT["NUMBA_NUM_THREADS"] == "14"
        assert v1.CAMPAIGN_KIND == campaign.CAMPAIGN_KIND
        assert Path(v1.__file__).name == "run_v011_m2_broad_panel_v2.py"
        assert v1._OVERRIDES["EXPECTED_CHILD_CPUS"] == 14
        assert campaign.frozen_protocol()["successor"]["supersedes_contract_id"] == (
            "v011-m2-broad-panel-20260722-v1"
        )
        with v1.configured_base():
            assert base.EXPECTED_CHILD_CPUS == 14
            assert base.CAMPAIGN_KIND == campaign.CAMPAIGN_KIND
            assert base.SOURCE_FILES == campaign.SOURCE_FILES
            assert base.protocol_sha256() == campaign.protocol_sha256()
    assert (
        v1.EXPECTED_CHILD_CPUS,
        v1.WORKER_ENVIRONMENT,
        v1.CAMPAIGN_KIND,
        v1.__file__,
        v1._OVERRIDES,
    ) == before
    assert (
        base.EXPECTED_CHILD_CPUS,
        base.CAMPAIGN_KIND,
        base.SOURCE_FILES,
        base.protocol_sha256,
    ) == base_before


@pytest.mark.parametrize(
    "relative",
    [
        "benchmarks/run_v011_m2_broad_panel_v2.py",
        "benchmarks/analyze_v011_m2_broad_panel_v2.py",
        "benchmarks/freeze_v011_m2_broad_panel_v2.py",
    ],
)
def test_v2_clis_bootstrap_from_a_clean_direct_invocation(relative, tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", str(campaign.ROOT / relative), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_v2_contract_is_bound_when_frozen():
    if not campaign.CONTRACT_PATH.exists():
        pytest.skip("prospective M2 v2 contract has not been frozen yet")
    contract = campaign.load_contract()
    assert contract["contract_id"] == campaign.CONTRACT_ID
    assert contract["outcome_blind"] is True
    assert contract["protocol"] == campaign.frozen_protocol()
    assert set(contract["bindings"]) == set(campaign.BOUND_PATHS)
