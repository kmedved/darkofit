"""Contract tests for the warmup-corrected v0.11 M2 successor."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import run_v011_m2_broad_panel_v2 as v2
from benchmarks import run_v011_m2_broad_panel_v3 as campaign


def test_v3_preserves_the_v2_scientific_protocol():
    first = v2.frozen_protocol()
    second = campaign.frozen_protocol()
    successor = second.pop("successor")
    first.pop("successor")
    first_pin = first.pop("darkofit_execution_source_pin")
    second_pin = second.pop("darkofit_execution_source_pin")
    assert first == second
    assert first_pin["only_path_added_after_harness_freeze"].endswith(
        "contract_v2_20260722.json"
    )
    assert second_pin["only_path_added_after_harness_freeze"].endswith(
        "contract_v3_20260722.json"
    )
    assert successor == {
        "supersedes_contract_id": "v011-m2-broad-panel-20260722-v2",
        "reason": "v2_pre_execution_warmup_constant_remained_18",
        "v2_fit_count": 0,
        "v2_completed_worker_count": 0,
        "v2_campaign_artifact_count": 2,
        "scientific_protocol_change": "none",
        "only_harness_change": "bind_warmup_thread_constant_to_frozen_14",
    }


def test_v3_successor_configuration_scopes_warmup_and_inherited_state(monkeypatch):
    base = campaign._v1._base

    class FakeWarmup:
        THREAD_COUNT = 18

        @classmethod
        def _normalize_thread_count(cls, value):
            if value != cls.THREAD_COUNT:
                raise ValueError
            return value

    warmup = FakeWarmup
    monkeypatch.setattr(campaign._v1, "_import_warmup_module", lambda: warmup)
    import_warmup_before = campaign._v1._import_warmup_module
    before = (
        campaign._v1.EXPECTED_CHILD_CPUS,
        campaign._v1.CAMPAIGN_KIND,
        campaign._v1.__file__,
        campaign._v1._OVERRIDES,
        warmup.THREAD_COUNT,
    )
    base_before = (
        base.EXPECTED_CHILD_CPUS,
        base.CAMPAIGN_KIND,
        base.SOURCE_FILES,
        base.protocol_sha256,
    )
    with campaign.configured_successor():
        assert campaign._v1.EXPECTED_CHILD_CPUS == 14
        assert campaign._v1.CAMPAIGN_KIND == campaign.CAMPAIGN_KIND
        assert Path(campaign._v1.__file__).name == "run_v011_m2_broad_panel_v3.py"
        assert campaign._v1._import_warmup_module() is warmup
        assert warmup.THREAD_COUNT == 14
        assert warmup._normalize_thread_count(14) == 14
        with campaign._v1.configured_base():
            assert base.EXPECTED_CHILD_CPUS == 14
            assert base.CAMPAIGN_KIND == campaign.CAMPAIGN_KIND
            assert base.SOURCE_FILES == campaign.SOURCE_FILES
            assert base.protocol_sha256() == campaign.protocol_sha256()
    assert (
        campaign._v1.EXPECTED_CHILD_CPUS,
        campaign._v1.CAMPAIGN_KIND,
        campaign._v1.__file__,
        campaign._v1._OVERRIDES,
        warmup.THREAD_COUNT,
    ) == before
    assert campaign._v1._import_warmup_module is import_warmup_before
    assert (
        base.EXPECTED_CHILD_CPUS,
        base.CAMPAIGN_KIND,
        base.SOURCE_FILES,
        base.protocol_sha256,
    ) == base_before


@pytest.mark.parametrize(
    "relative",
    [
        "benchmarks/run_v011_m2_broad_panel_v3.py",
        "benchmarks/analyze_v011_m2_broad_panel_v3.py",
        "benchmarks/freeze_v011_m2_broad_panel_v3.py",
    ],
)
def test_v3_clis_bootstrap_from_a_clean_direct_invocation(relative, tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", str(campaign.ROOT / relative), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_v3_contract_is_bound_when_frozen():
    if not campaign.CONTRACT_PATH.exists():
        pytest.skip("prospective M2 v3 contract has not been frozen yet")
    contract = campaign.load_contract()
    assert contract["contract_id"] == campaign.CONTRACT_ID
    assert contract["outcome_blind"] is True
    assert contract["protocol"] == campaign.frozen_protocol()
    assert set(contract["bindings"]) == set(campaign.BOUND_PATHS)
