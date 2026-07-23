from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import tier_d_fresh_power_design as power
from benchmarks.campaign_lib import provenance


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "benchmarks"
    / "t7b_automatic_depth_fresh_tier_d_power_design_result_20260723.json"
)
RESULT_FILE_SHA256 = (
    "5b767ce0a27e09d479bb18d6314d9adce3bbac78380aeff481639b13152714ad"
)
RESULT_SELF_SHA256 = (
    "735604d24828f6294e60e023ceda053caf272095c50ae83310593833ccdd07d1"
)


def _result():
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_power_result_is_create_only_hash_bound_and_valid():
    result = _result()

    assert provenance.file_sha256(RESULT) == RESULT_FILE_SHA256
    assert result["result_sha256"] == RESULT_SELF_SHA256
    power.validate_result(result)
    assert result["source_head"] == (
        "f895e480fcd2ffc117dc85fd9bd0b9bf0d492414"
    )
    assert result["source_sha256"] == {
        "builder": "f1482d20fbc6ad2f84d4bdc9a338adf4d6d87cb7a4fe640d997aeb9f9ee93fce",
        "contract": "1aa89083b16ed31ee816f005bc961751b3b22785b2ec1ec2a54fc1e2d0d94595",
        "protocol": "c325fe1bc32aea9bd0298394b8179861b4a354a17ee71b0c3227809b2824f012",
        "shipping_policy": "32070321337e8c08da1d2c7e97aa6e8980f3a3c84b52aed1694edcb132ff7e82",
        "tests": "e0f0bb48d1606733f8fb7b21d7f534e429604d519d88d86392217e8382da8405",
    }


def test_primary_power_recomputes_from_frozen_contract_and_inputs():
    result = _result()
    contract = power.load_contract()
    model = power.derive_effect_model(contract)
    recomputed = power.simulate_scenario(
        contract,
        model,
        retained_fraction=contract["effect_scenario"][
            "primary_retained_log_effect_fraction"
        ],
    )

    assert recomputed == result["primary_scenario"]
    assert recomputed["pass_probability"] == pytest.approx(0.998)
    assert recomputed["wilson_lower_bound"] == pytest.approx(
        0.99665735839545
    )
    assert recomputed["power_floor_passes"]


def test_power_result_grants_no_fresh_or_product_authority():
    result = _result()

    assert result["disposition"] == "design_power_qualified"
    assert result["power_qualified"]
    for key in (
        "fresh_access_authorized",
        "registry_build_authorized",
        "confirmation_run_authorized",
        "candidate_merge_authorized",
        "default_change_authorized",
        "release_authorized",
        "lockbox_access_authorized",
    ):
        assert result[key] is False


def test_sensitivity_discloses_the_design_detection_boundary():
    scenarios = {
        row["retained_log_effect_fraction"]: row
        for row in _result()["sensitivity_scenarios"]
    }

    assert scenarios[0.10]["pass_probability"] == pytest.approx(0.2176)
    assert not scenarios[0.10]["power_floor_passes"]
    assert scenarios[0.15]["pass_probability"] == pytest.approx(0.9576)
    assert scenarios[0.15]["power_floor_passes"]
    assert scenarios[0.25]["pass_probability"] == pytest.approx(0.9918)
    assert scenarios[0.25]["power_floor_passes"]
