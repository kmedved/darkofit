from __future__ import annotations

import copy
import json
from pathlib import Path

from benchmarks import tier_d_fresh_power_design as engine
from benchmarks import tier_d_fresh_power_design_v3 as design


ROOT = Path(__file__).resolve().parents[1]


def _contract():
    return json.loads(design.CONTRACT.read_text())


def test_as_built_contract_binds_verified_registry_and_branch_counts():
    contract = _contract()
    design.validate_contract(contract)
    enumeration = json.loads(design.ENUMERATION.read_text())
    eligible = [row for row in enumeration["identities"] if row["status"] == "eligible"]

    assert len(eligible) == 32
    assert contract["panel_template"]["branch_counts"] == {
        "depth_4": 17,
        "depth_8": 15,
    }
    assert contract["panel_template"]["group_safe_lineages"] == 3
    assert contract["verified_registry"]["eligible_lineage_ids"] == [
        row["lineage_id"] for row in eligible
    ]
    assert not any(contract["authorization"].values())


def test_as_built_power_engine_covers_both_branches_with_small_simulation():
    contract = copy.deepcopy(_contract())
    contract["simulation"].update(
        {
            "outer_panel_simulations": 20,
            "lineage_bootstrap_replicates": 30,
            "outer_batch": 10,
        }
    )
    effect = engine.derive_effect_model(contract)
    result = engine.simulate_scenario(
        contract,
        effect,
        retained_fraction=contract["effect_scenario"][
            "primary_retained_log_effect_fraction"
        ],
    )

    assert result["outer_panel_simulations"] == 20
    assert "each_branch_direction" in result["component_pass_probability"]
    assert result["metric_percentiles"]["branch_depth_4"]
    assert result["metric_percentiles"]["branch_depth_8"]
