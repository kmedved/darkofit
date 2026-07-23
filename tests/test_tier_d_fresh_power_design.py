from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import tier_d_fresh_power_design as power


ROOT = Path(__file__).resolve().parents[1]


def test_contract_is_shared_balanced_and_grants_no_authority():
    contract = power.load_contract()
    template = contract["panel_template"]

    assert template["lineage_count"] == 32
    assert template["paired_model_fits"] == 192
    assert {row["lineages"] for row in template["strata"]} == {8}
    assert {
        row["branch"] for row in template["strata"]
    } == {"depth_4", "depth_8"}
    assert {
        row["feature_family"] for row in template["strata"]
    } == {"numeric", "categorical_or_grouped"}
    assert not any(contract["authorization"].values())


def test_contract_binds_exact_spent_sizing_inputs_and_derivation():
    contract = power.load_contract()
    model = power.derive_effect_model(contract)

    assert model["source_lineage_count"] == 5
    assert model["source_coordinate_count"] == 21
    assert model["source_lineage_ratios"] == pytest.approx(
        [
            1.0111235759924502,
            0.9211784578411847,
            0.9720275932640814,
            0.9236046298511094,
            0.9558090968676863,
        ]
    )
    assert model["source_geometric_mean_ratio"] == pytest.approx(
        0.9561736534103875
    )


def test_contract_rejects_source_or_gate_drift():
    contract = power.load_contract()
    drifted = copy.deepcopy(contract)
    drifted["quality_gates"]["worst_lineage_ratio_at_most"] = 1.03
    with pytest.raises(RuntimeError, match="quality gates changed"):
        power.validate_contract(drifted)

    drifted = copy.deepcopy(contract)
    drifted["candidate"]["source_commit"] = "0" * 40
    with pytest.raises(RuntimeError, match="candidate source changed"):
        power.validate_contract(drifted)


def _bootstrap_counts(lineages: int, replicates: int = 500) -> np.ndarray:
    rng = np.random.default_rng(17)
    return rng.multinomial(
        lineages,
        np.full(lineages, 1.0 / lineages),
        size=replicates,
    ).astype(np.float64)


def test_panel_evaluator_passes_uniform_help_and_catches_harm():
    contract = power.load_contract()
    labels = ["depth_4"] * 4 + ["depth_8"] * 4
    good = np.full((1, 8), np.log(0.99))
    evaluated = power.evaluate_panel_logs(
        good,
        labels,
        _bootstrap_counts(8),
        contract["quality_gates"],
    )
    assert evaluated["passes"].tolist() == [True]

    harmful = good.copy()
    harmful[0, 0] = np.log(1.021)
    evaluated = power.evaluate_panel_logs(
        harmful,
        labels,
        _bootstrap_counts(8),
        contract["quality_gates"],
    )
    assert not evaluated["component_worst_lineage"][0]
    assert not evaluated["passes"][0]


def test_panel_evaluator_prevents_one_branch_from_carrying_the_other():
    contract = power.load_contract()
    labels = ["depth_4"] * 4 + ["depth_8"] * 4
    logs = np.log(
        np.asarray([[0.97] * 4 + [1.004] * 4], dtype=np.float64)
    )
    evaluated = power.evaluate_panel_logs(
        logs,
        labels,
        _bootstrap_counts(8),
        contract["quality_gates"],
    )

    assert evaluated["point"][0] < 0.995
    assert evaluated["branch_depth_8"][0] > 1.0
    assert not evaluated["component_each_branch_direction"][0]
    assert not evaluated["passes"][0]


def test_wilson_lower_is_one_sided_and_below_point_estimate():
    lower = power._wilson_lower(4500, 5000, 0.95)
    assert 0.89 < lower < 0.90
    assert lower < 0.90


def test_result_validator_rejects_authority_and_self_hash_tampering():
    result = {
        "primary_scenario": {"power_floor_passes": True},
        "power_qualified": True,
        "fresh_access_authorized": False,
        "registry_build_authorized": False,
        "confirmation_run_authorized": False,
        "candidate_merge_authorized": False,
        "default_change_authorized": False,
        "release_authorized": False,
        "lockbox_access_authorized": False,
    }
    result["result_sha256"] = power.provenance.canonical_json_sha256(result)
    power.validate_result(result)

    authorized = copy.deepcopy(result)
    authorized["fresh_access_authorized"] = True
    authorized.pop("result_sha256")
    authorized["result_sha256"] = power.provenance.canonical_json_sha256(
        authorized
    )
    with pytest.raises(RuntimeError, match="forbidden authority"):
        power.validate_result(authorized)

    tampered = copy.deepcopy(result)
    tampered["power_qualified"] = False
    with pytest.raises(RuntimeError, match="self-hash mismatch"):
        power.validate_result(tampered)


def test_cli_paths_are_canonical_and_create_only(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="contract path changed"):
        power.main(["--contract", str(tmp_path / "contract.json")])
    with pytest.raises(RuntimeError, match="output path changed"):
        power.main(["--output", str(tmp_path / "result.json")])

    monkeypatch.setattr(power, "DEFAULT_OUTPUT", tmp_path / "existing.json")
    power.DEFAULT_OUTPUT.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="create-only"):
        power.main([])


def test_contract_json_has_no_prospective_dataset_identity():
    contract = json.loads(power.CONTRACT.read_text(encoding="utf-8"))

    def keys(value):
        if isinstance(value, dict):
            return set(value).union(
                *(keys(child) for child in value.values())
            )
        if isinstance(value, list):
            return set().union(*(keys(child) for child in value))
        return set()

    observed = keys(contract["panel_template"])
    assert "task_id" not in observed
    assert "dataset_id" not in observed
    assert "target" not in observed
    assert "target_name" not in observed
