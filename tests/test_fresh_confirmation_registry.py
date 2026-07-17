from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from benchmarks import build_fresh_confirmation_registry as registry


RECORDED_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "fresh_confirmation_registry.json"
)
EXPECTED_ARTIFACT_SHA256 = (
    "37799ed0b788af3c1d69c8f0f7cf37a656fde998ce2d54b5c4a2196c369df4c3"
)


def test_name_hit_is_conservative_and_normalized():
    names = ["bike_sharing_demand", "airfoil_self_noise"]

    assert registry._name_hit("Bike-Sharing-Demand", names) == (
        "bike_sharing_demand"
    )
    assert registry._name_hit(
        "new-bike-sharing-demand-version", names
    ) == "bike_sharing_demand"
    assert registry._name_hit("new_unrelated_source", names) is None


def test_power_analysis_is_deterministic_and_requires_all_gates():
    per_dataset = {
        str(index): {"split_ratios": [0.9] * 7}
        for index in range(3)
    }
    artifact = {
        "analysis": {
            "contrasts": {
                "selector_over_default": {"per_dataset": per_dataset}
            }
        }
    }

    first = registry._power_analysis(artifact)
    second = registry._power_analysis(copy.deepcopy(artifact))

    assert first == second
    assert first["passes"] is True
    assert first["pass_probability"] == 1.0


def test_repository_grep_does_not_treat_numeric_substrings_as_task_ids():
    task = {
        "openml_task_id": 363132,
        "openml_dataset_id": 45718,
        "dataset_name": "3D_Estimation_using_RSSI_of_WLAN_dataset",
    }

    assert registry._repository_literals(task) == (
        "3D_Estimation_using_RSSI_of_WLAN_dataset",
    )


def test_recorded_registry_authorizes_only_fresh_confirmation():
    raw = RECORDED_ARTIFACT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_ARTIFACT_SHA256
    artifact = json.loads(raw)

    assert artifact["sources"]["darkofit_execution_head"] == (
        "8664f7c1102dddfb3fab834869f4c2db4c001858"
    )
    assert artifact["sources"]["darkofit_prefreeze_head"] == (
        "a3c55d6cfcc62d951b17db7c9be53c1ff0ce6137"
    )
    assert artifact["sources"]["chimeraboost_head"] == (
        "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d"
    )
    assert artifact["builder_source_sha256"] == hashlib.sha256(
        Path(registry.__file__).read_bytes()
    ).hexdigest()
    assert artifact["protocol_sha256"] == hashlib.sha256(
        registry.PROTOCOL.read_bytes()
    ).hexdigest()
    assert artifact["declarations_sha256"] == hashlib.sha256(
        registry.DECLARATIONS.read_bytes()
    ).hexdigest()
    assert artifact["task_count"] == artifact["lineage_count"] == 20
    assert artifact["stratum_counts"] == {
        "categorical": 3,
        "noisy_tabular": 3,
        "smooth_numeric": 14,
    }
    assert artifact["coordinate_count"] == 60
    assert all(row["status"] == "eligible" for row in artifact["tasks"])
    assert all(not row["exclusion_reasons"] for row in artifact["tasks"])
    assert artifact["power_analysis"]["pass_probability"] == 0.999965
    assert artifact["power_analysis"]["passes"] is True
    assert artifact["confirmation_data_scored"] is False
    assert artifact["selector_promotion_authorized"] is False
    assert artifact["lockbox_run_authorized"] is False
