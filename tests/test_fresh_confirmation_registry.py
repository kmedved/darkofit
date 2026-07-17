from __future__ import annotations

import copy

from benchmarks import build_fresh_confirmation_registry as registry


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
