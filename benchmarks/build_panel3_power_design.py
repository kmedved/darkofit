#!/usr/bin/env python3
"""Build the immutable Panel 3 power and candidate-retention decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import analyze_panel3_cross_power_calibration as calibration  # noqa: E402
from benchmarks import panel3_registry_common as common  # noqa: E402
from benchmarks.campaign_lib import provenance  # noqa: E402
from benchmarks import run_panel3_confirmation as confirmation  # noqa: E402


CONTRACT = ROOT / "benchmarks" / "panel3_power_design_contract.json"
PROTOCOL = ROOT / "benchmarks" / "panel3_power_design_protocol.md"
DEFAULT_SUMMARY = (
    ROOT / "benchmarks" / "panel3_cross_power_calibration_summary.json"
)
DEFAULT_RAW = (
    ROOT / "benchmarks" / "panel3_cross_power_calibration_raw.json"
)
CALIBRATION_FREEZE = (
    ROOT
    / "benchmarks"
    / "panel3_cross_power_calibration_source_freeze.json"
)
DEFAULT_OUTPUT = ROOT / "benchmarks" / "panel3_power_design_decision.json"
SOURCE_PATHS = tuple(
    ROOT / relative
    for relative in calibration.freeze.source_paths()
)
CANDIDATES = (
    "t5_composite_policy",
    "guarded_cross_features_policy",
)
STRATA = ("smooth_numeric", "mixed_categorical", "applied_noisy")
PANEL3_V1_RUNTIME_PACKAGE_NAMES = (
    "catboost",
    "darkofit",
    "numba",
    "numpy",
    "openml",
    "pandas",
    "scikit-learn",
    "scipy",
)
PANEL3_V1_RUNTIME_CONTRACT = {
    "contract_kind": "exact_active_environment_versions_v1",
    "python_implementation": "cpython",
    "python_version": "3.12.13",
    "packages": {
        "catboost": "1.2.10",
        "darkofit": "0.10.0",
        "numba": "0.66.0",
        "numpy": "2.4.6",
        "openml": "0.15.1",
        "pandas": "2.3.3",
        "scikit-learn": "1.7.2",
        "scipy": "1.16.3",
    },
}
PANEL3_V1_CALIBRATION_TASKS = [
    {
        "dataset_name": "airfoil_self_noise",
        "task_id": 363612,
        "stratum": "smooth_numeric",
        "expected_t5_size_gate_applicable_coordinates": 0,
    },
    {
        "dataset_name": "Another-Dataset-on-used-Fiat-500",
        "task_id": 363615,
        "stratum": "mixed_categorical",
        "expected_t5_size_gate_applicable_coordinates": 0,
    },
    {
        "dataset_name": "concrete_compressive_strength",
        "task_id": 363625,
        "stratum": "smooth_numeric",
        "expected_t5_size_gate_applicable_coordinates": 0,
    },
    {
        "dataset_name": "diamonds",
        "task_id": 363631,
        "stratum": "mixed_categorical",
        "expected_t5_size_gate_applicable_coordinates": 3,
    },
    {
        "dataset_name": "Food_Delivery_Time",
        "task_id": 363672,
        "stratum": "mixed_categorical",
        "expected_t5_size_gate_applicable_coordinates": 3,
    },
    {
        "dataset_name": "healthcare_insurance_expenses",
        "task_id": 363675,
        "stratum": "mixed_categorical",
        "expected_t5_size_gate_applicable_coordinates": 0,
    },
    {
        "dataset_name": "houses",
        "task_id": 363678,
        "stratum": "applied_noisy",
        "expected_t5_size_gate_applicable_coordinates": 3,
    },
    {
        "dataset_name": "miami_housing",
        "task_id": 363686,
        "stratum": "applied_noisy",
        "expected_t5_size_gate_applicable_coordinates": 3,
    },
    {
        "dataset_name": "physiochemical_protein",
        "task_id": 363693,
        "stratum": "smooth_numeric",
        "expected_t5_size_gate_applicable_coordinates": 3,
    },
    {
        "dataset_name": "QSAR-TID-11",
        "task_id": 363697,
        "stratum": "applied_noisy",
        "expected_t5_size_gate_applicable_coordinates": 3,
    },
    {
        "dataset_name": "QSAR_fish_toxicity",
        "task_id": 363698,
        "stratum": "applied_noisy",
        "expected_t5_size_gate_applicable_coordinates": 0,
    },
    {
        "dataset_name": "superconductivity",
        "task_id": 363705,
        "stratum": "smooth_numeric",
        "expected_t5_size_gate_applicable_coordinates": 3,
    },
    {
        "dataset_name": "wine_quality",
        "task_id": 363708,
        "stratum": "applied_noisy",
        "expected_t5_size_gate_applicable_coordinates": 3,
    },
]
PANEL3_V1_EXCHANGEABILITY = {
    "sampling_unit": "complete_three_coordinate_dataset_triplet",
    "within_stratum_sampling": "independent_with_replacement",
    "coordinate_values_resampled_or_recombined_across_source_tasks": False,
    "t5_applicable_slots_sample_only_fully_applicable_source_triplets": True,
    "t5_known_nonapplicable_slots_are_exact_one_one_one_triplets": True,
    "guarded_cross_slots_sample_all_source_triplets_in_the_matching_stratum": (
        True
    ),
    "assumption": (
        "Within each frozen semantic stratum and matching T5 applicability "
        "state, the 13-task spent panel is an exchangeable empirical proxy "
        "for the prospective panel. This is a transport assumption, not "
        "confirmation evidence."
    ),
}
PANEL3_V1_PROSPECTIVE_PANEL = {
    "required_strata": [
        "smooth_numeric",
        "mixed_categorical",
        "applied_noisy",
    ],
    "required_tasks_per_stratum": 4,
    "required_task_count": 12,
    "coordinates_per_task": 3,
    "authorized_only_for_exact_primary_selection": True,
    "slots": [
        {
            "lineage_cluster": "asa_usnews_colleges_1993_94",
            "task_id": 5166,
            "stratum": "applied_noisy",
            "t5_size_gate_applicability": [False, False, False],
        },
        {
            "lineage_cluster": "brien_payne_wine_sensory_experiment",
            "task_id": 359931,
            "stratum": "mixed_categorical",
            "t5_size_gate_applicability": [False, False, False],
        },
        {
            "lineage_cluster": "cps_1988_wages",
            "task_id": 361261,
            "stratum": "mixed_categorical",
            "t5_size_gate_applicability": [True, True, True],
        },
        {
            "lineage_cluster": "kaggle_imperial_loan_default_loss",
            "task_id": 361290,
            "stratum": "applied_noisy",
            "t5_size_gate_applicability": [True, True, True],
        },
        {
            "lineage_cluster": "sarcos_robot_arm",
            "task_id": 361254,
            "stratum": "smooth_numeric",
            "t5_size_gate_applicability": [True, True, True],
        },
        {
            "lineage_cluster": "statlib_1987_mlb_hitter_salary",
            "task_id": 4851,
            "stratum": "applied_noisy",
            "t5_size_gate_applicability": [False, False, False],
        },
        {
            "lineage_cluster": "uci_energy_efficiency",
            "task_id": 361617,
            "stratum": "smooth_numeric",
            "t5_size_gate_applicability": [False, False, False],
        },
        {
            "lineage_cluster": "uci_forest_fires",
            "task_id": 361618,
            "stratum": "mixed_categorical",
            "t5_size_gate_applicability": [False, False, False],
        },
        {
            "lineage_cluster": "uci_garment_employee_productivity",
            "task_id": 360993,
            "stratum": "mixed_categorical",
            "t5_size_gate_applicability": [False, False, False],
        },
        {
            "lineage_cluster": "uci_naval_propulsion",
            "task_id": 361247,
            "stratum": "smooth_numeric",
            "t5_size_gate_applicability": [True, True, True],
        },
        {
            "lineage_cluster": "uk_wheat_milling_baking_quality",
            "task_id": 363396,
            "stratum": "applied_noisy",
            "t5_size_gate_applicability": [False, False, False],
        },
        {
            "lineage_cluster": "wave_energy_converters",
            "task_id": 361253,
            "stratum": "smooth_numeric",
            "t5_size_gate_applicability": [True, True, True],
        },
    ],
}
PANEL3_V1_SOURCE_RELATIVE_PATHS = (
    "pyproject.toml",
    "benchmarks/panel3_candidate_contract.json",
    "benchmarks/panel3_environment_contract.json",
    "benchmarks/panel3_cross_power_calibration_protocol.md",
    "benchmarks/panel3_power_design_contract.json",
    "benchmarks/panel3_power_design_protocol.md",
    "benchmarks/build_panel3_power_design.py",
    "benchmarks/panel3_registry_declarations.json",
    "benchmarks/freeze_panel3_cross_power_calibration.py",
    "benchmarks/run_panel3_cross_power_calibration.py",
    "benchmarks/analyze_panel3_cross_power_calibration.py",
    "benchmarks/run_panel3_confirmation.py",
    "benchmarks/analyze_panel3_confirmation.py",
    "benchmarks/run_t5_composite_confirmation.py",
    "benchmarks/run_smooth_cross_features.py",
    "benchmarks/basketball_harness.py",
    "benchmarks/basketball_guardrails.py",
    "benchmarks/run_basketball_creator_benchmark.py",
    "benchmarks/build_ctr23_contamination_registry.py",
    "benchmarks/panel3_data_contract.py",
    "benchmarks/panel3_registry_common.py",
    "benchmarks/run_tabarena_regression_followon_screen.py",
    "benchmarks/run_tabarena_regression_cap_horizon.py",
    "benchmarks/tabarena_followon_warmup.py",
    "benchmarks/tabarena_warmup.py",
    "tests/conftest.py",
    "tests/test_campaign_partition.py",
    "tests/test_panel3_cross_power_calibration.py",
    "tests/test_panel3_execution.py",
    "tests/test_panel3_power_design.py",
    "tests/test_panel3_registry.py",
    "benchmarks/panel3_registry_protocol.md",
    "benchmarks/campaign_lib/__init__.py",
    "benchmarks/campaign_lib/provenance.py",
    "benchmarks/preflight_panel3_registry.py",
    "benchmarks/confirmation_target_preflight.py",
    "benchmarks/build_fresh_confirmation_registry.py",
    "benchmarks/build_panel3_registry.py",
    "benchmarks/t5_composite_registry_protocol.md",
    "benchmarks/smooth_cross_features_protocol.md",
    "benchmarks/smooth_cross_margin_analysis.json",
    "benchmarks/panel3_registry_declarations_invalid_draft.md",
    "darkofit/__init__.py",
    "darkofit/_numba_runtime.py",
    "darkofit/_validation.py",
    "darkofit/auto_params.py",
    "darkofit/binning.py",
    "darkofit/booster.py",
    "darkofit/callbacks.py",
    "darkofit/flat_model.py",
    "darkofit/linear_residual.py",
    "darkofit/losses.py",
    "darkofit/preprocessing.py",
    "darkofit/serialization.py",
    "darkofit/shap.py",
    "darkofit/sklearn_api.py",
    "darkofit/target_encoding.py",
    "darkofit/tree.py",
    "darkofit/tuning/__init__.py",
    "darkofit/tuning/optuna_backend.py",
    "darkofit/tuning/results.py",
    "darkofit/tuning/scoring.py",
    "darkofit/tuning/search.py",
    "darkofit/tuning/spaces.py",
    "darkofit/tuning/validation.py",
    "darkofit/warmup.py",
)
PANEL3_V1_CANDIDATE_CONTRACT_SHA256 = (
    "873f5ca2ea284ba8b178e03882a364e43bc11f5ff5050a337c2f7b9c2af41e48"
)
PANEL3_V1_ENVIRONMENT_CONTRACT_SHA256 = (
    "9755529964eda51024a6a02b9bad46b4099b13d38621067e4ade1cc4232046a0"
)
PANEL3_V1_POWER_CONTRACT_SHA256 = (
    "cd010883698aa294205c0c5cf25ac08a47edcbee2ebc62b45d39b1b68a44a925"
)
PANEL3_V1_QUALITY_GATES = {
    "equal_dataset_geomean_ratio_at_most": 0.995,
    "bootstrap_upper_ratio_at_most": 1.002,
    "leave_one_favorable_dataset_out_ratio_at_most": 0.998,
    "worst_dataset_ratio_at_most": 1.005,
}
PANEL3_V1_SIMULATION = {
    "outer_panel_simulations": 5_000,
    "outer_seed": 20_260_718,
    "hierarchical_bootstrap_seed": 20_260_717,
    "hierarchical_bootstrap_replicates": 100_000,
    "hierarchical_bootstrap_batch": 10_000,
    "outer_matrix_batch": 50,
    "numpy_percentile_method": "linear",
    "power_floor": 0.8,
    "power_estimate_rule": (
        "both_point_estimate_and_one_sided_wilson_lower_bound_must_meet_floor"
    ),
    "familywise_one_sided_alpha": 0.05,
    "initial_candidate_count": 2,
    "initial_per_candidate_one_sided_alpha": 0.025,
    "initial_bootstrap_percentile": 97.5,
    "initial_power_wilson_confidence": 0.975,
    "singleton_fallback_allowed": True,
    "singleton_per_candidate_one_sided_alpha": 0.05,
    "singleton_bootstrap_percentile": 95.0,
    "singleton_power_wilson_confidence": 0.95,
    "quality_gates": PANEL3_V1_QUALITY_GATES,
}
SUMMARY_FIELDS = {
    "schema_version",
    "name",
    "created_at",
    "raw_path",
    "raw_file_sha256",
    "raw_artifact_sha256",
    "source_freeze_sha256",
    "estimand",
    "candidate_results",
    "fixed_panel_power_inputs",
    "complete_unfiltered_coordinate_census",
    "ties_and_losses_preserved",
    "development_only",
    "may_inform_separately_frozen_power_design",
    "independent_confirmation",
    "panel3_authorized",
    "default_promotion_authorized",
    "product_claim_authorized",
    "summary_sha256",
}
DECISION_FIELDS = {
    "schema_version",
    "name",
    "created_at",
    "source_head",
    "decision_execution_head",
    "source_sha256",
    "contract",
    "calibration",
    "runtime",
    "mapping",
    "pre_h1_target_statistic_exclusions",
    "prospective_panel",
    "simulation",
    "initial_bonferroni_screen",
    "singleton_fallback",
    "retained_candidates",
    "candidate_count",
    "familywise_one_sided_alpha",
    "per_candidate_one_sided_alpha",
    "bootstrap_percentile",
    "power_floor",
    "checks",
    "target_preflight_authorized",
    "registry_build_authorized",
    "confirmation_run_authorized",
    "default_promotion_authorized",
    "product_claim_authorized",
    "decision_sha256",
}


def _git(*arguments: str) -> str:
    return provenance.git_output(ROOT, *arguments)


def _is_sha256(value: Any) -> bool:
    return provenance.is_sha256(value)


def _validate_decision_runtime(
    runtime: Any,
    *,
    require_current_sources: bool,
    candidate_contract: dict[str, Any] | None = None,
) -> None:
    package_names = PANEL3_V1_RUNTIME_PACKAGE_NAMES
    if (
        not isinstance(runtime, dict)
        or set(runtime)
        != {
            "contract_kind",
            "python_implementation",
            "python_version",
            "packages",
        }
        or runtime.get("contract_kind")
        != "exact_active_environment_versions_v1"
        or not isinstance(runtime.get("python_implementation"), str)
        or not runtime["python_implementation"]
        or not isinstance(runtime.get("python_version"), str)
        or not runtime["python_version"]
        or not isinstance(runtime.get("packages"), dict)
        or tuple(sorted(runtime["packages"])) != tuple(sorted(package_names))
        or any(
            not isinstance(runtime["packages"][name], str)
            or not runtime["packages"][name]
            for name in package_names
        )
    ):
        raise RuntimeError("Panel 3 power-design runtime binding changed")
    if (
        not require_current_sources
        and runtime != PANEL3_V1_RUNTIME_CONTRACT
    ):
        raise RuntimeError(
            "Panel 3 historical runtime contract changed"
        )
    if require_current_sources:
        if tuple(confirmation.RUNTIME_PACKAGE_NAMES) != package_names:
            raise RuntimeError(
                "Panel 3 runner runtime package schema changed"
            )
        if not isinstance(candidate_contract, dict):
            raise RuntimeError(
                "Panel 3 candidate runtime contract is unavailable"
            )
        observed = confirmation._validate_runtime_contract(
            candidate_contract
        )
        if runtime != observed:
            raise RuntimeError(
                "Panel 3 power-design runtime environment changed"
            )


def _positive_triplet(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise RuntimeError(f"{label} must preserve exactly three coordinates")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise RuntimeError(f"{label} contains an invalid ratio")
    return result


def _validate_contract(contract: Any) -> dict[str, Any]:
    if (
        not isinstance(contract, dict)
        or contract.get("schema_version") != 1
        or contract.get("contract_name")
        != "darkofit_panel3_authorization_power_design_v1"
        or set(contract)
        != {
            "schema_version",
            "contract_name",
            "calibration",
            "prospective_panel",
            "simulation",
            "candidate_retention",
            "exchangeability",
        }
    ):
        raise RuntimeError("Panel 3 power-design contract changed")
    calibration_contract = contract["calibration"]
    tasks = calibration_contract.get("tasks")
    if (
        calibration_contract.get("summary_name")
        != "darkofit_panel3_cross_power_calibration_summary_v1"
        or calibration_contract.get("summary_path")
        != str(DEFAULT_SUMMARY.relative_to(ROOT))
        or calibration_contract.get("raw_path")
        != str(DEFAULT_RAW.relative_to(ROOT))
        or calibration_contract.get("required_coordinate_count") != 39
        or calibration_contract.get("coordinates_per_task") != 3
        or calibration_contract.get(
            "minimum_distinct_source_tasks_per_stratum"
        )
        != 2
        or calibration_contract.get(
            "minimum_distinct_applicable_t5_source_tasks_per_stratum"
        )
        != 2
        or not isinstance(tasks, list)
        or len(tasks) != 13
    ):
        raise RuntimeError("Panel 3 calibration mapping changed")
    task_keys = {
        "dataset_name",
        "task_id",
        "stratum",
        "expected_t5_size_gate_applicable_coordinates",
    }
    if any(
        not isinstance(row, dict)
        or set(row) != task_keys
        or not isinstance(row["dataset_name"], str)
        or not row["dataset_name"]
        or type(row["task_id"]) is not int
        or row["task_id"] <= 0
        or row["stratum"] not in STRATA
        or row["expected_t5_size_gate_applicable_coordinates"] not in (0, 3)
        for row in tasks
    ):
        raise RuntimeError("Panel 3 calibration-task mapping is invalid")
    if (
        len({row["dataset_name"] for row in tasks}) != 13
        or len({row["task_id"] for row in tasks}) != 13
        or Counter(row["stratum"] for row in tasks)
        != Counter(
            {
                "smooth_numeric": 4,
                "mixed_categorical": 4,
                "applied_noisy": 5,
            }
        )
    ):
        raise RuntimeError("Panel 3 calibration-task census changed")

    panel = contract["prospective_panel"]
    slots = panel.get("slots")
    if (
        panel.get("required_strata") != list(STRATA)
        or panel.get("required_tasks_per_stratum") != 4
        or panel.get("required_task_count") != 12
        or panel.get("coordinates_per_task") != 3
        or panel.get("authorized_only_for_exact_primary_selection") is not True
        or not isinstance(slots, list)
        or len(slots) != 12
    ):
        raise RuntimeError("Panel 3 prospective composition changed")
    slot_keys = {
        "lineage_cluster",
        "task_id",
        "stratum",
        "t5_size_gate_applicability",
    }
    if any(
        not isinstance(row, dict)
        or set(row) != slot_keys
        or not isinstance(row["lineage_cluster"], str)
        or not row["lineage_cluster"]
        or type(row["task_id"]) is not int
        or row["task_id"] <= 0
        or row["stratum"] not in STRATA
        or not isinstance(row["t5_size_gate_applicability"], list)
        or len(row["t5_size_gate_applicability"]) != 3
        or any(
            type(value) is not bool
            for value in row["t5_size_gate_applicability"]
        )
        for row in slots
    ):
        raise RuntimeError("Panel 3 prospective slots are invalid")
    if (
        slots
        != sorted(slots, key=lambda row: row["lineage_cluster"])
        or len({row["lineage_cluster"] for row in slots}) != 12
        or len({row["task_id"] for row in slots}) != 12
        or Counter(row["stratum"] for row in slots)
        != Counter({stratum: 4 for stratum in STRATA})
        or {
            row["lineage_cluster"]
            for row in slots
            if not any(row["t5_size_gate_applicability"])
        }
        != {
            "asa_usnews_colleges_1993_94",
            "brien_payne_wine_sensory_experiment",
            "statlib_1987_mlb_hitter_salary",
            "uci_energy_efficiency",
            "uci_forest_fires",
            "uci_garment_employee_productivity",
            "uk_wheat_milling_baking_quality",
        }
    ):
        raise RuntimeError(
            "Panel 3 prospective ordering, strata, or T5 no-op slots changed"
        )

    simulation = contract["simulation"]
    if simulation != PANEL3_V1_SIMULATION:
        raise RuntimeError("Panel 3 power simulation contract changed")
    retention = contract["candidate_retention"]
    if (
        retention.get("candidate_order") != list(CANDIDATES)
        or retention.get("post_calibration_discretion_allowed") is not False
        or not all(
            isinstance(retention.get(field), str) and retention[field]
            for field in (
                "initial_rule",
                "zero_initial_survivors",
                "one_initial_survivor",
                "two_initial_survivors",
            )
        )
    ):
        raise RuntimeError("Panel 3 candidate-retention rule changed")
    exchangeability = contract["exchangeability"]
    if (
        exchangeability.get("sampling_unit")
        != "complete_three_coordinate_dataset_triplet"
        or exchangeability.get("within_stratum_sampling")
        != "independent_with_replacement"
        or exchangeability.get(
            "coordinate_values_resampled_or_recombined_across_source_tasks"
        )
        is not False
        or exchangeability.get(
            "t5_applicable_slots_sample_only_fully_applicable_source_triplets"
        )
        is not True
        or exchangeability.get(
            "t5_known_nonapplicable_slots_are_exact_one_one_one_triplets"
        )
        is not True
        or exchangeability.get(
            "guarded_cross_slots_sample_all_source_triplets_in_the_matching_stratum"
        )
        is not True
    ):
        raise RuntimeError("Panel 3 exchangeability rule changed")
    return contract


def load_contract_snapshot() -> tuple[dict[str, Any], str]:
    contract, file_sha256 = common.secure_load_json(CONTRACT)
    return _validate_contract(contract), file_sha256


def load_contract() -> dict[str, Any]:
    return load_contract_snapshot()[0]


def _validate_pre_h1_target_exclusion_slots(
    panel: Any,
    exclusions: Any,
    *,
    require_current_sources: bool,
) -> None:
    """Reject permanently exposed target-statistic lineages, including archival."""
    slots = panel.get("slots") if isinstance(panel, dict) else None
    if not isinstance(slots, list):
        raise RuntimeError(
            "Panel 3 pre-H1 target-statistic slot boundary is incomplete"
        )
    expected_keys = {
        "task_id",
        "dataset_id",
        "lineage_cluster",
        "stratum",
        "exposure_kind",
        "reason",
        "replacement_task_id",
    }
    if not isinstance(exclusions, list) or len(exclusions) != 3:
        raise RuntimeError(
            "Panel 3 pre-H1 target-statistic exclusion ledger is invalid"
        )
    for exclusion in exclusions:
        if (
            not isinstance(exclusion, dict)
            or set(exclusion) != expected_keys
            or type(exclusion["task_id"]) is not int
            or exclusion["task_id"] <= 0
            or type(exclusion["dataset_id"]) is not int
            or exclusion["dataset_id"] <= 0
            or type(exclusion["replacement_task_id"]) is not int
            or exclusion["replacement_task_id"] <= 0
            or exclusion["replacement_task_id"] == exclusion["task_id"]
            or not isinstance(exclusion["lineage_cluster"], str)
            or not exclusion["lineage_cluster"]
            or exclusion["stratum"] not in STRATA
            or exclusion["exposure_kind"]
            != "parquet_footer_target_min_max_statistics"
            or exclusion["reason"]
            != "target_parquet_footer_min_max_observed_before_h1"
        ):
            raise RuntimeError(
                "Panel 3 pre-H1 target-statistic exclusion ledger "
                "is invalid"
            )
    for key in (
        "task_id",
        "dataset_id",
        "lineage_cluster",
        "replacement_task_id",
    ):
        values = [row[key] for row in exclusions]
        if len(values) != len(set(values)):
            raise RuntimeError(
                "Panel 3 pre-H1 target-statistic exclusion ledger "
                "contains duplicates"
            )
    if (
        require_current_sources
        and exclusions != common.PRE_H1_TARGET_STATISTIC_EXCLUSIONS
    ):
        raise RuntimeError(
            "Panel 3 pre-H1 target-statistic exclusion ledger changed"
        )
    excluded_task_ids = {row["task_id"] for row in exclusions}
    excluded_lineages = {row["lineage_cluster"] for row in exclusions}
    if any(
        not isinstance(slot, dict)
        or slot.get("task_id") in excluded_task_ids
        or slot.get("lineage_cluster") in excluded_lineages
        for slot in slots
    ):
        raise RuntimeError(
            "Panel 3 pre-H1 exposed target-statistic lineage re-entered "
            "the power design"
        )
    by_task = {
        slot.get("task_id"): slot
        for slot in slots
        if isinstance(slot, dict)
    }
    if any(
        not isinstance(by_task.get(row["replacement_task_id"]), dict)
        or by_task[row["replacement_task_id"]].get("stratum")
        != row["stratum"]
        for row in exclusions
    ):
        raise RuntimeError(
            "Panel 3 pre-H1 replacement is absent from its frozen stratum"
        )


def _validate_summary_shape(
    summary: Any,
    contract: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    if (
        not isinstance(summary, dict)
        or set(summary) != SUMMARY_FIELDS
        or summary.get("schema_version") != 1
        or summary.get("name") != contract["calibration"]["summary_name"]
        or summary.get("estimand") != "exact_candidate/current_default"
        or summary.get("complete_unfiltered_coordinate_census") is not True
        or summary.get("ties_and_losses_preserved") is not True
        or summary.get("development_only") is not True
        or summary.get("may_inform_separately_frozen_power_design") is not True
        or summary.get("independent_confirmation") is not False
        or summary.get("panel3_authorized") is not False
        or summary.get("default_promotion_authorized") is not False
        or summary.get("product_claim_authorized") is not False
        or not _is_sha256(summary.get("raw_file_sha256"))
        or not _is_sha256(summary.get("raw_artifact_sha256"))
        or not _is_sha256(summary.get("source_freeze_sha256"))
        or not _is_sha256(summary.get("summary_sha256"))
    ):
        raise RuntimeError("Panel 3 calibration summary contract changed")
    common.verify_artifact_sha256(summary, "summary_sha256")
    if (
        summary["raw_path"] != str(DEFAULT_RAW.relative_to(ROOT))
        or set(summary["candidate_results"]) != set(CANDIDATES)
        or set(summary["fixed_panel_power_inputs"]) != set(CANDIDATES)
    ):
        raise RuntimeError("Panel 3 calibration summary boundary changed")

    expected_tasks = contract["calibration"]["tasks"]
    expected_names = [row["dataset_name"] for row in expected_tasks]
    profiles: dict[str, list[dict[str, Any]]] = {}
    for candidate in CANDIDATES:
        candidate_result = summary["candidate_results"][candidate]
        rows = summary["fixed_panel_power_inputs"][candidate]
        if (
            not isinstance(candidate_result, dict)
            or candidate_result.get("coordinate_count") != 39
            or candidate_result.get("dataset_count") != 13
            or not isinstance(rows, list)
            or len(rows) != 13
            or [row.get("dataset_name") for row in rows] != expected_names
        ):
            raise RuntimeError(
                f"Panel 3 {candidate} calibration census changed"
            )
        validated = []
        for row, expected in zip(rows, expected_tasks, strict=True):
            if (
                not isinstance(row, dict)
                or set(row)
                != {
                    "source",
                    "dataset_name",
                    "task_id",
                    "ratio",
                    "coordinate_ratios",
                    "t5_size_gate_applicable_coordinates",
                    "engaged_coordinates",
                }
                or row["source"]
                != "spent_tabarena_13x3_exact_policy_complete_census"
                or row["dataset_name"] != expected["dataset_name"]
                or row["task_id"] != expected["task_id"]
                or row["t5_size_gate_applicable_coordinates"]
                != expected[
                    "expected_t5_size_gate_applicable_coordinates"
                ]
                or type(row["engaged_coordinates"]) is not int
                or not 0 <= row["engaged_coordinates"] <= 3
            ):
                raise RuntimeError(
                    f"Panel 3 {candidate} calibration mapping changed"
                )
            triplet = _positive_triplet(
                row["coordinate_ratios"],
                f"{candidate}/{row['dataset_name']}",
            )
            ratio = float(row["ratio"])
            triplet_geomean = math.exp(
                sum(math.log(value) for value in triplet) / 3.0
            )
            if (
                not math.isfinite(ratio)
                or ratio <= 0.0
                or not math.isclose(
                    ratio,
                    triplet_geomean,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                or (
                    candidate == "t5_composite_policy"
                    and not expected[
                        "expected_t5_size_gate_applicable_coordinates"
                    ]
                    and triplet != (1.0, 1.0, 1.0)
                )
            ):
                raise RuntimeError(
                    f"Panel 3 {candidate} triplet aggregation changed"
                )
            validated.append(
                {
                    "dataset_name": row["dataset_name"],
                    "task_id": row["task_id"],
                    "stratum": expected["stratum"],
                    "t5_size_gate_applicable_coordinates": row[
                        "t5_size_gate_applicable_coordinates"
                    ],
                    "coordinate_ratios": list(triplet),
                }
            )
        profiles[candidate] = validated
    return profiles


def validate_calibration(
    summary: dict[str, Any],
    *,
    summary_path: Path = DEFAULT_SUMMARY,
    raw_path: Path = DEFAULT_RAW,
    verify_raw: bool = True,
    contract: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    contract = load_contract() if contract is None else contract
    profiles = _validate_summary_shape(summary, contract)
    if verify_raw:
        if (
            summary_path.expanduser().absolute() != DEFAULT_SUMMARY
            or raw_path.expanduser().absolute() != DEFAULT_RAW
        ):
            raise RuntimeError("Panel 3 calibration files changed")
        stored_summary, _summary_file_sha256 = common.secure_load_json(
            summary_path
        )
        raw, raw_file_sha256 = common.secure_load_json(raw_path)
        if (
            stored_summary != summary
            or raw_file_sha256 != summary["raw_file_sha256"]
            or not isinstance(raw, dict)
            or raw.get("raw_artifact_sha256")
            != summary["raw_artifact_sha256"]
            or raw.get("source_freeze_sha256")
            != summary["source_freeze_sha256"]
        ):
            raise RuntimeError("Panel 3 calibration raw binding changed")
        recomputed = calibration.analyze(
            raw,
            raw_path=raw_path,
            raw_file_sha256=raw_file_sha256,
            verify_source=True,
            verify_spool=True,
        )
        for field in (
            "raw_artifact_sha256",
            "source_freeze_sha256",
            "estimand",
            "candidate_results",
            "fixed_panel_power_inputs",
            "complete_unfiltered_coordinate_census",
            "ties_and_losses_preserved",
            "development_only",
            "may_inform_separately_frozen_power_design",
            "independent_confirmation",
            "panel3_authorized",
            "default_promotion_authorized",
            "product_claim_authorized",
        ):
            if recomputed[field] != summary[field]:
                raise RuntimeError(
                    f"Panel 3 calibration summary differs from raw: {field}"
                )
    return profiles


def validate_h1_design_sources(
    summary: dict[str, Any],
) -> tuple[str, dict[str, str]]:
    """Require every design byte to have existed at outcome-blind H1."""
    try:
        source_freeze, _freeze_file_sha256 = common.secure_load_json(
            CALIBRATION_FREEZE
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Panel 3 calibration H1 freeze is unavailable"
        ) from exc
    if not isinstance(source_freeze, dict):
        raise RuntimeError("Panel 3 calibration H1 freeze is unavailable")
    common.verify_artifact_sha256(source_freeze, "source_freeze_sha256")
    if (
        source_freeze.get("name")
        != "darkofit_panel3_cross_power_calibration_source_freeze_v1"
        or source_freeze.get("outcome_blind_source_freeze") is not True
        or source_freeze.get("source_head_clean") is not True
        or source_freeze.get("source_freeze_sha256")
        != summary["source_freeze_sha256"]
        or not isinstance(source_freeze.get("source_head"), str)
        or len(source_freeze["source_head"]) != 40
        or not isinstance(source_freeze.get("source_file_sha256"), dict)
    ):
        raise RuntimeError("Panel 3 calibration H1 freeze binding changed")
    h1_hashes = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        frozen = source_freeze["source_file_sha256"].get(relative)
        if (
            not _is_sha256(frozen)
            or common.sha256_file(path) != frozen
        ):
            raise RuntimeError(
                "Panel 3 power-design source was not frozen unchanged at "
                f"calibration H1: {relative}"
            )
        h1_hashes[relative] = frozen
    return source_freeze["source_head"], h1_hashes


def _profile_pools(
    profiles: list[dict[str, Any]],
    candidate: str,
    contract: dict[str, Any],
) -> dict[str, list[np.ndarray]]:
    pools: dict[str, list[np.ndarray]] = defaultdict(list)
    applicable: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in profiles:
        triplet = np.asarray(row["coordinate_ratios"], dtype=np.float64)
        pools[row["stratum"]].append(triplet)
        if row["t5_size_gate_applicable_coordinates"] == 3:
            applicable[row["stratum"]].append(triplet)
    minimum = contract["calibration"][
        "minimum_distinct_source_tasks_per_stratum"
    ]
    minimum_applicable = contract["calibration"][
        "minimum_distinct_applicable_t5_source_tasks_per_stratum"
    ]
    if any(len(pools[stratum]) < minimum for stratum in STRATA):
        raise RuntimeError("Panel 3 stratum has inadequate calibration support")
    if candidate == "t5_composite_policy" and any(
        len(applicable[stratum]) < minimum_applicable
            for stratum in STRATA
            if any(
                slot["stratum"] == stratum
                and any(slot["t5_size_gate_applicability"])
                for slot in contract["prospective_panel"]["slots"]
            )
    ):
        raise RuntimeError(
            "Panel 3 T5-applicable stratum has inadequate calibration support"
        )
    return applicable if candidate == "t5_composite_policy" else pools


def _bootstrap_draws(
    *,
    seed: int,
    replicates: int,
    batch: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    result = []
    for start in range(0, replicates, batch):
        count = min(batch, replicates - start)
        result.append(
            (
                rng.integers(0, 12, size=(count, 12)),
                rng.integers(0, 3, size=(count, 12, 3)),
            )
        )
    return result


def _bootstrap_weight_matrix(
    draws: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Losslessly count the 36 task-coordinate selections per replicate."""
    rows = []
    for task_draws, fold_draws in draws:
        if (
            task_draws.ndim != 2
            or task_draws.shape[1] != 12
            or fold_draws.shape != (*task_draws.shape, 3)
        ):
            raise ValueError("invalid hierarchical-bootstrap draws")
        flat = task_draws[..., None] * 3 + fold_draws
        weights = np.zeros((task_draws.shape[0], 36), dtype=np.uint8)
        row_index = np.broadcast_to(
            np.arange(task_draws.shape[0], dtype=np.int64)[:, None, None],
            flat.shape,
        )
        np.add.at(weights, (row_index.ravel(), flat.ravel()), 1)
        if not np.all(weights.sum(axis=1) == 36):
            raise RuntimeError("hierarchical-bootstrap weights changed")
        rows.append(weights)
    return np.asarray(np.concatenate(rows, axis=0), dtype=np.float64)


def _hierarchical_upper(
    logs: np.ndarray,
    draws: list[tuple[np.ndarray, np.ndarray]],
    *,
    percentile: float,
) -> float:
    statistics = np.concatenate(
        [
            logs[task_draws[..., None], fold_draws].mean(axis=(1, 2))
            for task_draws, fold_draws in draws
        ]
    )
    return float(
        np.exp(
            np.percentile(
                statistics,
                percentile,
                method="linear",
            )
        )
    )


def _hierarchical_uppers(
    panel_logs: np.ndarray,
    weights: np.ndarray,
    *,
    percentile: float,
    panel_batch: int,
) -> np.ndarray:
    """Evaluate the frozen, mathematically equivalent bootstrap via BLAS."""
    if (
        panel_logs.ndim != 3
        or panel_logs.shape[1:] != (12, 3)
        or weights.ndim != 2
        or weights.shape[1] != 36
        or type(panel_batch) is not int
        or panel_batch <= 0
    ):
        raise ValueError("invalid batched hierarchical-bootstrap inputs")
    result = np.empty(panel_logs.shape[0], dtype=np.float64)
    flattened = panel_logs.reshape(panel_logs.shape[0], 36)
    for start in range(0, panel_logs.shape[0], panel_batch):
        stop = min(start + panel_batch, panel_logs.shape[0])
        statistics = weights @ flattened[start:stop].T / 36.0
        result[start:stop] = np.exp(
            np.percentile(
                statistics,
                percentile,
                axis=0,
                method="linear",
            )
        )
    return result


def _wilson_lower(successes: int, total: int, confidence: float) -> float:
    if (
        type(successes) is not int
        or type(total) is not int
        or not 0 <= successes <= total
        or total <= 0
        or not 0.5 < confidence < 1.0
    ):
        raise ValueError("invalid Wilson interval inputs")
    probability = successes / total
    z = NormalDist().inv_cdf(confidence)
    denominator = 1.0 + z * z / total
    center = probability + z * z / (2.0 * total)
    radius = z * math.sqrt(
        probability * (1.0 - probability) / total
        + z * z / (4.0 * total * total)
    )
    return (center - radius) / denominator


def simulate_candidate_power(
    profiles: list[dict[str, Any]],
    *,
    candidate: str,
    percentile: float,
    wilson_confidence: float,
    contract: dict[str, Any] | None = None,
    outer_simulations: int | None = None,
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    """Simulate the frozen four-gate prospective statistical decision."""
    contract = load_contract() if contract is None else _validate_contract(contract)
    if candidate not in CANDIDATES:
        raise ValueError("unknown Panel 3 candidate")
    simulation = contract["simulation"]
    outer = (
        simulation["outer_panel_simulations"]
        if outer_simulations is None
        else outer_simulations
    )
    replicates = (
        simulation["hierarchical_bootstrap_replicates"]
        if bootstrap_replicates is None
        else bootstrap_replicates
    )
    if type(outer) is not int or outer <= 0:
        raise ValueError("outer simulation count must be positive")
    if type(replicates) is not int or replicates <= 0:
        raise ValueError("bootstrap replicate count must be positive")
    pools = _profile_pools(profiles, candidate, contract)
    slots = contract["prospective_panel"]["slots"]
    rng = np.random.default_rng(simulation["outer_seed"])
    draws = _bootstrap_draws(
        seed=simulation["hierarchical_bootstrap_seed"],
        replicates=replicates,
        batch=min(simulation["hierarchical_bootstrap_batch"], replicates),
    )
    weights = _bootstrap_weight_matrix(draws)
    panels = np.empty((outer, 12, 3), dtype=np.float64)
    for simulation_index in range(outer):
        panel = []
        for slot in slots:
            pool = pools[slot["stratum"]]
            index = int(rng.integers(0, len(pool)))
            triplet = np.asarray(pool[index], dtype=np.float64).copy()
            if candidate == "t5_composite_policy":
                applicability = np.asarray(
                    slot["t5_size_gate_applicability"],
                    dtype=np.bool_,
                )
                triplet[~applicability] = 1.0
            panel.append(triplet)
        panels[simulation_index] = np.asarray(panel, dtype=np.float64)
    logs = np.log(panels)
    dataset_logs = logs.mean(axis=2)
    point = np.exp(dataset_logs.mean(axis=1))
    leave_one = np.exp(
        (dataset_logs.sum(axis=1) - dataset_logs.min(axis=1)) / 11.0
    )
    worst = np.exp(dataset_logs.max(axis=1))
    upper = _hierarchical_uppers(
        logs,
        weights,
        percentile=percentile,
        panel_batch=min(simulation["outer_matrix_batch"], outer),
    )
    gates = simulation["quality_gates"]
    checks = {
        "point": (
            point <= gates["equal_dataset_geomean_ratio_at_most"]
        ),
        "hierarchical_bootstrap_upper": (
            upper <= gates["bootstrap_upper_ratio_at_most"]
        ),
        "leave_one_favorable_out": (
            leave_one
            <= gates[
                "leave_one_favorable_dataset_out_ratio_at_most"
            ]
        ),
        "worst_dataset": (
            worst <= gates["worst_dataset_ratio_at_most"]
        ),
    }
    component_counts = {
        name: int(np.count_nonzero(values))
        for name, values in checks.items()
    }
    passing = int(
        np.count_nonzero(
            np.logical_and.reduce(
                tuple(checks[name] for name in checks)
            )
        )
    )
    estimate = passing / outer
    lower = _wilson_lower(passing, outer, wilson_confidence)
    floor = simulation["power_floor"]
    return {
        "candidate": candidate,
        "outer_panel_simulations": outer,
        "outer_seed": simulation["outer_seed"],
        "complete_triplets_preserved": True,
        "prospective_tasks": 12,
        "prospective_coordinates": 36,
        "stratum_composition": {
            stratum: sum(slot["stratum"] == stratum for slot in slots)
            for stratum in STRATA
        },
        "fixed_t5_noop_slots": (
            sum(
                candidate == "t5_composite_policy"
                and not any(slot["t5_size_gate_applicability"])
                for slot in slots
            )
        ),
        "fixed_t5_noop_coordinates": (
            sum(
                candidate == "t5_composite_policy"
                and not applicable
                for slot in slots
                for applicable in slot["t5_size_gate_applicability"]
            )
        ),
        "hierarchical_bootstrap": {
            "seed": simulation["hierarchical_bootstrap_seed"],
            "replicates": replicates,
            "batch": min(
                simulation["hierarchical_bootstrap_batch"], replicates
            ),
            "hierarchy": "lineage_then_three_coordinates_within_lineage",
            "percentile": percentile,
            "numpy_percentile_method": "linear",
        },
        "component_passing_simulations": component_counts,
        "passing_simulations": passing,
        "pass_probability": estimate,
        "wilson_one_sided_confidence": wilson_confidence,
        "wilson_lower_bound": lower,
        "minimum_required_probability": floor,
        "point_estimate_passes": estimate >= floor,
        "wilson_lower_bound_passes": lower >= floor,
        "passes": estimate >= floor and lower >= floor,
        "statistical_gates_only": True,
        "operational_gates_remain_required_at_confirmation": True,
    }


def decide_retention(
    profiles: dict[str, list[dict[str, Any]]],
    *,
    contract: dict[str, Any] | None = None,
    outer_simulations: int | None = None,
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    """Apply the prospectively frozen Bonferroni-then-singleton rule."""
    contract = load_contract() if contract is None else _validate_contract(contract)
    simulation = contract["simulation"]
    initial = {
        candidate: simulate_candidate_power(
            profiles[candidate],
            candidate=candidate,
            percentile=simulation["initial_bootstrap_percentile"],
            wilson_confidence=simulation[
                "initial_power_wilson_confidence"
            ],
            contract=contract,
            outer_simulations=outer_simulations,
            bootstrap_replicates=bootstrap_replicates,
        )
        for candidate in CANDIDATES
    }
    initial_survivors = [
        candidate for candidate in CANDIDATES if initial[candidate]["passes"]
    ]
    singleton = None
    if len(initial_survivors) == 1:
        candidate = initial_survivors[0]
        singleton = simulate_candidate_power(
            profiles[candidate],
            candidate=candidate,
            percentile=simulation["singleton_bootstrap_percentile"],
            wilson_confidence=simulation[
                "singleton_power_wilson_confidence"
            ],
            contract=contract,
            outer_simulations=outer_simulations,
            bootstrap_replicates=bootstrap_replicates,
        )
        retained = [candidate] if singleton["passes"] else []
        alpha = (
            simulation["singleton_per_candidate_one_sided_alpha"]
            if retained
            else None
        )
        percentile = (
            simulation["singleton_bootstrap_percentile"]
            if retained
            else None
        )
    elif len(initial_survivors) == 2:
        retained = list(CANDIDATES)
        alpha = simulation["initial_per_candidate_one_sided_alpha"]
        percentile = simulation["initial_bootstrap_percentile"]
    else:
        retained = []
        alpha = None
        percentile = None
    return {
        "initial_bonferroni_screen": initial,
        "initial_survivors": initial_survivors,
        "singleton_fallback": singleton,
        "retained_candidates": retained,
        "candidate_count": len(retained),
        "familywise_one_sided_alpha": (
            simulation["familywise_one_sided_alpha"]
        ),
        "per_candidate_one_sided_alpha": alpha,
        "bootstrap_percentile": percentile,
        "power_floor": simulation["power_floor"],
        "passes": bool(retained),
    }


def _require_clean_committed_sources() -> str:
    missing = [str(path) for path in SOURCE_PATHS if not path.is_file()]
    if missing:
        raise RuntimeError(f"Panel 3 power-design source is missing: {missing}")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Panel 3 power design requires a clean tree")
    head = _git("rev-parse", "HEAD")
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        try:
            committed = subprocess.run(
                ["git", "show", f"{head}:{relative}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
            ).stdout
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Panel 3 power source is not committed: {relative}"
            ) from exc
        if hashlib.sha256(committed).hexdigest() != common.sha256_file(path):
            raise RuntimeError(
                f"Panel 3 power source differs from HEAD: {relative}"
            )
    return head


def build(
    *,
    summary_path: Path = DEFAULT_SUMMARY,
    raw_path: Path = DEFAULT_RAW,
    require_clean_source: bool = True,
) -> dict[str, Any]:
    execution_head = (
        _require_clean_committed_sources()
        if require_clean_source
        else _git("rev-parse", "HEAD")
    )
    contract, contract_file_sha256 = load_contract_snapshot()
    candidate_contract, candidate_contract_file_sha256 = (
        common.secure_load_json(common.CANDIDATE_CONTRACT)
    )
    if not isinstance(candidate_contract, dict):
        raise RuntimeError("Panel 3 candidate contract is unavailable")
    if (
        summary_path.expanduser().absolute() != DEFAULT_SUMMARY
        or raw_path.expanduser().absolute() != DEFAULT_RAW
    ):
        raise RuntimeError("Panel 3 calibration summary path is unavailable")
    summary, summary_file_sha256 = common.secure_load_json(summary_path)
    profiles = validate_calibration(
        summary,
        summary_path=summary_path,
        raw_path=raw_path,
        verify_raw=True,
        contract=contract,
    )
    h1_head, h1_hashes = validate_h1_design_sources(summary)
    if (
        h1_hashes[str(CONTRACT.relative_to(ROOT))]
        != contract_file_sha256
        or h1_hashes[
            str(common.CANDIDATE_CONTRACT.relative_to(ROOT))
        ]
        != candidate_contract_file_sha256
    ):
        raise RuntimeError(
            "Panel 3 contract snapshots differ from calibration H1"
        )
    runtime = confirmation._validate_runtime_contract(
        candidate_contract
    )
    retention = decide_retention(profiles, contract=contract)
    retained = retention["retained_candidates"]
    checks = {
        "calibration_summary_valid": True,
        "calibration_raw_and_spool_valid": True,
        "complete_39_coordinate_triplets_preserved": True,
        "frozen_4_4_4_composition_preserved": True,
        "known_t5_size_gate_applicability_preserved": True,
        "minimum_stratum_support_preserved": True,
        "candidate_retention_rule_applied_without_discretion": True,
        "design_sources_bound_at_calibration_h1": True,
        "power_decision_computed_from_bound_calibration": True,
        "at_least_one_candidate_meets_power_floor": bool(retained),
    }
    artifact = {
        "schema_version": 1,
        "name": "darkofit_panel3_power_design_decision_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_head": h1_head,
        "decision_execution_head": execution_head,
        "source_sha256": h1_hashes,
        "contract": {
            "path": str(CONTRACT.relative_to(ROOT)),
            "file_sha256": contract_file_sha256,
            "contract_name": contract["contract_name"],
        },
        "calibration": {
            "summary_path": str(summary_path.relative_to(ROOT)),
            "summary_file_sha256": summary_file_sha256,
            "summary_sha256": summary["summary_sha256"],
            "raw_path": str(raw_path.relative_to(ROOT)),
            "raw_file_sha256": summary["raw_file_sha256"],
            "raw_artifact_sha256": summary["raw_artifact_sha256"],
            "source_freeze_sha256": summary["source_freeze_sha256"],
        },
        "runtime": runtime,
        "mapping": {
            "calibration_tasks": contract["calibration"]["tasks"],
            "exchangeability": contract["exchangeability"],
        },
        "pre_h1_target_statistic_exclusions": [
            dict(row)
            for row in common.PRE_H1_TARGET_STATISTIC_EXCLUSIONS
        ],
        "prospective_panel": contract["prospective_panel"],
        "simulation": contract["simulation"],
        "initial_bonferroni_screen": retention[
            "initial_bonferroni_screen"
        ],
        "singleton_fallback": retention["singleton_fallback"],
        "retained_candidates": retained,
        "candidate_count": len(retained),
        "familywise_one_sided_alpha": retention[
            "familywise_one_sided_alpha"
        ],
        "per_candidate_one_sided_alpha": retention[
            "per_candidate_one_sided_alpha"
        ],
        "bootstrap_percentile": retention["bootstrap_percentile"],
        "power_floor": retention["power_floor"],
        "checks": checks,
        "target_preflight_authorized": all(checks.values()),
        "registry_build_authorized": False,
        "confirmation_run_authorized": False,
        "default_promotion_authorized": False,
        "product_claim_authorized": False,
    }
    decision = common.bind_artifact_sha256(artifact, "decision_sha256")
    final_summary, final_summary_file_sha256 = common.secure_load_json(
        summary_path
    )
    final_contract, final_contract_file_sha256 = load_contract_snapshot()
    final_candidate, final_candidate_file_sha256 = common.secure_load_json(
        common.CANDIDATE_CONTRACT
    )
    if (
        final_summary != summary
        or final_summary_file_sha256 != summary_file_sha256
        or final_contract != contract
        or final_contract_file_sha256 != contract_file_sha256
        or final_candidate != candidate_contract
        or final_candidate_file_sha256
        != candidate_contract_file_sha256
    ):
        raise RuntimeError(
            "Panel 3 calibration summary changed during power design"
        )
    return decision


def _validate_power_result(
    result: Any,
    *,
    candidate: str,
    panel: dict[str, Any],
    simulation: dict[str, Any],
    percentile: float,
    wilson_confidence: float,
) -> None:
    expected_fields = {
        "candidate",
        "outer_panel_simulations",
        "outer_seed",
        "complete_triplets_preserved",
        "prospective_tasks",
        "prospective_coordinates",
        "stratum_composition",
        "fixed_t5_noop_slots",
        "fixed_t5_noop_coordinates",
        "hierarchical_bootstrap",
        "component_passing_simulations",
        "passing_simulations",
        "pass_probability",
        "wilson_one_sided_confidence",
        "wilson_lower_bound",
        "minimum_required_probability",
        "point_estimate_passes",
        "wilson_lower_bound_passes",
        "passes",
        "statistical_gates_only",
        "operational_gates_remain_required_at_confirmation",
    }
    slots = panel.get("slots") if isinstance(panel, dict) else None
    if not isinstance(slots, list) or len(slots) != 12:
        raise RuntimeError("Panel 3 power-result panel changed")
    outer = simulation.get("outer_panel_simulations")
    replicates = simulation.get("hierarchical_bootstrap_replicates")
    batch = simulation.get("hierarchical_bootstrap_batch")
    seed = simulation.get("hierarchical_bootstrap_seed")
    outer_seed = simulation.get("outer_seed")
    method = simulation.get("numpy_percentile_method")
    floor = simulation.get("power_floor")
    if (
        not isinstance(result, dict)
        or set(result) != expected_fields
        or result.get("candidate") != candidate
        or type(outer) is not int
        or outer <= 0
        or type(replicates) is not int
        or replicates <= 0
        or type(batch) is not int
        or batch <= 0
        or type(seed) is not int
        or type(outer_seed) is not int
        or method != "linear"
        or not isinstance(floor, (int, float))
        or not math.isfinite(float(floor))
        or result.get("outer_panel_simulations") != outer
        or result.get("outer_seed") != outer_seed
        or result.get("complete_triplets_preserved") is not True
        or result.get("prospective_tasks") != 12
        or result.get("prospective_coordinates") != 36
        or result.get("statistical_gates_only") is not True
        or result.get("operational_gates_remain_required_at_confirmation")
        is not True
    ):
        raise RuntimeError(f"Panel 3 {candidate} power result changed")
    applicability = []
    strata = Counter()
    for slot in slots:
        vector = (
            slot.get("t5_size_gate_applicability")
            if isinstance(slot, dict)
            else None
        )
        stratum = slot.get("stratum") if isinstance(slot, dict) else None
        if (
            not isinstance(vector, list)
            or len(vector) != 3
            or any(type(value) is not bool for value in vector)
            or stratum not in STRATA
        ):
            raise RuntimeError("Panel 3 power-result slot changed")
        applicability.append(vector)
        strata[stratum] += 1
    expected_noop_slots = (
        sum(not any(vector) for vector in applicability)
        if candidate == "t5_composite_policy"
        else 0
    )
    expected_noop_coordinates = (
        sum(not value for vector in applicability for value in vector)
        if candidate == "t5_composite_policy"
        else 0
    )
    bootstrap = result.get("hierarchical_bootstrap")
    if (
        result.get("stratum_composition")
        != {stratum: strata[stratum] for stratum in STRATA}
        or result.get("fixed_t5_noop_slots") != expected_noop_slots
        or result.get("fixed_t5_noop_coordinates")
        != expected_noop_coordinates
        or not isinstance(bootstrap, dict)
        or set(bootstrap)
        != {
            "seed",
            "replicates",
            "batch",
            "hierarchy",
            "percentile",
            "numpy_percentile_method",
        }
        or bootstrap.get("seed") != seed
        or bootstrap.get("replicates") != replicates
        or bootstrap.get("batch") != min(batch, replicates)
        or bootstrap.get("hierarchy")
        != "lineage_then_three_coordinates_within_lineage"
        or bootstrap.get("percentile") != percentile
        or bootstrap.get("numpy_percentile_method") != method
    ):
        raise RuntimeError(f"Panel 3 {candidate} power geometry changed")
    components = result.get("component_passing_simulations")
    component_names = {
        "point",
        "hierarchical_bootstrap_upper",
        "leave_one_favorable_out",
        "worst_dataset",
    }
    passing = result.get("passing_simulations")
    probability = result.get("pass_probability")
    lower = result.get("wilson_lower_bound")
    if (
        not isinstance(components, dict)
        or set(components) != component_names
        or any(
            type(count) is not int or not 0 <= count <= outer
            for count in components.values()
        )
        or type(passing) is not int
        or not 0 <= passing <= min(components.values())
        or not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
        or float(probability) != passing / outer
        or result.get("wilson_one_sided_confidence")
        != wilson_confidence
        or not isinstance(lower, (int, float))
        or not math.isfinite(float(lower))
        or not math.isclose(
            float(lower),
            _wilson_lower(passing, outer, wilson_confidence),
            rel_tol=1e-15,
            abs_tol=1e-15,
        )
        or result.get("minimum_required_probability") != floor
        or result.get("point_estimate_passes")
        is not (float(probability) >= float(floor))
        or result.get("wilson_lower_bound_passes")
        is not (float(lower) >= float(floor))
        or result.get("passes")
        is not (
            float(probability) >= float(floor)
            and float(lower) >= float(floor)
        )
    ):
        raise RuntimeError(f"Panel 3 {candidate} power arithmetic changed")


def validate_decision(
    artifact: Any,
    *,
    decision_path: Path = DEFAULT_OUTPUT,
    require_current_sources: bool = True,
    recompute: bool = True,
) -> dict[str, Any]:
    if recompute and not require_current_sources:
        raise RuntimeError(
            "Panel 3 historical recomputation requires the bound "
            "calibration files"
        )
    if require_current_sources:
        contract, contract_file_sha256 = load_contract_snapshot()
        candidate_contract, candidate_contract_file_sha256 = (
            common.secure_load_json(common.CANDIDATE_CONTRACT)
        )
        if not isinstance(candidate_contract, dict):
            raise RuntimeError("Panel 3 candidate contract is unavailable")
    else:
        contract = None
        contract_file_sha256 = None
        candidate_contract = None
        candidate_contract_file_sha256 = None
    if (
        not isinstance(artifact, dict)
        or set(artifact) != DECISION_FIELDS
        or artifact.get("schema_version") != 1
        or artifact.get("name")
        != "darkofit_panel3_power_design_decision_v1"
        or not _is_sha256(artifact.get("decision_sha256"))
    ):
        raise RuntimeError("Panel 3 power-design decision contract changed")
    common.verify_artifact_sha256(artifact, "decision_sha256")
    created_at = artifact.get("created_at")
    try:
        parsed_created_at = datetime.fromisoformat(created_at)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Panel 3 power-design creation timestamp changed"
        ) from exc
    if (
        parsed_created_at.tzinfo is None
        or parsed_created_at.utcoffset() is None
    ):
        raise RuntimeError(
            "Panel 3 power-design creation timestamp changed"
        )
    _validate_decision_runtime(
        artifact.get("runtime"),
        require_current_sources=require_current_sources,
        candidate_contract=candidate_contract,
    )
    if require_current_sources:
        confirmation._validate_candidate_power_coherence(
            candidate_contract,
            artifact,
        )
    for field in ("source_head", "decision_execution_head"):
        value = artifact.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"Panel 3 {field} is invalid")
    contract_binding = artifact.get("contract")
    if (
        not isinstance(contract_binding, dict)
        or set(contract_binding)
        != {"path", "file_sha256", "contract_name"}
        or contract_binding.get("path")
        != "benchmarks/panel3_power_design_contract.json"
        or not _is_sha256(contract_binding.get("file_sha256"))
        or contract_binding.get("contract_name")
        != "darkofit_panel3_authorization_power_design_v1"
        or (
            require_current_sources
            and contract_binding
            != {
                "path": str(CONTRACT.relative_to(ROOT)),
                "file_sha256": contract_file_sha256,
                "contract_name": contract["contract_name"],
            }
        )
    ):
        raise RuntimeError("Panel 3 power-design contract binding changed")
    source_hashes = artifact.get("source_sha256")
    if (
        not isinstance(source_hashes, dict)
        or not source_hashes
        or set(source_hashes) != set(PANEL3_V1_SOURCE_RELATIVE_PATHS)
        or any(not _is_sha256(value) for value in source_hashes.values())
        or any(
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            for relative in source_hashes
        )
    ):
        raise RuntimeError("Panel 3 power-design source map changed")
    if (
        contract_binding["file_sha256"]
        != source_hashes[str(CONTRACT.relative_to(ROOT))]
        or contract_binding["file_sha256"]
        != PANEL3_V1_POWER_CONTRACT_SHA256
        or source_hashes[
            str(common.CANDIDATE_CONTRACT.relative_to(ROOT))
        ]
        != PANEL3_V1_CANDIDATE_CONTRACT_SHA256
        or source_hashes[
            str(common.ENVIRONMENT_CONTRACT.relative_to(ROOT))
        ]
        != PANEL3_V1_ENVIRONMENT_CONTRACT_SHA256
    ):
        raise RuntimeError("Panel 3 power-design provenance binding changed")
    if require_current_sources:
        expected_sources = {
            str(path.relative_to(ROOT)): common.sha256_file(path)
            for path in SOURCE_PATHS
        }
        expected_sources[str(CONTRACT.relative_to(ROOT))] = (
            contract_file_sha256
        )
        expected_sources[
            str(common.CANDIDATE_CONTRACT.relative_to(ROOT))
        ] = candidate_contract_file_sha256
        if source_hashes != expected_sources:
            raise RuntimeError("Panel 3 power-design source binding changed")
        for relative, digest in source_hashes.items():
            try:
                frozen = subprocess.run(
                    ["git", "show", f"{artifact['source_head']}:{relative}"],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                ).stdout
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Panel 3 H1 source is unavailable: {relative}"
                ) from exc
            if hashlib.sha256(frozen).hexdigest() != digest:
                raise RuntimeError(
                    f"Panel 3 H1 source digest changed: {relative}"
                )
        if (
            artifact.get("mapping")
            != {
                "calibration_tasks": contract["calibration"]["tasks"],
                "exchangeability": contract["exchangeability"],
            }
            or artifact.get("prospective_panel")
            != contract["prospective_panel"]
            or artifact.get("simulation") != contract["simulation"]
            or artifact.get("power_floor")
            != contract["simulation"]["power_floor"]
            or artifact.get("familywise_one_sided_alpha")
            != contract["simulation"]["familywise_one_sided_alpha"]
        ):
            raise RuntimeError(
                "Panel 3 power-design simulation binding changed"
            )
    else:
        mapping = artifact.get("mapping")
        panel = artifact.get("prospective_panel")
        simulation = artifact.get("simulation")
        if (
            mapping
            != {
                "calibration_tasks": PANEL3_V1_CALIBRATION_TASKS,
                "exchangeability": PANEL3_V1_EXCHANGEABILITY,
            }
            or panel != PANEL3_V1_PROSPECTIVE_PANEL
            or simulation != PANEL3_V1_SIMULATION
            or artifact.get("power_floor") != simulation.get("power_floor")
            or artifact.get("familywise_one_sided_alpha")
            != simulation.get("familywise_one_sided_alpha")
        ):
            raise RuntimeError(
                "Panel 3 historical power-design structure changed"
            )
    panel = artifact.get("prospective_panel")
    simulation = artifact.get("simulation")
    _validate_pre_h1_target_exclusion_slots(
        panel,
        artifact.get("pre_h1_target_statistic_exclusions"),
        require_current_sources=require_current_sources,
    )
    retained = artifact.get("retained_candidates")
    if (
        not isinstance(retained, list)
        or any(candidate not in CANDIDATES for candidate in retained)
        or retained != [
            candidate for candidate in CANDIDATES if candidate in retained
        ]
        or artifact.get("candidate_count") != len(retained)
    ):
        raise RuntimeError("Panel 3 retained-candidate set changed")
    initial = artifact.get("initial_bonferroni_screen")
    if not isinstance(initial, dict) or set(initial) != set(CANDIDATES):
        raise RuntimeError("Panel 3 initial power screen changed")
    for candidate in CANDIDATES:
        _validate_power_result(
            initial[candidate],
            candidate=candidate,
            panel=panel,
            simulation=simulation,
            percentile=simulation["initial_bootstrap_percentile"],
            wilson_confidence=simulation[
                "initial_power_wilson_confidence"
            ],
        )
    initial_survivors = [
        candidate
        for candidate in CANDIDATES
        if isinstance(initial, dict)
        and isinstance(initial.get(candidate), dict)
        and initial[candidate].get("passes") is True
    ]
    singleton = artifact.get("singleton_fallback")
    if singleton is not None:
        singleton_candidate = (
            singleton.get("candidate")
            if isinstance(singleton, dict)
            else None
        )
        if singleton_candidate not in CANDIDATES:
            raise RuntimeError("Panel 3 singleton power result changed")
        _validate_power_result(
            singleton,
            candidate=singleton_candidate,
            panel=panel,
            simulation=simulation,
            percentile=simulation["singleton_bootstrap_percentile"],
            wilson_confidence=simulation[
                "singleton_power_wilson_confidence"
            ],
        )
    if len(initial_survivors) == 2:
        expected_retained = list(CANDIDATES)
        expected_alpha = 0.025
        expected_percentile = 97.5
        if singleton is not None:
            raise RuntimeError("Panel 3 unexpected singleton fallback")
    elif len(initial_survivors) == 1:
        if (
            not isinstance(singleton, dict)
            or singleton.get("candidate") != initial_survivors[0]
        ):
            raise RuntimeError(
                "Panel 3 singleton fallback does not match the "
                "Bonferroni survivor"
            )
        expected_retained = (
            initial_survivors if singleton.get("passes") is True else []
        )
        expected_alpha = 0.05 if expected_retained else None
        expected_percentile = 95.0 if expected_retained else None
    else:
        expected_retained = []
        expected_alpha = None
        expected_percentile = None
        if singleton is not None:
            raise RuntimeError("Panel 3 singleton rescue is not allowed")
    if (
        retained != expected_retained
        or artifact.get("per_candidate_one_sided_alpha") != expected_alpha
        or artifact.get("bootstrap_percentile") != expected_percentile
    ):
        raise RuntimeError("Panel 3 candidate-retention decision changed")
    checks = artifact.get("checks")
    expected_check_keys = {
        "calibration_summary_valid",
        "calibration_raw_and_spool_valid",
        "complete_39_coordinate_triplets_preserved",
        "frozen_4_4_4_composition_preserved",
        "known_t5_size_gate_applicability_preserved",
        "minimum_stratum_support_preserved",
        "candidate_retention_rule_applied_without_discretion",
        "design_sources_bound_at_calibration_h1",
        "power_decision_computed_from_bound_calibration",
        "at_least_one_candidate_meets_power_floor",
    }
    if (
        not isinstance(checks, dict)
        or set(checks) != expected_check_keys
        or any(type(value) is not bool for value in checks.values())
        or checks["at_least_one_candidate_meets_power_floor"]
        is not bool(retained)
        or artifact.get("target_preflight_authorized")
        is not all(checks.values())
        or artifact.get("registry_build_authorized") is not False
        or artifact.get("confirmation_run_authorized") is not False
        or artifact.get("default_promotion_authorized") is not False
        or artifact.get("product_claim_authorized") is not False
    ):
        raise RuntimeError("Panel 3 authorization flags changed")
    calibration_binding = artifact.get("calibration")
    if (
        not isinstance(calibration_binding, dict)
        or calibration_binding.get("summary_path")
        != str(DEFAULT_SUMMARY.relative_to(ROOT))
        or calibration_binding.get("raw_path")
        != str(DEFAULT_RAW.relative_to(ROOT))
        or not all(
            _is_sha256(calibration_binding.get(field))
            for field in (
                "summary_file_sha256",
                "summary_sha256",
                "raw_file_sha256",
                "raw_artifact_sha256",
                "source_freeze_sha256",
            )
        )
    ):
        raise RuntimeError("Panel 3 calibration binding changed")
    if require_current_sources:
        current_artifact, _decision_file_sha256 = (
            common.secure_load_json(decision_path)
        )
        summary, summary_file_sha256 = common.secure_load_json(
            DEFAULT_SUMMARY
        )
        _raw, raw_file_sha256 = common.secure_load_json(DEFAULT_RAW)
        if (
            decision_path.expanduser().absolute() != DEFAULT_OUTPUT
            or summary_file_sha256
            != calibration_binding["summary_file_sha256"]
            or raw_file_sha256 != calibration_binding["raw_file_sha256"]
        ):
            raise RuntimeError("Panel 3 power-design artifact files changed")
        if current_artifact != artifact:
            raise RuntimeError(
                "Panel 3 embedded power decision differs from its "
                "create-only file"
            )
        if summary.get("summary_sha256") != calibration_binding[
            "summary_sha256"
        ]:
            raise RuntimeError("Panel 3 calibration summary hash changed")
        h1_head, h1_hashes = validate_h1_design_sources(summary)
        if (
            artifact["source_head"] != h1_head
            or artifact["source_sha256"] != h1_hashes
        ):
            raise RuntimeError("Panel 3 calibration H1 source binding changed")
        if recompute:
            profiles = validate_calibration(
                summary,
                summary_path=DEFAULT_SUMMARY,
                raw_path=DEFAULT_RAW,
                verify_raw=True,
                contract=contract,
            )
            retention = decide_retention(profiles, contract=contract)
            expected_projection = {
                "initial_bonferroni_screen": retention[
                    "initial_bonferroni_screen"
                ],
                "singleton_fallback": retention["singleton_fallback"],
                "retained_candidates": retention["retained_candidates"],
                "candidate_count": retention["candidate_count"],
                "familywise_one_sided_alpha": retention[
                    "familywise_one_sided_alpha"
                ],
                "per_candidate_one_sided_alpha": retention[
                    "per_candidate_one_sided_alpha"
                ],
                "bootstrap_percentile": retention[
                    "bootstrap_percentile"
                ],
                "power_floor": retention["power_floor"],
            }
            observed_projection = {
                field: artifact[field] for field in expected_projection
            }
            if observed_projection != expected_projection:
                raise RuntimeError(
                    "Panel 3 power decision differs from frozen recomputation"
                )
    return artifact


def load_decision_snapshot(
    *,
    require_current_sources: bool = True,
    recompute: bool = True,
) -> tuple[dict[str, Any], str]:
    try:
        artifact, file_sha256 = common.secure_load_json(DEFAULT_OUTPUT)
    except RuntimeError as exc:
        raise RuntimeError(
            "Panel 3 power-design decision is absent; target access is blocked"
        ) from exc
    if not isinstance(artifact, dict):
        raise RuntimeError(
            "Panel 3 power-design decision is absent; target access is blocked"
        )
    return (
        validate_decision(
            artifact,
            require_current_sources=require_current_sources,
            recompute=recompute,
        ),
        file_sha256,
    )


def load_decision(
    *,
    require_current_sources: bool = True,
    recompute: bool = True,
) -> dict[str, Any]:
    return load_decision_snapshot(
        require_current_sources=require_current_sources,
        recompute=recompute,
    )[0]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        args.summary.expanduser().absolute() != DEFAULT_SUMMARY
        or args.raw.expanduser().absolute() != DEFAULT_RAW
        or args.output.expanduser().absolute() != DEFAULT_OUTPUT
    ):
        raise RuntimeError("Panel 3 power decision path changed")
    artifact = build(summary_path=args.summary, raw_path=args.raw)
    encoded = (
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if (
        _require_clean_committed_sources()
        != artifact["decision_execution_head"]
    ):
        raise RuntimeError(
            "Panel 3 power-design source changed before publication"
        )
    summary, summary_file_sha256 = common.secure_load_json(args.summary)
    raw, raw_file_sha256 = common.secure_load_json(args.raw)
    contract, contract_file_sha256 = load_contract_snapshot()
    candidate_contract, candidate_contract_file_sha256 = (
        common.secure_load_json(common.CANDIDATE_CONTRACT)
    )
    calibration_binding = artifact["calibration"]
    if (
        not isinstance(summary, dict)
        or not isinstance(raw, dict)
        or not isinstance(candidate_contract, dict)
        or summary_file_sha256
        != calibration_binding["summary_file_sha256"]
        or summary.get("summary_sha256")
        != calibration_binding["summary_sha256"]
        or raw_file_sha256 != calibration_binding["raw_file_sha256"]
        or raw.get("raw_artifact_sha256")
        != calibration_binding["raw_artifact_sha256"]
        or contract_file_sha256 != artifact["contract"]["file_sha256"]
        or contract.get("contract_name")
        != artifact["contract"]["contract_name"]
        or candidate_contract_file_sha256
        != artifact["source_sha256"][
            str(common.CANDIDATE_CONTRACT.relative_to(ROOT))
        ]
    ):
        raise RuntimeError(
            "Panel 3 calibration evidence changed before publication"
        )
    # Reopen and revalidate the complete raw -> source-freeze -> spool chain
    # after the power simulation. A top-level raw hash alone cannot detect a
    # spool replacement that leaves the immutable raw bytes unchanged.
    validate_calibration(
        summary,
        summary_path=args.summary,
        raw_path=args.raw,
        verify_raw=True,
        contract=contract,
    )
    final_summary, final_summary_file_sha256 = common.secure_load_json(
        args.summary
    )
    final_raw, final_raw_file_sha256 = common.secure_load_json(args.raw)
    if (
        final_summary != summary
        or final_summary_file_sha256 != summary_file_sha256
        or final_raw != raw
        or final_raw_file_sha256 != raw_file_sha256
    ):
        raise RuntimeError(
            "Panel 3 calibration evidence changed during final validation"
        )
    common.atomic_create(args.output, encoded)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision_sha256": artifact["decision_sha256"],
                "retained_candidates": artifact["retained_candidates"],
                "target_preflight_authorized": artifact[
                    "target_preflight_authorized"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
