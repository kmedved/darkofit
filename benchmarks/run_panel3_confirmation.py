#!/usr/bin/env python3
"""Run the frozen Panel 3 retained-candidate confirmation campaign."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
CHIMERA_ROOT = ROOT.parent / "chimeraboost"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import basketball_harness as basketball  # noqa: E402
from benchmarks import build_panel3_registry as registry_builder  # noqa: E402
from benchmarks import build_panel3_power_design as power_design  # noqa: E402
from benchmarks import build_ctr23_contamination_registry as fingerprints  # noqa: E402
from benchmarks import panel3_data_contract as data_contract  # noqa: E402
from benchmarks import panel3_registry_common as common  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402
from benchmarks import run_t5_composite_confirmation as t5  # noqa: E402
from benchmarks.run_smooth_cross_features import candidate_pairs  # noqa: E402


DEFAULT_REGISTRY = ROOT / "benchmarks" / "panel3_registry.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "panel3_confirmation_raw.json"
DEFAULT_SPOOL_DIRECTORY = ROOT / ".cache" / "panel3-confirmation-spool-v1"
ARM_ORDER = (
    "current_default",
    "t5_composite_policy",
    "guarded_cross_features_policy",
    "chimeraboost_0_15_0",
    "catboost_product_default",
)
CANDIDATE_ARMS = (
    "t5_composite_policy",
    "guarded_cross_features_policy",
)
COMPARATOR_ARMS = (
    "chimeraboost_0_15_0",
    "catboost_product_default",
)
CONTROL_ARM = "current_default"
DECISION_ARMS = (CONTROL_ARM, *CANDIDATE_ARMS)
RUNNER_IMPLEMENTATION_COMPLETE = True
RANDOM_STATE = 4
THREADS_PER_WORKER = 6
CONCURRENT_WORKERS = 3
VALIDATION_FRACTION = 0.20
GUARDED_CROSS_RATIO = 0.95
PREDICTION_BLOCK_SECONDS = 0.25
PREDICTION_MIN_CALLS = 5
PREDICTION_MAX_CALLS = 20_000
WORKER_PREFIX = "PANEL3_WORKER_RESULT="
CAMPAIGN_INVALIDATION_FILENAME = "CAMPAIGN_INVALIDATED.json"
WORKER_SPOOL_DIRECTORY_ENV = "DARKOFIT_PANEL3_SPOOL_DIRECTORY"
WORKER_BINDING_SHA256_ENV = "DARKOFIT_PANEL3_BINDING_SHA256"
WORKER_BINDING_JSON_ENV = "DARKOFIT_PANEL3_BINDING_JSON"
RUNTIME_PACKAGE_NAMES = (
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
    "schema_version": 1,
    "contract_name": "darkofit_panel3_exact_runtime_environment_v1",
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
PANEL3_V1_REGISTRY_SOURCE_RELATIVE_PATHS = (
    "benchmarks/panel3_registry_declarations.json",
    "benchmarks/panel3_registry_protocol.md",
    "benchmarks/panel3_registry_common.py",
    "benchmarks/panel3_data_contract.py",
    "benchmarks/panel3_candidate_contract.json",
    "benchmarks/panel3_environment_contract.json",
    "benchmarks/panel3_power_design_contract.json",
    "benchmarks/panel3_power_design_protocol.md",
    "benchmarks/build_panel3_power_design.py",
    "benchmarks/preflight_panel3_registry.py",
    "benchmarks/confirmation_target_preflight.py",
    "benchmarks/build_ctr23_contamination_registry.py",
    "benchmarks/build_fresh_confirmation_registry.py",
    "benchmarks/build_panel3_registry.py",
    "benchmarks/run_panel3_confirmation.py",
    "benchmarks/analyze_panel3_confirmation.py",
    "benchmarks/t5_composite_registry_protocol.md",
    "benchmarks/run_t5_composite_confirmation.py",
    "benchmarks/smooth_cross_features_protocol.md",
    "benchmarks/smooth_cross_margin_analysis.json",
    "benchmarks/panel3_registry_declarations_invalid_draft.md",
)
PANEL3_V1_FROZEN_EVIDENCE_RELATIVE_PATHS = (
    "benchmarks/ctr23_suite_snapshot.json",
    "benchmarks/ctr23_partition.json",
    "benchmarks/ctr23_contamination_sources.json",
    "benchmarks/fresh_confirmation_registry.json",
    "benchmarks/fresh_confirmation_registry_v2.json",
    "benchmarks/native_ordinal_c2_registry.json",
    "benchmarks/t5_composite_registry_declarations.json",
    "benchmarks/t5_composite_registry.json",
    "benchmarks/t5_composite_registry_invalid_attempt.md",
    "benchmarks/t7_catboost_attribution_raw.json",
    "benchmarks/t7_catboost_attribution_summary.json",
    "benchmarks/t8_distributional_flagship_protocol.md",
    "benchmarks/t8_distributional_flagship_raw.csv",
    "benchmarks/t8_distributional_flagship_result.md",
    "benchmarks/smooth_cross_features.json",
    "benchmarks/smooth_cross_margin_analysis.json",
    "benchmarks/SHIPPING_POLICY.md",
)
PANEL3_V1_LOCKBOX_REFERENCE_ALLOWLIST = (
    "BEYOND_PARITY_PLAN.md",
    "benchmarks/ctr23_contamination_registry.json",
    "benchmarks/ctr23_contamination_sources.json",
    "benchmarks/ctr23_partition.json",
    "benchmarks/ctr23_suite_snapshot.json",
)
PANEL3_V1_CHIMERA_EXPOSURE_SOURCE_PATHS = (
    "benchmarks/run_benchmarks.py",
    "benchmarks/synthgen/corpus_marginals.json",
    "tests/test_highcard.py",
)
PANEL3_V1_CANDIDATES = [
    {
        "name": "t5_composite_policy",
        "source_protocol": "benchmarks/t5_composite_registry_protocol.md",
        "source_runner": "benchmarks/run_t5_composite_confirmation.py",
        "definition": {
            "minimum_outer_training_rows": 2_000,
            "n_estimators": 10_000,
            "learning_rate": 0.1,
            "l2_leaf_reg": 3.0,
            "max_bins": 128,
            "ts_permutations": 1,
            "tree_mode": "auto",
            "tree_mode_selection_rounds": 100,
            "linear_leaf_race": (
                "full_budget_only_when_catboost_tree_mode_wins"
            ),
            "cross_features": (
                "top_six_numeric_split_gain_diff_and_product_pairs"
            ),
            "cross_guard_ratio": 0.95,
            "candidate_guard_ratio": 0.995,
            "refit": (
                "selected_resolved_learning_rate_and_best_prefix_on_all_"
                "outer_training_rows"
            ),
            "decline": "byte_exact_current_default_full_data_fit",
        },
    },
    {
        "name": "guarded_cross_features_policy",
        "source_protocol": "benchmarks/smooth_cross_features_protocol.md",
        "source_analysis": "benchmarks/smooth_cross_margin_analysis.json",
        "definition": {
            "n_estimators": 2_000,
            "learning_rate": 0.1,
            "depth": 6,
            "l2_leaf_reg": 1.0,
            "max_bins": 128,
            "min_child_weight": 1.0,
            "tree_mode": "catboost",
            "constant_linear_race": "full_budget",
            "cross_features": (
                "top_six_numeric_split_gain_diff_and_product_pairs"
            ),
            "cross_guard_ratio": 0.95,
            "refit": (
                "selected_lane_and_best_prefix_on_all_outer_training_rows"
            ),
            "decline": (
                "uncrossed_constant_or_linear_validation_winner"
            ),
        },
    },
]
PANEL3_V1_CANDIDATE_CONTRACT = {
    "schema_version": 1,
    "contract_name": "darkofit_panel3_dual_candidate_contract_v1",
    "random_state": 4,
    "thread_count": 6,
    "runtime": {
        "path": "benchmarks/panel3_environment_contract.json",
        "sha256": (
            "9755529964eda51024a6a02b9bad46b4099b13d38621067e4ade1cc4232046a0"
        ),
    },
    "ordinal_features": {
        "source": (
            "panel3_registry_declarations.json:ordinal_features_by_task"
        ),
        "policy": "explicit_per_task_only",
        "all_declared_maps_empty": True,
        "reason": (
            "no complete source-ordered category maps survived the source audit"
        ),
    },
    "inner_validation": {
        "kind": "ShuffleSplit",
        "test_size": 0.2,
        "random_state": 4,
    },
    "control": {
        "name": "current_default",
        "estimator": "DarkoRegressor",
        "constructor": {"random_state": 4, "thread_count": 6},
        "fit_on_all_outer_training_rows": True,
    },
    "candidates": PANEL3_V1_CANDIDATES,
    "comparators": [
        {
            "name": "chimeraboost_0_15_0",
            "decision_role": "descriptive_only",
            "required_for_candidate_decision": False,
            "failure_policy": (
                "persist_immutable_failure_record_and_continue"
            ),
        },
        {
            "name": "catboost_product_default",
            "decision_role": "descriptive_only",
            "required_for_candidate_decision": False,
            "failure_policy": (
                "persist_immutable_failure_record_and_continue"
            ),
        },
    ],
    "decision": {
        "frozen_candidate_hypothesis_count": 2,
        "post_outcome_winner_selection_allowed": False,
        "each_candidate_adjudicated_separately": True,
        "default_selection_mapping": {
            "neither_passes": None,
            "only_guarded_cross_features_policy_passes": (
                "guarded_cross_features_policy"
            ),
            "only_t5_composite_policy_passes": "t5_composite_policy",
            "both_pass": "t5_composite_policy",
        },
        "both_pass_reason": (
            "t5_composite_policy contains guarded crosses as a constituent, "
            "so the broader frozen policy has fixed precedence without metric "
            "ranking or post-outcome discretion"
        ),
        "independent_constituent_confirmation_preserved": True,
        "familywise_one_sided_alpha": 0.05,
        "multiplicity_source": (
            "benchmarks/panel3_power_design_decision.json"
        ),
        "retained_candidate_multiplicity": {
            "two": {
                "candidate_count": 2,
                "per_candidate_one_sided_alpha": 0.025,
                "bootstrap_percentile": 97.5,
                "execution_authorized": True,
            },
            "one": {
                "candidate_count": 1,
                "per_candidate_one_sided_alpha": 0.05,
                "bootstrap_percentile": 95.0,
                "execution_authorized": True,
            },
            "zero": {
                "candidate_count": 0,
                "execution_authorized": False,
            },
        },
        "equal_dataset_geomean_ratio_at_most": 0.995,
        "bootstrap_upper_ratio_at_most": 1.002,
        "leave_one_favorable_dataset_out_ratio_at_most": 0.998,
        "worst_dataset_ratio_at_most": 1.005,
        "fit_seconds_ratio_at_most": 6.0,
        "worst_dataset_fit_seconds_ratio_at_most": 12.0,
        "predict_seconds_ratio_at_most": 1.5,
        "peak_rss_ratio_at_most": 2.5,
    },
}


def _candidate_arms(registry: dict[str, Any]) -> tuple[str, ...]:
    retained = registry.get("retained_candidates")
    if (
        not isinstance(retained, list)
        or not retained
        or retained
        != [candidate for candidate in CANDIDATE_ARMS if candidate in retained]
    ):
        raise RuntimeError("panel-3 retained candidate set is invalid")
    return tuple(retained)


def _decision_arms(registry: dict[str, Any]) -> tuple[str, ...]:
    return (CONTROL_ARM, *_candidate_arms(registry))


def _arm_order(registry: dict[str, Any]) -> tuple[str, ...]:
    return (*_decision_arms(registry), *COMPARATOR_ARMS)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value: Any, dtype: str | None = "<f8") -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _json_file_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _json_file_sha256(value: Any) -> str:
    return hashlib.sha256(_json_file_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite JSON number: {value}")
    return result


def _json_int(value: str) -> int:
    result = int(value)
    if not -(2**63) <= result <= 2**63 - 1:
        raise ValueError(f"out-of-range JSON integer: {value}")
    return result


def _json_loads(encoded: str, label: str) -> Any:
    try:
        return json.loads(
            encoded,
            object_pairs_hook=_json_object,
            parse_float=_json_float,
            parse_int=_json_int,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid panel-3 {label} JSON") from exc


def _artifact_path(path: Path) -> str:
    absolute = path.expanduser().absolute()
    try:
        return str(absolute.relative_to(ROOT))
    except ValueError:
        return str(absolute)


def _validate_candidate_contract_taxonomy(
    candidate_contract: dict[str, Any],
    *,
    source_sha256: dict[str, str] | None = None,
) -> None:
    """Validate the complete arm taxonomy and its immutable references."""
    runtime = candidate_contract.get("runtime")
    environment_relative = "benchmarks/panel3_environment_contract.json"
    if (
        not isinstance(candidate_contract, dict)
        or candidate_contract != PANEL3_V1_CANDIDATE_CONTRACT
        or not isinstance(runtime, dict)
        or set(runtime) != {"path", "sha256"}
        or runtime["path"] != environment_relative
        or not _is_sha256(runtime["sha256"])
    ):
        raise RuntimeError("panel-3 candidate contract taxonomy changed")
    if source_sha256 is not None and (
        not isinstance(source_sha256, dict)
        or source_sha256.get(environment_relative) != runtime["sha256"]
    ):
        raise RuntimeError(
            "panel-3 runtime-contract source binding changed"
        )


def _validate_embedded_runtime_contract(
    runtime: Any,
    candidate_contract: dict[str, Any],
    *,
    source_sha256: dict[str, str] | None = None,
) -> None:
    """Validate archived runtime evidence without reopening a live file."""
    _validate_candidate_contract_taxonomy(
        candidate_contract,
        source_sha256=source_sha256,
    )
    if runtime != PANEL3_V1_RUNTIME_CONTRACT:
        raise RuntimeError("panel-3 embedded runtime contract changed")


def _validate_runtime_contract(
    candidate_contract: dict[str, Any],
) -> dict[str, Any]:
    """Reject any interpreter or package drift before loading data."""
    _validate_candidate_contract_taxonomy(candidate_contract)
    reference = candidate_contract.get("runtime")
    if (
        not isinstance(reference, dict)
        or set(reference) != {"path", "sha256"}
        or not isinstance(reference["path"], str)
        or not _is_sha256(reference["sha256"])
    ):
        raise RuntimeError("panel-3 runtime-contract reference changed")
    path = (ROOT / reference["path"]).absolute()
    if not path.is_relative_to(ROOT.absolute()):
        raise RuntimeError("panel-3 runtime-contract file changed")
    runtime, runtime_file_sha256 = common.secure_load_json(
        path,
        allowed_root=ROOT,
    )
    if runtime_file_sha256 != reference["sha256"]:
        raise RuntimeError("panel-3 runtime-contract file changed")
    if (
        runtime != PANEL3_V1_RUNTIME_CONTRACT
        or runtime["python_implementation"] != sys.implementation.name
        or runtime["python_version"]
        != ".".join(str(value) for value in sys.version_info[:3])
    ):
        raise RuntimeError("panel-3 runtime contract changed")
    installed = {}
    for package in RUNTIME_PACKAGE_NAMES:
        expected = runtime["packages"].get(package)
        if not isinstance(expected, str) or not expected:
            raise RuntimeError(
                f"panel-3 expected version is invalid for {package}"
            )
        try:
            observed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"panel-3 required package is missing: {package}"
            ) from exc
        if observed != expected:
            raise RuntimeError(
                f"panel-3 {package} version changed: "
                f"{observed} != {expected}"
            )
        installed[package] = observed
    evidence = {**runtime, "packages": installed}
    _validate_embedded_runtime_contract(evidence, candidate_contract)
    return evidence


def _validate_candidate_power_coherence(
    candidate_contract: dict[str, Any],
    power_decision: dict[str, Any],
) -> None:
    """Require the candidate contract and retained-family decision to agree."""
    _validate_candidate_contract_taxonomy(candidate_contract)
    decision_contract = candidate_contract.get("decision")
    retained = power_decision.get("retained_candidates")
    simulation = power_decision.get("simulation")
    quality_gates = (
        simulation.get("quality_gates")
        if isinstance(simulation, dict)
        else None
    )
    contract_quality_gates = (
        {
            "equal_dataset_geomean_ratio_at_most": decision_contract.get(
                "equal_dataset_geomean_ratio_at_most"
            ),
            "bootstrap_upper_ratio_at_most": decision_contract.get(
                "bootstrap_upper_ratio_at_most"
            ),
            "leave_one_favorable_dataset_out_ratio_at_most": (
                decision_contract.get(
                    "leave_one_favorable_dataset_out_ratio_at_most"
                )
            ),
            "worst_dataset_ratio_at_most": decision_contract.get(
                "worst_dataset_ratio_at_most"
            ),
        }
        if isinstance(decision_contract, dict)
        else None
    )
    expected_selection_mapping = {
        "neither_passes": None,
        "only_guarded_cross_features_policy_passes": (
            "guarded_cross_features_policy"
        ),
        "only_t5_composite_policy_passes": "t5_composite_policy",
        "both_pass": "t5_composite_policy",
    }
    expected_operational_gates = {
        "fit_seconds_ratio_at_most": 6.0,
        "worst_dataset_fit_seconds_ratio_at_most": 12.0,
        "predict_seconds_ratio_at_most": 1.5,
        "peak_rss_ratio_at_most": 2.5,
    }
    if (
        not isinstance(decision_contract, dict)
        or decision_contract.get("frozen_candidate_hypothesis_count") != 2
        or decision_contract.get("post_outcome_winner_selection_allowed")
        is not False
        or decision_contract.get("each_candidate_adjudicated_separately")
        is not True
        or decision_contract.get("default_selection_mapping")
        != expected_selection_mapping
        or not isinstance(decision_contract.get("both_pass_reason"), str)
        or not decision_contract["both_pass_reason"]
        or decision_contract.get(
            "independent_constituent_confirmation_preserved"
        )
        is not True
        or decision_contract.get("familywise_one_sided_alpha")
        != power_decision.get("familywise_one_sided_alpha")
        or decision_contract.get("multiplicity_source")
        != "benchmarks/panel3_power_design_decision.json"
        or contract_quality_gates != quality_gates
        or {
            key: decision_contract.get(key)
            for key in expected_operational_gates
        }
        != expected_operational_gates
        or not isinstance(retained, list)
        or retained
        != [candidate for candidate in CANDIDATE_ARMS if candidate in retained]
        or len(retained) not in (0, 1, 2)
        or power_decision.get("candidate_count") != len(retained)
        or power_decision.get("familywise_one_sided_alpha") != 0.05
    ):
        raise RuntimeError(
            "panel-3 candidate/power multiplicity contract changed"
        )
    multiplicity = decision_contract.get("retained_candidate_multiplicity")
    label = {0: "zero", 1: "one", 2: "two"}[len(retained)]
    rule = multiplicity.get(label) if isinstance(multiplicity, dict) else None
    if len(retained) == 0:
        expected = {
            "candidate_count": 0,
            "execution_authorized": False,
        }
        coherent = (
            rule == expected
            and power_decision.get("per_candidate_one_sided_alpha") is None
            and power_decision.get("bootstrap_percentile") is None
            and power_decision.get("target_preflight_authorized") is False
        )
    else:
        expected = {
            "candidate_count": len(retained),
            "per_candidate_one_sided_alpha": (
                0.05 if len(retained) == 1 else 0.025
            ),
            "bootstrap_percentile": 95.0 if len(retained) == 1 else 97.5,
            "execution_authorized": True,
        }
        coherent = (
            rule == expected
            and power_decision.get("per_candidate_one_sided_alpha")
            == expected["per_candidate_one_sided_alpha"]
            and power_decision.get("bootstrap_percentile")
            == expected["bootstrap_percentile"]
            and power_decision.get("target_preflight_authorized") is True
        )
    if (
        not isinstance(multiplicity, dict)
        or set(multiplicity) != {"zero", "one", "two"}
        or not coherent
    ):
        raise RuntimeError(
            "panel-3 candidate/power multiplicity contract changed"
        )


def _validate_t5_size_gate_binding(
    registry: dict[str, Any],
    power_decision: dict[str, Any],
    *,
    verify_executed_source: bool = True,
) -> None:
    """Bind every selected split's T5 gate state to the power simulation."""
    panel = power_decision.get("prospective_panel")
    slots = panel.get("slots") if isinstance(panel, dict) else None
    if not isinstance(slots, list) or len(slots) != 12:
        raise RuntimeError("panel-3 power-design slot ledger changed")
    slot_by_task = {}
    for slot in slots:
        task_id = slot.get("task_id") if isinstance(slot, dict) else None
        if type(task_id) is not int or task_id in slot_by_task:
            raise RuntimeError("panel-3 power-design slot ledger changed")
        slot_by_task[task_id] = slot
    tasks = registry.get("tasks")
    selected = (
        [
            row
            for row in tasks
            if isinstance(row, dict) and row.get("status") == "selected"
        ]
        if isinstance(tasks, list)
        else []
    )
    if len(selected) != 12 or {
        int(row.get("task_id", -1)) for row in selected
    } != set(slot_by_task):
        raise RuntimeError("panel-3 selected T5 size-gate ledger changed")
    minimum = common.t5_minimum_outer_training_rows(
        registry.get("candidate_contract")
    )
    if verify_executed_source and t5.SIZE_GATE != minimum:
        raise RuntimeError("panel-3 executed T5 size gate changed")
    for row in selected:
        task_id = int(row["task_id"])
        slot = slot_by_task[task_id]
        observed = common.t5_size_gate_applicability(
            row,
            minimum_rows=minimum,
        )
        if (
            row.get("t5_size_gate_applicability") != observed
            or slot.get("t5_size_gate_applicability") != observed
            or slot.get("lineage_cluster") != row.get("lineage_cluster")
            or slot.get("stratum") != row.get("stratum")
        ):
            raise RuntimeError(
                f"panel-3 task {task_id} T5 size-gate binding changed"
            )


def _git(path: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _is_ancestor(path: Path, ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=path,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("panel-3 worker peak RSS is unavailable")
    return value


def _machine_fingerprint() -> dict[str, Any]:
    """Return a stable, nonsecret identity for resume compatibility."""
    details = creator._machine_details()
    try:
        import psutil

        physical_cpu_count = psutil.cpu_count(logical=False)
        logical_cpu_count = psutil.cpu_count(logical=True)
        memory_bytes = int(psutil.virtual_memory().total)
    except ImportError:
        physical_cpu_count = None
        logical_cpu_count = os.cpu_count()
        page_size = (
            os.sysconf("SC_PAGE_SIZE")
            if hasattr(os, "sysconf")
            else None
        )
        page_count = (
            os.sysconf("SC_PHYS_PAGES")
            if hasattr(os, "sysconf")
            else None
        )
        memory_bytes = (
            int(page_size * page_count)
            if isinstance(page_size, int)
            and isinstance(page_count, int)
            and page_size > 0
            and page_count > 0
            else None
        )
    payload = {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu_identifier": details.get("cpu_brand"),
        "physical_cpu_count": physical_cpu_count,
        "logical_cpu_count": logical_cpu_count,
        "memory_bytes": memory_bytes,
    }
    if (
        any(
            value is not None and not isinstance(value, (str, int))
            for value in payload.values()
        )
        or not all(
            isinstance(payload[field], str) and payload[field]
            for field in ("os", "os_release", "architecture")
        )
        or any(
            payload[field] is not None
            and (
                type(payload[field]) is not int
                or payload[field] <= 0
            )
            for field in (
                "physical_cpu_count",
                "logical_cpu_count",
                "memory_bytes",
            )
        )
    ):
        raise RuntimeError("panel-3 machine fingerprint is unavailable")
    return {
        **payload,
        "sha256": _json_sha256(payload),
    }


def categorical_column_indices(
    X: Any,
    declared_categorical: list[bool] | tuple[bool, ...],
) -> tuple[int, ...]:
    """Union declared categorical flags with nonnumeric loaded dtypes."""
    if getattr(X, "ndim", None) != 2:
        raise ValueError("X must be a two-dimensional feature matrix")
    if len(declared_categorical) != X.shape[1]:
        raise ValueError("categorical declaration width differs from X")
    dtypes = getattr(X, "dtypes", None)
    if dtypes is None:
        array = np.asarray(X)
        if array.dtype.kind in "biuf":
            nonnumeric = [False] * array.shape[1]
        else:
            nonnumeric = [
                np.asarray(array[:, index]).dtype.kind not in "biuf"
                for index in range(array.shape[1])
            ]
    else:
        nonnumeric = [not is_numeric_dtype(dtype) for dtype in dtypes]
    return tuple(
        index
        for index, (declared, inferred) in enumerate(
            zip(declared_categorical, nonnumeric, strict=True)
        )
        if bool(declared) or inferred
    )


def _validate_pre_h1_target_exclusion_boundary(
    registry: dict[str, Any],
    *,
    require_current_sources: bool,
) -> None:
    """Keep permanently exposed target-statistic lineages out of Panel 3."""
    exclusions = registry.get("pre_h1_target_statistic_exclusions")
    expected_keys = {
        "task_id",
        "dataset_id",
        "lineage_cluster",
        "stratum",
        "exposure_kind",
        "reason",
        "replacement_task_id",
    }
    frozen_strata = {
        "smooth_numeric",
        "mixed_categorical",
        "applied_noisy",
    }
    if not isinstance(exclusions, list) or len(exclusions) != 3:
        raise RuntimeError(
            "panel-3 pre-H1 target-statistic exclusion ledger is invalid"
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
            or exclusion["stratum"] not in frozen_strata
            or exclusion["exposure_kind"]
            != "parquet_footer_target_min_max_statistics"
            or exclusion["reason"]
            != "target_parquet_footer_min_max_observed_before_h1"
        ):
            raise RuntimeError(
                "panel-3 pre-H1 target-statistic exclusion ledger "
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
                "panel-3 pre-H1 target-statistic exclusion ledger "
                "contains duplicates"
            )
    if (
        require_current_sources
        and exclusions != common.PRE_H1_TARGET_STATISTIC_EXCLUSIONS
    ):
        raise RuntimeError(
            "panel-3 pre-H1 target-statistic exclusion ledger changed"
        )
    tasks = registry.get("tasks")
    coordinates = registry.get("coordinates")
    decision = registry.get("power_design_decision")
    panel = (
        decision.get("prospective_panel")
        if isinstance(decision, dict)
        else None
    )
    slots = panel.get("slots") if isinstance(panel, dict) else None
    if (
        not isinstance(tasks, list)
        or not isinstance(coordinates, list)
        or not isinstance(slots, list)
    ):
        raise RuntimeError(
            "panel-3 pre-H1 target-statistic boundary is incomplete"
        )
    excluded_task_ids = {row["task_id"] for row in exclusions}
    excluded_dataset_ids = {row["dataset_id"] for row in exclusions}
    excluded_lineages = {row["lineage_cluster"] for row in exclusions}
    task_ids = {
        row.get("task_id") for row in tasks if isinstance(row, dict)
    }
    dataset_ids = {
        row.get("dataset_id") for row in tasks if isinstance(row, dict)
    }
    lineages = {
        row.get("lineage_cluster") for row in tasks if isinstance(row, dict)
    }
    related_task_ids = {
        related
        for row in tasks
        if isinstance(row, dict)
        and isinstance(row.get("related_task_ids"), list)
        for related in row["related_task_ids"]
    }
    coordinate_task_ids = {
        row.get("task_id")
        for row in coordinates
        if isinstance(row, dict)
    }
    slot_task_ids = {
        row.get("task_id") for row in slots if isinstance(row, dict)
    }
    slot_lineages = {
        row.get("lineage_cluster") for row in slots if isinstance(row, dict)
    }
    if (
        excluded_task_ids
        & (task_ids | related_task_ids | coordinate_task_ids | slot_task_ids)
        or excluded_dataset_ids & dataset_ids
        or excluded_lineages & (lineages | slot_lineages)
    ):
        raise RuntimeError(
            "panel-3 pre-H1 exposed target-statistic lineage re-entered"
        )
    selected = {
        row.get("task_id"): row
        for row in tasks
        if isinstance(row, dict) and row.get("status") == "selected"
    }
    slot_by_task = {
        row.get("task_id"): row
        for row in slots
        if isinstance(row, dict)
    }
    for exclusion in exclusions:
        replacement = exclusion["replacement_task_id"]
        selected_row = selected.get(replacement)
        slot = slot_by_task.get(replacement)
        if (
            not isinstance(selected_row, dict)
            or selected_row.get("stratum") != exclusion["stratum"]
            or not isinstance(slot, dict)
            or slot.get("stratum") != exclusion["stratum"]
        ):
            raise RuntimeError(
                "panel-3 pre-H1 replacement is not selected in its "
                "frozen stratum"
            )


def validate_registry(
    registry: dict[str, Any],
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    verify_deterministic_output: bool = True,
) -> None:
    """Validate the complete prospective authorization boundary."""
    common.verify_artifact_sha256(registry, "registry_sha256")
    candidate_contract = common.load_json(common.CANDIDATE_CONTRACT)
    if (
        registry.get("schema_version") != 1
        or registry.get("selected_task_count") != 12
        or registry.get("selected_lineage_count") != 12
        or registry.get("coordinate_count") != 36
        or registry.get("outcome_blind") is not True
        or registry.get("target_statistics_used") is not False
        or registry.get("candidate_or_control_models_fitted") is not False
        or registry.get("candidate_or_control_outcomes_inspected") is not False
        or registry.get("lockbox_outcomes_used") is not False
        or registry.get("registry_freeze_complete") is not True
        or registry.get("runner_implementation_complete") is not True
        or registry.get("confirmation_run_authorized") is not True
        or registry.get("default_promotion_authorized") is not False
        or registry.get("created_from_clean_sources") is not True
    ):
        raise RuntimeError("panel-3 registry boundary is invalid")
    _validate_pre_h1_target_exclusion_boundary(
        registry,
        require_current_sources=True,
    )
    decision = registry.get("power_design_decision")
    power_design.validate_decision(
        decision,
        decision_path=common.POWER_DESIGN_DECISION,
        require_current_sources=True,
        recompute=False,
    )
    if (
        registry.get("power_design_path")
        != str(common.POWER_DESIGN_DECISION.relative_to(ROOT))
        or registry.get("power_design_file_sha256")
        != common.sha256_file(common.POWER_DESIGN_DECISION)
        or registry.get("power_design_decision_sha256")
        != decision["decision_sha256"]
        or registry.get("retained_candidates")
        != decision["retained_candidates"]
        or registry.get("power_analysis") != decision
        or registry.get("pre_h1_target_statistic_exclusions")
        != decision.get("pre_h1_target_statistic_exclusions")
        or decision["target_preflight_authorized"] is not True
    ):
        raise RuntimeError("panel-3 power-design registry binding changed")
    _validate_candidate_power_coherence(candidate_contract, decision)
    _validate_t5_size_gate_binding(registry, decision)
    _candidate_arms(registry)
    if registry.get("candidate_contract") != candidate_contract:
        raise RuntimeError("panel-3 candidate contract changed")
    _validate_runtime_contract(candidate_contract)
    coordinates = registry.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) != 36:
        raise RuntimeError("panel-3 coordinate ledger is invalid")
    identities = {
        (
            int(row["task_id"]),
            int(row["repeat"]),
            int(row["fold"]),
            int(row["sample"]),
        )
        for row in coordinates
        if isinstance(row, dict)
        and set(row) == {"task_id", "repeat", "fold", "sample"}
    }
    if len(identities) != 36:
        raise RuntimeError("panel-3 coordinate ledger repeats or is malformed")
    selected = {
        int(row["task_id"])
        for row in registry.get("tasks", [])
        if isinstance(row, dict) and row.get("status") == "selected"
    }
    if len(selected) != 12 or {row[0] for row in identities} != selected:
        raise RuntimeError("panel-3 selected task ledger changed")
    if selected != {
        int(slot["task_id"])
        for slot in decision["prospective_panel"]["slots"]
    }:
        raise RuntimeError("panel-3 selected tasks differ from power design")
    for field in ("source_sha256", "frozen_evidence_sha256"):
        bindings = registry.get(field)
        if not isinstance(bindings, dict) or not bindings:
            raise RuntimeError(f"panel-3 {field} ledger is invalid")
        for relative, digest in bindings.items():
            if (
                not isinstance(relative, str)
                or not relative
                or not _is_sha256(digest)
            ):
                raise RuntimeError(f"panel-3 {field} binding is invalid")
            path = (ROOT / relative).resolve()
            if (
                not path.is_relative_to(ROOT.resolve())
                or not path.is_file()
                or common.sha256_file(path) != digest
            ):
                raise RuntimeError(
                    f"panel-3 {field} binding changed: {relative}"
                )
    expected_source_paths = {
        str(path.relative_to(ROOT)) for path in common.PANEL3_SOURCE_PATHS
    }
    expected_evidence_paths = {
        str(path.relative_to(ROOT))
        for path in registry_builder.FROZEN_EVIDENCE
    }
    if (
        set(registry["source_sha256"]) != expected_source_paths
        or set(registry["frozen_evidence_sha256"])
        != expected_evidence_paths
        or registry.get("lockbox_darkofit_reference_allowlist")
        != sorted(registry_builder.LOCKBOX_DARKOFIT_REFERENCE_ALLOWLIST)
    ):
        raise RuntimeError("panel-3 source/evidence map changed")
    preflight_relative = registry.get("target_preflight_path")
    if (
        preflight_relative != "benchmarks/panel3_target_preflight.json"
    ):
        raise RuntimeError("panel-3 target-preflight path is invalid")
    preflight_path = (ROOT / preflight_relative).resolve()
    if (
        not preflight_path.is_relative_to(ROOT.resolve())
        or not preflight_path.is_file()
        or common.sha256_file(preflight_path)
        != registry.get("target_preflight_file_sha256")
    ):
        raise RuntimeError("panel-3 target-preflight file changed")
    preflight = common.load_json(preflight_path)
    common.verify_artifact_sha256(preflight, "target_preflight_sha256")
    registry_builder._validate_preflight(
        preflight,
        common.validate_declarations(),
    )
    if (
        preflight.get("target_preflight_sha256")
        != registry.get("target_preflight_sha256")
    ):
        raise RuntimeError("panel-3 target-preflight binding changed")
    sources = registry.get("sources")
    if (
        not isinstance(sources, dict)
        or set(sources)
        != {
            "darkofit_registry_head",
            "darkofit_model_head",
            "darkofit_prefreeze_head",
            "chimeraboost_head",
        }
        or any(
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
            for value in sources.values()
        )
        or preflight.get("sources")
        != {
            "darkofit_execution_head": sources["darkofit_model_head"],
            "darkofit_prefreeze_head": sources["darkofit_prefreeze_head"],
            "chimeraboost_head": sources["chimeraboost_head"],
        }
        or not _is_ancestor(
            ROOT,
            sources["darkofit_model_head"],
            sources["darkofit_registry_head"],
        )
        or not _is_ancestor(
            ROOT,
            sources["darkofit_prefreeze_head"],
            sources["darkofit_model_head"],
        )
    ):
        raise RuntimeError("panel-3 frozen source lineage changed")
    if not registry_path.is_file():
        raise RuntimeError("panel-3 registry file is missing")
    if verify_deterministic_output:
        registry_builder.validate_deterministic_registry_output(
            registry,
            preflight_path=preflight_path,
        )


def validate_registry_historical(
    registry: dict[str, Any],
    *,
    registry_path: Path = DEFAULT_REGISTRY,
) -> None:
    """Validate an archived registry without consulting mutable live sources."""
    common.verify_artifact_sha256(registry, "registry_sha256")
    if (
        registry.get("schema_version") != 1
        or registry.get("selected_task_count") != 12
        or registry.get("selected_lineage_count") != 12
        or registry.get("coordinate_count") != 36
        or registry.get("outcome_blind") is not True
        or registry.get("target_statistics_used") is not False
        or registry.get("candidate_or_control_models_fitted") is not False
        or registry.get("candidate_or_control_outcomes_inspected") is not False
        or registry.get("lockbox_outcomes_used") is not False
        or registry.get("registry_freeze_complete") is not True
        or registry.get("runner_implementation_complete") is not True
        or registry.get("confirmation_run_authorized") is not True
        or registry.get("default_promotion_authorized") is not False
        or registry.get("created_from_clean_sources") is not True
        or not isinstance(registry.get("candidate_contract"), dict)
        or not isinstance(registry.get("power_analysis"), dict)
    ):
        raise RuntimeError("panel-3 historical registry boundary is invalid")
    source_sha256 = registry.get("source_sha256")
    expected_source_paths = set(
        PANEL3_V1_REGISTRY_SOURCE_RELATIVE_PATHS
    )
    if (
        not isinstance(source_sha256, dict)
        or set(source_sha256) != expected_source_paths
        or any(not _is_sha256(value) for value in source_sha256.values())
    ):
        raise RuntimeError(
            "panel-3 historical source_sha256 ledger is invalid"
        )
    _validate_candidate_contract_taxonomy(
        registry["candidate_contract"],
        source_sha256=source_sha256,
    )
    frozen_evidence = registry.get("frozen_evidence_sha256")
    exposure = registry.get("exposure_catalog")
    exposure_sources = (
        exposure.get("source_files")
        if isinstance(exposure, dict)
        else None
    )
    normalized_names = (
        exposure.get("normalized_names")
        if isinstance(exposure, dict)
        else None
    )
    dataset_ids = (
        exposure.get("openml_dataset_ids")
        if isinstance(exposure, dict)
        else None
    )
    if (
        not isinstance(frozen_evidence, dict)
        or set(frozen_evidence)
        != set(PANEL3_V1_FROZEN_EVIDENCE_RELATIVE_PATHS)
        or any(not _is_sha256(value) for value in frozen_evidence.values())
        or registry.get("target_preflight_path")
        != "benchmarks/panel3_target_preflight.json"
        or registry.get("lockbox_darkofit_reference_allowlist")
        != list(PANEL3_V1_LOCKBOX_REFERENCE_ALLOWLIST)
        or not isinstance(exposure, dict)
        or set(exposure)
        != {
            "normalized_names",
            "openml_dataset_ids",
            "source_files",
            "tabarena_name_count",
            "resolved_name_count",
        }
        or not isinstance(exposure_sources, dict)
        or set(exposure_sources)
        != set(PANEL3_V1_CHIMERA_EXPOSURE_SOURCE_PATHS)
        or any(not _is_sha256(value) for value in exposure_sources.values())
        or not isinstance(normalized_names, list)
        or normalized_names != sorted(set(normalized_names))
        or any(
            not isinstance(value, str) or not value
            for value in normalized_names
        )
        or not isinstance(dataset_ids, list)
        or dataset_ids != sorted(set(dataset_ids))
        or any(type(value) is not int or value <= 0 for value in dataset_ids)
        or type(exposure.get("tabarena_name_count")) is not int
        or exposure["tabarena_name_count"] <= 0
        or type(exposure.get("resolved_name_count")) is not int
        or exposure["resolved_name_count"] < len(normalized_names)
    ):
        raise RuntimeError(
            "panel-3 historical provenance ledger changed"
        )
    _validate_pre_h1_target_exclusion_boundary(
        registry,
        require_current_sources=False,
    )
    decision = registry.get("power_design_decision")
    power_design.validate_decision(
        decision,
        require_current_sources=False,
        recompute=False,
    )
    power_relative = "benchmarks/panel3_power_design_decision.json"
    if (
        registry.get("power_design_path") != power_relative
        or not _is_sha256(registry.get("power_design_file_sha256"))
        or
        registry.get("power_design_decision_sha256")
        != decision["decision_sha256"]
        or registry.get("retained_candidates")
        != decision["retained_candidates"]
        or registry.get("power_analysis") != decision
        or registry.get("pre_h1_target_statistic_exclusions")
        != decision.get("pre_h1_target_statistic_exclusions")
        or decision["target_preflight_authorized"] is not True
        or registry["candidate_contract"]["decision"].get(
            "multiplicity_source"
        )
        != power_relative
    ):
        raise RuntimeError(
            "panel-3 historical power-design binding changed"
        )
    _validate_candidate_power_coherence(
        registry["candidate_contract"],
        decision,
    )
    _validate_t5_size_gate_binding(
        registry,
        decision,
        verify_executed_source=False,
    )
    _candidate_arms(registry)
    coordinates = registry.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) != 36:
        raise RuntimeError("panel-3 historical coordinate ledger is invalid")
    identities = {
        (
            int(row["task_id"]),
            int(row["repeat"]),
            int(row["fold"]),
            int(row["sample"]),
        )
        for row in coordinates
        if isinstance(row, dict)
        and set(row) == {"task_id", "repeat", "fold", "sample"}
        and all(type(value) is int and value >= 0 for value in row.values())
    }
    tasks = registry.get("tasks")
    selected = {
        int(row["task_id"])
        for row in tasks
        if isinstance(row, dict)
        and type(row.get("task_id")) is int
        and row.get("status") == "selected"
    } if isinstance(tasks, list) else set()
    if (
        len(identities) != 36
        or len(selected) != 12
        or {row[0] for row in identities} != selected
        or selected
        != {
            int(slot["task_id"])
            for slot in decision["prospective_panel"]["slots"]
        }
    ):
        raise RuntimeError("panel-3 historical selected-task ledger changed")
    for field in ("source_sha256", "frozen_evidence_sha256"):
        bindings = registry.get(field)
        if not isinstance(bindings, dict) or not bindings:
            raise RuntimeError(f"panel-3 historical {field} ledger is invalid")
        for relative, digest in bindings.items():
            candidate = Path(relative)
            if (
                not isinstance(relative, str)
                or not relative
                or candidate.is_absolute()
                or ".." in candidate.parts
                or not _is_sha256(digest)
            ):
                raise RuntimeError(
                    f"panel-3 historical {field} binding is invalid"
                )
    sources = registry.get("sources")
    if (
        not isinstance(sources, dict)
        or set(sources)
        != {
            "darkofit_registry_head",
            "darkofit_model_head",
            "darkofit_prefreeze_head",
            "chimeraboost_head",
        }
        or any(
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
            for value in sources.values()
        )
        or not isinstance(registry.get("target_preflight_path"), str)
        or not _is_sha256(registry.get("target_preflight_file_sha256"))
        or not _is_sha256(registry.get("target_preflight_sha256"))
    ):
        raise RuntimeError("panel-3 historical source ledger is invalid")


def _source_state(registry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    darko = creator.git_state(ROOT)
    chimera = creator.git_state(CHIMERA_ROOT)
    if not darko["clean"] or not chimera["clean"]:
        raise RuntimeError("panel-3 execution requires clean source trees")
    frozen = registry["sources"]
    if not _is_ancestor(
        ROOT, str(frozen["darkofit_registry_head"]), darko["head"]
    ):
        raise RuntimeError("panel-3 DarkoFit source left the frozen lineage")
    changed_since_registry_freeze = {
        value
        for value in _git(
            ROOT,
            "diff",
            "--name-only",
            f"{frozen['darkofit_registry_head']}..{darko['head']}",
        ).splitlines()
        if value
    }
    if not changed_since_registry_freeze <= {
        "benchmarks/panel3_registry.json"
    }:
        raise RuntimeError(
            "panel-3 tracked source changed after the registry freeze: "
            f"{sorted(changed_since_registry_freeze)}"
        )
    if chimera["head"] != str(frozen["chimeraboost_head"]):
        raise RuntimeError("panel-3 ChimeraBoost source changed")
    return darko, chimera


def _selected_rows(registry: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = {
        int(row["task_id"]): row
        for row in registry["tasks"]
        if row["status"] == "selected"
    }
    if len(rows) != 12:
        raise RuntimeError("panel-3 selected task ledger changed")
    return rows


def _load_task(task_id: int, row: dict[str, Any]):
    import openml

    task = openml.tasks.get_task(task_id, download_splits=True)
    dataset = task.get_dataset()
    X, y, categorical, names = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
        include_row_id=False,
        include_ignore_attribute=False,
    )
    record = row["task_record"]
    if (
        int(dataset.dataset_id) != int(row["dataset_id"])
        or str(dataset.name) != str(row["dataset_name"])
        or str(task.target_name) != str(row["target_name"])
        or list(X.columns) != list(names)
        or len(X) != int(record["fingerprint"]["n_rows"])
        or X.shape[1] != int(record["fingerprint"]["n_features"])
        or str(dataset.md5_checksum) != str(record["openml_declared_md5"])
    ):
        raise RuntimeError(f"panel-3 task {task_id} metadata changed")
    observed = fingerprints.dataset_fingerprint(X, y)
    if observed != record["fingerprint"]:
        raise RuntimeError(f"panel-3 task {task_id} data fingerprint changed")
    y = pd.to_numeric(y, errors="raise").astype(np.float64)
    if not np.isfinite(y.to_numpy(dtype=np.float64)).all():
        raise RuntimeError(f"panel-3 task {task_id} target is nonfinite")
    X, categorical, feature_policy = _apply_feature_policy(
        X,
        list(categorical),
        row.get("feature_policy", {"kind": "none"}),
    )
    expected_feature_policy = row.get("feature_policy_attestation")
    if (
        not isinstance(expected_feature_policy, dict)
        or feature_policy != expected_feature_policy
    ):
        raise RuntimeError(
            f"panel-3 task {task_id} feature-policy attestation changed"
        )
    categorical_indices = categorical_column_indices(X, categorical)
    categorical_names = [
        str(X.columns[index]) for index in categorical_indices
    ]
    if categorical_names != row.get("resolved_categorical_columns"):
        raise RuntimeError(
            f"panel-3 task {task_id} categorical resolution changed"
        )
    return (
        task,
        X,
        y,
        list(categorical_indices),
        categorical_names,
        feature_policy,
    )


def _apply_feature_policy(
    X: pd.DataFrame,
    categorical: list[bool],
    policy: Any,
) -> tuple[pd.DataFrame, list[bool], dict[str, Any]]:
    """Apply the shared target-free contract and remap category flags."""
    source_columns = list(X.columns)
    result, metadata = data_contract.apply_feature_policy(X, policy)
    retained_categorical = data_contract.categorical_flags_after_policy(
        source_columns,
        categorical,
        metadata,
    )
    return result, retained_categorical, metadata


def _coordinate_key(coordinate: dict[str, int]) -> tuple[int, int, int, int]:
    return (
        int(coordinate["task_id"]),
        int(coordinate["repeat"]),
        int(coordinate["fold"]),
        int(coordinate["sample"]),
    )


def _expected_split(
    row: dict[str, Any],
    coordinate: dict[str, int],
) -> dict[str, Any]:
    policy = row.get("split_policy", {"kind": "openml_official"})
    if not isinstance(policy, dict):
        raise RuntimeError("panel-3 split policy is invalid")
    kind = policy.get("kind")
    if kind == "openml_official":
        if set(policy) != {"kind"}:
            raise RuntimeError("panel-3 official split policy changed")
        source = row["task_record"]["official_splits"]["coordinates"]
    elif kind == "frozen_explicit":
        if set(policy) not in ({
            "kind",
            "coordinates",
            "allow_unused_rows",
        }, {
            "kind",
            "coordinates",
            "allow_unused_rows",
            "construction",
        }):
            raise RuntimeError("panel-3 explicit split policy changed")
        if type(policy["allow_unused_rows"]) is not bool:
            raise RuntimeError("panel-3 explicit split omission flag is invalid")
        if "construction" in policy and not isinstance(
            policy["construction"], dict
        ):
            raise RuntimeError(
                "panel-3 explicit split construction is invalid"
            )
        source = policy["coordinates"]
        if not isinstance(source, list):
            raise RuntimeError("panel-3 explicit split ledger is invalid")
    else:
        raise RuntimeError("panel-3 split policy kind is unsupported")
    matches = [
        split
        for split in source
        if (
            int(split["repeat"]) == int(coordinate["repeat"])
            and int(split["fold"]) == int(coordinate["fold"])
            and int(split["sample"]) == int(coordinate["sample"])
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"panel-3 coordinate {_coordinate_key(coordinate)} is not frozen"
        )
    return matches[0]


def _resolve_split(
    task: Any,
    row: dict[str, Any],
    coordinate: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    expected = _expected_split(row, coordinate)
    policy = row.get("split_policy", {"kind": "openml_official"})
    if policy["kind"] == "openml_official":
        train, test = task.get_train_test_split_indices(
            repeat=int(coordinate["repeat"]),
            fold=int(coordinate["fold"]),
            sample=int(coordinate["sample"]),
        )
    else:
        required = {
            "repeat",
            "fold",
            "sample",
            "train_indices",
            "test_indices",
            "train_size",
            "test_size",
            "train_index_sha256",
            "test_index_sha256",
        }
        if not isinstance(expected, dict) or set(expected) != required:
            raise RuntimeError(
                "panel-3 explicit split coordinate is malformed"
            )
        for label in ("train_indices", "test_indices"):
            values = expected[label]
            if (
                not isinstance(values, list)
                or not values
                or any(type(value) is not int for value in values)
            ):
                raise RuntimeError(
                    f"panel-3 explicit {label} are invalid"
                )
        train = np.asarray(expected["train_indices"], dtype=np.int64)
        test = np.asarray(expected["test_indices"], dtype=np.int64)
    train = np.asarray(train, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    n_rows = int(row["task_record"]["fingerprint"]["n_rows"])
    if (
        train.ndim != 1
        or test.ndim != 1
        or train.size == 0
        or test.size == 0
        or np.any(train < 0)
        or np.any(test < 0)
        or np.any(train >= n_rows)
        or np.any(test >= n_rows)
        or len(np.unique(train)) != len(train)
        or len(np.unique(test)) != len(test)
        or np.intersect1d(train, test).size
    ):
        raise RuntimeError("panel-3 split indices are invalid")
    if (
        policy["kind"] == "frozen_explicit"
        and not policy["allow_unused_rows"]
        and len(train) + len(test) != n_rows
    ):
        raise RuntimeError("panel-3 explicit split unexpectedly omits rows")
    observed = {
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "train_index_sha256": _array_sha256(train, dtype="<i8"),
        "test_index_sha256": _array_sha256(test, dtype="<i8"),
    }
    for key, value in observed.items():
        if expected.get(key) != value:
            raise RuntimeError(
                f"panel-3 coordinate {_coordinate_key(coordinate)} "
                f"{key} changed"
            )
    return train, test, {
        **observed,
        "kind": policy["kind"],
        "allow_unused_rows": bool(
            policy.get("allow_unused_rows", False)
        ),
        "construction_sha256": (
            None
            if "construction" not in policy
            else _json_sha256(policy["construction"])
        ),
    }


def _take(frame: Any, indices: np.ndarray):
    positions = np.asarray(indices, dtype=np.int64)
    if hasattr(frame, "iloc"):
        return frame.iloc[positions]
    return np.asarray(frame)[positions]


def _selection_split(n_rows: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train, validation = next(
        ShuffleSplit(
            n_splits=1,
            test_size=VALIDATION_FRACTION,
            random_state=RANDOM_STATE,
        ).split(np.arange(n_rows))
    )
    return train, validation, {
        "policy": "ShuffleSplit",
        "random_state": RANDOM_STATE,
        "validation_fraction": VALIDATION_FRACTION,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_positions_sha256": _array_sha256(train, dtype="<i8"),
        "validation_positions_sha256": _array_sha256(
            validation, dtype="<i8"
        ),
    }


def _fit(
    model: Any,
    X: Any,
    y: Any,
    categorical_indices: list[int],
    *,
    eval_set: tuple[Any, Any] | None = None,
) -> tuple[Any, float]:
    started = time.perf_counter_ns()
    model.fit(
        X,
        y,
        cat_features=categorical_indices or None,
        eval_set=eval_set,
    )
    return model, float((time.perf_counter_ns() - started) / 1e9)


def _guarded_model(
    *,
    linear_leaves: bool,
    iterations: int = 2_000,
    selection: bool,
):
    from darkofit import DarkoRegressor

    return DarkoRegressor(
        iterations=int(iterations),
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=1.0,
        max_bins=128,
        min_child_weight=1.0,
        tree_mode="catboost",
        selection_rounds=None,
        linear_leaves=bool(linear_leaves),
        early_stopping=bool(selection),
        use_best_model=bool(selection),
        refit=False,
        random_state=RANDOM_STATE,
        thread_count=THREADS_PER_WORKER,
        diagnostic_warnings="never",
    )


def _selection_record(name: str, model: Any, seconds: float) -> dict[str, Any]:
    score = float(model.best_score_)
    if not math.isfinite(score) or score <= 0.0:
        raise RuntimeError(f"panel-3 selection score is invalid for {name}")
    validation = dict(model.model_.auto_params_.get("validation_split", {}))
    if validation.get("source") != "explicit_eval_set":
        raise RuntimeError(
            f"panel-3 selection fit {name} missed explicit eval set"
        )
    metadata = basketball.extract_fit_metadata(model)
    if metadata["final_fit"]["stop_reason"] not in {
        "early_stopping",
        "iteration_limit",
        "no_split",
    }:
        raise RuntimeError(
            f"panel-3 selection fit {name} stopped unexpectedly"
        )
    return {
        "name": name,
        "validation_rmse": score,
        "fit_seconds": float(seconds),
        "fit_metadata": metadata,
        "validation": validation,
    }


def _augment_crosses(
    X: pd.DataFrame,
    pairs: list[tuple[int, int, str]],
) -> pd.DataFrame:
    if not pairs:
        return X.copy()
    result = X.copy()
    for left, right, operation in pairs:
        left_values = pd.to_numeric(
            X.iloc[:, left], errors="raise"
        ).to_numpy(dtype=np.float64)
        right_values = pd.to_numeric(
            X.iloc[:, right], errors="raise"
        ).to_numpy(dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            values = (
                left_values - right_values
                if operation == "diff"
                else left_values * right_values
            )
        values = np.asarray(values, dtype=np.float64)
        values[~np.isfinite(values)] = np.nan
        name = f"__darkofit_cross_{left}_{right}_{operation}"
        if name in result.columns:
            raise RuntimeError(f"panel-3 cross column collision: {name}")
        result[name] = values
    return result


def _timed_predict(
    predict: Callable[[], np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    prediction = np.asarray(predict(), dtype=np.float64)
    if prediction.ndim != 1 or not np.isfinite(prediction).all():
        raise RuntimeError("panel-3 prediction is invalid")
    durations: list[float] = []
    total = 0.0
    last = None
    while (
        len(durations) < PREDICTION_MIN_CALLS
        or total < PREDICTION_BLOCK_SECONDS
    ):
        if len(durations) >= PREDICTION_MAX_CALLS:
            raise RuntimeError(
                "panel-3 prediction timing did not reach block length"
            )
        started = time.perf_counter_ns()
        last = np.asarray(predict(), dtype=np.float64)
        elapsed = (time.perf_counter_ns() - started) / 1e9
        durations.append(float(elapsed))
        total += elapsed
    if not np.array_equal(prediction, last):
        raise RuntimeError("panel-3 repeated prediction changed")
    return prediction, {
        "per_call_median_seconds": float(np.median(durations)),
        "per_call_min_seconds": float(np.min(durations)),
        "per_call_max_seconds": float(np.max(durations)),
        "total_seconds": float(total),
        "call_count": len(durations),
        "minimum_block_seconds": PREDICTION_BLOCK_SECONDS,
    }


def _fit_guarded_cross(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_indices: list[int],
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, float, dict[str, Any], dict[str, Any]]:
    policy_started = time.perf_counter_ns()
    inner_train, validation, split = _selection_split(len(X_train))
    X_select = _take(X_train, inner_train)
    y_select = _take(y_train, inner_train)
    X_validation = _take(X_train, validation)
    y_validation = _take(y_train, validation)
    eval_set = (X_validation, y_validation)

    base_candidates = []
    records = []
    for linear in (False, True):
        model, seconds = _fit(
            _guarded_model(linear_leaves=linear, selection=True),
            X_select,
            y_select,
            categorical_indices,
            eval_set=eval_set,
        )
        record = _selection_record(
            "uncrossed_linear" if linear else "uncrossed_constant",
            model,
            seconds,
        )
        records.append(record)
        base_candidates.append((model, record))
    base, base_record = min(
        base_candidates,
        key=lambda item: (
            item[1]["validation_rmse"],
            bool(item[0].linear_leaves),
        ),
    )
    selected_linear = bool(base.linear_leaves)
    pairs = candidate_pairs(
        np.asarray(base.feature_importances_, dtype=np.float64),
        categorical_indices,
        X_train.shape[1],
    )
    crossed = None
    crossed_record = None
    cross_transform_seconds = 0.0
    if pairs:
        started = time.perf_counter_ns()
        X_select_cross = _augment_crosses(X_select, pairs)
        X_validation_cross = _augment_crosses(X_validation, pairs)
        cross_transform_seconds = (
            time.perf_counter_ns() - started
        ) / 1e9
        crossed, seconds = _fit(
            _guarded_model(
                linear_leaves=selected_linear,
                selection=True,
            ),
            X_select_cross,
            y_select,
            categorical_indices,
            eval_set=(X_validation_cross, y_validation),
        )
        crossed_record = _selection_record(
            "crossed_selected_leaf_lane",
            crossed,
            seconds + cross_transform_seconds,
        )
        crossed_record["transform_seconds"] = float(
            cross_transform_seconds
        )
        crossed_record["pairs"] = [list(pair) for pair in pairs]
        records.append(crossed_record)

    crossed_score = (
        None
        if crossed_record is None
        else float(crossed_record["validation_rmse"])
    )
    selected_crosses = (
        crossed_score is not None
        and crossed_score
        <= GUARDED_CROSS_RATIO * float(base_record["validation_rmse"])
    )
    selected = crossed if selected_crosses else base
    selected_record = crossed_record if selected_crosses else base_record
    selected_pairs = pairs if selected_crosses else []
    best_rounds = int(selected.best_n_estimators_)

    started = time.perf_counter_ns()
    X_final = (
        _augment_crosses(X_train, selected_pairs)
        if selected_pairs
        else X_train
    )
    final_transform_seconds = (time.perf_counter_ns() - started) / 1e9
    final, final_model_seconds = _fit(
        _guarded_model(
            linear_leaves=selected_linear,
            iterations=best_rounds,
            selection=False,
        ),
        X_final,
        y_train,
        categorical_indices,
    )
    final_fit_seconds = final_transform_seconds + final_model_seconds

    def predict() -> np.ndarray:
        frame = (
            _augment_crosses(X_test, selected_pairs)
            if selected_pairs
            else X_test
        )
        return final.predict(frame)

    selection_seconds = float(
        sum(float(record["fit_seconds"]) for record in records)
    )
    policy_fit_seconds = (
        time.perf_counter_ns() - policy_started
    ) / 1e9
    policy_overhead_seconds = (
        policy_fit_seconds - selection_seconds - final_fit_seconds
    )
    if policy_overhead_seconds < -max(
        1e-12,
        1e-9 * policy_fit_seconds,
    ):
        raise RuntimeError(
            "panel-3 guarded-cross fit-time components are inconsistent"
        )
    policy_overhead_seconds = max(0.0, policy_overhead_seconds)
    policy_fit_seconds = (
        selection_seconds
        + final_fit_seconds
        + policy_overhead_seconds
    )
    prediction, timing = _timed_predict(predict)
    metadata = {
        "kind": "guarded_cross_features_policy",
        "engaged": bool(selected_crosses),
        "decline_reason": None if selected_crosses else "cross_guard",
        "split": split,
        "cross_guard_ratio": GUARDED_CROSS_RATIO,
        "selected_configuration": (
            "crossed" if selected_crosses else "uncrossed"
        ),
        "selected_linear_leaves": selected_linear,
        "selected_crosses": bool(selected_crosses),
        "candidate_cross_pairs": [list(pair) for pair in pairs],
        "selected_cross_pairs": [
            list(pair) for pair in selected_pairs
        ],
        "selected_cross_pair_count": len(selected_pairs),
        "uncrossed_validation_rmse": float(
            base_record["validation_rmse"]
        ),
        "crossed_validation_rmse": crossed_score,
        "relative_crossed_validation_ratio": (
            None
            if crossed_score is None
            else float(crossed_score / base_record["validation_rmse"])
        ),
        "selected_best_iteration": best_rounds,
        "selected_resolved_learning_rate": float(
            selected.learning_rate_
        ),
        "selected_selection_fit": selected_record,
        "selection_fits": records,
        "total_selection_fit_seconds": selection_seconds,
        "policy_overhead_seconds": float(policy_overhead_seconds),
        "final_transform_seconds": float(final_transform_seconds),
        "final_model_fit_seconds": float(final_model_seconds),
        "final_fit_seconds": float(final_fit_seconds),
        "final_refit_parameters": {
            "iterations": best_rounds,
            "learning_rate": 0.1,
            "tree_mode": "catboost",
            "linear_leaves": selected_linear,
            "crossed": bool(selected_crosses),
        },
        "final_fit": basketball.extract_fit_metadata(final),
    }
    return (
        prediction,
        policy_fit_seconds,
        timing,
        metadata,
    )


def _ordinal_features(row: dict[str, Any]) -> dict[str, list[Any]]:
    if "ordinal_features" not in row:
        raise RuntimeError("panel-3 ordinal declaration is missing")
    value = row["ordinal_features"]
    if not isinstance(value, dict) or any(
        not isinstance(key, str)
        or not isinstance(levels, list)
        or len(levels) < 2
        for key, levels in value.items()
    ):
        raise RuntimeError("panel-3 ordinal declaration is invalid")
    return value


def _fit_chimera(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_indices: list[int],
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, float, dict[str, Any], dict[str, Any]]:
    if str(CHIMERA_ROOT) not in sys.path:
        sys.path.insert(0, str(CHIMERA_ROOT))
    from chimeraboost import ChimeraBoostRegressor

    model = ChimeraBoostRegressor(
        random_state=RANDOM_STATE,
        thread_count=THREADS_PER_WORKER,
    )
    started = time.perf_counter_ns()
    model.fit(
        X_train,
        y_train,
        cat_features=categorical_indices or None,
    )
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    prediction, timing = _timed_predict(lambda: model.predict(X_test))
    fitted = int(len(model.model_.trees_))
    requested = int(model.n_estimators)
    history = list(model.validation_history_)
    attempted = len(history)
    if attempted > fitted:
        stop_reason = "early_stopping"
    elif fitted >= requested:
        stop_reason = "iteration_limit"
    elif fitted == 0:
        stop_reason = "no_legal_split"
    else:
        stop_reason = "no_legal_split_or_internal_selection"
    linear = bool(model.linear_leaves_selected_)
    crossed = bool(model.cross_features_selected_)
    lane = (
        "linear_leaves_with_crosses"
        if linear and crossed
        else "linear_leaves"
        if linear
        else "constant_leaves_with_crosses"
        if crossed
        else "constant_leaves"
    )
    metadata = {
        "kind": "chimeraboost_0_15_0",
        "requested_iterations": requested,
        "attempted_iterations": attempted,
        "best_iteration": int(model.best_iteration_),
        "fitted_tree_count": fitted,
        "resolved_learning_rate": float(model.model_.lr_),
        "selected_mode": "symmetric_oblivious",
        "selected_lane": lane,
        "stop_reason": stop_reason,
        "early_stopping": bool(model.early_stopping),
        "selection_rounds": (
            None
            if model.selection_rounds is None
            else int(model.selection_rounds)
        ),
        "linear_leaves_selected": linear,
        "cross_features_selected": crossed,
        "cross_pairs": [
            list(pair) for pair in (model.cross_pairs_ or ())
        ],
    }
    return prediction, float(fit_seconds), timing, metadata


def _catboost_frame(
    X: pd.DataFrame,
    categorical_indices: list[int],
) -> pd.DataFrame:
    result = X.copy()
    for index in categorical_indices:
        column = result.columns[index]
        values = result.iloc[:, index].astype(object)
        result[column] = values.map(
            lambda value: (
                "__DARKOFIT_MISSING_CATEGORY__"
                if pd.isna(value)
                else f"{type(value).__name__}:{value}"
            )
        )
    return result


def _fit_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_indices: list[int],
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, float, dict[str, Any], dict[str, Any]]:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        random_seed=RANDOM_STATE,
        thread_count=THREADS_PER_WORKER,
        verbose=False,
        allow_writing_files=False,
    )
    started = time.perf_counter_ns()
    train = _catboost_frame(X_train, categorical_indices)
    model.fit(
        train,
        y_train,
        cat_features=categorical_indices or None,
    )
    fit_seconds = (time.perf_counter_ns() - started) / 1e9

    def predict() -> np.ndarray:
        # This conversion is part of the product-level prediction path and is
        # intentionally repeated inside every timed call.
        return model.predict(_catboost_frame(X_test, categorical_indices))

    prediction, timing = _timed_predict(predict)
    parameters = dict(model.get_all_params())
    requested = int(parameters["iterations"])
    fitted = int(model.tree_count_)
    raw_best = model.get_best_iteration()
    best = -1 if raw_best is None else int(raw_best)
    if fitted >= requested:
        stop_reason = "iteration_limit"
    elif best >= 0:
        stop_reason = "early_stopping"
    else:
        stop_reason = "no_legal_split_or_other"
    metadata = {
        "kind": "catboost_product_default",
        "requested_iterations": requested,
        "attempted_iterations": fitted,
        "best_iteration": best,
        "fitted_tree_count": fitted,
        "resolved_learning_rate": float(parameters["learning_rate"]),
        "selected_mode": str(parameters["grow_policy"]),
        "selected_lane": str(parameters["boosting_type"]),
        "stop_reason": stop_reason,
        "external_categorical_transform_included_in_fit_timing": True,
        "external_categorical_transform_included_in_predict_timing": True,
    }
    return prediction, float(fit_seconds), timing, metadata


def _warmup(arm: str) -> float:
    mapping = {
        CONTROL_ARM: t5.CONTROL,
        "t5_composite_policy": t5.COMPOSITE,
        "guarded_cross_features_policy": t5.COMPOSITE,
        "chimeraboost_0_15_0": t5.CHIMERA,
        "catboost_product_default": t5.CATBOOST,
    }
    if arm == "catboost_product_default":
        try:
            installed = importlib.metadata.version("catboost")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "panel-3 requires CatBoost 1.2.10"
            ) from exc
        if installed != "1.2.10":
            raise RuntimeError(
                "panel-3 CatBoost version changed: "
                f"{installed} != 1.2.10"
            )
    seconds = t5._warmup(mapping[arm])
    if arm == "chimeraboost_0_15_0":
        module = sys.modules.get("chimeraboost")
        source = getattr(module, "__file__", None)
        if (
            not isinstance(source, str)
            or not Path(source).resolve().is_relative_to(
                CHIMERA_ROOT.resolve()
            )
            or getattr(module, "__version__", None) != "0.15.0"
        ):
            raise RuntimeError(
                "panel-3 imported the wrong ChimeraBoost build"
            )
    return seconds


def _evaluate_coordinate(
    task: Any,
    row: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    categorical_indices: list[int],
    coordinate: dict[str, int],
    arm: str,
) -> dict[str, Any]:
    train, test, split = _resolve_split(task, row, coordinate)
    X_train = _take(X, train)
    y_train = _take(y, train)
    X_test = _take(X, test)

    if arm == CONTROL_ARM:
        prediction, fit_seconds, timing, metadata = t5._fit_control(
            X_train,
            y_train,
            categorical_indices,
            X_test,
        )
        metadata["kind"] = CONTROL_ARM
    elif arm == "t5_composite_policy":
        prediction, fit_seconds, timing, metadata = t5._fit_composite(
            X_train,
            y_train,
            categorical_indices,
            X_test,
            _ordinal_features(row),
        )
        metadata["kind"] = arm
    elif arm == "guarded_cross_features_policy":
        prediction, fit_seconds, timing, metadata = _fit_guarded_cross(
            X_train,
            y_train,
            categorical_indices,
            X_test,
        )
    elif arm == "chimeraboost_0_15_0":
        prediction, fit_seconds, timing, metadata = _fit_chimera(
            X_train,
            y_train,
            categorical_indices,
            X_test,
        )
    elif arm == "catboost_product_default":
        prediction, fit_seconds, timing, metadata = _fit_catboost(
            X_train,
            y_train,
            categorical_indices,
            X_test,
        )
    else:
        raise ValueError(f"unknown panel-3 arm: {arm}")

    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != (len(test),) or not np.isfinite(prediction).all():
        raise RuntimeError("panel-3 prediction shape or values changed")
    target = y.iloc[np.asarray(test, dtype=np.int64)].to_numpy(
        dtype=np.float64
    )
    rmse = float(mean_squared_error(target, prediction) ** 0.5)
    if not math.isfinite(rmse) or rmse <= 0.0:
        raise RuntimeError("panel-3 RMSE is invalid")
    return {
        "train_rows": split["train_size"],
        "test_rows": split["test_size"],
        "train_index_sha256": split["train_index_sha256"],
        "test_index_sha256": split["test_index_sha256"],
        "split_policy": {
            "kind": split["kind"],
            "allow_unused_rows": split["allow_unused_rows"],
            "construction_sha256": split["construction_sha256"],
        },
        "target_sha256": _array_sha256(target),
        "rmse": rmse,
        "fit_seconds": float(fit_seconds),
        "prediction_timing": timing,
        "prediction_sha256": _array_sha256(prediction),
        "metadata": metadata,
    }


def run_worker(
    registry: dict[str, Any],
    registry_path: Path,
    coordinate: dict[str, int],
    arm: str,
) -> dict[str, Any]:
    try:
        source_before = _campaign_source_attestation(
            registry,
            registry_path,
        )
    except Exception as exc:
        _invalidate_worker_campaign(str(exc))
        raise
    validate_registry(
        registry,
        registry_path=registry_path,
        verify_deterministic_output=False,
    )
    rows = _selected_rows(registry)
    task_id = int(coordinate["task_id"])
    if task_id not in rows:
        raise RuntimeError("panel-3 worker task is outside the registry")
    if arm not in _arm_order(registry):
        raise RuntimeError("panel-3 worker arm is invalid")
    if _coordinate_key(coordinate) not in {
        _coordinate_key(item) for item in registry["coordinates"]
    }:
        raise RuntimeError("panel-3 worker coordinate is outside the registry")
    row = rows[task_id]
    (
        task,
        X,
        y,
        categorical_indices,
        categorical_names,
        feature_policy,
    ) = _load_task(task_id, row)
    warmup_seconds = _warmup(arm)
    started = time.perf_counter_ns()
    result = _evaluate_coordinate(
        task,
        row,
        X,
        y,
        categorical_indices,
        coordinate,
        arm,
    )
    wall_seconds = (time.perf_counter_ns() - started) / 1e9
    try:
        source_after = _campaign_source_attestation(
            registry,
            registry_path,
        )
    except Exception as exc:
        _invalidate_worker_campaign(str(exc))
        raise
    if source_after != source_before:
        message = "panel-3 source or registry changed during worker fit"
        _invalidate_worker_campaign(message)
        raise RuntimeError(message)
    source_attestation = {
        "before": source_before,
        "after": source_after,
    }
    behavior = {
        "coordinate": coordinate,
        "arm": arm,
        "rmse": result["rmse"],
        "prediction_sha256": result["prediction_sha256"],
        "metadata": result["metadata"],
        "source_attestation": source_attestation,
    }
    gc.collect()
    return {
        "worker_key": _worker_key(coordinate, arm),
        "task_id": task_id,
        "dataset_id": int(row["dataset_id"]),
        "dataset_name": row["dataset_name"],
        "lineage_cluster": row["lineage_cluster"],
        "stratum": row["stratum"],
        "coordinate": {
            key: int(coordinate[key])
            for key in ("repeat", "fold", "sample")
        },
        "arm": arm,
        "categorical_feature_indices": categorical_indices,
        "categorical_feature_names": categorical_names,
        "ordinal_features": _ordinal_features(row),
        "feature_policy": feature_policy,
        **result,
        "source_attestation": source_attestation,
        "warmup_seconds": float(warmup_seconds),
        "wall_seconds": float(wall_seconds),
        "peak_rss_bytes": _peak_rss_bytes(),
        "behavior_fingerprint_sha256": _json_sha256(behavior),
    }


def _worker_key(coordinate: dict[str, int], arm: str) -> str:
    return (
        f"{int(coordinate['task_id'])}-r{int(coordinate['repeat'])}"
        f"-f{int(coordinate['fold'])}-s{int(coordinate['sample'])}-{arm}"
    )


def execution_plan(
    registry: dict[str, Any],
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    registry_file_sha256: str | None = None,
    validate_registry_boundary: bool = True,
) -> dict[str, Any]:
    if validate_registry_boundary:
        validate_registry(registry, registry_path=registry_path)
    if registry_file_sha256 is None:
        registry_file_sha256 = common.sha256_file(registry_path)
    if not _is_sha256(registry_file_sha256):
        raise RuntimeError("panel-3 registry file hash is invalid")
    arm_order = _arm_order(registry)
    candidate_arms = _candidate_arms(registry)
    decision_arms = _decision_arms(registry)
    workers = [
        {
            **coordinate,
            "arm": arm,
            "worker_key": _worker_key(coordinate, arm),
        }
        for arm in arm_order
        for coordinate in registry["coordinates"]
    ]
    if len({row["worker_key"] for row in workers}) != len(workers):
        raise RuntimeError("panel-3 execution plan repeats a worker")
    return common.bind_artifact_sha256(
        {
            "schema_version": 1,
            "name": "darkofit_panel3_execution_plan_v1",
            "registry_sha256": registry["registry_sha256"],
            "registry_file_sha256": registry_file_sha256,
            "arm_order": list(arm_order),
            "candidate_arms": list(candidate_arms),
            "decision_arms": list(decision_arms),
            "comparator_arms": list(COMPARATOR_ARMS),
            "comparators_affect_candidate_gates": False,
            "comparator_failure_policy": (
                "persist_immutable_failure_record_and_continue"
            ),
            "decision_worker_count": (
                len(registry["coordinates"]) * len(decision_arms)
            ),
            "comparator_worker_count": (
                len(registry["coordinates"]) * len(COMPARATOR_ARMS)
            ),
            "both_pass_default": "t5_composite_policy",
            "coordinate_count": len(registry["coordinates"]),
            "worker_count": len(workers),
            "workers": workers,
            "runner_implementation_complete": True,
            "model_fits_authorized_by_this_plan": True,
            "candidate_or_control_models_fitted": False,
            "candidate_or_control_outcomes_inspected": False,
        },
        "execution_plan_sha256",
    )


def _worker_environment() -> dict[str, str]:
    environment = basketball.worker_environment(THREADS_PER_WORKER)
    environment.update(
        {
            "DARKOFIT_WARMUP": "0",
            "CHIMERABOOST_WARMUP": "0",
            "OPENML_CACHE_DIRECTORY": str(Path.home() / ".cache" / "openml"),
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": os.pathsep.join(
                [
                    str(ROOT),
                    str(CHIMERA_ROOT),
                    environment.get("PYTHONPATH", ""),
                ]
            ),
        }
    )
    return environment


def _worker_command(
    registry_path: Path,
    coordinate: dict[str, int],
    arm: str,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--registry",
        str(registry_path),
        "--worker-task",
        str(int(coordinate["task_id"])),
        "--worker-repeat",
        str(int(coordinate["repeat"])),
        "--worker-fold",
        str(int(coordinate["fold"])),
        "--worker-sample",
        str(int(coordinate["sample"])),
        "--worker-arm",
        arm,
    ]


def _spool_binding(
    registry: dict[str, Any],
    registry_path: Path,
    registry_file_sha256: str,
    darko_source: dict[str, Any],
    chimera_source: dict[str, Any],
    runtime_contract_normalized_sha256: str,
    machine_fingerprint_sha256: str,
) -> dict[str, Any]:
    if (
        registry_path.expanduser().absolute() != DEFAULT_REGISTRY
        or not _is_sha256(registry_file_sha256)
        or not _is_sha256(runtime_contract_normalized_sha256)
        or not _is_sha256(machine_fingerprint_sha256)
    ):
        raise RuntimeError("panel-3 registry file binding is invalid")
    return {
        "schema_version": 1,
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "analyzer_sha256": _sha256(common.ANALYZER),
        "protocol_sha256": _sha256(common.PROTOCOL),
        "candidate_contract_sha256": _sha256(common.CANDIDATE_CONTRACT),
        "power_design_decision_sha256": registry[
            "power_design_decision_sha256"
        ],
        "registry_file_sha256": registry_file_sha256,
        "registry_canonical_sha256": registry["registry_sha256"],
        "runtime_contract_normalized_sha256": (
            runtime_contract_normalized_sha256
        ),
        "machine_fingerprint_sha256": machine_fingerprint_sha256,
        "darkofit_head": darko_source["head"],
        "chimeraboost_head": chimera_source["head"],
        "arms": list(_arm_order(registry)),
        "coordinate_count": len(registry["coordinates"]),
    }


def _spool_path(
    spool_directory: Path,
    coordinate: dict[str, int],
    arm: str,
) -> Path:
    return spool_directory / f"{_worker_key(coordinate, arm)}.json"


def _attempt_path(
    spool_directory: Path,
    coordinate: dict[str, int],
    arm: str,
) -> Path:
    return spool_directory / f"{_worker_key(coordinate, arm)}.attempt.json"


def _campaign_invalidation_path(spool_directory: Path) -> Path:
    return spool_directory / CAMPAIGN_INVALIDATION_FILENAME


def _attempt_payload(
    binding: dict[str, Any],
    coordinate: dict[str, int],
    arm: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "name": "darkofit_panel3_worker_attempt_v1",
        "binding": binding,
        "worker_key": _worker_key(coordinate, arm),
        "coordinate": {
            key: int(coordinate[key])
            for key in ("task_id", "repeat", "fold", "sample")
        },
        "arm": arm,
    }
    payload["attempt_sha256"] = _json_sha256(payload)
    return payload


def _load_attempt(
    path: Path,
    binding: dict[str, Any],
    coordinate: dict[str, int],
    arm: str,
    *,
    allowed_root: Path = ROOT,
) -> tuple[str, str]:
    try:
        encoded = common.secure_read_bytes(
            path,
            allowed_root=allowed_root,
        ).decode("utf-8")
    except (RuntimeError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"invalid panel-3 worker attempt: {path}"
        ) from exc
    payload = _json_loads(encoded, "worker attempt")
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"panel-3 worker attempt is not an object: {path}"
        )
    expected = _attempt_payload(binding, coordinate, arm)
    if payload != expected:
        raise RuntimeError(f"panel-3 worker attempt changed: {path}")
    return expected["attempt_sha256"], hashlib.sha256(
        encoded.encode("utf-8")
    ).hexdigest()


def _create_attempt(
    path: Path,
    binding: dict[str, Any],
    coordinate: dict[str, int],
    arm: str,
    *,
    allowed_root: Path = ROOT,
) -> tuple[str, str]:
    if path.is_symlink():
        raise RuntimeError(
            f"refusing symlink panel-3 worker attempt: {path}"
        )
    payload = _attempt_payload(binding, coordinate, arm)
    encoded = _json_file_bytes(payload)
    common.atomic_create(path, encoded, allowed_root=allowed_root)
    return payload["attempt_sha256"], hashlib.sha256(encoded).hexdigest()


def _refuse_invalidated_campaign(spool_directory: Path) -> None:
    path = _campaign_invalidation_path(spool_directory)
    if path.exists() or path.is_symlink():
        raise RuntimeError(
            "panel-3 campaign is permanently invalidated by "
            f"{CAMPAIGN_INVALIDATION_FILENAME}"
        )


def _invalidate_campaign(
    spool_directory: Path,
    binding: dict[str, Any] | None,
    message: str,
) -> None:
    path = _campaign_invalidation_path(spool_directory)
    payload = {
        "schema_version": 1,
        "name": "darkofit_panel3_campaign_invalidation_v1",
        "reason": "source_or_registry_drift",
        "binding_sha256": (
            _json_sha256(binding) if isinstance(binding, dict) else None
        ),
        "message": str(message),
    }
    payload["invalidation_sha256"] = _json_sha256(payload)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        common.atomic_create(path, encoded, allowed_root=spool_directory)
    except FileExistsError:
        pass


def _campaign_source_attestation(
    registry: dict[str, Any],
    registry_path: Path,
) -> dict[str, Any]:
    observed_registry, registry_file_sha256 = common.secure_load_json(
        registry_path
    )
    if observed_registry != registry:
        raise RuntimeError("panel-3 registry changed during execution")
    darko_source, chimera_source = _source_state(registry)
    return {
        "registry_file_sha256": registry_file_sha256,
        "registry_canonical_sha256": registry["registry_sha256"],
        "darkofit": darko_source,
        "chimeraboost": chimera_source,
    }


def _validate_parent_campaign_boundary(
    registry_path: Path,
    binding: dict[str, Any],
) -> dict[str, Any]:
    registry, registry_file_sha256 = common.secure_load_json(registry_path)
    if (
        not isinstance(registry, dict)
        or registry_file_sha256 != binding["registry_file_sha256"]
        or registry.get("registry_sha256")
        != binding["registry_canonical_sha256"]
    ):
        raise RuntimeError("panel-3 registry changed during execution")
    attestation = _campaign_source_attestation(registry, registry_path)
    if (
        attestation["darkofit"].get("head") != binding["darkofit_head"]
        or attestation["chimeraboost"].get("head")
        != binding["chimeraboost_head"]
    ):
        raise RuntimeError("panel-3 source changed during execution")
    return attestation


def _guard_parent_campaign_boundary(
    registry_path: Path,
    spool_directory: Path,
    binding: dict[str, Any],
) -> dict[str, Any]:
    _refuse_invalidated_campaign(spool_directory)
    try:
        return _validate_parent_campaign_boundary(registry_path, binding)
    except Exception as exc:
        _invalidate_campaign(spool_directory, binding, str(exc))
        raise


def _invalidate_worker_campaign(message: str) -> None:
    raw_directory = os.environ.get(WORKER_SPOOL_DIRECTORY_ENV)
    binding_sha256 = os.environ.get(WORKER_BINDING_SHA256_ENV)
    if (
        not raw_directory
        or not _is_sha256(binding_sha256)
    ):
        return
    directory = Path(raw_directory).expanduser().absolute()
    _invalidate_campaign(
        directory,
        {"worker_binding_sha256": binding_sha256},
        message,
    )


def _validate_worker_claim(
    registry: dict[str, Any],
    registry_path: Path,
    spool_directory: Path,
    coordinate: dict[str, int],
    arm: str,
) -> dict[str, Any]:
    raw_directory = os.environ.get(WORKER_SPOOL_DIRECTORY_ENV)
    raw_binding = os.environ.get(WORKER_BINDING_JSON_ENV)
    binding_sha256 = os.environ.get(WORKER_BINDING_SHA256_ENV)
    if (
        not raw_directory
        or not raw_binding
        or not _is_sha256(binding_sha256)
        or Path(raw_directory).expanduser().absolute()
        != spool_directory.expanduser().absolute()
    ):
        raise RuntimeError(
            "panel-3 worker requires a durable parent attempt claim"
        )
    binding = _json_loads(raw_binding, "worker binding")
    if (
        not isinstance(binding, dict)
        or _json_sha256(binding) != binding_sha256
    ):
        raise RuntimeError("panel-3 worker parent binding changed")
    _refuse_invalidated_campaign(spool_directory)
    registry_snapshot, registry_file_sha256 = common.secure_load_json(
        registry_path
    )
    if (
        registry_snapshot != registry
        or registry_file_sha256 != binding.get("registry_file_sha256")
        or registry.get("registry_sha256")
        != binding.get("registry_canonical_sha256")
    ):
        raise RuntimeError("panel-3 worker parent binding changed")
    attempt = _attempt_path(spool_directory, coordinate, arm)
    completion = _spool_path(spool_directory, coordinate, arm)
    if completion.exists() or completion.is_symlink():
        raise RuntimeError(
            "panel-3 worker attempt already has a completed spool"
        )
    _load_attempt(
        attempt,
        binding,
        coordinate,
        arm,
        allowed_root=spool_directory,
    )
    return binding


def _validate_spool_result(result: dict[str, Any]) -> None:
    """Apply the analyzer's exact public raw-result contract before persistence."""
    from benchmarks import analyze_panel3_confirmation as analyzer

    if result.get("status") == "failed":
        analyzer._validate_comparator_failure(result)
    else:
        analyzer._validate_result(result)
        analyzer._validate_fitted_metadata(result, strict=True)


def _spool_payload(
    binding: dict[str, Any],
    coordinate: dict[str, int],
    arm: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "name": "darkofit_panel3_worker_spool_v1",
        "binding": binding,
        "worker_key": _worker_key(coordinate, arm),
        "result_sha256": _json_sha256(result),
        "result": result,
    }
    payload["spool_record_sha256"] = _json_sha256(payload)
    return payload


def _load_spool(
    path: Path,
    binding: dict[str, Any],
    coordinate: dict[str, int],
    arm: str,
    *,
    allowed_root: Path = ROOT,
) -> tuple[dict[str, Any], str, str]:
    try:
        encoded = common.secure_read_bytes(
            path,
            allowed_root=allowed_root,
        ).decode("utf-8")
    except (RuntimeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"invalid panel-3 spool file: {path}") from exc
    payload = _json_loads(encoded, "spool")
    if not isinstance(payload, dict):
        raise RuntimeError(f"panel-3 spool record is not an object: {path}")
    expected_hash = payload.get("spool_record_sha256")
    unhashed = dict(payload)
    unhashed.pop("spool_record_sha256", None)
    if expected_hash != _json_sha256(unhashed):
        raise RuntimeError(f"panel-3 spool record hash is invalid: {path}")
    worker_key = _worker_key(coordinate, arm)
    if (
        payload.get("binding") != binding
        or payload.get("worker_key") != worker_key
        or payload.get("result_sha256")
        != _json_sha256(payload.get("result"))
    ):
        raise RuntimeError(f"panel-3 spool binding changed: {path}")
    result = payload["result"]
    if (
        not isinstance(result, dict)
        or result.get("worker_key") != worker_key
        or result.get("arm") != arm
    ):
        raise RuntimeError(f"panel-3 spool result changed: {path}")
    _validate_spool_result(result)
    return (
        result,
        expected_hash,
        hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    )


def _create_spool(
    path: Path,
    binding: dict[str, Any],
    coordinate: dict[str, int],
    arm: str,
    result: dict[str, Any],
    *,
    allowed_root: Path = ROOT,
) -> tuple[dict[str, Any], str, str]:
    if path.is_symlink():
        raise RuntimeError(f"refusing symlink panel-3 spool record: {path}")
    _validate_spool_result(result)
    payload = _spool_payload(binding, coordinate, arm, result)
    encoded = _json_file_bytes(payload)
    try:
        common.atomic_create(path, encoded, allowed_root=allowed_root)
    except FileExistsError:
        return _load_spool(
            path,
            binding,
            coordinate,
            arm,
            allowed_root=allowed_root,
        )
    return (
        result,
        payload["spool_record_sha256"],
        hashlib.sha256(encoded).hexdigest(),
    )


def _comparator_failure(
    coordinate: dict[str, int],
    arm: str,
    *,
    returncode: int | None,
    stdout: str | None,
    stderr: str | None,
    failure_kind: str,
    message: str,
) -> dict[str, Any]:
    if arm not in COMPARATOR_ARMS:
        raise ValueError("only descriptive comparators may record failures")
    if failure_kind not in {
        "worker_launch_failure",
        "worker_process_failure",
        "worker_protocol_failure",
    }:
        raise ValueError("invalid panel-3 comparator failure kind")
    payload = {
        "worker_key": _worker_key(coordinate, arm),
        "task_id": int(coordinate["task_id"]),
        "coordinate": {
            key: int(coordinate[key])
            for key in ("repeat", "fold", "sample")
        },
        "arm": arm,
        "status": "failed",
        "failure_kind": failure_kind,
        "returncode": returncode,
        "worker_stdout": stdout,
        "worker_stderr": stderr,
        "message": str(message),
    }
    payload["failure_fingerprint_sha256"] = _json_sha256(payload)
    return payload


def _run_one(
    registry_path: Path,
    coordinate: dict[str, int],
    arm: str,
    spool_directory: Path,
    binding: dict[str, Any],
) -> tuple[dict[str, Any], str, str, str, str, bool]:
    path = _spool_path(spool_directory, coordinate, arm)
    attempt_path = _attempt_path(spool_directory, coordinate, arm)
    _refuse_invalidated_campaign(spool_directory)
    if path.exists() or path.is_symlink():
        if not (attempt_path.exists() or attempt_path.is_symlink()):
            raise RuntimeError(
                "panel-3 completed spool has no durable worker attempt"
            )
        attempt_hash, attempt_file_hash = _load_attempt(
            attempt_path,
            binding,
            coordinate,
            arm,
            allowed_root=spool_directory,
        )
        _guard_parent_campaign_boundary(
            registry_path,
            spool_directory,
            binding,
        )
        result, spool_hash, spool_file_hash = _load_spool(
            path,
            binding,
            coordinate,
            arm,
            allowed_root=spool_directory,
        )
        if result.get("status") == "failed" and arm not in COMPARATOR_ARMS:
            raise RuntimeError(
                "panel-3 decision-arm spool contains a failure record"
            )
        return (
            result,
            spool_hash,
            spool_file_hash,
            attempt_hash,
            attempt_file_hash,
            True,
        )
    if attempt_path.exists() or attempt_path.is_symlink():
        _load_attempt(
            attempt_path,
            binding,
            coordinate,
            arm,
            allowed_root=spool_directory,
        )
        raise RuntimeError(
            "panel-3 worker attempt exists without a valid completed spool; "
            "this v1 coordinate is permanently invalid"
        )
    _guard_parent_campaign_boundary(
        registry_path,
        spool_directory,
        binding,
    )
    try:
        attempt_hash, attempt_file_hash = _create_attempt(
            attempt_path,
            binding,
            coordinate,
            arm,
            allowed_root=spool_directory,
        )
    except FileExistsError:
        _load_attempt(
            attempt_path,
            binding,
            coordinate,
            arm,
            allowed_root=spool_directory,
        )
        raise RuntimeError(
            "panel-3 worker attempt was claimed concurrently without a "
            "valid completed spool; this v1 coordinate is permanently invalid"
        ) from None
    environment = _worker_environment()
    environment.update(
        {
            WORKER_SPOOL_DIRECTORY_ENV: str(spool_directory),
            WORKER_BINDING_SHA256_ENV: _json_sha256(binding),
            WORKER_BINDING_JSON_ENV: json.dumps(
                binding,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        }
    )

    def publish(
        result: dict[str, Any],
    ) -> tuple[dict[str, Any], str, str]:
        _guard_parent_campaign_boundary(
            registry_path,
            spool_directory,
            binding,
        )
        return _create_spool(
            path,
            binding,
            coordinate,
            arm,
            result,
            allowed_root=spool_directory,
        )

    try:
        completed = subprocess.run(
            _worker_command(registry_path, coordinate, arm),
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        _guard_parent_campaign_boundary(
            registry_path,
            spool_directory,
            binding,
        )
        if arm not in COMPARATOR_ARMS:
            raise
        failure = _comparator_failure(
            coordinate,
            arm,
            returncode=None,
            stdout=None,
            stderr=None,
            failure_kind="worker_launch_failure",
            message=str(exc),
        )
        failure, spool_hash, spool_file_hash = publish(failure)
        return (
            failure,
            spool_hash,
            spool_file_hash,
            attempt_hash,
            attempt_file_hash,
            False,
        )
    _guard_parent_campaign_boundary(
        registry_path,
        spool_directory,
        binding,
    )
    worker_key = _worker_key(coordinate, arm)
    if completed.returncode:
        message = (
            f"panel-3 worker {worker_key} failed with "
            f"{completed.returncode}\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
        if arm not in COMPARATOR_ARMS:
            raise RuntimeError(message)
        failure = _comparator_failure(
            coordinate,
            arm,
            returncode=int(completed.returncode),
            stdout=completed.stdout.strip() or None,
            stderr=completed.stderr.strip() or None,
            failure_kind="worker_process_failure",
            message=message,
        )
        failure, spool_hash, spool_file_hash = publish(failure)
        return (
            failure,
            spool_hash,
            spool_file_hash,
            attempt_hash,
            attempt_file_hash,
            False,
        )
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if len(lines) != 1:
        message = (
            f"panel-3 worker {worker_key} emitted {len(lines)} results\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
        if arm not in COMPARATOR_ARMS:
            raise RuntimeError(message)
        failure = _comparator_failure(
            coordinate,
            arm,
            returncode=0,
            stdout=completed.stdout.strip() or None,
            stderr=completed.stderr.strip() or None,
            failure_kind="worker_protocol_failure",
            message=message,
        )
        failure, spool_hash, spool_file_hash = publish(failure)
        return (
            failure,
            spool_hash,
            spool_file_hash,
            attempt_hash,
            attempt_file_hash,
            False,
        )
    try:
        result = _json_loads(
            lines[0][len(WORKER_PREFIX) :], "worker result"
        )
    except RuntimeError as exc:
        if arm not in COMPARATOR_ARMS:
            raise
        failure = _comparator_failure(
            coordinate,
            arm,
            returncode=0,
            stdout=completed.stdout.strip() or None,
            stderr=completed.stderr.strip() or None,
            failure_kind="worker_protocol_failure",
            message=str(exc),
        )
        failure, spool_hash, spool_file_hash = publish(failure)
        return (
            failure,
            spool_hash,
            spool_file_hash,
            attempt_hash,
            attempt_file_hash,
            False,
        )
    if (
        not isinstance(result, dict)
        or result.get("worker_key") != worker_key
        or result.get("arm") != arm
    ):
        message = f"panel-3 worker {worker_key} identity changed"
        if arm not in COMPARATOR_ARMS:
            raise RuntimeError(message)
        failure = _comparator_failure(
            coordinate,
            arm,
            returncode=0,
            stdout=completed.stdout.strip() or None,
            stderr=completed.stderr.strip() or None,
            failure_kind="worker_protocol_failure",
            message=message,
        )
        failure, spool_hash, spool_file_hash = publish(failure)
        return (
            failure,
            spool_hash,
            spool_file_hash,
            attempt_hash,
            attempt_file_hash,
            False,
        )
    result["worker_stdout"] = (
        "\n".join(
            line
            for line in completed.stdout.splitlines()
            if not line.startswith(WORKER_PREFIX)
        ).strip()
        or None
    )
    result["worker_stderr"] = completed.stderr.strip() or None
    try:
        _validate_spool_result(result)
    except Exception as exc:
        if arm not in COMPARATOR_ARMS:
            raise
        failure = _comparator_failure(
            coordinate,
            arm,
            returncode=0,
            stdout=completed.stdout.strip() or None,
            stderr=completed.stderr.strip() or None,
            failure_kind="worker_protocol_failure",
            message=str(exc),
        )
        failure, spool_hash, spool_file_hash = publish(failure)
        return (
            failure,
            spool_hash,
            spool_file_hash,
            attempt_hash,
            attempt_file_hash,
            False,
        )
    result, spool_hash, spool_file_hash = publish(result)
    return (
        result,
        spool_hash,
        spool_file_hash,
        attempt_hash,
        attempt_file_hash,
        False,
    )


def _run_wave(
    registry_path: Path,
    coordinates: list[dict[str, int]],
    arm: str,
    spool_directory: Path,
    binding: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results = []
    spool_records = []
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        futures = {
            executor.submit(
                _run_one,
                registry_path,
                coordinate,
                arm,
                spool_directory,
                binding,
            ): coordinate
            for coordinate in coordinates
        }
        completed = 0
        for future in as_completed(futures):
            coordinate = futures[future]
            (
                result,
                spool_hash,
                spool_file_hash,
                attempt_hash,
                attempt_file_hash,
                resumed,
            ) = future.result()
            results.append(result)
            spool_records.append(
                {
                    "worker_key": _worker_key(coordinate, arm),
                    "filename": _spool_path(
                        spool_directory, coordinate, arm
                    ).name,
                    "spool_record_sha256": spool_hash,
                    "spool_file_sha256": spool_file_hash,
                    "attempt_filename": _attempt_path(
                        spool_directory, coordinate, arm
                    ).name,
                    "attempt_sha256": attempt_hash,
                    "attempt_file_sha256": attempt_file_hash,
                    "resumed": bool(resumed),
                }
            )
            completed += 1
            print(
                f"{arm}: {completed}/{len(coordinates)} "
                f"({_worker_key(coordinate, arm)}, "
                f"{'resumed' if resumed else 'fresh'})",
                flush=True,
            )
    return (
        sorted(results, key=lambda row: row["worker_key"]),
        sorted(spool_records, key=lambda row: row["worker_key"]),
    )


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    common.validate_create_path(args.output)
    common.ensure_output_directory(args.spool_directory)
    _refuse_invalidated_campaign(args.spool_directory)
    registry, registry_file_sha256 = common.secure_load_json(args.registry)
    if not isinstance(registry, dict):
        raise RuntimeError("panel-3 registry is not an object")
    validate_registry(registry, registry_path=args.registry)
    darko_source, chimera_source = _source_state(registry)
    runtime = _validate_runtime_contract(registry["candidate_contract"])
    runtime_contract_normalized_sha256 = _json_sha256(runtime)
    machine_fingerprint = _machine_fingerprint()
    plan = execution_plan(
        registry,
        registry_path=args.registry,
        registry_file_sha256=registry_file_sha256,
        validate_registry_boundary=False,
    )
    binding = _spool_binding(
        registry,
        args.registry,
        registry_file_sha256,
        darko_source,
        chimera_source,
        runtime_contract_normalized_sha256,
        machine_fingerprint["sha256"],
    )
    coordinates = list(registry["coordinates"])
    arm_order = _arm_order(registry)
    decision_arms = _decision_arms(registry)
    results = []
    comparator_failures = []
    spool_records = []
    for wave, arm in enumerate(arm_order, start=1):
        print(f"wave {wave}/{len(arm_order)}: {arm}", flush=True)
        fold_values = sorted(
            {
                (
                    int(coordinate["repeat"]),
                    int(coordinate["fold"]),
                    int(coordinate["sample"]),
                )
                for coordinate in coordinates
            }
        )
        for repeat, fold, sample in fold_values:
            _guard_parent_campaign_boundary(
                args.registry,
                args.spool_directory,
                binding,
            )
            fold_coordinates = [
                coordinate
                for coordinate in coordinates
                if (
                    int(coordinate["repeat"]) == repeat
                    and int(coordinate["fold"]) == fold
                    and int(coordinate["sample"]) == sample
                )
            ]
            if len({row["task_id"] for row in fold_coordinates}) != len(
                fold_coordinates
            ):
                raise RuntimeError(
                    "panel-3 fold wave repeats a task"
                )
            wave_results, wave_spool = _run_wave(
                args.registry,
                fold_coordinates,
                arm,
                args.spool_directory,
                binding,
            )
            results.extend(
                result
                for result in wave_results
                if result.get("status") != "failed"
            )
            comparator_failures.extend(
                result
                for result in wave_results
                if result.get("status") == "failed"
            )
            spool_records.extend(wave_spool)
    _guard_parent_campaign_boundary(
        args.registry,
        args.spool_directory,
        binding,
    )
    final_registry, final_registry_file_sha256 = common.secure_load_json(
        args.registry
    )
    if (
        final_registry != registry
        or final_registry_file_sha256 != registry_file_sha256
    ):
        _invalidate_campaign(
            args.spool_directory,
            binding,
            "panel-3 registry changed during execution",
        )
        raise RuntimeError("panel-3 registry changed during execution")
    expected = {
        _worker_key(coordinate, arm)
        for coordinate in coordinates
        for arm in arm_order
    }
    expected_decision = {
        _worker_key(coordinate, arm)
        for coordinate in coordinates
        for arm in decision_arms
    }
    observed_results = {row["worker_key"] for row in results}
    observed_failures = {
        row["worker_key"] for row in comparator_failures
    }
    if (
        observed_results & observed_failures
        or observed_results | observed_failures != expected
        or expected_decision - observed_results
        or len(results) + len(comparator_failures) != len(expected)
        or any(
            row["arm"] not in COMPARATOR_ARMS
            for row in comparator_failures
        )
    ):
        raise RuntimeError("panel-3 raw execution is incomplete")
    if (
        _validate_runtime_contract(registry["candidate_contract"]) != runtime
        or _machine_fingerprint() != machine_fingerprint
    ):
        raise RuntimeError(
            "panel-3 runtime or machine changed during execution"
        )
    artifact = {
        "schema_version": 1,
        "name": "darkofit_panel3_confirmation_raw_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "registry": {
            "path": _artifact_path(args.registry),
            "file_sha256": registry_file_sha256,
            "canonical_sha256": registry["registry_sha256"],
        },
        "execution_plan": plan,
        "protocol": {
            "path": _artifact_path(common.PROTOCOL),
            "sha256": _sha256(common.PROTOCOL),
            "runner_path": _artifact_path(Path(__file__).resolve()),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "analyzer_path": _artifact_path(common.ANALYZER),
            "analyzer_sha256": _sha256(common.ANALYZER),
            "candidate_contract_path": _artifact_path(
                common.CANDIDATE_CONTRACT
            ),
            "candidate_contract_sha256": _sha256(
                common.CANDIDATE_CONTRACT
            ),
            "power_design_decision_path": _artifact_path(
                common.POWER_DESIGN_DECISION
            ),
            "power_design_file_sha256": _sha256(
                common.POWER_DESIGN_DECISION
            ),
            "power_design_decision_sha256": registry[
                "power_design_decision_sha256"
            ],
            "arms": list(arm_order),
            "coordinate_count": len(coordinates),
            "worker_count": len(expected),
            "decision_worker_count": len(expected_decision),
            "successful_worker_count": len(results),
            "comparator_failure_count": len(comparator_failures),
            "comparator_failures_affect_candidate_gates": False,
            "threads_per_worker": THREADS_PER_WORKER,
            "concurrent_workers": CONCURRENT_WORKERS,
            "validation_fraction": VALIDATION_FRACTION,
            "guarded_cross_ratio": GUARDED_CROSS_RATIO,
            "prediction_block_seconds": PREDICTION_BLOCK_SECONDS,
            "prediction_min_calls": PREDICTION_MIN_CALLS,
            "prediction_max_calls": PREDICTION_MAX_CALLS,
            "task_drop_allowed": False,
            "task_imputation_allowed": False,
            "outcome_dependent_rerun_allowed": False,
        },
        "sources": {
            "darkofit": darko_source,
            "chimeraboost": chimera_source,
        },
        "environment": {
            "python": sys.version,
            "runtime_contract": runtime,
            "runtime_contract_normalized_sha256": (
                runtime_contract_normalized_sha256
            ),
            "machine_fingerprint": machine_fingerprint,
            "machine": creator._machine_details(),
            "dependencies": creator._dependency_versions(),
        },
        "spool": {
            "directory": _artifact_path(args.spool_directory),
            "binding": binding,
            "record_count": len(spool_records),
            "resumed_record_count": sum(
                bool(row["resumed"]) for row in spool_records
            ),
            "records": sorted(
                spool_records, key=lambda row: row["worker_key"]
            ),
        },
        "results": sorted(results, key=lambda row: row["worker_key"]),
        "comparator_failures": sorted(
            comparator_failures, key=lambda row: row["worker_key"]
        ),
        "outcomes_scored": True,
        "analysis_performed": False,
        "default_promotion_authorized": False,
        "protocol_deviations": [],
        "task_imputation_used": False,
        "task_drop_used": False,
    }
    artifact["raw_artifact_sha256"] = _json_sha256(artifact)
    from benchmarks import analyze_panel3_confirmation as analyzer

    analyzer.validate_raw(
        artifact,
        registry,
        registry_path=args.registry,
        registry_file_sha256=registry_file_sha256,
        verify_current_files=True,
    )
    _guard_parent_campaign_boundary(
        args.registry,
        args.spool_directory,
        binding,
    )
    publish_registry, publish_registry_file_sha256 = (
        common.secure_load_json(args.registry)
    )
    if (
        publish_registry != registry
        or publish_registry_file_sha256 != registry_file_sha256
    ):
        _invalidate_campaign(
            args.spool_directory,
            binding,
            "panel-3 registry changed before raw publication",
        )
        raise RuntimeError("panel-3 registry changed before raw publication")
    common.atomic_create(
        args.output,
        (
            json.dumps(
                artifact, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n"
        ).encode("utf-8"),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "raw_file_sha256": common.sha256_file(args.output),
                "raw_artifact_sha256": artifact[
                    "raw_artifact_sha256"
                ],
                "worker_count": len(expected),
                "successful_worker_count": len(results),
                "comparator_failure_count": len(comparator_failures),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--spool-directory",
        type=Path,
        default=DEFAULT_SPOOL_DIRECTORY,
    )
    parser.add_argument("--worker-task", type=int)
    parser.add_argument("--worker-repeat", type=int)
    parser.add_argument("--worker-fold", type=int)
    parser.add_argument("--worker-sample", type=int)
    parser.add_argument("--worker-arm", choices=ARM_ORDER)
    args = parser.parse_args(argv)
    args.registry = Path(
        os.path.abspath(os.path.expanduser(args.registry))
    )
    args.output = Path(os.path.abspath(os.path.expanduser(args.output)))
    args.spool_directory = Path(
        os.path.abspath(os.path.expanduser(args.spool_directory))
    )
    worker_values = (
        args.worker_task,
        args.worker_repeat,
        args.worker_fold,
        args.worker_sample,
        args.worker_arm,
    )
    if any(value is not None for value in worker_values) and not all(
        value is not None for value in worker_values
    ):
        parser.error("all panel-3 worker arguments must be supplied together")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        args.registry != DEFAULT_REGISTRY
        or args.output != DEFAULT_OUTPUT
        or args.spool_directory != DEFAULT_SPOOL_DIRECTORY
    ):
        raise RuntimeError("panel-3 execution path changed")
    if args.worker_arm is not None:
        registry, _registry_file_sha256 = common.secure_load_json(
            args.registry
        )
        if not isinstance(registry, dict):
            raise RuntimeError("panel-3 registry is not an object")
        coordinate = {
            "task_id": int(args.worker_task),
            "repeat": int(args.worker_repeat),
            "fold": int(args.worker_fold),
            "sample": int(args.worker_sample),
        }
        _validate_worker_claim(
            registry,
            args.registry,
            args.spool_directory,
            coordinate,
            args.worker_arm,
        )
        result = run_worker(
            registry,
            args.registry,
            coordinate,
            args.worker_arm,
        )
        print(
            WORKER_PREFIX
            + json.dumps(result, sort_keys=True, allow_nan=False),
            flush=True,
        )
        return 0
    run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
