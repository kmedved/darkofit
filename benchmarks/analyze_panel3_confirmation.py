#!/usr/bin/env python3
"""Validate and analyze the immutable raw panel-3 confirmation artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import panel3_registry_common as common  # noqa: E402
from benchmarks import panel3_data_contract as data_contract  # noqa: E402
from benchmarks import run_panel3_confirmation as runner  # noqa: E402


DEFAULT_INPUT = ROOT / "benchmarks" / "panel3_confirmation_raw.json"
DEFAULT_REGISTRY = ROOT / "benchmarks" / "panel3_registry.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "panel3_confirmation_summary.json"
DEFAULT_MARKDOWN = ROOT / "benchmarks" / "panel3_confirmation_result.md"
BOOTSTRAP_SEED = 20_260_717
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_BATCH = 10_000
PANEL3_V1_LINEAGES = 12
PANEL3_V1_STRATA = (
    "smooth_numeric",
    "mixed_categorical",
    "applied_noisy",
)
PANEL3_V1_QUALITY_GATES = {
    "equal_dataset_geomean_ratio_at_most": 0.995,
    "bootstrap_upper_ratio_at_most": 1.002,
    "leave_one_favorable_dataset_out_ratio_at_most": 0.998,
    "worst_dataset_ratio_at_most": 1.005,
}
PANEL3_V1_OPERATIONAL_GATES = {
    "fit_seconds_ratio_at_most": 6.0,
    "worst_dataset_fit_seconds_ratio_at_most": 12.0,
    "predict_seconds_ratio_at_most": 1.5,
    "peak_rss_ratio_at_most": 2.5,
}
PANEL3_V1_SELECTION_MAPPING = {
    "neither_passes": None,
    "only_guarded_cross_features_policy_passes": (
        "guarded_cross_features_policy"
    ),
    "only_t5_composite_policy_passes": "t5_composite_policy",
    "both_pass": "t5_composite_policy",
}
PANEL3_V1_BOTH_PASS_REASON = (
    "t5_composite_policy contains guarded crosses as a constituent, so the "
    "broader frozen policy has fixed precedence without metric ranking or "
    "post-outcome discretion"
)
PANEL3_V1_PREDICTION_TIMING = {
    "minimum_block_seconds": 0.25,
    "minimum_calls": 5,
    "maximum_calls": 20_000,
}
RAW_FIELDS = {
    "schema_version",
    "name",
    "created_at",
    "registry",
    "execution_plan",
    "protocol",
    "sources",
    "environment",
    "spool",
    "results",
    "comparator_failures",
    "outcomes_scored",
    "analysis_performed",
    "default_promotion_authorized",
    "protocol_deviations",
    "task_imputation_used",
    "task_drop_used",
    "raw_artifact_sha256",
}
PROTOCOL_FIELDS = {
    "path",
    "sha256",
    "runner_path",
    "runner_sha256",
    "analyzer_path",
    "analyzer_sha256",
    "candidate_contract_path",
    "candidate_contract_sha256",
    "power_design_decision_path",
    "power_design_file_sha256",
    "power_design_decision_sha256",
    "arms",
    "coordinate_count",
    "worker_count",
    "decision_worker_count",
    "successful_worker_count",
    "comparator_failure_count",
    "comparator_failures_affect_candidate_gates",
    "threads_per_worker",
    "concurrent_workers",
    "validation_fraction",
    "guarded_cross_ratio",
    "prediction_block_seconds",
    "prediction_min_calls",
    "prediction_max_calls",
    "task_drop_allowed",
    "task_imputation_allowed",
    "outcome_dependent_rerun_allowed",
}


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


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


def _json_loads(encoded: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            encoded,
            object_pairs_hook=_json_object,
            parse_float=_json_float,
            parse_int=_json_int,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {constant}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid panel-3 {label} JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"panel-3 {label} must be a JSON object")
    return value


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"panel-3 {label} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError(f"panel-3 {label} must be finite and positive")
    return result


def _nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"panel-3 {label} must be finite and nonnegative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RuntimeError(f"panel-3 {label} must be finite and nonnegative")
    return result


def geomean(values: list[float] | tuple[float, ...]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.size == 0
        or not np.isfinite(array).all()
        or np.any(array <= 0.0)
    ):
        raise ValueError("geomean requires finite positive values")
    return float(np.exp(np.mean(np.log(array))))


def quality_metrics(dataset_ratios: dict[str, float]) -> dict[str, Any]:
    if len(dataset_ratios) != PANEL3_V1_LINEAGES:
        raise ValueError("panel-3 requires exactly 12 dataset ratios")
    if any(
        not isinstance(name, str)
        or not name
        or not math.isfinite(float(ratio))
        or float(ratio) <= 0.0
        for name, ratio in dataset_ratios.items()
    ):
        raise ValueError("panel-3 dataset ratio ledger is invalid")
    ordered = {
        name: float(dataset_ratios[name]) for name in sorted(dataset_ratios)
    }
    aggregate = geomean(list(ordered.values()))
    leave_one_out = {
        name: geomean(
            [ratio for other, ratio in ordered.items() if other != name]
        )
        for name in ordered
    }
    return {
        "dataset_ratios": ordered,
        "equal_dataset_geomean_ratio": aggregate,
        "leave_one_out_ratios": leave_one_out,
        "least_favorable_leave_one_out_ratio": max(
            leave_one_out.values()
        ),
        "worst_dataset_ratio": max(ordered.values()),
    }


def adjudicate_candidate(
    dataset_ratios: dict[str, float],
    *,
    bonferroni_bootstrap_upper: float,
    equal_dataset_fit_seconds_ratio: float,
    worst_dataset_fit_seconds_ratio: float,
    equal_dataset_predict_seconds_ratio: float,
    equal_dataset_peak_rss_ratio: float,
    complete: bool,
    integrity_ok: bool,
    deviations: list[str],
    quality_gates: dict[str, float] | None = None,
    operational_gates: dict[str, float] | None = None,
    bootstrap_percentile: float = 97.5,
    per_candidate_one_sided_alpha: float = 0.025,
) -> dict[str, Any]:
    quality_gates = (
        PANEL3_V1_QUALITY_GATES
        if quality_gates is None
        else quality_gates
    )
    if (
        not isinstance(quality_gates, dict)
        or set(quality_gates) != set(PANEL3_V1_QUALITY_GATES)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in quality_gates.values()
        )
    ):
        raise ValueError("panel-3 quality-gate contract is invalid")
    operational_gates = (
        PANEL3_V1_OPERATIONAL_GATES
        if operational_gates is None
        else operational_gates
    )
    if (
        not isinstance(operational_gates, dict)
        or set(operational_gates) != set(PANEL3_V1_OPERATIONAL_GATES)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in operational_gates.values()
        )
    ):
        raise ValueError("panel-3 operational-gate contract is invalid")
    metrics = quality_metrics(dataset_ratios)
    scalars = {
        "bonferroni_bootstrap_upper": bonferroni_bootstrap_upper,
        "equal_dataset_fit_seconds_ratio": equal_dataset_fit_seconds_ratio,
        "worst_dataset_fit_seconds_ratio": worst_dataset_fit_seconds_ratio,
        "equal_dataset_predict_seconds_ratio": (
            equal_dataset_predict_seconds_ratio
        ),
        "equal_dataset_peak_rss_ratio": equal_dataset_peak_rss_ratio,
    }
    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in scalars.values()
    ):
        raise ValueError("panel-3 candidate scalar is invalid")
    if not isinstance(deviations, list) or any(
        not isinstance(value, str) or not value for value in deviations
    ):
        raise ValueError("panel-3 deviation ledger is invalid")
    gates = {
        "point": (
            metrics["equal_dataset_geomean_ratio"]
            <= quality_gates["equal_dataset_geomean_ratio_at_most"]
        ),
        "bonferroni_uncertainty": (
            bonferroni_bootstrap_upper
            <= quality_gates["bootstrap_upper_ratio_at_most"]
        ),
        "concentration": (
            metrics["least_favorable_leave_one_out_ratio"]
            <= quality_gates[
                "leave_one_favorable_dataset_out_ratio_at_most"
            ]
        ),
        "harm": (
            metrics["worst_dataset_ratio"]
            <= quality_gates["worst_dataset_ratio_at_most"]
        ),
        "fit_cost": (
            equal_dataset_fit_seconds_ratio
            <= operational_gates["fit_seconds_ratio_at_most"]
            and worst_dataset_fit_seconds_ratio
            <= operational_gates[
                "worst_dataset_fit_seconds_ratio_at_most"
            ]
        ),
        "predict_cost": (
            equal_dataset_predict_seconds_ratio
            <= operational_gates["predict_seconds_ratio_at_most"]
        ),
        "rss_cost": (
            equal_dataset_peak_rss_ratio
            <= operational_gates["peak_rss_ratio_at_most"]
        ),
        "complete": complete is True,
        "integrity": integrity_ok is True,
        "no_deviations": deviations == [],
    }
    return {
        **metrics,
        **{name: float(value) for name, value in scalars.items()},
        "bootstrap_percentile": float(bootstrap_percentile),
        "per_candidate_one_sided_alpha": float(
            per_candidate_one_sided_alpha
        ),
        "gates": gates,
        "passes": all(gates.values()),
    }


def adjudicate_two_candidates(
    candidates: dict[str, dict[str, Any]],
    *,
    retained_candidates: tuple[str, ...] = runner.CANDIDATE_ARMS,
    default_selection_mapping: dict[str, str | None] | None = None,
    both_pass_reason: str = PANEL3_V1_BOTH_PASS_REASON,
    familywise_one_sided_alpha: float = 0.05,
    per_candidate_one_sided_alpha: float = 0.025,
    bootstrap_percentile: float = 97.5,
) -> dict[str, Any]:
    expected = set(retained_candidates)
    if set(candidates) != expected:
        raise ValueError("panel-3 must adjudicate every retained candidate")
    if any(
        not isinstance(result, dict) or type(result.get("passes")) is not bool
        for result in candidates.values()
    ):
        raise ValueError("panel-3 candidate result is malformed")
    if (
        isinstance(familywise_one_sided_alpha, bool)
        or not isinstance(familywise_one_sided_alpha, (int, float))
        or not math.isfinite(float(familywise_one_sided_alpha))
        or not 0.0 < float(familywise_one_sided_alpha) < 1.0
    ):
        raise ValueError("panel-3 familywise alpha is invalid")
    default_selection_mapping = (
        PANEL3_V1_SELECTION_MAPPING
        if default_selection_mapping is None
        else default_selection_mapping
    )
    if (
        not isinstance(default_selection_mapping, dict)
        or default_selection_mapping != PANEL3_V1_SELECTION_MAPPING
        or not isinstance(both_pass_reason, str)
        or not both_pass_reason
    ):
        raise ValueError("panel-3 default-selection contract is invalid")
    confirmed = sorted(
        name for name, result in candidates.items() if result["passes"]
    )
    if len(confirmed) == 2:
        mapping_key = "both_pass"
    elif confirmed == ["guarded_cross_features_policy"]:
        mapping_key = "only_guarded_cross_features_policy_passes"
    elif confirmed == ["t5_composite_policy"]:
        mapping_key = "only_t5_composite_policy_passes"
    else:
        mapping_key = "neither_passes"
    selected_default = default_selection_mapping[mapping_key]
    return {
        "candidate_results": candidates,
        "multiplicity_method": (
            (
                "Bonferroni two one-sided 97.5% hierarchical-bootstrap "
                "upper bounds"
            )
            if len(retained_candidates) == 2
            else (
                "single preregistered one-sided 95% "
                "hierarchical-bootstrap upper bound"
            )
        ),
        "post_outcome_winner_selection_used": False,
        "fixed_default_precedence": [
            default_selection_mapping["both_pass"],
            "guarded_cross_features_policy",
        ],
        "both_pass_default": default_selection_mapping["both_pass"],
        "both_pass_reason": both_pass_reason,
        "familywise_one_sided_alpha": float(
            familywise_one_sided_alpha
        ),
        "per_candidate_one_sided_alpha": float(
            per_candidate_one_sided_alpha
        ),
        "bootstrap_percentile": float(bootstrap_percentile),
        "each_candidate_independently_adjudicated": True,
        "independently_confirmed_candidates": confirmed,
        "selected_default_candidate": selected_default,
        "shipping_candidates": (
            [] if selected_default is None else [selected_default]
        ),
    }


def _result_coordinate(result: dict[str, Any]) -> tuple[int, int, int, int]:
    coordinate = result.get("coordinate")
    if not isinstance(coordinate, dict) or set(coordinate) != {
        "repeat",
        "fold",
        "sample",
    }:
        raise RuntimeError("panel-3 result coordinate is invalid")
    values = (
        result.get("task_id"),
        coordinate.get("repeat"),
        coordinate.get("fold"),
        coordinate.get("sample"),
    )
    if any(type(value) is not int or value < 0 for value in values):
        raise RuntimeError("panel-3 result coordinate is invalid")
    return tuple(values)


def _validate_source_attestation(value: Any) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"before", "after"}
        or value["before"] != value["after"]
    ):
        raise RuntimeError("panel-3 worker source attestation changed")
    attestation = value["before"]
    if (
        not isinstance(attestation, dict)
        or set(attestation)
        != {
            "registry_file_sha256",
            "registry_canonical_sha256",
            "darkofit",
            "chimeraboost",
        }
        or not _is_sha256(attestation["registry_file_sha256"])
        or not _is_sha256(attestation["registry_canonical_sha256"])
    ):
        raise RuntimeError("panel-3 worker source attestation changed")
    source_fields = {
        "path",
        "head",
        "branch",
        "clean",
        "status",
        "describe",
        "remotes",
        "tracked_main_refs",
    }
    for source in (attestation["darkofit"], attestation["chimeraboost"]):
        if (
            not isinstance(source, dict)
            or set(source) != source_fields
            or not isinstance(source["path"], str)
            or not Path(source["path"]).is_absolute()
            or not isinstance(source["head"], str)
            or len(source["head"]) != 40
            or any(
                character not in "0123456789abcdef"
                for character in source["head"]
            )
            or not isinstance(source["branch"], str)
            or source["clean"] is not True
            or source["status"] != []
            or (
                source["describe"] is not None
                and not isinstance(source["describe"], str)
            )
            or not isinstance(source["remotes"], dict)
            or any(
                not isinstance(key, str)
                or not key
                or not isinstance(remote, str)
                or not remote
                for key, remote in source["remotes"].items()
            )
            or not isinstance(source["tracked_main_refs"], dict)
            or any(
                ref not in {"origin/main", "upstream/main"}
                or not isinstance(digest, str)
                or len(digest) != 40
                or any(
                    character not in "0123456789abcdef"
                    for character in digest
                )
                for ref, digest in source["tracked_main_refs"].items()
            )
        ):
            raise RuntimeError(
                "panel-3 worker source attestation changed"
            )


def _validate_prediction_timing(
    value: Any,
    timing_policy: dict[str, float | int] | None = None,
) -> None:
    timing_policy = (
        PANEL3_V1_PREDICTION_TIMING
        if timing_policy is None
        else timing_policy
    )
    if (
        not isinstance(timing_policy, dict)
        or set(timing_policy) != set(PANEL3_V1_PREDICTION_TIMING)
        or isinstance(timing_policy["minimum_block_seconds"], bool)
        or not isinstance(
            timing_policy["minimum_block_seconds"], (int, float)
        )
        or not math.isfinite(
            float(timing_policy["minimum_block_seconds"])
        )
        or float(timing_policy["minimum_block_seconds"]) <= 0.0
        or type(timing_policy["minimum_calls"]) is not int
        or timing_policy["minimum_calls"] <= 0
        or type(timing_policy["maximum_calls"]) is not int
        or timing_policy["maximum_calls"]
        < timing_policy["minimum_calls"]
    ):
        raise RuntimeError("panel-3 prediction timing policy is invalid")
    if not isinstance(value, dict) or set(value) != {
        "per_call_median_seconds",
        "per_call_min_seconds",
        "per_call_max_seconds",
        "total_seconds",
        "call_count",
        "minimum_block_seconds",
    }:
        raise RuntimeError("panel-3 prediction timing is malformed")
    minimum = _positive_float(
        value["per_call_min_seconds"], "prediction minimum"
    )
    median = _positive_float(
        value["per_call_median_seconds"], "prediction median"
    )
    maximum = _positive_float(
        value["per_call_max_seconds"], "prediction maximum"
    )
    total = _positive_float(value["total_seconds"], "prediction block")
    call_count = value["call_count"]
    if (
        not minimum <= median <= maximum
        or total < timing_policy["minimum_block_seconds"]
        or type(call_count) is not int
        or call_count < timing_policy["minimum_calls"]
        or call_count > timing_policy["maximum_calls"]
    ):
        raise RuntimeError("panel-3 prediction timing is inconsistent")
    tolerance = max(1e-15, 1e-12 * total)
    if (
        total + tolerance < call_count * minimum
        or total - tolerance > call_count * maximum
        or float(value["minimum_block_seconds"])
        != timing_policy["minimum_block_seconds"]
    ):
        raise RuntimeError("panel-3 prediction timing policy changed")


def _validate_result(
    result: Any,
    prediction_timing_policy: dict[str, float | int] | None = None,
) -> None:
    required = {
        "worker_key",
        "task_id",
        "dataset_id",
        "dataset_name",
        "lineage_cluster",
        "stratum",
        "coordinate",
        "arm",
        "categorical_feature_indices",
        "categorical_feature_names",
        "ordinal_features",
        "feature_policy",
        "train_rows",
        "test_rows",
        "train_index_sha256",
        "test_index_sha256",
        "split_policy",
        "target_sha256",
        "rmse",
        "fit_seconds",
        "prediction_timing",
        "prediction_sha256",
        "metadata",
        "source_attestation",
        "warmup_seconds",
        "wall_seconds",
        "peak_rss_bytes",
        "behavior_fingerprint_sha256",
        "worker_stdout",
        "worker_stderr",
    }
    if not isinstance(result, dict) or set(result) != required:
        raise RuntimeError("panel-3 worker result fields changed")
    coordinate = _result_coordinate(result)
    arm = result["arm"]
    full_coordinate = {
        "task_id": coordinate[0],
        "repeat": coordinate[1],
        "fold": coordinate[2],
        "sample": coordinate[3],
    }
    if (
        arm not in runner.ARM_ORDER
        or result["worker_key"] != runner._worker_key(full_coordinate, arm)
        or type(result["dataset_id"]) is not int
        or result["dataset_id"] <= 0
        or not isinstance(result["dataset_name"], str)
        or not result["dataset_name"]
        or not isinstance(result["lineage_cluster"], str)
        or not result["lineage_cluster"]
        or result["stratum"] not in PANEL3_V1_STRATA
    ):
        raise RuntimeError("panel-3 worker identity changed")
    if (
        not isinstance(result["categorical_feature_indices"], list)
        or any(
            type(value) is not int or value < 0
            for value in result["categorical_feature_indices"]
        )
        or len(set(result["categorical_feature_indices"]))
        != len(result["categorical_feature_indices"])
        or not isinstance(result["categorical_feature_names"], list)
        or any(
            not isinstance(value, str) or not value
            for value in result["categorical_feature_names"]
        )
        or len(result["categorical_feature_names"])
        != len(result["categorical_feature_indices"])
        or len(set(result["categorical_feature_names"]))
        != len(result["categorical_feature_names"])
        or not isinstance(result["ordinal_features"], dict)
        or not isinstance(result["feature_policy"], dict)
        or not isinstance(result["split_policy"], dict)
        or not isinstance(result["metadata"], dict)
    ):
        raise RuntimeError("panel-3 worker feature metadata is invalid")
    if (
        set(result["feature_policy"])
        != {
            "kind",
            "policy_sha256",
            "source_columns_sha256",
            "dropped_columns",
            "generated_columns",
            "generated_values_sha256",
            "retained_source_columns",
            "retained_columns",
            "retained_feature_count",
            "retained_columns_sha256",
            "output_schema",
            "output_schema_sha256",
        }
        or result["feature_policy"]["kind"]
        not in {"none", "drop_columns", "target_free_transform_v1"}
        or not isinstance(
            result["feature_policy"]["dropped_columns"], list
        )
        or not isinstance(
            result["feature_policy"]["generated_columns"], list
        )
        or not isinstance(
            result["feature_policy"]["retained_columns"], list
        )
        or not isinstance(
            result["feature_policy"]["retained_source_columns"], list
        )
        or not isinstance(
            result["feature_policy"]["output_schema"], list
        )
        or any(
            not isinstance(value, str) or not value
            for field in (
                "dropped_columns",
                "generated_columns",
                "retained_columns",
                "retained_source_columns",
            )
            for value in result["feature_policy"][field]
        )
        or any(
            not isinstance(value, dict)
            or set(value) != {"name", "dtype"}
            or not isinstance(value["name"], str)
            or not value["name"]
            or not isinstance(value["dtype"], str)
            or not value["dtype"]
            for value in result["feature_policy"]["output_schema"]
        )
        or type(result["feature_policy"]["retained_feature_count"]) is not int
        or result["feature_policy"]["retained_feature_count"]
        != len(result["feature_policy"]["retained_columns"])
        or result["feature_policy"]["retained_feature_count"] <= 0
        or len(set(result["feature_policy"]["dropped_columns"]))
        != len(result["feature_policy"]["dropped_columns"])
        or len(set(result["feature_policy"]["generated_columns"]))
        != len(result["feature_policy"]["generated_columns"])
        or len(set(result["feature_policy"]["retained_columns"]))
        != len(result["feature_policy"]["retained_columns"])
        or len(set(result["feature_policy"]["retained_source_columns"]))
        != len(result["feature_policy"]["retained_source_columns"])
        or not set(result["feature_policy"]["generated_columns"])
        <= set(result["feature_policy"]["retained_columns"])
        or result["feature_policy"]["retained_columns"]
        != [
            *result["feature_policy"]["retained_source_columns"],
            *result["feature_policy"]["generated_columns"],
        ]
        or [
            value["name"]
            for value in result["feature_policy"]["output_schema"]
        ]
        != result["feature_policy"]["retained_columns"]
        or result["feature_policy"]["retained_columns_sha256"]
        != data_contract.canonical_json_sha256(
            result["feature_policy"]["retained_columns"]
        )
        or not _is_sha256(result["feature_policy"]["policy_sha256"])
        or not _is_sha256(
            result["feature_policy"]["source_columns_sha256"]
        )
        or not _is_sha256(
            result["feature_policy"]["generated_values_sha256"]
        )
        or result["feature_policy"]["output_schema_sha256"]
        != data_contract.canonical_json_sha256(
            result["feature_policy"]["output_schema"]
        )
        or set(result["split_policy"])
        != {"kind", "allow_unused_rows", "construction_sha256"}
        or result["split_policy"]["kind"]
        not in {"openml_official", "frozen_explicit"}
        or type(result["split_policy"]["allow_unused_rows"]) is not bool
        or (
            result["split_policy"]["construction_sha256"] is not None
            and not _is_sha256(
                result["split_policy"]["construction_sha256"]
            )
        )
    ):
        raise RuntimeError("panel-3 worker policy metadata is invalid")
    if (
        any(
            index >= result["feature_policy"]["retained_feature_count"]
            for index in result["categorical_feature_indices"]
        )
        or result["categorical_feature_names"]
        != [
            result["feature_policy"]["retained_columns"][index]
            for index in result["categorical_feature_indices"]
        ]
    ):
        raise RuntimeError(
            "panel-3 categorical index/name resolution is invalid"
        )
    for field in ("train_rows", "test_rows", "peak_rss_bytes"):
        if type(result[field]) is not int or result[field] <= 0:
            raise RuntimeError(f"panel-3 {field} is invalid")
    for field in (
        "train_index_sha256",
        "test_index_sha256",
        "target_sha256",
        "prediction_sha256",
        "behavior_fingerprint_sha256",
    ):
        if not _is_sha256(result[field]):
            raise RuntimeError(f"panel-3 {field} is invalid")
    _positive_float(result["rmse"], "RMSE")
    fit_seconds = _positive_float(result["fit_seconds"], "fit seconds")
    _nonnegative_float(result["warmup_seconds"], "warmup seconds")
    wall_seconds = _positive_float(result["wall_seconds"], "wall seconds")
    _validate_prediction_timing(
        result["prediction_timing"],
        prediction_timing_policy,
    )
    _validate_source_attestation(result["source_attestation"])
    prediction_seconds = float(result["prediction_timing"]["total_seconds"])
    if wall_seconds + max(1e-15, 1e-12 * wall_seconds) < (
        fit_seconds + prediction_seconds
    ):
        raise RuntimeError(
            "panel-3 wall time is shorter than fitted and timed work"
        )
    if result["metadata"].get("kind") != arm:
        raise RuntimeError("panel-3 fitted metadata arm changed")
    if result["worker_stdout"] is not None and not isinstance(
        result["worker_stdout"], str
    ):
        raise RuntimeError("panel-3 worker stdout is invalid")
    if result["worker_stderr"] is not None and not isinstance(
        result["worker_stderr"], str
    ):
        raise RuntimeError("panel-3 worker stderr is invalid")
    behavior = {
        "coordinate": full_coordinate,
        "arm": arm,
        "rmse": result["rmse"],
        "prediction_sha256": result["prediction_sha256"],
        "metadata": result["metadata"],
        "source_attestation": result["source_attestation"],
    }
    if result["behavior_fingerprint_sha256"] != _json_sha256(behavior):
        raise RuntimeError("panel-3 behavior fingerprint changed")


def _validate_comparator_failure(value: Any) -> None:
    fields = {
        "worker_key",
        "task_id",
        "coordinate",
        "arm",
        "status",
        "failure_kind",
        "returncode",
        "worker_stdout",
        "worker_stderr",
        "message",
        "failure_fingerprint_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise RuntimeError("panel-3 comparator-failure fields changed")
    coordinate = _result_coordinate(value)
    full_coordinate = {
        "task_id": coordinate[0],
        "repeat": coordinate[1],
        "fold": coordinate[2],
        "sample": coordinate[3],
    }
    if (
        value["arm"] not in runner.COMPARATOR_ARMS
        or value["status"] != "failed"
        or value["worker_key"]
        != runner._worker_key(full_coordinate, value["arm"])
        or value["failure_kind"]
        not in {
            "worker_launch_failure",
            "worker_process_failure",
            "worker_protocol_failure",
        }
        or (
            value["returncode"] is not None
            and type(value["returncode"]) is not int
        )
        or not isinstance(value["message"], str)
        or not value["message"]
        or any(
            item is not None and not isinstance(item, str)
            for item in (value["worker_stdout"], value["worker_stderr"])
        )
        or not _is_sha256(value["failure_fingerprint_sha256"])
    ):
        raise RuntimeError("panel-3 comparator-failure record is invalid")
    unhashed = dict(value)
    observed = unhashed.pop("failure_fingerprint_sha256")
    if observed != _json_sha256(unhashed):
        raise RuntimeError(
            "panel-3 comparator-failure fingerprint changed"
        )


def _validate_darkofit_fit_metadata(value: Any, label: str) -> None:
    if (
        not isinstance(value, dict)
        or type(value.get("best_iteration")) is not int
        or value["best_iteration"] <= 0
        or type(value.get("fitted_tree_count")) is not int
        or value["fitted_tree_count"] <= 0
        or isinstance(value.get("resolved_learning_rate"), bool)
        or not isinstance(value.get("resolved_learning_rate"), (int, float))
        or not math.isfinite(float(value["resolved_learning_rate"]))
        or float(value["resolved_learning_rate"]) <= 0.0
        or not isinstance(value.get("selected_tree_mode"), str)
        or not value["selected_tree_mode"]
        or not isinstance(value.get("selected_lane"), str)
        or not value["selected_lane"]
        or not isinstance(value.get("final_fit"), dict)
        or value["final_fit"].get("stop_reason")
        not in {
            "early_stopping",
            "iteration_limit",
            "no_split",
            "selection_round_limit",
            "time_limit",
        }
    ):
        raise RuntimeError(
            f"panel-3 {label} fitted metadata is incomplete"
        )


def _validate_selection_fit_metadata(value: Any, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"panel-3 {label} selection metadata is incomplete")
    for index, record in enumerate(value):
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("name"), str)
            or not record["name"]
            or "fit_metadata" not in record
        ):
            raise RuntimeError(
                f"panel-3 {label} selection metadata is incomplete"
            )
        _validate_darkofit_fit_metadata(
            record["fit_metadata"],
            f"{label} selection fit {index}",
        )


def _selected_fit_record(
    metadata: dict[str, Any],
    expected_name: str,
    label: str,
) -> dict[str, Any]:
    matches = [
        record
        for record in metadata["selection_fits"]
        if record.get("name") == expected_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"panel-3 {label} selected fit metadata is incoherent"
        )
    return matches[0]


def _validate_selected_fit_coherence(
    *,
    metadata: dict[str, Any],
    selected_record: dict[str, Any],
    final_fit: dict[str, Any],
    selected_tree_mode: str,
    selected_lane: str,
    label: str,
) -> None:
    selected = selected_record["fit_metadata"]
    selected_best = metadata["selected_best_iteration"]
    selected_lr = float(metadata["selected_resolved_learning_rate"])
    if (
        selected["best_iteration"] != selected_best
        or float(selected["resolved_learning_rate"]) != selected_lr
        or selected["selected_tree_mode"] != selected_tree_mode
        or selected["selected_lane"] != selected_lane
        or float(final_fit["resolved_learning_rate"]) != selected_lr
        or final_fit.get(
            "requested_tree_mode",
            final_fit.get("selected_tree_mode"),
        )
        != selected_tree_mode
        or final_fit["selected_tree_mode"] != selected_tree_mode
        or final_fit["selected_lane"] != selected_lane
        or final_fit["best_iteration"] > selected_best
        or final_fit["fitted_tree_count"] > selected_best
    ):
        raise RuntimeError(
            f"panel-3 {label} selected/final fit metadata is incoherent"
        )


def _validate_fitted_metadata(
    result: dict[str, Any],
    *,
    strict: bool = False,
) -> None:
    metadata = result["metadata"]
    arm = result["arm"]
    if arm == runner.CONTROL_ARM:
        if (
            metadata.get("engaged") is not False
            or metadata.get("selected_configuration") != "product_default"
        ):
            raise RuntimeError(
                "panel-3 current-default metadata is incomplete"
            )
        _validate_darkofit_fit_metadata(
            metadata.get("final_fit"), "current-default"
        )
    elif arm == "t5_composite_policy":
        if type(metadata.get("engaged")) is not bool:
            raise RuntimeError("panel-3 T5 engagement ledger is invalid")
        decline_reason = metadata.get("decline_reason")
        if (
            metadata["engaged"]
            and decline_reason is not None
        ) or (
            not metadata["engaged"]
            and decline_reason
            not in {"below_size_gate", "outer_validation_guard"}
        ):
            raise RuntimeError("panel-3 T5 engagement ledger is invalid")
        final = metadata.get("final_fit")
        _validate_darkofit_fit_metadata(final, "T5")
        # Size-gate declines expose the best iteration and LR inside the
        # product-default final-fit record; engaged/guarded paths additionally
        # expose the selected audition values at the policy level.
        if metadata["engaged"] and (
            type(metadata.get("selected_best_iteration")) is not int
            or metadata["selected_best_iteration"] <= 0
            or isinstance(
                metadata.get("selected_resolved_learning_rate"),
                bool,
            )
            or not isinstance(
                metadata.get("selected_resolved_learning_rate"),
                (int, float),
            )
            or not math.isfinite(
                float(metadata["selected_resolved_learning_rate"])
            )
            or float(metadata["selected_resolved_learning_rate"]) <= 0.0
        ):
            raise RuntimeError("panel-3 T5 selection metadata is incomplete")
        if decline_reason != "below_size_gate":
            _validate_selection_fit_metadata(
                metadata.get("selection_fits"),
                "T5",
            )
        if metadata["engaged"]:
            if metadata.get("selected_configuration") != "challenger":
                raise RuntimeError(
                    "panel-3 T5 selected/final fit metadata is incoherent"
                )
            selected_crosses = metadata.get("selected_crosses")
            selected_linear = metadata.get("selected_linear_leaves")
            selected_mode = metadata.get("selected_tree_mode")
            if (
                type(selected_crosses) is not bool
                or type(selected_linear) is not bool
                or not isinstance(selected_mode, str)
                or not selected_mode
            ):
                raise RuntimeError(
                    "panel-3 T5 selected/final fit metadata is incoherent"
                )
            selected_name = (
                "challenger_crossed"
                if selected_crosses
                else (
                    "challenger_catboost_linear"
                    if selected_linear
                    else "challenger_auto"
                )
            )
            selected_record = _selected_fit_record(
                metadata,
                selected_name,
                "T5",
            )
            _validate_selected_fit_coherence(
                metadata=metadata,
                selected_record=selected_record,
                final_fit=final,
                selected_tree_mode=selected_mode,
                selected_lane=(
                    "linear_leaves" if selected_linear else "boosting"
                ),
                label="T5",
            )
    elif arm == "guarded_cross_features_policy":
        required = {
            "engaged",
            "selected_linear_leaves",
            "selected_crosses",
            "candidate_cross_pairs",
            "selected_cross_pairs",
            "selected_best_iteration",
            "selected_resolved_learning_rate",
            "selection_fits",
            "final_refit_parameters",
            "final_fit",
        }
        if not required <= set(metadata):
            raise RuntimeError(
                "panel-3 guarded-cross metadata is incomplete"
            )
        if (
            type(metadata["engaged"]) is not bool
            or type(metadata["selected_crosses"]) is not bool
            or metadata["engaged"] != metadata["selected_crosses"]
            or metadata.get("decline_reason")
            != (None if metadata["engaged"] else "cross_guard")
            or type(metadata["selected_best_iteration"]) is not int
            or metadata["selected_best_iteration"] <= 0
            or isinstance(
                metadata["selected_resolved_learning_rate"],
                bool,
            )
            or not isinstance(
                metadata["selected_resolved_learning_rate"],
                (int, float),
            )
            or not math.isfinite(
                float(metadata["selected_resolved_learning_rate"])
            )
            or float(metadata["selected_resolved_learning_rate"]) <= 0.0
        ):
            raise RuntimeError(
                "panel-3 guarded-cross selection ledger is invalid"
            )
        _validate_selection_fit_metadata(
            metadata["selection_fits"],
            "guarded-cross",
        )
        _validate_darkofit_fit_metadata(
            metadata["final_fit"], "guarded-cross"
        )
        selected_name = (
            "crossed_selected_leaf_lane"
            if metadata["selected_crosses"]
            else (
                "uncrossed_linear"
                if metadata["selected_linear_leaves"]
                else "uncrossed_constant"
            )
        )
        selected_record = _selected_fit_record(
            metadata,
            selected_name,
            "guarded-cross",
        )
        parameters = metadata["final_refit_parameters"]
        if (
            metadata.get("selected_selection_fit") != selected_record
            or not isinstance(parameters, dict)
            or set(parameters)
            != {
                "iterations",
                "learning_rate",
                "tree_mode",
                "linear_leaves",
                "crossed",
            }
            or parameters["iterations"]
            != metadata["selected_best_iteration"]
            or float(parameters["learning_rate"])
            != float(metadata["selected_resolved_learning_rate"])
            or parameters["tree_mode"] != "catboost"
            or parameters["linear_leaves"]
            is not metadata["selected_linear_leaves"]
            or parameters["crossed"] is not metadata["selected_crosses"]
        ):
            raise RuntimeError(
                "panel-3 guarded-cross selected/final fit metadata "
                "is incoherent"
            )
        _validate_selected_fit_coherence(
            metadata=metadata,
            selected_record=selected_record,
            final_fit=metadata["final_fit"],
            selected_tree_mode="catboost",
            selected_lane=(
                "linear_leaves"
                if metadata["selected_linear_leaves"]
                else "boosting"
            ),
            label="guarded-cross",
        )
    elif arm == "chimeraboost_0_15_0":
        required = {
            "requested_iterations",
            "attempted_iterations",
            "best_iteration",
            "fitted_tree_count",
            "resolved_learning_rate",
            "selected_mode",
            "selected_lane",
            "stop_reason",
            "linear_leaves_selected",
            "cross_features_selected",
            "cross_pairs",
        }
        if (
            not required <= set(metadata)
            or type(metadata["best_iteration"]) is not int
            or metadata["best_iteration"] < 0
            or type(metadata["fitted_tree_count"]) is not int
            or metadata["fitted_tree_count"] < 0
            or isinstance(metadata["resolved_learning_rate"], bool)
            or not isinstance(metadata["resolved_learning_rate"], (int, float))
            or not math.isfinite(float(metadata["resolved_learning_rate"]))
            or float(metadata["resolved_learning_rate"]) <= 0.0
            or not isinstance(metadata["selected_mode"], str)
            or not metadata["selected_mode"]
            or not isinstance(metadata["selected_lane"], str)
            or not metadata["selected_lane"]
            or metadata["stop_reason"]
            not in {
                "early_stopping",
                "iteration_limit",
                "no_legal_split",
                "no_legal_split_or_internal_selection",
            }
        ):
            raise RuntimeError(
                "panel-3 ChimeraBoost fitted metadata is incomplete"
            )
    elif arm == "catboost_product_default":
        required = {
            "requested_iterations",
            "attempted_iterations",
            "best_iteration",
            "fitted_tree_count",
            "resolved_learning_rate",
            "selected_mode",
            "selected_lane",
            "stop_reason",
            "external_categorical_transform_included_in_fit_timing",
            "external_categorical_transform_included_in_predict_timing",
        }
        if (
            not required <= set(metadata)
            or type(metadata["best_iteration"]) is not int
            or metadata["best_iteration"] < -1
            or type(metadata["fitted_tree_count"]) is not int
            or metadata["fitted_tree_count"] < 0
            or isinstance(metadata["resolved_learning_rate"], bool)
            or not isinstance(metadata["resolved_learning_rate"], (int, float))
            or not math.isfinite(float(metadata["resolved_learning_rate"]))
            or float(metadata["resolved_learning_rate"]) <= 0.0
            or not isinstance(metadata["selected_mode"], str)
            or not metadata["selected_mode"]
            or not isinstance(metadata["selected_lane"], str)
            or not metadata["selected_lane"]
            or metadata["stop_reason"]
            not in {
                "early_stopping",
                "iteration_limit",
                "no_legal_split_or_other",
            }
            or metadata[
                "external_categorical_transform_included_in_fit_timing"
            ]
            is not True
            or metadata[
                "external_categorical_transform_included_in_predict_timing"
            ]
            is not True
        ):
            raise RuntimeError(
                "panel-3 CatBoost fitted metadata is incomplete"
            )
    else:
        raise RuntimeError("panel-3 fitted metadata arm is unknown")
    if strict:
        _validate_fitted_metadata_strict(result)


def _close(left: float, right: float) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=1e-12,
        abs_tol=1e-15,
    )


def _validate_cross_pairs(
    value: Any,
    *,
    feature_count: int,
    categorical_indices: tuple[int, ...] = (),
    label: str,
) -> list[tuple[int, int, str]]:
    if not isinstance(value, list):
        raise RuntimeError(f"panel-3 {label} cross-pair ledger changed")
    pairs = []
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != 3
            or type(row[0]) is not int
            or type(row[1]) is not int
            or not 0 <= row[0] < feature_count
            or not 0 <= row[1] < feature_count
            or row[0] == row[1]
            or row[0] in categorical_indices
            or row[1] in categorical_indices
            or row[2] not in {"diff", "prod"}
        ):
            raise RuntimeError(
                f"panel-3 {label} cross-pair ledger changed"
            )
        pairs.append((row[0], row[1], row[2]))
    if len(pairs) != len(set(pairs)):
        raise RuntimeError(f"panel-3 {label} cross-pair ledger changed")
    return pairs


def _strict_selection_records(
    metadata: dict[str, Any],
    *,
    required_names: set[str],
    allowed_names: set[str],
    label: str,
) -> dict[str, dict[str, Any]]:
    records = metadata.get("selection_fits")
    _validate_selection_fit_metadata(records, label)
    by_name = {
        record["name"]: record
        for record in records
        if isinstance(record, dict)
    }
    if (
        len(by_name) != len(records)
        or not required_names <= set(by_name)
        or not set(by_name) <= allowed_names
    ):
        raise RuntimeError(
            f"panel-3 {label} selection-fit names changed"
        )
    base_fields = {
        "name",
        "validation_rmse",
        "fit_seconds",
        "fit_metadata",
        "validation",
    }
    for name, record in by_name.items():
        extras = set(record) - base_fields
        if name == "challenger_crossed":
            expected_extras = {
                "pairs",
                "pair_count",
                "transform_seconds",
            }
        elif name == "crossed_selected_leaf_lane":
            expected_extras = {"pairs", "transform_seconds"}
        else:
            expected_extras = extras & {"tree_mode_selection"}
        if (
            set(record) < base_fields
            or extras != expected_extras
        ):
            raise RuntimeError(
                f"panel-3 {label} selection-fit fields changed"
            )
        _positive_float(
            record["validation_rmse"],
            f"{label} validation RMSE",
        )
        _positive_float(record["fit_seconds"], f"{label} selection time")
        validation = record["validation"]
        if (
            not isinstance(validation, dict)
            or validation.get("source") != "explicit_eval_set"
        ):
            raise RuntimeError(
                f"panel-3 {label} selection validation changed"
            )
    selection_total = sum(
        float(record["fit_seconds"]) for record in records
    )
    if not _close(
        selection_total,
        metadata["total_selection_fit_seconds"],
    ):
        raise RuntimeError(
            f"panel-3 {label} selection-time ledger changed"
        )
    return by_name


def _validate_inner_split(
    split: Any,
    *,
    outer_rows: int,
    label: str,
) -> None:
    fields = {
        "policy",
        "random_state",
        "validation_fraction",
        "train_rows",
        "validation_rows",
        "train_positions_sha256",
        "validation_positions_sha256",
    }
    if (
        not isinstance(split, dict)
        or set(split) != fields
        or split["policy"] != "ShuffleSplit"
        or split["random_state"] != 4
        or split["validation_fraction"] != 0.2
        or type(split["train_rows"]) is not int
        or type(split["validation_rows"]) is not int
        or split["train_rows"] <= 0
        or split["validation_rows"] <= 0
        or split["train_rows"] + split["validation_rows"] != outer_rows
        or not _is_sha256(split["train_positions_sha256"])
        or not _is_sha256(split["validation_positions_sha256"])
    ):
        raise RuntimeError(f"panel-3 {label} selection split changed")


def _validate_policy_time(
    result: dict[str, Any],
    *,
    selection_seconds: float,
    final_seconds: float,
    overhead_seconds: Any,
    label: str,
) -> None:
    overhead = _nonnegative_float(
        overhead_seconds,
        f"{label} policy overhead",
    )
    if not _close(
        result["fit_seconds"],
        selection_seconds + final_seconds + overhead,
    ):
        raise RuntimeError(f"panel-3 {label} fit-time ledger changed")


def _validate_fitted_metadata_strict(result: dict[str, Any]) -> None:
    metadata = result["metadata"]
    arm = result["arm"]
    if arm == runner.CONTROL_ARM:
        expected = {
            "kind",
            "engaged",
            "selected_configuration",
            "final_fit",
        }
        if set(metadata) != expected:
            raise RuntimeError("panel-3 current-default fields changed")
        return
    if arm == "t5_composite_policy":
        if metadata["decline_reason"] == "below_size_gate":
            expected = {
                "kind",
                "engaged",
                "selected_configuration",
                "decline_reason",
                "size_gate",
                "total_selection_fit_seconds",
                "policy_overhead_seconds",
                "final_fit_seconds",
                "final_fit",
            }
            if set(metadata) != expected:
                raise RuntimeError("panel-3 T5 size-gate fields changed")
            final_seconds = _positive_float(
                metadata["final_fit_seconds"],
                "T5 final fit time",
            )
            _validate_policy_time(
                result,
                selection_seconds=0.0,
                final_seconds=final_seconds,
                overhead_seconds=metadata["policy_overhead_seconds"],
                label="T5",
            )
            return
        expected = {
            "kind",
            "engaged",
            "decline_reason",
            "size_gate",
            "split",
            "outer_guard_ratio",
            "cross_guard_ratio",
            "selection_rounds",
            "control_validation_rmse",
            "challenger_validation_rmse",
            "relative_challenger_validation_ratio",
            "selected_configuration",
            "selected_tree_mode",
            "selected_linear_leaves",
            "selected_crosses",
            "selected_cross_pairs",
            "selected_cross_pair_count",
            "selected_best_iteration",
            "selected_resolved_learning_rate",
            "selection_fits",
            "total_selection_fit_seconds",
            "policy_overhead_seconds",
            "final_transform_seconds",
            "final_fit_seconds",
            "final_fit",
        }
        if (
            set(metadata) != expected
            or type(metadata["selected_linear_leaves"]) is not bool
            or type(metadata["selected_crosses"]) is not bool
            or metadata["outer_guard_ratio"] != 0.995
            or metadata["cross_guard_ratio"] != 0.95
            or metadata["selection_rounds"] != 100
        ):
            raise RuntimeError("panel-3 T5 producer fields changed")
        _validate_inner_split(
            metadata["split"],
            outer_rows=result["train_rows"],
            label="T5",
        )
        records = _strict_selection_records(
            metadata,
            required_names={"control_audition", "challenger_auto"},
            allowed_names={
                "control_audition",
                "challenger_auto",
                "challenger_catboost_linear",
                "challenger_crossed",
            },
            label="T5",
        )
        control_record = records["control_audition"]
        auto_record = records["challenger_auto"]
        auto_mode = auto_record["fit_metadata"]["selected_tree_mode"]
        linear_required = auto_mode == "catboost"
        expected_names = {"control_audition", "challenger_auto"}
        if linear_required:
            expected_names.add("challenger_catboost_linear")
        crossed_record = records.get("challenger_crossed")
        if crossed_record is not None:
            expected_names.add("challenger_crossed")
        if (
            set(records) != expected_names
            or metadata["selected_tree_mode"] != auto_mode
        ):
            raise RuntimeError("panel-3 T5 selection-fit names changed")
        linear_record = records.get("challenger_catboost_linear")
        selected_linear = (
            linear_record is not None
            and float(linear_record["validation_rmse"])
            < float(auto_record["validation_rmse"])
        )
        selected_uncrossed = (
            linear_record if selected_linear else auto_record
        )
        if metadata["selected_linear_leaves"] is not selected_linear:
            raise RuntimeError("panel-3 T5 selected lane changed")
        cross_pairs: list[tuple[int, int, str]] = []
        if crossed_record is not None:
            cross_pairs = _validate_cross_pairs(
                crossed_record["pairs"],
                feature_count=result["feature_policy"][
                    "retained_feature_count"
                ],
                categorical_indices=tuple(
                    result["categorical_feature_indices"]
                ),
                label="T5 audition",
            )
            transform_seconds = _nonnegative_float(
                crossed_record["transform_seconds"],
                "T5 cross transform time",
            )
            if (
                not cross_pairs
                or crossed_record["pair_count"] != len(cross_pairs)
                or float(crossed_record["fit_seconds"])
                + 1e-15
                < transform_seconds
                or crossed_record["fit_metadata"][
                    "selected_tree_mode"
                ]
                != auto_mode
                or crossed_record["fit_metadata"]["selected_lane"]
                != (
                    "linear_leaves" if selected_linear else "boosting"
                )
            ):
                raise RuntimeError("panel-3 T5 cross record changed")
        selected_crosses = (
            crossed_record is not None
            and float(crossed_record["validation_rmse"])
            <= 0.95
            * float(selected_uncrossed["validation_rmse"])
        )
        selected_record = (
            crossed_record if selected_crosses else selected_uncrossed
        )
        control = _positive_float(
            metadata["control_validation_rmse"],
            "T5 control validation RMSE",
        )
        challenger = _positive_float(
            metadata["challenger_validation_rmse"],
            "T5 challenger validation RMSE",
        )
        ratio = _positive_float(
            metadata["relative_challenger_validation_ratio"],
            "T5 validation ratio",
        )
        if (
            not _close(
                control,
                control_record["validation_rmse"],
            )
            or not _close(
                challenger,
                selected_record["validation_rmse"],
            )
            or not _close(ratio, challenger / control)
            or metadata["engaged"] != (ratio <= 0.995)
            or metadata["decline_reason"]
            != (
                None
                if metadata["engaged"]
                else "outer_validation_guard"
            )
            or metadata["selected_configuration"]
            != ("challenger" if metadata["engaged"] else "product_default")
        ):
            raise RuntimeError("panel-3 T5 outer guard changed")
        pairs = _validate_cross_pairs(
            metadata["selected_cross_pairs"],
            feature_count=result["feature_policy"]["retained_feature_count"],
            categorical_indices=tuple(
                result["categorical_feature_indices"]
            ),
            label="T5 selected",
        )
        if (
            metadata["selected_crosses"] is not selected_crosses
            or metadata["selected_cross_pair_count"] != len(pairs)
            or pairs != (cross_pairs if selected_crosses else [])
            or metadata["selected_best_iteration"]
            != selected_record["fit_metadata"]["best_iteration"]
            or not _close(
                metadata["selected_resolved_learning_rate"],
                selected_record["fit_metadata"][
                    "resolved_learning_rate"
                ],
            )
        ):
            raise RuntimeError("panel-3 T5 cross guard changed")
        final_seconds = _positive_float(
            metadata["final_fit_seconds"],
            "T5 final fit time",
        )
        final_transform = _nonnegative_float(
            metadata["final_transform_seconds"],
            "T5 final transform time",
        )
        if (
            not (metadata["engaged"] and selected_crosses)
            and final_transform != 0.0
        ):
            raise RuntimeError("panel-3 T5 final transform ledger changed")
        _validate_policy_time(
            result,
            selection_seconds=float(
                metadata["total_selection_fit_seconds"]
            ),
            final_seconds=final_seconds,
            overhead_seconds=metadata["policy_overhead_seconds"],
            label="T5",
        )
        return
    if arm == "guarded_cross_features_policy":
        expected = {
            "kind",
            "engaged",
            "decline_reason",
            "split",
            "cross_guard_ratio",
            "selected_configuration",
            "selected_linear_leaves",
            "selected_crosses",
            "candidate_cross_pairs",
            "selected_cross_pairs",
            "selected_cross_pair_count",
            "uncrossed_validation_rmse",
            "crossed_validation_rmse",
            "relative_crossed_validation_ratio",
            "selected_best_iteration",
            "selected_resolved_learning_rate",
            "selected_selection_fit",
            "selection_fits",
            "total_selection_fit_seconds",
            "policy_overhead_seconds",
            "final_transform_seconds",
            "final_model_fit_seconds",
            "final_fit_seconds",
            "final_refit_parameters",
            "final_fit",
        }
        if (
            set(metadata) != expected
            or type(metadata["selected_linear_leaves"]) is not bool
            or metadata["cross_guard_ratio"] != 0.95
        ):
            raise RuntimeError(
                "panel-3 guarded-cross producer fields changed"
            )
        _validate_inner_split(
            metadata["split"],
            outer_rows=result["train_rows"],
            label="guarded-cross",
        )
        candidate_pairs = _validate_cross_pairs(
            metadata["candidate_cross_pairs"],
            feature_count=result["feature_policy"]["retained_feature_count"],
            categorical_indices=tuple(
                result["categorical_feature_indices"]
            ),
            label="guarded candidate",
        )
        selected_pairs = _validate_cross_pairs(
            metadata["selected_cross_pairs"],
            feature_count=result["feature_policy"]["retained_feature_count"],
            categorical_indices=tuple(
                result["categorical_feature_indices"]
            ),
            label="guarded selected",
        )
        required_names = {"uncrossed_constant", "uncrossed_linear"}
        if candidate_pairs:
            required_names.add("crossed_selected_leaf_lane")
        records = _strict_selection_records(
            metadata,
            required_names=required_names,
            allowed_names={
                "uncrossed_constant",
                "uncrossed_linear",
                "crossed_selected_leaf_lane",
            },
            label="guarded-cross",
        )
        if set(records) != required_names:
            raise RuntimeError(
                "panel-3 guarded-cross selection-fit names changed"
            )
        constant_record = records["uncrossed_constant"]
        linear_record = records["uncrossed_linear"]
        selected_linear = (
            float(linear_record["validation_rmse"])
            < float(constant_record["validation_rmse"])
        )
        uncrossed_record = (
            linear_record if selected_linear else constant_record
        )
        if metadata["selected_linear_leaves"] is not selected_linear:
            raise RuntimeError(
                "panel-3 guarded-cross selected lane changed"
            )
        uncrossed = _positive_float(
            metadata["uncrossed_validation_rmse"],
            "guarded uncrossed validation RMSE",
        )
        if not _close(
            uncrossed,
            uncrossed_record["validation_rmse"],
        ):
            raise RuntimeError(
                "panel-3 guarded-cross uncrossed score changed"
            )
        crossed = metadata["crossed_validation_rmse"]
        ratio = metadata["relative_crossed_validation_ratio"]
        crossed_record = records.get("crossed_selected_leaf_lane")
        if candidate_pairs:
            record_pairs = _validate_cross_pairs(
                crossed_record["pairs"],
                feature_count=result["feature_policy"][
                    "retained_feature_count"
                ],
                categorical_indices=tuple(
                    result["categorical_feature_indices"]
                ),
                label="guarded audition",
            )
            transform_seconds = _nonnegative_float(
                crossed_record["transform_seconds"],
                "guarded cross transform time",
            )
            crossed_value = _positive_float(
                crossed,
                "guarded crossed validation RMSE",
            )
            ratio_value = _positive_float(
                ratio,
                "guarded validation ratio",
            )
            expected_engaged = ratio_value <= 0.95
            ratio_ok = (
                record_pairs == candidate_pairs
                and _close(
                    crossed_value,
                    crossed_record["validation_rmse"],
                )
                and _close(
                    ratio_value,
                    crossed_value / uncrossed,
                )
                and float(crossed_record["fit_seconds"]) + 1e-15
                >= transform_seconds
                and crossed_record["fit_metadata"][
                    "selected_tree_mode"
                ]
                == "catboost"
                and crossed_record["fit_metadata"]["selected_lane"]
                == ("linear_leaves" if selected_linear else "boosting")
            )
        else:
            expected_engaged = False
            ratio_ok = (
                crossed is None
                and ratio is None
                and crossed_record is None
            )
        selected_record = (
            crossed_record if expected_engaged else uncrossed_record
        )
        if (
            not ratio_ok
            or metadata["engaged"] is not expected_engaged
            or metadata["selected_crosses"] is not expected_engaged
            or metadata["decline_reason"]
            != (None if expected_engaged else "cross_guard")
            or metadata["selected_configuration"]
            != ("crossed" if expected_engaged else "uncrossed")
            or selected_pairs
            != (candidate_pairs if expected_engaged else [])
            or metadata["selected_cross_pair_count"] != len(selected_pairs)
            or metadata["selected_selection_fit"] != selected_record
            or metadata["selected_best_iteration"]
            != selected_record["fit_metadata"]["best_iteration"]
            or not _close(
                metadata["selected_resolved_learning_rate"],
                selected_record["fit_metadata"][
                    "resolved_learning_rate"
                ],
            )
        ):
            raise RuntimeError("panel-3 guarded cross guard changed")
        final_transform = _nonnegative_float(
            metadata["final_transform_seconds"],
            "guarded final transform time",
        )
        final_model = _positive_float(
            metadata["final_model_fit_seconds"],
            "guarded final model time",
        )
        final_total = _positive_float(
            metadata["final_fit_seconds"],
            "guarded final fit time",
        )
        if not _close(final_total, final_transform + final_model):
            raise RuntimeError(
                "panel-3 guarded final fit-time ledger changed"
            )
        _validate_policy_time(
            result,
            selection_seconds=float(
                metadata["total_selection_fit_seconds"]
            ),
            final_seconds=final_total,
            overhead_seconds=metadata["policy_overhead_seconds"],
            label="guarded-cross",
        )
        return
    if arm == "chimeraboost_0_15_0":
        expected = {
            "kind",
            "requested_iterations",
            "attempted_iterations",
            "best_iteration",
            "fitted_tree_count",
            "resolved_learning_rate",
            "selected_mode",
            "selected_lane",
            "stop_reason",
            "early_stopping",
            "selection_rounds",
            "linear_leaves_selected",
            "cross_features_selected",
            "cross_pairs",
        }
    else:
        expected = {
            "kind",
            "requested_iterations",
            "attempted_iterations",
            "best_iteration",
            "fitted_tree_count",
            "resolved_learning_rate",
            "selected_mode",
            "selected_lane",
            "stop_reason",
            "external_categorical_transform_included_in_fit_timing",
            "external_categorical_transform_included_in_predict_timing",
        }
    if set(metadata) != expected:
        raise RuntimeError(f"panel-3 {arm} producer fields changed")


def validate_raw(
    raw: dict[str, Any],
    registry: dict[str, Any],
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    registry_file_sha256: str | None = None,
    verify_current_files: bool = True,
) -> dict[tuple[int, int, int, int, str], dict[str, Any]]:
    """Validate the raw/derived boundary without computing a gate."""
    if not isinstance(raw, dict):
        raise RuntimeError("panel-3 raw artifact must be an object")
    if set(raw) != RAW_FIELDS:
        raise RuntimeError("panel-3 raw artifact fields changed")
    expected_hash = raw.get("raw_artifact_sha256")
    unhashed = dict(raw)
    unhashed.pop("raw_artifact_sha256", None)
    if expected_hash != _json_sha256(unhashed):
        raise RuntimeError("panel-3 raw artifact hash changed")
    if verify_current_files:
        runner.validate_registry(registry, registry_path=registry_path)
    else:
        runner.validate_registry_historical(
            registry,
            registry_path=registry_path,
        )
    arm_order = runner._arm_order(registry)
    candidate_arms = runner._candidate_arms(registry)
    decision_arms = runner._decision_arms(registry)
    coordinate_count = len(registry["coordinates"])
    worker_count = coordinate_count * len(arm_order)
    decision_worker_count = coordinate_count * len(decision_arms)
    comparator_worker_count = (
        coordinate_count * len(runner.COMPARATOR_ARMS)
    )
    registry_record = raw.get("registry")
    if registry_file_sha256 is None:
        registry_file_sha256 = (
            registry_record.get("file_sha256")
            if not verify_current_files
            and isinstance(registry_record, dict)
            else common.sha256_file(registry_path)
        )
    if not _is_sha256(registry_file_sha256):
        raise RuntimeError("panel-3 registry file hash is invalid")
    if (
        raw.get("schema_version") != 1
        or raw.get("name") != "darkofit_panel3_confirmation_raw_v1"
        or not isinstance(registry_record, dict)
        or set(registry_record)
        != {"path", "file_sha256", "canonical_sha256"}
        or not isinstance(raw.get("created_at"), str)
        or registry_record.get("path")
        != runner._artifact_path(registry_path)
        or registry_record.get("canonical_sha256")
        != registry["registry_sha256"]
        or registry_record.get("file_sha256")
        != registry_file_sha256
        or raw.get("outcomes_scored") is not True
        or raw.get("analysis_performed") is not False
        or raw.get("default_promotion_authorized") is not False
        or raw.get("task_imputation_used") is not False
        or raw.get("task_drop_used") is not False
        or not isinstance(raw.get("protocol_deviations"), list)
    ):
        raise RuntimeError("panel-3 raw boundary is invalid")
    plan = raw.get("execution_plan")
    if not isinstance(plan, dict):
        raise RuntimeError("panel-3 execution plan is missing")
    common.verify_artifact_sha256(plan, "execution_plan_sha256")
    expected_plan = runner.execution_plan(
        registry,
        registry_path=registry_path,
        registry_file_sha256=registry_file_sha256,
        validate_registry_boundary=False,
    )
    if plan != expected_plan:
        raise RuntimeError("panel-3 execution plan changed")
    protocol = raw.get("protocol")
    if (
        not isinstance(protocol, dict)
        or set(protocol) != PROTOCOL_FIELDS
        or protocol.get("arms") != list(arm_order)
        or protocol.get("path") != runner._artifact_path(common.PROTOCOL)
        or protocol.get("runner_path")
        != runner._artifact_path(Path(runner.__file__).resolve())
        or protocol.get("analyzer_path")
        != runner._artifact_path(Path(__file__).resolve())
        or protocol.get("candidate_contract_path")
        != runner._artifact_path(common.CANDIDATE_CONTRACT)
        or protocol.get("power_design_decision_path")
        != runner._artifact_path(common.POWER_DESIGN_DECISION)
        or protocol.get("power_design_file_sha256")
        != registry["power_design_file_sha256"]
        or protocol.get("power_design_decision_sha256")
        != registry["power_design_decision_sha256"]
        or protocol.get("coordinate_count") != 36
        or protocol.get("worker_count") != worker_count
        or protocol.get("decision_worker_count") != decision_worker_count
        or type(protocol.get("successful_worker_count")) is not int
        or not decision_worker_count
        <= protocol["successful_worker_count"]
        <= worker_count
        or type(protocol.get("comparator_failure_count")) is not int
        or not 0
        <= protocol["comparator_failure_count"]
        <= comparator_worker_count
        or protocol["successful_worker_count"]
        + protocol["comparator_failure_count"]
        != worker_count
        or protocol.get("comparator_failures_affect_candidate_gates")
        is not False
        or type(protocol.get("threads_per_worker")) is not int
        or protocol["threads_per_worker"] <= 0
        or type(protocol.get("concurrent_workers")) is not int
        or protocol["concurrent_workers"] <= 0
        or protocol.get("task_drop_allowed") is not False
        or protocol.get("task_imputation_allowed") is not False
        or protocol.get("outcome_dependent_rerun_allowed") is not False
        or isinstance(protocol.get("validation_fraction"), bool)
        or not isinstance(protocol["validation_fraction"], (int, float))
        or not 0.0 < float(protocol["validation_fraction"]) < 1.0
        or isinstance(protocol.get("guarded_cross_ratio"), bool)
        or not isinstance(protocol["guarded_cross_ratio"], (int, float))
        or not 0.0 < float(protocol["guarded_cross_ratio"]) <= 1.0
        or isinstance(protocol.get("prediction_block_seconds"), bool)
        or not isinstance(
            protocol["prediction_block_seconds"], (int, float)
        )
        or not math.isfinite(float(protocol["prediction_block_seconds"]))
        or float(protocol["prediction_block_seconds"]) <= 0.0
        or type(protocol.get("prediction_min_calls")) is not int
        or protocol["prediction_min_calls"] <= 0
        or type(protocol.get("prediction_max_calls")) is not int
        or protocol["prediction_max_calls"]
        < protocol["prediction_min_calls"]
    ):
        raise RuntimeError("panel-3 raw protocol ledger changed")
    if verify_current_files:
        if {
            "threads_per_worker": protocol["threads_per_worker"],
            "concurrent_workers": protocol["concurrent_workers"],
            "validation_fraction": protocol["validation_fraction"],
            "guarded_cross_ratio": protocol["guarded_cross_ratio"],
            "prediction_block_seconds": protocol[
                "prediction_block_seconds"
            ],
            "prediction_min_calls": protocol["prediction_min_calls"],
            "prediction_max_calls": protocol["prediction_max_calls"],
        } != {
            "threads_per_worker": runner.THREADS_PER_WORKER,
            "concurrent_workers": runner.CONCURRENT_WORKERS,
            "validation_fraction": runner.VALIDATION_FRACTION,
            "guarded_cross_ratio": runner.GUARDED_CROSS_RATIO,
            "prediction_block_seconds": runner.PREDICTION_BLOCK_SECONDS,
            "prediction_min_calls": runner.PREDICTION_MIN_CALLS,
            "prediction_max_calls": runner.PREDICTION_MAX_CALLS,
        }:
            raise RuntimeError("panel-3 live execution policy changed")
        bindings = {
            "sha256": common.PROTOCOL,
            "runner_sha256": Path(runner.__file__).resolve(),
            "analyzer_sha256": Path(__file__).resolve(),
            "candidate_contract_sha256": common.CANDIDATE_CONTRACT,
            "power_design_file_sha256": common.POWER_DESIGN_DECISION,
        }
        for field, path in bindings.items():
            if protocol.get(field) != common.sha256_file(path):
                raise RuntimeError(
                    f"panel-3 raw source binding changed: {field}"
                )
    source_sha256 = registry["source_sha256"]
    protocol_source_bindings = {
        "sha256": protocol["path"],
        "runner_sha256": protocol["runner_path"],
        "analyzer_sha256": protocol["analyzer_path"],
        "candidate_contract_sha256": protocol[
            "candidate_contract_path"
        ],
    }
    if (
        any(
            source_sha256.get(relative) != protocol[field]
            for field, relative in protocol_source_bindings.items()
        )
        or protocol["power_design_file_sha256"]
        != registry["power_design_file_sha256"]
        or protocol["power_design_decision_path"]
        != registry["power_design_path"]
    ):
        raise RuntimeError(
            "panel-3 raw protocol/source digest binding changed"
        )
    environment = raw.get("environment")
    runtime_contract = (
        environment.get("runtime_contract")
        if isinstance(environment, dict)
        else None
    )
    machine_fingerprint = (
        environment.get("machine_fingerprint")
        if isinstance(environment, dict)
        else None
    )
    if (
        not isinstance(runtime_contract, dict)
        or environment.get("runtime_contract_normalized_sha256")
        != _json_sha256(runtime_contract)
        or not isinstance(machine_fingerprint, dict)
        or set(machine_fingerprint)
        != {
            "os",
            "os_release",
            "architecture",
            "cpu_identifier",
            "physical_cpu_count",
            "logical_cpu_count",
            "memory_bytes",
            "sha256",
        }
        or machine_fingerprint.get("sha256")
        != _json_sha256(
            {
                key: machine_fingerprint[key]
                for key in machine_fingerprint
                if key != "sha256"
            }
        )
    ):
        raise RuntimeError("panel-3 runtime evidence changed")
    runner._validate_embedded_runtime_contract(
        runtime_contract,
        registry["candidate_contract"],
        source_sha256=source_sha256,
    )
    if (
        verify_current_files
        and environment["runtime_contract"]
        != runner._validate_runtime_contract(
            registry["candidate_contract"]
        )
    ):
        raise RuntimeError("panel-3 runtime evidence changed")

    results = raw.get("results")
    failures = raw.get("comparator_failures")
    if (
        not isinstance(results, list)
        or not decision_worker_count <= len(results) <= worker_count
        or not isinstance(failures, list)
        or not 0 <= len(failures) <= comparator_worker_count
        or len(results) + len(failures) != worker_count
        or protocol["successful_worker_count"] != len(results)
        or protocol["comparator_failure_count"] != len(failures)
    ):
        raise RuntimeError("panel-3 raw result count changed")
    by_key = {}
    prediction_timing_policy = {
        "minimum_block_seconds": protocol["prediction_block_seconds"],
        "minimum_calls": protocol["prediction_min_calls"],
        "maximum_calls": protocol["prediction_max_calls"],
    }
    for result in results:
        _validate_result(result, prediction_timing_policy)
        _validate_fitted_metadata(result, strict=True)
        coordinate = _result_coordinate(result)
        key = (*coordinate, result["arm"])
        if key in by_key:
            raise RuntimeError("panel-3 raw results repeat a worker")
        by_key[key] = result
    failures_by_key = {}
    for failure in failures:
        _validate_comparator_failure(failure)
        coordinate = _result_coordinate(failure)
        key = (*coordinate, failure["arm"])
        if key in by_key or key in failures_by_key:
            raise RuntimeError("panel-3 raw workers repeat")
        failures_by_key[key] = failure
    expected = {
        (
            int(coordinate["task_id"]),
            int(coordinate["repeat"]),
            int(coordinate["fold"]),
            int(coordinate["sample"]),
            arm,
        )
        for coordinate in registry["coordinates"]
        for arm in arm_order
    }
    expected_decision = {
        (
            int(coordinate["task_id"]),
            int(coordinate["repeat"]),
            int(coordinate["fold"]),
            int(coordinate["sample"]),
            arm,
        )
        for coordinate in registry["coordinates"]
        for arm in decision_arms
    }
    if (
        not expected_decision <= set(by_key)
        or set(by_key) | set(failures_by_key) != expected
        or set(by_key) & set(failures_by_key)
        or any(key[4] not in runner.COMPARATOR_ARMS for key in failures_by_key)
    ):
        raise RuntimeError("panel-3 raw coordinate/arm set is incomplete")

    registry_rows = {
        int(row["task_id"]): row
        for row in registry["tasks"]
        if row["status"] == "selected"
    }
    for coordinate in {
        key[:4] for key in by_key
    }:
        rows = [
            by_key[(*coordinate, arm)] for arm in decision_arms
        ]
        rows.extend(
            by_key[(*coordinate, arm)]
            for arm in runner.COMPARATOR_ARMS
            if (*coordinate, arm) in by_key
        )
        registry_row = registry_rows[coordinate[0]]
        invariants = (
            "dataset_id",
            "dataset_name",
            "lineage_cluster",
            "stratum",
            "train_rows",
            "test_rows",
            "train_index_sha256",
            "test_index_sha256",
            "target_sha256",
            "categorical_feature_indices",
            "categorical_feature_names",
            "ordinal_features",
            "feature_policy",
            "split_policy",
        )
        for field in invariants:
            if any(row[field] != rows[0][field] for row in rows[1:]):
                raise RuntimeError(
                    f"panel-3 coordinate invariant changed: {field}"
                )
        if (
            rows[0]["dataset_id"] != int(registry_row["dataset_id"])
            or rows[0]["dataset_name"] != registry_row["dataset_name"]
            or rows[0]["lineage_cluster"]
            != registry_row["lineage_cluster"]
            or rows[0]["stratum"] != registry_row["stratum"]
            or rows[0]["ordinal_features"]
            != registry_row.get("ordinal_features", {})
            or rows[0]["categorical_feature_names"]
            != registry_row.get("resolved_categorical_columns")
            or rows[0]["feature_policy"]["policy_sha256"]
            != data_contract.canonical_json_sha256(
                registry_row.get("feature_policy", {"kind": "none"})
            )
            or rows[0]["feature_policy"]
            != registry_row.get("feature_policy_attestation")
        ):
            raise RuntimeError(
                "panel-3 result differs from its frozen task contract"
            )
        full_coordinate = {
            "task_id": coordinate[0],
            "repeat": coordinate[1],
            "fold": coordinate[2],
            "sample": coordinate[3],
        }
        frozen_split = runner._expected_split(
            registry_row, full_coordinate
        )
        for result_field, frozen_field in (
            ("train_rows", "train_size"),
            ("test_rows", "test_size"),
            ("train_index_sha256", "train_index_sha256"),
            ("test_index_sha256", "test_index_sha256"),
        ):
            if rows[0][result_field] != frozen_split[frozen_field]:
                raise RuntimeError(
                    "panel-3 result split differs from the registry"
                )
        split_policy = registry_row.get(
            "split_policy", {"kind": "openml_official"}
        )
        expected_construction_sha256 = (
            None
            if "construction" not in split_policy
            else data_contract.canonical_json_sha256(
                split_policy["construction"]
            )
        )
        if rows[0]["split_policy"] != {
            "kind": split_policy["kind"],
            "allow_unused_rows": bool(
                split_policy.get("allow_unused_rows", False)
            ),
            "construction_sha256": expected_construction_sha256,
        }:
            raise RuntimeError(
                "panel-3 result split policy differs from the registry"
            )
        control = by_key[(*coordinate, runner.CONTROL_ARM)]
        if "t5_composite_policy" in candidate_arms:
            composite = by_key[
                (*coordinate, "t5_composite_policy")
            ]
            applicability = registry_row.get(
                "t5_size_gate_applicability"
            )
            fold = coordinate[2]
            if (
                coordinate[1] != 0
                or coordinate[3] != 0
                or fold not in (0, 1, 2)
                or not isinstance(applicability, list)
                or len(applicability) != 3
                or type(applicability[fold]) is not bool
                or applicability[fold]
                != (
                    composite["train_rows"]
                    >= common.t5_minimum_outer_training_rows(
                        registry["candidate_contract"]
                    )
                )
                or composite["metadata"].get("size_gate")
                != common.t5_minimum_outer_training_rows(
                    registry["candidate_contract"]
                )
                or (
                    not applicability[fold]
                    and (
                        composite["metadata"]["engaged"] is not False
                        or composite["metadata"].get("decline_reason")
                        != "below_size_gate"
                        or composite["metadata"].get(
                            "total_selection_fit_seconds"
                        )
                        != 0.0
                    )
                )
                or (
                    applicability[fold]
                    and composite["metadata"].get("decline_reason")
                    == "below_size_gate"
                )
            ):
                raise RuntimeError(
                    "panel-3 T5 coordinate size-gate decision changed"
                )
            if (
                composite["metadata"]["engaged"] is False
                and composite["prediction_sha256"]
                != control["prediction_sha256"]
            ):
                raise RuntimeError(
                    "panel-3 declined T5 policy is not byte-identical "
                    "to current default"
                )

    spool = raw.get("spool")
    records = spool.get("records") if isinstance(spool, dict) else None
    payloads = [*results, *failures]
    if (
        not isinstance(records, list)
        or not isinstance(spool.get("directory"), str)
        or not spool["directory"]
        or spool.get("record_count") != worker_count
        or len(records) != worker_count
        or type(spool.get("resumed_record_count")) is not int
        or not 0 <= spool["resumed_record_count"] <= worker_count
        or len({record.get("worker_key") for record in records})
        != worker_count
        or {record.get("worker_key") for record in records}
        != {result["worker_key"] for result in payloads}
        or any(
            not isinstance(record, dict)
            or set(record)
            != {
                "worker_key",
                "filename",
                "spool_record_sha256",
                "spool_file_sha256",
                "attempt_filename",
                "attempt_sha256",
                "attempt_file_sha256",
                "claim_filename",
                "claim_sha256",
                "claim_file_sha256",
                "resumed",
            }
            or record.get("filename") != f"{record.get('worker_key')}.json"
            or record.get("attempt_filename")
            != f"{record.get('worker_key')}.attempt.json"
            or record.get("claim_filename")
            != f"{record.get('worker_key')}.claim.json"
            or not _is_sha256(record.get("spool_record_sha256"))
            or not _is_sha256(record.get("spool_file_sha256"))
            or not _is_sha256(record.get("attempt_sha256"))
            or not _is_sha256(record.get("attempt_file_sha256"))
            or not _is_sha256(record.get("claim_sha256"))
            or not _is_sha256(record.get("claim_file_sha256"))
            or type(record.get("resumed")) is not bool
            for record in records
        )
        or sum(record["resumed"] for record in records)
        != spool["resumed_record_count"]
    ):
        raise RuntimeError("panel-3 spool ledger is invalid")
    binding = spool.get("binding")
    sources = raw.get("sources")
    if (
        not isinstance(binding, dict)
        or set(binding)
        != {
            "schema_version",
            "runner_sha256",
            "analyzer_sha256",
            "protocol_sha256",
            "candidate_contract_sha256",
            "power_design_decision_sha256",
            "registry_file_sha256",
            "registry_canonical_sha256",
            "runtime_contract_normalized_sha256",
            "machine_fingerprint_sha256",
            "darkofit_head",
            "chimeraboost_head",
            "arms",
            "coordinate_count",
        }
        or binding["schema_version"] != 1
        or binding["runner_sha256"] != protocol["runner_sha256"]
        or binding["analyzer_sha256"] != protocol["analyzer_sha256"]
        or binding["protocol_sha256"] != protocol["sha256"]
        or binding["candidate_contract_sha256"]
        != protocol["candidate_contract_sha256"]
        or binding["power_design_decision_sha256"]
        != protocol["power_design_decision_sha256"]
        or binding["registry_file_sha256"]
        != registry_record["file_sha256"]
        or binding["registry_canonical_sha256"]
        != registry_record["canonical_sha256"]
        or binding["runtime_contract_normalized_sha256"]
        != environment["runtime_contract_normalized_sha256"]
        or binding["machine_fingerprint_sha256"]
        != machine_fingerprint["sha256"]
        or binding["arms"] != list(arm_order)
        or binding["coordinate_count"] != 36
        or not isinstance(sources, dict)
        or set(sources) != {"darkofit", "chimeraboost"}
        or not isinstance(sources["darkofit"], dict)
        or not isinstance(sources["chimeraboost"], dict)
        or binding["darkofit_head"] != sources["darkofit"].get("head")
        or binding["chimeraboost_head"]
        != sources["chimeraboost"].get("head")
        or sources["darkofit"].get("clean") is not True
        or sources["chimeraboost"].get("clean") is not True
        or sources["chimeraboost"].get("head")
        != registry["sources"]["chimeraboost_head"]
    ):
        raise RuntimeError("panel-3 spool/source binding is invalid")
    if verify_current_files and runner._machine_fingerprint() != (
        machine_fingerprint
    ):
        raise RuntimeError("panel-3 execution machine changed")
    payload_by_worker = {
        payload["worker_key"]: payload for payload in payloads
    }
    record_by_worker = {
        record["worker_key"]: record for record in records
    }
    for worker_key, payload in payload_by_worker.items():
        coordinate_tuple = _result_coordinate(payload)
        coordinate = {
            "task_id": coordinate_tuple[0],
            "repeat": coordinate_tuple[1],
            "fold": coordinate_tuple[2],
            "sample": coordinate_tuple[3],
        }
        reconstructed = runner._spool_payload(
            binding,
            coordinate,
            payload["arm"],
            payload,
        )
        reconstructed_attempt = runner._attempt_payload(
            binding,
            coordinate,
            payload["arm"],
        )
        reconstructed_claim = runner._claim_payload(
            binding,
            coordinate,
            payload["arm"],
            reconstructed_attempt["attempt_sha256"],
            runner._json_file_sha256(reconstructed_attempt),
        )
        if (
            reconstructed["spool_record_sha256"]
            != record_by_worker[worker_key]["spool_record_sha256"]
            or runner._json_file_sha256(reconstructed)
            != record_by_worker[worker_key]["spool_file_sha256"]
            or reconstructed_attempt["attempt_sha256"]
            != record_by_worker[worker_key]["attempt_sha256"]
            or runner._json_file_sha256(reconstructed_attempt)
            != record_by_worker[worker_key]["attempt_file_sha256"]
            or reconstructed_claim["claim_sha256"]
            != record_by_worker[worker_key]["claim_sha256"]
            or runner._json_file_sha256(reconstructed_claim)
            != record_by_worker[worker_key]["claim_file_sha256"]
        ):
            raise RuntimeError(
                "panel-3 offline worker ledger digest changed"
            )
        if payload.get("status") != "failed":
            source_attestation = payload["source_attestation"]["before"]
            if (
                source_attestation["registry_file_sha256"]
                != binding["registry_file_sha256"]
                or source_attestation["registry_canonical_sha256"]
                != binding["registry_canonical_sha256"]
                or source_attestation["darkofit"] != sources["darkofit"]
                or source_attestation["chimeraboost"]
                != sources["chimeraboost"]
            ):
                raise RuntimeError(
                    "panel-3 worker source attestation differs from "
                    "the campaign binding"
                )
    if verify_current_files:
        directory = Path(spool["directory"]).expanduser()
        if not directory.is_absolute():
            directory = ROOT / directory
        directory = directory.absolute()
        if (
            directory.is_symlink()
            or not directory.is_dir()
        ):
            raise RuntimeError("panel-3 spool directory is unavailable")
        for worker_key, payload in payload_by_worker.items():
            coordinate_tuple = _result_coordinate(payload)
            coordinate = {
                "task_id": coordinate_tuple[0],
                "repeat": coordinate_tuple[1],
                "fold": coordinate_tuple[2],
                "sample": coordinate_tuple[3],
            }
            path = directory / record_by_worker[worker_key]["filename"]
            attempt_path = (
                directory
                / record_by_worker[worker_key]["attempt_filename"]
            )
            claim_path = (
                directory
                / record_by_worker[worker_key]["claim_filename"]
            )
            attempt_digest, attempt_file_digest = runner._load_attempt(
                attempt_path,
                binding,
                coordinate,
                payload["arm"],
            )
            claim_digest, claim_file_digest = runner._load_claim(
                claim_path,
                binding,
                coordinate,
                payload["arm"],
                attempt_digest,
                attempt_file_digest,
            )
            observed, digest, spool_file_digest = runner._load_spool(
                path,
                binding,
                coordinate,
                payload["arm"],
            )
            if (
                observed != payload
                or digest
                != record_by_worker[worker_key]["spool_record_sha256"]
                or spool_file_digest
                != record_by_worker[worker_key]["spool_file_sha256"]
                or attempt_digest
                != record_by_worker[worker_key]["attempt_sha256"]
                or attempt_file_digest
                != record_by_worker[worker_key]["attempt_file_sha256"]
                or claim_digest
                != record_by_worker[worker_key]["claim_sha256"]
                or claim_file_digest
                != record_by_worker[worker_key]["claim_file_sha256"]
            ):
                raise RuntimeError(
                    "panel-3 spool record differs from raw artifact"
                )
    return by_key


def hierarchical_bootstrap_upper(
    coordinate_ratios: dict[str, list[float]],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
    batch: int = BOOTSTRAP_BATCH,
    percentile: float = 97.5,
) -> float:
    """Resample lineages and then coordinates within sampled lineages."""
    if (
        len(coordinate_ratios) != 12
        or type(seed) is not int
        or type(replicates) is not int
        or replicates <= 0
        or type(batch) is not int
        or batch <= 0
        or not 50.0 < float(percentile) < 100.0
    ):
        raise ValueError("panel-3 bootstrap inputs are invalid")
    ordered = []
    for name in sorted(coordinate_ratios):
        values = np.asarray(coordinate_ratios[name], dtype=np.float64)
        if (
            values.shape != (3,)
            or not np.isfinite(values).all()
            or np.any(values <= 0.0)
        ):
            raise ValueError(
                "panel-3 bootstrap requires three positive ratios per task"
            )
        ordered.append(np.log(values))
    logs = np.asarray(ordered, dtype=np.float64)
    rng = np.random.default_rng(seed)
    statistics = np.empty(replicates, dtype=np.float64)
    tasks = logs.shape[0]
    folds = logs.shape[1]
    for start in range(0, replicates, batch):
        count = min(batch, replicates - start)
        task_draws = rng.integers(0, tasks, size=(count, tasks))
        fold_draws = rng.integers(
            0,
            folds,
            size=(count, tasks, folds),
        )
        sampled = logs[task_draws[..., None], fold_draws]
        statistics[start : start + count] = sampled.mean(axis=(1, 2))
    return float(
        np.exp(np.percentile(statistics, percentile, method="linear"))
    )


def _dataset_ledger(
    by_key: dict[tuple[int, int, int, int, str], dict[str, Any]],
    arm: str,
    *,
    require_complete: bool = True,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[float]],
]:
    task_ids = sorted(
        {key[0] for key in by_key if key[4] == runner.CONTROL_ARM}
    )
    datasets = {}
    coordinate_ratios = {}
    for task_id in task_ids:
        arm_rows = sorted(
            (
                result
                for key, result in by_key.items()
                if key[0] == task_id and key[4] == arm
            ),
            key=_result_coordinate,
        )
        control_rows = sorted(
            (
                result
                for key, result in by_key.items()
                if key[0] == task_id
                and key[4] == runner.CONTROL_ARM
            ),
            key=_result_coordinate,
        )
        if len(control_rows) != 3:
            raise RuntimeError("panel-3 control coordinate count changed")
        if len(arm_rows) != 3:
            if require_complete:
                raise RuntimeError("panel-3 task coordinate count changed")
            continue
        ratios = [
            float(candidate["rmse"] / control["rmse"])
            for candidate, control in zip(
                arm_rows, control_rows, strict=True
            )
        ]
        fit_ratio = float(
            sum(row["fit_seconds"] for row in arm_rows)
            / sum(row["fit_seconds"] for row in control_rows)
        )
        prediction_ratio = float(
            sum(
                row["prediction_timing"]["per_call_median_seconds"]
                for row in arm_rows
            )
            / sum(
                row["prediction_timing"]["per_call_median_seconds"]
                for row in control_rows
            )
        )
        rss_ratio = float(
            max(row["peak_rss_bytes"] for row in arm_rows)
            / max(row["peak_rss_bytes"] for row in control_rows)
        )
        name = arm_rows[0]["lineage_cluster"]
        coordinate_ratios[name] = ratios
        datasets[name] = {
            "task_id": task_id,
            "dataset_id": arm_rows[0]["dataset_id"],
            "dataset_name": arm_rows[0]["dataset_name"],
            "stratum": arm_rows[0]["stratum"],
            "coordinate_ratios": ratios,
            "geomean_rmse_ratio": geomean(ratios),
            "worst_coordinate_rmse_ratio": max(ratios),
            "fit_seconds_ratio": fit_ratio,
            "predict_seconds_ratio": prediction_ratio,
            "peak_rss_ratio": rss_ratio,
        }
    return datasets, coordinate_ratios


def _aggregate_ledger(
    datasets: dict[str, dict[str, Any]],
    *,
    require_full_panel: bool,
) -> dict[str, Any]:
    if require_full_panel:
        quality = quality_metrics(
            {
                name: row["geomean_rmse_ratio"]
                for name, row in datasets.items()
            }
        )
    elif datasets:
        ratios = {
            name: row["geomean_rmse_ratio"]
            for name, row in sorted(datasets.items())
        }
        quality = {
            "dataset_ratios": ratios,
            "equal_dataset_geomean_ratio": geomean(
                list(ratios.values())
            ),
            "worst_dataset_ratio": max(ratios.values()),
        }
    else:
        quality = {
            "dataset_ratios": {},
            "equal_dataset_geomean_ratio": None,
            "worst_dataset_ratio": None,
        }
    if not datasets:
        return {
            **quality,
            "datasets": {},
            "coordinate_count": 0,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "equal_dataset_fit_seconds_ratio": None,
            "worst_dataset_fit_seconds_ratio": None,
            "equal_dataset_predict_seconds_ratio": None,
            "equal_dataset_peak_rss_ratio": None,
        }
    return {
        **quality,
        "datasets": datasets,
        "coordinate_count": 3 * len(datasets),
        "wins": sum(
            row["geomean_rmse_ratio"] < 1.0
            for row in datasets.values()
        ),
        "ties": sum(
            row["geomean_rmse_ratio"] == 1.0
            for row in datasets.values()
        ),
        "losses": sum(
            row["geomean_rmse_ratio"] > 1.0
            for row in datasets.values()
        ),
        "equal_dataset_fit_seconds_ratio": geomean(
            [row["fit_seconds_ratio"] for row in datasets.values()]
        ),
        "worst_dataset_fit_seconds_ratio": max(
            row["fit_seconds_ratio"] for row in datasets.values()
        ),
        "equal_dataset_predict_seconds_ratio": geomean(
            [row["predict_seconds_ratio"] for row in datasets.values()]
        ),
        "equal_dataset_peak_rss_ratio": geomean(
            [row["peak_rss_ratio"] for row in datasets.values()]
        ),
    }


def analyze_raw(
    raw: dict[str, Any],
    registry: dict[str, Any],
    *,
    raw_file_sha256: str,
    registry_path: Path = DEFAULT_REGISTRY,
    registry_file_sha256: str | None = None,
    verify_current_files: bool = True,
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    if not _is_sha256(raw_file_sha256):
        raise ValueError("panel-3 raw file SHA-256 is invalid")
    by_key = validate_raw(
        raw,
        registry,
        registry_path=registry_path,
        registry_file_sha256=registry_file_sha256,
        verify_current_files=verify_current_files,
    )
    candidate_arms = runner._candidate_arms(registry)
    decision_arms = runner._decision_arms(registry)
    arm_order = runner._arm_order(registry)
    power_decision = registry["power_design_decision"]
    simulation = power_decision["simulation"]
    bootstrap_seed = int(simulation["hierarchical_bootstrap_seed"])
    bootstrap_batch = int(simulation["hierarchical_bootstrap_batch"])
    embedded_bootstrap_replicates = int(
        simulation["hierarchical_bootstrap_replicates"]
    )
    bootstrap_override_used = (
        bootstrap_replicates is not None
        and bootstrap_replicates != embedded_bootstrap_replicates
    )
    if bootstrap_replicates is None:
        bootstrap_replicates = embedded_bootstrap_replicates
    if type(bootstrap_replicates) is not int or bootstrap_replicates <= 0:
        raise ValueError("panel-3 bootstrap replicate count is invalid")
    quality_gates = simulation["quality_gates"]
    decision_contract = registry["candidate_contract"]["decision"]
    operational_gates = {
        key: decision_contract[key]
        for key in PANEL3_V1_OPERATIONAL_GATES
    }
    default_selection_mapping = decision_contract[
        "default_selection_mapping"
    ]
    both_pass_reason = decision_contract["both_pass_reason"]
    familywise_alpha = float(
        power_decision["familywise_one_sided_alpha"]
    )
    bootstrap_percentile = float(
        power_decision["bootstrap_percentile"]
    )
    per_candidate_alpha = float(
        power_decision["per_candidate_one_sided_alpha"]
    )
    decision_worker_count = len(registry["coordinates"]) * len(
        decision_arms
    )
    candidates = {}
    all_arms = {}
    for arm in candidate_arms:
        datasets, coordinate_ratios = _dataset_ledger(
            by_key, arm, require_complete=True
        )
        dataset_ratios = {
            name: row["geomean_rmse_ratio"]
            for name, row in datasets.items()
        }
        aggregate = _aggregate_ledger(
            datasets, require_full_panel=True
        )
        upper = hierarchical_bootstrap_upper(
            coordinate_ratios,
            seed=bootstrap_seed,
            replicates=bootstrap_replicates,
            batch=bootstrap_batch,
            percentile=bootstrap_percentile,
        )
        decision = adjudicate_candidate(
            dataset_ratios,
            bonferroni_bootstrap_upper=upper,
            equal_dataset_fit_seconds_ratio=aggregate[
                "equal_dataset_fit_seconds_ratio"
            ],
            worst_dataset_fit_seconds_ratio=aggregate[
                "worst_dataset_fit_seconds_ratio"
            ],
            equal_dataset_predict_seconds_ratio=aggregate[
                "equal_dataset_predict_seconds_ratio"
            ],
            equal_dataset_peak_rss_ratio=aggregate[
                "equal_dataset_peak_rss_ratio"
            ],
            complete=sum(
                key[4] in decision_arms for key in by_key
            )
            == decision_worker_count,
            integrity_ok=not bootstrap_override_used,
            deviations=raw["protocol_deviations"],
            quality_gates=quality_gates,
            operational_gates=operational_gates,
            bootstrap_percentile=bootstrap_percentile,
            per_candidate_one_sided_alpha=per_candidate_alpha,
        )
        engagement = [
            result["metadata"]["engaged"]
            for key, result in by_key.items()
            if key[4] == arm
        ]
        decision.update(
            {
                "datasets": datasets,
                "coordinate_count": len(engagement),
                "engaged_coordinate_count": int(sum(engagement)),
                "declined_coordinate_count": int(
                    len(engagement) - sum(engagement)
                ),
                "wins": aggregate["wins"],
                "ties": aggregate["ties"],
                "losses": aggregate["losses"],
            }
        )
        candidates[arm] = decision
    for arm in runner.COMPARATOR_ARMS:
        datasets, _ = _dataset_ledger(
            by_key, arm, require_complete=False
        )
        aggregate = _aggregate_ledger(
            datasets, require_full_panel=False
        )
        failures = [
            value
            for value in raw["comparator_failures"]
            if value["arm"] == arm
        ]
        aggregate.update(
            {
                "decision_role": "descriptive_only",
                "affects_candidate_gates": False,
                "complete": len(datasets) == 12 and not failures,
                "complete_dataset_count": len(datasets),
                "failed_coordinate_count": len(failures),
                "failures": failures,
            }
        )
        all_arms[arm] = aggregate
    adjudication = adjudicate_two_candidates(
        candidates,
        retained_candidates=candidate_arms,
        default_selection_mapping=default_selection_mapping,
        both_pass_reason=both_pass_reason,
        familywise_one_sided_alpha=familywise_alpha,
        per_candidate_one_sided_alpha=per_candidate_alpha,
        bootstrap_percentile=bootstrap_percentile,
    )
    return {
        "schema_version": 1,
        "name": "darkofit_panel3_confirmation_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_file_sha256": raw_file_sha256,
        "raw_artifact_sha256": raw["raw_artifact_sha256"],
        "registry_file_sha256": raw["registry"]["file_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "bootstrap": {
            "seed": bootstrap_seed,
            "replicates": bootstrap_replicates,
            "batch": bootstrap_batch,
            "hierarchy": "lineage_then_coordinate_within_lineage",
            "percentile": bootstrap_percentile,
            "numpy_percentile_method": "linear",
            "embedded_replicates": embedded_bootstrap_replicates,
            "override_used": bootstrap_override_used,
            "authorization_eligible": not bootstrap_override_used,
            "multiplicity": (
                "Bonferroni_for_two_candidates"
                if len(candidate_arms) == 2
                else "single_preregistered_candidate"
            ),
        },
        **adjudication,
        "comparators": all_arms,
        "complete_coordinate_count": 36,
        "complete_decision_worker_count": decision_worker_count,
        "planned_worker_count": (
            len(registry["coordinates"]) * len(arm_order)
        ),
        "successful_worker_count": len(by_key),
        "comparator_failure_count": len(raw["comparator_failures"]),
        "comparator_failures_affect_candidate_gates": False,
        "task_imputation_used": False,
        "task_drop_used": False,
        "protocol_deviations": list(raw["protocol_deviations"]),
        "default_promotion_authorized": bool(
            adjudication["shipping_candidates"]
        ),
    }


def _markdown(summary: dict[str, Any]) -> str:
    candidate_names = list(summary["candidate_results"])
    percentile = float(summary["bootstrap"]["percentile"])
    lines = [
        "# Panel-3 retained-candidate confirmation",
        "",
        "This report is derived only from the immutable raw artifact bound "
        f"by file SHA-256 `{summary['raw_file_sha256']}` and canonical "
        f"payload SHA-256 `{summary['raw_artifact_sha256']}`.",
        "",
        f"| Candidate | RMSE ratio | {percentile:g}% upper | "
        "Worst dataset | "
        "Fit ratio | Predict ratio | RSS ratio | Decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for arm in candidate_names:
        row = summary["candidate_results"][arm]
        lines.append(
            f"| `{arm}` | "
            f"{row['equal_dataset_geomean_ratio']:.6f} | "
            f"{row['bonferroni_bootstrap_upper']:.6f} | "
            f"{row['worst_dataset_ratio']:.6f} | "
            f"{row['equal_dataset_fit_seconds_ratio']:.3f} | "
            f"{row['equal_dataset_predict_seconds_ratio']:.3f} | "
            f"{row['equal_dataset_peak_rss_ratio']:.3f} | "
            f"{'PASS' if row['passes'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"{len(candidate_names)} preregistered retained candidate"
            f"{'s were' if len(candidate_names) != 1 else ' was'} "
            "adjudicated independently. ChimeraBoost "
            "0.15.0 and CatBoost 1.2.10 are descriptive comparators and do "
            "not affect either candidate's gates; comparator failures are "
            "persisted but nonbinding.",
            "",
            "Independently confirmed candidates: "
            + (
                ", ".join(
                    f"`{value}`"
                    for value in summary[
                        "independently_confirmed_candidates"
                    ]
                )
                if summary["independently_confirmed_candidates"]
                else "none"
            )
            + ".",
            "",
            "Fixed-precedence default selection: "
            + (
                ", ".join(
                    f"`{value}`"
                    for value in summary["shipping_candidates"]
                )
                if summary["shipping_candidates"]
                else "none"
            )
            + ".",
            "",
            (
                "If both pass, T5 alone is selected because it already "
                "contains guarded crosses; no metric ranking or "
                "post-outcome discretion is used."
                if len(candidate_names) == 2
                else (
                    "The singleton candidate was retained before fresh "
                    "target access; no post-outcome candidate selection "
                    "is used."
                )
            ),
            "",
            "No task was dropped or imputed, and outcome-dependent reruns "
            "were forbidden by the frozen protocol.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args(argv)


def _publish_artifacts(
    summary: dict[str, Any],
    *,
    output: Path,
    markdown: Path,
) -> None:
    """Publish the display first and the canonical summary as commit marker.

    A failed summary publication may leave the create-only display behind.
    Treat an exact display match as a resumable partial publication; any
    differing pre-existing display remains a hard failure.
    """
    display = _markdown(summary).encode("utf-8")
    try:
        common.atomic_create(markdown, display)
    except FileExistsError:
        if common.secure_read_bytes(markdown) != display:
            raise RuntimeError(
                "existing panel-3 markdown differs from derived display"
            ) from None
    common.atomic_create(
        output,
        (
            json.dumps(
                summary, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n"
        ).encode("utf-8"),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    canonical = [
        DEFAULT_INPUT,
        DEFAULT_REGISTRY,
        DEFAULT_OUTPUT,
        DEFAULT_MARKDOWN,
    ]
    observed = [
        args.input.expanduser().absolute(),
        args.registry.expanduser().absolute(),
        args.output.expanduser().absolute(),
        args.markdown.expanduser().absolute(),
    ]
    if observed != canonical:
        raise RuntimeError("panel-3 analysis path changed")
    paths = observed[2:]
    if len(set(paths)) != len(paths):
        raise RuntimeError("refusing existing or aliased panel-3 output")
    common.validate_create_path(paths[0])
    try:
        common.validate_create_path(paths[1])
    except FileExistsError:
        # An exact-content comparison is deferred until after the summary is
        # re-derived. This only permits recovery from a display-first partial
        # publication; the canonical summary must still be absent.
        common.secure_read_bytes(paths[1])
    raw, raw_file_sha256 = common.secure_load_json(args.input)
    registry, registry_file_sha256 = common.secure_load_json(args.registry)
    if not isinstance(raw, dict) or not isinstance(registry, dict):
        raise RuntimeError("invalid panel-3 raw or registry JSON")
    summary = analyze_raw(
        raw,
        registry,
        raw_file_sha256=raw_file_sha256,
        registry_path=args.registry,
        registry_file_sha256=registry_file_sha256,
    )
    summary["summary_artifact_sha256"] = _json_sha256(summary)
    final_raw, final_raw_file_sha256 = common.secure_load_json(args.input)
    final_registry, final_registry_file_sha256 = common.secure_load_json(
        args.registry
    )
    if (
        final_raw != raw
        or final_raw_file_sha256 != raw_file_sha256
        or final_registry != registry
        or final_registry_file_sha256 != registry_file_sha256
    ):
        raise RuntimeError(
            "panel-3 raw or registry changed before analysis publication"
        )
    _publish_artifacts(
        summary,
        output=paths[0],
        markdown=paths[1],
    )
    print(
        json.dumps(
            {
                "output": str(paths[0]),
                "markdown": str(paths[1]),
                "summary_artifact_sha256": summary[
                    "summary_artifact_sha256"
                ],
                "shipping_candidates": summary["shipping_candidates"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
