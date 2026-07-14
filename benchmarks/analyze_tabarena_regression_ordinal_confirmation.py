"""Verify and analyze the frozen safe-ordinal replication campaign.

The campaign runner is the only process allowed to decode TabArena result
pickles.  It emits an attested JSON snapshot containing the normalized fields
needed here.  This analyzer treats every ``results.pkl`` as opaque bytes: it
verifies path, size, and SHA-256, but never unpickles executable data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import stat
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from benchmarks import analyze_tabarena_regression_cap_horizon as hardened
    from benchmarks import run_tabarena_regression_ordinal_confirmation as campaign
except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
    import analyze_tabarena_regression_cap_horizon as hardened
    import run_tabarena_regression_ordinal_confirmation as campaign


BOOTSTRAP_SEED = 20_260_718
BOOTSTRAP_DRAWS = 10_000
METRICS = (
    "test_rmse",
    "val_rmse",
    "train_time_s",
    "infer_time_s",
    "peak_memory_bytes",
)
ARM_CODES = {arm: spec["code"] for arm, spec in campaign.ARM_SPECS.items()}
CONTRAST_SPECS: dict[str, dict[str, str]] = {
    "ordinal_vs_fixed_base": {
        "code": "O/B",
        "numerator": "fixed_base_safe_ordinal",
        "denominator": "fixed_base_native",
        "role": "primary_causal",
    },
    "ordinal_vs_product_default": {
        "code": "O/P",
        "numerator": "fixed_base_safe_ordinal",
        "denominator": "product_default_native",
        "role": "deployment",
    },
    "fixed_base_vs_product_default": {
        "code": "B/P",
        "numerator": "fixed_base_native",
        "denominator": "product_default_native",
        "role": "attribution_only",
    },
}
ATTRIBUTION_CAN_ADVANCE = False
GATE_THRESHOLDS: dict[str, dict[str, float]] = {
    "O/B": {
        "test_ratio_max": 0.995,
        "bootstrap_upper95_max": 1.0,
        "each_dataset_test_ratio_max": 0.995,
        "sign_test_p_max": 0.05,
        "validation_ratio_max": 1.002,
        "each_dataset_validation_ratio_max": 1.005,
        "train_time_ratio_max": 1.5,
        "infer_time_ratio_max": 1.25,
        "peak_memory_ratio_max": 1.25,
    },
    "O/P": {
        "test_ratio_max": 0.995,
        "bootstrap_upper95_max": 1.0,
        "each_dataset_test_ratio_max": 1.005,
        "sign_test_p_max": 0.05,
        "validation_ratio_max": 1.002,
        "train_time_ratio_max": 1.5,
        "infer_time_ratio_max": 1.25,
        "peak_memory_ratio_max": 1.25,
    },
}
OUTPUT_KEYS = (
    "split_csv",
    "repeat_csv",
    "child_csv",
    "summary_json",
    "report_md",
)
OUTPUT_NAMES = tuple(campaign.DEFAULT_ANALYSIS_OUTPUT_FILENAMES)
VALID_STOP_REASONS = {"iteration_limit", "early_stopping", "no_split", "time_limit"}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"{field} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RuntimeError(f"{field} must be a SHA-256 digest") from exc
    return value


def _read_json_stable(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"could not inspect {field}: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{field} must be a regular file: {path}")
    payload = hardened._read_stable(path, field)
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{field} is not valid JSON") from exc
    return dict(hardened._as_mapping(decoded, field)), payload


def _artifact_bytes(
    input_dir: Path,
    relative: str,
    metadata: Mapping[str, Any],
    field: str,
) -> bytes:
    if set(metadata) != {"sha256", "size_bytes"}:
        raise RuntimeError(f"{field} attestation fields are not exact")
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or not relative_path.parts
    ):
        raise RuntimeError(f"unsafe attested path: {relative!r}")
    raw_path = input_dir / relative_path
    cursor = input_dir
    for component in relative_path.parts:
        cursor = cursor / component
        try:
            component_metadata = cursor.lstat()
        except OSError as exc:
            raise RuntimeError(f"could not inspect {field}: {cursor}") from exc
        if stat.S_ISLNK(component_metadata.st_mode):
            raise RuntimeError(f"{field} must not contain symbolic-link components")
    try:
        raw_metadata = raw_path.lstat()
    except OSError as exc:
        raise RuntimeError(f"could not inspect {field}: {raw_path}") from exc
    if not stat.S_ISREG(raw_metadata.st_mode):
        raise RuntimeError(f"{field} must be a regular file")
    path = raw_path.resolve(strict=True)
    try:
        path.relative_to(input_dir.resolve(strict=True))
    except ValueError as exc:
        raise RuntimeError(f"attested path escapes campaign: {relative}") from exc
    payload = hardened._read_stable(path, field)
    size = hardened._exact_int(metadata.get("size_bytes"), f"{field} size")
    digest = _validate_digest(metadata.get("sha256"), f"{field} digest")
    if len(payload) != size or _sha256(payload) != digest:
        raise RuntimeError(f"{field} does not match its attestation")
    return payload


def _verify_repository_source(
    source: Mapping[str, Any], input_dir: Path
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1].resolve(strict=True)
    expected_fields = {
        "repository",
        "git_head",
        "git_tree",
        "relevant_status",
        "files",
        "darkofit_import",
        "tabarena",
    }
    if set(source) != expected_fields or source.get("relevant_status") != "":
        raise RuntimeError("run manifest source provenance is incomplete or dirty")
    recorded_repository = hardened._manifest_path(
        source.get("repository"), "recorded ordinal-confirmation repository"
    )
    if recorded_repository != repository:
        raise RuntimeError("executing analyzer repository does not match the run")
    files = hardened._as_mapping(source.get("files"), "source files")
    expected_files = {str(path) for path in campaign.SOURCE_FILES}
    if set(files) != expected_files:
        raise RuntimeError("run manifest source file set is not exact")
    for relative in campaign.SOURCE_FILES:
        key = str(relative)
        path = (repository / relative).resolve(strict=True)
        try:
            path.relative_to(repository)
        except ValueError as exc:
            raise RuntimeError(f"source file escapes repository: {relative}") from exc
        metadata = hardened._as_mapping(files[key], f"source metadata for {key}")
        if set(metadata) != {"sha256", "git_blob"}:
            raise RuntimeError(f"source metadata is incomplete for {key}")
        payload = hardened._read_stable(path, f"source {key}")
        if metadata.get("sha256") != _sha256(payload):
            raise RuntimeError(f"source SHA-256 mismatch for {key}")
        if metadata.get("git_blob") != hardened._git_hash_payload(
            repository, payload, key
        ):
            raise RuntimeError(f"source Git-blob mismatch for {key}")
    head = hardened._git_output(repository, ["rev-parse", "HEAD"], "Git HEAD")
    tree = hardened._git_output(
        repository, ["rev-parse", "HEAD^{tree}"], "Git tree"
    )
    if source.get("git_head") != head or source.get("git_tree") != tree:
        raise RuntimeError("executing Git revision does not match the run")
    changes = hardened._repository_changes(repository, input_dir)
    if changes:
        raise RuntimeError(
            "executing ordinal-confirmation repository has dirty or unrecorded "
            "code: " + ", ".join(changes)
        )
    return {
        "executing_repository": str(repository),
        "executing_git_head": head,
        "executing_git_tree": tree,
    }


def verify_execution_provenance(
    manifest: Mapping[str, Any], input_dir: Path
) -> dict[str, Any]:
    source = hardened._as_mapping(manifest.get("source"), "manifest source")
    diagnostics = _verify_repository_source(source, input_dir)
    repository = Path(__file__).resolve().parents[1]
    hardened._verify_dependency_provenance(
        source.get("darkofit_import"),
        "darkofit",
        input_dir,
        required_repository=repository,
    )
    hardened._verify_dependency_provenance(
        source.get("tabarena"), "tabarena", input_dir
    )
    hardened._verify_runtime_provenance(manifest.get("runtime"))
    campaign.validate_product_effective_defaults()
    return {
        **diagnostics,
        "executing_source_verified": True,
        "analysis_runtime_verified": True,
        "dependency_provenance_verified": True,
        "hardware_identity_verified": True,
    }


def _positive(value: Any, field: str) -> float:
    return hardened._positive_finite(value, field)


def _nonnegative(value: Any, field: str) -> float:
    return hardened._nonnegative_finite(value, field)


def _expected_ordered_grid() -> list[tuple[str, int, int, str]]:
    rows = list(campaign.expected_ordered_grid())
    if len(rows) != campaign.EXPECTED_JOBS or len(set(rows)) != len(rows):
        raise RuntimeError("runner's frozen ordered grid is not exact")
    if set(rows) != campaign.expected_grid():
        raise RuntimeError("runner's ordered and unordered grids disagree")
    return rows


def _validate_ordering_metadata(
    manifest: Mapping[str, Any],
    attestation: Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
) -> str:
    expected_digest = campaign.job_order_sha256()
    if manifest.get("job_order_sha256") != expected_digest:
        raise RuntimeError("manifest job-order digest does not match the frozen order")
    expected_audit = _expected_ordering_audit()
    if manifest.get("ordering_audit") != expected_audit:
        raise RuntimeError("manifest ordering audit does not match the frozen order")
    for name, value in (("attestation", attestation), ("analysis payload", payload)):
        if value is not None and value.get("job_order_sha256") != expected_digest:
            raise RuntimeError(f"{name} job-order digest does not match")
    return expected_digest


def _expected_ordering_audit() -> dict[str, Any]:
    """Recompute the runner's position audit from the frozen tuple grid."""
    ordered = _expected_ordered_grid()
    per_dataset: dict[str, dict[str, dict[str, int]]] = {}
    overall = {arm: [0, 0, 0] for arm in campaign.ARM_SPECS}
    cursor = 0
    for dataset in campaign.TASKS:
        coordinates = [
            coordinate
            for coordinate in campaign.expected_coordinates()
            if coordinate[0] == dataset
        ]
        counts = {arm: [0, 0, 0] for arm in campaign.ARM_SPECS}
        for coordinate in coordinates:
            group = ordered[cursor : cursor + 3]
            cursor += 3
            if any(tuple(row[:3]) != tuple(coordinate) for row in group):
                raise RuntimeError("frozen job-order groups are not coordinate-local")
            for position, row in enumerate(group):
                arm = row[3]
                counts[arm][position] += 1
                overall[arm][position] += 1
        per_dataset[dataset] = {
            arm: {
                "first": values[0],
                "second": values[1],
                "third": values[2],
            }
            for arm, values in counts.items()
        }
    if cursor != campaign.EXPECTED_JOBS or any(
        values != [11, 11, 11] for values in overall.values()
    ):
        raise RuntimeError("frozen job-order position balance changed")
    return {
        "job_order_sha256": campaign.job_order_sha256(),
        "per_dataset_position_counts": per_dataset,
        "overall_position_counts": {
            arm: {
                "first": values[0],
                "second": values[1],
                "third": values[2],
            }
            for arm, values in overall.items()
        },
    }


def _expected_framework(arm: str) -> str:
    return f"DarkoFit_c1_ordinal_confirm_{arm}_BAG_L1"


def _validate_outer_rows(
    rows: Any,
    artifacts: Mapping[str, Any],
    child_cpus: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int, str], dict[str, Any]]]:
    if not isinstance(rows, list) or len(rows) != campaign.EXPECTED_JOBS:
        raise RuntimeError("analysis payload outer result count is wrong")
    fields = {
        "dataset",
        "task_id",
        "repeat",
        "fold",
        "registered_fold",
        "arm",
        "arm_code",
        "test_rmse",
        "val_rmse",
        "train_time_s",
        "infer_time_s",
        "peak_memory_bytes",
        "framework",
        "num_cpus",
        "num_gpus",
        "num_cpus_child",
        "num_gpus_child",
        "source",
    }
    expected_order = _expected_ordered_grid()
    normalized: list[dict[str, Any]] = []
    index: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    source_counts: Counter[str] = Counter()
    for position, raw in enumerate(rows):
        row = dict(hardened._as_mapping(raw, f"outer_rows[{position}]"))
        if set(row) != fields:
            raise RuntimeError(f"outer_rows[{position}] fields are not exact")
        dataset = row["dataset"]
        arm = row["arm"]
        repeat = hardened._exact_int(row["repeat"], "outer repeat")
        fold = hardened._exact_int(row["fold"], "outer fold")
        registered = hardened._exact_int(
            row["registered_fold"], "outer registered fold"
        )
        task_id = hardened._exact_int(row["task_id"], "outer task id")
        key = (dataset, repeat, fold, arm)
        if key != expected_order[position]:
            raise RuntimeError("safe outer rows are not in frozen execution order")
        if (
            dataset not in campaign.TASKS
            or arm not in campaign.ARM_SPECS
            or task_id != campaign.TASKS[dataset]
            or registered != 3 * repeat + fold
            or key in index
            or row["arm_code"] != campaign.ARM_SPECS[arm]["code"]
            or row["framework"] != _expected_framework(arm)
        ):
            raise RuntimeError(f"safe outer row does not match frozen grid: {key}")
        num_cpus = hardened._exact_int(row["num_cpus"], "outer CPUs")
        num_gpus = hardened._exact_int(row["num_gpus"], "outer GPUs")
        num_cpus_child = hardened._exact_int(
            row["num_cpus_child"], "outer child CPUs"
        )
        num_gpus_child = hardened._exact_int(
            row["num_gpus_child"], "outer child GPUs"
        )
        if (
            num_cpus != child_cpus
            or num_cpus_child != child_cpus
            or num_gpus != 0
            or num_gpus_child != 0
        ):
            raise RuntimeError(f"outer resource allocation changed for {key}")
        source = row["source"]
        if not isinstance(source, str) or source not in artifacts:
            raise RuntimeError(f"outer source is not attested for {key}")
        campaign._validate_result_source_binding(
            source,
            dataset=dataset,
            repeat=repeat,
            fold=fold,
            arm=arm,
        )
        source_counts[source] += 1
        clean = {
            **row,
            "task_id": task_id,
            "repeat": repeat,
            "fold": fold,
            "registered_fold": registered,
            "num_cpus": num_cpus,
            "num_gpus": num_gpus,
            "num_cpus_child": num_cpus_child,
            "num_gpus_child": num_gpus_child,
        }
        for metric in METRICS:
            clean[metric] = _positive(row[metric], f"outer {metric}")
        normalized.append(clean)
        index[key] = clean
    if set(index) != campaign.expected_grid():
        raise RuntimeError("safe outer-result grid is not exact")
    if set(source_counts) != set(artifacts) or any(
        count != 1 for count in source_counts.values()
    ):
        raise RuntimeError("safe outer rows do not bind one-to-one to raw results")
    return normalized, index


def _validate_child_policy(
    row: dict[str, Any],
    *,
    index: int,
) -> None:
    arm = row["arm"]
    child_fold = row["child_fold"]
    expected_user = campaign.ARM_SPECS[arm]["config"]
    if row["user_hyperparameters"] != expected_user:
        raise RuntimeError(f"child_rows[{index}] user policy is not frozen")
    expected_initial = campaign.expected_child_hyperparameters(arm, child_fold)
    if row["initial_hyperparameters"] != expected_initial:
        raise RuntimeError(f"child_rows[{index}] initialized policy is not frozen")
    if row["effective_hyperparameters"] != campaign.expected_effective_child_hyperparameters(
        arm, child_fold
    ):
        raise RuntimeError(f"child_rows[{index}] effective policy is not frozen")

    requested = hardened._exact_int(
        row["iterations_requested"], "iterations requested"
    )
    attempted = hardened._exact_int(
        row["iterations_attempted"], "iterations attempted"
    )
    completed = hardened._exact_int(row["rounds_completed"], "rounds completed")
    retained = hardened._exact_int(row["rounds_retained"], "rounds retained")
    best = hardened._exact_int(row["best_iteration"], "best iteration")
    if requested != 1_000 or not (
        0 <= retained == best <= completed <= attempted <= requested
    ):
        raise RuntimeError(f"child_rows[{index}] round counters are inconsistent")
    resolved_lr = _positive(row["resolved_learning_rate"], "resolved learning rate")
    if arm != "product_default_native" and resolved_lr != 0.1:
        raise RuntimeError(f"child_rows[{index}] fixed arm did not use LR=0.1")
    if (
        row["requested_tree_mode"] != "catboost"
        or row["selected_tree_mode"] != "catboost"
        or row["selected_lane"] != "boosting"
        or row["linear_residual_active"] is not False
    ):
        raise RuntimeError(f"child_rows[{index}] mode/lane policy changed")
    early_rounds = hardened._exact_int(
        row["early_stopping_rounds"], "early-stopping rounds"
    )
    if early_rounds != campaign._expected_patience(resolved_lr):
        raise RuntimeError(f"child_rows[{index}] early-stopping patience changed")
    reason = row["stop_reason"]
    if reason not in VALID_STOP_REASONS:
        raise RuntimeError(f"child_rows[{index}] stop reason is invalid")
    hardened._validate_stop_reason_causality(
        reason,
        requested=requested,
        attempted=attempted,
        completed=completed,
        field=f"child_rows[{index}]",
    )
    if (
        reason == "time_limit"
        or row["deadline_hit"] is not False
        or row["deadline_is_soft"] is not True
    ):
        raise RuntimeError(f"child_rows[{index}] hit a forbidden deadline")
    wall_limit = _positive(
        row["wall_clock_limit_seconds"],
        f"child_rows[{index}].wall_clock_limit_seconds",
    )
    wall_margin = _nonnegative(
        row["wall_clock_safety_margin_seconds"],
        f"child_rows[{index}].wall_clock_safety_margin_seconds",
    )
    wall_effective = _nonnegative(
        row["wall_clock_effective_seconds"],
        f"child_rows[{index}].wall_clock_effective_seconds",
    )
    wall_elapsed = _nonnegative(
        row["wall_clock_elapsed_seconds"],
        f"child_rows[{index}].wall_clock_elapsed_seconds",
    )
    if (
        wall_limit > campaign.TIME_LIMIT_SECONDS
        or wall_elapsed < 0.0
        or not math.isclose(
            wall_margin,
            min(5.0, 0.05 * wall_limit),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            wall_effective,
            max(0.0, wall_limit - wall_margin),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(f"child_rows[{index}] wall-clock audit is inconsistent")
    refit = campaign._validate_refit_params(
        row["refit_params"],
        arm=arm,
        expected_iterations=best,
        expected_learning_rate=resolved_lr,
        field=f"child_rows[{index}].refit_params",
    )
    if _positive(refit["learning_rate"], "refit learning rate") != resolved_lr:
        raise RuntimeError(f"child_rows[{index}] refit LR does not match fitted LR")


def _validate_child_rows(
    rows: Any,
    outer_index: Mapping[tuple[str, int, int, str], Mapping[str, Any]],
    child_cpus: int,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != campaign.EXPECTED_CHILD_FITS:
        raise RuntimeError("analysis payload child result count is wrong")
    fields = {
        "dataset",
        "task_id",
        "repeat",
        "fold",
        "registered_fold",
        "arm",
        "arm_code",
        "child",
        "child_fold",
        "iterations_requested",
        "iterations_attempted",
        "rounds_completed",
        "rounds_retained",
        "best_iteration",
        "resolved_learning_rate",
        "early_stopping_rounds",
        "requested_tree_mode",
        "selected_tree_mode",
        "selected_lane",
        "linear_residual_active",
        "stop_reason",
        "wall_clock_limit_seconds",
        "wall_clock_safety_margin_seconds",
        "wall_clock_effective_seconds",
        "deadline_hit",
        "deadline_is_soft",
        "wall_clock_elapsed_seconds",
        "child_features",
        "representation",
        "refit_params",
        "initial_hyperparameters",
        "user_hyperparameters",
        "effective_hyperparameters",
        "num_cpus",
        "num_gpus",
        "source",
    }
    expected_order = [
        (*outer, child_fold)
        for outer in _expected_ordered_grid()
        for child_fold in range(8)
    ]
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str, int]] = set()
    per_outer: Counter[tuple[str, int, int, str]] = Counter()
    for position, raw in enumerate(rows):
        row = dict(hardened._as_mapping(raw, f"child_rows[{position}]"))
        if set(row) != fields:
            raise RuntimeError(f"child_rows[{position}] fields are not exact")
        dataset = row["dataset"]
        arm = row["arm"]
        repeat = hardened._exact_int(row["repeat"], "child repeat")
        fold = hardened._exact_int(row["fold"], "child fold coordinate")
        registered = hardened._exact_int(
            row["registered_fold"], "child registered fold"
        )
        task_id = hardened._exact_int(row["task_id"], "child task id")
        child_fold = hardened._exact_int(row["child_fold"], "child fold")
        outer_key = (dataset, repeat, fold, arm)
        key = (*outer_key, child_fold)
        if key != expected_order[position]:
            raise RuntimeError("safe child rows are not in frozen execution order")
        outer = outer_index.get(outer_key)
        if (
            outer is None
            or key in seen
            or child_fold not in range(8)
            or row["arm_code"] != campaign.ARM_SPECS[arm]["code"]
            or row["child"] != f"S1F{child_fold + 1}"
            or task_id != campaign.TASKS[dataset]
            or registered != 3 * repeat + fold
            or row["source"] != outer["source"]
            or hardened._exact_int(row["num_cpus"], "child CPUs") != child_cpus
            or hardened._exact_int(row["num_gpus"], "child GPUs") != 0
        ):
            raise RuntimeError(f"safe child row does not match outer row: {key}")
        child_features = row["child_features"]
        campaign.followon._feature_schema_sha256(
            child_features, f"child_rows[{position}].child_features"
        )
        campaign._validate_representation_metadata(
            row["representation"],
            arm=arm,
            dataset=dataset,
            field=f"child_rows[{position}].representation",
            child_features=child_features,
        )
        clean = {
            **row,
            "task_id": task_id,
            "repeat": repeat,
            "fold": fold,
            "registered_fold": registered,
            "child_fold": child_fold,
        }
        _validate_child_policy(clean, index=position)
        normalized.append(clean)
        seen.add(key)
        per_outer[outer_key] += 1
    expected_children = {
        (*outer, child_fold)
        for outer in campaign.expected_grid()
        for child_fold in range(8)
    }
    if seen != expected_children or any(
        per_outer[outer] != 8 for outer in campaign.expected_grid()
    ):
        raise RuntimeError("safe child-fit grid is not exact")
    _validate_cross_arm_representation_activity(normalized)
    return normalized


def _validate_cross_arm_representation_activity(
    child_rows: Sequence[Mapping[str, Any]],
) -> None:
    by_coordinate: dict[
        tuple[str, int, int, int], dict[str, Mapping[str, Any]]
    ] = defaultdict(dict)
    activity = Counter()
    for row in child_rows:
        key = (
            str(row["dataset"]),
            int(row["repeat"]),
            int(row["fold"]),
            int(row["child_fold"]),
        )
        by_coordinate[key][str(row["arm"])] = row
        representation = hardened._as_mapping(
            row["representation"], "child representation"
        )
        feature_count = len(row["child_features"])
        representation_input_count = hardened._exact_int(
            representation.get("input_feature_count"),
            "representation input feature count",
        )
        representation_output_count = hardened._exact_int(
            representation.get("output_feature_count"),
            "representation output feature count",
        )
        if representation_input_count != feature_count or (
            row["arm"] == "fixed_base_safe_ordinal"
            and representation_output_count != feature_count
        ):
            raise RuntimeError(
                "representation input/output schema is not bound to child features"
            )
        kind = representation.get("kind")
        activity[(row["arm"], kind)] += 1
        if row["arm"] == "fixed_base_safe_ordinal":
            positions = representation.get("categorical_input_positions")
            observed = representation.get("observed_training_category_counts")
            if (
                kind != "safe_ordinal"
                or not isinstance(positions, list)
                or not positions
                or not isinstance(observed, list)
                or len(observed) != len(positions)
                or any(hardened._exact_int(value, "observed categories") < 1 for value in observed)
                or representation.get("remaining_native_target_stat_positions") != []
                or representation.get("target_used_by_representation") is not False
            ):
                raise RuntimeError("safe ordinal representation was not active")
        else:
            if (
                kind != "native"
                or not representation.get("fitted_categorical_input_columns")
                or representation.get("target_used_by_representation") is not True
            ):
                raise RuntimeError("native target-stat representation was not active")
    if len(by_coordinate) * 3 != len(child_rows):
        raise RuntimeError("child representations do not form three-arm blocks")
    for key, arms in by_coordinate.items():
        if set(arms) != set(ARM_CODES):
            raise RuntimeError(f"child representation block is incomplete: {key}")
        schemas = [arms[arm]["child_features"] for arm in ARM_CODES]
        if schemas[1:] != schemas[:-1]:
            raise RuntimeError(f"external child feature schema differs by arm: {key}")
        product = arms["product_default_native"]["representation"]
        fixed = arms["fixed_base_native"]["representation"]
        if product != fixed:
            raise RuntimeError(f"native child preprocessing differs by arm: {key}")
    expected_per_arm = campaign.EXPECTED_CHILD_FITS // 3
    expected_activity = {
        ("product_default_native", "native"): expected_per_arm,
        ("fixed_base_native", "native"): expected_per_arm,
        ("fixed_base_safe_ordinal", "safe_ordinal"): expected_per_arm,
    }
    if dict(activity) != expected_activity:
        raise RuntimeError("representation activity counts are not exact")


def load_safe_rows(
    payload: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    child_cpus: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outer_rows, outer_index = _validate_outer_rows(
        payload.get("outer_rows"), artifacts, child_cpus
    )
    child_rows = _validate_child_rows(
        payload.get("child_rows"), outer_index, child_cpus
    )
    return outer_rows, child_rows


def verify_campaign_integrity(
    input_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify every frozen campaign byte and load only attested safe JSON."""
    input_dir = input_dir.resolve(strict=True)
    manifest_path = input_dir / campaign.MANIFEST_FILENAME
    attestation_path = input_dir / campaign.COMPLETION_ATTESTATION_FILENAME
    manifest, manifest_bytes = _read_json_stable(manifest_path, "run manifest")
    if set(manifest) != {
        "schema_version",
        "kind",
        "created_at_utc",
        "output_dir",
        "time_limit_seconds",
        "resolved_child_num_cpus",
        "protocol_sha256",
        "job_order_sha256",
        "protocol",
        "ordering_audit",
        "source",
        "runtime",
    }:
        raise RuntimeError("run manifest fields are not exact")
    protocol = campaign.frozen_protocol()
    protocol_digest = _sha256(_canonical_json(protocol))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != campaign.CAMPAIGN_KIND
        or Path(str(manifest.get("output_dir", ""))).resolve() != input_dir
        or manifest.get("protocol") != protocol
        or manifest.get("protocol_sha256") != protocol_digest
        or manifest.get("time_limit_seconds") != campaign.TIME_LIMIT_SECONDS
    ):
        raise RuntimeError("run manifest does not match the frozen confirmation")
    child_cpus = hardened._exact_int(
        manifest.get("resolved_child_num_cpus"), "manifest child CPUs"
    )
    if child_cpus < 1:
        raise RuntimeError("manifest child CPUs must be positive")
    execution = verify_execution_provenance(manifest, input_dir)
    expected_order_digest = _validate_ordering_metadata(manifest)

    attestation, attestation_bytes = _read_json_stable(
        attestation_path, "completion attestation"
    )
    if set(attestation) != {
        "schema_version",
        "kind",
        "completed_at_utc",
        "pid",
        "result_count",
        "expected_result_count",
        "expected_child_fits",
        "expected_paired_comparisons",
        "expected_independent_contrasts",
        "warmup_thread_count",
        "warmup_stage_count",
        "warmup_expected_counts",
        "protocol_sha256",
        "job_order_sha256",
        "git_head",
        "manifest_sha256",
        "result_artifacts",
        "analysis_payload_artifact",
        "warmup_history_artifact",
        "resume_history_artifact",
        "validation",
    }:
        raise RuntimeError("completion attestation fields are not exact")
    expected_counts = {
        "result_count": campaign.EXPECTED_JOBS,
        "expected_result_count": campaign.EXPECTED_JOBS,
        "expected_child_fits": campaign.EXPECTED_CHILD_FITS,
        "expected_paired_comparisons": campaign.EXPECTED_PAIRED_COMPARISONS,
    }
    if (
        attestation.get("schema_version") != 1
        or attestation.get("kind") != campaign.COMPLETION_KIND
        or any(attestation.get(key) != value for key, value in expected_counts.items())
        or attestation.get("protocol_sha256") != protocol_digest
        or attestation.get("git_head") != manifest["source"]["git_head"]
        or attestation.get("manifest_sha256") != _sha256(manifest_bytes)
        or attestation.get("warmup_thread_count") != child_cpus
        or attestation.get("expected_independent_contrasts")
        != campaign.EXPECTED_INDEPENDENT_CONTRASTS
        or attestation.get("warmup_stage_count")
        != len(campaign.WARMUP_STAGE_SPECS)
        or attestation.get("warmup_expected_counts")
        != campaign.EXPECTED_WARMUP_COUNTS
    ):
        raise RuntimeError("completion attestation does not match the confirmation")
    _validate_ordering_metadata(manifest, attestation=attestation)

    artifacts = hardened._as_mapping(
        attestation.get("result_artifacts"), "result artifacts"
    )
    if len(artifacts) != campaign.EXPECTED_JOBS:
        raise RuntimeError("attested result count does not match the confirmation")
    observed = {
        str(path.relative_to(input_dir))
        for path in (input_dir / "experiments").rglob("results.pkl")
    }
    if observed != set(artifacts):
        raise RuntimeError("on-disk result set does not match the attestation")
    for relative, raw_metadata in artifacts.items():
        if not isinstance(relative, str) or Path(relative).name != "results.pkl":
            raise RuntimeError("attested result has an unsafe filename")
        _artifact_bytes(
            input_dir,
            relative,
            hardened._as_mapping(raw_metadata, f"result artifact {relative}"),
            f"result {relative}",
        )

    payload_artifact = hardened._as_mapping(
        attestation.get("analysis_payload_artifact"), "analysis payload artifact"
    )
    if set(payload_artifact) != {"path", "sha256", "size_bytes"}:
        raise RuntimeError("analysis payload attestation fields are not exact")
    if payload_artifact.get("path") != campaign.ANALYSIS_PAYLOAD_FILENAME:
        raise RuntimeError("analysis payload path is not frozen")
    payload_bytes = _artifact_bytes(
        input_dir,
        campaign.ANALYSIS_PAYLOAD_FILENAME,
        {key: payload_artifact[key] for key in ("sha256", "size_bytes")},
        "safe analysis payload",
    )
    try:
        payload = dict(
            hardened._as_mapping(json.loads(payload_bytes), "safe analysis payload")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("safe analysis payload is not valid JSON") from exc
    if set(payload) != {
        "schema_version",
        "kind",
        "protocol_sha256",
        "result_artifacts_sha256",
        "job_order_sha256",
        "outer_rows",
        "child_rows",
    }:
        raise RuntimeError("safe analysis payload fields are not exact")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != campaign.PAYLOAD_KIND
        or payload.get("protocol_sha256") != protocol_digest
        or payload.get("result_artifacts_sha256") != _sha256(_canonical_json(artifacts))
        or payload.get("job_order_sha256") != expected_order_digest
    ):
        raise RuntimeError("safe analysis payload does not bind the campaign")
    _validate_ordering_metadata(manifest, payload=payload)
    outer_rows, child_rows = load_safe_rows(payload, artifacts, child_cpus)
    payload["outer_rows"] = outer_rows
    payload["child_rows"] = child_rows

    warmup_history_digest = hardened._verify_history_artifact(
        input_dir,
        attestation,
        attestation_field="warmup_history_artifact",
        filename=campaign.WARMUP_HISTORY_FILENAME,
        required=True,
        validator=lambda value: campaign.followon._validate_followon_warmup_history(
            value,
            expected_thread_count=child_cpus,
            expected_latest_pid=hardened._exact_int(
                attestation.get("pid"), "completion pid"
            ),
        ),
    )
    resume_history_digest = hardened._verify_history_artifact(
        input_dir,
        attestation,
        attestation_field="resume_history_artifact",
        filename=campaign.RESUME_HISTORY_FILENAME,
        required=False,
        validator=lambda value: campaign._validate_resume_history(value, input_dir),
    )
    validation = hardened._as_mapping(
        attestation.get("validation"), "completion validation"
    )
    if set(validation) != {
        "result_count",
        "child_fit_count",
        "paired_comparison_count",
        "independent_contrast_count",
        "native_representation_pair_count",
        "failure_count",
        "deadline_hit_count",
        "time_limit_stop_count",
        "stop_reason_counts",
        "job_order_sha256",
        "resource_allocation",
    }:
        raise RuntimeError("completion validation fields are not exact")
    expected_validation = {
        "result_count": campaign.EXPECTED_JOBS,
        "child_fit_count": campaign.EXPECTED_CHILD_FITS,
        "paired_comparison_count": campaign.EXPECTED_PAIRED_COMPARISONS,
        "independent_contrast_count": campaign.EXPECTED_INDEPENDENT_CONTRASTS,
        "native_representation_pair_count": campaign.EXPECTED_NATIVE_REPRESENTATION_PAIRS,
        "failure_count": 0,
        "deadline_hit_count": 0,
        "time_limit_stop_count": 0,
    }
    if any(validation.get(key) != value for key, value in expected_validation.items()):
        raise RuntimeError("completion validation counts do not match the campaign")
    if validation.get("job_order_sha256") != expected_order_digest:
        raise RuntimeError("completion validation job-order digest does not match")
    resources = hardened._as_mapping(
        validation.get("resource_allocation"), "validation resource allocation"
    )
    if (
        resources.get("num_cpus") != child_cpus
        or resources.get("num_cpus_child") != child_cpus
        or resources.get("num_gpus") != 0
        or resources.get("num_gpus_child") != 0
    ):
        raise RuntimeError("completion resource allocation does not match")
    stop_counts = hardened._as_mapping(
        validation.get("stop_reason_counts"), "validation stop reasons"
    )
    expected_stop_counts = dict(
        sorted(Counter(row["stop_reason"] for row in child_rows).items())
    )
    if dict(stop_counts) != expected_stop_counts or stop_counts.get(
        "time_limit", 0
    ) != 0:
        raise RuntimeError(
            "completion stop-reason counts do not match validated child metadata"
        )
    return manifest, attestation, payload, {
        "manifest_sha256": _sha256(manifest_bytes),
        "attestation_sha256": _sha256(attestation_bytes),
        "analysis_payload_sha256": _sha256(payload_bytes),
        "warmup_history_sha256": warmup_history_digest,
        "resume_history_sha256": resume_history_digest,
        **execution,
    }


def _ratio_fields(metric: str, numerator: float, denominator: float) -> dict[str, float]:
    ratio = _positive(numerator, f"{metric} numerator") / _positive(
        denominator, f"{metric} denominator"
    )
    return {
        f"{metric}_ratio": ratio,
        f"{metric}_log_ratio": math.log(ratio),
        f"{metric}_pct": 100.0 * (ratio - 1.0),
    }


def pair_outer_rows(outer_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (
            str(row["dataset"]),
            int(row["repeat"]),
            int(row["fold"]),
            str(row["arm"]),
        ): row
        for row in outer_rows
    }
    if len(index) != len(outer_rows):
        raise RuntimeError("duplicate outer row before contrast pairing")
    paired: list[dict[str, Any]] = []
    for dataset, repeat, fold in campaign.expected_coordinates():
        for contrast_name, spec in CONTRAST_SPECS.items():
            numerator = index[(dataset, repeat, fold, spec["numerator"])]
            denominator = index[(dataset, repeat, fold, spec["denominator"])]
            row: dict[str, Any] = {
                "contrast": contrast_name,
                "contrast_code": spec["code"],
                "contrast_role": spec["role"],
                "numerator_arm": spec["numerator"],
                "denominator_arm": spec["denominator"],
                "dataset": dataset,
                "task_id": campaign.TASKS[dataset],
                "repeat": repeat,
                "fold": fold,
                "registered_fold": 3 * repeat + fold,
            }
            for metric in METRICS:
                numerator_value = float(numerator[metric])
                denominator_value = float(denominator[metric])
                row[f"numerator_{metric}"] = numerator_value
                row[f"denominator_{metric}"] = denominator_value
                row.update(_ratio_fields(metric, numerator_value, denominator_value))
            paired.append(row)
    expected = len(campaign.expected_coordinates()) * len(CONTRAST_SPECS)
    if len(paired) != expected:
        raise RuntimeError("paired split contrast grid is not exact")
    return paired


def pair_child_rows(child_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (
            str(row["dataset"]),
            int(row["repeat"]),
            int(row["fold"]),
            int(row["child_fold"]),
            str(row["arm"]),
        ): row
        for row in child_rows
    }
    if len(index) != len(child_rows):
        raise RuntimeError("duplicate child row before contrast pairing")
    paired: list[dict[str, Any]] = []
    metadata_fields = (
        "iterations_requested",
        "iterations_attempted",
        "rounds_retained",
        "best_iteration",
        "rounds_completed",
        "resolved_learning_rate",
        "early_stopping_rounds",
        "requested_tree_mode",
        "stop_reason",
        "deadline_hit",
        "deadline_is_soft",
        "wall_clock_limit_seconds",
        "wall_clock_safety_margin_seconds",
        "wall_clock_effective_seconds",
        "wall_clock_elapsed_seconds",
        "selected_tree_mode",
        "selected_lane",
        "linear_residual_active",
    )
    for dataset, repeat, fold in campaign.expected_coordinates():
        for child_fold in range(8):
            for contrast_name, spec in CONTRAST_SPECS.items():
                numerator = index[
                    (dataset, repeat, fold, child_fold, spec["numerator"])
                ]
                denominator = index[
                    (dataset, repeat, fold, child_fold, spec["denominator"])
                ]
                row: dict[str, Any] = {
                    "contrast": contrast_name,
                    "contrast_code": spec["code"],
                    "dataset": dataset,
                    "task_id": campaign.TASKS[dataset],
                    "repeat": repeat,
                    "fold": fold,
                    "registered_fold": 3 * repeat + fold,
                    "child": f"S1F{child_fold + 1}",
                    "child_fold": child_fold,
                    "numerator_arm": spec["numerator"],
                    "denominator_arm": spec["denominator"],
                    "numerator_representation_kind": numerator["representation"]["kind"],
                    "denominator_representation_kind": denominator["representation"]["kind"],
                    "external_feature_schema_sha256": campaign.followon._feature_schema_sha256(
                        list(numerator["child_features"]),
                        "paired child external features",
                    ),
                    "numerator_initial_hyperparameters_json": json.dumps(
                        numerator["initial_hyperparameters"],
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "denominator_initial_hyperparameters_json": json.dumps(
                        denominator["initial_hyperparameters"],
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "numerator_effective_hyperparameters_json": json.dumps(
                        numerator["effective_hyperparameters"],
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "denominator_effective_hyperparameters_json": json.dumps(
                        denominator["effective_hyperparameters"],
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "numerator_refit_params_json": json.dumps(
                        numerator["refit_params"],
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "denominator_refit_params_json": json.dumps(
                        denominator["refit_params"],
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "numerator_representation_json": json.dumps(
                        numerator["representation"],
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "denominator_representation_json": json.dumps(
                        denominator["representation"],
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                }
                for field in metadata_fields:
                    row[f"numerator_{field}"] = numerator[field]
                    row[f"denominator_{field}"] = denominator[field]
                paired.append(row)
    expected = len(campaign.expected_coordinates()) * 8 * len(CONTRAST_SPECS)
    if len(paired) != expected:
        raise RuntimeError("paired child contrast grid is not exact")
    return paired


def _nested_log_values(
    split_rows: Sequence[Mapping[str, Any]], metric_log_key: str
) -> dict[str, dict[int, list[float]]]:
    nested: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    seen = set()
    contrast_codes = {str(row.get("contrast_code")) for row in split_rows}
    if len(contrast_codes) != 1:
        raise RuntimeError("hierarchical aggregation requires exactly one contrast")
    for row in split_rows:
        key = (str(row["dataset"]), int(row["repeat"]), int(row["fold"]))
        if key in seen:
            raise RuntimeError(f"duplicate paired split for {key}")
        seen.add(key)
        value = float(row[metric_log_key])
        if not math.isfinite(value):
            raise RuntimeError(f"nonfinite paired log ratio for {key}")
        nested[key[0]][key[1]].append(value)
    expected_datasets = set(campaign.TASKS)
    if set(nested) != expected_datasets:
        raise RuntimeError("paired contrast does not contain both frozen datasets")
    expected_coordinates = set(campaign.expected_coordinates())
    if seen != expected_coordinates:
        raise RuntimeError("paired contrast coordinates are not exact")
    return {dataset: dict(repeats) for dataset, repeats in nested.items()}


def hierarchical_point_log_ratio(
    split_rows: Sequence[Mapping[str, Any]], metric_log_key: str
) -> tuple[float, dict[str, float], dict[str, dict[int, float]]]:
    """Average folds within repeats, repeats within datasets, then datasets 50/50."""
    nested = _nested_log_values(split_rows, metric_log_key)
    repeat_logs: dict[str, dict[int, float]] = {}
    dataset_logs: dict[str, float] = {}
    for dataset in sorted(nested):
        repeat_logs[dataset] = {
            repeat: math.fsum(folds) / len(folds)
            for repeat, folds in sorted(nested[dataset].items())
        }
        values = list(repeat_logs[dataset].values())
        dataset_logs[dataset] = math.fsum(values) / len(values)
    if len(dataset_logs) != 2:
        raise RuntimeError("frozen confirmation requires exactly two datasets")
    overall = 0.5 * math.fsum(dataset_logs.values())
    return overall, dataset_logs, repeat_logs


def hierarchical_bootstrap_log_ratios(
    split_rows: Sequence[Mapping[str, Any]],
    metric_log_key: str = "test_rmse_log_ratio",
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Keep datasets fixed; resample repeats, then folds within repeats."""
    if draws <= 0:
        raise ValueError("draws must be positive")
    nested = _nested_log_values(split_rows, metric_log_key)
    datasets = sorted(nested)
    rng = np.random.default_rng(seed)
    output = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        dataset_draws = []
        for dataset in datasets:
            repeats = nested[dataset]
            repeat_ids = sorted(repeats)
            sampled_repeat_indices = rng.integers(
                0, len(repeat_ids), size=len(repeat_ids)
            )
            repeat_draws = []
            for repeat_index in sampled_repeat_indices:
                folds = np.asarray(
                    repeats[repeat_ids[int(repeat_index)]], dtype=np.float64
                )
                sampled_fold_indices = rng.integers(0, len(folds), size=len(folds))
                repeat_draws.append(float(np.mean(folds[sampled_fold_indices])))
            dataset_draws.append(math.fsum(repeat_draws) / len(repeat_draws))
        output[draw] = 0.5 * math.fsum(dataset_draws)
    return output


def exact_one_sided_sign_test_pvalue(wins: int, nonwins: int) -> float:
    """Return the frozen Binomial(13, .5) upper tail; ties are non-wins."""
    if wins < 0 or nonwins < 0:
        raise ValueError("wins and nonwins must be nonnegative")
    if wins + nonwins != 13:
        raise ValueError("the frozen sign test requires exactly 13 repeat blocks")
    return math.fsum(math.comb(13, value) for value in range(wins, 14)) / (2**13)


def _ratio_summary(log_ratio: float) -> dict[str, float]:
    ratio = math.exp(log_ratio)
    return {
        "log_ratio": log_ratio,
        "ratio": ratio,
        "pct": 100.0 * (ratio - 1.0),
    }


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": _quantile(array, 0.90),
        "max": float(np.max(array)),
    }


def _contrast_gates(
    code: str,
    metrics: Mapping[str, Mapping[str, float]],
    datasets: Sequence[Mapping[str, Any]],
    bootstrap_upper95: float,
    sign_p: float,
    *,
    campaign_clean: bool,
) -> dict[str, bool] | None:
    thresholds = GATE_THRESHOLDS.get(code)
    if thresholds is None:
        return None
    gates = {
        "complete_failure_free_campaign": campaign_clean,
        "test_ratio_at_most_0_995": metrics["test_rmse"]["ratio"]
        <= thresholds["test_ratio_max"],
        "one_sided_bootstrap_upper95_below_1": bootstrap_upper95
        < thresholds["bootstrap_upper95_max"],
        "dataset_test_guardrail": max(item["test_rmse_ratio"] for item in datasets)
        <= thresholds["each_dataset_test_ratio_max"],
        "repeat_block_sign_test_p_below_0_05": sign_p
        < thresholds["sign_test_p_max"],
        "validation_ratio_at_most_1_002": metrics["val_rmse"]["ratio"]
        <= thresholds["validation_ratio_max"],
        "train_time_ratio_at_most_1_5": metrics["train_time_s"]["ratio"]
        <= thresholds["train_time_ratio_max"],
        "infer_time_ratio_at_most_1_25": metrics["infer_time_s"]["ratio"]
        <= thresholds["infer_time_ratio_max"],
        "peak_memory_ratio_at_most_1_25": metrics["peak_memory_bytes"]["ratio"]
        <= thresholds["peak_memory_ratio_max"],
    }
    if code == "O/B":
        gates["each_dataset_validation_ratio_at_most_1_005"] = max(
            item["val_rmse_ratio"] for item in datasets
        ) <= thresholds["each_dataset_validation_ratio_max"]
    gates["advance"] = all(gates.values())
    return gates


def analyze_paired_rows(
    split_rows: Sequence[Mapping[str, Any]],
    child_pairs: Sequence[Mapping[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    campaign_clean: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contrast_summaries = []
    per_repeat: list[dict[str, Any]] = []
    for contrast_name, spec in CONTRAST_SPECS.items():
        selected = [row for row in split_rows if row["contrast"] == contrast_name]
        if len(selected) != len(campaign.expected_coordinates()):
            raise RuntimeError(f"{spec['code']} split panel is incomplete")
        metrics: dict[str, dict[str, float]] = {}
        metric_dataset_logs: dict[str, dict[str, float]] = {}
        repeat_logs: dict[str, dict[int, float]] | None = None
        for metric in METRICS:
            point, datasets_for_metric, repeats_for_metric = (
                hierarchical_point_log_ratio(selected, f"{metric}_log_ratio")
            )
            metrics[metric] = _ratio_summary(point)
            metric_dataset_logs[metric] = datasets_for_metric
            if metric == "test_rmse":
                repeat_logs = repeats_for_metric
        if repeat_logs is None:
            raise RuntimeError(f"{spec['code']} test-RMSE hierarchy is missing")
        bootstrap = hierarchical_bootstrap_log_ratios(
            selected, draws=draws, seed=seed
        )
        bootstrap_summary = {
            "draws": draws,
            "seed": seed,
            "datasets_resampled": False,
            "ratio_lower95_two_sided": math.exp(_quantile(bootstrap, 0.025)),
            "ratio_upper95_two_sided": math.exp(_quantile(bootstrap, 0.975)),
            "ratio_upper95_one_sided": math.exp(_quantile(bootstrap, 0.95)),
        }
        datasets = []
        for dataset in sorted(metric_dataset_logs["test_rmse"]):
            dataset_rows = [row for row in selected if row["dataset"] == dataset]
            worst = max(dataset_rows, key=lambda row: float(row["test_rmse_ratio"]))
            datasets.append(
                {
                    "dataset": dataset,
                    "split_count": len(dataset_rows),
                    "repeat_count": len(repeat_logs[dataset]),
                    "test_rmse_ratio": math.exp(
                        metric_dataset_logs["test_rmse"][dataset]
                    ),
                    "test_rmse_pct": 100.0
                    * (
                        math.exp(metric_dataset_logs["test_rmse"][dataset]) - 1.0
                    ),
                    "val_rmse_ratio": math.exp(metric_dataset_logs["val_rmse"][dataset]),
                    "val_rmse_pct": 100.0
                    * (math.exp(metric_dataset_logs["val_rmse"][dataset]) - 1.0),
                    "split_wins": sum(row["test_rmse_ratio"] < 1.0 for row in dataset_rows),
                    "split_losses": sum(row["test_rmse_ratio"] > 1.0 for row in dataset_rows),
                    "split_ties": sum(row["test_rmse_ratio"] == 1.0 for row in dataset_rows),
                    "worst_split_dataset": dataset,
                    "worst_split_repeat": int(worst["repeat"]),
                    "worst_split_fold": int(worst["fold"]),
                    "worst_split_ratio": float(worst["test_rmse_ratio"]),
                }
            )
        repeat_values = []
        for dataset in sorted(repeat_logs):
            for repeat, log_ratio in sorted(repeat_logs[dataset].items()):
                rows_for_repeat = [
                    row
                    for row in selected
                    if row["dataset"] == dataset and int(row["repeat"]) == repeat
                ]
                val_log = math.fsum(
                    float(row["val_rmse_log_ratio"]) for row in rows_for_repeat
                ) / len(rows_for_repeat)
                item = {
                    "contrast": contrast_name,
                    "contrast_code": spec["code"],
                    "dataset": dataset,
                    "repeat": repeat,
                    "fold_count": len(rows_for_repeat),
                    **_ratio_summary(log_ratio),
                    "val_rmse_ratio": math.exp(val_log),
                    "val_rmse_pct": 100.0 * (math.exp(val_log) - 1.0),
                    "fold_wins": sum(row["test_rmse_ratio"] < 1.0 for row in rows_for_repeat),
                    "fold_losses": sum(row["test_rmse_ratio"] > 1.0 for row in rows_for_repeat),
                    "fold_ties": sum(row["test_rmse_ratio"] == 1.0 for row in rows_for_repeat),
                }
                repeat_values.append(item)
                per_repeat.append(item)
        if len(repeat_values) != 13:
            raise RuntimeError("confirmation must contain exactly 13 repeat blocks")
        wins = sum(item["ratio"] < 1.0 for item in repeat_values)
        losses = sum(item["ratio"] > 1.0 for item in repeat_values)
        ties = 13 - wins - losses
        sign_p = exact_one_sided_sign_test_pvalue(wins, losses + ties)
        worst_coordinate = max(
            selected, key=lambda row: float(row["test_rmse_ratio"])
        )
        children = [row for row in child_pairs if row["contrast"] == contrast_name]
        time_stops = sum(
            row[side + "_stop_reason"] == "time_limit"
            for row in children
            for side in ("numerator", "denominator")
        )
        deadline_hits = sum(
            row[side + "_deadline_hit"] is True
            for row in children
            for side in ("numerator", "denominator")
        )
        contrast_clean = campaign_clean and time_stops == 0 and deadline_hits == 0
        gates = _contrast_gates(
            spec["code"],
            metrics,
            datasets,
            bootstrap_summary["ratio_upper95_one_sided"],
            sign_p,
            campaign_clean=contrast_clean,
        )
        contrast_summaries.append(
            {
                "contrast": contrast_name,
                **spec,
                "paired_splits": len(selected),
                "paired_children": len(children),
                "metrics": metrics,
                "test_bootstrap": bootstrap_summary,
                "datasets": datasets,
                "repeat_blocks": {
                    "count": len(repeat_values),
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "one_sided_sign_test_p": sign_p,
                },
                "split_counts": {
                    "wins": sum(row["test_rmse_ratio"] < 1.0 for row in selected),
                    "losses": sum(row["test_rmse_ratio"] > 1.0 for row in selected),
                    "ties": sum(row["test_rmse_ratio"] == 1.0 for row in selected),
                },
                "worst_coordinate_diagnostic": {
                    "dataset": str(worst_coordinate["dataset"]),
                    "repeat": int(worst_coordinate["repeat"]),
                    "fold": int(worst_coordinate["fold"]),
                    "test_rmse_ratio": float(worst_coordinate["test_rmse_ratio"]),
                },
                "child_metadata": {
                    "time_limit_stops": time_stops,
                    "deadline_hits": deadline_hits,
                    "numerator_stop_reason_counts": dict(
                        sorted(Counter(row["numerator_stop_reason"] for row in children).items())
                    ),
                    "denominator_stop_reason_counts": dict(
                        sorted(Counter(row["denominator_stop_reason"] for row in children).items())
                    ),
                    "numerator_best_iteration": _distribution(
                        [float(row["numerator_best_iteration"]) for row in children]
                    ),
                    "denominator_best_iteration": _distribution(
                        [float(row["denominator_best_iteration"]) for row in children]
                    ),
                    "numerator_representation_counts": dict(
                        sorted(Counter(row["numerator_representation_kind"] for row in children).items())
                    ),
                    "denominator_representation_counts": dict(
                        sorted(Counter(row["denominator_representation_kind"] for row in children).items())
                    ),
                },
                "gates": gates,
                "decision": (
                    "report_only"
                    if gates is None
                    else ("pass" if gates["advance"] else "fail")
                ),
            }
        )
    by_code = {item["code"]: item for item in contrast_summaries}
    advance = bool(
        by_code["O/B"]["gates"]["advance"]
        and by_code["O/P"]["gates"]["advance"]
    )
    primary_children = [
        row for row in child_pairs if row["contrast_code"] == "O/B"
    ]
    deployment_children = [
        row for row in child_pairs if row["contrast_code"] == "O/P"
    ]
    if len(primary_children) != 264 or len(deployment_children) != 264:
        raise RuntimeError("representation activity child panels are incomplete")
    ordinal_representations = [
        json.loads(row["numerator_representation_json"])
        for row in primary_children
    ]
    fixed_native_representations = [
        json.loads(row["denominator_representation_json"])
        for row in primary_children
    ]
    product_native_representations = [
        json.loads(row["denominator_representation_json"])
        for row in deployment_children
    ]
    ordinal_activity_count = sum(
        representation.get("kind") == "safe_ordinal"
        and representation.get("mapping_source") == "source_frozen_before_campaign"
        and representation.get("target_used_by_representation") is False
        and representation.get("fit_calls") == 1
        for representation in ordinal_representations
    )
    ordinal_unknown_count = sum(
        sum(representation["eval_unknown_counts"])
        for representation in ordinal_representations
    )
    fixed_native_activity_count = sum(
        representation.get("kind") == "native"
        and representation.get("target_used_by_representation") is True
        for representation in fixed_native_representations
    )
    product_native_activity_count = sum(
        representation.get("kind") == "native"
        and representation.get("target_used_by_representation") is True
        for representation in product_native_representations
    )
    if (
        ordinal_activity_count != 264
        or ordinal_unknown_count != 0
        or fixed_native_activity_count != 264
        or product_native_activity_count != 264
    ):
        raise RuntimeError("paired representation activity diagnostics changed")
    representation_activity = {
        "cross_arm_external_schema_match_count": 264,
        "fixed_base_safe_ordinal": {
            "child_count": len(primary_children),
            "representation_counts": dict(
                sorted(
                    Counter(
                        row["numerator_representation_kind"]
                        for row in primary_children
                    ).items()
                )
            ),
            "dataset_child_counts": dict(
                sorted(Counter(row["dataset"] for row in primary_children).items())
            ),
            "target_free_source_declared_activity_count": ordinal_activity_count,
            "unknown_validation_value_count": ordinal_unknown_count,
        },
        "fixed_base_native": {
            "child_count": len(primary_children),
            "representation_counts": dict(
                sorted(
                    Counter(
                        row["denominator_representation_kind"]
                        for row in primary_children
                    ).items()
                )
            ),
            "native_target_stat_activity_count": fixed_native_activity_count,
        },
        "product_default_native": {
            "child_count": len(deployment_children),
            "representation_counts": dict(
                sorted(
                    Counter(
                        row["denominator_representation_kind"]
                        for row in deployment_children
                    ).items()
                )
            ),
            "native_target_stat_activity_count": product_native_activity_count,
        },
    }
    return {
        "protocol": (
            "frozen safe-ordinal replication on mechanism-unused TabArena "
            "coordinates previously evaluated by the cap campaign"
        ),
        "bootstrap": {
            "draws": draws,
            "seed": seed,
            "datasets_fixed": True,
            "resampling_order": "repeats_within_dataset_then_folds_within_repeat",
            "dataset_weights": {
                "airfoil_self_noise": 0.5,
                "diamonds": 0.5,
            },
        },
        "thresholds": GATE_THRESHOLDS,
        "counts": {
            "datasets": 2,
            "dataset_repeat_blocks": 13,
            "outer_jobs": campaign.EXPECTED_JOBS,
            "child_fits": campaign.EXPECTED_CHILD_FITS,
            "coordinates": len(campaign.expected_coordinates()),
            "contrast_split_rows": len(split_rows),
            "contrast_child_rows": len(child_pairs),
        },
        "contrasts": contrast_summaries,
        "representation_activity": representation_activity,
        "advance_safe_ordinal_policy": advance,
        "decision": "advance" if advance else "do_not_advance",
    }, per_repeat


def _csv_bytes(rows: Sequence[Mapping[str, Any]], field: str) -> bytes:
    if not rows:
        raise RuntimeError(f"refusing to write empty {field}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def render_markdown_report(summary: Mapping[str, Any]) -> str:
    decision = (
        "ADVANCE the explicit safe-ordinal policy"
        if summary["advance_safe_ordinal_policy"]
        else "DO NOT ADVANCE the safe-ordinal policy"
    )
    lines = [
        "# TabArena safe-ordinal confirmation",
        "",
        "This frozen replication uses only the 33 mechanism-unused Airfoil and "
        "Diamonds coordinates. The cap campaign previously evaluated these "
        "coordinates, so this is mechanism replication rather than independent "
        "dataset generalization. Ratios below one favor the numerator. Datasets "
        "remain fixed at 50/50 weight in both the estimand and bootstrap.",
        "",
        "| Contrast | Role | Test RMSE | Upper 95% | Validation | Train | Infer | RSS | Decision |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summary["contrasts"]:
        metrics = item["metrics"]
        lines.append(
            f"| {item['code']} | {item['role']} | "
            f"{metrics['test_rmse']['ratio']:.6f} | "
            f"{item['test_bootstrap']['ratio_upper95_one_sided']:.6f} | "
            f"{metrics['val_rmse']['ratio']:.6f} | "
            f"{metrics['train_time_s']['ratio']:.4f} | "
            f"{metrics['infer_time_s']['ratio']:.4f} | "
            f"{metrics['peak_memory_bytes']['ratio']:.4f} | "
            f"{item['decision']} |"
        )
    for item in summary["contrasts"]:
        lines.extend(
            [
                "",
                f"## {item['code']}: {item['contrast']}",
                "",
                f"Role: `{item['role']}`. Repeat-block wins/losses/ties: "
                f"{item['repeat_blocks']['wins']}/{item['repeat_blocks']['losses']}/"
                f"{item['repeat_blocks']['ties']}; exact one-sided sign-test "
                f"p={item['repeat_blocks']['one_sided_sign_test_p']:.6g}.",
                "",
                "Split wins/losses/ties: "
                f"{item['split_counts']['wins']}/{item['split_counts']['losses']}/"
                f"{item['split_counts']['ties']}. Worst coordinate: "
                f"{item['worst_coordinate_diagnostic']['dataset']} "
                f"r{item['worst_coordinate_diagnostic']['repeat']}"
                f"f{item['worst_coordinate_diagnostic']['fold']} at ratio "
                f"{item['worst_coordinate_diagnostic']['test_rmse_ratio']:.6f} "
                "(diagnostic only).",
                "",
                "| Dataset | Splits | Repeats | Test ratio | Validation ratio |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for dataset in item["datasets"]:
            lines.append(
                f"| {dataset['dataset']} | {dataset['split_count']} | "
                f"{dataset['repeat_count']} | {dataset['test_rmse_ratio']:.6f} | "
                f"{dataset['val_rmse_ratio']:.6f} |"
            )
        lines.extend(["", "Gates:", ""])
        if item["gates"] is None:
            lines.append("- Report only; no advancement gate was declared.")
        else:
            for name, passed in item["gates"].items():
                if name != "advance":
                    lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    activity = summary["representation_activity"]
    lines.extend(
        [
            "",
            "## Representation activity",
            "",
            f"- Safe ordinal: {activity['fixed_base_safe_ordinal']['child_count']} "
            "children; target-free source-declared transform active in "
            f"{activity['fixed_base_safe_ordinal']['target_free_source_declared_activity_count']}; "
            f"unknown validation values "
            f"{activity['fixed_base_safe_ordinal']['unknown_validation_value_count']}.",
            f"- Fixed native: {activity['fixed_base_native']['child_count']} "
            "children with native target-stat representation active.",
            f"- Product native: {activity['product_default_native']['child_count']} "
            "children with native target-stat representation active.",
            f"- Exact ordered external schemas matched across P/B/O for "
            f"{activity['cross_arm_external_schema_match_count']} child blocks.",
            "",
            "## Integrity",
            "",
            f"- Outer jobs: {summary['counts']['outer_jobs']}/99.",
            f"- Child fits: {summary['counts']['child_fits']}/792.",
            "- Zero missing, failed, imputed, deadline-hit, or time-limit results.",
            "- Raw result files were verified only as opaque byte artifacts; this "
            "analyzer never unpickled them.",
            "- Exact source, dependency, runtime, hardware, schema, representation, "
            "and job-order provenance matched the completion attestation.",
            "",
            "## Provenance",
            "",
            f"- Git commit: `{summary['provenance']['git_head']}`.",
            f"- Frozen protocol semantic SHA-256: `{summary['provenance']['protocol_sha256']}`.",
            f"- Ordered-grid SHA-256: `{summary['provenance']['job_order_sha256']}`.",
            f"- Manifest SHA-256: `{summary['provenance']['manifest_sha256']}`.",
            f"- Completion attestation SHA-256: "
            f"`{summary['provenance']['attestation_sha256']}`.",
            f"- Safe analysis payload SHA-256: "
            f"`{summary['provenance']['analysis_payload_sha256']}`.",
            "",
            "## Decision",
            "",
            f"**{decision}.**",
        ]
    )
    return "\n".join(lines)


def _publish_outputs_atomically(
    outputs: Mapping[str, Path],
    payloads: Mapping[str, bytes],
    *,
    post_write_check,
) -> None:
    if set(outputs) != set(OUTPUT_KEYS) or set(payloads) != set(OUTPUT_KEYS):
        raise RuntimeError("managed analysis output fields are not exact")
    hardened._atomic_write_group(
        [(outputs[name], payloads[name]) for name in OUTPUT_KEYS],
        post_write_check=post_write_check,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve(strict=True)
    manifest_path = (input_dir / campaign.MANIFEST_FILENAME).resolve(strict=True)
    attestation_path = (
        input_dir / campaign.COMPLETION_ATTESTATION_FILENAME
    ).resolve(strict=True)
    manifest, attestation, payload, digests = verify_campaign_integrity(input_dir)
    outputs_requested = {
        key: input_dir / name for key, name in zip(OUTPUT_KEYS, OUTPUT_NAMES)
    }
    protected = hardened._protected_campaign_paths(
        input_dir,
        manifest_path=manifest_path,
        attestation_path=attestation_path,
        attestation=attestation,
    )
    outputs = hardened._canonical_output_targets(
        input_dir, outputs_requested, protected_paths=protected
    )
    paired_splits = pair_outer_rows(payload["outer_rows"])
    paired_children = pair_child_rows(payload["child_rows"])
    validation = hardened._as_mapping(
        attestation["validation"], "completion validation"
    )
    campaign_clean = all(
        validation.get(field) == 0
        for field in ("failure_count", "deadline_hit_count", "time_limit_stop_count")
    )
    summary, per_repeat = analyze_paired_rows(
        paired_splits, paired_children, campaign_clean=campaign_clean
    )
    if summary["counts"] != {
        "datasets": 2,
        "dataset_repeat_blocks": 13,
        "outer_jobs": 99,
        "child_fits": 792,
        "coordinates": 33,
        "contrast_split_rows": 99,
        "contrast_child_rows": 792,
    }:
        raise RuntimeError("analysis counts do not match the frozen campaign")
    summary["integrity_diagnostics"] = {
        "validation_basis": (
            "completion attestation, opaque raw-result hashes, runner-normalized "
            "safe JSON, and analyzer exact-grid/schema/order revalidation"
        ),
        "expected_outer_jobs": 99,
        "observed_outer_jobs": 99,
        "missing_outer_jobs": 0,
        "failed_outer_jobs": 0,
        "imputed_outer_jobs": 0,
        "duplicate_outer_jobs": 0,
        "expected_child_fits": 792,
        "observed_child_fits": 792,
        "missing_child_metadata": 0,
        "duplicate_child_metadata": 0,
        "deadline_hit_children": 0,
        "time_limit_children": 0,
        "representation_activity_verified": True,
        "job_order_verified": True,
    }
    summary["provenance"] = {
        **digests,
        "manifest_path": str(manifest_path),
        "attestation_path": str(attestation_path),
        "protocol_sha256": campaign.protocol_sha256(),
        "job_order_sha256": campaign.job_order_sha256(),
        "git_head": manifest["source"]["git_head"],
        "completed_at_utc": attestation.get("completed_at_utc"),
    }
    payloads = {
        "split_csv": _csv_bytes(paired_splits, "paired split CSV"),
        "repeat_csv": _csv_bytes(per_repeat, "per-repeat CSV"),
        "child_csv": _csv_bytes(paired_children, "paired child CSV"),
        "summary_json": (
            json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "report_md": render_markdown_report(summary).encode("utf-8"),
    }
    outputs = hardened._canonical_output_targets(
        input_dir, outputs, protected_paths=protected
    )
    baseline_snapshot = (manifest, attestation, payload, digests)

    def assert_campaign_unchanged() -> None:
        if verify_campaign_integrity(input_dir) != baseline_snapshot:
            raise RuntimeError("campaign artifacts changed during analysis")

    assert_campaign_unchanged()
    _publish_outputs_atomically(
        outputs, payloads, post_write_check=assert_campaign_unchanged
    )
    print(
        f"analyzed {campaign.EXPECTED_JOBS} jobs and "
        f"{campaign.EXPECTED_CHILD_FITS} child fits; "
        f"advance={summary['advance_safe_ordinal_policy']}; "
        f"wrote {outputs['summary_json']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
