#!/usr/bin/env python3
"""Create the eligibility-aware, outcome-blind panel-3 target attestation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import build_ctr23_contamination_registry as ctr  # noqa: E402
from benchmarks import confirmation_target_preflight as target_check  # noqa: E402
from benchmarks import build_panel3_power_design as power_design  # noqa: E402
from benchmarks import panel3_registry_common as common  # noqa: E402
from benchmarks.campaign_lib import provenance  # noqa: E402


DEFAULT_OUTPUT = ROOT / "benchmarks" / "panel3_target_preflight.json"
COORDINATE_FOLDS = (0, 1, 2)


def _git(*args: str) -> str:
    return provenance.git_output(ROOT, *args)


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )


def _require_frozen_clean_source(
    declarations: dict[str, Any],
    source_sha256: dict[str, str],
) -> str:
    head = _git("rev-parse", "HEAD")
    prefreeze = str(declarations["darkofit_prefreeze_head"])
    if head == prefreeze:
        raise RuntimeError(
            "panel-3 preflight sources must be committed before execution"
        )
    if not _is_ancestor(prefreeze, head):
        raise RuntimeError(
            "panel-3 execution head does not descend from prefreeze"
        )
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("panel-3 target preflight requires a clean tree")
    expected_paths = {
        str(path.relative_to(ROOT)) for path in common.PANEL3_SOURCE_PATHS
    }
    if set(source_sha256) != expected_paths:
        raise RuntimeError("panel-3 target-preflight source snapshot changed")
    for relative, digest in source_sha256.items():
        try:
            committed = subprocess.run(
                ["git", "show", f"{head}:{relative}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
            ).stdout
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"panel-3 preflight source is not committed: {relative}"
            ) from exc
        if hashlib.sha256(committed).hexdigest() != digest:
            raise RuntimeError(
                f"panel-3 preflight source differs from HEAD: {relative}"
            )
    return head


def _recheck_authorization_inputs(
    *,
    decision: dict[str, Any],
    decision_file_sha256: str,
    source_sha256: dict[str, str],
    execution_head: str | None = None,
    require_clean_source: bool = False,
) -> None:
    current_decision, current_decision_file_sha256 = (
        common.secure_load_json(common.POWER_DESIGN_DECISION)
    )
    if (
        current_decision != decision
        or current_decision_file_sha256 != decision_file_sha256
    ):
        raise RuntimeError(
            "panel-3 power decision changed before preflight publication"
        )
    common.recheck_snapshot_files(
        list(common.PANEL3_SOURCE_PATHS),
        source_sha256,
    )
    if execution_head is not None and _git("rev-parse", "HEAD") != (
        execution_head
    ):
        raise RuntimeError(
            "panel-3 source head changed before preflight publication"
        )
    if require_clean_source and _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError(
            "panel-3 tree changed before preflight publication"
        )


def _load_task_record(task_id: int) -> dict[str, Any]:
    return ctr._task_record(int(task_id), include_splits=False)


def _official_split_geometry(
    task: Any,
    declaration: dict[str, Any],
) -> tuple[int, list[int]]:
    """Read split sizes without loading any dataset column."""
    expected = declaration["expected_split_dimensions"]
    observed = tuple(int(value) for value in task.get_split_dimensions())
    if observed != (
        int(expected["repeats"]),
        int(expected["folds"]),
        int(expected["samples"]),
    ):
        raise RuntimeError(
            f"task {declaration['task_id']} split dimensions changed"
        )
    row_count = None
    training_rows = []
    for fold in COORDINATE_FOLDS:
        train, test = task.get_train_test_split_indices(
            repeat=0,
            fold=int(fold),
            sample=0,
        )
        train = np.asarray(train, dtype="<i8")
        test = np.asarray(test, dtype="<i8")
        covered = np.sort(np.concatenate([train, test]))
        if (
            train.ndim != 1
            or test.ndim != 1
            or train.size == 0
            or test.size == 0
            or np.any(train < 0)
            or np.any(test < 0)
            or np.unique(train).size != train.size
            or np.unique(test).size != test.size
            or np.intersect1d(train, test).size
            or not np.array_equal(
                covered,
                np.arange(covered.size, dtype="<i8"),
            )
        ):
            raise RuntimeError(
                f"task {declaration['task_id']} official split is invalid"
            )
        if row_count is None:
            row_count = int(covered.size)
        elif row_count != int(covered.size):
            raise RuntimeError(
                f"task {declaration['task_id']} split row count changed"
            )
        training_rows.append(int(train.size))
    if row_count is None:
        raise RuntimeError(
            f"task {declaration['task_id']} has no structural split evidence"
        )
    return row_count, training_rows


def _target_free_split_features(
    task: Any,
    declaration: dict[str, Any],
    *,
    row_count: int,
) -> pd.DataFrame:
    """Materialize only feature columns required by a custom split.

    OpenML may first cache a container that physically includes the target
    column.  The Parquet projection below is the enforceable outcome-blind
    boundary: target values are never decoded, materialized, or inspected by
    this structural pass.
    """
    constructor = declaration["split_policy"]["constructor"]
    if constructor["kind"] == "size_balanced_group_kfold_v1":
        columns = list(constructor["group_key"]["source_columns"])
    elif constructor["kind"] == "expanding_unique_datetime_blocks_v1":
        columns = [constructor["source_column"]]
    else:
        raise RuntimeError(
            f"task {declaration['task_id']} split projection is unsupported"
        )
    if (
        not columns
        or len(set(columns)) != len(columns)
        or any(not isinstance(column, str) or not column for column in columns)
        or str(task.target_name) in columns
    ):
        raise RuntimeError(
            f"task {declaration['task_id']} target-free split columns changed"
        )
    dataset = task.get_dataset(
        download_data=True,
        cache_format="feather",
        download_qualities=False,
        download_features_meta_data=False,
    )
    parquet_file = getattr(dataset, "parquet_file", None)
    path = Path(parquet_file) if parquet_file is not None else None
    if (
        int(dataset.dataset_id) != int(declaration["dataset_id"])
        or path is None
        or not path.is_file()
    ):
        raise RuntimeError(
            f"task {declaration['task_id']} target-free Parquet view is unavailable"
        )
    try:
        frame = pd.read_parquet(path, columns=columns)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise RuntimeError(
            f"task {declaration['task_id']} target-free split projection failed"
        ) from exc
    if (
        not isinstance(frame, pd.DataFrame)
        or list(frame.columns) != columns
        or len(frame) != row_count
    ):
        raise RuntimeError(
            f"task {declaration['task_id']} target-free split view changed"
        )
    return frame


def _custom_split_training_rows(
    task: Any,
    declaration: dict[str, Any],
    *,
    row_count: int,
) -> list[int]:
    """Construct the declared target-free folds and return train sizes only."""
    from benchmarks import build_panel3_registry as registry_builder

    constructor = declaration["split_policy"]["constructor"]
    kind = constructor["kind"]
    if kind == "shuffled_kfold_v1":
        pairs = registry_builder._shuffled_kfold_indices(
            row_count,
            constructor,
        )
    elif kind == "size_balanced_group_kfold_v1":
        pairs = registry_builder._group_kfold_indices(
            _target_free_split_features(
                task,
                declaration,
                row_count=row_count,
            ),
            constructor,
        )
    elif kind == "expanding_unique_datetime_blocks_v1":
        pairs = registry_builder._chronological_indices(
            _target_free_split_features(
                task,
                declaration,
                row_count=row_count,
            ),
            constructor,
        )
    else:
        raise RuntimeError(
            f"task {declaration['task_id']} split constructor is unsupported"
        )
    if len(pairs) != len(COORDINATE_FOLDS):
        raise RuntimeError(
            f"task {declaration['task_id']} custom split count changed"
        )
    allow_unused = constructor["allow_unused_rows"]
    test_parts = []
    training_rows = []
    for fold, (train, test) in zip(
        COORDINATE_FOLDS,
        pairs,
        strict=True,
    ):
        train = np.asarray(train, dtype="<i8")
        test = np.asarray(test, dtype="<i8")
        if (
            train.ndim != 1
            or test.ndim != 1
            or train.size == 0
            or test.size == 0
            or np.any(train < 0)
            or np.any(test < 0)
            or np.any(train >= row_count)
            or np.any(test >= row_count)
            or np.unique(train).size != train.size
            or np.unique(test).size != test.size
            or np.intersect1d(train, test).size
            or (
                not allow_unused
                and train.size + test.size != row_count
            )
        ):
            raise RuntimeError(
                f"task {declaration['task_id']} custom split fold {fold} "
                "is invalid"
            )
        test_parts.append(test)
        training_rows.append(int(train.size))
    all_test = np.sort(np.concatenate(test_parts))
    if (
        np.unique(all_test).size != all_test.size
        or (
            not allow_unused
            and not np.array_equal(
                all_test,
                np.arange(row_count, dtype="<i8"),
            )
        )
    ):
        raise RuntimeError(
            f"task {declaration['task_id']} custom test folds changed"
        )
    return training_rows


def _split_applicability_attestation(
    declaration: dict[str, Any],
    slot: dict[str, Any],
    *,
    minimum_rows: int,
) -> dict[str, Any]:
    """Bind one slot before any target-value materialization or inspection."""
    task_id = int(declaration["task_id"])
    if (
        set(slot)
        != {
            "lineage_cluster",
            "task_id",
            "stratum",
            "t5_size_gate_applicability",
        }
        or int(slot["task_id"]) != task_id
        or slot["lineage_cluster"] != declaration["lineage_cluster"]
        or slot["stratum"] != declaration["stratum"]
    ):
        raise RuntimeError(f"task {task_id} power-design slot changed")
    try:
        import openml
    except ImportError as exc:  # pragma: no cover - CLI dependency error.
        raise RuntimeError("openml is required for Panel 3 preflight") from exc
    task = openml.tasks.get_task(
        task_id,
        download_splits=True,
        download_data=False,
        download_qualities=False,
        download_features_meta_data=False,
    )
    task_type_id = getattr(task.task_type_id, "value", task.task_type_id)
    if (
        int(task.task_id) != task_id
        or int(task.dataset_id) != int(declaration["dataset_id"])
        or int(task_type_id) != 2
        or str(task.target_name)
        != str(declaration["expected_target_name"])
    ):
        raise RuntimeError(f"task {task_id} structural identity changed")
    row_count, official_training_rows = _official_split_geometry(
        task,
        declaration,
    )
    policy = declaration["split_policy"]
    if policy == {"kind": "openml_official"}:
        evidence_kind = "exact_openml_official_training_rows"
        evidence_rows = official_training_rows
    elif (
        isinstance(policy, dict)
        and policy.get("kind") == "target_free_split_construction_v1"
        and isinstance(policy.get("constructor"), dict)
    ):
        if row_count < minimum_rows:
            evidence_kind = "dataset_row_upper_bound_below_gate"
            evidence_rows = [row_count] * len(COORDINATE_FOLDS)
        else:
            evidence_kind = "exact_target_free_constructed_training_rows"
            evidence_rows = _custom_split_training_rows(
                task,
                declaration,
                row_count=row_count,
            )
    else:
        raise RuntimeError(f"task {task_id} split policy changed")
    applicability = [
        rows >= minimum_rows
        for rows in evidence_rows
    ]
    if slot["t5_size_gate_applicability"] != applicability:
        raise RuntimeError(
            f"task {task_id} T5 size-gate applicability differs from "
            "the power design before target materialization"
        )
    return {
        "task_id": task_id,
        "dataset_id": int(declaration["dataset_id"]),
        "lineage_cluster": declaration["lineage_cluster"],
        "stratum": declaration["stratum"],
        "coordinate_folds": list(COORDINATE_FOLDS),
        "minimum_outer_training_rows": minimum_rows,
        "evidence_kind": evidence_kind,
        "outer_training_rows_or_upper_bound": evidence_rows,
        "t5_size_gate_applicability": applicability,
    }


def _validate_split_applicability_binding(
    binding: Any,
    decision: dict[str, Any],
    declarations: dict[str, Any],
    *,
    minimum_outer_training_rows: int | None = None,
) -> None:
    slots = decision.get("prospective_panel", {}).get("slots")
    rows = binding.get("attestations") if isinstance(binding, dict) else None
    minimum = (
        common.t5_minimum_outer_training_rows()
        if minimum_outer_training_rows is None
        else minimum_outer_training_rows
    )
    if (
        not isinstance(binding, dict)
        or set(binding)
        != {
            "kind",
            "verified_before_target_materialization_or_inspection",
            "target_values_materialized_or_inspected",
            "target_bearing_openml_container_may_be_cached",
            "target_column_excluded_from_projection",
            "minimum_outer_training_rows",
            "attestations",
        }
        or binding["kind"]
        != "panel3_pre_target_split_applicability_v1"
        or binding[
            "verified_before_target_materialization_or_inspection"
        ]
        is not True
        or binding["target_values_materialized_or_inspected"] is not False
        or binding["target_bearing_openml_container_may_be_cached"] is not True
        or binding["target_column_excluded_from_projection"] is not True
        or binding["minimum_outer_training_rows"] != minimum
        or not isinstance(slots, list)
        or not isinstance(rows, list)
        or len(slots) != 12
        or len(rows) != len(slots)
    ):
        raise RuntimeError(
            "panel-3 pre-target split applicability binding changed"
        )
    declarations_by_task = {
        int(row["task_id"]): row
        for row in declarations["candidates"]
        if row["selection_role"] == "selected"
    }
    for row, slot in zip(rows, slots, strict=True):
        task_id = slot.get("task_id") if isinstance(slot, dict) else None
        declaration = (
            declarations_by_task.get(task_id)
            if type(task_id) is int
            else None
        )
        evidence = (
            row.get("outer_training_rows_or_upper_bound")
            if isinstance(row, dict)
            else None
        )
        applicability = (
            row.get("t5_size_gate_applicability")
            if isinstance(row, dict)
            else None
        )
        declared_split = (
            declaration.get("split_policy")
            if isinstance(declaration, dict)
            else None
        )
        evidence_kind = (
            row.get("evidence_kind")
            if isinstance(row, dict)
            else None
        )
        evidence_matches_declared_split = (
            evidence_kind == "exact_openml_official_training_rows"
            if declared_split == {"kind": "openml_official"}
            else (
                isinstance(declared_split, dict)
                and declared_split.get("kind")
                == "target_free_split_construction_v1"
                and evidence_kind
                in {
                    "exact_target_free_constructed_training_rows",
                    "dataset_row_upper_bound_below_gate",
                }
            )
        )
        if (
            declaration is None
            or not isinstance(row, dict)
            or set(row)
            != {
                "task_id",
                "dataset_id",
                "lineage_cluster",
                "stratum",
                "coordinate_folds",
                "minimum_outer_training_rows",
                "evidence_kind",
                "outer_training_rows_or_upper_bound",
                "t5_size_gate_applicability",
            }
            or row["task_id"] != task_id
            or row["dataset_id"] != int(declaration["dataset_id"])
            or row["lineage_cluster"] != slot["lineage_cluster"]
            or row["stratum"] != slot["stratum"]
            or row["coordinate_folds"] != list(COORDINATE_FOLDS)
            or row["minimum_outer_training_rows"] != minimum
            or not evidence_matches_declared_split
            or not isinstance(evidence, list)
            or len(evidence) != len(COORDINATE_FOLDS)
            or any(type(value) is not int or value <= 0 for value in evidence)
            or not isinstance(applicability, list)
            or len(applicability) != len(COORDINATE_FOLDS)
            or any(type(value) is not bool for value in applicability)
            or applicability != [value >= minimum for value in evidence]
            or applicability != slot["t5_size_gate_applicability"]
            or (
                row["evidence_kind"] == "dataset_row_upper_bound_below_gate"
                and (any(applicability) or len(set(evidence)) != 1)
            )
        ):
            raise RuntimeError(
                "panel-3 pre-target split applicability binding changed"
            )


def _build_split_applicability_binding(
    decision: dict[str, Any],
    declarations: dict[str, Any],
    *,
    minimum_outer_training_rows: int | None = None,
) -> dict[str, Any]:
    minimum = (
        common.t5_minimum_outer_training_rows()
        if minimum_outer_training_rows is None
        else minimum_outer_training_rows
    )
    declarations_by_task = {
        int(row["task_id"]): row
        for row in declarations["candidates"]
        if row["selection_role"] == "selected"
    }
    slots = decision["prospective_panel"]["slots"]
    rows = []
    for slot in slots:
        task_id = int(slot["task_id"])
        declaration = declarations_by_task.get(task_id)
        if declaration is None:
            raise RuntimeError(
                f"task {task_id} power slot is not an exact primary"
            )
        rows.append(
            _split_applicability_attestation(
                declaration,
                slot,
                minimum_rows=minimum,
            )
        )
    binding = {
        "kind": "panel3_pre_target_split_applicability_v1",
        "verified_before_target_materialization_or_inspection": True,
        "target_values_materialized_or_inspected": False,
        "target_bearing_openml_container_may_be_cached": True,
        "target_column_excluded_from_projection": True,
        "minimum_outer_training_rows": minimum,
        "attestations": rows,
    }
    _validate_split_applicability_binding(
        binding,
        decision,
        declarations,
        minimum_outer_training_rows=minimum,
    )
    return binding


def _target_ineligibility_reason(exc: BaseException) -> str | None:
    message = str(exc).lower()
    if isinstance(exc, target_check.TargetPreflightError):
        if "metadata drifted" in message or "fingerprint drifted" in message:
            return None
        if "target" in message:
            return "target_policy_rejection"
    if isinstance(exc, RuntimeError) and (
        "did not return a target" in message
        or "differs from the dataset default target" in message
    ):
        return "target_metadata_ambiguity"
    return None


def _validate_record_against_declaration(
    record: dict[str, Any],
    declaration: dict[str, Any],
) -> None:
    task_id = int(declaration["task_id"])
    if int(record["openml_task_id"]) != task_id:
        raise RuntimeError(f"task {task_id} identity drifted")
    if int(record["openml_dataset_id"]) != int(declaration["dataset_id"]):
        raise RuntimeError(f"task {task_id} dataset ID drifted")
    if record["normalized_name"] != declaration["expected_normalized_name"]:
        raise RuntimeError(f"task {task_id} name drifted")
    if str(record["target_name"]) != str(
        declaration["expected_target_name"]
    ):
        raise RuntimeError(f"task {task_id} target drifted")
    if int(record["openml_task_type_id"]) != 2:
        raise RuntimeError(f"task {task_id} is not supervised regression")


def _validate_target_attestation(
    attestation: Any,
    record: dict[str, Any],
) -> None:
    fields = {
        "policy",
        "checked",
        "passed",
        "target_outcome_statistics_computed",
        "target_values_persisted",
        "binding",
    }
    binding = (
        attestation.get("binding")
        if isinstance(attestation, dict)
        else None
    )
    if (
        not isinstance(attestation, dict)
        or set(attestation) != fields
        or attestation["policy"] != target_check.TARGET_POLICY
        or attestation["checked"] is not True
        or attestation["passed"] is not True
        or attestation["target_outcome_statistics_computed"] is not False
        or attestation["target_values_persisted"] is not False
        or not isinstance(binding, dict)
        or set(binding)
        != {
            "openml_task_id",
            "openml_dataset_id",
            "target_name",
            "dataset_fingerprint_sha256",
            "ordered_task_view_sha256",
        }
        or binding["openml_task_id"] != int(record["openml_task_id"])
        or binding["openml_dataset_id"] != int(record["openml_dataset_id"])
        or binding["target_name"] != str(record["target_name"])
        or binding["dataset_fingerprint_sha256"]
        != ctr.sha256_json(record["fingerprint"])
        or not provenance.is_sha256(
            binding["ordered_task_view_sha256"]
        )
    ):
        raise RuntimeError(
            f"task {record['openml_task_id']} target attestation is malformed"
        )


def build(
    *,
    require_clean_source: bool = True,
) -> dict[str, Any]:
    # This must precede declaration loading and every OpenML target-column
    # materialization or inspection. It recomputes the H1-frozen power
    # decision from the bound spent census.
    decision, decision_file_sha256 = power_design.load_decision_snapshot(
        require_current_sources=True,
        recompute=True,
    )
    if decision["target_preflight_authorized"] is not True:
        raise RuntimeError("panel-3 power design does not authorize preflight")
    source_snapshots, source_sha256 = common.secure_snapshot_files(
        list(common.PANEL3_SOURCE_PATHS)
    )
    declarations = common.validate_declarations(
        common.decode_json_bytes(
            source_snapshots[common.DECLARATIONS.absolute()],
            source=common.DECLARATIONS,
        )
    )
    candidate_contract = common.decode_json_bytes(
        source_snapshots[common.CANDIDATE_CONTRACT.absolute()],
        source=common.CANDIDATE_CONTRACT,
    )
    decision_source_sha256 = decision.get("source_sha256")
    if (
        not isinstance(decision_source_sha256, dict)
        or any(
            decision_source_sha256.get(relative) != digest
            for relative, digest in source_sha256.items()
        )
    ):
        raise RuntimeError(
            "panel-3 source snapshot differs from the power decision"
        )
    minimum_outer_training_rows = (
        common.t5_minimum_outer_training_rows(candidate_contract)
    )
    execution_head = (
        _require_frozen_clean_source(declarations, source_sha256)
        if require_clean_source
        else _git("rev-parse", "HEAD")
    )
    # Freeze the exact power-design split geometry before `_load_task_record`
    # or the target attestor can materialize a fresh target value.
    split_applicability_binding = _build_split_applicability_binding(
        decision,
        declarations,
        minimum_outer_training_rows=minimum_outer_training_rows,
    )

    rows = []
    for declaration in declarations["candidates"]:
        task_id = int(declaration["task_id"])
        try:
            record = _load_task_record(task_id)
            _validate_record_against_declaration(record, declaration)
            target_attestation = target_check.attest_openml_target(record)
        except BaseException as exc:
            reason = _target_ineligibility_reason(exc)
            if reason is None:
                raise
            rows.append(
                {
                    "task_id": task_id,
                    "dataset_id": int(declaration["dataset_id"]),
                    "lineage_cluster": declaration["lineage_cluster"],
                    "stratum": declaration["stratum"],
                    "priority": int(declaration["priority"]),
                    "origin": declaration["origin"],
                    "status": "target_ineligible",
                    "target_eligibility_reason": reason,
                    "target_outcome_statistics_computed": False,
                    "target_values_persisted": False,
                }
            )
            continue
        _validate_target_attestation(target_attestation, record)
        rows.append(
            {
                "task_id": task_id,
                "dataset_id": int(declaration["dataset_id"]),
                "lineage_cluster": declaration["lineage_cluster"],
                "stratum": declaration["stratum"],
                "priority": int(declaration["priority"]),
                "origin": declaration["origin"],
                "status": "target_eligible",
                "target_attestation": target_attestation,
                "task_record": record,
            }
        )

    eligible_counts = Counter(
        row["stratum"]
        for row in rows
        if row["status"] == "target_eligible"
    )
    insufficient = {
        stratum: eligible_counts[stratum]
        for stratum in common.STRATA
        if eligible_counts[stratum] < common.REQUIRED_PER_STRATUM
    }
    if insufficient:
        raise RuntimeError(
            "panel-3 target preflight lacks four eligible declarations: "
            f"{insufficient}"
        )
    primary_task_ids = {
        int(slot["task_id"])
        for slot in decision["prospective_panel"]["slots"]
    }
    eligible_task_ids = {
        int(row["task_id"])
        for row in rows
        if row["status"] == "target_eligible"
    }
    exact_primary_tasks_target_eligible = (
        primary_task_ids <= eligible_task_ids
    )

    artifact = {
        "schema_version": 1,
        "name": "darkofit_panel3_target_preflight_v1",
        "created_from_clean_sources": require_clean_source,
        "eligibility_policy": target_check.TARGET_POLICY,
        "outcome_blind": True,
        "eligibility_aware": True,
        "target_statistics_used": False,
        "target_values_persisted": False,
        "candidate_or_control_models_fitted": False,
        "candidate_or_control_outcomes_inspected": False,
        "selection_performed": False,
        "registry_authorized": False,
        "exact_power_authorized_primary_tasks_target_eligible": (
            exact_primary_tasks_target_eligible
        ),
        "registry_build_authorized": (
            decision["target_preflight_authorized"] is True
            and exact_primary_tasks_target_eligible
            and bool(decision["retained_candidates"])
        ),
        "power_design_path": str(
            common.POWER_DESIGN_DECISION.relative_to(ROOT)
        ),
        "power_design_file_sha256": decision_file_sha256,
        "power_design_decision_sha256": decision["decision_sha256"],
        "power_design_decision": decision,
        "power_design_split_applicability_binding": (
            split_applicability_binding
        ),
        "retained_candidates": decision["retained_candidates"],
        "declaration_count": len(rows),
        "target_eligible_count": sum(
            row["status"] == "target_eligible" for row in rows
        ),
        "target_ineligible_count": sum(
            row["status"] != "target_eligible" for row in rows
        ),
        "eligible_counts_by_stratum": {
            stratum: eligible_counts[stratum] for stratum in common.STRATA
        },
        "declarations_sha256": source_sha256[
            str(common.DECLARATIONS.relative_to(ROOT))
        ],
        "source_sha256": source_sha256,
        "sources": {
            "darkofit_execution_head": execution_head,
            "darkofit_prefreeze_head": declarations[
                "darkofit_prefreeze_head"
            ],
            "chimeraboost_head": declarations["chimeraboost_head"],
        },
        "tasks": rows,
    }
    artifact = common.bind_artifact_sha256(
        artifact,
        "target_preflight_sha256",
    )
    _recheck_authorization_inputs(
        decision=decision,
        decision_file_sha256=decision_file_sha256,
        source_sha256=source_sha256,
        execution_head=execution_head,
        require_clean_source=require_clean_source,
    )
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.expanduser().absolute() != DEFAULT_OUTPUT:
        raise RuntimeError("panel-3 target-preflight output path changed")
    common.validate_create_path(args.output)
    # Build and validate every declaration before a pathname is created. A
    # technical or insufficient-reserve failure therefore leaves no output.
    artifact = build()
    _recheck_authorization_inputs(
        decision=artifact["power_design_decision"],
        decision_file_sha256=artifact["power_design_file_sha256"],
        source_sha256=artifact["source_sha256"],
        execution_head=artifact["sources"]["darkofit_execution_head"],
        require_clean_source=True,
    )
    common.atomic_create(args.output, ctr.canonical_json_bytes(artifact))
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().absolute()),
                "target_preflight_sha256": artifact[
                    "target_preflight_sha256"
                ],
                "eligible": artifact["target_eligible_count"],
                "ineligible": artifact["target_ineligible_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
