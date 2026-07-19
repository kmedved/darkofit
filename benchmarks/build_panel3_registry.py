#!/usr/bin/env python3
"""Build the deterministic, contamination-screened panel-3 registry."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CHIMERA_ROOT = ROOT.parent / "chimeraboost"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import build_ctr23_contamination_registry as ctr  # noqa: E402
from benchmarks import build_fresh_confirmation_registry as fresh  # noqa: E402
from benchmarks import build_panel3_power_design as power_design  # noqa: E402
from benchmarks import panel3_data_contract as data_contract  # noqa: E402
from benchmarks import panel3_registry_common as common  # noqa: E402
from benchmarks.campaign_lib import provenance  # noqa: E402
from benchmarks import preflight_panel3_registry as preflight_builder  # noqa: E402


DEFAULT_PREFLIGHT = (
    ROOT / "benchmarks" / "panel3_target_preflight.json"
)
DEFAULT_OUTPUT = ROOT / "benchmarks" / "panel3_registry.json"

CTR_SNAPSHOT = ROOT / "benchmarks" / "ctr23_suite_snapshot.json"
CTR_PARTITION = ROOT / "benchmarks" / "ctr23_partition.json"
CTR_DECLARATIONS = (
    ROOT / "benchmarks" / "ctr23_contamination_sources.json"
)
FRESH_REGISTRY = ROOT / "benchmarks" / "fresh_confirmation_registry.json"
FRESH_REGISTRY_V2 = (
    ROOT / "benchmarks" / "fresh_confirmation_registry_v2.json"
)
ORDINAL_REGISTRY = (
    ROOT / "benchmarks" / "native_ordinal_c2_registry.json"
)
T5_DECLARATIONS = (
    ROOT / "benchmarks" / "t5_composite_registry_declarations.json"
)
T5_REGISTRY = ROOT / "benchmarks" / "t5_composite_registry.json"
T5_INVALID_ATTEMPT = (
    ROOT / "benchmarks" / "t5_composite_registry_invalid_attempt.md"
)
T7_RAW = ROOT / "benchmarks" / "t7_catboost_attribution_raw.json"
T7_SUMMARY = ROOT / "benchmarks" / "t7_catboost_attribution_summary.json"
T8_PROTOCOL = ROOT / "benchmarks" / "t8_distributional_flagship_protocol.md"
T8_RAW = ROOT / "benchmarks" / "t8_distributional_flagship_raw.csv"
T8_RESULT = ROOT / "benchmarks" / "t8_distributional_flagship_result.md"
SMOOTH_CROSS_RAW = ROOT / "benchmarks" / "smooth_cross_features.json"
SMOOTH_CROSS_ANALYSIS = (
    ROOT / "benchmarks" / "smooth_cross_margin_analysis.json"
)
SHIPPING_POLICY = ROOT / "benchmarks" / "SHIPPING_POLICY.md"

FROZEN_EVIDENCE = (
    CTR_SNAPSHOT,
    CTR_PARTITION,
    CTR_DECLARATIONS,
    FRESH_REGISTRY,
    FRESH_REGISTRY_V2,
    ORDINAL_REGISTRY,
    T5_DECLARATIONS,
    T5_REGISTRY,
    T5_INVALID_ATTEMPT,
    T7_RAW,
    T7_SUMMARY,
    T8_PROTOCOL,
    T8_RAW,
    T8_RESULT,
    SMOOTH_CROSS_RAW,
    SMOOTH_CROSS_ANALYSIS,
    SHIPPING_POLICY,
)
CHIMERA_EXPOSURE_PATHS = (
    CHIMERA_ROOT / "benchmarks" / "run_benchmarks.py",
    CHIMERA_ROOT / "tests" / "test_highcard.py",
    CHIMERA_ROOT
    / "benchmarks"
    / "synthgen"
    / "corpus_marginals.json",
)
SPENT_JSON_PATHS = (
    CTR_SNAPSHOT,
    CTR_PARTITION,
    FRESH_REGISTRY,
    FRESH_REGISTRY_V2,
    ORDINAL_REGISTRY,
    T5_DECLARATIONS,
    T5_REGISTRY,
    T7_RAW,
    T7_SUMMARY,
)

# These are deliberately explicit even when another catalog currently implies
# them. The list documents failed T5 nominees and alternate task identities
# whose absence from the older helper caused the original spent-data gap.
EXPLICIT_SPENT_TASK_IDS = {
    168887,  # CD4: invalid default-target metadata.
    362418,  # avocado_sales: catalog exposure.
    363040,  # fifa: catalog and near-lineage exposure.
    360050,  # alternate Mauna Loa task.
    363690,  # alternate NHANES task.
}
EXPLICIT_SPENT_DATASET_IDS = {
    197,  # T8 cpu_act.
    287,  # T8 wine_quality.
    531,  # T8 boston.
    41542,  # invalid CD4 nominee.
    43927,  # invalid avocado nominee.
    45002,  # invalid fifa nominee.
}
EXPLICIT_SPENT_NORMALIZED_NAMES = {
    "cd4",
    "avocado_sales",
    "fifa",
    "mauna_loa_atmospheric_co2",
    "nhanes_age",
}

LOCKBOX_TASK_IDS = frozenset(
    {
        361247,
        361253,
        361254,
        361261,
        361264,
        361272,
        361616,
        361617,
        361618,
    }
)
LATER_INVALIDATED_LOCKBOX_TASK_IDS = frozenset(
    {
        361264,  # Socmob: current ChimeraBoost HIGHCARD_PLAN exposure.
        361272,  # FIFA: later T5/catalog exposure.
        361616,  # Moneyball: current ChimeraBoost catalog exposure.
    }
)
AUTHORIZED_LOCKBOX_TASK_IDS = (
    LOCKBOX_TASK_IDS - LATER_INVALIDATED_LOCKBOX_TASK_IDS
)
LOCKBOX_DARKOFIT_REFERENCE_ALLOWLIST = frozenset(
    {
        "BEYOND_PARITY_PLAN.md",
        "benchmarks/ctr23_suite_snapshot.json",
        "benchmarks/ctr23_partition.json",
        "benchmarks/ctr23_contamination_sources.json",
        "benchmarks/ctr23_contamination_registry.json",
    }
)


def _git(path: Path, *args: str) -> str:
    return provenance.git_output(path, *args)


def _is_ancestor(path: Path, ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=path,
            check=False,
        ).returncode
        == 0
    )


def _validate_post_preflight_boundary(
    preflight: dict[str, Any],
    preflight_path: Path,
    head: str,
    *,
    require_clean_source: bool,
) -> None:
    preflight_head = str(
        preflight["sources"]["darkofit_execution_head"]
    )
    if not _is_ancestor(ROOT, preflight_head, head):
        raise RuntimeError(
            "panel-3 registry head does not descend from target preflight"
        )
    if not require_clean_source:
        return
    try:
        preflight_relative = str(
            preflight_path.resolve().relative_to(ROOT.resolve())
        )
    except ValueError as exc:
        raise RuntimeError(
            "panel-3 preflight artifact is outside the repository"
        ) from exc
    changed_after_preflight = {
        value
        for value in _git(
            ROOT,
            "diff",
            "--name-only",
            f"{preflight_head}..{head}",
        ).splitlines()
        if value
    }
    if changed_after_preflight != {preflight_relative}:
        raise RuntimeError(
            "panel-3 post-preflight source boundary changed: "
            f"{sorted(changed_after_preflight)}"
        )


def _validate_source_snapshots_at_head(
    head: str,
    source_sha256: dict[str, str],
) -> None:
    expected = {
        str(path.relative_to(ROOT)) for path in common.PANEL3_SOURCE_PATHS
    }
    if set(source_sha256) != expected:
        raise RuntimeError("panel-3 registry source snapshot changed")
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
                f"panel-3 registry source is not committed: {relative}"
            ) from exc
        if hashlib.sha256(committed).hexdigest() != digest:
            raise RuntimeError(
                f"panel-3 registry source differs from HEAD: {relative}"
            )


def _literal_assignment_from_snapshot(
    payload: bytes,
    *,
    source: Path,
    name: str,
) -> Any:
    class _DictionaryCallNormalizer(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            node = self.generic_visit(node)
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "dict"
                and not node.args
                and all(keyword.arg is not None for keyword in node.keywords)
            ):
                return ast.copy_location(
                    ast.Dict(
                        keys=[
                            ast.Constant(keyword.arg)
                            for keyword in node.keywords
                        ],
                        values=[
                            keyword.value for keyword in node.keywords
                        ],
                    ),
                    node,
                )
            return node

    try:
        tree = ast.parse(payload.decode("utf-8"), filename=str(source))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise RuntimeError(
            f"panel-3 Chimera exposure source is invalid: {source}"
        ) from exc
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
        )
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in targets
        ):
            try:
                normalized = _DictionaryCallNormalizer().visit(node.value)
                return ast.literal_eval(normalized)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"panel-3 Chimera exposure value is not literal: {name}"
                ) from exc
    raise RuntimeError(f"{name} is absent from {source}")


def _chimera_exposure_catalog_from_snapshots(
    snapshots: dict[Path, bytes],
    source_sha256: dict[str, str],
) -> dict[str, Any]:
    benchmark_path, highcard_path, _marginals_path = (
        CHIMERA_EXPOSURE_PATHS
    )
    benchmark_bytes = snapshots[benchmark_path.absolute()]
    openml_suite = _literal_assignment_from_snapshot(
        benchmark_bytes,
        source=benchmark_path,
        name="OPENML_SUITE",
    )
    grinsztajn = _literal_assignment_from_snapshot(
        benchmark_bytes,
        source=benchmark_path,
        name="GRINSZTAJN_DATASETS",
    )
    pmlb = _literal_assignment_from_snapshot(
        benchmark_bytes,
        source=benchmark_path,
        name="PMLB_DATASETS",
    )
    high_cardinality = _literal_assignment_from_snapshot(
        benchmark_bytes,
        source=benchmark_path,
        name="HC_DATASETS",
    )
    tabarena = _literal_assignment_from_snapshot(
        snapshots[highcard_path.absolute()],
        source=highcard_path,
        name="TABARENA_51",
    )
    names = set(openml_suite)
    names.update(
        name for values in grinsztajn.values() for name in values
    )
    names.update(
        name for values in pmlb.values() for name, _task in values
    )
    names.update(high_cardinality)
    names.update(tabarena)
    dataset_ids = {
        int(spec["data_id"]) for spec in openml_suite.values()
    }
    dataset_ids.update(
        int(spec["data_id"]) for spec in high_cardinality.values()
    )
    return {
        "normalized_names": sorted(
            {ctr.normalize_name(name) for name in names}
        ),
        "openml_dataset_ids": sorted(dataset_ids),
        "source_files": dict(source_sha256),
        "tabarena_name_count": len(tabarena),
        "resolved_name_count": len(names),
    }


def _validate_chimera_snapshot_at_head(
    source_sha256: dict[str, str],
    *,
    head: str,
) -> None:
    expected = {
        str(path.relative_to(CHIMERA_ROOT))
        for path in CHIMERA_EXPOSURE_PATHS
    }
    if set(source_sha256) != expected:
        raise RuntimeError("panel-3 Chimera exposure snapshot changed")
    for relative, digest in source_sha256.items():
        try:
            committed = subprocess.run(
                ["git", "show", f"{head}:{relative}"],
                cwd=CHIMERA_ROOT,
                check=True,
                capture_output=True,
            ).stdout
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"panel-3 Chimera exposure source is unavailable: {relative}"
            ) from exc
        if hashlib.sha256(committed).hexdigest() != digest:
            raise RuntimeError(
                f"panel-3 Chimera exposure differs from HEAD: {relative}"
            )


def _recheck_registry_inputs(
    artifact: dict[str, Any],
    *,
    preflight_path: Path,
) -> None:
    current_preflight, preflight_file_sha256 = common.secure_load_json(
        preflight_path
    )
    current_decision, power_file_sha256 = common.secure_load_json(
        common.POWER_DESIGN_DECISION
    )
    if (
        not isinstance(current_preflight, dict)
        or preflight_file_sha256
        != artifact["target_preflight_file_sha256"]
        or current_preflight.get("target_preflight_sha256")
        != artifact["target_preflight_sha256"]
        or current_preflight.get("power_design_decision")
        != artifact["power_design_decision"]
        or current_decision != artifact["power_design_decision"]
        or power_file_sha256 != artifact["power_design_file_sha256"]
    ):
        raise RuntimeError(
            "panel-3 authorization artifact changed before registry "
            "publication"
        )
    common.recheck_snapshot_files(
        list(common.PANEL3_SOURCE_PATHS),
        artifact["source_sha256"],
    )
    common.recheck_snapshot_files(
        list(FROZEN_EVIDENCE),
        artifact["frozen_evidence_sha256"],
    )
    if _git(ROOT, "rev-parse", "HEAD") != artifact["sources"][
        "darkofit_registry_head"
    ]:
        raise RuntimeError(
            "panel-3 DarkoFit head changed before registry publication"
        )
    if artifact["created_from_clean_sources"] and _git(
        ROOT,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ):
        raise RuntimeError(
            "panel-3 DarkoFit tree changed before registry publication"
        )
    chimera_head = artifact["sources"]["chimeraboost_head"]
    if _git(CHIMERA_ROOT, "rev-parse", "HEAD") != chimera_head:
        raise RuntimeError(
            "panel-3 Chimera head changed before registry publication"
        )
    if artifact["created_from_clean_sources"] and _git(
        CHIMERA_ROOT,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ):
        raise RuntimeError(
            "panel-3 Chimera tree changed before registry publication"
        )
    common.recheck_snapshot_files(
        list(CHIMERA_EXPOSURE_PATHS),
        artifact["exposure_catalog"]["source_files"],
        allowed_root=CHIMERA_ROOT,
    )
    _validate_chimera_snapshot_at_head(
        artifact["exposure_catalog"]["source_files"],
        head=chimera_head,
    )


def _task_records(payload: Any) -> list[dict[str, Any]]:
    records = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            record = value.get("task_record")
            if (
                isinstance(record, dict)
                and "openml_task_id" in record
                and "fingerprint" in record
            ):
                records.append(record)
            if (
                "openml_task_id" in value
                and "fingerprint" in value
                and "openml_dataset_id" in value
            ):
                records.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    unique = {
        int(record["openml_task_id"]): record for record in records
    }
    return [unique[key] for key in sorted(unique)]


def _integer_values(payload: Any, keys: set[str]) -> set[int]:
    values = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in keys:
                    candidates = child if isinstance(child, list) else [child]
                    values.update(
                        candidate
                        for candidate in candidates
                        if type(candidate) is int and candidate > 0
                    )
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return values


def spent_evidence(
    *,
    payloads: dict[Path, Any] | None = None,
    source_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    json_paths = SPENT_JSON_PATHS
    if payloads is None:
        payloads = {path: common.load_json(path) for path in json_paths}
    elif set(payloads) != set(json_paths):
        raise RuntimeError("panel-3 frozen evidence JSON snapshot changed")
    if source_sha256 is not None and set(source_sha256) != {
        str(path.relative_to(ROOT)) for path in FROZEN_EVIDENCE
    }:
        raise RuntimeError("panel-3 frozen evidence source snapshot changed")
    partition_task_ids = frozenset(
        int(value) for value in payloads[CTR_PARTITION]["lockbox_task_ids"]
    )
    if partition_task_ids != LOCKBOX_TASK_IDS:
        raise RuntimeError("panel-3 lockbox task authorization changed")
    records = {}
    for payload in payloads.values():
        for record in _task_records(payload):
            records[int(record["openml_task_id"])] = record

    task_ids = set(records) | set(EXPLICIT_SPENT_TASK_IDS)
    dataset_ids = {
        int(record["openml_dataset_id"]) for record in records.values()
    } | set(EXPLICIT_SPENT_DATASET_IDS)
    names = {
        str(record["normalized_name"])
        for record in records.values()
        if "normalized_name" in record
    } | set(EXPLICIT_SPENT_NORMALIZED_NAMES)
    for payload in payloads.values():
        task_ids |= _integer_values(
            payload,
            {"task_id", "openml_task_id", "related_task_ids"},
        )
        dataset_ids |= _integer_values(
            payload,
            {"dataset_id", "openml_dataset_id"},
        )
    # related_task_ids are lists, so add them explicitly rather than relying
    # on a key/value walker intended for scalar IDs.
    for row in payloads[T5_DECLARATIONS]["candidates"]:
        task_ids.update(int(value) for value in row["related_task_ids"])
        names.add(str(row["expected_normalized_name"]))

    return {
        "task_records": [records[key] for key in sorted(records)],
        "openml_task_ids": sorted(task_ids),
        "openml_dataset_ids": sorted(dataset_ids),
        "normalized_names": sorted(names),
        "source_sha256": (
            {
                str(path.relative_to(ROOT)): common.sha256_file(path)
                for path in FROZEN_EVIDENCE
            }
            if source_sha256 is None
            else dict(source_sha256)
        ),
    }


def _repository_literal_is_discriminating(value: str) -> bool:
    normalized = ctr.normalize_name(value).replace("_", "")
    return len(normalized) >= 6


def _repository_match_path(match: str, revision: str) -> str:
    prefix = f"{revision}:"
    if not isinstance(match, str) or not match.startswith(prefix):
        raise RuntimeError("panel-3 git-grep result changed")
    relative = match[len(prefix) :]
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise RuntimeError("panel-3 git-grep path is invalid")
    return relative


def _preflight_rows(preflight: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = preflight.get("tasks")
    if not isinstance(rows, list):
        raise RuntimeError("panel-3 preflight task ledger is missing")
    mapped = {}
    for row in rows:
        task_id = int(row["task_id"])
        if task_id in mapped:
            raise RuntimeError("panel-3 preflight repeats a task")
        mapped[task_id] = row
    return mapped


def _validate_preflight(
    preflight: dict[str, Any],
    declarations: dict[str, Any],
    *,
    declarations_file_sha256: str | None = None,
    power_design_file_sha256: str | None = None,
    source_sha256: dict[str, str] | None = None,
    candidate_contract: dict[str, Any] | None = None,
) -> None:
    common.verify_artifact_sha256(preflight, "target_preflight_sha256")
    expected_fields = {
        "schema_version",
        "name",
        "created_from_clean_sources",
        "eligibility_policy",
        "outcome_blind",
        "eligibility_aware",
        "target_statistics_used",
        "target_values_persisted",
        "candidate_or_control_models_fitted",
        "candidate_or_control_outcomes_inspected",
        "selection_performed",
        "registry_authorized",
        "exact_power_authorized_primary_tasks_target_eligible",
        "registry_build_authorized",
        "power_design_path",
        "power_design_file_sha256",
        "power_design_decision_sha256",
        "power_design_decision",
        "power_design_split_applicability_binding",
        "retained_candidates",
        "declaration_count",
        "target_eligible_count",
        "target_ineligible_count",
        "eligible_counts_by_stratum",
        "declarations_sha256",
        "source_sha256",
        "sources",
        "tasks",
        "target_preflight_sha256",
    }
    if (
        set(preflight) != expected_fields
        or preflight.get("schema_version") != 1
        or preflight.get("name")
        != "darkofit_panel3_target_preflight_v1"
        or preflight.get("created_from_clean_sources") is not True
        or preflight.get("eligibility_policy")
        != preflight_builder.target_check.TARGET_POLICY
        or preflight.get("outcome_blind") is not True
        or preflight.get("eligibility_aware") is not True
        or preflight.get("target_statistics_used") is not False
        or preflight.get("target_values_persisted") is not False
        or preflight.get("candidate_or_control_models_fitted") is not False
        or preflight.get("candidate_or_control_outcomes_inspected")
        is not False
        or preflight.get("selection_performed") is not False
        or preflight.get("registry_authorized") is not False
        or preflight.get(
            "exact_power_authorized_primary_tasks_target_eligible"
        )
        is not True
        or preflight.get("registry_build_authorized") is not True
    ):
        raise RuntimeError("panel-3 target preflight boundary is invalid")
    expected_declarations_sha256 = (
        common.sha256_file(common.DECLARATIONS)
        if declarations_file_sha256 is None
        else declarations_file_sha256
    )
    if preflight.get("declarations_sha256") != (
        expected_declarations_sha256
    ):
        raise RuntimeError("panel-3 preflight declaration binding changed")
    decision = preflight.get("power_design_decision")
    power_design.validate_decision(
        decision,
        decision_path=common.POWER_DESIGN_DECISION,
        require_current_sources=True,
        recompute=False,
    )
    if (
        preflight.get("power_design_path")
        != str(common.POWER_DESIGN_DECISION.relative_to(ROOT))
        or preflight.get("power_design_file_sha256")
        != (
            common.sha256_file(common.POWER_DESIGN_DECISION)
            if power_design_file_sha256 is None
            else power_design_file_sha256
        )
        or preflight.get("power_design_decision_sha256")
        != decision["decision_sha256"]
        or preflight.get("retained_candidates")
        != decision["retained_candidates"]
        or decision["target_preflight_authorized"] is not True
    ):
        raise RuntimeError("panel-3 preflight power-design binding changed")
    preflight_builder._validate_split_applicability_binding(
        preflight.get("power_design_split_applicability_binding"),
        decision,
        declarations,
        minimum_outer_training_rows=(
            None
            if candidate_contract is None
            else common.t5_minimum_outer_training_rows(
                candidate_contract
            )
        ),
    )
    expected_sources = (
        {
            str(path.relative_to(ROOT)): common.sha256_file(path)
            for path in common.PANEL3_SOURCE_PATHS
        }
        if source_sha256 is None
        else source_sha256
    )
    if preflight.get("source_sha256") != expected_sources:
        raise RuntimeError("panel-3 preflight source map changed")
    declared_ids = {
        int(row["task_id"]) for row in declarations["candidates"]
    }
    rows = _preflight_rows(preflight)
    if set(rows) != declared_ids:
        raise RuntimeError("panel-3 preflight declaration ledger changed")
    declarations_by_id = {
        int(row["task_id"]): row for row in declarations["candidates"]
    }
    for task_id, row in rows.items():
        declaration = declarations_by_id[task_id]
        base = {
            "task_id": task_id,
            "dataset_id": int(declaration["dataset_id"]),
            "lineage_cluster": declaration["lineage_cluster"],
            "stratum": declaration["stratum"],
            "priority": int(declaration["priority"]),
            "origin": declaration["origin"],
        }
        if any(row.get(key) != value for key, value in base.items()):
            raise RuntimeError(
                f"panel-3 preflight task {task_id} identity changed"
            )
        if row.get("status") == "target_eligible":
            if set(row) != {
                *base,
                "status",
                "target_attestation",
                "task_record",
            }:
                raise RuntimeError(
                    f"panel-3 eligible preflight task {task_id} changed"
                )
            record = row["task_record"]
            preflight_builder._validate_record_against_declaration(
                record, declaration
            )
            preflight_builder._validate_target_attestation(
                row["target_attestation"], record
            )
        elif row.get("status") == "target_ineligible":
            if (
                set(row)
                != {
                    *base,
                    "status",
                    "target_eligibility_reason",
                    "target_outcome_statistics_computed",
                    "target_values_persisted",
                }
                or not isinstance(
                    row["target_eligibility_reason"], str
                )
                or not row["target_eligibility_reason"]
                or row["target_outcome_statistics_computed"] is not False
                or row["target_values_persisted"] is not False
            ):
                raise RuntimeError(
                    f"panel-3 ineligible preflight task {task_id} changed"
                )
        else:
            raise RuntimeError(
                f"panel-3 preflight task {task_id} status changed"
            )
    eligible = sum(
        row["status"] == "target_eligible" for row in rows.values()
    )
    expected_counts = {
        stratum: sum(
            row["status"] == "target_eligible"
            and row["stratum"] == stratum
            for row in rows.values()
        )
        for stratum in common.STRATA
    }
    if (
        preflight["declaration_count"] != len(rows)
        or preflight["target_eligible_count"] != eligible
        or preflight["target_ineligible_count"] != len(rows) - eligible
        or preflight["eligible_counts_by_stratum"] != expected_counts
    ):
        raise RuntimeError("panel-3 preflight counts changed")


def _load_task_record_with_splits(task_id: int) -> dict[str, Any]:
    return ctr._task_record(int(task_id), include_splits=True)


def _load_task_feature_view(
    task_id: int,
    record: dict[str, Any],
) -> tuple[Any, pd.DataFrame, list[bool], str]:
    """Load the exact target-separated task view without outcome statistics."""
    import openml

    task = openml.tasks.get_task(int(task_id), download_splits=True)
    dataset = task.get_dataset()
    X, target, categorical, names = dataset.get_data(
        target=task.target_name,
        include_row_id=False,
        include_ignore_attribute=False,
        dataset_format="dataframe",
    )
    if (
        not isinstance(X, pd.DataFrame)
        or int(task.task_id) != int(task_id)
        or int(dataset.dataset_id) != int(record["openml_dataset_id"])
        or str(task.target_name) != str(record["target_name"])
        or str(dataset.md5_checksum) != str(record["openml_declared_md5"])
        or list(X.columns) != list(names)
        or len(X) != int(record["fingerprint"]["n_rows"])
        or X.shape[1] != int(record["fingerprint"]["n_features"])
        or not isinstance(categorical, list)
        or len(categorical) != X.shape[1]
        or any(type(value) is not bool for value in categorical)
    ):
        raise RuntimeError(
            f"panel-3 task {task_id} feature view changed"
        )
    numeric_target = pd.to_numeric(target, errors="raise").astype(np.float64)
    ordered_view_sha256 = data_contract.ordered_task_view_sha256(
        X,
        numeric_target,
    )
    return task, X, categorical, ordered_view_sha256


def _validate_preflight_ordered_view(
    task_id: int,
    preflight_row: dict[str, Any],
    ordered_view_sha256: str,
) -> None:
    expected = (
        preflight_row.get("target_attestation", {})
        .get("binding", {})
        .get("ordered_task_view_sha256")
    )
    if (
        not provenance.is_sha256(ordered_view_sha256)
        or ordered_view_sha256 != expected
    ):
        raise RuntimeError(
            f"panel-3 task {task_id} ordered view drifted after "
            "target preflight"
        )


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<i8")
    if array.ndim != 1:
        raise RuntimeError("panel-3 split index array must be one-dimensional")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _frozen_coordinate(
    *,
    fold: int,
    train: np.ndarray,
    test: np.ndarray,
    n_rows: int,
    allow_unused_rows: bool,
) -> dict[str, Any]:
    train = np.ascontiguousarray(train, dtype="<i8")
    test = np.ascontiguousarray(test, dtype="<i8")
    if (
        train.ndim != 1
        or test.ndim != 1
        or train.size == 0
        or test.size == 0
        or np.any(train < 0)
        or np.any(test < 0)
        or np.any(train >= n_rows)
        or np.any(test >= n_rows)
        or np.unique(train).size != train.size
        or np.unique(test).size != test.size
        or np.intersect1d(train, test).size
        or (
            not allow_unused_rows
            and train.size + test.size != n_rows
        )
    ):
        raise RuntimeError(
            f"panel-3 frozen split fold {fold} is invalid"
        )
    return {
        "repeat": 0,
        "fold": int(fold),
        "sample": 0,
        "train_indices": [int(value) for value in train],
        "test_indices": [int(value) for value in test],
        "train_size": int(train.size),
        "test_size": int(test.size),
        "train_index_sha256": _array_sha256(train),
        "test_index_sha256": _array_sha256(test),
    }


def _shuffled_kfold_indices(
    n_rows: int,
    constructor: dict[str, Any],
) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import KFold

    required = {
        "kind",
        "n_splits",
        "shuffle",
        "random_state",
        "allow_unused_rows",
    }
    if (
        set(constructor) != required
        or constructor["kind"] != "shuffled_kfold_v1"
        or constructor["n_splits"] != 3
        or constructor["shuffle"] is not True
        or type(constructor["random_state"]) is not int
        or constructor["allow_unused_rows"] is not False
    ):
        raise RuntimeError("panel-3 shuffled split declaration changed")
    splitter = KFold(
        n_splits=3,
        shuffle=True,
        random_state=int(constructor["random_state"]),
    )
    rows = np.arange(n_rows, dtype="<i8")
    return [
        (
            np.asarray(train, dtype="<i8"),
            np.asarray(test, dtype="<i8"),
        )
        for train, test in splitter.split(rows)
    ]


def _group_kfold_indices(
    X: pd.DataFrame,
    constructor: dict[str, Any],
) -> list[tuple[np.ndarray, np.ndarray]]:
    required = {
        "kind",
        "n_splits",
        "group_key",
        "group_order",
        "fold_assignment",
        "allow_unused_rows",
    }
    if (
        set(constructor) != required
        or constructor["kind"] != "size_balanced_group_kfold_v1"
        or constructor["n_splits"] != 3
        or constructor["group_order"]
        != "descending_row_count_then_group_sha256"
        or constructor["fold_assignment"]
        != "minimum_row_count_then_lowest_fold"
        or constructor["allow_unused_rows"] is not False
    ):
        raise RuntimeError("panel-3 grouped split declaration changed")
    hashes = data_contract.canonical_group_hashes(
        X,
        constructor["group_key"],
    )
    fold_ids = np.asarray(
        data_contract.greedy_group_fold_ids(hashes, n_splits=3),
        dtype="<i8",
    )
    rows = np.arange(len(X), dtype="<i8")
    return [
        (rows[fold_ids != fold], rows[fold_ids == fold])
        for fold in range(3)
    ]


def _chronological_indices(
    X: pd.DataFrame,
    constructor: dict[str, Any],
) -> list[tuple[np.ndarray, np.ndarray]]:
    required = {
        "kind",
        "source_column",
        "format",
        "utc",
        "block_count",
        "never_split_equal_values",
        "folds",
        "allow_unused_rows",
    }
    expected_folds = [
        {"fold": 0, "train_blocks": [0], "test_blocks": [1]},
        {"fold": 1, "train_blocks": [0, 1], "test_blocks": [2]},
        {
            "fold": 2,
            "train_blocks": [0, 1, 2],
            "test_blocks": [3],
        },
    ]
    if (
        set(constructor) != required
        or constructor["kind"] != "expanding_unique_datetime_blocks_v1"
        or not isinstance(constructor["source_column"], str)
        or not constructor["source_column"]
        or not isinstance(constructor["format"], str)
        or not constructor["format"]
        or type(constructor["utc"]) is not bool
        or constructor["block_count"] != 4
        or constructor["never_split_equal_values"] is not True
        or constructor["folds"] != expected_folds
        or constructor["allow_unused_rows"] is not True
        or constructor["source_column"] not in X.columns
    ):
        raise RuntimeError("panel-3 chronological split declaration changed")
    try:
        parsed = pd.to_datetime(
            X[constructor["source_column"]],
            format=constructor["format"],
            utc=constructor["utc"],
            errors="raise",
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("panel-3 chronological split parse failed") from exc
    if bool(parsed.isna().any()):
        raise RuntimeError("panel-3 chronological split date is missing")
    values = parsed.astype("int64").to_numpy(dtype="<i8")
    unique_values = np.unique(values)
    blocks = np.array_split(unique_values, 4)
    if any(block.size == 0 for block in blocks):
        raise RuntimeError(
            "panel-3 chronological split has fewer dates than blocks"
        )
    block_by_value = {
        int(value): block
        for block, values_in_block in enumerate(blocks)
        for value in values_in_block
    }
    row_blocks = np.asarray(
        [block_by_value[int(value)] for value in values],
        dtype="<i8",
    )
    rows = np.arange(len(X), dtype="<i8")
    return [
        (
            rows[np.isin(row_blocks, declaration["train_blocks"])],
            rows[np.isin(row_blocks, declaration["test_blocks"])],
        )
        for declaration in expected_folds
    ]


def _freeze_split_policy(
    X: pd.DataFrame,
    declared: dict[str, Any],
) -> dict[str, Any]:
    if declared == {"kind": "openml_official"}:
        return declared
    if (
        not isinstance(declared, dict)
        or set(declared) != {"kind", "constructor"}
        or declared["kind"] != "target_free_split_construction_v1"
        or not isinstance(declared["constructor"], dict)
    ):
        raise RuntimeError("panel-3 split policy declaration is invalid")
    constructor = declared["constructor"]
    if constructor.get("kind") == "shuffled_kfold_v1":
        pairs = _shuffled_kfold_indices(len(X), constructor)
    elif constructor.get("kind") == "size_balanced_group_kfold_v1":
        pairs = _group_kfold_indices(X, constructor)
    elif (
        constructor.get("kind")
        == "expanding_unique_datetime_blocks_v1"
    ):
        pairs = _chronological_indices(X, constructor)
    else:
        raise RuntimeError("panel-3 split constructor is unsupported")
    if len(pairs) != 3:
        raise RuntimeError("panel-3 split constructor did not create 3 folds")
    allow_unused = bool(constructor["allow_unused_rows"])
    coordinates = [
        _frozen_coordinate(
            fold=fold,
            train=train,
            test=test,
            n_rows=len(X),
            allow_unused_rows=allow_unused,
        )
        for fold, (train, test) in enumerate(pairs)
    ]
    if not allow_unused:
        partition = np.sort(
            np.concatenate(
                [
                    np.asarray(row["test_indices"], dtype="<i8")
                    for row in coordinates
                ]
            )
        )
        if not np.array_equal(partition, np.arange(len(X), dtype="<i8")):
            raise RuntimeError(
                "panel-3 custom test folds do not partition every row"
            )
    return {
        "kind": "frozen_explicit",
        "allow_unused_rows": allow_unused,
        "construction": constructor,
        "coordinates": coordinates,
    }


def select_first_four(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Mutate eligibility labels and return frozen-priority selections."""
    selected = []
    for stratum in common.STRATA:
        eligible = [
            row
            for row in records
            if row["stratum"] == stratum and row["status"] == "eligible"
        ]
        chosen = eligible[: common.REQUIRED_PER_STRATUM]
        if len(chosen) != common.REQUIRED_PER_STRATUM:
            raise RuntimeError(
                f"panel-3 {stratum} has only {len(chosen)} eligible lineages"
            )
        selected.extend(chosen)
        selected_ids = {row["task_id"] for row in chosen}
        for row in records:
            if row["stratum"] != stratum or row["status"] != "eligible":
                continue
            row["status"] = (
                "selected"
                if row["task_id"] in selected_ids
                else "eligible_reserve"
            )
    return selected


def _base_exclusion_reasons(
    declaration: dict[str, Any],
    preflight_row: dict[str, Any],
    *,
    spent: dict[str, Any],
    exposure: dict[str, Any],
    known_names: list[str],
    thresholds: dict[str, Any],
    prefreeze: str,
    chimera_head: str,
) -> list[dict[str, Any]]:
    if preflight_row["status"] != "target_eligible":
        return [
            {
                "kind": "target_ineligible",
                "reason": preflight_row["target_eligibility_reason"],
            }
        ]
    record = preflight_row["task_record"]
    task_id = int(record["openml_task_id"])
    dataset_id = int(record["openml_dataset_id"])
    reasons = []
    sealed_lockbox = declaration["origin"] == "ctr23_sealed_lockbox"
    if sealed_lockbox:
        if task_id not in AUTHORIZED_LOCKBOX_TASK_IDS:
            reasons.append(
                {"kind": "lockbox_task_not_authorized", "match": task_id}
            )
    else:
        if task_id in set(spent["openml_task_ids"]):
            reasons.append(
                {"kind": "spent_openml_task_id", "match": task_id}
            )
        if dataset_id in set(spent["openml_dataset_ids"]):
            reasons.append(
                {"kind": "spent_openml_dataset_id", "match": dataset_id}
            )
        related_spent = sorted(
            set(int(value) for value in declaration["related_task_ids"])
            & set(spent["openml_task_ids"])
        )
        if related_spent:
            reasons.append(
                {"kind": "spent_related_task_id", "matches": related_spent}
            )
        name_hit = fresh._name_hit(record["normalized_name"], known_names)
        if name_hit is not None:
            reasons.append({"kind": "known_name", "match": name_hit})
    if dataset_id in set(exposure["openml_dataset_ids"]):
        reasons.append(
            {"kind": "chimeraboost_openml_dataset_id", "match": dataset_id}
        )
    exposure_name_hit = fresh._name_hit(
        record["normalized_name"],
        sorted(exposure["normalized_names"]),
    )
    if exposure_name_hit is not None:
        reasons.append(
            {"kind": "chimeraboost_known_name", "match": exposure_name_hit}
        )
    if _repository_literal_is_discriminating(record["dataset_name"]):
        for repository, revision, label in (
            (ROOT, prefreeze, "darkofit"),
            (CHIMERA_ROOT, chimera_head, "chimeraboost"),
        ):
            matches = fresh._git_grep(
                repository,
                revision,
                str(record["dataset_name"]),
            )
            if matches:
                unexpected = matches
                if sealed_lockbox and label == "darkofit":
                    unexpected = [
                        match
                        for match in matches
                        if _repository_match_path(match, revision)
                        not in LOCKBOX_DARKOFIT_REFERENCE_ALLOWLIST
                    ]
                if not unexpected:
                    continue
                reasons.append(
                    {
                        "kind": "repository_reference",
                        "repository": label,
                        "literal": str(record["dataset_name"]),
                        "paths": unexpected,
                    }
                )
    if record["fingerprint"]["canonicalization_ambiguous"]:
        reasons.append({"kind": "canonicalization_ambiguous"})
    near_matches = []
    for source in spent["task_records"]:
        if sealed_lockbox and int(source["openml_task_id"]) == task_id:
            continue
        evidence = ctr.near_match_evidence(
            record["fingerprint"],
            source["fingerprint"],
            **thresholds,
        )
        if evidence["ambiguous"]:
            near_matches.append(
                {
                    "source_task_id": int(source["openml_task_id"]),
                    **evidence,
                }
            )
    if near_matches:
        reasons.append(
            {"kind": "spent_near_lineage_alarm", "matches": near_matches}
        )
    return reasons


def _manual_contamination_adjudication(
    declaration: dict[str, Any],
    record: dict[str, Any],
    reasons: list[dict[str, Any]],
    *,
    exposure: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Apply the one frozen source-distinction exception, or fail closed."""
    manual = declaration.get("manual_contamination_adjudication")
    if manual is None:
        return reasons, None
    expected = common.COLLEGES_MANUAL_ADJUDICATION
    allowed_kinds = {"known_name", "chimeraboost_known_name"}
    matching = [
        row
        for row in reasons
        if row.get("kind") in allowed_kinds
        and row.get("match") == "colleges"
    ]
    remaining = [row for row in reasons if row not in matching]
    exact_boundary = (
        manual == expected
        and int(declaration["task_id"]) == 5166
        and int(declaration["dataset_id"]) == 538
        and int(record["openml_dataset_id"]) == 538
        and record["normalized_name"] == "colleges_usnews"
        and 538 not in set(exposure["openml_dataset_ids"])
        and 42727 in set(exposure["openml_dataset_ids"])
        and {row["kind"] for row in matching} == allowed_kinds
        and not remaining
    )
    if not exact_boundary:
        return (
            [
                *reasons,
                {
                    "kind": "manual_contamination_adjudication_failed",
                    "observed_name_alarm_kinds": sorted(
                        row["kind"] for row in matching
                    ),
                    "other_reason_kinds": sorted(
                        str(row.get("kind")) for row in remaining
                    ),
                },
            ],
            None,
        )
    applied = {
        **manual,
        "suppressed_exact_alarm_kinds": sorted(allowed_kinds),
        "suppressed_exact_match": "colleges",
        "automated_checks_performed": {
            "candidate_openml_dataset_id_absent_from_catalog": 538,
            "colliding_catalog_openml_dataset_ids_observed": [42727],
            "only_exact_name_containment_alarms_suppressed": True,
        },
        "declared_colliding_dataset_ids_not_independently_observed": [
            42159
        ],
        "manual_source_review_not_reperformed_by_builder": [
            "original source families remain distinct",
            "semantic fingerprints show no row or feature-table lineage match",
        ],
        "other_contamination_alarms": [],
    }
    return [], applied


def _prospective_near_matches(
    fingerprint: dict[str, Any],
    earlier: list[tuple[int, dict[str, Any]]],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    matches = []
    for earlier_id, earlier_fingerprint in earlier:
        evidence = ctr.near_match_evidence(
            fingerprint,
            earlier_fingerprint,
            **thresholds,
        )
        if evidence["ambiguous"]:
            matches.append(
                {
                    "earlier_task_id": int(earlier_id),
                    **evidence,
                }
            )
    return matches


def build(
    *,
    preflight_path: Path = DEFAULT_PREFLIGHT,
    require_clean_source: bool = True,
) -> dict[str, Any]:
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
    preflight, preflight_file_sha256 = common.secure_load_json(
        preflight_path
    )
    current_decision, power_design_file_sha256 = common.secure_load_json(
        common.POWER_DESIGN_DECISION
    )
    if (
        not isinstance(preflight, dict)
        or current_decision != preflight.get("power_design_decision")
    ):
        raise RuntimeError(
            "panel-3 preflight and power-decision snapshots disagree"
        )
    evidence_snapshots, evidence_sha256 = common.secure_snapshot_files(
        list(FROZEN_EVIDENCE)
    )
    evidence_payloads = {
        path: common.decode_json_bytes(
            evidence_snapshots[path.absolute()],
            source=path,
        )
        for path in SPENT_JSON_PATHS
    }
    _validate_preflight(
        preflight,
        declarations,
        declarations_file_sha256=source_sha256[
            str(common.DECLARATIONS.relative_to(ROOT))
        ],
        power_design_file_sha256=power_design_file_sha256,
        source_sha256=source_sha256,
        candidate_contract=candidate_contract,
    )

    head = _git(ROOT, "rev-parse", "HEAD")
    prefreeze = str(declarations["darkofit_prefreeze_head"])
    if not _is_ancestor(ROOT, prefreeze, head):
        raise RuntimeError("panel-3 head does not descend from prefreeze")
    _validate_post_preflight_boundary(
        preflight,
        preflight_path,
        head,
        require_clean_source=require_clean_source,
    )
    if require_clean_source and _git(
        ROOT, "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("panel-3 registry requires a clean DarkoFit tree")
    if require_clean_source:
        _validate_source_snapshots_at_head(head, source_sha256)
    chimera_head = str(declarations["chimeraboost_head"])
    if _git(CHIMERA_ROOT, "rev-parse", "HEAD") != chimera_head:
        raise RuntimeError("panel-3 ChimeraBoost head changed")
    if require_clean_source and _git(
        CHIMERA_ROOT, "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("panel-3 registry requires clean ChimeraBoost")

    chimera_snapshots, chimera_source_sha256 = (
        common.secure_snapshot_files(
            list(CHIMERA_EXPOSURE_PATHS),
            allowed_root=CHIMERA_ROOT,
        )
    )
    _validate_chimera_snapshot_at_head(
        chimera_source_sha256,
        head=chimera_head,
    )
    exposure = _chimera_exposure_catalog_from_snapshots(
        chimera_snapshots,
        chimera_source_sha256,
    )
    spent = spent_evidence(
        payloads=evidence_payloads,
        source_sha256=evidence_sha256,
    )
    known_names = sorted(
        set(exposure["normalized_names"])
        | set(spent["normalized_names"])
    )
    thresholds = evidence_payloads[CTR_DECLARATIONS][
        "near_match_thresholds"
    ]
    preflight_rows = _preflight_rows(preflight)
    records = []
    prospective_fingerprints: list[tuple[int, dict[str, Any]]] = []
    for declaration in declarations["candidates"]:
        task_id = int(declaration["task_id"])
        preflight_row = preflight_rows[task_id]
        reasons = _base_exclusion_reasons(
            declaration,
            preflight_row,
            spent=spent,
            exposure=exposure,
            known_names=known_names,
            thresholds=thresholds,
            prefreeze=prefreeze,
            chimera_head=chimera_head,
        )
        task_record = preflight_row.get("task_record")
        manual_applied = None
        if task_record is not None:
            reasons, manual_applied = _manual_contamination_adjudication(
                declaration,
                task_record,
                reasons,
                exposure=exposure,
            )
        if task_record is not None:
            prospective_matches = _prospective_near_matches(
                task_record["fingerprint"],
                prospective_fingerprints,
                thresholds,
            )
            if prospective_matches:
                reasons.append(
                    {
                        "kind": "prospective_near_lineage_alarm",
                        "matches": prospective_matches,
                    }
                )
        if manual_applied is not None and reasons:
            reasons.append(
                {
                    "kind": (
                        "manual_contamination_adjudication_invalidated_by_"
                        "prospective_alarm"
                    ),
                    "other_reason_kinds": sorted(
                        str(row.get("kind")) for row in reasons
                    ),
                }
            )
            manual_applied = None
        if task_record is not None:
            prospective_fingerprints.append(
                (task_id, task_record["fingerprint"])
            )
        records.append(
            {
                "task_id": task_id,
                "dataset_id": int(declaration["dataset_id"]),
                "dataset_name": (
                    task_record["dataset_name"]
                    if task_record is not None
                    else None
                ),
                "normalized_name": (
                    task_record["normalized_name"]
                    if task_record is not None
                    else declaration["expected_normalized_name"]
                ),
                "target_name": declaration["expected_target_name"],
                "expected_split_dimensions": declaration[
                    "expected_split_dimensions"
                ],
                "lineage_cluster": declaration["lineage_cluster"],
                "stratum": declaration["stratum"],
                "priority": int(declaration["priority"]),
                "related_task_ids": declaration["related_task_ids"],
                "semantic_sources": declaration["semantic_sources"],
                "semantic_caveat": declaration["semantic_caveat"],
                "origin": declaration["origin"],
                "selection_role": declaration["selection_role"],
                "feature_policy": declaration["feature_policy"],
                "ordinal_features": declarations[
                    "ordinal_features_by_task"
                ][str(task_id)],
                "declared_split_policy": declaration["split_policy"],
                "manual_contamination_adjudication": declaration.get(
                    "manual_contamination_adjudication"
                ),
                "manual_contamination_adjudication_applied": manual_applied,
                "status": "eligible" if not reasons else "excluded",
                "exclusion_reasons": reasons,
                "target_preflight_status": preflight_row["status"],
                "task_record": task_record,
            }
        )

    selected = select_first_four(records)
    decision = preflight["power_design_decision"]
    selected_task_ids = {
        int(row["task_id"]) for row in selected
    }
    authorized_slots = {
        int(row["task_id"]): row
        for row in decision["prospective_panel"]["slots"]
    }
    authorized_task_ids = set(authorized_slots)
    if selected_task_ids != authorized_task_ids:
        raise RuntimeError(
            "panel-3 target preflight changed the exact power-authorized "
            "primary task set"
        )

    for row in selected:
        task_id = int(row["task_id"])
        split_record = _load_task_record_with_splits(task_id)
        if split_record["fingerprint"] != row["task_record"]["fingerprint"]:
            raise RuntimeError(
                f"panel-3 task {task_id} drifted after target preflight"
            )
        if (
            split_record["official_splits"]["dimensions"]
            != row["expected_split_dimensions"]
        ):
            raise RuntimeError(
                f"panel-3 task {task_id} split dimensions changed"
            )
        row["task_record"] = split_record
        _task, X, categorical, ordered_view_sha256 = _load_task_feature_view(
            task_id,
            split_record,
        )
        _validate_preflight_ordered_view(
            task_id,
            preflight_rows[task_id],
            ordered_view_sha256,
        )
        row["ordered_task_view_sha256"] = ordered_view_sha256
        transformed, feature_attestation = (
            data_contract.apply_feature_policy(
                X,
                row["feature_policy"],
            )
        )
        retained_flags = data_contract.categorical_flags_after_policy(
            list(X.columns),
            categorical,
            feature_attestation,
        )
        row["feature_policy_attestation"] = feature_attestation
        row["resolved_categorical_columns"] = [
            str(name)
            for name, declared, dtype in zip(
                transformed.columns,
                retained_flags,
                transformed.dtypes,
                strict=True,
            )
            if bool(declared)
            or not pd.api.types.is_numeric_dtype(dtype)
        ]
        row["split_policy"] = _freeze_split_policy(
            X,
            row["declared_split_policy"],
        )
        applicability = common.t5_size_gate_applicability(
            row,
            minimum_rows=common.t5_minimum_outer_training_rows(
                candidate_contract
            ),
        )
        authorized_slot = authorized_slots[task_id]
        if (
            authorized_slot.get("lineage_cluster")
            != row["lineage_cluster"]
            or authorized_slot.get("stratum") != row["stratum"]
            or authorized_slot.get("t5_size_gate_applicability")
            != applicability
        ):
            raise RuntimeError(
                f"panel-3 task {task_id} T5 size-gate applicability "
                "differs from the power design"
            )
        row["t5_size_gate_applicability"] = applicability

    coordinates = [
        {
            "task_id": int(row["task_id"]),
            "repeat": 0,
            "fold": fold,
            "sample": 0,
        }
        for row in selected
        for fold in declarations["coordinate_folds"]
    ]
    power = decision

    selected_by_stratum = defaultdict(list)
    for row in selected:
        selected_by_stratum[row["stratum"]].append(row["task_id"])
    artifact = {
        "schema_version": 1,
        "registry_name": declarations["registry_name"],
        "created_from_clean_sources": require_clean_source,
        "sources": {
            "darkofit_registry_head": head,
            "darkofit_model_head": preflight["sources"][
                "darkofit_execution_head"
            ],
            "darkofit_prefreeze_head": prefreeze,
            "chimeraboost_head": chimera_head,
        },
        "source_sha256": source_sha256,
        "frozen_evidence_sha256": spent["source_sha256"],
        "target_preflight_path": str(preflight_path.relative_to(ROOT)),
        "target_preflight_file_sha256": preflight_file_sha256,
        "target_preflight_sha256": preflight["target_preflight_sha256"],
        "power_design_path": preflight["power_design_path"],
        "power_design_file_sha256": preflight[
            "power_design_file_sha256"
        ],
        "power_design_decision_sha256": decision["decision_sha256"],
        "power_design_decision": decision,
        "retained_candidates": decision["retained_candidates"],
        "pre_h1_target_statistic_exclusions": copy.deepcopy(
            declarations["pre_h1_target_statistic_exclusions"]
        ),
        "spent_evidence_counts": {
            "task_records": len(spent["task_records"]),
            "openml_task_ids": len(spent["openml_task_ids"]),
            "openml_dataset_ids": len(spent["openml_dataset_ids"]),
            "normalized_names": len(spent["normalized_names"]),
        },
        "exposure_catalog": exposure,
        "lockbox_darkofit_reference_allowlist": sorted(
            LOCKBOX_DARKOFIT_REFERENCE_ALLOWLIST
        ),
        "declaration_count": len(records),
        "selected_task_count": len(selected),
        "selected_lineage_count": len(
            {row["lineage_cluster"] for row in selected}
        ),
        "selected_by_stratum": dict(selected_by_stratum),
        "coordinate_count": len(coordinates),
        "coordinates": coordinates,
        "tasks": records,
        "power_analysis": power,
        "candidate_contract": candidate_contract,
        "selection_rule": (
            "first four target-eligible and contamination-clean declarations "
            "in frozen priority order within each stratum"
        ),
        "outcome_blind": True,
        "eligibility_aware": True,
        "target_statistics_used": False,
        "candidate_or_control_models_fitted": False,
        "candidate_or_control_outcomes_inspected": False,
        "lockbox_metadata_used": True,
        "lockbox_outcomes_used": False,
        "registry_freeze_complete": True,
        "runner_implementation_complete": True,
        "confirmation_run_authorized": True,
        "default_promotion_authorized": False,
    }
    artifact = common.bind_artifact_sha256(
        artifact,
        "registry_sha256",
    )
    _recheck_registry_inputs(
        artifact,
        preflight_path=preflight_path,
    )
    return artifact


def deterministic_registry_projection(
    artifact: dict[str, Any],
) -> dict[str, Any]:
    """Return the exact builder-controlled payload, excluding freeze mechanics."""
    if not isinstance(artifact, dict):
        raise RuntimeError("panel-3 registry must be an object")
    projected = copy.deepcopy(artifact)
    projected.pop("registry_sha256", None)
    projected.pop("created_from_clean_sources", None)
    sources = projected.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("panel-3 registry source ledger is missing")
    sources.pop("darkofit_registry_head", None)
    return projected


def validate_deterministic_registry_output(
    artifact: dict[str, Any],
    *,
    preflight_path: Path = DEFAULT_PREFLIGHT,
) -> None:
    """Rebuild the outcome-blind registry and compare every controlled byte."""
    expected = build(
        preflight_path=preflight_path,
        require_clean_source=False,
    )
    if deterministic_registry_projection(artifact) != (
        deterministic_registry_projection(expected)
    ):
        raise RuntimeError(
            "panel-3 registry differs from deterministic builder output"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        type=Path,
        default=DEFAULT_PREFLIGHT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        args.preflight.expanduser().absolute() != DEFAULT_PREFLIGHT
        or args.output.expanduser().absolute() != DEFAULT_OUTPUT
    ):
        raise RuntimeError("panel-3 registry path changed")
    common.validate_create_path(args.output)
    artifact = build(preflight_path=args.preflight)
    _recheck_registry_inputs(
        artifact,
        preflight_path=args.preflight,
    )
    common.atomic_create(args.output, ctr.canonical_json_bytes(artifact))
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().absolute()),
                "registry_sha256": artifact["registry_sha256"],
                "selected_tasks": artifact["selected_task_count"],
                "coordinates": artifact["coordinate_count"],
                "retained_candidates": artifact[
                    "retained_candidates"
                ],
                "power_decision_sha256": artifact[
                    "power_design_decision_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
