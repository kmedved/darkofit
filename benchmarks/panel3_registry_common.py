"""Shared, prospective-only helpers for the panel-3 confirmation freeze."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import stat
from collections import Counter
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from benchmarks import build_ctr23_contamination_registry as ctr
from benchmarks import panel3_data_contract as data_contract
from benchmarks.campaign_lib import provenance


ROOT = Path(__file__).resolve().parents[1]
DECLARATIONS = ROOT / "benchmarks" / "panel3_registry_declarations.json"
PROTOCOL = ROOT / "benchmarks" / "panel3_registry_protocol.md"
COMMON = ROOT / "benchmarks" / "panel3_registry_common.py"
DATA_CONTRACT = ROOT / "benchmarks" / "panel3_data_contract.py"
CAMPAIGN_LIB_INIT = ROOT / "benchmarks" / "campaign_lib" / "__init__.py"
CAMPAIGN_LIB_PROVENANCE = (
    ROOT / "benchmarks" / "campaign_lib" / "provenance.py"
)
CANDIDATE_CONTRACT = (
    ROOT / "benchmarks" / "panel3_candidate_contract.json"
)
ENVIRONMENT_CONTRACT = (
    ROOT / "benchmarks" / "panel3_environment_contract.json"
)
POWER_DESIGN_CONTRACT = (
    ROOT / "benchmarks" / "panel3_power_design_contract.json"
)
POWER_DESIGN_PROTOCOL = (
    ROOT / "benchmarks" / "panel3_power_design_protocol.md"
)
POWER_DESIGN_BUILDER = (
    ROOT / "benchmarks" / "build_panel3_power_design.py"
)
POWER_DESIGN_DECISION = (
    ROOT / "benchmarks" / "panel3_power_design_decision.json"
)
PREFLIGHT_BUILDER = ROOT / "benchmarks" / "preflight_panel3_registry.py"
REGISTRY_BUILDER = ROOT / "benchmarks" / "build_panel3_registry.py"
RUNNER = ROOT / "benchmarks" / "run_panel3_confirmation.py"
ANALYZER = ROOT / "benchmarks" / "analyze_panel3_confirmation.py"
TARGET_PREFLIGHT_HELPER = (
    ROOT / "benchmarks" / "confirmation_target_preflight.py"
)
CTR_CONTAMINATION_BUILDER = (
    ROOT / "benchmarks" / "build_ctr23_contamination_registry.py"
)
FRESH_REGISTRY_BUILDER = (
    ROOT / "benchmarks" / "build_fresh_confirmation_registry.py"
)
INVALID_DECLARATION_DRAFT = (
    ROOT / "benchmarks" / "panel3_registry_declarations_invalid_draft.md"
)
T5_COMPOSITE_PROTOCOL = (
    ROOT / "benchmarks" / "t5_composite_registry_protocol.md"
)
T5_COMPOSITE_RUNNER = (
    ROOT / "benchmarks" / "run_t5_composite_confirmation.py"
)
SMOOTH_CROSS_PROTOCOL = (
    ROOT / "benchmarks" / "smooth_cross_features_protocol.md"
)
SMOOTH_CROSS_ANALYSIS = (
    ROOT / "benchmarks" / "smooth_cross_margin_analysis.json"
)
PANEL3_SOURCE_PATHS = (
    DECLARATIONS,
    PROTOCOL,
    COMMON,
    DATA_CONTRACT,
    CAMPAIGN_LIB_INIT,
    CAMPAIGN_LIB_PROVENANCE,
    CANDIDATE_CONTRACT,
    ENVIRONMENT_CONTRACT,
    POWER_DESIGN_CONTRACT,
    POWER_DESIGN_PROTOCOL,
    POWER_DESIGN_BUILDER,
    PREFLIGHT_BUILDER,
    TARGET_PREFLIGHT_HELPER,
    CTR_CONTAMINATION_BUILDER,
    FRESH_REGISTRY_BUILDER,
    REGISTRY_BUILDER,
    RUNNER,
    ANALYZER,
    T5_COMPOSITE_PROTOCOL,
    T5_COMPOSITE_RUNNER,
    SMOOTH_CROSS_PROTOCOL,
    SMOOTH_CROSS_ANALYSIS,
    INVALID_DECLARATION_DRAFT,
)
T5_REGISTRY = ROOT / "benchmarks" / "t5_composite_registry.json"

STRATA = ("smooth_numeric", "mixed_categorical", "applied_noisy")
REQUIRED_PER_STRATUM = 4
POWER_SEED = 20_260_717
POWER_SIMULATIONS = 200_000
POWER_LINEAGES = 12
MIN_POWER = 0.80
QUALITY_BAR = 0.995
UNCERTAINTY_BAR = 1.002
LOO_BAR = 0.998
HARM_BAR = 1.005
CANDIDATE_COUNT = 2
DECLARATION_COUNT = 17
DECLARATION_SCHEMA_KEYS = {
    "schema_version",
    "registry_name",
    "source_audit_version",
    "darkofit_prefreeze_head",
    "chimeraboost_head",
    "required_per_stratum",
    "selection_rule",
    "ordinal_features_by_task",
    "coordinate_folds",
    "panel_split_dimensions",
    "stratum_order",
    "ctr23_lockbox_policy",
    "pre_h1_target_statistic_exclusions",
    "candidates",
}
DECLARATION_ROW_KEYS = {
    "priority",
    "task_id",
    "dataset_id",
    "origin",
    "expected_normalized_name",
    "expected_target_name",
    "expected_split_dimensions",
    "lineage_cluster",
    "stratum",
    "related_task_ids",
    "semantic_sources",
    "semantic_caveat",
    "selection_role",
    "feature_policy",
    "split_policy",
}
REGISTRY_NAME = "darkofit_panel3_dual_candidate_confirmation_v1"
SOURCE_AUDIT_VERSION = "hybrid_6_ctr23_plus_6_new_v1"
DARKOFIT_PREFREEZE_HEAD = "120a2e7d0a87c03e6b308ea0a6f4bbd2acb58b16"
CHIMERABOOST_HEAD = "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d"
PANEL_SPLIT_DIMENSIONS = {"repeats": 1, "folds": 3, "samples": 1}
DECLARATION_COUNTS = {
    "smooth_numeric": 6,
    "mixed_categorical": 4,
    "applied_noisy": 7,
}
LOCKBOX_SELECTED_TASK_IDS = [
    361247,
    361253,
    361254,
    361617,
    361261,
    361618,
]
CTR23_LOCKBOX_POLICY = {
    "status": "not_opened_by_source_review",
    "selected_task_ids": LOCKBOX_SELECTED_TASK_IDS,
    "excluded_after_refresh": {
        "361264": {
            "normalized_name": "socmob",
            "reason": (
                "Current ChimeraBoost v0.15.0 explicitly inspected and cut "
                "socmob in benchmarks/HIGHCARD_PLAN.md, so the old seven-task "
                "catalog exception is stale."
            ),
        }
    },
    "claim_boundary": (
        "The four simulator or trajectory tasks support only the semantic "
        "caveats recorded on their declarations. They do not support broad "
        "real-world deployment claims."
    ),
}
PRE_H1_TARGET_STATISTIC_EXCLUSIONS = [
    {
        "task_id": 363370,
        "dataset_id": 46657,
        "lineage_cluster": "kaggle_google_quest_qa_annotations",
        "stratum": "mixed_categorical",
        "exposure_kind": "parquet_footer_target_min_max_statistics",
        "reason": "target_parquet_footer_min_max_observed_before_h1",
        "replacement_task_id": 359931,
    },
    {
        "task_id": 363377,
        "dataset_id": 46660,
        "lineage_cluster": "kaggle_mercari_price_suggestion",
        "stratum": "mixed_categorical",
        "exposure_kind": "parquet_footer_target_min_max_statistics",
        "reason": "target_parquet_footer_min_max_observed_before_h1",
        "replacement_task_id": 360993,
    },
    {
        "task_id": 363495,
        "dataset_id": 46787,
        "lineage_cluster": "kaggle_sberbank_russian_housing_market",
        "stratum": "applied_noisy",
        "exposure_kind": "parquet_footer_target_min_max_statistics",
        "reason": "target_parquet_footer_min_max_observed_before_h1",
        "replacement_task_id": 4851,
    },
]
FAMILYWISE_ONE_SIDED_ALPHA = 0.05
PER_CANDIDATE_ONE_SIDED_ALPHA = (
    FAMILYWISE_ONE_SIDED_ALPHA / CANDIDATE_COUNT
)
BONFERRONI_Z = NormalDist().inv_cdf(
    1.0 - PER_CANDIDATE_ONE_SIDED_ALPHA
)
EXPECTED_EFFECT_POOL_SHA256 = (
    "76737ab531b139e1ab2db7ae717bbd65409251e0a8773db3e2994951e52e4e0a"
)
EXPECTED_CROSS_EFFECT_POOL_SHA256 = (
    "1af3835e31ac5ccdc95922c604854b5946e0475fd40c765ae3d21b6badbcc767"
)
SELECTION_RULE = (
    "Within each stratum, select the first four eligible declarations by "
    "ascending priority. selection_role is descriptive only and must agree "
    "with priority; consumers must not use it to select tasks."
)
COLLEGES_MANUAL_ADJUDICATION = {
    "status": "source_distinct_name_containment_exception",
    "candidate_dataset_id": 538,
    "candidate_lineage": (
        "ASA 1995 Data Analysis Exposition / U.S. News 1993-94 "
        "graduation-rate data"
    ),
    "colliding_catalog_normalized_name": "colleges",
    "colliding_catalog_dataset_ids": [42159, 42727],
    "colliding_catalog_lineage": (
        "U.S. College Scorecard percent-Pell-grant data"
    ),
    "basis": (
        "The sources, years, targets, dataset IDs, and schemas are distinct. "
        "The alarm is caused only by conservative normalized-name containment."
    ),
    "required_registry_checks": [
        "dataset IDs remain distinct",
        "original source families remain distinct",
        "semantic fingerprints show no row or feature-table lineage match",
    ],
}


def load_json(path: Path) -> Any:
    """Load a campaign JSON artifact with a fail-closed numeric grammar."""
    try:
        return decode_json_bytes(path.read_bytes(), source=path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid JSON artifact {path}: {exc}") from exc


def decode_json_bytes(payload: bytes, *, source: Path | str) -> Any:
    """Strictly decode an already-captured JSON byte snapshot."""
    try:
        return provenance.strict_json_loads(payload.decode("utf-8"))
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise RuntimeError(f"invalid JSON artifact {source}: {exc}") from exc


def t5_minimum_outer_training_rows(
    candidate_contract: dict[str, Any] | None = None,
) -> int:
    """Return the exact T5 size gate from the frozen candidate contract."""
    contract = (
        load_json(CANDIDATE_CONTRACT)
        if candidate_contract is None
        else candidate_contract
    )
    candidates = contract.get("candidates") if isinstance(contract, dict) else None
    rows = (
        [
            row
            for row in candidates
            if isinstance(row, dict)
            and row.get("name") == "t5_composite_policy"
        ]
        if isinstance(candidates, list)
        else []
    )
    minimum = (
        rows[0].get("definition", {}).get("minimum_outer_training_rows")
        if len(rows) == 1 and isinstance(rows[0].get("definition"), dict)
        else None
    )
    if type(minimum) is not int or minimum != 2_000:
        raise RuntimeError("panel-3 T5 outer-training size gate changed")
    return minimum


def t5_size_gate_applicability(
    task_row: dict[str, Any],
    *,
    folds: tuple[int, ...] = (0, 1, 2),
    minimum_rows: int | None = None,
) -> list[bool]:
    """Derive the per-coordinate T5 gate from frozen split metadata."""
    if not isinstance(task_row, dict):
        raise RuntimeError("panel-3 selected task row is invalid")
    policy = task_row.get("split_policy")
    if not isinstance(policy, dict):
        raise RuntimeError("panel-3 selected split policy is invalid")
    if policy.get("kind") == "openml_official":
        task_record = task_row.get("task_record")
        official = (
            task_record.get("official_splits")
            if isinstance(task_record, dict)
            else None
        )
        source = (
            official.get("coordinates")
            if isinstance(official, dict)
            else None
        )
    elif policy.get("kind") == "frozen_explicit":
        source = policy.get("coordinates")
    else:
        raise RuntimeError("panel-3 selected split policy kind changed")
    if not isinstance(source, list):
        raise RuntimeError("panel-3 selected split ledger is missing")
    threshold = (
        t5_minimum_outer_training_rows()
        if minimum_rows is None
        else minimum_rows
    )
    if type(threshold) is not int or threshold <= 0:
        raise RuntimeError("panel-3 T5 outer-training size gate is invalid")
    result = []
    for fold in folds:
        matches = [
            row
            for row in source
            if isinstance(row, dict)
            and row.get("repeat") == 0
            and row.get("fold") == fold
            and row.get("sample") == 0
        ]
        if (
            len(matches) != 1
            or type(matches[0].get("train_size")) is not int
            or matches[0]["train_size"] <= 0
        ):
            raise RuntimeError(
                "panel-3 selected split train-size ledger changed"
            )
        result.append(matches[0]["train_size"] >= threshold)
    return result


def sha256_file(path: Path) -> str:
    return provenance.file_sha256(path)


def artifact_sha256(artifact: dict[str, Any], field: str) -> str:
    payload = {key: value for key, value in artifact.items() if key != field}
    return ctr.sha256_json(payload)


def bind_artifact_sha256(
    artifact: dict[str, Any],
    field: str,
) -> dict[str, Any]:
    if field in artifact:
        raise ValueError(f"artifact already contains {field}")
    artifact[field] = artifact_sha256(artifact, field)
    return artifact


def verify_artifact_sha256(
    artifact: dict[str, Any],
    field: str,
) -> None:
    if artifact.get(field) != artifact_sha256(artifact, field):
        raise RuntimeError(f"{field} binding changed")


def _secure_output_parent(
    path: Path,
    allowed_root: Path,
) -> tuple[Path, Path, int]:
    """Open a lexical parent by descriptor without following symlink ancestors."""
    root = Path(allowed_root).expanduser().resolve(strict=True)
    lexical = Path(os.path.abspath(os.path.expanduser(path)))
    if not lexical.is_relative_to(root) or lexical == root:
        raise RuntimeError(
            f"panel-3 output must remain below {root}: {lexical}"
        )
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root, directory_flags)
    current = root
    try:
        for part in lexical.parent.relative_to(root).parts:
            candidate = current / part
            try:
                os.mkdir(part, dir_fd=descriptor)
            except FileExistsError:
                pass
            try:
                child_descriptor = os.open(
                    part,
                    directory_flags,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise RuntimeError(
                    "refusing symlink ancestor or unsafe directory for "
                    f"panel-3 output: {candidate}"
                ) from exc
            os.close(descriptor)
            descriptor = child_descriptor
            current = candidate
    except BaseException:
        os.close(descriptor)
        raise
    return lexical, root, descriptor


def _secure_input_parent(
    path: Path,
    allowed_root: Path,
) -> tuple[Path, Path, int]:
    """Open an existing lexical parent without following symlink ancestors."""
    root = Path(allowed_root).expanduser().resolve(strict=True)
    lexical = Path(os.path.abspath(os.path.expanduser(path)))
    if not lexical.is_relative_to(root) or lexical == root:
        raise RuntimeError(
            f"panel-3 input must remain below {root}: {lexical}"
        )
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root, directory_flags)
    current = root
    try:
        for part in lexical.parent.relative_to(root).parts:
            candidate = current / part
            try:
                child_descriptor = os.open(
                    part,
                    directory_flags,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise RuntimeError(
                    "refusing symlink ancestor or unsafe directory for "
                    f"panel-3 input: {candidate}"
                ) from exc
            os.close(descriptor)
            descriptor = child_descriptor
            current = candidate
    except BaseException:
        os.close(descriptor)
        raise
    return lexical, root, descriptor


def secure_read_bytes(
    path: Path,
    *,
    allowed_root: Path = ROOT,
) -> bytes:
    """Read one regular file through held, no-follow directory descriptors."""
    lexical, _root, parent_descriptor = _secure_input_parent(
        Path(path),
        Path(allowed_root),
    )
    descriptor = None
    try:
        try:
            descriptor = os.open(
                lexical.name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_descriptor,
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise RuntimeError(
                    f"panel-3 input is not a regular file: {lexical}"
                )
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = None
                return handle.read()
        except OSError as exc:
            raise RuntimeError(
                f"refusing symlink leaf or unsafe panel-3 input: {lexical}"
            ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def secure_load_json(
    path: Path,
    *,
    allowed_root: Path = ROOT,
) -> tuple[Any, str]:
    """Strictly decode and hash one securely opened JSON byte snapshot."""
    payload = secure_read_bytes(path, allowed_root=allowed_root)
    value = decode_json_bytes(payload, source=path)
    return value, hashlib.sha256(payload).hexdigest()


def secure_snapshot_files(
    paths: tuple[Path, ...] | list[Path],
    *,
    allowed_root: Path = ROOT,
) -> tuple[dict[Path, bytes], dict[str, str]]:
    """Capture regular-file bytes and their exact repository-relative hashes."""
    snapshots: dict[Path, bytes] = {}
    digests: dict[str, str] = {}
    for path in paths:
        normalized = Path(path).expanduser().absolute()
        try:
            relative = str(normalized.relative_to(allowed_root.absolute()))
        except ValueError as exc:
            raise RuntimeError(
                f"panel-3 snapshot input is outside its root: {normalized}"
            ) from exc
        if normalized in snapshots or relative in digests:
            raise RuntimeError(f"panel-3 snapshot input repeats: {relative}")
        payload = secure_read_bytes(normalized, allowed_root=allowed_root)
        snapshots[normalized] = payload
        digests[relative] = hashlib.sha256(payload).hexdigest()
    return snapshots, digests


def recheck_snapshot_files(
    paths: tuple[Path, ...] | list[Path],
    expected_sha256: dict[str, str],
    *,
    allowed_root: Path = ROOT,
) -> None:
    """Fail if any securely reopened input differs from its captured bytes."""
    _snapshots, observed = secure_snapshot_files(
        paths,
        allowed_root=allowed_root,
    )
    if observed != expected_sha256:
        raise RuntimeError("panel-3 input snapshot changed before publication")


def _require_absent_leaf(path: Path, parent_descriptor: int) -> None:
    try:
        os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    raise FileExistsError(f"refusing existing output: {path}")


def validate_create_path(
    path: Path,
    *,
    allowed_root: Path = ROOT,
) -> Path:
    """Validate and prepare a create-only output path without writing its leaf."""
    lexical, _root, parent_descriptor = _secure_output_parent(
        Path(path),
        Path(allowed_root),
    )
    try:
        _require_absent_leaf(lexical, parent_descriptor)
        return lexical
    finally:
        os.close(parent_descriptor)


def ensure_output_directory(
    path: Path,
    *,
    allowed_root: Path = ROOT,
) -> Path:
    """Create a directory below ``allowed_root`` without symlink traversal."""
    lexical = Path(os.path.abspath(os.path.expanduser(path)))
    probe = lexical / ".panel3-directory-boundary"
    prepared, _root, descriptor = _secure_output_parent(
        probe,
        Path(allowed_root),
    )
    try:
        if prepared.parent != lexical:
            raise RuntimeError(
                f"unsafe panel-3 output directory: {lexical}"
            )
        return lexical
    finally:
        os.close(descriptor)


def atomic_create(
    path: Path,
    payload: bytes,
    *,
    allowed_root: Path = ROOT,
) -> None:
    """Publish one immutable file below ``allowed_root`` without symlink traversal."""
    path, _root, parent_descriptor = _secure_output_parent(
        Path(path),
        Path(allowed_root),
    )
    descriptor = None
    temporary_name = (
        f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    )
    try:
        _require_absent_leaf(path, parent_descriptor)
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        os.fsync(parent_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)


def _validate_split_declaration(policy: Any) -> None:
    if policy == {"kind": "openml_official"}:
        return
    if (
        not isinstance(policy, dict)
        or set(policy) != {"kind", "constructor"}
        or policy["kind"] != "target_free_split_construction_v1"
        or not isinstance(policy["constructor"], dict)
    ):
        raise RuntimeError("panel-3 split declaration is invalid")
    constructor = policy["constructor"]
    kind = constructor.get("kind")
    if kind == "shuffled_kfold_v1":
        if (
            set(constructor)
            != {
                "kind",
                "n_splits",
                "shuffle",
                "random_state",
                "allow_unused_rows",
            }
            or constructor["n_splits"] != 3
            or constructor["shuffle"] is not True
            or constructor["random_state"] != POWER_SEED
            or constructor["allow_unused_rows"] is not False
        ):
            raise RuntimeError("panel-3 shuffled split declaration changed")
        return
    if kind == "size_balanced_group_kfold_v1":
        if (
            set(constructor)
            != {
                "kind",
                "n_splits",
                "group_key",
                "group_order",
                "fold_assignment",
                "allow_unused_rows",
            }
            or constructor["n_splits"] != 3
            or constructor["group_order"]
            != "descending_row_count_then_group_sha256"
            or constructor["fold_assignment"]
            != "minimum_row_count_then_lowest_fold"
            or constructor["allow_unused_rows"] is not False
        ):
            raise RuntimeError("panel-3 grouped split declaration changed")
        data_contract.validate_group_key_spec(constructor["group_key"])
        return
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
        kind != "expanding_unique_datetime_blocks_v1"
        or set(constructor)
        != {
            "kind",
            "source_column",
            "format",
            "utc",
            "block_count",
            "never_split_equal_values",
            "folds",
            "allow_unused_rows",
        }
        or not isinstance(constructor["source_column"], str)
        or not constructor["source_column"]
        or not isinstance(constructor["format"], str)
        or not constructor["format"]
        or type(constructor["utc"]) is not bool
        or constructor["block_count"] != 4
        or constructor["never_split_equal_values"] is not True
        or constructor["folds"] != expected_folds
        or constructor["allow_unused_rows"] is not True
    ):
        raise RuntimeError("panel-3 chronological split declaration changed")


def validate_declarations(
    declarations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if declarations is None:
        declarations = load_json(DECLARATIONS)
    if not isinstance(declarations, dict) or set(declarations) != DECLARATION_SCHEMA_KEYS:
        raise RuntimeError("panel-3 declaration top-level schema changed")
    if type(declarations["schema_version"]) is not int or declarations["schema_version"] != 1:
        raise RuntimeError("unsupported panel-3 declaration schema")
    if declarations["registry_name"] != REGISTRY_NAME:
        raise RuntimeError("panel-3 registry name changed")
    if declarations["source_audit_version"] != SOURCE_AUDIT_VERSION:
        raise RuntimeError("panel-3 source-audit boundary changed")
    if declarations["darkofit_prefreeze_head"] != DARKOFIT_PREFREEZE_HEAD:
        raise RuntimeError("panel-3 DarkoFit prefreeze head changed")
    if declarations["chimeraboost_head"] != CHIMERABOOST_HEAD:
        raise RuntimeError("panel-3 ChimeraBoost audit head changed")
    if declarations["required_per_stratum"] != REQUIRED_PER_STRATUM:
        raise RuntimeError("panel-3 required stratum size changed")
    if declarations["coordinate_folds"] != [0, 1, 2]:
        raise RuntimeError("panel-3 coordinate folds changed")
    if declarations["panel_split_dimensions"] != PANEL_SPLIT_DIMENSIONS:
        raise RuntimeError("panel-3 split dimensions changed")
    if declarations["stratum_order"] != list(STRATA):
        raise RuntimeError("panel-3 stratum order changed")
    if declarations["selection_rule"] != SELECTION_RULE:
        raise RuntimeError("panel-3 selection rule changed")
    if declarations["ctr23_lockbox_policy"] != CTR23_LOCKBOX_POLICY:
        raise RuntimeError("panel-3 CTR23 lockbox policy changed")
    exclusions = declarations["pre_h1_target_statistic_exclusions"]
    if exclusions != PRE_H1_TARGET_STATISTIC_EXCLUSIONS:
        raise RuntimeError(
            "panel-3 pre-H1 target-statistic exclusion ledger changed"
        )
    candidates = declarations["candidates"]
    if not isinstance(candidates, list) or len(candidates) != DECLARATION_COUNT:
        raise RuntimeError("panel-3 candidate declaration count changed")
    for row in candidates:
        expected_keys = set(DECLARATION_ROW_KEYS)
        if isinstance(row, dict) and row.get("task_id") == 5166:
            expected_keys.add("manual_contamination_adjudication")
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise RuntimeError("panel-3 candidate declaration schema changed")
    task_ids = [int(row["task_id"]) for row in candidates]
    dataset_ids = [int(row["dataset_id"]) for row in candidates]
    lineages = [str(row["lineage_cluster"]) for row in candidates]
    if (
        len(task_ids) != len(set(task_ids))
        or len(dataset_ids) != len(set(dataset_ids))
        or len(lineages) != len(set(lineages))
    ):
        raise RuntimeError(
            "panel-3 tasks, datasets, and lineage clusters must be unique"
        )
    excluded_task_ids = {row["task_id"] for row in exclusions}
    excluded_dataset_ids = {row["dataset_id"] for row in exclusions}
    excluded_lineages = {row["lineage_cluster"] for row in exclusions}
    related_task_ids = {
        value for row in candidates for value in row["related_task_ids"]
    }
    if (
        excluded_task_ids.intersection(task_ids)
        or excluded_task_ids.intersection(related_task_ids)
        or excluded_dataset_ids.intersection(dataset_ids)
        or excluded_lineages.intersection(lineages)
    ):
        raise RuntimeError(
            "panel-3 pre-H1 excluded lineage remains declaration-eligible"
        )
    selected_rows_by_task = {
        int(row["task_id"]): row
        for row in candidates
        if row["selection_role"] == "selected"
    }
    replacement_task_ids = {
        row["replacement_task_id"] for row in exclusions
    }
    if (
        len(replacement_task_ids) != len(exclusions)
        or not replacement_task_ids.issubset(selected_rows_by_task)
        or any(
            selected_rows_by_task[row["replacement_task_id"]]["stratum"]
            != row["stratum"]
            for row in exclusions
        )
    ):
        raise RuntimeError(
            "panel-3 pre-H1 replacement is not unique, selected, and "
            "same-stratum"
        )
    ordinal_features = declarations.get("ordinal_features_by_task")
    if (
        not isinstance(ordinal_features, dict)
        or set(ordinal_features) != {str(value) for value in task_ids}
        or any(value != {} for value in ordinal_features.values())
    ):
        raise RuntimeError(
            "panel-3 per-task ordinal declarations changed"
        )
    counts = Counter(str(row["stratum"]) for row in candidates)
    if dict(counts) != DECLARATION_COUNTS:
        raise RuntimeError("panel-3 per-stratum declaration counts changed")
    lockbox_task_ids = [
        int(row["task_id"])
        for row in candidates
        if row["origin"] == "ctr23_sealed_lockbox"
    ]
    if lockbox_task_ids != LOCKBOX_SELECTED_TASK_IDS:
        raise RuntimeError("panel-3 selected CTR23 lockbox tasks changed")
    for stratum in STRATA:
        priorities = [
            int(row["priority"])
            for row in candidates
            if row["stratum"] == stratum
        ]
        if priorities != list(range(1, len(priorities) + 1)):
            raise RuntimeError(
                f"panel-3 {stratum} reserve priority is not contiguous"
            )
    for row in candidates:
        if type(row["task_id"]) is not int or type(row["dataset_id"]) is not int:
            raise RuntimeError("panel-3 task and dataset IDs must be integers")
        if row["origin"] not in {
            "ctr23_sealed_lockbox",
            "new_source_reviewed",
        }:
            raise RuntimeError("panel-3 declaration origin is invalid")
        if type(row["priority"]) is not int or row["priority"] <= 0:
            raise RuntimeError("panel-3 priorities must be positive integers")
        for field in (
            "expected_normalized_name",
            "expected_target_name",
            "lineage_cluster",
        ):
            if not isinstance(row[field], str) or not row[field].strip():
                raise RuntimeError(
                    f"panel-3 candidate {field} must be a nonempty string"
                )
        if (
            not isinstance(row["related_task_ids"], list)
            or row["task_id"] not in row["related_task_ids"]
            or any(type(value) is not int for value in row["related_task_ids"])
            or len(row["related_task_ids"]) != len(set(row["related_task_ids"]))
        ):
            raise RuntimeError("panel-3 related task IDs are invalid")
        expected_role = (
            "selected"
            if int(row["priority"]) <= REQUIRED_PER_STRATUM
            else "reserve"
        )
        if row["selection_role"] != expected_role:
            raise RuntimeError(
                "panel-3 descriptive selection role conflicts with priority"
            )
        if (
            not isinstance(row["semantic_caveat"], str)
            or not row["semantic_caveat"].strip()
        ):
            raise RuntimeError("panel-3 semantic caveat is missing")
        data_contract.validate_feature_policy(row["feature_policy"])
        _validate_split_declaration(row["split_policy"])
        if row["origin"] == "ctr23_sealed_lockbox" and (
            row["feature_policy"] != {"kind": "none"}
            or row["split_policy"] != {"kind": "openml_official"}
        ):
            raise RuntimeError(
                "panel-3 lockbox task must preserve its exact task view"
            )
        manual = row.get("manual_contamination_adjudication")
        if int(row["task_id"]) == 5166:
            if manual != COLLEGES_MANUAL_ADJUDICATION:
                raise RuntimeError(
                    "panel-3 colleges adjudication changed"
                )
        elif manual is not None:
            raise RuntimeError(
                "panel-3 unexpected manual contamination exception"
            )
        dimensions = row["expected_split_dimensions"]
        if (
            not isinstance(dimensions, dict)
            or set(dimensions) != {"repeats", "folds", "samples"}
            or any(type(value) is not int or value <= 0 for value in dimensions.values())
            or dimensions["folds"] < 3
        ):
            raise RuntimeError(
                "panel-3 expected split dimensions are invalid"
            )
        if (
            not isinstance(row["semantic_sources"], list)
            or not row["semantic_sources"]
            or any(
                not isinstance(value, str)
                or not value.startswith("https://")
                for value in row["semantic_sources"]
            )
        ):
            raise RuntimeError("panel-3 semantic sources must use HTTPS")
    return declarations


def _composite_effect_profiles() -> list[dict[str, Any]]:
    payload = load_json(T5_REGISTRY)
    profiles = payload["power_analysis"]["effect_profiles"]
    if ctr.sha256_json(profiles) != EXPECTED_EFFECT_POOL_SHA256:
        raise RuntimeError("frozen T5 plausible-effect pool changed")
    if len(profiles) != 15:
        raise RuntimeError("frozen T5 plausible-effect pool size changed")
    if any(
        not math.isfinite(float(row["ratio"])) or float(row["ratio"]) <= 0.0
        for row in profiles
    ):
        raise RuntimeError("frozen T5 plausible-effect ratio is invalid")
    return profiles


def _cross_effect_profiles() -> list[dict[str, Any]]:
    payload = load_json(SMOOTH_CROSS_ANALYSIS)
    nominee = payload["analysis"]["nominee"]
    ratios = nominee["dataset_ratios"]
    if ctr.sha256_json(ratios) != EXPECTED_CROSS_EFFECT_POOL_SHA256:
        raise RuntimeError("frozen guarded-cross effect pool changed")
    profiles = [
        {
            "source": "smooth_cross_margin_analysis_nominee",
            "lineage": lineage,
            "ratio": ratio,
        }
        for lineage, ratio in sorted(ratios.items())
    ]
    if len(profiles) != 3:
        raise RuntimeError("frozen guarded-cross effect pool size changed")
    if any(
        not math.isfinite(float(row["ratio"])) or float(row["ratio"]) <= 0.0
        for row in profiles
    ):
        raise RuntimeError("frozen guarded-cross effect ratio is invalid")
    return profiles


def _provisional_cross_sensitivity_profiles() -> list[dict[str, Any]]:
    """Include non-nominee declines instead of sampling only selected wins."""
    non_nominees = [
        {
            "source": "optimistic_safe_non_nominee_decline_sensitivity",
            "lineage": row["lineage"],
            "ratio": 1.0,
        }
        for row in _composite_effect_profiles()
        if row["source"] == "guarded_a10_over_product_default"
    ]
    if len(non_nominees) != 12:
        raise RuntimeError(
            "provisional guarded-cross decline census changed"
        )
    return [*non_nominees, *_cross_effect_profiles()]


def _candidate_power(
    *,
    candidate_name: str,
    profiles: list[dict[str, Any]],
    effect_pool_sha256: str,
    effect_pool_sources: list[Path],
    seed: int,
    sampled_lineages: int,
    fixed_tie_lineages: int,
    interpretation: str,
) -> dict[str, Any]:
    if (
        sampled_lineages <= 0
        or fixed_tie_lineages < 0
        or sampled_lineages + fixed_tie_lineages != POWER_LINEAGES
    ):
        raise RuntimeError("panel-3 provisional power shape changed")
    log_pool = np.log(
        np.asarray([row["ratio"] for row in profiles], dtype=np.float64)
    )
    rng = np.random.default_rng(seed)
    component_counts = {
        "point": 0,
        "bonferroni_upper": 0,
        "leave_one_favorable_out": 0,
        "worst_dataset": 0,
    }
    passing = 0
    for start in range(0, POWER_SIMULATIONS, 10_000):
        count = min(10_000, POWER_SIMULATIONS - start)
        draws = rng.choice(
            log_pool,
            size=(count, sampled_lineages),
            replace=True,
        )
        if fixed_tie_lineages:
            draws = np.concatenate(
                (
                    draws,
                    np.zeros(
                        (count, fixed_tie_lineages),
                        dtype=np.float64,
                    ),
                ),
                axis=1,
            )
        means = draws.mean(axis=1)
        checks = {
            "point": np.exp(means) <= QUALITY_BAR,
            "bonferroni_upper": np.exp(
                means
                + BONFERRONI_Z
                * draws.std(axis=1, ddof=1)
                / math.sqrt(POWER_LINEAGES)
            )
            <= UNCERTAINTY_BAR,
            "leave_one_favorable_out": np.exp(
                (draws.sum(axis=1) - draws.min(axis=1))
                / (POWER_LINEAGES - 1)
            )
            <= LOO_BAR,
            "worst_dataset": np.exp(draws.max(axis=1)) <= HARM_BAR,
        }
        for name, values in checks.items():
            component_counts[name] += int(np.count_nonzero(values))
        passing += int(
            np.count_nonzero(np.logical_and.reduce(tuple(checks.values())))
        )
    marginal_probability = passing / POWER_SIMULATIONS
    return {
        "candidate": candidate_name,
        "seed": seed,
        "simulations": POWER_SIMULATIONS,
        "simulated_lineages": POWER_LINEAGES,
        "sampled_effect_lineages": sampled_lineages,
        "fixed_tie_lineages": fixed_tie_lineages,
        "effect_profile_count": len(profiles),
        "effect_pool_sha256": effect_pool_sha256,
        "effect_pool_sources": [
            str(path.relative_to(ROOT)) for path in effect_pool_sources
        ],
        "effect_pool_exchangeable_across_strata": True,
        "effect_pool_is_stratum_specific": False,
        "familywise_one_sided_alpha": FAMILYWISE_ONE_SIDED_ALPHA,
        "per_candidate_one_sided_alpha": PER_CANDIDATE_ONE_SIDED_ALPHA,
        "bonferroni_one_sided_percentile": 97.5,
        "bonferroni_normal_z": BONFERRONI_Z,
        "component_passing_simulations": component_counts,
        "marginal_passing_simulations": passing,
        "marginal_pass_probability": marginal_probability,
        "minimum_required_probability": MIN_POWER,
        "marginal_passes": marginal_probability >= MIN_POWER,
        "passes": marginal_probability >= MIN_POWER,
        "gates": {
            "equal_dataset_geomean_ratio_at_most": QUALITY_BAR,
            "bonferroni_one_sided_97_5_upper_at_most": UNCERTAINTY_BAR,
            "least_favorable_leave_one_out_ratio_at_most": LOO_BAR,
            "worst_dataset_ratio_at_most": HARM_BAR,
        },
        "interpretation": interpretation,
    }


def power_analysis() -> dict[str, Any]:
    """Return honest pre-calibration sensitivities that block authorization."""
    provisional_cross = _provisional_cross_sensitivity_profiles()
    candidates = {
        "t5_composite_policy": _candidate_power(
            candidate_name="t5_composite_policy",
            profiles=_composite_effect_profiles(),
            effect_pool_sha256=EXPECTED_EFFECT_POOL_SHA256,
            effect_pool_sources=[T5_REGISTRY],
            seed=POWER_SEED,
            sampled_lineages=5,
            fixed_tie_lineages=7,
            interpretation=(
                "provisional sensitivity using five empirical development "
                "draws plus seven exact size-gate ties known before outcomes; "
                "the component pool is not the exact composite policy"
            ),
        ),
        "guarded_cross_features_policy": _candidate_power(
            candidate_name="guarded_cross_features_policy",
            profiles=provisional_cross,
            effect_pool_sha256=ctr.sha256_json(provisional_cross),
            effect_pool_sources=[
                T5_REGISTRY,
                SMOOTH_CROSS_ANALYSIS,
            ],
            seed=POWER_SEED,
            sampled_lineages=12,
            fixed_tie_lineages=0,
            interpretation=(
                "provisional sensitivity using three selected guarded-cross "
                "nominees plus 12 optimistic safe non-nominee exact declines; "
                "not an observed exact-policy census"
            ),
        ),
    }
    probabilities = [
        row["marginal_pass_probability"] for row in candidates.values()
    ]
    # Fréchet's lower bound is valid for any dependence between the two
    # nominees. It avoids inventing an independence assumption at design time.
    joint_lower_bound = max(0.0, sum(probabilities) - (len(probabilities) - 1))
    diagnostics = {
        "seed": POWER_SEED,
        "simulations_per_candidate": POWER_SIMULATIONS,
        "simulated_lineages": POWER_LINEAGES,
        "candidate_count": CANDIDATE_COUNT,
        "familywise_one_sided_alpha": FAMILYWISE_ONE_SIDED_ALPHA,
        "per_candidate_one_sided_alpha": PER_CANDIDATE_ONE_SIDED_ALPHA,
        "bonferroni_one_sided_percentile": 97.5,
        "bonferroni_normal_z": BONFERRONI_Z,
        "candidates": candidates,
        "dependence_agnostic_joint_probability_lower_bound": joint_lower_bound,
        "minimum_required_probability": MIN_POWER,
        "all_marginals_pass": all(row["passes"] for row in candidates.values()),
        "joint_lower_bound_passes": joint_lower_bound >= MIN_POWER,
        "passes": False,
        "authorization_blocked": True,
        "authorization_status": (
            "blocked_pending_exact_policy_spent_data_calibration"
        ),
        "calibration_protocol": (
            "benchmarks/panel3_cross_power_calibration_protocol.md"
        ),
        "calibration_result_required": (
            "benchmarks/panel3_cross_power_calibration_summary.json"
        ),
        "interpretation": (
            "pre-outcome sensitivity audit only; the prior power claim was "
            "structurally optimistic and cannot authorize Panel 3"
        ),
    }
    return diagnostics
