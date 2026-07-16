"""Run the preregistered minimal CTR23 regression confirmation.

The campaign spends all nine confirmation tasks at exactly ``r0f0``, ``r0f1``,
and ``r0f2``.  A10, ChimeraBoost 0.14.1, and the current DarkoFit default run
on all 27 coordinates; CatBoost 1.2.10 is descriptive context on ``r0f0`` only.
No result from a partial or concurrent-invalid attempt is reusable.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import pickle
import platform
import shutil
import stat
import subprocess
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    from benchmarks import run_tabarena_regression_accuracy_shootout as hardened
    from benchmarks import run_tabarena_regression_same_machine as comparators
except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
    import run_tabarena_regression_accuracy_shootout as hardened
    import run_tabarena_regression_same_machine as comparators


CAMPAIGN_KIND = "darkofit_ctr23_minimal_confirmation"
COMPLETION_KIND = CAMPAIGN_KIND + "_completion"
PAYLOAD_KIND = CAMPAIGN_KIND + "_analysis_payload"
HARNESS_SCHEMA_VERSION = 2
MANIFEST_FILENAME = "run_manifest.json"
COMPLETION_ATTESTATION_FILENAME = "completion_attestation.json"
ANALYSIS_PAYLOAD_FILENAME = "analysis_payload.json"
SCHEDULE_FILENAME = "wave_schedule.json"
WARMUP_HISTORY_FILENAME = "warmup_history.json"
CONCURRENCY_HISTORY_FILENAME = "concurrency_history.json"
PREFLIGHT_REPORT_FILENAME = "preflight_report.json"
INVALID_ATTEMPT_FILENAME = "invalid_attempt.json"
COORDINATE_MANIFEST_FILENAME = "ctr23_minimal_confirmation_coordinates.json"
_FAILURE_SWAP_TELEMETRY_ATTRIBUTE = "_ctr23_failure_swap_telemetry"
_WORKER_STARTUP_CLEANUP_CONFIRMED_ATTRIBUTE = (
    "_ctr23_worker_startup_cleanup_confirmed"
)
_FAILURE_SWAP_TELEMETRY_FIELDS = frozenset(
    {
        "capture_status",
        "teardown_confirmed",
        "post_teardown_sample_recorded",
        "worker_session_swap_telemetry",
        "swap_in_bytes",
        "swap_out_bytes",
        "diagnostic",
    }
)
ANALYZER_CAMPAIGN_JSON_FILENAMES = (
    MANIFEST_FILENAME,
    COMPLETION_ATTESTATION_FILENAME,
    ANALYSIS_PAYLOAD_FILENAME,
    SCHEDULE_FILENAME,
    PREFLIGHT_REPORT_FILENAME,
    CONCURRENCY_HISTORY_FILENAME,
    WARMUP_HISTORY_FILENAME,
)

TIME_LIMIT_SECONDS = 3_600.0
EXPECTED_CHILD_CPUS = 18
WORKER_COUNT = 2
EXPECTED_PRIMARY_COORDINATES = 27
EXPECTED_CATBOOST_COORDINATES = 9
EXPECTED_JOBS = 90
EXPECTED_CHILD_FITS = 720
EXPECTED_WAVES = 45
BOOTSTRAP_DRAWS = 10_000
PRIMARY_BOOTSTRAP_SEED = 20_260_719
GUARDRAIL_BOOTSTRAP_SEED = 20_260_720
CATBOOST_BOOTSTRAP_SEED = 20_260_721
BOOTSTRAP_BIT_GENERATOR = "PCG64"
BOOTSTRAP_QUANTILE_METHOD = "higher"
SWAP_POLICY = "quality_only_swap_in"
DEFAULT_OUTPUT_DIR = Path(".cache/ctr23-minimal-confirmation-20260715")
DEFAULT_CHIMERABOOST_PATH = Path("/Users/kmedved/.cache/chimeraboost-v0.14.1")
CHIMERABOOST_TAG_COMMIT = "9c9ea6e704a9fe2bfe6d6c284b22de73914be048"
TABARENA_COMMIT = "4cd1d2526874962daae048a6f2dcf34aa272f3fa"
DARKOFIT_SUBTREE = "52278b0326419a45a72bdfd3afcfc13019087838"

A10_CONFIG: dict[str, Any] = {
    "iterations": 10_000,
    "tree_mode": "auto",
    "l2_leaf_reg": 3.0,
    "max_bins": 128,
    "learning_rate": 0.1,
    "ts_permutations": 1,
    "linear_residual": False,
    "early_stopping": True,
    "use_best_model": True,
}

ARM_SPECS: dict[str, dict[str, Any]] = {
    "A10": {
        "code": "A",
        "engine": "darkofit",
        "version": "current-tree",
        "model_cls": "ScreenNativeDarkoFitModel",
        "config": dict(A10_CONFIG),
        "coordinates": "all",
    },
    "M": {
        "code": "M",
        "engine": "chimeraboost",
        "version": "0.14.1",
        "model_cls": "CTR23ComparatorChimeraBoostModel",
        "config": {},
        "coordinates": "all",
    },
    "D": {
        "code": "D",
        "engine": "darkofit",
        "version": "current-tree",
        "model_cls": "ComparatorDarkoFitModel",
        "config": {},
        "coordinates": "all",
    },
    "C": {
        "code": "C",
        "engine": "catboost",
        "version": "1.2.10",
        "model_cls": "CTR23ComparatorCatBoostModel",
        "config": {},
        "coordinates": "r0f0-only",
    },
}

OUTER_PAYLOAD_FIELDS = (
    "dataset",
    "task_id",
    "repeat",
    "fold",
    "sample",
    "arm",
    "test_rmse",
    "val_rmse",
    "source",
    "num_cpus",
    "num_cpus_child",
    "num_gpus",
    "num_gpus_child",
)
CHILD_PAYLOAD_FIELDS = (
    "dataset",
    "task_id",
    "repeat",
    "fold",
    "sample",
    "arm",
    "child_fold",
    "source",
    "num_cpus",
    "num_gpus",
    "iterations_requested",
    "iterations_attempted",
    "rounds_completed",
    "rounds_retained",
    "best_iteration",
    "resolved_learning_rate",
    "requested_tree_mode",
    "selected_tree_mode",
    "selected_lane",
    "stop_reason",
    "deadline_hit",
    "deadline_is_soft",
    "time_callback_hit",
    "time_callback_instance_count",
    "time_callback_call_count",
    "candidate_metadata",
)

_COORDINATE_MANIFEST_PATH = Path(__file__).with_name(
    COORDINATE_MANIFEST_FILENAME
)
_SOURCE_FREEZE_ARTIFACTS = {
    "benchmarks/ctr23_suite_snapshot.json": {
        "file_sha256": (
            "ce676c7dde7576aee8c5c8f074aa76fa004746d0ed4381a89b12335282e1d33c"
        ),
        "declared_suite_snapshot_sha256": (
            "95bb2bb5d9c65ea21cb7642151bedb831ed67712bae28166a0bddc64670f0364"
        ),
    },
    "benchmarks/ctr23_contamination_registry.json": {
        "file_sha256": (
            "002dba27713237c33f8b09de91160ad54b4f7d4c6e44b76ba5ba0e0e98c0adc2"
        ),
        "declared_registry_sha256": (
            "9bda6f8b94b71575fa8275ed724ab80976c93555d898fbec8f474fcc78c6639d"
        ),
    },
    "benchmarks/ctr23_partition.json": {
        "file_sha256": (
            "125e70cbe49241fb4fda1ed3f79b504f5d8d0b20b8f85d665c1cb49aa5c5fab6"
        ),
        "declared_partition_sha256": (
            "24e060ed3626fed23967294138d5768c3d9e7241f4ed06cf9b8180d512e81ee8"
        ),
    },
    "benchmarks/ctr23_manual_evidence_catalog.json": {
        "file_sha256": (
            "abb3dc9c875db5119b2bfbbf23d49cbb1cc2176c39d0a8237d5e01ba533941f3"
        ),
        "declared_manual_evidence_sha256": (
            "66529d85f9f1caea2d04784ae6666704cb6c3b5e56e06460066a482c5358ce75"
        ),
    },
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _read_json_regular(path: Path, field: str) -> Any:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise RuntimeError(f"{field} must be a regular file: {path}")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"could not read {field}: {path}") from exc
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(payload) != after.st_size:
        raise RuntimeError(f"{field} changed while it was read: {path}")
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{field} is not valid JSON: {path}") from exc


def _exists_including_broken_symlink(path: Path, field: str) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeError(f"could not inspect {field}: {path}") from exc
    return True


def _coordinate_document() -> dict[str, Any]:
    value = _read_json_regular(_COORDINATE_MANIFEST_PATH, "coordinate manifest")
    if not isinstance(value, dict):
        raise RuntimeError("coordinate manifest must be a mapping")
    return value


def expected_coordinate_manifest() -> list[dict[str, Any]]:
    """Return the exact 27 official split rows in their preregistered order."""
    document = _coordinate_document()
    rows: list[dict[str, Any]] = []
    tasks = document.get("tasks")
    if not isinstance(tasks, list):
        raise RuntimeError("coordinate manifest tasks must be a list")
    for task in tasks:
        if not isinstance(task, Mapping) or not isinstance(
            task.get("coordinates"), list
        ):
            raise RuntimeError("coordinate manifest task is malformed")
        for coordinate in task["coordinates"]:
            rows.append(
                {
                    "dataset": task.get("dataset_name"),
                    "fold": coordinate.get("fold"),
                    "openml_task_id": task.get("task_id"),
                    "repeat": coordinate.get("repeat"),
                    "sample": coordinate.get("sample"),
                    "test_index_sha256": coordinate.get("test_index_sha256"),
                    "test_size": coordinate.get("test_size"),
                    "train_index_sha256": coordinate.get("train_index_sha256"),
                    "train_size": coordinate.get("train_size"),
                }
            )
    if len(rows) != EXPECTED_PRIMARY_COORDINATES:
        raise RuntimeError("coordinate manifest does not contain exactly 27 rows")
    return rows


COORDINATE_MANIFEST_SHA256 = (
    "6cef3b771c20440c9dad6b737797f50650d84217ee99cf8fc6fcfcbd85829c0b"
)


def _validate_coordinate_digest() -> None:
    observed = hashlib.sha256(
        _canonical_json(expected_coordinate_manifest())
    ).hexdigest()
    if observed != COORDINATE_MANIFEST_SHA256:
        raise RuntimeError("frozen coordinate manifest digest changed")


def _task_rows() -> list[dict[str, Any]]:
    tasks = _coordinate_document().get("tasks")
    if not isinstance(tasks, list):
        raise RuntimeError("coordinate manifest tasks must be a list")
    return [dict(task) for task in tasks]


def expected_grid() -> set[tuple[str, int, int, int, int, str]]:
    grid: set[tuple[str, int, int, int, int, str]] = set()
    for row in expected_coordinate_manifest():
        coordinate = (
            str(row["dataset"]),
            int(row["openml_task_id"]),
            int(row["repeat"]),
            int(row["fold"]),
            int(row["sample"]),
        )
        for arm in ("A10", "M", "D"):
            grid.add((*coordinate, arm))
        if row["repeat"] == 0 and row["fold"] == 0 and row["sample"] == 0:
            grid.add((*coordinate, "C"))
    if len(grid) != EXPECTED_JOBS:
        raise RuntimeError("frozen job grid does not contain exactly 90 jobs")
    return grid


def expected_child_grid() -> set[tuple[str, int, int, int, int, str, int]]:
    grid = {
        (*job, child_fold)
        for job in expected_grid()
        for child_fold in range(8)
    }
    if len(grid) != EXPECTED_CHILD_FITS:
        raise RuntimeError("frozen child grid does not contain exactly 720 fits")
    return grid


_TASK_WAVE_PAIRS = (
    (("A10", 0), ("M", 0)),
    (("D", 0), ("C", 0)),
    (("A10", 1), ("M", 1)),
    (("D", 1), ("A10", 2)),
    (("M", 2), ("D", 2)),
)
# The masks select which member of each pair occupies slot zero.  Their global
# exposure is A=14/13, M=14/13, D=13/14, C=4/5 across slots 0/1.
_TASK_SLOT_MASKS = (0, 7, 0, 3, 0, 15, 0, 2, 0)


def _schedule_key(dataset: str, arm: str, fold: int) -> dict[str, Any]:
    task_id = next(
        int(row["task_id"])
        for row in _task_rows()
        if row["dataset_name"] == dataset
    )
    return {
        "dataset": dataset,
        "task_id": task_id,
        "repeat": 0,
        "fold": fold,
        "sample": 0,
        "arm": arm,
        "arm_code": ARM_SPECS[arm]["code"],
    }


def expected_schedule() -> list[dict[str, Any]]:
    waves: list[dict[str, Any]] = []
    for task_index, task in enumerate(_task_rows()):
        dataset = str(task["dataset_name"])
        mask = _TASK_SLOT_MASKS[task_index]
        for local_index, pair in enumerate(_TASK_WAVE_PAIRS):
            reverse = bool(mask & (1 << local_index))
            jobs = []
            for pair_index, (arm, fold) in enumerate(pair):
                slot = (1 - pair_index) if reverse else pair_index
                jobs.append(
                    {
                        "worker_slot": slot,
                        "key": _schedule_key(dataset, arm, fold),
                    }
                )
            waves.append(
                {
                    "wave_index": len(waves),
                    "task_index": task_index,
                    "local_wave_index": local_index,
                    "dataset": dataset,
                    "jobs": sorted(jobs, key=lambda item: item["worker_slot"]),
                }
            )
    validate_schedule(waves)
    return waves


def validate_schedule(waves: Sequence[Mapping[str, Any]]) -> None:
    if len(waves) != EXPECTED_WAVES:
        raise RuntimeError("frozen schedule must contain exactly 45 waves")
    observed: set[tuple[str, int, int, int, int, str]] = set()
    slot_counts = {arm: [0, 0] for arm in ARM_SPECS}
    for index, wave in enumerate(waves):
        jobs = wave.get("jobs")
        if wave.get("wave_index") != index or not isinstance(jobs, list) or len(jobs) != 2:
            raise RuntimeError("schedule wave header is malformed")
        slots = set()
        for item in jobs:
            if not isinstance(item, Mapping) or set(item) != {"worker_slot", "key"}:
                raise RuntimeError("schedule job fields are not exact")
            slot = item["worker_slot"]
            key = item["key"]
            if slot not in (0, 1) or slot in slots or not isinstance(key, Mapping):
                raise RuntimeError("schedule worker slots are invalid")
            public = _key_tuple(key)
            if public in observed:
                raise RuntimeError("schedule contains an invalid or duplicate job")
            observed.add(public)
            slots.add(slot)
            slot_counts[str(public[5])][int(slot)] += 1
        if slots != {0, 1}:
            raise RuntimeError("schedule wave does not occupy both worker slots")
    if observed != expected_grid():
        raise RuntimeError("schedule does not cover the exact 90-job grid")
    if any(abs(counts[0] - counts[1]) > 1 for counts in slot_counts.values()):
        raise RuntimeError("schedule arm exposure is not slot-balanced")


def schedule_sha256() -> str:
    return hashlib.sha256(_canonical_json(expected_schedule())).hexdigest()


def analysis_boundary() -> dict[str, Any]:
    """Return the exact campaign/result-namespace analyzer trust boundary.

    Raw result verification and safe extraction belong exclusively to the
    runner. Source-frozen repository/dependency provenance is validated under
    its separate manifest contract; campaign and recovery namespaces are
    limited to the JSON documents and absence check below.
    """
    return {
        "schema_version": 1,
        "source_registry_json_allowlist": list(_SOURCE_FREEZE_ARTIFACTS),
        "source_registry_path_policy": (
            "hard_coded_repo_relative_paths_before_filesystem_access"
        ),
        "campaign_root_json_allowlist": list(ANALYZER_CAMPAIGN_JSON_FILENAMES),
        "sequential_recovery_json_allowlist": [
            INVALID_ATTEMPT_FILENAME,
            MANIFEST_FILENAME,
        ],
        "sequential_recovery_absence_checks": [
            COMPLETION_ATTESTATION_FILENAME,
        ],
        "raw_result_access": "forbidden_no_stat_enumerate_open_hash_or_decode",
        "raw_result_manifest_authority": "runner",
        "completion_attestation_role": (
            "runner_root_attestation_without_external_signature"
        ),
        "operational_input_policy": (
            "strict_json_with_runner_attested_hash_and_size_chain"
        ),
    }


def frozen_protocol() -> dict[str, Any]:
    return {
        "schema_version": HARNESS_SCHEMA_VERSION,
        "kind": CAMPAIGN_KIND,
        "coordinate_manifest_sha256": COORDINATE_MANIFEST_SHA256,
        "coordinates": expected_coordinate_manifest(),
        "arms": json.loads(json.dumps(ARM_SPECS, sort_keys=True)),
        "expected_primary_coordinates": EXPECTED_PRIMARY_COORDINATES,
        "expected_catboost_coordinates": EXPECTED_CATBOOST_COORDINATES,
        "expected_jobs": EXPECTED_JOBS,
        "expected_child_fits": EXPECTED_CHILD_FITS,
        "bag_folds": 8,
        "bag_sets": 1,
        "seed_policy": "fold-wise",
        "fold_fitting_strategy": "sequential_local",
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "configured_child_cpus": EXPECTED_CHILD_CPUS,
        "execution": {
            "start_method": "spawn",
            "persistent_worker_count": WORKER_COUNT,
            "private_worker_scratch": True,
            "serial_process_local_warmup": True,
            "hard_wave_barriers": True,
            "swap_policy": SWAP_POLICY,
            "swap_in_allowed_and_recorded": True,
            "swap_out_allowed": False,
            "maximum_combined_rss_fraction": 0.8,
            "timing_or_memory_performance_claims_allowed": False,
            "recovery": "fresh_namespace_all_sequential_from_wave_zero_only",
            "partial_results_reusable": False,
        },
        "schedule_sha256": schedule_sha256(),
        "schedule": expected_schedule(),
        "inference": {
            "primary": "equal-task geometric mean of within-task A10/M ratios",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bit_generator": BOOTSTRAP_BIT_GENERATOR,
            "quantile_method": BOOTSTRAP_QUANTILE_METHOD,
            "primary_seed": PRIMARY_BOOTSTRAP_SEED,
            "primary_gate": "one-sided 95% upper bound below 1.000",
            "guardrail_seed": GUARDRAIL_BOOTSTRAP_SEED,
            "guardrail_gate": "95th percentile of maximum task A10/D ratio <= 1.02",
            "catboost_seed": CATBOOST_BOOTSTRAP_SEED,
            "catboost_status": "descriptive_r0f0_only",
        },
        "safe_payload": {
            "outer_fields": list(OUTER_PAYLOAD_FIELDS),
            "child_fields": list(CHILD_PAYLOAD_FIELDS),
            "timing_fields_allowed": False,
            "memory_performance_fields_allowed": False,
            "analysis_boundary": analysis_boundary(),
            "swap_telemetry_retained": True,
        },
    }


def protocol_sha256() -> str:
    return hashlib.sha256(
        Path(__file__)
        .with_name("ctr23_minimal_confirmation_protocol.md")
        .read_bytes()
    ).hexdigest()


def frozen_protocol_sha256() -> str:
    return hashlib.sha256(_canonical_json(frozen_protocol())).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source_freeze() -> dict[str, Any]:
    """Validate the committed target-blind registry without touching results."""
    _validate_coordinate_digest()
    document = _coordinate_document()
    if (
        set(document)
        != {
            "schema_version",
            "kind",
            "ctr23_suite_id",
            "coordinate_policy",
            "source_artifacts",
            "expected_task_count",
            "expected_coordinate_count",
            "tasks",
        }
        or document["schema_version"] != 1
        or document["kind"]
        != "darkofit_ctr23_minimal_confirmation_coordinates"
        or document["ctr23_suite_id"] != 353
        or document["expected_task_count"] != 9
        or document["expected_coordinate_count"] != 27
        or document["coordinate_policy"]
        != {
            "repeat": 0,
            "folds": [0, 1, 2],
            "sample": 0,
            "split_indices": ["r0f0", "r0f1", "r0f2"],
        }
    ):
        raise RuntimeError("coordinate document header changed")

    source_artifacts = document.get("source_artifacts")
    if (
        not isinstance(source_artifacts, Mapping)
        or source_artifacts != _SOURCE_FREEZE_ARTIFACTS
    ):
        # This check deliberately precedes every path construction and file
        # operation.  Source-registry keys are data, not filesystem authority.
        raise RuntimeError("coordinate source artifact registry changed")

    source_payloads: dict[str, Any] = {}
    for relative, metadata in _SOURCE_FREEZE_ARTIFACTS.items():
        path = REPOSITORY_ROOT / relative
        expected_file_hash = metadata["file_sha256"]
        if _sha256_file(path) != expected_file_hash:
            raise RuntimeError(f"source artifact byte hash changed: {relative}")
        source_payloads[relative] = _read_json_regular(path, relative)

    suite = source_payloads["benchmarks/ctr23_suite_snapshot.json"]
    registry = source_payloads["benchmarks/ctr23_contamination_registry.json"]
    partition = source_payloads["benchmarks/ctr23_partition.json"]
    source_payloads["benchmarks/ctr23_manual_evidence_catalog.json"]
    semantic = {
        "suite_snapshot_sha256": (
            "95bb2bb5d9c65ea21cb7642151bedb831ed67712bae28166a0bddc64670f0364"
        ),
        "contamination_registry_sha256": (
            "9bda6f8b94b71575fa8275ed724ab80976c93555d898fbec8f474fcc78c6639d"
        ),
        "partition_sha256": (
            "24e060ed3626fed23967294138d5768c3d9e7241f4ed06cf9b8180d512e81ee8"
        ),
        "registry_bundle_sha256": (
            "21980c6ddaf3f5b70e866fbcc6c59c04a98b666687234147a7cedcc0b8271516"
        ),
        "manual_evidence_sha256": (
            "66529d85f9f1caea2d04784ae6666704cb6c3b5e56e06460066a482c5358ce75"
        ),
        "declarations_sha256": (
            "bd20852afdacdbd55d20fd4adfe7331c760f651061a912fa8424f5d77675dcc9"
        ),
    }
    if (
        suite.get("schema_version") != 3
        or registry.get("schema_version") != 3
        or partition.get("schema_version") != 3
        or suite.get("suite_snapshot_sha256")
        != semantic["suite_snapshot_sha256"]
        or registry.get("suite_snapshot_sha256")
        != semantic["suite_snapshot_sha256"]
        or partition.get("suite_snapshot_sha256")
        != semantic["suite_snapshot_sha256"]
        or registry.get("contamination_registry_sha256")
        != semantic["contamination_registry_sha256"]
        or partition.get("contamination_registry_sha256")
        != semantic["contamination_registry_sha256"]
        or partition.get("partition_sha256") != semantic["partition_sha256"]
        or partition.get("registry_bundle_sha256")
        != semantic["registry_bundle_sha256"]
        or suite.get("manual_evidence_sha256")
        != semantic["manual_evidence_sha256"]
        or registry.get("manual_evidence_sha256")
        != semantic["manual_evidence_sha256"]
        or partition.get("manual_evidence_sha256")
        != semantic["manual_evidence_sha256"]
        or suite.get("declarations_sha256")
        != semantic["declarations_sha256"]
        or registry.get("declarations_sha256")
        != semantic["declarations_sha256"]
    ):
        raise RuntimeError("CTR23 semantic registry bundle changed")

    task_ids = tuple(int(task["task_id"]) for task in _task_rows())
    if (
        task_ids
        != (
            361236,
            361251,
            361252,
            361258,
            361268,
            361269,
            361619,
            361622,
            361623,
        )
        or tuple(partition.get("confirmation_task_ids", ())) != task_ids
        or set(task_ids).intersection(partition.get("lockbox_task_ids", ()))
    ):
        raise RuntimeError("CTR23 confirmation/lockbox allocation changed")
    registry_tasks = {
        int(item["openml_task_id"]): item for item in registry.get("tasks", [])
    }
    for task_id in task_ids:
        item = registry_tasks.get(task_id)
        if (
            not isinstance(item, Mapping)
            or item.get("status") != "eligible"
            or item.get("exclusion_reasons") != []
            or item.get("ambiguous_matches") != []
            or item.get("exposure_scope") is not None
        ):
            raise RuntimeError(f"CTR23 task {task_id} is no longer cleanly eligible")

    suite_tasks = {
        int(item["openml_task_id"]): item
        for item in suite.get("ctr23_tasks", [])
    }
    regenerated: list[dict[str, Any]] = []
    for task in _task_rows():
        task_id = int(task["task_id"])
        if (
            type(task.get("n_rows")) is not int
            or type(task.get("n_features")) is not int
            or task["n_rows"] <= 0
            or task["n_features"] <= 0
            or task["n_rows"] * task["n_features"] >= 5_000_000
        ):
            raise RuntimeError(
                f"CTR23 task {task_id} is not CatBoost memory-callback ineligible"
            )
        source = suite_tasks.get(task_id)
        if not isinstance(source, Mapping):
            raise RuntimeError(f"CTR23 suite task {task_id} is missing")
        fingerprint = source.get("fingerprint", {})
        official = source.get("official_splits", {})
        if (
            source.get("normalized_name") != task["dataset_name"]
            or source.get("openml_dataset_id") != task["dataset_id"]
            or source.get("target_name") != task["target_name"]
            or fingerprint.get("n_rows") != task["n_rows"]
            or fingerprint.get("n_features") != task["n_features"]
            or fingerprint.get("has_categorical") != task["has_categorical"]
            or fingerprint.get("has_missing_features")
            != task["has_missing_features"]
            or fingerprint.get("categorical_feature_count")
            != task["categorical_feature_count"]
            or official.get("dimensions") != task["split_dimensions"]
            or official.get("raw_split_file_sha256")
            != task["raw_split_file_sha256"]
            or official.get("semantic_split_sha256")
            != task["semantic_split_sha256"]
        ):
            raise RuntimeError(f"CTR23 suite metadata changed for task {task_id}")
        coordinate_index = {
            (item["repeat"], item["fold"], item["sample"]): item
            for item in official.get("coordinates", [])
        }
        for selected in task["coordinates"]:
            key = (
                selected["repeat"],
                selected["fold"],
                selected["sample"],
            )
            if coordinate_index.get(key) != selected:
                raise RuntimeError(
                    f"official split record changed for task {task_id} {key}"
                )
            regenerated.append(
                {
                    "dataset": task["dataset_name"],
                    "fold": selected["fold"],
                    "openml_task_id": task_id,
                    "repeat": selected["repeat"],
                    "sample": selected["sample"],
                    "test_index_sha256": selected["test_index_sha256"],
                    "test_size": selected["test_size"],
                    "train_index_sha256": selected["train_index_sha256"],
                    "train_size": selected["train_size"],
                }
            )
    if regenerated != expected_coordinate_manifest():
        raise RuntimeError("coordinate file and official suite snapshot diverged")
    return {
        "ctr23_suite_id": 353,
        "coordinate_manifest_sha256": COORDINATE_MANIFEST_SHA256,
        "semantic_hashes": semantic,
        "source_file_sha256": {
            relative: metadata["file_sha256"]
            for relative, metadata in document["source_artifacts"].items()
        },
        "confirmation_task_ids": list(task_ids),
        "lockbox_task_ids": list(partition["lockbox_task_ids"]),
    }


def verify_live_official_splits() -> dict[str, Any]:
    """Bind the locally materialized OpenML task objects to committed indices."""
    import numpy as np
    import openml

    verified = []
    for task in _task_rows():
        task_id = int(task["task_id"])
        openml_task = openml.tasks.get_task(
            task_id,
            download_data=True,
            download_qualities=True,
            download_splits=True,
        )
        dimensions = tuple(int(value) for value in openml_task.get_split_dimensions())
        expected_dimensions = task["split_dimensions"]
        if dimensions != (
            expected_dimensions["repeats"],
            expected_dimensions["folds"],
            expected_dimensions["samples"],
        ):
            raise RuntimeError(f"live OpenML split dimensions changed for task {task_id}")
        for coordinate in task["coordinates"]:
            train, test = openml_task.get_train_test_split_indices(
                repeat=coordinate["repeat"],
                fold=coordinate["fold"],
                sample=coordinate["sample"],
            )
            train = np.asarray(train, dtype="<i8")
            test = np.asarray(test, dtype="<i8")
            observed = {
                "train_size": int(train.size),
                "test_size": int(test.size),
                "train_index_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
                "test_index_sha256": hashlib.sha256(test.tobytes()).hexdigest(),
            }
            if any(observed[name] != coordinate[name] for name in observed):
                raise RuntimeError(
                    f"live OpenML indices changed for task {task_id} "
                    f"r{coordinate['repeat']}f{coordinate['fold']}s{coordinate['sample']}"
                )
            verified.append(
                {
                    "task_id": task_id,
                    "repeat": coordinate["repeat"],
                    "fold": coordinate["fold"],
                    "sample": coordinate["sample"],
                    **observed,
                }
            )
    if len(verified) != EXPECTED_PRIMARY_COORDINATES:
        raise RuntimeError("live OpenML split verification is incomplete")
    return {
        "coordinate_manifest_sha256": COORDINATE_MANIFEST_SHA256,
        "verified_coordinate_count": len(verified),
        "coordinates_sha256": hashlib.sha256(_canonical_json(verified)).hexdigest(),
    }


def build_task_metadata_collection():
    """Build exact OpenML metadata; never call TabArena's lossy fallback."""
    from tabarena.benchmark.task.metadata import (
        OpenMLTaskMetadataSource,
        SplitMetadata,
        TabArenaTaskMetadata,
        TaskMetadataCollection,
    )

    tasks = []
    for task in _task_rows():
        split_metadata = {
            f"r{coordinate['repeat']}f{coordinate['fold']}": SplitMetadata(
                repeat=int(coordinate["repeat"]),
                fold=int(coordinate["fold"]),
                num_instances_train=int(coordinate["train_size"]),
                num_instances_test=int(coordinate["test_size"]),
                num_instance_groups_train=int(coordinate["train_size"]),
                num_instance_groups_test=int(coordinate["test_size"]),
                num_classes_train=-1,
                num_classes_test=-1,
                num_features_train=int(task["n_features"]),
                num_features_test=int(task["n_features"]),
            )
            for coordinate in task["coordinates"]
        }
        tasks.append(
            TabArenaTaskMetadata(
                dataset_name=str(task["dataset_name"]),
                problem_type="regression",
                is_classification=False,
                target_name=str(task["target_name"]),
                eval_metric="rmse",
                splits_metadata=split_metadata,
                split_time_horizon=None,
                split_time_horizon_unit=None,
                stratify_on=None,
                time_on=None,
                group_on=None,
                group_time_on=None,
                group_labels=None,
                multiclass_min_n_classes_over_splits=None,
                multiclass_max_n_classes_over_splits=None,
                class_consistency_over_splits=None,
                num_instances=int(task["n_rows"]),
                num_features=int(task["n_features"]),
                num_classes=-1,
                num_instance_groups=int(task["n_rows"]),
                tabarena_task_name=str(task["dataset_name"]),
                task_id_str=str(task["task_id"]),
                has_datetime=False,
                has_text=False,
                has_categorical=bool(task["has_categorical"]),
                has_numerical=True,
                has_binary=None,
                has_high_cardinality_categorical=None,
                task_type="random",
                num_text_cols=0,
                num_high_cardinality_cats=None,
                num_cols_after_preprocessing=None,
                missing_value_fraction=None,
                domain=None,
                dataset_year=None,
                source="OpenML CTR23 suite 353",
            )
        )
    class _ExactOpenMLSource(OpenMLTaskMetadataSource):
        def load(self, *, verbose: bool = False):
            del verbose
            return list(tasks)

    collection = TaskMetadataCollection(tasks, source=_ExactOpenMLSource())
    observed = {
        (dataset, repeat, fold)
        for dataset, fold, repeat in collection.dataset_fold_repeats()
    }
    expected = {
        (row["dataset"], row["repeat"], row["fold"])
        for row in expected_coordinate_manifest()
    }
    if observed != expected or len(collection) != 9:
        raise RuntimeError("exact TaskMetadataCollection changed the frozen grid")
    return collection


def _experiment_suffix(arm: str) -> str:
    if arm not in ARM_SPECS:
        raise RuntimeError(f"unknown CTR23 arm: {arm!r}")
    return f"_c1_ctr23_minimal_{ARM_SPECS[arm]['code']}"


def _experiment_name(arm: str) -> str:
    display = {
        "A10": "DarkoFit",
        "D": "DarkoFit",
        "M": "ChimeraBoost",
        "C": "CatBoost",
    }[arm]
    return f"{display}{_experiment_suffix(arm)}_BAG_L1"


def _expected_ag_ensemble_config() -> dict[str, Any]:
    return {
        "model_random_seed": 0,
        "vary_seed_across_folds": True,
        "fold_fitting_strategy": "sequential_local",
        "ag.max_time_limit": TIME_LIMIT_SECONDS,
    }


def _load_model_classes(chimeraboost_path: Path = DEFAULT_CHIMERABOOST_PATH):
    comparators.activate_chimeraboost_checkout(chimeraboost_path)
    classes = comparators._load_model_classes()
    comparators.validate_official_defaults(classes)
    try:
        from benchmarks.tabarena_screen_adapters import ScreenNativeDarkoFitModel
        from benchmarks.tabarena_ctr23_adapters import (
            CTR23ComparatorCatBoostModel,
            CTR23ComparatorChimeraBoostModel,
        )
    except ModuleNotFoundError:
        from tabarena_screen_adapters import ScreenNativeDarkoFitModel
        from tabarena_ctr23_adapters import (
            CTR23ComparatorCatBoostModel,
            CTR23ComparatorChimeraBoostModel,
        )
    classes[ScreenNativeDarkoFitModel.__name__] = ScreenNativeDarkoFitModel
    classes[CTR23ComparatorChimeraBoostModel.__name__] = (
        CTR23ComparatorChimeraBoostModel
    )
    classes[CTR23ComparatorCatBoostModel.__name__] = CTR23ComparatorCatBoostModel
    return classes


def build_experiments(
    *, model_classes: Mapping[str, type], config_generator_cls, time_limit: float
) -> dict[str, Any]:
    if not math.isfinite(float(time_limit)) or float(time_limit) != TIME_LIMIT_SECONDS:
        raise ValueError(
            f"frozen campaign time limit must equal {TIME_LIMIT_SECONDS:g} seconds"
        )
    experiments: dict[str, Any] = {}
    for arm, spec in ARM_SPECS.items():
        generator = config_generator_cls(
            model_cls=model_classes[spec["model_cls"]],
            manual_configs=[dict(spec["config"])],
            search_space={},
        )
        generated = generator.generate_all_bag_experiments(
            num_random_configs=0,
            name_id_suffix=f"_ctr23_minimal_{spec['code']}",
            add_seed="fold-wise",
            fold_fitting_strategy="sequential_local",
            time_limit=time_limit,
        )
        if len(generated) != 1 or generated[0].name != _experiment_name(arm):
            raise RuntimeError(f"unexpected generated experiment for arm {arm}")
        experiments[arm] = generated[0]
    return experiments


def _job_arm(job: Any) -> str:
    name = getattr(getattr(job, "experiment", None), "name", "")
    matches = [arm for arm in ARM_SPECS if name == _experiment_name(arm)]
    if len(matches) != 1:
        raise RuntimeError(f"could not identify CTR23 job arm from {name!r}")
    arm = matches[0]
    method = getattr(job.experiment, "method_kwargs", None)
    if not isinstance(method, Mapping):
        raise RuntimeError("CTR23 job has no resolved method settings")
    raw = dict(method.get("model_hyperparameters", {}))
    ag_args = raw.pop("ag_args", None)
    ag_ensemble = raw.pop("ag_args_ensemble", None)
    if (
        raw != ARM_SPECS[arm]["config"]
        or ag_args != {"name_suffix": _experiment_suffix(arm)}
        or not isinstance(ag_ensemble, Mapping)
        or dict(ag_ensemble) != _expected_ag_ensemble_config()
        or getattr(method.get("model_cls"), "__name__", None)
        != ARM_SPECS[arm]["model_cls"]
    ):
        raise RuntimeError(f"CTR23 job does not match frozen arm {arm}")
    return arm


def _job_key(job: Any) -> tuple[str, int, int, int, int, str]:
    dataset = str(job.task.dataset)
    task_id = next(
        int(task["task_id"])
        for task in _task_rows()
        if task["dataset_name"] == dataset
    )
    return (
        dataset,
        task_id,
        int(job.task.repeat),
        int(job.task.fold),
        0,
        _job_arm(job),
    )


def _key_payload(key: tuple[str, int, int, int, int, str]) -> dict[str, Any]:
    if key not in expected_grid():
        raise RuntimeError(f"job key is outside the frozen grid: {key}")
    dataset, task_id, repeat, fold, sample, arm = key
    return {
        "dataset": dataset,
        "task_id": task_id,
        "repeat": repeat,
        "fold": fold,
        "sample": sample,
        "arm": arm,
        "arm_code": ARM_SPECS[arm]["code"],
    }


def _key_tuple(value: Mapping[str, Any]) -> tuple[str, int, int, int, int, str]:
    if not isinstance(value, Mapping):
        raise RuntimeError("job key must be a mapping")
    if set(value) != {
        "dataset",
        "task_id",
        "repeat",
        "fold",
        "sample",
        "arm",
        "arm_code",
    }:
        raise RuntimeError("job key fields are not exact")
    key = (
        value.get("dataset"),
        _exact_int(value.get("task_id"), "job task id"),
        _exact_int(value.get("repeat"), "job repeat"),
        _exact_int(value.get("fold"), "job fold"),
        _exact_int(value.get("sample"), "job sample"),
        value.get("arm"),
    )
    if not isinstance(key[0], str) or not isinstance(key[5], str):
        raise RuntimeError("job dataset and arm must be strings")
    if key not in expected_grid() or dict(value) != _key_payload(key):
        raise RuntimeError(f"job key is not canonical: {value!r}")
    return key


def _resolve_child_resources(jobs: Sequence[Any]) -> int:
    seen = set()
    arms = set()
    for job in jobs:
        experiment = job.experiment
        if id(experiment) in seen:
            continue
        seen.add(id(experiment))
        arm = _job_arm(job)
        arms.add(arm)
        method = experiment.method_kwargs
        fit_kwargs = method.get("fit_kwargs")
        if not isinstance(fit_kwargs, dict):
            raise RuntimeError("CTR23 experiment has no mutable fit kwargs")
        probe = method["model_cls"](
            path="",
            name="CTR23ResourceProbe",
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            hyperparameters=dict(ARM_SPECS[arm]["config"]),
        )
        default_cpus, default_gpus = probe._get_default_resources()
        if int(default_cpus) < EXPECTED_CHILD_CPUS or float(default_gpus) != 0.0:
            raise RuntimeError("host cannot satisfy frozen 18-CPU child allocation")
        fit_kwargs["num_cpus"] = EXPECTED_CHILD_CPUS
    if arms != set(ARM_SPECS):
        raise RuntimeError("resource audit did not cover every CTR23 arm")
    return EXPECTED_CHILD_CPUS


def build_runtime_jobs(
    time_limit: float = TIME_LIMIT_SECONDS,
    *,
    chimeraboost_path: Path = DEFAULT_CHIMERABOOST_PATH,
) -> tuple[Any, list[Any], int]:
    from tabarena.benchmark.experiment import Job
    from tabarena.contexts import TabArenaContext
    from tabarena.utils.config_utils import ConfigGenerator

    model_classes = _load_model_classes(chimeraboost_path)
    context = TabArenaContext(task_metadata=build_task_metadata_collection())
    experiments = build_experiments(
        model_classes=model_classes,
        config_generator_cls=ConfigGenerator,
        time_limit=time_limit,
    )
    jobs = []
    for dataset, task_id, repeat, fold, sample, arm in sorted(
        expected_grid(),
        key=lambda key: (
            next(
                index
                for index, task in enumerate(_task_rows())
                if task["task_id"] == key[1]
            ),
            key[3],
            ("A10", "M", "D", "C").index(key[5]),
        ),
    ):
        del task_id, sample
        jobs.append(
            Job.create(
                experiments[arm],
                dataset=dataset,
                fold=fold,
                repeat=repeat,
            )
        )
    child_cpus = _resolve_child_resources(jobs)
    observed = {_job_key(job) for job in jobs}
    if len(jobs) != EXPECTED_JOBS or observed != expected_grid():
        raise RuntimeError("built jobs do not match the frozen CTR23 grid")
    return context, jobs, child_cpus


SOURCE_FILES = (
    Path("pyproject.toml"),
    Path("darkofit/__init__.py"),
    Path("darkofit/booster.py"),
    Path("darkofit/callbacks.py"),
    Path("darkofit/preprocessing.py"),
    Path("darkofit/sklearn_api.py"),
    Path("benchmarks/tabarena_adapter.py"),
    Path("benchmarks/tabarena_screen_adapters.py"),
    Path("benchmarks/tabarena_comparator_adapters.py"),
    Path("benchmarks/tabarena_ctr23_adapters.py"),
    Path("benchmarks/tabarena_comparator_warmup.py"),
    Path("benchmarks/tabarena_followon_warmup.py"),
    Path("benchmarks/run_tabarena_regression_accuracy_shootout.py"),
    Path("benchmarks/run_ctr23_minimal_confirmation.py"),
    Path("benchmarks/analyze_ctr23_minimal_confirmation.py"),
    Path("benchmarks/ctr23_minimal_confirmation_coordinates.json"),
    Path("benchmarks/ctr23_minimal_confirmation_protocol.md"),
    Path("benchmarks/ctr23_suite_snapshot.json"),
    Path("benchmarks/ctr23_contamination_registry.json"),
    Path("benchmarks/ctr23_partition.json"),
    Path("benchmarks/ctr23_manual_evidence_catalog.json"),
)


def _validate_campaign_namespace(path: Path, *, field: str) -> Path:
    """Require in-repository campaign namespaces to be wholly Git-ignored."""
    repository = REPOSITORY_ROOT.resolve(strict=True)
    namespace = path.resolve()
    repository_parts = tuple(part.casefold() for part in repository.parts)
    namespace_parts = tuple(part.casefold() for part in namespace.parts)
    if namespace_parts[: len(repository_parts)] != repository_parts:
        return namespace
    suffix = namespace.parts[len(repository.parts) :]
    relative = Path(*suffix) if suffix else Path(".")
    if relative == Path("."):
        raise RuntimeError(f"{field} cannot be the campaign repository")
    try:
        ignored = subprocess.run(
            [
                "git",
                "check-ignore",
                "--quiet",
                "--no-index",
                "--",
                str(relative),
            ],
            cwd=repository,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise RuntimeError(f"could not validate {field} Git-ignore state") from exc
    if ignored.returncode == 0:
        return namespace
    if ignored.returncode == 1:
        raise RuntimeError(
            f"{field} is inside the campaign repository but is not Git-ignored"
        )
    raise RuntimeError(
        f"could not validate {field} Git-ignore state: git check-ignore exited "
        f"with status {ignored.returncode}"
    )


def _validated_campaign_output_path(path: Path) -> Path:
    """Reserve raw-result filenames before a campaign namespace is created."""
    lexical_parts = tuple(part.casefold() for part in path.parts)
    if ".." in lexical_parts or "results.pkl" in lexical_parts:
        raise RuntimeError("campaign output path uses a reserved raw-result name")
    resolved = path.resolve()
    if "results.pkl" in (part.casefold() for part in resolved.parts):
        raise RuntimeError("campaign output path uses a reserved raw-result name")
    return resolved


def collect_source_provenance(
    output_dir: Path | None = None,
    *,
    chimeraboost_path: Path = DEFAULT_CHIMERABOOST_PATH,
) -> dict[str, Any]:
    if output_dir is not None:
        _validate_campaign_namespace(output_dir, field="campaign output namespace")
    base = comparators.collect_source_provenance(
        # The CTR23 namespace validator permits in-repository output only when
        # Git ignores it.  Do not delegate the broader one-directory status
        # exemption: a negated or otherwise visible descendant must still make
        # the authenticated source tree dirty.
        output_dir=None,
        chimeraboost_path=chimeraboost_path,
    )
    inherited_files = _as_mapping(
        base.get("files"), "inherited comparator source files"
    )
    files: dict[str, Any] = {
        str(relative): dict(_as_mapping(metadata, f"source file {relative}"))
        for relative, metadata in inherited_files.items()
    }
    for relative in SOURCE_FILES:
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"required CTR23 source is missing: {relative}")
        metadata = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
            "git_blob": subprocess.run(
                ["git", "hash-object", str(path)],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        }
        inherited = files.get(str(relative))
        if inherited is not None and any(
            inherited.get(field) != metadata[field]
            for field in ("sha256", "git_blob")
        ):
            raise RuntimeError(f"inherited source provenance changed: {relative}")
        files[str(relative)] = {**(inherited or {}), **metadata}
    git_head = str(base.get("git_head"))
    git_tree = str(base.get("git_tree"))
    darkofit_subtree = subprocess.run(
        ["git", "rev-parse", "HEAD:darkofit"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if darkofit_subtree != DARKOFIT_SUBTREE:
        raise RuntimeError("DarkoFit package subtree changed from the frozen revision")
    tabarena = _as_mapping(base.get("tabarena"), "TabArena provenance")
    tabarena_head = tabarena.get("git_head", tabarena.get("head"))
    if tabarena_head != TABARENA_COMMIT:
        raise RuntimeError("TabArena checkout changed from the frozen revision")
    chimera = _as_mapping(base.get("chimeraboost"), "ChimeraBoost provenance")
    if chimera.get("git_head", chimera.get("head")) != CHIMERABOOST_TAG_COMMIT:
        raise RuntimeError("ChimeraBoost checkout changed from the frozen tag")
    catboost = _as_mapping(base.get("catboost"), "CatBoost provenance")
    if catboost.get("version") != "1.2.10" or not catboost.get("files"):
        raise RuntimeError("CatBoost wheel provenance changed")
    if not isinstance(base.get("external_adapter_sources"), Mapping):
        raise RuntimeError("external adapter source provenance is missing")
    return {
        **base,
        "git_head": git_head,
        "git_tree": git_tree,
        "darkofit_subtree": darkofit_subtree,
        "files": files,
    }


def collect_runtime_provenance() -> dict[str, Any]:
    value = comparators.collect_runtime_provenance()
    packages = value.get("packages")
    hardware = value.get("hardware")
    expected_packages = {
        "darkofit": "0.9.0",
        "tabarena": "0.0.1",
        "autogluon.common": "1.5.1b20260712",
        "autogluon.core": "1.5.1b20260712",
        "autogluon.features": "1.5.1b20260712",
        "autogluon.tabular": "1.5.1b20260712",
        "numpy": "2.4.6",
        "pandas": "2.3.3",
        "scikit-learn": "1.7.2",
        "scipy": "1.16.3",
        "numba": "0.66.0",
        "llvmlite": "0.48.0",
        "psutil": "7.1.3",
        "catboost": "1.2.10",
        "graphviz": "0.21",
        "chimeraboost": "0.14.1",
        "openml": "0.15.1",
        "pyarrow": "24.0.0",
        "liac-arff": "2.5.0",
    }
    if not isinstance(packages, dict) or not isinstance(hardware, Mapping):
        raise RuntimeError("runtime provenance is incomplete")
    for name, version in expected_packages.items():
        observed = (
            "0.14.1"
            if name == "chimeraboost"
            else importlib.metadata.version(name)
        )
        if observed != version:
            raise RuntimeError(f"{name} version changed: {observed!r}")
        if packages.get(name) not in (None, observed):
            raise RuntimeError(f"{name} delegated provenance changed")
        packages[name] = observed
    if (
        value.get("python_version") != "3.12.13"
        or value.get("machine") != "arm64"
        or hardware.get("logical_cpu_count") != 18
        or hardware.get("physical_cpu_count") != 18
        or hardware.get("process_cpu_affinity_count") != 18
        or hardware.get("total_memory_bytes") != 137_438_953_472
        or platform.machine() != "arm64"
    ):
        raise RuntimeError("runtime host changed from the frozen CTR23 machine")
    return value


def _as_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field} must be a mapping")
    return value


def _exact_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise RuntimeError(f"{field} must be an exact integer")
    return value


def _finite_positive(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise RuntimeError(f"{field} must be finite and strictly positive")
    return number


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or number < 0.0:
        raise RuntimeError(f"{field} must be finite and nonnegative")
    return number


def _strict_json_mapping(value: Any, field: str) -> dict[str, Any]:
    mapping = dict(_as_mapping(value, field))
    try:
        return json.loads(_canonical_json(mapping))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{field} is not strict finite JSON") from exc


def _decode_result_pickle(path: Path) -> Mapping[str, Any]:
    """Decode a runner-owned result.  The detached analyzer never calls this."""
    try:
        payload = path.read_bytes()
        if payload[:2] == b"\x1f\x8b":
            payload = gzip.decompress(payload)
        value = pickle.loads(payload)
    except Exception as exc:
        raise RuntimeError(f"could not decode trusted result artifact {path}") from exc
    return _as_mapping(value, f"result artifact {path}")


def expected_result_relative_path(
    dataset: str,
    task_id: int,
    repeat: int,
    fold: int,
    sample: int,
    arm: str,
) -> str:
    key = (dataset, task_id, repeat, fold, sample, arm)
    if key not in expected_grid():
        raise RuntimeError(f"result path key is outside the frozen grid: {key}")
    return str(
        Path("experiments")
        / "data"
        / _experiment_name(arm)
        / str(task_id)
        / f"{repeat}_{fold}"
        / "results.pkl"
    )


def _result_path(output_dir: Path, job: Any) -> Path:
    return output_dir / expected_result_relative_path(*_job_key(job))


_COMPARATOR_ARM = {
    "D": "darkofit_product_default",
    "M": "chimeraboost_0_14_1_default",
    "C": "catboost_1_2_10_default",
}
_COMPARATOR_ENGINE = {"D": "darkofit", "M": "chimeraboost", "C": "catboost"}


def _expected_resolved_hyperparameters(arm: str) -> dict[str, Any]:
    ensemble = _expected_ag_ensemble_config()
    maximum = ensemble.pop("ag.max_time_limit")
    ensemble["ag_args_fit"] = {"max_time_limit": maximum}
    return {
        **(A10_CONFIG if arm == "A10" else {}),
        "ag_args_ensemble": ensemble,
    }


def _expected_fit_kwargs(num_cpus: int) -> dict[str, Any]:
    if num_cpus != EXPECTED_CHILD_CPUS:
        raise RuntimeError("CTR23 child allocation must equal 18 CPUs")
    return {
        "num_bag_folds": 8,
        "num_bag_sets": 1,
        "raise_on_model_failure": True,
        "calibrate": False,
        "num_cpus": EXPECTED_CHILD_CPUS,
    }


def _validate_native_representation(
    value: Any,
    *,
    field: str,
    child_features: list[str],
    expected_scope: str,
) -> dict[str, Any]:
    representation = _strict_json_mapping(value, field)
    expected_fields = {
        "schema_version",
        "kind",
        "fit_scope",
        "feature_alignment_policy",
        "target_used_by_representation",
        "input_feature_count",
        "output_feature_count",
        "external_feature_schema_sha256",
        "fitted_feature_schema_sha256",
        "categorical_input_columns",
        "fitted_categorical_input_columns",
        "dropped_constant_input_columns",
        "dropped_constant_input_unique_counts",
    }
    if set(representation) != expected_fields:
        raise RuntimeError(f"{field} native representation fields are not exact")
    if (
        representation["schema_version"] != 2
        or representation["kind"] != "native"
        or representation["fit_scope"] != expected_scope
        or representation["feature_alignment_policy"]
        != "autogluon_child_drop_unique"
    ):
        raise RuntimeError(f"{field} native representation identity changed")
    for name in (
        "categorical_input_columns",
        "fitted_categorical_input_columns",
        "dropped_constant_input_columns",
        "dropped_constant_input_unique_counts",
    ):
        if not isinstance(representation[name], list):
            raise RuntimeError(f"{field}.{name} must be a list")
    categorical = representation["categorical_input_columns"]
    fitted_categorical = representation["fitted_categorical_input_columns"]
    dropped = representation["dropped_constant_input_columns"]
    dropped_counts = representation["dropped_constant_input_unique_counts"]
    if any(
        any(not isinstance(item, str) for item in values)
        or len(set(values)) != len(values)
        or values != [name for name in child_features if name in set(values)]
        for values in (categorical, fitted_categorical, dropped)
    ):
        raise RuntimeError(f"{field} ordered feature subsets are invalid")
    if (
        any(type(count) is not int or count != 1 for count in dropped_counts)
        or len(dropped_counts) != len(dropped)
    ):
        raise RuntimeError(f"{field} dropped-feature audit is invalid")
    dropped_set = set(dropped)
    fitted_features = [name for name in child_features if name not in dropped_set]
    expected_fitted_categorical = [
        name for name in categorical if name not in dropped_set
    ]
    feature_digest = hardened.screen._feature_schema_sha256
    if (
        representation["input_feature_count"] != len(child_features)
        or representation["output_feature_count"] != len(fitted_features)
        or representation["external_feature_schema_sha256"]
        != feature_digest(child_features, f"{field}.external")
        or representation["fitted_feature_schema_sha256"]
        != feature_digest(fitted_features, f"{field}.fitted")
        or fitted_categorical != expected_fitted_categorical
        or representation["target_used_by_representation"]
        is not bool(expected_fitted_categorical)
    ):
        raise RuntimeError(f"{field} native representation audit is inconsistent")
    return representation


def _compact_candidate_metadata(
    selection: Mapping[str, Any], *, field: str
) -> dict[str, Any]:
    order = ["catboost", "lightgbm", "hybrid"]
    candidates = selection.get("candidates")
    selected = _exact_int(
        selection.get("selected_candidate_index"), f"{field}.selected index"
    )
    if (
        selection.get("candidate_count") != 3
        or selection.get("fitted_candidate_count") != 3
        or not isinstance(candidates, list)
        or len(candidates) != 3
        or selected not in range(3)
    ):
        raise RuntimeError(f"{field} does not contain all frozen candidates")
    compact = []
    scores = []
    for index, raw in enumerate(candidates):
        candidate = _as_mapping(raw, f"{field}.candidate {index}")
        score = _finite_nonnegative(
            candidate.get("validation_score"),
            f"{field}.candidate {index} validation RMSE",
        )
        reason = candidate.get("stop_reason")
        if (
            candidate.get("tree_mode") != order[index]
            or candidate.get("fit_status") != "fitted"
            or candidate.get("deadline_hit") is not False
            or reason == "time_limit"
        ):
            raise RuntimeError(f"{field}.candidate {index} fitted state is invalid")
        compact.append(
            {
                "candidate_index": index,
                "tree_mode": order[index],
                "fitted": True,
                "validation_rmse": score,
                "deadline_hit": False,
                "stop_reason": reason,
            }
        )
        scores.append(score)
    if selected != min(range(3), key=scores.__getitem__):
        raise RuntimeError(f"{field} selected candidate is not the first argmin")
    return {
        "candidate_count": 3,
        "fitted_candidate_count": 3,
        "candidate_order": order,
        "selected_candidate_index": selected,
        "candidates": compact,
    }


def _parse_a10_fit(
    value: Any,
    *,
    selected_params: Any,
    field: str,
) -> dict[str, Any]:
    fitted = _strict_json_mapping(value, field)
    required = set(hardened.screen.REQUIRED_FIT_METADATA) | {"tree_mode_selection"}
    if set(fitted) != required:
        raise RuntimeError(f"{field} fitted metadata fields are not exact")
    requested = _exact_int(fitted["iterations_requested"], f"{field}.requested")
    attempted = _exact_int(fitted["iterations_attempted"], f"{field}.attempted")
    completed = _exact_int(fitted["rounds_completed"], f"{field}.completed")
    retained = _exact_int(fitted["rounds_retained"], f"{field}.retained")
    best = _exact_int(fitted["best_iteration"], f"{field}.best")
    if requested != 10_000 or not (
        0 <= retained == best <= completed <= attempted <= requested
    ):
        raise RuntimeError(f"{field} round counters are inconsistent")
    hardened.screen._validate_early_stopping_rounds(
        fitted["early_stopping_rounds"],
        field=f"{field}.early_stopping_rounds",
    )
    if _finite_positive(fitted["resolved_learning_rate"], f"{field}.LR") != 0.1:
        raise RuntimeError(f"{field} learning rate changed")
    selected_mode = fitted.get("selected_tree_mode")
    if (
        fitted.get("requested_tree_mode") != "auto"
        or selected_mode not in {"catboost", "lightgbm", "hybrid"}
        or fitted.get("selected_lane") != "boosting"
        or fitted.get("linear_residual_active") is not False
        or fitted.get("deadline_hit") is not False
        or fitted.get("deadline_is_soft") is not True
    ):
        raise RuntimeError(f"{field} resolved A10 lane changed")
    reason = fitted.get("stop_reason")
    hardened.screen.hardened.validate_stop_reason_causality(
        reason,
        requested=requested,
        attempted=attempted,
        completed=completed,
        field=field,
    )
    if reason == "time_limit":
        raise RuntimeError(f"{field} hit the wall-clock limit")
    hardened.screen._validate_child_wall_clock_audit(fitted, field=field)
    selection = _as_mapping(fitted["tree_mode_selection"], f"{field}.selection")
    hardened.screen._validate_tree_mode_selection(
        selection,
        expected_iterations=10_000,
        selected_tree_mode=str(selected_mode),
        deadline_hit=False,
        top_level=fitted,
        field=f"{field}.selection",
    )
    hardened.screen._validate_refit_params(
        selected_params,
        expected_iterations=best,
        selected_tree_mode=str(selected_mode),
        field=f"{field}.refit",
    )
    return {
        "iterations_requested": requested,
        "iterations_attempted": attempted,
        "rounds_completed": completed,
        "rounds_retained": retained,
        "best_iteration": best,
        "resolved_learning_rate": 0.1,
        "requested_tree_mode": "auto",
        "selected_tree_mode": selected_mode,
        "selected_lane": "boosting",
        "stop_reason": reason,
        "deadline_hit": False,
        "deadline_is_soft": True,
        "time_callback_hit": False,
        "time_callback_instance_count": 0,
        "time_callback_call_count": 0,
        "candidate_metadata": _compact_candidate_metadata(
            selection, field=f"{field}.selection"
        ),
    }


def _parse_comparator_fit(
    value: Any,
    *,
    audit_value: Any,
    arm: str,
    field: str,
) -> dict[str, Any]:
    base_arm = _COMPARATOR_ARM[arm]
    fit = comparators._validate_comparator_fit(value, arm=base_arm, field=field)
    requested = _exact_int(fit["iterations_requested"], f"{field}.requested")
    attempted = _exact_int(
        fit.get("iterations_attempted", fit["rounds_retained"]),
        f"{field}.attempted",
    )
    retained = _exact_int(fit["rounds_retained"], f"{field}.retained")
    best = _exact_int(fit["best_iteration"], f"{field}.best")
    completed = _exact_int(
        fit.get("rounds_completed", attempted), f"{field}.completed"
    )
    if not (0 <= retained == best <= completed <= attempted <= requested):
        raise RuntimeError(f"{field} comparator counters are inconsistent")
    reason = fit.get("stop_reason")
    resolved_lr = _finite_positive(fit["resolved_learning_rate"], f"{field}.LR")
    if arm == "D":
        if audit_value is not None:
            raise RuntimeError(f"{field} product default has an unexpected sidecar")
        if (
            requested != 1_000
            or fit.get("requested_tree_mode") != "catboost"
            or fit.get("selected_tree_mode") != "catboost"
            or fit.get("selected_lane") != "boosting"
            or fit.get("deadline_hit") is not False
            or fit.get("deadline_is_soft") is not True
        ):
            raise RuntimeError(f"{field} DarkoFit default contract changed")
        hardened.screen.hardened.validate_stop_reason_causality(
            reason,
            requested=requested,
            attempted=attempted,
            completed=completed,
            field=field,
        )
        requested_mode = selected_mode = "catboost"
        selected_lane = "boosting"
        callback_hit = False
        callback_instances = callback_calls = 0
    elif arm == "M":
        audit = _strict_json_mapping(audit_value, f"{field}.CTR23 callback audit")
        callback_limit = _finite_positive(
            audit.get("time_limit_seconds"), f"{field}.time limit"
        )
        callback_hit = audit.get("time_callback_hit")
        callback_instances = _exact_int(
            audit.get("time_callback_instance_count"),
            f"{field}.callback model count",
        )
        callback_calls = _exact_int(
            audit.get("time_callback_call_count"), f"{field}.callback invocations"
        )
        if (
            requested != 10_000
            or not math.isclose(resolved_lr, 0.1, rel_tol=1e-7, abs_tol=1e-12)
            or fit.get("selected_lane") not in {"constant", "linear"}
            or callback_limit > TIME_LIMIT_SECONDS
            or audit.get("time_callback_instrumented") is not True
            or callback_hit is not False
            or callback_calls < callback_instances
            or callback_instances
            != (2 if fit.get("linear_selection_performed") is True else 1)
            or set(audit)
            != {
                "schema_version",
                "kind",
                "engine",
                "time_limit_seconds",
                "time_callback_instrumented",
                "time_callback_instance_count",
                "time_callback_call_count",
                "time_callback_hit",
            }
            or audit.get("schema_version") != 1
            or audit.get("kind") != "darkofit_ctr23_time_callback_audit"
            or audit.get("engine") != "chimeraboost"
        ):
            raise RuntimeError(f"{field} ChimeraBoost default contract changed")
        requested_mode = selected_mode = None
        selected_lane = fit["selected_lane"]
    else:
        audit = _strict_json_mapping(audit_value, f"{field}.CTR23 callback audit")
        callback_limit = _finite_positive(
            audit.get("time_limit_seconds"), f"{field}.time limit"
        )
        callback_hit = audit.get("time_callback_hit")
        callback_instances = _exact_int(
            audit.get("time_callback_instance_count"), f"{field}.callback instances"
        )
        callback_calls = _exact_int(
            audit.get("time_callback_call_count"), f"{field}.callback invocations"
        )
        if (
            requested != 10_000
            or not math.isclose(resolved_lr, 0.05, rel_tol=1e-7, abs_tol=1e-12)
            or callback_limit > TIME_LIMIT_SECONDS
            or audit.get("time_callback_instrumented") is not True
            or callback_hit is not False
            or callback_instances != 1
            or callback_calls < 1
            or set(audit)
            != {
                "schema_version",
                "kind",
                "engine",
                "time_limit_seconds",
                "time_callback_instrumented",
                "time_callback_instance_count",
                "time_callback_call_count",
                "time_callback_hit",
            }
            or audit.get("schema_version") != 1
            or audit.get("kind") != "darkofit_ctr23_time_callback_audit"
            or audit.get("engine") != "catboost"
        ):
            raise RuntimeError(f"{field} CatBoost default contract changed")
        requested_mode = selected_mode = None
        selected_lane = "cpu"
    if reason == "time_limit" or callback_hit is not False:
        raise RuntimeError(f"{field} hit the wall-clock limit")
    return {
        "iterations_requested": requested,
        "iterations_attempted": attempted,
        "rounds_completed": completed,
        "rounds_retained": retained,
        "best_iteration": best,
        "resolved_learning_rate": resolved_lr,
        "requested_tree_mode": requested_mode,
        "selected_tree_mode": selected_mode,
        "selected_lane": selected_lane,
        "stop_reason": reason,
        # M/C persist an explicit audited callback state; a null inferred stop
        # reason is retained rather than guessed.
        "deadline_hit": False,
        "deadline_is_soft": True,
        "time_callback_hit": False,
        "time_callback_instance_count": callback_instances,
        "time_callback_call_count": callback_calls,
        "candidate_metadata": None,
    }


def _arm_from_record_method(method: Mapping[str, Any], field: str) -> str:
    raw = dict(_as_mapping(method.get("model_hyperparameters"), field))
    ag_args = raw.pop("ag_args", None)
    ensemble = raw.pop("ag_args_ensemble", None)
    model_cls = method.get("model_cls")
    matches = []
    for arm, spec in ARM_SPECS.items():
        if (
            raw == spec["config"]
            and model_cls == spec["model_cls"]
            and ag_args == {"name_suffix": _experiment_suffix(arm)}
        ):
            matches.append(arm)
    if len(matches) != 1:
        raise RuntimeError(f"{field} does not identify exactly one frozen arm")
    if not isinstance(ensemble, Mapping) or dict(ensemble) != _expected_ag_ensemble_config():
        raise RuntimeError(f"{field} ensemble configuration changed")
    return matches[0]


def parse_result_record(
    record: Mapping[str, Any], *, source: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one trusted result and project only the preregistered safe fields."""
    if record.get("problem_type") != "regression" or record.get("metric") != "rmse":
        raise RuntimeError(f"{source}: wrong problem type or metric")
    if record.get("imputed", False) not in (False, None):
        raise RuntimeError(f"{source}: result was imputed")
    test_rmse = _finite_positive(record.get("metric_error"), f"{source}: test RMSE")
    val_rmse = _finite_positive(
        record.get("metric_error_val"), f"{source}: validation RMSE"
    )
    _finite_positive(record.get("time_train_s"), f"{source}: train time")
    _finite_positive(record.get("time_infer_s"), f"{source}: inference time")
    memory = _as_mapping(record.get("memory_usage"), f"{source}: memory")
    _finite_positive(memory.get("peak_mem_cpu"), f"{source}: peak memory")
    experiment = _as_mapping(
        record.get("experiment_metadata"), f"{source}: experiment metadata"
    )
    if (
        experiment.get("experiment_cls") != "OOFExperimentRunner"
        or experiment.get("method_cls") != "AGSingleBagWrapper"
    ):
        raise RuntimeError(f"{source}: wrong experiment implementation")

    method = _as_mapping(record.get("method_metadata"), f"{source}: method")
    arm = _arm_from_record_method(method, f"{source}: model hyperparameters")
    if record.get("framework") != _experiment_name(arm):
        raise RuntimeError(f"{source}: framework name does not match arm")
    if dict(
        _as_mapping(method.get("hyperparameters"), f"{source}: resolved params")
    ) != _expected_resolved_hyperparameters(arm):
        raise RuntimeError(f"{source}: resolved method policy changed")
    expected_identity = {
        "A10": ("DARKO", "DarkoFit"),
        "D": ("DARKO", "DarkoFit"),
        "M": ("CHIMERA", "ChimeraBoost"),
        "C": ("CAT", "CatBoost"),
    }[arm]
    if (
        (method.get("model_type"), method.get("name_prefix")) != expected_identity
        or method.get("init_kwargs_extra") != {}
    ):
        raise RuntimeError(f"{source}: resolved model identity changed")
    num_cpus = _exact_int(method.get("num_cpus"), f"{source}: num_cpus")
    num_gpus = _exact_int(method.get("num_gpus"), f"{source}: num_gpus")
    num_cpus_child = _exact_int(
        method.get("num_cpus_child"), f"{source}: num_cpus_child"
    )
    num_gpus_child = _exact_int(
        method.get("num_gpus_child"), f"{source}: num_gpus_child"
    )
    if (
        num_cpus != EXPECTED_CHILD_CPUS
        or num_cpus_child != EXPECTED_CHILD_CPUS
        or num_gpus != 0
        or num_gpus_child != 0
        or dict(
            _as_mapping(method.get("fit_kwargs_extra"), f"{source}: fit kwargs")
        )
        != _expected_fit_kwargs(num_cpus)
    ):
        raise RuntimeError(f"{source}: resolved resource/bag settings changed")

    task = _as_mapping(record.get("task_metadata"), f"{source}: task")
    dataset = str(task.get("name"))
    task_row = next(
        (item for item in _task_rows() if item["dataset_name"] == dataset), None
    )
    if task_row is None:
        raise RuntimeError(f"{source}: unexpected dataset {dataset}")
    task_id = _exact_int(task.get("tid"), f"{source}: task id")
    repeat = _exact_int(task.get("repeat"), f"{source}: repeat")
    fold = _exact_int(task.get("fold"), f"{source}: fold")
    key = (dataset, task_id, repeat, fold, 0, arm)
    if (
        key not in expected_grid()
        or task_id != task_row["task_id"]
        or task.get("split_idx") not in (None, 3 * repeat + fold)
    ):
        raise RuntimeError(f"{source}: task coordinate is outside the frozen grid")
    if source != expected_result_relative_path(*key):
        raise RuntimeError(f"{source}: result source does not match its coordinate")

    outer_fit = _as_mapping(method.get("fit_metadata"), f"{source}: outer fit")
    if (
        outer_fit.get("num_cpus") != EXPECTED_CHILD_CPUS
        or outer_fit.get("num_gpus") != 0
        or outer_fit.get("val_in_fit") is not False
        or outer_fit.get("unlabeled_in_fit") is not False
    ):
        raise RuntimeError(f"{source}: outer fit metadata changed")
    info = _as_mapping(method.get("info"), f"{source}: model info")
    if (
        info.get("is_valid") is not True
        or info.get("can_infer") is not True
        or info.get("model_type") != "StackerEnsembleModel"
        or info.get("problem_type") != "regression"
        or info.get("eval_metric") != "root_mean_squared_error"
        or info.get("stopping_metric") != "root_mean_squared_error"
        or info.get("num_cpus") != EXPECTED_CHILD_CPUS
        or info.get("num_gpus") != 0
        or info.get("val_in_fit") is not False
        or info.get("unlabeled_in_fit") is not False
    ):
        raise RuntimeError(f"{source}: outer model metadata changed")
    outer_features = info.get("features")
    hardened.screen._feature_schema_sha256(
        outer_features, f"{source}: outer features"
    )
    if _exact_int(info.get("num_features"), f"{source}: feature count") != len(
        outer_features
    ):
        raise RuntimeError(f"{source}: outer feature count changed")

    bag = _as_mapping(info.get("bagged_info"), f"{source}: bag")
    child_names = [f"S1F{index}" for index in range(1, 9)]
    expected_model_cls = ARM_SPECS[arm]["model_cls"]
    expected_user = A10_CONFIG if arm == "A10" else {}
    first_expected = (
        {
            **A10_CONFIG,
            "diagnostic_warnings": "never",
            "random_state": 0,
        }
        if arm == "A10"
        else comparators.expected_child_hyperparameters(
            _COMPARATOR_ENGINE[arm], 0
        )
    )
    if (
        bag.get("num_child_models") != 8
        or bag.get("child_model_type") != expected_model_cls
        or bag.get("child_model_names") != child_names
        or bag.get("_n_repeats") != 1
        or bag.get("_k_per_n_repeat") != [8]
        or bag.get("_random_state") != 1
        or bag.get("bagged_mode") is not True
        or dict(
            _as_mapping(
                bag.get("child_hyperparameters_user"),
                f"{source}: child user params",
            )
        )
        != expected_user
        or dict(
            _as_mapping(bag.get("child_hyperparameters"), f"{source}: child params")
        )
        != first_expected
    ):
        raise RuntimeError(f"{source}: bag construction changed")
    ag_fit = _as_mapping(bag.get("child_ag_args_fit"), f"{source}: child ag args")
    expected_ag_fit = {
        "max_memory_usage_ratio": 1.0,
        "max_time_limit_ratio": 1.0,
        "max_time_limit": None,
        "min_time_limit": 0,
    }
    if any(ag_fit.get(name) != value for name, value in expected_ag_fit.items()):
        raise RuntimeError(f"{source}: child budget ratios changed")
    children = _as_mapping(info.get("children_info"), f"{source}: children")
    if set(children) != set(child_names):
        raise RuntimeError(f"{source}: child set is incomplete")

    child_rows = []
    child_best = []
    for child_fold, child_name in enumerate(child_names):
        field = f"{source}: {child_name}"
        child = _as_mapping(children[child_name], field)
        child_features = child.get("features")
        hardened.screen._feature_schema_sha256(
            child_features, f"{field}.features"
        )
        expected_child = (
            {
                **A10_CONFIG,
                "diagnostic_warnings": "never",
                "random_state": child_fold,
            }
            if arm == "A10"
            else comparators.expected_child_hyperparameters(
                _COMPARATOR_ENGINE[arm], child_fold
            )
        )
        if (
            child.get("name") != child_name
            or child.get("model_type") != expected_model_cls
            or child.get("is_valid") is not True
            or child.get("can_infer") is not True
            or child.get("problem_type") != "regression"
            or child.get("eval_metric") != "root_mean_squared_error"
            or child.get("stopping_metric") != "root_mean_squared_error"
            or child.get("num_cpus") != EXPECTED_CHILD_CPUS
            or child.get("num_gpus") != 0
            or child.get("val_in_fit") is not True
            or child.get("unlabeled_in_fit") is not False
            or _exact_int(child.get("num_features"), f"{field}.feature count")
            != len(child_features)
            or set(child_features) != set(outer_features)
            or dict(
                _as_mapping(child.get("hyperparameters"), f"{field}.params")
            )
            != expected_child
            or dict(
                _as_mapping(child.get("hyperparameters_user"), f"{field}.user params")
            )
            != expected_user
        ):
            raise RuntimeError(f"{field} initialized/fitted state changed")
        child_ag = _as_mapping(child.get("ag_args_fit"), f"{field}.ag args")
        if any(child_ag.get(name) != value for name, value in expected_ag_fit.items()):
            raise RuntimeError(f"{field} budget ratios changed")
        _validate_native_representation(
            child.get("benchmark_representation"),
            field=f"{field}.representation",
            child_features=list(child_features),
            expected_scope=(
                "darkofit_child_training_fold"
                if arm == "A10"
                else "comparator_child_training_fold"
            ),
        )
        fitted = (
            _parse_a10_fit(
                child.get("darkofit_fit"),
                selected_params=child.get("hyperparameters_fit"),
                field=f"{field}.darkofit_fit",
            )
            if arm == "A10"
            else _parse_comparator_fit(
                child.get("comparator_fit"),
                audit_value=child.get("ctr23_time_callback_audit"),
                arm=arm,
                field=f"{field}.comparator_fit",
            )
        )
        child_best.append(int(fitted["best_iteration"]))
        child_rows.append(
            {
                "dataset": dataset,
                "task_id": task_id,
                "repeat": repeat,
                "fold": fold,
                "sample": 0,
                "arm": arm,
                "child_fold": child_fold,
                "source": source,
                "num_cpus": EXPECTED_CHILD_CPUS,
                "num_gpus": 0,
                **fitted,
            }
        )

    if arm == "A10":
        hardened.screen._validate_auto_compressed_refit_params(
            bag.get("child_hyperparameters_fit"),
            child_best=child_best,
            field=f"{source}: compressed A10 refit params",
        )

    outer = {
        "dataset": dataset,
        "task_id": task_id,
        "repeat": repeat,
        "fold": fold,
        "sample": 0,
        "arm": arm,
        "test_rmse": test_rmse,
        "val_rmse": val_rmse,
        "source": source,
        "num_cpus": EXPECTED_CHILD_CPUS,
        "num_cpus_child": EXPECTED_CHILD_CPUS,
        "num_gpus": 0,
        "num_gpus_child": 0,
    }
    if tuple(outer) != OUTER_PAYLOAD_FIELDS or any(
        tuple(row) != CHILD_PAYLOAD_FIELDS for row in child_rows
    ):
        raise RuntimeError(f"{source}: safe projection field order changed")
    return outer, child_rows


def parse_result_path(path: Path, *, output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    current = path.absolute()
    root_absolute = output_dir.absolute()
    while current != root_absolute:
        if current.is_symlink():
            raise RuntimeError("result path contains a symlink component")
        if current.parent == current:
            raise RuntimeError("result path is not beneath the campaign directory")
        current = current.parent
    try:
        relative = path.resolve(strict=True).relative_to(output_dir.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise RuntimeError("result path is not confined to the campaign directory") from exc
    if path.is_symlink() or path.name != "results.pkl":
        raise RuntimeError("result artifact must be a regular non-symlink results.pkl")
    return parse_result_record(_decode_result_pickle(path), source=str(relative))


def behavior_fingerprint(path: Path, *, output_dir: Path) -> tuple[str, int, bool, int]:
    outer, children = parse_result_path(path, output_dir=output_dir)
    normalized = {
        "outer": {key: value for key, value in outer.items() if key != "source"},
        "children": [
            {key: value for key, value in row.items() if key != "source"}
            for row in children
        ],
    }
    digest = hashlib.sha256(_canonical_json(normalized)).hexdigest()
    deadline = any(
        row["deadline_hit"] is not False or row["stop_reason"] == "time_limit"
        for row in children
    )
    candidate_count = sum(
        row["candidate_metadata"]["fitted_candidate_count"]
        for row in children
        if row["arm"] == "A10"
    )
    return digest, len(children), deadline, candidate_count


def _wait_until(release_ns: int) -> None:
    while True:
        remaining = (release_ns - time.monotonic_ns()) / 1e9
        if remaining <= 0.0:
            return
        time.sleep(min(remaining, 0.01))


def _self_peak_rss_bytes() -> int:
    import resource

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1_024


def _combined_synthetic_warmup(thread_count: int) -> dict[str, Any]:
    try:
        from benchmarks.tabarena_comparator_warmup import (
            warmup_tabarena_comparators,
        )
        from benchmarks.tabarena_followon_warmup import (
            warmup_tabarena_followon_screen,
        )
    except ModuleNotFoundError:
        from tabarena_comparator_warmup import warmup_tabarena_comparators
        from tabarena_followon_warmup import warmup_tabarena_followon_screen
    return {
        "darkofit": warmup_tabarena_followon_screen(thread_count=thread_count),
        "comparators": warmup_tabarena_comparators(thread_count=thread_count),
    }


def _synthetic_behavior_projection(value: Any) -> Any:
    """Drop operational observations while preserving model/config fingerprints."""
    if isinstance(value, Mapping):
        return {
            str(key): _synthetic_behavior_projection(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if "seconds" not in str(key)
            and "duration" not in str(key)
            and "memory" not in str(key)
            and "rss" not in str(key)
        }
    if isinstance(value, (list, tuple)):
        return [_synthetic_behavior_projection(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError("synthetic behavior contains a non-finite value")
        return {"float_hex": value.hex()}
    return value


def _worker_main(
    slot: int,
    connection: Any,
    scratch_root: str,
    time_limit: float,
) -> None:
    """Own one context for the worker lifetime and run only parent commands."""
    try:
        scratch = Path(scratch_root).resolve()
        scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(scratch, 0o700)
        os.chdir(scratch)
        context, jobs, child_cpus = build_runtime_jobs(time_limit)
        lookup = {_job_key(job): job for job in jobs}
        if set(lookup) != expected_grid():
            raise RuntimeError("worker runtime lookup is incomplete")
        connection.send(
            {
                "type": "ready",
                "slot": slot,
                "pid": os.getpid(),
                "child_cpus": child_cpus,
                "start_method": mp.get_start_method(),
                "scratch_root": str(scratch),
            }
        )
        while True:
            command = connection.recv()
            command_id = str(command.get("command_id"))
            kind = command.get("kind")
            if kind == "stop":
                connection.send(
                    {"type": "stopped", "command_id": command_id, "slot": slot}
                )
                return
            if kind == "warmup":
                connection.send(
                    {
                        "type": "warmup",
                        "command_id": command_id,
                        "slot": slot,
                        "pid": os.getpid(),
                        "warmup": _combined_synthetic_warmup(child_cpus),
                    }
                )
                continue
            if kind == "synthetic_probe":
                release_ns = _exact_int(
                    command["release_monotonic_ns"], "synthetic barrier release"
                )
                _wait_until(release_ns)
                started_ns = time.monotonic_ns()
                warmup = _combined_synthetic_warmup(child_cpus)
                ended_ns = time.monotonic_ns()
                connection.send(
                    {
                        "type": "synthetic_probe",
                        "command_id": command_id,
                        "slot": slot,
                        "pid": os.getpid(),
                        "behavior_sha256": hashlib.sha256(
                            _canonical_json(_synthetic_behavior_projection(warmup))
                        ).hexdigest(),
                        "barrier_release_monotonic_ns": release_ns,
                        "started_monotonic_ns": started_ns,
                        "ended_monotonic_ns": ended_ns,
                        "process_peak_rss_bytes": _self_peak_rss_bytes(),
                    }
                )
                continue
            if kind == "prime":
                keys = [_key_tuple(value) for value in command.get("keys", [])]
                if not keys:
                    raise RuntimeError("worker prime requires frozen job keys")
                with context._cache_scope():
                    context.task_metadata_collection.subset_to_jobs(
                        [lookup[key] for key in keys]
                    ).materialize()
                connection.send(
                    {
                        "type": "prime",
                        "command_id": command_id,
                        "slot": slot,
                        "pid": os.getpid(),
                        "keys": [_key_payload(key) for key in keys],
                    }
                )
                continue
            if kind == "resource":
                connection.send(
                    {
                        "type": "resource",
                        "command_id": command_id,
                        "slot": slot,
                        "pid": os.getpid(),
                        "process_peak_rss_bytes": _self_peak_rss_bytes(),
                    }
                )
                continue
            if kind != "run":
                raise RuntimeError(f"unknown worker command: {kind!r}")
            key = _key_tuple(command["key"])
            result_root = Path(command["result_root"]).resolve()
            release_ns = _exact_int(
                command["release_monotonic_ns"], "barrier release"
            )
            _wait_until(release_ns)
            started_ns = time.monotonic_ns()
            results = context.run_jobs(
                [lookup[key]],
                expname=str(result_root / "experiments"),
                new_result_prefix="[CTR23 minimal confirmation] ",
                debug_mode=True,
                register=False,
            )
            ended_ns = time.monotonic_ns()
            path = _result_path(result_root, lookup[key])
            behavior, child_count, deadline_hit, candidate_count = (
                behavior_fingerprint(path, output_dir=result_root)
            )
            _, child_rows = parse_result_path(path, output_dir=result_root)
            callback_hits = sum(row["time_callback_hit"] is True for row in child_rows)
            connection.send(
                {
                    "type": "result",
                    "command_id": command_id,
                    "status": "ok",
                    "slot": slot,
                    "pid": os.getpid(),
                    "key": _key_payload(key),
                    "result_root": str(result_root),
                    "result_path": str(path),
                    "result_count": len(results),
                    "child_count": child_count,
                    "deadline_hit": deadline_hit,
                    "time_callback_hit_count": callback_hits,
                    "a10_candidate_fit_count": candidate_count,
                    "behavior_sha256": behavior,
                    "result_sha256": _sha256_file(path),
                    "result_size_bytes": path.stat().st_size,
                    "process_peak_rss_bytes": _self_peak_rss_bytes(),
                    "barrier_release_monotonic_ns": release_ns,
                    "started_monotonic_ns": started_ns,
                    "ended_monotonic_ns": ended_ns,
                    "start_method": mp.get_start_method(),
                }
            )
    except Exception as exc:
        try:
            connection.send(
                {
                    "type": "error",
                    "slot": slot,
                    "pid": os.getpid(),
                    "command_id": locals().get("command_id"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            connection.close()


def _start_workers(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(root, 0o700)
    original = hardened._worker_main
    hardened._worker_main = _worker_main
    try:
        try:
            return hardened._start_workers(root, worker_count=WORKER_COUNT)
        except BaseException as exc:
            cleanup_message = exc.args[0] if exc.args else None
            cleanup_unconfirmed = (
                isinstance(cleanup_message, str)
                and cleanup_message.startswith(
                    "worker startup cleanup could not be confirmed:"
                )
            )
            if not cleanup_unconfirmed:
                try:
                    setattr(
                        exc,
                        _WORKER_STARTUP_CLEANUP_CONFIRMED_ATTRIBUTE,
                        True,
                    )
                except BaseException:
                    pass
            raise
    finally:
        hardened._worker_main = original


def _stop_workers(workers: Sequence[Mapping[str, Any]], *, force: bool = False) -> None:
    hardened._stop_workers(workers, force=force)


def _warm_workers(workers: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    try:
        from benchmarks.tabarena_comparator_warmup import (
            validate_comparator_warmup_history,
        )
    except ModuleNotFoundError:
        from tabarena_comparator_warmup import validate_comparator_warmup_history

    records = []
    for worker in workers:  # Deliberately serial: never warm both at once.
        command_id = f"warmup-{worker['slot']}-{time.monotonic_ns()}"
        worker["connection"].send({"kind": "warmup", "command_id": command_id})
        messages, _ = hardened._await_commands(
            workers,
            {command_id},
            timeout_seconds=1_800.0,
            command_slots={command_id: worker["slot"]},
        )
        message = messages[0]
        warmup = _as_mapping(message.get("warmup"), "worker warmup")
        record = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "pid": message.get("pid"),
            "worker_slot": message.get("slot"),
            "warmup": dict(warmup),
        }
        if (
            message.get("type") != "warmup"
            or message.get("slot") != worker["slot"]
            or message.get("pid") != worker["process"].pid
            or set(warmup) != {"darkofit", "comparators"}
        ):
            raise RuntimeError("worker warmup identity changed")
        base_record = {
            "completed_at_utc": record["completed_at_utc"],
            "pid": record["pid"],
        }
        hardened.screen._validate_followon_warmup_history(
            [{**base_record, "warmup": warmup["darkofit"]}],
            expected_thread_count=EXPECTED_CHILD_CPUS,
            expected_latest_pid=record["pid"],
        )
        validate_comparator_warmup_history(
            [{**base_record, "warmup": warmup["comparators"]}],
            expected_thread_count=EXPECTED_CHILD_CPUS,
            expected_latest_pid=record["pid"],
        )
        records.append(record)
    return records


def _prime_workers(
    workers: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    prime_keys = sorted(
        [key for key in expected_grid() if key[-1] == "A10"],
        key=lambda key: (key[1], key[3]),
    )
    records = []
    for worker in workers:
        command_id = f"prime-{worker['slot']}-{time.monotonic_ns()}"
        worker["connection"].send(
            {
                "kind": "prime",
                "command_id": command_id,
                "keys": [_key_payload(key) for key in prime_keys],
            }
        )
        messages, _ = hardened._await_commands(
            workers,
            {command_id},
            timeout_seconds=1_800.0,
            command_slots={command_id: worker["slot"]},
        )
        message = messages[0]
        if (
            message.get("type") != "prime"
            or message.get("slot") != worker["slot"]
            or [_key_tuple(value) for value in message.get("keys", [])]
            != prime_keys
        ):
            raise RuntimeError("worker data prime changed")
        records.append(dict(message))
    return records


def _new_worker_session_swap_telemetry() -> dict[str, Any]:
    return hardened._new_swap_session_telemetry()


def _checkpoint_worker_session_swap_telemetry(
    telemetry: dict[str, Any],
) -> None:
    hardened._checkpoint_swap_session(telemetry, SWAP_POLICY)


def _validated_worker_session_swap_telemetry(
    value: Any, field: str,
) -> dict[str, Any]:
    telemetry = dict(_as_mapping(value, field))
    if not hardened._swap_session_telemetry_valid(telemetry, SWAP_POLICY):
        raise RuntimeError(f"{field} violates the swap policy")
    return telemetry


def _validated_failure_swap_telemetry(
    value: Any, field: str,
) -> dict[str, Any]:
    """Validate the bounded lifecycle snapshot retained for a failed attempt."""
    record = dict(_as_mapping(value, field))
    if (
        set(record) != _FAILURE_SWAP_TELEMETRY_FIELDS
        or record.get("capture_status") not in {"captured", "capture_failed"}
        or type(record.get("teardown_confirmed")) is not bool
        or type(record.get("post_teardown_sample_recorded")) is not bool
        or (
            record.get("diagnostic") is not None
            and (
                not isinstance(record.get("diagnostic"), str)
                or not 0 < len(record["diagnostic"]) <= 4_096
            )
        )
    ):
        raise RuntimeError(f"{field} fields are invalid")

    telemetry_value = record.get("worker_session_swap_telemetry")
    if telemetry_value is None:
        if (
            record.get("swap_in_bytes") is not None
            or record.get("swap_out_bytes") is not None
        ):
            raise RuntimeError(f"{field} deltas have no lifecycle telemetry")
    else:
        telemetry = dict(_as_mapping(telemetry_value, f"{field} lifecycle"))
        if not hardened._swap_session_telemetry_structurally_valid(telemetry):
            raise RuntimeError(f"{field} lifecycle telemetry is malformed")
        if (
            type(record.get("swap_in_bytes")) is not int
            or record.get("swap_in_bytes") != telemetry["swap_in_delta"]
            or type(record.get("swap_out_bytes")) is not int
            or record.get("swap_out_bytes") != telemetry["swap_out_delta"]
        ):
            raise RuntimeError(f"{field} deltas do not match the lifecycle")
        record["worker_session_swap_telemetry"] = telemetry

    if record["capture_status"] == "captured":
        if (
            record["post_teardown_sample_recorded"] is not True
            or telemetry_value is None
            or record["worker_session_swap_telemetry"]["sample_count"] < 2
        ):
            raise RuntimeError(f"{field} does not retain the teardown boundary")
    elif record.get("diagnostic") is None:
        raise RuntimeError(f"{field} capture failure has no diagnostic")
    if not record["teardown_confirmed"] and record.get("diagnostic") is None:
        raise RuntimeError(f"{field} teardown failure has no diagnostic")
    if (
        record["capture_status"] == "captured"
        and record["teardown_confirmed"]
        and record.get("diagnostic") is not None
    ):
        raise RuntimeError(f"{field} has a spurious diagnostic")
    return record


def _completed_session_failure_swap_telemetry(value: Any) -> dict[str, Any]:
    """Project an already validated, post-shutdown session into failure evidence."""
    telemetry = _validated_worker_session_swap_telemetry(
        value, "completed worker-session swap telemetry"
    )
    return _validated_failure_swap_telemetry(
        {
            "capture_status": "captured",
            "teardown_confirmed": True,
            "post_teardown_sample_recorded": True,
            "worker_session_swap_telemetry": telemetry,
            "swap_in_bytes": telemetry["swap_in_delta"],
            "swap_out_bytes": telemetry["swap_out_delta"],
            "diagnostic": None,
        },
        "completed worker-session failure projection",
    )


def _safe_exception_text(error: BaseException) -> str:
    try:
        return str(error) or type(error).__name__
    except BaseException:
        return f"{type(error).__name__}: <unprintable exception>"


def _failure_diagnostic(error: BaseException, prefix: str) -> str:
    detail = _safe_exception_text(error)
    return f"{prefix}: {type(error).__name__}: {detail}"[:4_096]


def _capture_post_teardown_swap_telemetry(
    telemetry: dict[str, Any], *, teardown_error: BaseException | None,
) -> dict[str, Any]:
    """Best-effort snapshot after worker teardown without masking the run error."""
    diagnostics: list[str] = []
    sample_recorded = False
    try:
        hardened._append_swap_session_sample(telemetry)
        sample_recorded = True
    except BaseException as exc:
        diagnostics.append(_failure_diagnostic(exc, "swap capture failed"))

    snapshot: dict[str, Any] | None = None
    try:
        candidate = json.loads(_canonical_json(telemetry))
        if not hardened._swap_session_telemetry_structurally_valid(candidate):
            raise RuntimeError("worker-session swap telemetry is malformed")
        snapshot = dict(candidate)
    except BaseException as exc:
        diagnostics.append(_failure_diagnostic(exc, "swap snapshot failed"))

    if teardown_error is not None:
        diagnostics.append(_failure_diagnostic(teardown_error, "worker teardown failed"))
    captured = sample_recorded and snapshot is not None
    record = {
        "capture_status": "captured" if captured else "capture_failed",
        "teardown_confirmed": teardown_error is None,
        "post_teardown_sample_recorded": sample_recorded,
        "worker_session_swap_telemetry": snapshot,
        "swap_in_bytes": None if snapshot is None else snapshot["swap_in_delta"],
        "swap_out_bytes": None if snapshot is None else snapshot["swap_out_delta"],
        "diagnostic": None if not diagnostics else "; ".join(diagnostics)[:4_096],
    }
    return _validated_failure_swap_telemetry(
        record, "failed-attempt swap telemetry"
    )


def _attach_failure_swap_telemetry(
    error: BaseException, record: Mapping[str, Any],
) -> None:
    try:
        setattr(error, _FAILURE_SWAP_TELEMETRY_ATTRIBUTE, dict(record))
    except BaseException:
        pass


def _add_error_note(error: BaseException, note: str) -> None:
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        try:
            add_note(note)
        except BaseException:
            pass


def _finalize_worker_session(
    workers: Sequence[Mapping[str, Any]],
    telemetry: dict[str, Any],
    *,
    active_error: BaseException | None,
    worker_start_completed: bool = True,
    startup_cleanup_confirmed: bool = False,
) -> dict[str, Any]:
    """Stop workers, retain the final counter boundary, and preserve root cause."""
    teardown_error: BaseException | None = None
    try:
        _stop_workers(workers, force=active_error is not None)
    except BaseException as exc:
        teardown_error = exc
    if (
        not worker_start_completed
        and not startup_cleanup_confirmed
        and teardown_error is None
    ):
        teardown_error = RuntimeError(
            "worker startup did not return ownership; cleanup is not independently "
            "confirmed"
        )

    try:
        record = _capture_post_teardown_swap_telemetry(
            telemetry, teardown_error=teardown_error
        )
    except BaseException as exc:
        diagnostics = [_failure_diagnostic(exc, "swap capture failed")]
        if teardown_error is not None:
            diagnostics.append(
                _failure_diagnostic(teardown_error, "worker teardown failed")
            )
        record = {
            "capture_status": "capture_failed",
            "teardown_confirmed": teardown_error is None,
            "post_teardown_sample_recorded": False,
            "worker_session_swap_telemetry": None,
            "swap_in_bytes": None,
            "swap_out_bytes": None,
            "diagnostic": "; ".join(diagnostics)[:4_096],
        }
    if active_error is not None:
        _attach_failure_swap_telemetry(active_error, record)
        return record
    if teardown_error is not None:
        _attach_failure_swap_telemetry(teardown_error, record)
        raise teardown_error
    if record["capture_status"] != "captured":
        error = RuntimeError("could not retain post-shutdown swap telemetry")
        _attach_failure_swap_telemetry(error, record)
        raise error
    try:
        _validated_worker_session_swap_telemetry(
            telemetry, "post-shutdown worker-session swap telemetry"
        )
    except BaseException as exc:
        _attach_failure_swap_telemetry(exc, record)
        raise
    return record


def _validated_dispatch_swap_deltas(
    value: Any, field: str,
) -> tuple[int, int]:
    """Recompute dispatch swap deltas from retained monotonic host counters."""
    telemetry = _as_mapping(value, field)
    samples = telemetry.get("samples")
    if (
        not isinstance(samples, list)
        or len(samples) < 2
        or type(telemetry.get("sample_count")) is not int
        or telemetry.get("sample_count") != len(samples)
    ):
        raise RuntimeError(f"{field} swap samples are incomplete")
    previous_monotonic_ns = -1
    previous_swap_in = -1
    previous_swap_out = -1
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise RuntimeError(f"{field} swap sample is malformed")
        monotonic_ns = sample.get("monotonic_ns")
        swap_in = sample.get("swap_in_bytes")
        swap_out = sample.get("swap_out_bytes")
        if (
            type(monotonic_ns) is not int
            or monotonic_ns <= previous_monotonic_ns
            or type(swap_in) is not int
            or swap_in < 0
            or swap_in < previous_swap_in
            or type(swap_out) is not int
            or swap_out < 0
            or swap_out < previous_swap_out
        ):
            raise RuntimeError(f"{field} swap counters are not monotonic")
        previous_monotonic_ns = monotonic_ns
        previous_swap_in = swap_in
        previous_swap_out = swap_out
    swap_in_delta = samples[-1]["swap_in_bytes"] - samples[0]["swap_in_bytes"]
    swap_out_delta = samples[-1]["swap_out_bytes"] - samples[0]["swap_out_bytes"]
    if (
        type(telemetry.get("swap_in_delta")) is not int
        or type(telemetry.get("swap_out_delta")) is not int
        or telemetry.get("swap_in_delta") != swap_in_delta
        or telemetry.get("swap_out_delta") != swap_out_delta
        or swap_out_delta != 0
    ):
        raise RuntimeError(f"{field} swap deltas violate the swap policy")
    return swap_in_delta, swap_out_delta


def _dispatch_runs(
    workers: Sequence[Mapping[str, Any]],
    assignments: Sequence[
        tuple[int, tuple[str, int, int, int, int, str], Path]
    ],
    *,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not assignments or len(assignments) > WORKER_COUNT:
        raise RuntimeError("dispatch must contain one or two jobs")
    slots = [slot for slot, _, _ in assignments]
    if len(set(slots)) != len(slots) or any(slot not in range(WORKER_COUNT) for slot in slots):
        raise RuntimeError("dispatch worker slots are invalid")
    release_ns = time.monotonic_ns() + 250_000_000
    expected: dict[str, tuple[int, tuple[str, int, int, int, int, str], Path]] = {}
    for slot, key, result_root in assignments:
        if key not in expected_grid():
            raise RuntimeError("dispatch key is outside the frozen grid")
        command_id = f"{label}-{slot}-{time.monotonic_ns()}"
        expected[command_id] = (slot, key, result_root.resolve())
        workers[slot]["connection"].send(
            {
                "kind": "run",
                "command_id": command_id,
                "key": _key_payload(key),
                "result_root": str(result_root.resolve()),
                "release_monotonic_ns": release_ns,
            }
        )
    messages, telemetry = hardened._await_commands(
        workers,
        set(expected),
        timeout_seconds=TIME_LIMIT_SECONDS + 900.0,
        command_slots={key: value[0] for key, value in expected.items()},
    )
    _validated_dispatch_swap_deltas(telemetry, f"{label} resource telemetry")
    reports = []
    for message in messages:
        slot, key, root = expected[message["command_id"]]
        path = root / expected_result_relative_path(*key)
        if (
            message.get("type") != "result"
            or message.get("status") != "ok"
            or message.get("slot") != slot
            or message.get("pid") != workers[slot]["process"].pid
            or message.get("start_method") != "spawn"
            or _key_tuple(message.get("key", {})) != key
            or Path(message.get("result_path", "")).resolve() != path.resolve()
            or message.get("result_count") != 1
            or message.get("child_count") != 8
            or message.get("deadline_hit") is not False
            or message.get("time_callback_hit_count") != 0
            or message.get("a10_candidate_fit_count")
            != (24 if key[-1] == "A10" else 0)
            or not path.is_file()
            or path.is_symlink()
            or message.get("result_sha256") != _sha256_file(path)
            or message.get("result_size_bytes") != path.stat().st_size
            or type(message.get("process_peak_rss_bytes")) is not int
            or message["process_peak_rss_bytes"] <= 0
            or message.get("barrier_release_monotonic_ns") != release_ns
            or type(message.get("started_monotonic_ns")) is not int
            or type(message.get("ended_monotonic_ns")) is not int
            or not (
                release_ns
                <= message["started_monotonic_ns"]
                < message["ended_monotonic_ns"]
            )
        ):
            raise RuntimeError(f"worker result identity changed: {message}")
        reports.append(dict(message))
    reports.sort(key=lambda value: value["slot"])
    starts = [value["started_monotonic_ns"] for value in reports]
    ends = [value["ended_monotonic_ns"] for value in reports]
    telemetry["barrier_release_monotonic_ns"] = release_ns
    telemetry["start_skew_ns"] = max(starts) - min(starts)
    telemetry["overlap_ns"] = max(0, min(ends) - max(starts))
    telemetry["wave_elapsed_ns"] = max(ends) - min(starts)
    physical = int(telemetry["physical_memory_bytes"])
    peak = int(telemetry["peak_combined_rss_bytes"])
    telemetry["peak_combined_rss_fraction"] = peak / physical
    if (
        telemetry.get("swap_out_delta") != 0
        or not 0.0 <= telemetry["peak_combined_rss_fraction"] < 0.8
        or telemetry["wave_elapsed_ns"] >= int(TIME_LIMIT_SECONDS * 1e9)
        or (len(reports) == 2 and (
            telemetry["start_skew_ns"] > 1_000_000_000
            or telemetry["overlap_ns"] <= 0
        ))
    ):
        raise RuntimeError("dispatch violated the resource/barrier contract")
    return reports, telemetry


def _atomic_write_json(path: Path, value: Any) -> None:
    hardened.screen.hardened._atomic_write_json(path, value)


def _artifact_metadata(path: Path, root: Path) -> dict[str, Any]:
    try:
        relative = path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise RuntimeError("artifact is not confined to the campaign directory") from exc
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("artifact must be a regular non-symlink file")
    return {
        "path": str(relative),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def collect_result_artifacts(output_dir: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for key in sorted(expected_grid(), key=lambda value: (value[1], value[3], value[5])):
        relative = expected_result_relative_path(*key)
        path = output_dir / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"required result is missing or unsafe: {relative}")
        artifacts[relative] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    observed = {
        str(path.relative_to(output_dir))
        for path in (output_dir / "experiments").rglob("results.pkl")
        if path.is_file()
    }
    if observed != set(artifacts):
        raise RuntimeError("on-disk result set differs from the frozen 90 jobs")
    return artifacts


def validate_completed_results(
    output_dir: Path,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if set(artifacts) != {
        expected_result_relative_path(*key) for key in expected_grid()
    }:
        raise RuntimeError("result artifacts do not cover the frozen grid")
    outer_rows: list[dict[str, Any]] = []
    child_rows: list[dict[str, Any]] = []
    for relative in sorted(artifacts):
        path = output_dir / relative
        metadata = artifacts[relative]
        if (
            set(metadata) != {"sha256", "size_bytes"}
            or metadata.get("sha256") != _sha256_file(path)
            or metadata.get("size_bytes") != path.stat().st_size
        ):
            raise RuntimeError(f"result artifact changed: {relative}")
        outer, children = parse_result_path(path, output_dir=output_dir)
        outer_rows.append(outer)
        child_rows.extend(children)
    if (
        {tuple(row[name] for name in OUTER_PAYLOAD_FIELDS[:6]) for row in outer_rows}
        != expected_grid()
        or {
            (*tuple(row[name] for name in OUTER_PAYLOAD_FIELDS[:6]), row["child_fold"])
            for row in child_rows
        }
        != expected_child_grid()
        or len(outer_rows) != EXPECTED_JOBS
        or len(child_rows) != EXPECTED_CHILD_FITS
    ):
        raise RuntimeError("normalized result rows do not match the frozen grids")
    candidate_count = sum(
        row["candidate_metadata"]["fitted_candidate_count"]
        for row in child_rows
        if row["arm"] == "A10"
    )
    callback_hits = sum(row["time_callback_hit"] is True for row in child_rows)
    deadline_hits = sum(row["deadline_hit"] is True for row in child_rows)
    unresolved = sum(
        row["arm"] in {"M", "C"} and row["stop_reason"] is None
        for row in child_rows
    )
    validation = {
        "result_count": len(outer_rows),
        "child_fit_count": len(child_rows),
        "a10_candidate_fit_count": candidate_count,
        "failure_count": 0,
        "imputation_count": 0,
        "deadline_hit_count": deadline_hits,
        "time_callback_hit_count": callback_hits,
        "worker_failure_count": 0,
        "recovery_mixing_count": 0,
        "swap_in_audit_evidence_retained": False,
        "preflight_swap_in_bytes": 0,
        "production_swap_in_bytes": 0,
        "swap_dispatch_count": 0,
        "swap_wave_count": 0,
        "swap_out_bytes": 0,
        "peak_combined_rss_fraction": 0.0,
        "unresolved_comparator_stop_count": unresolved,
        "resource_allocation": {
            "num_cpus": EXPECTED_CHILD_CPUS,
            "num_gpus": 0,
            "num_cpus_child": EXPECTED_CHILD_CPUS,
            "num_gpus_child": 0,
        },
    }
    return validation, outer_rows, child_rows


def build_swap_audit(
    preflight_value: Any,
    concurrency_value: Any,
    *,
    execution_mode: str,
) -> dict[str, Any]:
    """Build the bounded swap summary duplicated into safe completion JSON."""
    if execution_mode not in {"concurrent", "sequential_recovery"}:
        raise RuntimeError("swap audit execution mode is invalid")
    preflight = _as_mapping(preflight_value, "preflight swap audit source")
    concurrency = _as_mapping(concurrency_value, "production swap audit source")
    preflight_session = _as_mapping(
        preflight.get("worker_session_swap_telemetry"),
        "preflight worker-session swap telemetry",
    )
    preflight_measured = _as_mapping(
        preflight.get("measured_phase_swap_window"),
        "preflight measured swap window",
    )
    production_session = _as_mapping(
        concurrency.get("worker_session_swap_telemetry"),
        "production worker-session swap telemetry",
    )
    production_measured = _as_mapping(
        concurrency.get("measured_phase_swap_window"),
        "production measured swap window",
    )
    entries = concurrency.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("production swap audit waves are missing")
    waves = []
    for expected_index, raw_entry in enumerate(entries):
        entry = _as_mapping(raw_entry, "production swap audit wave")
        start = _exact_int(
            entry.get("swap_start_sample_index"), "swap audit wave start"
        )
        end = _exact_int(entry.get("swap_end_sample_index"), "swap audit wave end")
        waves.append(
            {
                "wave_index": _exact_int(
                    entry.get("wave_index"), "swap audit wave index"
                ),
                "dispatch_count": end - start,
                "swap_in_bytes": _exact_int(
                    entry.get("swap_in_delta"), "swap audit wave swap-in"
                ),
                "swap_out_bytes": _exact_int(
                    entry.get("swap_out_delta"), "swap audit wave swap-out"
                ),
            }
        )
        if (
            waves[-1]["wave_index"] != expected_index
            or waves[-1]["dispatch_count"]
            != (1 if execution_mode == "concurrent" else WORKER_COUNT)
            or waves[-1]["swap_in_bytes"] < 0
            or waves[-1]["swap_out_bytes"] != 0
        ):
            raise RuntimeError("production swap audit wave is invalid")
    audit = {
        "policy": SWAP_POLICY,
        "preflight": {
            "worker_lifecycle_swap_in_bytes": _exact_int(
                preflight_session.get("swap_in_delta"),
                "preflight lifecycle swap-in",
            ),
            "worker_lifecycle_swap_out_bytes": _exact_int(
                preflight_session.get("swap_out_delta"),
                "preflight lifecycle swap-out",
            ),
            "measured_phase_swap_in_bytes": _exact_int(
                preflight_measured.get("swap_in_delta"),
                "preflight measured swap-in",
            ),
            "measured_phase_swap_out_bytes": _exact_int(
                preflight_measured.get("swap_out_delta"),
                "preflight measured swap-out",
            ),
            "measured_dispatch_count": len(
                preflight_measured.get("dispatches", [])
            ),
        },
        "production": {
            "worker_lifecycle_swap_in_bytes": _exact_int(
                production_session.get("swap_in_delta"),
                "production lifecycle swap-in",
            ),
            "worker_lifecycle_swap_out_bytes": _exact_int(
                production_session.get("swap_out_delta"),
                "production lifecycle swap-out",
            ),
            "measured_phase_swap_in_bytes": _exact_int(
                production_measured.get("swap_in_delta"),
                "production measured swap-in",
            ),
            "measured_phase_swap_out_bytes": _exact_int(
                production_measured.get("swap_out_delta"),
                "production measured swap-out",
            ),
            "measured_dispatch_count": len(
                production_measured.get("dispatches", [])
            ),
            "wave_count": len(waves),
            "waves": waves,
        },
    }
    expected_dispatches = EXPECTED_WAVES * (
        1 if execution_mode == "concurrent" else WORKER_COUNT
    )
    if (
        audit["policy"] != SWAP_POLICY
        or audit["preflight"]["worker_lifecycle_swap_in_bytes"] < 0
        or audit["preflight"]["worker_lifecycle_swap_out_bytes"] != 0
        or audit["preflight"]["measured_phase_swap_in_bytes"] < 0
        or audit["preflight"]["measured_phase_swap_out_bytes"] != 0
        or audit["preflight"]["measured_dispatch_count"] != 1
        or audit["production"]["worker_lifecycle_swap_in_bytes"] < 0
        or audit["production"]["worker_lifecycle_swap_out_bytes"] != 0
        or audit["production"]["measured_phase_swap_in_bytes"] < 0
        or audit["production"]["measured_phase_swap_out_bytes"] != 0
        or audit["production"]["measured_dispatch_count"] != expected_dispatches
        or audit["production"]["wave_count"] != EXPECTED_WAVES
        or sum(wave["swap_in_bytes"] for wave in waves)
        != audit["production"]["measured_phase_swap_in_bytes"]
    ):
        raise RuntimeError("swap audit does not prove retained quality-only telemetry")
    return audit


def validate_completion_for_analysis(
    value: Any,
    *,
    manifest: Mapping[str, Any],
    outer_rows: Sequence[Mapping[str, Any]],
    child_rows: Sequence[Mapping[str, Any]],
    swap_audit: Mapping[str, Any],
) -> None:
    validation = dict(_as_mapping(value, "completion validation"))
    expected_fields = {
        "result_count",
        "child_fit_count",
        "a10_candidate_fit_count",
        "failure_count",
        "imputation_count",
        "deadline_hit_count",
        "time_callback_hit_count",
        "worker_failure_count",
        "recovery_mixing_count",
        "swap_in_audit_evidence_retained",
        "preflight_swap_in_bytes",
        "production_swap_in_bytes",
        "swap_dispatch_count",
        "swap_wave_count",
        "swap_out_bytes",
        "peak_combined_rss_fraction",
        "unresolved_comparator_stop_count",
        "resource_allocation",
    }
    unresolved = sum(
        row.get("arm") in {"M", "C"} and row.get("stop_reason") is None
        for row in child_rows
    )
    candidate_count = sum(
        row.get("candidate_metadata", {}).get("fitted_candidate_count", 0)
        for row in child_rows
        if row.get("arm") == "A10"
    )
    expected_dispatches = EXPECTED_WAVES * (
        1 if manifest.get("execution_mode") == "concurrent" else WORKER_COUNT
    )
    preflight_swap = _as_mapping(swap_audit.get("preflight"), "preflight swap audit")
    production_swap = _as_mapping(
        swap_audit.get("production"), "production swap audit"
    )
    integer_fields = {
        "result_count",
        "child_fit_count",
        "a10_candidate_fit_count",
        "failure_count",
        "imputation_count",
        "deadline_hit_count",
        "time_callback_hit_count",
        "worker_failure_count",
        "recovery_mixing_count",
        "preflight_swap_in_bytes",
        "production_swap_in_bytes",
        "swap_dispatch_count",
        "swap_wave_count",
        "swap_out_bytes",
        "unresolved_comparator_stop_count",
    }
    try:
        exact = {
            name: _exact_int(validation.get(name), f"completion {name}")
            for name in integer_fields
        }
        audit_preflight_swap_in = _exact_int(
            preflight_swap.get("worker_lifecycle_swap_in_bytes"),
            "completion preflight audit swap-in",
        )
        audit_production_swap_in = _exact_int(
            production_swap.get("worker_lifecycle_swap_in_bytes"),
            "completion production audit swap-in",
        )
        audit_dispatch_count = _exact_int(
            production_swap.get("measured_dispatch_count"),
            "completion production dispatch count",
        )
        audit_wave_count = _exact_int(
            production_swap.get("wave_count"),
            "completion production wave count",
        )
        resource = _as_mapping(
            validation.get("resource_allocation"),
            "completion resource allocation",
        )
        exact_resource = {
            name: _exact_int(resource.get(name), f"completion resource {name}")
            for name in ("num_cpus", "num_gpus", "num_cpus_child", "num_gpus_child")
        }
    except RuntimeError as exc:
        raise RuntimeError(
            "completion validation does not prove the frozen campaign"
        ) from exc
    if (
        set(validation) != expected_fields
        or exact["result_count"] != EXPECTED_JOBS
        or len(outer_rows) != EXPECTED_JOBS
        or exact["child_fit_count"] != EXPECTED_CHILD_FITS
        or len(child_rows) != EXPECTED_CHILD_FITS
        or exact["a10_candidate_fit_count"] != candidate_count
        or candidate_count != 648
        or validation.get("swap_in_audit_evidence_retained") is not True
        or exact["preflight_swap_in_bytes"] != audit_preflight_swap_in
        or exact["production_swap_in_bytes"] != audit_production_swap_in
        or exact["swap_dispatch_count"] != expected_dispatches
        or exact["swap_dispatch_count"] != audit_dispatch_count
        or exact["swap_wave_count"] != EXPECTED_WAVES
        or exact["swap_wave_count"] != audit_wave_count
        or any(
            exact[name] != 0
            for name in (
                "failure_count",
                "imputation_count",
                "deadline_hit_count",
                "time_callback_hit_count",
                "worker_failure_count",
                "recovery_mixing_count",
                "swap_out_bytes",
            )
        )
        or exact["unresolved_comparator_stop_count"] != unresolved
        or set(resource)
        != {"num_cpus", "num_gpus", "num_cpus_child", "num_gpus_child"}
        or exact_resource
        != {
            "num_cpus": 18,
            "num_gpus": 0,
            "num_cpus_child": 18,
            "num_gpus_child": 0,
        }
        or not isinstance(validation.get("peak_combined_rss_fraction"), (int, float))
        or isinstance(validation.get("peak_combined_rss_fraction"), bool)
        or not 0.0 <= float(validation["peak_combined_rss_fraction"]) < 0.8
        or manifest.get("swap_policy") != SWAP_POLICY
        or manifest.get("timing_admissible") is not False
    ):
        raise RuntimeError("completion validation does not prove the frozen campaign")


def _validate_persisted_preflight(value: Any) -> None:
    """Rebuild the synthetic concurrency proof from its persisted observations."""
    try:
        from benchmarks.tabarena_comparator_warmup import (
            validate_comparator_warmup_history,
        )
    except ModuleNotFoundError:
        from tabarena_comparator_warmup import validate_comparator_warmup_history

    preflight = dict(_as_mapping(value, "preflight report"))
    fields = {
        "schema_version",
        "kind",
        "completed_at_utc",
        "status",
        "swap_policy",
        "timing_admissible",
        "worker_ready",
        "worker_warmup",
        "ctr23_fit_count",
        "synthetic_probes",
        "start_skew_ns",
        "overlap_ns",
        "worker_restarts",
        "failure_count",
        "worker_session_swap_telemetry",
        "measured_phase_swap_window",
        "synthetic_dispatch_telemetry",
        "swap_in_bytes",
        "swap_out_bytes",
        "peak_combined_rss_fraction",
    }
    if set(preflight) != fields:
        raise RuntimeError("preflight report fields are not exact")
    ready_value = preflight.get("worker_ready")
    warmup_value = preflight.get("worker_warmup")
    probes_value = preflight.get("synthetic_probes")
    if (
        not isinstance(ready_value, list)
        or not isinstance(warmup_value, list)
        or not isinstance(probes_value, list)
        or len(ready_value) != WORKER_COUNT
        or len(warmup_value) != WORKER_COUNT
        or len(probes_value) != WORKER_COUNT
    ):
        raise RuntimeError("preflight worker evidence is incomplete")
    ready_by_slot: dict[int, Mapping[str, Any]] = {}
    for raw_ready in ready_value:
        ready = _as_mapping(raw_ready, "preflight worker readiness")
        slot = _exact_int(ready.get("slot"), "preflight ready slot")
        pid = _exact_int(ready.get("pid"), "preflight ready pid")
        if (
            set(ready)
            != {"type", "slot", "pid", "child_cpus", "start_method", "scratch_root"}
            or slot not in range(WORKER_COUNT)
            or slot in ready_by_slot
            or pid <= 0
            or ready.get("type") != "ready"
            or ready.get("child_cpus") != EXPECTED_CHILD_CPUS
            or ready.get("start_method") != "spawn"
            or not isinstance(ready.get("scratch_root"), str)
            or not ready["scratch_root"]
        ):
            raise RuntimeError("preflight worker readiness changed")
        ready_by_slot[slot] = ready
    if set(ready_by_slot) != set(range(WORKER_COUNT)):
        raise RuntimeError("preflight readiness does not cover both workers")

    warmup_by_slot: dict[int, Mapping[str, Any]] = {}
    for raw_warmup in warmup_value:
        record = _as_mapping(raw_warmup, "preflight worker warmup")
        slot = _exact_int(record.get("worker_slot"), "preflight warmup slot")
        pid = _exact_int(record.get("pid"), "preflight warmup pid")
        payload = _as_mapping(record.get("warmup"), "preflight warmup payload")
        if (
            set(record) != {"completed_at_utc", "pid", "worker_slot", "warmup"}
            or slot not in ready_by_slot
            or slot in warmup_by_slot
            or pid != ready_by_slot[slot]["pid"]
            or set(payload) != {"darkofit", "comparators"}
            or not isinstance(record.get("completed_at_utc"), str)
            or not record["completed_at_utc"]
        ):
            raise RuntimeError("preflight worker warmup changed")
        base = {
            "completed_at_utc": record["completed_at_utc"],
            "pid": pid,
        }
        hardened.screen._validate_followon_warmup_history(
            [{**base, "warmup": payload["darkofit"]}],
            expected_thread_count=EXPECTED_CHILD_CPUS,
            expected_latest_pid=pid,
        )
        validate_comparator_warmup_history(
            [{**base, "warmup": payload["comparators"]}],
            expected_thread_count=EXPECTED_CHILD_CPUS,
            expected_latest_pid=pid,
        )
        warmup_by_slot[slot] = record
    if set(warmup_by_slot) != set(range(WORKER_COUNT)):
        raise RuntimeError("preflight warmup does not cover both workers")

    probe_by_slot: dict[int, Mapping[str, Any]] = {}
    for raw_probe in probes_value:
        probe = _as_mapping(raw_probe, "preflight synthetic probe")
        slot = _exact_int(probe.get("worker_slot"), "preflight probe slot")
        pid = _exact_int(probe.get("pid"), "preflight probe pid")
        release = _exact_int(
            probe.get("barrier_release_monotonic_ns"),
            "preflight barrier release",
        )
        started = _exact_int(
            probe.get("started_monotonic_ns"), "preflight start timestamp"
        )
        ended = _exact_int(
            probe.get("ended_monotonic_ns"), "preflight end timestamp"
        )
        expected_digest = None
        if slot in warmup_by_slot:
            expected_digest = hashlib.sha256(
                _canonical_json(
                    _synthetic_behavior_projection(
                        warmup_by_slot[slot]["warmup"]
                    )
                )
            ).hexdigest()
        if (
            set(probe)
            != {
                "worker_slot",
                "pid",
                "behavior_sha256",
                "barrier_release_monotonic_ns",
                "started_monotonic_ns",
                "ended_monotonic_ns",
            }
            or slot not in ready_by_slot
            or slot in probe_by_slot
            or pid != ready_by_slot[slot]["pid"]
            or probe.get("behavior_sha256") != expected_digest
            or release <= 0
            or not release <= started < ended
        ):
            raise RuntimeError("preflight synthetic probe changed")
        probe_by_slot[slot] = probe
    if set(probe_by_slot) != set(range(WORKER_COUNT)):
        raise RuntimeError("preflight probes do not cover both workers")
    probes = [probe_by_slot[slot] for slot in range(WORKER_COUNT)]
    releases = [int(probe["barrier_release_monotonic_ns"]) for probe in probes]
    starts = [int(probe["started_monotonic_ns"]) for probe in probes]
    ends = [int(probe["ended_monotonic_ns"]) for probe in probes]
    skew = max(starts) - min(starts)
    overlap = max(0, min(ends) - max(starts))
    if (
        len(set(releases)) != 1
        or _exact_int(preflight.get("start_skew_ns"), "preflight start skew")
        != skew
        or _exact_int(preflight.get("overlap_ns"), "preflight overlap")
        != overlap
        or skew > 1_000_000_000
        or overlap <= 0
    ):
        raise RuntimeError("preflight concurrency evidence changed")
    session_swap = _validated_worker_session_swap_telemetry(
        preflight.get("worker_session_swap_telemetry"),
        "preflight worker-session swap telemetry",
    )
    measured_swap = preflight.get("measured_phase_swap_window")
    dispatch_telemetry = _as_mapping(
        preflight.get("synthetic_dispatch_telemetry"),
        "preflight synthetic dispatch telemetry",
    )
    _validated_dispatch_swap_deltas(
        dispatch_telemetry, "preflight synthetic dispatch telemetry"
    )
    if (
        session_swap.get("sample_count") != 6
        or _exact_int(preflight.get("swap_in_bytes"), "preflight swap-in bytes")
        != session_swap.get("swap_in_delta")
        or _exact_int(preflight.get("swap_out_bytes"), "preflight swap-out bytes")
        != session_swap.get("swap_out_delta")
        or dispatch_telemetry.get("barrier_release_monotonic_ns") != releases[0]
        or not hardened._measured_swap_window_valid(
            measured_swap,
            session_swap,
            swap_policy=SWAP_POLICY,
            expected_dispatches=[
                ("preflight-synthetic-probe", dispatch_telemetry, probes)
            ],
            require_shutdown_sample=True,
            expected_start_sample_index=3,
            require_exact_shutdown_suffix=True,
        )
    ):
        raise RuntimeError("preflight swap evidence changed")


def validate_operational_artifacts_for_analysis(
    value: Any,
    *,
    manifest: Mapping[str, Any],
    attestation: Mapping[str, Any],
    output_dir: Path,
) -> None:
    operational = dict(_as_mapping(value, "operational artifacts"))
    if set(operational) != {
        "preflight_report_artifact",
        "concurrency_history_artifact",
        "warmup_history_artifact",
    }:
        raise RuntimeError("operational artifact set is not exact")
    preflight = _as_mapping(
        operational["preflight_report_artifact"], "preflight report"
    )
    concurrency = _as_mapping(
        operational["concurrency_history_artifact"], "concurrency history"
    )
    warmup = _as_mapping(operational["warmup_history_artifact"], "warmup history")
    _validate_persisted_preflight(preflight)
    if set(concurrency) != {
        "schema_version",
        "kind",
        "execution_mode",
        "swap_policy",
        "timing_admissible",
        "wave_count",
        "entries",
        "failure_count",
        "worker_restart_count",
        "recovery_mixing_count",
        "worker_session_swap_telemetry",
        "measured_phase_swap_window",
        "swap_dispatch_count",
        "swap_in_bytes",
        "swap_out_bytes",
        "peak_combined_rss_fraction",
    }:
        raise RuntimeError("concurrency history fields are not exact")
    if set(warmup) != {
        "schema_version",
        "kind",
        "execution_mode",
        "worker_ready",
        "worker_warmup",
    }:
        raise RuntimeError("production warmup fields are not exact")
    mode = manifest.get("execution_mode")
    try:
        preflight_schema = _exact_int(
            preflight.get("schema_version"), "preflight schema version"
        )
        preflight_failures = _exact_int(
            preflight.get("failure_count"), "preflight failure count"
        )
        preflight_fits = _exact_int(
            preflight.get("ctr23_fit_count"), "preflight CTR23 fit count"
        )
        preflight_swap_out = _exact_int(
            preflight.get("swap_out_bytes"), "preflight swap-out bytes"
        )
        preflight_peak_value = preflight.get("peak_combined_rss_fraction")
        if isinstance(preflight_peak_value, bool) or not isinstance(
            preflight_peak_value, (int, float)
        ):
            raise RuntimeError("preflight peak RSS fraction must be numeric")
        preflight_peak = _finite_nonnegative(
            preflight_peak_value, "preflight peak RSS fraction"
        )
        concurrency_schema = _exact_int(
            concurrency.get("schema_version"), "concurrency schema version"
        )
        concurrency_waves = _exact_int(
            concurrency.get("wave_count"), "concurrency wave count"
        )
        concurrency_failures = _exact_int(
            concurrency.get("failure_count"), "concurrency failure count"
        )
        worker_restarts = _exact_int(
            concurrency.get("worker_restart_count"), "worker restart count"
        )
        recovery_mixing = _exact_int(
            concurrency.get("recovery_mixing_count"), "recovery mixing count"
        )
        concurrency_swap_out = _exact_int(
            concurrency.get("swap_out_bytes"), "concurrency swap-out bytes"
        )
        concurrency_peak_value = concurrency.get("peak_combined_rss_fraction")
        if isinstance(concurrency_peak_value, bool) or not isinstance(
            concurrency_peak_value, (int, float)
        ):
            raise RuntimeError("concurrency peak RSS fraction must be numeric")
        concurrency_peak = _finite_nonnegative(
            concurrency_peak_value, "concurrency peak RSS fraction"
        )
        warmup_schema = _exact_int(
            warmup.get("schema_version"), "warmup schema version"
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "operational artifacts do not prove the campaign"
        ) from exc
    if (
        preflight_schema != HARNESS_SCHEMA_VERSION
        or preflight.get("kind") != CAMPAIGN_KIND + "_preflight"
        or preflight.get("status") != "passed"
        or preflight.get("swap_policy") != SWAP_POLICY
        or preflight.get("timing_admissible") is not False
        or preflight.get("worker_restarts") is not False
        or preflight_failures != 0
        or preflight_fits != 0
        or len(preflight.get("synthetic_probes", [])) != WORKER_COUNT
        or preflight_swap_out != 0
        or not preflight_peak < 0.8
        or concurrency_schema != HARNESS_SCHEMA_VERSION
        or concurrency.get("kind") != CAMPAIGN_KIND + "_concurrency_history"
        or concurrency.get("execution_mode") != mode
        or concurrency.get("swap_policy") != SWAP_POLICY
        or concurrency.get("timing_admissible") is not False
        or concurrency_waves != EXPECTED_WAVES
        or concurrency_failures != 0
        or worker_restarts != 0
        or recovery_mixing != 0
        or concurrency_swap_out != 0
        or not concurrency_peak < 0.8
        or warmup_schema != HARNESS_SCHEMA_VERSION
        or warmup.get("kind") != CAMPAIGN_KIND + "_warmup_history"
        or warmup.get("execution_mode") != mode
        or len(warmup.get("worker_ready", [])) != WORKER_COUNT
        or len(warmup.get("worker_warmup", [])) != WORKER_COUNT
        or manifest.get("output_dir") != str(output_dir)
        or attestation.get("execution_mode") != mode
    ):
        raise RuntimeError("operational artifacts do not prove the campaign")
    recovery = manifest.get("sequential_recovery")
    if mode == "concurrent":
        if recovery is not None:
            raise RuntimeError("concurrent campaign unexpectedly carries recovery state")
    else:
        validated_recovery = validate_sequential_recovery_record_shape(recovery)
        if validated_recovery["source_output_dir"] == str(output_dir.resolve()):
            raise RuntimeError("sequential recovery mixed its source and destination")
    entries = concurrency.get("entries")
    if not isinstance(entries, list) or len(entries) != EXPECTED_WAVES:
        raise RuntimeError("concurrency history does not contain 45 waves")
    session_swap = _validated_worker_session_swap_telemetry(
        concurrency.get("worker_session_swap_telemetry"),
        "production worker-session swap telemetry",
    )
    measured_swap = concurrency.get("measured_phase_swap_window")
    expected_dispatch_count = EXPECTED_WAVES * (
        1 if mode == "concurrent" else WORKER_COUNT
    )
    if (
        _exact_int(
            concurrency.get("swap_dispatch_count"),
            "production swap dispatch count",
        )
        != expected_dispatch_count
        or _exact_int(
            concurrency.get("swap_in_bytes"), "production swap-in bytes"
        )
        != session_swap.get("swap_in_delta")
        or _exact_int(
            concurrency.get("swap_out_bytes"), "production swap-out bytes"
        )
        != session_swap.get("swap_out_delta")
        or session_swap.get("sample_count") != expected_dispatch_count + 4
        or not hardened._measured_swap_window_valid(
            measured_swap,
            session_swap,
            swap_policy=SWAP_POLICY,
            require_shutdown_sample=True,
            expected_start_sample_index=2,
            require_exact_shutdown_suffix=True,
        )
        or not isinstance(measured_swap, Mapping)
        or measured_swap.get("sample_count") != expected_dispatch_count + 1
        or len(measured_swap.get("dispatches", [])) != expected_dispatch_count
    ):
        raise RuntimeError("production swap evidence is incomplete")
    ready_by_slot: dict[int, Mapping[str, Any]] = {}
    scratch_roots = set()
    for item in warmup["worker_ready"]:
        if not isinstance(item, Mapping):
            raise RuntimeError("warmup worker readiness is malformed")
        slot = _exact_int(item.get("slot"), "warmup worker slot")
        pid = _exact_int(item.get("pid"), "warmup worker pid")
        expected_scratch = str(
            output_dir.resolve() / "worker_scratch" / f"worker-{slot}"
        )
        scratch = item.get("scratch_root")
        if (
            set(item)
            != {"type", "slot", "pid", "child_cpus", "start_method", "scratch_root"}
            or slot not in range(WORKER_COUNT)
            or slot in ready_by_slot
            or pid <= 0
            or item.get("type") != "ready"
            or item.get("child_cpus") != EXPECTED_CHILD_CPUS
            or item.get("start_method") != "spawn"
            or scratch != expected_scratch
        ):
            raise RuntimeError("warmup worker readiness changed")
        ready_by_slot[slot] = item
        scratch_roots.add(scratch)
    if len(scratch_roots) != WORKER_COUNT:
        raise RuntimeError("worker scratch roots overlap")
    try:
        from benchmarks.tabarena_comparator_warmup import (
            validate_comparator_warmup_history,
        )
    except ModuleNotFoundError:
        from tabarena_comparator_warmup import validate_comparator_warmup_history
    for record in warmup["worker_warmup"]:
        if not isinstance(record, Mapping):
            raise RuntimeError("worker warmup record is malformed")
        slot = _exact_int(record.get("worker_slot"), "worker warmup slot")
        pid = _exact_int(record.get("pid"), "worker warmup pid")
        payload = _as_mapping(record.get("warmup"), "worker warmup payload")
        if (
            set(record) != {"completed_at_utc", "pid", "worker_slot", "warmup"}
            or not isinstance(record.get("completed_at_utc"), str)
            or not record["completed_at_utc"]
            or slot not in ready_by_slot
            or pid != ready_by_slot[slot]["pid"]
            or set(payload) != {"darkofit", "comparators"}
        ):
            raise RuntimeError("worker warmup is not bound to its process")
        base = {
            "completed_at_utc": record.get("completed_at_utc"),
            "pid": pid,
        }
        hardened.screen._validate_followon_warmup_history(
            [{**base, "warmup": payload["darkofit"]}],
            expected_thread_count=EXPECTED_CHILD_CPUS,
            expected_latest_pid=pid,
        )
        validate_comparator_warmup_history(
            [{**base, "warmup": payload["comparators"]}],
            expected_thread_count=EXPECTED_CHILD_CPUS,
            expected_latest_pid=pid,
        )
    warmed_slots = [
        _exact_int(record.get("worker_slot"), "worker warmup slot")
        for record in warmup["worker_warmup"]
    ]
    if sorted(warmed_slots) != list(range(WORKER_COUNT)):
        raise RuntimeError("production warmup does not cover each worker exactly once")
    seen_command_ids: set[str] = set()
    previous_wave_end_ns: int | None = None
    previous_swap_sample_index = int(measured_swap["start_sample_index"])
    measured_dispatch_offset = 0
    wave_swap_in_total = 0
    for index, (entry, expected_wave) in enumerate(zip(entries, expected_schedule())):
        wave_timing = _validate_persisted_wave(
            entry,
            expected_wave=expected_wave,
            execution_mode=str(mode),
            ready_by_slot=ready_by_slot,
            result_artifacts=attestation.get("result_artifacts"),
            output_dir=output_dir,
            seen_command_ids=seen_command_ids,
        )
        dispatch_width = 1 if mode == "concurrent" else WORKER_COUNT
        swap_start = wave_timing["swap_start_sample_index"]
        swap_end = wave_timing["swap_end_sample_index"]
        samples = session_swap["samples"]
        expected_swap_in = (
            samples[swap_end]["swap_in_bytes"]
            - samples[swap_start]["swap_in_bytes"]
            if 0 <= swap_start < swap_end < len(samples)
            else None
        )
        expected_swap_out = (
            samples[swap_end]["swap_out_bytes"]
            - samples[swap_start]["swap_out_bytes"]
            if 0 <= swap_start < swap_end < len(samples)
            else None
        )
        dispatches = measured_swap["dispatches"][
            measured_dispatch_offset : measured_dispatch_offset + dispatch_width
        ]
        expected_labels = (
            [f"production-wave-{index}"]
            if mode == "concurrent"
            else [
                f"recovery-wave-{index}-job-{order}"
                for order in range(WORKER_COUNT)
            ]
        )
        reports_by_slot = {
            _exact_int(report.get("slot"), "operational report slot"): report
            for report in entry["reports"]
        }
        for order, (dispatch, expected_label) in enumerate(
            zip(dispatches, expected_labels)
        ):
            relevant_reports = (
                list(reports_by_slot.values())
                if mode == "concurrent"
                else [
                    reports_by_slot[
                        _exact_int(
                            expected_wave["jobs"][order]["worker_slot"],
                            "scheduled worker slot",
                        )
                    ]
                ]
            )
            if (
                dispatch.get("label") != expected_label
                or dispatch.get("barrier_release_monotonic_ns")
                != relevant_reports[0].get("barrier_release_monotonic_ns")
                or any(
                    report.get("barrier_release_monotonic_ns")
                    != relevant_reports[0].get("barrier_release_monotonic_ns")
                    for report in relevant_reports
                )
                or dispatch.get("max_report_end_monotonic_ns")
                != max(int(report["ended_monotonic_ns"]) for report in relevant_reports)
            ):
                raise RuntimeError(
                    f"operational wave {index} swap dispatch binding changed"
                )
        if (
            len(dispatches) != dispatch_width
            or swap_start != previous_swap_sample_index
            or swap_end - swap_start != dispatch_width
            or wave_timing["swap_in_delta"] != expected_swap_in
            or wave_timing["swap_out_delta"] != expected_swap_out
        ):
            raise RuntimeError(f"operational wave {index} swap range changed")
        previous_swap_sample_index = swap_end
        measured_dispatch_offset += dispatch_width
        wave_swap_in_total += wave_timing["swap_in_delta"]
        if (
            previous_wave_end_ns is not None
            and wave_timing["first_release_ns"] <= previous_wave_end_ns
        ):
            raise RuntimeError(f"operational wave {index} precedes its release")
        previous_wave_end_ns = wave_timing["last_end_ns"]
    if (
        previous_swap_sample_index != measured_swap["end_sample_index"]
        or measured_dispatch_offset != expected_dispatch_count
        or wave_swap_in_total != measured_swap["swap_in_delta"]
    ):
        raise RuntimeError("production swap waves do not cover the measured window")
    rebuilt_swap_audit = build_swap_audit(
        preflight, concurrency, execution_mode=str(mode)
    )
    raw_verification = _as_mapping(
        attestation.get("raw_result_verification"),
        "completion raw-result verification",
    )
    if (
        _canonical_json(attestation.get("swap_audit"))
        != _canonical_json(rebuilt_swap_audit)
        or _canonical_json(attestation.get("analysis_boundary"))
        != _canonical_json(analysis_boundary())
        or set(raw_verification)
        != {"authority", "count", "method", "analyzer_access"}
        or raw_verification.get("authority") != "runner"
        or _exact_int(raw_verification.get("count"), "raw result count")
        != EXPECTED_JOBS
        or raw_verification.get("method")
        != "sha256_size_and_safe_extraction"
        or raw_verification.get("analyzer_access") != "forbidden"
    ):
        raise RuntimeError("completion attestation detaches operational evidence")


def _validate_persisted_wave(
    value: Any,
    *,
    expected_wave: Mapping[str, Any],
    execution_mode: str,
    ready_by_slot: Mapping[int, Mapping[str, Any]],
    result_artifacts: Any,
    output_dir: Path,
    seen_command_ids: set[str] | None = None,
) -> dict[str, int]:
    """Authenticate one persisted wave from its raw worker timestamps."""
    entry = dict(_as_mapping(value, "operational wave"))
    entry_fields = {
        "wave_index",
        "jobs",
        "reports",
        "swap_start_sample_index",
        "swap_end_sample_index",
        "swap_in_delta",
        "swap_out_delta",
        "peak_combined_rss_fraction",
        "start_skew_ns",
        "overlap_ns",
        "wave_elapsed_ns",
    }
    index = _exact_int(entry.get("wave_index"), "operational wave index")
    swap_start_sample_index = _exact_int(
        entry.get("swap_start_sample_index"), "operational swap start sample"
    )
    swap_end_sample_index = _exact_int(
        entry.get("swap_end_sample_index"), "operational swap end sample"
    )
    swap_in_delta = _exact_int(
        entry.get("swap_in_delta"), "operational swap-in delta"
    )
    swap_out_delta = _exact_int(
        entry.get("swap_out_delta"), "operational swap-out delta"
    )
    if (
        set(entry) != entry_fields
        or index != expected_wave.get("wave_index")
        or entry.get("jobs") != expected_wave.get("jobs")
        or swap_start_sample_index < 0
        or swap_end_sample_index <= swap_start_sample_index
        or swap_in_delta < 0
        or swap_out_delta != 0
        or not 0.0
        <= _finite_nonnegative(
            entry.get("peak_combined_rss_fraction"),
            "operational peak combined RSS fraction",
        )
        < 0.8
    ):
        raise RuntimeError(f"operational wave {index} is incomplete")
    reports_value = entry.get("reports")
    if not isinstance(reports_value, list) or len(reports_value) != WORKER_COUNT:
        raise RuntimeError(f"operational wave {index} reports are incomplete")
    artifacts = _as_mapping(result_artifacts, "attested result artifacts")
    expected_jobs = expected_wave["jobs"]
    expected_by_key = {
        _key_tuple(item["key"]): (
            _exact_int(item["worker_slot"], "scheduled worker slot"),
            order,
        )
        for order, item in enumerate(expected_jobs)
    }
    report_fields = {
        "type",
        "command_id",
        "status",
        "slot",
        "pid",
        "key",
        "result_root",
        "result_path",
        "result_count",
        "child_count",
        "deadline_hit",
        "time_callback_hit_count",
        "a10_candidate_fit_count",
        "behavior_sha256",
        "result_sha256",
        "result_size_bytes",
        "process_peak_rss_bytes",
        "barrier_release_monotonic_ns",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "start_method",
    }
    reports_by_key: dict[
        tuple[str, int, int, int, int, str], Mapping[str, Any]
    ] = {}
    ordered_reports: list[Mapping[str, Any] | None] = [None] * WORKER_COUNT
    campaign_root_path = output_dir.resolve()
    campaign_root = str(campaign_root_path)
    local_command_ids: set[str] = set()
    for raw_report in reports_value:
        report = _as_mapping(raw_report, f"operational wave {index} report")
        if set(report) != report_fields:
            raise RuntimeError(f"operational wave {index} report fields changed")
        key = _key_tuple(report.get("key"))
        slot = _exact_int(report.get("slot"), "operational report slot")
        expected = expected_by_key.get(key)
        if expected is None or expected[0] != slot or slot not in ready_by_slot:
            raise RuntimeError(f"operational wave {index} report identity changed")
        order = expected[1]
        relative = expected_result_relative_path(*key)
        metadata = _as_mapping(
            artifacts.get(relative), f"attested result artifact {relative}"
        )
        result_size = _exact_int(
            report.get("result_size_bytes"), "operational result size"
        )
        process_rss = _exact_int(
            report.get("process_peak_rss_bytes"), "operational process RSS"
        )
        release = _exact_int(
            report.get("barrier_release_monotonic_ns"),
            "operational barrier release",
        )
        started = _exact_int(
            report.get("started_monotonic_ns"), "operational start timestamp"
        )
        ended = _exact_int(
            report.get("ended_monotonic_ns"), "operational end timestamp"
        )
        pid = _exact_int(report.get("pid"), "operational report pid")
        result_count = _exact_int(
            report.get("result_count"), "operational result count"
        )
        child_count = _exact_int(
            report.get("child_count"), "operational child count"
        )
        callback_hits = _exact_int(
            report.get("time_callback_hit_count"),
            "operational callback hit count",
        )
        candidate_count = _exact_int(
            report.get("a10_candidate_fit_count"),
            "operational A10 candidate count",
        )
        command_id = report.get("command_id")
        if execution_mode == "concurrent":
            command_prefix = f"production-wave-{index}-{slot}-"
        elif execution_mode == "sequential_recovery":
            command_prefix = f"recovery-wave-{index}-job-{order}-{slot}-"
        else:
            raise RuntimeError("operational execution mode changed")
        command_nonce = (
            command_id[len(command_prefix) :]
            if isinstance(command_id, str) and command_id.startswith(command_prefix)
            else ""
        )
        if (
            not command_nonce.isdigit()
            or int(command_nonce) <= 0
            or command_id in local_command_ids
            or (seen_command_ids is not None and command_id in seen_command_ids)
        ):
            raise RuntimeError(f"operational wave {index} command identity changed")
        local_command_ids.add(command_id)
        digest = report.get("behavior_sha256")
        if (
            report.get("type") != "result"
            or report.get("status") != "ok"
            or report.get("start_method") != "spawn"
            or pid != ready_by_slot[slot].get("pid")
            or result_count != 1
            or child_count != 8
            or report.get("deadline_hit") is not False
            or callback_hits != 0
            or candidate_count != (24 if key[-1] == "A10" else 0)
            or report.get("result_root") != campaign_root
            or report.get("result_path")
            != str(campaign_root_path / relative)
            or set(metadata) != {"sha256", "size_bytes"}
            or report.get("result_sha256") != metadata.get("sha256")
            or result_size != metadata.get("size_bytes")
            or process_rss <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or release <= 0
            or not release <= started < ended
            or ended - started >= int(TIME_LIMIT_SECONDS * 1e9)
        ):
            raise RuntimeError(f"operational wave {index} report changed")
        if key in reports_by_key or ordered_reports[order] is not None:
            raise RuntimeError(f"operational wave {index} report is duplicated")
        reports_by_key[key] = report
        ordered_reports[order] = report
    if set(reports_by_key) != set(expected_by_key) or any(
        report is None for report in ordered_reports
    ):
        raise RuntimeError(f"operational wave {index} report grid changed")
    if seen_command_ids is not None:
        seen_command_ids.update(local_command_ids)

    reports = [report for report in ordered_reports if report is not None]
    releases = [int(report["barrier_release_monotonic_ns"]) for report in reports]
    starts = [int(report["started_monotonic_ns"]) for report in reports]
    ends = [int(report["ended_monotonic_ns"]) for report in reports]
    recomputed_skew = max(starts) - min(starts)
    recomputed_overlap = max(0, min(ends) - max(starts))
    recomputed_elapsed = max(ends) - min(starts)
    persisted_skew = _exact_int(
        entry.get("start_skew_ns"), "operational start skew"
    )
    persisted_overlap = _exact_int(
        entry.get("overlap_ns"), "operational overlap"
    )
    persisted_elapsed = _exact_int(
        entry.get("wave_elapsed_ns"), "operational wave elapsed"
    )
    if persisted_elapsed != recomputed_elapsed or recomputed_elapsed <= 0:
        raise RuntimeError(f"operational wave {index} elapsed time changed")
    if execution_mode == "concurrent":
        if (
            len(set(releases)) != 1
            or persisted_skew != recomputed_skew
            or persisted_overlap != recomputed_overlap
            or recomputed_skew > 1_000_000_000
            or recomputed_overlap <= 0
            or recomputed_elapsed >= int(TIME_LIMIT_SECONDS * 1e9)
        ):
            raise RuntimeError(f"operational wave {index} concurrency changed")
    elif (
        persisted_skew != recomputed_skew
        or persisted_overlap != recomputed_overlap
        or recomputed_overlap != 0
        or not ends[0] <= releases[1] <= starts[1]
    ):
        raise RuntimeError(f"operational wave {index} sequential ordering changed")
    return {
        "first_release_ns": min(releases),
        "last_end_ns": max(ends),
        "swap_start_sample_index": swap_start_sample_index,
        "swap_end_sample_index": swap_end_sample_index,
        "swap_in_delta": swap_in_delta,
        "swap_out_delta": swap_out_delta,
    }


def run_preflight(output_dir: Path) -> dict[str, Any]:
    failure_context: dict[str, Any] = {}
    try:
        return _run_preflight_impl(
            output_dir, failure_context=failure_context
        )
    except BaseException as exc:
        retained = failure_context.get("failure_swap_telemetry")
        if isinstance(retained, Mapping):
            _attach_failure_swap_telemetry(exc, retained)
        raise


def _run_preflight_impl(
    output_dir: Path, *, failure_context: dict[str, Any]
) -> dict[str, Any]:
    """Prove synthetic two-process safety without spending a CTR23 coordinate."""
    scratch = output_dir / ".preflight-nonreusable"
    if scratch.exists():
        raise RuntimeError("preflight scratch namespace already exists")
    scratch.mkdir(mode=0o700)
    os.chmod(scratch, 0o700)
    workers: list[dict[str, Any]] = []
    ready: list[dict[str, Any]] = []
    warmup: list[dict[str, Any]] = []
    synthetic_probes: list[dict[str, Any]] = []
    all_telemetry: list[Mapping[str, Any]] = []
    session_swap = _new_worker_session_swap_telemetry()
    measured_swap: dict[str, Any] | None = None
    synthetic_dispatch_telemetry: dict[str, Any] | None = None
    error: BaseException | None = None
    worker_start_completed = False
    startup_cleanup_confirmed = False
    try:
        workers, ready = _start_workers(scratch / "worker_scratch")
        worker_start_completed = True
        _checkpoint_worker_session_swap_telemetry(session_swap)
        warmup = _warm_workers(workers)
        _checkpoint_worker_session_swap_telemetry(session_swap)
        measured_swap = hardened._start_measured_swap_window(
            session_swap, swap_policy=SWAP_POLICY
        )
        release_ns = time.monotonic_ns() + 250_000_000
        commands = {}
        for worker in workers:
            command_id = f"synthetic-probe-{worker['slot']}-{time.monotonic_ns()}"
            commands[command_id] = worker["slot"]
            worker["connection"].send(
                {
                    "kind": "synthetic_probe",
                    "command_id": command_id,
                    "release_monotonic_ns": release_ns,
                }
            )
        messages, telemetry = hardened._await_commands(
            workers,
            set(commands),
            timeout_seconds=1_800.0,
            command_slots=commands,
        )
        telemetry["barrier_release_monotonic_ns"] = release_ns
        _validated_dispatch_swap_deltas(
            telemetry, "preflight synthetic dispatch telemetry"
        )
        all_telemetry.append(telemetry)
        serial_digests = {
            int(record["worker_slot"]): hashlib.sha256(
                _canonical_json(
                    _synthetic_behavior_projection(record["warmup"])
                )
            ).hexdigest()
            for record in warmup
        }
        messages.sort(key=lambda item: item["slot"])
        starts = [int(item["started_monotonic_ns"]) for item in messages]
        ends = [int(item["ended_monotonic_ns"]) for item in messages]
        overlap_ns = max(0, min(ends) - max(starts))
        start_skew_ns = max(starts) - min(starts)
        for message in messages:
            slot = int(message["slot"])
            if (
                message.get("type") != "synthetic_probe"
                or message.get("pid") != workers[slot]["process"].pid
                or message.get("behavior_sha256") != serial_digests[slot]
                or message.get("barrier_release_monotonic_ns") != release_ns
                or type(message.get("started_monotonic_ns")) is not int
                or type(message.get("ended_monotonic_ns")) is not int
                or not (
                    release_ns
                    <= message["started_monotonic_ns"]
                    < message["ended_monotonic_ns"]
                )
                or start_skew_ns > 1_000_000_000
                or overlap_ns <= 0
            ):
                raise RuntimeError("synthetic concurrent preflight changed behavior")
            synthetic_probes.append(
                {
                    "worker_slot": slot,
                    "pid": message["pid"],
                    "behavior_sha256": message["behavior_sha256"],
                    "barrier_release_monotonic_ns": message[
                        "barrier_release_monotonic_ns"
                    ],
                    "started_monotonic_ns": message["started_monotonic_ns"],
                    "ended_monotonic_ns": message["ended_monotonic_ns"],
                }
            )
        hardened._checkpoint_measured_swap_window(
            session_swap,
            measured_swap,
            label="preflight-synthetic-probe",
            resource_telemetry=telemetry,
            reports=messages,
            swap_policy=SWAP_POLICY,
        )
        synthetic_dispatch_telemetry = dict(telemetry)
    except BaseException as exc:
        error = exc
        startup_cleanup_confirmed = (
            getattr(
                exc,
                _WORKER_STARTUP_CLEANUP_CONFIRMED_ATTRIBUTE,
                False,
            )
            is True
        )
        raise
    finally:
        final_record: dict[str, Any] | None = None
        finalization_error: BaseException | None = None
        try:
            final_record = _finalize_worker_session(
                workers,
                session_swap,
                active_error=error,
                worker_start_completed=worker_start_completed,
                startup_cleanup_confirmed=startup_cleanup_confirmed,
            )
        except BaseException as exc:
            finalization_error = exc
        cleanup_error: BaseException | None = None
        try:
            shutil.rmtree(scratch, ignore_errors=False)
        except BaseException as exc:
            cleanup_error = exc
        if final_record is not None:
            failure_context["failure_swap_telemetry"] = final_record
        if error is not None:
            if cleanup_error is not None:
                _add_error_note(
                    error,
                    _failure_diagnostic(cleanup_error, "scratch cleanup failed"),
                )
        elif finalization_error is not None:
            if cleanup_error is not None:
                _add_error_note(
                    finalization_error,
                    _failure_diagnostic(cleanup_error, "scratch cleanup failed")
                )
            raise finalization_error
        elif cleanup_error is not None:
            if final_record is not None:
                _attach_failure_swap_telemetry(cleanup_error, final_record)
            raise cleanup_error
    session_swap = _validated_worker_session_swap_telemetry(
        session_swap, "preflight worker-session swap telemetry"
    )
    if (
        measured_swap is None
        or synthetic_dispatch_telemetry is None
        or not hardened._measured_swap_window_valid(
            measured_swap,
            session_swap,
            swap_policy=SWAP_POLICY,
            expected_dispatches=[
                (
                    "preflight-synthetic-probe",
                    synthetic_dispatch_telemetry,
                    messages,
                )
            ],
            require_shutdown_sample=True,
            expected_start_sample_index=3,
            require_exact_shutdown_suffix=True,
        )
    ):
        raise RuntimeError("preflight swap telemetry is incomplete")
    peak = max(
        (
            int(item["peak_combined_rss_bytes"])
            / int(item["physical_memory_bytes"])
            for item in all_telemetry
        ),
        default=0.0,
    )
    if messages and all_telemetry:
        peak = max(
            peak,
            sum(int(item["process_peak_rss_bytes"]) for item in messages)
            / int(all_telemetry[-1]["physical_memory_bytes"]),
        )
    swap_in = int(session_swap["swap_in_delta"])
    swap_out = int(session_swap["swap_out_delta"])
    report = {
        "schema_version": HARNESS_SCHEMA_VERSION,
        "kind": CAMPAIGN_KIND + "_preflight",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "swap_policy": SWAP_POLICY,
        "timing_admissible": False,
        "worker_ready": ready,
        "worker_warmup": warmup,
        "ctr23_fit_count": 0,
        "synthetic_probes": synthetic_probes,
        "start_skew_ns": start_skew_ns,
        "overlap_ns": overlap_ns,
        "worker_restarts": False,
        "failure_count": 0,
        "worker_session_swap_telemetry": session_swap,
        "measured_phase_swap_window": measured_swap,
        "synthetic_dispatch_telemetry": synthetic_dispatch_telemetry,
        "swap_in_bytes": swap_in,
        "swap_out_bytes": swap_out,
        "peak_combined_rss_fraction": peak,
    }
    if swap_out != 0 or peak >= 0.8:
        raise RuntimeError("preflight violated the resource contract")
    _atomic_write_json(output_dir / PREFLIGHT_REPORT_FILENAME, report)
    return report


def _validate_manifest_static(
    value: Any, *, output_dir: Path, execution_mode: str
) -> dict[str, Any]:
    manifest = dict(_as_mapping(value, "run manifest"))
    fields = {
        "schema_version",
        "kind",
        "created_at_utc",
        "output_dir",
        "protocol_sha256",
        "frozen_protocol_sha256",
        "coordinate_manifest_sha256",
        "schedule_sha256",
        "schedule",
        "expected_jobs",
        "expected_child_fits",
        "time_limit_seconds",
        "resolved_child_num_cpus",
        "execution_mode",
        "swap_policy",
        "timing_admissible",
        "source_freeze",
        "source",
        "runtime",
        "sequential_recovery",
    }
    schedule = manifest.get("schedule")
    if not isinstance(schedule, list):
        raise RuntimeError("run manifest schedule must be a list")
    validate_schedule(schedule)
    if _canonical_json(schedule) != _canonical_json(expected_schedule()):
        raise RuntimeError("run manifest schedule is not canonical")
    if (
        set(manifest) != fields
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != CAMPAIGN_KIND
        or not isinstance(manifest.get("created_at_utc"), str)
        or not manifest.get("created_at_utc")
        or not isinstance(manifest.get("output_dir"), str)
        or manifest.get("output_dir") != str(output_dir)
        or manifest.get("protocol_sha256") != protocol_sha256()
        or manifest.get("frozen_protocol_sha256") != frozen_protocol_sha256()
        or manifest.get("coordinate_manifest_sha256")
        != COORDINATE_MANIFEST_SHA256
        or manifest.get("schedule_sha256") != schedule_sha256()
        or type(manifest.get("expected_jobs")) is not int
        or manifest.get("expected_jobs") != EXPECTED_JOBS
        or type(manifest.get("expected_child_fits")) is not int
        or manifest.get("expected_child_fits") != EXPECTED_CHILD_FITS
        or type(manifest.get("time_limit_seconds")) is not float
        or manifest.get("time_limit_seconds") != TIME_LIMIT_SECONDS
        or type(manifest.get("resolved_child_num_cpus")) is not int
        or manifest.get("resolved_child_num_cpus") != EXPECTED_CHILD_CPUS
        or manifest.get("execution_mode") != execution_mode
        or manifest.get("swap_policy") != SWAP_POLICY
        or manifest.get("timing_admissible") is not False
    ):
        raise RuntimeError("run manifest does not match the frozen campaign")
    return manifest


def _sequential_recovery_record(
    source: Path | None, *, destination: Path | None = None
) -> dict[str, Any] | None:
    if source is None:
        return None
    if "results.pkl" in (part.casefold() for part in source.parts):
        raise RuntimeError("sequential recovery source path is unsafe")
    root = _validate_campaign_namespace(
        source.resolve(strict=True), field="sequential recovery source namespace"
    )
    if "results.pkl" in (part.casefold() for part in root.parts):
        # Recheck the canonical target: a benign lexical name may be a symlink
        # into a reserved raw-result namespace.
        raise RuntimeError("sequential recovery source path is unsafe")
    if destination is not None:
        resolved_destination = _validate_campaign_namespace(
            destination, field="sequential recovery destination namespace"
        )
        root_parts = tuple(part.casefold() for part in root.parts)
        destination_parts = tuple(
            part.casefold() for part in resolved_destination.parts
        )
        if (
            root_parts == destination_parts[: len(root_parts)]
            or destination_parts == root_parts[: len(destination_parts)]
        ):
            raise RuntimeError(
                "recovery source and destination must be disjoint namespaces"
            )
    marker = root / INVALID_ATTEMPT_FILENAME
    if not root.is_dir() or not marker.is_file() or marker.is_symlink():
        raise RuntimeError("sequential recovery source has no regular invalid marker")
    if _exists_including_broken_symlink(
        root / COMPLETION_ATTESTATION_FILENAME,
        "recovery completion attestation",
    ):
        raise RuntimeError("cannot recover from a completed campaign")
    marker_value = _read_json_regular(marker, "invalid attempt marker")
    marker_fields = {
        "schema_version",
        "kind",
        "invalidated_at_utc",
        "execution_mode",
        "stage",
        "reuse_allowed",
        "recovery_policy",
        "manifest_sha256",
        "worker_shutdown_confirmed",
        "failure_swap_telemetry",
        "error_type",
        "error",
    }
    source_manifest = root / MANIFEST_FILENAME
    if not source_manifest.is_file() or source_manifest.is_symlink():
        raise RuntimeError("recovery requires a regular production manifest")
    manifest = _validate_manifest_static(
        _read_json_regular(source_manifest, "recovery source manifest"),
        output_dir=root,
        execution_mode="concurrent",
    )
    manifest_digest = _sha256_file(source_manifest)
    if not isinstance(marker_value, Mapping) or set(marker_value) != marker_fields:
        raise RuntimeError("recovery source invalid marker is not canonical")
    failure_swap = _validated_failure_swap_telemetry(
        marker_value.get("failure_swap_telemetry"),
        "recovery invalid-marker swap telemetry",
    )
    if (
        type(marker_value.get("schema_version")) is not int
        or marker_value.get("schema_version") != HARNESS_SCHEMA_VERSION
        or marker_value.get("kind") != CAMPAIGN_KIND + "_invalid_attempt"
        or marker_value.get("execution_mode") != "concurrent"
        or marker_value.get("stage") != "production"
        or marker_value.get("reuse_allowed") is not False
        or marker_value.get("recovery_policy")
        != "fresh_sequential_namespace_from_wave_zero_only"
        or marker_value.get("manifest_sha256") != manifest_digest
        or marker_value.get("worker_shutdown_confirmed") is not True
        or failure_swap.get("capture_status") != "captured"
        or failure_swap.get("teardown_confirmed") is not True
        or manifest.get("sequential_recovery") is not None
        or not isinstance(marker_value.get("invalidated_at_utc"), str)
        or not marker_value.get("invalidated_at_utc")
        or not isinstance(marker_value.get("error_type"), str)
        or not 0 < len(marker_value.get("error_type")) <= 256
        or not isinstance(marker_value.get("error"), str)
        or not 0 < len(marker_value.get("error")) <= 4_096
        or _canonical_json(manifest.get("source_freeze"))
        != _canonical_json(validate_source_freeze())
        or _canonical_json(manifest.get("source"))
        != _canonical_json(collect_source_provenance(output_dir=root))
        or _canonical_json(manifest.get("runtime"))
        != _canonical_json(collect_runtime_provenance())
    ):
        raise RuntimeError("recovery source invalid marker is not canonical")
    record: dict[str, Any] = {
        "source_output_dir": str(root),
        "invalid_attempt_artifact": {
            "path": str(marker),
            "sha256": _sha256_file(marker),
            "size_bytes": marker.stat().st_size,
        },
        "source_manifest_artifact": {
            "path": str(source_manifest),
            "sha256": manifest_digest,
            "size_bytes": source_manifest.stat().st_size,
        },
        "reuse_policy": "no_results_reused_fresh_wave_zero",
    }
    return record


def validate_sequential_recovery_record_shape(value: Any) -> dict[str, Any]:
    """Validate recovery metadata without accessing its source namespace."""
    record = dict(_as_mapping(value, "sequential recovery record"))
    if set(record) != {
        "source_output_dir",
        "invalid_attempt_artifact",
        "source_manifest_artifact",
        "reuse_policy",
    } or record.get("reuse_policy") != "no_results_reused_fresh_wave_zero":
        raise RuntimeError("sequential recovery record fields changed")
    source_raw = record.get("source_output_dir")
    if not isinstance(source_raw, str):
        raise RuntimeError("sequential recovery source path changed")
    source = Path(source_raw)
    if (
        not source.is_absolute()
        or str(source) != source_raw
        or any(part.casefold() in {".", ".."} for part in source.parts)
        or "results.pkl" in (part.casefold() for part in source.parts)
    ):
        raise RuntimeError("sequential recovery source path is unsafe")
    for key, filename in (
        ("invalid_attempt_artifact", INVALID_ATTEMPT_FILENAME),
        ("source_manifest_artifact", MANIFEST_FILENAME),
    ):
        metadata = _as_mapping(record.get(key), key)
        digest = metadata.get("sha256")
        if (
            set(metadata) != {"path", "sha256", "size_bytes"}
            or metadata.get("path") != str(source / filename)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or _exact_int(metadata.get("size_bytes"), f"{key} size") < 0
        ):
            raise RuntimeError(f"{key} metadata changed")
    return record


def validate_sequential_recovery_record(value: Any) -> dict[str, Any]:
    record = validate_sequential_recovery_record_shape(value)
    regenerated = _sequential_recovery_record(
        Path(str(record["source_output_dir"]))
    )
    if regenerated != record:
        raise RuntimeError("sequential recovery source artifacts changed")
    return record


def build_run_manifest(
    *,
    output_dir: Path,
    execution_mode: str,
    source_freeze: Mapping[str, Any],
    source: Mapping[str, Any],
    runtime: Mapping[str, Any],
    sequential_recovery: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if execution_mode not in {"concurrent", "sequential_recovery"}:
        raise RuntimeError("invalid CTR23 execution mode")
    return {
        "schema_version": 1,
        "kind": CAMPAIGN_KIND,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir.resolve()),
        "protocol_sha256": protocol_sha256(),
        "frozen_protocol_sha256": frozen_protocol_sha256(),
        "coordinate_manifest_sha256": COORDINATE_MANIFEST_SHA256,
        "schedule_sha256": schedule_sha256(),
        "schedule": expected_schedule(),
        "expected_jobs": EXPECTED_JOBS,
        "expected_child_fits": EXPECTED_CHILD_FITS,
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "resolved_child_num_cpus": EXPECTED_CHILD_CPUS,
        "execution_mode": execution_mode,
        "swap_policy": SWAP_POLICY,
        "timing_admissible": False,
        "source_freeze": dict(source_freeze),
        "source": dict(source),
        "runtime": dict(runtime),
        "sequential_recovery": (
            None if sequential_recovery is None else dict(sequential_recovery)
        ),
    }


def _write_invalid_attempt(
    output_dir: Path,
    *,
    execution_mode: str,
    error: BaseException,
) -> None:
    try:
        manifest_path = output_dir / MANIFEST_FILENAME
        production_started = manifest_path.is_file()
        failure_swap: dict[str, Any] | None = None
        attached = getattr(error, _FAILURE_SWAP_TELEMETRY_ATTRIBUTE, None)
        if attached is not None:
            try:
                failure_swap = _validated_failure_swap_telemetry(
                    attached, "failed-attempt swap telemetry"
                )
            except BaseException:
                failure_swap = None
        worker_shutdown_confirmed = bool(
            failure_swap is not None
            and failure_swap.get("teardown_confirmed") is True
        )
        recoverable = bool(
            production_started
            and execution_mode == "concurrent"
            and worker_shutdown_confirmed
            and failure_swap is not None
            and failure_swap.get("capture_status") == "captured"
        )
        marker = {
            "schema_version": HARNESS_SCHEMA_VERSION,
            "kind": CAMPAIGN_KIND + "_invalid_attempt",
            "invalidated_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution_mode": execution_mode,
            "stage": "production" if production_started else "preflight",
            "reuse_allowed": False,
            "recovery_policy": (
                "fresh_sequential_namespace_from_wave_zero_only"
                if recoverable
                else "not_recoverable"
            ),
            "manifest_sha256": (
                _sha256_file(manifest_path) if production_started else None
            ),
            "worker_shutdown_confirmed": worker_shutdown_confirmed,
            "failure_swap_telemetry": failure_swap,
            "error_type": type(error).__name__[:256],
            "error": _safe_exception_text(error)[:4_096],
        }
        _atomic_write_json(output_dir / INVALID_ATTEMPT_FILENAME, marker)
    except BaseException:
        pass


def execute_production(output_dir: Path, *, execution_mode: str) -> dict[str, Any]:
    failure_context: dict[str, Any] = {}
    try:
        return _execute_production_impl(
            output_dir,
            execution_mode=execution_mode,
            failure_context=failure_context,
        )
    except BaseException as exc:
        retained = failure_context.get("failure_swap_telemetry")
        if isinstance(retained, Mapping):
            _attach_failure_swap_telemetry(exc, retained)
        raise


def _execute_production_impl(
    output_dir: Path,
    *,
    execution_mode: str,
    failure_context: dict[str, Any],
) -> dict[str, Any]:
    workers: list[dict[str, Any]] = []
    ready: list[dict[str, Any]] = []
    warmup: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    session_swap = _new_worker_session_swap_telemetry()
    measured_swap: dict[str, Any] | None = None
    peak_fraction = 0.0
    error: BaseException | None = None
    worker_start_completed = False
    startup_cleanup_confirmed = False
    try:
        workers, ready = _start_workers(output_dir / "worker_scratch")
        worker_start_completed = True
        _checkpoint_worker_session_swap_telemetry(session_swap)
        warmup = _warm_workers(workers)
        _prime_workers(workers)
        warmup_document = {
            "schema_version": HARNESS_SCHEMA_VERSION,
            "kind": CAMPAIGN_KIND + "_warmup_history",
            "execution_mode": execution_mode,
            "worker_ready": ready,
            "worker_warmup": warmup,
        }
        _atomic_write_json(output_dir / WARMUP_HISTORY_FILENAME, warmup_document)
        measured_swap = hardened._start_measured_swap_window(
            session_swap, swap_policy=SWAP_POLICY
        )
        for wave in expected_schedule():
            wave_index = int(wave["wave_index"])
            wave_swap_start = int(measured_swap["end_sample_index"])
            assignments = [
                (
                    int(item["worker_slot"]),
                    _key_tuple(item["key"]),
                    output_dir,
                )
                for item in wave["jobs"]
            ]
            if execution_mode == "concurrent":
                reports, telemetry = _dispatch_runs(
                    workers,
                    assignments,
                    label=f"production-wave-{wave_index}",
                )
                hardened._checkpoint_measured_swap_window(
                    session_swap,
                    measured_swap,
                    label=f"production-wave-{wave_index}",
                    resource_telemetry=telemetry,
                    reports=reports,
                    swap_policy=SWAP_POLICY,
                )
                wave_peak = float(telemetry["peak_combined_rss_fraction"])
                start_skew = int(telemetry["start_skew_ns"])
                overlap = int(telemetry["overlap_ns"])
                wave_elapsed = int(telemetry["wave_elapsed_ns"])
            else:
                reports = []
                telemetry_parts = []
                for order, assignment in enumerate(assignments):
                    part_reports, part_telemetry = _dispatch_runs(
                        workers,
                        [assignment],
                        label=f"recovery-wave-{wave_index}-job-{order}",
                    )
                    reports.extend(part_reports)
                    telemetry_parts.append(part_telemetry)
                    hardened._checkpoint_measured_swap_window(
                        session_swap,
                        measured_swap,
                        label=f"recovery-wave-{wave_index}-job-{order}",
                        resource_telemetry=part_telemetry,
                        reports=part_reports,
                        swap_policy=SWAP_POLICY,
                    )
                reports.sort(key=lambda value: value["slot"])
                wave_peak = max(
                    float(item["peak_combined_rss_fraction"])
                    for item in telemetry_parts
                )
                starts = [int(report["started_monotonic_ns"]) for report in reports]
                ends = [int(report["ended_monotonic_ns"]) for report in reports]
                start_skew = max(starts) - min(starts)
                overlap = max(0, min(ends) - max(starts))
                wave_elapsed = max(ends) - min(starts)
            wave_swap_end = int(measured_swap["end_sample_index"])
            swap_samples = session_swap["samples"]
            wave_swap_in = (
                swap_samples[wave_swap_end]["swap_in_bytes"]
                - swap_samples[wave_swap_start]["swap_in_bytes"]
            )
            wave_swap_out = (
                swap_samples[wave_swap_end]["swap_out_bytes"]
                - swap_samples[wave_swap_start]["swap_out_bytes"]
            )
            if wave_swap_out != 0 or wave_peak >= 0.8:
                raise RuntimeError(f"production wave {wave_index} violated resources")
            peak_fraction = max(peak_fraction, wave_peak)
            entries.append(
                {
                    "wave_index": wave_index,
                    "jobs": wave["jobs"],
                    "reports": reports,
                    "swap_start_sample_index": wave_swap_start,
                    "swap_end_sample_index": wave_swap_end,
                    "swap_in_delta": wave_swap_in,
                    "swap_out_delta": wave_swap_out,
                    "peak_combined_rss_fraction": wave_peak,
                    "start_skew_ns": start_skew,
                    "overlap_ns": overlap,
                    "wave_elapsed_ns": wave_elapsed,
                }
            )
            checkpoint = {
                "schema_version": HARNESS_SCHEMA_VERSION,
                "kind": CAMPAIGN_KIND + "_concurrency_history",
                "execution_mode": execution_mode,
                "swap_policy": SWAP_POLICY,
                "timing_admissible": False,
                "wave_count": len(entries),
                "entries": entries,
                "failure_count": 0,
                "worker_restart_count": 0,
                "recovery_mixing_count": 0,
                "worker_session_swap_telemetry": session_swap,
                "measured_phase_swap_window": measured_swap,
                "swap_dispatch_count": len(measured_swap["dispatches"]),
                "swap_in_bytes": int(session_swap["swap_in_delta"]),
                "swap_out_bytes": int(session_swap["swap_out_delta"]),
                "peak_combined_rss_fraction": peak_fraction,
            }
            _atomic_write_json(output_dir / CONCURRENCY_HISTORY_FILENAME, checkpoint)
        high_water, high_water_telemetry = hardened._query_worker_high_water(workers)
        high_water_fraction = (
            sum(int(item["process_peak_rss_bytes"]) for item in high_water)
            / int(high_water_telemetry["physical_memory_bytes"])
        )
        peak_fraction = max(peak_fraction, high_water_fraction)
        if peak_fraction >= 0.8:
            raise RuntimeError("production worker high-water RSS reached 80%")
    except BaseException as exc:
        error = exc
        startup_cleanup_confirmed = (
            getattr(
                exc,
                _WORKER_STARTUP_CLEANUP_CONFIRMED_ATTRIBUTE,
                False,
            )
            is True
        )
        raise
    finally:
        final_record = _finalize_worker_session(
            workers,
            session_swap,
            active_error=error,
            worker_start_completed=worker_start_completed,
            startup_cleanup_confirmed=startup_cleanup_confirmed,
        )
        failure_context["failure_swap_telemetry"] = final_record
    session_swap = _validated_worker_session_swap_telemetry(
        session_swap, "production worker-session swap telemetry"
    )
    expected_dispatches = EXPECTED_WAVES * (
        1 if execution_mode == "concurrent" else WORKER_COUNT
    )
    expected_session_samples = expected_dispatches + 4
    expected_measured_samples = expected_dispatches + 1
    if (
        measured_swap is None
        or session_swap.get("sample_count") != expected_session_samples
        or measured_swap.get("sample_count") != expected_measured_samples
        or len(measured_swap.get("dispatches", [])) != expected_dispatches
        or not hardened._measured_swap_window_valid(
            measured_swap,
            session_swap,
            swap_policy=SWAP_POLICY,
            require_shutdown_sample=True,
            expected_start_sample_index=2,
            require_exact_shutdown_suffix=True,
        )
    ):
        raise RuntimeError("production swap telemetry is incomplete")
    swap_in = int(session_swap["swap_in_delta"])
    swap_out = int(session_swap["swap_out_delta"])
    history = {
        "schema_version": HARNESS_SCHEMA_VERSION,
        "kind": CAMPAIGN_KIND + "_concurrency_history",
        "execution_mode": execution_mode,
        "swap_policy": SWAP_POLICY,
        "timing_admissible": False,
        "wave_count": len(entries),
        "entries": entries,
        "failure_count": 0,
        "worker_restart_count": 0,
        "recovery_mixing_count": 0,
        "worker_session_swap_telemetry": session_swap,
        "measured_phase_swap_window": measured_swap,
        "swap_dispatch_count": expected_dispatches,
        "swap_in_bytes": swap_in,
        "swap_out_bytes": swap_out,
        "peak_combined_rss_fraction": peak_fraction,
    }
    if len(entries) != EXPECTED_WAVES:
        raise RuntimeError("production did not complete all 45 waves")
    _atomic_write_json(output_dir / CONCURRENCY_HISTORY_FILENAME, history)
    return history


def write_completion_attestation(
    output_dir: Path,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_bytes = manifest_path.read_bytes()
    artifacts = collect_result_artifacts(output_dir)
    validation, outer_rows, child_rows = validate_completed_results(
        output_dir, artifacts
    )
    preflight = _read_json_regular(
        output_dir / PREFLIGHT_REPORT_FILENAME, "preflight report"
    )
    concurrency = _read_json_regular(
        output_dir / CONCURRENCY_HISTORY_FILENAME, "concurrency history"
    )
    swap_audit = build_swap_audit(
        preflight,
        concurrency,
        execution_mode=str(manifest["execution_mode"]),
    )
    validation["swap_in_audit_evidence_retained"] = True
    validation["preflight_swap_in_bytes"] = swap_audit["preflight"][
        "worker_lifecycle_swap_in_bytes"
    ]
    validation["production_swap_in_bytes"] = swap_audit["production"][
        "worker_lifecycle_swap_in_bytes"
    ]
    validation["swap_dispatch_count"] = swap_audit["production"][
        "measured_dispatch_count"
    ]
    validation["swap_wave_count"] = swap_audit["production"]["wave_count"]
    validation["swap_out_bytes"] = (
        swap_audit["preflight"]["worker_lifecycle_swap_out_bytes"]
        + swap_audit["production"]["worker_lifecycle_swap_out_bytes"]
    )
    validation["peak_combined_rss_fraction"] = concurrency[
        "peak_combined_rss_fraction"
    ]
    validate_completion_for_analysis(
        validation,
        manifest=manifest,
        outer_rows=outer_rows,
        child_rows=child_rows,
        swap_audit=swap_audit,
    )
    boundary = analysis_boundary()
    payload = {
        "schema_version": HARNESS_SCHEMA_VERSION,
        "kind": PAYLOAD_KIND,
        "protocol_sha256": protocol_sha256(),
        "frozen_protocol_sha256": frozen_protocol_sha256(),
        "coordinate_manifest_sha256": COORDINATE_MANIFEST_SHA256,
        "schedule_sha256": schedule_sha256(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "result_artifacts_sha256": hashlib.sha256(
            _canonical_json(artifacts)
        ).hexdigest(),
        "analysis_boundary_sha256": hashlib.sha256(
            _canonical_json(boundary)
        ).hexdigest(),
        "swap_policy": SWAP_POLICY,
        "timing_admissible": False,
        "swap_audit": swap_audit,
        "outer_rows": outer_rows,
        "child_rows": child_rows,
    }
    payload_path = output_dir / ANALYSIS_PAYLOAD_FILENAME
    _atomic_write_json(payload_path, payload)
    singleton_paths = {
        "analysis_payload_artifact": payload_path,
        "schedule_artifact": output_dir / SCHEDULE_FILENAME,
        "preflight_report_artifact": output_dir / PREFLIGHT_REPORT_FILENAME,
        "concurrency_history_artifact": output_dir / CONCURRENCY_HISTORY_FILENAME,
        "warmup_history_artifact": output_dir / WARMUP_HISTORY_FILENAME,
    }
    attestation = {
        "schema_version": HARNESS_SCHEMA_VERSION,
        "kind": COMPLETION_KIND,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "execution_mode": manifest["execution_mode"],
        "swap_policy": SWAP_POLICY,
        "timing_admissible": False,
        "protocol_sha256": protocol_sha256(),
        "frozen_protocol_sha256": frozen_protocol_sha256(),
        "coordinate_manifest_sha256": COORDINATE_MANIFEST_SHA256,
        "schedule_sha256": schedule_sha256(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "result_count": len(artifacts),
        "expected_result_count": EXPECTED_JOBS,
        "expected_child_fits": EXPECTED_CHILD_FITS,
        "result_artifacts": artifacts,
        "raw_result_verification": {
            "authority": "runner",
            "count": len(artifacts),
            "method": "sha256_size_and_safe_extraction",
            "analyzer_access": "forbidden",
        },
        "analysis_boundary": boundary,
        "swap_audit": swap_audit,
        **{
            field: _artifact_metadata(path, output_dir)
            for field, path in singleton_paths.items()
        },
        "validation": validation,
    }
    operational = {
        "preflight_report_artifact": preflight,
        "concurrency_history_artifact": concurrency,
        "warmup_history_artifact": _read_json_regular(
            output_dir / WARMUP_HISTORY_FILENAME, "warmup history"
        ),
    }
    validate_operational_artifacts_for_analysis(
        operational,
        manifest=manifest,
        attestation=attestation,
        output_dir=output_dir,
    )
    if (
        validate_source_freeze() != manifest["source_freeze"]
        or collect_source_provenance(output_dir=output_dir) != manifest["source"]
        or collect_runtime_provenance() != manifest["runtime"]
        or collect_result_artifacts(output_dir) != artifacts
    ):
        raise RuntimeError("campaign provenance or results changed before completion")
    if manifest["execution_mode"] == "sequential_recovery":
        current_recovery = validate_sequential_recovery_record(
            manifest.get("sequential_recovery")
        )
        if _canonical_json(current_recovery) != _canonical_json(
            manifest.get("sequential_recovery")
        ):
            raise RuntimeError("sequential recovery source changed before completion")
    _atomic_write_json(output_dir / COMPLETION_ATTESTATION_FILENAME, attestation)
    return attestation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--sequential-recovery-from", type=Path)
    parser.add_argument(
        "--chimeraboost-path", type=Path, default=DEFAULT_CHIMERABOOST_PATH
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    chimeraboost_path = args.chimeraboost_path.resolve(strict=True)
    if chimeraboost_path != DEFAULT_CHIMERABOOST_PATH.resolve(strict=True):
        raise RuntimeError("frozen campaign requires the preregistered Chimera checkout")
    source_freeze = validate_source_freeze()
    runtime = collect_runtime_provenance()
    # This materializes experiment metadata and resource probes only.  It does
    # not load a CTR23 target, fit a model, or inspect any score.
    _, jobs, child_cpus = build_runtime_jobs(chimeraboost_path=chimeraboost_path)
    if len(jobs) != EXPECTED_JOBS or child_cpus != EXPECTED_CHILD_CPUS:
        raise RuntimeError("dry construction did not reproduce the frozen grid")
    validate_schedule(expected_schedule())
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_valid",
                    "job_count": len(jobs),
                    "child_fit_count": EXPECTED_CHILD_FITS,
                    "wave_count": EXPECTED_WAVES,
                    "coordinate_manifest_sha256": COORDINATE_MANIFEST_SHA256,
                    "schedule_sha256": schedule_sha256(),
                    "runtime": runtime,
                },
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output_dir = _validated_campaign_output_path(args.output_dir)
    if output_dir.exists():
        raise RuntimeError("campaign output must be a fresh zero-start namespace")
    _validate_campaign_namespace(output_dir, field="campaign output namespace")
    recovery = _sequential_recovery_record(
        args.sequential_recovery_from, destination=output_dir
    )
    execution_mode = "sequential_recovery" if recovery is not None else "concurrent"
    source = collect_source_provenance(
        output_dir=output_dir,
        chimeraboost_path=chimeraboost_path,
    )
    verify_live_official_splits()
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    production_history: Mapping[str, Any] | None = None
    try:
        run_preflight(output_dir)
        if args.preflight_only:
            return 0
        manifest = build_run_manifest(
            output_dir=output_dir,
            execution_mode=execution_mode,
            source_freeze=source_freeze,
            source=source,
            runtime=runtime,
            sequential_recovery=recovery,
        )
        _atomic_write_json(output_dir / MANIFEST_FILENAME, manifest)
        _atomic_write_json(output_dir / SCHEDULE_FILENAME, expected_schedule())
        production_history = execute_production(
            output_dir, execution_mode=execution_mode
        )
        write_completion_attestation(output_dir, manifest=manifest)
    except BaseException as exc:
        if (
            getattr(exc, _FAILURE_SWAP_TELEMETRY_ATTRIBUTE, None) is None
            and production_history is not None
        ):
            try:
                retained = _completed_session_failure_swap_telemetry(
                    production_history.get("worker_session_swap_telemetry")
                )
            except BaseException:
                pass
            else:
                _attach_failure_swap_telemetry(exc, retained)
        _write_invalid_attempt(
            output_dir,
            execution_mode=execution_mode,
            error=exc,
        )
        raise
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "result_count": EXPECTED_JOBS,
                "child_fit_count": EXPECTED_CHILD_FITS,
                "timing_admissible": False,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
