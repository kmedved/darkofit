"""Verify and analyze the frozen v0.11 M2 broad comparison panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import stat
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:  # Direct execution from a clean checkout.
    _REPO_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

try:
    from benchmarks import analyze_tabarena_regression_same_machine as _legacy
    from benchmarks import run_v011_m2_broad_panel as campaign
except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
    import analyze_tabarena_regression_same_machine as _legacy
    import run_v011_m2_broad_panel as campaign


METRICS = (
    "test_rmse",
    "val_rmse",
    "train_time_s",
    "infer_time_s",
    "incremental_memory_bytes",
    "peak_memory_bytes",
)
CONTRAST_CODES = (("D", "M"), ("D", "C"), ("M", "C"))
ARM_BY_CODE = {
    spec["code"]: {"arm": arm, "engine": spec["engine"]}
    for arm, spec in campaign.ARM_SPECS.items()
}
OUTPUT_NAMES = campaign.DEFAULT_ANALYSIS_OUTPUT_FILENAMES
MANIFEST_FIELDS = {
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
}
ATTESTATION_FIELDS = {
    "schema_version",
    "kind",
    "completed_at_utc",
    "pid",
    "result_count",
    "expected_result_count",
    "expected_primary_result_count",
    "expected_ordinal_diagnostic_result_count",
    "expected_child_fits",
    "warmup_thread_count",
    "warmup_stage_count",
    "protocol_sha256",
    "job_order_sha256",
    "git_head",
    "manifest_sha256",
    "result_artifacts",
    "analysis_payload_artifact",
    "warmup_history_artifact",
    "resume_history_artifact",
    "validation",
    "fresh_worker_count",
    "worker_attestation_artifacts",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    return _legacy._read_json_stable(path, field)


def _as_mapping(value: Any, field: str) -> dict[str, Any]:
    return dict(_legacy.hardened._as_mapping(value, field))


def _artifact_bytes(
    input_dir: Path, relative: str, metadata: Mapping[str, Any], field: str
) -> bytes:
    return _legacy._artifact_bytes(input_dir, relative, metadata, field)


def _expected_framework(arm: str) -> str:
    with campaign.configured_base():
        return campaign._base._experiment_name(arm)


def _verify_chimeraboost_source(value: Any, input_dir: Path) -> None:
    recorded = _as_mapping(value, "ChimeraBoost source")
    expected_fields = {
        "repository",
        "git_head",
        "git_tree",
        "git_describe",
        "git_remote_origin",
        "status",
        "module_file",
        "module_sha256",
        "hidden_import_warmup",
    }
    if set(recorded) != expected_fields or recorded.get("status") != "":
        raise RuntimeError("ChimeraBoost f14be60 provenance fields are not exact")
    repository = _legacy.hardened._manifest_path(
        recorded.get("repository"), "ChimeraBoost repository"
    ).resolve(strict=True)
    if _legacy.hardened._repository_changes(repository, input_dir):
        raise RuntimeError("ChimeraBoost checkout changed after execution")
    head = _legacy.hardened._git_output(
        repository, ["rev-parse", "HEAD"], "ChimeraBoost HEAD"
    )
    tree = _legacy.hardened._git_output(
        repository, ["rev-parse", "HEAD^{tree}"], "ChimeraBoost tree"
    )
    describe = _legacy.hardened._git_output(
        repository, ["describe", "--tags", "--always"], "ChimeraBoost describe"
    )
    remote = _legacy.hardened._sanitize_git_remote(
        _legacy.hardened._git_output(
            repository, ["remote", "get-url", "origin"], "ChimeraBoost origin"
        )
    )
    module_path = _legacy.hardened._manifest_path(
        recorded.get("module_file"), "ChimeraBoost module"
    ).resolve(strict=True)
    module_path.relative_to(repository)
    module_payload = _legacy.hardened._read_stable(
        module_path, "ChimeraBoost module"
    )
    if (
        head != campaign.CHIMERABOOST_TAG_COMMIT
        or recorded.get("git_head") != head
        or recorded.get("git_tree") != tree
        or describe != campaign.CHIMERABOOST_DESCRIBE
        or recorded.get("git_describe") != describe
        or recorded.get("git_remote_origin") != remote
        or recorded.get("hidden_import_warmup") != "disabled"
        or recorded.get("module_sha256") != _sha256(module_payload)
    ):
        raise RuntimeError("ChimeraBoost f14be60 provenance changed")


def _verify_execution_provenance(
    manifest: Mapping[str, Any], input_dir: Path
) -> dict[str, Any]:
    source = _as_mapping(manifest.get("source"), "manifest source")
    with campaign.configured_base():
        campaign.validate_framework_pins(source)
        diagnostics = _legacy._verify_repository_source(source, input_dir)
        repository = Path(__file__).resolve().parents[1]
        _legacy.hardened._verify_dependency_provenance(
            source.get("darkofit_import"),
            "darkofit",
            input_dir,
            required_repository=repository,
        )
        _legacy.hardened._verify_dependency_provenance(
            source.get("tabarena"), "tabarena", input_dir
        )
        _verify_chimeraboost_source(source.get("chimeraboost"), input_dir)
        _legacy._verify_catboost_source(source.get("catboost"))
        _legacy._verify_external_adapter_sources(
            source.get("external_adapter_sources")
        )
        _legacy._verify_runtime_provenance(manifest.get("runtime"))
    return {
        **diagnostics,
        "executing_source_verified": True,
        "analysis_runtime_verified": True,
        "dependency_provenance_verified": True,
        "hardware_identity_verified": True,
    }


def _verify_outer_rows(
    rows: Any, artifacts: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != campaign.EXPECTED_JOBS:
        raise RuntimeError("M2 outer row count is wrong")
    expected = campaign.expected_ordered_grid()
    fields = {
        "lane",
        "dataset",
        "task_id",
        "repeat",
        "fold",
        "registered_fold",
        "arm",
        "arm_code",
        "engine",
        "representation",
        "test_rmse",
        "val_rmse",
        "train_time_s",
        "infer_time_s",
        "incremental_memory_bytes",
        "baseline_memory_bytes",
        "peak_memory_bytes",
        "framework",
        "num_cpus",
        "num_gpus",
        "num_cpus_child",
        "num_gpus_child",
        "source",
    }
    normalized: list[dict[str, Any]] = []
    seen_sources: Counter[str] = Counter()
    for index, raw in enumerate(rows):
        row = _as_mapping(raw, f"outer_rows[{index}]")
        if set(row) != fields:
            raise RuntimeError(f"outer_rows[{index}] fields are not exact")
        lane, dataset, repeat, fold, arm = expected[index]
        spec = campaign.ARM_SPECS[arm]
        if (
            row["lane"] != lane
            or row["dataset"] != dataset
            or int(row["repeat"]) != repeat
            or int(row["fold"]) != fold
            or row["arm"] != arm
            or row["arm_code"] != spec["code"]
            or row["engine"] != spec["engine"]
            or row["representation"] != "native"
            or int(row["task_id"]) != campaign.TASKS[dataset]
            or int(row["registered_fold"]) != 3 * repeat + fold
            or int(row["num_cpus"]) != campaign.EXPECTED_CHILD_CPUS
            or int(row["num_cpus_child"]) != campaign.EXPECTED_CHILD_CPUS
            or float(row["num_gpus"]) != 0.0
            or float(row["num_gpus_child"]) != 0.0
        ):
            raise RuntimeError(f"outer row does not match frozen coordinate {expected[index]}")
        source = row["source"]
        if not isinstance(source, str) or source not in artifacts:
            raise RuntimeError("outer row is not bound to an attested raw result")
        framework = _expected_framework(arm)
        if row["framework"] != framework:
            raise RuntimeError("outer row framework does not match its frozen arm")
        _legacy._validate_result_source_binding(
            source,
            framework=framework,
            task_id=campaign.TASKS[dataset],
            repeat=repeat,
            fold=fold,
        )
        seen_sources[source] += 1
        clean = dict(row)
        for metric in METRICS:
            value = float(row[metric])
            if not math.isfinite(value) or value < 0.0 or (
                metric not in {"incremental_memory_bytes"} and value == 0.0
            ):
                raise RuntimeError(f"outer metric {metric} is invalid")
            clean[metric] = value
        baseline = float(row["baseline_memory_bytes"])
        peak = clean["peak_memory_bytes"]
        if (
            not math.isfinite(baseline)
            or baseline < 0.0
            or peak < baseline
            or not math.isclose(
                clean["incremental_memory_bytes"],
                peak - baseline,
                rel_tol=1e-12,
                abs_tol=1e-6,
            )
        ):
            raise RuntimeError("outer memory telemetry is inconsistent")
        normalized.append(clean)
    if set(seen_sources) != set(artifacts) or any(
        count != 1 for count in seen_sources.values()
    ):
        raise RuntimeError("outer rows do not bind one-to-one to raw results")
    return normalized


def _verify_child_rows(
    rows: Any, outer_rows: Sequence[Mapping[str, Any]]
) -> tuple[Counter[str], Counter[str]]:
    if not isinstance(rows, list) or len(rows) != campaign.EXPECTED_CHILD_FITS:
        raise RuntimeError("M2 child row count is wrong")
    outer_index = {
        (row["lane"], row["dataset"], row["repeat"], row["fold"], row["arm"]): row
        for row in outer_rows
    }
    expected = [
        (*outer, child_fold)
        for outer in campaign.expected_ordered_grid()
        for child_fold in range(8)
    ]
    blocks: dict[tuple[str, int, int, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    stop_counts: Counter[str] = Counter()
    inferred_counts: Counter[str] = Counter()
    fields = {
        "lane",
        "dataset",
        "task_id",
        "repeat",
        "fold",
        "registered_fold",
        "arm",
        "arm_code",
        "engine",
        "child",
        "child_fold",
        "child_features",
        "representation",
        "initial_hyperparameters",
        "user_hyperparameters",
        "effective_hyperparameters",
        "comparator_fit",
        "refit_params",
        "num_cpus",
        "num_gpus",
        "source",
    }
    for index, raw in enumerate(rows):
        row = _as_mapping(raw, f"child_rows[{index}]")
        if set(row) != fields:
            raise RuntimeError(f"child_rows[{index}] fields are not exact")
        lane, dataset, repeat, fold, arm, child_fold = expected[index]
        key = (lane, dataset, repeat, fold, arm)
        spec = campaign.ARM_SPECS[arm]
        if (
            row.get("lane") != lane
            or row.get("dataset") != dataset
            or int(row.get("repeat")) != repeat
            or int(row.get("fold")) != fold
            or row.get("arm") != arm
            or row.get("engine") != spec["engine"]
            or row.get("arm_code") != spec["code"]
            or int(row.get("child_fold")) != child_fold
            or row.get("child") != f"S1F{child_fold + 1}"
            or int(row.get("task_id")) != campaign.TASKS[dataset]
            or int(row.get("registered_fold")) != 3 * repeat + fold
            or int(row.get("num_cpus")) != campaign.EXPECTED_CHILD_CPUS
            or float(row.get("num_gpus")) != 0.0
            or key not in outer_index
            or row.get("source") != outer_index[key]["source"]
        ):
            raise RuntimeError(f"child row does not match frozen position {index}")
        child_features = row.get("child_features")
        _legacy._feature_schema_sha256(
            child_features, f"child_rows[{index}].child_features"
        )
        _legacy._validate_representation(
            row.get("representation"),
            lane=lane,
            dataset=dataset,
            child_features=child_features,
            field=f"child_rows[{index}].representation",
        )
        fit = _legacy._validate_common_comparator_fit(
            row.get("comparator_fit"),
            engine=spec["engine"],
            child_cpus=campaign.EXPECTED_CHILD_CPUS,
            field=f"child_rows[{index}].comparator_fit",
        )
        _legacy._validate_child_config(
            row,
            engine=spec["engine"],
            child_fold=child_fold,
            field=f"child_rows[{index}]",
        )
        _legacy._finite_json(
            row.get("refit_params"), f"child_rows[{index}].refit_params"
        )
        stop_counts[str(fit.get("stop_reason") or "unknown")] += 1
        inferred = fit.get("stop_reason_inferred")
        if inferred is not None:
            inferred_counts["inferred" if inferred is True else "unresolved"] += 1
        block = (dataset, repeat, fold, child_fold)
        engine = spec["engine"]
        if engine in blocks[block]:
            raise RuntimeError("duplicate engine in M2 child block")
        blocks[block][engine] = row
    if len(blocks) != campaign.EXPECTED_PRIMARY_COORDINATES * 8:
        raise RuntimeError("M2 child block count is wrong")
    for key, engines in blocks.items():
        if set(engines) != set(campaign.ENGINE_SPECS):
            raise RuntimeError(f"incomplete M2 child block {key}")
        schemas = [engines[name]["child_features"] for name in sorted(engines)]
        if schemas[1:] != schemas[:-1]:
            raise RuntimeError(f"cross-engine child schema differs at {key}")
        representations = [
            engines[name]["representation"] for name in sorted(engines)
        ]
        if representations[1:] != representations[:-1]:
            raise RuntimeError(f"cross-engine representation differs at {key}")
    return stop_counts, inferred_counts


def _verify_worker_attestations(
    input_dir: Path,
    metadata: Any,
    result_artifacts: Mapping[str, Any],
    *,
    parent_pid: int,
) -> None:
    artifacts = _as_mapping(metadata, "worker attestation artifacts")
    if len(artifacts) != campaign.EXPECTED_JOBS:
        raise RuntimeError("worker-attestation artifact count is wrong")
    expected_paths = {
        str(campaign._worker_attestation_path(input_dir, index).relative_to(input_dir))
        for index in range(campaign.EXPECTED_JOBS)
    }
    if set(artifacts) != expected_paths:
        raise RuntimeError("worker-attestation artifact path set is wrong")
    root = input_dir / campaign.WORKER_ATTESTATION_DIRNAME
    observed = set()
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("worker-attestation tree contains an unsafe entry")
        observed.add(str(path.relative_to(input_dir)))
    if observed != expected_paths:
        raise RuntimeError("on-disk worker-attestation set changed")
    grid = campaign.expected_ordered_grid()
    for worker_index, expected in enumerate(grid):
        relative = str(
            campaign._worker_attestation_path(input_dir, worker_index).relative_to(
                input_dir
            )
        )
        payload_bytes = _artifact_bytes(
            input_dir,
            relative,
            _as_mapping(artifacts[relative], f"worker artifact {worker_index}"),
            f"worker attestation {worker_index}",
        )
        payload = _legacy._decode_json(
            payload_bytes, f"worker attestation {worker_index}"
        )
        engine = campaign.ARM_SPECS[expected[4]]["engine"]
        expected_fields = {
            "schema_version",
            "kind",
            "worker_index",
            "pid",
            "parent_pid",
            "started_at_utc",
            "completed_at_utc",
            "coordinate",
            "same_arm_warmup",
            "environment",
            "numba_thread_ceiling",
            "numba_current_threads_after_fit",
            "result_artifact",
        }
        coordinate = {
            "lane": expected[0],
            "dataset": expected[1],
            "repeat": expected[2],
            "fold": expected[3],
            "arm": expected[4],
            "engine": engine,
        }
        result_relative = str(
            Path("experiments")
            / "data"
            / _expected_framework(expected[4])
            / str(campaign.TASKS[expected[1]])
            / f"{expected[2]}_{expected[3]}"
            / "results.pkl"
        )
        if (
            set(payload) != expected_fields
            or payload.get("schema_version") != 1
            or payload.get("kind") != campaign.CAMPAIGN_KIND + "_worker"
            or payload.get("worker_index") != worker_index
            or type(payload.get("pid")) is not int
            or payload.get("pid", 0) <= 0
            or payload.get("pid") == parent_pid
            or payload.get("parent_pid") != parent_pid
            or payload.get("coordinate") != coordinate
            or payload.get("environment") != campaign.WORKER_ENVIRONMENT
            or payload.get("numba_thread_ceiling") != campaign.EXPECTED_CHILD_CPUS
            or payload.get("numba_current_threads_after_fit")
            != campaign.EXPECTED_CHILD_CPUS
            or not isinstance(payload.get("started_at_utc"), str)
            or not payload.get("started_at_utc")
            or not isinstance(payload.get("completed_at_utc"), str)
            or not payload.get("completed_at_utc")
            or payload.get("result_artifact")
            != {"path": result_relative, **result_artifacts[result_relative]}
        ):
            raise RuntimeError(f"worker attestation {worker_index} is inconsistent")
        warmup = _as_mapping(
            payload.get("same_arm_warmup"), f"worker {worker_index} warmup"
        )
        if (
            set(warmup) != {"engine", "stage_names", "stages", "warnings"}
            or warmup.get("engine") != engine
            or warmup.get("stage_names")
            != [f"{engine}_numeric", f"{engine}_categorical"]
            or not isinstance(warmup.get("warnings"), list)
            or not isinstance(warmup.get("stages"), list)
            or len(warmup["stages"]) != 2
        ):
            raise RuntimeError(f"worker {worker_index} warmup fields drifted")
        for position, stage in enumerate(warmup["stages"]):
            stage = _as_mapping(stage, f"worker {worker_index} warmup stage")
            input_kind = ("numeric", "categorical")[position]
            if (
                stage.get("name") != f"{engine}_{input_kind}"
                or stage.get("engine") != engine
                or stage.get("input_kind") != input_kind
                or stage.get("thread_count") != campaign.EXPECTED_CHILD_CPUS
                or stage.get("representation", {}).get("kind") != "native"
                or stage.get("comparator_fit", {}).get("engine") != engine
                or stage.get("comparator_fit", {}).get("num_cpus")
                != campaign.EXPECTED_CHILD_CPUS
            ):
                raise RuntimeError(f"worker {worker_index} warmup stage drifted")
        for warning in warmup["warnings"]:
            if (
                not isinstance(warning, Mapping)
                or set(warning) != {"category", "message"}
                or not isinstance(warning["category"], str)
                or not isinstance(warning["message"], str)
            ):
                raise RuntimeError(f"worker {worker_index} warning record drifted")


def verify_campaign(
    input_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    contract = campaign.load_contract()
    campaign.validate_execution_source_pin(contract)
    input_dir = Path(os.path.abspath(input_dir.expanduser()))
    campaign._base._reject_symlink_components(input_dir, "M2 campaign input")
    try:
        input_metadata = input_dir.lstat()
    except OSError as exc:
        raise RuntimeError(f"could not inspect M2 campaign input: {input_dir}") from exc
    if not stat.S_ISDIR(input_metadata.st_mode):
        raise RuntimeError("M2 campaign input must be a real directory")
    manifest, manifest_bytes = _read_json(
        input_dir / campaign._base.MANIFEST_FILENAME, "run manifest"
    )
    attestation, attestation_bytes = _read_json(
        input_dir / campaign._base.COMPLETION_ATTESTATION_FILENAME,
        "completion attestation",
    )
    protocol = campaign.frozen_protocol()
    protocol_sha = _sha256(_canonical_json(protocol))
    if (
        set(manifest) != MANIFEST_FIELDS
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != campaign.CAMPAIGN_KIND
        or manifest.get("protocol") != protocol
        or manifest.get("protocol_sha256") != protocol_sha
        or manifest.get("job_order_sha256") != campaign.job_order_sha256()
        or manifest.get("ordering_audit") != campaign.expected_position_audit()
        or Path(str(manifest.get("output_dir"))).resolve() != input_dir
        or int(manifest.get("resolved_child_num_cpus")) != campaign.EXPECTED_CHILD_CPUS
        or float(manifest.get("time_limit_seconds")) != campaign.TIME_LIMIT_SECONDS
    ):
        raise RuntimeError("run manifest does not match the frozen M2 contract")
    execution = _verify_execution_provenance(manifest, input_dir)
    expected_attestation_counts = {
        "result_count": campaign.EXPECTED_JOBS,
        "expected_result_count": campaign.EXPECTED_JOBS,
        "expected_primary_result_count": campaign.EXPECTED_JOBS,
        "expected_ordinal_diagnostic_result_count": 0,
        "expected_child_fits": campaign.EXPECTED_CHILD_FITS,
    }
    if (
        set(attestation) != ATTESTATION_FIELDS
        or attestation.get("schema_version") != 1
        or attestation.get("kind") != campaign.COMPLETION_KIND
        or any(
            attestation.get(name) != value
            for name, value in expected_attestation_counts.items()
        )
        or attestation.get("protocol_sha256") != protocol_sha
        or attestation.get("job_order_sha256") != campaign.job_order_sha256()
        or attestation.get("manifest_sha256") != _sha256(manifest_bytes)
        or attestation.get("git_head") != manifest["source"]["git_head"]
        or attestation.get("warmup_thread_count") != campaign.EXPECTED_CHILD_CPUS
        or attestation.get("warmup_stage_count")
        != len(campaign._base.WARMUP_STAGE_NAMES)
        or attestation.get("fresh_worker_count") != campaign.EXPECTED_JOBS
    ):
        raise RuntimeError("completion attestation does not match frozen M2")
    artifacts = _as_mapping(attestation.get("result_artifacts"), "result artifacts")
    if len(artifacts) != campaign.EXPECTED_JOBS:
        raise RuntimeError("attested raw-result count is wrong")
    observed = {
        str(path.relative_to(input_dir))
        for path in (input_dir / "experiments").rglob("results.pkl")
    }
    if observed != set(artifacts):
        raise RuntimeError("raw-result files do not match the attestation")
    for relative, metadata in artifacts.items():
        _artifact_bytes(
            input_dir,
            relative,
            _as_mapping(metadata, f"artifact {relative}"),
            f"raw result {relative}",
        )
    _verify_worker_attestations(
        input_dir,
        attestation.get("worker_attestation_artifacts"),
        artifacts,
        parent_pid=int(attestation["pid"]),
    )
    payload_meta = _as_mapping(
        attestation.get("analysis_payload_artifact"), "analysis payload artifact"
    )
    payload_name = campaign._base.ANALYSIS_PAYLOAD_FILENAME
    if set(payload_meta) != {"path", "sha256", "size_bytes"} or payload_meta.get(
        "path"
    ) != payload_name:
        raise RuntimeError("analysis payload path changed")
    payload_bytes = _artifact_bytes(
        input_dir,
        payload_name,
        {name: payload_meta[name] for name in ("sha256", "size_bytes")},
        "analysis payload",
    )
    payload = _legacy._decode_json(payload_bytes, "analysis payload")
    if (
        set(payload)
        != {
            "schema_version",
            "kind",
            "protocol_sha256",
            "job_order_sha256",
            "result_artifacts_sha256",
            "outer_rows",
            "child_rows",
        }
        or payload.get("schema_version") != 1
        or payload.get("kind") != campaign.PAYLOAD_KIND
        or payload.get("protocol_sha256") != protocol_sha
        or payload.get("job_order_sha256") != campaign.job_order_sha256()
        or payload.get("result_artifacts_sha256") != _sha256(_canonical_json(artifacts))
    ):
        raise RuntimeError("analysis payload does not bind the M2 campaign")
    outer_rows = _verify_outer_rows(payload.get("outer_rows"), artifacts)
    stop_counts, inferred_counts = _verify_child_rows(
        payload.get("child_rows"), outer_rows
    )
    try:
        from benchmarks.tabarena_comparator_warmup import (
            validate_comparator_warmup_history,
        )
    except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
        from tabarena_comparator_warmup import validate_comparator_warmup_history

    warmup_history_sha = _legacy.hardened._verify_history_artifact(
        input_dir,
        attestation,
        attestation_field="warmup_history_artifact",
        filename=campaign._base.WARMUP_HISTORY_FILENAME,
        required=True,
        validator=lambda value: validate_comparator_warmup_history(
            value,
            expected_thread_count=campaign.EXPECTED_CHILD_CPUS,
            expected_latest_pid=int(attestation["pid"]),
        ),
    )
    resume_path = input_dir / campaign._base.RESUME_HISTORY_FILENAME
    if (
        attestation.get("resume_history_artifact") is not None
        or resume_path.exists()
        or resume_path.is_symlink()
    ):
        raise RuntimeError("M2 forbids resume history")
    terminal_path = input_dir / campaign.TERMINAL_FILENAME
    if terminal_path.exists() or terminal_path.is_symlink():
        raise RuntimeError("a terminal M2 campaign cannot be analyzed as complete")
    validation = _as_mapping(attestation.get("validation"), "validation")
    validation_fields = {
        "result_count",
        "child_fit_count",
        "lane_result_counts",
        "lane_child_counts",
        "cross_engine_representation_blocks",
        "failure_count",
        "imputation_count",
        "known_deadline_hit_count",
        "known_time_limit_stop_count",
        "stop_reason_counts",
        "competitor_stop_reason_inference_counts",
        "job_order_sha256",
        "resource_allocation",
        "memory_metric",
    }
    if set(validation) != validation_fields:
        raise RuntimeError("completion validation fields are not exact")
    expected_validation = {
        "result_count": campaign.EXPECTED_JOBS,
        "child_fit_count": campaign.EXPECTED_CHILD_FITS,
        "lane_result_counts": {campaign.PRIMARY_LANE: campaign.EXPECTED_JOBS},
        "lane_child_counts": {
            campaign.PRIMARY_LANE: campaign.EXPECTED_CHILD_FITS
        },
        "cross_engine_representation_blocks": (
            campaign.EXPECTED_PRIMARY_COORDINATES * 8
        ),
        "failure_count": 0,
        "imputation_count": 0,
        "known_deadline_hit_count": 0,
        "known_time_limit_stop_count": 0,
        "job_order_sha256": campaign.job_order_sha256(),
        "resource_allocation": {
            "num_cpus": campaign.EXPECTED_CHILD_CPUS,
            "num_gpus": 0,
            "num_cpus_child": campaign.EXPECTED_CHILD_CPUS,
            "num_gpus_child": 0,
        },
        "memory_metric": "peak_mem_cpu_minus_min_mem_cpu",
    }
    if any(validation.get(name) != value for name, value in expected_validation.items()):
        raise RuntimeError("completion validation does not match normalized M2 rows")
    if validation.get("stop_reason_counts") != dict(sorted(stop_counts.items())):
        raise RuntimeError("attested stop-reason counts do not match M2 children")
    if validation.get("competitor_stop_reason_inference_counts") != dict(
        sorted(inferred_counts.items())
    ):
        raise RuntimeError("attested stop inference does not match M2 children")
    return manifest, attestation, outer_rows, {
        "manifest_sha256": _sha256(manifest_bytes),
        "attestation_sha256": _sha256(attestation_bytes),
        "analysis_payload_sha256": _sha256(payload_bytes),
        "raw_result_set_sha256": _sha256(_canonical_json(artifacts)),
        "warmup_history_sha256": warmup_history_sha,
        **execution,
    }


def _ratio_fields(metric: str, numerator: float, denominator: float) -> dict[str, Any]:
    if metric == "incremental_memory_bytes" and (
        numerator == 0.0 or denominator == 0.0
    ):
        return {
            f"{metric}_ratio": None,
            f"{metric}_log_ratio": None,
            f"{metric}_pct": None,
        }
    ratio = numerator / denominator
    return {
        f"{metric}_ratio": ratio,
        f"{metric}_log_ratio": math.log(ratio),
        f"{metric}_pct": 100.0 * (ratio - 1.0),
    }


def pair_rows(outer_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (row["dataset"], row["repeat"], row["fold"], row["arm"]): row
        for row in outer_rows
    }
    paired: list[dict[str, Any]] = []
    for numerator_code, denominator_code in CONTRAST_CODES:
        numerator_spec = ARM_BY_CODE[numerator_code]
        denominator_spec = ARM_BY_CODE[denominator_code]
        contrast = f"{numerator_code}/{denominator_code}"
        for dataset in campaign.TASKS:
            for repeat, fold in campaign.PRIMARY_COORDINATE_PAIRS:
                numerator = index[(dataset, repeat, fold, numerator_spec["arm"])]
                denominator = index[(dataset, repeat, fold, denominator_spec["arm"])]
                row: dict[str, Any] = {
                    "contrast": contrast,
                    "numerator_engine": numerator_spec["engine"],
                    "denominator_engine": denominator_spec["engine"],
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
    if len(paired) != len(CONTRAST_CODES) * 39:
        raise RuntimeError("paired M2 coordinate grid is incomplete")
    return paired


def _bootstrap(logs_by_dataset: Mapping[str, Sequence[float]]) -> np.ndarray:
    rng = np.random.default_rng(campaign.BOOTSTRAP_SEED)
    datasets = list(campaign.TASKS)
    result = np.empty(campaign.BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(campaign.BOOTSTRAP_DRAWS):
        dataset_points = []
        for dataset in datasets:
            values = np.asarray(logs_by_dataset[dataset], dtype=np.float64)
            indices = rng.integers(0, len(values), size=len(values))
            dataset_points.append(float(np.mean(values[indices])))
        result[draw] = float(np.mean(dataset_points))
    return result


def _head_to_head(values: Sequence[float]) -> dict[str, Any]:
    wins = sum(value < 1.0 for value in values)
    losses = sum(value > 1.0 for value in values)
    ties = len(values) - wins - losses
    decisive = wins + losses
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate_excluding_ties": None if decisive == 0 else wins / decisive,
        "win_share_with_half_ties": (wins + 0.5 * ties) / len(values),
    }


def summarize(
    paired: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    comparisons: list[dict[str, Any]] = []
    per_dataset: list[dict[str, Any]] = []
    for numerator_code, denominator_code in CONTRAST_CODES:
        code = f"{numerator_code}/{denominator_code}"
        selected = [row for row in paired if row["contrast"] == code]
        metrics: dict[str, Any] = {}
        dataset_metric_ratios: dict[str, dict[str, float | None]] = defaultdict(dict)
        for metric in METRICS:
            logs_by_dataset: dict[str, list[float]] = {}
            unavailable = False
            for dataset in campaign.TASKS:
                values = [
                    row[f"{metric}_log_ratio"]
                    for row in selected
                    if row["dataset"] == dataset
                ]
                if len(values) != 3 or any(value is None for value in values):
                    unavailable = True
                    break
                logs_by_dataset[dataset] = [float(value) for value in values]
            if unavailable:
                metrics[metric] = {
                    "available": False,
                    "reason": "zero_incremental_memory_observation",
                }
                for dataset in campaign.TASKS:
                    dataset_metric_ratios[dataset][metric] = None
                continue
            dataset_logs = {
                dataset: math.fsum(values) / len(values)
                for dataset, values in logs_by_dataset.items()
            }
            point = math.fsum(dataset_logs.values()) / len(dataset_logs)
            draws = _bootstrap(logs_by_dataset)
            ratio = math.exp(point)
            metrics[metric] = {
                "available": True,
                "ratio": ratio,
                "pct": 100.0 * (ratio - 1.0),
                "bootstrap_ratio": {
                    "draws": campaign.BOOTSTRAP_DRAWS,
                    "p025": math.exp(float(np.quantile(draws, 0.025))),
                    "median": math.exp(float(np.quantile(draws, 0.5))),
                    "p975": math.exp(float(np.quantile(draws, 0.975))),
                },
            }
            for dataset, value in dataset_logs.items():
                dataset_metric_ratios[dataset][metric] = math.exp(value)
        dataset_items = []
        for dataset in campaign.TASKS:
            rows = [row for row in selected if row["dataset"] == dataset]
            quality_ratio = dataset_metric_ratios[dataset]["test_rmse"]
            item: dict[str, Any] = {
                "contrast": code,
                "numerator_engine": ARM_BY_CODE[numerator_code]["engine"],
                "denominator_engine": ARM_BY_CODE[denominator_code]["engine"],
                "dataset": dataset,
                "task_id": campaign.TASKS[dataset],
                "coordinate_count": 3,
                "test_rmse_wins": sum(row["test_rmse_ratio"] < 1.0 for row in rows),
                "test_rmse_losses": sum(row["test_rmse_ratio"] > 1.0 for row in rows),
                "test_rmse_ties": sum(row["test_rmse_ratio"] == 1.0 for row in rows),
            }
            for metric in METRICS:
                item[f"{metric}_ratio"] = dataset_metric_ratios[dataset][metric]
            dataset_items.append(item)
            per_dataset.append(item)
        quality_dataset_ratios = [
            float(item["test_rmse_ratio"]) for item in dataset_items
        ]
        quality_coordinate_ratios = [
            float(row["test_rmse_ratio"]) for row in selected
        ]
        comparisons.append(
            {
                "contrast": code,
                "numerator_engine": ARM_BY_CODE[numerator_code]["engine"],
                "denominator_engine": ARM_BY_CODE[denominator_code]["engine"],
                "paired_coordinates": 39,
                "datasets": 13,
                "metrics": metrics,
                "head_to_head": {
                    "coordinate_quality": _head_to_head(
                        quality_coordinate_ratios
                    ),
                    "equal_dataset_quality": _head_to_head(
                        quality_dataset_ratios
                    ),
                    "descriptive_only": True,
                },
            }
        )
    summary = {
        "schema_version": 1,
        "campaign": campaign.CAMPAIGN_KIND,
        "decision": "descriptive_only",
        "policy_advancement_allowed": False,
        "counts": {
            "outer_jobs": campaign.EXPECTED_JOBS,
            "child_fits": campaign.EXPECTED_CHILD_FITS,
            "coordinates": campaign.EXPECTED_PRIMARY_COORDINATES,
            "paired_coordinate_rows": len(paired),
            "per_dataset_rows": len(per_dataset),
        },
        "aggregation": {
            "datasets_fixed_and_equally_weighted": True,
            "coordinates_per_dataset": 3,
            "bootstrap_draws": campaign.BOOTSTRAP_DRAWS,
            "bootstrap_seed": campaign.BOOTSTRAP_SEED,
            "independent_dataset_claim": False,
        },
        "comparisons": comparisons,
        "provenance": dict(provenance),
    }
    return summary, per_dataset


def _csv_bytes(rows: Sequence[Mapping[str, Any]], field: str) -> bytes:
    if not rows:
        raise RuntimeError(f"{field} is empty")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# v0.11 M2 broad-panel result",
        "",
        "Status: **descriptive, spent evidence; no default or release decision is authorized.**",
        "",
        "All ratios are numerator / denominator; lower is better for every listed metric.",
        "The point estimate equally weights the 13 fixed datasets after averaging each",
        "dataset's three paired log ratios. Intervals resample coordinates within each",
        "fixed dataset and do not imply 13 independent datasets.",
        "",
        "| Contrast | Test RMSE | Fit time | Predict time | Incremental RSS | Dataset W-L-T |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["comparisons"]:
        metrics = item["metrics"]
        def ratio(name: str) -> str:
            value = metrics[name]
            return "n/a" if not value["available"] else f"{value['ratio']:.4f}x"
        counts = item["head_to_head"]["equal_dataset_quality"]
        lines.append(
            f"| {item['contrast']} | {ratio('test_rmse')} | {ratio('train_time_s')} | "
            f"{ratio('infer_time_s')} | {ratio('incremental_memory_bytes')} | "
            f"{counts['wins']}-{counts['losses']}-{counts['ties']} |"
        )
    lines.extend(
        [
            "",
            "The complete per-coordinate and per-dataset rows accompany this report.",
            "The panel is defaults-only at the pinned sources; the private ensemble is absent.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    input_dir = Path(os.path.abspath(args.input_dir.expanduser()))
    baseline = verify_campaign(input_dir)
    manifest, attestation, outer_rows, provenance = baseline
    provenance = {
        **provenance,
        "protocol_sha256": campaign.protocol_sha256(),
        "job_order_sha256": campaign.job_order_sha256(),
        "git_head": manifest["source"]["git_head"],
        "chimeraboost_git_head": manifest["source"]["chimeraboost"]["git_head"],
        "catboost_version": manifest["source"]["catboost"]["version"],
        "tabarena_git_head": manifest["source"]["tabarena"]["git_head"],
        "autogluon_version": campaign.AUTOGLUON_VERSION,
        "completed_at_utc": attestation["completed_at_utc"],
    }
    paired = pair_rows(outer_rows)
    summary, per_dataset = summarize(paired, provenance)
    summary["official_default_disclosure"] = manifest["protocol"][
        "official_default_disclosure"
    ]
    summary["framework_source"] = manifest["protocol"]["framework_source"]
    outputs = {
        input_dir / "paired_coordinates.csv": _csv_bytes(paired, "paired coordinates"),
        input_dir / "per_dataset.csv": _csv_bytes(per_dataset, "per-dataset rows"),
        input_dir / "summary.json": (
            json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        input_dir / "report.md": render_report(summary).encode("utf-8"),
    }
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise RuntimeError("M2 analysis outputs are create-only and already exist")
    if verify_campaign(input_dir) != baseline:
        raise RuntimeError("M2 campaign changed during analysis")
    written: list[Path] = []
    try:
        for path, payload in outputs.items():
            _write_create_only(path, payload)
            written.append(path)
        if verify_campaign(input_dir) != baseline:
            raise RuntimeError("M2 campaign changed while outputs were published")
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    print(
        f"analyzed {campaign.EXPECTED_JOBS} jobs and "
        f"{campaign.EXPECTED_CHILD_FITS} child fits; descriptive_only"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
