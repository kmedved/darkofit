"""Verify and analyze the frozen isolated TabArena follow-on screen."""

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
from statistics import fmean
from typing import Any

import numpy as np

try:
    from benchmarks import analyze_tabarena_regression_cap_horizon as hardened_analysis
    from benchmarks import run_tabarena_regression_followon_screen as screen
except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
    import analyze_tabarena_regression_cap_horizon as hardened_analysis
    import run_tabarena_regression_followon_screen as screen


BOOTSTRAP_SEED = 20_260_713
BOOTSTRAP_DRAWS = 10_000
METRICS = (
    "test_rmse",
    "val_rmse",
    "train_time_s",
    "infer_time_s",
    "peak_memory_bytes",
)
SCREEN_THRESHOLDS = {
    "test_ratio_max": 0.995,
    "validation_ratio_max": 1.002,
    "dataset_regret_max": 1.005,
    "train_time_ratio_max": 4.0,
    "infer_time_ratio_max": 1.25,
    "peak_memory_ratio_max": 2.0,
    "time_limit_stops_max": 0,
}
OUTPUT_KEYS = (
    "split_csv",
    "repeat_csv",
    "child_csv",
    "summary_json",
    "report_md",
)
OUTPUT_NAMES = (
    "paired_splits.csv",
    "per_repeat.csv",
    "paired_children.csv",
    "summary.json",
    "report.md",
)


def _read_json_stable(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"could not inspect {field}: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{field} must be a regular file: {path}")
    payload = hardened_analysis._read_stable(path, field)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{field} is not valid JSON") from exc
    return dict(hardened_analysis._as_mapping(value, field)), payload


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
    recorded_repository = hardened_analysis._manifest_path(
        source.get("repository"), "recorded screen repository"
    )
    if recorded_repository != repository:
        raise RuntimeError("executing analyzer repository does not match the run")
    files = hardened_analysis._as_mapping(source.get("files"), "source files")
    if set(files) != {str(path) for path in screen.SOURCE_FILES}:
        raise RuntimeError("run manifest source file set is not exact")
    for relative in screen.SOURCE_FILES:
        path = (repository / relative).resolve(strict=True)
        try:
            path.relative_to(repository)
        except ValueError as exc:
            raise RuntimeError(f"source file escapes repository: {relative}") from exc
        metadata = hardened_analysis._as_mapping(
            files[str(relative)], f"source metadata for {relative}"
        )
        if set(metadata) != {"sha256", "git_blob"}:
            raise RuntimeError(f"source metadata is incomplete for {relative}")
        payload = hardened_analysis._read_stable(path, f"source {relative}")
        if metadata.get("sha256") != hashlib.sha256(payload).hexdigest():
            raise RuntimeError(f"source SHA-256 mismatch for {relative}")
        if metadata.get("git_blob") != hardened_analysis._git_hash_payload(
            repository, payload, str(relative)
        ):
            raise RuntimeError(f"source Git-blob mismatch for {relative}")
    head = hardened_analysis._git_output(repository, ["rev-parse", "HEAD"], "Git HEAD")
    tree = hardened_analysis._git_output(
        repository, ["rev-parse", "HEAD^{tree}"], "Git tree"
    )
    if source.get("git_head") != head or source.get("git_tree") != tree:
        raise RuntimeError("executing Git revision does not match the run")
    changes = hardened_analysis._repository_changes(repository, input_dir)
    if changes:
        raise RuntimeError(
            "executing screen repository has dirty or unrecorded code: "
            + ", ".join(changes)
        )
    return {
        "executing_repository": str(repository),
        "executing_git_head": head,
        "executing_git_tree": tree,
    }


def verify_execution_provenance(
    manifest: Mapping[str, Any], input_dir: Path
) -> dict[str, Any]:
    source = hardened_analysis._as_mapping(manifest.get("source"), "manifest source")
    diagnostics = _verify_repository_source(source, input_dir)
    repository = Path(__file__).resolve().parents[1]
    hardened_analysis._verify_dependency_provenance(
        source.get("darkofit_import"),
        "darkofit",
        input_dir,
        required_repository=repository,
    )
    hardened_analysis._verify_dependency_provenance(
        source.get("tabarena"), "tabarena", input_dir
    )
    hardened_analysis._verify_runtime_provenance(manifest.get("runtime"))
    return {
        **diagnostics,
        "executing_source_verified": True,
        "analysis_runtime_verified": True,
        "dependency_provenance_verified": True,
    }


def _artifact_bytes(
    input_dir: Path, relative: str, metadata: Mapping[str, Any], field: str
) -> bytes:
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path.name == ""
    ):
        raise RuntimeError(f"unsafe attested path: {relative!r}")
    raw_path = input_dir / relative_path
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
    payload = hardened_analysis._read_stable(path, field)
    size = hardened_analysis._exact_int(metadata.get("size_bytes"), f"{field} size")
    digest = metadata.get("sha256")
    screen._validate_sha256(digest, f"{field} digest")
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
        raise RuntimeError(f"{field} does not match its attestation")
    return payload


def _validate_safe_payload(
    payload: Mapping[str, Any], artifacts: Mapping[str, Any], child_cpus: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outer_rows = payload.get("outer_rows")
    child_rows = payload.get("child_rows")
    if not isinstance(outer_rows, list) or not isinstance(child_rows, list):
        raise RuntimeError("analysis payload rows must be lists")
    if len(outer_rows) != screen.EXPECTED_JOBS:
        raise RuntimeError("analysis payload outer result count is wrong")
    if len(child_rows) != screen.EXPECTED_CHILD_FITS:
        raise RuntimeError("analysis payload child result count is wrong")

    outer_index = {}
    source_counts = Counter()
    expected_outer_fields = {
        "dataset",
        "task_id",
        "repeat",
        "fold",
        "registered_fold",
        "arm",
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
    for raw in outer_rows:
        row = dict(hardened_analysis._as_mapping(raw, "outer row"))
        if set(row) != expected_outer_fields:
            raise RuntimeError("safe outer row fields are not exact")
        dataset = row["dataset"]
        repeat = hardened_analysis._exact_int(row["repeat"], "outer repeat")
        fold = hardened_analysis._exact_int(row["fold"], "outer fold")
        arm = row["arm"]
        key = (dataset, repeat, fold, arm)
        if (
            key not in screen.expected_grid()
            or key in outer_index
            or row["task_id"] != screen.TASKS[dataset]
            or row["registered_fold"] != 3 * repeat + fold
            or row["num_cpus"] != child_cpus
            or row["num_cpus_child"] != child_cpus
            or row["num_gpus"] != 0
            or row["num_gpus_child"] != 0
        ):
            raise RuntimeError(f"safe outer row does not match screen grid: {key}")
        for metric in METRICS:
            screen._finite(row[metric], f"outer {metric}", positive=True)
        if row["source"] not in artifacts:
            raise RuntimeError("outer row source is not an attested result")
        screen._validate_result_source_binding(
            row["source"],
            dataset=dataset,
            repeat=repeat,
            fold=fold,
            arm=arm,
        )
        source_counts[row["source"]] += 1
        outer_index[key] = row
    if set(outer_index) != screen.expected_grid() or any(
        count != 1 for count in source_counts.values()
    ) or set(source_counts) != set(artifacts):
        raise RuntimeError("safe outer rows do not bind one-to-one to raw results")

    expected_child_fields = {
        "dataset",
        "task_id",
        "repeat",
        "fold",
        "registered_fold",
        "arm",
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
        "tree_mode_selection",
        "stop_reason",
        "deadline_hit",
        "wall_clock_elapsed_seconds",
        "representation",
        "refit_params",
        "num_cpus",
        "num_gpus",
        "source",
    }
    child_index = set()
    child_per_outer = Counter()
    normalized_children = []
    for raw in child_rows:
        row = dict(hardened_analysis._as_mapping(raw, "child row"))
        if set(row) != expected_child_fields:
            raise RuntimeError("safe child row fields are not exact")
        dataset = row["dataset"]
        if dataset not in screen.TASKS or row["arm"] not in screen.ARM_SPECS:
            raise RuntimeError("safe child dataset or arm is not declared")
        repeat = hardened_analysis._exact_int(row["repeat"], "child repeat")
        fold = hardened_analysis._exact_int(row["fold"], "child fold coordinate")
        task_id = hardened_analysis._exact_int(row["task_id"], "child task id")
        registered_fold = hardened_analysis._exact_int(
            row["registered_fold"], "child registered fold"
        )
        if (
            task_id != screen.TASKS[dataset]
            or registered_fold != 3 * repeat + fold
        ):
            raise RuntimeError("safe child task or registered fold does not match")
        outer_key = (dataset, repeat, fold, row["arm"])
        child_fold = hardened_analysis._exact_int(row["child_fold"], "child fold")
        key = (*outer_key, child_fold)
        if (
            outer_key not in outer_index
            or key in child_index
            or child_fold not in range(8)
            or row["child"] != f"S1F{child_fold + 1}"
            or row["source"] != outer_index[outer_key]["source"]
            or hardened_analysis._exact_int(row["num_cpus"], "child CPUs")
            != child_cpus
            or hardened_analysis._exact_int(row["num_gpus"], "child GPUs") != 0
        ):
            raise RuntimeError(f"safe child row does not match outer row: {key}")
        requested = hardened_analysis._exact_int(
            row["iterations_requested"], "iterations requested"
        )
        attempted = hardened_analysis._exact_int(
            row["iterations_attempted"], "iterations attempted"
        )
        completed = hardened_analysis._exact_int(
            row["rounds_completed"], "rounds completed"
        )
        retained = hardened_analysis._exact_int(
            row["rounds_retained"], "rounds retained"
        )
        best = hardened_analysis._exact_int(row["best_iteration"], "best iteration")
        if requested != 1_000 or not (
            0 <= retained == best <= completed <= attempted <= requested
        ):
            raise RuntimeError("safe child round counters are inconsistent")
        if (
            float(row["resolved_learning_rate"]) != 0.1
        ):
            raise RuntimeError("safe child learning-rate policy changed")
        screen._validate_early_stopping_rounds(
            row["early_stopping_rounds"], field="safe child early stopping rounds"
        )
        requested_mode = screen.ARM_SPECS[row["arm"]]["config"]["tree_mode"]
        selected_mode = row["selected_tree_mode"]
        if (
            row["requested_tree_mode"] != requested_mode
            or selected_mode not in {"catboost", "lightgbm", "hybrid"}
            or (requested_mode != "auto" and selected_mode != requested_mode)
        ):
            raise RuntimeError("safe child selected mode does not match its arm")
        screen._validate_linear_lane(
            arm=row["arm"],
            linear_active=row["linear_residual_active"],
            selected_lane=row["selected_lane"],
            field="safe child",
        )
        _validate_normalized_tree_mode_selection(row, selected_mode=selected_mode)
        if row["stop_reason"] not in screen.VALID_STOP_REASONS:
            raise RuntimeError("safe child stop reason is invalid")
        screen.hardened.validate_stop_reason_causality(
            row["stop_reason"],
            requested=requested,
            attempted=attempted,
            completed=completed,
            field="safe child",
        )
        if row["stop_reason"] == "time_limit" or row["deadline_hit"] is not False:
            raise RuntimeError("screen payload contains a deadline-hit child")
        if screen._finite(
            row["wall_clock_elapsed_seconds"], "safe child wall-clock elapsed"
        ) < 0.0:
            raise RuntimeError("safe child wall-clock elapsed must be nonnegative")
        screen._validate_representation_metadata(
            row["representation"],
            arm=row["arm"],
            dataset=row["dataset"],
            field="safe child representation",
        )
        screen._validate_refit_params(
            row["refit_params"],
            expected_iterations=best,
            selected_tree_mode=row["selected_tree_mode"],
            field="safe child refit params",
        )
        child_index.add(key)
        child_per_outer[outer_key] += 1
        normalized_children.append(row)
    if any(child_per_outer[key] != 8 for key in outer_index):
        raise RuntimeError("safe payload does not contain eight children per outer fit")
    return list(outer_index.values()), normalized_children


def _validate_normalized_tree_mode_selection(
    row: Mapping[str, Any], *, selected_mode: str
) -> None:
    """Revalidate runner-normalized auto metadata without unpickling results."""
    if row["arm"] == "auto":
        screen._validate_tree_mode_selection(
            row["tree_mode_selection"],
            selected_tree_mode=selected_mode,
            deadline_hit=row["deadline_hit"],
            top_level=row,
            field="safe child tree selection",
        )
    elif row["tree_mode_selection"] is not None:
        raise RuntimeError("non-auto safe child carries tree selection metadata")


def verify_campaign_integrity(
    input_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    input_dir = input_dir.resolve(strict=True)
    manifest_path = input_dir / screen.MANIFEST_FILENAME
    attestation_path = input_dir / screen.COMPLETION_ATTESTATION_FILENAME
    manifest, manifest_bytes = _read_json_stable(manifest_path, "run manifest")
    protocol = screen.frozen_protocol()
    protocol_digest = hashlib.sha256(screen.hardened._canonical_json(protocol)).hexdigest()
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != screen.CAMPAIGN_KIND
        or Path(str(manifest.get("output_dir", ""))).resolve() != input_dir
        or manifest.get("protocol") != protocol
        or manifest.get("protocol_sha256") != protocol_digest
        or manifest.get("time_limit_seconds") != screen.TIME_LIMIT_SECONDS
    ):
        raise RuntimeError("run manifest does not match the frozen screen")
    child_cpus = hardened_analysis._exact_int(
        manifest.get("resolved_child_num_cpus"), "manifest child CPUs"
    )
    if child_cpus < 1:
        raise RuntimeError("manifest child CPUs must be positive")
    execution = verify_execution_provenance(manifest, input_dir)

    attestation, attestation_bytes = _read_json_stable(
        attestation_path, "completion attestation"
    )
    expected_counts = {
        "result_count": screen.EXPECTED_JOBS,
        "expected_result_count": screen.EXPECTED_JOBS,
        "expected_child_fits": screen.EXPECTED_CHILD_FITS,
        "expected_paired_comparisons": screen.EXPECTED_PAIRED_COMPARISONS,
    }
    if (
        attestation.get("schema_version") != 1
        or attestation.get("kind") != screen.COMPLETION_KIND
        or any(attestation.get(key) != value for key, value in expected_counts.items())
        or attestation.get("protocol_sha256") != protocol_digest
        or attestation.get("git_head") != manifest["source"]["git_head"]
        or attestation.get("manifest_sha256")
        != hashlib.sha256(manifest_bytes).hexdigest()
        or attestation.get("warmup_thread_count") != child_cpus
        or attestation.get("warmup_stage_count") != len(screen.WARMUP_STAGE_SPECS)
        or attestation.get("warmup_expected_counts")
        != screen.EXPECTED_WARMUP_COUNTS
    ):
        raise RuntimeError("completion attestation does not match the screen")

    artifacts = hardened_analysis._as_mapping(
        attestation.get("result_artifacts"), "result artifacts"
    )
    if len(artifacts) != screen.EXPECTED_JOBS:
        raise RuntimeError("attested result count does not match the screen")
    observed = {
        str(path.relative_to(input_dir))
        for path in (input_dir / "experiments").rglob("results.pkl")
    }
    if observed != set(artifacts):
        raise RuntimeError("on-disk result set does not match the attestation")
    for relative, metadata in artifacts.items():
        if Path(relative).name != "results.pkl":
            raise RuntimeError("attested result has an unsafe filename")
        _artifact_bytes(
            input_dir,
            relative,
            hardened_analysis._as_mapping(metadata, "result artifact"),
            f"result {relative}",
        )

    payload_artifact = hardened_analysis._as_mapping(
        attestation.get("analysis_payload_artifact"), "analysis payload artifact"
    )
    if payload_artifact.get("path") != screen.ANALYSIS_PAYLOAD_FILENAME:
        raise RuntimeError("analysis payload path is not frozen")
    payload_bytes = _artifact_bytes(
        input_dir,
        screen.ANALYSIS_PAYLOAD_FILENAME,
        payload_artifact,
        "safe analysis payload",
    )
    try:
        payload = dict(
            hardened_analysis._as_mapping(
                json.loads(payload_bytes), "safe analysis payload"
            )
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("safe analysis payload is not valid JSON") from exc
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != screen.PAYLOAD_KIND
        or payload.get("protocol_sha256") != protocol_digest
        or payload.get("result_artifacts_sha256")
        != hashlib.sha256(screen.hardened._canonical_json(artifacts)).hexdigest()
    ):
        raise RuntimeError("safe analysis payload does not bind the campaign")
    outer_rows, child_rows = _validate_safe_payload(payload, artifacts, child_cpus)
    payload["outer_rows"] = outer_rows
    payload["child_rows"] = child_rows

    hardened_analysis._verify_history_artifact(
        input_dir,
        attestation,
        attestation_field="warmup_history_artifact",
        filename=screen.WARMUP_HISTORY_FILENAME,
        required=True,
        validator=lambda value: screen._validate_followon_warmup_history(
            value,
            expected_thread_count=child_cpus,
            expected_latest_pid=hardened_analysis._exact_int(
                attestation.get("pid"), "completion pid"
            ),
        ),
    )
    hardened_analysis._verify_history_artifact(
        input_dir,
        attestation,
        attestation_field="resume_history_artifact",
        filename=screen.RESUME_HISTORY_FILENAME,
        required=False,
        validator=lambda value: screen._validate_resume_history(value, input_dir),
    )
    validation = hardened_analysis._as_mapping(
        attestation.get("validation"), "completion validation"
    )
    if (
        validation.get("result_count") != screen.EXPECTED_JOBS
        or validation.get("child_fit_count") != screen.EXPECTED_CHILD_FITS
        or validation.get("paired_comparison_count")
        != screen.EXPECTED_PAIRED_COMPARISONS
        or validation.get("resource_allocation", {}).get("num_cpus_child")
        != child_cpus
    ):
        raise RuntimeError("completion validation summary does not match payload")
    return manifest, attestation, payload, {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "attestation_sha256": hashlib.sha256(attestation_bytes).hexdigest(),
        "analysis_payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        **execution,
    }


def pair_outer_rows(outer_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (row["dataset"], int(row["repeat"]), int(row["fold"]), row["arm"]): row
        for row in outer_rows
    }
    paired = []
    for arm in screen.CANDIDATE_ARMS:
        for dataset, repeat, fold in screen.expected_arm_coordinates(arm):
            baseline = index[(dataset, repeat, fold, "baseline")]
            candidate = index[(dataset, repeat, fold, arm)]
            row = {
                "candidate_arm": arm,
                "dataset": dataset,
                "task_id": screen.TASKS[dataset],
                "repeat": repeat,
                "fold": fold,
                "registered_fold": 3 * repeat + fold,
            }
            for metric in METRICS:
                denominator = float(baseline[metric])
                numerator = float(candidate[metric])
                ratio = numerator / denominator
                if not math.isfinite(ratio) or ratio <= 0.0:
                    raise RuntimeError(f"nonfinite paired ratio for {arm} {dataset}")
                row[f"baseline_{metric}"] = denominator
                row[f"candidate_{metric}"] = numerator
                row[f"{metric}_ratio"] = ratio
                row[f"{metric}_log_ratio"] = math.log(ratio)
                row[f"{metric}_pct"] = 100.0 * (ratio - 1.0)
            paired.append(row)
    if len(paired) != screen.EXPECTED_PAIRED_COMPARISONS:
        raise RuntimeError("paired split count does not match candidate grid")
    return paired


def pair_child_rows(child_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (
            row["dataset"],
            int(row["repeat"]),
            int(row["fold"]),
            int(row["child_fold"]),
            row["arm"],
        ): row
        for row in child_rows
    }
    paired = []
    for arm in screen.CANDIDATE_ARMS:
        for dataset, repeat, fold in screen.expected_arm_coordinates(arm):
            for child_fold in range(8):
                baseline = index[(dataset, repeat, fold, child_fold, "baseline")]
                candidate = index[(dataset, repeat, fold, child_fold, arm)]
                paired.append(
                    {
                        "candidate_arm": arm,
                        "dataset": dataset,
                        "repeat": repeat,
                        "fold": fold,
                        "registered_fold": 3 * repeat + fold,
                        "child": f"S1F{child_fold + 1}",
                        "child_fold": child_fold,
                        "baseline_best_iteration": baseline["best_iteration"],
                        "candidate_best_iteration": candidate["best_iteration"],
                        "baseline_rounds_completed": baseline["rounds_completed"],
                        "candidate_rounds_completed": candidate["rounds_completed"],
                        "baseline_stop_reason": baseline["stop_reason"],
                        "candidate_stop_reason": candidate["stop_reason"],
                        "baseline_selected_tree_mode": baseline["selected_tree_mode"],
                        "candidate_selected_tree_mode": candidate["selected_tree_mode"],
                        "baseline_selected_lane": baseline["selected_lane"],
                        "candidate_selected_lane": candidate["selected_lane"],
                        "candidate_linear_residual_active": candidate[
                            "linear_residual_active"
                        ],
                        "candidate_representation_kind": candidate[
                            "representation"
                        ]["kind"],
                        "candidate_representation_json": json.dumps(
                            candidate["representation"],
                            allow_nan=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    }
                )
    if len(paired) != screen.EXPECTED_PAIRED_COMPARISONS * 8:
        raise RuntimeError("paired child count does not match candidate grid")
    return paired


def _ratio_summary(log_ratio: float) -> dict[str, float]:
    ratio = math.exp(log_ratio)
    return {
        "log_ratio": log_ratio,
        "ratio": ratio,
        "pct": 100.0 * (ratio - 1.0),
    }


def _arm_nested(
    rows: Sequence[Mapping[str, Any]], arm: str, metric: str
) -> dict[str, list[float]]:
    nested = defaultdict(list)
    seen = set()
    for row in rows:
        if row["candidate_arm"] != arm:
            continue
        key = (row["dataset"], int(row["repeat"]), int(row["fold"]))
        if key in seen:
            raise RuntimeError(f"duplicate paired screen row: {key} {arm}")
        seen.add(key)
        nested[row["dataset"]].append(float(row[f"{metric}_log_ratio"]))
    expected_datasets = set(screen.ARM_SPECS[arm]["datasets"])
    if set(nested) != expected_datasets or any(len(values) != 3 for values in nested.values()):
        raise RuntimeError(f"{arm} paired screen scope is incomplete")
    return dict(nested)


def _equal_dataset_log_ratio(
    rows: Sequence[Mapping[str, Any]], arm: str, metric: str
) -> tuple[float, dict[str, float]]:
    nested = _arm_nested(rows, arm, metric)
    datasets = {
        dataset: math.fsum(values) / len(values)
        for dataset, values in nested.items()
    }
    return math.fsum(datasets.values()) / len(datasets), datasets


def hierarchical_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    metric: str = "test_rmse",
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    nested = _arm_nested(rows, arm, metric)
    datasets = sorted(nested)
    rng = np.random.default_rng(seed)
    out = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled_datasets = rng.integers(0, len(datasets), size=len(datasets))
        values = []
        for dataset_index in sampled_datasets:
            splits = np.asarray(nested[datasets[int(dataset_index)]])
            sampled_splits = rng.integers(0, len(splits), size=len(splits))
            values.append(float(np.mean(splits[sampled_splits])))
        out[draw] = math.fsum(values) / len(values)
    return out


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def analyze(
    paired_rows: Sequence[Mapping[str, Any]],
    paired_children: Sequence[Mapping[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    arm_summaries = []
    per_repeat = []
    for arm_index, arm in enumerate(screen.CANDIDATE_ARMS):
        metric_summaries = {}
        dataset_logs = None
        for metric in METRICS:
            point, logs = _equal_dataset_log_ratio(paired_rows, arm, metric)
            metric_summaries[metric] = _ratio_summary(point)
            if metric == "test_rmse":
                dataset_logs = logs
        assert dataset_logs is not None
        boot = hierarchical_bootstrap(
            paired_rows,
            arm,
            draws=draws,
            seed=seed + arm_index,
        )
        datasets = []
        for dataset in sorted(dataset_logs):
            selected = [
                row
                for row in paired_rows
                if row["candidate_arm"] == arm and row["dataset"] == dataset
            ]
            worst = max(selected, key=lambda row: float(row["test_rmse_ratio"]))
            datasets.append(
                {
                    "dataset": dataset,
                    **_ratio_summary(dataset_logs[dataset]),
                    "split_wins": sum(row["test_rmse_ratio"] < 1.0 for row in selected),
                    "split_losses": sum(row["test_rmse_ratio"] > 1.0 for row in selected),
                    "split_ties": sum(row["test_rmse_ratio"] == 1.0 for row in selected),
                    "worst_split": f"r{worst['repeat']}f{worst['fold']}",
                    "worst_split_ratio": worst["test_rmse_ratio"],
                }
            )
            for row in selected:
                per_repeat.append(
                    {
                        "candidate_arm": arm,
                        "dataset": dataset,
                        "repeat": row["repeat"],
                        "fold": row["fold"],
                        "test_rmse_ratio": row["test_rmse_ratio"],
                        "test_rmse_pct": row["test_rmse_pct"],
                        "val_rmse_ratio": row["val_rmse_ratio"],
                        "val_rmse_pct": row["val_rmse_pct"],
                    }
                )
        children = [
            row for row in paired_children if row["candidate_arm"] == arm
        ]
        stop_counts = Counter(row["candidate_stop_reason"] for row in children)
        mode_counts = Counter(row["candidate_selected_tree_mode"] for row in children)
        lane_counts = Counter(row["candidate_selected_lane"] for row in children)
        linear_active_count = sum(
            row["candidate_linear_residual_active"] for row in children
        )
        representation_counts = Counter(
            row["candidate_representation_kind"] for row in children
        )
        max_dataset_ratio = max(item["ratio"] for item in datasets)
        dataset_wins = sum(item["ratio"] < 1.0 for item in datasets)
        dataset_losses = sum(item["ratio"] > 1.0 for item in datasets)
        dataset_ties = sum(item["ratio"] == 1.0 for item in datasets)
        majority_wins_required = len(datasets) // 2 + 1
        time_limit_stops = stop_counts.get("time_limit", 0)
        gates = {
            "complete_declared_scope": True,
            "test_ratio_at_most_0_995": metric_summaries["test_rmse"]["ratio"]
            <= SCREEN_THRESHOLDS["test_ratio_max"],
            "validation_ratio_at_most_1_002": metric_summaries["val_rmse"]["ratio"]
            <= SCREEN_THRESHOLDS["validation_ratio_max"],
            "no_dataset_point_regret_above_0_5pct": max_dataset_ratio
            <= SCREEN_THRESHOLDS["dataset_regret_max"],
            "majority_of_applicable_datasets_improve": dataset_wins
            >= majority_wins_required,
            "zero_time_limit_stops": time_limit_stops
            <= SCREEN_THRESHOLDS["time_limit_stops_max"],
            "train_time_ratio_at_most_4": metric_summaries["train_time_s"]["ratio"]
            <= SCREEN_THRESHOLDS["train_time_ratio_max"],
            "infer_time_ratio_at_most_1_25": metric_summaries["infer_time_s"]["ratio"]
            <= SCREEN_THRESHOLDS["infer_time_ratio_max"],
            "peak_memory_ratio_at_most_2": metric_summaries[
                "peak_memory_bytes"
            ]["ratio"]
            <= SCREEN_THRESHOLDS["peak_memory_ratio_max"],
        }
        gates["survives_exploratory_screen"] = all(gates.values())
        arm_summaries.append(
            {
                "arm": arm,
                "scope_datasets": list(screen.ARM_SPECS[arm]["datasets"]),
                "paired_splits": len(
                    [row for row in paired_rows if row["candidate_arm"] == arm]
                ),
                "paired_children": len(children),
                "metrics": metric_summaries,
                "test_bootstrap": {
                    "draws": draws,
                    "seed": seed + arm_index,
                    "ratio_lower95": math.exp(float(np.quantile(boot, 0.025))),
                    "ratio_upper95": math.exp(float(np.quantile(boot, 0.975))),
                    "ratio_upper95_one_sided": math.exp(float(np.quantile(boot, 0.95))),
                },
                "dataset_wins": dataset_wins,
                "dataset_losses": dataset_losses,
                "dataset_ties": dataset_ties,
                "majority_wins_required": majority_wins_required,
                "datasets": datasets,
                "child_metadata": {
                    "stop_reason_counts": dict(sorted(stop_counts.items())),
                    "selected_tree_mode_counts": dict(sorted(mode_counts.items())),
                    "selected_lane_counts": dict(sorted(lane_counts.items())),
                    "selected_lane_rates": {
                        lane: count / len(children)
                        for lane, count in sorted(lane_counts.items())
                    },
                    "linear_residual_active_count": linear_active_count,
                    "linear_residual_active_rate": linear_active_count
                    / len(children),
                    "representation_counts": dict(sorted(representation_counts.items())),
                    "best_iteration": _distribution(
                        [row["candidate_best_iteration"] for row in children]
                    ),
                    "rounds_completed": _distribution(
                        [row["candidate_rounds_completed"] for row in children]
                    ),
                },
                "gates": gates,
            }
        )
    survivors = [
        item["arm"]
        for item in arm_summaries
        if item["gates"]["survives_exploratory_screen"]
    ]
    summary = {
        "protocol": "frozen isolated TabArena regression follow-on screen",
        "thresholds": dict(SCREEN_THRESHOLDS),
        "counts": {
            "shared_control_jobs": screen.EXPECTED_CONTROL_JOBS,
            "candidate_jobs": screen.EXPECTED_CANDIDATE_JOBS,
            "outer_jobs": screen.EXPECTED_JOBS,
            "child_fits": screen.EXPECTED_CHILD_FITS,
            "paired_comparisons": screen.EXPECTED_PAIRED_COMPARISONS,
        },
        "arms": arm_summaries,
        "survivors": survivors,
        "next_stage": (
            "Freeze each survivor independently, then prefer genuinely unseen "
            "datasets. The other 27 panel coordinates are mechanism-level "
            "holdouts only and were already used for cap selection; do not "
            "combine survivors in this screen."
        ),
        "exploratory_test_use_disclosure": (
            "Outer test RMSE is used for research screening. These r0f0/r1f1/r2f2 "
            "coordinates cannot be used as confirmation data."
        ),
        "auto_validation_disclosure": (
            "Auto uses child validation to select its tree mode, so validation "
            "is supportive rather than independent; outer test drives screening."
        ),
    }
    return summary, per_repeat


def _csv_bytes(rows: Sequence[Mapping[str, Any]], field: str) -> bytes:
    if not rows:
        raise RuntimeError(f"refusing to write empty {field}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _publish_outputs_atomically(
    outputs: Mapping[str, Path],
    payloads: Mapping[str, bytes],
    *,
    post_write_check,
) -> None:
    """Publish the complete managed analysis generation or roll it all back."""
    if set(outputs) != set(OUTPUT_KEYS) or set(payloads) != set(OUTPUT_KEYS):
        raise RuntimeError("managed analysis output fields are not exact")
    hardened_analysis._atomic_write_group(
        [(outputs[name], payloads[name]) for name in OUTPUT_KEYS],
        post_write_check=post_write_check,
    )


def render_report(summary: Mapping[str, Any]) -> str:
    def counts_text(values: Mapping[str, int]) -> str:
        return ", ".join(f"{name}={count}" for name, count in values.items()) or "none"

    def distribution_text(values: Mapping[str, float | int]) -> str:
        return (
            f"n={values['count']}, min={values['min']:.0f}, "
            f"median={values['median']:.1f}, p90={values['p90']:.1f}, "
            f"max={values['max']:.0f}"
        )

    lines = [
        "# DarkoFit isolated regression follow-on screen",
        "",
        "This is an exploratory shared-control screen. Each candidate changes one "
        "declared lever and is compared only with the same native baseline outer "
        "coordinates. Negative percentages favor the candidate.",
        "",
        "| Arm | Scope | Test RMSE | Validation RMSE | Train time | Infer time | "
        "Peak memory | Worst dataset | Survivor |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for item in summary["arms"]:
        metrics = item["metrics"]
        worst = max(dataset["ratio"] for dataset in item["datasets"])
        lines.append(
            "| {arm} | {scope} | {test:+.3f}% | {val:+.3f}% | {train:+.1f}% | "
            "{infer:+.1f}% | {memory:+.1f}% | {worst:+.3f}% | {survivor} |".format(
                arm=item["arm"],
                scope=len(item["scope_datasets"]),
                test=metrics["test_rmse"]["pct"],
                val=metrics["val_rmse"]["pct"],
                train=metrics["train_time_s"]["pct"],
                infer=metrics["infer_time_s"]["pct"],
                memory=metrics["peak_memory_bytes"]["pct"],
                worst=100.0 * (worst - 1.0),
                survivor="yes" if item["gates"]["survives_exploratory_screen"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Per-arm details",
            "",
        ]
    )
    for item in summary["arms"]:
        boot = item["test_bootstrap"]
        lines.extend(
            [
                f"### {item['arm']}",
                "",
                f"Scope: {', '.join(item['scope_datasets'])}.",
                "",
                f"Hierarchical 95% test-RMSE ratio interval: "
                f"{boot['ratio_lower95']:.6f} to {boot['ratio_upper95']:.6f}; "
                f"one-sided upper 95% bound {boot['ratio_upper95_one_sided']:.6f}.",
                "",
                f"Dataset point-estimate wins/losses/ties: "
                f"{item['dataset_wins']}/{item['dataset_losses']}/"
                f"{item['dataset_ties']} (majority requires "
                f"{item['majority_wins_required']} wins).",
                "",
                "Selected child lanes: "
                + ", ".join(
                    f"{lane}={count} ({item['child_metadata']['selected_lane_rates'][lane]:.1%})"
                    for lane, count in item["child_metadata"][
                        "selected_lane_counts"
                    ].items()
                )
                + "; linear-residual active in "
                + f"{item['child_metadata']['linear_residual_active_count']}/"
                + f"{item['paired_children']} children "
                + f"({item['child_metadata']['linear_residual_active_rate']:.1%}).",
                "",
                "Selected tree modes: "
                + counts_text(item["child_metadata"]["selected_tree_mode_counts"])
                + ".",
                "",
                "Stop reasons: "
                + counts_text(item["child_metadata"]["stop_reason_counts"])
                + ".",
                "",
                "Representations: "
                + counts_text(item["child_metadata"]["representation_counts"])
                + ".",
                "",
                "Best iteration distribution: "
                + distribution_text(item["child_metadata"]["best_iteration"])
                + ".",
                "",
                "Completed-round distribution: "
                + distribution_text(item["child_metadata"]["rounds_completed"])
                + ".",
                "",
                "| Dataset | Test RMSE change | Wins | Worst split |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for dataset in item["datasets"]:
            lines.append(
                f"| {dataset['dataset']} | {dataset['pct']:+.3f}% | "
                f"{dataset['split_wins']}/3 | {dataset['worst_split']} "
                f"({100.0 * (dataset['worst_split_ratio'] - 1.0):+.3f}%) |"
            )
        lines.extend(["", "Gates:", ""])
        for name, passed in item["gates"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
        lines.append("")
    lines.extend(
        [
            "## Decision boundary",
            "",
            f"Survivors: {', '.join(summary['survivors']) or 'none'}.",
            "",
            summary["next_stage"],
            "",
            summary["exploratory_test_use_disclosure"],
            "",
            summary["auto_validation_disclosure"],
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve(strict=True)
    manifest_path = (input_dir / screen.MANIFEST_FILENAME).resolve(strict=True)
    attestation_path = (
        input_dir / screen.COMPLETION_ATTESTATION_FILENAME
    ).resolve(strict=True)
    manifest, attestation, payload, digests = verify_campaign_integrity(input_dir)
    expected_ordering = {
        arm: {
            "candidate_before": (len(screen.expected_arm_coordinates(arm)) + 1) // 2,
            "candidate_after": len(screen.expected_arm_coordinates(arm)) // 2,
        }
        for arm in screen.CANDIDATE_ARMS
    }
    if manifest.get("ordering_balance") != expected_ordering:
        raise RuntimeError("manifest ordering balance does not match frozen reversal")

    requested_outputs = {
        "split_csv": input_dir / OUTPUT_NAMES[0],
        "repeat_csv": input_dir / OUTPUT_NAMES[1],
        "child_csv": input_dir / OUTPUT_NAMES[2],
        "summary_json": input_dir / OUTPUT_NAMES[3],
        "report_md": input_dir / OUTPUT_NAMES[4],
    }
    protected = hardened_analysis._protected_campaign_paths(
        input_dir,
        manifest_path=manifest_path,
        attestation_path=attestation_path,
        attestation=attestation,
    )
    outputs = hardened_analysis._canonical_output_targets(
        input_dir, requested_outputs, protected_paths=protected
    )
    paired = pair_outer_rows(payload["outer_rows"])
    paired_children = pair_child_rows(payload["child_rows"])
    summary, per_repeat = analyze(paired, paired_children)
    summary["integrity_diagnostics"] = {
        "validation_basis": (
            "completion attestation, runner normalization, analyzer exact-grid "
            "and representation-safety revalidation"
        ),
        "expected_outer_jobs": screen.EXPECTED_JOBS,
        "observed_outer_jobs": screen.EXPECTED_JOBS,
        "missing_outer_jobs": 0,
        "failed_outer_jobs": 0,
        "imputed_outer_jobs": 0,
        "duplicate_outer_jobs": 0,
        "expected_child_fits": screen.EXPECTED_CHILD_FITS,
        "observed_child_fits": screen.EXPECTED_CHILD_FITS,
        "missing_child_metadata": 0,
        "duplicate_child_metadata": 0,
        "deadline_hit_auto_candidates": 0,
    }
    summary["provenance"] = {
        **digests,
        "manifest_path": str(manifest_path),
        "attestation_path": str(attestation_path),
        "protocol_sha256": screen.protocol_sha256(),
        "git_head": manifest["source"]["git_head"],
        "completed_at_utc": attestation.get("completed_at_utc"),
    }
    payloads = {
        "split_csv": _csv_bytes(paired, "paired split CSV"),
        "repeat_csv": _csv_bytes(per_repeat, "per-repeat CSV"),
        "child_csv": _csv_bytes(paired_children, "paired child CSV"),
        "summary_json": (
            json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "report_md": render_report(summary).encode("utf-8"),
    }
    outputs = hardened_analysis._canonical_output_targets(
        input_dir, outputs, protected_paths=protected
    )

    baseline_snapshot = (manifest, attestation, payload, digests)

    def assert_campaign_unchanged() -> None:
        current = verify_campaign_integrity(input_dir)
        if current != baseline_snapshot:
            raise RuntimeError("campaign artifacts changed during analysis")

    assert_campaign_unchanged()
    _publish_outputs_atomically(
        outputs,
        payloads,
        post_write_check=assert_campaign_unchanged,
    )
    print(
        f"analyzed {screen.EXPECTED_JOBS} jobs and "
        f"{screen.EXPECTED_CHILD_FITS} child fits; survivors="
        f"{summary['survivors']}; wrote {outputs['summary_json']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
