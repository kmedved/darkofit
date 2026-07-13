"""Analyze the frozen 1,000-vs-10,000-round TabArena campaign.

The runner validates its locally generated TabArena pickles and emits a
non-executable JSON snapshot containing only the fields needed for analysis.
This analyzer verifies the manifest, completion attestation, every raw-result
digest, and that JSON snapshot before it computes any decision statistic.  It
never unpickles data supplied through ``--input-dir``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import math
import os
import platform
import stat
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np

try:
    from benchmarks.run_tabarena_regression_cap_horizon import (
        ANALYSIS_PAYLOAD_FILENAME,
        COMPLETION_ATTESTATION_FILENAME,
        DEFAULT_ANALYSIS_OUTPUT_FILENAMES,
        EXPECTED_CHILD_FITS,
        EXPECTED_JOBS,
        HORIZON_ARMS,
        MANIFEST_FILENAME,
        PACKAGE_DISTRIBUTIONS,
        RESUME_HISTORY_FILENAME,
        RUNTIME_ENVIRONMENT_KEYS,
        SOURCE_FILES,
        TASK_SPLIT_COUNTS,
        TIME_LIMIT_SECONDS,
        WARMUP_HISTORY_FILENAME,
        collect_runtime_hardware_provenance,
        expected_ag_ensemble_config,
        expected_child_hyperparameters,
        expected_fit_kwargs_extra,
        expected_resolved_method_hyperparameters,
        frozen_protocol,
        protocol_sha256,
        _sanitize_git_remote,
        _validate_resume_history,
        _validate_warmup_history,
    )
except ModuleNotFoundError:  # Direct execution: python benchmarks/analyze_*.py
    from run_tabarena_regression_cap_horizon import (
        ANALYSIS_PAYLOAD_FILENAME,
        COMPLETION_ATTESTATION_FILENAME,
        DEFAULT_ANALYSIS_OUTPUT_FILENAMES,
        EXPECTED_CHILD_FITS,
        EXPECTED_JOBS,
        HORIZON_ARMS,
        MANIFEST_FILENAME,
        PACKAGE_DISTRIBUTIONS,
        RESUME_HISTORY_FILENAME,
        RUNTIME_ENVIRONMENT_KEYS,
        SOURCE_FILES,
        TASK_SPLIT_COUNTS,
        TIME_LIMIT_SECONDS,
        WARMUP_HISTORY_FILENAME,
        collect_runtime_hardware_provenance,
        expected_ag_ensemble_config,
        expected_child_hyperparameters,
        expected_fit_kwargs_extra,
        expected_resolved_method_hyperparameters,
        frozen_protocol,
        protocol_sha256,
        _sanitize_git_remote,
        _validate_resume_history,
        _validate_warmup_history,
    )


BOOTSTRAP_SEED = 20_260_713
BOOTSTRAP_DRAWS = 10_000
ARMS = ("cap1000", "cap10000")
METRICS = (
    "test_rmse",
    "val_rmse",
    "train_time_s",
    "infer_time_s",
    "peak_memory_bytes",
)
THRESHOLDS = {
    "test_ratio_max": 0.995,
    "bootstrap_upper95_max": 1.0,
    "dataset_wins_min": 10,
    "sign_test_p_max": 0.05,
    "conditional_dataset_point_harm": 1.005,
    "dataset_hard_ratio_max": 1.02,
    "validation_ratio_max": 1.002,
    "paired_10k_over_1000_fraction_min": 0.20,
    "time_limit_stops_max": 0,
    "train_time_ratio_max": 2.0,
    "infer_time_ratio_max": 1.10,
    "peak_memory_ratio_max": 1.10,
}
REQUIRED_REFIT_PARAMS = {
    "iterations",
    "learning_rate",
    "tree_mode",
    "early_stopping",
    "early_stopping_rounds",
    "use_best_model",
    "refit",
    "depth",
    "num_leaves",
    "l2_leaf_reg",
    "min_child_samples",
    "min_child_weight",
    "cat_smoothing",
}
REQUIRED_FIT_METADATA = {
    "iterations_requested",
    "iterations_attempted",
    "rounds_completed",
    "rounds_retained",
    "best_iteration",
    "resolved_learning_rate",
    "requested_tree_mode",
    "selected_tree_mode",
    "selected_lane",
    "linear_residual_active",
    "early_stopping_rounds",
    "stop_reason",
    "wall_clock_limit_seconds",
    "wall_clock_safety_margin_seconds",
    "wall_clock_effective_seconds",
    "wall_clock_elapsed_seconds",
    "deadline_hit",
    "deadline_is_soft",
}
VALID_STOP_REASONS = {"iteration_limit", "early_stopping", "no_split", "time_limit"}
STOP_REASON_ORDER = ("iteration_limit", "early_stopping", "no_split", "time_limit")
EXECUTING_REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT_TARGET_NAMES = (
    "split_csv",
    "repeat_csv",
    "child_csv",
    "summary_json",
    "report_md",
)
DECISION_OUTPUT_NAMES = ("summary_json", "report_md")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_stable(path: Path, field: str) -> bytes:
    try:
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise RuntimeError(f"could not read {field}: {path}") from exc
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError(f"{field} changed while being read: {path}")
    return payload


def _git_output(repository: Path, args: Sequence[str], field: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"could not inspect {field} at {repository}") from exc
    return result.stdout.strip()


def _git_hash_payload(repository: Path, payload: bytes, field: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "hash-object", "--stdin"],
            input=payload,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"could not hash {field} as a Git blob") from exc
    return result.stdout.decode("ascii").strip()


def _nul_paths(payload: bytes) -> list[str]:
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in payload.split(b"\0")
        if item
    ]


def _git_path_list(repository: Path, args: Sequence[str], field: str) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"could not inspect {field} at {repository}") from exc
    return _nul_paths(result.stdout)


def _is_within_relative_path(path: str, parent: Path | None) -> bool:
    if parent is None:
        return False
    candidate = Path(path)
    return candidate == parent or parent in candidate.parents


def _repository_changes(repository: Path, input_dir: Path) -> list[str]:
    """Return code-tree changes while ignoring only untracked run output.

    The result directory can live below the repository and is necessarily
    created after the clean run manifest.  Tracked changes are never ignored;
    only untracked files lexically contained by the resolved input directory
    are excluded.
    """
    tracked = _git_path_list(
        repository,
        ["diff", "--name-only", "--no-ext-diff", "--no-renames", "-z", "HEAD", "--"],
        "tracked repository changes",
    )
    untracked = _git_path_list(
        repository,
        ["ls-files", "--others", "--exclude-standard", "-z", "--"],
        "untracked repository files",
    )
    try:
        output_relative = input_dir.resolve().relative_to(repository.resolve())
    except ValueError:
        output_relative = None
    if output_relative == Path("."):
        # Never exempt the whole repository merely because a malformed
        # manifest calls it the output directory.
        output_relative = None
    relevant_untracked = [
        path
        for path in untracked
        if not _is_within_relative_path(path, output_relative)
    ]
    return [
        *(f"tracked:{path}" for path in tracked),
        *(f"untracked:{path}" for path in relevant_untracked),
    ]


def _manifest_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{field} must be a nonempty canonical path")
    try:
        return Path(value).resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{field} does not resolve: {value!r}") from exc


def _verify_repository_source(
    source: Mapping[str, Any],
    input_dir: Path,
    *,
    repository: Path = EXECUTING_REPOSITORY,
) -> dict[str, Any]:
    """Bind analysis to the exact committed source tree that produced the run."""
    expected_source_fields = {
        "repository",
        "git_head",
        "git_tree",
        "relevant_status",
        "files",
        "darkofit_import",
        "tabarena",
    }
    if set(source) != expected_source_fields:
        raise RuntimeError("run manifest source fields are incomplete")

    repository = repository.resolve(strict=True)
    recorded_repository = _manifest_path(
        source.get("repository"), "run manifest source repository"
    )
    if recorded_repository != repository:
        raise RuntimeError(
            "executing analyzer repository does not match the run manifest"
        )
    if source.get("relevant_status") != "":
        raise RuntimeError("run manifest recorded a dirty campaign repository")

    files = _as_mapping(source.get("files"), "run manifest source files")
    expected_files = {str(path) for path in SOURCE_FILES}
    if set(files) != expected_files:
        raise RuntimeError("run manifest source file set is not exact")
    for relative in SOURCE_FILES:
        key = str(relative)
        metadata = _as_mapping(files[key], f"source metadata for {key}")
        if set(metadata) != {"sha256", "git_blob"}:
            raise RuntimeError(f"source metadata is incomplete for {key}")
        path = (repository / relative).resolve(strict=True)
        try:
            path.relative_to(repository)
        except ValueError as exc:
            raise RuntimeError(
                f"source file escapes executing repository: {key}"
            ) from exc
        payload = _read_stable(path, f"executing source {key}")
        if metadata.get("sha256") != _sha256(payload):
            raise RuntimeError(f"source SHA-256 mismatch for {key}")
        if metadata.get("git_blob") != _git_hash_payload(repository, payload, key):
            raise RuntimeError(f"source Git-blob mismatch for {key}")

    current_head = _git_output(repository, ["rev-parse", "HEAD"], "Git HEAD")
    current_tree = _git_output(
        repository, ["rev-parse", "HEAD^{tree}"], "Git tree"
    )
    if source.get("git_head") != current_head:
        raise RuntimeError("executing Git HEAD does not match the run manifest")
    if source.get("git_tree") != current_tree:
        raise RuntimeError("executing Git tree does not match the run manifest")
    changes = _repository_changes(repository, input_dir)
    if changes:
        raise RuntimeError(
            "executing campaign repository has dirty or unrecorded code: "
            + ", ".join(changes)
        )
    return {
        "executing_repository": str(repository),
        "executing_git_head": current_head,
        "executing_git_tree": current_tree,
    }


def _current_dependency_provenance(
    module_name: str, input_dir: Path
) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"could not import recorded dependency {module_name}"
        ) from exc
    module_file = Path(module.__file__).resolve(strict=True)
    repository = Path(
        _git_output(
            module_file.parent,
            ["rev-parse", "--show-toplevel"],
            f"{module_name} repository",
        )
    ).resolve(strict=True)
    changes = _repository_changes(repository, input_dir)
    return {
        "module": module_name,
        "module_file": str(module_file),
        "repository": str(repository),
        "git_head": _git_output(repository, ["rev-parse", "HEAD"], "Git HEAD"),
        "git_tree": _git_output(
            repository, ["rev-parse", "HEAD^{tree}"], "Git tree"
        ),
        "git_remote_origin": _sanitize_git_remote(
            _git_output(
                repository, ["remote", "get-url", "origin"], "Git origin"
            )
        ),
        "status": "" if not changes else ", ".join(changes),
    }


def _verify_dependency_provenance(
    recorded: Any,
    module_name: str,
    input_dir: Path,
    *,
    required_repository: Path | None = None,
) -> None:
    recorded = dict(_as_mapping(recorded, f"recorded {module_name} provenance"))
    expected_fields = {
        "module",
        "module_file",
        "repository",
        "git_head",
        "git_tree",
        "git_remote_origin",
        "status",
    }
    if set(recorded) != expected_fields:
        raise RuntimeError(f"recorded {module_name} provenance fields are incomplete")
    if recorded.get("status") != "":
        raise RuntimeError(f"run manifest recorded dirty {module_name} source")
    current = _current_dependency_provenance(module_name, input_dir)
    if current["status"]:
        raise RuntimeError(
            f"executing {module_name} dependency has dirty or unrecorded code: "
            f"{current['status']}"
        )
    for field in ("module_file", "repository"):
        recorded_path = _manifest_path(
            recorded.get(field), f"recorded {module_name} {field}"
        )
        if recorded_path != Path(current[field]).resolve(strict=True):
            raise RuntimeError(
                f"executing {module_name} {field} does not match the run manifest"
            )
    if required_repository is not None and Path(current["repository"]).resolve() != (
        required_repository.resolve()
    ):
        raise RuntimeError(
            f"executing {module_name} is not imported from the analyzer repository"
        )
    for field in ("module", "git_head", "git_tree", "git_remote_origin"):
        if recorded.get(field) != current[field]:
            raise RuntimeError(
                f"executing {module_name} {field} does not match the run manifest"
            )


def _current_runtime_provenance() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            packages[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "environment": {
            key: os.environ.get(key) for key in RUNTIME_ENVIRONMENT_KEYS
        },
        "hardware": collect_runtime_hardware_provenance(),
    }


def _verify_runtime_provenance(recorded: Any) -> None:
    recorded = dict(_as_mapping(recorded, "run manifest runtime"))
    current = _current_runtime_provenance()
    if set(recorded) != set(current):
        raise RuntimeError("run manifest runtime fields are incomplete")
    for field in ("python_executable", "python_version", "platform", "machine"):
        if recorded.get(field) != current[field]:
            raise RuntimeError(
                f"analysis runtime {field} does not match the run manifest"
            )

    recorded_packages = _as_mapping(recorded.get("packages"), "runtime packages")
    if set(recorded_packages) != set(current["packages"]):
        raise RuntimeError("run manifest runtime package set is not exact")
    for distribution, version in current["packages"].items():
        if recorded_packages.get(distribution) != version:
            raise RuntimeError(
                f"analysis runtime package {distribution} does not match the "
                "run manifest"
            )

    recorded_environment = _as_mapping(
        recorded.get("environment"), "runtime environment"
    )
    if set(recorded_environment) != set(current["environment"]):
        raise RuntimeError("run manifest runtime environment set is not exact")
    for key, value in current["environment"].items():
        if recorded_environment.get(key) != value:
            raise RuntimeError(
                f"analysis runtime environment {key} does not match the run manifest"
            )

    recorded_hardware = _as_mapping(
        recorded.get("hardware"), "runtime hardware provenance"
    )
    if dict(recorded_hardware) != current["hardware"]:
        raise RuntimeError(
            "analysis runtime hardware does not match the run manifest"
        )


def verify_execution_provenance(
    manifest: Mapping[str, Any], input_dir: Path
) -> dict[str, Any]:
    """Verify that analysis executes from the run's exact source and runtime."""
    source = _as_mapping(manifest.get("source"), "run manifest source")
    diagnostics = _verify_repository_source(source, input_dir)
    _verify_dependency_provenance(
        source.get("darkofit_import"),
        "darkofit",
        input_dir,
        required_repository=EXECUTING_REPOSITORY,
    )
    _verify_dependency_provenance(source.get("tabarena"), "tabarena", input_dir)
    _verify_runtime_provenance(manifest.get("runtime"))
    return {
        **diagnostics,
        "executing_source_verified": True,
        "analysis_runtime_verified": True,
        "dependency_provenance_verified": True,
    }


def _as_mapping(value: Any, field: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field} must be a mapping")
    return value


def _positive_finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise RuntimeError(f"{field} must be positive and finite")
    return number


def _nonnegative_finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or number < 0.0:
        raise RuntimeError(f"{field} must be nonnegative and finite")
    return number


def _exact_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{field} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{field} must be an integer") from exc
    if value != number:
        raise RuntimeError(f"{field} must be an integer")
    return number


def _validate_stop_reason_causality(
    reason: Any,
    *,
    requested: int,
    attempted: int,
    completed: int,
    field: str,
) -> None:
    """Reject counters that cannot causally produce their recorded stop reason."""
    if reason not in VALID_STOP_REASONS:
        raise RuntimeError(f"{field} has an invalid stop reason")
    if reason == "iteration_limit" and attempted != requested:
        raise RuntimeError(
            f"{field} iteration_limit requires all requested iterations attempted"
        )
    if reason == "time_limit" and attempted >= requested:
        raise RuntimeError(
            f"{field} time_limit requires stopping before the iteration limit"
        )
    if reason == "no_split" and attempted <= completed:
        raise RuntimeError(
            f"{field} no_split requires a failed attempted iteration"
        )
    if reason == "early_stopping" and (attempted == 0 or completed == 0):
        raise RuntimeError(
            f"{field} early_stopping requires at least one completed iteration"
        )


def _autogluon_compressed_refit_iterations(
    child_iterations: Sequence[Any], *, field: str
) -> int:
    """Mirror AutoGluon's integer compression: round(mean(child values))."""
    values = [
        _exact_int(value, f"{field}[{index}]")
        for index, value in enumerate(child_iterations)
    ]
    if len(values) != 8:
        raise RuntimeError(f"{field} must contain exactly eight child values")
    return round(statistics.mean(values))


def _validate_compressed_refit_iterations(
    compressed: Mapping[str, Any],
    child_iterations: Sequence[Any],
    *,
    field: str,
) -> int:
    observed = _exact_int(compressed.get("iterations"), f"{field}.iterations")
    expected = _autogluon_compressed_refit_iterations(
        child_iterations, field=f"{field}.child_iterations"
    )
    if observed != expected:
        raise RuntimeError(
            f"{field}.iterations does not match AutoGluon's child aggregation"
        )
    return observed


def _validate_frozen_refit_params(
    value: Any,
    field: str,
    *,
    max_iterations: int,
    expected_iterations: int | None = None,
) -> dict[str, Any]:
    """Validate and normalize every field in a CatBoost refit policy."""
    params = dict(_as_mapping(value, field))
    if set(params) != REQUIRED_REFIT_PARAMS:
        raise RuntimeError(f"{field} fields are not exact")
    iterations = _exact_int(params["iterations"], f"{field}.iterations")
    if not 0 <= iterations <= max_iterations:
        raise RuntimeError(f"{field}.iterations is outside the frozen horizon")
    if expected_iterations is not None and iterations != expected_iterations:
        raise RuntimeError(f"{field}.iterations does not match the best prefix")
    if _exact_int(params["depth"], f"{field}.depth") != 6:
        raise RuntimeError(f"{field}.depth does not match the frozen policy")
    if params["num_leaves"] is not None:
        raise RuntimeError(f"{field}.num_leaves does not match the frozen policy")
    if _exact_int(
        params["min_child_samples"], f"{field}.min_child_samples"
    ) != 20:
        raise RuntimeError(
            f"{field}.min_child_samples does not match the frozen policy"
        )
    for name, expected in {
        "learning_rate": 0.1,
        "l2_leaf_reg": 3.0,
        "min_child_weight": 1.0,
        "cat_smoothing": 1.0,
    }.items():
        if _positive_finite(params[name], f"{field}.{name}") != expected:
            raise RuntimeError(f"{field}.{name} does not match the frozen policy")
    for name, expected in {
        "tree_mode": "catboost",
        "early_stopping": False,
        "early_stopping_rounds": None,
        "use_best_model": False,
        "refit": False,
    }.items():
        if params[name] is not expected and params[name] != expected:
            raise RuntimeError(f"{field}.{name} does not match the frozen policy")
        if isinstance(expected, bool) and params[name] is not expected:
            raise RuntimeError(f"{field}.{name} must be a boolean")
    return {
        **params,
        "iterations": iterations,
        "depth": 6,
        "min_child_samples": 20,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3.0,
        "min_child_weight": 1.0,
        "cat_smoothing": 1.0,
    }


def _validate_resolved_method_metadata(
    method: Mapping[str, Any], arm: str, *, field: str
) -> dict[str, Any]:
    resolved = _as_mapping(method.get("hyperparameters"), f"{field}.hyperparameters")
    if dict(resolved) != expected_resolved_method_hyperparameters(arm):
        raise RuntimeError(f"{field} resolved hyperparameters do not match")
    fit_kwargs = _as_mapping(
        method.get("fit_kwargs_extra"), f"{field}.fit_kwargs_extra"
    )
    if (
        method.get("model_cls") != "DarkoFitModel"
        or method.get("model_type") != "DARKO"
        or method.get("name_prefix") != "DarkoFit"
        or method.get("init_kwargs_extra") != {}
    ):
        raise RuntimeError(f"{field} resolved model identity does not match")
    resources = {
        name: _exact_int(method.get(name), f"{field}.{name}")
        for name in ("num_cpus", "num_gpus", "num_cpus_child", "num_gpus_child")
    }
    if dict(fit_kwargs) != expected_fit_kwargs_extra(resources["num_cpus"]):
        raise RuntimeError(f"{field} resolved bag settings do not match")
    if (
        resources["num_cpus"] < 1
        or resources["num_cpus_child"] != resources["num_cpus"]
        or resources["num_gpus"] != 0
        or resources["num_gpus_child"] != 0
    ):
        raise RuntimeError(f"{field} resolved resources do not match")
    fit_metadata = _as_mapping(method.get("fit_metadata"), f"{field}.fit_metadata")
    if (
        _exact_int(fit_metadata.get("num_cpus"), f"{field}.fit_metadata.num_cpus")
        != resources["num_cpus"]
        or _exact_int(
            fit_metadata.get("num_gpus"), f"{field}.fit_metadata.num_gpus"
        )
        != 0
        or fit_metadata.get("val_in_fit") is not False
        or fit_metadata.get("unlabeled_in_fit") is not False
    ):
        raise RuntimeError(f"{field} fit resources or validation policy do not match")
    return {
        "resolved_method_hyperparameters": dict(resolved),
        "fit_kwargs_extra": dict(fit_kwargs),
        **resources,
    }


def _validate_resolved_bag_metadata(
    bag: Mapping[str, Any], arm: str, *, field: str
) -> None:
    if (
        _exact_int(bag.get("_n_repeats"), f"{field}._n_repeats") != 1
        or bag.get("_k_per_n_repeat") != [8]
        or _exact_int(bag.get("_random_state"), f"{field}._random_state") != 1
        or bag.get("bagged_mode") is not True
    ):
        raise RuntimeError(f"{field} is not the frozen one-set/eight-fold bag")
    child_user = _as_mapping(
        bag.get("child_hyperparameters_user"),
        f"{field}.child_hyperparameters_user",
    )
    child_resolved = _as_mapping(
        bag.get("child_hyperparameters"), f"{field}.child_hyperparameters"
    )
    if dict(child_user) != HORIZON_ARMS[arm] or dict(
        child_resolved
    ) != expected_child_hyperparameters(arm, 0):
        raise RuntimeError(f"{field} resolved child policy does not match")
    child_ag_args = _as_mapping(
        bag.get("child_ag_args_fit"), f"{field}.child_ag_args_fit"
    )
    if any(
        child_ag_args.get(name) != expected
        for name, expected in {
            "max_memory_usage_ratio": 1.0,
            "max_time_limit_ratio": 1.0,
            "max_time_limit": None,
            "min_time_limit": 0,
        }.items()
    ):
        raise RuntimeError(f"{field} child budget ratios do not match")


def _validate_initial_child_metadata(
    child: Mapping[str, Any],
    arm: str,
    child_fold: int,
    *,
    num_cpus: int,
    num_gpus: int,
    field: str,
) -> dict[str, Any]:
    initial = _as_mapping(child.get("hyperparameters"), f"{field}.hyperparameters")
    user = _as_mapping(
        child.get("hyperparameters_user"), f"{field}.hyperparameters_user"
    )
    if dict(initial) != expected_child_hyperparameters(arm, child_fold):
        raise RuntimeError(f"{field} initialized policy or fold seed does not match")
    if dict(user) != HORIZON_ARMS[arm]:
        raise RuntimeError(f"{field} user hyperparameters do not match")
    if (
        _exact_int(child.get("num_cpus"), f"{field}.num_cpus") != num_cpus
        or _exact_int(child.get("num_gpus"), f"{field}.num_gpus") != num_gpus
        or child.get("problem_type") != "regression"
        or child.get("eval_metric") != "root_mean_squared_error"
        or child.get("stopping_metric") != "root_mean_squared_error"
        or child.get("val_in_fit") is not True
        or child.get("unlabeled_in_fit") is not False
    ):
        raise RuntimeError(f"{field} resources/evaluation policy do not match")
    child_ag_args = _as_mapping(child.get("ag_args_fit"), f"{field}.ag_args_fit")
    if any(
        child_ag_args.get(name) != expected
        for name, expected in {
            "max_memory_usage_ratio": 1.0,
            "max_time_limit_ratio": 1.0,
            "max_time_limit": None,
            "min_time_limit": 0,
        }.items()
    ):
        raise RuntimeError(f"{field} budget ratios do not match")
    return dict(initial)


def _verify_history_artifact(
    input_dir: Path,
    attestation: Mapping[str, Any],
    *,
    attestation_field: str,
    filename: str,
    required: bool,
    validator: Callable[[Any], None],
) -> str | None:
    raw_artifact = attestation.get(attestation_field)
    path = input_dir / filename
    if raw_artifact is None:
        if required or path.exists():
            raise RuntimeError(f"{filename} is not bound by the completion attestation")
        return None
    if path.is_symlink():
        raise RuntimeError(f"{filename} must not be a symbolic link")
    artifact = _as_mapping(raw_artifact, f"attested {filename}")
    if set(artifact) != {"path", "sha256", "size_bytes"}:
        raise RuntimeError(f"attested {filename} fields are incomplete")
    if artifact.get("path") != filename:
        raise RuntimeError(f"attested {filename} path does not match")
    resolved_path = path.resolve(strict=True)
    try:
        resolved_path.relative_to(input_dir.resolve(strict=True))
    except ValueError as exc:
        raise RuntimeError(f"{filename} escapes the campaign directory") from exc
    payload = _read_stable(resolved_path, filename)
    size = _exact_int(artifact.get("size_bytes"), f"{filename} size_bytes")
    digest = artifact.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"attested {filename} digest is invalid")
    if len(payload) != size or _sha256(payload) != digest:
        raise RuntimeError(f"{filename} does not match its completion attestation")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{filename} is not valid JSON") from exc
    validator(value)
    return digest


def verify_campaign_integrity(
    input_dir: Path,
    *,
    manifest_path: Path | None = None,
    attestation_path: Path | None = None,
    expected_protocol: Mapping[str, Any] | None = None,
    expected_jobs: int = EXPECTED_JOBS,
    expected_child_fits: int = EXPECTED_CHILD_FITS,
) -> tuple[dict, dict, dict, dict[str, Any]]:
    """Verify the complete byte set and load only the safe JSON snapshot."""
    input_dir = input_dir.resolve()
    manifest_path = (manifest_path or input_dir / MANIFEST_FILENAME).resolve()
    attestation_path = (
        attestation_path or input_dir / COMPLETION_ATTESTATION_FILENAME
    ).resolve()
    expected_protocol = dict(expected_protocol or frozen_protocol())
    expected_protocol_digest = _sha256(_canonical_json(expected_protocol))

    manifest_payload = _read_stable(manifest_path, "run manifest")
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("run manifest is not valid JSON") from exc
    manifest = dict(_as_mapping(manifest, "run manifest"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("run manifest has an unsupported schema version")
    if manifest.get("kind") != "darkofit_tabarena_regression_cap_horizon":
        raise RuntimeError("run manifest has the wrong campaign kind")
    if Path(str(manifest.get("output_dir", ""))).resolve() != input_dir:
        raise RuntimeError("run manifest output_dir does not match the input directory")
    if manifest.get("protocol") != expected_protocol:
        raise RuntimeError("run manifest does not contain the frozen protocol")
    if manifest.get("protocol_sha256") != expected_protocol_digest:
        raise RuntimeError("run manifest protocol digest does not match")
    if manifest.get("time_limit_seconds") != expected_protocol.get(
        "time_limit_seconds"
    ):
        raise RuntimeError("run manifest time limit does not match the protocol")
    resolved_child_num_cpus = _exact_int(
        manifest.get("resolved_child_num_cpus"),
        "run manifest resolved_child_num_cpus",
    )
    if resolved_child_num_cpus < 1:
        raise RuntimeError("run manifest child CPU allocation must be positive")
    source = _as_mapping(manifest.get("source"), "run manifest source")
    git_head = source.get("git_head")
    if not isinstance(git_head, str) or not git_head:
        raise RuntimeError("run manifest has no Git commit")
    execution_provenance = verify_execution_provenance(manifest, input_dir)
    manifest_digest = _sha256(manifest_payload)

    attestation_payload = _read_stable(attestation_path, "completion attestation")
    try:
        attestation = json.loads(attestation_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("completion attestation is not valid JSON") from exc
    attestation = dict(_as_mapping(attestation, "completion attestation"))
    if attestation.get("schema_version") != 1:
        raise RuntimeError("completion attestation has an unsupported schema version")
    if (
        attestation.get("kind")
        != "darkofit_tabarena_regression_cap_horizon_completion"
    ):
        raise RuntimeError("completion attestation has the wrong campaign kind")
    count_fields = {
        "result_count": expected_jobs,
        "expected_result_count": expected_jobs,
        "expected_child_fits": expected_child_fits,
    }
    for field, expected in count_fields.items():
        if attestation.get(field) != expected:
            raise RuntimeError(f"completion attestation {field} is not {expected}")
    if attestation.get("protocol_sha256") != expected_protocol_digest:
        raise RuntimeError("completion attestation protocol digest does not match")
    if attestation.get("git_head") != git_head:
        raise RuntimeError("completion attestation Git commit does not match")
    if attestation.get("manifest_sha256") != manifest_digest:
        raise RuntimeError("completion attestation does not bind this run manifest")
    validation = _as_mapping(attestation.get("validation"), "attestation validation")
    if validation.get("result_count") != expected_jobs:
        raise RuntimeError("attestation validation result count does not match")
    if validation.get("child_fit_count") != expected_child_fits:
        raise RuntimeError("attestation validation child-fit count does not match")
    resource_allocation = _as_mapping(
        validation.get("resource_allocation"),
        "attestation resource allocation",
    )
    if set(resource_allocation) != {
        "num_cpus",
        "num_gpus",
        "num_cpus_child",
        "num_gpus_child",
    }:
        raise RuntimeError("attestation resource allocation fields are incomplete")
    outer_cpus = _exact_int(resource_allocation["num_cpus"], "attested num_cpus")
    child_cpus = _exact_int(
        resource_allocation["num_cpus_child"], "attested child num_cpus"
    )
    if (
        outer_cpus < 1
        or child_cpus != outer_cpus
        or child_cpus != resolved_child_num_cpus
        or _exact_int(resource_allocation["num_gpus"], "attested num_gpus") != 0
        or _exact_int(
            resource_allocation["num_gpus_child"], "attested child num_gpus"
        )
        != 0
        or _exact_int(
            attestation.get("warmup_thread_count"),
            "attested warmup thread count",
        )
        != child_cpus
    ):
        raise RuntimeError(
            "attested resources, manifest, and warmup thread count do not match"
        )

    artifacts = _as_mapping(
        attestation.get("result_artifacts"), "attested result artifacts"
    )
    if len(artifacts) != expected_jobs:
        raise RuntimeError(
            f"expected {expected_jobs} attested results, found {len(artifacts)}"
        )
    observed_paths = {
        str(path.relative_to(input_dir))
        for path in sorted((input_dir / "experiments").rglob("results.pkl"))
    }
    if observed_paths != set(artifacts):
        raise RuntimeError("on-disk result set does not exactly match the attestation")

    # Raw pickles are hashed for integrity but are deliberately never decoded
    # by this analyzer. Only the runner-created JSON snapshot below is parsed.
    for relative, raw_metadata in sorted(artifacts.items()):
        if not isinstance(relative, str):
            raise RuntimeError("attested result path must be a string")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.name != "results.pkl"
        ):
            raise RuntimeError(f"unsafe attested result path: {relative!r}")
        path = (input_dir / relative_path).resolve()
        try:
            path.relative_to(input_dir)
        except ValueError as exc:
            raise RuntimeError(f"attested result escapes input directory: {relative}") from exc
        metadata = _as_mapping(raw_metadata, f"artifact metadata for {relative}")
        size = _exact_int(metadata.get("size_bytes"), f"{relative}: size_bytes")
        digest = metadata.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"{relative}: invalid SHA-256 digest")
        payload = _read_stable(path, f"result artifact {relative}")
        if len(payload) != size:
            raise RuntimeError(f"attested result size changed: {relative}")
        if _sha256(payload) != digest:
            raise RuntimeError(f"attested result digest changed: {relative}")

    analysis_artifact = _as_mapping(
        attestation.get("analysis_payload_artifact"),
        "attested analysis payload",
    )
    if set(analysis_artifact) != {"path", "sha256", "size_bytes"}:
        raise RuntimeError("analysis payload attestation fields are incomplete")
    if analysis_artifact.get("path") != ANALYSIS_PAYLOAD_FILENAME:
        raise RuntimeError("analysis payload path is not the frozen filename")
    analysis_path = (input_dir / ANALYSIS_PAYLOAD_FILENAME).resolve()
    try:
        analysis_path.relative_to(input_dir)
    except ValueError as exc:
        raise RuntimeError("analysis payload escapes input directory") from exc
    analysis_bytes = _read_stable(analysis_path, "safe analysis payload")
    analysis_size = _exact_int(
        analysis_artifact.get("size_bytes"), "analysis payload size_bytes"
    )
    analysis_digest = analysis_artifact.get("sha256")
    if not isinstance(analysis_digest, str) or len(analysis_digest) != 64:
        raise RuntimeError("analysis payload has an invalid SHA-256 digest")
    if len(analysis_bytes) != analysis_size or _sha256(analysis_bytes) != analysis_digest:
        raise RuntimeError("analysis payload does not match its attestation")
    try:
        analysis_payload = json.loads(analysis_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("analysis payload is not valid JSON") from exc
    analysis_payload = dict(_as_mapping(analysis_payload, "analysis payload"))
    if analysis_payload.get("schema_version") != 1:
        raise RuntimeError("analysis payload has an unsupported schema version")
    if (
        analysis_payload.get("kind")
        != "darkofit_tabarena_regression_cap_horizon_analysis_payload"
    ):
        raise RuntimeError("analysis payload has the wrong campaign kind")
    if analysis_payload.get("protocol_sha256") != expected_protocol_digest:
        raise RuntimeError("analysis payload protocol digest does not match")
    if analysis_payload.get("result_artifacts_sha256") != _sha256(
        _canonical_json(artifacts)
    ):
        raise RuntimeError("analysis payload does not bind the raw result set")

    warmup_history_digest = _verify_history_artifact(
        input_dir,
        attestation,
        attestation_field="warmup_history_artifact",
        filename=WARMUP_HISTORY_FILENAME,
        required=True,
        validator=lambda value: _validate_warmup_history(
            value,
            expected_thread_count=resolved_child_num_cpus,
        ),
    )
    resume_history_digest = _verify_history_artifact(
        input_dir,
        attestation,
        attestation_field="resume_history_artifact",
        filename=RESUME_HISTORY_FILENAME,
        required=False,
        validator=lambda value: _validate_resume_history(value, input_dir),
    )

    return manifest, attestation, analysis_payload, {
        "manifest_sha256": manifest_digest,
        "attestation_sha256": _sha256(attestation_payload),
        "analysis_payload_sha256": analysis_digest,
        "warmup_history_sha256": warmup_history_digest,
        "resume_history_sha256": resume_history_digest,
        **execution_provenance,
    }


def _assert_campaign_snapshot_unchanged(
    input_dir: Path,
    *,
    manifest_path: Path,
    attestation_path: Path,
    baseline_manifest: Mapping[str, Any],
    baseline_attestation: Mapping[str, Any],
    baseline_analysis_payload: Mapping[str, Any],
    baseline_digests: Mapping[str, Any],
) -> None:
    """Revalidate every attested input and require the initial byte snapshot."""
    (
        current_manifest,
        current_attestation,
        current_analysis_payload,
        current_digests,
    ) = verify_campaign_integrity(
        input_dir,
        manifest_path=manifest_path,
        attestation_path=attestation_path,
    )
    if (
        current_manifest != dict(baseline_manifest)
        or current_attestation != dict(baseline_attestation)
        or current_analysis_payload != dict(baseline_analysis_payload)
        or current_digests != dict(baseline_digests)
    ):
        raise RuntimeError(
            "campaign artifacts or execution provenance changed during analysis"
        )


def _identify_arm(
    record: Mapping,
    source: str,
    horizon_arms: Mapping[str, Mapping[str, Any]],
) -> tuple[str, Mapping, Mapping]:
    method = _as_mapping(record.get("method_metadata"), f"{source}: method_metadata")
    raw = dict(
        _as_mapping(
            method.get("model_hyperparameters"),
            f"{source}: model_hyperparameters",
        )
    )
    ag_args = raw.pop("ag_args", None)
    ag_ensemble = raw.pop("ag_args_ensemble", None)
    matches = [name for name, config in horizon_arms.items() if raw == dict(config)]
    if len(matches) != 1:
        raise RuntimeError(f"{source}: result does not match exactly one frozen arm")
    arm = matches[0]
    expected_suffix = f"_c1_{arm}_horizon"
    if ag_args != {"name_suffix": expected_suffix}:
        raise RuntimeError(f"{source}: unexpected AutoGluon name suffix")
    ensemble = _as_mapping(ag_ensemble, f"{source}: ag_args_ensemble")
    if dict(ensemble) != expected_ag_ensemble_config():
        raise RuntimeError(
            f"{source}: bag seed/resource configuration is not frozen"
        )
    if record.get("framework") != f"DarkoFit{expected_suffix}_BAG_L1":
        raise RuntimeError(f"{source}: unexpected framework name")
    return arm, method, raw


def parse_result_record(
    record: Mapping,
    *,
    source: str,
    task_split_counts: Mapping[str, tuple[int, int]] = TASK_SPLIT_COUNTS,
    horizon_arms: Mapping[str, Mapping[str, Any]] = HORIZON_ARMS,
) -> tuple[dict, list[dict]]:
    """Strictly normalize one outer result and its eight child fits."""
    if record.get("problem_type") != "regression" or record.get("metric") != "rmse":
        raise RuntimeError(f"{source}: result is not regression/RMSE")
    imputed = record.get("imputed", False)
    if imputed is not False and imputed is not None:
        raise RuntimeError(f"{source}: result is imputed")
    experiment = _as_mapping(
        record.get("experiment_metadata"), f"{source}: experiment_metadata"
    )
    if (
        experiment.get("experiment_cls") != "OOFExperimentRunner"
        or experiment.get("method_cls") != "AGSingleBagWrapper"
    ):
        raise RuntimeError(f"{source}: unexpected experiment implementation")

    task = _as_mapping(record.get("task_metadata"), f"{source}: task_metadata")
    dataset = task.get("name")
    if dataset not in task_split_counts:
        raise RuntimeError(f"{source}: unexpected dataset {dataset!r}")
    task_id, split_count = task_split_counts[str(dataset)]
    if _exact_int(task.get("tid"), f"{source}: task id") != task_id:
        raise RuntimeError(f"{source}: task id does not match dataset")
    repeat = _exact_int(task.get("repeat"), f"{source}: repeat")
    fold = _exact_int(task.get("fold"), f"{source}: fold")
    registered_fold = 3 * repeat + fold
    if fold not in range(3) or registered_fold not in range(split_count):
        raise RuntimeError(f"{source}: split is outside the frozen coordinate map")
    if task.get("split_idx") is not None and _exact_int(
        task["split_idx"], f"{source}: split_idx"
    ) != registered_fold:
        raise RuntimeError(f"{source}: split_idx does not match repeat/fold")

    arm, method, _ = _identify_arm(record, source, horizon_arms)
    resolved_method = _validate_resolved_method_metadata(
        method,
        arm,
        field=f"{source}: resolved method metadata",
    )
    info = _as_mapping(method.get("info"), f"{source}: model info")
    if info.get("is_valid") is not True or info.get("can_infer") is not True:
        raise RuntimeError(f"{source}: outer model is not valid and inferable")
    if info.get("model_type") != "StackerEnsembleModel":
        raise RuntimeError(f"{source}: unexpected outer model type")
    if (
        _exact_int(info.get("num_cpus"), f"{source}: outer num_cpus")
        != resolved_method["num_cpus"]
        or _exact_int(info.get("num_gpus"), f"{source}: outer num_gpus")
        != resolved_method["num_gpus"]
        or info.get("problem_type") != "regression"
        or info.get("eval_metric") != "root_mean_squared_error"
        or info.get("stopping_metric") != "root_mean_squared_error"
        or info.get("val_in_fit") is not False
        or info.get("unlabeled_in_fit") is not False
    ):
        raise RuntimeError(f"{source}: outer resources/evaluation policy mismatch")
    bag = _as_mapping(info.get("bagged_info"), f"{source}: bagged_info")
    if bag.get("num_child_models") != 8:
        raise RuntimeError(f"{source}: expected eight bag children")
    if bag.get("child_model_type") != "DarkoFitModel":
        raise RuntimeError(f"{source}: unexpected child model type")
    expected_child_names = [f"S1F{index}" for index in range(1, 9)]
    if bag.get("child_model_names") != expected_child_names:
        raise RuntimeError(f"{source}: unexpected bag child names")
    _validate_resolved_bag_metadata(
        bag,
        arm,
        field=f"{source}: bagged_info",
    )
    compressed = _as_mapping(
        bag.get("child_hyperparameters_fit"),
        f"{source}: child_hyperparameters_fit",
    )
    requested_horizon = _exact_int(
        horizon_arms[arm]["iterations"], f"{source}: frozen horizon"
    )
    compressed = _validate_frozen_refit_params(
        compressed,
        f"{source}: compressed refit parameters",
        max_iterations=requested_horizon,
    )
    children = _as_mapping(info.get("children_info"), f"{source}: children_info")
    if set(children) != set(expected_child_names):
        raise RuntimeError(f"{source}: unexpected child set")

    outer = {
        "dataset": str(dataset),
        "task_id": task_id,
        "repeat": repeat,
        "fold": fold,
        "registered_fold": registered_fold,
        "arm": arm,
        "test_rmse": _positive_finite(record.get("metric_error"), f"{source}: RMSE"),
        "val_rmse": _positive_finite(
            record.get("metric_error_val"), f"{source}: validation RMSE"
        ),
        "train_time_s": _positive_finite(
            record.get("time_train_s"), f"{source}: training time"
        ),
        "infer_time_s": _positive_finite(
            record.get("time_infer_s"), f"{source}: inference time"
        ),
        "peak_memory_bytes": _positive_finite(
            _as_mapping(record.get("memory_usage"), f"{source}: memory usage").get(
                "peak_mem_cpu"
            ),
            f"{source}: peak CPU memory",
        ),
        "framework": str(record.get("framework")),
        **resolved_method,
        "bag_folds": 8,
        "bag_sets": 1,
        "bag_random_state": 1,
        "child_ag_args_fit": {
            name: bag["child_ag_args_fit"][name]
            for name in (
                "max_memory_usage_ratio",
                "max_time_limit_ratio",
                "max_time_limit",
                "min_time_limit",
            )
        },
        "compressed_refit_params": compressed,
        "source": source,
    }

    child_rows = []
    child_refit_iterations = []
    for child_name in expected_child_names:
        child = _as_mapping(children[child_name], f"{source}: {child_name}")
        if (
            child.get("name") != child_name
            or child.get("model_type") != "DarkoFitModel"
            or child.get("is_valid") is not True
            or child.get("can_infer") is not True
        ):
            raise RuntimeError(f"{source}: {child_name} is not a valid DarkoFit child")
        child_fold = int(child_name.removeprefix("S1F")) - 1
        initial_hyperparameters = _validate_initial_child_metadata(
            child,
            arm,
            child_fold,
            num_cpus=resolved_method["num_cpus_child"],
            num_gpus=resolved_method["num_gpus_child"],
            field=f"{source}: {child_name}",
        )
        trained = _as_mapping(
            child.get("hyperparameters_fit"),
            f"{source}: {child_name}.hyperparameters_fit",
        )
        fitted = _as_mapping(
            child.get("darkofit_fit"), f"{source}: {child_name}.darkofit_fit"
        )
        if set(fitted) != REQUIRED_FIT_METADATA:
            raise RuntimeError(f"{source}: {child_name} fit metadata incomplete")

        requested = _exact_int(
            fitted["iterations_requested"], f"{source}: {child_name} requested"
        )
        attempted = _exact_int(
            fitted["iterations_attempted"], f"{source}: {child_name} attempted"
        )
        completed = _exact_int(
            fitted["rounds_completed"], f"{source}: {child_name} completed"
        )
        retained = _exact_int(
            fitted["rounds_retained"], f"{source}: {child_name} retained"
        )
        best = _exact_int(
            fitted["best_iteration"], f"{source}: {child_name} best iteration"
        )
        if requested != requested_horizon:
            raise RuntimeError(f"{source}: {child_name} requested the wrong horizon")
        if not (0 <= retained == best <= completed <= attempted <= requested):
            raise RuntimeError(f"{source}: {child_name} round counts are inconsistent")
        trained = _validate_frozen_refit_params(
            trained,
            f"{source}: {child_name} refit parameters",
            expected_iterations=best,
            max_iterations=requested,
        )
        resolved_lr = _positive_finite(
            fitted["resolved_learning_rate"], f"{source}: resolved learning rate"
        )
        if resolved_lr != 0.1 or _positive_finite(
            trained["learning_rate"], f"{source}: refit learning rate"
        ) != 0.1:
            raise RuntimeError(f"{source}: {child_name} resolved the wrong learning rate")
        if (
            fitted["requested_tree_mode"] != "catboost"
            or fitted["selected_tree_mode"] != "catboost"
            or trained["tree_mode"] != "catboost"
            or fitted["selected_lane"] != "boosting"
            or fitted["linear_residual_active"] is not False
            or fitted["early_stopping_rounds"] != 50
            or trained["early_stopping"] is not False
            or trained["early_stopping_rounds"] is not None
            or trained["use_best_model"] is not False
            or trained["refit"] is not False
        ):
            raise RuntimeError(f"{source}: {child_name} resolved policy mismatch")
        reason = fitted["stop_reason"]
        _validate_stop_reason_causality(
            reason,
            requested=requested,
            attempted=attempted,
            completed=completed,
            field=f"{source}: {child_name}",
        )
        if fitted["deadline_is_soft"] is not True:
            raise RuntimeError(f"{source}: {child_name} did not use a soft deadline")
        if fitted["deadline_hit"] is not (reason == "time_limit"):
            raise RuntimeError(f"{source}: {child_name} deadline flag mismatch")
        wall_limit = _positive_finite(
            fitted["wall_clock_limit_seconds"], f"{source}: wall-clock limit"
        )
        wall_margin = _nonnegative_finite(
            fitted["wall_clock_safety_margin_seconds"],
            f"{source}: wall-clock safety margin",
        )
        wall_effective = _nonnegative_finite(
            fitted["wall_clock_effective_seconds"],
            f"{source}: effective wall-clock limit",
        )
        wall_elapsed = _nonnegative_finite(
            fitted["wall_clock_elapsed_seconds"], f"{source}: wall-clock elapsed"
        )
        if (
            wall_margin > wall_limit
            or wall_limit > TIME_LIMIT_SECONDS
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
            raise RuntimeError(f"{source}: {child_name} deadline metadata is inconsistent")

        child_rows.append(
            {
                "dataset": str(dataset),
                "task_id": task_id,
                "repeat": repeat,
                "fold": fold,
                "registered_fold": registered_fold,
                "child": child_name,
                "child_fold": child_fold,
                "arm": arm,
                "iterations_requested": requested,
                "iterations_attempted": attempted,
                "rounds_completed": completed,
                "rounds_retained": retained,
                "best_iteration": best,
                "resolved_learning_rate": resolved_lr,
                "requested_tree_mode": str(fitted["requested_tree_mode"]),
                "selected_tree_mode": str(fitted["selected_tree_mode"]),
                "selected_lane": str(fitted["selected_lane"]),
                "linear_residual_active": bool(fitted["linear_residual_active"]),
                "early_stopping_rounds": int(fitted["early_stopping_rounds"]),
                "stop_reason": str(reason),
                "wall_clock_limit_seconds": wall_limit,
                "wall_clock_safety_margin_seconds": wall_margin,
                "wall_clock_effective_seconds": wall_effective,
                "wall_clock_elapsed_seconds": wall_elapsed,
                "deadline_hit": bool(fitted["deadline_hit"]),
                "deadline_is_soft": bool(fitted["deadline_is_soft"]),
                "initial_hyperparameters": initial_hyperparameters,
                "num_cpus": resolved_method["num_cpus_child"],
                "num_gpus": resolved_method["num_gpus_child"],
                "problem_type": "regression",
                "eval_metric": "root_mean_squared_error",
                "stopping_metric": "root_mean_squared_error",
                "val_in_fit": True,
                "unlabeled_in_fit": False,
                "ag_args_fit": {
                    name: child["ag_args_fit"][name]
                    for name in (
                        "max_memory_usage_ratio",
                        "max_time_limit_ratio",
                        "max_time_limit",
                        "min_time_limit",
                    )
                },
                "source": source,
            }
        )
        child_refit_iterations.append(trained["iterations"])
    _validate_compressed_refit_iterations(
        compressed,
        child_refit_iterations,
        field=f"{source}: compressed refit parameters",
    )
    return outer, child_rows


def load_safe_rows(
    analysis_payload: Mapping[str, Any],
    *,
    task_split_counts: Mapping[str, tuple[int, int]] = TASK_SPLIT_COUNTS,
    horizon_arms: Mapping[str, Mapping[str, Any]] = HORIZON_ARMS,
) -> tuple[list[dict], list[dict]]:
    """Validate the non-executable normalized rows emitted by the runner."""
    raw_outer = analysis_payload.get("outer_rows")
    raw_children = analysis_payload.get("child_rows")
    if not isinstance(raw_outer, list) or not isinstance(raw_children, list):
        raise RuntimeError("analysis payload rows must be JSON arrays")

    outer_fields = {
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
        "imputed",
        "experiment_cls",
        "method_cls",
        "outer_model_type",
        "ag_ensemble",
        "resolved_method_hyperparameters",
        "fit_kwargs_extra",
        "num_cpus",
        "num_gpus",
        "num_cpus_child",
        "num_gpus_child",
        "bag_folds",
        "bag_sets",
        "bag_random_state",
        "child_ag_args_fit",
        "compressed_refit_params",
        "source",
    }
    child_fields = {
        "dataset",
        "task_id",
        "repeat",
        "fold",
        "registered_fold",
        "child",
        "child_fold",
        "arm",
        "iterations_requested",
        "iterations_attempted",
        "rounds_completed",
        "rounds_retained",
        "best_iteration",
        "resolved_learning_rate",
        "requested_tree_mode",
        "selected_tree_mode",
        "selected_lane",
        "linear_residual_active",
        "early_stopping_rounds",
        "stop_reason",
        "wall_clock_limit_seconds",
        "wall_clock_safety_margin_seconds",
        "wall_clock_effective_seconds",
        "wall_clock_elapsed_seconds",
        "deadline_hit",
        "deadline_is_soft",
        "refit_params",
        "initial_hyperparameters",
        "num_cpus",
        "num_gpus",
        "problem_type",
        "eval_metric",
        "stopping_metric",
        "val_in_fit",
        "unlabeled_in_fit",
        "ag_args_fit",
        "source",
    }

    outer_rows: list[dict] = []
    outer_sources: dict[tuple[str, int, int, str], str] = {}
    for index, value in enumerate(raw_outer):
        row = dict(_as_mapping(value, f"outer_rows[{index}]"))
        if set(row) != outer_fields:
            raise RuntimeError(f"outer_rows[{index}] fields are not exact")
        dataset = row["dataset"]
        if dataset not in task_split_counts:
            raise RuntimeError(f"outer_rows[{index}] has an unexpected dataset")
        task_id, split_count = task_split_counts[str(dataset)]
        repeat = _exact_int(row["repeat"], f"outer_rows[{index}].repeat")
        fold = _exact_int(row["fold"], f"outer_rows[{index}].fold")
        registered = _exact_int(
            row["registered_fold"], f"outer_rows[{index}].registered_fold"
        )
        if (
            _exact_int(row["task_id"], f"outer_rows[{index}].task_id") != task_id
            or fold not in range(3)
            or registered != 3 * repeat + fold
            or registered not in range(split_count)
        ):
            raise RuntimeError(f"outer_rows[{index}] coordinate is inconsistent")
        arm = row["arm"]
        if arm not in horizon_arms:
            raise RuntimeError(f"outer_rows[{index}] has an unexpected arm")
        compressed_refit_params = _validate_frozen_refit_params(
            row["compressed_refit_params"],
            f"outer_rows[{index}].compressed_refit_params",
            max_iterations=_exact_int(
                horizon_arms[str(arm)]["iterations"], "frozen horizon"
            ),
        )
        source = row["source"]
        if not isinstance(source, str) or not source.endswith("/results.pkl"):
            raise RuntimeError(f"outer_rows[{index}] has an invalid source")
        if row["framework"] != f"DarkoFit_c1_{arm}_horizon_BAG_L1":
            raise RuntimeError(f"outer_rows[{index}] has an unexpected framework")
        if (
            row["imputed"] is not False
            or row["experiment_cls"] != "OOFExperimentRunner"
            or row["method_cls"] != "AGSingleBagWrapper"
            or row["outer_model_type"] != "StackerEnsembleModel"
        ):
            raise RuntimeError(
                f"outer_rows[{index}] experiment/imputation metadata is invalid"
            )
        bag_configuration = _as_mapping(
            row["ag_ensemble"], f"outer_rows[{index}].ag_ensemble"
        )
        if dict(bag_configuration) != expected_ag_ensemble_config():
            raise RuntimeError(
                f"outer_rows[{index}] bag seed/resource configuration is not frozen"
            )
        num_cpus = _exact_int(row["num_cpus"], f"outer_rows[{index}].num_cpus")
        num_gpus = _exact_int(row["num_gpus"], f"outer_rows[{index}].num_gpus")
        num_cpus_child = _exact_int(
            row["num_cpus_child"], f"outer_rows[{index}].num_cpus_child"
        )
        num_gpus_child = _exact_int(
            row["num_gpus_child"], f"outer_rows[{index}].num_gpus_child"
        )
        if row[
            "resolved_method_hyperparameters"
        ] != expected_resolved_method_hyperparameters(
            str(arm)
        ) or row["fit_kwargs_extra"] != expected_fit_kwargs_extra(num_cpus):
            raise RuntimeError(
                f"outer_rows[{index}] resolved method configuration is not frozen"
            )
        expected_child_ag_args = {
            "max_memory_usage_ratio": 1.0,
            "max_time_limit_ratio": 1.0,
            "max_time_limit": None,
            "min_time_limit": 0,
        }
        if (
            num_cpus < 1
            or num_cpus_child != num_cpus
            or num_gpus != 0
            or num_gpus_child != 0
            or _exact_int(row["bag_folds"], f"outer_rows[{index}].bag_folds") != 8
            or _exact_int(row["bag_sets"], f"outer_rows[{index}].bag_sets") != 1
            or _exact_int(
                row["bag_random_state"], f"outer_rows[{index}].bag_random_state"
            )
            != 1
            or row["child_ag_args_fit"] != expected_child_ag_args
        ):
            raise RuntimeError(f"outer_rows[{index}] resolved resources/bag do not match")
        normalized = {
            **row,
            "dataset": str(dataset),
            "task_id": int(task_id),
            "repeat": repeat,
            "fold": fold,
            "registered_fold": registered,
            "compressed_refit_params": compressed_refit_params,
            "num_cpus": num_cpus,
            "num_gpus": num_gpus,
            "num_cpus_child": num_cpus_child,
            "num_gpus_child": num_gpus_child,
            "test_rmse": _positive_finite(
                row["test_rmse"], f"outer_rows[{index}].test_rmse"
            ),
            "val_rmse": _positive_finite(
                row["val_rmse"], f"outer_rows[{index}].val_rmse"
            ),
            "train_time_s": _positive_finite(
                row["train_time_s"], f"outer_rows[{index}].train_time_s"
            ),
            "infer_time_s": _positive_finite(
                row["infer_time_s"], f"outer_rows[{index}].infer_time_s"
            ),
            "peak_memory_bytes": _positive_finite(
                row["peak_memory_bytes"],
                f"outer_rows[{index}].peak_memory_bytes",
            ),
        }
        key = (str(dataset), repeat, fold, str(arm))
        if key in outer_sources:
            raise RuntimeError(f"duplicate outer result for {key}")
        outer_sources[key] = source
        outer_rows.append(normalized)

    child_rows: list[dict] = []
    for index, value in enumerate(raw_children):
        row = dict(_as_mapping(value, f"child_rows[{index}]"))
        if set(row) != child_fields:
            raise RuntimeError(f"child_rows[{index}] fields are not exact")
        dataset = row["dataset"]
        if dataset not in task_split_counts:
            raise RuntimeError(f"child_rows[{index}] has an unexpected dataset")
        task_id, split_count = task_split_counts[str(dataset)]
        repeat = _exact_int(row["repeat"], f"child_rows[{index}].repeat")
        fold = _exact_int(row["fold"], f"child_rows[{index}].fold")
        registered = _exact_int(
            row["registered_fold"], f"child_rows[{index}].registered_fold"
        )
        child_fold = _exact_int(
            row["child_fold"], f"child_rows[{index}].child_fold"
        )
        child_name = row["child"]
        arm = row["arm"]
        if (
            _exact_int(row["task_id"], f"child_rows[{index}].task_id") != task_id
            or fold not in range(3)
            or registered != 3 * repeat + fold
            or registered not in range(split_count)
            or child_fold not in range(8)
            or child_name != f"S1F{child_fold + 1}"
            or arm not in horizon_arms
        ):
            raise RuntimeError(f"child_rows[{index}] coordinate is inconsistent")
        outer_key = (str(dataset), repeat, fold, str(arm))
        if row["source"] != outer_sources.get(outer_key):
            raise RuntimeError(f"child_rows[{index}] source does not match its outer row")

        requested = _exact_int(
            row["iterations_requested"],
            f"child_rows[{index}].iterations_requested",
        )
        attempted = _exact_int(
            row["iterations_attempted"],
            f"child_rows[{index}].iterations_attempted",
        )
        completed = _exact_int(
            row["rounds_completed"], f"child_rows[{index}].rounds_completed"
        )
        retained = _exact_int(
            row["rounds_retained"], f"child_rows[{index}].rounds_retained"
        )
        best = _exact_int(
            row["best_iteration"], f"child_rows[{index}].best_iteration"
        )
        if (
            requested != _exact_int(
                horizon_arms[str(arm)]["iterations"], "frozen horizon"
            )
            or not (0 <= retained == best <= completed <= attempted <= requested)
        ):
            raise RuntimeError(f"child_rows[{index}] round counts are inconsistent")
        resolved_lr = _positive_finite(
            row["resolved_learning_rate"],
            f"child_rows[{index}].resolved_learning_rate",
        )
        early_rounds = _exact_int(
            row["early_stopping_rounds"],
            f"child_rows[{index}].early_stopping_rounds",
        )
        reason = row["stop_reason"]
        _validate_stop_reason_causality(
            reason,
            requested=requested,
            attempted=attempted,
            completed=completed,
            field=f"child_rows[{index}]",
        )
        if (
            resolved_lr != 0.1
            or row["requested_tree_mode"] != "catboost"
            or row["selected_tree_mode"] != "catboost"
            or row["selected_lane"] != "boosting"
            or row["linear_residual_active"] is not False
            or early_rounds != 50
            or row["deadline_is_soft"] is not True
            or row["deadline_hit"] is not (reason == "time_limit")
        ):
            raise RuntimeError(f"child_rows[{index}] resolved policy is inconsistent")
        refit_params = _validate_frozen_refit_params(
            row["refit_params"],
            f"child_rows[{index}] refit policy",
            expected_iterations=best,
            max_iterations=requested,
        )
        initial_hyperparameters = row["initial_hyperparameters"]
        if initial_hyperparameters != expected_child_hyperparameters(
            str(arm), child_fold
        ):
            raise RuntimeError(f"child_rows[{index}] initial policy or seed is wrong")
        outer_resources = next(
            outer
            for outer in outer_rows
            if (
                outer["dataset"],
                outer["repeat"],
                outer["fold"],
                outer["arm"],
            )
            == outer_key
        )
        if (
            _exact_int(row["num_cpus"], f"child_rows[{index}].num_cpus")
            != outer_resources["num_cpus_child"]
            or _exact_int(row["num_gpus"], f"child_rows[{index}].num_gpus")
            != outer_resources["num_gpus_child"]
            or row["problem_type"] != "regression"
            or row["eval_metric"] != "root_mean_squared_error"
            or row["stopping_metric"] != "root_mean_squared_error"
            or row["val_in_fit"] is not True
            or row["unlabeled_in_fit"] is not False
            or row["ag_args_fit"] != expected_child_ag_args
        ):
            raise RuntimeError(f"child_rows[{index}] resolved resources/policy is wrong")
        wall_limit = _positive_finite(
            row["wall_clock_limit_seconds"],
            f"child_rows[{index}].wall_clock_limit_seconds",
        )
        wall_margin = _nonnegative_finite(
            row["wall_clock_safety_margin_seconds"],
            f"child_rows[{index}].wall_clock_safety_margin_seconds",
        )
        wall_effective = _nonnegative_finite(
            row["wall_clock_effective_seconds"],
            f"child_rows[{index}].wall_clock_effective_seconds",
        )
        wall_elapsed = _nonnegative_finite(
            row["wall_clock_elapsed_seconds"],
            f"child_rows[{index}].wall_clock_elapsed_seconds",
        )
        if (
            wall_limit > TIME_LIMIT_SECONDS
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
            raise RuntimeError(f"child_rows[{index}] deadline metadata is inconsistent")
        child_rows.append(
            {
                **row,
                "dataset": str(dataset),
                "task_id": int(task_id),
                "repeat": repeat,
                "fold": fold,
                "registered_fold": registered,
                "child_fold": child_fold,
                "iterations_requested": requested,
                "iterations_attempted": attempted,
                "rounds_completed": completed,
                "rounds_retained": retained,
                "best_iteration": best,
                "resolved_learning_rate": resolved_lr,
                "early_stopping_rounds": early_rounds,
                "refit_params": refit_params,
                "initial_hyperparameters": dict(initial_hyperparameters),
                "wall_clock_limit_seconds": wall_limit,
                "wall_clock_safety_margin_seconds": wall_margin,
                "wall_clock_effective_seconds": wall_effective,
                "wall_clock_elapsed_seconds": wall_elapsed,
            }
        )

    expected_outer = {
        (dataset, registered // 3, registered % 3, arm)
        for dataset, (_, split_count) in task_split_counts.items()
        for registered in range(split_count)
        for arm in horizon_arms
    }
    observed_outer = set(outer_sources)
    if len(observed_outer) != len(outer_rows):
        raise RuntimeError("completed campaign contains duplicate outer results")
    if observed_outer != expected_outer:
        raise RuntimeError("completed campaign outer-result grid is not exact")
    resource_allocations = {
        (
            row["num_cpus"],
            row["num_gpus"],
            row["num_cpus_child"],
            row["num_gpus_child"],
        )
        for row in outer_rows
    }
    if len(resource_allocations) != 1:
        raise RuntimeError(
            "completed campaign did not use one identical CPU/GPU allocation"
        )
    expected_children = {
        (*key, f"S1F{child}") for key in expected_outer for child in range(1, 9)
    }
    observed_children = {
        (
            row["dataset"],
            row["repeat"],
            row["fold"],
            row["arm"],
            row["child"],
        )
        for row in child_rows
    }
    if len(observed_children) != len(child_rows):
        raise RuntimeError("completed campaign contains duplicate child-fit metadata")
    if observed_children != expected_children:
        raise RuntimeError("completed campaign child-fit grid is not exact")
    child_iterations_by_outer: dict[tuple[str, int, int, str], list[int]] = (
        defaultdict(list)
    )
    for row in child_rows:
        child_iterations_by_outer[
            (row["dataset"], row["repeat"], row["fold"], row["arm"])
        ].append(row["refit_params"]["iterations"])
    for row in outer_rows:
        key = (row["dataset"], row["repeat"], row["fold"], row["arm"])
        _validate_compressed_refit_iterations(
            row["compressed_refit_params"],
            child_iterations_by_outer[key],
            field=f"outer result {key} compressed_refit_params",
        )
    return outer_rows, child_rows


def _ratio_fields(metric: str, numerator: float, denominator: float) -> dict:
    ratio = _positive_finite(numerator, f"{metric} numerator") / _positive_finite(
        denominator, f"{metric} denominator"
    )
    return {
        f"{metric}_log_ratio": math.log(ratio),
        f"{metric}_ratio": ratio,
        f"{metric}_pct": 100.0 * (ratio - 1.0),
    }


def pair_outer_rows(
    outer_rows: Sequence[Mapping],
    *,
    task_split_counts: Mapping[str, tuple[int, int]] = TASK_SPLIT_COUNTS,
) -> list[dict]:
    """Return one paired row per frozen dataset/repeat/fold coordinate."""
    index: dict[tuple[str, int, int, str], Mapping] = {}
    for row in outer_rows:
        key = (
            str(row["dataset"]),
            int(row["repeat"]),
            int(row["fold"]),
            str(row["arm"]),
        )
        if key in index:
            raise RuntimeError(f"duplicate outer result for {key}")
        index[key] = row
    paired = []
    for dataset, (task_id, split_count) in task_split_counts.items():
        for registered_fold in range(split_count):
            repeat, fold = divmod(registered_fold, 3)
            try:
                short = index[(dataset, repeat, fold, "cap1000")]
                long = index[(dataset, repeat, fold, "cap10000")]
            except KeyError as exc:
                raise RuntimeError(
                    f"missing horizon arm for {(dataset, repeat, fold)}"
                ) from exc
            row = {
                "dataset": dataset,
                "task_id": task_id,
                "repeat": repeat,
                "fold": fold,
                "registered_fold": registered_fold,
            }
            for metric in METRICS:
                short_value = _positive_finite(short[metric], f"cap1000 {metric}")
                long_value = _positive_finite(long[metric], f"cap10000 {metric}")
                row[f"cap1000_{metric}"] = short_value
                row[f"cap10000_{metric}"] = long_value
                row.update(_ratio_fields(metric, long_value, short_value))
            row["cap10000_test_win"] = row["test_rmse_ratio"] < 1.0
            row["cap10000_test_tie"] = row["test_rmse_ratio"] == 1.0
            paired.append(row)
    if len(index) != 2 * len(paired):
        raise RuntimeError("unexpected outer results remain after pairing")
    return paired


def pair_child_rows(child_rows: Sequence[Mapping]) -> list[dict]:
    """Return one row per coordinate/child with both fitted metadata blocks."""
    index: dict[tuple[str, int, int, str, str], Mapping] = {}
    for child in child_rows:
        key = (
            str(child["dataset"]),
            int(child["repeat"]),
            int(child["fold"]),
            str(child["child"]),
            str(child["arm"]),
        )
        if key in index:
            raise RuntimeError(f"duplicate child fit for {key}")
        index[key] = child
    pair_keys = sorted({key[:-1] for key in index})
    paired = []
    metadata_fields = (
        "iterations_requested",
        "iterations_attempted",
        "rounds_completed",
        "rounds_retained",
        "best_iteration",
        "resolved_learning_rate",
        "requested_tree_mode",
        "selected_tree_mode",
        "selected_lane",
        "linear_residual_active",
        "early_stopping_rounds",
        "stop_reason",
        "wall_clock_limit_seconds",
        "wall_clock_safety_margin_seconds",
        "wall_clock_effective_seconds",
        "wall_clock_elapsed_seconds",
        "deadline_hit",
        "deadline_is_soft",
    )
    for dataset, repeat, fold, child_name in pair_keys:
        try:
            short = index[(dataset, repeat, fold, child_name, "cap1000")]
            long = index[(dataset, repeat, fold, child_name, "cap10000")]
        except KeyError as exc:
            raise RuntimeError(
                f"missing child horizon arm for {(dataset, repeat, fold, child_name)}"
            ) from exc
        row = {
            "dataset": dataset,
            "task_id": int(short["task_id"]),
            "repeat": repeat,
            "fold": fold,
            "registered_fold": int(short["registered_fold"]),
            "child": child_name,
            "child_fold": int(short["child_fold"]),
        }
        for arm, source in (("cap1000", short), ("cap10000", long)):
            for field in metadata_fields:
                row[f"{arm}_{field}"] = source[field]
        row["cap10000_completed_over_1000"] = int(long["rounds_completed"]) > 1_000
        row["cap1000_hit_cap"] = short["stop_reason"] == "iteration_limit"
        row["rounds_completed_delta"] = int(long["rounds_completed"]) - int(
            short["rounds_completed"]
        )
        row["best_iteration_delta"] = int(long["best_iteration"]) - int(
            short["best_iteration"]
        )
        paired.append(row)
    if len(index) != 2 * len(paired):
        raise RuntimeError("unexpected child fits remain after pairing")
    return paired


def _nested_log_values(
    split_rows: Sequence[Mapping], metric_log_key: str
) -> dict[str, dict[int, list[float]]]:
    nested: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in split_rows:
        key = (str(row["dataset"]), int(row["repeat"]), int(row["fold"]))
        if key in seen:
            raise RuntimeError(f"duplicate paired split for {key}")
        seen.add(key)
        value = float(row[metric_log_key])
        if not math.isfinite(value):
            raise RuntimeError(f"nonfinite paired log ratio for {key}")
        nested[key[0]][key[1]].append(value)
    if not nested:
        raise RuntimeError("cannot aggregate an empty paired panel")
    for dataset, repeats in nested.items():
        for repeat, folds in repeats.items():
            if len(folds) != 3:
                raise RuntimeError(
                    f"{dataset} repeat {repeat} has {len(folds)} folds, expected 3"
                )
    return {dataset: dict(repeats) for dataset, repeats in nested.items()}


def hierarchical_point_log_ratio(
    split_rows: Sequence[Mapping], metric_log_key: str
) -> tuple[float, dict[str, float], dict[str, dict[int, float]]]:
    """Average folds, repeats, then datasets exactly as predeclared."""
    nested = _nested_log_values(split_rows, metric_log_key)
    repeat_estimates: dict[str, dict[int, float]] = {}
    dataset_estimates: dict[str, float] = {}
    for dataset, repeats in nested.items():
        repeat_estimates[dataset] = {
            repeat: math.fsum(folds) / len(folds)
            for repeat, folds in sorted(repeats.items())
        }
        values = list(repeat_estimates[dataset].values())
        dataset_estimates[dataset] = math.fsum(values) / len(values)
    point = math.fsum(dataset_estimates.values()) / len(dataset_estimates)
    return point, dataset_estimates, repeat_estimates


def hierarchical_bootstrap_log_ratios(
    split_rows: Sequence[Mapping],
    metric_log_key: str = "test_rmse_log_ratio",
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Resample datasets, repeats within datasets, and folds within repeats."""
    if draws <= 0:
        raise ValueError("draws must be positive")
    nested = _nested_log_values(split_rows, metric_log_key)
    datasets = sorted(nested)
    rng = np.random.default_rng(seed)
    out = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled_datasets = rng.integers(0, len(datasets), size=len(datasets))
        dataset_values = []
        for dataset_index in sampled_datasets:
            repeats = nested[datasets[int(dataset_index)]]
            repeat_ids = sorted(repeats)
            sampled_repeats = rng.integers(0, len(repeat_ids), size=len(repeat_ids))
            repeat_values = []
            for repeat_index in sampled_repeats:
                folds = np.asarray(
                    repeats[repeat_ids[int(repeat_index)]], dtype=np.float64
                )
                sampled_folds = rng.integers(0, len(folds), size=len(folds))
                repeat_values.append(float(np.mean(folds[sampled_folds])))
            dataset_values.append(math.fsum(repeat_values) / len(repeat_values))
        out[draw] = math.fsum(dataset_values) / len(dataset_values)
    return out


def repeat_block_bootstrap_log_ratios(
    repeat_log_ratios: Sequence[float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Condition on one dataset and resample whole repeat blocks."""
    values = np.asarray(repeat_log_ratios, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise RuntimeError("repeat-block bootstrap input must be finite and nonempty")
    if draws <= 0:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    return np.mean(values[indices], axis=1)


def exact_one_sided_sign_test_pvalue(wins: int, losses: int) -> float:
    """Return P[Binomial(n, .5) >= wins], excluding exact ties from n."""
    if wins < 0 or losses < 0:
        raise ValueError("wins and losses must be nonnegative")
    n = wins + losses
    if n == 0:
        return 1.0
    return math.fsum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n)


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def _ratio_summary(log_ratio: float) -> dict[str, float]:
    ratio = math.exp(log_ratio)
    return {
        "log_ratio": log_ratio,
        "ratio": ratio,
        "pct": 100.0 * (ratio - 1.0),
    }


def _t_sensitivity(dataset_logs: Sequence[float]) -> dict[str, float]:
    if len(dataset_logs) != 13:
        raise RuntimeError("the frozen t sensitivity check requires 13 datasets")
    mean = statistics.fmean(dataset_logs)
    standard_error = statistics.stdev(dataset_logs) / math.sqrt(len(dataset_logs))
    # scipy.stats.t.ppf(.975, 12), frozen here to avoid an analysis-only runtime
    # dependency beyond NumPy.
    critical = 2.1788128296634177
    return {
        "degrees_of_freedom": 12,
        "critical_975": critical,
        "log_lower95": mean - critical * standard_error,
        "log_upper95": mean + critical * standard_error,
        "ratio_lower95": math.exp(mean - critical * standard_error),
        "ratio_upper95": math.exp(mean + critical * standard_error),
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": _quantile(array, 0.90),
        "max": float(np.max(array)),
    }


def analyze_paired_rows(
    split_rows: Sequence[Mapping],
    child_pairs: Sequence[Mapping],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Compute the frozen estimands, diagnostics, and all decision gates."""
    test_log, dataset_logs, repeat_logs = hierarchical_point_log_ratio(
        split_rows, "test_rmse_log_ratio"
    )
    if len(dataset_logs) != 13:
        raise RuntimeError(f"expected 13 datasets, got {len(dataset_logs)}")
    metric_summaries = {}
    for metric in METRICS:
        point, _, _ = hierarchical_point_log_ratio(
            split_rows, f"{metric}_log_ratio"
        )
        metric_summaries[metric] = _ratio_summary(point)

    bootstrap = hierarchical_bootstrap_log_ratios(
        split_rows, draws=draws, seed=seed
    )
    primary_bootstrap = {
        "draws": draws,
        "seed": seed,
        "ratio_lower95_two_sided": math.exp(_quantile(bootstrap, 0.025)),
        "ratio_upper95_two_sided": math.exp(_quantile(bootstrap, 0.975)),
        "ratio_upper95_one_sided": math.exp(_quantile(bootstrap, 0.95)),
    }

    dataset_summaries = []
    for dataset_index, dataset in enumerate(sorted(dataset_logs)):
        log_ratio = dataset_logs[dataset]
        repeat_values = [repeat_logs[dataset][key] for key in sorted(repeat_logs[dataset])]
        conditional = repeat_block_bootstrap_log_ratios(
            repeat_values,
            draws=draws,
            seed=seed + dataset_index + 1,
        )
        selected = [row for row in split_rows if row["dataset"] == dataset]
        worst = max(selected, key=lambda row: float(row["test_rmse_ratio"]))
        item = {
            "dataset": dataset,
            "split_count": len(selected),
            "repeat_count": len(repeat_values),
            **_ratio_summary(log_ratio),
            "repeat_wins": sum(value < 0.0 for value in repeat_values),
            "repeat_losses": sum(value > 0.0 for value in repeat_values),
            "repeat_ties": sum(value == 0.0 for value in repeat_values),
            "repeat_block_bootstrap_ratio_lower90": math.exp(
                _quantile(conditional, 0.10)
            ),
            "worst_split_ratio": float(worst["test_rmse_ratio"]),
            "worst_split": f"r{worst['repeat']}f{worst['fold']}",
        }
        item["conditional_harm"] = (
            item["ratio"] > THRESHOLDS["conditional_dataset_point_harm"]
            and item["repeat_block_bootstrap_ratio_lower90"] > 1.0
        )
        dataset_summaries.append(item)

    repeat_summaries = []
    for dataset in sorted(repeat_logs):
        for repeat, log_ratio in sorted(repeat_logs[dataset].items()):
            selected = [
                row
                for row in split_rows
                if row["dataset"] == dataset and int(row["repeat"]) == repeat
            ]
            worst = max(selected, key=lambda row: float(row["test_rmse_ratio"]))
            repeat_summaries.append(
                {
                    "dataset": dataset,
                    "repeat": int(repeat),
                    "fold_count": len(selected),
                    **_ratio_summary(log_ratio),
                    "fold_wins": sum(
                        float(row["test_rmse_ratio"]) < 1.0 for row in selected
                    ),
                    "fold_losses": sum(
                        float(row["test_rmse_ratio"]) > 1.0 for row in selected
                    ),
                    "fold_ties": sum(
                        float(row["test_rmse_ratio"]) == 1.0 for row in selected
                    ),
                    "worst_fold": int(worst["fold"]),
                    "worst_split_ratio": float(worst["test_rmse_ratio"]),
                }
            )

    wins = sum(item["ratio"] < 1.0 for item in dataset_summaries)
    losses = sum(item["ratio"] > 1.0 for item in dataset_summaries)
    ties = len(dataset_summaries) - wins - losses
    sign_p = exact_one_sided_sign_test_pvalue(wins, losses)

    cap_children = [row for row in child_pairs if row["cap1000_hit_cap"] is True]
    capped_pairs_over = sum(
        row["cap10000_completed_over_1000"] is True for row in cap_children
    )
    all_pairs_over = sum(
        row["cap10000_completed_over_1000"] is True for row in child_pairs
    )
    all_pairs_fraction = all_pairs_over / len(child_pairs) if child_pairs else 0.0
    capped_pairs_fraction = (
        capped_pairs_over / len(cap_children) if cap_children else 0.0
    )
    time_stops = sum(
        row[f"{arm}_stop_reason"] == "time_limit"
        for row in child_pairs
        for arm in ARMS
    )
    mechanism = {
        "cap1000_iteration_limit_children": len(cap_children),
        "cap10000_children_over_1000": all_pairs_over,
        "cap10000_children_over_1000_fraction": all_pairs_fraction,
        "among_capped_cap10000_children_over_1000": capped_pairs_over,
        "among_capped_cap10000_children_over_1000_fraction": (
            capped_pairs_fraction
        ),
        "time_limit_stops": time_stops,
    }

    child_metadata = {}
    for arm in ARMS:
        stop_counts = Counter(str(row[f"{arm}_stop_reason"]) for row in child_pairs)
        stop_reason_counts = {
            reason: int(stop_counts.get(reason, 0)) for reason in STOP_REASON_ORDER
        }
        stop_reason_diagnostics = {
            reason: {
                "count": count,
                "denominator": len(child_pairs),
                "fraction": count / len(child_pairs),
            }
            for reason, count in stop_reason_counts.items()
        }
        requested_horizon = int(HORIZON_ARMS[arm]["iterations"])
        near_cap_threshold = int(math.ceil(0.95 * requested_horizon))
        at_cap_count = sum(
            int(row[f"{arm}_rounds_completed"]) == requested_horizon
            for row in child_pairs
        )
        near_cap_count = sum(
            int(row[f"{arm}_rounds_completed"]) >= near_cap_threshold
            for row in child_pairs
        )
        child_metadata[arm] = {
            "child_fit_count": len(child_pairs),
            "stop_reason_counts": stop_reason_counts,
            "stop_reason_diagnostics": stop_reason_diagnostics,
            "requested_horizon": requested_horizon,
            "near_cap_definition": "rounds_completed >= ceil(0.95 * requested_horizon)",
            "near_cap_threshold": near_cap_threshold,
            "at_cap_count": at_cap_count,
            "at_cap_fraction": at_cap_count / len(child_pairs),
            "near_cap_count": near_cap_count,
            "near_cap_fraction": near_cap_count / len(child_pairs),
            "best_iteration": _distribution(
                [float(row[f"{arm}_best_iteration"]) for row in child_pairs]
            ),
            "rounds_completed": _distribution(
                [float(row[f"{arm}_rounds_completed"]) for row in child_pairs]
            ),
            "resolved_learning_rate_counts": dict(
                sorted(
                    Counter(
                        str(row[f"{arm}_resolved_learning_rate"])
                        for row in child_pairs
                    ).items()
                )
            ),
            "selected_tree_mode_counts": dict(
                sorted(
                    Counter(
                        str(row[f"{arm}_selected_tree_mode"])
                        for row in child_pairs
                    ).items()
                )
            ),
            "selected_lane_counts": dict(
                sorted(
                    Counter(str(row[f"{arm}_selected_lane"]) for row in child_pairs).items()
                )
            ),
        }

    gates = {
        "complete_provenance_matched_panel": True,
        "test_point_ratio_at_most_0_995": metric_summaries["test_rmse"]["ratio"]
        <= THRESHOLDS["test_ratio_max"],
        "hierarchical_bootstrap_upper95_below_1": primary_bootstrap[
            "ratio_upper95_one_sided"
        ]
        < THRESHOLDS["bootstrap_upper95_max"],
        "at_least_10_of_13_dataset_wins": wins
        >= THRESHOLDS["dataset_wins_min"],
        "one_sided_sign_test_p_below_0_05": sign_p
        < THRESHOLDS["sign_test_p_max"],
        "no_conditional_dataset_harm": not any(
            item["conditional_harm"] for item in dataset_summaries
        ),
        "no_dataset_point_ratio_above_1_02": max(
            item["ratio"] for item in dataset_summaries
        )
        <= THRESHOLDS["dataset_hard_ratio_max"],
        "validation_ratio_at_most_1_002": metric_summaries["val_rmse"]["ratio"]
        <= THRESHOLDS["validation_ratio_max"],
        "cap1000_has_iteration_limit_child": len(cap_children) > 0,
        "at_least_20pct_paired_cap10000_children_exceed_1000": all_pairs_fraction
        >= THRESHOLDS["paired_10k_over_1000_fraction_min"],
        "zero_time_limit_stops": time_stops
        <= THRESHOLDS["time_limit_stops_max"],
        "train_time_ratio_at_most_2": metric_summaries["train_time_s"]["ratio"]
        <= THRESHOLDS["train_time_ratio_max"],
        "inference_time_ratio_at_most_1_10": metric_summaries["infer_time_s"][
            "ratio"
        ]
        <= THRESHOLDS["infer_time_ratio_max"],
        "peak_memory_ratio_at_most_1_10": metric_summaries[
            "peak_memory_bytes"
        ]["ratio"]
        <= THRESHOLDS["peak_memory_ratio_max"],
    }
    gates["advance"] = all(gates.values())

    worst_split = max(split_rows, key=lambda row: float(row["test_rmse_ratio"]))
    return {
        "protocol": "frozen TabArena scalar-regression cap-horizon experiment",
        "arms": {name: dict(config) for name, config in HORIZON_ARMS.items()},
        "thresholds": dict(THRESHOLDS),
        "counts": {
            "datasets": len(dataset_logs),
            "paired_splits": len(split_rows),
            "outer_results": 2 * len(split_rows),
            "paired_children": len(child_pairs),
            "child_fit_blocks": 2 * len(child_pairs),
        },
        "equal_dataset": metric_summaries,
        "primary_test": {
            **_ratio_summary(test_log),
            "dataset_wins": wins,
            "dataset_losses": losses,
            "dataset_ties": ties,
            "one_sided_sign_test_p": sign_p,
            "bootstrap": primary_bootstrap,
            "t_interval_sensitivity": _t_sensitivity(list(dataset_logs.values())),
            "split_wins": sum(float(row["test_rmse_ratio"]) < 1.0 for row in split_rows),
            "split_losses": sum(float(row["test_rmse_ratio"]) > 1.0 for row in split_rows),
            "split_ties": sum(float(row["test_rmse_ratio"]) == 1.0 for row in split_rows),
            "worst_split_dataset": str(worst_split["dataset"]),
            "worst_split": f"r{worst_split['repeat']}f{worst_split['fold']}",
            "worst_split_ratio": float(worst_split["test_rmse_ratio"]),
        },
        "datasets": dataset_summaries,
        "repeats": repeat_summaries,
        "mechanism": mechanism,
        "child_metadata": child_metadata,
        "gates": gates,
    }


def _csv_bytes(path: Path, rows: Sequence[Mapping]) -> bytes:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _existing_regular_file(path: Path, field: str) -> None:
    """Reject a non-regular existing target without following a final symlink."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeError(f"could not inspect {field}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"{field} must not be a symbolic link: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{field} must be a regular-file target: {path}")


def _canonical_output_targets(
    input_dir: Path,
    targets: Mapping[str, Path],
    *,
    protected_paths: Sequence[Path],
) -> dict[str, Path]:
    """Validate and canonicalize analyzer output paths before any write.

    This boundary protects campaign inputs from accidental or adversarial CLI
    path selection: every output must be a distinct regular-file target below
    the canonical campaign directory and must not alias a protected input,
    including by hard link.  Atomic replacement below also avoids following a
    final-component symlink.  This is not intended to defend against a
    privileged process concurrently rewriting directory entries after this
    check; the analyzer revalidates immediately before publishing decisions.
    """
    if set(targets) != set(OUTPUT_TARGET_NAMES):
        raise RuntimeError("analysis output target fields are not exact")
    try:
        canonical_input = input_dir.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            f"analysis input directory does not resolve: {input_dir}"
        ) from exc
    if not canonical_input.is_dir():
        raise RuntimeError("analysis input path is not a directory")

    canonical_protected: list[Path] = []
    for protected in protected_paths:
        try:
            canonical_protected.append(Path(protected).resolve(strict=True))
        except OSError as exc:
            raise RuntimeError(
                f"protected campaign artifact does not resolve: {protected}"
            ) from exc

    canonical: dict[str, Path] = {}
    for name in OUTPUT_TARGET_NAMES:
        raw = Path(targets[name])
        try:
            target = raw.resolve(strict=False)
            relative = target.relative_to(canonical_input)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"analysis output {name} must be strictly under the input directory"
            ) from exc
        if relative == Path("."):
            raise RuntimeError(
                f"analysis output {name} must not be the input directory itself"
            )
        _existing_regular_file(raw, f"analysis output {name}")

        # All existing descendants on the canonical route must be real
        # directories.  Nonexistent suffixes are allowed and created later.
        cursor = canonical_input
        for component in relative.parts[:-1]:
            cursor = cursor / component
            try:
                metadata = cursor.lstat()
            except FileNotFoundError:
                break
            except OSError as exc:
                raise RuntimeError(
                    f"could not inspect parent of analysis output {name}: {cursor}"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(
                    f"analysis output {name} has a symbolic-link parent: {cursor}"
                )
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(
                    f"analysis output {name} has a non-directory parent: {cursor}"
                )
        _existing_regular_file(target, f"analysis output {name}")

        for protected in canonical_protected:
            if target == protected:
                raise RuntimeError(
                    f"analysis output {name} collides with a protected "
                    "campaign artifact"
                )
            if target.exists() and protected.exists():
                try:
                    aliases_protected = os.path.samefile(target, protected)
                except OSError as exc:
                    raise RuntimeError(
                        f"could not compare analysis output {name} with "
                        "protected artifacts"
                    ) from exc
                if aliases_protected:
                    raise RuntimeError(
                        f"analysis output {name} aliases a protected campaign artifact"
                    )
        if target.name == "results.pkl":
            raise RuntimeError(
                f"analysis output {name} must not use the protected results.pkl name"
            )
        canonical[name] = target

    for index, left_name in enumerate(OUTPUT_TARGET_NAMES):
        left = canonical[left_name]
        for right_name in OUTPUT_TARGET_NAMES[index + 1 :]:
            right = canonical[right_name]
            collision = left == right or left in right.parents or right in left.parents
            if not collision and left.exists() and right.exists():
                try:
                    collision = os.path.samefile(left, right)
                except OSError as exc:
                    raise RuntimeError(
                        "could not compare analysis output targets"
                    ) from exc
            if collision:
                raise RuntimeError(
                    f"analysis outputs {left_name} and {right_name} are not distinct "
                    "regular-file targets"
                )
    return canonical


def _protected_campaign_paths(
    input_dir: Path,
    *,
    manifest_path: Path,
    attestation_path: Path,
    attestation: Mapping[str, Any],
) -> list[Path]:
    """Return every input artifact an analysis output must never replace."""
    artifacts = _as_mapping(
        attestation.get("result_artifacts"), "attested result artifacts"
    )
    protected = {
        manifest_path.resolve(strict=True),
        attestation_path.resolve(strict=True),
        (input_dir / ANALYSIS_PAYLOAD_FILENAME).resolve(strict=True),
    }
    for frozen_name in (
        MANIFEST_FILENAME,
        COMPLETION_ATTESTATION_FILENAME,
        WARMUP_HISTORY_FILENAME,
        RESUME_HISTORY_FILENAME,
    ):
        frozen_path = input_dir / frozen_name
        if frozen_path.exists():
            protected.add(frozen_path.resolve(strict=True))
    for relative in artifacts:
        if not isinstance(relative, str):
            raise RuntimeError("attested result path must be a string")
        protected.add((input_dir / relative).resolve(strict=True))
    # The exact-result-set verification already rejects unattested raw result
    # files.  Keeping this scan here makes the output boundary independently
    # refuse every extant results.pkl even when a unit-sized campaign is used.
    protected.update(
        path.resolve(strict=True) for path in input_dir.rglob("results.pkl")
    )
    return sorted(protected)


def _stage_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _existing_regular_file(path, "analysis output")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Replace one regular output atomically, preserving the old file on error."""
    temporary = _stage_bytes(path, payload)
    try:
        _existing_regular_file(path, "analysis output")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reserve_backup_path(path: Path) -> Path:
    descriptor, backup_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".backup",
    )
    os.close(descriptor)
    backup = Path(backup_name)
    backup.unlink()
    return backup


def _atomic_write_group(
    outputs: Sequence[tuple[Path, bytes]],
    *,
    post_write_check: Callable[[], None] | None = None,
) -> None:
    """Publish related decision files with best-effort transactional rollback.

    Each payload is fully written and fsynced before publication.  If any
    replacement or the optional final provenance check fails, newly published
    files are removed and prior regular files are restored.  There is no
    portable multi-file filesystem transaction, but this prevents partial
    decisions for ordinary write/check failures; each individual replacement
    remains atomic if the process or host stops abruptly.
    """
    targets = [path for path, _ in outputs]
    if len(set(targets)) != len(targets):
        raise RuntimeError("atomic output group contains duplicate targets")
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for path, payload in outputs:
            staged[path] = _stage_bytes(path, payload)
        for path in targets:
            _existing_regular_file(path, "decision output")
            if path.exists():
                backup = _reserve_backup_path(path)
                os.replace(path, backup)
                backups[path] = backup
        for path in targets:
            os.replace(staged[path], path)
            installed.append(path)
        if post_write_check is not None:
            post_write_check()
    except BaseException as exc:
        rollback_errors: list[BaseException] = []
        for path in reversed(installed):
            try:
                path.unlink(missing_ok=True)
            except BaseException as rollback_exc:
                rollback_errors.append(rollback_exc)
        for path in reversed(targets):
            backup = backups.get(path)
            if backup is not None and backup.exists():
                try:
                    os.replace(backup, path)
                except BaseException as rollback_exc:
                    rollback_errors.append(rollback_exc)
        if rollback_errors:
            raise RuntimeError(
                "decision output publication failed and rollback was incomplete"
            ) from exc
        raise
    finally:
        for temporary in (*staged.values(), *backups.values()):
            temporary.unlink(missing_ok=True)


def render_markdown_report(summary: Mapping) -> str:
    """Render an auditable report with every diagnostic before the decision."""
    primary = summary["primary_test"]
    equal = summary["equal_dataset"]
    mechanism = summary["mechanism"]
    integrity = summary["integrity_diagnostics"]
    decision = (
        "ADVANCE 10,000 rounds"
        if summary["gates"]["advance"]
        else "RETAIN 1,000 rounds"
    )
    lines = [
        "# TabArena regression cap-horizon result",
        "",
        (
            "The frozen comparison changes only the maximum boosting horizon. "
            "Ratios below one favor the 10,000-round arm."
        ),
        "",
        "## Primary result",
        "",
        f"- Equal-dataset test RMSE ratio: **{primary['ratio']:.6f}** ({primary['pct']:+.3f}%).",
        (
            "- Hierarchical one-sided 95% upper bound: "
            f"**{primary['bootstrap']['ratio_upper95_one_sided']:.6f}**."
        ),
        (
            "- Hierarchical bootstrap two-sided 95% interval: "
            f"[{primary['bootstrap']['ratio_lower95_two_sided']:.6f}, "
            f"{primary['bootstrap']['ratio_upper95_two_sided']:.6f}]."
        ),
        (
            "- Dataset-level t-interval sensitivity: "
            f"[{primary['t_interval_sensitivity']['ratio_lower95']:.6f}, "
            f"{primary['t_interval_sensitivity']['ratio_upper95']:.6f}] "
            f"(df={primary['t_interval_sensitivity']['degrees_of_freedom']})."
        ),
        (
            "- Dataset wins/losses/ties: "
            f"**{primary['dataset_wins']}/{primary['dataset_losses']}/"
            f"{primary['dataset_ties']}**; exact one-sided sign-test "
            f"p={primary['one_sided_sign_test_p']:.6g}."
        ),
        (
            f"- Validation ratio: {equal['val_rmse']['ratio']:.6f}; "
            "training/inference/memory ratios: "
            f"{equal['train_time_s']['ratio']:.4f}/"
            f"{equal['infer_time_s']['ratio']:.4f}/"
            f"{equal['peak_memory_bytes']['ratio']:.4f}."
        ),
        "",
        "## Dataset estimates",
        "",
        "| Dataset | Test ratio | Repeat W/L/T | Repeat-block lower 90% | Worst split | Worst split ratio | Conditional harm |",
        "|---|---:|---:|---:|---|---:|:---:|",
    ]
    for item in summary["datasets"]:
        lines.append(
            f"| {item['dataset']} | {item['ratio']:.6f} | "
            f"{item['repeat_wins']}/{item['repeat_losses']}/{item['repeat_ties']} | "
            f"{item['repeat_block_bootstrap_ratio_lower90']:.6f} | "
            f"{item['worst_split']} | "
            f"{item['worst_split_ratio']:.6f} | "
            f"{'yes' if item['conditional_harm'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Repeat estimates",
            "",
            "| Dataset | Repeat | Test ratio | Folds W/L/T | Worst fold ratio |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in summary["repeats"]:
        lines.append(
            f"| {item['dataset']} | {item['repeat']} | {item['ratio']:.6f} | "
            f"{item['fold_wins']}/{item['fold_losses']}/{item['fold_ties']} | "
            f"f{item['worst_fold']} {item['worst_split_ratio']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Mechanism and fitted metadata",
            "",
            (
                "- 1,000-round children stopped by iteration limit: "
                f"{mechanism['cap1000_iteration_limit_children']}."
            ),
            (
                "- All paired 10,000-round children exceeding 1,000 rounds: "
                f"{mechanism['cap10000_children_over_1000']} "
                f"({mechanism['cap10000_children_over_1000_fraction']:.1%})."
            ),
            (
                "- Among capped 1,000-round children, paired 10,000-round "
                "children exceeding 1,000: "
                f"{mechanism['among_capped_cap10000_children_over_1000']} "
                f"({mechanism['among_capped_cap10000_children_over_1000_fraction']:.1%})."
            ),
            f"- Wall-clock stops: {mechanism['time_limit_stops']}.",
            "",
            "### Fit-iteration distributions",
            "",
            "| Arm | Field | Count | Min | Median | P90 | Max |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        for field in ("best_iteration", "rounds_completed"):
            distribution = summary["child_metadata"][arm][field]
            lines.append(
                f"| {arm} | `{field}` | {distribution['count']} | "
                f"{distribution['min']:.0f} | {distribution['median']:.0f} | "
                f"{distribution['p90']:.0f} | {distribution['max']:.0f} |"
            )

    lines.extend(
        [
            "",
            "### Resolved configuration diagnostics",
            "",
            "| Arm | Field | Value | Count |",
            "|---|---|---|---:|",
        ]
    )
    for arm in ARMS:
        for field in (
            "resolved_learning_rate_counts",
            "selected_tree_mode_counts",
            "selected_lane_counts",
        ):
            for value, count in summary["child_metadata"][arm][field].items():
                lines.append(f"| {arm} | `{field}` | `{value}` | {count} |")

    lines.extend(
        [
            "",
            "### Stop-reason diagnostics",
            "",
            "| Arm | Stop reason | Count | Denominator | Fraction |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        for reason in STOP_REASON_ORDER:
            item = summary["child_metadata"][arm]["stop_reason_diagnostics"][reason]
            lines.append(
                f"| {arm} | `{reason}` | {item['count']} | "
                f"{item['denominator']} | {item['fraction']:.1%} |"
            )

    lines.extend(
        [
            "",
            "### At/near-cap diagnostics",
            "",
            (
                "Near cap is frozen as `rounds_completed >= "
                "ceil(0.95 * requested_horizon)`."
            ),
            "",
            "| Arm | At cap | Near cap | Near-cap threshold |",
            "|---|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        item = summary["child_metadata"][arm]
        lines.append(
            f"| {arm} | {item['at_cap_count']} / {item['child_fit_count']} "
            f"({item['at_cap_fraction']:.1%}) | {item['near_cap_count']} / "
            f"{item['child_fit_count']} ({item['near_cap_fraction']:.1%}) | "
            f"{item['near_cap_threshold']} |"
        )

    lines.extend(
        [
            "",
            "## Campaign integrity diagnostics",
            "",
            f"Validation basis: {integrity['validation_basis']}.",
            "",
            "| Diagnostic | Count |",
            "|---|---:|",
            f"| Expected outer results | {integrity['expected_outer_results']} |",
            f"| Observed outer results | {integrity['observed_outer_results']} |",
            f"| Missing outer results | {integrity['missing_outer_results']} |",
            f"| Failed outer results | {integrity['failed_outer_results']} |",
            f"| Imputed outer results | {integrity['imputed_outer_results']} |",
            f"| Duplicated outer results | {integrity['duplicate_outer_results']} |",
            f"| Expected child-fit blocks | {integrity['expected_child_fit_blocks']} |",
            f"| Observed child-fit blocks | {integrity['observed_child_fit_blocks']} |",
            f"| Missing child-fit metadata | {integrity['missing_child_fit_metadata']} |",
            f"| Duplicated child-fit metadata | {integrity['duplicate_child_fit_metadata']} |",
            f"| Metadata-incomplete child fits | {integrity['metadata_incomplete_child_fits']} |",
            "",
            "## Frozen gates",
            "",
            "| Gate | Pass |",
            "|---|:---:|",
        ]
    )
    for name, passed in summary["gates"].items():
        if name != "advance":
            lines.append(f"| `{name}` | {'yes' if passed else '**no**'} |")

    lines.extend(["", "## Provenance", ""])
    preferred_provenance_order = (
        "git_head",
        "protocol_sha256",
        "manifest_sha256",
        "attestation_sha256",
        "analysis_payload_sha256",
        "completed_at_utc",
        "manifest_path",
        "attestation_path",
    )
    provenance = summary["provenance"]
    provenance_fields = [
        field for field in preferred_provenance_order if field in provenance
    ]
    provenance_fields.extend(sorted(set(provenance).difference(provenance_fields)))
    for field in provenance_fields:
        lines.append(f"- `{field}`: `{provenance[field]}`")

    lines.extend(["", "## Decision", "", f"**{decision}.**"])
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--attestation", type=Path)
    parser.add_argument("--split-csv", type=Path)
    parser.add_argument("--repeat-csv", type=Path)
    parser.add_argument("--child-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--report-md", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve(strict=True)
    requested_outputs = {
        "split_csv": args.split_csv or input_dir / DEFAULT_ANALYSIS_OUTPUT_FILENAMES[0],
        "repeat_csv": args.repeat_csv or input_dir / DEFAULT_ANALYSIS_OUTPUT_FILENAMES[1],
        "child_csv": args.child_csv or input_dir / DEFAULT_ANALYSIS_OUTPUT_FILENAMES[2],
        "summary_json": args.summary_json or input_dir / DEFAULT_ANALYSIS_OUTPUT_FILENAMES[3],
        "report_md": args.report_md or input_dir / DEFAULT_ANALYSIS_OUTPUT_FILENAMES[4],
    }
    manifest_path = (args.manifest or input_dir / MANIFEST_FILENAME).resolve(
        strict=True
    )
    attestation_path = (
        args.attestation or input_dir / COMPLETION_ATTESTATION_FILENAME
    ).resolve(strict=True)

    manifest, attestation, analysis_payload, digests = verify_campaign_integrity(
        input_dir,
        manifest_path=manifest_path,
        attestation_path=attestation_path,
    )
    protected_paths = _protected_campaign_paths(
        input_dir,
        manifest_path=manifest_path,
        attestation_path=attestation_path,
        attestation=attestation,
    )
    outputs = _canonical_output_targets(
        input_dir,
        requested_outputs,
        protected_paths=protected_paths,
    )
    outer_rows, child_rows = load_safe_rows(analysis_payload)
    split_rows = pair_outer_rows(outer_rows)
    child_pairs = pair_child_rows(child_rows)
    summary = analyze_paired_rows(split_rows, child_pairs)
    if summary["counts"] != {
        "datasets": 13,
        "paired_splits": 222,
        "outer_results": 444,
        "paired_children": 1_776,
        "child_fit_blocks": 3_552,
    }:
        raise RuntimeError("analysis counts do not match the frozen campaign")
    summary["integrity_diagnostics"] = {
        "validation_basis": (
            "completion attestation, runner normalization, and analyzer exact-grid "
            "revalidation"
        ),
        "expected_outer_results": EXPECTED_JOBS,
        "observed_outer_results": summary["counts"]["outer_results"],
        "missing_outer_results": 0,
        "failed_outer_results": 0,
        "imputed_outer_results": 0,
        "duplicate_outer_results": 0,
        "expected_child_fit_blocks": EXPECTED_CHILD_FITS,
        "observed_child_fit_blocks": summary["counts"]["child_fit_blocks"],
        "missing_child_fit_metadata": 0,
        "duplicate_child_fit_metadata": 0,
        "metadata_incomplete_child_fits": 0,
    }
    summary["provenance"] = {
        **digests,
        "manifest_path": str(manifest_path),
        "attestation_path": str(attestation_path),
        "protocol_sha256": protocol_sha256(),
        "git_head": manifest["source"]["git_head"],
        "completed_at_utc": attestation.get("completed_at_utc"),
    }

    # Serialize every artifact before the first write. The three CSVs contain
    # diagnostics but not the campaign Decision section. They are published
    # first to canonical, non-protected paths. Source/runtime provenance is
    # then checked again before the summary/report decision pair is installed.
    payloads = {
        "split_csv": _csv_bytes(outputs["split_csv"], split_rows),
        "repeat_csv": _csv_bytes(outputs["repeat_csv"], summary["repeats"]),
        "child_csv": _csv_bytes(outputs["child_csv"], child_pairs),
        "summary_json": (
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
        "report_md": render_markdown_report(summary).encode("utf-8"),
    }
    for name in ("split_csv", "repeat_csv", "child_csv"):
        _atomic_write_bytes(outputs[name], payloads[name])

    # Revalidate aliases/types after the non-decision writes, then re-read and
    # hash the complete attested campaign snapshot. The decision pair has
    # rollback on replacement or a final post-publication integrity failure.
    outputs = _canonical_output_targets(
        input_dir,
        outputs,
        protected_paths=protected_paths,
    )

    def assert_campaign_unchanged() -> None:
        _assert_campaign_snapshot_unchanged(
            input_dir,
            manifest_path=manifest_path,
            attestation_path=attestation_path,
            baseline_manifest=manifest,
            baseline_attestation=attestation,
            baseline_analysis_payload=analysis_payload,
            baseline_digests=digests,
        )

    assert_campaign_unchanged()
    _atomic_write_group(
        [
            (outputs["summary_json"], payloads["summary_json"]),
            (outputs["report_md"], payloads["report_md"]),
        ],
        post_write_check=assert_campaign_unchanged,
    )
    print(
        f"analyzed 444 jobs and 3,552 child fits; "
        f"advance={summary['gates']['advance']}; wrote {outputs['summary_json']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
