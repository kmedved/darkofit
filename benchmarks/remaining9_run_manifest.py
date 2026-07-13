"""Capture and verify provenance for the remaining-nine confirmation run.

The confirmation runner uses a resumable TabArena cache.  A cache key does not
include the DarkoFit or TabArena revision, so a run manifest is required to
prove that every cached result belongs to the frozen source and environment.
This module supports the current in-flight run as well as future reruns without
reading any quality metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Mapping

try:
    from benchmarks.run_tabarena_regression_remaining9 import (
        EXPECTED_DATASET_SPLITS,
        EXPECTED_JOBS,
        FROZEN_CANDIDATE,
        SPLIT_INDICES,
        TASK_SPLIT_COUNTS,
    )
except ModuleNotFoundError:  # Direct execution: python benchmarks/remaining9_*.py
    from run_tabarena_regression_remaining9 import (
        EXPECTED_DATASET_SPLITS,
        EXPECTED_JOBS,
        FROZEN_CANDIDATE,
        SPLIT_INDICES,
        TASK_SPLIT_COUNTS,
    )


SCHEMA_VERSION = 1
ATTESTATION_SCHEMA_VERSION = 1
RUNNER_FILENAME = "run_tabarena_regression_remaining9.py"
RUNNER_MODULE = "benchmarks.run_tabarena_regression_remaining9"
RUNNER_PATH = Path(__file__).with_name(RUNNER_FILENAME).resolve()
ADAPTER_PATH = Path(__file__).with_name("tabarena_adapter.py").resolve()
DEFAULT_OUTPUT_DIR = Path(".cache/tabarena-regression-remaining9-0.9.0-20260712")
DEFAULT_SOURCE_COMMIT = "224bd46"
FROZEN_RUNNER_REF = "2f390a7"
FROZEN_RUNNER_GIT_BLOB = "4620d449ccaf692fec858398e4fbc2e015660897"
FROZEN_ADAPTER_REF = DEFAULT_SOURCE_COMMIT
FROZEN_ADAPTER_GIT_BLOB = "361cf266d588907112e5cd7fb8fdeb9e315a1097"
PACKAGE_DISTRIBUTIONS = (
    "darkofit",
    "tabarena",
    "autogluon.common",
    "autogluon.core",
    "autogluon.tabular",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "numba",
    "llvmlite",
    "psutil",
)
RELEVANT_ENVIRONMENT = (
    "NUMBA_CACHE_DIR",
    "NUMBA_NUM_THREADS",
    "NUMBA_THREADING_LAYER",
    "NUMBA_DISABLE_JIT",
    "NUMBA_CPU_NAME",
    "NUMBA_CPU_FEATURES",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "PYTHONPATH",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_command(args: list[str], *, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"command failed: {args[0]}") from exc
    return result.stdout.strip()


def _git(args: list[str], *, cwd: Path) -> str:
    return _run_command(["git", *args], cwd=cwd)


def _git_root(path: Path) -> Path:
    return Path(_git(["rev-parse", "--show-toplevel"], cwd=path)).resolve()


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _runtime_configuration() -> dict:
    import numba

    try:
        import psutil

        physical_cpus = psutil.cpu_count(logical=False)
        logical_cpus = psutil.cpu_count(logical=True)
    except ImportError:
        physical_cpus = None
        logical_cpus = os.cpu_count()
    return {
        "numba_num_threads": int(numba.config.NUMBA_NUM_THREADS),
        "numba_default_num_threads": int(numba.config.NUMBA_DEFAULT_NUM_THREADS),
        "numba_threading_layer": str(numba.config.THREADING_LAYER),
        "numba_disable_jit": int(numba.config.DISABLE_JIT),
        "numba_cpu_name": numba.config.CPU_NAME,
        "numba_cpu_features": numba.config.CPU_FEATURES,
        "physical_cpu_count": physical_cpus,
        "logical_cpu_count": logical_cpus,
    }


def _frozen_file_identity(
    *,
    path: Path,
    repo_root: Path,
    frozen_ref: str,
    expected_blob: str,
) -> dict:
    relative = path.relative_to(repo_root)
    current_blob = _git(["hash-object", str(path)], cwd=repo_root)
    head_blob = _git(["rev-parse", f"HEAD:{relative}"], cwd=repo_root)
    frozen_blob = _git(["rev-parse", f"{frozen_ref}:{relative}"], cwd=repo_root)
    status = _git(
        ["status", "--porcelain=v1", "--untracked-files=all", "--", str(relative)],
        cwd=repo_root,
    )
    if (
        current_blob != expected_blob
        or head_blob != expected_blob
        or frozen_blob != expected_blob
        or status
    ):
        raise RuntimeError(f"{relative} does not match frozen blob {expected_blob}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "git_blob": current_blob,
        "frozen_ref": frozen_ref,
        "status": status,
        "mtime_utc": datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ).isoformat(),
    }


def frozen_protocol_identity() -> dict:
    """Return the immutable experiment matrix and model policy."""
    return {
        "task_split_counts": {
            dataset: {"task_id": task_id, "split_count": split_count}
            for dataset, (task_id, split_count) in TASK_SPLIT_COUNTS.items()
        },
        "split_indices": list(SPLIT_INDICES),
        "expected_dataset_splits": EXPECTED_DATASET_SPLITS,
        "expected_jobs": EXPECTED_JOBS,
        "declared_source_commit": DEFAULT_SOURCE_COMMIT,
        "runner_git_blob": FROZEN_RUNNER_GIT_BLOB,
        "adapter_git_blob": FROZEN_ADAPTER_GIT_BLOB,
        "control_user_hyperparameters": {},
        "candidate_user_hyperparameters": dict(FROZEN_CANDIDATE),
        "bag_folds": 8,
        "bag_sets": 1,
        "seed_policy": "fold-wise",
        "fold_fitting_strategy": "sequential_local",
        "time_limit_seconds": 3_600,
        "model_defaults": {
            "iterations": 1_000,
            "early_stopping": True,
            "tree_mode": "catboost",
            "diagnostic_warnings": "never",
        },
    }


def _process_snapshot(pid: int) -> dict:
    """Read a minimal, non-secret process snapshot from ``ps`` and ``lsof``."""
    if pid < 1:
        raise RuntimeError("runner PID must be positive")
    try:
        os.kill(pid, 0)
    except OSError as exc:
        raise RuntimeError(f"runner PID {pid} is not active") from exc

    command = _run_command(["ps", "-p", str(pid), "-o", "command="])
    parent_pid = int(_run_command(["ps", "-p", str(pid), "-o", "ppid="]))
    started = _run_command(["ps", "-p", str(pid), "-o", "lstart="])
    started_utc = (
        datetime.strptime(started, "%a %b %d %H:%M:%S %Y")
        .astimezone()
        .astimezone(timezone.utc)
        .isoformat()
    )
    if not _command_invokes_remaining9_runner(command):
        raise RuntimeError(
            f"PID {pid} command is not the remaining-nine runner: {command!r}"
        )

    lsof = _run_command(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"])
    cwd_lines = [line[1:] for line in lsof.splitlines() if line.startswith("n")]
    if len(cwd_lines) != 1:
        raise RuntimeError(f"could not determine cwd for runner PID {pid}")
    process_with_environment = _run_command(
        ["ps", "eww", "-p", str(pid), "-o", "command="]
    )
    process_environment = {}
    for name in RELEVANT_ENVIRONMENT:
        match = re.search(
            rf"(?:^|\s){re.escape(name)}=([^\s]*)", process_with_environment
        )
        process_environment[name] = match.group(1) if match else None
    return {
        "pid": pid,
        "parent_pid": parent_pid,
        "started": started,
        "started_utc": started_utc,
        "command": command,
        "cwd": str(Path(cwd_lines[0]).resolve()),
        "environment": process_environment,
        "running_at_capture": True,
    }


def _command_invokes_remaining9_runner(command: str) -> bool:
    """Recognize direct, module, flagged, and shell-wrapped runner commands."""
    try:
        parts = shlex.split(command)
    except ValueError:
        return False

    def invokes(tokens: list[str]) -> bool:
        python_seen = False
        for index, token in enumerate(tokens):
            if Path(token).name.startswith("python"):
                python_seen = True
            if python_seen and Path(token).name == RUNNER_FILENAME:
                return True
            if (
                python_seen
                and token == RUNNER_MODULE
                and index > 0
                and tokens[index - 1] == "-m"
            ):
                return True
        for token in tokens:
            if " " not in token:
                continue
            try:
                nested = shlex.split(token)
            except ValueError:
                continue
            if nested != tokens and invokes(nested):
                return True
        return False

    return invokes(parts)


def _matching_runner_pids() -> list[int]:
    """Return all remaining-nine runners, including flagged or wrapped calls."""
    matches = []
    process_rows = _run_command(["ps", "-axo", "pid=,command="])
    for row in process_rows.splitlines():
        fields = row.strip().split(maxsplit=1)
        if len(fields) != 2:
            continue
        if _command_invokes_remaining9_runner(fields[1]):
            matches.append(int(fields[0]))
    return sorted(matches)


def collect_current_identity(
    *,
    repo_root: Path | None = None,
    declared_source_commit: str = DEFAULT_SOURCE_COMMIT,
) -> dict:
    """Collect the source and interpreter identity used to verify a manifest."""
    repo_root = _git_root(repo_root or RUNNER_PATH.parent)
    declared_source_commit = _git(
        ["rev-parse", f"{declared_source_commit}^{{commit}}"], cwd=repo_root
    )
    darkofit_tree = _git(
        ["rev-parse", f"{declared_source_commit}:darkofit"], cwd=repo_root
    )
    current_darkofit_tree = _git(["rev-parse", "HEAD:darkofit"], cwd=repo_root)
    darkofit_status = _git(
        ["status", "--porcelain=v1", "--untracked-files=all", "--", "darkofit"],
        cwd=repo_root,
    )
    if current_darkofit_tree != darkofit_tree or darkofit_status:
        raise RuntimeError(
            "current DarkoFit library tree does not exactly match the declared source commit"
        )

    import tabarena

    tabarena_module = Path(tabarena.__file__).resolve()
    tabarena_root = _git_root(tabarena_module.parent)
    tabarena_status = _git(
        ["status", "--porcelain=v1", "--untracked-files=all"], cwd=tabarena_root
    )
    if tabarena_status:
        raise RuntimeError("TabArena source checkout must be clean")

    runner_identity = _frozen_file_identity(
        path=RUNNER_PATH,
        repo_root=repo_root,
        frozen_ref=FROZEN_RUNNER_REF,
        expected_blob=FROZEN_RUNNER_GIT_BLOB,
    )
    adapter_identity = _frozen_file_identity(
        path=ADAPTER_PATH,
        repo_root=repo_root,
        frozen_ref=FROZEN_ADAPTER_REF,
        expected_blob=FROZEN_ADAPTER_GIT_BLOB,
    )

    return {
        "runner": runner_identity,
        "adapter": adapter_identity,
        "darkofit": {
            "repository_path": str(repo_root),
            "repository_head_at_capture": _git(["rev-parse", "HEAD"], cwd=repo_root),
            "declared_source_commit": declared_source_commit,
            "library_tree": darkofit_tree,
            "library_status": darkofit_status,
        },
        "tabarena": {
            "repository_path": str(tabarena_root),
            "repository_head": _git(["rev-parse", "HEAD"], cwd=tabarena_root),
            "repository_status": tabarena_status,
            "module_path": str(tabarena_module),
        },
        "python": {
            "executable": str(Path(sys.executable).absolute()),
            "resolved_executable": str(Path(sys.executable).resolve()),
            "prefix": str(Path(sys.prefix).resolve()),
            "base_prefix": str(Path(sys.base_prefix).resolve()),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "packages": _package_versions(),
        "runtime_configuration": _runtime_configuration(),
        "environment": {
            name: os.environ.get(name) for name in RELEVANT_ENVIRONMENT
        },
    }


def _result_snapshot(experiments_dir: Path) -> dict:
    paths = _result_paths(experiments_dir)
    mtimes = [path.stat().st_mtime for path in paths]
    return {
        "completed_result_files_at_capture": len(paths),
        "earliest_result_mtime_utc": (
            datetime.fromtimestamp(min(mtimes), timezone.utc).isoformat()
            if mtimes
            else None
        ),
        "latest_result_mtime_utc": (
            datetime.fromtimestamp(max(mtimes), timezone.utc).isoformat()
            if mtimes
            else None
        ),
    }


def _result_paths(experiments_dir: Path) -> list[Path]:
    paths = sorted(experiments_dir.rglob("results.pkl"))
    paths.extend(sorted(experiments_dir.rglob("results.pkl.gz")))
    return paths


def build_manifest(
    *,
    pid: int,
    output_dir: Path,
    declared_source_commit: str = DEFAULT_SOURCE_COMMIT,
    process_snapshot: Mapping | None = None,
    current_identity: Mapping | None = None,
) -> dict:
    """Build an in-flight manifest without reading any result metrics."""
    output_dir = output_dir.resolve()
    experiments_dir = output_dir / "experiments"
    identity = dict(
        current_identity
        or collect_current_identity(declared_source_commit=declared_source_commit)
    )
    if not str(identity["darkofit"]["declared_source_commit"]).startswith(
        DEFAULT_SOURCE_COMMIT
    ):
        raise RuntimeError(
            f"remaining-nine source must be frozen at {DEFAULT_SOURCE_COMMIT}"
        )
    process = dict(process_snapshot or _process_snapshot(pid))
    process_cwd = process.get("cwd")
    if not isinstance(process_cwd, str) or not process_cwd:
        raise RuntimeError("runner process snapshot is missing its cwd")
    if Path(process_cwd).resolve() != Path(
        identity["darkofit"]["repository_path"]
    ).resolve():
        raise RuntimeError("runner process cwd does not match the DarkoFit repository")
    expected_output_dir = (
        Path(identity["darkofit"]["repository_path"]) / DEFAULT_OUTPUT_DIR
    ).resolve()
    if output_dir != expected_output_dir:
        raise RuntimeError(
            f"frozen runner output must be {expected_output_dir}, got {output_dir}"
        )
    _validate_process_command(process, current_identity=identity)
    if not process.get("running_at_capture"):
        raise RuntimeError("manifest must be captured while the runner is active")
    if process.get("environment") != identity.get("environment"):
        raise RuntimeError(
            "capture process must use the runner's NUMBA_CACHE_DIR and PYTHONPATH"
        )

    started_utc = datetime.fromisoformat(str(process.get("started_utc")))
    for source_name in ("runner", "adapter"):
        source_mtime = datetime.fromisoformat(identity[source_name]["mtime_utc"])
        if source_mtime > started_utc:
            raise RuntimeError(
                f"{source_name} file was modified after the target process started"
            )
    result_snapshot = _result_snapshot(experiments_dir)
    earliest = result_snapshot["earliest_result_mtime_utc"]
    if earliest is not None and datetime.fromisoformat(earliest) < started_utc:
        raise RuntimeError("result cache contains files older than the runner process")

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "darkofit_remaining9_in_flight_run_manifest",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "experiments_dir": str(experiments_dir),
        "protocol": frozen_protocol_identity(),
        "process": process,
        "result_snapshot": result_snapshot,
        **identity,
    }


def _parse_aware_timestamp(value, field: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid {field} timestamp") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise RuntimeError(f"{field} timestamp must include a timezone")
    return timestamp


def _timestamp_ns(timestamp: datetime) -> int:
    utc_timestamp = timestamp.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = utc_timestamp - epoch
    return (
        (delta.days * 86_400 + delta.seconds) * 1_000_000_000
        + delta.microseconds * 1_000
    )


def validate_manifest_payload(
    manifest: Mapping,
    *,
    input_dir: Path,
    current_identity: Mapping,
) -> None:
    """Require a manifest to match the frozen protocol and current analysis env."""
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported remaining-nine run-manifest schema")
    if manifest.get("kind") != "darkofit_remaining9_in_flight_run_manifest":
        raise RuntimeError("unexpected remaining-nine run-manifest kind")
    if manifest.get("protocol") != frozen_protocol_identity():
        raise RuntimeError("run manifest does not match the frozen protocol")
    if Path(str(manifest.get("experiments_dir", ""))).resolve() != input_dir.resolve():
        raise RuntimeError("run manifest experiments_dir does not match --input-dir")
    expected_output_dir = (
        Path(current_identity["darkofit"]["repository_path"]) / DEFAULT_OUTPUT_DIR
    ).resolve()
    if Path(str(manifest.get("output_dir", ""))).resolve() != expected_output_dir:
        raise RuntimeError("run manifest output_dir is not the frozen runner output")

    process = manifest.get("process")
    if not isinstance(process, Mapping) or process.get("running_at_capture") is not True:
        raise RuntimeError("run manifest was not captured while the runner was active")
    if not isinstance(process.get("pid"), int) or process["pid"] < 1:
        raise RuntimeError("run manifest has an invalid runner PID")
    if RUNNER_FILENAME not in str(process.get("command", "")):
        raise RuntimeError("run manifest process command is not the frozen runner")
    process_cwd = process.get("cwd")
    if not isinstance(process_cwd, str) or not process_cwd:
        raise RuntimeError("run manifest is missing the runner cwd")
    if Path(process_cwd).resolve() != Path(
        current_identity["darkofit"]["repository_path"]
    ).resolve():
        raise RuntimeError("run manifest process cwd is not the DarkoFit repository")
    _validate_process_command(process, current_identity=current_identity)
    if process.get("environment") != manifest.get("environment"):
        raise RuntimeError("run manifest runner environment is inconsistent")
    try:
        started_value = process["started_utc"]
    except KeyError as exc:
        raise RuntimeError("run manifest is missing the process start time") from exc
    started_utc = _parse_aware_timestamp(started_value, "process start")
    captured_at = _parse_aware_timestamp(
        manifest.get("captured_at_utc"), "manifest capture"
    )
    if captured_at < started_utc:
        raise RuntimeError("run manifest predates its runner process")
    for source_name in ("runner", "adapter"):
        try:
            source_mtime_value = manifest[source_name]["mtime_utc"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"run manifest has an invalid {source_name} timestamp"
            ) from exc
        source_mtime = _parse_aware_timestamp(
            source_mtime_value, f"{source_name} mtime"
        )
        if source_mtime > started_utc:
            raise RuntimeError(
                f"run manifest {source_name} was modified after process start"
            )

    snapshot = manifest.get("result_snapshot")
    if not isinstance(snapshot, Mapping):
        raise RuntimeError("run manifest is missing the in-flight result snapshot")
    result_count = snapshot.get("completed_result_files_at_capture")
    if (
        not isinstance(result_count, int)
        or isinstance(result_count, bool)
        or result_count not in range(EXPECTED_JOBS + 1)
    ):
        raise RuntimeError("run manifest has an invalid in-flight result count")
    earliest = snapshot.get("earliest_result_mtime_utc")
    latest = snapshot.get("latest_result_mtime_utc")
    if result_count == 0:
        if earliest is not None or latest is not None:
            raise RuntimeError("empty run manifest snapshot has result timestamps")
    else:
        if earliest is None or latest is None:
            raise RuntimeError("run manifest has incomplete result timestamps")
        earliest_result = _parse_aware_timestamp(
            earliest, "earliest in-flight result mtime"
        )
        latest_result = _parse_aware_timestamp(
            latest, "latest in-flight result mtime"
        )
        if earliest_result < started_utc:
            raise RuntimeError("run manifest includes stale pre-process result files")
        if latest_result < earliest_result or latest_result > captured_at:
            raise RuntimeError("run manifest has inconsistent result timestamps")

    for section in (
        "runner",
        "adapter",
        "tabarena",
        "python",
        "packages",
        "runtime_configuration",
        "environment",
    ):
        if manifest.get(section) != current_identity.get(section):
            raise RuntimeError(f"run manifest {section} does not match analysis environment")

    recorded_darkofit = manifest.get("darkofit")
    current_darkofit = current_identity.get("darkofit")
    if not isinstance(recorded_darkofit, Mapping) or not isinstance(
        current_darkofit, Mapping
    ):
        raise RuntimeError("run manifest is missing DarkoFit source identity")
    if not str(recorded_darkofit.get("declared_source_commit", "")).startswith(
        DEFAULT_SOURCE_COMMIT
    ):
        raise RuntimeError("run manifest does not identify the frozen source commit")
    for field in (
        "repository_path",
        "declared_source_commit",
        "library_tree",
        "library_status",
    ):
        if recorded_darkofit.get(field) != current_darkofit.get(field):
            raise RuntimeError(
                f"run manifest DarkoFit {field} does not match analysis source"
            )


def validate_completion_attestation(
    attestation: Mapping,
    *,
    manifest: Mapping,
    manifest_sha256: str,
    input_dir: Path,
) -> dict[str, bytes]:
    """Bind every final result byte to the captured live runner and manifest."""
    if attestation.get("schema_version") != ATTESTATION_SCHEMA_VERSION:
        raise RuntimeError("unsupported completion-attestation schema")
    if attestation.get("kind") != "remaining9_live_completion_attestation":
        raise RuntimeError("unexpected completion-attestation kind")

    process = _as_manifest_mapping(manifest.get("process"), "process")
    runner_pid = process.get("pid")
    if (
        isinstance(runner_pid, bool)
        or not isinstance(runner_pid, int)
        or runner_pid < 1
    ):
        raise RuntimeError("run manifest has an invalid runner PID")
    if attestation.get("runner_pid") != runner_pid:
        raise RuntimeError("completion attestation runner PID does not match manifest")
    if attestation.get("runner_pids_at_completion") != [runner_pid]:
        raise RuntimeError(
            "completion attestation does not identify the sole frozen runner PID"
        )
    if attestation.get("runner_alive_at_completion") is not True:
        raise RuntimeError("runner was not alive at completion attestation")
    if attestation.get("expected_results") != EXPECTED_JOBS:
        raise RuntimeError(
            f"completion attestation must cover exactly {EXPECTED_JOBS} results"
        )
    if attestation.get("run_manifest_sha256") != manifest_sha256:
        raise RuntimeError("completion attestation run-manifest digest does not match")

    captured_at = _parse_aware_timestamp(
        manifest.get("captured_at_utc"), "manifest capture"
    )
    watch_started = _parse_aware_timestamp(
        attestation.get("watch_started_utc"), "watch start"
    )
    completed = _parse_aware_timestamp(
        attestation.get("completed_utc"), "completion"
    )
    if watch_started <= captured_at:
        raise RuntimeError(
            "completion attestation watcher did not start after manifest capture"
        )
    if completed < watch_started:
        raise RuntimeError("completion attestation predates its watcher start")

    observed = attestation.get("observed_results")
    if not isinstance(observed, Mapping) or len(observed) != EXPECTED_JOBS:
        count = len(observed) if isinstance(observed, Mapping) else "non-mapping"
        raise RuntimeError(
            f"completion attestation has {count} observed results; "
            f"expected {EXPECTED_JOBS}"
        )
    paths = _result_paths(input_dir)
    if len(paths) != EXPECTED_JOBS:
        raise RuntimeError(
            f"expected {EXPECTED_JOBS} final result files, got {len(paths)}"
        )
    actual_paths = {
        path.relative_to(input_dir).as_posix(): path for path in paths
    }
    if set(observed) != set(actual_paths):
        raise RuntimeError(
            "completion attestation result paths do not match the final cache"
        )

    try:
        runner_started = _parse_aware_timestamp(
            process["started_utc"], "process start"
        )
    except KeyError as exc:
        raise RuntimeError("run manifest is missing the process start time") from exc
    runner_started_ns = _timestamp_ns(runner_started)
    verified_payloads = {}
    for relative, path in actual_paths.items():
        item = observed[relative]
        if not isinstance(item, Mapping):
            raise RuntimeError(
                f"completion attestation entry for {relative} is not an object"
            )
        if item.get("runner_pid_alive") is not True:
            raise RuntimeError(
                f"completion attestation did not observe {relative} with runner alive"
            )
        first_stable = _parse_aware_timestamp(
            item.get("first_stable_seen_utc"),
            f"first stable observation for {relative}",
        )
        if first_stable < watch_started or first_stable > completed:
            raise RuntimeError(
                f"completion attestation observation time is invalid for {relative}"
            )
        expected_size = item.get("size_bytes")
        expected_mtime = item.get("mtime_ns")
        expected_hash = item.get("sha256")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
            or isinstance(expected_mtime, bool)
            or not isinstance(expected_mtime, int)
            or expected_mtime < 0
            or not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
        ):
            raise RuntimeError(
                f"completion attestation has invalid file metadata for {relative}"
            )
        if expected_mtime < runner_started_ns:
            raise RuntimeError(
                f"completion attestation result predates runner: {relative}"
            )
        if expected_mtime > _timestamp_ns(first_stable):
            raise RuntimeError(
                f"completion attestation result mtime follows observation: {relative}"
            )
        stat_before = path.stat()
        payload = path.read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        stat_after = path.stat()
        if (
            stat_before.st_size != stat_after.st_size
            or stat_before.st_mtime_ns != stat_after.st_mtime_ns
        ):
            raise RuntimeError(f"result file changed during verification: {relative}")
        if (
            stat_after.st_size != expected_size
            or stat_after.st_mtime_ns != expected_mtime
            or actual_hash != expected_hash
        ):
            raise RuntimeError(
                f"completion attestation does not match final result {relative}"
            )
        verified_payloads[relative] = payload
    return verified_payloads


def load_and_verify_completion_attestation(
    path: Path,
    *,
    manifest: Mapping,
    manifest_sha256: str,
    input_dir: Path,
) -> tuple[dict, str, dict[str, bytes]]:
    """Load one attestation read and verify it against all final result bytes."""
    try:
        payload = path.read_bytes()
        attestation = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"failed to read completion attestation {path}: {exc}"
        ) from exc
    if not isinstance(attestation, Mapping):
        raise RuntimeError("completion-attestation payload must be an object")
    verified_payloads = validate_completion_attestation(
        attestation,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        input_dir=input_dir,
    )
    return (
        dict(attestation),
        hashlib.sha256(payload).hexdigest(),
        verified_payloads,
    )


def _validate_process_command(
    process: Mapping, *, current_identity: Mapping
) -> None:
    """Bind a captured PID to the exact Python, runner, and default arguments."""
    try:
        parts = shlex.split(str(process["command"]))
    except (KeyError, ValueError) as exc:
        raise RuntimeError("run manifest has an invalid runner command") from exc
    if len(parts) != 2:
        raise RuntimeError("frozen runner command must use its declared defaults")
    executable = Path(parts[0]).resolve()
    recorded_executable = Path(parts[0]).absolute()
    expected_recorded = Path(current_identity["python"]["executable"]).absolute()
    expected_resolved = Path(
        current_identity["python"]["resolved_executable"]
    ).resolve()
    if recorded_executable != expected_recorded or executable != expected_resolved:
        raise RuntimeError("runner PID does not use the recorded Python executable")
    cwd = Path(str(process["cwd"]))
    runner = (cwd / parts[1]).resolve()
    if runner != Path(current_identity["runner"]["path"]).resolve():
        raise RuntimeError("runner PID is not executing the recorded runner path")


def _as_manifest_mapping(value, field: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"run manifest {field} must be an object")
    return value


def load_and_verify_manifest(path: Path, *, input_dir: Path) -> tuple[dict, str]:
    """Load a JSON manifest and verify it against the current source/env."""
    try:
        payload = path.read_bytes()
        manifest = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to read run manifest {path}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise RuntimeError("run manifest payload must be an object")
    darkofit = manifest.get("darkofit")
    if not isinstance(darkofit, Mapping):
        raise RuntimeError("run manifest is missing DarkoFit source identity")
    source_commit = str(darkofit.get("declared_source_commit", ""))
    current = collect_current_identity(declared_source_commit=source_commit)
    validate_manifest_payload(manifest, input_dir=input_dir, current_identity=current)
    return dict(manifest), hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def watch_completion(
    *,
    manifest_path: Path,
    attestation_path: Path,
    pid: int,
    poll_interval: float = 0.1,
    timeout: float = 14_400.0,
) -> dict:
    """Hash all final results while the sole captured runner is still alive."""
    if poll_interval <= 0.0 or timeout <= 0.0:
        raise RuntimeError("watch poll interval and timeout must be positive")
    if attestation_path.exists():
        raise RuntimeError(
            f"refusing to overwrite existing attestation {attestation_path}"
        )

    raw_manifest = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
    manifest, verified_manifest_sha256 = load_and_verify_manifest(
        manifest_path,
        input_dir=Path(json.loads(raw_manifest)["experiments_dir"]),
    )
    if verified_manifest_sha256 != manifest_sha256:
        raise RuntimeError("run manifest changed while starting completion watcher")
    process = _as_manifest_mapping(manifest.get("process"), "process")
    if process.get("pid") != pid:
        raise RuntimeError("watch PID does not match the run manifest")
    watch_started = datetime.now(timezone.utc)
    captured_at = _parse_aware_timestamp(
        manifest.get("captured_at_utc"), "manifest capture"
    )
    if watch_started <= captured_at:
        raise RuntimeError("completion watcher must start after manifest capture")

    initial_process = _process_snapshot(pid)
    for field in ("pid", "started_utc", "command", "cwd", "environment"):
        if initial_process.get(field) != process.get(field):
            raise RuntimeError(f"runner process {field} changed before watch start")
    experiments_dir = Path(str(manifest["experiments_dir"]))
    previous_stats: dict[str, tuple[int, int]] = {}
    observed: dict[str, dict] = {}
    deadline = time.monotonic() + timeout

    while True:
        if time.monotonic() > deadline:
            raise RuntimeError("timed out waiting for remaining-nine completion")
        if _matching_runner_pids() != [pid]:
            raise RuntimeError("frozen runner is not the sole matching live process")
        paths = _result_paths(experiments_dir)
        if len(paths) > EXPECTED_JOBS:
            raise RuntimeError(f"result cache contains more than {EXPECTED_JOBS} files")

        current_relatives = set()
        for path in paths:
            relative = path.relative_to(experiments_dir).as_posix()
            current_relatives.add(relative)
            if relative in observed:
                continue
            stat = path.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            if previous_stats.get(relative) == signature:
                payload = path.read_bytes()
                final_stat = path.stat()
                if signature == (final_stat.st_size, final_stat.st_mtime_ns):
                    observed[relative] = {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": final_stat.st_size,
                        "mtime_ns": final_stat.st_mtime_ns,
                        "first_stable_seen_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "runner_pid_alive": True,
                    }
            previous_stats[relative] = signature
        previous_stats = {
            relative: signature
            for relative, signature in previous_stats.items()
            if relative in current_relatives
        }

        if len(paths) == EXPECTED_JOBS and len(observed) == EXPECTED_JOBS:
            if set(observed) != current_relatives:
                raise RuntimeError("result cache changed during completion watch")
            for path in paths:
                relative = path.relative_to(experiments_dir).as_posix()
                item = observed[relative]
                stat = path.stat()
                if (
                    stat.st_size != item["size_bytes"]
                    or stat.st_mtime_ns != item["mtime_ns"]
                    or sha256_file(path) != item["sha256"]
                ):
                    raise RuntimeError(
                        f"result file changed after stable observation: {relative}"
                    )
            if _matching_runner_pids() != [pid]:
                raise RuntimeError("runner exited before completion attestation")
            completion_process = _process_snapshot(pid)
            for field in ("pid", "started_utc", "command", "cwd", "environment"):
                if completion_process.get(field) != process.get(field):
                    raise RuntimeError(
                        f"runner process {field} changed before completion"
                    )
            final_manifest = manifest_path.read_bytes()
            if hashlib.sha256(final_manifest).hexdigest() != manifest_sha256:
                raise RuntimeError("run manifest changed during completion watch")
            attestation = {
                "schema_version": ATTESTATION_SCHEMA_VERSION,
                "kind": "remaining9_live_completion_attestation",
                "watch_started_utc": watch_started.isoformat(),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "runner_pid": pid,
                "runner_pids_at_completion": [pid],
                "runner_alive_at_completion": True,
                "expected_results": EXPECTED_JOBS,
                "observed_results": observed,
                "run_manifest_sha256": manifest_sha256,
            }
            _write_json_atomic(attestation_path, attestation)
            return attestation
        time.sleep(poll_interval)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--watch-completion", action="store_true")
    parser.add_argument("--attestation", type=Path)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=14_400.0)
    parser.add_argument("--source-commit", default=DEFAULT_SOURCE_COMMIT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else output_dir / "run_manifest.json"
    )
    if args.watch_completion:
        attestation_path = (
            args.attestation.resolve()
            if args.attestation is not None
            else output_dir / "completion_attestation.live.json"
        )
        if args.force and attestation_path.exists():
            attestation_path.unlink()
        attestation = watch_completion(
            manifest_path=manifest_path,
            attestation_path=attestation_path,
            pid=args.pid,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
        )
        print(
            "REMAINING9_COMPLETION_ATTESTED "
            f"pid={args.pid} results={len(attestation['observed_results'])} "
            f"path={attestation_path}"
        )
        return 0
    if args.attestation is not None:
        raise RuntimeError("--attestation requires --watch-completion")
    if manifest_path.exists() and not args.force:
        raise RuntimeError(f"refusing to overwrite existing manifest {manifest_path}")
    manifest = build_manifest(
        pid=args.pid,
        output_dir=output_dir,
        declared_source_commit=args.source_commit,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    print(
        "REMAINING9_MANIFEST_CAPTURED "
        f"pid={args.pid} results={manifest['result_snapshot']['completed_result_files_at_capture']} "
        f"path={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
