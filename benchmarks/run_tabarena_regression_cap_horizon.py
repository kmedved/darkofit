"""Run the frozen 1,000-vs-10,000-round TabArena regression campaign."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import math
import os
import platform
import pickle
import re
import stat
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from statistics import mean
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit


# Keep the tasks in ascending TabArena task-id order.  The insertion order is
# also the canonical execution order used by ``interleave_horizon_jobs``.
TASK_SPLIT_COUNTS = {
    "airfoil_self_noise": (363612, 30),
    "Another-Dataset-on-used-Fiat-500": (363615, 30),
    "concrete_compressive_strength": (363625, 30),
    "diamonds": (363631, 9),
    "Food_Delivery_Time": (363672, 9),
    "healthcare_insurance_expenses": (363675, 30),
    "houses": (363678, 9),
    "miami_housing": (363686, 9),
    "physiochemical_protein": (363693, 9),
    "QSAR-TID-11": (363697, 9),
    "QSAR_fish_toxicity": (363698, 30),
    "superconductivity": (363705, 9),
    "wine_quality": (363708, 9),
}
TASK_IDS = [task_id for task_id, _ in TASK_SPLIT_COUNTS.values()]
SPLIT_INDICES = [f"r{repeat}f{fold}" for repeat in range(10) for fold in range(3)]

COMMON_CONFIG: dict[str, Any] = {
    "tree_mode": "catboost",
    "l2_leaf_reg": 3.0,
    "max_bins": 128,
    "learning_rate": 0.1,
    "ts_permutations": 1,
    "early_stopping": True,
    "use_best_model": True,
}
HORIZON_ARMS = {
    "cap1000": {**COMMON_CONFIG, "iterations": 1_000},
    "cap10000": {**COMMON_CONFIG, "iterations": 10_000},
}
HORIZONS = tuple(config["iterations"] for config in HORIZON_ARMS.values())

EXPECTED_DATASET_SPLITS = sum(count for _, count in TASK_SPLIT_COUNTS.values())
EXPECTED_JOBS = len(HORIZON_ARMS) * EXPECTED_DATASET_SPLITS
EXPECTED_CHILD_FITS = EXPECTED_JOBS * 8
TIME_LIMIT_SECONDS = 3_600.0

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "run_manifest.json"
COMPLETION_ATTESTATION_FILENAME = "completion_attestation.json"
ANALYSIS_PAYLOAD_FILENAME = "analysis_payload.json"
WARMUP_HISTORY_FILENAME = "warmup_history.json"
RESUME_HISTORY_FILENAME = "resume_history.json"
DEFAULT_ANALYSIS_OUTPUT_FILENAMES = (
    "paired_splits.csv",
    "per_repeat.csv",
    "paired_children.csv",
    "summary.json",
    "report.md",
)
DEFAULT_OUTPUT_DIR = Path(
    ".cache/tabarena-regression-cap-horizon-0.9.0-20260713"
)
SOURCE_FILES = (
    Path("pyproject.toml"),
    Path("darkofit/__init__.py"),
    Path("darkofit/booster.py"),
    Path("darkofit/callbacks.py"),
    Path("darkofit/serialization.py"),
    Path("darkofit/sklearn_api.py"),
    Path("benchmarks/tabarena_adapter.py"),
    Path("benchmarks/tabarena_warmup.py"),
    Path("benchmarks/run_tabarena_regression_cap_horizon.py"),
    Path("benchmarks/analyze_tabarena_regression_cap_horizon.py"),
    Path("benchmarks/tabarena_regression_cap_horizon_protocol.md"),
)
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
    # TabArena's peak-RSS metric is implemented by
    # ``tabarena.utils.memory_utils.CPUMemoryTracker`` using psutil.
    "psutil",
)
RUNTIME_ENVIRONMENT_KEYS = (
    "TABARENA_DISABLE_RESULT_COMPRESSION",
    "NUMBA_CACHE_DIR",
    "NUMBA_NUM_THREADS",
    # These variables change Numba's execution backend or whether compiled
    # kernels execute at all.
    "NUMBA_THREADING_LAYER",
    "NUMBA_THREADING_LAYER_PRIORITY",
    "NUMBA_DISABLE_JIT",
    # These variables change generated CPU code and can therefore affect both
    # the numerical and resource metrics used by this campaign.
    "NUMBA_CPU_NAME",
    "NUMBA_CPU_FEATURES",
    "NUMBA_OPT",
    "NUMBA_LOOP_VECTORIZE",
    "NUMBA_SLP_VECTORIZE",
    "NUMBA_ENABLE_AVX",
    "NUMBA_BOUNDSCHECK",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "PYTHONHASHSEED",
)
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


def expected_resolved_method_hyperparameters(arm: str) -> dict[str, Any]:
    """Return AutoGluon's resolved bag-level view of one frozen arm."""
    if arm not in HORIZON_ARMS:
        raise RuntimeError(f"unexpected horizon arm: {arm}")
    ensemble = expected_ag_ensemble_config()
    max_time_limit = ensemble.pop("ag.max_time_limit")
    ensemble["ag_args_fit"] = {"max_time_limit": max_time_limit}
    return {**HORIZON_ARMS[arm], "ag_args_ensemble": ensemble}


def expected_fit_kwargs_extra(num_cpus: int) -> dict[str, Any]:
    """Return the resolved TabArena/AutoGluon bag construction settings."""
    num_cpus = _exact_int(num_cpus, "expected fit num_cpus")
    if num_cpus < 1:
        raise RuntimeError("expected fit num_cpus must be positive")
    return {
        "num_bag_folds": 8,
        "num_bag_sets": 1,
        "raise_on_model_failure": True,
        "calibrate": False,
        "num_cpus": num_cpus,
    }


def expected_child_hyperparameters(arm: str, child_fold: int) -> dict[str, Any]:
    """Return the exact initialized child policy for one fold-wise seed."""
    if isinstance(child_fold, bool) or child_fold not in range(8):
        raise RuntimeError("child_fold must be an integer from 0 through 7")
    return {
        **HORIZON_ARMS[arm],
        "diagnostic_warnings": "never",
        "random_state": int(child_fold),
    }


def expected_coordinates() -> list[tuple[str, int, int]]:
    """Return the exact dataset/repeat/fold coordinates in execution order."""
    coordinates = []
    for dataset, (_, split_count) in TASK_SPLIT_COUNTS.items():
        if split_count % 3:
            raise RuntimeError(f"split count for {dataset} is not fold-aligned")
        coordinates.extend(
            (dataset, registered_fold // 3, registered_fold % 3)
            for registered_fold in range(split_count)
        )
    return coordinates


def frozen_protocol() -> dict[str, Any]:
    """Return the JSON-serializable causal experiment specification."""
    return {
        "task_split_counts": {
            dataset: {"task_id": task_id, "split_count": split_count}
            for dataset, (task_id, split_count) in TASK_SPLIT_COUNTS.items()
        },
        "split_indices_requested": list(SPLIT_INDICES),
        "coordinates": [
            {"dataset": dataset, "repeat": repeat, "fold": fold}
            for dataset, repeat, fold in expected_coordinates()
        ],
        "arms": {name: dict(config) for name, config in HORIZON_ARMS.items()},
        "expected_dataset_splits": EXPECTED_DATASET_SPLITS,
        "expected_jobs": EXPECTED_JOBS,
        "expected_child_fits": EXPECTED_CHILD_FITS,
        "bag_folds": 8,
        "bag_sets": 1,
        "seed_policy": "fold-wise",
        "seed_configuration": {
            "model_random_seed": 0,
            "vary_seed_across_folds": True,
        },
        "fold_fitting_strategy": "sequential_local",
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "job_order": (
            "adjacent task/split pairs; cap1000 first on even coordinates and "
            "cap10000 first on odd coordinates"
        ),
        "warmup_outside_run_jobs": True,
        "warmup_thread_policy": (
            "resolve and pin the built sequential child num_cpus; warmup config "
            "and fitted thread count must equal it"
        ),
        "chimera_role": "coverage validation only; excluded from horizon selection",
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def protocol_sha256() -> str:
    """Return a stable fingerprint of the frozen campaign matrix."""
    return hashlib.sha256(_canonical_json(frozen_protocol())).hexdigest()


def validate_chimera_coverage(results) -> None:
    """Validate matching registered ChimeraBoost rows without scoring them."""
    required = {"dataset", "method", "fold", "imputed"}
    missing_columns = required.difference(results.columns)
    if missing_columns:
        raise RuntimeError(
            "registered CHIMERA results lack columns: "
            + ", ".join(sorted(missing_columns))
        )

    selected = results[
        results["dataset"].isin(TASK_SPLIT_COUNTS)
        & (results["method"] == "CHIMERA (default)")
    ]
    if selected["imputed"].isna().any() or selected["imputed"].astype(bool).any():
        raise RuntimeError("registered CHIMERA coverage contains imputed rows")
    if selected.duplicated(["dataset", "fold"], keep=False).any():
        raise RuntimeError("registered CHIMERA coverage contains duplicate rows")

    for dataset, (_, split_count) in TASK_SPLIT_COUNTS.items():
        observed = sorted(
            int(value)
            for value in selected.loc[selected["dataset"] == dataset, "fold"]
        )
        expected = list(range(split_count))
        if observed != expected:
            raise RuntimeError(
                f"unexpected registered CHIMERA folds for {dataset}: "
                f"expected {expected[0]}..{expected[-1]}, got {observed}"
            )
    if len(selected) != EXPECTED_DATASET_SPLITS:
        raise RuntimeError(
            f"expected {EXPECTED_DATASET_SPLITS} CHIMERA rows, got {len(selected)}"
        )


def build_experiments(*, model_cls, config_generator_cls, time_limit: float):
    """Create one explicitly named bag experiment for each frozen arm."""
    if float(time_limit) != TIME_LIMIT_SECONDS:
        raise ValueError(
            f"frozen campaign time_limit must be {TIME_LIMIT_SECONDS:g} seconds"
        )
    experiments = []
    for arm_name, config in HORIZON_ARMS.items():
        generator = config_generator_cls(
            model_cls=model_cls,
            manual_configs=[dict(config)],
            search_space={},
        )
        generated = generator.generate_all_bag_experiments(
            num_random_configs=0,
            name_id_suffix=f"_{arm_name}_horizon",
            add_seed="fold-wise",
            fold_fitting_strategy="sequential_local",
            time_limit=time_limit,
        )
        if len(generated) != 1:
            raise RuntimeError(
                f"expected one experiment for {arm_name}, got {len(generated)}"
            )
        experiments.extend(generated)
    return experiments


def expected_ag_ensemble_config() -> dict[str, Any]:
    """Return the exact TabArena bag/seed/resource metadata for each arm."""
    return {
        "model_random_seed": 0,
        "vary_seed_across_folds": True,
        "fold_fitting_strategy": "sequential_local",
        "ag.max_time_limit": TIME_LIMIT_SECONDS,
    }


def _job_coordinate(job) -> tuple[str, int, int]:
    return job.task.dataset, int(job.task.repeat), int(job.task.fold)


def _job_horizon(job) -> int:
    try:
        value = job.experiment.method_kwargs["model_hyperparameters"]["iterations"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise RuntimeError("could not identify a job's horizon arm") from exc
    return int(value)


def interleave_horizon_jobs(jobs: Iterable[Any]) -> list[Any]:
    """Pair arms per coordinate and reverse pair order on alternating splits."""
    grouped: dict[tuple[str, int, int], list[Any]] = defaultdict(list)
    for job in jobs:
        grouped[_job_coordinate(job)].append(job)

    expected = expected_coordinates()
    unexpected = set(grouped).difference(expected)
    if unexpected:
        raise RuntimeError(f"built jobs contain unexpected coordinates: {unexpected}")

    ordered = []
    for coordinate_index, coordinate in enumerate(expected):
        horizon_jobs: dict[int, Any] = {}
        for job in grouped.get(coordinate, []):
            horizon = _job_horizon(job)
            if horizon in horizon_jobs:
                raise RuntimeError(
                    f"duplicate {horizon}-round job for coordinate {coordinate}"
                )
            horizon_jobs[horizon] = job
        if set(horizon_jobs) != set(HORIZONS):
            raise RuntimeError(
                f"coordinate {coordinate} has horizons "
                f"{sorted(horizon_jobs)}, expected {sorted(HORIZONS)}"
            )
        pair_order = HORIZONS if coordinate_index % 2 == 0 else HORIZONS[::-1]
        ordered.extend(horizon_jobs[horizon] for horizon in pair_order)

    if len(ordered) != EXPECTED_JOBS:
        raise RuntimeError(f"expected {EXPECTED_JOBS} jobs, built {len(ordered)}")
    return ordered


def _autogluon_cpu_count() -> int:
    """Resolve the CPU pool through the same manager AutoGluon fits use."""
    from autogluon.common.utils.resource_utils import get_resource_manager

    return _exact_int(
        get_resource_manager().get_cpu_count(), "AutoGluon CPU allocation"
    )


def resolve_and_pin_child_cpu_allocation(jobs: Iterable[Any]) -> int:
    """Resolve and pin the sequential-child CPU allocation on built jobs.

    The campaign deliberately leaves predictor resources automatic.  Resolve
    that automatic total with AutoGluon's own resource manager, then apply the
    sequential fold strategy's ``min(child default, total)`` rule using the
    model class carried by each built experiment.  Any explicit total resource
    is honored, and every arm must resolve identically.
    """
    allocations: set[int] = set()
    resolved_methods: list[dict[str, Any]] = []
    seen_experiments: set[int] = set()
    seen_arms: set[str] = set()
    for job in jobs:
        experiment = getattr(job, "experiment", None)
        if experiment is None or id(experiment) in seen_experiments:
            continue
        seen_experiments.add(id(experiment))
        method = getattr(experiment, "method_kwargs", None)
        if not isinstance(method, dict):
            raise RuntimeError("built experiment has no resolved method configuration")
        resolved_methods.append(method)
        model_hyperparameters = method.get("model_hyperparameters")
        if not isinstance(model_hyperparameters, Mapping):
            raise RuntimeError("built experiment has no model hyperparameters")
        raw = dict(model_hyperparameters)
        ag_args = raw.pop("ag_args", None)
        ag_ensemble = raw.pop("ag_args_ensemble", None)
        matching_arms = [arm for arm, config in HORIZON_ARMS.items() if raw == config]
        if len(matching_arms) != 1:
            raise RuntimeError("built experiment does not match one frozen arm")
        arm = matching_arms[0]
        seen_arms.add(arm)
        if ag_args != {"name_suffix": f"_c1_{arm}_horizon"} or not isinstance(
            ag_ensemble, Mapping
        ) or dict(ag_ensemble) != expected_ag_ensemble_config():
            raise RuntimeError("built experiment bag configuration is not frozen")

        fit_kwargs = method.get("fit_kwargs")
        if not isinstance(fit_kwargs, dict):
            raise RuntimeError("built experiment has no resolved fit configuration")
        raw_total = fit_kwargs.get("num_cpus", "auto")
        total_cpus = (
            _autogluon_cpu_count()
            if raw_total is None or raw_total == "auto"
            else _exact_int(raw_total, "built experiment num_cpus")
        )
        if total_cpus < 1:
            raise RuntimeError("built experiment CPU allocation must be positive")

        model_cls = method.get("model_cls")
        if not isinstance(model_cls, type):
            raise RuntimeError("built experiment model class is not resolved")
        try:
            resource_probe = model_cls(
                path="",
                name="DarkoFitResourceProbe",
                problem_type="regression",
                eval_metric="root_mean_squared_error",
                hyperparameters=dict(HORIZON_ARMS[arm]),
            )
            default_cpus, default_gpus = resource_probe._get_default_resources()
        except Exception as exc:
            raise RuntimeError("could not resolve child model resources") from exc
        default_cpus = _exact_int(default_cpus, "child default num_cpus")
        if default_cpus < 1 or float(default_gpus) != 0.0:
            raise RuntimeError("child default resources are not CPU-only")
        allocations.add(min(total_cpus, default_cpus))

    if seen_arms != set(HORIZON_ARMS) or len(allocations) != 1:
        raise RuntimeError(
            "built horizon arms do not share one child CPU allocation"
        )
    allocation = next(iter(allocations))
    for method in resolved_methods:
        # Make the resource implicit in AutoGluon's sequential strategy explicit
        # on the exact experiment objects that ``run_jobs`` will execute.
        method["fit_kwargs"]["num_cpus"] = allocation
    return allocation


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_command(args: list[str], *, cwd: Path) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"command failed: {' '.join(args)}") from exc
    return result.stdout.strip()


def _git_path_list(args: list[str], *, cwd: Path) -> list[str]:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"command failed: {' '.join(args)}") from exc
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in result.stdout.split(b"\0")
        if item
    ]


def _repository_status(repository: Path, output_dir: Path | None) -> str:
    """Report every code-tree change, ignoring only untracked run output."""
    tracked = _git_path_list(
        [
            "git",
            "diff",
            "--name-only",
            "--no-ext-diff",
            "--no-renames",
            "-z",
            "HEAD",
            "--",
        ],
        cwd=repository,
    )
    untracked = _git_path_list(
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "--"],
        cwd=repository,
    )
    output_relative = None
    if output_dir is not None:
        try:
            output_relative = output_dir.resolve().relative_to(repository.resolve())
        except ValueError:
            pass
        if output_relative == Path("."):
            output_relative = None
    relevant_untracked = []
    for path in untracked:
        candidate = Path(path)
        if output_relative is not None and (
            candidate == output_relative or output_relative in candidate.parents
        ):
            continue
        relevant_untracked.append(path)
    return "\n".join(
        [
            *(f"tracked:{path}" for path in tracked),
            *(f"untracked:{path}" for path in relevant_untracked),
        ]
    )


def _sanitize_git_remote(remote: str) -> str:
    """Remove URL credentials/query material before persisting provenance."""
    if "://" not in remote:
        if "@" in remote:
            _, suffix = remote.rsplit("@", 1)
            if ":" in suffix:
                return suffix
        return remote
    parsed = urlsplit(remote)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    if parsed.port is not None:
        hostname = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))


def collect_source_provenance(output_dir: Path | None = None) -> dict[str, Any]:
    """Bind a run to committed model, adapter, warmup, and runner sources."""
    repo_root = Path(__file__).resolve().parents[1]
    status = _repository_status(repo_root, output_dir)
    if status:
        raise RuntimeError(
            "campaign repository is not clean; commit or remove every tracked "
            f"and untracked change before executing:\n{status}"
        )

    files = {}
    for relative in SOURCE_FILES:
        path = repo_root / relative
        files[str(relative)] = {
            "sha256": _sha256_file(path),
            "git_blob": _run_command(["git", "hash-object", str(path)], cwd=repo_root),
        }
    darkofit_import = collect_git_dependency_provenance(
        "darkofit", output_dir=output_dir
    )
    if Path(darkofit_import["repository"]).resolve() != repo_root.resolve():
        raise RuntimeError(
            "imported darkofit does not resolve to the campaign repository: "
            f"{darkofit_import['module_file']}"
        )
    return {
        "repository": str(repo_root),
        "git_head": _run_command(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "git_tree": _run_command(["git", "rev-parse", "HEAD^{tree}"], cwd=repo_root),
        "relevant_status": status,
        "files": files,
        "darkofit_import": darkofit_import,
        "tabarena": collect_git_dependency_provenance(
            "tabarena", output_dir=output_dir
        ),
    }


def collect_git_dependency_provenance(
    module_name: str, *, output_dir: Path | None = None
) -> dict[str, Any]:
    """Require an imported source checkout to be clean and Git-pinned."""
    module = importlib.import_module(module_name)
    module_file = Path(module.__file__).resolve()
    result = subprocess.run(
        ["git", "-C", str(module_file.parent), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{module_name} import at {module_file} is not from a Git checkout"
        )
    repository = Path(result.stdout.strip()).resolve()
    status = _repository_status(repository, output_dir)
    if status:
        raise RuntimeError(
            f"editable dependency {module_name} is not clean at {repository}:\n"
            f"{status}"
        )
    remote = _sanitize_git_remote(
        _run_command(
            ["git", "remote", "get-url", "origin"],
            cwd=repository,
        )
    )
    return {
        "module": module_name,
        "module_file": str(module_file),
        "repository": str(repository),
        "git_head": _run_command(["git", "rev-parse", "HEAD"], cwd=repository),
        "git_tree": _run_command(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=repository
        ),
        "git_remote_origin": remote,
        "status": status,
    }


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _stable_host_identity_sha256() -> str:
    """Return a privacy-preserving identifier for the physical host.

    Prefer an operating-system machine identifier over a hostname so an
    explicit resume cannot silently combine timing or memory observations from
    two otherwise identical machines.  Only a digest is persisted.
    """
    identifiers: list[str] = []
    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        try:
            value = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            identifiers.append(f"machine-id:{value}")
            break
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            pass
        else:
            match = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', result.stdout)
            if match:
                identifiers.append(f"io-platform-uuid:{match.group(1)}")
    if not identifiers:
        node = platform.node().strip()
        if not node:
            raise RuntimeError("could not determine a stable host identity")
        identifiers.append(f"node:{node}")
    return hashlib.sha256("\0".join(identifiers).encode("utf-8")).hexdigest()


def collect_runtime_hardware_provenance() -> dict[str, Any]:
    """Capture host identity and resources that affect benchmark timing."""
    logical_cpu_count = os.cpu_count()
    physical_cpu_count = None
    total_memory_bytes = None
    if platform.system() == "Darwin":
        for field, key in (
            ("physical", "hw.physicalcpu"),
            ("memory", "hw.memsize"),
        ):
            try:
                result = subprocess.run(
                    ["sysctl", "-n", key],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                value = int(result.stdout.strip())
            except (OSError, ValueError, subprocess.CalledProcessError):
                continue
            if field == "physical":
                physical_cpu_count = value
            else:
                total_memory_bytes = value
    else:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            page_count = os.sysconf("SC_PHYS_PAGES")
        except (AttributeError, OSError, ValueError):
            pass
        else:
            total_memory_bytes = int(page_size) * int(page_count)

    affinity_count = logical_cpu_count
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, NotImplementedError, OSError):
        pass
    else:
        affinity_count = len(affinity)
    return {
        "host_identity_sha256": _stable_host_identity_sha256(),
        "logical_cpu_count": logical_cpu_count,
        "physical_cpu_count": physical_cpu_count,
        "process_cpu_affinity_count": affinity_count,
        "total_memory_bytes": total_memory_bytes,
    }


def build_run_manifest(
    *,
    output_dir: Path,
    time_limit: float,
    source: dict[str, Any],
    resolved_child_num_cpus: int,
) -> dict[str, Any]:
    """Build the provenance record that owns one resumable result cache."""
    if float(time_limit) != TIME_LIMIT_SECONDS:
        raise ValueError("manifest time limit does not match the frozen protocol")
    resolved_child_num_cpus = _exact_int(
        resolved_child_num_cpus, "resolved child num_cpus"
    )
    if resolved_child_num_cpus < 1:
        raise ValueError("resolved child num_cpus must be positive")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "darkofit_tabarena_regression_cap_horizon",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir.resolve()),
        "time_limit_seconds": float(time_limit),
        "resolved_child_num_cpus": resolved_child_num_cpus,
        "protocol_sha256": protocol_sha256(),
        "protocol": frozen_protocol(),
        "source": source,
        "runtime": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": _package_versions(),
            "environment": {
                key: os.environ.get(key) for key in RUNTIME_ENVIRONMENT_KEYS
            },
            "hardware": collect_runtime_hardware_provenance(),
        },
    }


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Durably replace JSON without following a predictable temp symlink."""
    try:
        target_metadata = path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise RuntimeError(f"could not inspect JSON target: {path}") from exc
    else:
        if stat.S_ISLNK(target_metadata.st_mode):
            raise RuntimeError(f"JSON target must not be a symbolic link: {path}")
        if not stat.S_ISREG(target_metadata.st_mode):
            raise RuntimeError(f"JSON target must be a regular file: {path}")
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_output_state(output_dir: Path, *, resume: bool) -> None:
    """Reject accidental cache reuse before performing any measured work."""
    if not output_dir.exists():
        if resume:
            raise RuntimeError(f"cannot resume missing output directory: {output_dir}")
        return
    entries = list(output_dir.iterdir())
    if resume:
        if not (output_dir / MANIFEST_FILENAME).is_file():
            raise RuntimeError("resume requested but run_manifest.json is missing")
    elif entries:
        raise RuntimeError(
            f"output directory is not empty: {output_dir}; choose a new directory "
            "or pass --resume after verifying its manifest"
        )


def write_or_validate_run_manifest(
    output_dir: Path, manifest: dict[str, Any], *, resume: bool
) -> dict[str, Any]:
    """Create a zero-start cache or prove an explicit resume is compatible."""
    validate_output_state(output_dir, resume=resume)
    manifest_path = output_dir / MANIFEST_FILENAME
    if not resume:
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(manifest_path, manifest)
        return manifest

    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("could not read existing run manifest") from exc
    stable_fields = (
        "schema_version",
        "kind",
        "output_dir",
        "time_limit_seconds",
        "resolved_child_num_cpus",
        "protocol_sha256",
        "protocol",
        "source",
        "runtime",
    )
    mismatches = [field for field in stable_fields if existing.get(field) != manifest.get(field)]
    if mismatches:
        raise RuntimeError(
            "resume manifest does not match this campaign: " + ", ".join(mismatches)
        )
    return existing


def record_warmup(output_dir: Path, warmup: dict[str, Any]) -> None:
    """Append one process-local warmup record before entering ``run_jobs``."""
    path = output_dir / WARMUP_HISTORY_FILENAME
    history = []
    if path.exists():
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("could not read warmup history") from exc
    if not isinstance(history, list):
        raise RuntimeError("warmup history must contain a JSON list")
    history.append(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "warmup": warmup,
        }
    )
    _atomic_write_json(path, history)


def _arm_for_horizon(horizon: int) -> str:
    matches = [
        arm
        for arm, config in HORIZON_ARMS.items()
        if config["iterations"] == horizon
    ]
    if len(matches) != 1:
        raise RuntimeError(f"unexpected horizon arm: {horizon}")
    return matches[0]


def _cached_result_issue(path: Path, job: Any) -> str | None:
    """Return why a trusted runner cache cannot count as a complete result."""
    if not path.exists():
        return "missing"
    if not path.is_file():
        return "not_a_file"
    try:
        record = _decode_result_pickle(path.read_bytes(), str(path))
    except (OSError, RuntimeError):
        return "unreadable"

    try:
        def require_mapping(value: Any, field: str) -> Mapping[str, Any]:
            if not isinstance(value, Mapping):
                raise RuntimeError(f"{field} must be a mapping")
            return value

        dataset, repeat, fold = _job_coordinate(job)
        horizon = _job_horizon(job)
        arm = _arm_for_horizon(horizon)
        expected_suffix = f"_c1_{arm}_horizon"
        task_id = TASK_SPLIT_COUNTS[dataset][0]

        if record.get("problem_type") != "regression" or record.get("metric") != "rmse":
            raise RuntimeError("wrong problem type or metric")
        imputed = record.get("imputed", False)
        if imputed is not False and imputed is not None:
            raise RuntimeError("imputed result")
        for field in (
            "metric_error",
            "metric_error_val",
            "time_train_s",
            "time_infer_s",
        ):
            _positive_finite(record.get(field), field)
        memory = require_mapping(record.get("memory_usage"), "memory_usage")
        _positive_finite(memory.get("peak_mem_cpu"), "peak_mem_cpu")

        experiment = require_mapping(
            record.get("experiment_metadata"), "experiment_metadata"
        )
        if (
            experiment.get("experiment_cls") != "OOFExperimentRunner"
            or experiment.get("method_cls") != "AGSingleBagWrapper"
        ):
            raise RuntimeError("wrong experiment implementation")
        task = require_mapping(record.get("task_metadata"), "task_metadata")
        if (
            task.get("name") != dataset
            or _exact_int(task.get("tid"), "task id") != task_id
            or _exact_int(task.get("repeat"), "repeat") != repeat
            or _exact_int(task.get("fold"), "fold") != fold
        ):
            raise RuntimeError("wrong task coordinate")
        registered_fold = 3 * repeat + fold
        if task.get("split_idx") is not None and _exact_int(
            task["split_idx"], "split_idx"
        ) != registered_fold:
            raise RuntimeError("wrong registered split")

        method = require_mapping(record.get("method_metadata"), "method_metadata")
        raw = dict(
            require_mapping(
                method.get("model_hyperparameters"), "model_hyperparameters"
            )
        )
        ag_args = raw.pop("ag_args", None)
        ag_ensemble = raw.pop("ag_args_ensemble", None)
        if raw != HORIZON_ARMS[arm]:
            raise RuntimeError("wrong horizon configuration")
        if ag_args != {"name_suffix": expected_suffix}:
            raise RuntimeError("wrong model suffix")
        if not isinstance(ag_ensemble, Mapping) or dict(
            ag_ensemble
        ) != expected_ag_ensemble_config():
            raise RuntimeError("wrong bag configuration")
        if record.get("framework") != f"DarkoFit{expected_suffix}_BAG_L1":
            raise RuntimeError("wrong framework")
        resolved_method = _validate_resolved_method_metadata(
            method,
            arm,
            field="resolved method metadata",
        )

        info = require_mapping(method.get("info"), "model info")
        if (
            info.get("is_valid") is not True
            or info.get("can_infer") is not True
            or info.get("model_type") != "StackerEnsembleModel"
        ):
            raise RuntimeError("invalid outer model")
        _validate_outer_model_info(
            info,
            resolved_method,
            field="outer model info",
        )
        bag = require_mapping(info.get("bagged_info"), "bagged_info")
        child_names = [f"S1F{index}" for index in range(1, 9)]
        if (
            bag.get("num_child_models") != 8
            or bag.get("child_model_type") != "DarkoFitModel"
            or bag.get("child_model_names") != child_names
        ):
            raise RuntimeError("incomplete bag")
        _validate_resolved_bag_metadata(bag, arm, field="bagged_info")
        compressed = _validate_refit_params(
            bag.get("child_hyperparameters_fit"),
            "compressed refit parameters",
            max_iterations=horizon,
        )
        children = require_mapping(info.get("children_info"), "children_info")
        if set(children) != set(child_names):
            raise RuntimeError("incomplete child set")
        child_refit_iterations = []
        for child_fold, child_name in enumerate(child_names):
            child = require_mapping(children[child_name], child_name)
            if (
                child.get("name") != child_name
                or child.get("model_type") != "DarkoFitModel"
                or child.get("is_valid") is not True
                or child.get("can_infer") is not True
            ):
                raise RuntimeError(f"invalid child {child_name}")
            _validate_child_initial_hyperparameters(
                child,
                arm,
                child_fold,
                num_cpus=resolved_method["num_cpus_child"],
                num_gpus=resolved_method["num_gpus_child"],
                field=child_name,
            )
            fitted = require_mapping(child.get("darkofit_fit"), "darkofit_fit")
            if set(fitted) != REQUIRED_FIT_METADATA:
                raise RuntimeError(f"incomplete fit metadata for {child_name}")
            requested = _exact_int(fitted["iterations_requested"], "requested")
            attempted = _exact_int(fitted["iterations_attempted"], "attempted")
            completed = _exact_int(fitted["rounds_completed"], "completed")
            retained = _exact_int(fitted["rounds_retained"], "retained")
            best = _exact_int(fitted["best_iteration"], "best_iteration")
            if requested != horizon or not (
                0 <= retained == best <= completed <= attempted <= requested
            ):
                raise RuntimeError(f"invalid round counts for {child_name}")
            _validate_refit_params(
                child.get("hyperparameters_fit"),
                f"{child_name} refit parameters",
                expected_iterations=best,
                max_iterations=horizon,
            )
            if (
                float(fitted["resolved_learning_rate"]) != 0.1
                or fitted["requested_tree_mode"] != "catboost"
                or fitted["selected_tree_mode"] != "catboost"
                or fitted["selected_lane"] != "boosting"
                or fitted["linear_residual_active"] is not False
                or fitted["early_stopping_rounds"] != 50
            ):
                raise RuntimeError(f"wrong fitted policy for {child_name}")
            reason = fitted["stop_reason"]
            if reason not in {
                "iteration_limit",
                "early_stopping",
                "no_split",
                "time_limit",
            }:
                raise RuntimeError(f"wrong stop reason for {child_name}")
            validate_stop_reason_causality(
                reason,
                requested=requested,
                attempted=attempted,
                completed=completed,
                field=child_name,
            )
            if fitted["deadline_is_soft"] is not True or fitted[
                "deadline_hit"
            ] is not (reason == "time_limit"):
                raise RuntimeError(f"wrong deadline state for {child_name}")
            wall_limit = _positive_finite(
                fitted["wall_clock_limit_seconds"], "wall-clock limit"
            )
            wall_margin = _nonnegative_finite(
                fitted["wall_clock_safety_margin_seconds"],
                "wall-clock safety margin",
            )
            wall_effective = _nonnegative_finite(
                fitted["wall_clock_effective_seconds"],
                "effective wall-clock limit",
            )
            wall_elapsed = _nonnegative_finite(
                fitted["wall_clock_elapsed_seconds"], "wall-clock elapsed"
            )
            if (
                wall_limit > TIME_LIMIT_SECONDS
                or wall_margin < 0.0
                or wall_margin > wall_limit
                or wall_effective < 0.0
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
                raise RuntimeError(f"wrong deadline metadata for {child_name}")
            child_refit_iterations.append(best)
        validate_compressed_refit_iterations(
            compressed,
            child_refit_iterations,
            field="compressed refit parameters",
        )
    except (KeyError, RuntimeError, TypeError, ValueError, OverflowError):
        return "incomplete_or_mismatched"
    return None


def _require_regular_archive_source(path: Path, field: str) -> bool:
    """Return whether an entry exists, rejecting links and non-regular files."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeError(f"could not inspect {field}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{field} is not a regular file: {path}")
    return True


def prepare_paired_resume(
    output_dir: Path, jobs: Iterable[Any], *, resume: bool
) -> dict[str, Any] | None:
    """Archive any incomplete pair so resumed timing pairs rerun together."""
    if not resume:
        return None
    grouped: dict[tuple[str, int, int], dict[int, tuple[Path, Any]]] = defaultdict(dict)
    for job in jobs:
        dataset, repeat, fold = _job_coordinate(job)
        task_id = TASK_SPLIT_COUNTS[dataset][0]
        path = (
            output_dir
            / "experiments"
            / "data"
            / job.experiment.name
            / str(task_id)
            / f"{repeat}_{fold}"
            / "results.pkl"
        )
        grouped[(dataset, repeat, fold)][_job_horizon(job)] = (path, job)

    stale_attestation = output_dir / COMPLETION_ATTESTATION_FILENAME
    stale_analysis_payload = output_dir / ANALYSIS_PAYLOAD_FILENAME
    stale_analysis_outputs = [
        output_dir / filename for filename in DEFAULT_ANALYSIS_OUTPUT_FILENAMES
    ]
    # This is a pre-mutation gate: an invalid filesystem object must not leave
    # a partially archived resume cache behind.
    for pair in grouped.values():
        for path, _ in pair.values():
            _require_regular_archive_source(path, "cached result")
    for path in (
        stale_attestation,
        stale_analysis_payload,
        *stale_analysis_outputs,
        output_dir / MANIFEST_FILENAME,
        output_dir / WARMUP_HISTORY_FILENAME,
        output_dir / RESUME_HISTORY_FILENAME,
    ):
        _require_regular_archive_source(path, "campaign state artifact")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_root = output_dir / "resume_invalidated" / timestamp
    invalidated_pairs = []
    invalidated_result_count = 0
    for coordinate, pair in sorted(grouped.items()):
        statuses = {
            horizon: _cached_result_issue(path, job)
            for horizon, (path, job) in pair.items()
        }
        existing = [
            (horizon, path)
            for horizon, (path, _) in pair.items()
            if path.exists() or path.is_symlink()
        ]
        if not existing or all(issue is None for issue in statuses.values()):
            continue
        moved = []
        for horizon, source in existing:
            relative = source.relative_to(output_dir)
            destination = archive_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            moved.append(
                {
                    "arm": _arm_for_horizon(horizon),
                    "source": str(relative),
                    "archive": str(destination.relative_to(output_dir)),
                }
            )
        invalidated_result_count += len(moved)
        invalidated_pairs.append(
            {
                "coordinate": {
                    "dataset": coordinate[0],
                    "repeat": coordinate[1],
                    "fold": coordinate[2],
                },
                "arm_status": {
                    _arm_for_horizon(horizon): issue or "valid"
                    for horizon, issue in sorted(statuses.items())
                },
                "archived_results": moved,
            }
        )

    archived_attestation = None
    if stale_attestation.exists():
        destination = archive_root / COMPLETION_ATTESTATION_FILENAME
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stale_attestation, destination)
        archived_attestation = str(destination.relative_to(output_dir))

    archived_analysis_payload = None
    if stale_analysis_payload.exists():
        destination = archive_root / ANALYSIS_PAYLOAD_FILENAME
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stale_analysis_payload, destination)
        archived_analysis_payload = str(destination.relative_to(output_dir))

    archived_analysis_outputs = []
    for filename, source in zip(
        DEFAULT_ANALYSIS_OUTPUT_FILENAMES,
        stale_analysis_outputs,
        strict=True,
    ):
        if not source.exists():
            continue
        destination = archive_root / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        archived_analysis_outputs.append(
            {
                "source": filename,
                "archive": str(destination.relative_to(output_dir)),
            }
        )

    record = {
        "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "invalidated_pair_count": len(invalidated_pairs),
        "invalidated_result_count": invalidated_result_count,
        "invalidated_pairs": invalidated_pairs,
        "archived_completion_attestation": archived_attestation,
        "archived_analysis_payload": archived_analysis_payload,
        "archived_analysis_outputs": archived_analysis_outputs,
    }
    history_path = output_dir / RESUME_HISTORY_FILENAME
    history = []
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            raise RuntimeError("resume history must contain a JSON list")
    history.append(record)
    _atomic_write_json(history_path, history)
    return record


def collect_result_artifacts(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Hash the complete immutable result-pickle set before attestation."""
    files = sorted((output_dir / "experiments").rglob("results.pkl"))
    if len(files) != EXPECTED_JOBS:
        raise RuntimeError(
            f"expected {EXPECTED_JOBS} result files, found {len(files)}"
        )
    artifacts = {}
    for path in files:
        stat_before = path.lstat()
        if not stat.S_ISREG(stat_before.st_mode):
            raise RuntimeError(f"result artifact is not a regular file: {path}")
        digest = _sha256_file(path)
        stat_after = path.lstat()
        if (
            not stat.S_ISREG(stat_after.st_mode)
            or stat_before.st_dev != stat_after.st_dev
            or stat_before.st_ino != stat_after.st_ino
            or stat_before.st_size != stat_after.st_size
            or stat_before.st_mtime_ns != stat_after.st_mtime_ns
        ):
            raise RuntimeError(f"result changed while hashing: {path}")
        relative = str(path.relative_to(output_dir))
        artifacts[relative] = {
            "sha256": digest,
            "size_bytes": int(stat_after.st_size),
        }
    return artifacts


def _decode_result_pickle(payload: bytes, source: str) -> Mapping[str, Any]:
    """Decode either TabArena's default gzip cache or its raw-pickle mode."""
    try:
        pickle_payload = (
            gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
        )
        record = pickle.loads(pickle_payload)
    except Exception as exc:
        raise RuntimeError(f"could not decode result artifact {source}") from exc
    if not isinstance(record, Mapping):
        raise RuntimeError(f"{source}: result must be a mapping")
    return record


def _exact_int(value: Any, field: str) -> int:
    """Accept integral scalar values without silently truncating them."""
    if isinstance(value, bool):
        raise RuntimeError(f"{field} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{field} must be an integer") from exc
    if value != number:
        raise RuntimeError(f"{field} must be an integer")
    return number


def _positive_finite(value: Any, field: str) -> float:
    """Return a positive finite number while rejecting booleans explicitly."""
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
    """Return a nonnegative finite number while rejecting booleans explicitly."""
    if isinstance(value, bool):
        raise RuntimeError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or number < 0.0:
        raise RuntimeError(f"{field} must be nonnegative and finite")
    return number


def _validate_resolved_method_metadata(
    method: Mapping[str, Any], arm: str, *, field: str
) -> dict[str, Any]:
    """Validate AutoGluon's resolved budget, bag settings, and resources."""
    resolved = method.get("hyperparameters")
    if not isinstance(resolved, Mapping) or dict(
        resolved
    ) != expected_resolved_method_hyperparameters(arm):
        raise RuntimeError(f"{field} resolved hyperparameters do not match the arm")
    if (
        method.get("model_cls") != "DarkoFitModel"
        or method.get("model_type") != "DARKO"
        or method.get("name_prefix") != "DarkoFit"
        or method.get("init_kwargs_extra") != {}
    ):
        raise RuntimeError(f"{field} resolved model identity does not match")

    num_cpus = _exact_int(method.get("num_cpus"), f"{field}.num_cpus")
    num_gpus = _exact_int(method.get("num_gpus"), f"{field}.num_gpus")
    num_cpus_child = _exact_int(
        method.get("num_cpus_child"), f"{field}.num_cpus_child"
    )
    num_gpus_child = _exact_int(
        method.get("num_gpus_child"), f"{field}.num_gpus_child"
    )
    fit_kwargs = method.get("fit_kwargs_extra")
    if not isinstance(fit_kwargs, Mapping) or dict(
        fit_kwargs
    ) != expected_fit_kwargs_extra(num_cpus):
        raise RuntimeError(f"{field} resolved bag fit settings do not match")
    if num_cpus < 1 or num_cpus_child != num_cpus:
        raise RuntimeError(f"{field} CPU allocation does not match sequential fitting")
    if num_gpus != 0 or num_gpus_child != 0:
        raise RuntimeError(f"{field} GPU allocation does not match the CPU campaign")
    fit_metadata = method.get("fit_metadata")
    if not isinstance(fit_metadata, Mapping):
        raise RuntimeError(f"{field}.fit_metadata must be a mapping")
    if (
        _exact_int(fit_metadata.get("num_cpus"), f"{field}.fit_metadata.num_cpus")
        != num_cpus
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
        "num_cpus": num_cpus,
        "num_gpus": num_gpus,
        "num_cpus_child": num_cpus_child,
        "num_gpus_child": num_gpus_child,
    }


def _validate_resolved_bag_metadata(
    bag: Mapping[str, Any], arm: str, *, field: str
) -> None:
    """Validate the resolved one-set/eight-fold bag and its base child policy."""
    if (
        _exact_int(bag.get("_n_repeats"), f"{field}._n_repeats") != 1
        or bag.get("_k_per_n_repeat") != [8]
        or _exact_int(bag.get("_random_state"), f"{field}._random_state") != 1
        or bag.get("bagged_mode") is not True
    ):
        raise RuntimeError(f"{field} is not the frozen one-set/eight-fold bag")
    child_user = bag.get("child_hyperparameters_user")
    if not isinstance(child_user, Mapping) or dict(child_user) != HORIZON_ARMS[arm]:
        raise RuntimeError(f"{field} child user hyperparameters do not match the arm")
    child_resolved = bag.get("child_hyperparameters")
    if not isinstance(child_resolved, Mapping) or dict(
        child_resolved
    ) != expected_child_hyperparameters(arm, 0):
        raise RuntimeError(f"{field} resolved base-child policy does not match")
    child_ag_args = bag.get("child_ag_args_fit")
    if not isinstance(child_ag_args, Mapping) or any(
        child_ag_args.get(name) != expected
        for name, expected in {
            "max_memory_usage_ratio": 1.0,
            "max_time_limit_ratio": 1.0,
            "max_time_limit": None,
            "min_time_limit": 0,
        }.items()
    ):
        raise RuntimeError(f"{field} child resource/budget ratios do not match")


def _validate_outer_model_info(
    info: Mapping[str, Any], resolved_method: Mapping[str, Any], *, field: str
) -> None:
    if (
        _exact_int(info.get("num_cpus"), f"{field}.num_cpus")
        != resolved_method["num_cpus"]
        or _exact_int(info.get("num_gpus"), f"{field}.num_gpus")
        != resolved_method["num_gpus"]
        or info.get("problem_type") != "regression"
        or info.get("eval_metric") != "root_mean_squared_error"
        or info.get("stopping_metric") != "root_mean_squared_error"
        or info.get("val_in_fit") is not False
        or info.get("unlabeled_in_fit") is not False
    ):
        raise RuntimeError(f"{field} resources/evaluation policy do not match")


def _validate_child_initial_hyperparameters(
    child: Mapping[str, Any],
    arm: str,
    child_fold: int,
    *,
    num_cpus: int,
    num_gpus: int,
    field: str,
) -> dict[str, Any]:
    initial = child.get("hyperparameters")
    if not isinstance(initial, Mapping) or dict(
        initial
    ) != expected_child_hyperparameters(arm, child_fold):
        raise RuntimeError(f"{field} initialized hyperparameters or seed do not match")
    user = child.get("hyperparameters_user")
    if not isinstance(user, Mapping) or dict(user) != HORIZON_ARMS[arm]:
        raise RuntimeError(f"{field} user hyperparameters do not match the arm")
    if (
        _exact_int(child.get("num_cpus"), f"{field}.num_cpus") != num_cpus
        or _exact_int(child.get("num_gpus"), f"{field}.num_gpus") != num_gpus
        or child.get("problem_type") != "regression"
        or child.get("eval_metric") != "root_mean_squared_error"
        or child.get("stopping_metric") != "root_mean_squared_error"
        or child.get("val_in_fit") is not True
        or child.get("unlabeled_in_fit") is not False
    ):
        raise RuntimeError(f"{field} resolved resources/evaluation policy do not match")
    child_ag_args = child.get("ag_args_fit")
    if not isinstance(child_ag_args, Mapping) or any(
        child_ag_args.get(name) != expected
        for name, expected in {
            "max_memory_usage_ratio": 1.0,
            "max_time_limit_ratio": 1.0,
            "max_time_limit": None,
            "min_time_limit": 0,
        }.items()
    ):
        raise RuntimeError(f"{field} resource/budget ratios do not match")
    return dict(initial)


def validate_stop_reason_causality(
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


def autogluon_compressed_refit_iterations(
    child_iterations: Iterable[Any], *, field: str
) -> int:
    """Mirror AutoGluon's integer compression: round(mean(child values))."""
    values = [
        _exact_int(value, f"{field}[{index}]")
        for index, value in enumerate(child_iterations)
    ]
    if len(values) != 8:
        raise RuntimeError(f"{field} must contain exactly eight child values")
    return round(mean(values))


def validate_compressed_refit_iterations(
    compressed: Mapping[str, Any],
    child_iterations: Iterable[Any],
    *,
    field: str,
) -> int:
    """Bind AutoGluon's bag-level compressed iterations to all eight children."""
    observed = _exact_int(compressed.get("iterations"), f"{field}.iterations")
    expected = autogluon_compressed_refit_iterations(
        child_iterations, field=f"{field}.child_iterations"
    )
    if observed != expected:
        raise RuntimeError(
            f"{field}.iterations does not match AutoGluon's child aggregation"
        )
    return observed


def _validate_refit_params(
    value: Any,
    field: str,
    *,
    max_iterations: int,
    expected_iterations: int | None = None,
) -> Mapping[str, Any]:
    """Validate every refit field against the frozen CatBoost policy."""
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field} must be a mapping")
    if set(value) != REQUIRED_REFIT_PARAMS:
        raise RuntimeError(f"{field} are incomplete")

    iterations = _exact_int(value["iterations"], f"{field}.iterations")
    if not 0 <= iterations <= max_iterations:
        raise RuntimeError(f"{field}.iterations is outside the frozen horizon")
    if expected_iterations is not None and iterations != expected_iterations:
        raise RuntimeError(f"{field}.iterations does not match the best prefix")
    if _exact_int(value["depth"], f"{field}.depth") != 6:
        raise RuntimeError(f"{field}.depth does not match the frozen policy")
    if value["num_leaves"] is not None:
        raise RuntimeError(f"{field}.num_leaves does not match the frozen policy")
    if _exact_int(
        value["min_child_samples"], f"{field}.min_child_samples"
    ) != 20:
        raise RuntimeError(
            f"{field}.min_child_samples does not match the frozen policy"
        )

    numeric_expected = {
        "learning_rate": 0.1,
        "l2_leaf_reg": 3.0,
        "min_child_weight": 1.0,
        "cat_smoothing": 1.0,
    }
    for name, expected in numeric_expected.items():
        raw = value[name]
        if isinstance(raw, bool):
            raise RuntimeError(f"{field}.{name} must be numeric")
        try:
            number = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"{field}.{name} must be numeric") from exc
        if not math.isfinite(number) or number != expected:
            raise RuntimeError(f"{field}.{name} does not match the frozen policy")

    expected_exact = {
        "tree_mode": "catboost",
        "early_stopping": False,
        "early_stopping_rounds": None,
        "use_best_model": False,
        "refit": False,
    }
    for name, expected in expected_exact.items():
        if value[name] is not expected and value[name] != expected:
            raise RuntimeError(f"{field}.{name} does not match the frozen policy")
        if isinstance(expected, bool) and value[name] is not expected:
            raise RuntimeError(f"{field}.{name} must be a boolean")
    return value


def validate_completed_result_artifacts(
    output_dir: Path, artifacts: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate result pickles and return a non-executable analysis snapshot."""
    expected = set(expected_coordinates())
    seen: set[tuple[str, int, int, str]] = set()
    child_count = 0
    stop_reasons: dict[str, int] = defaultdict(int)
    resource_allocations: set[tuple[int, int, int, int]] = set()
    outer_rows: list[dict[str, Any]] = []
    child_rows: list[dict[str, Any]] = []
    def as_mapping(value, field):
        if not isinstance(value, Mapping):
            raise RuntimeError(f"{field} must be a mapping")
        return value

    def finite_number(value, field, *, positive=False):
        validator = _positive_finite if positive else _nonnegative_finite
        return validator(value, field)

    for relative, expected_artifact in sorted(artifacts.items()):
        path = output_dir / relative
        payload = path.read_bytes()
        if len(payload) != int(expected_artifact["size_bytes"]):
            raise RuntimeError(f"attested result size changed: {relative}")
        if hashlib.sha256(payload).hexdigest() != expected_artifact["sha256"]:
            raise RuntimeError(f"attested result digest changed: {relative}")
        record = _decode_result_pickle(payload, relative)
        if record.get("problem_type") != "regression" or record.get("metric") != "rmse":
            raise RuntimeError(f"{relative}: result is not regression/RMSE")
        imputed = record.get("imputed", False)
        if imputed is not False and imputed is not None:
            raise RuntimeError(f"{relative}: result is imputed")
        experiment = as_mapping(
            record.get("experiment_metadata"),
            f"{relative}: experiment_metadata",
        )
        if (
            experiment.get("experiment_cls") != "OOFExperimentRunner"
            or experiment.get("method_cls") != "AGSingleBagWrapper"
        ):
            raise RuntimeError(f"{relative}: unexpected experiment implementation")
        metric_values = {
            field: finite_number(
                record.get(field), f"{relative}: {field}", positive=True
            )
            for field in (
                "metric_error",
                "metric_error_val",
                "time_train_s",
                "time_infer_s",
            )
        }
        memory = as_mapping(record.get("memory_usage"), f"{relative}: memory_usage")
        peak_memory = finite_number(
            memory.get("peak_mem_cpu"),
            f"{relative}: peak_mem_cpu",
            positive=True,
        )

        task = as_mapping(record.get("task_metadata"), f"{relative}: task_metadata")
        dataset = task.get("name")
        if dataset not in TASK_SPLIT_COUNTS:
            raise RuntimeError(f"{relative}: unexpected dataset {dataset!r}")
        task_id, split_count = TASK_SPLIT_COUNTS[dataset]
        repeat = _exact_int(task.get("repeat"), f"{relative}: repeat")
        fold = _exact_int(task.get("fold"), f"{relative}: fold")
        coordinate = (str(dataset), repeat, fold)
        if (
            _exact_int(task.get("tid"), f"{relative}: task id") != task_id
            or coordinate not in expected
        ):
            raise RuntimeError(f"{relative}: unexpected task coordinate {coordinate}")
        registered_fold = 3 * repeat + fold
        if registered_fold >= split_count:
            raise RuntimeError(f"{relative}: out-of-range task coordinate {coordinate}")
        if task.get("split_idx") is not None and _exact_int(
            task["split_idx"], f"{relative}: split_idx"
        ) != registered_fold:
            raise RuntimeError(f"{relative}: split_idx does not match repeat/fold")

        method = as_mapping(
            record.get("method_metadata"), f"{relative}: method_metadata"
        )
        raw = dict(
            as_mapping(
                method.get("model_hyperparameters"),
                f"{relative}: model_hyperparameters",
            )
        )
        ag_args = raw.pop("ag_args", None)
        ag_ensemble = raw.pop("ag_args_ensemble", None)
        matching_arms = [name for name, config in HORIZON_ARMS.items() if raw == config]
        if len(matching_arms) != 1:
            raise RuntimeError(f"{relative}: result does not match one frozen arm")
        arm = matching_arms[0]
        expected_suffix = f"_c1_{arm}_horizon"
        if ag_args != {"name_suffix": expected_suffix}:
            raise RuntimeError(f"{relative}: unexpected AutoGluon name suffix")
        if (
            not isinstance(ag_ensemble, Mapping)
            or dict(ag_ensemble) != expected_ag_ensemble_config()
        ):
            raise RuntimeError(
                f"{relative}: unexpected bag seed/resource configuration"
            )
        if record.get("framework") != f"DarkoFit{expected_suffix}_BAG_L1":
            raise RuntimeError(f"{relative}: unexpected framework name")
        resolved_method = _validate_resolved_method_metadata(
            method,
            arm,
            field=f"{relative}: resolved method metadata",
        )
        resource_allocations.add(
            (
                resolved_method["num_cpus"],
                resolved_method["num_gpus"],
                resolved_method["num_cpus_child"],
                resolved_method["num_gpus_child"],
            )
        )
        info = as_mapping(method.get("info"), f"{relative}: model info")
        if info.get("is_valid") is not True or info.get("can_infer") is not True:
            raise RuntimeError(f"{relative}: fitted bag is not valid/inferable")
        if info.get("model_type") != "StackerEnsembleModel":
            raise RuntimeError(f"{relative}: unexpected outer model type")
        _validate_outer_model_info(
            info,
            resolved_method,
            field=f"{relative}: outer model info",
        )
        key = (*coordinate, arm)
        if key in seen:
            raise RuntimeError(f"duplicate completed result for {key}")
        seen.add(key)
        outer_rows.append(
            {
                "dataset": str(dataset),
                "task_id": int(task_id),
                "repeat": repeat,
                "fold": fold,
                "registered_fold": registered_fold,
                "arm": arm,
                "test_rmse": metric_values["metric_error"],
                "val_rmse": metric_values["metric_error_val"],
                "train_time_s": metric_values["time_train_s"],
                "infer_time_s": metric_values["time_infer_s"],
                "peak_memory_bytes": peak_memory,
                "framework": str(record.get("framework")),
                "imputed": False,
                "experiment_cls": str(experiment.get("experiment_cls")),
                "method_cls": str(experiment.get("method_cls")),
                "outer_model_type": str(info.get("model_type")),
                "ag_ensemble": dict(ag_ensemble),
                **resolved_method,
                "source": relative,
            }
        )
        bag = as_mapping(info.get("bagged_info"), f"{relative}: bagged_info")
        if (
            bag.get("num_child_models") != 8
            or bag.get("child_model_type") != "DarkoFitModel"
            or bag.get("child_model_names")
            != [f"S1F{index}" for index in range(1, 9)]
        ):
            raise RuntimeError(f"{relative}: expected eight bag children")
        _validate_resolved_bag_metadata(
            bag,
            arm,
            field=f"{relative}: bagged_info",
        )
        outer_rows[-1]["bag_folds"] = 8
        outer_rows[-1]["bag_sets"] = 1
        outer_rows[-1]["bag_random_state"] = 1
        outer_rows[-1]["child_ag_args_fit"] = {
            name: bag["child_ag_args_fit"][name]
            for name in (
                "max_memory_usage_ratio",
                "max_time_limit_ratio",
                "max_time_limit",
                "min_time_limit",
            )
        }
        compressed = as_mapping(
            bag.get("child_hyperparameters_fit"),
            f"{relative}: child_hyperparameters_fit",
        )
        _validate_refit_params(
            compressed,
            f"{relative}: compressed refit parameters",
            max_iterations=HORIZON_ARMS[arm]["iterations"],
        )
        outer_rows[-1]["compressed_refit_params"] = {
            name: compressed[name] for name in sorted(REQUIRED_REFIT_PARAMS)
        }
        children = as_mapping(info.get("children_info"), f"{relative}: children_info")
        if set(children) != {f"S1F{index}" for index in range(1, 9)}:
            raise RuntimeError(f"{relative}: unexpected child set")
        child_refit_iterations = []
        for child_name, child_value in children.items():
            child = as_mapping(child_value, f"{relative}: {child_name}")
            if (
                child.get("name") != child_name
                or child.get("model_type") != "DarkoFitModel"
            ):
                raise RuntimeError(f"{relative}: {child_name} has wrong model type")
            if child.get("is_valid") is not True or child.get("can_infer") is not True:
                raise RuntimeError(f"{relative}: {child_name} is not valid/inferable")
            child_fold = int(str(child_name).removeprefix("S1F")) - 1
            initial_hyperparameters = _validate_child_initial_hyperparameters(
                child,
                arm,
                child_fold,
                num_cpus=resolved_method["num_cpus_child"],
                num_gpus=resolved_method["num_gpus_child"],
                field=f"{relative}: {child_name}",
            )
            trained = as_mapping(
                child.get("hyperparameters_fit"),
                f"{relative}: {child_name}.hyperparameters_fit",
            )
            fitted = as_mapping(
                child.get("darkofit_fit"),
                f"{relative}: {child_name}.darkofit_fit",
            )
            if set(fitted) != REQUIRED_FIT_METADATA:
                raise RuntimeError(f"{relative}: {child_name} fit metadata incomplete")
            requested = _exact_int(
                fitted["iterations_requested"],
                f"{relative}: {child_name} iterations_requested",
            )
            attempted = _exact_int(
                fitted["iterations_attempted"],
                f"{relative}: {child_name} iterations_attempted",
            )
            completed = _exact_int(
                fitted["rounds_completed"],
                f"{relative}: {child_name} rounds_completed",
            )
            retained = _exact_int(
                fitted["rounds_retained"],
                f"{relative}: {child_name} rounds_retained",
            )
            best = _exact_int(
                fitted["best_iteration"],
                f"{relative}: {child_name} best_iteration",
            )
            if requested != HORIZON_ARMS[arm]["iterations"]:
                raise RuntimeError(f"{relative}: {child_name} requested wrong horizon")
            if not (0 <= retained == best <= completed <= attempted <= requested):
                raise RuntimeError(f"{relative}: {child_name} round counts are inconsistent")
            _validate_refit_params(
                trained,
                f"{relative}: {child_name} refit parameters",
                expected_iterations=best,
                max_iterations=requested,
            )
            if (
                finite_number(fitted["resolved_learning_rate"], "resolved LR") != 0.1
                or finite_number(trained["learning_rate"], "refit LR") != 0.1
                or fitted["requested_tree_mode"] != "catboost"
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
                raise RuntimeError(f"{relative}: {child_name} resolved policy mismatch")
            reason = fitted["stop_reason"]
            validate_stop_reason_causality(
                reason,
                requested=requested,
                attempted=attempted,
                completed=completed,
                field=f"{relative}: {child_name}",
            )
            if fitted["deadline_is_soft"] is not True:
                raise RuntimeError(f"{relative}: {child_name} deadline was not active")
            if fitted["deadline_hit"] is not (reason == "time_limit"):
                raise RuntimeError(f"{relative}: {child_name} deadline flag mismatch")
            wall_limit = finite_number(
                fitted["wall_clock_limit_seconds"], "wall limit", positive=True
            )
            wall_margin = finite_number(
                fitted["wall_clock_safety_margin_seconds"], "wall margin"
            )
            wall_effective = finite_number(
                fitted["wall_clock_effective_seconds"], "effective wall limit"
            )
            wall_elapsed = finite_number(
                fitted["wall_clock_elapsed_seconds"], "wall elapsed"
            )
            if (
                wall_limit <= 0.0
                or wall_limit > TIME_LIMIT_SECONDS
                or wall_margin < 0.0
                or wall_margin > wall_limit
                or wall_effective < 0.0
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
                raise RuntimeError(
                    f"{relative}: {child_name} deadline metadata is inconsistent"
                )
            child_rows.append(
                {
                    "dataset": str(dataset),
                    "task_id": int(task_id),
                    "repeat": repeat,
                    "fold": fold,
                    "registered_fold": registered_fold,
                    "child": str(child_name),
                    "child_fold": child_fold,
                    "arm": arm,
                    "iterations_requested": requested,
                    "iterations_attempted": attempted,
                    "rounds_completed": completed,
                    "rounds_retained": retained,
                    "best_iteration": best,
                    "resolved_learning_rate": float(
                        fitted["resolved_learning_rate"]
                    ),
                    "requested_tree_mode": str(fitted["requested_tree_mode"]),
                    "selected_tree_mode": str(fitted["selected_tree_mode"]),
                    "selected_lane": str(fitted["selected_lane"]),
                    "linear_residual_active": bool(
                        fitted["linear_residual_active"]
                    ),
                    "early_stopping_rounds": int(
                        fitted["early_stopping_rounds"]
                    ),
                    "stop_reason": str(reason),
                    "wall_clock_limit_seconds": wall_limit,
                    "wall_clock_safety_margin_seconds": wall_margin,
                    "wall_clock_effective_seconds": wall_effective,
                    "wall_clock_elapsed_seconds": wall_elapsed,
                    "deadline_hit": bool(fitted["deadline_hit"]),
                    "deadline_is_soft": bool(fitted["deadline_is_soft"]),
                    "refit_params": {
                        name: trained[name] for name in sorted(REQUIRED_REFIT_PARAMS)
                    },
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
                    "source": relative,
                }
            )
            child_count += 1
            stop_reasons[str(reason)] += 1
            child_refit_iterations.append(best)
        validate_compressed_refit_iterations(
            compressed,
            child_refit_iterations,
            field=f"{relative}: compressed refit parameters",
        )

    expected_keys = {
        (dataset, repeat, fold, arm)
        for dataset, repeat, fold in expected
        for arm in HORIZON_ARMS
    }
    if seen != expected_keys:
        raise RuntimeError("completed result grid is missing or unexpected")
    if child_count != EXPECTED_CHILD_FITS:
        raise RuntimeError(
            f"expected {EXPECTED_CHILD_FITS} child metadata records, got {child_count}"
        )
    if len(resource_allocations) != 1:
        raise RuntimeError(
            "completed campaign did not use one identical CPU/GPU allocation"
        )
    outer_rows.sort(
        key=lambda row: (
            row["task_id"], row["repeat"], row["fold"], row["arm"]
        )
    )
    child_rows.sort(
        key=lambda row: (
            row["task_id"],
            row["repeat"],
            row["fold"],
            row["child_fold"],
            row["arm"],
        )
    )
    resolved_resources = next(iter(resource_allocations))
    validation = {
        "result_count": len(seen),
        "child_fit_count": child_count,
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
        "resource_allocation": {
            "num_cpus": resolved_resources[0],
            "num_gpus": resolved_resources[1],
            "num_cpus_child": resolved_resources[2],
            "num_gpus_child": resolved_resources[3],
        },
    }
    return validation, outer_rows, child_rows


def _validate_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"{field} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RuntimeError(f"{field} must be a SHA-256 digest") from exc
    return value


def _validate_warmup_history(value: Any, *, expected_thread_count: int) -> None:
    """Validate the audit record proving process-local warmup preceded fitting."""
    expected_thread_count = _exact_int(
        expected_thread_count, "expected warmup thread count"
    )
    if expected_thread_count < 1:
        raise RuntimeError("expected warmup thread count must be positive")
    if not isinstance(value, list) or not value:
        raise RuntimeError("warmup history must contain at least one record")
    expected_stage_names = ("numeric", "categorical")
    for record_index, raw_record in enumerate(value):
        if not isinstance(raw_record, Mapping) or set(raw_record) != {
            "completed_at_utc",
            "pid",
            "warmup",
        }:
            raise RuntimeError(f"warmup history record {record_index} is incomplete")
        if not isinstance(raw_record["completed_at_utc"], str) or not raw_record[
            "completed_at_utc"
        ]:
            raise RuntimeError("warmup completion timestamp is missing")
        if _exact_int(raw_record["pid"], "warmup pid") <= 0:
            raise RuntimeError("warmup pid must be positive")
        warmup = raw_record["warmup"]
        if not isinstance(warmup, Mapping) or set(warmup) != {
            "schema_version",
            "clock",
            "duration_seconds",
            "config",
            "stages",
        }:
            raise RuntimeError("warmup metadata fields are incomplete")
        if warmup["schema_version"] != 2 or warmup["clock"] != "time.monotonic_ns":
            raise RuntimeError("warmup schema or clock does not match")
        _nonnegative_finite(warmup["duration_seconds"], "warmup duration")
        config = warmup["config"]
        if not isinstance(config, Mapping):
            raise RuntimeError("warmup config must be a mapping")
        expected_config = {
            "iterations": 5,
            "loss": "RMSE",
            "tree_mode": "catboost",
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "max_bins": 128,
            "learning_rate": 0.1,
            "ts_permutations": 1,
            "ordered_boosting": "auto",
            "sampling": "uniform",
            "linear_residual": False,
            "early_stopping": False,
            "use_best_model": False,
            "diagnostic_warnings": "never",
            "random_state": 20_260_713,
        }
        thread_count = _exact_int(config.get("thread_count"), "warmup thread_count")
        if thread_count != expected_thread_count:
            raise RuntimeError(
                "warmup thread count does not match resolved child resources"
            )
        if dict(config) != {**expected_config, "thread_count": thread_count}:
            raise RuntimeError("warmup config does not match the frozen policy")
        stages = warmup["stages"]
        if not isinstance(stages, list) or [
            stage.get("name") if isinstance(stage, Mapping) else None
            for stage in stages
        ] != list(expected_stage_names):
            raise RuntimeError("warmup must contain numeric and categorical stages")
        for stage_index, stage in enumerate(stages):
            expected_fields = {
                "name",
                "categorical_features",
                "train_rows",
                "validation_rows",
                "fit_seconds",
                "iterations_fitted",
                "tree_depths",
                "resolved_learning_rate",
                "resolved_tree_mode",
                "resolved_ordered_boosting",
                "resolved_thread_count",
                "flat_ensemble_type",
                "flat_prediction_router_selected",
                "prediction_parallel_min_rows",
                "prediction_batches",
            }
            if set(stage) != expected_fields:
                raise RuntimeError("warmup stage fields are incomplete")
            if stage["categorical_features"] != ([] if stage_index == 0 else [12]):
                raise RuntimeError("warmup categorical lane does not match")
            if (
                _exact_int(stage["train_rows"], "warmup train rows") != 2048
                or _exact_int(stage["validation_rows"], "warmup validation rows")
                != 512
                or _exact_int(stage["iterations_fitted"], "warmup iterations") != 5
                or stage["tree_depths"] != [6] * 5
                or float(stage["resolved_learning_rate"]) != 0.1
                or stage["resolved_tree_mode"] != "catboost"
                or stage["resolved_ordered_boosting"] is not False
            ):
                raise RuntimeError("warmup stage did not resolve the frozen policy")
            _nonnegative_finite(stage["fit_seconds"], "warmup fit duration")
            resolved_threads = _exact_int(
                stage["resolved_thread_count"], "warmup resolved threads"
            )
            if resolved_threads != expected_thread_count:
                raise RuntimeError(
                    "warmup resolved thread count does not match child resources"
                )
            threshold = _exact_int(
                stage["prediction_parallel_min_rows"], "warmup prediction threshold"
            )
            if threshold != 8192:
                raise RuntimeError("warmup prediction threshold does not match")
            batches = stage["prediction_batches"]
            if not isinstance(batches, list) or len(batches) != 2:
                raise RuntimeError("warmup prediction batches are incomplete")
            for batch_index, batch in enumerate(batches):
                if not isinstance(batch, Mapping) or set(batch) != {
                    "name",
                    "route",
                    "input_shape",
                    "prediction_shape",
                    "predict_seconds",
                    "prediction_sha256",
                }:
                    raise RuntimeError("warmup prediction batch fields are incomplete")
                expected_rows = threshold - 1 if batch_index == 0 else threshold
                expected_name = (
                    "serial_subthreshold"
                    if batch_index == 0
                    else "parallel_at_threshold"
                )
                expected_route = (
                    "tree_loop"
                    if resolved_threads == 1
                    else ("flat_serial" if batch_index == 0 else "flat_parallel")
                )
                if (
                    batch["name"] != expected_name
                    or batch["route"] != expected_route
                    or batch["input_shape"] != [expected_rows, 12 + stage_index]
                    or batch["prediction_shape"] != [expected_rows]
                ):
                    raise RuntimeError("warmup prediction route or shape does not match")
                _nonnegative_finite(
                    batch["predict_seconds"], "warmup prediction duration"
                )
                _validate_sha256(
                    batch["prediction_sha256"], "warmup prediction fingerprint"
                )


def _validate_resume_history(value: Any, output_dir: Path) -> None:
    if not isinstance(value, list) or not value:
        raise RuntimeError("resume history must contain at least one record")
    for index, record in enumerate(value):
        if not isinstance(record, Mapping) or set(record) != {
            "resumed_at_utc",
            "pid",
            "invalidated_pair_count",
            "invalidated_result_count",
            "invalidated_pairs",
            "archived_completion_attestation",
            "archived_analysis_payload",
            "archived_analysis_outputs",
        }:
            raise RuntimeError(f"resume history record {index} is incomplete")
        if not isinstance(record["resumed_at_utc"], str) or not record[
            "resumed_at_utc"
        ]:
            raise RuntimeError("resume timestamp is missing")
        if _exact_int(record["pid"], "resume pid") <= 0:
            raise RuntimeError("resume pid must be positive")
        pairs = record["invalidated_pairs"]
        archived_outputs = record["archived_analysis_outputs"]
        if not isinstance(pairs, list) or not isinstance(archived_outputs, list):
            raise RuntimeError("resume history lists are malformed")
        valid_statuses = {
            "valid",
            "missing",
            "not_a_file",
            "unreadable",
            "incomplete_or_mismatched",
        }
        for pair in pairs:
            if not isinstance(pair, Mapping) or set(pair) != {
                "coordinate",
                "arm_status",
                "archived_results",
            }:
                raise RuntimeError("resume invalidated-pair record is incomplete")
            coordinate = pair["coordinate"]
            if not isinstance(coordinate, Mapping) or set(coordinate) != {
                "dataset",
                "repeat",
                "fold",
            }:
                raise RuntimeError("resume invalidated coordinate is incomplete")
            dataset = coordinate["dataset"]
            repeat = _exact_int(coordinate["repeat"], "resume repeat")
            fold = _exact_int(coordinate["fold"], "resume fold")
            if (dataset, repeat, fold) not in set(expected_coordinates()):
                raise RuntimeError("resume invalidated coordinate is not frozen")
            arm_status = pair["arm_status"]
            if (
                not isinstance(arm_status, Mapping)
                or set(arm_status) != set(HORIZON_ARMS)
                or any(status not in valid_statuses for status in arm_status.values())
            ):
                raise RuntimeError("resume arm status is malformed")
            archived_results = pair["archived_results"]
            if not isinstance(archived_results, list):
                raise RuntimeError("resume archived result list is malformed")
            for archived in archived_results:
                if not isinstance(archived, Mapping) or set(archived) != {
                    "arm",
                    "source",
                    "archive",
                }:
                    raise RuntimeError("resume archived result record is incomplete")
                if archived["arm"] not in HORIZON_ARMS:
                    raise RuntimeError("resume archived result arm is invalid")
        for archived in archived_outputs:
            if not isinstance(archived, Mapping) or set(archived) != {
                "source",
                "archive",
            }:
                raise RuntimeError("resume archived analysis output is incomplete")
            if archived["source"] not in DEFAULT_ANALYSIS_OUTPUT_FILENAMES:
                raise RuntimeError("resume archived analysis output name is invalid")
        for field in (
            "archived_completion_attestation",
            "archived_analysis_payload",
        ):
            if record[field] is not None and not isinstance(record[field], str):
                raise RuntimeError(f"resume {field} must be a string or null")
        if _exact_int(record["invalidated_pair_count"], "invalidated pair count") != len(
            pairs
        ):
            raise RuntimeError("resume invalidated-pair count does not match")
        moved_results = sum(
            len(pair.get("archived_results", []))
            for pair in pairs
            if isinstance(pair, Mapping)
        )
        if _exact_int(
            record["invalidated_result_count"], "invalidated result count"
        ) != moved_results:
            raise RuntimeError("resume invalidated-result count does not match")
        archived_paths = [
            record["archived_completion_attestation"],
            record["archived_analysis_payload"],
            *(
                item.get("archive") if isinstance(item, Mapping) else None
                for item in archived_outputs
            ),
            *(
                moved.get("archive")
                for pair in pairs
                if isinstance(pair, Mapping)
                for moved in pair.get("archived_results", [])
                if isinstance(moved, Mapping)
            ),
        ]
        for relative in (path for path in archived_paths if path is not None):
            if not isinstance(relative, str):
                raise RuntimeError("resume archive path must be a string")
            relative_path = Path(relative)
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or not relative_path.parts
                or relative_path.parts[0] != "resume_invalidated"
            ):
                raise RuntimeError("resume archive path is not a safe relative path")
            archive = output_dir / relative_path
            try:
                archive.parent.resolve(strict=True).relative_to(
                    output_dir.resolve(strict=True)
                )
            except (OSError, ValueError) as exc:
                raise RuntimeError("resume archive path escapes the output directory") from exc
            try:
                archive_metadata = archive.lstat()
            except FileNotFoundError as exc:
                raise RuntimeError(f"resume archive is missing: {relative}") from exc
            if not stat.S_ISREG(archive_metadata.st_mode):
                raise RuntimeError(
                    f"resume archive is not a regular file: {relative}"
                )


def _history_artifact(
    output_dir: Path,
    filename: str,
    *,
    required: bool,
    validator,
) -> dict[str, Any] | None:
    path = output_dir / filename
    try:
        stat_before = path.lstat()
    except FileNotFoundError:
        if required:
            raise RuntimeError(f"required campaign history is missing: {filename}")
        return None
    if not stat.S_ISREG(stat_before.st_mode):
        raise RuntimeError(f"campaign history is not a regular file: {filename}")
    payload = path.read_bytes()
    stat_after = path.lstat()
    if (
        not stat.S_ISREG(stat_after.st_mode)
        or stat_before.st_dev != stat_after.st_dev
        or stat_before.st_ino != stat_after.st_ino
        or stat_before.st_size != stat_after.st_size
        or stat_before.st_mtime_ns != stat_after.st_mtime_ns
    ):
        raise RuntimeError(f"campaign history changed while reading: {filename}")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"campaign history is not valid JSON: {filename}") from exc
    validator(value)
    return {
        "path": filename,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def write_completion_attestation(
    output_dir: Path, *, manifest: dict[str, Any], result_count: int
) -> dict[str, Any]:
    """Seal a complete run so analysis cannot accept a partial cache."""
    if result_count != EXPECTED_JOBS:
        raise RuntimeError(
            f"expected {EXPECTED_JOBS} completed results, got {result_count}"
        )
    manifest_path = output_dir / MANIFEST_FILENAME
    artifacts = collect_result_artifacts(output_dir)
    validation, outer_rows, child_rows = validate_completed_result_artifacts(
        output_dir, artifacts
    )
    resource_allocation = validation.get("resource_allocation")
    if not isinstance(resource_allocation, Mapping):
        raise RuntimeError("completed result resource allocation is missing")
    resolved_child_num_cpus = _exact_int(
        resource_allocation.get("num_cpus_child"),
        "completed child num_cpus",
    )
    if resolved_child_num_cpus != _exact_int(
        manifest.get("resolved_child_num_cpus"),
        "manifest resolved child num_cpus",
    ):
        raise RuntimeError(
            "completed child CPU allocation does not match the run manifest"
        )
    analysis_payload = {
        "schema_version": 1,
        "kind": "darkofit_tabarena_regression_cap_horizon_analysis_payload",
        "protocol_sha256": manifest["protocol_sha256"],
        "result_artifacts_sha256": hashlib.sha256(
            _canonical_json(artifacts)
        ).hexdigest(),
        "outer_rows": outer_rows,
        "child_rows": child_rows,
    }
    analysis_path = output_dir / ANALYSIS_PAYLOAD_FILENAME
    _atomic_write_json(analysis_path, analysis_payload)
    analysis_before = analysis_path.stat()
    analysis_digest = _sha256_file(analysis_path)
    analysis_after = analysis_path.stat()
    if (
        analysis_before.st_size != analysis_after.st_size
        or analysis_before.st_mtime_ns != analysis_after.st_mtime_ns
    ):
        raise RuntimeError("analysis payload changed while being hashed")
    analysis_artifact = {
        "path": ANALYSIS_PAYLOAD_FILENAME,
        "sha256": analysis_digest,
        "size_bytes": int(analysis_after.st_size),
    }
    warmup_history_artifact = _history_artifact(
        output_dir,
        WARMUP_HISTORY_FILENAME,
        required=True,
        validator=lambda value: _validate_warmup_history(
            value,
            expected_thread_count=resolved_child_num_cpus,
        ),
    )
    resume_history_artifact = _history_artifact(
        output_dir,
        RESUME_HISTORY_FILENAME,
        required=False,
        validator=lambda value: _validate_resume_history(value, output_dir),
    )
    attestation = {
        "schema_version": 1,
        "kind": "darkofit_tabarena_regression_cap_horizon_completion",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "result_count": int(result_count),
        "expected_result_count": EXPECTED_JOBS,
        "expected_child_fits": EXPECTED_CHILD_FITS,
        "warmup_thread_count": resolved_child_num_cpus,
        "protocol_sha256": manifest["protocol_sha256"],
        "git_head": manifest["source"]["git_head"],
        "manifest_sha256": _sha256_file(manifest_path),
        "result_artifacts": artifacts,
        "analysis_payload_artifact": analysis_artifact,
        "warmup_history_artifact": warmup_history_artifact,
        "resume_history_artifact": resume_history_artifact,
        "validation": validation,
    }
    _atomic_write_json(
        output_dir / COMPLETION_ATTESTATION_FILENAME,
        attestation,
    )
    return attestation


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--time-limit",
        type=float,
        default=TIME_LIMIT_SECONDS,
        help=(
            "Frozen TabArena wall-clock budget in seconds for each outer "
            f"bagged job (must be {TIME_LIMIT_SECONDS:g})."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only a cache whose manifest exactly matches this run.",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.time_limit) or args.time_limit != TIME_LIMIT_SECONDS:
        parser.error(
            f"--time-limit is frozen at {TIME_LIMIT_SECONDS:g} seconds"
        )
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    from tabarena.contexts import TabArenaContext
    from tabarena.utils.config_utils import ConfigGenerator

    try:
        from benchmarks.tabarena_adapter import DarkoFitModel
    except ModuleNotFoundError:  # Direct execution: python benchmarks/run_*.py
        from tabarena_adapter import DarkoFitModel

    output_dir = args.output_dir.resolve()
    validate_output_state(output_dir, resume=args.resume)

    context = TabArenaContext()
    # This is a coverage/provenance gate only.  No ChimeraBoost metric enters
    # horizon selection, aggregation, or job construction.
    validate_chimera_coverage(context.load_results(methods=["ChimeraBoost"]))

    experiments = build_experiments(
        model_cls=DarkoFitModel,
        config_generator_cls=ConfigGenerator,
        time_limit=args.time_limit,
    )
    jobs = context.build_jobs(
        experiments,
        task_ids=TASK_IDS,
        split_indices=SPLIT_INDICES,
    )
    jobs = interleave_horizon_jobs(jobs)
    resolved_child_num_cpus = resolve_and_pin_child_cpu_allocation(jobs)
    print(
        f"validated {EXPECTED_DATASET_SPLITS} registered CHIMERA rows; "
        f"built {len(jobs)} paired DarkoFit jobs ({EXPECTED_CHILD_FITS} child fits); "
        f"pinned child CPUs={resolved_child_num_cpus}"
    )
    if args.dry_run:
        return 0

    source = collect_source_provenance(output_dir=output_dir)
    manifest = build_run_manifest(
        output_dir=output_dir,
        time_limit=args.time_limit,
        source=source,
        resolved_child_num_cpus=resolved_child_num_cpus,
    )
    write_or_validate_run_manifest(output_dir, manifest, resume=args.resume)
    prepare_paired_resume(output_dir, jobs, resume=args.resume)

    # Warmup is deliberately outside ``context.run_jobs`` so its one-time JIT
    # cost cannot be assigned to either horizon arm.
    try:
        from benchmarks.tabarena_warmup import warmup_tabarena_regression
    except ModuleNotFoundError:  # Direct execution: python benchmarks/run_*.py
        from tabarena_warmup import warmup_tabarena_regression

    warmup = warmup_tabarena_regression(thread_count=resolved_child_num_cpus)
    record_warmup(output_dir, warmup)

    results = context.run_jobs(
        jobs,
        expname=str(output_dir / "experiments"),
        new_result_prefix="[DarkoFit cap-horizon] ",
        debug_mode=True,
    )
    write_completion_attestation(
        output_dir,
        manifest=manifest,
        result_count=len(results),
    )
    print(f"CAP_HORIZON_COMPLETE {len(results)} {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
