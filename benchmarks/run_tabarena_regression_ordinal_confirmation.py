"""Run the frozen mechanism-unused ordinal-representation confirmation campaign.

The campaign is deliberately narrow.  It compares the real DarkoFit product
defaults (P), the fixed follow-on baseline (B), and the same fixed baseline
with the source-frozen safe ordinal representation (O).  The three diagonal
screen coordinates used to discover the mechanism are excluded from each
dataset.  All remaining jobs are bagged over eight child folds.

This runner owns and validates every artifact below its output directory.
``--resume`` must therefore be used only with a trusted cache created by this
runner; validating cached TabArena records requires unpickling those records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import stat
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

try:
    from benchmarks import run_tabarena_regression_cap_horizon as hardened
    from benchmarks import run_tabarena_regression_followon_screen as followon
except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
    import run_tabarena_regression_cap_horizon as hardened
    import run_tabarena_regression_followon_screen as followon

try:
    from benchmarks.tabarena_followon_warmup import (
        EXPECTED_WARMUP_COUNTS,
        WARMUP_KIND,
        WARMUP_SCHEMA_VERSION,
        WARMUP_STAGE_SPECS,
    )
except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
    from tabarena_followon_warmup import (
        EXPECTED_WARMUP_COUNTS,
        WARMUP_KIND,
        WARMUP_SCHEMA_VERSION,
        WARMUP_STAGE_SPECS,
    )


# Keep datasets in task-id order.  The insertion order is part of the frozen
# execution order and the order cycle restarts for each dataset.
TASK_SPLIT_COUNTS = {
    "airfoil_self_noise": (363612, 30),
    "diamonds": (363631, 9),
}
TASKS = {dataset: task_id for dataset, (task_id, _) in TASK_SPLIT_COUNTS.items()}
TASK_IDS = list(TASKS.values())
SPLIT_INDICES = [f"r{repeat}f{fold}" for repeat in range(10) for fold in range(3)]
EXCLUDED_SPLITS = ((0, 0), (1, 1), (2, 2))
EXCLUDED_COORDINATES = {
    (dataset, repeat, fold)
    for dataset in TASKS
    for repeat, fold in EXCLUDED_SPLITS
}

FIXED_BASE_CONFIG: dict[str, Any] = {
    "iterations": 1_000,
    "tree_mode": "catboost",
    "l2_leaf_reg": 3.0,
    "max_bins": 128,
    "learning_rate": 0.1,
    "ts_permutations": 1,
    "linear_residual": False,
    "early_stopping": True,
    "use_best_model": True,
}

# P must remain an empty manual configuration.  The effective values below are
# documentation and validation expectations for the product estimator whose
# source hash is sealed in the manifest; they are not passed to ConfigGenerator.
PRODUCT_EFFECTIVE_DEFAULTS: dict[str, Any] = {
    "iterations": 1_000,
    "tree_mode": "catboost",
    "l2_leaf_reg": "auto",
    "max_bins": 254,
    "learning_rate": None,
    "ts_permutations": 1,
    "linear_residual": False,
    "early_stopping": True,
    "use_best_model": True,
}
PRODUCT_ESTIMATOR_DEFAULTS = {
    **PRODUCT_EFFECTIVE_DEFAULTS,
    "early_stopping": False,
}

ARM_SPECS: dict[str, dict[str, Any]] = {
    "product_default_native": {
        "code": "P",
        "config": {},
        "effective_defaults": dict(PRODUCT_EFFECTIVE_DEFAULTS),
        "model_cls": "ScreenNativeDarkoFitModel",
        "representation": "native",
    },
    "fixed_base_native": {
        "code": "B",
        "config": dict(FIXED_BASE_CONFIG),
        "effective_defaults": dict(FIXED_BASE_CONFIG),
        "model_cls": "ScreenNativeDarkoFitModel",
        "representation": "native",
    },
    "fixed_base_safe_ordinal": {
        "code": "O",
        "config": dict(FIXED_BASE_CONFIG),
        "effective_defaults": dict(FIXED_BASE_CONFIG),
        "model_cls": "SafeOrdinalDarkoFitModel",
        "representation": "safe_ordinal",
    },
}
ARM_CODES = {spec["code"]: arm for arm, spec in ARM_SPECS.items()}
ARM_ORDER_CYCLE = (
    ("P", "B", "O"),
    ("B", "O", "P"),
    ("O", "P", "B"),
    ("O", "B", "P"),
    ("P", "O", "B"),
    ("B", "P", "O"),
)

EXPECTED_COORDINATES = 33
EXPECTED_DATASET_SPLITS = EXPECTED_COORDINATES
EXPECTED_JOBS = EXPECTED_COORDINATES * len(ARM_SPECS)
EXPECTED_CHILD_FITS = EXPECTED_JOBS * 8
# All three pairwise analytical contrasts are reported per coordinate.
EXPECTED_PAIRED_COMPARISONS = EXPECTED_COORDINATES * 3
EXPECTED_CONTRAST_PAIRS = EXPECTED_PAIRED_COMPARISONS
EXPECTED_INDEPENDENT_CONTRASTS = EXPECTED_COORDINATES * 2
EXPECTED_NATIVE_REPRESENTATION_PAIRS = EXPECTED_COORDINATES * 8
TIME_LIMIT_SECONDS = 3_600.0

MANIFEST_FILENAME = hardened.MANIFEST_FILENAME
COMPLETION_ATTESTATION_FILENAME = hardened.COMPLETION_ATTESTATION_FILENAME
ANALYSIS_PAYLOAD_FILENAME = hardened.ANALYSIS_PAYLOAD_FILENAME
WARMUP_HISTORY_FILENAME = hardened.WARMUP_HISTORY_FILENAME
RESUME_HISTORY_FILENAME = hardened.RESUME_HISTORY_FILENAME
DEFAULT_ANALYSIS_OUTPUT_FILENAMES = (
    "paired_splits.csv",
    "per_repeat.csv",
    "paired_children.csv",
    "summary.json",
    "report.md",
)
DEFAULT_OUTPUT_DIR = Path(
    ".cache/tabarena-regression-ordinal-confirmation-0.9.0-20260713"
)
CAMPAIGN_KIND = "darkofit_tabarena_regression_ordinal_confirmation"
COMPLETION_KIND = CAMPAIGN_KIND + "_completion"
PAYLOAD_KIND = CAMPAIGN_KIND + "_analysis_payload"

SOURCE_FILES = (
    Path("pyproject.toml"),
    Path("darkofit/__init__.py"),
    Path("darkofit/booster.py"),
    Path("darkofit/callbacks.py"),
    Path("darkofit/preprocessing.py"),
    Path("darkofit/serialization.py"),
    Path("darkofit/sklearn_api.py"),
    Path("benchmarks/tabarena_adapter.py"),
    Path("benchmarks/tabarena_warmup.py"),
    Path("benchmarks/tabarena_followon_warmup.py"),
    Path("benchmarks/tabarena_screen_adapters.py"),
    Path("benchmarks/run_tabarena_regression_cap_horizon.py"),
    Path("benchmarks/analyze_tabarena_regression_cap_horizon.py"),
    Path("benchmarks/run_tabarena_regression_followon_screen.py"),
    Path("benchmarks/analyze_tabarena_regression_followon_screen.py"),
    Path("benchmarks/run_tabarena_regression_ordinal_confirmation.py"),
    Path("benchmarks/analyze_tabarena_regression_ordinal_confirmation.py"),
    Path("benchmarks/tabarena_regression_ordinal_confirmation_protocol.md"),
)
PACKAGE_DISTRIBUTIONS = hardened.PACKAGE_DISTRIBUTIONS
RUNTIME_ENVIRONMENT_KEYS = hardened.RUNTIME_ENVIRONMENT_KEYS
REQUIRED_FIT_METADATA = hardened.REQUIRED_FIT_METADATA
REQUIRED_REFIT_PARAMS = hardened.REQUIRED_REFIT_PARAMS

EXPECTED_NATIVE_CATEGORICAL_COLUMNS = {
    "airfoil_self_noise": ["attack-angle"],
    "diamonds": ["cut", "color", "clarity"],
}


def expected_coordinates() -> list[tuple[str, int, int]]:
    """Return the 33 mechanism-unused dataset/repeat/fold coordinates in order."""
    coordinates = []
    for dataset, (_, split_count) in TASK_SPLIT_COUNTS.items():
        if split_count % 3:
            raise RuntimeError(f"split count for {dataset} is not fold-aligned")
        for registered_fold in range(split_count):
            coordinate = (dataset, registered_fold // 3, registered_fold % 3)
            if coordinate not in EXCLUDED_COORDINATES:
                coordinates.append(coordinate)
    if len(coordinates) != EXPECTED_COORDINATES:
        raise RuntimeError("frozen ordinal coordinate count changed")
    return coordinates


def expected_arm_coordinates(arm: str) -> list[tuple[str, int, int]]:
    if arm not in ARM_SPECS:
        raise RuntimeError(f"unexpected ordinal confirmation arm: {arm}")
    return expected_coordinates()


def expected_grid() -> set[tuple[str, int, int, str]]:
    return {
        (*coordinate, arm)
        for coordinate in expected_coordinates()
        for arm in ARM_SPECS
    }


def expected_ordered_grid() -> list[tuple[str, int, int, str]]:
    """Return the exact 99-job order, resetting the six-cycle per dataset."""
    by_dataset: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for coordinate in expected_coordinates():
        by_dataset[coordinate[0]].append(coordinate)
    ordered = []
    for dataset in TASKS:
        for index, coordinate in enumerate(by_dataset[dataset]):
            codes = ARM_ORDER_CYCLE[index % len(ARM_ORDER_CYCLE)]
            ordered.extend((*coordinate, ARM_CODES[code]) for code in codes)
    if len(ordered) != EXPECTED_JOBS or set(ordered) != expected_grid():
        raise RuntimeError("frozen ordinal job order is incomplete")
    return ordered


def job_order_sha256() -> str:
    payload = [
        {"dataset": dataset, "repeat": repeat, "fold": fold, "arm": arm}
        for dataset, repeat, fold, arm in expected_ordered_grid()
    ]
    return hashlib.sha256(hardened._canonical_json(payload)).hexdigest()


def expected_ag_ensemble_config() -> dict[str, Any]:
    return {
        "model_random_seed": 0,
        "vary_seed_across_folds": True,
        "fold_fitting_strategy": "sequential_local",
        "ag.max_time_limit": TIME_LIMIT_SECONDS,
    }


def expected_resolved_method_hyperparameters(arm: str) -> dict[str, Any]:
    if arm not in ARM_SPECS:
        raise RuntimeError(f"unexpected ordinal confirmation arm: {arm}")
    ensemble = expected_ag_ensemble_config()
    max_time_limit = ensemble.pop("ag.max_time_limit")
    ensemble["ag_args_fit"] = {"max_time_limit": max_time_limit}
    return {**ARM_SPECS[arm]["config"], "ag_args_ensemble": ensemble}


def expected_fit_kwargs_extra(num_cpus: int) -> dict[str, Any]:
    num_cpus = hardened._exact_int(num_cpus, "expected fit num_cpus")
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
    if arm not in ARM_SPECS:
        raise RuntimeError(f"unexpected ordinal confirmation arm: {arm}")
    if isinstance(child_fold, bool) or child_fold not in range(8):
        raise RuntimeError("child_fold must be an integer from 0 through 7")
    if arm == "product_default_native":
        params = {
            "iterations": 1_000,
            "early_stopping": True,
            "tree_mode": "catboost",
            "diagnostic_warnings": "never",
        }
    else:
        params = {
            **ARM_SPECS[arm]["config"],
            "diagnostic_warnings": "never",
        }
    return {**params, "random_state": int(child_fold)}


def expected_effective_child_hyperparameters(
    arm: str, child_fold: int
) -> dict[str, Any]:
    """Return the selected estimator defaults plus adapter-only diagnostics."""
    if arm not in ARM_SPECS:
        raise RuntimeError(f"unexpected ordinal confirmation arm: {arm}")
    if isinstance(child_fold, bool) or child_fold not in range(8):
        raise RuntimeError("child_fold must be an integer from 0 through 7")
    return {
        **ARM_SPECS[arm]["effective_defaults"],
        "diagnostic_warnings": "never",
        "random_state": int(child_fold),
    }


def frozen_protocol() -> dict[str, Any]:
    """Return the JSON-serializable mechanism-replication specification."""
    return {
        "task_split_counts": {
            dataset: {"task_id": task_id, "registered_split_count": split_count}
            for dataset, (task_id, split_count) in TASK_SPLIT_COUNTS.items()
        },
        "excluded_discovery_coordinates": [
            {"dataset": dataset, "repeat": repeat, "fold": fold}
            for dataset, repeat, fold in sorted(EXCLUDED_COORDINATES)
        ],
        "coordinates": [
            {"dataset": dataset, "repeat": repeat, "fold": fold}
            for dataset, repeat, fold in expected_coordinates()
        ],
        "arms": {
            arm: {
                "code": spec["code"],
                "manual_config": dict(spec["config"]),
                "effective_defaults": dict(spec["effective_defaults"]),
                "model_cls": spec["model_cls"],
                "representation": spec["representation"],
            }
            for arm, spec in ARM_SPECS.items()
        },
        "contrasts": [
            {
                "candidate": "fixed_base_safe_ordinal",
                "reference": "fixed_base_native",
                "role": "isolated_ordinal_mechanism",
            },
            {
                "candidate": "fixed_base_safe_ordinal",
                "reference": "product_default_native",
                "role": "product_relevance",
            },
            {
                "candidate": "fixed_base_native",
                "reference": "product_default_native",
                "role": "fixed_policy_context",
            },
        ],
        "order_cycle_codes": [list(order) for order in ARM_ORDER_CYCLE],
        "order_cycle_scope": "restart_for_each_dataset",
        "ordered_job_sha256": job_order_sha256(),
        "expected_coordinates": EXPECTED_COORDINATES,
        "expected_jobs": EXPECTED_JOBS,
        "expected_child_fits": EXPECTED_CHILD_FITS,
        "expected_paired_comparisons": EXPECTED_PAIRED_COMPARISONS,
        "expected_independent_contrasts": EXPECTED_INDEPENDENT_CONTRASTS,
        "bag_folds": 8,
        "bag_sets": 1,
        "seed_policy": "fold-wise",
        "seed_configuration": {
            "model_random_seed": 0,
            "vary_seed_across_folds": True,
        },
        "fold_fitting_strategy": "sequential_local",
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "warmup_outside_run_jobs": True,
        "warmup": {
            "kind": WARMUP_KIND,
            "schema_version": WARMUP_SCHEMA_VERSION,
            "stage_names": [spec["name"] for spec in WARMUP_STAGE_SPECS],
            "expected_counts": EXPECTED_WARMUP_COUNTS,
            "thread_policy": "same pinned CPU count as every measured child",
        },
        "representation_safety": {
            "native_categorical_columns": {
                dataset: list(columns)
                for dataset, columns in EXPECTED_NATIVE_CATEGORICAL_COLUMNS.items()
            },
            "native_metadata_schema_version": 2,
            "native_pair_child_count": EXPECTED_NATIVE_REPRESENTATION_PAIRS,
            "ordinal_fit_scope": "child_training_rows_only",
            "ordinal_mapping_source": "source_frozen_before_campaign",
            "ordinal_target_used": False,
            "ordinal_unknown_policy": "fail_closed",
        },
        "evidence_boundary": {
            "mechanism_unused_coordinates": EXPECTED_COORDINATES,
            "cap_campaign_previously_inspected_coordinates": True,
            "globally_unseen_confirmation": False,
            "semantics_scope": "dataset_specific_airfoil_and_diamonds_only",
        },
        "freshness": (
            "non-resume execution requires a nonexistent/empty output and all "
            "99 jobs; resume invalidates whole three-arm coordinate groups"
        ),
    }


def protocol_sha256() -> str:
    return hashlib.sha256(hardened._canonical_json(frozen_protocol())).hexdigest()


def build_experiments(
    *, model_classes: Mapping[str, type], config_generator_cls, time_limit: float
) -> dict[str, Any]:
    """Build one explicitly named experiment for each arm."""
    if float(time_limit) != TIME_LIMIT_SECONDS:
        raise ValueError(
            f"frozen campaign time_limit must be {TIME_LIMIT_SECONDS:g} seconds"
        )
    experiments = {}
    for arm, spec in ARM_SPECS.items():
        generator = config_generator_cls(
            model_cls=model_classes[spec["model_cls"]],
            manual_configs=[dict(spec["config"])],
            search_space={},
        )
        generated = generator.generate_all_bag_experiments(
            num_random_configs=0,
            name_id_suffix=f"_ordinal_confirm_{arm}",
            add_seed="fold-wise",
            fold_fitting_strategy="sequential_local",
            time_limit=time_limit,
        )
        if len(generated) != 1:
            raise RuntimeError(f"expected one ordinal experiment for {arm}")
        experiments[arm] = generated[0]
    return experiments


def _job_coordinate(job) -> tuple[str, int, int]:
    return job.task.dataset, int(job.task.repeat), int(job.task.fold)


def _experiment_name(arm: str) -> str:
    return f"DarkoFit_c1_ordinal_confirm_{arm}_BAG_L1"


def _job_arm(job) -> str:
    experiment = getattr(job, "experiment", None)
    name = getattr(experiment, "name", "")
    matches = [arm for arm in ARM_SPECS if name == _experiment_name(arm)]
    if len(matches) != 1:
        raise RuntimeError(f"could not identify ordinal arm from {name!r}")
    arm = matches[0]
    method = getattr(experiment, "method_kwargs", None)
    if not isinstance(method, Mapping):
        raise RuntimeError("ordinal job has no resolved method settings")
    raw = dict(method.get("model_hyperparameters", {}))
    ag_args = raw.pop("ag_args", None)
    ag_ensemble = raw.pop("ag_args_ensemble", None)
    if (
        raw != ARM_SPECS[arm]["config"]
        or ag_args != {"name_suffix": f"_c1_ordinal_confirm_{arm}"}
        or not isinstance(ag_ensemble, Mapping)
        or dict(ag_ensemble) != expected_ag_ensemble_config()
        or getattr(method.get("model_cls"), "__name__", None)
        != ARM_SPECS[arm]["model_cls"]
    ):
        raise RuntimeError(f"ordinal job does not match frozen arm {arm}")
    return arm


def build_confirmation_jobs(context, experiments: Mapping[str, Any]) -> list[Any]:
    jobs = []
    allowed = set(expected_coordinates())
    for experiment in experiments.values():
        built = context.build_jobs(
            [experiment], task_ids=TASK_IDS, split_indices=SPLIT_INDICES
        )
        jobs.extend(job for job in built if _job_coordinate(job) in allowed)
    return order_confirmation_jobs(jobs)


def order_confirmation_jobs(jobs: Iterable[Any]) -> list[Any]:
    """Apply the exact PBO/BOP/OPB/OBP/POB/BPO cycle per dataset."""
    grouped: dict[tuple[str, int, int], dict[str, Any]] = defaultdict(dict)
    for job in jobs:
        coordinate = _job_coordinate(job)
        arm = _job_arm(job)
        if coordinate not in set(expected_coordinates()):
            raise RuntimeError(f"unexpected ordinal coordinate: {coordinate}")
        if arm in grouped[coordinate]:
            raise RuntimeError(f"duplicate {arm} job for {coordinate}")
        grouped[coordinate][arm] = job
    if set(grouped) != set(expected_coordinates()):
        raise RuntimeError("built jobs do not match the frozen ordinal coordinates")
    ordered = []
    for dataset, repeat, fold, arm in expected_ordered_grid():
        coordinate = (dataset, repeat, fold)
        if set(grouped[coordinate]) != set(ARM_SPECS):
            raise RuntimeError(f"coordinate {coordinate} does not contain all arms")
        ordered.append(grouped[coordinate][arm])
    if len(ordered) != EXPECTED_JOBS:
        raise RuntimeError(f"expected {EXPECTED_JOBS} jobs, built {len(ordered)}")
    return ordered


def ordering_audit(jobs: Iterable[Any]) -> dict[str, Any]:
    observed = [(*_job_coordinate(job), _job_arm(job)) for job in jobs]
    expected = expected_ordered_grid()
    if observed != expected:
        raise RuntimeError("built job order does not match the frozen order cycle")
    per_dataset = {}
    overall = {arm: [0, 0, 0] for arm in ARM_SPECS}
    cursor = 0
    for dataset in TASKS:
        coordinates = [c for c in expected_coordinates() if c[0] == dataset]
        counts = {arm: [0, 0, 0] for arm in ARM_SPECS}
        for coordinate_index, coordinate in enumerate(coordinates):
            group = observed[cursor : cursor + 3]
            cursor += 3
            expected_codes = ARM_ORDER_CYCLE[
                coordinate_index % len(ARM_ORDER_CYCLE)
            ]
            expected_arms = [ARM_CODES[code] for code in expected_codes]
            if [row[3] for row in group] != expected_arms or any(
                row[:3] != coordinate for row in group
            ):
                raise RuntimeError("ordinal order cycle does not match")
            for position, arm in enumerate(expected_arms):
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
    if cursor != EXPECTED_JOBS or any(values != [11, 11, 11] for values in overall.values()):
        raise RuntimeError("ordinal job positions are not perfectly balanced")
    return {
        "job_order_sha256": job_order_sha256(),
        "per_dataset_position_counts": per_dataset,
        "overall_position_counts": {
            arm: {"first": values[0], "second": values[1], "third": values[2]}
            for arm, values in overall.items()
        },
    }


def resolve_and_pin_child_cpu_allocation(jobs: Iterable[Any]) -> int:
    """Resolve AutoGluon's automatic resource total and pin every arm alike."""
    allocations = set()
    methods = []
    seen_experiments = set()
    seen_arms = set()
    for job in jobs:
        experiment = job.experiment
        if id(experiment) in seen_experiments:
            continue
        seen_experiments.add(id(experiment))
        arm = _job_arm(job)
        seen_arms.add(arm)
        method = experiment.method_kwargs
        methods.append(method)
        fit_kwargs = method.get("fit_kwargs")
        if not isinstance(fit_kwargs, dict):
            raise RuntimeError("ordinal experiment has no fit_kwargs")
        raw_total = fit_kwargs.get("num_cpus", "auto")
        total = (
            hardened._autogluon_cpu_count()
            if raw_total in (None, "auto")
            else hardened._exact_int(raw_total, "experiment num_cpus")
        )
        model_cls = method.get("model_cls")
        probe = model_cls(
            path="",
            name="DarkoFitOrdinalResourceProbe",
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            hyperparameters=dict(ARM_SPECS[arm]["config"]),
        )
        default_cpus, default_gpus = probe._get_default_resources()
        if float(default_gpus) != 0.0:
            raise RuntimeError("ordinal child resources must be CPU-only")
        allocations.add(min(total, hardened._exact_int(default_cpus, "child CPUs")))
    if seen_arms != set(ARM_SPECS) or len(allocations) != 1:
        raise RuntimeError("ordinal arms do not share one child CPU allocation")
    allocation = next(iter(allocations))
    if allocation < 1:
        raise RuntimeError("ordinal child CPU allocation must be positive")
    for method in methods:
        method["fit_kwargs"]["num_cpus"] = allocation
    return allocation


def collect_source_provenance(output_dir: Path | None = None) -> dict[str, Any]:
    """Bind execution to a clean committed tree and exact imported sources."""
    repo_root = Path(__file__).resolve().parents[1]
    status = hardened._repository_status(repo_root, output_dir)
    if status:
        raise RuntimeError(
            "campaign repository is not clean; commit or remove every change "
            f"before executing:\n{status}"
        )
    files = {}
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise RuntimeError(f"required ordinal campaign source is missing: {relative}")
        files[str(relative)] = {
            "sha256": hardened._sha256_file(path),
            "git_blob": hardened._run_command(
                ["git", "hash-object", str(path)], cwd=repo_root
            ),
        }
    darkofit_import = hardened.collect_git_dependency_provenance(
        "darkofit", output_dir=output_dir
    )
    if Path(darkofit_import["repository"]).resolve() != repo_root.resolve():
        raise RuntimeError("imported darkofit is not this ordinal repository")
    return {
        "repository": str(repo_root),
        "git_head": hardened._run_command(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "git_tree": hardened._run_command(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=repo_root
        ),
        "relevant_status": status,
        "files": files,
        "darkofit_import": darkofit_import,
        "tabarena": hardened.collect_git_dependency_provenance(
            "tabarena", output_dir=output_dir
        ),
    }


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def collect_runtime_provenance() -> dict[str, Any]:
    """Capture exact interpreter, packages, environment, and host hardware."""
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": _package_versions(),
        "environment": {
            key: os.environ.get(key) for key in RUNTIME_ENVIRONMENT_KEYS
        },
        "hardware": hardened.collect_runtime_hardware_provenance(),
    }


def build_run_manifest(
    *,
    output_dir: Path,
    source: Mapping[str, Any],
    resolved_child_num_cpus: int,
    ordering: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": CAMPAIGN_KIND,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir.resolve()),
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "resolved_child_num_cpus": int(resolved_child_num_cpus),
        "protocol_sha256": protocol_sha256(),
        "job_order_sha256": job_order_sha256(),
        "protocol": frozen_protocol(),
        "ordering_audit": dict(ordering),
        "source": dict(source),
        "runtime": collect_runtime_provenance(),
    }


def write_or_validate_run_manifest(
    output_dir: Path, manifest: Mapping[str, Any], *, resume: bool
) -> dict[str, Any]:
    hardened.validate_output_state(output_dir, resume=resume)
    path = output_dir / MANIFEST_FILENAME
    if not resume:
        output_dir.mkdir(parents=True, exist_ok=True)
        hardened._atomic_write_json(path, manifest)
        return dict(manifest)
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("could not read existing ordinal manifest") from exc
    stable = (
        "schema_version",
        "kind",
        "output_dir",
        "time_limit_seconds",
        "resolved_child_num_cpus",
        "protocol_sha256",
        "job_order_sha256",
        "protocol",
        "ordering_audit",
        "source",
        "runtime",
    )
    mismatches = [name for name in stable if existing.get(name) != manifest.get(name)]
    if mismatches:
        raise RuntimeError(
            "resume manifest does not match the frozen ordinal campaign: "
            + ", ".join(mismatches)
        )
    return existing


def _as_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field} must be a mapping")
    return value


def _finite(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive and " if positive else ""
        raise RuntimeError(f"{field} must be {qualifier}finite")
    return number


def _arm_from_method(method: Mapping[str, Any], field: str) -> str:
    raw = dict(_as_mapping(method.get("model_hyperparameters"), field))
    ag_args = raw.pop("ag_args", None)
    ag_ensemble = raw.pop("ag_args_ensemble", None)
    model_cls_name = method.get("model_cls")
    matches = [
        arm
        for arm, spec in ARM_SPECS.items()
        if raw == spec["config"]
        and model_cls_name == spec["model_cls"]
        and ag_args == {"name_suffix": f"_c1_ordinal_confirm_{arm}"}
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{field} does not match exactly one ordinal arm")
    if not isinstance(ag_ensemble, Mapping) or dict(
        ag_ensemble
    ) != expected_ag_ensemble_config():
        raise RuntimeError(f"{field} bag seed/time configuration is not frozen")
    return matches[0]


def _expected_patience(resolved_learning_rate: float) -> int:
    return int(min(200, max(20, math.ceil(5.0 / resolved_learning_rate))))


def _validate_child_fit_metadata(
    value: Any, *, arm: str, field: str
) -> dict[str, Any]:
    """Validate and normalize one child's fitted policy and wall-clock audit."""
    if arm not in ARM_SPECS:
        raise RuntimeError(f"{field}: unexpected arm {arm}")
    fitted = dict(_as_mapping(value, field))
    if set(fitted) != set(REQUIRED_FIT_METADATA):
        raise RuntimeError(f"{field} fitted metadata fields are incomplete")
    requested = hardened._exact_int(
        fitted["iterations_requested"], f"{field}.iterations_requested"
    )
    attempted = hardened._exact_int(
        fitted["iterations_attempted"], f"{field}.iterations_attempted"
    )
    completed = hardened._exact_int(
        fitted["rounds_completed"], f"{field}.rounds_completed"
    )
    retained = hardened._exact_int(
        fitted["rounds_retained"], f"{field}.rounds_retained"
    )
    best = hardened._exact_int(fitted["best_iteration"], f"{field}.best_iteration")
    if requested != 1_000 or not (
        0 <= retained == best <= completed <= attempted <= requested
    ):
        raise RuntimeError(f"{field} round counters are inconsistent")

    learning_rate = _finite(
        fitted["resolved_learning_rate"],
        f"{field} learning rate",
        positive=True,
    )
    if arm != "product_default_native" and learning_rate != 0.1:
        raise RuntimeError(f"{field} fixed-arm learning rate does not match")
    patience = hardened._exact_int(
        fitted["early_stopping_rounds"], f"{field}.early_stopping_rounds"
    )
    if patience != _expected_patience(learning_rate):
        raise RuntimeError(f"{field} early-stopping patience does not match LR")
    if (
        fitted["requested_tree_mode"] != "catboost"
        or fitted["selected_tree_mode"] != "catboost"
        or fitted["selected_lane"] != "boosting"
        or fitted["linear_residual_active"] is not False
    ):
        raise RuntimeError(f"{field} selected mode/lane does not match")

    reason = fitted["stop_reason"]
    hardened.validate_stop_reason_causality(
        reason,
        requested=requested,
        attempted=attempted,
        completed=completed,
        field=field,
    )
    deadline_hit = fitted["deadline_hit"]
    if (
        not isinstance(deadline_hit, bool)
        or deadline_hit is not (reason == "time_limit")
        or fitted["deadline_is_soft"] is not True
    ):
        raise RuntimeError(f"{field} deadline causality does not match")
    wall_limit = _finite(
        fitted["wall_clock_limit_seconds"],
        f"{field}.wall_clock_limit_seconds",
        positive=True,
    )
    wall_margin = _finite(
        fitted["wall_clock_safety_margin_seconds"],
        f"{field}.wall_clock_safety_margin_seconds",
    )
    wall_effective = _finite(
        fitted["wall_clock_effective_seconds"],
        f"{field}.wall_clock_effective_seconds",
    )
    wall_elapsed = _finite(
        fitted["wall_clock_elapsed_seconds"],
        f"{field}.wall_clock_elapsed_seconds",
    )
    if (
        wall_limit > TIME_LIMIT_SECONDS
        or min(wall_margin, wall_effective, wall_elapsed) < 0.0
        or not math.isclose(
            wall_margin, min(5.0, 0.05 * wall_limit), abs_tol=1e-12
        )
        or not math.isclose(
            wall_effective,
            max(0.0, wall_limit - wall_margin),
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(f"{field} wall-clock audit does not match")
    return {
        "iterations_requested": requested,
        "iterations_attempted": attempted,
        "rounds_completed": completed,
        "rounds_retained": retained,
        "best_iteration": best,
        "resolved_learning_rate": learning_rate,
        "early_stopping_rounds": patience,
        "requested_tree_mode": "catboost",
        "selected_tree_mode": "catboost",
        "selected_lane": "boosting",
        "linear_residual_active": False,
        "stop_reason": reason,
        "wall_clock_limit_seconds": wall_limit,
        "wall_clock_safety_margin_seconds": wall_margin,
        "wall_clock_effective_seconds": wall_effective,
        "deadline_hit": deadline_hit,
        "deadline_is_soft": True,
        "wall_clock_elapsed_seconds": wall_elapsed,
    }


def _validate_refit_params(
    value: Any,
    *,
    arm: str,
    expected_iterations: int | None,
    expected_learning_rate: float | None,
    field: str,
) -> dict[str, Any]:
    """Validate exact-refit metadata, including product auto resolutions."""
    if arm not in ARM_SPECS:
        raise RuntimeError(f"{field}: unexpected arm {arm}")
    params = dict(_as_mapping(value, field))
    if set(params) != set(REQUIRED_REFIT_PARAMS):
        raise RuntimeError(f"{field} fields are incomplete")
    iterations = hardened._exact_int(params["iterations"], f"{field}.iterations")
    if not 0 <= iterations <= 1_000:
        raise RuntimeError(f"{field}.iterations is outside the frozen cap")
    if expected_iterations is not None and iterations != expected_iterations:
        raise RuntimeError(f"{field}.iterations does not match best iteration")
    learning_rate = _finite(
        params["learning_rate"], f"{field}.learning_rate", positive=True
    )
    if expected_learning_rate is not None and learning_rate != expected_learning_rate:
        raise RuntimeError(f"{field}.learning_rate does not match fitted metadata")
    if arm != "product_default_native" and learning_rate != 0.1:
        raise RuntimeError(f"{field}.learning_rate does not match fixed policy")
    l2 = _finite(params["l2_leaf_reg"], f"{field}.l2_leaf_reg", positive=True)
    if arm == "product_default_native":
        if not 3.0 <= l2 <= 20.0:
            raise RuntimeError(f"{field}.l2_leaf_reg is outside auto-policy bounds")
    elif l2 != 3.0:
        raise RuntimeError(f"{field}.l2_leaf_reg does not match fixed policy")
    for name, expected in {"min_child_weight": 1.0, "cat_smoothing": 1.0}.items():
        if _finite(params[name], f"{field}.{name}") != expected:
            raise RuntimeError(f"{field}.{name} does not match")
    if hardened._exact_int(
        params["min_child_samples"], f"{field}.min_child_samples"
    ) != 20:
        raise RuntimeError(f"{field}.min_child_samples does not match")
    if (
        params["tree_mode"] != "catboost"
        or hardened._exact_int(params["depth"], f"{field}.depth") != 6
        or params["num_leaves"] is not None
    ):
        raise RuntimeError(f"{field} CatBoost structure does not match")
    for name, expected in {
        "early_stopping": False,
        "early_stopping_rounds": None,
        "use_best_model": False,
        "refit": False,
    }.items():
        if params[name] is not expected:
            raise RuntimeError(f"{field}.{name} does not match")
    return params


def _validate_representation_metadata(
    value: Any,
    *,
    arm: str,
    dataset: str,
    field: str,
    child_features: list[str] | None = None,
) -> dict[str, Any]:
    """Reuse the audited schema validator under this campaign's arm names."""
    alias = {
        "product_default_native": "baseline",
        "fixed_base_native": "baseline",
        "fixed_base_safe_ordinal": "ordinal",
    }.get(arm)
    if alias is None:
        raise RuntimeError(f"{field}: unexpected arm {arm}")
    return followon._validate_representation_metadata(
        value,
        arm=alias,
        dataset=dataset,
        field=field,
        child_features=child_features,
    )


def validate_product_effective_defaults(
    model_cls=None, adapter_cls=None
) -> dict[str, Any]:
    """Prove that the empty P config resolves to the frozen product policy."""
    if model_cls is None:
        from darkofit import DarkoRegressor

        model_cls = DarkoRegressor
    params = model_cls().get_params(deep=False)
    observed = {name: params.get(name) for name in PRODUCT_EFFECTIVE_DEFAULTS}
    if observed != PRODUCT_ESTIMATOR_DEFAULTS:
        raise RuntimeError(
            "DarkoRegressor estimator defaults changed: "
            f"expected {PRODUCT_ESTIMATOR_DEFAULTS!r}, observed {observed!r}"
        )
    if adapter_cls is None:
        try:
            from benchmarks.tabarena_screen_adapters import (
                ScreenNativeDarkoFitModel,
            )
        except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
            from tabarena_screen_adapters import ScreenNativeDarkoFitModel

        adapter_cls = ScreenNativeDarkoFitModel
    adapter = adapter_cls(
        path="",
        name="DarkoFitProductDefaultProbe",
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        hyperparameters={},
    )
    adapter._set_default_params()
    adapter_params = adapter._get_model_params()
    expected_adapter_params = {
        "iterations": 1_000,
        "early_stopping": True,
        "tree_mode": "catboost",
        "diagnostic_warnings": "never",
    }
    if adapter_params != expected_adapter_params:
        raise RuntimeError(
            "DarkoFit adapter defaults changed: "
            f"expected {expected_adapter_params!r}, observed {adapter_params!r}"
        )
    return dict(PRODUCT_EFFECTIVE_DEFAULTS)


def parse_result_record(
    record: Mapping[str, Any], *, source: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one runner-owned TabArena result and normalize it to JSON."""
    if record.get("problem_type") != "regression" or record.get("metric") != "rmse":
        raise RuntimeError(f"{source}: wrong problem type or metric")
    if record.get("imputed", False) not in (False, None):
        raise RuntimeError(f"{source}: imputed result")
    metrics = {
        "test_rmse": _finite(
            record.get("metric_error"), f"{source}: test RMSE", positive=True
        ),
        "val_rmse": _finite(
            record.get("metric_error_val"), f"{source}: val RMSE", positive=True
        ),
        "train_time_s": _finite(
            record.get("time_train_s"), f"{source}: train time", positive=True
        ),
        "infer_time_s": _finite(
            record.get("time_infer_s"), f"{source}: infer time", positive=True
        ),
    }
    memory = _as_mapping(record.get("memory_usage"), f"{source}: memory")
    peak_memory = _finite(
        memory.get("peak_mem_cpu"), f"{source}: peak memory", positive=True
    )
    experiment = _as_mapping(
        record.get("experiment_metadata"), f"{source}: experiment metadata"
    )
    if (
        experiment.get("experiment_cls") != "OOFExperimentRunner"
        or experiment.get("method_cls") != "AGSingleBagWrapper"
    ):
        raise RuntimeError(f"{source}: wrong experiment implementation")

    task = _as_mapping(record.get("task_metadata"), f"{source}: task metadata")
    dataset = str(task.get("name"))
    repeat = hardened._exact_int(task.get("repeat"), f"{source}: repeat")
    fold = hardened._exact_int(task.get("fold"), f"{source}: fold")
    coordinate = (dataset, repeat, fold)
    if (
        dataset not in TASKS
        or coordinate not in set(expected_coordinates())
        or hardened._exact_int(task.get("tid"), f"{source}: task id")
        != TASKS[dataset]
        or task.get("split_idx") not in (None, 3 * repeat + fold)
    ):
        raise RuntimeError(f"{source}: task coordinate does not match confirmation")

    method = _as_mapping(record.get("method_metadata"), f"{source}: method metadata")
    arm = _arm_from_method(method, f"{source}: model hyperparameters")
    expected_suffix = f"_c1_ordinal_confirm_{arm}"
    if record.get("framework") != f"DarkoFit{expected_suffix}_BAG_L1":
        raise RuntimeError(f"{source}: framework name does not match arm")
    resolved = _as_mapping(method.get("hyperparameters"), f"{source}: resolved params")
    if dict(resolved) != expected_resolved_method_hyperparameters(arm):
        raise RuntimeError(f"{source}: resolved method policy does not match")
    if (
        method.get("model_type") != "DARKO"
        or method.get("name_prefix") != "DarkoFit"
        or method.get("init_kwargs_extra") != {}
    ):
        raise RuntimeError(f"{source}: resolved model identity does not match")
    num_cpus = hardened._exact_int(method.get("num_cpus"), f"{source}: num_cpus")
    num_gpus = hardened._exact_int(method.get("num_gpus"), f"{source}: num_gpus")
    num_cpus_child = hardened._exact_int(
        method.get("num_cpus_child"), f"{source}: num_cpus_child"
    )
    num_gpus_child = hardened._exact_int(
        method.get("num_gpus_child"), f"{source}: num_gpus_child"
    )
    if (
        num_cpus < 1
        or num_cpus_child != num_cpus
        or num_gpus != 0
        or num_gpus_child != 0
    ):
        raise RuntimeError(f"{source}: resolved resources do not match")
    if dict(
        _as_mapping(method.get("fit_kwargs_extra"), f"{source}: fit kwargs")
    ) != expected_fit_kwargs_extra(num_cpus):
        raise RuntimeError(f"{source}: bag fit settings do not match")
    outer_fit_metadata = _as_mapping(
        method.get("fit_metadata"), f"{source}: outer fit metadata"
    )
    if (
        outer_fit_metadata.get("num_cpus") != num_cpus
        or outer_fit_metadata.get("num_gpus") != 0
        or outer_fit_metadata.get("val_in_fit") is not False
        or outer_fit_metadata.get("unlabeled_in_fit") is not False
    ):
        raise RuntimeError(f"{source}: outer fit metadata does not match")

    info = _as_mapping(method.get("info"), f"{source}: model info")
    if (
        info.get("is_valid") is not True
        or info.get("can_infer") is not True
        or info.get("model_type") != "StackerEnsembleModel"
        or info.get("problem_type") != "regression"
        or info.get("eval_metric") != "root_mean_squared_error"
        or info.get("stopping_metric") != "root_mean_squared_error"
        or info.get("num_cpus") != num_cpus
        or info.get("num_gpus") != 0
        or info.get("val_in_fit") is not False
        or info.get("unlabeled_in_fit") is not False
    ):
        raise RuntimeError(f"{source}: outer model metadata does not match")
    outer_features = info.get("features")
    outer_num_features = hardened._exact_int(
        info.get("num_features"), f"{source}: outer feature count"
    )
    followon._feature_schema_sha256(outer_features, f"{source}: outer features")
    if outer_num_features != len(outer_features):
        raise RuntimeError(f"{source}: outer feature schema does not match")

    bag = _as_mapping(info.get("bagged_info"), f"{source}: bag info")
    child_names = [f"S1F{index}" for index in range(1, 9)]
    expected_model_cls = ARM_SPECS[arm]["model_cls"]
    user_config = dict(ARM_SPECS[arm]["config"])
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
        != user_config
        or dict(
            _as_mapping(
                bag.get("child_hyperparameters"), f"{source}: child params"
            )
        )
        != expected_child_hyperparameters(arm, 0)
    ):
        raise RuntimeError(f"{source}: bag construction does not match")
    expected_ag_args_fit = {
        "max_memory_usage_ratio": 1.0,
        "max_time_limit_ratio": 1.0,
        "max_time_limit": None,
        "min_time_limit": 0,
    }
    bag_ag_args = _as_mapping(
        bag.get("child_ag_args_fit"), f"{source}: child ag args"
    )
    if any(
        bag_ag_args.get(name) != value
        for name, value in expected_ag_args_fit.items()
    ):
        raise RuntimeError(f"{source}: child budget ratios do not match")

    children = _as_mapping(info.get("children_info"), f"{source}: children")
    if set(children) != set(child_names):
        raise RuntimeError(f"{source}: child set is incomplete")
    child_rows = []
    child_best = []
    child_learning_rates = []
    child_l2_values = []
    for child_fold, child_name in enumerate(child_names):
        child = _as_mapping(children[child_name], f"{source}: {child_name}")
        child_features = child.get("features")
        child_num_features = hardened._exact_int(
            child.get("num_features"), f"{source}: {child_name} feature count"
        )
        followon._feature_schema_sha256(
            child_features, f"{source}: {child_name} features"
        )
        initial_hyperparameters = dict(
            _as_mapping(
                child.get("hyperparameters"),
                f"{source}: {child_name} hyperparameters",
            )
        )
        user_hyperparameters = dict(
            _as_mapping(
                child.get("hyperparameters_user"),
                f"{source}: {child_name} user hyperparameters",
            )
        )
        if (
            child.get("name") != child_name
            or child.get("model_type") != expected_model_cls
            or child.get("is_valid") is not True
            or child.get("can_infer") is not True
            or initial_hyperparameters
            != expected_child_hyperparameters(arm, child_fold)
            or user_hyperparameters != user_config
            or child.get("num_cpus") != num_cpus_child
            or child.get("num_gpus") != num_gpus_child
            or child.get("val_in_fit") is not True
            or child.get("unlabeled_in_fit") is not False
            or child_num_features != len(child_features)
            or set(child_features) != set(outer_features)
        ):
            raise RuntimeError(f"{source}: {child_name} initialized policy mismatch")
        child_ag_args = _as_mapping(
            child.get("ag_args_fit"), f"{source}: {child_name} ag args"
        )
        if any(
            child_ag_args.get(name) != value
            for name, value in expected_ag_args_fit.items()
        ):
            raise RuntimeError(f"{source}: {child_name} budget ratios mismatch")

        fitted = _validate_child_fit_metadata(
            child.get("darkofit_fit"),
            arm=arm,
            field=f"{source}: {child_name} fitted metadata",
        )
        trained = _validate_refit_params(
            child.get("hyperparameters_fit"),
            arm=arm,
            expected_iterations=fitted["best_iteration"],
            expected_learning_rate=fitted["resolved_learning_rate"],
            field=f"{source}: {child_name} refit params",
        )
        representation = _validate_representation_metadata(
            child.get("benchmark_representation"),
            arm=arm,
            dataset=dataset,
            field=f"{source}: {child_name} representation",
            child_features=child_features,
        )
        child_best.append(fitted["best_iteration"])
        child_learning_rates.append(fitted["resolved_learning_rate"])
        child_l2_values.append(float(trained["l2_leaf_reg"]))
        child_rows.append(
            {
                "dataset": dataset,
                "task_id": TASKS[dataset],
                "repeat": repeat,
                "fold": fold,
                "registered_fold": 3 * repeat + fold,
                "arm": arm,
                "arm_code": ARM_SPECS[arm]["code"],
                "child": child_name,
                "child_fold": child_fold,
                **fitted,
                "initial_hyperparameters": initial_hyperparameters,
                "user_hyperparameters": user_hyperparameters,
                "effective_hyperparameters": expected_effective_child_hyperparameters(
                    arm, child_fold
                ),
                "child_features": list(child_features),
                "representation": representation,
                "refit_params": {name: trained[name] for name in sorted(trained)},
                "num_cpus": num_cpus_child,
                "num_gpus": num_gpus_child,
                "source": source,
            }
        )

    compressed = _validate_refit_params(
        bag.get("child_hyperparameters_fit"),
        arm=arm,
        expected_iterations=None,
        expected_learning_rate=None,
        field=f"{source}: compressed refit params",
    )
    hardened.validate_compressed_refit_iterations(
        compressed,
        child_best,
        field=f"{source}: compressed refit params",
    )
    if arm != "product_default_native" and compressed["learning_rate"] != 0.1:
        raise RuntimeError(f"{source}: compressed fixed learning rate changed")
    if arm == "product_default_native" and not math.isclose(
        float(compressed["learning_rate"]),
        sum(child_learning_rates) / len(child_learning_rates),
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise RuntimeError(
            f"{source}: compressed product learning rate is not the child mean"
        )
    if not math.isclose(
        float(compressed["l2_leaf_reg"]),
        sum(child_l2_values) / len(child_l2_values),
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise RuntimeError(f"{source}: compressed L2 is not the child mean")

    outer = {
        "dataset": dataset,
        "task_id": TASKS[dataset],
        "repeat": repeat,
        "fold": fold,
        "registered_fold": 3 * repeat + fold,
        "arm": arm,
        "arm_code": ARM_SPECS[arm]["code"],
        **metrics,
        "peak_memory_bytes": peak_memory,
        "framework": str(record["framework"]),
        "num_cpus": num_cpus,
        "num_gpus": num_gpus,
        "num_cpus_child": num_cpus_child,
        "num_gpus_child": num_gpus_child,
        "source": source,
    }
    return outer, child_rows


def _decode_result_pickle(path: Path) -> Mapping[str, Any]:
    # The underlying helper accepts gzip/plain runner-owned TabArena records
    # and rejects non-mappings.  Never call it on an untrusted cache.
    return followon._decode_result_pickle(path)


def _result_path(output_dir: Path, job: Any) -> Path:
    dataset, repeat, fold = _job_coordinate(job)
    return (
        output_dir
        / "experiments"
        / "data"
        / job.experiment.name
        / str(TASKS[dataset])
        / f"{repeat}_{fold}"
        / "results.pkl"
    )


def expected_result_relative_path(
    dataset: str, repeat: int, fold: int, arm: str
) -> str:
    if (
        dataset not in TASKS
        or arm not in ARM_SPECS
        or (dataset, repeat, fold) not in set(expected_coordinates())
    ):
        raise RuntimeError("result coordinate is outside the frozen confirmation")
    return str(
        Path("experiments")
        / "data"
        / _experiment_name(arm)
        / str(TASKS[dataset])
        / f"{repeat}_{fold}"
        / "results.pkl"
    )


def _validate_result_source_binding(
    relative: str, *, dataset: str, repeat: int, fold: int, arm: str
) -> None:
    if relative != expected_result_relative_path(dataset, repeat, fold, arm):
        raise RuntimeError("result payload is bound to the wrong frozen path")


def _cached_result_issue(path: Path, job: Any) -> str | None:
    try:
        metadata_record = path.lstat()
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "unreadable"
    if not stat.S_ISREG(metadata_record.st_mode):
        return "not_a_regular_file"
    try:
        outer, children = parse_result_record(
            _decode_result_pickle(path), source=str(path)
        )
        if (
            (outer["dataset"], outer["repeat"], outer["fold"])
            != _job_coordinate(job)
            or outer["arm"] != _job_arm(job)
            or len(children) != 8
        ):
            return "mismatched"
    except (KeyError, RuntimeError, TypeError, ValueError, OverflowError):
        return "incomplete_or_mismatched"
    return None


def _validate_resume_history(value: Any, output_dir: Path) -> None:
    """Validate archived coordinate groups without following path aliases."""
    if not isinstance(value, list) or not value:
        raise RuntimeError("resume history must be a nonempty list")
    allowed_statuses = {
        "valid",
        "missing",
        "not_a_regular_file",
        "unreadable",
        "mismatched",
        "incomplete_or_mismatched",
    }
    allowed_campaign_names = {
        COMPLETION_ATTESTATION_FILENAME,
        ANALYSIS_PAYLOAD_FILENAME,
        *DEFAULT_ANALYSIS_OUTPUT_FILENAMES,
    }

    def validate_archive_path(relative: Any, field: str) -> Path:
        if not isinstance(relative, str):
            raise RuntimeError(f"{field} must be a relative path")
        lexical = Path(relative)
        if (
            lexical.is_absolute()
            or ".." in lexical.parts
            or len(lexical.parts) < 3
            or lexical.parts[0] != "resume_invalidated"
            or not lexical.parts[1]
        ):
            raise RuntimeError(f"{field} is outside the resume archive root")
        path = output_dir / lexical
        try:
            metadata_record = path.lstat()
        except OSError as exc:
            raise RuntimeError(f"{field} does not exist") from exc
        if not stat.S_ISREG(metadata_record.st_mode):
            raise RuntimeError(f"{field} is not a regular archived file")
        try:
            path.resolve().relative_to(output_dir.resolve())
        except ValueError as exc:
            raise RuntimeError(f"{field} escapes the campaign output") from exc
        return Path(*lexical.parts[:2])

    for index, item in enumerate(value):
        item = _as_mapping(item, f"resume history[{index}]")
        required = {
            "resumed_at_utc",
            "pid",
            "invalidated_coordinate_count",
            "invalidated_result_count",
            "invalidated_coordinates",
            "archived_campaign_artifacts",
        }
        if set(item) != required:
            raise RuntimeError("resume history fields are incomplete")
        coordinate_count = hardened._exact_int(
            item["invalidated_coordinate_count"], "invalidated coordinate count"
        )
        result_count = hardened._exact_int(
            item["invalidated_result_count"], "invalidated result count"
        )
        coordinates = item["invalidated_coordinates"]
        artifacts = item["archived_campaign_artifacts"]
        if (
            not isinstance(item["resumed_at_utc"], str)
            or not item["resumed_at_utc"]
            or hardened._exact_int(item["pid"], "resume pid") <= 0
            or coordinate_count < 0
            or result_count < 0
            or not isinstance(coordinates, list)
            or len(coordinates) != coordinate_count
            or not isinstance(artifacts, list)
        ):
            raise RuntimeError("resume history counts are inconsistent")
        seen_coordinates = set()
        archived_result_total = 0
        archive_roots = set()
        for entry in coordinates:
            entry = _as_mapping(entry, "invalidated coordinate")
            if set(entry) != {"dataset", "repeat", "fold", "arm_status", "archived"}:
                raise RuntimeError("invalidated coordinate fields are incomplete")
            coordinate = (
                entry["dataset"],
                hardened._exact_int(entry["repeat"], "resume repeat"),
                hardened._exact_int(entry["fold"], "resume fold"),
            )
            if (
                coordinate not in set(expected_coordinates())
                or coordinate in seen_coordinates
                or not isinstance(entry["archived"], list)
            ):
                raise RuntimeError("invalidated coordinate is not in the campaign")
            seen_coordinates.add(coordinate)
            arm_status = entry["arm_status"]
            if (
                not isinstance(arm_status, Mapping)
                or set(arm_status) != set(ARM_SPECS)
                or any(status not in allowed_statuses for status in arm_status.values())
                or all(status == "valid" for status in arm_status.values())
            ):
                raise RuntimeError("invalidated coordinate arm status is inconsistent")
            for relative in entry["archived"]:
                lexical = Path(relative)
                root = validate_archive_path(relative, "archived result")
                if (
                    len(lexical.parts) < 5
                    or lexical.parts[2] != "experiments"
                    or lexical.name != "results.pkl"
                ):
                    raise RuntimeError("archived result path is outside experiments")
                archive_roots.add(root)
                archived_result_total += 1
        for relative in artifacts:
            lexical = Path(relative)
            root = validate_archive_path(relative, "archived campaign artifact")
            if len(lexical.parts) != 3 or lexical.name not in allowed_campaign_names:
                raise RuntimeError("archived campaign artifact path is not allowed")
            archive_roots.add(root)
        if (
            len(seen_coordinates) != coordinate_count
            or archived_result_total != result_count
            or len(archive_roots) > 1
        ):
            raise RuntimeError("resume history archive totals or roots are inconsistent")


def prepare_grouped_resume(
    output_dir: Path, jobs: Iterable[Any], *, resume: bool
) -> dict[str, Any] | None:
    """Invalidate an entire P/B/O coordinate group if any member is bad."""
    if not resume:
        return None
    grouped: dict[tuple[str, int, int], dict[str, tuple[Path, Any]]] = defaultdict(dict)
    expected_paths = set()
    for job in jobs:
        path = _result_path(output_dir, job)
        followon._require_no_symlink_components(
            path, root=output_dir, field="expected cached ordinal result"
        )
        expected_paths.add(str(path.relative_to(output_dir)))
        grouped[_job_coordinate(job)][_job_arm(job)] = (path, job)
    observed = followon._observed_regular_result_paths(output_dir)
    unexpected = set(observed).difference(expected_paths)
    if unexpected:
        raise RuntimeError(f"resume cache contains unexpected result files: {unexpected}")

    stale_paths = [
        output_dir / COMPLETION_ATTESTATION_FILENAME,
        output_dir / ANALYSIS_PAYLOAD_FILENAME,
        *(output_dir / name for name in DEFAULT_ANALYSIS_OUTPUT_FILENAMES),
    ]
    for group in grouped.values():
        for path, _ in group.values():
            hardened._require_regular_archive_source(path, "cached ordinal result")
    for path in (
        *stale_paths,
        output_dir / WARMUP_HISTORY_FILENAME,
        output_dir / RESUME_HISTORY_FILENAME,
        output_dir / MANIFEST_FILENAME,
    ):
        hardened._require_regular_archive_source(path, "ordinal campaign artifact")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_root = output_dir / "resume_invalidated" / timestamp
    invalidated = []
    invalidated_result_count = 0
    for coordinate in expected_coordinates():
        group = grouped[coordinate]
        if set(group) != set(ARM_SPECS):
            raise RuntimeError(f"resume group is incomplete for {coordinate}")
        statuses = {
            arm: _cached_result_issue(path, job)
            for arm, (path, job) in group.items()
        }
        existing = [
            (arm, path)
            for arm, (path, _) in group.items()
            if path.exists() or path.is_symlink()
        ]
        if not existing or all(issue is None for issue in statuses.values()):
            continue
        archived = []
        for arm, source in existing:
            relative = source.relative_to(output_dir)
            destination = archive_root / relative
            followon._prepare_archive_destination(destination, output_dir=output_dir)
            os.replace(source, destination)
            archived.append(str(destination.relative_to(output_dir)))
        invalidated_result_count += len(archived)
        invalidated.append(
            {
                "dataset": coordinate[0],
                "repeat": coordinate[1],
                "fold": coordinate[2],
                "arm_status": {
                    arm: issue or "valid" for arm, issue in sorted(statuses.items())
                },
                "archived": archived,
            }
        )

    archived_campaign_artifacts = []
    for source in stale_paths:
        if not source.exists():
            continue
        destination = archive_root / source.name
        followon._prepare_archive_destination(destination, output_dir=output_dir)
        os.replace(source, destination)
        archived_campaign_artifacts.append(str(destination.relative_to(output_dir)))
    record = {
        "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "invalidated_coordinate_count": len(invalidated),
        "invalidated_result_count": invalidated_result_count,
        "invalidated_coordinates": invalidated,
        "archived_campaign_artifacts": archived_campaign_artifacts,
    }
    history_path = output_dir / RESUME_HISTORY_FILENAME
    history = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("could not read ordinal resume history") from exc
        _validate_resume_history(history, output_dir)
    history.append(record)
    hardened._atomic_write_json(history_path, history)
    return record


def _stable_file_artifact(path: Path, output_dir: Path) -> dict[str, Any]:
    return followon._stable_file_artifact(path, output_dir)


def collect_result_artifacts(
    output_dir: Path, jobs: Iterable[Any]
) -> dict[str, dict[str, Any]]:
    expected = set()
    for job in jobs:
        path = _result_path(output_dir, job)
        followon._require_no_symlink_components(
            path, root=output_dir, field="expected completed ordinal result"
        )
        expected.add(str(path.relative_to(output_dir)))
    observed = followon._observed_regular_result_paths(output_dir)
    if set(observed) != expected or len(observed) != EXPECTED_JOBS:
        raise RuntimeError(
            f"completed result grid mismatch: expected {EXPECTED_JOBS}, "
            f"found {len(observed)}"
        )
    artifacts = {}
    for path in (observed[relative] for relative in sorted(observed)):
        artifact = _stable_file_artifact(path, output_dir)
        relative = artifact.pop("path")
        artifacts[relative] = artifact
    return artifacts


def validate_native_representation_pairs(
    child_rows: Iterable[Mapping[str, Any]],
) -> int:
    """Prove P and B received identical child-local native preprocessing."""
    rows = list(child_rows)
    index = {
        (
            row["dataset"],
            int(row["repeat"]),
            int(row["fold"]),
            row["arm"],
            row["child"],
        ): row
        for row in rows
    }
    if len(index) != len(rows):
        raise RuntimeError("ordinal child metadata contains duplicate rows")
    comparisons = 0
    for dataset, repeat, fold in expected_coordinates():
        for child_index in range(1, 9):
            child = f"S1F{child_index}"
            product = index.get(
                (dataset, repeat, fold, "product_default_native", child)
            )
            fixed = index.get((dataset, repeat, fold, "fixed_base_native", child))
            if (
                product is None
                or fixed is None
                or product["child_features"] != fixed["child_features"]
                or product["representation"] != fixed["representation"]
            ):
                raise RuntimeError(
                    "paired product/fixed native preprocessing is not identical"
                )
            comparisons += 1
    if comparisons != EXPECTED_NATIVE_REPRESENTATION_PAIRS:
        raise RuntimeError("native child preprocessing pair count changed")
    return comparisons


def validate_completed_results(
    output_dir: Path, artifacts: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate all 99 result files and return execution-ordered JSON rows."""
    outer_rows = []
    child_rows = []
    seen = set()
    resources = set()
    stop_reasons = Counter()
    deadline_hit_count = 0
    for relative in sorted(artifacts):
        path = output_dir / relative
        outer, children = parse_result_record(
            _decode_result_pickle(path), source=relative
        )
        key = (outer["dataset"], outer["repeat"], outer["fold"], outer["arm"])
        _validate_result_source_binding(
            relative,
            dataset=outer["dataset"],
            repeat=outer["repeat"],
            fold=outer["fold"],
            arm=outer["arm"],
        )
        if key in seen:
            raise RuntimeError(f"duplicate completed ordinal result: {key}")
        seen.add(key)
        resources.add(
            (
                outer["num_cpus"],
                outer["num_gpus"],
                outer["num_cpus_child"],
                outer["num_gpus_child"],
            )
        )
        outer_rows.append(outer)
        child_rows.extend(children)
        stop_reasons.update(child["stop_reason"] for child in children)
        deadline_hit_count += sum(child["deadline_hit"] is True for child in children)
    if seen != expected_grid():
        raise RuntimeError("completed ordinal result grid is missing or unexpected")
    if len(child_rows) != EXPECTED_CHILD_FITS:
        raise RuntimeError(
            f"expected {EXPECTED_CHILD_FITS} child fits, got {len(child_rows)}"
        )
    if len(resources) != 1:
        raise RuntimeError("ordinal results do not share one resource allocation")
    if stop_reasons.get("time_limit", 0) or deadline_hit_count:
        raise RuntimeError("ordinal confirmation contains a wall-clock-stopped child")
    resource = next(iter(resources))
    native_pairs = validate_native_representation_pairs(child_rows)

    order_rank = {key: index for index, key in enumerate(expected_ordered_grid())}
    outer_rows.sort(
        key=lambda row: order_rank[
            (row["dataset"], row["repeat"], row["fold"], row["arm"])
        ]
    )
    child_rows.sort(
        key=lambda row: (
            order_rank[
                (row["dataset"], row["repeat"], row["fold"], row["arm"])
            ],
            row["child_fold"],
        )
    )
    if [
        (row["dataset"], row["repeat"], row["fold"], row["arm"])
        for row in outer_rows
    ] != expected_ordered_grid():
        raise RuntimeError("normalized outer rows lost the frozen execution order")
    validation = {
        "result_count": len(outer_rows),
        "child_fit_count": len(child_rows),
        "paired_comparison_count": EXPECTED_PAIRED_COMPARISONS,
        "independent_contrast_count": EXPECTED_INDEPENDENT_CONTRASTS,
        "native_representation_pair_count": native_pairs,
        "failure_count": 0,
        "deadline_hit_count": deadline_hit_count,
        "time_limit_stop_count": int(stop_reasons.get("time_limit", 0)),
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
        "job_order_sha256": job_order_sha256(),
        "resource_allocation": {
            "num_cpus": resource[0],
            "num_gpus": resource[1],
            "num_cpus_child": resource[2],
            "num_gpus_child": resource[3],
        },
    }
    return validation, outer_rows, child_rows


def _history_artifact(
    output_dir: Path,
    filename: str,
    *,
    required: bool,
    validator,
) -> dict[str, Any] | None:
    path = output_dir / filename
    if not path.exists():
        if required:
            raise RuntimeError(f"required campaign history is missing: {filename}")
        return None
    hardened._require_regular_archive_source(path, "campaign history")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"campaign history is invalid: {filename}") from exc
    validator(value)
    return _stable_file_artifact(path, output_dir)


def write_completion_attestation(
    output_dir: Path,
    *,
    manifest: Mapping[str, Any],
    jobs: Iterable[Any],
    result_count: int,
) -> dict[str, Any]:
    """Normalize, hash, and seal a complete campaign after all validations."""
    jobs = list(jobs)
    if result_count != EXPECTED_JOBS:
        raise RuntimeError(
            f"expected {EXPECTED_JOBS} completed results, got {result_count}"
        )
    if ordering_audit(jobs) != manifest.get("ordering_audit"):
        raise RuntimeError("completion job order does not match the manifest")
    artifacts = collect_result_artifacts(output_dir, jobs)
    validation, outer_rows, child_rows = validate_completed_results(
        output_dir, artifacts
    )
    resources = validation["resource_allocation"]
    child_cpus = resources["num_cpus_child"]
    if child_cpus != manifest["resolved_child_num_cpus"]:
        raise RuntimeError("completed child CPUs do not match the manifest")
    payload = {
        "schema_version": 1,
        "kind": PAYLOAD_KIND,
        "protocol_sha256": manifest["protocol_sha256"],
        "job_order_sha256": manifest["job_order_sha256"],
        "result_artifacts_sha256": hashlib.sha256(
            hardened._canonical_json(artifacts)
        ).hexdigest(),
        "outer_rows": outer_rows,
        "child_rows": child_rows,
    }
    payload_path = output_dir / ANALYSIS_PAYLOAD_FILENAME
    hardened._atomic_write_json(payload_path, payload)
    payload_artifact = _stable_file_artifact(payload_path, output_dir)
    warmup_artifact = _history_artifact(
        output_dir,
        WARMUP_HISTORY_FILENAME,
        required=True,
        validator=lambda value: followon._validate_followon_warmup_history(
            value,
            expected_thread_count=child_cpus,
            expected_latest_pid=os.getpid(),
        ),
    )
    resume_artifact = _history_artifact(
        output_dir,
        RESUME_HISTORY_FILENAME,
        required=False,
        validator=lambda value: _validate_resume_history(value, output_dir),
    )
    sealed_artifacts = collect_result_artifacts(output_dir, jobs)
    if sealed_artifacts != artifacts:
        raise RuntimeError("result artifacts changed during normalization")
    final_source = collect_source_provenance(output_dir=output_dir)
    final_runtime = collect_runtime_provenance()
    if final_source != manifest.get("source"):
        raise RuntimeError("source provenance changed during the campaign")
    if final_runtime != manifest.get("runtime"):
        raise RuntimeError("runtime provenance changed during the campaign")
    manifest_path = output_dir / MANIFEST_FILENAME
    attestation = {
        "schema_version": 1,
        "kind": COMPLETION_KIND,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "result_count": result_count,
        "expected_result_count": EXPECTED_JOBS,
        "expected_child_fits": EXPECTED_CHILD_FITS,
        "expected_paired_comparisons": EXPECTED_PAIRED_COMPARISONS,
        "expected_independent_contrasts": EXPECTED_INDEPENDENT_CONTRASTS,
        "warmup_thread_count": child_cpus,
        "warmup_stage_count": len(WARMUP_STAGE_SPECS),
        "warmup_expected_counts": EXPECTED_WARMUP_COUNTS,
        "protocol_sha256": manifest["protocol_sha256"],
        "job_order_sha256": manifest["job_order_sha256"],
        "git_head": manifest["source"]["git_head"],
        "manifest_sha256": hardened._sha256_file(manifest_path),
        "result_artifacts": artifacts,
        "analysis_payload_artifact": payload_artifact,
        "warmup_history_artifact": warmup_artifact,
        "resume_history_artifact": resume_artifact,
        "validation": validation,
    }
    hardened._atomic_write_json(
        output_dir / COMPLETION_ATTESTATION_FILENAME, attestation
    )
    return attestation


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="fresh campaign directory (or this runner's trusted cache with --resume)",
    )
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_SECONDS)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume only a trusted cache created by this runner; cached result "
            "validation unpickles runner-owned artifacts and invalidates whole "
            "P/B/O coordinate groups"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and audit the exact 99-job grid without writing or fitting",
    )
    args = parser.parse_args(argv)
    if args.time_limit != TIME_LIMIT_SECONDS:
        parser.error(f"--time-limit is frozen at {TIME_LIMIT_SECONDS:g} seconds")
    if args.dry_run and args.resume:
        parser.error("--dry-run and --resume cannot be combined")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    hardened.validate_output_state(output_dir, resume=args.resume)
    validate_product_effective_defaults()

    from tabarena.contexts import TabArenaContext
    from tabarena.utils.config_utils import ConfigGenerator

    try:
        from benchmarks.tabarena_screen_adapters import (
            SafeOrdinalDarkoFitModel,
            ScreenNativeDarkoFitModel,
        )
    except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
        from tabarena_screen_adapters import (
            SafeOrdinalDarkoFitModel,
            ScreenNativeDarkoFitModel,
        )

    model_classes = {
        cls.__name__: cls for cls in (ScreenNativeDarkoFitModel, SafeOrdinalDarkoFitModel)
    }
    context = TabArenaContext()
    experiments = build_experiments(
        model_classes=model_classes,
        config_generator_cls=ConfigGenerator,
        time_limit=args.time_limit,
    )
    jobs = build_confirmation_jobs(context, experiments)
    ordering = ordering_audit(jobs)
    child_cpus = resolve_and_pin_child_cpu_allocation(jobs)
    print(
        f"built {len(jobs)} ordinal confirmation jobs "
        f"({EXPECTED_COORDINATES} mechanism-unused coordinates, "
        f"{EXPECTED_CHILD_FITS} child fits); pinned child CPUs={child_cpus}"
    )
    print(
        "order audit "
        + json.dumps(
            {
                "job_order_sha256": ordering["job_order_sha256"],
                "overall_position_counts": ordering["overall_position_counts"],
            },
            sort_keys=True,
        )
    )
    if args.dry_run:
        return 0

    source = collect_source_provenance(output_dir=output_dir)
    manifest = build_run_manifest(
        output_dir=output_dir,
        source=source,
        resolved_child_num_cpus=child_cpus,
        ordering=ordering,
    )
    manifest = write_or_validate_run_manifest(
        output_dir, manifest, resume=args.resume
    )
    prepare_grouped_resume(output_dir, jobs, resume=args.resume)

    try:
        from benchmarks.tabarena_followon_warmup import (
            warmup_tabarena_followon_screen,
        )
    except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
        from tabarena_followon_warmup import warmup_tabarena_followon_screen

    warmup = warmup_tabarena_followon_screen(thread_count=child_cpus)
    hardened.record_warmup(output_dir, warmup)
    results = context.run_jobs(
        jobs,
        expname=str(output_dir / "experiments"),
        new_result_prefix="[DarkoFit ordinal confirmation] ",
        debug_mode=True,
    )
    write_completion_attestation(
        output_dir,
        manifest=manifest,
        jobs=jobs,
        result_count=len(results),
    )
    print(f"ORDINAL_CONFIRMATION_COMPLETE {len(results)} {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
