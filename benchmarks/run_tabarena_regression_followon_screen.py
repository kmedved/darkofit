"""Run the frozen isolated TabArena regression follow-on screen.

The campaign shares one native control per outer coordinate and changes one
lever at a time: tree-mode selection, target-statistic permutations, ordinal
categories, one-hot categories, or the linear-residual lane.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import pickle
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
except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
    import run_tabarena_regression_cap_horizon as hardened

try:
    from benchmarks.tabarena_followon_warmup import (
        EXPECTED_WARMUP_COUNTS,
        WARMUP_BASE_CONFIG,
        WARMUP_KIND,
        WARMUP_SCHEMA_VERSION,
        WARMUP_STAGE_SPECS,
    )
except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
    from tabarena_followon_warmup import (
        EXPECTED_WARMUP_COUNTS,
        WARMUP_BASE_CONFIG,
        WARMUP_KIND,
        WARMUP_SCHEMA_VERSION,
        WARMUP_STAGE_SPECS,
    )


TASKS = {
    "airfoil_self_noise": 363612,
    "Another-Dataset-on-used-Fiat-500": 363615,
    "concrete_compressive_strength": 363625,
    "diamonds": 363631,
    "Food_Delivery_Time": 363672,
    "healthcare_insurance_expenses": 363675,
    "houses": 363678,
    "miami_housing": 363686,
    "physiochemical_protein": 363693,
    "QSAR-TID-11": 363697,
    "QSAR_fish_toxicity": 363698,
    "superconductivity": 363705,
    "wine_quality": 363708,
}
SCREEN_SPLITS = ((0, 0), (1, 1), (2, 2))
SCREEN_SPLIT_INDICES = tuple(f"r{repeat}f{fold}" for repeat, fold in SCREEN_SPLITS)

TS4_DATASETS = (
    "airfoil_self_noise",
    "Another-Dataset-on-used-Fiat-500",
    "diamonds",
    "Food_Delivery_Time",
    "healthcare_insurance_expenses",
)
ORDINAL_DATASETS = (
    "airfoil_self_noise",
    "diamonds",
)
ONE_HOT_DATASETS = (
    "Another-Dataset-on-used-Fiat-500",
    "diamonds",
    "Food_Delivery_Time",
    "healthcare_insurance_expenses",
    "miami_housing",
    "wine_quality",
)
EXPECTED_NATIVE_CATEGORICAL_COLUMNS = {
    "airfoil_self_noise": ["attack-angle"],
    "Another-Dataset-on-used-Fiat-500": ["model"],
    "concrete_compressive_strength": [],
    "diamonds": ["cut", "color", "clarity"],
    "Food_Delivery_Time": [
        "Delivery_person_ID",
        "Type_of_order",
        "Type_of_vehicle",
    ],
    "healthcare_insurance_expenses": ["region"],
    "houses": [],
    "miami_housing": [],
    "physiochemical_protein": [],
    "QSAR-TID-11": [],
    "QSAR_fish_toxicity": [],
    "superconductivity": [],
    "wine_quality": [],
}

BASELINE_CONFIG: dict[str, Any] = {
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
ARM_SPECS: dict[str, dict[str, Any]] = {
    "baseline": {
        "config": dict(BASELINE_CONFIG),
        "model_cls": "ScreenNativeDarkoFitModel",
        "representation": "native",
        "datasets": tuple(TASKS),
    },
    "auto": {
        "config": {**BASELINE_CONFIG, "tree_mode": "auto"},
        "model_cls": "ScreenNativeDarkoFitModel",
        "representation": "native",
        "datasets": tuple(TASKS),
    },
    "ts4": {
        "config": {**BASELINE_CONFIG, "ts_permutations": 4},
        "model_cls": "ScreenNativeDarkoFitModel",
        "representation": "native",
        "datasets": TS4_DATASETS,
    },
    "ordinal": {
        "config": dict(BASELINE_CONFIG),
        "model_cls": "SafeOrdinalDarkoFitModel",
        "representation": "safe_ordinal",
        "datasets": ORDINAL_DATASETS,
    },
    "onehot": {
        "config": dict(BASELINE_CONFIG),
        "model_cls": "SafeOneHotDarkoFitModel",
        "representation": "safe_one_hot",
        "datasets": ONE_HOT_DATASETS,
    },
    "linear": {
        "config": {**BASELINE_CONFIG, "linear_residual": True},
        "model_cls": "ScreenNativeDarkoFitModel",
        "representation": "native",
        "datasets": tuple(TASKS),
    },
}
CANDIDATE_ARMS = tuple(arm for arm in ARM_SPECS if arm != "baseline")

EXPECTED_CONTROL_JOBS = len(TASKS) * len(SCREEN_SPLITS)
EXPECTED_CANDIDATE_JOBS = sum(
    len(ARM_SPECS[arm]["datasets"]) * len(SCREEN_SPLITS)
    for arm in CANDIDATE_ARMS
)
EXPECTED_JOBS = EXPECTED_CONTROL_JOBS + EXPECTED_CANDIDATE_JOBS
EXPECTED_CHILD_FITS = EXPECTED_JOBS * 8
EXPECTED_PAIRED_COMPARISONS = EXPECTED_CANDIDATE_JOBS
EXPECTED_NATIVE_REPRESENTATION_PAIRS = 8 * sum(
    len(spec["datasets"]) * len(SCREEN_SPLITS)
    for arm, spec in ARM_SPECS.items()
    if arm != "baseline" and spec["representation"] == "native"
)
TIME_LIMIT_SECONDS = 3_600.0

MANIFEST_FILENAME = hardened.MANIFEST_FILENAME
COMPLETION_ATTESTATION_FILENAME = hardened.COMPLETION_ATTESTATION_FILENAME
ANALYSIS_PAYLOAD_FILENAME = hardened.ANALYSIS_PAYLOAD_FILENAME
WARMUP_HISTORY_FILENAME = hardened.WARMUP_HISTORY_FILENAME
RESUME_HISTORY_FILENAME = hardened.RESUME_HISTORY_FILENAME
DEFAULT_ANALYSIS_OUTPUT_FILENAMES = hardened.DEFAULT_ANALYSIS_OUTPUT_FILENAMES
DEFAULT_OUTPUT_DIR = Path(
    ".cache/tabarena-regression-followon-screen-0.9.0-20260713"
)
CAMPAIGN_KIND = "darkofit_tabarena_regression_followon_screen"
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
    Path("benchmarks/tabarena_regression_followon_screen_protocol.md"),
)
PACKAGE_DISTRIBUTIONS = hardened.PACKAGE_DISTRIBUTIONS
RUNTIME_ENVIRONMENT_KEYS = hardened.RUNTIME_ENVIRONMENT_KEYS
REQUIRED_FIT_METADATA = hardened.REQUIRED_FIT_METADATA
REQUIRED_REFIT_PARAMS = hardened.REQUIRED_REFIT_PARAMS
VALID_STOP_REASONS = hardened.VALID_STOP_REASONS


def expected_coordinates() -> list[tuple[str, int, int]]:
    return [
        (dataset, repeat, fold)
        for dataset in TASKS
        for repeat, fold in SCREEN_SPLITS
    ]


def expected_arm_coordinates(arm: str) -> list[tuple[str, int, int]]:
    if arm not in ARM_SPECS:
        raise RuntimeError(f"unexpected screen arm: {arm}")
    scope = set(ARM_SPECS[arm]["datasets"])
    return [coordinate for coordinate in expected_coordinates() if coordinate[0] in scope]


def expected_grid() -> set[tuple[str, int, int, str]]:
    return {
        (*coordinate, arm)
        for arm in ARM_SPECS
        for coordinate in expected_arm_coordinates(arm)
    }


def expected_ag_ensemble_config() -> dict[str, Any]:
    return {
        "model_random_seed": 0,
        "vary_seed_across_folds": True,
        "fold_fitting_strategy": "sequential_local",
        "ag.max_time_limit": TIME_LIMIT_SECONDS,
    }


def expected_resolved_method_hyperparameters(arm: str) -> dict[str, Any]:
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
    if isinstance(child_fold, bool) or child_fold not in range(8):
        raise RuntimeError("child_fold must be an integer from 0 through 7")
    return {
        **ARM_SPECS[arm]["config"],
        "diagnostic_warnings": "never",
        "random_state": int(child_fold),
    }


def frozen_protocol() -> dict[str, Any]:
    return {
        "tasks": dict(TASKS),
        "screen_splits": [
            {"repeat": repeat, "fold": fold, "registered_fold": 3 * repeat + fold}
            for repeat, fold in SCREEN_SPLITS
        ],
        "arms": {
            arm: {
                "config": dict(spec["config"]),
                "model_cls": spec["model_cls"],
                "representation": spec["representation"],
                "datasets": list(spec["datasets"]),
            }
            for arm, spec in ARM_SPECS.items()
        },
        "shared_control": "one baseline outer job per dataset/screen split",
        "isolation": "each candidate changes exactly one declared lever",
        "job_order": (
            "coordinate groups; each candidate alternates before/after the shared "
            "baseline within its own scoped occurrence sequence"
        ),
        "expected_control_jobs": EXPECTED_CONTROL_JOBS,
        "expected_candidate_jobs": EXPECTED_CANDIDATE_JOBS,
        "expected_jobs": EXPECTED_JOBS,
        "expected_child_fits": EXPECTED_CHILD_FITS,
        "expected_paired_comparisons": EXPECTED_PAIRED_COMPARISONS,
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
            "stage_count": len(WARMUP_STAGE_SPECS),
            "expected_counts": EXPECTED_WARMUP_COUNTS,
            "thread_policy": "same pinned CPU count as every measured child",
        },
        "representation_safety": {
            "native_categorical_columns": {
                dataset: list(columns)
                for dataset, columns in EXPECTED_NATIVE_CATEGORICAL_COLUMNS.items()
            },
            "native_metadata_schema_version": 2,
            "native_feature_alignment_policy": (
                "exact external/internal schemas with only audited AutoGluon "
                "child-fold constants removed"
            ),
            "native_representation_pair_count": (
                EXPECTED_NATIVE_REPRESENTATION_PAIRS
            ),
            "ordinal_mapping_source": "source-frozen domain semantics",
            "ordinal_compact_domains": EXPECTED_ORDINAL_COMPACT_DOMAINS,
            "ordinal_schema_sha256": EXPECTED_ORDINAL_SCHEMA_SHA256,
            "onehot_fit_scope": "child_training_rows_only",
            "target_used_by_ordinal_or_onehot": False,
            "unknown_policy": {
                "ordinal": "fail closed outside declared domain schema",
                "onehot": "all-zero unknown with a separate missing indicator",
            },
            "onehot_max_categories_per_feature": 8,
            "onehot_max_output_features": 256,
            "onehot_high_cardinality_policy": (
                "leave >8-cardinality categoricals native for DarkoFit target stats"
            ),
        },
        "cap_dependency": (
            "The preceding attested cap experiment formally retained 1,000 "
            "rounds; this source freezes that result without runtime coupling."
        ),
    }


def protocol_sha256() -> str:
    return hashlib.sha256(hardened._canonical_json(frozen_protocol())).hexdigest()


def build_experiments(*, model_classes: Mapping[str, type], config_generator_cls, time_limit: float):
    if float(time_limit) != TIME_LIMIT_SECONDS:
        raise ValueError(
            f"frozen campaign time_limit must be {TIME_LIMIT_SECONDS:g} seconds"
        )
    experiments = {}
    for arm, spec in ARM_SPECS.items():
        model_cls = model_classes[spec["model_cls"]]
        generator = config_generator_cls(
            model_cls=model_cls,
            manual_configs=[dict(spec["config"])],
            search_space={},
        )
        generated = generator.generate_all_bag_experiments(
            num_random_configs=0,
            name_id_suffix=f"_screen_{arm}",
            add_seed="fold-wise",
            fold_fitting_strategy="sequential_local",
            time_limit=time_limit,
        )
        if len(generated) != 1:
            raise RuntimeError(f"expected one experiment for {arm}")
        experiments[arm] = generated[0]
    return experiments


def _job_coordinate(job) -> tuple[str, int, int]:
    return job.task.dataset, int(job.task.repeat), int(job.task.fold)


def _job_arm(job) -> str:
    experiment = getattr(job, "experiment", None)
    name = getattr(experiment, "name", "")
    matches = [arm for arm in ARM_SPECS if name == f"DarkoFit_c1_screen_{arm}_BAG_L1"]
    if len(matches) != 1:
        raise RuntimeError(f"could not identify screen arm from experiment {name!r}")
    arm = matches[0]
    method = getattr(experiment, "method_kwargs", None)
    if not isinstance(method, Mapping):
        raise RuntimeError("screen job has no resolved method settings")
    raw = dict(method.get("model_hyperparameters", {}))
    ag_args = raw.pop("ag_args", None)
    ag_ensemble = raw.pop("ag_args_ensemble", None)
    if (
        raw != ARM_SPECS[arm]["config"]
        or ag_args != {"name_suffix": f"_c1_screen_{arm}"}
        or not isinstance(ag_ensemble, Mapping)
        or dict(ag_ensemble) != expected_ag_ensemble_config()
        or getattr(method.get("model_cls"), "__name__", None)
        != ARM_SPECS[arm]["model_cls"]
    ):
        raise RuntimeError(f"screen job does not match frozen arm {arm}")
    return arm


def build_screen_jobs(context, experiments: Mapping[str, Any]) -> list[Any]:
    jobs = []
    for arm, experiment in experiments.items():
        task_ids = [TASKS[dataset] for dataset in ARM_SPECS[arm]["datasets"]]
        jobs.extend(
            context.build_jobs(
                [experiment],
                task_ids=task_ids,
                split_indices=list(SCREEN_SPLIT_INDICES),
            )
        )
    return order_screen_jobs(jobs)


def order_screen_jobs(jobs: Iterable[Any]) -> list[Any]:
    """Balance every candidate's before/after-control exposure."""
    grouped: dict[tuple[str, int, int], dict[str, Any]] = defaultdict(dict)
    for job in jobs:
        coordinate = _job_coordinate(job)
        arm = _job_arm(job)
        if arm in grouped[coordinate]:
            raise RuntimeError(f"duplicate {arm} job for {coordinate}")
        grouped[coordinate][arm] = job

    expected_coordinates_set = set(expected_coordinates())
    if set(grouped) != expected_coordinates_set:
        raise RuntimeError("built jobs do not match the frozen screen coordinates")

    occurrence = Counter()
    ordered = []
    for coordinate in expected_coordinates():
        observed = grouped[coordinate]
        expected_arms = {
            arm
            for arm in ARM_SPECS
            if coordinate[0] in ARM_SPECS[arm]["datasets"]
        }
        if set(observed) != expected_arms:
            raise RuntimeError(
                f"coordinate {coordinate} has arms {sorted(observed)}, "
                f"expected {sorted(expected_arms)}"
            )
        before = []
        after = []
        for arm in CANDIDATE_ARMS:
            if arm not in observed:
                continue
            target = before if occurrence[arm] % 2 == 0 else after
            target.append(arm)
            occurrence[arm] += 1
        # Reverse both sides on alternate coordinate groups. Some arms share
        # the same before/after parity, so reversing only one (possibly empty)
        # side would leave their relative order fixed across the campaign.
        coordinate_index = expected_coordinates().index(coordinate)
        if coordinate_index % 2:
            before.reverse()
            after.reverse()
        ordered.extend(observed[arm] for arm in before)
        ordered.append(observed["baseline"])
        ordered.extend(observed[arm] for arm in after)

    if len(ordered) != EXPECTED_JOBS or occurrence != Counter(
        {arm: len(expected_arm_coordinates(arm)) for arm in CANDIDATE_ARMS}
    ):
        raise RuntimeError("ordered screen job count does not match the frozen grid")
    return ordered


def ordering_balance(jobs: Iterable[Any]) -> dict[str, dict[str, int]]:
    positions = defaultdict(dict)
    for index, job in enumerate(jobs):
        positions[_job_coordinate(job)][_job_arm(job)] = index
    out = {}
    for arm in CANDIDATE_ARMS:
        before = after = 0
        for coordinate in expected_arm_coordinates(arm):
            if positions[coordinate][arm] < positions[coordinate]["baseline"]:
                before += 1
            else:
                after += 1
        out[arm] = {"candidate_before": before, "candidate_after": after}
        if abs(before - after) > 1:
            raise RuntimeError(f"{arm} ordering is not balanced")
    return out


def resolve_and_pin_child_cpu_allocation(jobs: Iterable[Any]) -> int:
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
            raise RuntimeError("screen experiment has no fit_kwargs")
        raw_total = fit_kwargs.get("num_cpus", "auto")
        total = (
            hardened._autogluon_cpu_count()
            if raw_total in (None, "auto")
            else hardened._exact_int(raw_total, "experiment num_cpus")
        )
        model_cls = method.get("model_cls")
        probe = model_cls(
            path="",
            name="DarkoFitScreenResourceProbe",
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            hyperparameters=dict(ARM_SPECS[arm]["config"]),
        )
        default_cpus, default_gpus = probe._get_default_resources()
        if float(default_gpus) != 0.0:
            raise RuntimeError("screen child resources must be CPU-only")
        allocations.add(min(total, hardened._exact_int(default_cpus, "child CPUs")))
    if seen_arms != set(ARM_SPECS) or len(allocations) != 1:
        raise RuntimeError("screen arms do not share one child CPU allocation")
    allocation = next(iter(allocations))
    if allocation < 1:
        raise RuntimeError("screen child CPU allocation must be positive")
    for method in methods:
        method["fit_kwargs"]["num_cpus"] = allocation
    return allocation


def collect_source_provenance(output_dir: Path | None = None) -> dict[str, Any]:
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
        raise RuntimeError("imported darkofit is not this screen repository")
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
    ordering: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": CAMPAIGN_KIND,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir.resolve()),
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "resolved_child_num_cpus": int(resolved_child_num_cpus),
        "protocol_sha256": protocol_sha256(),
        "protocol": frozen_protocol(),
        "ordering_balance": {arm: dict(values) for arm, values in ordering.items()},
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
        raise RuntimeError("could not read existing screen manifest") from exc
    stable = (
        "schema_version",
        "kind",
        "output_dir",
        "time_limit_seconds",
        "resolved_child_num_cpus",
        "protocol_sha256",
        "protocol",
        "ordering_balance",
        "source",
        "runtime",
    )
    mismatches = [name for name in stable if existing.get(name) != manifest.get(name)]
    if mismatches:
        raise RuntimeError(
            "resume manifest does not match the frozen screen: "
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
        and ag_args == {"name_suffix": f"_c1_screen_{arm}"}
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{field} does not match exactly one screen arm")
    if not isinstance(ag_ensemble, Mapping) or dict(
        ag_ensemble
    ) != expected_ag_ensemble_config():
        raise RuntimeError(f"{field} bag seed/time configuration is not frozen")
    return matches[0]


def _validate_refit_params(
    value: Any,
    *,
    expected_iterations: int | None,
    selected_tree_mode: str | None,
    field: str,
) -> Mapping[str, Any]:
    value = _as_mapping(value, field)
    if set(value) != REQUIRED_REFIT_PARAMS:
        raise RuntimeError(f"{field} fields are incomplete")
    iterations = hardened._exact_int(value["iterations"], f"{field}.iterations")
    if not 0 <= iterations <= BASELINE_CONFIG["iterations"]:
        raise RuntimeError(f"{field}.iterations is outside the frozen cap")
    if expected_iterations is not None and iterations != expected_iterations:
        raise RuntimeError(f"{field}.iterations does not match best iteration")
    if _finite(value["learning_rate"], f"{field}.learning_rate") != 0.1:
        raise RuntimeError(f"{field}.learning_rate does not match")
    for name, expected in {
        "l2_leaf_reg": 3.0,
        "min_child_weight": 1.0,
        "cat_smoothing": 1.0,
    }.items():
        if _finite(value[name], f"{field}.{name}") != expected:
            raise RuntimeError(f"{field}.{name} does not match")
    if hardened._exact_int(
        value["min_child_samples"], f"{field}.min_child_samples"
    ) != 20:
        raise RuntimeError(f"{field}.min_child_samples does not match")
    mode = value["tree_mode"]
    if mode not in {"catboost", "lightgbm", "hybrid"}:
        raise RuntimeError(f"{field}.tree_mode is not a fitted mode")
    if selected_tree_mode is not None and mode != selected_tree_mode:
        raise RuntimeError(f"{field}.tree_mode does not match selected mode")
    depth = hardened._exact_int(value["depth"], f"{field}.depth")
    num_leaves = value["num_leaves"]
    if mode == "catboost":
        if depth != 6 or num_leaves is not None:
            raise RuntimeError(f"{field} CatBoost capacity does not match")
    else:
        # The wrapper freezes ``None``; the bound production builder resolves
        # that default internally to its 31-leaf capacity.
        if depth != -1 or num_leaves is not None:
            raise RuntimeError(f"{field} leaf-wise capacity does not match")
    for name, expected in {
        "early_stopping": False,
        "early_stopping_rounds": None,
        "use_best_model": False,
        "refit": False,
    }.items():
        if value[name] is not expected:
            raise RuntimeError(f"{field}.{name} does not match")
    return value


def _validate_auto_compressed_refit_params(
    value: Any, *, child_best: list[int], field: str
) -> Mapping[str, Any]:
    """Validate AutoGluon's fieldwise compression without inventing coherence."""
    value = _as_mapping(value, field)
    if set(value) != REQUIRED_REFIT_PARAMS:
        raise RuntimeError(f"{field} fields are incomplete")
    iterations = hardened._exact_int(value["iterations"], f"{field}.iterations")
    if not 0 <= iterations <= BASELINE_CONFIG["iterations"]:
        raise RuntimeError(f"{field}.iterations is outside the frozen cap")
    if _finite(value["learning_rate"], f"{field}.learning_rate") != 0.1:
        raise RuntimeError(f"{field}.learning_rate does not match")
    for name, expected in {
        "l2_leaf_reg": 3.0,
        "min_child_weight": 1.0,
        "cat_smoothing": 1.0,
    }.items():
        if _finite(value[name], f"{field}.{name}") != expected:
            raise RuntimeError(f"{field}.{name} does not match")
    if hardened._exact_int(
        value["min_child_samples"], f"{field}.min_child_samples"
    ) != 20:
        raise RuntimeError(f"{field}.min_child_samples does not match")
    for name, expected in {
        "early_stopping": False,
        "early_stopping_rounds": None,
        "use_best_model": False,
        "refit": False,
    }.items():
        if value[name] is not expected:
            raise RuntimeError(f"{field}.{name} does not match")
    if value["tree_mode"] not in {"catboost", "lightgbm", "hybrid"}:
        raise RuntimeError(f"{field}.tree_mode is outside the fitted modes")
    compressed_depth = hardened._exact_int(value["depth"], f"{field}.depth")
    if compressed_depth not in range(-1, 7) or value["num_leaves"] is not None:
        raise RuntimeError(f"{field} fieldwise structural values are invalid")
    hardened.validate_compressed_refit_iterations(
        value,
        child_best,
        field=field,
    )
    return value


def _validate_tree_mode_selection(
    value: Any,
    *,
    selected_tree_mode: str,
    deadline_hit: bool,
    top_level: Mapping[str, Any],
    field: str,
    expected_iterations: int | None = None,
) -> None:
    if expected_iterations is None:
        expected_iterations = hardened._exact_int(
            top_level.get("iterations_requested"),
            f"{field}.expected_iterations",
        )
    selection = _as_mapping(value, field)
    if (
        selection.get("enabled") is not True
        or selection.get("input") != "auto"
        or selection.get("selected_tree_mode") != selected_tree_mode
        or selection.get("selected_lane") != "boosting"
        or selection.get("deadline_hit") is not False
        or deadline_hit is not False
        or selection.get("candidate_count") != 3
        or selection.get("fitted_candidate_count") != 3
        or selection.get("skipped_deadline_candidate_count") != 0
        or selection.get("candidate_fit_status_counts")
        != {"fitted": 3, "skipped_deadline": 0}
        or hardened._exact_int(
            selection.get("wall_clock_stopper_count"),
            f"{field}.wall_clock_stopper_count",
        )
        != 1
    ):
        raise RuntimeError(f"{field} selection summary does not match")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or [
        item.get("tree_mode") if isinstance(item, Mapping) else None
        for item in candidates
    ] != ["catboost", "lightgbm", "hybrid"]:
        raise RuntimeError(f"{field} candidate set is incomplete")
    if sum(item.get("selected") is True for item in candidates) != 1:
        raise RuntimeError(f"{field} must select exactly one candidate")
    selected_index = hardened._exact_int(
        selection.get("selected_candidate_index"), f"{field}.selected_candidate_index"
    )
    if selected_index not in range(3) or candidates[selected_index].get(
        "selected"
    ) is not True:
        raise RuntimeError(f"{field} selected candidate index does not match")
    if not any(
        item.get("selected") is True
        and item.get("tree_mode") == selected_tree_mode
        for item in candidates
    ):
        raise RuntimeError(f"{field} selected candidate does not match")
    scores = []
    for index, item in enumerate(candidates):
        item = _as_mapping(item, f"{field}.candidates[{index}]")
        if (
            item.get("fit_status") != "fitted"
            or item.get("deadline_hit") is not False
            or item.get("deadline_hit_start") is not False
            or item.get("deadline_hit_end") is not False
        ):
            raise RuntimeError(
                f"{field} contains a deadline-hit or skipped auto candidate"
            )
        requested = hardened._exact_int(
            item.get("iterations_requested"), f"{field}.iterations_requested"
        )
        attempted = hardened._exact_int(
            item.get("iterations_attempted"), f"{field}.iterations_attempted"
        )
        completed = hardened._exact_int(
            item.get("rounds_completed"), f"{field}.rounds_completed"
        )
        retained = hardened._exact_int(
            item.get("rounds_retained"), f"{field}.rounds_retained"
        )
        best = hardened._exact_int(
            item.get("best_iteration"), f"{field}.best_iteration"
        )
        if requested != expected_iterations or not (
            0 <= retained == best <= completed <= attempted <= requested
        ):
            raise RuntimeError(f"{field} candidate round counters are inconsistent")
        if _finite(item.get("resolved_learning_rate"), f"{field}.resolved LR") != 0.1:
            raise RuntimeError(f"{field} candidate learning rate does not match")
        if item.get("lane") != "boosting":
            raise RuntimeError(f"{field} candidate lane does not match")
        reason = item.get("stop_reason")
        if reason == "time_limit":
            raise RuntimeError(f"{field} contains a wall-clock-stopped candidate")
        hardened.validate_stop_reason_causality(
            reason,
            requested=requested,
            attempted=attempted,
            completed=completed,
            field=f"{field}.candidates[{index}]",
        )
        score = _finite(
            item.get("validation_score"), f"{field}.validation_score"
        )
        if _finite(item.get("score"), f"{field}.score") != score:
            raise RuntimeError(f"{field} candidate score fields disagree")
        scores.append(score)
    expected_selected_index = min(range(len(scores)), key=scores.__getitem__)
    selected_score = _finite(selection.get("selected_score"), f"{field}.selected_score")
    if (
        selected_index != expected_selected_index
        or selected_score != scores[expected_selected_index]
        or candidates[expected_selected_index].get("tree_mode")
        != selected_tree_mode
    ):
        raise RuntimeError(f"{field} selection does not follow the minimum-score tie rule")
    selected_candidate = _as_mapping(
        candidates[expected_selected_index], f"{field}.selected_candidate"
    )
    exact_top_level_fields = (
        "iterations_requested",
        "iterations_attempted",
        "rounds_completed",
        "rounds_retained",
        "best_iteration",
        "resolved_learning_rate",
        "stop_reason",
        "deadline_hit",
    )
    if any(
        selected_candidate.get(name) != top_level.get(name)
        for name in exact_top_level_fields
    ) or (
        selected_candidate.get("tree_mode") != top_level.get("selected_tree_mode")
        or selected_candidate.get("lane") != top_level.get("selected_lane")
    ):
        raise RuntimeError(f"{field} selected candidate disagrees with child metadata")


def _validate_linear_lane(
    *, arm: str, linear_active: Any, selected_lane: Any, field: str
) -> str:
    expected_active = arm == "linear"
    if not isinstance(linear_active, bool) or linear_active is not expected_active:
        raise RuntimeError(f"{field} linear lane activation mismatch")
    expected_lane = "linear_residual" if expected_active else "boosting"
    if selected_lane != expected_lane:
        raise RuntimeError(f"{field} selected lane mismatch")
    return expected_lane


def _validate_early_stopping_rounds(value: Any, *, field: str) -> int:
    rounds = hardened._exact_int(value, field)
    if rounds != 50:
        raise RuntimeError(f"{field} must equal the frozen 50-round patience")
    return rounds


EXPECTED_CATEGORICAL_POSITIONS = {
    "airfoil_self_noise": [4],
    "Another-Dataset-on-used-Fiat-500": [6],
    "diamonds": [6, 7, 8],
    "Food_Delivery_Time": [6, 7, 8],
    "healthcare_insurance_expenses": [1, 4, 5],
    "miami_housing": [12],
    "wine_quality": [11],
}
EXPECTED_ORDINAL_DOMAINS = {
    "airfoil_self_noise": "airfoil_attack_angle_numeric",
    "diamonds": "diamonds_declared_orders",
    "miami_housing": "miami_avno60plus_binary",
}
EXPECTED_ORDINAL_COMPACT_DOMAINS = {
    "airfoil_self_noise": {"attack-angle": list(range(27))},
    "diamonds": {
        "cut": list(range(5)),
        "color": list(range(7)),
        "clarity": list(range(8)),
    },
    "miami_housing": {},
}
EXPECTED_ORDINAL_OBSERVED_MAX = {
    "airfoil_self_noise": [27],
    "diamonds": [5, 7, 8],
    "miami_housing": [2],
}
EXPECTED_ORDINAL_SCHEMA_SHA256 = {
    "airfoil_self_noise": "c592ba78a13e5434afb8980820a0c0ab668db594fb2491950286c66a3fe071a1",
    "diamonds": "98bc5774a472b47a6c7fd9fbf14fa7ca784877e7896d1cef3c43c41673f21cf6",
    "miami_housing": "6023d16bf8d362478a9f99dd468b4f724ccb1240898f34645f31a7304b4acb47",
}
EXPECTED_ONEHOT_DOMAINS = {
    "Another-Dataset-on-used-Fiat-500": "fiat",
    "diamonds": "diamonds",
    "Food_Delivery_Time": "food_delivery",
    "healthcare_insurance_expenses": "healthcare",
    "miami_housing": "miami",
    "wine_quality": "wine",
}


def _validate_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"{field} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RuntimeError(f"{field} must be a SHA-256 digest") from exc
    return value


def _feature_schema_sha256(columns: list[str], field: str) -> str:
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(column, str) for column in columns)
        or len(set(columns)) != len(columns)
    ):
        raise RuntimeError(f"{field} must contain unique string feature names")
    return hashlib.sha256(
        json.dumps(
            columns,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validate_representation_metadata(
    value: Any,
    *,
    arm: str,
    dataset: str,
    field: str,
    child_features: list[str] | None = None,
) -> dict[str, Any]:
    value = dict(_as_mapping(value, field))
    expected_kind = ARM_SPECS[arm]["representation"]
    if value.get("kind") != expected_kind:
        raise RuntimeError(f"{field}.kind does not match arm {arm}")
    input_count = hardened._exact_int(
        value.get("input_feature_count"), f"{field}.input_feature_count"
    )
    output_count = hardened._exact_int(
        value.get("output_feature_count"), f"{field}.output_feature_count"
    )
    if input_count < 1 or output_count < 1:
        raise RuntimeError(f"{field} feature counts must be positive")
    if expected_kind == "native":
        required = {
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
        if (
            set(value) != required
            or value.get("schema_version") != 2
            or value.get("fit_scope") != "darkofit_child_training_fold"
            or value.get("feature_alignment_policy")
            != "autogluon_child_drop_unique"
        ):
            raise RuntimeError(f"{field} native metadata is incomplete")
        external_digest = _validate_sha256(
            value.get("external_feature_schema_sha256"),
            f"{field}.external_feature_schema_sha256",
        )
        fitted_digest = _validate_sha256(
            value.get("fitted_feature_schema_sha256"),
            f"{field}.fitted_feature_schema_sha256",
        )
        dropped_columns = value.get("dropped_constant_input_columns")
        dropped_unique_counts = value.get(
            "dropped_constant_input_unique_counts"
        )
        if (
            not isinstance(dropped_columns, list)
            or any(not isinstance(column, str) for column in dropped_columns)
            or len(set(dropped_columns)) != len(dropped_columns)
            or not isinstance(dropped_unique_counts, list)
            or len(dropped_unique_counts) != len(dropped_columns)
            or any(
                hardened._exact_int(count, f"{field}.dropped unique count") != 1
                for count in dropped_unique_counts
            )
            or output_count != input_count - len(dropped_columns)
        ):
            raise RuntimeError(f"{field} native constant-drop audit is inconsistent")
        expected_columns = EXPECTED_NATIVE_CATEGORICAL_COLUMNS[dataset]
        fitted_categorical_columns = value.get(
            "fitted_categorical_input_columns"
        )
        expected_fitted_categorical = [
            column for column in expected_columns if column not in dropped_columns
        ]
        if (
            value.get("categorical_input_columns") != expected_columns
            or fitted_categorical_columns != expected_fitted_categorical
            or value.get("target_used_by_representation")
            is not bool(expected_fitted_categorical)
        ):
            raise RuntimeError(f"{field} native feature schema is inconsistent")
        if arm == "ts4" and fitted_categorical_columns != expected_columns:
            raise RuntimeError(
                f"{field} TS4 categorical schema was removed before fitting"
            )
        if child_features is not None:
            expected_external_digest = _feature_schema_sha256(
                child_features, f"{field}.child_features"
            )
            if (
                input_count != len(child_features)
                or external_digest != expected_external_digest
                or any(column not in child_features for column in dropped_columns)
            ):
                raise RuntimeError(
                    f"{field} native external schema is not bound to the child"
                )
            dropped_set = set(dropped_columns)
            fitted_features = [
                column for column in child_features if column not in dropped_set
            ]
            if fitted_digest != _feature_schema_sha256(
                fitted_features, f"{field}.fitted_features"
            ):
                raise RuntimeError(
                    f"{field} native fitted schema does not match audited drops"
                )
        return value

    common_required = {
        "kind",
        "fit_scope",
        "target_used_by_representation",
        "fit_calls",
        "eval_transform_calls_during_fit",
        "eval_unknown_counts",
        "input_feature_count",
        "output_feature_count",
        "categorical_input_positions",
        "category_schema_sha256",
    }
    if not common_required.issubset(value):
        raise RuntimeError(f"{field} safe representation metadata is incomplete")
    if (
        value.get("fit_scope") != "child_training_rows_only"
        or value.get("target_used_by_representation") is not False
        or hardened._exact_int(value.get("fit_calls"), f"{field}.fit_calls") != 1
    ):
        raise RuntimeError(f"{field} does not prove target-free child-only fitting")
    eval_calls = hardened._exact_int(
        value.get("eval_transform_calls_during_fit"), f"{field}.eval calls"
    )
    unknown_counts = value.get("eval_unknown_counts")
    if (
        eval_calls < 1
        or not isinstance(unknown_counts, list)
        or len(unknown_counts) != eval_calls
        or any(
            hardened._exact_int(count, f"{field}.unknown count") < 0
            for count in unknown_counts
        )
    ):
        raise RuntimeError(f"{field} eval transform audit is inconsistent")
    schema_sha256 = _validate_sha256(
        value.get("category_schema_sha256"), f"{field}.schema"
    )
    categorical_positions = value.get("categorical_input_positions")
    if categorical_positions != EXPECTED_CATEGORICAL_POSITIONS[dataset]:
        raise RuntimeError(f"{field} categorical positions do not match dataset")

    if expected_kind == "safe_ordinal":
        required = common_required | {
            "domain",
            "mapping_source",
            "observed_training_category_counts",
            "compact_category_domains",
            "missing_policy",
            "unknown_policy",
            "remaining_native_target_stat_positions",
        }
        if set(value) != required:
            raise RuntimeError(f"{field} ordinal metadata fields are not exact")
        observed_counts = value.get("observed_training_category_counts")
        expected_maxima = EXPECTED_ORDINAL_OBSERVED_MAX[dataset]
        if (
            value.get("domain") != EXPECTED_ORDINAL_DOMAINS[dataset]
            or value.get("compact_category_domains")
            != EXPECTED_ORDINAL_COMPACT_DOMAINS[dataset]
            or value.get("mapping_source") != "source_frozen_before_campaign"
            or value.get("missing_policy") != "numeric_nan"
            or value.get("unknown_policy") != "fail_closed"
            or value.get("remaining_native_target_stat_positions") != []
            or schema_sha256 != EXPECTED_ORDINAL_SCHEMA_SHA256[dataset]
            or not isinstance(observed_counts, list)
            or len(observed_counts) != len(expected_maxima)
            or any(
                hardened._exact_int(count, f"{field}.observed category count")
                not in range(1, maximum + 1)
                for count, maximum in zip(observed_counts, expected_maxima)
            )
            or any(count != 0 for count in unknown_counts)
            or input_count != output_count
        ):
            raise RuntimeError(f"{field} ordinal safety policy does not match")
        return value

    required = common_required | {
        "domain",
        "target_free_one_hot_input_positions",
        "target_free_one_hot_category_counts",
        "remaining_native_target_stat_input_positions",
        "remaining_native_target_stat_output_positions",
        "remaining_native_target_stats_use_target",
        "unknown_policy",
        "missing_indicator_per_categorical",
        "max_categories_per_feature",
        "max_output_features",
    }
    if set(value) != required:
        raise RuntimeError(f"{field} one-hot metadata fields are not exact")
    onehot_positions = value["target_free_one_hot_input_positions"]
    native_positions = value["remaining_native_target_stat_input_positions"]
    native_output_positions = value[
        "remaining_native_target_stat_output_positions"
    ]
    counts = value["target_free_one_hot_category_counts"]
    expected_native = [6] if dataset == "Food_Delivery_Time" else []
    if (
        value.get("domain") != EXPECTED_ONEHOT_DOMAINS[dataset]
        or sorted(onehot_positions + native_positions) != categorical_positions
        or native_positions != expected_native
        or native_output_positions != expected_native
        or len(counts) != len(onehot_positions)
        or not counts
        or any(
            hardened._exact_int(count, f"{field}.category count") not in range(1, 9)
            for count in counts
        )
        or value.get("remaining_native_target_stats_use_target")
        is not bool(native_positions)
        or value.get("unknown_policy") != "all_zero"
        or value.get("missing_indicator_per_categorical") is not True
        or value.get("max_categories_per_feature") != 8
        or value.get("max_output_features") != 256
        or output_count > 256
    ):
        raise RuntimeError(f"{field} one-hot/native partition is unsafe")
    return value


def parse_result_record(
    record: Mapping[str, Any], *, source: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if record.get("problem_type") != "regression" or record.get("metric") != "rmse":
        raise RuntimeError(f"{source}: wrong problem type or metric")
    if record.get("imputed", False) not in (False, None):
        raise RuntimeError(f"{source}: imputed result")
    metrics = {
        "test_rmse": _finite(record.get("metric_error"), f"{source}: test RMSE", positive=True),
        "val_rmse": _finite(record.get("metric_error_val"), f"{source}: val RMSE", positive=True),
        "train_time_s": _finite(record.get("time_train_s"), f"{source}: train time", positive=True),
        "infer_time_s": _finite(record.get("time_infer_s"), f"{source}: infer time", positive=True),
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
    if dataset not in TASKS:
        raise RuntimeError(f"{source}: unexpected dataset {dataset}")
    repeat = hardened._exact_int(task.get("repeat"), f"{source}: repeat")
    fold = hardened._exact_int(task.get("fold"), f"{source}: fold")
    coordinate = (dataset, repeat, fold)
    if (
        coordinate not in set(expected_coordinates())
        or hardened._exact_int(task.get("tid"), f"{source}: task id") != TASKS[dataset]
        or task.get("split_idx") not in (None, 3 * repeat + fold)
    ):
        raise RuntimeError(f"{source}: task coordinate does not match screen")

    method = _as_mapping(record.get("method_metadata"), f"{source}: method metadata")
    arm = _arm_from_method(method, f"{source}: model hyperparameters")
    if coordinate not in set(expected_arm_coordinates(arm)):
        raise RuntimeError(f"{source}: {arm} is outside its declared scope")
    expected_suffix = f"_c1_screen_{arm}"
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
    if num_cpus < 1 or num_cpus_child != num_cpus or num_gpus != 0 or num_gpus_child != 0:
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
    _feature_schema_sha256(outer_features, f"{source}: outer features")
    if outer_num_features != len(outer_features):
        raise RuntimeError(f"{source}: outer feature schema does not match")

    bag = _as_mapping(info.get("bagged_info"), f"{source}: bag info")
    child_names = [f"S1F{index}" for index in range(1, 9)]
    expected_model_cls = ARM_SPECS[arm]["model_cls"]
    if (
        bag.get("num_child_models") != 8
        or bag.get("child_model_type") != expected_model_cls
        or bag.get("child_model_names") != child_names
        or bag.get("_n_repeats") != 1
        or bag.get("_k_per_n_repeat") != [8]
        or bag.get("_random_state") != 1
        or bag.get("bagged_mode") is not True
        or dict(_as_mapping(bag.get("child_hyperparameters_user"), "child user params"))
        != ARM_SPECS[arm]["config"]
        or dict(_as_mapping(bag.get("child_hyperparameters"), "child params"))
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
    if any(bag_ag_args.get(name) != value for name, value in expected_ag_args_fit.items()):
        raise RuntimeError(f"{source}: child budget ratios do not match")

    children = _as_mapping(info.get("children_info"), f"{source}: children")
    if set(children) != set(child_names):
        raise RuntimeError(f"{source}: child set is incomplete")
    child_rows = []
    child_best = []
    for child_fold, child_name in enumerate(child_names):
        child = _as_mapping(children[child_name], f"{source}: {child_name}")
        child_features = child.get("features")
        child_num_features = hardened._exact_int(
            child.get("num_features"), f"{source}: {child_name} feature count"
        )
        _feature_schema_sha256(
            child_features, f"{source}: {child_name} features"
        )
        if (
            child.get("name") != child_name
            or child.get("model_type") != expected_model_cls
            or child.get("is_valid") is not True
            or child.get("can_infer") is not True
            or dict(_as_mapping(child.get("hyperparameters"), "child hyperparameters"))
            != expected_child_hyperparameters(arm, child_fold)
            or dict(_as_mapping(child.get("hyperparameters_user"), "child user params"))
            != ARM_SPECS[arm]["config"]
            or child.get("num_cpus") != num_cpus_child
            or child.get("num_gpus") != num_gpus_child
            or child.get("val_in_fit") is not True
            or child.get("unlabeled_in_fit") is not False
            or child_num_features != len(child_features)
            or set(child_features) != set(outer_features)
        ):
            raise RuntimeError(f"{source}: {child_name} initialized policy mismatch")
        child_ag_args = _as_mapping(
            child.get("ag_args_fit"), f"{source}: child ag args"
        )
        if any(
            child_ag_args.get(name) != value
            for name, value in expected_ag_args_fit.items()
        ):
            raise RuntimeError(f"{source}: {child_name} budget ratios mismatch")
        fitted = _as_mapping(child.get("darkofit_fit"), f"{source}: {child_name} fitted")
        expected_fit_fields = set(REQUIRED_FIT_METADATA)
        if arm == "auto":
            expected_fit_fields.add("tree_mode_selection")
        if set(fitted) != expected_fit_fields:
            raise RuntimeError(f"{source}: {child_name} fitted metadata is incomplete")
        requested = hardened._exact_int(fitted["iterations_requested"], "requested")
        attempted = hardened._exact_int(fitted["iterations_attempted"], "attempted")
        completed = hardened._exact_int(fitted["rounds_completed"], "completed")
        retained = hardened._exact_int(fitted["rounds_retained"], "retained")
        best = hardened._exact_int(fitted["best_iteration"], "best")
        expected_iterations = hardened._exact_int(
            ARM_SPECS[arm]["config"]["iterations"], "configured iterations"
        )
        if requested != expected_iterations or not (
            0 <= retained == best <= completed <= attempted <= requested
        ):
            raise RuntimeError(f"{source}: {child_name} round counters are inconsistent")
        if _finite(fitted["resolved_learning_rate"], "resolved LR") != 0.1:
            raise RuntimeError(f"{source}: {child_name} learning rate mismatch")
        _validate_early_stopping_rounds(
            fitted["early_stopping_rounds"],
            field=f"{source}: {child_name} early stopping rounds",
        )
        requested_mode = ARM_SPECS[arm]["config"]["tree_mode"]
        selected_mode = str(fitted["selected_tree_mode"])
        if (
            fitted["requested_tree_mode"] != requested_mode
            or selected_mode not in {"catboost", "lightgbm", "hybrid"}
            or (requested_mode != "auto" and selected_mode != requested_mode)
        ):
            raise RuntimeError(f"{source}: {child_name} selected mode mismatch")
        linear_active = fitted["linear_residual_active"]
        expected_lane = _validate_linear_lane(
            arm=arm,
            linear_active=linear_active,
            selected_lane=fitted["selected_lane"],
            field=f"{source}: {child_name}",
        )
        reason = fitted["stop_reason"]
        hardened.validate_stop_reason_causality(
            reason,
            requested=requested,
            attempted=attempted,
            completed=completed,
            field=f"{source}: {child_name}",
        )
        deadline_hit = fitted["deadline_hit"]
        if not isinstance(deadline_hit, bool) or fitted["deadline_is_soft"] is not True:
            raise RuntimeError(f"{source}: {child_name} deadline flags invalid")
        if arm != "auto" and deadline_hit is not (reason == "time_limit"):
            raise RuntimeError(f"{source}: {child_name} deadline causality mismatch")
        wall_limit = _finite(fitted["wall_clock_limit_seconds"], "wall limit", positive=True)
        wall_margin = _finite(fitted["wall_clock_safety_margin_seconds"], "wall margin")
        wall_effective = _finite(fitted["wall_clock_effective_seconds"], "wall effective")
        wall_elapsed = _finite(fitted["wall_clock_elapsed_seconds"], "wall elapsed")
        if (
            wall_limit > TIME_LIMIT_SECONDS
            or min(wall_margin, wall_effective, wall_elapsed) < 0.0
            or not math.isclose(wall_margin, min(5.0, 0.05 * wall_limit), abs_tol=1e-12)
            or not math.isclose(wall_effective, max(0.0, wall_limit - wall_margin), abs_tol=1e-12)
        ):
            raise RuntimeError(f"{source}: {child_name} wall-clock audit mismatch")
        trained = _validate_refit_params(
            child.get("hyperparameters_fit"),
            expected_iterations=best,
            selected_tree_mode=selected_mode,
            field=f"{source}: {child_name} refit params",
        )
        tree_mode_selection = None
        if arm == "auto":
            tree_mode_selection = dict(
                _as_mapping(
                    fitted["tree_mode_selection"],
                    f"{source}: {child_name} tree selection",
                )
            )
            _validate_tree_mode_selection(
                tree_mode_selection,
                expected_iterations=expected_iterations,
                selected_tree_mode=selected_mode,
                deadline_hit=deadline_hit,
                top_level=fitted,
                field=f"{source}: {child_name} tree selection",
            )
        representation = _validate_representation_metadata(
            child.get("benchmark_representation"),
            arm=arm,
            dataset=dataset,
            field=f"{source}: {child_name} representation",
            child_features=child_features,
        )
        child_best.append(best)
        child_rows.append(
            {
                "dataset": dataset,
                "task_id": TASKS[dataset],
                "repeat": repeat,
                "fold": fold,
                "registered_fold": 3 * repeat + fold,
                "arm": arm,
                "child": child_name,
                "child_fold": child_fold,
                "iterations_requested": requested,
                "iterations_attempted": attempted,
                "rounds_completed": completed,
                "rounds_retained": retained,
                "best_iteration": best,
                "resolved_learning_rate": 0.1,
                "early_stopping_rounds": 50,
                "requested_tree_mode": requested_mode,
                "selected_tree_mode": selected_mode,
                "selected_lane": expected_lane,
                "linear_residual_active": linear_active,
                "tree_mode_selection": tree_mode_selection,
                "stop_reason": reason,
                "deadline_hit": deadline_hit,
                "wall_clock_elapsed_seconds": wall_elapsed,
                "child_features": list(child_features),
                "representation": representation,
                "refit_params": {name: trained[name] for name in sorted(trained)},
                "num_cpus": num_cpus_child,
                "num_gpus": num_gpus_child,
                "source": source,
            }
        )
    compressed_field = f"{source}: compressed refit params"
    if arm == "auto":
        compressed = _validate_auto_compressed_refit_params(
            bag.get("child_hyperparameters_fit"),
            child_best=child_best,
            field=compressed_field,
        )
    else:
        compressed = _validate_refit_params(
            bag.get("child_hyperparameters_fit"),
            expected_iterations=None,
            selected_tree_mode=None,
            field=compressed_field,
        )
        hardened.validate_compressed_refit_iterations(
            compressed,
            child_best,
            field=compressed_field,
        )
    outer = {
        "dataset": dataset,
        "task_id": TASKS[dataset],
        "repeat": repeat,
        "fold": fold,
        "registered_fold": 3 * repeat + fold,
        "arm": arm,
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
    try:
        payload = path.read_bytes()
        if payload[:2] == b"\x1f\x8b":
            payload = gzip.decompress(payload)
        record = pickle.loads(payload)
    except Exception as exc:
        raise RuntimeError(f"could not decode result artifact {path}") from exc
    return _as_mapping(record, f"result artifact {path}")


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
        or (dataset, repeat, fold) not in set(expected_arm_coordinates(arm))
    ):
        raise RuntimeError("result coordinate is outside the frozen screen")
    return str(
        Path("experiments")
        / "data"
        / f"DarkoFit_c1_screen_{arm}_BAG_L1"
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
        metadata = path.lstat()
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "unreadable"
    if not stat.S_ISREG(metadata.st_mode):
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
            metadata = path.lstat()
        except OSError as exc:
            raise RuntimeError(f"{field} does not exist") from exc
        if not stat.S_ISREG(metadata.st_mode):
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
        count = hardened._exact_int(
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
            or
            count < 0
            or result_count < 0
            or not isinstance(coordinates, list)
            or len(coordinates) != count
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
                raise RuntimeError("invalidated coordinate is not in the screen")
            seen_coordinates.add(coordinate)
            expected_arms = {
                arm
                for arm, spec in ARM_SPECS.items()
                if coordinate[0] in spec["datasets"]
            }
            arm_status = entry["arm_status"]
            if (
                not isinstance(arm_status, Mapping)
                or set(arm_status) != expected_arms
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
            len(seen_coordinates) != count
            or archived_result_total != result_count
            or len(archive_roots) > 1
        ):
            raise RuntimeError("resume history archive totals or roots are inconsistent")


def _require_no_symlink_components(
    path: Path, *, root: Path, field: str
) -> None:
    """Reject any existing symlink from ``root`` through ``path`` itself."""
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{field} escapes its trusted root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise RuntimeError(f"could not inspect {field}: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{field} contains a symlinked path component: {current}")


def _prepare_archive_destination(path: Path, *, output_dir: Path) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"resume archive destination already exists: {path}")
    _require_no_symlink_components(
        path.parent, root=output_dir, field="resume archive destination"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_components(
        path.parent, root=output_dir, field="resume archive destination"
    )


def _observed_regular_result_paths(output_dir: Path) -> dict[str, Path]:
    """Return result files by lexical path, rejecting aliases and links."""
    observed: dict[str, Path] = {}
    for path in (output_dir / "experiments").rglob("results.pkl"):
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise RuntimeError(f"could not inspect result artifact: {path}") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"result artifact is not a regular file: {path}")
        relative = str(path.relative_to(output_dir))
        try:
            resolved_relative = str(path.resolve().relative_to(output_dir.resolve()))
        except ValueError as exc:
            raise RuntimeError(f"result artifact escapes campaign output: {path}") from exc
        if resolved_relative != relative:
            raise RuntimeError(f"result artifact uses a symlinked path alias: {path}")
        if relative in observed:
            raise RuntimeError(f"duplicate lexical result path: {relative}")
        observed[relative] = path
    return observed


def prepare_grouped_resume(
    output_dir: Path, jobs: Iterable[Any], *, resume: bool
) -> dict[str, Any] | None:
    """Invalidate a coordinate's whole shared-control group atomically."""
    if not resume:
        return None
    grouped: dict[tuple[str, int, int], dict[str, tuple[Path, Any]]] = defaultdict(dict)
    expected_paths = set()
    for job in jobs:
        path = _result_path(output_dir, job)
        _require_no_symlink_components(
            path, root=output_dir, field="expected cached screen result"
        )
        expected_paths.add(str(path.relative_to(output_dir)))
        grouped[_job_coordinate(job)][_job_arm(job)] = (path, job)
    observed = _observed_regular_result_paths(output_dir)
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
            hardened._require_regular_archive_source(path, "cached screen result")
    for path in (
        *stale_paths,
        output_dir / WARMUP_HISTORY_FILENAME,
        output_dir / RESUME_HISTORY_FILENAME,
        output_dir / MANIFEST_FILENAME,
    ):
        hardened._require_regular_archive_source(path, "screen campaign artifact")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_root = output_dir / "resume_invalidated" / timestamp
    invalidated = []
    invalidated_result_count = 0
    for coordinate in expected_coordinates():
        group = grouped[coordinate]
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
            _prepare_archive_destination(destination, output_dir=output_dir)
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
        _prepare_archive_destination(destination, output_dir=output_dir)
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
        # This branch is reachable only if a prior history was deliberately
        # retained; normally it was moved with the other campaign artifacts.
        history = json.loads(history_path.read_text(encoding="utf-8"))
    history.append(record)
    hardened._atomic_write_json(history_path, history)
    return record


def _stable_file_artifact(path: Path, output_dir: Path) -> dict[str, Any]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"campaign artifact is not a regular file: {path}")
    digest = hardened._sha256_file(path)
    after = path.lstat()
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError(f"campaign artifact changed while hashing: {path}")
    return {
        "path": str(path.relative_to(output_dir)),
        "sha256": digest,
        "size_bytes": int(after.st_size),
    }


def collect_result_artifacts(
    output_dir: Path, jobs: Iterable[Any]
) -> dict[str, dict[str, Any]]:
    expected = set()
    for job in jobs:
        path = _result_path(output_dir, job)
        _require_no_symlink_components(
            path, root=output_dir, field="expected completed screen result"
        )
        expected.add(str(path.relative_to(output_dir)))
    observed = _observed_regular_result_paths(output_dir)
    if set(observed) != expected or len(observed) != EXPECTED_JOBS:
        raise RuntimeError(
            f"completed result grid mismatch: expected {EXPECTED_JOBS}, found {len(observed)}"
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
        raise RuntimeError("screen child metadata contains duplicate rows")
    comparisons = 0
    for key, row in index.items():
        dataset, repeat, fold, arm, child = key
        if (
            arm == "baseline"
            or ARM_SPECS[arm]["representation"] != "native"
        ):
            continue
        baseline = index.get((dataset, repeat, fold, "baseline", child))
        if baseline is None or row["representation"] != baseline["representation"]:
            raise RuntimeError(
                "paired native arms do not share identical child preprocessing"
            )
        comparisons += 1
    if comparisons != EXPECTED_NATIVE_REPRESENTATION_PAIRS:
        raise RuntimeError(
            "native child preprocessing comparison count does not match"
        )
    return comparisons


def validate_completed_results(
    output_dir: Path, artifacts: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    outer_rows = []
    child_rows = []
    seen = set()
    resources = set()
    stop_reasons = Counter()
    for relative in sorted(artifacts):
        path = output_dir / relative
        outer, children = parse_result_record(_decode_result_pickle(path), source=relative)
        key = (outer["dataset"], outer["repeat"], outer["fold"], outer["arm"])
        _validate_result_source_binding(
            relative,
            dataset=outer["dataset"],
            repeat=outer["repeat"],
            fold=outer["fold"],
            arm=outer["arm"],
        )
        if key in seen:
            raise RuntimeError(f"duplicate completed screen result: {key}")
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
    if seen != expected_grid():
        raise RuntimeError("completed screen result grid is missing or unexpected")
    if len(child_rows) != EXPECTED_CHILD_FITS:
        raise RuntimeError(
            f"expected {EXPECTED_CHILD_FITS} child fits, got {len(child_rows)}"
        )
    if len(resources) != 1:
        raise RuntimeError("screen results do not share one resource allocation")
    resource = next(iter(resources))
    if stop_reasons.get("time_limit", 0):
        raise RuntimeError("screen contains a fitted child with a wall-clock stop")
    native_representation_pairs = validate_native_representation_pairs(
        child_rows
    )
    outer_rows.sort(key=lambda row: (row["task_id"], row["repeat"], row["arm"]))
    child_rows.sort(
        key=lambda row: (
            row["task_id"],
            row["repeat"],
            row["arm"],
            row["child_fold"],
        )
    )
    validation = {
        "result_count": len(outer_rows),
        "child_fit_count": len(child_rows),
        "paired_comparison_count": EXPECTED_PAIRED_COMPARISONS,
        "native_representation_pair_count": native_representation_pairs,
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
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


def _validate_followon_warmup_history(
    value: Any,
    *,
    expected_thread_count: int,
    expected_latest_pid: int | None = None,
) -> None:
    """Prove every timed follow-on mode/lane was warmed in this process."""
    expected_thread_count = hardened._exact_int(
        expected_thread_count, "expected warmup thread count"
    )
    if expected_thread_count < 1:
        raise RuntimeError("expected warmup thread count must be positive")
    if not isinstance(value, list) or not value:
        raise RuntimeError("warmup history must contain at least one record")
    if expected_latest_pid is not None:
        expected_latest_pid = hardened._exact_int(
            expected_latest_pid, "expected latest warmup pid"
        )
        if expected_latest_pid <= 0:
            raise RuntimeError("expected latest warmup pid must be positive")
    expected_names = [spec["name"] for spec in WARMUP_STAGE_SPECS]
    stage_fields = {
        "name",
        "input_kind",
        "categorical_features",
        "config",
        "train_rows",
        "validation_rows",
        "fit_seconds",
        "iterations_fitted",
        "tree_depths",
        "requested_tree_mode",
        "resolved_tree_mode",
        "selected_lane",
        "linear_residual_active",
        "resolved_learning_rate",
        "resolved_ordered_boosting",
        "resolved_thread_count",
        "resolved_target_encoding_mode",
        "resolved_include_cat_codes",
        "resolved_ts_permutations",
        "encoder_modes",
        "encoder_ts_permutations",
        "flat_ensemble_type",
        "flat_prediction_router_selected",
        "prediction_parallel_min_rows",
        "prediction_batches",
    }
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
        if hardened._exact_int(raw_record["pid"], "warmup pid") <= 0:
            raise RuntimeError("warmup pid must be positive")
        warmup = raw_record["warmup"]
        if not isinstance(warmup, Mapping) or set(warmup) != {
            "schema_version",
            "kind",
            "clock",
            "duration_seconds",
            "thread_count",
            "stage_count",
            "counts",
            "stages",
        }:
            raise RuntimeError("follow-on warmup metadata fields are incomplete")
        if (
            warmup["schema_version"] != WARMUP_SCHEMA_VERSION
            or warmup["kind"] != WARMUP_KIND
            or warmup["clock"] != "time.monotonic_ns"
        ):
            raise RuntimeError("follow-on warmup schema, kind, or clock does not match")
        hardened._nonnegative_finite(
            warmup["duration_seconds"], "follow-on warmup duration"
        )
        if (
            hardened._exact_int(warmup["thread_count"], "warmup thread_count")
            != expected_thread_count
            or hardened._exact_int(warmup["stage_count"], "warmup stage count")
            != len(WARMUP_STAGE_SPECS)
            or warmup["counts"] != EXPECTED_WARMUP_COUNTS
        ):
            raise RuntimeError("follow-on warmup counts or resources do not match")
        stages = warmup["stages"]
        if not isinstance(stages, list) or [
            stage.get("name") if isinstance(stage, Mapping) else None
            for stage in stages
        ] != expected_names:
            raise RuntimeError("follow-on warmup stages do not match the frozen order")
        for spec, stage in zip(WARMUP_STAGE_SPECS, stages):
            if not isinstance(stage, Mapping) or set(stage) != stage_fields:
                raise RuntimeError("follow-on warmup stage fields are incomplete")
            input_kind = spec["input_kind"]
            expected_cats = [] if input_kind == "numeric" else [12]
            expected_mode = spec["tree_mode"]
            expected_lane = (
                "linear_residual" if spec["linear_residual"] else "boosting"
            )
            expected_encoding = (
                "kfold" if expected_mode in {"lightgbm", "hybrid"} else "ordered"
            )
            expected_resolved_threads = (
                min(expected_thread_count, 2)
                if expected_mode in {"lightgbm", "hybrid"}
                else expected_thread_count
            )
            expected_config = {
                **WARMUP_BASE_CONFIG,
                "tree_mode": expected_mode,
                "linear_residual": spec["linear_residual"],
                "ts_permutations": spec["ts_permutations"],
                "thread_count": expected_thread_count,
            }
            if (
                stage["input_kind"] != input_kind
                or stage["categorical_features"] != expected_cats
                or stage["config"] != expected_config
                or hardened._exact_int(stage["train_rows"], "warmup train rows")
                != 2_048
                or hardened._exact_int(
                    stage["validation_rows"], "warmup validation rows"
                )
                != 512
                or hardened._exact_int(
                    stage["iterations_fitted"], "warmup iterations"
                )
                != 5
                or stage["requested_tree_mode"] != expected_mode
                or stage["resolved_tree_mode"] != expected_mode
                or stage["selected_lane"] != expected_lane
                or stage["linear_residual_active"] is not spec["linear_residual"]
                or float(stage["resolved_learning_rate"]) != 0.1
                or stage["resolved_ordered_boosting"] is not False
                or hardened._exact_int(
                    stage["resolved_thread_count"], "warmup resolved threads"
                )
                != expected_resolved_threads
                or stage["resolved_target_encoding_mode"] != expected_encoding
                or stage["resolved_include_cat_codes"]
                is not (expected_mode in {"lightgbm", "hybrid"})
                or hardened._exact_int(
                    stage["resolved_ts_permutations"], "warmup TS permutations"
                )
                != spec["ts_permutations"]
            ):
                raise RuntimeError(f"follow-on warmup stage mismatch: {spec['name']}")
            hardened._nonnegative_finite(stage["fit_seconds"], "warmup fit duration")
            depths = stage["tree_depths"]
            if (
                not isinstance(depths, list)
                or len(depths) != 5
                or any(
                    hardened._exact_int(depth, "warmup tree depth") < 0
                    for depth in depths
                )
            ):
                raise RuntimeError("follow-on warmup tree depths are incomplete")
            expected_encoder_modes = [] if input_kind == "numeric" else [expected_encoding]
            expected_encoder_ts = (
                [] if input_kind == "numeric" else [spec["ts_permutations"]]
            )
            if (
                stage["encoder_modes"] != expected_encoder_modes
                or stage["encoder_ts_permutations"] != expected_encoder_ts
            ):
                raise RuntimeError("follow-on categorical warmup path does not match")
            threshold = hardened._exact_int(
                stage["prediction_parallel_min_rows"], "warmup prediction threshold"
            )
            if threshold != 8_192:
                raise RuntimeError("follow-on warmup prediction threshold changed")
            flat_selected = stage["flat_prediction_router_selected"]
            if not isinstance(flat_selected, bool):
                raise RuntimeError("warmup flat-router selection must be boolean")
            expected_flat_type = (
                "FlatObliviousEnsemble"
                if expected_mode == "catboost"
                else "FlatNonObliviousEnsemble"
            )
            expected_router = expected_mode == "catboost" and expected_thread_count > 1
            if (
                stage["flat_ensemble_type"] != expected_flat_type
                or flat_selected is not expected_router
            ):
                raise RuntimeError("follow-on warmup predictor route does not match")
            batches = stage["prediction_batches"]
            if not isinstance(batches, list) or len(batches) != 2:
                raise RuntimeError("follow-on warmup prediction batches are incomplete")
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
                rows = threshold - 1 if batch_index == 0 else threshold
                name = "serial_subthreshold" if batch_index == 0 else "parallel_at_threshold"
                route = "tree_loop"
                if expected_router:
                    route = "flat_serial" if batch_index == 0 else "flat_parallel"
                width = 12 if input_kind == "numeric" else 13
                if (
                    batch["name"] != name
                    or batch["route"] != route
                    or batch["input_shape"] != [rows, width]
                    or batch["prediction_shape"] != [rows]
                ):
                    raise RuntimeError("follow-on warmup prediction route is incomplete")
                hardened._nonnegative_finite(
                    batch["predict_seconds"], "warmup prediction duration"
                )
                hardened._validate_sha256(
                    batch["prediction_sha256"], "warmup prediction fingerprint"
                )
    if expected_latest_pid is not None and hardened._exact_int(
        value[-1]["pid"], "latest warmup pid"
    ) != expected_latest_pid:
        raise RuntimeError("latest warmup record was not produced by this run")


def write_completion_attestation(
    output_dir: Path,
    *,
    manifest: Mapping[str, Any],
    jobs: Iterable[Any],
    result_count: int,
) -> dict[str, Any]:
    jobs = list(jobs)
    if result_count != EXPECTED_JOBS:
        raise RuntimeError(
            f"expected {EXPECTED_JOBS} completed results, got {result_count}"
        )
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
        validator=lambda value: _validate_followon_warmup_history(
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
        "warmup_thread_count": child_cpus,
        "warmup_stage_count": len(WARMUP_STAGE_SPECS),
        "warmup_expected_counts": EXPECTED_WARMUP_COUNTS,
        "protocol_sha256": manifest["protocol_sha256"],
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_SECONDS)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume only a trusted cache directory created by this runner; "
            "cached result validation unpickles runner-owned artifacts"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
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

    from tabarena.contexts import TabArenaContext
    from tabarena.utils.config_utils import ConfigGenerator

    try:
        from benchmarks.tabarena_screen_adapters import (
            SafeOneHotDarkoFitModel,
            SafeOrdinalDarkoFitModel,
            ScreenNativeDarkoFitModel,
        )
    except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
        from tabarena_screen_adapters import (
            SafeOneHotDarkoFitModel,
            SafeOrdinalDarkoFitModel,
            ScreenNativeDarkoFitModel,
        )

    model_classes = {
        cls.__name__: cls
        for cls in (
            ScreenNativeDarkoFitModel,
            SafeOrdinalDarkoFitModel,
            SafeOneHotDarkoFitModel,
        )
    }
    context = TabArenaContext()
    experiments = build_experiments(
        model_classes=model_classes,
        config_generator_cls=ConfigGenerator,
        time_limit=args.time_limit,
    )
    jobs = build_screen_jobs(context, experiments)
    ordering = ordering_balance(jobs)
    child_cpus = resolve_and_pin_child_cpu_allocation(jobs)
    print(
        f"built {len(jobs)} isolated screen jobs "
        f"({EXPECTED_CONTROL_JOBS} shared controls, "
        f"{EXPECTED_CANDIDATE_JOBS} candidate comparisons, "
        f"{EXPECTED_CHILD_FITS} child fits); pinned child CPUs={child_cpus}"
    )
    print("ordering balance " + json.dumps(ordering, sort_keys=True))
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
        new_result_prefix="[DarkoFit isolated follow-on screen] ",
        debug_mode=True,
    )
    write_completion_attestation(
        output_dir,
        manifest=manifest,
        jobs=jobs,
        result_count=len(results),
    )
    print(f"FOLLOWON_SCREEN_COMPLETE {len(results)} {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
