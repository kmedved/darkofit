"""Run the frozen two-process B10/A10 TabArena accuracy shootout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import shutil
import stat
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _load_private_screen_runner():
    """Load the reusable runner without mutating its process-global policy."""
    module_name = "_darkofit_accuracy_shootout_screen_runner"
    path = Path(__file__).with_name("run_tabarena_regression_followon_screen.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load reusable screen runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


screen = _load_private_screen_runner()
# The reused ChimeraBoost comparison manifest predates the package rename and
# records this legacy warmup toggle.  Keep it in the private runtime lock so
# the new campaign can prove the complete environment is unchanged.
if "CHIMERABOOST_WARMUP" not in screen.RUNTIME_ENVIRONMENT_KEYS:
    screen.RUNTIME_ENVIRONMENT_KEYS = (
        *screen.RUNTIME_ENVIRONMENT_KEYS,
        "CHIMERABOOST_WARMUP",
    )
for _distribution in (
    "autogluon.features",
    "graphviz",
    "catboost",
    "chimeraboost",
):
    if _distribution not in screen.PACKAGE_DISTRIBUTIONS:
        screen.PACKAGE_DISTRIBUTIONS = (
            *screen.PACKAGE_DISTRIBUTIONS,
            _distribution,
        )


TASKS = dict(screen.TASKS)
SHOOTOUT_SPLITS = ((0, 0), (1, 1), (2, 2))
EXPECTED_CHILD_CPUS = 18
WORKER_COUNT = 2
TIME_LIMIT_SECONDS = 3_600.0
MAX_PREFLIGHT_RECIPROCAL_ASYMMETRY = 1.5
SWAP_POLICY_STRICT = "strict"
SWAP_POLICY_QUALITY_ONLY_SWAP_IN = "quality_only_swap_in"
SWAP_POLICIES = (SWAP_POLICY_STRICT, SWAP_POLICY_QUALITY_ONLY_SWAP_IN)
DEFAULT_SWAP_POLICY = SWAP_POLICY_STRICT

B10_CONFIG: dict[str, Any] = {
    "iterations": 10_000,
    "tree_mode": "catboost",
    "l2_leaf_reg": 3.0,
    "max_bins": 128,
    "learning_rate": 0.1,
    "ts_permutations": 1,
    "linear_residual": False,
    "early_stopping": True,
    "use_best_model": True,
}
A10_CONFIG = {**B10_CONFIG, "tree_mode": "auto"}
PUBLIC_TO_INTERNAL_ARM = {"B10": "baseline", "A10": "auto"}
INTERNAL_TO_PUBLIC_ARM = {
    internal: public for public, internal in PUBLIC_TO_INTERNAL_ARM.items()
}

EXPECTED_COORDINATES = len(TASKS) * len(SHOOTOUT_SPLITS)
EXPECTED_JOBS = EXPECTED_COORDINATES * len(PUBLIC_TO_INTERNAL_ARM)
EXPECTED_CHILD_FITS = EXPECTED_JOBS * 8
EXPECTED_PAIRED_CHILDREN = EXPECTED_COORDINATES * 8
EXPECTED_WAVES = EXPECTED_COORDINATES

CAMPAIGN_KIND = "darkofit_tabarena_regression_accuracy_shootout"
COMPLETION_KIND = CAMPAIGN_KIND + "_completion"
PAYLOAD_KIND = CAMPAIGN_KIND + "_analysis_payload"
CONCURRENCY_HISTORY_FILENAME = "concurrency_history.json"
PREFLIGHT_REPORT_FILENAME = "preflight_report.json"
INVALID_ATTEMPT_FILENAME = "invalid_attempt.json"
WAVE_SCHEDULE_FILENAME = "wave_schedule.json"
OWNER_LOCK_FILENAME = ".owner_session.lock"
OWNER_STATE_FILENAME = "owner_sessions.json"
OWNER_STATE_KIND = CAMPAIGN_KIND + "_owner_sessions"
ANALYSIS_OUTPUT_FILENAMES = (
    "paired_splits.csv",
    "per_dataset.csv",
    "paired_children.csv",
    "summary.json",
    "report.md",
)
PREFLIGHT_COORDINATES = {
    "physiochemical_protein": (0, 0),
    "QSAR-TID-11": (2, 2),
}
REUSED_EVIDENCE = {
    "source_commit": "a1ff4b74510b5e314bb41c27b40544910741543d",
    "source_darkofit_subtree": "52278b0326419a45a72bdfd3afcfc13019087838",
    "chimeraboost_tag_commit": "9c9ea6e704a9fe2bfe6d6c284b22de73914be048",
    "catboost_version": "1.2.10",
    "artifacts": {
        "tabarena_regression_same_machine_primary_paired_splits.csv": "3e7bbe21e0ffe40771f2065dc252dbd4314550f8ab350f2fbed9641401b341b1",
        "tabarena_regression_same_machine_summary.json": "ca23618bdc3d9e0ab38557e7738c66e95827945ad34e3eb63005f253c92ccf01",
        "tabarena_regression_same_machine_completion_attestation.json": "213f462aa06103e97864ecd786b75e8fd8e11743c77f556262fa39bdb3e1b7d9",
        "tabarena_regression_same_machine_run_manifest.json": "2869acaaa4bcc8319d9ba03744a4a9ca8602ed349553a031c3d84ab537de72ee",
    },
}
REUSED_COMMON_PACKAGE_DISTRIBUTIONS = (
    "darkofit",
    "tabarena",
    "autogluon.common",
    "autogluon.core",
    "autogluon.features",
    "autogluon.tabular",
    "graphviz",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "numba",
    "llvmlite",
    "psutil",
)
REUSED_OPTIONAL_COMPARATOR_DISTRIBUTIONS = ("catboost", "chimeraboost")
DEFAULT_OUTPUT_DIR = Path(
    ".cache/tabarena-regression-accuracy-shootout-20260715"
)

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
    Path("benchmarks/run_tabarena_regression_accuracy_shootout.py"),
    Path("benchmarks/analyze_tabarena_regression_accuracy_shootout.py"),
    Path("benchmarks/tabarena_regression_accuracy_shootout_protocol.md"),
)


def _configure_reused_runner() -> None:
    """Specialize the hardened follow-on runner for the two shootout arms."""
    screen.SCREEN_SPLITS = SHOOTOUT_SPLITS
    screen.SCREEN_SPLIT_INDICES = tuple(
        f"r{repeat}f{fold}" for repeat, fold in SHOOTOUT_SPLITS
    )
    screen.BASELINE_CONFIG = dict(B10_CONFIG)
    screen.ARM_SPECS = {
        "baseline": {
            "config": dict(B10_CONFIG),
            "model_cls": "ScreenNativeDarkoFitModel",
            "representation": "native",
            "datasets": tuple(TASKS),
        },
        "auto": {
            "config": dict(A10_CONFIG),
            "model_cls": "ScreenNativeDarkoFitModel",
            "representation": "native",
            "datasets": tuple(TASKS),
        },
    }
    screen.CANDIDATE_ARMS = ("auto",)
    screen.EXPECTED_CONTROL_JOBS = EXPECTED_COORDINATES
    screen.EXPECTED_CANDIDATE_JOBS = EXPECTED_COORDINATES
    screen.EXPECTED_JOBS = EXPECTED_JOBS
    screen.EXPECTED_CHILD_FITS = EXPECTED_CHILD_FITS
    screen.EXPECTED_PAIRED_COMPARISONS = EXPECTED_COORDINATES
    screen.EXPECTED_NATIVE_REPRESENTATION_PAIRS = EXPECTED_PAIRED_CHILDREN
    screen.TIME_LIMIT_SECONDS = TIME_LIMIT_SECONDS
    screen.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    screen.CAMPAIGN_KIND = CAMPAIGN_KIND
    screen.COMPLETION_KIND = COMPLETION_KIND
    screen.PAYLOAD_KIND = PAYLOAD_KIND
    screen.SOURCE_FILES = SOURCE_FILES


_configure_reused_runner()


def expected_coordinates() -> list[tuple[str, int, int]]:
    return [
        (dataset, repeat, fold)
        for dataset in TASKS
        for repeat, fold in SHOOTOUT_SPLITS
    ]


def _job_key(
    coordinate: tuple[str, int, int], public_arm: str
) -> tuple[str, int, int, str]:
    if public_arm not in PUBLIC_TO_INTERNAL_ARM:
        raise RuntimeError(f"unknown shootout arm: {public_arm}")
    return (*coordinate, public_arm)


def _key_payload(key: tuple[str, int, int, str]) -> dict[str, Any]:
    dataset, repeat, fold, arm = key
    return {
        "dataset": dataset,
        "task_id": TASKS[dataset],
        "repeat": repeat,
        "fold": fold,
        "registered_fold": 3 * repeat + fold,
        "arm": arm,
        "internal_arm": PUBLIC_TO_INTERNAL_ARM[arm],
    }


def expected_wave_schedule() -> list[dict[str, Any]]:
    """Return 39 arm-balanced waves with a within-dataset derangement."""
    waves = []
    for dataset_index, dataset in enumerate(TASKS):
        coordinates = [
            (dataset, repeat, fold) for repeat, fold in SHOOTOUT_SPLITS
        ]
        for local_index, a_coordinate in enumerate(coordinates):
            b_coordinate = coordinates[(local_index + 1) % len(coordinates)]
            global_index = dataset_index * len(coordinates) + local_index
            a_slot = global_index % WORKER_COUNT
            b_slot = 1 - a_slot
            jobs = [
                {
                    "worker_slot": a_slot,
                    "key": _key_payload(_job_key(a_coordinate, "A10")),
                },
                {
                    "worker_slot": b_slot,
                    "key": _key_payload(_job_key(b_coordinate, "B10")),
                },
            ]
            waves.append(
                {
                    "wave_index": global_index,
                    "dataset_index": dataset_index,
                    "dataset": dataset,
                    "local_wave_index": local_index,
                    "jobs": sorted(jobs, key=lambda item: item["worker_slot"]),
                }
            )
    validate_wave_schedule(waves)
    return waves


def validate_wave_schedule(waves: Sequence[Mapping[str, Any]]) -> None:
    if len(waves) != EXPECTED_WAVES:
        raise RuntimeError(
            f"expected {EXPECTED_WAVES} waves, observed {len(waves)}"
        )
    observed = set()
    slot_counts = {"A10": [0, 0], "B10": [0, 0]}
    for expected_index, wave in enumerate(waves):
        if wave.get("wave_index") != expected_index:
            raise RuntimeError("wave indices are not contiguous")
        jobs = wave.get("jobs")
        if not isinstance(jobs, list) or len(jobs) != WORKER_COUNT:
            raise RuntimeError("every production wave must contain two jobs")
        slots = set()
        arms = set()
        coordinates = set()
        for raw in jobs:
            if not isinstance(raw, Mapping) or set(raw) != {
                "worker_slot",
                "key",
            }:
                raise RuntimeError("wave job metadata is incomplete")
            slot = raw["worker_slot"]
            key = raw["key"]
            if slot not in range(WORKER_COUNT) or slot in slots:
                raise RuntimeError("wave worker slots are invalid")
            if not isinstance(key, Mapping):
                raise RuntimeError("wave key must be a mapping")
            dataset = key.get("dataset")
            repeat = key.get("repeat")
            fold = key.get("fold")
            arm = key.get("arm")
            public_key = (dataset, repeat, fold, arm)
            if (
                dataset not in TASKS
                or (repeat, fold) not in SHOOTOUT_SPLITS
                or arm not in PUBLIC_TO_INTERNAL_ARM
                or public_key in observed
            ):
                raise RuntimeError("wave contains an invalid or duplicate job")
            if key != _key_payload(public_key):
                raise RuntimeError("wave key metadata is not canonical")
            slots.add(slot)
            arms.add(arm)
            coordinates.add((dataset, repeat, fold))
            observed.add(public_key)
            slot_counts[arm][slot] += 1
        if slots != set(range(WORKER_COUNT)) or arms != set(
            PUBLIC_TO_INTERNAL_ARM
        ):
            raise RuntimeError("each wave must pair one A10 with one B10")
        if len(coordinates) != WORKER_COUNT:
            raise RuntimeError("wave arms must use different coordinates")
        if {coordinate[0] for coordinate in coordinates} != {wave["dataset"]}:
            raise RuntimeError("wave partners must belong to one dataset")
    expected = {
        _job_key(coordinate, arm)
        for coordinate in expected_coordinates()
        for arm in PUBLIC_TO_INTERNAL_ARM
    }
    if observed != expected:
        raise RuntimeError("wave schedule does not cover the exact shootout grid")
    if any(abs(counts[0] - counts[1]) > 1 for counts in slot_counts.values()):
        raise RuntimeError("arm exposure is not balanced across worker slots")


def wave_schedule_sha256() -> str:
    return hashlib.sha256(
        screen.hardened._canonical_json(expected_wave_schedule())
    ).hexdigest()


def frozen_protocol() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": CAMPAIGN_KIND,
        "tasks": dict(TASKS),
        "coordinates": [
            {"dataset": dataset, "repeat": repeat, "fold": fold}
            for dataset, repeat, fold in expected_coordinates()
        ],
        "arms": {
            "B10": dict(B10_CONFIG),
            "A10": dict(A10_CONFIG),
        },
        "internal_arm_names": dict(PUBLIC_TO_INTERNAL_ARM),
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
            "intentional_max_runnable_threads": (
                WORKER_COUNT * EXPECTED_CHILD_CPUS
            ),
            "barrier_between_waves": True,
            "one_a10_and_one_b10_per_wave": True,
            "within_dataset_partner_derangement": "B10 coordinate j+1 mod 3",
            "worker_slot_policy": "A10 slot alternates by global wave parity",
            "private_worker_cwd": True,
            "parent_only_campaign_metadata_writes": True,
            "safe_zero_start_resume": True,
            "failure_policy": (
                "stop releasing waves, drain or terminate the active partner, "
                "and emit no completion attestation"
            ),
            "swap_policies": {
                "default": SWAP_POLICY_STRICT,
                "allowed": list(SWAP_POLICIES),
                "strict": "zero host swap-in and swap-out for the full worker session",
                "quality_only_swap_in": (
                    "host swap-in is allowed and recorded anywhere; host swap-out "
                    "is forbidden for the full worker session and every dispatch; "
                    "timing and memory-performance evidence is inadmissible by policy"
                ),
                "measured_phase_uses_shared_lifecycle_samples": True,
            },
        },
        "wave_schedule_sha256": wave_schedule_sha256(),
        "wave_schedule": expected_wave_schedule(),
        "timing_interpretation": {
            "quality_is_primary": True,
            "per_arm_wall_time_is_contention_exposed": True,
            "causal_arm_timing_claim_allowed": False,
            "isolated_timing_rerun_required_if_freeze_depends_on_resources": True,
            "by_swap_policy": {
                SWAP_POLICY_STRICT: {
                    "campaign_throughput_status": "descriptive",
                    "preflight_performance_comparison_criteria": [
                        "throughput_speedup_at_least_1_10",
                        "reciprocal_arm_slot_symmetry",
                    ],
                },
                SWAP_POLICY_QUALITY_ONLY_SWAP_IN: {
                    "campaign_throughput_status": (
                        "inadmissible_raw_operational_audit_only"
                    ),
                    "preflight_performance_comparison_criteria": [],
                },
            },
        },
        "warmup": {
            "process_local": True,
            "worker_count": WORKER_COUNT,
            "thread_count_per_worker": EXPECTED_CHILD_CPUS,
            "workers_warmed_serially_before_wave_zero": True,
        },
        "preflight": {
            "namespace_is_non_reusable": True,
            "untimed_data_prime_both_tasks_per_worker": True,
            "datasets": ["physiochemical_protein", "QSAR-TID-11"],
            "coordinates": {
                dataset: {"repeat": repeat, "fold": fold}
                for dataset, (repeat, fold) in PREFLIGHT_COORDINATES.items()
            },
            "maximum_start_skew_seconds": 1.0,
            "maximum_concurrent_job_seconds": 1_800.0,
            "performance_comparison_limits_by_swap_policy": {
                SWAP_POLICY_STRICT: {
                    "minimum_throughput_speedup": 1.10,
                    "maximum_reciprocal_asymmetry_ratio": (
                        MAX_PREFLIGHT_RECIPROCAL_ASYMMETRY
                    ),
                },
                SWAP_POLICY_QUALITY_ONLY_SWAP_IN: {
                    "minimum_throughput_speedup": None,
                    "maximum_reciprocal_asymmetry_ratio": None,
                },
            },
            "require_exact_quality_and_structure_fingerprints": True,
            "require_zero_deadlines_time_limits_restarts_or_oom": True,
            "require_policy_valid_swap_during_every_dispatch_and_gap": True,
            "require_zero_full_session_swap_out": True,
            "require_os_high_water_rss": True,
        },
    }


screen.frozen_protocol = frozen_protocol


def _public_job_key(job: Any) -> tuple[str, int, int, str]:
    dataset, repeat, fold = screen._job_coordinate(job)
    internal_arm = screen._job_arm(job)
    return dataset, repeat, fold, INTERNAL_TO_PUBLIC_ARM[internal_arm]


def _load_model_classes() -> dict[str, type]:
    try:
        from benchmarks.tabarena_screen_adapters import ScreenNativeDarkoFitModel
    except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
        from tabarena_screen_adapters import ScreenNativeDarkoFitModel

    return {ScreenNativeDarkoFitModel.__name__: ScreenNativeDarkoFitModel}


def build_runtime_jobs(time_limit: float) -> tuple[Any, list[Any], int]:
    from tabarena.contexts import TabArenaContext
    from tabarena.utils.config_utils import ConfigGenerator

    context = TabArenaContext()
    experiments = screen.build_experiments(
        model_classes=_load_model_classes(),
        config_generator_cls=ConfigGenerator,
        time_limit=time_limit,
    )
    jobs = screen.build_screen_jobs(context, experiments)
    child_cpus = screen.resolve_and_pin_child_cpu_allocation(jobs)
    if child_cpus != EXPECTED_CHILD_CPUS:
        raise RuntimeError(
            f"shootout requires {EXPECTED_CHILD_CPUS} child CPUs, got {child_cpus}"
        )
    job_keys = [_public_job_key(job) for job in jobs]
    expected = {
        _job_key(coordinate, arm)
        for coordinate in expected_coordinates()
        for arm in PUBLIC_TO_INTERNAL_ARM
    }
    if set(job_keys) != expected or len(job_keys) != EXPECTED_JOBS:
        raise RuntimeError("built jobs do not match the frozen shootout grid")
    return context, jobs, child_cpus


def _read_stable_regular_file(path: Path, field: str) -> bytes:
    """Read one regular file while rejecting links and concurrent replacement."""
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise RuntimeError(f"{field} must be a regular file: {path}")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"could not read {field}: {path}") from exc
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(payload) != after.st_size:
        raise RuntimeError(f"{field} changed while it was read: {path}")
    return payload


def _as_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field} must be a mapping")
    return value


def verify_reused_evidence(repository: Path | None = None) -> dict[str, Any]:
    """Revalidate the complete P/M/C reuse contract before any measured fit."""
    repository = (repository or Path(__file__).resolve().parents[1]).resolve()
    current_subtree = subprocess.run(
        ["git", "rev-parse", "HEAD:darkofit"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_subtree = subprocess.run(
        ["git", "rev-parse", f"{REUSED_EVIDENCE['source_commit']}:darkofit"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_subtree = REUSED_EVIDENCE["source_darkofit_subtree"]
    if current_subtree != expected_subtree or source_subtree != expected_subtree:
        raise RuntimeError("DarkoFit package subtree no longer matches reused evidence")
    artifact_payloads = {}
    for filename, expected_sha256 in REUSED_EVIDENCE["artifacts"].items():
        path = repository / "benchmarks" / filename
        payload = _read_stable_regular_file(path, f"reused evidence {filename}")
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise RuntimeError(f"reused evidence hash mismatch: {filename}")
        artifact_payloads[filename] = payload

    try:
        source_manifest = json.loads(
            artifact_payloads[
                "tabarena_regression_same_machine_run_manifest.json"
            ]
        )
        source_attestation = json.loads(
            artifact_payloads[
                "tabarena_regression_same_machine_completion_attestation.json"
            ]
        )
        source_summary = json.loads(
            artifact_payloads["tabarena_regression_same_machine_summary.json"]
        )
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("reused evidence metadata is not valid JSON") from exc
    source_protocol = _as_mapping(
        source_manifest.get("protocol"), "reused source protocol"
    )
    primary = _as_mapping(
        _as_mapping(
            source_protocol.get("lanes"), "reused source lanes"
        ).get("primary"),
        "reused source primary lane",
    )
    expected_primary_coordinates = [
        {"dataset": dataset, "repeat": repeat, "fold": fold}
        for dataset, repeat, fold in expected_coordinates()
    ]
    if (
        source_manifest.get("resolved_child_num_cpus") != EXPECTED_CHILD_CPUS
        or source_protocol.get("bag_folds") != 8
        or source_protocol.get("bag_sets") != 1
        or source_protocol.get("fold_fitting_strategy") != "sequential_local"
        or source_protocol.get("chimera_source")
        != {
            "exact_git_commit": REUSED_EVIDENCE["chimeraboost_tag_commit"],
            "hidden_import_warmup": "disabled",
            "version": "0.14.1",
        }
        or source_protocol.get("catboost_source")
        != {"version": REUSED_EVIDENCE["catboost_version"]}
        or primary.get("coordinates") != expected_primary_coordinates
        or primary.get("expected_jobs") != 117
    ):
        raise RuntimeError("reused source protocol does not match the shootout")

    source_validation = _as_mapping(
        source_attestation.get("validation"), "reused completion validation"
    )
    if (
        source_attestation.get("manifest_sha256")
        != REUSED_EVIDENCE["artifacts"][
            "tabarena_regression_same_machine_run_manifest.json"
        ]
        or source_attestation.get("result_count") != 135
        or source_attestation.get("expected_child_fits") != 1_080
        or source_validation.get("failure_count") != 0
        or source_validation.get("imputation_count") != 0
        or source_validation.get("known_deadline_hit_count") != 0
        or source_validation.get("known_time_limit_stop_count") != 0
        or source_validation.get("resource_allocation")
        != {
            "num_cpus": 18,
            "num_gpus": 0,
            "num_cpus_child": 18,
            "num_gpus_child": 0,
        }
    ):
        raise RuntimeError("reused source completion is not admissible")

    summary_provenance = _as_mapping(
        source_summary.get("provenance"), "reused summary provenance"
    )
    if (
        summary_provenance.get("manifest_sha256")
        != REUSED_EVIDENCE["artifacts"][
            "tabarena_regression_same_machine_run_manifest.json"
        ]
        or summary_provenance.get("attestation_sha256")
        != REUSED_EVIDENCE["artifacts"][
            "tabarena_regression_same_machine_completion_attestation.json"
        ]
        or summary_provenance.get("chimeraboost_git_head")
        != REUSED_EVIDENCE["chimeraboost_tag_commit"]
        or summary_provenance.get("catboost_version")
        != REUSED_EVIDENCE["catboost_version"]
        or source_summary.get("counts", {}).get("primary_coordinates") != 39
    ):
        raise RuntimeError("reused source summary does not bind its inputs")

    old_runtime = _as_mapping(
        source_manifest.get("runtime"), "reused source runtime"
    )
    current_runtime = screen.collect_runtime_provenance()
    for field in ("python_version", "platform", "machine", "environment", "hardware"):
        if current_runtime.get(field) != old_runtime.get(field):
            raise RuntimeError(f"runtime differs from reused evidence: {field}")
    old_packages = _as_mapping(
        old_runtime.get("packages"), "reused source packages"
    )
    current_packages = _as_mapping(
        current_runtime.get("packages"), "current packages"
    )
    expected_source_packages = set(REUSED_COMMON_PACKAGE_DISTRIBUTIONS) | set(
        REUSED_OPTIONAL_COMPARATOR_DISTRIBUTIONS
    )
    if (
        set(old_packages) != expected_source_packages
        or set(current_packages) != expected_source_packages
        or any(
            old_packages.get(name) != current_packages.get(name)
            for name in REUSED_COMMON_PACKAGE_DISTRIBUTIONS
        )
        or any(
            current_packages.get(name) is not None
            and current_packages.get(name) != old_packages.get(name)
            for name in REUSED_OPTIONAL_COMPARATOR_DISTRIBUTIONS
        )
    ):
        raise RuntimeError("dependency lock differs from reused evidence")

    source_provenance = _as_mapping(
        source_manifest.get("source"), "reused source provenance"
    )
    source_tabarena = _as_mapping(
        source_provenance.get("tabarena"), "reused TabArena provenance"
    )
    current_tabarena = screen.hardened.collect_git_dependency_provenance(
        "tabarena", output_dir=None
    )
    if current_tabarena != source_tabarena:
        raise RuntimeError("TabArena revision differs from reused evidence")

    source_files = _as_mapping(
        source_provenance.get("files"),
        "reused source file hashes",
    )
    for relative in (
        "benchmarks/tabarena_adapter.py",
        "benchmarks/tabarena_screen_adapters.py",
    ):
        recorded = _as_mapping(
            source_files.get(relative), f"reused source hash for {relative}"
        )
        payload = _read_stable_regular_file(
            repository / relative, f"current adapter {relative}"
        )
        if recorded.get("sha256") != hashlib.sha256(payload).hexdigest():
            raise RuntimeError(f"base adapter changed since reused evidence: {relative}")
    return json.loads(json.dumps(REUSED_EVIDENCE, sort_keys=True))


def _job_lookup(jobs: Sequence[Any]) -> dict[tuple[str, int, int, str], Any]:
    lookup = {_public_job_key(job): job for job in jobs}
    expected = {
        _job_key(coordinate, arm)
        for coordinate in expected_coordinates()
        for arm in PUBLIC_TO_INTERNAL_ARM
    }
    if len(jobs) != EXPECTED_JOBS or set(lookup) != expected:
        raise RuntimeError("runtime job lookup is incomplete")
    return lookup


def _key_tuple(value: Mapping[str, Any]) -> tuple[str, int, int, str]:
    key = (
        str(value.get("dataset")),
        screen.hardened._exact_int(value.get("repeat"), "job repeat"),
        screen.hardened._exact_int(value.get("fold"), "job fold"),
        str(value.get("arm")),
    )
    if key not in {
        _job_key(coordinate, arm)
        for coordinate in expected_coordinates()
        for arm in PUBLIC_TO_INTERNAL_ARM
    }:
        raise RuntimeError(f"job key is outside the frozen grid: {key}")
    return key


def _behavior_value(value: Any) -> Any:
    """Canonicalize behavior while excluding operational observations."""
    excluded = {
        "fit_seconds",
        "wall_clock_elapsed_seconds",
        "wall_clock_limit_seconds",
        "wall_clock_effective_seconds",
        "wall_clock_safety_margin_seconds",
        "source",
    }
    if isinstance(value, Mapping):
        return {
            str(key): _behavior_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in excluded
            and not str(key).endswith("_seconds")
            and "memory" not in str(key)
        }
    if isinstance(value, list):
        return [_behavior_value(item) for item in value]
    if isinstance(value, tuple):
        return [_behavior_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError("behavior fingerprint contains a non-finite float")
        return {"float_hex": value.hex()}
    return value


def behavior_fingerprint(path: Path) -> tuple[str, int, bool, int]:
    outer, children = screen.parse_result_record(
        screen._decode_result_pickle(path), source=str(path)
    )
    payload = {
        "outer": {
            key: value
            for key, value in outer.items()
            if key not in {"train_time_s", "infer_time_s", "peak_memory_bytes", "source"}
        },
        "children": children,
    }
    normalized = _behavior_value(payload)
    digest = hashlib.sha256(screen.hardened._canonical_json(normalized)).hexdigest()
    deadline_hit = any(
        child["deadline_hit"] is not False or child["stop_reason"] == "time_limit"
        for child in children
    )
    auto_candidate_fit_count = 0
    if outer["arm"] == "auto":
        for child in children:
            selection = child["tree_mode_selection"]
            auto_candidate_fit_count += sum(
                candidate.get("fit_status") == "fitted"
                and math.isfinite(float(candidate.get("validation_score")))
                for candidate in selection["candidates"]
            )
    return digest, len(children), deadline_hit, auto_candidate_fit_count


def _wait_until(release_ns: int) -> None:
    while True:
        remaining = (release_ns - time.monotonic_ns()) / 1e9
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.01))


def _self_peak_rss_bytes() -> int:
    import resource

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1_024


def _worker_main(
    slot: int,
    connection: Any,
    scratch_root: str,
    time_limit: float,
) -> None:
    """Own one TabArena context and execute commands from the parent."""
    try:
        scratch = Path(scratch_root).resolve()
        scratch.mkdir(parents=True, exist_ok=True)
        os.chdir(scratch)
        context, jobs, child_cpus = build_runtime_jobs(time_limit)
        lookup = _job_lookup(jobs)
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
                try:
                    from benchmarks.tabarena_followon_warmup import (
                        warmup_tabarena_followon_screen,
                    )
                except ModuleNotFoundError:
                    from tabarena_followon_warmup import (
                        warmup_tabarena_followon_screen,
                    )
                warmup = warmup_tabarena_followon_screen(thread_count=child_cpus)
                connection.send(
                    {
                        "type": "warmup",
                        "command_id": command_id,
                        "slot": slot,
                        "pid": os.getpid(),
                        "warmup": warmup,
                    }
                )
                continue
            if kind == "prime":
                keys = [_key_tuple(item) for item in command.get("keys", [])]
                if not keys:
                    raise RuntimeError("worker prime requires at least one frozen job key")
                prime_jobs = [lookup[key] for key in keys]
                # Front-load OpenML task/dataset/split materialization outside every
                # measured run.  This removes the otherwise systematic cold-cache
                # advantage of running all concurrent observations second.
                with context._cache_scope():
                    context.task_metadata_collection.subset_to_jobs(
                        prime_jobs
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
            job = lookup[key]
            result_root = Path(command["result_root"]).resolve()
            release_ns = screen.hardened._exact_int(
                command["release_monotonic_ns"], "barrier release"
            )
            _wait_until(release_ns)
            started_ns = time.monotonic_ns()
            cpu_started = time.process_time()
            results = context.run_jobs(
                [job],
                expname=str(result_root / "experiments"),
                new_result_prefix="[DarkoFit accuracy shootout] ",
                debug_mode=True,
                register=False,
            )
            ended_ns = time.monotonic_ns()
            path = screen._result_path(result_root, job)
            digest, child_count, deadline_hit, auto_count = behavior_fingerprint(path)
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
                    "auto_candidate_fit_count": auto_count,
                    "behavior_sha256": digest,
                    "result_sha256": screen.hardened._sha256_file(path),
                    "result_size_bytes": path.stat().st_size,
                    "process_peak_rss_bytes": _self_peak_rss_bytes(),
                    "barrier_release_monotonic_ns": release_ns,
                    "started_monotonic_ns": started_ns,
                    "ended_monotonic_ns": ended_ns,
                    "elapsed_seconds": (ended_ns - started_ns) / 1e9,
                    "cpu_time_seconds": time.process_time() - cpu_started,
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


def _start_workers(root: Path, *, worker_count: int = WORKER_COUNT):
    context = mp.get_context("spawn")
    workers = []
    # These refer only to the pair currently being constructed.  Once a
    # worker mapping is appended, _stop_workers owns its parent endpoint and
    # process; the child endpoint remains locally owned until close succeeds.
    pending_parent = None
    pending_child = None
    parent_is_worker_owned = False
    try:
        for slot in range(worker_count):
            pending_parent, pending_child = context.Pipe(duplex=True)
            parent_is_worker_owned = False
            scratch = root / f"worker-{slot}"
            process = context.Process(
                target=_worker_main,
                args=(slot, pending_child, str(scratch), TIME_LIMIT_SECONDS),
                name=f"darkofit-shootout-{slot}",
            )
            worker = {
                "slot": slot,
                "process": process,
                "connection": pending_parent,
            }
            # Register before start(): multiprocessing may spawn a child and
            # still raise while completing its parent-side bookkeeping.
            workers.append(worker)
            parent_is_worker_owned = True
            process.start()
            pending_child.close()
            pending_child = None
            pending_parent = None
            parent_is_worker_owned = False

        ready = []
        deadline = time.monotonic() + 300.0
        for worker in workers:
            remaining = max(0.0, deadline - time.monotonic())
            if not worker["connection"].poll(remaining):
                raise RuntimeError("worker readiness timed out")
            message = worker["connection"].recv()
            if (
                message.get("type") != "ready"
                or message.get("slot") != worker["slot"]
                or message.get("pid") != worker["process"].pid
                or message.get("start_method") != "spawn"
                or message.get("child_cpus") != EXPECTED_CHILD_CPUS
            ):
                raise RuntimeError(f"invalid worker readiness report: {message}")
            ready.append(message)
        return workers, ready
    except BaseException as startup_error:
        cleanup_errors = []
        # Process construction can fail after Pipe() succeeds but before a
        # worker owns either endpoint.  A child-end close can itself fail, so
        # retry it while unwinding and treat a second failure as unsafe.
        if pending_child is not None:
            try:
                pending_child.close()
            except BaseException as exc:
                cleanup_errors.append(f"child endpoint close failed: {exc}")
        if pending_parent is not None and not parent_is_worker_owned:
            try:
                pending_parent.close()
            except BaseException as exc:
                cleanup_errors.append(f"parent endpoint close failed: {exc}")
        try:
            _stop_workers(workers, force=True)
        except BaseException as exc:
            cleanup_errors.append(f"spawned worker cleanup failed: {exc}")
        if cleanup_errors:
            raise RuntimeError(
                "worker startup cleanup could not be confirmed: "
                + "; ".join(cleanup_errors)
            ) from startup_error
        # In particular, preserve KeyboardInterrupt instead of converting it
        # into a sequential-fallback preflight result.
        raise


def _stop_workers(workers: Sequence[Mapping[str, Any]], *, force: bool = False) -> None:
    errors = []
    stop_ids = {}
    if not force:
        for worker in workers:
            process = worker["process"]
            command_id = f"stop-{worker['slot']}-{time.monotonic_ns()}"
            try:
                alive = process.is_alive()
            except BaseException as exc:
                errors.append(
                    f"worker {worker['slot']} state check before shutdown failed: {exc}"
                )
                continue
            if not alive:
                errors.append(f"worker {worker['slot']} exited before shutdown")
                continue
            try:
                worker["connection"].send(
                    {"kind": "stop", "command_id": command_id}
                )
            except BaseException as exc:
                errors.append(f"worker {worker['slot']} stop send failed: {exc}")
                continue
            stop_ids[worker["slot"]] = command_id
        for worker in workers:
            if worker["slot"] not in stop_ids:
                continue
            try:
                if not worker["connection"].poll(30.0):
                    errors.append(f"worker {worker['slot']} stop acknowledgement timed out")
                    continue
                message = worker["connection"].recv()
                if message != {
                    "type": "stopped",
                    "command_id": stop_ids[worker["slot"]],
                    "slot": worker["slot"],
                }:
                    errors.append(
                        f"worker {worker['slot']} sent invalid stop acknowledgement"
                    )
            except BaseException as exc:
                errors.append(f"worker {worker['slot']} stop acknowledgement failed: {exc}")
    for worker in workers:
        slot = worker["slot"]
        process = worker["process"]
        required_termination = False
        required_kill = False
        # A forced shutdown skips the graceful wait because it is used only
        # while unwinding an interrupted or failed command.  Both paths then
        # use the same bounded terminate -> kill escalation and prove that the
        # process has actually exited before returning.  Intermediate method
        # failures are deliberately best-effort: a later kill plus confirmed
        # exit is still a safe teardown.
        if not force:
            try:
                process.join(timeout=30.0)
            except BaseException as exc:
                errors.append(f"worker {slot} graceful join failed: {exc}")
        try:
            needs_termination = process.is_alive()
        except BaseException:
            needs_termination = True
        if needs_termination:
            required_termination = True
            try:
                process.terminate()
            except BaseException:
                pass
            try:
                process.join(timeout=10.0)
            except BaseException:
                pass

        try:
            needs_kill = process.is_alive()
        except BaseException:
            needs_kill = True
        if needs_kill:
            required_kill = True
            try:
                process.kill()
            except BaseException:
                pass
            try:
                process.join(timeout=10.0)
            except BaseException:
                pass

        try:
            alive = process.is_alive()
            exitcode = process.exitcode
        except BaseException as exc:
            errors.append(f"worker {slot} exit could not be confirmed: {exc}")
        else:
            if alive:
                errors.append(
                    f"worker {slot} failed to quiesce after terminate/kill escalation"
                )
            elif exitcode is None:
                # A Process registered before start() is safe only when it is
                # provably unstarted.  A non-null pid with no exit status may
                # be a partially spawned process and must fail closed.
                try:
                    pid = process.pid
                except BaseException as exc:
                    errors.append(
                        f"worker {slot} unstarted state could not be confirmed: {exc}"
                    )
                else:
                    if pid is not None:
                        errors.append(
                            f"worker {slot} has pid {pid} without a confirmed exit"
                        )
            elif not force and exitcode != 0:
                errors.append(f"worker {slot} exited with code {exitcode}")

        if not force and required_termination:
            errors.append(f"worker {slot} required termination")
        if not force and required_kill:
            errors.append(f"worker {slot} required kill")
        try:
            worker["connection"].close()
        except BaseException as exc:
            errors.append(f"worker {slot} connection close failed: {exc}")
        else:
            try:
                connection_closed = worker["connection"].closed
            except AttributeError:
                # Third-party/fault-injection connection facades need only
                # honor close(); multiprocessing.Connection exposes .closed.
                connection_closed = True
            except BaseException as exc:
                errors.append(
                    f"worker {slot} connection state could not be confirmed: {exc}"
                )
            else:
                if not connection_closed:
                    errors.append(
                        f"worker {slot} connection remained open after close"
                    )
    if errors:
        raise RuntimeError("; ".join(errors))


def _resource_sample(workers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import psutil

    processes = []
    combined_rss = 0
    combined_threads = 0
    for worker in workers:
        process = psutil.Process(worker["process"].pid)
        try:
            rss = process.memory_info().rss
            threads = process.num_threads()
            cpu = sum(process.cpu_times()[:2])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            rss = threads = 0
            cpu = 0.0
        combined_rss += rss
        combined_threads += threads
        processes.append(
            {"slot": worker["slot"], "pid": process.pid, "rss_bytes": rss,
             "thread_count": threads, "cpu_time_seconds": cpu}
        )
    virtual = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "monotonic_ns": time.monotonic_ns(),
        "load_average": list(os.getloadavg()),
        "available_memory_bytes": virtual.available,
        "physical_memory_bytes": virtual.total,
        "swap_in_bytes": swap.sin,
        "swap_out_bytes": swap.sout,
        "combined_rss_bytes": combined_rss,
        "combined_thread_count": combined_threads,
        "workers": processes,
    }


def _swap_counter_sample() -> dict[str, int]:
    """Read host swap counters independently of any dispatch boundary."""
    import psutil

    swap = psutil.swap_memory()
    return {
        "monotonic_ns": time.monotonic_ns(),
        "swap_in_bytes": int(swap.sin),
        "swap_out_bytes": int(swap.sout),
    }


def _validate_swap_policy(value: Any) -> str:
    if type(value) is not str or value not in SWAP_POLICIES:
        raise RuntimeError(f"unknown swap policy: {value!r}")
    return str(value)


def _new_swap_session_telemetry() -> dict[str, Any]:
    telemetry: dict[str, Any] = {
        "sample_count": 0,
        "samples": [],
        "swap_in_delta": 0,
        "swap_out_delta": 0,
    }
    _append_swap_session_sample(telemetry)
    return telemetry


def _append_swap_session_sample(
    telemetry: dict[str, Any], sample: Mapping[str, Any] | None = None
) -> int:
    """Extend one continuous host-counter record for a worker session."""
    observed = dict(_swap_counter_sample() if sample is None else sample)
    samples = telemetry.get("samples")
    if not isinstance(samples, list):
        raise RuntimeError("worker-session swap samples are invalid")
    if set(observed) != {"monotonic_ns", "swap_in_bytes", "swap_out_bytes"} or any(
        type(observed[name]) is not int or observed[name] < 0
        for name in observed
    ):
        raise RuntimeError("worker-session swap sample is invalid")
    if samples and observed["monotonic_ns"] <= samples[-1].get("monotonic_ns", -1):
        raise RuntimeError("worker-session swap clock did not advance")
    samples.append(observed)
    telemetry["sample_count"] = len(samples)
    telemetry["swap_in_delta"] = (
        samples[-1]["swap_in_bytes"] - samples[0]["swap_in_bytes"]
    )
    telemetry["swap_out_delta"] = (
        samples[-1]["swap_out_bytes"] - samples[0]["swap_out_bytes"]
    )
    return len(samples) - 1


def _swap_session_telemetry_structurally_valid(value: Any) -> bool:
    """Validate one monotonic swap-counter record without applying its policy gate."""
    if not isinstance(value, Mapping) or set(value) != {
        "sample_count",
        "samples",
        "swap_in_delta",
        "swap_out_delta",
    }:
        return False
    samples = value.get("samples")
    if (
        not isinstance(samples, list)
        or not samples
        or type(value.get("sample_count")) is not int
        or value.get("sample_count") != len(samples)
        or type(value.get("swap_in_delta")) is not int
        or type(value.get("swap_out_delta")) is not int
    ):
        return False
    previous_monotonic_ns = -1
    previous_swap_in = -1
    previous_swap_out = -1
    for sample in samples:
        if not isinstance(sample, Mapping) or set(sample) != {
            "monotonic_ns",
            "swap_in_bytes",
            "swap_out_bytes",
        }:
            return False
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
            return False
        previous_monotonic_ns = monotonic_ns
        previous_swap_in = swap_in
        previous_swap_out = swap_out
    swap_in_delta = samples[-1]["swap_in_bytes"] - samples[0]["swap_in_bytes"]
    swap_out_delta = samples[-1]["swap_out_bytes"] - samples[0]["swap_out_bytes"]
    return (
        value.get("swap_in_delta") == swap_in_delta
        and value.get("swap_out_delta") == swap_out_delta
    )


def _swap_session_telemetry_valid(
    value: Any, swap_policy: str = DEFAULT_SWAP_POLICY
) -> bool:
    """Apply the selected full-lifecycle swap policy to one worker session."""
    try:
        policy = _validate_swap_policy(swap_policy)
    except RuntimeError:
        return False
    return bool(
        _swap_session_telemetry_structurally_valid(value)
        and value.get("swap_out_delta") == 0
        and (
            policy == SWAP_POLICY_QUALITY_ONLY_SWAP_IN
            or value.get("swap_in_delta") == 0
        )
    )


def _checkpoint_swap_session(
    telemetry: dict[str, Any], swap_policy: str = DEFAULT_SWAP_POLICY
) -> None:
    _append_swap_session_sample(telemetry)
    if not _swap_session_telemetry_valid(telemetry, swap_policy):
        raise RuntimeError("worker session observed swap I/O")


_MEASURED_SWAP_WINDOW_FIELDS = {
    "start_sample_index",
    "end_sample_index",
    "sample_count",
    "swap_in_delta",
    "swap_out_delta",
    "dispatches",
}
_MEASURED_SWAP_DISPATCH_FIELDS = {
    "label",
    "sample_index",
    "resource_first_monotonic_ns",
    "resource_last_monotonic_ns",
    "resource_first_swap_in_bytes",
    "resource_last_swap_in_bytes",
    "resource_first_swap_out_bytes",
    "resource_last_swap_out_bytes",
    "barrier_release_monotonic_ns",
    "max_report_end_monotonic_ns",
}


def _start_measured_swap_window(
    session_telemetry: dict[str, Any], *, swap_policy: str
) -> dict[str, Any]:
    """Start timing after setup using one sample owned by the full lifecycle."""
    _validate_swap_policy(swap_policy)
    sample_index = _append_swap_session_sample(session_telemetry)
    if not _swap_session_telemetry_valid(session_telemetry, swap_policy):
        raise RuntimeError("worker session observed swap I/O")
    return {
        "start_sample_index": sample_index,
        "end_sample_index": sample_index,
        "sample_count": 1,
        "swap_in_delta": 0,
        "swap_out_delta": 0,
        "dispatches": [],
    }


def _resource_swap_boundary(telemetry: Mapping[str, Any]) -> tuple[int, int]:
    samples = telemetry.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("dispatch resource samples are missing")
    try:
        first = screen.hardened._exact_int(
            samples[0].get("monotonic_ns"), "dispatch first resource sample"
        )
        last = screen.hardened._exact_int(
            samples[-1].get("monotonic_ns"), "dispatch last resource sample"
        )
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError("dispatch resource sample clock is invalid") from exc
    if first > last:
        raise RuntimeError("dispatch resource sample clock is reversed")
    return first, last


def _checkpoint_measured_swap_window(
    session_telemetry: dict[str, Any],
    measured_window: dict[str, Any],
    *,
    label: str,
    resource_telemetry: Mapping[str, Any],
    reports: Sequence[Mapping[str, Any]],
    swap_policy: str,
) -> int:
    """Close one measured dispatch and cross-bind it to the lifecycle record."""
    if not isinstance(label, str) or not label:
        raise RuntimeError("measured dispatch label is invalid")
    first, last = _resource_swap_boundary(resource_telemetry)
    resource_samples = resource_telemetry["samples"]
    if not reports:
        raise RuntimeError("measured dispatch reports are missing")
    release = screen.hardened._exact_int(
        resource_telemetry.get("barrier_release_monotonic_ns"),
        "measured dispatch barrier release",
    )
    report_ends = [
        screen.hardened._exact_int(
            report.get("ended_monotonic_ns"), "measured dispatch report end"
        )
        for report in reports
    ]
    if any(report.get("barrier_release_monotonic_ns") != release for report in reports):
        raise RuntimeError("measured dispatch barrier identity changed")
    sample_index = _append_swap_session_sample(session_telemetry)
    measured_window["end_sample_index"] = sample_index
    measured_window["sample_count"] = (
        sample_index - measured_window["start_sample_index"] + 1
    )
    samples = session_telemetry["samples"]
    start = measured_window["start_sample_index"]
    measured_window["swap_in_delta"] = (
        samples[sample_index]["swap_in_bytes"] - samples[start]["swap_in_bytes"]
    )
    measured_window["swap_out_delta"] = (
        samples[sample_index]["swap_out_bytes"] - samples[start]["swap_out_bytes"]
    )
    measured_window["dispatches"].append(
        {
            "label": label,
            "sample_index": sample_index,
            "resource_first_monotonic_ns": first,
            "resource_last_monotonic_ns": last,
            "resource_first_swap_in_bytes": resource_samples[0]["swap_in_bytes"],
            "resource_last_swap_in_bytes": resource_samples[-1]["swap_in_bytes"],
            "resource_first_swap_out_bytes": resource_samples[0]["swap_out_bytes"],
            "resource_last_swap_out_bytes": resource_samples[-1]["swap_out_bytes"],
            "barrier_release_monotonic_ns": release,
            "max_report_end_monotonic_ns": max(report_ends),
        }
    )
    if not _swap_session_telemetry_valid(session_telemetry, swap_policy):
        raise RuntimeError("worker session observed swap I/O")
    if not _measured_swap_window_valid(
        measured_window,
        session_telemetry,
        swap_policy=swap_policy,
        expected_dispatches=[(label, resource_telemetry, reports)]
        if len(measured_window["dispatches"]) == 1
        else None,
        require_shutdown_sample=False,
    ):
        raise RuntimeError("measured phase observed swap I/O or invalid boundaries")
    return sample_index


def _measured_swap_window_structurally_valid(
    value: Any,
    session_telemetry: Any,
    *,
    expected_dispatches: Sequence[
        tuple[str, Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ]
    | None = None,
    require_shutdown_sample: bool,
    expected_start_sample_index: int | None = None,
    require_exact_shutdown_suffix: bool = False,
) -> bool:
    """Validate a derived measured window against shared lifecycle samples."""
    if (
        not isinstance(value, Mapping)
        or set(value) != _MEASURED_SWAP_WINDOW_FIELDS
        or not _swap_session_telemetry_structurally_valid(session_telemetry)
    ):
        return False
    try:
        start = screen.hardened._exact_int(
            value.get("start_sample_index"), "measured swap start index"
        )
        end = screen.hardened._exact_int(
            value.get("end_sample_index"), "measured swap end index"
        )
        dispatches = value.get("dispatches")
        samples = session_telemetry["samples"]
        if (
            start <= 0
            or (
                expected_start_sample_index is not None
                and start != expected_start_sample_index
            )
            or end < start
            or end >= len(samples)
            or not isinstance(dispatches, list)
            or type(value.get("sample_count")) is not int
            or value.get("sample_count") != end - start + 1
            or len(dispatches) != end - start
            or (require_shutdown_sample and end >= len(samples) - 1)
            or (require_exact_shutdown_suffix and end != len(samples) - 2)
            or type(value.get("swap_in_delta")) is not int
            or type(value.get("swap_out_delta")) is not int
        ):
            return False
        if (
            value.get("swap_in_delta")
            != samples[end]["swap_in_bytes"] - samples[start]["swap_in_bytes"]
            or value.get("swap_out_delta")
            != samples[end]["swap_out_bytes"] - samples[start]["swap_out_bytes"]
        ):
            return False
        observed_labels = set()
        for ordinal, dispatch in enumerate(dispatches, start=1):
            if (
                not isinstance(dispatch, Mapping)
                or set(dispatch) != _MEASURED_SWAP_DISPATCH_FIELDS
            ):
                return False
            label = dispatch.get("label")
            sample_index = dispatch.get("sample_index")
            resource_first = dispatch.get("resource_first_monotonic_ns")
            resource_last = dispatch.get("resource_last_monotonic_ns")
            resource_first_swap_in = dispatch.get("resource_first_swap_in_bytes")
            resource_last_swap_in = dispatch.get("resource_last_swap_in_bytes")
            resource_first_swap_out = dispatch.get("resource_first_swap_out_bytes")
            resource_last_swap_out = dispatch.get("resource_last_swap_out_bytes")
            release = dispatch.get("barrier_release_monotonic_ns")
            report_end = dispatch.get("max_report_end_monotonic_ns")
            expected_index = start + ordinal
            if (
                not isinstance(label, str)
                or not label
                or label in observed_labels
                or type(sample_index) is not int
                or sample_index != expected_index
                or type(resource_first) is not int
                or type(resource_last) is not int
                or type(resource_first_swap_in) is not int
                or type(resource_last_swap_in) is not int
                or type(resource_first_swap_out) is not int
                or type(resource_last_swap_out) is not int
                or type(release) is not int
                or type(report_end) is not int
                or not samples[expected_index - 1]["monotonic_ns"]
                < resource_first
                <= resource_last
                < samples[expected_index]["monotonic_ns"]
                or not samples[expected_index - 1]["monotonic_ns"]
                < release
                <= report_end
                <= resource_last
                or not samples[expected_index - 1]["swap_in_bytes"]
                <= resource_first_swap_in
                <= resource_last_swap_in
                <= samples[expected_index]["swap_in_bytes"]
                or not samples[expected_index - 1]["swap_out_bytes"]
                <= resource_first_swap_out
                <= resource_last_swap_out
                <= samples[expected_index]["swap_out_bytes"]
            ):
                return False
            observed_labels.add(label)
        if expected_dispatches is not None:
            if len(expected_dispatches) != len(dispatches):
                return False
            for dispatch, (expected_label, telemetry, reports) in zip(
                dispatches, expected_dispatches
            ):
                first, last = _resource_swap_boundary(telemetry)
                release = screen.hardened._exact_int(
                    telemetry.get("barrier_release_monotonic_ns"),
                    "expected measured dispatch release",
                )
                if not reports:
                    return False
                report_end = max(
                    screen.hardened._exact_int(
                        report.get("ended_monotonic_ns"),
                        "expected measured dispatch report end",
                    )
                    for report in reports
                )
                if (
                    dispatch["label"] != expected_label
                    or dispatch["resource_first_monotonic_ns"] != first
                    or dispatch["resource_last_monotonic_ns"] != last
                    or dispatch["resource_first_swap_in_bytes"]
                    != telemetry["samples"][0]["swap_in_bytes"]
                    or dispatch["resource_last_swap_in_bytes"]
                    != telemetry["samples"][-1]["swap_in_bytes"]
                    or dispatch["resource_first_swap_out_bytes"]
                    != telemetry["samples"][0]["swap_out_bytes"]
                    or dispatch["resource_last_swap_out_bytes"]
                    != telemetry["samples"][-1]["swap_out_bytes"]
                    or dispatch["barrier_release_monotonic_ns"] != release
                    or dispatch["max_report_end_monotonic_ns"] != report_end
                    or any(
                        report.get("barrier_release_monotonic_ns") != release
                        for report in reports
                    )
                ):
                    return False
    except (KeyError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _measured_swap_window_valid(
    value: Any,
    session_telemetry: Any,
    *,
    swap_policy: str = DEFAULT_SWAP_POLICY,
    expected_dispatches: Sequence[
        tuple[str, Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ]
    | None = None,
    require_shutdown_sample: bool,
    expected_start_sample_index: int | None = None,
    require_exact_shutdown_suffix: bool = False,
) -> bool:
    try:
        policy = _validate_swap_policy(swap_policy)
    except RuntimeError:
        return False
    return bool(
        _measured_swap_window_structurally_valid(
            value,
            session_telemetry,
            expected_dispatches=expected_dispatches,
            require_shutdown_sample=require_shutdown_sample,
            expected_start_sample_index=expected_start_sample_index,
            require_exact_shutdown_suffix=require_exact_shutdown_suffix,
        )
        and value.get("swap_out_delta") == 0
        and (
            policy == SWAP_POLICY_QUALITY_ONLY_SWAP_IN
            or value.get("swap_in_delta") == 0
        )
    )


def _truncate_measured_swap_window(
    measured_window: dict[str, Any],
    session_telemetry: Mapping[str, Any],
    dispatch_count: int,
) -> None:
    """Roll an interrupted in-memory transaction back to its durable prefix."""
    dispatches = measured_window.get("dispatches")
    if (
        type(dispatch_count) is not int
        or dispatch_count < 0
        or not isinstance(dispatches, list)
        or dispatch_count > len(dispatches)
    ):
        raise RuntimeError("cannot truncate measured swap coverage")
    del dispatches[dispatch_count:]
    start = measured_window["start_sample_index"]
    end = start + dispatch_count
    samples = session_telemetry["samples"]
    measured_window["end_sample_index"] = end
    measured_window["sample_count"] = dispatch_count + 1
    measured_window["swap_in_delta"] = (
        samples[end]["swap_in_bytes"] - samples[start]["swap_in_bytes"]
    )
    measured_window["swap_out_delta"] = (
        samples[end]["swap_out_bytes"] - samples[start]["swap_out_bytes"]
    )


def _physical_memory_bytes() -> int:
    import psutil

    return int(psutil.virtual_memory().total)


def _await_commands(
    workers: Sequence[Mapping[str, Any]],
    command_ids: set[str],
    *,
    timeout_seconds: float,
    command_slots: Mapping[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pending = set(command_ids)
    messages = []
    samples = []
    deadline = time.monotonic() + timeout_seconds
    while pending:
        if time.monotonic() >= deadline:
            raise RuntimeError(f"worker commands timed out: {sorted(pending)}")
        samples.append(_resource_sample(workers))
        for worker in workers:
            connection = worker["connection"]
            while connection.poll(0):
                message = connection.recv()
                command_id = message.get("command_id")
                if command_id not in pending:
                    raise RuntimeError(f"unexpected worker report: {message}")
                messages.append(message)
                pending.remove(command_id)
            if (
                not worker["process"].is_alive()
                and command_slots is not None
            ):
                abandoned = [
                    command_id
                    for command_id in pending
                    if command_slots.get(command_id) == worker["slot"]
                ]
                for command_id in abandoned:
                    messages.append(
                        {
                            "type": "error",
                            "command_id": command_id,
                            "slot": worker["slot"],
                            "pid": worker["process"].pid,
                            "error_type": "WorkerExit",
                            "error": f"worker exited with code {worker['process'].exitcode}",
                        }
                    )
                    pending.remove(command_id)
        if pending:
            time.sleep(1.0)
    samples.append(_resource_sample(workers))
    errors = [message for message in messages if message.get("type") == "error"]
    if errors:
        raise RuntimeError("worker command failed: " + json.dumps(errors, sort_keys=True))
    if len(messages) != len(command_ids):
        raise RuntimeError("worker report count does not match command count")
    initial, final = samples[0], samples[-1]
    telemetry = {
        "sample_count": len(samples),
        "samples": samples,
        "physical_memory_bytes": initial["physical_memory_bytes"],
        "peak_combined_rss_bytes": max(item["combined_rss_bytes"] for item in samples),
        "peak_combined_thread_count": max(
            item["combined_thread_count"] for item in samples
        ),
        "swap_in_delta": final["swap_in_bytes"] - initial["swap_in_bytes"],
        "swap_out_delta": final["swap_out_bytes"] - initial["swap_out_bytes"],
    }
    return messages, telemetry


def _query_worker_high_water(
    workers: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    command_slots = {}
    for worker in workers:
        command_id = f"resource-{worker['slot']}-{time.monotonic_ns()}"
        command_slots[command_id] = worker["slot"]
        worker["connection"].send(
            {"kind": "resource", "command_id": command_id}
        )
    messages, telemetry = _await_commands(
        workers,
        set(command_slots),
        timeout_seconds=60.0,
        command_slots=command_slots,
    )
    messages.sort(key=lambda item: item.get("slot"))
    for message in messages:
        slot = message.get("slot")
        if (
            message.get("type") != "resource"
            or message.get("pid") != workers[slot]["process"].pid
            or not isinstance(message.get("process_peak_rss_bytes"), int)
            or message["process_peak_rss_bytes"] <= 0
        ):
            raise RuntimeError(f"invalid worker high-water report: {message}")
    return messages, telemetry


def _warm_workers(
    workers: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for worker in workers:
        command_id = f"warmup-{worker['slot']}-{time.monotonic_ns()}"
        worker["connection"].send({"kind": "warmup", "command_id": command_id})
        messages, _ = _await_commands(
            workers,
            {command_id},
            timeout_seconds=900.0,
            command_slots={command_id: worker["slot"]},
        )
        message = messages[0]
        if message.get("type") != "warmup" or message.get("slot") != worker["slot"]:
            raise RuntimeError(f"invalid worker warmup report: {message}")
        record = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "pid": message["pid"],
            "worker_slot": worker["slot"],
            "warmup": message["warmup"],
        }
        # Reuse the exhaustive warmup validator after removing the runner-owned slot.
        screen._validate_followon_warmup_history(
            [{key: value for key, value in record.items() if key != "worker_slot"}],
            expected_thread_count=EXPECTED_CHILD_CPUS,
            expected_latest_pid=message["pid"],
        )
        records.append(record)
    return records


def _prime_preflight_workers(
    workers: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Materialize both pilot datasets in every worker before timing."""
    keys = [
        _job_key(("physiochemical_protein", 0, 0), "A10"),
        _job_key(("QSAR-TID-11", 2, 2), "A10"),
    ]
    records = []
    for worker in workers:
        command_id = f"prime-{worker['slot']}-{time.monotonic_ns()}"
        worker["connection"].send(
            {
                "kind": "prime",
                "command_id": command_id,
                "keys": [_key_payload(key) for key in keys],
            }
        )
        messages, _ = _await_commands(
            workers,
            {command_id},
            timeout_seconds=900.0,
            command_slots={command_id: worker["slot"]},
        )
        message = messages[0]
        if (
            message.get("type") != "prime"
            or message.get("slot") != worker["slot"]
            or message.get("pid") != worker["process"].pid
            or [_key_tuple(item) for item in message.get("keys", [])] != keys
        ):
            raise RuntimeError(f"invalid worker data-prime report: {message}")
        records.append(
            {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "pid": message["pid"],
                "worker_slot": worker["slot"],
                "keys": message["keys"],
            }
        )
    return records


def _dispatch_runs(
    workers: Sequence[Mapping[str, Any]],
    assignments: Sequence[tuple[int, tuple[str, int, int, str], Path]],
    *,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    release_ns = time.monotonic_ns() + 250_000_000
    command_ids = set()
    expected = {}
    for slot, key, result_root in assignments:
        command_id = f"{label}-{slot}-{time.monotonic_ns()}"
        command_ids.add(command_id)
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
    messages, telemetry = _await_commands(
        workers,
        command_ids,
        timeout_seconds=TIME_LIMIT_SECONDS + 900.0,
        command_slots={command_id: slot for command_id, (slot, _, _) in expected.items()},
    )
    reports = []
    for message in messages:
        command_id = message["command_id"]
        slot, key, result_root = expected[command_id]
        expected_path = result_root / screen.expected_result_relative_path(
            key[0], key[1], key[2], PUBLIC_TO_INTERNAL_ARM[key[3]]
        )
        if (
            message.get("type") != "result"
            or message.get("status") != "ok"
            or message.get("slot") != slot
            or message.get("pid") != workers[slot]["process"].pid
            or message.get("start_method") != "spawn"
            or _key_tuple(message.get("key", {})) != key
            or Path(message.get("result_path", "")).resolve() != expected_path.resolve()
            or message.get("result_count") != 1
            or not expected_path.is_file()
            or message.get("result_sha256")
            != screen.hardened._sha256_file(expected_path)
            or message.get("result_size_bytes") != expected_path.stat().st_size
            or not isinstance(message.get("process_peak_rss_bytes"), int)
            or message["process_peak_rss_bytes"] <= 0
        ):
            raise RuntimeError(f"worker result identity mismatch: {message}")
        reports.append(message)
    reports.sort(key=lambda item: item["slot"])
    high_water_reports, high_water_telemetry = _query_worker_high_water(workers)
    telemetry["samples"].extend(high_water_telemetry["samples"])
    telemetry["sample_count"] = len(telemetry["samples"])
    telemetry["swap_in_delta"] = (
        telemetry["samples"][-1]["swap_in_bytes"]
        - telemetry["samples"][0]["swap_in_bytes"]
    )
    telemetry["swap_out_delta"] = (
        telemetry["samples"][-1]["swap_out_bytes"]
        - telemetry["samples"][0]["swap_out_bytes"]
    )
    high_water_by_worker = [
        {
            "slot": item["slot"],
            "pid": item["pid"],
            "process_peak_rss_bytes": item["process_peak_rss_bytes"],
        }
        for item in high_water_reports
    ]
    high_water_combined = sum(
        item["process_peak_rss_bytes"] for item in high_water_by_worker
    )
    telemetry["worker_high_water_rss_bytes"] = high_water_by_worker
    telemetry["high_water_combined_rss_bytes"] = high_water_combined
    telemetry["peak_combined_rss_bytes"] = max(
        telemetry["peak_combined_rss_bytes"], high_water_combined
    )
    telemetry["peak_combined_thread_count"] = max(
        telemetry["peak_combined_thread_count"],
        high_water_telemetry["peak_combined_thread_count"],
    )
    starts = [item["started_monotonic_ns"] for item in reports]
    ends = [item["ended_monotonic_ns"] for item in reports]
    telemetry["barrier_release_monotonic_ns"] = release_ns
    telemetry["start_skew_seconds"] = (
        (max(starts) - min(starts)) / 1e9 if len(starts) > 1 else 0.0
    )
    telemetry["wave_seconds"] = (max(ends) - release_ns) / 1e9
    telemetry["overlap_seconds"] = (
        max(0.0, (min(ends) - max(starts)) / 1e9) if len(starts) > 1 else 0.0
    )
    telemetry["solo_tail_seconds"] = (
        max(0.0, (max(ends) - min(ends)) / 1e9) if len(ends) > 1 else 0.0
    )
    return reports, telemetry


def preflight_throughput_speedup(
    isolated_seconds: Sequence[float], concurrent_wave_seconds: Sequence[float]
) -> float:
    if len(isolated_seconds) != 4 or len(concurrent_wave_seconds) != 2:
        raise ValueError("preflight requires four isolated jobs and two waves")
    values = [float(value) for value in (*isolated_seconds, *concurrent_wave_seconds)]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("preflight durations must be positive and finite")
    return sum(values[:4]) / sum(values[4:])


def preflight_reciprocal_asymmetry(
    isolated_by_key: Mapping[tuple[str, int, int, str], Mapping[str, Any]],
    concurrent_by_key: Mapping[tuple[str, int, int, str], Mapping[str, Any]],
) -> float:
    """Return the worst multiplicative slowdown asymmetry by arm or slot."""
    if set(isolated_by_key) != set(concurrent_by_key) or len(isolated_by_key) != 4:
        raise ValueError("reciprocal asymmetry requires four matched executions")
    ratios_by_arm = {arm: [] for arm in PUBLIC_TO_INTERNAL_ARM}
    ratios_by_slot = {slot: [] for slot in range(WORKER_COUNT)}
    ratios_by_dataset: dict[str, dict[str, float]] = {}
    for key in sorted(isolated_by_key):
        isolated = float(isolated_by_key[key].get("elapsed_seconds", math.nan))
        concurrent = float(concurrent_by_key[key].get("elapsed_seconds", math.nan))
        slot = screen.hardened._exact_int(
            concurrent_by_key[key].get("slot"), "preflight reciprocal slot"
        )
        if (
            not math.isfinite(isolated)
            or not math.isfinite(concurrent)
            or isolated <= 0.0
            or concurrent <= 0.0
            or slot not in ratios_by_slot
        ):
            raise ValueError("reciprocal durations must be finite and positive")
        ratio = concurrent / isolated
        arm = key[3]
        ratios_by_arm[arm].append(ratio)
        ratios_by_slot[slot].append(ratio)
        ratios_by_dataset.setdefault(key[0], {})[arm] = ratio

    def geomean(values: Sequence[float]) -> float:
        if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
            raise ValueError("reciprocal ratio group is incomplete")
        return math.exp(sum(math.log(value) for value in values) / len(values))

    arm_means = [geomean(values) for values in ratios_by_arm.values()]
    slot_means = [geomean(values) for values in ratios_by_slot.values()]
    factors = [
        max(arm_means) / min(arm_means),
        max(slot_means) / min(slot_means),
    ]
    for values in ratios_by_dataset.values():
        if set(values) != set(PUBLIC_TO_INTERNAL_ARM):
            raise ValueError("reciprocal dataset arm coverage is incomplete")
        factors.append(max(values.values()) / min(values.values()))
    result = max(factors)
    if not math.isfinite(result) or result < 1.0:
        raise ValueError("reciprocal asymmetry is invalid")
    return result


def _preflight_measured_dispatches(
    report: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any], Sequence[Mapping[str, Any]]]]:
    dispatches: list[
        tuple[str, Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ] = []
    for index, item in enumerate(report.get("isolated_runs", [])):
        if not isinstance(item, Mapping) or not isinstance(
            item.get("telemetry"), Mapping
        ):
            raise RuntimeError("preflight isolated dispatch evidence is invalid")
        dispatches.append(
            (f"preflight-isolated-{index}", item["telemetry"], [item])
        )
    for index, wave in enumerate(report.get("concurrent_waves", [])):
        if not isinstance(wave, Mapping) or not isinstance(
            wave.get("telemetry"), Mapping
        ):
            raise RuntimeError("preflight concurrent dispatch evidence is invalid")
        reports = wave.get("reports")
        if not isinstance(reports, list):
            raise RuntimeError("preflight concurrent reports are invalid")
        dispatches.append(
            (f"preflight-concurrent-{index}", wave["telemetry"], reports)
        )
    return dispatches


def evaluate_preflight(report: Mapping[str, Any]) -> dict[str, Any]:
    isolated = report.get("isolated_runs")
    waves = report.get("concurrent_waves")
    if not isinstance(isolated, list) or not isinstance(waves, list):
        raise ValueError("preflight report is missing run lists")
    try:
        swap_policy = _validate_swap_policy(report.get("swap_policy"))
    except RuntimeError:
        swap_policy = ""
    isolated_timing_valid = len(isolated) == 4 and all(
        isinstance(item, Mapping)
        and _dispatch_timing_valid([item], item.get("telemetry"))
        for item in isolated
    )
    concurrent_timing_valid = len(waves) == 2 and all(
        isinstance(wave, Mapping)
        and isinstance(wave.get("reports"), list)
        and _dispatch_timing_valid(
            wave["reports"], wave.get("telemetry"), wrapper=wave
        )
        for wave in waves
    )
    try:
        if not isolated_timing_valid or not concurrent_timing_valid:
            raise ValueError("preflight timing evidence is inconsistent")
        speedup = preflight_throughput_speedup(
            [item["elapsed_seconds"] for item in isolated],
            [item["wave_seconds"] for item in waves],
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        speedup = float("nan")
    concurrent = [item for wave in waves for item in wave.get("reports", [])]
    isolated_by_key = {
        _key_tuple(item["key"]): item for item in isolated if isinstance(item, Mapping)
    }
    concurrent_by_key = {
        _key_tuple(item["key"]): item
        for item in concurrent
        if isinstance(item, Mapping)
    }
    expected_keys = {
        (dataset, repeat, fold, arm)
        for dataset, (repeat, fold) in PREFLIGHT_COORDINATES.items()
        for arm in PUBLIC_TO_INTERNAL_ARM
    }
    reciprocal = [
        {
            ("physiochemical_protein", 0, 0, "A10"): 0,
            ("QSAR-TID-11", 2, 2, "B10"): 1,
        },
        {
            ("physiochemical_protein", 0, 0, "B10"): 0,
            ("QSAR-TID-11", 2, 2, "A10"): 1,
        },
    ]

    def valid_sha256(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    run_reports = [*isolated, *concurrent]
    run_valid = (
        len(isolated) == 4
        and len(waves) == 2
        and len(concurrent) == 4
        and all(
            item.get("status") == "ok"
            and item.get("result_count") == 1
            and item.get("child_count") == 8
            and item.get("deadline_hit") is False
            and valid_sha256(item.get("behavior_sha256"))
            and (
                item["key"]["arm"] != "A10"
                or item.get("auto_candidate_fit_count") == 24
            )
            for item in run_reports
        )
    )
    fingerprints_match = (
        len(isolated_by_key) == 4
        and set(isolated_by_key) == set(concurrent_by_key) == expected_keys
        and all(
            isolated_by_key[key].get("behavior_sha256")
            == concurrent_by_key[key].get("behavior_sha256")
            for key in isolated_by_key
        )
    )
    reciprocal_pairing = len(waves) == 2 and all(
        {
            _key_tuple(item["key"]): item.get("slot")
            for item in wave.get("reports", [])
        }
        == reciprocal[index]
        for index, wave in enumerate(waves)
    )
    try:
        reciprocal_asymmetry = preflight_reciprocal_asymmetry(
            isolated_by_key, concurrent_by_key
        )
    except (RuntimeError, TypeError, ValueError, OverflowError):
        reciprocal_asymmetry = math.inf
    try:
        expected_pids = {
            screen.hardened._exact_int(item.get("slot"), "preflight worker slot"):
            screen.hardened._exact_int(item.get("pid"), "preflight worker pid")
            for item in report.get("worker_ready", [])
        }
    except (AttributeError, RuntimeError, TypeError, ValueError):
        expected_pids = {}
    expected_prime_keys = [
        _job_key(("physiochemical_protein", 0, 0), "A10"),
        _job_key(("QSAR-TID-11", 2, 2), "A10"),
    ]
    try:
        prime_by_slot = {
            screen.hardened._exact_int(item.get("worker_slot"), "data-prime slot"):
            item
            for item in report.get("worker_data_prime", [])
        }
        data_prime_complete = (
            set(prime_by_slot) == set(range(WORKER_COUNT))
            and all(
                item.get("pid") == expected_pids.get(slot)
                and [_key_tuple(key) for key in item.get("keys", [])]
                == expected_prime_keys
                for slot, item in prime_by_slot.items()
            )
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        data_prime_complete = False
    isolated_operational = (
        isolated_timing_valid
        and set(expected_pids) == set(range(WORKER_COUNT))
        and all(
            _resource_telemetry_valid(
                item.get("telemetry"),
                expected_pids=expected_pids,
                reports=[item],
                swap_policy=swap_policy,
            )
            for item in isolated
        )
    )
    operational = (
        isolated_operational
        and concurrent_timing_valid
        and reciprocal_pairing
        and all(
            math.isfinite(float(wave.get("start_skew_seconds", math.inf)))
            and 0.0 <= float(wave.get("start_skew_seconds", math.inf)) <= 1.0
            and math.isfinite(float(wave.get("wave_seconds", math.inf)))
            and 0.0 < float(wave.get("wave_seconds", math.inf)) < 1_800.0
            and all(
                math.isfinite(float(item.get("elapsed_seconds", math.inf)))
                and 0.0 < float(item.get("elapsed_seconds", math.inf)) < 1_800.0
                for item in wave.get("reports", [])
            )
            and _resource_telemetry_valid(
                wave.get("telemetry"),
                expected_pids=expected_pids,
                reports=wave.get("reports", []),
                swap_policy=swap_policy,
            )
            for wave in waves
        )
    )
    try:
        measured_dispatches = _preflight_measured_dispatches(report)
    except RuntimeError:
        measured_dispatches = []
    session_swap = report.get("worker_session_swap_telemetry")
    measured_swap = report.get("measured_phase_swap_window")
    criteria = {
        "throughput_speedup_at_least_1_10": math.isfinite(speedup)
        and speedup >= 1.10,
        "eight_valid_executions": run_valid,
        "exact_behavior_fingerprints": fingerprints_match,
        "reciprocal_pairing": reciprocal_pairing,
        "reciprocal_arm_slot_symmetry": (
            reciprocal_pairing
            and math.isfinite(reciprocal_asymmetry)
            and reciprocal_asymmetry <= MAX_PREFLIGHT_RECIPROCAL_ASYMMETRY
        ),
        "data_prime_complete": data_prime_complete,
        "operational_limits": operational,
        "swap_policy_attested": swap_policy in SWAP_POLICIES,
        "full_session_swap_policy": _swap_session_telemetry_valid(
            session_swap, swap_policy
        ),
        "measured_phase_swap_policy": (
            len(measured_dispatches) == 6
            and _measured_swap_window_valid(
                measured_swap,
                session_swap,
                swap_policy=swap_policy,
                expected_dispatches=measured_dispatches,
                require_shutdown_sample=True,
                expected_start_sample_index=3,
                require_exact_shutdown_suffix=True,
            )
            and measured_swap.get("sample_count") == 7
            and session_swap.get("sample_count") == 11
        ),
        "no_worker_restarts": report.get("worker_restarts") is False,
        "no_preflight_error": report.get("preflight_error") is None,
        "no_sequential_recovery_override": report.get("sequential_recovery") is None,
    }
    timing_admissible = swap_policy == SWAP_POLICY_STRICT
    scientific_criteria = dict(criteria)
    if swap_policy == SWAP_POLICY_QUALITY_ONLY_SWAP_IN:
        scientific_criteria.pop("throughput_speedup_at_least_1_10", None)
        scientific_criteria.pop("reciprocal_arm_slot_symmetry", None)
    passed = all(scientific_criteria.values())
    return {
        "passed": passed,
        "execution_mode": "concurrent" if passed else "sequential_fallback",
        "throughput_speedup": speedup if math.isfinite(speedup) else None,
        "reciprocal_asymmetry_ratio": (
            reciprocal_asymmetry if math.isfinite(reciprocal_asymmetry) else None
        ),
        "timing_admissible": timing_admissible,
        "timing_inadmissibility_reason": (
            None if timing_admissible else "quality_only_swap_in_policy"
        ),
        "criteria": criteria,
        "mode_selection_criteria": sorted(scientific_criteria),
    }


def _validate_sequential_recovery_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "source_output_dir",
        "source_manifest_sha256",
        "source_invalid_attempt_sha256",
        "source_owner_state_sha256",
        "source_preflight_sha256",
        "source_execution_grid_sha256",
        "source_git_head",
        "source_execution_mode",
        "source_swap_policy",
        "invalid_attempt",
    }:
        raise RuntimeError("sequential recovery record fields are incomplete")
    marker = value["invalid_attempt"]
    hashes = (
        value["source_manifest_sha256"],
        value["source_invalid_attempt_sha256"],
        value["source_owner_state_sha256"],
        value["source_preflight_sha256"],
    )
    if (
        value["schema_version"] != 1
        or not Path(str(value["source_output_dir"])).is_absolute()
        or any(
            not isinstance(item, str)
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in hashes
        )
        or not isinstance(value["source_git_head"], str)
        or len(value["source_git_head"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in value["source_git_head"]
        )
        or value["source_execution_mode"] != "concurrent"
        or value["source_swap_policy"] not in SWAP_POLICIES
        or not isinstance(value["source_execution_grid_sha256"], str)
        or len(value["source_execution_grid_sha256"]) != 64
        or not isinstance(marker, Mapping)
        or set(marker) != {
            "schema_version",
            "kind",
            "failed_at_utc",
            "pid",
            "output_dir",
            "execution_mode",
            "swap_policy",
            "wave_schedule_sha256",
            "protocol_sha256",
            "manifest_sha256",
            "preflight_report_sha256",
            "execution_grid_sha256",
            "git_head",
            "error_type",
            "error",
            "recovery",
        }
        or marker.get("schema_version") != 1
        or marker.get("kind") != CAMPAIGN_KIND + "_invalid_attempt"
        or marker.get("output_dir") != value["source_output_dir"]
        or marker.get("execution_mode") != "concurrent"
        or marker.get("swap_policy") != value["source_swap_policy"]
        or marker.get("wave_schedule_sha256") != wave_schedule_sha256()
        or marker.get("protocol_sha256") != screen.protocol_sha256()
        or marker.get("manifest_sha256") != value["source_manifest_sha256"]
        or marker.get("preflight_report_sha256") != value["source_preflight_sha256"]
        or marker.get("execution_grid_sha256")
        != value["source_execution_grid_sha256"]
        or marker.get("git_head") != value["source_git_head"]
        or marker.get("recovery")
        != "use --sequential-recovery-from with a fresh output namespace"
        or not isinstance(marker.get("pid"), int)
        or marker["pid"] <= 0
        or not all(
            isinstance(marker.get(name), str) and marker[name]
            for name in ("failed_at_utc", "execution_grid_sha256", "error_type", "error")
        )
    ):
        raise RuntimeError("sequential recovery record does not match a failed attempt")
    return dict(value)


def collect_sequential_recovery(
    source_output_dir: Path,
    *,
    current_source: Mapping[str, Any],
    swap_policy: str = DEFAULT_SWAP_POLICY,
) -> dict[str, Any]:
    """Bind a fresh sequential run to one invalid concurrent attempt."""
    swap_policy = _validate_swap_policy(swap_policy)
    source_output_dir = source_output_dir.resolve(strict=True)
    manifest_path = source_output_dir / screen.MANIFEST_FILENAME
    marker_path = source_output_dir / INVALID_ATTEMPT_FILENAME
    completion_path = source_output_dir / screen.COMPLETION_ATTESTATION_FILENAME
    for path in (manifest_path, marker_path):
        screen.hardened._require_regular_archive_source(path, "recovery source artifact")
        if not path.is_file():
            raise RuntimeError(f"sequential recovery source is incomplete: {path}")
    if completion_path.exists() or completion_path.is_symlink():
        raise RuntimeError("a completed campaign cannot authorize sequential recovery")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("could not read sequential recovery source") from exc
    preflight, preflight_sha256 = _load_preflight_report(
        source_output_dir, swap_policy
    )
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("kind") != CAMPAIGN_KIND
        or manifest.get("protocol_sha256") != screen.protocol_sha256()
        or manifest.get("wave_schedule_sha256") != wave_schedule_sha256()
        or manifest.get("execution_mode") != "concurrent"
        or manifest.get("swap_policy") != swap_policy
        or manifest.get("preflight_report_sha256") != preflight_sha256
        or manifest.get("source") != dict(current_source)
        or preflight["decision"].get("execution_mode") != "concurrent"
        or preflight["decision"].get("passed") is not True
    ):
        raise RuntimeError("sequential recovery source is not the current failed campaign")
    owner_state_sha256 = validate_invalid_owner_session(source_output_dir, marker)
    record = {
        "schema_version": 1,
        "source_output_dir": str(source_output_dir),
        "source_manifest_sha256": screen.hardened._sha256_file(manifest_path),
        "source_invalid_attempt_sha256": screen.hardened._sha256_file(marker_path),
        "source_owner_state_sha256": owner_state_sha256,
        "source_preflight_sha256": preflight_sha256,
        "source_execution_grid_sha256": manifest["execution_grid_sha256"],
        "source_git_head": manifest["source"]["git_head"],
        "source_execution_mode": "concurrent",
        "source_swap_policy": swap_policy,
        "invalid_attempt": marker,
    }
    return _validate_sequential_recovery_record(record)


_TIMING_TOLERANCE_SECONDS = 1e-9
_DISPATCH_TELEMETRY_FIELDS = {
    "sample_count",
    "samples",
    "physical_memory_bytes",
    "peak_combined_rss_bytes",
    "peak_combined_thread_count",
    "swap_in_delta",
    "swap_out_delta",
    "worker_high_water_rss_bytes",
    "high_water_combined_rss_bytes",
    "barrier_release_monotonic_ns",
    "start_skew_seconds",
    "wave_seconds",
    "overlap_seconds",
    "solo_tail_seconds",
}
_SEQUENTIAL_AGGREGATE_TELEMETRY_FIELDS = _DISPATCH_TELEMETRY_FIELDS - {
    "barrier_release_monotonic_ns"
}


def _seconds_match(value: Any, expected: float) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    observed = float(value)
    return math.isfinite(observed) and math.isclose(
        observed,
        expected,
        rel_tol=0.0,
        abs_tol=_TIMING_TOLERANCE_SECONDS,
    )


def _dispatch_timing_valid(
    reports: Sequence[Mapping[str, Any]],
    telemetry: Any,
    *,
    wrapper: Mapping[str, Any] | None = None,
) -> bool:
    """Recompute one dispatch's timing from worker monotonic timestamps."""
    if len(reports) not in {1, WORKER_COUNT} or not isinstance(telemetry, Mapping):
        return False
    try:
        release = screen.hardened._exact_int(
            telemetry.get("barrier_release_monotonic_ns"),
            "telemetry barrier release",
        )
        starts = []
        ends = []
        for report in reports:
            if not isinstance(report, Mapping):
                return False
            report_release = screen.hardened._exact_int(
                report.get("barrier_release_monotonic_ns"), "report barrier release"
            )
            started = screen.hardened._exact_int(
                report.get("started_monotonic_ns"), "worker start"
            )
            ended = screen.hardened._exact_int(
                report.get("ended_monotonic_ns"), "worker end"
            )
            if (
                report_release != release
                or not release <= started < ended
                or not _seconds_match(
                    report.get("elapsed_seconds"), (ended - started) / 1e9
                )
            ):
                return False
            starts.append(started)
            ends.append(ended)
    except (RuntimeError, TypeError, ValueError):
        return False

    expected = {
        "start_skew_seconds": (
            (max(starts) - min(starts)) / 1e9 if len(starts) > 1 else 0.0
        ),
        "wave_seconds": (max(ends) - release) / 1e9,
        "overlap_seconds": (
            max(0.0, (min(ends) - max(starts)) / 1e9)
            if len(starts) > 1
            else 0.0
        ),
        "solo_tail_seconds": (
            max(0.0, (max(ends) - min(ends)) / 1e9)
            if len(ends) > 1
            else 0.0
        ),
    }
    if expected["wave_seconds"] <= 0.0 or any(
        not _seconds_match(telemetry.get(field), value)
        for field, value in expected.items()
    ):
        return False
    return wrapper is None or (
        _seconds_match(wrapper.get("wave_seconds"), expected["wave_seconds"])
        and _seconds_match(
            wrapper.get("start_skew_seconds"), expected["start_skew_seconds"]
        )
    )


def _resource_telemetry_structurally_valid(
    telemetry: Any,
    *,
    expected_pids: Mapping[int, int],
    reports: Sequence[Mapping[str, Any]],
    expected_physical_memory_bytes: int | None = None,
) -> bool:
    """Validate and reconcile resource evidence without applying resource limits."""
    if not isinstance(telemetry, Mapping):
        return False
    physical = telemetry.get("physical_memory_bytes")
    peak_rss = telemetry.get("peak_combined_rss_bytes")
    high_water = telemetry.get("worker_high_water_rss_bytes")
    high_water_combined = telemetry.get("high_water_combined_rss_bytes")
    samples = telemetry.get("samples")
    if (
        type(physical) is not int
        or physical <= 0
        or (
            expected_physical_memory_bytes is not None
            and physical != expected_physical_memory_bytes
        )
        or type(peak_rss) is not int
        or peak_rss < 0
        or not isinstance(high_water, list)
        or len(high_water) != WORKER_COUNT
        or set(expected_pids) != set(range(WORKER_COUNT))
        or not isinstance(samples, list)
        or not samples
        or telemetry.get("sample_count") != len(samples)
    ):
        return False
    previous_monotonic_ns = -1
    previous_swap_in = -1
    previous_swap_out = -1
    for sample in samples:
        if not isinstance(sample, Mapping):
            return False
        monotonic_ns = sample.get("monotonic_ns")
        swap_in = sample.get("swap_in_bytes")
        swap_out = sample.get("swap_out_bytes")
        combined_rss = sample.get("combined_rss_bytes")
        if (
            type(monotonic_ns) is not int
            or monotonic_ns <= previous_monotonic_ns
            or type(swap_in) is not int
            or swap_in < 0
            or swap_in < previous_swap_in
            or type(swap_out) is not int
            or swap_out < 0
            or swap_out < previous_swap_out
            or type(combined_rss) is not int
            or combined_rss < 0
            or sample.get("physical_memory_bytes") != physical
        ):
            return False
        previous_monotonic_ns = monotonic_ns
        previous_swap_in = swap_in
        previous_swap_out = swap_out
    swap_in_delta = samples[-1]["swap_in_bytes"] - samples[0]["swap_in_bytes"]
    swap_out_delta = samples[-1]["swap_out_bytes"] - samples[0]["swap_out_bytes"]
    by_slot: dict[int, Mapping[str, Any]] = {}
    for raw in high_water:
        if not isinstance(raw, Mapping):
            return False
        slot = raw.get("slot")
        pid = raw.get("pid")
        process_peak = raw.get("process_peak_rss_bytes")
        if (
            type(slot) is not int
            or slot not in range(WORKER_COUNT)
            or slot in by_slot
            or type(pid) is not int
            or pid != expected_pids.get(slot)
            or type(process_peak) is not int
            or process_peak <= 0
        ):
            return False
        by_slot[slot] = raw
    if (
        set(by_slot) != set(range(WORKER_COUNT))
        or type(high_water_combined) is not int
        or high_water_combined
        != sum(item["process_peak_rss_bytes"] for item in by_slot.values())
        or peak_rss < high_water_combined
        or peak_rss < max(item["combined_rss_bytes"] for item in samples)
        or swap_in_delta < 0
        or swap_out_delta < 0
        or telemetry.get("swap_in_delta") != swap_in_delta
        or telemetry.get("swap_out_delta") != swap_out_delta
    ):
        return False
    for report in reports:
        slot = report.get("slot")
        report_peak = report.get("process_peak_rss_bytes")
        if (
            type(slot) is not int
            or slot not in by_slot
            or type(report_peak) is not int
            or report_peak <= 0
            or report.get("pid") != expected_pids[slot]
            or report_peak > by_slot[slot]["process_peak_rss_bytes"]
        ):
            return False
    return True


def _resource_telemetry_valid(
    telemetry: Any,
    *,
    expected_pids: Mapping[int, int],
    reports: Sequence[Mapping[str, Any]],
    expected_physical_memory_bytes: int | None = None,
    swap_policy: str = DEFAULT_SWAP_POLICY,
) -> bool:
    """Require structurally valid telemetry that also passes resource policy."""
    try:
        policy = _validate_swap_policy(swap_policy)
    except RuntimeError:
        return False
    return (
        _resource_telemetry_structurally_valid(
            telemetry,
            expected_pids=expected_pids,
            reports=reports,
            expected_physical_memory_bytes=expected_physical_memory_bytes,
        )
        and telemetry.get("peak_combined_rss_bytes")
        < 0.8 * telemetry["physical_memory_bytes"]
        and telemetry.get("swap_out_delta") == 0
        and (
            policy == SWAP_POLICY_QUALITY_ONLY_SWAP_IN
            or telemetry.get("swap_in_delta") == 0
        )
    )


def _sequential_fallback_telemetry_valid(
    reports: Sequence[Mapping[str, Any]],
    telemetry: Any,
    mode_details: Any,
    *,
    swap_policy: str = DEFAULT_SWAP_POLICY,
) -> bool:
    """Reconcile two serial dispatch segments with their aggregate evidence."""
    if (
        not isinstance(mode_details, Mapping)
        or set(mode_details) != {"segments", "segment_telemetry"}
        or mode_details.get("segments") != WORKER_COUNT
        or not isinstance(mode_details.get("segment_telemetry"), list)
        or len(mode_details["segment_telemetry"]) != WORKER_COUNT
        or not isinstance(telemetry, Mapping)
        or set(telemetry) != _SEQUENTIAL_AGGREGATE_TELEMETRY_FIELDS
    ):
        return False
    segments = mode_details["segment_telemetry"]
    if any(
        not isinstance(segment, Mapping)
        or set(segment) != _DISPATCH_TELEMETRY_FIELDS
        for segment in segments
    ):
        return False
    try:
        by_slot = {
            screen.hardened._exact_int(report.get("slot"), "sequential report slot"):
            report
            for report in reports
        }
        expected_pids = {
            slot: screen.hardened._exact_int(
                by_slot[slot].get("pid"), "sequential worker pid"
            )
            for slot in range(WORKER_COUNT)
        }
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return False
    if (
        set(by_slot) != set(range(WORKER_COUNT))
        or len(reports) != WORKER_COUNT
        or len(set(expected_pids.values())) != WORKER_COUNT
    ):
        return False

    physical = telemetry.get("physical_memory_bytes")
    ordered_reports = [by_slot[slot] for slot in range(WORKER_COUNT)]
    for report, segment in zip(ordered_reports, segments):
        if (
            not _dispatch_timing_valid([report], segment)
            or not _resource_telemetry_valid(
                segment,
                expected_pids=expected_pids,
                reports=[report],
                expected_physical_memory_bytes=physical,
                swap_policy=swap_policy,
            )
        ):
            return False

    first_report, second_report = ordered_reports
    first_samples = segments[0]["samples"]
    second_samples = segments[1]["samples"]
    if (
        first_report["ended_monotonic_ns"]
        > second_report["barrier_release_monotonic_ns"]
        or first_samples[-1]["monotonic_ns"] >= second_samples[0]["monotonic_ns"]
    ):
        return False

    combined_samples = [
        sample for segment in segments for sample in segment["samples"]
    ]
    high_water_by_slot = {
        slot: max(
            item["process_peak_rss_bytes"]
            for segment in segments
            for item in segment["worker_high_water_rss_bytes"]
            if item["slot"] == slot
        )
        for slot in range(WORKER_COUNT)
    }
    expected_high_water = [
        {
            "slot": slot,
            "pid": expected_pids[slot],
            "process_peak_rss_bytes": high_water_by_slot[slot],
        }
        for slot in range(WORKER_COUNT)
    ]
    expected_high_water_combined = sum(high_water_by_slot.values())
    expected_peak_rss = max(
        [segment["peak_combined_rss_bytes"] for segment in segments]
        + [expected_high_water_combined]
    )
    return (
        telemetry.get("samples") == combined_samples
        and telemetry.get("sample_count") == len(combined_samples)
        and telemetry.get("physical_memory_bytes") == segments[0]["physical_memory_bytes"]
        and telemetry.get("worker_high_water_rss_bytes") == expected_high_water
        and telemetry.get("high_water_combined_rss_bytes")
        == expected_high_water_combined
        and telemetry.get("peak_combined_rss_bytes") == expected_peak_rss
        and telemetry.get("peak_combined_thread_count")
        == max(segment["peak_combined_thread_count"] for segment in segments)
        and telemetry.get("swap_in_delta")
        == combined_samples[-1]["swap_in_bytes"]
        - combined_samples[0]["swap_in_bytes"]
        and telemetry.get("swap_out_delta")
        == combined_samples[-1]["swap_out_bytes"]
        - combined_samples[0]["swap_out_bytes"]
        and _seconds_match(
            telemetry.get("wave_seconds"),
            sum(float(segment["wave_seconds"]) for segment in segments),
        )
        and _seconds_match(telemetry.get("start_skew_seconds"), 0.0)
        and _seconds_match(telemetry.get("overlap_seconds"), 0.0)
        and _seconds_match(telemetry.get("solo_tail_seconds"), 0.0)
    )


def validate_preflight_attestation(value: Mapping[str, Any], output_dir: Path) -> None:
    expected_fields = {
        "schema_version",
        "kind",
        "completed_at_utc",
        "protocol_sha256",
        "wave_schedule_sha256",
        "swap_policy",
        "worker_ready",
        "worker_warmup",
        "worker_data_prime",
        "worker_restarts",
        "isolated_runs",
        "concurrent_waves",
        "worker_session_swap_telemetry",
        "measured_phase_swap_window",
        "preflight_error",
        "sequential_recovery",
        "decision",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_fields
        or value.get("schema_version") != 1
        or value.get("kind") != CAMPAIGN_KIND + "_preflight"
        or not isinstance(value.get("completed_at_utc"), str)
        or not value["completed_at_utc"]
        or value.get("protocol_sha256") != screen.protocol_sha256()
        or value.get("wave_schedule_sha256") != wave_schedule_sha256()
    ):
        raise RuntimeError("preflight report header or fields are incomplete")
    ready = value.get("worker_ready")
    warmup = value.get("worker_warmup")
    data_prime = value.get("worker_data_prime")
    recovery = value.get("sequential_recovery")
    swap_policy = _validate_swap_policy(value.get("swap_policy"))
    session_swap = value.get("worker_session_swap_telemetry")
    measured_swap = value.get("measured_phase_swap_window")
    if not _swap_session_telemetry_structurally_valid(session_swap):
        raise RuntimeError("preflight worker-session swap evidence is invalid")
    expected_decision = evaluate_preflight(value)
    if value.get("decision") != expected_decision:
        raise RuntimeError("preflight decision does not match its evidence")
    if recovery is not None:
        validated_recovery = _validate_sequential_recovery_record(recovery)
        if validated_recovery["source_swap_policy"] != swap_policy:
            raise RuntimeError("preflight recovery swap policy does not match")
    preflight_error = value.get("preflight_error")
    if preflight_error is not None:
        if measured_swap is not None and not _measured_swap_window_structurally_valid(
            measured_swap,
            session_swap,
            require_shutdown_sample=True,
            expected_start_sample_index=3,
            require_exact_shutdown_suffix=True,
        ):
            raise RuntimeError("preflight measured swap evidence is invalid")
        if (
            not isinstance(preflight_error, Mapping)
            or set(preflight_error) != {"error_type", "error"}
            or not all(
                isinstance(preflight_error[name], str) and preflight_error[name]
                for name in preflight_error
            )
            or not isinstance(ready, list)
            or not isinstance(warmup, list)
            or not isinstance(data_prime, list)
            or value.get("decision", {}).get("execution_mode")
            != "sequential_fallback"
        ):
            raise RuntimeError("incomplete preflight failure is not attested")
        return
    if (
        not isinstance(ready, list)
        or not isinstance(warmup, list)
        or not isinstance(data_prime, list)
        or len(ready) != WORKER_COUNT
        or len(warmup) != WORKER_COUNT
        or len(data_prime) != WORKER_COUNT
        or value.get("worker_restarts") is not False
    ):
        raise RuntimeError("preflight workers were not completely attested")
    measured_dispatches = _preflight_measured_dispatches(value)
    if (
        len(measured_dispatches) != 6
        or not _measured_swap_window_structurally_valid(
            measured_swap,
            session_swap,
            expected_dispatches=measured_dispatches,
            require_shutdown_sample=True,
            expected_start_sample_index=3,
            require_exact_shutdown_suffix=True,
        )
        or measured_swap.get("sample_count") != 7
        or session_swap.get("sample_count") != 11
    ):
        raise RuntimeError("preflight measured swap coverage is incomplete")
    # A policy failure is a valid, attested fallback; the evidence still has to
    # be internally consistent and the recomputed decision must reject it.
    if _swap_session_telemetry_valid(session_swap, swap_policy) != bool(
        value["decision"]["criteria"]["full_session_swap_policy"]
    ):
        raise RuntimeError("preflight full-session swap decision is inconsistent")
    ready_by_slot = {}
    scratch_roots = set()
    for item in ready:
        slot = screen.hardened._exact_int(item.get("slot"), "preflight worker slot")
        pid = screen.hardened._exact_int(item.get("pid"), "preflight worker pid")
        scratch = Path(item.get("scratch_root", "")).resolve()
        if (
            slot not in range(WORKER_COUNT)
            or slot in ready_by_slot
            or pid <= 0
            or item.get("child_cpus") != EXPECTED_CHILD_CPUS
            or item.get("start_method") != "spawn"
        ):
            raise RuntimeError("preflight worker readiness is invalid")
        try:
            scratch.relative_to((output_dir / "preflight" / "worker_scratch").resolve())
        except ValueError as exc:
            raise RuntimeError("preflight worker scratch path is not private") from exc
        ready_by_slot[slot] = item
        scratch_roots.add(str(scratch))
    if set(ready_by_slot) != set(range(WORKER_COUNT)) or len(scratch_roots) != WORKER_COUNT:
        raise RuntimeError("preflight worker slots or scratch roots overlap")
    warmed_slots = set()
    for record in warmup:
        slot = screen.hardened._exact_int(record.get("worker_slot"), "warmup slot")
        if (
            slot not in ready_by_slot
            or slot in warmed_slots
            or record.get("pid") != ready_by_slot[slot]["pid"]
        ):
            raise RuntimeError("preflight warmup worker identity changed")
        screen._validate_followon_warmup_history(
            [{key: item for key, item in record.items() if key != "worker_slot"}],
            expected_thread_count=EXPECTED_CHILD_CPUS,
            expected_latest_pid=record["pid"],
        )
        warmed_slots.add(slot)
    if warmed_slots != set(range(WORKER_COUNT)):
        raise RuntimeError("preflight warmup worker coverage is incomplete")
    expected_prime_keys = [
        _job_key(("physiochemical_protein", 0, 0), "A10"),
        _job_key(("QSAR-TID-11", 2, 2), "A10"),
    ]
    primed_slots = set()
    for record in data_prime:
        slot = screen.hardened._exact_int(
            record.get("worker_slot"), "data-prime slot"
        )
        keys = [_key_tuple(item) for item in record.get("keys", [])]
        if (
            slot not in ready_by_slot
            or slot in primed_slots
            or record.get("pid") != ready_by_slot[slot]["pid"]
            or not isinstance(record.get("completed_at_utc"), str)
            or not record["completed_at_utc"]
            or keys != expected_prime_keys
        ):
            raise RuntimeError("preflight data-prime evidence is invalid")
        primed_slots.add(slot)
    if primed_slots != set(range(WORKER_COUNT)):
        raise RuntimeError("preflight data-prime worker coverage is incomplete")
    all_reports = [
        *value.get("isolated_runs", []),
        *[
            report
            for wave in value.get("concurrent_waves", [])
            for report in wave.get("reports", [])
        ],
    ]
    if any(
        report.get("pid") != ready_by_slot[report.get("slot")]["pid"]
        for report in all_reports
    ):
        raise RuntimeError("preflight report worker identity changed")
    expected_pids = {slot: item["pid"] for slot, item in ready_by_slot.items()}
    expected_physical = _physical_memory_bytes()
    for report in value.get("isolated_runs", []):
        if (
            not _dispatch_timing_valid([report], report.get("telemetry"))
            or not _resource_telemetry_structurally_valid(
                report.get("telemetry"),
                expected_pids=expected_pids,
                reports=[report],
                expected_physical_memory_bytes=expected_physical,
            )
        ):
            raise RuntimeError("isolated preflight timing or resource evidence is invalid")
    for wave in value.get("concurrent_waves", []):
        reports = wave.get("reports", [])
        if (
            not _dispatch_timing_valid(
                reports, wave.get("telemetry"), wrapper=wave
            )
            or not _resource_telemetry_structurally_valid(
                wave.get("telemetry"),
                expected_pids=expected_pids,
                reports=reports,
                expected_physical_memory_bytes=expected_physical,
            )
        ):
            raise RuntimeError("concurrent preflight timing or resource evidence is invalid")


def _validate_completed_wave(
    reports: Sequence[Mapping[str, Any]],
    telemetry: Mapping[str, Any],
    *,
    execution_mode: str,
    wave_index: int,
    swap_policy: str = DEFAULT_SWAP_POLICY,
) -> None:
    if len(reports) != 2:
        raise RuntimeError(f"wave {wave_index} did not return two results")
    for report in reports:
        elapsed = float(report.get("elapsed_seconds", math.inf))
        if (
            report.get("status") != "ok"
            or report.get("result_count") != 1
            or report.get("child_count") != 8
            or report.get("deadline_hit") is not False
            or not math.isfinite(elapsed)
            or not 0.0 < elapsed < TIME_LIMIT_SECONDS
            or (
                report.get("key", {}).get("arm") == "A10"
                and report.get("auto_candidate_fit_count") != 24
            )
        ):
            raise RuntimeError(
                f"wave {wave_index} contains an incomplete or deadline-hit result"
            )
    expected_pids = {
        report.get("slot"): report.get("pid") for report in reports
    }
    wave_seconds = float(telemetry.get("wave_seconds", math.inf))
    if (
        not _resource_telemetry_valid(
            telemetry,
            expected_pids=expected_pids,
            reports=reports,
            swap_policy=swap_policy,
        )
        or not math.isfinite(wave_seconds)
        or wave_seconds <= 0.0
    ):
        raise RuntimeError(f"wave {wave_index} violated the resource contract")
    if execution_mode == "concurrent":
        skew = float(telemetry.get("start_skew_seconds", math.inf))
        if (
            not _dispatch_timing_valid(reports, telemetry)
            or not math.isfinite(skew)
            or not 0.0 <= skew <= 1.0
            or wave_seconds >= TIME_LIMIT_SECONDS
            or float(telemetry.get("overlap_seconds", 0.0)) <= 0.0
        ):
            raise RuntimeError(f"wave {wave_index} violated the concurrent barrier")
    elif execution_mode == "sequential_fallback":
        if telemetry.get("overlap_seconds") != 0.0:
            raise RuntimeError(f"wave {wave_index} overlapped in sequential mode")
    else:
        raise RuntimeError(f"wave {wave_index} has an invalid execution mode")


def run_preflight(
    output_dir: Path,
    *,
    sequential_recovery: Mapping[str, Any] | None = None,
    owner_session: Mapping[str, Any] | None = None,
    swap_policy: str = DEFAULT_SWAP_POLICY,
) -> dict[str, Any]:
    """Compare isolated and concurrent behavior in a non-reusable namespace."""
    root = output_dir / "preflight"
    swap_policy = _validate_swap_policy(swap_policy)
    if root.exists():
        raise RuntimeError(f"preflight namespace already exists: {root}")
    root.mkdir(parents=True)
    workers = []
    ready = []
    warmup = []
    data_prime = []
    isolated_runs = []
    concurrent_waves = []
    preflight_error = None
    owner_binding_error = None
    owner_workers_bound = False
    session_swap = _new_swap_session_telemetry()
    measured_swap = None
    try:
        worker_root = root / "worker_scratch"
        _create_private_worker_root(worker_root, output_dir=output_dir)
        workers, ready = _start_workers(worker_root)
        if owner_session is not None:
            try:
                bind_owner_workers(owner_session, "preflight", workers)
                owner_workers_bound = True
            except Exception as exc:
                owner_binding_error = exc
                raise
        _checkpoint_swap_session(session_swap, swap_policy)
        warmup = _warm_workers(workers)
        _checkpoint_swap_session(session_swap, swap_policy)
        data_prime = _prime_preflight_workers(workers)
        measured_swap = _start_measured_swap_window(
            session_swap, swap_policy=swap_policy
        )
        protein = ("physiochemical_protein", 0, 0)
        qsar = ("QSAR-TID-11", 2, 2)
        isolated_plan = [
            (0, _job_key(protein, "A10")),
            (0, _job_key(protein, "B10")),
            (1, _job_key(qsar, "A10")),
            (1, _job_key(qsar, "B10")),
        ]
        for index, (slot, key) in enumerate(isolated_plan):
            reports, telemetry = _dispatch_runs(
                workers,
                [(slot, key, root / "isolated" / f"run-{index}")],
                label=f"preflight-isolated-{index}",
            )
            _checkpoint_measured_swap_window(
                session_swap,
                measured_swap,
                label=f"preflight-isolated-{index}",
                resource_telemetry=telemetry,
                reports=reports,
                swap_policy=swap_policy,
            )
            isolated_runs.append({**reports[0], "telemetry": telemetry})
        concurrent_plan = [
            ((0, _job_key(protein, "A10")), (1, _job_key(qsar, "B10"))),
            ((0, _job_key(protein, "B10")), (1, _job_key(qsar, "A10"))),
        ]
        for index, pair in enumerate(concurrent_plan):
            reports, telemetry = _dispatch_runs(
                workers,
                [
                    (slot, key, root / "concurrent" / f"wave-{index}")
                    for slot, key in pair
                ],
                label=f"preflight-concurrent-{index}",
            )
            _checkpoint_measured_swap_window(
                session_swap,
                measured_swap,
                label=f"preflight-concurrent-{index}",
                resource_telemetry=telemetry,
                reports=reports,
                swap_policy=swap_policy,
            )
            concurrent_waves.append(
                {
                    "wave_index": index,
                    "wave_seconds": telemetry["wave_seconds"],
                    "start_skew_seconds": telemetry["start_skew_seconds"],
                    "reports": reports,
                    "telemetry": telemetry,
                }
            )
    except Exception as exc:
        preflight_error = {
            "error_type": type(exc).__name__,
            "error": str(exc) or type(exc).__name__,
        }
    finally:
        shutdown_error = None
        try:
            _stop_workers(
                workers,
                force=preflight_error is not None or sys.exc_info()[0] is not None,
            )
        except Exception as exc:
            shutdown_error = exc
        if shutdown_error is None and owner_workers_bound:
            try:
                mark_owner_workers_quiesced(owner_session, "preflight")
            except Exception as exc:
                shutdown_error = exc
        try:
            _append_swap_session_sample(session_swap)
        finally:
            if shutdown_error is not None:
                detail = str(shutdown_error) or type(shutdown_error).__name__
                raise RuntimeError(
                    "preflight worker shutdown could not be confirmed; "
                    f"refusing fallback or production: {detail}"
                ) from shutdown_error
    if owner_binding_error is not None:
        raise RuntimeError(
            "preflight workers could not be durably bound before commands"
        ) from owner_binding_error
    report = {
        "schema_version": 1,
        "kind": CAMPAIGN_KIND + "_preflight",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": screen.protocol_sha256(),
        "wave_schedule_sha256": wave_schedule_sha256(),
        "swap_policy": swap_policy,
        "worker_ready": ready,
        "worker_warmup": warmup,
        "worker_data_prime": data_prime,
        "worker_restarts": preflight_error is not None
        or len(ready) != WORKER_COUNT
        or any(
            worker["process"].pid != ready[worker["slot"]]["pid"]
            for worker in workers
        ),
        "isolated_runs": isolated_runs,
        "concurrent_waves": concurrent_waves,
        "worker_session_swap_telemetry": session_swap,
        "measured_phase_swap_window": measured_swap,
        "preflight_error": preflight_error,
        "sequential_recovery": (
            _validate_sequential_recovery_record(sequential_recovery)
            if sequential_recovery is not None
            else None
        ),
    }
    report["decision"] = evaluate_preflight(report)
    validate_preflight_attestation(report, output_dir)
    screen.hardened._atomic_write_json(
        output_dir / PREFLIGHT_REPORT_FILENAME, report
    )
    return report


def execution_grid_payload(
    execution_mode: str, swap_policy: str = DEFAULT_SWAP_POLICY
) -> dict[str, Any]:
    if execution_mode not in {"concurrent", "sequential_fallback"}:
        raise ValueError(f"invalid execution mode: {execution_mode}")
    swap_policy = _validate_swap_policy(swap_policy)
    return {
        "schema_version": 1,
        "kind": CAMPAIGN_KIND + "_execution_grid",
        "execution_mode": execution_mode,
        "swap_policy": swap_policy,
        "timing_admissible": swap_policy == SWAP_POLICY_STRICT,
        "worker_count": WORKER_COUNT,
        "start_method": "spawn",
        "configured_child_cpus": EXPECTED_CHILD_CPUS,
        "wave_schedule_sha256": wave_schedule_sha256(),
        "waves": expected_wave_schedule(),
        "sequential_fallback_order": [
            item["key"]
            for wave in expected_wave_schedule()
            for item in sorted(wave["jobs"], key=lambda value: value["worker_slot"])
        ],
    }


def execution_grid_sha256(
    execution_mode: str, swap_policy: str = DEFAULT_SWAP_POLICY
) -> str:
    return hashlib.sha256(
        screen.hardened._canonical_json(
            execution_grid_payload(execution_mode, swap_policy)
        )
    ).hexdigest()


def build_run_manifest(
    *,
    output_dir: Path,
    source: Mapping[str, Any],
    ordering: Mapping[str, Mapping[str, int]],
    execution_mode: str,
    preflight_sha256: str,
    reused_evidence: Mapping[str, Any],
    swap_policy: str = DEFAULT_SWAP_POLICY,
) -> dict[str, Any]:
    swap_policy = _validate_swap_policy(swap_policy)
    manifest = screen.build_run_manifest(
        output_dir=output_dir,
        source=source,
        resolved_child_num_cpus=EXPECTED_CHILD_CPUS,
        ordering=ordering,
    )
    manifest.update(
        {
            "execution_mode": execution_mode,
            "swap_policy": swap_policy,
            "timing_admissible": swap_policy == SWAP_POLICY_STRICT,
            "worker_count": WORKER_COUNT,
            "start_method": "spawn",
            "wave_count": EXPECTED_WAVES,
            "wave_schedule_sha256": wave_schedule_sha256(),
            "execution_grid_sha256": execution_grid_sha256(
                execution_mode, swap_policy
            ),
            "preflight_report_sha256": preflight_sha256,
            "reused_evidence": dict(reused_evidence),
        }
    )
    return manifest


def write_or_validate_manifest(
    output_dir: Path, manifest: Mapping[str, Any], *, resume: bool
) -> dict[str, Any]:
    path = output_dir / screen.MANIFEST_FILENAME
    if not resume:
        if path.exists():
            raise RuntimeError("run manifest already exists")
        screen.hardened._atomic_write_json(path, manifest)
        return dict(manifest)
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("could not read existing shootout manifest") from exc
    stable = set(manifest).difference({"created_at_utc"})
    mismatches = [name for name in sorted(stable) if existing.get(name) != manifest.get(name)]
    if mismatches:
        raise RuntimeError(
            "resume manifest does not match the accuracy shootout: "
            + ", ".join(mismatches)
        )
    return existing


def _load_preflight_report(
    output_dir: Path, swap_policy: str = DEFAULT_SWAP_POLICY
) -> tuple[dict[str, Any], str]:
    swap_policy = _validate_swap_policy(swap_policy)
    path = output_dir / PREFLIGHT_REPORT_FILENAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("could not read the preflight report") from exc
    if (
        value.get("kind") != CAMPAIGN_KIND + "_preflight"
        or value.get("protocol_sha256") != screen.protocol_sha256()
        or value.get("wave_schedule_sha256") != wave_schedule_sha256()
        or value.get("swap_policy") != swap_policy
        or value.get("decision") != evaluate_preflight(value)
    ):
        raise RuntimeError("preflight report does not match the frozen protocol")
    validate_preflight_attestation(value, output_dir)
    return value, screen.hardened._sha256_file(path)


def _require_confined_archive_source(
    path: Path, *, output_dir: Path, field: str
) -> bool:
    """Reject symlinked/non-directory parents before any resume mutation."""
    root = output_dir.resolve(strict=True)
    try:
        relative = path.relative_to(output_dir)
    except ValueError as exc:
        raise RuntimeError(f"{field} is outside the campaign: {path}") from exc
    current = output_dir
    for part in relative.parts[:-1]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise RuntimeError(f"could not inspect {field} parent: {current}") from exc
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{field} parent is not a real directory: {current}")
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{field} resolves outside the campaign: {path}") from exc
    return screen.hardened._require_regular_archive_source(path, field)


def _require_confined_directory_if_present(
    path: Path, *, output_dir: Path, field: str
) -> bool:
    """Validate an existing directory chain without following any symlink."""
    root = output_dir.resolve(strict=True)
    try:
        relative = path.relative_to(output_dir)
    except ValueError as exc:
        raise RuntimeError(f"{field} is outside the campaign: {path}") from exc
    current = output_dir
    present = True
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            present = False
            break
        except OSError as exc:
            raise RuntimeError(f"could not inspect {field}: {current}") from exc
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{field} is not a real directory: {current}")
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{field} resolves outside the campaign: {path}") from exc
    return present


def _create_private_worker_root(root: Path, *, output_dir: Path) -> None:
    """Create a new worker root component-by-component without following links."""
    if _require_confined_directory_if_present(
        root, output_dir=output_dir, field="worker scratch root"
    ):
        raise RuntimeError(f"worker scratch root already exists: {root}")
    relative = root.relative_to(output_dir)
    current = output_dir
    for part in relative.parts:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            metadata = current.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(
                    f"worker scratch parent is not a real directory: {current}"
                )
    for slot in range(WORKER_COUNT):
        private = root / f"worker-{slot}"
        private.mkdir()
        metadata = private.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"worker scratch path is not private: {private}")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_process_identity(pid: int) -> dict[str, int] | None:
    """Return a PID-reuse-safe live-process identity, or ``None`` if dead."""
    import psutil

    try:
        process = psutil.Process(pid)
        created_us = int(round(process.create_time() * 1_000_000))
        status = process.status()
        if status == psutil.STATUS_ZOMBIE or not process.is_running():
            return None
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None
    except (psutil.AccessDenied, OSError) as exc:
        raise RuntimeError(f"could not inspect process identity for PID {pid}") from exc
    if created_us <= 0:
        raise RuntimeError(f"process identity has an invalid start time for PID {pid}")
    return {"pid": int(pid), "create_time_us": created_us}


def _validate_process_identity(value: Any, field: str) -> dict[str, int]:
    identity = _as_mapping(value, field)
    if set(identity) != {"pid", "create_time_us"}:
        raise RuntimeError(f"{field} fields are incomplete")
    pid = screen.hardened._exact_int(identity["pid"], f"{field} PID")
    created_us = screen.hardened._exact_int(
        identity["create_time_us"], f"{field} process start"
    )
    if pid <= 0 or created_us <= 0:
        raise RuntimeError(f"{field} is invalid")
    return {"pid": pid, "create_time_us": created_us}


def _process_identity_is_live(value: Mapping[str, Any]) -> bool:
    identity = _validate_process_identity(value, "recorded process identity")
    observed = _read_process_identity(identity["pid"])
    # A live process with the same PID but a different start time is PID reuse,
    # not the prior campaign process.
    return observed == identity


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_owner_state(value: Any, output_dir: Path) -> dict[str, Any]:
    """Validate the durable history without consulting current PID liveness."""
    state = dict(_as_mapping(value, "owner session state"))
    if set(state) != {
        "schema_version",
        "kind",
        "output_dir",
        "protocol_sha256",
        "sessions",
    }:
        raise RuntimeError("owner session state fields are incomplete")
    if (
        state["schema_version"] != 1
        or state["kind"] != OWNER_STATE_KIND
        or state["output_dir"] != str(output_dir.resolve())
        or state["protocol_sha256"] != screen.protocol_sha256()
        or not isinstance(state["sessions"], list)
        or not state["sessions"]
    ):
        raise RuntimeError("owner session state does not match the campaign")
    session_ids = set()
    for session_index, raw_session in enumerate(state["sessions"]):
        session = _as_mapping(raw_session, f"owner session {session_index}")
        if set(session) != {
            "session_id",
            "started_at_utc",
            "finalized_at_utc",
            "state",
            "phase",
            "terminal_status",
            "parent",
            "worker_cohorts",
            "execution_mode",
            "manifest_sha256",
            "terminal_artifact",
        }:
            raise RuntimeError("owner session fields are incomplete")
        session_id = session["session_id"]
        if (
            not isinstance(session_id, str)
            or len(session_id) != 32
            or any(character not in "0123456789abcdef" for character in session_id)
            or session_id in session_ids
            or not isinstance(session["started_at_utc"], str)
            or not session["started_at_utc"]
        ):
            raise RuntimeError("owner session identity is invalid")
        session_ids.add(session_id)
        _validate_process_identity(session["parent"], "owner parent")
        cohorts = session["worker_cohorts"]
        if not isinstance(cohorts, list):
            raise RuntimeError("owner worker cohorts must be a list")
        cohort_phases = set()
        for cohort_index, raw_cohort in enumerate(cohorts):
            cohort = _as_mapping(raw_cohort, f"owner worker cohort {cohort_index}")
            if set(cohort) != {
                "phase",
                "bound_at_utc",
                "quiesced_at_utc",
                "workers",
            }:
                raise RuntimeError("owner worker cohort fields are incomplete")
            phase = cohort["phase"]
            if (
                phase not in {"preflight", "production"}
                or phase in cohort_phases
                or not isinstance(cohort["bound_at_utc"], str)
                or not cohort["bound_at_utc"]
                or (
                    cohort["quiesced_at_utc"] is not None
                    and (
                        not isinstance(cohort["quiesced_at_utc"], str)
                        or not cohort["quiesced_at_utc"]
                    )
                )
                or not isinstance(cohort["workers"], list)
                or len(cohort["workers"]) != WORKER_COUNT
            ):
                raise RuntimeError("owner worker cohort is invalid")
            cohort_phases.add(phase)
            slots = set()
            process_identities = set()
            for raw_worker in cohort["workers"]:
                worker = _as_mapping(raw_worker, "owner worker identity")
                if set(worker) != {"slot", "pid", "create_time_us"}:
                    raise RuntimeError("owner worker identity fields are incomplete")
                slot = screen.hardened._exact_int(worker["slot"], "owner worker slot")
                if slot not in range(WORKER_COUNT) or slot in slots:
                    raise RuntimeError("owner worker slots are invalid")
                slots.add(slot)
                process_identity = _validate_process_identity(
                    {"pid": worker["pid"], "create_time_us": worker["create_time_us"]},
                    "owner worker",
                )
                identity_key = (
                    process_identity["pid"],
                    process_identity["create_time_us"],
                )
                if identity_key in process_identities:
                    raise RuntimeError("owner worker process identities are duplicated")
                process_identities.add(identity_key)
            if slots != set(range(WORKER_COUNT)):
                raise RuntimeError("owner worker cohort is incomplete")
        state_name = session["state"]
        phase = session["phase"]
        status = session["terminal_status"]
        finalized = session["finalized_at_utc"]
        if state_name == "active":
            if (
                phase not in {"preflight", "resume_validation", "production"}
                or finalized is not None
                or status is not None
                or session["terminal_artifact"] is not None
            ):
                raise RuntimeError("active owner session is inconsistent")
        elif state_name == "finalized":
            if (
                phase != "terminal"
                or not isinstance(finalized, str)
                or not finalized
                or status
                not in {
                    "abandoned_after_crash",
                    "completed",
                    "interrupted",
                    "invalid",
                    "preflight_only",
                }
            ):
                raise RuntimeError("finalized owner session is inconsistent")
        else:
            raise RuntimeError("owner session state is invalid")
        execution_mode = session["execution_mode"]
        manifest_sha256 = session["manifest_sha256"]
        if (execution_mode is None) != (manifest_sha256 is None):
            raise RuntimeError("owner manifest binding is incomplete")
        if execution_mode is not None and (
            execution_mode not in {"concurrent", "sequential_fallback"}
            or not _valid_sha256(manifest_sha256)
        ):
            raise RuntimeError("owner manifest binding is invalid")
        if (
            (phase == "production" or "production" in cohort_phases)
            and execution_mode is None
        ):
            raise RuntimeError("production owner session lacks its manifest binding")
        if phase in {"preflight", "resume_validation"} and execution_mode is not None:
            raise RuntimeError("pre-production owner session has a manifest binding")
        if status in {"completed", "interrupted"} and execution_mode is None:
            raise RuntimeError("terminal production owner lacks its manifest binding")
        if status == "preflight_only" and execution_mode is not None:
            raise RuntimeError("preflight-only owner unexpectedly binds a manifest")
        artifact = session["terminal_artifact"]
        if artifact is not None:
            artifact = _as_mapping(artifact, "owner terminal artifact")
            if (
                set(artifact) != {"path", "sha256", "size_bytes"}
                or artifact["path"]
                not in {
                    INVALID_ATTEMPT_FILENAME,
                    screen.COMPLETION_ATTESTATION_FILENAME,
                }
                or not _valid_sha256(artifact["sha256"])
                or screen.hardened._exact_int(
                    artifact["size_bytes"], "owner terminal artifact size"
                )
                <= 0
            ):
                raise RuntimeError("owner terminal artifact is invalid")
        if status == "completed" and (
            artifact is None
            or artifact["path"] != screen.COMPLETION_ATTESTATION_FILENAME
        ):
            raise RuntimeError("completed owner session lacks its attestation")
        if status == "invalid" and artifact is not None and (
            artifact["path"] != INVALID_ATTEMPT_FILENAME
        ):
            raise RuntimeError("invalid owner session has the wrong terminal artifact")
        if status not in {"completed", "invalid"} and artifact is not None:
            raise RuntimeError("nonterminal owner status unexpectedly binds an artifact")
    if any(
        session["state"] != "finalized" for session in state["sessions"][:-1]
    ):
        raise RuntimeError("only the latest owner session may remain active")
    return state


def _read_owner_state(path: Path, output_dir: Path) -> dict[str, Any]:
    if not _require_confined_archive_source(
        path, output_dir=output_dir, field="owner session state"
    ):
        raise RuntimeError("owner session state is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("could not read owner session state") from exc
    return validate_owner_state(value, output_dir)


def _assert_owner_lock_handle(handle: Mapping[str, Any]) -> None:
    if handle.get("released"):
        raise RuntimeError("owner lock has already been released")
    descriptor = screen.hardened._exact_int(handle["lock_fd"], "owner lock fd")
    path = Path(handle["lock_path"])
    try:
        opened = os.fstat(descriptor)
        current = path.lstat()
    except OSError as exc:
        raise RuntimeError("owner lock handle is no longer stable") from exc
    if (
        not stat.S_ISREG(opened.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
    ):
        raise RuntimeError("owner lock inode changed during the invocation")


def _write_owner_state(handle: Mapping[str, Any], value: Mapping[str, Any]) -> None:
    _assert_owner_lock_handle(handle)
    output_dir = Path(handle["output_dir"])
    path = Path(handle["state_path"])
    validate_owner_state(value, output_dir)
    screen.hardened._atomic_write_json(path, value)
    _fsync_directory(output_dir)


def _open_owner_lock(output_dir: Path) -> tuple[int, Path]:
    import fcntl

    path = output_dir / OWNER_LOCK_FILENAME
    try:
        prior = path.lstat()
    except FileNotFoundError:
        prior = None
    except OSError as exc:
        raise RuntimeError("could not inspect owner lock") from exc
    if prior is not None and (
        not stat.S_ISREG(prior.st_mode) or stat.S_ISLNK(prior.st_mode)
    ):
        raise RuntimeError("owner lock must be a real regular file")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise RuntimeError("could not open owner lock without following links") from exc
    try:
        opened = os.fstat(descriptor)
        current = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise RuntimeError("owner lock path changed while opening")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            raise RuntimeError("campaign namespace is owned by another process") from exc
        if prior is None:
            _fsync_directory(output_dir)
        return descriptor, path
    except BaseException:
        os.close(descriptor)
        raise


def _owner_current_session(
    handle: Mapping[str, Any], *, require_active: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    _assert_owner_lock_handle(handle)
    output_dir = Path(handle["output_dir"])
    state = _read_owner_state(Path(handle["state_path"]), output_dir)
    session = state["sessions"][-1]
    if session["session_id"] != handle["session_id"] or (
        require_active and session["state"] != "active"
    ):
        raise RuntimeError("owner lock no longer names this invocation")
    return state, session


def acquire_owner_session(
    output_dir: Path, *, resume: bool, phase: str
) -> dict[str, Any]:
    """Exclusively claim a campaign namespace before any worker is started."""
    if phase not in {"preflight", "resume_validation"}:
        raise RuntimeError("initial owner phase is invalid")
    output_dir = output_dir.resolve(strict=True)
    descriptor, lock_path = _open_owner_lock(output_dir)
    state_path = output_dir / OWNER_STATE_FILENAME
    handle: dict[str, Any] = {
        "lock_fd": descriptor,
        "lock_path": str(lock_path),
        "state_path": str(state_path),
        "output_dir": str(output_dir.resolve()),
        "session_id": "",
        "released": False,
    }
    try:
        state_exists = _require_confined_archive_source(
            state_path, output_dir=output_dir, field="owner session state"
        )
        if resume:
            if not state_exists:
                raise RuntimeError("resume requires a durable prior owner session")
            state = _read_owner_state(state_path, output_dir)
            prior = state["sessions"][-1]
            if prior["state"] == "active":
                live = []
                if _process_identity_is_live(prior["parent"]):
                    live.append(f"parent PID {prior['parent']['pid']}")
                for cohort in prior["worker_cohorts"]:
                    if cohort["quiesced_at_utc"] is not None:
                        continue
                    for worker in cohort["workers"]:
                        if _process_identity_is_live(
                            {
                                "pid": worker["pid"],
                                "create_time_us": worker["create_time_us"],
                            }
                        ):
                            live.append(
                                f"{cohort['phase']} worker slot {worker['slot']} "
                                f"PID {worker['pid']}"
                            )
                if live:
                    raise RuntimeError(
                        "prior campaign processes are still active: " + ", ".join(live)
                    )
                finalized_at = datetime.now(timezone.utc).isoformat()
                for cohort in prior["worker_cohorts"]:
                    if cohort["quiesced_at_utc"] is None:
                        cohort["quiesced_at_utc"] = finalized_at
                prior.update(
                    {
                        "state": "finalized",
                        "phase": "terminal",
                        "finalized_at_utc": finalized_at,
                        "terminal_status": "abandoned_after_crash",
                    }
                )
            elif prior["terminal_status"] not in {
                "interrupted",
                "abandoned_after_crash",
            }:
                raise RuntimeError(
                    "prior owner session is terminal and cannot be resumed: "
                    f"{prior['terminal_status']}"
                )
        else:
            if state_exists:
                raise RuntimeError("new campaign unexpectedly has owner session history")
            state = {
                "schema_version": 1,
                "kind": OWNER_STATE_KIND,
                "output_dir": str(output_dir.resolve()),
                "protocol_sha256": screen.protocol_sha256(),
                "sessions": [],
            }
        parent = _read_process_identity(os.getpid())
        if parent is None:
            raise RuntimeError("could not establish current owner process identity")
        session_id = os.urandom(16).hex()
        handle["session_id"] = session_id
        state["sessions"].append(
            {
                "session_id": session_id,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "finalized_at_utc": None,
                "state": "active",
                "phase": phase,
                "terminal_status": None,
                "parent": parent,
                "worker_cohorts": [],
                "execution_mode": None,
                "manifest_sha256": None,
                "terminal_artifact": None,
            }
        )
        _write_owner_state(handle, state)
        return handle
    except BaseException:
        release_owner_session(handle)
        raise


def release_owner_session(handle: Mapping[str, Any]) -> None:
    import fcntl

    if handle.get("released"):
        return
    descriptor = screen.hardened._exact_int(handle["lock_fd"], "owner lock fd")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
        if isinstance(handle, dict):
            handle["released"] = True


def bind_owner_workers(
    handle: Mapping[str, Any], phase: str, workers: Sequence[Mapping[str, Any]]
) -> None:
    """Durably bind ready workers before issuing any worker command."""
    if phase not in {"preflight", "production"}:
        raise RuntimeError("owner worker phase is invalid")
    state, session = _owner_current_session(handle)
    if session["phase"] != phase or any(
        cohort["phase"] == phase for cohort in session["worker_cohorts"]
    ):
        raise RuntimeError("owner worker cohort does not match the active phase")
    if len(workers) != WORKER_COUNT:
        raise RuntimeError("owner worker cohort has the wrong size")
    identities = []
    for worker in sorted(workers, key=lambda item: item["slot"]):
        slot = screen.hardened._exact_int(worker["slot"], "owner worker slot")
        identity = _read_process_identity(worker["process"].pid)
        if identity is None:
            raise RuntimeError(f"worker slot {slot} exited before owner binding")
        identities.append({"slot": slot, **identity})
    session["worker_cohorts"].append(
        {
            "phase": phase,
            "bound_at_utc": datetime.now(timezone.utc).isoformat(),
            "quiesced_at_utc": None,
            "workers": identities,
        }
    )
    _write_owner_state(handle, state)


def mark_owner_workers_quiesced(handle: Mapping[str, Any], phase: str) -> None:
    state, session = _owner_current_session(handle)
    matches = [
        cohort for cohort in session["worker_cohorts"] if cohort["phase"] == phase
    ]
    if len(matches) != 1 or matches[0]["quiesced_at_utc"] is not None:
        raise RuntimeError("owner worker cohort cannot be marked quiescent")
    matches[0]["quiesced_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_owner_state(handle, state)


def bind_owner_manifest(
    handle: Mapping[str, Any], *, execution_mode: str, manifest_path: Path
) -> None:
    state, session = _owner_current_session(handle)
    if execution_mode not in {"concurrent", "sequential_fallback"} or any(
        cohort["quiesced_at_utc"] is None for cohort in session["worker_cohorts"]
    ):
        raise RuntimeError("owner session cannot enter production")
    session["phase"] = "production"
    session["execution_mode"] = execution_mode
    session["manifest_sha256"] = screen.hardened._sha256_file(manifest_path)
    _write_owner_state(handle, state)


def finalize_owner_session(handle: Mapping[str, Any], status: str) -> None:
    if status not in {"completed", "interrupted", "invalid", "preflight_only"}:
        raise RuntimeError("owner terminal status is invalid")
    state, session = _owner_current_session(handle)
    if any(
        cohort["quiesced_at_utc"] is None for cohort in session["worker_cohorts"]
    ):
        raise RuntimeError("cannot finalize owner session with active workers")
    output_dir = Path(handle["output_dir"])
    artifact = None
    artifact_path = None
    if status == "completed":
        artifact_path = output_dir / screen.COMPLETION_ATTESTATION_FILENAME
    elif status == "invalid" and (output_dir / INVALID_ATTEMPT_FILENAME).exists():
        artifact_path = output_dir / INVALID_ATTEMPT_FILENAME
    if artifact_path is not None:
        artifact = screen._stable_file_artifact(artifact_path, output_dir)
    finalized_at = datetime.now(timezone.utc).isoformat()
    session.update(
        {
            "state": "finalized",
            "phase": "terminal",
            "finalized_at_utc": finalized_at,
            "terminal_status": status,
            "terminal_artifact": artifact,
        }
    )
    _write_owner_state(handle, state)


def validate_completed_owner_session(
    output_dir: Path, attestation: Mapping[str, Any]
) -> dict[str, Any]:
    lock_path = output_dir / OWNER_LOCK_FILENAME
    if not _require_confined_archive_source(
        lock_path, output_dir=output_dir, field="owner lock"
    ):
        raise RuntimeError("owner lock is missing")
    if lock_path.stat().st_size != 0:
        raise RuntimeError("owner lock contents are invalid")
    state = _read_owner_state(output_dir / OWNER_STATE_FILENAME, output_dir)
    session = state["sessions"][-1]
    if (
        session["state"] != "finalized"
        or session["terminal_status"] != "completed"
        or session["session_id"] != attestation.get("owner_session_id")
        or session["execution_mode"] != attestation.get("execution_mode")
        or session["manifest_sha256"] != attestation.get("manifest_sha256")
    ):
        raise RuntimeError("completed owner session does not bind the attestation")
    expected_artifact = screen._stable_file_artifact(
        output_dir / screen.COMPLETION_ATTESTATION_FILENAME, output_dir
    )
    if session["terminal_artifact"] != expected_artifact:
        raise RuntimeError("owner session completion artifact changed")
    production = [
        cohort for cohort in session["worker_cohorts"] if cohort["phase"] == "production"
    ]
    if len(production) != 1 or production[0]["quiesced_at_utc"] is None:
        raise RuntimeError("completed owner session lacks quiesced production workers")
    return state


def validate_invalid_owner_session(
    output_dir: Path, marker: Mapping[str, Any]
) -> str:
    """Prove an invalid recovery source is finalized and no longer owned."""
    import fcntl

    lock_path = output_dir / OWNER_LOCK_FILENAME
    state_path = output_dir / OWNER_STATE_FILENAME
    for path, field in (
        (lock_path, "owner lock"),
        (state_path, "owner session state"),
    ):
        if not _require_confined_archive_source(
            path, output_dir=output_dir, field=field
        ):
            raise RuntimeError(f"sequential recovery source lacks {field}")
    descriptor, _path = _open_owner_lock(output_dir)
    try:
        state = _read_owner_state(state_path, output_dir)
        session = state["sessions"][-1]
        expected_artifact = screen._stable_file_artifact(
            output_dir / INVALID_ATTEMPT_FILENAME, output_dir
        )
        if (
            session["state"] != "finalized"
            or session["terminal_status"] != "invalid"
            or session["execution_mode"] != "concurrent"
            or session["manifest_sha256"] != marker.get("manifest_sha256")
            or session["terminal_artifact"] != expected_artifact
            or any(
                cohort["quiesced_at_utc"] is None
                for cohort in session["worker_cohorts"]
            )
        ):
            raise RuntimeError(
                "sequential recovery source is not bound to a quiescent invalid owner"
            )
        return screen.hardened._sha256_file(state_path)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _validate_resume_archive_path(
    output_dir: Path, value: Any, field: str
) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{field} path must be a string")
    relative = Path(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] != "resume_invalidated"
    ):
        raise RuntimeError(f"{field} path is unsafe")
    if not _require_confined_archive_source(
        output_dir / relative, output_dir=output_dir, field=field
    ):
        raise RuntimeError(f"{field} archive is missing: {relative}")
    return value


def validate_resume_history(value: Any, output_dir: Path) -> None:
    """Validate every retained zero-start resume record and archived byte."""
    if not isinstance(value, list) or not value:
        raise RuntimeError("resume history must contain at least one record")
    schedule = expected_wave_schedule()
    all_waves = set(range(EXPECTED_WAVES))
    allowed_statuses = {
        "valid",
        "missing",
        "unreadable",
        "not_a_regular_file",
        "mismatched",
        "incomplete_or_mismatched",
        "unattested_or_changed",
        "prior_process_pickle_archived",
    }
    for record_index, raw_record in enumerate(value):
        record = _as_mapping(raw_record, f"resume record {record_index}")
        if set(record) != {
            "resumed_at_utc",
            "pid",
            "wave_schedule_sha256",
            "reusable_wave_indices",
            "pending_wave_indices",
            "invalidated_waves",
            "archived_campaign_artifacts",
        }:
            raise RuntimeError("resume record fields are incomplete")
        if (
            not isinstance(record["resumed_at_utc"], str)
            or not record["resumed_at_utc"]
            or screen.hardened._exact_int(record["pid"], "resume pid") <= 0
            or record["wave_schedule_sha256"] != wave_schedule_sha256()
            or record["reusable_wave_indices"] != []
            or record["pending_wave_indices"] != list(range(EXPECTED_WAVES))
        ):
            raise RuntimeError("resume record does not describe a full zero-start rerun")
        pending = {
            screen.hardened._exact_int(item, "pending wave index")
            for item in record["pending_wave_indices"]
        }
        if pending != all_waves:
            raise RuntimeError("resume wave partition is not exact")
        invalidated = record["invalidated_waves"]
        if not isinstance(invalidated, list):
            raise RuntimeError("invalidated waves must be a list")
        invalidated_indices = []
        archived_paths = set()
        for raw_wave in invalidated:
            wave = _as_mapping(raw_wave, "invalidated wave")
            if set(wave) != {"wave_index", "members"}:
                raise RuntimeError("invalidated wave fields are incomplete")
            wave_index = screen.hardened._exact_int(
                wave["wave_index"], "invalidated wave index"
            )
            members = wave["members"]
            if (
                wave_index not in pending
                or wave_index in invalidated_indices
                or not isinstance(members, list)
                or not 1 <= len(members) <= 2
            ):
                raise RuntimeError("invalidated wave does not match pending work")
            invalidated_indices.append(wave_index)
            expected_keys = {
                _key_tuple(item["key"]) for item in schedule[wave_index]["jobs"]
            }
            member_keys = set()
            for raw_member in members:
                member = _as_mapping(raw_member, "invalidated member")
                if set(member) != {"key", "status", "path"}:
                    raise RuntimeError("invalidated member fields are incomplete")
                key = _key_tuple(_as_mapping(member["key"], "member key"))
                path = _validate_resume_archive_path(
                    output_dir, member["path"], "invalidated result"
                )
                if (
                    key not in expected_keys
                    or key in member_keys
                    or member["key"] != _key_payload(key)
                    or member["status"] not in allowed_statuses
                    or path in archived_paths
                ):
                    raise RuntimeError("invalidated member does not match its wave")
                member_keys.add(key)
                archived_paths.add(path)
        if invalidated_indices != sorted(invalidated_indices):
            raise RuntimeError("invalidated waves are not in canonical order")
        archived = record["archived_campaign_artifacts"]
        if not isinstance(archived, list):
            raise RuntimeError("archived campaign artifacts must be a list")
        for raw_path in archived:
            path = _validate_resume_archive_path(
                output_dir, raw_path, "archived campaign artifact"
            )
            if path in archived_paths:
                raise RuntimeError("resume archive path is duplicated")
            archived_paths.add(path)


def prepare_wave_resume(
    output_dir: Path,
    jobs: Sequence[Any],
    schedule: Sequence[Mapping[str, Any]],
    *,
    resume: bool,
    execution_mode: str = "concurrent",
    swap_policy: str = DEFAULT_SWAP_POLICY,
) -> dict[str, Any]:
    """Validate metadata, archive prior-process pickles, and restart at wave zero."""
    if execution_mode not in {"concurrent", "sequential_fallback"}:
        raise RuntimeError("resume execution mode is invalid")
    swap_policy = _validate_swap_policy(swap_policy)
    validate_wave_schedule(schedule)
    lookup = _job_lookup(jobs)
    if not resume:
        return {
            "reusable_wave_indices": [],
            "pending_wave_indices": list(range(EXPECTED_WAVES)),
            "invalidated_wave_indices": [],
        }
    expected_paths = {
        str(screen._result_path(output_dir, job).relative_to(output_dir))
        for job in jobs
    }
    observed = screen._observed_regular_result_paths(output_dir)
    unexpected = set(observed).difference(expected_paths)
    if unexpected:
        raise RuntimeError(f"resume cache contains unexpected result files: {unexpected}")
    _require_confined_directory_if_present(
        output_dir / "worker_scratch",
        output_dir=output_dir,
        field="production worker scratch directory",
    )
    stale_paths = [
        output_dir / screen.COMPLETION_ATTESTATION_FILENAME,
        output_dir / screen.ANALYSIS_PAYLOAD_FILENAME,
        *(output_dir / name for name in ANALYSIS_OUTPUT_FILENAMES),
    ]
    for job in jobs:
        _require_confined_archive_source(
            screen._result_path(output_dir, job),
            output_dir=output_dir,
            field="cached shootout result",
        )
    for path in (
        *stale_paths,
        output_dir / screen.MANIFEST_FILENAME,
        output_dir / PREFLIGHT_REPORT_FILENAME,
        output_dir / WAVE_SCHEDULE_FILENAME,
        output_dir / CONCURRENCY_HISTORY_FILENAME,
        output_dir / screen.WARMUP_HISTORY_FILENAME,
        output_dir / screen.RESUME_HISTORY_FILENAME,
        output_dir / OWNER_LOCK_FILENAME,
        output_dir / OWNER_STATE_FILENAME,
    ):
        _require_confined_archive_source(
            path,
            output_dir=output_dir,
            field="shootout campaign artifact",
        )
    journal_path = output_dir / CONCURRENCY_HISTORY_FILENAME
    if journal_path.exists():
        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("could not read concurrency history before resume") from exc
        if (
            not isinstance(journal, Mapping)
            or journal.get("execution_mode") != execution_mode
            or journal.get("swap_policy") != swap_policy
            or journal.get("wave_schedule_sha256") != wave_schedule_sha256()
            or not isinstance(journal.get("entries"), list)
        ):
            raise RuntimeError("concurrency history is incompatible with resume")
        validate_concurrency_history(
            journal,
            execution_mode=execution_mode,
            output_dir=output_dir,
            require_complete=False,
            swap_policy=swap_policy,
            # A hard crash cannot append the post-shutdown sample.  This journal
            # is validated only so it can be archived for a zero-start rerun.
            require_shutdown_sample=False,
        )
    prior_resume_history = []
    history_path = output_dir / screen.RESUME_HISTORY_FILENAME
    if history_path.exists():
        try:
            prior_resume_history = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("could not read resume history before mutation") from exc
        if not isinstance(prior_resume_history, list):
            raise RuntimeError("resume history must be a list")
        validate_resume_history(prior_resume_history, output_dir)
    warmup_path = output_dir / screen.WARMUP_HISTORY_FILENAME
    if warmup_path.exists():
        try:
            prior_warmup = json.loads(warmup_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("could not read warmup history before mutation") from exc
        if not isinstance(prior_warmup, list):
            raise RuntimeError("warmup history must be a list")
        validate_warmup_sessions(
            prior_warmup,
            execution_mode=execution_mode,
            output_dir=output_dir,
        )
        if len(prior_warmup) > len(prior_resume_history) + 1:
            raise RuntimeError("warmup history exceeds recorded campaign invocations")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_root = output_dir / "resume_invalidated" / timestamp
    archived_campaign_artifacts = []
    if journal_path.exists():
        destination = archive_root / journal_path.name
        screen._prepare_archive_destination(destination, output_dir=output_dir)
        os.replace(journal_path, destination)
        archived_campaign_artifacts.append(
            str(destination.relative_to(output_dir))
        )
    screen.hardened._atomic_write_json(
        journal_path, _empty_concurrency_history(execution_mode, swap_policy)
    )
    reusable: list[int] = []
    pending = list(range(EXPECTED_WAVES))
    invalidated = []
    for wave in schedule:
        wave_index = screen.hardened._exact_int(wave["wave_index"], "wave index")
        members = []
        for item in wave["jobs"]:
            key = _key_tuple(item["key"])
            job = lookup[key]
            path = screen._result_path(output_dir, job)
            issue = "prior_process_pickle_archived" if path.exists() else "missing"
            members.append((key, path, issue))
        existing = [member for member in members if member[1].exists() or member[1].is_symlink()]
        if not existing:
            continue
        archived = []
        for key, source, issue in existing:
            relative = source.relative_to(output_dir)
            destination = archive_root / relative
            screen._prepare_archive_destination(destination, output_dir=output_dir)
            os.replace(source, destination)
            archived.append(
                {
                    "key": _key_payload(key),
                    "status": issue or "valid",
                    "path": str(destination.relative_to(output_dir)),
                }
            )
        invalidated.append({"wave_index": wave_index, "members": archived})
    for source in stale_paths:
        if not source.exists():
            continue
        destination = archive_root / source.name
        screen._prepare_archive_destination(destination, output_dir=output_dir)
        os.replace(source, destination)
        archived_campaign_artifacts.append(str(destination.relative_to(output_dir)))
    record = {
        "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "wave_schedule_sha256": wave_schedule_sha256(),
        "reusable_wave_indices": reusable,
        "pending_wave_indices": pending,
        "invalidated_waves": invalidated,
        "archived_campaign_artifacts": archived_campaign_artifacts,
    }
    history = prior_resume_history
    history.append(record)
    validate_resume_history(history, output_dir)
    screen.hardened._atomic_write_json(history_path, history)
    return {
        "reusable_wave_indices": reusable,
        "pending_wave_indices": pending,
        "invalidated_wave_indices": [item["wave_index"] for item in invalidated],
    }


def _production_measured_dispatches(
    entries: Sequence[Mapping[str, Any]], execution_mode: str
) -> list[tuple[str, Mapping[str, Any], Sequence[Mapping[str, Any]]]]:
    dispatches: list[
        tuple[str, Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ] = []
    for entry in entries:
        wave_index = screen.hardened._exact_int(
            entry.get("wave_index"), "measured production wave index"
        )
        if execution_mode == "concurrent":
            telemetry = entry.get("telemetry")
            if not isinstance(telemetry, Mapping):
                raise RuntimeError("production dispatch telemetry is invalid")
            reports = entry.get("reports")
            if not isinstance(reports, list):
                raise RuntimeError("production dispatch reports are invalid")
            dispatches.append(
                (f"production-wave-{wave_index}", telemetry, reports)
            )
        elif execution_mode == "sequential_fallback":
            details = entry.get("mode_details")
            segments = details.get("segment_telemetry") if isinstance(details, Mapping) else None
            if (
                not isinstance(segments, list)
                or len(segments) != WORKER_COUNT
                or any(not isinstance(item, Mapping) for item in segments)
            ):
                raise RuntimeError("sequential production dispatches are incomplete")
            dispatches.extend(
                (
                    f"production-wave-{wave_index}-job-{ordinal}",
                    telemetry,
                    [sorted(entry["reports"], key=lambda item: item["slot"])[ordinal]],
                )
                for ordinal, telemetry in enumerate(segments)
            )
        else:
            raise RuntimeError("production execution mode is invalid")
    return dispatches


def validate_concurrency_history(
    value: Any,
    *,
    execution_mode: str,
    output_dir: Path,
    require_complete: bool = True,
    swap_policy: str = DEFAULT_SWAP_POLICY,
    require_shutdown_sample: bool = True,
) -> None:
    swap_policy = _validate_swap_policy(swap_policy)
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "kind",
        "execution_mode",
        "swap_policy",
        "timing_admissible",
        "wave_schedule_sha256",
        "worker_session_swap_telemetry",
        "measured_phase_swap_window",
        "entries",
    }:
        raise RuntimeError("concurrency history fields are incomplete")
    if (
        value["schema_version"] != 1
        or value["kind"] != CAMPAIGN_KIND + "_concurrency_history"
        or value["execution_mode"] != execution_mode
        or value["swap_policy"] != swap_policy
        or value["timing_admissible"] is not (swap_policy == SWAP_POLICY_STRICT)
        or value["wave_schedule_sha256"] != wave_schedule_sha256()
        or not isinstance(value["entries"], list)
    ):
        raise RuntimeError("concurrency history header does not match")
    session_swap = value["worker_session_swap_telemetry"]
    measured_swap = value["measured_phase_swap_window"]
    if session_swap is None:
        if measured_swap is not None or value["entries"] or require_complete:
            raise RuntimeError("concurrency worker-session swap evidence is missing")
    elif not _swap_session_telemetry_valid(session_swap, swap_policy):
        raise RuntimeError("concurrency worker session observed swap I/O")
    schedule = expected_wave_schedule()
    seen = set()
    observed_order = []
    observed_pids_by_slot = {slot: set() for slot in range(WORKER_COUNT)}
    for entry in value["entries"]:
        if not isinstance(entry, Mapping) or set(entry) != {
            "wave_index",
            "dataset",
            "execution_mode",
            "reports",
            "telemetry",
            "mode_details",
        }:
            raise RuntimeError("concurrency history entry fields are incomplete")
        wave_index = screen.hardened._exact_int(entry.get("wave_index"), "wave index")
        if wave_index not in range(EXPECTED_WAVES) or wave_index in seen:
            raise RuntimeError("concurrency history has invalid wave indices")
        seen.add(wave_index)
        observed_order.append(wave_index)
        if (
            entry["dataset"] != schedule[wave_index]["dataset"]
            or entry["execution_mode"] != execution_mode
        ):
            raise RuntimeError("concurrency history entry header does not match")
        reports = entry.get("reports")
        if not isinstance(reports, list) or len(reports) != 2:
            raise RuntimeError("completed wave must contain two reports")
        expected = {
            _key_tuple(item["key"]): item["worker_slot"]
            for item in schedule[wave_index]["jobs"]
        }
        observed = {_key_tuple(item["key"]): item.get("slot") for item in reports}
        if observed != expected:
            raise RuntimeError("concurrency history wave identity does not match")
        by_slot = {report["slot"]: report for report in reports}
        for report in reports:
            key = _key_tuple(report["key"])
            expected_relative = screen.expected_result_relative_path(
                key[0], key[1], key[2], PUBLIC_TO_INTERNAL_ARM[key[3]]
            )
            path = Path(report.get("result_path", ""))
            observed_pids_by_slot[report["slot"]].add(
                screen.hardened._exact_int(report.get("pid"), "worker pid")
            )
            try:
                relative = str(path.resolve().relative_to(output_dir.resolve()))
            except ValueError as exc:
                raise RuntimeError("concurrency result path escapes output") from exc
            release = screen.hardened._exact_int(
                report.get("barrier_release_monotonic_ns"), "barrier release"
            )
            started = screen.hardened._exact_int(
                report.get("started_monotonic_ns"), "worker start"
            )
            ended = screen.hardened._exact_int(
                report.get("ended_monotonic_ns"), "worker end"
            )
            elapsed = float(report.get("elapsed_seconds", math.inf))
            slot = report["slot"]
            pid = screen.hardened._exact_int(report.get("pid"), "worker pid")
            partner = by_slot[1 - slot]
            if (
                relative != expected_relative
                or Path(report.get("result_root", "")).resolve()
                != output_dir.resolve()
                or report.get("status") != "ok"
                or report.get("type") != "result"
                or not isinstance(report.get("command_id"), str)
                or not report["command_id"]
                or report.get("result_count") != 1
                or not isinstance(report.get("result_sha256"), str)
                or len(report["result_sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in report["result_sha256"]
                )
                or not isinstance(report.get("result_size_bytes"), int)
                or report["result_size_bytes"] <= 0
                or not isinstance(report.get("process_peak_rss_bytes"), int)
                or report["process_peak_rss_bytes"] <= 0
                or report.get("child_count") != 8
                or report.get("deadline_hit") is not False
                or report.get("auto_candidate_fit_count")
                != (24 if key[3] == "A10" else 0)
                or not isinstance(report.get("behavior_sha256"), str)
                or len(report["behavior_sha256"]) != 64
                or report.get("start_method") != "spawn"
                or report.get("wave_index") != wave_index
                or report.get("wave_schedule_sha256") != wave_schedule_sha256()
                or report.get("partner_key") != partner.get("key")
                or pid <= 0
                or not release <= started <= ended
                or not math.isfinite(elapsed)
                or not 0.0 < elapsed < TIME_LIMIT_SECONDS
                or not math.isclose(
                    elapsed, (ended - started) / 1e9, rel_tol=1e-9, abs_tol=1e-9
                )
                or not math.isfinite(float(report.get("cpu_time_seconds", math.inf)))
                or float(report.get("cpu_time_seconds", -1.0)) < 0.0
            ):
                raise RuntimeError("concurrency result report does not match artifact")
        telemetry = entry["telemetry"]
        if not isinstance(telemetry, Mapping):
            raise RuntimeError("concurrency telemetry must be a mapping")
        _validate_completed_wave(
            reports,
            telemetry,
            execution_mode=execution_mode,
            wave_index=wave_index,
            swap_policy=swap_policy,
        )
        samples = telemetry.get("samples")
        if (
            not isinstance(samples, list)
            or not samples
            or telemetry.get("sample_count") != len(samples)
            or not isinstance(telemetry.get("peak_combined_thread_count"), int)
            or telemetry["peak_combined_thread_count"] < 1
        ):
            raise RuntimeError("concurrency telemetry samples are incomplete")
        if execution_mode == "concurrent":
            if (
                not _dispatch_timing_valid(reports, telemetry)
                or entry["mode_details"] != {"segments": 1}
            ):
                raise RuntimeError("concurrency barrier telemetry is inconsistent")
        elif not _sequential_fallback_telemetry_valid(
            reports,
            telemetry,
            entry["mode_details"],
            swap_policy=swap_policy,
        ):
            raise RuntimeError("sequential fallback segments are incomplete")
    expected_order = list(range(EXPECTED_WAVES if require_complete else len(observed_order)))
    if observed_order != expected_order:
        raise RuntimeError("concurrency history is not a contiguous wave prefix")
    if any(len(pids) > 1 for pids in observed_pids_by_slot.values()):
        raise RuntimeError("production worker identity changed between waves")
    if require_complete and seen != set(range(EXPECTED_WAVES)):
        raise RuntimeError("concurrency history does not cover all frozen waves")
    if session_swap is not None:
        measured_dispatches = _production_measured_dispatches(
            value["entries"], execution_mode
        )
        expected_sample_count = len(measured_dispatches) + 1
        if (
            measured_swap is None
            or not _measured_swap_window_valid(
                measured_swap,
                session_swap,
                swap_policy=swap_policy,
                expected_dispatches=measured_dispatches,
                require_shutdown_sample=require_shutdown_sample,
                expected_start_sample_index=2,
                require_exact_shutdown_suffix=(
                    require_complete and require_shutdown_sample
                ),
            )
            or measured_swap.get("sample_count") != expected_sample_count
        ):
            raise RuntimeError("production measured swap coverage is incomplete")
        if require_complete:
            frozen_sample_count = (
                EXPECTED_WAVES + 1
                if execution_mode == "concurrent"
                else EXPECTED_WAVES * WORKER_COUNT + 1
            )
            if expected_sample_count != frozen_sample_count:
                raise RuntimeError("production measured swap coverage is not exact")
            expected_session_sample_count = frozen_sample_count + (
                3 if require_shutdown_sample else 2
            )
            if session_swap.get("sample_count") != expected_session_sample_count:
                raise RuntimeError("production worker-session coverage is not exact")


def _record_warmup_session(
    output_dir: Path,
    *,
    execution_mode: str,
    ready: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> None:
    path = output_dir / screen.WARMUP_HISTORY_FILENAME
    history = []
    if path.exists():
        history = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            raise RuntimeError("warmup history must be a list")
    history.append(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution_mode": execution_mode,
            "worker_ready": list(ready),
            "worker_warmup": list(records),
        }
    )
    validate_warmup_sessions(
        history, execution_mode=execution_mode, output_dir=output_dir
    )
    screen.hardened._atomic_write_json(path, history)


def validate_warmup_sessions(
    value: Any, *, execution_mode: str, output_dir: Path
) -> None:
    if not isinstance(value, list) or not value:
        raise RuntimeError("production warmup history must be a nonempty list")
    for session in value:
        if not isinstance(session, Mapping) or set(session) != {
            "completed_at_utc",
            "execution_mode",
            "worker_ready",
            "worker_warmup",
        }:
            raise RuntimeError("production warmup session fields are incomplete")
        ready = session["worker_ready"]
        records = session["worker_warmup"]
        if (
            session["execution_mode"] != execution_mode
            or not isinstance(session["completed_at_utc"], str)
            or not isinstance(ready, list)
            or not isinstance(records, list)
            or len(ready) != WORKER_COUNT
            or len(records) != WORKER_COUNT
        ):
            raise RuntimeError("production warmup session does not match")
        ready_by_slot = {}
        scratch_roots = set()
        for item in ready:
            slot = screen.hardened._exact_int(item.get("slot"), "warmup worker slot")
            pid = screen.hardened._exact_int(item.get("pid"), "warmup worker pid")
            scratch = Path(item.get("scratch_root", "")).resolve()
            try:
                scratch.relative_to((output_dir / "worker_scratch").resolve())
            except ValueError as exc:
                raise RuntimeError("production worker scratch escapes output") from exc
            if (
                slot not in range(WORKER_COUNT)
                or slot in ready_by_slot
                or pid <= 0
                or item.get("child_cpus") != EXPECTED_CHILD_CPUS
                or item.get("start_method") != "spawn"
            ):
                raise RuntimeError("production worker readiness is invalid")
            ready_by_slot[slot] = item
            scratch_roots.add(str(scratch))
        if len(scratch_roots) != WORKER_COUNT:
            raise RuntimeError("production worker scratch roots overlap")
        for record in records:
            slot = screen.hardened._exact_int(record.get("worker_slot"), "warmup slot")
            if slot not in ready_by_slot or record.get("pid") != ready_by_slot[slot]["pid"]:
                raise RuntimeError("production warmup worker identity changed")
            screen._validate_followon_warmup_history(
                [{key: item for key, item in record.items() if key != "worker_slot"}],
                expected_thread_count=EXPECTED_CHILD_CPUS,
                expected_latest_pid=record["pid"],
            )


def _load_concurrency_history(
    output_dir: Path,
    execution_mode: str,
    invalidated: set[int],
    swap_policy: str = DEFAULT_SWAP_POLICY,
) -> dict[str, Any]:
    swap_policy = _validate_swap_policy(swap_policy)
    path = output_dir / CONCURRENCY_HISTORY_FILENAME
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping) or not isinstance(value.get("entries"), list):
            raise RuntimeError("could not resume concurrency history")
        if (
            value.get("execution_mode") != execution_mode
            or value.get("swap_policy") != swap_policy
            or value.get("wave_schedule_sha256") != wave_schedule_sha256()
        ):
            raise RuntimeError("concurrency history does not match execution mode")
        value = dict(value)
        value["entries"] = [
            item for item in value["entries"] if item.get("wave_index") not in invalidated
        ]
        validate_concurrency_history(
            value,
            execution_mode=execution_mode,
            output_dir=output_dir,
            require_complete=False,
            swap_policy=swap_policy,
        )
        return value
    return _empty_concurrency_history(execution_mode, swap_policy)


def _empty_concurrency_history(
    execution_mode: str, swap_policy: str = DEFAULT_SWAP_POLICY
) -> dict[str, Any]:
    swap_policy = _validate_swap_policy(swap_policy)
    return {
        "schema_version": 1,
        "kind": CAMPAIGN_KIND + "_concurrency_history",
        "execution_mode": execution_mode,
        "swap_policy": swap_policy,
        "timing_admissible": swap_policy == SWAP_POLICY_STRICT,
        "wave_schedule_sha256": wave_schedule_sha256(),
        "worker_session_swap_telemetry": None,
        "measured_phase_swap_window": None,
        "entries": [],
    }


def _write_invalid_attempt(
    output_dir: Path, *, execution_mode: str, error: BaseException
) -> None:
    if execution_mode not in {"concurrent", "sequential_fallback"}:
        raise RuntimeError("cannot mark an attempt with an unknown execution mode")
    manifest_path = output_dir / screen.MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("cannot bind invalid attempt to its manifest") from exc
    screen.hardened._atomic_write_json(
        output_dir / INVALID_ATTEMPT_FILENAME,
        {
            "schema_version": 1,
            "kind": CAMPAIGN_KIND + "_invalid_attempt",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "output_dir": str(output_dir.resolve()),
            "execution_mode": execution_mode,
            "swap_policy": manifest["swap_policy"],
            "wave_schedule_sha256": wave_schedule_sha256(),
            "protocol_sha256": manifest["protocol_sha256"],
            "manifest_sha256": screen.hardened._sha256_file(manifest_path),
            "preflight_report_sha256": manifest["preflight_report_sha256"],
            "execution_grid_sha256": manifest["execution_grid_sha256"],
            "git_head": manifest["source"]["git_head"],
            "error_type": type(error).__name__,
            "error": str(error) or type(error).__name__,
            "recovery": (
                "use --sequential-recovery-from with a fresh output namespace"
                if execution_mode == "concurrent"
                else (
                    "sequential fallback failed; restart the frozen campaign "
                    "without a recovery flag in a fresh output namespace"
                )
            ),
        },
    )


def _close_production_swap_session(
    session_swap: dict[str, Any],
    measured_swap: dict[str, Any] | None,
    history: dict[str, Any] | None,
    *,
    history_path: Path,
    execution_mode: str,
    output_dir: Path,
    require_complete: bool,
    swap_policy: str = DEFAULT_SWAP_POLICY,
) -> None:
    """Capture shutdown and durably bind the full worker-session swap record."""
    swap_policy = _validate_swap_policy(swap_policy)
    _checkpoint_swap_session(session_swap, swap_policy)
    if history is None:
        return
    history["worker_session_swap_telemetry"] = session_swap
    history["measured_phase_swap_window"] = measured_swap
    validate_concurrency_history(
        history,
        execution_mode=execution_mode,
        output_dir=output_dir,
        require_complete=require_complete,
        swap_policy=swap_policy,
        require_shutdown_sample=True,
    )
    screen.hardened._atomic_write_json(history_path, history)


def execute_production(
    output_dir: Path,
    *,
    execution_mode: str,
    pending_wave_indices: Sequence[int],
    invalidated_wave_indices: Sequence[int],
    owner_session: Mapping[str, Any] | None = None,
    swap_policy: str = DEFAULT_SWAP_POLICY,
) -> dict[str, Any]:
    """Execute strict waves and durably journal only completed barriers."""
    swap_policy = _validate_swap_policy(swap_policy)
    session_root = (
        output_dir
        / "worker_scratch"
        / f"session-{os.getpid()}-{time.monotonic_ns()}"
    )
    workers = []
    session_swap = _new_swap_session_telemetry()
    measured_swap: dict[str, Any] | None = None
    history: dict[str, Any] | None = None
    history_path = output_dir / CONCURRENCY_HISTORY_FILENAME
    owner_workers_bound = False
    durable_entry_count = 0
    try:
        _create_private_worker_root(session_root, output_dir=output_dir)
        workers, ready = _start_workers(session_root)
        if owner_session is not None:
            bind_owner_workers(owner_session, "production", workers)
            owner_workers_bound = True
        _checkpoint_swap_session(session_swap, swap_policy)
        warmup = _warm_workers(workers)
        measured_swap = _start_measured_swap_window(
            session_swap, swap_policy=swap_policy
        )
        _record_warmup_session(
            output_dir,
            execution_mode=execution_mode,
            ready=ready,
            records=warmup,
        )
        history = _load_concurrency_history(
            output_dir,
            execution_mode,
            set(pending_wave_indices),
            swap_policy,
        )
        if history["entries"]:
            raise RuntimeError("production worker session cannot adopt prior waves")
        history["worker_session_swap_telemetry"] = session_swap
        history["measured_phase_swap_window"] = measured_swap
        schedule = expected_wave_schedule()
        completed = {entry["wave_index"] for entry in history["entries"]}
        for wave_index in pending_wave_indices:
            if wave_index in completed:
                raise RuntimeError(f"pending wave already exists in journal: {wave_index}")
            wave = schedule[wave_index]
            assignments = [
                (
                    item["worker_slot"],
                    _key_tuple(item["key"]),
                    output_dir,
                )
                for item in wave["jobs"]
            ]
            if execution_mode == "concurrent":
                dispatch_label = f"production-wave-{wave_index}"
                reports, telemetry = _dispatch_runs(
                    workers, assignments, label=dispatch_label
                )
                _checkpoint_measured_swap_window(
                    session_swap,
                    measured_swap,
                    label=dispatch_label,
                    resource_telemetry=telemetry,
                    reports=reports,
                    swap_policy=swap_policy,
                )
                mode_details = {"segments": 1}
            else:
                reports = []
                segments = []
                for ordinal, assignment in enumerate(
                    sorted(assignments, key=lambda item: item[0])
                ):
                    current_reports, current_telemetry = _dispatch_runs(
                        workers,
                        [assignment],
                        label=f"production-wave-{wave_index}-job-{ordinal}",
                    )
                    _checkpoint_measured_swap_window(
                        session_swap,
                        measured_swap,
                        label=f"production-wave-{wave_index}-job-{ordinal}",
                        resource_telemetry=current_telemetry,
                        reports=current_reports,
                        swap_policy=swap_policy,
                    )
                    reports.extend(current_reports)
                    segments.append(current_telemetry)
                reports.sort(key=lambda item: item["slot"])
                high_water_by_slot = {
                    slot: max(
                        item["process_peak_rss_bytes"]
                        for segment in segments
                        for item in segment["worker_high_water_rss_bytes"]
                        if item["slot"] == slot
                    )
                    for slot in range(WORKER_COUNT)
                }
                combined_samples = [
                    sample for item in segments for sample in item["samples"]
                ]
                telemetry = {
                    "sample_count": len(combined_samples),
                    "samples": combined_samples,
                    "physical_memory_bytes": segments[0]["physical_memory_bytes"],
                    "peak_combined_rss_bytes": max(
                        max(item["peak_combined_rss_bytes"] for item in segments),
                        sum(high_water_by_slot.values()),
                    ),
                    "worker_high_water_rss_bytes": [
                        {
                            "slot": slot,
                            "pid": workers[slot]["process"].pid,
                            "process_peak_rss_bytes": high_water_by_slot[slot],
                        }
                        for slot in range(WORKER_COUNT)
                    ],
                    "high_water_combined_rss_bytes": sum(
                        high_water_by_slot.values()
                    ),
                    "peak_combined_thread_count": max(
                        item["peak_combined_thread_count"] for item in segments
                    ),
                    "swap_in_delta": (
                        combined_samples[-1]["swap_in_bytes"]
                        - combined_samples[0]["swap_in_bytes"]
                    ),
                    "swap_out_delta": (
                        combined_samples[-1]["swap_out_bytes"]
                        - combined_samples[0]["swap_out_bytes"]
                    ),
                    "wave_seconds": sum(item["wave_seconds"] for item in segments),
                    "start_skew_seconds": 0.0,
                    "overlap_seconds": 0.0,
                    "solo_tail_seconds": 0.0,
                }
                mode_details = {"segments": 2, "segment_telemetry": segments}
            _validate_completed_wave(
                reports,
                telemetry,
                execution_mode=execution_mode,
                wave_index=wave_index,
                swap_policy=swap_policy,
            )
            enriched = []
            by_slot = {item["slot"]: item for item in reports}
            for report in reports:
                partner = by_slot[1 - report["slot"]]
                enriched.append(
                    {
                        **report,
                        "wave_index": wave_index,
                        "partner_key": partner["key"],
                        "wave_schedule_sha256": wave_schedule_sha256(),
                    }
                )
            history["entries"].append(
                {
                    "wave_index": wave_index,
                    "dataset": wave["dataset"],
                    "execution_mode": execution_mode,
                    "reports": enriched,
                    "telemetry": telemetry,
                    "mode_details": mode_details,
                }
            )
            history["entries"].sort(key=lambda item: item["wave_index"])
            screen.hardened._atomic_write_json(history_path, history)
            durable_entry_count = len(history["entries"])
            print(
                f"SHOOTOUT_WAVE_COMPLETE {wave_index + 1}/{EXPECTED_WAVES} "
                f"{wave['dataset']} {execution_mode}",
                flush=True,
            )
        validate_concurrency_history(
            history,
            execution_mode=execution_mode,
            output_dir=output_dir,
            swap_policy=swap_policy,
            require_shutdown_sample=False,
        )
        return history
    except Exception as exc:
        _write_invalid_attempt(
            output_dir, execution_mode=execution_mode, error=exc
        )
        raise
    finally:
        if sys.exc_info()[0] is not None:
            active_error = sys.exc_info()[1]
            if (
                isinstance(active_error, KeyboardInterrupt)
                and history is not None
                and measured_swap is not None
                and not (output_dir / INVALID_ATTEMPT_FILENAME).exists()
            ):
                del history["entries"][durable_entry_count:]
                dispatch_count = durable_entry_count * (
                    1 if execution_mode == "concurrent" else WORKER_COUNT
                )
                _truncate_measured_swap_window(
                    measured_swap, session_swap, dispatch_count
                )
            try:
                _stop_workers(workers, force=True)
                if owner_workers_bound:
                    mark_owner_workers_quiesced(owner_session, "production")
            except Exception as shutdown_error:
                # A clean KeyboardInterrupt is resumable only after every worker
                # is confirmed dead.  If teardown cannot prove that, bind an
                # invalid marker so a potentially active writer cannot be reused.
                if not (output_dir / INVALID_ATTEMPT_FILENAME).exists():
                    _write_invalid_attempt(
                        output_dir,
                        execution_mode=execution_mode,
                        error=shutdown_error,
                    )
                raise RuntimeError(
                    "worker shutdown could not be confirmed; attempt is invalid"
                ) from shutdown_error
            try:
                resumable_history = (
                    None
                    if (output_dir / INVALID_ATTEMPT_FILENAME).exists()
                    else history
                )
                _close_production_swap_session(
                    session_swap,
                    measured_swap,
                    resumable_history,
                    history_path=history_path,
                    execution_mode=execution_mode,
                    output_dir=output_dir,
                    require_complete=False,
                    swap_policy=swap_policy,
                )
            except Exception as resource_error:
                if not (output_dir / INVALID_ATTEMPT_FILENAME).exists():
                    _write_invalid_attempt(
                        output_dir,
                        execution_mode=execution_mode,
                        error=resource_error,
                    )
                raise RuntimeError(
                    "worker-session resource contract failed during shutdown; "
                    "attempt is invalid"
                ) from resource_error
        else:
            try:
                _stop_workers(workers)
                if owner_workers_bound:
                    mark_owner_workers_quiesced(owner_session, "production")
                _close_production_swap_session(
                    session_swap,
                    measured_swap,
                    history,
                    history_path=history_path,
                    execution_mode=execution_mode,
                    output_dir=output_dir,
                    require_complete=True,
                    swap_policy=swap_policy,
                )
            except Exception as exc:
                _write_invalid_attempt(
                    output_dir, execution_mode=execution_mode, error=exc
                )
                raise


def _stable_optional_artifact(output_dir: Path, filename: str) -> dict[str, Any] | None:
    path = output_dir / filename
    return screen._stable_file_artifact(path, output_dir) if path.exists() else None


def _validate_owner_completion_ready(
    handle: Mapping[str, Any],
    *,
    execution_mode: str,
    manifest_path: Path,
    observed_workers: set[tuple[int, int]],
) -> dict[str, Any]:
    _state, session = _owner_current_session(handle)
    production = [
        cohort for cohort in session["worker_cohorts"] if cohort["phase"] == "production"
    ]
    recorded_workers = (
        {(item["slot"], item["pid"]) for item in production[0]["workers"]}
        if len(production) == 1
        else set()
    )
    if (
        session["phase"] != "production"
        or session["execution_mode"] != execution_mode
        or session["manifest_sha256"]
        != screen.hardened._sha256_file(manifest_path)
        or _read_process_identity(os.getpid()) != session["parent"]
        or len(production) != 1
        or production[0]["quiesced_at_utc"] is None
        or recorded_workers != observed_workers
    ):
        raise RuntimeError("owner session is not ready for completion")
    return session


def write_completion_attestation(
    output_dir: Path,
    *,
    manifest: Mapping[str, Any],
    jobs: Sequence[Any],
    owner_session: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = screen.collect_result_artifacts(output_dir, jobs)
    validation, outer_rows, child_rows = screen.validate_completed_results(
        output_dir, artifacts
    )
    payload = {
        "schema_version": 1,
        "kind": PAYLOAD_KIND,
        "protocol_sha256": manifest["protocol_sha256"],
        "wave_schedule_sha256": manifest["wave_schedule_sha256"],
        "execution_grid_sha256": manifest["execution_grid_sha256"],
        "preflight_report_sha256": manifest["preflight_report_sha256"],
        "swap_policy": manifest["swap_policy"],
        "timing_admissible": manifest["timing_admissible"],
        "result_artifacts_sha256": hashlib.sha256(
            screen.hardened._canonical_json(artifacts)
        ).hexdigest(),
        "outer_rows": outer_rows,
        "child_rows": child_rows,
    }
    payload_path = output_dir / screen.ANALYSIS_PAYLOAD_FILENAME
    screen.hardened._atomic_write_json(payload_path, payload)
    concurrency_path = output_dir / CONCURRENCY_HISTORY_FILENAME
    concurrency = json.loads(concurrency_path.read_text(encoding="utf-8"))
    validate_concurrency_history(
        concurrency,
        execution_mode=manifest["execution_mode"],
        output_dir=output_dir,
        swap_policy=manifest["swap_policy"],
    )
    warmup_path = output_dir / screen.WARMUP_HISTORY_FILENAME
    warmup_sessions = json.loads(warmup_path.read_text(encoding="utf-8"))
    validate_warmup_sessions(
        warmup_sessions,
        execution_mode=manifest["execution_mode"],
        output_dir=output_dir,
    )
    latest_warmed_workers = {
        (item["slot"], item["pid"])
        for item in warmup_sessions[-1]["worker_ready"]
    }
    observed_workers = {
        (report["slot"], report["pid"])
        for entry in concurrency["entries"]
        for report in entry["reports"]
    }
    resume_path = output_dir / screen.RESUME_HISTORY_FILENAME
    if resume_path.exists():
        resume_history = json.loads(resume_path.read_text(encoding="utf-8"))
        if not isinstance(resume_history, list):
            raise RuntimeError("resume history must be a list")
        validate_resume_history(resume_history, output_dir)
    else:
        resume_history = []
    if (
        observed_workers != latest_warmed_workers
        or len(warmup_sessions) > len(resume_history) + 1
    ):
        raise RuntimeError(
            "completed results do not bind to one persistent warmed worker pair"
        )
    manifest_path = output_dir / screen.MANIFEST_FILENAME
    owner = _validate_owner_completion_ready(
        owner_session,
        execution_mode=manifest["execution_mode"],
        manifest_path=manifest_path,
        observed_workers=observed_workers,
    )
    if screen.collect_source_provenance(output_dir=output_dir) != manifest["source"]:
        raise RuntimeError("source provenance changed during the campaign")
    if screen.collect_runtime_provenance() != manifest["runtime"]:
        raise RuntimeError("runtime provenance changed during the campaign")
    preflight_path = output_dir / PREFLIGHT_REPORT_FILENAME
    if screen.hardened._sha256_file(preflight_path) != manifest["preflight_report_sha256"]:
        raise RuntimeError("preflight report changed during the campaign")
    attestation = {
        "schema_version": 1,
        "kind": COMPLETION_KIND,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "execution_mode": manifest["execution_mode"],
        "swap_policy": manifest["swap_policy"],
        "timing_admissible": manifest["timing_admissible"],
        "wave_schedule_sha256": manifest["wave_schedule_sha256"],
        "execution_grid_sha256": manifest["execution_grid_sha256"],
        "protocol_sha256": manifest["protocol_sha256"],
        "preflight_report_sha256": manifest["preflight_report_sha256"],
        "git_head": manifest["source"]["git_head"],
        "manifest_sha256": screen.hardened._sha256_file(manifest_path),
        "owner_session_id": owner["session_id"],
        "result_count": len(artifacts),
        "expected_result_count": EXPECTED_JOBS,
        "expected_child_fits": EXPECTED_CHILD_FITS,
        "expected_paired_comparisons": EXPECTED_COORDINATES,
        "result_artifacts": artifacts,
        "analysis_payload_artifact": screen._stable_file_artifact(
            payload_path, output_dir
        ),
        "preflight_report_artifact": screen._stable_file_artifact(
            preflight_path, output_dir
        ),
        "concurrency_history_artifact": screen._stable_file_artifact(
            concurrency_path, output_dir
        ),
        "warmup_history_artifact": screen._stable_file_artifact(
            warmup_path, output_dir
        ),
        "resume_history_artifact": _stable_optional_artifact(
            output_dir, screen.RESUME_HISTORY_FILENAME
        ),
        "validation": validation,
    }
    screen.hardened._atomic_write_json(
        output_dir / screen.COMPLETION_ATTESTATION_FILENAME, attestation
    )
    return attestation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_SECONDS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--swap-policy",
        choices=SWAP_POLICIES,
        default=None,
        help=(
            "host swap policy; quality_only_swap_in preserves quality/safety "
            "gates but makes performance evidence inadmissible"
        ),
    )
    parser.add_argument(
        "--sequential-recovery-from",
        type=Path,
        help=(
            "run the provenance-bound sequential fallback from a prior invalid "
            "concurrent attempt in a fresh output directory"
        ),
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.time_limit) or args.time_limit != TIME_LIMIT_SECONDS:
        parser.error(f"--time-limit is frozen at {TIME_LIMIT_SECONDS:g} seconds")
    if sum(
        (
            args.resume,
            args.dry_run,
            args.preflight,
            args.sequential_recovery_from is not None,
        )
    ) > 1:
        parser.error(
            "--resume, --dry-run, --preflight, and --sequential-recovery-from "
            "are mutually exclusive"
        )
    return args


def _read_bound_swap_policy(output_dir: Path) -> str:
    """Read a campaign policy without mutating or acquiring its namespace."""
    output_dir = output_dir.resolve(strict=True)
    path = output_dir / screen.MANIFEST_FILENAME
    screen.hardened._require_regular_archive_source(path, "swap-policy manifest")
    if not path.is_file():
        raise RuntimeError("swap-policy manifest is missing")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("could not read swap-policy manifest") from exc
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("kind") != CAMPAIGN_KIND
        or manifest.get("protocol_sha256") != screen.protocol_sha256()
    ):
        raise RuntimeError("swap-policy manifest does not match this campaign")
    policy = _validate_swap_policy(manifest.get("swap_policy"))
    if manifest.get("timing_admissible") is not (policy == SWAP_POLICY_STRICT):
        raise RuntimeError("swap-policy manifest timing status is inconsistent")
    return policy


def resolve_invocation_swap_policy(
    args: argparse.Namespace, output_dir: Path
) -> str:
    """Resolve omitted resume/recovery policy before any namespace mutation."""
    requested = args.swap_policy
    source = None
    if args.resume:
        source = output_dir
    elif args.sequential_recovery_from is not None:
        source = args.sequential_recovery_from
    if source is None:
        return _validate_swap_policy(requested or DEFAULT_SWAP_POLICY)
    bound = _read_bound_swap_policy(Path(source))
    if requested is not None and requested != bound:
        raise RuntimeError(
            f"requested swap policy {requested!r} does not match bound policy {bound!r}"
        )
    return bound


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    args.swap_policy = resolve_invocation_swap_policy(args, output_dir)
    schedule = expected_wave_schedule()
    _, jobs, child_cpus = build_runtime_jobs(args.time_limit)
    ordering = screen.ordering_balance(jobs)
    print(
        f"built {len(jobs)} B10/A10 jobs ({EXPECTED_CHILD_FITS} child fits); "
        f"waves={len(schedule)} workers={WORKER_COUNT} child_cpus={child_cpus} "
        f"swap_policy={args.swap_policy}"
    )
    print(
        "schedule "
        + json.dumps(
            {
                "sha256": wave_schedule_sha256(),
                "ordering_balance": ordering,
            },
            sort_keys=True,
        )
    )
    if args.dry_run:
        return 0

    reused_evidence = verify_reused_evidence()
    source = screen.collect_source_provenance(output_dir=output_dir)
    if args.resume:
        screen.hardened.validate_output_state(output_dir, resume=True)
        owner_session = acquire_owner_session(
            output_dir, resume=True, phase="resume_validation"
        )
    else:
        screen.hardened.validate_output_state(output_dir, resume=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        owner_session = acquire_owner_session(
            output_dir, resume=False, phase="preflight"
        )
    resume_prepared = False
    try:
        if args.resume:
            if (output_dir / INVALID_ATTEMPT_FILENAME).exists():
                raise RuntimeError(
                    "this execution attempt is invalid and cannot be resumed; "
                    "use a fresh output directory"
                )
            preflight, preflight_sha256 = _load_preflight_report(
                output_dir, args.swap_policy
            )
        else:
            recovery = (
                collect_sequential_recovery(
                    args.sequential_recovery_from,
                    current_source=source,
                    swap_policy=args.swap_policy,
                )
                if args.sequential_recovery_from is not None
                else None
            )
            preflight = run_preflight(
                output_dir,
                sequential_recovery=recovery,
                owner_session=owner_session,
                swap_policy=args.swap_policy,
            )
            preflight_sha256 = screen.hardened._sha256_file(
                output_dir / PREFLIGHT_REPORT_FILENAME
            )
            if args.preflight:
                finalize_owner_session(owner_session, "preflight_only")
                decision = preflight["decision"]
                timing_detail = (
                    f"speedup={decision['throughput_speedup']}"
                    if decision["timing_admissible"]
                    else "timing=inadmissible_raw_operational_audit_only"
                )
                print(
                    "SHOOTOUT_PREFLIGHT_COMPLETE "
                    f"{decision['execution_mode']} "
                    f"{timing_detail} {output_dir}",
                    flush=True,
                )
                return 0
        execution_mode = preflight["decision"]["execution_mode"]
        grid = execution_grid_payload(execution_mode, args.swap_policy)
        grid_path = output_dir / WAVE_SCHEDULE_FILENAME
        if args.resume:
            try:
                existing_grid = json.loads(grid_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("could not read the execution grid") from exc
            if existing_grid != grid:
                raise RuntimeError("execution grid does not match the frozen schedule")
        else:
            screen.hardened._atomic_write_json(grid_path, grid)
        manifest = build_run_manifest(
            output_dir=output_dir,
            source=source,
            ordering=ordering,
            execution_mode=execution_mode,
            preflight_sha256=preflight_sha256,
            reused_evidence=reused_evidence,
            swap_policy=args.swap_policy,
        )
        manifest["execution_grid_artifact_sha256"] = screen.hardened._sha256_file(
            grid_path
        )
        manifest = write_or_validate_manifest(output_dir, manifest, resume=args.resume)
        manifest_path = output_dir / screen.MANIFEST_FILENAME
        bind_owner_manifest(
            owner_session,
            execution_mode=execution_mode,
            manifest_path=manifest_path,
        )
        resume_state = prepare_wave_resume(
            output_dir,
            jobs,
            schedule,
            resume=args.resume,
            execution_mode=execution_mode,
            swap_policy=args.swap_policy,
        )
        resume_prepared = True
        execute_production(
            output_dir,
            execution_mode=execution_mode,
            pending_wave_indices=resume_state["pending_wave_indices"],
            invalidated_wave_indices=resume_state["invalidated_wave_indices"],
            owner_session=owner_session,
            swap_policy=args.swap_policy,
        )
        attestation = write_completion_attestation(
            output_dir,
            manifest=manifest,
            jobs=jobs,
            owner_session=owner_session,
        )
        finalize_owner_session(owner_session, "completed")
        validate_completed_owner_session(output_dir, attestation)
        print(f"ACCURACY_SHOOTOUT_COMPLETE {EXPECTED_JOBS} {output_dir}", flush=True)
        return 0
    except BaseException as exc:
        try:
            _state, current_owner = _owner_current_session(
                owner_session, require_active=False
            )
            if current_owner["state"] == "active":
                resumable_interrupt = (
                    isinstance(exc, KeyboardInterrupt)
                    and resume_prepared
                    and not (output_dir / INVALID_ATTEMPT_FILENAME).exists()
                )
                finalize_owner_session(
                    owner_session,
                    "interrupted" if resumable_interrupt else "invalid",
                )
        except Exception as owner_error:
            raise RuntimeError(
                "owner session could not be safely finalized"
            ) from owner_error
        raise
    finally:
        release_owner_session(owner_session)


if __name__ == "__main__":
    raise SystemExit(main())
