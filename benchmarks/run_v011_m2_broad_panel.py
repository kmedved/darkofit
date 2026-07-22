"""Run the frozen v0.11 M2 defaults-only broad comparison panel.

This successor deliberately reuses the hardened same-machine runner's job,
telemetry, provenance, and normalization machinery.  It removes the
historical ordinal diagnostic and updates only the three declared product
pins.  The historical campaign module is configured only while ``main`` is
running, so importing this module cannot mutate the old campaign contract.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import stat
import subprocess
import sys
import warnings
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

if __package__ in {None, ""}:  # Direct execution from a clean checkout.
    _REPO_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

try:
    from benchmarks import run_tabarena_regression_same_machine as _base
except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
    import run_tabarena_regression_same_machine as _base


CAMPAIGN_DATE = "2026-07-22"
ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ID = "v011-m2-broad-panel-20260722-v1"
CONTRACT_PATH = ROOT / "benchmarks/v011_m2_broad_panel_contract_20260722.json"
PRIMARY_LANE = _base.PRIMARY_LANE
ORDINAL_DIAGNOSTIC_LANE = _base.ORDINAL_DIAGNOSTIC_LANE
LANES = (PRIMARY_LANE,)
TASK_SPLIT_COUNTS = dict(_base.TASK_SPLIT_COUNTS)
TASKS = dict(_base.TASKS)
PRIMARY_COORDINATE_PAIRS = tuple(_base.PRIMARY_COORDINATE_PAIRS)
SPLIT_INDICES = list(_base.SPLIT_INDICES)
ORDER_CYCLE = tuple(_base.ORDER_CYCLE)

ENGINE_SPECS: dict[str, dict[str, Any]] = {
    "darkofit": {
        "code": "D",
        "display_name": "DarkoFit",
        "model_type": "DARKO",
        "version": "0.10.1",
        "native_class": "ComparatorDarkoFitModel",
        "ordinal_class": "ComparatorOrdinalDarkoFitModel",
    },
    "chimeraboost": {
        "code": "M",
        "display_name": "ChimeraBoost",
        "model_type": "CHIMERA",
        "version": "0.18.0",
        "native_class": "ComparatorChimeraBoostModel",
        "ordinal_class": "ComparatorOrdinalChimeraBoostModel",
    },
    "catboost": {
        "code": "C",
        "display_name": "CatBoost",
        "model_type": "CAT",
        "version": "1.2.10",
        "native_class": "ComparatorCatBoostModel",
        "ordinal_class": "ComparatorOrdinalCatBoostModel",
    },
}
ENGINE_CODES = {spec["code"]: engine for engine, spec in ENGINE_SPECS.items()}
ARM_SPECS = {
    "darkofit_0_10_1_default": {
        "lane": PRIMARY_LANE,
        "engine": "darkofit",
        "code": "D",
        "config": {},
        "model_cls": "ComparatorDarkoFitModel",
        "representation": "native",
    },
    "chimeraboost_f14be60_default": {
        "lane": PRIMARY_LANE,
        "engine": "chimeraboost",
        "code": "M",
        "config": {},
        "model_cls": "ComparatorChimeraBoostModel",
        "representation": "native",
    },
    "catboost_1_2_10_default": {
        "lane": PRIMARY_LANE,
        "engine": "catboost",
        "code": "C",
        "config": {},
        "model_cls": "ComparatorCatBoostModel",
        "representation": "native",
    },
}
ARM_BY_LANE_CODE = {
    (spec["lane"], spec["code"]): arm for arm, spec in ARM_SPECS.items()
}

EXPECTED_PRIMARY_COORDINATES = len(TASKS) * len(PRIMARY_COORDINATE_PAIRS)
EXPECTED_DIAGNOSTIC_COORDINATES = 0
EXPECTED_COORDINATES = EXPECTED_PRIMARY_COORDINATES
EXPECTED_PRIMARY_JOBS = EXPECTED_PRIMARY_COORDINATES * len(ENGINE_SPECS)
EXPECTED_DIAGNOSTIC_JOBS = 0
EXPECTED_JOBS = EXPECTED_PRIMARY_JOBS
EXPECTED_CHILD_FITS = EXPECTED_JOBS * 8
EXPECTED_CHILD_CPUS = 18
TIME_LIMIT_SECONDS = 3_600.0
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_260_722

CHIMERABOOST_TAG_COMMIT = "f14be606b641f1bf0dc92bb14b3951f1fe631c6b"
CHIMERABOOST_VERSION = "0.18.0"
CATBOOST_VERSION = "1.2.10"
CHIMERABOOST_DESCRIBE = "v0.18.0-6-gf14be60"
TABARENA_TAG_COMMIT = "4cd1d2526874962daae048a6f2dcf34aa272f3fa"
TABARENA_GIT_TREE = "a293df372a613c7358ba5fcd746f58d580cde7d6"
TABARENA_VERSION = "0.0.1"
AUTOGLUON_VERSION = "1.5.1b20260712"

DEFAULT_CHIMERABOOST_PATH = Path("/private/tmp/darkofit-v011-chimera-f14be60")
DEFAULT_OUTPUT_DIR = Path(".cache/v011-m2-broad-panel-20260722")
DEFAULT_ANALYSIS_OUTPUT_FILENAMES = (
    "paired_coordinates.csv",
    "per_dataset.csv",
    "summary.json",
    "report.md",
)
WORKER_ATTESTATION_DIRNAME = "worker_attestations"
TERMINAL_FILENAME = "terminal.json"
WORKER_ENVIRONMENT = {
    "OMP_NUM_THREADS": str(EXPECTED_CHILD_CPUS),
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": str(EXPECTED_CHILD_CPUS),
    "MKL_NUM_THREADS": str(EXPECTED_CHILD_CPUS),
    "MKL_DYNAMIC": "FALSE",
    "NUMEXPR_NUM_THREADS": str(EXPECTED_CHILD_CPUS),
    "NUMBA_NUM_THREADS": str(EXPECTED_CHILD_CPUS),
    "DARKOFIT_WARMUP": "0",
    "CHIMERABOOST_WARMUP": "0",
    "PYTHONHASHSEED": "0",
}
CAMPAIGN_KIND = "darkofit_v011_m2_broad_panel_20260722"
COMPLETION_KIND = CAMPAIGN_KIND + "_completion"
PAYLOAD_KIND = CAMPAIGN_KIND + "_analysis_payload"
WARMUP_KIND = CAMPAIGN_WARMUP_KIND = _base.WARMUP_KIND

SOURCE_FILES = (
    *_base.SOURCE_FILES[:-3],
    Path("benchmarks/run_tabarena_regression_same_machine.py"),
    Path("benchmarks/analyze_tabarena_regression_same_machine.py"),
    Path("benchmarks/run_v011_m2_broad_panel.py"),
    Path("benchmarks/analyze_v011_m2_broad_panel.py"),
    Path("benchmarks/freeze_v011_m2_broad_panel.py"),
    Path("benchmarks/v011_m2_broad_panel_protocol_20260722.md"),
    Path("benchmarks/v011_m2_broad_panel_contract_20260722.json"),
)

BOUND_PATHS = {
    "authorization": Path("benchmarks/v011_evidence_phase_instruction_20260721.md"),
    "protocol": Path("benchmarks/v011_m2_broad_panel_protocol_20260722.md"),
    "runner": Path("benchmarks/run_v011_m2_broad_panel.py"),
    "analyzer": Path("benchmarks/analyze_v011_m2_broad_panel.py"),
    "freezer": Path("benchmarks/freeze_v011_m2_broad_panel.py"),
    "tests": Path("tests/test_v011_m2_broad_panel.py"),
}


def _bound_record(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"could not inspect bound source: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"bound source is not a regular file: {path}")
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def expected_coordinates() -> list[tuple[str, str, int, int]]:
    return [
        (PRIMARY_LANE, dataset, repeat, fold)
        for dataset in TASKS
        for repeat, fold in PRIMARY_COORDINATE_PAIRS
    ]


def expected_ordered_grid() -> list[tuple[str, str, int, int, str]]:
    ordered: list[tuple[str, str, int, int, str]] = []
    for index, coordinate in enumerate(expected_coordinates()):
        for code in ORDER_CYCLE[index % len(ORDER_CYCLE)]:
            ordered.append((*coordinate, ARM_BY_LANE_CODE[(PRIMARY_LANE, code)]))
    if len(ordered) != EXPECTED_JOBS or len(set(ordered)) != EXPECTED_JOBS:
        raise RuntimeError("frozen M2 ordered grid is incomplete")
    return ordered


def job_order_sha256() -> str:
    payload = [
        {
            "lane": lane,
            "dataset": dataset,
            "repeat": repeat,
            "fold": fold,
            "arm": arm,
            "arm_code": ARM_SPECS[arm]["code"],
        }
        for lane, dataset, repeat, fold, arm in expected_ordered_grid()
    ]
    return hashlib.sha256(_base.hardened._canonical_json(payload)).hexdigest()


def expected_position_audit() -> dict[str, Any]:
    counts = {engine: [0, 0, 0] for engine in ENGINE_SPECS}
    for index, _coordinate in enumerate(expected_coordinates()):
        for position, code in enumerate(ORDER_CYCLE[index % len(ORDER_CYCLE)]):
            counts[ENGINE_CODES[code]][position] += 1
    if any(values != [13, 13, 13] for values in counts.values()):
        raise RuntimeError("M2 arm positions are not perfectly balanced")
    return {
        "job_order_sha256": job_order_sha256(),
        "lane_position_counts": {
            PRIMARY_LANE: {
                engine: {
                    "first": values[0],
                    "second": values[1],
                    "third": values[2],
                }
                for engine, values in counts.items()
            }
        },
    }


def frozen_protocol() -> dict[str, Any]:
    return {
        "campaign_date": CAMPAIGN_DATE,
        "claim": "descriptive_current_version_calibrated_yardstick",
        "policy_advancement_allowed": False,
        "task_split_counts": {
            dataset: {"task_id": task_id, "registered_split_count": count}
            for dataset, (task_id, count) in TASK_SPLIT_COUNTS.items()
        },
        "coordinates": [
            {"dataset": dataset, "repeat": repeat, "fold": fold}
            for _, dataset, repeat, fold in expected_coordinates()
        ],
        "arms": {
            arm: {
                **{name: value for name, value in spec.items() if name != "config"},
                "manual_config": {},
                "official_defaults": True,
                "single_product_arm": True,
            }
            for arm, spec in ARM_SPECS.items()
        },
        "ensemble_candidate_included": False,
        "framework_evaluation": {
            "bag_folds": 8,
            "bag_sets": 1,
            "model_random_seed": 0,
            "vary_seed_across_folds": True,
            "fold_fitting_strategy": "sequential_local",
            "calibrate": False,
        },
        "official_default_disclosure": {
            "darkofit": {
                "version": "0.10.1",
                "manual_config": {},
                "promoted_dispatch_active_on_eligible_fits": True,
            },
            "chimeraboost": {
                "version": CHIMERABOOST_VERSION,
                "git_commit": CHIMERABOOST_TAG_COMMIT,
                "git_describe": CHIMERABOOST_DESCRIBE,
                "manual_config": {},
            },
            "catboost": {"version": CATBOOST_VERSION, "manual_config": {}},
        },
        "darkofit_execution_source_pin": {
            "policy": "published_contract_commit_only",
            "required_parent": "harness_freeze_git_head",
            "only_path_added_after_harness_freeze": str(CONTRACT_PATH.relative_to(ROOT)),
            "required_remote_ref": "origin/main",
        },
        "framework_source": {
            "tabarena_git_commit": TABARENA_TAG_COMMIT,
            "tabarena_git_tree": TABARENA_GIT_TREE,
            "tabarena_version": TABARENA_VERSION,
            "autogluon_version": AUTOGLUON_VERSION,
        },
        "ordered_job_sha256": job_order_sha256(),
        "order_cycle_codes": [list(row) for row in ORDER_CYCLE],
        "order_cycle_scope": "continuous_across_all_39_coordinate_groups",
        "execution_dispatch": {
            "lane": PRIMARY_LANE,
            "outer_jobs": EXPECTED_JOBS,
            "child_fits": EXPECTED_CHILD_FITS,
            "task_groups_are_input_contiguous": True,
            "outer_jobs_sequential": True,
            "fresh_worker_boundary": "one_new_python_process_per_outer_job",
            "same_arm_worker_warmup": ["numeric_regression", "categorical_regression"],
            "worker_count": EXPECTED_JOBS,
            "resume_allowed": False,
            "worker_environment": dict(WORKER_ENVIRONMENT),
        },
        "num_cpus": EXPECTED_CHILD_CPUS,
        "num_gpus": 0,
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "raise_on_model_failure": True,
        "warmup_outside_measured_jobs": True,
        "measurements": [
            "test_rmse",
            "validation_rmse",
            "fit_wall_time",
            "predict_wall_time",
            "incremental_peak_rss",
            "absolute_peak_rss",
        ],
        "analysis": {
            "dataset_weighting": "equal_1_over_13",
            "estimator": "paired_log_ratio",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resampling": "coordinates_within_each_fixed_dataset",
            "head_to_head_supplement": {
                "coordinate_wins": True,
                "equal_dataset_geomean_wins": True,
                "ties_reported": True,
                "descriptive_only": True,
            },
        },
        "prior_artifacts_untouched": True,
        "fresh_confirmation_or_lockbox_used": False,
    }


def protocol_sha256() -> str:
    return hashlib.sha256(_base.hardened._canonical_json(frozen_protocol())).hexdigest()


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    contract_path = Path(os.path.abspath(path.expanduser()))
    payload = _read_finite_json(contract_path)
    if (
        set(payload)
        != {
            "schema_version",
            "contract_id",
            "created_at_utc",
            "contract_frozen",
            "outcome_blind",
            "authorization",
            "harness_freeze_git_head",
            "bindings",
            "protocol_sha256",
            "protocol",
        }
        or payload.get("schema_version") != 1
        or payload.get("contract_id") != CONTRACT_ID
        or payload.get("contract_frozen") is not True
        or payload.get("outcome_blind") is not True
        or payload.get("authorization")
        != "Phase 2 of v011_evidence_phase_instruction_20260721.md"
        or payload.get("protocol") != frozen_protocol()
        or payload.get("protocol_sha256") != protocol_sha256()
        or set(payload.get("bindings", {})) != set(BOUND_PATHS)
    ):
        raise RuntimeError("v0.11 M2 contract is invalid")
    for name, relative in BOUND_PATHS.items():
        if payload["bindings"][name] != _bound_record(ROOT / relative):
            raise RuntimeError(f"v0.11 M2 contract binding drifted: {name}")
    freeze_head = str(payload.get("harness_freeze_git_head", ""))
    if len(freeze_head) != 40:
        raise RuntimeError("v0.11 M2 harness freeze commit is invalid")
    current = _base._run_git(["rev-parse", "HEAD"], cwd=ROOT)
    try:
        _base._run_git(["merge-base", "--is-ancestor", freeze_head, current], cwd=ROOT)
    except RuntimeError as exc:
        raise RuntimeError("current source does not descend from the M2 harness freeze") from exc
    return payload


def validate_execution_source_pin(contract: Mapping[str, Any]) -> str:
    """Require the unique published commit that added only this contract."""
    freeze_head = str(contract["harness_freeze_git_head"])
    current = _base._run_git(["rev-parse", "HEAD"], cwd=ROOT)
    revision = _base._run_git(
        ["rev-list", "--parents", "-n", "1", current], cwd=ROOT
    ).split()
    if revision != [current, freeze_head]:
        raise RuntimeError(
            "M2 execution source must be the direct contract-commit child of the "
            "harness freeze"
        )
    contract_relative = str(CONTRACT_PATH.relative_to(ROOT))
    changed = _base._run_git(
        ["diff", "--name-only", freeze_head, current], cwd=ROOT
    ).splitlines()
    if changed != [contract_relative]:
        raise RuntimeError("M2 contract commit changed more than the frozen contract")
    if _base._run_git(["rev-parse", "origin/main"], cwd=ROOT) != current:
        raise RuntimeError("M2 execution source is not the published origin/main commit")
    return current


def activate_chimeraboost_checkout(path: Path) -> dict[str, Any]:
    checkout = path.expanduser().resolve()
    if not checkout.is_dir() or not (checkout / "chimeraboost/__init__.py").is_file():
        raise RuntimeError(f"ChimeraBoost checkout is missing: {checkout}")
    if _base._run_git(["rev-parse", "HEAD"], cwd=checkout) != CHIMERABOOST_TAG_COMMIT:
        raise RuntimeError("ChimeraBoost checkout is not the exact f14be60 commit")
    if _base._run_git(["status", "--porcelain", "--untracked-files=all"], cwd=checkout):
        raise RuntimeError("ChimeraBoost f14be60 checkout is not clean")
    describe = _base._run_git(["describe", "--tags", "--always"], cwd=checkout)
    if describe != CHIMERABOOST_DESCRIBE:
        raise RuntimeError("ChimeraBoost f14be60 describe identity changed")
    _base._validate_chimeraboost_warmup_environment(
        os.environ.get("CHIMERABOOST_WARMUP")
    )
    original_modules = _base._loaded_chimeraboost_modules()
    origins = {
        _base._module_is_from_checkout(module, checkout)
        for module in original_modules.values()
    }
    if origins == {False, True}:
        raise RuntimeError("mixed ChimeraBoost imports are already loaded")
    if origins == {False}:
        for name in original_modules:
            sys.modules.pop(name, None)
    original_sys_path = list(sys.path)
    checkout_text = str(checkout)
    while checkout_text in sys.path:
        sys.path.remove(checkout_text)
    sys.path.insert(0, checkout_text)
    importlib.invalidate_caches()
    try:
        module = importlib.import_module("chimeraboost")
        module_file = Path(module.__file__).resolve()
        module_file.relative_to(checkout)
        if getattr(module, "__version__", None) != CHIMERABOOST_VERSION:
            raise RuntimeError("imported ChimeraBoost version is not 0.18.0")
        if any(
            not _base._module_is_from_checkout(loaded, checkout)
            for loaded in _base._loaded_chimeraboost_modules().values()
        ):
            raise RuntimeError("mixed ChimeraBoost imports are loaded")
    except BaseException:
        for name in _base._loaded_chimeraboost_modules():
            sys.modules.pop(name, None)
        sys.modules.update(original_modules)
        sys.path[:] = original_sys_path
        raise
    return {
        "repository": str(checkout),
        "git_head": CHIMERABOOST_TAG_COMMIT,
        "git_tree": _base._run_git(["rev-parse", "HEAD^{tree}"], cwd=checkout),
        "git_describe": describe,
        "git_remote_origin": _base.hardened._sanitize_git_remote(
            _base._run_git(["remote", "get-url", "origin"], cwd=checkout)
        ),
        "status": "",
        "module_file": str(module_file),
        "module_sha256": _base.hardened._sha256_file(module_file),
        "hidden_import_warmup": "disabled",
    }


def validate_framework_pins(source: Mapping[str, Any]) -> None:
    tabarena = source.get("tabarena")
    if (
        not isinstance(tabarena, Mapping)
        or tabarena.get("git_head") != TABARENA_TAG_COMMIT
        or tabarena.get("git_tree") != TABARENA_GIT_TREE
        or tabarena.get("status") != ""
    ):
        raise RuntimeError("TabArena source is not the frozen historical commit")
    expected_packages = {
        "tabarena": TABARENA_VERSION,
        "autogluon.common": AUTOGLUON_VERSION,
        "autogluon.core": AUTOGLUON_VERSION,
        "autogluon.features": AUTOGLUON_VERSION,
        "autogluon.tabular": AUTOGLUON_VERSION,
        "catboost": CATBOOST_VERSION,
    }
    observed = _base.collect_runtime_provenance()["packages"]
    mismatches = {
        name: observed.get(name)
        for name, version in expected_packages.items()
        if observed.get(name) != version
    }
    if mismatches:
        raise RuntimeError(f"M2 framework package pins drifted: {mismatches}")


def _import_warmup_module():
    try:
        from benchmarks import tabarena_comparator_warmup as warmup
    except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
        import tabarena_comparator_warmup as warmup
    return warmup


def _worker_attestation_path(output_dir: Path, worker_index: int) -> Path:
    return output_dir / WORKER_ATTESTATION_DIRNAME / f"{worker_index:03d}.json"


def _worker_warmup(engine: str, child_cpus: int) -> dict[str, Any]:
    warmup = _import_warmup_module()
    data = warmup._make_data()
    stages = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for input_kind in ("numeric", "categorical"):
            stages.append(
                warmup._run_stage(
                    engine=engine,
                    input_kind=input_kind,
                    data=data[input_kind],
                    thread_count=child_cpus,
                )
            )
    return {
        "engine": engine,
        "stage_names": [stage["name"] for stage in stages],
        "stages": stages,
        "warnings": [
            {
                "category": item.category.__name__,
                "message": str(item.message),
            }
            for item in caught
        ],
    }


def _write_create_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(
                json.dumps(
                    payload,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _read_finite_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"could not inspect JSON artifact: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"JSON artifact is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    return value


def _build_runtime_jobs(chimeraboost_path: Path):
    activate_chimeraboost_checkout(chimeraboost_path)
    model_classes = _base._load_model_classes()
    _base.validate_official_defaults(model_classes)
    from tabarena.contexts import TabArenaContext
    from tabarena.utils.config_utils import ConfigGenerator

    context = TabArenaContext()
    experiments = _base.build_experiments(
        model_classes=model_classes,
        config_generator_cls=ConfigGenerator,
        time_limit=TIME_LIMIT_SECONDS,
    )
    jobs = _base.build_comparator_jobs(context, experiments)
    ordering = _base.ordering_audit(jobs)
    child_cpus = _base.resolve_and_pin_child_cpu_allocation(jobs)
    return context, jobs, ordering, child_cpus


def _run_worker(args: argparse.Namespace) -> int:
    output_dir = Path(os.path.abspath(args.output_dir.expanduser()))
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise RuntimeError("M2 worker requires the parent-owned output directory")
    actual_environment = {key: os.environ.get(key) for key in WORKER_ENVIRONMENT}
    if actual_environment != WORKER_ENVIRONMENT:
        raise RuntimeError(f"M2 worker environment drifted: {actual_environment}")
    import numba

    if (
        int(numba.config.NUMBA_NUM_THREADS) != EXPECTED_CHILD_CPUS
        or int(numba.get_num_threads()) != EXPECTED_CHILD_CPUS
    ):
        raise RuntimeError("M2 worker Numba thread budget is not exactly 18")
    context, jobs, ordering, child_cpus = _build_runtime_jobs(args.chimeraboost_path)
    if ordering != expected_position_audit() or child_cpus != EXPECTED_CHILD_CPUS:
        raise RuntimeError("M2 worker schedule or resource allocation drifted")
    worker_index = int(args.worker_index)
    if worker_index not in range(EXPECTED_JOBS):
        raise RuntimeError("M2 worker index is outside the frozen grid")
    job = jobs[worker_index]
    expected = expected_ordered_grid()[worker_index]
    if (*_base._job_coordinate(job), _base._job_arm(job)) != expected:
        raise RuntimeError("M2 worker resolved the wrong frozen job")
    result_path = _base._result_path(output_dir, job)
    attestation_path = _worker_attestation_path(output_dir, worker_index)
    if result_path.exists() or result_path.is_symlink() or attestation_path.exists():
        raise RuntimeError("M2 worker output is create-only")
    engine = ARM_SPECS[expected[4]]["engine"]
    warmup = _worker_warmup(engine, child_cpus)
    started = datetime.now(timezone.utc).isoformat()
    results = context.run_jobs(
        [job],
        expname=str(output_dir / "experiments"),
        register=False,
        new_result_prefix="[v0.11 M2 fresh worker] ",
        debug_mode=True,
    )
    if len(results) != 1 or _base._cached_result_issue(result_path, job) is not None:
        raise RuntimeError("M2 fresh worker did not publish one valid result")
    result_artifact = _base._stable_file_artifact(result_path, output_dir)
    _write_create_only_json(
        attestation_path,
        {
            "schema_version": 1,
            "kind": CAMPAIGN_KIND + "_worker",
            "worker_index": worker_index,
            "pid": os.getpid(),
            "parent_pid": int(args.parent_pid),
            "started_at_utc": started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "coordinate": {
                "lane": expected[0],
                "dataset": expected[1],
                "repeat": expected[2],
                "fold": expected[3],
                "arm": expected[4],
                "engine": engine,
            },
            "same_arm_warmup": warmup,
            "environment": actual_environment,
            "numba_thread_ceiling": int(numba.config.NUMBA_NUM_THREADS),
            "numba_current_threads_after_fit": int(numba.get_num_threads()),
            "result_artifact": result_artifact,
        },
    )
    print(f"V011_M2_WORKER_COMPLETE {worker_index} {os.getpid()}")
    return 0


def _validate_worker_attestations(
    output_dir: Path, jobs: list[Any], *, parent_pid: int
) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    observed_paths = set()
    root = output_dir / WORKER_ATTESTATION_DIRNAME
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("M2 worker-attestation directory is missing or unsafe")
    for worker_index, job in enumerate(jobs):
        path = _worker_attestation_path(output_dir, worker_index)
        payload = _read_finite_json(path)
        expected = expected_ordered_grid()[worker_index]
        engine = ARM_SPECS[expected[4]]["engine"]
        result_path = _base._result_path(output_dir, job)
        expected_result = _base._stable_file_artifact(result_path, output_dir)
        expected_coordinate = {
            "lane": expected[0],
            "dataset": expected[1],
            "repeat": expected[2],
            "fold": expected[3],
            "arm": expected[4],
            "engine": engine,
        }
        if (
            set(payload)
            != {
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
            or type(payload.get("pid")) is not int
            or payload.get("pid", 0) <= 0
            or payload.get("schema_version") != 1
            or payload.get("kind") != CAMPAIGN_KIND + "_worker"
            or payload.get("worker_index") != worker_index
            or payload.get("parent_pid") != parent_pid
            or payload.get("pid") == parent_pid
            or payload.get("coordinate") != expected_coordinate
            or payload.get("result_artifact") != expected_result
            or payload.get("environment") != WORKER_ENVIRONMENT
            or payload.get("numba_thread_ceiling") != EXPECTED_CHILD_CPUS
            or payload.get("numba_current_threads_after_fit") != EXPECTED_CHILD_CPUS
            or not isinstance(payload.get("started_at_utc"), str)
            or not payload.get("started_at_utc")
            or not isinstance(payload.get("completed_at_utc"), str)
            or not payload.get("completed_at_utc")
        ):
            raise RuntimeError(f"M2 worker attestation {worker_index} is invalid")
        warmup = payload.get("same_arm_warmup")
        if (
            not isinstance(warmup, Mapping)
            or warmup.get("engine") != engine
            or warmup.get("stage_names")
            != [f"{engine}_numeric", f"{engine}_categorical"]
            or not isinstance(warmup.get("stages"), list)
            or len(warmup["stages"]) != 2
            or not isinstance(warmup.get("warnings"), list)
        ):
            raise RuntimeError(f"M2 worker warmup {worker_index} is invalid")
        for stage in warmup["stages"]:
            if (
                stage.get("engine") != engine
                or stage.get("thread_count") != EXPECTED_CHILD_CPUS
                or stage.get("representation", {}).get("kind") != "native"
            ):
                raise RuntimeError(f"M2 worker warmup stage {worker_index} drifted")
        relative = str(path.relative_to(output_dir))
        observed_paths.add(relative)
        artifact = _base._stable_file_artifact(path, output_dir)
        if artifact.pop("path") != relative:
            raise RuntimeError("M2 worker-attestation artifact path changed")
        artifacts[relative] = artifact
    actual = set()
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("M2 worker-attestation tree contains an unsafe entry")
        actual.add(str(path.relative_to(output_dir)))
    if actual != observed_paths or len(artifacts) != EXPECTED_JOBS:
        raise RuntimeError("M2 worker-attestation set is not exact")
    return artifacts


def write_completion_attestation(
    output_dir: Path,
    *,
    manifest: Mapping[str, Any],
    jobs: list[Any],
    result_count: int,
) -> dict[str, Any]:
    if result_count != EXPECTED_JOBS:
        raise RuntimeError(
            f"expected {EXPECTED_JOBS} completed M2 results, got {result_count}"
        )
    if _base.ordering_audit(jobs) != manifest.get("ordering_audit"):
        raise RuntimeError("M2 completion job order does not match the manifest")
    worker_artifacts = _validate_worker_attestations(
        output_dir, jobs, parent_pid=os.getpid()
    )
    artifacts = _base.collect_result_artifacts(output_dir, jobs)
    validation, outer_rows, child_rows = validate_completed_results(
        output_dir, artifacts
    )
    payload = {
        "schema_version": 1,
        "kind": PAYLOAD_KIND,
        "protocol_sha256": manifest["protocol_sha256"],
        "job_order_sha256": manifest["job_order_sha256"],
        "result_artifacts_sha256": hashlib.sha256(
            _base.hardened._canonical_json(artifacts)
        ).hexdigest(),
        "outer_rows": outer_rows,
        "child_rows": child_rows,
    }
    payload_path = output_dir / _base.ANALYSIS_PAYLOAD_FILENAME
    if payload_path.exists() or payload_path.is_symlink():
        raise RuntimeError("M2 normalized analysis payload is create-only")
    _base.hardened._atomic_write_json(payload_path, payload)
    payload_artifact = _base._stable_file_artifact(payload_path, output_dir)
    warmup = _import_warmup_module()
    warmup_artifact = _base._history_artifact(
        output_dir,
        _base.WARMUP_HISTORY_FILENAME,
        required=True,
        validator=lambda value: warmup.validate_comparator_warmup_history(
            value,
            expected_thread_count=EXPECTED_CHILD_CPUS,
            expected_latest_pid=os.getpid(),
        ),
    )
    if (output_dir / _base.RESUME_HISTORY_FILENAME).exists():
        raise RuntimeError("M2 resume history is forbidden")
    if _base.collect_result_artifacts(output_dir, jobs) != artifacts:
        raise RuntimeError("M2 raw results changed during normalization")
    final_source = _base.collect_source_provenance(
        output_dir=output_dir,
        chimeraboost_path=Path(manifest["source"]["chimeraboost"]["repository"]),
    )
    validate_framework_pins(final_source)
    final_runtime = _base.collect_runtime_provenance()
    if final_source != manifest.get("source"):
        raise RuntimeError("source provenance changed during M2")
    if final_runtime != manifest.get("runtime"):
        raise RuntimeError("runtime provenance changed during M2")
    manifest_path = output_dir / _base.MANIFEST_FILENAME
    attestation = {
        "schema_version": 1,
        "kind": COMPLETION_KIND,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "result_count": result_count,
        "expected_result_count": EXPECTED_JOBS,
        "expected_primary_result_count": EXPECTED_PRIMARY_JOBS,
        "expected_ordinal_diagnostic_result_count": 0,
        "expected_child_fits": EXPECTED_CHILD_FITS,
        "warmup_thread_count": EXPECTED_CHILD_CPUS,
        "warmup_stage_count": len(_base.WARMUP_STAGE_NAMES),
        "protocol_sha256": manifest["protocol_sha256"],
        "job_order_sha256": manifest["job_order_sha256"],
        "git_head": manifest["source"]["git_head"],
        "manifest_sha256": _base.hardened._sha256_file(manifest_path),
        "result_artifacts": artifacts,
        "analysis_payload_artifact": payload_artifact,
        "warmup_history_artifact": warmup_artifact,
        "resume_history_artifact": None,
        "validation": validation,
        "fresh_worker_count": EXPECTED_JOBS,
        "worker_attestation_artifacts": worker_artifacts,
    }
    completion_path = output_dir / _base.COMPLETION_ATTESTATION_FILENAME
    if completion_path.exists() or completion_path.is_symlink():
        raise RuntimeError("M2 completion attestation is create-only")
    _base.hardened._atomic_write_json(
        completion_path, attestation
    )
    return attestation


def _write_terminal(
    output_dir: Path, *, worker_index: int, returncode: int, completed: int
) -> None:
    _write_create_only_json(
        output_dir / TERMINAL_FILENAME,
        {
            "schema_version": 1,
            "kind": CAMPAIGN_KIND + "_terminal",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "worker_index": worker_index,
            "returncode": returncode,
            "completed_worker_count": completed,
            "rerun_same_identity_allowed": False,
        },
    )


def _run_parent(args: argparse.Namespace) -> int:
    output_dir = Path(os.path.abspath(args.output_dir.expanduser()))
    _base._validate_comparator_output_state(output_dir, resume=False)
    chimera_source = activate_chimeraboost_checkout(args.chimeraboost_path)
    context, jobs, ordering, child_cpus = _build_runtime_jobs(args.chimeraboost_path)
    del context
    source = _base.collect_source_provenance(
        output_dir=output_dir, chimeraboost_path=args.chimeraboost_path
    )
    validate_framework_pins(source)
    if source["chimeraboost"] != chimera_source:
        raise RuntimeError("ChimeraBoost source changed during M2 parent setup")
    manifest = _base.build_run_manifest(
        output_dir=output_dir,
        source=source,
        ordering=ordering,
        resolved_child_num_cpus=child_cpus,
    )
    if args.dry_run:
        print(
            "V011_M2_DRY_RUN "
            + json.dumps(
                {
                    "expected_jobs": EXPECTED_JOBS,
                    "protocol_sha256": manifest["protocol_sha256"],
                    "job_order_sha256": manifest["job_order_sha256"],
                    "git_head": source["git_head"],
                },
                sort_keys=True,
            )
        )
        return 0
    manifest = _base.write_or_validate_run_manifest(
        output_dir, manifest, resume=False
    )
    warmup = _import_warmup_module()
    try:
        _base.hardened.record_warmup(
            output_dir, warmup.warmup_tabarena_comparators(thread_count=child_cpus)
        )
    except BaseException:
        _write_terminal(
            output_dir,
            worker_index=-1,
            returncode=-3,
            completed=0,
        )
        raise
    environment = os.environ.copy()
    environment.update(WORKER_ENVIRONMENT)
    completed = 0
    for worker_index in range(EXPECTED_JOBS):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output-dir",
            str(output_dir),
            "--chimeraboost-path",
            str(args.chimeraboost_path),
            "--worker-index",
            str(worker_index),
            "--parent-pid",
            str(os.getpid()),
        ]
        try:
            result = subprocess.run(command, env=environment, cwd=ROOT, check=False)
        except BaseException:
            _write_terminal(
                output_dir,
                worker_index=worker_index,
                returncode=-1,
                completed=completed,
            )
            raise
        if result.returncode != 0:
            _write_terminal(
                output_dir,
                worker_index=worker_index,
                returncode=result.returncode,
                completed=completed,
            )
            raise RuntimeError(
                f"M2 fresh worker {worker_index} failed with {result.returncode}; "
                "this campaign identity is terminal"
            )
        completed += 1
    try:
        write_completion_attestation(
            output_dir,
            manifest=manifest,
            jobs=jobs,
            result_count=completed,
        )
    except BaseException:
        _write_terminal(
            output_dir,
            worker_index=EXPECTED_JOBS,
            returncode=-2,
            completed=completed,
        )
        raise
    print(f"V011_M2_BROAD_PANEL_COMPLETE {completed} {output_dir}")
    return 0


def validate_completed_results(
    output_dir: Path, artifacts: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize the inherited result format for the primary-only grid."""
    outer_rows: list[dict[str, Any]] = []
    child_rows: list[dict[str, Any]] = []
    seen = set()
    resources = set()
    stop_reasons: Counter[str] = Counter()
    inferred_stop_reasons: Counter[str] = Counter()
    deadline_hits = 0
    for relative in sorted(artifacts):
        path = output_dir / relative
        outer, children = _base.parse_result_record(
            _base._decode_result_pickle(path), source=relative
        )
        key = (
            outer["lane"],
            outer["dataset"],
            outer["repeat"],
            outer["fold"],
            outer["arm"],
        )
        if relative != _base.expected_result_relative_path(*key[:4], key[4]):
            raise RuntimeError("result payload is bound to the wrong M2 path")
        if key in seen:
            raise RuntimeError(f"duplicate completed M2 result: {key}")
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
        for child in children:
            fit = child["comparator_fit"]
            reason = fit.get("stop_reason")
            stop_reasons[str(reason) if reason is not None else "unknown"] += 1
            inferred = fit.get("stop_reason_inferred")
            if inferred is not None:
                inferred_stop_reasons[
                    "inferred" if inferred is True else "unresolved"
                ] += 1
            deadline_hits += int(fit.get("deadline_hit") is True)
    if seen != _base.expected_grid():
        raise RuntimeError("completed M2 result grid is incomplete")
    if len(child_rows) != EXPECTED_CHILD_FITS:
        raise RuntimeError(
            f"expected {EXPECTED_CHILD_FITS} M2 child fits, got {len(child_rows)}"
        )
    if resources != {(EXPECTED_CHILD_CPUS, 0, EXPECTED_CHILD_CPUS, 0)}:
        raise RuntimeError("M2 results do not share frozen resources")
    if stop_reasons.get("time_limit", 0) or deadline_hits:
        raise RuntimeError("M2 campaign contains a known deadline stop")
    representation_blocks = _base.validate_cross_engine_representations(child_rows)
    order_rank = {
        key: index for index, key in enumerate(_base.expected_ordered_grid())
    }
    outer_rows.sort(
        key=lambda row: order_rank[
            (
                row["lane"],
                row["dataset"],
                row["repeat"],
                row["fold"],
                row["arm"],
            )
        ]
    )
    child_rows.sort(
        key=lambda row: (
            order_rank[
                (
                    row["lane"],
                    row["dataset"],
                    row["repeat"],
                    row["fold"],
                    row["arm"],
                )
            ],
            row["child_fold"],
        )
    )
    observed_order = [
        (row["lane"], row["dataset"], row["repeat"], row["fold"], row["arm"])
        for row in outer_rows
    ]
    if observed_order != _base.expected_ordered_grid():
        raise RuntimeError("normalized M2 rows lost frozen execution order")
    lane_result_counts = Counter(row["lane"] for row in outer_rows)
    lane_child_counts = Counter(row["lane"] for row in child_rows)
    if lane_result_counts != {PRIMARY_LANE: EXPECTED_JOBS} or lane_child_counts != {
        PRIMARY_LANE: EXPECTED_CHILD_FITS
    }:
        raise RuntimeError("normalized M2 lane counts changed")
    validation = {
        "result_count": len(outer_rows),
        "child_fit_count": len(child_rows),
        "lane_result_counts": dict(sorted(lane_result_counts.items())),
        "lane_child_counts": dict(sorted(lane_child_counts.items())),
        "cross_engine_representation_blocks": representation_blocks,
        "failure_count": 0,
        "imputation_count": 0,
        "known_deadline_hit_count": deadline_hits,
        "known_time_limit_stop_count": int(stop_reasons.get("time_limit", 0)),
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
        "competitor_stop_reason_inference_counts": dict(
            sorted(inferred_stop_reasons.items())
        ),
        "job_order_sha256": _base.job_order_sha256(),
        "resource_allocation": {
            "num_cpus": EXPECTED_CHILD_CPUS,
            "num_gpus": 0,
            "num_cpus_child": EXPECTED_CHILD_CPUS,
            "num_gpus_child": 0,
        },
        "memory_metric": "peak_mem_cpu_minus_min_mem_cpu",
    }
    return validation, outer_rows, child_rows


_OVERRIDES = {
    "LANES": LANES,
    "DIAGNOSTIC_DATASETS": (),
    "ENGINE_SPECS": ENGINE_SPECS,
    "ENGINE_CODES": ENGINE_CODES,
    "ARM_SPECS": ARM_SPECS,
    "ARM_BY_LANE_CODE": ARM_BY_LANE_CODE,
    "EXPECTED_PRIMARY_COORDINATES": EXPECTED_PRIMARY_COORDINATES,
    "EXPECTED_DIAGNOSTIC_COORDINATES": EXPECTED_DIAGNOSTIC_COORDINATES,
    "EXPECTED_COORDINATES": EXPECTED_COORDINATES,
    "EXPECTED_PRIMARY_JOBS": EXPECTED_PRIMARY_JOBS,
    "EXPECTED_DIAGNOSTIC_JOBS": EXPECTED_DIAGNOSTIC_JOBS,
    "EXPECTED_JOBS": EXPECTED_JOBS,
    "EXPECTED_CHILD_FITS": EXPECTED_CHILD_FITS,
    "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
    "CHIMERABOOST_TAG_COMMIT": CHIMERABOOST_TAG_COMMIT,
    "CHIMERABOOST_VERSION": CHIMERABOOST_VERSION,
    "DEFAULT_CHIMERABOOST_PATH": DEFAULT_CHIMERABOOST_PATH,
    "DEFAULT_OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
    "DEFAULT_ANALYSIS_OUTPUT_FILENAMES": DEFAULT_ANALYSIS_OUTPUT_FILENAMES,
    "CAMPAIGN_KIND": CAMPAIGN_KIND,
    "COMPLETION_KIND": COMPLETION_KIND,
    "PAYLOAD_KIND": PAYLOAD_KIND,
    "WARMUP_KIND": WARMUP_KIND,
    "CAMPAIGN_WARMUP_KIND": CAMPAIGN_WARMUP_KIND,
    "SOURCE_FILES": SOURCE_FILES,
    "frozen_protocol": frozen_protocol,
    "protocol_sha256": protocol_sha256,
    "activate_chimeraboost_checkout": activate_chimeraboost_checkout,
    "validate_completed_results": validate_completed_results,
}


@contextmanager
def configured_base() -> Iterator[Any]:
    saved = {name: getattr(_base, name) for name in _OVERRIDES}
    try:
        for name, value in _OVERRIDES.items():
            setattr(_base, name, value)
        yield _base
    finally:
        for name, value in saved.items():
            setattr(_base, name, value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--chimeraboost-path", type=Path, default=DEFAULT_CHIMERABOOST_PATH
    )
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.time_limit != TIME_LIMIT_SECONDS:
        parser.error(f"--time-limit is frozen at {TIME_LIMIT_SECONDS:g} seconds")
    if (args.worker_index is None) != (args.parent_pid is None):
        parser.error("internal worker arguments must be supplied together")
    if args.worker_index is not None and args.dry_run:
        parser.error("an M2 worker cannot be a dry run")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contract = load_contract()
    validate_execution_source_pin(contract)
    with configured_base():
        if args.worker_index is not None:
            return _run_worker(args)
        return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
