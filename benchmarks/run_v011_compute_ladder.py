#!/usr/bin/env python3
"""Run the frozen DarkoFit v0.11 versus ChimeraBoost v0.20 compute ladder."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    _REPO_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from benchmarks import run_tabarena_regression_same_machine as historical_m2


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
PROTOCOL_PATH = BENCH / "v011_compute_ladder_protocol_20260722.md"
CONTRACT_PATH = BENCH / "v011_compute_ladder_contract_20260722.json"
ANALYZER_PATH = BENCH / "analyze_v011_compute_ladder.py"
FREEZER_PATH = BENCH / "freeze_v011_compute_ladder.py"
DEFAULT_OUTPUT_DIR = Path(".cache/v011-compute-ladder-20260722")
DEFAULT_DARKOFIT_SOURCE = Path("/private/tmp/darkofit-v011-release-source")
DEFAULT_CHIMERABOOST_SOURCE = Path("/private/tmp/chimeraboost-v020-release-source")
DEFAULT_TABARENA_SOURCE = Path("/private/tmp/tabarena-m2-4cd1d25")

CONTRACT_ID = "v011-release-compute-ladder-20260722-v1"
DARKOFIT_VERSION = "0.11.0"
DARKOFIT_COMMIT = "0b820e332cec2c083b1dd89eef0fe306d69cfc0e"
DARKOFIT_TAG = "v0.11.0"
CHIMERABOOST_VERSION = "0.20.0"
CHIMERABOOST_COMMIT = "7d48e053e5bd3c7aded1126871aeb0f1f6b84c46"
CHIMERABOOST_TAG = "v0.20.0"
CHIMERABOOST_RELEASE_PUBLISHED_AT = "2026-07-21T02:44:50Z"
CHIMERABOOST_RELEASE_REPOSITORY = "bbstats/chimeraboost"
TABARENA_COMMIT = "4cd1d2526874962daae048a6f2dcf34aa272f3fa"
TABARENA_TREE = "a293df372a613c7358ba5fcd746f58d580cde7d6"
TABARENA_VERSION = "0.0.1"
THREADS = 14
WORKER_TIMEOUT_SECONDS = 7_200.0
RSS_INTERVAL_SECONDS = 0.005
PREDICTION_PILOTS = 3
PREDICTION_TARGET_SECONDS = 1.0
PREDICTION_MIN_SECONDS = 0.5
PREDICTION_MIN_CALLS = 3
PREDICTION_MAX_CALLS = 65_536
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_260_722
WORKER_PREFIX = "V011_COMPUTE_LADDER_RESULT="

TASKS = dict(historical_m2.TASKS)
TASK_SPLIT_COUNTS = dict(historical_m2.TASK_SPLIT_COUNTS)
COORDINATE_PAIRS = tuple(historical_m2.PRIMARY_COORDINATE_PAIRS)

DARKO_DEFAULT = "darkofit_v011_default"
DARKO_ACCURACY = "darkofit_v011_accuracy"
DARKO_ENSEMBLE = "darkofit_v011_ensemble8"
CHIMERA_DEFAULT = "chimeraboost_v020_default"
CHIMERA_ACCURACY = "chimeraboost_v020_depth10"
CHIMERA_ENSEMBLE = "chimeraboost_v020_ensemble8"

ARM_SPECS: dict[str, dict[str, Any]] = {
    DARKO_DEFAULT: {
        "code": "D0",
        "engine": "darkofit",
        "profile": "default",
        "config": {},
    },
    DARKO_ACCURACY: {
        "code": "DA",
        "engine": "darkofit",
        "profile": "accuracy",
        "config": {"preset": "accuracy"},
    },
    DARKO_ENSEMBLE: {
        "code": "D8",
        "engine": "darkofit",
        "profile": "ensemble",
        "config": {"ensemble_mode": "v3", "n_ensembles": 8},
    },
    CHIMERA_DEFAULT: {
        "code": "M0",
        "engine": "chimeraboost",
        "profile": "default",
        "config": {},
    },
    CHIMERA_ACCURACY: {
        "code": "MA",
        "engine": "chimeraboost",
        "profile": "accuracy",
        "config": {"depth": 10},
    },
    CHIMERA_ENSEMBLE: {
        "code": "M8",
        "engine": "chimeraboost",
        "profile": "ensemble",
        "config": {"n_ensembles": 8},
    },
}
BASE_ORDER = tuple(ARM_SPECS)
EXPECTED_COORDINATES = len(TASKS) * len(COORDINATE_PAIRS)
EXPECTED_WORKERS = EXPECTED_COORDINATES * len(ARM_SPECS)

WORKER_ENVIRONMENT = {
    "OMP_NUM_THREADS": str(THREADS),
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": str(THREADS),
    "MKL_NUM_THREADS": str(THREADS),
    "MKL_DYNAMIC": "FALSE",
    "NUMEXPR_NUM_THREADS": str(THREADS),
    "NUMBA_NUM_THREADS": str(THREADS),
    "DARKOFIT_WARMUP": "0",
    "CHIMERABOOST_WARMUP": "0",
    "PYTHONHASHSEED": "0",
}

BOUND_PATHS = {
    "governing_plan": Path("BEAT_CHIMERABOOST_PLAN.md"),
    "agent_rules": Path("AGENTS.md"),
    "protocol": PROTOCOL_PATH.relative_to(ROOT),
    "runner": Path("benchmarks/run_v011_compute_ladder.py"),
    "analyzer": Path("benchmarks/analyze_v011_compute_ladder.py"),
    "freezer": Path("benchmarks/freeze_v011_compute_ladder.py"),
    "tests": Path("tests/test_v011_compute_ladder.py"),
    "historical_task_registry": Path(
        "benchmarks/run_tabarena_regression_same_machine.py"
    ),
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_record(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"bound path is not a regular file: {path}")
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": int(metadata.st_size),
        "sha256": sha256(path),
    }


def expected_coordinates() -> list[tuple[str, int, int]]:
    return [
        (dataset, repeat, fold)
        for dataset in TASKS
        for repeat, fold in COORDINATE_PAIRS
    ]


def expected_ordered_grid() -> list[tuple[str, int, int, str]]:
    rows: list[tuple[str, int, int, str]] = []
    for index, (dataset, repeat, fold) in enumerate(expected_coordinates()):
        shift = index % len(BASE_ORDER)
        order = BASE_ORDER[shift:] + BASE_ORDER[:shift]
        rows.extend((dataset, repeat, fold, arm) for arm in order)
    if len(rows) != EXPECTED_WORKERS or len(set(rows)) != EXPECTED_WORKERS:
        raise RuntimeError("compute-ladder ordered grid is incomplete")
    return rows


def position_audit() -> dict[str, list[int]]:
    counts = {arm: [0] * len(BASE_ORDER) for arm in BASE_ORDER}
    for index, _coordinate in enumerate(expected_coordinates()):
        shift = index % len(BASE_ORDER)
        order = BASE_ORDER[shift:] + BASE_ORDER[:shift]
        for position, arm in enumerate(order):
            counts[arm][position] += 1
    if any(max(values) - min(values) > 1 for values in counts.values()):
        raise RuntimeError("compute-ladder order is not position-balanced")
    return counts


def ordered_grid_sha256() -> str:
    rows = [
        {
            "dataset": dataset,
            "repeat": repeat,
            "fold": fold,
            "arm": arm,
            "code": ARM_SPECS[arm]["code"],
        }
        for dataset, repeat, fold, arm in expected_ordered_grid()
    ]
    return _sha256_bytes(_canonical_json(rows))


def execution_spec() -> dict[str, Any]:
    return {
        "contract_id": CONTRACT_ID,
        "darkofit": {
            "version": DARKOFIT_VERSION,
            "tag": DARKOFIT_TAG,
            "commit": DARKOFIT_COMMIT,
        },
        "chimeraboost": {
            "version": CHIMERABOOST_VERSION,
            "tag": CHIMERABOOST_TAG,
            "commit": CHIMERABOOST_COMMIT,
            "release_repository": CHIMERABOOST_RELEASE_REPOSITORY,
            "published_at": CHIMERABOOST_RELEASE_PUBLISHED_AT,
            "must_remain_latest_before_worker_zero": True,
        },
        "tabarena": {
            "version": TABARENA_VERSION,
            "commit": TABARENA_COMMIT,
            "tree": TABARENA_TREE,
            "role": "task_and_registered_split_source_only",
        },
        "tasks": [
            {
                "dataset": dataset,
                "task_id": TASKS[dataset],
                "registered_split_count": TASK_SPLIT_COUNTS[dataset][1],
            }
            for dataset in TASKS
        ],
        "coordinates": [
            {"dataset": dataset, "repeat": repeat, "fold": fold}
            for dataset, repeat, fold in expected_coordinates()
        ],
        "arms": ARM_SPECS,
        "order": {
            "base": list(BASE_ORDER),
            "method": "cyclic_latin_rotation",
            "position_counts": position_audit(),
            "ordered_grid_sha256": ordered_grid_sha256(),
        },
        "workers": {
            "count": EXPECTED_WORKERS,
            "fresh_process_per_row": True,
            "sequential": True,
            "resume": False,
            "timeout_seconds": WORKER_TIMEOUT_SECONDS,
            "environment": WORKER_ENVIRONMENT,
        },
        "resources": {
            "threads": THREADS,
            "gpus": 0,
            "chimera_ensemble_parallelism": "public_default_equal_total_budget",
            "darkofit_ensemble_parallelism": "public_v3_sequential",
        },
        "prediction_timing": {
            "pilots": PREDICTION_PILOTS,
            "target_seconds": PREDICTION_TARGET_SECONDS,
            "minimum_seconds": PREDICTION_MIN_SECONDS,
            "minimum_calls": PREDICTION_MIN_CALLS,
            "maximum_calls": PREDICTION_MAX_CALLS,
            "actual_registered_test_batch": True,
        },
        "rss": {
            "scope": "worker_plus_recursive_children",
            "interval_seconds": RSS_INTERVAL_SECONDS,
            "absolute_peak": True,
            "peak_minus_start": True,
        },
        "warmup": {
            "outside_measurement": True,
            "same_arm_routes": True,
            "reduced_iteration_budget": True,
        },
        "product_direct": True,
        "autogluon_outer_bag": False,
        "output_create_only": True,
        "silent_worker_retry": False,
    }


def analysis_spec() -> dict[str, Any]:
    return {
        "reference_arm": CHIMERA_DEFAULT,
        "equal_dataset_geometric_mean": True,
        "coordinate_pairing": True,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "coordinates_within_each_fixed_dataset",
            "percentiles": [2.5, 50.0, 97.5],
            "datasets_random_population": False,
        },
        "metrics": [
            "test_rmse",
            "fit_seconds",
            "prediction_seconds_per_call",
            "fit_rss_peak_bytes",
            "fit_rss_peak_delta_bytes",
        ],
        "head_to_head": ["coordinate_wlt", "dataset_wlt"],
        "counterpart_contrasts": [
            [DARKO_DEFAULT, CHIMERA_DEFAULT],
            [DARKO_ACCURACY, CHIMERA_ACCURACY],
            [DARKO_ENSEMBLE, CHIMERA_ENSEMBLE],
        ],
        "frontiers": {
            "axes": ["fit_seconds", "prediction_seconds_per_call"],
            "reference": CHIMERA_DEFAULT,
            "y": "test_rmse_ratio_to_chimeraboost_default",
            "within_engine_dominated_points_removed_per_axis": True,
            "union_observed_budgets": True,
            "interpolation": False,
            "dominance_requires_all_comparable_budgets": True,
        },
        "strict_program_verdict": {
            "basis": "predeclared_equal_dataset_point_estimates",
            "uncertainty_adjacent_not_certificate": True,
            "fit_frontier_dominance": True,
            "prediction_frontier_dominance": True,
            "counterpart_peak_rss_no_worse": True,
        },
    }


def claim_spec() -> dict[str, Any]:
    return {
        "tier": "E",
        "spent_descriptive_release_scoreboard": True,
        "default_or_policy_advancement": False,
        "fresh_confirmation": False,
        "lockbox": False,
        "tabarena_placement": False,
        "catboost_comparison": False,
        "no_rerun_to_improve": True,
    }


def protocol_sha256() -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "execution": execution_spec(),
                "analysis": analysis_spec(),
                "claims": claim_spec(),
            }
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    metadata = path.lstat()
    if not path.is_file() or path.is_symlink() or metadata.st_size <= 0:
        raise RuntimeError(f"unsafe JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return value


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    path = path.expanduser().resolve()
    payload = _read_json(path)
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
            "execution",
            "analysis",
            "claims",
        }
        or payload.get("schema_version") != 1
        or payload.get("contract_id") != CONTRACT_ID
        or payload.get("contract_frozen") is not True
        or payload.get("outcome_blind") is not True
        or payload.get("authorization")
        != "BEAT_CHIMERABOOST_PLAN.md Phase E release milestone"
        or payload.get("execution") != execution_spec()
        or payload.get("analysis") != analysis_spec()
        or payload.get("claims") != claim_spec()
        or payload.get("protocol_sha256") != protocol_sha256()
        or set(payload.get("bindings", {})) != set(BOUND_PATHS)
    ):
        raise RuntimeError("compute-ladder contract is invalid")
    for name, relative in BOUND_PATHS.items():
        if payload["bindings"][name] != _bound_record(ROOT / relative):
            raise RuntimeError(f"compute-ladder contract binding drifted: {name}")
    freeze_head = str(payload.get("harness_freeze_git_head", ""))
    if len(freeze_head) != 40:
        raise RuntimeError("compute-ladder harness freeze commit is invalid")
    return payload


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: float | None = None,
) -> str:
    completed = subprocess.run(
        list(command),
        cwd=None if cwd is None else str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _git(repo: Path, *args: str) -> str:
    return _run(["git", *args], cwd=repo)


def validate_execution_source_pin(contract: Mapping[str, Any]) -> str:
    freeze_head = str(contract["harness_freeze_git_head"])
    current = _git(ROOT, "rev-parse", "HEAD")
    parents = _git(ROOT, "rev-list", "--parents", "-n", "1", current).split()
    if parents != [current, freeze_head]:
        raise RuntimeError("execution source is not the direct contract child")
    changed = _git(ROOT, "diff", "--name-only", freeze_head, current).splitlines()
    if changed != [str(CONTRACT_PATH.relative_to(ROOT))]:
        raise RuntimeError("contract commit changed more than the contract")
    if _git(ROOT, "rev-parse", "origin/main") != current:
        raise RuntimeError("compute-ladder execution source is not published")
    return current


def _validate_source_checkout(
    path: Path,
    *,
    commit: str,
    tag: str,
    package_init: str,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_dir() or not (path / package_init).is_file():
        raise RuntimeError(f"product source checkout is missing: {path}")
    if _git(path, "rev-parse", "HEAD") != commit:
        raise RuntimeError(f"product source is not exact commit {commit}")
    if _git(path, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError(f"product source checkout is not clean: {path}")
    if tag not in _git(path, "tag", "--points-at", "HEAD").splitlines():
        raise RuntimeError(f"product source does not carry tag {tag}")
    return {
        "path": str(path),
        "commit": commit,
        "tree": _git(path, "rev-parse", "HEAD^{tree}"),
        "tag": tag,
        "status": "",
        "package_init_sha256": sha256(path / package_init),
    }


def validate_tabarena_source(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    module = path / "packages/tabarena/src/tabarena/__init__.py"
    if not module.is_file():
        raise RuntimeError(f"TabArena source checkout is missing: {path}")
    if _git(path, "rev-parse", "HEAD") != TABARENA_COMMIT:
        raise RuntimeError("TabArena source commit drifted")
    if _git(path, "rev-parse", "HEAD^{tree}") != TABARENA_TREE:
        raise RuntimeError("TabArena source tree drifted")
    if _git(path, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("TabArena source checkout is not clean")
    return {
        "path": str(path),
        "commit": TABARENA_COMMIT,
        "tree": TABARENA_TREE,
        "status": "",
        "module_sha256": sha256(module),
    }


def validate_product_sources(
    darkofit_source: Path,
    chimeraboost_source: Path,
    tabarena_source: Path,
) -> dict[str, dict[str, Any]]:
    """Validate every immutable product/data checkout at one time boundary."""
    return {
        "darkofit": _validate_source_checkout(
            darkofit_source,
            commit=DARKOFIT_COMMIT,
            tag=DARKOFIT_TAG,
            package_init="darkofit/__init__.py",
        ),
        "chimeraboost": _validate_source_checkout(
            chimeraboost_source,
            commit=CHIMERABOOST_COMMIT,
            tag=CHIMERABOOST_TAG,
            package_init="chimeraboost/__init__.py",
        ),
        "tabarena": validate_tabarena_source(tabarena_source),
    }


def validate_latest_chimeraboost_release() -> dict[str, Any]:
    raw = _run(
        [
            "gh",
            "api",
            f"repos/{CHIMERABOOST_RELEASE_REPOSITORY}/releases/latest",
        ],
        timeout=30.0,
    )
    payload = json.loads(raw)
    observed = {
        "tag_name": payload.get("tag_name"),
        "published_at": payload.get("published_at"),
        "html_url": payload.get("html_url"),
    }
    if (
        observed["tag_name"] != CHIMERABOOST_TAG
        or observed["published_at"] != CHIMERABOOST_RELEASE_PUBLISHED_AT
    ):
        raise RuntimeError(
            "ChimeraBoost latest release changed after protocol freeze: " f"{observed}"
        )
    observed["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    return observed


def _hardware() -> dict[str, Any]:
    import psutil

    logical = psutil.cpu_count(logical=True)
    physical = psutil.cpu_count(logical=False)
    if logical != THREADS or physical != THREADS:
        raise RuntimeError(
            f"compute ladder requires the frozen 14/14 CPU host, got "
            f"{logical}/{physical}"
        )
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "logical_cpus": logical,
        "physical_cpus": physical,
        "memory_bytes": int(psutil.virtual_memory().total),
    }


def _exclusive_machine_audit() -> dict[str, Any]:
    import psutil

    conflicts = []
    self_pid = os.getpid()
    markers = (
        "run_v011_compute_ladder",
        "run_v011_m2_broad_panel",
        "run_v011_ensemble_evidence",
        "run_m3",
        "run_tabarena",
    )
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess, ValueError):
            continue
        if pid != self_pid and any(marker in command for marker in markers):
            conflicts.append({"pid": pid, "command": command})
    if conflicts:
        raise RuntimeError(f"another benchmark process is active: {conflicts}")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "conflicting_benchmark_processes": [],
        "load_average": [float(value) for value in os.getloadavg()],
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


def _stable_artifact(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"unsafe artifact: {path}")
    return {
        "path": str(path.relative_to(root)),
        "bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }


def _json_safe(value: Any, *, field: str = "value") -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise RuntimeError(f"{field} is nonfinite")
        return value
    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RuntimeError(f"{field} has a non-string key")
            output[key] = _json_safe(item, field=f"{field}.{key}")
        return output
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise RuntimeError(f"{field} has unsupported type {type(value)!r}")


def _loaded_package_matches_source(package: str, source: Path) -> bool:
    root_module = sys.modules.get(package)
    root_path = None if root_module is None else getattr(root_module, "__file__", None)
    if root_path is None:
        return False
    names = [
        name
        for name in sys.modules
        if name == package or name.startswith(f"{package}.")
    ]
    for name in names:
        module_path = getattr(sys.modules[name], "__file__", None)
        if module_path is None:
            continue
        try:
            Path(module_path).resolve().relative_to(source)
        except ValueError:
            return False
    return True


def _activate_product_sources(darkofit_source: Path, chimera_source: Path) -> None:
    darkofit_source = darkofit_source.resolve()
    chimera_source = chimera_source.resolve()
    for package, source in (
        ("darkofit", darkofit_source),
        ("chimeraboost", chimera_source),
    ):
        if _loaded_package_matches_source(package, source):
            continue
        for name in list(sys.modules):
            if name == package or name.startswith(f"{package}."):
                sys.modules.pop(name, None)
    blocked = {str(ROOT.resolve()), str(darkofit_source), str(chimera_source)}
    sys.path[:] = [
        entry for entry in sys.path if str(Path(entry or ".").resolve()) not in blocked
    ]
    sys.path.insert(0, str(chimera_source))
    sys.path.insert(0, str(darkofit_source))
    importlib.invalidate_caches()


def _implementation(model: Any, expected_source: Path) -> dict[str, Any]:
    module = importlib.import_module(model.__class__.__module__)
    path = Path(inspect.getfile(module)).resolve()
    try:
        path.relative_to(expected_source.resolve())
    except ValueError as exc:
        raise RuntimeError(f"model imported from unexpected source: {path}") from exc
    return {
        "class": model.__class__.__name__,
        "module": model.__class__.__module__,
        "module_path": str(path),
        "module_sha256": sha256(path),
    }


def _coordinate_seed(repeat: int, fold: int) -> int:
    if (repeat, fold) not in COORDINATE_PAIRS:
        raise ValueError("coordinate is outside the frozen grid")
    return 1_000 * int(repeat) + int(fold)


def _build_model(
    arm: str,
    *,
    seed: int,
    darkofit_source: Path,
    chimeraboost_source: Path,
) -> tuple[Any, Path]:
    _activate_product_sources(darkofit_source, chimeraboost_source)
    spec = ARM_SPECS[arm]
    if spec["engine"] == "darkofit":
        from darkofit import DarkoRegressor, __version__

        if __version__ != DARKOFIT_VERSION:
            raise RuntimeError("imported DarkoFit version drifted")
        model = DarkoRegressor(
            random_state=seed,
            thread_count=THREADS,
            diagnostic_warnings="never",
            **spec["config"],
        )
        return model, darkofit_source
    from chimeraboost import ChimeraBoostRegressor, __version__

    if __version__ != CHIMERABOOST_VERSION:
        raise RuntimeError("imported ChimeraBoost version drifted")
    model = ChimeraBoostRegressor(
        random_state=seed,
        thread_count=THREADS,
        **spec["config"],
    )
    return model, chimeraboost_source


def _load_split(
    dataset: str,
    repeat: int,
    fold: int,
    tabarena_source: Path,
) -> dict[str, Any]:
    from importlib.metadata import version as distribution_version

    from tabarena.benchmark.task.spec import task_spec_from_task_id_str
    from tabarena.contexts import TabArenaContext
    import tabarena

    module_path = Path(tabarena.__file__).resolve()
    try:
        module_path.relative_to(tabarena_source.resolve())
    except ValueError as exc:
        raise RuntimeError("TabArena imported from an unpinned source") from exc
    if distribution_version("tabarena") != TABARENA_VERSION:
        raise RuntimeError("TabArena package version drifted")
    context = TabArenaContext()
    metadata = context.task_metadata_collection.task_metadata_by_dataset()[dataset]
    if int(metadata.task_id_str) != TASKS[dataset]:
        raise RuntimeError("TabArena task ID drifted")
    task = (
        task_spec_from_task_id_str(metadata.task_id_str)
        .with_task_metadata(metadata)
        .load()
    )
    if task.problem_type != "regression" or task.eval_metric != "rmse":
        raise RuntimeError("compute-ladder task is not RMSE regression")
    X_train, y_train, X_test, y_test = task.get_train_test_split(
        fold=fold,
        repeat=repeat,
    )
    return {
        "task_id": int(task.task_id),
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def _pandas_sha256(value: Any) -> str:
    import pandas as pd

    if isinstance(value, pd.Series):
        frame = value.to_frame(name=str(value.name))
    else:
        frame = value
    schema = [
        {"column": str(column), "dtype": str(frame[column].dtype)}
        for column in frame.columns
    ]
    hashed = pd.util.hash_pandas_object(
        frame,
        index=True,
        categorize=True,
    ).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update(_canonical_json(schema))
    digest.update(hashed.tobytes(order="C"))
    return digest.hexdigest()


def _split_fingerprints(data: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        name: _pandas_sha256(data[name])
        for name in ("X_train", "y_train", "X_test", "y_test")
    }
    fields["combined_sha256"] = _sha256_bytes(_canonical_json(fields))
    return fields


def _categorical_features(X: Any) -> list[str]:
    return [str(column) for column in X.select_dtypes(include=["category"]).columns]


def _fit(model: Any, X: Any, y: Any, cat_features: Sequence[str]) -> Any:
    return model.fit(
        X,
        y,
        cat_features=list(cat_features) or None,
    )


def _warmup(
    arm: str,
    data: Mapping[str, Any],
    *,
    seed: int,
    darkofit_source: Path,
    chimeraboost_source: Path,
) -> dict[str, Any]:
    count = min(512, len(data["y_train"]))
    X = data["X_train"].iloc[:count].copy()
    y = data["y_train"].iloc[:count].copy()
    cat_features = _categorical_features(X)
    spec = ARM_SPECS[arm]
    routes = []
    if arm == DARKO_ACCURACY:
        _activate_product_sources(darkofit_source, chimeraboost_source)
        from darkofit import DarkoRegressor

        for mode in ("catboost", "lightgbm", "hybrid"):
            model = DarkoRegressor(
                iterations=2,
                learning_rate=0.1,
                l2_leaf_reg=3.0,
                max_bins=128,
                early_stopping=False,
                tree_mode=mode,
                random_state=seed,
                thread_count=THREADS,
                diagnostic_warnings="never",
            )
            _fit(model, X, y, cat_features)
            model.predict(X.iloc[: min(32, count)])
            routes.append(f"darkofit_{mode}")
            del model
    else:
        model, _source = _build_model(
            arm,
            seed=seed,
            darkofit_source=darkofit_source,
            chimeraboost_source=chimeraboost_source,
        )
        if spec["engine"] == "darkofit":
            model.set_params(iterations=2)
        else:
            model.set_params(n_estimators=2, early_stopping=False)
        _fit(model, X, y, cat_features)
        model.predict(X.iloc[: min(32, count)])
        routes.append(f"{spec['engine']}_{spec['profile']}")
        del model
    gc.collect()
    return {
        "rows": count,
        "categorical_feature_count": len(cat_features),
        "routes": routes,
        "reduced_iteration_budget": True,
    }


class ProcessTreeRSSSampler:
    """Sample aggregate RSS for this worker and every recursive child."""

    def __init__(self, interval_seconds: float = RSS_INTERVAL_SECONDS):
        self.interval_seconds = float(interval_seconds)
        self.start_bytes = 0
        self.peak_bytes = 0
        self.end_bytes = 0
        self.samples = 0
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def current_bytes() -> int:
        import psutil

        root = psutil.Process()
        total = 0
        seen: set[int] = set()
        for process in (root, *root.children(recursive=True)):
            if process.pid in seen:
                continue
            seen.add(process.pid)
            try:
                total += int(process.memory_info().rss)
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        if total <= 0:
            raise RuntimeError("process-tree RSS is unavailable")
        return total

    def _sample_once(self) -> None:
        value = self.current_bytes()
        self.peak_bytes = max(self.peak_bytes, value)
        self.samples += 1

    def _run_sampler(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._sample_once()
            except Exception as exc:  # pragma: no cover - platform telemetry
                self.errors.append(f"{type(exc).__name__}: {exc}")

    def __enter__(self):
        self.start_bytes = self.current_bytes()
        self.peak_bytes = self.start_bytes
        self.samples = 1
        self._thread = threading.Thread(target=self._run_sampler, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self._sample_once()
            self.end_bytes = self.current_bytes()
            self.peak_bytes = max(self.peak_bytes, self.end_bytes)
        except Exception as cleanup_exc:
            self.errors.append(f"{type(cleanup_exc).__name__}: {cleanup_exc}")
        if exc_type is None and (self.errors or self.samples < 2):
            raise RuntimeError(f"process-tree RSS sampling failed: {self.errors}")
        return False


def _coerce_prediction(value: Any, n_rows: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape == (n_rows, 1):
        value = value.reshape(n_rows)
    if value.shape != (n_rows,) or not np.isfinite(value).all():
        raise RuntimeError(f"invalid regression prediction: {value.shape}")
    return value


def _prediction_array(model: Any, X: Any) -> np.ndarray:
    return _coerce_prediction(model.predict(X), len(X))


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _timed_prediction(model: Any, X: Any) -> dict[str, Any]:
    reference = _prediction_array(model, X)
    pilots = []
    for _ in range(PREDICTION_PILOTS):
        started = time.perf_counter_ns()
        raw_value = model.predict(X)
        pilots.append((time.perf_counter_ns() - started) / 1e9)
        value = _coerce_prediction(raw_value, len(X))
        if not np.array_equal(reference, value):
            raise RuntimeError("prediction pilot changed output")
    pilot_median = float(np.median(np.asarray(pilots, dtype=np.float64)))
    calls = int(
        min(
            PREDICTION_MAX_CALLS,
            max(
                PREDICTION_MIN_CALLS,
                math.ceil(PREDICTION_TARGET_SECONDS / max(pilot_median, 1e-9)),
            ),
        )
    )
    final_raw = None
    gc.disable()
    started = time.perf_counter_ns()
    try:
        for _ in range(calls):
            final_raw = model.predict(X)
    finally:
        elapsed = (time.perf_counter_ns() - started) / 1e9
        gc.enable()
    final = None if final_raw is None else _coerce_prediction(final_raw, len(X))
    if final is None or not np.array_equal(reference, final):
        raise RuntimeError("formal prediction interval changed output")
    if elapsed < PREDICTION_MIN_SECONDS:
        raise RuntimeError(
            f"prediction interval {elapsed:.6f}s missed the frozen floor"
        )
    return {
        "rows": int(len(X)),
        "pilots_seconds": [float(value) for value in pilots],
        "pilot_median_seconds": pilot_median,
        "calls": calls,
        "interval_seconds": float(elapsed),
        "seconds_per_call": float(elapsed / calls),
        "rows_per_second": float(len(X) * calls / elapsed),
        "prediction_sha256": _array_sha256(reference),
    }


def _member_models(model: Any) -> list[Any]:
    members = getattr(model, "estimators_", None)
    if members is None:
        return [model]
    members = list(members)
    if not members:
        raise RuntimeError("fitted ensemble has no members")
    return members


def _model_metadata(model: Any, arm: str) -> dict[str, Any]:
    spec = ARM_SPECS[arm]
    members = _member_models(model)
    member_rows = []
    for member in members:
        core = getattr(member, "model_", None)
        if core is None:
            raise RuntimeError("fitted member core is missing")
        trees = getattr(core, "trees_", None)
        tree_count = len(trees) if trees is not None else 0
        thread_count = int(getattr(core, "n_threads_", -1))
        if thread_count <= 0:
            raise RuntimeError("fitted member thread count is missing")
        row = {
            "tree_count": int(tree_count),
            "thread_count": thread_count,
        }
        if spec["engine"] == "darkofit":
            row["tree_mode"] = str(getattr(core, "tree_mode_", "unknown"))
            row["stop_reason"] = str(getattr(core, "stop_reason_", "unknown"))
        else:
            row["linear_leaves_selected"] = getattr(
                member,
                "linear_leaves_selected_",
                None,
            )
            row["cross_features_selected"] = getattr(
                member,
                "cross_features_selected_",
                None,
            )
            pairs = getattr(member, "cross_pairs_", None)
            row["cross_pair_count"] = 0 if pairs is None else len(pairs)
        member_rows.append(row)
    metadata = {
        "engine": spec["engine"],
        "profile": spec["profile"],
        "public_config": dict(spec["config"]),
        "member_count": len(members),
        "total_tree_count": sum(row["tree_count"] for row in member_rows),
        "members": member_rows,
    }
    if spec["engine"] == "darkofit":
        metadata["preset"] = getattr(model, "preset_", None)
        metadata["tree_mode_selection"] = getattr(
            model,
            "tree_mode_selection_",
            None,
        )
        metadata["ensemble_mode"] = getattr(model, "ensemble_mode", None)
    else:
        metadata["ensemble_n_jobs"] = int(model.ensemble_n_jobs)
        metadata["max_samples"] = float(model.max_samples)
    return _json_safe(metadata, field="model_metadata")


def _worker_result(args: argparse.Namespace) -> dict[str, Any]:
    validate_product_sources(
        args.darkofit_source,
        args.chimeraboost_source,
        args.tabarena_source,
    )
    actual_environment = {key: os.environ.get(key) for key in WORKER_ENVIRONMENT}
    if actual_environment != WORKER_ENVIRONMENT:
        raise RuntimeError(f"worker environment drifted: {actual_environment}")
    import numba

    if (
        int(numba.config.NUMBA_NUM_THREADS) != THREADS
        or int(numba.get_num_threads()) != THREADS
    ):
        raise RuntimeError("worker Numba thread budget is not exactly 14")
    dataset, repeat, fold, arm = expected_ordered_grid()[args.worker_index]
    if args.arm != arm:
        raise RuntimeError("worker arm does not match frozen order")
    data = _load_split(dataset, repeat, fold, args.tabarena_source)
    fingerprints = _split_fingerprints(data)
    seed = _coordinate_seed(repeat, fold)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warmup = _warmup(
            arm,
            data,
            seed=seed,
            darkofit_source=args.darkofit_source,
            chimeraboost_source=args.chimeraboost_source,
        )
        model, expected_source = _build_model(
            arm,
            seed=seed,
            darkofit_source=args.darkofit_source,
            chimeraboost_source=args.chimeraboost_source,
        )
        implementation = _implementation(model, expected_source)
        cat_features = _categorical_features(data["X_train"])
        ambient_before = int(numba.get_num_threads())
        with ProcessTreeRSSSampler() as rss:
            started = time.perf_counter_ns()
            _fit(model, data["X_train"], data["y_train"], cat_features)
            fit_seconds = (time.perf_counter_ns() - started) / 1e9
        ambient_after = int(numba.get_num_threads())
        if ambient_after != ambient_before:
            raise RuntimeError("fit leaked the worker's ambient Numba thread mask")
        prediction = _prediction_array(model, data["X_test"])
        truth = np.asarray(data["y_test"], dtype=np.float64)
        rmse = float(np.sqrt(np.mean(np.square(prediction - truth))))
        if not math.isfinite(rmse) or rmse <= 0.0:
            raise RuntimeError("test RMSE is invalid")
        prediction_timing = _timed_prediction(model, data["X_test"])
        model_metadata = _model_metadata(model, arm)
    return {
        "schema_version": 1,
        "kind": "v011_compute_ladder_worker",
        "worker_index": int(args.worker_index),
        "pid": os.getpid(),
        "parent_pid": int(args.parent_pid),
        "started_at_utc": args.worker_started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "task_id": data["task_id"],
        "repeat": repeat,
        "fold": fold,
        "arm": arm,
        "code": ARM_SPECS[arm]["code"],
        "engine": ARM_SPECS[arm]["engine"],
        "profile": ARM_SPECS[arm]["profile"],
        "seed": seed,
        "train_rows": int(len(data["y_train"])),
        "test_rows": int(len(data["y_test"])),
        "feature_count": int(data["X_train"].shape[1]),
        "categorical_features": cat_features,
        "fingerprints": fingerprints,
        "test_rmse": rmse,
        "fit_seconds": float(fit_seconds),
        "fit_rss": {
            "scope": "worker_plus_recursive_children",
            "start_bytes": int(rss.start_bytes),
            "peak_bytes": int(rss.peak_bytes),
            "peak_delta_bytes": int(max(0, rss.peak_bytes - rss.start_bytes)),
            "end_bytes": int(rss.end_bytes),
            "samples": int(rss.samples),
            "errors": list(rss.errors),
            "interval_seconds": rss.interval_seconds,
        },
        "prediction": prediction_timing,
        "prediction_sha256": _array_sha256(prediction),
        "model": model_metadata,
        "implementation": implementation,
        "warmup": warmup,
        "environment": actual_environment,
        "numba_threads_before_fit": ambient_before,
        "numba_threads_after_fit": ambient_after,
        "warnings": [
            {"category": item.category.__name__, "message": str(item.message)}
            for item in caught
        ],
    }


def _manifest(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    *,
    harness_head: str,
) -> dict[str, Any]:
    sources = validate_product_sources(
        args.darkofit_source,
        args.chimeraboost_source,
        args.tabarena_source,
    )
    return {
        "schema_version": 1,
        "kind": "v011_compute_ladder_manifest",
        "contract_id": CONTRACT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "path": str(args.contract.resolve()),
            "sha256": sha256(args.contract),
            "protocol_sha256": contract["protocol_sha256"],
        },
        "harness_head": harness_head,
        "darkofit_source": sources["darkofit"],
        "chimeraboost_source": sources["chimeraboost"],
        "tabarena_source": sources["tabarena"],
        "latest_chimeraboost_release": validate_latest_chimeraboost_release(),
        "hardware": _hardware(),
        "exclusive_machine": _exclusive_machine_audit(),
        "worker_environment": WORKER_ENVIRONMENT,
        "expected_worker_count": EXPECTED_WORKERS,
        "ordered_grid_sha256": ordered_grid_sha256(),
        "ordered_grid": [
            {
                "dataset": dataset,
                "repeat": repeat,
                "fold": fold,
                "arm": arm,
            }
            for dataset, repeat, fold, arm in expected_ordered_grid()
        ],
    }


def _validate_output_state(output_dir: Path) -> None:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError(f"compute-ladder output is create-only: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(mode=0o755)
    (output_dir / "workers").mkdir(mode=0o755)


def _worker_command(
    args: argparse.Namespace,
    *,
    worker_index: int,
    arm: str,
    parent_pid: int,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--contract",
        str(args.contract.resolve()),
        "--darkofit-source",
        str(args.darkofit_source.resolve()),
        "--chimeraboost-source",
        str(args.chimeraboost_source.resolve()),
        "--tabarena-source",
        str(args.tabarena_source.resolve()),
        "--worker-index",
        str(worker_index),
        "--arm",
        arm,
        "--parent-pid",
        str(parent_pid),
        "--worker-started-at",
        datetime.now(timezone.utc).isoformat(),
    ]


def _parse_worker_stdout(stdout: str) -> dict[str, Any]:
    matches = [line for line in stdout.splitlines() if line.startswith(WORKER_PREFIX)]
    if len(matches) != 1:
        raise RuntimeError("worker did not emit exactly one result marker")
    payload = json.loads(matches[0][len(WORKER_PREFIX) :])
    if not isinstance(payload, dict):
        raise RuntimeError("worker result marker is not an object")
    return payload


def _run_parent(args: argparse.Namespace) -> int:
    contract = load_contract(args.contract)
    harness_head = validate_execution_source_pin(contract)
    if _git(ROOT, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("formal compute ladder requires a clean harness checkout")
    output_dir = args.output_dir.expanduser().resolve()
    _validate_output_state(output_dir)
    manifest_path = output_dir / "manifest.json"
    parent_pid = os.getpid()
    worker_artifacts = []
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        manifest = _manifest(args, contract, harness_head=harness_head)
        _write_create_only_json(manifest_path, manifest)
        for worker_index, (_dataset, _repeat, _fold, arm) in enumerate(
            expected_ordered_grid()
        ):
            environment = os.environ.copy()
            environment.update(WORKER_ENVIRONMENT)
            completed = subprocess.run(
                _worker_command(
                    args,
                    worker_index=worker_index,
                    arm=arm,
                    parent_pid=parent_pid,
                ),
                cwd=str(ROOT),
                env=environment,
                capture_output=True,
                text=True,
                timeout=WORKER_TIMEOUT_SECONDS,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"worker {worker_index} failed ({completed.returncode}):\n"
                    f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
                )
            payload = _parse_worker_stdout(completed.stdout)
            payload["launcher_output"] = {
                "returncode": int(completed.returncode),
                "stdout_without_result": [
                    line
                    for line in completed.stdout.splitlines()
                    if not line.startswith(WORKER_PREFIX)
                ],
                "stderr": completed.stderr,
            }
            expected = expected_ordered_grid()[worker_index]
            if (
                payload.get("worker_index") != worker_index
                or (
                    payload.get("dataset"),
                    payload.get("repeat"),
                    payload.get("fold"),
                    payload.get("arm"),
                )
                != expected
                or payload.get("parent_pid") != parent_pid
            ):
                raise RuntimeError(f"worker {worker_index} identity drifted")
            worker_path = output_dir / "workers" / f"{worker_index:03d}.json"
            _write_create_only_json(worker_path, payload)
            worker_artifacts.append(_stable_artifact(worker_path, output_dir))
            print(
                f"compute ladder {worker_index + 1}/{EXPECTED_WORKERS}: "
                f"{expected[0]} r{expected[1]}f{expected[2]} {arm}",
                flush=True,
            )
        completed_at = datetime.now(timezone.utc).isoformat()
        raw = {
            "schema_version": 1,
            "kind": "v011_compute_ladder_raw",
            "contract_id": CONTRACT_ID,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "manifest": _stable_artifact(manifest_path, output_dir),
            "workers": worker_artifacts,
            "rows": [
                _read_json(output_dir / artifact["path"])
                for artifact in worker_artifacts
            ],
        }
        final_sources = validate_product_sources(
            args.darkofit_source,
            args.chimeraboost_source,
            args.tabarena_source,
        )
        if final_sources != {
            "darkofit": manifest["darkofit_source"],
            "chimeraboost": manifest["chimeraboost_source"],
            "tabarena": manifest["tabarena_source"],
        }:
            raise RuntimeError("product or data source changed during execution")
        raw_path = output_dir / "raw.json"
        _write_create_only_json(raw_path, raw)
        terminal = {
            "schema_version": 1,
            "kind": "v011_compute_ladder_terminal",
            "status": "complete",
            "contract_id": CONTRACT_ID,
            "completed_worker_count": len(worker_artifacts),
            "raw": _stable_artifact(raw_path, output_dir),
            "completed_at_utc": completed_at,
        }
        _write_create_only_json(output_dir / "terminal.json", terminal)
        return 0
    except BaseException as exc:
        terminal_path = output_dir / "terminal.json"
        if not terminal_path.exists():
            _write_create_only_json(
                terminal_path,
                {
                    "schema_version": 1,
                    "kind": "v011_compute_ladder_terminal",
                    "status": "failed",
                    "contract_id": CONTRACT_ID,
                    "completed_worker_count": len(worker_artifacts),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--darkofit-source",
        type=Path,
        default=DEFAULT_DARKOFIT_SOURCE,
    )
    parser.add_argument(
        "--chimeraboost-source",
        type=Path,
        default=DEFAULT_CHIMERABOOST_SOURCE,
    )
    parser.add_argument(
        "--tabarena-source",
        type=Path,
        default=DEFAULT_TABARENA_SOURCE,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--worker-index", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--arm", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-started-at",
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    worker_fields = (
        args.worker_index,
        args.arm,
        args.parent_pid,
        args.worker_started_at,
    )
    if any(value is not None for value in worker_fields) and not all(
        value is not None for value in worker_fields
    ):
        parser.error("internal worker arguments must be supplied together")
    if args.worker_index is not None:
        if args.worker_index not in range(EXPECTED_WORKERS):
            parser.error("worker index is outside the frozen grid")
        if args.arm not in ARM_SPECS:
            parser.error("worker arm is invalid")
        if args.dry_run:
            parser.error("worker mode cannot be a dry run")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_index is not None:
        load_contract(args.contract)
        result = _worker_result(args)
        print(WORKER_PREFIX + json.dumps(result, allow_nan=False, sort_keys=True))
        return 0
    contract = load_contract(args.contract)
    harness_head = validate_execution_source_pin(contract)
    if args.dry_run:
        manifest = _manifest(args, contract, harness_head=harness_head)
        print(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True))
        return 0
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
