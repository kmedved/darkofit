#!/usr/bin/env python3
"""Run the DarkoFit v0.12 versus ChimeraBoost v0.23 release compute ladder."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    _REPO_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
PROTOCOL_PATH = BENCH / "v012_compute_ladder_protocol_20260724.md"
ANALYZER_PATH = BENCH / "analyze_v012_compute_ladder.py"
DEFAULT_OUTPUT_DIR = Path(".cache/v012-compute-ladder-20260724")
DEFAULT_DARKOFIT_SOURCE = Path("/private/tmp/darkofit-v012-release-source")
DEFAULT_CHIMERABOOST_SOURCE = Path("/private/tmp/chimeraboost-v023-release-source")
DEFAULT_TABARENA_SOURCE = Path("/private/tmp/tabarena-m2-4cd1d25")


def _load_isolated_legacy_worker() -> ModuleType:
    name = "benchmarks._v012_isolated_legacy_worker"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = BENCH / "run_v011_compute_ladder.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load legacy worker helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


legacy = _load_isolated_legacy_worker()

RUN_ID = "v012-release-compute-ladder-20260724"
DARKOFIT_VERSION = "0.12.0"
DARKOFIT_COMMIT = "a9eb4dbbf8af0e6db42e9ace433e7a267c80fca7"
DARKOFIT_TAG = "v0.12.0"
CHIMERABOOST_VERSION = "0.23.0"
CHIMERABOOST_COMMIT = "6667843b8970454b0f582ffd1ab2be033989c578"
CHIMERABOOST_TAG = "v0.23.0"
CHIMERABOOST_RELEASE_PUBLISHED_AT = "2026-07-24T01:06:38Z"
CHIMERABOOST_RELEASE_REPOSITORY = "bbstats/chimeraboost"
TABARENA_COMMIT = legacy.TABARENA_COMMIT
TABARENA_TREE = legacy.TABARENA_TREE
TABARENA_VERSION = legacy.TABARENA_VERSION

THREADS = 14
WORKER_TIMEOUT_SECONDS = legacy.WORKER_TIMEOUT_SECONDS
RSS_INTERVAL_SECONDS = legacy.RSS_INTERVAL_SECONDS
PREDICTION_PILOTS = legacy.PREDICTION_PILOTS
PREDICTION_TARGET_SECONDS = legacy.PREDICTION_TARGET_SECONDS
PREDICTION_MIN_SECONDS = legacy.PREDICTION_MIN_SECONDS
PREDICTION_MIN_CALLS = legacy.PREDICTION_MIN_CALLS
PREDICTION_MAX_CALLS = legacy.PREDICTION_MAX_CALLS
BOOTSTRAP_DRAWS = legacy.BOOTSTRAP_DRAWS
BOOTSTRAP_SEED = 20_260_724
WORKER_PREFIX = "V012_COMPUTE_LADDER_RESULT="

TASKS = dict(legacy.TASKS)
TASK_SPLIT_COUNTS = dict(legacy.TASK_SPLIT_COUNTS)
COORDINATE_PAIRS = tuple(legacy.COORDINATE_PAIRS)

DARKO_DEFAULT = "darkofit_v012_default"
DARKO_ACCURACY = "darkofit_v012_accuracy"
DARKO_ENSEMBLE = "darkofit_v012_ensemble8"
CHIMERA_DEFAULT = "chimeraboost_v023_default"
CHIMERA_ACCURACY = "chimeraboost_v023_depth10"
CHIMERA_ENSEMBLE = "chimeraboost_v023_ensemble8"

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
    **legacy.WORKER_ENVIRONMENT,
    "OMP_NUM_THREADS": str(THREADS),
    "OPENBLAS_NUM_THREADS": str(THREADS),
    "MKL_NUM_THREADS": str(THREADS),
    "NUMEXPR_NUM_THREADS": str(THREADS),
    "NUMBA_NUM_THREADS": str(THREADS),
}


def _configure_legacy() -> None:
    """Point the proven worker machinery at the current public releases."""
    values = {
        "ROOT": ROOT,
        "BENCH": BENCH,
        "DARKOFIT_VERSION": DARKOFIT_VERSION,
        "DARKOFIT_COMMIT": DARKOFIT_COMMIT,
        "DARKOFIT_TAG": DARKOFIT_TAG,
        "CHIMERABOOST_VERSION": CHIMERABOOST_VERSION,
        "CHIMERABOOST_COMMIT": CHIMERABOOST_COMMIT,
        "CHIMERABOOST_TAG": CHIMERABOOST_TAG,
        "CHIMERABOOST_RELEASE_PUBLISHED_AT": CHIMERABOOST_RELEASE_PUBLISHED_AT,
        "CHIMERABOOST_RELEASE_REPOSITORY": CHIMERABOOST_RELEASE_REPOSITORY,
        "TABARENA_COMMIT": TABARENA_COMMIT,
        "TABARENA_TREE": TABARENA_TREE,
        "TABARENA_VERSION": TABARENA_VERSION,
        "THREADS": THREADS,
        "BOOTSTRAP_DRAWS": BOOTSTRAP_DRAWS,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        "WORKER_PREFIX": WORKER_PREFIX,
        "TASKS": TASKS,
        "TASK_SPLIT_COUNTS": TASK_SPLIT_COUNTS,
        "COORDINATE_PAIRS": COORDINATE_PAIRS,
        "DARKO_DEFAULT": DARKO_DEFAULT,
        "DARKO_ACCURACY": DARKO_ACCURACY,
        "DARKO_ENSEMBLE": DARKO_ENSEMBLE,
        "CHIMERA_DEFAULT": CHIMERA_DEFAULT,
        "CHIMERA_ACCURACY": CHIMERA_ACCURACY,
        "CHIMERA_ENSEMBLE": CHIMERA_ENSEMBLE,
        "ARM_SPECS": ARM_SPECS,
        "BASE_ORDER": BASE_ORDER,
        "EXPECTED_COORDINATES": EXPECTED_COORDINATES,
        "EXPECTED_WORKERS": EXPECTED_WORKERS,
        "WORKER_ENVIRONMENT": WORKER_ENVIRONMENT,
    }
    for name, value in values.items():
        setattr(legacy, name, value)


_configure_legacy()

sha256 = legacy.sha256
_sha256_bytes = legacy._sha256_bytes
_stable_artifact = legacy._stable_artifact
_write_create_only_json = legacy._write_create_only_json
_read_json = legacy._read_json
_coordinate_seed = legacy._coordinate_seed


def expected_coordinates() -> list[tuple[str, int, int]]:
    return legacy.expected_coordinates()


def expected_ordered_grid() -> list[tuple[str, int, int, str]]:
    return legacy.expected_ordered_grid()


def position_audit() -> dict[str, list[int]]:
    return legacy.position_audit()


def ordered_grid_sha256() -> str:
    return legacy.ordered_grid_sha256()


def validate_product_sources(
    darkofit_source: Path,
    chimeraboost_source: Path,
    tabarena_source: Path,
) -> dict[str, dict[str, Any]]:
    return legacy.validate_product_sources(
        darkofit_source,
        chimeraboost_source,
        tabarena_source,
    )


def validate_latest_chimeraboost_release() -> dict[str, Any]:
    return legacy.validate_latest_chimeraboost_release()


def _harness_head() -> str:
    head = legacy._git(ROOT, "rev-parse", "HEAD")
    if legacy._git(ROOT, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("compute-ladder harness checkout is not clean")
    if legacy._git(ROOT, "rev-parse", "origin/main") != head:
        raise RuntimeError("compute-ladder harness commit is not published on main")
    return head


def _exclusive_machine_audit() -> dict[str, Any]:
    import psutil

    self_pid = os.getpid()
    own_chain = {self_pid}
    ancestor = psutil.Process(self_pid).parent()
    while ancestor is not None:
        own_chain.add(ancestor.pid)
        try:
            ancestor = ancestor.parent()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            break
    markers = (
        "run_v012_compute_ladder",
        "run_v011_compute_ladder",
        "run_v011_m2_broad_panel",
        "run_v011_ensemble_evidence",
        "run_m3",
        "run_tabarena",
        "run_gpboost",
    )
    conflicts: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess, ValueError):
            continue
        if pid not in own_chain and any(marker in command for marker in markers):
            conflicts.append({"pid": pid, "command": command})
    if conflicts:
        raise RuntimeError(f"another benchmark process is active: {conflicts}")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "conflicting_benchmark_processes": [],
        "ignored_launch_ancestor_pids": sorted(own_chain - {self_pid}),
        "load_average": [float(value) for value in os.getloadavg()],
    }


def _manifest(args: argparse.Namespace) -> dict[str, Any]:
    sources = validate_product_sources(
        args.darkofit_source,
        args.chimeraboost_source,
        args.tabarena_source,
    )
    return {
        "schema_version": 1,
        "kind": "v012_compute_ladder_manifest",
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(PROTOCOL_PATH.relative_to(ROOT)),
            "sha256": sha256(PROTOCOL_PATH),
        },
        "runner": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "analyzer": {
            "path": str(ANALYZER_PATH.relative_to(ROOT)),
            "sha256": sha256(ANALYZER_PATH),
        },
        "harness_head": _harness_head(),
        "darkofit_source": sources["darkofit"],
        "chimeraboost_source": sources["chimeraboost"],
        "tabarena_source": sources["tabarena"],
        "latest_chimeraboost_release": validate_latest_chimeraboost_release(),
        "hardware": legacy._hardware(),
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


def _worker_result(args: argparse.Namespace) -> dict[str, Any]:
    payload = legacy._worker_result(args)
    payload["kind"] = "v012_compute_ladder_worker"
    return payload


def _run_parent(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    legacy._validate_output_state(output_dir)
    manifest_path = output_dir / "manifest.json"
    parent_pid = os.getpid()
    worker_artifacts: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        manifest = _manifest(args)
        _write_create_only_json(manifest_path, manifest)
        grid = expected_ordered_grid()
        for worker_index, (_dataset, _repeat, _fold, arm) in enumerate(grid):
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
            expected = grid[worker_index]
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
            "kind": "v012_compute_ladder_raw",
            "run_id": RUN_ID,
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
            "kind": "v012_compute_ladder_terminal",
            "status": "complete",
            "run_id": RUN_ID,
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
                    "kind": "v012_compute_ladder_terminal",
                    "status": "failed",
                    "run_id": RUN_ID,
                    "completed_worker_count": len(worker_artifacts),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--darkofit-source", type=Path, default=DEFAULT_DARKOFIT_SOURCE
    )
    parser.add_argument(
        "--chimeraboost-source", type=Path, default=DEFAULT_CHIMERABOOST_SOURCE
    )
    parser.add_argument(
        "--tabarena-source", type=Path, default=DEFAULT_TABARENA_SOURCE
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--worker-index", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--arm", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-started-at", default=None, help=argparse.SUPPRESS
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
            parser.error("worker index is outside the release grid")
        if args.arm not in ARM_SPECS:
            parser.error("worker arm is invalid")
        if args.dry_run:
            parser.error("worker mode cannot be a dry run")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_index is not None:
        result = _worker_result(args)
        print(WORKER_PREFIX + json.dumps(result, allow_nan=False, sort_keys=True))
        return 0
    if args.dry_run:
        print(json.dumps(_manifest(args), allow_nan=False, indent=2, sort_keys=True))
        return 0
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
