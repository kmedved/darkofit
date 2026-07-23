#!/usr/bin/env python3
"""Run the 32-lineage automatic-depth paired development benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path[:1]:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_t7b_automatic_depth_fresh_tier_d as legacy
from benchmarks import run_t7b_automatic_depth_fresh_tier_d_v3 as registry


BENCHMARK_ID = "t7b-automatic-depth-development-v1-20260723"
CONTROL_HEAD = legacy.CONTROL_HEAD
CANDIDATE_HEAD = legacy.CANDIDATE_HEAD
ANALYZER = ROOT / "benchmarks" / "analyze_t7b_automatic_depth_development_v1.py"
THREADS = 14


def _candidate_policy_is_consistent(row: Mapping[str, Any]) -> bool:
    depth = int(row["fitted_depth"])
    policy = row.get("automatic_depth_policy")
    expected_branch = {
        4: "low_density",
        6: "middle_density",
        8: "high_density",
    }.get(depth)
    return bool(
        isinstance(policy, Mapping)
        and expected_branch is not None
        and policy.get("branch") == expected_branch
        and policy.get("rule")
        == "scalar_rmse_catboost_n_eff_per_input_feature_4_6_8"
    )


def _row_integrity(row: Mapping[str, Any], arm: str) -> bool:
    policy_ok = (
        _candidate_policy_is_consistent(row)
        if arm == "candidate"
        else int(row["fitted_depth"]) == 6
        and row.get("automatic_depth_policy") is None
    )
    prediction_times = np.asarray(
        row.get("predict_seconds_repeats", ()), dtype=np.float64
    )
    return bool(
        policy_ok
        and row.get("safe_npz_exact") is True
        and row.get("ambient_thread_restored") is True
        and np.isfinite(float(row["rmse"]))
        and float(row["rmse"]) > 0.0
        and np.isfinite(float(row["fit_seconds"]))
        and float(row["fit_seconds"]) > 0.0
        and prediction_times.shape == (3,)
        and np.isfinite(prediction_times).all()
        and np.all(prediction_times > 0.0)
    )


def run_worker(
    lineage: Mapping[str, Any],
    *,
    coordinate: int,
    arm: str,
    source: Path,
) -> dict[str, Any]:
    """Run the historical worker, replacing only its wrong branch assertion."""
    row = legacy.run_worker(
        lineage,
        coordinate=coordinate,
        arm=arm,
        source=source,
    )
    integrity = _row_integrity(row, arm)
    return {
        **row,
        "contract_id": BENCHMARK_ID,
        "status": "ok" if integrity else "integrity_failed",
        "integrity_passes": integrity,
        "panel_branch": row["branch"],
        "resolved_depth": int(row["fitted_depth"]),
        "resolved_policy_branch": (
            None
            if row.get("automatic_depth_policy") is None
            else row["automatic_depth_policy"].get("branch")
        ),
    }


def build_preflight() -> dict[str, Any]:
    """Project the verified 32-lineage registry into a normal dev manifest."""
    historical = registry.build_preflight()
    return {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "status": "preflight_passed",
        "historical_registry_sha256": legacy.file_sha256(registry.ENUMERATION),
        "active_lineage_count": historical["active_lineage_count"],
        "active_lineages": historical["active_lineages"],
        "notes": [
            "This is a development benchmark, not a fresh confirmation.",
            "Registry branch labels describe the panel construction only.",
            "Candidate policy resolution is recorded from fitted metadata.",
        ],
    }


def _worker_command(
    lineage_path: Path,
    coordinate: int,
    arm: str,
    source: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--lineage",
        str(lineage_path),
        "--coordinate",
        str(coordinate),
        "--arm",
        arm,
        "--source",
        str(source),
    ]


def output_paths(prefix: Path) -> dict[str, Path]:
    prefix = prefix.expanduser().resolve()
    try:
        prefix.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("benchmark outputs must be outside the source tree")
    return {
        "launch": Path(str(prefix) + "_launch.json"),
        "raw": Path(str(prefix) + "_raw.json"),
        "result": Path(str(prefix) + "_result.json"),
    }


def exclusive_machine_audit() -> dict[str, Any]:
    """Run the standing audit and also exclude another copy of this runner."""
    import psutil

    audit = legacy._exclusive_machine_audit()
    own = {os.getpid()}
    parent = psutil.Process().parent()
    while parent is not None:
        own.add(parent.pid)
        try:
            parent = parent.parent()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            break
    marker = Path(__file__).name
    conflicts = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess, ValueError):
            continue
        if pid not in own and marker in command:
            conflicts.append({"pid": pid, "command": command})
    if conflicts:
        raise RuntimeError(f"another automatic-depth benchmark is active: {conflicts}")
    return audit


def execute(
    *,
    preflight_path: Path,
    control: Path,
    candidate: Path,
    prefix: Path,
) -> dict[str, Any]:
    preflight = legacy._load_json(preflight_path)
    if (
        preflight.get("benchmark_id") != BENCHMARK_ID
        or preflight.get("status") != "preflight_passed"
        or preflight.get("active_lineage_count") != 32
    ):
        raise RuntimeError("development preflight is invalid")
    paths = output_paths(prefix)
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions:
        raise RuntimeError(f"benchmark output collision: {collisions}")

    harness_state = legacy.source_state(ROOT)
    control_state = legacy.source_state(control)
    candidate_state = legacy.source_state(candidate)
    if not harness_state["clean"]:
        raise RuntimeError("benchmark harness source tree must be clean")
    if not control_state["clean"] or control_state["head"] != CONTROL_HEAD:
        raise RuntimeError("control source pin changed")
    if not candidate_state["clean"] or candidate_state["head"] != CANDIDATE_HEAD:
        raise RuntimeError("candidate source pin changed")
    audit = exclusive_machine_audit()
    environment = legacy._environment()
    if environment["logical_cpu_count"] != THREADS:
        raise RuntimeError(f"benchmark requires exactly {THREADS} logical CPUs")

    launch = {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "paired_development",
        "rerunnable_after_harness_or_environment_failure": True,
        "sources": {
            "harness": harness_state,
            "control": control_state,
            "candidate": candidate_state,
        },
        "source_hashes": {
            "runner": legacy.file_sha256(Path(__file__)),
            "analyzer": legacy.file_sha256(ANALYZER),
            "preflight": legacy.file_sha256(preflight_path),
        },
        "environment": environment,
        "exclusive_machine_audit": audit,
        "planned_arm_rows": 192,
        "output_paths": {key: str(path) for key, path in paths.items()},
    }
    legacy._write_create_only(paths["launch"], launch)

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="t7b-depth-development-") as temp:
        temp_path = Path(temp)
        caches = {
            "control": temp_path / "numba-control",
            "candidate": temp_path / "numba-candidate",
        }
        for cache in caches.values():
            cache.mkdir()
        for lineage_index, lineage in enumerate(preflight["active_lineages"]):
            lineage_path = temp_path / f"lineage-{lineage_index:02d}.json"
            lineage_path.write_bytes(legacy.canonical_json_bytes(lineage))
            for coordinate in (0, 1, 2):
                arms = (
                    ("control", "candidate")
                    if (lineage_index + coordinate) % 2 == 0
                    else ("candidate", "control")
                )
                for arm in arms:
                    source = control if arm == "control" else candidate
                    completed = subprocess.run(
                        _worker_command(lineage_path, coordinate, arm, source),
                        cwd=ROOT,
                        env=legacy._worker_env(source, caches[arm]),
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    if completed.returncode:
                        raise RuntimeError(
                            f"worker failed for {lineage['lineage_id']}/"
                            f"{coordinate}/{arm}: {completed.stderr[-4000:]}"
                        )
                    output_lines = [
                        line for line in completed.stdout.splitlines() if line.strip()
                    ]
                    if not output_lines:
                        raise RuntimeError("worker returned no JSON row")
                    row = json.loads(output_lines[-1])
                    if row.get("status") != "ok":
                        raise RuntimeError(f"worker integrity failed: {row}")
                    rows.append(row)

    if len(rows) != 192:
        raise RuntimeError("development row census changed")
    raw = {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "complete": True,
        "launch_sha256": legacy.file_sha256(paths["launch"]),
        "environment": environment,
        "rows": rows,
    }
    legacy._write_create_only(paths["raw"], raw)
    completed = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--raw",
            str(paths["raw"]),
            "--output",
            str(paths["result"]),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError("development analyzer failed: " + completed.stderr[-4000:])
    return legacy._load_json(paths["result"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--output", type=Path, required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--lineage", type=Path, required=True)
    worker.add_argument("--coordinate", type=int, choices=(0, 1, 2), required=True)
    worker.add_argument("--arm", choices=("control", "candidate"), required=True)
    worker.add_argument("--source", type=Path, required=True)
    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument("--preflight", type=Path, required=True)
    execute_parser.add_argument("--control", type=Path, required=True)
    execute_parser.add_argument("--candidate", type=Path, required=True)
    execute_parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "preflight":
        if args.output.exists() or args.output.is_symlink():
            raise RuntimeError(f"refusing existing output: {args.output}")
        artifact = build_preflight()
        legacy._write_create_only(args.output, artifact)
        print(json.dumps({"output": str(args.output), "lineages": 32}))
        return 0
    if args.command == "worker":
        row = run_worker(
            legacy._load_json(args.lineage),
            coordinate=args.coordinate,
            arm=args.arm,
            source=args.source,
        )
        print(json.dumps(row, sort_keys=True, allow_nan=False))
        return 0
    result = execute(
        preflight_path=args.preflight,
        control=args.control,
        candidate=args.candidate,
        prefix=args.output_prefix,
    )
    print(
        json.dumps(
            {
                "quality_ratio": result["quality"][
                    "equal_lineage_geomean_ratio"
                ],
                "bootstrap_upper_ratio": result["quality"][
                    "bootstrap_upper_ratio"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
