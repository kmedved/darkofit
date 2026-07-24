#!/usr/bin/env python3
"""Characterize activation-gated B3-v2 on the spent v1 timing grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from . import run_b3_parallel_ensemble_v1 as v1
except ImportError:  # direct script execution
    import run_b3_parallel_ensemble_v1 as v1


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ID = "b3-activation-gated-parallel-members-v2-20260723"
SOURCE_FILES = (
    "darkofit/sklearn_api.py",
    "tests/test_b3_parallel_ensemble_candidate.py",
)
EXPECTED_ROUTES = {
    "general_friedman_numeric": "sequential_fallback",
    "general_categorical_reg": "sequential_fallback",
    "general_numeric_binary": "process_parallel",
    "general_categorical_multiclass": "process_parallel",
}
MINIMUM_WORK = 80_000_000
ARMS = v1.ARMS
CASES = v1.CASES
MODES = v1.MODES
BLOCKS = v1.BLOCKS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _write_create_only(path: Path, value: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _geomean(values) -> float:
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise RuntimeError("B3-v2 ratio is not finite and positive")
    return float(math.exp(np.mean(np.log(values))))


def _record(row: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    records = [item for item in row["records"] if item["mode"] == mode]
    if len(records) != 1:
        raise RuntimeError("B3-v2 worker mode census differs")
    return records[0]


def _logical_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(record["model"])
    for name in (
        "fitted_thread_counts",
        "prediction_thread_counts",
        "schedule",
        "sequential",
    ):
        identity.pop(name, None)
    return identity


def _observed_route(record: Mapping[str, Any]) -> str:
    model = record["model"]
    schedule = model["schedule"]
    if (
        model["sequential"] is True
        and schedule is None
        and model["fitted_thread_counts"] == [v1.THREADS] * 8
        and model["prediction_thread_counts"] == [v1.THREADS] * 8
    ):
        return "sequential_fallback"
    if (
        model["sequential"] is False
        and model["fitted_thread_counts"] == [v1.MEMBER_THREADS] * 8
        and model["prediction_thread_counts"] == [v1.MEMBER_THREADS] * 8
        and isinstance(schedule, Mapping)
        and schedule.get("contract") == v1.CONTRACT_ID
        and schedule.get("mode") == "private_process_workers"
        and schedule.get("workers") == v1.WORKERS
        and schedule.get("member_threads") == v1.MEMBER_THREADS
        and schedule.get("total_thread_budget") == v1.THREADS
        and schedule.get("maximum_model_threads") == v1.THREADS
        and schedule.get("result_order") == "member_index"
    ):
        return "process_parallel"
    return "invalid"


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = len(CASES) * BLOCKS * len(ARMS)
    indexed = {
        (row["case_id"], row["block"], row["arm"]): row for row in rows
    }
    if len(rows) != expected or len(indexed) != expected:
        raise RuntimeError("B3-v2 worker census differs")

    exact = []
    route_checks = []
    resource_checks = []
    memory_checks = []
    ratios = {mode: {} for mode in MODES}
    for case_id in CASES:
        expected_route = EXPECTED_ROUTES[case_id]
        for block in range(BLOCKS):
            control = indexed[(case_id, block, "sequential_1x14")]
            candidate = indexed[(case_id, block, "parallel_7x2")]
            if control["fingerprints"] != candidate["fingerprints"]:
                raise RuntimeError("B3-v2 paired fingerprints differ")
            for mode in MODES:
                control_record = _record(control, mode)
                candidate_record = _record(candidate, mode)
                exact.append(
                    control_record["prediction_sha256"]
                    == candidate_record["prediction_sha256"]
                    and control_record["probability_sha256"]
                    == candidate_record["probability_sha256"]
                    and _logical_identity(control_record)
                    == _logical_identity(candidate_record)
                )
                route_checks.append(
                    _observed_route(control_record) == "sequential_fallback"
                )
                route_checks.append(
                    _observed_route(candidate_record) == expected_route
                )
                ratios[mode][(case_id, block)] = (
                    candidate_record["fit_seconds"]
                    / control_record["fit_seconds"]
                )
                candidate_peak = candidate_record["fit_rss"]["peak_bytes"]
                control_peak = control_record["fit_rss"]["peak_bytes"]
                resource_checks.append(
                    not control_record["fit_rss"].get("errors")
                    and not candidate_record["fit_rss"].get("errors")
                )
                ratio = candidate_peak / control_peak
                delta = candidate_peak - control_peak
                memory_checks.append({
                    "case_id": case_id,
                    "block": block,
                    "mode": mode,
                    "candidate_peak_bytes": candidate_peak,
                    "control_peak_bytes": control_peak,
                    "ratio": ratio,
                    "delta_bytes": delta,
                    "passes": (
                        candidate_peak <= v1.ABSOLUTE_RSS_CEILING
                        and not (
                            ratio > v1.RSS_RATIO_ALLOWANCE
                            and delta > v1.RSS_DELTA_ALLOWANCE
                        )
                    ),
                })

    speed = {}
    for mode, mode_values in ratios.items():
        case_medians = {
            case_id: float(np.median([
                mode_values[(case_id, block)] for block in range(BLOCKS)
            ]))
            for case_id in CASES
        }
        engaged = [
            value
            for case_id, value in case_medians.items()
            if EXPECTED_ROUTES[case_id] == "process_parallel"
        ]
        fallback = [
            value
            for case_id, value in case_medians.items()
            if EXPECTED_ROUTES[case_id] == "sequential_fallback"
        ]
        speed[mode] = {
            "case_median_ratios": case_medians,
            "all_case_geomean_ratio": _geomean(case_medians.values()),
            "engaged_geomean_ratio": _geomean(engaged),
            "engaged_worst_case_ratio": max(engaged),
            "fallback_geomean_ratio": _geomean(fallback),
            "fallback_worst_case_ratio": max(fallback),
        }

    checks = {
        "behavior_exact": all(exact),
        "routes_match_frozen_work_rule": all(route_checks),
        "resource_sampling_clean": all(resource_checks),
        "memory_bounded": all(item["passes"] for item in memory_checks),
        "engaged_cold_direction_stable": (
            speed["cold_executor"]["engaged_worst_case_ratio"] <= 1.0
        ),
        "engaged_steady_direction_stable": (
            speed["steady_executor"]["engaged_worst_case_ratio"] <= 1.0
        ),
        "fallback_cold_not_materially_slower": (
            speed["cold_executor"]["fallback_worst_case_ratio"] <= 1.05
        ),
        "fallback_steady_not_materially_slower": (
            speed["steady_executor"]["fallback_worst_case_ratio"] <= 1.05
        ),
    }
    return {
        "checks": checks,
        "routes": dict(EXPECTED_ROUTES),
        "speed": speed,
        "memory": {
            "absolute_ceiling_bytes": v1.ABSOLUTE_RSS_CEILING,
            "ratio_allowance": v1.RSS_RATIO_ALLOWANCE,
            "delta_allowance_bytes": v1.RSS_DELTA_ALLOWANCE,
            "maximum_candidate_peak_bytes": max(
                item["candidate_peak_bytes"] for item in memory_checks
            ),
            "checks": memory_checks,
        },
        "disposition": (
            "ready_to_productize" if all(checks.values()) else "needs_revision"
        ),
    }


def run(source: Path, output_prefix: Path) -> Path:
    source = source.expanduser().resolve()
    output_prefix = output_prefix.expanduser().resolve()
    status = v1.git(
        source, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if status:
        raise RuntimeError("B3-v2 source must be clean")
    state = {
        "path": str(source),
        "head": v1.git(source, "rev-parse", "HEAD"),
        "tree": v1.git(source, "rev-parse", "HEAD^{tree}"),
        "clean": True,
    }
    if source != ROOT:
        raise RuntimeError("B3-v2 runs one clean source for both paired arms")
    if os.cpu_count() != v1.THREADS:
        raise RuntimeError("B3-v2 characterization requires the 14-CPU host")
    audit = v1.exclusive_machine_audit()
    paths = {
        "launch": Path(str(output_prefix) + "_launch.json"),
        "raw": Path(str(output_prefix) + "_raw.json"),
        "result": Path(str(output_prefix) + "_result.json"),
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("B3-v2 output path already exists")
    launch = {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": state,
        "source_files": {
            name: _sha256(source / name) for name in SOURCE_FILES
        },
        "runner": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": _sha256(Path(__file__)),
        },
        "minimum_work": MINIMUM_WORK,
        "expected_routes": dict(EXPECTED_ROUTES),
        "cases": list(CASES),
        "blocks": BLOCKS,
        "modes": list(MODES),
        "exclusive_machine": audit,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
        },
    }
    _write_create_only(paths["launch"], launch)
    rows = []
    for case_id in CASES:
        for block in range(BLOCKS):
            for arm in v1.order_for(case_id, block):
                row = v1._run_worker(source, case_id, arm)
                row["block"] = block
                rows.append(row)
                print(f"ok {case_id:32s} block={block} arm={arm}", flush=True)
    _write_create_only(paths["raw"], {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "rows": rows,
    })
    result = {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": analyze(rows),
        "launch_sha256": _sha256(paths["launch"]),
        "raw_sha256": _sha256(paths["raw"]),
    }
    _write_create_only(paths["result"], result)
    return paths["result"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.source, args.output_prefix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
