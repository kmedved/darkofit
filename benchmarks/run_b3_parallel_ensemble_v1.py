#!/usr/bin/env python3
"""Run the one permitted private B3 v1 timing/resource inspection."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from . import run_ensemble_v3_characterization as characterization
    from . import run_m3b_ensemble_v3 as m3b
    from .run_b3_parallel_ensemble_v1_invariants import (
        CANDIDATE_FILES,
        CANDIDATE_BASE_HEAD,
        CANDIDATE_HEAD,
        CONTROL_HEAD,
        git,
        sha256,
        source_state,
    )
except ImportError:  # direct script execution
    import run_ensemble_v3_characterization as characterization
    import run_m3b_ensemble_v3 as m3b
    from run_b3_parallel_ensemble_v1_invariants import (
        CANDIDATE_FILES,
        CANDIDATE_BASE_HEAD,
        CANDIDATE_HEAD,
        CONTROL_HEAD,
        git,
        sha256,
        source_state,
    )


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "benchmarks/b3_parallel_ensemble_v1_contract.md"
INVARIANT_RUNNER_PATH = (
    ROOT / "benchmarks/run_b3_parallel_ensemble_v1_invariants.py"
)
CONTRACT_ID = "b3-parallel-ensemble-members-v1-20260723"
INSPECTION_INDEX = 1
THREADS = 14
MEMBER_THREADS = 2
WORKERS = 7
ITERATIONS = 600
PATIENCE = 30
RANDOM_STATE = 4
VALIDATION_FRACTION = 0.15
BLOCKS = 3
ARMS = ("sequential_1x14", "parallel_7x2")
MODES = ("cold_executor", "steady_executor")
CASES = (
    "general_friedman_numeric",
    "general_categorical_reg",
    "general_numeric_binary",
    "general_categorical_multiclass",
)
WORKER_PREFIX = "B3_PARALLEL_ENSEMBLE_RESULT="
GIB = 1024**3
ABSOLUTE_RSS_CEILING = 6 * GIB
RSS_RATIO_ALLOWANCE = 5.0
RSS_DELTA_ALLOWANCE = 2 * GIB
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def write_create_only(path: Path, payload: Any) -> None:
    path = path.expanduser().resolve()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def output_paths(prefix: Path) -> dict[str, Path]:
    prefix = prefix.expanduser().resolve()
    return {
        "launch": Path(str(prefix) + "_launch_manifest.json"),
        "raw": Path(str(prefix) + "_raw.json"),
        "result": Path(str(prefix) + "_result.json"),
        "terminal": Path(str(prefix) + "_terminal_attestation.json"),
    }


def _case_spec(case_id: str) -> Mapping[str, Any]:
    for spec in m3b.case_specs():
        if spec["case_id"] == case_id:
            return spec
    raise ValueError(f"unknown B3 case: {case_id}")


def order_for(case_id: str, block: int) -> tuple[str, ...]:
    offset = (CASES.index(case_id) + int(block)) % 2
    return ARMS[offset:] + ARMS[:offset]


def _worker_environment(source: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith(("NUMBA_", "OMP_", "KMP_", "MKL_", "OPENBLAS_", "VECLIB_", "NUMEXPR_")):
            environment.pop(key)
    for key in THREAD_ENV_KEYS:
        environment[key] = str(THREADS)
    environment.update({
        "DARKOFIT_WARMUP": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": os.pathsep.join((str(source.resolve()), str(ROOT))),
        "NUMBA_CACHE_DIR": "/private/tmp/darkofit-b3-v1-numba-cache",
        "JOBLIB_TEMP_FOLDER": "/private/tmp/darkofit-b3-v1-joblib",
        "LOKY_MAX_CPU_COUNT": str(THREADS),
        "NUMBA_DISABLE_JIT": "0",
        "NUMBA_NUM_THREADS": str(THREADS),
        "NUMBA_THREADING_LAYER": "default",
        "OMP_DYNAMIC": "FALSE",
        "OMP_THREAD_LIMIT": str(THREADS),
        "MKL_DYNAMIC": "FALSE",
    })
    return environment


def _runtime_record() -> dict[str, Any]:
    import numba

    return {
        "numba_thread_ceiling": int(numba.config.NUMBA_NUM_THREADS),
        "numba_current_threads": int(numba.get_num_threads()),
        "numba_threading_layer": numba.threading_layer(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (*THREAD_ENV_KEYS, "NUMBA_NUM_THREADS", "OMP_THREAD_LIMIT")
        },
    }


def _activate_source(source: Path) -> None:
    value = str(source.expanduser().resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _prediction_hash(value: Any) -> str:
    return characterization.array_sha256(np.asarray(value))


def _model_identity(model) -> dict[str, Any]:
    metadata = model.ensemble_metadata_
    return {
        "member_seeds": list(metadata["member_seeds"]),
        "sampled_indices_sha256": [
            record["sampled_indices_sha256"] for record in metadata["members"]
        ],
        "oob_indices_sha256": [
            record["oob_indices_sha256"] for record in metadata["members"]
        ],
        "best_iterations": [
            int(record["best_iteration"]) for record in metadata["members"]
        ],
        "fitted_thread_counts": [
            int(record["fitted_thread_count"]) for record in metadata["members"]
        ],
        "prediction_thread_counts": [
            int(record.get("prediction_thread_count", record["fitted_thread_count"]))
            for record in metadata["members"]
        ],
        "schedule": metadata.get("private_b3_schedule"),
        "sequential": metadata["sequential"],
    }


def _build_model(task: str):
    from darkofit import DarkoClassifier, DarkoRegressor

    estimator = DarkoRegressor if task == "regression" else DarkoClassifier
    return estimator(
        iterations=ITERATIONS,
        early_stopping_rounds=PATIENCE,
        early_stopping=True,
        use_best_model=True,
        refit=False,
        validation_fraction=VALIDATION_FRACTION,
        validation_strategy="random",
        random_state=RANDOM_STATE,
        thread_count=THREADS,
        diagnostic_warnings="never",
        ensemble_shared_preprocessing=True,
        n_ensembles=8,
        ensemble_mode="v3",
    )


def _fit_once(arm: str, spec: Mapping[str, Any], data: Mapping[str, Any]) -> dict:
    from darkofit import DarkoClassifier, DarkoRegressor

    model = _build_model(str(spec["task"]))
    fit_kwargs = {
        "cat_features": data["cat_features"],
        "groups": data["groups_fit"],
        "sample_weight": data["w_fit"],
    }
    gc.collect()
    with characterization.ProcessTreeRSSSampler() as rss:
        start = time.perf_counter_ns()
        if arm == "sequential_1x14":
            model.fit(data["X_fit"], data["y_fit"], **fit_kwargs)
        elif arm == "parallel_7x2":
            from darkofit.sklearn_api import (
                _fit_public_ensemble_v3_parallel_candidate,
            )

            _fit_public_ensemble_v3_parallel_candidate(
                model,
                data["X_fit"],
                data["y_fit"],
                total_thread_budget=THREADS,
                **fit_kwargs,
            )
        else:
            raise ValueError(f"unknown B3 arm: {arm}")
        fit_seconds = (time.perf_counter_ns() - start) / 1e9
    if spec["task"] == "regression":
        start = time.perf_counter_ns()
        prediction = model.predict(data["X_test"])
        predict_seconds = (time.perf_counter_ns() - start) / 1e9
        probability = None
    else:
        start = time.perf_counter_ns()
        probability = model.predict_proba(data["X_test"])
        predict_seconds = (time.perf_counter_ns() - start) / 1e9
        prediction = model.predict(data["X_test"])
    with tempfile.TemporaryDirectory(prefix="darkofit-b3-v1-") as directory:
        archive = Path(directory) / "model.npz"
        model.save_model(archive)
        archive_bytes = archive.stat().st_size
        estimator = DarkoRegressor if spec["task"] == "regression" else DarkoClassifier
        loaded = estimator.load_model(archive)
        loaded_prediction = loaded.predict(data["X_test"])
        if not np.array_equal(np.asarray(prediction), np.asarray(loaded_prediction)):
            raise RuntimeError("B3 archive changed predictions")
        if probability is not None and not np.array_equal(
            np.asarray(probability), np.asarray(loaded.predict_proba(data["X_test"]))
        ):
            raise RuntimeError("B3 archive changed probabilities")
    return {
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "fit_rss": {
            "scope": "parent_plus_recursive_children",
            "start_bytes": int(rss.start_bytes),
            "peak_bytes": int(rss.peak_bytes),
            "peak_delta_bytes": int(max(0, rss.peak_bytes - rss.start_bytes)),
            "end_bytes": int(rss.end_bytes),
            "samples": int(rss.samples),
            "errors": list(rss.errors),
        },
        "archive_bytes": int(archive_bytes),
        "prediction_sha256": _prediction_hash(prediction),
        "probability_sha256": (
            None if probability is None else _prediction_hash(probability)
        ),
        "model": _model_identity(model),
    }


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.expanduser().resolve()
    _activate_source(source)
    before = _runtime_record()
    if before["numba_thread_ceiling"] != THREADS or before["numba_current_threads"] != THREADS:
        raise RuntimeError(f"B3 worker thread precondition failed: {before}")
    spec = _case_spec(args.case_id)
    data = m3b.build_case(spec)
    fingerprints = m3b.case_fingerprints(spec, data)
    records = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for mode in MODES:
            record = _fit_once(args.arm, spec, data)
            record["mode"] = mode
            records.append(record)
    after = _runtime_record()
    if after["numba_current_threads"] != THREADS:
        raise RuntimeError(f"B3 worker leaked its ambient thread mask: {after}")
    return {
        "case_id": args.case_id,
        "task": spec["task"],
        "arm": args.arm,
        "fingerprints": fingerprints,
        "fit_rows": len(data["y_fit"]),
        "test_rows": len(data["y_test"]),
        "records": records,
        "runtime_before": before,
        "runtime_after": after,
        "warnings": [
            {"category": item.category.__name__, "message": str(item.message)}
            for item in caught
        ],
    }


def _run_worker(source: Path, case_id: str, arm: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--source",
        str(source),
        "--case-id",
        case_id,
        "--arm",
        arm,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_worker_environment(source),
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith(WORKER_PREFIX)]
    if completed.returncode or len(lines) != 1:
        raise RuntimeError(
            f"B3 worker failed for {case_id}/{arm}\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    payload = json.loads(lines[0][len(WORKER_PREFIX):])
    payload["worker_stdout"] = "\n".join(
        line for line in completed.stdout.splitlines() if not line.startswith(WORKER_PREFIX)
    ).strip() or None
    payload["worker_stderr"] = completed.stderr.strip() or None
    return payload


def _geomean(values) -> float:
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise RuntimeError("B3 ratio is not finite and positive")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = len(CASES) * BLOCKS * len(ARMS)
    if len(rows) != expected:
        raise RuntimeError(f"B3 row count differs: {len(rows)} != {expected}")
    indexed = {(row["case_id"], row["block"], row["arm"]): row for row in rows}
    if len(indexed) != expected:
        raise RuntimeError("B3 row identities are duplicated")
    exactness = []
    execution_integrity = []
    memory_checks = []
    mode_ratios = {mode: {} for mode in MODES}
    for case_id in CASES:
        for block in range(BLOCKS):
            control = indexed[(case_id, block, "sequential_1x14")]
            candidate = indexed[(case_id, block, "parallel_7x2")]
            if control["fingerprints"] != candidate["fingerprints"]:
                raise RuntimeError("B3 paired fingerprints differ")
            for row in (control, candidate):
                cold = next(item for item in row["records"] if item["mode"] == "cold_executor")
                steady = next(item for item in row["records"] if item["mode"] == "steady_executor")
                execution_integrity.append(
                    cold["prediction_sha256"] == steady["prediction_sha256"]
                    and cold["probability_sha256"] == steady["probability_sha256"]
                    and cold["model"] == steady["model"]
                    and not cold["fit_rss"].get("errors")
                    and not steady["fit_rss"].get("errors")
                )
            for mode in MODES:
                control_record = next(row for row in control["records"] if row["mode"] == mode)
                candidate_record = next(row for row in candidate["records"] if row["mode"] == mode)
                same_output = (
                    control_record["prediction_sha256"] == candidate_record["prediction_sha256"]
                    and control_record["probability_sha256"] == candidate_record["probability_sha256"]
                )
                control_identity = dict(control_record["model"])
                candidate_identity = dict(candidate_record["model"])
                for identity in (control_identity, candidate_identity):
                    identity.pop("fitted_thread_counts", None)
                    identity.pop("prediction_thread_counts", None)
                    identity.pop("schedule", None)
                    identity.pop("sequential", None)
                same_identity = control_identity == candidate_identity
                exactness.append(same_output and same_identity)
                control_schedule_ok = (
                    control_record["model"]["schedule"] is None
                    and control_record["model"]["sequential"] is True
                    and control_record["model"]["fitted_thread_counts"] == [14] * 8
                    and control_record["model"]["prediction_thread_counts"] == [14] * 8
                )
                candidate_schedule = candidate_record["model"]["schedule"]
                candidate_schedule_ok = (
                    candidate_record["model"]["sequential"] is False
                    and candidate_record["model"]["fitted_thread_counts"] == [2] * 8
                    and candidate_record["model"]["prediction_thread_counts"] == [2] * 8
                    and isinstance(candidate_schedule, Mapping)
                    and candidate_schedule.get("contract") == CONTRACT_ID
                    and candidate_schedule.get("workers") == WORKERS
                    and candidate_schedule.get("member_threads") == MEMBER_THREADS
                    and candidate_schedule.get("total_thread_budget") == THREADS
                    and candidate_schedule.get("maximum_model_threads") == THREADS
                )
                execution_integrity.append(control_schedule_ok and candidate_schedule_ok)
                ratio = candidate_record["fit_seconds"] / control_record["fit_seconds"]
                mode_ratios[mode][(case_id, block)] = ratio
                candidate_peak = candidate_record["fit_rss"]["peak_bytes"]
                control_peak = control_record["fit_rss"]["peak_bytes"]
                memory_checks.append({
                    "case_id": case_id,
                    "block": block,
                    "mode": mode,
                    "candidate_peak_bytes": candidate_peak,
                    "control_peak_bytes": control_peak,
                    "ratio": candidate_peak / control_peak,
                    "delta_bytes": candidate_peak - control_peak,
                    "passes": (
                        candidate_peak <= ABSOLUTE_RSS_CEILING
                        and not (
                            candidate_peak / control_peak > RSS_RATIO_ALLOWANCE
                            and candidate_peak - control_peak > RSS_DELTA_ALLOWANCE
                        )
                    ),
                })
    speed = {}
    for mode, ratios in mode_ratios.items():
        case_medians = {
            case_id: float(np.median([ratios[(case_id, block)] for block in range(BLOCKS)]))
            for case_id in CASES
        }
        aggregate = _geomean(ratios.values())
        loo = {
            omitted: _geomean(
                ratio for (case_id, _), ratio in ratios.items() if case_id != omitted
            )
            for omitted in CASES
        }
        speed[mode] = {
            "equal_case_geometric_mean_ratio": aggregate,
            "case_median_ratios": case_medians,
            "leave_one_case_out_ratios": loo,
            "worst_case_median_ratio": max(case_medians.values()),
            "worst_leave_one_out_ratio": max(loo.values()),
            "passes": (
                aggregate <= 1.0
                and max(case_medians.values()) <= 1.0
                and max(loo.values()) <= 1.0
            ),
        }
    gates = {
        "behavior_exact": all(exactness),
        "execution_integrity": all(execution_integrity),
        "cold_speed_stable": speed["cold_executor"]["passes"],
        "steady_speed_stable": speed["steady_executor"]["passes"],
        "hybrid_rss": all(item["passes"] for item in memory_checks),
    }
    return {
        "gates": gates,
        "speed": speed,
        "memory": {
            "absolute_ceiling_bytes": ABSOLUTE_RSS_CEILING,
            "ratio_allowance": RSS_RATIO_ALLOWANCE,
            "delta_allowance_bytes": RSS_DELTA_ALLOWANCE,
            "checks": memory_checks,
            "maximum_candidate_peak_bytes": max(item["candidate_peak_bytes"] for item in memory_checks),
        },
        "disposition": "advance" if all(gates.values()) else "kill",
    }


def exclusive_machine_audit() -> dict[str, Any]:
    import psutil

    own = {os.getpid()}
    parent = psutil.Process().parent()
    while parent is not None:
        own.add(parent.pid)
        try:
            parent = parent.parent()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            break
    markers = ("run_b3_parallel_ensemble_v1", "run_m6_quality_successor", "run_tabarena", "run_v011")
    conflicts = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess, ValueError):
            continue
        if pid not in own and any(marker in command for marker in markers):
            conflicts.append({"pid": pid, "command": command})
    if conflicts:
        raise RuntimeError(f"another benchmark process is active: {conflicts}")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "conflicting_benchmark_processes": [],
        "load_average": list(os.getloadavg()),
    }


def validate_sources(control: Path, candidate: Path) -> dict[str, Any]:
    states = {
        "harness": source_state(ROOT),
        "control": source_state(control, CONTROL_HEAD),
        "candidate": source_state(candidate, CANDIDATE_HEAD),
    }
    changed = set(
        git(
            candidate,
            "diff",
            "--name-only",
            f"{CANDIDATE_BASE_HEAD}..{CANDIDATE_HEAD}",
        ).splitlines()
    )
    if changed != CANDIDATE_FILES:
        raise RuntimeError(f"candidate file allowlist drifted: {sorted(changed)}")
    return states


def validate_invariants(path: Path, sources: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if (
        payload.get("contract_id") != CONTRACT_ID
        or payload.get("passed") is not True
        or payload.get("sources", {}).get("control", {}).get("head") != CONTROL_HEAD
        or payload.get("sources", {}).get("candidate", {}).get("head") != CANDIDATE_HEAD
        or payload.get("sources", {}).get("harness", {}).get("head") != sources["harness"]["head"]
        or payload.get("contract", {}).get("sha256") != sha256(CONTRACT_PATH)
    ):
        raise RuntimeError("B3 invariant artifact is invalid")
    return file_record(path)


def run(args: argparse.Namespace) -> Path:
    paths = output_paths(args.output_prefix)
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise FileExistsError("B3 outputs are create-only")
    control = args.control.expanduser().resolve()
    candidate = args.candidate.expanduser().resolve()
    for path in paths.values():
        if any(path.is_relative_to(root) for root in (ROOT, control, candidate)):
            raise ValueError("B3 outputs must be outside all source checkouts")
    sources = validate_sources(control, candidate)
    invariants = validate_invariants(args.invariants.expanduser().resolve(), sources)
    import psutil

    physical_memory = int(psutil.virtual_memory().total)
    if physical_memory != 24 * GIB:
        raise RuntimeError(
            f"B3 contract requires the 24 GiB machine class; got {physical_memory}"
        )
    launch = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "inspection_index": INSPECTION_INDEX,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inspection_spent_on_manifest_creation": True,
        "rerun_authorized": False,
        "sources": sources,
        "invariants": invariants,
        "bindings": {
            "contract": file_record(CONTRACT_PATH),
            "runner": file_record(Path(__file__).resolve()),
            "invariant_runner": file_record(INVARIANT_RUNNER_PATH),
        },
        "exclusive_machine": exclusive_machine_audit(),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "physical_memory_bytes": physical_memory,
            "physical_cores": THREADS,
        },
        "execution": {
            "cases": list(CASES),
            "blocks": BLOCKS,
            "arms": list(ARMS),
            "modes": list(MODES),
            "iterations": ITERATIONS,
            "patience": PATIENCE,
            "random_state": RANDOM_STATE,
            "validation_fraction": VALIDATION_FRACTION,
            "control_topology": {"workers": 1, "member_threads": 14},
            "candidate_topology": {"workers": WORKERS, "member_threads": MEMBER_THREADS},
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    write_create_only(paths["launch"], launch)
    rows = []
    try:
        for case_id in CASES:
            for block in range(BLOCKS):
                for arm in order_for(case_id, block):
                    source = control if arm == "sequential_1x14" else candidate
                    row = _run_worker(source, case_id, arm)
                    row["block"] = block
                    rows.append(row)
                    print(f"ok {case_id:32s} block={block} arm={arm}", flush=True)
        write_create_only(paths["raw"], {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "inspection_index": INSPECTION_INDEX,
            "rows": rows,
        })
        analysis = analyze(rows)
        result = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "inspection_index": INSPECTION_INDEX,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "analysis": analysis,
            "candidate_public_or_default_eligible": False,
            "candidate_merge_eligible": False,
            "next_eligibility": (
                "eligible_for_public_b3_contract_design"
                if analysis["disposition"] == "advance"
                else "closed"
            ),
        }
        write_create_only(paths["result"], result)
        terminal = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "inspection_index": INSPECTION_INDEX,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "disposition": analysis["disposition"],
            "rerun_authorized": False,
            "launch_sha256": sha256(paths["launch"]),
            "raw_sha256": sha256(paths["raw"]),
            "result_sha256": sha256(paths["result"]),
            "candidate_merge_eligible": False,
        }
        write_create_only(paths["terminal"], terminal)
        return paths["result"]
    except BaseException as exc:
        if not paths["raw"].exists():
            write_create_only(paths["raw"], {
                "schema_version": 1,
                "contract_id": CONTRACT_ID,
                "inspection_index": INSPECTION_INDEX,
                "rows": rows,
                "execution_failed": True,
            })
        if not paths["terminal"].exists():
            write_create_only(paths["terminal"], {
                "schema_version": 1,
                "contract_id": CONTRACT_ID,
                "inspection_index": INSPECTION_INDEX,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "disposition": "execution_failed_closed",
                "rerun_authorized": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "launch_sha256": sha256(paths["launch"]),
                "raw_sha256": sha256(paths["raw"]),
                "candidate_merge_eligible": False,
            })
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--case-id", choices=CASES)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--invariants", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.source is None or args.case_id is None or args.arm is None:
            parser.error("worker mode requires source, case-id, and arm")
        print(WORKER_PREFIX + json.dumps(run_worker(args), separators=(",", ":")))
        return 0
    if any(value is None for value in (args.control, args.candidate, args.invariants, args.output_prefix)):
        parser.error("run mode requires control, candidate, invariants, and output-prefix")
    print(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
