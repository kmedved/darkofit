#!/usr/bin/env python3
"""Run the spent-development T7 CatBoost mechanism attribution."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import basketball_harness as basketball  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402
from benchmarks import run_native_ordinal_c2 as c2  # noqa: E402
from benchmarks.run_t5_composite_confirmation import (  # noqa: E402
    _catboost_frame,
)


ARMS = {
    "default": {},
    "ordered": {"boosting_type": "Ordered"},
    "plain": {"boosting_type": "Plain"},
    "border_128": {"border_count": 128},
    "leaf10_no_backtracking": {
        "leaf_estimation_iterations": 10,
        "leaf_estimation_backtracking": "No",
    },
    "leaf10_any_improvement": {
        "leaf_estimation_iterations": 10,
        "leaf_estimation_backtracking": "AnyImprovement",
    },
    "ctr_complexity_2": {"max_ctr_complexity": 2},
    "depth_4": {"depth": 4},
    "depth_8": {"depth": 8},
}
ARM_NAMES = tuple(ARMS)
FOLDS = (0, 1, 2)
THREADS_PER_WORKER = 6
CONCURRENT_WORKERS = 3
PREDICTION_CALLS = 20
RANDOM_STATE = 4
WORKER_PREFIX = "T7_CATBOOST_ATTRIBUTION_RESULT="
REGISTRY = ROOT / "benchmarks" / "native_ordinal_c2_registry.json"
C2_RAW = ROOT / "benchmarks" / "native_ordinal_c2_development_raw.json"
PROTOCOL = ROOT / "benchmarks" / "t7_catboost_attribution_protocol.md"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "t7_catboost_attribution_raw.json"
DEFAULT_SPOOL = ROOT / ".cache" / "t7-catboost-attribution-v1"
EXPECTED_REGISTRY_SHA256 = (
    "34343d5296698ad7ac728fbef40961f384ca61923e6524afa8a2c7eeda7080d3"
)
EXPECTED_C2_RAW_SHA256 = (
    "2599029d7f4c8f7464c26af27d0aadf8e8443f47f1a52b0f794b8dd912c10d8a"
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _array_sha256(value):
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _peak_rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _rows():
    if _sha256(REGISTRY) != EXPECTED_REGISTRY_SHA256:
        raise RuntimeError("T7 C2 registry changed")
    registry = c2._load_registry()
    rows = {
        int(row["task_id"]): row
        for row in registry["development_tasks"]
    }
    if len(rows) != 8:
        raise RuntimeError("T7 development task count changed")
    if _sha256(C2_RAW) != EXPECTED_C2_RAW_SHA256:
        raise RuntimeError("T7 immutable DarkoFit anchor changed")
    return registry, rows


def _source_state():
    state = creator.git_state(ROOT)
    remote_head = subprocess.run(
        ["git", "rev-parse", "origin/codex/product-offense"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state["remote_branch_head"] = remote_head
    if (
        not state["clean"]
        or state["branch"] != "codex/product-offense"
        or remote_head != state["head"]
    ):
        raise RuntimeError("T7 requires the clean pushed offense branch")
    return state


def _arm_order(coordinate_index):
    shift = int(coordinate_index) % len(ARM_NAMES)
    return ARM_NAMES[shift:] + ARM_NAMES[:shift]


def _warmup():
    from catboost import CatBoostRegressor

    X = np.arange(320, dtype=np.float64).reshape(80, 4)
    y = X[:, 0] - X[:, 1]
    started = time.perf_counter_ns()
    model = CatBoostRegressor(
        iterations=2,
        random_seed=RANDOM_STATE,
        thread_count=THREADS_PER_WORKER,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X, y)
    model.predict(X[:8])
    return (time.perf_counter_ns() - started) / 1e9


def _score(model, X, y):
    prediction = np.asarray(model.predict(X), dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    if prediction.shape != target.shape or not np.all(np.isfinite(prediction)):
        raise RuntimeError("T7 CatBoost prediction is invalid")
    rmse = float(np.sqrt(np.mean(np.square(target - prediction))))
    if not math.isfinite(rmse) or rmse <= 0:
        raise RuntimeError("T7 CatBoost RMSE is invalid")
    return {
        "rows": int(len(prediction)),
        "rmse": rmse,
        "prediction_sha256": _array_sha256(prediction),
    }


def _timed_predict(model, X, expected):
    durations = []
    last = None
    for _ in range(PREDICTION_CALLS):
        started = time.perf_counter_ns()
        last = model.predict(X)
        durations.append((time.perf_counter_ns() - started) / 1e9)
    if _array_sha256(last) != expected:
        raise RuntimeError("T7 repeated prediction changed")
    return {
        "calls": PREDICTION_CALLS,
        "median_seconds": float(np.median(durations)),
        "total_seconds": float(sum(durations)),
    }


def _resolved_params(model):
    params = model.get_all_params()
    keys = (
        "boosting_type",
        "border_count",
        "leaf_estimation_iterations",
        "leaf_estimation_backtracking",
        "max_ctr_complexity",
        "depth",
        "grow_policy",
        "learning_rate",
        "iterations",
        "random_seed",
    )
    return {key: params.get(key) for key in keys}


def run_worker(task_id, fold, coordinate_index):
    import catboost
    from catboost import CatBoostRegressor

    if catboost.__version__ != "1.2.10":
        raise RuntimeError("T7 requires CatBoost 1.2.10")
    _registry, rows = _rows()
    row = rows[int(task_id)]
    task, X, y, categorical = c2._load_task(row)
    outer_train, outer_test = task.get_train_test_split_indices(
        repeat=0, fold=int(fold), sample=0
    )
    outer = c2._verify_outer_split(
        row, int(fold), outer_train, outer_test
    )
    fit_indices, validation_indices, inner = c2.development_split(
        outer_train, task_id=int(task_id), fold=int(fold)
    )
    X_fit = _catboost_frame(X.iloc[fit_indices], categorical)
    X_validation = _catboost_frame(
        X.iloc[validation_indices], categorical
    )
    X_test = _catboost_frame(X.iloc[outer_test], categorical)
    y_fit = y.iloc[fit_indices]
    y_validation = y.iloc[validation_indices]
    y_test = y.iloc[outer_test]
    warmup_seconds = _warmup()
    arm_results = []
    for position, arm in enumerate(_arm_order(coordinate_index)):
        model = CatBoostRegressor(
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            thread_count=THREADS_PER_WORKER,
            verbose=False,
            allow_writing_files=False,
            **ARMS[arm],
        )
        started = time.perf_counter_ns()
        model.fit(X_fit, y_fit, cat_features=categorical or None)
        fit_seconds = (time.perf_counter_ns() - started) / 1e9
        validation = _score(model, X_validation, y_validation)
        test = _score(model, X_test, y_test)
        timing = _timed_predict(
            model, X_test, test["prediction_sha256"]
        )
        arm_results.append(
            {
                "arm": arm,
                "position": position,
                "overrides": ARMS[arm],
                "fit_seconds": float(fit_seconds),
                "validation": validation,
                "test": test,
                "prediction_timing": timing,
                "tree_count": int(model.tree_count_),
                "resolved_params": _resolved_params(model),
            }
        )
        del model
        gc.collect()
    result = {
        "task_id": int(task_id),
        "dataset_id": int(row["dataset_id"]),
        "dataset_name": row["dataset_name"],
        "lineage_cluster": row["lineage_cluster"],
        "fold": int(fold),
        "coordinate_index": int(coordinate_index),
        "n_features": int(X.shape[1]),
        "categorical_count": len(categorical),
        "outer_split": outer,
        "inner_split": inner,
        "arm_order": list(_arm_order(coordinate_index)),
        "arms": arm_results,
        "warmup_seconds": float(warmup_seconds),
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    result["behavior_sha256"] = _json_sha256(
        {
            "task_id": result["task_id"],
            "fold": result["fold"],
            "arms": [
                {
                    "arm": arm["arm"],
                    "validation": arm["validation"],
                    "test": arm["test"],
                    "resolved_params": arm["resolved_params"],
                }
                for arm in arm_results
            ],
        }
    )
    return result


def _environment():
    environment = basketball.worker_environment(THREADS_PER_WORKER)
    environment.update(
        {
            "DARKOFIT_WARMUP": "0",
            "PYTHONHASHSEED": "0",
            "OPENML_CACHE_DIRECTORY": str(Path.home() / ".cache" / "openml"),
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT), environment.get("PYTHONPATH", "")]
            ),
        }
    )
    return environment


def _binding(source):
    return {
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "protocol_sha256": _sha256(PROTOCOL),
        "registry_sha256": EXPECTED_REGISTRY_SHA256,
        "c2_raw_sha256": EXPECTED_C2_RAW_SHA256,
        "source_head": source["head"],
        "catboost_version": "1.2.10",
        "arms": list(ARM_NAMES),
        "folds": list(FOLDS),
    }


def _spool_path(directory, task_id, fold):
    return directory / f"task-{task_id}--fold-{fold}.json"


def _load_spool(path, binding, task_id, fold):
    payload = json.loads(path.read_text())
    expected = payload.pop("spool_sha256", None)
    if expected != _json_sha256(payload):
        raise RuntimeError(f"T7 spool hash changed: {path}")
    if (
        payload["binding"] != binding
        or int(payload["task_id"]) != int(task_id)
        or int(payload["fold"]) != int(fold)
        or payload["result_sha256"] != _json_sha256(payload["result"])
    ):
        raise RuntimeError(f"T7 spool binding changed: {path}")
    return payload["result"], expected


def _create_spool(path, binding, result):
    payload = {
        "binding": binding,
        "task_id": result["task_id"],
        "fold": result["fold"],
        "result_sha256": _json_sha256(result),
        "result": result,
    }
    payload["spool_sha256"] = _json_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return result, payload["spool_sha256"]


def _run_one(task_id, fold, coordinate_index, spool, binding):
    path = _spool_path(spool, task_id, fold)
    if path.exists():
        result, digest = _load_spool(
            path, binding, task_id, fold
        )
        return result, digest, True
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-task",
        str(task_id),
        "--worker-fold",
        str(fold),
        "--coordinate-index",
        str(coordinate_index),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"T7 worker {task_id}/{fold} failed\n{completed.stderr}"
        )
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if len(lines) != 1:
        raise RuntimeError(f"T7 worker {task_id}/{fold} protocol failed")
    result = json.loads(lines[0][len(WORKER_PREFIX) :])
    result["worker_stderr"] = completed.stderr.strip() or None
    result["worker_stdout"] = (
        "\n".join(
            line
            for line in completed.stdout.splitlines()
            if not line.startswith(WORKER_PREFIX)
        ).strip()
        or None
    )
    result, digest = _create_spool(path, binding, result)
    return result, digest, False


def run_parent(args):
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    source = _source_state()
    registry, rows = _rows()
    binding = _binding(source)
    coordinates = [
        (int(row["task_id"]), fold)
        for row in registry["development_tasks"]
        for fold in FOLDS
    ]
    results = []
    spool_records = []
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        futures = {
            executor.submit(
                _run_one,
                task_id,
                fold,
                index,
                args.spool,
                binding,
            ): (task_id, fold)
            for index, (task_id, fold) in enumerate(coordinates)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            task_id, fold = futures[future]
            result, digest, resumed = future.result()
            results.append(result)
            spool_records.append(
                {
                    "task_id": task_id,
                    "fold": fold,
                    "sha256": digest,
                    "resumed": resumed,
                }
            )
            print(
                f"T7 {completed}/{len(coordinates)}: "
                f"{task_id}/{fold} "
                f"({'resumed' if resumed else 'fresh'})",
                flush=True,
            )
    if _source_state() != source:
        raise RuntimeError("T7 source changed during execution")
    results.sort(key=lambda row: int(row["coordinate_index"]))
    artifact = {
        "schema_version": 1,
        "name": "darkofit_t7_catboost_attribution_raw_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "runtime": {
            "python": sys.version,
            "machine": creator._machine_details(),
            "dependencies": creator._dependency_versions(),
        },
        "protocol": binding,
        "task_count": len(rows),
        "coordinate_count": len(coordinates),
        "fit_count": len(coordinates) * len(ARM_NAMES),
        "spool_records": spool_records,
        "results": results,
        "development_data_only": True,
        "confirmation_outcomes_inspected": False,
        "lockbox_data_used": False,
        "default_change_authorized": False,
    }
    artifact["raw_sha256"] = _json_sha256(artifact)
    creator._atomic_write_bytes(
        args.output,
        (
            json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode(),
    )
    return artifact


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spool", type=Path, default=DEFAULT_SPOOL)
    parser.add_argument("--worker-task", type=int)
    parser.add_argument("--worker-fold", type=int)
    parser.add_argument("--coordinate-index", type=int)
    args = parser.parse_args(argv)
    args.output = args.output.expanduser().absolute()
    args.spool = args.spool.expanduser().absolute()
    worker = (
        args.worker_task,
        args.worker_fold,
        args.coordinate_index,
    )
    if any(value is not None for value in worker) != all(
        value is not None for value in worker
    ):
        parser.error("all T7 worker arguments must be supplied together")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.worker_task is not None:
        result = run_worker(
            args.worker_task,
            args.worker_fold,
            args.coordinate_index,
        )
        print(
            WORKER_PREFIX
            + json.dumps(result, sort_keys=True, allow_nan=False),
            flush=True,
        )
        return 0
    run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
