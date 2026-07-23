#!/usr/bin/env python3
"""Run the automatic-depth paired ship-check on the sealed CTR23 tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path[:1]:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_t7b_automatic_depth_development_v1 as development
from benchmarks import run_t7b_automatic_depth_fresh_tier_d as helpers


SHIP_CHECK_ID = "t7b-automatic-depth-ctr23-ship-check-v1-20260723"
PARTITION = ROOT / "benchmarks" / "ctr23_partition.json"
SUITE = ROOT / "benchmarks" / "ctr23_suite_snapshot.json"
ANALYZER = ROOT / "benchmarks" / "analyze_t7b_automatic_depth_ctr23_ship_check_v1.py"
FOLDS = (0, 1, 2)
THREADS = 14


def build_manifest() -> dict[str, Any]:
    partition = helpers._load_json(PARTITION)
    suite = helpers._load_json(SUITE)
    task_ids = [int(value) for value in partition["lockbox_task_ids"]]
    by_id = {
        int(row["openml_task_id"]): row
        for row in suite["ctr23_tasks"]
    }
    if len(task_ids) != 9 or any(task_id not in by_id for task_id in task_ids):
        raise RuntimeError("CTR23 lockbox membership changed")
    tasks = [
        {
            "task_id": task_id,
            "dataset_id": int(by_id[task_id]["openml_dataset_id"]),
            "dataset_version": int(by_id[task_id]["openml_dataset_version"]),
            "dataset_name": str(by_id[task_id]["dataset_name"]),
            "target_name": str(by_id[task_id]["target_name"]),
            "coordinates": sorted(
                (
                    coordinate
                    for coordinate in by_id[task_id]["official_splits"]["coordinates"]
                    if int(coordinate["repeat"]) == 0
                    and int(coordinate["fold"]) in FOLDS
                    and int(coordinate["sample"]) == 0
                ),
                key=lambda coordinate: int(coordinate["fold"]),
            ),
        }
        for task_id in task_ids
    ]
    if any(
        [int(coordinate["fold"]) for coordinate in task["coordinates"]]
        != list(FOLDS)
        for task in tasks
    ):
        raise RuntimeError("CTR23 ship-check coordinates changed")
    return {
        "schema_version": 1,
        "ship_check_id": SHIP_CHECK_ID,
        "status": "ready",
        "kind": "holdout_ship_check",
        "holdout": "CTR23 lockbox",
        "partition_sha256": helpers.file_sha256(PARTITION),
        "suite_sha256": helpers.file_sha256(SUITE),
        "task_count": len(tasks),
        "tasks": tasks,
        "notes": [
            "CTR23 is relabeled observed release-validation after this run.",
            "The candidate is unchanged from development.",
            "No candidate tuning is permitted from these outcomes.",
        ],
    }


def _array_sha256(values, dtype="<i8") -> str:
    array = np.asarray(values, dtype=dtype)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load_task(task_spec: Mapping[str, Any]):
    import openml

    task = openml.tasks.get_task(int(task_spec["task_id"]), download_splits=True)
    dataset = task.get_dataset()
    if (
        int(dataset.dataset_id) != int(task_spec["dataset_id"])
        or int(dataset.version) != int(task_spec["dataset_version"])
        or str(dataset.name) != str(task_spec["dataset_name"])
        or str(task.target_name) != str(task_spec["target_name"])
    ):
        raise RuntimeError("CTR23 task binding changed")
    X, target, _categorical, _names = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )
    X, categorical = helpers._normalize_features(X)
    y = helpers._target_attestation(target, expected_rows=len(X))
    dimensions = tuple(int(value) for value in task.get_split_dimensions())
    if dimensions[1] < len(FOLDS) or dimensions[2] != 1:
        raise RuntimeError("CTR23 official split dimensions changed")
    return task, dataset, X, y, categorical


def run_worker(
    task_spec: Mapping[str, Any],
    *,
    fold: int,
    arm: str,
    source: Path,
) -> dict[str, Any]:
    import numba

    source = source.expanduser().resolve()
    expected = (
        development.CONTROL_HEAD
        if arm == "control"
        else development.CANDIDATE_HEAD
    )
    state = helpers.source_state(source)
    if not state["clean"] or state["head"] != expected:
        raise RuntimeError(f"{arm} source state changed")
    sys.path.insert(0, str(source))
    from darkofit import DarkoRegressor
    import darkofit

    if Path(darkofit.__file__).resolve().parents[1] != source:
        raise RuntimeError("worker imported DarkoFit from the wrong source")
    helpers._warmup(DarkoRegressor)
    task, dataset, X, y, categorical = _load_task(task_spec)
    train, test = task.get_train_test_split_indices(
        repeat=0,
        fold=int(fold),
        sample=0,
    )
    train = np.asarray(train, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    if train.size == 0 or test.size == 0 or np.intersect1d(train, test).size:
        raise RuntimeError("CTR23 official split is invalid")
    expected_coordinate = next(
        coordinate
        for coordinate in task_spec["coordinates"]
        if int(coordinate["fold"]) == int(fold)
    )
    observed_coordinate = {
        "train_size": int(train.size),
        "test_size": int(test.size),
        "train_index_sha256": _array_sha256(train),
        "test_index_sha256": _array_sha256(test),
    }
    if any(
        observed_coordinate[name] != expected_coordinate[name]
        for name in observed_coordinate
    ):
        raise RuntimeError("CTR23 official split differs from the committed snapshot")
    model = DarkoRegressor(**helpers._model_params())
    ambient = int(numba.get_num_threads())
    with helpers._PeakRSS() as rss:
        started = time.perf_counter()
        model.fit(
            X.iloc[train].reset_index(drop=True),
            y[train],
            cat_features=categorical or None,
        )
        fit_seconds = time.perf_counter() - started
        prediction_started = time.perf_counter()
        prediction = np.asarray(
            model.predict(X.iloc[test].reset_index(drop=True)),
            dtype=np.float64,
        )
        predict_seconds = time.perf_counter() - prediction_started
        with tempfile.TemporaryDirectory(prefix="t7b-ctr23-worker-") as temp:
            archive = Path(temp) / "model.npz"
            model.save_model(archive)
            restored = DarkoRegressor.load_model(archive)
            check_rows = min(256, len(test))
            exact = np.array_equal(
                restored.predict(X.iloc[test[:check_rows]].reset_index(drop=True)),
                prediction[:check_rows],
            )
    fitted_depth = int(model.model_.depth)
    policy = (
        None
        if arm == "control"
        else model.model_.auto_params_["auto_structure"]["candidates"]["depth"]
    )
    base = {
        "fitted_depth": fitted_depth,
        "automatic_depth_policy": policy,
        "safe_npz_exact": bool(exact),
        "ambient_thread_restored": int(numba.get_num_threads()) == ambient,
        "rmse": helpers._rmse(y[test], prediction, None),
        "fit_seconds": fit_seconds,
        "predict_seconds_repeats": [predict_seconds] * 3,
    }
    integrity = development._row_integrity(base, arm)
    return {
        "schema_version": 1,
        "ship_check_id": SHIP_CHECK_ID,
        "status": "ok" if integrity else "integrity_failed",
        "arm": arm,
        "source": state,
        "task_id": int(task_spec["task_id"]),
        "dataset_id": int(dataset.dataset_id),
        "dataset_name": str(dataset.name),
        "fold": int(fold),
        "train_rows": int(train.size),
        "test_rows": int(test.size),
        "input_features": int(X.shape[1]),
        "train_index_sha256": observed_coordinate["train_index_sha256"],
        "test_index_sha256": observed_coordinate["test_index_sha256"],
        "rmse": base["rmse"],
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "peak_process_tree_rss_bytes": rss.peak,
        "fitted_depth": fitted_depth,
        "automatic_depth_policy": policy,
        "safe_npz_exact": bool(exact),
        "ambient_thread_restored": base["ambient_thread_restored"],
        "integrity_passes": integrity,
    }


def _worker_command(spec: Path, fold: int, arm: str, source: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--task-spec",
        str(spec),
        "--fold",
        str(fold),
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
        raise ValueError("ship-check outputs must be outside the source tree")
    return {
        "launch": Path(str(prefix) + "_launch.json"),
        "raw": Path(str(prefix) + "_raw.json"),
        "result": Path(str(prefix) + "_result.json"),
    }


def exclusive_machine_audit() -> dict[str, Any]:
    import psutil

    audit = development.exclusive_machine_audit()
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
        raise RuntimeError(f"another CTR23 ship-check is active: {conflicts}")
    return audit


def execute(
    *,
    manifest_path: Path,
    control: Path,
    candidate: Path,
    prefix: Path,
) -> dict[str, Any]:
    manifest = helpers._load_json(manifest_path)
    if (
        manifest.get("ship_check_id") != SHIP_CHECK_ID
        or manifest.get("status") != "ready"
        or manifest.get("task_count") != 9
    ):
        raise RuntimeError("CTR23 ship-check manifest is invalid")
    paths = output_paths(prefix)
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions:
        raise RuntimeError(f"ship-check output collision: {collisions}")
    states = {
        "harness": helpers.source_state(ROOT),
        "control": helpers.source_state(control),
        "candidate": helpers.source_state(candidate),
    }
    if not states["harness"]["clean"]:
        raise RuntimeError("ship-check harness must be clean")
    if (
        not states["control"]["clean"]
        or states["control"]["head"] != development.CONTROL_HEAD
        or not states["candidate"]["clean"]
        or states["candidate"]["head"] != development.CANDIDATE_HEAD
    ):
        raise RuntimeError("ship-check source pins changed")
    audit = exclusive_machine_audit()
    environment = helpers._environment()
    if environment["logical_cpu_count"] != THREADS:
        raise RuntimeError("ship-check requires 14 logical CPUs")
    launch = {
        "schema_version": 1,
        "ship_check_id": SHIP_CHECK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "holdout_ship_check",
        "sources": states,
        "source_hashes": {
            "manifest": helpers.file_sha256(manifest_path),
            "runner": helpers.file_sha256(Path(__file__)),
            "analyzer": helpers.file_sha256(ANALYZER),
        },
        "environment": environment,
        "exclusive_machine_audit": audit,
        "planned_rows": 54,
    }
    helpers._write_create_only(paths["launch"], launch)
    rows = []
    with tempfile.TemporaryDirectory(prefix="t7b-ctr23-ship-check-") as temp:
        temp_path = Path(temp)
        caches = {
            "control": temp_path / "numba-control",
            "candidate": temp_path / "numba-candidate",
        }
        for cache in caches.values():
            cache.mkdir()
        for task_index, task_spec in enumerate(manifest["tasks"]):
            spec_path = temp_path / f"task-{task_index:02d}.json"
            spec_path.write_bytes(helpers.canonical_json_bytes(task_spec))
            for fold in FOLDS:
                arms = (
                    ("control", "candidate")
                    if (task_index + fold) % 2 == 0
                    else ("candidate", "control")
                )
                for arm in arms:
                    source = control if arm == "control" else candidate
                    completed = subprocess.run(
                        _worker_command(spec_path, fold, arm, source),
                        cwd=ROOT,
                        env=helpers._worker_env(source, caches[arm]),
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    if completed.returncode:
                        raise RuntimeError(
                            f"CTR23 worker failed for task {task_spec['task_id']}/"
                            f"{fold}/{arm}: {completed.stderr[-4000:]}"
                        )
                    lines = [
                        line for line in completed.stdout.splitlines() if line.strip()
                    ]
                    if not lines:
                        raise RuntimeError("CTR23 worker returned no row")
                    row = json.loads(lines[-1])
                    if row.get("status") != "ok":
                        raise RuntimeError(f"CTR23 worker integrity failed: {row}")
                    rows.append(row)
    if len(rows) != 54:
        raise RuntimeError("CTR23 ship-check row census changed")
    raw = {
        "schema_version": 1,
        "ship_check_id": SHIP_CHECK_ID,
        "complete": True,
        "holdout": "CTR23 lockbox",
        "environment": environment,
        "launch_sha256": helpers.file_sha256(paths["launch"]),
        "rows": rows,
    }
    helpers._write_create_only(paths["raw"], raw)
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
        raise RuntimeError("CTR23 analyzer failed: " + completed.stderr[-4000:])
    return helpers._load_json(paths["result"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--output", type=Path, required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--task-spec", type=Path, required=True)
    worker.add_argument("--fold", type=int, choices=FOLDS, required=True)
    worker.add_argument("--arm", choices=("control", "candidate"), required=True)
    worker.add_argument("--source", type=Path, required=True)
    run = sub.add_parser("execute")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--control", type=Path, required=True)
    run.add_argument("--candidate", type=Path, required=True)
    run.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "manifest":
        if args.output.exists() or args.output.is_symlink():
            raise RuntimeError(f"refusing existing output: {args.output}")
        artifact = build_manifest()
        helpers._write_create_only(args.output, artifact)
        print(json.dumps({"output": str(args.output), "tasks": 9}))
        return 0
    if args.command == "worker":
        row = run_worker(
            helpers._load_json(args.task_spec),
            fold=args.fold,
            arm=args.arm,
            source=args.source,
        )
        print(json.dumps(row, sort_keys=True, allow_nan=False))
        return 0
    result = execute(
        manifest_path=args.manifest,
        control=args.control,
        candidate=args.candidate,
        prefix=args.output_prefix,
    )
    print(
        json.dumps(
            {
                "quality_ratio": result["quality"]["task_geomean_ratio"],
                "bootstrap_upper_ratio": result["quality"]["bootstrap_upper_ratio"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
