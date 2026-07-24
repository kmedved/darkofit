#!/usr/bin/env python3
"""Run the automatic linear-selector v3 CTR23 ship-check."""

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

from benchmarks import run_t7b_automatic_depth_ctr23_ship_check_v1 as ctr23
from benchmarks import run_t7b_automatic_depth_fresh_tier_d as helpers


SHIP_CHECK_ID = "automatic-linear-selector-v3-ctr23-ship-check-20260723"
ANALYZER = (
    ROOT
    / "benchmarks"
    / "analyze_automatic_linear_selector_v3_ctr23_ship_check.py"
)
PROTOCOL = (
    ROOT / "benchmarks" / "automatic_linear_selector_v3_ctr23_ship_check.md"
)
FOLDS = (0, 1, 2)
THREADS = 14
ARMS = {"control": False, "automatic": "auto"}


def build_manifest() -> dict[str, Any]:
    manifest = ctr23.build_manifest()
    manifest.update({
        "ship_check_id": SHIP_CHECK_ID,
        "holdout": "CTR23 observed release-validation",
        "notes": [
            "CTR23 was previously opened by the automatic-depth ship-check.",
            "The selector candidate is unchanged after Protein development.",
            "No candidate tuning is permitted from these outcomes.",
        ],
    })
    return manifest


def _selector_integrity(arm: str, model, restored) -> bool:
    selector = getattr(model, "automatic_linear_selector_", None)
    restored_selector = getattr(restored, "automatic_linear_selector_", None)
    if arm == "control":
        return bool(
            selector is None
            and restored_selector is None
            and not bool(getattr(model.model_, "linear_leaves", False))
        )
    if (
        not isinstance(selector, Mapping)
        or selector != restored_selector
        or selector.get("version") != 2
        or selector.get("minimum_gain_z") != 2.0
        or selector.get("requested") != "auto"
    ):
        return False
    selected = selector.get("resolved_linear_leaves")
    if not isinstance(selected, bool):
        return False
    return bool(
        selector.get("final_booster_linear_leaves") is selected
        and bool(getattr(model.model_, "linear_leaves", False)) is selected
    )


def run_worker(
    task_spec: Mapping[str, Any],
    *,
    fold: int,
    arm: str,
    source: Path,
    expected_head: str,
) -> dict[str, Any]:
    import numba

    source = source.expanduser().resolve()
    state = helpers.source_state(source)
    if not state["clean"] or state["head"] != expected_head:
        raise RuntimeError("selector CTR23 source state changed")
    sys.path.insert(0, str(source))
    from darkofit import DarkoRegressor
    import darkofit

    if Path(darkofit.__file__).resolve().parents[1] != source:
        raise RuntimeError("worker imported DarkoFit from the wrong source")
    helpers._warmup(DarkoRegressor)
    task, dataset, X, y, categorical = ctr23._load_task(task_spec)
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
        "train_index_sha256": ctr23._array_sha256(train),
        "test_index_sha256": ctr23._array_sha256(test),
    }
    if any(
        observed_coordinate[name] != expected_coordinate[name]
        for name in observed_coordinate
    ):
        raise RuntimeError("CTR23 official split differs from committed snapshot")

    params = helpers._model_params()
    params["linear_leaves"] = ARMS[arm]
    model = DarkoRegressor(**params)
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
        with tempfile.TemporaryDirectory(prefix="selector-v3-ctr23-worker-") as temp:
            archive = Path(temp) / "model.npz"
            model.save_model(archive)
            restored = DarkoRegressor.load_model(archive)
            check_rows = min(256, len(test))
            archive_exact = np.array_equal(
                restored.predict(X.iloc[test[:check_rows]].reset_index(drop=True)),
                prediction[:check_rows],
            )

    selector = getattr(model, "automatic_linear_selector_", None)
    prediction_sha256 = hashlib.sha256(
        np.ascontiguousarray(prediction, dtype="<f8").tobytes()
    ).hexdigest()
    integrity = bool(
        archive_exact
        and int(numba.get_num_threads()) == ambient
        and _selector_integrity(arm, model, restored)
        and np.isfinite(prediction).all()
        and prediction.shape == (len(test),)
    )
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
        "rmse": helpers._rmse(y[test], prediction, None),
        "prediction_sha256": prediction_sha256,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "peak_process_tree_rss_bytes": rss.peak,
        "automatic_linear_selector": (
            None if selector is None else dict(selector)
        ),
        "final_linear_leaves": bool(
            getattr(model.model_, "linear_leaves", False)
        ),
        "safe_npz_exact": bool(archive_exact),
        "ambient_thread_restored": int(numba.get_num_threads()) == ambient,
        "integrity_passes": integrity,
    }


def _worker_command(
    spec: Path,
    fold: int,
    arm: str,
    source: Path,
    expected_head: str,
) -> list[str]:
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
        "--expected-head",
        expected_head,
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

    audit = ctr23.exclusive_machine_audit()
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
        raise RuntimeError(f"another selector CTR23 ship-check is active: {conflicts}")
    return audit


def execute(
    *,
    manifest_path: Path,
    source: Path,
    expected_head: str,
    prefix: Path,
) -> dict[str, Any]:
    manifest = helpers._load_json(manifest_path)
    if (
        manifest.get("ship_check_id") != SHIP_CHECK_ID
        or manifest.get("status") != "ready"
        or manifest.get("task_count") != 9
    ):
        raise RuntimeError("selector CTR23 manifest is invalid")
    paths = output_paths(prefix)
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions:
        raise RuntimeError(f"selector CTR23 output collision: {collisions}")
    states = {
        "harness": helpers.source_state(ROOT),
        "candidate": helpers.source_state(source),
    }
    if (
        not states["harness"]["clean"]
        or states["harness"]["head"] != expected_head
        or states["candidate"] != states["harness"]
    ):
        raise RuntimeError("selector CTR23 source pin changed")
    audit = exclusive_machine_audit()
    environment = helpers._environment()
    if environment["logical_cpu_count"] != THREADS:
        raise RuntimeError("selector CTR23 ship-check requires 14 logical CPUs")
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
            "protocol": helpers.file_sha256(PROTOCOL),
        },
        "environment": environment,
        "exclusive_machine_audit": audit,
        "planned_rows": 54,
    }
    helpers._write_create_only(paths["launch"], launch)
    rows = []
    with tempfile.TemporaryDirectory(prefix="selector-v3-ctr23-") as temp:
        temp_path = Path(temp)
        caches = {
            arm: temp_path / f"numba-{arm}"
            for arm in ARMS
        }
        for cache in caches.values():
            cache.mkdir()
        for task_index, task_spec in enumerate(manifest["tasks"]):
            spec_path = temp_path / f"task-{task_index:02d}.json"
            spec_path.write_bytes(helpers.canonical_json_bytes(task_spec))
            for fold in FOLDS:
                arms = (
                    ("control", "automatic")
                    if (task_index + fold) % 2 == 0
                    else ("automatic", "control")
                )
                for arm in arms:
                    completed = subprocess.run(
                        _worker_command(
                            spec_path, fold, arm, source, expected_head
                        ),
                        cwd=ROOT,
                        env=helpers._worker_env(source, caches[arm]),
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    if completed.returncode:
                        raise RuntimeError(
                            f"selector CTR23 worker failed for "
                            f"{task_spec['task_id']}/{fold}/{arm}: "
                            f"{completed.stderr[-4000:]}"
                        )
                    lines = [
                        line
                        for line in completed.stdout.splitlines()
                        if line.strip()
                    ]
                    if not lines:
                        raise RuntimeError("selector CTR23 worker returned no row")
                    row = json.loads(lines[-1])
                    if row.get("status") != "ok":
                        raise RuntimeError(
                            f"selector CTR23 worker integrity failed: {row}"
                        )
                    rows.append(row)
                    print(
                        f"ok {len(rows)}/54 task={task_spec['task_id']} "
                        f"fold={fold} arm={arm}",
                        flush=True,
                    )
    if len(rows) != 54:
        raise RuntimeError("selector CTR23 ship-check row census changed")
    raw = {
        "schema_version": 1,
        "ship_check_id": SHIP_CHECK_ID,
        "complete": True,
        "holdout": "CTR23 observed release-validation",
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
        raise RuntimeError("selector CTR23 analyzer failed: " + completed.stderr[-4000:])
    return helpers._load_json(paths["result"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--output", type=Path, required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--task-spec", type=Path, required=True)
    worker.add_argument("--fold", type=int, choices=FOLDS, required=True)
    worker.add_argument("--arm", choices=tuple(ARMS), required=True)
    worker.add_argument("--source", type=Path, required=True)
    worker.add_argument("--expected-head", required=True)
    run = sub.add_parser("execute")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--expected-head", required=True)
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
            expected_head=args.expected_head,
        )
        print(json.dumps(row, sort_keys=True, allow_nan=False))
        return 0
    result = execute(
        manifest_path=args.manifest,
        source=args.source,
        expected_head=args.expected_head,
        prefix=args.output_prefix,
    )
    print(
        json.dumps(
            {
                "disposition": result["disposition"],
                "quality_ratio": result["quality"]["task_geomean_ratio"],
                "worst_task_ratio": result["quality"]["worst_task_ratio"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
