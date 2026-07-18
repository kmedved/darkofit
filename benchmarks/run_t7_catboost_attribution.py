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
import stat
import subprocess
import sys
import tempfile
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


def _json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number: {value}")
    return number


def _json_int(value):
    number = int(value)
    if not -(2**63) <= number <= 2**63 - 1:
        raise ValueError(f"out-of-range JSON integer: {value}")
    return number


def _json_loads(encoded, context):
    try:
        if isinstance(encoded, (bytes, bytearray)):
            encoded = bytes(encoded).decode("utf-8")
        elif not isinstance(encoded, str):
            raise ValueError("JSON input must be UTF-8 bytes or text")
        return json.loads(
            encoded,
            object_pairs_hook=_json_object,
            parse_float=_json_float,
            parse_int=_json_int,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"invalid {context} JSON") from error


def _same_typed_value(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_typed_value(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_typed_value(a, b) for a, b in zip(left, right)
        )
    return left == right


def _reject_symlink_directory(path, message):
    absolute = Path(os.path.abspath(os.path.expanduser(path)))
    for component in (absolute, *absolute.parents):
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"{message}: {component}")


def _create_owned_directories(path, message):
    missing = []
    current = path
    while True:
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            missing.append(current)
            current = current.parent
            continue
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            raise RuntimeError(f"{message}: {current}")
        break
    owned = []
    try:
        for directory in reversed(missing):
            try:
                directory.mkdir()
            except FileExistsError:
                mode = directory.lstat().st_mode
                if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
                    raise RuntimeError(f"{message}: {directory}")
                continue
            identity = directory.lstat()
            if not stat.S_ISDIR(identity.st_mode):
                raise RuntimeError(f"{message}: {directory}")
            owned.append(
                (directory, (identity.st_dev, identity.st_ino))
            )
    except BaseException:
        _remove_owned_directories(owned)
        raise
    return owned


def _remove_owned_directories(owned):
    for directory, expected in reversed(owned):
        try:
            current = directory.lstat()
            if (
                stat.S_ISDIR(current.st_mode)
                and (current.st_dev, current.st_ino) == expected
            ):
                directory.rmdir()
        except OSError:
            pass


def _verify_published_identity(path, expected, message):
    try:
        current = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"{message}: {path}") from error
    if (
        not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino) != expected
    ):
        raise RuntimeError(f"{message}: {path}")


def _array_sha256(value):
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _peak_rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _rows(registry_bytes=None, c2_raw_bytes=None):
    if registry_bytes is None:
        registry_bytes = REGISTRY.read_bytes()
    if hashlib.sha256(registry_bytes).hexdigest() != EXPECTED_REGISTRY_SHA256:
        raise RuntimeError("T7 C2 registry changed")
    registry = _json_loads(registry_bytes, "T7 registry")
    rows = {
        int(row["task_id"]): row
        for row in registry["development_tasks"]
    }
    if len(rows) != 8:
        raise RuntimeError("T7 development task count changed")
    if c2_raw_bytes is None:
        c2_raw_bytes = C2_RAW.read_bytes()
    if hashlib.sha256(c2_raw_bytes).hexdigest() != EXPECTED_C2_RAW_SHA256:
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
    if type(coordinate_index) is not int or coordinate_index < 0:
        raise ValueError("T7 coordinate index must be a nonnegative integer")
    shift = coordinate_index % len(ARM_NAMES)
    return ARM_NAMES[shift:] + ARM_NAMES[:shift]


def _warmup():
    from catboost import CatBoostRegressor

    X = np.arange(320, dtype=np.float64).reshape(80, 4)
    y = np.square(X[:, 0]) + 0.5 * X[:, 1]
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
    registry, rows = _rows()
    coordinates = [
        (int(row["task_id"]), current_fold)
        for row in registry["development_tasks"]
        for current_fold in FOLDS
    ]
    if (
        type(task_id) is not int
        or type(fold) is not int
        or type(coordinate_index) is not int
        or coordinate_index < 0
        or coordinate_index >= len(coordinates)
        or coordinates[coordinate_index] != (task_id, fold)
    ):
        raise ValueError(
            "T7 worker coordinate does not match the frozen schedule"
        )
    import catboost
    from catboost import CatBoostRegressor

    if catboost.__version__ != "1.2.10":
        raise RuntimeError("T7 requires CatBoost 1.2.10")
    row = rows[task_id]
    task, X, y, categorical = c2._load_task(row)
    outer_train, outer_test = task.get_train_test_split_indices(
        repeat=0, fold=fold, sample=0
    )
    outer = c2._verify_outer_split(
        row, fold, outer_train, outer_test
    )
    fit_indices, validation_indices, inner = c2.development_split(
        outer_train, task_id=task_id, fold=fold
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


def _ordered_spool_records(records, coordinates):
    coordinate_order = {
        coordinate: index for index, coordinate in enumerate(coordinates)
    }
    return sorted(
        records,
        key=lambda row: coordinate_order[(row["task_id"], row["fold"])],
    )


def _load_spool(path, binding, task_id, fold):
    _reject_symlink_directory(
        path.parent, "refusing symlink T7 spool directory"
    )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"refusing invalid T7 spool record: {path}") from error
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise RuntimeError(f"refusing invalid T7 spool record: {path}")
        encoded = handle.read()
    payload = _json_loads(encoded, "T7 spool record")
    if not isinstance(payload, dict):
        raise RuntimeError(f"refusing invalid T7 spool record: {path}")
    expected = payload.pop("spool_sha256", None)
    if (
        set(payload)
        != {"binding", "task_id", "fold", "result_sha256", "result"}
        or expected != _json_sha256(payload)
    ):
        raise RuntimeError(f"T7 spool hash changed: {path}")
    result = payload.get("result")
    if (
        not _same_typed_value(payload.get("binding"), binding)
        or type(payload.get("task_id")) is not int
        or payload["task_id"] != task_id
        or type(payload.get("fold")) is not int
        or payload["fold"] != fold
        or not isinstance(result, dict)
        or type(result.get("task_id")) is not int
        or result["task_id"] != task_id
        or type(result.get("fold")) is not int
        or result["fold"] != fold
        or payload["result_sha256"] != _json_sha256(result)
    ):
        raise RuntimeError(f"T7 spool binding changed: {path}")
    return result, expected


def _unlink_if_owned(path, identity):
    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    if (
        stat.S_ISREG(current.st_mode)
        and (current.st_dev, current.st_ino) == identity
    ):
        path.unlink()


def _create_spool(path, binding, result, *, return_publish_state=False):
    if path.is_symlink():
        raise RuntimeError(f"refusing symlink T7 spool record: {path}")
    payload = {
        "binding": binding,
        "task_id": result["task_id"],
        "fold": result["fold"],
        "result_sha256": _json_sha256(result),
        "result": result,
    }
    payload["spool_sha256"] = _json_sha256(payload)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    message = "refusing symlink T7 spool directory"
    _reject_symlink_directory(path.parent, message)
    owned_directories = _create_owned_directories(path.parent, message)
    temporary = None
    published_identity = None
    try:
        _reject_symlink_directory(path.parent, message)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            identity = os.fstat(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            outcome = _load_spool(
                path,
                binding,
                result["task_id"],
                result["fold"],
            )
            published = False
        else:
            published_identity = (identity.st_dev, identity.st_ino)
            _verify_published_identity(
                path,
                published_identity,
                "T7 spool publish identity changed",
            )
            outcome = (result, payload["spool_sha256"])
            published = True
    except BaseException:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        if published_identity is not None:
            try:
                _unlink_if_owned(path, published_identity)
            except OSError:
                pass
        _remove_owned_directories(owned_directories)
        raise
    try:
        temporary.unlink(missing_ok=True)
    except BaseException:
        if published_identity is not None:
            try:
                _unlink_if_owned(path, published_identity)
            except OSError:
                pass
        _remove_owned_directories(owned_directories)
        raise
    return (*outcome, published) if return_publish_state else outcome


def _create_output(path, encoded):
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing existing output: {path}")
    message = "refusing symlink T7 output directory"
    _reject_symlink_directory(path.parent, message)
    owned_directories = _create_owned_directories(path.parent, message)
    temporary = None
    published_identity = None
    try:
        _reject_symlink_directory(path.parent, message)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            identity = os.fstat(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(f"refusing existing output: {path}") from error
        published_identity = (identity.st_dev, identity.st_ino)
        _verify_published_identity(
            path,
            published_identity,
            "T7 output publish identity changed",
        )
    except BaseException:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        if published_identity is not None:
            try:
                _unlink_if_owned(path, published_identity)
            except OSError:
                pass
        _remove_owned_directories(owned_directories)
        raise
    try:
        temporary.unlink(missing_ok=True)
    except BaseException:
        try:
            _unlink_if_owned(path, published_identity)
        except OSError:
            pass
        _remove_owned_directories(owned_directories)
        raise


def _validate_worker_identity(result, task_id, fold, coordinate_index):
    if (
        not isinstance(result, dict)
        or type(result.get("task_id")) is not int
        or result["task_id"] != task_id
        or type(result.get("fold")) is not int
        or result["fold"] != fold
        or type(result.get("coordinate_index")) is not int
        or result["coordinate_index"] != coordinate_index
        or not isinstance(result.get("arm_order"), list)
        or tuple(result["arm_order"]) != _arm_order(coordinate_index)
        or not isinstance(result.get("arms"), list)
        or len(result["arms"]) != len(ARM_NAMES)
        or any(
            not isinstance(arm, dict)
            or type(arm.get("position")) is not int
            or arm["position"] != position
            or arm.get("arm") != result["arm_order"][position]
            for position, arm in enumerate(result["arms"])
        )
    ):
        raise RuntimeError(f"T7 worker {task_id}/{fold} identity changed")


def _run_one(task_id, fold, coordinate_index, spool, binding):
    path = _spool_path(spool, task_id, fold)
    if path.exists() or path.is_symlink():
        result, digest = _load_spool(
            path, binding, task_id, fold
        )
        _validate_worker_identity(result, task_id, fold, coordinate_index)
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
    result = _json_loads(
        lines[0][len(WORKER_PREFIX) :], "T7 worker result"
    )
    _validate_worker_identity(result, task_id, fold, coordinate_index)
    result["worker_stderr"] = completed.stderr.strip() or None
    result["worker_stdout"] = (
        "\n".join(
            line
            for line in completed.stdout.splitlines()
            if not line.startswith(WORKER_PREFIX)
        ).strip()
        or None
    )
    result, digest, published = _create_spool(
        path, binding, result, return_publish_state=True
    )
    _validate_worker_identity(result, task_id, fold, coordinate_index)
    return result, digest, not published


def run_parent(args):
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing existing output: {args.output}")
    _reject_symlink_directory(
        args.output.parent, "refusing symlink T7 output directory"
    )
    _reject_symlink_directory(
        args.spool, "refusing symlink T7 spool directory"
    )
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
    spool_records = _ordered_spool_records(spool_records, coordinates)
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
    _create_output(
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
    args.output = Path(os.path.abspath(os.path.expanduser(args.output)))
    args.spool = Path(os.path.abspath(os.path.expanduser(args.spool)))
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
