#!/usr/bin/env python3
"""Fresh-process runner for the frozen fused-lane dispatch campaign."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from . import fused_lane_dispatch_campaign as campaign
except ImportError:  # direct script execution
    import fused_lane_dispatch_campaign as campaign


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
CALIBRATION_PREFIX = "FUSED_LANE_CALIBRATION_RESULT="
VALIDATION_PREFIX = "FUSED_LANE_VALIDATION_RESULT="
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
WORKER_ENV_KEYS = (
    *THREAD_ENV_KEYS,
    "DARKOFIT_WARMUP",
    "NUMBA_CACHE_DIR",
    "NUMBA_DISABLE_JIT",
    "NUMBA_NUM_THREADS",
    "NUMBA_THREADING_LAYER",
    "OMP_DYNAMIC",
    "OMP_THREAD_LIMIT",
    "MKL_DYNAMIC",
    "PYTHONHASHSEED",
    "PYTHONPATH",
)


def _is_hex_digest(value: Any, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def git_state(repo: Path = ROOT) -> dict[str, str]:
    repo = repo.expanduser().resolve()
    return {
        "head": _git(repo, "rev-parse", "HEAD"),
        "status": _git(
            repo,
            "status", "--porcelain=v1", "--untracked-files=all"
        ),
    }


def _sysctl_value(name: str) -> str | None:
    if sys.platform != "darwin":
        return None
    completed = subprocess.run(
        ("sysctl", "-n", name),
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _positive_int(value: str | int | None) -> int | None:
    try:
        parsed = int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed is not None and parsed > 0 else None


def _memory_bytes() -> int | None:
    value = _positive_int(_sysctl_value("hw.memsize"))
    if value is not None:
        return value
    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            page_count = int(os.sysconf("SC_PHYS_PAGES"))
        except (OSError, TypeError, ValueError):
            return None
        value = page_size * page_count
        return value if value > 0 else None
    return None


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def runtime_fingerprint() -> dict[str, Any]:
    """Return the exact runtime/hardware identity bound by formal execution."""
    payload = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": _package_version("numpy"),
        "numba": _package_version("numba"),
        "llvmlite": _package_version("llvmlite"),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "hardware_model": _sysctl_value("hw.model"),
        "cpu_identifier": (
            _sysctl_value("machdep.cpu.brand_string")
            or platform.processor()
            or platform.machine()
        ),
        "physical_cpu_count": _positive_int(_sysctl_value("hw.physicalcpu")),
        "logical_cpu_count": int(os.cpu_count() or 1),
        "memory_bytes": _memory_bytes(),
    }
    return {**payload, "sha256": campaign.json_sha256(payload)}


def write_create_only(path: Path, payload: bytes) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _json_payload(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def fixed_worker_environment(threads: int, cache_dir: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONSTARTUP", None)
    environment.pop("PYTHONINSPECT", None)
    for key in tuple(environment):
        if key.startswith(("NUMBA_", "OMP_", "KMP_", "MKL_", "OPENBLAS_")):
            environment.pop(key)
    for key in THREAD_ENV_KEYS:
        environment[key] = str(int(threads))
    environment.update(
        {
            "DARKOFIT_WARMUP": "0",
            "NUMBA_CACHE_DIR": str(cache_dir.expanduser().resolve()),
            "NUMBA_DISABLE_JIT": "0",
            "NUMBA_NUM_THREADS": str(int(threads)),
            "NUMBA_THREADING_LAYER": "default",
            "OMP_DYNAMIC": "FALSE",
            "OMP_THREAD_LIMIT": str(int(threads)),
            "MKL_DYNAMIC": "FALSE",
            "PYTHONHASHSEED": "0",
        }
    )
    return environment


def _worker_environment_record(
    contract: Mapping[str, Any], threads: int
) -> dict[str, str | None]:
    runtime = contract.get("runtime")
    records = (
        runtime.get("worker_environments")
        if isinstance(runtime, Mapping)
        else None
    )
    record = records.get(str(int(threads))) if isinstance(records, Mapping) else None
    if not isinstance(record, Mapping) or set(record) != set(WORKER_ENV_KEYS):
        raise RuntimeError("fused-lane frozen worker environment is invalid")
    expected = dict(record)
    if any(
        value is not None and not isinstance(value, str)
        for value in expected.values()
    ):
        raise RuntimeError("fused-lane frozen worker environment is invalid")
    cache_dir = expected.get("NUMBA_CACHE_DIR")
    if (
        not isinstance(cache_dir, str)
        or not cache_dir
        or not Path(cache_dir).is_absolute()
        or any(expected.get(key) != str(int(threads)) for key in THREAD_ENV_KEYS)
        or expected.get("NUMBA_NUM_THREADS") != str(int(threads))
        or expected.get("OMP_THREAD_LIMIT") != str(int(threads))
    ):
        raise RuntimeError("fused-lane frozen worker environment is invalid")
    return expected


def _worker_process_environment(
    contract: Mapping[str, Any], threads: int
) -> dict[str, str]:
    expected = _worker_environment_record(contract, threads)
    environment = fixed_worker_environment(
        threads, Path(str(expected["NUMBA_CACHE_DIR"]))
    )
    actual = {name: environment.get(name) for name in WORKER_ENV_KEYS}
    if actual != expected:
        raise RuntimeError("fused-lane parent worker environment drifted")
    return environment


def assert_worker_environment(
    threads: int, expected_environment: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        not isinstance(expected_environment, Mapping)
        or set(expected_environment) != set(WORKER_ENV_KEYS)
    ):
        raise RuntimeError("fused-lane frozen worker environment is invalid")
    expected = dict(expected_environment)
    actual = {name: os.environ.get(name) for name in WORKER_ENV_KEYS}
    mismatches = {
        name: {"expected": value, "actual": actual.get(name)}
        for name, value in expected.items()
        if actual.get(name) != value
    }
    if mismatches:
        raise RuntimeError(
            "fused-lane worker environment drifted: "
            + json.dumps(mismatches, sort_keys=True)
        )
    import numba

    ceiling = int(numba.config.NUMBA_NUM_THREADS)
    current = int(numba.get_num_threads())
    if ceiling != int(threads) or current != int(threads):
        raise RuntimeError(
            "fused-lane Numba runtime drifted: "
            f"ceiling={ceiling}, current={current}, expected={threads}"
        )
    return {
        "ceiling": ceiling,
        "current": current,
        "threading_layer": str(numba.threading_layer()),
        "environment": actual,
    }


def _validate_bound_files(contract: Mapping[str, Any]) -> None:
    records = contract.get("bound_files")
    if not isinstance(records, Mapping) or not records:
        raise RuntimeError("fused-lane execution contract has no bound files")
    for name, record in records.items():
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise RuntimeError(f"fused-lane bound file is invalid: {name}")
        path = (ROOT / record["path"]).resolve()
        if (
            not path.is_relative_to(ROOT)
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != record.get("bytes")
            or campaign.file_sha256(path) != record.get("sha256")
        ):
            raise RuntimeError(f"fused-lane bound file drifted: {name}")


def _declared_output_path(contract: Mapping[str, Any], name: str) -> Path:
    outputs = contract.get("outputs")
    relative = outputs.get(name) if isinstance(outputs, Mapping) else None
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"fused-lane contract has no declared {name} path")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise RuntimeError(f"fused-lane contract {name} path escapes the repository")
    return path


def _require_declared_path(
    path: Path, contract: Mapping[str, Any], name: str
) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != _declared_output_path(contract, name):
        raise RuntimeError(f"fused-lane {name} path does not match the contract")
    return resolved


def load_execution_contract(path: Path, *, phase: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("fused-lane execution contract is unavailable")
    contract = json.loads(path.read_text(encoding="utf-8"))
    state = git_state(ROOT)
    runtime = contract.get("runtime") if isinstance(contract, dict) else None
    if (
        not isinstance(contract, dict)
        or contract.get("schema_version") != campaign.SCHEMA_VERSION
        or contract.get("campaign") != campaign.CAMPAIGN_NAME
        or contract.get("phase") != phase
        or contract.get("contract_frozen") is not True
        or contract.get("outcomes_opened") is not False
        or not _is_hex_digest(contract.get("source"), 40)
        or not isinstance(runtime, Mapping)
        or runtime.get("fingerprint") != runtime_fingerprint()
        or state["status"]
    ):
        raise RuntimeError("fused-lane execution contract or source is not frozen")
    generator = contract.get("generator")
    specs = generator.get("specs") if isinstance(generator, Mapping) else None
    if not isinstance(specs, list) or not specs or any(
        not isinstance(spec, Mapping) for spec in specs
    ):
        raise RuntimeError("fused-lane execution contract has no frozen specs")
    thread_counts = set()
    for spec in specs:
        threads = spec.get("threads")
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise RuntimeError("fused-lane execution contract has invalid threads")
        thread_counts.add(int(threads))
    runtime_environments = runtime.get("worker_environments")
    if (
        not isinstance(runtime_environments, Mapping)
        or set(runtime_environments) != {str(value) for value in thread_counts}
    ):
        raise RuntimeError("fused-lane execution contract environments drifted")
    for threads in thread_counts:
        _worker_environment_record(contract, threads)
    _validate_bound_files(contract)
    return contract


def require_authorization(
    authorization_path: Path,
    *,
    contract_path: Path,
    contract: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    authorization_path = _require_declared_path(
        authorization_path, contract, "authorization"
    )
    if not authorization_path.is_file() or authorization_path.is_symlink():
        raise RuntimeError("fused-lane execution is not owner-authorized")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    expected_contract_hash = campaign.file_sha256(contract_path.resolve())
    execution_identity = contract.get("execution_identity")
    if (
        not isinstance(authorization, dict)
        or authorization.get("schema_version") != campaign.SCHEMA_VERSION
        or authorization.get("campaign") != campaign.CAMPAIGN_NAME
        or authorization.get("phase") != phase
        or authorization.get("execution_authorized") is not True
        or authorization.get("execution_contract_sha256")
        != expected_contract_hash
        or authorization.get("source") != contract.get("source")
        or (
            execution_identity is not None
            and authorization.get("execution_identity") != execution_identity
        )
        or not isinstance(authorization.get("owner_decision"), str)
        or not authorization["owner_decision"].strip()
    ):
        raise RuntimeError("fused-lane owner authorization is invalid")
    return authorization


def _require_frozen_worker_spec(
    contract: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise RuntimeError("fused-lane worker spec is not frozen")
    generator = contract.get("generator")
    specs = generator.get("specs") if isinstance(generator, Mapping) else None
    if not isinstance(specs, list):
        raise RuntimeError("fused-lane worker spec is not frozen")
    matches = [candidate for candidate in specs if candidate == dict(spec)]
    if len(matches) != 1:
        raise RuntimeError("fused-lane worker spec is not a frozen coordinate")
    return dict(matches[0])


def _require_frozen_validation_coordinate(
    contract: Mapping[str, Any], *, arm: str, block: int
) -> None:
    execution = contract.get("execution")
    orders = execution.get("block_orders") if isinstance(execution, Mapping) else None
    if (
        not isinstance(orders, list)
        or any(
            not isinstance(order, list)
            or not order
            or any(value not in {"fused", "auto"} for value in order)
            for order in orders
        )
        or isinstance(block, bool)
        or not isinstance(block, int)
        or block < 0
        or block >= len(orders)
        or arm not in orders[block]
    ):
        raise RuntimeError("fused-lane validation worker coordinate is not frozen")


def _load_validation_threshold(
    contract: Mapping[str, Any], threshold_path: Path
) -> tuple[dict[str, Any], int, Path]:
    relative = contract.get("calibration_threshold_path")
    if not isinstance(relative, str) or not relative:
        raise RuntimeError("validation contract has no frozen threshold path")
    expected_path = (ROOT / relative).resolve()
    threshold_path = threshold_path.expanduser().resolve()
    if (
        not expected_path.is_relative_to(ROOT)
        or threshold_path != expected_path
        or not threshold_path.is_file()
        or threshold_path.is_symlink()
        or campaign.file_sha256(threshold_path)
        != contract.get("calibration_threshold_sha256")
    ):
        raise RuntimeError("validation threshold artifact is invalid")
    threshold_record = json.loads(threshold_path.read_text(encoding="utf-8"))
    threshold = (
        threshold_record.get("selected", {}).get("threshold")
        if isinstance(threshold_record, Mapping)
        else None
    )
    if (
        not isinstance(threshold_record, dict)
        or threshold_record.get("qualifies") is not True
        or isinstance(threshold, bool)
        or not isinstance(threshold, int)
        or threshold < 0
    ):
        raise RuntimeError("validation threshold artifact is invalid")
    return threshold_record, int(threshold), threshold_path


def _authorize_worker_invocation(
    *,
    contract_path: Path,
    authorization_path: Path,
    phase: str,
    spec: Mapping[str, Any],
    source: str,
    arm: str | None = None,
    block: int | None = None,
    threshold_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int | None]:
    contract = load_execution_contract(contract_path, phase=phase)
    authorization = require_authorization(
        authorization_path,
        contract_path=contract_path,
        contract=contract,
        phase=phase,
    )
    if source != contract.get("source"):
        raise RuntimeError("fused-lane worker source does not match the contract")
    frozen_spec = _require_frozen_worker_spec(contract, spec)
    threshold = None
    if phase == "calibration":
        if arm is not None or block is not None or threshold_path is not None:
            raise RuntimeError("fused-lane calibration worker coordinate is invalid")
    elif phase == "validation":
        if arm is None or block is None or threshold_path is None:
            raise RuntimeError("fused-lane validation worker coordinate is invalid")
        _require_frozen_validation_coordinate(contract, arm=arm, block=block)
        _threshold_record, threshold, _threshold_path = _load_validation_threshold(
            contract, threshold_path
        )
    else:
        raise RuntimeError("fused-lane worker phase is invalid")
    expected_environment = _worker_environment_record(
        contract, int(frozen_spec["threads"])
    )
    runtime = assert_worker_environment(
        int(frozen_spec["threads"]), expected_environment
    )
    return contract, authorization, runtime, threshold


def _tree_state(tree, leaf, leaf_g, leaf_h, probe_X) -> dict[str, Any]:
    arrays = {
        "splits_feat": tree.splits_feat,
        "splits_thr": tree.splits_thr,
        "values": tree.values,
        "gains": tree.gains,
        "leaf": leaf,
        "leaf_g": leaf_g,
        "leaf_h": leaf_h,
    }
    return {
        "state_sha256": campaign.named_arrays_sha256(arrays),
        "prediction_sha256": campaign.array_sha256(
            "prediction", tree.predict(probe_X)
        ),
        "depth": int(tree.depth),
        "arrays": arrays,
    }


def _tree_states_exact(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        left["state_sha256"] == right["state_sha256"]
        and left["prediction_sha256"] == right["prediction_sha256"]
        and left["depth"] == right["depth"]
        and all(
            np.array_equal(left["arrays"][name], right["arrays"][name])
            for name in left["arrays"]
        )
    )


def _calibration_buffers(spec: Mapping[str, Any]):
    features = int(spec["features"])
    leaves = 1 << int(spec["depth"])
    bins = int(spec["bins"])
    hist = tuple(
        np.empty((features, leaves, bins), dtype=np.float64)
        for _ in range(2)
    )
    split = (
        *(np.empty((features, leaves), dtype=np.float64) for _ in range(5)),
        np.empty((features, leaves), dtype=np.int64),
    )
    return hist, split


def _activate_source(source_root: Path, source: str) -> dict[str, str]:
    source_root = source_root.expanduser().resolve()
    state = git_state(source_root)
    if state != {"head": source, "status": ""}:
        raise RuntimeError("fused-lane source worktree is not the exact clean pin")
    sys.path.insert(0, str(source_root))
    return state


def calibration_worker(
    spec: Mapping[str, Any],
    *,
    contract_path: Path,
    authorization_path: Path,
    source: str,
    source_root: Path,
) -> dict[str, Any]:
    contract, _authorization, runtime_before, _threshold = (
        _authorize_worker_invocation(
            contract_path=contract_path,
            authorization_path=authorization_path,
            phase="calibration",
            spec=spec,
            source=source,
        )
    )
    state_before = _activate_source(source_root, source)
    if state_before != {"head": source, "status": ""}:
        raise RuntimeError("calibration worker source is not frozen")
    import numba
    import darkofit.tree as tree_module

    build_oblivious_tree = tree_module.build_oblivious_tree
    implementation_path = Path(tree_module.__file__).resolve()
    if not implementation_path.is_relative_to(source_root.expanduser().resolve()):
        raise RuntimeError("calibration imported DarkoFit outside the source pin")

    values, fingerprints = campaign.generate_calibration_case(spec)
    X = values["X"]
    X_hist = np.asfortranarray(X)
    grad = values["grad"]
    hess = values["hess"]
    n_bins = values["n_bins"]
    buffers = {lane: _calibration_buffers(spec) for lane in campaign.LANES}
    probe_X = X[: min(len(X), 4096)]

    def build(lane: str):
        fused_counter = np.zeros(1, dtype=np.int64)
        unfused_counter = np.zeros(1, dtype=np.int64)
        hist_buffers, split_buffers = buffers[lane]
        started = time.perf_counter_ns()
        tree, leaf, leaf_g, leaf_h = build_oblivious_tree(
            X,
            grad,
            hess,
            n_bins,
            int(spec["depth"]),
            3.0,
            0.1,
            min_child_weight=0.0,
            hist_buffers=hist_buffers,
            split_buffers=split_buffers,
            return_training_state=True,
            X_hist_binned=X_hist,
            X_route_binned=X,
            constant_hessian=spec["hessian"] == "unit",
            level_histogram_subtraction=False,
            random_strength=0.0,
            fused_oblivious_kernel=lane == "fused",
            fused_oblivious_counter=fused_counter,
            unfused_oblivious_counter=unfused_counter,
        )
        elapsed = (time.perf_counter_ns() - started) / 1e9
        return {
            "seconds": float(elapsed),
            "fused_level_count": int(fused_counter[0]),
            "unfused_level_count": int(unfused_counter[0]),
            "state": _tree_state(tree, leaf, leaf_g, leaf_h, probe_X),
        }

    for _ in range(campaign.CALIBRATION_WARMUPS):
        for lane in campaign.LANES:
            build(lane)
    repetitions = []
    for repeat in range(campaign.CALIBRATION_REPEATS):
        order = campaign.calibration_order(repeat)
        results = {lane: build(lane) for lane in order}
        fused = results["fused"]
        unfused = results["unfused"]
        repetitions.append(
            {
                "repeat": repeat,
                "order": list(order),
                "fused_seconds": fused["seconds"],
                "unfused_seconds": unfused["seconds"],
                "exact": _tree_states_exact(fused["state"], unfused["state"]),
                "state_sha256": fused["state"]["state_sha256"],
                "prediction_sha256": fused["state"]["prediction_sha256"],
                "tree_depth": fused["state"]["depth"],
                "fused_level_count": fused["fused_level_count"],
                "fused_opposite_level_count": fused["unfused_level_count"],
                "unfused_level_count": unfused["unfused_level_count"],
                "unfused_opposite_level_count": unfused["fused_level_count"],
            }
        )
    runtime_after = {
        **runtime_before,
        "current": int(numba.get_num_threads()),
    }
    state_after = git_state(source_root)
    if state_after != state_before:
        raise RuntimeError("calibration worker source changed during execution")
    return {
        **dict(spec),
        "seed": campaign.CALIBRATION_SEED,
        "warmups_per_lane": campaign.CALIBRATION_WARMUPS,
        "fingerprints": fingerprints,
        "runtime_before": runtime_before,
        "runtime_after": runtime_after,
        "thread_mask_restored": runtime_after["current"] == int(spec["threads"]),
        "source": source,
        "execution_identity": contract.get("execution_identity"),
        "execution_contract_sha256": campaign.file_sha256(contract_path),
        "authorization_sha256": campaign.file_sha256(authorization_path),
        "implementation_path": str(implementation_path),
        "repetitions": repetitions,
    }


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _full_fit_model(spec: Mapping[str, Any], *, arm: str, rounds: int):
    from darkofit.booster import GradientBoosting

    return GradientBoosting(
        loss=("Logloss" if spec["task"] == "binary_logloss" else "RMSE"),
        iterations=int(rounds),
        learning_rate=0.1,
        depth=int(spec["depth"]),
        l2_leaf_reg=3.0,
        max_bins=int(spec["max_bins"]),
        subsample=1.0,
        colsample=1.0,
        thread_count=int(spec["threads"]),
        random_state=campaign.VALIDATION_SEED,
        ordered_boosting=False,
        tree_mode="catboost",
        random_strength=0.0,
        use_best_model=False,
        diagnostic_warnings="never",
        verbose_timing=True,
        oblivious_kernel=arm,
    )


def validation_worker(
    spec: Mapping[str, Any],
    *,
    arm: str,
    block: int,
    threshold_path: Path,
    contract_path: Path,
    authorization_path: Path,
    source: str,
    source_root: Path,
) -> dict[str, Any]:
    contract, _authorization, runtime_before, threshold = (
        _authorize_worker_invocation(
            contract_path=contract_path,
            authorization_path=authorization_path,
            phase="validation",
            spec=spec,
            source=source,
            arm=arm,
            block=block,
            threshold_path=threshold_path,
        )
    )
    if threshold is None:
        raise RuntimeError("validation threshold artifact is invalid")
    state_before = _activate_source(source_root, source)
    if state_before != {"head": source, "status": ""}:
        raise RuntimeError("validation worker source is not frozen")
    import numba
    import darkofit.booster as booster_module
    from darkofit.booster import GradientBoosting

    if booster_module._OBLIVIOUS_KERNEL_AUTO_THRESHOLD != int(threshold):
        raise RuntimeError("validation source does not contain the frozen threshold")
    implementation_path = Path(booster_module.__file__).resolve()
    if not implementation_path.is_relative_to(source_root.expanduser().resolve()):
        raise RuntimeError("validation imported DarkoFit outside the source pin")
    values, fingerprints = campaign.generate_validation_case(spec)
    X = values["X"]
    y = values["y"]
    sample_weight = values["sample_weight"]
    ambient = int(numba.get_num_threads())
    warmup = _full_fit_model(spec, arm=arm, rounds=3)
    warmup.fit(X, y, sample_weight=sample_weight)
    after_warmup = int(numba.get_num_threads())
    del warmup
    gc.collect()

    model = _full_fit_model(spec, arm=arm, rounds=int(spec["rounds"]))
    started = time.perf_counter_ns()
    model.fit(X, y, sample_weight=sample_weight)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    tree_seconds = float(model.timing_["tree_build"])
    # Timing is external evidence, not canonical fitted state. Keep it in the
    # worker row and remove it before the archive exactness oracle.
    model.timing_ = None
    after_fit = int(numba.get_num_threads())
    peak_rss_bytes = _peak_rss_bytes()
    probe = X[: min(len(X), 4096)]
    raw_prediction = model.predict_raw(probe)
    if spec["task"] == "binary_logloss":
        from darkofit.losses import _sigmoid

        positive = _sigmoid(raw_prediction)
        probability = np.column_stack((1.0 - positive, positive))
        probability_sha256 = campaign.array_sha256(
            "probability", probability
        )
    else:
        probability_sha256 = None
    after_predict = int(numba.get_num_threads())
    dispatch = model.oblivious_kernel_dispatch_
    selected_count = dispatch[f"{dispatch['resolved']}_level_count"]
    opposite = "unfused" if dispatch["resolved"] == "fused" else "fused"
    opposite_count = dispatch[f"{opposite}_level_count"]

    with tempfile.TemporaryDirectory(prefix="darkofit-fused-lane-") as directory:
        model_path = Path(directory) / "model.npz"
        resaved_path = Path(directory) / "resaved.npz"
        model.save_model(model_path)
        projected_digest = campaign.canonical_archive_sha256(
            model_path, project_dispatch=True
        )
        full_digest = campaign.canonical_archive_sha256(
            model_path, project_dispatch=False
        )
        archive_bytes = model_path.stat().st_size
        loaded = GradientBoosting.load_model(model_path)
        loaded_prediction = loaded.predict_raw(probe)
        loaded.save_model(resaved_path)
        roundtrip_exact = bool(
            np.array_equal(raw_prediction, loaded_prediction)
            and campaign.canonical_archive_sha256(
                resaved_path, project_dispatch=False
            )
            == full_digest
            and loaded.oblivious_kernel_dispatch_ == dispatch
        )
    runtime_after = {
        **runtime_before,
        "current": int(numba.get_num_threads()),
    }
    state_after = git_state(source_root)
    if state_after != state_before:
        raise RuntimeError("validation worker source changed during execution")
    return {
        **dict(spec),
        "arm": arm,
        "block": int(block),
        "threshold": int(threshold),
        "source": source,
        "execution_identity": contract.get("execution_identity"),
        "execution_contract_sha256": campaign.file_sha256(contract_path),
        "authorization_sha256": campaign.file_sha256(authorization_path),
        "threshold_sha256": campaign.file_sha256(threshold_path),
        "implementation_path": str(implementation_path),
        "seed": campaign.VALIDATION_SEED,
        "fingerprints": fingerprints,
        "dataset_sha256": fingerprints["dataset_sha256"],
        "fit_seconds": float(fit_seconds),
        "tree_seconds": tree_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "projected_archive_sha256": projected_digest,
        "archive_sha256": full_digest,
        "archive_bytes": int(archive_bytes),
        "prediction_sha256": campaign.array_sha256(
            "prediction", raw_prediction
        ),
        "probability_sha256": probability_sha256,
        "feature_importance_sha256": campaign.array_sha256(
            "feature_importances", model.feature_importances_
        ),
        "safe_roundtrip_exact": roundtrip_exact,
        "requested_lane": dispatch["requested"],
        "resolved_lane": dispatch["resolved"],
        "dispatch_reason": dispatch["reason"],
        "dispatch_metadata": dispatch,
        "selected_level_count": int(selected_count),
        "opposite_level_count": int(opposite_count),
        "thread_counts": {
            "ambient": ambient,
            "after_warmup": after_warmup,
            "after_fit": after_fit,
            "after_predict": after_predict,
            "after_roundtrip": runtime_after["current"],
        },
        "thread_mask_restored": (
            ambient
            == after_warmup
            == after_fit
            == after_predict
            == runtime_after["current"]
        ),
        "runtime_before": runtime_before,
        "runtime_after": runtime_after,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def _extract_worker_payload(stdout: str, prefix: str) -> dict[str, Any]:
    records = [
        line[len(prefix):]
        for line in stdout.splitlines()
        if line.startswith(prefix)
    ]
    if len(records) != 1:
        raise RuntimeError("fused-lane worker emitted an invalid result count")
    value = json.loads(records[0])
    if not isinstance(value, dict):
        raise RuntimeError("fused-lane worker result is not an object")
    return value


def _run_worker(
    args: list[str],
    *,
    threads: int,
    contract: Mapping[str, Any],
    contract_path: Path,
    authorization_path: Path,
    source: str,
    source_root: Path,
    prefix: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(RUNNER_PATH),
        *args,
        "--contract",
        str(contract_path.expanduser().resolve()),
        "--authorization",
        str(authorization_path.expanduser().resolve()),
        "--source",
        source,
        "--source-root",
        str(source_root.expanduser().resolve()),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_worker_process_environment(contract, threads),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "fused-lane worker failed: "
            + json.dumps(
                {
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                },
                sort_keys=True,
            )
        )
    return _extract_worker_payload(completed.stdout, prefix)


def _terminal_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_terminal.json")


def _assert_fresh_output(output: Path) -> Path:
    output = output.expanduser().resolve()
    terminal = _terminal_path(output)
    if (
        output.exists()
        or output.is_symlink()
        or terminal.exists()
        or terminal.is_symlink()
    ):
        raise RuntimeError("fused-lane execution identity was already used")
    return output


def _run_or_terminal(
    operation,
    *,
    output: Path,
    phase: str,
    contract_path: Path,
    source: str,
    execution_identity: str | None,
) -> dict[str, Any]:
    try:
        result = operation()
        write_create_only(output, _json_payload(result))
        return result
    except BaseException as exc:
        terminal = {
            "schema_version": campaign.SCHEMA_VERSION,
            "campaign": campaign.CAMPAIGN_NAME,
            "phase": phase,
            "execution_identity": execution_identity,
            "terminal": True,
            "rerun_allowed": False,
            "source": source,
            "execution_contract_sha256": campaign.file_sha256(contract_path),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "partial_rows_published": False,
        }
        write_create_only(_terminal_path(output), _json_payload(terminal))
        raise


def run_calibration(
    *,
    contract_path: Path,
    authorization_path: Path,
    source_root: Path,
    output: Path,
) -> dict[str, Any]:
    contract = load_execution_contract(contract_path, phase="calibration")
    output = _require_declared_path(output, contract, "raw")
    if _terminal_path(output) != _declared_output_path(contract, "terminal"):
        raise RuntimeError("fused-lane terminal path does not match the contract")
    authorization = require_authorization(
        authorization_path,
        contract_path=contract_path,
        contract=contract,
        phase="calibration",
    )
    source = str(contract["source"])
    harness_state = git_state(ROOT)
    source_state = git_state(source_root)
    if source_state != {"head": source, "status": ""}:
        raise RuntimeError("calibration source worktree is not the clean pin")
    output = _assert_fresh_output(output)

    def operation():
        rows = []
        for spec in campaign.calibration_specs():
            rows.append(
                _run_worker(
                    ["calibration-worker", "--spec-json", json.dumps(spec)],
                    threads=int(spec["threads"]),
                    contract=contract,
                    contract_path=contract_path,
                    authorization_path=authorization_path,
                    source=source,
                    source_root=source_root,
                    prefix=CALIBRATION_PREFIX,
                )
            )
        if git_state(ROOT) != harness_state or git_state(source_root) != source_state:
            raise RuntimeError("calibration harness or source changed during execution")
        return {
            "schema_version": campaign.SCHEMA_VERSION,
            "campaign": campaign.CAMPAIGN_NAME,
            "phase": "calibration",
            "execution_identity": contract.get("execution_identity"),
            "source": source,
            "execution_contract_sha256": campaign.file_sha256(contract_path),
            "authorization_sha256": campaign.file_sha256(authorization_path),
            "authorization": authorization,
            "harness_state": harness_state,
            "source_state": source_state,
            "rows": rows,
        }

    return _run_or_terminal(
        operation,
        output=output,
        phase="calibration",
        contract_path=contract_path,
        source=source,
        execution_identity=contract.get("execution_identity"),
    )


def run_validation(
    *,
    contract_path: Path,
    authorization_path: Path,
    source_root: Path,
    threshold_path: Path,
    output: Path,
) -> dict[str, Any]:
    contract = load_execution_contract(contract_path, phase="validation")
    output = _require_declared_path(output, contract, "raw")
    if _terminal_path(output) != _declared_output_path(contract, "terminal"):
        raise RuntimeError("fused-lane terminal path does not match the contract")
    authorization = require_authorization(
        authorization_path,
        contract_path=contract_path,
        contract=contract,
        phase="validation",
    )
    _threshold_record, threshold, threshold_path = _load_validation_threshold(
        contract, threshold_path
    )
    source = str(contract["source"])
    harness_state = git_state(ROOT)
    source_state = git_state(source_root)
    if source_state != {"head": source, "status": ""}:
        raise RuntimeError("validation source worktree is not the clean pin")
    output = _assert_fresh_output(output)

    def operation():
        rows = []
        for spec in campaign.validation_specs():
            for block, order in enumerate(campaign.VALIDATION_BLOCK_ORDERS):
                for arm in order:
                    rows.append(
                        _run_worker(
                            [
                                "validation-worker",
                                "--spec-json",
                                json.dumps(spec),
                                "--arm",
                                arm,
                                "--block",
                                str(block),
                                "--threshold-artifact",
                                str(threshold_path),
                            ],
                            threads=int(spec["threads"]),
                            contract=contract,
                            contract_path=contract_path,
                            authorization_path=authorization_path,
                            source=source,
                            source_root=source_root,
                            prefix=VALIDATION_PREFIX,
                        )
                    )
        if git_state(ROOT) != harness_state or git_state(source_root) != source_state:
            raise RuntimeError("validation harness or source changed during execution")
        return {
            "schema_version": campaign.SCHEMA_VERSION,
            "campaign": campaign.CAMPAIGN_NAME,
            "phase": "validation",
            "execution_identity": contract.get("execution_identity"),
            "source": source,
            "threshold": threshold,
            "threshold_sha256": campaign.file_sha256(threshold_path),
            "execution_contract_sha256": campaign.file_sha256(contract_path),
            "authorization_sha256": campaign.file_sha256(authorization_path),
            "authorization": authorization,
            "harness_state": harness_state,
            "source_state": source_state,
            "rows": rows,
        }

    return _run_or_terminal(
        operation,
        output=output,
        phase="validation",
        contract_path=contract_path,
        source=source,
        execution_identity=contract.get("execution_identity"),
    )


def analyze_raw(
    *,
    raw_path: Path,
    contract_path: Path,
    output: Path,
    threshold_path: Path | None = None,
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise RuntimeError("fused-lane analysis contract is invalid")
    raw_path = _require_declared_path(raw_path, contract, "raw")
    output = _require_declared_path(output, contract, "analysis")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    phase = raw.get("phase")
    authorization_path = _declared_output_path(contract, "authorization")
    authorization = require_authorization(
        authorization_path,
        contract_path=contract_path,
        contract=contract,
        phase=str(phase),
    )
    if (
        raw.get("schema_version") != campaign.SCHEMA_VERSION
        or raw.get("campaign") != campaign.CAMPAIGN_NAME
        or contract.get("phase") != phase
        or contract.get("source") != raw.get("source")
        or contract.get("execution_identity")
        != raw.get("execution_identity")
        or campaign.file_sha256(contract_path)
        != raw.get("execution_contract_sha256")
        or raw.get("authorization") != authorization
        or raw.get("authorization_sha256")
        != campaign.file_sha256(authorization_path)
        or raw.get("source_state")
        != {"head": raw.get("source"), "status": ""}
        or not isinstance(raw.get("harness_state"), Mapping)
        or raw["harness_state"].get("status") != ""
    ):
        raise RuntimeError("fused-lane raw artifact binding is invalid")
    _validate_bound_files(contract)
    rows = raw.get("rows")
    contract_sha256 = campaign.file_sha256(contract_path)
    authorization_sha256 = campaign.file_sha256(authorization_path)
    if (
        not isinstance(rows, list)
        or any(
            not isinstance(row, Mapping)
            or row.get("source") != raw.get("source")
            or row.get("execution_identity") != raw.get("execution_identity")
            or row.get("execution_contract_sha256") != contract_sha256
            or row.get("authorization_sha256") != authorization_sha256
            for row in rows
        )
    ):
        raise RuntimeError("fused-lane raw artifact has no rows")
    if phase == "calibration":
        if threshold_path is not None:
            raise RuntimeError("calibration analysis cannot accept a threshold")
        result = campaign.analyze_calibration(rows)
    elif phase == "validation" and threshold_path is not None:
        _threshold_record, threshold, threshold_path = (
            _load_validation_threshold(contract, threshold_path)
        )
        threshold_sha256 = campaign.file_sha256(threshold_path)
        if (
            raw.get("threshold") != threshold
            or raw.get("threshold_sha256") != threshold_sha256
            or any(
                row.get("threshold") != threshold
                or row.get("threshold_sha256") != threshold_sha256
                for row in rows
            )
        ):
            raise RuntimeError("fused-lane validation threshold binding is invalid")
        result = campaign.analyze_validation(rows, threshold=int(threshold))
    else:
        raise RuntimeError("fused-lane analysis phase is invalid")
    result.update(
        {
            "raw_sha256": campaign.file_sha256(raw_path),
            "source": raw.get("source"),
            "execution_identity": raw.get("execution_identity"),
            "execution_contract_sha256": raw.get("execution_contract_sha256"),
        }
    )
    write_create_only(output, _json_payload(result))
    return result


def _parse_spec(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("spec must be a JSON object")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    calibration_worker_parser = commands.add_parser("calibration-worker")
    calibration_worker_parser.add_argument(
        "--spec-json", type=_parse_spec, required=True
    )
    calibration_worker_parser.add_argument("--contract", type=Path, required=True)
    calibration_worker_parser.add_argument("--authorization", type=Path, required=True)
    calibration_worker_parser.add_argument("--source", required=True)
    calibration_worker_parser.add_argument("--source-root", type=Path, required=True)

    validation_worker_parser = commands.add_parser("validation-worker")
    validation_worker_parser.add_argument(
        "--spec-json", type=_parse_spec, required=True
    )
    validation_worker_parser.add_argument(
        "--arm", choices=("fused", "auto"), required=True
    )
    validation_worker_parser.add_argument("--block", type=int, required=True)
    validation_worker_parser.add_argument(
        "--threshold-artifact", type=Path, required=True
    )
    validation_worker_parser.add_argument("--contract", type=Path, required=True)
    validation_worker_parser.add_argument("--authorization", type=Path, required=True)
    validation_worker_parser.add_argument("--source", required=True)
    validation_worker_parser.add_argument("--source-root", type=Path, required=True)

    for name in ("calibration", "validation"):
        phase_parser = commands.add_parser(name)
        phase_parser.add_argument("--contract", type=Path, required=True)
        phase_parser.add_argument("--authorization", type=Path, required=True)
        phase_parser.add_argument("--source-root", type=Path, required=True)
        phase_parser.add_argument("--output", type=Path, required=True)
        if name == "validation":
            phase_parser.add_argument("--threshold-artifact", type=Path, required=True)

    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--raw", type=Path, required=True)
    analyze_parser.add_argument("--contract", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--threshold-artifact", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "calibration-worker":
        result = calibration_worker(
            args.spec_json,
            contract_path=args.contract,
            authorization_path=args.authorization,
            source=args.source,
            source_root=args.source_root,
        )
        print(CALIBRATION_PREFIX + json.dumps(result, sort_keys=True, allow_nan=False))
    elif args.command == "validation-worker":
        result = validation_worker(
            args.spec_json,
            arm=args.arm,
            block=args.block,
            threshold_path=args.threshold_artifact,
            contract_path=args.contract,
            authorization_path=args.authorization,
            source=args.source,
            source_root=args.source_root,
        )
        print(VALIDATION_PREFIX + json.dumps(result, sort_keys=True, allow_nan=False))
    elif args.command == "calibration":
        run_calibration(
            contract_path=args.contract,
            authorization_path=args.authorization,
            source_root=args.source_root,
            output=args.output,
        )
    elif args.command == "validation":
        run_validation(
            contract_path=args.contract,
            authorization_path=args.authorization,
            source_root=args.source_root,
            threshold_path=args.threshold_artifact,
            output=args.output,
        )
    else:
        analyze_raw(
            raw_path=args.raw,
            contract_path=args.contract,
            output=args.output,
            threshold_path=args.threshold_artifact,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
