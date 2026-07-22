#!/usr/bin/env python3
"""Run the frozen ensemble-v3 fit/resource/prediction characterization."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import inspect
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from . import run_m3b_ensemble_v3 as m3b
except ImportError:  # direct script execution
    import run_m3b_ensemble_v3 as m3b


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmarks" / "ensemble_v3_characterization_protocol.md"
CONTRACT_PATH = ROOT / "benchmarks" / "ensemble_v3_characterization_contract.json"
ANALYZER_PATH = ROOT / "benchmarks" / "analyze_ensemble_v3_characterization.py"
FREEZER_PATH = ROOT / "benchmarks" / "freeze_ensemble_v3_characterization.py"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "ensemble_v3_characterization_raw.json"
DEFAULT_TERMINAL = ROOT / "benchmarks" / "ensemble_v3_characterization_terminal.json"

CONTRACT_ID = "ensemble-v3-characterization-v1"
DARKOFIT_HEAD = "c5e66ef7e6bdcf5665b55b81c6b870f42d76237b"
CHIMERABOOST_HEAD = "f14be606b641f1bf0dc92bb14b3951f1fe631c6b"
THREADS = 14
ITERATIONS = 600
PATIENCE = 30
RANDOM_STATE = 4
VALIDATION_FRACTION = 0.15
BATCH_SIZES = (8_192, 65_536, 524_288, 2_000_000)
TARGET_INTERVAL_SECONDS = 1.0
MIN_INTERVAL_SECONDS = 0.75
MIN_CALLS = 2
MAX_CALLS = 8192
BLOCKS = 3
WORKER_PREFIX = "ENSEMBLE_V3_CHARACTERIZATION_RESULT="
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

DARKO_SINGLE = "darkofit_single"
DARKO_V3 = "darkofit_ensemble_v3"
CHIMERA_SINGLE = "chimeraboost_0_18_single"
ARMS = (DARKO_SINGLE, DARKO_V3, CHIMERA_SINGLE)
CASES = (
    "general_friedman_numeric",
    "general_categorical_reg",
    "general_numeric_binary",
    "general_categorical_multiclass",
)

BOUND_PATHS = {
    "protocol": "benchmarks/ensemble_v3_characterization_protocol.md",
    "runner": "benchmarks/run_ensemble_v3_characterization.py",
    "analyzer": "benchmarks/analyze_ensemble_v3_characterization.py",
    "freezer": "benchmarks/freeze_ensemble_v3_characterization.py",
    "tests": "tests/test_ensemble_v3_characterization.py",
    "public_contract": "benchmarks/ensemble_v3_public_contract.md",
    "implementation": "darkofit/sklearn_api.py",
    "implementation_tests": "tests/test_ensemble_v3_release_candidate.py",
    "m3b_runner": "benchmarks/run_m3b_ensemble_v3.py",
    "m6_adapter": "benchmarks/benchmark_adapters.py",
    "paired_execution": "benchmarks/paired_evidence_contract.py",
    "historical_result": "benchmarks/m3b_ensemble_v3_r3_result.json",
    "historical_quality": "benchmarks/m3b_ensemble_v3_r3_quality.json",
    "historical_timing": "benchmarks/m3b_ensemble_v3_r3_timing.json",
    "historical_readout": (
        "benchmarks/m3b_ensemble_v3_r3_vs_single_readout_20260721.json"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def array_sha256(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(array.shape), "dtype": str(array.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if array.dtype.kind in "OUS":
        payload = json.dumps(
            array.tolist(), separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    else:
        payload = np.ascontiguousarray(array).tobytes()
    digest.update(payload)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args), cwd=repo, check=False, capture_output=True, text=True
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout.strip()


def source_state(repo: Path) -> dict[str, Any]:
    repo = repo.expanduser().resolve()
    root = Path(git(repo, "rev-parse", "--show-toplevel")).resolve()
    if root != repo:
        raise RuntimeError(f"source path must name its Git root: {repo}")
    status = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "path": str(repo),
        "head": git(repo, "rev-parse", "HEAD"),
        "tree": git(repo, "rev-parse", "HEAD^{tree}"),
        "clean": not bool(status),
        "status": status.splitlines(),
    }


def _bound_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def execution_spec() -> dict[str, Any]:
    return {
        "contract_id": CONTRACT_ID,
        "darkofit_head": DARKOFIT_HEAD,
        "chimeraboost_head": CHIMERABOOST_HEAD,
        "threads": THREADS,
        "iterations": ITERATIONS,
        "patience": PATIENCE,
        "random_state": RANDOM_STATE,
        "validation_fraction": VALIDATION_FRACTION,
        "cases": list(CASES),
        "arms": list(ARMS),
        "blocks": BLOCKS,
        "orders": [list(order_for(case, block)) for case in CASES for block in range(BLOCKS)],
        "batch_sizes": list(BATCH_SIZES),
        "target_interval_seconds": TARGET_INTERVAL_SECONDS,
        "minimum_interval_seconds": MIN_INTERVAL_SECONDS,
        "minimum_calls": MIN_CALLS,
        "maximum_calls": MAX_CALLS,
        "fresh_worker_per_case_arm_block": True,
        "no_reruns": True,
        "output_create_only": True,
    }


def order_for(case_id: str, block: int) -> tuple[str, ...]:
    case_index = CASES.index(case_id)
    offset = (case_index + int(block)) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("contract_id") != CONTRACT_ID
        or payload.get("contract_frozen") is not True
        or payload.get("execution") != execution_spec()
        or set(payload.get("bindings", {})) != set(BOUND_PATHS)
    ):
        raise RuntimeError("ensemble-v3 characterization contract is invalid")
    for name, relative in BOUND_PATHS.items():
        path_value = ROOT / relative
        record = payload["bindings"][name]
        if record != _bound_record(path_value):
            raise RuntimeError(f"characterization binding drifted: {name}")
    return payload


def _activate_sources(darkofit_source: Path, chimeraboost_source: Path) -> None:
    for source in (darkofit_source, chimeraboost_source):
        value = str(source.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)


def _implementation(model: Any, expected_source: Path) -> dict[str, Any]:
    module = importlib.import_module(model.__class__.__module__)
    path = Path(inspect.getfile(module)).resolve()
    if not path.is_relative_to(expected_source.resolve()):
        raise RuntimeError(f"model imported from unexpected source: {path}")
    return {"class": model.__class__.__name__, "module_path": str(path)}


def _case_spec(case_id: str) -> Mapping[str, Any]:
    for spec in m3b.case_specs():
        if spec["case_id"] == case_id:
            return spec
    raise ValueError(f"unknown characterization case: {case_id}")


def _build_model(
    arm: str, spec: Mapping[str, Any], darkofit_source: Path, chimeraboost_source: Path
):
    _activate_sources(darkofit_source, chimeraboost_source)
    task = str(spec["task"])
    if arm in {DARKO_SINGLE, DARKO_V3}:
        from darkofit import DarkoClassifier, DarkoRegressor

        estimator = DarkoRegressor if task == "regression" else DarkoClassifier
        model = estimator(
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
            n_ensembles=8 if arm == DARKO_V3 else 1,
        )
        return model, darkofit_source
    if arm != CHIMERA_SINGLE:
        raise ValueError(f"unknown characterization arm: {arm}")
    from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

    estimator = ChimeraBoostRegressor if task == "regression" else ChimeraBoostClassifier
    model = estimator(
        n_estimators=ITERATIONS,
        early_stopping_rounds=PATIENCE,
        early_stopping=True,
        validation_fraction=VALIDATION_FRACTION,
        random_state=RANDOM_STATE,
        thread_count=THREADS,
        n_ensembles=None,
        quantize_gradients=True,
    )
    return model, chimeraboost_source


def _fit(model: Any, arm: str, data: Mapping[str, Any]) -> Any:
    kwargs = {
        "cat_features": data["cat_features"],
        "sample_weight": data["w_fit"],
    }
    if arm == DARKO_V3:
        from darkofit.sklearn_api import _fit_ensemble_v3_release_candidate

        return _fit_ensemble_v3_release_candidate(
            model, data["X_fit"], data["y_fit"], **kwargs
        )
    return model.fit(data["X_fit"], data["y_fit"], **kwargs)


def _warmup(
    arm: str,
    spec: Mapping[str, Any],
    data: Mapping[str, Any],
    darkofit_source: Path,
    chimeraboost_source: Path,
) -> None:
    model, _ = _build_model(arm, spec, darkofit_source, chimeraboost_source)
    count = min(600, len(data["y_fit"]))
    indices = np.arange(count, dtype=np.int64)
    warm = {
        "X_fit": _take_rows(data["X_fit"], indices),
        "y_fit": np.asarray(data["y_fit"])[indices],
        "w_fit": None if data["w_fit"] is None else np.asarray(data["w_fit"])[indices],
        "cat_features": data["cat_features"],
    }
    if arm == CHIMERA_SINGLE:
        model.set_params(n_estimators=2)
    else:
        model.set_params(iterations=2)
    _fit(model, arm, warm)
    gc.collect()


def _take_rows(values: Any, indices: np.ndarray) -> Any:
    iloc = getattr(values, "iloc", None)
    if iloc is not None:
        return iloc[indices]
    return np.asarray(values)[indices]


def _repeat_rows(values: Any, rows: int) -> Any:
    indices = np.arange(int(rows), dtype=np.int64) % len(values)
    repeated = _take_rows(values, indices)
    reset = getattr(repeated, "reset_index", None)
    return reset(drop=True) if callable(reset) else repeated


def _valid_prediction(value: Any, rows: int) -> np.ndarray:
    output = np.asarray(value)
    if output.shape != (int(rows),):
        raise RuntimeError(f"invalid prediction shape: {output.shape}")
    if output.dtype.kind in "fc" and not np.all(np.isfinite(output)):
        raise RuntimeError("prediction contains non-finite values")
    return output


class ProcessTreeRSSSampler:
    """Sample aggregate RSS for this worker and recursive child processes."""

    def __init__(self, interval_seconds: float = 0.01):
        self.interval_seconds = float(interval_seconds)
        self.peak_bytes = 0
        self.samples = 0
        self.errors: list[str] = []
        self.start_bytes = 0
        self.end_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def current_bytes() -> int:
        import psutil

        root = psutil.Process()
        processes = [root, *root.children(recursive=True)]
        total = 0
        seen: set[int] = set()
        for process in processes:
            if process.pid in seen:
                continue
            seen.add(process.pid)
            try:
                total += int(process.memory_info().rss)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        if total <= 0:
            raise RuntimeError("process-tree RSS is unavailable")
        return total

    def _sample_once(self) -> None:
        value = self.current_bytes()
        self.peak_bytes = max(self.peak_bytes, value)
        self.samples += 1

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._sample_once()
            except Exception as exc:  # pragma: no cover - platform telemetry
                self.errors.append(f"{type(exc).__name__}: {exc}")

    def __enter__(self):
        self.start_bytes = self.current_bytes()
        self.peak_bytes = self.start_bytes
        self.samples = 1
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample_once()
        self.end_bytes = self.current_bytes()
        self.peak_bytes = max(self.peak_bytes, self.end_bytes)
        if self.errors or self.samples < 2:
            raise RuntimeError(f"process-tree RSS sampling failed: {self.errors}")
        return False


def _core_metadata(model: Any) -> dict[str, Any]:
    members = list(getattr(model, "estimators_", None) or [model])
    rows = []
    for member in members:
        core = member.model_
        trees = list(getattr(core, "trees_", ()))
        rows.append(
            {
                "tree_count": len(trees),
                "thread_count": int(getattr(core, "n_threads_", 0)),
                "learning_rate": float(getattr(core, "lr_", np.nan)),
            }
        )
    return {
        "member_count": len(rows),
        "total_tree_count": sum(row["tree_count"] for row in rows),
        "members": rows,
    }


def _archive(model: Any, arm: str, reference_X: Any) -> dict[str, Any]:
    reference = _valid_prediction(model.predict(reference_X), len(reference_X))
    if arm in {DARKO_SINGLE, DARKO_V3}:
        with tempfile.TemporaryDirectory(prefix="darkofit-v3-characterization-") as tmp:
            path = Path(tmp) / "model.npz"
            model.save_model(path)
            archive_bytes = path.stat().st_size
            restored = model.__class__.load_model(path)
            restored_prediction = _valid_prediction(
                restored.predict(reference_X), len(reference_X)
            )
        if not np.array_equal(reference, restored_prediction):
            raise RuntimeError("safe-NPZ prediction roundtrip is not exact")
        return {
            "format": "darkofit_safe_npz",
            "bytes": int(archive_bytes),
            "roundtrip_exact": True,
        }
    return {
        "format": "python_pickle_telemetry",
        "bytes": len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)),
        "roundtrip_exact": None,
    }


def _timed_prediction(model: Any, X: Any) -> dict[str, Any]:
    rows = len(X)
    started = time.perf_counter_ns()
    warm = _valid_prediction(model.predict(X), rows)
    warm_seconds = (time.perf_counter_ns() - started) / 1e9
    calls = int(
        min(
            MAX_CALLS,
            max(MIN_CALLS, math.ceil(TARGET_INTERVAL_SECONDS / max(warm_seconds, 1e-9))),
        )
    )
    final = None
    gc.disable()
    started = time.perf_counter_ns()
    try:
        for _ in range(calls):
            final = _valid_prediction(model.predict(X), rows)
    finally:
        interval_seconds = (time.perf_counter_ns() - started) / 1e9
        gc.enable()
    if final is None or not np.array_equal(warm, final):
        raise RuntimeError("integrated prediction changed output")
    return {
        "rows": int(rows),
        "calls": calls,
        "warm_seconds": float(warm_seconds),
        "interval_seconds": float(interval_seconds),
        "minimum_interval_met": bool(interval_seconds >= MIN_INTERVAL_SECONDS),
        "seconds_per_call": float(interval_seconds / calls),
        "rows_per_second": float(rows * calls / interval_seconds),
        "prediction_sha256": array_sha256(final),
        "method": "predict",
    }


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    runtime_before = _assert_worker_contract()
    contract = load_contract(Path(args.contract))
    del contract
    spec = _case_spec(args.case_id)
    data = m3b.build_case(spec)
    _warmup(args.arm, spec, data, args.darkofit_source, args.chimeraboost_source)
    model, expected_source = _build_model(
        args.arm, spec, args.darkofit_source, args.chimeraboost_source
    )
    implementation = _implementation(model, expected_source)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with ProcessTreeRSSSampler() as rss:
            started = time.perf_counter_ns()
            _fit(model, args.arm, data)
            fit_seconds = (time.perf_counter_ns() - started) / 1e9
        archive = _archive(model, args.arm, data["X_test"])
        predictions = {}
        for rows in BATCH_SIZES:
            X = _repeat_rows(data["X_test"], rows)
            input_hash = array_sha256(X)
            timing = _timed_prediction(model, X)
            timing["input_sha256"] = input_hash
            predictions[str(rows)] = timing
            del X
            gc.collect()
    metadata = _core_metadata(model)
    if any(row["thread_count"] != THREADS for row in metadata["members"]):
        raise RuntimeError("fitted thread count differs from the contract")
    return {
        "case_id": args.case_id,
        "task": spec["task"],
        "arm": args.arm,
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
        "archive": archive,
        "predictions": predictions,
        "model": metadata,
        "implementation": implementation,
        "warnings": [
            {"category": row.category.__name__, "message": str(row.message)}
            for row in caught
        ],
        "runtime_before": runtime_before,
        "runtime_after": _assert_worker_contract(),
    }


def _worker_environment(
    darkofit_source: Path, chimeraboost_source: Path
) -> dict[str, str]:
    environment = os.environ.copy()
    prefixes = (
        "NUMBA_",
        "OMP_",
        "KMP_",
        "MKL_",
        "OPENBLAS_",
        "VECLIB_",
        "NUMEXPR_",
    )
    for key in tuple(environment):
        if key.startswith(prefixes):
            environment.pop(key)
    for key in THREAD_ENV_KEYS:
        environment[key] = str(THREADS)
    environment.update(
        {
            "DARKOFIT_WARMUP": "0",
            "CHIMERABOOST_WARMUP": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": os.pathsep.join(
                (
                    str(darkofit_source.resolve()),
                    str(chimeraboost_source.resolve()),
                    str(ROOT),
                )
            ),
            "NUMBA_CACHE_DIR": "/private/tmp/darkofit-v3-characterization-numba",
            "JOBLIB_TEMP_FOLDER": "/private/tmp/darkofit-v3-characterization-joblib",
            "LOKY_MAX_CPU_COUNT": str(THREADS),
            "NUMBA_DISABLE_JIT": "0",
            "NUMBA_NUM_THREADS": str(THREADS),
            "NUMBA_THREADING_LAYER": "default",
            "OMP_DYNAMIC": "FALSE",
            "OMP_THREAD_LIMIT": str(THREADS),
            "MKL_DYNAMIC": "FALSE",
        }
    )
    return environment


def _assert_worker_contract() -> dict[str, Any]:
    expected = {key: str(THREADS) for key in THREAD_ENV_KEYS}
    expected.update(
        {
            "NUMBA_DISABLE_JIT": "0",
            "NUMBA_NUM_THREADS": str(THREADS),
            "NUMBA_THREADING_LAYER": "default",
            "OMP_DYNAMIC": "FALSE",
            "OMP_THREAD_LIMIT": str(THREADS),
            "MKL_DYNAMIC": "FALSE",
            "DARKOFIT_WARMUP": "0",
            "CHIMERABOOST_WARMUP": "0",
            "PYTHONHASHSEED": "0",
        }
    )
    actual = {key: os.environ.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"characterization worker environment drifted: {actual}")
    import numba

    ceiling = int(numba.config.NUMBA_NUM_THREADS)
    current = int(numba.get_num_threads())
    if ceiling != THREADS or current != THREADS:
        raise RuntimeError(
            f"Numba thread contract drifted: ceiling={ceiling}, current={current}"
        )
    return {
        "environment": actual,
        "numba_thread_ceiling": ceiling,
        "numba_current_threads": current,
        "numba_threading_layer": numba.threading_layer(),
    }


def _run_worker(args: argparse.Namespace, case_id: str, arm: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--contract",
        str(args.contract),
        "--darkofit-source",
        str(args.darkofit_source),
        "--chimeraboost-source",
        str(args.chimeraboost_source),
        "--case-id",
        case_id,
        "--arm",
        arm,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_worker_environment(args.darkofit_source, args.chimeraboost_source),
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if completed.returncode or len(lines) != 1:
        raise RuntimeError(
            f"worker failed for {case_id}/{arm} ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    result = json.loads(lines[0][len(WORKER_PREFIX) :])
    result["worker_stdout"] = "\n".join(
        line for line in completed.stdout.splitlines() if not line.startswith(WORKER_PREFIX)
    ).strip() or None
    result["worker_stderr"] = completed.stderr.strip() or None
    return result


def _require_clean_sources(args: argparse.Namespace) -> dict[str, Any]:
    states = {
        "harness": source_state(ROOT),
        "darkofit": source_state(args.darkofit_source),
        "chimeraboost": source_state(args.chimeraboost_source),
    }
    if not all(value["clean"] for value in states.values()):
        raise RuntimeError("characterization requires three clean checkouts")
    if states["darkofit"]["head"] != DARKOFIT_HEAD:
        raise RuntimeError("DarkoFit source is not the published candidate pin")
    if states["chimeraboost"]["head"] != CHIMERABOOST_HEAD:
        raise RuntimeError("ChimeraBoost source is not the pinned 0.18 comparator")
    return states


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"characterization artifact is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json_bytes(payload))


def _require_external_outputs(args: argparse.Namespace) -> None:
    roots = (ROOT, args.darkofit_source, args.chimeraboost_source)
    for path in (args.output, args.terminal):
        for root in roots:
            if path.is_relative_to(root):
                raise ValueError(
                    f"characterization outputs must be outside all checkouts: {path}"
                )


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    _require_external_outputs(args)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"raw output is create-only: {args.output}")
    if args.terminal.exists() or args.terminal.is_symlink():
        raise FileExistsError(f"terminal output is create-only: {args.terminal}")
    contract = load_contract(args.contract)
    states = _require_clean_sources(args)
    rows = []
    try:
        for block in range(BLOCKS):
            for case_id in CASES:
                for position, arm in enumerate(order_for(case_id, block)):
                    if _require_clean_sources(args) != states:
                        raise RuntimeError("source state changed during characterization")
                    print(
                        f"block={block + 1}/{BLOCKS} case={case_id} "
                        f"position={position + 1}/{len(ARMS)} arm={arm}",
                        flush=True,
                    )
                    row = _run_worker(args, case_id, arm)
                    row.update(block=block, position=position)
                    rows.append(row)
        if _require_clean_sources(args) != states:
            raise RuntimeError("source state changed during characterization")
    except BaseException as exc:
        terminal = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "terminal_failure",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_rows_discarded": len(rows),
            "rerun_authorized": False,
            "contract_sha256": sha256(args.contract),
            "sources": states,
        }
        _write_create_only(args.terminal, terminal)
        raise
    artifact = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "evidence_scope": "tier_e_spent_general_characterization",
        "contract": {
            "path": str(args.contract.resolve()),
            "sha256": sha256(args.contract),
            "bindings": contract["bindings"],
        },
        "sources": states,
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "python": sys.version,
            "executable": sys.executable,
            "cpu_count": os.cpu_count(),
        },
        "execution": execution_spec(),
        "rows": rows,
        "shipping_or_default_claim_authorized": False,
        "m2_or_m4": False,
        "fresh_or_lockbox_data_used": False,
    }
    _write_create_only(args.output, artifact)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--terminal", type=Path, default=DEFAULT_TERMINAL)
    parser.add_argument("--darkofit-source", type=Path, required=True)
    parser.add_argument("--chimeraboost-source", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-id", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ("contract", "output", "terminal", "darkofit_source", "chimeraboost_source"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.worker and (args.case_id is None or args.arm is None):
        parser.error("worker mode requires --case-id and --arm")
    if not args.worker and (args.case_id is not None or args.arm is not None):
        parser.error("--case-id/--arm are private worker arguments")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        result = run_worker(args)
        print(WORKER_PREFIX + json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
