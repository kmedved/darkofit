#!/usr/bin/env python3
"""Run the frozen basketball leafwise packed-prediction confirmation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numba
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_TEXT = str(REPO_ROOT)
if REPO_ROOT_TEXT in sys.path:
    sys.path.remove(REPO_ROOT_TEXT)
sys.path.insert(0, REPO_ROOT_TEXT)

import darkofit as darkofit_package  # noqa: E402
import darkofit.booster as booster_module  # noqa: E402
import darkofit.flat_model as flat_module  # noqa: E402
from benchmarks import basketball_harness as harness  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402
from darkofit import DarkoRegressor  # noqa: E402
from darkofit.flat_model import FlatNonObliviousEnsemble  # noqa: E402
from darkofit.sklearn_api import _check_predict_input  # noqa: E402


PRE_PROTOCOL_COMMIT = "96413f2c71faf4fd4b2caf05c411c661e5958f21"
EXPECTED_DARKOFIT_TREE = "1b93ea2f52bb563e81cefc9102f4a5b0ad29308b"
PROTOCOL_PATH = (
    REPO_ROOT / "benchmarks" / "basketball_leafwise_packed_prediction_protocol.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "73184115ccf9c208a032df70166f974e51ac5a60d0c8355b1ccf07785fcb1d14"
)
EXPECTED_NORMALIZED_RUNNER_SHA256 = (
    "7d2abc94d80658bb4ac399e5fd0872f1b200185e608715d1aa19a6f5859e8237"
)
EXPECTED_ORACLE_FUNCTION_SHA256 = (
    "9b7f61f8ef16be9ec822117afe5cf77b0f2f4b8d7fe0698d07e707862943b787"
)
EXPECTED_SUPPORT_SHA256 = {
    "benchmarks/basketball_harness.py": (
        "40360ca54d1954d540bd72bec47c891a179fb0f764a0cb6061d3a33b8593aba1"
    ),
    "benchmarks/basketball_guardrails.py": (
        "4492a65161b2261e5d437b9733c286368534c54f7fd23b6f6b95a804507fff52"
    ),
    "benchmarks/run_basketball_creator_benchmark.py": (
        "9a2d6b3dc8f3a8586cd4ac20bdb3607c17675cab1d89e6bbd7c438d2bba53fec"
    ),
}
EXPECTED_GOLDEN_MANIFEST_SHA256 = (
    "2443509cded5e8ec3a725b50ce9fcc403be4c3ad9339260a9ccb85dec3ca17ff"
)
EXPECTED_RAW_SHA256 = "43af3be46074da9130a757aa1db643e207e2c0ea5405d2086d698f85555fbcb2"
EXPECTED_X_SHA256 = "05efa554a983942885b72b1b3fdcc97a10ecf4ccbb4b10843ee8b970129fe71b"
EXPECTED_Y_SHA256 = "7c06b45d4649c392fdb0c3dc91a857650f2f3fc77341fe06ffdbee2b3a44aecf"
EXPECTED_FOLD_SHA256 = (
    "7270350a38a687b6e783d18d9c02e5a45f2c7367faa4a6955e74c710f5e8daea"
)
EXPECTED_COLD_MASK_SHA256 = (
    "e17887c9240fd95aee21d37d7e1e8a533c38ef87f4492acd28fb3aa496a3dd19"
)
EXPECTED_STATE_SHA256 = (
    "b588ddf2e09857479421bd490d394f7667d29e998b5d931dcff089455672604e"
)
EXPECTED_ARCHIVE_SHA256 = (
    "eb16c2f24f884f9661debd029897e3e7b1403d9e4189d86ff4f2c7ac4aeaf5bc"
)
EXPECTED_ARCHIVE_BYTES = 606056
EXPECTED_TRAIN_PREDICTION_SHA256 = (
    "1544903cc4f52b361dc21327f43f848e9471337c88e44b5d2578f11b9bf515d1"
)
EXPECTED_PACKED_STORAGE_BYTES = 2146616
EXPECTED_PACKED_STORAGE_SHA256 = (
    "8c43bdcd713e86b3d9fa76ec2d908b7f8ce0669797e64c652c4651b825808135"
)
EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "platform": "macOS-26.5.2-arm64-arm-64bit",
    "machine": "arm64",
    "cpu_brand": "Apple M5 Max",
    "logical_cpu_count": 18,
    "dependencies": {
        "numpy": "2.4.6",
        "numba": "0.66.0",
        "llvmlite": "0.48.0",
        "pandas": "3.0.3",
        "scikit-learn": "1.9.0",
        "scipy": "1.18.0",
        "joblib": "1.5.3",
        "threadpoolctl": "3.6.0",
        "pytest": "9.1.1",
        "iniconfig": "2.3.0",
        "packaging": "26.2",
        "pluggy": "1.6.0",
        "Pygments": "2.20.0",
        "optuna": "4.9.0",
        "alembic": "1.18.5",
        "colorlog": "6.10.1",
        "Mako": "1.3.12",
        "MarkupSafe": "3.0.3",
        "PyYAML": "6.0.3",
        "SQLAlchemy": "2.0.51",
        "tqdm": "4.68.4",
        "typing_extensions": "4.16.0",
    },
}
DEFAULT_OUTPUT = (
    REPO_ROOT / "benchmarks" / "basketball_leafwise_packed_prediction.json"
)
TIMING_BLOCKS = 11
MEMORY_CALLS = 5
MAX_IQR_FRACTION = 0.20
MAX_EXTRA_TRACED_BYTES = 256 * 1024
EXPECTED_RESOLVED_THREADS = 2
EXPECTED_TREES = 1000
CONFIRMATION_FOLD = 1

MODEL_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.1,
    "depth": 6,
    "l2_leaf_reg": 1,
    "max_bins": 128,
    "tree_mode": "lightgbm",
    "ordered_boosting": False,
    "use_best_model": False,
    "early_stopping": False,
    "thread_count": 18,
    "random_state": 4,
    "diagnostic_warnings": "never",
}
BOUNDARY_CASES = (
    (1, 32768, False),
    (5, 8192, True),
    (16, 2409, True),
    (25, 525, False),
    (62, 525, False),
    (63, 525, True),
    (258, 127, False),
    (259, 127, True),
    (1000, 32768, True),
    (1000, 65536, False),
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_array(values: Any, dtype: str | None = None) -> str:
    array = np.asarray(values, dtype=dtype)
    return _sha256_bytes(np.ascontiguousarray(array).tobytes())


def _normalized_runner_sha256() -> str:
    payload = Path(__file__).resolve().read_bytes()
    for field in (
        b'EXPECTED_NORMALIZED_RUNNER_SHA256 = (\n    "',
        b'EXPECTED_ORACLE_FUNCTION_SHA256 = (\n    "',
    ):
        start = payload.find(field)
        if start < 0:
            raise RuntimeError("runner self-hash marker is missing")
        value_start = start + len(field)
        value_end = value_start + 64
        if payload[value_end : value_end + 2] != b'"\n':
            raise RuntimeError("runner self-hash field is malformed")
        payload = payload[:value_start] + (b"0" * 64) + payload[value_end:]
    return _sha256_bytes(payload)


def _support_sha256() -> dict[str, str]:
    return {
        relative: _sha256_bytes((REPO_ROOT / relative).read_bytes())
        for relative in EXPECTED_SUPPORT_SHA256
    }


def _git(*args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def _runtime_state() -> dict[str, Any]:
    machine = creator._machine_details()
    dependencies = {
        package: importlib.metadata.version(package)
        for package in EXPECTED_RUNTIME["dependencies"]
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_brand": machine["cpu_brand"],
        "logical_cpu_count": os.cpu_count(),
        "dependencies": dependencies,
        "python_executable": sys.executable,
        "numba_max_threads": int(numba.config.NUMBA_NUM_THREADS),
    }


def canonical_fitted_state_sha256(model):
    array_fields = (
        "features", "thresholds", "left_child", "right_child",
        "leaf_index", "values", "splits_feat", "splits_thr", "gains",
    )

    def add_array(digest, name, value):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
        digest.update(array.tobytes(order="C"))

    core = model.model_
    digest = hashlib.sha256()
    metadata = {
        "tree_count": len(core.trees_),
        "tree_types": [type(tree).__name__ for tree in core.trees_],
        "best_iteration": int(core.best_iteration_),
        "learning_rate": float(core.lr_),
        "stop_reason": str(core.stop_reason_),
        "tree_mode": str(core.tree_mode_),
        "ordered_boosting": bool(core.ordered_boosting_),
        "threads": int(core.n_threads_),
        "depth": int(core.depth),
        "l2_leaf_reg": float(core.l2_leaf_reg),
        "max_bins": int(core.max_bins),
    }
    digest.update(
        json.dumps(
            metadata, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    add_array(digest, "feature_importances", core.feature_importances_)
    for index, tree in enumerate(core.trees_):
        digest.update(
            f"tree:{index}:{tree.depth}:{tree.n_leaves}:"
            f"{tree.n_splits}".encode("utf-8")
        )
        for field in array_fields:
            add_array(digest, field, getattr(tree, field))
    return digest.hexdigest()


def _validate_binding(output: Path) -> dict[str, Any]:
    if output != DEFAULT_OUTPUT:
        raise RuntimeError("formal output path is not exact")
    if output.is_symlink() or output.exists():
        raise RuntimeError(f"refusing existing benchmark output: {output}")
    if _sha256_bytes(PROTOCOL_PATH.read_bytes()) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("leafwise packed-prediction protocol changed")
    if _normalized_runner_sha256() != EXPECTED_NORMALIZED_RUNNER_SHA256:
        raise RuntimeError("leafwise packed-prediction runner changed")
    oracle_source = inspect.getsource(canonical_fitted_state_sha256).encode()
    oracle_hash = _sha256_bytes(oracle_source)
    if oracle_hash != EXPECTED_ORACLE_FUNCTION_SHA256:
        raise RuntimeError("canonical fitted-state oracle changed")
    support = _support_sha256()
    if support != EXPECTED_SUPPORT_SHA256:
        raise RuntimeError("basketball support files changed")
    golden_manifest_hash = _sha256_bytes(
        (REPO_ROOT / "tests" / "golden_predictions.json").read_bytes()
    )
    if golden_manifest_hash != EXPECTED_GOLDEN_MANIFEST_SHA256:
        raise RuntimeError("Phase 0 prediction-golden manifest changed")
    imported = Path(darkofit_package.__file__).resolve()
    if not imported.is_relative_to(REPO_ROOT):
        raise RuntimeError("DarkoFit was imported outside the frozen repository")
    if _git("rev-parse", "HEAD:darkofit") != EXPECTED_DARKOFIT_TREE:
        raise RuntimeError("DarkoFit package tree changed")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PRE_PROTOCOL_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode:
        raise RuntimeError("source no longer descends from the protocol base")
    source = creator.git_state(REPO_ROOT)
    if not source["clean"]:
        raise RuntimeError("formal run requires clean committed source")
    if (
        source["branch"] != "main"
        or source["tracked_main_refs"].get("origin/main") != source["head"]
    ):
        raise RuntimeError("formal run requires main == origin/main")
    runtime = _runtime_state()
    for key in (
        "python", "platform", "machine", "cpu_brand", "logical_cpu_count"
    ):
        if runtime[key] != EXPECTED_RUNTIME[key]:
            raise RuntimeError(f"frozen runtime mismatch: {key}")
    if runtime["dependencies"] != EXPECTED_RUNTIME["dependencies"]:
        raise RuntimeError("frozen dependency versions changed")
    if runtime["numba_max_threads"] != 18:
        raise RuntimeError("frozen Numba thread ceiling changed")
    return {
        "source": source,
        "runtime": runtime,
        "runner_normalized_sha256": _normalized_runner_sha256(),
        "oracle_function_sha256": oracle_hash,
        "support_sha256": support,
        "golden_manifest_sha256": golden_manifest_hash,
        "imported_darkofit_path": str(imported),
    }


def _validate_data(dataset: Any) -> None:
    if dataset.raw_metadata.get("sha256") != EXPECTED_RAW_SHA256:
        raise RuntimeError("basketball CSV changed")
    if dataset.processed_metadata.get("x_train_sha256") != EXPECTED_X_SHA256:
        raise RuntimeError("basketball creator features changed")
    if dataset.processed_metadata.get("y_train_sha256") != EXPECTED_Y_SHA256:
        raise RuntimeError("basketball creator target changed")
    if dataset.fold_fingerprint_sha256 != EXPECTED_FOLD_SHA256:
        raise RuntimeError("basketball creator folds changed")
    if (
        dataset.player_guardrail.metadata.get("cold_player_mask_sha256")
        != EXPECTED_COLD_MASK_SHA256
    ):
        raise RuntimeError("basketball cold-player mask changed")


def _repeat_frame(frame: Any, rows: int):
    indices = np.arange(int(rows), dtype=np.int64) % len(frame)
    return frame.iloc[indices].reset_index(drop=True)


def _repeat_array(array: np.ndarray, rows: int) -> np.ndarray:
    indices = np.arange(int(rows), dtype=np.int64) % len(array)
    return np.ascontiguousarray(array[indices])


def _inner_calls(rows: int) -> int:
    return max(1, min(64, math.ceil(65536 / int(rows))))


def _timing_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    iqr = float(np.subtract(*np.percentile(array, [75, 25])))
    return {
        "seconds_per_call": [float(value) for value in array],
        "minimum_seconds": float(array.min()),
        "maximum_seconds": float(array.max()),
        "median_seconds": median,
        "p50_seconds": float(np.percentile(array, 50)),
        "p95_seconds": float(np.percentile(array, 95)),
        "p99_seconds": float(np.percentile(array, 99)),
        "iqr_seconds": iqr,
        "iqr_fraction": iqr / median if median > 0.0 else float("inf"),
    }


def _time_pair(
    candidate: Callable[[], np.ndarray],
    reference: Callable[[], np.ndarray],
    expected: np.ndarray,
    inner_calls: int,
) -> dict[str, Any]:
    functions = {"candidate": candidate, "reference": reference}
    durations = {name: [] for name in functions}
    blocks = []
    for block in range(TIMING_BLOCKS):
        order = (
            ("candidate", "reference")
            if block % 2 == 0
            else ("reference", "candidate")
        )
        record = {"block": block, "order": list(order), "arms": {}}
        for name in order:
            gc.disable()
            started = time.perf_counter_ns()
            try:
                observed = None
                for _ in range(inner_calls):
                    observed = functions[name]()
            finally:
                elapsed = (time.perf_counter_ns() - started) / 1e9
                gc.enable()
            if observed is None or not np.array_equal(observed, expected):
                raise RuntimeError("timed prediction differs from reference")
            per_call = elapsed / inner_calls
            durations[name].append(per_call)
            record["arms"][name] = {
                "total_seconds": float(elapsed),
                "per_call_seconds": float(per_call),
            }
        blocks.append(record)
    summaries = {
        name: _timing_summary(values) for name, values in durations.items()
    }
    return {
        "inner_calls": int(inner_calls),
        "blocks": blocks,
        "summaries": summaries,
        "candidate_over_reference": float(
            summaries["candidate"]["median_seconds"]
            / summaries["reference"]["median_seconds"]
        ),
    }


def _trace_pair(
    candidate: Callable[[], np.ndarray],
    reference: Callable[[], np.ndarray],
    expected: np.ndarray,
) -> dict[str, Any]:
    result = {}
    for name, function in (("candidate", candidate), ("reference", reference)):
        peaks = []
        for _ in range(MEMORY_CALLS):
            gc.collect()
            tracemalloc.start()
            try:
                observed = function()
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            if not np.array_equal(observed, expected):
                raise RuntimeError("traced prediction differs from reference")
            peaks.append(int(peak))
        result[name] = {
            "calls": MEMORY_CALLS,
            "peak_bytes_per_call": peaks,
            "maximum_peak_bytes": max(peaks),
        }
    result["candidate_extra_peak_bytes"] = int(
        result["candidate"]["maximum_peak_bytes"]
        - result["reference"]["maximum_peak_bytes"]
    )
    return result


def _loop_from_binned(core: Any, X_binned: np.ndarray) -> np.ndarray:
    output = np.full(len(X_binned), core.init_, dtype=np.float64)
    for tree in core.trees_:
        tree.add_predict(X_binned, output)
    return output


def _reference_public(model: Any, X: Any) -> np.ndarray:
    core = model.model_
    validated = _check_predict_input(model, X)
    model._check_fitted_loss_matches_params("predict")
    trend = model._linear_residual_trend(validated)
    X_binned = core.prep_.transform(
        core._prepare_predict_X(validated, validated=True)
    )
    raw = _loop_from_binned(core, X_binned)
    if model._fitted_distributional():
        raise RuntimeError("frozen scalar-RMSE reference became distributional")
    return raw if trend is None else raw + trend


def _force_tree_loop(*_args, **_kwargs) -> bool:
    return False


def _public_predict_route(model: Any, X: Any, *, packed: bool) -> np.ndarray:
    original_selector = booster_module.flat_predict_preferred
    booster_module.flat_predict_preferred = (
        original_selector if packed else _force_tree_loop
    )
    try:
        return np.asarray(model.predict(X), dtype=np.float64)
    finally:
        booster_module.flat_predict_preferred = original_selector


def _candidate_core(
    flat: FlatNonObliviousEnsemble,
    X_binned: np.ndarray,
    initial: float,
) -> np.ndarray:
    output = np.full(len(X_binned), initial, dtype=np.float64)
    flat.add_predict_scalar_packed(X_binned, output)
    return output


def _observe_public_route(model: Any, X: Any) -> tuple[dict[str, Any], np.ndarray]:
    original_selector = booster_module.flat_predict_preferred
    original_kernel = flat_module._flat_nonoblivious_scalar_add_parallel
    observations = {"outer": [], "scalar_kernel_calls": 0}

    def observed_selector(flat, n_rows=None, tree_mode=None):
        selected = original_selector(flat, n_rows, tree_mode)
        observations["outer"].append(
            {
                "rows": int(n_rows),
                "tree_mode": str(tree_mode),
                "selected": bool(selected),
            }
        )
        return selected

    def observed_kernel(*args):
        observations["scalar_kernel_calls"] += 1
        return original_kernel(*args)

    booster_module.flat_predict_preferred = observed_selector
    flat_module._flat_nonoblivious_scalar_add_parallel = observed_kernel
    try:
        prediction = np.asarray(model.predict(X), dtype=np.float64)
    finally:
        booster_module.flat_predict_preferred = original_selector
        flat_module._flat_nonoblivious_scalar_add_parallel = original_kernel
    if len(observations["outer"]) != 1:
        raise RuntimeError(f"unexpected outer route observations: {observations}")
    return observations, prediction


def _last_staged(core: Any, X: Any) -> np.ndarray:
    last = None
    for last in core.staged_predict_raw(X):
        pass
    if last is None:
        raise RuntimeError("fitted forest produced no staged prediction")
    return last


def _packed_storage(flat: FlatNonObliviousEnsemble) -> dict[str, Any]:
    arrays = {
        slot: getattr(flat, slot)
        for slot in flat.__slots__
        if isinstance(getattr(flat, slot), np.ndarray)
    }
    return {
        "bytes": int(sum(value.nbytes for value in arrays.values())),
        "arrays": {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": _sha256_array(value),
            }
            for name, value in arrays.items()
        },
    }


def _packed_storage_sha256(storage: dict[str, Any]) -> str:
    payload = json.dumps(
        storage, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return _sha256_bytes(payload)


def _case_result(name: str, X: Any, model: Any) -> dict[str, Any]:
    core = model.model_
    flat = core._flat_ensemble()
    X_binned = core.prep_.transform(
        core._prepare_predict_X(_check_predict_input(model, X), validated=True)
    )
    reference = _loop_from_binned(core, X_binned)
    route, public_candidate = _observe_public_route(model, X)
    core_candidate = _candidate_core(flat, X_binned, core.init_)
    public_reference = _reference_public(model, X)
    staged = _last_staged(core, X)
    exactness = {
        "public_candidate_vs_loop": np.array_equal(public_candidate, reference),
        "public_reference_vs_loop": np.array_equal(public_reference, reference),
        "packed_core_vs_loop": np.array_equal(core_candidate, reference),
        "staged_final_vs_public": np.array_equal(staged, public_candidate),
    }
    if not all(exactness.values()):
        raise RuntimeError(f"{name}: prediction exactness failed: {exactness}")
    expected_selected = len(X) <= 32768
    if route["outer"][0]["selected"] != expected_selected:
        raise RuntimeError(f"{name}: unexpected outer route: {route}")
    expected_kernel_calls = 1 if expected_selected else 0
    if route["scalar_kernel_calls"] != expected_kernel_calls:
        raise RuntimeError(f"{name}: unexpected scalar-kernel route: {route}")

    core_functions = {
        "candidate": lambda: _candidate_core(flat, X_binned, core.init_),
        "reference": lambda: _loop_from_binned(core, X_binned),
    }
    public_functions = {
        "candidate": lambda: _public_predict_route(model, X, packed=True),
        "reference": lambda: _public_predict_route(model, X, packed=False),
    }
    for functions in (core_functions, public_functions):
        for function in functions.values():
            if not np.array_equal(function(), reference):
                raise RuntimeError(f"{name}: warmup output changed")
    calls = _inner_calls(len(X))
    core_timing = _time_pair(
        core_functions["candidate"],
        core_functions["reference"],
        reference,
        calls,
    )
    public_timing = _time_pair(
        public_functions["candidate"],
        public_functions["reference"],
        reference,
        calls,
    )
    memory = _trace_pair(
        public_functions["candidate"],
        public_functions["reference"],
        reference,
    )
    return {
        "name": name,
        "rows": int(len(X)),
        "columns": int(X.shape[1]),
        "binned_sha256": _sha256_array(X_binned),
        "prediction_sha256": _sha256_array(reference, "<f8"),
        "route": route,
        "exactness": exactness,
        "core_timing": core_timing,
        "public_timing": public_timing,
        "memory": memory,
    }


def _boundary_results(core: Any, seed_binned: np.ndarray) -> list[dict[str, Any]]:
    records = []
    original_selector = flat_module.flat_predict_preferred
    original_kernel = flat_module._flat_nonoblivious_scalar_add_parallel
    for tree_count, rows, expected_selected in BOUNDARY_CASES:
        flat = FlatNonObliviousEnsemble(core.trees_[:tree_count])
        X_binned = _repeat_array(seed_binned, rows)
        selected_calls = []
        kernel_calls = [0]

        def observed_selector(candidate, n_rows=None, tree_mode=None):
            selected = original_selector(candidate, n_rows, tree_mode)
            selected_calls.append(bool(selected))
            return selected

        def observed_kernel(*args):
            kernel_calls[0] += 1
            return original_kernel(*args)

        flat_module.flat_predict_preferred = observed_selector
        flat_module._flat_nonoblivious_scalar_add_parallel = observed_kernel
        try:
            output = np.full(rows, core.init_, dtype=np.float64)
            if flat_module.flat_predict_preferred(flat, rows, "lightgbm"):
                flat.add_predict_scalar_packed(X_binned, output)
            else:
                for tree in core.trees_[:tree_count]:
                    tree.add_predict(X_binned, output)
        finally:
            flat_module.flat_predict_preferred = original_selector
            flat_module._flat_nonoblivious_scalar_add_parallel = original_kernel
        reference = np.full(rows, core.init_, dtype=np.float64)
        for tree in core.trees_[:tree_count]:
            tree.add_predict(X_binned, reference)
        observed_selected = selected_calls == [expected_selected]
        observed_kernel = kernel_calls[0] == (1 if expected_selected else 0)
        exact = np.array_equal(output, reference)
        if not (observed_selected and observed_kernel and exact):
            raise RuntimeError(
                f"boundary route failed for trees={tree_count}, rows={rows}"
            )
        records.append(
            {
                "trees": tree_count,
                "rows": rows,
                "expected_selected": expected_selected,
                "observed_selected": selected_calls[0],
                "scalar_kernel_calls": kernel_calls[0],
                "prediction_array_exact": exact,
                "prediction_sha256": _sha256_array(reference, "<f8"),
            }
        )
    return records


def _model_metadata(model: Any, fit_seconds: float) -> dict[str, Any]:
    core = model.model_
    result = {
        "requested_params": MODEL_PARAMS,
        "fit_seconds_directional": float(fit_seconds),
        "tree_count": len(core.trees_),
        "tree_types": sorted({type(tree).__name__ for tree in core.trees_}),
        "best_iteration": int(core.best_iteration_),
        "learning_rate": float(core.lr_),
        "stop_reason": str(core.stop_reason_),
        "tree_mode": str(core.tree_mode_),
        "ordered_boosting": bool(core.ordered_boosting_),
        "resolved_threads": int(core.n_threads_),
        "selection_model_present": getattr(model, "selection_model_", None) is not None,
        "refit": bool(getattr(model, "refit_", False)),
        "linear_residual_active": bool(
            getattr(model, "linear_residual_active_", False)
        ),
        "linear_leaves_active": bool(
            getattr(core, "auto_params_", {})
            .get("linear_leaves", {})
            .get("active", False)
        ),
    }
    expected = {
        "tree_count": EXPECTED_TREES,
        "tree_types": ["NonObliviousTree"],
        "best_iteration": EXPECTED_TREES,
        "learning_rate": 0.1,
        "tree_mode": "lightgbm",
        "ordered_boosting": False,
        "resolved_threads": EXPECTED_RESOLVED_THREADS,
        "selection_model_present": False,
        "refit": False,
        "linear_residual_active": False,
        "linear_leaves_active": False,
    }
    result["frozen_configuration_exact"] = all(
        result[key] == value for key, value in expected.items()
    )
    if not result["frozen_configuration_exact"]:
        raise RuntimeError(f"fitted configuration changed: {result}")
    return result


def _numeric_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_numeric_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_numeric_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _analyze(
    cases: dict[str, dict[str, Any]],
    boundaries: list[dict[str, Any]],
    model_record: dict[str, Any],
    invariants: dict[str, Any],
) -> dict[str, Any]:
    core_080 = (
        "tiny_127",
        "fold_1_524",
        "cold_player_585",
        "held_team_2409",
        "repeated_8192",
    )
    public_090 = ("fold_1_524", "cold_player_585", "held_team_2409")
    fallbacks = ("fallback_65536", "fallback_100000")
    stability_pairs = (
        *(("core_timing", name) for name in (*core_080, "repeated_32768")),
        *(("public_timing", name) for name in (*public_090, *fallbacks)),
    )
    gates = {
        "frozen_model_configuration": model_record["frozen_configuration_exact"],
        "prechange_fitted_state": invariants["state_sha256_exact"],
        "prechange_archive": invariants["archive_sha256_exact"]
        and invariants["archive_length_exact"],
        "prechange_train_prediction": invariants["train_prediction_sha256_exact"],
        "prechange_packed_storage": invariants[
            "prechange_packed_storage_exact"
        ],
        "predictions_array_exact": all(
            all(case["exactness"].values()) for case in cases.values()
        ),
        "routes_observed_exact": all(
            case["route"]["outer"][0]["selected"] == (case["rows"] <= 32768)
            and case["route"]["scalar_kernel_calls"]
            == (1 if case["rows"] <= 32768 else 0)
            for case in cases.values()
        ),
        "boundary_table_exact": all(
            record["expected_selected"] == record["observed_selected"]
            and record["scalar_kernel_calls"]
            == (1 if record["expected_selected"] else 0)
            and record["prediction_array_exact"]
            for record in boundaries
        ),
        "core_ratio_at_most_080": all(
            cases[name]["core_timing"]["candidate_over_reference"] <= 0.80
            for name in core_080
        ),
        "core_32768_ratio_at_most_098": (
            cases["repeated_32768"]["core_timing"]["candidate_over_reference"]
            <= 0.98
        ),
        "public_ratio_at_most_090": all(
            cases[name]["public_timing"]["candidate_over_reference"] <= 0.90
            for name in public_090
        ),
        "fallback_ratio_at_most_110": all(
            cases[name]["public_timing"]["candidate_over_reference"] <= 1.10
            for name in fallbacks
        ),
        "timing_stability": all(
            cases[name][group]["summaries"][arm]["iqr_fraction"]
            <= MAX_IQR_FRACTION
            for group, name in stability_pairs
            for arm in ("candidate", "reference")
        ),
        "transient_memory": all(
            case["memory"]["candidate_extra_peak_bytes"]
            <= MAX_EXTRA_TRACED_BYTES
            for case in cases.values()
        ),
        "zero_persistent_bytes": invariants["persistent_extra_bytes"] == 0,
        "packed_cache_identity": invariants["packed_cache_identity"],
        "archive_unchanged_after_prediction": invariants[
            "archive_unchanged_after_prediction"
        ],
        "fitted_state_unchanged_after_prediction": invariants[
            "fitted_state_unchanged_after_prediction"
        ],
    }
    finite = _numeric_finite(
        {
            "cases": cases,
            "boundaries": boundaries,
            "model": model_record,
            "invariants": invariants,
        }
    )
    gates["all_numeric_values_finite"] = finite
    passed = all(gates.values())
    return {
        "gates": gates,
        "passes_all_gates": passed,
        "recommendation": (
            "retain_bounded_leafwise_packed_prediction"
            if passed
            else "revert_bounded_leafwise_packed_prediction_without_retuning"
        ),
        "default_policy_change_authorized": False,
        "ctr23_or_lockbox_used": False,
    }


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def run(output: Path, data_cache: Path) -> dict[str, Any]:
    binding = _validate_binding(output)
    source_head = binding["source"]["head"]
    dataset = harness.load_basketball_dataset(data_cache)
    _validate_data(dataset)

    model = DarkoRegressor(**MODEL_PARAMS)
    started = time.perf_counter_ns()
    model.fit(dataset.X, dataset.y)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    model_record = _model_metadata(model, fit_seconds)
    core = model.model_
    if numba.get_num_threads() != EXPECTED_RESOLVED_THREADS:
        raise RuntimeError("fitted model did not restore two Numba threads")

    with tempfile.TemporaryDirectory(prefix="darkofit-leafwise-") as directory:
        archive_before_path = Path(directory) / "before.npz"
        archive_after_path = Path(directory) / "after.npz"
        model.save_model(archive_before_path)
        archive_before = archive_before_path.read_bytes()
        state_before = canonical_fitted_state_sha256(model)
        train_prediction = np.asarray(model.predict(dataset.X), dtype=np.float64)

        splits = list(creator.creator_cv().split(dataset.X, dataset.y))
        _, fold_1_test = splits[CONFIRMATION_FOLD]
        fold_1 = dataset.X.iloc[np.asarray(fold_1_test, dtype=np.int64)]
        guardrail = dataset.player_guardrail
        cold = np.asarray(guardrail.cold_player_mask, dtype=bool)
        case_frames = {
            "tiny_127": fold_1.iloc[:127],
            "fold_1_524": fold_1,
            "cold_player_585": guardrail.X_holdout.iloc[np.flatnonzero(cold)],
            "held_team_2409": guardrail.X_holdout,
            "repeated_8192": _repeat_frame(fold_1, 8192),
            "repeated_32768": _repeat_frame(fold_1, 32768),
            "fallback_65536": _repeat_frame(fold_1, 65536),
            "fallback_100000": _repeat_frame(fold_1, 100000),
        }
        expected_rows = {
            "tiny_127": 127,
            "fold_1_524": 524,
            "cold_player_585": 585,
            "held_team_2409": 2409,
            "repeated_8192": 8192,
            "repeated_32768": 32768,
            "fallback_65536": 65536,
            "fallback_100000": 100000,
        }
        observed_rows = {name: len(frame) for name, frame in case_frames.items()}
        if observed_rows != expected_rows:
            raise RuntimeError(f"basketball case rows changed: {observed_rows}")

        flat = core._flat_ensemble()
        if not isinstance(flat, FlatNonObliviousEnsemble):
            raise RuntimeError("frozen fit did not produce a packed leafwise forest")
        cache_before = core._flat_cache_
        storage_before = _packed_storage(flat)
        cases = {}
        for name, frame in case_frames.items():
            print(f"running {name} ({len(frame):,} rows)", flush=True)
            cases[name] = _case_result(name, frame, model)
        seed_binned = core.prep_.transform(
            core._prepare_predict_X(
                _check_predict_input(model, fold_1), validated=True
            )
        )
        boundaries = _boundary_results(core, seed_binned)
        storage_after = _packed_storage(core._flat_ensemble())
        state_after = canonical_fitted_state_sha256(model)
        model.save_model(archive_after_path)
        archive_after = archive_after_path.read_bytes()

    archive_before_sha = _sha256_bytes(archive_before)
    train_prediction_sha = _sha256_array(train_prediction, "<f8")
    invariants = {
        "state_sha256_before": state_before,
        "state_sha256_after": state_after,
        "state_sha256_exact": state_before == EXPECTED_STATE_SHA256,
        "archive_sha256_before": archive_before_sha,
        "archive_bytes_before": len(archive_before),
        "archive_sha256_exact": archive_before_sha == EXPECTED_ARCHIVE_SHA256,
        "archive_length_exact": len(archive_before) == EXPECTED_ARCHIVE_BYTES,
        "train_prediction_sha256": train_prediction_sha,
        "train_prediction_sha256_exact": (
            train_prediction_sha == EXPECTED_TRAIN_PREDICTION_SHA256
        ),
        "archive_unchanged_after_prediction": archive_after == archive_before,
        "fitted_state_unchanged_after_prediction": state_after == state_before,
        "packed_storage_before": storage_before,
        "packed_storage_after": storage_after,
        "packed_storage_sha256_before": _packed_storage_sha256(storage_before),
        "prechange_packed_storage_exact": (
            storage_before["bytes"] == EXPECTED_PACKED_STORAGE_BYTES
            and _packed_storage_sha256(storage_before)
            == EXPECTED_PACKED_STORAGE_SHA256
        ),
        "persistent_extra_bytes": (
            storage_after["bytes"] - storage_before["bytes"]
        ),
        "packed_cache_identity": (
            core._flat_cache_ is cache_before
            and core._flat_cache_[1] is flat
            and storage_after == storage_before
        ),
    }
    decision = _analyze(cases, boundaries, model_record, invariants)
    final_source = creator.git_state(REPO_ROOT)
    if final_source != binding["source"] or final_source["head"] != source_head:
        raise RuntimeError("source checkout changed during formal run")

    artifact = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "basketball_leafwise_packed_prediction_confirmation",
            "sha256": EXPECTED_PROTOCOL_SHA256,
            "single_lever": "bounded_64_row_scalar_leafwise_packed_prediction",
            "confirmation_fold": CONFIRMATION_FOLD,
            "exploratory_fold": 0,
            "basketball_is_primary_fatal_boundary": True,
            "lockbox_data_used": False,
        },
        "binding": binding,
        "data": {
            "raw": dataset.raw_metadata,
            "processed": dataset.processed_metadata,
            "fold_fingerprint_sha256": dataset.fold_fingerprint_sha256,
            "fold_test_sizes": dataset.fold_test_sizes,
            "cold_player_guardrail": dataset.player_guardrail.metadata,
        },
        "model": model_record,
        "invariants": invariants,
        "boundaries": boundaries,
        "cases": cases,
        "decision": decision,
        "literal_chimeraboost_code_copied": False,
    }
    _write_create_only(output, artifact)
    print(f"decision: {decision['recommendation']}")
    print(f"wrote {output}")
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-cache", type=Path, default=harness.DEFAULT_CACHE)
    args = parser.parse_args(argv)
    args.output = creator._absolute_lexical_path(args.output)
    args.data_cache = creator._absolute_lexical_path(args.data_cache)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args.output, args.data_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
