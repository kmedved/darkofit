#!/usr/bin/env python3
"""Run the one-shot spent Protein attribution for the automatic selector."""

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
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


RUNNER_PATH = Path(__file__).resolve()
ROOT = RUNNER_PATH.parents[1]
BENCH = RUNNER_PATH.parent
PROTOCOL_PATH = BENCH / "automatic_linear_selector_v2_protein_attribution_protocol.md"
TEST_PATH = ROOT / "tests/test_automatic_linear_selector_v2_protein_attribution.py"
DEVELOPMENT_CONTRACT_PATH = (
    BENCH / "automatic_linear_selector_v2_development_contract.md"
)
M6_RESULT_PATH = (
    BENCH / "automatic_linear_selector_v2_m6_v3_inspection1_result_20260722.json"
)
M6_MANIFEST_PATH = Path(str(M6_RESULT_PATH) + ".manifest.json")
RELEASE_CONTRACT_PATH = BENCH / "v011_compute_ladder_contract_v3_20260722.json"
RELEASE_RUNNER_PATH = BENCH / "run_v011_compute_ladder.py"

CONTRACT_ID = "automatic-linear-selector-v2-protein-attribution-20260722"
MECHANISM_ID = "automatic_linear_selector_v2"
CANDIDATE_COMMIT = "a53d4bf543534678189d87d88dcad87dd2a8bd8f"
CANDIDATE_VERSION = "0.11.0"
DATASET = "physiochemical_protein"
TASK_ID = 363693
COORDINATES = (
    {"coordinate": 0, "repeat": 0, "fold": 0, "seed": 0},
    {"coordinate": 1, "repeat": 1, "fold": 1, "seed": 1001},
    {"coordinate": 2, "repeat": 2, "fold": 2, "seed": 2002},
)
ARMS = {
    "constant": False,
    "automatic": "auto",
    "explicit_linear": True,
}
BASE_ARM_ORDER = tuple(ARMS)
THREADS = 14
TABARENA_COMMIT = "4cd1d2526874962daae048a6f2dcf34aa272f3fa"
TABARENA_TREE = "a293df372a613c7358ba5fcd746f58d580cde7d6"
TABARENA_VERSION = "0.0.1"
WARMUP_ROWS = 1_400
WARMUP_ITERATIONS = 2
HARM_BOUND = 1.02
ATTEMPT_INDEX = 1
WORKER_TIMEOUT_SECONDS = 7_200.0
RSS_INTERVAL_SECONDS = 0.005
PREDICTION_PILOTS = 3
PREDICTION_TARGET_SECONDS = 1.0
PREDICTION_MIN_SECONDS = 0.5
PREDICTION_MIN_CALLS = 3
PREDICTION_MAX_CALLS = 65_536
WORKER_PREFIX = "AUTOMATIC_SELECTOR_PROTEIN_RESULT="
WORKER_ENVIRONMENT = {
    "OMP_NUM_THREADS": str(THREADS),
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": str(THREADS),
    "MKL_NUM_THREADS": str(THREADS),
    "MKL_DYNAMIC": "FALSE",
    "NUMEXPR_NUM_THREADS": str(THREADS),
    "NUMBA_NUM_THREADS": str(THREADS),
    "DARKOFIT_WARMUP": "0",
    "CHIMERABOOST_WARMUP": "0",
    "PYTHONHASHSEED": "0",
}

EXPECTED_HASHES = {
    DEVELOPMENT_CONTRACT_PATH: (
        "fe2d476417e8e8087a3c7342eee0d5cb82a6b8a4ee3f360a1806ee4c0922163b"
    ),
    M6_RESULT_PATH: (
        "7445b70ca3bc727bb24f8990ceef590ca933eb1dd45ccefe9ee5788eff211948"
    ),
    M6_MANIFEST_PATH: (
        "601f069896cdf664fcab470abe8c3643f0c0aacf5f79572a6663e304af3d7782"
    ),
    RELEASE_CONTRACT_PATH: (
        "61e788f06b88eefcc2e3c08a38402bf93246e7334980a77061b46763650b581a"
    ),
    RELEASE_RUNNER_PATH: (
        "db5b47af68fa0d74458c9d48d0c441caee8621cf1922542df2a27668118d14fb"
    ),
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_safe(value: Any, *, field: str = "value") -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise RuntimeError(f"{field} is nonfinite")
        return value
    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RuntimeError(f"{field} has a non-string key")
            output[key] = _json_safe(item, field=f"{field}.{key}")
        return output
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise RuntimeError(f"{field} has unsupported type {type(value)!r}")


def _git(repository: Path, *args: str, binary: bool = False):
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=False,
        capture_output=True,
        text=not binary,
    )
    if result.returncode:
        stderr = result.stderr if not binary else result.stderr.decode(errors="replace")
        stdout = result.stdout if not binary else result.stdout.decode(errors="replace")
        raise RuntimeError(stderr.strip() or stdout.strip())
    return result.stdout if binary else result.stdout.strip()


def source_state(repository: Path) -> dict[str, Any]:
    repository = repository.expanduser().resolve()
    root = Path(_git(repository, "rev-parse", "--show-toplevel")).resolve()
    if root != repository:
        raise RuntimeError(f"source must name its Git root: {repository}")
    status = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    return {
        "path": str(repository),
        "head": _git(repository, "rev-parse", "HEAD"),
        "tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "clean": not status,
        "status": status,
    }


def _tracked_head_bytes(path: Path) -> bytes:
    relative = path.resolve().relative_to(ROOT)
    return _git(ROOT, "show", f"HEAD:{relative}", binary=True)


def expected_ordered_grid() -> list[dict[str, Any]]:
    grid = []
    for index, coordinate in enumerate(COORDINATES):
        rotation = index % len(BASE_ARM_ORDER)
        order = BASE_ARM_ORDER[rotation:] + BASE_ARM_ORDER[:rotation]
        for position, arm in enumerate(order):
            grid.append({**coordinate, "position": position, "arm": arm})
    return grid


def ordered_grid_sha256() -> str:
    return hashlib.sha256(_canonical_json(expected_ordered_grid())).hexdigest()


def _validate_harness() -> dict[str, Any]:
    state = source_state(ROOT)
    if not state["clean"]:
        raise RuntimeError("Protein attribution requires a clean harness")
    if _git(ROOT, "rev-parse", "origin/main") != state["head"]:
        raise RuntimeError("Protein attribution harness is not published origin/main")
    for path in (
        PROTOCOL_PATH,
        RUNNER_PATH,
        TEST_PATH,
        *EXPECTED_HASHES,
    ):
        if not path.is_file() or path.read_bytes() != _tracked_head_bytes(path):
            raise RuntimeError(f"bound harness file differs from HEAD: {path.name}")
    return state


def _validate_candidate(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not (path / "darkofit/__init__.py").is_file():
        raise RuntimeError(f"candidate checkout is missing: {path}")
    state = source_state(path)
    if state["head"] != CANDIDATE_COMMIT or not state["clean"]:
        raise RuntimeError("candidate source is not the clean frozen commit")
    state["package_init_sha256"] = sha256(path / "darkofit/__init__.py")
    return state


def validate_tabarena_source(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    module = path / "packages/tabarena/src/tabarena/__init__.py"
    if not module.is_file():
        raise RuntimeError(f"TabArena source checkout is missing: {path}")
    if _git(path, "rev-parse", "HEAD") != TABARENA_COMMIT:
        raise RuntimeError("TabArena source commit drifted")
    if _git(path, "rev-parse", "HEAD^{tree}") != TABARENA_TREE:
        raise RuntimeError("TabArena source tree drifted")
    if _git(path, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("TabArena source checkout is not clean")
    return {
        "path": str(path),
        "commit": TABARENA_COMMIT,
        "tree": TABARENA_TREE,
        "status": "",
        "module_sha256": sha256(module),
    }


def validate_bound_evidence() -> dict[str, Any]:
    actual = {path: sha256(path) for path in EXPECTED_HASHES}
    if actual != EXPECTED_HASHES:
        raise RuntimeError("Protein attribution evidence binding drifted")

    m6 = json.loads(M6_RESULT_PATH.read_text())
    m6_manifest = json.loads(M6_MANIFEST_PATH.read_text())
    if (
        m6.get("contract_id") != "m6-quality-successor-v3"
        or m6.get("mechanism_id") != MECHANISM_ID
        or m6.get("inspection_index") != 1
        or m6.get("analysis", {}).get("disposition") != "advance"
        or m6_manifest.get("inspection_spent") is not True
        or m6_manifest.get("sources_before_and_after", {})
        .get("candidate_default", {})
        .get("head")
        != CANDIDATE_COMMIT
    ):
        raise RuntimeError("M6 advancement binding is invalid")

    release = json.loads(RELEASE_CONTRACT_PATH.read_text())
    release_coordinates = {
        (item["dataset"], item["repeat"], item["fold"])
        for item in release.get("execution", {}).get("coordinates", ())
    }
    expected_coordinates = {
        (DATASET, item["repeat"], item["fold"]) for item in COORDINATES
    }
    tasks = {
        (item["dataset"], item["task_id"])
        for item in release.get("execution", {}).get("tasks", ())
    }
    if (
        release.get("contract_id") != "v011-release-compute-ladder-20260722-v3"
        or release.get("contract_frozen") is not True
        or not expected_coordinates.issubset(release_coordinates)
        or (DATASET, TASK_ID) not in tasks
        or release.get("execution", {}).get("tabarena", {}).get("commit")
        != TABARENA_COMMIT
    ):
        raise RuntimeError("release-ladder Protein coordinate binding is invalid")

    return {
        str(path.relative_to(ROOT)): {"sha256": digest}
        for path, digest in actual.items()
    }


def _hardware() -> dict[str, Any]:
    import psutil

    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    if physical != THREADS or logical != THREADS:
        raise RuntimeError(
            f"Protein attribution requires the frozen 14/14 host, got "
            f"{physical}/{logical}"
        )
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "physical_cpus": physical,
        "logical_cpus": logical,
        "memory_bytes": int(psutil.virtual_memory().total),
        "load_average": list(os.getloadavg()),
    }


def _exclusive_machine_audit() -> dict[str, Any]:
    import psutil

    own_chain = {os.getpid()}
    ancestor = psutil.Process().parent()
    while ancestor is not None:
        own_chain.add(ancestor.pid)
        try:
            ancestor = ancestor.parent()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            break
    markers = (
        "run_automatic_linear_selector_v2_protein_attribution",
        "run_m6_quality_successor",
        "run_v011_compute_ladder",
        "run_v011_m2_broad_panel",
        "run_v011_ensemble_evidence",
        "run_m3",
        "run_tabarena",
    )
    conflicts = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess, ValueError):
            continue
        if pid not in own_chain and any(marker in command for marker in markers):
            conflicts.append({"pid": pid, "command": command})
    if conflicts:
        raise RuntimeError(f"another benchmark process is active: {conflicts}")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "conflicting_benchmark_processes": [],
    }


def _output_paths(prefix: Path) -> dict[str, Path]:
    prefix = prefix.expanduser().resolve()
    try:
        prefix.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise RuntimeError("Protein attribution outputs must be external to harness")
    return {
        "manifest": Path(str(prefix) + "_manifest.json"),
        "raw": Path(str(prefix) + "_raw.json"),
        "result": Path(str(prefix) + "_result.json"),
    }


def _write_create_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _activate_sources(candidate: Path, tabarena_source: Path) -> None:
    if "darkofit" in sys.modules or "tabarena" in sys.modules:
        raise RuntimeError("product modules loaded before source activation")
    candidate = candidate.resolve()
    tabarena_package = (
        tabarena_source.resolve() / "packages/tabarena/src"
    )
    sys.path.insert(0, str(tabarena_package))
    sys.path.insert(0, str(candidate))


def _load_split(
    repeat: int,
    fold: int,
    tabarena_source: Path,
) -> dict[str, Any]:
    from importlib.metadata import version as distribution_version

    from tabarena.benchmark.task.spec import task_spec_from_task_id_str
    from tabarena.contexts import TabArenaContext
    import tabarena

    module_path = Path(tabarena.__file__).resolve()
    try:
        module_path.relative_to(tabarena_source.resolve())
    except ValueError as exc:
        raise RuntimeError("TabArena imported from an unpinned source") from exc
    if distribution_version("tabarena") != TABARENA_VERSION:
        raise RuntimeError("TabArena package version drifted")
    context = TabArenaContext()
    metadata = context.task_metadata_collection.task_metadata_by_dataset()[DATASET]
    if int(metadata.task_id_str) != TASK_ID:
        raise RuntimeError("Protein task ID drifted")
    task = (
        task_spec_from_task_id_str(metadata.task_id_str)
        .with_task_metadata(metadata)
        .load()
    )
    if task.problem_type != "regression" or task.eval_metric != "rmse":
        raise RuntimeError("Protein task is not RMSE regression")
    X_train, y_train, X_test, y_test = task.get_train_test_split(
        fold=fold,
        repeat=repeat,
    )
    return {
        "task_id": int(task.task_id),
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def _data_loader_preflight(
    candidate: Path,
    tabarena_source: Path,
) -> dict[str, Any]:
    """Prove the exact data loader works before an attempt can be spent."""
    product_modules_before = {
        "darkofit": "darkofit" in sys.modules,
        "tabarena": "tabarena" in sys.modules,
    }
    _activate_sources(candidate.resolve(), tabarena_source.resolve())
    records = []
    for coordinate in COORDINATES:
        data = _load_split(
            coordinate["repeat"],
            coordinate["fold"],
            tabarena_source,
        )
        if int(data["task_id"]) != TASK_ID:
            raise RuntimeError("Protein preflight task ID drifted")
        fingerprints = _split_fingerprints(data)
        records.append(
            {
                **coordinate,
                "task_id": int(data["task_id"]),
                "train_rows": int(len(data["y_train"])),
                "test_rows": int(len(data["y_test"])),
                "feature_count": int(data["X_train"].shape[1]),
                "combined_split_sha256": fingerprints["combined_sha256"],
            }
        )
        del data
    if "darkofit" in sys.modules:
        raise RuntimeError("data-loader preflight imported DarkoFit")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "product_modules_before": product_modules_before,
        "darkofit_remained_unloaded": True,
        "coordinates": records,
    }


def _pandas_sha256(value: Any) -> str:
    import pandas as pd

    if isinstance(value, pd.Series):
        frame = value.to_frame(name=str(value.name))
    else:
        frame = value
    schema = [
        {"column": str(column), "dtype": str(frame[column].dtype)}
        for column in frame.columns
    ]
    hashed = pd.util.hash_pandas_object(
        frame,
        index=True,
        categorize=True,
    ).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update(_canonical_json(schema))
    digest.update(hashed.tobytes(order="C"))
    return digest.hexdigest()


def _split_fingerprints(data: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        name: _pandas_sha256(data[name])
        for name in ("X_train", "y_train", "X_test", "y_test")
    }
    fields["combined_sha256"] = _sha256_bytes(_canonical_json(fields))
    return fields


def _categorical_features(X: Any) -> list[str]:
    return [str(column) for column in X.select_dtypes(include=["category"]).columns]


class ProcessTreeRSSSampler:
    """Sample aggregate RSS for this worker and every recursive child."""

    def __init__(self, interval_seconds: float = RSS_INTERVAL_SECONDS):
        self.interval_seconds = float(interval_seconds)
        self.start_bytes = 0
        self.peak_bytes = 0
        self.end_bytes = 0
        self.samples = 0
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def current_bytes() -> int:
        import psutil

        root = psutil.Process()
        total = 0
        seen: set[int] = set()
        for process in (root, *root.children(recursive=True)):
            if process.pid in seen:
                continue
            seen.add(process.pid)
            try:
                total += int(process.memory_info().rss)
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        if total <= 0:
            raise RuntimeError("process-tree RSS is unavailable")
        return total

    def _sample_once(self) -> None:
        value = self.current_bytes()
        self.peak_bytes = max(self.peak_bytes, value)
        self.samples += 1

    def _run_sampler(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._sample_once()
            except Exception as exc:  # pragma: no cover - platform telemetry
                self.errors.append(f"{type(exc).__name__}: {exc}")

    def __enter__(self):
        self.start_bytes = self.current_bytes()
        self.peak_bytes = self.start_bytes
        self.samples = 1
        self._thread = threading.Thread(target=self._run_sampler, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self._sample_once()
            self.end_bytes = self.current_bytes()
            self.peak_bytes = max(self.peak_bytes, self.end_bytes)
        except Exception as cleanup_exc:
            self.errors.append(f"{type(cleanup_exc).__name__}: {cleanup_exc}")
        if exc_type is None and (self.errors or self.samples < 2):
            raise RuntimeError(f"process-tree RSS sampling failed: {self.errors}")
        return False


def _coerce_prediction(value: Any, n_rows: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape == (n_rows, 1):
        value = value.reshape(n_rows)
    if value.shape != (n_rows,) or not np.isfinite(value).all():
        raise RuntimeError(f"invalid regression prediction: {value.shape}")
    return value


def _prediction_array(model: Any, X: Any) -> np.ndarray:
    return _coerce_prediction(model.predict(X), len(X))


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _timed_prediction(model: Any, X: Any) -> dict[str, Any]:
    reference = _prediction_array(model, X)
    pilots = []
    for _ in range(PREDICTION_PILOTS):
        started = time.perf_counter_ns()
        raw_value = model.predict(X)
        pilots.append((time.perf_counter_ns() - started) / 1e9)
        value = _coerce_prediction(raw_value, len(X))
        if not np.array_equal(reference, value):
            raise RuntimeError("prediction pilot changed output")
    pilot_median = float(np.median(np.asarray(pilots, dtype=np.float64)))
    calls = int(
        min(
            PREDICTION_MAX_CALLS,
            max(
                PREDICTION_MIN_CALLS,
                math.ceil(PREDICTION_TARGET_SECONDS / max(pilot_median, 1e-9)),
            ),
        )
    )
    final_raw = None
    gc.disable()
    started = time.perf_counter_ns()
    try:
        for _ in range(calls):
            final_raw = model.predict(X)
    finally:
        elapsed = (time.perf_counter_ns() - started) / 1e9
        gc.enable()
    final = None if final_raw is None else _coerce_prediction(final_raw, len(X))
    if final is None or not np.array_equal(reference, final):
        raise RuntimeError("formal prediction interval changed output")
    if elapsed < PREDICTION_MIN_SECONDS:
        raise RuntimeError(
            f"prediction interval {elapsed:.6f}s missed the frozen floor"
        )
    return {
        "rows": int(len(X)),
        "pilots_seconds": [float(value) for value in pilots],
        "pilot_median_seconds": pilot_median,
        "calls": calls,
        "interval_seconds": float(elapsed),
        "seconds_per_call": float(elapsed / calls),
        "rows_per_second": float(len(X) * calls / elapsed),
        "prediction_sha256": _array_sha256(reference),
    }


def _implementation(model: Any, candidate: Path) -> dict[str, Any]:
    module = sys.modules[model.__class__.__module__]
    path = Path(module.__file__).resolve()
    try:
        path.relative_to(candidate.resolve())
    except ValueError as exc:
        raise RuntimeError(f"model imported outside candidate source: {path}") from exc
    return {
        "class": model.__class__.__name__,
        "module": model.__class__.__module__,
        "module_path": str(path),
        "module_sha256": sha256(path),
    }


def _core_booster_state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix="darkofit-protein-state-") as temp:
        path = Path(temp) / "booster.npz"
        model.model_.save_model(path)
        with np.load(path, allow_pickle=False) as archive:
            for name in sorted(archive.files):
                value = archive[name]
                digest.update(name.encode())
                if name == "header":
                    if value.shape != () or not np.issubdtype(
                        value.dtype, np.str_
                    ):
                        raise RuntimeError("booster header has an invalid shape")
                    header = json.loads(str(value.item()))
                    auto_params = header.get("auto_params")
                    if not isinstance(auto_params, dict):
                        raise RuntimeError("booster auto_params header is invalid")
                    auto_params.pop("automatic_linear_selector", None)
                    diagnostics = auto_params.get("diagnostics")
                    if isinstance(diagnostics, dict):
                        diagnostics.pop("automatic_linear_selector", None)
                        if not diagnostics:
                            auto_params.pop("diagnostics")
                    digest.update(b"normalized-selector-provenance-v1")
                    digest.update(_canonical_json(header))
                    continue
                value = np.ascontiguousarray(value)
                digest.update(str(value.dtype).encode())
                digest.update(_canonical_json(list(value.shape)))
                digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _build_model(arm: str, seed: int):
    from darkofit import DarkoRegressor, __version__

    if __version__ != CANDIDATE_VERSION:
        raise RuntimeError("candidate DarkoFit version drifted")
    return DarkoRegressor(
        random_state=seed,
        thread_count=THREADS,
        diagnostic_warnings="never",
        linear_leaves=ARMS[arm],
    )


def _fit(model: Any, X: Any, y: Any, cat_features: Sequence[str]) -> Any:
    return model.fit(X, y, cat_features=list(cat_features) or None)


def _fitted_model_metadata(model: Any) -> dict[str, Any]:
    resolved_linear_leaves = bool(
        getattr(
            model,
            "linear_leaves_selected_",
            getattr(model.model_, "linear_leaves", False),
        )
    )
    return {
        "requested_linear_leaves": model.linear_leaves,
        "selected_linear_leaves": resolved_linear_leaves,
        "linear_leaves_active": bool(
            getattr(model.model_, "linear_leaves_active_", False)
        ),
        "best_n_estimators": int(model.best_n_estimators_),
        "learning_rate": float(model.learning_rate_),
        "tree_count": int(len(model.model_.trees_)),
        "resolved_thread_count": int(model.model_.n_threads_),
    }


def _warmup(
    arm: str,
    seed: int,
    data: Mapping[str, Any],
    cat_features: Sequence[str],
) -> dict[str, Any]:
    count = min(WARMUP_ROWS, len(data["y_train"]))
    model = _build_model(arm, seed)
    model.set_params(iterations=WARMUP_ITERATIONS)
    _fit(
        model,
        data["X_train"].iloc[:count].copy(),
        data["y_train"].iloc[:count].copy(),
        cat_features,
    )
    return {
        "rows": count,
        "iterations": WARMUP_ITERATIONS,
        "linear_leaves": ARMS[arm],
    }


def _worker_result(args: argparse.Namespace) -> dict[str, Any]:
    candidate = args.candidate_source.expanduser().resolve()
    tabarena_source = args.tabarena_source.expanduser().resolve()
    _validate_candidate(candidate)
    validate_tabarena_source(tabarena_source)
    actual_environment = {
        key: os.environ.get(key) for key in WORKER_ENVIRONMENT
    }
    if actual_environment != WORKER_ENVIRONMENT:
        raise RuntimeError(f"worker environment drifted: {actual_environment}")

    import numba

    if (
        int(numba.config.NUMBA_NUM_THREADS) != THREADS
        or int(numba.get_num_threads()) != THREADS
    ):
        raise RuntimeError("worker Numba thread budget is not exactly 14")

    expected = expected_ordered_grid()[args.worker_index]
    if args.arm != expected["arm"]:
        raise RuntimeError("worker arm does not match frozen order")
    _activate_sources(candidate, tabarena_source)
    data = _load_split(
        expected["repeat"],
        expected["fold"],
        tabarena_source,
    )
    if int(data["task_id"]) != TASK_ID:
        raise RuntimeError("Protein task ID drifted")
    fingerprints = _split_fingerprints(data)
    cat_features = _categorical_features(data["X_train"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warmup = _warmup(expected["arm"], expected["seed"], data, cat_features)
        model = _build_model(expected["arm"], expected["seed"])
        implementation = _implementation(model, candidate)
        ambient_before = int(numba.get_num_threads())
        with ProcessTreeRSSSampler() as rss:
            started = time.perf_counter_ns()
            _fit(model, data["X_train"], data["y_train"], cat_features)
            fit_seconds = (time.perf_counter_ns() - started) / 1e9
        ambient_after_fit = int(numba.get_num_threads())
        if ambient_after_fit != ambient_before:
            raise RuntimeError("fit leaked the worker Numba thread mask")
        core_state_sha256 = _core_booster_state_sha256(model)
        prediction = _prediction_array(model, data["X_test"])
        ambient_after_predict = int(numba.get_num_threads())
        if ambient_after_predict != ambient_before:
            raise RuntimeError("predict leaked the worker Numba thread mask")
        truth = np.asarray(data["y_test"], dtype=np.float64)
        rmse = float(np.sqrt(np.mean(np.square(prediction - truth))))
        prediction_timing = _timed_prediction(model, data["X_test"])
        ambient_after_timing = int(numba.get_num_threads())
        if ambient_after_timing != ambient_before:
            raise RuntimeError("timed predict leaked the worker Numba thread mask")
        selector = getattr(model, "automatic_linear_selector_", None)
        model_metadata = _fitted_model_metadata(model)

    if not math.isfinite(rmse) or rmse <= 0.0:
        raise RuntimeError("Protein RMSE is invalid")
    return {
        "schema_version": 1,
        "kind": "automatic_linear_selector_v2_protein_worker",
        **expected,
        "dataset": DATASET,
        "task_id": TASK_ID,
        "pid": os.getpid(),
        "parent_pid": int(args.parent_pid),
        "started_at_utc": args.worker_started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(len(data["y_train"])),
        "test_rows": int(len(data["y_test"])),
        "feature_count": int(data["X_train"].shape[1]),
        "categorical_features": cat_features,
        "fingerprints": fingerprints,
        "test_rmse": rmse,
        "prediction_sha256": _array_sha256(prediction),
        "core_booster_state_sha256": core_state_sha256,
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
        "prediction": prediction_timing,
        "selector": _json_safe(selector, field="selector"),
        "model": model_metadata,
        "implementation": implementation,
        "warmup": warmup,
        "environment": actual_environment,
        "numba_threads_before_fit": ambient_before,
        "numba_threads_after_fit": ambient_after_fit,
        "numba_threads_after_predict": ambient_after_predict,
        "numba_threads_after_timing": ambient_after_timing,
        "warnings": [
            {"category": item.category.__name__, "message": str(item.message)}
            for item in caught
        ],
    }


def _geometric_mean(values: Sequence[float]) -> float:
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise RuntimeError("geometric mean requires finite positive values")
    return float(math.exp(math.fsum(math.log(value) for value in values) / len(values)))


def analyze_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = expected_ordered_grid()
    if len(rows) != len(expected):
        raise RuntimeError("Protein attribution row count is incomplete")
    by_key = {}
    for row in rows:
        key = (int(row["coordinate"]), str(row["arm"]))
        if key in by_key:
            raise RuntimeError(f"duplicate Protein attribution row: {key}")
        by_key[key] = row
    expected_keys = {
        (item["coordinate"], item["arm"]) for item in expected
    }
    if set(by_key) != expected_keys:
        raise RuntimeError("Protein attribution grid is incomplete or drifted")
    for row, cell in zip(rows, expected):
        identity = {
            key: row.get(key)
            for key in (
                "coordinate",
                "repeat",
                "fold",
                "seed",
                "position",
                "arm",
            )
        }
        if identity != cell:
            raise RuntimeError("Protein attribution row order or identity drifted")
        if (
            row.get("schema_version") != 1
            or row.get("kind")
            != "automatic_linear_selector_v2_protein_worker"
            or row.get("dataset") != DATASET
            or row.get("task_id") != TASK_ID
            or row.get("environment") != WORKER_ENVIRONMENT
            or any(
                row.get(key) != THREADS
                for key in (
                    "numba_threads_before_fit",
                    "numba_threads_after_fit",
                    "numba_threads_after_predict",
                    "numba_threads_after_timing",
                )
            )
        ):
            raise RuntimeError("Protein attribution worker contract drifted")
        model = row.get("model")
        if not isinstance(model, Mapping) or model.get(
            "requested_linear_leaves"
        ) != ARMS[cell["arm"]]:
            raise RuntimeError("Protein attribution fitted arm drifted")
        if cell["arm"] != "automatic" and model.get(
            "selected_linear_leaves"
        ) is not (cell["arm"] == "explicit_linear"):
            raise RuntimeError("explicit Protein arm resolved incorrectly")
        if cell["arm"] != "automatic" and model.get(
            "linear_leaves_active"
        ) is not (cell["arm"] == "explicit_linear"):
            raise RuntimeError("explicit Protein arm activation drifted")
        if cell["arm"] != "automatic" and row.get("selector") is not None:
            raise RuntimeError("explicit Protein arm has automatic selector state")
        prediction = row.get("prediction")
        fit_rss = row.get("fit_rss")
        if (
            not isinstance(prediction, Mapping)
            or prediction.get("rows") != row.get("test_rows")
            or prediction.get("prediction_sha256")
            != row.get("prediction_sha256")
            or len(prediction.get("pilots_seconds", ())) != PREDICTION_PILOTS
            or not (
                PREDICTION_MIN_CALLS
                <= prediction.get("calls", 0)
                <= PREDICTION_MAX_CALLS
            )
            or prediction.get("interval_seconds", 0.0)
            < PREDICTION_MIN_SECONDS
            or prediction.get("seconds_per_call", 0.0) <= 0.0
            or not isinstance(fit_rss, Mapping)
            or fit_rss.get("scope") != "worker_plus_recursive_children"
            or fit_rss.get("errors") != []
            or fit_rss.get("interval_seconds") != RSS_INTERVAL_SECONDS
        ):
            raise RuntimeError("Protein attribution timing or RSS record is invalid")
        for digest_key in (
            "prediction_sha256",
            "core_booster_state_sha256",
        ):
            digest = row.get(digest_key)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise RuntimeError(f"invalid Protein {digest_key}")

    coordinate_results = []
    invariant_rows = []
    ratios = []
    margins = []
    for coordinate in COORDINATES:
        index = coordinate["coordinate"]
        constant = by_key[(index, "constant")]
        automatic = by_key[(index, "automatic")]
        explicit = by_key[(index, "explicit_linear")]
        fingerprints_equal = (
            constant["fingerprints"]
            == automatic["fingerprints"]
            == explicit["fingerprints"]
        )
        selector = automatic.get("selector")
        selector_mapping = selector if isinstance(selector, Mapping) else {}
        split = selector_mapping.get("split")
        split_mapping = split if isinstance(split, Mapping) else {}
        margin = selector_mapping.get("relative_validation_improvement")
        margin_value = float(margin) if isinstance(margin, (int, float)) else None
        invariants = {
            "fingerprints_equal": fingerprints_equal,
            "eligible": selector_mapping.get("eligible") is True,
            "selected_linear_reason": (
                selector_mapping.get("reason") == "selected_linear"
                and automatic.get("model", {}).get("selected_linear_leaves") is True
                and automatic.get("model", {}).get("linear_leaves_active") is True
            ),
            "margin_at_least_0_03": (
                margin_value is not None and margin_value >= 0.03
            ),
            "selector_provenance": (
                selector_mapping.get("fit_random_state_seed")
                == coordinate["seed"]
                and split_mapping.get("source") == "automatic_holdout"
                and split_mapping.get("policy")
                == "weighted_target_stratified"
            ),
            "selection_rows_disjoint": split_mapping.get("rows_disjoint") is True,
            "final_linear_leaves_active": (
                selector_mapping.get("final_booster_linear_leaves") is True
                and selector_mapping.get("final_linear_leaves_active") is True
            ),
            "prediction_exact": (
                automatic["prediction_sha256"]
                == explicit["prediction_sha256"]
            ),
            "core_state_exact": (
                automatic["core_booster_state_sha256"]
                == explicit["core_booster_state_sha256"]
            ),
        }
        ratio = float(automatic["test_rmse"]) / float(constant["test_rmse"])
        ratios.append(ratio)
        if margin_value is not None:
            margins.append(margin_value)
        invariant_rows.append({"coordinate": index, **invariants})
        coordinate_results.append(
            {
                **coordinate,
                "rmse": {
                    arm: float(by_key[(index, arm)]["test_rmse"])
                    for arm in BASE_ARM_ORDER
                },
                "automatic_over_constant_rmse": ratio,
                "selector_margin": margin_value,
                "invariants": invariants,
                "fit_seconds": {
                    arm: float(by_key[(index, arm)]["fit_seconds"])
                    for arm in BASE_ARM_ORDER
                },
                "prediction_seconds_per_call": {
                    arm: float(
                        by_key[(index, arm)]["prediction"]["seconds_per_call"]
                    )
                    for arm in BASE_ARM_ORDER
                },
                "fit_rss_peak_bytes": {
                    arm: int(by_key[(index, arm)]["fit_rss"]["peak_bytes"])
                    for arm in BASE_ARM_ORDER
                },
                "fit_rss_peak_delta_bytes": {
                    arm: int(
                        by_key[(index, arm)]["fit_rss"]["peak_delta_bytes"]
                    )
                    for arm in BASE_ARM_ORDER
                },
                "prediction_sha256": {
                    arm: by_key[(index, arm)]["prediction_sha256"]
                    for arm in BASE_ARM_ORDER
                },
                "core_booster_state_sha256": {
                    arm: by_key[(index, arm)]["core_booster_state_sha256"]
                    for arm in BASE_ARM_ORDER
                },
            }
        )

    aggregate_ratio = _geometric_mean(ratios)
    worst_index = max(range(len(ratios)), key=ratios.__getitem__)
    invariants_pass = all(
        all(value for key, value in row.items() if key != "coordinate")
        for row in invariant_rows
    )
    aggregate_harm_pass = aggregate_ratio <= HARM_BOUND
    coordinate_harm_pass = max(ratios) <= HARM_BOUND
    passes = invariants_pass and aggregate_harm_pass and coordinate_harm_pass

    cost_summary = {}
    for arm in BASE_ARM_ORDER:
        arm_rows = [by_key[(item["coordinate"], arm)] for item in COORDINATES]
        cost_summary[arm] = {
            "fit_seconds_geometric_mean": _geometric_mean(
                [float(row["fit_seconds"]) for row in arm_rows]
            ),
            "prediction_seconds_per_call_geometric_mean": _geometric_mean(
                [float(row["prediction"]["seconds_per_call"]) for row in arm_rows]
            ),
            "fit_rss_peak_bytes_geometric_mean": _geometric_mean(
                [float(row["fit_rss"]["peak_bytes"]) for row in arm_rows]
            ),
            "fit_rss_peak_delta_bytes": [
                int(row["fit_rss"]["peak_delta_bytes"]) for row in arm_rows
            ],
        }

    return {
        "disposition": (
            "ready_for_powered_fresh_design" if passes else "terminal_close"
        ),
        "all_conditions_pass": passes,
        "gates": {
            "all_selector_and_exactness_invariants": invariants_pass,
            "aggregate_ratio_at_most_1_02": aggregate_harm_pass,
            "every_coordinate_ratio_at_most_1_02": coordinate_harm_pass,
        },
        "aggregate_automatic_over_constant_rmse": aggregate_ratio,
        "worst_coordinate": int(COORDINATES[worst_index]["coordinate"]),
        "worst_coordinate_ratio": float(ratios[worst_index]),
        "minimum_selector_margin": min(margins) if margins else None,
        "coordinates": coordinate_results,
        "invariants": invariant_rows,
        "cost_summary": cost_summary,
    }


def _worker_command(
    args: argparse.Namespace,
    worker_index: int,
    arm: str,
) -> list[str]:
    return [
        sys.executable,
        str(RUNNER_PATH),
        "--candidate-source",
        str(args.candidate_source.resolve()),
        "--tabarena-source",
        str(args.tabarena_source.resolve()),
        "--output-prefix",
        str(args.output_prefix.resolve()),
        "--worker-index",
        str(worker_index),
        "--arm",
        arm,
        "--parent-pid",
        str(os.getpid()),
        "--worker-started-at",
        datetime.now(timezone.utc).isoformat(),
    ]


def _parse_worker_stdout(stdout: str) -> dict[str, Any]:
    matches = [
        line[len(WORKER_PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if len(matches) != 1:
        raise RuntimeError("worker did not emit exactly one result")
    return json.loads(matches[0])


def _run_parent(args: argparse.Namespace) -> int:
    paths = _output_paths(args.output_prefix)
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise FileExistsError("Protein attribution output is create-only")

    harness = _validate_harness()
    candidate_before = _validate_candidate(args.candidate_source)
    tabarena_before = validate_tabarena_source(args.tabarena_source)
    bindings = validate_bound_evidence()
    hardware = _hardware()
    exclusivity = _exclusive_machine_audit()
    data_loader_preflight = _data_loader_preflight(
        args.candidate_source,
        args.tabarena_source,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "mechanism_id": MECHANISM_ID,
        "attempt_index": ATTEMPT_INDEX,
        "attempt_spent": True,
        "created_at_utc": started_at,
        "status": "launched_before_worker_zero",
        "grid": expected_ordered_grid(),
        "ordered_grid_sha256": ordered_grid_sha256(),
        "worker_environment": WORKER_ENVIRONMENT,
        "sources": {
            "harness": harness,
            "candidate": candidate_before,
            "tabarena": tabarena_before,
        },
        "bindings": {
            **bindings,
            str(PROTOCOL_PATH.relative_to(ROOT)): {
                "sha256": sha256(PROTOCOL_PATH)
            },
            str(RUNNER_PATH.relative_to(ROOT)): {"sha256": sha256(RUNNER_PATH)},
            str(TEST_PATH.relative_to(ROOT)): {"sha256": sha256(TEST_PATH)},
        },
        "hardware": hardware,
        "exclusive_machine_audit": exclusivity,
        "data_loader_preflight": data_loader_preflight,
        "planned_outputs": {key: str(path) for key, path in paths.items()},
        "command": [sys.executable, *sys.argv],
    }
    _write_create_only_json(paths["manifest"], manifest)

    rows = []
    try:
        for worker_index, cell in enumerate(expected_ordered_grid()):
            command = _worker_command(args, worker_index, cell["arm"])
            environment = os.environ.copy()
            environment.update(WORKER_ENVIRONMENT)
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=WORKER_TIMEOUT_SECONDS,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"worker {worker_index} failed with {completed.returncode}: "
                    f"{completed.stderr.strip() or completed.stdout.strip()}"
                )
            row = _parse_worker_stdout(completed.stdout)
            if row["arm"] != cell["arm"] or row["coordinate"] != cell["coordinate"]:
                raise RuntimeError("worker result identity drifted")
            rows.append(row)
            print(
                f"ok {worker_index + 1}/{len(expected_ordered_grid())} "
                f"coordinate={cell['coordinate']} arm={cell['arm']}",
                flush=True,
            )

        candidate_after = _validate_candidate(args.candidate_source)
        tabarena_after = validate_tabarena_source(args.tabarena_source)
        harness_after = _validate_harness()
        if (
            candidate_after != candidate_before
            or tabarena_after != tabarena_before
            or harness_after != harness
        ):
            raise RuntimeError("bound source state changed during execution")
        analysis = analyze_rows(rows)
        raw = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "mechanism_id": MECHANISM_ID,
            "attempt_index": ATTEMPT_INDEX,
            "rows": rows,
        }
        _write_create_only_json(paths["raw"], raw)
        result = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "mechanism_id": MECHANISM_ID,
            "attempt_index": ATTEMPT_INDEX,
            "attempt_spent": True,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_scope": "spent_protein_development_attribution",
            "shipping_or_default_claim_eligible": False,
            "fresh_or_lockbox_accessed": False,
            "analysis": analysis,
            "artifacts": {
                "manifest": {
                    "path": str(paths["manifest"]),
                    "sha256": sha256(paths["manifest"]),
                },
                "raw": {
                    "path": str(paths["raw"]),
                    "sha256": sha256(paths["raw"]),
                },
            },
        }
        _write_create_only_json(paths["result"], result)
        return 0
    except BaseException as exc:
        failure = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "mechanism_id": MECHANISM_ID,
            "attempt_index": ATTEMPT_INDEX,
            "attempt_spent": True,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_scope": "spent_protein_development_attribution",
            "shipping_or_default_claim_eligible": False,
            "fresh_or_lockbox_accessed": False,
            "analysis": {
                "disposition": "terminal_execution_failure",
                "completed_worker_count": len(rows),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            "artifacts": {
                "manifest": {
                    "path": str(paths["manifest"]),
                    "sha256": sha256(paths["manifest"]),
                }
            },
        }
        if not paths["result"].exists():
            _write_create_only_json(paths["result"], failure)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--tabarena-source", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--arm", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-started-at",
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    worker_fields = (
        args.worker_index,
        args.arm,
        args.parent_pid,
        args.worker_started_at,
    )
    if any(value is not None for value in worker_fields) and not all(
        value is not None for value in worker_fields
    ):
        parser.error("internal worker arguments must be supplied together")
    if args.worker_index is not None:
        if args.worker_index not in range(len(expected_ordered_grid())):
            parser.error("worker index is outside the frozen grid")
        if args.arm not in ARMS:
            parser.error("worker arm is invalid")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_index is not None:
        result = _worker_result(args)
        print(WORKER_PREFIX + json.dumps(result, allow_nan=False, sort_keys=True))
        return 0
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
