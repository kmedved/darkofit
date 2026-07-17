#!/usr/bin/env python3
"""Run the frozen native-ordinal C2 development or confirmation tier."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CHIMERA_ROOT = ROOT.parent / "chimeraboost"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import basketball_harness as basketball  # noqa: E402
from benchmarks import build_ctr23_contamination_registry as ctr  # noqa: E402
from benchmarks import build_native_ordinal_c2_registry as registry_builder  # noqa: E402


CONTROL = "control"
CANDIDATE = "candidate"
ARMS = (CONTROL, CANDIDATE)
DEVELOPMENT = "development"
CONFIRMATION = "confirmation"
TIERS = (DEVELOPMENT, CONFIRMATION)
RANDOM_STATE = 4
SPLIT_SEED = 20260717
VALIDATION_FRACTION = 0.20
THREADS_PER_WORKER = 6
CONCURRENT_WORKERS = 3
PREDICTION_CALLS = 20
WORKER_RESULT_PREFIX = "NATIVE_ORDINAL_C2_RESULT="

REGISTRY_PATH = ROOT / "benchmarks" / "native_ordinal_c2_registry.json"
PROTOCOL_PATH = ROOT / "benchmarks" / "native_ordinal_c2_protocol.md"
ANALYZER_PATH = ROOT / "benchmarks" / "analyze_native_ordinal_c2.py"
DEFAULT_DEVELOPMENT_OUTPUT = (
    ROOT / "benchmarks" / "native_ordinal_c2_development_raw.json"
)
DEFAULT_CONFIRMATION_OUTPUT = (
    ROOT / "benchmarks" / "native_ordinal_c2_confirmation_raw.json"
)
CACHE_ROOT = ROOT / ".cache" / "native-ordinal-c2"

EXPECTED_REGISTRY_FILE_SHA256 = (
    "34343d5296698ad7ac728fbef40961f384ca61923e6524afa8a2c7eeda7080d3"
)
EXPECTED_REGISTRY_CONTENT_SHA256 = (
    "e7493131eb0cb1da00f1118c39f29130a44381e12f38bd2e2bd972132f953b28"
)
EXPECTED_PROTOCOL_SHA256 = (
    "c9be15732180fa6202db212ec74b95b6592cf31aaf70dde5c96dc7ee72f354e8"
)
EXPECTED_PACKAGE_TREE = "d4661e0d4a919d2e0f0da4385b12034a5c853a6c"
EXPECTED_CHIMERA_HEAD = "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d"
EXPECTED_NORMALIZED_RUNNER_SHA256 = (
    "b1f5a716552ab562c1edc8275256d82fca757c2ab7212e6338cf66302a54d1fd"
)
EXPECTED_ANALYZER_SHA256 = (
    "2e9ac551ca2092c83c6d689053be4fb75e698a1d6a39c8444e32bbbf63a5bd2c"
)
EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "platform": "macOS-26.5.2-arm64-arm-64bit",
    "machine": "arm64",
    "logical_cpu_count": 18,
    "dependencies": {
        "numpy": "2.4.6",
        "pandas": "2.3.3",
        "scikit-learn": "1.7.2",
        "numba": "0.66.0",
        "llvmlite": "0.48.0",
        "openml": "0.15.1",
    },
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _array_sha256(value: Any, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return _sha256_bytes(array.tobytes())


def _normalized_runner_sha256(path: Path | None = None) -> str:
    payload = (path or Path(__file__).resolve()).read_bytes()
    pattern = (
        rb'EXPECTED_NORMALIZED_RUNNER_SHA256 = \(\n'
        rb'    "[0-9a-f]{64}"\n'
        rb"\)"
    )
    replacement = (
        b'EXPECTED_NORMALIZED_RUNNER_SHA256 = (\n'
        b'    "' + (b"0" * 64) + b'"\n'
        b")"
    )
    normalized, count = re.subn(pattern, replacement, payload)
    if count != 1:
        raise RuntimeError("native-ordinal C2 runner normalization changed")
    return _sha256_bytes(normalized)


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True
    ).strip()


def _runtime_record() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in EXPECTED_RUNTIME["dependencies"]
        },
    }


def _source_state() -> dict[str, Any]:
    return {
        "head": _git("rev-parse", "HEAD"),
        "origin_main": _git("rev-parse", "origin/main"),
        "branch": _git("branch", "--show-current"),
        "status": _git("status", "--porcelain=v1", "--untracked-files=all"),
        "package_tree": _git("rev-parse", "HEAD:darkofit"),
        "registry_file_sha256": _sha256_file(REGISTRY_PATH),
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "runner_normalized_sha256": _normalized_runner_sha256(),
        "analyzer_sha256": (
            _sha256_file(ANALYZER_PATH) if ANALYZER_PATH.is_file() else None
        ),
        "chimeraboost_head": _git("rev-parse", "HEAD", cwd=CHIMERA_ROOT),
        "chimeraboost_status": _git(
            "status", "--porcelain=v1", "--untracked-files=all", cwd=CHIMERA_ROOT
        ),
    }


def _load_registry() -> dict[str, Any]:
    if _sha256_file(REGISTRY_PATH) != EXPECTED_REGISTRY_FILE_SHA256:
        raise RuntimeError("native-ordinal C2 registry file changed")
    artifact = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    content_hash = artifact.pop("registry_sha256", None)
    if (
        content_hash != EXPECTED_REGISTRY_CONTENT_SHA256
        or ctr.sha256_json(artifact) != EXPECTED_REGISTRY_CONTENT_SHA256
    ):
        raise RuntimeError("native-ordinal C2 registry content changed")
    artifact["registry_sha256"] = content_hash
    current_sources = {
        "builder_source_sha256": _sha256_file(
            Path(registry_builder.__file__).resolve()
        ),
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "declarations_sha256": _sha256_file(registry_builder.DECLARATIONS),
    }
    if any(
        artifact[key] != value for key, value in current_sources.items()
    ):
        raise RuntimeError("native-ordinal C2 registry source changed")
    if any(
        _sha256_file(ROOT / relative_path) != expected
        for relative_path, expected in artifact["source_artifacts"].items()
    ):
        raise RuntimeError("native-ordinal C2 registry dependency changed")
    return artifact


def require_frozen_source() -> tuple[dict[str, Any], dict[str, Any]]:
    source = _source_state()
    if (
        source["branch"] != "main"
        or source["head"] != source["origin_main"]
        or source["status"]
    ):
        raise RuntimeError(
            "formal native-ordinal C2 execution requires clean pushed main"
        )
    if source["package_tree"] != EXPECTED_PACKAGE_TREE:
        raise RuntimeError("DarkoFit package tree changed after C1")
    if source["registry_file_sha256"] != EXPECTED_REGISTRY_FILE_SHA256:
        raise RuntimeError("native-ordinal C2 registry changed")
    if source["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("native-ordinal C2 protocol changed")
    if (
        source["runner_normalized_sha256"]
        != EXPECTED_NORMALIZED_RUNNER_SHA256
    ):
        raise RuntimeError("native-ordinal C2 runner changed")
    if source["analyzer_sha256"] != EXPECTED_ANALYZER_SHA256:
        raise RuntimeError("native-ordinal C2 analyzer changed")
    if (
        source["chimeraboost_head"] != EXPECTED_CHIMERA_HEAD
        or source["chimeraboost_status"]
    ):
        raise RuntimeError("frozen ChimeraBoost source changed")
    runtime = _runtime_record()
    for key in ("python", "platform", "machine", "logical_cpu_count"):
        if runtime[key] != EXPECTED_RUNTIME[key]:
            raise RuntimeError(f"native-ordinal C2 runtime changed: {key}")
    if runtime["dependencies"] != EXPECTED_RUNTIME["dependencies"]:
        raise RuntimeError("native-ordinal C2 dependency versions changed")
    registry = _load_registry()
    return source, registry


def _atomic_create(path: Path, payload: bytes) -> None:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cache_stats(path: Path) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    compiled = [item for item in files if item.suffix in {".nbc", ".nbi"}]
    return {
        "file_count": len(files),
        "compiled_file_count": len(compiled),
        "bytes": int(sum(item.stat().st_size for item in files)),
    }


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("worker peak RSS is unavailable")
    return value


def _warning_records(
    caught: list[warnings.WarningMessage],
) -> list[dict[str, str]]:
    return [
        {"category": item.category.__name__, "message": str(item.message)}
        for item in caught
    ]


def _task_row(
    registry: dict[str, Any], tier: str, task_id: int
) -> dict[str, Any]:
    rows = registry[f"{tier}_tasks"]
    matches = [row for row in rows if int(row["task_id"]) == int(task_id)]
    if len(matches) != 1:
        raise RuntimeError(f"task {task_id} is not uniquely frozen for {tier}")
    if tier == CONFIRMATION and matches[0].get("status") != "eligible":
        raise RuntimeError(f"confirmation task {task_id} is not eligible")
    return matches[0]


def _load_task(row: dict[str, Any]):
    import openml

    task_id = int(row["task_id"])
    task = openml.tasks.get_task(task_id, download_splits=True)
    dataset = task.get_dataset()
    X, y, categorical, names = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )
    expected = row["task_record"]
    if (
        int(dataset.dataset_id) != int(row["dataset_id"])
        or str(dataset.name) != str(row["dataset_name"])
        or str(task.target_name) != str(row["target_name"])
        or str(dataset.md5_checksum) != str(expected["openml_declared_md5"])
        or X.shape
        != (
            int(expected["fingerprint"]["n_rows"]),
            int(expected["fingerprint"]["n_features"]),
        )
    ):
        raise RuntimeError(f"native-ordinal C2 task {task_id} identity changed")
    if list(names) != row["feature_record"]["feature_names"]:
        raise RuntimeError(f"native-ordinal C2 task {task_id} schema changed")
    if ctr.dataset_fingerprint(X, y) != expected["fingerprint"]:
        raise RuntimeError(
            f"native-ordinal C2 task {task_id} fingerprint changed"
        )
    model_categorical, inferred = (
        registry_builder._model_categorical_indices(X, list(categorical))
    )
    feature_record = row["feature_record"]
    if (
        model_categorical != feature_record["categorical_indices"]
        or inferred
        != feature_record["inferred_nonnumeric_categorical_indices"]
    ):
        raise RuntimeError(
            f"native-ordinal C2 task {task_id} categorical policy changed"
        )
    y = pd.to_numeric(y, errors="raise").astype(np.float64)
    if not np.all(np.isfinite(y.to_numpy())):
        raise RuntimeError(f"native-ordinal C2 task {task_id} target is invalid")
    return task, X, y, model_categorical


def _expected_outer_split(row: dict[str, Any], fold: int) -> dict[str, Any]:
    coordinates = row["task_record"]["official_splits"]["coordinates"]
    matches = [
        item
        for item in coordinates
        if int(item["repeat"]) == 0
        and int(item["fold"]) == int(fold)
        and int(item["sample"]) == 0
    ]
    if len(matches) != 1:
        raise RuntimeError("native-ordinal C2 split is not uniquely frozen")
    return matches[0]


def _verify_outer_split(
    row: dict[str, Any], fold: int, train: Any, test: Any
) -> dict[str, Any]:
    observed = {
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "train_index_sha256": _array_sha256(train, "<i8"),
        "test_index_sha256": _array_sha256(test, "<i8"),
    }
    expected = _expected_outer_split(row, fold)
    for key, value in observed.items():
        if value != expected[key]:
            raise RuntimeError(
                f"native-ordinal C2 task {row['task_id']} fold {fold} "
                f"{key} changed"
            )
    return observed


def development_split(
    outer_train: Any, *, task_id: int, fold: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    outer_train = np.asarray(outer_train, dtype=np.int64)
    if outer_train.ndim != 1 or len(outer_train) < 5:
        raise ValueError("native-ordinal C2 outer training split is invalid")
    seed = int(SPLIT_SEED + int(task_id) + int(fold))
    order = np.random.default_rng(seed).permutation(len(outer_train))
    validation_rows = max(
        1, int(math.ceil(VALIDATION_FRACTION * len(outer_train)))
    )
    validation_positions = np.sort(order[-validation_rows:])
    fit_positions = np.sort(order[:-validation_rows])
    fit = outer_train[fit_positions]
    validation = outer_train[validation_positions]
    if (
        len(fit) + len(validation) != len(outer_train)
        or np.intersect1d(fit, validation).size
        or not np.array_equal(
            np.sort(np.concatenate((fit, validation))),
            np.sort(outer_train),
        )
    ):
        raise RuntimeError("native-ordinal C2 inner split is invalid")
    return fit, validation, {
        "policy": "seeded_permutation_tail_20_percent",
        "seed": seed,
        "validation_fraction": VALIDATION_FRACTION,
        "fit_rows": int(len(fit)),
        "validation_rows": int(len(validation)),
        "fit_index_sha256": _array_sha256(fit, "<i8"),
        "validation_index_sha256": _array_sha256(validation, "<i8"),
    }


def _ordinal_state(model: Any) -> dict[str, Any]:
    return {
        "mode": getattr(model, "ordinal_features_mode_", None),
        "records": copy.deepcopy(getattr(model, "ordinal_features_", None)),
        "indices": np.asarray(
            getattr(model, "ordinal_feature_indices_", ()), dtype=np.int64
        ).tolist(),
        "metadata": copy.deepcopy(
            model.model_.auto_params_.get("ordinal_features")
        ),
    }


def _preprocessor_state(model: Any) -> dict[str, Any]:
    prep = model.model_.prep_
    binner = prep.binner_
    return {
        "n_input_features": int(prep.n_input_features_),
        "num_features": [int(value) for value in prep.num_features_],
        "cat_features": [int(value) for value in prep.cat_features_],
        "feature_map": np.asarray(prep.feature_map_, dtype=np.int64).tolist(),
        "n_bins": np.asarray(binner.n_bins_, dtype=np.int64).tolist(),
        "borders_flat_sha256": _array_sha256(
            binner._borders_flat_, "<f8"
        ),
        "border_offsets": np.asarray(
            binner._border_offsets_, dtype=np.int64
        ).tolist(),
        "block_widths": [
            int(value) for value in getattr(binner, "_block_widths_", ())
        ],
        "encoder_count": len(getattr(prep, "encoders_", ())),
        "target_stat_blocks": len(getattr(prep, "encoders_", ())),
    }


def _normalized_header(header: dict[str, Any], arm: str) -> dict[str, Any]:
    normalized = copy.deepcopy(header)
    normalized.pop("timing", None)
    diagnostics = normalized.get("auto_params", {}).get("diagnostics")
    if isinstance(diagnostics, dict):
        diagnostics.pop("runtime_warnings_emitted", None)
    if arm == CANDIDATE:
        auto_params = normalized.get("auto_params")
        state = normalized.get("wrapper", {}).get("state", {})
        if not isinstance(auto_params, dict) or not isinstance(state, dict):
            raise RuntimeError("candidate archive ordinal state is absent")
        ordinal = auto_params.pop("ordinal_features", None)
        diagnostics = auto_params.get("diagnostics")
        diagnostic_ordinal = (
            diagnostics.pop("ordinal_features", None)
            if isinstance(diagnostics, dict)
            else None
        )
        mode = state.pop("ordinal_features_mode", None)
        records = state.pop("ordinal_features", None)
        if (
            not isinstance(ordinal, dict)
            or diagnostic_ordinal != ordinal
            or mode != "explicit"
            or not isinstance(records, list)
        ):
            raise RuntimeError("candidate archive ordinal metadata changed")
    return normalized


def _archive_identity(model: Any, arm: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="darkofit-c2-archive-") as root:
        path = Path(root) / "model.npz"
        model.save_model(path)
        raw = path.read_bytes()
        digest = hashlib.sha256()
        with np.load(path, allow_pickle=False) as archive:
            header = json.loads(archive["header"].item())
            encoded = _canonical_json(_normalized_header(header, arm))
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
            for name in sorted(item for item in archive.files if item != "header"):
                values = np.asarray(archive[name])
                for payload in (
                    name.encode("utf-8"),
                    values.dtype.str.encode("ascii"),
                    _canonical_json(list(values.shape)),
                    np.ascontiguousarray(values).tobytes(),
                ):
                    digest.update(len(payload).to_bytes(8, "little"))
                    digest.update(payload)
    return {
        "raw_bytes": len(raw),
        "raw_sha256": _sha256_bytes(raw),
        "normalized_logical_sha256": digest.hexdigest(),
    }


def _fit_model(
    arm: str,
    row: dict[str, Any],
    X_fit: Any,
    y_fit: Any,
    categorical_indices: list[int],
):
    from darkofit import DarkoRegressor

    model = DarkoRegressor(random_state=RANDOM_STATE)
    started = time.perf_counter_ns()
    if arm == CONTROL:
        model.fit(X_fit, y_fit, cat_features=categorical_indices)
    elif arm == CANDIDATE:
        model.fit(
            X_fit,
            y_fit,
            cat_features=categorical_indices,
            ordinal_features=row["ordinal_features"],
        )
    else:
        raise ValueError(f"unknown native-ordinal C2 arm {arm!r}")
    seconds = (time.perf_counter_ns() - started) / 1e9
    return model, float(seconds)


def _prediction(model: Any, X: Any, y: Any) -> dict[str, Any]:
    prediction = np.asarray(model.predict(X), dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    if (
        prediction.shape != target.shape
        or prediction.ndim != 1
        or not np.all(np.isfinite(prediction))
    ):
        raise RuntimeError("native-ordinal C2 prediction is invalid")
    rmse = float(np.sqrt(np.mean(np.square(target - prediction))))
    if not math.isfinite(rmse) or rmse < 0.0:
        raise RuntimeError("native-ordinal C2 RMSE is invalid")
    return {
        "rows": int(len(prediction)),
        "rmse": rmse,
        "prediction_sha256": _array_sha256(prediction, "<f8"),
    }


def _timed_prediction(model: Any, X: Any, expected_hash: str) -> dict[str, Any]:
    started = time.perf_counter_ns()
    prediction = None
    for _ in range(PREDICTION_CALLS):
        prediction = model.predict(X)
    seconds = (time.perf_counter_ns() - started) / 1e9
    digest = _array_sha256(prediction, "<f8")
    if digest != expected_hash:
        raise RuntimeError("timed native-ordinal C2 prediction changed")
    return {
        "calls": PREDICTION_CALLS,
        "total_seconds": float(seconds),
        "seconds_per_call": float(seconds / PREDICTION_CALLS),
        "last_prediction_sha256": digest,
    }


def _behavior_payload(result: dict[str, Any]) -> dict[str, Any]:
    behavior = copy.deepcopy(result)
    behavior.pop("behavior_fingerprint_sha256", None)
    for key in (
        "worker_returncode",
        "worker_stdout",
        "worker_stderr",
        "coordinate_index",
        "position",
        "fit_seconds",
        "peak_rss_bytes",
    ):
        behavior.pop(key, None)
    behavior.get("public_predict_timing", {}).pop("total_seconds", None)
    behavior.get("public_predict_timing", {}).pop(
        "seconds_per_call", None
    )
    return behavior


def run_worker(
    tier: str,
    task_id: int,
    fold: int,
    arm: str,
    cache_dir: Path,
    authorization: Path | None,
) -> dict[str, Any]:
    if tier not in TIERS or arm not in ARMS:
        raise ValueError("native-ordinal C2 worker tier or arm is invalid")
    if os.environ.get("DARKOFIT_WARMUP") != "0":
        raise RuntimeError("worker import warmup must be disabled")
    if Path(os.environ.get("NUMBA_CACHE_DIR", "")) != cache_dir:
        raise RuntimeError("worker Numba cache binding changed")
    if _cache_stats(cache_dir)["file_count"]:
        raise RuntimeError("worker Numba cache was not empty")

    source, registry = require_frozen_source()
    authorization_record = _validate_authorization(
        tier, authorization, registry
    )
    row = _task_row(registry, tier, task_id)
    before_import = _cache_stats(cache_dir)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import darkofit

        module_path = Path(darkofit.__file__).resolve()
        if not module_path.is_relative_to(ROOT.resolve()):
            raise RuntimeError("darkofit imported outside frozen repository")
        after_import = _cache_stats(cache_dir)
        warmup_started = time.perf_counter_ns()
        returned_warmup = float(darkofit.warmup())
        warmup_seconds = (time.perf_counter_ns() - warmup_started) / 1e9
        if not math.isclose(
            returned_warmup,
            warmup_seconds,
            rel_tol=0.05,
            abs_tol=0.05,
        ):
            raise RuntimeError("public warmup returned inconsistent timing")
        after_warmup = _cache_stats(cache_dir)
        if after_warmup["compiled_file_count"] <= 0:
            raise RuntimeError("explicit warmup did not populate Numba cache")

        task, X, y, categorical = _load_task(row)
        outer_train, outer_test = task.get_train_test_split_indices(
            repeat=0, fold=int(fold), sample=0
        )
        outer = _verify_outer_split(
            row, fold, outer_train, outer_test
        )
        fit_indices, validation_indices, inner = development_split(
            outer_train, task_id=task_id, fold=fold
        )
        model, fit_seconds = _fit_model(
            arm,
            row,
            X.iloc[fit_indices],
            y.iloc[fit_indices],
            categorical,
        )
        validation = _prediction(
            model, X.iloc[validation_indices], y.iloc[validation_indices]
        )
        test = _prediction(model, X.iloc[outer_test], y.iloc[outer_test])
        timed = _timed_prediction(
            model, X.iloc[outer_test], test["prediction_sha256"]
        )
        model_record = {
            "fit_metadata": basketball.extract_fit_metadata(model),
            "ordinal_state": _ordinal_state(model),
            "preprocessor": _preprocessor_state(model),
            "archive": _archive_identity(model, arm),
        }

    result = {
        "ok": True,
        "tier": tier,
        "task_id": int(task_id),
        "dataset_id": int(row["dataset_id"]),
        "dataset_name": row["dataset_name"],
        "lineage_cluster": row["lineage_cluster"],
        "role": row["role"],
        "fold": int(fold),
        "arm": arm,
        "source": source,
        "runtime": _runtime_record(),
        "registry_sha256": registry["registry_sha256"],
        "authorization": authorization_record,
        "categorical_indices": categorical,
        "declared_ordinal_features": row["ordinal_features"],
        "outer_split": outer,
        "inner_split": inner,
        "warmup_seconds": float(warmup_seconds),
        "warmup_returned_seconds": float(returned_warmup),
        "fit_seconds": fit_seconds,
        "validation": validation,
        "test": test,
        "public_predict_timing": timed,
        "peak_rss_bytes": _peak_rss_bytes(),
        "model": model_record,
        "warnings": _warning_records(caught),
        "cache": {
            "before_import": before_import,
            "after_import": after_import,
            "after_warmup": after_warmup,
            "after_workload": _cache_stats(cache_dir),
        },
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "DARKOFIT_WARMUP",
                "NUMBA_CACHE_DIR",
                "NUMBA_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }
    result["behavior_fingerprint_sha256"] = _json_sha256(
        _behavior_payload(result)
    )
    return result


def _validate_authorization(
    tier: str,
    path: Path | None,
    registry: dict[str, Any],
) -> dict[str, Any] | None:
    if tier == DEVELOPMENT:
        if path is not None:
            raise RuntimeError("development tier must not use authorization")
        return None
    if path is None:
        raise RuntimeError("confirmation tier requires authorization")
    path = path.expanduser().absolute()
    if path != (
        ROOT / "benchmarks" / "native_ordinal_c2_development_result.json"
    ).absolute():
        raise RuntimeError(
            "confirmation requires the frozen development result path"
        )
    payload = path.read_bytes()
    record = json.loads(payload)
    if (
        record.get("decision")
        != "authorize_native_ordinal_c2_confirmation_once"
        or record.get("passes") is not True
        or record.get("confirmation_run_authorized") is not True
        or record.get("registry_sha256") != registry["registry_sha256"]
        or record.get("lockbox_touched") is not False
        or record.get("tier") != DEVELOPMENT
        or record.get("development_outcomes_inspected") is not True
        or record.get("confirmation_outcomes_inspected") is not False
        or not isinstance(record.get("raw_sha256"), str)
        or len(record["raw_sha256"]) != 64
    ):
        raise RuntimeError("native-ordinal C2 confirmation is not authorized")
    return {
        "path": str(path),
        "sha256": _sha256_bytes(payload),
        "decision": record["decision"],
        "development_raw_sha256": record.get("raw_sha256"),
    }


def _worker_environment(cache_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    threads = str(THREADS_PER_WORKER)
    environment.update(
        {
            "DARKOFIT_WARMUP": "0",
            "CHIMERABOOST_WARMUP": "0",
            "NUMBA_CACHE_DIR": str(cache_dir),
            "NUMBA_NUM_THREADS": threads,
            "OMP_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "VECLIB_MAXIMUM_THREADS": threads,
            "PYTHONHASHSEED": "0",
            "OPENML_CACHE_DIRECTORY": str(Path.home() / ".cache" / "openml"),
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT), environment.get("PYTHONPATH", "")]
            ),
        }
    )
    return environment


def _run_worker_process(
    tier: str,
    coordinate: dict[str, Any],
    arm: str,
    *,
    coordinate_index: int,
    position: int,
    authorization: Path | None,
) -> dict[str, Any]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(
        tempfile.mkdtemp(
            prefix=(
                f"{tier}-c{coordinate_index}-p{position}-{arm}-"
            ),
            dir=CACHE_ROOT,
        )
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-tier",
        tier,
        "--worker-task",
        str(coordinate["task_id"]),
        "--worker-fold",
        str(coordinate["fold"]),
        "--worker-arm",
        arm,
        "--cache-dir",
        str(cache_dir),
    ]
    if authorization is not None:
        command.extend(["--authorization", str(authorization)])
    try:
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=_worker_environment(cache_dir),
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            return {
                "ok": False,
                "tier": tier,
                "task_id": int(coordinate["task_id"]),
                "fold": int(coordinate["fold"]),
                "arm": arm,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "worker_returncode": None,
                "worker_stdout": None,
                "worker_stderr": None,
                "coordinate_index": int(coordinate_index),
                "position": int(position),
            }
        lines = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith(WORKER_RESULT_PREFIX)
        ]
        if len(lines) == 1:
            result = json.loads(lines[0][len(WORKER_RESULT_PREFIX) :])
        else:
            result = {
                "ok": False,
                "tier": tier,
                "task_id": int(coordinate["task_id"]),
                "fold": int(coordinate["fold"]),
                "arm": arm,
                "error_type": "WorkerProtocolError",
                "error": (
                    f"expected one worker result line, found {len(lines)}"
                ),
            }
        result["worker_returncode"] = int(completed.returncode)
        result["worker_stdout"] = (
            "\n".join(
                line
                for line in completed.stdout.splitlines()
                if not line.startswith(WORKER_RESULT_PREFIX)
            ).strip()
            or None
        )
        result["worker_stderr"] = completed.stderr.strip() or None
        result["coordinate_index"] = int(coordinate_index)
        result["position"] = int(position)
        return result
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def reciprocal_order(coordinate_index: int) -> tuple[str, str]:
    return (
        (CONTROL, CANDIDATE)
        if int(coordinate_index) % 2 == 0
        else (CANDIDATE, CONTROL)
    )


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().absolute()
    expected_output = (
        DEFAULT_DEVELOPMENT_OUTPUT
        if args.tier == DEVELOPMENT
        else DEFAULT_CONFIRMATION_OUTPUT
    ).absolute()
    if output != expected_output:
        raise RuntimeError(
            "formal native-ordinal C2 execution requires its frozen output path"
        )
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing existing output: {output}")
    if (
        args.threads_per_worker != THREADS_PER_WORKER
        or args.concurrent_workers != CONCURRENT_WORKERS
        or args.predict_calls != PREDICTION_CALLS
    ):
        raise RuntimeError("native-ordinal C2 execution settings changed")
    source, registry = require_frozen_source()
    authorization = _validate_authorization(
        args.tier, args.authorization, registry
    )
    coordinates = registry["coordinates"][args.tier]
    expected_count = 24 if args.tier == DEVELOPMENT else 15
    if len(coordinates) != expected_count:
        raise RuntimeError("native-ordinal C2 coordinate count changed")

    results = []
    for wave_start in range(0, len(coordinates), CONCURRENT_WORKERS):
        wave = list(enumerate(
            coordinates[wave_start : wave_start + CONCURRENT_WORKERS],
            start=wave_start,
        ))
        for position in range(2):
            if _source_state() != source:
                raise RuntimeError("source changed during native-ordinal C2 run")
            print(
                f"[native-ordinal-c2] tier={args.tier} "
                f"wave={wave_start // CONCURRENT_WORKERS} "
                f"position={position}",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
                futures = [
                    executor.submit(
                        _run_worker_process,
                        args.tier,
                        coordinate,
                        reciprocal_order(index)[position],
                        coordinate_index=index,
                        position=position,
                        authorization=args.authorization,
                    )
                    for index, coordinate in wave
                ]
                results.extend(future.result() for future in futures)
    if _source_state() != source:
        raise RuntimeError("source changed during native-ordinal C2 run")

    results.sort(key=lambda row: (row["coordinate_index"], row["position"]))
    artifact = {
        "schema_version": 1,
        "name": f"darkofit_native_ordinal_c2_{args.tier}_raw_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tier": args.tier,
        "source": source,
        "runtime": _runtime_record(),
        "registry": {
            "path": str(REGISTRY_PATH.relative_to(ROOT)),
            "file_sha256": EXPECTED_REGISTRY_FILE_SHA256,
            "content_sha256": EXPECTED_REGISTRY_CONTENT_SHA256,
        },
        "protocol": {
            "path": str(PROTOCOL_PATH.relative_to(ROOT)),
            "sha256": EXPECTED_PROTOCOL_SHA256,
        },
        "runner": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "normalized_sha256": EXPECTED_NORMALIZED_RUNNER_SHA256,
        },
        "analyzer": {
            "path": str(ANALYZER_PATH.relative_to(ROOT)),
            "sha256": EXPECTED_ANALYZER_SHA256,
        },
        "execution": {
            "threads_per_worker": THREADS_PER_WORKER,
            "concurrent_workers": CONCURRENT_WORKERS,
            "prediction_calls": PREDICTION_CALLS,
            "coordinate_count": len(coordinates),
            "worker_count": len(results),
            "reciprocal_order": "alternating_by_coordinate_index",
            "partial_resume_supported": False,
        },
        "authorization": authorization,
        "results": results,
        "development_outcomes_inspected": args.tier == DEVELOPMENT,
        "confirmation_outcomes_inspected": args.tier == CONFIRMATION,
        "lockbox_touched": False,
    }
    _atomic_create(
        output,
        (
            json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8"),
    )
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=TIERS, default=DEVELOPMENT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument(
        "--threads-per-worker", type=int, default=THREADS_PER_WORKER
    )
    parser.add_argument(
        "--concurrent-workers", type=int, default=CONCURRENT_WORKERS
    )
    parser.add_argument("--predict-calls", type=int, default=PREDICTION_CALLS)
    parser.add_argument("--worker-tier", choices=TIERS)
    parser.add_argument("--worker-task", type=int)
    parser.add_argument("--worker-fold", type=int)
    parser.add_argument("--worker-arm", choices=ARMS)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    worker_values = (
        args.worker_tier,
        args.worker_task,
        args.worker_fold,
        args.worker_arm,
        args.cache_dir,
    )
    if any(value is not None for value in worker_values) and not all(
        value is not None for value in worker_values
    ):
        parser.error("all worker arguments must be supplied together")
    if args.output is None:
        args.output = (
            DEFAULT_DEVELOPMENT_OUTPUT
            if args.tier == DEVELOPMENT
            else DEFAULT_CONFIRMATION_OUTPUT
        )
    return args


def main() -> int:
    args = parse_args()
    if args.worker_tier is not None:
        try:
            result = run_worker(
                args.worker_tier,
                args.worker_task,
                args.worker_fold,
                args.worker_arm,
                args.cache_dir,
                args.authorization,
            )
            returncode = 0
        except Exception as exc:
            result = {
                "ok": False,
                "tier": args.worker_tier,
                "task_id": args.worker_task,
                "fold": args.worker_fold,
                "arm": args.worker_arm,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            returncode = 1
        print(
            WORKER_RESULT_PREFIX
            + json.dumps(result, sort_keys=True, allow_nan=False),
            flush=True,
        )
        return returncode

    artifact = run_parent(args)
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().absolute()),
                "sha256": _sha256_file(args.output.expanduser().absolute()),
                "tier": args.tier,
                "workers": len(artifact["results"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
