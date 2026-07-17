#!/usr/bin/env python3
"""Run the frozen basketball native-ordinal no-engagement campaign."""

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
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import r2_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import basketball_harness as harness  # noqa: E402
from benchmarks import run_basketball_creator_benchmark as creator  # noqa: E402


CONTROL = "control"
CANDIDATE = "candidate"
ARMS = (CONTROL, CANDIDATE)
TIMING_BLOCKS = 3
BLOCK_ORDERS = harness.reciprocal_schedule(
    CONTROL, CANDIDATE, repetitions=TIMING_BLOCKS
)
EXPECTED_THREADS = 18
HELD_PREDICTION_CALLS = 200
COLD_PREDICTION_CALLS = 500

IMPLEMENTATION_COMMIT = "ceb96d191e316ab5f88204cfe767bc96f78239e8"
IMPLEMENTATION_PACKAGE_TREE = "d4661e0d4a919d2e0f0da4385b12034a5c853a6c"
PROTOCOL_COMMIT = "ae0e25aaf7fb44ab07d4cfd690ac33ddf47055ef"
PROTOCOL_PATH = (
    REPO_ROOT / "benchmarks" / "basketball_native_ordinal_protocol.md"
)
ANALYZER_PATH = (
    REPO_ROOT / "benchmarks" / "analyze_basketball_native_ordinal.py"
)
EXPECTED_PROTOCOL_SHA256 = (
    "35f260c6fd2403195ce8dfda658bc0bb7ad90af8ee74383a539eb3c1ca491b9b"
)
EXPECTED_NORMALIZED_RUNNER_SHA256 = (
    "fa09739b220201aced753b775826a3f41e814ddeca19a4ecd2a9c3c8d2a3be75"
)
EXPECTED_SUPPORT_SHA256 = {
    "benchmarks/basketball_campaign_harness.py": (
        "14df91eb9c99912bd0fdf5bce81434934ff2ae84d588006a34fb513743283433"
    ),
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
EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "platform": "macOS-26.5.2-arm64-arm-64bit",
    "machine": "arm64",
    "logical_cpu_count": 18,
    "dependencies": {
        "numpy": "2.4.6",
        "pandas": "3.0.3",
        "scikit-learn": "1.9.0",
        "numba": "0.66.0",
        "llvmlite": "0.48.0",
        "joblib": "1.5.3",
        "threadpoolctl": "3.6.0",
    },
}
EXPECTED_DATA = {
    "raw_bytes": 2_549_434,
    "raw_sha256": (
        "43af3be46074da9130a757aa1db643e207e2c0ea5405d2086d698f85555fbcb2"
    ),
    "x_sha256": (
        "05efa554a983942885b72b1b3fdcc97a10ecf4ccbb4b10843ee8b970129fe71b"
    ),
    "y_sha256": (
        "7c06b45d4649c392fdb0c3dc91a857650f2f3fc77341fe06ffdbee2b3a44aecf"
    ),
    "fold_sha256": (
        "7270350a38a687b6e783d18d9c02e5a45f2c7367faa4a6955e74c710f5e8daea"
    ),
    "cold_mask_sha256": (
        "e17887c9240fd95aee21d37d7e1e8a533c38ef87f4492acd28fb3aa496a3dd19"
    ),
}
EXPECTED_PREDICTION_SHA256 = {
    "folds": [
        "6200db22da190d8c0787d7794c1fb8d859af737ab7e3026716e23aa1be95125f",
        "96ad500c63ac3701fe769b03a369d3a01ed1af9695d71c7ea68936d36479da44",
        "230b3cb530dee9ba8f5196b2b12b77f8d62751c545828ca13bad3fe04e54261b",
        "4603c6b3036bbdee060faaa92e6eee18a1f803e4abe9bc4aa7906745db5bd1c1",
        "e00b84d4aa7b8640aad72f5aed6e5e578cef2035459aa146b972145dc8d19fef",
        "12852587a9d1cd729cde1b28d714ff0c30b8051e806d6bb2f3f68088f22912d8",
        "514663b32f0adaf0fc7591def75632f5ea1103598b2d7aaeeaf37fdc2560bb04",
        "45374906a6931f90a6fff29ba0544c4d66311bb6152e3f250d54db55e0c03384",
        "32167d2ad1ba4ee34297a812be85ae67675f638383d61fe709b130bdbb3931a5",
        "f51972e8f896568291b259d698726b224a2399711f8e8cdf451e68b5090ae38d",
    ],
    "held": (
        "5d910ae8f6b0dca563b99f9f881dcb17ee092711a46b2890452eaa3b8e68367a"
    ),
    "cold": (
        "998a14f530ed284865a50726191da067f72d69da3001614d664a4b90e7aa6376"
    ),
    "seen": (
        "c9b506afbfb3eb660dd918ee9635d996c0285b0320ba250cbf39c80df9122425"
    ),
}
EXPECTED_MEAN_R2 = 0.5267495183883605
EXPECTED_FOLD_LEARNING_RATES = (0.052312, *([0.052314] * 9))
EXPECTED_GUARDRAIL_LEARNING_RATE = 0.053192
EXPECTED_TREE_COUNT = 1_000

WORKER_RESULT_PREFIX = "BASKETBALL_NATIVE_ORDINAL_RESULT="
DEFAULT_OUTPUT = (
    REPO_ROOT / "benchmarks" / "basketball_native_ordinal_raw.json"
)
CACHE_ROOT = REPO_ROOT / ".cache" / "basketball-native-ordinal"


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
        raise RuntimeError("runner normalization field changed")
    return _sha256_bytes(normalized)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args], text=True
    ).strip()


def _is_ancestor(commit: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
            check=False,
        ).returncode
        == 0
    )


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
        "branch": _git("branch", "--show-current"),
        "origin_main": _git("rev-parse", "origin/main"),
        "status": _git("status", "--porcelain=v1", "--untracked-files=all"),
        "package_tree": _git("rev-parse", "HEAD:darkofit"),
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "runner_normalized_sha256": _normalized_runner_sha256(),
        "analyzer_sha256": (
            _sha256_file(ANALYZER_PATH) if ANALYZER_PATH.is_file() else None
        ),
        "support_sha256": {
            relative: _sha256_file(REPO_ROOT / relative)
            for relative in EXPECTED_SUPPORT_SHA256
        },
    }


def require_frozen_source() -> dict[str, Any]:
    state = _source_state()
    if state["branch"] != "main" or state["head"] != state["origin_main"]:
        raise RuntimeError("formal native-ordinal campaign requires pushed main")
    if state["status"]:
        raise RuntimeError("formal native-ordinal campaign requires clean source")
    if not _is_ancestor(IMPLEMENTATION_COMMIT):
        raise RuntimeError("native-ordinal implementation commit is not an ancestor")
    if not _is_ancestor(PROTOCOL_COMMIT):
        raise RuntimeError("native-ordinal protocol commit is not an ancestor")
    if state["package_tree"] != IMPLEMENTATION_PACKAGE_TREE:
        raise RuntimeError("DarkoFit package tree changed after C1 implementation")
    if state["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("native-ordinal protocol changed")
    if state["runner_normalized_sha256"] != EXPECTED_NORMALIZED_RUNNER_SHA256:
        raise RuntimeError("native-ordinal runner changed")
    if state["support_sha256"] != EXPECTED_SUPPORT_SHA256:
        raise RuntimeError("native-ordinal support files changed")
    if state["analyzer_sha256"] is None:
        raise RuntimeError("native-ordinal analyzer is missing")
    runtime = _runtime_record()
    for key in ("python", "platform", "machine", "logical_cpu_count"):
        if runtime[key] != EXPECTED_RUNTIME[key]:
            raise RuntimeError(f"native-ordinal runtime changed: {key}")
    if runtime["dependencies"] != EXPECTED_RUNTIME["dependencies"]:
        raise RuntimeError("native-ordinal dependency versions changed")
    return state


def _atomic_create(path: Path, payload: bytes) -> None:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
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
        "compiled_files": [str(item.relative_to(path)) for item in compiled],
    }


def _validate_data_cache(path: Path) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(
            "formal native-ordinal campaign requires a regular cached dataset"
        )
    if path.stat().st_size != EXPECTED_DATA["raw_bytes"]:
        raise RuntimeError("basketball cached byte count changed")
    if _sha256_file(path) != EXPECTED_DATA["raw_sha256"]:
        raise RuntimeError("basketball cached hash changed")
    return path


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


def _fit_model(arm: str, X: Any, y: Any):
    from darkofit import DarkoRegressor

    model = DarkoRegressor(random_state=creator.RANDOM_STATE)
    started = time.perf_counter_ns()
    if arm == CONTROL:
        model.fit(X, y)
    elif arm == CANDIDATE:
        model.fit(X, y, ordinal_features="auto")
    else:
        raise ValueError(f"unknown native-ordinal arm {arm!r}")
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    return model, float(fit_seconds)


def _ordinal_state(model: Any) -> dict[str, Any]:
    metadata = model.model_.auto_params_.get("ordinal_features")
    return {
        "mode": getattr(model, "ordinal_features_mode_", None),
        "records": copy.deepcopy(getattr(model, "ordinal_features_", None)),
        "indices": np.asarray(
            getattr(model, "ordinal_feature_indices_", ()), dtype=np.int64
        ).tolist(),
        "metadata": copy.deepcopy(metadata),
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
        "borders_flat_sha256": _sha256_bytes(
            np.ascontiguousarray(
                np.asarray(binner._borders_flat_, dtype="<f8")
            ).tobytes()
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
    auto_params = normalized.get("auto_params")
    state = (
        normalized.get("wrapper", {}).get("state", {})
        if isinstance(normalized.get("wrapper"), dict)
        else {}
    )
    if arm == CANDIDATE:
        if not isinstance(auto_params, dict):
            raise RuntimeError("candidate archive lacks auto_params")
        ordinal = auto_params.pop("ordinal_features", None)
        diagnostics = auto_params.get("diagnostics")
        diagnostic_ordinal = (
            diagnostics.pop("ordinal_features", None)
            if isinstance(diagnostics, dict)
            else None
        )
        mode = state.pop("ordinal_features_mode", None)
        records = state.pop("ordinal_features", None)
        if ordinal is None or diagnostic_ordinal != ordinal:
            raise RuntimeError("candidate archive ordinal metadata changed")
        if mode != "auto" or records != []:
            raise RuntimeError("candidate archive ordinal wrapper state changed")
    else:
        if (
            isinstance(auto_params, dict)
            and "ordinal_features" in auto_params
        ):
            raise RuntimeError("control archive contains ordinal metadata")
        if isinstance(auto_params, dict):
            diagnostics = auto_params.get("diagnostics")
            if isinstance(diagnostics, dict) and "ordinal_features" in diagnostics:
                raise RuntimeError("control archive diagnostics contain ordinal metadata")
        if "ordinal_features_mode" in state or "ordinal_features" in state:
            raise RuntimeError("control archive contains ordinal wrapper state")
    return normalized


def _archive_identity(model: Any, arm: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="darkofit-native-ordinal-archive-"
    ) as root:
        path = Path(root) / "model.npz"
        model.save_model(path)
        raw = path.read_bytes()
        digest = hashlib.sha256()
        with np.load(path, allow_pickle=False) as archive:
            header = json.loads(archive["header"].item())
            normalized = _normalized_header(header, arm)
            encoded_header = _canonical_json(normalized)
            digest.update(len(encoded_header).to_bytes(8, "little"))
            digest.update(encoded_header)
            members = sorted(name for name in archive.files if name != "header")
            for name in members:
                values = np.asarray(archive[name])
                name_bytes = name.encode("utf-8")
                dtype_bytes = values.dtype.str.encode("ascii")
                shape_bytes = _canonical_json(list(values.shape))
                value_bytes = np.ascontiguousarray(values).tobytes()
                for payload in (
                    name_bytes,
                    dtype_bytes,
                    shape_bytes,
                    value_bytes,
                ):
                    digest.update(len(payload).to_bytes(8, "little"))
                    digest.update(payload)
    return {
        "raw_bytes": len(raw),
        "raw_sha256": _sha256_bytes(raw),
        "normalized_sha256": digest.hexdigest(),
        "normalized_removed_fields": (
            [
                "auto_params.ordinal_features",
                "auto_params.diagnostics.ordinal_features",
                "wrapper.state.ordinal_features_mode",
                "wrapper.state.ordinal_features",
                "timing",
            ]
            if arm == CANDIDATE
            else ["timing"]
        ),
    }


def _fit_observation(model: Any, arm: str) -> dict[str, Any]:
    importance = np.ascontiguousarray(
        np.asarray(model.feature_importances_, dtype="<f8")
    )
    return {
        "fit_metadata": harness.extract_fit_metadata(model),
        "ordinal_state": _ordinal_state(model),
        "preprocessor": _preprocessor_state(model),
        "feature_importance": importance.tolist(),
        "feature_importance_sha256": _sha256_bytes(importance.tobytes()),
        "archive": _archive_identity(model, arm),
    }


def _prediction_record(target: Any, prediction: Any) -> dict[str, Any]:
    prediction = harness.validate_prediction(prediction, len(target))
    return {
        "r2": float(r2_score(target, prediction)),
        "prediction_sha256": harness.prediction_sha256(prediction),
        "predictions": prediction.tolist(),
    }


def _behavior_payload(result: dict[str, Any]) -> dict[str, Any]:
    def model_payload(observation: dict[str, Any]) -> dict[str, Any]:
        archive = observation["archive"]
        return {
            "fit_metadata": observation["fit_metadata"],
            "ordinal_state": observation["ordinal_state"],
            "preprocessor": observation["preprocessor"],
            "feature_importance": observation["feature_importance"],
            "feature_importance_sha256": observation[
                "feature_importance_sha256"
            ],
            "archive": {
                "normalized_sha256": archive["normalized_sha256"],
                "normalized_removed_fields": archive[
                    "normalized_removed_fields"
                ],
            },
        }

    guardrail = result["guardrail"]
    return {
        "arm": result["arm"],
        "folds": [
            {
                "fold": row["fold"],
                "test_indices": row["test_indices"],
                "prediction_sha256": row["prediction"]["prediction_sha256"],
                "predictions": row["prediction"]["predictions"],
                "model": model_payload(row["model"]),
            }
            for row in result["folds"]
        ],
        "guardrail": {
            "model": model_payload(guardrail["model"]),
            "held": guardrail["held"],
            "cold": guardrail["cold"],
            "seen": guardrail["seen"],
            "scores_from_full_prediction": guardrail[
                "scores_from_full_prediction"
            ],
        },
        "runtime": result["runtime"],
        "data": result["data"],
    }


def _time_prediction_loop(
    model: Any,
    X: Any,
    *,
    calls: int,
) -> tuple[float, str]:
    started = time.perf_counter_ns()
    prediction = None
    for _ in range(int(calls)):
        prediction = model.predict(X)
    seconds = (time.perf_counter_ns() - started) / 1e9
    prediction = harness.validate_prediction(prediction, len(X))
    return float(seconds), harness.prediction_sha256(prediction)


def _validate_dataset(dataset: harness.BasketballDataset) -> dict[str, Any]:
    if dataset.raw_metadata["bytes"] != EXPECTED_DATA["raw_bytes"]:
        raise RuntimeError("basketball raw byte count changed")
    if dataset.raw_metadata["sha256"] != EXPECTED_DATA["raw_sha256"]:
        raise RuntimeError("basketball raw hash changed")
    if (
        dataset.processed_metadata["x_train_sha256"]
        != EXPECTED_DATA["x_sha256"]
    ):
        raise RuntimeError("basketball X hash changed")
    if (
        dataset.processed_metadata["y_train_sha256"]
        != EXPECTED_DATA["y_sha256"]
    ):
        raise RuntimeError("basketball y hash changed")
    if dataset.fold_fingerprint_sha256 != EXPECTED_DATA["fold_sha256"]:
        raise RuntimeError("basketball fold hash changed")
    guardrail = dataset.player_guardrail.metadata
    if guardrail["cold_player_mask_sha256"] != EXPECTED_DATA["cold_mask_sha256"]:
        raise RuntimeError("basketball cold-player mask changed")
    if dataset.X.shape != (5_241, 15):
        raise RuntimeError("basketball feature shape changed")
    if dataset.processed_metadata["missing_train_feature_cells"] != 0:
        raise RuntimeError("basketball training features gained missing values")
    dtypes = {str(name): str(dtype) for name, dtype in dataset.X.dtypes.items()}
    expected = {str(name): "float64" for name in creator.FEATURES}
    expected["Age"] = "int64"
    if dtypes != expected:
        raise RuntimeError("basketball feature dtypes changed")
    return {
        "raw": dataset.raw_metadata,
        "processed": dataset.processed_metadata,
        "fold_sha256": dataset.fold_fingerprint_sha256,
        "fold_test_sizes": dataset.fold_test_sizes,
        "feature_dtypes": dtypes,
        "guardrail": guardrail,
    }


def run_worker(
    arm: str,
    data_cache: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown native-ordinal arm {arm!r}")
    if os.environ.get("DARKOFIT_WARMUP") != "0":
        raise RuntimeError("worker import warmup must be disabled")
    if Path(os.environ.get("NUMBA_CACHE_DIR", "")) != cache_dir:
        raise RuntimeError("worker Numba cache binding changed")
    if _cache_stats(cache_dir)["file_count"] != 0:
        raise RuntimeError("worker Numba cache was not empty")

    source = require_frozen_source()
    runtime = _runtime_record()
    data_cache = _validate_data_cache(data_cache)
    dataset = harness.load_basketball_dataset(data_cache)
    data = _validate_dataset(dataset)
    before_import = _cache_stats(cache_dir)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import_started = time.perf_counter_ns()
        import darkofit

        import_seconds = (time.perf_counter_ns() - import_started) / 1e9
        module_path = Path(darkofit.__file__).resolve()
        if not module_path.is_relative_to(REPO_ROOT.resolve()):
            raise RuntimeError("darkofit imported outside the frozen repository")
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

        folds = []
        total_fit_seconds = 0.0
        worker_started = time.perf_counter_ns()
        for fold, (train, test) in enumerate(
            creator.creator_cv().split(dataset.X, dataset.y)
        ):
            model, fit_seconds = _fit_model(
                arm, dataset.X.iloc[train], dataset.y.iloc[train]
            )
            total_fit_seconds += fit_seconds
            prediction = model.predict(dataset.X.iloc[test])
            folds.append(
                {
                    "fold": int(fold),
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "test_indices": [int(value) for value in test],
                    "fit_seconds": fit_seconds,
                    "prediction": _prediction_record(
                        dataset.y.iloc[test], prediction
                    ),
                    "model": _fit_observation(model, arm),
                }
            )

        guardrail = dataset.player_guardrail
        model, full_fit_seconds = _fit_model(
            arm, guardrail.X_train, guardrail.y_train
        )
        total_fit_seconds += full_fit_seconds
        held_prediction = harness.validate_prediction(
            model.predict(guardrail.X_holdout), len(guardrail.X_holdout)
        )
        cold_input = guardrail.X_holdout.iloc[
            np.flatnonzero(guardrail.cold_player_mask)
        ]
        seen_input = guardrail.X_holdout.iloc[
            np.flatnonzero(~guardrail.cold_player_mask)
        ]
        cold_prediction = harness.validate_prediction(
            model.predict(cold_input), len(cold_input)
        )
        seen_prediction = harness.validate_prediction(
            model.predict(seen_input), len(seen_input)
        )
        held_loop_seconds, held_loop_sha = _time_prediction_loop(
            model,
            guardrail.X_holdout,
            calls=HELD_PREDICTION_CALLS,
        )
        cold_loop_seconds, cold_loop_sha = _time_prediction_loop(
            model,
            cold_input,
            calls=COLD_PREDICTION_CALLS,
        )
        worker_wall_seconds = (time.perf_counter_ns() - worker_started) / 1e9

    fold_scores = [row["prediction"]["r2"] for row in folds]
    guardrail_scores = harness.guardrails.score_player_guardrails(
        guardrail.y_holdout,
        held_prediction,
        guardrail.cold_player_mask,
    )
    result = {
        "ok": True,
        "arm": arm,
        "source": source,
        "runtime": runtime,
        "data": data,
        "import_seconds": float(import_seconds),
        "warmup_seconds": float(warmup_seconds),
        "cache": {
            "before_import": before_import,
            "after_import": after_import,
            "after_warmup": after_warmup,
            "after_workload": _cache_stats(cache_dir),
        },
        "folds": folds,
        "mean_r2": float(np.mean(fold_scores)),
        "fold_scores": fold_scores,
        "guardrail": {
            "full_fit_seconds": full_fit_seconds,
            "model": _fit_observation(model, arm),
            "held": {
                **_prediction_record(
                    guardrail.y_holdout, held_prediction
                ),
                "loop_calls": HELD_PREDICTION_CALLS,
                "loop_seconds": held_loop_seconds,
                "loop_last_prediction_sha256": held_loop_sha,
            },
            "cold": {
                **_prediction_record(
                    guardrail.y_holdout.iloc[
                        np.flatnonzero(guardrail.cold_player_mask)
                    ],
                    cold_prediction,
                ),
                "loop_calls": COLD_PREDICTION_CALLS,
                "loop_seconds": cold_loop_seconds,
                "loop_last_prediction_sha256": cold_loop_sha,
            },
            "seen": _prediction_record(
                guardrail.y_holdout.iloc[
                    np.flatnonzero(~guardrail.cold_player_mask)
                ],
                seen_prediction,
            ),
            "scores_from_full_prediction": guardrail_scores,
        },
        "total_fit_seconds": float(total_fit_seconds),
        "worker_wall_seconds": float(worker_wall_seconds),
        "peak_rss_bytes": _peak_rss_bytes(),
        "warnings": _warning_records(caught),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("DARKOFIT_WARMUP", *creator.THREAD_ENV_KEYS)
        },
    }
    result["behavior_fingerprint_sha256"] = harness.behavior_fingerprint(
        _behavior_payload(result)
    )
    return result


def _worker_environment(cache_dir: Path, threads: int) -> dict[str, str]:
    environment = harness.worker_environment(threads)
    environment.update(
        {
            "DARKOFIT_WARMUP": "0",
            "CHIMERABOOST_WARMUP": "0",
            "NUMBA_CACHE_DIR": str(cache_dir),
            "PYTHONHASHSEED": "0",
        }
    )
    return environment


def _run_worker_process(
    args: argparse.Namespace,
    arm: str,
    *,
    block: int,
    position: int,
    cache_dir: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-arm",
        arm,
        "--threads",
        str(args.threads),
        "--data-cache",
        str(args.data_cache),
        "--cache-dir",
        str(cache_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_worker_environment(cache_dir, args.threads),
        check=False,
        capture_output=True,
        text=True,
    )
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
            "arm": arm,
            "error_type": "WorkerProtocolError",
            "error": f"expected one worker result line, found {len(lines)}",
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
    result["block"] = int(block)
    result["position"] = int(position)
    return result


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(os.path.abspath(os.path.expanduser(os.fspath(args.output))))
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing existing output: {output}")
    if args.threads != EXPECTED_THREADS:
        raise RuntimeError(
            f"native-ordinal campaign requires {EXPECTED_THREADS} threads"
        )
    args.data_cache = _validate_data_cache(args.data_cache)
    initial_source = require_frozen_source()
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    repeats = []
    for block, order in enumerate(BLOCK_ORDERS):
        for position, arm in enumerate(order):
            if _source_state() != initial_source:
                raise RuntimeError("source changed before native-ordinal worker")
            print(
                f"[native-ordinal] block={block} position={position} arm={arm}",
                flush=True,
            )
            with tempfile.TemporaryDirectory(
                prefix=f"b{block}-{arm}-",
                dir=CACHE_ROOT,
            ) as cache_name:
                cache_dir = Path(cache_name)
                result = _run_worker_process(
                    args,
                    arm,
                    block=block,
                    position=position,
                    cache_dir=cache_dir,
                )
            repeats.append(
                {
                    "block": int(block),
                    "position": int(position),
                    "order": list(order),
                    "arm": arm,
                    "result": result,
                }
            )
    final_source = _source_state()
    if final_source != initial_source:
        raise RuntimeError("source changed during native-ordinal campaign")

    artifact = {
        "schema_version": 1,
        "name": "darkofit_basketball_native_ordinal_raw_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
            "commit": PROTOCOL_COMMIT,
            "sha256": EXPECTED_PROTOCOL_SHA256,
        },
        "runner": {
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "normalized_sha256": EXPECTED_NORMALIZED_RUNNER_SHA256,
        },
        "analyzer": {
            "path": str(ANALYZER_PATH.relative_to(REPO_ROOT)),
            "sha256": initial_source["analyzer_sha256"],
        },
        "source": initial_source,
        "execution": {
            "threads": args.threads,
            "timing_blocks": TIMING_BLOCKS,
            "block_orders": [list(order) for order in BLOCK_ORDERS],
            "held_prediction_calls": HELD_PREDICTION_CALLS,
            "cold_prediction_calls": COLD_PREDICTION_CALLS,
            "worker_count": len(repeats),
            "partial_resume_supported": False,
        },
        "repeats": repeats,
        "categorical_outcomes_inspected": False,
        "lockbox_touched": False,
    }
    payload = (
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_create(output, payload)
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-arm", choices=ARMS)
    parser.add_argument("--threads", type=int, default=EXPECTED_THREADS)
    parser.add_argument(
        "--data-cache",
        type=Path,
        default=harness.DEFAULT_CACHE,
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.worker_arm is not None and args.cache_dir is None:
        parser.error("--cache-dir is required for worker mode")
    return args


def main() -> int:
    args = parse_args()
    if args.worker_arm is not None:
        try:
            result = run_worker(
                args.worker_arm,
                args.data_cache,
                args.cache_dir,
            )
            returncode = 0
        except Exception as exc:
            result = {
                "ok": False,
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
    output = Path(
        os.path.abspath(os.path.expanduser(os.fspath(args.output)))
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256_file(output),
                "workers": len(artifact["repeats"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
