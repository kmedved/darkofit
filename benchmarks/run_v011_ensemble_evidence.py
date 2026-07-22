#!/usr/bin/env python3
"""Run the frozen v0.11 private ensemble evidence campaign."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import inspect
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

try:
    from . import run_ensemble_v3_characterization as retired
    from . import run_m3b_ensemble_v3 as m3b
except ImportError:  # direct script execution
    import run_ensemble_v3_characterization as retired
    import run_m3b_ensemble_v3 as m3b


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
PROTOCOL_PATH = BENCH / "v011_ensemble_evidence_protocol.md"
CONTRACT_PATH = BENCH / "v011_ensemble_evidence_contract.json"
ANALYZER_PATH = BENCH / "analyze_v011_ensemble_evidence.py"
FREEZER_PATH = BENCH / "freeze_v011_ensemble_evidence.py"
DEFAULT_OUTPUT = BENCH / "v011_ensemble_evidence_raw.json"
DEFAULT_TERMINAL = BENCH / "v011_ensemble_evidence_terminal.json"
READOUT_PATH = BENCH / "m3b_ensemble_v3_r3_vs_single_readout_20260721.json"

CONTRACT_ID = "v011-private-ensemble-evidence-v1"
DARKOFIT_HEAD = "543604dd9860a28c30912f914b2cfccfcb99d783"
CHIMERABOOST_HEAD = "f14be606b641f1bf0dc92bb14b3951f1fe631c6b"
CATBOOST_VERSION = "1.2.10"
THREADS = 14
ITERATIONS = 600
PATIENCE = 30
RANDOM_STATE = 4
VALIDATION_FRACTION = 0.15
BLOCKS = 3
REPRODUCTION_ABS_TOLERANCE = 1e-10
SPORTS_BOOTSTRAP_DRAWS = 100_000
SPORTS_BOOTSTRAP_SEED = 20_260_720
GENERAL_BOOTSTRAP_DRAWS = 100_000
GENERAL_BOOTSTRAP_SEED = 20_260_721
BATCH_SIZES = (8_192, 65_536, 524_288, 2_000_000)
PILOT_CALLS = 5
TARGET_INTERVAL_SECONDS = 2.0
MIN_INTERVAL_SECONDS = 1.0
MIN_CALLS = 3
MAX_CALLS = 65_536
RSS_INTERVAL_SECONDS = 0.01
WORKER_PREFIX = "V011_ENSEMBLE_EVIDENCE_RESULT="

DARKO_SINGLE = "darkofit_single"
DARKO_BOOTSTRAP = "darkofit_existing_bootstrap8"
DARKO_V3 = "darkofit_ensemble_v3"
CHIMERA_SINGLE = "chimeraboost_0_18_single"
CHIMERA_ENSEMBLE = "chimeraboost_0_18_ensemble8"
CATBOOST_SINGLE = "catboost_1_2_10_single"
QUALITY_ARMS = (DARKO_SINGLE, DARKO_BOOTSTRAP, DARKO_V3)
PREDICTION_ARMS = (
    DARKO_SINGLE,
    DARKO_V3,
    CHIMERA_SINGLE,
    CHIMERA_ENSEMBLE,
    CATBOOST_SINGLE,
)
QUALITY_CASES = tuple(spec["case_id"] for spec in m3b.case_specs())
PREDICTION_CASES = (
    "general_friedman_numeric",
    "general_categorical_reg",
    "general_numeric_binary",
    "general_categorical_multiclass",
)
THREAD_ENV_KEYS = retired.THREAD_ENV_KEYS

BOUND_PATHS = {
    "protocol": "benchmarks/v011_ensemble_evidence_protocol.md",
    "runner": "benchmarks/run_v011_ensemble_evidence.py",
    "analyzer": "benchmarks/analyze_v011_ensemble_evidence.py",
    "freezer": "benchmarks/freeze_v011_ensemble_evidence.py",
    "tests": "tests/test_v011_ensemble_evidence.py",
    "authorization": "benchmarks/v011_evidence_phase_instruction_20260721.md",
    "public_contract": "benchmarks/ensemble_v3_public_contract.md",
    "implementation": "darkofit/sklearn_api.py",
    "implementation_tests": "tests/test_ensemble_v3_release_candidate.py",
    "m3b_runner": "benchmarks/run_m3b_ensemble_v3.py",
    "m3a_runner": "benchmarks/run_m3a_wave1.py",
    "m3a_contract": "benchmarks/m3a_wave1_contract.json",
    "sports_manifest": "benchmarks/basketball_sports_panel_v2_manifest.json",
    "m6_adapter": "benchmarks/benchmark_adapters.py",
    "paired_execution": "benchmarks/paired_evidence_contract.py",
    "immutable_readout": (
        "benchmarks/m3b_ensemble_v3_r3_vs_single_readout_20260721.json"
    ),
    "retired_characterization": "benchmarks/run_ensemble_v3_characterization.py",
    "retired_audit": (
        "benchmarks/ensemble_v3_characterization_post_run_audit_20260721.json"
    ),
}


class ReproductionMismatch(RuntimeError):
    """The current private wrapper diverged from immutable M3b behavior."""


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


def _bound_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def source_state(repo: Path) -> dict[str, Any]:
    return retired.source_state(repo)


def _case_spec(case_id: str) -> Mapping[str, Any]:
    for spec in m3b.case_specs():
        if spec["case_id"] == case_id:
            return spec
    raise ValueError(f"unknown v0.11 evidence case: {case_id}")


def quality_order(case_id: str, block: int) -> tuple[str, ...]:
    offset = (QUALITY_CASES.index(case_id) + int(block)) % len(QUALITY_ARMS)
    return QUALITY_ARMS[offset:] + QUALITY_ARMS[:offset]


def prediction_order(case_id: str, block: int) -> tuple[str, ...]:
    case_index = PREDICTION_CASES.index(case_id)
    offset = (2 * case_index + int(block)) % len(PREDICTION_ARMS)
    return PREDICTION_ARMS[offset:] + PREDICTION_ARMS[:offset]


def execution_spec() -> dict[str, Any]:
    return {
        "contract_id": CONTRACT_ID,
        "darkofit_head": DARKOFIT_HEAD,
        "chimeraboost_head": CHIMERABOOST_HEAD,
        "catboost_version": CATBOOST_VERSION,
        "threads": THREADS,
        "iterations": ITERATIONS,
        "patience": PATIENCE,
        "random_state": RANDOM_STATE,
        "validation_fraction": VALIDATION_FRACTION,
        "blocks": BLOCKS,
        "quality_cases": list(QUALITY_CASES),
        "quality_arms": list(QUALITY_ARMS),
        "quality_orders": {
            f"{case_id}:block{block}": list(quality_order(case_id, block))
            for case_id in QUALITY_CASES
            for block in range(BLOCKS)
        },
        "reproduction_absolute_ratio_tolerance": REPRODUCTION_ABS_TOLERANCE,
        "prediction_cases": list(PREDICTION_CASES),
        "prediction_arms": list(PREDICTION_ARMS),
        "prediction_orders": {
            f"{case_id}:block{block}": list(prediction_order(case_id, block))
            for case_id in PREDICTION_CASES
            for block in range(BLOCKS)
        },
        "batch_sizes": list(BATCH_SIZES),
        "pilot_calls": PILOT_CALLS,
        "target_interval_seconds": TARGET_INTERVAL_SECONDS,
        "minimum_interval_seconds": MIN_INTERVAL_SECONDS,
        "minimum_calls": MIN_CALLS,
        "maximum_calls": MAX_CALLS,
        "rss_interval_seconds": RSS_INTERVAL_SECONDS,
        "fresh_worker_per_case_arm_block": True,
        "same_case_arm_warmup_outside_measurement": True,
        "output_create_only": True,
        "silent_worker_retry": False,
    }


def uncertainty_spec() -> dict[str, Any]:
    return {
        "sports": {
            "cluster_unit": "season",
            "clusters": list(m3b.SPORTS_SEASONS),
            "draws": SPORTS_BOOTSTRAP_DRAWS,
            "seed": SPORTS_BOOTSTRAP_SEED,
            "percentiles": [2.5, 50.0, 97.5],
            "leave_one_season_out": True,
        },
        "general": {
            "case_count": len(PREDICTION_CASES),
            "draws": GENERAL_BOOTSTRAP_DRAWS,
            "seed": GENERAL_BOOTSTRAP_SEED,
            "percentiles": [2.5, 50.0, 97.5],
            "leave_one_case_out": True,
        },
    }


def claim_spec() -> dict[str, Any]:
    return {
        "tier": "E",
        "private_characterization_only": True,
        "public_exposure_authorized": False,
        "default_change_authorized": False,
        "m2_or_m4": False,
        "release_authorized": False,
        "fresh_or_lockbox_data": False,
        "exposure_stop_conditions": [
            "correctness_failure",
            "unresolved_reproduction_failure",
        ],
        "performance_or_cost_gate": False,
    }


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("contract_id") != CONTRACT_ID
        or payload.get("contract_frozen") is not True
        or payload.get("outcome_blind") is not True
        or payload.get("execution") != execution_spec()
        or payload.get("uncertainty") != uncertainty_spec()
        or payload.get("claims") != claim_spec()
        or set(payload.get("bindings", {})) != set(BOUND_PATHS)
    ):
        raise RuntimeError("v0.11 ensemble evidence contract is invalid")
    for name, relative in BOUND_PATHS.items():
        if payload["bindings"][name] != _bound_record(ROOT / relative):
            raise RuntimeError(f"v0.11 ensemble evidence binding drifted: {name}")
    manifests = payload.get("case_manifests")
    if (
        not isinstance(manifests, Mapping)
        or set(manifests) != set(QUALITY_CASES)
        or any(
            not isinstance(manifests[case_id], Mapping)
            or set(manifests[case_id].get("fingerprints", {}))
            != {"case_sha256", "dataset_sha256", "split_sha256", "weight_sha256"}
            for case_id in QUALITY_CASES
        )
    ):
        raise RuntimeError("v0.11 ensemble evidence case manifests are invalid")
    if payload.get("immutable_ratios") != immutable_ratios():
        raise RuntimeError("v0.11 immutable reproduction ratios drifted")
    return payload


def immutable_ratios() -> dict[str, Any]:
    payload = json.loads(READOUT_PATH.read_text(encoding="utf-8"))
    combined = payload["arms_vs_single"]["b1_b2_combined"]
    return {
        "per_case": {
            str(case_id): float(value)
            for case_id, value in combined["per_case_primary_ratio"].items()
        },
        "pooled": float(combined["all_case_geometric_mean"]),
        "sports": float(combined["sports_geometric_mean"]),
        "general": float(combined["general_geometric_mean"]),
    }


def _activate_sources(darkofit_source: Path, chimeraboost_source: Path) -> None:
    for source in (darkofit_source, chimeraboost_source):
        value = str(source.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)


def _implementation(model: Any, expected_source: Path | None) -> dict[str, Any]:
    module = importlib.import_module(model.__class__.__module__)
    path = Path(inspect.getfile(module)).resolve()
    if expected_source is not None and not path.is_relative_to(expected_source.resolve()):
        raise RuntimeError(f"model imported from unexpected source: {path}")
    return {
        "class": model.__class__.__name__,
        "module_path": str(path),
        "expected_source": None if expected_source is None else str(expected_source),
    }


def _darkofit_model(arm: str, spec: Mapping[str, Any]):
    from darkofit import DarkoClassifier, DarkoRegressor

    estimator = DarkoRegressor if spec["task"] == "regression" else DarkoClassifier
    return estimator(
        iterations=ITERATIONS,
        early_stopping_rounds=PATIENCE,
        early_stopping=True,
        use_best_model=True,
        refit=False,
        validation_fraction=VALIDATION_FRACTION,
        validation_strategy=("group" if spec["domain"] == "sports" else "random"),
        random_state=RANDOM_STATE,
        thread_count=THREADS,
        diagnostic_warnings="never",
        ensemble_shared_preprocessing=True,
        ensemble_bootstrap=spec["sampling_unit"],
        n_ensembles=8 if arm != DARKO_SINGLE else 1,
    )


def _chimeraboost_model(arm: str, spec: Mapping[str, Any]):
    from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

    estimator = (
        ChimeraBoostRegressor if spec["task"] == "regression" else ChimeraBoostClassifier
    )
    return estimator(
        n_estimators=ITERATIONS,
        early_stopping_rounds=PATIENCE,
        early_stopping=True,
        validation_fraction=VALIDATION_FRACTION,
        random_state=RANDOM_STATE,
        thread_count=THREADS,
        n_ensembles=8 if arm == CHIMERA_ENSEMBLE else None,
        ensemble_n_jobs=-1,
        max_samples=0.8,
        quantize_gradients=True,
    )


def _catboost_model(spec: Mapping[str, Any]):
    from catboost import CatBoostClassifier, CatBoostRegressor

    common = {
        "iterations": ITERATIONS,
        "random_seed": RANDOM_STATE,
        "thread_count": THREADS,
        "verbose": False,
        "allow_writing_files": False,
    }
    if spec["task"] == "regression":
        return CatBoostRegressor(loss_function="RMSE", **common)
    return CatBoostClassifier(
        loss_function=("Logloss" if spec["task"] == "binary" else "MultiClass"),
        **common,
    )


def _build_model(
    arm: str,
    spec: Mapping[str, Any],
    darkofit_source: Path,
    chimeraboost_source: Path,
) -> tuple[Any, Path | None]:
    _activate_sources(darkofit_source, chimeraboost_source)
    if arm in {DARKO_SINGLE, DARKO_BOOTSTRAP, DARKO_V3}:
        return _darkofit_model(arm, spec), darkofit_source
    if arm in {CHIMERA_SINGLE, CHIMERA_ENSEMBLE}:
        return _chimeraboost_model(arm, spec), chimeraboost_source
    if arm == CATBOOST_SINGLE:
        return _catboost_model(spec), None
    raise ValueError(f"unknown v0.11 evidence arm: {arm}")


def _take_rows(values: Any, indices: np.ndarray) -> Any:
    return retired._take_rows(values, indices)


def _fit_darkofit(model: Any, arm: str, spec: Mapping[str, Any], data: Mapping[str, Any]):
    kwargs = {
        "cat_features": data["cat_features"],
        "groups": data.get("groups_fit"),
        "sample_weight": data.get("w_fit"),
    }
    if arm == DARKO_SINGLE:
        return model.fit(data["X_fit"], data["y_fit"], **kwargs)
    if arm == DARKO_V3:
        from darkofit.sklearn_api import _fit_ensemble_v3_release_candidate

        return _fit_ensemble_v3_release_candidate(
            model, data["X_fit"], data["y_fit"], **kwargs
        )
    from darkofit.sklearn_api import _fit_private_ensemble_v3

    return _fit_private_ensemble_v3(
        model,
        data["X_fit"],
        data["y_fit"],
        sampling="bootstrap",
        sampling_unit=spec["sampling_unit"],
        sample_fraction=None,
        member_policy="none",
        **kwargs,
    )


def _catboost_split(spec: Mapping[str, Any], data: Mapping[str, Any]):
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(data["y_fit"]), dtype=np.int64)
    stratify = None if spec["task"] == "regression" else np.asarray(data["y_fit"])
    train, valid = train_test_split(
        indices,
        test_size=VALIDATION_FRACTION,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    return train, valid


def _fit_prediction_model(
    model: Any, arm: str, spec: Mapping[str, Any], data: Mapping[str, Any]
):
    if arm in {DARKO_SINGLE, DARKO_V3}:
        return _fit_darkofit(model, arm, spec, data)
    if arm in {CHIMERA_SINGLE, CHIMERA_ENSEMBLE}:
        return model.fit(
            data["X_fit"],
            data["y_fit"],
            cat_features=data["cat_features"],
            sample_weight=data["w_fit"],
        )
    train, valid = _catboost_split(spec, data)
    fit_kwargs = {
        "cat_features": data["cat_features"],
        "sample_weight": (
            None if data["w_fit"] is None else np.asarray(data["w_fit"])[train]
        ),
        "eval_set": (
            _take_rows(data["X_fit"], valid),
            np.asarray(data["y_fit"])[valid],
        ),
        "early_stopping_rounds": PATIENCE,
        "use_best_model": True,
        "verbose": False,
    }
    return model.fit(
        _take_rows(data["X_fit"], train),
        np.asarray(data["y_fit"])[train],
        **fit_kwargs,
    )


def _warmup(
    kind: str,
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
        "groups_fit": (
            None
            if data.get("groups_fit") is None
            else np.asarray(data["groups_fit"])[indices]
        ),
        "cat_features": data["cat_features"],
    }
    if arm in {CHIMERA_SINGLE, CHIMERA_ENSEMBLE}:
        model.set_params(n_estimators=2)
    elif arm == CATBOOST_SINGLE:
        model.set_params(iterations=2)
    else:
        model.set_params(iterations=2)
    if kind == "quality":
        _fit_darkofit(model, arm, spec, warm)
    else:
        _fit_prediction_model(model, arm, spec, warm)
    gc.collect()


class ProcessTreeRSSSampler:
    """Failure-safe aggregate RSS for a worker and recursive children."""

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
        try:
            self._sample_once()
            self.end_bytes = self.current_bytes()
            self.peak_bytes = max(self.peak_bytes, self.end_bytes)
        except Exception as cleanup_exc:  # preserve a primary fit exception
            self.errors.append(f"{type(cleanup_exc).__name__}: {cleanup_exc}")
        if exc_type is None and (self.errors or self.samples < 2):
            raise RuntimeError(f"process-tree RSS sampling failed: {self.errors}")
        return False


def _valid_prediction(value: Any, rows: int) -> np.ndarray:
    output = np.asarray(value)
    # CatBoost's sklearn classifier surface returns labels as (n, 1), while
    # DarkoFit and ChimeraBoost return (n,).  Both are complete public
    # ``predict`` results; normalize only the singleton label axis for hashing.
    if output.shape == (int(rows), 1):
        output = output.reshape(int(rows))
    if output.shape != (int(rows),):
        raise RuntimeError(f"invalid prediction shape: {output.shape}")
    if output.dtype.kind in "fc" and not np.all(np.isfinite(output)):
        raise RuntimeError("prediction contains non-finite values")
    return output


def _probability(model: Any, spec: Mapping[str, Any], X: Any) -> np.ndarray | None:
    if spec["task"] == "regression":
        return None
    value = np.asarray(model.predict_proba(X), dtype=np.float64)
    if (
        value.shape[0] != len(X)
        or value.ndim != 2
        or not np.isfinite(value).all()
        or np.min(value) < -1e-12
        or np.max(value) > 1.0 + 1e-12
        or np.max(np.abs(value.sum(axis=1) - 1.0)) > 1e-8
    ):
        raise RuntimeError("invalid classification probability output")
    return value


def _array_hash(value: Any) -> str:
    return retired.array_sha256(value)


def _core_metadata(model: Any, arm: str) -> dict[str, Any]:
    if arm == CATBOOST_SINGLE:
        return {
            "member_count": 1,
            "total_tree_count": int(model.tree_count_),
            "members": [{"tree_count": int(model.tree_count_), "thread_count": THREADS}],
        }
    return retired._core_metadata(model)


def _safe_archive(model: Any, reference_X: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    reference = _valid_prediction(model.predict(reference_X), len(reference_X))
    probability = _probability(model, spec, reference_X)
    with tempfile.TemporaryDirectory(prefix="darkofit-v011-ensemble-") as directory:
        path = Path(directory) / "model.npz"
        model.save_model(path)
        size = path.stat().st_size
        restored = model.__class__.load_model(path)
        restored_prediction = _valid_prediction(
            restored.predict(reference_X), len(reference_X)
        )
        restored_probability = _probability(restored, spec, reference_X)
    if not np.array_equal(reference, restored_prediction):
        raise RuntimeError("safe-NPZ prediction roundtrip is not exact")
    if probability is not None and not np.array_equal(probability, restored_probability):
        raise RuntimeError("safe-NPZ probability roundtrip is not exact")
    return {"format": "darkofit_safe_npz", "bytes": int(size), "roundtrip_exact": True}


def _quality_worker(
    args: argparse.Namespace, spec: Mapping[str, Any], data: Mapping[str, Any]
) -> dict[str, Any]:
    if args.arm not in QUALITY_ARMS:
        raise ValueError(f"invalid quality arm: {args.arm}")
    _warmup(
        "quality",
        args.arm,
        spec,
        data,
        args.darkofit_source,
        args.chimeraboost_source,
    )
    model, expected_source = _build_model(
        args.arm, spec, args.darkofit_source, args.chimeraboost_source
    )
    implementation = _implementation(model, expected_source)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with ProcessTreeRSSSampler() as rss:
            started = time.perf_counter_ns()
            _fit_darkofit(model, args.arm, spec, data)
            fit_seconds = (time.perf_counter_ns() - started) / 1e9
        prediction = _valid_prediction(model.predict(data["X_test"]), len(data["X_test"]))
        probability = _probability(model, spec, data["X_test"])
        metrics = m3b._metrics(model, spec, data, prediction, probability)
        archive = _safe_archive(model, data["X_test"], spec)
    metadata = _core_metadata(model, args.arm)
    if any(row["thread_count"] != THREADS for row in metadata["members"]):
        raise RuntimeError("fitted thread count differs from the frozen contract")
    return {
        "kind": "quality",
        "case_id": args.case_id,
        "domain": spec["domain"],
        "task": spec["task"],
        "arm": args.arm,
        **m3b.case_fingerprints(spec, data),
        **metrics,
        "fit_rows": int(len(data["y_fit"])),
        "feature_count": int(np.asarray(data["X_fit"]).shape[1]),
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
        "prediction_sha256": _array_hash(prediction),
        "probability_sha256": None if probability is None else _array_hash(probability),
        "model": metadata,
        "implementation": implementation,
        "warnings": [
            {"category": item.category.__name__, "message": str(item.message)}
            for item in caught
        ],
    }


def _timed_prediction(model: Any, X: Any) -> dict[str, Any]:
    rows = len(X)
    warm = _valid_prediction(model.predict(X), rows)
    pilots = []
    for _ in range(PILOT_CALLS):
        started = time.perf_counter_ns()
        value = _valid_prediction(model.predict(X), rows)
        pilots.append((time.perf_counter_ns() - started) / 1e9)
        if not np.array_equal(warm, value):
            raise RuntimeError("prediction pilot changed output")
    pilot_median = float(np.median(np.asarray(pilots, dtype=np.float64)))
    calls = int(
        min(
            MAX_CALLS,
            max(MIN_CALLS, math.ceil(TARGET_INTERVAL_SECONDS / max(pilot_median, 1e-9))),
        )
    )
    final = None
    gc.disable()
    started = time.perf_counter_ns()
    try:
        for _ in range(calls):
            final = _valid_prediction(model.predict(X), rows)
    finally:
        elapsed = (time.perf_counter_ns() - started) / 1e9
        gc.enable()
    if final is None or not np.array_equal(warm, final):
        raise RuntimeError("formal prediction interval changed output")
    if elapsed < MIN_INTERVAL_SECONDS:
        raise RuntimeError(
            f"prediction interval {elapsed:.6f}s missed frozen {MIN_INTERVAL_SECONDS:.6f}s floor"
        )
    return {
        "rows": int(rows),
        "calls": calls,
        "pilot_seconds": [float(value) for value in pilots],
        "pilot_median_seconds": pilot_median,
        "interval_seconds": float(elapsed),
        "minimum_interval_met": True,
        "seconds_per_call": float(elapsed / calls),
        "rows_per_second": float(rows * calls / elapsed),
        "prediction_sha256": _array_hash(final),
        "method": "predict",
    }


def _prediction_worker(
    args: argparse.Namespace, spec: Mapping[str, Any], data: Mapping[str, Any]
) -> dict[str, Any]:
    if args.arm not in PREDICTION_ARMS:
        raise ValueError(f"invalid prediction arm: {args.arm}")
    _warmup(
        "prediction",
        args.arm,
        spec,
        data,
        args.darkofit_source,
        args.chimeraboost_source,
    )
    model, expected_source = _build_model(
        args.arm, spec, args.darkofit_source, args.chimeraboost_source
    )
    implementation = _implementation(model, expected_source)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter_ns()
        _fit_prediction_model(model, args.arm, spec, data)
        fit_seconds = (time.perf_counter_ns() - started) / 1e9
        predictions = {}
        for rows in BATCH_SIZES:
            X = retired._repeat_rows(data["X_test"], rows)
            record = _timed_prediction(model, X)
            record["input_sha256"] = _array_hash(X)
            predictions[str(rows)] = record
            del X
            gc.collect()
    metadata = _core_metadata(model, args.arm)
    fitted_threads = [row["thread_count"] for row in metadata["members"]]
    if args.arm not in {CATBOOST_SINGLE, CHIMERA_ENSEMBLE} and any(
        value != THREADS for value in fitted_threads
    ):
        raise RuntimeError("fitted thread count differs from the frozen contract")
    if args.arm == CHIMERA_ENSEMBLE and any(
        value < 1 or value > THREADS for value in fitted_threads
    ):
        raise RuntimeError("ChimeraBoost ensemble exceeded the total thread budget")
    return {
        "kind": "prediction",
        "case_id": args.case_id,
        "domain": spec["domain"],
        "task": spec["task"],
        "arm": args.arm,
        **m3b.case_fingerprints(spec, data),
        "fit_rows": int(len(data["y_fit"])),
        "feature_count": int(np.asarray(data["X_fit"]).shape[1]),
        "fit_seconds_telemetry": float(fit_seconds),
        "predictions": predictions,
        "model": metadata,
        "implementation": implementation,
        "warnings": [
            {"category": item.category.__name__, "message": str(item.message)}
            for item in caught
        ],
    }


def _worker_environment(
    darkofit_source: Path, chimeraboost_source: Path
) -> dict[str, str]:
    environment = os.environ.copy()
    prefixes = ("NUMBA_", "OMP_", "KMP_", "MKL_", "OPENBLAS_", "VECLIB_", "NUMEXPR_")
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
                (str(darkofit_source), str(chimeraboost_source), str(ROOT))
            ),
            "NUMBA_CACHE_DIR": "/private/tmp/darkofit-v011-evidence-numba",
            "JOBLIB_TEMP_FOLDER": "/private/tmp/darkofit-v011-evidence-joblib",
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
        raise RuntimeError(f"v0.11 evidence worker environment drifted: {actual}")
    import catboost
    import numba

    if catboost.__version__ != CATBOOST_VERSION:
        raise RuntimeError(f"CatBoost version drifted: {catboost.__version__}")
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
        "catboost_version": catboost.__version__,
    }


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    runtime_before = _assert_worker_contract()
    contract = load_contract(args.contract)
    spec = _case_spec(args.case_id)
    data = m3b.build_case(spec, args.panel_cache)
    manifest = contract["case_manifests"][args.case_id]
    if m3b.case_fingerprints(spec, data) != manifest["fingerprints"]:
        raise RuntimeError(f"case fingerprint drifted: {args.case_id}")
    if args.kind == "quality":
        result = _quality_worker(args, spec, data)
    else:
        result = _prediction_worker(args, spec, data)
    result["runtime_before"] = runtime_before
    result["runtime_after"] = _assert_worker_contract()
    return result


def _run_worker(
    args: argparse.Namespace, kind: str, case_id: str, arm: str
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--kind",
        kind,
        "--contract",
        str(args.contract),
        "--darkofit-source",
        str(args.darkofit_source),
        "--chimeraboost-source",
        str(args.chimeraboost_source),
        "--panel-cache",
        str(args.panel_cache),
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
        line for line in completed.stdout.splitlines() if line.startswith(WORKER_PREFIX)
    ]
    if completed.returncode or len(lines) != 1:
        raise RuntimeError(
            f"{kind} worker failed for {case_id}/{arm} ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    result = json.loads(lines[0][len(WORKER_PREFIX) :])
    result["worker_stdout"] = "\n".join(
        line for line in completed.stdout.splitlines() if not line.startswith(WORKER_PREFIX)
    ).strip() or None
    result["worker_stderr"] = completed.stderr.strip() or None
    return result


def _catboost_state() -> dict[str, Any]:
    import catboost

    path = Path(inspect.getfile(catboost)).resolve()
    return {"version": catboost.__version__, "module_path": str(path)}


def _require_clean_sources(args: argparse.Namespace) -> dict[str, Any]:
    states = {
        "harness": source_state(ROOT),
        "darkofit": source_state(args.darkofit_source),
        "chimeraboost": source_state(args.chimeraboost_source),
        "catboost": _catboost_state(),
    }
    if not all(states[name]["clean"] for name in ("harness", "darkofit", "chimeraboost")):
        raise RuntimeError("v0.11 ensemble evidence requires three clean checkouts")
    if states["darkofit"]["head"] != DARKOFIT_HEAD:
        raise RuntimeError("DarkoFit source is not the published Phase 0 pin")
    if states["chimeraboost"]["head"] != CHIMERABOOST_HEAD:
        raise RuntimeError("ChimeraBoost source is not the frozen 0.18 pin")
    if states["catboost"]["version"] != CATBOOST_VERSION:
        raise RuntimeError("CatBoost distribution is not the frozen version")
    return states


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"v0.11 ensemble evidence artifact is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json_bytes(payload))


def _require_external_outputs(args: argparse.Namespace) -> None:
    roots = (ROOT, args.darkofit_source, args.chimeraboost_source)
    for path in (args.output, args.terminal):
        for root in roots:
            if path.is_relative_to(root):
                raise ValueError(f"formal outputs must be outside all checkouts: {path}")


def _geomean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 1 or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise RuntimeError("geometric mean requires finite positive values")
    return float(np.exp(np.mean(np.log(array))))


def _check_reproduction(rows: Sequence[Mapping[str, Any]], block: int) -> None:
    expected = immutable_ratios()
    ratios = {}
    for case_id in QUALITY_CASES:
        selected = {
            row["arm"]: row
            for row in rows
            if row["kind"] == "quality"
            and row["block"] == block
            and row["case_id"] == case_id
        }
        if DARKO_SINGLE not in selected or DARKO_V3 not in selected:
            raise RuntimeError(f"incomplete reproduction pair: {case_id}/block{block}")
        ratio = float(selected[DARKO_V3]["primary_loss"]) / float(
            selected[DARKO_SINGLE]["primary_loss"]
        )
        ratios[case_id] = ratio
        if abs(ratio - expected["per_case"][case_id]) > REPRODUCTION_ABS_TOLERANCE:
            raise ReproductionMismatch(
                f"{case_id}/block{block} ratio {ratio:.17g} differs from immutable "
                f"{expected['per_case'][case_id]:.17g} by more than "
                f"{REPRODUCTION_ABS_TOLERANCE:.1e}"
            )
    domains = {
        "pooled": list(QUALITY_CASES),
        "sports": [case for case in QUALITY_CASES if case.startswith("sports_")],
        "general": [case for case in QUALITY_CASES if case.startswith("general_")],
    }
    for name, cases in domains.items():
        value = _geomean([ratios[case] for case in cases])
        if abs(value - expected[name]) > REPRODUCTION_ABS_TOLERANCE:
            raise ReproductionMismatch(
                f"{name}/block{block} ratio {value:.17g} differs from immutable "
                f"{expected[name]:.17g} by more than {REPRODUCTION_ABS_TOLERANCE:.1e}"
            )


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    _require_external_outputs(args)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"raw output is create-only: {args.output}")
    if args.terminal.exists() or args.terminal.is_symlink():
        raise FileExistsError(f"terminal output is create-only: {args.terminal}")
    contract = load_contract(args.contract)
    states = _require_clean_sources(args)
    rows: list[dict[str, Any]] = []
    try:
        for block in range(BLOCKS):
            for case_id in QUALITY_CASES:
                for position, arm in enumerate(quality_order(case_id, block)):
                    if _require_clean_sources(args) != states:
                        raise RuntimeError("source state changed during quality execution")
                    print(
                        f"quality block={block + 1}/{BLOCKS} case={case_id} "
                        f"position={position + 1}/{len(QUALITY_ARMS)} arm={arm}",
                        flush=True,
                    )
                    row = _run_worker(args, "quality", case_id, arm)
                    row.update(block=block, position=position)
                    rows.append(row)
            _check_reproduction(rows, block)
        for block in range(BLOCKS):
            for case_id in PREDICTION_CASES:
                for position, arm in enumerate(prediction_order(case_id, block)):
                    if _require_clean_sources(args) != states:
                        raise RuntimeError("source state changed during prediction execution")
                    print(
                        f"prediction block={block + 1}/{BLOCKS} case={case_id} "
                        f"position={position + 1}/{len(PREDICTION_ARMS)} arm={arm}",
                        flush=True,
                    )
                    row = _run_worker(args, "prediction", case_id, arm)
                    row.update(block=block, position=position)
                    rows.append(row)
        if _require_clean_sources(args) != states:
            raise RuntimeError("source state changed before artifact publication")
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
            "successor_after_named_correctness_fix_allowed": isinstance(
                exc, ReproductionMismatch
            ),
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
        "evidence_scope": "tier_e_private_v011_ensemble_characterization",
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
        "reproduction_checked": True,
        "rows": rows,
        "claims": claim_spec(),
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
    parser.add_argument("--panel-cache", type=Path, default=m3b.DEFAULT_PANEL_CACHE)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--kind", choices=("quality", "prediction"), help=argparse.SUPPRESS)
    parser.add_argument("--case-id", choices=QUALITY_CASES, help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=tuple(sorted(set(QUALITY_ARMS + PREDICTION_ARMS))), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in (
        "contract",
        "output",
        "terminal",
        "darkofit_source",
        "chimeraboost_source",
        "panel_cache",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.worker and (args.kind is None or args.case_id is None or args.arm is None):
        parser.error("worker mode requires --kind, --case-id, and --arm")
    if not args.worker and any(
        value is not None for value in (args.kind, args.case_id, args.arm)
    ):
        parser.error("--kind/--case-id/--arm are private worker arguments")
    if args.kind == "prediction" and args.case_id not in PREDICTION_CASES:
        parser.error("prediction worker case is outside the frozen grid")
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
