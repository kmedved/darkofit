#!/usr/bin/env python3
"""Run the frozen DarkoFit df1 SynthGen adoption ledger.

The runner writes one atomic shard per dataset so an interrupted campaign can
resume without rerunning completed coordinates. It never scores the ledger;
``analyze_synthgen_darkofit_ledger.py`` consumes only the combined raw JSON.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BENCHMARKS_ROOT = REPO_ROOT / "benchmarks"
if str(BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_ROOT))

import synthgen  # noqa: E402
from synthgen.suites import CANARIES, SUITES  # noqa: E402


SCHEMA_VERSION = 1
SPLIT_SEEDS = (17, 29, 43)
CANARY_SEEDS = (0, 1, 2)
THREAD_COUNT = 6
EPSILON = 1e-9

CONTROL = "control"
STUDENT_T = "student_t"
MAE = "mae"
RANDOM_STRENGTH_05 = "random_strength_0_5"
RANDOM_STRENGTH_10 = "random_strength_1_0"
LINEAR_LEAVES = "linear_leaves"
LINEAR_RESIDUAL = "linear_residual"
TS4 = "ts_permutations_4"
ORDERED = "ordered_boosting_true"
CORE = "core_profile"

CONFIG_ORDER = (
    CONTROL,
    STUDENT_T,
    MAE,
    RANDOM_STRENGTH_05,
    RANDOM_STRENGTH_10,
    LINEAR_LEAVES,
    LINEAR_RESIDUAL,
    TS4,
    ORDERED,
    CORE,
)
CONFIGS = {
    CONTROL: {},
    STUDENT_T: {"loss": "StudentT", "tree_mode": "lightgbm"},
    MAE: {"loss": "MAE"},
    RANDOM_STRENGTH_05: {"random_strength": 0.5},
    RANDOM_STRENGTH_10: {"random_strength": 1.0},
    LINEAR_LEAVES: {"linear_leaves": True},
    LINEAR_RESIDUAL: {"linear_residual": True},
    TS4: {"ts_permutations": 4},
    ORDERED: {"ordered_boosting": True},
    CORE: {
        "l2_leaf_reg": 1.0,
        "max_bins": 128,
        "learning_rate": 0.1,
    },
}
COMMON_PARAMS = {
    "iterations": 600,
    "depth": 6,
    "thread_count": THREAD_COUNT,
    "early_stopping": False,
    "refit": False,
    "diagnostic_warnings": "never",
    "verbose_timing": True,
}
CANARY_CONFIGS = {
    CONTROL: {"ts_permutations": 1},
    TS4: {"ts_permutations": 4},
}
CANARY_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.1,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "max_bins": 128,
    "tree_mode": "catboost",
    "ordered_boosting": False,
    "early_stopping": True,
    "early_stopping_rounds": 50,
    "validation_fraction": 0.2,
    "refit": False,
    "thread_count": THREAD_COUNT,
    "random_state": 0,
    "diagnostic_warnings": "never",
    "verbose_timing": True,
}
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

PROTOCOL_PATH = BENCHMARKS_ROOT / "synthgen_darkofit_protocol.md"
CORPUS_PATH = BENCHMARKS_ROOT / "synthgen" / "corpus_marginals.json"
SUITES_PATH = BENCHMARKS_ROOT / "synthgen" / "suites.py"
GOLDENS_PATH = REPO_ROOT / "tests" / "golden_synthgen.json"
FREEZE_PATH = BENCHMARKS_ROOT / "synthgen_df1_freeze.json"
RUNNER_PATH = Path(__file__).resolve()
ANALYZER_PATH = BENCHMARKS_ROOT / "analyze_synthgen_darkofit_ledger.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / ".cache" / "synthgen-darkofit-df1-ledger"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def source_attestation() -> dict[str, Any]:
    status = _git_output(
        "status", "--porcelain=v1", "--untracked-files=all"
    )
    if status:
        raise RuntimeError(
            "formal SynthGen ledger requires a clean source tree:\n" + status
        )
    commit = _git_output("rev-parse", "HEAD")
    origin_main = _git_output("rev-parse", "origin/main")
    if commit != origin_main:
        raise RuntimeError(
            "formal SynthGen ledger requires HEAD to equal published origin/main"
        )
    branch = _git_output("branch", "--show-current")
    if branch != "main":
        raise RuntimeError(
            "formal SynthGen ledger must run from the main branch"
        )
    return {
        "commit": commit,
        "branch": branch,
        "clean": True,
        "origin_main": origin_main,
    }


def environment_metadata() -> dict[str, Any]:
    import darkofit

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "darkofit": getattr(darkofit, "__version__", None),
        "thread_environment": {
            key: os.environ.get(key) for key in THREAD_ENV_KEYS
        },
    }


def split_indices(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n_rows < 4:
        raise ValueError("SynthGen split requires at least four rows")
    permutation = np.random.RandomState(seed).permutation(n_rows)
    n_test = int(math.ceil(0.25 * n_rows))
    test = np.asarray(permutation[:n_test], dtype=np.int64)
    train = np.asarray(permutation[n_test:], dtype=np.int64)
    return train, test


def split_hash(train: np.ndarray, test: np.ndarray) -> str:
    return _sha256_bytes(
        np.asarray(train, dtype="<i8").tobytes()
        + b"\x00"
        + np.asarray(test, dtype="<i8").tobytes()
    )


def frozen_slices() -> dict[str, list[int]]:
    slices = {
        "ordinary_regression": [],
        "noisy_nonlinear": [],
        "smooth_linear": [],
        "categorical_regression": [],
    }
    for dataset_id in SUITES["screen"]:
        _, _, _, task, meta = synthgen.build_dataset(
            synthgen.key_for(dataset_id)
        )
        if task != "regression" or meta["saturated"]:
            synthgen.build_dataset.cache_clear()
            continue
        slices["ordinary_regression"].append(dataset_id)
        if (
            meta["noise_level"] >= 0.25
            and meta["interaction_depth"] >= 2
            and meta["func_dominant"] != "linear"
        ):
            slices["noisy_nonlinear"].append(dataset_id)
        if meta["func_dominant"] == "linear":
            slices["smooth_linear"].append(dataset_id)
        if meta["n_cat"] > 0:
            slices["categorical_regression"].append(dataset_id)
        synthgen.build_dataset.cache_clear()
    for name, members in slices.items():
        if len(members) < 8:
            raise RuntimeError(
                f"frozen slice {name!r} has {len(members)} datasets; need 8"
            )
    return slices


def regression_dataset_ids() -> list[int]:
    result = []
    for dataset_id in SUITES["screen"]:
        if synthgen.task_of(synthgen.key_for(dataset_id)) == "regression":
            result.append(dataset_id)
    return result


def categorical_screen_canaries() -> list[int]:
    result = []
    for dataset_id in sorted(set(SUITES["screen"]) & set(CANARIES)):
        _, _, _, task, meta = synthgen.build_dataset(
            synthgen.key_for(dataset_id)
        )
        if task != "regression" and meta["n_cat"] > 0:
            result.append(dataset_id)
        synthgen.build_dataset.cache_clear()
    if not result:
        raise RuntimeError("categorical screen canary slice is empty")
    return result


def validate_freeze_evidence() -> dict[str, Any]:
    evidence = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if evidence["generator_version"] != synthgen.VERSION:
        raise RuntimeError("freeze evidence generator version changed")
    if evidence["selection"]["suites"] != SUITES:
        raise RuntimeError("freeze evidence suite membership changed")
    if set(evidence["selection"]["canaries"]) != set(CANARIES):
        raise RuntimeError("freeze evidence canary membership changed")
    records = {
        int(record["id"]): record for record in evidence["scan"]["records"]
    }
    for dataset_id in CANARIES:
        record = records.get(dataset_id)
        if record is None or record.get("canary") is not True:
            raise RuntimeError(f"canary {dataset_id} lacks earned evidence")
        values = np.asarray(record.get("ceiling_values"), dtype=np.float64)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise RuntimeError(f"canary {dataset_id} lacks exact seed metrics")
    return evidence


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _selected_lane(model: Any) -> str:
    if bool(getattr(model, "linear_residual_active_", False)):
        return "linear_residual"
    linear = dict(
        getattr(model.model_, "auto_params_", {}).get("linear_leaves", {}) or {}
    )
    return "linear_leaves" if linear.get("active", False) else "boosting"


def fitted_metadata(model: Any) -> dict[str, Any]:
    core = model.model_
    training = dict(getattr(core, "training_metadata_", {}) or {})
    return {
        "best_iteration": int(model.best_n_estimators_),
        "fitted_tree_count": int(model.n_estimators_),
        "resolved_learning_rate": float(model.learning_rate_),
        "requested_tree_mode": str(model.tree_mode),
        "selected_tree_mode": str(core.tree_mode_),
        "selected_lane": _selected_lane(model),
        "resolved_thread_count": int(core.n_threads_),
        "stop_reason": str(
            training.get("stop_reason", getattr(core, "stop_reason_", "unknown"))
        ),
        "iterations_requested": int(
            training.get("iterations_requested", core.iterations_)
        ),
        "iterations_attempted": int(
            training.get(
                "iterations_attempted",
                getattr(core, "iterations_attempted_", 0),
            )
        ),
        "rounds_completed": int(
            training.get("rounds_completed", getattr(core, "rounds_completed_", 0))
        ),
        "rounds_retained": int(
            training.get("rounds_retained", len(getattr(core, "trees_", ())))
        ),
        "early_stopping_rounds": (
            None
            if core.early_stopping_rounds_ is None
            else int(core.early_stopping_rounds_)
        ),
        "refit": bool(getattr(model, "refit_", False)),
    }


def _validate_fit_metadata(
    metadata: dict[str, Any], *, canary: bool
) -> None:
    expected_iterations = (
        CANARY_PARAMS["iterations"] if canary else COMMON_PARAMS["iterations"]
    )
    if metadata["iterations_requested"] != expected_iterations:
        raise RuntimeError("fitted model changed the frozen iteration budget")
    if metadata["resolved_thread_count"] != THREAD_COUNT:
        raise RuntimeError("fitted model changed the frozen thread count")
    if metadata["refit"]:
        raise RuntimeError("fitted model unexpectedly refit")
    if not canary and metadata["early_stopping_rounds"] is not None:
        raise RuntimeError("regression ledger unexpectedly used early stopping")


def _prediction_hash(prediction: np.ndarray) -> str:
    return _sha256_bytes(np.asarray(prediction, dtype="<f8").tobytes())


def _run_regression_dataset(
    dataset_id: int, run_fingerprint: str
) -> dict[str, Any]:
    from darkofit import DarkoRegressor

    key = synthgen.key_for(dataset_id)
    X, y, cat_features, task, meta = synthgen.build_dataset(key)
    if task != "regression":
        raise RuntimeError(f"{key} is not regression")
    dataset_hash = synthgen.hash_dataset(key)
    records = []
    split_records = []
    for seed_index, seed in enumerate(SPLIT_SEEDS):
        train, test = split_indices(len(y), seed)
        split_records.append(
            {
                "seed": seed,
                "train_rows": len(train),
                "test_rows": len(test),
                "indices_sha256": split_hash(train, test),
            }
        )
        X_train, y_train = X[train], np.asarray(y)[train]
        X_test, y_test = X[test], np.asarray(y)[test]
        offset = (dataset_id + seed_index) % len(CONFIG_ORDER)
        arm_order = CONFIG_ORDER[offset:] + CONFIG_ORDER[:offset]
        for arm in arm_order:
            params = dict(COMMON_PARAMS)
            params.update(CONFIGS[arm])
            params["random_state"] = seed
            model = DarkoRegressor(**params)
            started = time.perf_counter_ns()
            model.fit(X_train, y_train, cat_features=cat_features)
            fit_seconds = (time.perf_counter_ns() - started) / 1e9
            started = time.perf_counter_ns()
            prediction = np.asarray(model.predict(X_test), dtype=np.float64)
            predict_seconds = (time.perf_counter_ns() - started) / 1e9
            if prediction.shape != (len(test),) or not np.isfinite(prediction).all():
                raise RuntimeError(
                    f"{key}/{seed}/{arm} produced an invalid prediction"
                )
            rmse = float(np.sqrt(np.mean((y_test - prediction) ** 2)))
            if not math.isfinite(rmse):
                raise RuntimeError(f"{key}/{seed}/{arm} produced invalid RMSE")
            metadata = fitted_metadata(model)
            _validate_fit_metadata(metadata, canary=False)
            records.append(
                {
                    "kind": "regression_ledger",
                    "dataset_id": dataset_id,
                    "dataset_key": key,
                    "dataset_sha256": dataset_hash,
                    "seed": seed,
                    "arm": arm,
                    "rmse": rmse,
                    "noise_sigma": float(meta["noise_sigma"]),
                    "bayes_excess_rmse_ratio": (
                        rmse - float(meta["noise_sigma"])
                    )
                    / max(float(meta["noise_sigma"]), 1e-12),
                    "fit_seconds": float(fit_seconds),
                    "predict_seconds": float(predict_seconds),
                    "worker_peak_rss_bytes": _peak_rss_bytes(),
                    "serialized_model_bytes": len(
                        pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
                    ),
                    "prediction_sha256": _prediction_hash(prediction),
                    "fit_metadata": metadata,
                }
            )
    synthgen.build_dataset.cache_clear()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "kind": "regression_ledger",
        "dataset_id": dataset_id,
        "dataset_key": key,
        "dataset_sha256": dataset_hash,
        "metadata": meta,
        "splits": split_records,
        "records": records,
    }


def _sum_brier(
    classes: np.ndarray, y_true: np.ndarray, probability: np.ndarray
) -> float:
    onehot = (
        np.asarray(y_true)[:, None] == np.asarray(classes)[None, :]
    ).astype(np.float64)
    return float(np.mean(np.sum((probability - onehot) ** 2, axis=1)))


def _run_canary_dataset(
    dataset_id: int, run_fingerprint: str
) -> dict[str, Any]:
    from darkofit import DarkoClassifier
    from sklearn.model_selection import train_test_split

    key = synthgen.key_for(dataset_id)
    X, y, cat_features, task, meta = synthgen.build_dataset(key)
    if task == "regression" or not meta["saturated"] or meta["n_cat"] <= 0:
        raise RuntimeError(f"{key} is not a categorical classification canary")
    dataset_hash = synthgen.hash_dataset(key)
    records = []
    split_records = []
    for seed_index, seed in enumerate(CANARY_SEEDS):
        train, test = train_test_split(
            np.arange(len(y), dtype=np.int64),
            test_size=0.25,
            random_state=seed,
            stratify=y,
        )
        train = np.asarray(train, dtype=np.int64)
        test = np.asarray(test, dtype=np.int64)
        split_records.append(
            {
                "seed": seed,
                "train_rows": len(train),
                "test_rows": len(test),
                "indices_sha256": split_hash(train, test),
            }
        )
        arm_order = (
            (CONTROL, TS4) if seed_index % 2 == 0 else (TS4, CONTROL)
        )
        for arm in arm_order:
            model = DarkoClassifier(**CANARY_PARAMS, **CANARY_CONFIGS[arm])
            started = time.perf_counter_ns()
            model.fit(X[train], np.asarray(y)[train], cat_features=cat_features)
            fit_seconds = (time.perf_counter_ns() - started) / 1e9
            started = time.perf_counter_ns()
            probability = np.asarray(
                model.predict_proba(X[test]), dtype=np.float64
            )
            predict_seconds = (time.perf_counter_ns() - started) / 1e9
            if (
                probability.shape != (len(test), int(meta["n_classes"]))
                or not np.isfinite(probability).all()
            ):
                raise RuntimeError(
                    f"{key}/{seed}/{arm} produced invalid probabilities"
                )
            brier = _sum_brier(model.classes_, np.asarray(y)[test], probability)
            metadata = fitted_metadata(model)
            _validate_fit_metadata(metadata, canary=True)
            records.append(
                {
                    "kind": "canary_no_variance",
                    "dataset_id": dataset_id,
                    "dataset_key": key,
                    "dataset_sha256": dataset_hash,
                    "seed": seed,
                    "arm": arm,
                    "brier": brier,
                    "bayes_brier": float(meta["bayes_brier"]),
                    "excess_brier": brier - float(meta["bayes_brier"]),
                    "fit_seconds": float(fit_seconds),
                    "predict_seconds": float(predict_seconds),
                    "worker_peak_rss_bytes": _peak_rss_bytes(),
                    "serialized_model_bytes": len(
                        pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
                    ),
                    "probability_sha256": _prediction_hash(probability.ravel()),
                    "fit_metadata": metadata,
                }
            )
    synthgen.build_dataset.cache_clear()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "kind": "canary_no_variance",
        "dataset_id": dataset_id,
        "dataset_key": key,
        "dataset_sha256": dataset_hash,
        "metadata": meta,
        "splits": split_records,
        "records": records,
    }


def _worker(task: tuple[str, int], run_fingerprint: str) -> dict[str, Any]:
    kind, dataset_id = task
    if kind == "regression_ledger":
        return _run_regression_dataset(dataset_id, run_fingerprint)
    if kind == "canary_no_variance":
        return _run_canary_dataset(dataset_id, run_fingerprint)
    raise RuntimeError(f"unknown SynthGen worker kind {kind!r}")


def _shard_path(output_dir: Path, task: tuple[str, int]) -> Path:
    prefix = "reg" if task[0] == "regression_ledger" else "canary"
    return output_dir / "shards" / f"{prefix}-{task[1]:03d}.json"


def _validate_shard(
    shard: dict[str, Any], task: tuple[str, int], run_fingerprint: str
) -> None:
    kind, dataset_id = task
    if (
        shard.get("schema_version") != SCHEMA_VERSION
        or shard.get("run_fingerprint") != run_fingerprint
        or shard.get("kind") != kind
        or shard.get("dataset_id") != dataset_id
    ):
        raise RuntimeError(f"stale or malformed shard for {task}")
    expected = (
        len(SPLIT_SEEDS) * len(CONFIG_ORDER)
        if kind == "regression_ledger"
        else len(CANARY_SEEDS) * len(CANARY_CONFIGS)
    )
    if len(shard.get("records", ())) != expected:
        raise RuntimeError(f"incomplete shard for {task}")
    expected_seeds = SPLIT_SEEDS if kind == "regression_ledger" else CANARY_SEEDS
    expected_arms = (
        CONFIG_ORDER
        if kind == "regression_ledger"
        else tuple(CANARY_CONFIGS)
    )
    expected_coordinates = {
        (dataset_id, seed, arm)
        for seed in expected_seeds
        for arm in expected_arms
    }
    coordinates = []
    dataset_hash = shard.get("dataset_sha256")
    for record in shard["records"]:
        if (
            record.get("kind") != kind
            or record.get("dataset_id") != dataset_id
            or record.get("dataset_sha256") != dataset_hash
        ):
            raise RuntimeError(f"record boundary changed in shard for {task}")
        coordinates.append(
            (record["dataset_id"], record["seed"], record["arm"])
        )
    if set(coordinates) != expected_coordinates or len(coordinates) != len(
        expected_coordinates
    ):
        raise RuntimeError(f"coordinate boundary changed in shard for {task}")
    splits = shard.get("splits")
    if (
        not isinstance(splits, list)
        or {row.get("seed") for row in splits} != set(expected_seeds)
        or len(splits) != len(expected_seeds)
        or any(not row.get("indices_sha256") for row in splits)
    ):
        raise RuntimeError(f"split boundary changed in shard for {task}")


def build_manifest(source: dict[str, Any]) -> dict[str, Any]:
    freeze = validate_freeze_evidence()
    slices = frozen_slices()
    regression_ids = regression_dataset_ids()
    canary_ids = categorical_screen_canaries()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "generator_version": synthgen.VERSION,
        "protected_outcome_sources_accessed": False,
        "inputs": {
            "protocol_sha256": _sha256_file(PROTOCOL_PATH),
            "corpus_sha256": _sha256_file(CORPUS_PATH),
            "suites_sha256": _sha256_file(SUITES_PATH),
            "goldens_sha256": _sha256_file(GOLDENS_PATH),
            "freeze_sha256": _sha256_file(FREEZE_PATH),
            "runner_sha256": _sha256_file(RUNNER_PATH),
            "analyzer_sha256": _sha256_file(ANALYZER_PATH),
            "freeze_source": freeze["source"],
        },
        "benchmark": {
            "regression_dataset_ids": regression_ids,
            "categorical_canary_ids": canary_ids,
            "slices": slices,
            "split_seeds": list(SPLIT_SEEDS),
            "model_random_state_policy": "split_seed",
            "canary_seeds": list(CANARY_SEEDS),
            "thread_count": THREAD_COUNT,
            "config_order": list(CONFIG_ORDER),
            "configs": CONFIGS,
            "common_params": COMMON_PARAMS,
            "canary_configs": CANARY_CONFIGS,
            "canary_params": CANARY_PARAMS,
        },
    }
    manifest["run_fingerprint"] = _sha256_bytes(_canonical_json(manifest))
    return manifest


def run(output_dir: Path, workers: int) -> Path:
    if workers < 1:
        raise ValueError("workers must be positive")
    source = source_attestation()
    manifest = build_manifest(source)
    run_fingerprint = manifest["run_fingerprint"]
    tasks = [
        ("regression_ledger", dataset_id)
        for dataset_id in manifest["benchmark"]["regression_dataset_ids"]
    ] + [
        ("canary_no_variance", dataset_id)
        for dataset_id in manifest["benchmark"]["categorical_canary_ids"]
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError("output directory belongs to a different run")
    else:
        _atomic_json(manifest_path, manifest)

    shards: dict[tuple[str, int], dict[str, Any]] = {}
    pending = []
    for task in tasks:
        path = _shard_path(output_dir, task)
        if path.exists():
            shard = json.loads(path.read_text(encoding="utf-8"))
            _validate_shard(shard, task, run_fingerprint)
            shards[task] = shard
        else:
            pending.append(task)
    print(
        f"SynthGen df1: {len(tasks)} dataset jobs; "
        f"{len(shards)} resumed, {len(pending)} pending; workers={workers}",
        flush=True,
    )

    context = multiprocessing.get_context("spawn")
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context
    )
    future_to_task = {}
    try:
        future_to_task = {
            executor.submit(_worker, task, run_fingerprint): task
            for task in pending
        }
        completed = len(shards)
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            shard = future.result()
            _validate_shard(shard, task, run_fingerprint)
            _atomic_json(_shard_path(output_dir, task), shard)
            shards[task] = shard
            completed += 1
            print(
                f"  completed {completed}/{len(tasks)}: {task[0]} "
                f"{task[1]:03d}",
                flush=True,
            )
    except BaseException:
        for future in future_to_task:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    if set(shards) != set(tasks):
        raise RuntimeError("campaign ended with missing dataset shards")
    ending_source = source_attestation()
    if ending_source != source:
        raise RuntimeError("source attestation changed during the campaign")
    ordered_shards = [shards[task] for task in tasks]
    records = [
        record
        for shard in ordered_shards
        for record in shard["records"]
    ]
    raw = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "synthgen_darkofit_ledger_raw",
        "manifest": manifest,
        "environment": environment_metadata(),
        "shards": [
            {
                "kind": shard["kind"],
                "dataset_id": shard["dataset_id"],
                "sha256": _sha256_bytes(_canonical_json(shard)),
            }
            for shard in ordered_shards
        ],
        "datasets": [
            {
                "kind": shard["kind"],
                "dataset_id": shard["dataset_id"],
                "dataset_key": shard["dataset_key"],
                "dataset_sha256": shard["dataset_sha256"],
                "metadata": shard["metadata"],
                "splits": shard["splits"],
            }
            for shard in ordered_shards
        ],
        "records": records,
    }
    raw_path = output_dir / "raw.json"
    _atomic_json(raw_path, raw)
    print(
        f"wrote {raw_path} ({len(records)} coordinates; "
        f"sha256={_sha256_file(raw_path)})",
        flush=True,
    )
    return raw_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate clean source, freeze evidence, slices, and manifest only",
    )
    args = parser.parse_args()
    if args.dry_run:
        manifest = build_manifest(source_attestation())
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    run(args.output_dir.resolve(), args.workers)


if __name__ == "__main__":
    main()
