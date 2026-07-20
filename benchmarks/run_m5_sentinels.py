#!/usr/bin/env python3
"""Establish or check the non-ranking M5 diversity sentinel baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split

try:
    from benchmark_adapters import (
        build_dataset,
        make_sample_weight,
        split_case,
    )
    from campaign_lib.provenance import canonical_json_sha256, file_sha256
    from standing_evidence import (
        M5_ARMS,
        M5_BASELINE_EVIDENCE_PATH,
        M5_BASELINE_EVIDENCE_SHA256,
        M5_CONTRACT_FROZEN,
        M5_CONTROL_SOURCE,
        M5_SENTINEL_CASES,
        M5_SENTINEL_DOMAINS,
        M5_THREADS,
        contract_payload,
        m5_expected_grid,
    )
    from weighted_metrics import metric_bundle
except ImportError:  # pragma: no cover - supports `python -m benchmarks...`
    from benchmarks.benchmark_adapters import (
        build_dataset,
        make_sample_weight,
        split_case,
    )
    from benchmarks.campaign_lib.provenance import (
        canonical_json_sha256,
        file_sha256,
    )
    from benchmarks.standing_evidence import (
        M5_ARMS,
        M5_BASELINE_EVIDENCE_PATH,
        M5_BASELINE_EVIDENCE_SHA256,
        M5_CONTRACT_FROZEN,
        M5_CONTROL_SOURCE,
        M5_SENTINEL_CASES,
        M5_SENTINEL_DOMAINS,
        M5_THREADS,
        contract_payload,
        m5_expected_grid,
    )
    from benchmarks.weighted_metrics import metric_bundle


RUNNER_VERSION = "m5-sentinels-v1"
SCHEMA_VERSION = 1
WORKER_PREFIX = "M5_SENTINEL_RESULT="
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
CANARY_EXCESS_BRIER_MEAN_MAX = 0.005
CANARY_EXCESS_BRIER_WORST_MAX = 0.01


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"git {' '.join(arguments)} failed in {repository}: {detail}"
        )
    return result.stdout.strip()


def source_state(repository: Path) -> dict[str, Any]:
    repository = repository.expanduser().resolve()
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
        "package_tree": _git(repository, "rev-parse", "HEAD:darkofit"),
        "branch": _git(repository, "branch", "--show-current"),
        "clean": not status,
        "status": status,
    }


def validate_sources(
    harness: dict[str, Any],
    control: dict[str, Any],
    candidate: dict[str, Any],
    *,
    establishing: bool,
) -> None:
    dirty = [
        name
        for name, state in (
            ("harness", harness),
            ("control", control),
            ("candidate", candidate),
        )
        if not state["clean"]
    ]
    if dirty:
        raise RuntimeError(f"M5 sentinel sources are dirty: {dirty}")
    if control["head"] != M5_CONTROL_SOURCE:
        raise RuntimeError(
            f"M5 control is {control['head']}, expected {M5_CONTROL_SOURCE}"
        )
    if establishing and control["package_tree"] != candidate["package_tree"]:
        raise RuntimeError(
            "M5 baseline establishment requires behavior-identical package trees"
        )


def _case(domain_id: str):
    matches = [
        case for case in M5_SENTINEL_CASES if case.domain_id == domain_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"missing unique M5 case: {domain_id}")
    return matches[0]


def _domain(domain_id: str):
    matches = [
        domain for domain in M5_SENTINEL_DOMAINS if domain.id == domain_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"missing unique M5 domain: {domain_id}")
    return matches[0]


def _array_sha256(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    if array.dtype == object:
        for item in array.ravel(order="C"):
            encoded = (
                f"{type(item).__name__}:{item!r}".encode(
                    "utf-8", errors="backslashreplace"
                )
            )
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    else:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _split_hash(train: np.ndarray, test: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(train, dtype="<i8").tobytes()
        + b"\0"
        + np.asarray(test, dtype="<i8").tobytes()
    ).hexdigest()


def _grouped_dataset(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(20_260_720 + seed)
    group = np.repeat(np.arange(300, dtype=np.int64), 20)
    X = rng.normal(size=(len(group), 12))
    effects = rng.normal(0.0, 2.5, size=300)
    y = (
        3.0 * np.sin(X[:, 0])
        + 1.5 * X[:, 1]
        - 1.2 * X[:, 2] * X[:, 3]
        + effects[group]
        + rng.normal(0.0, 0.8, size=len(group))
    )
    order = rng.permutation(len(group))
    X, y, group = X[order], y[order], group[order]
    train, test = next(
        GroupShuffleSplit(
            n_splits=1, test_size=0.25, random_state=seed
        ).split(X, y, groups=group)
    )
    return {
        "task": "regression",
        "X_train": X[train],
        "X_test": X[test],
        "y_train": y[train],
        "y_test": y[test],
        "cat_features": [],
        "groups_train": group[train],
        "w_train": None,
        "w_test": None,
        "dataset_sha256": canonical_json_sha256(
            {
                "X": _array_sha256(X),
                "y": _array_sha256(y),
                "groups": _array_sha256(group),
            }
        ),
        "split_sha256": _split_hash(train, test),
        "bayes_brier": None,
    }


def _synthgen_dataset(dataset_key: str, seed: int) -> dict[str, Any]:
    import synthgen

    X, y, cat_features, task, metadata = synthgen.build_dataset(dataset_key)
    dataset_sha256 = synthgen.hash_dataset(dataset_key)
    indices = np.arange(len(y), dtype=np.int64)
    train, test = train_test_split(
        indices,
        test_size=0.25,
        random_state=seed,
        stratify=None if task == "regression" else y,
    )
    result = {
        "task": task,
        "X_train": X[train],
        "X_test": X[test],
        "y_train": np.asarray(y)[train],
        "y_test": np.asarray(y)[test],
        "cat_features": list(cat_features or []),
        "groups_train": None,
        "w_train": None,
        "w_test": None,
        "dataset_sha256": dataset_sha256,
        "split_sha256": _split_hash(train, test),
        "bayes_brier": metadata.get("bayes_brier"),
    }
    synthgen.build_dataset.cache_clear()
    return result


def _adapter_dataset(
    dataset_key: str,
    seed: int,
    *,
    weighted: bool,
) -> dict[str, Any]:
    _, dataset, size = dataset_key.split(":", 2)
    spec, X, y, cat_features = build_dataset(dataset, size, seed)
    weights = (
        make_sample_weight(y, spec.task, "stress") if weighted else None
    )
    split = split_case(X, y, spec.task, seed, weights)
    X_train = np.concatenate((split["X_fit"], split["X_val"]), axis=0)
    y_train = np.concatenate((split["y_fit"], split["y_val"]), axis=0)
    if split["w_fit"] is None:
        w_train = None
    else:
        w_train = np.concatenate((split["w_fit"], split["w_val"]), axis=0)
    return {
        "task": spec.task,
        "X_train": X_train,
        "X_test": split["X_test"],
        "y_train": y_train,
        "y_test": split["y_test"],
        "cat_features": list(cat_features or []),
        "groups_train": None,
        "w_train": w_train,
        "w_test": split["w_test"],
        "dataset_sha256": canonical_json_sha256(
            {
                "X": _array_sha256(X),
                "y": _array_sha256(y),
                "weight": (
                    None if weights is None else _array_sha256(weights)
                ),
            }
        ),
        "split_sha256": canonical_json_sha256(
            {
                "X_train": _array_sha256(X_train),
                "X_test": _array_sha256(split["X_test"]),
                "y_train": _array_sha256(y_train),
                "y_test": _array_sha256(split["y_test"]),
            }
        ),
        "bayes_brier": None,
    }


def build_case(domain_id: str, seed: int) -> dict[str, Any]:
    case = _case(domain_id)
    domain = _domain(domain_id)
    if case.dataset_key == "generic:grouped-entity-v1":
        result = _grouped_dataset(seed)
    elif case.dataset_key.startswith("syn:"):
        result = _synthgen_dataset(case.dataset_key, seed)
    elif case.dataset_key.startswith("adapter:"):
        result = _adapter_dataset(
            case.dataset_key, seed, weighted=domain.weighted
        )
    else:  # pragma: no cover - contract validation owns membership
        raise RuntimeError(f"unknown M5 dataset key: {case.dataset_key}")
    if result["task"] != domain.task:
        raise RuntimeError(f"M5 task drifted for {domain_id}")
    if case.dataset_sha256 and (
        result["dataset_sha256"] != case.dataset_sha256
    ):
        raise RuntimeError(f"M5 dataset hash drifted for {domain_id}")
    return result


def _prepare_source(source: Path) -> None:
    source = source.resolve()
    for name in list(sys.modules):
        if name == "darkofit" or name.startswith("darkofit."):
            del sys.modules[name]
    sys.path = [
        entry
        for entry in sys.path
        if entry and Path(entry).resolve() not in {REPO_ROOT, source}
    ]
    sys.path.insert(0, str(source))


def _model_params(profile: str, seed: int, *, warmup: bool) -> dict[str, Any]:
    iterations = {
        "standard": 300,
        "high_row": 120,
        "canary": 2000,
        "group_ensemble": 150,
    }[profile]
    params = {
        "iterations": 3 if warmup else iterations,
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "max_bins": 128,
        "tree_mode": "catboost",
        "ordered_boosting": False,
        "early_stopping": False,
        "refit": False,
        "thread_count": M5_THREADS,
        "random_state": seed,
        "diagnostic_warnings": "never",
        "verbose_timing": True,
    }
    if profile == "canary" and not warmup:
        params.update(
            {
                "early_stopping": True,
                "early_stopping_rounds": 50,
                "validation_fraction": 0.2,
                "use_best_model": True,
                "ts_permutations": 1,
            }
        )
    if profile == "group_ensemble":
        params.update(
            {
                "n_ensembles": 3,
                "ensemble_bootstrap": "groups",
                "ensemble_shared_preprocessing": True,
            }
        )
    return params


def _fit_kwargs(data: dict[str, Any], rows: Optional[int] = None):
    limit = slice(None) if rows is None else slice(0, rows)
    kwargs = {"cat_features": data["cat_features"]}
    if data["groups_train"] is not None:
        kwargs["groups"] = data["groups_train"][limit]
    if data["w_train"] is not None:
        kwargs["sample_weight"] = data["w_train"][limit]
    return kwargs


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("M5 worker peak RSS is unavailable")
    return value


def _quality_baseline(
    task: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    w_train: Optional[np.ndarray],
    w_test: Optional[np.ndarray],
) -> float:
    if task == "regression":
        center = float(np.average(y_train, weights=w_train))
        return float(
            mean_squared_error(
                y_test,
                np.full(len(y_test), center),
                sample_weight=w_test,
            )
            ** 0.5
        )
    classes = np.unique(y_train)
    counts = np.asarray(
        [
            np.sum(
                (y_train == label)
                * (1.0 if w_train is None else w_train)
            )
            for label in classes
        ],
        dtype=np.float64,
    )
    prior = counts / counts.sum()
    return float(
        log_loss(
            y_test,
            np.tile(prior, (len(y_test), 1)),
            labels=classes,
            sample_weight=w_test,
        )
    )


def _model_metadata(model: Any) -> dict[str, Any]:
    members = list(getattr(model, "estimators_", ()) or ())
    fitted = members if members else [model]
    cores = [member.model_ for member in fitted]
    return {
        "member_count": len(fitted),
        "tree_count": int(sum(len(core.trees_) for core in cores)),
        "tree_modes": sorted({str(core.tree_mode_) for core in cores}),
        "resolved_thread_counts": sorted(
            {int(core.n_threads_) for core in cores}
        ),
        "classes": (
            None
            if not hasattr(model, "classes_")
            else np.asarray(model.classes_).tolist()
        ),
    }


def run_worker(payload: dict[str, Any]) -> dict[str, Any]:
    domain_id = payload["domain_id"]
    seed = int(payload["seed"])
    case = _case(domain_id)
    data = build_case(domain_id, seed)
    _prepare_source(Path(payload["source"]))
    from darkofit import DarkoClassifier, DarkoRegressor

    estimator = (
        DarkoRegressor
        if data["task"] == "regression"
        else DarkoClassifier
    )
    warm_rows = min(512, len(data["y_train"]))
    warm = estimator(**_model_params(case.model_profile, seed, warmup=True))
    warm.fit(
        data["X_train"][:warm_rows],
        data["y_train"][:warm_rows],
        **_fit_kwargs(data, warm_rows),
    )
    if data["task"] == "regression":
        warm.predict(data["X_test"][:32])
    else:
        warm.predict_proba(data["X_test"][:32])

    model = estimator(**_model_params(case.model_profile, seed, warmup=False))
    started = time.perf_counter_ns()
    model.fit(
        data["X_train"],
        data["y_train"],
        **_fit_kwargs(data),
    )
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    started = time.perf_counter_ns()
    prediction = np.asarray(model.predict(data["X_test"]))
    probability = (
        None
        if data["task"] == "regression"
        else np.asarray(model.predict_proba(data["X_test"]), dtype=np.float64)
    )
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    if (
        prediction.shape != (len(data["y_test"]),)
        or not np.isfinite(np.asarray(prediction, dtype=np.float64)).all()
        or (
            probability is not None
            and (
                probability.shape[0] != len(data["y_test"])
                or not np.isfinite(probability).all()
                or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-12)
            )
        )
    ):
        raise RuntimeError("M5 prediction invariant failed")

    with tempfile.TemporaryDirectory(prefix="darkofit-m5-model-") as directory:
        model_path = Path(directory) / "model.npz"
        model.save_model(model_path)
        model_bytes = model_path.stat().st_size
        loaded = estimator.load_model(model_path)
        loaded_prediction = np.asarray(loaded.predict(data["X_test"]))
        loaded_probability = (
            None
            if data["task"] == "regression"
            else np.asarray(
                loaded.predict_proba(data["X_test"]), dtype=np.float64
            )
        )
    roundtrip_exact = np.array_equal(prediction, loaded_prediction) and (
        probability is None
        or np.array_equal(probability, loaded_probability)
    )
    if not roundtrip_exact:
        raise RuntimeError("M5 serialization roundtrip changed predictions")

    metrics = metric_bundle(
        data["task"],
        data["y_test"],
        prediction,
        proba=probability,
        labels=getattr(model, "classes_", None),
        sample_weight=data["w_test"],
    )
    primary_value = float(metrics["primary_value"])
    trivial_loss = _quality_baseline(
        data["task"],
        np.asarray(data["y_train"]),
        np.asarray(data["y_test"]),
        data["w_train"],
        data["w_test"],
    )
    normalized_loss = primary_value / trivial_loss
    if (
        not math.isfinite(normalized_loss)
        or normalized_loss < 0.0
        or normalized_loss > case.expected_normalized_loss_max
    ):
        raise RuntimeError(
            f"M5 normalized loss {normalized_loss} is outside [0, "
            f"{case.expected_normalized_loss_max}]"
        )
    metadata = _model_metadata(model)
    behavior = {
        "prediction_sha256": _array_sha256(prediction),
        "probability_sha256": (
            None if probability is None else _array_sha256(probability)
        ),
        "model_metadata": metadata,
        "primary_metric": metrics["primary_metric"],
        "primary_value": primary_value,
        "normalized_loss": normalized_loss,
    }
    return {
        "status": "ok",
        "arm": payload["arm"],
        "source": str(Path(payload["source"]).resolve()),
        "domain_id": domain_id,
        "dataset_key": case.dataset_key,
        "task": data["task"],
        "seed": seed,
        "weighted": _domain(domain_id).weighted,
        "n_train": int(len(data["y_train"])),
        "n_test": int(len(data["y_test"])),
        "n_features": int(np.asarray(data["X_train"]).shape[1]),
        "dataset_sha256": data["dataset_sha256"],
        "split_sha256": data["split_sha256"],
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "worker_peak_rss_bytes": _peak_rss_bytes(),
        "serialized_model_bytes": int(model_bytes),
        "roundtrip_exact": roundtrip_exact,
        "metrics": metrics,
        "trivial_primary_loss": trivial_loss,
        "normalized_loss": normalized_loss,
        "excess_brier": (
            None
            if not case.known_floor
            else float(metrics["brier"] - float(data["bayes_brier"] or 0.0))
        ),
        "model_metadata": metadata,
        "prediction_sha256": behavior["prediction_sha256"],
        "probability_sha256": behavior["probability_sha256"],
        "behavior_fingerprint_sha256": canonical_json_sha256(behavior),
        "thread_environment": {
            key: os.environ.get(key) for key in THREAD_ENV_KEYS
        },
    }


def _worker_main(payload_path: Path) -> None:
    try:
        row = run_worker(json.loads(payload_path.read_text()))
    except Exception:
        row = {"status": "error", "error": traceback.format_exc()}
    print(WORKER_PREFIX + json.dumps(row, sort_keys=True, allow_nan=False))


def _worker_environment(cache_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    for key in THREAD_ENV_KEYS:
        environment[key] = str(M5_THREADS)
    environment.update(
        {
            "DARKOFIT_WARMUP": "0",
            "NUMBA_DISABLE_JIT": "0",
            "NUMBA_CACHE_DIR": str(cache_dir),
            "PYTHONHASHSEED": "0",
        }
    )
    return environment


def _run_worker(payload: dict[str, Any], temporary: Path) -> dict[str, Any]:
    identity = (
        f"{payload['arm']}-{payload['domain_id']}-{payload['seed']}"
    )
    payload_path = temporary / f"{identity}.json"
    payload_path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False)
    )
    process = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--worker", str(payload_path)],
        check=False,
        capture_output=True,
        text=True,
        env=_worker_environment(temporary / "numba-cache"),
    )
    matches = [
        line[len(WORKER_PREFIX) :]
        for line in process.stdout.splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if process.returncode != 0 or len(matches) != 1:
        raise RuntimeError(
            f"M5 worker failed for {identity}: returncode={process.returncode}"
            f"\nstdout={process.stdout}\nstderr={process.stderr}"
        )
    row = json.loads(matches[0])
    row["stderr"] = process.stderr.strip()
    if row.get("status") != "ok":
        raise RuntimeError(f"M5 worker failed for {identity}:\n{row.get('error')}")
    return row


def _identity(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["arm"], row["domain_id"], int(row["seed"])


def _ratio_summary(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def analyze_rows(
    rows: list[dict[str, Any]],
    *,
    establishing: bool,
    baseline: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    expected = set(m5_expected_grid())
    identities = [_identity(row) for row in rows]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise RuntimeError("M5 row grid is incomplete or duplicated")
    if any(row.get("status") != "ok" for row in rows):
        raise RuntimeError("M5 contains a failed row")
    by_cell: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_cell[(row["domain_id"], int(row["seed"]))][row["arm"]] = row
    behavior_equal = True
    fit_ratios = []
    predict_ratios = []
    rss_ratios = []
    for key, pair in by_cell.items():
        if set(pair) != set(M5_ARMS):
            raise RuntimeError(f"M5 pair is incomplete: {key}")
        if (
            pair["control"]["dataset_sha256"]
            != pair["candidate"]["dataset_sha256"]
            or pair["control"]["split_sha256"]
            != pair["candidate"]["split_sha256"]
        ):
            raise RuntimeError(f"M5 pair data drifted: {key}")
        behavior_equal &= (
            pair["control"]["behavior_fingerprint_sha256"]
            == pair["candidate"]["behavior_fingerprint_sha256"]
        )
        fit_ratios.append(
            float(pair["candidate"]["fit_seconds"])
            / float(pair["control"]["fit_seconds"])
        )
        predict_ratios.append(
            float(pair["candidate"]["predict_seconds"])
            / float(pair["control"]["predict_seconds"])
        )
        rss_ratios.append(
            float(pair["candidate"]["worker_peak_rss_bytes"])
            / float(pair["control"]["worker_peak_rss_bytes"])
        )
    if establishing and not behavior_equal:
        raise RuntimeError(
            "behavior-identical M5 baseline sources produced different fingerprints"
        )

    canary = {}
    for arm in M5_ARMS:
        for case in M5_SENTINEL_CASES:
            if not case.known_floor:
                continue
            values = [
                float(row["excess_brier"])
                for row in rows
                if row["arm"] == arm and row["domain_id"] == case.domain_id
            ]
            mean = float(np.mean(values))
            worst = float(max(values))
            passed = (
                mean <= CANARY_EXCESS_BRIER_MEAN_MAX
                and worst <= CANARY_EXCESS_BRIER_WORST_MAX
            )
            canary[f"{arm}:{case.domain_id}"] = {
                "values": values,
                "mean": mean,
                "worst": worst,
                "passed": passed,
            }
            if not passed:
                raise RuntimeError(
                    f"M5 known floor failed for {arm}/{case.domain_id}"
                )

    baseline_drift = []
    if baseline is not None:
        frozen = {
            _identity(row): row for row in baseline["rows"]
        }
        for row in rows:
            if row["arm"] != "control":
                continue
            prior = frozen.get(_identity(row))
            if prior is None:
                baseline_drift.append(
                    {"identity": _identity(row), "reason": "missing"}
                )
                continue
            for field in (
                "dataset_sha256",
                "split_sha256",
                "behavior_fingerprint_sha256",
            ):
                if row[field] != prior[field]:
                    baseline_drift.append(
                        {
                            "identity": _identity(row),
                            "reason": field,
                        }
                    )
    return {
        "paired_cells": len(by_cell),
        "behavior_fingerprints_equal_between_arms": behavior_equal,
        "known_floor_checks": canary,
        "candidate_over_control_fit_seconds_ratio": _ratio_summary(fit_ratios),
        "candidate_over_control_predict_seconds_ratio": _ratio_summary(
            predict_ratios
        ),
        "candidate_over_control_peak_rss_ratio": _ratio_summary(rss_ratios),
        "baseline_drift": baseline_drift,
        "advancement_blocked_for_drift": bool(baseline_drift),
        "ranking_or_acceptance_score": False,
    }


def _write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    baseline = None
    establishing = args.baseline is None
    if establishing and M5_CONTRACT_FROZEN:
        raise RuntimeError("frozen M5 requires its baseline for every check")
    if not establishing:
        baseline_path = args.baseline.expanduser().resolve()
        if (
            not M5_CONTRACT_FROZEN
            or str(baseline_path.relative_to(REPO_ROOT))
            != M5_BASELINE_EVIDENCE_PATH
            or file_sha256(baseline_path) != M5_BASELINE_EVIDENCE_SHA256
        ):
            raise RuntimeError("M5 baseline evidence is not contract-bound")
        baseline = json.loads(baseline_path.read_text())

    control = args.control.expanduser().resolve()
    candidate = args.candidate.expanduser().resolve()
    sources = {
        "harness": source_state(REPO_ROOT),
        "control": source_state(control),
        "candidate": source_state(candidate),
    }
    validate_sources(**sources, establishing=establishing)
    with tempfile.TemporaryDirectory(prefix="darkofit-m5-") as directory:
        temporary = Path(directory)
        grouped: dict[tuple[str, int], list[tuple[str, str, int]]] = defaultdict(list)
        for identity in m5_expected_grid():
            grouped[(identity[1], identity[2])].append(identity)
        rows = []
        for cell_index, cell in enumerate(sorted(grouped, key=str)):
            identities = sorted(grouped[cell])
            if cell_index % 2:
                identities.reverse()
            for arm, domain_id, seed in identities:
                source = control if arm == "control" else candidate
                row = _run_worker(
                    {
                        "arm": arm,
                        "source": str(source),
                        "domain_id": domain_id,
                        "seed": seed,
                    },
                    temporary,
                )
                rows.append(row)
                print(
                    f"ok {arm:9s} {domain_id:32s} seed={seed}",
                    flush=True,
                )
        analysis = analyze_rows(
            rows, establishing=establishing, baseline=baseline
        )
    sources_after = {
        "harness": source_state(REPO_ROOT),
        "control": source_state(control),
        "candidate": source_state(candidate),
    }
    if sources_after != sources:
        raise RuntimeError("M5 source changed during execution")
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": (
            "baseline_establishment" if establishing else "sentinel_check"
        ),
        "non_ranking": True,
        "shipping_or_default_claim_authorized": False,
        "contract": contract_payload(),
        "contract_sha256": canonical_json_sha256(contract_payload()),
        "runner_sha256": file_sha256(RUNNER_PATH),
        "sources": sources,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "logical_cpu_count": os.cpu_count(),
        },
        "execution": {
            "threads": M5_THREADS,
            "fresh_worker_per_arm_cell": True,
            "arm_order": "alternating by domain/seed cell",
            "warmup": "same-source 3-tree fit and prediction outside timing",
            "row_count": len(rows),
        },
        "rows": rows,
        "analysis": analysis,
    }
    _write_create_only(
        output,
        (
            json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8"),
    )
    print(f"wrote M5 sentinels to {output}")
    print(f"artifact sha256: {file_sha256(output)}")
    return output


def main(argv: Optional[list[str]] = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--worker"]:
        if len(arguments) != 2:
            raise SystemExit("--worker requires one payload path")
        _worker_main(Path(arguments[1]))
        return
    run(parse_args(arguments))


if __name__ == "__main__":
    main()
