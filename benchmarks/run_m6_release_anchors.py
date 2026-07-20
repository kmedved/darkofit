#!/usr/bin/env python3
"""Establish the pinned ChimeraBoost and CatBoost release anchors for M6."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from benchmark_adapters import (
        DATASETS,
        build_dataset,
        make_sample_weight,
        split_case,
    )
    from campaign_lib.provenance import canonical_json_sha256, file_sha256
    from standing_evidence import (
        M6_DATASETS,
        M6_RELEASE_ANCHORS,
        M6_SEED_COUNT,
        M6_SIZES,
        M6_SMOKE_DATASETS,
        M6_THREADS,
        M6_WEIGHT_MODES,
        contract_payload,
    )
    from weighted_metrics import metric_bundle
except ImportError:  # pragma: no cover - supports `python -m benchmarks...`
    from benchmarks.benchmark_adapters import (
        DATASETS,
        build_dataset,
        make_sample_weight,
        split_case,
    )
    from benchmarks.campaign_lib.provenance import (
        canonical_json_sha256,
        file_sha256,
    )
    from benchmarks.standing_evidence import (
        M6_DATASETS,
        M6_RELEASE_ANCHORS,
        M6_SEED_COUNT,
        M6_SIZES,
        M6_SMOKE_DATASETS,
        M6_THREADS,
        M6_WEIGHT_MODES,
        contract_payload,
    )
    from benchmarks.weighted_metrics import metric_bundle


RUNNER_VERSION = "m6-release-anchors-v1"
SCHEMA_VERSION = 1
WORKER_PREFIX = "M6_RELEASE_ANCHOR_RESULT="
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


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
        "branch": _git(repository, "branch", "--show-current"),
        "clean": not status,
        "status": status,
    }


def _catboost_record() -> Path:
    distribution = importlib.metadata.distribution("catboost")
    return Path(
        distribution.locate_file(
            f"catboost-{distribution.version}.dist-info/RECORD"
        )
    )


def installed_catboost_state() -> dict[str, str]:
    record = _catboost_record()
    return {
        "version": importlib.metadata.version("catboost"),
        "record_path": str(record),
        "record_sha256": file_sha256(record),
    }


def expected_anchor_map() -> dict[str, dict[str, str]]:
    return {
        anchor.id: {
            "version": anchor.version,
            "source_pin": anchor.source_pin,
        }
        for anchor in M6_RELEASE_ANCHORS
    }


def validate_sources(
    harness: dict[str, Any],
    chimera: dict[str, Any],
    catboost: dict[str, str],
) -> None:
    if not harness["clean"]:
        raise RuntimeError("M6 release anchors require a clean harness source")
    if not chimera["clean"]:
        raise RuntimeError("M6 release anchors require a clean ChimeraBoost source")
    expected = expected_anchor_map()
    if set(expected) != {"chimeraboost", "catboost"}:
        raise RuntimeError("M6 release-anchor membership drifted")
    chimera_pin = expected["chimeraboost"]["source_pin"]
    if chimera_pin != f"git:{chimera['head']}":
        raise RuntimeError(
            f"ChimeraBoost source is {chimera['head']}, expected {chimera_pin}"
        )
    if expected["catboost"]["version"] != catboost["version"]:
        raise RuntimeError("CatBoost release-anchor version drifted")
    record_pin = f"record-sha256:{catboost['record_sha256']}"
    if expected["catboost"]["source_pin"] != record_pin:
        raise RuntimeError(
            f"CatBoost installation is {record_pin}, expected "
            f"{expected['catboost']['source_pin']}"
        )


def expected_coordinates(*, smoke: bool) -> tuple[tuple[Any, ...], ...]:
    datasets = M6_SMOKE_DATASETS if smoke else M6_DATASETS
    sizes = ("small",) if smoke else M6_SIZES
    seeds = range(1 if smoke else M6_SEED_COUNT)
    weights = ("none",) if smoke else M6_WEIGHT_MODES
    return tuple(
        product(
            ("chimeraboost", "catboost"),
            datasets,
            sizes,
            seeds,
            weights,
        )
    )


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


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    if value <= 0:
        raise RuntimeError("worker peak RSS is unavailable")
    return value


def _catboost_frames(
    X_train: Any,
    X_test: Any,
    categorical_indices: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(np.asarray(X_train).copy())
    test = pd.DataFrame(np.asarray(X_test).copy())
    categorical = set(categorical_indices)
    for index in range(train.shape[1]):
        if index not in categorical:
            train[index] = pd.to_numeric(train[index], errors="raise")
            test[index] = pd.to_numeric(test[index], errors="raise")
            continue
        combined = pd.concat((train[index], test[index]), ignore_index=True)
        codes, _ = pd.factorize(combined, sort=False)
        tokens = np.asarray(
            [
                "__DARKOFIT_MISSING_CATEGORY__"
                if code < 0
                else f"__DARKOFIT_CATEGORY_{code}__"
                for code in codes
            ],
            dtype=object,
        )
        train[index] = tokens[: len(train)]
        test[index] = tokens[len(train) :]
    return train, test


def _chimera_estimator(task: str, seed: int, *, warmup: bool):
    from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

    estimator = (
        ChimeraBoostRegressor
        if task == "regression"
        else ChimeraBoostClassifier
    )
    kwargs = {"thread_count": M6_THREADS, "random_state": seed}
    if warmup:
        kwargs["n_estimators"] = 3
        kwargs["early_stopping"] = False
    return estimator(**kwargs)


def _catboost_estimator(task: str, seed: int, *, warmup: bool):
    from catboost import CatBoostClassifier, CatBoostRegressor

    estimator = CatBoostRegressor if task == "regression" else CatBoostClassifier
    kwargs = {
        "random_seed": seed,
        "thread_count": M6_THREADS,
        "verbose": False,
        "allow_writing_files": False,
    }
    if warmup:
        kwargs["iterations"] = 3
    return estimator(**kwargs)


def _fit_model(
    anchor_id: str,
    task: str,
    seed: int,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    cat_features: list[int],
    sample_weight: Any,
):
    if anchor_id == "chimeraboost":
        estimator_factory = _chimera_estimator
        train_data, test_data = X_train, X_test
    elif anchor_id == "catboost":
        estimator_factory = _catboost_estimator
        train_data, test_data = _catboost_frames(
            X_train, X_test, cat_features
        )
    else:  # pragma: no cover - validated by the parent
        raise ValueError(f"unknown release anchor: {anchor_id}")

    warm_rows = min(256, len(y_train))
    warm = estimator_factory(task, seed, warmup=True)
    warm_kwargs = {"cat_features": cat_features}
    if sample_weight is not None:
        warm_kwargs["sample_weight"] = np.asarray(sample_weight)[:warm_rows]
    warm.fit(
        train_data.iloc[:warm_rows]
        if isinstance(train_data, pd.DataFrame)
        else train_data[:warm_rows],
        np.asarray(y_train)[:warm_rows],
        **warm_kwargs,
    )
    warm_test = (
        test_data.iloc[: min(32, len(test_data))]
        if isinstance(test_data, pd.DataFrame)
        else test_data[: min(32, len(test_data))]
    )
    if task == "regression":
        warm.predict(warm_test)
    else:
        warm.predict_proba(warm_test)

    model = estimator_factory(task, seed, warmup=False)
    fit_kwargs = {"cat_features": cat_features}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    started = time.perf_counter_ns()
    model.fit(train_data, y_train, **fit_kwargs)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    started = time.perf_counter_ns()
    prediction = np.asarray(model.predict(test_data)).reshape(-1)
    if task == "regression":
        probability = None
    else:
        probability = np.asarray(model.predict_proba(test_data), dtype=np.float64)
    predict_seconds = (time.perf_counter_ns() - started) / 1e9
    return model, prediction, probability, fit_seconds, predict_seconds


def _model_metadata(anchor_id: str, model: Any) -> dict[str, Any]:
    if anchor_id == "catboost":
        return {
            "tree_count": int(model.tree_count_),
            "classes": (
                None
                if not hasattr(model, "classes_")
                else np.asarray(model.classes_).tolist()
            ),
        }
    core = model.model_
    return {
        "tree_count": int(len(core.trees_)),
        "resolved_learning_rate": float(core.lr_),
        "linear_leaves_selected": bool(
            getattr(model, "linear_leaves_selected_", False)
        ),
        "cross_features_selected": bool(
            getattr(model, "cross_features_selected_", False)
        ),
        "classes": (
            None
            if not hasattr(model, "classes_")
            else np.asarray(model.classes_).tolist()
        ),
    }


def run_worker(payload: dict[str, Any]) -> dict[str, Any]:
    anchor_id = payload["anchor_id"]
    if anchor_id == "chimeraboost":
        source = str(Path(payload["chimeraboost_source"]).resolve())
        sys.path = [entry for entry in sys.path if entry != source]
        sys.path.insert(0, source)
    spec, X, y, cat_features = build_dataset(
        payload["dataset"], payload["size"], int(payload["seed"])
    )
    cat_features = list(cat_features or [])
    weights = make_sample_weight(y, spec.task, payload["weight_mode"])
    split = split_case(X, y, spec.task, int(payload["seed"]), weights)
    X_train = np.concatenate((split["X_fit"], split["X_val"]), axis=0)
    y_train = np.concatenate((split["y_fit"], split["y_val"]), axis=0)
    if split["w_fit"] is None:
        w_train = None
    else:
        w_train = np.concatenate((split["w_fit"], split["w_val"]), axis=0)
    model, prediction, probability, fit_seconds, predict_seconds = _fit_model(
        anchor_id,
        spec.task,
        int(payload["seed"]),
        X_train,
        y_train,
        split["X_test"],
        cat_features,
        w_train,
    )
    if prediction.shape != (len(split["y_test"]),):
        raise RuntimeError("release-anchor prediction shape drifted")
    if not np.isfinite(np.asarray(prediction, dtype=np.float64)).all():
        raise RuntimeError("release-anchor prediction is non-finite")
    if probability is not None and (
        probability.ndim != 2
        or probability.shape[0] != len(split["y_test"])
        or not np.isfinite(probability).all()
    ):
        raise RuntimeError("release-anchor probability is invalid")
    labels = getattr(model, "classes_", None)
    metrics = metric_bundle(
        spec.task,
        split["y_test"],
        prediction,
        proba=probability,
        labels=labels,
        sample_weight=split["w_test"],
    )
    values = np.asarray(list(metrics.values()), dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("release-anchor metrics are non-finite")
    return {
        "status": "ok",
        "anchor_id": anchor_id,
        "dataset": payload["dataset"],
        "task": spec.task,
        "size": payload["size"],
        "seed": int(payload["seed"]),
        "weight_mode": payload["weight_mode"],
        "n_train": int(len(y_train)),
        "n_test": int(len(split["y_test"])),
        "n_features": int(split["n_features"]),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "worker_peak_rss_bytes": _peak_rss_bytes(),
        "dataset_sha256": canonical_json_sha256(
            {
                "X": _array_sha256(X),
                "y": _array_sha256(y),
                "weight": None if weights is None else _array_sha256(weights),
            }
        ),
        "prediction_sha256": _array_sha256(prediction),
        "probability_sha256": (
            None if probability is None else _array_sha256(probability)
        ),
        "metrics": metrics,
        "model_metadata": _model_metadata(anchor_id, model),
        "thread_environment": {
            key: os.environ.get(key) for key in THREAD_ENV_KEYS
        },
    }


def _worker_main(payload_path: Path) -> None:
    try:
        row = run_worker(json.loads(payload_path.read_text()))
    except Exception:
        row = {
            "status": "error",
            "error": traceback.format_exc(),
        }
    print(WORKER_PREFIX + json.dumps(row, sort_keys=True, allow_nan=False))


def _worker_environment(cache_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for key in THREAD_ENV_KEYS:
        environment[key] = str(M6_THREADS)
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
    identity = "-".join(
        str(payload[key])
        for key in ("anchor_id", "dataset", "size", "seed", "weight_mode")
    )
    payload_path = temporary / f"{identity}.json"
    payload_path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False)
    )
    process = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--worker",
            str(payload_path),
        ],
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
            f"release-anchor worker failed for {identity}: "
            f"returncode={process.returncode}\nstdout={process.stdout}\n"
            f"stderr={process.stderr}"
        )
    row = json.loads(matches[0])
    row["stderr"] = process.stderr.strip()
    if row.get("status") != "ok":
        raise RuntimeError(
            f"release-anchor worker failed for {identity}:\n"
            f"{row.get('error')}"
        )
    return row


def _coordinate(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["anchor_id"],
        row["dataset"],
        row["size"],
        int(row["seed"]),
        row["weight_mode"],
    )


def validate_rows(rows: list[dict[str, Any]], *, smoke: bool) -> None:
    expected = set(expected_coordinates(smoke=smoke))
    observed = [_coordinate(row) for row in rows]
    if len(observed) != len(set(observed)):
        raise RuntimeError("duplicate release-anchor coordinates")
    if set(observed) != expected:
        raise RuntimeError(
            f"release-anchor grid mismatch: missing={sorted(expected-set(observed))}, "
            f"unexpected={sorted(set(observed)-expected)}"
        )
    for row in rows:
        if row.get("status") != "ok":
            raise RuntimeError(f"release-anchor row failed: {_coordinate(row)}")
        for field in (
            "fit_seconds",
            "predict_seconds",
            "worker_peak_rss_bytes",
        ):
            value = float(row[field])
            if not math.isfinite(value) or value <= 0.0:
                raise RuntimeError(
                    f"release-anchor {field} is invalid: {_coordinate(row)}"
                )


def _write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chimeraboost-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    chimera_root = args.chimeraboost_source.expanduser().resolve()
    harness_before = source_state(REPO_ROOT)
    chimera_before = source_state(chimera_root)
    catboost = installed_catboost_state()
    validate_sources(harness_before, chimera_before, catboost)

    import tempfile

    rows = []
    with tempfile.TemporaryDirectory(prefix="darkofit-m6-anchors-") as directory:
        temporary = Path(directory)
        coordinates = expected_coordinates(smoke=args.smoke)
        # Alternate product order by cell without changing the frozen grid.
        grouped: dict[tuple[Any, ...], list[tuple[Any, ...]]] = {}
        for coordinate in coordinates:
            grouped.setdefault(coordinate[1:], []).append(coordinate)
        for cell_index, cell in enumerate(sorted(grouped, key=str)):
            arms = sorted(grouped[cell], key=lambda item: item[0])
            if cell_index % 2:
                arms.reverse()
            for anchor_id, dataset, size, seed, weight_mode in arms:
                row = _run_worker(
                    {
                        "anchor_id": anchor_id,
                        "dataset": dataset,
                        "size": size,
                        "seed": seed,
                        "weight_mode": weight_mode,
                        "chimeraboost_source": str(chimera_root),
                    },
                    temporary,
                )
                rows.append(row)
                print(
                    f"ok {anchor_id:13s} {dataset:24s} {size:6s} "
                    f"seed={seed} weights={weight_mode}",
                    flush=True,
                )
    validate_rows(rows, smoke=args.smoke)
    harness_after = source_state(REPO_ROOT)
    chimera_after = source_state(chimera_root)
    if harness_after != harness_before or chimera_after != chimera_before:
        raise RuntimeError("release-anchor source changed during execution")
    contract = contract_payload()
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "smoke": bool(args.smoke),
        "evidence_status": (
            "harness_smoke" if args.smoke else "release_anchor_establishment"
        ),
        "contract": contract,
        "contract_sha256": canonical_json_sha256(contract),
        "runner_sha256": file_sha256(RUNNER_PATH),
        "harness_source": harness_before,
        "chimeraboost_source": chimera_before,
        "catboost_installation": catboost,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "logical_cpu_count": os.cpu_count(),
        },
        "execution": {
            "threads": M6_THREADS,
            "fresh_worker_per_row": True,
            "warmup": "same-product 3-tree fit and prediction outside timing",
            "public_default_policy": (
                "only thread count and random seed are fixed"
            ),
            "row_count": len(rows),
        },
        "rows": rows,
    }
    _write_create_only(
        output,
        (
            json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8"),
    )
    print(f"wrote M6 release anchors to {output}")
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
