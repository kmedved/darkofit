"""Attested out-of-band warmup for the same-machine comparator campaign."""

from __future__ import annotations

import hashlib
import json
import math
import operator
import os
import time
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from autogluon.core.metrics import get_metric

try:
    from benchmarks.tabarena_comparator_adapters import (
        COMPARATOR_METADATA_KEY,
        REPRESENTATION_METADATA_KEY,
        ComparatorCatBoostModel,
        ComparatorChimeraBoostModel,
        ComparatorDarkoFitModel,
    )
except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
    from tabarena_comparator_adapters import (
        COMPARATOR_METADATA_KEY,
        REPRESENTATION_METADATA_KEY,
        ComparatorCatBoostModel,
        ComparatorChimeraBoostModel,
        ComparatorDarkoFitModel,
    )


WARMUP_SCHEMA_VERSION = 1
WARMUP_KIND = "darkofit_tabarena_regression_same_machine_warmup"
THREAD_COUNT = 18
SEED = 20_260_719
TRAIN_ROWS = 1_200
VALIDATION_ROWS = 400
FEATURE_COUNT = 12
SMALL_PREDICTION_ROWS = 31
LARGE_PREDICTION_ROWS = 16_384
TIME_LIMIT_SECONDS = 300.0

MODEL_CLASSES = {
    "darkofit": ComparatorDarkoFitModel,
    "chimeraboost": ComparatorChimeraBoostModel,
    "catboost": ComparatorCatBoostModel,
}
STAGE_NAMES = tuple(
    f"{engine}_{input_kind}"
    for engine in MODEL_CLASSES
    for input_kind in ("numeric", "categorical")
)


def _elapsed_seconds(start_ns: int) -> float:
    return float(time.monotonic_ns() - start_ns) / 1_000_000_000.0


def _prediction_fingerprint(prediction: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(prediction, dtype="<f8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _normalize_thread_count(thread_count: int) -> int:
    if isinstance(thread_count, (bool, np.bool_)):
        raise TypeError("thread_count must be a positive integer")
    try:
        value = operator.index(thread_count)
    except TypeError as exc:
        raise TypeError("thread_count must be a positive integer") from exc
    if value != THREAD_COUNT:
        raise ValueError(f"thread_count is frozen at {THREAD_COUNT}")
    return int(value)


def _warmup_environment_value() -> str | None:
    value = os.environ.get("CHIMERABOOST_WARMUP")
    if value is not None and value != "" and value.strip() != "0":
        raise RuntimeError(
            "CHIMERABOOST_WARMUP must be unset, empty, or zero"
        )
    return value


def _make_data() -> dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    rng = np.random.default_rng(SEED)
    n_rows = TRAIN_ROWS + VALIDATION_ROWS
    numeric = rng.normal(size=(n_rows, FEATURE_COUNT))
    category_codes = rng.integers(0, 17, size=n_rows)
    noise = rng.normal(scale=0.05, size=n_rows)
    target = (
        1.7 * numeric[:, 0]
        - 0.8 * numeric[:, 1]
        + 0.35 * numeric[:, 2] * numeric[:, 3]
        + np.sin(numeric[:, 4])
        + np.linspace(-0.8, 0.8, 17)[category_codes]
        + noise
    )
    columns = [f"feature_{index}" for index in range(FEATURE_COUNT)]
    numeric_frame = pd.DataFrame(numeric, columns=columns)
    categorical_frame = numeric_frame.copy()
    categorical_frame["category"] = pd.Categorical(
        [f"category_{code}" for code in category_codes],
        categories=[f"category_{code}" for code in range(17)],
    )
    target_series = pd.Series(target, name="target")

    def split(frame: pd.DataFrame):
        return (
            frame.iloc[:TRAIN_ROWS].reset_index(drop=True),
            target_series.iloc[:TRAIN_ROWS].reset_index(drop=True),
            frame.iloc[TRAIN_ROWS:].reset_index(drop=True),
            target_series.iloc[TRAIN_ROWS:].reset_index(drop=True),
        )

    return {"numeric": split(numeric_frame), "categorical": split(categorical_frame)}


def _prediction_batch(frame: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    repeats = (n_rows + len(frame) - 1) // len(frame)
    return pd.concat([frame] * repeats, ignore_index=True).iloc[:n_rows].copy()


def _run_stage(
    *,
    engine: str,
    input_kind: str,
    data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    thread_count: int,
) -> dict[str, Any]:
    model_cls = MODEL_CLASSES[engine]
    X_train, y_train, X_val, y_val = data
    metric = get_metric("root_mean_squared_error")
    model = model_cls(
        path="",
        name=f"ComparatorWarmup{engine.title()}{input_kind.title()}",
        problem_type="regression",
        eval_metric=metric,
        hyperparameters={},
    )
    fit_started_ns = time.monotonic_ns()
    model.fit(
        X=X_train,
        y=y_train,
        X_val=X_val,
        y_val=y_val,
        time_limit=TIME_LIMIT_SECONDS,
        num_cpus=thread_count,
        num_gpus=0,
        verbosity=0,
    )
    fit_seconds = _elapsed_seconds(fit_started_ns)
    comparator_fit = model._fit_metadata.get(COMPARATOR_METADATA_KEY)
    representation = model._fit_metadata.get(REPRESENTATION_METADATA_KEY)
    if not isinstance(comparator_fit, dict) or not isinstance(representation, dict):
        raise RuntimeError(f"warmup telemetry missing for {engine}/{input_kind}")
    if (
        comparator_fit.get("engine") != engine
        or comparator_fit.get("num_cpus") != thread_count
        or float(comparator_fit.get("num_gpus")) != 0.0
        or representation.get("kind") != "native"
    ):
        raise RuntimeError(f"warmup telemetry mismatch for {engine}/{input_kind}")

    prediction_batches = []
    for name, n_rows in (
        ("small", SMALL_PREDICTION_ROWS),
        ("large", LARGE_PREDICTION_ROWS),
    ):
        batch = _prediction_batch(X_val, n_rows)
        predict_started_ns = time.monotonic_ns()
        prediction = np.asarray(model.predict(batch), dtype=np.float64)
        predict_seconds = _elapsed_seconds(predict_started_ns)
        if prediction.shape != (n_rows,) or not np.all(np.isfinite(prediction)):
            raise RuntimeError(f"warmup prediction failed for {engine}/{input_kind}")
        prediction_batches.append(
            {
                "name": name,
                "rows": n_rows,
                "input_shape": [int(value) for value in batch.shape],
                "prediction_shape": [int(value) for value in prediction.shape],
                "predict_seconds": predict_seconds,
                "prediction_sha256": _prediction_fingerprint(prediction),
            }
        )
    stage = {
        "name": f"{engine}_{input_kind}",
        "engine": engine,
        "input_kind": input_kind,
        "model_class": model_cls.__name__,
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "categorical_columns": list(
            X_train.select_dtypes(include="category").columns
        ),
        "fit_seconds": fit_seconds,
        "thread_count": thread_count,
        "explicit_eval_set": True,
        "comparator_fit": comparator_fit,
        "representation": representation,
        "prediction_batches": prediction_batches,
    }
    json.dumps(stage, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return stage


def warmup_tabarena_comparators(*, thread_count: int) -> dict[str, Any]:
    """Warm the exact native fit/preprocess/predict routes outside measurement."""
    thread_count = _normalize_thread_count(thread_count)
    warmup_env = _warmup_environment_value()
    data = _make_data()
    started_ns = time.monotonic_ns()
    stages = []
    for engine in MODEL_CLASSES:
        for input_kind in ("numeric", "categorical"):
            stages.append(
                _run_stage(
                    engine=engine,
                    input_kind=input_kind,
                    data=data[input_kind],
                    thread_count=thread_count,
                )
            )
    metadata = {
        "schema_version": WARMUP_SCHEMA_VERSION,
        "kind": WARMUP_KIND,
        "clock": "time.monotonic_ns",
        "duration_seconds": _elapsed_seconds(started_ns),
        "thread_count": thread_count,
        "stage_count": len(stages),
        "stage_names": [stage["name"] for stage in stages],
        "counts": {
            "engine": dict(sorted(Counter(stage["engine"] for stage in stages).items())),
            "input_kind": dict(
                sorted(Counter(stage["input_kind"] for stage in stages).items())
            ),
            "prediction_batch": {"large": len(stages), "small": len(stages)},
        },
        "chimeraboost_warmup_environment": warmup_env,
        "stages": stages,
    }
    validate_comparator_warmup(metadata, expected_thread_count=thread_count)
    return metadata


def validate_comparator_warmup(
    value: Any, *, expected_thread_count: int
) -> None:
    if not isinstance(value, dict):
        raise RuntimeError("comparator warmup must be a mapping")
    if set(value) != {
        "schema_version",
        "kind",
        "clock",
        "duration_seconds",
        "thread_count",
        "stage_count",
        "stage_names",
        "counts",
        "chimeraboost_warmup_environment",
        "stages",
    }:
        raise RuntimeError("comparator warmup fields are incomplete")
    if (
        value["schema_version"] != WARMUP_SCHEMA_VERSION
        or value["kind"] != WARMUP_KIND
        or value["clock"] != "time.monotonic_ns"
        or value["thread_count"] != expected_thread_count
        or value["stage_count"] != len(STAGE_NAMES)
        or value["stage_names"] != list(STAGE_NAMES)
        or not isinstance(value["duration_seconds"], (int, float))
        or not math.isfinite(float(value["duration_seconds"]))
        or float(value["duration_seconds"]) <= 0.0
    ):
        raise RuntimeError("comparator warmup identity or counts changed")
    environment = value["chimeraboost_warmup_environment"]
    if environment is not None and (
        not isinstance(environment, str)
        or (environment != "" and environment.strip() != "0")
    ):
        raise RuntimeError("hidden ChimeraBoost import warmup was enabled")
    expected_counts = {
        "engine": {engine: 2 for engine in MODEL_CLASSES},
        "input_kind": {"categorical": 3, "numeric": 3},
        "prediction_batch": {"large": 6, "small": 6},
    }
    if value["counts"] != expected_counts or not isinstance(value["stages"], list):
        raise RuntimeError("comparator warmup coverage counts changed")
    for stage, expected_name in zip(value["stages"], STAGE_NAMES, strict=True):
        if not isinstance(stage, dict) or stage.get("name") != expected_name:
            raise RuntimeError("comparator warmup stage order changed")
        engine, input_kind = expected_name.rsplit("_", 1)
        expected_class = MODEL_CLASSES[engine].__name__
        if (
            stage.get("engine") != engine
            or stage.get("input_kind") != input_kind
            or stage.get("model_class") != expected_class
            or stage.get("train_rows") != TRAIN_ROWS
            or stage.get("validation_rows") != VALIDATION_ROWS
            or stage.get("thread_count") != expected_thread_count
            or stage.get("explicit_eval_set") is not True
            or (engine == "chimeraboost" and stage.get("train_rows", 0) < 1_000)
        ):
            raise RuntimeError("comparator warmup stage policy changed")
        fit_seconds = stage.get("fit_seconds")
        if (
            isinstance(fit_seconds, bool)
            or not isinstance(fit_seconds, (int, float))
            or not math.isfinite(float(fit_seconds))
            or float(fit_seconds) <= 0.0
        ):
            raise RuntimeError("comparator warmup fit duration is invalid")
        comparator_fit = stage.get("comparator_fit")
        representation = stage.get("representation")
        fitted_num_gpus = (
            comparator_fit.get("num_gpus")
            if isinstance(comparator_fit, dict)
            else None
        )
        if (
            not isinstance(comparator_fit, dict)
            or comparator_fit.get("engine") != engine
            or comparator_fit.get("num_cpus") != expected_thread_count
            or type(fitted_num_gpus) not in (int, float)
            or not math.isfinite(float(fitted_num_gpus))
            or float(fitted_num_gpus) != 0.0
            or not isinstance(representation, dict)
            or representation.get("kind") != "native"
        ):
            raise RuntimeError("comparator warmup fitted telemetry changed")
        resolved_params = comparator_fit.get("resolved_params")
        if (
            not isinstance(resolved_params, dict)
            or resolved_params.get("thread_count") != expected_thread_count
            or (
                engine == "catboost"
                and resolved_params.get("task_type", "CPU") != "CPU"
            )
        ):
            raise RuntimeError("comparator warmup core resources changed")
        if (
            engine == "chimeraboost"
            and comparator_fit.get("linear_selection_performed") is not True
        ):
            raise RuntimeError(
                "ChimeraBoost warmup did not exercise automatic lane selection"
            )
        batches = stage.get("prediction_batches")
        if not isinstance(batches, list) or len(batches) != 2:
            raise RuntimeError("comparator warmup prediction coverage changed")
        for batch, (name, rows) in zip(
            batches,
            (("small", SMALL_PREDICTION_ROWS), ("large", LARGE_PREDICTION_ROWS)),
            strict=True,
        ):
            if (
                batch.get("name") != name
                or batch.get("rows") != rows
                or batch.get("prediction_shape") != [rows]
                or not isinstance(batch.get("prediction_sha256"), str)
                or len(batch["prediction_sha256"]) != 64
            ):
                raise RuntimeError("comparator warmup prediction audit changed")
    json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def validate_comparator_warmup_history(
    value: Any,
    *,
    expected_thread_count: int,
    expected_latest_pid: int | None = None,
) -> None:
    if not isinstance(value, list) or not value:
        raise RuntimeError("comparator warmup history must be a nonempty list")
    for index, record in enumerate(value):
        if not isinstance(record, dict) or set(record) != {
            "completed_at_utc",
            "pid",
            "warmup",
        }:
            raise RuntimeError("comparator warmup history record is incomplete")
        if not isinstance(record["completed_at_utc"], str) or not record[
            "completed_at_utc"
        ]:
            raise RuntimeError("comparator warmup completion time is invalid")
        if type(record["pid"]) is not int or record["pid"] <= 0:
            raise RuntimeError("comparator warmup pid is invalid")
        validate_comparator_warmup(
            record["warmup"], expected_thread_count=expected_thread_count
        )
        if (
            expected_latest_pid is not None
            and index == len(value) - 1
            and record["pid"] != expected_latest_pid
        ):
            raise RuntimeError("latest comparator warmup belongs to another process")


__all__ = [
    "STAGE_NAMES",
    "WARMUP_KIND",
    "WARMUP_SCHEMA_VERSION",
    "validate_comparator_warmup",
    "validate_comparator_warmup_history",
    "warmup_tabarena_comparators",
]
