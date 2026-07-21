"""Strict execution and row contract for future paired evidence runners.

This module is deliberately separate from the frozen M6 v3 contract.  It is
an execution foundation for a successor standing runner or M3b; using it does
not make a result ranking-eligible or authorize a shipping/default claim.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


CONTRACT_VERSION = "paired-evidence-v1"
CONTRACT_THREADS = 4
PROBABILITY_SUM_TOLERANCE = 1e-7
PROBABILITY_VALUE_TOLERANCE = 1e-12

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
THREAD_ENV_DEFAULTS = {
    "OMP_DYNAMIC": "FALSE",
    "OMP_THREAD_LIMIT": str(CONTRACT_THREADS),
    "MKL_DYNAMIC": "FALSE",
}
CONTRACT_ENV_KEYS = (
    *THREAD_ENV_KEYS,
    *THREAD_ENV_DEFAULTS,
    "NUMBA_NUM_THREADS",
    "NUMBA_DISABLE_JIT",
    "NUMBA_THREADING_LAYER",
    "NUMBA_CACHE_DIR",
    "DARKOFIT_WARMUP",
    "PYTHONHASHSEED",
    "PYTHONPATH",
)
EVIDENCE_EXTRA_FIELDS = (
    "evidence_contract",
    "candidate_ranking_eligible",
    "shipping_or_default_claim_eligible",
    "implementation_path",
    "case_sha256",
    "dataset_sha256",
    "split_sha256",
    "weight_sha256",
    "prediction_sha256",
    "probability_sha256",
    "expected_class_count",
    "class_count",
    "probability_width",
    "probability_min",
    "probability_max",
    "probability_row_sum_max_error",
    "model_metadata",
    "requested_thread_count",
    "fitted_thread_counts",
    "numba_thread_ceiling",
    "numba_current_thread_count",
    "numba_threading_layer",
    "thread_environment",
)

_REGRESSION_METRIC_FIELDS = (
    "rmse",
    "mae",
    "r2",
    "weighted_rmse",
    "weighted_mae",
    "weighted_r2",
)
_CLASSIFICATION_METRIC_FIELDS = (
    "accuracy",
    "f1_macro",
    "log_loss",
    "brier",
    "weighted_accuracy",
    "weighted_f1_macro",
    "weighted_log_loss",
    "weighted_brier",
)
_HEX_DIGITS = frozenset("0123456789abcdef")
_NUMBA_THREADING_LAYERS = frozenset({"omp", "tbb", "workqueue"})


def contract_payload() -> dict[str, Any]:
    """Return the bindable, explicitly non-ranking draft contract."""
    return {
        "contract_version": CONTRACT_VERSION,
        "contract_frozen": False,
        "candidate_ranking_eligible": False,
        "shipping_or_default_claim_eligible": False,
        "threads": CONTRACT_THREADS,
        "probability_sum_tolerance": PROBABILITY_SUM_TOLERANCE,
        "probability_value_tolerance": PROBABILITY_VALUE_TOLERANCE,
        "required_environment": _expected_environment(CONTRACT_THREADS),
        "required_extra_fields": list(EVIDENCE_EXTRA_FIELDS),
    }


def fixed_worker_environment(
    cache_dir: Path,
    *,
    threads: int = CONTRACT_THREADS,
    base: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Return a worker environment independent of caller thread settings."""
    if threads != CONTRACT_THREADS:
        raise ValueError(
            f"{CONTRACT_VERSION} requires exactly {CONTRACT_THREADS} threads"
        )
    environment = dict(os.environ if base is None else base)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONSTARTUP", None)
    environment.pop("PYTHONINSPECT", None)
    environment.pop("PYTHONOPTIMIZE", None)
    thread_prefixes = (
        "NUMBA_",
        "OMP_",
        "KMP_",
        "MKL_",
        "OPENBLAS_",
        "VECLIB_",
        "NUMEXPR_",
    )
    for key in tuple(environment):
        if key.startswith(thread_prefixes):
            environment.pop(key)
    for key in THREAD_ENV_KEYS:
        environment[key] = str(threads)
    environment.update(
        {
            "DARKOFIT_WARMUP": "0",
            **THREAD_ENV_DEFAULTS,
            "NUMBA_CACHE_DIR": str(cache_dir.resolve()),
            "NUMBA_DISABLE_JIT": "0",
            "NUMBA_NUM_THREADS": str(threads),
            "NUMBA_THREADING_LAYER": "default",
            "PYTHONHASHSEED": "0",
        }
    )
    return environment


def _expected_environment(threads: int) -> dict[str, Optional[str]]:
    expected: dict[str, Optional[str]] = {key: str(threads) for key in THREAD_ENV_KEYS}
    expected.update(
        {
            **THREAD_ENV_DEFAULTS,
            "NUMBA_NUM_THREADS": str(threads),
            "NUMBA_DISABLE_JIT": "0",
            "NUMBA_THREADING_LAYER": "default",
            "DARKOFIT_WARMUP": "0",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": None,
        }
    )
    return expected


def assert_worker_contract(threads: int = CONTRACT_THREADS) -> dict[str, Any]:
    """Fail before model import when the interpreter missed the contract."""
    if threads != CONTRACT_THREADS:
        raise ValueError(
            f"{CONTRACT_VERSION} requires exactly {CONTRACT_THREADS} threads"
        )
    expected = _expected_environment(threads)
    actual = {key: os.environ.get(key) for key in CONTRACT_ENV_KEYS}
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    cache_dir = actual.get("NUMBA_CACHE_DIR")
    if not cache_dir:
        mismatches["NUMBA_CACHE_DIR"] = {
            "expected": "non-empty",
            "actual": cache_dir,
        }
    if mismatches:
        raise RuntimeError(
            "paired evidence worker environment drifted: "
            + json.dumps(mismatches, sort_keys=True)
        )

    import numba

    ceiling = int(numba.config.NUMBA_NUM_THREADS)
    current = int(numba.get_num_threads())
    if ceiling != threads or current != threads:
        raise RuntimeError(
            "paired evidence Numba runtime drifted before model access: "
            f"ceiling={ceiling}, current={current}, expected={threads}"
        )
    return {
        "ceiling": ceiling,
        "current": current,
        "threading_layer": str(numba.threading_layer()),
        "environment": actual,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_create_only(path: Path, payload: bytes) -> None:
    """Publish validated bytes without overwriting or retaining a partial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _scalar_payload(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return {"type": type(value).__name__, "value": value}
    if isinstance(value, float):
        if math.isnan(value):
            return {"type": "float", "value": "nan"}
        if math.isinf(value):
            return {
                "type": "float",
                "value": "inf" if value > 0 else "-inf",
            }
        return {"type": "float", "value": value}
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _update_array_hash(digest: Any, name: str, value: Any) -> None:
    digest.update(name.encode("utf-8"))
    if value is None:
        digest.update(b"<none>")
        return
    array = np.asarray(value)
    descriptor = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(descriptor)
    if array.dtype.hasobject:
        payload = [_scalar_payload(item) for item in array.reshape(-1)]
        digest.update(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
    else:
        digest.update(np.ascontiguousarray(array).tobytes())


def _named_arrays_sha256(data: Mapping[str, Any], names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        _update_array_hash(digest, name, data.get(name))
    return digest.hexdigest()


def case_fingerprints(
    data_path: Path,
    data: Mapping[str, Any],
) -> dict[str, str]:
    """Bind the exact case pack, split arrays, targets, and weights."""
    dataset_names = (
        "X_fit",
        "X_val",
        "X_test",
        "y_fit",
        "y_val",
        "y_test",
    )
    weight_names = ("w_fit", "w_val", "w_test")
    return {
        "case_sha256": _file_sha256(data_path),
        "dataset_sha256": _named_arrays_sha256(data, dataset_names),
        "split_sha256": _named_arrays_sha256(
            data,
            (*dataset_names, *weight_names),
        ),
        "weight_sha256": _named_arrays_sha256(data, weight_names),
    }


def _best_iteration(model: Any) -> Optional[int]:
    value = getattr(model, "best_iteration_", None)
    if value is None:
        value = getattr(model, "best_iteration", None)
    if callable(value):
        value = value()
    return None if value is None else int(value)


def fitted_model_metadata(model: Any) -> dict[str, Any]:
    """Return metadata resolved from fitted DarkoFit cores, not arguments."""
    members = list(getattr(model, "estimators_", ()) or ())
    fitted = members if members else [model]
    cores = [getattr(member, "model_", member) for member in fitted]
    tree_counts = []
    thread_counts = []
    tree_modes = []
    best_iterations = []
    for member, core in zip(fitted, cores):
        trees = getattr(core, "trees_", None)
        thread_count = getattr(core, "n_threads_", None)
        tree_mode = getattr(core, "tree_mode_", None)
        if trees is None or thread_count is None or not tree_mode:
            raise RuntimeError("paired evidence model metadata is unresolved")
        tree_counts.append(len(trees))
        thread_counts.append(int(thread_count))
        tree_modes.append(str(tree_mode))
        best_iterations.append(_best_iteration(member))
    return {
        "member_count": len(fitted),
        "tree_count": int(sum(tree_counts)),
        "tree_counts": tree_counts,
        "tree_modes": sorted(set(tree_modes)),
        "resolved_thread_counts": sorted(set(thread_counts)),
        "best_iterations": best_iterations,
    }


def _array_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_array_hash(digest, "array", value)
    return digest.hexdigest()


def evidence_row_metadata(
    *,
    model: Any,
    implementation_path: str,
    data_path: Path,
    data: Mapping[str, Any],
    task: str,
    prediction: Any,
    probability: Any,
    labels: Any,
    requested_threads: int,
) -> dict[str, Any]:
    """Build strict provenance after fit and validate classification output."""
    runtime = assert_worker_contract(requested_threads)
    metadata = fitted_model_metadata(model)
    fitted_threads = metadata["resolved_thread_counts"]
    if fitted_threads != [requested_threads]:
        raise RuntimeError(
            "paired evidence fitted thread mask drifted: "
            f"{fitted_threads!r} != {[requested_threads]!r}"
        )

    predictions = np.asarray(prediction)
    expected_rows = len(np.asarray(data["y_test"]))
    if predictions.shape != (expected_rows,):
        raise RuntimeError(
            "paired evidence prediction shape is invalid: "
            f"{predictions.shape!r} != {(expected_rows,)!r}"
        )
    if task == "regression":
        try:
            numeric_predictions = np.asarray(predictions, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "paired evidence regression prediction is non-numeric"
            ) from exc
        if not np.isfinite(numeric_predictions).all():
            raise RuntimeError(
                "paired evidence regression prediction is non-finite"
            )

    probability_sha256 = ""
    class_count = 0
    probability_width = 0
    probability_min: Any = ""
    probability_max: Any = ""
    probability_row_sum_max_error: Any = ""
    if task == "regression":
        if probability is not None:
            raise RuntimeError("paired evidence regression emitted probabilities")
    else:
        probabilities = np.asarray(probability, dtype=np.float64)
        classes = np.asarray(labels)
        class_count = int(classes.size)
        if (
            probabilities.ndim != 2
            or probabilities.shape[0] != expected_rows
            or probabilities.shape[1] != class_count
            or class_count < 2
            or not np.isfinite(probabilities).all()
        ):
            raise RuntimeError("paired evidence probability shape is invalid")
        probability_width = int(probabilities.shape[1])
        probability_min = float(np.min(probabilities))
        probability_max = float(np.max(probabilities))
        if (
            probability_min < -PROBABILITY_VALUE_TOLERANCE
            or probability_max > 1.0 + PROBABILITY_VALUE_TOLERANCE
        ):
            raise RuntimeError(
                "paired evidence probabilities are outside [0, 1]: "
                f"minimum={probability_min}, maximum={probability_max}"
            )
        probability_row_sum_max_error = float(
            np.max(np.abs(probabilities.sum(axis=1) - 1.0))
        )
        if probability_row_sum_max_error > PROBABILITY_SUM_TOLERANCE:
            raise RuntimeError(
                "paired evidence probability rows are not normalized: "
                f"max_error={probability_row_sum_max_error}"
            )
        probability_sha256 = _array_sha256(probabilities)

    fingerprints = case_fingerprints(data_path, data)
    environment = runtime["environment"]
    return {
        "evidence_contract": CONTRACT_VERSION,
        "candidate_ranking_eligible": False,
        "shipping_or_default_claim_eligible": False,
        "implementation_path": implementation_path,
        **fingerprints,
        "prediction_sha256": _array_sha256(predictions),
        "probability_sha256": probability_sha256,
        "class_count": class_count,
        "probability_width": probability_width,
        "probability_min": probability_min,
        "probability_max": probability_max,
        "probability_row_sum_max_error": probability_row_sum_max_error,
        "model_metadata": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        "requested_thread_count": requested_threads,
        "fitted_thread_counts": json.dumps(fitted_threads),
        "numba_thread_ceiling": runtime["ceiling"],
        "numba_current_thread_count": runtime["current"],
        "numba_threading_layer": runtime["threading_layer"],
        "thread_environment": json.dumps(
            environment, sort_keys=True, separators=(",", ":")
        ),
    }


def _text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None or not str(value).strip():
        raise RuntimeError(f"paired evidence row has blank {field!r}")
    return str(value)


def _integer(
    row: Mapping[str, Any],
    field: str,
    *,
    minimum: int,
) -> int:
    value = row.get(field)
    if isinstance(value, bool):
        raise RuntimeError(f"paired evidence row has invalid {field!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"paired evidence row has invalid {field!r}: {value!r}"
        ) from exc
    if parsed < minimum or str(value).strip() not in {str(parsed), f"{parsed}.0"}:
        raise RuntimeError(f"paired evidence row has invalid {field!r}: {value!r}")
    return parsed


def _finite(
    row: Mapping[str, Any],
    field: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    value = row.get(field)
    if isinstance(value, bool):
        raise RuntimeError(f"paired evidence row has invalid {field!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"paired evidence row has invalid {field!r}: {value!r}"
        ) from exc
    if (
        not math.isfinite(parsed)
        or (positive and parsed <= 0.0)
        or (nonnegative and parsed < 0.0)
    ):
        raise RuntimeError(f"paired evidence row has invalid {field!r}: {value!r}")
    return parsed


def _sha256(row: Mapping[str, Any], field: str) -> str:
    value = _text(row, field)
    if len(value) != 64 or set(value) - _HEX_DIGITS:
        raise RuntimeError(f"paired evidence row has invalid {field!r}")
    return value


def _json_object(row: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = row.get(field)
    if isinstance(value, dict):
        parsed = value
    else:
        try:
            parsed = json.loads(_text(row, field))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"paired evidence row has invalid {field!r} JSON"
            ) from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"paired evidence row has invalid {field!r}")
    return parsed


def _json_list(row: Mapping[str, Any], field: str) -> list[Any]:
    value = row.get(field)
    if isinstance(value, list):
        parsed = value
    else:
        try:
            parsed = json.loads(_text(row, field))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"paired evidence row has invalid {field!r} JSON"
            ) from exc
    if not isinstance(parsed, list):
        raise RuntimeError(f"paired evidence row has invalid {field!r}")
    return parsed


def _validate_row(
    row: Mapping[str, Any],
    *,
    expected_sources: Mapping[str, Path],
    threads: int,
) -> None:
    if None in row:
        raise RuntimeError("paired evidence CSV has extra unnamed fields")
    if row.get("status") != "ok" or str(row.get("error", "")).strip():
        raise RuntimeError(f"paired evidence worker failed: {row.get('error')!r}")
    if row.get("evidence_contract") != CONTRACT_VERSION:
        raise RuntimeError("paired evidence contract identity drifted")
    for field in (
        "candidate_ranking_eligible",
        "shipping_or_default_claim_eligible",
    ):
        if row.get(field) not in {False, "False"}:
            raise RuntimeError(f"paired evidence cannot authorize claims: {field}")
    variant = _text(row, "variant")
    if variant not in expected_sources:
        raise RuntimeError(f"paired evidence variant is unexpected: {variant}")
    source = expected_sources[variant].expanduser().resolve()
    if Path(_text(row, "revision_path")).resolve() != source:
        raise RuntimeError(f"paired evidence source path drifted: {variant}")
    implementation = Path(_text(row, "implementation_path")).resolve()
    try:
        implementation.relative_to(source / "darkofit")
    except ValueError as exc:
        raise RuntimeError(
            f"paired evidence implementation path drifted: {variant}"
        ) from exc

    task = _text(row, "task")
    if task not in {"regression", "binary", "multiclass"}:
        raise RuntimeError(f"paired evidence task is invalid: {task!r}")
    for field in ("dataset", "size", "selected_tree_mode"):
        _text(row, field)
    weight_mode = _text(row, "weight_mode")
    if weight_mode not in {"none", "uniform", "stress"}:
        raise RuntimeError("paired evidence weight mode is invalid")
    if row.get("use_defaults") not in {True, "True"}:
        raise RuntimeError("paired evidence row did not use public defaults")
    counts = {}
    for field, minimum in (
        ("seed", 0),
        ("n_train", 1),
        ("n_val", 0),
        ("n_test", 1),
        ("n_features", 1),
    ):
        counts[field] = _integer(row, field, minimum=minimum)
    if counts["n_val"] != 0:
        raise RuntimeError(
            "paired evidence public-default row has an external validation set"
        )
    for field in ("fit_seconds", "predict_seconds"):
        _finite(row, field, positive=True)
    _integer(row, "worker_peak_rss_bytes", minimum=1)
    for field in (
        "case_sha256",
        "dataset_sha256",
        "split_sha256",
        "weight_sha256",
        "prediction_sha256",
    ):
        _sha256(row, field)

    requested = _integer(row, "requested_thread_count", minimum=1)
    ceiling = _integer(row, "numba_thread_ceiling", minimum=1)
    current = _integer(row, "numba_current_thread_count", minimum=1)
    fitted = _json_list(row, "fitted_thread_counts")
    if requested != threads or ceiling != threads or current != threads:
        raise RuntimeError("paired evidence resolved thread budget drifted")
    if fitted != [threads] or any(
        isinstance(value, bool) or not isinstance(value, int) for value in fitted
    ):
        raise RuntimeError("paired evidence fitted thread mask drifted")
    threading_layer = _text(row, "numba_threading_layer")
    if threading_layer not in _NUMBA_THREADING_LAYERS:
        raise RuntimeError("paired evidence Numba threading layer is invalid")

    environment = _json_object(row, "thread_environment")
    if set(environment) != set(CONTRACT_ENV_KEYS):
        raise RuntimeError("paired evidence thread environment schema drifted")
    expected_environment = _expected_environment(threads)
    for key, value in expected_environment.items():
        if environment.get(key) != value:
            raise RuntimeError(f"paired evidence thread environment drifted at {key}")
    cache_dir = environment.get("NUMBA_CACHE_DIR")
    if (
        not isinstance(cache_dir, str)
        or not cache_dir
        or not Path(cache_dir).is_absolute()
    ):
        raise RuntimeError("paired evidence Numba cache path is missing")

    metadata = _json_object(row, "model_metadata")
    member_count = metadata.get("member_count")
    tree_count = metadata.get("tree_count")
    tree_counts = metadata.get("tree_counts")
    tree_modes = metadata.get("tree_modes")
    best_iterations = metadata.get("best_iterations")
    if (
        isinstance(member_count, bool)
        or not isinstance(member_count, int)
        or member_count < 1
        or isinstance(tree_count, bool)
        or not isinstance(tree_count, int)
        or tree_count <= 0
        or not isinstance(tree_counts, list)
        or len(tree_counts) != member_count
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in tree_counts
        )
        or sum(tree_counts) != tree_count
        or metadata.get("resolved_thread_counts") != [threads]
        or not isinstance(tree_modes, list)
        or not tree_modes
        or any(not isinstance(value, str) or not value for value in tree_modes)
        or row.get("selected_tree_mode") not in tree_modes
        or not isinstance(best_iterations, list)
        or len(best_iterations) != member_count
        or any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
            for value in best_iterations
        )
    ):
        raise RuntimeError("paired evidence fitted model metadata is invalid")

    primary_metric = _text(row, "primary_metric")
    primary_value = _finite(row, "primary_value", nonnegative=True)
    metric_fields = (
        _REGRESSION_METRIC_FIELDS
        if task == "regression"
        else _CLASSIFICATION_METRIC_FIELDS
    )
    metrics = {field: _finite(row, field) for field in metric_fields}
    weighted = weight_mode != "none"
    expected_primary = (
        "weighted_rmse"
        if task == "regression" and weighted
        else (
            "rmse"
            if task == "regression"
            else "weighted_log_loss" if weighted else "log_loss"
        )
    )
    if task == "regression":
        nonnegative_metrics = (
            "rmse",
            "mae",
            "weighted_rmse",
            "weighted_mae",
        )
        bounded_metric_maxima = {
            "r2": 1.0,
            "weighted_r2": 1.0,
        }
    else:
        nonnegative_metrics = (
            "accuracy",
            "f1_macro",
            "log_loss",
            "brier",
            "weighted_accuracy",
            "weighted_f1_macro",
            "weighted_log_loss",
            "weighted_brier",
        )
        bounded_metric_maxima = {
            "accuracy": 1.0,
            "f1_macro": 1.0,
            "brier": 2.0,
            "weighted_accuracy": 1.0,
            "weighted_f1_macro": 1.0,
            "weighted_brier": 2.0,
        }
    if (
        primary_metric != expected_primary
        or primary_value != metrics[expected_primary]
        or any(metrics[field] < 0.0 for field in nonnegative_metrics)
        or any(
            metrics[field] > maximum
            for field, maximum in bounded_metric_maxima.items()
        )
    ):
        raise RuntimeError("paired evidence primary metric is invalid")

    expected_class_count = _integer(row, "expected_class_count", minimum=0)
    class_count = _integer(row, "class_count", minimum=0)
    probability_width = _integer(row, "probability_width", minimum=0)
    probability_hash = str(row.get("probability_sha256", "")).strip()
    probability_min = str(row.get("probability_min", "")).strip()
    probability_max = str(row.get("probability_max", "")).strip()
    probability_error = str(row.get("probability_row_sum_max_error", "")).strip()
    if task == "regression":
        if (
            expected_class_count != 0
            or class_count != 0
            or probability_width != 0
        ):
            raise RuntimeError("paired evidence regression class metadata drifted")
        if (
            probability_hash
            or probability_min
            or probability_max
            or probability_error
        ):
            raise RuntimeError("paired evidence regression probability drifted")
    else:
        _sha256(row, "probability_sha256")
        minimum = _finite(row, "probability_min")
        maximum = _finite(row, "probability_max")
        error = _finite(
            row,
            "probability_row_sum_max_error",
            nonnegative=True,
        )
        expected_count_is_valid = (
            expected_class_count == 2
            if task == "binary"
            else expected_class_count >= 3
        )
        if (
            not expected_count_is_valid
            or class_count != expected_class_count
            or probability_width != class_count
            or minimum > maximum
            or minimum < -PROBABILITY_VALUE_TOLERANCE
            or maximum > 1.0 + PROBABILITY_VALUE_TOLERANCE
            or error > PROBABILITY_SUM_TOLERANCE
        ):
            raise RuntimeError(
                "paired evidence classification probability metadata drifted"
            )


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_sources: Mapping[str, Path],
    threads: int = CONTRACT_THREADS,
    expected_pair_keys: Optional[Sequence[tuple[str, str, str, str]]] = None,
) -> dict[str, Any]:
    """Validate strict rows and exact control/candidate pairing provenance."""
    if threads != CONTRACT_THREADS:
        raise ValueError(
            f"{CONTRACT_VERSION} requires exactly {CONTRACT_THREADS} threads"
        )
    if len(expected_sources) != 2:
        raise ValueError("paired evidence requires exactly two source arms")
    for row in rows:
        _validate_row(row, expected_sources=expected_sources, threads=threads)

    pair_fields = (
        "task",
        "n_train",
        "n_val",
        "n_test",
        "n_features",
        "expected_class_count",
        "case_sha256",
        "dataset_sha256",
        "split_sha256",
        "weight_sha256",
        "numba_threading_layer",
    )
    pairs: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        identity = (
            str(row["dataset"]),
            str(row["size"]),
            str(row["seed"]),
            str(row["weight_mode"]),
        )
        pairs.setdefault(identity, []).append(row)
    if expected_pair_keys is not None:
        expected = {
            tuple(str(value) for value in identity) for identity in expected_pair_keys
        }
        actual = set(pairs)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            raise RuntimeError(
                "paired evidence grid mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
    expected_variants = set(expected_sources)
    for identity, pair in pairs.items():
        variants = [str(row["variant"]) for row in pair]
        if len(pair) != 2 or set(variants) != expected_variants:
            raise RuntimeError(f"paired evidence pair is incomplete: {identity}")
        for field in pair_fields:
            if len({str(row.get(field)) for row in pair}) != 1:
                raise RuntimeError(
                    f"paired evidence pair differs on {field}: {identity}"
                )
    if not pairs:
        raise RuntimeError("paired evidence contains no comparison pairs")
    return {
        "row_count": len(rows),
        "paired_cells": len(pairs),
        "resolved_threads": threads,
        "contract_version": CONTRACT_VERSION,
    }


def load_and_validate_csv(
    path: Path,
    *,
    expected_fields: Sequence[str],
    expected_sources: Mapping[str, Path],
    threads: int = CONTRACT_THREADS,
    expected_pair_keys: Optional[Sequence[tuple[str, str, str, str]]] = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Load an exact-schema CSV and reject extra or blank required fields."""
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"paired evidence CSV is not a regular file: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(expected_fields):
            raise RuntimeError("paired evidence CSV schema drifted")
        rows = list(reader)
    if any(None in row for row in rows):
        raise RuntimeError("paired evidence CSV has extra unnamed fields")
    validation = validate_rows(
        rows,
        expected_sources=expected_sources,
        threads=threads,
        expected_pair_keys=expected_pair_keys,
    )
    return rows, validation
