#!/usr/bin/env python3
"""Analyze the frozen native-ordinal C2 development or confirmation tier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from benchmarks import run_native_ordinal_c2 as runner  # noqa: E402


DEVELOPMENT_MAX_GEOMEAN = 0.980
DEVELOPMENT_MIN_WINS = 3
DEVELOPMENT_MAX_TASK = 1.020
DEVELOPMENT_MAX_SPLIT = 1.050
CONFIRMATION_MAX_GEOMEAN = 0.995
CONFIRMATION_MIN_WINS = 3
CONFIRMATION_MAX_TASK = 1.020
CONFIRMATION_MAX_SPLIT = 1.050
MAX_VALIDATION_TASK = 1.020
MAX_MEDIAN_FIT = 1.150
MAX_MEDIAN_PREDICT = 1.100
MAX_MEDIAN_RSS = 1.100
MAX_TIMING_RELATIVE_IQR = 0.150
BOOTSTRAP_SEED = 20260718
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_LEVEL = 0.95
EXPECTED_WARNING_PREFIXES = (
    "DarkoFit automatic learning rate clipped to max ",
)

DEFAULT_DEVELOPMENT_RAW = runner.DEFAULT_DEVELOPMENT_OUTPUT
DEFAULT_CONFIRMATION_RAW = runner.DEFAULT_CONFIRMATION_OUTPUT
DEFAULT_DEVELOPMENT_SUMMARY = (
    ROOT / "benchmarks" / "native_ordinal_c2_development_result.json"
)
DEFAULT_CONFIRMATION_SUMMARY = (
    ROOT / "benchmarks" / "native_ordinal_c2_confirmation_result.json"
)
DEFAULT_DEVELOPMENT_REPORT = (
    ROOT / "benchmarks" / "native_ordinal_c2_development_result.md"
)
DEFAULT_CONFIRMATION_REPORT = (
    ROOT / "benchmarks" / "native_ordinal_c2_confirmation_result.md"
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _geomean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 1
        or array.size == 0
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
    ):
        raise ValueError("geometric mean requires nonnegative finite values")
    if np.any(array == 0.0):
        return 0.0
    return float(np.exp(np.mean(np.log(array))))


def _quality_ratio(candidate: float, control: float) -> float:
    candidate = float(candidate)
    control = float(control)
    if (
        not math.isfinite(candidate)
        or not math.isfinite(control)
        or candidate < 0.0
        or control < 0.0
    ):
        raise ValueError("quality metrics must be nonnegative and finite")
    if control == 0.0:
        if candidate == 0.0:
            return 1.0
        raise ValueError("candidate/control ratio is undefined at zero control")
    return candidate / control


def _ratio_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 1
        or array.size == 0
        or not np.all(np.isfinite(array))
        or np.any(array <= 0.0)
    ):
        raise ValueError("ratio summary requires positive finite values")
    q25, median, q75 = np.quantile(array, [0.25, 0.50, 0.75])
    iqr = float(q75 - q25)
    return {
        "count": int(array.size),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": iqr,
        "iqr_over_median": float(iqr / median),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def confirmation_bootstrap_upper(
    task_ratios: list[float],
) -> dict[str, Any]:
    ratios = np.asarray(task_ratios, dtype=np.float64)
    if (
        ratios.shape != (5,)
        or not np.all(np.isfinite(ratios))
        or np.any(ratios < 0.0)
    ):
        raise ValueError("confirmation bootstrap requires five task ratios")
    logs = np.log(np.maximum(ratios, np.finfo(np.float64).tiny))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, 5, size=(BOOTSTRAP_DRAWS, 5))
    estimates = np.mean(logs[indices], axis=1)
    upper_log = float(
        np.quantile(estimates, BOOTSTRAP_LEVEL, method="higher")
    )
    return {
        "seed": BOOTSTRAP_SEED,
        "draws": BOOTSTRAP_DRAWS,
        "level": BOOTSTRAP_LEVEL,
        "method": "task_cluster_bootstrap_higher",
        "upper_ratio": float(math.exp(upper_log)),
    }


def _task_rows(registry: dict[str, Any], tier: str) -> dict[int, dict[str, Any]]:
    return {
        int(row["task_id"]): row
        for row in registry[f"{tier}_tasks"]
    }


def _coordinate_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["task_id"]), int(row["fold"])


def _expected_warning(warning: dict[str, Any]) -> bool:
    return (
        warning.get("category") == "RuntimeWarning"
        and isinstance(warning.get("message"), str)
        and warning["message"].startswith(EXPECTED_WARNING_PREFIXES)
    )


def _validate_worker_static(
    result: dict[str, Any],
    raw: dict[str, Any],
    registry: dict[str, Any],
    task: dict[str, Any],
    failures: list[str],
) -> None:
    label = (
        f"task {result.get('task_id')} fold {result.get('fold')} "
        f"arm {result.get('arm')}"
    )
    if not result.get("ok"):
        failures.append(
            f"{label}: worker failed: {result.get('error_type')}: "
            f"{result.get('error')}"
        )
        return
    if result.get("worker_returncode") != 0:
        failures.append(f"{label}: worker return code was nonzero")
    if result.get("worker_stdout") or result.get("worker_stderr"):
        failures.append(f"{label}: worker emitted uncaptured output")
    if result.get("tier") != raw["tier"]:
        failures.append(f"{label}: tier binding changed")
    if result.get("registry_sha256") != registry["registry_sha256"]:
        failures.append(f"{label}: registry binding changed")
    if result.get("source") != raw["source"]:
        failures.append(f"{label}: source binding changed")
    if result.get("runtime") != raw["runtime"]:
        failures.append(f"{label}: runtime binding changed")
    if int(result.get("dataset_id", -1)) != int(task["dataset_id"]):
        failures.append(f"{label}: dataset ID changed")
    if result.get("dataset_name") != task["dataset_name"]:
        failures.append(f"{label}: dataset name changed")
    if result.get("lineage_cluster") != task["lineage_cluster"]:
        failures.append(f"{label}: lineage changed")
    if result.get("declared_ordinal_features") != task["ordinal_features"]:
        failures.append(f"{label}: ordinal declaration changed")
    if result.get("categorical_indices") != task["feature_record"][
        "categorical_indices"
    ]:
        failures.append(f"{label}: categorical input policy changed")
    if result.get("peak_rss_bytes", 0) <= 0:
        failures.append(f"{label}: peak RSS is invalid")
    if (
        result.get("warmup_seconds", 0.0) <= 0.0
        or result.get("warmup_returned_seconds", 0.0) <= 0.0
        or not math.isclose(
            result["warmup_seconds"],
            result["warmup_returned_seconds"],
            rel_tol=0.05,
            abs_tol=0.05,
        )
    ):
        failures.append(f"{label}: explicit warmup timing is invalid")
    if result.get("fit_seconds", 0.0) <= 0.0:
        failures.append(f"{label}: fit timing is invalid")
    timing = result.get("public_predict_timing", {})
    if (
        timing.get("calls") != runner.PREDICTION_CALLS
        or timing.get("seconds_per_call", 0.0) <= 0.0
        or timing.get("last_prediction_sha256")
        != result.get("test", {}).get("prediction_sha256")
    ):
        failures.append(f"{label}: public prediction timing is invalid")
    for view in ("validation", "test"):
        record = result.get(view, {})
        if (
            record.get("rows", 0) <= 0
            or record.get("rmse", -1.0) < 0.0
            or not isinstance(record.get("prediction_sha256"), str)
        ):
            failures.append(f"{label}: {view} prediction record is invalid")
    expected_outer = runner._expected_outer_split(task, int(result["fold"]))
    outer = result.get("outer_split", {})
    if any(
        outer.get(key) != expected_outer[key]
        for key in (
            "train_size",
            "test_size",
            "train_index_sha256",
            "test_index_sha256",
        )
    ):
        failures.append(f"{label}: official split binding changed")
    inner = result.get("inner_split", {})
    expected_validation_rows = max(
        1,
        int(
            math.ceil(
                runner.VALIDATION_FRACTION * expected_outer["train_size"]
            )
        ),
    )
    if (
        inner.get("policy") != "seeded_permutation_tail_20_percent"
        or inner.get("seed")
        != runner.SPLIT_SEED + int(result["task_id"]) + int(result["fold"])
        or inner.get("validation_fraction") != runner.VALIDATION_FRACTION
        or inner.get("validation_rows") != expected_validation_rows
        or inner.get("fit_rows", 0) + inner.get("validation_rows", 0)
        != expected_outer["train_size"]
        or not isinstance(inner.get("fit_index_sha256"), str)
        or not isinstance(inner.get("validation_index_sha256"), str)
        or result.get("validation", {}).get("rows")
        != expected_validation_rows
        or result.get("test", {}).get("rows") != expected_outer["test_size"]
    ):
        failures.append(f"{label}: inner split binding changed")
    cache = result.get("cache", {})
    if (
        cache.get("before_import", {}).get("file_count") != 0
        or cache.get("after_import", {}).get("file_count") != 0
        or cache.get("after_warmup", {}).get("compiled_file_count", 0) <= 0
        or cache.get("after_workload", {}).get("compiled_file_count", 0) <= 0
    ):
        failures.append(f"{label}: cache warmup contract changed")
    environment = result.get("thread_environment", {})
    if (
        environment.get("DARKOFIT_WARMUP") != "0"
        or environment.get("NUMBA_NUM_THREADS")
        != str(runner.THREADS_PER_WORKER)
        or any(
            environment.get(name) != str(runner.THREADS_PER_WORKER)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        )
    ):
        failures.append(f"{label}: worker thread environment changed")
    unexpected = [
        warning
        for warning in result.get("warnings", [])
        if not _expected_warning(warning)
    ]
    if unexpected:
        failures.append(f"{label}: unexpected warnings: {unexpected!r}")
    fit_metadata = result.get("model", {}).get("fit_metadata", {})
    final_fit = fit_metadata.get("final_fit", {})
    if (
        not isinstance(fit_metadata.get("best_iteration"), int)
        or fit_metadata["best_iteration"] <= 0
        or fit_metadata.get("fitted_tree_count")
        != fit_metadata["best_iteration"]
        or not isinstance(fit_metadata.get("resolved_learning_rate"), float)
        or not math.isfinite(fit_metadata["resolved_learning_rate"])
        or fit_metadata["resolved_learning_rate"] <= 0.0
        or fit_metadata.get("resolved_thread_count")
        != runner.THREADS_PER_WORKER
        or fit_metadata.get("requested_tree_mode") != "catboost"
        or fit_metadata.get("selected_tree_mode") != "catboost"
        or fit_metadata.get("selected_lane") != "boosting"
        or fit_metadata.get("refit") is not False
        or final_fit.get("stop_reason") != "iteration_limit"
        or final_fit.get("rounds_retained")
        != fit_metadata.get("fitted_tree_count")
    ):
        failures.append(f"{label}: fitted metadata changed")
    if result.get("behavior_fingerprint_sha256") != runner._json_sha256(
        runner._behavior_payload(result)
    ):
        failures.append(f"{label}: behavior fingerprint changed")


def _expected_ordinal_records(task: dict[str, Any]) -> list[dict[str, Any]]:
    records = [
        {
            "index": int(record["index"]),
            "name": record["feature"],
            "categories": record["categories"],
            "source": "explicit",
        }
        for record in task["feature_record"]["ordinal_features"]
    ]
    return sorted(records, key=lambda record: record["index"])


def _validate_telemetry(
    result: dict[str, Any],
    task: dict[str, Any],
    failures: list[str],
) -> None:
    label = (
        f"task {result['task_id']} fold {result['fold']} arm {result['arm']}"
    )
    state = result["model"]["ordinal_state"]
    categorical = task["feature_record"]["categorical_indices"]
    ordinal_records = _expected_ordinal_records(task)
    ordinal_indices = [int(row["index"]) for row in ordinal_records]
    preprocessor = result["model"]["preprocessor"]
    if result["arm"] == runner.CONTROL:
        if (
            state.get("mode") != "off"
            or state.get("records") != []
            or state.get("indices") != []
            or state.get("metadata") is not None
        ):
            failures.append(f"{label}: control ordinal state is not off")
        if preprocessor.get("cat_features") != categorical:
            failures.append(f"{label}: control categorical preprocessing changed")
        return

    metadata = state.get("metadata")
    expected_metadata = {
        "mode": "explicit",
        "active": bool(ordinal_records),
        "feature_count": len(ordinal_records),
        "feature_indices": ordinal_indices,
        "feature_names": [row["name"] for row in ordinal_records],
        "sources": ["explicit"] * len(ordinal_records),
        "nominal_categorical_count": len(categorical) - len(ordinal_records),
        "added_columns": 0,
        "target_stat_blocks_added": 0,
        "target_used": False,
        "unknown_policy": "fail_closed",
        "missing_policy": "numeric_missing_bin",
    }
    if (
        state.get("mode") != "explicit"
        or state.get("records") != ordinal_records
        or state.get("indices") != ordinal_indices
        or metadata != expected_metadata
    ):
        failures.append(f"{label}: candidate ordinal telemetry changed")
    expected_nominal = [
        index for index in categorical if index not in set(ordinal_indices)
    ]
    if preprocessor.get("cat_features") != expected_nominal:
        failures.append(f"{label}: candidate nominal feature set changed")
    if any(
        index not in preprocessor.get("num_features", [])
        for index in ordinal_indices
    ):
        failures.append(f"{label}: ordinal feature did not enter numeric binner")


def _validate_pair(
    control: dict[str, Any],
    candidate: dict[str, Any],
    task: dict[str, Any],
    failures: list[str],
) -> None:
    label = f"task {control['task_id']} fold {control['fold']}"
    for field in (
        "tier",
        "task_id",
        "dataset_id",
        "dataset_name",
        "lineage_cluster",
        "role",
        "fold",
        "source",
        "runtime",
        "registry_sha256",
        "authorization",
        "categorical_indices",
        "declared_ordinal_features",
        "outer_split",
        "inner_split",
    ):
        if control.get(field) != candidate.get(field):
            failures.append(f"{label}: paired {field} differs")
    if not task["ordinal_features"]:
        for view in ("validation", "test"):
            if (
                control[view]["prediction_sha256"]
                != candidate[view]["prediction_sha256"]
                or control[view]["rmse"] != candidate[view]["rmse"]
            ):
                failures.append(
                    f"{label}: no-engagement {view} predictions differ"
                )
        if (
            control["model"]["preprocessor"]
            != candidate["model"]["preprocessor"]
        ):
            failures.append(
                f"{label}: no-engagement preprocessing state differs"
            )
        if (
            control["model"]["archive"]["normalized_logical_sha256"]
            != candidate["model"]["archive"]["normalized_logical_sha256"]
        ):
            failures.append(
                f"{label}: no-engagement logical model state differs"
            )


def validate_raw(
    raw: dict[str, Any],
    registry: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    failures: list[str] = []
    tier = raw.get("tier")
    expected_coordinates = (
        24 if tier == runner.DEVELOPMENT else 15
    )
    expected_workers = expected_coordinates * 2
    if tier not in runner.TIERS:
        return [], ["raw tier is invalid"], {}
    if raw.get("name") != f"darkofit_native_ordinal_c2_{tier}_raw_v1":
        failures.append("raw artifact name changed")
    if raw.get("schema_version") != 1:
        failures.append("raw schema version changed")
    if raw.get("lockbox_touched") is not False:
        failures.append("raw artifact reports lockbox access")
    if raw.get("development_outcomes_inspected") is not (
        tier == runner.DEVELOPMENT
    ):
        failures.append("development outcome flag changed")
    if raw.get("confirmation_outcomes_inspected") is not (
        tier == runner.CONFIRMATION
    ):
        failures.append("confirmation outcome flag changed")
    source = raw.get("source", {})
    if (
        source.get("branch") != "main"
        or source.get("head") != source.get("origin_main")
        or source.get("status") != ""
        or source.get("package_tree") != runner.EXPECTED_PACKAGE_TREE
        or source.get("registry_file_sha256")
        != runner.EXPECTED_REGISTRY_FILE_SHA256
        or source.get("protocol_sha256") != runner.EXPECTED_PROTOCOL_SHA256
        or source.get("runner_normalized_sha256")
        != runner.EXPECTED_NORMALIZED_RUNNER_SHA256
        or source.get("analyzer_sha256")
        != _sha256_file(Path(__file__).resolve())
        or source.get("chimeraboost_head") != runner.EXPECTED_CHIMERA_HEAD
        or source.get("chimeraboost_status") != ""
    ):
        failures.append("raw source attestation changed")
    runtime = raw.get("runtime", {})
    if any(
        runtime.get(key) != runner.EXPECTED_RUNTIME[key]
        for key in ("python", "platform", "machine", "logical_cpu_count")
    ) or runtime.get("dependencies") != runner.EXPECTED_RUNTIME[
        "dependencies"
    ]:
        failures.append("raw runtime attestation changed")
    if tier == runner.DEVELOPMENT:
        if raw.get("authorization") is not None:
            failures.append("development raw unexpectedly has authorization")
    else:
        authorization = raw.get("authorization")
        if (
            not isinstance(authorization, dict)
            or authorization.get("decision")
            != "authorize_native_ordinal_c2_confirmation_once"
            or not isinstance(
                authorization.get("development_raw_sha256"), str
            )
        ):
            failures.append("confirmation authorization binding changed")
    execution = raw.get("execution", {})
    if execution != {
        "threads_per_worker": runner.THREADS_PER_WORKER,
        "concurrent_workers": runner.CONCURRENT_WORKERS,
        "prediction_calls": runner.PREDICTION_CALLS,
        "coordinate_count": expected_coordinates,
        "worker_count": expected_workers,
        "reciprocal_order": "alternating_by_coordinate_index",
        "partial_resume_supported": False,
    }:
        failures.append("raw execution contract changed")
    if raw.get("registry") != {
        "path": str(runner.REGISTRY_PATH.relative_to(ROOT)),
        "file_sha256": runner.EXPECTED_REGISTRY_FILE_SHA256,
        "content_sha256": runner.EXPECTED_REGISTRY_CONTENT_SHA256,
    }:
        failures.append("raw registry binding changed")
    if raw.get("protocol") != {
        "path": str(runner.PROTOCOL_PATH.relative_to(ROOT)),
        "sha256": runner.EXPECTED_PROTOCOL_SHA256,
    }:
        failures.append("raw protocol binding changed")
    if raw.get("runner") != {
        "path": str(Path(runner.__file__).resolve().relative_to(ROOT)),
        "normalized_sha256": runner.EXPECTED_NORMALIZED_RUNNER_SHA256,
    }:
        failures.append("raw runner binding changed")
    if raw.get("analyzer") != {
        "path": str(Path(__file__).resolve().relative_to(ROOT)),
        "sha256": _sha256_file(Path(__file__).resolve()),
    }:
        failures.append("raw analyzer binding changed")

    results = raw.get("results", [])
    if len(results) != expected_workers:
        failures.append(
            f"worker count is {len(results)}, expected {expected_workers}"
        )
    cache_paths = [
        result.get("thread_environment", {}).get("NUMBA_CACHE_DIR")
        for result in results
        if result.get("ok")
    ]
    expected_cache_root = str(runner.CACHE_ROOT.absolute()) + os.sep
    if (
        len(cache_paths) != expected_workers
        or len(set(cache_paths)) != expected_workers
        or any(
            not isinstance(path, str)
            or not path.startswith(expected_cache_root)
            for path in cache_paths
        )
    ):
        failures.append("worker cache paths are not unique and frozen")
    tasks = _task_rows(registry, tier)
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    expected_warnings = 0
    for result in results:
        task = tasks.get(int(result.get("task_id", -1)))
        if task is None:
            failures.append(
                f"unexpected task {result.get('task_id')} in raw results"
            )
            continue
        _validate_worker_static(result, raw, registry, task, failures)
        if not result.get("ok"):
            continue
        _validate_telemetry(result, task, failures)
        expected_warnings += sum(
            _expected_warning(warning)
            for warning in result.get("warnings", [])
        )
        key = _coordinate_key(result)
        arm = result.get("arm")
        if arm in grouped.setdefault(key, {}):
            failures.append(f"duplicate worker result for {key} arm {arm}")
        grouped[key][arm] = result

    expected_keys = {
        (int(row["task_id"]), int(row["fold"]))
        for row in registry["coordinates"][tier]
    }
    if set(grouped) != expected_keys:
        failures.append("raw coordinate membership changed")
    pairs = []
    for coordinate_index, coordinate in enumerate(
        registry["coordinates"][tier]
    ):
        key = (int(coordinate["task_id"]), int(coordinate["fold"]))
        arms = grouped.get(key, {})
        if set(arms) != set(runner.ARMS):
            failures.append(f"coordinate {key} does not have both arms")
            continue
        control = arms[runner.CONTROL]
        candidate = arms[runner.CANDIDATE]
        expected_order = runner.reciprocal_order(coordinate_index)
        observed_positions = {
            int(control["position"]): runner.CONTROL,
            int(candidate["position"]): runner.CANDIDATE,
        }
        if observed_positions != {
            0: expected_order[0],
            1: expected_order[1],
        }:
            failures.append(f"coordinate {key} reciprocal order changed")
        if (
            int(control["coordinate_index"]) != coordinate_index
            or int(candidate["coordinate_index"]) != coordinate_index
        ):
            failures.append(f"coordinate {key} index changed")
        _validate_pair(control, candidate, tasks[key[0]], failures)
        pairs.append({
            "task": tasks[key[0]],
            "control": control,
            "candidate": candidate,
        })
    return pairs, failures, {"expected_warning_count": expected_warnings}


def _aggregate(
    tier: str,
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    engaged = [
        pair for pair in pairs if bool(pair["task"]["ordinal_features"])
    ]
    by_task: dict[int, list[dict[str, Any]]] = {}
    for pair in engaged:
        by_task.setdefault(int(pair["task"]["task_id"]), []).append(pair)
    task_rows = []
    for task_id, task_pairs in by_task.items():
        test_ratios = [
            _quality_ratio(
                pair["candidate"]["test"]["rmse"],
                pair["control"]["test"]["rmse"],
            )
            for pair in task_pairs
        ]
        validation_ratios = [
            _quality_ratio(
                pair["candidate"]["validation"]["rmse"],
                pair["control"]["validation"]["rmse"],
            )
            for pair in task_pairs
        ]
        task_rows.append({
            "task_id": task_id,
            "dataset_name": task_pairs[0]["task"]["dataset_name"],
            "lineage_cluster": task_pairs[0]["task"]["lineage_cluster"],
            "split_count": len(task_pairs),
            "test_rmse_ratio": _geomean(test_ratios),
            "validation_rmse_ratio": _geomean(validation_ratios),
            "test_split_ratios": test_ratios,
            "validation_split_ratios": validation_ratios,
        })
    task_rows.sort(key=lambda row: row["task_id"])
    task_ratios = [row["test_rmse_ratio"] for row in task_rows]
    split_ratios = [
        ratio for row in task_rows for ratio in row["test_split_ratios"]
    ]
    fit_ratios = [
        pair["candidate"]["fit_seconds"] / pair["control"]["fit_seconds"]
        for pair in pairs
    ]
    predict_ratios = [
        pair["candidate"]["public_predict_timing"]["seconds_per_call"]
        / pair["control"]["public_predict_timing"]["seconds_per_call"]
        for pair in pairs
    ]
    rss_ratios = [
        pair["candidate"]["peak_rss_bytes"]
        / pair["control"]["peak_rss_bytes"]
        for pair in pairs
    ]
    result = {
        "engaged_task_count": len(task_rows),
        "engaged_split_count": len(split_ratios),
        "task_rows": task_rows,
        "equal_task_test_rmse_ratio": _geomean(task_ratios),
        "task_wins": sum(ratio < 1.0 for ratio in task_ratios),
        "worst_task_ratio": max(task_ratios),
        "worst_split_ratio": max(split_ratios),
        "worst_validation_task_ratio": max(
            row["validation_rmse_ratio"] for row in task_rows
        ),
        "fit_ratio": _ratio_summary(fit_ratios),
        "predict_ratio": _ratio_summary(predict_ratios),
        "peak_rss_ratio": _ratio_summary(rss_ratios),
    }
    if tier == runner.CONFIRMATION:
        result["task_bootstrap"] = confirmation_bootstrap_upper(task_ratios)
    return result


def _gates(
    tier: str,
    aggregate: dict[str, Any],
    integrity_failures: list[str],
) -> dict[str, dict[str, Any]]:
    max_geomean = (
        DEVELOPMENT_MAX_GEOMEAN
        if tier == runner.DEVELOPMENT
        else CONFIRMATION_MAX_GEOMEAN
    )
    min_wins = (
        DEVELOPMENT_MIN_WINS
        if tier == runner.DEVELOPMENT
        else CONFIRMATION_MIN_WINS
    )
    max_task = (
        DEVELOPMENT_MAX_TASK
        if tier == runner.DEVELOPMENT
        else CONFIRMATION_MAX_TASK
    )
    max_split = (
        DEVELOPMENT_MAX_SPLIT
        if tier == runner.DEVELOPMENT
        else CONFIRMATION_MAX_SPLIT
    )
    gates = {
        "integrity": {
            "passes": not integrity_failures,
            "failure_count": len(integrity_failures),
        },
        "equal_task_test_rmse": {
            "value": aggregate["equal_task_test_rmse_ratio"],
            "threshold_at_most": max_geomean,
            "passes": aggregate["equal_task_test_rmse_ratio"] <= max_geomean,
        },
        "task_wins": {
            "value": aggregate["task_wins"],
            "threshold_at_least": min_wins,
            "passes": aggregate["task_wins"] >= min_wins,
        },
        "worst_task": {
            "value": aggregate["worst_task_ratio"],
            "threshold_at_most": max_task,
            "passes": aggregate["worst_task_ratio"] <= max_task,
        },
        "worst_split": {
            "value": aggregate["worst_split_ratio"],
            "threshold_at_most": max_split,
            "passes": aggregate["worst_split_ratio"] <= max_split,
        },
        "validation": {
            "value": aggregate["worst_validation_task_ratio"],
            "threshold_at_most": MAX_VALIDATION_TASK,
            "passes": (
                aggregate["worst_validation_task_ratio"]
                <= MAX_VALIDATION_TASK
            ),
        },
        "fit_time": {
            "value": aggregate["fit_ratio"]["median"],
            "threshold_at_most": MAX_MEDIAN_FIT,
            "passes": aggregate["fit_ratio"]["median"] <= MAX_MEDIAN_FIT,
        },
        "predict_time": {
            "value": aggregate["predict_ratio"]["median"],
            "threshold_at_most": MAX_MEDIAN_PREDICT,
            "passes": (
                aggregate["predict_ratio"]["median"] <= MAX_MEDIAN_PREDICT
            ),
        },
        "peak_rss": {
            "value": aggregate["peak_rss_ratio"]["median"],
            "threshold_at_most": MAX_MEDIAN_RSS,
            "passes": (
                aggregate["peak_rss_ratio"]["median"] <= MAX_MEDIAN_RSS
            ),
        },
        "fit_dispersion": {
            "value": aggregate["fit_ratio"]["iqr_over_median"],
            "threshold_at_most": MAX_TIMING_RELATIVE_IQR,
            "passes": (
                aggregate["fit_ratio"]["iqr_over_median"]
                <= MAX_TIMING_RELATIVE_IQR
            ),
        },
        "predict_dispersion": {
            "value": aggregate["predict_ratio"]["iqr_over_median"],
            "threshold_at_most": MAX_TIMING_RELATIVE_IQR,
            "passes": (
                aggregate["predict_ratio"]["iqr_over_median"]
                <= MAX_TIMING_RELATIVE_IQR
            ),
        },
    }
    if tier == runner.CONFIRMATION:
        gates["task_bootstrap"] = {
            "value": aggregate["task_bootstrap"]["upper_ratio"],
            "threshold_strictly_below": 1.0,
            "passes": aggregate["task_bootstrap"]["upper_ratio"] < 1.0,
        }
    return gates


def analyze(raw_path: Path) -> dict[str, Any]:
    raw_payload = raw_path.read_bytes()
    raw = json.loads(raw_payload)
    registry = runner._load_registry()
    pairs, failures, diagnostics = validate_raw(raw, registry)
    expected_pairs = 24 if raw.get("tier") == runner.DEVELOPMENT else 15
    if len(pairs) != expected_pairs:
        failures.append(
            f"complete pair count is {len(pairs)}, expected {expected_pairs}"
        )
    if not pairs:
        raise RuntimeError("native-ordinal C2 raw artifact has no valid pairs")
    aggregate = _aggregate(raw["tier"], pairs)
    gates = _gates(raw["tier"], aggregate, failures)
    if raw["tier"] == runner.DEVELOPMENT:
        probability = float(registry["power_analysis"]["pass_probability"])
        minimum = float(
            registry["power_analysis"]["minimum_required_probability"]
        )
        gates["confirmation_power"] = {
            "value": probability,
            "threshold_at_least": minimum,
            "passes": (
                registry["power_analysis"]["passes"] is True
                and probability >= minimum
            ),
        }
    passes = all(record["passes"] for record in gates.values())
    if raw["tier"] == runner.DEVELOPMENT:
        decision = (
            "authorize_native_ordinal_c2_confirmation_once"
            if passes
            else "close_native_ordinal_c2_development"
        )
    else:
        decision = (
            "retain_explicit_native_ordinal_and_run_mode_mix_attribution"
            if passes
            else "close_native_ordinal_c2_promotion"
        )
    return {
        "schema_version": 1,
        "name": f"darkofit_native_ordinal_c2_{raw['tier']}_result_v1",
        "tier": raw["tier"],
        "raw_path": str(raw_path.relative_to(ROOT)),
        "raw_sha256": _sha256_bytes(raw_payload),
        "registry_sha256": registry["registry_sha256"],
        "passes": passes,
        "decision": decision,
        "confirmation_run_authorized": (
            raw["tier"] == runner.DEVELOPMENT and passes
        ),
        "gates": gates,
        "aggregate": aggregate,
        "integrity_failures": failures,
        "diagnostics": diagnostics,
        "expected_warning_prefixes": list(EXPECTED_WARNING_PREFIXES),
        "development_outcomes_inspected": (
            raw["tier"] == runner.DEVELOPMENT
        ),
        "confirmation_outcomes_inspected": (
            raw["tier"] == runner.CONFIRMATION
        ),
        "lockbox_touched": False,
    }


def _report(summary: dict[str, Any]) -> str:
    status = "PASS" if summary["passes"] else "FAIL"
    aggregate = summary["aggregate"]
    lines = [
        f"# Native ordinal C2 {summary['tier']} result",
        "",
        f"**Decision: {summary['decision']} ({status}).**",
        "",
        "## Headline",
        "",
        (
            f"- Equal-task test RMSE ratio: "
            f"`{aggregate['equal_task_test_rmse_ratio']:.6f}`."
        ),
        (
            f"- Task wins: `{aggregate['task_wins']}` of "
            f"`{aggregate['engaged_task_count']}`."
        ),
        f"- Worst task ratio: `{aggregate['worst_task_ratio']:.6f}`.",
        f"- Worst split ratio: `{aggregate['worst_split_ratio']:.6f}`.",
        (
            f"- Worst validation task ratio: "
            f"`{aggregate['worst_validation_task_ratio']:.6f}`."
        ),
        (
            f"- Median fit / predict / RSS ratios: "
            f"`{aggregate['fit_ratio']['median']:.6f}` / "
            f"`{aggregate['predict_ratio']['median']:.6f}` / "
            f"`{aggregate['peak_rss_ratio']['median']:.6f}`."
        ),
        "",
        "## Per-task quality",
        "",
        "| Task | Dataset | Test ratio | Validation ratio |",
        "|---:|---|---:|---:|",
    ]
    for row in aggregate["task_rows"]:
        lines.append(
            f"| {row['task_id']} | {row['dataset_name']} | "
            f"{row['test_rmse_ratio']:.6f} | "
            f"{row['validation_rmse_ratio']:.6f} |"
        )
    lines.extend([
        "",
        "## Gates",
        "",
        "| Gate | Value | Pass |",
        "|---|---:|:---:|",
    ])
    for name, gate in summary["gates"].items():
        value = gate.get("value", gate.get("failure_count"))
        rendered = (
            f"{value:.6f}" if isinstance(value, float) else str(value)
        )
        lines.append(
            f"| {name} | {rendered} | "
            f"{'yes' if gate['passes'] else 'no'} |"
        )
    if summary["integrity_failures"]:
        lines.extend(["", "## Integrity failures", ""])
        lines.extend(
            f"- {failure}" for failure in summary["integrity_failures"]
        )
    lines.extend([
        "",
        (
            f"Raw artifact: `{summary['raw_path']}` "
            f"(`{summary['raw_sha256']}`)."
        ),
        "",
        "CTR23 lockbox touched: **no**.",
        "",
    ])
    return "\n".join(lines)


def _atomic_create_many(payloads: dict[Path, bytes]) -> None:
    normalized = {
        Path(os.path.abspath(os.path.expanduser(os.fspath(path)))): payload
        for path, payload in payloads.items()
    }
    if len(normalized) != len(payloads):
        raise ValueError("native-ordinal C2 output paths collide")
    for path in normalized:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary: dict[Path, Path] = {}
    created: list[Path] = []
    try:
        for path, payload in normalized.items():
            descriptor, name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            temp = Path(name)
            temporary[path] = temp
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for path, temp in temporary.items():
            os.link(temp, path)
            created.append(path)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        for temp in temporary.values():
            temp.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=runner.TIERS, default=runner.DEVELOPMENT)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.raw is None:
        args.raw = (
            DEFAULT_DEVELOPMENT_RAW
            if args.tier == runner.DEVELOPMENT
            else DEFAULT_CONFIRMATION_RAW
        )
    if args.summary is None:
        args.summary = (
            DEFAULT_DEVELOPMENT_SUMMARY
            if args.tier == runner.DEVELOPMENT
            else DEFAULT_CONFIRMATION_SUMMARY
        )
    if args.report is None:
        args.report = (
            DEFAULT_DEVELOPMENT_REPORT
            if args.tier == runner.DEVELOPMENT
            else DEFAULT_CONFIRMATION_REPORT
        )
    return args


def main() -> int:
    args = parse_args()
    expected_paths = {
        runner.DEVELOPMENT: (
            DEFAULT_DEVELOPMENT_RAW,
            DEFAULT_DEVELOPMENT_SUMMARY,
            DEFAULT_DEVELOPMENT_REPORT,
        ),
        runner.CONFIRMATION: (
            DEFAULT_CONFIRMATION_RAW,
            DEFAULT_CONFIRMATION_SUMMARY,
            DEFAULT_CONFIRMATION_REPORT,
        ),
    }
    observed_paths = tuple(
        path.expanduser().absolute()
        for path in (args.raw, args.summary, args.report)
    )
    if observed_paths != tuple(
        path.absolute() for path in expected_paths[args.tier]
    ):
        raise RuntimeError(
            "formal native-ordinal C2 analysis requires frozen output paths"
        )
    summary = analyze(args.raw.expanduser().absolute())
    if summary["tier"] != args.tier:
        raise RuntimeError("requested analysis tier differs from raw artifact")
    _atomic_create_many({
        args.summary: (
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8"),
        args.report: _report(summary).encode("utf-8"),
    })
    print(
        json.dumps(
            {
                "summary": str(args.summary.expanduser().absolute()),
                "summary_sha256": _sha256_file(
                    args.summary.expanduser().absolute()
                ),
                "report": str(args.report.expanduser().absolute()),
                "report_sha256": _sha256_file(
                    args.report.expanduser().absolute()
                ),
                "passes": summary["passes"],
                "decision": summary["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
