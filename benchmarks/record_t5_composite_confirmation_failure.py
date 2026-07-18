#!/usr/bin/env python3
"""Record the fail-closed T5 execution without fitting another model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_t5_composite_confirmation as runner  # noqa: E402


DEFAULT_SPOOL_DIRECTORY = runner.DEFAULT_SPOOL_DIRECTORY
DEFAULT_OUTPUT = (
    ROOT / "benchmarks" / "t5_composite_confirmation_failure.json"
)
DEFAULT_MARKDOWN = (
    ROOT / "benchmarks" / "t5_composite_confirmation_failure.md"
)
EXPECTED_DARKOFIT_HEAD = "da6881ecf1f58f251c9b3a6486c03000126d292c"
INVALID_TARGETS = {
    362367: {
        "dataset_id": 43462,
        "dataset_name": "Riga-real-estate-dataset",
        "target_name": "price",
        "rows": 4_689,
        "finite": 4_219,
        "nonfinite": 470,
        "nan": 470,
        "posinf": 0,
        "neginf": 0,
    },
    362395: {
        "dataset_id": 43853,
        "dataset_name": "Nintendo3DS-Games",
        "target_name": "metacritic",
        "rows": 1_680,
        "finite": 138,
        "nonfinite": 1_542,
        "nan": 1_542,
        "posinf": 0,
        "neginf": 0,
    },
}


def _json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number: {value}")
    return number


def _json_int(value):
    number = int(value)
    if not -(2**63) <= number <= 2**63 - 1:
        raise ValueError(f"out-of-range JSON integer: {value}")
    return number


def _json_loads(encoded, context):
    try:
        if isinstance(encoded, (bytes, bytearray)):
            encoded = bytes(encoded).decode("utf-8")
        elif not isinstance(encoded, str):
            raise ValueError("JSON input must be UTF-8 bytes or text")
        return json.loads(
            encoded,
            object_pairs_hook=_json_object,
            parse_float=_json_float,
            parse_int=_json_int,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"invalid {context} JSON") from error


def _same_typed_value(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_typed_value(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_typed_value(a, b) for a, b in zip(left, right)
        )
    return left == right


def _reject_symlink_directory(path, message):
    absolute = Path(os.path.abspath(os.path.expanduser(path)))
    for component in (absolute, *absolute.parents):
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"{message}: {component}")


def _create_owned_directories(path, message):
    missing = []
    current = path
    while True:
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            missing.append(current)
            current = current.parent
            continue
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            raise RuntimeError(f"{message}: {current}")
        break
    owned = []
    try:
        for directory in reversed(missing):
            try:
                directory.mkdir()
            except FileExistsError:
                mode = directory.lstat().st_mode
                if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
                    raise RuntimeError(f"{message}: {directory}")
                continue
            identity = directory.lstat()
            if not stat.S_ISDIR(identity.st_mode):
                raise RuntimeError(f"{message}: {directory}")
            owned.append(
                (directory, (identity.st_dev, identity.st_ino))
            )
    except BaseException:
        _remove_owned_directories(owned)
        raise
    return owned


def _remove_owned_directories(owned):
    for directory, expected in reversed(owned):
        try:
            current = directory.lstat()
            if (
                stat.S_ISDIR(current.st_mode)
                and (current.st_dev, current.st_ino) == expected
            ):
                directory.rmdir()
        except OSError:
            pass


def _verify_published_identity(path, expected, message):
    try:
        current = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"{message}: {path}") from error
    if (
        not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino) != expected
    ):
        raise RuntimeError(f"{message}: {path}")


def _dependency_snapshot():
    runner_bytes = Path(runner.__file__).resolve().read_bytes()
    protocol_bytes = runner.PROTOCOL.read_bytes()
    registry_bytes = runner.REGISTRY.read_bytes()
    return {
        "runner_sha256": hashlib.sha256(runner_bytes).hexdigest(),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "registry_bytes": registry_bytes,
    }


def _is_int(value):
    return type(value) is int


def _is_positive_float(value):
    return type(value) is float and math.isfinite(value) and value > 0


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_json_metadata(value):
    if value is None or type(value) in {bool, str}:
        return True
    if type(value) is int:
        return -(2**63) <= value <= 2**63 - 1
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_valid_json_metadata(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _valid_json_metadata(item)
            for key, item in value.items()
        )
    return False


def _valid_linear_leaf_metadata(value):
    if not isinstance(value, dict) or set(value) != {
        "requested",
        "active",
        "inactive_reason",
        "min_samples",
        "linear_lambda",
        "numeric_feature_count",
        "linear_tree_count",
        "linear_leaf_count",
    }:
        return False
    requested = value["requested"]
    active = value["active"]
    reason = value["inactive_reason"]
    numeric_count = value["numeric_feature_count"]
    tree_count = value["linear_tree_count"]
    leaf_count = value["linear_leaf_count"]
    if (
        not isinstance(requested, bool)
        or not isinstance(active, bool)
        or not _is_int(value["min_samples"])
        or value["min_samples"] != 1_000
        or type(value["linear_lambda"]) is not float
        or value["linear_lambda"] != 1.0
        or not _is_int(numeric_count)
        or numeric_count < 0
        or not _is_int(tree_count)
        or tree_count < 0
        or not _is_int(leaf_count)
        or leaf_count < 0
    ):
        return False
    if not requested:
        return (
            active is False
            and reason == "disabled"
            and numeric_count == tree_count == leaf_count == 0
        )
    if active:
        return (
            reason is None
            and numeric_count > 0
            and tree_count > 0
            and leaf_count >= tree_count
        )
    if reason == "no_retained_linear_trees":
        return numeric_count > 0 and tree_count == leaf_count == 0
    return (
        reason in {"below_min_samples", "no_numeric_features"}
        and numeric_count == tree_count == leaf_count == 0
    )


def _valid_core_fit_metadata(value):
    integer_fields = (
        "iterations_requested",
        "iterations_attempted",
        "rounds_completed",
        "rounds_retained",
    )
    return (
        isinstance(value, dict)
        and set(value)
        == {
            *integer_fields,
            "stop_reason",
            "phase_seconds",
        }
        and all(
            _is_int(value.get(name)) and value[name] >= 0
            for name in integer_fields
        )
        and value["rounds_retained"] <= value["rounds_completed"]
        and value["rounds_completed"] <= value["iterations_attempted"]
        and value["iterations_attempted"] <= value["iterations_requested"]
        and isinstance(value.get("stop_reason"), str)
        and bool(value["stop_reason"])
        and isinstance(value.get("phase_seconds"), dict)
        and all(
            isinstance(name, str)
            and bool(name)
            and type(seconds) is float
            and math.isfinite(seconds)
            and seconds >= 0.0
            for name, seconds in value["phase_seconds"].items()
        )
    )


def _valid_fit_metadata(value):
    if not isinstance(value, dict) or set(value) != {
        "best_iteration",
        "fitted_tree_count",
        "resolved_learning_rate",
        "requested_tree_mode",
        "selected_tree_mode",
        "selected_lane",
        "linear_residual_active",
        "linear_leaves_active",
        "linear_leaves",
        "resolved_thread_count",
        "refit",
        "refit_strategy",
        "final_fit",
        "selection_fit",
        "selection_early_stopping_rounds",
        "final_early_stopping_rounds",
    }:
        return False
    selection_fit = value["selection_fit"]
    optional_rounds = (
        value["selection_early_stopping_rounds"],
        value["final_early_stopping_rounds"],
    )
    expected_lane = (
        "linear_residual"
        if value["linear_residual_active"]
        else (
            "linear_leaves"
            if value["linear_leaves_active"]
            else "boosting"
        )
    )
    return (
        _is_int(value["best_iteration"])
        and value["best_iteration"] > 0
        and _is_int(value["fitted_tree_count"])
        and value["fitted_tree_count"] > 0
        and value["best_iteration"] <= value["fitted_tree_count"]
        and _is_positive_float(value["resolved_learning_rate"])
        and isinstance(value["requested_tree_mode"], str)
        and bool(value["requested_tree_mode"])
        and isinstance(value["selected_tree_mode"], str)
        and bool(value["selected_tree_mode"])
        and value["selected_lane"]
        in {"boosting", "linear_residual", "linear_leaves"}
        and isinstance(value["linear_residual_active"], bool)
        and isinstance(value["linear_leaves_active"], bool)
        and value["selected_lane"] == expected_lane
        and _valid_linear_leaf_metadata(value["linear_leaves"])
        and value["linear_leaves_active"]
        is value["linear_leaves"]["active"]
        and _is_int(value["resolved_thread_count"])
        and value["resolved_thread_count"] == runner.THREADS_PER_WORKER
        and isinstance(value["refit"], bool)
        and (
            value["refit_strategy"] is None
            or isinstance(value["refit_strategy"], str)
        )
        and _valid_core_fit_metadata(value["final_fit"])
        and value["final_fit"]["rounds_retained"]
        == value["fitted_tree_count"]
        and (
            selection_fit is None
            or _valid_core_fit_metadata(selection_fit)
        )
        and (
            selection_fit is not None
            or value["selection_early_stopping_rounds"] is None
        )
        and all(
            rounds is None or (_is_int(rounds) and rounds > 0)
            for rounds in optional_rounds
        )
    )


def _valid_product_default_fit(value):
    if not _valid_fit_metadata(value):
        return False
    core = value["final_fit"]
    return (
        value["best_iteration"] == 1_000
        and value["fitted_tree_count"] == 1_000
        and value["requested_tree_mode"] == "catboost"
        and value["selected_tree_mode"] == "catboost"
        and value["selected_lane"] == "boosting"
        and value["linear_residual_active"] is False
        and value["linear_leaves_active"] is False
        and _same_typed_value(
            value["linear_leaves"],
            {
                "requested": False,
                "active": False,
                "inactive_reason": "disabled",
                "min_samples": 1_000,
                "linear_lambda": 1.0,
                "numeric_feature_count": 0,
                "linear_tree_count": 0,
                "linear_leaf_count": 0,
            },
        )
        and value["refit"] is False
        and value["refit_strategy"] is None
        and value["selection_fit"] is None
        and value["selection_early_stopping_rounds"] is None
        and value["final_early_stopping_rounds"] is None
        and core["iterations_requested"] == 1_000
        and core["iterations_attempted"] == 1_000
        and core["rounds_completed"] == 1_000
        and core["rounds_retained"] == 1_000
        and core["stop_reason"] == "iteration_limit"
    )


def _valid_control_metadata(metadata):
    return (
        isinstance(metadata, dict)
        and set(metadata)
        == {"kind", "engaged", "selected_configuration", "final_fit"}
        and metadata.get("kind") == runner.CONTROL
        and metadata.get("engaged") is False
        and metadata.get("selected_configuration") == "product_default"
        and _valid_product_default_fit(metadata.get("final_fit"))
    )


def _registry_snapshot(encoded=None):
    if encoded is None:
        encoded = runner.REGISTRY.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != (
        runner.EXPECTED_REGISTRY_FILE_SHA256
    ):
        raise RuntimeError("T5 registry file identity changed")
    payload = _json_loads(encoded, "T5 registry")
    if (
        not isinstance(payload, dict)
        or payload.get("registry_sha256")
        != runner.EXPECTED_REGISTRY_CANONICAL_SHA256
        or payload.get("confirmation_outcomes_inspected") is not False
        or payload.get("confirmation_run_authorized") is not True
        or payload.get("default_promotion_authorized") is not False
        or payload.get("lockbox_data_used") is not False
        or not _is_int(payload.get("task_count"))
        or payload["task_count"] != 25
        or not _is_int(payload.get("coordinate_count"))
        or payload["coordinate_count"] != 75
        or not isinstance(payload.get("power_analysis"), dict)
        or payload["power_analysis"].get("passes") is not True
        or not isinstance(payload.get("tasks"), list)
    ):
        raise RuntimeError("T5 registry authorization state is invalid")
    rows = {}
    for row in payload["tasks"]:
        if not isinstance(row, dict) or not _is_int(row.get("task_id")):
            raise RuntimeError("T5 registry task matrix changed")
        if row["task_id"] in rows:
            raise RuntimeError("T5 registry task matrix changed")
        rows[row["task_id"]] = row
    if len(rows) != 25:
        raise RuntimeError("T5 registry task matrix changed")
    return payload, rows


def _git_head(repo):
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _target_finiteness(task_id):
    import openml

    task = openml.tasks.get_task(task_id, download_splits=True)
    dataset = task.get_dataset()
    _X, y, _categorical, _names = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )
    values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    return {
        "dataset_id": int(dataset.dataset_id),
        "dataset_name": str(dataset.name),
        "target_name": str(task.target_name),
        "rows": int(len(values)),
        "finite": int(finite.sum()),
        "nonfinite": int((~finite).sum()),
        "nan": int(np.isnan(values).sum()),
        "posinf": int(np.isposinf(values).sum()),
        "neginf": int(np.isneginf(values).sum()),
    }


def _validate_behavior(result):
    if (
        not isinstance(result, dict)
        or not _is_int(result.get("task_id"))
        or result.get("config") != runner.CONTROL
        or not isinstance(result.get("folds"), list)
        or any(not isinstance(fold, dict) for fold in result["folds"])
    ):
        raise RuntimeError("T5 completed worker behavior is invalid")
    behavior = {
        "task_id": result["task_id"],
        "config": result["config"],
        "folds": [
            {
                "fold": fold["fold"],
                "rmse": fold["rmse"],
                "prediction_sha256": fold["prediction_sha256"],
                "metadata": fold["metadata"],
            }
            for fold in result["folds"]
        ],
    }
    if (
        result["behavior_fingerprint_sha256"]
        != runner._json_sha256(behavior)
    ):
        raise RuntimeError("T5 completed worker behavior hash changed")


def _validate_completed_result(result, registry_row):
    if not isinstance(result, dict):
        raise RuntimeError("T5 completed worker result changed")
    expected_result_fields = {
        "task_id",
        "dataset_id",
        "dataset_name",
        "lineage_cluster",
        "stratum",
        "categorical_feature_indices",
        "ordinal_features",
        "config",
        "folds",
        "fold_count",
        "warmup_seconds",
        "wall_seconds",
        "summed_fit_seconds",
        "summed_prediction_block_seconds",
        "peak_rss_bytes",
        "worker_stdout",
        "worker_stderr",
        "behavior_fingerprint_sha256",
    }
    task_id = registry_row["task_id"]
    categorical = result.get("categorical_feature_indices")
    fingerprint = registry_row["task_record"]["fingerprint"]
    expected_categorical = [
        index
        for index, column in enumerate(fingerprint["columns"])
        if column["dtype_family"] != "numeric"
    ]
    if (
        set(result) != expected_result_fields
        or not _is_int(result.get("task_id"))
        or result["task_id"] != task_id
        or result.get("config") != runner.CONTROL
        or not _is_int(result.get("dataset_id"))
        or result["dataset_id"] != registry_row["dataset_id"]
        or result.get("dataset_name") != registry_row["dataset_name"]
        or result.get("lineage_cluster") != registry_row["lineage_cluster"]
        or result.get("stratum") != registry_row["stratum"]
        or not _same_typed_value(
            result.get("ordinal_features"),
            registry_row["ordinal_features"],
        )
        or not isinstance(categorical, list)
        or any(not _is_int(index) for index in categorical)
        or categorical != expected_categorical
        or not _is_int(result.get("fold_count"))
        or result["fold_count"] != len(runner.FOLDS)
        or not _is_positive_float(result.get("warmup_seconds"))
        or not _is_positive_float(result.get("wall_seconds"))
        or not _is_positive_float(result.get("summed_fit_seconds"))
        or not _is_positive_float(
            result.get("summed_prediction_block_seconds")
        )
        or not _is_int(result.get("peak_rss_bytes"))
        or result["peak_rss_bytes"] <= 0
        or (
            result.get("worker_stdout") is not None
            and not isinstance(result.get("worker_stdout"), str)
        )
        or (
            result.get("worker_stderr") is not None
            and not isinstance(result.get("worker_stderr"), str)
        )
    ):
        raise RuntimeError("T5 completed worker result changed")
    folds = result.get("folds")
    if (
        not isinstance(folds, list)
        or any(
            not isinstance(fold, dict) or not _is_int(fold.get("fold"))
            for fold in folds
        )
        or tuple(fold["fold"] for fold in folds) != runner.FOLDS
    ):
        raise RuntimeError("T5 completed worker fold matrix changed")
    for fold in folds:
        expected = runner._expected_split(registry_row, fold["fold"])
        timing = fold.get("prediction_timing")
        metadata = fold.get("metadata")
        if (
            set(fold)
            != {
                "fold",
                "train_rows",
                "test_rows",
                "train_index_sha256",
                "test_index_sha256",
                "rmse",
                "fit_seconds",
                "prediction_timing",
                "prediction_sha256",
                "metadata",
            }
            or not _is_int(fold.get("train_rows"))
            or fold["train_rows"] != expected["train_size"]
            or not _is_int(fold.get("test_rows"))
            or fold["test_rows"] != expected["test_size"]
            or fold.get("train_index_sha256")
            != expected["train_index_sha256"]
            or fold.get("test_index_sha256") != expected["test_index_sha256"]
            or not _is_positive_float(fold.get("rmse"))
            or not _is_positive_float(fold.get("fit_seconds"))
            or not _is_sha256(fold.get("prediction_sha256"))
            or not isinstance(timing, dict)
            or set(timing)
            != {
                "call_count",
                "total_seconds",
                "minimum_block_seconds",
                "per_call_min_seconds",
                "per_call_median_seconds",
                "per_call_max_seconds",
            }
            or not _is_int(timing.get("call_count"))
            or not runner.PREDICTION_MIN_CALLS
            <= timing["call_count"]
            <= runner.PREDICTION_MAX_CALLS
            or not _is_positive_float(timing.get("total_seconds"))
            or timing["total_seconds"] < runner.PREDICTION_BLOCK_SECONDS
            or type(timing.get("minimum_block_seconds")) is not float
            or timing["minimum_block_seconds"]
            != runner.PREDICTION_BLOCK_SECONDS
            or not _is_positive_float(timing.get("per_call_min_seconds"))
            or not _is_positive_float(timing.get("per_call_median_seconds"))
            or not _is_positive_float(timing.get("per_call_max_seconds"))
            or not timing["per_call_min_seconds"]
            <= timing["per_call_median_seconds"]
            <= timing["per_call_max_seconds"]
            <= timing["total_seconds"]
            or timing["total_seconds"]
            < (
                (timing["call_count"] + 1) // 2
                * timing["per_call_median_seconds"]
            )
            or not _valid_control_metadata(metadata)
        ):
            raise RuntimeError("T5 completed worker metric changed")
    if (
        not math.isclose(
            result["summed_fit_seconds"],
            sum(fold["fit_seconds"] for fold in folds),
            rel_tol=1e-15,
            abs_tol=1e-15,
        )
        or not math.isclose(
            result["summed_prediction_block_seconds"],
            sum(fold["prediction_timing"]["total_seconds"] for fold in folds),
            rel_tol=1e-15,
            abs_tol=1e-15,
        )
        or result["wall_seconds"]
        < (
            result["summed_fit_seconds"]
            + result["summed_prediction_block_seconds"]
        )
    ):
        raise RuntimeError("T5 completed worker timing changed")
    _validate_behavior(result)


def _validate_completed_binding(binding, snapshot=None):
    if not isinstance(binding, dict):
        raise RuntimeError("T5 completed-worker binding changed")
    if snapshot is None:
        snapshot = _dependency_snapshot()
    if (
        set(binding)
        != {
            "schema_version",
            "runner_sha256",
            "protocol_sha256",
            "registry_file_sha256",
            "registry_canonical_sha256",
            "darkofit_head",
            "chimeraboost_head",
            "configs",
            "folds",
        }
        or type(binding.get("schema_version")) is not int
        or binding["schema_version"] != 1
        or binding.get("runner_sha256")
        != snapshot["runner_sha256"]
        or binding.get("protocol_sha256") != snapshot["protocol_sha256"]
        or binding.get("registry_file_sha256")
        != runner.EXPECTED_REGISTRY_FILE_SHA256
        or binding.get("registry_canonical_sha256")
        != runner.EXPECTED_REGISTRY_CANONICAL_SHA256
        or binding.get("darkofit_head") != EXPECTED_DARKOFIT_HEAD
        or binding.get("chimeraboost_head") != runner.EXPECTED_CHIMERA_HEAD
        or not isinstance(binding.get("configs"), list)
        or any(not isinstance(config, str) for config in binding["configs"])
        or tuple(binding["configs"]) != runner.CONFIGS
        or not isinstance(binding.get("folds"), list)
        or any(not _is_int(fold) for fold in binding["folds"])
        or tuple(binding["folds"]) != runner.FOLDS
    ):
        raise RuntimeError("T5 completed-worker binding changed")


def _completed_control_paths(spool_directory, expected_task_ids):
    _reject_symlink_directory(
        spool_directory, "refusing symlink T5 failure spool directory"
    )
    if not spool_directory.is_dir():
        raise RuntimeError("T5 failure spool directory is invalid")
    expected = {
        runner._spool_path(
            Path("."), task_id, runner.CONTROL
        ).name: int(task_id)
        for task_id in expected_task_ids
    }
    entries = sorted(spool_directory.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise RuntimeError("T5 failure spool inventory contains a non-file")
    observed = {path.name for path in entries}
    if observed != set(expected):
        unexpected = sorted(observed - set(expected))
        missing = sorted(set(expected) - observed)
        raise RuntimeError(
            "T5 failure spool inventory changed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return [(expected[path.name], path) for path in entries]


def _load_completed_control_spool(path, expected_task_id):
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(
            f"T5 completed-worker spool is invalid: {path}"
        ) from error
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise RuntimeError(
                f"T5 completed-worker spool is not a file: {path}"
            )
        encoded = handle.read()
    payload = _json_loads(encoded, "T5 completed-worker spool")
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "schema_version",
            "name",
            "binding",
            "task_id",
            "config",
            "result_sha256",
            "result",
            "spool_record_sha256",
        }
    ):
        raise RuntimeError("T5 completed-worker payload is invalid")
    spool_hash = payload.get("spool_record_sha256")
    unhashed = dict(payload)
    unhashed.pop("spool_record_sha256", None)
    if spool_hash != runner._json_sha256(unhashed):
        raise RuntimeError("T5 completed-worker spool hash changed")
    result = payload.get("result")
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or payload.get("name") != "darkofit_t5_composite_worker_spool_v1"
        or type(payload.get("task_id")) is not int
        or payload.get("task_id") != expected_task_id
        or payload.get("config") != runner.CONTROL
        or not isinstance(payload.get("binding"), dict)
        or not isinstance(result, dict)
        or type(result.get("task_id")) is not int
        or result.get("task_id") != expected_task_id
        or result.get("config") != runner.CONTROL
        or payload.get("result_sha256") != runner._json_sha256(result)
    ):
        raise RuntimeError("T5 completed-worker spool binding changed")
    return (
        payload,
        result,
        spool_hash,
        hashlib.sha256(encoded).hexdigest(),
    )


def build_failure_record(spool_directory):
    snapshot = _dependency_snapshot()
    registry, rows_by_task = _registry_snapshot(snapshot["registry_bytes"])
    expected_task_ids = {
        int(row["task_id"]) for row in registry["tasks"]
    }
    invalid_task_ids = set(INVALID_TARGETS)
    if not invalid_task_ids < expected_task_ids:
        raise RuntimeError("T5 invalid-task declaration changed")
    observed_invalid = {
        task_id: _target_finiteness(task_id)
        for task_id in sorted(invalid_task_ids)
    }
    if observed_invalid != INVALID_TARGETS:
        raise RuntimeError("T5 invalid-target evidence changed")

    expected_completed = expected_task_ids - invalid_task_ids
    paths = _completed_control_paths(
        spool_directory, expected_completed
    )
    records = []
    for expected_task_id, path in paths:
        payload, result, spool_hash, file_sha256 = (
            _load_completed_control_spool(path, expected_task_id)
        )
        task_id = payload["task_id"]
        binding = payload["binding"]
        _validate_completed_binding(binding, snapshot)
        _validate_completed_result(result, rows_by_task[task_id])
        records.append(
            {
                "task_id": task_id,
                "config": runner.CONTROL,
                "filename": path.name,
                "file_sha256": file_sha256,
                "spool_record_sha256": spool_hash,
                "result_sha256": payload["result_sha256"],
                "behavior_fingerprint_sha256": result[
                    "behavior_fingerprint_sha256"
                ],
            }
        )
    if {row["task_id"] for row in records} != expected_completed:
        raise RuntimeError("T5 completed-worker identities changed")

    if _git_head(ROOT) != EXPECTED_DARKOFIT_HEAD:
        raise RuntimeError("record the T5 failure from its bound commit")
    if _git_head(runner.CHIMERA_ROOT) != runner.EXPECTED_CHIMERA_HEAD:
        raise RuntimeError("T5 ChimeraBoost source changed")
    artifact = {
        "schema_version": 1,
        "name": "darkofit_t5_composite_confirmation_failure_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "close_t5_composite_candidate",
        "failure_reason": "frozen_panel_contains_nonfinite_targets",
        "campaign_complete": False,
        "outcomes_scored": True,
        "candidate_arm_started": False,
        "analysis_performed": False,
        "default_promotion_authorized": False,
        "lockbox_data_used": False,
        "task_drop_allowed": False,
        "task_imputation_allowed": False,
        "rerun_authorized": False,
        "protocol": {
            "path": str(runner.PROTOCOL.relative_to(ROOT)),
            "sha256": snapshot["protocol_sha256"],
            "runner_path": str(
                Path(runner.__file__).resolve().relative_to(ROOT)
            ),
            "runner_sha256": snapshot["runner_sha256"],
            "registry_file_sha256": (
                runner.EXPECTED_REGISTRY_FILE_SHA256
            ),
            "registry_canonical_sha256": (
                runner.EXPECTED_REGISTRY_CANONICAL_SHA256
            ),
        },
        "sources": {
            "darkofit_head": EXPECTED_DARKOFIT_HEAD,
            "chimeraboost_head": runner.EXPECTED_CHIMERA_HEAD,
        },
        "execution": {
            "python": sys.version,
            "interpreter": sys.executable,
            "attempted_wave": runner.CONTROL,
            "expected_worker_count": 25,
            "completed_worker_count": len(records),
            "failed_before_fit_count": len(INVALID_TARGETS),
            "completed_workers": records,
            "invalid_targets": [
                {"task_id": task_id, **observed_invalid[task_id]}
                for task_id in sorted(observed_invalid)
            ],
            "primary_reported_failure_task_id": 362395,
        },
        "panel_disposition": {
            "all_25_lineages_spent_for_confirmation": True,
            "reason": (
                "control outcomes were scored before the frozen data-validity "
                "failure; the panel cannot be repaired or reused"
            ),
        },
    }
    artifact["failure_artifact_sha256"] = runner._json_sha256(artifact)
    return artifact


def _markdown(artifact):
    invalid = artifact["execution"]["invalid_targets"]
    rows = "\n".join(
        f"| {row['task_id']} | {row['dataset_name']} | "
        f"{row['target_name']} | {row['nonfinite']:,} / {row['rows']:,} |"
        for row in invalid
    )
    return f"""# T5 composite confirmation: fail-closed result

**Decision: `close_t5_composite_candidate`.**

The frozen T5 run stopped in its first wave. Twenty-three current-default
workers completed and were persisted; two tasks failed target validation
before fitting. The composite, ChimeraBoost, and CatBoost waves never started.

| Task | Dataset | Target | Non-finite rows |
|---:|---|---|---:|
{rows}

The protocol forbids dropping or imputing a task after outcomes exist. No task
was changed, no run was resumed, and no default promotion is authorized. All
25 lineages are now spent for confirmation because control outcomes were
scored before the failure.

This is a panel-construction failure, not evidence for or against the T5 model
policy. A future nomination needs a new outcome-unseen panel whose target
validity is checked before authorization.
"""


def _atomic_create(path, value, *, _keep_open=False):
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing existing output: {path}")
    message = "refusing symlink T5 failure output directory"
    _reject_symlink_directory(
        path.parent, message
    )
    owned_directories = _create_owned_directories(path.parent, message)
    temporary = None
    temporary_identity = None
    published_identity = None
    handle = None
    descriptor = None
    try:
        try:
            _reject_symlink_directory(path.parent, message)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
            )
            temporary = Path(temporary_name)
            identity = os.fstat(descriptor)
            temporary_identity = (identity.st_dev, identity.st_ino)
            handle = os.fdopen(descriptor, "wb")
            descriptor = None
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise RuntimeError(
                    f"refusing existing output: {path}"
                ) from error
            published_identity = (identity.st_dev, identity.st_ino)
            _verify_published_identity(
                path,
                published_identity,
                "T5 failure output publish identity changed",
            )
        except BaseException:
            if temporary is not None and temporary_identity is not None:
                try:
                    _unlink_if_owned(temporary, temporary_identity)
                except OSError:
                    pass
            if published_identity is not None:
                try:
                    _unlink_if_owned(path, published_identity)
                except OSError:
                    pass
            _remove_owned_directories(owned_directories)
            raise
        try:
            _unlink_if_owned(temporary, temporary_identity)
        except BaseException:
            try:
                _unlink_if_owned(path, published_identity)
            except OSError:
                pass
            _remove_owned_directories(owned_directories)
            raise
        if _keep_open:
            pinned_handle = handle
            handle = None
            return published_identity, owned_directories, pinned_handle
        return published_identity, owned_directories
    finally:
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        elif descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _unlink_if_owned(path, identity):
    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    if (
        stat.S_ISREG(current.st_mode)
        and (current.st_dev, current.st_ino) == identity
    ):
        path.unlink()


def _atomic_create_pair(first_path, first_value, second_path, second_value):
    for path in (first_path, second_path):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing existing output: {path}")
    created = []
    try:
        created.append(
            (
                first_path,
                _atomic_create(first_path, first_value, _keep_open=True),
            )
        )
        created.append(
            (
                second_path,
                _atomic_create(second_path, second_value, _keep_open=True),
            )
        )
    except BaseException:
        for path, (identity, owned_directories, _handle) in reversed(created):
            try:
                _unlink_if_owned(path, identity)
            except OSError:
                pass
            _remove_owned_directories(owned_directories)
        raise
    finally:
        for _path, (_identity, _owned_directories, handle) in created:
            try:
                handle.close()
            except OSError:
                pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spool-directory",
        type=Path,
        default=DEFAULT_SPOOL_DIRECTORY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args(argv)
    args.spool_directory = Path(
        os.path.abspath(os.path.expanduser(args.spool_directory))
    )
    args.output = Path(os.path.abspath(os.path.expanduser(args.output)))
    args.markdown = Path(os.path.abspath(os.path.expanduser(args.markdown)))
    return args


def main(argv=None):
    args = parse_args(argv)
    artifact = build_failure_record(args.spool_directory)
    _atomic_create_pair(
        args.output,
        (
            json.dumps(
                artifact,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode(),
        args.markdown,
        _markdown(artifact).encode(),
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
