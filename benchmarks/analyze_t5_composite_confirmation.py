#!/usr/bin/env python3
"""Validate and analyze the immutable raw T5 composite campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_t5_composite_confirmation as runner  # noqa: E402


DEFAULT_INPUT = ROOT / "benchmarks" / "t5_composite_confirmation_raw.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "t5_composite_confirmation_summary.json"
DEFAULT_MARKDOWN = ROOT / "benchmarks" / "t5_composite_confirmation_result.md"
BOOTSTRAP_SEED = 20260717
BOOTSTRAP_REPLICATES = 100_000
QUALITY_BAR = 0.995
UNCERTAINTY_BAR = 1.002
LOO_BAR = 0.998
HARM_BAR = 1.005
FIT_AGGREGATE_BAR = 6.0
FIT_WORST_DATASET_BAR = 12.0
PREDICT_AGGREGATE_BAR = 1.5
RSS_AGGREGATE_BAR = 2.5
FROZEN_DARKOFIT_HEAD = "da6881ecf1f58f251c9b3a6486c03000126d292c"
GATE_NAMES = (
    "quality_ratio_at_most_0_995",
    "bootstrap_upper_at_most_1_002",
    "least_favorable_loo_at_most_0_998",
    "worst_dataset_at_most_1_005",
    "fit_aggregate_at_most_6",
    "fit_worst_dataset_at_most_12",
    "prediction_aggregate_at_most_1_5",
    "rss_aggregate_at_most_2_5",
    "complete_without_imputation_or_lockbox",
)
RAW_FIELDS = frozenset(
    {
        "schema_version",
        "name",
        "created_at",
        "protocol",
        "sources",
        "environment",
        "registry_power_probability",
        "spool",
        "results",
        "outcomes_scored",
        "analysis_performed",
        "default_promotion_authorized",
        "lockbox_data_used",
        "raw_artifact_sha256",
    }
)
PROTOCOL_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "runner_path",
        "runner_sha256",
        "registry_file_sha256",
        "registry_canonical_sha256",
        "configs",
        "folds",
        "task_count",
        "coordinate_count",
        "worker_count",
        "threads_per_worker",
        "concurrent_workers",
        "size_gate",
        "validation_fraction",
        "outer_guard_ratio",
        "cross_guard_ratio",
        "selection_rounds",
        "prediction_block_seconds",
        "lockbox_data_used",
        "task_drop_allowed",
        "task_imputation_allowed",
    }
)
RESULT_FIELDS = frozenset(
    {
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
)
FOLD_FIELDS = frozenset(
    {
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
)
SOURCE_FIELDS = frozenset(
    {
        "path",
        "head",
        "branch",
        "clean",
        "status",
        "describe",
        "remotes",
        "tracked_main_refs",
    }
)
PREDICTION_TIMING_FIELDS = frozenset(
    {
        "call_count",
        "total_seconds",
        "minimum_block_seconds",
        "per_call_min_seconds",
        "per_call_median_seconds",
        "per_call_max_seconds",
    }
)
MACHINE_FIELDS = frozenset(
    {
        "platform",
        "machine",
        "cpu_brand",
        "logical_cpu_count",
        "python",
        "python_executable",
    }
)
DEPENDENCY_FIELDS = frozenset(
    {
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
        "numba",
        "catboost",
    }
)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


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
    analyzer_bytes = Path(__file__).resolve().read_bytes()
    runner_bytes = Path(runner.__file__).resolve().read_bytes()
    protocol_bytes = runner.PROTOCOL.read_bytes()
    registry_bytes = runner.REGISTRY.read_bytes()
    return {
        "analyzer_sha256": hashlib.sha256(analyzer_bytes).hexdigest(),
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


def _valid_string_mapping(value):
    return isinstance(value, dict) and all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value.items()
    )


def _valid_source(source):
    return (
        isinstance(source, dict)
        and set(source) == SOURCE_FIELDS
        and isinstance(source.get("path"), str)
        and bool(source["path"])
        and isinstance(source.get("head"), str)
        and isinstance(source.get("branch"), str)
        and isinstance(source.get("clean"), bool)
        and isinstance(source.get("status"), list)
        and all(isinstance(line, str) for line in source["status"])
        and (
            source.get("describe") is None
            or isinstance(source.get("describe"), str)
        )
        and _valid_string_mapping(source.get("remotes"))
        and _valid_string_mapping(source.get("tracked_main_refs"))
    )


def _valid_environment(environment):
    machine = (
        environment.get("machine") if isinstance(environment, dict) else None
    )
    dependencies = (
        environment.get("dependencies")
        if isinstance(environment, dict)
        else None
    )
    return (
        isinstance(environment, dict)
        and set(environment) == {"python", "machine", "dependencies"}
        and isinstance(environment.get("python"), str)
        and bool(environment["python"])
        and isinstance(machine, dict)
        and set(machine) == MACHINE_FIELDS
        and isinstance(machine.get("platform"), str)
        and isinstance(machine.get("machine"), str)
        and (
            machine.get("cpu_brand") is None
            or isinstance(machine.get("cpu_brand"), str)
        )
        and (
            machine.get("logical_cpu_count") is None
            or (
                _is_int(machine.get("logical_cpu_count"))
                and machine["logical_cpu_count"] > 0
            )
        )
        and isinstance(machine.get("python"), str)
        and isinstance(machine.get("python_executable"), str)
        and isinstance(dependencies, dict)
        and set(dependencies) == DEPENDENCY_FIELDS
        and all(
            value is None or isinstance(value, str)
            for value in dependencies.values()
        )
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


def _valid_selection_fit(value):
    if not isinstance(value, dict):
        return False
    base_fields = {
        "name",
        "validation_rmse",
        "fit_seconds",
        "fit_metadata",
        "validation",
    }
    optional_fields = {
        "tree_mode_selection",
        "pair_count",
        "pairs",
        "transform_seconds",
    }
    if (
        not base_fields <= set(value)
        or not set(value) - base_fields <= optional_fields
    ):
        return False
    cross_fields = {"pair_count", "pairs", "transform_seconds"}
    present_cross_fields = set(value) & cross_fields
    if (
        value["name"] == "challenger_crossed"
        and present_cross_fields != cross_fields
    ) or (
        value["name"] != "challenger_crossed"
        and present_cross_fields
    ):
        return False
    if (
        value["name"]
        not in {
            "control_audition",
            "challenger_auto",
            "challenger_catboost_linear",
            "challenger_crossed",
        }
        or not _is_positive_float(value["validation_rmse"])
        or not _is_positive_float(value["fit_seconds"])
        or not _valid_fit_metadata(value["fit_metadata"])
        or (
            value["name"] in {"control_audition", "challenger_auto"}
            and value["fit_metadata"]["linear_leaves"]["requested"] is not False
        )
        or not isinstance(value["validation"], dict)
        or value["validation"].get("source") != "explicit_eval_set"
        or not _valid_json_metadata(value["validation"])
        or (
            "tree_mode_selection" in value
            and not _valid_json_metadata(value["tree_mode_selection"])
        )
    ):
        return False
    if cross_fields <= set(value):
        pairs = value["pairs"]
        return (
            _is_int(value["pair_count"])
            and value["pair_count"] > 0
            and isinstance(pairs, list)
            and len(pairs) == value["pair_count"]
            and all(
                isinstance(pair, list)
                and len(pair) == 3
                and all(
                    _is_int(index) and index >= 0 for index in pair[:2]
                )
                and pair[0] != pair[1]
                and pair[2] in {"diff", "prod"}
                for pair in pairs
            )
            and type(value["transform_seconds"]) is float
            and math.isfinite(value["transform_seconds"])
            and value["transform_seconds"] >= 0.0
        )
    return True


def _valid_cross_pair_matrix(pairs, n_features, categorical_indices):
    if (
        not _is_int(n_features)
        or n_features <= 0
        or not isinstance(categorical_indices, list)
        or any(
            not _is_int(index) or index < 0 or index >= n_features
            for index in categorical_indices
        )
    ):
        return False
    categorical = set(categorical_indices)
    if len(categorical) != len(categorical_indices):
        return False
    numeric_count = n_features - len(categorical)
    selected_count = min(6, numeric_count)
    expected_pair_count = selected_count * (selected_count - 1)
    if not isinstance(pairs, list) or len(pairs) != expected_pair_count:
        return False
    base_pairs = []
    for index in range(0, len(pairs), 2):
        difference = pairs[index]
        product = pairs[index + 1]
        if (
            not isinstance(difference, list)
            or not isinstance(product, list)
            or len(difference) != 3
            or len(product) != 3
            or difference[:2] != product[:2]
            or difference[2] != "diff"
            or product[2] != "prod"
        ):
            return False
        left, right = difference[:2]
        if (
            not _is_int(left)
            or not _is_int(right)
            or left < 0
            or right < 0
            or left >= n_features
            or right >= n_features
            or left == right
            or left in categorical
            or right in categorical
        ):
            return False
        base_pairs.append(frozenset((left, right)))
    selected_features = set().union(*base_pairs) if base_pairs else set()
    return (
        len(selected_features) == selected_count
        and len(set(base_pairs)) == selected_count * (selected_count - 1) // 2
    )


def _valid_final_challenger_fit(final_fit, selected_fit, metadata):
    selected_rounds = metadata["selected_best_iteration"]
    core = final_fit["final_fit"]
    return (
        selected_fit["selected_tree_mode"] == metadata["selected_tree_mode"]
        and selected_fit["linear_residual_active"] is False
        and selected_fit["linear_leaves_active"]
        is metadata["selected_linear_leaves"]
        and selected_fit["linear_leaves"]["requested"]
        is metadata["selected_linear_leaves"]
        and final_fit["best_iteration"] == selected_rounds
        and final_fit["fitted_tree_count"] == selected_rounds
        and final_fit["resolved_learning_rate"]
        == metadata["selected_resolved_learning_rate"]
        and final_fit["requested_tree_mode"]
        == metadata["selected_tree_mode"]
        and final_fit["selected_tree_mode"] == metadata["selected_tree_mode"]
        and final_fit["linear_residual_active"] is False
        and final_fit["linear_leaves_active"]
        is metadata["selected_linear_leaves"]
        and final_fit["linear_leaves"]["requested"]
        is metadata["selected_linear_leaves"]
        and final_fit["refit"] is False
        and final_fit["refit_strategy"] is None
        and final_fit["selection_fit"] is None
        and final_fit["selection_early_stopping_rounds"] is None
        and final_fit["final_early_stopping_rounds"] is None
        and core["iterations_requested"] == selected_rounds
        and core["iterations_attempted"] == selected_rounds
        and core["rounds_completed"] == selected_rounds
        and core["rounds_retained"] == selected_rounds
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


def _valid_composite_metadata(
    metadata,
    train_rows,
    fit_seconds,
    n_features,
    categorical_indices,
):
    if not isinstance(metadata, dict):
        return False
    if train_rows < runner.SIZE_GATE:
        return (
            set(metadata)
            == {
                "kind",
                "engaged",
                "selected_configuration",
                "final_fit",
                "decline_reason",
                "size_gate",
                "total_selection_fit_seconds",
            }
            and metadata.get("kind") == runner.COMPOSITE
            and metadata.get("engaged") is False
            and metadata.get("selected_configuration") == "product_default"
            and metadata.get("decline_reason") == "below_size_gate"
            and _is_int(metadata.get("size_gate"))
            and metadata["size_gate"] == runner.SIZE_GATE
            and type(metadata.get("total_selection_fit_seconds")) is float
            and metadata["total_selection_fit_seconds"] == 0.0
            and _valid_product_default_fit(metadata.get("final_fit"))
        )
    if set(metadata) != {
        "kind",
        "engaged",
        "decline_reason",
        "size_gate",
        "split",
        "outer_guard_ratio",
        "cross_guard_ratio",
        "selection_rounds",
        "control_validation_rmse",
        "challenger_validation_rmse",
        "relative_challenger_validation_ratio",
        "selected_configuration",
        "selected_tree_mode",
        "selected_linear_leaves",
        "selected_crosses",
        "selected_cross_pairs",
        "selected_cross_pair_count",
        "selected_best_iteration",
        "selected_resolved_learning_rate",
        "selection_fits",
        "total_selection_fit_seconds",
        "final_transform_seconds",
        "final_fit_seconds",
        "final_fit",
    }:
        return False
    engaged = metadata.get("engaged")
    control = metadata.get("control_validation_rmse")
    challenger = metadata.get("challenger_validation_rmse")
    ratio = metadata.get("relative_challenger_validation_ratio")
    pairs = metadata.get("selected_cross_pairs")
    selection_fits = metadata.get("selection_fits")
    expected_split = runner._selection_split(train_rows)[2]
    if (
        metadata.get("kind") != runner.COMPOSITE
        or not isinstance(engaged, bool)
        or not _is_int(metadata.get("size_gate"))
        or metadata["size_gate"] != runner.SIZE_GATE
        or not _same_typed_value(metadata.get("split"), expected_split)
        or type(metadata.get("outer_guard_ratio")) is not float
        or metadata["outer_guard_ratio"] != runner.OUTER_GUARD_RATIO
        or type(metadata.get("cross_guard_ratio")) is not float
        or metadata["cross_guard_ratio"] != runner.CROSS_GUARD_RATIO
        or not _is_int(metadata.get("selection_rounds"))
        or metadata["selection_rounds"] != runner.SELECTION_ROUNDS
        or not _is_positive_float(control)
        or not _is_positive_float(challenger)
        or not _is_positive_float(ratio)
        or not math.isclose(
            ratio,
            challenger / control,
            rel_tol=1e-15,
            abs_tol=1e-15,
        )
        or engaged
        is not (challenger <= runner.OUTER_GUARD_RATIO * control)
        or metadata.get("selected_configuration")
        != ("challenger" if engaged else "product_default")
        or metadata.get("decline_reason")
        != (None if engaged else "outer_validation_guard")
        or not isinstance(metadata.get("selected_tree_mode"), str)
        or not metadata["selected_tree_mode"]
        or not isinstance(metadata.get("selected_linear_leaves"), bool)
        or not isinstance(metadata.get("selected_crosses"), bool)
        or not isinstance(pairs, list)
        or any(
            not isinstance(pair, list)
            or len(pair) != 3
            or any(not _is_int(index) or index < 0 for index in pair[:2])
            or pair[0] == pair[1]
            or pair[2] not in {"diff", "prod"}
            for pair in pairs
        )
        or not _is_int(metadata.get("selected_cross_pair_count"))
        or metadata["selected_cross_pair_count"] != len(pairs)
        or metadata["selected_crosses"] is not bool(pairs)
        or not _is_int(metadata.get("selected_best_iteration"))
        or metadata["selected_best_iteration"] <= 0
        or not _is_positive_float(
            metadata.get("selected_resolved_learning_rate")
        )
        or not isinstance(selection_fits, list)
        or not 2 <= len(selection_fits) <= 4
        or any(not _valid_selection_fit(row) for row in selection_fits)
        or [row["name"] for row in selection_fits[:2]]
        != ["control_audition", "challenger_auto"]
        or len({row["name"] for row in selection_fits})
        != len(selection_fits)
        or not _is_positive_float(
            metadata.get("total_selection_fit_seconds")
        )
        or not math.isclose(
            metadata["total_selection_fit_seconds"],
            sum(row["fit_seconds"] for row in selection_fits),
            rel_tol=1e-15,
            abs_tol=1e-15,
        )
        or type(metadata.get("final_transform_seconds")) is not float
        or not math.isfinite(metadata["final_transform_seconds"])
        or metadata["final_transform_seconds"] < 0.0
        or not _is_positive_float(metadata.get("final_fit_seconds"))
        or metadata["final_transform_seconds"] > metadata["final_fit_seconds"]
        or not math.isclose(
            fit_seconds,
            metadata["total_selection_fit_seconds"]
            + metadata["final_fit_seconds"],
            rel_tol=1e-15,
            abs_tol=1e-15,
        )
        or not _valid_fit_metadata(metadata.get("final_fit"))
    ):
        return False
    names = [row["name"] for row in selection_fits]
    if names not in (
        ["control_audition", "challenger_auto"],
        ["control_audition", "challenger_auto", "challenger_crossed"],
        [
            "control_audition",
            "challenger_auto",
            "challenger_catboost_linear",
        ],
        [
            "control_audition",
            "challenger_auto",
            "challenger_catboost_linear",
            "challenger_crossed",
        ],
    ):
        return False
    by_name = {row["name"]: row for row in selection_fits}
    uncrossed = by_name["challenger_auto"]
    uncrossed_fit = uncrossed["fit_metadata"]
    if (
        uncrossed_fit["requested_tree_mode"] != "auto"
        or uncrossed_fit["linear_residual_active"] is not False
        or uncrossed_fit["linear_leaves_active"] is not False
    ):
        return False
    linear = by_name.get("challenger_catboost_linear")
    if linear is not None:
        linear_fit = linear["fit_metadata"]
        if (
            uncrossed_fit["selected_tree_mode"] != "catboost"
            or linear_fit["requested_tree_mode"] != "catboost"
            or linear_fit["selected_tree_mode"] != "catboost"
            or linear_fit["linear_residual_active"] is not False
            or linear_fit["linear_leaves_active"] is not True
        ):
            return False
    expected_linear = (
        linear is not None
        and linear["validation_rmse"] < uncrossed["validation_rmse"]
    )
    if expected_linear:
        uncrossed = linear
    crossed = by_name.get("challenger_crossed")
    if crossed is not None and (
        crossed["transform_seconds"] > crossed["fit_seconds"]
        or crossed["fit_metadata"]["requested_tree_mode"]
        != uncrossed["fit_metadata"]["selected_tree_mode"]
        or crossed["fit_metadata"]["selected_tree_mode"]
        != uncrossed["fit_metadata"]["selected_tree_mode"]
        or crossed["fit_metadata"]["linear_residual_active"] is not False
        or crossed["fit_metadata"]["linear_leaves_active"]
        is not expected_linear
        or crossed["fit_metadata"]["linear_leaves"]["requested"]
        is not expected_linear
        or not _valid_cross_pair_matrix(
            crossed["pairs"],
            n_features,
            categorical_indices,
        )
    ):
        return False
    expected_crosses = (
        crossed is not None
        and crossed["validation_rmse"]
        <= runner.CROSS_GUARD_RATIO * uncrossed["validation_rmse"]
    )
    selected = crossed if expected_crosses else uncrossed
    expected_pairs = crossed["pairs"] if expected_crosses else []
    return (
        metadata["control_validation_rmse"]
        == by_name["control_audition"]["validation_rmse"]
        and metadata["challenger_validation_rmse"]
        == selected["validation_rmse"]
        and metadata["selected_linear_leaves"] is expected_linear
        and metadata["selected_crosses"] is expected_crosses
        and _same_typed_value(pairs, expected_pairs)
        and metadata["selected_best_iteration"]
        == selected["fit_metadata"]["best_iteration"]
        and metadata["selected_resolved_learning_rate"]
        == selected["fit_metadata"]["resolved_learning_rate"]
        and metadata["selected_tree_mode"]
        == selected["fit_metadata"]["selected_tree_mode"]
        and (
            metadata["final_transform_seconds"] == 0.0
            if not engaged or not expected_crosses
            else True
        )
        and (
            _valid_final_challenger_fit(
                metadata["final_fit"],
                selected["fit_metadata"],
                metadata,
            )
            if engaged
            else _valid_product_default_fit(metadata["final_fit"])
        )
    )


def _valid_fold_metadata(
    metadata,
    config,
    train_rows,
    fit_seconds,
    n_features,
    categorical_indices,
):
    if config == runner.CONTROL:
        return _valid_control_metadata(metadata)
    if config == runner.COMPOSITE:
        return _valid_composite_metadata(
            metadata,
            train_rows,
            fit_seconds,
            n_features,
            categorical_indices,
        )
    if config == runner.CHIMERA:
        return (
            isinstance(metadata, dict)
            and set(metadata)
            == {
                "kind",
                "fitted_tree_count",
                "resolved_learning_rate",
                "linear_leaves_selected",
                "cross_features_selected",
                "cross_pair_count",
            }
            and metadata.get("kind") == runner.CHIMERA
            and _is_int(metadata.get("fitted_tree_count"))
            and metadata["fitted_tree_count"] > 0
            and _is_positive_float(metadata.get("resolved_learning_rate"))
            and isinstance(metadata.get("linear_leaves_selected"), bool)
            and isinstance(metadata.get("cross_features_selected"), bool)
            and _is_int(metadata.get("cross_pair_count"))
            and metadata["cross_pair_count"] >= 0
        )
    if config == runner.CATBOOST:
        return (
            isinstance(metadata, dict)
            and set(metadata) == {"kind", "fitted_tree_count", "best_iteration"}
            and metadata.get("kind") == runner.CATBOOST
            and _is_int(metadata.get("fitted_tree_count"))
            and metadata["fitted_tree_count"] > 0
            and _is_int(metadata.get("best_iteration"))
            and -1 <= metadata["best_iteration"] < metadata["fitted_tree_count"]
        )
    return False


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


def _geomean(values):
    values = np.asarray(values, dtype=np.float64)
    if (
        values.ndim != 1
        or not values.size
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
    ):
        raise RuntimeError("T5 geometric mean received invalid values")
    return float(np.exp(np.mean(np.log(values))))


def _validate(raw, snapshot=None):
    if (
        not isinstance(raw, dict)
        or set(raw) != RAW_FIELDS
        or not _is_int(raw.get("schema_version"))
        or raw.get("schema_version") != 1
        or raw.get("name") != "darkofit_t5_composite_confirmation_raw_v1"
    ):
        raise RuntimeError("T5 raw artifact name changed")
    expected_hash = raw.get("raw_artifact_sha256")
    unhashed = dict(raw)
    unhashed.pop("raw_artifact_sha256", None)
    try:
        observed_hash = _json_sha256(unhashed)
    except (TypeError, ValueError) as error:
        raise RuntimeError("T5 raw artifact hash is invalid") from error
    if not _is_sha256(expected_hash) or expected_hash != observed_hash:
        raise RuntimeError("T5 raw artifact hash is invalid")
    if snapshot is None:
        snapshot = _dependency_snapshot()
    try:
        created_at = datetime.fromisoformat(raw["created_at"])
    except (TypeError, ValueError) as error:
        raise RuntimeError("T5 raw creation timestamp changed") from error
    environment = raw.get("environment")
    if (
        created_at.tzinfo is None
        or created_at.utcoffset() != timezone.utc.utcoffset(created_at)
        or not _valid_environment(environment)
    ):
        raise RuntimeError("T5 raw execution environment changed")
    if (
        raw.get("outcomes_scored") is not True
        or raw.get("analysis_performed") is not False
        or raw.get("default_promotion_authorized") is not False
        or raw.get("lockbox_data_used") is not False
    ):
        raise RuntimeError("T5 raw artifact state is invalid")
    protocol = raw.get("protocol")
    if (
        not isinstance(protocol, dict)
        or set(protocol) != PROTOCOL_FIELDS
        or protocol.get("path")
        != str(runner.PROTOCOL.relative_to(ROOT))
        or protocol.get("sha256") != snapshot["protocol_sha256"]
        or protocol.get("runner_path")
        != str(Path(runner.__file__).resolve().relative_to(ROOT))
        or protocol.get("runner_sha256") != snapshot["runner_sha256"]
        or protocol.get("registry_file_sha256")
        != runner.EXPECTED_REGISTRY_FILE_SHA256
        or protocol.get("registry_canonical_sha256")
        != runner.EXPECTED_REGISTRY_CANONICAL_SHA256
        or not isinstance(protocol.get("configs"), list)
        or tuple(protocol["configs"]) != runner.CONFIGS
        or any(not isinstance(config, str) for config in protocol["configs"])
        or not isinstance(protocol.get("folds"), list)
        or tuple(protocol["folds"]) != runner.FOLDS
        or any(not _is_int(fold) for fold in protocol["folds"])
        or not _is_int(protocol.get("task_count"))
        or protocol["task_count"] != 25
        or not _is_int(protocol.get("coordinate_count"))
        or protocol["coordinate_count"] != 75
        or not _is_int(protocol.get("worker_count"))
        or protocol["worker_count"] != 100
        or not _is_int(protocol.get("threads_per_worker"))
        or protocol["threads_per_worker"] != runner.THREADS_PER_WORKER
        or not _is_int(protocol.get("concurrent_workers"))
        or protocol["concurrent_workers"] != runner.CONCURRENT_WORKERS
        or not _is_int(protocol.get("size_gate"))
        or protocol["size_gate"] != runner.SIZE_GATE
        or type(protocol.get("validation_fraction")) is not float
        or protocol["validation_fraction"] != runner.VALIDATION_FRACTION
        or type(protocol.get("outer_guard_ratio")) is not float
        or protocol["outer_guard_ratio"] != runner.OUTER_GUARD_RATIO
        or type(protocol.get("cross_guard_ratio")) is not float
        or protocol["cross_guard_ratio"] != runner.CROSS_GUARD_RATIO
        or not _is_int(protocol.get("selection_rounds"))
        or protocol["selection_rounds"] != runner.SELECTION_ROUNDS
        or type(protocol.get("prediction_block_seconds")) is not float
        or protocol["prediction_block_seconds"]
        != runner.PREDICTION_BLOCK_SECONDS
        or protocol.get("lockbox_data_used") is not False
        or protocol.get("task_drop_allowed") is not False
        or protocol.get("task_imputation_allowed") is not False
    ):
        raise RuntimeError("T5 raw protocol changed")
    registry, _rows = _registry_snapshot(snapshot["registry_bytes"])
    expected_task_ids = {
        row["task_id"] for row in registry["tasks"]
    }
    if (
        type(raw.get("registry_power_probability")) is not float
        or raw["registry_power_probability"]
        != registry["power_analysis"]["pass_probability"]
    ):
        raise RuntimeError("T5 registry power record changed")
    sources = raw.get("sources")
    if (
        not isinstance(sources, dict)
        or set(sources) != {"darkofit", "chimeraboost"}
        or not _valid_source(sources.get("darkofit"))
        or not _valid_source(sources.get("chimeraboost"))
        or sources["darkofit"].get("clean") is not True
        or sources["darkofit"].get("head") != FROZEN_DARKOFIT_HEAD
        or sources["darkofit"].get("status") != []
        or sources["chimeraboost"].get("clean") is not True
        or sources["chimeraboost"].get("head")
        != runner.EXPECTED_CHIMERA_HEAD
        or sources["chimeraboost"].get("status") != []
    ):
        raise RuntimeError("T5 raw source state is invalid")
    results = raw.get("results")
    if not isinstance(results, list) or any(
        not isinstance(row, dict)
        or set(row) != RESULT_FIELDS
        or not _is_int(row.get("task_id"))
        or not isinstance(row.get("config"), str)
        for row in results
    ):
        raise RuntimeError("T5 raw worker matrix is incomplete")
    expected = {
        (task_id, config)
        for task_id in expected_task_ids
        for config in runner.CONFIGS
    }
    observed = {(row["task_id"], row["config"]) for row in results}
    if (
        len(results) != 100
        or len({task_id for task_id, _config in observed}) != 25
        or observed != expected
    ):
        raise RuntimeError("T5 raw worker matrix is incomplete")
    spool = raw.get("spool")
    if (
        not isinstance(spool, dict)
        or set(spool)
        != {"binding", "record_count", "resumed_record_count", "records"}
        or not isinstance(spool.get("binding"), dict)
        or not isinstance(spool.get("records"), list)
    ):
        raise RuntimeError("T5 raw spool binding changed")
    binding = spool["binding"]
    if any(
        not isinstance(row, dict)
        or set(row)
        != {"task_id", "config", "filename", "sha256", "resumed"}
        or not _is_int(row.get("task_id"))
        or not isinstance(row.get("config"), str)
        or not isinstance(row.get("filename"), str)
        or not isinstance(row.get("resumed"), bool)
        or not _is_sha256(row.get("sha256"))
        for row in spool["records"]
    ):
        raise RuntimeError("T5 raw spool matrix is incomplete")
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
        or binding.get("runner_sha256") != protocol["runner_sha256"]
        or binding.get("protocol_sha256") != protocol["sha256"]
        or binding.get("registry_file_sha256")
        != runner.EXPECTED_REGISTRY_FILE_SHA256
        or binding.get("registry_canonical_sha256")
        != runner.EXPECTED_REGISTRY_CANONICAL_SHA256
        or binding.get("darkofit_head") != sources["darkofit"]["head"]
        or binding.get("chimeraboost_head")
        != sources["chimeraboost"]["head"]
        or not isinstance(binding.get("configs"), list)
        or tuple(binding["configs"]) != runner.CONFIGS
        or any(not isinstance(config, str) for config in binding["configs"])
        or not isinstance(binding.get("folds"), list)
        or tuple(binding["folds"]) != runner.FOLDS
        or any(not _is_int(fold) for fold in binding["folds"])
        or not _is_int(spool.get("record_count"))
        or spool["record_count"] != len(expected)
        or not _is_int(spool.get("resumed_record_count"))
        or spool["resumed_record_count"]
        != sum(row.get("resumed") is True for row in spool["records"])
    ):
        raise RuntimeError("T5 raw spool binding changed")
    spool_coordinates = {
        (row["task_id"], row["config"])
        for row in spool["records"]
    }
    if (
        not _is_int(binding.get("schema_version"))
        or binding.get("schema_version") != 1
        or len(spool["records"]) != len(expected)
        or spool_coordinates != expected
        or len({row["sha256"] for row in spool["records"]})
        != len(expected)
        or any(
            row["filename"]
            != runner._spool_path(
                Path("."), row["task_id"], row["config"]
            ).name
            for row in spool["records"]
        )
    ):
        raise RuntimeError("T5 raw spool matrix is incomplete")
    by_key = {}
    identity = {}
    for row in results:
        key = (row["task_id"], row["config"])
        if key in by_key:
            raise RuntimeError(f"T5 raw duplicate worker: {key}")
        by_key[key] = row
        current_identity = (
            row.get("dataset_name"),
            row.get("lineage_cluster"),
            row.get("stratum"),
        )
        registry_row = _rows[key[0]]
        expected_identity = (
            str(registry_row["dataset_name"]),
            str(registry_row["lineage_cluster"]),
            str(registry_row["stratum"]),
        )
        if (
            not all(isinstance(value, str) for value in current_identity)
            or current_identity != expected_identity
        ):
            raise RuntimeError(f"T5 registry identity changed: {key[0]}")
        if (
            not _is_int(row.get("dataset_id"))
            or row["dataset_id"] != int(registry_row["dataset_id"])
            or not _same_typed_value(
                row.get("ordinal_features"),
                registry_row["ordinal_features"],
            )
            or not _is_int(row.get("fold_count"))
            or row["fold_count"] != len(runner.FOLDS)
        ):
            raise RuntimeError(f"T5 worker declaration changed: {key}")
        categorical = row.get("categorical_feature_indices")
        fingerprint = registry_row["task_record"]["fingerprint"]
        expected_categorical = [
            index
            for index, column in enumerate(fingerprint["columns"])
            if column["dtype_family"] != "numeric"
        ]
        if (
            not isinstance(categorical, list)
            or any(not _is_int(index) for index in categorical)
            or categorical != expected_categorical
            or not _is_positive_float(row.get("warmup_seconds"))
            or not _is_positive_float(row.get("wall_seconds"))
            or not _is_positive_float(
                row.get("summed_prediction_block_seconds")
            )
            or (
                row.get("worker_stdout") is not None
                and not isinstance(row.get("worker_stdout"), str)
            )
            or (
                row.get("worker_stderr") is not None
                and not isinstance(row.get("worker_stderr"), str)
            )
        ):
            raise RuntimeError(f"T5 worker resource record is invalid: {key}")
        previous = identity.setdefault(key[0], current_identity)
        if previous != current_identity:
            raise RuntimeError(f"T5 task identity changed: {key[0]}")
        folds = row.get("folds")
        if (
            not isinstance(folds, list)
            or any(
                not isinstance(fold, dict)
                or set(fold) != FOLD_FIELDS
                or not _is_int(fold.get("fold"))
                for fold in folds
            )
            or tuple(fold["fold"] for fold in folds) != runner.FOLDS
        ):
            raise RuntimeError(f"T5 fold order changed: {key}")
        if (
            not _is_int(row.get("peak_rss_bytes"))
            or row["peak_rss_bytes"] <= 0
        ):
            raise RuntimeError(f"T5 resource record is invalid: {key}")
        for fold in folds:
            expected_split = runner._expected_split(
                registry_row, fold["fold"]
            )
            if (
                not _is_int(fold.get("train_rows"))
                or fold["train_rows"] != expected_split["train_size"]
                or not _is_int(fold.get("test_rows"))
                or fold["test_rows"] != expected_split["test_size"]
                or fold["train_index_sha256"]
                != expected_split["train_index_sha256"]
                or fold["test_index_sha256"]
                != expected_split["test_index_sha256"]
                or not _is_positive_float(fold.get("rmse"))
                or not _is_positive_float(fold.get("fit_seconds"))
                or not _is_sha256(fold.get("prediction_sha256"))
            ):
                raise RuntimeError(f"T5 metric is invalid: {key}")
            timing = fold.get("prediction_timing")
            if (
                not isinstance(timing, dict)
                or set(timing) != PREDICTION_TIMING_FIELDS
                or not _is_int(timing.get("call_count"))
                or timing["call_count"] < runner.PREDICTION_MIN_CALLS
                or timing["call_count"] > runner.PREDICTION_MAX_CALLS
                or not _is_positive_float(timing.get("total_seconds"))
                or timing["total_seconds"]
                < runner.PREDICTION_BLOCK_SECONDS
                or type(timing.get("minimum_block_seconds")) is not float
                or timing["minimum_block_seconds"]
                != runner.PREDICTION_BLOCK_SECONDS
                or not _is_positive_float(timing.get("per_call_min_seconds"))
                or not _is_positive_float(
                    timing.get("per_call_median_seconds")
                )
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
            ):
                raise RuntimeError(f"T5 prediction block is invalid: {key}")
            metadata = fold.get("metadata")
            if not _valid_fold_metadata(
                metadata,
                key[1],
                fold["train_rows"],
                fold["fit_seconds"],
                int(fingerprint["n_features"]),
                categorical,
            ):
                raise RuntimeError(f"T5 fold metadata is invalid: {key}")
        expected_behavior = {
            "task_id": key[0],
            "config": key[1],
            "folds": [
                {
                    "fold": fold["fold"],
                    "rmse": fold["rmse"],
                    "prediction_sha256": fold["prediction_sha256"],
                    "metadata": fold["metadata"],
                }
                for fold in folds
            ],
        }
        if (
            not _is_sha256(row.get("behavior_fingerprint_sha256"))
            or row["behavior_fingerprint_sha256"]
            != _json_sha256(expected_behavior)
            or not _is_positive_float(row.get("summed_fit_seconds"))
            or not math.isclose(
                row["summed_fit_seconds"],
                sum(fold["fit_seconds"] for fold in folds),
                rel_tol=1e-15,
                abs_tol=1e-15,
            )
            or not math.isclose(
                row["summed_prediction_block_seconds"],
                sum(
                    fold["prediction_timing"]["total_seconds"]
                    for fold in folds
                ),
                rel_tol=1e-15,
                abs_tol=1e-15,
            )
            or row["wall_seconds"]
            < (
                row["summed_fit_seconds"]
                + row["summed_prediction_block_seconds"]
            )
        ):
            raise RuntimeError(f"T5 worker behavior changed: {key}")
    if len(set(identity.values())) != 25:
        raise RuntimeError("T5 lineage identity is not unique")

    for task_id in identity:
        control = by_key[(task_id, runner.CONTROL)]
        composite = by_key[(task_id, runner.COMPOSITE)]
        for control_fold, composite_fold in zip(
            control["folds"], composite["folds"]
        ):
            engaged = composite_fold["metadata"]["engaged"]
            if not engaged and (
                composite_fold["prediction_sha256"]
                != control_fold["prediction_sha256"]
                or float(composite_fold["rmse"])
                != float(control_fold["rmse"])
                or composite_fold["metadata"]["final_fit"][
                    "resolved_learning_rate"
                ]
                != control_fold["metadata"]["final_fit"][
                    "resolved_learning_rate"
                ]
            ):
                raise RuntimeError(
                    f"T5 exact decline differs from control: "
                    f"{task_id}/{control_fold['fold']}"
                )
    for record in spool["records"]:
        key = (record["task_id"], record["config"])
        result = by_key[key]
        payload = {
            "schema_version": 1,
            "name": "darkofit_t5_composite_worker_spool_v1",
            "binding": binding,
            "task_id": key[0],
            "config": key[1],
            "result_sha256": _json_sha256(result),
            "result": result,
        }
        if record["sha256"] != _json_sha256(payload):
            raise RuntimeError(f"T5 raw spool result changed: {key}")
    return by_key, identity


def _quality_contrast(by_key, identity, numerator, denominator):
    per_dataset = {}
    logs = []
    for task_id in sorted(identity):
        dataset, lineage, stratum = identity[task_id]
        top = by_key[(task_id, numerator)]
        bottom = by_key[(task_id, denominator)]
        ratios = np.asarray(
            [
                float(a["rmse"]) / float(b["rmse"])
                for a, b in zip(top["folds"], bottom["folds"])
            ],
            dtype=np.float64,
        )
        ratio = _geomean(ratios)
        logs.append(np.log(ratios))
        per_dataset[lineage] = {
            "task_id": task_id,
            "dataset_name": dataset,
            "stratum": stratum,
            "ratio": ratio,
            "split_ratios": ratios.tolist(),
        }
    dataset_ratios = [row["ratio"] for row in per_dataset.values()]
    return {
        "numerator": numerator,
        "denominator": denominator,
        "equal_dataset_geomean_ratio": _geomean(dataset_ratios),
        "equal_dataset_pct": (_geomean(dataset_ratios) - 1.0) * 100.0,
        "worst_dataset_ratio": float(max(dataset_ratios)),
        "worst_split_ratio": float(
            max(max(row["split_ratios"]) for row in per_dataset.values())
        ),
        "dataset_wins": int(np.count_nonzero(np.asarray(dataset_ratios) < 1)),
        "dataset_losses": int(np.count_nonzero(np.asarray(dataset_ratios) > 1)),
        "dataset_ties": int(np.count_nonzero(np.asarray(dataset_ratios) == 1)),
        "per_dataset": per_dataset,
        "_log_split_ratios": np.asarray(logs, dtype=np.float64),
    }


def _hierarchical_bootstrap_upper(log_split_ratios):
    logs = np.asarray(log_split_ratios, dtype=np.float64)
    if logs.shape != (25, 3):
        raise RuntimeError("T5 bootstrap requires 25 datasets x 3 folds")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    block = 2_000
    for start in range(0, BOOTSTRAP_REPLICATES, block):
        count = min(block, BOOTSTRAP_REPLICATES - start)
        datasets = rng.integers(0, 25, size=(count, 25))
        folds = rng.integers(0, 3, size=(count, 25, 3))
        selected = logs[datasets[..., None], folds]
        estimates[start : start + count] = np.exp(
            selected.mean(axis=(1, 2))
        )
    return {
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "one_sided_95_upper": float(np.quantile(estimates, 0.95)),
        "median": float(np.median(estimates)),
        "lower_5": float(np.quantile(estimates, 0.05)),
    }


def _cost_contrast(by_key, identity):
    per_dataset = {}
    for task_id in sorted(identity):
        dataset, lineage, stratum = identity[task_id]
        candidate = by_key[(task_id, runner.COMPOSITE)]
        control = by_key[(task_id, runner.CONTROL)]
        fit_ratio = float(
            candidate["summed_fit_seconds"] / control["summed_fit_seconds"]
        )
        predict_ratios = [
            float(a["prediction_timing"]["per_call_median_seconds"])
            / float(b["prediction_timing"]["per_call_median_seconds"])
            for a, b in zip(candidate["folds"], control["folds"])
        ]
        rss_ratio = float(
            candidate["peak_rss_bytes"] / control["peak_rss_bytes"]
        )
        per_dataset[lineage] = {
            "task_id": task_id,
            "dataset_name": dataset,
            "stratum": stratum,
            "fit_seconds_ratio": fit_ratio,
            "prediction_seconds_ratio": _geomean(predict_ratios),
            "peak_rss_ratio": rss_ratio,
        }
    return {
        "equal_dataset_fit_seconds_ratio": _geomean(
            [row["fit_seconds_ratio"] for row in per_dataset.values()]
        ),
        "worst_dataset_fit_seconds_ratio": float(
            max(row["fit_seconds_ratio"] for row in per_dataset.values())
        ),
        "equal_dataset_prediction_seconds_ratio": _geomean(
            [row["prediction_seconds_ratio"] for row in per_dataset.values()]
        ),
        "equal_dataset_peak_rss_ratio": _geomean(
            [row["peak_rss_ratio"] for row in per_dataset.values()]
        ),
        "per_dataset": per_dataset,
    }


def analyze(raw):
    snapshot = _dependency_snapshot()
    by_key, identity = _validate(raw, snapshot)
    primary = _quality_contrast(
        by_key, identity, runner.COMPOSITE, runner.CONTROL
    )
    bootstrap = _hierarchical_bootstrap_upper(
        primary.pop("_log_split_ratios")
    )
    dataset_logs = np.log(
        [row["ratio"] for row in primary["per_dataset"].values()]
    )
    loo = []
    lineages = list(primary["per_dataset"])
    for index, lineage in enumerate(lineages):
        ratio = float(
            np.exp(
                (dataset_logs.sum() - dataset_logs[index])
                / (len(dataset_logs) - 1)
            )
        )
        loo.append({"omitted_lineage": lineage, "ratio": ratio})
    least_favorable_loo = max(loo, key=lambda row: row["ratio"])
    cost = _cost_contrast(by_key, identity)
    comparisons = {}
    for name, denominator in (
        ("composite_over_chimeraboost", runner.CHIMERA),
        ("composite_over_catboost", runner.CATBOOST),
        ("control_over_chimeraboost", runner.CHIMERA),
        ("control_over_catboost", runner.CATBOOST),
    ):
        numerator = (
            runner.COMPOSITE if name.startswith("composite") else runner.CONTROL
        )
        contrast = _quality_contrast(by_key, identity, numerator, denominator)
        contrast.pop("_log_split_ratios")
        comparisons[name] = contrast
    gates = {
        "quality_ratio_at_most_0_995": (
            primary["equal_dataset_geomean_ratio"] <= QUALITY_BAR
        ),
        "bootstrap_upper_at_most_1_002": (
            bootstrap["one_sided_95_upper"] <= UNCERTAINTY_BAR
        ),
        "least_favorable_loo_at_most_0_998": (
            least_favorable_loo["ratio"] <= LOO_BAR
        ),
        "worst_dataset_at_most_1_005": (
            primary["worst_dataset_ratio"] <= HARM_BAR
        ),
        "fit_aggregate_at_most_6": (
            cost["equal_dataset_fit_seconds_ratio"] <= FIT_AGGREGATE_BAR
        ),
        "fit_worst_dataset_at_most_12": (
            cost["worst_dataset_fit_seconds_ratio"]
            <= FIT_WORST_DATASET_BAR
        ),
        "prediction_aggregate_at_most_1_5": (
            cost["equal_dataset_prediction_seconds_ratio"]
            <= PREDICT_AGGREGATE_BAR
        ),
        "rss_aggregate_at_most_2_5": (
            cost["equal_dataset_peak_rss_ratio"] <= RSS_AGGREGATE_BAR
        ),
        "complete_without_imputation_or_lockbox": True,
    }
    composite_folds = [
        fold
        for task_id in sorted(identity)
        for fold in by_key[(task_id, runner.COMPOSITE)]["folds"]
    ]
    engaged = sum(bool(fold["metadata"]["engaged"]) for fold in composite_folds)
    passes = all(gates.values())
    summary = {
        "schema_version": 1,
        "name": "darkofit_t5_composite_confirmation_summary_v1",
        "raw_artifact_sha256": raw["raw_artifact_sha256"],
        "analyzer_sha256": snapshot["analyzer_sha256"],
        "primary": primary,
        "hierarchical_bootstrap": bootstrap,
        "leave_one_out": loo,
        "least_favorable_leave_one_out": least_favorable_loo,
        "cost": cost,
        "competitive_comparisons": comparisons,
        "selection": {
            "coordinate_count": len(composite_folds),
            "engaged_count": engaged,
            "declined_count": len(composite_folds) - engaged,
            "exact_declines_verified": True,
        },
        "gates": gates,
        "passes_all_gates": passes,
        "decision": (
            "promote_t5_composite_automatic_policy"
            if passes
            else "close_t5_composite_candidate"
        ),
        "default_change_implemented": False,
        "lockbox_data_used": False,
    }
    summary["summary_sha256"] = _json_sha256(summary)
    return summary


def _markdown(summary):
    primary = summary["primary"]
    cost = summary["cost"]
    comp = summary["competitive_comparisons"]
    rows = [
        ("T5 / current default", primary["equal_dataset_geomean_ratio"]),
        (
            "T5 / ChimeraBoost 0.15.0",
            comp["composite_over_chimeraboost"][
                "equal_dataset_geomean_ratio"
            ],
        ),
        (
            "T5 / CatBoost 1.2.10",
            comp["composite_over_catboost"]["equal_dataset_geomean_ratio"],
        ),
    ]
    table = "\n".join(
        f"| {name} | {ratio:.6f} | {(ratio - 1) * 100:+.3f}% |"
        for name, ratio in rows
    )
    gates = "\n".join(
        f"- [{'x' if passed else ' '}] `{name}`"
        for name in GATE_NAMES
        for passed in (summary["gates"][name],)
    )
    return f"""# T5 composite confirmation result

**Decision: `{summary['decision']}`.**

All 25 outcome-unseen lineages and 75 frozen coordinates completed. The
selection-guarded candidate engaged on
{summary['selection']['engaged_count']} coordinates and declined exactly on
{summary['selection']['declined_count']}.

| Contrast | Equal-dataset RMSE ratio | Difference |
|---|---:|---:|
{table}

The one-sided hierarchical 95% upper bound is
`{summary['hierarchical_bootstrap']['one_sided_95_upper']:.6f}`; the
least-favorable leave-one-out ratio is
`{summary['least_favorable_leave_one_out']['ratio']:.6f}`; and the worst
dataset ratio is `{primary['worst_dataset_ratio']:.6f}`.

Cost versus current default: fit `{cost['equal_dataset_fit_seconds_ratio']:.3f}x`
(worst dataset `{cost['worst_dataset_fit_seconds_ratio']:.3f}x`), prediction
`{cost['equal_dataset_prediction_seconds_ratio']:.3f}x`, and peak RSS
`{cost['equal_dataset_peak_rss_ratio']:.3f}x`.

## Frozen gates

{gates}

Competitive comparisons are descriptive. No earlier lockbox was opened, no
task was dropped or imputed, and no default changes until the recorded
decision is implemented and verified.
"""


def _atomic_create(path, text):
    encoded = text.encode()
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing existing output: {path}")
    message = "refusing symlink T5 output directory"
    _reject_symlink_directory(
        path.parent, message
    )
    owned_directories = _create_owned_directories(path.parent, message)
    temporary = None
    published_identity = None
    try:
        _reject_symlink_directory(path.parent, message)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            identity = os.fstat(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(f"refusing existing output: {path}") from error
        published_identity = (identity.st_dev, identity.st_ino)
        _verify_published_identity(
            path,
            published_identity,
            "T5 output publish identity changed",
        )
    except BaseException:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
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
        temporary.unlink(missing_ok=True)
    except BaseException:
        try:
            _unlink_if_owned(path, published_identity)
        except OSError:
            pass
        _remove_owned_directories(owned_directories)
        raise
    return published_identity, owned_directories


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


def _atomic_create_pair(first_path, first_text, second_path, second_text):
    for path in (first_path, second_path):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing existing output: {path}")
    created = []
    try:
        created.append(
            (first_path, _atomic_create(first_path, first_text))
        )
        created.append(
            (second_path, _atomic_create(second_path, second_text))
        )
    except BaseException:
        for path, (identity, owned_directories) in reversed(created):
            try:
                _unlink_if_owned(path, identity)
            except OSError:
                pass
            _remove_owned_directories(owned_directories)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args(argv)
    args.input = Path(os.path.abspath(os.path.expanduser(args.input)))
    args.output = Path(os.path.abspath(os.path.expanduser(args.output)))
    args.markdown = Path(os.path.abspath(os.path.expanduser(args.markdown)))
    return args


def main(argv=None):
    args = parse_args(argv)
    raw = _json_loads(args.input.read_bytes(), "T5 raw artifact")
    summary = analyze(raw)
    _atomic_create_pair(
        args.output,
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        args.markdown,
        _markdown(summary),
    )
    print(summary["decision"])
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
