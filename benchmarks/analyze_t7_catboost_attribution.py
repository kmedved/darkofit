#!/usr/bin/env python3
"""Analyze the T7 CatBoost attribution development run."""

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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_native_ordinal_c2 as c2  # noqa: E402
from benchmarks import run_t7_catboost_attribution as runner  # noqa: E402


DEFAULT_INPUT = runner.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT / "benchmarks" / "t7_catboost_attribution_summary.json"
DEFAULT_MARKDOWN = ROOT / "benchmarks" / "t7_catboost_attribution_result.md"
ELIGIBLE = (
    "ordered",
    "border_128",
    "leaf10_no_backtracking",
    "leaf10_any_improvement",
    "ctr_complexity_2",
    "depth_by_n_p",
)
CONTRAST_NAMES = (
    "ordered_over_plain",
    "plain_over_default",
    "border_128_over_default",
    "leaf10_no_backtracking_over_default",
    "backtracking_over_no_backtracking",
    "ctr_complexity_2_over_default",
    "depth_4_over_default",
    "depth_8_over_default",
    "depth_by_n_p_over_default",
)
LINEAGE_ORDER = (
    "auction_verification",
    "video_transcoding",
    "fps_benchmark",
    "cars_ctr23",
    "diamonds",
    "book_price",
    "apparel_price",
    "munich_rent",
)
FROZEN_RUNNER_SHA256 = (
    "be1178f8593d3ff52a19963812932b399fbfbc3fd1942b97ad663ee9fe728a49"
)
FROZEN_C2_HELPER_SHA256 = (
    "8da023ee1c6ab1311d0b8b152c8bcd82f80d6f323020efb4e86c71870caa8952"
)
FROZEN_PROTOCOL_SHA256 = (
    "18200d9bd8f6b43ec345be5755ce795f6284ae399a43eeae0144cd860718f460"
)
FROZEN_SOURCE_HEAD = "027bfdb29caf5b1320aac4be76aab950c4e0da15"
FROZEN_SOURCE_BRANCH = "codex/product-offense"
FROZEN_RAW_CANONICAL_SHA256 = (
    "6673fe69c5e09d1e020252237c322e7795c14effdf375e9dd1c0db3ecc4772ee"
)
FROZEN_RAW_FILE_SHA256 = (
    "cf199793c5e3349ee4a8e3575870f9cacec2905e54b84fc4bcf2703a70cb518f"
)
RESOLVED_PARAM_KEYS = (
    "boosting_type",
    "border_count",
    "leaf_estimation_iterations",
    "leaf_estimation_backtracking",
    "max_ctr_complexity",
    "depth",
    "grow_policy",
    "learning_rate",
    "iterations",
    "random_seed",
)
RAW_FIELDS = frozenset(
    {
        "schema_version",
        "name",
        "created_at",
        "source",
        "runtime",
        "protocol",
        "task_count",
        "coordinate_count",
        "fit_count",
        "spool_records",
        "results",
        "development_data_only",
        "confirmation_outcomes_inspected",
        "lockbox_data_used",
        "default_change_authorized",
        "raw_sha256",
    }
)
RESULT_FIELDS = frozenset(
    {
        "task_id",
        "dataset_id",
        "dataset_name",
        "lineage_cluster",
        "fold",
        "coordinate_index",
        "n_features",
        "categorical_count",
        "outer_split",
        "inner_split",
        "arm_order",
        "arms",
        "warmup_seconds",
        "peak_rss_bytes",
        "behavior_sha256",
        "worker_stdout",
        "worker_stderr",
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
        "remote_branch_head",
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


def _json_sha256(value):
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
    c2_bytes = Path(c2.__file__).resolve().read_bytes()
    protocol_bytes = runner.PROTOCOL.read_bytes()
    registry_bytes = runner.REGISTRY.read_bytes()
    c2_raw_bytes = runner.C2_RAW.read_bytes()
    return {
        "analyzer_sha256": hashlib.sha256(analyzer_bytes).hexdigest(),
        "runner_sha256": hashlib.sha256(runner_bytes).hexdigest(),
        "c2_helper_sha256": hashlib.sha256(c2_bytes).hexdigest(),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "registry_bytes": registry_bytes,
        "c2_raw_bytes": c2_raw_bytes,
    }


def _geomean(values):
    values = np.asarray(values, dtype=np.float64)
    if (
        values.ndim != 1
        or not values.size
        or np.any(values <= 0)
        or not np.all(np.isfinite(values))
    ):
        raise RuntimeError("T7 geometric mean received invalid values")
    return float(np.exp(np.mean(np.log(values))))


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_int(value):
    return type(value) is int


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
        and isinstance(source.get("remote_branch_head"), str)
    )


def _valid_runtime(runtime):
    machine = runtime.get("machine") if isinstance(runtime, dict) else None
    dependencies = (
        runtime.get("dependencies") if isinstance(runtime, dict) else None
    )
    return (
        isinstance(runtime, dict)
        and set(runtime) == {"python", "machine", "dependencies"}
        and isinstance(runtime.get("python"), str)
        and bool(runtime["python"])
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


def _is_positive_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _is_positive_float(value):
    return type(value) is float and math.isfinite(value) and value > 0


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


def _resolved_param_types_are_valid(params):
    return (
        isinstance(params.get("boosting_type"), str)
        and _is_int(params.get("border_count"))
        and _is_int(params.get("leaf_estimation_iterations"))
        and isinstance(params.get("leaf_estimation_backtracking"), str)
        and (
            params.get("max_ctr_complexity") is None
            or _is_int(params.get("max_ctr_complexity"))
        )
        and _is_int(params.get("depth"))
        and isinstance(params.get("grow_policy"), str)
        and _is_positive_float(params.get("learning_rate"))
        and _is_int(params.get("iterations"))
        and _is_int(params.get("random_seed"))
    )


def depth_policy_arm(fit_rows, n_features):
    if (
        not _is_int(fit_rows)
        or fit_rows <= 0
        or not _is_int(n_features)
        or n_features <= 0
    ):
        raise ValueError("T7 depth policy requires positive integer dimensions")
    density = fit_rows / n_features
    if density < 100:
        return "depth_4"
    if density >= 2_500:
        return "depth_8"
    return "default"


def _behavior(result):
    return {
        "task_id": result["task_id"],
        "fold": result["fold"],
        "arms": [
            {
                "arm": arm["arm"],
                "validation": arm["validation"],
                "test": arm["test"],
                "resolved_params": arm["resolved_params"],
            }
            for arm in result["arms"]
        ],
    }


def _validate(raw, snapshot=None):
    if not isinstance(raw, dict):
        raise RuntimeError("T7 raw artifact is not an object")
    if (
        set(raw) != RAW_FIELDS
        or not _is_int(raw.get("schema_version"))
        or raw.get("schema_version") != 1
        or raw.get("name") != "darkofit_t7_catboost_attribution_raw_v1"
    ):
        raise RuntimeError("T7 raw artifact name changed")
    expected_hash = raw.get("raw_sha256")
    if not _is_sha256(expected_hash):
        raise RuntimeError("T7 raw artifact hash changed")
    unhashed = dict(raw)
    unhashed.pop("raw_sha256")
    try:
        observed_hash = runner._json_sha256(unhashed)
    except (TypeError, ValueError) as error:
        raise RuntimeError("T7 raw artifact hash changed") from error
    if expected_hash != observed_hash:
        raise RuntimeError("T7 raw artifact hash changed")
    if snapshot is None:
        snapshot = _dependency_snapshot()
    if snapshot.get("c2_helper_sha256") != FROZEN_C2_HELPER_SHA256:
        raise RuntimeError("T7 frozen C2 split helper changed")
    try:
        created_at = datetime.fromisoformat(raw["created_at"])
    except (TypeError, ValueError) as error:
        raise RuntimeError("T7 raw creation timestamp changed") from error
    protocol = raw.get("protocol")
    source = raw.get("source")
    runtime = raw.get("runtime")
    if (
        not isinstance(protocol, dict)
        or set(protocol)
        != {
            "runner_sha256",
            "protocol_sha256",
            "registry_sha256",
            "c2_raw_sha256",
            "source_head",
            "catboost_version",
            "arms",
            "folds",
        }
        or not _valid_source(source)
        or not _valid_runtime(runtime)
        or created_at.tzinfo is None
        or created_at.utcoffset() != timezone.utc.utcoffset(created_at)
        or not isinstance(protocol.get("arms"), list)
        or not isinstance(protocol.get("folds"), list)
        or not isinstance(source.get("status"), list)
    ):
        raise RuntimeError("T7 raw protocol structure changed")
    if (
        raw.get("development_data_only") is not True
        or raw.get("confirmation_outcomes_inspected") is not False
        or raw.get("lockbox_data_used") is not False
        or raw.get("default_change_authorized") is not False
        or protocol.get("runner_sha256") != FROZEN_RUNNER_SHA256
        or protocol.get("protocol_sha256") != FROZEN_PROTOCOL_SHA256
        or protocol.get("protocol_sha256") != snapshot["protocol_sha256"]
        or protocol.get("registry_sha256")
        != runner.EXPECTED_REGISTRY_SHA256
        or protocol.get("c2_raw_sha256") != runner.EXPECTED_C2_RAW_SHA256
        or protocol.get("source_head") != FROZEN_SOURCE_HEAD
        or protocol.get("catboost_version") != "1.2.10"
        or tuple(protocol["arms"]) != runner.ARM_NAMES
        or tuple(protocol["folds"]) != runner.FOLDS
        or source.get("head") != FROZEN_SOURCE_HEAD
        or source.get("remote_branch_head") != FROZEN_SOURCE_HEAD
        or source.get("branch") != FROZEN_SOURCE_BRANCH
        or source.get("clean") is not True
        or source.get("status") != []
        or not _is_int(raw.get("task_count"))
        or raw["task_count"] != 8
        or not _is_int(raw.get("coordinate_count"))
        or raw["coordinate_count"] != 24
        or not _is_int(raw.get("fit_count"))
        or raw["fit_count"] != 216
    ):
        raise RuntimeError("T7 raw protocol changed")
    if (
        any(not isinstance(arm, str) for arm in protocol["arms"])
        or any(not _is_int(fold) for fold in protocol["folds"])
    ):
        raise RuntimeError("T7 raw protocol changed")
    registry_bytes = snapshot["registry_bytes"]
    c2_raw_bytes = snapshot["c2_raw_bytes"]
    registry, rows = runner._rows(registry_bytes, c2_raw_bytes)
    expected = {
        (int(row["task_id"]), fold)
        for row in registry["development_tasks"]
        for fold in runner.FOLDS
    }
    expected_order = {
        coordinate: index
        for index, coordinate in enumerate(
            (
                (int(row["task_id"]), fold)
                for row in registry["development_tasks"]
                for fold in runner.FOLDS
            )
        )
    }
    c2_raw = _json_loads(c2_raw_bytes, "T7 C2 raw artifact")
    c2_controls = {}
    for c2_result in c2_raw["results"]:
        if c2_result["arm"] != "control":
            continue
        key = (int(c2_result["task_id"]), int(c2_result["fold"]))
        if key in c2_controls:
            raise RuntimeError(f"T7 duplicate C2 control coordinate: {key}")
        c2_controls[key] = c2_result
    if set(c2_controls) != expected:
        raise RuntimeError("T7 C2 control matrix changed")
    results = raw.get("results")
    if not isinstance(results, list) or any(
        not isinstance(row, dict)
        or set(row) != RESULT_FIELDS
        or not _is_int(row.get("task_id"))
        or not _is_int(row.get("fold"))
        or not _is_int(row.get("coordinate_index"))
        for row in results
    ):
        raise RuntimeError("T7 coordinate matrix is not a list")
    observed = {
        (row["task_id"], row["fold"]) for row in results
    }
    if (
        len(results) != 24
        or observed != expected
        or [row["coordinate_index"] for row in results]
        != list(range(len(expected)))
    ):
        raise RuntimeError("T7 coordinate matrix is incomplete")
    by_coordinate = {}
    for result in results:
        key = (result["task_id"], result["fold"])
        if key in by_coordinate:
            raise RuntimeError(f"T7 duplicate coordinate: {key}")
        row = rows[key[0]]
        coordinate_index = expected_order[key]
        expected_split = c2._expected_outer_split(row, key[1])
        outer = result.get("outer_split")
        inner = result.get("inner_split")
        if (
            not isinstance(outer, dict)
            or set(outer)
            != {
                "train_size",
                "test_size",
                "train_index_sha256",
                "test_index_sha256",
            }
            or not isinstance(inner, dict)
            or not isinstance(result.get("arm_order"), list)
            or not isinstance(result.get("arms"), list)
        ):
            raise RuntimeError(f"T7 coordinate structure changed: {key}")
        expected_categorical = row["feature_record"]["categorical_indices"]
        expected_validation_rows = max(
            1,
            int(
                math.ceil(
                    c2.VALIDATION_FRACTION * int(outer["train_size"])
                )
            ),
        )
        if (
            not _is_int(result.get("dataset_id"))
            or result["dataset_id"] != int(row["dataset_id"])
            or result.get("dataset_name") != row["dataset_name"]
            or result.get("lineage_cluster") != row["lineage_cluster"]
            or result["coordinate_index"] != coordinate_index
            or not _is_int(result.get("n_features"))
            or result["n_features"] <= 0
            or result["n_features"]
            != row["task_record"]["fingerprint"]["n_features"]
            or not _is_int(result.get("categorical_count"))
            or result["categorical_count"] < 0
            or result["categorical_count"] != len(expected_categorical)
            or not _is_int(outer.get("train_size"))
            or outer["train_size"] != expected_split["train_size"]
            or not _is_int(outer.get("test_size"))
            or outer["test_size"] != expected_split["test_size"]
            or outer.get("train_index_sha256")
            != expected_split["train_index_sha256"]
            or outer.get("test_index_sha256")
            != expected_split["test_index_sha256"]
            or tuple(result["arm_order"])
            != runner._arm_order(coordinate_index)
            or inner.get("policy") != "seeded_permutation_tail_20_percent"
            or not _is_int(inner.get("seed"))
            or inner["seed"] != c2.SPLIT_SEED + key[0] + key[1]
            or not isinstance(
                inner.get("validation_fraction"), float
            )
            or inner["validation_fraction"] != c2.VALIDATION_FRACTION
            or not _is_int(inner.get("validation_rows"))
            or inner["validation_rows"] != expected_validation_rows
            or not _is_int(inner.get("fit_rows"))
            or inner["fit_rows"]
            != outer["train_size"] - expected_validation_rows
            or inner != c2_controls[key]["inner_split"]
            or not _is_sha256(inner.get("fit_index_sha256"))
            or not _is_sha256(inner.get("validation_index_sha256"))
            or not _is_sha256(result.get("behavior_sha256"))
            or not _is_positive_float(result.get("warmup_seconds"))
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
            raise RuntimeError(f"T7 coordinate binding changed: {key}")
        if (
            any(
                not isinstance(arm, dict)
                or set(arm)
                != {
                    "arm",
                    "position",
                    "overrides",
                    "fit_seconds",
                    "validation",
                    "test",
                    "prediction_timing",
                    "tree_count",
                    "resolved_params",
                }
                or not isinstance(arm.get("arm"), str)
                for arm in result["arms"]
            )
            or [arm["arm"] for arm in result["arms"]]
            != result["arm_order"]
        ):
            raise RuntimeError(f"T7 arm execution order changed: {key}")
        if result["behavior_sha256"] != runner._json_sha256(
            _behavior(result)
        ):
            raise RuntimeError(f"T7 coordinate behavior changed: {key}")
        arms = {arm["arm"]: arm for arm in result["arms"]}
        if len(arms) != len(runner.ARM_NAMES) or set(arms) != set(
            runner.ARM_NAMES
        ):
            raise RuntimeError(f"T7 arm matrix changed: {key}")
        baseline_params = arms["default"].get("resolved_params")
        expected_baseline = {
            "boosting_type": "Plain",
            "border_count": 254,
            "leaf_estimation_iterations": 1,
            "leaf_estimation_backtracking": "AnyImprovement",
            "max_ctr_complexity": 4 if result["categorical_count"] else None,
            "depth": 6,
            "grow_policy": "SymmetricTree",
            "learning_rate": baseline_params.get("learning_rate")
            if isinstance(baseline_params, dict)
            else None,
            "iterations": 1000,
            "random_seed": runner.RANDOM_STATE,
        }
        if (
            not isinstance(baseline_params, dict)
            or set(baseline_params) != set(RESOLVED_PARAM_KEYS)
            or not _resolved_param_types_are_valid(baseline_params)
            or baseline_params != expected_baseline
        ):
            raise RuntimeError(f"T7 default resolution changed: {key}")
        for position, arm_name in enumerate(result["arm_order"]):
            arm = arms[arm_name]
            resolved = arm.get("resolved_params")
            if (
                not _is_int(arm.get("position"))
                or arm["position"] != position
                or not _same_typed_value(
                    arm.get("overrides"), runner.ARMS[arm_name]
                )
                or not isinstance(resolved, dict)
                or set(resolved) != set(RESOLVED_PARAM_KEYS)
                or not _resolved_param_types_are_valid(resolved)
                or not _is_int(arm.get("tree_count"))
                or arm["tree_count"] < 1
                or arm["tree_count"] != resolved["iterations"]
            ):
                raise RuntimeError(f"T7 arm binding changed: {key}/{arm_name}")
            expected_resolved = dict(baseline_params)
            for parameter, expected_value in runner.ARMS[arm_name].items():
                if (
                    parameter == "max_ctr_complexity"
                    and not result["categorical_count"]
                    and resolved[parameter] is None
                ):
                    continue
                expected_resolved[parameter] = expected_value
            if arm_name.startswith("leaf10_"):
                expected_resolved["learning_rate"] = resolved["learning_rate"]
            if resolved != expected_resolved:
                raise RuntimeError(
                    f"T7 resolved arm changed: {key}/{arm_name}"
                )
        leaf_learning_rates = {
            arms[arm_name]["resolved_params"]["learning_rate"]
            for arm_name in (
                "leaf10_no_backtracking",
                "leaf10_any_improvement",
            )
        }
        if len(leaf_learning_rates) != 1:
            raise RuntimeError(f"T7 leaf-arm learning rates changed: {key}")
        for arm_name in result["arm_order"]:
            arm = arms[arm_name]
            for split in ("validation", "test"):
                metric = arm.get(split)
                expected_rows = (
                    inner["validation_rows"]
                    if split == "validation"
                    else outer["test_size"]
                )
                if (
                    not isinstance(metric, dict)
                    or set(metric) != {"rows", "rmse", "prediction_sha256"}
                    or not _is_int(metric.get("rows"))
                    or metric["rows"] != expected_rows
                    or not _is_positive_float(metric.get("rmse"))
                    or not _is_sha256(metric["prediction_sha256"])
                ):
                    raise RuntimeError(f"T7 metric changed: {key}")
            timing = arm.get("prediction_timing")
            if (
                not _is_positive_float(arm.get("fit_seconds"))
                or not isinstance(timing, dict)
                or set(timing) != {"calls", "median_seconds", "total_seconds"}
                or not _is_int(timing.get("calls"))
                or timing["calls"] != runner.PREDICTION_CALLS
                or not _is_positive_float(timing.get("median_seconds"))
                or not _is_positive_float(timing.get("total_seconds"))
                or timing["total_seconds"]
                < (
                    (timing["calls"] + 1) // 2
                    * timing["median_seconds"]
                )
            ):
                raise RuntimeError(f"T7 timing changed: {key}")
        equivalent_behaviors = {}
        for arm_name in result["arm_order"]:
            arm = arms[arm_name]
            parameter_sha256 = runner._json_sha256(
                arm["resolved_params"]
            )
            behavior = {
                "tree_count": arm["tree_count"],
                "validation": arm["validation"],
                "test": arm["test"],
            }
            previous = equivalent_behaviors.setdefault(
                parameter_sha256, behavior
            )
            if previous != behavior:
                raise RuntimeError(
                    f"T7 equivalent arm behavior changed: {key}/{arm_name}"
                )
        by_coordinate[key] = {
            "result": result,
            "arms": arms,
        }
    spool_records = raw.get("spool_records")
    if not isinstance(spool_records, list) or any(
        not isinstance(row, dict)
        or set(row) != {"task_id", "fold", "sha256", "resumed"}
        or not _is_int(row.get("task_id"))
        or not _is_int(row.get("fold"))
        for row in spool_records
    ):
        raise RuntimeError("T7 spool matrix is not a list")
    spool_coordinates = {
        (row["task_id"], row["fold"]) for row in spool_records
    }
    expected_spool_hashes = {}
    for key, coordinate in by_coordinate.items():
        result = coordinate["result"]
        payload = {
            "binding": protocol,
            "task_id": result["task_id"],
            "fold": result["fold"],
            "result_sha256": runner._json_sha256(result),
            "result": result,
        }
        expected_spool_hashes[key] = runner._json_sha256(payload)
    if (
        len(spool_records) != len(expected)
        or len(spool_coordinates) != len(expected)
        or spool_coordinates != expected
        or any(
            not _is_sha256(row.get("sha256"))
            or not isinstance(row.get("resumed"), bool)
            or row["sha256"]
            != expected_spool_hashes[
                (row["task_id"], row["fold"])
            ]
            for row in spool_records
        )
        or len({row["sha256"] for row in spool_records}) != len(expected)
    ):
        raise RuntimeError("T7 spool matrix changed")
    if raw["raw_sha256"] != FROZEN_RAW_CANONICAL_SHA256:
        raise RuntimeError("T7 frozen raw content changed")
    return by_coordinate, rows, c2_controls


def _selected_arm(coordinate, name):
    if name == "depth_by_n_p":
        result = coordinate["result"]
        return depth_policy_arm(
            result["inner_split"]["fit_rows"],
            result["n_features"],
        )
    return name


def _contrast(by_coordinate, rows, numerator, denominator="default"):
    per_task = {}
    for task_id, row in rows.items():
        test_ratios = []
        validation_ratios = []
        selected_arms = []
        for fold in runner.FOLDS:
            coordinate = by_coordinate[(task_id, fold)]
            top_name = _selected_arm(coordinate, numerator)
            bottom_name = _selected_arm(coordinate, denominator)
            top = coordinate["arms"][top_name]
            bottom = coordinate["arms"][bottom_name]
            test_ratios.append(
                float(top["test"]["rmse"]) / float(bottom["test"]["rmse"])
            )
            validation_ratios.append(
                float(top["validation"]["rmse"])
                / float(bottom["validation"]["rmse"])
            )
            selected_arms.append(top_name)
        per_task[row["lineage_cluster"]] = {
            "task_id": task_id,
            "dataset_name": row["dataset_name"],
            "test_ratio": _geomean(test_ratios),
            "validation_ratio": _geomean(validation_ratios),
            "split_test_ratios": test_ratios,
            "selected_arms": selected_arms,
        }
    task_ratios = [row["test_ratio"] for row in per_task.values()]
    validation_ratios = [
        row["validation_ratio"] for row in per_task.values()
    ]
    logs = np.log(task_ratios)
    loo = [
        {
            "omitted_lineage": lineage,
            "ratio": float(
                np.exp((logs.sum() - logs[index]) / (len(logs) - 1))
            ),
        }
        for index, lineage in enumerate(per_task)
    ]
    return {
        "numerator": numerator,
        "denominator": denominator,
        "equal_dataset_test_ratio": _geomean(task_ratios),
        "equal_dataset_validation_ratio": _geomean(validation_ratios),
        "worst_task_test_ratio": float(max(task_ratios)),
        "worst_split_test_ratio": float(
            max(
                max(row["split_test_ratios"])
                for row in per_task.values()
            )
        ),
        "least_favorable_loo_test_ratio": float(
            max(row["ratio"] for row in loo)
        ),
        "wins": int(np.count_nonzero(np.asarray(task_ratios) < 1)),
        "losses": int(np.count_nonzero(np.asarray(task_ratios) > 1)),
        "ties": int(np.count_nonzero(np.asarray(task_ratios) == 1)),
        "leave_one_out": loo,
        "per_task": per_task,
    }


def _darkofit_anchor(by_coordinate, rows, controls, arm):
    if len(controls) != 24:
        raise RuntimeError("T7 DarkoFit anchor matrix changed")
    per_task = {}
    for task_id, row in rows.items():
        ratios = []
        for fold in runner.FOLDS:
            coordinate = by_coordinate[(task_id, fold)]
            arm_name = _selected_arm(coordinate, arm)
            cat_rmse = coordinate["arms"][arm_name]["test"]["rmse"]
            darko_rmse = controls[(task_id, fold)]["test"]["rmse"]
            ratios.append(float(darko_rmse) / float(cat_rmse))
        per_task[row["lineage_cluster"]] = {
            "task_id": task_id,
            "dataset_name": row["dataset_name"],
            "darkofit_over_catboost_ratio": _geomean(ratios),
            "split_ratios": ratios,
        }
    ratios = [
        row["darkofit_over_catboost_ratio"] for row in per_task.values()
    ]
    return {
        "catboost_arm": arm,
        "equal_dataset_darkofit_over_catboost_ratio": _geomean(ratios),
        "darkofit_wins": int(np.count_nonzero(np.asarray(ratios) < 1)),
        "darkofit_losses": int(np.count_nonzero(np.asarray(ratios) > 1)),
        "per_task": per_task,
        "evidence_scope": (
            "immutable C2 DarkoFit control; descriptive, not current-release "
            "confirmation"
        ),
    }


def analyze(raw):
    snapshot = _dependency_snapshot()
    by_coordinate, rows, c2_controls = _validate(raw, snapshot)
    contrasts = {
        "ordered_over_plain": _contrast(
            by_coordinate, rows, "ordered", "plain"
        ),
        "plain_over_default": _contrast(
            by_coordinate, rows, "plain", "default"
        ),
        "border_128_over_default": _contrast(
            by_coordinate, rows, "border_128"
        ),
        "leaf10_no_backtracking_over_default": _contrast(
            by_coordinate, rows, "leaf10_no_backtracking"
        ),
        "backtracking_over_no_backtracking": _contrast(
            by_coordinate,
            rows,
            "leaf10_any_improvement",
            "leaf10_no_backtracking",
        ),
        "ctr_complexity_2_over_default": _contrast(
            by_coordinate, rows, "ctr_complexity_2"
        ),
        "depth_4_over_default": _contrast(
            by_coordinate, rows, "depth_4"
        ),
        "depth_8_over_default": _contrast(
            by_coordinate, rows, "depth_8"
        ),
        "depth_by_n_p_over_default": _contrast(
            by_coordinate, rows, "depth_by_n_p"
        ),
    }
    candidate_contrasts = {
        name: _contrast(by_coordinate, rows, name)
        for name in ELIGIBLE
    }
    nominations = []
    for name, contrast in candidate_contrasts.items():
        gates = {
            "test_ratio_at_most_0_995": (
                contrast["equal_dataset_test_ratio"] <= 0.995
            ),
            "validation_ratio_at_most_1_005": (
                contrast["equal_dataset_validation_ratio"] <= 1.005
            ),
            "worst_task_at_most_1_02": (
                contrast["worst_task_test_ratio"] <= 1.02
            ),
            "least_favorable_loo_at_most_1": (
                contrast["least_favorable_loo_test_ratio"] <= 1.0
            ),
        }
        nominations.append(
            {
                "candidate": name,
                "passes": all(gates.values()),
                "gates": gates,
                "test_ratio": contrast["equal_dataset_test_ratio"],
            }
        )
    survivors = sorted(
        (row for row in nominations if row["passes"]),
        key=lambda row: (row["test_ratio"], row["candidate"]),
    )[:3]
    anchor_arms = (*runner.ARM_NAMES, "depth_by_n_p")
    all_anchor_details = {
        arm: _darkofit_anchor(
            by_coordinate, rows, c2_controls, arm
        )
        for arm in anchor_arms
    }
    anchors = {
        arm: all_anchor_details[arm]
        for arm in ("default", *[row["candidate"] for row in survivors])
    }
    anchor_ratio_key = "equal_dataset_darkofit_over_catboost_ratio"
    all_anchor_ratios = {
        arm: all_anchor_details[arm][anchor_ratio_key]
        for arm in anchor_arms
    }
    default_resolution = {}
    for task_id, row in rows.items():
        params = [
            by_coordinate[(task_id, fold)]["arms"]["default"][
                "resolved_params"
            ]
            for fold in runner.FOLDS
        ]
        default_resolution[row["lineage_cluster"]] = {
            "task_id": task_id,
            "dataset_name": row["dataset_name"],
            "resolved_params_by_fold": params,
        }
    summary = {
        "schema_version": 1,
        "name": "darkofit_t7_catboost_attribution_summary_v1",
        "raw_sha256": raw["raw_sha256"],
        "analyzer_sha256": snapshot["analyzer_sha256"],
        "analysis_evidence": {
            "raw_file_sha256": FROZEN_RAW_FILE_SHA256,
            "raw_canonical_sha256": FROZEN_RAW_CANONICAL_SHA256,
            "frozen_protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "frozen_runner_sha256": FROZEN_RUNNER_SHA256,
            "current_runner_sha256": snapshot["runner_sha256"],
            "frozen_c2_helper_sha256": FROZEN_C2_HELPER_SHA256,
            "current_c2_helper_sha256": snapshot["c2_helper_sha256"],
        },
        "contrasts": contrasts,
        "candidate_contrasts": candidate_contrasts,
        "candidate_gate_results": nominations,
        "frozen_research_candidates": [
            row["candidate"] for row in survivors
        ],
        "candidate_limit": 3,
        "darkofit_anchors": anchors,
        "darkofit_all_arm_anchor_ratios": all_anchor_ratios,
        "catboost_default_resolution": default_resolution,
        "development_data_only": True,
        "confirmation_outcomes_inspected": False,
        "lockbox_data_used": False,
        "default_change_authorized": False,
        "decision": (
            "freeze_t7_research_candidates"
            if survivors
            else "close_t7_without_candidates"
        ),
    }
    summary["summary_sha256"] = _json_sha256(summary)
    return summary


def _markdown(summary):
    rows = []
    for name in CONTRAST_NAMES:
        contrast = summary["contrasts"][name]
        rows.append(
            f"| `{name}` | {contrast['equal_dataset_test_ratio']:.6f} | "
            f"{contrast['equal_dataset_validation_ratio']:.6f} | "
            f"{contrast['worst_task_test_ratio']:.6f} | "
            f"{contrast['wins']}/{contrast['losses']}/"
            f"{contrast['ties']} |"
        )
    table = "\n".join(rows)
    candidates = summary["frozen_research_candidates"]
    candidate_text = (
        ", ".join(f"`{name}`" for name in candidates)
        if candidates
        else "none"
    )
    anchor = summary["darkofit_anchors"]["default"]
    candidate_rows = []
    for name in candidates:
        contrast = summary["candidate_contrasts"][name]
        for lineage in LINEAGE_ORDER:
            row = contrast["per_task"][lineage]
            candidate_rows.append(
                f"| `{name}` | {row['dataset_name']} | "
                f"{row['test_ratio']:.6f} | "
                f"{row['validation_ratio']:.6f} | "
                f"`{row['selected_arms'][0]}` |"
            )
    candidate_table = (
        "\n".join(candidate_rows)
        if candidate_rows
        else "| — | — | — | — | — |"
    )
    anchor_rows = "\n".join(
        f"| `{name}` | "
        f"{summary['darkofit_all_arm_anchor_ratios'][name]:.6f} |"
        for name in (*runner.ARM_NAMES, "depth_by_n_p")
    )
    return f"""# T7 CatBoost mechanism attribution

**Decision: `{summary['decision']}`.**

## Evidence bindings

- Frozen raw file SHA-256: `{summary['analysis_evidence']['raw_file_sha256']}`
- Frozen raw canonical SHA-256: `{summary['analysis_evidence']['raw_canonical_sha256']}`
- Frozen protocol SHA-256: `{summary['analysis_evidence']['frozen_protocol_sha256']}`
- Original run-time runner SHA-256: `{summary['analysis_evidence']['frozen_runner_sha256']}`
- Current hardened analyzer SHA-256: `{summary['analyzer_sha256']}`
- Current hardened runner SHA-256: `{summary['analysis_evidence']['current_runner_sha256']}`
- Frozen C2 split-helper SHA-256: `{summary['analysis_evidence']['frozen_c2_helper_sha256']}`
- Current C2 split-helper SHA-256: `{summary['analysis_evidence']['current_c2_helper_sha256']}`

| Contrast | Test ratio | Validation ratio | Worst task | W/L/T |
|---|---:|---:|---:|---:|
{table}

Frozen research candidates (maximum three): {candidate_text}.

## Surviving candidate by dataset

| Candidate | Dataset | Test ratio | Validation ratio | Selected arm |
|---|---|---:|---:|---|
{candidate_table}

The fixed depth policy uses depth 4 below 100 inner-fit rows per feature,
depth 8 at or above 2,500, and CatBoost's default depth 6 otherwise. It
declines exactly to the default on the five middle-density datasets.

## DarkoFit anchor

Against CatBoost's product default, the immutable C2 DarkoFit control has an
equal-dataset RMSE ratio of
`{anchor['equal_dataset_darkofit_over_catboost_ratio']:.6f}`.

| CatBoost arm | DarkoFit / CatBoost RMSE |
|---|---:|
{anchor_rows}

This is a descriptive historical anchor, not a current-release confirmation
claim. All nine measured CatBoost arms and the assembled depth policy are
reported. The surviving CatBoost depth policy widens rather than closes the
historical competitive gap; porting the rule to DarkoFit would require a
separate implementation and outcome-unseen evaluation.

All results use spent development tasks. No confirmation panel or lockbox was
opened, and no default change is authorized.
"""


def _atomic_create(path, value):
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing existing output: {path}")
    message = "refusing symlink T7 output directory"
    _reject_symlink_directory(path.parent, message)
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
            handle.write(value)
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
            "T7 analysis output publish identity changed",
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


def _atomic_create_pair(first_path, first_value, second_path, second_value):
    for path in (first_path, second_path):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing existing output: {path}")
    created = []
    try:
        created.append(
            (first_path, _atomic_create(first_path, first_value))
        )
        created.append(
            (second_path, _atomic_create(second_path, second_value))
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
    encoded = args.input.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != FROZEN_RAW_FILE_SHA256:
        raise RuntimeError("T7 frozen raw file changed")
    raw = _json_loads(encoded, "T7 raw artifact")
    summary = analyze(raw)
    _atomic_create_pair(
        args.output,
        (
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode(),
        args.markdown,
        _markdown(summary).encode(),
    )
    print(summary["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
