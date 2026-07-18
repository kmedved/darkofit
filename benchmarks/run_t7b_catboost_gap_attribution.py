#!/usr/bin/env python3
"""Run the prospective, spent-development T7b CatBoost-gap attribution."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_native_ordinal_c2 as c2  # noqa: E402
from benchmarks import run_t7_catboost_attribution as t7  # noqa: E402
from benchmarks.run_t5_composite_confirmation import (  # noqa: E402
    _catboost_frame,
)


ARMS = {
    "baseline": {},
    "random_strength_0": {"random_strength": 0},
    "bootstrap_no": {"bootstrap_type": "No"},
    "no_split_noise_or_row_sampling": {
        "random_strength": 0,
        "bootstrap_type": "No",
    },
    "l2_leaf_reg_1": {"l2_leaf_reg": 1},
    "one_hot_max_size_0": {"one_hot_max_size": 0},
    "one_hot_max_size_255": {"one_hot_max_size": 255},
    "leaf10_any_improvement": {
        "leaf_estimation_iterations": 10,
        "leaf_estimation_backtracking": "AnyImprovement",
    },
}
ARM_NAMES = tuple(ARMS)
SEEDS = (4, 17, 29)
FOLDS = (0, 1, 2)
THREADS_PER_WORKER = 6
CONCURRENT_WORKERS = 3
ITERATIONS = 1_000
CARS_TASK_ID = 361622
WORKER_PREFIX = "T7B_CATBOOST_GAP_ATTRIBUTION_RESULT="
REQUESTED_STATIC_PARAMS = {
    "loss_function": "RMSE",
    "iterations": ITERATIONS,
    "task_type": "CPU",
    "thread_count": THREADS_PER_WORKER,
    "verbose": False,
    "allow_writing_files": False,
}
FIT_POLICY = {
    "cat_features": "frozen_c2_indices_or_none",
    "eval_set": None,
    "early_stopping_rounds": None,
}
RESOLVED_KEYS = (
    "task_type",
    "loss_function",
    "eval_metric",
    "use_best_model",
    "eval_fraction",
    "thread_count",
    "boosting_type",
    "random_strength",
    "bootstrap_type",
    "bagging_temperature",
    "subsample",
    "l2_leaf_reg",
    "one_hot_max_size",
    "leaf_estimation_iterations",
    "leaf_estimation_backtracking",
    "depth",
    "grow_policy",
    "learning_rate",
    "iterations",
    "random_seed",
)

PROTOCOL = ROOT / "benchmarks" / "t7b_catboost_gap_attribution_protocol.md"
COORDINATES = (
    ROOT / "benchmarks" / "t7b_catboost_gap_attribution_coordinates.json"
)
FREEZE = ROOT / "benchmarks" / "t7b_catboost_gap_attribution_freeze.json"
ANALYZER = ROOT / "benchmarks" / "analyze_t7b_catboost_gap_attribution.py"
TEST_FILE = ROOT / "tests" / "test_t7b_catboost_gap_attribution.py"
T7_RAW = ROOT / "benchmarks" / "t7_catboost_attribution_raw.json"
T7_SUMMARY = ROOT / "benchmarks" / "t7_catboost_attribution_summary.json"
DEFAULT_OUTPUT = (
    ROOT / "benchmarks" / "t7b_catboost_gap_attribution_raw.json"
)
DEFAULT_SPOOL = ROOT / ".cache" / "t7b-catboost-gap-attribution-v1"
EXPECTED_ENVIRONMENT = {
    "python": {
        "implementation": "CPython",
        "version": "3.12.13",
    },
    "dependencies": {
        "numpy": "2.4.6",
        "pandas": "2.3.3",
        "scipy": "1.16.3",
        "scikit-learn": "1.7.2",
        "numba": "0.66.0",
        "openml": "0.15.1",
        "catboost": "1.2.10",
    },
}
ALLOWED_POST_MODEL_SOURCE_PATHS = frozenset(
    {FREEZE.relative_to(ROOT).as_posix()}
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _load_json(path, context):
    return t7._json_loads(path.read_bytes(), context)


def _load_json_bytes(encoded, context):
    return t7._json_loads(encoded, context)


def _is_hex(value, length):
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _runtime_environment():
    dependencies = {}
    for package in EXPECTED_ENVIRONMENT["dependencies"]:
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "dependencies": dependencies,
    }


def _validate_runtime_environment(expected):
    observed = _runtime_environment()
    if not t7._same_typed_value(observed, expected):
        raise RuntimeError(
            "T7b requires the exact frozen Python and dependency environment"
        )
    return observed


def _source_paths():
    return {
        "runner": Path(__file__).resolve(),
        "analyzer": ANALYZER,
        "protocol": PROTOCOL,
        "coordinates": COORDINATES,
        "tests": TEST_FILE,
        "t7_runner": Path(t7.__file__).resolve(),
        "t7_analyzer": (
            ROOT / "benchmarks" / "analyze_t7_catboost_attribution.py"
        ),
        "c2_runner": Path(c2.__file__).resolve(),
        "catboost_frame_provider": (
            ROOT / "benchmarks" / "run_t5_composite_confirmation.py"
        ),
        "basketball_harness": Path(t7.basketball.__file__).resolve(),
        "creator_harness": Path(t7.creator.__file__).resolve(),
        "c2_registry_builder": Path(c2.registry_builder.__file__).resolve(),
        "fingerprint_provider": Path(c2.ctr.__file__).resolve(),
    }


def _git_blob(path, revision):
    relative = path.resolve().relative_to(ROOT).as_posix()
    completed = subprocess.run(
        ["git", "show", f"{revision}:{relative}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"T7b frozen source is absent from model head: {relative}"
        )
    return completed.stdout


def _git_blob_sha256(path, revision):
    return hashlib.sha256(_git_blob(path, revision)).hexdigest()


def _source_freeze(*, verify_live=True):
    freeze = _load_json(FREEZE, "T7b source freeze")
    if (
        not isinstance(freeze, dict)
        or set(freeze)
        != {
            "schema_version",
            "name",
            "status",
            "catboost_version",
            "darkofit_model_head",
            "environment",
            "runtime",
            "source_sha256",
            "freeze_sha256",
        }
        or freeze["schema_version"] != 1
        or freeze["name"]
        != "darkofit_t7b_catboost_gap_attribution_freeze_v1"
        or freeze["status"] != "frozen_not_executed"
        or freeze["catboost_version"] != "1.2.10"
        or not _is_hex(freeze["darkofit_model_head"], 40)
        or not t7._same_typed_value(
            freeze["environment"], EXPECTED_ENVIRONMENT
        )
        or freeze["environment"]["dependencies"]["catboost"]
        != freeze["catboost_version"]
        or freeze["runtime"]
        != {
            "task_type": "CPU",
            "threads_per_worker": THREADS_PER_WORKER,
            "concurrent_workers": CONCURRENT_WORKERS,
            "iterations": ITERATIONS,
            "seeds": list(SEEDS),
            "arms": list(ARM_NAMES),
        }
        or set(freeze["source_sha256"]) != set(_source_paths())
        or any(
            not _is_hex(digest, 64)
            for digest in freeze["source_sha256"].values()
        )
    ):
        raise RuntimeError("T7b source freeze schema changed")
    canonical = dict(freeze)
    content_hash = canonical.pop("freeze_sha256")
    if content_hash != _json_sha256(canonical):
        raise RuntimeError("T7b source freeze hash changed")
    model_head = freeze["darkofit_model_head"]
    for name, path in _source_paths().items():
        expected = freeze["source_sha256"][name]
        if _git_blob_sha256(path, model_head) != expected:
            raise RuntimeError(f"T7b frozen git source changed: {name}")
        if verify_live and (
            not path.is_file() or _sha256(path) != expected
        ):
            raise RuntimeError(f"T7b live source changed: {name}")
    return freeze


def _t7_raw(encoded=None):
    if encoded is None:
        encoded = T7_RAW.read_bytes()
    raw = _load_json_bytes(encoded, "frozen T7 raw")
    raw_hash = raw.get("raw_sha256")
    canonical = dict(raw)
    canonical.pop("raw_sha256", None)
    if (
        type(raw_hash) is not str
        or raw_hash != _json_sha256(canonical)
        or raw.get("name") != "darkofit_t7_catboost_attribution_raw_v1"
        or raw.get("coordinate_count") != 24
        or not isinstance(raw.get("results"), list)
        or len(raw["results"]) != 24
    ):
        raise RuntimeError("T7b frozen T7 raw boundary changed")
    return raw


def _coordinates(*, source_head=None):
    if source_head is None:
        coordinate_bytes = COORDINATES.read_bytes()
        registry_bytes = t7.REGISTRY.read_bytes()
        c2_raw_bytes = t7.C2_RAW.read_bytes()
        t7_raw_bytes = T7_RAW.read_bytes()
        t7_summary_bytes = T7_SUMMARY.read_bytes()
    else:
        if not _is_hex(source_head, 40):
            raise RuntimeError("T7b historical coordinate head changed")
        coordinate_bytes = _git_blob(COORDINATES, source_head)
        registry_bytes = _git_blob(t7.REGISTRY, source_head)
        c2_raw_bytes = _git_blob(t7.C2_RAW, source_head)
        t7_raw_bytes = _git_blob(T7_RAW, source_head)
        t7_summary_bytes = _git_blob(T7_SUMMARY, source_head)
    declaration = _load_json_bytes(
        coordinate_bytes, "T7b coordinate declaration"
    )
    expected_keys = {
        "schema_version",
        "name",
        "development_data_only",
        "seeds",
        "arms",
        "input_boundary",
        "coordinate_count",
        "coordinates",
        "coordinates_sha256",
    }
    if (
        not isinstance(declaration, dict)
        or set(declaration) != expected_keys
        or declaration["schema_version"] != 1
        or declaration["name"]
        != "darkofit_t7b_catboost_gap_attribution_coordinates_v1"
        or declaration["development_data_only"] is not True
        or declaration["seeds"] != list(SEEDS)
        or declaration["arms"] != list(ARM_NAMES)
        or declaration["coordinate_count"] != 24
        or not isinstance(declaration["coordinates"], list)
        or len(declaration["coordinates"]) != 24
        or declaration["coordinates_sha256"]
        != _json_sha256(declaration["coordinates"])
    ):
        raise RuntimeError("T7b coordinate declaration changed")
    boundary = declaration["input_boundary"]
    if (
        not isinstance(boundary, dict)
        or set(boundary)
        != {
            "native_ordinal_c2_registry_sha256",
            "native_ordinal_c2_development_raw_sha256",
            "t7_raw_file_sha256",
            "t7_raw_canonical_sha256",
            "t7_summary_file_sha256",
            "historical_darkofit_over_catboost_default_ratio",
        }
        or boundary["native_ordinal_c2_registry_sha256"]
        != t7.EXPECTED_REGISTRY_SHA256
        or boundary["native_ordinal_c2_development_raw_sha256"]
        != t7.EXPECTED_C2_RAW_SHA256
        or hashlib.sha256(registry_bytes).hexdigest()
        != boundary["native_ordinal_c2_registry_sha256"]
        or hashlib.sha256(c2_raw_bytes).hexdigest()
        != boundary["native_ordinal_c2_development_raw_sha256"]
        or hashlib.sha256(t7_raw_bytes).hexdigest()
        != boundary["t7_raw_file_sha256"]
        or hashlib.sha256(t7_summary_bytes).hexdigest()
        != boundary["t7_summary_file_sha256"]
        or type(
            boundary["historical_darkofit_over_catboost_default_ratio"]
        )
        is not float
        or boundary["historical_darkofit_over_catboost_default_ratio"] <= 1
    ):
        raise RuntimeError("T7b input boundary changed")
    raw = _t7_raw(t7_raw_bytes)
    if raw["raw_sha256"] != boundary["t7_raw_canonical_sha256"]:
        raise RuntimeError("T7b frozen T7 canonical hash changed")
    summary = _load_json_bytes(t7_summary_bytes, "frozen T7 summary")
    summary_hash = summary.pop("summary_sha256", None)
    if (
        type(summary_hash) is not str
        or summary_hash != _json_sha256(summary)
        or summary.get("darkofit_all_arm_anchor_ratios", {}).get("default")
        != boundary["historical_darkofit_over_catboost_default_ratio"]
    ):
        raise RuntimeError("T7b historical CatBoost/DarkoFit gap changed")
    registry, _rows = t7._rows(registry_bytes, c2_raw_bytes)
    schedule = [
        (int(row["task_id"]), fold)
        for row in registry["development_tasks"]
        for fold in FOLDS
    ]
    raw_by_coordinate = {}
    for result in raw["results"]:
        key = result.get("coordinate_index")
        if type(key) is not int or key in raw_by_coordinate:
            raise RuntimeError("T7b frozen T7 coordinate identity changed")
        raw_by_coordinate[key] = result
    expected_coordinate_keys = {
        "coordinate_index",
        "task_id",
        "fold",
        "frozen_learning_rate",
        "seed4_validation_prediction_sha256",
        "seed4_test_prediction_sha256",
    }
    for index, coordinate in enumerate(declaration["coordinates"]):
        if (
            not isinstance(coordinate, dict)
            or set(coordinate) != expected_coordinate_keys
            or type(coordinate["coordinate_index"]) is not int
            or coordinate["coordinate_index"] != index
            or type(coordinate["task_id"]) is not int
            or type(coordinate["fold"]) is not int
            or (coordinate["task_id"], coordinate["fold"]) != schedule[index]
            or type(coordinate["frozen_learning_rate"]) is not float
            or not math.isfinite(coordinate["frozen_learning_rate"])
            or coordinate["frozen_learning_rate"] <= 0
        ):
            raise RuntimeError("T7b coordinate schedule changed")
        frozen = raw_by_coordinate.get(index)
        if (
            not isinstance(frozen, dict)
            or frozen.get("task_id") != coordinate["task_id"]
            or frozen.get("fold") != coordinate["fold"]
        ):
            raise RuntimeError("T7b coordinate/T7 identity changed")
        defaults = [
            arm for arm in frozen.get("arms", []) if arm.get("arm") == "default"
        ]
        if len(defaults) != 1:
            raise RuntimeError("T7b frozen T7 default changed")
        default = defaults[0]
        if (
            default.get("resolved_params", {}).get("learning_rate")
            != coordinate["frozen_learning_rate"]
            or default.get("validation", {}).get("prediction_sha256")
            != coordinate["seed4_validation_prediction_sha256"]
            or default.get("test", {}).get("prediction_sha256")
            != coordinate["seed4_test_prediction_sha256"]
        ):
            raise RuntimeError("T7b frozen T7 baseline changed")
    return declaration, raw_by_coordinate


def _schedule(coordinates):
    return [
        {
            "execution_index": coordinate["coordinate_index"] * len(SEEDS)
            + seed_index,
            "coordinate_index": coordinate["coordinate_index"],
            "task_id": coordinate["task_id"],
            "fold": coordinate["fold"],
            "seed": seed,
        }
        for coordinate in coordinates
        for seed_index, seed in enumerate(SEEDS)
    ]


def _arm_order(execution_index):
    if type(execution_index) is not int or execution_index < 0:
        raise ValueError("T7b execution index must be nonnegative")
    shift = execution_index % len(ARM_NAMES)
    return ARM_NAMES[shift:] + ARM_NAMES[:shift]


def _resolved_params(model):
    params = model.get_all_params()
    return {key: params.get(key) for key in RESOLVED_KEYS}


def _requested_policy(coordinate, seed, arm):
    constructor = dict(REQUESTED_STATIC_PARAMS)
    constructor.update(
        {
            "learning_rate": coordinate["frozen_learning_rate"],
            "random_seed": seed,
        }
    )
    constructor.update(ARMS[arm])
    return {
        "constructor_params": constructor,
        "fit_policy": FIT_POLICY,
    }


def _valid_optional_number(value, *, minimum=0.0, maximum=None):
    if value is None:
        return True
    if (
        type(value) not in (int, float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        return False
    return maximum is None or float(value) <= maximum


def _validate_resolved(
    params, coordinate, seed, arm=None, *, categorical_count=None
):
    if (
        not isinstance(params, dict)
        or set(params) != set(RESOLVED_KEYS)
        or params.get("task_type") != "CPU"
        or params.get("loss_function") != "RMSE"
        or params.get("eval_metric") != "RMSE"
        or params.get("use_best_model") is not False
        or params.get("eval_fraction") != 0
        or params.get("thread_count") != THREADS_PER_WORKER
        or params.get("boosting_type") != "Plain"
        or params.get("depth") != 6
        or params.get("grow_policy") != "SymmetricTree"
        or params.get("iterations") != ITERATIONS
        or params.get("random_seed") != seed
        or params.get("learning_rate")
        != coordinate["frozen_learning_rate"]
        or not _valid_optional_number(params.get("random_strength"))
        or not isinstance(params.get("bootstrap_type"), str)
        or not _valid_optional_number(params.get("bagging_temperature"))
        or not _valid_optional_number(
            params.get("subsample"), minimum=0.0, maximum=1.0
        )
        or not _valid_optional_number(params.get("l2_leaf_reg"))
        or (
            params.get("one_hot_max_size") is not None
            and (
                type(params.get("one_hot_max_size")) is not int
                or params["one_hot_max_size"] < 0
            )
        )
        or type(params.get("leaf_estimation_iterations")) is not int
        or params["leaf_estimation_iterations"] <= 0
        or not isinstance(
            params.get("leaf_estimation_backtracking"), str
        )
    ):
        raise RuntimeError("T7b CatBoost resolved shared policy changed")
    if arm is None:
        return
    expected = ARMS[arm]
    resolved_overrides = {
        "random_strength": params.get("random_strength"),
        "bootstrap_type": params.get("bootstrap_type"),
        "l2_leaf_reg": params.get("l2_leaf_reg"),
        "one_hot_max_size": params.get("one_hot_max_size"),
        "leaf_estimation_iterations": params.get(
            "leaf_estimation_iterations"
        ),
        "leaf_estimation_backtracking": params.get(
            "leaf_estimation_backtracking"
        ),
    }
    if any(
        resolved_overrides.get(key) != value
        and not (
            key == "one_hot_max_size"
            and categorical_count == 0
            and resolved_overrides[key] is None
        )
        for key, value in expected.items()
    ):
        raise RuntimeError(f"T7b CatBoost arm override changed: {arm}")


def _validate_requested(policy, coordinate, seed, arm):
    expected = _requested_policy(coordinate, seed, arm)
    if not t7._same_typed_value(policy, expected):
        raise RuntimeError(f"T7b CatBoost requested policy changed: {arm}")


def _validate_arm_isolation(result):
    by_arm = {arm["arm"]: arm for arm in result["arms"]}
    baseline = by_arm["baseline"]["resolved_params"]
    coupled = {
        "bootstrap_type": {
            "bootstrap_type",
            "bagging_temperature",
            "subsample",
        }
    }
    for name, arm in by_arm.items():
        params = arm["resolved_params"]
        allowed = set(ARMS[name])
        for key in ARMS[name]:
            allowed.update(coupled.get(key, ()))
        for key in RESOLVED_KEYS:
            if key not in allowed and params[key] != baseline[key]:
                raise RuntimeError(
                    f"T7b CatBoost arm changed an undeclared parameter: "
                    f"{name}/{key}"
                )


def _integrity_checks(result, coordinate):
    by_arm = {arm["arm"]: arm for arm in result["arms"]}
    baseline = by_arm["baseline"]
    if result["seed"] == 4 and (
        baseline["validation"]["prediction_sha256"]
        != coordinate["seed4_validation_prediction_sha256"]
        or baseline["test"]["prediction_sha256"]
        != coordinate["seed4_test_prediction_sha256"]
    ):
        raise RuntimeError("T7b seed-4 baseline does not byte-match T7")
    cars_control = None
    if result["task_id"] == CARS_TASK_ID:
        if result["categorical_count"] != 0:
            raise RuntimeError("T7b Cars negative control is not numeric")
        cars_control = {}
        for name in ("one_hot_max_size_0", "one_hot_max_size_255"):
            exact = (
                by_arm[name]["validation"] == baseline["validation"]
                and by_arm[name]["test"] == baseline["test"]
            )
            cars_control[name] = exact
            if not exact:
                raise RuntimeError(
                    f"T7b Cars one-hot negative control changed: {name}"
                )
    _validate_arm_isolation(result)
    result["integrity"] = {
        "seed4_t7_baseline_byte_match": (
            True if result["seed"] == 4 else None
        ),
        "cars_one_hot_negative_control": cars_control,
        "arm_isolation_verified": True,
    }


def run_worker(task_id, fold, seed, execution_index):
    freeze = _source_freeze()
    declaration, raw_by_coordinate = _coordinates()
    schedule = _schedule(declaration["coordinates"])
    if (
        type(task_id) is not int
        or type(fold) is not int
        or type(seed) is not int
        or type(execution_index) is not int
        or execution_index < 0
        or execution_index >= len(schedule)
        or schedule[execution_index]
        != {
            "execution_index": execution_index,
            "coordinate_index": execution_index // len(SEEDS),
            "task_id": task_id,
            "fold": fold,
            "seed": seed,
        }
    ):
        raise ValueError("T7b worker does not match the frozen schedule")
    _validate_runtime_environment(freeze["environment"])
    source = _git_state()
    _validate_execution_source(source, freeze)
    import catboost
    from catboost import CatBoostRegressor

    if catboost.__version__ != freeze["catboost_version"]:
        raise RuntimeError("T7b requires CatBoost 1.2.10")
    coordinate = declaration["coordinates"][execution_index // len(SEEDS)]
    _registry, rows = t7._rows()
    row = rows[task_id]
    task, X, y, categorical = c2._load_task(row)
    outer_train, outer_test = task.get_train_test_split_indices(
        repeat=0, fold=fold, sample=0
    )
    outer = c2._verify_outer_split(row, fold, outer_train, outer_test)
    fit_indices, validation_indices, inner = c2.development_split(
        outer_train, task_id=task_id, fold=fold
    )
    frozen = raw_by_coordinate[coordinate["coordinate_index"]]
    if (
        not t7._same_typed_value(outer, frozen["outer_split"])
        or not t7._same_typed_value(inner, frozen["inner_split"])
        or int(X.shape[1]) != frozen["n_features"]
        or len(categorical) != frozen["categorical_count"]
    ):
        raise RuntimeError("T7b C2/T7 input boundary changed")
    X_fit = _catboost_frame(X.iloc[fit_indices], categorical)
    X_validation = _catboost_frame(
        X.iloc[validation_indices], categorical
    )
    X_test = _catboost_frame(X.iloc[outer_test], categorical)
    y_fit = y.iloc[fit_indices]
    y_validation = y.iloc[validation_indices]
    y_test = y.iloc[outer_test]
    warmup_seconds = t7._warmup()
    arm_results = []
    for position, arm in enumerate(_arm_order(execution_index)):
        model = CatBoostRegressor(
            **REQUESTED_STATIC_PARAMS,
            learning_rate=coordinate["frozen_learning_rate"],
            random_seed=seed,
            **ARMS[arm],
        )
        started = time.perf_counter_ns()
        model.fit(X_fit, y_fit, cat_features=categorical or None)
        fit_seconds = (time.perf_counter_ns() - started) / 1e9
        validation = t7._score(model, X_validation, y_validation)
        test = t7._score(model, X_test, y_test)
        timing = t7._timed_predict(
            model, X_test, test["prediction_sha256"]
        )
        resolved = _resolved_params(model)
        _validate_resolved(
            resolved,
            coordinate,
            seed,
            arm,
            categorical_count=len(categorical),
        )
        requested = _requested_policy(coordinate, seed, arm)
        _validate_requested(requested, coordinate, seed, arm)
        arm_results.append(
            {
                "arm": arm,
                "position": position,
                "overrides": ARMS[arm],
                "fit_seconds": float(fit_seconds),
                "validation": validation,
                "test": test,
                "prediction_timing": timing,
                "tree_count": int(model.tree_count_),
                "requested_policy": requested,
                "resolved_params": resolved,
            }
        )
        del model
        gc.collect()
    result = {
        "execution_index": execution_index,
        "coordinate_index": coordinate["coordinate_index"],
        "task_id": task_id,
        "dataset_id": int(row["dataset_id"]),
        "dataset_name": row["dataset_name"],
        "lineage_cluster": row["lineage_cluster"],
        "fold": fold,
        "seed": seed,
        "frozen_learning_rate": coordinate["frozen_learning_rate"],
        "n_features": int(X.shape[1]),
        "categorical_count": len(categorical),
        "outer_split": outer,
        "inner_split": inner,
        "arm_order": list(_arm_order(execution_index)),
        "arms": arm_results,
        "warmup_seconds": float(warmup_seconds),
        "peak_rss_bytes": t7._peak_rss_bytes(),
    }
    _integrity_checks(result, coordinate)
    result["behavior_sha256"] = _json_sha256(
        {
            "execution_index": execution_index,
            "task_id": task_id,
            "fold": fold,
            "seed": seed,
            "arms": [
                {
                    "arm": arm["arm"],
                    "validation": arm["validation"],
                    "test": arm["test"],
                    "requested_policy": arm["requested_policy"],
                    "resolved_params": arm["resolved_params"],
                }
                for arm in arm_results
            ],
            "integrity": result["integrity"],
        }
    )
    return result


def _git_state():
    state = t7.creator.git_state(ROOT)
    state["remote_branch_head"] = subprocess.run(
        ["git", "rev-parse", "origin/main"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        state["branch"] != "main"
        or state["head"] != state["remote_branch_head"]
        or not state["clean"]
        or state["status"]
    ):
        raise RuntimeError("T7b requires clean pushed main")
    return state


def _validate_execution_source(source, freeze):
    model_head = freeze["darkofit_model_head"]
    execution_head = source["head"]
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", model_head, execution_head],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            "T7b model source head is not an ancestor of execution head"
        )
    completed = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "-z",
            model_head,
            execution_head,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    changed = {
        value.decode("utf-8")
        for value in completed.stdout.split(b"\0")
        if value
    }
    if changed != set(ALLOWED_POST_MODEL_SOURCE_PATHS):
        raise RuntimeError(
            "T7b execution head may differ from model head only by the "
            "dedicated freeze JSON"
        )
    return changed


def _binding(source, freeze, declaration):
    return {
        "freeze_file_sha256": _sha256(FREEZE),
        "freeze_canonical_sha256": freeze["freeze_sha256"],
        "source_sha256": freeze["source_sha256"],
        "coordinates_sha256": declaration["coordinates_sha256"],
        "source_head": source["head"],
        "model_source_head": freeze["darkofit_model_head"],
        "environment": freeze["environment"],
        "catboost_version": freeze["catboost_version"],
        "task_type": "CPU",
        "seeds": list(SEEDS),
        "arms": list(ARM_NAMES),
        "historical_darkofit_over_catboost_default_ratio": declaration[
            "input_boundary"
        ]["historical_darkofit_over_catboost_default_ratio"],
    }


def _spool_path(directory, task_id, fold, seed):
    return directory / f"task-{task_id}--fold-{fold}--seed-{seed}.json"


def _validate_worker(result, expected):
    if (
        not isinstance(result, dict)
        or type(result.get("execution_index")) is not int
        or result["execution_index"] != expected["execution_index"]
        or type(result.get("coordinate_index")) is not int
        or result["coordinate_index"] != expected["coordinate_index"]
        or type(result.get("task_id")) is not int
        or result["task_id"] != expected["task_id"]
        or type(result.get("fold")) is not int
        or result["fold"] != expected["fold"]
        or type(result.get("seed")) is not int
        or result["seed"] != expected["seed"]
        or result.get("arm_order") != list(_arm_order(expected["execution_index"]))
        or not isinstance(result.get("arms"), list)
        or [arm.get("arm") for arm in result["arms"]]
        != list(_arm_order(expected["execution_index"]))
        or type(result.get("behavior_sha256")) is not str
    ):
        raise RuntimeError("T7b worker identity changed")


def _verify_closing_state(source, freeze, declaration):
    closing_source = _git_state()
    closing_freeze = _source_freeze()
    closing_declaration, _raw = _coordinates()
    if (
        not t7._same_typed_value(closing_source, source)
        or not t7._same_typed_value(closing_freeze, freeze)
        or not t7._same_typed_value(closing_declaration, declaration)
    ):
        raise RuntimeError("T7b source or input boundary changed during execution")
    _validate_runtime_environment(closing_freeze["environment"])
    _validate_execution_source(closing_source, closing_freeze)


def _run_one(expected, spool, binding):
    path = _spool_path(
        spool, expected["task_id"], expected["fold"], expected["seed"]
    )
    if path.exists() or path.is_symlink():
        result, digest = t7._load_spool(
            path, binding, expected["task_id"], expected["fold"]
        )
        _validate_worker(result, expected)
        return result, {"path": path.name, "sha256": digest}, True
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-task-id",
        str(expected["task_id"]),
        "--worker-fold",
        str(expected["fold"]),
        "--worker-seed",
        str(expected["seed"]),
        "--execution-index",
        str(expected["execution_index"]),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=t7._environment(),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"T7b worker {expected['execution_index']} failed\n"
            f"{completed.stderr}"
        )
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if len(lines) != 1:
        raise RuntimeError("T7b worker protocol failed")
    result = t7._json_loads(
        lines[0][len(WORKER_PREFIX) :], "T7b worker result"
    )
    _validate_worker(result, expected)
    stored, digest = t7._create_spool(
        path, binding, result, return_publish_state=False
    )
    _validate_worker(stored, expected)
    return stored, {"path": path.name, "sha256": digest}, False


def run(output, spool):
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing existing T7b output: {output}")
    t7._reject_symlink_directory(
        output.parent, "refusing symlink T7b output directory"
    )
    t7._reject_symlink_directory(
        spool, "refusing symlink T7b spool directory"
    )
    freeze = _source_freeze()
    environment = _validate_runtime_environment(freeze["environment"])
    declaration, _raw = _coordinates()
    source = _git_state()
    _validate_execution_source(source, freeze)
    binding = _binding(source, freeze, declaration)
    schedule = _schedule(declaration["coordinates"])
    results = []
    spool_records = []
    resumed = 0
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        futures = {
            executor.submit(_run_one, expected, spool, binding): expected
            for expected in schedule
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            expected = futures[future]
            result, spool_record, was_resumed = future.result()
            results.append(result)
            spool_record["resumed"] = bool(was_resumed)
            spool_records.append(spool_record)
            resumed += int(was_resumed)
            print(
                f"T7b {completed}/{len(schedule)}: "
                f"{expected['task_id']}/{expected['fold']}/"
                f"{expected['seed']}",
                flush=True,
            )
    results.sort(key=lambda row: row["execution_index"])
    spool_records.sort(key=lambda row: row["path"])
    _verify_closing_state(source, freeze, declaration)
    payload = {
        "schema_version": 1,
        "name": "darkofit_t7b_catboost_gap_attribution_raw_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "development_data_only": True,
        "lockbox_data_used": False,
        "confirmation_outcomes_inspected": False,
        "default_change_authorized": False,
        "source": source,
        "runtime": {
            "python": sys.version,
            "machine": t7.creator._machine_details(),
            "dependencies": t7.creator._dependency_versions(),
            "environment": environment,
        },
        "protocol": binding,
        "coordinate_count": len(declaration["coordinates"]),
        "execution_count": len(schedule),
        "fit_count": len(schedule) * len(ARM_NAMES),
        "resumed_execution_count": resumed,
        "results": results,
        "spool_records": spool_records,
    }
    payload["raw_sha256"] = _json_sha256(payload)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    t7._create_output(output, encoded)
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spool", type=Path, default=DEFAULT_SPOOL)
    parser.add_argument("--worker-task-id", type=int)
    parser.add_argument("--worker-fold", type=int)
    parser.add_argument("--worker-seed", type=int)
    parser.add_argument("--execution-index", type=int)
    args = parser.parse_args(argv)
    args.output = Path(os.path.abspath(os.path.expanduser(args.output)))
    args.spool = Path(os.path.abspath(os.path.expanduser(args.spool)))
    worker_values = (
        args.worker_task_id,
        args.worker_fold,
        args.worker_seed,
        args.execution_index,
    )
    if any(value is not None for value in worker_values) and any(
        value is None for value in worker_values
    ):
        parser.error("all T7b worker arguments are required together")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.worker_task_id is not None:
        result = run_worker(
            args.worker_task_id,
            args.worker_fold,
            args.worker_seed,
            args.execution_index,
        )
        print(
            WORKER_PREFIX
            + json.dumps(
                result, sort_keys=True, separators=(",", ":"), allow_nan=False
            ),
            flush=True,
        )
        return
    run(args.output, args.spool)


if __name__ == "__main__":
    main()
