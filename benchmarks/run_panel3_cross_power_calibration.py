#!/usr/bin/env python3
"""Run the spent 13-task exact-policy Panel 3 power calibration."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import build_ctr23_contamination_registry as fingerprints  # noqa: E402
from benchmarks import freeze_panel3_cross_power_calibration as freeze  # noqa: E402
from benchmarks import panel3_data_contract as data_contract  # noqa: E402
from benchmarks import panel3_registry_common as common  # noqa: E402
from benchmarks import run_panel3_confirmation as panel3  # noqa: E402
from benchmarks import run_t5_composite_confirmation as t5  # noqa: E402


DEFAULT_FREEZE = freeze.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    ROOT / "benchmarks" / "panel3_cross_power_calibration_raw.json"
)
DEFAULT_SPOOL_DIRECTORY = (
    ROOT / ".cache" / "panel3_cross_power_calibration_spool_v1"
)
WORKER_PREFIX = "PANEL3_CROSS_POWER_CALIBRATION_RESULT="
CONTROL_ARM = "current_default"
CANDIDATE_ARMS = (
    "t5_composite_policy",
    "guarded_cross_features_policy",
)
ARM_ORDER = (CONTROL_ARM, *CANDIDATE_ARMS)
THREAD_COUNT = 6
RANDOM_STATE = 4
T5_SIZE_GATE = common.t5_minimum_outer_training_rows()
if t5.SIZE_GATE != T5_SIZE_GATE:
    raise RuntimeError("Panel 3 calibration T5 size gate changed")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _array_sha256(value: Any, dtype: str = "<f8") -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _git(*arguments: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=ROOT,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def _validate_post_freeze_history(source_head: str, head: str) -> None:
    """Require one create-only H2 freeze and no other committed changes."""
    expected = freeze.FREEZE_RELATIVE
    final_change = _git(
        "diff",
        "--name-status",
        f"{source_head}..{head}",
    ).splitlines()
    commits = [
        value
        for value in _git(
            "rev-list",
            f"{source_head}..{head}",
        ).splitlines()
        if value
    ]
    touched_by_commit = {}
    for commit in commits:
        touched_by_commit[commit] = {
            value
            for value in _git(
                "diff-tree",
                "--root",
                "--no-commit-id",
                "--name-only",
                "-r",
                "-m",
                commit,
            ).splitlines()
            if value
        }
    nonempty = {
        commit: paths
        for commit, paths in touched_by_commit.items()
        if paths
    }
    if (
        final_change != [f"A\t{expected}"]
        or len(nonempty) != 1
        or next(iter(nonempty.values()), set()) != {expected}
    ):
        raise RuntimeError(
            "calibration H2 must add only the create-only source freeze"
        )


def coordinate_key(
    task_id: int,
    repeat: int,
    fold: int,
    sample: int = 0,
) -> str:
    return f"{task_id}-r{repeat}-f{fold}-s{sample}"


def worker_key(coordinate: dict[str, int], arm: str) -> str:
    return f"{coordinate_key(**coordinate)}-{arm}"


def expected_coordinates() -> list[dict[str, int]]:
    return [
        {
            "task_id": task_id,
            "repeat": coordinate["repeat"],
            "fold": coordinate["fold"],
            "sample": coordinate["sample"],
        }
        for task_id in freeze.TASKS.values()
        for coordinate in freeze.COORDINATES
    ]


def validate_source_freeze(
    artifact: dict[str, Any],
    *,
    freeze_path: Path = DEFAULT_FREEZE,
    require_repository_state: bool = True,
) -> dict[str, Any]:
    common.verify_artifact_sha256(artifact, "source_freeze_sha256")
    required = {
        "schema_version",
        "name",
        "created_at",
        "source_head",
        "source_tree",
        "source_head_clean",
        "post_freeze_allowed_tracked_path",
        "source_file_sha256",
        "source_file_set_sha256",
        "protocol_sha256",
        "runner_sha256",
        "analyzer_sha256",
        "candidate_contract_sha256",
        "runtime_contract",
        "runtime_environment",
        "spent_provenance_sha256",
        "tasks",
        "native_categorical_columns",
        "task_view_attestations",
        "coordinates",
        "arms",
        "coordinate_count",
        "result_count",
        "outcome_blind_source_freeze",
        "candidate_or_control_models_fitted",
        "candidate_or_control_outcomes_inspected",
        "development_only",
        "panel3_authorized",
        "default_promotion_authorized",
        "product_claim_authorized",
        "source_freeze_sha256",
    }
    if (
        set(artifact) != required
        or artifact["schema_version"] != 1
        or artifact["name"]
        != "darkofit_panel3_cross_power_calibration_source_freeze_v1"
        or artifact["source_head_clean"] is not True
        or artifact["post_freeze_allowed_tracked_path"]
        != freeze.FREEZE_RELATIVE
        or artifact["tasks"] != freeze.TASKS
        or artifact["native_categorical_columns"]
        != freeze.EXPECTED_NATIVE_CATEGORICAL_COLUMNS
        or not isinstance(artifact["task_view_attestations"], dict)
        or set(artifact["task_view_attestations"])
        != {str(value) for value in freeze.TASKS.values()}
        or artifact["coordinates"] != list(freeze.COORDINATES)
        or artifact["arms"] != list(ARM_ORDER)
        or artifact["coordinate_count"] != 39
        or artifact["result_count"] != 117
        or artifact["outcome_blind_source_freeze"] is not True
        or artifact["candidate_or_control_models_fitted"] is not False
        or artifact["candidate_or_control_outcomes_inspected"] is not False
        or artifact["development_only"] is not True
        or artifact["panel3_authorized"] is not False
        or artifact["default_promotion_authorized"] is not False
        or artifact["product_claim_authorized"] is not False
    ):
        raise RuntimeError("calibration source-freeze contract changed")
    source_head = str(artifact["source_head"])
    frozen_paths = tuple(artifact["source_file_sha256"])
    historical_files = freeze.source_file_sha256_at_head(
        source_head,
        frozen_paths,
    )
    if (
        artifact["source_file_sha256"] != historical_files
        or artifact["source_file_set_sha256"]
        != freeze.source_tree_sha256(historical_files)
        or artifact["source_tree"]
        != _git("rev-parse", f"{source_head}^{{tree}}")
        or artifact["protocol_sha256"]
        != historical_files[
            str(freeze.PROTOCOL.relative_to(ROOT))
        ]
        or artifact["runner_sha256"]
        != historical_files[
            str(Path(__file__).resolve().relative_to(ROOT))
        ]
        or artifact["analyzer_sha256"]
        != historical_files[
            str(freeze.ANALYZER.relative_to(ROOT))
        ]
        or artifact["candidate_contract_sha256"]
        != historical_files[
            str(freeze.CANDIDATE_CONTRACT.relative_to(ROOT))
        ]
        or artifact["runtime_contract"].get("path")
        != str(freeze.RUNTIME_CONTRACT.relative_to(ROOT))
        or artifact["runtime_contract"].get("sha256")
        != historical_files[
            str(freeze.RUNTIME_CONTRACT.relative_to(ROOT))
        ]
        or artifact["spent_provenance_sha256"]
        != {
            relative: hashlib.sha256(
                freeze._blob(relative, source_head)
            ).hexdigest()
            for relative in freeze.SPENT_PROVENANCE_PATHS
        }
    ):
        raise RuntimeError("calibration frozen source bytes changed")
    runtime_environment = artifact["runtime_environment"]
    if (
        not isinstance(runtime_environment, dict)
        or runtime_environment.get("contract_name")
        != "darkofit_panel3_exact_runtime_environment_v1"
        or runtime_environment.get("contract_kind")
        != "exact_active_environment_versions_v1"
        or not isinstance(runtime_environment.get("packages"), dict)
    ):
        raise RuntimeError("calibration frozen runtime changed")
    runtime = {
        "contract_kind": runtime_environment["contract_kind"],
        "python_implementation": runtime_environment[
            "python_implementation"
        ],
        "python_version": runtime_environment["python_version"],
        "packages": runtime_environment["packages"],
    }
    if require_repository_state:
        observed_files = freeze.source_file_sha256()
        if (
            observed_files != historical_files
            or _sha256(freeze.RUNTIME_CONTRACT)
            != artifact["runtime_contract"]["sha256"]
        ):
            raise RuntimeError("calibration live source bytes changed")
        candidate_contract = common.load_json(freeze.CANDIDATE_CONTRACT)
        if panel3._validate_runtime_contract(candidate_contract) != runtime:
            raise RuntimeError("calibration live runtime changed")
        if _git("status", "--porcelain"):
            raise RuntimeError("calibration execution requires a clean tree")
        head = _git("rev-parse", "HEAD")
        if not _is_ancestor(source_head, head):
            raise RuntimeError(
                "calibration execution no longer descends from source freeze"
            )
        _validate_post_freeze_history(source_head, head)
    return runtime


def load_source_freeze(
    path: Path = DEFAULT_FREEZE,
    *,
    require_repository_state: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    artifact, file_sha256 = common.secure_load_json(path)
    if not isinstance(artifact, dict):
        raise RuntimeError("calibration source freeze is not an object")
    runtime = validate_source_freeze(
        artifact,
        freeze_path=path,
        require_repository_state=require_repository_state,
    )
    return artifact, runtime, file_sha256


def load_task(task_id: int, source_freeze: dict[str, Any]):
    import openml

    if task_id not in freeze.TASKS.values():
        raise RuntimeError("calibration task is outside the frozen panel")
    task = openml.tasks.get_task(task_id, download_splits=True)
    dataset = task.get_dataset()
    X, y, categorical, names = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
        include_row_id=False,
        include_ignore_attribute=False,
    )
    if (
        not isinstance(X, pd.DataFrame)
        or list(X.columns) != list(names)
        or len(categorical) != X.shape[1]
    ):
        raise RuntimeError(f"calibration task {task_id} feature view changed")
    try:
        target = pd.to_numeric(y, errors="raise").astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"calibration task {task_id} target is not numeric"
        ) from exc
    target_values = target.to_numpy(dtype=np.float64)
    if (
        target_values.ndim != 1
        or len(target_values) != len(X)
        or not np.isfinite(target_values).all()
    ):
        raise RuntimeError(
            f"calibration task {task_id} target is nonfinite"
        )
    categorical_indices = list(
        panel3.categorical_column_indices(X, list(categorical))
    )
    metadata = {
        "task_id": int(task_id),
        "dataset_id": int(dataset.dataset_id),
        "dataset_name": str(dataset.name),
        "target_name": str(task.target_name),
        "openml_declared_md5": str(dataset.md5_checksum),
        "split_dimensions": list(task.get_split_dimensions()),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "feature_names": [str(value) for value in X.columns],
        "feature_schema": data_contract.feature_schema(X),
        "feature_schema_sha256": data_contract.feature_schema_sha256(X),
        "categorical_feature_indices": categorical_indices,
        "categorical_feature_names": [
            str(X.columns[index]) for index in categorical_indices
        ],
        "ordered_task_view_sha256": freeze.ordered_task_view_sha256(
            X,
            target,
        ),
        "dataset_fingerprint": fingerprints.dataset_fingerprint(X, target),
    }
    expected = source_freeze["task_view_attestations"].get(str(task_id))
    if (
        not isinstance(expected, dict)
        or {key: value for key, value in expected.items() if key != "coordinates"}
        != metadata
        or metadata["categorical_feature_names"]
        != source_freeze["native_categorical_columns"][str(dataset.name)]
    ):
        raise RuntimeError(
            f"calibration task {task_id} frozen view changed"
        )
    return task, X, target, categorical_indices, metadata


def resolve_split(
    task: Any,
    X: pd.DataFrame,
    expected: dict[str, Any],
    *,
    repeat: int,
    fold: int,
    sample: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train, test = task.get_train_test_split_indices(
        repeat=repeat,
        fold=fold,
        sample=sample,
    )
    train = np.asarray(train, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    if (
        train.ndim != 1
        or test.ndim != 1
        or len(train) == 0
        or len(test) == 0
        or np.any(train < 0)
        or np.any(test < 0)
        or np.any(train >= len(X))
        or np.any(test >= len(X))
        or len(np.unique(train)) != len(train)
        or len(np.unique(test)) != len(test)
        or np.intersect1d(train, test).size
        or len(train) + len(test) != len(X)
    ):
        raise RuntimeError("calibration OpenML split is invalid")
    observed = {
        "repeat": int(repeat),
        "fold": int(fold),
        "sample": int(sample),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_index_sha256": _array_sha256(train, "<i8"),
        "test_index_sha256": _array_sha256(test, "<i8"),
    }
    if observed != expected:
        raise RuntimeError("calibration frozen OpenML split changed")
    return train, test, observed


def fit_arm(
    arm: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_indices: list[int],
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, float, dict[str, Any], dict[str, Any]]:
    if arm == CONTROL_ARM:
        prediction, seconds, timing, metadata = t5._fit_control(
            X_train,
            y_train,
            categorical_indices,
            X_test,
        )
        metadata["kind"] = CONTROL_ARM
        return prediction, seconds, timing, metadata
    if arm == "t5_composite_policy":
        prediction, seconds, timing, metadata = t5._fit_composite(
            X_train,
            y_train,
            categorical_indices,
            X_test,
            {},
        )
        metadata["kind"] = arm
        return prediction, seconds, timing, metadata
    if arm == "guarded_cross_features_policy":
        return panel3._fit_guarded_cross(
            X_train,
            y_train,
            categorical_indices,
            X_test,
        )
    raise ValueError(f"unknown calibration arm: {arm}")


def _finite_measurement(
    value: Any,
    *,
    positive: bool,
    label: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"calibration {label} is invalid")
    result = float(value)
    if (
        not math.isfinite(result)
        or (result <= 0.0 if positive else result < 0.0)
    ):
        raise RuntimeError(f"calibration {label} is invalid")
    return result


def _validate_selection_fit(value: Any) -> None:
    from benchmarks import analyze_panel3_confirmation as confirmation_analyzer

    required = {
        "name",
        "validation_rmse",
        "fit_seconds",
        "fit_metadata",
        "validation",
    }
    optional = {
        "tree_mode_selection",
        "pair_count",
        "pairs",
        "transform_seconds",
    }
    if (
        not isinstance(value, dict)
        or not required <= set(value)
        or not set(value) <= required | optional
        or not isinstance(value["name"], str)
        or not value["name"]
        or not isinstance(value["validation"], dict)
    ):
        raise RuntimeError("calibration selection-fit metadata is invalid")
    _finite_measurement(
        value["validation_rmse"],
        positive=True,
        label="selection validation RMSE",
    )
    _finite_measurement(
        value["fit_seconds"],
        positive=False,
        label="selection fit seconds",
    )
    confirmation_analyzer._validate_darkofit_fit_metadata(
        value["fit_metadata"],
        "calibration selection",
    )
    if "transform_seconds" in value:
        _finite_measurement(
            value["transform_seconds"],
            positive=False,
            label="selection transform seconds",
        )
    if "pair_count" in value and (
        type(value["pair_count"]) is not int or value["pair_count"] < 0
    ):
        raise RuntimeError("calibration selection pair count is invalid")
    if "pairs" in value and not isinstance(value["pairs"], list):
        raise RuntimeError("calibration selection pairs are invalid")


def validate_arm_metadata(
    metadata: Any,
    *,
    arm: str,
    t5_size_gate_applicable: bool,
    fit_seconds: float,
    train_rows: int,
    feature_count: int,
    categorical_indices: Sequence[int],
) -> None:
    """Validate the exact policy metadata needed by the calibration census."""
    from benchmarks import analyze_panel3_confirmation as confirmation_analyzer

    if not isinstance(metadata, dict) or metadata.get("kind") != arm:
        raise RuntimeError("calibration fitted metadata arm changed")
    if arm == CONTROL_ARM:
        if (
            set(metadata)
            != {
                "kind",
                "engaged",
                "selected_configuration",
                "final_fit",
            }
            or metadata["engaged"] is not False
            or metadata["selected_configuration"] != "product_default"
        ):
            raise RuntimeError("calibration control metadata changed")
        confirmation_analyzer._validate_fitted_metadata(
            {
                "arm": arm,
                "metadata": metadata,
                "fit_seconds": fit_seconds,
                "train_rows": train_rows,
                "feature_policy": {
                    "retained_feature_count": feature_count
                },
                "categorical_feature_indices": list(
                    categorical_indices
                ),
            },
            strict=True,
        )
        return

    if arm == "t5_composite_policy":
        if not t5_size_gate_applicable:
            if (
                set(metadata)
                != {
                    "kind",
                    "engaged",
                    "selected_configuration",
                    "final_fit",
                    "decline_reason",
                    "size_gate",
                    "total_selection_fit_seconds",
                    "policy_overhead_seconds",
                    "final_fit_seconds",
                }
                or metadata["engaged"] is not False
                or metadata["selected_configuration"] != "product_default"
                or metadata["decline_reason"] != "below_size_gate"
                or metadata["size_gate"] != T5_SIZE_GATE
                or metadata["total_selection_fit_seconds"] != 0.0
                or not math.isclose(
                    float(fit_seconds),
                    float(metadata["policy_overhead_seconds"])
                    + float(metadata["final_fit_seconds"]),
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
            ):
                raise RuntimeError(
                    "calibration T5 size-gate metadata changed"
                )
            confirmation_analyzer._validate_fitted_metadata(
                {
                    "arm": arm,
                    "metadata": metadata,
                    "fit_seconds": fit_seconds,
                    "train_rows": train_rows,
                    "feature_policy": {
                        "retained_feature_count": feature_count
                    },
                    "categorical_feature_indices": list(
                        categorical_indices
                    ),
                },
                strict=True,
            )
            return
        required = {
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
            "policy_overhead_seconds",
            "final_transform_seconds",
            "final_fit_seconds",
            "final_fit",
        }
        if set(metadata) != required:
            raise RuntimeError("calibration T5 metadata fields changed")
        control = _finite_measurement(
            metadata["control_validation_rmse"],
            positive=True,
            label="T5 control validation RMSE",
        )
        challenger = _finite_measurement(
            metadata["challenger_validation_rmse"],
            positive=True,
            label="T5 challenger validation RMSE",
        )
        ratio = _finite_measurement(
            metadata["relative_challenger_validation_ratio"],
            positive=True,
            label="T5 validation ratio",
        )
        engaged = type(metadata["engaged"]) is bool and metadata["engaged"]
        expected_engaged = ratio <= t5.OUTER_GUARD_RATIO
        if (
            type(metadata["engaged"]) is not bool
            or engaged != expected_engaged
            or not math.isclose(
                ratio,
                challenger / control,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
            or metadata["size_gate"] != T5_SIZE_GATE
            or metadata["outer_guard_ratio"]
            != t5.OUTER_GUARD_RATIO
            or metadata["cross_guard_ratio"]
            != t5.CROSS_GUARD_RATIO
            or metadata["selection_rounds"] != t5.SELECTION_ROUNDS
            or metadata["decline_reason"]
            != (None if engaged else "outer_validation_guard")
            or metadata["selected_configuration"]
            != ("challenger" if engaged else "product_default")
            or not isinstance(metadata["selected_tree_mode"], str)
            or not metadata["selected_tree_mode"]
            or type(metadata["selected_linear_leaves"]) is not bool
            or type(metadata["selected_crosses"]) is not bool
            or not isinstance(metadata["selected_cross_pairs"], list)
            or metadata["selected_cross_pair_count"]
            != len(metadata["selected_cross_pairs"])
            or type(metadata["selected_best_iteration"]) is not int
            or metadata["selected_best_iteration"] <= 0
            or not isinstance(metadata["selection_fits"], list)
            or not metadata["selection_fits"]
            or not isinstance(metadata["split"], dict)
        ):
            raise RuntimeError("calibration T5 metadata changed")
        for record in metadata["selection_fits"]:
            _validate_selection_fit(record)
        for field in (
            "selected_resolved_learning_rate",
            "total_selection_fit_seconds",
            "policy_overhead_seconds",
            "final_transform_seconds",
            "final_fit_seconds",
        ):
            _finite_measurement(
                metadata[field],
                positive=field == "selected_resolved_learning_rate",
                label=f"T5 {field}",
            )
        if not math.isclose(
            float(fit_seconds),
            float(metadata["total_selection_fit_seconds"])
            + float(metadata["final_fit_seconds"])
            + float(metadata["policy_overhead_seconds"]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise RuntimeError("calibration T5 fit-time ledger changed")
        confirmation_analyzer._validate_fitted_metadata(
            {
                "arm": arm,
                "metadata": metadata,
                "fit_seconds": fit_seconds,
                "train_rows": train_rows,
                "feature_policy": {
                    "retained_feature_count": feature_count
                },
                "categorical_feature_indices": list(
                    categorical_indices
                ),
            },
            strict=True,
        )
        return

    if arm != "guarded_cross_features_policy":
        raise RuntimeError("calibration fitted metadata arm is unknown")
    required = {
        "kind",
        "engaged",
        "decline_reason",
        "split",
        "cross_guard_ratio",
        "selected_configuration",
        "selected_linear_leaves",
        "selected_crosses",
        "candidate_cross_pairs",
        "selected_cross_pairs",
        "selected_cross_pair_count",
        "uncrossed_validation_rmse",
        "crossed_validation_rmse",
        "relative_crossed_validation_ratio",
        "selected_best_iteration",
        "selected_resolved_learning_rate",
        "selected_selection_fit",
        "selection_fits",
        "total_selection_fit_seconds",
        "policy_overhead_seconds",
        "final_transform_seconds",
        "final_model_fit_seconds",
        "final_fit_seconds",
        "final_refit_parameters",
        "final_fit",
    }
    if set(metadata) != required:
        raise RuntimeError("calibration guarded-cross metadata fields changed")
    uncrossed = _finite_measurement(
        metadata["uncrossed_validation_rmse"],
        positive=True,
        label="guarded uncrossed validation RMSE",
    )
    crossed = metadata["crossed_validation_rmse"]
    ratio = metadata["relative_crossed_validation_ratio"]
    if crossed is None:
        if ratio is not None or metadata["candidate_cross_pairs"] != []:
            raise RuntimeError("calibration guarded-cross decline changed")
        expected_engaged = False
    else:
        crossed = _finite_measurement(
            crossed,
            positive=True,
            label="guarded crossed validation RMSE",
        )
        ratio = _finite_measurement(
            ratio,
            positive=True,
            label="guarded validation ratio",
        )
        if not math.isclose(
            ratio,
            crossed / uncrossed,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise RuntimeError("calibration guarded validation ratio changed")
        expected_engaged = ratio <= panel3.GUARDED_CROSS_RATIO
    engaged = metadata["engaged"]
    refit = metadata["final_refit_parameters"]
    if (
        type(engaged) is not bool
        or engaged != expected_engaged
        or metadata["selected_crosses"] is not engaged
        or metadata["cross_guard_ratio"] != panel3.GUARDED_CROSS_RATIO
        or metadata["decline_reason"]
        != (None if engaged else "cross_guard")
        or metadata["selected_configuration"]
        != ("crossed" if engaged else "uncrossed")
        or type(metadata["selected_linear_leaves"]) is not bool
        or not isinstance(metadata["candidate_cross_pairs"], list)
        or not isinstance(metadata["selected_cross_pairs"], list)
        or metadata["selected_cross_pairs"]
        != (metadata["candidate_cross_pairs"] if engaged else [])
        or metadata["selected_cross_pair_count"]
        != len(metadata["selected_cross_pairs"])
        or type(metadata["selected_best_iteration"]) is not int
        or metadata["selected_best_iteration"] <= 0
        or not isinstance(metadata["selection_fits"], list)
        or not metadata["selection_fits"]
        or metadata["selected_selection_fit"] not in metadata["selection_fits"]
        or not isinstance(metadata["split"], dict)
        or not isinstance(refit, dict)
        or set(refit)
        != {"iterations", "learning_rate", "tree_mode", "linear_leaves", "crossed"}
        or refit["iterations"] != metadata["selected_best_iteration"]
        or refit["learning_rate"] != 0.1
        or refit["tree_mode"] != "catboost"
        or refit["linear_leaves"] is not metadata["selected_linear_leaves"]
        or refit["crossed"] is not engaged
    ):
        raise RuntimeError("calibration guarded-cross metadata changed")
    for record in metadata["selection_fits"]:
        _validate_selection_fit(record)
    for field in (
        "selected_resolved_learning_rate",
        "total_selection_fit_seconds",
        "policy_overhead_seconds",
        "final_transform_seconds",
        "final_model_fit_seconds",
        "final_fit_seconds",
    ):
        _finite_measurement(
            metadata[field],
            positive=field == "selected_resolved_learning_rate",
            label=f"guarded {field}",
        )
    if (
        not math.isclose(
            float(metadata["final_fit_seconds"]),
            float(metadata["final_transform_seconds"])
            + float(metadata["final_model_fit_seconds"]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or not math.isclose(
            float(fit_seconds),
            float(metadata["total_selection_fit_seconds"])
            + float(metadata["final_fit_seconds"])
            + float(metadata["policy_overhead_seconds"]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    ):
        raise RuntimeError("calibration guarded fit-time ledger changed")
    confirmation_analyzer._validate_fitted_metadata(
        {
            "arm": arm,
            "metadata": metadata,
            "fit_seconds": fit_seconds,
            "train_rows": train_rows,
            "feature_policy": {
                "retained_feature_count": feature_count
            },
            "categorical_feature_indices": list(categorical_indices),
        },
        strict=True,
    )


def validate_worker_result(
    result: Any,
    source_freeze: dict[str, Any],
    coordinate: dict[str, int],
    arm: str,
) -> dict[str, Any]:
    required = {
        "worker_key",
        "coordinate",
        "arm",
        "task",
        "split",
        "t5_size_gate_applicable",
        "rmse",
        "fit_seconds",
        "wall_seconds",
        "prediction_timing",
        "prediction_sha256",
        "test_target_sha256",
        "metadata",
        "peak_rss_bytes",
        "behavior_fingerprint_sha256",
    }
    if (
        not isinstance(result, dict)
        or set(result) != required
        or result["worker_key"] != worker_key(coordinate, arm)
        or result["coordinate"] != coordinate
        or result["arm"] != arm
        or not isinstance(result["task"], dict)
        or not isinstance(result["split"], dict)
        or type(result["t5_size_gate_applicable"]) is not bool
        or not isinstance(result["metadata"], dict)
        or type(result["peak_rss_bytes"]) is not int
        or result["peak_rss_bytes"] <= 0
    ):
        raise RuntimeError("calibration worker result contract changed")
    expected_task = source_freeze["task_view_attestations"].get(
        str(coordinate["task_id"])
    )
    expected_splits = (
        []
        if not isinstance(expected_task, dict)
        else [
            row
            for row in expected_task["coordinates"]
            if all(
                row[key] == coordinate[key]
                for key in ("repeat", "fold", "sample")
            )
        ]
    )
    if (
        not isinstance(expected_task, dict)
        or {key: value for key, value in expected_task.items() if key != "coordinates"}
        != result["task"]
        or len(expected_splits) != 1
        or result["split"] != expected_splits[0]
        or result["t5_size_gate_applicable"]
        is not (result["split"]["train_rows"] >= T5_SIZE_GATE)
    ):
        raise RuntimeError("calibration worker data boundary changed")
    _finite_measurement(
        result["rmse"],
        positive=True,
        label="worker RMSE",
    )
    _finite_measurement(
        result["fit_seconds"],
        positive=False,
        label="worker fit seconds",
    )
    _finite_measurement(
        result["wall_seconds"],
        positive=False,
        label="worker wall seconds",
    )
    metadata = result["metadata"]
    behavior = {
        "coordinate": coordinate,
        "arm": arm,
        "rmse": result["rmse"],
        "prediction_sha256": result["prediction_sha256"],
        "metadata": metadata,
    }
    if (
        not panel3._is_sha256(result["prediction_sha256"])
        or not panel3._is_sha256(result["test_target_sha256"])
        or metadata.get("kind") != arm
        or result["behavior_fingerprint_sha256"] != _json_sha256(behavior)
    ):
        raise RuntimeError("calibration worker behavior changed")
    from benchmarks import analyze_panel3_confirmation as confirmation_analyzer

    confirmation_analyzer._validate_prediction_timing(
        result["prediction_timing"]
    )
    validate_arm_metadata(
        metadata,
        arm=arm,
        t5_size_gate_applicable=result["t5_size_gate_applicable"],
        fit_seconds=float(result["fit_seconds"]),
        train_rows=int(result["split"]["train_rows"]),
        feature_count=int(result["task"]["n_features"]),
        categorical_indices=result["task"][
            "categorical_feature_indices"
        ],
    )
    return result


def validate_complete_results(
    results: Sequence[dict[str, Any]],
    source_freeze: dict[str, Any],
) -> None:
    expected = [
        (coordinate, arm)
        for coordinate in expected_coordinates()
        for arm in ARM_ORDER
    ]
    if len(results) != len(expected):
        raise RuntimeError("calibration result grid is incomplete")
    validated = [
        validate_worker_result(result, source_freeze, coordinate, arm)
        for result, (coordinate, arm) in zip(results, expected, strict=True)
    ]
    for start in range(0, len(validated), len(ARM_ORDER)):
        control, t5_result, guarded = validated[
            start : start + len(ARM_ORDER)
        ]
        if (
            t5_result["task"] != control["task"]
            or guarded["task"] != control["task"]
            or t5_result["split"] != control["split"]
            or guarded["split"] != control["split"]
            or t5_result["test_target_sha256"]
            != control["test_target_sha256"]
            or guarded["test_target_sha256"]
            != control["test_target_sha256"]
        ):
            raise RuntimeError("calibration paired-arm data boundary changed")
        if not t5_result["metadata"].get("engaged") and (
            t5_result["prediction_sha256"]
            != control["prediction_sha256"]
            or t5_result["rmse"] != control["rmse"]
        ):
            raise RuntimeError("calibration T5 decline is not byte-exact")


def evaluate_coordinate(
    source_freeze: dict[str, Any],
    task_id: int,
    repeat: int,
    fold: int,
    sample: int,
    arms: Sequence[str],
) -> list[dict[str, Any]]:
    if (
        len(arms) == 0
        or len(set(arms)) != len(arms)
        or any(arm not in ARM_ORDER for arm in arms)
        or tuple(arms) != tuple(arm for arm in ARM_ORDER if arm in arms)
    ):
        raise RuntimeError("calibration worker arm order changed")
    task, X, y, categorical_indices, task_metadata = load_task(
        task_id,
        source_freeze,
    )
    expected_task = source_freeze["task_view_attestations"][str(task_id)]
    expected_splits = [
        row
        for row in expected_task["coordinates"]
        if (
            row["repeat"] == repeat
            and row["fold"] == fold
            and row["sample"] == sample
        )
    ]
    if len(expected_splits) != 1:
        raise RuntimeError("calibration frozen split ledger changed")
    train, test, split = resolve_split(
        task,
        X,
        expected_splits[0],
        repeat=repeat,
        fold=fold,
        sample=sample,
    )
    X_train = X.iloc[train]
    y_train = y.iloc[train]
    X_test = X.iloc[test]
    y_test = y.iloc[test].to_numpy(dtype=np.float64)
    coordinate = {
        "task_id": int(task_id),
        "repeat": int(repeat),
        "fold": int(fold),
        "sample": int(sample),
    }
    results = []
    for arm in arms:
        started = time.perf_counter_ns()
        prediction, fit_seconds, timing, metadata = fit_arm(
            arm,
            X_train,
            y_train,
            categorical_indices,
            X_test,
        )
        wall_seconds = (time.perf_counter_ns() - started) / 1e9
        prediction = np.asarray(prediction, dtype=np.float64)
        if (
            prediction.shape != y_test.shape
            or not np.isfinite(prediction).all()
        ):
            raise RuntimeError("calibration prediction is invalid")
        rmse = float(mean_squared_error(y_test, prediction) ** 0.5)
        if not math.isfinite(rmse) or rmse <= 0.0:
            raise RuntimeError("calibration RMSE is invalid")
        behavior = {
            "coordinate": coordinate,
            "arm": arm,
            "rmse": rmse,
            "prediction_sha256": _array_sha256(prediction),
            "metadata": metadata,
        }
        results.append(
            {
                "worker_key": worker_key(coordinate, arm),
                "coordinate": coordinate,
                "arm": arm,
                "task": task_metadata,
                "split": split,
                "t5_size_gate_applicable": bool(
                    split["train_rows"] >= T5_SIZE_GATE
                ),
                "rmse": rmse,
                "fit_seconds": float(fit_seconds),
                "wall_seconds": float(wall_seconds),
                "prediction_timing": timing,
                "prediction_sha256": _array_sha256(prediction),
                "test_target_sha256": _array_sha256(y_test),
                "metadata": metadata,
                "peak_rss_bytes": panel3._peak_rss_bytes(),
                "behavior_fingerprint_sha256": _json_sha256(behavior),
            }
        )
        gc.collect()
    return results


def spool_binding(
    source_freeze: dict[str, Any],
    source_freeze_path: Path,
    source_freeze_file_sha256: str,
) -> dict[str, Any]:
    if (
        source_freeze_path.expanduser().absolute() != DEFAULT_FREEZE
        or not panel3._is_sha256(source_freeze_file_sha256)
    ):
        raise RuntimeError("calibration source-freeze file hash is invalid")
    return {
        "schema_version": 1,
        "source_freeze_sha256": source_freeze["source_freeze_sha256"],
        "source_freeze_file_sha256": source_freeze_file_sha256,
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "analyzer_sha256": _sha256(freeze.ANALYZER),
        "protocol_sha256": _sha256(freeze.PROTOCOL),
        "candidate_contract_sha256": _sha256(freeze.CANDIDATE_CONTRACT),
        "arms": list(ARM_ORDER),
        "coordinate_count": 39,
        "result_count": 117,
    }


def spool_path(
    directory: Path,
    coordinate: dict[str, int],
    arm: str,
) -> Path:
    return directory / f"{worker_key(coordinate, arm)}.json"


def load_spool(
    path: Path,
    binding: dict[str, Any],
    coordinate: dict[str, int],
    arm: str,
    *,
    allowed_root: Path = ROOT,
) -> tuple[dict[str, Any], str, str]:
    try:
        encoded = common.secure_read_bytes(
            path,
            allowed_root=allowed_root,
        )
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=common._json_object,
            parse_float=common._json_float,
            parse_int=common._json_int,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (
        RuntimeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise RuntimeError(f"invalid calibration spool: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"calibration spool is not an object: {path}")
    common.verify_artifact_sha256(payload, "spool_record_sha256")
    expected_key = worker_key(coordinate, arm)
    if (
        set(payload)
        != {
            "schema_version",
            "name",
            "binding",
            "worker_key",
            "result_sha256",
            "result",
            "spool_record_sha256",
        }
        or payload["schema_version"] != 1
        or payload["name"]
        != "darkofit_panel3_cross_power_calibration_spool_v1"
        or payload["binding"] != binding
        or payload["worker_key"] != expected_key
        or payload["result_sha256"] != _json_sha256(payload["result"])
        or payload["result"].get("worker_key") != expected_key
        or payload["result"].get("coordinate") != coordinate
        or payload["result"].get("arm") != arm
    ):
        raise RuntimeError(f"calibration spool binding changed: {path}")
    return (
        payload["result"],
        payload["spool_record_sha256"],
        hashlib.sha256(encoded).hexdigest(),
    )


def create_spool(
    path: Path,
    binding: dict[str, Any],
    result: dict[str, Any],
) -> tuple[dict[str, Any], str, str]:
    payload = common.bind_artifact_sha256(
        {
            "schema_version": 1,
            "name": "darkofit_panel3_cross_power_calibration_spool_v1",
            "binding": binding,
            "worker_key": result["worker_key"],
            "result_sha256": _json_sha256(result),
            "result": result,
        },
        "spool_record_sha256",
    )
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    common.atomic_create(path, encoded)
    return (
        result,
        payload["spool_record_sha256"],
        hashlib.sha256(encoded).hexdigest(),
    )


def verify_spool_publication_snapshot(
    results: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    binding: dict[str, Any],
    *,
    spool_directory: Path = DEFAULT_SPOOL_DIRECTORY,
) -> None:
    expected = [
        (coordinate, arm)
        for coordinate in expected_coordinates()
        for arm in ARM_ORDER
    ]
    if len(results) != 117 or len(records) != 117:
        raise RuntimeError("calibration publication spool grid changed")
    for result, record, (coordinate, arm) in zip(
        results,
        records,
        expected,
        strict=True,
    ):
        path = spool_path(spool_directory, coordinate, arm)
        reopened, record_sha256, file_sha256 = load_spool(
            path,
            binding,
            coordinate,
            arm,
            allowed_root=spool_directory,
        )
        if (
            record
            != {
                "worker_key": worker_key(coordinate, arm),
                "path": str(path.relative_to(ROOT)),
                "file_sha256": file_sha256,
                "spool_record_sha256": record_sha256,
                "result_sha256": _json_sha256(result),
                "resumed": record.get("resumed"),
            }
            or type(record.get("resumed")) is not bool
            or reopened != result
        ):
            raise RuntimeError(
                "calibration publication spool changed: "
                f"{worker_key(coordinate, arm)}"
            )


def _worker_environment() -> dict[str, str]:
    environment = panel3.basketball.worker_environment(THREAD_COUNT)
    environment.update(
        {
            "DARKOFIT_WARMUP": "0",
            "CHIMERABOOST_WARMUP": "0",
            "OPENML_CACHE_DIRECTORY": str(Path.home() / ".cache" / "openml"),
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT), environment.get("PYTHONPATH", "")]
            ),
        }
    )
    return environment


def run_worker_subprocess(
    source_freeze_path: Path,
    coordinate: dict[str, int],
    arm: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--source-freeze",
        str(source_freeze_path),
        "--worker-task",
        str(coordinate["task_id"]),
        "--worker-repeat",
        str(coordinate["repeat"]),
        "--worker-fold",
        str(coordinate["fold"]),
        "--worker-sample",
        str(coordinate["sample"]),
        "--worker-arm",
        arm,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_worker_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "calibration worker failed for "
            f"{coordinate_key(**coordinate)}:\n{completed.stderr}"
        )
    lines = [
        line[len(WORKER_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if len(lines) != 1:
        raise RuntimeError("calibration worker protocol changed")
    payload = panel3._json_loads(lines[0], "calibration worker")
    if (
        not isinstance(payload, dict)
        or payload.get("arm") != arm
        or payload.get("coordinate") != coordinate
    ):
        raise RuntimeError("calibration worker result grid changed")
    return payload


def run_parent(
    *,
    source_freeze_path: Path = DEFAULT_FREEZE,
    output: Path = DEFAULT_OUTPUT,
    spool_directory: Path = DEFAULT_SPOOL_DIRECTORY,
) -> dict[str, Any]:
    common.validate_create_path(output)
    common.ensure_output_directory(spool_directory)
    source_freeze, runtime, source_freeze_file_sha256 = load_source_freeze(
        source_freeze_path
    )
    binding = spool_binding(
        source_freeze,
        source_freeze_path,
        source_freeze_file_sha256,
    )
    results: list[dict[str, Any]] = []
    spool_records = []
    for coordinate in expected_coordinates():
        existing: dict[str, tuple[dict[str, Any], str, str]] = {}
        missing = []
        for arm in ARM_ORDER:
            path = spool_path(spool_directory, coordinate, arm)
            if path.exists() or path.is_symlink():
                loaded = load_spool(
                    path, binding, coordinate, arm
                )
                validate_worker_result(
                    loaded[0],
                    source_freeze,
                    coordinate,
                    arm,
                )
                existing[arm] = loaded
            else:
                missing.append(arm)
        generated: dict[str, dict[str, Any]] = {}
        failures = []
        if missing:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(
                        run_worker_subprocess,
                        source_freeze_path,
                        coordinate,
                        arm,
                    ): arm
                    for arm in missing
                }
                for future in as_completed(futures):
                    arm = futures[future]
                    try:
                        result = future.result()
                        validate_worker_result(
                            result,
                            source_freeze,
                            coordinate,
                            arm,
                        )
                        path = spool_path(
                            spool_directory,
                            coordinate,
                            arm,
                        )
                        generated[arm], digest, file_sha256 = create_spool(
                            path,
                            binding,
                            result,
                        )
                        existing[arm] = (
                            generated[arm],
                            digest,
                            file_sha256,
                        )
                    except Exception as exc:  # noqa: BLE001
                        failures.append((arm, exc))
        if failures:
            details = "; ".join(
                f"{arm}: {type(exc).__name__}: {exc}"
                for arm, exc in sorted(failures)
            )
            raise RuntimeError(
                "calibration coordinate worker failure; successful arm "
                f"spools were preserved: {details}"
            )
        if set(existing) != set(ARM_ORDER):
            raise RuntimeError("calibration worker omitted an arm")
        for arm in ARM_ORDER:
            path = spool_path(spool_directory, coordinate, arm)
            resumed = arm not in generated
            result, digest, file_sha256 = existing[arm]
            results.append(result)
            spool_records.append(
                {
                    "worker_key": worker_key(coordinate, arm),
                    "path": str(path.relative_to(ROOT)),
                    "file_sha256": file_sha256,
                    "spool_record_sha256": digest,
                    "result_sha256": _json_sha256(result),
                    "resumed": resumed,
                }
            )
    if len(results) != 117 or len(spool_records) != 117:
        raise RuntimeError("calibration result grid is incomplete")
    validate_complete_results(results, source_freeze)
    final_freeze, final_runtime, final_freeze_file_sha256 = (
        load_source_freeze(source_freeze_path)
    )
    if (
        final_freeze != source_freeze
        or final_runtime != runtime
        or final_freeze_file_sha256 != source_freeze_file_sha256
    ):
        raise RuntimeError("calibration source or runtime changed during run")
    artifact = common.bind_artifact_sha256(
        {
            "schema_version": 1,
            "name": "darkofit_panel3_cross_power_calibration_raw_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_freeze_path": str(source_freeze_path.relative_to(ROOT)),
            "source_freeze_file_sha256": source_freeze_file_sha256,
            "source_freeze_sha256": source_freeze["source_freeze_sha256"],
            "runtime": runtime,
            "tasks": freeze.TASKS,
            "coordinates": expected_coordinates(),
            "arms": list(ARM_ORDER),
            "execution": {
                "kind": (
                    "coordinate_waves_three_concurrent_isolated_arm_processes"
                ),
                "concurrent_processes": 3,
                "worker_thread_count": THREAD_COUNT,
                "random_state": RANDOM_STATE,
                "timing_and_memory_claim_eligible": False,
            },
            "spool": {
                "directory": str(spool_directory.relative_to(ROOT)),
                "binding": binding,
                "record_count": len(spool_records),
                "resumed_record_count": sum(
                    bool(row["resumed"]) for row in spool_records
                ),
                "records": spool_records,
            },
            "results": results,
            "result_count": len(results),
            "all_results_preserved_without_filtering": True,
            "outcomes_scored": True,
            "analysis_performed": False,
            "development_only": True,
            "panel3_authorized": False,
            "default_promotion_authorized": False,
            "product_claim_authorized": False,
        },
        "raw_artifact_sha256",
    )
    encoded = (
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    final_freeze, final_runtime, final_freeze_file_sha256 = (
        load_source_freeze(source_freeze_path)
    )
    if (
        final_freeze != source_freeze
        or final_runtime != runtime
        or final_freeze_file_sha256 != source_freeze_file_sha256
    ):
        raise RuntimeError("calibration source changed before raw publish")
    verify_spool_publication_snapshot(
        results,
        spool_records,
        binding,
        spool_directory=spool_directory,
    )
    final_freeze, final_runtime, final_freeze_file_sha256 = (
        load_source_freeze(source_freeze_path)
    )
    if (
        final_freeze != source_freeze
        or final_runtime != runtime
        or final_freeze_file_sha256 != source_freeze_file_sha256
    ):
        raise RuntimeError("calibration source changed before raw publish")
    common.atomic_create(output, encoded)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--spool-directory",
        type=Path,
        default=DEFAULT_SPOOL_DIRECTORY,
    )
    parser.add_argument("--worker-task", type=int)
    parser.add_argument("--worker-repeat", type=int)
    parser.add_argument("--worker-fold", type=int)
    parser.add_argument("--worker-sample", type=int, default=0)
    parser.add_argument("--worker-arm", choices=ARM_ORDER)
    args = parser.parse_args(argv)
    worker_fields = (
        args.worker_task,
        args.worker_repeat,
        args.worker_fold,
        args.worker_arm,
    )
    if any(value is not None for value in worker_fields) and not all(
        value is not None for value in worker_fields
    ):
        parser.error("all calibration worker arguments must be supplied")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.source_freeze.expanduser().absolute() != DEFAULT_FREEZE:
        raise RuntimeError("calibration source-freeze path changed")
    if args.worker_task is not None:
        (
            source_freeze,
            worker_runtime,
            source_freeze_file_sha256,
        ) = load_source_freeze(args.source_freeze)
        coordinate = {
            "task_id": args.worker_task,
            "repeat": args.worker_repeat,
            "fold": args.worker_fold,
            "sample": args.worker_sample,
        }
        if coordinate not in expected_coordinates():
            raise RuntimeError("calibration worker coordinate is not frozen")
        results = evaluate_coordinate(
            source_freeze,
            **coordinate,
            arms=[args.worker_arm],
        )
        if len(results) != 1:
            raise RuntimeError("calibration worker returned multiple arms")
        validate_worker_result(
            results[0],
            source_freeze,
            coordinate,
            args.worker_arm,
        )
        final_freeze, final_runtime, final_file_sha256 = load_source_freeze(
            args.source_freeze
        )
        if (
            final_freeze != source_freeze
            or final_runtime != worker_runtime
            or final_file_sha256 != source_freeze_file_sha256
        ):
            raise RuntimeError(
                "calibration source changed during worker execution"
            )
        print(
            WORKER_PREFIX
            + json.dumps(results[0], sort_keys=True, allow_nan=False),
            flush=True,
        )
        return 0
    if (
        args.output.expanduser().absolute() != DEFAULT_OUTPUT
        or args.spool_directory.expanduser().absolute()
        != DEFAULT_SPOOL_DIRECTORY
    ):
        raise RuntimeError("calibration output path changed")
    artifact = run_parent(
        source_freeze_path=args.source_freeze,
        output=args.output,
        spool_directory=args.spool_directory,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "result_count": artifact["result_count"],
                "raw_artifact_sha256": artifact["raw_artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
