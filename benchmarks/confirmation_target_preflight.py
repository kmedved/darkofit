"""Prospective target-validity checks for confirmation registries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from benchmarks import build_ctr23_contamination_registry as ctr
from benchmarks import panel3_data_contract as data_contract


TARGET_POLICY = "numeric_float64_all_finite_v1"


class TargetPreflightError(RuntimeError):
    """Raised when a prospective confirmation target is not eligible."""


def validate_finite_regression_target(
    target: Any,
    *,
    expected_rows: int,
) -> dict[str, Any]:
    """Return a value-free eligibility attestation for a regression target."""
    if type(expected_rows) is not int or expected_rows <= 0:
        raise ValueError("expected_rows must be a positive integer")
    if np.ma.isMaskedArray(target):
        raise TargetPreflightError("target must not be a masked array")
    try:
        values = np.asarray(target)
    except (TypeError, ValueError) as exc:
        raise TargetPreflightError(
            "target must be a one-dimensional numeric vector"
        ) from exc
    if values.ndim != 1 or values.size == 0:
        raise TargetPreflightError(
            "target must be a nonempty one-dimensional vector"
        )
    if values.shape[0] != expected_rows:
        raise TargetPreflightError(
            "target row count differs from the bound dataset"
        )
    if np.iscomplexobj(values) or (
        values.dtype.kind == "O"
        and any(
            isinstance(value, (complex, np.complexfloating))
            for value in values
        )
    ):
        raise TargetPreflightError("target must not contain complex values")
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TargetPreflightError(
            "target must be convertible to float64"
        ) from exc
    if not np.isfinite(numeric).all():
        raise TargetPreflightError("target must contain only finite values")
    return {
        "policy": TARGET_POLICY,
        "checked": True,
        "passed": True,
        "target_outcome_statistics_computed": False,
        "target_values_persisted": False,
    }


def attest_openml_target(
    task_record: Mapping[str, Any],
    *,
    openml_module=None,
) -> dict[str, Any]:
    """Reload and bind an OpenML target before campaign authorization."""
    required = {
        "openml_task_id",
        "openml_dataset_id",
        "openml_dataset_version",
        "dataset_name",
        "target_name",
        "dataset_default_target_attribute",
        "openml_task_type_id",
        "fingerprint",
    }
    if not isinstance(task_record, Mapping) or not required.issubset(task_record):
        raise ValueError("task_record is missing target-preflight bindings")
    if openml_module is None:
        try:
            import openml as openml_module
        except ImportError as exc:  # pragma: no cover - CLI dependency error.
            raise RuntimeError(
                "openml is required for target preflight"
            ) from exc

    task_id = int(task_record["openml_task_id"])
    task = openml_module.tasks.get_task(
        task_id,
        download_splits=False,
        download_data=False,
        download_qualities=False,
        download_features_meta_data=False,
    )
    dataset_id = int(task_record["openml_dataset_id"])
    if int(task.dataset_id) != dataset_id:
        raise TargetPreflightError(
            f"task {task_id} dataset metadata drifted"
        )
    dataset = openml_module.datasets.get_dataset(
        dataset_id,
        download_data=False,
        download_qualities=False,
    )
    task_type = getattr(task.task_type_id, "value", task.task_type_id)
    metadata_matches = (
        str(task.target_name) == str(task_record["target_name"])
        and str(dataset.default_target_attribute)
        == str(task_record["dataset_default_target_attribute"])
        and str(dataset.name) == str(task_record["dataset_name"])
        and int(dataset.version) == int(task_record["openml_dataset_version"])
        and int(task_type) == int(task_record["openml_task_type_id"]) == 2
    )
    if not metadata_matches:
        raise TargetPreflightError(
            f"task {task_id} target metadata drifted"
        )
    X, target, _categorical, _names = dataset.get_data(
        target=task.target_name,
        include_row_id=False,
        include_ignore_attribute=False,
        dataset_format="dataframe",
    )
    if target is None:
        raise TargetPreflightError(f"task {task_id} returned no target")
    attestation = validate_finite_regression_target(
        target,
        expected_rows=int(task_record["fingerprint"]["n_rows"]),
    )
    fingerprint = ctr.dataset_fingerprint(X, target)
    if fingerprint != task_record["fingerprint"]:
        raise TargetPreflightError(
            f"task {task_id} dataset fingerprint drifted"
        )
    numeric_target = pd.to_numeric(target, errors="raise").astype(np.float64)
    return {
        **attestation,
        "binding": {
            "openml_task_id": task_id,
            "openml_dataset_id": dataset_id,
            "target_name": str(task.target_name),
            "dataset_fingerprint_sha256": ctr.sha256_json(fingerprint),
            "ordered_task_view_sha256": (
                data_contract.ordered_task_view_sha256(
                    X,
                    numeric_target,
                )
            ),
        },
    }
