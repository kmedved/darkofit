from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmarks import confirmation_target_preflight as preflight


@pytest.mark.parametrize(
    "target",
    [
        [1, 2],
        np.array([1.5, -2.0]),
        np.array(["1.5", "-2"], dtype=object),
        pd.Series([1.5, -2.0], dtype="Float64"),
    ],
)
def test_finite_numeric_targets_receive_value_free_attestation(target):
    result = preflight.validate_finite_regression_target(
        target,
        expected_rows=2,
    )

    assert result == {
        "policy": "numeric_float64_all_finite_v1",
        "checked": True,
        "passed": True,
        "target_outcome_statistics_computed": False,
        "target_values_persisted": False,
    }


@pytest.mark.parametrize(
    "target",
    [
        [],
        [[1.0], [2.0]],
        [np.nan, 1.0],
        [np.inf, 1.0],
        [-np.inf, 1.0],
        [pd.NA, 1.0],
        ["not-numeric", "1.0"],
        [1.0 + 2.0j, 3.0],
        np.array([np.complex64(1.0 + 2.0j), 3.0]),
        np.ma.array([1.0, 2.0], mask=[False, True]),
        ["1e9999", "1.0"],
    ],
)
def test_invalid_targets_fail_without_returning_outcome_details(target):
    with pytest.raises(preflight.TargetPreflightError):
        preflight.validate_finite_regression_target(
            target,
            expected_rows=2,
        )


def test_target_preflight_rejects_row_binding_mismatch():
    with pytest.raises(
        preflight.TargetPreflightError,
        match="row count differs",
    ):
        preflight.validate_finite_regression_target(
            [1.0, 2.0],
            expected_rows=3,
        )


class _FakeDataset:
    def __init__(self, X, target):
        self.name = "prospective-regression"
        self.version = 7
        self.default_target_attribute = "score"
        self._X = X
        self._target = target

    def get_data(self, **kwargs):
        assert kwargs == {
            "target": "score",
            "include_row_id": False,
            "include_ignore_attribute": False,
            "dataset_format": "dataframe",
        }
        return self._X, self._target, None, list(self._X.columns)


def _fake_openml(X, target):
    task = SimpleNamespace(
        dataset_id=321,
        target_name="score",
        task_type_id=SimpleNamespace(value=2),
    )
    dataset = _FakeDataset(X, target)
    return SimpleNamespace(
        tasks=SimpleNamespace(get_task=lambda task_id, **kwargs: task),
        datasets=SimpleNamespace(
            get_dataset=lambda dataset_id, **kwargs: dataset
        ),
    )


def _task_record(X, target):
    return {
        "openml_task_id": 123,
        "openml_dataset_id": 321,
        "openml_dataset_version": 7,
        "dataset_name": "prospective-regression",
        "target_name": "score",
        "dataset_default_target_attribute": "score",
        "openml_task_type_id": 2,
        "fingerprint": preflight.ctr.dataset_fingerprint(X, target),
    }


def test_openml_attestation_is_bound_to_exact_dataset_fingerprint():
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    target = pd.Series([2.0, 4.0, 6.0], name="score")
    record = _task_record(X, target)

    result = preflight.attest_openml_target(
        record,
        openml_module=_fake_openml(X, target),
    )

    assert result["passed"] is True
    assert result["binding"] == {
        "openml_task_id": 123,
        "openml_dataset_id": 321,
        "target_name": "score",
        "dataset_fingerprint_sha256": preflight.ctr.sha256_json(
            record["fingerprint"]
        ),
    }
    assert set(result) == {
        "policy",
        "checked",
        "passed",
        "target_outcome_statistics_computed",
        "target_values_persisted",
        "binding",
    }


def test_openml_attestation_fails_on_fingerprint_drift():
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    target = pd.Series([2.0, 4.0, 6.0], name="score")
    record = _task_record(X, target)
    changed = target.copy()
    changed.iloc[0] = 99.0

    with pytest.raises(
        preflight.TargetPreflightError,
        match="fingerprint drifted",
    ):
        preflight.attest_openml_target(
            record,
            openml_module=_fake_openml(X, changed),
        )


def test_openml_attestation_normalizes_complex_target_failure():
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    finite = pd.Series([2.0, 4.0, 6.0], name="score")
    complex_target = pd.Series(
        [2.0 + 1.0j, 4.0, 6.0],
        name="score",
    )

    with pytest.raises(
        preflight.TargetPreflightError,
        match="complex",
    ):
        preflight.attest_openml_target(
            _task_record(X, finite),
            openml_module=_fake_openml(X, complex_target),
        )
