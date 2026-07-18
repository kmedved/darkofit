import copy

import numpy as np
import pytest

from benchmarks import run_smooth_cross_features as experiment


def test_candidate_pairs_are_deterministic_and_skip_categoricals():
    pairs = experiment.candidate_pairs(
        [0.1, 0.5, 0.5, 0.2], [1], n_features=4
    )
    assert pairs == [
        (2, 3, "diff"),
        (2, 3, "prod"),
        (2, 0, "diff"),
        (2, 0, "prod"),
        (3, 0, "diff"),
        (3, 0, "prod"),
    ]
    assert all(1 not in pair[:2] for pair in pairs)


def test_augmentation_appends_diff_and_product_with_nan_propagation():
    X = np.array([[2.0, 3.0], [np.nan, 4.0]])
    augmented = experiment.augment_numeric_crosses(
        X, [(0, 1, "diff"), (0, 1, "prod")]
    )
    np.testing.assert_array_equal(augmented[0], [2.0, 3.0, -1.0, 6.0])
    assert np.isnan(augmented[1, 2:]).all()


def _fake_rows():
    rows = []
    for task_id, name in experiment.TASKS.items():
        for fold in experiment.FOLDS:
            ratio = 0.98 if task_id != 361623 else 1.01
            rows.append(
                {
                    "task_id": task_id,
                    "dataset_name": name,
                    "fold": fold,
                    "base_linear_selected": fold % 2 == 0,
                    "cross_selected": fold % 3 == 0,
                    "base": {"test_rmse": 1.0},
                    "selected": {"test_rmse": ratio},
                    "external_native_exact": True,
                    "darko_total_fit_seconds": 1.0,
                    "chimera_total_fit_seconds": 1.0,
                }
            )
    return rows


def test_analysis_reports_magnitude_concentration_and_harm_without_win_gate():
    analysis = experiment.analyze(_fake_rows())
    assert analysis["coordinate_count"] == 21
    assert analysis["external_native_exact"] is True
    assert analysis["fresh_claim_eligible"] is False
    assert analysis["worst_dataset_ratio"] == pytest.approx(1.01)
    assert set(analysis["leave_one_out_equal_dataset_ratios"]) == set(
        experiment.TASKS.values()
    )
    assert "wins" not in analysis


def test_analysis_rejects_inexact_or_incomplete_artifacts():
    rows = _fake_rows()
    rows[0]["external_native_exact"] = False
    with pytest.raises(RuntimeError, match="parity"):
        experiment.analyze(rows)
    incomplete = copy.deepcopy(_fake_rows()[:-1])
    with pytest.raises(RuntimeError, match="incomplete"):
        experiment.analyze(incomplete)
