from __future__ import annotations

import copy

import numpy as np
import pytest

from benchmarks import benchmark_adapters
from benchmarks import run_accuracy_v2_ablation as bench
from darkofit import DarkoRegressor


def test_accuracy_v2_grid_and_components_are_exact():
    assert bench.DATASETS == (
        "diabetes_resampled",
        "friedman_numeric",
        "wide_numeric_reg",
        "categorical_reg",
    )
    assert len(bench.expected_cell_keys()) == 24
    assert bench.HORIZONS == (1_000, 10_000)
    assert bench.CROSS_GUARD_RATIO == 0.95
    assert bench.ARMS == ("a1", "a1_cross", "a10", "a10_cross")


def test_cross_augmentation_preserves_original_columns_and_values():
    X = np.array([[1.0, 3.0], [2.0, 5.0]])
    out = bench.augment_crosses(X, [(0, 1, "diff"), (0, 1, "prod")])
    np.testing.assert_array_equal(out[:, :2], X)
    np.testing.assert_array_equal(out[:, 2], [-2.0, -3.0])
    np.testing.assert_array_equal(out[:, 3], [3.0, 10.0])


def _rows(values, *, engaged=True):
    rows = []
    for dataset, seed, weight_mode in bench.expected_cell_keys():
        for arm in bench.ARMS:
            rows.append({
                "dataset": dataset,
                "seed": seed,
                "weight_mode": weight_mode,
                "arm": arm,
                "primary_metric": "rmse",
                "primary_value": values[arm],
                "fit_seconds": 1.0,
                "predict_seconds": 1.0,
                "engaged": bool(engaged and arm.endswith("_cross")),
            })
    return rows


def test_analysis_selects_v2_only_for_bounded_strict_a10_cross_gain():
    summary = bench.analyze_rows(_rows({
        "a1": 1.02,
        "a1_cross": 1.00,
        "a10": 1.00,
        "a10_cross": 0.98,
    }))
    assert summary["selected_accuracy_profile"] == "accuracy_v2"
    assert summary["public_profile_changed"] is True


def test_analysis_keeps_a10_for_no_engagement_or_regression():
    no_engagement = bench.analyze_rows(_rows({
        "a1": 1.02,
        "a1_cross": 1.02,
        "a10": 1.00,
        "a10_cross": 1.00,
    }, engaged=False))
    regression = bench.analyze_rows(_rows({
        "a1": 1.02,
        "a1_cross": 1.02,
        "a10": 1.00,
        "a10_cross": 1.001,
    }))
    assert no_engagement["selected_accuracy_profile"] == "accuracy"
    assert regression["selected_accuracy_profile"] == "accuracy"


def test_analysis_rejects_missing_or_duplicate_rows():
    rows = _rows({
        "a1": 1.02,
        "a1_cross": 1.00,
        "a10": 1.00,
        "a10_cross": 0.98,
    })
    with pytest.raises(RuntimeError, match="exact grid"):
        bench.analyze_rows(rows[:-1])
    duplicate = copy.deepcopy(rows)
    duplicate.append(copy.deepcopy(rows[0]))
    with pytest.raises(RuntimeError, match="duplicate"):
        bench.analyze_rows(duplicate)


def test_outputs_are_create_only(tmp_path, monkeypatch):
    raw = tmp_path / "raw.json"
    result = tmp_path / "result.md"
    raw.write_text("existing")
    args = bench.parse_args([
        "--expected-source-sha",
        "a" * 40,
        "--raw-output",
        str(raw),
        "--result-output",
        str(result),
    ])
    monkeypatch.setattr(bench, "source_state", lambda expected: pytest.fail(
        "source must not be inspected after an output collision"
    ))
    with pytest.raises(FileExistsError, match="create-only"):
        bench.run(args)


def test_horizon_worker_has_exact_decline_fallback(monkeypatch):
    def tiny_model(horizon, *, seed):
        return DarkoRegressor(
            iterations=4,
            learning_rate=0.1,
            depth=2,
            tree_mode="catboost",
            linear_leaves=False,
            early_stopping=True,
            early_stopping_rounds=2,
            use_best_model=True,
            refit=False,
            random_state=seed,
            thread_count=1,
            diagnostic_warnings="never",
        )

    monkeypatch.setattr(bench, "_model", tiny_model)
    monkeypatch.setattr(bench, "CROSS_GUARD_RATIO", 0.0)
    spec, X, y, cat_features = benchmark_adapters.build_dataset(
        "friedman_numeric", "tiny", 0
    )
    split = benchmark_adapters.split_case(X, y, spec.task, 0)
    records = bench.run_horizon(split, cat_features, horizon=1_000, seed=0)
    assert records["guarded"]["engaged"] is False
    assert records["guarded"]["fallback_prediction_exact"] is True
    np.testing.assert_array_equal(
        records["guarded"]["prediction"], records["base"]["prediction"]
    )
