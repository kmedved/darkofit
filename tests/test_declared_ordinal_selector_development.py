from __future__ import annotations

import numpy as np

from benchmarks import run_declared_ordinal_selector_development as bench


def test_split_positions_are_complete_and_disjoint():
    train, validation, test = bench._split_positions(1_503, 4)
    combined = np.concatenate([train, validation, test])
    assert len(train) == 961
    assert len(validation) == 241
    assert len(test) == 301
    assert np.array_equal(np.sort(combined), np.arange(1_503))
    assert not np.intersect1d(train, validation).size
    assert not np.intersect1d(train, test).size
    assert not np.intersect1d(validation, test).size


def test_summary_is_equal_dataset_and_preserves_worst_coordinate():
    rows = [
        {
            "dataset": "a",
            "selector_over_native_ratio": 0.8,
            "forced_over_native_ratio": 0.7,
            "selector_selected": True,
            "selector_final_exact": True,
        },
        {
            "dataset": "a",
            "selector_over_native_ratio": 1.0,
            "forced_over_native_ratio": 0.9,
            "selector_selected": False,
            "selector_final_exact": True,
        },
        {
            "dataset": "b",
            "selector_over_native_ratio": 0.9,
            "forced_over_native_ratio": 0.8,
            "selector_selected": True,
            "selector_final_exact": True,
        },
    ]
    summary = bench._summarize(rows)
    assert np.isclose(
        summary["equal_dataset_selector_over_native_ratio"],
        np.sqrt(np.sqrt(0.8) * 0.9),
    )
    assert summary["worst_coordinate_selector_over_native_ratio"] == 1.0
    assert summary["engagements"] == 2
    assert summary["all_final_refits_exact"] is True
