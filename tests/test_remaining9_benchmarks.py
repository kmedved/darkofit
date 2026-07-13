import pandas as pd
import pytest

from benchmarks.analyze_tabarena_regression_remaining9 import analyze_rows
from benchmarks.preflight_hotpaths import _speedup, _timing_summary
from benchmarks.run_tabarena_regression_remaining9 import (
    EXPECTED_DATASET_SPLITS,
    EXPECTED_JOBS,
    FROZEN_CANDIDATE,
    TASK_SPLIT_COUNTS,
    validate_chimera_coverage,
)


def test_remaining9_frozen_matrix_and_registered_coverage():
    assert EXPECTED_DATASET_SPLITS == 165
    assert EXPECTED_JOBS == 330
    assert FROZEN_CANDIDATE == {
        "l2_leaf_reg": 1.0,
        "max_bins": 128,
        "learning_rate": 0.1,
        "ts_permutations": 1,
    }
    rows = []
    for dataset, (_, split_count) in TASK_SPLIT_COUNTS.items():
        rows.extend(
            {
                "dataset": dataset,
                "method": "CHIMERA (default)",
                "fold": fold,
                "imputed": False,
            }
            for fold in range(split_count)
        )
    validate_chimera_coverage(pd.DataFrame(rows))
    rows[-1]["imputed"] = True
    with pytest.raises(RuntimeError, match="imputed"):
        validate_chimera_coverage(pd.DataFrame(rows))


def test_preflight_timing_helpers():
    optimized = _timing_summary([1.0, 2.0, 3.0])
    reference = _timing_summary([2.0, 4.0, 6.0])
    assert optimized["median_seconds"] == 2.0
    assert optimized["iqr_seconds"] == 1.0
    assert optimized["iqr_fraction"] == 0.5
    assert _speedup(optimized, reference) == 2.0


def test_remaining9_analysis_applies_equal_dataset_and_repeat_gates():
    tasks = {"small_a": (1, 9), "small_b": (2, 9)}
    local = []
    chimera = []
    for dataset, (task_id, split_count) in tasks.items():
        for registered_fold in range(split_count):
            repeat, fold = divmod(registered_fold, 3)
            chimera.append(
                {
                    "dataset": dataset,
                    "repeat": repeat,
                    "fold": fold,
                    "registered_fold": registered_fold,
                    "rmse": 1.1,
                    "val_rmse": 1.1,
                }
            )
            for config, rmse in (("default", 1.0), ("candidate", 0.99)):
                local.append(
                    {
                        "dataset": dataset,
                        "task_id": task_id,
                        "repeat": repeat,
                        "fold": fold,
                        "registered_fold": registered_fold,
                        "config": config,
                        "rmse": rmse,
                        "val_rmse": rmse,
                        "train_time_s": 1.0 if config == "default" else 0.9,
                        "infer_time_s": 1.0 if config == "default" else 0.9,
                        "peak_memory_bytes": 100.0,
                    }
                )
    tidy, summary = analyze_rows(local, chimera, task_split_counts=tasks)
    assert len(tidy) == 18
    assert summary["equal_dataset"]["candidate_default_rmse"]["ratio"] == pytest.approx(
        0.99
    )
    assert summary["gates"]["advance"] is True
    assert all(item["repeat_wins"] == 3 for item in summary["datasets"])
