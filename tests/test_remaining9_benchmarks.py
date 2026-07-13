import pandas as pd
import pytest

from benchmarks.analyze_tabarena_regression_remaining9 import analyze_rows
from benchmarks.preflight_hotpaths import _speedup, _timing_summary
from benchmarks.preprocessing_instrumentation import (
    capture_preprocessing,
    instrument_feature_preprocessors,
)
from benchmarks.run_tabarena_regression_remaining9 import (
    EXPECTED_DATASET_SPLITS,
    EXPECTED_JOBS,
    FROZEN_CANDIDATE,
    TASK_SPLIT_COUNTS,
    validate_chimera_coverage,
)
from benchmarks.run_tabarena_same_machine_performance import (
    EXPECTED_JOBS as EXPECTED_PERFORMANCE_JOBS,
    EXPECTED_REGISTERED_ROWS,
    FROZEN_CHIMERA_COMMIT,
    FROZEN_CHIMERA_VERSION,
    REGISTERED_FOLDS,
    SPLIT_INDICES as PERFORMANCE_SPLITS,
    TIME_LIMIT_SECONDS,
    resolve_chimera_repo,
    validate_registered_splits,
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


def test_same_machine_performance_protocol_is_frozen():
    assert PERFORMANCE_SPLITS == ["r0f0", "r1f1", "r2f2"]
    assert REGISTERED_FOLDS == [0, 4, 8]
    assert TIME_LIMIT_SECONDS == 3600
    assert EXPECTED_REGISTERED_ROWS == 27
    assert EXPECTED_PERFORMANCE_JOBS == 81
    assert FROZEN_CHIMERA_VERSION == "0.14.1"
    assert FROZEN_CHIMERA_COMMIT == "07995af9e2b6212a41975a49931ee20af8f2cc14"


def test_same_machine_registered_split_validation_is_exact():
    rows = [
        {
            "dataset": dataset,
            "method": "CHIMERA (default)",
            "fold": fold,
            "imputed": False,
            "problem_type": "regression",
            "metric": "rmse",
        }
        for dataset in TASK_SPLIT_COUNTS
        for fold in REGISTERED_FOLDS
    ]
    validate_registered_splits(rows)

    rows[-1]["imputed"] = True
    with pytest.raises(RuntimeError, match="imputed"):
        validate_registered_splits(rows)


def test_same_machine_sibling_chimera_discovery(tmp_path):
    darkofit_repo = tmp_path / "darkofit"
    chimera_repo = tmp_path / "chimeraboost"
    darkofit_repo.mkdir()
    (chimera_repo / "chimeraboost").mkdir(parents=True)
    (chimera_repo / "chimeraboost" / "__init__.py").write_text("")
    (chimera_repo / ".git").mkdir()

    assert resolve_chimera_repo(darkofit_repo=darkofit_repo) == chimera_repo


def test_preprocessing_instrumentation_accumulates_only_active_package():
    class DarkPreprocessor:
        def fit_transform(self, value):
            return value + 1

    class ChimeraPreprocessor:
        def fit_transform(self, value):
            return value * 2

    original_dark = DarkPreprocessor.fit_transform
    times = iter([0.0, 0.1, 0.1, 0.3, 0.3, 0.7])
    with instrument_feature_preprocessors(
        {"darkofit": DarkPreprocessor, "chimeraboost": ChimeraPreprocessor},
        clock=lambda: next(times),
    ):
        with capture_preprocessing("darkofit") as captured:
            assert DarkPreprocessor().fit_transform(1) == 2
            assert DarkPreprocessor().fit_transform(2) == 3
            assert ChimeraPreprocessor().fit_transform(3) == 6
        assert captured["calls"] == 2
        assert captured["seconds"] == pytest.approx(0.3)

    assert DarkPreprocessor.fit_transform is original_dark
