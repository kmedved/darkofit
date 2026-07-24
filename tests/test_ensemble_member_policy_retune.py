from __future__ import annotations

import copy
import subprocess
import sys

import numpy as np
import pytest

from benchmarks import benchmark_adapters
from benchmarks import run_ensemble_member_policy_retune as bench


def test_retune_grid_and_recipes_are_exact():
    assert len(bench.expected_cell_keys()) == 60
    assert len(set(bench.expected_cell_keys())) == 60
    assert bench.ARMS["current"] == {
        "ensemble_member_learning_rate": "policy",
        "ensemble_member_colsample": "policy",
    }
    assert bench.ARMS["intermediate"] == {
        "ensemble_member_learning_rate": 0.125,
        "ensemble_member_colsample": 0.925,
    }
    assert bench.ARMS["legacy_auto"] == {
        "ensemble_member_learning_rate": None,
        "ensemble_member_colsample": 1.0,
    }
    assert bench.arm_params("single").get("n_ensembles") is None
    assert bench.arm_params("current")["ensemble_parallelism"] == "sequential"


def _rows(ratios):
    rows = []
    for dataset, seed, weight_mode in bench.expected_cell_keys():
        for arm in bench.ARMS:
            rows.append({
                "dataset": dataset,
                "seed": seed,
                "weight_mode": weight_mode,
                "arm": arm,
                "primary_metric": "rmse",
                "primary_value": ratios[arm],
                "fit_seconds": 1.0,
                "predict_seconds": 1.0,
                "archive_bytes": 100,
            })
    return rows


def test_analysis_selects_improving_bounded_intermediate():
    result = bench.analyze_rows(_rows({
        "single": 1.05,
        "legacy_auto": 1.01,
        "intermediate": 0.99,
        "current": 1.0,
    }))
    assert result["selected_recipe"] == "intermediate"
    assert result["policy_changed"] is True
    assert result["comparisons"]["intermediate"]["vs_current"]["disposition"] == (
        "advance"
    )


def test_analysis_keeps_current_when_alternatives_do_not_improve():
    result = bench.analyze_rows(_rows({
        "single": 1.05,
        "legacy_auto": 1.01,
        "intermediate": 1.001,
        "current": 1.0,
    }))
    assert result["selected_recipe"] == "current"
    assert result["policy_changed"] is False


def test_analysis_rejects_missing_or_duplicate_rows():
    rows = _rows({
        "single": 1.05,
        "legacy_auto": 1.01,
        "intermediate": 0.99,
        "current": 1.0,
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


def test_current_recipe_worker_records_resolved_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "ITERATIONS", 4)
    monkeypatch.setattr(bench, "PATIENCE", 2)
    spec, X, y, cat_features = benchmark_adapters.build_dataset(
        "friedman_numeric", "tiny", 0
    )
    split = benchmark_adapters.split_case(X, y, spec.task, 0)
    row = bench._fit_one(
        spec,
        split,
        cat_features,
        "current",
        0,
        tmp_path / "current.npz",
    )
    assert row["resolved_member_policy"] == {
        "learning_rate": 0.15,
        "colsample": 0.85,
    }
    assert len(row["member_tree_counts"]) == 8
    assert np.isfinite(row["primary_value"])


def test_runner_script_mode_can_import_darkofit():
    command = (
        "import importlib.util, pathlib; "
        "p=pathlib.Path('benchmarks/run_ensemble_member_policy_retune.py'); "
        "s=importlib.util.spec_from_file_location('member_retune_script', p); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "import darkofit; print(darkofit.__file__)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=bench.ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert str(bench.ROOT / "darkofit") in completed.stdout
