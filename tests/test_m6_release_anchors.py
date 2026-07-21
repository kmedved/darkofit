"""Fail-closed tests for the M6 release-anchor campaign."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from run_m6_release_anchors import (  # noqa: E402
    _worker_environment,
    _write_create_only,
    _catboost_frames,
    _numeric_metric_values,
    _assert_module_under,
    expected_coordinates,
    main as main_release_anchors,
    run as run_release_anchors,
    source_state,
    validate_rows,
    validate_sources,
)


def test_terminal_m6_v3_release_anchor_runner_refuses_new_execution():
    with pytest.raises(RuntimeError, match="terminal"):
        run_release_anchors(SimpleNamespace())


def test_terminal_m6_v3_release_anchor_worker_cli_refuses_new_execution():
    with pytest.raises(RuntimeError, match="terminal"):
        main_release_anchors(["--worker", "/nonexistent/payload.json"])


def test_release_anchor_grid_includes_full_small_medium_weighted_slice():
    full = expected_coordinates(smoke=False)
    smoke = expected_coordinates(smoke=True)

    assert len(full) == 240
    assert len(set(full)) == 240
    assert {row[2] for row in full} == {"small", "medium"}
    assert {row[4] for row in full} == {"none", "stress"}
    assert len(smoke) == 6


def test_metric_validation_excludes_the_primary_metric_name():
    values = _numeric_metric_values(
        {
            "primary_metric": "log_loss",
            "primary_value": 0.4,
            "accuracy": 0.8,
        }
    )

    assert np.array_equal(values, np.asarray([0.4, 0.8]))


def test_catboost_transport_preserves_numeric_columns_and_tokens_categories():
    X_train = np.asarray(
        [["a", 1.5], [None, np.nan], ["b", -2.0]], dtype=object
    )
    X_test = np.asarray([["b", 3.0], [None, 4.0]], dtype=object)

    train, test = _catboost_frames(X_train, X_test, [0])

    assert isinstance(train, pd.DataFrame)
    assert train[1].dtype.kind == "f"
    assert test[1].dtype.kind == "f"
    assert train.iloc[0, 0] == "__DARKOFIT_CATEGORY_0__"
    assert test.iloc[0, 0] == "__DARKOFIT_CATEGORY_1__"
    assert train.iloc[1, 0] == "__DARKOFIT_MISSING_CATEGORY__"
    assert test.iloc[1, 0] == "__DARKOFIT_MISSING_CATEGORY__"


def test_release_anchor_source_validation_checks_exact_pins():
    harness = {"clean": True}
    chimera = {
        "clean": True,
        "head": "f14be606b641f1bf0dc92bb14b3951f1fe631c6b",
    }
    catboost = {
        "version": "1.2.10",
        "record_sha256": (
            "9c20fb35750d9ff814309323b225e836b"
            "538c1496745f357c8fd50187e7824ed"
        ),
    }

    validate_sources(harness, chimera, catboost)

    with pytest.raises(RuntimeError, match="expected"):
        validate_sources(harness, {**chimera, "head": "0" * 40}, catboost)


def test_release_anchor_source_state_rejects_a_git_subdirectory():
    with pytest.raises(RuntimeError, match="source checkout"):
        source_state(BENCH_DIR, package="chimeraboost")


def test_release_anchor_workers_drop_inherited_pythonpath(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("PYTHONPATH", "/wrong/repository")

    environment = _worker_environment(tmp_path)

    assert "PYTHONPATH" not in environment


def test_release_anchor_import_binding_rejects_shadowed_module(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    expected_module = source / "chimeraboost" / "__init__.py"
    expected_module.parent.mkdir()
    expected_module.touch()

    assert _assert_module_under(
        SimpleNamespace(__file__=str(expected_module)),
        source,
        "chimeraboost",
    ) == str(expected_module)
    with pytest.raises(RuntimeError, match="not"):
        _assert_module_under(
            SimpleNamespace(__file__=str(tmp_path / "shadow.py")),
            source,
            "chimeraboost",
        )


def test_release_anchor_create_only_write_removes_partial_output(
    monkeypatch, tmp_path
):
    output = tmp_path / "partial.json"

    def fail_fsync(_descriptor):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="simulated fsync failure"):
        _write_create_only(output, b"partial")

    assert not output.exists()


def test_release_anchor_rows_reject_duplicate_or_nonpositive_resources():
    coordinates = expected_coordinates(smoke=True)
    rows = []
    for anchor, dataset, size, seed, weight in coordinates:
        task = {
            "friedman_numeric": "regression",
            "numeric_binary": "binary",
            "categorical_binary": "binary",
        }[dataset]
        primary = (
            "weighted_rmse"
            if task == "regression" and weight == "stress"
            else (
                "rmse"
                if task == "regression"
                else (
                    "weighted_log_loss" if weight == "stress" else "log_loss"
                )
            )
        )
        metrics = {
            "primary_metric": primary,
            "primary_value": 0.5,
            primary: 0.5,
        }
        rows.append({
            "status": "ok",
            "anchor_id": anchor,
            "dataset": dataset,
            "dataset_sha256": "a" * 64,
            "size": size,
            "seed": seed,
            "weight_mode": weight,
            "task": task,
            "n_train": 100,
            "n_test": 25,
            "n_features": 4,
            "fit_seconds": 1.0,
            "predict_seconds": 0.1,
            "worker_peak_rss_bytes": 1000,
            "prediction_sha256": "b" * 64,
            "probability_sha256": None if task == "regression" else "c" * 64,
            "metrics": metrics,
        })
    validate_rows(rows, smoke=True)

    mismatched = [dict(row) for row in rows]
    mismatched[1]["dataset_sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="differs on dataset_sha256"):
        validate_rows(mismatched, smoke=True)

    rows[-1] = dict(rows[0])
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_rows(rows, smoke=True)
