"""Fail-closed tests for the M6 release-anchor campaign."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from run_m6_release_anchors import (  # noqa: E402
    _catboost_frames,
    _numeric_metric_values,
    expected_coordinates,
    validate_rows,
    validate_sources,
)


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


def test_release_anchor_rows_reject_duplicate_or_nonpositive_resources():
    coordinates = expected_coordinates(smoke=True)
    rows = [
        {
            "status": "ok",
            "anchor_id": anchor,
            "dataset": dataset,
            "size": size,
            "seed": seed,
            "weight_mode": weight,
            "fit_seconds": 1.0,
            "predict_seconds": 0.1,
            "worker_peak_rss_bytes": 1000,
        }
        for anchor, dataset, size, seed, weight in coordinates
    ]
    validate_rows(rows, smoke=True)

    rows[-1] = dict(rows[0])
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_rows(rows, smoke=True)
