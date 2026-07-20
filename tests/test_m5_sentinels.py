"""Contract and analyzer tests for the M5 diversity sentinels."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from run_m5_sentinels import (  # noqa: E402
    _quality_baseline,
    analyze_rows,
    validate_sources,
)
from standing_evidence import m5_expected_grid  # noqa: E402


def test_quality_baseline_handles_weighted_regression_and_classification():
    regression = _quality_baseline(
        "regression",
        np.asarray([0.0, 10.0]),
        np.asarray([0.0, 10.0]),
        np.asarray([9.0, 1.0]),
        np.asarray([1.0, 9.0]),
    )
    classification = _quality_baseline(
        "binary",
        np.asarray([0, 0, 1]),
        np.asarray([0, 1]),
        None,
        None,
    )

    assert regression > 0.0
    assert classification > 0.0


def test_m5_source_validation_requires_exact_control_and_baseline_parity():
    harness = {"clean": True}
    control = {
        "clean": True,
        "head": "726e5d8e6131c580bce948db833a5007d0692dca",
        "package_tree": "same",
    }
    candidate = {
        "clean": True,
        "head": "candidate",
        "package_tree": "same",
    }

    validate_sources(harness, control, candidate, establishing=True)

    with pytest.raises(RuntimeError, match="behavior-identical"):
        validate_sources(
            harness,
            control,
            {**candidate, "package_tree": "different"},
            establishing=True,
        )


def _row(identity):
    arm, domain, seed = identity
    canary = domain in {
        "binary_classification",
        "multiclass_classification",
    }
    return {
        "status": "ok",
        "arm": arm,
        "domain_id": domain,
        "seed": seed,
        "dataset_sha256": f"data-{domain}-{seed}",
        "split_sha256": f"split-{domain}-{seed}",
        "behavior_fingerprint_sha256": f"behavior-{domain}-{seed}",
        "fit_seconds": 1.0,
        "predict_seconds": 0.1,
        "worker_peak_rss_bytes": 1000,
        "excess_brier": 0.001 if canary else None,
    }


def test_m5_analyzer_is_nonranking_and_enforces_known_floors():
    rows = [_row(identity) for identity in m5_expected_grid()]

    analysis = analyze_rows(rows, establishing=True)

    assert analysis["paired_cells"] == 19
    assert analysis["ranking_or_acceptance_score"] is False
    assert all(
        check["passed"]
        for check in analysis["known_floor_checks"].values()
    )

    failed_index = next(
        index
        for index, row in enumerate(rows)
        if row["domain_id"] == "binary_classification"
    )
    rows[failed_index] = dict(rows[failed_index], excess_brier=0.02)
    with pytest.raises(RuntimeError, match="known floor"):
        analyze_rows(rows, establishing=True)
