"""Contract and analyzer tests for the M5 diversity sentinels."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from run_m5_sentinels import (  # noqa: E402
    _assert_darkofit_source,
    _write_create_only,
    _quality_baseline,
    analyze_rows,
    source_state,
    validate_sources,
)


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


def test_m5_source_state_rejects_a_git_subdirectory():
    with pytest.raises(RuntimeError, match="source checkout"):
        source_state(BENCH_DIR)


def test_m5_import_binding_rejects_shadowed_module(tmp_path):
    source = tmp_path / "source"
    module_path = source / "darkofit" / "__init__.py"
    module_path.parent.mkdir(parents=True)
    module_path.touch()

    assert _assert_darkofit_source(
        SimpleNamespace(__file__=str(module_path)), source
    ) == str(module_path)
    with pytest.raises(RuntimeError, match="not"):
        _assert_darkofit_source(
            SimpleNamespace(__file__=str(tmp_path / "shadow.py")), source
        )


def test_m5_create_only_write_removes_partial_output(monkeypatch, tmp_path):
    output = tmp_path / "partial.json"

    def fail_fsync(_descriptor):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="simulated fsync failure"):
        _write_create_only(output, b"partial")

    assert not output.exists()


def _baseline_rows():
    artifact = json.loads(
        (BENCH_DIR / "m5_sentinel_baseline.json").read_text()
    )
    return copy.deepcopy(artifact["rows"])


def test_m5_analyzer_is_nonranking_and_enforces_known_floors():
    rows = _baseline_rows()

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


def test_m5_analyzer_revalidates_worker_invariants():
    rows = _baseline_rows()
    rows[0]["roundtrip_exact"] = False

    with pytest.raises(RuntimeError, match="row invariant"):
        analyze_rows(rows, establishing=True)
