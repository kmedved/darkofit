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
    M5_THREADS,
    _assert_darkofit_source,
    _valid_probability_matrix,
    _worker_environment,
    _write_create_only,
    _quality_baseline,
    analyze_rows,
    source_state,
    validate_sources,
)


def test_m5_worker_environment_overrides_inherited_numba_ceiling(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("NUMBA_NUM_THREADS", "1")
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    monkeypatch.setenv("NUMBA_BOUNDSCHECK", "1")
    monkeypatch.setenv("NUMBA_THREADING_LAYER", "workqueue")
    monkeypatch.setenv("OMP_DYNAMIC", "TRUE")
    monkeypatch.setenv("OMP_THREAD_LIMIT", "1")
    monkeypatch.setenv("KMP_AFFINITY", "compact")
    monkeypatch.setenv("MKL_DYNAMIC", "TRUE")
    monkeypatch.setenv("PYTHONOPTIMIZE", "2")
    monkeypatch.setenv("PYTHONWARNINGS", "error")
    monkeypatch.setenv("PYTHONPATH", "/wrong/repository")

    environment = _worker_environment(tmp_path)

    assert environment["NUMBA_NUM_THREADS"] == str(M5_THREADS)
    assert environment["NUMBA_DISABLE_JIT"] == "0"
    assert environment["OMP_DYNAMIC"] == "FALSE"
    assert environment["OMP_THREAD_LIMIT"] == str(M5_THREADS)
    assert environment["MKL_DYNAMIC"] == "FALSE"
    assert "NUMBA_BOUNDSCHECK" not in environment
    assert "NUMBA_THREADING_LAYER" not in environment
    assert "KMP_AFFINITY" not in environment
    assert "PYTHONOPTIMIZE" not in environment
    assert "PYTHONWARNINGS" not in environment
    assert "PYTHONPATH" not in environment


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


def test_m5_probability_validation_uses_an_absolute_tolerance_and_bounds():
    valid = np.asarray([[0.25, 0.75], [0.6, 0.4]])
    relative_tolerance_only = np.asarray([[0.25, 0.750005], [0.6, 0.4]])
    out_of_range = np.asarray([[-0.1, 1.1], [0.6, 0.4]])
    phantom_binary_class = np.asarray(
        [[0.2, 0.7, 0.1], [0.3, 0.6, 0.1]]
    )

    assert _valid_probability_matrix(
        valid,
        expected_rows=2,
        expected_columns=2,
    )
    assert not _valid_probability_matrix(
        relative_tolerance_only,
        expected_rows=2,
    )
    assert not _valid_probability_matrix(out_of_range, expected_rows=2)
    assert not _valid_probability_matrix(
        phantom_binary_class,
        expected_rows=2,
        expected_columns=2,
    )


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


def test_m5_analyzer_rejects_thread_environment_drift():
    rows = _baseline_rows()
    rows[0]["thread_environment"]["OMP_NUM_THREADS"] = "1"

    with pytest.raises(RuntimeError, match="thread environment"):
        analyze_rows(rows, establishing=True)


def test_m5_analyzer_rejects_task_class_cardinality_drift():
    rows = _baseline_rows()
    row = next(
        item for item in rows if item["domain_id"] == "binary_classification"
    )
    row["model_metadata"]["classes"] = [0, 1, 2]

    with pytest.raises(RuntimeError, match="model metadata"):
        analyze_rows(rows, establishing=True)


def test_m5_analyzer_rejects_duplicate_class_metadata():
    rows = _baseline_rows()
    row = next(
        item for item in rows if item["domain_id"] == "binary_classification"
    )
    row["model_metadata"]["classes"] = [0, 0]

    with pytest.raises(RuntimeError, match="model metadata"):
        analyze_rows(rows, establishing=True)
