"""Regression tests for the successor paired-evidence execution contract."""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
REPO_ROOT = BENCH_DIR.parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from benchmark_adapters import FitConfig, RevisionSpec  # noqa: E402
import bench_compare_revisions as comparison  # noqa: E402
import paired_evidence_contract as evidence_contract  # noqa: E402
from bench_compare_revisions import (  # noqa: E402
    EVIDENCE_CSV_FIELDS,
    _run_worker,
    _save_case,
)
from paired_evidence_contract import (  # noqa: E402
    CONTRACT_ENV_KEYS,
    CONTRACT_THREADS,
    CONTRACT_VERSION,
    THREAD_ENV_DEFAULTS,
    THREAD_ENV_KEYS,
    contract_payload,
    fixed_worker_environment,
    load_and_validate_csv,
    validate_rows,
    write_create_only,
)


def _thread_environment(tmp_path: Path) -> dict[str, object]:
    environment: dict[str, object] = {
        key: str(CONTRACT_THREADS) for key in THREAD_ENV_KEYS
    }
    environment.update(
        {
            **THREAD_ENV_DEFAULTS,
            "NUMBA_NUM_THREADS": str(CONTRACT_THREADS),
            "NUMBA_DISABLE_JIT": "0",
            "NUMBA_THREADING_LAYER": "default",
            "NUMBA_CACHE_DIR": str(tmp_path / "numba-cache"),
            "DARKOFIT_WARMUP": "0",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": None,
        }
    )
    return environment


def test_successor_contract_is_explicitly_draft_and_nonranking():
    payload = contract_payload()

    assert payload["contract_version"] == CONTRACT_VERSION
    assert payload["contract_frozen"] is False
    assert payload["candidate_ranking_eligible"] is False
    assert payload["shipping_or_default_claim_eligible"] is False
    assert payload["threads"] == CONTRACT_THREADS
    assert payload["required_environment"]["NUMBA_NUM_THREADS"] == str(
        CONTRACT_THREADS
    )


def _row(
    variant: str,
    source: Path,
    tmp_path: Path,
) -> dict[str, object]:
    metadata = {
        "member_count": 1,
        "tree_count": 3,
        "tree_counts": [3],
        "tree_modes": ["lightgbm"],
        "resolved_thread_counts": [CONTRACT_THREADS],
        "best_iterations": [2],
    }
    return {
        "status": "ok",
        "error": "",
        "variant": variant,
        "revision_path": str(source.resolve()),
        "use_defaults": "True",
        "selected_tree_mode": "lightgbm",
        "dataset": "numeric_binary",
        "task": "binary",
        "size": "tiny",
        "seed": "0",
        "weight_mode": "none",
        "n_train": "100",
        "n_val": "0",
        "n_test": "30",
        "n_features": "4",
        "fit_seconds": "1.0",
        "predict_seconds": "0.1",
        "worker_peak_rss_bytes": "1000000",
        "primary_metric": "log_loss",
        "primary_value": "0.5",
        "accuracy": "0.8",
        "f1_macro": "0.75",
        "log_loss": "0.5",
        "brier": "0.3",
        "weighted_accuracy": "0.8",
        "weighted_f1_macro": "0.75",
        "weighted_log_loss": "0.5",
        "weighted_brier": "0.3",
        "evidence_contract": CONTRACT_VERSION,
        "candidate_ranking_eligible": "False",
        "shipping_or_default_claim_eligible": "False",
        "implementation_path": str(source.resolve() / "darkofit" / "__init__.py"),
        "case_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "split_sha256": "c" * 64,
        "weight_sha256": "d" * 64,
        "prediction_sha256": (
            "e" * 63 + ("0" if variant == "control_default" else "1")
        ),
        "probability_sha256": (
            "f" * 63 + ("0" if variant == "control_default" else "1")
        ),
        "expected_class_count": "2",
        "class_count": "2",
        "probability_width": "2",
        "probability_min": "0.1",
        "probability_max": "0.9",
        "probability_row_sum_max_error": "1e-12",
        "model_metadata": json.dumps(metadata, sort_keys=True),
        "requested_thread_count": str(CONTRACT_THREADS),
        "fitted_thread_counts": json.dumps([CONTRACT_THREADS]),
        "numba_thread_ceiling": str(CONTRACT_THREADS),
        "numba_current_thread_count": str(CONTRACT_THREADS),
        "numba_threading_layer": "omp",
        "thread_environment": json.dumps(_thread_environment(tmp_path), sort_keys=True),
    }


def _paired_rows(tmp_path: Path):
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    return (
        [
            _row("control_default", control, tmp_path),
            _row("candidate_default", candidate, tmp_path),
        ],
        {
            "control_default": control,
            "candidate_default": candidate,
        },
    )


def _regression_rows(tmp_path: Path):
    rows, sources = _paired_rows(tmp_path)
    for row in rows:
        row.update(
            {
                "task": "regression",
                "primary_metric": "rmse",
                "primary_value": "1.0",
                "rmse": "1.0",
                "mae": "0.8",
                "r2": "0.2",
                "weighted_rmse": "1.0",
                "weighted_mae": "0.8",
                "weighted_r2": "0.2",
                "expected_class_count": "0",
                "class_count": "0",
                "probability_width": "0",
                "probability_sha256": "",
                "probability_min": "",
                "probability_max": "",
                "probability_row_sum_max_error": "",
            }
        )
    return rows, sources


def test_fixed_worker_environment_overrides_contaminated_parent(tmp_path):
    base = dict(os.environ)
    base.update(
        {
            "NUMBA_NUM_THREADS": "1",
            "NUMBA_DISABLE_JIT": "1",
            "NUMBA_BOUNDSCHECK": "1",
            "PYTHONPATH": "/wrong/repository",
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "TRUE",
            "KMP_AFFINITY": "compact",
        }
    )

    environment = fixed_worker_environment(tmp_path, base=base)

    assert environment["NUMBA_NUM_THREADS"] == str(CONTRACT_THREADS)
    assert environment["NUMBA_DISABLE_JIT"] == "0"
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["DARKOFIT_WARMUP"] == "0"
    assert environment["NUMBA_THREADING_LAYER"] == "default"
    assert environment["OMP_DYNAMIC"] == "FALSE"
    assert environment["OMP_THREAD_LIMIT"] == str(CONTRACT_THREADS)
    assert environment["MKL_DYNAMIC"] == "FALSE"
    assert "KMP_AFFINITY" not in environment
    assert "NUMBA_BOUNDSCHECK" not in environment
    assert "PYTHONPATH" not in environment
    assert all(environment[key] == str(CONTRACT_THREADS) for key in THREAD_ENV_KEYS)


def test_paired_rows_require_strict_resources_and_split_identity(tmp_path):
    rows, sources = _paired_rows(tmp_path)

    assert validate_rows(rows, expected_sources=sources) == {
        "row_count": 2,
        "paired_cells": 1,
        "resolved_threads": CONTRACT_THREADS,
        "contract_version": CONTRACT_VERSION,
    }

    rows[1]["split_sha256"] = "9" * 64
    with pytest.raises(RuntimeError, match="differs on split_sha256"):
        validate_rows(rows, expected_sources=sources)


def test_paired_rows_require_the_predeclared_complete_grid(tmp_path):
    rows, sources = _paired_rows(tmp_path)

    with pytest.raises(RuntimeError, match="grid mismatch"):
        validate_rows(
            rows,
            expected_sources=sources,
            expected_pair_keys=[
                ("numeric_binary", "tiny", "0", "none"),
                ("numeric_binary", "tiny", "1", "none"),
            ],
        )


def test_paired_validator_cannot_reinterpret_the_fixed_thread_contract(
    tmp_path,
):
    rows, sources = _paired_rows(tmp_path)

    with pytest.raises(ValueError, match="requires exactly"):
        validate_rows(rows, expected_sources=sources, threads=1)


def test_paired_public_defaults_forbid_external_validation_rows(tmp_path):
    rows, sources = _paired_rows(tmp_path)
    rows[0]["n_val"] = "1"

    with pytest.raises(RuntimeError, match="external validation"):
        validate_rows(rows, expected_sources=sources)


def test_paired_rows_require_the_task_appropriate_primary_loss(tmp_path):
    rows, sources = _paired_rows(tmp_path)
    rows[0]["primary_metric"] = "accuracy"
    rows[0]["primary_value"] = rows[0]["accuracy"]

    with pytest.raises(RuntimeError, match="primary metric"):
        validate_rows(rows, expected_sources=sources)


def test_paired_rows_reject_out_of_range_secondary_metrics(tmp_path):
    rows, sources = _paired_rows(tmp_path)
    rows[0]["accuracy"] = "1.1"

    with pytest.raises(RuntimeError, match="primary metric"):
        validate_rows(rows, expected_sources=sources)


@pytest.mark.parametrize(
    "field",
    ["accuracy", "f1_macro", "weighted_accuracy", "weighted_f1_macro"],
)
def test_paired_rows_reject_negative_classification_scores(tmp_path, field):
    rows, sources = _paired_rows(tmp_path)
    rows[0][field] = "-0.01"

    with pytest.raises(RuntimeError, match="primary metric"):
        validate_rows(rows, expected_sources=sources)


@pytest.mark.parametrize("field", ["brier", "weighted_brier"])
def test_paired_rows_reject_impossible_brier_scores(tmp_path, field):
    rows, sources = _paired_rows(tmp_path)
    rows[0][field] = "2.01"

    with pytest.raises(RuntimeError, match="primary metric"):
        validate_rows(rows, expected_sources=sources)


@pytest.mark.parametrize("field", ["r2", "weighted_r2"])
def test_paired_rows_reject_r2_above_one(tmp_path, field):
    rows, sources = _regression_rows(tmp_path)
    rows[0][field] = "1.01"

    with pytest.raises(RuntimeError, match="primary metric"):
        validate_rows(rows, expected_sources=sources)


def test_paired_rows_require_matching_numba_backends(tmp_path):
    rows, sources = _paired_rows(tmp_path)
    rows[1]["numba_threading_layer"] = "workqueue"

    with pytest.raises(RuntimeError, match="numba_threading_layer"):
        validate_rows(rows, expected_sources=sources)


def test_paired_rows_require_an_absolute_numba_cache_path(tmp_path):
    rows, sources = _paired_rows(tmp_path)
    environment = json.loads(rows[0]["thread_environment"])
    environment["NUMBA_CACHE_DIR"] = "relative-cache"
    rows[0]["thread_environment"] = json.dumps(environment)

    with pytest.raises(RuntimeError, match="cache path"):
        validate_rows(rows, expected_sources=sources)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_metadata", "", "blank 'model_metadata'"),
        ("fitted_thread_counts", "[1]", "fitted thread mask"),
        ("numba_thread_ceiling", "1", "resolved thread budget"),
        ("probability_width", "3", "probability metadata"),
        (
            "probability_row_sum_max_error",
            "0.01",
            "probability metadata",
        ),
        ("probability_min", "-0.01", "probability metadata"),
        ("probability_max", "1.01", "probability metadata"),
    ],
)
def test_paired_rows_reject_missing_or_false_provenance(
    tmp_path, field, value, message
):
    rows, sources = _paired_rows(tmp_path)
    rows[0][field] = value

    with pytest.raises(RuntimeError, match=message):
        validate_rows(rows, expected_sources=sources)


def test_paired_rows_bind_implementation_to_expected_package_root(tmp_path):
    rows, sources = _paired_rows(tmp_path)
    rows[0]["implementation_path"] = str(tmp_path / "shadow.py")

    with pytest.raises(RuntimeError, match="implementation path drifted"):
        validate_rows(rows, expected_sources=sources)


def test_paired_rows_require_exactly_two_classes_for_binary_tasks(tmp_path):
    rows, sources = _paired_rows(tmp_path)
    for row in rows:
        row["expected_class_count"] = "3"
        row["class_count"] = "3"
        row["probability_width"] = "3"

    with pytest.raises(RuntimeError, match="probability metadata"):
        validate_rows(rows, expected_sources=sources)


def test_paired_metadata_rejects_nonvector_predictions(monkeypatch, tmp_path):
    runtime = {
        "ceiling": CONTRACT_THREADS,
        "current": CONTRACT_THREADS,
        "threading_layer": "omp",
        "environment": _thread_environment(tmp_path),
    }
    monkeypatch.setattr(
        evidence_contract,
        "assert_worker_contract",
        lambda _threads: runtime,
    )
    monkeypatch.setattr(
        evidence_contract,
        "fitted_model_metadata",
        lambda _model: {"resolved_thread_counts": [CONTRACT_THREADS]},
    )
    data_path = tmp_path / "case.npz"
    data_path.write_bytes(b"case")
    y = np.asarray([0, 1])
    data = {
        "X_fit": np.zeros((2, 1)),
        "X_val": np.zeros((2, 1)),
        "X_test": np.zeros((2, 1)),
        "y_fit": y,
        "y_val": y,
        "y_test": y,
        "w_fit": None,
        "w_val": None,
        "w_test": None,
    }

    with pytest.raises(RuntimeError, match="prediction shape"):
        evidence_contract.evidence_row_metadata(
            model=object(),
            implementation_path=str(REPO_ROOT / "darkofit" / "booster.py"),
            data_path=data_path,
            data=data,
            task="binary",
            prediction=np.asarray([[0], [1]]),
            probability=np.asarray([[0.8, 0.2], [0.1, 0.9]]),
            labels=np.asarray([0, 1]),
            requested_threads=CONTRACT_THREADS,
        )


def test_paired_create_only_write_removes_partial_output(
    monkeypatch, tmp_path
):
    output = tmp_path / "partial.csv"

    def fail_fsync(_descriptor):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="simulated fsync failure"):
        write_create_only(output, b"partial")

    assert not output.exists()


def test_csv_loader_rejects_schema_drift_and_unnamed_extra_values(tmp_path):
    rows, sources = _paired_rows(tmp_path)
    output = tmp_path / "paired.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVIDENCE_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field, "") for field in EVIDENCE_CSV_FIELDS}
            )

    _, validation = load_and_validate_csv(
        output,
        expected_fields=EVIDENCE_CSV_FIELDS,
        expected_sources=sources,
    )
    assert validation["paired_cells"] == 1

    lines = output.read_text().splitlines()
    lines[1] += ",unexpected"
    output.write_text("\n".join(lines) + "\n")
    with pytest.raises(RuntimeError, match="extra unnamed fields"):
        load_and_validate_csv(
            output,
            expected_fields=EVIDENCE_CSV_FIELDS,
            expected_sources=sources,
        )


def test_paired_runner_aborts_after_first_worker_failure(monkeypatch, tmp_path):
    calls = []

    def fail_worker(payload_path, *, environment=None):
        calls.append((payload_path, environment))
        return {"status": "error", "error": "injected failure"}

    monkeypatch.setattr(comparison, "_run_worker", fail_worker)
    output = tmp_path / "failed.csv"

    with pytest.raises(RuntimeError, match="no output was published"):
        comparison.main(
            [
                "--policy-suite",
                "standing-slice",
                "--control",
                str(REPO_ROOT),
                "--candidate",
                str(REPO_ROOT),
                "--datasets",
                "friedman_numeric",
                "--sizes",
                "tiny",
                "--seeds",
                "1",
                "--repeat",
                "1",
                "--threads",
                str(CONTRACT_THREADS),
                "--weight-modes",
                "none",
                "--models",
                "control_default",
                "candidate_default",
                "--evidence-contract",
                CONTRACT_VERSION,
                "--csv",
                str(output),
            ]
        )

    assert len(calls) == 1
    assert not output.exists()


@pytest.mark.parametrize("option", ["--seeds", "--repeat"])
def test_benchmark_runner_rejects_nonpositive_work_counts(tmp_path, option):
    output = tmp_path / "empty.csv"

    with pytest.raises(SystemExit, match="positive integer"):
        comparison.main([option, "0", "--csv", str(output)])

    assert not output.exists()


def test_paired_worker_establishes_four_threads_from_contaminated_shell(
    tmp_path,
):
    rng = np.random.default_rng(20260720)
    X = rng.normal(size=(120, 4))
    y = X[:, 0] - 0.5 * X[:, 1]
    split = {
        "X_fit": X[:80],
        "X_val": X[80:100],
        "X_test": X[100:],
        "y_fit": y[:80],
        "y_val": y[80:100],
        "y_test": y[100:],
        "w_fit": None,
        "w_val": None,
        "w_test": None,
    }
    data_path = tmp_path / "case.npz"
    _save_case(data_path, split)
    payload = {
        "variant": asdict(
            RevisionSpec(
                "control_default",
                str(REPO_ROOT),
                tree_mode="catboost",
            )
        ),
        "fit_config": asdict(
            FitConfig(
                iterations=3,
                patience=2,
                depth=3,
                max_bins=32,
                threads=CONTRACT_THREADS,
                verbose_timing=False,
            )
        ),
        "data_path": str(data_path),
        "task": "regression",
        "cat_features": [],
        "seed": 0,
        "repeat": 1,
        "evidence_contract": CONTRACT_VERSION,
    }
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload))
    contaminated = dict(os.environ)
    contaminated.update(
        {
            "NUMBA_NUM_THREADS": "1",
            "NUMBA_DISABLE_JIT": "1",
            "PYTHONPATH": "/wrong/repository",
        }
    )
    environment = fixed_worker_environment(
        tmp_path / "numba-cache",
        base=contaminated,
    )

    row = _run_worker(payload_path, environment=environment)

    assert row["status"] == "ok", row.get("error")
    assert row["evidence_contract"] == CONTRACT_VERSION
    assert row["candidate_ranking_eligible"] is False
    assert row["shipping_or_default_claim_eligible"] is False
    assert row["numba_thread_ceiling"] == CONTRACT_THREADS
    assert row["numba_current_thread_count"] == CONTRACT_THREADS
    assert json.loads(row["fitted_thread_counts"]) == [CONTRACT_THREADS]
    recorded_environment = json.loads(row["thread_environment"])
    assert recorded_environment["PYTHONPATH"] is None
    assert recorded_environment["NUMBA_NUM_THREADS"] == str(CONTRACT_THREADS)
    assert set(recorded_environment) == set(CONTRACT_ENV_KEYS)
