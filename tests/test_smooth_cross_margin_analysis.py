import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from benchmarks import analyze_smooth_cross_margin as analysis
from benchmarks import run_smooth_cross_features as runner


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks" / "smooth_cross_margin_analysis.json"
RAW_ARTIFACT = ROOT / "benchmarks" / "smooth_cross_features.json"
REPORT = ROOT / "benchmarks" / "smooth_cross_features_result.md"
ORIGINAL_RAW_SOURCE_COMMIT = "f6d7983f537f9995739dbfd327b31e97f28cd747"
ORIGINAL_RUNNER_SHA256 = (
    "473f82baaf25d692ffc298f563b5a290b6f38089c5705d268044a79e9570f308"
)
ORIGINAL_ANALYZER_SOURCE_COMMIT = "da5e2d313e522fb9da0abe3c93853bbc8a052512"
ORIGINAL_ANALYZER_SHA256 = (
    "0c370e3380857bd86e2c632a2c26632f1c36f7e4893b6be9f865b76d02617e85"
)
HARDENED_SOURCE_COMMIT = "816101476bb65cf5a0e2f59cd11edaf96f46a1cc"
HARDENED_RUNNER_SHA256 = (
    "4e831fa0ff26f7c64b4e130259d1a3fcb565b51c0310ab5ee0bd2a8da7a248eb"
)
HARDENED_ANALYZER_SHA256 = (
    "dbb233f6d4d9776881bdfeaa839480c8b3e353a75a8ae60233c7da1b5a1463da"
)
FROZEN_RAW_FILE_SHA256 = (
    "b5544bc598601862c443c237214124007ae49b72b11e2cf2888f03112450d30c"
)
FROZEN_ANALYSIS_FILE_SHA256 = (
    "0e115192938137c8bb713f5ca533c84bad315460aae12a900b32603970ad0190"
)


def _row(dataset, fold, margin, ratio, *, selected=True):
    task_ids = {"a": 1, "b": 2}
    return {
        "task_id": task_ids[dataset],
        "dataset_name": dataset,
        "fold": fold,
        "cross_selected": selected,
        "base": {
            "best_validation_rmse": 1.0,
            "test_rmse": 1.0,
        },
        "selected": {
            "best_validation_rmse": 1.0 - margin,
            "test_rmse": ratio,
        },
    }


def _source():
    return {
        "results": [
            _row("a", 0, 0.02, 1.10),
            _row("a", 1, 0.06, 0.80),
            _row("b", 0, 0.07, 0.90),
            _row("b", 1, 0.00, 1.00, selected=False),
        ]
    }


def test_margin_evaluation_declines_below_threshold_as_exact_tie():
    result = analysis.evaluate_margin(_source()["results"], 0.05)
    records = {
        (row["dataset_name"], row["fold"]): row
        for row in result["split_records"]
    }
    assert records[("a", 0)]["engaged"] is False
    assert records[("a", 0)]["test_ratio"] == 1.0
    assert records[("a", 1)]["engaged"] is True
    assert result["worst_split_ratio"] == 1.0


def test_result_discloses_original_and_hardened_evidence_bindings():
    report = REPORT.read_text()
    for source_commit, source_path, expected_sha256 in (
        (
            ORIGINAL_RAW_SOURCE_COMMIT,
            "benchmarks/run_smooth_cross_features.py",
            ORIGINAL_RUNNER_SHA256,
        ),
        (
            ORIGINAL_ANALYZER_SOURCE_COMMIT,
            "benchmarks/analyze_smooth_cross_margin.py",
            ORIGINAL_ANALYZER_SHA256,
        ),
    ):
        source = subprocess.run(
            ["git", "show", f"{source_commit}:{source_path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(source).hexdigest() == expected_sha256
    assert hashlib.sha256(RAW_ARTIFACT.read_bytes()).hexdigest() == (
        FROZEN_RAW_FILE_SHA256
    )
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        FROZEN_ANALYSIS_FILE_SHA256
    )
    assert analysis._sha256(Path(runner.__file__).resolve()) == (
        HARDENED_RUNNER_SHA256
    )
    assert analysis._sha256(Path(analysis.__file__).resolve()) == (
        HARDENED_ANALYZER_SHA256
    )
    for label, value in (
        ("Frozen raw file SHA-256", FROZEN_RAW_FILE_SHA256),
        ("Frozen margin-analysis file SHA-256", FROZEN_ANALYSIS_FILE_SHA256),
        ("Original raw-run source commit", ORIGINAL_RAW_SOURCE_COMMIT),
        ("Original run-time runner SHA-256", ORIGINAL_RUNNER_SHA256),
        (
            "Original margin-analyzer source commit",
            ORIGINAL_ANALYZER_SOURCE_COMMIT,
        ),
        ("Original margin-analyzer SHA-256", ORIGINAL_ANALYZER_SHA256),
        ("Current hardened source commit", HARDENED_SOURCE_COMMIT),
        (
            "Current hardened runner/verifier SHA-256",
            HARDENED_RUNNER_SHA256,
        ),
        (
            "Current hardened margin-analyzer SHA-256",
            HARDENED_ANALYZER_SHA256,
        ),
    ):
        assert f"- {label}:\n  `{value}`" in report
    assert "no benchmark outcome was\nrerun or replaced" in report


def test_analysis_nominates_smallest_grid_margin_with_zero_observed_harm(
    monkeypatch,
):
    monkeypatch.setattr(analysis, "MARGIN_GRID", (0.0, 0.01, 0.05, 0.10))
    result = analysis.analyze(_source(), validate_source=False)
    assert result["nominee"]["minimum_validation_improvement"] == 0.05
    assert result["nominee"]["worst_split_ratio"] == 1.0
    assert result["fresh_claim_eligible"] is False
    assert result["nominee_requires_fresh_confirmation"] is True


def test_no_nominee_when_every_grid_point_has_harm(monkeypatch):
    source = _source()
    changed = copy.deepcopy(source)
    changed["results"][0]["selected"]["best_validation_rmse"] = 0.5
    monkeypatch.setattr(analysis, "MARGIN_GRID", (0.0, 0.10))
    result = analysis.analyze(changed, validate_source=False)
    assert result["nominee"] is None
    assert result["nominee_requires_fresh_confirmation"] is False


def test_geomean_rejects_nonpositive_values():
    with pytest.raises(RuntimeError, match="positive"):
        analysis._geomean([1.0, 0.0])


def test_margin_evaluation_rejects_wrong_domain_types():
    rows = _source()["results"]
    with pytest.raises(RuntimeError, match="threshold"):
        analysis.evaluate_margin(rows, "0.05")

    changed = copy.deepcopy(rows)
    changed[0]["cross_selected"] = 1
    with pytest.raises(RuntimeError, match="row ledger"):
        analysis.evaluate_margin(changed, 0.05)

    changed = copy.deepcopy(rows)
    changed[0]["base"]["best_validation_rmse"] = "1.0"
    with pytest.raises(RuntimeError, match="finite and positive"):
        analysis.evaluate_margin(changed, 0.05)


def test_margin_evaluation_rejects_coordinate_and_selection_contradictions():
    changed = copy.deepcopy(_source()["results"])
    changed.append(copy.deepcopy(changed[0]))
    with pytest.raises(RuntimeError, match="repeats a split"):
        analysis.evaluate_margin(changed, 0.05)

    changed = copy.deepcopy(_source()["results"])
    changed[0]["cross_selected"] = False
    with pytest.raises(RuntimeError, match="cross-selection ledger"):
        analysis.evaluate_margin(changed, 0.05)

    changed = copy.deepcopy(_source()["results"])
    changed[0]["task_id"] = changed[2]["task_id"]
    with pytest.raises(RuntimeError, match="identity changed"):
        analysis.evaluate_margin(changed, 0.05)


def test_margin_artifact_create_is_atomic_and_create_only(tmp_path):
    output = tmp_path / "result.json"
    analysis._atomic_create(output, b"first")
    assert output.read_bytes() == b"first"
    with pytest.raises(FileExistsError, match="refusing to replace"):
        analysis._atomic_create(output, b"second")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        analysis.run(output)


def test_margin_artifact_rejects_mutable_symlink_parent(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink output"):
        analysis._atomic_create(
            linked / "missing" / "result.json",
            b"result",
        )
    assert list(real.iterdir()) == []


def test_margin_analysis_rejects_forged_source_provenance():
    source = json.loads(analysis.SOURCE.read_text())
    changed = copy.deepcopy(source)
    changed["protocol"]["sha256"] = "forged"
    with pytest.raises(RuntimeError, match="protocol ledger"):
        analysis.analyze(changed)

    changed = copy.deepcopy(source)
    changed["partition_boundary"]["partition_sha256"] = "forged"
    with pytest.raises(RuntimeError, match="partition ledger"):
        analysis.analyze(changed)

    changed = copy.deepcopy(source)
    changed["analysis"]["equal_dataset_geomean_ratio"] = 99.0
    with pytest.raises(RuntimeError, match="not reproducible"):
        analysis.analyze(changed)


def test_recorded_analysis_nominates_five_percent_for_fresh_confirmation(
    assert_analysis_equal,
):
    artifact = json.loads(ARTIFACT.read_text())
    source = json.loads(analysis.SOURCE.read_text())
    assert artifact["source"]["sha256"] == analysis._sha256(analysis.SOURCE)
    analysis.validate_artifact(artifact)
    assert_analysis_equal(
        artifact["analysis"],
        analysis.analyze(source),
    )
    nominee = artifact["analysis"]["nominee"]
    assert nominee["minimum_validation_improvement"] == 0.05
    assert nominee["worst_split_ratio"] == 1.0
    assert artifact["analysis"]["fresh_claim_eligible"] is False


def test_recorded_margin_artifact_binds_source_and_stored_analysis():
    artifact = json.loads(ARTIFACT.read_text())
    changed = copy.deepcopy(artifact)
    changed["source"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="source ledger changed"):
        analysis.validate_artifact(changed)

    changed = copy.deepcopy(artifact)
    changed["analysis"]["nominee"] = None
    with pytest.raises(RuntimeError, match="not reproducible"):
        analysis.validate_artifact(changed)

    changed = copy.deepcopy(artifact)
    changed["created_at"] = "2026-07-18T00:00:00+00:00"
    with pytest.raises(RuntimeError, match="frozen artifact changed"):
        analysis.validate_artifact(changed)
