import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from benchmarks import run_rssi_linear_leaf_diagnosis as diagnosis


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks" / "rssi_linear_leaf_diagnosis.json"
REPORT = ROOT / "benchmarks" / "rssi_linear_leaf_diagnosis_result.md"
ORIGINAL_SOURCE_COMMIT = "dcd6e298e61aaf114d922cef4e1666fefcd66add"
ORIGINAL_RUNNER_SHA256 = (
    "136296297733f24d31f5bc82ad049411f1baec806a88497821c38ff1e4771c05"
)
HARDENED_SOURCE_COMMIT = "816101476bb65cf5a0e2f59cd11edaf96f46a1cc"
HARDENED_RUNNER_SHA256 = (
    "8b4b9ec41cfa9178ff93c143bd9d09abce98cf0194f943ed9c003a145e944104"
)
FROZEN_RAW_FILE_SHA256 = (
    "02c2d36a12b3a452363cc9b8a62b1cf246b09829a5544d233eae375b10d17ef6"
)


def _row(arm, *, marker="same", best=1.0, selected=None, cross=None):
    return {
        "arm": arm,
        "library": (
            "darkofit" if arm.startswith("darko_") else "chimeraboost"
        ),
        "fit_seconds": 1.0,
        "resolved_learning_rate": 0.1,
        "borders_sha256": marker,
        "validation_history_sha256": marker,
        "model_sha256": marker,
        "prediction_sha256": marker,
        "fitted_tree_count": 10,
        "best_validation_rmse": (
            None if arm == "darko_default" else best
        ),
        "test_rmse": best,
        "linear_leaves_selected": selected,
        "cross_features_selected": cross,
        "cross_pair_count": 30 if cross is True else 0,
    }


def _valid_rows():
    rows = []
    for arm in diagnosis.ARMS:
        if arm in {
            "darko_shared_constant",
            "chimera_shared_constant",
        }:
            row = _row(arm, best=0.8)
        elif arm == "chimera_full_selector":
            row = _row(arm, best=0.8, selected=False)
        elif arm in {
            "darko_shared_linear",
            "darko_matched_auto20_linear",
            "chimera_shared_linear",
        }:
            row = _row(arm, best=1.0)
        elif arm == "chimera_capped_selector":
            row = _row(arm, best=1.0, selected=True)
        elif arm == "chimera_product":
            row = _row(arm, best=1.0, selected=True, cross=False)
        elif arm == "chimera_full_product":
            row = _row(arm, best=0.8, selected=False, cross=False)
        else:
            row = _row(arm, best=1.0)
        rows.append(row)
    return rows


def test_protocol_uses_only_previously_scored_coordinate():
    registry = json.loads(
        (ROOT / "benchmarks" / "fresh_confirmation_registry.json").read_text()
    )
    coordinate = {
        "task_id": diagnosis.TASK_ID,
        "repeat": diagnosis.OUTER_REPEAT,
        "fold": diagnosis.OUTER_FOLD,
        "sample": diagnosis.OUTER_SAMPLE,
    }
    assert coordinate in registry["coordinates"]

    prior = json.loads(
        (ROOT / "benchmarks" / "fresh_selector_confirmation.json").read_text()
    )
    matching = [
        row for row in prior["results"] if row["task_id"] == diagnosis.TASK_ID
    ]
    assert matching
    assert all(
        diagnosis.OUTER_FOLD in [fold["fold"] for fold in row["folds"]]
        for row in matching
    )


def test_result_discloses_original_and_hardened_evidence_bindings():
    report = REPORT.read_text()
    original_runner = subprocess.run(
        [
            "git",
            "show",
            (
                f"{ORIGINAL_SOURCE_COMMIT}:"
                "benchmarks/run_rssi_linear_leaf_diagnosis.py"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert hashlib.sha256(original_runner).hexdigest() == (
        ORIGINAL_RUNNER_SHA256
    )
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        FROZEN_RAW_FILE_SHA256
    )
    assert diagnosis._sha256(Path(diagnosis.__file__).resolve()) == (
        HARDENED_RUNNER_SHA256
    )
    for label, value in (
        ("Frozen raw file SHA-256", FROZEN_RAW_FILE_SHA256),
        ("Original source commit", ORIGINAL_SOURCE_COMMIT),
        ("Original run-time runner SHA-256", ORIGINAL_RUNNER_SHA256),
        ("Current hardened source commit", HARDENED_SOURCE_COMMIT),
        (
            "Current hardened runner/verifier SHA-256",
            HARDENED_RUNNER_SHA256,
        ),
    ):
        assert f"- {label}:\n  `{value}`" in report
    assert "it did not generate new\nbenchmark outcomes" in report


def test_analysis_reports_capped_misselection_without_making_shipping_claim():
    analysis = diagnosis.analyze(_valid_rows())
    assert analysis["forced_full_budget_validation_winner"] == "constant"
    assert analysis["full_selector_winner"] == "constant"
    assert analysis["capped_selector_winner"] == "linear"
    assert analysis["capped_selector_disagrees_with_full"] is True
    assert analysis["fresh_claim_eligible"] is False
    assert analysis["claim_tier"] == "development_diagnostic_only"


def test_analysis_rejects_nonexact_matched_engines():
    rows = _valid_rows()
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["arm"] == "chimera_shared_linear":
            row["prediction_sha256"] = "different"
    with pytest.raises(RuntimeError, match="prediction_sha256"):
        diagnosis.analyze(changed)


def test_analysis_rejects_full_selector_that_disagrees_with_forced_race():
    rows = _valid_rows()
    for row in rows:
        if row["arm"] == "chimera_full_selector":
            row["linear_leaves_selected"] = True
    with pytest.raises(RuntimeError, match="full selector disagrees"):
        diagnosis.analyze(rows)


def test_analysis_rejects_selector_model_that_disagrees_with_winning_lane():
    rows = _valid_rows()
    for row in rows:
        if row["arm"] == "chimera_full_selector":
            row["model_sha256"] = "different"
    with pytest.raises(RuntimeError, match="model_sha256"):
        diagnosis.analyze(rows)


def test_analysis_rejects_missing_selector_decisions():
    rows = _valid_rows()
    for row in rows:
        if row["arm"] == "chimera_capped_selector":
            row["linear_leaves_selected"] = None
    with pytest.raises(RuntimeError, match="resolved linear-leaf"):
        diagnosis.analyze(rows)


def test_analysis_rejects_invalid_numeric_or_library_evidence():
    rows = _valid_rows()
    for row in rows:
        if row["arm"] == "chimera_product":
            row["test_rmse"] = -1.0
    with pytest.raises(RuntimeError, match="invalid numeric evidence"):
        diagnosis.analyze(rows)

    rows = _valid_rows()
    for row in rows:
        if row["arm"] == "chimera_product":
            row["library"] = "darkofit"
    with pytest.raises(RuntimeError, match="invalid library ledger"):
        diagnosis.analyze(rows)

    rows = _valid_rows()
    for row in rows:
        if row["arm"] == "chimera_product":
            row["test_rmse"] = "1.0"
    with pytest.raises(RuntimeError, match="invalid numeric evidence"):
        diagnosis.analyze(rows)


def test_analysis_rejects_contradictory_cross_pair_ledger():
    rows = _valid_rows()
    for row in rows:
        if row["arm"] == "chimera_product":
            row["cross_pair_count"] = 30
    with pytest.raises(RuntimeError, match="invalid selector ledger"):
        diagnosis.analyze(rows)

    rows = _valid_rows()
    for row in rows:
        if row["arm"] == "chimera_product":
            row["cross_features_selected"] = True
            row["cross_pair_count"] = 1
    with pytest.raises(RuntimeError, match="invalid selector ledger"):
        diagnosis.analyze(rows)


def test_arm_manifest_has_no_duplicate_or_undeclared_lane():
    assert len(diagnosis.ARMS) == len(set(diagnosis.ARMS))
    assert set(diagnosis.ARMS) == {
        "darko_default",
        "darko_matched_auto10_linear",
        "darko_matched_auto20_linear",
        "darko_shared_constant",
        "darko_shared_linear",
        "chimera_shared_constant",
        "chimera_shared_linear",
        "chimera_full_selector",
        "chimera_capped_selector",
        "chimera_full_product",
        "chimera_product",
    }


def test_rssi_artifact_create_is_atomic_and_create_only(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    diagnosis._atomic_create(output, b"first")
    assert output.read_bytes() == b"first"
    with pytest.raises(FileExistsError, match="refusing to replace"):
        diagnosis._atomic_create(output, b"second")
    monkeypatch.setattr(
        diagnosis,
        "_source_state",
        lambda *_args, **_kwargs: pytest.fail("diagnosis should not start"),
    )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        diagnosis.run(output)


def test_rssi_artifact_rejects_mutable_symlink_parent(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink output"):
        diagnosis._atomic_create(
            linked / "missing" / "result.json",
            b"result",
        )
    assert list(real.iterdir()) == []


def test_recorded_artifact_reproduces_binding_diagnosis(
    assert_analysis_equal,
):
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema_version"] == 1
    assert artifact["protocol"]["sha256"] == diagnosis._sha256(
        diagnosis.PROTOCOL
    )
    assert artifact["spent_boundary"][
        "registry_sha256"
    ] == diagnosis._sha256(diagnosis.REGISTRY)
    assert artifact["spent_boundary"][
        "prior_result_sha256"
    ] == diagnosis._sha256(diagnosis.PRIOR_RESULT)
    diagnosis.validate_artifact(artifact)
    analysis = diagnosis.analyze(artifact["results"])
    assert_analysis_equal(artifact["analysis"], analysis)
    assert artifact["spent_boundary"]["fresh_claim_eligible"] is False
    assert artifact["protocol"]["timing_claim_eligible"] is False
    assert analysis["capped_selector_disagrees_with_full"] is True
    assert analysis["chimera_product_cross_selected"] is False
    assert (
        analysis["test_rmse_ratios"][
            "darko_shared_constant_over_chimera_product"
        ]
        < 1.0
    )


def test_artifact_validation_rejects_forged_analysis_and_split_hash():
    artifact = json.loads(ARTIFACT.read_text())
    changed = copy.deepcopy(artifact)
    changed["analysis"]["forced_full_budget_validation_winner"] = "linear"
    with pytest.raises(RuntimeError, match="not reproducible"):
        diagnosis.validate_artifact(changed)

    changed = copy.deepcopy(artifact)
    changed["data"]["shared_fit_index_sha256"] = "forged"
    with pytest.raises(RuntimeError, match="data ledger"):
        diagnosis.validate_artifact(changed)

    changed = copy.deepcopy(artifact)
    changed["data"]["shared_fit_index_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="frozen evidence ledger"):
        diagnosis.validate_artifact(changed)

    changed = copy.deepcopy(artifact)
    changed["results"][0]["fit_seconds"] += 1.0
    with pytest.raises(RuntimeError, match="frozen artifact changed"):
        diagnosis.validate_artifact(changed)
