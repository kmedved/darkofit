from __future__ import annotations

import json

import pytest

from benchmarks import run_group_centered_categorical_crosses_v1 as campaign
from benchmarks import (
    run_group_centered_categorical_crosses_v1_engagement as engagement,
)
from benchmarks import (
    run_group_centered_categorical_crosses_v1_invariants as invariants,
)


def _cross_record(*, eligible: bool, selected: bool):
    record = {
        "version": 1,
        "eligible": eligible,
        "reason": "selected_augmented" if selected else "control_won",
        "selected": selected,
        "pairs": [[1, 0]],
        "split": {},
        "control_validation_rmse": 1.0,
        "augmented_validation_rmse": 0.9 if selected else 1.1,
        "relative_validation_improvement": 0.1 if selected else -0.1,
        "selection_total_seconds": 1.0,
        "final_pairs": [[1, 0]] if selected else [],
        "final_preprocessing": {},
    }
    if eligible:
        record["fit_random_state_seed"] = 3
        record["selection_fits"] = []
    return record


def test_engagement_companion_binds_exact_m6_grid_and_foundation() -> None:
    identities = engagement.expected_identities()
    assert len(identities) == len(set(identities)) == 60
    assert {identity[1] for identity in identities} == {"medium"}
    assert {identity[2] for identity in identities} == {0, 1, 2}
    assert {identity[3] for identity in identities} == {"none", "stress"}
    assert (
        engagement.file_sha256(engagement.FOUNDATION_PATH)
        == engagement.FOUNDATION_SHA256
    )


def test_engagement_schema_accepts_regression_and_rejects_classification() -> None:
    selected = _cross_record(eligible=True, selected=True)
    row = {
        "model_metadata": json.dumps(
            {"group_centered_categorical_crosses": selected}
        )
    }
    assert engagement._cross_record(row, task="regression") == selected

    classification = {
        "model_metadata": json.dumps(
            {"group_centered_categorical_crosses": None}
        )
    }
    assert engagement._cross_record(classification, task="binary") == {
        "eligible": False,
        "selected": False,
        "reason": "classification_not_applicable",
    }
    with pytest.raises(RuntimeError, match="unexpectedly"):
        engagement._cross_record(row, task="multiclass")


def test_campaign_frozen_source_allowlists_are_exact() -> None:
    assert campaign.CONTROL_HEAD == (
        "01ae675bcebdf435988ce9e0d493d0fc0017f54a"
    )
    assert campaign.CANDIDATE_HEAD == (
        "c3f2608cd3033cfc00aa0737897a92ed868b5865"
    )
    assert campaign.INSPECTION_INDEX == 1
    assert campaign.MECHANISM_ID == (
        "group_centered_categorical_crosses_v1"
    )
    assert campaign.CANDIDATE_FILES == set(invariants.CANDIDATE_FILES)
    assert campaign.HARNESS_FILES == {
        "benchmarks/group_centered_categorical_crosses_v1_m6_engagement_companion.md",
        "benchmarks/run_group_centered_categorical_crosses_v1.py",
        "benchmarks/run_group_centered_categorical_crosses_v1_engagement.py",
        "benchmarks/run_group_centered_categorical_crosses_v1_invariants.py",
        "tests/test_group_centered_campaign_harness.py",
    }


def test_m5_changed_cell_parser_requires_complete_pairs() -> None:
    rows = []
    for seed in (0, 1):
        rows.extend(
            [
                {
                    "domain_id": "categorical_missing_regression",
                    "seed": seed,
                    "arm": "control",
                    "behavior_fingerprint_sha256": "control",
                },
                {
                    "domain_id": "categorical_missing_regression",
                    "seed": seed,
                    "arm": "candidate",
                    "behavior_fingerprint_sha256": "candidate",
                },
            ]
        )
    assert campaign._changed_m5_cells({"rows": rows}) == {
        ("categorical_missing_regression", 0),
        ("categorical_missing_regression", 1),
    }
    with pytest.raises(RuntimeError, match="incomplete"):
        campaign._changed_m5_cells({"rows": rows[:-1]})


def test_campaign_output_paths_are_external_and_create_only(tmp_path) -> None:
    paths = campaign.output_paths(tmp_path / "campaign")
    assert set(paths) == {
        "launch_manifest",
        "raw",
        "result",
        "m6_manifest",
        "terminal_attestation",
    }
    assert len(set(paths.values())) == len(paths)
