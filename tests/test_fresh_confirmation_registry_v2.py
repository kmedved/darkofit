from __future__ import annotations

import copy

import pytest

from benchmarks import amend_fresh_confirmation_registry_v2 as amendment


def test_v2_relabels_only_primary_stratum():
    artifact = amendment.build()

    assert artifact["stratum_counts"] == {
        "smooth_process": 14,
        "categorical": 3,
        "noisy_tabular": 3,
    }
    assert artifact["smooth_process_feature_profile_counts"] == {
        "numeric_only_complete": 5,
        "categorical_complete": 7,
        "categorical_with_missing": 2,
    }
    assert artifact["contamination_decisions_changed"] is False
    assert artifact["coordinates_changed"] is False
    assert artifact["power_design_changed"] is False


def test_v2_refuses_post_score_parent(monkeypatch):
    parent = {
        "registry_sha256": amendment.EXPECTED_V1_REGISTRY_SHA256,
        "confirmation_data_scored": True,
    }
    monkeypatch.setattr(amendment, "_sha256", lambda path: (
        amendment.EXPECTED_V1_FILE_SHA256
    ))
    monkeypatch.setattr(
        amendment.json,
        "loads",
        lambda text: copy.deepcopy(parent),
    )

    with pytest.raises(RuntimeError, match="after confirmation scoring"):
        amendment.build()
