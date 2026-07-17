from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import amend_fresh_confirmation_registry_v2 as amendment


RECORDED_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "fresh_confirmation_registry_v2.json"
)
EXPECTED_ARTIFACT_SHA256 = (
    "0d878d690e32f6781a170fa3e5c232eef13d20d51d25b352c96a20ddc87e3970"
)


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


def test_recorded_v2_supersedes_only_the_primary_label():
    raw = RECORDED_ARTIFACT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_ARTIFACT_SHA256
    artifact = json.loads(raw)

    assert artifact["registry_v2_sha256"] == (
        "29e14fd855e0190e175e0aa27915d00f41c39978759c6f415f6e43fe245fba5d"
    )
    assert artifact["parent"]["file_sha256"] == (
        amendment.EXPECTED_V1_FILE_SHA256
    )
    assert artifact["protocol_sha256"] == hashlib.sha256(
        amendment.PROTOCOL.read_bytes()
    ).hexdigest()
    assert artifact["builder_source_sha256"] == hashlib.sha256(
        Path(amendment.__file__).read_bytes()
    ).hexdigest()
    assert artifact["task_count"] == artifact["lineage_count"] == 20
    assert artifact["coordinate_count"] == 60
    assert artifact["stratum_counts"]["smooth_process"] == 14
    assert artifact["contamination_decisions_changed"] is False
    assert artifact["coordinates_changed"] is False
    assert artifact["power_design_changed"] is False
    assert artifact["confirmation_data_scored"] is False
    assert artifact["selector_promotion_authorized"] is False
    assert artifact["lockbox_run_authorized"] is False
