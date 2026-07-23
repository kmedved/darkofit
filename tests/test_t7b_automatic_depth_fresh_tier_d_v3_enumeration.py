from __future__ import annotations

import json
from pathlib import Path

from benchmarks import enumerate_t7b_automatic_depth_fresh_tier_d_v3 as enumeration


ROOT = Path(__file__).resolve().parents[1]


def test_only_declared_p1_resource_files_are_disclosure_paths():
    assert enumeration._is_disclosure_path(
        "benchmarks/t7b_automatic_depth_fresh_tier_d_execution_contract.json"
    )
    assert enumeration._is_disclosure_path(
        "benchmarks/enumerate_t7b_automatic_depth_fresh_tier_d_v3.py"
    )
    assert enumeration._is_disclosure_path("R2_PLAN.md")
    assert not enumeration._is_disclosure_path("benchmarks/m6_v3_raw.csv")
    assert not enumeration._is_disclosure_path("darkofit/booster.py")


def test_enumeration_evaluates_identities_independently(monkeypatch):
    registry = json.loads(
        (
            ROOT
            / "benchmarks"
            / "t7b_automatic_depth_fresh_tier_d_contamination_registry.json"
        ).read_text()
    )
    monkeypatch.setattr(
        enumeration,
        "_source_state",
        lambda path: {
            "path": str(path),
            "head": "a" * 40 if path == enumeration.ROOT else "b" * 40,
            "tree": "c" * 40,
            "clean": True,
            "status": [],
        },
    )
    monkeypatch.setattr(
        enumeration,
        "_git",
        lambda *_args, **_kwargs: "origin/test-enumeration",
    )
    monkeypatch.setattr(enumeration, "_history_hits", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(enumeration.v2_runner, "_known_fingerprints", lambda: ([], {}))
    monkeypatch.setattr(
        enumeration.v2_runner,
        "_preflight_lineage",
        lambda lineage, _known, _thresholds: (
            (_ for _ in ()).throw(
                enumeration.v2_runner.EligibilityError("synthetic ineligible")
            )
            if lineage["lineage_id"] == registry["lineages"][0]["lineage_id"]
            else {
                **lineage,
                "coordinates": [{"coordinate": value} for value in (0, 1, 2)],
            }
        ),
    )
    monkeypatch.setattr(
        enumeration,
        "_module_environment",
        lambda: {"python": "test", "modules": {}},
    )

    artifact = enumeration.enumerate_resources()

    assert artifact["candidate_pool"]["declared_identities"] == 40
    assert artifact["eligible_identity_count"] == 39
    assert artifact["ineligible_identity_count"] == 1
    rejected = [row for row in artifact["identities"] if row["status"] == "ineligible"]
    assert rejected == [
        {
            "lineage_id": registry["lineages"][0]["lineage_id"],
            "dataset_name": registry["lineages"][0]["dataset_name"],
            "task_id": registry["lineages"][0]["task_id"],
            "dataset_id": registry["lineages"][0]["dataset_id"],
            "stratum": registry["lineages"][0]["stratum"],
            "branch": registry["lineages"][0]["branch"],
            "split_kind": registry["lineages"][0]["split_kind"],
            "status": "ineligible",
            "reason": "synthetic ineligible",
            "history_hits": [],
            "resource_loaded": True,
        }
    ]
    assert artifact["attestations"]["abstract_slot_substitution_performed"] is False
    assert artifact["attestations"]["confirmation_panel_frozen"] is False
