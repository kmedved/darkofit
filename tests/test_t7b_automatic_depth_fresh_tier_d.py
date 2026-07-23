from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks import analyze_t7b_automatic_depth_fresh_tier_d as analyzer
from benchmarks import run_t7b_automatic_depth_fresh_tier_d as runner


ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    return json.loads((ROOT / "benchmarks" / name).read_text())


def test_execution_contract_registry_and_power_bindings_are_frozen():
    contract = _load("t7b_automatic_depth_fresh_tier_d_execution_contract_v2.json")
    registry = _load("t7b_automatic_depth_fresh_tier_d_contamination_registry.json")
    runner.validate_contract(contract)

    unhashed = {
        key: value for key, value in registry.items() if key != "registry_sha256"
    }
    assert registry["registry_sha256"] == runner.json_sha256(unhashed)
    primary = [row for row in registry["lineages"] if row["priority"] == 0]
    reserves = [row for row in registry["lineages"] if row["priority"] == 1]
    assert len(primary) == 32
    assert len(reserves) == 8
    assert all(
        row["target_blind_contamination_status"] == "eligible"
        and not row["target_blind_exposure_hits"]
        for row in registry["lineages"]
    )
    assert {
        stratum: sum(row["stratum"] == stratum for row in primary)
        for stratum in {
            "low_density_numeric",
            "low_density_categorical_or_grouped",
            "high_density_numeric",
            "high_density_categorical_or_grouped",
        }
    } == {
        "low_density_numeric": 8,
        "low_density_categorical_or_grouped": 8,
        "high_density_numeric": 8,
        "high_density_categorical_or_grouped": 8,
    }
    assert sum(row["split_kind"] == "group_hash_3fold" for row in primary) >= 4


def test_frozen_high_density_cap_keeps_weighted_coordinate_in_depth8():
    rows = runner.CAP_ROWS_PER_FEATURE * 4
    X = pd.DataFrame(np.arange(rows * 4, dtype=np.float64).reshape(rows, 4))
    y = np.arange(rows, dtype=np.float64)
    lineage = {
        "lineage_id": "synthetic_high",
        "branch": "depth_8",
        "split_kind": "row_hash_5fold",
    }
    view = runner._split_view(lineage, X, y, None, 1)

    assert view["weight_mode"] == "nonuniform"
    assert view["effective_rows_per_feature"] >= 2500.0
    assert view["effective_rows_per_feature"] < 2700.0
    assert (
        view["split_sha256"]
        == runner._split_view(lineage, X, y, None, 1)["split_sha256"]
    )


def test_group_safe_outer_split_is_disjoint_and_low_density():
    groups = pd.Series(np.repeat([f"g{i}" for i in range(30)], 3))
    X = pd.DataFrame(
        {
            "group": groups,
            "x": np.arange(len(groups), dtype=np.float64),
            "z": np.arange(len(groups), dtype=np.float64) ** 2,
        }
    )
    y = np.arange(len(groups), dtype=np.float64)
    lineage = {
        "lineage_id": "synthetic_grouped_low",
        "branch": "depth_4",
        "split_kind": "group_hash_3fold",
    }
    view = runner._split_view(lineage, X, y, groups, 1)
    train_groups = set(view["groups_train"].astype(str))
    test_mask = np.asarray(
        [
            runner._hash_mod(3, "synthetic_grouped_low", str(value), 20260724) == 1
            for value in groups
        ]
    )
    test_groups = set(groups[test_mask].astype(str))

    assert train_groups.isdisjoint(test_groups)
    assert view["effective_rows_per_feature"] < 100.0
    assert view["weight_mode"] == "nonuniform"


def test_preflight_uses_frozen_same_stratum_reserve_queue(monkeypatch):
    failed = "galaxy_velocity"

    monkeypatch.setattr(runner, "validate_contract", lambda _contract: None)
    monkeypatch.setattr(runner, "_known_fingerprints", lambda: ([], {}))

    def fake_preflight(lineage, _known, _thresholds):
        if lineage["lineage_id"] == failed:
            raise runner.EligibilityError("synthetic value-free failure")
        return {
            **lineage,
            "status": "eligible",
            "coordinates": [{"coordinate": coordinate} for coordinate in (0, 1, 2)],
        }

    monkeypatch.setattr(runner, "_preflight_lineage", fake_preflight)
    artifact = runner.build_preflight()
    selected = {row["slot"]: row for row in artifact["active_lineages"]}

    replacement = selected["low_density_numeric_06"]
    assert replacement["priority"] == 1
    assert replacement["lineage_id"] == "chscase_census5"
    assert replacement["registry_identity_slot"] == ("reserve_low_density_numeric_01")
    assert artifact["rejected_frozen_candidates"] == [
        {
            "slot": "low_density_numeric_06",
            "priority": 0,
            "lineage_id": failed,
            "reason": "synthetic value-free failure",
            "value_free": True,
        }
    ]


def _synthetic_raw(*, candidate_quality=0.98, depth8_quality=None):
    rows = []
    for lineage_index in range(32):
        branch = "depth_4" if lineage_index < 16 else "depth_8"
        quality = (
            candidate_quality
            if depth8_quality is None or branch == "depth_4"
            else depth8_quality
        )
        for coordinate in range(3):
            common = {
                "status": "ok",
                "lineage_id": f"lineage_{lineage_index:02d}",
                "slot": f"slot_{lineage_index:02d}",
                "stratum": (
                    "low_density_numeric"
                    if branch == "depth_4"
                    else "high_density_numeric"
                ),
                "branch": branch,
                "coordinate": coordinate,
                "weight_mode": ("nonuniform" if coordinate == 1 else "ordinary"),
                "task_id": 10_000 + lineage_index,
                "dataset_id": 20_000 + lineage_index,
                "split_sha256": f"split-{lineage_index}-{coordinate}",
                "train_rows": 1000,
                "test_rows": 250,
                "input_features": 10,
                "predict_seconds_repeats": [1.0, 1.01, 0.99],
                "peak_process_tree_rss_bytes": 200_000_000,
                "integrity_passes": True,
                "ambient_thread_restored": True,
                "safe_npz_exact": True,
            }
            rows.append(
                {
                    **common,
                    "arm": "control",
                    "rmse": 1.0,
                    "fit_seconds": 2.0,
                }
            )
            rows.append(
                {
                    **common,
                    "arm": "candidate",
                    "rmse": quality,
                    "fit_seconds": 1.8,
                    "predict_seconds_repeats": [0.9, 0.91, 0.89],
                    "peak_process_tree_rss_bytes": 195_000_000,
                }
            )
    return {
        "schema_version": 1,
        "contract_id": runner.CONTRACT_ID,
        "complete": True,
        "environment": {"physical_memory_bytes": 64 * 1024**3},
        "rows": rows,
    }


def test_analyzer_go_requires_all_quality_cost_and_integrity_gates():
    contract = _load("t7b_automatic_depth_fresh_tier_d_execution_contract_v2.json")
    power = _load("t7b_automatic_depth_fresh_tier_d_power_design_contract.json")
    result = analyzer.analyze(_synthetic_raw(), contract, power)

    assert result["go"] is True
    assert result["quality"]["passes"] is True
    assert result["costs"]["passes"] is True
    assert result["integrity"]["passes"] is True
    assert result["disposition"] == ("go_promote_automatic_depth_default_v0_12")


def test_analyzer_no_go_when_depth8_branch_is_harmful():
    contract = _load("t7b_automatic_depth_fresh_tier_d_execution_contract_v2.json")
    power = _load("t7b_automatic_depth_fresh_tier_d_power_design_contract.json")
    result = analyzer.analyze(
        _synthetic_raw(candidate_quality=0.97, depth8_quality=1.001),
        contract,
        power,
    )

    assert result["go"] is False
    assert result["quality"]["component_passes"]["each_branch_direction"] is False
    assert result["disposition"] == ("no_go_close_automatic_depth_default_candidate")


def test_target_attestation_rejects_nonfinite_without_reporting_statistics():
    with pytest.raises(runner.EligibilityError, match="non-finite"):
        runner._target_attestation(np.array([1.0, np.nan]), expected_rows=2)
