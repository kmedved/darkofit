import copy
import json
from pathlib import Path

import pytest

from benchmarks import build_t5_composite_registry as registry


ROOT = Path(__file__).resolve().parents[1]
DECLARATIONS = json.loads(registry.DECLARATIONS.read_text())


def test_declarations_freeze_25_unique_lineages_and_75_coordinates():
    rows = DECLARATIONS["candidates"]
    assert len(rows) == 25
    assert len({row["task_id"] for row in rows}) == 25
    assert len({row["dataset_id"] for row in rows}) == 25
    assert len({row["lineage_cluster"] for row in rows}) == 25
    assert DECLARATIONS["coordinate_folds"] == [0, 1, 2]
    assert {
        name: sum(row["stratum"] == name for row in rows)
        for name in registry.EXPECTED_STRATA
    } == registry.EXPECTED_STRATA


def test_power_analysis_is_deterministic_and_covers_shipping_rule():
    first = registry.power_analysis()
    second = registry.power_analysis()
    assert first == second
    assert first["simulations"] == 200_000
    assert first["simulated_lineages"] == 25
    assert len(first["effect_profiles"]) == 15
    assert first["effect_pool_construction"]["a10_omitted_lineage"] == "diamonds"
    assert first["pass_probability"] >= 0.80
    assert first["passes"] is True
    assert first["gates"] == {
        "equal_dataset_geomean_ratio_at_most": 0.995,
        "normal_approximation_one_sided_95_upper_at_most": 1.002,
        "least_favorable_leave_one_out_ratio_at_most": 0.998,
        "worst_dataset_ratio_at_most": 1.005,
    }


def test_power_fails_closed_if_effect_pool_is_null(monkeypatch):
    profiles, construction = registry._power_sources()
    null = [{**row, "ratio": 1.0} for row in profiles]
    monkeypatch.setattr(
        registry, "_power_sources", lambda: (copy.deepcopy(null), construction)
    )
    result = registry.power_analysis()
    assert result["pass_probability"] == 0.0
    assert result["passes"] is False


def test_protocol_uses_no_win_count_and_freezes_guarded_candidate():
    text = registry.PROTOCOL.read_text()
    assert "Win counts are descriptive only" in text
    assert "challenger <= 0.995 * control" in text
    assert "cross <= 0.95 * uncrossed" in text
    assert "100,000 replicates" in text
    assert "No-rerun rule" in text


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        ("UCC", False),
        ("child", False),
        ("fifa", False),
        ("colrec", True),
        ("bike-sharing-domain-generalization", True),
    ],
)
def test_repository_literal_requires_six_discriminating_characters(
    literal, expected
):
    assert registry._repository_literal_is_discriminating(literal) is expected


def test_recorded_registry_is_target_blind_and_only_authorizes_execution():
    if not registry.DEFAULT_OUTPUT.exists():
        pytest.skip("registry is generated only after its builder is committed")
    artifact = json.loads(registry.DEFAULT_OUTPUT.read_text())
    assert artifact["task_count"] == 25
    assert artifact["lineage_count"] == 25
    assert artifact["coordinate_count"] == 75
    assert artifact["stratum_counts"] == registry.EXPECTED_STRATA
    assert artifact["pairwise_candidate_near_match_alarms"] == []
    assert artifact["confirmation_outcomes_inspected"] is False
    assert artifact["target_statistics_used"] is False
    assert artifact["lockbox_data_used"] is False
    assert artifact["confirmation_run_authorized"] is True
    assert artifact["default_promotion_authorized"] is False
    assert artifact["power_analysis"]["passes"] is True
    assert all(row["status"] == "eligible" for row in artifact["tasks"])
