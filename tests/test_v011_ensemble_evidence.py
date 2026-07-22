import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import analyze_v011_ensemble_evidence as analysis  # noqa: E402
import run_v011_ensemble_evidence as campaign  # noqa: E402


def test_v011_contract_constants_and_complete_exposure_stop_list():
    assert campaign.DARKOFIT_HEAD == "543604dd9860a28c30912f914b2cfccfcb99d783"
    assert campaign.CHIMERABOOST_HEAD == "f14be606b641f1bf0dc92bb14b3951f1fe631c6b"
    assert campaign.CATBOOST_VERSION == "1.2.10"
    assert campaign.REPRODUCTION_ABS_TOLERANCE == 1e-10
    assert campaign.claim_spec()["exposure_stop_conditions"] == [
        "correctness_failure",
        "unresolved_reproduction_failure",
    ]
    assert campaign.claim_spec()["performance_or_cost_gate"] is False
    assert len(campaign.QUALITY_CASES) == 13
    assert len(campaign.PREDICTION_ARMS) == 5
    assert len(campaign.BATCH_SIZES) == 4


def test_v011_orders_are_deterministic_complete_rotations():
    for case_id in campaign.QUALITY_CASES:
        orders = [campaign.quality_order(case_id, block) for block in range(3)]
        assert all(set(order) == set(campaign.QUALITY_ARMS) for order in orders)
        assert len({order[0] for order in orders}) == 3
    observed = {arm: set() for arm in campaign.PREDICTION_ARMS}
    for case_id in campaign.PREDICTION_CASES:
        for block in range(campaign.BLOCKS):
            order = campaign.prediction_order(case_id, block)
            assert set(order) == set(campaign.PREDICTION_ARMS)
            for position, arm in enumerate(order):
                observed[arm].add(position)
    assert all(len(positions) >= 4 for positions in observed.values())


def test_immutable_reproduction_ratios_are_complete_and_self_consistent():
    expected = campaign.immutable_ratios()
    assert set(expected["per_case"]) == set(campaign.QUALITY_CASES)
    assert analysis._geomean(expected["per_case"].values()) == pytest.approx(
        expected["pooled"], abs=1e-15
    )
    sports = [
        value
        for case_id, value in expected["per_case"].items()
        if case_id.startswith("sports_")
    ]
    general = [
        value
        for case_id, value in expected["per_case"].items()
        if case_id.startswith("general_")
    ]
    assert analysis._geomean(sports) == pytest.approx(expected["sports"], abs=1e-15)
    assert analysis._geomean(general) == pytest.approx(expected["general"], abs=1e-15)


def _reproduction_rows(delta=0.0):
    expected = campaign.immutable_ratios()["per_case"]
    rows = []
    for block in range(campaign.BLOCKS):
        for case_id in campaign.QUALITY_CASES:
            rows.extend(
                [
                    {
                        "kind": "quality",
                        "block": block,
                        "case_id": case_id,
                        "arm": campaign.DARKO_SINGLE,
                        "primary_loss": 2.0,
                    },
                    {
                        "kind": "quality",
                        "block": block,
                        "case_id": case_id,
                        "arm": campaign.DARKO_V3,
                        "primary_loss": 2.0 * (expected[case_id] + delta),
                    },
                ]
            )
    return rows


def test_reproduction_check_uses_frozen_per_case_and_aggregate_band():
    rows = _reproduction_rows(delta=0.0)
    for block in range(campaign.BLOCKS):
        campaign._check_reproduction(rows, block)
    result = analysis._reproduction(rows)
    assert result["passed"] is True
    assert all(
        record["within_frozen_tolerance"]
        for record in result["per_case"].values()
    )

    with pytest.raises(campaign.ReproductionMismatch):
        campaign._check_reproduction(
            _reproduction_rows(delta=campaign.REPRODUCTION_ABS_TOLERANCE * 2),
            0,
        )


def test_quality_uncertainty_is_seeded_and_clustered():
    reproduction = analysis._reproduction(_reproduction_rows())
    first = analysis._quality_uncertainty(reproduction)
    second = analysis._quality_uncertainty(reproduction)
    assert first == second
    assert set(first["sports"]["season_ratios"]) == {"2014", "2015", "2016"}
    assert len(first["sports"]["leave_one_season_out"]) == 3
    assert len(first["general"]["leave_one_case_out"]) == 4
    assert "not independent" in first["general"]["scope"]


def test_process_tree_sampler_preserves_primary_exception(monkeypatch):
    samples = iter([100, RuntimeError("cleanup failed")])

    def current_bytes():
        value = next(samples)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(
        campaign.ProcessTreeRSSSampler,
        "current_bytes",
        staticmethod(current_bytes),
    )
    with pytest.raises(ValueError, match="primary"):
        with campaign.ProcessTreeRSSSampler(interval_seconds=60):
            raise ValueError("primary")


def test_frozen_contract_loads_when_present():
    if not campaign.CONTRACT_PATH.exists():
        pytest.skip("prospective contract has not been frozen yet")
    contract = campaign.load_contract()
    assert contract["contract_frozen"] is True
    assert contract["outcome_blind"] is True
    assert contract["execution"] == campaign.execution_spec()
    assert contract["uncertainty"] == campaign.uncertainty_spec()
    assert contract["claims"] == campaign.claim_spec()


def test_protocol_names_catboost_stable_timing_and_no_performance_gate():
    text = campaign.PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "CatBoost: exact distribution version `1.2.10`" in text
    assert "five additional untimed pilot calls" in text
    assert "must last at least 1.0 second" in text
    assert "Every timing, memory, archive-size, dispersion" in text
    assert "No post-outcome acceptance bar may be added" in text


def test_execution_spec_roundtrips_strict_json():
    encoded = json.dumps(campaign.execution_spec(), allow_nan=False, sort_keys=True)
    assert json.loads(encoded) == campaign.execution_spec()
