"""Frozen-rule tests for the M6 historical backtest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from run_m6_historical_backtest import (  # noqa: E402
    SELECTOR_CASES,
    analyze_fused,
    analyze_packed,
    analyze_selector,
    validate_sources,
)


def test_fused_rule_reproduces_the_declared_positive_verdict():
    artifact = json.loads(
        (BENCH_DIR / "fused_variable_hessian.json").read_text()
    )

    analysis = analyze_fused(artifact)

    assert analysis["observed_disposition"] == "advance"
    assert analysis["agreement"] is True
    assert analysis["fit_geomean_ratio"] <= 0.90


def test_packed_rule_reproduces_the_declared_negative_verdict():
    artifact = json.loads(
        (BENCH_DIR / "basketball_packed_prediction.json").read_text()
    )

    analysis = analyze_packed(artifact)

    assert analysis["observed_disposition"] == "kill"
    assert analysis["agreement"] is True
    assert (
        analysis["gates"]["large_candidate_over_legacy_at_most_1_10"]
        is False
    )


def _selector_rows(ratios, improvements=None):
    improvements = improvements or [0.0] * len(ratios)
    return [
        {
            "dataset": dataset,
            "size": size,
            "default_rmse": 1.0,
            "selector_rmse": ratio,
            "selected_linear": improvement >= 0.03,
            "relative_validation_improvement": improvement,
        }
        for (dataset, size), ratio, improvement in zip(
            SELECTOR_CASES, ratios, improvements
        )
    ]


def test_selector_rule_requires_effect_wins_and_no_harm_together():
    advance = analyze_selector(
        _selector_rows([0.97, 0.97, 0.98, 0.99, 0.97, 0.98])
    )
    kill = analyze_selector(
        _selector_rows([0.90, 0.90, 0.90, 0.90, 1.03, 1.03])
    )

    assert advance["observed_disposition"] == "advance"
    assert kill["observed_disposition"] == "kill"
    assert kill["gates"]["no_cell_ratio_above_1_02"] is False


def test_backtest_source_validation_requires_all_exact_clean_pins():
    state = {"clean": True, "head": "ignored"}
    fused = {
        "clean": True,
        "head": "1016e7e8d70c403a70feab7762de8837ea8fd09c",
    }
    packed = {
        "clean": True,
        "head": "e961bcc2ea64706169641722b5935f9f31402fa3",
    }
    selector = {
        "clean": True,
        "head": "29bd30cdcf476139c30efe4e09773ca812ba443f",
    }
    chimera = {
        "clean": True,
        "head": "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d",
    }

    validate_sources(
        **{
            "harness": state,
            "fused": fused,
            "packed": packed,
            "selector": selector,
            "chimeraboost_015": chimera,
        }
    )

    with pytest.raises(RuntimeError, match="expected"):
        validate_sources(
            **{
                "harness": state,
                "fused": {**fused, "head": "0" * 40},
                "packed": packed,
                "selector": selector,
                "chimeraboost_015": chimera,
            }
        )
