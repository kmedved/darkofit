from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import m6_quality_successor as successor  # noqa: E402
import run_m6_quality_successor_backtest as backtest  # noqa: E402


def test_successor_contract_freezes_medium_diverse_exact_grid():
    keys = successor.expected_pair_keys()
    assert len(keys) == 60
    assert len(set(keys)) == 60
    assert {key[1] for key in keys} == {"medium"}
    assert {key[2] for key in keys} == {"0", "1", "2"}
    assert {key[3] for key in keys} == {"none", "stress"}
    assert len({key[0] for key in keys}) == 10
    assert successor.REPEAT == 3
    assert successor.THREADS == 4


def test_quality_rule_advances_broad_gain_and_kills_narrow_gain():
    advance = successor.quality_decision({
        f"case-{index}": 0.97 + 0.001 * (index % 3)
        for index in range(10)
    })
    narrow = successor.quality_decision({
        **{f"win-{index}": 0.90 for index in range(2)},
        **{f"tie-{index}": 1.0 for index in range(8)},
    })
    harm = successor.quality_decision({
        **{f"win-{index}": 0.94 for index in range(9)},
        "harm": 1.03,
    })
    assert advance["disposition"] == "advance"
    assert narrow["disposition"] == "kill"
    assert narrow["gates"]["wins_at_least_60_percent"] is False
    assert harm["disposition"] == "kill"
    assert harm["gates"]["no_cell_above_1_02"] is False


@pytest.mark.parametrize(
    "ratios",
    [{}, {"x": -0.1}, {"x": float("nan")}, {"x": True}, {"": 0.9}],
)
def test_quality_rule_rejects_invalid_ratios(ratios):
    with pytest.raises(RuntimeError):
        successor.quality_decision(ratios)


def test_predeclared_backtest_reproduces_known_advance_and_kill():
    positive = json.loads(backtest.POSITIVE_PATH.read_text())
    negative = json.loads(backtest.NEGATIVE_PATH.read_text())
    positive_result = backtest.replay_positive(positive)
    negative_result = backtest.replay_negative(negative)
    assert positive_result["observed_disposition"] == "advance"
    assert positive_result["agreement"] is True
    assert negative_result["observed_disposition"] == "kill"
    assert negative_result["agreement"] is True


def test_backtest_inputs_match_predeclared_hashes():
    assert successor.file_sha256(backtest.POSITIVE_PATH) == backtest.POSITIVE_SHA256
    assert successor.file_sha256(backtest.NEGATIVE_PATH) == backtest.NEGATIVE_SHA256


def test_unbound_successor_refuses_candidate_ranking():
    assert successor.BACKTEST_COMPLETE is False
    with pytest.raises(RuntimeError, match="not complete"):
        successor.validate_backtest_binding()


def test_analyze_rows_requires_exact_grid_and_computes_loo():
    rows = []
    for dataset, size, seed, weight in successor.expected_pair_keys():
        for arm, value in (("control_default", 1.0), ("candidate_default", 0.97)):
            rows.append({
                "dataset": dataset,
                "size": size,
                "seed": seed,
                "weight_mode": weight,
                "variant": arm,
                "primary_metric": "rmse",
                "primary_value": value,
            })
    result = successor.analyze_rows(rows)
    assert result["disposition"] == "advance"
    assert result["case_count"] == 60
    assert set(result["per_dataset_geometric_mean_ratio"]) == set(successor.DATASETS)
    assert set(result["leave_one_dataset_out_geometric_mean_ratio"]) == set(successor.DATASETS)

    with pytest.raises(RuntimeError, match="exact paired grid"):
        successor.analyze_rows(rows[:-1])


def test_contract_declares_inspection_and_nonshipping_boundaries():
    text = successor.CONTRACT_PATH.read_text()
    for token in (
        successor.CONTRACT_ID,
        "positive one-based",
        "inspection index",
        "10,000-row `medium` cells",
        "60 paired cells",
        "can never authorize shipping",
        backtest.POSITIVE_SHA256,
        backtest.NEGATIVE_SHA256,
    ):
        assert token in text
