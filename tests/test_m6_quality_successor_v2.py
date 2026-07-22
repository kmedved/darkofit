from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import m6_quality_rule_v2 as rule  # noqa: E402
import run_m6_quality_successor_v2 as execution  # noqa: E402
import run_m6_quality_successor_v2_backtest as backtest  # noqa: E402


def test_v2_freezes_exact_medium_grid_and_unchanged_rule():
    keys = rule.expected_pair_keys()
    assert rule.CONTRACT_ID == "m6-quality-successor-v2"
    assert len(keys) == len(set(keys)) == 60
    assert {key[1] for key in keys} == {"medium"}
    assert rule.REPEAT == 3
    assert rule.THREADS == 4
    assert rule.MAX_GEOMEAN_RATIO == 0.98
    assert rule.MIN_WIN_FRACTION == 0.60
    assert rule.MAX_CELL_RATIO == 1.02


def test_v2_backtest_reproduces_declared_advance_and_kill():
    positive = backtest.replay_positive(json.loads(backtest.POSITIVE_PATH.read_text()))
    negative = backtest.replay_negative(json.loads(backtest.NEGATIVE_PATH.read_text()))
    assert positive["observed_disposition"] == "advance"
    assert positive["agreement"] is True
    assert negative["observed_disposition"] == "kill"
    assert negative["agreement"] is True


def test_v2_comparison_command_attests_exact_coordinates_and_repeats(tmp_path):
    command = execution.comparison_command(
        control=tmp_path / "control",
        candidate=tmp_path / "candidate",
        raw_csv=tmp_path / "raw.csv",
    )
    assert command[command.index("--repeat") + 1] == "3"
    assert command[command.index("--threads") + 1] == "4"
    dataset_start = command.index("--datasets") + 1
    dataset_end = command.index("--sizes")
    assert tuple(command[dataset_start:dataset_end]) == rule.DATASETS
    assert "all" not in command[dataset_start:dataset_end]


def test_v2_prebacktest_binding_refuses_ranking():
    assert not execution.BACKTEST_RESULT_PATH.exists()
    with pytest.raises(RuntimeError, match="not committed"):
        execution.validate_backtest_binding()


def test_v2_rule_rejects_duplicate_and_incomplete_rows():
    rows = []
    for dataset, size, seed, weight in rule.expected_pair_keys():
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
    assert rule.analyze_rows(rows)["disposition"] == "advance"
    with pytest.raises(RuntimeError, match="duplicate arm"):
        rule.analyze_rows([*rows, rows[0]])
    with pytest.raises(RuntimeError, match="exact paired grid"):
        rule.analyze_rows(rows[:-1])


def test_v2_contract_discloses_known_outcome_and_structural_fixes():
    text = execution.CONTRACT_PATH.read_text()
    for token in (
        rule.CONTRACT_ID,
        "claims no outcome blindness",
        "contains no activation flag",
        "three repeats",
        "never `--datasets all`",
        "inspection index",
        backtest.POSITIVE_SHA256,
        backtest.NEGATIVE_SHA256,
    ):
        assert token in text
