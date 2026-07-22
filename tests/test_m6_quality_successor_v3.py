from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import m6_quality_rule_v3 as rule  # noqa: E402
import run_m6_quality_successor_v3 as execution  # noqa: E402
import run_m6_quality_successor_v3_backtest as backtest  # noqa: E402


EXPECTED_RESULT_SHA256 = (
    "35cc54acfeb7de7950966445ed8248654f945072e5e5900e3333fff4b15129b6"
)


def test_v3_freezes_exact_medium_grid_without_win_gate():
    keys = rule.expected_pair_keys()
    assert rule.CONTRACT_ID == "m6-quality-successor-v3"
    assert len(keys) == len(set(keys)) == 60
    assert {key[1] for key in keys} == {"medium"}
    assert rule.REPEAT == 3
    assert rule.THREADS == 4
    assert rule.MAX_AGGREGATE_RATIO == 1.0
    assert rule.MAX_GROUP_RATIO == 1.02
    assert rule.MAX_LOO_RATIO == 1.003
    assert not hasattr(rule, "MIN_WIN_FRACTION")


def test_v3_advances_selective_no_harm_and_kills_real_harm_or_concentration():
    selective = rule.quality_decision({
        **{f"tie-{index}": 1.0 for index in range(12)},
        "gain-a": 0.88,
        "gain-b": 0.98,
    })
    harm = rule.quality_decision({
        **{f"gain-{index}": 0.95 for index in range(9)},
        "harm": 1.03,
    })
    concentrated = rule.quality_decision({
        "large-gain": 0.50,
        "small-harm-a": 1.01,
        "small-harm-b": 1.01,
    })
    assert selective["disposition"] == "advance"
    assert harm["disposition"] == "kill"
    assert harm["gates"]["worst_group_at_most_1_02"] is False
    assert concentrated["disposition"] == "kill"
    assert concentrated["gates"]["loo_concentration_at_most_1_003"] is False
    assert "wins" not in selective


@pytest.mark.parametrize(
    ("ratios", "groups"),
    [
        ({}, None),
        ({"x": -0.1}, None),
        ({"x": float("nan")}, None),
        ({"x": True}, None),
        ({"": 0.9}, None),
        ({"a": 0.9, "b": 1.0}, {"a": "same", "b": "same"}),
        ({"a": 0.9, "b": 1.0}, {"a": "a"}),
    ],
)
def test_v3_rule_rejects_invalid_ratios_or_groups(ratios, groups):
    with pytest.raises(RuntimeError):
        rule.quality_decision(ratios, groups=groups)


def test_v3_backtest_replays_surviving_verdicts_and_retired_tripwire():
    positive = backtest.replay_positive(json.loads(backtest.POSITIVE_PATH.read_text()))
    negative = backtest.replay_negative(json.loads(backtest.NEGATIVE_PATH.read_text()))
    retired = backtest.replay_retired_selector(
        json.loads(backtest.RETIRED_SELECTOR_PATH.read_text())
    )
    assert positive["observed_disposition"] == "advance"
    assert positive["agreement"] is True
    assert negative["observed_disposition"] == "kill"
    assert negative["agreement"] is True
    assert retired["observed_disposition"] == "advance"
    assert retired["agreement"] is True
    assert retired["audit_role"] == "abolished_verdict_tripwire_not_new_evidence"


def test_v3_backtest_inputs_match_predeclared_hashes():
    assert execution.file_sha256(backtest.POSITIVE_PATH) == backtest.POSITIVE_SHA256
    assert execution.file_sha256(backtest.NEGATIVE_PATH) == backtest.NEGATIVE_SHA256
    assert execution.file_sha256(backtest.RETIRED_SELECTOR_PATH) == (
        backtest.RETIRED_SELECTOR_SHA256
    )


def test_v3_comparison_command_attests_coordinates_and_repeats(tmp_path):
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


def test_v3_analyze_rows_groups_coordinates_by_dataset():
    rows = []
    for dataset, size, seed, weight in rule.expected_pair_keys():
        for arm, value in (("control_default", 1.0), ("candidate_default", 0.99)):
            rows.append({
                "dataset": dataset,
                "size": size,
                "seed": seed,
                "weight_mode": weight,
                "variant": arm,
                "primary_metric": "rmse",
                "primary_value": value,
            })
    result = rule.analyze_rows(rows)
    assert result["disposition"] == "advance"
    assert result["case_count"] == 60
    assert result["group_count"] == 10
    assert set(result["group_geometric_mean_ratio"]) == set(rule.DATASETS)
    assert set(result["leave_one_group_out_geometric_mean_ratio"]) == set(
        rule.DATASETS
    )
    with pytest.raises(RuntimeError, match="duplicate arm"):
        rule.analyze_rows([*rows, rows[0]])
    with pytest.raises(RuntimeError, match="exact paired grid"):
        rule.analyze_rows(rows[:-1])


def test_v3_contract_discloses_supersession_and_nonshipping_boundary():
    text = execution.CONTRACT_PATH.read_text()
    for token in (
        rule.CONTRACT_ID,
        "no win count",
        "no minimum worthwhile effect size",
        "never `--datasets all`",
        "claims no outcome blindness",
        "obsolete-verdict tripwire",
        backtest.POSITIVE_SHA256,
        backtest.NEGATIVE_SHA256,
        backtest.RETIRED_SELECTOR_SHA256,
    ):
        assert token in text
    supersession = execution.SUPERSESSION_PATH.read_text()
    assert "no forward ranking\n  authority" in supersession
    assert "no v2 outcome is relabeled" in supersession


def test_v3_binding_matches_create_only_result_when_present(monkeypatch):
    if not execution.BACKTEST_RESULT_PATH.exists():
        pytest.skip("v3 one-shot result is created only after harness commit")
    payload = json.loads(execution.BACKTEST_RESULT_PATH.read_text())
    monkeypatch.setattr(
        execution,
        "source_state",
        lambda _path: {
            "clean": True,
            "head": payload["harness"]["head"],
            "tree": payload["harness"]["tree"],
        },
    )
    monkeypatch.setattr(
        execution,
        "_tracked_head_bytes",
        lambda path: path.read_bytes(),
    )
    monkeypatch.setattr(execution, "_is_ancestor", lambda _a, _b: True)
    binding = execution.validate_backtest_binding()
    assert len(binding["result_sha256"]) == 64


def test_v3_create_only_result_has_expected_terminal_identity():
    assert execution.BACKTEST_RESULT_PATH.is_file()
    assert hashlib.sha256(execution.BACKTEST_RESULT_PATH.read_bytes()).hexdigest() == (
        EXPECTED_RESULT_SHA256
    )
    payload = json.loads(execution.BACKTEST_RESULT_PATH.read_text())
    assert payload["backtest_complete"] is True
    assert payload["candidate_ranking_eligible"] is True
    assert payload["rerun_authorized"] is False
    assert payload["harness"]["head"] == (
        "f3d19ebb4d9306e278a52534a7856650675d1166"
    )
    assert all(replay["agreement"] is True for replay in payload["replays"])
