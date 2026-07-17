from __future__ import annotations

import copy
from pathlib import Path

import pytest

from benchmarks import analyze_basketball_sports_panel as analyzer
from benchmarks import build_basketball_sports_panel as panel
from benchmarks import run_basketball_sports_panel as runner


def _cell(season: int, target: str, score: float, guard: float) -> dict:
    return {
        "season": season,
        "target": target,
        "primary_mean_r2": score,
        "folds": [
            {
                "fold": fold,
                "r2": score,
                "test_indices": [fold],
                "prediction_sha256": f"prediction-{fold}",
                "fit_metadata": {"fitted_tree_count": 1000},
            }
            for fold in range(10)
        ],
        "guardrail": {
            "scores": {
                view: {
                    "rows": 10,
                    "r2": guard,
                    "target_sha256": "target",
                    "prediction_sha256": "prediction",
                }
                for view in (
                    "overlap_exposed_team_holdout",
                    "seen_player_subset",
                    "cold_player_subset",
                )
            },
            "prediction_sha256": "guard-prediction",
            "fit_metadata": {"fitted_tree_count": 1000},
        },
    }


def _result(
    arm: str,
    score: float,
    guard: float,
    fit: float,
    rss: float,
) -> dict:
    cells = [
        _cell(season, target, score, guard)
        for season in panel.SEASONS
        for target in panel.TARGET_COLUMNS
    ]
    return {
        "arm": arm,
        "implementation": {"package": arm},
        "cells": cells,
        "equal_cell_mean_r2": score,
        "total_fit_seconds": fit,
        "total_predict_seconds": fit / 10.0,
        "steady_wall_seconds": fit * 1.1,
        "peak_rss_bytes": rss,
        "behavior_fingerprint_sha256": f"fingerprint-{arm}",
    }


def _raw() -> dict:
    results = {
        runner.CONTROL: _result(runner.CONTROL, 0.50, 0.40, 10.0, 100.0),
        runner.CANDIDATE: _result(runner.CANDIDATE, 0.502, 0.401, 12.0, 105.0),
        runner.CHIMERABOOST: _result(runner.CHIMERABOOST, 0.499, 0.399, 14.0, 110.0),
        runner.CATBOOST: _result(runner.CATBOOST, 0.501, 0.400, 16.0, 120.0),
    }
    repeats = []
    for block, order in enumerate(runner.BLOCK_ORDERS):
        for position, arm in enumerate(order):
            repeats.append(
                {
                    "block": block,
                    "position": position,
                    "order": list(order),
                    "arm": arm,
                    "result": copy.deepcopy(results[arm]),
                }
            )
    return {
        "schema_version": 1,
        "name": "darkofit_basketball_sports_panel_raw_v1",
        "runner": {"sha256": "runner"},
        "protocol": {"sha256": "protocol"},
        "panel_manifest": {"processed_panel_sha256": "panel"},
        "behavior_fingerprints": {
            arm: f"fingerprint-{arm}" for arm in runner.ARM_ORDER
        },
        "repeats": repeats,
    }


def test_block_orders_are_reciprocal_and_complete():
    assert len(runner.BLOCK_ORDERS) == 3
    assert all(set(order) == set(runner.ARM_ORDER) for order in runner.BLOCK_ORDERS)
    assert runner.BLOCK_ORDERS[0][:2] == (runner.CONTROL, runner.CANDIDATE)
    assert runner.BLOCK_ORDERS[1][-2:] == (runner.CANDIDATE, runner.CONTROL)


def test_analyzer_passes_candidate_and_external_claims():
    result = analyzer.analyze(_raw(), "raw")
    assert result["candidate"]["passes"] is True
    assert result["eligible_darkofit_arm"] == runner.CANDIDATE
    assert result["candidate"]["global_default_change_authorized"] is False
    assert result["claims"] == {
        "beats_chimeraboost_on_s4": True,
        "beats_catboost_on_s4": True,
    }
    assert result["panel_spent"] is True
    assert result["retuning_on_panel_authorized"] is False


def test_analyzer_closes_candidate_on_cold_player_regression():
    raw = _raw()
    for record in raw["repeats"]:
        if record["arm"] == runner.CANDIDATE:
            for cell in record["result"]["cells"]:
                cell["guardrail"]["scores"]["cold_player_subset"]["r2"] = 0.39
    result = analyzer.analyze(raw, "raw")
    assert result["candidate"]["passes"] is False
    assert result["eligible_darkofit_arm"] == runner.CONTROL
    assert result["candidate"]["decision"] == (
        "close_random_strength_0_5_without_s4_confirmation"
    )


def test_analyzer_rejects_nonreproducing_behavior():
    raw = _raw()
    raw["repeats"][0]["result"]["behavior_fingerprint_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="behavior changed"):
        analyzer.analyze(raw, "raw")


def test_analyzer_rejects_ambiguous_paths(tmp_path: Path):
    raw = tmp_path / "raw.json"
    raw.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must be distinct"):
        analyzer._validate_paths(raw, raw, tmp_path / "report.md")
