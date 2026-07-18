from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from benchmarks import analyze_basketball_sports_panel_v2 as analyzer
from benchmarks import build_basketball_sports_panel_v2 as panel
from benchmarks import run_basketball_sports_panel_v2 as runner


def _raw_like_frame() -> pd.DataFrame:
    rows = []
    for season in panel.SEASONS:
        for team_number in range(30):
            for game in range(2):
                value = float(team_number + game + 1)
                rows.append(
                    {
                        "Date": f"{season - 1}-11-{game + 1:02d}",
                        "Age": "25-183",
                        "Tm": f"T{team_number:02d}",
                        "GS": float(game == 0),
                        "Minutes": 260.0 + game,
                        "TS.": value / 100.0,
                        "eFG.": value / 101.0,
                        "ORB.": value / 10.0,
                        "DRB.": value / 9.0,
                        "TRB.": value / 8.0,
                        "AST.": value / 7.0,
                        "STL.": value / 6.0,
                        "BLK.": value / 5.0,
                        "TOV.": value / 4.0,
                        "USG.": value / 3.0,
                        "ORtg": 100.0 + value,
                        "DRtg": 110.0 - value,
                        "GmSc": value,
                        "BPM": value - 5.0,
                        "bref_id": f"p{team_number:02d}",
                        "Player": f"Player {team_number:02d}",
                        "year": season,
                    }
                )
    return pd.DataFrame(rows, columns=panel.RAW_COLUMNS)


def test_v2_panel_uses_only_fresh_seasons_and_middle_team_third():
    prepared = panel.prepare_panel(_raw_like_frame())
    assert tuple(sorted(prepared["year"].unique())) == panel.SEASONS
    for season in panel.SEASONS:
        assert panel.held_teams(prepared, season) == tuple(
            f"T{value:02d}" for value in range(10, 20)
        )


def test_v2_split_manifest_is_player_disjoint_and_partitions_rows():
    prepared = panel.prepare_panel(_raw_like_frame())
    manifest = panel.split_manifest(prepared)
    for season in panel.SEASONS:
        record = manifest["seasons"][str(season)]
        assert record["primary_rows"] == 20
        assert record["held_team_rows"] == 10
        assert record["seen_player_rows"] == 0
        assert record["cold_player_rows"] == 10
        observed = []
        primary = prepared.loc[
            (prepared["year"] == season) & ~prepared["Tm"].isin(record["held_teams"])
        ].reset_index(drop=True)
        groups = primary["bref_id"].astype(str).to_numpy()
        for fold in record["folds"]:
            train = np.asarray(fold["train_indices"], dtype=np.int64)
            test = np.asarray(fold["test_indices"], dtype=np.int64)
            assert set(groups[train]).isdisjoint(groups[test])
            observed.extend(test.tolist())
        assert sorted(observed) == list(range(len(primary)))


def test_v2_power_analysis_is_deterministic_and_sufficient():
    first = panel.power_analysis()
    second = panel.power_analysis()
    assert first == second
    assert first["cells"] == 9
    assert first["bootstrap_resamples"] == 2_000
    assert first["source_geometric_mean_rmse_ratio"] == pytest.approx(
        0.9967283642622231
    )
    assert 0.84 <= first["pass_probability"] <= 0.87
    assert first["passes"] is True


def _result(arm: str, ratio: float) -> dict:
    cells = []
    for season in panel.SEASONS:
        for target in panel.TARGET_COLUMNS:
            cells.append(
                {
                    "season": season,
                    "target": target,
                    "primary": {"rmse": ratio},
                    "guardrail": {
                        "scores": {
                            "held_team": {"rmse": ratio},
                            "seen_player": {"rmse": ratio},
                            "cold_player": {"rmse": ratio},
                        }
                    },
                }
            )
    return {"arm": arm, "cells": cells}


def test_v2_quality_rule_credits_safe_aggregate_improvement():
    comparison = analyzer._quality_comparison(
        _result("candidate", 0.995),
        _result("control", 1.0),
    )
    assert comparison["aggregate_rmse_ratio"] == pytest.approx(0.995)
    assert comparison["bootstrap_95_upper"] == pytest.approx(0.995)
    assert comparison["passes_quality"] is True


def test_v2_quality_rule_rejects_single_lineage_harm():
    candidate = _result("candidate", 0.99)
    candidate["cells"][0]["primary"]["rmse"] = 1.03
    comparison = analyzer._quality_comparison(
        candidate,
        _result("control", 1.0),
    )
    assert comparison["worst_lineage_ratio"] == pytest.approx(1.03)
    assert comparison["primary_gates"]["worst_lineage_at_most_1_020"] is False
    assert comparison["passes_quality"] is False


def test_v2_analyzer_requires_distinct_raw_and_output_paths(tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text("{}")
    with pytest.raises(RuntimeError, match="must be distinct"):
        analyzer._validate_paths(raw, raw, tmp_path / "report.md")


def test_v2_runner_freezes_public_row_oob_ensemble():
    model = runner.build_estimator(
        runner.CANDIDATE,
        runner.DEFAULT_CHIMERABOOST_REPO,
    )
    assert model.n_ensembles == 5
    assert model.ensemble_bootstrap == "rows"
    assert model.ensemble_shared_preprocessing is True
    assert model.random_state == 4
    assert model.thread_count == 18


def test_v2_ratio_summary_uses_seconds_scale_paired_ratios():
    summary = analyzer._ratio_summary([2.0, 2.1, 1.9])
    assert math.isclose(summary["median"], 2.0)
    assert summary["stable"] is True
