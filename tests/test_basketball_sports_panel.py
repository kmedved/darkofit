from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import pytest

from benchmarks import build_basketball_sports_panel as panel


def _raw_like_frame() -> pd.DataFrame:
    rows = []
    for season in panel.SEASONS:
        for team_number in range(30):
            for game in range(2):
                minutes = 260.0 + game
                value = float(team_number + game + 1)
                rows.append(
                    {
                        "Date": f"{season - 1}-11-{game + 1:02d}",
                        "Age": "25-183",
                        "Tm": f"T{team_number:02d}",
                        "GS": float(game == 0),
                        "Minutes": minutes,
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


def test_parse_age_accepts_canonical_values_and_rejects_bad_days():
    assert panel.parse_age("25-000") == 25.0
    assert math.isclose(panel.parse_age("25-183"), 25.0 + 183.0 / 365.25)
    with pytest.raises(ValueError, match="invalid Basketball Reference age day"):
        panel.parse_age("25-366")
    with pytest.raises(ValueError, match="invalid Basketball Reference age"):
        panel.parse_age("25")


def test_prepare_panel_has_frozen_columns_order_and_weighting():
    frame = _raw_like_frame()
    prepared = panel.prepare_panel(frame)
    assert prepared.columns.tolist() == [
        *panel.IDENTITY_COLUMNS,
        *panel.FEATURE_COLUMNS,
        *panel.TARGET_COLUMNS,
    ]
    assert len(prepared) == 90
    assert prepared.groupby("year").size().tolist() == [30, 30, 30]
    first = prepared.iloc[0]
    expected_weighted_bpm = ((1.0 - 5.0) * 260.0 + (2.0 - 5.0) * 261.0) / 521.0
    assert math.isclose(first["box_plus_minus"], expected_weighted_bpm)
    assert math.isclose(first["game_score"], 1.5)
    assert math.isclose(first["minutes_per_game"], 260.5)
    assert math.isclose(first["start_rate"], 0.5)
    assert prepared["age"].nunique() == 1
    assert (
        np.isfinite(prepared.loc[:, [*panel.FEATURE_COLUMNS, *panel.TARGET_COLUMNS]])
        .all()
        .all()
    )


def test_panel_bytes_and_split_manifest_are_deterministic():
    prepared = panel.prepare_panel(_raw_like_frame())
    shuffled = panel.prepare_panel(_raw_like_frame().sample(frac=1.0, random_state=9))
    assert panel.panel_csv_bytes(prepared) == panel.panel_csv_bytes(shuffled)
    first = panel.split_manifest(prepared)
    second = panel.split_manifest(shuffled)
    assert first == second
    assert first["held_teams"] == [f"T{value:02d}" for value in range(10)]
    for season in panel.SEASONS:
        record = first["seasons"][str(season)]
        assert record["primary_rows"] == 20
        assert record["held_team_rows"] == 10
        assert record["seen_player_rows"] == 0
        assert record["cold_player_rows"] == 10
        assert sum(row["test_rows"] for row in record["folds"]) == 20


def test_power_analysis_is_deterministic_and_sufficient():
    first = panel.power_analysis()
    second = panel.power_analysis()
    assert first == second
    assert first["cells"] == 9
    assert first["folds_per_cell"] == 10
    assert first["minimum_equal_cell_mean_delta"] == 0.0005
    assert 0.80 <= first["pass_probability"] <= 0.83
    assert first["passes"] is True


def test_prepare_panel_rejects_missing_numeric_values():
    frame = _raw_like_frame()
    frame.loc[0, "BPM"] = np.nan
    with pytest.raises(RuntimeError, match="contain missing values"):
        panel.prepare_panel(frame)


def test_split_manifest_changes_when_identity_changes():
    prepared = panel.prepare_panel(_raw_like_frame())
    changed = copy.deepcopy(prepared)
    changed.loc[0, "bref_id"] = "different"
    assert (
        panel.split_manifest(prepared)["split_manifest_sha256"]
        != panel.split_manifest(changed)["split_manifest_sha256"]
    )
