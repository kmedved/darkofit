from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from benchmarks import analyze_basketball_cold_player_guardrail as analysis
from benchmarks import basketball_guardrails as guardrails


def _frame():
    rows = []
    assignments = (
        ("A", "seen-1"),
        ("A", "seen-2"),
        ("B", "cold-1"),
        ("B", "cold-2"),
        ("C", "seen-1"),
        ("C", "seen-2"),
        ("D", "train-2"),
        ("E", "train-3"),
        ("F", "train-4"),
    )
    for index, (team, player) in enumerate(assignments):
        row = {
            feature: float(index + offset)
            for offset, feature in enumerate(guardrails.creator.FEATURES)
        }
        row.update(
            {
                "Player": player,
                "Tm": team,
                "MP": 600.0 + index,
                "G": 60.0,
                "GS": 30.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _patch_creator(monkeypatch, frame):
    filtered = frame.loc[frame["MP"] > 500].copy()
    filtered["MPG"] = filtered["MP"] / filtered["G"]
    train = filtered.loc[~filtered["Tm"].isin(["A", "B"])]
    X = train.loc[:, guardrails.creator.FEATURES]
    y = train.loc[:, "MPG"]
    monkeypatch.setattr(
        guardrails.creator,
        "prepare_creator_data",
        lambda value: (X, y, {"test_rows": 4}),
    )


def test_prepare_player_guardrail_discloses_overlap(monkeypatch):
    frame = _frame()
    _patch_creator(monkeypatch, frame)

    data = guardrails.prepare_player_guardrail(frame)

    assert data.metadata["test_teams"] == ["A", "B"]
    assert data.metadata["train_unique_players"] == 5
    assert data.metadata["holdout_unique_players"] == 4
    assert data.metadata["overlapping_unique_players"] == 2
    assert data.metadata["overlapping_unique_player_rate"] == 0.5
    assert data.metadata["seen_player_holdout_rows"] == 2
    assert data.metadata["cold_player_holdout_rows"] == 2
    assert data.metadata["cold_unique_players"] == 2
    assert data.metadata["player_identifier_used_as_model_feature"] is False
    assert data.metadata["temporal_guardrail_available"] is False
    assert data.cold_player_mask.tolist() == [False, False, True, True]


def test_score_player_guardrails_records_all_prediction_hashes():
    target = np.array([1.0, 2.0, 4.0, 8.0])
    prediction = np.array([1.1, 1.9, 4.2, 7.7])
    cold = np.array([False, True, False, True])

    scores = guardrails.score_player_guardrails(target, prediction, cold)

    assert scores["overlap_exposed_team_holdout"]["r2"] == pytest.approx(
        r2_score(target, prediction)
    )
    assert scores["cold_player_subset"]["rows"] == 2
    assert scores["seen_player_subset"]["rows"] == 2
    assert scores["overlap_exposed_team_holdout"]["prediction_sha256"] == (
        guardrails.prediction_sha256(prediction)
    )
    assert scores["cold_player_subset"]["prediction_sha256"] == (
        guardrails.prediction_sha256(prediction[cold])
    )


@pytest.mark.parametrize(
    "target,prediction,mask",
    [
        ([1.0], [1.0, 2.0], [True]),
        ([1.0, 2.0], [1.0, 2.0], [True, True]),
        ([1.0, np.nan], [1.0, 2.0], [True, False]),
    ],
)
def test_score_player_guardrails_rejects_invalid_inputs(
    target, prediction, mask
):
    with pytest.raises(ValueError):
        guardrails.score_player_guardrails(target, prediction, mask)


def _source_artifact(data, predictions_by_config):
    results = []
    for config, prediction in predictions_by_config.items():
        prediction = np.asarray(prediction, dtype=np.float64)
        results.append(
            {
                "config": config,
                "held_team": {
                    "predictions": prediction.tolist(),
                    "prediction_sha256": guardrails.prediction_sha256(prediction),
                    "r2": float(r2_score(data.y_holdout, prediction)),
                },
            }
        )
    return {
        "protocol": {"name": "darkofit_basketball_frozen_five_arm_ablation"},
        "decision": {"recommendation": "advance_none"},
        "sources": {"darkofit": {"head": "abc123"}},
        "results": results,
    }


def test_build_analysis_reuses_predictions_without_refitting(monkeypatch):
    frame = _frame()
    _patch_creator(monkeypatch, frame)
    data = guardrails.prepare_player_guardrail(frame)
    artifact = _source_artifact(
        data,
        {
            "default": [10.0, 10.0, 10.0, 10.0],
            "candidate": [10.1, 10.0, 10.0, 10.1],
        },
    )

    result = analysis.build_analysis(
        artifact,
        frame,
        source_artifact_sha256="source-hash",
    )

    assert result["scope"]["models_refit"] is False
    assert result["scope"]["creator_benchmark_changed"] is False
    assert result["guardrail"]["team_holdout_interpretation"] == (
        "player_overlap_exposed"
    )
    assert result["results"][0]["cold_player_subset"]["rows"] == 2
    assert result["decision"]["recommendation"] == "advance_none"


def test_build_analysis_rejects_prediction_hash_drift(monkeypatch):
    frame = _frame()
    _patch_creator(monkeypatch, frame)
    data = guardrails.prepare_player_guardrail(frame)
    artifact = _source_artifact(data, {"default": [10.0, 10.0, 10.0, 10.0]})
    artifact["results"][0]["held_team"]["prediction_sha256"] = "bad"

    with pytest.raises(RuntimeError, match="hash mismatch"):
        analysis.build_analysis(
            artifact,
            frame,
            source_artifact_sha256="source-hash",
        )
