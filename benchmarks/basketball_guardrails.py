"""Player-overlap diagnostics for the frozen basketball benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from benchmarks import run_basketball_creator_benchmark as creator


PLAYER_COLUMN = "Player"


@dataclass(frozen=True)
class BasketballGuardrailData:
    """Creator team split plus an unseen-player evaluation mask."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_holdout: pd.DataFrame
    y_holdout: pd.Series
    cold_player_mask: np.ndarray
    metadata: dict[str, Any]


def prediction_sha256(prediction: Any) -> str:
    values = np.ascontiguousarray(np.asarray(prediction, dtype="<f8"))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _mask_sha256(mask: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _strings_sha256(values: pd.Series) -> str:
    payload = json.dumps(
        values.astype(str).tolist(),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_player_guardrail(frame: pd.DataFrame) -> BasketballGuardrailData:
    """Rebuild the creator team holdout and identify unseen-player rows.

    The creator's feature matrix does not include ``Player``. Repeated players
    therefore are not direct feature leakage, but they make the alphabetical
    team holdout dependent at the athlete level. The cold-player mask selects
    only holdout rows whose exact source ``Player`` value is absent from the
    training rows.
    """
    if PLAYER_COLUMN not in frame.columns:
        raise ValueError(f"basketball data is missing {PLAYER_COLUMN!r}")
    filtered = frame.loc[frame["MP"] > 500].copy()
    if filtered[PLAYER_COLUMN].isna().any():
        raise ValueError("basketball Player values must be non-missing")
    filtered["MPG"] = filtered["MP"] / filtered["G"]
    filtered["starter"] = np.where(
        filtered["GS"] / filtered["G"] >= 0.5, 1, 0
    )

    teams = filtered["Tm"].sort_values().drop_duplicates().tolist()
    test_teams = teams[: len(teams) // 3]
    test_team_set = set(test_teams)
    train = filtered.loc[~filtered["Tm"].isin(test_team_set)]
    holdout = filtered.loc[filtered["Tm"].isin(test_team_set)]

    X_train = train.loc[:, creator.FEATURES]
    y_train = train.loc[:, "MPG"]
    X_holdout = holdout.loc[:, creator.FEATURES]
    y_holdout = holdout.loc[:, "MPG"]
    creator_X, creator_y, creator_metadata = creator.prepare_creator_data(frame)
    if not X_train.equals(creator_X) or not y_train.equals(creator_y):
        raise RuntimeError("player guardrail changed the creator training rows")
    if int(len(holdout)) != int(creator_metadata["test_rows"]):
        raise RuntimeError("player guardrail changed the creator team holdout")
    if PLAYER_COLUMN in creator.FEATURES:
        raise RuntimeError("Player unexpectedly became a creator model feature")

    train_players = train[PLAYER_COLUMN].astype(str)
    holdout_players = holdout[PLAYER_COLUMN].astype(str)
    train_player_set = set(train_players)
    holdout_player_set = set(holdout_players)
    overlap_players = train_player_set & holdout_player_set
    cold_mask = ~holdout_players.isin(train_player_set).to_numpy(dtype=bool)
    seen_mask = ~cold_mask
    cold_players = set(holdout_players.iloc[np.flatnonzero(cold_mask)])
    if cold_players & train_player_set:
        raise RuntimeError("cold-player mask contains a training player")
    if not cold_mask.any() or not seen_mask.any():
        raise RuntimeError("player guardrail requires both cold and seen rows")

    metadata = {
        "name": "alphabetical_team_holdout_with_cold_player_subset",
        "creator_team_holdout_unchanged": True,
        "team_holdout_interpretation": "player_overlap_exposed",
        "player_identifier_column": PLAYER_COLUMN,
        "player_identifier_policy": "exact_source_string",
        "player_identifier_used_as_model_feature": False,
        "temporal_guardrail_available": False,
        "temporal_guardrail_unavailable_reason": "source_has_no_season_or_date_column",
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "train_unique_players": int(len(train_player_set)),
        "holdout_unique_players": int(len(holdout_player_set)),
        "overlapping_unique_players": int(len(overlap_players)),
        "overlapping_unique_player_rate": float(
            len(overlap_players) / len(holdout_player_set)
        ),
        "seen_player_holdout_rows": int(seen_mask.sum()),
        "seen_player_holdout_row_rate": float(seen_mask.mean()),
        "cold_player_holdout_rows": int(cold_mask.sum()),
        "cold_player_holdout_row_rate": float(cold_mask.mean()),
        "cold_unique_players": int(len(cold_players)),
        "cold_players_absent_from_training": True,
        "cold_player_mask_sha256": _mask_sha256(cold_mask),
        "train_player_sequence_sha256": _strings_sha256(train_players),
        "holdout_player_sequence_sha256": _strings_sha256(holdout_players),
        "test_teams": list(test_teams),
    }
    return BasketballGuardrailData(
        X_train=X_train,
        y_train=y_train,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        cold_player_mask=cold_mask,
        metadata=metadata,
    )


def score_player_guardrails(
    y_holdout: Any,
    prediction: Any,
    cold_player_mask: Any,
) -> dict[str, Any]:
    """Score the unchanged team holdout and its cold/seen-player subsets."""
    target = np.asarray(y_holdout, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    cold = np.asarray(cold_player_mask, dtype=bool)
    if target.ndim != 1 or predicted.ndim != 1 or cold.ndim != 1:
        raise ValueError("basketball guardrail inputs must be one-dimensional")
    if not (len(target) == len(predicted) == len(cold)):
        raise ValueError("basketball guardrail inputs have different lengths")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(predicted)):
        raise ValueError("basketball guardrail values must be finite")
    cold_rows = int(cold.sum())
    seen_rows = int((~cold).sum())
    if cold_rows < 2 or seen_rows < 2:
        raise ValueError(
            "cold-player and seen-player subsets each require at least two rows"
        )
    seen = ~cold
    return {
        "overlap_exposed_team_holdout": {
            "rows": int(len(target)),
            "r2": float(r2_score(target, predicted)),
            "prediction_sha256": prediction_sha256(predicted),
        },
        "cold_player_subset": {
            "rows": int(cold.sum()),
            "r2": float(r2_score(target[cold], predicted[cold])),
            "prediction_sha256": prediction_sha256(predicted[cold]),
        },
        "seen_player_subset": {
            "rows": int(seen.sum()),
            "r2": float(r2_score(target[seen], predicted[seen])),
            "prediction_sha256": prediction_sha256(predicted[seen]),
        },
    }
