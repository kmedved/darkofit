#!/usr/bin/env python3
"""Build and attest the preregistered T10 basketball sports panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from benchmarks import build_basketball_sports_panel as base


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = base.DEFAULT_SOURCE
DEFAULT_CACHE = REPO_ROOT / ".cache" / "basketball-sports-panel-v2" / "panel.csv"
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "basketball_sports_panel_v2_manifest.json"
PROTOCOL_PATH = REPO_ROOT / "benchmarks" / "basketball_sports_panel_v2_protocol.md"
POWER_SOURCE = REPO_ROOT / "benchmarks" / "basketball_oob_ensemble_confirmation.json"

SOURCE_BYTES = base.SOURCE_BYTES
SOURCE_SHA256 = base.SOURCE_SHA256
SEASONS = (2014, 2015, 2016)
MIN_TOTAL_MINUTES = base.MIN_TOTAL_MINUTES
N_SPLITS = 10
RANDOM_STATE = 4
HELD_TEAM_START = 10
HELD_TEAM_STOP = 20

IDENTITY_COLUMNS = base.IDENTITY_COLUMNS
FEATURE_COLUMNS = base.FEATURE_COLUMNS
TARGET_COLUMNS = base.TARGET_COLUMNS
RAW_COLUMNS = base.RAW_COLUMNS

AGGREGATE_BAR = 1.0
BOOTSTRAP_UPPER_BAR = 1.002
LEAVE_ONE_OUT_BAR = 1.003
WORST_LINEAGE_BAR = 1.02
POWER_SEED = 20_260_718
POWER_SIMULATIONS = 20_000
POWER_BOOTSTRAPS = 2_000
POWER_CELLS = len(SEASONS) * len(TARGET_COLUMNS)
POWER_FOLDS = 10
POWER_BETWEEN_CELL_LOG_SD = 0.005
MIN_POWER = 0.80
POWER_SOURCE_SHA256 = "b7c4619c80ff3f25521b23b5f2df086a55c68deba31edc695f14b2d5bf55c824"


def _sha256(path: Path) -> str:
    return base._sha256(path)


def _json_sha256(value: Any) -> str:
    return base._json_sha256(value)


def load_raw_source(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    return base.load_raw_source(path)


def prepare_panel(frame: pd.DataFrame) -> pd.DataFrame:
    return base.prepare_panel(frame, seasons=SEASONS)


def panel_csv_bytes(panel: pd.DataFrame) -> bytes:
    return base.panel_csv_bytes(panel)


def held_teams(panel: pd.DataFrame, season: int) -> tuple[str, ...]:
    seasonal = panel.loc[panel["year"] == int(season)]
    teams = tuple(sorted(seasonal["Tm"].astype(str).unique().tolist()))
    if len(teams) != 30:
        raise RuntimeError(f"basketball season {season} does not contain 30 teams")
    return teams[HELD_TEAM_START:HELD_TEAM_STOP]


def _fold_records(primary: pd.DataFrame) -> list[dict[str, Any]]:
    groups = primary["bref_id"].astype(str).to_numpy()
    records = []
    for fold, (train, test) in enumerate(
        GroupKFold(n_splits=N_SPLITS).split(primary, groups=groups)
    ):
        train_players = frozenset(groups[train])
        test_players = frozenset(groups[test])
        if not train_players.isdisjoint(test_players):
            raise RuntimeError("player-disjoint fold construction leaked a player")
        records.append(
            {
                "fold": fold,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_indices": [int(value) for value in train],
                "test_indices": [int(value) for value in test],
                "train_player_count": len(train_players),
                "test_player_count": len(test_players),
                "test_identities_sha256": _json_sha256(
                    primary.loc[test, list(IDENTITY_COLUMNS)].values.tolist()
                ),
            }
        )
    if sorted(value for row in records for value in row["test_indices"]) != list(
        range(len(primary))
    ):
        raise RuntimeError("player-disjoint folds do not partition primary rows")
    return records


def split_manifest(panel: pd.DataFrame) -> dict[str, Any]:
    seasons: dict[str, Any] = {}
    for season in SEASONS:
        seasonal = panel.loc[panel["year"] == season].reset_index(drop=True)
        held = frozenset(held_teams(panel, season))
        primary = seasonal.loc[~seasonal["Tm"].isin(held)].reset_index(drop=True)
        holdout = seasonal.loc[seasonal["Tm"].isin(held)].reset_index(drop=True)
        train_players = frozenset(primary["bref_id"].astype(str))
        seen = holdout["bref_id"].astype(str).isin(train_players).to_numpy()
        folds = _fold_records(primary)
        seasons[str(season)] = {
            "rows": int(len(seasonal)),
            "held_teams": sorted(held),
            "primary_rows": int(len(primary)),
            "held_team_rows": int(len(holdout)),
            "seen_player_rows": int(np.sum(seen)),
            "cold_player_rows": int(np.sum(~seen)),
            "primary_identities_sha256": _json_sha256(
                primary.loc[:, list(IDENTITY_COLUMNS)].values.tolist()
            ),
            "held_identities_sha256": _json_sha256(
                holdout.loc[:, list(IDENTITY_COLUMNS)].values.tolist()
            ),
            "target_sha256": {
                target: _json_sha256(
                    primary[target].to_numpy(dtype=np.float64).tolist()
                )
                for target in TARGET_COLUMNS
            },
            "folds": folds,
            "folds_sha256": _json_sha256(folds),
        }
    return {
        "fold_strategy": "GroupKFold(bref_id, n_splits=10)",
        "held_team_slice": [HELD_TEAM_START, HELD_TEAM_STOP],
        "seasons": seasons,
        "split_manifest_sha256": _json_sha256(seasons),
    }


def _source_log_ratios() -> tuple[np.ndarray, str]:
    source_sha256 = _sha256(POWER_SOURCE)
    if source_sha256 != POWER_SOURCE_SHA256:
        raise RuntimeError("OOB-ensemble power source changed")
    artifact = json.loads(POWER_SOURCE.read_text(encoding="utf-8"))
    rows = {row["config"]: row for row in artifact["canonical_results"]}
    control = np.asarray(rows["default"]["fold_scores"], dtype=np.float64)
    candidate = np.asarray(rows["oob_ensemble5"]["fold_scores"], dtype=np.float64)
    if control.shape != (POWER_FOLDS,) or candidate.shape != (POWER_FOLDS,):
        raise RuntimeError("OOB-ensemble power source no longer has ten folds")
    if np.any(control >= 1.0) or np.any(candidate >= 1.0):
        raise RuntimeError("OOB-ensemble power source has invalid fold R2")
    ratios = np.sqrt((1.0 - candidate) / (1.0 - control))
    return np.log(ratios), source_sha256


def power_analysis() -> dict[str, Any]:
    source, source_sha256 = _source_log_ratios()
    rng = np.random.default_rng(POWER_SEED)
    true_effects = rng.normal(
        float(np.mean(source)),
        POWER_BETWEEN_CELL_LOG_SD,
        size=(POWER_SIMULATIONS, POWER_CELLS),
    )
    observed = true_effects + rng.normal(
        0.0,
        float(np.std(source, ddof=1)) / math.sqrt(POWER_FOLDS),
        size=(POWER_SIMULATIONS, POWER_CELLS),
    )
    bootstrap_indices = rng.integers(
        0,
        POWER_CELLS,
        size=(POWER_BOOTSTRAPS, POWER_CELLS),
    )
    bootstrap_weights = np.stack(
        [np.sum(bootstrap_indices == cell, axis=1) for cell in range(POWER_CELLS)],
        axis=1,
    ) / float(POWER_CELLS)
    upper = np.empty(POWER_SIMULATIONS, dtype=np.float64)
    for start in range(0, POWER_SIMULATIONS, 250):
        stop = min(start + 250, POWER_SIMULATIONS)
        bootstrap_means = observed[start:stop] @ bootstrap_weights.T
        upper[start:stop] = np.quantile(bootstrap_means, 0.95, axis=1)
    aggregate = np.mean(observed, axis=1)
    leave_one_out = (np.sum(observed, axis=1)[:, None] - observed) / float(
        POWER_CELLS - 1
    )
    passed = (
        (aggregate <= math.log(AGGREGATE_BAR))
        & (upper <= math.log(BOOTSTRAP_UPPER_BAR))
        & (np.max(leave_one_out, axis=1) <= math.log(LEAVE_ONE_OUT_BAR))
        & (np.max(observed, axis=1) <= math.log(WORST_LINEAGE_BAR))
    )
    probability = float(np.mean(passed))
    return {
        "source_path": str(POWER_SOURCE.relative_to(REPO_ROOT)),
        "source_sha256": source_sha256,
        "source_fold_log_rmse_ratios": source.tolist(),
        "source_geometric_mean_rmse_ratio": float(np.exp(np.mean(source))),
        "source_log_ratio_sample_sd": float(np.std(source, ddof=1)),
        "between_lineage_log_sd": POWER_BETWEEN_CELL_LOG_SD,
        "cells": POWER_CELLS,
        "folds_per_cell": POWER_FOLDS,
        "seed": POWER_SEED,
        "simulations": POWER_SIMULATIONS,
        "bootstrap_resamples": POWER_BOOTSTRAPS,
        "gates": {
            "aggregate_ratio_at_most": AGGREGATE_BAR,
            "bootstrap_upper_at_most": BOOTSTRAP_UPPER_BAR,
            "leave_one_out_ratio_at_most": LEAVE_ONE_OUT_BAR,
            "worst_lineage_ratio_at_most": WORST_LINEAGE_BAR,
        },
        "passing_simulations": int(np.sum(passed)),
        "pass_probability": probability,
        "minimum_probability": MIN_POWER,
        "passes": probability >= MIN_POWER,
        "scope": "Tier-D quality gates 1-4; guardrail and cost gates excluded",
    }


def build_manifest(
    panel: pd.DataFrame,
    panel_bytes: bytes,
    raw_metadata: dict[str, Any],
) -> dict[str, Any]:
    power = power_analysis()
    if not power["passes"]:
        raise RuntimeError("basketball sports panel 2 lacks 80% simulated power")
    split = split_manifest(panel)
    return {
        "schema_version": 1,
        "name": "darkofit_basketball_sports_panel_v2",
        "raw_source": raw_metadata,
        "transformation": {
            "seasons": list(SEASONS),
            "minimum_total_minutes_exclusive": MIN_TOTAL_MINUTES,
            "identity_columns": list(IDENTITY_COLUMNS),
            "feature_columns": list(FEATURE_COLUMNS),
            "target_columns": list(TARGET_COLUMNS),
            "shared_builder_semantics": str(
                Path(base.__file__).resolve().relative_to(REPO_ROOT)
            ),
        },
        "processed_panel": {
            "rows": int(len(panel)),
            "columns": int(panel.shape[1]),
            "bytes": len(panel_bytes),
            "sha256": hashlib.sha256(panel_bytes).hexdigest(),
            "identities_sha256": _json_sha256(
                panel.loc[:, list(IDENTITY_COLUMNS)].values.tolist()
            ),
        },
        "split": split,
        "power_analysis": power,
        "protocol": {
            "path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256(PROTOCOL_PATH),
        },
        "builder": {
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "sha256": _sha256(Path(__file__).resolve()),
            "shared_builder_path": str(
                Path(base.__file__).resolve().relative_to(REPO_ROOT)
            ),
            "shared_builder_sha256": _sha256(Path(base.__file__).resolve()),
        },
        "candidate_data_scored": False,
        "comparators_scored": False,
        "panel_spent": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(os.environ.get("DARKOFIT_SPORTS_PANEL_SOURCE", DEFAULT_SOURCE)),
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame, raw_metadata = load_raw_source(args.source)
    panel = prepare_panel(frame)
    payload = panel_csv_bytes(panel)
    manifest = build_manifest(panel, payload, raw_metadata)
    base._atomic_create(args.cache, payload)
    base._atomic_create(
        args.manifest,
        (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )
    print(
        json.dumps(
            {
                "cache": str(args.cache),
                "manifest": str(args.manifest),
                "rows": len(panel),
                "panel_sha256": manifest["processed_panel"]["sha256"],
                "power": manifest["power_analysis"]["pass_probability"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
