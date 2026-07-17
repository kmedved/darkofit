#!/usr/bin/env python3
"""Build and attest the preregistered S4 basketball sports panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    "/Users/kmedved/Library/CloudStorage/Dropbox/github/darko/"
    "calculated_data/temp/bbr_advanced_game_logs.csv"
)
DEFAULT_CACHE = REPO_ROOT / ".cache" / "basketball-sports-panel-v1" / "panel.csv"
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "basketball_sports_panel_manifest.json"
PROTOCOL_PATH = REPO_ROOT / "benchmarks" / "basketball_sports_panel_protocol.md"
POWER_SOURCE = REPO_ROOT / "benchmarks" / "basketball_random_strength.json"

SOURCE_BYTES = 214_366_516
SOURCE_SHA256 = "96e0efffb09e27f64cee395faa1783b025757c88efb74f5fb98cbd82c583d826"
SEASONS = (2017, 2018, 2019)
MIN_TOTAL_MINUTES = 500.0
N_SPLITS = 10
RANDOM_STATE = 4

IDENTITY_COLUMNS = ("bref_id", "Player", "year", "Tm")
FEATURE_COLUMNS = (
    "age",
    "games",
    "start_rate",
    "ts_pct",
    "efg_pct",
    "orb_pct",
    "drb_pct",
    "trb_pct",
    "ast_pct",
    "stl_pct",
    "blk_pct",
    "tov_pct",
    "usg_pct",
    "offensive_rating",
    "defensive_rating",
)
TARGET_COLUMNS = ("minutes_per_game", "game_score", "box_plus_minus")
RAW_COLUMNS = (
    "Date",
    "Age",
    "Tm",
    "GS",
    "Minutes",
    "TS.",
    "eFG.",
    "ORB.",
    "DRB.",
    "TRB.",
    "AST.",
    "STL.",
    "BLK.",
    "TOV.",
    "USG.",
    "ORtg",
    "DRtg",
    "GmSc",
    "BPM",
    "bref_id",
    "Player",
    "year",
)
RAW_WEIGHTED_TO_PANEL = {
    "Age_decimal": "age",
    "TS.": "ts_pct",
    "eFG.": "efg_pct",
    "ORB.": "orb_pct",
    "DRB.": "drb_pct",
    "TRB.": "trb_pct",
    "AST.": "ast_pct",
    "STL.": "stl_pct",
    "BLK.": "blk_pct",
    "TOV.": "tov_pct",
    "USG.": "usg_pct",
    "ORtg": "offensive_rating",
    "DRtg": "defensive_rating",
    "BPM": "box_plus_minus",
}

MIN_PRIMARY_MEAN_DELTA = 0.0005
POWER_SEED = 20_260_717
POWER_SIMULATIONS = 200_000
POWER_CELLS = len(SEASONS) * len(TARGET_COLUMNS)
POWER_BETWEEN_CELL_SD = 0.004
MIN_POWER = 0.80
EXPECTED_SCREEN_MEAN = 0.002123761281670389
EXPECTED_SCREEN_SD = 0.005482924018693017

_AGE_PATTERN = re.compile(r"^(?P<years>[0-9]{1,3})-(?P<days>[0-9]{3})$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_create(path: Path, value: bytes) -> None:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_age(value: Any) -> float:
    """Parse Basketball Reference's years-days age encoding exactly."""
    match = _AGE_PATTERN.fullmatch(str(value))
    if match is None:
        raise ValueError(f"invalid Basketball Reference age: {value!r}")
    years = int(match.group("years"))
    days = int(match.group("days"))
    if days > 365:
        raise ValueError(f"invalid Basketball Reference age day: {value!r}")
    return float(years + days / 365.25)


def _validate_raw_source(path: Path) -> dict[str, Any]:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"basketball sports source is not a regular file: {path}")
    size = path.stat().st_size
    digest = _sha256(path)
    if size != SOURCE_BYTES or digest != SOURCE_SHA256:
        raise RuntimeError(
            "basketball sports source does not match the frozen export: "
            f"bytes={size}, sha256={digest}"
        )
    return {"path": str(path), "bytes": size, "sha256": digest}


def load_raw_source(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = _validate_raw_source(path)
    frame = pd.read_csv(
        path,
        encoding="latin1",
        usecols=list(RAW_COLUMNS),
        low_memory=False,
    )
    missing = sorted(set(RAW_COLUMNS).difference(frame.columns))
    if missing:
        raise RuntimeError(f"basketball sports source lost columns: {missing}")
    metadata.update(
        {
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "encoding": "latin1",
        }
    )
    return frame, metadata


def prepare_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the canonical player-team-season panel from a raw-like frame."""
    missing = sorted(set(RAW_COLUMNS).difference(frame.columns))
    if missing:
        raise RuntimeError(f"basketball sports frame lost columns: {missing}")

    work = frame.loc[:, list(RAW_COLUMNS)].copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work["Minutes"] = pd.to_numeric(work["Minutes"], errors="coerce")
    work = work.loc[
        work["year"].isin(SEASONS)
        & np.isfinite(work["Minutes"])
        & (work["Minutes"] > 0.0)
        & work["bref_id"].notna()
        & work["Player"].notna()
        & work["Tm"].notna()
    ].copy()
    if work.empty:
        raise RuntimeError("basketball sports source has no eligible game rows")

    work["year"] = work["year"].astype(np.int64)
    for column in (
        "GS",
        "TS.",
        "eFG.",
        "ORB.",
        "DRB.",
        "TRB.",
        "AST.",
        "STL.",
        "BLK.",
        "TOV.",
        "USG.",
        "ORtg",
        "DRtg",
        "GmSc",
        "BPM",
    ):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["Age_decimal"] = work["Age"].map(parse_age)

    required_numeric = ["Minutes", "GS", "GmSc", "BPM", "Age_decimal"]
    numeric = work.loc[:, required_numeric].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        bad = {
            column: int(work[column].isna().sum())
            for column in required_numeric
            if work[column].isna().any()
        }
        raise RuntimeError(
            f"eligible basketball game rows contain missing values: {bad}"
        )

    keys = list(IDENTITY_COLUMNS)
    grouped = work.groupby(keys, sort=True, observed=True)
    base = grouped.agg(
        games=("Date", "size"),
        total_minutes=("Minutes", "sum"),
        start_rate=("GS", "mean"),
        game_score=("GmSc", "mean"),
    )
    for raw_column, panel_column in RAW_WEIGHTED_TO_PANEL.items():
        values = work[raw_column].astype(np.float64)
        available = np.isfinite(values)
        weights = work["Minutes"].astype(np.float64).where(available, 0.0)
        numerator = (
            (values.where(available, 0.0) * weights)
            .groupby([work[key] for key in keys], sort=True, observed=True)
            .sum()
        )
        denominator = weights.groupby(
            [work[key] for key in keys], sort=True, observed=True
        ).sum()
        base[panel_column] = numerator / denominator

    base["minutes_per_game"] = base["total_minutes"] / base["games"]
    panel = base.loc[base["total_minutes"] > MIN_TOTAL_MINUTES].reset_index()
    panel = panel.loc[
        :,
        [*IDENTITY_COLUMNS, *FEATURE_COLUMNS, *TARGET_COLUMNS],
    ]
    panel["_player_sort"] = panel["Player"].astype(str)
    panel = (
        panel.sort_values(["year", "_player_sort", "bref_id", "Tm"], kind="mergesort")
        .drop(columns="_player_sort")
        .reset_index(drop=True)
    )

    values = panel.loc[:, [*FEATURE_COLUMNS, *TARGET_COLUMNS]].to_numpy(
        dtype=np.float64
    )
    if not np.all(np.isfinite(values)):
        raise RuntimeError("canonical basketball panel contains non-finite values")
    if tuple(sorted(panel["year"].unique().tolist())) != SEASONS:
        raise RuntimeError("canonical basketball panel lost a frozen season")
    for season in SEASONS:
        if panel.loc[panel["year"] == season, "Tm"].nunique() != 30:
            raise RuntimeError(f"season {season} does not contain 30 teams")
    return panel


def panel_csv_bytes(panel: pd.DataFrame) -> bytes:
    return panel.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode(
        "utf-8"
    )


def held_teams(panel: pd.DataFrame) -> tuple[str, ...]:
    teams = tuple(sorted(panel["Tm"].astype(str).unique().tolist()))
    if len(teams) != 30:
        raise RuntimeError("basketball sports panel does not contain 30 teams")
    return teams[: len(teams) // 3]


def split_manifest(panel: pd.DataFrame) -> dict[str, Any]:
    held = frozenset(held_teams(panel))
    seasons: dict[str, Any] = {}
    for season in SEASONS:
        seasonal = panel.loc[panel["year"] == season].reset_index(drop=True)
        primary = seasonal.loc[~seasonal["Tm"].isin(held)].reset_index(drop=True)
        holdout = seasonal.loc[seasonal["Tm"].isin(held)].reset_index(drop=True)
        train_players = frozenset(primary["bref_id"].astype(str))
        seen_mask = holdout["bref_id"].astype(str).isin(train_players).to_numpy()
        folds = []
        for fold, (train, test) in enumerate(
            KFold(n_splits=N_SPLITS, shuffle=False).split(primary)
        ):
            folds.append(
                {
                    "fold": fold,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "test_indices": [int(value) for value in test],
                    "test_identities_sha256": _json_sha256(
                        primary.loc[test, list(IDENTITY_COLUMNS)].values.tolist()
                    ),
                }
            )
        target_hashes = {
            target: _json_sha256(primary[target].to_numpy(dtype=np.float64).tolist())
            for target in TARGET_COLUMNS
        }
        seasons[str(season)] = {
            "rows": int(len(seasonal)),
            "primary_rows": int(len(primary)),
            "held_team_rows": int(len(holdout)),
            "seen_player_rows": int(np.sum(seen_mask)),
            "cold_player_rows": int(np.sum(~seen_mask)),
            "primary_identities_sha256": _json_sha256(
                primary.loc[:, list(IDENTITY_COLUMNS)].values.tolist()
            ),
            "held_identities_sha256": _json_sha256(
                holdout.loc[:, list(IDENTITY_COLUMNS)].values.tolist()
            ),
            "target_sha256": target_hashes,
            "folds": folds,
            "folds_sha256": _json_sha256(folds),
        }
    return {
        "held_teams": list(sorted(held)),
        "seasons": seasons,
        "split_manifest_sha256": _json_sha256(seasons),
    }


def _screen_fold_deltas() -> tuple[np.ndarray, str]:
    source_sha256 = _sha256(POWER_SOURCE)
    artifact = json.loads(POWER_SOURCE.read_text(encoding="utf-8"))
    runs = {row["config"]: row for row in artifact["results"]}
    control = np.asarray(runs["control"]["fold_scores"], dtype=np.float64)
    candidate = np.asarray(runs["random_strength_0_5"]["fold_scores"], dtype=np.float64)
    if control.shape != (10,) or candidate.shape != (10,):
        raise RuntimeError("random-strength power source no longer has ten folds")
    deltas = candidate - control
    if not math.isclose(
        float(np.mean(deltas)), EXPECTED_SCREEN_MEAN, rel_tol=0.0, abs_tol=1e-15
    ) or not math.isclose(
        float(np.std(deltas, ddof=1)),
        EXPECTED_SCREEN_SD,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("random-strength power-source deltas changed")
    return deltas, source_sha256


def power_analysis() -> dict[str, Any]:
    deltas, source_sha256 = _screen_fold_deltas()
    within_sd = float(np.std(deltas, ddof=1))
    effect_mean = float(np.mean(deltas))
    rng = np.random.default_rng(POWER_SEED)
    true_effects = rng.normal(
        loc=effect_mean,
        scale=POWER_BETWEEN_CELL_SD,
        size=(POWER_SIMULATIONS, POWER_CELLS),
    )
    observed = rng.normal(
        loc=true_effects[:, :, None],
        scale=within_sd,
        size=(POWER_SIMULATIONS, POWER_CELLS, N_SPLITS),
    )
    cells = np.mean(observed, axis=2)
    primary = np.mean(cells, axis=1)
    leave_one_out = (np.sum(cells, axis=1)[:, None] - cells) / float(POWER_CELLS - 1)
    passed = (primary >= MIN_PRIMARY_MEAN_DELTA) & np.all(leave_one_out >= 0.0, axis=1)
    passing = int(np.sum(passed))
    probability = float(passing / POWER_SIMULATIONS)
    return {
        "source_path": str(POWER_SOURCE.relative_to(REPO_ROOT)),
        "source_sha256": source_sha256,
        "source_fold_deltas": deltas.tolist(),
        "source_mean": effect_mean,
        "source_sample_sd": within_sd,
        "between_cell_sd": POWER_BETWEEN_CELL_SD,
        "cells": POWER_CELLS,
        "folds_per_cell": N_SPLITS,
        "seed": POWER_SEED,
        "simulations": POWER_SIMULATIONS,
        "minimum_equal_cell_mean_delta": MIN_PRIMARY_MEAN_DELTA,
        "minimum_probability": MIN_POWER,
        "passing_simulations": passing,
        "pass_probability": probability,
        "passes": probability >= MIN_POWER,
        "scope": "primary_mean_and_leave_one_cell_out_gates_only",
    }


def build_manifest(
    panel: pd.DataFrame,
    panel_bytes: bytes,
    raw_metadata: dict[str, Any],
) -> dict[str, Any]:
    split = split_manifest(panel)
    power = power_analysis()
    if not power["passes"]:
        raise RuntimeError("basketball sports design lacks 80% simulated power")
    panel_sha256 = hashlib.sha256(panel_bytes).hexdigest()
    return {
        "schema_version": 1,
        "name": "darkofit_basketball_sports_panel_v1",
        "raw_source": raw_metadata,
        "transformation": {
            "seasons": list(SEASONS),
            "minimum_total_minutes_exclusive": MIN_TOTAL_MINUTES,
            "identity_columns": list(IDENTITY_COLUMNS),
            "feature_columns": list(FEATURE_COLUMNS),
            "target_columns": list(TARGET_COLUMNS),
            "age_formula": "years + days / 365.25",
            "weighted_by_minutes": list(RAW_WEIGHTED_TO_PANEL.values()),
            "game_score_aggregation": "arithmetic_mean",
            "canonical_order": ["year", "Player", "bref_id", "Tm"],
        },
        "processed_panel": {
            "rows": int(len(panel)),
            "columns": int(panel.shape[1]),
            "bytes": len(panel_bytes),
            "sha256": panel_sha256,
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
    _atomic_create(args.cache, payload)
    _atomic_create(
        args.manifest,
        (
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
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
