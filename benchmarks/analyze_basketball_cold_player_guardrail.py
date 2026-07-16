#!/usr/bin/env python3
"""Add a cold-player supplement to the frozen basketball ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import basketball_guardrails as guardrails
from benchmarks import run_basketball_creator_benchmark as creator


DEFAULT_SOURCE = REPO_ROOT / "benchmarks" / "basketball_darkofit_ablation.json"
DEFAULT_OUTPUT = (
    REPO_ROOT / "benchmarks" / "basketball_darkofit_cold_player_guardrail.json"
)
SOURCE_ARTIFACT_SHA256 = (
    "eb7cd737331e270714228e1bf2a7cf61db755b0bd22689f7fa4a3b7bdd881b6f"
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_analysis(
    artifact: dict[str, Any],
    frame,
    *,
    source_artifact_sha256: str,
) -> dict[str, Any]:
    protocol = artifact.get("protocol", {})
    if protocol.get("name") != "darkofit_basketball_frozen_five_arm_ablation":
        raise RuntimeError("unexpected basketball ablation protocol")
    if artifact.get("decision", {}).get("recommendation") != "advance_none":
        raise RuntimeError("source basketball decision changed")
    data = guardrails.prepare_player_guardrail(frame)
    results = []
    for source_result in artifact.get("results", []):
        config = str(source_result["config"])
        held = source_result["held_team"]
        prediction = np.asarray(held["predictions"], dtype=np.float64)
        if len(prediction) != len(data.y_holdout):
            raise RuntimeError(f"{config} held prediction length changed")
        digest = guardrails.prediction_sha256(prediction)
        if digest != held["prediction_sha256"]:
            raise RuntimeError(f"{config} held prediction hash mismatch")
        scores = guardrails.score_player_guardrails(
            data.y_holdout, prediction, data.cold_player_mask
        )
        recomputed = scores["overlap_exposed_team_holdout"]["r2"]
        if not math.isclose(recomputed, float(held["r2"]), rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(f"{config} held score does not reproduce")
        results.append({"config": config, **scores})
    if not results:
        raise RuntimeError("source basketball artifact has no results")
    default = next(row for row in results if row["config"] == "default")
    default_cold = default["cold_player_subset"]["r2"]
    default_team = default["overlap_exposed_team_holdout"]["r2"]
    comparisons = []
    for row in results:
        if row["config"] == "default":
            continue
        comparisons.append(
            {
                "config": row["config"],
                "cold_player_r2_delta_vs_default": float(
                    row["cold_player_subset"]["r2"] - default_cold
                ),
                "overlap_exposed_team_r2_delta_vs_default": float(
                    row["overlap_exposed_team_holdout"]["r2"] - default_team
                ),
            }
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "diagnostic_only": True,
            "creator_benchmark_changed": False,
            "models_refit": False,
            "source_predictions_reused": True,
            "source_artifact_sha256": source_artifact_sha256,
            "source_artifact_expected_sha256": SOURCE_ARTIFACT_SHA256,
            "source_artifact_head": artifact["sources"]["darkofit"]["head"],
        },
        "guardrail": data.metadata,
        "results": results,
        "comparisons": comparisons,
        "decision": {
            "previous_recommendation": "advance_none",
            "recommendation_changed": False,
            "recommendation": "advance_none",
            "reason": "cold-player supplement is diagnostic and no source model was refit",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--data-cache",
        type=Path,
        default=creator.DEFAULT_CACHE / "basketball_reference_toy_data.csv",
    )
    parser.add_argument("--allow-source-drift", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = creator._absolute_lexical_path(args.source)
    output = creator._absolute_lexical_path(args.output)
    data_cache = creator._absolute_lexical_path(args.data_cache)
    if output.is_symlink():
        raise RuntimeError(f"refusing symlink analysis output: {output}")
    digest = _file_sha256(source)
    if not args.allow_source_drift and digest != SOURCE_ARTIFACT_SHA256:
        raise RuntimeError(
            "basketball ablation artifact drifted: "
            f"expected {SOURCE_ARTIFACT_SHA256}, found {digest}"
        )
    artifact = json.loads(source.read_text(encoding="utf-8"))
    frame, _ = creator.load_raw_data(data_cache)
    analysis = build_analysis(
        artifact,
        frame,
        source_artifact_sha256=digest,
    )
    creator._atomic_write_bytes(
        output,
        (
            json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
