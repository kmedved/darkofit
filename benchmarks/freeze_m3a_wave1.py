#!/usr/bin/env python3
"""Create the pre-outcome Wave-1 M3a contract manifest."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "m3a_wave1_contract.json"
RUNNER_PATH = ROOT / "benchmarks" / "run_m3a_wave1.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "m3a_runner_for_freeze", RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M3a runner for contract freeze")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bound(path: str) -> dict[str, Any]:
    target = ROOT / path
    return {
        "path": path,
        "bytes": target.stat().st_size,
        "sha256": _sha256(target),
    }


def build_contract() -> dict[str, Any]:
    runner = _load_runner()
    sports_manifest = json.loads(
        (
            ROOT / "benchmarks" / "basketball_sports_panel_v2_manifest.json"
        ).read_text(encoding="utf-8")
    )
    seasons = sports_manifest["transformation"]["seasons"]
    creator_folds = {}
    for season in seasons:
        primary_rows = sports_manifest["split"]["seasons"][str(season)][
            "primary_rows"
        ]
        records = runner.creator_fold_records(primary_rows, int(season))
        creator_folds[str(season)] = {
            "rows": int(primary_rows),
            "folds": len(records),
            "sha256": _json_sha256(records),
        }
    return {
        "schema_version": 1,
        "name": "wave1_m3a_20260720",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "contract_frozen": True,
        "outcomes_opened": False,
        "sources": {
            "darkofit": runner.DARKO_SOURCE_HEAD,
            "chimeraboost": runner.CHIMERA_SOURCE_HEAD,
        },
        "threads": runner.THREADS,
        "bound_files": {
            "protocol": _bound("benchmarks/m3a_wave1_protocol.md"),
            "runner": _bound("benchmarks/run_m3a_wave1.py"),
            "analyzer": _bound("benchmarks/analyze_m3a_wave1.py"),
            "freezer": _bound("benchmarks/freeze_m3a_wave1.py"),
            "sports_manifest": _bound(
                "benchmarks/basketball_sports_panel_v2_manifest.json"
            ),
            "m6_adapter": _bound("benchmarks/benchmark_adapters.py"),
            "m5_baseline": _bound("benchmarks/m5_sentinel_baseline.json"),
        },
        "sports_panel": {
            "processed_sha256": sports_manifest["processed_panel"]["sha256"],
            "processed_bytes": sports_manifest["processed_panel"]["bytes"],
            "seasons": seasons,
            "targets": sports_manifest["transformation"]["target_columns"],
            "player_fold_strategy": sports_manifest["split"]["fold_strategy"],
            "split_manifest_sha256": sports_manifest["split"][
                "split_manifest_sha256"
            ],
            "spent": True,
        },
        "creator_folds": {
            "strategy": "KFold(n_splits=10, shuffle=True)",
            "seed_formula": "20260720 + season",
            "seasons": creator_folds,
        },
        "arms": {
            "primary": list(runner.PRIMARY_ARMS),
            "diagnostic": list(runner.DIAGNOSTIC_ARMS),
            "disclosures": {
                runner.DARKO_SINGLE: (
                    "single; group-aware internal validation on sports rows"
                ),
                runner.DARKO_GROUP8: (
                    "group-bootstrap member/OOB selection; player-disjoint"
                ),
                runner.CHIMERA_SINGLE: (
                    "single; group-aware internal validation on sports rows"
                ),
                runner.CHIMERA_ENSEMBLE8: (
                    "0.8 row subagging; player-overlap exposed"
                ),
                runner.DARKO_ROW5: (
                    "row bootstrap; player-overlap exposed"
                ),
                runner.DARKO_ROW8: (
                    "row bootstrap; player-overlap exposed"
                ),
                runner.DARKO_GROUP5: (
                    "group bootstrap; player-disjoint"
                ),
                runner.CHIMERA_FLOAT_SINGLE: (
                    "float single; group-aware internal validation on sports rows"
                ),
                runner.CHIMERA_FLOAT_ENSEMBLE8: (
                    "float 0.8 row subagging; player-overlap exposed"
                ),
            },
        },
        "execution": {
            "quality_first": True,
            "orders": {
                "primary-quality": [
                    [
                        runner.DARKO_SINGLE,
                        runner.CHIMERA_SINGLE,
                        runner.DARKO_GROUP8,
                        runner.CHIMERA_ENSEMBLE8,
                    ]
                ],
                "diagnostics": [
                    [
                        runner.DARKO_ROW5,
                        runner.CHIMERA_FLOAT_SINGLE,
                        runner.DARKO_GROUP5,
                        runner.CHIMERA_FLOAT_ENSEMBLE8,
                        runner.DARKO_ROW8,
                    ]
                ],
                "primary-repeats": [
                    [
                        runner.CHIMERA_ENSEMBLE8,
                        runner.DARKO_GROUP8,
                        runner.CHIMERA_SINGLE,
                        runner.DARKO_SINGLE,
                    ],
                    [
                        runner.DARKO_GROUP8,
                        runner.DARKO_SINGLE,
                        runner.CHIMERA_ENSEMBLE8,
                        runner.CHIMERA_SINGLE,
                    ],
                ],
            },
            "repeat_rule": (
                "run primary-repeats only when frozen primary decision survives"
            ),
            "fresh_worker_per_arm": True,
            "same_arm_warmup": True,
        },
        "inference": {
            "unit": "season",
            "clusters": 3,
            "seed": 20_260_720,
            "resamples": 100_000,
            "descriptive_only": True,
        },
        "survival_gates": {
            "player_geomean_at_most": 0.995,
            "player_cluster_p95_at_most": 1.000,
            "held_geomean_at_most": 1.005,
            "cold_geomean_at_most": 1.005,
            "worst_season_at_most": 1.010,
            "worst_player_cell_at_most": 1.030,
            "fit_ratio_at_most": 9.0,
            "predict_ratio_at_most": 9.0,
            "model_bytes_ratio_at_most": 9.0,
            "peak_rss_ratio_at_most": 4.0,
        },
        "general": {
            "adapter": "frozen M6 benchmark_adapters",
            "datasets": [
                "friedman_numeric",
                "wide_numeric_reg",
                "categorical_reg",
            ],
            "size": "medium",
            "rows": 10_000,
            "seeds": [0, 1],
            "weight_mode": "none",
            "arms": list(runner.GENERAL_ARMS),
            "descriptive_only": True,
            "m6_ranking_authorized": False,
        },
        "claims": {
            "tier": "E",
            "default_change_authorized": False,
            "cross_season_generalization_authorized": False,
            "m5_used_as_scoreboard": False,
            "m6_ranking_authorized": False,
        },
    }


def main() -> int:
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise RuntimeError(f"refusing existing M3a contract: {OUTPUT}")
    contract = build_contract()
    payload = (
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "creator_folds": contract["creator_folds"]["seasons"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
