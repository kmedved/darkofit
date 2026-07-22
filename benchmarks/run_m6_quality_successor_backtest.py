#!/usr/bin/env python3
"""Execute the frozen artifact-only M6 quality-successor backtest."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    import m6_quality_successor as successor
    from paired_evidence_contract import write_create_only
except ImportError:  # pragma: no cover - supports module execution
    from benchmarks import m6_quality_successor as successor
    from benchmarks.paired_evidence_contract import write_create_only


RUNNER_PATH = Path(__file__).resolve()
REPO_ROOT = RUNNER_PATH.parents[1]
POSITIVE_PATH = REPO_ROOT / "benchmarks/m3b_ensemble_v3_r3_vs_single_readout_20260721.json"
NEGATIVE_PATH = REPO_ROOT / "benchmarks/fresh_selector_confirmation.json"
POSITIVE_SHA256 = "99d693063c46a0708eb45a704af0b46611fa8ed89dbe4d6469b47c7cd4a27c64"
NEGATIVE_SHA256 = "4dc158ec4fd11cf29a5822dc2a09aa76715ce9446773673fa9a2828da1b71a7d"


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _require_clean_committed_harness() -> dict[str, Any]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    if status:
        raise RuntimeError("M6 successor backtest requires a clean harness")
    return {
        "head": _git("rev-parse", "HEAD"),
        "tree": _git("rev-parse", "HEAD^{tree}"),
        "clean": True,
    }


def replay_positive(payload: dict[str, Any]) -> dict[str, Any]:
    arm = payload["arms_vs_single"]["b1_b2_combined"]
    ratios = arm["per_case_primary_ratio"]
    if (
        payload.get("finding", {}).get("combined_beats_single_all_cases") is not True
        or arm.get("case_count") != 13
        or set(ratios) != {
            "general_categorical_multiclass",
            "general_categorical_reg",
            "general_friedman_numeric",
            "general_numeric_binary",
            "sports_2014_box_plus_minus",
            "sports_2014_game_score",
            "sports_2014_minutes_per_game",
            "sports_2015_box_plus_minus",
            "sports_2015_game_score",
            "sports_2015_minutes_per_game",
            "sports_2016_box_plus_minus",
            "sports_2016_game_score",
            "sports_2016_minutes_per_game",
        }
    ):
        raise RuntimeError("M6 positive backtest artifact schema drifted")
    decision = successor.quality_decision(ratios)
    return {
        "mechanism_id": "combined_b1_b2_ensemble_v3",
        "expected_disposition": "advance",
        "observed_disposition": decision["disposition"],
        "agreement": decision["disposition"] == "advance",
        "decision": decision,
    }


def replay_negative(payload: dict[str, Any]) -> dict[str, Any]:
    analysis = payload["analysis"]
    contrast = analysis["contrasts"]["smooth_process_selector_over_default"]
    ratios = {
        name: record["ratio"]
        for name, record in contrast["per_lineage"].items()
    }
    if (
        analysis.get("recommendation") != "close_fresh_smooth_margin_selector"
        or analysis.get("passes_all_gates") is not False
        or contrast.get("lineage_count") != 14
        or len(ratios) != 14
    ):
        raise RuntimeError("M6 negative backtest artifact schema drifted")
    decision = successor.quality_decision(ratios)
    return {
        "mechanism_id": "linear_leaf_selector_3pct",
        "expected_disposition": "kill",
        "observed_disposition": decision["disposition"],
        "agreement": decision["disposition"] == "kill",
        "decision": decision,
    }


def build_result() -> dict[str, Any]:
    if successor.BACKTEST_COMPLETE:
        raise RuntimeError("M6 successor backtest is already complete")
    harness = _require_clean_committed_harness()
    hashes = {
        "positive": successor.file_sha256(POSITIVE_PATH),
        "negative": successor.file_sha256(NEGATIVE_PATH),
    }
    if hashes != {"positive": POSITIVE_SHA256, "negative": NEGATIVE_SHA256}:
        raise RuntimeError("M6 successor backtest artifact hash drifted")
    positive = replay_positive(json.loads(POSITIVE_PATH.read_text()))
    negative = replay_negative(json.loads(NEGATIVE_PATH.read_text()))
    replays = [positive, negative]
    complete = all(replay["agreement"] for replay in replays)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": successor.CONTRACT_ID,
        "evidence_status": "complete" if complete else "terminal_failure",
        "backtest_complete": complete,
        "candidate_ranking_eligible": complete,
        "shipping_or_default_claim_eligible": False,
        "rerun_authorized": False,
        "harness": harness,
        "bindings": {
            "contract": {
                "path": str(successor.CONTRACT_PATH.relative_to(REPO_ROOT)),
                "sha256": successor.file_sha256(successor.CONTRACT_PATH),
            },
            "analyzer": {
                "path": str(successor.ANALYZER_PATH.relative_to(REPO_ROOT)),
                "sha256": successor.file_sha256(successor.ANALYZER_PATH),
            },
            "runner": {
                "path": str(RUNNER_PATH.relative_to(REPO_ROOT)),
                "sha256": successor.file_sha256(RUNNER_PATH),
            },
            "positive_artifact": {
                "path": str(POSITIVE_PATH.relative_to(REPO_ROOT)),
                "sha256": hashes["positive"],
            },
            "negative_artifact": {
                "path": str(NEGATIVE_PATH.relative_to(REPO_ROOT)),
                "sha256": hashes["negative"],
            },
        },
        "replays": replays,
        "decision": (
            "enable_quality_development_ranking"
            if complete
            else "keep_m6_nonranking_and_close_contract_identity"
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"M6 backtest output is create-only: {args.output}")
    result = build_result()
    write_create_only(
        args.output,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    return 0 if result["backtest_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
