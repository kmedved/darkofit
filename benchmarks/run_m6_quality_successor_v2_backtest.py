#!/usr/bin/env python3
"""Execute the one-shot structural M6 quality-successor v2 backtest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    import m6_quality_rule_v2 as rule
    import run_m6_quality_successor_v2 as execution
    from paired_evidence_contract import write_create_only
except ImportError:  # pragma: no cover
    from benchmarks import m6_quality_rule_v2 as rule
    from benchmarks import run_m6_quality_successor_v2 as execution
    from benchmarks.paired_evidence_contract import write_create_only


RUNNER_PATH = Path(__file__).resolve()
REPO_ROOT = RUNNER_PATH.parents[1]
POSITIVE_PATH = REPO_ROOT / "benchmarks/m3b_ensemble_v3_r3_vs_single_readout_20260721.json"
NEGATIVE_PATH = REPO_ROOT / "benchmarks/fresh_selector_confirmation.json"
POSITIVE_SHA256 = "99d693063c46a0708eb45a704af0b46611fa8ed89dbe4d6469b47c7cd4a27c64"
NEGATIVE_SHA256 = "4dc158ec4fd11cf29a5822dc2a09aa76715ce9446773673fa9a2828da1b71a7d"


def replay_positive(payload: dict[str, Any]) -> dict[str, Any]:
    arm = payload["arms_vs_single"]["b1_b2_combined"]
    ratios = arm["per_case_primary_ratio"]
    if arm.get("case_count") != 13 or len(ratios) != 13:
        raise RuntimeError("M6 v2 positive artifact schema drifted")
    decision = rule.quality_decision(ratios)
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
        or contrast.get("lineage_count") != 14
        or len(ratios) != 14
    ):
        raise RuntimeError("M6 v2 negative artifact schema drifted")
    decision = rule.quality_decision(ratios)
    return {
        "mechanism_id": "linear_leaf_selector_3pct",
        "expected_disposition": "kill",
        "observed_disposition": decision["disposition"],
        "agreement": decision["disposition"] == "kill",
        "decision": decision,
    }


def build_result() -> dict[str, Any]:
    harness = execution.source_state(REPO_ROOT)
    if not harness["clean"]:
        raise RuntimeError("M6 v2 backtest requires a clean harness")
    if execution.BACKTEST_RESULT_PATH.exists():
        raise RuntimeError("M6 v2 backtest result already exists")
    if (
        execution.file_sha256(POSITIVE_PATH) != POSITIVE_SHA256
        or execution.file_sha256(NEGATIVE_PATH) != NEGATIVE_SHA256
    ):
        raise RuntimeError("M6 v2 backtest artifact hash drifted")
    replays = [
        replay_positive(json.loads(POSITIVE_PATH.read_text())),
        replay_negative(json.loads(NEGATIVE_PATH.read_text())),
    ]
    complete = all(replay["agreement"] for replay in replays)
    bindings = {
        "contract": execution.CONTRACT_PATH,
        "rule": execution.RULE_PATH,
        "execution_runner": execution.RUNNER_PATH,
        "comparison_runner": execution.COMPARISON_PATH,
        "paired_execution": execution.EXECUTION_PATH,
        "backtest_runner": RUNNER_PATH,
        "positive_artifact": POSITIVE_PATH,
        "negative_artifact": NEGATIVE_PATH,
    }
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": rule.CONTRACT_ID,
        "evidence_status": "complete" if complete else "terminal_failure",
        "backtest_complete": complete,
        "candidate_ranking_eligible": complete,
        "shipping_or_default_claim_eligible": False,
        "rerun_authorized": False,
        "outcome_blind": False,
        "v1_structural_invalidation": (
            "benchmarks/m6_quality_successor_v1_invalidation_20260721.md"
        ),
        "harness": harness,
        "bindings": {
            name: {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": execution.file_sha256(path),
            }
            for name, path in bindings.items()
        },
        "replays": replays,
        "decision": (
            "enable_quality_development_ranking"
            if complete
            else "keep_m6_nonranking_and_close_v2"
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.name != execution.BACKTEST_RESULT_PATH.name:
        raise ValueError("M6 v2 backtest must use its canonical result name")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("M6 v2 backtest result is create-only")
    result = build_result()
    write_create_only(
        args.output,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    return 0 if result["backtest_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
