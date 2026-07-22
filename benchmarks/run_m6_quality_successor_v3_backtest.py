#!/usr/bin/env python3
"""Execute the one-shot Phase-F-corrected M6 v3 historical backtest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    import m6_quality_rule_v3 as rule
    import run_m6_quality_successor_v3 as execution
    from paired_evidence_contract import write_create_only
except ImportError:  # pragma: no cover
    from benchmarks import m6_quality_rule_v3 as rule
    from benchmarks import run_m6_quality_successor_v3 as execution
    from benchmarks.paired_evidence_contract import write_create_only


RUNNER_PATH = Path(__file__).resolve()
REPO_ROOT = RUNNER_PATH.parents[1]
POSITIVE_PATH = (
    REPO_ROOT / "benchmarks/m3b_ensemble_v3_r3_vs_single_readout_20260721.json"
)
NEGATIVE_PATH = REPO_ROOT / "benchmarks/native_ordinal_c2_development_result.json"
RETIRED_SELECTOR_PATH = REPO_ROOT / "benchmarks/fresh_selector_confirmation.json"
POSITIVE_SHA256 = "99d693063c46a0708eb45a704af0b46611fa8ed89dbe4d6469b47c7cd4a27c64"
NEGATIVE_SHA256 = "7aeb83131bb7604a3eaabc2789f048d40dabb58791b6ab6aad0ac26f0f0f566f"
RETIRED_SELECTOR_SHA256 = (
    "4dc158ec4fd11cf29a5822dc2a09aa76715ce9446773673fa9a2828da1b71a7d"
)


def replay_positive(payload: dict[str, Any]) -> dict[str, Any]:
    arm = payload["arms_vs_single"]["b1_b2_combined"]
    ratios = arm["per_case_primary_ratio"]
    if arm.get("case_count") != 13 or len(ratios) != 13:
        raise RuntimeError("M6 v3 positive artifact schema drifted")
    decision = rule.quality_decision(ratios)
    return {
        "mechanism_id": "combined_b1_b2_ensemble_v3",
        "audit_role": "surviving_known_advance",
        "expected_disposition": "advance",
        "observed_disposition": decision["disposition"],
        "agreement": decision["disposition"] == "advance",
        "decision": decision,
    }


def replay_negative(payload: dict[str, Any]) -> dict[str, Any]:
    aggregate = payload["aggregate"]
    rows = aggregate["task_rows"]
    ratios = {str(row["task_id"]): row["test_rmse_ratio"] for row in rows}
    if (
        payload.get("decision") != "close_native_ordinal_c2_development"
        or len(rows) != 4
        or len(ratios) != 4
        or aggregate.get("worst_task_ratio", 0.0) <= rule.MAX_GROUP_RATIO
    ):
        raise RuntimeError("M6 v3 negative artifact schema drifted")
    decision = rule.quality_decision(ratios)
    return {
        "mechanism_id": "native_ordinal_c2",
        "audit_role": "surviving_known_kill",
        "expected_disposition": "kill",
        "observed_disposition": decision["disposition"],
        "agreement": decision["disposition"] == "kill",
        "decision": decision,
    }


def replay_retired_selector(payload: dict[str, Any]) -> dict[str, Any]:
    analysis = payload["analysis"]
    contrast = analysis["contrasts"]["smooth_process_selector_over_default"]
    ratios = {
        name: record["ratio"]
        for name, record in contrast["per_lineage"].items()
    }
    if (
        analysis.get("recommendation") != "close_fresh_smooth_margin_selector"
        or contrast.get("lineage_count") != 14
        or contrast.get("lineage_losses") != 0
        or len(ratios) != 14
    ):
        raise RuntimeError("M6 v3 retired-selector artifact schema drifted")
    decision = rule.quality_decision(ratios)
    return {
        "mechanism_id": "linear_leaf_selector_3pct",
        "audit_role": "abolished_verdict_tripwire_not_new_evidence",
        "expected_disposition": "advance",
        "observed_disposition": decision["disposition"],
        "agreement": decision["disposition"] == "advance",
        "decision": decision,
    }


def build_result() -> dict[str, Any]:
    harness = execution.source_state(REPO_ROOT)
    if not harness["clean"]:
        raise RuntimeError("M6 v3 backtest requires a clean harness")
    if execution.BACKTEST_RESULT_PATH.exists():
        raise RuntimeError("M6 v3 backtest result already exists")
    expected_artifacts = {
        POSITIVE_PATH: POSITIVE_SHA256,
        NEGATIVE_PATH: NEGATIVE_SHA256,
        RETIRED_SELECTOR_PATH: RETIRED_SELECTOR_SHA256,
    }
    if any(
        execution.file_sha256(path) != expected
        for path, expected in expected_artifacts.items()
    ):
        raise RuntimeError("M6 v3 backtest artifact hash drifted")

    audit = json.loads(execution.AUDIT_PATH.read_text())
    if audit.get("m6_v3_backtest_selection") != {
        "known_advance": "b1_b2_combined_archive_gate",
        "known_kill": "c2_native_ordinal",
        "retired_verdict_tripwire": "smooth_linear_leaf_selector_3pct",
    }:
        raise RuntimeError("M6 v3 Phase F selection drifted")

    replays = [
        replay_positive(json.loads(POSITIVE_PATH.read_text())),
        replay_negative(json.loads(NEGATIVE_PATH.read_text())),
        replay_retired_selector(json.loads(RETIRED_SELECTOR_PATH.read_text())),
    ]
    complete = all(replay["agreement"] for replay in replays)
    bindings = {
        "audit": execution.AUDIT_PATH,
        "supersession": execution.SUPERSESSION_PATH,
        "contract": execution.CONTRACT_PATH,
        "rule": execution.RULE_PATH,
        "execution_runner": execution.RUNNER_PATH,
        "backtest_runner": RUNNER_PATH,
        "comparison_runner": execution.COMPARISON_PATH,
        "paired_execution": execution.EXECUTION_PATH,
        "positive_artifact": POSITIVE_PATH,
        "negative_artifact": NEGATIVE_PATH,
        "retired_selector_artifact": RETIRED_SELECTOR_PATH,
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
        "supersedes_forward_ranking_authority_of": "m6-quality-successor-v2",
        "frozen_v1_v2_artifacts_edited": False,
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
            "enable_phase_f_corrected_quality_development_ranking"
            if complete
            else "keep_m6_nonranking_and_close_v3"
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.name != execution.BACKTEST_RESULT_PATH.name:
        raise ValueError("M6 v3 backtest must use its canonical result name")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("M6 v3 backtest result is create-only")
    result = build_result()
    write_create_only(
        args.output,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    return 0 if result["backtest_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
