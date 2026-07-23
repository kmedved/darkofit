#!/usr/bin/env python3
"""Freeze the prospective T7b automatic-depth spent-sports contract."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from . import paired_evidence_contract as paired
    from . import run_t7b_automatic_depth_sports_v1 as runner
except ImportError:  # direct script execution
    import paired_evidence_contract as paired
    import run_t7b_automatic_depth_sports_v1 as runner


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_M3B_CONTRACT = runner.HISTORICAL_M3B_CONTRACT_PATH


def _bound(relative: str) -> dict[str, object]:
    path = ROOT / relative
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"spent-sports bound file is invalid: {relative}")
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": runner.file_sha256(path),
    }


def build_contract(
    *, control: Path, candidate: Path, panel_cache: Path
) -> dict[str, object]:
    harness = runner.source_state(ROOT)
    if not harness["clean"]:
        raise RuntimeError("spent-sports freeze requires a clean harness")
    control_state = runner.validate_source(
        control, {"head": runner.CONTROL_HEAD, "tree": runner.CONTROL_TREE}
    )
    candidate_state = runner.validate_source(
        candidate, {"head": runner.CANDIDATE_HEAD, "tree": runner.CANDIDATE_TREE}
    )
    panel = runner.panel_record(panel_cache)
    historical = json.loads(HISTORICAL_M3B_CONTRACT.read_text(encoding="utf-8"))
    if panel != {
        "bytes": historical["panel_cache"]["bytes"],
        "sha256": historical["panel_cache"]["sha256"],
    }:
        raise RuntimeError("spent-sports panel differs from frozen M3b r3")
    manifests = runner.case_manifests(panel_cache)
    historical_manifests = {
        case_id: value
        for case_id, value in historical["case_manifests"].items()
        if case_id.startswith("sports_")
    }
    if manifests != historical_manifests:
        raise RuntimeError("spent-sports case manifests differ from M3b r3")
    general = runner.validate_general_preconditions()
    return {
        "schema_version": 1,
        "contract_id": runner.CONTRACT_ID,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_frozen": True,
        "outcomes_opened": False,
        "sources": {
            runner.CONTROL: {
                "head": control_state["head"],
                "tree": control_state["tree"],
            },
            runner.CANDIDATE: {
                "head": candidate_state["head"],
                "tree": candidate_state["tree"],
            },
            "harness": harness["head"],
        },
        "bound_files": {
            name: _bound(relative)
            for name, relative in runner.BOUND_PATHS.items()
        },
        "historical_m3b_contract": {
            "path": str(HISTORICAL_M3B_CONTRACT.relative_to(ROOT)),
            "sha256": runner.file_sha256(HISTORICAL_M3B_CONTRACT),
        },
        "panel_cache": panel,
        "cases": list(runner.case_specs()),
        "case_manifests": manifests,
        "quality_orders": runner.quality_orders(),
        "execution": runner.execution_spec(),
        "decision_rules": runner.decision_rules(),
        "claims": runner.claim_spec(),
        "general_preconditions": general,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--panel-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=runner.CONTRACT_PATH)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing existing spent-sports contract: {output}")
    before = runner.source_state(ROOT)
    contract = build_contract(
        control=args.control.expanduser().resolve(),
        candidate=args.candidate.expanduser().resolve(),
        panel_cache=args.panel_cache.expanduser().resolve(),
    )
    if runner.source_state(ROOT) != before:
        raise RuntimeError("spent-sports harness changed during freeze")
    payload = (
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    paired.write_create_only(output, payload)
    print(
        json.dumps(
            {
                "contract": str(output),
                "sha256": runner.file_sha256(output),
                "harness": before["head"],
                "outcomes_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
