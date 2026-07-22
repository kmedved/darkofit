#!/usr/bin/env python3
"""Create the warmup-corrected v0.11 M2 successor contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:  # Direct execution from a clean checkout.
    _REPO_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from benchmarks import run_v011_m2_broad_panel_v3 as campaign


def freeze(output: Path) -> dict:
    output = Path(os.path.abspath(output.expanduser()))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"v0.11 M2 v3 contract is create-only: {output}")
    status = campaign._v1._base.hardened._repository_status(campaign.ROOT, None)
    if status:
        raise RuntimeError("freeze from a clean committed M2 v3 harness checkout")
    head = campaign._v1._base._run_git(["rev-parse", "HEAD"], cwd=campaign.ROOT)
    bindings = {
        name: campaign._bound_record(campaign.ROOT / relative)
        for name, relative in campaign.BOUND_PATHS.items()
    }
    contract = {
        "schema_version": 1,
        "contract_id": campaign.CONTRACT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_frozen": True,
        "outcome_blind": True,
        "authorization": "Phase 2 of v011_evidence_phase_instruction_20260721.md",
        "harness_freeze_git_head": head,
        "bindings": bindings,
        "protocol_sha256": campaign.protocol_sha256(),
        "protocol": campaign.frozen_protocol(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(
                (json.dumps(contract, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
                    "utf-8"
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return contract


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=campaign.CONTRACT_PATH)
    args = parser.parse_args(argv)
    freeze(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
