#!/usr/bin/env python3
"""Create the prospective v0.11 private ensemble evidence v2 contract."""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

try:
    from . import run_v011_ensemble_evidence_v2 as campaign
except ImportError:
    import run_v011_ensemble_evidence_v2 as campaign


def freeze(output: Path, panel_cache: Path) -> dict:
    output = output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"v0.11 ensemble evidence v2 contract is create-only: {output}")
    state = campaign.source_state(campaign.ROOT)
    if not state["clean"]:
        raise RuntimeError("freeze from a clean committed v2 harness checkout")
    import catboost

    if catboost.__version__ != campaign.CATBOOST_VERSION:
        raise RuntimeError(
            f"expected CatBoost {campaign.CATBOOST_VERSION}, got {catboost.__version__}"
        )
    bindings = {}
    for name, relative in campaign.BOUND_PATHS.items():
        path = campaign.ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"bound input is not a regular file: {relative}")
        bindings[name] = campaign._bound_record(path)
    contract = {
        "schema_version": 1,
        "contract_id": campaign.CONTRACT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_frozen": True,
        "outcome_blind": True,
        "authorization": "Phase 1 of v011_evidence_phase_instruction_20260721.md",
        "attempt_lineage": {
            "v1_formal_workers_started": 0,
            "v1_outcomes_opened": False,
            "v1_raw_or_terminal_published": False,
            "sole_amendment": "capture and discard unmeasured warmup warnings",
        },
        "harness_freeze_state": state,
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "catboost_version": catboost.__version__,
            "catboost_module": str(Path(inspect.getfile(catboost)).resolve()),
        },
        "bindings": bindings,
        "case_manifests": campaign.m3b.expected_case_manifests(panel_cache),
        "immutable_ratios": campaign.immutable_ratios(),
        "execution": campaign.execution_spec(),
        "uncertainty": campaign.uncertainty_spec(),
        "claims": campaign.claim_spec(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(campaign.json_bytes(contract))
    return contract


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=campaign.CONTRACT_PATH)
    parser.add_argument(
        "--panel-cache", type=Path, default=campaign.m3b.DEFAULT_PANEL_CACHE
    )
    args = parser.parse_args(argv)
    args.output = args.output.expanduser().resolve()
    args.panel_cache = args.panel_cache.expanduser().resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    freeze(args.output, args.panel_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
