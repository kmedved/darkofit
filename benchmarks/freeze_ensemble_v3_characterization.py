#!/usr/bin/env python3
"""Create the prospective ensemble-v3 characterization contract once."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

try:
    from . import run_ensemble_v3_characterization as campaign
except ImportError:  # direct script execution
    import run_ensemble_v3_characterization as campaign


def freeze(output: Path) -> dict:
    output = output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"characterization contract is create-only: {output}")
    bindings = {
        name: campaign._bound_record(campaign.ROOT / relative)
        for name, relative in campaign.BOUND_PATHS.items()
    }
    payload = {
        "schema_version": 1,
        "contract_id": campaign.CONTRACT_ID,
        "contract_frozen": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outcome_blind": True,
        "execution": campaign.execution_spec(),
        "bindings": bindings,
        "quality_uncertainty": {
            "sports_cluster_unit": "season",
            "sports_clusters": [2014, 2015, 2016],
            "sports_bootstrap_draws": 100_000,
            "sports_bootstrap_seed": 20_260_720,
            "general_case_count": 4,
            "general_bootstrap_draws": 100_000,
            "general_bootstrap_seed": 20_260_721,
            "percentiles": [2.5, 50.0, 97.5],
            "leave_one_out": True,
        },
        "claims": {
            "tier": "E",
            "characterization_only": True,
            "shipping_or_default_change_authorized": False,
            "m2_or_m4": False,
            "fresh_or_lockbox_data": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=campaign.CONTRACT_PATH)
    return parser.parse_args(argv)


if __name__ == "__main__":
    freeze(parse_args().output)
