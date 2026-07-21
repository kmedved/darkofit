#!/usr/bin/env python3
"""Freeze the outcome-blind fused-lane calibration v2 execution contract."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from . import freeze_fused_lane_dispatch_calibration as v1
    from . import fused_lane_dispatch_campaign as campaign
    from . import run_fused_lane_dispatch as runner
except ImportError:  # direct script execution
    import freeze_fused_lane_dispatch_calibration as v1
    import fused_lane_dispatch_campaign as campaign
    import run_fused_lane_dispatch as runner


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT / "benchmarks" / "fused_lane_dispatch_calibration_contract_v2.json"
)
EXECUTION_IDENTITY = "calibration_v2"
V1_CONTRACT = "benchmarks/fused_lane_dispatch_calibration_contract.json"
V1_CONTRACT_SHA256 = (
    "3d7f8a653a71d6a9712f57f51bb01421765b42fcd105902f1fb0c6a611f7712d"
)
BOUND_PATHS = {
    **{
        name: path
        for name, path in v1.BOUND_PATHS.items()
        if name not in {"execution_protocol", "freezer"}
    },
    "execution_protocol_v1": (
        "benchmarks/fused_lane_dispatch_calibration_protocol.md"
    ),
    "execution_protocol_v2": (
        "benchmarks/fused_lane_dispatch_calibration_v2_protocol.md"
    ),
    "freezer_v1": "benchmarks/freeze_fused_lane_dispatch_calibration.py",
    "freezer_v2": (
        "benchmarks/freeze_fused_lane_dispatch_calibration_v2.py"
    ),
    "execution_contract_v1": V1_CONTRACT,
}


def build_contract():
    contract = v1.build_contract()
    if campaign.file_sha256(ROOT / V1_CONTRACT) != V1_CONTRACT_SHA256:
        raise RuntimeError("fused-lane v1 execution contract drifted")
    contract["execution_identity"] = EXECUTION_IDENTITY
    contract["supersedes"] = {
        "execution_identity": "calibration_v1",
        "contract_path": V1_CONTRACT,
        "contract_sha256": V1_CONTRACT_SHA256,
        "formal_worker_started": False,
        "outcomes_opened": False,
        "reason": "host-platform-dependent product test expectation",
        "scientific_grid_or_gate_changed": False,
    }
    contract["bound_files"] = {
        name: v1._bound(path) for name, path in BOUND_PATHS.items()
    }
    contract["execution"]["v1_formal_worker_started"] = False
    contract["execution"]["scientific_change_from_v1"] = False
    contract["outputs"] = {
        "authorization": (
            "benchmarks/fused_lane_dispatch_calibration_authorization_v2.json"
        ),
        "raw": "benchmarks/fused_lane_dispatch_calibration_raw_v2.json",
        "terminal": (
            "benchmarks/fused_lane_dispatch_calibration_raw_v2_terminal.json"
        ),
        "analysis": (
            "benchmarks/fused_lane_dispatch_calibration_analysis_v2.json"
        ),
    }
    contract["authorization_contract"]["required_fields"][
        "execution_identity"
    ] = EXECUTION_IDENTITY
    return contract


def main() -> int:
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise RuntimeError(f"refusing existing fused-lane v2 contract: {OUTPUT}")
    state_before = runner.git_state(ROOT)
    contract = build_contract()
    if runner.git_state(ROOT) != state_before:
        raise RuntimeError("fused-lane source changed during v2 contract freeze")
    payload = (
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    runner.write_create_only(OUTPUT, payload)
    print(
        json.dumps(
            {
                "contract": str(OUTPUT),
                "sha256": campaign.file_sha256(OUTPUT),
                "source": contract["source"],
                "execution_identity": EXECUTION_IDENTITY,
                "execution_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
