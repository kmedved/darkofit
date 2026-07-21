#!/usr/bin/env python3
"""Freeze the outcome-blind fused-lane calibration v3 execution contract."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

try:
    from . import freeze_fused_lane_dispatch_calibration as v1
    from . import freeze_fused_lane_dispatch_calibration_v2 as v2
    from . import fused_lane_dispatch_campaign as campaign
    from . import run_fused_lane_dispatch as runner
except ImportError:  # direct script execution
    import freeze_fused_lane_dispatch_calibration as v1
    import freeze_fused_lane_dispatch_calibration_v2 as v2
    import fused_lane_dispatch_campaign as campaign
    import run_fused_lane_dispatch as runner


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT / "benchmarks" / "fused_lane_dispatch_calibration_contract_v3.json"
)
EXECUTION_IDENTITY = "calibration_v3"
V2_CONTRACT = "benchmarks/fused_lane_dispatch_calibration_contract_v2.json"
V2_CONTRACT_SHA256 = (
    "b2075f9c45df3b3fb674c74fe0b47cd9ddd1ec3bae790f5379308e15a327061a"
)
BOUND_PATHS = {
    **v2.BOUND_PATHS,
    "execution_protocol_v3": (
        "benchmarks/fused_lane_dispatch_calibration_v3_protocol.md"
    ),
    "freezer_v3": (
        "benchmarks/freeze_fused_lane_dispatch_calibration_v3.py"
    ),
    "execution_contract_v2": V2_CONTRACT,
    "tree_builder": "darkofit/tree.py",
    "library_tests": "tests/test_darkofit.py",
    "private_ensemble_tests": "tests/test_private_ensemble_v3.py",
    "v3_freezer_tests": "tests/test_fused_lane_dispatch_v3_freezer.py",
}


def build_contract():
    contract = v2.build_contract()
    if campaign.file_sha256(ROOT / V2_CONTRACT) != V2_CONTRACT_SHA256:
        raise RuntimeError("fused-lane v2 execution contract drifted")
    v2_contract = json.loads((ROOT / V2_CONTRACT).read_text(encoding="utf-8"))
    v2_runtime = v2_contract.get("runtime")
    v2_environments = (
        v2_runtime.get("worker_environments")
        if isinstance(v2_runtime, dict)
        else None
    )
    if not isinstance(v2_environments, dict) or not v2_environments:
        raise RuntimeError("fused-lane v2 worker environments are invalid")
    contract["runtime"]["worker_environments"] = deepcopy(v2_environments)
    contract["execution_identity"] = EXECUTION_IDENTITY
    contract["supersedes"] = {
        "execution_identity": "calibration_v2",
        "contract_path": V2_CONTRACT,
        "contract_sha256": V2_CONTRACT_SHA256,
        "formal_worker_started": False,
        "outcomes_opened": False,
        "reason": "pre-authorization execution and provenance gate repair",
        "scientific_grid_or_gate_changed": False,
    }
    contract["bound_files"] = {
        name: v1._bound(path) for name, path in BOUND_PATHS.items()
    }
    contract["execution"]["v2_formal_worker_started"] = False
    contract["execution"]["scientific_change_from_v2"] = False
    contract["execution"]["worker_self_authorization_required"] = True
    contract["execution"]["exact_frozen_worker_environment_required"] = True
    contract["execution"]["actual_builder_counters_required"] = True
    contract["execution"]["hash_bound_threshold_analysis_required"] = True
    contract["outputs"] = {
        "authorization": (
            "benchmarks/fused_lane_dispatch_calibration_authorization_v3.json"
        ),
        "raw": "benchmarks/fused_lane_dispatch_calibration_raw_v3.json",
        "terminal": (
            "benchmarks/fused_lane_dispatch_calibration_raw_v3_terminal.json"
        ),
        "analysis": (
            "benchmarks/fused_lane_dispatch_calibration_analysis_v3.json"
        ),
    }
    contract["authorization_contract"]["required_fields"][
        "execution_identity"
    ] = EXECUTION_IDENTITY
    return contract


def main() -> int:
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise RuntimeError(f"refusing existing fused-lane v3 contract: {OUTPUT}")
    state_before = runner.git_state(ROOT)
    contract = build_contract()
    if runner.git_state(ROOT) != state_before:
        raise RuntimeError("fused-lane source changed during v3 contract freeze")
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
