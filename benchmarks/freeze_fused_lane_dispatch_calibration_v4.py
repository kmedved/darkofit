#!/usr/bin/env python3
"""Freeze the outcome-blind fused-lane calibration v4 execution contract."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

try:
    from . import freeze_fused_lane_dispatch_calibration as v1
    from . import freeze_fused_lane_dispatch_calibration_v3 as v3
    from . import fused_lane_dispatch_campaign as campaign
    from . import run_fused_lane_dispatch as runner
except ImportError:  # direct script execution
    import freeze_fused_lane_dispatch_calibration as v1
    import freeze_fused_lane_dispatch_calibration_v3 as v3
    import fused_lane_dispatch_campaign as campaign
    import run_fused_lane_dispatch as runner


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT / "benchmarks" / "fused_lane_dispatch_calibration_contract_v4.json"
)
EXECUTION_IDENTITY = "calibration_v4"
V3_CONTRACT = "benchmarks/fused_lane_dispatch_calibration_contract_v3.json"
V3_CONTRACT_SHA256 = (
    "c55ee50fccda5b9ba24e004ae8a27285e4db92e52a9c17a668bc1b417b0fa648"
)
BOUND_PATHS = {
    **v3.BOUND_PATHS,
    "execution_protocol_v4": (
        "benchmarks/fused_lane_dispatch_calibration_v4_protocol.md"
    ),
    "freezer_v4": (
        "benchmarks/freeze_fused_lane_dispatch_calibration_v4.py"
    ),
    "execution_contract_v3": V3_CONTRACT,
    "ensemble_tests": "tests/test_ensemble_api.py",
    "v4_freezer_tests": "tests/test_fused_lane_dispatch_v4_freezer.py",
}


def build_contract():
    contract = v3.build_contract()
    if campaign.file_sha256(ROOT / V3_CONTRACT) != V3_CONTRACT_SHA256:
        raise RuntimeError("fused-lane v3 execution contract drifted")
    v3_contract = json.loads((ROOT / V3_CONTRACT).read_text(encoding="utf-8"))
    v3_runtime = v3_contract.get("runtime")
    v3_environments = (
        v3_runtime.get("worker_environments")
        if isinstance(v3_runtime, dict)
        else None
    )
    if not isinstance(v3_environments, dict) or not v3_environments:
        raise RuntimeError("fused-lane v3 worker environments are invalid")
    contract["runtime"]["worker_environments"] = deepcopy(v3_environments)
    contract["execution_identity"] = EXECUTION_IDENTITY
    contract["supersedes"] = {
        "execution_identity": "calibration_v3",
        "contract_path": V3_CONTRACT,
        "contract_sha256": V3_CONTRACT_SHA256,
        "formal_worker_started": False,
        "outcomes_opened": False,
        "reason": (
            "pre-authorization parent-capability, constructor-provenance, "
            "weighted-class, and production-layout repair"
        ),
        "scientific_grid_or_gate_changed": False,
    }
    contract["bound_files"] = {
        name: v1._bound(path) for name, path in BOUND_PATHS.items()
    }
    contract["execution"]["v3_formal_worker_started"] = False
    contract["execution"]["scientific_change_from_v3"] = False
    contract["execution"]["parent_pipe_capability_required"] = True
    contract["execution"]["authorization_alone_rejected_by_workers"] = True
    contract["execution"]["production_routing_layout_required"] = True
    contract["execution"]["wrapper_booster_kernel_binding_required"] = True
    contract["execution"]["positive_mass_class_semantics_required"] = True
    contract["outputs"] = {
        "authorization": (
            "benchmarks/fused_lane_dispatch_calibration_authorization_v4.json"
        ),
        "raw": "benchmarks/fused_lane_dispatch_calibration_raw_v4.json",
        "terminal": (
            "benchmarks/fused_lane_dispatch_calibration_raw_v4_terminal.json"
        ),
        "analysis": (
            "benchmarks/fused_lane_dispatch_calibration_analysis_v4.json"
        ),
    }
    contract["authorization_contract"]["required_fields"][
        "execution_identity"
    ] = EXECUTION_IDENTITY
    return contract


def main() -> int:
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise RuntimeError(f"refusing existing fused-lane v4 contract: {OUTPUT}")
    state_before = runner.git_state(ROOT)
    contract = build_contract()
    if runner.git_state(ROOT) != state_before:
        raise RuntimeError("fused-lane source changed during v4 contract freeze")
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
