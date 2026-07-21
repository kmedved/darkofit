#!/usr/bin/env python3
"""Freeze the outcome-blind fused-lane calibration execution contract."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from . import fused_lane_dispatch_campaign as campaign
    from . import run_fused_lane_dispatch as runner
except ImportError:  # direct script execution
    import fused_lane_dispatch_campaign as campaign
    import run_fused_lane_dispatch as runner


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "fused_lane_dispatch_calibration_contract.json"
BOUND_PATHS = {
    "design_v1": "benchmarks/fused_lane_dispatch_v1_contract.md",
    "design_v2": "benchmarks/fused_lane_dispatch_v2_contract.md",
    "bin_width_erratum": (
        "benchmarks/fused_lane_dispatch_v2_bin_width_erratum_20260721.md"
    ),
    "execution_protocol": (
        "benchmarks/fused_lane_dispatch_calibration_protocol.md"
    ),
    "campaign": "benchmarks/fused_lane_dispatch_campaign.py",
    "runner": "benchmarks/run_fused_lane_dispatch.py",
    "freezer": "benchmarks/freeze_fused_lane_dispatch_calibration.py",
    "selector": "darkofit/booster.py",
    "persistence": "darkofit/serialization.py",
    "public_api": "darkofit/sklearn_api.py",
    "loss_determinism": "darkofit/losses.py",
    "campaign_tests": "tests/test_fused_lane_dispatch_campaign.py",
    "dispatch_tests": "tests/test_oblivious_kernel_dispatch.py",
    "fused_kernel_tests": "tests/test_fused_oblivious_kernel.py",
    "fused_expanded_tests": "tests/test_fused_oblivious_expanded.py",
    "thread_tests": "tests/test_thread_state_restoration.py",
    "loss_tests": "tests/test_loss_determinism.py",
}


def _bound(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"fused-lane bound input is invalid: {relative}")
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": campaign.file_sha256(path),
    }


def build_contract() -> dict[str, Any]:
    state = runner.git_state(ROOT)
    if state["status"]:
        raise RuntimeError("fused-lane freeze requires a clean harness tree")
    cache_root = ROOT / ".cache" / campaign.CAMPAIGN_NAME / "calibration"
    thread_environments = {}
    for threads in sorted({spec["threads"] for spec in campaign.calibration_specs()}):
        environment = runner.fixed_worker_environment(
            threads, cache_root / f"threads-{threads}"
        )
        thread_environments[str(threads)] = {
            name: environment.get(name)
            for name in (
                *runner.THREAD_ENV_KEYS,
                "DARKOFIT_WARMUP",
                "NUMBA_CACHE_DIR",
                "NUMBA_DISABLE_JIT",
                "NUMBA_NUM_THREADS",
                "NUMBA_THREADING_LAYER",
                "OMP_DYNAMIC",
                "OMP_THREAD_LIMIT",
                "MKL_DYNAMIC",
                "PYTHONHASHSEED",
                "PYTHONPATH",
            )
        }
    return {
        "schema_version": campaign.SCHEMA_VERSION,
        "campaign": campaign.CAMPAIGN_NAME,
        "phase": "calibration",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "contract_frozen": True,
        "outcomes_opened": False,
        "execution_authorized": False,
        "source": state["head"],
        "source_worktree": {
            "required": True,
            "clean": True,
            "detached_at_source": True,
            "import_must_resolve_inside_worktree": True,
        },
        "bound_files": {
            name: _bound(path) for name, path in BOUND_PATHS.items()
        },
        "runtime": {
            "fingerprint": runner.runtime_fingerprint(),
            "worker_environments": thread_environments,
        },
        "generator": {
            "seed": campaign.CALIBRATION_SEED,
            "specs": list(campaign.calibration_specs()),
        },
        "execution": {
            "fresh_worker_per_coordinate": True,
            "both_lanes_per_worker": True,
            "warmups_per_lane": campaign.CALIBRATION_WARMUPS,
            "paired_repetitions": campaign.CALIBRATION_REPEATS,
            "orders": [
                list(campaign.calibration_order(repeat))
                for repeat in range(campaign.CALIBRATION_REPEATS)
            ],
            "timed_region": "build_oblivious_tree_only",
            "shared_input_and_buffer_allocation_timed": False,
            "create_only": True,
            "terminal_failure_nonrerunnable": True,
            "partial_rows_published": False,
        },
        "decision_rules": {
            "stability_iqr_over_median_at_most": campaign.STABILITY_LIMIT,
            "selected_fused_geomean_at_most": (
                campaign.CALIBRATION_GEOMEAN_LIMIT
            ),
            "worst_selected_fused_at_most": campaign.CALIBRATION_WORST_LIMIT,
            "threshold_tie_tolerance": 0.001,
            "threshold_tie_policy": "largest_never_switch_largest",
            "both_lanes_must_be_selected": True,
            "all_cells_exact": True,
        },
        "outputs": {
            "authorization": (
                "benchmarks/fused_lane_dispatch_calibration_authorization_v1.json"
            ),
            "raw": "benchmarks/fused_lane_dispatch_calibration_raw_v1.json",
            "terminal": (
                "benchmarks/fused_lane_dispatch_calibration_raw_v1_terminal.json"
            ),
            "analysis": (
                "benchmarks/fused_lane_dispatch_calibration_analysis_v1.json"
            ),
        },
        "authorization_contract": {
            "separate_create_only_record_required": True,
            "required_fields": {
                "schema_version": campaign.SCHEMA_VERSION,
                "campaign": campaign.CAMPAIGN_NAME,
                "phase": "calibration",
                "execution_authorized": True,
                "execution_contract_sha256": "<this contract SHA-256>",
                "source": state["head"],
                "owner_decision": "<nonempty explicit decision>",
            },
        },
        "downstream": {
            "calibration_execution_authorized": False,
            "validation_execution_authorized": False,
            "auto_threshold_change_authorized": False,
            "speed_claim_authorized": False,
            "release_authorized": False,
            "m2_m4_q_or_lockbox_authorized": False,
            "next_mechanism_slot": "quality_first",
        },
    }


def main() -> int:
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise RuntimeError(f"refusing existing fused-lane contract: {OUTPUT}")
    state_before = runner.git_state(ROOT)
    contract = build_contract()
    if runner.git_state(ROOT) != state_before:
        raise RuntimeError("fused-lane source changed during contract freeze")
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
                "execution_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
