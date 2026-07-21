#!/usr/bin/env python3
"""Create the prospective, pre-outcome Wave-2 M3b contract."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from . import paired_evidence_contract as paired
    from . import run_m3b_ensemble_v3 as runner
except ImportError:  # direct script execution
    import paired_evidence_contract as paired
    import run_m3b_ensemble_v3 as runner


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "m3b_ensemble_v3_contract.json"

BOUND_PATHS = runner.BOUND_PATHS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"M3b bound input is not a regular file: {relative}")
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build_contract() -> dict[str, Any]:
    source_state = runner.git_state(ROOT)
    panel_cache = runner.DEFAULT_PANEL_CACHE.resolve()
    if not panel_cache.is_file() or panel_cache.is_symlink():
        raise RuntimeError("M3b sports-panel cache is unavailable")
    m3a_contract = json.loads(runner.M3A_CONTRACT_PATH.read_text(encoding="utf-8"))
    manifests = runner.expected_case_manifests(panel_cache)
    fingerprints = {
        case_id: record["fingerprints"] for case_id, record in manifests.items()
    }
    panel_record = {
        "contract_path": str(panel_cache.relative_to(ROOT)),
        "bytes": panel_cache.stat().st_size,
        "sha256": _sha256(panel_cache),
    }
    if (
        panel_record["bytes"] != m3a_contract["sports_panel"]["processed_bytes"]
        or panel_record["sha256"] != m3a_contract["sports_panel"]["processed_sha256"]
    ):
        raise RuntimeError("M3b sports-panel cache drifted from frozen M3a")
    return {
        "schema_version": runner.SCHEMA_VERSION,
        "name": runner.CONTRACT_NAME,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "contract_frozen": True,
        "outcomes_opened": False,
        "paired_execution_contract": paired.CONTRACT_VERSION,
        "threads": runner.THREADS,
        "sources": {
            "darkofit": source_state["head"],
            "m3a_darkofit": m3a_contract["sources"]["darkofit"],
            "m3a_chimeraboost": m3a_contract["sources"]["chimeraboost"],
        },
        "bound_files": {name: _bound(path) for name, path in BOUND_PATHS.items()},
        "panel_cache": panel_record,
        "m3a_sports_archive": {
            "contract_sha256": _sha256(runner.M3A_CONTRACT_PATH),
            "spent": True,
            "historical_files_preserved": True,
        },
        "cases": list(runner.case_specs()),
        "case_manifests": manifests,
        "case_fingerprints": fingerprints,
        "arms": {arm: runner.arm_config(arm) for arm in runner.ARMS},
        "quality_orders": runner.quality_orders(),
        "execution": runner.execution_contract(),
        "decision_rules": runner.decision_rules(),
        "claims": runner.claim_contract(),
    }


def main() -> int:
    state = runner.git_state(ROOT)
    if state["status"]:
        raise RuntimeError("M3b freeze requires a clean harness tree")
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise RuntimeError(f"refusing existing M3b contract: {OUTPUT}")
    contract = build_contract()
    state_after = runner.git_state(ROOT)
    if state_after != state or contract["sources"]["darkofit"] != state["head"]:
        raise RuntimeError("M3b source changed during contract construction")
    payload = (
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    paired.write_create_only(OUTPUT, payload)
    print(
        json.dumps(
            {
                "contract": str(OUTPUT),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source": state["head"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
