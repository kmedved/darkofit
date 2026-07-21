#!/usr/bin/env python3
"""Create the prospective Wave-3 B-archive feasibility contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from . import paired_evidence_contract as paired
    from . import run_barchive_v1 as runner
except ImportError:  # direct script execution
    import paired_evidence_contract as paired
    import run_barchive_v1 as runner


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = runner.CONTRACT_PATH
HISTORICAL_CONTRACT = ROOT / "benchmarks" / "m3b_ensemble_v3_r3_contract.json"
HISTORICAL_RESULT = ROOT / "benchmarks" / "m3b_ensemble_v3_r3_result.json"
VS_SINGLE_READOUT = (
    ROOT / "benchmarks" / "m3b_ensemble_v3_r3_vs_single_readout_20260721.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"B-archive freeze input is invalid: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"B-archive freeze input is not an object: {path}")
    return value


def _git_blob(relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{runner.MODEL_SOURCE_HEAD}:{relative}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(f"B-archive source pin is missing {relative}")
    return completed.stdout


def _bound(relative: str) -> dict[str, Any]:
    if relative in runner.PINNED_SOURCE_PATHS:
        payload = _git_blob(relative)
        return {
            "path": relative,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    path = ROOT / relative
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"B-archive bound input is invalid: {relative}")
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build_contract() -> dict[str, Any]:
    runner.assert_frozen_runtime()
    state = runner.m3b.git_state(ROOT)
    historical = _load_object(HISTORICAL_CONTRACT)
    result = _load_object(HISTORICAL_RESULT)
    readout = _load_object(VS_SINGLE_READOUT)
    manifests = runner.m3b.expected_case_manifests(runner.DEFAULT_PANEL_CACHE)
    fingerprints = {
        case_id: record["fingerprints"] for case_id, record in manifests.items()
    }
    expected_cases = list(runner.m3b.case_specs())
    expected_panel = {
        "contract_path": str(runner.DEFAULT_PANEL_CACHE.relative_to(ROOT)),
        "bytes": runner.DEFAULT_PANEL_CACHE.stat().st_size,
        "sha256": _sha256(runner.DEFAULT_PANEL_CACHE),
    }
    changed_source = runner.m3b._git(
        ROOT,
        "diff",
        "--name-only",
        runner.MODEL_SOURCE_HEAD,
        state["head"],
        "--",
        *sorted(runner.PINNED_SOURCE_PATHS),
    )
    if changed_source:
        raise RuntimeError("B-archive implementation differs from its source pin")
    if (
        expected_cases != historical.get("cases")
        or manifests != historical.get("case_manifests")
        or fingerprints != historical.get("case_fingerprints")
        or expected_panel != historical.get("panel_cache")
        or result.get("disposition") != "close_b1_b2_preserve_existing_opt_in"
        or result.get("candidates", {})
        .get(runner.COMBINED, {})
        .get("resources", {})
        .get("median_archive_to_single")
        != 5.534767493867151
        or readout.get("finding", {}).get("combined_beats_single_all_cases") is not True
        or readout.get("finding", {}).get("combined_case_count") != 13
        or readout.get("finding", {}).get("serialization_authorized") is not False
        or readout.get("amends_frozen_m3b_result") is not False
        or not str(readout.get("sports_primary_scope", "")).startswith(
            "player-disjoint"
        )
    ):
        raise RuntimeError("B-archive immutable M3b lineage is invalid")
    return {
        "schema_version": runner.SCHEMA_VERSION,
        "name": runner.CONTRACT_NAME,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "contract_frozen": True,
        "outcomes_opened": False,
        "sources": {
            "darkofit": runner.MODEL_SOURCE_HEAD,
            "harness": state["head"],
        },
        "bound_files": {
            name: _bound(relative) for name, relative in runner.BOUND_PATHS.items()
        },
        "m3b_r3_lineage": runner.m3b_r3_lineage(),
        "panel_cache": expected_panel,
        "cases": expected_cases,
        "case_manifests": manifests,
        "case_fingerprints": fingerprints,
        "expected_shared_preprocessing": (runner.expected_shared_preprocessing()),
        "execution": runner.execution_contract(),
        "decision_rules": runner.decision_rules(),
        "claims": runner.claim_contract(),
    }


def main() -> int:
    state = runner.m3b.git_state(ROOT)
    if state["status"]:
        raise RuntimeError("B-archive freeze requires a clean harness tree")
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise RuntimeError(f"refusing existing B-archive contract: {OUTPUT}")
    contract = build_contract()
    if (
        runner.m3b.git_state(ROOT) != state
        or contract["sources"]["harness"] != state["head"]
    ):
        raise RuntimeError("B-archive harness changed during freeze")
    payload = (
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    paired.write_create_only(OUTPUT, payload)
    print(
        json.dumps(
            {
                "contract": str(OUTPUT),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source": runner.MODEL_SOURCE_HEAD,
                "harness": state["head"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
