#!/usr/bin/env python3
"""Create the prospective attempt-2 M3b contract."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from . import paired_evidence_contract as paired
    from . import run_m3b_ensemble_v3_r2 as runner
except ImportError:  # direct script execution
    import paired_evidence_contract as paired
    import run_m3b_ensemble_v3_r2 as runner


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = runner.CONTRACT_PATH
ATTEMPT1_CONTRACT = ROOT / "benchmarks" / "m3b_ensemble_v3_contract.json"
ATTEMPT1_FAILURE_RECORD = (
    ROOT / "benchmarks" / "m3b_ensemble_v3_attempt1_failure_record.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound(relative: str):
    path = ROOT / relative
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"M3b attempt-2 bound input is invalid: {relative}")
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_object(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"M3b attempt-2 input is not an object: {path}")
    return value


def build_contract():
    runner.assert_rss_capability()
    harness_state = runner.git_state(ROOT)
    attempt1 = _load_object(ATTEMPT1_CONTRACT)
    failure = _load_object(ATTEMPT1_FAILURE_RECORD)
    if (
        _sha256(ATTEMPT1_CONTRACT) != failure["failed_contract"]["sha256"]
        or failure["terminal_artifact"]["completed_rows_discarded"] != 0
        or failure["failure"]["model_outcomes_opened"] is not False
        or failure["disposition"]["attempt_terminal"] is not True
        or failure["disposition"]["rerun_same_identity"] is not False
    ):
        raise RuntimeError("M3b attempt-1 failure lineage is invalid")
    manifests = runner.expected_case_manifests(runner.DEFAULT_PANEL_CACHE)
    fingerprints = {
        case_id: record["fingerprints"] for case_id, record in manifests.items()
    }
    if (
        manifests != attempt1["case_manifests"]
        or fingerprints != attempt1["case_fingerprints"]
        or list(runner.case_specs()) != attempt1["cases"]
        or {arm: runner.arm_config(arm) for arm in runner.ARMS} != attempt1["arms"]
        or runner.quality_orders() != attempt1["quality_orders"]
        or runner.decision_rules() != attempt1["decision_rules"]
    ):
        raise RuntimeError("M3b attempt-2 scientific grid drifted from attempt 1")
    implementation = _bound("darkofit/sklearn_api.py")
    if implementation["sha256"] != attempt1["bound_files"]["implementation"]["sha256"]:
        raise RuntimeError("M3b attempt-2 model implementation changed")
    panel_cache = {
        "contract_path": str(runner.DEFAULT_PANEL_CACHE.relative_to(ROOT)),
        "bytes": runner.DEFAULT_PANEL_CACHE.stat().st_size,
        "sha256": _sha256(runner.DEFAULT_PANEL_CACHE),
    }
    if panel_cache != attempt1["panel_cache"]:
        raise RuntimeError("M3b attempt-2 panel cache changed")
    return {
        "schema_version": runner.SCHEMA_VERSION,
        "name": runner.CONTRACT_NAME,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "contract_frozen": True,
        "outcomes_opened": False,
        "paired_execution_contract": paired.CONTRACT_VERSION,
        "threads": runner.THREADS,
        "rss_scope": runner.RSS_SCOPE,
        "sources": {
            "darkofit": attempt1["sources"]["darkofit"],
            "harness": harness_state["head"],
            "m3a_darkofit": attempt1["sources"]["m3a_darkofit"],
            "m3a_chimeraboost": attempt1["sources"]["m3a_chimeraboost"],
        },
        "attempt_lineage": {
            "attempt1_contract_sha256": _sha256(ATTEMPT1_CONTRACT),
            "attempt1_failure_record_sha256": _sha256(ATTEMPT1_FAILURE_RECORD),
            "attempt1_terminal_artifact_sha256": failure["terminal_artifact"]["sha256"],
            "attempt1_completed_rows": 0,
            "attempt1_model_outcomes_opened": False,
            "attempt1_rerun": False,
            "sole_amendment": "self-worker RSS with capability preflight",
        },
        "bound_files": {
            name: _bound(path) for name, path in runner.BOUND_PATHS.items()
        },
        "panel_cache": panel_cache,
        "m3a_sports_archive": attempt1["m3a_sports_archive"],
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
    runner.assert_rss_capability()
    state = runner.git_state(ROOT)
    if state["status"]:
        raise RuntimeError("M3b attempt-2 freeze requires a clean harness tree")
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise RuntimeError(f"refusing existing M3b attempt-2 contract: {OUTPUT}")
    contract = build_contract()
    if (
        runner.git_state(ROOT) != state
        or contract["sources"]["harness"] != state["head"]
    ):
        raise RuntimeError("M3b attempt-2 harness changed during freeze")
    payload = (
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    paired.write_create_only(OUTPUT, payload)
    print(
        json.dumps(
            {
                "contract": str(OUTPUT),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source": contract["sources"]["darkofit"],
                "harness": state["head"],
                "rss_scope": runner.RSS_SCOPE,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
