#!/usr/bin/env python3
"""Create the prospective attempt-3 M3b contract."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from . import paired_evidence_contract as paired
    from . import run_m3b_ensemble_v3_r3 as runner
except ImportError:  # direct script execution
    import paired_evidence_contract as paired
    import run_m3b_ensemble_v3_r3 as runner


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = runner.CONTRACT_PATH
ATTEMPT1_CONTRACT = ROOT / "benchmarks" / "m3b_ensemble_v3_contract.json"
ATTEMPT1_FAILURE = ROOT / "benchmarks" / "m3b_ensemble_v3_attempt1_failure_record.json"
ATTEMPT2_CONTRACT = ROOT / "benchmarks" / "m3b_ensemble_v3_r2_contract.json"
ATTEMPT2_FAILURE = ROOT / "benchmarks" / "m3b_ensemble_v3_attempt2_failure_record.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"M3b attempt-3 input is not an object: {path}")
    return value


def _bound(relative: str):
    path = ROOT / relative
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"M3b attempt-3 bound input is invalid: {relative}")
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_lineage():
    attempt1 = _load_object(ATTEMPT1_CONTRACT)
    failure1 = _load_object(ATTEMPT1_FAILURE)
    attempt2 = _load_object(ATTEMPT2_CONTRACT)
    failure2 = _load_object(ATTEMPT2_FAILURE)
    if (
        _sha256(ATTEMPT1_CONTRACT) != failure1["failed_contract"]["sha256"]
        or _sha256(ATTEMPT2_CONTRACT) != failure2["failed_contract"]["sha256"]
        or failure1["terminal_artifact"]["completed_rows_discarded"] != 0
        or failure1["failure"]["model_outcomes_opened"] is not False
        or failure2["terminal_artifact"]["completed_rows_discarded"] != 1
        or failure2["failure"]["completed_results_published"] is not False
        or failure2["failure"]["completed_results_inspected"] is not False
        or failure1["disposition"]["rerun_same_identity"] is not False
        or failure2["disposition"]["rerun_same_identity"] is not False
    ):
        raise RuntimeError("M3b attempt-3 predecessor lineage is invalid")
    return attempt1, failure1, attempt2, failure2


def build_contract():
    runner.assert_rss_capability()
    harness_state = runner.git_state(ROOT)
    attempt1, failure1, attempt2, failure2 = _validate_lineage()
    manifests = runner.expected_case_manifests(runner.DEFAULT_PANEL_CACHE)
    fingerprints = {
        case_id: record["fingerprints"] for case_id, record in manifests.items()
    }
    if (
        manifests != attempt2["case_manifests"]
        or fingerprints != attempt2["case_fingerprints"]
        or list(runner.case_specs()) != attempt2["cases"]
        or {arm: runner.arm_config(arm) for arm in runner.ARMS} != attempt2["arms"]
        or runner.quality_orders() != attempt2["quality_orders"]
        or runner.decision_rules() != attempt2["decision_rules"]
    ):
        raise RuntimeError("M3b attempt-3 scientific grid drifted")
    changed_source_files = runner._git(
        ROOT,
        "diff",
        "--name-only",
        runner.MODEL_SOURCE_HEAD,
        "--",
        "darkofit/sklearn_api.py",
        "tests/test_private_ensemble_v3.py",
    )
    if changed_source_files:
        raise RuntimeError("M3b attempt-3 implementation differs from its source pin")
    panel_cache = {
        "contract_path": str(runner.DEFAULT_PANEL_CACHE.relative_to(ROOT)),
        "bytes": runner.DEFAULT_PANEL_CACHE.stat().st_size,
        "sha256": _sha256(runner.DEFAULT_PANEL_CACHE),
    }
    if panel_cache != attempt2["panel_cache"]:
        raise RuntimeError("M3b attempt-3 panel cache changed")
    attempt1_lineage = attempt2["attempt_lineage"]
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
            "darkofit": runner.MODEL_SOURCE_HEAD,
            "harness": harness_state["head"],
            "m3a_darkofit": attempt2["sources"]["m3a_darkofit"],
            "m3a_chimeraboost": attempt2["sources"]["m3a_chimeraboost"],
        },
        "attempt_lineage": {
            **attempt1_lineage,
            "attempt2_contract_sha256": _sha256(ATTEMPT2_CONTRACT),
            "attempt2_failure_record_sha256": _sha256(ATTEMPT2_FAILURE),
            "attempt2_terminal_artifact_sha256": failure2["terminal_artifact"][
                "sha256"
            ],
            "attempt2_completed_rows_discarded": 1,
            "attempt2_results_published": False,
            "attempt2_results_inspected": False,
            "attempt2_rerun": False,
            "source_fix": ("allow variable sampled row counts for group bootstrap"),
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
        raise RuntimeError("M3b attempt-3 freeze requires a clean harness tree")
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise RuntimeError(f"refusing existing M3b attempt-3 contract: {OUTPUT}")
    contract = build_contract()
    if (
        runner.git_state(ROOT) != state
        or contract["sources"]["harness"] != state["head"]
    ):
        raise RuntimeError("M3b attempt-3 harness changed during freeze")
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
                "rss_scope": runner.RSS_SCOPE,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
