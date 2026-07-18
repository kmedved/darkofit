import json
from pathlib import Path

from benchmarks import record_t5_composite_confirmation_failure as failure
from benchmarks import run_t5_composite_confirmation as runner


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks" / "t5_composite_confirmation_failure.json"


def test_t5_failure_artifact_is_hash_bound_and_fail_closed():
    artifact = json.loads(ARTIFACT.read_text())
    expected_hash = artifact.pop("failure_artifact_sha256")
    assert runner._json_sha256(artifact) == expected_hash
    assert artifact["decision"] == "close_t5_composite_candidate"
    assert artifact["candidate_arm_started"] is False
    assert artifact["default_promotion_authorized"] is False
    assert artifact["rerun_authorized"] is False
    assert artifact["execution"]["completed_worker_count"] == 23
    assert artifact["execution"]["failed_before_fit_count"] == 2
    assert {
        row["task_id"] for row in artifact["execution"]["invalid_targets"]
    } == set(failure.INVALID_TARGETS)
    assert artifact["panel_disposition"][
        "all_25_lineages_spent_for_confirmation"
    ]
