"""Integrity checks for the committed v0.11 compute-ladder result."""

from __future__ import annotations

import json

from benchmarks import run_v011_compute_ladder as campaign


def test_committed_v3_result_artifacts_match_terminal_and_attestation():
    prefix = campaign.ROOT / "benchmarks" / "v011_compute_ladder_v3"
    name = prefix.name
    raw_path = prefix.with_name(f"{name}_raw.json")
    manifest_path = prefix.with_name(f"{name}_manifest.json")
    terminal_path = prefix.with_name(f"{name}_terminal.json")
    attestation_path = prefix.with_name(f"{name}_analysis_attestation.json")
    result_path = prefix.with_name(f"{name}_result.json")
    coordinate_path = prefix.with_name(f"{name}_coordinate_ratios.csv")
    per_dataset_path = prefix.with_name(f"{name}_per_dataset.csv")
    generated_report_path = prefix.with_name(f"{name}_generated_report.md")

    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert terminal["status"] == "complete"
    assert terminal["completed_worker_count"] == campaign.EXPECTED_WORKERS
    assert terminal["raw"] == {
        "bytes": raw_path.stat().st_size,
        "path": "raw.json",
        "sha256": campaign.sha256(raw_path),
    }
    assert attestation["input"] == {
        "contract_sha256": campaign.sha256(campaign.CONTRACT_PATH),
        "manifest_sha256": campaign.sha256(manifest_path),
        "raw_sha256": campaign.sha256(raw_path),
    }
    assert attestation["outputs"]["summary.json"]["sha256"] == campaign.sha256(
        result_path
    )
    coordinate_sha = attestation["outputs"]["coordinate_ratios.csv"]["sha256"]
    assert coordinate_sha == campaign.sha256(coordinate_path)
    per_dataset_sha = attestation["outputs"]["per_dataset.csv"]["sha256"]
    assert per_dataset_sha == campaign.sha256(per_dataset_path)
    generated_report_sha = attestation["outputs"]["report.md"]["sha256"]
    assert generated_report_sha == campaign.sha256(generated_report_path)
    assert result["counts"] == {
        "arms": 6,
        "coordinates": 39,
        "datasets": 13,
        "paired_rows": 351,
        "per_dataset_rows": 117,
        "workers": 234,
    }
    assert result["strict_program_verdict"]["strict_pareto_victory"] is False
    assert result["policy_advancement_allowed"] is False
