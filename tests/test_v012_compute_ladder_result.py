"""Integrity checks for the committed v0.12 compute-ladder result."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks import analyze_v012_compute_ladder as analysis
from benchmarks import run_v012_compute_ladder as campaign


def test_committed_v012_result_artifacts_match_terminal_and_attestation():
    prefix = campaign.ROOT / "benchmarks" / "v012_compute_ladder_20260724"
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
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert terminal["status"] == "complete"
    assert terminal["completed_worker_count"] == campaign.EXPECTED_WORKERS
    assert terminal["raw"] == {
        "bytes": raw_path.stat().st_size,
        "path": "raw.json",
        "sha256": campaign.sha256(raw_path),
    }
    assert manifest["analyzer"]["sha256"] == analysis.PLANNED_ANALYZER_SHA256
    assert attestation["input"] == {
        "executed_analyzer_sha256": campaign.sha256(Path(analysis.__file__)),
        "manifest_sha256": campaign.sha256(manifest_path),
        "planned_analyzer_sha256": analysis.PLANNED_ANALYZER_SHA256,
        "raw_sha256": campaign.sha256(raw_path),
        "terminal_sha256": campaign.sha256(terminal_path),
    }
    committed_outputs = {
        "coordinate_ratios.csv": coordinate_path,
        "per_dataset.csv": per_dataset_path,
        "report.md": generated_report_path,
        "summary.json": result_path,
    }
    expected_output_hashes = {
        "coordinate_ratios.csv": (
            "5d03c3c73ffb6d11fe8de8e64f2cec511b29bf4689d94ce976c7a17ea2b07b34"
        ),
        "per_dataset.csv": (
            "4429bd1043ef11b7fbf755e822a61069489b50a177d92fc6565db40ee09e4e48"
        ),
        "report.md": (
            "5c40287f6a85f3ac535951711facb9478b1cb5c30ecd5a9ae3257263ece1009a"
        ),
        "summary.json": (
            "d2c1d814f549f7a5b12aef241a8670f20dedc489a413e4762a7c82b3ccc347f2"
        ),
    }
    assert set(attestation["outputs"]) == set(committed_outputs)
    for output_name, committed_path in committed_outputs.items():
        record = attestation["outputs"][output_name]
        assert record["path"] == output_name
        assert record["bytes"] == committed_path.stat().st_size
        assert record["sha256"] == expected_output_hashes[output_name]
        assert campaign.sha256(committed_path) == expected_output_hashes[output_name]

    assert campaign.sha256(raw_path) == (
        "404692f6f89d517bfeb470127267e3b18857a1c3e8bb12acf9cda6fcf9984809"
    )
    assert campaign.sha256(manifest_path) == (
        "3ddc1a8b1a2a0fab80afd0a5d6d7e7e895230d8a731a2e391c6b48965330f63b"
    )
    assert campaign.sha256(terminal_path) == (
        "d54580ce7453ab4e8b572947722c5ab81c0f360d5c09e748985f0d395ef19037"
    )
    assert campaign.sha256(attestation_path) == (
        "133ae800ac0a95479bc9e2940c8a92fec192fe8c524f9797a3051b9e171a6852"
    )

    assert result["counts"] == {
        "arms": 6,
        "coordinates": 39,
        "datasets": 13,
        "paired_rows": 351,
        "per_dataset_rows": 117,
        "workers": 234,
    }
    assert result["decision"] == "descriptive_release_scoreboard"
    assert result["strict_program_verdict"] == {
        "basis": "equal_dataset_point_estimates",
        "counterpart_peak_rss_no_worse": False,
        "fit_frontier_dominance": False,
        "prediction_frontier_dominance": False,
        "strict_pareto_victory": False,
    }


def test_committed_v012_matched_profile_readout_is_exact():
    path = campaign.ROOT / "benchmarks" / "v012_compute_ladder_20260724_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    contrasts = {row["contrast"]: row for row in result["matched_profile_contrasts"]}

    expected = {
        "D0/M0": (1.0097093808128663, 2.601617953137915, 3.2710422491561904),
        "DA/MA": (0.9880948142729871, 1.250034629905852, 3.3451064317543002),
        "D8/M8": (1.0362909194675003, 3.570807795565795, 1.824998584658173),
    }
    for contrast, (quality, fit, predict) in expected.items():
        metrics = contrasts[contrast]["metrics"]
        assert metrics["test_rmse"]["ratio"] == quality
        assert metrics["fit_seconds"]["ratio"] == fit
        assert metrics["prediction_seconds_per_call"]["ratio"] == predict


def test_committed_v012_analysis_rerun_disclosure_is_retained():
    path = (
        campaign.ROOT
        / "benchmarks"
        / "v012_compute_ladder_20260724_generated_report.md"
    )
    report = path.read_text(encoding="utf-8")
    assert "The completed 234-worker measurement was not rerun or changed" in report
    assert "both analyzer hashes are retained" in report
