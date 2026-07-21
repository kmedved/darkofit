"""Regression checks for the published Wave 2 M3b evidence chain."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from benchmarks import analyze_m3b_ensemble_v3_r3 as analyzer
from benchmarks import run_m3b_ensemble_v3_r3 as runner


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
CONTRACT = BENCHMARKS / "m3b_ensemble_v3_r3_contract.json"
QUALITY = BENCHMARKS / "m3b_ensemble_v3_r3_quality.json"
GATE = BENCHMARKS / "m3b_ensemble_v3_r3_gate.json"
TIMING = BENCHMARKS / "m3b_ensemble_v3_r3_timing.json"
RESULT = BENCHMARKS / "m3b_ensemble_v3_r3_result.json"
NOTE = BENCHMARKS / "m3b_ensemble_v3_r3_result.md"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_m3b_terminal_attempts_match_their_failure_records():
    for attempt in (1, 2):
        terminal = BENCHMARKS / f"m3b_ensemble_v3_attempt{attempt}_terminal.json"
        record = _load(
            BENCHMARKS / f"m3b_ensemble_v3_attempt{attempt}_failure_record.json"
        )
        payload = _load(terminal)

        assert record["terminal_artifact"]["sha256"] == _sha256(terminal)
        assert record["terminal_artifact"]["bytes"] == terminal.stat().st_size
        assert payload["status"] == "failed"
        assert payload["phase"] == "quality"
        assert payload["rows"] is None
        assert payload["completed_rows_discarded"] == attempt - 1
        assert record["disposition"]["rerun_same_identity"] is False


def test_m3b_r3_published_artifacts_validate_and_cover_the_frozen_grids():
    quality = analyzer.validate_artifact(QUALITY, CONTRACT, phase="quality")
    gate = _load(GATE)
    timing = analyzer.validate_artifact(
        TIMING,
        CONTRACT,
        phase="timing",
        gate=gate,
        gate_path=GATE,
    )

    expected_quality = len(runner.case_specs()) * len(runner.ARMS)
    expected_timing = (
        len(runner.decision_rules()["timing_repeats"])
        * len(runner.case_specs())
        * (2 + len(gate["eligible_candidates"]))
    )
    assert len(quality["rows"]) == expected_quality == 65
    assert len(timing["rows"]) == expected_timing == 130
    assert gate["eligible_candidates"] == list(runner.CANDIDATE_ARMS)


def test_m3b_r3_result_hash_chain_and_frozen_disposition_regenerate_exactly():
    stored = _load(RESULT)
    regenerated = analyzer.build_final_result(QUALITY, GATE, TIMING, CONTRACT)

    assert stored["contract_sha256"] == _sha256(CONTRACT)
    assert stored["quality_artifact_sha256"] == _sha256(QUALITY)
    assert stored["gate_sha256"] == _sha256(GATE)
    assert stored["timing_artifact_sha256"] == _sha256(TIMING)
    regenerated["analyzed_at"] = stored["analyzed_at"]
    assert regenerated == stored

    rendered = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from pathlib import Path; "
                "from benchmarks import analyze_m3b_ensemble_v3_r3 as analyzer; "
                "path = Path('benchmarks/m3b_ensemble_v3_r3_result.json'); "
                "print(analyzer.render_note(json.loads(path.read_text())), end='')"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert rendered == NOTE.read_text(encoding="utf-8")

    assert stored["disposition"] == "close_b1_b2_preserve_existing_opt_in"
    assert stored["retained_private_arms"] == []
    assert all(
        record["quality"]["eligible"] and not record["survives"]
        for record in stored["candidates"].values()
    )
    assert stored["candidates"][runner.COMBINED]["checks"] == {
        "archive_to_control": True,
        "archive_to_single": False,
        "predict": True,
        "quality_eligible": True,
        "rss_to_control": True,
        "rss_to_single": True,
        "value": True,
    }
    assert stored["public_or_default_change_authorized"] is False
    assert stored["b3_authorized"] is False
    assert stored["fresh_confirmation_authorized"] is False
