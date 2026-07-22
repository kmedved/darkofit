from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import run_t7b_automatic_l2_v1 as runner


def test_contract_pins_one_mechanism_and_control():
    text = runner.CONTRACT_PATH.read_text()

    assert runner.CONTRACT_ID in text
    assert runner.CONTROL_HEAD in text
    assert "samples-per-feature depth idea is a separate mechanism" in text
    assert "No new TabArena coordinate" in text


def test_output_paths_are_external_and_create_only_names(tmp_path):
    paths = runner.output_paths(tmp_path / "t7b")

    assert set(paths) == {
        "launch_manifest",
        "raw",
        "result",
        "m6_manifest",
        "terminal_attestation",
    }
    assert paths["result"].name == "t7b_result.json"
    assert paths["m6_manifest"].name == "t7b_result.json.manifest.json"
    with pytest.raises(ValueError, match="outside"):
        runner.output_paths(runner.ROOT / "inside")


def test_m6_command_is_exact(tmp_path):
    paths = runner.output_paths(tmp_path / "t7b")
    command = runner.m6_command(
        control=tmp_path / "control",
        candidate=tmp_path / "candidate",
        paths=paths,
    )

    assert command[1] == str(runner.M6_RUNNER_PATH)
    assert command[command.index("--mechanism-id") + 1] == runner.MECHANISM_ID
    assert command[command.index("--inspection-index") + 1] == "1"
    assert command[command.index("--raw-csv") + 1] == str(paths["raw"])
    assert command[command.index("--output") + 1] == str(paths["result"])


def test_launch_manifest_precedes_quality_subprocess(tmp_path, monkeypatch):
    paths = runner.output_paths(tmp_path / "t7b")
    source_payload = {
        "harness": {"head": "h"},
        "control": {"head": runner.CONTROL_HEAD},
        "candidate": {"head": "c"},
        "candidate_changed_files": sorted(runner.CANDIDATE_FILES),
    }
    monkeypatch.setattr(runner, "validate_sources", lambda *_: source_payload)
    monkeypatch.setattr(
        runner,
        "_exclusive_machine_audit",
        lambda: {"conflicting_benchmark_processes": []},
    )
    monkeypatch.setattr(runner.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(runner.platform, "machine", lambda: "test-machine")

    def fail_after_manifest(command, **kwargs):
        assert paths["launch_manifest"].is_file()
        payload = json.loads(paths["launch_manifest"].read_text())
        assert payload["inspection_spent_on_manifest_creation"] is True
        assert payload["rerun_authorized"] is False
        raise subprocess_error

    subprocess_error = RuntimeError("stop before quality")
    monkeypatch.setattr(runner.subprocess, "run", fail_after_manifest)

    with pytest.raises(RuntimeError, match="stop before quality"):
        runner.run(
            SimpleNamespace(
                control=tmp_path / "control",
                candidate=tmp_path / "candidate",
                output_prefix=tmp_path / "t7b",
            )
        )
    assert paths["launch_manifest"].is_file()
    assert not paths["terminal_attestation"].exists()


def test_terminal_validation_binds_all_generic_artifacts(tmp_path):
    paths = runner.output_paths(tmp_path / "t7b")
    sources = {
        "control": {"head": runner.CONTROL_HEAD},
        "candidate": {"head": "candidate"},
    }
    paths["raw"].write_text("raw\n")
    paths["result"].write_text(
        json.dumps(
            {
                "contract_id": "m6-quality-successor-v3",
                "mechanism_id": runner.MECHANISM_ID,
                "inspection_index": 1,
                "candidate_ranking_eligible": True,
                "shipping_or_default_claim_eligible": False,
                "analysis": {"disposition": "advance"},
            }
        )
    )
    paths["m6_manifest"].write_text(
        json.dumps(
            {
                "contract_id": "m6-quality-successor-v3",
                "mechanism_id": runner.MECHANISM_ID,
                "inspection_index": 1,
                "inspection_spent": True,
                "sources_before_and_after": {
                    "control_default": sources["control"],
                    "candidate_default": sources["candidate"],
                },
                "raw_csv": {"sha256": runner.file_sha256(paths["raw"])},
            }
        )
    )

    terminal = runner._validate_terminal(paths, sources=sources)

    assert terminal["disposition"] == "advance"
    paths["m6_manifest"].write_text("{}")
    with pytest.raises(RuntimeError, match="invalid"):
        runner._validate_terminal(paths, sources=sources)
