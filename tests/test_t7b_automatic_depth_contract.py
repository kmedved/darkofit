from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import run_t7b_automatic_depth_v1 as runner


def test_contract_pins_one_mechanism_and_control():
    text = runner.CONTRACT_PATH.read_text()

    assert runner.CONTRACT_ID in text
    assert runner.CONTROL_HEAD in text
    assert "L2 v1 is already terminal and remains closed" in text
    assert "It does not authorize" in text and "TabArena" in text
    assert "depth 4 below 100" in text
    assert "depth 8 at or above" in text
    assert 'literal `depth="auto"` keeps its existing' in text
    assert runner.CANDIDATE_FILES == {
        "darkofit/booster.py",
        "tests/test_darkofit.py",
        "tests/test_t7b_automatic_depth_policy.py",
    }
    assert runner.HARNESS_FILES == {
        "benchmarks/check_t7b_automatic_depth_invariants.py",
        "benchmarks/run_t7b_automatic_depth_v1.py",
        "benchmarks/t7b_automatic_depth_development_contract.md",
        "tests/test_t7b_automatic_depth_contract.py",
        "tests/test_t7b_automatic_depth_invariants.py",
    }


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


def test_source_validation_binds_harness_and_candidate_diffs(
    tmp_path, monkeypatch
):
    control = (tmp_path / "control").resolve()
    candidate = (tmp_path / "candidate").resolve()
    harness_state = {
        "path": str(runner.ROOT),
        "head": "harness",
        "tree": "harness-tree",
        "clean": True,
        "status": [],
    }
    control_state = {
        "path": str(control),
        "head": runner.CONTROL_HEAD,
        "tree": "control-tree",
        "clean": True,
        "status": [],
    }
    candidate_state = {
        "path": str(candidate),
        "head": "candidate",
        "tree": "candidate-tree",
        "clean": True,
        "status": [],
    }

    def fake_source_state(repository):
        resolved = repository.resolve()
        return {
            runner.ROOT: harness_state,
            control: control_state,
            candidate: candidate_state,
        }[resolved]

    monkeypatch.setattr(runner, "source_state", fake_source_state)
    monkeypatch.setattr(runner, "_git", lambda *_args, **_kwargs: runner.CONTROL_HEAD)
    monkeypatch.setattr(runner, "_is_ancestor", lambda *_args: True)
    monkeypatch.setattr(
        runner,
        "_tracked_bytes",
        lambda *_args: runner.CONTRACT_PATH.read_bytes(),
    )

    def valid_changed_files(repository, **_kwargs):
        return (
            runner.HARNESS_FILES
            if repository.resolve() == runner.ROOT
            else runner.CANDIDATE_FILES
        )

    monkeypatch.setattr(runner, "_candidate_changed_files", valid_changed_files)
    sources = runner.validate_sources(control, candidate)
    assert sources["harness_changed_files"] == sorted(runner.HARNESS_FILES)
    assert sources["candidate_changed_files"] == sorted(runner.CANDIDATE_FILES)

    monkeypatch.setattr(
        runner,
        "_candidate_changed_files",
        lambda repository, **_kwargs: (
            runner.HARNESS_FILES | {"darkofit/booster.py"}
            if repository.resolve() == runner.ROOT
            else runner.CANDIDATE_FILES
        ),
    )
    with pytest.raises(RuntimeError, match="harness changed files"):
        runner.validate_sources(control, candidate)


def test_precondition_artifacts_bind_same_sources_and_passed_checks(tmp_path):
    state = {
        "path": str(tmp_path / "source"),
        "head": "head",
        "tree": "tree",
        "clean": True,
        "status": [],
    }
    sources = {
        "harness": state,
        "control": {**state, "head": runner.CONTROL_HEAD},
        "candidate": {**state, "head": "candidate"},
        "candidate_changed_files": sorted(runner.CANDIDATE_FILES),
    }
    invariant = tmp_path / "invariant.json"
    invariant.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": runner.INVARIANT_IDENTITY,
                "contract_id": runner.CONTRACT_ID,
                "quality_outcomes_inspected": False,
                "sources": sources,
                "bindings": {
                    "contract_sha256": runner.file_sha256(runner.CONTRACT_PATH),
                    "campaign_runner_sha256": runner.file_sha256(
                        runner.RUNNER_PATH
                    ),
                    "invariant_runner_sha256": runner.file_sha256(
                        runner.INVARIANT_RUNNER_PATH
                    ),
                },
                "analysis": {
                    "all_noop_cases_exact": True,
                    "all_depth_branches_engaged": True,
                },
            }
        )
    )
    m5 = tmp_path / "m5.json"
    m5.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runner_version": "m5-sentinels-v1",
                "contract": {
                    "m5": {
                        "contract_frozen": True,
                        "control_source": runner.M5_CONTROL_HEAD,
                    }
                },
                "evidence_status": "sentinel_check",
                "non_ranking": True,
                "shipping_or_default_claim_authorized": False,
                "runner_sha256": runner.file_sha256(runner.M5_RUNNER_PATH),
                "sources": {
                    "harness": state,
                    "candidate": {**state, "head": "candidate"},
                    "control": {
                        **state,
                        "head": runner.M5_CONTROL_HEAD,
                    },
                },
                "analysis": {
                    "behavior_fingerprints_equal_between_arms": True,
                    "baseline_drift": [],
                    "advancement_blocked_for_drift": False,
                    "known_floor_checks": {"floor": {"passed": True}},
                },
            }
        )
    )

    binding = runner.validate_preconditions(
        invariant, m5, sources=sources
    )
    assert set(binding) == {"invariant", "m5"}

    payload = json.loads(m5.read_text())
    payload["analysis"]["baseline_drift"] = ["drift"]
    m5.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="precondition"):
        runner.validate_preconditions(invariant, m5, sources=sources)


def test_invalid_precondition_does_not_spend_inspection(tmp_path, monkeypatch):
    paths = runner.output_paths(tmp_path / "t7b")
    monkeypatch.setattr(
        runner,
        "validate_sources",
        lambda *_: {"harness": {}, "control": {}, "candidate": {}},
    )
    monkeypatch.setattr(
        runner,
        "validate_preconditions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("invalid prerequisite")
        ),
    )

    with pytest.raises(RuntimeError, match="invalid prerequisite"):
        runner.run(
            SimpleNamespace(
                control=tmp_path / "control",
                candidate=tmp_path / "candidate",
                invariants=tmp_path / "invariant.json",
                m5_result=tmp_path / "m5.json",
                output_prefix=tmp_path / "t7b",
            )
        )
    assert not any(path.exists() for path in paths.values())


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
        "validate_preconditions",
        lambda *_args, **_kwargs: {"invariant": {}, "m5": {}},
    )
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
                invariants=tmp_path / "invariant.json",
                m5_result=tmp_path / "m5.json",
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
