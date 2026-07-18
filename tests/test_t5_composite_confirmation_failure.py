import json
import hashlib
import os
from pathlib import Path

import pytest

from benchmarks import record_t5_composite_confirmation_failure as failure
from benchmarks import run_t5_composite_confirmation as runner


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks" / "t5_composite_confirmation_failure.json"


def test_t5_failure_artifact_is_hash_bound_and_fail_closed():
    artifact = json.loads(ARTIFACT.read_text())
    assert (
        ROOT
        / "benchmarks"
        / "t5_composite_confirmation_failure.md"
    ).read_text() == failure._markdown(artifact)
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


def test_t5_failure_inventory_requires_exact_control_only_spools(tmp_path):
    expected = {101, 102}
    for task_id in expected:
        runner._spool_path(
            tmp_path, task_id, runner.CONTROL
        ).write_text("{}")
    paths = failure._completed_control_paths(tmp_path, expected)
    assert {task_id for task_id, _path in paths} == expected

    runner._spool_path(
        tmp_path, 101, runner.COMPOSITE
    ).write_text("{}")
    with pytest.raises(RuntimeError, match="unexpected"):
        failure._completed_control_paths(tmp_path, expected)


def test_t5_failure_binding_requires_frozen_registry_identity():
    binding = {
        "schema_version": 1,
        "runner_sha256": runner._sha256(Path(runner.__file__).resolve()),
        "protocol_sha256": runner._sha256(runner.PROTOCOL),
        "registry_file_sha256": runner.EXPECTED_REGISTRY_FILE_SHA256,
        "registry_canonical_sha256": (
            runner.EXPECTED_REGISTRY_CANONICAL_SHA256
        ),
        "darkofit_head": failure.EXPECTED_DARKOFIT_HEAD,
        "chimeraboost_head": runner.EXPECTED_CHIMERA_HEAD,
        "configs": list(runner.CONFIGS),
        "folds": list(runner.FOLDS),
    }
    failure._validate_completed_binding(binding)
    binding["folds"] = [False, 1, 2]
    with pytest.raises(RuntimeError, match="binding changed"):
        failure._validate_completed_binding(binding)
    binding["folds"] = list(runner.FOLDS)
    binding["registry_file_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="binding changed"):
        failure._validate_completed_binding(binding)


def test_t5_failure_spool_is_validated_from_one_byte_snapshot(tmp_path):
    binding = {
        "schema_version": 1,
        "runner_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
        "registry_file_sha256": "c" * 64,
        "registry_canonical_sha256": "d" * 64,
        "darkofit_head": "e" * 40,
        "chimeraboost_head": "f" * 40,
        "configs": list(runner.CONFIGS),
        "folds": list(runner.FOLDS),
    }
    result = {
        "task_id": 101,
        "config": runner.CONTROL,
        "folds": [],
    }
    path = runner._spool_path(tmp_path, 101, runner.CONTROL)
    runner._create_spool(
        path, binding, 101, runner.CONTROL, result
    )
    payload, loaded, spool_hash, file_sha256 = (
        failure._load_completed_control_spool(path, 101)
    )
    assert loaded == result
    assert spool_hash == payload["spool_record_sha256"]
    assert file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()

    payload["schema_version"] = True
    payload["spool_record_sha256"] = runner._json_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "spool_record_sha256"
        }
    )
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="spool binding"):
        failure._load_completed_control_spool(path, 101)


def test_t5_failure_spool_rejects_duplicate_json_keys(tmp_path):
    path = runner._spool_path(tmp_path, 101, runner.CONTROL)
    path.write_text('{"task_id":101,"task_id":102}')
    with pytest.raises(RuntimeError, match="invalid T5 completed-worker spool JSON"):
        failure._load_completed_control_spool(path, 101)


def test_t5_failure_json_requires_utf8_and_bounded_integers():
    with pytest.raises(RuntimeError, match="invalid T5 failure JSON"):
        failure._json_loads(
            '{"schema_version":1}'.encode("utf-16"),
            "T5 failure",
        )
    with pytest.raises(RuntimeError, match="invalid T5 failure JSON"):
        failure._json_loads(
            '{"task_id":9223372036854775808}',
            "T5 failure",
        )


def test_t5_failure_control_metadata_schema_is_exact():
    core = {
        "iterations_requested": 1_000,
        "iterations_attempted": 1_000,
        "rounds_completed": 1_000,
        "rounds_retained": 1_000,
        "stop_reason": "iteration_limit",
        "phase_seconds": {"tree_build": 0.01},
    }
    fit = {
        "best_iteration": 1_000,
        "fitted_tree_count": 1_000,
        "resolved_learning_rate": 0.1,
        "requested_tree_mode": "catboost",
        "selected_tree_mode": "catboost",
        "selected_lane": "boosting",
        "linear_residual_active": False,
        "linear_leaves_active": False,
        "linear_leaves": {
            "requested": False,
            "active": False,
            "inactive_reason": "disabled",
            "min_samples": 1_000,
            "linear_lambda": 1.0,
            "numeric_feature_count": 0,
            "linear_tree_count": 0,
            "linear_leaf_count": 0,
        },
        "resolved_thread_count": runner.THREADS_PER_WORKER,
        "refit": False,
        "refit_strategy": None,
        "final_fit": core,
        "selection_fit": None,
        "selection_early_stopping_rounds": None,
        "final_early_stopping_rounds": None,
    }
    metadata = {
        "kind": runner.CONTROL,
        "engaged": False,
        "selected_configuration": "product_default",
        "final_fit": fit,
    }
    assert failure._valid_control_metadata(metadata)
    metadata["final_fit"]["resolved_thread_count"] = 1
    assert not failure._valid_control_metadata(metadata)
    metadata["final_fit"]["resolved_thread_count"] = runner.THREADS_PER_WORKER
    metadata["final_fit"] = {"forged": True}
    assert not failure._valid_control_metadata(metadata)


def test_t5_failure_pair_publish_rolls_back_first_output(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "failure.json"
    markdown = tmp_path / "failure.md"
    original = failure._atomic_create

    def fail_markdown(path, value, **kwargs):
        if path == markdown:
            raise OSError("injected second-output failure")
        return original(path, value, **kwargs)

    monkeypatch.setattr(failure, "_atomic_create", fail_markdown)
    with pytest.raises(OSError, match="second-output"):
        failure._atomic_create_pair(
            artifact, b"{}\n", markdown, b"# failure\n"
        )
    assert not artifact.exists()
    assert not markdown.exists()


def test_t5_failure_cli_defaults_are_repository_anchored(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    args = failure.parse_args([])
    assert args.spool_directory == failure.DEFAULT_SPOOL_DIRECTORY
    assert args.output == failure.DEFAULT_OUTPUT
    assert args.markdown == failure.DEFAULT_MARKDOWN


def test_t5_failure_publish_rolls_back_if_temp_cleanup_fails(
    tmp_path, monkeypatch
):
    output = tmp_path / "failure.json"
    original = Path.unlink

    def fail_temporary(path, *args, **kwargs):
        if path.suffix == ".tmp":
            raise OSError("injected temp cleanup failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_temporary)
    with pytest.raises(OSError, match="temp cleanup"):
        failure._atomic_create(output, b"{}\n")
    assert not output.exists()


def test_t5_failure_publish_removes_only_new_directories_on_failure(
    tmp_path, monkeypatch
):
    created_root = tmp_path / "new" / "nested"
    output = created_root / "failure.json"

    def fail_publish(_source, _destination):
        raise OSError("injected publish failure")

    monkeypatch.setattr(failure.os, "link", fail_publish)
    with pytest.raises(OSError, match="publish failure"):
        failure._atomic_create(output, b"{}\n")
    assert not (tmp_path / "new").exists()
    assert tmp_path.exists()


def test_t5_failure_publish_rejects_substituted_temporary_inode(
    tmp_path, monkeypatch
):
    output = tmp_path / "failure.json"
    original = failure.os.link
    original_fdopen = failure.os.fdopen
    foreign_temporaries = []
    handles = []

    def track_fdopen(*args, **kwargs):
        handle = original_fdopen(*args, **kwargs)
        handles.append(handle)
        return handle

    def substitute_temporary(source, destination):
        source = Path(source)
        source.unlink()
        source.write_bytes(b"foreign\n")
        foreign_temporaries.append(source)
        original(source, destination)

    monkeypatch.setattr(failure.os, "fdopen", track_fdopen)
    monkeypatch.setattr(failure.os, "link", substitute_temporary)
    with pytest.raises(RuntimeError, match="publish identity changed"):
        failure._atomic_create(output, b"ours\n")
    assert output.read_bytes() == b"foreign\n"
    assert len(foreign_temporaries) == 1
    assert foreign_temporaries[0].read_bytes() == b"foreign\n"
    assert os.path.samefile(foreign_temporaries[0], output)
    assert handles and all(handle.closed for handle in handles)


def test_t5_failure_pair_rollback_preserves_replacement_inode(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "failure.json"
    markdown = tmp_path / "failure.md"
    original = failure._atomic_create
    original_fdopen = failure.os.fdopen
    handles = []

    def track_fdopen(*args, **kwargs):
        handle = original_fdopen(*args, **kwargs)
        handles.append(handle)
        return handle

    def replace_then_fail(path, value, **kwargs):
        if path == markdown:
            artifact.unlink()
            artifact.write_bytes(b"other writer\n")
            raise OSError("injected second-output failure")
        return original(path, value, **kwargs)

    monkeypatch.setattr(failure.os, "fdopen", track_fdopen)
    monkeypatch.setattr(failure, "_atomic_create", replace_then_fail)
    with pytest.raises(OSError, match="second-output"):
        failure._atomic_create_pair(
            artifact, b"ours\n", markdown, b"# failure\n"
        )
    assert artifact.read_bytes() == b"other writer\n"
    assert not markdown.exists()
    assert handles and all(handle.closed for handle in handles)


def test_t5_failure_registry_is_read_from_one_snapshot(monkeypatch):
    original = Path.read_bytes
    reads = 0

    def tracked(path):
        nonlocal reads
        if path == runner.REGISTRY:
            reads += 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    registry, rows = failure._registry_snapshot()
    assert registry["task_count"] == len(rows) == 25
    assert reads == 1


def test_t5_failure_dependency_binding_uses_one_snapshot(monkeypatch):
    original = Path.read_bytes
    expected = {
        Path(runner.__file__).resolve(): 0,
        runner.PROTOCOL: 0,
        runner.REGISTRY: 0,
    }

    def tracked(path):
        if path in expected:
            expected[path] += 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    snapshot = failure._dependency_snapshot()
    assert snapshot["runner_sha256"]
    assert snapshot["protocol_sha256"]
    assert expected == {path: 1 for path in expected}


def test_t5_failure_behavior_rejects_coerced_identity():
    result = {
        "task_id": "101",
        "config": runner.CONTROL,
        "folds": [],
        "behavior_fingerprint_sha256": "0" * 64,
    }
    with pytest.raises(RuntimeError, match="behavior is invalid"):
        failure._validate_behavior(result)


def test_t5_failure_output_rejects_nested_symlink_before_mkdir(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    output = alias / "nested" / "failure.json"
    with pytest.raises(
        RuntimeError, match="symlink T5 failure output directory"
    ):
        failure._atomic_create(output, b"{}\n")
    assert not (real / "nested").exists()
