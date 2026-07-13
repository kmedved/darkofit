import hashlib
import json
import os
import threading

import pandas as pd
import pytest

import benchmarks.remaining9_run_manifest as remaining9_provenance

from benchmarks.analyze_tabarena_regression_remaining9 import (
    FROZEN_COMPLETION_ATTESTATION_SHA256,
    FROZEN_RUN_MANIFEST_SHA256,
    analyze_rows,
    load_local_rows,
    normalized_chimera_rows_sha256,
    require_frozen_execution_digests,
)
from benchmarks.preflight_hotpaths import _speedup, _timing_summary
from benchmarks.remaining9_run_manifest import (
    SCHEMA_VERSION,
    frozen_protocol_identity,
    validate_completion_attestation,
    validate_manifest_payload,
)
from benchmarks.run_tabarena_regression_remaining9 import (
    EXPECTED_DATASET_SPLITS,
    EXPECTED_JOBS,
    FROZEN_CANDIDATE,
    TASK_SPLIT_COUNTS,
    validate_chimera_coverage,
)
def test_remaining9_frozen_matrix_and_registered_coverage():
    assert EXPECTED_DATASET_SPLITS == 165
    assert EXPECTED_JOBS == 330
    assert FROZEN_CANDIDATE == {
        "l2_leaf_reg": 1.0,
        "max_bins": 128,
        "learning_rate": 0.1,
        "ts_permutations": 1,
    }
    rows = []
    for dataset, (_, split_count) in TASK_SPLIT_COUNTS.items():
        rows.extend(
            {
                "dataset": dataset,
                "method": "CHIMERA (default)",
                "fold": fold,
                "imputed": False,
            }
            for fold in range(split_count)
        )
    validate_chimera_coverage(pd.DataFrame(rows))
    rows[-1]["imputed"] = True
    with pytest.raises(RuntimeError, match="imputed"):
        validate_chimera_coverage(pd.DataFrame(rows))


def test_preflight_timing_helpers():
    optimized = _timing_summary([1.0, 2.0, 3.0])
    reference = _timing_summary([2.0, 4.0, 6.0])
    assert optimized["median_seconds"] == 2.0
    assert optimized["iqr_seconds"] == 1.0
    assert optimized["iqr_fraction"] == 0.5
    assert _speedup(optimized, reference) == 2.0


def test_remaining9_analyzer_requires_trusted_execution_digests(tmp_path):
    require_frozen_execution_digests(
        FROZEN_RUN_MANIFEST_SHA256,
        FROZEN_COMPLETION_ATTESTATION_SHA256,
    )
    with pytest.raises(RuntimeError, match="manifest is not the trusted"):
        require_frozen_execution_digests(
            "0" * 64,
            FROZEN_COMPLETION_ATTESTATION_SHA256,
        )
    with pytest.raises(RuntimeError, match="attestation is not the trusted"):
        require_frozen_execution_digests(
            FROZEN_RUN_MANIFEST_SHA256,
            "0" * 64,
        )
    with pytest.raises(RuntimeError, match="without trusted attestation"):
        load_local_rows(tmp_path)


def test_remaining9_analysis_applies_equal_dataset_and_repeat_gates():
    tasks = {"small_a": (1, 9), "small_b": (2, 9)}
    local = []
    chimera = []
    for dataset, (task_id, split_count) in tasks.items():
        for registered_fold in range(split_count):
            repeat, fold = divmod(registered_fold, 3)
            chimera.append(
                {
                    "dataset": dataset,
                    "repeat": repeat,
                    "fold": fold,
                    "registered_fold": registered_fold,
                    "rmse": 1.1,
                    "val_rmse": 1.1,
                }
            )
            for config, rmse in (("default", 1.0), ("candidate", 0.99)):
                local.append(
                    {
                        "dataset": dataset,
                        "task_id": task_id,
                        "repeat": repeat,
                        "fold": fold,
                        "registered_fold": registered_fold,
                        "config": config,
                        "rmse": rmse,
                        "val_rmse": rmse,
                        "train_time_s": 1.0 if config == "default" else 0.9,
                        "infer_time_s": 1.0 if config == "default" else 0.9,
                        "peak_memory_bytes": 100.0,
                    }
                )
    tidy, summary = analyze_rows(local, chimera, task_split_counts=tasks)
    assert len(tidy) == 18
    assert summary["equal_dataset"]["candidate_default_rmse"]["ratio"] == pytest.approx(
        0.99
    )
    assert summary["gates"]["advance"] is True
    assert all(item["repeat_wins"] == 3 for item in summary["datasets"])
    assert summary["counts"]["expected_child_fits"] == 16 * 18
    digest = normalized_chimera_rows_sha256(chimera)
    assert normalized_chimera_rows_sha256(reversed(chimera)) == digest
    changed = [dict(row) for row in chimera]
    changed[0]["rmse"] += 0.01
    assert normalized_chimera_rows_sha256(changed) != digest


def test_remaining9_run_manifest_requires_exact_protocol_source_and_environment(
    tmp_path,
):
    repository = (tmp_path / "darkofit").resolve()
    experiments = (
        repository
        / ".cache/tabarena-regression-remaining9-0.9.0-20260712/experiments"
    ).resolve()
    repository.mkdir()
    experiments.mkdir(parents=True)
    runner_path = repository / "benchmarks/run_tabarena_regression_remaining9.py"
    identity = {
        "runner": {
            "path": str(runner_path),
            "sha256": "runner-sha",
            "git_blob": "blob",
            "mtime_utc": "2026-07-12T23:59:00+00:00",
        },
        "adapter": {
            "path": str(repository / "benchmarks/tabarena_adapter.py"),
            "sha256": "adapter-sha",
            "git_blob": "adapter-blob",
            "mtime_utc": "2026-07-12T23:58:00+00:00",
        },
        "darkofit": {
            "repository_path": str(repository),
            "repository_head_at_capture": "head-after-benchmark-only-change",
            "declared_source_commit": "224bd46-source-commit",
            "library_tree": "library-tree",
            "library_status": "",
        },
        "tabarena": {
            "repository_path": "/tabarena",
            "repository_head": "tabarena-head",
            "repository_status": "",
            "module_path": "/tabarena/tabarena/__init__.py",
        },
        "python": {
            "executable": "/venv/bin/python",
            "resolved_executable": "/venv/bin/python",
            "prefix": "/venv",
            "base_prefix": "/base",
            "version": "3.12.0",
            "implementation": "CPython",
            "platform": "test-platform",
            "machine": "test-machine",
        },
        "packages": {"darkofit": "0.9.0", "tabarena": "0.0.1"},
        "environment": {"NUMBA_CACHE_DIR": "/cache", "PYTHONPATH": "."},
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "darkofit_remaining9_in_flight_run_manifest",
        "captured_at_utc": "2026-07-13T00:00:02+00:00",
        "output_dir": str(experiments.parent),
        "experiments_dir": str(experiments),
        "protocol": frozen_protocol_identity(),
        "process": {
            "pid": 123,
            "command": (
                "/venv/bin/python "
                "benchmarks/run_tabarena_regression_remaining9.py"
            ),
            "cwd": str(repository),
            "environment": identity["environment"],
            "started_utc": "2026-07-13T00:00:00+00:00",
            "running_at_capture": True,
        },
        "result_snapshot": {
            "completed_result_files_at_capture": 100,
            "earliest_result_mtime_utc": "2026-07-13T00:00:01+00:00",
            "latest_result_mtime_utc": "2026-07-13T00:00:01+00:00",
        },
        **identity,
    }

    validate_manifest_payload(
        manifest,
        input_dir=experiments,
        current_identity=identity,
    )

    empty_snapshot = {
        **manifest,
        "result_snapshot": {
            "completed_result_files_at_capture": 0,
            "earliest_result_mtime_utc": None,
            "latest_result_mtime_utc": None,
        },
    }
    validate_manifest_payload(
        empty_snapshot,
        input_dir=experiments,
        current_identity=identity,
    )
    invalid_empty_snapshot = {
        **empty_snapshot,
        "result_snapshot": {
            **empty_snapshot["result_snapshot"],
            "earliest_result_mtime_utc": "2026-07-13T00:00:01+00:00",
        },
    }
    with pytest.raises(RuntimeError, match="empty run manifest snapshot"):
        validate_manifest_payload(
            invalid_empty_snapshot,
            input_dir=experiments,
            current_identity=identity,
        )

    wrong_runner = {**manifest, "runner": {**manifest["runner"], "sha256": "wrong"}}
    with pytest.raises(RuntimeError, match="runner"):
        validate_manifest_payload(
            wrong_runner,
            input_dir=experiments,
            current_identity=identity,
        )

    wrong_tree = {
        **manifest,
        "darkofit": {**manifest["darkofit"], "library_tree": "wrong"},
    }
    with pytest.raises(RuntimeError, match="library_tree"):
        validate_manifest_payload(
            wrong_tree,
            input_dir=experiments,
            current_identity=identity,
        )


def test_remaining9_completion_attestation_binds_every_final_file(tmp_path):
    experiments = tmp_path / "experiments"
    observed = {}
    for index in range(EXPECTED_JOBS):
        path = experiments / str(index) / "results.pkl"
        path.parent.mkdir(parents=True)
        payload = f"result-{index}".encode()
        path.write_bytes(payload)
        stat = path.stat()
        observed[path.relative_to(experiments).as_posix()] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "first_stable_seen_utc": "2099-07-13T00:00:02+00:00",
            "runner_pid_alive": True,
        }

    manifest_sha256 = "a" * 64
    manifest = {
        "captured_at_utc": "2026-07-13T00:00:00+00:00",
        "process": {
            "pid": 123,
            "started_utc": "2026-07-12T23:59:59+00:00",
        },
    }
    attestation = {
        "schema_version": 1,
        "kind": "remaining9_live_completion_attestation",
        "watch_started_utc": "2026-07-13T00:00:01+00:00",
        "completed_utc": "2099-07-13T00:00:03+00:00",
        "runner_pid": 123,
        "runner_pids_at_completion": [123],
        "runner_alive_at_completion": True,
        "expected_results": EXPECTED_JOBS,
        "observed_results": observed,
        "run_manifest_sha256": manifest_sha256,
    }

    verified_payloads = validate_completion_attestation(
        attestation,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        input_dir=experiments,
    )
    assert len(verified_payloads) == EXPECTED_JOBS
    assert verified_payloads["0/results.pkl"] == b"result-0"

    wrong_manifest = {**attestation, "run_manifest_sha256": "b" * 64}
    with pytest.raises(RuntimeError, match="run-manifest digest"):
        validate_completion_attestation(
            wrong_manifest,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            input_dir=experiments,
        )

    wrong_pid = {**attestation, "runner_pids_at_completion": [123, 456]}
    with pytest.raises(RuntimeError, match="sole frozen runner PID"):
        validate_completion_attestation(
            wrong_pid,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            input_dir=experiments,
        )

    bad_hashes = {name: dict(item) for name, item in observed.items()}
    bad_hashes["0/results.pkl"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="does not match final result"):
        validate_completion_attestation(
            {**attestation, "observed_results": bad_hashes},
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            input_dir=experiments,
        )

    wrong_chronology = {
        **attestation,
        "watch_started_utc": "2026-07-12T23:59:59+00:00",
    }
    with pytest.raises(RuntimeError, match="after manifest capture"):
        validate_completion_attestation(
            wrong_chronology,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            input_dir=experiments,
        )

    stale_path = experiments / "0/results.pkl"
    original_mtime_ns = stale_path.stat().st_mtime_ns
    os.utime(stale_path, ns=(1, 1))
    stale_results = {name: dict(item) for name, item in observed.items()}
    stale_results["0/results.pkl"]["mtime_ns"] = 1
    with pytest.raises(RuntimeError, match="predates runner"):
        validate_completion_attestation(
            {**attestation, "observed_results": stale_results},
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            input_dir=experiments,
        )
    os.utime(stale_path, ns=(original_mtime_ns, original_mtime_ns))


@pytest.mark.parametrize(
    "command",
    [
        "/venv/bin/python benchmarks/run_tabarena_regression_remaining9.py",
        (
            "/venv/bin/python benchmarks/run_tabarena_regression_remaining9.py "
            "--output-dir /tmp/results --time-limit 10"
        ),
        (
            "/venv/bin/python -m benchmarks.run_tabarena_regression_remaining9 "
            "--output-dir /tmp/results"
        ),
        (
            "/bin/zsh -c 'PYTHONPATH=. /venv/bin/python "
            "benchmarks/run_tabarena_regression_remaining9.py --time-limit 10'"
        ),
    ],
)
def test_remaining9_runner_detector_covers_flagged_and_wrapped_commands(command):
    assert remaining9_provenance._command_invokes_remaining9_runner(command)


def test_remaining9_runner_detector_rejects_non_runner_commands():
    assert not remaining9_provenance._command_invokes_remaining9_runner(
        "rg run_tabarena_regression_remaining9.py"
    )
    assert not remaining9_provenance._command_invokes_remaining9_runner(
        "/venv/bin/python benchmarks/other_benchmark.py"
    )


def test_remaining9_versioned_watcher_emits_verifiable_attestation(
    tmp_path,
    monkeypatch,
):
    experiments = tmp_path / "experiments"

    process = {
        "pid": 123,
        "started_utc": "2000-01-01T00:00:00+00:00",
        "command": "python benchmarks/run_tabarena_regression_remaining9.py",
        "cwd": str(tmp_path),
        "environment": {},
    }
    manifest = {
        "captured_at_utc": "2000-01-01T00:00:01+00:00",
        "experiments_dir": str(experiments),
        "process": process,
        "result_snapshot": {"completed_result_files_at_capture": 0},
    }
    manifest_path = tmp_path / "run_manifest.json"
    manifest_payload = json.dumps(manifest, sort_keys=True).encode()
    manifest_path.write_bytes(manifest_payload)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    monkeypatch.setattr(
        remaining9_provenance,
        "load_and_verify_manifest",
        lambda path, input_dir: (manifest, manifest_sha256),
    )
    monkeypatch.setattr(
        remaining9_provenance,
        "_matching_runner_pids",
        lambda: [123],
    )
    writer = None

    def write_results():
        for index in range(EXPECTED_JOBS):
            path = experiments / str(index) / "results.pkl"
            path.parent.mkdir(parents=True)
            path.write_bytes(f"result-{index}".encode())

    def process_snapshot(pid):
        nonlocal writer
        if writer is None:
            writer = threading.Thread(target=write_results)
            writer.start()
        return process

    monkeypatch.setattr(remaining9_provenance, "_process_snapshot", process_snapshot)

    attestation_path = tmp_path / "completion_attestation.live.json"
    attestation = remaining9_provenance.watch_completion(
        manifest_path=manifest_path,
        attestation_path=attestation_path,
        pid=123,
        poll_interval=0.001,
        timeout=2.0,
    )
    writer.join()
    assert attestation_path.is_file()
    assert len(attestation["observed_results"]) == EXPECTED_JOBS
    verified = validate_completion_attestation(
        attestation,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        input_dir=experiments,
    )
    assert len(verified) == EXPECTED_JOBS


def test_remaining9_versioned_watcher_rejects_late_start(tmp_path, monkeypatch):
    experiments = tmp_path / "experiments"
    manifest = {
        "captured_at_utc": "2000-01-01T00:00:01+00:00",
        "experiments_dir": str(experiments),
        "process": {"pid": 123},
        "result_snapshot": {"completed_result_files_at_capture": 1},
    }
    manifest_path = tmp_path / "run_manifest.json"
    payload = json.dumps(manifest, sort_keys=True).encode()
    manifest_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        remaining9_provenance,
        "load_and_verify_manifest",
        lambda path, input_dir: (manifest, digest),
    )

    with pytest.raises(RuntimeError, match="zero-result manifest"):
        remaining9_provenance.watch_completion(
            manifest_path=manifest_path,
            attestation_path=tmp_path / "attestation.json",
            pid=123,
        )
