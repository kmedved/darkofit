"""Hostile unit tests for the executable CTR23 confirmation runner."""

from __future__ import annotations

import hashlib
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest

from benchmarks import run_ctr23_minimal_confirmation as campaign


def _pinned_campaign_stack_available() -> bool:
    try:
        campaign.DEFAULT_CHIMERABOOST_PATH.resolve(strict=True)
        campaign.collect_runtime_provenance()
    except (ImportError, OSError, RuntimeError):
        return False
    return True


requires_pinned_campaign_stack = pytest.mark.skipif(
    not _pinned_campaign_stack_available(),
    reason="the frozen CTR23 benchmark stack and host are unavailable",
)


def _install_raw_path_tripwire(monkeypatch) -> list[Path]:
    touched: list[Path] = []
    for method_name in (
        "resolve",
        "stat",
        "lstat",
        "exists",
        "is_file",
        "open",
        "read_bytes",
        "iterdir",
        "glob",
        "rglob",
    ):
        original = getattr(Path, method_name)

        def guarded(
            path,
            *args,
            _original=original,
            _method_name=method_name,
            **kwargs,
        ):
            folded = {part.casefold() for part in path.parts}
            if "experiments" in folded or "results.pkl" in folded:
                touched.append(path)
                raise AssertionError(f"raw path {_method_name} touched {path}")
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(Path, method_name, guarded)
    return touched


def test_source_freeze_rejects_hostile_artifact_key_before_file_access(
    monkeypatch,
):
    document = deepcopy(campaign._coordinate_document())
    document["source_artifacts"]["../experiments/results.pkl"] = {
        "file_sha256": "0" * 64,
    }
    monkeypatch.setattr(
        campaign,
        "_coordinate_document",
        lambda: deepcopy(document),
    )

    touched: list[Path] = []

    def forbidden_file_access(path, *_args, **_kwargs):
        touched.append(Path(path))
        raise AssertionError(f"source registry touched {path}")

    monkeypatch.setattr(campaign, "_sha256_file", forbidden_file_access)
    monkeypatch.setattr(campaign, "_read_json_regular", forbidden_file_access)
    for method_name in (
        "resolve",
        "stat",
        "lstat",
        "exists",
        "is_file",
        "open",
        "read_bytes",
        "iterdir",
        "glob",
        "rglob",
    ):
        original = getattr(Path, method_name)

        def guarded_path_access(
            path,
            *args,
            _original=original,
            _method_name=method_name,
            **kwargs,
        ):
            if "experiments" in path.parts or path.name == "results.pkl":
                touched.append(path)
                raise AssertionError(
                    f"source registry {_method_name} touched {path}"
                )
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(Path, method_name, guarded_path_access)

    with pytest.raises(RuntimeError, match="source artifact registry changed"):
        campaign.validate_source_freeze()
    assert touched == []


def test_manifest_validator_rejects_hostile_output_before_path_access(
    tmp_path, monkeypatch,
):
    root = (tmp_path / "campaign").resolve()
    manifest = campaign.build_run_manifest(
        output_dir=root,
        execution_mode="concurrent",
        source_freeze={},
        source={},
        runtime={},
        sequential_recovery=None,
    )
    manifest["output_dir"] = str(
        root / "ExPeRiMeNtS" / "hostile" / "ReSuLtS.PkL"
    )
    touched = _install_raw_path_tripwire(monkeypatch)

    with pytest.raises(RuntimeError, match="run manifest"):
        campaign._validate_manifest_static(
            manifest, output_dir=root, execution_mode="concurrent"
        )
    assert touched == []


def test_campaign_namespace_git_ignore_gate_is_case_insensitive(
    tmp_path, monkeypatch,
):
    repository = tmp_path / "Repository"
    repository.mkdir()
    monkeypatch.setattr(campaign, "REPOSITORY_ROOT", repository)
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    variant = repository.with_name(repository.name.swapcase()) / ".cache" / "run"

    campaign._validate_campaign_namespace(variant, field="fixture")

    assert len(calls) == 1
    assert calls[0][0][-1] == str(Path(".cache/run"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("expected_jobs", True),
        ("expected_child_fits", 720.0),
        ("time_limit_seconds", 3_600),
        ("resolved_child_num_cpus", 18.0),
    ],
)
def test_manifest_validator_rejects_numeric_type_coercions(
    tmp_path, field, value,
):
    root = tmp_path.resolve()
    manifest = campaign.build_run_manifest(
        output_dir=root,
        execution_mode="concurrent",
        source_freeze={},
        source={},
        runtime={},
        sequential_recovery=None,
    )
    manifest[field] = value

    with pytest.raises(RuntimeError, match="run manifest"):
        campaign._validate_manifest_static(
            manifest, output_dir=root, execution_mode="concurrent"
        )


def _candidate_selection(scores=(0.0, 1.0, 2.0), selected=0):
    modes = ("catboost", "lightgbm", "hybrid")
    return {
        "candidate_count": 3,
        "fitted_candidate_count": 3,
        "selected_candidate_index": selected,
        "candidates": [
            {
                "tree_mode": mode,
                "fit_status": "fitted",
                "validation_score": score,
                "deadline_hit": False,
                "stop_reason": "early_stopping",
            }
            for mode, score in zip(modes, scores)
        ],
    }


def _comparator_fit(arm: str) -> dict:
    if arm == "D":
        return {
            "iterations_requested": 1_000,
            "iterations_attempted": 40,
            "rounds_completed": 40,
            "rounds_retained": 35,
            "best_iteration": 35,
            "resolved_learning_rate": 0.08,
            "requested_tree_mode": "catboost",
            "selected_tree_mode": "catboost",
            "selected_lane": "boosting",
            "deadline_hit": False,
            "deadline_is_soft": True,
            "stop_reason": "early_stopping",
        }
    requested = 10_000
    return {
        "iterations_requested": requested,
        "iterations_attempted": 100,
        "rounds_retained": 90,
        "best_iteration": 90,
        "resolved_learning_rate": 0.1 if arm == "M" else 0.05,
        "selected_lane": "constant" if arm == "M" else None,
        "linear_selection_performed": False,
        "stop_reason": None,
    }


def _callback_audit(engine: str, *, instances: int = 1, calls: int = 10):
    return {
        "schema_version": 1,
        "kind": "darkofit_ctr23_time_callback_audit",
        "engine": engine,
        "time_limit_seconds": 100.0,
        "time_callback_instrumented": True,
        "time_callback_instance_count": instances,
        "time_callback_call_count": calls,
        "time_callback_hit": False,
    }


def _swap_session(samples: list[tuple[int, int, int]]) -> dict:
    records = [
        {
            "monotonic_ns": monotonic_ns,
            "swap_in_bytes": swap_in,
            "swap_out_bytes": swap_out,
        }
        for monotonic_ns, swap_in, swap_out in samples
    ]
    return {
        "sample_count": len(records),
        "samples": records,
        "swap_in_delta": records[-1]["swap_in_bytes"] - records[0]["swap_in_bytes"],
        "swap_out_delta": records[-1]["swap_out_bytes"] - records[0]["swap_out_bytes"],
    }


def _failure_swap_telemetry(
    *, teardown_confirmed: bool = True,
) -> dict:
    session = _swap_session([(100, 10, 20), (200, 15, 20)])
    return {
        "capture_status": "captured",
        "teardown_confirmed": teardown_confirmed,
        "post_teardown_sample_recorded": True,
        "worker_session_swap_telemetry": session,
        "swap_in_bytes": 5,
        "swap_out_bytes": 0,
        "diagnostic": None if teardown_confirmed else "worker teardown failed",
    }


def _dispatch_swap_telemetry(
    *,
    release: int = 1_000_000_000,
    first_ns: int = 950_000_000,
    last_ns: int = 1_100_000_000,
    first_swap_in: int = 112,
    last_swap_in: int = 120,
    swap_out: int = 200,
) -> dict:
    return {
        "sample_count": 2,
        "samples": [
            {
                "monotonic_ns": first_ns,
                "swap_in_bytes": first_swap_in,
                "swap_out_bytes": swap_out,
            },
            {
                "monotonic_ns": last_ns,
                "swap_in_bytes": last_swap_in,
                "swap_out_bytes": swap_out,
            },
        ],
        "swap_in_delta": last_swap_in - first_swap_in,
        "swap_out_delta": 0,
        "barrier_release_monotonic_ns": release,
    }


def _persisted_wave_fixture(tmp_path: Path, execution_mode: str):
    expected_wave = deepcopy(campaign.expected_schedule()[0])
    ready_by_slot = {0: {"pid": 10_000}, 1: {"pid": 10_001}}
    reports = []
    artifacts = {}
    prior_end = None
    shared_release = 1_000_000_000
    for order, job in enumerate(expected_wave["jobs"]):
        slot = job["worker_slot"]
        key = campaign._key_tuple(job["key"])
        relative = campaign.expected_result_relative_path(*key)
        digest = hashlib.sha256(relative.encode()).hexdigest()
        size = 1_000 + slot
        artifacts[relative] = {"sha256": digest, "size_bytes": size}
        release = (
            shared_release
            if execution_mode == "concurrent"
            else (shared_release if prior_end is None else prior_end + 100)
        )
        started = release + 100 + (slot * 100 if execution_mode == "concurrent" else 0)
        ended = started + 5_000 + slot * 100
        if execution_mode == "concurrent":
            command_id = f"production-wave-0-{slot}-{1_000 + slot}"
        else:
            command_id = f"recovery-wave-0-job-{order}-{slot}-{1_000 + slot}"
        reports.append(
            {
                "type": "result",
                "command_id": command_id,
                "status": "ok",
                "slot": slot,
                "pid": ready_by_slot[slot]["pid"],
                "key": deepcopy(job["key"]),
                "result_root": str(tmp_path.resolve()),
                "result_path": str((tmp_path / relative).resolve()),
                "result_count": 1,
                "child_count": 8,
                "deadline_hit": False,
                "time_callback_hit_count": 0,
                "a10_candidate_fit_count": 24 if key[-1] == "A10" else 0,
                "behavior_sha256": "a" * 64,
                "result_sha256": digest,
                "result_size_bytes": size,
                "process_peak_rss_bytes": 10_000_000 + slot,
                "barrier_release_monotonic_ns": release,
                "started_monotonic_ns": started,
                "ended_monotonic_ns": ended,
                "start_method": "spawn",
            }
        )
        prior_end = ended
    starts = [report["started_monotonic_ns"] for report in reports]
    ends = [report["ended_monotonic_ns"] for report in reports]
    entry = {
        "wave_index": 0,
        "jobs": deepcopy(expected_wave["jobs"]),
        "reports": reports,
        "swap_start_sample_index": 2,
        "swap_end_sample_index": 3 if execution_mode == "concurrent" else 4,
        "swap_in_delta": 5 if execution_mode == "concurrent" else 10,
        "swap_out_delta": 0,
        "peak_combined_rss_fraction": 0.1,
        "start_skew_ns": max(starts) - min(starts),
        "overlap_ns": max(0, min(ends) - max(starts)),
        "wave_elapsed_ns": max(ends) - min(starts),
    }
    return entry, expected_wave, ready_by_slot, artifacts


def _refresh_persisted_wave_timing(entry: dict, execution_mode: str) -> None:
    starts = [report["started_monotonic_ns"] for report in entry["reports"]]
    ends = [report["ended_monotonic_ns"] for report in entry["reports"]]
    entry["start_skew_ns"] = max(starts) - min(starts)
    entry["overlap_ns"] = max(0, min(ends) - max(starts))
    entry["wave_elapsed_ns"] = max(ends) - min(starts)


def _validate_persisted_wave_fixture(
    tmp_path: Path,
    execution_mode: str,
    entry: dict,
    expected_wave: dict,
    ready_by_slot: dict,
    artifacts: dict,
) -> None:
    campaign._validate_persisted_wave(
        entry,
        expected_wave=expected_wave,
        execution_mode=execution_mode,
        ready_by_slot=ready_by_slot,
        result_artifacts=artifacts,
        output_dir=tmp_path,
        seen_command_ids=set(),
    )


def _persisted_preflight_fixture(tmp_path: Path) -> dict:
    ready = []
    warmup = []
    probes = []
    release = 1_000_000_000
    for slot in range(campaign.WORKER_COUNT):
        pid = 20_000 + slot
        payload = {
            "darkofit": {"engine": "darkofit", "slot": slot},
            "comparators": {"engine": "comparators", "slot": slot},
        }
        ready.append(
            {
                "type": "ready",
                "slot": slot,
                "pid": pid,
                "child_cpus": campaign.EXPECTED_CHILD_CPUS,
                "start_method": "spawn",
                "scratch_root": str(tmp_path / f"preflight-worker-{slot}"),
            }
        )
        warmup.append(
            {
                "completed_at_utc": "2026-07-16T00:00:00+00:00",
                "pid": pid,
                "worker_slot": slot,
                "warmup": payload,
            }
        )
        digest = hashlib.sha256(
            campaign._canonical_json(
                campaign._synthetic_behavior_projection(payload)
            )
        ).hexdigest()
        probes.append(
            {
                "worker_slot": slot,
                "pid": pid,
                "behavior_sha256": digest,
                "barrier_release_monotonic_ns": release,
                "started_monotonic_ns": release + 100 + slot * 100,
                "ended_monotonic_ns": release + 5_000 + slot * 100,
            }
        )
    starts = [probe["started_monotonic_ns"] for probe in probes]
    ends = [probe["ended_monotonic_ns"] for probe in probes]
    session_swap = _swap_session(
        [
            (100, 100, 200),
            (200, 105, 200),
            (300, 105, 200),
            (900_000_000, 110, 200),
            (1_200_000_000, 125, 200),
            (1_300_000_000, 130, 200),
        ]
    )
    dispatch = _dispatch_swap_telemetry(release=release)
    measured = {
        "start_sample_index": 3,
        "end_sample_index": 4,
        "sample_count": 2,
        "swap_in_delta": 15,
        "swap_out_delta": 0,
        "dispatches": [
            {
                "label": "preflight-synthetic-probe",
                "sample_index": 4,
                "resource_first_monotonic_ns": 950_000_000,
                "resource_last_monotonic_ns": 1_100_000_000,
                "resource_first_swap_in_bytes": 112,
                "resource_last_swap_in_bytes": 120,
                "resource_first_swap_out_bytes": 200,
                "resource_last_swap_out_bytes": 200,
                "barrier_release_monotonic_ns": release,
                "max_report_end_monotonic_ns": max(ends),
            }
        ],
    }
    return {
        "schema_version": campaign.HARNESS_SCHEMA_VERSION,
        "kind": campaign.CAMPAIGN_KIND + "_preflight",
        "completed_at_utc": "2026-07-16T00:00:01+00:00",
        "status": "passed",
        "swap_policy": campaign.SWAP_POLICY,
        "timing_admissible": False,
        "worker_ready": ready,
        "worker_warmup": warmup,
        "ctr23_fit_count": 0,
        "synthetic_probes": probes,
        "start_skew_ns": max(starts) - min(starts),
        "overlap_ns": max(0, min(ends) - max(starts)),
        "worker_restarts": False,
        "failure_count": 0,
        "worker_session_swap_telemetry": session_swap,
        "measured_phase_swap_window": measured,
        "synthetic_dispatch_telemetry": dispatch,
        "swap_in_bytes": 30,
        "swap_out_bytes": 0,
        "peak_combined_rss_fraction": 0.1,
    }


def _stub_persisted_warmup_validators(monkeypatch) -> None:
    monkeypatch.setattr(
        campaign.hardened.screen,
        "_validate_followon_warmup_history",
        lambda *args, **kwargs: None,
    )
    module = ModuleType("benchmarks.tabarena_comparator_warmup")
    module.validate_comparator_warmup_history = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_grid_schedule_and_child_counts_are_exact():
    assert len(campaign.expected_grid()) == 90
    assert len(campaign.expected_child_grid()) == 720
    assert len(campaign.expected_schedule()) == 45
    assert campaign.COORDINATE_MANIFEST_SHA256 == (
        "6cef3b771c20440c9dad6b737797f50650d84217ee99cf8fc6fcfcbd85829c0b"
    )
    assert campaign.schedule_sha256() == (
        "0285ca4242bd544f368578519f52f1a7157c1aa0f5c1c0ddcec9fcff5722055e"
    )


def test_quality_only_policy_accepts_retained_positive_swap_in():
    dispatch = _dispatch_swap_telemetry()
    session = _swap_session([(100, 10, 20), (200, 35, 20)])

    assert campaign._validated_dispatch_swap_deltas(dispatch, "dispatch") == (8, 0)
    assert campaign._validated_worker_session_swap_telemetry(
        session, "session"
    ) == session


def test_failed_worker_session_retains_post_teardown_swap_snapshot(
    tmp_path, monkeypatch,
):
    session = _swap_session([(100, 10, 20)])
    root_error = RuntimeError("root workload failure")
    monkeypatch.setattr(campaign, "_stop_workers", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        campaign.hardened,
        "_swap_counter_sample",
        lambda: {
            "monotonic_ns": 200,
            "swap_in_bytes": 15,
            "swap_out_bytes": 20,
        },
    )

    campaign._finalize_worker_session([], session, active_error=root_error)

    retained = getattr(root_error, campaign._FAILURE_SWAP_TELEMETRY_ATTRIBUTE)
    assert retained == _failure_swap_telemetry()
    (tmp_path / campaign.MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    campaign._write_invalid_attempt(
        tmp_path, execution_mode="concurrent", error=root_error
    )
    marker = campaign._read_json_regular(
        tmp_path / campaign.INVALID_ATTEMPT_FILENAME, "invalid marker"
    )
    assert marker["schema_version"] == campaign.HARNESS_SCHEMA_VERSION
    assert marker["worker_shutdown_confirmed"] is True
    assert marker["failure_swap_telemetry"] == retained
    assert (
        marker["recovery_policy"]
        == "fresh_sequential_namespace_from_wave_zero_only"
    )


def test_failed_worker_teardown_does_not_mask_root_error_or_authorize_recovery(
    tmp_path, monkeypatch,
):
    session = _swap_session([(100, 10, 20)])
    root_error = RuntimeError("root workload failure")

    def fail_stop(*_args, **_kwargs):
        raise RuntimeError("worker stop failed")

    monkeypatch.setattr(campaign, "_stop_workers", fail_stop)
    monkeypatch.setattr(
        campaign.hardened,
        "_swap_counter_sample",
        lambda: {
            "monotonic_ns": 200,
            "swap_in_bytes": 15,
            "swap_out_bytes": 20,
        },
    )

    campaign._finalize_worker_session([], session, active_error=root_error)

    retained = getattr(root_error, campaign._FAILURE_SWAP_TELEMETRY_ATTRIBUTE)
    assert retained["capture_status"] == "captured"
    assert retained["teardown_confirmed"] is False
    assert "worker stop failed" in retained["diagnostic"]
    (tmp_path / campaign.MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    campaign._write_invalid_attempt(
        tmp_path, execution_mode="concurrent", error=root_error
    )
    marker = campaign._read_json_regular(
        tmp_path / campaign.INVALID_ATTEMPT_FILENAME, "invalid marker"
    )
    assert marker["worker_shutdown_confirmed"] is False
    assert marker["recovery_policy"] == "not_recoverable"


def test_swap_capture_failure_does_not_mask_or_authorize_recovery(
    tmp_path, monkeypatch,
):
    root_error = RuntimeError("root workload failure")
    monkeypatch.setattr(campaign, "_stop_workers", lambda *args, **kwargs: None)

    def fail_capture(*_args, **_kwargs):
        raise RuntimeError("counter backend unavailable")

    monkeypatch.setattr(
        campaign, "_capture_post_teardown_swap_telemetry", fail_capture
    )

    campaign._finalize_worker_session(
        [], _swap_session([(100, 10, 20)]), active_error=root_error
    )

    retained = getattr(root_error, campaign._FAILURE_SWAP_TELEMETRY_ATTRIBUTE)
    assert retained["capture_status"] == "capture_failed"
    assert retained["teardown_confirmed"] is True
    assert retained["worker_session_swap_telemetry"] is None
    assert "counter backend unavailable" in retained["diagnostic"]
    (tmp_path / campaign.MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    campaign._write_invalid_attempt(
        tmp_path, execution_mode="concurrent", error=root_error
    )
    marker = campaign._read_json_regular(
        tmp_path / campaign.INVALID_ATTEMPT_FILENAME, "invalid marker"
    )
    assert marker["worker_shutdown_confirmed"] is True
    assert marker["failure_swap_telemetry"]["capture_status"] == "capture_failed"
    assert marker["recovery_policy"] == "not_recoverable"


def test_preflight_cleanup_failure_preserves_workload_error_and_swap_snapshot(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        campaign,
        "_new_worker_session_swap_telemetry",
        lambda: _swap_session([(100, 10, 20)]),
    )

    def fail_start(_scratch):
        raise RuntimeError("preflight worker start failed")

    monkeypatch.setattr(campaign, "_start_workers", fail_start)
    monkeypatch.setattr(campaign, "_stop_workers", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        campaign.hardened,
        "_swap_counter_sample",
        lambda: {
            "monotonic_ns": 200,
            "swap_in_bytes": 15,
            "swap_out_bytes": 20,
        },
    )

    def fail_cleanup(*_args, **_kwargs):
        raise OSError("scratch cleanup failed")

    monkeypatch.setattr(campaign.shutil, "rmtree", fail_cleanup)

    with pytest.raises(RuntimeError, match="preflight worker start failed") as raised:
        campaign.run_preflight(tmp_path)

    retained = getattr(
        raised.value, campaign._FAILURE_SWAP_TELEMETRY_ATTRIBUTE
    )
    assert retained["capture_status"] == "captured"
    assert retained["teardown_confirmed"] is False
    assert retained["worker_session_swap_telemetry"]["sample_count"] == 2


def test_failure_swap_telemetry_rejects_collapsed_shutdown_boundary():
    retained = _failure_swap_telemetry()
    retained["worker_session_swap_telemetry"] = _swap_session([(100, 10, 20)])
    retained["swap_in_bytes"] = 0

    with pytest.raises(RuntimeError, match="teardown boundary"):
        campaign._validated_failure_swap_telemetry(retained, "fixture")


def test_uncertain_worker_startup_cleanup_never_authorizes_recovery(
    tmp_path, monkeypatch,
):
    samples = iter(
        [
            {"monotonic_ns": 100, "swap_in_bytes": 10, "swap_out_bytes": 20},
            {"monotonic_ns": 200, "swap_in_bytes": 15, "swap_out_bytes": 20},
        ]
    )
    monkeypatch.setattr(
        campaign.hardened, "_swap_counter_sample", lambda: next(samples)
    )

    def fail_start(_root):
        raise RuntimeError("worker startup cleanup could not be confirmed")

    monkeypatch.setattr(campaign, "_start_workers", fail_start)
    monkeypatch.setattr(campaign, "_stop_workers", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="startup cleanup") as raised:
        campaign.execute_production(tmp_path, execution_mode="concurrent")

    retained = getattr(
        raised.value, campaign._FAILURE_SWAP_TELEMETRY_ATTRIBUTE
    )
    assert retained["capture_status"] == "captured"
    assert retained["teardown_confirmed"] is False
    (tmp_path / campaign.MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    campaign._write_invalid_attempt(
        tmp_path, execution_mode="concurrent", error=raised.value
    )
    marker = campaign._read_json_regular(
        tmp_path / campaign.INVALID_ATTEMPT_FILENAME, "invalid marker"
    )
    assert marker["worker_shutdown_confirmed"] is False
    assert marker["recovery_policy"] == "not_recoverable"


def test_confirmed_hardened_startup_cleanup_authorizes_fresh_recovery(
    tmp_path, monkeypatch,
):
    samples = iter(
        [
            {"monotonic_ns": 100, "swap_in_bytes": 10, "swap_out_bytes": 20},
            {"monotonic_ns": 200, "swap_in_bytes": 15, "swap_out_bytes": 20},
        ]
    )
    monkeypatch.setattr(
        campaign.hardened, "_swap_counter_sample", lambda: next(samples)
    )

    def fail_after_confirmed_cleanup(_root, *, worker_count):
        assert worker_count == campaign.WORKER_COUNT
        raise RuntimeError("worker readiness timed out")

    monkeypatch.setattr(
        campaign.hardened, "_start_workers", fail_after_confirmed_cleanup
    )
    monkeypatch.setattr(campaign, "_stop_workers", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="worker readiness timed out") as raised:
        campaign.execute_production(tmp_path, execution_mode="concurrent")

    assert (
        getattr(
            raised.value,
            campaign._WORKER_STARTUP_CLEANUP_CONFIRMED_ATTRIBUTE,
        )
        is True
    )
    retained = getattr(
        raised.value, campaign._FAILURE_SWAP_TELEMETRY_ATTRIBUTE
    )
    assert retained["capture_status"] == "captured"
    assert retained["teardown_confirmed"] is True
    (tmp_path / campaign.MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    campaign._write_invalid_attempt(
        tmp_path, execution_mode="concurrent", error=raised.value
    )
    marker = campaign._read_json_regular(
        tmp_path / campaign.INVALID_ATTEMPT_FILENAME, "invalid marker"
    )
    assert (
        marker["recovery_policy"]
        == "fresh_sequential_namespace_from_wave_zero_only"
    )


def test_execute_wrapper_attaches_retained_swap_to_post_shutdown_error(
    monkeypatch,
):
    retained = _failure_swap_telemetry()

    def fail_after_shutdown(_output_dir, *, execution_mode, failure_context):
        assert execution_mode == "concurrent"
        failure_context["failure_swap_telemetry"] = retained
        raise RuntimeError("post-shutdown validation failed")

    monkeypatch.setattr(campaign, "_execute_production_impl", fail_after_shutdown)

    with pytest.raises(RuntimeError, match="post-shutdown") as raised:
        campaign.execute_production(Path("unused"), execution_mode="concurrent")

    assert (
        getattr(raised.value, campaign._FAILURE_SWAP_TELEMETRY_ATTRIBUTE)
        == retained
    )


def test_unprintable_teardown_error_does_not_mask_workload_error(monkeypatch):
    class UnprintableTeardownError(RuntimeError):
        def __str__(self):
            raise RuntimeError("stringification failed")

    root_error = RuntimeError("root workload failure")

    def fail_stop(*_args, **_kwargs):
        raise UnprintableTeardownError()

    monkeypatch.setattr(campaign, "_stop_workers", fail_stop)
    monkeypatch.setattr(
        campaign.hardened,
        "_swap_counter_sample",
        lambda: {
            "monotonic_ns": 200,
            "swap_in_bytes": 15,
            "swap_out_bytes": 20,
        },
    )

    campaign._finalize_worker_session(
        [], _swap_session([(100, 10, 20)]), active_error=root_error
    )

    retained = getattr(root_error, campaign._FAILURE_SWAP_TELEMETRY_ATTRIBUTE)
    assert retained["teardown_confirmed"] is False
    assert "<unprintable exception>" in retained["diagnostic"]


def test_invalid_marker_survives_unprintable_root_exception(tmp_path):
    class UnprintableRootError(RuntimeError):
        def __str__(self):
            raise RuntimeError("stringification failed")

    campaign._write_invalid_attempt(
        tmp_path,
        execution_mode="concurrent",
        error=UnprintableRootError(),
    )

    marker = campaign._read_json_regular(
        tmp_path / campaign.INVALID_ATTEMPT_FILENAME, "invalid marker"
    )
    assert marker["error_type"] == "UnprintableRootError"
    assert "<unprintable exception>" in marker["error"]
    assert marker["recovery_policy"] == "not_recoverable"


def test_sequential_failure_marker_preserves_production_identity(
    tmp_path,
):
    (tmp_path / campaign.MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    error = RuntimeError("sequential production failed")
    setattr(
        error,
        campaign._FAILURE_SWAP_TELEMETRY_ATTRIBUTE,
        _failure_swap_telemetry(),
    )

    campaign._write_invalid_attempt(
        tmp_path,
        execution_mode="sequential_recovery",
        error=error,
    )

    marker = campaign._read_json_regular(
        tmp_path / campaign.INVALID_ATTEMPT_FILENAME, "invalid marker"
    )
    assert marker["stage"] == "production"
    assert marker["manifest_sha256"] == campaign._sha256_file(
        tmp_path / campaign.MANIFEST_FILENAME
    )
    assert marker["worker_shutdown_confirmed"] is True
    assert marker["recovery_policy"] == "not_recoverable"


def test_main_retains_production_swap_when_completion_validation_fails(
    tmp_path, monkeypatch,
):
    chimeraboost = tmp_path / "chimeraboost"
    chimeraboost.mkdir()
    output = tmp_path / "campaign"
    monkeypatch.setattr(campaign, "DEFAULT_CHIMERABOOST_PATH", chimeraboost)
    monkeypatch.setattr(campaign, "validate_source_freeze", lambda: {})
    monkeypatch.setattr(campaign, "collect_runtime_provenance", lambda: {})
    monkeypatch.setattr(
        campaign,
        "build_runtime_jobs",
        lambda **kwargs: (None, [object()] * campaign.EXPECTED_JOBS, 18),
    )
    monkeypatch.setattr(campaign, "validate_schedule", lambda value: None)
    monkeypatch.setattr(
        campaign,
        "_validate_campaign_namespace",
        lambda path, **kwargs: path,
    )
    monkeypatch.setattr(
        campaign, "collect_source_provenance", lambda **kwargs: {}
    )
    monkeypatch.setattr(campaign, "verify_live_official_splits", lambda: {})
    monkeypatch.setattr(campaign, "run_preflight", lambda output_dir: {})
    monkeypatch.setattr(
        campaign,
        "execute_production",
        lambda output_dir, **kwargs: {
            "worker_session_swap_telemetry": _swap_session(
                [(100, 10, 20), (200, 15, 20)]
            )
        },
    )

    def fail_completion(*_args, **_kwargs):
        raise RuntimeError("completion validation failed")

    monkeypatch.setattr(campaign, "write_completion_attestation", fail_completion)

    with pytest.raises(RuntimeError, match="completion validation"):
        campaign.main(
            [
                "--output-dir",
                str(output),
                "--chimeraboost-path",
                str(chimeraboost),
            ]
        )

    marker = campaign._read_json_regular(
        output / campaign.INVALID_ATTEMPT_FILENAME, "invalid marker"
    )
    assert marker["worker_shutdown_confirmed"] is True
    assert marker["failure_swap_telemetry"] == _failure_swap_telemetry()
    assert (
        marker["recovery_policy"]
        == "fresh_sequential_namespace_from_wave_zero_only"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__(
            "swap_in_delta", value["swap_in_delta"] + 1
        ),
        lambda value: (
            value["samples"][0].__setitem__("swap_in_bytes", 119),
            value.__setitem__("swap_in_delta", True),
        ),
        lambda value: value["samples"][1].__setitem__("swap_in_bytes", 111),
        lambda value: value["samples"][0].__setitem__("swap_in_bytes", -1),
        lambda value: value.__setitem__("sample_count", True),
        lambda value: (
            value.__setitem__("samples", value["samples"][:1]),
            value.__setitem__("sample_count", 1),
            value.__setitem__("swap_in_delta", 0),
            value.__setitem__("swap_out_delta", 0),
        ),
        lambda value: (
            value["samples"][1].__setitem__("swap_out_bytes", 201),
            value.__setitem__("swap_out_delta", 1),
        ),
    ],
)
def test_dispatch_swap_telemetry_rejects_hostile_mutations(mutation):
    telemetry = _dispatch_swap_telemetry()
    mutation(telemetry)

    with pytest.raises(RuntimeError, match="swap"):
        campaign._validated_dispatch_swap_deltas(telemetry, "dispatch")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("fold", True),
        lambda value: value.pop("sample"),
        lambda value: value.__setitem__("extra", 1),
        lambda value: value.__setitem__("arm_code", "X"),
    ],
)
def test_key_decoder_rejects_coercion_missing_and_extra_fields(mutation):
    key = min(campaign.expected_grid(), key=lambda value: (value[1], value[3], value[5]))
    payload = campaign._key_payload(key)
    mutation(payload)

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        campaign._key_tuple(payload)


@pytest.mark.parametrize("field", ["task_id", "repeat", "fold", "sample"])
@pytest.mark.parametrize("bad", [True, 1.0, "1"])
def test_key_decoder_rejects_every_numeric_coercion(field, bad):
    key = min(campaign.expected_grid(), key=lambda value: (value[1], value[3], value[5]))
    payload = campaign._key_payload(key)
    payload[field] = bad
    with pytest.raises(RuntimeError, match="exact integer"):
        campaign._key_tuple(payload)


def test_schedule_rejects_duplicates_slot_tampering_and_reordering():
    schedule = campaign.expected_schedule()
    duplicate = deepcopy(schedule)
    duplicate[1]["jobs"][0]["key"] = duplicate[0]["jobs"][0]["key"]
    with pytest.raises(RuntimeError, match="invalid or duplicate"):
        campaign.validate_schedule(duplicate)

    bad_slot = deepcopy(schedule)
    bad_slot[0]["jobs"][1]["worker_slot"] = 0
    with pytest.raises(RuntimeError, match="slots"):
        campaign.validate_schedule(bad_slot)

    reordered = deepcopy(schedule)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(RuntimeError, match="header"):
        campaign.validate_schedule(reordered)

    bool_key = deepcopy(schedule)
    target = next(
        item
        for wave in bool_key
        for item in wave["jobs"]
        if item["key"]["fold"] == 1
    )
    target["key"]["fold"] = True
    with pytest.raises(RuntimeError, match="exact integer"):
        campaign.validate_schedule(bool_key)


def test_persisted_concurrent_wave_recomputes_raw_timing_and_artifact_identity(
    tmp_path,
):
    entry, expected, ready, artifacts = _persisted_wave_fixture(
        tmp_path, "concurrent"
    )

    _validate_persisted_wave_fixture(
        tmp_path, "concurrent", entry, expected, ready, artifacts
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda entry, artifacts: entry.__setitem__(
            "start_skew_ns", entry["start_skew_ns"] + 1
        ),
        lambda entry, artifacts: entry.__setitem__(
            "wave_elapsed_ns", entry["wave_elapsed_ns"] + 1
        ),
        lambda entry, artifacts: entry["reports"][1].__setitem__(
            "barrier_release_monotonic_ns",
            entry["reports"][1]["barrier_release_monotonic_ns"] + 1,
        ),
        lambda entry, artifacts: entry["reports"][0].__setitem__(
            "started_monotonic_ns",
            entry["reports"][0]["barrier_release_monotonic_ns"] - 1,
        ),
        lambda entry, artifacts: entry["reports"][0].__setitem__(
            "result_sha256", "f" * 64
        ),
        lambda entry, artifacts: entry.__setitem__("swap_in_delta", -1),
        lambda entry, artifacts: entry.__setitem__(
            "swap_end_sample_index", entry["swap_start_sample_index"]
        ),
        lambda entry, artifacts: entry.__setitem__("unregistered_field", 1),
    ],
)
def test_persisted_concurrent_wave_rejects_detached_mutations(
    tmp_path, mutation
):
    entry, expected, ready, artifacts = _persisted_wave_fixture(
        tmp_path, "concurrent"
    )
    mutation(entry, artifacts)

    with pytest.raises(RuntimeError, match="operational wave"):
        _validate_persisted_wave_fixture(
            tmp_path, "concurrent", entry, expected, ready, artifacts
        )


def test_persisted_concurrent_wave_rejects_forged_nonoverlap(tmp_path):
    entry, expected, ready, artifacts = _persisted_wave_fixture(
        tmp_path, "concurrent"
    )
    entry["reports"][0]["ended_monotonic_ns"] = (
        entry["reports"][0]["started_monotonic_ns"] + 1
    )
    _refresh_persisted_wave_timing(entry, "concurrent")

    with pytest.raises(RuntimeError, match="concurrency changed"):
        _validate_persisted_wave_fixture(
            tmp_path, "concurrent", entry, expected, ready, artifacts
        )


def test_persisted_sequential_wave_recomputes_skew_and_enforces_order(tmp_path):
    entry, expected, ready, artifacts = _persisted_wave_fixture(
        tmp_path, "sequential_recovery"
    )

    _validate_persisted_wave_fixture(
        tmp_path, "sequential_recovery", entry, expected, ready, artifacts
    )

    bad_aggregate = deepcopy(entry)
    bad_aggregate["start_skew_ns"] += 1
    with pytest.raises(RuntimeError, match="sequential ordering changed"):
        _validate_persisted_wave_fixture(
            tmp_path,
            "sequential_recovery",
            bad_aggregate,
            expected,
            ready,
            artifacts,
        )

    early_dispatch = deepcopy(entry)
    first_end = early_dispatch["reports"][0]["ended_monotonic_ns"]
    early_dispatch["reports"][1]["barrier_release_monotonic_ns"] = first_end - 1
    _refresh_persisted_wave_timing(early_dispatch, "sequential_recovery")
    with pytest.raises(RuntimeError, match="sequential ordering changed"):
        _validate_persisted_wave_fixture(
            tmp_path,
            "sequential_recovery",
            early_dispatch,
            expected,
            ready,
            artifacts,
        )

    overlapping = deepcopy(entry)
    overlapping["reports"][1]["barrier_release_monotonic_ns"] = first_end - 2
    overlapping["reports"][1]["started_monotonic_ns"] = first_end - 1
    _refresh_persisted_wave_timing(overlapping, "sequential_recovery")
    with pytest.raises(RuntimeError, match="sequential ordering changed"):
        _validate_persisted_wave_fixture(
            tmp_path,
            "sequential_recovery",
            overlapping,
            expected,
            ready,
            artifacts,
        )


def test_persisted_preflight_recomputes_probe_concurrency_and_warmup_digest(
    tmp_path, monkeypatch,
):
    _stub_persisted_warmup_validators(monkeypatch)
    campaign._validate_persisted_preflight(
        _persisted_preflight_fixture(tmp_path)
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__(
            "start_skew_ns", value["start_skew_ns"] + 1
        ),
        lambda value: value["synthetic_probes"][1].__setitem__(
            "barrier_release_monotonic_ns",
            value["synthetic_probes"][1]["barrier_release_monotonic_ns"] + 1,
        ),
        lambda value: value["synthetic_probes"][0].__setitem__(
            "started_monotonic_ns",
            value["synthetic_probes"][0]["barrier_release_monotonic_ns"] - 1,
        ),
        lambda value: value["synthetic_probes"][0].__setitem__(
            "behavior_sha256", "f" * 64
        ),
        lambda value: value["worker_warmup"][1].update(
            {
                "worker_slot": 0,
                "pid": value["worker_ready"][0]["pid"],
                }
            ),
        lambda value: value.__setitem__(
            "swap_in_bytes", value["swap_in_bytes"] + 1
        ),
        lambda value: value["worker_session_swap_telemetry"].__setitem__(
            "swap_in_delta",
            value["worker_session_swap_telemetry"]["swap_in_delta"] + 1,
        ),
        lambda value: value["synthetic_dispatch_telemetry"].__setitem__(
            "swap_in_delta",
            value["synthetic_dispatch_telemetry"]["swap_in_delta"] + 1,
        ),
        lambda value: value.__setitem__("extra", True),
    ],
)
def test_persisted_preflight_rejects_detached_mutations(
    tmp_path, monkeypatch, mutation
):
    _stub_persisted_warmup_validators(monkeypatch)
    value = _persisted_preflight_fixture(tmp_path)
    mutation(value)

    with pytest.raises(RuntimeError, match="preflight"):
        campaign._validate_persisted_preflight(value)


def test_persisted_preflight_rejects_forged_nonoverlap(tmp_path, monkeypatch):
    _stub_persisted_warmup_validators(monkeypatch)
    value = _persisted_preflight_fixture(tmp_path)
    probe = value["synthetic_probes"][0]
    probe["ended_monotonic_ns"] = probe["started_monotonic_ns"] + 1
    starts = [item["started_monotonic_ns"] for item in value["synthetic_probes"]]
    ends = [item["ended_monotonic_ns"] for item in value["synthetic_probes"]]
    value["start_skew_ns"] = max(starts) - min(starts)
    value["overlap_ns"] = max(0, min(ends) - max(starts))

    with pytest.raises(RuntimeError, match="preflight concurrency"):
        campaign._validate_persisted_preflight(value)


def test_compact_candidates_accept_zero_and_enforce_first_argmin():
    value = campaign._compact_candidate_metadata(
        _candidate_selection(), field="fixture"
    )
    assert value["selected_candidate_index"] == 0
    assert value["candidates"][0]["validation_rmse"] == 0.0

    with pytest.raises(RuntimeError, match="first argmin"):
        campaign._compact_candidate_metadata(
            _candidate_selection(scores=(1.0, 1.0, 2.0), selected=1),
            field="fixture",
        )
    bad = _candidate_selection()
    bad["candidates"][1]["validation_score"] = math.nan
    with pytest.raises(RuntimeError, match="finite and nonnegative"):
        campaign._compact_candidate_metadata(bad, field="fixture")


@pytest.mark.parametrize("bad_patience", [None, 1, True, "50"])
def test_a10_projection_rejects_non_frozen_early_stopping_patience(bad_patience):
    fitted = {
        field: None
        for field in (
            set(campaign.hardened.screen.REQUIRED_FIT_METADATA)
            | {"tree_mode_selection"}
        )
    }
    fitted.update(
        {
            "iterations_requested": 10_000,
            "iterations_attempted": 100,
            "rounds_completed": 100,
            "rounds_retained": 90,
            "best_iteration": 90,
            "early_stopping_rounds": bad_patience,
        }
    )

    with pytest.raises(RuntimeError, match="must be an integer|50-round patience"):
        campaign._parse_a10_fit(
            fitted,
            selected_params={},
            field="fixture",
        )


@pytest.mark.parametrize(
    "arm,audit,expected_lane",
    [
        ("D", None, "boosting"),
        ("M", _callback_audit("chimeraboost"), "constant"),
        ("C", _callback_audit("catboost"), "cpu"),
    ],
)
def test_comparator_projection_freezes_arm_contracts(
    monkeypatch, arm, audit, expected_lane
):
    fit = _comparator_fit(arm)
    monkeypatch.setattr(
        campaign.comparators,
        "_validate_comparator_fit",
        lambda value, **_: dict(value),
    )

    row = campaign._parse_comparator_fit(
        fit, audit_value=audit, arm=arm, field="fixture"
    )

    assert row["selected_lane"] == expected_lane
    assert row["time_callback_hit"] is False
    assert row["candidate_metadata"] is None


def test_comparator_projection_rejects_uninstrumented_or_hit_callback(monkeypatch):
    monkeypatch.setattr(
        campaign.comparators,
        "_validate_comparator_fit",
        lambda value, **_: dict(value),
    )
    fit = _comparator_fit("M")
    for mutation in (
        lambda value: value.__setitem__("time_callback_instrumented", False),
        lambda value: value.__setitem__("time_callback_hit", True),
        lambda value: value.__setitem__("time_callback_instance_count", 0),
        lambda value: value.__setitem__("time_limit_seconds", 3_601.0),
    ):
        audit = _callback_audit("chimeraboost")
        mutation(audit)
        with pytest.raises(RuntimeError, match="contract|wall-clock"):
            campaign._parse_comparator_fit(
                fit, audit_value=audit, arm="M", field="fixture"
            )


def test_result_path_rejects_symlink_component_before_pickle_decode(tmp_path):
    root = tmp_path / "campaign"
    real = tmp_path / "real"
    root.mkdir()
    real.mkdir()
    (real / "results.pkl").write_bytes(b"not-a-pickle")
    (root / "linked").symlink_to(real, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink component"):
        campaign.parse_result_path(root / "linked" / "results.pkl", output_dir=root)


def test_recovery_requires_invalid_marker_and_rejects_completed_source(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(campaign, "validate_source_freeze", lambda: {})
    monkeypatch.setattr(
        campaign, "collect_source_provenance", lambda **kwargs: {}
    )
    monkeypatch.setattr(campaign, "collect_runtime_provenance", lambda: {})
    source = tmp_path / "experiments" / "failed"
    source.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="invalid marker"):
        campaign._sequential_recovery_record(source)
    manifest = campaign.build_run_manifest(
        output_dir=source,
        execution_mode="concurrent",
        source_freeze={},
        source={},
        runtime={},
        sequential_recovery=None,
    )
    campaign._atomic_write_json(source / campaign.MANIFEST_FILENAME, manifest)
    manifest_sha = campaign._sha256_file(source / campaign.MANIFEST_FILENAME)
    campaign._atomic_write_json(
        source / campaign.INVALID_ATTEMPT_FILENAME,
        {
            "schema_version": campaign.HARNESS_SCHEMA_VERSION,
            "kind": campaign.CAMPAIGN_KIND + "_invalid_attempt",
            "invalidated_at_utc": "2026-07-15T00:00:00Z",
            "execution_mode": "concurrent",
            "stage": "production",
            "reuse_allowed": False,
            "recovery_policy": "fresh_sequential_namespace_from_wave_zero_only",
            "manifest_sha256": manifest_sha,
            "worker_shutdown_confirmed": True,
            "failure_swap_telemetry": _failure_swap_telemetry(),
            "error_type": "RuntimeError",
            "error": "fixture",
        },
    )
    record = campaign._sequential_recovery_record(source)
    assert record["reuse_policy"] == "no_results_reused_fresh_wave_zero"
    completion_path = source / campaign.COMPLETION_ATTESTATION_FILENAME
    completion_path.symlink_to(source / "missing-completion-target")
    with pytest.raises(RuntimeError, match="completed"):
        campaign._sequential_recovery_record(source)
    completion_path.unlink()
    marker_path = source / campaign.INVALID_ATTEMPT_FILENAME
    marker = campaign._read_json_regular(marker_path, "invalid marker")
    marker["schema_version"] = float(campaign.HARNESS_SCHEMA_VERSION)
    campaign._atomic_write_json(marker_path, marker)
    with pytest.raises(RuntimeError, match="invalid marker"):
        campaign._sequential_recovery_record(source)
    (source / campaign.COMPLETION_ATTESTATION_FILENAME).write_text("{}")
    with pytest.raises(RuntimeError, match="completed"):
        campaign._sequential_recovery_record(source)


def test_recovery_rejects_symlink_resolving_into_raw_result_namespace(tmp_path):
    raw_source = tmp_path / "ExPeRiMeNtS" / "failed-concurrent" / "ReSuLtS.PkL"
    raw_source.parent.mkdir(parents=True)
    raw_source.write_bytes(b"opaque raw result")
    alias = tmp_path / "benign-recovery-source"
    alias.symlink_to(raw_source)

    with pytest.raises(RuntimeError, match="source path is unsafe"):
        campaign._sequential_recovery_record(alias)

    with pytest.raises(RuntimeError, match="source path is unsafe"):
        campaign._sequential_recovery_record(
            tmp_path / "ReSuLtS.PkL" / "failed-concurrent"
        )


def test_completion_refuses_changed_sequential_recovery_source(
    tmp_path, monkeypatch,
):
    output_dir = tmp_path / "fresh-sequential"
    output_dir.mkdir()
    (output_dir / campaign.MANIFEST_FILENAME).write_text("{}")
    recovery = {"source_output_dir": str(tmp_path / "failed-concurrent")}
    manifest = {
        "execution_mode": "sequential_recovery",
        "sequential_recovery": recovery,
        "source_freeze": {},
        "source": {},
        "runtime": {},
    }
    swap_audit = {
        "preflight": {
            "worker_lifecycle_swap_in_bytes": 0,
            "worker_lifecycle_swap_out_bytes": 0,
        },
        "production": {
            "worker_lifecycle_swap_in_bytes": 0,
            "worker_lifecycle_swap_out_bytes": 0,
            "measured_dispatch_count": 90,
            "wave_count": 45,
        },
    }
    operational = {
        campaign.PREFLIGHT_REPORT_FILENAME: {},
        campaign.CONCURRENCY_HISTORY_FILENAME: {
            "peak_combined_rss_fraction": 0.1,
        },
        campaign.WARMUP_HISTORY_FILENAME: {},
    }
    monkeypatch.setattr(campaign, "collect_result_artifacts", lambda _: {})
    monkeypatch.setattr(
        campaign,
        "validate_completed_results",
        lambda *_: ({}, [], []),
    )
    original_read_json = campaign._read_json_regular
    monkeypatch.setattr(
        campaign,
        "_read_json_regular",
        lambda path, field: (
            operational[path.name]
            if path.name in operational
            else original_read_json(path, field)
        ),
    )
    monkeypatch.setattr(campaign, "build_swap_audit", lambda *_args, **_kwargs: swap_audit)
    monkeypatch.setattr(
        campaign, "validate_completion_for_analysis", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        campaign,
        "_artifact_metadata",
        lambda path, _root: {
            "path": path.name,
            "sha256": "0" * 64,
            "size_bytes": 0,
        },
    )
    monkeypatch.setattr(
        campaign,
        "validate_operational_artifacts_for_analysis",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(campaign, "validate_source_freeze", lambda: {})
    monkeypatch.setattr(
        campaign, "collect_source_provenance", lambda **_kwargs: {}
    )
    monkeypatch.setattr(campaign, "collect_runtime_provenance", lambda: {})

    def reject_changed_source(value):
        assert value is recovery
        raise RuntimeError("recovery source artifacts changed")

    monkeypatch.setattr(
        campaign,
        "validate_sequential_recovery_record",
        reject_changed_source,
    )

    with pytest.raises(RuntimeError, match="recovery source artifacts changed"):
        campaign.write_completion_attestation(output_dir, manifest=manifest)
    assert not (output_dir / campaign.COMPLETION_ATTESTATION_FILENAME).exists()


def test_recovery_rejects_ancestor_or_descendant_destination(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(RuntimeError, match="disjoint"):
        campaign._sequential_recovery_record(
            source, destination=source / "nested"
        )
    with pytest.raises(RuntimeError, match="disjoint"):
        campaign._sequential_recovery_record(source, destination=tmp_path)


def test_campaign_namespaces_inside_repository_must_be_git_ignored(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    (repository / ".gitignore").write_text(".cache/\n", encoding="utf-8")
    monkeypatch.setattr(campaign, "REPOSITORY_ROOT", repository)

    ignored_source = repository / ".cache" / "failed-concurrent"
    ignored_destination = repository / ".cache" / "fresh-sequential"
    outside = tmp_path / "outside-campaign"
    assert campaign._validate_campaign_namespace(
        ignored_source, field="source"
    ) == ignored_source.resolve()
    assert campaign._validate_campaign_namespace(
        ignored_destination, field="destination"
    ) == ignored_destination.resolve()
    assert campaign._validate_campaign_namespace(outside, field="output") == (
        outside.resolve()
    )

    with pytest.raises(RuntimeError, match="inside.*not Git-ignored"):
        campaign._validate_campaign_namespace(
            repository / "failed-concurrent", field="source"
        )
    with pytest.raises(RuntimeError, match="inside.*not Git-ignored"):
        campaign._validate_campaign_namespace(
            repository / "fresh-sequential", field="destination"
        )
    with pytest.raises(RuntimeError, match="cannot be the campaign repository"):
        campaign._validate_campaign_namespace(repository, field="output")


def test_campaign_output_path_reserves_results_name_but_allows_experiments(
    tmp_path,
):
    allowed = tmp_path / "experiments" / "campaign"
    assert campaign._validated_campaign_output_path(allowed) == allowed.resolve()

    for rejected in (
        tmp_path / "ReSuLtS.PkL" / "campaign",
        tmp_path / "campaign" / ".." / "ReSuLtS.PkL",
    ):
        with pytest.raises(RuntimeError, match="reserved raw-result name"):
            campaign._validated_campaign_output_path(rejected)

    canonical = tmp_path / "ReSuLtS.PkL" / "canonical"
    canonical.mkdir(parents=True)
    alias = tmp_path / "benign-output-alias"
    alias.symlink_to(canonical, target_is_directory=True)
    with pytest.raises(RuntimeError, match="reserved raw-result name"):
        campaign._validated_campaign_output_path(alias)


def test_recovery_rejects_nonignored_repository_namespaces_before_provenance(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    (repository / ".gitignore").write_text(".cache/\n", encoding="utf-8")
    monkeypatch.setattr(campaign, "REPOSITORY_ROOT", repository)

    nonignored_source = repository / "failed-concurrent"
    nonignored_source.mkdir()
    with pytest.raises(RuntimeError, match="source namespace.*not Git-ignored"):
        campaign._sequential_recovery_record(nonignored_source)

    ignored_source = repository / ".cache" / "failed-concurrent"
    ignored_source.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="invalid marker"):
        campaign._sequential_recovery_record(ignored_source)

    outside_source = tmp_path / "outside-failed-concurrent"
    outside_source.mkdir()
    with pytest.raises(RuntimeError, match="destination namespace.*not Git-ignored"):
        campaign._sequential_recovery_record(
            outside_source, destination=repository / "fresh-sequential"
        )


def test_source_provenance_does_not_delegate_output_exclusion_and_merges_files(
    tmp_path, monkeypatch
):
    assert Path("benchmarks/run_tabarena_regression_accuracy_shootout.py") in (
        campaign.SOURCE_FILES
    )
    repository = tmp_path / "repository"
    repository.mkdir()
    local_relative = Path("local_ctr23_source.py")
    local_path = repository / local_relative
    local_path.write_text("local source\n", encoding="utf-8")
    inherited_relative = "benchmarks/run_tabarena_regression_same_machine.py"
    inherited_metadata = {"sha256": "a" * 64, "git_blob": "b" * 40}
    delegated_calls = []

    def delegated(*, output_dir, chimeraboost_path):
        delegated_calls.append((output_dir, chimeraboost_path))
        return {
            "git_head": "c" * 40,
            "git_tree": "d" * 40,
            "files": {inherited_relative: inherited_metadata},
            "tabarena": {"git_head": campaign.TABARENA_COMMIT},
            "chimeraboost": {"git_head": campaign.CHIMERABOOST_TAG_COMMIT},
            "catboost": {"version": "1.2.10", "files": {"wheel": {}}},
            "external_adapter_sources": {},
        }

    def fake_run(args, **kwargs):
        del kwargs
        stdout = (
            campaign.DARKOFIT_SUBTREE + "\n"
            if args[1:3] == ["rev-parse", "HEAD:darkofit"]
            else "e" * 40 + "\n"
        )
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(campaign, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(campaign, "SOURCE_FILES", (local_relative,))
    monkeypatch.setattr(
        campaign.comparators, "collect_source_provenance", delegated
    )
    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    outside_output = tmp_path / "outside-campaign"
    chimera_path = tmp_path / "chimera"

    source = campaign.collect_source_provenance(
        output_dir=outside_output, chimeraboost_path=chimera_path
    )

    assert delegated_calls == [(None, chimera_path)]
    assert source["files"][inherited_relative] == inherited_metadata
    assert source["files"][str(local_relative)]["sha256"] == (
        campaign._sha256_file(local_path)
    )


@requires_pinned_campaign_stack
def test_runtime_provenance_enforces_frozen_host_and_dependency_lock():
    runtime = campaign.collect_runtime_provenance()
    assert runtime["python_version"] == "3.12.13"
    assert runtime["machine"] == "arm64"
    assert runtime["hardware"]["logical_cpu_count"] == 18
    assert runtime["hardware"]["total_memory_bytes"] == 137_438_953_472
    assert runtime["packages"]["openml"] == "0.15.1"
    assert runtime["packages"]["pyarrow"] == "24.0.0"
    assert runtime["packages"]["liac-arff"] == "2.5.0"


@requires_pinned_campaign_stack
def test_dry_run_does_not_create_output_namespace(tmp_path, capsys):
    output = tmp_path / "must-not-exist"

    assert campaign.main(["--dry-run", "--output-dir", str(output)]) == 0

    assert not output.exists()
    assert '"status": "dry_run_valid"' in capsys.readouterr().out
