"""Hostile unit tests for the executable CTR23 confirmation runner."""

from __future__ import annotations

import hashlib
import math
import subprocess
from copy import deepcopy
from pathlib import Path

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
    return {
        "schema_version": 1,
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
        "swap_out_bytes": 0,
        "peak_combined_rss_fraction": 0.1,
    }


def _stub_persisted_warmup_validators(monkeypatch) -> None:
    from benchmarks import tabarena_comparator_warmup

    monkeypatch.setattr(
        campaign.hardened.screen,
        "_validate_followon_warmup_history",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tabarena_comparator_warmup,
        "validate_comparator_warmup_history",
        lambda *args, **kwargs: None,
    )


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
    source = tmp_path / "failed"
    source.mkdir()
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
            "schema_version": 1,
            "kind": campaign.CAMPAIGN_KIND + "_invalid_attempt",
            "invalidated_at_utc": "2026-07-15T00:00:00Z",
            "execution_mode": "concurrent",
            "stage": "production",
            "reuse_allowed": False,
            "recovery_policy": "fresh_sequential_namespace_from_wave_zero_only",
            "manifest_sha256": manifest_sha,
            "error_type": "RuntimeError",
            "error": "fixture",
        },
    )
    record = campaign._sequential_recovery_record(source)
    assert record["reuse_policy"] == "no_results_reused_fresh_wave_zero"
    (source / campaign.COMPLETION_ATTESTATION_FILENAME).write_text("{}")
    with pytest.raises(RuntimeError, match="completed"):
        campaign._sequential_recovery_record(source)


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
