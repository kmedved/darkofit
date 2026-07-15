"""Frozen-protocol tests for the two-process TabArena accuracy shootout."""

from __future__ import annotations

from copy import deepcopy
import itertools
import math
import os
from types import SimpleNamespace

import pytest

import benchmarks.run_tabarena_regression_followon_screen as shared_screen
import benchmarks.run_tabarena_regression_accuracy_shootout as shootout


EXPECTED_SCHEDULE_SHA256 = (
    "6b4d727be3b4094b8cdf0627a13b26de513d0dbd78bcc558070403593c18bcbf"
)


@pytest.fixture(autouse=True)
def _stable_host_swap_counters(monkeypatch):
    monotonic_ns = itertools.count(1)
    monkeypatch.setattr(
        shootout,
        "_swap_counter_sample",
        lambda: {
            "monotonic_ns": next(monotonic_ns),
            "swap_in_bytes": 0,
            "swap_out_bytes": 0,
        },
    )


def _public_key(job: dict) -> tuple[str, int, int, str]:
    key = job["key"]
    return key["dataset"], key["repeat"], key["fold"], key["arm"]


def _zero_swap_session() -> dict:
    return {
        "sample_count": 2,
        "samples": [
            {"monotonic_ns": 1, "swap_in_bytes": 0, "swap_out_bytes": 0},
            {"monotonic_ns": 2, "swap_in_bytes": 0, "swap_out_bytes": 0},
        ],
        "swap_in_delta": 0,
        "swap_out_delta": 0,
    }


def _preflight_worker_report(dataset: str, arm: str) -> dict:
    repeat, fold = shootout.PREFLIGHT_COORDINATES[dataset]
    key = shootout._key_payload((dataset, repeat, fold, arm))
    slot = 0 if dataset == "physiochemical_protein" else 1
    physical_memory = 128_000_000_000
    release_ns = 1_000_000_000
    started_ns = 2_000_000_000
    ended_ns = 12_000_000_000
    return {
        "key": key,
        "slot": slot,
        "pid": 10_000 + slot,
        "status": "ok",
        "result_count": 1,
        "child_count": 8,
        "deadline_hit": False,
        "auto_candidate_fit_count": 24 if arm == "A10" else 0,
        "elapsed_seconds": 10.0,
        "barrier_release_monotonic_ns": release_ns,
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "process_peak_rss_bytes": 1_000_000_000,
        "behavior_sha256": shootout.hashlib.sha256(
            f"{dataset}:{arm}".encode()
        ).hexdigest(),
        "telemetry": {
            "sample_count": 1,
            "samples": [
                {
                    "monotonic_ns": 1,
                    "swap_in_bytes": 0,
                    "swap_out_bytes": 0,
                    "combined_rss_bytes": 2_000_000_000,
                    "physical_memory_bytes": physical_memory,
                }
            ],
            "swap_in_delta": 0,
            "swap_out_delta": 0,
            "peak_combined_rss_bytes": 2_000_000_000,
            "physical_memory_bytes": physical_memory,
            "worker_high_water_rss_bytes": [
                {
                    "slot": worker_slot,
                    "pid": 10_000 + worker_slot,
                    "process_peak_rss_bytes": 1_000_000_000,
                }
                for worker_slot in range(shootout.WORKER_COUNT)
            ],
            "high_water_combined_rss_bytes": 2_000_000_000,
            "barrier_release_monotonic_ns": release_ns,
            "start_skew_seconds": 0.0,
            "wave_seconds": 11.0,
            "overlap_seconds": 0.0,
            "solo_tail_seconds": 0.0,
        },
    }


def _passing_preflight_report() -> dict:
    protein = "physiochemical_protein"
    qsar = "QSAR-TID-11"
    isolated = [
        _preflight_worker_report(dataset, arm)
        for dataset in (protein, qsar)
        for arm in ("A10", "B10")
    ]
    by_key = {
        (report["key"]["dataset"], report["key"]["arm"]): report
        for report in isolated
    }

    def concurrent(dataset: str, arm: str, slot: int) -> dict:
        result = deepcopy(by_key[(dataset, arm)])
        result["slot"] = slot
        result["barrier_release_monotonic_ns"] = 100_000_000
        result["started_monotonic_ns"] = 2_000_000_000 + slot * 100_000_000
        result["ended_monotonic_ns"] = 12_000_000_000 + slot * 100_000_000
        return result

    return {
        "worker_ready": [
            {"slot": slot, "pid": 10_000 + slot}
            for slot in range(shootout.WORKER_COUNT)
        ],
        "worker_data_prime": [
            {
                "worker_slot": slot,
                "pid": 10_000 + slot,
                "keys": [
                    shootout._key_payload(
                        ("physiochemical_protein", 0, 0, "A10")
                    ),
                    shootout._key_payload(("QSAR-TID-11", 2, 2, "A10")),
                ],
            }
            for slot in range(shootout.WORKER_COUNT)
        ],
        "isolated_runs": isolated,
        "concurrent_waves": [
            {
                "wave_seconds": 12.0,
                "start_skew_seconds": 0.1,
                "reports": [
                    concurrent(protein, "A10", 0),
                    concurrent(qsar, "B10", 1),
                ],
                "telemetry": {
                    "sample_count": 1,
                    "samples": [
                        {
                            "monotonic_ns": 1,
                            "swap_in_bytes": 0,
                            "swap_out_bytes": 0,
                            "combined_rss_bytes": 2_000_000_000,
                            "physical_memory_bytes": 128_000_000_000,
                        }
                    ],
                    "swap_in_delta": 0,
                    "swap_out_delta": 0,
                    "peak_combined_rss_bytes": 2_000_000_000,
                    "physical_memory_bytes": 128_000_000_000,
                    "worker_high_water_rss_bytes": [
                        {
                            "slot": slot,
                            "pid": 10_000 + slot,
                            "process_peak_rss_bytes": 1_000_000_000,
                        }
                        for slot in range(shootout.WORKER_COUNT)
                    ],
                    "high_water_combined_rss_bytes": 2_000_000_000,
                    "barrier_release_monotonic_ns": 100_000_000,
                    "start_skew_seconds": 0.1,
                    "wave_seconds": 12.0,
                    "overlap_seconds": 9.9,
                    "solo_tail_seconds": 0.1,
                },
            },
            {
                "wave_seconds": 12.0,
                "start_skew_seconds": 0.1,
                "reports": [
                    concurrent(protein, "B10", 0),
                    concurrent(qsar, "A10", 1),
                ],
                "telemetry": {
                    "sample_count": 1,
                    "samples": [
                        {
                            "monotonic_ns": 1,
                            "swap_in_bytes": 0,
                            "swap_out_bytes": 0,
                            "combined_rss_bytes": 2_000_000_000,
                            "physical_memory_bytes": 128_000_000_000,
                        }
                    ],
                    "swap_in_delta": 0,
                    "swap_out_delta": 0,
                    "peak_combined_rss_bytes": 2_000_000_000,
                    "physical_memory_bytes": 128_000_000_000,
                    "worker_high_water_rss_bytes": [
                        {
                            "slot": slot,
                            "pid": 10_000 + slot,
                            "process_peak_rss_bytes": 1_000_000_000,
                        }
                        for slot in range(shootout.WORKER_COUNT)
                    ],
                    "high_water_combined_rss_bytes": 2_000_000_000,
                    "barrier_release_monotonic_ns": 100_000_000,
                    "start_skew_seconds": 0.1,
                    "wave_seconds": 12.0,
                    "overlap_seconds": 9.9,
                    "solo_tail_seconds": 0.1,
                },
            },
        ],
        "worker_session_swap_telemetry": _zero_swap_session(),
        "worker_restarts": False,
    }


def _fake_jobs() -> list[SimpleNamespace]:
    experiments = {}
    for internal_arm, spec in shootout.screen.ARM_SPECS.items():
        model_cls = type(spec["model_cls"], (), {})
        experiments[internal_arm] = SimpleNamespace(
            name=f"DarkoFit_c1_screen_{internal_arm}_BAG_L1",
            method_kwargs={
                "model_cls": model_cls,
                "model_hyperparameters": {
                    **spec["config"],
                    "ag_args": {"name_suffix": f"_c1_screen_{internal_arm}"},
                    "ag_args_ensemble": shootout.screen.expected_ag_ensemble_config(),
                },
            },
        )
    return [
        SimpleNamespace(
            experiment=experiments[shootout.PUBLIC_TO_INTERNAL_ARM[public_arm]],
            task=SimpleNamespace(dataset=dataset, repeat=repeat, fold=fold),
        )
        for dataset, repeat, fold in shootout.expected_coordinates()
        for public_arm in ("A10", "B10")
    ]


def _valid_concurrency_history(output_dir) -> dict:
    entries = []
    for wave in shootout.expected_wave_schedule():
        reports = []
        release_ns = 1_000_000_000 + wave["wave_index"] * 10_000_000
        jobs_by_slot = {item["worker_slot"]: item for item in wave["jobs"]}
        for item in wave["jobs"]:
            key = item["key"]
            relative = shootout.screen.expected_result_relative_path(
                key["dataset"],
                key["repeat"],
                key["fold"],
                key["internal_arm"],
            )
            reports.append(
                {
                    "type": "result",
                    "command_id": f"wave-{wave['wave_index']}-{item['worker_slot']}",
                    "key": deepcopy(key),
                    "slot": item["worker_slot"],
                    "status": "ok",
                    "pid": 10_000 + item["worker_slot"],
                    "result_root": str(output_dir),
                    "result_path": str(output_dir / relative),
                    "result_count": 1,
                    "result_sha256": shootout.hashlib.sha256(
                        repr(("result", wave["wave_index"], item["worker_slot"])).encode()
                    ).hexdigest(),
                    "result_size_bytes": 123,
                    "process_peak_rss_bytes": 1_000_000_000,
                    "child_count": 8,
                    "deadline_hit": False,
                    "auto_candidate_fit_count": 24 if key["arm"] == "A10" else 0,
                    "behavior_sha256": shootout.hashlib.sha256(
                        repr((wave["wave_index"], item["worker_slot"])).encode()
                    ).hexdigest(),
                    "barrier_release_monotonic_ns": release_ns,
                    "started_monotonic_ns": release_ns + 100_000,
                    "ended_monotonic_ns": release_ns + 5_000_000,
                    "elapsed_seconds": 0.0049,
                    "cpu_time_seconds": 0.004,
                    "start_method": "spawn",
                    "wave_index": wave["wave_index"],
                    "partner_key": deepcopy(
                        jobs_by_slot[1 - item["worker_slot"]]["key"]
                    ),
                    "wave_schedule_sha256": shootout.wave_schedule_sha256(),
                }
            )
        entries.append(
            {
                "wave_index": wave["wave_index"],
                "dataset": wave["dataset"],
                "execution_mode": "concurrent",
                "reports": reports,
                "telemetry": {
                    "sample_count": 1,
                    "samples": [
                        {
                            "monotonic_ns": release_ns + 200_000,
                            "load_average": [1.0, 1.0, 1.0],
                            "available_memory_bytes": 100_000_000_000,
                            "physical_memory_bytes": 128_000_000_000,
                            "swap_in_bytes": 0,
                            "swap_out_bytes": 0,
                            "combined_rss_bytes": 2_000_000_000,
                            "combined_thread_count": 36,
                            "workers": [
                                {
                                    "slot": slot,
                                    "pid": 10_000 + slot,
                                    "rss_bytes": 1_000_000_000,
                                    "thread_count": 18,
                                    "cpu_time_seconds": 0.004,
                                }
                                for slot in (0, 1)
                            ],
                        }
                    ],
                    "physical_memory_bytes": 128_000_000_000,
                    "peak_combined_rss_bytes": 2_000_000_000,
                    "worker_high_water_rss_bytes": [
                        {
                            "slot": slot,
                            "pid": 10_000 + slot,
                            "process_peak_rss_bytes": 1_000_000_000,
                        }
                        for slot in (0, 1)
                    ],
                    "high_water_combined_rss_bytes": 2_000_000_000,
                    "peak_combined_thread_count": 36,
                    "swap_in_delta": 0,
                    "swap_out_delta": 0,
                    "barrier_release_monotonic_ns": release_ns,
                    "start_skew_seconds": 0.0,
                    "wave_seconds": 0.005,
                    "overlap_seconds": 0.0049,
                    "solo_tail_seconds": 0.0,
                },
                "mode_details": {"segments": 1},
            }
        )
    return {
        "schema_version": 1,
        "kind": shootout.CAMPAIGN_KIND + "_concurrency_history",
        "execution_mode": "concurrent",
        "wave_schedule_sha256": shootout.wave_schedule_sha256(),
        "worker_session_swap_telemetry": _zero_swap_session(),
        "entries": entries,
    }


def _valid_sequential_concurrency_history(output_dir) -> dict:
    history = _valid_concurrency_history(output_dir)
    history["execution_mode"] = "sequential_fallback"
    for entry in history["entries"]:
        entry["execution_mode"] = "sequential_fallback"
        reports = sorted(entry["reports"], key=lambda report: report["slot"])
        base_release = reports[0]["barrier_release_monotonic_ns"]
        segments = []
        for slot, report in enumerate(reports):
            release_ns = base_release + slot * 10_000_000
            report["barrier_release_monotonic_ns"] = release_ns
            report["started_monotonic_ns"] = release_ns + 100_000
            report["ended_monotonic_ns"] = release_ns + 5_000_000
            report["elapsed_seconds"] = 0.0049
            segment = deepcopy(entry["telemetry"])
            segment["samples"] = [
                {**segment["samples"][0], "monotonic_ns": release_ns + 200_000}
            ]
            segment["sample_count"] = 1
            segment["barrier_release_monotonic_ns"] = release_ns
            segment["start_skew_seconds"] = 0.0
            segment["wave_seconds"] = 0.005
            segment["overlap_seconds"] = 0.0
            segment["solo_tail_seconds"] = 0.0
            segments.append(segment)
        combined_samples = [
            sample for segment in segments for sample in segment["samples"]
        ]
        aggregate = {
            "sample_count": len(combined_samples),
            "samples": combined_samples,
            "physical_memory_bytes": segments[0]["physical_memory_bytes"],
            "peak_combined_rss_bytes": 2_000_000_000,
            "worker_high_water_rss_bytes": deepcopy(
                segments[0]["worker_high_water_rss_bytes"]
            ),
            "high_water_combined_rss_bytes": 2_000_000_000,
            "peak_combined_thread_count": 36,
            "swap_in_delta": 0,
            "swap_out_delta": 0,
            "wave_seconds": 0.01,
            "start_skew_seconds": 0.0,
            "overlap_seconds": 0.0,
            "solo_tail_seconds": 0.0,
        }
        entry["reports"] = reports
        entry["telemetry"] = aggregate
        entry["mode_details"] = {
            "segments": 2,
            "segment_telemetry": segments,
        }
    return history


def test_shootout_policy_specialization_does_not_mutate_shared_runner():
    assert shootout.screen is not shared_screen
    assert set(shootout.screen.ARM_SPECS) == {"baseline", "auto"}


def test_runtime_lock_collects_common_and_optional_comparator_packages():
    assert set(shootout.screen.PACKAGE_DISTRIBUTIONS) == set(
        shootout.REUSED_COMMON_PACKAGE_DISTRIBUTIONS
    ) | set(shootout.REUSED_OPTIONAL_COMPARATOR_DISTRIBUTIONS)
    assert set(shared_screen.ARM_SPECS) == {
        "baseline",
        "auto",
        "ts4",
        "ordinal",
        "onehot",
        "linear",
    }
    assert shared_screen.BASELINE_CONFIG["iterations"] == 1_000
    assert shootout.screen.BASELINE_CONFIG["iterations"] == 10_000


def test_shootout_jobs_have_disjoint_result_paths_within_and_across_waves(
    tmp_path,
):
    jobs_by_key = {shootout._public_job_key(job): job for job in _fake_jobs()}
    paths = {
        key: shootout.screen._result_path(tmp_path, job)
        for key, job in jobs_by_key.items()
    }

    assert len(jobs_by_key) == len(paths) == shootout.EXPECTED_JOBS
    assert len(set(paths.values())) == shootout.EXPECTED_JOBS
    for wave in shootout.expected_wave_schedule():
        wave_paths = [
            paths[_public_key(job)]
            for job in wave["jobs"]
        ]
        assert len(set(wave_paths)) == shootout.WORKER_COUNT


def test_wave_resume_archives_both_members_when_one_partner_is_missing(
    tmp_path, monkeypatch
):
    jobs = _fake_jobs()
    jobs_by_key = {shootout._public_job_key(job): job for job in jobs}
    waves = shootout.expected_wave_schedule()

    def result_path(key):
        return shootout.screen._result_path(tmp_path, jobs_by_key[key])

    complete_paths = [result_path(_public_key(item)) for item in waves[0]["jobs"]]
    partial_path = result_path(_public_key(waves[1]["jobs"][0]))
    for path in (*complete_paths, partial_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic valid result")
    journal = _valid_concurrency_history(tmp_path)
    journal["entries"] = journal["entries"][:1]
    for report in journal["entries"][0]["reports"]:
        path = result_path(_public_key(report))
        report["result_sha256"] = shootout.screen.hardened._sha256_file(path)
        report["result_size_bytes"] = path.stat().st_size
    (tmp_path / shootout.CONCURRENCY_HISTORY_FILENAME).write_text(
        shootout.json.dumps(journal), encoding="utf-8"
    )

    monkeypatch.setattr(
        shootout.screen,
        "_cached_result_issue",
        lambda path, _job: None if path.is_file() else "missing",
    )
    state = shootout.prepare_wave_resume(tmp_path, jobs, waves, resume=True)

    assert state["reusable_wave_indices"] == []
    assert state["pending_wave_indices"] == list(range(shootout.EXPECTED_WAVES))
    assert state["invalidated_wave_indices"] == [0, 1]
    assert all(not path.exists() for path in complete_paths)
    assert not partial_path.exists()
    archived = list((tmp_path / "resume_invalidated").rglob("results.pkl"))
    assert len(archived) == 3
    assert all(path.read_bytes() == b"synthetic valid result" for path in archived)


def test_wave_resume_restarts_the_journal_before_replacement_wave_zero(tmp_path):
    jobs = _fake_jobs()
    jobs_by_key = {shootout._public_job_key(job): job for job in jobs}
    waves = shootout.expected_wave_schedule()
    journal = _valid_concurrency_history(tmp_path)
    journal["entries"] = journal["entries"][:1]
    for item in waves[0]["jobs"]:
        path = shootout.screen._result_path(
            tmp_path, jobs_by_key[_public_key(item)]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"prior process result")
    journal_path = tmp_path / shootout.CONCURRENCY_HISTORY_FILENAME
    journal_path.write_text(shootout.json.dumps(journal), encoding="utf-8")

    first = shootout.prepare_wave_resume(tmp_path, jobs, waves, resume=True)

    assert first["invalidated_wave_indices"] == [0]
    restarted = shootout.json.loads(journal_path.read_text(encoding="utf-8"))
    assert restarted == shootout._empty_concurrency_history("concurrent")
    first_history = shootout.json.loads(
        (tmp_path / shootout.screen.RESUME_HISTORY_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    archived_journal = tmp_path / next(
        path
        for path in first_history[0]["archived_campaign_artifacts"]
        if path.endswith(shootout.CONCURRENCY_HISTORY_FILENAME)
    )
    assert shootout.json.loads(archived_journal.read_text(encoding="utf-8")) == journal

    # Simulate a clean interruption after resume preparation but before wave zero.
    second = shootout.prepare_wave_resume(tmp_path, jobs, waves, resume=True)

    assert second["invalidated_wave_indices"] == []
    assert shootout.json.loads(journal_path.read_text(encoding="utf-8")) == (
        shootout._empty_concurrency_history("concurrent")
    )
    loaded = shootout._load_concurrency_history(
        tmp_path, "concurrent", set(second["invalidated_wave_indices"])
    )
    assert loaded["entries"] == []
    resume_history = shootout.json.loads(
        (tmp_path / shootout.screen.RESUME_HISTORY_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert len(resume_history) == 2
    shootout.validate_resume_history(resume_history, tmp_path)


def test_wave_resume_validates_the_complete_schedule_before_mutation(
    tmp_path, monkeypatch
):
    jobs = _fake_jobs()
    jobs_by_key = {shootout._public_job_key(job): job for job in jobs}
    waves = deepcopy(shootout.expected_wave_schedule())
    source = shootout.screen._result_path(
        tmp_path, jobs_by_key[_public_key(waves[0]["jobs"][0])]
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"must remain in place")
    waves[-1]["wave_index"] = 0
    monkeypatch.setattr(
        shootout.screen,
        "_cached_result_issue",
        lambda path, _job: None if path.is_file() else "missing",
    )

    with pytest.raises(RuntimeError):
        shootout.prepare_wave_resume(tmp_path, jobs, waves, resume=True)

    assert source.read_bytes() == b"must remain in place"
    assert not (tmp_path / "resume_invalidated").exists()
    assert not (tmp_path / shootout.screen.RESUME_HISTORY_FILENAME).exists()


def test_resume_never_decodes_an_unattested_pickle(tmp_path, monkeypatch):
    jobs = _fake_jobs()
    first = jobs[0]
    path = shootout.screen._result_path(tmp_path, first)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a trusted pickle")
    monkeypatch.setattr(
        shootout.screen,
        "_cached_result_issue",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unattested bytes were decoded")
        ),
    )

    state = shootout.prepare_wave_resume(
        tmp_path,
        jobs,
        shootout.expected_wave_schedule(),
        resume=True,
    )

    assert state["invalidated_wave_indices"]
    assert not path.exists()


def test_wave_resume_validates_existing_journal_before_result_mutation(
    tmp_path, monkeypatch
):
    jobs = _fake_jobs()
    jobs_by_key = {shootout._public_job_key(job): job for job in jobs}
    waves = shootout.expected_wave_schedule()
    source = shootout.screen._result_path(
        tmp_path, jobs_by_key[_public_key(waves[0]["jobs"][0])]
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"must remain in place")
    journal = _valid_concurrency_history(tmp_path)
    journal["entries"] = journal["entries"][:1]
    journal["entries"][0].pop("telemetry")
    (tmp_path / shootout.CONCURRENCY_HISTORY_FILENAME).write_text(
        shootout.json.dumps(journal), encoding="utf-8"
    )
    monkeypatch.setattr(
        shootout.screen,
        "_cached_result_issue",
        lambda path, _job: None if path.is_file() else "missing",
    )

    with pytest.raises(RuntimeError):
        shootout.prepare_wave_resume(tmp_path, jobs, waves, resume=True)

    assert source.read_bytes() == b"must remain in place"
    assert not (tmp_path / "resume_invalidated").exists()
    assert not (tmp_path / shootout.screen.RESUME_HISTORY_FILENAME).exists()


def test_wave_resume_rejects_journal_mode_before_result_mutation(tmp_path):
    jobs = _fake_jobs()
    waves = shootout.expected_wave_schedule()
    jobs_by_key = {shootout._public_job_key(job): job for job in jobs}
    source = shootout.screen._result_path(
        tmp_path, jobs_by_key[_public_key(waves[0]["jobs"][0])]
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"must remain in place")
    journal = {
        "schema_version": 1,
        "kind": shootout.CAMPAIGN_KIND + "_concurrency_history",
        "execution_mode": "sequential_fallback",
        "wave_schedule_sha256": shootout.wave_schedule_sha256(),
        "entries": [],
    }
    (tmp_path / shootout.CONCURRENCY_HISTORY_FILENAME).write_text(
        shootout.json.dumps(journal), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="execution mode|incompatible"):
        shootout.prepare_wave_resume(
            tmp_path,
            jobs,
            waves,
            resume=True,
            execution_mode="concurrent",
        )

    assert source.read_bytes() == b"must remain in place"
    assert not (tmp_path / "resume_invalidated").exists()


def test_wave_resume_semantically_validates_prior_history_before_mutation(tmp_path):
    jobs = _fake_jobs()
    waves = shootout.expected_wave_schedule()
    first = jobs[0]
    source = shootout.screen._result_path(tmp_path, first)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"first attempt")
    shootout.prepare_wave_resume(tmp_path, jobs, waves, resume=True)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"second attempt must remain")
    history_path = tmp_path / shootout.screen.RESUME_HISTORY_FILENAME
    history = shootout.json.loads(history_path.read_text(encoding="utf-8"))
    history[0]["wave_schedule_sha256"] = "0" * 64
    history_path.write_text(shootout.json.dumps(history), encoding="utf-8")

    with pytest.raises(RuntimeError, match="resume record"):
        shootout.prepare_wave_resume(tmp_path, jobs, waves, resume=True)

    assert source.read_bytes() == b"second attempt must remain"


def test_wave_resume_validates_prior_warmup_before_result_mutation(tmp_path):
    jobs = _fake_jobs()
    source = shootout.screen._result_path(tmp_path, jobs[0])
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"must remain in place")
    (tmp_path / shootout.screen.WARMUP_HISTORY_FILENAME).write_text(
        "[{}]", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="warmup"):
        shootout.prepare_wave_resume(
            tmp_path, jobs, shootout.expected_wave_schedule(), resume=True
        )

    assert source.read_bytes() == b"must remain in place"
    assert not (tmp_path / "resume_invalidated").exists()


def test_resume_history_rejects_a_missing_referenced_archive(tmp_path):
    jobs = _fake_jobs()
    source = shootout.screen._result_path(tmp_path, jobs[0])
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"archived result")
    shootout.prepare_wave_resume(
        tmp_path, jobs, shootout.expected_wave_schedule(), resume=True
    )
    history = shootout.json.loads(
        (tmp_path / shootout.screen.RESUME_HISTORY_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    archived = tmp_path / history[0]["invalidated_waves"][0]["members"][0]["path"]
    archived.unlink()

    with pytest.raises(RuntimeError, match="archive is missing"):
        shootout.validate_resume_history(history, tmp_path)


def test_wave_resume_archives_all_generated_analysis_outputs(tmp_path):
    jobs = _fake_jobs()
    for name in shootout.ANALYSIS_OUTPUT_FILENAMES:
        (tmp_path / name).write_text(f"stale {name}", encoding="utf-8")

    shootout.prepare_wave_resume(
        tmp_path, jobs, shootout.expected_wave_schedule(), resume=True
    )

    history = shootout.json.loads(
        (tmp_path / shootout.screen.RESUME_HISTORY_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    archived = {
        (tmp_path / relative).name
        for relative in history[0]["archived_campaign_artifacts"]
    }
    assert archived == set(shootout.ANALYSIS_OUTPUT_FILENAMES)
    assert all(not (tmp_path / name).exists() for name in archived)


@pytest.mark.parametrize("artifact", ["resume_history", "completion", "payload"])
def test_wave_resume_rejects_nonregular_campaign_artifacts_before_mutation(
    tmp_path, monkeypatch, artifact
):
    jobs = _fake_jobs()
    jobs_by_key = {shootout._public_job_key(job): job for job in jobs}
    waves = shootout.expected_wave_schedule()
    source = shootout.screen._result_path(
        tmp_path, jobs_by_key[_public_key(waves[0]["jobs"][0])]
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"must remain in place")
    victim = tmp_path / "victim.json"
    victim.write_text("[]", encoding="utf-8")
    artifact_names = {
        "resume_history": shootout.screen.RESUME_HISTORY_FILENAME,
        "completion": shootout.screen.COMPLETION_ATTESTATION_FILENAME,
        "payload": shootout.screen.ANALYSIS_PAYLOAD_FILENAME,
    }
    artifact_path = tmp_path / artifact_names[artifact]
    artifact_path.symlink_to(victim)
    monkeypatch.setattr(
        shootout.screen,
        "_cached_result_issue",
        lambda path, _job: None if path.is_file() else "missing",
    )

    with pytest.raises(RuntimeError):
        shootout.prepare_wave_resume(tmp_path, jobs, waves, resume=True)

    assert source.read_bytes() == b"must remain in place"
    assert artifact_path.is_symlink()
    assert victim.read_text(encoding="utf-8") == "[]"
    assert not (tmp_path / "resume_invalidated").exists()


def test_wave_resume_rejects_symlinked_result_parent_before_mutation(tmp_path):
    jobs = _fake_jobs()
    waves = shootout.expected_wave_schedule()
    jobs_by_key = {shootout._public_job_key(job): job for job in jobs}
    source = shootout.screen._result_path(
        tmp_path, jobs_by_key[_public_key(waves[0]["jobs"][0])]
    )
    external = tmp_path.parent / f"{tmp_path.name}-external-result"
    external.mkdir()
    victim = external / source.name
    victim.write_bytes(b"must not be moved")
    source.parent.parent.mkdir(parents=True, exist_ok=True)
    source.parent.symlink_to(external, target_is_directory=True)

    with pytest.raises(RuntimeError, match="parent|outside"):
        shootout.prepare_wave_resume(tmp_path, jobs, waves, resume=True)

    assert victim.read_bytes() == b"must not be moved"
    assert source.parent.is_symlink()
    assert not (tmp_path / "resume_invalidated").exists()


def test_wave_resume_rejects_symlinked_worker_scratch_before_mutation(tmp_path):
    jobs = _fake_jobs()
    waves = shootout.expected_wave_schedule()
    jobs_by_key = {shootout._public_job_key(job): job for job in jobs}
    source = shootout.screen._result_path(
        tmp_path, jobs_by_key[_public_key(waves[0]["jobs"][0])]
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"must remain in place")
    external = tmp_path.parent / f"{tmp_path.name}-external-scratch"
    external.mkdir()
    (tmp_path / "worker_scratch").symlink_to(external, target_is_directory=True)

    with pytest.raises(RuntimeError, match="scratch|directory"):
        shootout.prepare_wave_resume(tmp_path, jobs, waves, resume=True)

    assert source.read_bytes() == b"must remain in place"
    assert (tmp_path / "worker_scratch").is_symlink()
    assert list(external.iterdir()) == []
    assert not (tmp_path / "resume_invalidated").exists()


def test_wave_resume_requires_the_exact_runtime_job_grid(tmp_path):
    jobs = _fake_jobs()
    jobs[-1].task.dataset = "invented_dataset"

    with pytest.raises(RuntimeError, match="lookup|grid"):
        shootout.prepare_wave_resume(
            tmp_path,
            jobs,
            shootout.expected_wave_schedule(),
            resume=False,
        )

    assert list(tmp_path.iterdir()) == []


def test_resume_manifest_mismatch_fails_before_result_mutation(tmp_path):
    output_dir = tmp_path / "campaign"
    output_dir.mkdir()
    original = {"created_at_utc": "first", "execution_mode": "concurrent"}
    shootout.write_or_validate_manifest(output_dir, original, resume=False)
    result = output_dir / "experiments" / "data" / "job" / "results.pkl"
    result.parent.mkdir(parents=True)
    result.write_bytes(b"untouched")

    changed = {"created_at_utc": "second", "execution_mode": "sequential_fallback"}
    with pytest.raises(RuntimeError, match="execution_mode"):
        shootout.write_or_validate_manifest(output_dir, changed, resume=True)

    assert result.read_bytes() == b"untouched"
    assert not (output_dir / "resume_invalidated").exists()


def test_concurrency_history_requires_all_39_two_report_barrier_waves(tmp_path):
    history = _valid_concurrency_history(tmp_path)

    shootout.validate_concurrency_history(
        history, execution_mode="concurrent", output_dir=tmp_path
    )


def test_production_failure_marks_attempt_invalid_before_shutdown(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "campaign"
    output_dir.mkdir()
    shootout.screen.hardened._atomic_write_json(
        output_dir / shootout.screen.MANIFEST_FILENAME,
        {
            "protocol_sha256": shootout.screen.protocol_sha256(),
            "preflight_report_sha256": "1" * 64,
            "execution_grid_sha256": "2" * 64,
            "source": {"git_head": "3" * 40},
        },
    )
    workers = [{"slot": 0}, {"slot": 1}]
    stopped = []
    monkeypatch.setattr(shootout, "_start_workers", lambda _root: (workers, []))
    monkeypatch.setattr(shootout, "_warm_workers", lambda _workers: [])
    monkeypatch.setattr(
        shootout, "_record_warmup_session", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        shootout,
        "_dispatch_runs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic worker failure")
        ),
    )
    monkeypatch.setattr(
        shootout,
        "_stop_workers",
        lambda observed, **_kwargs: stopped.append(observed),
    )

    with pytest.raises(RuntimeError, match="synthetic worker failure"):
        shootout.execute_production(
            output_dir,
            execution_mode="concurrent",
            pending_wave_indices=[0],
            invalidated_wave_indices=[],
        )

    marker = shootout.json.loads(
        (output_dir / shootout.INVALID_ATTEMPT_FILENAME).read_text(encoding="utf-8")
    )
    assert marker["kind"] == shootout.CAMPAIGN_KIND + "_invalid_attempt"
    assert marker["execution_mode"] == "concurrent"
    assert marker["error_type"] == "RuntimeError"
    assert marker["error"] == "synthetic worker failure"
    assert "fresh output namespace" in marker["recovery"]
    assert stopped == [workers]
    assert not (output_dir / shootout.CONCURRENCY_HISTORY_FILENAME).exists()


class _FaultInjectedShutdownProcess:
    def __init__(
        self,
        *,
        terminate_error=None,
        terminate_stops: bool = False,
        kill_stops: bool = True,
    ):
        self._alive = True
        self._exitcode = None
        self._pending_signal = None
        self.terminate_error = terminate_error
        self.terminate_stops = terminate_stops
        self.kill_stops = kill_stops
        self.calls = []

    def is_alive(self):
        self.calls.append("is_alive")
        return self._alive

    @property
    def exitcode(self):
        self.calls.append("exitcode")
        return self._exitcode

    def terminate(self):
        self.calls.append("terminate")
        if self.terminate_error is not None:
            raise self.terminate_error
        self._pending_signal = "terminate"

    def kill(self):
        self.calls.append("kill")
        self._pending_signal = "kill"

    def join(self, timeout):
        self.calls.append(("join", timeout))
        if self._pending_signal == "terminate" and self.terminate_stops:
            self._alive = False
            self._exitcode = -15
        if self._pending_signal == "kill" and self.kill_stops:
            self._alive = False
            self._exitcode = -9


class _FaultInjectedShutdownConnection:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _fault_injected_worker(process):
    connection = _FaultInjectedShutdownConnection()
    return {"slot": 0, "process": process, "connection": connection}, connection


class _FaultInjectedStartupConnection:
    def __init__(
        self,
        *,
        message=None,
        poll_error=None,
        recv_error=None,
        close_effective=True,
    ):
        self.message = message
        self.poll_error = poll_error
        self.recv_error = recv_error
        self.close_effective = close_effective
        self.closed = False

    def poll(self, _timeout):
        if self.poll_error is not None:
            raise self.poll_error
        return True

    def recv(self):
        if self.recv_error is not None:
            raise self.recv_error
        return self.message

    def close(self):
        if self.close_effective:
            self.closed = True


class _FaultInjectedStartupProcess:
    def __init__(self, slot, *, start_error=None, partial_start=False):
        self.slot = slot
        self.start_error = start_error
        self.partial_start = partial_start
        self._pid = None
        self._alive = False
        self._exitcode = None
        self.calls = []

    @property
    def pid(self):
        return self._pid

    @property
    def exitcode(self):
        return self._exitcode

    def start(self):
        self.calls.append("start")
        if self.start_error is None or self.partial_start:
            self._pid = 20_000 + self.slot
            self._alive = True
        if self.start_error is not None:
            raise self.start_error

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.calls.append("terminate")
        self._alive = False
        self._exitcode = -15

    def kill(self):
        self.calls.append("kill")
        self._alive = False
        self._exitcode = -9

    def join(self, timeout):
        self.calls.append(("join", timeout))


class _FaultInjectedStartupContext:
    def __init__(
        self,
        *,
        recv_error_slot=None,
        poll_error_slot=None,
        process_error_slot=None,
        start_error_slot=None,
        pipe_error_slot=None,
        unclosable_parent_slot=None,
        partial_start=False,
    ):
        self.recv_error_slot = recv_error_slot
        self.poll_error_slot = poll_error_slot
        self.process_error_slot = process_error_slot
        self.start_error_slot = start_error_slot
        self.pipe_error_slot = pipe_error_slot
        self.unclosable_parent_slot = unclosable_parent_slot
        self.partial_start = partial_start
        self.pipe_count = 0
        self.parents = []
        self.children = []
        self.processes = []

    def Pipe(self, *, duplex):
        assert duplex is True
        slot = self.pipe_count
        self.pipe_count += 1
        if slot == self.pipe_error_slot:
            raise OSError("synthetic Pipe failure")
        parent = _FaultInjectedStartupConnection(
            message={
                "type": "ready",
                "slot": slot,
                "pid": 20_000 + slot,
                "start_method": "spawn",
                "child_cpus": shootout.EXPECTED_CHILD_CPUS,
            },
            poll_error=(
                KeyboardInterrupt() if slot == self.poll_error_slot else None
            ),
            recv_error=EOFError("synthetic readiness EOF")
            if slot == self.recv_error_slot
            else None,
            close_effective=slot != self.unclosable_parent_slot,
        )
        child = _FaultInjectedStartupConnection()
        self.parents.append(parent)
        self.children.append(child)
        return parent, child

    def Process(self, *, target, args, name):
        del target, name
        slot = args[0]
        if slot == self.process_error_slot:
            raise RuntimeError("synthetic Process construction failure")
        process = _FaultInjectedStartupProcess(
            slot,
            start_error=(
                RuntimeError("synthetic Process.start failure")
                if slot == self.start_error_slot
                else None
            ),
            partial_start=self.partial_start,
        )
        self.processes.append(process)
        return process


def _assert_startup_context_quiesced(context):
    assert all(not process.is_alive() for process in context.processes)
    assert all(parent.closed for parent in context.parents)
    assert all(child.closed for child in context.children)


def test_worker_readiness_eof_stops_every_spawned_worker_and_closes_pipes(
    tmp_path, monkeypatch
):
    context = _FaultInjectedStartupContext(recv_error_slot=1)
    monkeypatch.setattr(shootout.mp, "get_context", lambda _method: context)

    with pytest.raises(EOFError, match="readiness EOF"):
        shootout._start_workers(tmp_path)

    assert len(context.processes) == shootout.WORKER_COUNT
    _assert_startup_context_quiesced(context)


def test_keyboard_interrupt_during_readiness_aborts_without_fallback_or_leak(
    tmp_path, monkeypatch
):
    context = _FaultInjectedStartupContext(poll_error_slot=0)
    monkeypatch.setattr(shootout.mp, "get_context", lambda _method: context)

    with pytest.raises(KeyboardInterrupt):
        shootout.run_preflight(tmp_path)

    _assert_startup_context_quiesced(context)
    assert not (tmp_path / shootout.PREFLIGHT_REPORT_FILENAME).exists()


def test_worker_startup_fails_closed_when_pipe_closure_cannot_be_confirmed(
    tmp_path, monkeypatch
):
    context = _FaultInjectedStartupContext(
        recv_error_slot=0,
        unclosable_parent_slot=0,
    )
    monkeypatch.setattr(shootout.mp, "get_context", lambda _method: context)

    with pytest.raises(
        RuntimeError, match="worker startup cleanup could not be confirmed"
    ) as error:
        shootout._start_workers(tmp_path)

    assert isinstance(error.value.__cause__, EOFError)
    assert all(not process.is_alive() for process in context.processes)
    assert context.parents[0].closed is False
    assert all(child.closed for child in context.children)


@pytest.mark.parametrize(
    ("fault", "expected"),
    [
        ("pipe", "Pipe failure"),
        ("construct", "Process construction failure"),
        ("start", "Process.start failure"),
    ],
)
def test_worker_construction_failures_close_orphan_endpoints_and_workers(
    tmp_path, monkeypatch, fault, expected
):
    context = _FaultInjectedStartupContext(
        pipe_error_slot=1 if fault == "pipe" else None,
        process_error_slot=1 if fault == "construct" else None,
        start_error_slot=1 if fault == "start" else None,
        partial_start=fault == "start",
    )
    monkeypatch.setattr(shootout.mp, "get_context", lambda _method: context)

    with pytest.raises((OSError, RuntimeError), match=expected):
        shootout._start_workers(tmp_path)

    _assert_startup_context_quiesced(context)


def test_forced_worker_shutdown_escalates_to_kill_and_confirms_exit():
    process = _FaultInjectedShutdownProcess(
        terminate_error=KeyboardInterrupt(), kill_stops=True
    )
    worker, connection = _fault_injected_worker(process)

    shootout._stop_workers([worker], force=True)

    assert process.calls == [
        "is_alive",
        "terminate",
        ("join", 10.0),
        "is_alive",
        "kill",
        ("join", 10.0),
        "is_alive",
        "exitcode",
    ]
    assert connection.closed


def test_forced_worker_shutdown_fails_closed_when_kill_does_not_quiesce():
    process = _FaultInjectedShutdownProcess(kill_stops=False)
    worker, connection = _fault_injected_worker(process)

    with pytest.raises(RuntimeError, match="failed to quiesce"):
        shootout._stop_workers([worker], force=True)

    assert "terminate" in process.calls
    assert "kill" in process.calls
    assert process.calls[-2:] == ["is_alive", "exitcode"]
    assert connection.closed


def test_clean_keyboard_interrupt_remains_resumable(tmp_path, monkeypatch):
    output_dir = tmp_path / "campaign"
    output_dir.mkdir()
    workers = [{"slot": 0}, {"slot": 1}]
    stopped = []
    monkeypatch.setattr(shootout, "_start_workers", lambda _root: (workers, []))
    monkeypatch.setattr(shootout, "_warm_workers", lambda _workers: [])
    monkeypatch.setattr(
        shootout, "_record_warmup_session", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        shootout,
        "_dispatch_runs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(
        shootout,
        "_stop_workers",
        lambda observed, **kwargs: stopped.append((observed, kwargs)),
    )

    with pytest.raises(KeyboardInterrupt):
        shootout.execute_production(
            output_dir,
            execution_mode="concurrent",
            pending_wave_indices=[0],
            invalidated_wave_indices=[],
        )

    assert not (output_dir / shootout.INVALID_ATTEMPT_FILENAME).exists()
    assert stopped == [(workers, {"force": True})]


def test_keyboard_interrupt_with_unquiesced_worker_marks_attempt_invalid(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "campaign"
    output_dir.mkdir()
    shootout.screen.hardened._atomic_write_json(
        output_dir / shootout.screen.MANIFEST_FILENAME,
        {
            "protocol_sha256": shootout.screen.protocol_sha256(),
            "preflight_report_sha256": "1" * 64,
            "execution_grid_sha256": "2" * 64,
            "source": {"git_head": "3" * 40},
        },
    )
    process = _FaultInjectedShutdownProcess(kill_stops=False)
    worker, connection = _fault_injected_worker(process)
    monkeypatch.setattr(shootout, "_start_workers", lambda _root: ([worker], []))
    monkeypatch.setattr(shootout, "_warm_workers", lambda _workers: [])
    monkeypatch.setattr(
        shootout, "_record_warmup_session", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        shootout,
        "_dispatch_runs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(RuntimeError, match="attempt is invalid"):
        shootout.execute_production(
            output_dir,
            execution_mode="concurrent",
            pending_wave_indices=[0],
            invalidated_wave_indices=[],
        )

    marker = shootout.json.loads(
        (output_dir / shootout.INVALID_ATTEMPT_FILENAME).read_text(encoding="utf-8")
    )
    assert marker["error_type"] == "RuntimeError"
    assert "failed to quiesce" in marker["error"]
    assert connection.closed


def _test_process_identity(pid: int, created: int = 1) -> dict[str, int]:
    return {"pid": pid, "create_time_us": created}


def test_owner_lock_inode_is_stable_while_state_is_atomically_replaced(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        shootout,
        "_read_process_identity",
        lambda pid: _test_process_identity(pid),
    )
    handle = shootout.acquire_owner_session(
        tmp_path, resume=False, phase="preflight"
    )
    with pytest.raises(RuntimeError, match="owned by another process"):
        shootout._open_owner_lock(tmp_path)
    lock_before = os.fstat(handle["lock_fd"])
    state_before = (tmp_path / shootout.OWNER_STATE_FILENAME).lstat()

    shootout.finalize_owner_session(handle, "preflight_only")

    lock_after = (tmp_path / shootout.OWNER_LOCK_FILENAME).lstat()
    state_after = (tmp_path / shootout.OWNER_STATE_FILENAME).lstat()
    assert (lock_before.st_dev, lock_before.st_ino) == (
        lock_after.st_dev,
        lock_after.st_ino,
    )
    assert (state_before.st_dev, state_before.st_ino) != (
        state_after.st_dev,
        state_after.st_ino,
    )
    shootout.release_owner_session(handle)


def test_owner_paths_reject_lock_and_state_symlinks(tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("not campaign state", encoding="utf-8")

    lock_campaign = tmp_path / "lock-campaign"
    lock_campaign.mkdir()
    (lock_campaign / shootout.OWNER_LOCK_FILENAME).symlink_to(outside)
    with pytest.raises(RuntimeError, match="owner lock"):
        shootout.acquire_owner_session(
            lock_campaign, resume=False, phase="preflight"
        )

    state_campaign = tmp_path / "state-campaign"
    state_campaign.mkdir()
    (state_campaign / shootout.OWNER_LOCK_FILENAME).touch()
    (state_campaign / shootout.OWNER_STATE_FILENAME).symlink_to(outside)
    with pytest.raises(RuntimeError, match="symbolic|owner session state"):
        shootout.acquire_owner_session(
            state_campaign, resume=True, phase="resume_validation"
        )


def test_resume_refuses_a_live_prior_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(
        shootout,
        "_read_process_identity",
        lambda pid: _test_process_identity(pid),
    )
    first = shootout.acquire_owner_session(
        tmp_path, resume=False, phase="preflight"
    )
    shootout.release_owner_session(first)
    state_path = tmp_path / shootout.OWNER_STATE_FILENAME
    state = shootout._read_owner_state(state_path, tmp_path)
    state["sessions"][-1]["parent"] = _test_process_identity(8_000, 8)
    state["sessions"][-1]["worker_cohorts"] = [
        {
            "phase": "preflight",
            "bound_at_utc": "2026-07-15T12:00:00+00:00",
            "quiesced_at_utc": None,
            "workers": [
                {"slot": 0, **_test_process_identity(9_001, 9)},
                {"slot": 1, **_test_process_identity(9_002, 9)},
            ],
        }
    ]
    shootout.validate_owner_state(state, tmp_path)
    shootout.screen.hardened._atomic_write_json(state_path, state)
    monkeypatch.setattr(
        shootout,
        "_process_identity_is_live",
        lambda identity: identity["pid"] == 9_001,
    )

    with pytest.raises(RuntimeError, match="worker slot 0 PID 9001"):
        shootout.acquire_owner_session(
            tmp_path, resume=True, phase="resume_validation"
        )

    assert len(shootout._read_owner_state(state_path, tmp_path)["sessions"]) == 1


def test_resume_accepts_only_after_prior_identities_are_dead(tmp_path, monkeypatch):
    monkeypatch.setattr(
        shootout,
        "_read_process_identity",
        lambda pid: _test_process_identity(pid),
    )
    first = shootout.acquire_owner_session(
        tmp_path, resume=False, phase="preflight"
    )
    shootout.release_owner_session(first)
    monkeypatch.setattr(
        shootout, "_process_identity_is_live", lambda _identity: False
    )

    resumed = shootout.acquire_owner_session(
        tmp_path, resume=True, phase="resume_validation"
    )

    state = shootout._read_owner_state(
        tmp_path / shootout.OWNER_STATE_FILENAME, tmp_path
    )
    assert len(state["sessions"]) == 2
    assert state["sessions"][0]["terminal_status"] == "abandoned_after_crash"
    assert state["sessions"][1]["session_id"] == resumed["session_id"]
    malformed = deepcopy(state)
    malformed["sessions"][0].update(
        {
            "state": "active",
            "phase": "preflight",
            "finalized_at_utc": None,
            "terminal_status": None,
        }
    )
    with pytest.raises(RuntimeError, match="only the latest"):
        shootout.validate_owner_state(malformed, tmp_path)
    shootout.finalize_owner_session(resumed, "invalid")
    shootout.release_owner_session(resumed)


def test_process_start_identity_distinguishes_pid_reuse(monkeypatch):
    recorded = _test_process_identity(4_242, 100)
    monkeypatch.setattr(
        shootout,
        "_read_process_identity",
        lambda _pid: _test_process_identity(4_242, 200),
    )
    assert shootout._process_identity_is_live(recorded) is False

    monkeypatch.setattr(
        shootout,
        "_read_process_identity",
        lambda _pid: _test_process_identity(4_242, 100),
    )
    assert shootout._process_identity_is_live(recorded) is True


def test_invalid_recovery_requires_finalized_unlocked_owner_binding(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        shootout,
        "_read_process_identity",
        lambda pid: _test_process_identity(pid),
    )
    handle = shootout.acquire_owner_session(
        tmp_path, resume=False, phase="preflight"
    )
    manifest_path = tmp_path / shootout.screen.MANIFEST_FILENAME
    manifest_path.write_text('{"synthetic": true}\n', encoding="utf-8")
    manifest_sha256 = shootout.screen.hardened._sha256_file(manifest_path)
    shootout.bind_owner_manifest(
        handle, execution_mode="concurrent", manifest_path=manifest_path
    )
    marker = {"manifest_sha256": manifest_sha256}
    shootout.screen.hardened._atomic_write_json(
        tmp_path / shootout.INVALID_ATTEMPT_FILENAME, marker
    )
    shootout.finalize_owner_session(handle, "invalid")

    with pytest.raises(RuntimeError, match="owned by another process"):
        shootout.validate_invalid_owner_session(tmp_path, marker)

    shootout.release_owner_session(handle)
    assert shootout.validate_invalid_owner_session(
        tmp_path, marker
    ) == shootout.screen.hardened._sha256_file(
        tmp_path / shootout.OWNER_STATE_FILENAME
    )

    shootout.screen.hardened._atomic_write_json(
        tmp_path / shootout.INVALID_ATTEMPT_FILENAME,
        {**marker, "changed": True},
    )
    with pytest.raises(RuntimeError, match="not bound"):
        shootout.validate_invalid_owner_session(tmp_path, marker)


def test_production_issues_no_worker_command_before_durable_owner_binding(
    tmp_path, monkeypatch
):
    shootout.screen.hardened._atomic_write_json(
        tmp_path / shootout.screen.MANIFEST_FILENAME,
        {
            "protocol_sha256": shootout.screen.protocol_sha256(),
            "preflight_report_sha256": "1" * 64,
            "execution_grid_sha256": "2" * 64,
            "source": {"git_head": "3" * 40},
        },
    )
    workers = [{"slot": 0}, {"slot": 1}]
    calls = []
    monkeypatch.setattr(
        shootout,
        "_start_workers",
        lambda _root: (calls.append("start") or workers, []),
    )
    monkeypatch.setattr(
        shootout,
        "bind_owner_workers",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("durable bind failed")),
    )
    monkeypatch.setattr(
        shootout,
        "_warm_workers",
        lambda _workers: calls.append("warmup"),
    )
    monkeypatch.setattr(
        shootout,
        "_stop_workers",
        lambda _workers, **kwargs: calls.append(f"stop:{kwargs['force']}"),
    )
    monkeypatch.setattr(
        shootout,
        "_close_production_swap_session",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="durable bind failed"):
        shootout.execute_production(
            tmp_path,
            execution_mode="concurrent",
            pending_wave_indices=[0],
            invalidated_wave_indices=[],
            owner_session={"session_id": "synthetic"},
        )

    assert calls == ["start", "stop:True"]


def test_sequential_failure_emits_supported_fresh_start_guidance(tmp_path):
    output_dir = tmp_path / "campaign"
    output_dir.mkdir()
    shootout.screen.hardened._atomic_write_json(
        output_dir / shootout.screen.MANIFEST_FILENAME,
        {
            "protocol_sha256": shootout.screen.protocol_sha256(),
            "preflight_report_sha256": "1" * 64,
            "execution_grid_sha256": "2" * 64,
            "source": {"git_head": "3" * 40},
        },
    )

    shootout._write_invalid_attempt(
        output_dir,
        execution_mode="sequential_fallback",
        error=RuntimeError("synthetic failure"),
    )

    marker = shootout.json.loads(
        (output_dir / shootout.INVALID_ATTEMPT_FILENAME).read_text(encoding="utf-8")
    )
    assert "without a recovery flag" in marker["recovery"]
    assert "--sequential-recovery-from" not in marker["recovery"]


def test_worker_startup_failure_marks_concurrent_attempt_invalid(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "campaign"
    output_dir.mkdir()
    shootout.screen.hardened._atomic_write_json(
        output_dir / shootout.screen.MANIFEST_FILENAME,
        {
            "protocol_sha256": shootout.screen.protocol_sha256(),
            "preflight_report_sha256": "1" * 64,
            "execution_grid_sha256": "2" * 64,
            "source": {"git_head": "3" * 40},
        },
    )
    monkeypatch.setattr(
        shootout,
        "_start_workers",
        lambda _root: (_ for _ in ()).throw(RuntimeError("readiness failed")),
    )
    monkeypatch.setattr(shootout, "_stop_workers", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="readiness failed"):
        shootout.execute_production(
            output_dir,
            execution_mode="concurrent",
            pending_wave_indices=list(range(shootout.EXPECTED_WAVES)),
            invalidated_wave_indices=[],
        )

    marker = shootout.json.loads(
        (output_dir / shootout.INVALID_ATTEMPT_FILENAME).read_text(encoding="utf-8")
    )
    assert marker["error_type"] == "RuntimeError"
    assert marker["error"] == "readiness failed"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_wave",
        "duplicate_wave",
        "missing_partner",
        "wrong_slot",
        "failed_report",
        "escaped_result",
        "wrong_mode",
        "wrong_schedule_digest",
        "wrong_entry_dataset",
        "wrong_entry_mode",
        "missing_telemetry",
        "production_start_skew",
        "production_swap",
        "worker_session_swap",
        "production_memory",
        "wrong_result_count",
        "wrong_child_count",
        "deadline_hit",
        "wrong_candidate_count",
        "nonfinite_elapsed",
        "wrong_partner",
        "wrong_report_schedule_digest",
        "worker_pid_changed",
        "nonprefix_wave",
    ],
)
def test_concurrency_history_rejects_incomplete_or_mismatched_barriers(
    tmp_path, mutation
):
    history = _valid_concurrency_history(tmp_path)
    if mutation == "missing_wave":
        history["entries"].pop()
    elif mutation == "duplicate_wave":
        history["entries"][-1] = deepcopy(history["entries"][0])
    elif mutation == "missing_partner":
        history["entries"][0]["reports"].pop()
    elif mutation == "wrong_slot":
        history["entries"][0]["reports"][0]["slot"] = 1
    elif mutation == "failed_report":
        history["entries"][0]["reports"][0]["status"] = "error"
    elif mutation == "escaped_result":
        history["entries"][0]["reports"][0]["result_path"] = str(
            tmp_path.parent / "escaped" / "results.pkl"
        )
    elif mutation == "wrong_mode":
        history["execution_mode"] = "sequential_fallback"
    elif mutation == "wrong_schedule_digest":
        history["wave_schedule_sha256"] = "0" * 64
    elif mutation == "wrong_entry_dataset":
        history["entries"][0]["dataset"] = "diamonds"
    elif mutation == "wrong_entry_mode":
        history["entries"][0]["execution_mode"] = "sequential_fallback"
    elif mutation == "missing_telemetry":
        history["entries"][0].pop("telemetry")
    elif mutation == "production_start_skew":
        history["entries"][0]["telemetry"]["start_skew_seconds"] = 1.01
    elif mutation == "production_swap":
        history["entries"][0]["telemetry"]["swap_out_delta"] = 1
    elif mutation == "worker_session_swap":
        session = history["worker_session_swap_telemetry"]
        session["samples"][-1]["swap_out_bytes"] = 1
        session["swap_out_delta"] = 1
    elif mutation == "production_memory":
        telemetry = history["entries"][0]["telemetry"]
        telemetry["peak_combined_rss_bytes"] = (
            0.8 * telemetry["physical_memory_bytes"]
        )
    elif mutation == "wrong_result_count":
        history["entries"][0]["reports"][0]["result_count"] = 0
    elif mutation == "wrong_child_count":
        history["entries"][0]["reports"][0]["child_count"] = 7
    elif mutation == "deadline_hit":
        history["entries"][0]["reports"][0]["deadline_hit"] = True
    elif mutation == "wrong_candidate_count":
        report = next(
            item
            for item in history["entries"][0]["reports"]
            if item["key"]["arm"] == "A10"
        )
        report["auto_candidate_fit_count"] = 23
    elif mutation == "nonfinite_elapsed":
        history["entries"][0]["reports"][0]["elapsed_seconds"] = math.inf
    elif mutation == "wrong_partner":
        history["entries"][0]["reports"][0]["partner_key"] = deepcopy(
            history["entries"][1]["reports"][0]["key"]
        )
    elif mutation == "wrong_report_schedule_digest":
        history["entries"][0]["reports"][0]["wave_schedule_sha256"] = "0" * 64
    elif mutation == "worker_pid_changed":
        report = history["entries"][1]["reports"][0]
        report["pid"] += 100
        telemetry = history["entries"][1]["telemetry"]
        high_water = next(
            item
            for item in telemetry["worker_high_water_rss_bytes"]
            if item["slot"] == report["slot"]
        )
        high_water["pid"] = report["pid"]
    else:
        history["entries"] = [history["entries"][1]]

    with pytest.raises(RuntimeError):
        shootout.validate_concurrency_history(
            history, execution_mode="concurrent", output_dir=tmp_path
        )


@pytest.mark.parametrize(
    "mutation", ["bogus_segment_type", "aggregate_shorter_than_reports", "extra_mode_key"]
)
def test_sequential_history_reconciles_segment_telemetry(tmp_path, mutation):
    history = _valid_sequential_concurrency_history(tmp_path)
    first = history["entries"][0]
    if mutation == "bogus_segment_type":
        first["mode_details"]["segment_telemetry"][0] = "not telemetry"
    elif mutation == "aggregate_shorter_than_reports":
        first["telemetry"]["wave_seconds"] = 0.001
    else:
        first["mode_details"]["unexpected"] = True

    with pytest.raises(RuntimeError, match="sequential fallback segments"):
        shootout.validate_concurrency_history(
            history,
            execution_mode="sequential_fallback",
            output_dir=tmp_path,
        )


def test_valid_sequential_history_reconciles_both_segments(tmp_path):
    shootout.validate_concurrency_history(
        _valid_sequential_concurrency_history(tmp_path),
        execution_mode="sequential_fallback",
        output_dir=tmp_path,
    )


def test_shootout_schedule_is_exact_and_digest_frozen():
    waves = shootout.expected_wave_schedule()

    assert len(waves) == shootout.EXPECTED_WAVES == 39
    assert shootout.EXPECTED_JOBS == 78
    assert shootout.EXPECTED_CHILD_FITS == 624
    assert shootout.wave_schedule_sha256() == EXPECTED_SCHEDULE_SHA256

    observed = []
    for dataset_index, dataset in enumerate(shootout.TASKS):
        dataset_waves = waves[3 * dataset_index : 3 * dataset_index + 3]
        assert [wave["dataset"] for wave in dataset_waves] == [dataset] * 3
        assert [wave["local_wave_index"] for wave in dataset_waves] == [0, 1, 2]
        for local_index, wave in enumerate(dataset_waves):
            assert wave["wave_index"] == 3 * dataset_index + local_index
            jobs_by_arm = {job["key"]["arm"]: job for job in wave["jobs"]}
            assert set(jobs_by_arm) == {"A10", "B10"}
            a_key = jobs_by_arm["A10"]["key"]
            b_key = jobs_by_arm["B10"]["key"]
            assert (a_key["repeat"], a_key["fold"]) == (
                shootout.SHOOTOUT_SPLITS[local_index]
            )
            assert (b_key["repeat"], b_key["fold"]) == (
                shootout.SHOOTOUT_SPLITS[(local_index + 1) % 3]
            )
            assert jobs_by_arm["A10"]["worker_slot"] == wave["wave_index"] % 2
            assert jobs_by_arm["B10"]["worker_slot"] == 1 - (
                wave["wave_index"] % 2
            )
            assert (a_key["repeat"], a_key["fold"]) != (
                b_key["repeat"],
                b_key["fold"],
            )
            observed.extend(_public_key(job) for job in wave["jobs"])

    assert len(observed) == len(set(observed)) == shootout.EXPECTED_JOBS
    assert set(observed) == {
        (dataset, repeat, fold, arm)
        for dataset in shootout.TASKS
        for repeat, fold in shootout.SHOOTOUT_SPLITS
        for arm in ("A10", "B10")
    }


def test_shootout_schedule_balances_arm_exposure_across_worker_slots():
    counts = {(arm, slot): 0 for arm in ("A10", "B10") for slot in (0, 1)}
    for wave in shootout.expected_wave_schedule():
        for job in wave["jobs"]:
            counts[(job["key"]["arm"], job["worker_slot"])] += 1

    assert counts == {
        ("A10", 0): 20,
        ("A10", 1): 19,
        ("B10", 0): 19,
        ("B10", 1): 20,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "reorder_waves",
        "duplicate_job",
        "same_coordinate",
        "duplicate_slot",
        "wrong_dataset",
        "noncanonical_metadata",
    ],
)
def test_shootout_schedule_rejects_hostile_mutations(mutation):
    waves = deepcopy(shootout.expected_wave_schedule())
    if mutation == "reorder_waves":
        waves[0], waves[1] = waves[1], waves[0]
    elif mutation == "duplicate_job":
        waves[1]["jobs"][1]["key"] = deepcopy(waves[0]["jobs"][1]["key"])
    elif mutation == "same_coordinate":
        jobs_by_arm = {job["key"]["arm"]: job for job in waves[0]["jobs"]}
        a_key = jobs_by_arm["A10"]["key"]
        b_key = jobs_by_arm["B10"]["key"]
        for field in ("dataset", "task_id", "repeat", "fold", "registered_fold"):
            b_key[field] = a_key[field]
    elif mutation == "duplicate_slot":
        waves[0]["jobs"][1]["worker_slot"] = waves[0]["jobs"][0]["worker_slot"]
    elif mutation == "wrong_dataset":
        waves[0]["dataset"] = next(
            dataset for dataset in shootout.TASKS if dataset != waves[0]["dataset"]
        )
    else:
        waves[0]["jobs"][0]["key"]["registered_fold"] += 1

    with pytest.raises(RuntimeError):
        shootout.validate_wave_schedule(waves)


def test_shootout_schedule_digest_detects_any_ordered_grid_change():
    waves = deepcopy(shootout.expected_wave_schedule())
    waves[0]["jobs"].reverse()

    mutated_digest = shootout.hashlib.sha256(
        shootout.screen.hardened._canonical_json(waves)
    ).hexdigest()
    assert mutated_digest != EXPECTED_SCHEDULE_SHA256


def test_shootout_protocol_freezes_concurrency_and_fallback_contract():
    protocol = shootout.frozen_protocol()
    execution = protocol["execution"]
    timing = protocol["timing_interpretation"]
    warmup = protocol["warmup"]
    preflight = protocol["preflight"]

    assert protocol["wave_schedule_sha256"] == EXPECTED_SCHEDULE_SHA256
    assert protocol["wave_schedule"] == shootout.expected_wave_schedule()
    assert protocol["configured_child_cpus"] == 18
    assert execution == {
        "start_method": "spawn",
        "persistent_worker_count": 2,
        "intentional_max_runnable_threads": 36,
        "barrier_between_waves": True,
        "one_a10_and_one_b10_per_wave": True,
        "within_dataset_partner_derangement": "B10 coordinate j+1 mod 3",
        "worker_slot_policy": "A10 slot alternates by global wave parity",
        "private_worker_cwd": True,
        "parent_only_campaign_metadata_writes": True,
        "safe_zero_start_resume": True,
        "failure_policy": (
            "stop releasing waves, drain or terminate the active partner, "
            "and emit no completion attestation"
        ),
    }
    assert timing == {
        "quality_is_primary": True,
        "per_arm_wall_time_is_contention_exposed": True,
        "causal_arm_timing_claim_allowed": False,
        "campaign_throughput_is_descriptive": True,
        "isolated_timing_rerun_required_if_freeze_depends_on_resources": True,
    }
    assert warmup == {
        "process_local": True,
        "worker_count": 2,
        "thread_count_per_worker": 18,
        "workers_warmed_serially_before_wave_zero": True,
    }
    assert preflight == {
        "namespace_is_non_reusable": True,
        "untimed_data_prime_both_tasks_per_worker": True,
        "datasets": ["physiochemical_protein", "QSAR-TID-11"],
        "coordinates": {
            dataset: {"repeat": repeat, "fold": fold}
            for dataset, (repeat, fold) in shootout.PREFLIGHT_COORDINATES.items()
        },
        "minimum_throughput_speedup": 1.10,
        "maximum_start_skew_seconds": 1.0,
        "maximum_concurrent_job_seconds": 1_800.0,
        "maximum_reciprocal_asymmetry_ratio": 1.5,
        "require_exact_quality_and_structure_fingerprints": True,
        "require_zero_deadlines_time_limits_restarts_oom_or_swap": True,
        "require_os_high_water_rss": True,
    }


def test_shootout_arm_configs_differ_only_by_tree_mode():
    assert shootout.B10_CONFIG["iterations"] == 10_000
    assert shootout.B10_CONFIG["l2_leaf_reg"] == 3.0
    assert shootout.B10_CONFIG["max_bins"] == 128
    assert shootout.B10_CONFIG["learning_rate"] == 0.1
    assert shootout.B10_CONFIG["ts_permutations"] == 1
    assert shootout.B10_CONFIG["tree_mode"] == "catboost"
    assert shootout.A10_CONFIG["tree_mode"] == "auto"
    assert {
        key
        for key in shootout.B10_CONFIG
        if shootout.B10_CONFIG[key] != shootout.A10_CONFIG[key]
    } == {"tree_mode"}


def test_preflight_accepts_a_preexisting_empty_output_directory(
    tmp_path, monkeypatch
):
    jobs = _fake_jobs()
    monkeypatch.setattr(
        shootout, "build_runtime_jobs", lambda _limit: (None, jobs, 18)
    )
    monkeypatch.setattr(
        shootout.screen,
        "ordering_balance",
        lambda _jobs: {"auto": {"candidate_before": 0, "candidate_after": 0}},
    )
    monkeypatch.setattr(shootout, "verify_reused_evidence", lambda: {})
    monkeypatch.setattr(
        shootout,
        "_read_process_identity",
        lambda pid: {"pid": pid, "create_time_us": 1},
    )
    monkeypatch.setattr(
        shootout.screen, "collect_source_provenance", lambda **_kwargs: {}
    )

    def preflight(output_dir, **_kwargs):
        report = {
            "decision": {
                "execution_mode": "sequential_fallback",
                "throughput_speedup": None,
            }
        }
        shootout.screen.hardened._atomic_write_json(
            output_dir / shootout.PREFLIGHT_REPORT_FILENAME, report
        )
        return report

    monkeypatch.setattr(shootout, "run_preflight", preflight)

    assert (
        shootout.main(["--output-dir", str(tmp_path), "--preflight"]) == 0
    )


def test_preflight_throughput_speedup_uses_isolated_sum_over_wave_makespans():
    assert shootout.preflight_throughput_speedup(
        [10.0, 10.0, 10.0, 10.0], [12.0, 12.0]
    ) == pytest.approx(40.0 / 24.0)


@pytest.mark.parametrize(
    ("isolated", "concurrent"),
    [
        ([], [1.0, 1.0]),
        ([1.0] * 4, []),
        ([1.0, 1.0, 1.0, 0.0], [1.0, 1.0]),
        ([1.0] * 4, [1.0, math.nan]),
        ([1.0] * 4, [1.0, -1.0]),
    ],
)
def test_preflight_throughput_speedup_rejects_incomplete_or_nonpositive_inputs(
    isolated, concurrent
):
    with pytest.raises((RuntimeError, ValueError)):
        shootout.preflight_throughput_speedup(isolated, concurrent)


def test_passing_preflight_selects_two_worker_execution():
    decision = shootout.evaluate_preflight(_passing_preflight_report())

    assert decision["passed"] is True
    assert decision["execution_mode"] == "concurrent"
    assert decision["throughput_speedup"] == pytest.approx(40.0 / 24.0)
    assert decision["reciprocal_asymmetry_ratio"] == pytest.approx(1.0)
    assert all(decision["criteria"].values())


def _attestable_preflight_report(tmp_path, monkeypatch):
    report = _passing_preflight_report()
    report.update(
        {
            "preflight_error": None,
            "sequential_recovery": None,
            "worker_warmup": [
                {"worker_slot": slot, "pid": 10_000 + slot}
                for slot in range(shootout.WORKER_COUNT)
            ],
        }
    )
    for slot, ready in enumerate(report["worker_ready"]):
        ready.update(
            {
                "child_cpus": shootout.EXPECTED_CHILD_CPUS,
                "start_method": "spawn",
                "scratch_root": str(
                    tmp_path / "preflight" / "worker_scratch" / f"slot-{slot}"
                ),
            }
        )
        report["worker_data_prime"][slot]["completed_at_utc"] = (
            "2026-07-15T00:00:00+00:00"
        )
    monkeypatch.setattr(
        shootout.screen, "_validate_followon_warmup_history", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(shootout, "_physical_memory_bytes", lambda: 128_000_000_000)
    return report


@pytest.mark.parametrize(
    "failed_policy", ["memory_headroom", "resource_swap", "session_swap"]
)
def test_valid_resource_policy_failure_is_attested_and_written_as_fallback(
    failed_policy, tmp_path, monkeypatch
):
    report = _attestable_preflight_report(tmp_path, monkeypatch)

    if failed_policy == "memory_headroom":
        report["concurrent_waves"][0]["telemetry"][
            "peak_combined_rss_bytes"
        ] = 102_400_000_000
    elif failed_policy == "resource_swap":
        telemetry = report["concurrent_waves"][0]["telemetry"]
        telemetry["samples"].append(
            {
                **telemetry["samples"][0],
                "monotonic_ns": 2,
                "swap_out_bytes": 1,
            }
        )
        telemetry["sample_count"] = len(telemetry["samples"])
        telemetry["swap_out_delta"] = 1
    else:
        session = report["worker_session_swap_telemetry"]
        session["samples"][-1]["swap_out_bytes"] = 1
        session["swap_out_delta"] = 1

    report["decision"] = shootout.evaluate_preflight(report)
    assert report["decision"]["passed"] is False
    assert report["decision"]["execution_mode"] == "sequential_fallback"

    shootout.validate_preflight_attestation(report, tmp_path)
    report_path = tmp_path / shootout.PREFLIGHT_REPORT_FILENAME
    shootout.screen.hardened._atomic_write_json(report_path, report)
    written = shootout.json.loads(report_path.read_text())
    shootout.validate_preflight_attestation(written, tmp_path)

    assert written["decision"]["execution_mode"] == "sequential_fallback"


def test_preflight_attestation_rejects_duplicate_warmup_slot(tmp_path, monkeypatch):
    report = _attestable_preflight_report(tmp_path, monkeypatch)
    report["worker_warmup"][1] = deepcopy(report["worker_warmup"][0])
    report["decision"] = shootout.evaluate_preflight(report)

    with pytest.raises(RuntimeError, match="warmup worker identity changed"):
        shootout.validate_preflight_attestation(report, tmp_path)


def test_preflight_attestation_rejects_forged_telemetry_timing(
    tmp_path, monkeypatch
):
    report = _attestable_preflight_report(tmp_path, monkeypatch)
    telemetry = report["concurrent_waves"][0]["telemetry"]
    telemetry["wave_seconds"] = 10_000.0
    telemetry["start_skew_seconds"] = 10_000.0
    report["decision"] = shootout.evaluate_preflight(report)

    with pytest.raises(RuntimeError, match="timing or resource evidence"):
        shootout.validate_preflight_attestation(report, tmp_path)


def test_worker_session_swap_checkpoint_catches_io_between_dispatches(
    monkeypatch,
):
    samples = iter(
        [
            {"monotonic_ns": 1, "swap_in_bytes": 10, "swap_out_bytes": 20},
            {"monotonic_ns": 2, "swap_in_bytes": 10, "swap_out_bytes": 20},
            {"monotonic_ns": 3, "swap_in_bytes": 10, "swap_out_bytes": 21},
        ]
    )
    monkeypatch.setattr(shootout, "_swap_counter_sample", lambda: next(samples))

    telemetry = shootout._new_swap_session_telemetry()
    shootout._checkpoint_swap_session(telemetry)
    with pytest.raises(RuntimeError, match="observed swap I/O"):
        shootout._checkpoint_swap_session(telemetry)

    assert telemetry["swap_in_delta"] == 0
    assert telemetry["swap_out_delta"] == 1


def test_post_shutdown_swap_checkpoint_fails_before_completion(
    tmp_path, monkeypatch
):
    samples = iter(
        [
            {"monotonic_ns": 1, "swap_in_bytes": 0, "swap_out_bytes": 0},
            {"monotonic_ns": 2, "swap_in_bytes": 1, "swap_out_bytes": 0},
        ]
    )
    monkeypatch.setattr(shootout, "_swap_counter_sample", lambda: next(samples))
    telemetry = shootout._new_swap_session_telemetry()

    with pytest.raises(RuntimeError, match="observed swap I/O"):
        shootout._close_production_swap_session(
            telemetry,
            None,
            history_path=tmp_path / shootout.CONCURRENCY_HISTORY_FILENAME,
            execution_mode="concurrent",
            output_dir=tmp_path,
            require_complete=False,
        )


def test_incomplete_preflight_is_attested_as_sequential_fallback(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        shootout,
        "_start_workers",
        lambda _root: (_ for _ in ()).throw(RuntimeError("pilot worker failed")),
    )

    report = shootout.run_preflight(tmp_path)

    assert report["decision"]["execution_mode"] == "sequential_fallback"
    assert report["decision"]["passed"] is False
    assert report["preflight_error"] == {
        "error_type": "RuntimeError",
        "error": "pilot worker failed",
    }
    assert (tmp_path / shootout.PREFLIGHT_REPORT_FILENAME).is_file()


def test_unquiesced_preflight_worker_aborts_before_report_or_production(
    tmp_path, monkeypatch
):
    jobs = _fake_jobs()
    process = _FaultInjectedShutdownProcess(kill_stops=False)
    worker, connection = _fault_injected_worker(process)
    production_calls = []
    monkeypatch.setattr(
        shootout, "build_runtime_jobs", lambda _limit: (None, jobs, 18)
    )
    monkeypatch.setattr(
        shootout.screen,
        "ordering_balance",
        lambda _jobs: {"auto": {"candidate_before": 0, "candidate_after": 0}},
    )
    monkeypatch.setattr(shootout, "verify_reused_evidence", lambda: {})
    monkeypatch.setattr(
        shootout,
        "_read_process_identity",
        lambda pid: {"pid": pid, "create_time_us": 1},
    )
    monkeypatch.setattr(
        shootout.screen, "collect_source_provenance", lambda **_kwargs: {}
    )
    monkeypatch.setattr(
        shootout,
        "_start_workers",
        lambda _root: ([worker], [{"slot": 0, "pid": 1234}]),
    )
    monkeypatch.setattr(
        shootout,
        "_warm_workers",
        lambda _workers: (_ for _ in ()).throw(RuntimeError("pilot fit failed")),
    )
    monkeypatch.setattr(
        shootout,
        "execute_production",
        lambda *_args, **_kwargs: production_calls.append(True),
    )

    with pytest.raises(
        RuntimeError,
        match="preflight worker shutdown could not be confirmed",
    ):
        shootout.main(["--output-dir", str(tmp_path)])

    assert "terminate" in process.calls
    assert "kill" in process.calls
    assert connection.closed
    assert production_calls == []
    assert not (tmp_path / shootout.PREFLIGHT_REPORT_FILENAME).exists()


def test_provenance_bound_recovery_forces_sequential_decision():
    report = _passing_preflight_report()
    report["sequential_recovery"] = {"attested": True}

    decision = shootout.evaluate_preflight(report)

    assert decision["passed"] is False
    assert decision["execution_mode"] == "sequential_fallback"
    assert decision["criteria"]["no_sequential_recovery_override"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "behavior_mismatch",
        "candidate_count",
        "too_slow",
        "start_skew",
        "telemetry_timing",
        "deadline",
        "child_count",
        "result_count",
        "worker_status",
        "worker_restart",
        "swap_in",
        "swap_out",
        "swap_counter_gap",
        "memory_headroom",
        "missing_high_water",
        "high_water_sum_mismatch",
        "high_water_pid_mismatch",
        "missing_data_prime",
        "job_duration",
        "nonfinite_duration",
        "missing_behavior_digest",
        "wrong_coordinate",
        "wrong_reciprocal_pairing",
        "wrong_a10_slot_parity",
        "reciprocal_asymmetry",
        "isolated_swap",
        "worker_session_swap",
    ],
)
def test_preflight_gate_fails_closed_for_hostile_mutations(mutation):
    report = _passing_preflight_report()
    concurrent_report = report["concurrent_waves"][0]["reports"][0]
    if mutation == "behavior_mismatch":
        concurrent_report["behavior_sha256"] = "changed"
    elif mutation == "candidate_count":
        concurrent_report["auto_candidate_fit_count"] = 23
    elif mutation == "too_slow":
        for wave in report["concurrent_waves"]:
            wave["wave_seconds"] = 20.0
    elif mutation == "start_skew":
        report["concurrent_waves"][0]["start_skew_seconds"] = 1.01
    elif mutation == "telemetry_timing":
        telemetry = report["concurrent_waves"][0]["telemetry"]
        telemetry["wave_seconds"] = 10_000.0
        telemetry["start_skew_seconds"] = 10_000.0
    elif mutation == "deadline":
        concurrent_report["deadline_hit"] = True
    elif mutation == "child_count":
        concurrent_report["child_count"] = 7
    elif mutation == "result_count":
        concurrent_report["result_count"] = 0
    elif mutation == "worker_status":
        concurrent_report["status"] = "error"
    elif mutation == "worker_restart":
        report["worker_restarts"] = True
    elif mutation == "swap_in":
        report["concurrent_waves"][0]["telemetry"]["swap_in_delta"] = 1
    elif mutation == "swap_out":
        report["concurrent_waves"][0]["telemetry"]["swap_out_delta"] = 1
    elif mutation == "swap_counter_gap":
        telemetry = report["concurrent_waves"][0]["telemetry"]
        telemetry["samples"].append(
            {
                **telemetry["samples"][0],
                "monotonic_ns": 2,
                "swap_in_bytes": 1,
            }
        )
        telemetry["sample_count"] = 2
    elif mutation == "memory_headroom":
        telemetry = report["concurrent_waves"][0]["telemetry"]
        telemetry["peak_combined_rss_bytes"] = (
            0.8 * telemetry["physical_memory_bytes"]
        )
    elif mutation == "missing_high_water":
        report["concurrent_waves"][0]["telemetry"].pop(
            "worker_high_water_rss_bytes"
        )
    elif mutation == "high_water_sum_mismatch":
        report["concurrent_waves"][0]["telemetry"][
            "high_water_combined_rss_bytes"
        ] += 1
    elif mutation == "high_water_pid_mismatch":
        report["isolated_runs"][0]["telemetry"][
            "worker_high_water_rss_bytes"
        ][0]["pid"] += 1
    elif mutation == "missing_data_prime":
        report["worker_data_prime"] = []
    elif mutation == "job_duration":
        concurrent_report["elapsed_seconds"] = 1_800.0
    elif mutation == "nonfinite_duration":
        concurrent_report["elapsed_seconds"] = math.inf
    elif mutation == "missing_behavior_digest":
        key = _public_key(concurrent_report)
        concurrent_report.pop("behavior_sha256")
        matching_isolated = next(
            item for item in report["isolated_runs"] if _public_key(item) == key
        )
        matching_isolated.pop("behavior_sha256")
    elif mutation == "wrong_coordinate":
        key = _public_key(concurrent_report)
        matching_isolated = next(
            item for item in report["isolated_runs"] if _public_key(item) == key
        )
        for item in (concurrent_report, matching_isolated):
            item["key"]["repeat"] = 1
            item["key"]["fold"] = 1
            item["key"]["registered_fold"] = 4
    elif mutation == "wrong_reciprocal_pairing":
        first = report["concurrent_waves"][0]["reports"]
        second = report["concurrent_waves"][1]["reports"]
        first[1], second[0] = second[0], first[1]
    elif mutation == "wrong_a10_slot_parity":
        for wave in report["concurrent_waves"]:
            a10 = next(item for item in wave["reports"] if item["key"]["arm"] == "A10")
            a10["slot"] = 0
    elif mutation == "reciprocal_asymmetry":
        concurrent_report["elapsed_seconds"] = 40.0
    elif mutation == "isolated_swap":
        report["isolated_runs"][0]["telemetry"]["swap_in_delta"] = 1
    else:
        session = report["worker_session_swap_telemetry"]
        session["samples"][-1]["swap_in_bytes"] = 1
        session["swap_in_delta"] = 1

    decision = shootout.evaluate_preflight(report)
    assert decision["passed"] is False
    assert decision["execution_mode"] == "sequential_fallback"
    assert not all(decision["criteria"].values())
