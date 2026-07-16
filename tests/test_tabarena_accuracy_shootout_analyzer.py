"""Estimator and source-contract tests for the B10/A10 shootout analyzer."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks import analyze_tabarena_regression_accuracy_shootout as analysis


def _synthetic_inputs(
    *,
    a10_test_ratio: float = 0.99,
    a10_val_ratio: float = 0.995,
    b10_test_ratio: float = 1.0,
    b10_val_ratio: float = 1.0,
):
    outer = []
    reused = {}
    for dataset, repeat, fold in analysis.campaign.expected_coordinates():
        reused[(dataset, repeat, fold)] = {
            f"{arm}_{metric}": 1.0
            for arm in ("P", "M", "C")
            for metric in analysis.QUALITY_METRICS
        }
        for arm, test_ratio, val_ratio in (
            ("baseline", b10_test_ratio, b10_val_ratio),
            ("auto", a10_test_ratio, a10_val_ratio),
        ):
            outer.append(
                {
                    "dataset": dataset,
                    "repeat": repeat,
                    "fold": fold,
                    "arm": arm,
                    "test_rmse": test_ratio,
                    "val_rmse": val_ratio,
                    "train_time_s": 1.0,
                    "infer_time_s": 1.0,
                    "peak_memory_bytes": 1.0,
                }
            )
    paired_children = [
        {"A10_selected_tree_mode": "catboost"}
        for _ in range(analysis.campaign.EXPECTED_PAIRED_CHILDREN)
    ]
    return outer, reused, paired_children


def test_reused_evidence_contract_matches_runner_and_committed_source_hashes():
    assert analysis.REUSED_EVIDENCE_CONTRACT == analysis.campaign.REUSED_EVIDENCE
    repository = Path(analysis.__file__).resolve().parents[1]
    for name, digest in analysis.SOURCE_ARTIFACTS.items():
        payload = (repository / "benchmarks" / name).read_bytes()
        assert analysis._sha256(payload) == digest


def test_reused_comparator_rows_form_exact_39_coordinate_grid():
    repository = Path(analysis.__file__).resolve().parents[1]
    source = repository / "benchmarks" / (
        "tabarena_regression_same_machine_primary_paired_splits.csv"
    )
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    parsed = analysis._parse_reused_comparator_rows(rows)

    assert len(parsed) == 39
    assert set(parsed) == set(analysis.campaign.expected_coordinates())
    assert all(
        set(values)
        == {
            f"{arm}_{metric}"
            for arm in ("P", "M", "C")
            for metric in analysis.QUALITY_METRICS
        }
        for values in parsed.values()
    )


def test_complete_reused_evidence_contract_revalidates_frozen_inputs(tmp_path):
    repository = Path(analysis.__file__).resolve().parents[1]
    frozen_repository = tmp_path / "frozen-source"
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--shared",
            "--no-checkout",
            str(repository),
            str(frozen_repository),
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(frozen_repository),
            "checkout",
            "--quiet",
            analysis.SOURCE_COMMIT,
        ],
        check=True,
    )
    source_manifest = json.loads(
        (
            frozen_repository
            / "benchmarks/tabarena_regression_same_machine_run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    manifest = {
        "reused_evidence": analysis.REUSED_EVIDENCE_CONTRACT,
        "runtime": source_manifest["runtime"],
        "source": {
            "tabarena": source_manifest["source"]["tabarena"],
        },
    }

    rows, diagnostics = analysis._verify_source_reuse_contract(
        frozen_repository, manifest
    )

    assert len(rows) == 39
    assert diagnostics["source_artifact_count"] == 4
    assert diagnostics["runtime_lock_verified"] is True
    assert diagnostics["tabarena_provenance"]["git_head"] == (
        source_manifest["source"]["tabarena"]["git_head"]
    )


def test_reused_evidence_contract_rejects_package_subtree_drift(monkeypatch):
    def simulated_git(_repository, args, _label):
        if args == ["rev-parse", f"{analysis.SOURCE_COMMIT}:darkofit"]:
            return analysis.SOURCE_DARKOFIT_SUBTREE
        if args == ["rev-parse", "HEAD:darkofit"]:
            return "0" * 40
        raise AssertionError(f"unexpected git command: {args}")

    monkeypatch.setattr(analysis, "_git", simulated_git)

    with pytest.raises(RuntimeError, match="package subtree changed"):
        analysis._verify_source_reuse_contract(
            Path(analysis.__file__).resolve().parents[1],
            {"reused_evidence": analysis.REUSED_EVIDENCE_CONTRACT},
        )


def test_common_dependency_lock_is_explicit_and_only_comparators_may_be_omitted():
    repository = Path(analysis.__file__).resolve().parents[1]
    source_manifest = json.loads(
        (
            repository
            / "benchmarks/tabarena_regression_same_machine_run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    source = source_manifest["runtime"]["packages"]
    current = {
        name: version
        for name, version in source.items()
        if name not in analysis.COMPARATOR_ONLY_PACKAGES
    }

    diagnostics = analysis._validate_common_dependency_lock(source, current)

    assert "autogluon.features" in diagnostics["common_packages"]
    assert "graphviz" in diagnostics["common_packages"]
    assert diagnostics["omitted_comparator_packages"] == [
        "catboost",
        "chimeraboost",
    ]


def test_common_dependency_lock_treats_null_comparator_as_omitted():
    repository = Path(analysis.__file__).resolve().parents[1]
    source_manifest = json.loads(
        (
            repository
            / "benchmarks/tabarena_regression_same_machine_run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    source = source_manifest["runtime"]["packages"]
    current = dict(source)
    current["catboost"] = None

    diagnostics = analysis._validate_common_dependency_lock(source, current)

    assert diagnostics["omitted_comparator_packages"] == ["catboost"]


def test_common_dependency_lock_requires_exact_installed_comparator_version():
    repository = Path(analysis.__file__).resolve().parents[1]
    source_manifest = json.loads(
        (
            repository
            / "benchmarks/tabarena_regression_same_machine_run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    source = source_manifest["runtime"]["packages"]
    current = dict(source)
    current["catboost"] = "different"

    with pytest.raises(RuntimeError, match="version differs.*catboost"):
        analysis._validate_common_dependency_lock(source, current)


@pytest.mark.parametrize("missing", ["autogluon.features", "graphviz", "numpy"])
def test_common_dependency_lock_rejects_any_missing_common_package(missing):
    repository = Path(analysis.__file__).resolve().parents[1]
    source_manifest = json.loads(
        (
            repository
            / "benchmarks/tabarena_regression_same_machine_run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    source = source_manifest["runtime"]["packages"]
    current = dict(source)
    current.pop(missing)

    with pytest.raises(RuntimeError, match="missing common"):
        analysis._validate_common_dependency_lock(source, current)


def test_common_dependency_lock_rejects_unknown_or_changed_package():
    repository = Path(analysis.__file__).resolve().parents[1]
    source_manifest = json.loads(
        (
            repository
            / "benchmarks/tabarena_regression_same_machine_run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    source = source_manifest["runtime"]["packages"]
    unexpected = {**source, "not-frozen": "1.0"}
    with pytest.raises(RuntimeError, match="unexpected"):
        analysis._validate_common_dependency_lock(source, unexpected)

    changed = dict(source)
    changed["autogluon.features"] = "different"
    with pytest.raises(RuntimeError, match="version differs"):
        analysis._validate_common_dependency_lock(source, changed)


@pytest.mark.parametrize("field", ["git_head", "git_tree", "git_remote_origin"])
def test_reused_tabarena_provenance_is_exact(field):
    repository = Path(analysis.__file__).resolve().parents[1]
    source_manifest = json.loads(
        (
            repository
            / "benchmarks/tabarena_regression_same_machine_run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    source = source_manifest["source"]
    current = {"tabarena": deepcopy(source["tabarena"])}
    diagnostics = analysis._validate_reused_tabarena_provenance(source, current)
    assert diagnostics["git_head"] == source["tabarena"]["git_head"]

    current["tabarena"][field] = "changed"
    with pytest.raises(RuntimeError, match=field):
        analysis._validate_reused_tabarena_provenance(source, current)


def _warmup_session(
    output_dir: Path,
    *,
    session_name: str = "session-999-123456",
    pids: tuple[int, int] = (101, 102),
) -> list[dict]:
    repository = Path(analysis.__file__).resolve().parents[1]
    base_record = json.loads(
        (
            repository
            / "benchmarks/tabarena_regression_followon_screen_warmup_history.json"
        ).read_text(encoding="utf-8")
    )[0]
    ready = []
    records = []
    session_root = output_dir / "worker_scratch" / session_name
    for slot, pid in enumerate(pids):
        scratch_root = session_root / f"worker-{slot}"
        scratch_root.mkdir(parents=True)
        ready.append(
            {
                "type": "ready",
                "slot": slot,
                "pid": pid,
                "child_cpus": 18,
                "start_method": "spawn",
                "scratch_root": str(scratch_root.resolve()),
            }
        )
        records.append(
            {
                **deepcopy(base_record),
                "pid": pid,
                "worker_slot": slot,
            }
        )
    return [
        {
            "completed_at_utc": "2026-07-15T12:00:00+00:00",
            "execution_mode": "concurrent",
            "worker_ready": ready,
            "worker_warmup": records,
        }
    ]


def test_analyzer_revalidates_both_process_local_warmups(tmp_path):
    history = _warmup_session(tmp_path)

    identities = analysis._validate_warmup_sessions(
        history, execution_mode="concurrent", output_dir=tmp_path
    )
    assert identities == {(0, 101), (1, 102)}

    history[0]["worker_ready"][1]["pid"] = 101
    history[0]["worker_warmup"][1]["pid"] = 101
    with pytest.raises(RuntimeError, match="PIDs are not distinct"):
        analysis._validate_warmup_sessions(
            history, execution_mode="concurrent", output_dir=tmp_path
        )


def test_measured_workers_bind_exactly_to_newest_warmup_session(tmp_path):
    history = [
        *_warmup_session(
            tmp_path,
            session_name="session-999-123456",
            pids=(101, 102),
        ),
        *_warmup_session(
            tmp_path,
            session_name="session-1000-234567",
            pids=(201, 202),
        ),
    ]
    latest = analysis._validate_warmup_sessions(
        history, execution_mode="concurrent", output_dir=tmp_path
    )
    concurrency = {
        "entries": [
            {
                "reports": [
                    {"slot": 0, "pid": 201},
                    {"slot": 1, "pid": 202},
                ]
            }
        ]
    }

    analysis._validate_production_session_binding(
        history,
        latest_warmup_identities=latest,
        concurrency=concurrency,
        resume_history=[{"resumed_at_utc": "2026-07-15T11:00:00+00:00"}],
    )

    concurrency["entries"][0]["reports"][0]["pid"] = 101
    with pytest.raises(RuntimeError, match="newest warmup session"):
        analysis._validate_production_session_binding(
            history,
            latest_warmup_identities=latest,
            concurrency=concurrency,
            resume_history=[{"resumed_at_utc": "2026-07-15T11:00:00+00:00"}],
        )


def test_warmup_session_count_cannot_exceed_resume_attempt_count(tmp_path):
    history = [
        *_warmup_session(
            tmp_path,
            session_name="session-999-123456",
            pids=(101, 102),
        ),
        *_warmup_session(
            tmp_path,
            session_name="session-1000-234567",
            pids=(201, 202),
        ),
    ]
    latest = analysis._validate_warmup_sessions(
        history, execution_mode="concurrent", output_dir=tmp_path
    )
    concurrency = {
        "entries": [
            {
                "reports": [
                    {"slot": 0, "pid": 201},
                    {"slot": 1, "pid": 202},
                ]
            }
        ]
    }

    with pytest.raises(RuntimeError, match="resume history"):
        analysis._validate_production_session_binding(
            history,
            latest_warmup_identities=latest,
            concurrency=concurrency,
            resume_history=None,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "different_session",
        "traversal",
        "wrong_worker_leaf",
        "relative_path",
        "outside_campaign",
        "trailing_separator",
    ],
)
def test_warmup_validation_rejects_hostile_private_scratch_paths(
    tmp_path, mutation
):
    history = _warmup_session(tmp_path)
    ready = history[0]["worker_ready"]
    if mutation == "different_session":
        scratch = tmp_path / "worker_scratch/session-1000-654321/worker-1"
        scratch.mkdir(parents=True)
        ready[1]["scratch_root"] = str(scratch.resolve())
    elif mutation == "traversal":
        session = Path(ready[1]["scratch_root"]).parent
        ready[1]["scratch_root"] = str(session / "nested" / ".." / "worker-1")
    elif mutation == "wrong_worker_leaf":
        ready[1]["scratch_root"] = ready[0]["scratch_root"]
    elif mutation == "relative_path":
        ready[1]["scratch_root"] = "worker_scratch/session-999-123456/worker-1"
    elif mutation == "outside_campaign":
        scratch = tmp_path.parent / "session-999-123456/worker-1"
        scratch.mkdir(parents=True, exist_ok=True)
        ready[1]["scratch_root"] = str(scratch.resolve())
    else:
        ready[1]["scratch_root"] += "/"

    with pytest.raises(RuntimeError, match="scratch|session|private"):
        analysis._validate_warmup_sessions(
            history, execution_mode="concurrent", output_dir=tmp_path
        )


def test_warmup_validation_rejects_symlinked_private_scratch_path(tmp_path):
    history = _warmup_session(tmp_path)
    scratch = Path(history[0]["worker_ready"][1]["scratch_root"])
    scratch.rmdir()
    target = tmp_path / "symlink-target"
    target.mkdir()
    scratch.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        analysis._validate_warmup_sessions(
            history, execution_mode="concurrent", output_dir=tmp_path
        )


@pytest.mark.parametrize("validator", ["archive", "artifact"])
def test_campaign_paths_reject_symlinked_parent_escape(tmp_path, validator):
    outside = tmp_path.parent / f"{tmp_path.name}-outside-{validator}"
    outside.mkdir()
    payload = b"external payload"
    (outside / "payload.bin").write_bytes(payload)
    if validator == "archive":
        parent = tmp_path / "resume_invalidated"
        parent.mkdir()
        (parent / "link").symlink_to(outside, target_is_directory=True)
        with pytest.raises(RuntimeError, match="escape|symbolic-link"):
            analysis._validate_archived_path(
                tmp_path,
                "resume_invalidated/link/payload.bin",
                "archived result",
            )
    else:
        parent = tmp_path / "artifacts"
        parent.mkdir()
        (parent / "link").symlink_to(outside, target_is_directory=True)
        with pytest.raises(RuntimeError, match="escape|symbolic-link"):
            analysis._artifact_bytes(
                tmp_path,
                "artifacts/link/payload.bin",
                {"size_bytes": len(payload), "sha256": analysis._sha256(payload)},
                "attested artifact",
            )


def _resume_history_with_invalidated_member(output_dir: Path) -> list[dict]:
    wave = analysis.campaign.expected_wave_schedule()[0]
    member_key = deepcopy(wave["jobs"][0]["key"])
    archived = (
        output_dir
        / "resume_invalidated/20260715T120000000000Z"
        / analysis.campaign.screen.expected_result_relative_path(
            member_key["dataset"],
            member_key["repeat"],
            member_key["fold"],
            member_key["internal_arm"],
        )
    )
    archived.parent.mkdir(parents=True)
    archived.write_bytes(b"changed cached result")
    return [
        {
            "resumed_at_utc": "2026-07-15T12:00:00+00:00",
            "pid": 999,
            "wave_schedule_sha256": analysis.campaign.wave_schedule_sha256(),
            "reusable_wave_indices": [],
            "pending_wave_indices": list(range(analysis.campaign.EXPECTED_WAVES)),
            "invalidated_waves": [
                {
                    "wave_index": 0,
                    "members": [
                        {
                            "key": member_key,
                            "status": "unattested_or_changed",
                            "path": str(archived.relative_to(output_dir)),
                        }
                    ],
                }
            ],
            "archived_campaign_artifacts": [],
        }
    ]


def test_resume_analyzer_accepts_runner_unattested_or_changed_status(tmp_path):
    history = _resume_history_with_invalidated_member(tmp_path)

    analysis._validate_shootout_resume_history(history, tmp_path)

    history[0]["invalidated_waves"][0]["members"][0]["status"] = "invented"
    with pytest.raises(RuntimeError, match="wrong wave"):
        analysis._validate_shootout_resume_history(history, tmp_path)


def test_resume_analyzer_accepts_prior_process_pickle_archive_status(tmp_path):
    history = _resume_history_with_invalidated_member(tmp_path)
    history[0]["invalidated_waves"][0]["members"][0][
        "status"
    ] = "prior_process_pickle_archived"

    analysis._validate_shootout_resume_history(history, tmp_path)


def test_resume_analyzer_rejects_duplicate_member_key_or_archive_path(tmp_path):
    history = _resume_history_with_invalidated_member(tmp_path)
    duplicate = deepcopy(history[0]["invalidated_waves"][0]["members"][0])
    history[0]["invalidated_waves"][0]["members"].append(duplicate)

    with pytest.raises(RuntimeError, match="wrong wave|duplicated"):
        analysis._validate_shootout_resume_history(history, tmp_path)


def test_optional_resume_artifact_presence_must_match_attestation(tmp_path):
    filename = analysis.campaign.screen.RESUME_HISTORY_FILENAME
    attestation = {}
    analysis._validate_optional_artifact_presence(
        tmp_path,
        attestation,
        field="resume_history_artifact",
        filename=filename,
    )

    path = tmp_path / filename
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="presence"):
        analysis._validate_optional_artifact_presence(
            tmp_path,
            attestation,
            field="resume_history_artifact",
            filename=filename,
        )

    attestation["resume_history_artifact"] = {
        "path": filename,
        "sha256": "0" * 64,
        "size_bytes": path.stat().st_size,
    }
    analysis._validate_optional_artifact_presence(
        tmp_path,
        attestation,
        field="resume_history_artifact",
        filename=filename,
    )
    path.unlink()
    with pytest.raises(RuntimeError, match="presence"):
        analysis._validate_optional_artifact_presence(
            tmp_path,
            attestation,
            field="resume_history_artifact",
            filename=filename,
        )


def test_resume_analyzer_requires_full_rerun_without_cross_process_reuse(tmp_path):
    history = _resume_history_with_invalidated_member(tmp_path)
    history[0]["reusable_wave_indices"] = [0]
    history[0]["pending_wave_indices"] = list(
        range(1, analysis.campaign.EXPECTED_WAVES)
    )

    with pytest.raises(RuntimeError, match="rerun all waves"):
        analysis._validate_shootout_resume_history(history, tmp_path)


def test_preflight_analysis_invokes_runner_attestation_validator(
    tmp_path, monkeypatch
):
    decision = {
        "passed": True,
        "execution_mode": "concurrent",
        "throughput_speedup": 1.2,
        "timing_admissible": True,
        "mode_selection_criteria": ["synthetic"],
        "criteria": {"synthetic": True},
    }
    report = {
        "schema_version": 1,
        "kind": analysis.campaign.CAMPAIGN_KIND + "_preflight",
        "protocol_sha256": analysis.protocol_sha256(),
        "wave_schedule_sha256": analysis.campaign.wave_schedule_sha256(),
        "swap_policy": analysis.campaign.SWAP_POLICY_STRICT,
        "decision": decision,
    }
    observed = []

    def validate(value, output_dir):
        observed.append((value, output_dir))

    monkeypatch.setattr(
        analysis.campaign, "validate_preflight_attestation", validate
    )
    monkeypatch.setattr(
        analysis.campaign, "evaluate_preflight", lambda _value: decision
    )

    assert analysis._validate_preflight_artifact(
        report, execution_mode="concurrent", input_dir=tmp_path
    ) == decision
    assert observed == [(report, tmp_path)]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            {
                "swap_policy": analysis.campaign.SWAP_POLICY_STRICT,
                "timing_admissible": False,
            },
            "timing admissibility",
        ),
        (
            {
                "swap_policy": analysis.campaign.SWAP_POLICY_QUALITY_ONLY_SWAP_IN,
                "timing_admissible": True,
            },
            "timing admissibility",
        ),
        (
            {"swap_policy": "unknown", "timing_admissible": False},
            "swap policy",
        ),
    ],
)
def test_policy_timing_binding_rejects_tampering(value, expected):
    with pytest.raises(RuntimeError, match=expected):
        analysis._validate_policy_timing_binding(value, "synthetic artifact")


def test_preflight_policy_must_match_manifest_selection(tmp_path, monkeypatch):
    report = {
        "swap_policy": analysis.campaign.SWAP_POLICY_QUALITY_ONLY_SWAP_IN,
    }
    monkeypatch.setattr(
        analysis.campaign,
        "validate_preflight_attestation",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="preflight swap policy"):
        analysis._validate_preflight_artifact(
            report,
            execution_mode="concurrent",
            input_dir=tmp_path,
            swap_policy=analysis.campaign.SWAP_POLICY_STRICT,
            timing_admissible=True,
        )


def test_concurrency_validator_receives_exact_selected_policy(tmp_path, monkeypatch):
    observed = []
    history = {
        "swap_policy": analysis.campaign.SWAP_POLICY_QUALITY_ONLY_SWAP_IN,
        "timing_admissible": False,
    }

    def validate(value, **kwargs):
        observed.append((value, kwargs))

    monkeypatch.setattr(
        analysis.campaign, "validate_concurrency_history", validate
    )
    analysis._validate_concurrency_artifact(
        history,
        execution_mode="concurrent",
        input_dir=tmp_path,
        swap_policy=analysis.campaign.SWAP_POLICY_QUALITY_ONLY_SWAP_IN,
        timing_admissible=False,
    )

    assert observed == [
        (
            history,
            {
                "execution_mode": "concurrent",
                "output_dir": tmp_path,
                "swap_policy": analysis.campaign.SWAP_POLICY_QUALITY_ONLY_SWAP_IN,
            },
        )
    ]


def test_concurrency_reports_bind_bijectively_to_completed_artifacts(tmp_path):
    reports = []
    artifacts = {}
    for dataset, repeat, fold in analysis.campaign.expected_coordinates():
        for public_arm, internal_arm in analysis.campaign.PUBLIC_TO_INTERNAL_ARM.items():
            relative = analysis.campaign.screen.expected_result_relative_path(
                dataset, repeat, fold, internal_arm
            )
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"{dataset}:{repeat}:{fold}:{public_arm}".encode()
            path.write_bytes(payload)
            digest = analysis._sha256(payload)
            artifacts[relative] = {
                "sha256": digest,
                "size_bytes": len(payload),
            }
            reports.append(
                {
                    "result_path": str(path),
                    "result_sha256": digest,
                    "result_size_bytes": len(payload),
                }
            )
    concurrency = {"entries": [{"reports": reports}]}

    analysis._validate_concurrency_result_bindings(
        concurrency, artifacts, tmp_path
    )

    reports[0]["result_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="bind the completed result"):
        analysis._validate_concurrency_result_bindings(
            concurrency, artifacts, tmp_path
        )


def test_concurrency_result_binding_rejects_duplicate_artifact_path(tmp_path):
    path = tmp_path / "experiments/one/results.pkl"
    path.parent.mkdir(parents=True)
    payload = b"one"
    path.write_bytes(payload)
    relative = str(path.relative_to(tmp_path))
    report = {
        "result_path": str(path),
        "result_sha256": analysis._sha256(payload),
        "result_size_bytes": len(payload),
    }

    with pytest.raises(RuntimeError, match="duplicated"):
        analysis._validate_concurrency_result_bindings(
            {"entries": [{"reports": [report, deepcopy(report)]}]},
            {
                relative: {
                    "sha256": analysis._sha256(payload),
                    "size_bytes": len(payload),
                }
            },
            tmp_path,
        )


def test_reused_comparator_parser_rejects_cross_contrast_value_disagreement():
    repository = Path(analysis.__file__).resolve().parents[1]
    source = repository / "benchmarks" / (
        "tabarena_regression_same_machine_primary_paired_splits.csv"
    )
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    mutated = deepcopy(rows)
    dc = next(row for row in mutated if row["contrast_code"] == "D/C")
    dc["numerator_test_rmse"] = str(float(dc["numerator_test_rmse"]) + 1.0)

    with pytest.raises(RuntimeError, match="values disagree"):
        analysis._parse_reused_comparator_rows(mutated)


def test_frozen_hierarchy_uses_split_then_dataset_geometric_means():
    outer, reused, children = _synthetic_inputs()
    first_dataset = next(iter(analysis.campaign.TASKS))
    ratios = (0.8, 1.0, 1.2)
    for row in outer:
        if row["dataset"] == first_dataset and row["arm"] == "auto":
            row["test_rmse"] = ratios[row["repeat"]]
    paired = analysis.pair_outer_rows(outer, reused)

    summary, per_dataset = analysis.analyze(paired, children)

    expected_dataset = math.exp(sum(math.log(value) for value in ratios) / 3.0)
    expected_campaign = math.exp(
        math.log(expected_dataset) / 13.0 + 12 * math.log(0.99) / 13.0
    )
    first = next(
        row
        for row in per_dataset
        if row["contrast"] == "A10/M" and row["dataset"] == first_dataset
    )
    assert first["test_rmse_ratio"] == pytest.approx(expected_dataset)
    assert summary["contrasts"]["A10/M"]["test_rmse"]["ratio"] == pytest.approx(
        expected_campaign
    )


def test_attribution_chain_and_all_three_frozen_quality_gates():
    outer, reused, children = _synthetic_inputs(
        a10_test_ratio=0.98,
        b10_test_ratio=0.99,
    )
    paired = analysis.pair_outer_rows(outer, reused)

    summary, _ = analysis.analyze(paired, children)

    assert summary["contrasts"]["A10/M"]["test_rmse"]["ratio"] == pytest.approx(
        0.98
    )
    assert summary["contrasts"]["A10/P"]["test_rmse"]["ratio"] == pytest.approx(
        0.98
    )
    assert summary["contrasts"]["A10/B10"]["test_rmse"]["ratio"] == pytest.approx(
        0.98 / 0.99
    )
    assert summary["contrasts"]["B10/P"]["test_rmse"]["ratio"] == pytest.approx(
        0.99
    )
    assert len(summary["lodo_a10_over_m"]) == 13
    assert all(
        item["ratio"] == pytest.approx(0.98)
        for item in summary["lodo_a10_over_m"]
    )
    assert summary["gates"]["all_development_gates_pass"] is True


@pytest.mark.parametrize(
    ("a10_ratio", "expected_failed_gate"),
    [
        (1.001, "parity_g_a10_over_m_at_most_1"),
        (1.021, "worst_dataset_a10_over_p_at_most_1_02"),
    ],
)
def test_frozen_gates_fail_at_preregistered_thresholds(
    a10_ratio, expected_failed_gate
):
    outer, reused, children = _synthetic_inputs(a10_test_ratio=a10_ratio)
    summary, _ = analysis.analyze(
        analysis.pair_outer_rows(outer, reused), children
    )

    assert summary["gates"][expected_failed_gate] is False
    assert summary["gates"]["all_development_gates_pass"] is False


def test_lodo_gate_prevents_one_strong_dataset_from_hiding_panel_harm():
    outer, reused, children = _synthetic_inputs(a10_test_ratio=1.011)
    carrying_dataset = next(iter(analysis.campaign.TASKS))
    for row in outer:
        if row["dataset"] == carrying_dataset and row["arm"] == "auto":
            row["test_rmse"] = 0.85

    summary, _ = analysis.analyze(
        analysis.pair_outer_rows(outer, reused), children
    )

    assert summary["gates"]["parity_g_a10_over_m_at_most_1"] is True
    assert summary["gates"]["worst_dataset_a10_over_p_at_most_1_02"] is True
    assert summary["gates"]["worst_lodo_a10_over_m_at_most_1_01"] is False
    assert summary["worst_lodo_a10_over_m"]["omitted_dataset"] == carrying_dataset


def test_sequential_fallback_report_makes_no_contention_claim():
    outer, reused, children = _synthetic_inputs()
    summary, per_dataset = analysis.analyze(
        analysis.pair_outer_rows(outer, reused),
        children,
        execution_mode="sequential_fallback",
    )

    assert summary["execution_mode"] == "sequential_fallback"
    assert "sequential fallback" in summary["timing_disclosure"]
    assert "contention-exposed" not in summary["timing_disclosure"]
    report = analysis.render_report(summary, per_dataset)
    assert "Execution mode: **sequential_fallback**" in report


def test_quality_only_report_disclaims_performance_and_exposes_raw_swap_audit():
    outer, reused, children = _synthetic_inputs()
    swap_audit = {
        "swap_policy": analysis.campaign.SWAP_POLICY_QUALITY_ONLY_SWAP_IN,
        "timing_admissible": False,
        "production_zero_swap_out_verified": True,
        "preflight": {
            "decision_passed": True,
            "decision_execution_mode": "concurrent",
            "selected_policy_passed": True,
            "zero_swap_out_coverage_status": "complete",
            "zero_swap_out_observed": True,
            "preflight_error": None,
            "lifecycle": {
                "status": "complete",
                "swap_in_bytes": 11,
                "swap_out_bytes": 0,
                "policy_passed": True,
            },
            "measured": {
                "status": "complete",
                "swap_in_bytes": 7,
                "swap_out_bytes": 0,
                "policy_passed": True,
            },
        },
        "production": {
            "lifecycle": {
                "status": "complete",
                "swap_in_bytes": 19,
                "swap_out_bytes": 0,
                "policy_passed": True,
            },
            "measured": {
                "status": "complete",
                "swap_in_bytes": 13,
                "swap_out_bytes": 0,
                "policy_passed": True,
            },
        },
    }
    decision = {
        "execution_mode": "concurrent",
        "timing_admissible": False,
        "mode_selection_criteria": [
            "eight_valid_executions",
            "exact_behavior_fingerprints",
            "operational_limits",
            "full_session_swap_policy",
        ],
        "throughput_speedup": 1.23,
        "reciprocal_asymmetry_ratio": 1.08,
    }

    summary, per_dataset = analysis.analyze(
        analysis.pair_outer_rows(outer, reused),
        children,
        swap_policy=analysis.campaign.SWAP_POLICY_QUALITY_ONLY_SWAP_IN,
        timing_admissible=False,
        swap_audit=swap_audit,
        preflight_decision=decision,
    )

    assert summary["timing_admissible"] is False
    assert summary["memory_performance_admissible"] is False
    assert summary["swap_audit"] == swap_audit
    assert summary["execution_mode_selection"][
        "performance_comparison_criteria_used"
    ] is False
    assert "inadmissible" in summary["timing_disclosure"]
    assert "operational audit data" in summary["timing_disclosure"]
    assert "contention-exposed" not in summary["timing_disclosure"]
    report = analysis.render_report(summary, per_dataset)
    assert "Timing and memory-performance evidence: **inadmissible by policy**" in report
    assert "| Preflight lifecycle | complete | 11 B | 0 B | pass |" in report
    assert "| Production measured | complete | 13 B | 0 B | pass |" in report
    assert "Production zero swap-out verified" in report
    assert "raw host-counter deltas, not performance results" in report
    assert (
        "Zero swap-out observed across the complete preflight lifecycle and "
        "measured windows: **yes**."
    ) in report


@pytest.mark.parametrize(
    ("execution_mode", "swap_policy", "timing_admissible", "disposition"),
    [
        (
            "concurrent",
            analysis.campaign.SWAP_POLICY_STRICT,
            True,
            "timing_admissible_with_noncausal_schedule_limits",
        ),
        (
            "sequential_fallback",
            analysis.campaign.SWAP_POLICY_STRICT,
            True,
            "timing_admissible_with_noncausal_schedule_limits",
        ),
        (
            "concurrent",
            analysis.campaign.SWAP_POLICY_QUALITY_ONLY_SWAP_IN,
            False,
            "inadmissible_by_quality_only_swap_in_policy",
        ),
    ],
)
def test_standalone_comparative_csv_rows_carry_policy_disposition(
    execution_mode, swap_policy, timing_admissible, disposition
):
    outer, reused, children = _synthetic_inputs()
    paired = analysis.pair_outer_rows(outer, reused)
    summary, per_dataset = analysis.analyze(
        paired,
        children,
        execution_mode=execution_mode,
        swap_policy=swap_policy,
        timing_admissible=timing_admissible,
    )

    payloads = analysis._build_output_payloads(
        paired, per_dataset, children, summary
    )

    for key, expected_count in (("split_csv", 39), ("child_csv", 312)):
        rows = list(csv.DictReader(payloads[key].decode("utf-8").splitlines()))
        assert len(rows) == expected_count
        assert all(row["execution_mode"] == execution_mode for row in rows)
        assert all(row["swap_policy"] == swap_policy for row in rows)
        assert all(
            row["timing_admissible"] == str(timing_admissible) for row in rows
        )
        assert all(
            row["performance_evidence_disposition"] == disposition for row in rows
        )
    assert all(
        not set(row).intersection(analysis.EXPORT_CONTEXT_FIELDS) for row in paired
    )
    assert all(
        not set(row).intersection(analysis.EXPORT_CONTEXT_FIELDS) for row in children
    )


def test_quality_only_mode_selection_rejects_timing_criteria():
    outer, reused, children = _synthetic_inputs()
    with pytest.raises(RuntimeError, match="inadmissible timing evidence"):
        analysis.analyze(
            analysis.pair_outer_rows(outer, reused),
            children,
            swap_policy=analysis.campaign.SWAP_POLICY_QUALITY_ONLY_SWAP_IN,
            timing_admissible=False,
            preflight_decision={
                "execution_mode": "concurrent",
                "timing_admissible": False,
                "mode_selection_criteria": [
                    "throughput_speedup_at_least_1_10"
                ],
            },
        )


def test_swap_audit_rejects_production_swap_out():
    preflight = {
        "decision": {
            "passed": True,
            "execution_mode": "concurrent",
            "criteria": {
                "full_session_swap_policy": True,
                "measured_phase_swap_policy": True,
            },
        },
        "preflight_error": None,
        "worker_session_swap_telemetry": {
            "swap_in_delta": 1,
            "swap_out_delta": 0,
        },
        "measured_phase_swap_window": {
            "swap_in_delta": 1,
            "swap_out_delta": 0,
        },
    }
    concurrency = {
        "execution_mode": "concurrent",
        "worker_session_swap_telemetry": {
            "swap_in_delta": 2,
            "swap_out_delta": 0,
        },
        "measured_phase_swap_window": {
            "swap_in_delta": 2,
            "swap_out_delta": 1,
        },
    }

    with pytest.raises(RuntimeError, match="zero swap-out"):
        analysis._build_swap_audit(
            preflight,
            concurrency,
            swap_policy=analysis.campaign.SWAP_POLICY_QUALITY_ONLY_SWAP_IN,
            timing_admissible=False,
        )


def test_swap_audit_accepts_unavailable_measured_preflight_for_fallback():
    preflight = {
        "decision": {
            "passed": False,
            "execution_mode": "sequential_fallback",
            "criteria": {
                "full_session_swap_policy": True,
                "measured_phase_swap_policy": False,
            },
        },
        "preflight_error": {
            "error_type": "RuntimeError",
            "error": "pilot worker failed before measured dispatch",
        },
        "worker_session_swap_telemetry": {
            "swap_in_delta": 0,
            "swap_out_delta": 0,
        },
        "measured_phase_swap_window": None,
    }
    production = {
        "execution_mode": "sequential_fallback",
        "worker_session_swap_telemetry": {
            "swap_in_delta": 0,
            "swap_out_delta": 0,
        },
        "measured_phase_swap_window": {
            "swap_in_delta": 0,
            "swap_out_delta": 0,
        },
    }

    audit = analysis._build_swap_audit(
        preflight,
        production,
        swap_policy=analysis.campaign.SWAP_POLICY_STRICT,
        timing_admissible=True,
    )

    assert audit["preflight"]["measured"] == {
        "status": "unavailable",
        "swap_in_bytes": None,
        "swap_out_bytes": None,
        "policy_passed": False,
    }
    assert audit["preflight"]["zero_swap_out_coverage_status"] == "unavailable"
    assert audit["preflight"]["zero_swap_out_observed"] is None
    assert audit["preflight"]["decision_execution_mode"] == "sequential_fallback"
    assert audit["production_zero_swap_out_verified"] is True
    outer, reused, children = _synthetic_inputs()
    summary, per_dataset = analysis.analyze(
        analysis.pair_outer_rows(outer, reused),
        children,
        execution_mode="sequential_fallback",
        swap_policy=analysis.campaign.SWAP_POLICY_STRICT,
        timing_admissible=True,
        swap_audit=audit,
    )
    report = analysis.render_report(summary, per_dataset)
    assert "| Preflight measured | unavailable | unavailable | unavailable | fail |" in report
    assert "pilot worker failed before measured dispatch" in report
    assert "Production zero swap-out verified" in report
    assert "Preflight zero-swap-out coverage: **unavailable**" in report
    assert "across complete preflight windows" not in report


def test_partial_preflight_swap_coverage_has_no_complete_window_conclusion():
    preflight = {
        "decision": {
            "passed": False,
            "execution_mode": "sequential_fallback",
            "criteria": {
                "full_session_swap_policy": True,
                "measured_phase_swap_policy": False,
            },
        },
        "preflight_error": {
            "error_type": "RuntimeError",
            "error": "pilot failed after one measured dispatch",
        },
        "worker_session_swap_telemetry": {
            "swap_in_delta": 0,
            "swap_out_delta": 0,
        },
        "measured_phase_swap_window": {
            "swap_in_delta": 0,
            "swap_out_delta": 0,
        },
    }
    production = {
        "execution_mode": "sequential_fallback",
        "worker_session_swap_telemetry": {
            "swap_in_delta": 0,
            "swap_out_delta": 0,
        },
        "measured_phase_swap_window": {
            "swap_in_delta": 0,
            "swap_out_delta": 0,
        },
    }

    audit = analysis._build_swap_audit(
        preflight,
        production,
        swap_policy=analysis.campaign.SWAP_POLICY_STRICT,
        timing_admissible=True,
    )

    assert audit["preflight"]["measured"]["status"] == "partial"
    assert audit["preflight"]["zero_swap_out_coverage_status"] == "partial"
    assert audit["preflight"]["zero_swap_out_observed"] is None
    outer, reused, children = _synthetic_inputs()
    summary, per_dataset = analysis.analyze(
        analysis.pair_outer_rows(outer, reused),
        children,
        execution_mode="sequential_fallback",
        swap_policy=analysis.campaign.SWAP_POLICY_STRICT,
        timing_admissible=True,
        swap_audit=audit,
    )
    report = analysis.render_report(summary, per_dataset)
    assert "Preflight zero-swap-out coverage: **partial**" in report
    assert "no complete two-window zero-swap-out conclusion" in report
    assert "across the complete preflight lifecycle" not in report


def test_swap_audit_preserves_preflight_swapout_failure_for_fallback():
    preflight = {
        "decision": {
            "passed": False,
            "execution_mode": "sequential_fallback",
            "criteria": {
                "full_session_swap_policy": False,
                "measured_phase_swap_policy": False,
            },
        },
        "preflight_error": None,
        "worker_session_swap_telemetry": {
            "swap_in_delta": 5,
            "swap_out_delta": 3,
        },
        "measured_phase_swap_window": {
            "swap_in_delta": 4,
            "swap_out_delta": 2,
        },
    }
    production = {
        "execution_mode": "sequential_fallback",
        "worker_session_swap_telemetry": {
            "swap_in_delta": 9,
            "swap_out_delta": 0,
        },
        "measured_phase_swap_window": {
            "swap_in_delta": 4,
            "swap_out_delta": 0,
        },
    }

    audit = analysis._build_swap_audit(
        preflight,
        production,
        swap_policy=analysis.campaign.SWAP_POLICY_QUALITY_ONLY_SWAP_IN,
        timing_admissible=False,
    )

    assert audit["preflight"]["selected_policy_passed"] is False
    assert audit["preflight"]["zero_swap_out_observed"] is False
    assert audit["preflight"]["lifecycle"]["swap_out_bytes"] == 3
    assert audit["production_zero_swap_out_verified"] is True
    outer, reused, children = _synthetic_inputs()
    summary, _ = analysis.analyze(
        analysis.pair_outer_rows(outer, reused),
        children,
        execution_mode="sequential_fallback",
        swap_policy=analysis.campaign.SWAP_POLICY_QUALITY_ONLY_SWAP_IN,
        timing_admissible=False,
        swap_audit=audit,
    )
    assert summary["swap_audit"]["preflight"]["zero_swap_out_observed"] is False


def test_per_dataset_worst_validation_split_is_not_reused_from_test_metric():
    outer, reused, children = _synthetic_inputs()
    dataset = next(iter(analysis.campaign.TASKS))
    for row in outer:
        if row["dataset"] != dataset or row["arm"] != "auto":
            continue
        row["test_rmse"] = (1.2, 1.0, 0.8)[row["repeat"]]
        row["val_rmse"] = (0.8, 1.0, 1.2)[row["repeat"]]
    summary, per_dataset = analysis.analyze(
        analysis.pair_outer_rows(outer, reused), children
    )
    del summary
    row = next(
        item
        for item in per_dataset
        if item["contrast"] == "A10/M" and item["dataset"] == dataset
    )

    assert row["test_worst_split"] == "r0f0"
    assert row["val_worst_split"] == "r2f2"


def test_pair_outer_rows_requires_every_arm_and_reused_coordinate():
    outer, reused, _ = _synthetic_inputs()
    outer.pop()

    with pytest.raises(RuntimeError, match="grid is incomplete"):
        analysis.pair_outer_rows(outer, reused)


def test_paired_child_output_preserves_fitted_policy_metadata():
    child_rows = []
    for dataset, repeat, fold in analysis.campaign.expected_coordinates():
        for arm in ("baseline", "auto"):
            for child_fold in range(8):
                child_rows.append(
                    {
                        "dataset": dataset,
                        "repeat": repeat,
                        "fold": fold,
                        "arm": arm,
                        "child_fold": child_fold,
                        "best_iteration": 37,
                        "iterations_requested": 10_000,
                        "iterations_attempted": 52,
                        "rounds_completed": 51,
                        "rounds_retained": 37,
                        "resolved_learning_rate": 0.1,
                        "stop_reason": "early_stopping",
                        "selected_tree_mode": (
                            "lightgbm" if arm == "auto" else "catboost"
                        ),
                        "selected_lane": "boosting",
                        "deadline_hit": False,
                        "tree_mode_selection": (
                            {
                                "candidate_count": 3,
                                "fitted_candidate_count": 3,
                                "selected_candidate_index": 1,
                            }
                            if arm == "auto"
                            else None
                        ),
                        "wall_clock_elapsed_seconds": 1.25,
                    }
                )

    paired = analysis.pair_child_rows(child_rows)

    assert len(paired) == analysis.campaign.EXPECTED_PAIRED_CHILDREN
    first = paired[0]
    assert first["B10_iterations_requested"] == 10_000
    assert first["A10_iterations_attempted"] == 52
    assert first["B10_rounds_retained"] == 37
    assert first["A10_resolved_learning_rate"] == 0.1
    assert first["B10_selected_lane"] == "boosting"
    assert first["A10_deadline_hit"] is False
