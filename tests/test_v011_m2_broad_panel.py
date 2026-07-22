"""Contract tests for the frozen v0.11 M2 broad panel."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import analyze_v011_m2_broad_panel as analysis
from benchmarks import run_tabarena_regression_same_machine as historical
from benchmarks import run_v011_m2_broad_panel as campaign


@pytest.mark.parametrize(
    "relative",
    [
        "benchmarks/run_v011_m2_broad_panel.py",
        "benchmarks/analyze_v011_m2_broad_panel.py",
        "benchmarks/freeze_v011_m2_broad_panel.py",
    ],
)
def test_m2_clis_bootstrap_the_repo_from_a_clean_direct_invocation(
    relative, tmp_path
):
    result = subprocess.run(
        [sys.executable, "-I", str(campaign.ROOT / relative), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_m2_grid_is_exact_balanced_and_primary_only():
    grid = campaign.expected_ordered_grid()
    assert campaign.EXPECTED_PRIMARY_COORDINATES == 39
    assert campaign.EXPECTED_JOBS == 117
    assert campaign.EXPECTED_CHILD_FITS == 936
    assert len(grid) == len(set(grid)) == 117
    assert {row[0] for row in grid} == {"primary"}
    assert {(row[2], row[3]) for row in grid} == {(0, 0), (1, 1), (2, 2)}
    audit = campaign.expected_position_audit()["lane_position_counts"]["primary"]
    assert all(tuple(values.values()) == (13, 13, 13) for values in audit.values())
    assert campaign.job_order_sha256() == (
        "ed35ca18a759b74ab9f26373e2d253c5970d2dab3788e1139d23466429cf0385"
    )


def test_m2_uses_declared_product_pins_and_empty_manual_configs():
    assert campaign.CHIMERABOOST_TAG_COMMIT == (
        "f14be606b641f1bf0dc92bb14b3951f1fe631c6b"
    )
    assert campaign.CHIMERABOOST_VERSION == "0.18.0"
    assert campaign.CATBOOST_VERSION == "1.2.10"
    assert campaign.TABARENA_TAG_COMMIT == (
        "4cd1d2526874962daae048a6f2dcf34aa272f3fa"
    )
    assert campaign.AUTOGLUON_VERSION == "1.5.1b20260712"
    assert {spec["engine"] for spec in campaign.ARM_SPECS.values()} == {
        "darkofit",
        "chimeraboost",
        "catboost",
    }
    assert all(spec["config"] == {} for spec in campaign.ARM_SPECS.values())
    protocol = campaign.frozen_protocol()
    assert protocol["ensemble_candidate_included"] is False
    assert protocol["policy_advancement_allowed"] is False
    assert protocol["fresh_confirmation_or_lockbox_used"] is False
    execution = protocol["execution_dispatch"]
    assert execution["fresh_worker_boundary"] == "one_new_python_process_per_outer_job"
    assert execution["same_arm_worker_warmup"] == [
        "numeric_regression",
        "categorical_regression",
    ]
    assert execution["worker_count"] == 117
    assert execution["resume_allowed"] is False
    assert campaign.WORKER_ENVIRONMENT["NUMBA_NUM_THREADS"] == "18"
    assert protocol["darkofit_execution_source_pin"] == {
        "policy": "published_contract_commit_only",
        "required_parent": "harness_freeze_git_head",
        "only_path_added_after_harness_freeze": (
            "benchmarks/v011_m2_broad_panel_contract_20260722.json"
        ),
        "required_remote_ref": "origin/main",
    }


def test_chimeraboost_activation_imports_the_frozen_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "chimera"
    package = checkout / "chimeraboost"
    package.mkdir(parents=True)
    module_path = package / "__init__.py"
    module_path.write_text('__version__ = "0.18.0"\n', encoding="utf-8")
    responses = {
        ("rev-parse", "HEAD"): campaign.CHIMERABOOST_TAG_COMMIT,
        ("status", "--porcelain", "--untracked-files=all"): "",
        ("describe", "--tags", "--always"): campaign.CHIMERABOOST_DESCRIBE,
        ("rev-parse", "HEAD^{tree}"): "synthetic-tree",
        ("remote", "get-url", "origin"): "https://example.invalid/chimera.git",
    }

    def fake_git(args, *, cwd):
        assert cwd == checkout
        return responses[tuple(args)]

    monkeypatch.setattr(campaign._base, "_run_git", fake_git)
    original_path = list(sys.path)
    original_modules = campaign._base._loaded_chimeraboost_modules()
    for name in original_modules:
        sys.modules.pop(name, None)
    try:
        result = campaign.activate_chimeraboost_checkout(checkout)
        assert result["git_head"] == campaign.CHIMERABOOST_TAG_COMMIT
        assert Path(result["module_file"]) == module_path
        assert importlib.import_module("chimeraboost").__version__ == "0.18.0"
    finally:
        for name in campaign._base._loaded_chimeraboost_modules():
            sys.modules.pop(name, None)
        sys.modules.update(original_modules)
        sys.path[:] = original_path
        importlib.invalidate_caches()


def test_worker_attestation_metadata_round_trips_into_the_analyzer(
    tmp_path, monkeypatch
):
    expected = campaign.expected_ordered_grid()[0]
    engine = campaign.ARM_SPECS[expected[4]]["engine"]
    framework = analysis._expected_framework(expected[4])
    result_relative = str(
        Path("experiments")
        / "data"
        / framework
        / str(campaign.TASKS[expected[1]])
        / f"{expected[2]}_{expected[3]}"
        / "results.pkl"
    )
    result_path = tmp_path / result_relative
    result_path.parent.mkdir(parents=True)
    result_path.write_bytes(b"synthetic-result")
    result_artifact = campaign._base._stable_file_artifact(result_path, tmp_path)
    worker_path = campaign._worker_attestation_path(tmp_path, 0)
    worker_path.parent.mkdir(parents=True)
    stages = [
        {
            "name": f"{engine}_{input_kind}",
            "engine": engine,
            "input_kind": input_kind,
            "thread_count": campaign.EXPECTED_CHILD_CPUS,
            "representation": {"kind": "native"},
            "comparator_fit": {
                "engine": engine,
                "num_cpus": campaign.EXPECTED_CHILD_CPUS,
            },
        }
        for input_kind in ("numeric", "categorical")
    ]
    payload = {
        "schema_version": 1,
        "kind": campaign.CAMPAIGN_KIND + "_worker",
        "worker_index": 0,
        "pid": 456,
        "parent_pid": 123,
        "started_at_utc": "2026-07-22T00:00:00+00:00",
        "completed_at_utc": "2026-07-22T00:01:00+00:00",
        "coordinate": {
            "lane": expected[0],
            "dataset": expected[1],
            "repeat": expected[2],
            "fold": expected[3],
            "arm": expected[4],
            "engine": engine,
        },
        "same_arm_warmup": {
            "engine": engine,
            "stage_names": [stage["name"] for stage in stages],
            "stages": stages,
            "warnings": [],
        },
        "environment": campaign.WORKER_ENVIRONMENT,
        "numba_thread_ceiling": campaign.EXPECTED_CHILD_CPUS,
        "numba_current_threads_after_fit": campaign.EXPECTED_CHILD_CPUS,
        "result_artifact": result_artifact,
    }
    worker_path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True), encoding="utf-8"
    )
    job = object()
    monkeypatch.setattr(campaign, "EXPECTED_JOBS", 1)
    monkeypatch.setattr(campaign, "expected_ordered_grid", lambda: [expected])
    monkeypatch.setattr(campaign._base, "_result_path", lambda output, item: result_path)
    worker_artifacts = campaign._validate_worker_attestations(
        tmp_path, [job], parent_pid=123
    )
    worker_relative = str(worker_path.relative_to(tmp_path))
    assert set(worker_artifacts[worker_relative]) == {"sha256", "size_bytes"}
    analysis._verify_worker_attestations(
        tmp_path,
        worker_artifacts,
        {result_relative: {k: v for k, v in result_artifact.items() if k != "path"}},
        parent_pid=123,
    )


def test_m2_internal_worker_arguments_are_paired_and_resume_is_forbidden():
    with pytest.raises(SystemExit):
        campaign.parse_args(["--worker-index", "0"])
    with pytest.raises(SystemExit):
        campaign.parse_args(["--parent-pid", "123"])
    with pytest.raises(SystemExit):
        campaign.parse_args(["--resume"])
    args = campaign.parse_args(["--worker-index", "0", "--parent-pid", "123"])
    assert args.worker_index == 0
    assert args.parent_pid == 123


def test_execution_source_pin_requires_the_published_contract_only_commit(monkeypatch):
    freeze = "1" * 40
    current = "2" * 40
    responses = {
        ("rev-parse", "HEAD"): current,
        ("rev-list", "--parents", "-n", "1", current): f"{current} {freeze}",
        ("diff", "--name-only", freeze, current): (
            "benchmarks/v011_m2_broad_panel_contract_20260722.json"
        ),
        ("rev-parse", "origin/main"): current,
    }

    def fake_git(args, *, cwd):
        assert cwd == campaign.ROOT
        return responses[tuple(args)]

    monkeypatch.setattr(campaign._base, "_run_git", fake_git)
    assert campaign.validate_execution_source_pin(
        {"harness_freeze_git_head": freeze}
    ) == current
    responses[("diff", "--name-only", freeze, current)] += "\ndarkofit/booster.py"
    with pytest.raises(RuntimeError, match="changed more"):
        campaign.validate_execution_source_pin({"harness_freeze_git_head": freeze})


def test_configured_base_is_scoped_and_restored_after_failure():
    before = (
        historical.EXPECTED_JOBS,
        historical.LANES,
        historical.CHIMERABOOST_VERSION,
        historical.SOURCE_FILES,
    )
    with pytest.raises(RuntimeError, match="sentinel"):
        with campaign.configured_base() as configured:
            assert configured.EXPECTED_JOBS == 117
            assert configured.LANES == ("primary",)
            assert configured.CHIMERABOOST_VERSION == "0.18.0"
            assert len(configured.expected_ordered_grid()) == 117
            raise RuntimeError("sentinel")
    assert (
        historical.EXPECTED_JOBS,
        historical.LANES,
        historical.CHIMERABOOST_VERSION,
        historical.SOURCE_FILES,
    ) == before


def test_contract_is_create_only_and_binding_checked_when_frozen():
    if not campaign.CONTRACT_PATH.exists():
        pytest.skip("prospective M2 contract has not been frozen yet")
    contract = campaign.load_contract()
    assert contract["contract_id"] == campaign.CONTRACT_ID
    assert contract["outcome_blind"] is True
    assert contract["protocol"] == campaign.frozen_protocol()
    assert set(contract["bindings"]) == set(campaign.BOUND_PATHS)


def _synthetic_outer_rows() -> list[dict]:
    factors = {"D": 0.9, "M": 1.0, "C": 1.1}
    rows = []
    for lane, dataset, repeat, fold, arm in campaign.expected_ordered_grid():
        code = campaign.ARM_SPECS[arm]["code"]
        factor = factors[code]
        rows.append(
            {
                "lane": lane,
                "dataset": dataset,
                "repeat": repeat,
                "fold": fold,
                "arm": arm,
                "test_rmse": factor,
                "val_rmse": factor * 1.01,
                "train_time_s": factor * 10.0,
                "infer_time_s": factor * 0.1,
                "incremental_memory_bytes": factor * 1_000.0,
                "peak_memory_bytes": factor * 2_000.0,
            }
        )
    return rows


def test_m2_analysis_reports_equal_dataset_ratios_and_head_to_head():
    paired = analysis.pair_rows(_synthetic_outer_rows())
    summary, datasets = analysis.summarize(paired, {"synthetic": True})
    assert len(paired) == 117
    assert len(datasets) == 39
    comparisons = {item["contrast"]: item for item in summary["comparisons"]}
    assert comparisons["D/M"]["metrics"]["test_rmse"]["ratio"] == pytest.approx(0.9)
    assert comparisons["D/C"]["metrics"]["test_rmse"]["ratio"] == pytest.approx(
        0.9 / 1.1
    )
    head_to_head = comparisons["M/C"]["head_to_head"]["equal_dataset_quality"]
    assert head_to_head == {
        "wins": 13,
        "losses": 0,
        "ties": 0,
        "win_rate_excluding_ties": 1.0,
        "win_share_with_half_ties": 1.0,
    }
    assert summary["policy_advancement_allowed"] is False


def test_m2_protocol_document_names_non_authorized_actions():
    text = Path(
        "benchmarks/v011_m2_broad_panel_protocol_20260722.md"
    ).read_text(encoding="utf-8")
    assert "cannot select a default" in text
    assert "authorize TabArena-Lite" in text
    assert "authorize a release" in text
    assert "No private or public ensemble candidate is an arm" in text
    assert "each in a newly launched Python" in text
    assert "Resume is forbidden" in text
