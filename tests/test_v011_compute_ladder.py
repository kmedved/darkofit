"""Contract tests for the v0.11 release compute ladder."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from argparse import Namespace
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest

from benchmarks import analyze_v011_compute_ladder as analysis
from benchmarks import freeze_v011_compute_ladder as freezer
from benchmarks import run_v011_compute_ladder as campaign


@pytest.mark.parametrize(
    "relative",
    [
        "benchmarks/run_v011_compute_ladder.py",
        "benchmarks/analyze_v011_compute_ladder.py",
        "benchmarks/freeze_v011_compute_ladder.py",
    ],
)
def test_compute_ladder_clis_bootstrap_from_isolated_invocations(relative, tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", str(campaign.ROOT / relative), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_compute_ladder_grid_and_position_balance_are_exact():
    grid = campaign.expected_ordered_grid()
    assert campaign.EXPECTED_COORDINATES == 39
    assert campaign.EXPECTED_WORKERS == 234
    assert len(grid) == len(set(grid)) == 234
    assert {(row[1], row[2]) for row in grid} == {(0, 0), (1, 1), (2, 2)}
    assert {row[3] for row in grid} == set(campaign.ARM_SPECS)
    assert all(sum(values) == 39 for values in campaign.position_audit().values())
    assert all(
        max(values) - min(values) == 1 for values in campaign.position_audit().values()
    )
    assert campaign.ordered_grid_sha256() == (
        "5fbd9dc0d3e70c6c16f2daa2bc52f6c84d343211b145070c5afb73d66cd9e2be"
    )


def test_compute_ladder_public_points_and_claim_boundary_are_frozen():
    assert campaign.ARM_SPECS[campaign.DARKO_DEFAULT]["config"] == {}
    assert campaign.ARM_SPECS[campaign.DARKO_ACCURACY]["config"] == {
        "preset": "accuracy"
    }
    assert campaign.ARM_SPECS[campaign.DARKO_ENSEMBLE]["config"] == {
        "ensemble_mode": "v3",
        "n_ensembles": 8,
    }
    assert campaign.ARM_SPECS[campaign.CHIMERA_DEFAULT]["config"] == {}
    assert campaign.ARM_SPECS[campaign.CHIMERA_ACCURACY]["config"] == {"depth": 10}
    assert campaign.ARM_SPECS[campaign.CHIMERA_ENSEMBLE]["config"] == {"n_ensembles": 8}
    execution = campaign.execution_spec()
    assert execution["successor"] == {
        "supersedes_contract_id": "v011-release-compute-ladder-20260722-v2",
        "v2_fit_count": 0,
        "v2_worker_count": 0,
        "scientific_protocol_change": "none",
        "harness_change": "none",
        "only_execution_change": (
            "standalone_sleep_inhibitor_not_benchmark_command_wrapper"
        ),
    }
    assert execution["product_direct"] is True
    assert execution["autogluon_outer_bag"] is False
    assert execution["resources"]["threads"] == 14
    assert execution["prediction_timing"]["maximum_calls"] == 65_536
    assert campaign.claim_spec() == {
        "tier": "E",
        "spent_descriptive_release_scoreboard": True,
        "default_or_policy_advancement": False,
        "fresh_confirmation": False,
        "lockbox": False,
        "tabarena_placement": False,
        "catboost_comparison": False,
        "no_rerun_to_improve": True,
    }
    spec = campaign.analysis_spec()
    assert spec["frontiers"]["axes"] == [
        "fit_seconds",
        "prediction_seconds_per_call",
    ]
    assert spec["strict_program_verdict"] == {
        "basis": "predeclared_equal_dataset_point_estimates",
        "uncertainty_adjacent_not_certificate": True,
        "fit_frontier_dominance": True,
        "prediction_frontier_dominance": True,
        "counterpart_peak_rss_no_worse": True,
    }


def test_protocol_names_direct_product_boundary_and_all_victory_axes():
    text = campaign.PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "64-model nested bag" in text
    assert "65,536 calls" in text
    assert "fit and prediction frontiers" in text
    assert "process-tree peak RSS" in text
    assert "strict program verdict additionally requires" in text
    assert "all three points" in text
    assert "wrap product estimators" in text
    assert "AutoGluon's eight-fold bag" in text
    assert "CatBoost and TabArena placement are intentionally outside" in text
    assert "Contract v2 supersedes v1" in text
    assert "stopped before any" in text
    assert "worker or model fit" in text
    assert "Contract v3 supersedes v2" in text
    assert "zero workers and zero model fits" in text


def test_v1_terminal_is_hash_bound_and_records_zero_outcomes():
    terminal = json.loads(campaign.V1_TERMINAL_PATH.read_text(encoding="utf-8"))
    assert terminal["contract_id"] == campaign.V1_CONTRACT_ID
    assert terminal["status"] == "superseded_pre_execution"
    assert terminal["worker_count"] == terminal["fit_count"] == 0
    assert terminal["model_outcome_count"] == 0
    assert terminal["scientific_protocol_change"] == "none"
    assert terminal["v1_contract"] == {
        "bytes": campaign.V1_CONTRACT_PATH.stat().st_size,
        "path": str(campaign.V1_CONTRACT_PATH.relative_to(campaign.ROOT)),
        "sha256": campaign.sha256(campaign.V1_CONTRACT_PATH),
    }


def test_v2_terminal_is_hash_bound_and_records_zero_outcomes():
    terminal = json.loads(campaign.V2_TERMINAL_PATH.read_text(encoding="utf-8"))
    assert terminal["contract_id"] == campaign.V2_CONTRACT_ID
    assert terminal["status"] == "superseded_pre_execution"
    assert terminal["worker_count"] == terminal["fit_count"] == 0
    assert terminal["model_outcome_count"] == 0
    assert terminal["scientific_protocol_change"] == "none"
    assert terminal["v2_contract"] == {
        "bytes": campaign.V2_CONTRACT_PATH.stat().st_size,
        "path": str(campaign.V2_CONTRACT_PATH.relative_to(campaign.ROOT)),
        "sha256": campaign.sha256(campaign.V2_CONTRACT_PATH),
    }
    encoded = (
        json.dumps(
            terminal["terminal_payload"],
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    assert terminal["terminal_artifact"] == {
        "bytes": len(encoded),
        "sha256": campaign._sha256_bytes(encoded),
    }


def test_internal_worker_arguments_are_all_or_none():
    with pytest.raises(SystemExit):
        campaign.parse_args(["--worker-index", "0"])
    with pytest.raises(SystemExit):
        campaign.parse_args(["--worker-index", "0", "--arm", campaign.DARKO_DEFAULT])
    args = campaign.parse_args(
        [
            "--worker-index",
            "0",
            "--arm",
            campaign.expected_ordered_grid()[0][3],
            "--parent-pid",
            "123",
            "--worker-started-at",
            "2026-07-22T00:00:00+00:00",
        ]
    )
    assert args.worker_index == 0


def test_execution_source_pin_requires_one_published_contract_only_child(monkeypatch):
    freeze = "1" * 40
    current = "2" * 40
    responses = {
        ("rev-parse", "HEAD"): current,
        ("rev-list", "--parents", "-n", "1", current): f"{current} {freeze}",
        ("diff", "--name-only", freeze, current): str(
            campaign.CONTRACT_PATH.relative_to(campaign.ROOT)
        ),
        ("rev-parse", "origin/main"): current,
    }

    def fake_git(repo, *args):
        assert repo == campaign.ROOT
        return responses[args]

    monkeypatch.setattr(campaign, "_git", fake_git)
    assert (
        campaign.validate_execution_source_pin({"harness_freeze_git_head": freeze})
        == current
    )
    responses[("diff", "--name-only", freeze, current)] += "\ndarkofit/booster.py"
    with pytest.raises(RuntimeError, match="changed more"):
        campaign.validate_execution_source_pin({"harness_freeze_git_head": freeze})


def test_latest_release_preflight_closes_on_a_new_rival_release(monkeypatch):
    expected = {
        "tag_name": campaign.CHIMERABOOST_TAG,
        "published_at": campaign.CHIMERABOOST_RELEASE_PUBLISHED_AT,
        "html_url": "https://example.invalid/v0.20.0",
    }
    monkeypatch.setattr(campaign, "_run", lambda *args, **kwargs: json.dumps(expected))
    observed = campaign.validate_latest_chimeraboost_release()
    assert observed["tag_name"] == "v0.20.0"
    expected["tag_name"] = "v0.21.0"
    with pytest.raises(RuntimeError, match="latest release changed"):
        campaign.validate_latest_chimeraboost_release()


def test_exclusive_machine_scan_ignores_only_its_launch_ancestors(monkeypatch):
    class FakeError(Exception):
        pass

    class FakeProcess:
        def __init__(self, pid, command, parent=None):
            self.pid = pid
            self.info = {"pid": pid, "cmdline": command.split()}
            self._parent = parent

        def parent(self):
            return self._parent

    launch = FakeProcess(101, "zsh run_v011_compute_ladder.py --dry-run")
    current = FakeProcess(os.getpid(), "python runner", parent=launch)
    harmless = FakeProcess(202, "python unrelated.py")
    processes = [current, launch, harmless]
    fake_psutil = SimpleNamespace(
        Process=lambda pid=None: current,
        process_iter=lambda fields: list(processes),
        AccessDenied=FakeError,
        NoSuchProcess=FakeError,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    audit = campaign._exclusive_machine_audit()
    assert audit["ignored_launch_ancestor_pids"] == [101]
    assert audit["conflicting_benchmark_processes"] == []
    processes.append(FakeProcess(303, "python run_v011_compute_ladder.py"))
    with pytest.raises(RuntimeError, match="another benchmark"):
        campaign._exclusive_machine_audit()


def test_prediction_timer_excludes_post_call_validation_overhead(monkeypatch):
    class SlowArray:
        def __array__(self, dtype=None, copy=None):
            del copy
            time.sleep(0.01)
            return np.asarray([1.0, 2.0], dtype=dtype)

    class Model:
        def predict(self, X):
            assert len(X) == 2
            return SlowArray()

    monkeypatch.setattr(campaign, "PREDICTION_TARGET_SECONDS", 0.0001)
    monkeypatch.setattr(campaign, "PREDICTION_MIN_SECONDS", 0.0)
    monkeypatch.setattr(campaign, "PREDICTION_MAX_CALLS", 3)
    result = campaign._timed_prediction(Model(), [0, 1])
    assert result["calls"] == 3
    assert max(result["pilots_seconds"]) < 0.005
    assert result["prediction_sha256"]


def test_split_fingerprint_preserves_schema_and_values():
    pd = pytest.importorskip("pandas")
    first = pd.DataFrame(
        {
            "numeric": [1.0, 2.0],
            "category": pd.Series(["a", "b"], dtype="category"),
        }
    )
    same = first.copy()
    changed = first.copy()
    changed["numeric"] = changed["numeric"].astype("float32")
    assert campaign._pandas_sha256(first) == campaign._pandas_sha256(same)
    assert campaign._pandas_sha256(first) != campaign._pandas_sha256(changed)


def test_product_activation_preserves_warmed_modules_and_evicts_wrong_sources(
    tmp_path,
):
    darko_source = tmp_path / "darko"
    chimera_source = tmp_path / "chimera"
    darko_package = darko_source / "darkofit"
    chimera_package = chimera_source / "chimeraboost"
    darko_package.mkdir(parents=True)
    chimera_package.mkdir(parents=True)
    darko = ModuleType("darkofit")
    darko.__file__ = str(darko_package / "__init__.py")
    darko_core = ModuleType("darkofit.core")
    darko_core.__file__ = str(darko_package / "core.py")
    wrong_chimera = ModuleType("chimeraboost")
    wrong_chimera.__file__ = str(tmp_path / "wrong" / "chimeraboost" / "__init__.py")
    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == "darkofit"
        or name.startswith("darkofit.")
        or name == "chimeraboost"
        or name.startswith("chimeraboost.")
    }
    previous_path = list(sys.path)
    try:
        for name in previous:
            sys.modules.pop(name, None)
        sys.modules["darkofit"] = darko
        sys.modules["darkofit.core"] = darko_core
        sys.modules["chimeraboost"] = wrong_chimera
        campaign._activate_product_sources(darko_source, chimera_source)
        assert sys.modules["darkofit"] is darko
        assert sys.modules["darkofit.core"] is darko_core
        assert "chimeraboost" not in sys.modules
    finally:
        for name in list(sys.modules):
            if (
                name == "darkofit"
                or name.startswith("darkofit.")
                or name == "chimeraboost"
                or name.startswith("chimeraboost.")
            ):
                sys.modules.pop(name, None)
        sys.modules.update(previous)
        sys.path[:] = previous_path


def _synthetic_rows() -> list[dict]:
    factors = {
        campaign.DARKO_DEFAULT: (0.95, 0.8, 0.5, 0.8, 0.7),
        campaign.DARKO_ACCURACY: (0.85, 1.2, 0.6, 0.9, 0.8),
        campaign.DARKO_ENSEMBLE: (0.70, 4.0, 2.0, 0.95, 0.9),
        campaign.CHIMERA_DEFAULT: (1.0, 1.0, 1.0, 1.0, 1.0),
        campaign.CHIMERA_ACCURACY: (0.90, 1.5, 1.1, 1.1, 1.2),
        campaign.CHIMERA_ENSEMBLE: (0.75, 5.0, 8.0, 1.2, 1.4),
    }
    rows = []
    for dataset, repeat, fold, arm in campaign.expected_ordered_grid():
        quality, fit, predict, peak, delta = factors[arm]
        rows.append(
            {
                "dataset": dataset,
                "repeat": repeat,
                "fold": fold,
                "arm": arm,
                "engine": campaign.ARM_SPECS[arm]["engine"],
                "profile": campaign.ARM_SPECS[arm]["profile"],
                "test_rmse": quality,
                "fit_seconds": fit,
                "prediction": {"seconds_per_call": predict},
                "fit_rss": {
                    "peak_bytes": peak * 1_000_000,
                    "peak_delta_bytes": delta * 1_000_000,
                },
            }
        )
    return rows


def test_analysis_equal_weights_datasets_and_builds_both_frontiers(monkeypatch):
    monkeypatch.setattr(campaign, "BOOTSTRAP_DRAWS", 100)
    paired = analysis.pair_rows(_synthetic_rows())
    summary, per_dataset = analysis.summarize(paired, {"synthetic": True})
    references = {
        item["numerator_arm"]: item for item in summary["arms_vs_chimeraboost_default"]
    }
    assert references[campaign.DARKO_DEFAULT]["metrics"]["test_rmse"][
        "ratio"
    ] == pytest.approx(0.95)
    assert len(paired) == 9 * 39
    assert len(per_dataset) == 9 * 13
    assert summary["frontiers"]["fit_seconds"]["darkofit_full_curve_dominance"] is True
    assert (
        summary["frontiers"]["prediction_seconds_per_call"][
            "darkofit_full_curve_dominance"
        ]
        is True
    )
    assert summary["memory_retention"]["all_no_worse"] is True
    assert summary["strict_program_verdict"]["strict_pareto_victory"] is True


def _synthetic_worker(worker_index: int) -> dict:
    dataset, repeat, fold, arm = campaign.expected_ordered_grid()[worker_index]
    spec = campaign.ARM_SPECS[arm]
    routes = (
        ["darkofit_catboost", "darkofit_lightgbm", "darkofit_hybrid"]
        if arm == campaign.DARKO_ACCURACY
        else [f"{spec['engine']}_{spec['profile']}"]
    )
    return {
        "schema_version": 1,
        "kind": "v011_compute_ladder_worker",
        "worker_index": worker_index,
        "pid": 456 + worker_index,
        "parent_pid": 123,
        "started_at_utc": "2026-07-22T00:00:00+00:00",
        "completed_at_utc": "2026-07-22T00:01:00+00:00",
        "dataset": dataset,
        "task_id": campaign.TASKS[dataset],
        "repeat": repeat,
        "fold": fold,
        "arm": arm,
        "code": spec["code"],
        "engine": spec["engine"],
        "profile": spec["profile"],
        "seed": campaign._coordinate_seed(repeat, fold),
        "train_rows": 20,
        "test_rows": 10,
        "feature_count": 2,
        "categorical_features": [],
        "fingerprints": {
            "X_train": "a" * 64,
            "y_train": "b" * 64,
            "X_test": "c" * 64,
            "y_test": "d" * 64,
            "combined_sha256": "e" * 64,
        },
        "test_rmse": 1.0,
        "fit_seconds": 2.0,
        "fit_rss": {
            "scope": "worker_plus_recursive_children",
            "start_bytes": 100,
            "peak_bytes": 200,
            "peak_delta_bytes": 100,
            "end_bytes": 150,
            "samples": 2,
            "errors": [],
            "interval_seconds": campaign.RSS_INTERVAL_SECONDS,
        },
        "prediction": {
            "rows": 10,
            "pilots_seconds": [0.2, 0.2, 0.2],
            "pilot_median_seconds": 0.2,
            "calls": 3,
            "interval_seconds": 0.6,
            "seconds_per_call": 0.2,
            "rows_per_second": 50.0,
            "prediction_sha256": "f" * 64,
        },
        "prediction_sha256": "f" * 64,
        "model": {},
        "implementation": {},
        "warmup": {
            "rows": 20,
            "categorical_feature_count": 0,
            "routes": routes,
            "reduced_iteration_budget": True,
        },
        "environment": campaign.WORKER_ENVIRONMENT,
        "numba_threads_before_fit": campaign.THREADS,
        "numba_threads_after_fit": campaign.THREADS,
        "warnings": [],
        "launcher_output": {
            "returncode": 0,
            "stdout_without_result": [],
            "stderr": "",
        },
    }


def test_worker_validator_rejects_inconsistent_process_tree_rss(monkeypatch):
    row = _synthetic_worker(0)
    monkeypatch.setattr(analysis, "_verify_worker_model", lambda *args, **kwargs: None)
    clean, parent = analysis._verify_worker(
        row,
        expected_index=0,
        parent_pid=None,
        manifest={"darkofit_source": {"path": "."}},
    )
    assert clean["fit_seconds"] == 2.0
    assert parent == 123
    row["fit_rss"]["peak_delta_bytes"] = 99
    with pytest.raises(RuntimeError, match="RSS telemetry"):
        analysis._verify_worker(
            row,
            expected_index=0,
            parent_pid=None,
            manifest={"darkofit_source": {"path": "."}},
        )


def test_campaign_verifier_checks_the_complete_create_only_artifact_set(
    tmp_path, monkeypatch
):
    output = tmp_path / "campaign"
    workers = output / "workers"
    workers.mkdir(parents=True)
    rows = []
    artifacts = []
    for index in range(campaign.EXPECTED_WORKERS):
        row = _synthetic_worker(index)
        path = workers / f"{index:03d}.json"
        campaign._write_create_only_json(path, row)
        rows.append(row)
        artifacts.append(campaign._stable_artifact(path, output))
    manifest_path = output / "manifest.json"
    synthetic_manifest = {
        "synthetic": True,
        "darkofit_source": {"path": "."},
        "chimeraboost_source": {"path": "."},
    }
    campaign._write_create_only_json(manifest_path, synthetic_manifest)
    completed_at = "2026-07-22T01:00:00+00:00"
    raw = {
        "schema_version": 1,
        "kind": "v011_compute_ladder_raw",
        "contract_id": campaign.CONTRACT_ID,
        "started_at_utc": "2026-07-22T00:00:00+00:00",
        "completed_at_utc": completed_at,
        "manifest": campaign._stable_artifact(manifest_path, output),
        "workers": artifacts,
        "rows": rows,
    }
    raw_path = output / "raw.json"
    campaign._write_create_only_json(raw_path, raw)
    campaign._write_create_only_json(
        output / "terminal.json",
        {
            "schema_version": 1,
            "kind": "v011_compute_ladder_terminal",
            "status": "complete",
            "contract_id": campaign.CONTRACT_ID,
            "completed_worker_count": campaign.EXPECTED_WORKERS,
            "raw": campaign._stable_artifact(raw_path, output),
            "completed_at_utc": completed_at,
        },
    )
    monkeypatch.setattr(
        campaign,
        "load_contract",
        lambda: {
            "contract_id": campaign.CONTRACT_ID,
            "protocol_sha256": campaign.protocol_sha256(),
        },
    )
    monkeypatch.setattr(
        analysis,
        "_verify_source_manifest",
        lambda value: {"synthetic_manifest": value["synthetic"]},
    )
    monkeypatch.setattr(analysis, "_verify_worker_model", lambda *args, **kwargs: None)
    manifest, verified, provenance = analysis.verify_campaign(output)
    assert manifest == synthetic_manifest
    assert len(verified) == campaign.EXPECTED_WORKERS
    assert provenance["raw_sha256"] == campaign.sha256(raw_path)
    with (workers / "000.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(RuntimeError, match="artifact digest changed"):
        analysis.verify_campaign(output)


def test_freezer_is_create_only_and_binds_every_harness_file(tmp_path, monkeypatch):
    head = "1" * 40

    def fake_git(repo, *args):
        assert repo == campaign.ROOT
        if args[0] == "status":
            return ""
        assert args == ("rev-parse", "HEAD")
        return head

    monkeypatch.setattr(campaign, "_git", fake_git)
    output = tmp_path / "contract.json"
    contract = freezer.freeze(output)
    assert contract["harness_freeze_git_head"] == head
    assert contract["execution"] == campaign.execution_spec()
    assert contract["analysis"] == campaign.analysis_spec()
    assert set(contract["bindings"]) == set(campaign.BOUND_PATHS)
    with pytest.raises(FileExistsError):
        freezer.freeze(output)


def test_parent_binds_raw_and_terminal_to_one_completion_time(tmp_path, monkeypatch):
    coordinate = campaign.expected_ordered_grid()[0]
    arm = coordinate[3]
    output = tmp_path / "campaign"
    sources = {
        "darkofit": {"path": "darko"},
        "chimeraboost": {"path": "chimera"},
        "tabarena": {"path": "tabarena"},
    }
    manifest = {
        "darkofit_source": sources["darkofit"],
        "chimeraboost_source": sources["chimeraboost"],
        "tabarena_source": sources["tabarena"],
    }
    payload = {
        "worker_index": 0,
        "dataset": coordinate[0],
        "repeat": coordinate[1],
        "fold": coordinate[2],
        "arm": arm,
        "parent_pid": os.getpid(),
    }
    monkeypatch.setattr(campaign, "EXPECTED_WORKERS", 1)
    monkeypatch.setattr(campaign, "expected_ordered_grid", lambda: [coordinate])
    monkeypatch.setattr(campaign, "load_contract", lambda path: {})
    monkeypatch.setattr(campaign, "validate_execution_source_pin", lambda value: "h")
    monkeypatch.setattr(campaign, "_git", lambda *args: "")
    monkeypatch.setattr(campaign, "_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(campaign, "validate_product_sources", lambda *args: sources)
    monkeypatch.setattr(
        campaign.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=campaign.WORKER_PREFIX + json.dumps(payload) + "\n",
            stderr="",
        ),
    )
    args = Namespace(
        contract=tmp_path / "contract.json",
        output_dir=output,
        darkofit_source=tmp_path / "darko",
        chimeraboost_source=tmp_path / "chimera",
        tabarena_source=tmp_path / "tabarena",
    )
    assert campaign._run_parent(args) == 0
    raw = campaign._read_json(output / "raw.json")
    terminal = campaign._read_json(output / "terminal.json")
    assert raw["completed_at_utc"] == terminal["completed_at_utc"]
    assert terminal["status"] == "complete"


def test_parent_records_a_terminal_failure_when_manifest_preflight_closes(
    tmp_path, monkeypatch
):
    output = tmp_path / "campaign"
    monkeypatch.setattr(campaign, "load_contract", lambda path: {})
    monkeypatch.setattr(campaign, "validate_execution_source_pin", lambda value: "h")
    monkeypatch.setattr(campaign, "_git", lambda *args: "")
    monkeypatch.setattr(
        campaign,
        "_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("new release")),
    )
    args = Namespace(
        contract=tmp_path / "contract.json",
        output_dir=output,
        darkofit_source=tmp_path / "darko",
        chimeraboost_source=tmp_path / "chimera",
        tabarena_source=tmp_path / "tabarena",
    )
    with pytest.raises(RuntimeError, match="new release"):
        campaign._run_parent(args)
    terminal = campaign._read_json(output / "terminal.json")
    assert terminal["status"] == "failed"
    assert terminal["completed_worker_count"] == 0
    assert terminal["error"] == "new release"


def test_contract_is_binding_once_frozen():
    if not campaign.CONTRACT_PATH.exists():
        pytest.skip("prospective compute-ladder contract has not been frozen")
    contract = campaign.load_contract()
    assert contract["contract_id"] == campaign.CONTRACT_ID
    assert contract["outcome_blind"] is True
    assert contract["execution"] == campaign.execution_spec()
    assert contract["analysis"] == campaign.analysis_spec()
    assert set(contract["bindings"]) == set(campaign.BOUND_PATHS)
