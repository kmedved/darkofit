from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmarks import analyze_panel3_cross_power_calibration as analyzer
from benchmarks import freeze_panel3_cross_power_calibration as freeze
from benchmarks import panel3_registry_common as common
from benchmarks import run_panel3_cross_power_calibration as runner
from benchmarks import run_tabarena_regression_followon_screen as spent


def _hash(character: str) -> str:
    return character * 64


def _fit_metadata() -> dict:
    return {
        "best_iteration": 10,
        "fitted_tree_count": 10,
        "resolved_learning_rate": 0.1,
        "selected_tree_mode": "catboost",
        "selected_lane": "boosting",
        "final_fit": {"stop_reason": "iteration_limit"},
    }


def _selection_fit(name: str, score: float) -> dict:
    metadata = _fit_metadata()
    if "linear" in name:
        metadata["selected_lane"] = "linear_leaves"
    return {
        "name": name,
        "validation_rmse": score,
        "fit_seconds": 0.2,
        "fit_metadata": metadata,
        "validation": {"source": "explicit_eval_set"},
    }


def _prediction_timing() -> dict:
    return {
        "per_call_median_seconds": 0.01,
        "per_call_min_seconds": 0.009,
        "per_call_max_seconds": 0.011,
        "total_seconds": 0.25,
        "call_count": 25,
        "minimum_block_seconds": 0.25,
    }


def _metadata(arm: str, *, engaged: bool, applicable: bool) -> dict:
    outer_rows = 2_400 if applicable else 1_500
    validation_rows = int(outer_rows * 0.2)
    inner_split = {
        "policy": "ShuffleSplit",
        "random_state": 4,
        "validation_fraction": 0.2,
        "train_rows": outer_rows - validation_rows,
        "validation_rows": validation_rows,
        "train_positions_sha256": _hash("7"),
        "validation_positions_sha256": _hash("8"),
    }
    if arm == runner.CONTROL_ARM:
        return {
            "kind": arm,
            "engaged": False,
            "selected_configuration": "product_default",
            "final_fit": _fit_metadata(),
        }
    if arm == "t5_composite_policy":
        if not applicable:
            return {
                "kind": arm,
                "engaged": False,
                "selected_configuration": "product_default",
                "final_fit": _fit_metadata(),
                "decline_reason": "below_size_gate",
                "size_gate": runner.T5_SIZE_GATE,
                "total_selection_fit_seconds": 0.0,
                "policy_overhead_seconds": 0.1,
                "final_fit_seconds": 0.9,
            }
        ratio = 0.9 if engaged else 1.0
        control_score = 10.0
        challenger_score = control_score * ratio
        selection_fits = [
            _selection_fit("control_audition", control_score),
            _selection_fit("challenger_auto", challenger_score),
            _selection_fit(
                "challenger_catboost_linear",
                challenger_score + 1.0,
            ),
        ]
        return {
            "kind": arm,
            "engaged": engaged,
            "size_gate": runner.T5_SIZE_GATE,
            "decline_reason": None if engaged else "outer_validation_guard",
            "split": inner_split,
            "outer_guard_ratio": runner.panel3.t5.OUTER_GUARD_RATIO,
            "cross_guard_ratio": runner.panel3.t5.CROSS_GUARD_RATIO,
            "selection_rounds": runner.panel3.t5.SELECTION_ROUNDS,
            "control_validation_rmse": control_score,
            "challenger_validation_rmse": challenger_score,
            "relative_challenger_validation_ratio": ratio,
            "selected_configuration": (
                "challenger" if engaged else "product_default"
            ),
            "selected_tree_mode": "catboost",
            "selected_linear_leaves": False,
            "selected_crosses": False,
            "selected_cross_pairs": [],
            "selected_cross_pair_count": 0,
            "selected_best_iteration": 10,
            "selected_resolved_learning_rate": 0.1,
            "selection_fits": selection_fits,
            "total_selection_fit_seconds": 0.6,
            "policy_overhead_seconds": 0.1,
            "final_transform_seconds": 0.0,
            "final_fit_seconds": 0.3,
            "final_fit": _fit_metadata(),
        }
    ratio = 0.9 if engaged else 0.99
    uncrossed = 10.0
    crossed = uncrossed * ratio
    pairs = [[0, 1, "diff"]]
    base_fit = _selection_fit("uncrossed_constant", uncrossed)
    linear_fit = _selection_fit("uncrossed_linear", uncrossed + 1.0)
    crossed_fit = _selection_fit("crossed_selected_leaf_lane", crossed)
    crossed_fit["pairs"] = pairs
    crossed_fit["transform_seconds"] = 0.01
    selection_fits = [base_fit, linear_fit, crossed_fit]
    return {
        "kind": arm,
        "engaged": engaged,
        "selected_crosses": engaged,
        "decline_reason": None if engaged else "cross_guard",
        "cross_guard_ratio": runner.panel3.GUARDED_CROSS_RATIO,
        "split": inner_split,
        "selected_configuration": "crossed" if engaged else "uncrossed",
        "selected_linear_leaves": False,
        "candidate_cross_pairs": pairs,
        "selected_cross_pairs": pairs if engaged else [],
        "selected_cross_pair_count": 1 if engaged else 0,
        "uncrossed_validation_rmse": uncrossed,
        "crossed_validation_rmse": crossed,
        "relative_crossed_validation_ratio": ratio,
        "selected_best_iteration": 10,
        "selected_resolved_learning_rate": 0.1,
        "selected_selection_fit": crossed_fit if engaged else base_fit,
        "selection_fits": selection_fits,
        "total_selection_fit_seconds": 0.6,
        "policy_overhead_seconds": 0.1,
        "final_transform_seconds": 0.0,
        "final_model_fit_seconds": 0.3,
        "final_fit_seconds": 0.3,
        "final_refit_parameters": {
            "iterations": 10,
            "learning_rate": 0.1,
            "tree_mode": "catboost",
            "linear_leaves": False,
            "crossed": engaged,
        },
        "final_fit": _fit_metadata(),
    }


def _validate_metadata(
    metadata: dict,
    *,
    arm: str,
    applicable: bool,
) -> None:
    runner.validate_arm_metadata(
        metadata,
        arm=arm,
        t5_size_gate_applicable=applicable,
        fit_seconds=1.0,
        train_rows=2_400 if applicable else 1_500,
        feature_count=2,
        categorical_indices=[],
    )


def _fake_raw() -> dict:
    results = []
    for task_index, (dataset_name, task_id) in enumerate(
        freeze.TASKS.items()
    ):
        applicable = task_index != 12
        train_rows = 2_400 if applicable else 1_500
        for coordinate_part in freeze.COORDINATES:
            coordinate = {"task_id": task_id, **coordinate_part}
            split = {
                **coordinate_part,
                "train_rows": train_rows,
                "test_rows": 400,
                "train_index_sha256": _hash("a"),
                "test_index_sha256": _hash("b"),
            }
            task = {
                "task_id": task_id,
                "dataset_name": dataset_name,
                "n_features": 2,
                "categorical_feature_indices": [],
            }
            for arm in runner.ARM_ORDER:
                engaged = arm != runner.CONTROL_ARM and task_index < 2
                if arm == "t5_composite_policy" and not applicable:
                    engaged = False
                ratio = (
                    0.9
                    if task_index == 0 and arm != runner.CONTROL_ARM
                    else 1.1
                    if task_index == 1 and arm != runner.CONTROL_ARM
                    else 1.0
                )
                metadata = _metadata(
                    arm,
                    engaged=engaged,
                    applicable=applicable,
                )
                prediction_hash = (
                    _hash("c")
                    if arm == runner.CONTROL_ARM or not engaged
                    else _hash("d" if arm == "t5_composite_policy" else "e")
                )
                behavior = {
                    "coordinate": coordinate,
                    "arm": arm,
                    "rmse": 10.0 * ratio,
                    "prediction_sha256": prediction_hash,
                    "metadata": metadata,
                }
                results.append(
                    {
                        "worker_key": runner.worker_key(coordinate, arm),
                        "coordinate": coordinate,
                        "arm": arm,
                        "task": task,
                        "split": split,
                        "t5_size_gate_applicable": applicable,
                        "rmse": 10.0 * ratio,
                        "fit_seconds": 1.0,
                        "wall_seconds": 1.1,
                        "prediction_timing": _prediction_timing(),
                        "prediction_sha256": prediction_hash,
                        "test_target_sha256": _hash("f"),
                        "metadata": metadata,
                        "peak_rss_bytes": 1_000_000,
                        "behavior_fingerprint_sha256": runner._json_sha256(
                            behavior
                        ),
                    }
                )
    return common.bind_artifact_sha256(
        {
            "schema_version": 1,
            "name": "darkofit_panel3_cross_power_calibration_raw_v1",
            "created_at": "2026-07-18T00:00:00+00:00",
            "source_freeze_path": (
                "benchmarks/"
                "panel3_cross_power_calibration_source_freeze.json"
            ),
            "source_freeze_file_sha256": _hash("1"),
            "source_freeze_sha256": _hash("2"),
            "runtime": {},
            "tasks": freeze.TASKS,
            "coordinates": runner.expected_coordinates(),
            "arms": list(runner.ARM_ORDER),
            "execution": {
                "kind": (
                    "coordinate_waves_three_concurrent_isolated_arm_processes"
                ),
                "concurrent_processes": 3,
                "worker_thread_count": runner.THREAD_COUNT,
                "random_state": runner.RANDOM_STATE,
                "timing_and_memory_claim_eligible": False,
            },
            "spool": {
                "directory": str(
                    runner.DEFAULT_SPOOL_DIRECTORY.relative_to(runner.ROOT)
                ),
                "binding": {},
                "record_count": 117,
                "resumed_record_count": 0,
                "records": [{} for _ in range(117)],
            },
            "results": results,
            "result_count": 117,
            "all_results_preserved_without_filtering": True,
            "outcomes_scored": True,
            "analysis_performed": False,
            "development_only": True,
            "panel3_authorized": False,
            "default_promotion_authorized": False,
            "product_claim_authorized": False,
        },
        "raw_artifact_sha256",
    )


def _rebind_raw(raw: dict) -> dict:
    raw = copy.deepcopy(raw)
    raw.pop("raw_artifact_sha256", None)
    return common.bind_artifact_sha256(raw, "raw_artifact_sha256")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rmse", True),
        ("fit_seconds", False),
        ("wall_seconds", True),
        ("prediction_sha256", "z" * 64),
        ("test_target_sha256", "z" * 64),
    ],
)
def test_calibration_result_rejects_boolean_measurements_and_nonhex_hashes(
    field,
    value,
):
    result = copy.deepcopy(_fake_raw()["results"][0])
    result[field] = value
    if field in {"rmse", "prediction_sha256"}:
        result["behavior_fingerprint_sha256"] = runner._json_sha256(
            {
                "coordinate": result["coordinate"],
                "arm": result["arm"],
                "rmse": result["rmse"],
                "prediction_sha256": result["prediction_sha256"],
                "metadata": result["metadata"],
            }
        )

    with pytest.raises(RuntimeError):
        analyzer._validate_result(
            result,
            expected_key=result["worker_key"],
            expected_coordinate=result["coordinate"],
            expected_arm=result["arm"],
        )


def test_calibration_worker_rejects_boolean_measurement():
    result = copy.deepcopy(_fake_raw()["results"][0])
    result["rmse"] = True
    result["behavior_fingerprint_sha256"] = runner._json_sha256(
        {
            "coordinate": result["coordinate"],
            "arm": result["arm"],
            "rmse": result["rmse"],
            "prediction_sha256": result["prediction_sha256"],
            "metadata": result["metadata"],
        }
    )
    source_freeze = {
        "task_view_attestations": {
            str(result["coordinate"]["task_id"]): {
                **result["task"],
                "coordinates": [result["split"]],
            }
        }
    }

    with pytest.raises(RuntimeError, match="worker RMSE"):
        runner.validate_worker_result(
            result,
            source_freeze,
            result["coordinate"],
            result["arm"],
        )


def test_spool_resume_rejects_symlink_ancestor(tmp_path):
    coordinate = {
        "task_id": next(iter(freeze.TASKS.values())),
        "repeat": 0,
        "fold": 0,
        "sample": 0,
    }
    arm = runner.CONTROL_ARM
    binding = {"campaign": "panel3-cross-power-calibration"}
    worker_key = runner.worker_key(coordinate, arm)
    result = {
        "worker_key": worker_key,
        "coordinate": coordinate,
        "arm": arm,
    }
    payload = common.bind_artifact_sha256(
        {
            "schema_version": 1,
            "name": (
                "darkofit_panel3_cross_power_calibration_spool_v1"
            ),
            "binding": binding,
            "worker_key": worker_key,
            "result_sha256": runner._json_sha256(result),
            "result": result,
        },
        "spool_record_sha256",
    )
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (outside / "record.json").write_text(
        json.dumps(payload, sort_keys=True)
    )
    (allowed / "redirect").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="invalid calibration spool"):
        runner.load_spool(
            allowed / "redirect" / "record.json",
            binding,
            coordinate,
            arm,
            allowed_root=allowed,
        )


def test_dangling_spool_leaf_is_present_and_rejected_before_reuse(
    tmp_path,
):
    coordinate = {
        "task_id": next(iter(freeze.TASKS.values())),
        "repeat": 0,
        "fold": 0,
        "sample": 0,
    }
    arm = runner.CONTROL_ARM
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing.json")

    assert not dangling.exists()
    assert dangling.is_symlink()
    with pytest.raises(RuntimeError, match="invalid calibration spool"):
        runner.load_spool(
            dangling,
            {"campaign": "synthetic"},
            coordinate,
            arm,
            allowed_root=tmp_path,
        )


def test_secure_json_snapshot_hashes_the_exact_decoded_bytes(tmp_path):
    artifact = tmp_path / "artifact.json"
    encoded = b'{"value": 1}\n'
    artifact.write_bytes(encoded)

    value, digest = common.secure_load_json(
        artifact,
        allowed_root=tmp_path,
    )

    assert value == {"value": 1}
    assert digest == hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda runtime: runtime.__setitem__("unexpected", "field"),
        lambda runtime: runtime.__setitem__("schema_version", 2),
    ],
)
def test_source_freeze_rejects_mutated_runtime_contract_schema(mutation):
    candidate = copy.deepcopy(common.load_json(freeze.CANDIDATE_CONTRACT))
    runtime = copy.deepcopy(common.load_json(freeze.RUNTIME_CONTRACT))
    mutation(runtime)
    runtime_bytes = json.dumps(
        runtime,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    runtime_sha256 = hashlib.sha256(runtime_bytes).hexdigest()
    candidate["runtime"]["sha256"] = runtime_sha256
    candidate_bytes = json.dumps(
        candidate,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    snapshots = {
        freeze.CANDIDATE_CONTRACT.absolute(): candidate_bytes,
        freeze.RUNTIME_CONTRACT.absolute(): runtime_bytes,
    }
    files = {
        str(freeze.CANDIDATE_CONTRACT.relative_to(freeze.ROOT)): (
            hashlib.sha256(candidate_bytes).hexdigest()
        ),
        str(freeze.RUNTIME_CONTRACT.relative_to(freeze.ROOT)): (
            runtime_sha256
        ),
    }

    with pytest.raises(RuntimeError, match="runtime environment changed"):
        freeze._decode_contract_snapshots(snapshots, files)


def test_source_freeze_rejects_transient_contract_snapshot_restored_to_h1(
    monkeypatch,
):
    source_paths = (
        freeze.CANDIDATE_CONTRACT,
        freeze.RUNTIME_CONTRACT,
        freeze.PROTOCOL,
        freeze.RUNNER,
        freeze.ANALYZER,
    )
    snapshots, restored_files = common.secure_snapshot_files(
        list(source_paths)
    )
    transient_runtime = copy.deepcopy(
        common.load_json(freeze.RUNTIME_CONTRACT)
    )
    transient_runtime["unexpected"] = "transient"
    runtime_bytes = json.dumps(
        transient_runtime,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    runtime_sha256 = hashlib.sha256(runtime_bytes).hexdigest()
    transient_candidate = copy.deepcopy(
        common.load_json(freeze.CANDIDATE_CONTRACT)
    )
    transient_candidate["runtime"]["sha256"] = runtime_sha256
    candidate_bytes = json.dumps(
        transient_candidate,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    transient_snapshots = dict(snapshots)
    transient_snapshots[freeze.RUNTIME_CONTRACT.absolute()] = runtime_bytes
    transient_snapshots[freeze.CANDIDATE_CONTRACT.absolute()] = candidate_bytes
    transient_files = dict(restored_files)
    transient_files[
        str(freeze.RUNTIME_CONTRACT.relative_to(freeze.ROOT))
    ] = runtime_sha256
    transient_files[
        str(freeze.CANDIDATE_CONTRACT.relative_to(freeze.ROOT))
    ] = hashlib.sha256(candidate_bytes).hexdigest()
    real_secure_snapshot_files = common.secure_snapshot_files

    def snapshot_files(paths, **kwargs):
        if tuple(Path(path).absolute() for path in paths) == tuple(
            path.absolute() for path in source_paths
        ):
            return transient_snapshots, transient_files
        return real_secure_snapshot_files(paths, **kwargs)

    real_git = freeze._git

    def clean_git(*arguments):
        if arguments[:2] == ("status", "--porcelain"):
            return ""
        return real_git(*arguments)

    monkeypatch.setattr(
        freeze,
        "source_paths",
        lambda: tuple(
            str(path.relative_to(freeze.ROOT)) for path in source_paths
        ),
    )
    monkeypatch.setattr(common, "secure_snapshot_files", snapshot_files)
    monkeypatch.setattr(
        freeze,
        "source_file_sha256_at_head",
        lambda _head, _paths=None: restored_files,
    )
    monkeypatch.setattr(freeze, "_git", clean_git)
    monkeypatch.setattr(
        freeze,
        "task_view_attestations",
        lambda: pytest.fail("target views were accessed"),
    )

    with pytest.raises(
        RuntimeError,
        match="source snapshot differs from committed H1",
    ):
        freeze.build()


def test_boundary_is_imported_from_spent_tabarena_panel():
    assert freeze.TASKS == spent.TASKS
    assert list(freeze.COORDINATES) == [
        {"repeat": repeat, "fold": fold, "sample": 0}
        for repeat, fold in spent.SCREEN_SPLITS
    ]
    assert (
        freeze.EXPECTED_NATIVE_CATEGORICAL_COLUMNS
        == spent.EXPECTED_NATIVE_CATEGORICAL_COLUMNS
    )
    assert len(runner.expected_coordinates()) == 39
    assert len(analyzer._expected_result_keys()) == 117
    assert runner.ARM_ORDER == (
        "current_default",
        "t5_composite_policy",
        "guarded_cross_features_policy",
    )


def test_source_freeze_does_not_treat_uv_lock_as_runtime_contract():
    assert "uv.lock" not in freeze.EXPLICIT_SOURCE_PATHS
    assert (
        "benchmarks/panel3_environment_contract.json"
        in freeze.EXPLICIT_SOURCE_PATHS
    )
    assert (
        "benchmarks/run_tabarena_regression_followon_screen.py"
        in freeze.EXPLICIT_SOURCE_PATHS
    )
    assert {
        "tests/conftest.py",
        "tests/test_campaign_partition.py",
        "tests/test_panel3_cross_power_calibration.py",
        "tests/test_panel3_execution.py",
        "tests/test_panel3_power_design.py",
        "tests/test_panel3_registry.py",
    } <= set(freeze.EXPLICIT_SOURCE_PATHS)


def test_h2_history_requires_one_create_only_freeze_commit(monkeypatch):
    source_head = "1" * 40
    execution_head = "2" * 40
    freeze_path = freeze.FREEZE_RELATIVE

    def clean_git(*arguments):
        if arguments[:2] == ("diff", "--name-status"):
            return f"A\t{freeze_path}"
        if arguments[0] == "rev-list":
            return "3" * 40
        if arguments[0] == "diff-tree":
            return freeze_path
        raise AssertionError(arguments)

    monkeypatch.setattr(runner, "_git", clean_git)
    runner._validate_post_freeze_history(source_head, execution_head)

    def modified_existing_git(*arguments):
        if arguments[:2] == ("diff", "--name-status"):
            return f"M\t{freeze_path}"
        return clean_git(*arguments)

    monkeypatch.setattr(runner, "_git", modified_existing_git)
    with pytest.raises(RuntimeError, match="create-only source freeze"):
        runner._validate_post_freeze_history(source_head, execution_head)


def test_h2_history_rejects_intermediate_source_change_and_revert(
    monkeypatch,
):
    source_head = "1" * 40
    execution_head = "2" * 40
    freeze_commit = "3" * 40
    source_commit = "4" * 40
    freeze_path = freeze.FREEZE_RELATIVE

    def tampered_git(*arguments):
        if arguments[:2] == ("diff", "--name-status"):
            return f"A\t{freeze_path}"
        if arguments[0] == "rev-list":
            return f"{source_commit}\n{freeze_commit}"
        if arguments[0] == "diff-tree":
            commit = arguments[-1]
            return (
                freeze_path
                if commit == freeze_commit
                else "darkofit/booster.py"
            )
        raise AssertionError(arguments)

    monkeypatch.setattr(runner, "_git", tampered_git)
    with pytest.raises(RuntimeError, match="create-only source freeze"):
        runner._validate_post_freeze_history(source_head, execution_head)


def test_ordered_task_view_hash_rejects_joint_row_reordering():
    X = pd.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]})
    y = pd.Series([3.0, 4.0])
    original = freeze.ordered_task_view_sha256(X, y)
    reordered = freeze.ordered_task_view_sha256(
        X.iloc[::-1].reset_index(drop=True),
        y.iloc[::-1].reset_index(drop=True),
    )
    assert original != reordered


def test_analyzer_preserves_complete_exact_policy_census():
    raw = _fake_raw()
    summary = analyzer.analyze(
        raw,
        verify_source=False,
        verify_spool=False,
    )
    assert summary["raw_path"] == (
        "benchmarks/panel3_cross_power_calibration_raw.json"
    )
    assert not Path(summary["raw_path"]).is_absolute()
    for candidate in runner.CANDIDATE_ARMS:
        result = summary["candidate_results"][candidate]
        assert result["coordinate_count"] == 39
        assert result["dataset_count"] == 13
        assert result["coordinate_wins_losses_ties"] == {
            "win": 3,
            "loss": 3,
            "tie": 33,
        }
        assert result["dataset_wins_losses_ties"] == {
            "win": 1,
            "loss": 1,
            "tie": 11,
        }
        assert result["t5_size_gate_applicable_coordinates"] == 36
        assert result["engaged_coordinates"] == 6
        assert len(result["coordinates"]) == 39
        assert len(result["datasets"]) == 13
        assert len(summary["fixed_panel_power_inputs"][candidate]) == 13
        assert math.isclose(
            result["equal_dataset_geomean_ratio"],
            (0.9 * 1.1) ** (1.0 / 13.0),
        )


def test_analyzer_binds_supplied_raw_snapshot_without_reopening(
    monkeypatch,
):
    raw = _fake_raw()
    monkeypatch.setattr(
        analyzer,
        "_sha256",
        lambda _path: pytest.fail("raw path was reopened"),
    )

    summary = analyzer.analyze(
        raw,
        raw_path=Path("/unavailable/raw.json"),
        raw_file_sha256=_hash("9"),
        verify_source=False,
        verify_spool=False,
    )

    assert summary["raw_file_sha256"] == _hash("9")
    assert summary["complete_unfiltered_coordinate_census"] is True
    assert summary["ties_and_losses_preserved"] is True
    assert summary["panel3_authorized"] is False
    assert summary["default_promotion_authorized"] is False
    assert summary["product_claim_authorized"] is False


def test_analyzer_rejects_nonexact_t5_decline():
    raw = _fake_raw()
    result = next(
        row
        for row in raw["results"]
        if row["arm"] == "t5_composite_policy"
        and not row["metadata"]["engaged"]
    )
    result["prediction_sha256"] = _hash("9")
    behavior = {
        "coordinate": result["coordinate"],
        "arm": result["arm"],
        "rmse": result["rmse"],
        "prediction_sha256": result["prediction_sha256"],
        "metadata": result["metadata"],
    }
    result["behavior_fingerprint_sha256"] = runner._json_sha256(behavior)
    with pytest.raises(RuntimeError, match="decline is not exact"):
        analyzer.analyze(
            _rebind_raw(raw),
            verify_source=False,
            verify_spool=False,
        )


def test_analyzer_rejects_missing_result_instead_of_filtering():
    raw = _fake_raw()
    raw["results"].pop()
    raw["result_count"] -= 1
    with pytest.raises(RuntimeError, match="raw contract"):
        analyzer.analyze(
            _rebind_raw(raw),
            verify_source=False,
            verify_spool=False,
        )


@pytest.mark.parametrize(
    ("arm", "field", "message"),
    [
        (
            "t5_composite_policy",
            "relative_challenger_validation_ratio",
            "producer fields",
        ),
        (
            "guarded_cross_features_policy",
            "final_refit_parameters",
            "metadata is incomplete",
        ),
    ],
)
def test_analyzer_rejects_incomplete_policy_metadata(arm, field, message):
    raw = _fake_raw()
    result = next(
        row
        for row in raw["results"]
        if row["arm"] == arm and row["metadata"]["engaged"]
    )
    result["metadata"].pop(field)
    behavior = {
        "coordinate": result["coordinate"],
        "arm": result["arm"],
        "rmse": result["rmse"],
        "prediction_sha256": result["prediction_sha256"],
        "metadata": result["metadata"],
    }
    result["behavior_fingerprint_sha256"] = runner._json_sha256(behavior)
    with pytest.raises(RuntimeError, match=message):
        analyzer.analyze(
            _rebind_raw(raw),
            verify_source=False,
            verify_spool=False,
        )


@pytest.mark.parametrize(
    ("arm", "engaged", "applicable"),
    [
        (runner.CONTROL_ARM, False, True),
        ("t5_composite_policy", True, True),
        ("guarded_cross_features_policy", True, True),
    ],
)
def test_calibration_accepts_canonical_arm_metadata(
    arm,
    engaged,
    applicable,
):
    _validate_metadata(
        _metadata(arm, engaged=engaged, applicable=applicable),
        arm=arm,
        applicable=applicable,
    )


@pytest.mark.parametrize("applicable", [False, True])
def test_calibration_accepts_matching_t5_size_gate_applicability(
    applicable,
):
    _validate_metadata(
        _metadata(
            "t5_composite_policy",
            engaged=applicable,
            applicable=applicable,
        ),
        arm="t5_composite_policy",
        applicable=applicable,
    )


@pytest.mark.parametrize(
    ("metadata_applicable", "reported_applicable"),
    [(False, True), (True, False)],
)
def test_calibration_rejects_mismatched_t5_size_gate_applicability(
    metadata_applicable,
    reported_applicable,
):
    metadata = _metadata(
        "t5_composite_policy",
        engaged=metadata_applicable,
        applicable=metadata_applicable,
    )

    with pytest.raises(RuntimeError, match="size-gate applicability"):
        _validate_metadata(
            metadata,
            arm="t5_composite_policy",
            applicable=reported_applicable,
        )


def test_calibration_preserves_arm_error_order():
    with pytest.raises(RuntimeError, match="metadata arm changed"):
        _validate_metadata(
            _metadata(runner.CONTROL_ARM, engaged=False, applicable=True),
            arm="guarded_cross_features_policy",
            applicable=True,
        )

    with pytest.raises(RuntimeError, match="metadata arm is unknown"):
        _validate_metadata(
            {"kind": "unknown"},
            arm="unknown",
            applicable=True,
        )


def test_calibration_rejects_t5_aggregate_scores_detached_from_children():
    raw = _fake_raw()
    result = next(
        row
        for row in raw["results"]
        if row["arm"] == "t5_composite_policy"
        and row["t5_size_gate_applicable"]
    )
    metadata = result["metadata"]
    metadata["control_validation_rmse"] = 20.0
    metadata["challenger_validation_rmse"] = 18.0
    metadata["relative_challenger_validation_ratio"] = 0.9
    result["behavior_fingerprint_sha256"] = runner._json_sha256(
        {
            "coordinate": result["coordinate"],
            "arm": result["arm"],
            "rmse": result["rmse"],
            "prediction_sha256": result["prediction_sha256"],
            "metadata": metadata,
        }
    )

    with pytest.raises(RuntimeError, match="outer guard"):
        analyzer.analyze(
            _rebind_raw(raw),
            verify_source=False,
            verify_spool=False,
        )


def test_calibration_rejects_guarded_uncrossed_score_detached_from_children():
    raw = _fake_raw()
    result = next(
        row
        for row in raw["results"]
        if row["arm"] == "guarded_cross_features_policy"
        and row["metadata"]["engaged"]
    )
    metadata = result["metadata"]
    metadata["uncrossed_validation_rmse"] = 20.0
    metadata["crossed_validation_rmse"] = 18.0
    metadata["relative_crossed_validation_ratio"] = 0.9
    result["behavior_fingerprint_sha256"] = runner._json_sha256(
        {
            "coordinate": result["coordinate"],
            "arm": result["arm"],
            "rmse": result["rmse"],
            "prediction_sha256": result["prediction_sha256"],
            "metadata": metadata,
        }
    )

    with pytest.raises(RuntimeError, match="uncrossed score"):
        analyzer.analyze(
            _rebind_raw(raw),
            verify_source=False,
            verify_spool=False,
        )


def test_calibration_rejects_linear_lane_when_constant_child_wins():
    raw = _fake_raw()
    result = next(
        row
        for row in raw["results"]
        if row["arm"] == "guarded_cross_features_policy"
        and not row["metadata"]["engaged"]
    )
    metadata = result["metadata"]
    linear = next(
        record
        for record in metadata["selection_fits"]
        if record["name"] == "uncrossed_linear"
    )
    metadata["selected_linear_leaves"] = True
    metadata["selected_selection_fit"] = linear
    metadata["final_refit_parameters"]["linear_leaves"] = True
    metadata["final_fit"]["selected_lane"] = "linear_leaves"
    result["behavior_fingerprint_sha256"] = runner._json_sha256(
        {
            "coordinate": result["coordinate"],
            "arm": result["arm"],
            "rmse": result["rmse"],
            "prediction_sha256": result["prediction_sha256"],
            "metadata": metadata,
        }
    )

    with pytest.raises(RuntimeError, match="selected lane"):
        analyzer.analyze(
            _rebind_raw(raw),
            verify_source=False,
            verify_spool=False,
        )


def test_spool_verification_requires_source_verification():
    with pytest.raises(
        ValueError,
        match="spool verification requires source verification",
    ):
        analyzer.validate_raw(
            _fake_raw(),
            verify_source=False,
            verify_spool=True,
        )


def test_publication_snapshot_reopens_every_spool_and_rejects_mutation(
    monkeypatch,
):
    expected = [
        (coordinate, arm)
        for coordinate in runner.expected_coordinates()
        for arm in runner.ARM_ORDER
    ]
    results = [
        {"worker_key": runner.worker_key(coordinate, arm)}
        for coordinate, arm in expected
    ]
    records = []
    for result, (coordinate, arm) in zip(results, expected, strict=True):
        path = runner.spool_path(
            runner.DEFAULT_SPOOL_DIRECTORY,
            coordinate,
            arm,
        )
        records.append(
            {
                "worker_key": result["worker_key"],
                "path": str(path.relative_to(runner.ROOT)),
                "file_sha256": _hash("1"),
                "spool_record_sha256": _hash("2"),
                "result_sha256": runner._json_sha256(result),
                "resumed": False,
            }
        )
    calls = []

    def reopen(_path, _binding, coordinate, arm, **_kwargs):
        index = len(calls)
        calls.append(runner.worker_key(coordinate, arm))
        return (
            results[index],
            _hash("2"),
            _hash("9") if index == 116 else _hash("1"),
        )

    monkeypatch.setattr(runner, "load_spool", reopen)

    with pytest.raises(RuntimeError, match="publication spool changed"):
        runner.verify_spool_publication_snapshot(
            results,
            records,
            {},
        )
    assert len(calls) == 117


def test_analyzer_rechecks_raw_snapshot_before_create(monkeypatch):
    raw = _fake_raw()
    changed = {**raw, "created_at": "changed"}
    calls = iter(
        [
            (raw, _hash("1")),
            (changed, _hash("2")),
        ]
    )
    monkeypatch.setattr(
        analyzer.common,
        "secure_load_json",
        lambda _path: next(calls),
    )
    monkeypatch.setattr(
        analyzer,
        "analyze",
        lambda *_args, **_kwargs: {"summary_sha256": _hash("3")},
    )
    monkeypatch.setattr(
        analyzer.common,
        "validate_create_path",
        lambda _path: None,
    )
    monkeypatch.setattr(
        analyzer.common,
        "atomic_create",
        lambda *_args, **_kwargs: pytest.fail(
            "changed raw must not publish"
        ),
    )

    with pytest.raises(RuntimeError, match="changed before summary publish"):
        analyzer.main([])


def test_fit_dispatch_reuses_exact_panel3_helpers(monkeypatch):
    X = pd.DataFrame({"x": [0.0, 1.0]})
    y = pd.Series([0.0, 1.0])
    calls = []

    def fake_control(*args):
        calls.append(("control", args))
        return np.zeros(2), 1.0, {}, {
            "kind": "old",
            "engaged": False,
            "selected_configuration": "product_default",
        }

    def fake_composite(*args):
        calls.append(("composite", args))
        assert args[-1] == {}
        return np.zeros(2), 1.0, {}, {
            "kind": "old",
            "engaged": False,
            "size_gate": runner.T5_SIZE_GATE,
        }

    def fake_cross(*args):
        calls.append(("cross", args))
        return np.zeros(2), 1.0, {}, {
            "kind": "guarded_cross_features_policy",
            "engaged": False,
            "selected_crosses": False,
        }

    monkeypatch.setattr(runner.panel3.t5, "_fit_control", fake_control)
    monkeypatch.setattr(runner.panel3.t5, "_fit_composite", fake_composite)
    monkeypatch.setattr(runner.panel3, "_fit_guarded_cross", fake_cross)
    for arm in runner.ARM_ORDER:
        result = runner.fit_arm(arm, X, y, [], X)
        assert result[3]["kind"] == arm
    assert [name for name, _args in calls] == [
        "control",
        "composite",
        "cross",
    ]


def test_worker_stdout_uses_strict_panel3_json_loader(monkeypatch):
    coordinate = runner.expected_coordinates()[0]
    arm = runner.CONTROL_ARM
    payload = {"coordinate": coordinate, "arm": arm}
    called = []

    def strict_loads(value, label):
        called.append((value, label))
        return json.loads(value)

    monkeypatch.setattr(runner.panel3, "_json_loads", strict_loads)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=runner.WORKER_PREFIX + json.dumps(payload) + "\n",
            stderr="",
        ),
    )
    assert (
        runner.run_worker_subprocess(Path("freeze.json"), coordinate, arm)
        == payload
    )
    assert called and called[0][1] == "calibration worker"
