"""Focused regression tests for the isolated TabArena follow-on screen."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import benchmarks.analyze_tabarena_regression_followon_screen as analysis
import benchmarks.run_tabarena_regression_followon_screen as screen


def _fake_jobs():
    experiments = {}
    for arm, spec in screen.ARM_SPECS.items():
        model_cls = type(spec["model_cls"], (), {})
        experiments[arm] = SimpleNamespace(
            name=f"DarkoFit_c1_screen_{arm}_BAG_L1",
            method_kwargs={
                "model_cls": model_cls,
                "model_hyperparameters": {
                    **spec["config"],
                    "ag_args": {"name_suffix": f"_c1_screen_{arm}"},
                    "ag_args_ensemble": screen.expected_ag_ensemble_config(),
                },
            },
        )
    return [
        SimpleNamespace(
            experiment=experiments[arm],
            task=SimpleNamespace(dataset=dataset, repeat=repeat, fold=fold),
        )
        for dataset, repeat, fold, arm in sorted(screen.expected_grid(), reverse=True)
    ]


def test_followon_screen_grid_scopes_and_isolation_are_exact():
    assert screen.EXPECTED_CONTROL_JOBS == 39
    assert screen.EXPECTED_CANDIDATE_JOBS == 117
    assert screen.EXPECTED_JOBS == 156
    assert screen.EXPECTED_CHILD_FITS == 1_248
    assert screen.EXPECTED_PAIRED_COMPARISONS == 117
    assert {arm: len(screen.expected_arm_coordinates(arm)) for arm in screen.ARM_SPECS} == {
        "baseline": 39,
        "auto": 39,
        "ts4": 15,
        "ordinal": 6,
        "onehot": 18,
        "linear": 39,
    }
    assert set(screen.TS4_DATASETS) == {
        "airfoil_self_noise",
        "Another-Dataset-on-used-Fiat-500",
        "diamonds",
        "Food_Delivery_Time",
        "healthcare_insurance_expenses",
    }
    assert set(screen.ORDINAL_DATASETS) == {"airfoil_self_noise", "diamonds"}
    assert "miami_housing" not in screen.TS4_DATASETS
    assert "wine_quality" not in screen.TS4_DATASETS

    baseline = screen.ARM_SPECS["baseline"]
    expected_changes = {
        "auto": {"tree_mode"},
        "ts4": {"ts_permutations"},
        "linear": {"linear_residual"},
    }
    for arm, changed_names in expected_changes.items():
        config = screen.ARM_SPECS[arm]["config"]
        assert {
            name
            for name in baseline["config"]
            if config[name] != baseline["config"][name]
        } == changed_names
        assert screen.ARM_SPECS[arm]["model_cls"] == baseline["model_cls"]
        assert screen.ARM_SPECS[arm]["representation"] == "native"
    for arm, kind in (("ordinal", "safe_ordinal"), ("onehot", "safe_one_hot")):
        assert screen.ARM_SPECS[arm]["config"] == baseline["config"]
        assert screen.ARM_SPECS[arm]["model_cls"] != baseline["model_cls"]
        assert screen.ARM_SPECS[arm]["representation"] == kind


def test_followon_analyzer_rejects_unmanaged_custom_output_paths():
    with pytest.raises(SystemExit):
        analysis.parse_args(
            [
                "--input-dir",
                "/tmp/followon-screen",
                "--summary-json",
                "/tmp/followon-screen/reports/summary.json",
            ]
        )


def test_followon_analysis_publication_rolls_back_all_five_outputs(
    tmp_path, monkeypatch
):
    outputs = {
        name: tmp_path / filename
        for name, filename in zip(analysis.OUTPUT_KEYS, analysis.OUTPUT_NAMES)
    }
    payloads = {name: f"new {name}".encode() for name in analysis.OUTPUT_KEYS}
    for name, path in outputs.items():
        path.write_bytes(f"old {name}".encode())

    original_replace = analysis.hardened_analysis.os.replace
    failing_target = outputs["summary_json"]
    injected = False

    def fail_summary_install(source, destination):
        nonlocal injected
        source = Path(source)
        destination = Path(destination)
        if (
            not injected
            and destination == failing_target
            and source.name.startswith(f".{failing_target.name}.")
            and source.suffix == ".tmp"
        ):
            injected = True
            raise OSError("synthetic five-output publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(
        analysis.hardened_analysis.os, "replace", fail_summary_install
    )
    with pytest.raises(OSError, match="synthetic five-output publication failure"):
        analysis._publish_outputs_atomically(
            outputs,
            payloads,
            post_write_check=lambda: None,
        )

    assert injected is True
    for name, path in outputs.items():
        assert path.read_bytes() == f"old {name}".encode()
    assert list(tmp_path.glob(".*.tmp")) == []
    assert list(tmp_path.glob(".*.backup")) == []


def test_native_categorical_schemas_are_exact_and_prove_ts4_is_active():
    assert set(screen.EXPECTED_NATIVE_CATEGORICAL_COLUMNS) == set(screen.TASKS)
    assert all(
        screen.EXPECTED_NATIVE_CATEGORICAL_COLUMNS[dataset]
        for dataset in screen.TS4_DATASETS
    )
    assert screen.EXPECTED_NATIVE_CATEGORICAL_COLUMNS["miami_housing"] == []
    assert screen.EXPECTED_NATIVE_CATEGORICAL_COLUMNS["wine_quality"] == []
    for dataset, columns in screen.EXPECTED_NATIVE_CATEGORICAL_COLUMNS.items():
        features = list(columns) + [
            f"numeric_{index}" for index in range(max(1, 5 - len(columns)))
        ]
        metadata = {
            "schema_version": 2,
            "kind": "native",
            "fit_scope": "darkofit_child_training_fold",
            "feature_alignment_policy": "autogluon_child_drop_unique",
            "target_used_by_representation": bool(columns),
            "input_feature_count": len(features),
            "output_feature_count": len(features),
            "external_feature_schema_sha256": screen._feature_schema_sha256(
                features, "test features"
            ),
            "fitted_feature_schema_sha256": screen._feature_schema_sha256(
                features, "test fitted features"
            ),
            "categorical_input_columns": list(columns),
            "fitted_categorical_input_columns": list(columns),
            "dropped_constant_input_columns": [],
            "dropped_constant_input_unique_counts": [],
        }
        screen._validate_representation_metadata(
            metadata,
            arm="baseline",
            dataset=dataset,
            field=f"native/{dataset}",
            child_features=features,
        )

    invalid = dict(metadata)
    invalid["categorical_input_columns"] = ["invented"]
    with pytest.raises(RuntimeError, match="native feature schema"):
        screen._validate_representation_metadata(
            invalid,
            arm="baseline",
            dataset=dataset,
            field="invalid native columns",
            child_features=features,
        )
    invalid = dict(metadata)
    invalid["target_used_by_representation"] = not bool(columns)
    with pytest.raises(RuntimeError, match="native feature schema"):
        screen._validate_representation_metadata(
            invalid,
            arm="baseline",
            dataset=dataset,
            field="invalid native target flag",
            child_features=features,
        )


def test_native_schema_attests_only_exact_fold_local_constant_drops():
    features = ["signal", "rare", "noise"]
    fitted = ["signal", "noise"]
    metadata = {
        "schema_version": 2,
        "kind": "native",
        "fit_scope": "darkofit_child_training_fold",
        "feature_alignment_policy": "autogluon_child_drop_unique",
        "target_used_by_representation": False,
        "input_feature_count": 3,
        "output_feature_count": 2,
        "external_feature_schema_sha256": screen._feature_schema_sha256(
            features, "test features"
        ),
        "fitted_feature_schema_sha256": screen._feature_schema_sha256(
            fitted, "test fitted features"
        ),
        "categorical_input_columns": [],
        "fitted_categorical_input_columns": [],
        "dropped_constant_input_columns": ["rare"],
        "dropped_constant_input_unique_counts": [1],
    }
    screen._validate_representation_metadata(
        metadata,
        arm="baseline",
        dataset="QSAR-TID-11",
        field="fold-local constant",
        child_features=features,
    )
    normalized_row = {
        "arm": "baseline",
        "dataset": "QSAR-TID-11",
        "child_features": features,
        "representation": metadata,
    }
    analysis._validate_normalized_representation(normalized_row)

    normalized_row["representation"] = {
        **metadata,
        "dropped_constant_input_columns": ["invented_numeric"],
    }
    with pytest.raises(RuntimeError, match="not bound to the child"):
        analysis._validate_normalized_representation(normalized_row)

    invalid = dict(metadata, dropped_constant_input_unique_counts=[2])
    with pytest.raises(RuntimeError, match="constant-drop audit"):
        screen._validate_representation_metadata(
            invalid,
            arm="baseline",
            dataset="QSAR-TID-11",
            field="nonconstant drop",
            child_features=features,
        )
    invalid = dict(metadata, fitted_feature_schema_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="fitted schema"):
        screen._validate_representation_metadata(
            invalid,
            arm="baseline",
            dataset="QSAR-TID-11",
            field="tampered fitted schema",
            child_features=features,
        )

    categorical_features = ["attack-angle", "frequency"]
    categorical_metadata = dict(
        metadata,
        target_used_by_representation=False,
        input_feature_count=2,
        output_feature_count=1,
        external_feature_schema_sha256=screen._feature_schema_sha256(
            categorical_features, "categorical test features"
        ),
        fitted_feature_schema_sha256=screen._feature_schema_sha256(
            ["frequency"], "categorical fitted features"
        ),
        categorical_input_columns=["attack-angle"],
        fitted_categorical_input_columns=[],
        dropped_constant_input_columns=["attack-angle"],
        dropped_constant_input_unique_counts=[1],
    )
    with pytest.raises(RuntimeError, match="TS4 categorical schema"):
        screen._validate_representation_metadata(
            categorical_metadata,
            arm="ts4",
            dataset="airfoil_self_noise",
            field="removed TS4 categorical",
            child_features=categorical_features,
        )


def test_native_preprocessing_audit_must_match_the_paired_control(monkeypatch):
    monkeypatch.setattr(screen, "EXPECTED_NATIVE_REPRESENTATION_PAIRS", 1)
    representation = {"kind": "native", "schema_version": 2}
    baseline = {
        "dataset": "QSAR-TID-11",
        "repeat": 0,
        "fold": 0,
        "arm": "baseline",
        "child": "S1F7",
        "representation": representation,
    }
    candidate = {
        **baseline,
        "arm": "auto",
        "representation": dict(representation),
    }
    assert screen.validate_native_representation_pairs(
        [baseline, candidate]
    ) == 1

    candidate["representation"] = {
        **representation,
        "dropped_constant_input_columns": ["invented"],
    }
    with pytest.raises(RuntimeError, match="identical child preprocessing"):
        screen.validate_native_representation_pairs([baseline, candidate])


def _valid_followon_warmup_history(thread_count=18):
    stages = []
    for spec in screen.WARMUP_STAGE_SPECS:
        input_kind = spec["input_kind"]
        mode = spec["tree_mode"]
        router = mode == "catboost" and thread_count > 1
        encoding = "kfold" if mode in {"lightgbm", "hybrid"} else "ordered"
        width = 12 if input_kind == "numeric" else 13
        stages.append(
            {
                "name": spec["name"],
                "input_kind": input_kind,
                "categorical_features": [] if input_kind == "numeric" else [12],
                "config": {
                    **screen.WARMUP_BASE_CONFIG,
                    "tree_mode": mode,
                    "linear_residual": spec["linear_residual"],
                    "ts_permutations": spec["ts_permutations"],
                    "thread_count": thread_count,
                },
                "train_rows": 2_048,
                "validation_rows": 512,
                "fit_seconds": 0.1,
                "iterations_fitted": 5,
                "tree_depths": [6] * 5,
                "requested_tree_mode": mode,
                "resolved_tree_mode": mode,
                "selected_lane": (
                    "linear_residual" if spec["linear_residual"] else "boosting"
                ),
                "linear_residual_active": spec["linear_residual"],
                "resolved_learning_rate": 0.1,
                "resolved_ordered_boosting": False,
                "resolved_thread_count": (
                    min(thread_count, 2)
                    if mode in {"lightgbm", "hybrid"}
                    else thread_count
                ),
                "resolved_target_encoding_mode": encoding,
                "resolved_include_cat_codes": mode in {"lightgbm", "hybrid"},
                "resolved_ts_permutations": spec["ts_permutations"],
                "encoder_modes": [] if input_kind == "numeric" else [encoding],
                "encoder_ts_permutations": (
                    [] if input_kind == "numeric" else [spec["ts_permutations"]]
                ),
                "flat_ensemble_type": (
                    "FlatObliviousEnsemble"
                    if mode == "catboost"
                    else "FlatNonObliviousEnsemble"
                ),
                "flat_prediction_router_selected": router,
                "prediction_parallel_min_rows": 8_192,
                "prediction_batches": [
                    {
                        "name": "serial_subthreshold",
                        "route": "flat_serial" if router else "tree_loop",
                        "input_shape": [8_191, width],
                        "prediction_shape": [8_191],
                        "predict_seconds": 0.01,
                        "prediction_sha256": "0" * 64,
                    },
                    {
                        "name": "parallel_at_threshold",
                        "route": "flat_parallel" if router else "tree_loop",
                        "input_shape": [8_192, width],
                        "prediction_shape": [8_192],
                        "predict_seconds": 0.01,
                        "prediction_sha256": "1" * 64,
                    },
                ],
            }
        )
    return [
        {
            "completed_at_utc": "2026-07-13T00:00:00+00:00",
            "pid": 123,
            "warmup": {
                "schema_version": screen.WARMUP_SCHEMA_VERSION,
                "kind": screen.WARMUP_KIND,
                "clock": "time.monotonic_ns",
                "duration_seconds": 1.0,
                "thread_count": thread_count,
                "stage_count": len(stages),
                "counts": deepcopy(screen.EXPECTED_WARMUP_COUNTS),
                "stages": stages,
            },
        }
    ]


def test_followon_warmup_and_inherited_provenance_are_source_frozen():
    source_files = {str(path) for path in screen.SOURCE_FILES}
    assert {
        "benchmarks/tabarena_followon_warmup.py",
        "benchmarks/run_tabarena_regression_cap_horizon.py",
        "benchmarks/analyze_tabarena_regression_cap_horizon.py",
    } <= source_files
    protocol = screen.frozen_protocol()
    assert protocol["artifact_schema_versions"] == {
        "run_manifest": screen.RUN_MANIFEST_SCHEMA_VERSION,
        "completion_attestation": screen.COMPLETION_ATTESTATION_SCHEMA_VERSION,
        "analysis_payload": screen.ANALYSIS_PAYLOAD_SCHEMA_VERSION,
    }
    assert screen.ANALYSIS_PAYLOAD_SCHEMA_VERSION == 2
    legacy_protocol = deepcopy(protocol)
    legacy_protocol.pop("artifact_schema_versions")
    legacy_digest = screen.hashlib.sha256(
        screen.hardened._canonical_json(legacy_protocol)
    ).hexdigest()
    assert screen.protocol_sha256() != legacy_digest
    assert protocol["warmup"] == {
        "kind": screen.WARMUP_KIND,
        "schema_version": screen.WARMUP_SCHEMA_VERSION,
        "stage_names": [spec["name"] for spec in screen.WARMUP_STAGE_SPECS],
        "stage_count": 9,
        "expected_counts": screen.EXPECTED_WARMUP_COUNTS,
        "thread_policy": "same pinned CPU count as every measured child",
    }
    history = _valid_followon_warmup_history()
    screen._validate_followon_warmup_history(
        history, expected_thread_count=18, expected_latest_pid=123
    )
    with pytest.raises(RuntimeError, match="not produced by this run"):
        screen._validate_followon_warmup_history(
            history, expected_thread_count=18, expected_latest_pid=124
        )

    invalid = deepcopy(history)
    invalid[0]["warmup"]["stages"][2]["resolved_tree_mode"] = "catboost"
    with pytest.raises(RuntimeError, match="stage mismatch"):
        screen._validate_followon_warmup_history(
            invalid, expected_thread_count=18
        )

    invalid = deepcopy(history)
    invalid[0]["warmup"]["counts"]["selected_lane"] = {"boosting": 9}
    with pytest.raises(RuntimeError, match="counts or resources"):
        screen._validate_followon_warmup_history(
            invalid, expected_thread_count=18
        )


def test_followon_analyzer_accepts_current_payload_schema():
    payload = {"schema_version": screen.ANALYSIS_PAYLOAD_SCHEMA_VERSION}
    assert analysis._require_schema_version(
        payload,
        expected=screen.ANALYSIS_PAYLOAD_SCHEMA_VERSION,
        field="safe analysis payload",
    ) == 2


@pytest.mark.parametrize("schema_version", (1, 3))
def test_followon_analyzer_rejects_old_or_tampered_payload_schema(schema_version):
    with pytest.raises(RuntimeError, match="requires schema version 2"):
        analysis._require_schema_version(
            {"schema_version": schema_version},
            expected=screen.ANALYSIS_PAYLOAD_SCHEMA_VERSION,
            field="safe analysis payload",
        )


@pytest.mark.parametrize("value", [True, False, 1.5, "2", None])
def test_followon_warmup_rejects_nonintegral_thread_counts(value):
    from benchmarks.tabarena_followon_warmup import (
        warmup_tabarena_followon_screen,
    )

    with pytest.raises(TypeError, match="positive integer"):
        warmup_tabarena_followon_screen(thread_count=value)


def test_followon_screen_order_balances_every_candidate_against_shared_control():
    ordered = screen.order_screen_jobs(_fake_jobs())
    assert len(ordered) == screen.EXPECTED_JOBS
    assert screen.ordering_balance(ordered) == {
        "auto": {"candidate_before": 20, "candidate_after": 19},
        "ts4": {"candidate_before": 8, "candidate_after": 7},
        "ordinal": {"candidate_before": 3, "candidate_after": 3},
        "onehot": {"candidate_before": 9, "candidate_after": 9},
        "linear": {"candidate_before": 20, "candidate_after": 19},
    }
    positions = {}
    for index, job in enumerate(ordered):
        positions[(screen._job_coordinate(job), screen._job_arm(job))] = index
    for coordinate in screen.expected_coordinates():
        assert (coordinate, "baseline") in positions
    relative = {
        "auto_before_linear": sum(
            positions[(coordinate, "auto")] < positions[(coordinate, "linear")]
            for coordinate in screen.expected_coordinates()
        ),
        "linear_before_auto": sum(
            positions[(coordinate, "linear")] < positions[(coordinate, "auto")]
            for coordinate in screen.expected_coordinates()
        ),
    }
    assert relative == {"auto_before_linear": 20, "linear_before_auto": 19}


def test_result_discovery_rejects_a_symlink_alias_before_path_resolution(
    tmp_path, monkeypatch
):
    expected = tmp_path / "experiments" / "expected" / "results.pkl"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"trusted runner payload")
    alias = tmp_path / "experiments" / "alias" / "results.pkl"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(expected)
    monkeypatch.setattr(screen, "_result_path", lambda output_dir, job: expected)
    monkeypatch.setattr(screen, "EXPECTED_JOBS", 1)
    with pytest.raises(RuntimeError, match="not a regular file"):
        screen.collect_result_artifacts(tmp_path, [object()])


def test_expected_result_path_rejects_symlinked_ancestor(tmp_path, monkeypatch):
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    (real_parent / "results.pkl").write_bytes(b"runner payload")
    linked_parent = tmp_path / "experiments" / "linked_parent"
    linked_parent.parent.mkdir()
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    expected = linked_parent / "results.pkl"
    monkeypatch.setattr(screen, "_result_path", lambda output_dir, job: expected)
    monkeypatch.setattr(screen, "EXPECTED_JOBS", 1)
    with pytest.raises(RuntimeError, match="symlinked path component"):
        screen.collect_result_artifacts(tmp_path, [object()])

    archive_link = tmp_path / "resume_invalidated" / "linked"
    archive_link.parent.mkdir()
    archive_link.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlinked path component"):
        screen._prepare_archive_destination(
            archive_link / "artifact.json", output_dir=tmp_path
        )


def test_result_source_is_bound_to_its_exact_coordinate_path():
    expected = screen.expected_result_relative_path(
        "airfoil_self_noise", 0, 0, "baseline"
    )
    screen._validate_result_source_binding(
        expected,
        dataset="airfoil_self_noise",
        repeat=0,
        fold=0,
        arm="baseline",
    )
    with pytest.raises(RuntimeError, match="wrong frozen path"):
        screen._validate_result_source_binding(
            screen.expected_result_relative_path(
                "diamonds", 0, 0, "baseline"
            ),
            dataset="airfoil_self_noise",
            repeat=0,
            fold=0,
            arm="baseline",
        )


def test_completion_rehashes_results_after_normalization(tmp_path, monkeypatch):
    artifacts = [
        {"experiments/data/result/results.pkl": {"sha256": "0" * 64, "size_bytes": 1}},
        {"experiments/data/result/results.pkl": {"sha256": "1" * 64, "size_bytes": 1}},
    ]
    monkeypatch.setattr(screen, "EXPECTED_JOBS", 1)
    monkeypatch.setattr(
        screen, "collect_result_artifacts", lambda output_dir, jobs: artifacts.pop(0)
    )
    monkeypatch.setattr(
        screen,
        "validate_completed_results",
        lambda output_dir, value: (
            {
                "resource_allocation": {"num_cpus_child": 2},
                "result_count": 1,
                "child_fit_count": 8,
                "paired_comparison_count": 1,
            },
            [],
            [],
        ),
    )
    monkeypatch.setattr(screen.hardened, "_atomic_write_json", lambda *args: None)
    monkeypatch.setattr(
        screen,
        "_stable_file_artifact",
        lambda *args: {"path": "payload", "sha256": "2" * 64, "size_bytes": 1},
    )
    monkeypatch.setattr(screen, "_history_artifact", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="changed during normalization"):
        screen.write_completion_attestation(
            tmp_path,
            manifest={
                "resolved_child_num_cpus": 2,
                "protocol_sha256": "3" * 64,
                "source": {"git_head": "head"},
            },
            jobs=[object()],
            result_count=1,
        )


def test_resume_history_rejects_undeclared_coordinates_and_arm_status(tmp_path):
    base = {
        "resumed_at_utc": "2026-07-13T00:00:00+00:00",
        "pid": 123,
        "invalidated_coordinate_count": 0,
        "invalidated_result_count": 0,
        "invalidated_coordinates": [],
        "archived_campaign_artifacts": [],
    }
    screen._validate_resume_history([base], tmp_path)
    invalid = deepcopy(base)
    invalid["invalidated_coordinate_count"] = 1
    invalid["invalidated_coordinates"] = [
        {
            "dataset": "airfoil_self_noise",
            "repeat": 9,
            "fold": 9,
            "arm_status": {"baseline": "missing"},
            "archived": [],
        }
    ]
    with pytest.raises(RuntimeError, match="not in the screen"):
        screen._validate_resume_history([invalid], tmp_path)


def _representation_model(model_cls):
    pytest.importorskip("autogluon.core")
    model = model_cls(
        path="",
        name="representation_test",
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        hyperparameters={},
    )
    model._screen_representation_fit_active = True
    model._screen_representation_fit_calls = 0
    model._screen_representation_eval_transform_calls = 0
    model._screen_representation_eval_unknown_counts = []
    return model


def _numeric_frame(pd, columns, rows=12):
    return pd.DataFrame(
        {
            column: np.linspace(index, index + 1, rows)
            for index, column in enumerate(columns)
        }
    )


def test_native_adapter_records_autogluon_fold_constant_alignment(monkeypatch):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("autogluon.core")
    from benchmarks import tabarena_screen_adapters as adapters

    outer = pd.DataFrame(
        {
            "signal": np.arange(9, dtype=np.float64),
            "rare": [0, 1, 1, 1, 1, 1, 1, 1, 1],
            "noise": np.arange(9, 0, -1, dtype=np.float64),
        }
    )
    child_train = outer.iloc[1:].reset_index(drop=True)
    y = pd.Series(np.linspace(0.0, 1.0, len(child_train)))
    model = adapters.ScreenNativeDarkoFitModel(
        path="",
        name="native_constant_alignment",
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        hyperparameters={},
    )
    model.initialize(X=child_train, y=y)
    assert model._features_internal == ["signal", "noise"]

    def fake_parent_fit(self, X, y, **kwargs):
        del y, kwargs
        transformed = self.preprocess(X, is_train=True)
        self.model = SimpleNamespace(
            n_features_in_=int(transformed.shape[1]),
            feature_names_in_=np.asarray(transformed.columns, dtype=object),
        )

    monkeypatch.setattr(adapters.DarkoFitModel, "_fit", fake_parent_fit)
    model._fit(child_train, y)
    metadata = model._fit_metadata[adapters.REPRESENTATION_METADATA_KEY]
    assert metadata["input_feature_count"] == 3
    assert metadata["output_feature_count"] == 2
    assert metadata["dropped_constant_input_columns"] == ["rare"]
    assert metadata["dropped_constant_input_unique_counts"] == [1]
    screen._validate_representation_metadata(
        metadata,
        arm="baseline",
        dataset="QSAR-TID-11",
        field="native adapter constant alignment",
        child_features=list(child_train.columns),
    )


def test_source_frozen_ordinal_mappings_and_schema_fail_closed():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("autogluon.core")
    from benchmarks.tabarena_screen_adapters import (
        AIRFOIL_CHILD_CODE_VALUES,
        AIRFOIL_COLUMNS,
        DIAMONDS_CHILD_CODE_RANKS,
        DIAMONDS_COLUMNS,
        DIAMONDS_ORDERS,
        MIAMI_COLUMNS,
        SafeOrdinalDarkoFitModel,
    )

    airfoil = _numeric_frame(pd, AIRFOIL_COLUMNS, rows=3)
    airfoil["attack-angle"] = pd.Series(
        pd.Categorical([1, 13, 26], categories=range(27))
    )
    model = _representation_model(SafeOrdinalDarkoFitModel)
    transformed = model._preprocess(airfoil, is_train=True)
    np.testing.assert_array_equal(transformed["attack-angle"], [1.5, 3.0, 9.9])
    assert AIRFOIL_CHILD_CODE_VALUES[13] == 3.0
    assert model._representation_metadata()["domain"] == "airfoil_attack_angle_numeric"

    diamonds = _numeric_frame(pd, DIAMONDS_COLUMNS, rows=2)
    diamonds["cut"] = pd.Series(pd.Categorical([0, 2], categories=range(5)))
    diamonds["color"] = pd.Series(pd.Categorical([6, 0], categories=range(7)))
    diamonds["clarity"] = pd.Series(pd.Categorical([0, 1], categories=range(8)))
    model = _representation_model(SafeOrdinalDarkoFitModel)
    transformed = model._preprocess(diamonds, is_train=True)
    np.testing.assert_array_equal(transformed["cut"], [0.0, 4.0])
    np.testing.assert_array_equal(transformed["color"], [0.0, 6.0])
    np.testing.assert_array_equal(transformed["clarity"], [0.0, 7.0])
    assert DIAMONDS_ORDERS["cut"] == (
        "Fair",
        "Good",
        "Very Good",
        "Premium",
        "Ideal",
    )
    assert DIAMONDS_CHILD_CODE_RANKS["cut"] == (0, 1, 4, 3, 2)

    invalid = diamonds.copy()
    invalid["cut"] = pd.Series(pd.Categorical([0, 2], categories=[0, 1, 2, 3, 9]))
    model = _representation_model(SafeOrdinalDarkoFitModel)
    with pytest.raises(RuntimeError, match="compact category domain changed"):
        model._preprocess(invalid, is_train=True)

    wrong_order = airfoil[list(reversed(AIRFOIL_COLUMNS))]
    model = _representation_model(SafeOrdinalDarkoFitModel)
    with pytest.raises(RuntimeError, match="predeclared schema"):
        model._preprocess(wrong_order, is_train=True)

    invalid_airfoil = airfoil.copy()
    invalid_airfoil["attack-angle"] = pd.Series(
        pd.Categorical([1, 13, 26], categories=range(26))
    )
    model = _representation_model(SafeOrdinalDarkoFitModel)
    with pytest.raises(RuntimeError, match="compact category domain changed"):
        model._preprocess(invalid_airfoil, is_train=True)

    miami = _numeric_frame(pd, MIAMI_COLUMNS, rows=2)
    miami["avno60plus"] = [0, 1]
    model = _representation_model(SafeOrdinalDarkoFitModel)
    transformed = model._preprocess(miami, is_train=True)
    np.testing.assert_array_equal(transformed["avno60plus"], [0.0, 1.0])
    assert model._representation_metadata()["domain"] == "miami_avno60plus_binary"


def test_onehot_encodes_at_most_eight_and_keeps_food_id_native():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("autogluon.core")
    from benchmarks.tabarena_screen_adapters import FOOD_COLUMNS, SafeOneHotDarkoFitModel

    food = _numeric_frame(pd, FOOD_COLUMNS, rows=12)
    food["Delivery_person_ID"] = pd.Series(
        [f"ID{index}" for index in range(12)], dtype="category"
    )
    food["Type_of_order"] = pd.Series(
        (["Meal", "Snack", "Drinks", "Buffet"] * 3), dtype="category"
    )
    food["Type_of_vehicle"] = pd.Series(
        (["bike", "scooter", "car"] * 4), dtype="category"
    )
    model = _representation_model(SafeOneHotDarkoFitModel)
    transformed = model._preprocess(food, is_train=True)
    metadata = model._representation_metadata()
    assert metadata["domain"] == "food_delivery"
    assert metadata["target_free_one_hot_input_positions"] == [7, 8]
    assert metadata["target_free_one_hot_category_counts"] == [4, 3]
    assert metadata["remaining_native_target_stat_input_positions"] == [6]
    assert metadata["remaining_native_target_stat_output_positions"] == [6]
    assert model._categorical_indices == [6]
    assert str(transformed.iloc[:, 6].dtype) == "category"

    evaluation = food.iloc[:2].copy()
    evaluation["Type_of_order"] = pd.Series(
        ["Dessert", "Meal"], dtype="category", index=evaluation.index
    )
    transformed_eval = model._preprocess(evaluation, is_train=False)
    order_columns = [
        column for column in transformed_eval if "_ohe_7_" in str(column)
    ]
    assert transformed_eval.loc[evaluation.index[0], order_columns].sum() == 0
    assert model._screen_representation_eval_unknown_counts == [1]
    audit = model._representation_metadata()
    audit.update(
        {
            "fit_scope": "child_training_rows_only",
            "target_used_by_representation": False,
            "fit_calls": model._screen_representation_fit_calls,
            "eval_transform_calls_during_fit": (
                model._screen_representation_eval_transform_calls
            ),
            "eval_unknown_counts": list(
                model._screen_representation_eval_unknown_counts
            ),
        }
    )
    screen._validate_representation_metadata(
        audit,
        arm="onehot",
        dataset="Food_Delivery_Time",
        field="food one-hot test",
    )


def _valid_auto_selection():
    candidates = []
    scores = (2.0, 1.0, 3.0)
    for index, mode in enumerate(("catboost", "lightgbm", "hybrid")):
        elapsed_start = float(index)
        elapsed_end = float(index + 1)
        candidates.append(
            {
                "tree_mode": mode,
                "fit_status": "fitted",
                "selected": index == 1,
                "lane": "boosting",
                "score": scores[index],
                "validation_score": scores[index],
                "iterations_requested": 1_000,
                "iterations_attempted": 100,
                "rounds_completed": 100,
                "rounds_retained": 90,
                "best_iteration": 90,
                "resolved_learning_rate": 0.1,
                "stop_reason": "early_stopping",
                "wall_clock_elapsed_seconds_start": elapsed_start,
                "wall_clock_elapsed_seconds_end": elapsed_end,
                "wall_clock_elapsed_seconds": elapsed_end,
                "deadline_hit_start": False,
                "deadline_hit_end": False,
                "deadline_hit": False,
            }
        )
    return {
        "enabled": True,
        "input": "auto",
        "candidates": candidates,
        "selected_tree_mode": "lightgbm",
        "selected_lane": "boosting",
        "selected_candidate_index": 1,
        "selected_score": 1.0,
        "candidate_count": 3,
        "fitted_candidate_count": 3,
        "skipped_deadline_candidate_count": 0,
        "candidate_fit_status_counts": {"fitted": 3, "skipped_deadline": 0},
        "wall_clock_stopper_count": 1,
        "deadline_hit": False,
    }


def _valid_auto_top_level(selection):
    selected = selection["candidates"][selection["selected_candidate_index"]]
    return {
        **{
            name: selected[name]
            for name in (
                "iterations_requested",
                "iterations_attempted",
                "rounds_completed",
                "rounds_retained",
                "best_iteration",
                "resolved_learning_rate",
                "stop_reason",
                "deadline_hit",
            )
        },
        "selected_tree_mode": selected["tree_mode"],
        "selected_lane": selected["lane"],
        "wall_clock_limit_seconds": 10.0,
        "wall_clock_safety_margin_seconds": 0.5,
        "wall_clock_effective_seconds": 9.5,
        "wall_clock_elapsed_seconds": 4.0,
    }


def test_auto_candidate_audit_rejects_any_skipped_or_deadline_hit_candidate():
    selection = _valid_auto_selection()
    screen._validate_tree_mode_selection(
        selection,
        selected_tree_mode="lightgbm",
        deadline_hit=False,
        top_level=_valid_auto_top_level(selection),
        field="auto test",
    )
    selection["candidates"][2]["fit_status"] = "skipped_deadline"
    selection["candidates"][2]["deadline_hit"] = True
    with pytest.raises(RuntimeError, match="deadline-hit or skipped"):
        screen._validate_tree_mode_selection(
            selection,
            selected_tree_mode="lightgbm",
            deadline_hit=False,
            top_level=_valid_auto_top_level(selection),
            field="auto test",
        )


def test_auto_candidate_audit_rejects_bad_counters_and_nonminimum_selection():
    selection = _valid_auto_selection()
    selection["candidates"][0]["rounds_retained"] = 91
    with pytest.raises(RuntimeError, match="round counters"):
        screen._validate_tree_mode_selection(
            selection,
            selected_tree_mode="lightgbm",
            deadline_hit=False,
            top_level=_valid_auto_top_level(selection),
            field="auto test",
        )

    selection = _valid_auto_selection()
    selection["candidates"][2].update(
        {
            "iterations_attempted": 99,
            "rounds_completed": 99,
            "stop_reason": "time_limit",
        }
    )
    with pytest.raises(RuntimeError, match="wall-clock-stopped candidate"):
        screen._validate_tree_mode_selection(
            selection,
            selected_tree_mode="lightgbm",
            deadline_hit=False,
            top_level=_valid_auto_top_level(selection),
            field="auto test",
        )


@pytest.mark.parametrize("wall_elapsed", (9.5, 9.75, 10.0))
def test_child_deadline_audit_allows_soft_overrun_within_hard_limit(wall_elapsed):
    metadata = {
        "wall_clock_limit_seconds": 10.0,
        "wall_clock_safety_margin_seconds": 0.5,
        "wall_clock_effective_seconds": 9.5,
        "wall_clock_elapsed_seconds": wall_elapsed,
        "stop_reason": "early_stopping",
        "deadline_hit": False,
    }

    assert screen._validate_child_wall_clock_audit(
        metadata, field="child test"
    ) == (10.0, 0.5, 9.5, wall_elapsed)


@pytest.mark.parametrize(
    "updates",
    (
        {"wall_clock_elapsed_seconds": 10.000001},
        {
            "wall_clock_elapsed_seconds": 9.499999,
            "deadline_hit": True,
            "stop_reason": "time_limit",
        },
        {
            "wall_clock_elapsed_seconds": 9.75,
            "deadline_hit": False,
            "stop_reason": "time_limit",
        },
        {"stop_reason": "unknown"},
    ),
)
def test_child_deadline_audit_rejects_hard_overrun_or_impossible_hit(updates):
    metadata = {
        "wall_clock_limit_seconds": 10.0,
        "wall_clock_safety_margin_seconds": 0.5,
        "wall_clock_effective_seconds": 9.5,
        "wall_clock_elapsed_seconds": 9.0,
        "stop_reason": "early_stopping",
        "deadline_hit": False,
    }
    metadata.update(updates)

    with pytest.raises(RuntimeError, match="wall-clock audit mismatch"):
        screen._validate_child_wall_clock_audit(metadata, field="child test")


@pytest.mark.parametrize(
    ("candidate_index", "updates", "error"),
    (
        (
            0,
            {"wall_clock_elapsed_seconds_start": float("nan")},
            "must be finite",
        ),
        (
            1,
            {
                "wall_clock_elapsed_seconds_start": 2.5,
                "wall_clock_elapsed_seconds_end": 2.0,
                "wall_clock_elapsed_seconds": 2.0,
            },
            "timing fields are inconsistent",
        ),
        (
            1,
            {"wall_clock_elapsed_seconds": 2.5},
            "timing fields are inconsistent",
        ),
        (
            2,
            {
                "wall_clock_elapsed_seconds_start": -1.0,
                "wall_clock_elapsed_seconds_end": 3.0,
                "wall_clock_elapsed_seconds": 3.0,
            },
            "timing must be nonnegative",
        ),
    ),
)
def test_auto_candidate_audit_rejects_malformed_timing(
    candidate_index, updates, error
):
    selection = _valid_auto_selection()
    selection["candidates"][candidate_index].update(updates)

    with pytest.raises(RuntimeError, match=error):
        screen._validate_tree_mode_selection(
            selection,
            selected_tree_mode="lightgbm",
            deadline_hit=False,
            top_level=_valid_auto_top_level(selection),
            field="auto test",
        )


def test_auto_candidate_audit_rejects_overlapping_candidate_timeline():
    selection = _valid_auto_selection()
    selection["candidates"][1]["wall_clock_elapsed_seconds_start"] = 0.5

    with pytest.raises(RuntimeError, match="candidate timings are not ordered"):
        screen._validate_tree_mode_selection(
            selection,
            selected_tree_mode="lightgbm",
            deadline_hit=False,
            top_level=_valid_auto_top_level(selection),
            field="auto test",
        )


@pytest.mark.parametrize("candidate_end", (9.5, 9.75, 10.0))
def test_auto_candidate_audit_allows_final_candidate_soft_overrun(
    candidate_end,
):
    selection = _valid_auto_selection()
    candidate = selection["candidates"][2]
    candidate["wall_clock_elapsed_seconds_end"] = candidate_end
    candidate["wall_clock_elapsed_seconds"] = candidate_end
    top_level = _valid_auto_top_level(selection)
    top_level["wall_clock_elapsed_seconds"] = candidate_end

    screen._validate_tree_mode_selection(
        selection,
        selected_tree_mode="lightgbm",
        deadline_hit=False,
        top_level=top_level,
        field="auto test",
    )


@pytest.mark.parametrize("candidate_start", (9.5, 9.500001))
def test_auto_candidate_audit_rejects_candidate_start_at_soft_deadline(
    candidate_start,
):
    selection = _valid_auto_selection()
    candidate = selection["candidates"][2]
    candidate["wall_clock_elapsed_seconds_start"] = candidate_start
    candidate["wall_clock_elapsed_seconds_end"] = candidate_start + 0.1
    candidate["wall_clock_elapsed_seconds"] = candidate_start + 0.1
    top_level = _valid_auto_top_level(selection)
    top_level["wall_clock_elapsed_seconds"] = candidate_start + 0.1

    with pytest.raises(RuntimeError, match="starts at or after"):
        screen._validate_tree_mode_selection(
            selection,
            selected_tree_mode="lightgbm",
            deadline_hit=False,
            top_level=top_level,
            field="auto test",
        )


def test_auto_candidate_audit_rejects_hard_wall_clock_overrun():
    selection = _valid_auto_selection()
    candidate = selection["candidates"][2]
    candidate["wall_clock_elapsed_seconds_end"] = 10.000001
    candidate["wall_clock_elapsed_seconds"] = 10.000001
    top_level = _valid_auto_top_level(selection)
    top_level["wall_clock_elapsed_seconds"] = 10.0

    with pytest.raises(RuntimeError, match="exceeds the hard wall-clock limit"):
        screen._validate_tree_mode_selection(
            selection,
            selected_tree_mode="lightgbm",
            deadline_hit=False,
            top_level=top_level,
            field="auto test",
        )


def test_auto_candidate_audit_rejects_child_elapsed_before_final_candidate():
    selection = _valid_auto_selection()
    top_level = _valid_auto_top_level(selection)
    top_level["wall_clock_elapsed_seconds"] = 2.5

    with pytest.raises(RuntimeError, match="precedes its final candidate"):
        screen._validate_tree_mode_selection(
            selection,
            selected_tree_mode="lightgbm",
            deadline_hit=False,
            top_level=top_level,
            field="auto test",
        )


def test_analyzer_revalidates_nonselected_auto_candidate_metadata():
    selection = _valid_auto_selection()
    row = {
        "arm": "auto",
        "deadline_hit": False,
        "tree_mode_selection": selection,
        **_valid_auto_top_level(selection),
    }
    analysis._validate_normalized_tree_mode_selection(
        row, selected_mode="lightgbm"
    )
    row["tree_mode_selection"]["candidates"][2]["validation_score"] = 0.5
    row["tree_mode_selection"]["candidates"][2]["score"] = 0.5
    with pytest.raises(RuntimeError, match="minimum-score tie rule"):
        analysis._validate_normalized_tree_mode_selection(
            row, selected_mode="lightgbm"
        )

    selection = _valid_auto_selection()
    row = {
        "arm": "auto",
        "deadline_hit": False,
        "tree_mode_selection": selection,
        **_valid_auto_top_level(selection),
    }
    row["rounds_completed"] = 99
    with pytest.raises(RuntimeError, match="disagrees with child metadata"):
        analysis._validate_normalized_tree_mode_selection(
            row, selected_mode="lightgbm"
        )

    selection = _valid_auto_selection()
    selection["candidates"][0]["score"] = 0.5
    selection["candidates"][0]["validation_score"] = 0.5
    with pytest.raises(RuntimeError, match="minimum-score tie rule"):
        screen._validate_tree_mode_selection(
            selection,
            selected_tree_mode="lightgbm",
            deadline_hit=False,
            top_level=_valid_auto_top_level(selection),
            field="auto test",
        )


def test_linear_lane_must_be_active_only_for_the_linear_arm():
    assert screen._validate_linear_lane(
        arm="linear",
        linear_active=True,
        selected_lane="linear_residual",
        field="linear test",
    ) == "linear_residual"
    with pytest.raises(RuntimeError, match="activation mismatch"):
        screen._validate_linear_lane(
            arm="linear",
            linear_active=False,
            selected_lane="boosting",
            field="linear test",
        )
    assert screen._validate_early_stopping_rounds(
        50, field="early stopping test"
    ) == 50
    with pytest.raises(RuntimeError, match="50-round patience"):
        screen._validate_early_stopping_rounds(
            49, field="early stopping test"
        )


def _refit_params(*, mode, depth, iterations=90):
    return {
        "iterations": iterations,
        "learning_rate": 0.1,
        "tree_mode": mode,
        "early_stopping": False,
        "early_stopping_rounds": None,
        "use_best_model": False,
        "refit": False,
        "depth": depth,
        "num_leaves": None,
        "l2_leaf_reg": 3.0,
        "min_child_samples": 20,
        "min_child_weight": 1.0,
        "cat_smoothing": 1.0,
    }


def test_actual_leafwise_and_auto_compressed_refit_shapes_are_validated():
    leafwise = _refit_params(mode="lightgbm", depth=-1)
    screen._validate_refit_params(
        leafwise,
        expected_iterations=90,
        selected_tree_mode="lightgbm",
        field="leafwise child",
    )
    invalid = dict(leafwise)
    invalid["num_leaves"] = 31
    with pytest.raises(RuntimeError, match="leaf-wise capacity"):
        screen._validate_refit_params(
            invalid,
            expected_iterations=90,
            selected_tree_mode="lightgbm",
            field="leafwise child",
        )

    compressed = _refit_params(mode="catboost", depth=2)
    screen._validate_auto_compressed_refit_params(
        compressed,
        child_best=[90] * 8,
        field="auto compressed",
    )


def _synthetic_analysis_rows(ratios=None):
    ratios = ratios or {}
    outer = []
    for dataset, repeat, fold, arm in sorted(screen.expected_grid()):
        ratio = 1.0 if arm == "baseline" else ratios.get((arm, dataset), 0.99)
        outer.append(
            {
                "dataset": dataset,
                "repeat": repeat,
                "fold": fold,
                "arm": arm,
                "test_rmse": ratio,
                "val_rmse": ratio,
                "train_time_s": 1.0,
                "infer_time_s": 1.0,
                "peak_memory_bytes": 1.0,
            }
        )
    child_rows = []
    for dataset, repeat, fold, arm in sorted(screen.expected_grid()):
        for child_fold in range(8):
            child_rows.append(
                {
                    "dataset": dataset,
                    "repeat": repeat,
                    "fold": fold,
                    "arm": arm,
                    "child_fold": child_fold,
                    "best_iteration": 90,
                    "rounds_completed": 100,
                    "early_stopping_rounds": 50,
                    "stop_reason": "early_stopping",
                    "selected_tree_mode": "hybrid" if arm == "auto" else "catboost",
                    "selected_lane": "linear_residual" if arm == "linear" else "boosting",
                    "linear_residual_active": arm == "linear",
                    "representation": {"kind": screen.ARM_SPECS[arm]["representation"]},
                }
            )
    return analysis.pair_outer_rows(outer), analysis.pair_child_rows(child_rows)


def test_analyzer_predeclared_gates_include_majority_dataset_wins():
    paired, children = _synthetic_analysis_rows()
    summary, repeats = analysis.analyze(paired, children, draws=100, seed=7)
    assert len(paired) == screen.EXPECTED_PAIRED_COMPARISONS
    assert len(children) == screen.EXPECTED_PAIRED_COMPARISONS * 8
    assert len(repeats) == screen.EXPECTED_PAIRED_COMPARISONS
    assert set(summary["survivors"]) == set(screen.CANDIDATE_ARMS)
    assert all(
        item["gates"]["majority_of_applicable_datasets_improve"]
        for item in summary["arms"]
    )
    linear = next(item for item in summary["arms"] if item["arm"] == "linear")
    assert linear["child_metadata"]["selected_lane_counts"] == {
        "linear_residual": 312
    }
    assert linear["child_metadata"]["linear_residual_active_rate"] == 1.0
    report = analysis.render_report(summary)
    assert "| Infer time | Peak memory |" in report
    assert "Selected tree modes: hybrid=312." in report
    assert "Stop reasons: early_stopping=312." in report
    assert "Best iteration distribution: n=312" in report
    assert "Completed-round distribution: n=312" in report

    paired, children = _synthetic_analysis_rows(
        {
            ("ordinal", "airfoil_self_noise"): 0.98,
            ("ordinal", "diamonds"): 1.001,
        }
    )
    summary, _ = analysis.analyze(paired, children, draws=100, seed=7)
    ordinal = next(item for item in summary["arms"] if item["arm"] == "ordinal")
    assert ordinal["metrics"]["test_rmse"]["ratio"] <= 0.995
    assert ordinal["gates"]["no_dataset_point_regret_above_0_5pct"] is True
    assert ordinal["dataset_wins"] == 1
    assert ordinal["majority_wins_required"] == 2
    assert ordinal["gates"]["majority_of_applicable_datasets_improve"] is False
    assert ordinal["gates"]["survives_exploratory_screen"] is False
