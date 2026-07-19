from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmarks import analyze_panel3_confirmation as analyzer
from benchmarks import panel3_data_contract as data_contract
from benchmarks import panel3_registry_common as common
from benchmarks import run_panel3_confirmation as runner


def test_feature_policy_drops_exact_columns_and_reindexes_flags():
    X = pd.DataFrame(
        {
            "group": ["a", "b"],
            "numeric": [1.0, 2.0],
            "leak": [10, 20],
        }
    )

    result, categorical, record = runner._apply_feature_policy(
        X,
        [True, False, False],
        {"kind": "drop_columns", "columns": ["leak"]},
    )

    assert list(result.columns) == ["group", "numeric"]
    assert categorical == [True, False]
    assert record["dropped_columns"] == ["leak"]
    assert record["retained_feature_count"] == 2


def test_feature_policy_fails_closed_for_drift_and_total_drop():
    X = pd.DataFrame({"a": [1.0], "b": [2.0]})

    with pytest.raises(RuntimeError, match="missing"):
        runner._apply_feature_policy(
            X,
            [False, False],
            {"kind": "drop_columns", "columns": ["c"]},
        )
    with pytest.raises(RuntimeError, match="policy"):
        data_contract.validate_feature_policy(
            {"kind": "none", "unexpected": True}
        )
    with pytest.raises(RuntimeError, match="every feature"):
        runner._apply_feature_policy(
            X,
            [False, False],
            {"kind": "drop_columns", "columns": ["a", "b"]},
        )


def test_datetime_calendar_transform_is_deterministic_and_target_free():
    X = pd.DataFrame(
        {
            "timestamp": ["2020-01-02", "2021-12-31"],
            "value": [1.0, 2.0],
            "leak": [3.0, 4.0],
        }
    )
    policy = {
        "kind": "target_free_transform_v1",
        "drop_columns": ["leak"],
        "datetime_calendar": [
            {
                "source_column": "timestamp",
                "output_prefix": "event",
                "format": "%Y-%m-%d",
                "utc": False,
                "components": [
                    "ordinal_day",
                    "year",
                    "month",
                    "dayofweek",
                ],
                "drop_source": True,
                "missing": "reject",
            }
        ],
        "lexical_counts": [],
    }

    result, categorical, record = runner._apply_feature_policy(
        X, [False, False, False], policy
    )

    assert list(result.columns) == [
        "value",
        "event_ordinal_day",
        "event_year",
        "event_month",
        "event_dayofweek",
    ]
    assert result["event_year"].tolist() == [2020.0, 2021.0]
    assert result["event_dayofweek"].tolist() == [3.0, 4.0]
    assert categorical == [False, False, False, False, False]
    assert record["generated_columns"] == [
        "event_ordinal_day",
        "event_year",
        "event_month",
        "event_dayofweek",
    ]


def test_lexical_counts_use_nfkc_and_frozen_missing_policy():
    X = pd.DataFrame(
        {
            "text": ["Ａ B", None, "one  two"],
            "category": ["x", "y", "z"],
        }
    )
    policy = {
        "kind": "target_free_transform_v1",
        "drop_columns": [],
        "datetime_calendar": [],
        "lexical_counts": [
            {
                "source_column": "text",
                "output_prefix": "text",
                "counts": ["char_count", "token_count"],
                "unicode_normalization": "NFKC",
                "missing": "empty_string",
                "drop_source": True,
            }
        ],
    }

    result, categorical, metadata = runner._apply_feature_policy(
        X, [False, True], policy
    )

    assert list(result.columns) == [
        "category",
        "text_char_count",
        "text_token_count",
    ]
    assert result["text_char_count"].tolist() == [3, 0, 8]
    assert result["text_token_count"].tolist() == [2, 0, 2]
    assert categorical == [True, False, False]
    assert metadata["output_schema"][-1] == {
        "name": "text_token_count",
        "dtype": "int64",
    }


def test_group_hash_and_greedy_fold_assignment_are_canonical():
    X = pd.DataFrame(
        {
            "title": ["  Foo  BAR ", "foo bar", "Other", "Third"],
            "body": ["Ｂaz", "Baz", "Body", "Body"],
        }
    )

    spec = {
        "kind": "length_prefixed_nfkc_casefold_sha256_v1",
        "source_columns": ["title", "body"],
        "missing": "empty_string",
        "whitespace": "collapse",
    }
    hashes = data_contract.canonical_group_hashes(X, spec)
    folds = data_contract.greedy_group_fold_ids(hashes, n_splits=3)

    assert hashes[0] == hashes[1]
    assert folds[0] == folds[1]
    assert set(folds) == {0, 1, 2}
    assert folds == data_contract.greedy_group_fold_ids(
        hashes, n_splits=3
    )

    typed = data_contract.canonical_group_hashes(
        pd.DataFrame({"value": pd.Series([1, "1", 1.0], dtype=object)}),
        {
            "kind": "typed_value_tuple_sha256_v1",
            "source_columns": ["value"],
            "missing": "reject",
            "whitespace": "preserve",
        },
    )
    assert len(set(typed)) == 3


def _coordinate(task_id=1, repeat=0, fold=0, sample=0):
    return {
        "task_id": task_id,
        "repeat": repeat,
        "fold": fold,
        "sample": sample,
    }


def _explicit_split_row():
    train = np.asarray([0, 2, 4], dtype=np.int64)
    test = np.asarray([1, 3], dtype=np.int64)
    return {
        "task_record": {"fingerprint": {"n_rows": 5}},
        "split_policy": {
            "kind": "frozen_explicit",
            "allow_unused_rows": False,
            "coordinates": [
                {
                    "repeat": 0,
                    "fold": 0,
                    "sample": 0,
                    "train_indices": train.tolist(),
                    "test_indices": test.tolist(),
                    "train_size": 3,
                    "test_size": 2,
                    "train_index_sha256": runner._array_sha256(
                        train, dtype="<i8"
                    ),
                    "test_index_sha256": runner._array_sha256(
                        test, dtype="<i8"
                    ),
                }
            ],
        },
    }


def test_explicit_split_is_consumed_without_openml_split_access():
    class NoOfficialSplit:
        def get_train_test_split_indices(self, **_kwargs):
            raise AssertionError("official split must not be consulted")

    train, test, metadata = runner._resolve_split(
        NoOfficialSplit(),
        _explicit_split_row(),
        _coordinate(),
    )

    assert train.tolist() == [0, 2, 4]
    assert test.tolist() == [1, 3]
    assert metadata["kind"] == "frozen_explicit"


def test_explicit_split_rejects_overlap_hash_drift_and_unapproved_omission():
    row = _explicit_split_row()
    row["split_policy"]["coordinates"][0]["test_indices"] = [2, 3]

    with pytest.raises(RuntimeError, match="indices are invalid"):
        runner._resolve_split(
            object(),
            row,
            _coordinate(),
        )

    row = _explicit_split_row()
    row["split_policy"]["coordinates"][0]["train_index_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="changed"):
        runner._resolve_split(
            object(),
            row,
            _coordinate(),
        )

    row = _explicit_split_row()
    coordinate = row["split_policy"]["coordinates"][0]
    coordinate["train_indices"] = [0, 2]
    coordinate["train_size"] = 2
    coordinate["train_index_sha256"] = runner._array_sha256(
        [0, 2], dtype="<i8"
    )
    with pytest.raises(RuntimeError, match="omits rows"):
        runner._resolve_split(
            object(),
            row,
            _coordinate(),
        )


def test_hierarchical_bootstrap_is_deterministic_and_uses_three_coordinates():
    ratios = {
        f"lineage-{index}": [0.98, 0.99, 1.0]
        for index in range(12)
    }

    left = analyzer.hierarchical_bootstrap_upper(
        ratios, seed=7, replicates=2_000
    )
    right = analyzer.hierarchical_bootstrap_upper(
        ratios, seed=7, replicates=2_000
    )

    assert left == right
    assert 0.98 <= left <= 1.0
    with pytest.raises(ValueError, match="three positive ratios"):
        analyzer.hierarchical_bootstrap_upper(
            {**ratios, "lineage-0": [0.99, 1.0]},
            replicates=10,
        )


def test_hierarchical_bootstrap_resamples_three_folds_per_sampled_task():
    ratios = {
        f"lineage-{index}": [0.5, 1.0, 2.0]
        for index in range(12)
    }
    seed = 19
    replicates = 400

    observed = analyzer.hierarchical_bootstrap_upper(
        ratios,
        seed=seed,
        replicates=replicates,
        percentile=90.0,
    )

    logs = np.log(np.asarray(list(ratios.values()), dtype=np.float64))
    rng = np.random.default_rng(seed)
    task_draws = rng.integers(0, 12, size=(replicates, 12))
    fold_draws = rng.integers(0, 3, size=(replicates, 12, 3))
    expected = float(
        np.exp(
            np.percentile(
                logs[task_draws[..., None], fold_draws].mean(axis=(1, 2)),
                90.0,
                method="linear",
            )
        )
    )
    one_fold_rng = np.random.default_rng(seed)
    one_fold_tasks = one_fold_rng.integers(
        0, 12, size=(replicates, 12)
    )
    one_fold_draws = one_fold_rng.integers(
        0, 3, size=(replicates, 12)
    )
    one_fold = float(
        np.exp(
            np.percentile(
                logs[one_fold_tasks, one_fold_draws].mean(axis=1),
                90.0,
                method="linear",
            )
        )
    )

    assert observed == expected
    assert observed != one_fold


def test_analyzer_json_boundary_rejects_duplicates_and_nonfinite_numbers():
    with pytest.raises(RuntimeError, match="invalid"):
        analyzer._json_loads('{"value": 1, "value": 2}', "raw")
    with pytest.raises(RuntimeError, match="invalid"):
        analyzer._json_loads('{"value": 1e999}', "raw")


def _none_feature_attestation():
    return {
        "kind": "none",
        "policy_sha256": data_contract.canonical_json_sha256(
            {"kind": "none"}
        ),
        "source_columns_sha256": data_contract.canonical_json_sha256(
            ["x"]
        ),
        "dropped_columns": [],
        "generated_columns": [],
        "generated_values_sha256": (
            data_contract.generated_values_sha256({})
        ),
        "retained_source_columns": ["x"],
        "retained_columns": ["x"],
        "retained_feature_count": 1,
        "retained_columns_sha256": data_contract.canonical_json_sha256(
            ["x"]
        ),
        "output_schema": [{"name": "x", "dtype": "float64"}],
        "output_schema_sha256": data_contract.canonical_json_sha256(
            [{"name": "x", "dtype": "float64"}]
        ),
    }


def _synthetic_registry():
    coordinates = [
        _coordinate(task_id=task, fold=fold)
        for task in range(1, 13)
        for fold in range(3)
    ]
    tasks = []
    for task in range(1, 13):
        splits = [
            {
                "repeat": 0,
                "fold": fold,
                "sample": 0,
                "train_size": 80,
                "test_size": 20,
                "train_index_sha256": "2" * 64,
                "test_index_sha256": "3" * 64,
            }
            for fold in range(3)
        ]
        tasks.append(
            {
                "task_id": task,
                "dataset_id": 10_000 + task,
                "dataset_name": f"dataset-{task}",
                "lineage_cluster": f"lineage-{task}",
                "stratum": common.STRATA[(task - 1) // 4],
                "status": "selected",
                "t5_size_gate_applicability": [False, False, False],
                "ordinal_features": {},
                "feature_policy": {"kind": "none"},
                "feature_policy_attestation": (
                    _none_feature_attestation()
                ),
                "resolved_categorical_columns": [],
                "split_policy": {"kind": "openml_official"},
                "task_record": {
                    "official_splits": {"coordinates": splits}
                },
            }
        )
    exclusions = [
        {
            "task_id": 101,
            "dataset_id": 20_001,
            "lineage_cluster": "excluded-lineage-1",
            "stratum": common.STRATA[0],
            "exposure_kind": (
                "parquet_footer_target_min_max_statistics"
            ),
            "reason": (
                "target_parquet_footer_min_max_observed_before_h1"
            ),
            "replacement_task_id": 1,
        },
        {
            "task_id": 105,
            "dataset_id": 20_005,
            "lineage_cluster": "excluded-lineage-5",
            "stratum": common.STRATA[1],
            "exposure_kind": (
                "parquet_footer_target_min_max_statistics"
            ),
            "reason": (
                "target_parquet_footer_min_max_observed_before_h1"
            ),
            "replacement_task_id": 5,
        },
        {
            "task_id": 109,
            "dataset_id": 20_009,
            "lineage_cluster": "excluded-lineage-9",
            "stratum": common.STRATA[2],
            "exposure_kind": (
                "parquet_footer_target_min_max_statistics"
            ),
            "reason": (
                "target_parquet_footer_min_max_observed_before_h1"
            ),
            "replacement_task_id": 9,
        },
    ]
    source_sha256 = {
        str(path.relative_to(common.ROOT)): "1" * 64
        for path in common.PANEL3_SOURCE_PATHS
    }
    source_sha256["benchmarks/panel3_registry_protocol.md"] = "6" * 64
    source_sha256["benchmarks/run_panel3_confirmation.py"] = "7" * 64
    source_sha256["benchmarks/analyze_panel3_confirmation.py"] = "8" * 64
    source_sha256["benchmarks/panel3_candidate_contract.json"] = "9" * 64
    source_sha256["benchmarks/panel3_environment_contract.json"] = (
        common.load_json(common.CANDIDATE_CONTRACT)["runtime"]["sha256"]
    )
    return {
        "registry_sha256": "a" * 64,
        "pre_h1_target_statistic_exclusions": exclusions,
        "candidate_contract": common.load_json(common.CANDIDATE_CONTRACT),
        "power_design_file_sha256": "4" * 64,
        "power_design_path": (
            "benchmarks/panel3_power_design_decision.json"
        ),
        "power_design_decision_sha256": "5" * 64,
        "power_design_decision": {
            "decision_sha256": "5" * 64,
            "retained_candidates": list(runner.CANDIDATE_ARMS),
            "candidate_count": 2,
            "familywise_one_sided_alpha": 0.05,
            "bootstrap_percentile": 97.5,
            "per_candidate_one_sided_alpha": 0.025,
            "simulation": copy.deepcopy(
                runner.power_design.PANEL3_V1_SIMULATION
            ),
            "target_preflight_authorized": True,
            "pre_h1_target_statistic_exclusions": exclusions,
            "prospective_panel": {
                "slots": [
                    {
                        "task_id": task,
                        "lineage_cluster": f"lineage-{task}",
                        "stratum": common.STRATA[(task - 1) // 4],
                        "t5_size_gate_applicability": [
                            False,
                            False,
                            False,
                        ],
                    }
                    for task in range(1, 13)
                ]
            },
        },
        "retained_candidates": list(runner.CANDIDATE_ARMS),
        "source_sha256": source_sha256,
        "frozen_evidence_sha256": {
            relative: "2" * 64
            for relative in (
                runner.PANEL3_V1_FROZEN_EVIDENCE_RELATIVE_PATHS
            )
        },
        "exposure_catalog": {
            "normalized_names": ["example"],
            "openml_dataset_ids": [1],
            "source_files": {
                relative: "3" * 64
                for relative in (
                    runner.PANEL3_V1_CHIMERA_EXPOSURE_SOURCE_PATHS
                )
            },
            "tabarena_name_count": 1,
            "resolved_name_count": 1,
        },
        "lockbox_darkofit_reference_allowlist": list(
            runner.PANEL3_V1_LOCKBOX_REFERENCE_ALLOWLIST
        ),
        "sources": {
            "darkofit_registry_head": "a" * 40,
            "darkofit_model_head": "b" * 40,
            "darkofit_prefreeze_head": "c" * 40,
            "chimeraboost_head": "b" * 40,
        },
        "coordinates": coordinates,
        "tasks": tasks,
    }


def test_calibration_modules_import_from_clean_interpreter():
    root = Path(__file__).resolve().parents[1]
    commands = [
        [
            sys.executable,
            "-c",
            "import benchmarks.run_panel3_cross_power_calibration",
        ],
        [
            sys.executable,
            "-c",
            "import benchmarks.analyze_panel3_confirmation",
        ],
        [
            sys.executable,
            "-c",
            "import benchmarks.build_panel3_power_design",
        ],
        [
            sys.executable,
            "benchmarks/run_panel3_cross_power_calibration.py",
            "--help",
        ],
    ]
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr


def test_t5_size_gate_binding_rejects_split_size_drift():
    registry = _synthetic_registry()
    decision = registry["power_design_decision"]
    from benchmarks import run_panel3_cross_power_calibration as calibration

    minimum = common.t5_minimum_outer_training_rows(
        registry["candidate_contract"]
    )
    assert runner.t5.SIZE_GATE == minimum == calibration.T5_SIZE_GATE

    runner._validate_t5_size_gate_binding(registry, decision)

    registry["tasks"][0]["task_record"]["official_splits"]["coordinates"][0][
        "train_size"
    ] = 2_000
    with pytest.raises(RuntimeError, match="size-gate binding changed"):
        runner._validate_t5_size_gate_binding(registry, decision)


def test_candidate_contract_rejects_singleton_multiplicity_drift():
    registry = _synthetic_registry()
    decision = registry["power_design_decision"]
    decision["retained_candidates"] = ["guarded_cross_features_policy"]
    decision["candidate_count"] = 1
    decision["per_candidate_one_sided_alpha"] = 0.05
    decision["bootstrap_percentile"] = 95.0

    runner._validate_candidate_power_coherence(
        registry["candidate_contract"],
        decision,
    )

    decision["bootstrap_percentile"] = 97.5
    with pytest.raises(RuntimeError, match="multiplicity contract changed"):
        runner._validate_candidate_power_coherence(
            registry["candidate_contract"],
            decision,
        )


def test_singleton_registry_changes_plan_multiplicity_before_outcomes(
    tmp_path,
    monkeypatch,
):
    registry = _synthetic_registry()
    registry["retained_candidates"] = [
        "guarded_cross_features_policy"
    ]
    registry["power_design_decision"]["retained_candidates"] = [
        "guarded_cross_features_policy"
    ]
    registry["power_design_decision"]["candidate_count"] = 1
    registry["power_design_decision"]["bootstrap_percentile"] = 95.0
    registry["power_design_decision"][
        "per_candidate_one_sided_alpha"
    ] = 0.05
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}")
    monkeypatch.setattr(
        runner,
        "validate_registry",
        lambda *_args, **_kwargs: None,
    )

    plan = runner.execution_plan(registry, registry_path=registry_path)

    assert plan["candidate_arms"] == [
        "guarded_cross_features_policy"
    ]
    assert plan["decision_arms"] == [
        "current_default",
        "guarded_cross_features_policy",
    ]
    assert plan["arm_order"] == [
        "current_default",
        "guarded_cross_features_policy",
        *runner.COMPARATOR_ARMS,
    ]
    assert plan["decision_worker_count"] == 72
    assert plan["worker_count"] == 144


def test_singleton_adjudication_uses_frozen_alpha_and_precedence():
    candidate = {
        "guarded_cross_features_policy": {"passes": True}
    }

    result = analyzer.adjudicate_two_candidates(
        candidate,
        retained_candidates=("guarded_cross_features_policy",),
        per_candidate_one_sided_alpha=0.05,
        bootstrap_percentile=95.0,
    )

    assert result["selected_default_candidate"] == (
        "guarded_cross_features_policy"
    )
    assert result["per_candidate_one_sided_alpha"] == 0.05
    assert result["bootstrap_percentile"] == 95.0
    assert result["multiplicity_method"].startswith(
        "single preregistered"
    )


def _synthetic_historical_registry():
    registry = _synthetic_registry()
    registry.pop("registry_sha256")
    decision = {
        "decision_sha256": "5" * 64,
        "retained_candidates": list(runner.CANDIDATE_ARMS),
        "candidate_count": 2,
        "familywise_one_sided_alpha": 0.05,
        "per_candidate_one_sided_alpha": 0.025,
        "bootstrap_percentile": 97.5,
        "simulation": copy.deepcopy(
            runner.power_design.PANEL3_V1_SIMULATION
        ),
        "target_preflight_authorized": True,
        "pre_h1_target_statistic_exclusions": registry[
            "pre_h1_target_statistic_exclusions"
        ],
        "prospective_panel": registry["power_design_decision"][
            "prospective_panel"
        ],
    }
    registry.update(
        {
            "schema_version": 1,
            "selected_task_count": 12,
            "selected_lineage_count": 12,
            "coordinate_count": 36,
            "outcome_blind": True,
            "target_statistics_used": False,
            "candidate_or_control_models_fitted": False,
            "candidate_or_control_outcomes_inspected": False,
            "lockbox_outcomes_used": False,
            "registry_freeze_complete": True,
            "runner_implementation_complete": True,
            "confirmation_run_authorized": True,
            "default_promotion_authorized": False,
            "created_from_clean_sources": True,
            "candidate_contract": common.load_json(
                common.CANDIDATE_CONTRACT
            ),
            "power_design_decision": decision,
            "power_design_decision_sha256": "5" * 64,
            "retained_candidates": list(runner.CANDIDATE_ARMS),
            "power_analysis": decision,
            "target_preflight_path": (
                "benchmarks/panel3_target_preflight.json"
            ),
            "target_preflight_file_sha256": "3" * 64,
            "target_preflight_sha256": "4" * 64,
        }
    )
    return common.bind_artifact_sha256(registry, "registry_sha256")


def _synthetic_darkofit_fit():
    return {
        "best_iteration": 10,
        "fitted_tree_count": 10,
        "resolved_learning_rate": 0.1,
        "requested_tree_mode": "catboost",
        "selected_tree_mode": "catboost",
        "selected_lane": "boosting",
        "final_fit": {
            "stop_reason": "iteration_limit",
        },
    }


def _synthetic_selection_fit(name, score=1.0):
    return {
        "name": name,
        "validation_rmse": float(score),
        "fit_seconds": 0.2,
        "fit_metadata": _synthetic_darkofit_fit(),
        "validation": {"source": "explicit_eval_set"},
    }


def _synthetic_inner_split():
    return {
        "policy": "ShuffleSplit",
        "random_state": 4,
        "validation_fraction": 0.2,
        "train_rows": 64,
        "validation_rows": 16,
        "train_positions_sha256": "8" * 64,
        "validation_positions_sha256": "9" * 64,
    }


def _synthetic_source_state(repository, head):
    return {
        "repository": repository,
        "head": head,
        "branch": "main",
        "clean": True,
        "status": [],
        "describe": head[:7],
        "tracked_main_refs": {"origin/main": head},
    }


def _synthetic_source_attestation(
    registry_file_sha256="0" * 64,
    registry_canonical_sha256="a" * 64,
):
    attestation = {
        "registry_file_sha256": registry_file_sha256,
        "registry_canonical_sha256": registry_canonical_sha256,
        "darkofit": _synthetic_source_state(
            runner.SOURCE_REPOSITORY_IDS["darkofit"], "a" * 40
        ),
        "chimeraboost": _synthetic_source_state(
            runner.SOURCE_REPOSITORY_IDS["chimeraboost"], "b" * 40
        ),
    }
    return {
        "before": copy.deepcopy(attestation),
        "after": copy.deepcopy(attestation),
    }


def test_panel3_public_attestations_redact_host_paths(
    monkeypatch,
):
    source = {
        "path": "/Users/private-user/Code/private-checkout",
        "head": "a" * 40,
        "branch": "main",
        "clean": True,
        "status": [],
        "describe": "aaaaaaa",
        "remotes": {
            "origin": "file:///Users/private-user/private-remote.git"
        },
        "tracked_main_refs": {"origin/main": "a" * 40},
    }
    darkofit = runner._public_source_state(source, "darkofit")
    chimeraboost = runner._public_source_state(
        {**source, "head": "b" * 40},
        "chimeraboost",
    )
    monkeypatch.setattr(
        runner.creator,
        "_machine_details",
        lambda: {
            "platform": "test-platform",
            "machine": "arm64",
            "cpu_brand": "test-cpu",
            "logical_cpu_count": 8,
            "python": "3.12.13",
            "python_executable": (
                "/Users/private-user/.venvs/darkofit/bin/python"
            ),
        },
    )
    machine = runner._public_machine_details()
    encoded = json.dumps(
        {
            "sources": [darkofit, chimeraboost],
            "machine": machine,
        },
        sort_keys=True,
    )

    assert "/Users/private-user" not in encoded
    assert "private-remote" not in encoded
    assert darkofit["repository"] == "kmedved/darkofit"
    assert chimeraboost["repository"] == "bbstats/chimeraboost"
    assert "python_executable" not in machine

    attestation = {
        "registry_file_sha256": "0" * 64,
        "registry_canonical_sha256": "1" * 64,
        "darkofit": darkofit,
        "chimeraboost": chimeraboost,
    }
    analyzer._validate_source_attestation(
        {
            "before": copy.deepcopy(attestation),
            "after": copy.deepcopy(attestation),
        }
    )


def _synthetic_result(coordinate, arm, ratio):
    full = dict(coordinate)
    prediction = "1" * 64
    if arm == "guarded_cross_features_policy":
        constant = _synthetic_selection_fit("uncrossed_constant")
        linear = _synthetic_selection_fit("uncrossed_linear", 1.1)
        metadata = {
            "kind": arm,
            "engaged": False,
            "decline_reason": "cross_guard",
            "split": _synthetic_inner_split(),
            "cross_guard_ratio": 0.95,
            "selected_configuration": "uncrossed",
            "selected_linear_leaves": False,
            "selected_crosses": False,
            "candidate_cross_pairs": [],
            "selected_cross_pairs": [],
            "selected_cross_pair_count": 0,
            "uncrossed_validation_rmse": 1.0,
            "crossed_validation_rmse": None,
            "relative_crossed_validation_ratio": None,
            "selected_best_iteration": 10,
            "selected_resolved_learning_rate": 0.1,
            "selection_fits": [constant, linear],
            "selected_selection_fit": constant,
            "total_selection_fit_seconds": 0.4,
            "policy_overhead_seconds": 0.1,
            "final_transform_seconds": 0.0,
            "final_model_fit_seconds": 0.5,
            "final_fit_seconds": 0.5,
            "final_refit_parameters": {
                "iterations": 10,
                "learning_rate": 0.1,
                "tree_mode": "catboost",
                "linear_leaves": False,
                "crossed": False,
            },
            "final_fit": _synthetic_darkofit_fit(),
        }
    elif arm == "t5_composite_policy":
        metadata = {
            "kind": arm,
            "engaged": False,
            "decline_reason": "below_size_gate",
            "size_gate": 2_000,
            "total_selection_fit_seconds": 0.0,
            "policy_overhead_seconds": 0.1,
            "final_fit_seconds": 0.9,
            "selected_configuration": "product_default",
            "final_fit": _synthetic_darkofit_fit(),
        }
    elif arm == runner.CONTROL_ARM:
        metadata = {
            "kind": arm,
            "engaged": False,
            "selected_configuration": "product_default",
            "final_fit": _synthetic_darkofit_fit(),
        }
    elif arm == "chimeraboost_0_15_0":
        metadata = {
            "kind": arm,
            "requested_iterations": 2000,
            "attempted_iterations": 10,
            "best_iteration": 10,
            "fitted_tree_count": 10,
            "resolved_learning_rate": 0.1,
            "selected_mode": "symmetric_oblivious",
            "selected_lane": "constant_leaves",
            "stop_reason": "no_legal_split_or_internal_selection",
            "early_stopping": False,
            "selection_rounds": None,
            "linear_leaves_selected": False,
            "cross_features_selected": False,
            "cross_pairs": [],
        }
    else:
        metadata = {
            "kind": arm,
            "requested_iterations": 1000,
            "attempted_iterations": 1000,
            "best_iteration": -1,
            "fitted_tree_count": 1000,
            "resolved_learning_rate": 0.03,
            "selected_mode": "SymmetricTree",
            "selected_lane": "Plain",
            "stop_reason": "iteration_limit",
            "external_categorical_transform_included_in_fit_timing": True,
            "external_categorical_transform_included_in_predict_timing": True,
        }
    behavior = {
        "coordinate": full,
        "arm": arm,
        "rmse": float(ratio),
        "prediction_sha256": prediction,
        "metadata": metadata,
        "source_attestation": _synthetic_source_attestation(),
    }
    return {
        "worker_key": runner._worker_key(coordinate, arm),
        "task_id": coordinate["task_id"],
        "dataset_id": 10_000 + coordinate["task_id"],
        "dataset_name": f"dataset-{coordinate['task_id']}",
        "lineage_cluster": f"lineage-{coordinate['task_id']}",
        "stratum": common.STRATA[(coordinate["task_id"] - 1) // 4],
        "coordinate": {
            "repeat": coordinate["repeat"],
            "fold": coordinate["fold"],
            "sample": coordinate["sample"],
        },
        "arm": arm,
        "categorical_feature_indices": [],
        "categorical_feature_names": [],
        "ordinal_features": {},
        "feature_policy": _none_feature_attestation(),
        "train_rows": 80,
        "test_rows": 20,
        "train_index_sha256": "2" * 64,
        "test_index_sha256": "3" * 64,
        "split_policy": {
            "kind": "openml_official",
            "allow_unused_rows": False,
            "construction_sha256": None,
        },
        "target_sha256": "4" * 64,
        "rmse": float(ratio),
        "fit_seconds": 1.0,
        "prediction_timing": {
            "per_call_median_seconds": 0.01,
            "per_call_min_seconds": 0.009,
            "per_call_max_seconds": 0.011,
            "total_seconds": 0.25,
            "call_count": 25,
            "minimum_block_seconds": 0.25,
        },
        "prediction_sha256": prediction,
        "metadata": metadata,
        "source_attestation": behavior["source_attestation"],
        "warmup_seconds": 0.0,
        "wall_seconds": 1.3,
        "peak_rss_bytes": 1_000_000,
        "behavior_fingerprint_sha256": runner._json_sha256(behavior),
        "worker_stdout": None,
        "worker_stderr": None,
    }


def _synthetic_engaged_t5_result():
    coordinate = _coordinate()
    result = _synthetic_result(
        coordinate,
        "t5_composite_policy",
        1.0,
    )
    control = _synthetic_selection_fit("control_audition", 1.0)
    auto = _synthetic_selection_fit("challenger_auto", 0.9)
    linear = _synthetic_selection_fit(
        "challenger_catboost_linear",
        0.89,
    )
    linear["fit_metadata"]["selected_lane"] = "linear_leaves"
    final = _synthetic_darkofit_fit()
    final["selected_lane"] = "linear_leaves"
    result["metadata"] = {
        "kind": "t5_composite_policy",
        "engaged": True,
        "decline_reason": None,
        "size_gate": 2_000,
        "split": _synthetic_inner_split(),
        "outer_guard_ratio": 0.995,
        "cross_guard_ratio": 0.95,
        "selection_rounds": 100,
        "control_validation_rmse": 1.0,
        "challenger_validation_rmse": 0.89,
        "relative_challenger_validation_ratio": 0.89,
        "selected_configuration": "challenger",
        "selected_tree_mode": "catboost",
        "selected_linear_leaves": True,
        "selected_crosses": False,
        "selected_cross_pairs": [],
        "selected_cross_pair_count": 0,
        "selected_best_iteration": 10,
        "selected_resolved_learning_rate": 0.1,
        "selection_fits": [control, auto, linear],
        "total_selection_fit_seconds": 0.6,
        "policy_overhead_seconds": 0.1,
        "final_transform_seconds": 0.0,
        "final_fit_seconds": 0.3,
        "final_fit": final,
    }
    _refresh_result_behavior(result)
    return result


def _synthetic_declined_t5_result():
    result = _synthetic_engaged_t5_result()
    metadata = result["metadata"]
    for record in metadata["selection_fits"]:
        if record["name"] != "control_audition":
            record["validation_rmse"] = 1.0
    metadata.update(
        {
            "engaged": False,
            "decline_reason": "outer_validation_guard",
            "challenger_validation_rmse": 1.0,
            "relative_challenger_validation_ratio": 1.0,
            "selected_configuration": "product_default",
            "selected_linear_leaves": False,
        }
    )
    metadata["final_fit"]["selected_lane"] = "boosting"
    _refresh_result_behavior(result)
    return result


def _synthetic_raw(tmp_path, monkeypatch):
    registry = _synthetic_registry()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}")
    monkeypatch.setattr(
        runner,
        "validate_registry",
        lambda _registry, registry_path: None,
    )
    monkeypatch.setattr(
        runner,
        "validate_registry_historical",
        lambda _registry, registry_path: None,
    )
    runtime = copy.deepcopy(runner.PANEL3_V1_RUNTIME_CONTRACT)
    machine_payload = {
        "os": "SyntheticOS",
        "os_release": "1",
        "architecture": "synthetic64",
        "cpu_identifier": "Synthetic CPU",
        "physical_cpu_count": 4,
        "logical_cpu_count": 8,
        "memory_bytes": 16_000_000_000,
    }
    machine = {
        **machine_payload,
        "sha256": runner._json_sha256(machine_payload),
    }
    monkeypatch.setattr(
        runner,
        "_validate_runtime_contract",
        lambda _contract: runtime,
    )
    plan = runner.execution_plan(registry, registry_path=registry_path)
    results = []
    records = []
    ratios = {
        runner.CONTROL_ARM: 1.0,
        "t5_composite_policy": 1.0,
        "guarded_cross_features_policy": 0.99,
        "chimeraboost_0_15_0": 0.98,
        "catboost_product_default": 0.97,
    }
    for arm in runner.ARM_ORDER:
        for coordinate in registry["coordinates"]:
            result = _synthetic_result(coordinate, arm, ratios[arm])
            result["source_attestation"] = _synthetic_source_attestation(
                common.sha256_file(registry_path),
                registry["registry_sha256"],
            )
            _refresh_result_behavior(result)
            results.append(result)
    raw = {
        "schema_version": 1,
        "name": "darkofit_panel3_confirmation_raw_v1",
        "created_at": "2026-07-18T00:00:00+00:00",
        "registry": {
            "path": runner._artifact_path(registry_path),
            "file_sha256": common.sha256_file(registry_path),
            "canonical_sha256": registry["registry_sha256"],
        },
        "execution_plan": plan,
        "protocol": {
            "path": "benchmarks/panel3_registry_protocol.md",
            "sha256": "6" * 64,
            "runner_path": "benchmarks/run_panel3_confirmation.py",
            "runner_sha256": "7" * 64,
            "analyzer_path": "benchmarks/analyze_panel3_confirmation.py",
            "analyzer_sha256": "8" * 64,
            "candidate_contract_path": "benchmarks/panel3_candidate_contract.json",
            "candidate_contract_sha256": "9" * 64,
            "power_design_decision_path": (
                "benchmarks/panel3_power_design_decision.json"
            ),
            "power_design_file_sha256": "4" * 64,
            "power_design_decision_sha256": "5" * 64,
            "arms": list(runner._arm_order(registry)),
            "coordinate_count": 36,
            "worker_count": 36 * len(runner._arm_order(registry)),
            "decision_worker_count": (
                36 * len(runner._decision_arms(registry))
            ),
            "successful_worker_count": (
                36 * len(runner._arm_order(registry))
            ),
            "comparator_failure_count": 0,
            "comparator_failures_affect_candidate_gates": False,
            "threads_per_worker": runner.THREADS_PER_WORKER,
            "concurrent_workers": runner.CONCURRENT_WORKERS,
            "validation_fraction": runner.VALIDATION_FRACTION,
            "guarded_cross_ratio": runner.GUARDED_CROSS_RATIO,
            "prediction_block_seconds": runner.PREDICTION_BLOCK_SECONDS,
            "prediction_min_calls": runner.PREDICTION_MIN_CALLS,
            "prediction_max_calls": runner.PREDICTION_MAX_CALLS,
            "task_drop_allowed": False,
            "task_imputation_allowed": False,
            "outcome_dependent_rerun_allowed": False,
        },
        "sources": {
            "darkofit": _synthetic_source_state(
                runner.SOURCE_REPOSITORY_IDS["darkofit"], "a" * 40
            ),
            "chimeraboost": _synthetic_source_state(
                runner.SOURCE_REPOSITORY_IDS["chimeraboost"], "b" * 40
            ),
        },
        "environment": {
            "runtime_contract": runtime,
            "runtime_contract_normalized_sha256": runner._json_sha256(
                runtime
            ),
            "machine_fingerprint": machine,
        },
        "spool": {
            "directory": str(tmp_path / "spool"),
            "binding": {
                "schema_version": 1,
                "runner_sha256": "7" * 64,
                "analyzer_sha256": "8" * 64,
                "protocol_sha256": "6" * 64,
                "candidate_contract_sha256": "9" * 64,
                "power_design_decision_sha256": "5" * 64,
                "registry_file_sha256": common.sha256_file(registry_path),
                "registry_canonical_sha256": registry["registry_sha256"],
                "runtime_contract_normalized_sha256": (
                    runner._json_sha256(runtime)
                ),
                "machine_fingerprint_sha256": machine["sha256"],
                "darkofit_head": "a" * 40,
                "chimeraboost_head": "b" * 40,
                "arms": list(runner._arm_order(registry)),
                "coordinate_count": 36,
            },
            "record_count": 36 * len(runner._arm_order(registry)),
            "resumed_record_count": 0,
            "records": records,
        },
        "results": results,
        "comparator_failures": [],
        "outcomes_scored": True,
        "analysis_performed": False,
        "default_promotion_authorized": False,
        "protocol_deviations": [],
        "task_imputation_used": False,
        "task_drop_used": False,
    }
    binding = raw["spool"]["binding"]
    for result in results:
        coordinate = {"task_id": result["task_id"], **result["coordinate"]}
        payload = runner._spool_payload(
            binding,
            coordinate,
            result["arm"],
            result,
        )
        attempt = runner._attempt_payload(
            binding,
            coordinate,
            result["arm"],
        )
        claim = runner._claim_payload(
            binding,
            coordinate,
            result["arm"],
            attempt["attempt_sha256"],
            runner._json_file_sha256(attempt),
        )
        records.append(
            {
                "worker_key": result["worker_key"],
                "filename": result["worker_key"] + ".json",
                "spool_record_sha256": payload[
                    "spool_record_sha256"
                ],
                "spool_file_sha256": runner._json_file_sha256(payload),
                "attempt_filename": result["worker_key"]
                + ".attempt.json",
                "attempt_sha256": attempt["attempt_sha256"],
                "attempt_file_sha256": runner._json_file_sha256(
                    attempt
                ),
                "claim_filename": result["worker_key"] + ".claim.json",
                "claim_sha256": claim["claim_sha256"],
                "claim_file_sha256": runner._json_file_sha256(claim),
                "resumed": False,
            }
        )
    raw["raw_artifact_sha256"] = runner._json_sha256(raw)
    return registry, registry_path, raw


def _refresh_spool_digest(raw, payload):
    coordinate = {"task_id": payload["task_id"], **payload["coordinate"]}
    spool_payload = runner._spool_payload(
        raw["spool"]["binding"],
        coordinate,
        payload["arm"],
        payload,
    )
    record = next(
        row
        for row in raw["spool"]["records"]
        if row["worker_key"] == payload["worker_key"]
    )
    record["spool_record_sha256"] = spool_payload[
        "spool_record_sha256"
    ]
    record["spool_file_sha256"] = runner._json_file_sha256(
        spool_payload
    )


def _refresh_result_integrity(raw, result):
    _refresh_result_behavior(result)
    _refresh_spool_digest(raw, result)


def _refresh_result_behavior(result):
    result["behavior_fingerprint_sha256"] = runner._json_sha256(
        {
            "coordinate": {
                "task_id": result["task_id"],
                **result["coordinate"],
            },
            "arm": result["arm"],
            "rmse": result["rmse"],
            "prediction_sha256": result["prediction_sha256"],
            "metadata": result["metadata"],
            "source_attestation": result["source_attestation"],
        }
    )


def _refresh_raw_digest(raw):
    unhashed = dict(raw)
    unhashed.pop("raw_artifact_sha256", None)
    raw["raw_artifact_sha256"] = runner._json_sha256(unhashed)


def test_spool_rejects_right_identity_malformed_result_before_write(
    tmp_path,
    monkeypatch,
):
    coordinate = _coordinate()
    result = _synthetic_result(
        coordinate,
        runner.CONTROL_ARM,
        1.0,
    )
    result.pop("rmse")
    writes = []
    monkeypatch.setattr(
        common,
        "atomic_create",
        lambda *_args, **_kwargs: writes.append(True),
    )

    with pytest.raises(RuntimeError, match="result fields changed"):
        runner._create_spool(
            tmp_path / "malformed.json",
            {},
            coordinate,
            runner.CONTROL_ARM,
            result,
        )

    assert writes == []


def test_confirmation_cli_rejects_noncanonical_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner,
        "run_parent",
        lambda _args: pytest.fail("campaign must not run"),
    )

    with pytest.raises(RuntimeError, match="execution path changed"):
        runner.main(["--registry", str(tmp_path / "registry.json")])
    with pytest.raises(RuntimeError, match="execution path changed"):
        runner.main(["--output", str(tmp_path / "raw.json")])
    with pytest.raises(RuntimeError, match="execution path changed"):
        runner.main(["--spool-directory", str(tmp_path / "spool")])


def test_direct_worker_cli_refuses_before_model_work(monkeypatch):
    monkeypatch.delenv(runner.WORKER_SPOOL_DIRECTORY_ENV, raising=False)
    monkeypatch.delenv(runner.WORKER_BINDING_SHA256_ENV, raising=False)
    monkeypatch.delenv(runner.WORKER_BINDING_JSON_ENV, raising=False)
    monkeypatch.setattr(
        common,
        "secure_load_json",
        lambda _path: ({"registry_sha256": "a" * 64}, "b" * 64),
    )
    monkeypatch.setattr(
        runner,
        "run_worker",
        lambda *_args, **_kwargs: pytest.fail(
            "unclaimed worker reached model work"
        ),
    )

    with pytest.raises(RuntimeError, match="durable parent attempt"):
        runner.main(
            [
                "--worker-task",
                "1",
                "--worker-repeat",
                "0",
                "--worker-fold",
                "0",
                "--worker-sample",
                "0",
                "--worker-arm",
                runner.CONTROL_ARM,
            ]
        )


def test_worker_cli_rejects_mismatched_parent_binding_before_claim_load(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        runner.WORKER_SPOOL_DIRECTORY_ENV,
        str(tmp_path),
    )
    monkeypatch.setenv(runner.WORKER_BINDING_JSON_ENV, "{}")
    monkeypatch.setenv(runner.WORKER_BINDING_SHA256_ENV, "0" * 64)
    monkeypatch.setattr(
        runner,
        "_load_attempt",
        lambda *_args, **_kwargs: pytest.fail(
            "mismatched binding reached attempt load"
        ),
    )

    with pytest.raises(RuntimeError, match="parent binding changed"):
        runner._validate_worker_claim(
            {"registry_sha256": "a" * 64},
            tmp_path / "registry.json",
            tmp_path,
            _coordinate(),
            runner.CONTROL_ARM,
        )


def test_worker_attempt_can_be_consumed_only_once(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    coordinate = _coordinate()
    arm = runner.CONTROL_ARM
    registry = {"registry_sha256": "a" * 64}
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry))
    binding = {
        "registry_file_sha256": common.sha256_file(registry_path),
        "registry_canonical_sha256": registry["registry_sha256"],
    }
    runner._create_attempt(
        runner._attempt_path(tmp_path, coordinate, arm),
        binding,
        coordinate,
        arm,
        allowed_root=tmp_path,
    )
    monkeypatch.setenv(
        runner.WORKER_SPOOL_DIRECTORY_ENV,
        str(tmp_path),
    )
    monkeypatch.setenv(
        runner.WORKER_BINDING_JSON_ENV,
        json.dumps(binding, sort_keys=True, separators=(",", ":")),
    )
    monkeypatch.setenv(
        runner.WORKER_BINDING_SHA256_ENV,
        runner._json_sha256(binding),
    )

    assert runner._validate_worker_claim(
        registry,
        registry_path,
        tmp_path,
        coordinate,
        arm,
    ) == binding
    with pytest.raises(RuntimeError, match="already consumed"):
        runner._validate_worker_claim(
            registry,
            registry_path,
            tmp_path,
            coordinate,
            arm,
        )


def test_analysis_cli_rejects_noncanonical_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(
        analyzer,
        "analyze_raw",
        lambda *_args, **_kwargs: pytest.fail("analysis must not run"),
    )

    for flag, name in (
        ("--input", "raw.json"),
        ("--registry", "registry.json"),
        ("--output", "summary.json"),
        ("--markdown", "result.md"),
    ):
        with pytest.raises(RuntimeError, match="analysis path changed"):
            analyzer.main([flag, str(tmp_path / name)])


def test_analysis_partial_publication_resumes_from_identical_display(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "summary.json"
    markdown = tmp_path / "result.md"
    monkeypatch.setattr(analyzer, "_markdown", lambda _summary: "result\n")
    failed_once = False

    def create(path, payload):
        nonlocal failed_once
        if path == output and not failed_once:
            failed_once = True
            raise FileExistsError("simulated publication race")
        if path.exists():
            raise FileExistsError(f"refusing existing output: {path}")
        path.write_bytes(payload)

    monkeypatch.setattr(common, "atomic_create", create)
    monkeypatch.setattr(
        common,
        "secure_read_bytes",
        lambda path: path.read_bytes(),
    )

    with pytest.raises(FileExistsError, match="publication race"):
        analyzer._publish_artifacts(
            {"value": 1},
            output=output,
            markdown=markdown,
        )

    assert markdown.read_text() == "result\n"
    assert not output.exists()

    analyzer._publish_artifacts(
        {"value": 1},
        output=output,
        markdown=markdown,
    )

    assert markdown.read_text() == "result\n"
    assert json.loads(output.read_text()) == {"value": 1}


def test_analysis_partial_publication_rejects_different_display(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "summary.json"
    markdown = tmp_path / "result.md"
    markdown.write_text("different\n")
    monkeypatch.setattr(analyzer, "_markdown", lambda _summary: "result\n")
    monkeypatch.setattr(
        common,
        "atomic_create",
        lambda path, _payload: (_ for _ in ()).throw(
            FileExistsError(f"refusing existing output: {path}")
        ),
    )
    monkeypatch.setattr(
        common,
        "secure_read_bytes",
        lambda path: path.read_bytes(),
    )

    with pytest.raises(RuntimeError, match="differs from derived display"):
        analyzer._publish_artifacts(
            {"value": 1},
            output=output,
            markdown=markdown,
        )

    assert not output.exists()


def test_malformed_comparator_result_becomes_nonbinding_failure(
    tmp_path,
    monkeypatch,
):
    coordinate = _coordinate()
    arm = "chimeraboost_0_15_0"
    worker_key = runner._worker_key(coordinate, arm)
    malformed = {"worker_key": worker_key, "arm": arm}
    completed = SimpleNamespace(
        returncode=0,
        stdout=(
            runner.WORKER_PREFIX
            + json.dumps(malformed, sort_keys=True)
            + "\n"
        ),
        stderr="",
    )
    persisted = []
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: completed,
    )
    monkeypatch.setattr(runner, "_worker_environment", lambda: {})
    monkeypatch.setattr(
        runner,
        "_guard_parent_campaign_boundary",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runner,
        "_create_attempt",
        lambda *_args, **_kwargs: ("b" * 64, "d" * 64),
    )
    monkeypatch.setattr(
        runner,
        "_load_claim",
        lambda *_args, **_kwargs: ("e" * 64, "f" * 64),
    )

    def create(
        _path,
        _binding,
        _coordinate,
        _arm,
        result,
        **_kwargs,
    ):
        persisted.append(result)
        return result, "a" * 64, "c" * 64

    monkeypatch.setattr(runner, "_create_spool", create)

    (
        result,
        digest,
        file_digest,
        attempt_digest,
        attempt_file_digest,
        claim_digest,
        claim_file_digest,
        resumed,
    ) = runner._run_one(
        tmp_path / "registry.json",
        coordinate,
        arm,
        tmp_path,
        {},
    )

    assert result["status"] == "failed"
    assert result["failure_kind"] == "worker_protocol_failure"
    assert result["arm"] == arm
    assert digest == "a" * 64
    assert file_digest == "c" * 64
    assert attempt_digest == "b" * 64
    assert attempt_file_digest == "d" * 64
    assert claim_digest == "e" * 64
    assert claim_file_digest == "f" * 64
    assert resumed is False
    assert persisted == [result]


def test_guarded_cross_child_fit_metadata_is_required():
    coordinate = _coordinate()
    result = _synthetic_result(
        coordinate,
        "guarded_cross_features_policy",
        1.0,
    )
    result["metadata"]["selection_fits"][0].pop("fit_metadata")

    with pytest.raises(RuntimeError, match="selection metadata is incomplete"):
        analyzer._validate_fitted_metadata(result)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda metadata: metadata.__setitem__(
                "control_validation_rmse", 1.01
            ),
            "outer guard",
        ),
        (
            lambda metadata: metadata.__setitem__(
                "challenger_validation_rmse", 0.88
            ),
            "outer guard",
        ),
        (
            lambda metadata: metadata.__setitem__(
                "selected_linear_leaves", False
            ),
            "selected",
        ),
        (
            lambda metadata: metadata["selection_fits"].pop(),
            "selected",
        ),
    ],
)
def test_t5_strict_metadata_rejects_selection_ledger_mutations(
    mutation,
    message,
):
    result = _synthetic_engaged_t5_result()
    mutation(result["metadata"])
    _refresh_result_behavior(result)

    with pytest.raises(RuntimeError, match=message):
        analyzer._validate_fitted_metadata(result, strict=True)


@pytest.mark.parametrize(
    "arm",
    [
        runner.CONTROL_ARM,
        "t5_composite_policy",
        "guarded_cross_features_policy",
        "chimeraboost_0_15_0",
        "catboost_product_default",
    ],
)
def test_strict_fitted_metadata_requires_kind_to_match_arm(arm):
    result = _synthetic_result(
        _coordinate(),
        arm,
        1.0,
    )
    result["metadata"]["kind"] = "changed"

    with pytest.raises(RuntimeError, match="metadata arm changed"):
        analyzer._validate_fitted_metadata(result, strict=True)


@pytest.mark.parametrize("engaged", [False, True])
def test_t5_strict_metadata_requires_frozen_size_gate_on_both_paths(
    engaged,
):
    result = (
        _synthetic_engaged_t5_result()
        if engaged
        else _synthetic_result(
            _coordinate(),
            "t5_composite_policy",
            1.0,
        )
    )
    result["metadata"]["size_gate"] = 2_001

    with pytest.raises(RuntimeError, match="size gate changed"):
        analyzer._validate_fitted_metadata(result, strict=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selected_configuration", "challenger"),
        ("total_selection_fit_seconds", 0.1),
    ],
)
def test_t5_strict_metadata_binds_below_gate_decline_fields(field, value):
    result = _synthetic_result(
        _coordinate(),
        "t5_composite_policy",
        1.0,
    )
    result["metadata"][field] = value

    with pytest.raises(RuntimeError, match="size-gate fields changed"):
        analyzer._validate_fitted_metadata(result, strict=True)


def test_guarded_strict_metadata_binds_fixed_refit_learning_rate():
    result = _synthetic_result(
        _coordinate(),
        "guarded_cross_features_policy",
        1.0,
    )
    metadata = result["metadata"]
    metadata["selected_resolved_learning_rate"] = 0.2
    metadata["final_refit_parameters"]["learning_rate"] = 0.2
    metadata["selected_selection_fit"]["fit_metadata"][
        "resolved_learning_rate"
    ] = 0.2
    metadata["final_fit"]["resolved_learning_rate"] = 0.2

    with pytest.raises(RuntimeError, match="fixed learning rate changed"):
        analyzer._validate_fitted_metadata(result, strict=True)


@pytest.mark.parametrize(
    ("field", "value", "selected_child_lr"),
    [
        ("selected_best_iteration", 10.0, None),
        ("selected_resolved_learning_rate", "0.1", None),
        ("selected_resolved_learning_rate", True, 1.0),
    ],
)
def test_t5_strict_metadata_binds_declined_selection_types(
    field,
    value,
    selected_child_lr,
):
    result = _synthetic_declined_t5_result()
    metadata = result["metadata"]
    metadata[field] = value
    if selected_child_lr is not None:
        selected = next(
            record
            for record in metadata["selection_fits"]
            if record["name"] == "challenger_auto"
        )
        selected["fit_metadata"][
            "resolved_learning_rate"
        ] = selected_child_lr

    with pytest.raises(RuntimeError, match="producer fields changed"):
        analyzer._validate_fitted_metadata(result, strict=True)


@pytest.mark.parametrize(
    "path",
    ["t5_engaged", "t5_declined", "guarded_declined"],
)
def test_strict_metadata_rejects_string_selection_totals(path):
    if path == "t5_engaged":
        result = _synthetic_engaged_t5_result()
    elif path == "t5_declined":
        result = _synthetic_declined_t5_result()
    else:
        result = _synthetic_result(
            _coordinate(),
            "guarded_cross_features_policy",
            1.0,
        )
    result["metadata"]["total_selection_fit_seconds"] = str(
        result["metadata"]["total_selection_fit_seconds"]
    )

    with pytest.raises(RuntimeError, match="total selection time"):
        analyzer._validate_fitted_metadata(result, strict=True)


def test_t5_strict_metadata_requires_integer_cross_pair_count():
    result = _synthetic_engaged_t5_result()
    metadata = result["metadata"]
    crossed = _synthetic_selection_fit("challenger_crossed", 1.0)
    crossed["fit_metadata"]["selected_lane"] = "linear_leaves"
    crossed.update(
        {
            "pairs": [[0, 1, "diff"]],
            "pair_count": 1.0,
            "transform_seconds": 0.01,
        }
    )
    metadata["selection_fits"].append(crossed)
    metadata["total_selection_fit_seconds"] = 0.8
    result["feature_policy"]["retained_feature_count"] = 2
    result["fit_seconds"] = 1.2

    with pytest.raises(RuntimeError, match="selection-fit fields changed"):
        analyzer._validate_fitted_metadata(result, strict=True)


def test_t5_strict_ratio_identity_has_no_absolute_tolerance():
    result = _synthetic_engaged_t5_result()
    metadata = result["metadata"]
    selected = next(
        record
        for record in metadata["selection_fits"]
        if record["name"] == "challenger_catboost_linear"
    )
    selected["validation_rmse"] = 1e-18
    metadata["challenger_validation_rmse"] = 1e-18
    metadata["relative_challenger_validation_ratio"] = 2e-18

    with pytest.raises(RuntimeError, match="outer guard changed"):
        analyzer._validate_fitted_metadata(result, strict=True)


def test_guarded_strict_ratio_identity_has_no_absolute_tolerance():
    result = _synthetic_result(
        _coordinate(),
        "guarded_cross_features_policy",
        1.0,
    )
    metadata = result["metadata"]
    pairs = [[0, 1, "diff"]]
    crossed = _synthetic_selection_fit(
        "crossed_selected_leaf_lane",
        1e-18,
    )
    crossed.update(
        {
            "pairs": pairs,
            "transform_seconds": 0.01,
        }
    )
    metadata.update(
        {
            "engaged": True,
            "decline_reason": None,
            "selected_configuration": "crossed",
            "selected_crosses": True,
            "candidate_cross_pairs": pairs,
            "selected_cross_pairs": pairs,
            "selected_cross_pair_count": 1,
            "crossed_validation_rmse": 1e-18,
            "relative_crossed_validation_ratio": 2e-18,
            "selected_selection_fit": crossed,
            "selection_fits": [
                *metadata["selection_fits"],
                crossed,
            ],
            "total_selection_fit_seconds": 0.6,
        }
    )
    metadata["final_refit_parameters"]["crossed"] = True
    result["feature_policy"]["retained_feature_count"] = 2
    result["fit_seconds"] = 1.2

    with pytest.raises(RuntimeError, match="cross guard changed"):
        analyzer._validate_fitted_metadata(result, strict=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("uncrossed_validation_rmse", 1.01, "uncrossed score"),
        ("selected_configuration", "crossed", "cross guard"),
        ("selected_linear_leaves", True, "selected"),
        ("candidate_cross_pairs", [[0, 0, "prod"]], "cross-pair"),
    ],
)
def test_guarded_strict_metadata_rejects_selection_ledger_mutations(
    field,
    value,
    message,
):
    result = _synthetic_result(
        _coordinate(),
        "guarded_cross_features_policy",
        1.0,
    )
    result["metadata"][field] = value
    _refresh_result_behavior(result)

    with pytest.raises(RuntimeError, match=message):
        analyzer._validate_fitted_metadata(result, strict=True)


@pytest.mark.parametrize(
    "arm",
    [
        runner.CONTROL_ARM,
        "t5_composite_policy",
        "guarded_cross_features_policy",
        "chimeraboost_0_15_0",
        "catboost_product_default",
    ],
)
def test_fitted_metadata_rejects_boolean_learning_rates(arm):
    coordinate = _coordinate()
    result = _synthetic_result(coordinate, arm, 1.0)
    if arm in {
        runner.CONTROL_ARM,
        "t5_composite_policy",
    }:
        result["metadata"]["final_fit"]["resolved_learning_rate"] = True
    elif arm == "guarded_cross_features_policy":
        result["metadata"]["selected_resolved_learning_rate"] = True
    else:
        result["metadata"]["resolved_learning_rate"] = True

    with pytest.raises(RuntimeError):
        analyzer._validate_fitted_metadata(result)


@pytest.mark.parametrize(
    "arm",
    ["chimeraboost_0_15_0", "catboost_product_default"],
)
@pytest.mark.parametrize("field", ["selected_mode", "selected_lane"])
def test_comparator_metadata_requires_nonempty_mode_and_lane(arm, field):
    coordinate = _coordinate()
    result = _synthetic_result(coordinate, arm, 1.0)
    result["metadata"][field] = ""

    with pytest.raises(RuntimeError, match="metadata is incomplete"):
        analyzer._validate_fitted_metadata(result)


def test_parent_validates_complete_raw_before_publication(
    tmp_path,
    monkeypatch,
):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}")
    monkeypatch.setattr(runner, "DEFAULT_REGISTRY", registry_path)
    output = tmp_path / "raw.json"
    spool = tmp_path / "spool"
    spool.mkdir()
    power_design = tmp_path / "power-design.json"
    power_design.write_text("{}")
    registry = {
        "registry_sha256": "a" * 64,
        "candidate_contract": {},
        "power_design_file_sha256": "4" * 64,
        "power_design_decision_sha256": "5" * 64,
        "retained_candidates": list(runner.CANDIDATE_ARMS),
        "coordinates": [],
    }
    source = {
        "path": str(tmp_path),
        "head": "b" * 40,
        "branch": "main",
        "clean": True,
        "status": [],
        "describe": "bbbbbbb",
        "remotes": {"origin": "file:///private/source"},
        "tracked_main_refs": {"origin/main": "b" * 40},
    }
    events = []
    monkeypatch.setattr(
        common,
        "secure_load_json",
        lambda _path: (registry, "c" * 64),
    )
    monkeypatch.setattr(
        common,
        "POWER_DESIGN_DECISION",
        power_design,
    )
    monkeypatch.setattr(
        common,
        "validate_create_path",
        lambda path: Path(path),
    )
    monkeypatch.setattr(
        common,
        "ensure_output_directory",
        lambda path: Path(path),
    )
    monkeypatch.setattr(
        runner,
        "validate_registry",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runner,
        "_source_state",
        lambda _registry: (source, source),
    )
    monkeypatch.setattr(
        runner,
        "_validate_runtime_contract",
        lambda _contract: {"contract_kind": "synthetic"},
    )
    monkeypatch.setattr(
        runner.creator,
        "_machine_details",
        lambda: {},
    )
    monkeypatch.setattr(
        runner.creator,
        "_dependency_versions",
        lambda: {},
    )
    monkeypatch.setattr(
        analyzer,
        "validate_raw",
        lambda *_args, **_kwargs: events.append("validate"),
    )

    def publish(path, payload, **_kwargs):
        events.append("publish")
        Path(path).write_bytes(payload)

    monkeypatch.setattr(
        common,
        "atomic_create",
        publish,
    )
    args = SimpleNamespace(
        registry=registry_path,
        output=output,
        spool_directory=spool,
    )

    runner.run_parent(args)

    assert events == ["validate", "publish"]


def test_raw_validator_enforces_exact_decline_and_analysis_is_independent(
    tmp_path,
    monkeypatch,
):
    registry, registry_path, raw = _synthetic_raw(tmp_path, monkeypatch)

    summary = analyzer.analyze_raw(
        raw,
        registry,
        raw_file_sha256="e" * 64,
        registry_path=registry_path,
        verify_current_files=False,
        bootstrap_replicates=2_000,
    )

    assert summary["candidate_results"]["t5_composite_policy"]["passes"] is False
    assert (
        summary["candidate_results"]["guarded_cross_features_policy"]["passes"]
        is False
    )
    assert summary["candidate_results"]["guarded_cross_features_policy"][
        "gates"
    ]["integrity"] is False
    assert summary["shipping_candidates"] == []
    assert summary["bootstrap"]["override_used"] is True
    assert summary["bootstrap"]["authorization_eligible"] is False
    assert all(
        row["decision_role"] == "descriptive_only"
        for row in summary["comparators"].values()
    )

    corrupted = copy.deepcopy(raw)
    composite = next(
        row
        for row in corrupted["results"]
        if row["arm"] == "t5_composite_policy"
    )
    composite["prediction_sha256"] = "f" * 64
    _refresh_result_integrity(corrupted, composite)
    _refresh_raw_digest(corrupted)

    with pytest.raises(RuntimeError, match="byte-identical"):
        analyzer.validate_raw(
            corrupted,
            registry,
            registry_path=registry_path,
            verify_current_files=False,
        )

    corrupted = copy.deepcopy(raw)
    composite = next(
        row
        for row in corrupted["results"]
        if row["arm"] == "t5_composite_policy"
    )
    composite["rmse"] *= 0.5
    _refresh_result_integrity(corrupted, composite)
    _refresh_raw_digest(corrupted)

    with pytest.raises(RuntimeError, match="byte-identical"):
        analyzer.validate_raw(
            corrupted,
            registry,
            registry_path=registry_path,
            verify_current_files=False,
        )


def test_historical_raw_reconstructs_attempt_digest_offline(
    tmp_path,
    monkeypatch,
):
    registry, registry_path, raw = _synthetic_raw(tmp_path, monkeypatch)
    raw["spool"]["records"][0]["attempt_sha256"] = "f" * 64
    _refresh_raw_digest(raw)

    with pytest.raises(RuntimeError, match="worker ledger digest"):
        analyzer.validate_raw(
            raw,
            registry,
            registry_path=registry_path,
            verify_current_files=False,
        )


def test_historical_raw_reconstructs_consumed_claim_digest_offline(
    tmp_path,
    monkeypatch,
):
    registry, registry_path, raw = _synthetic_raw(tmp_path, monkeypatch)
    raw["spool"]["records"][0]["claim_sha256"] = "f" * 64
    _refresh_raw_digest(raw)

    with pytest.raises(RuntimeError, match="worker ledger digest"):
        analyzer.validate_raw(
            raw,
            registry,
            registry_path=registry_path,
            verify_current_files=False,
        )


def test_historical_raw_rejects_worker_source_attestation_drift(
    tmp_path,
    monkeypatch,
):
    registry, registry_path, raw = _synthetic_raw(tmp_path, monkeypatch)
    result = raw["results"][0]
    result["source_attestation"]["after"]["darkofit"]["head"] = "f" * 40
    _refresh_result_integrity(raw, result)
    _refresh_raw_digest(raw)

    with pytest.raises(RuntimeError, match="source attestation changed"):
        analyzer.validate_raw(
            raw,
            registry,
            registry_path=registry_path,
            verify_current_files=False,
        )


def test_historical_analysis_does_not_consult_live_registry_files(
    tmp_path,
    monkeypatch,
):
    registry, registry_path, raw = _synthetic_raw(tmp_path, monkeypatch)
    historical_calls = []

    def reject_live(*_args, **_kwargs):
        raise AssertionError("historical analysis consulted live validation")

    monkeypatch.setattr(runner, "validate_registry", reject_live)
    monkeypatch.setattr(
        runner,
        "validate_registry_historical",
        lambda _registry, registry_path: historical_calls.append(
            registry_path
        ),
    )

    analyzer.analyze_raw(
        raw,
        registry,
        raw_file_sha256="e" * 64,
        registry_path=registry_path,
        verify_current_files=False,
        bootstrap_replicates=100,
    )

    assert historical_calls == [registry_path]


def test_registry_snapshot_digest_is_threaded_without_path_reopen(
    tmp_path,
    monkeypatch,
):
    registry, registry_path, raw = _synthetic_raw(tmp_path, monkeypatch)
    registry_file_sha256 = raw["registry"]["file_sha256"]
    monkeypatch.setattr(
        common,
        "sha256_file",
        lambda _path: pytest.fail("registry path was reopened"),
    )

    plan = runner.execution_plan(
        registry,
        registry_path=registry_path,
        registry_file_sha256=registry_file_sha256,
        validate_registry_boundary=False,
    )
    summary = analyzer.analyze_raw(
        raw,
        registry,
        raw_file_sha256="e" * 64,
        registry_path=registry_path,
        registry_file_sha256=registry_file_sha256,
        verify_current_files=False,
        bootstrap_replicates=100,
    )

    assert plan["registry_file_sha256"] == registry_file_sha256
    assert summary["registry_file_sha256"] == registry_file_sha256


def test_historical_analysis_uses_embedded_gates_not_live_constants(
    tmp_path,
    monkeypatch,
):
    registry, registry_path, raw = _synthetic_raw(tmp_path, monkeypatch)
    baseline = analyzer.analyze_raw(
        raw,
        registry,
        raw_file_sha256="e" * 64,
        registry_path=registry_path,
        verify_current_files=False,
        bootstrap_replicates=100,
    )
    monkeypatch.setattr(common, "QUALITY_BAR", 0.01)
    monkeypatch.setattr(common, "UNCERTAINTY_BAR", 0.01)
    monkeypatch.setattr(common, "LOO_BAR", 0.01)
    monkeypatch.setattr(common, "HARM_BAR", 0.01)
    monkeypatch.setattr(common, "FAMILYWISE_ONE_SIDED_ALPHA", 0.9)
    monkeypatch.setattr(analyzer, "BOOTSTRAP_SEED", 1)
    monkeypatch.setattr(analyzer, "BOOTSTRAP_BATCH", 1)
    monkeypatch.setattr(runner, "THREADS_PER_WORKER", 99)
    monkeypatch.setattr(runner, "CONCURRENT_WORKERS", 99)
    monkeypatch.setattr(runner, "VALIDATION_FRACTION", 0.49)
    monkeypatch.setattr(runner, "GUARDED_CROSS_RATIO", 0.01)
    monkeypatch.setattr(runner, "PREDICTION_BLOCK_SECONDS", 99.0)
    monkeypatch.setattr(runner, "PREDICTION_MIN_CALLS", 99)
    monkeypatch.setattr(runner, "PREDICTION_MAX_CALLS", 100)

    replay = analyzer.analyze_raw(
        raw,
        registry,
        raw_file_sha256="e" * 64,
        registry_path=registry_path,
        verify_current_files=False,
        bootstrap_replicates=100,
    )

    assert replay["candidate_results"] == baseline["candidate_results"]
    assert replay["familywise_one_sided_alpha"] == 0.05


def test_historical_registry_validation_uses_embedded_bytes_only(
    tmp_path,
    monkeypatch,
):
    registry = _synthetic_historical_registry()
    registry_path = tmp_path / "historical-registry.json"
    registry_path.write_text("{}")
    monkeypatch.setattr(
        runner.power_design,
        "validate_decision",
        lambda artifact, **_kwargs: artifact,
    )
    monkeypatch.setattr(runner.t5, "SIZE_GATE", runner.t5.SIZE_GATE + 1)
    monkeypatch.setattr(
        common,
        "PRE_H1_TARGET_STATISTIC_EXCLUSIONS",
        [{**registry["pre_h1_target_statistic_exclusions"][0]}],
    )

    runner.validate_registry_historical(
        registry,
        registry_path=registry_path,
    )
    with pytest.raises(RuntimeError, match="ledger changed"):
        runner._validate_pre_h1_target_exclusion_boundary(
            registry,
            require_current_sources=True,
        )

    reintroduced = copy.deepcopy(registry)
    reintroduced["pre_h1_target_statistic_exclusions"][0][
        "task_id"
    ] = 2
    with pytest.raises(RuntimeError, match="lineage re-entered"):
        runner._validate_pre_h1_target_exclusion_boundary(
            reintroduced,
            require_current_sources=False,
        )


def test_descriptive_comparator_failure_cannot_veto_candidates(
    tmp_path,
    monkeypatch,
):
    registry, registry_path, raw = _synthetic_raw(tmp_path, monkeypatch)
    failed = next(
        result
        for result in raw["results"]
        if result["arm"] == "catboost_product_default"
    )
    raw["results"].remove(failed)
    coordinate = {
        "task_id": failed["task_id"],
        **failed["coordinate"],
    }
    raw["comparator_failures"].append(
        runner._comparator_failure(
            coordinate,
            failed["arm"],
            returncode=-6,
            stdout=None,
            stderr="synthetic crash",
            failure_kind="worker_process_failure",
            message="synthetic comparator crash",
        )
    )
    _refresh_spool_digest(raw, raw["comparator_failures"][-1])
    raw["protocol"]["successful_worker_count"] = 179
    raw["protocol"]["comparator_failure_count"] = 1
    _refresh_raw_digest(raw)

    summary = analyzer.analyze_raw(
        raw,
        registry,
        raw_file_sha256="e" * 64,
        registry_path=registry_path,
        verify_current_files=False,
        bootstrap_replicates=2_000,
    )

    assert summary["shipping_candidates"] == []
    assert summary["candidate_results"]["guarded_cross_features_policy"][
        "gates"
    ]["integrity"] is False
    comparator = summary["comparators"]["catboost_product_default"]
    assert comparator["complete"] is False
    assert comparator["failed_coordinate_count"] == 1
    assert comparator["affects_candidate_gates"] is False


def test_both_pass_uses_frozen_t5_precedence_without_metric_ranking():
    passing = {"passes": True}

    result = analyzer.adjudicate_two_candidates(
        {
            "t5_composite_policy": dict(passing),
            "guarded_cross_features_policy": dict(passing),
        }
    )

    assert result["independently_confirmed_candidates"] == [
        "guarded_cross_features_policy",
        "t5_composite_policy",
    ]
    assert result["selected_default_candidate"] == "t5_composite_policy"
    assert result["shipping_candidates"] == ["t5_composite_policy"]
    assert result["post_outcome_winner_selection_used"] is False


def test_runtime_contract_rejects_interpreter_file_and_package_drift(
    tmp_path,
    monkeypatch,
):
    payload = copy.deepcopy(runner.PANEL3_V1_RUNTIME_CONTRACT)
    packages = payload["packages"]
    path = tmp_path / "benchmarks" / "panel3_environment_contract.json"
    path.parent.mkdir()
    path.write_bytes(common.ENVIRONMENT_CONTRACT.read_bytes())
    contract = copy.deepcopy(common.load_json(common.CANDIDATE_CONTRACT))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.sys, "version_info", (3, 12, 13))
    monkeypatch.setattr(
        runner.importlib.metadata,
        "version",
        lambda package: packages[package],
    )

    assert runner._validate_runtime_contract(contract) == payload

    monkeypatch.setattr(
        runner.importlib.metadata,
        "version",
        lambda package: "2.0" if package == "numpy" else packages[package],
    )
    with pytest.raises(RuntimeError, match="numpy version changed"):
        runner._validate_runtime_contract(contract)

    path.write_text(path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="file changed"):
        runner._validate_runtime_contract(contract)


def test_candidate_constructors_and_gate_constants_match_frozen_contract():
    contract = common.load_json(common.CANDIDATE_CONTRACT)
    definitions = {
        row["name"]: row["definition"] for row in contract["candidates"]
    }
    guarded = runner._guarded_model(
        linear_leaves=False,
        iterations=2_000,
        selection=True,
    ).get_params()
    assert {
        "iterations": guarded["iterations"],
        "learning_rate": guarded["learning_rate"],
        "depth": guarded["depth"],
        "l2_leaf_reg": guarded["l2_leaf_reg"],
        "max_bins": guarded["max_bins"],
        "min_child_weight": guarded["min_child_weight"],
        "tree_mode": guarded["tree_mode"],
    } == {
        "iterations": definitions[
            "guarded_cross_features_policy"
        ]["n_estimators"],
        **{
            key: definitions["guarded_cross_features_policy"][key]
            for key in (
            "learning_rate",
            "depth",
            "l2_leaf_reg",
            "max_bins",
            "min_child_weight",
            "tree_mode",
            )
        },
    }

    decision = contract["decision"]
    assert runner.GUARDED_CROSS_RATIO == definitions[
        "guarded_cross_features_policy"
    ]["cross_guard_ratio"]
    assert common.QUALITY_BAR == decision[
        "equal_dataset_geomean_ratio_at_most"
    ]
    assert common.UNCERTAINTY_BAR == decision[
        "bootstrap_upper_ratio_at_most"
    ]
    assert common.LOO_BAR == decision[
        "leave_one_favorable_dataset_out_ratio_at_most"
    ]
    assert common.HARM_BAR == decision["worst_dataset_ratio_at_most"]


def test_guarded_cross_accepts_valid_no_split_selection_fit():
    rng = np.random.default_rng(10)
    X = pd.DataFrame(rng.normal(size=(80, 3)))
    y = pd.Series(X[0] - X[1])

    prediction, _fit_seconds, _timing, metadata = (
        runner._fit_guarded_cross(
            X.iloc[:64],
            y.iloc[:64],
            [],
            X.iloc[64:],
        )
    )

    crossed = next(
        row
        for row in metadata["selection_fits"]
        if row["name"] == "crossed_selected_leaf_lane"
    )
    assert crossed["fit_metadata"]["final_fit"]["stop_reason"] == "no_split"
    assert metadata["engaged"] is True
    assert np.isfinite(prediction).all()


def test_spool_record_is_create_only_and_tamper_evident(tmp_path):
    coordinate = _coordinate()
    arm = runner.CONTROL_ARM
    path = tmp_path / "record.json"
    binding = {"campaign": "panel3"}
    result = _synthetic_result(coordinate, arm, 1.0)
    different = copy.deepcopy(result)
    different["rmse"] = 2.0
    different["behavior_fingerprint_sha256"] = runner._json_sha256(
        {
            "coordinate": coordinate,
            "arm": arm,
            "rmse": different["rmse"],
            "prediction_sha256": different["prediction_sha256"],
            "metadata": different["metadata"],
            "source_attestation": different["source_attestation"],
        }
    )

    created, digest, file_digest = runner._create_spool(
        path,
        binding,
        coordinate,
        arm,
        result,
        allowed_root=tmp_path,
    )
    resumed, resumed_digest, resumed_file_digest = runner._create_spool(
        path,
        binding,
        coordinate,
        arm,
        different,
        allowed_root=tmp_path,
    )

    assert created == resumed == result
    assert digest == resumed_digest
    assert file_digest == resumed_file_digest
    payload = json.loads(path.read_text())
    payload["result"]["arm"] = "changed"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="hash is invalid"):
        runner._load_spool(
            path,
            binding,
            coordinate,
            arm,
            allowed_root=tmp_path,
        )


def test_worker_attempt_is_create_only_and_tamper_evident(tmp_path):
    coordinate = _coordinate()
    arm = runner.CONTROL_ARM
    path = runner._attempt_path(tmp_path, coordinate, arm)
    binding = {"campaign": "panel3"}

    digest, file_digest = runner._create_attempt(
        path,
        binding,
        coordinate,
        arm,
        allowed_root=tmp_path,
    )

    assert (digest, file_digest) == runner._load_attempt(
        path,
        binding,
        coordinate,
        arm,
        allowed_root=tmp_path,
    )
    with pytest.raises(FileExistsError):
        runner._create_attempt(
            path,
            binding,
            coordinate,
            arm,
            allowed_root=tmp_path,
        )
    payload = json.loads(path.read_text())
    payload["arm"] = "changed"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="attempt changed"):
        runner._load_attempt(
            path,
            binding,
            coordinate,
            arm,
            allowed_root=tmp_path,
        )


def test_worker_crash_leaves_permanent_attempt_and_restart_refuses(
    tmp_path,
    monkeypatch,
):
    coordinate = _coordinate()
    binding = {"campaign": "panel3"}
    calls = []
    monkeypatch.setattr(
        runner,
        "_guard_parent_campaign_boundary",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(runner, "_worker_environment", lambda: {})

    def crash(*_args, **_kwargs):
        calls.append("launched")
        attempt_digest, attempt_file_digest = runner._load_attempt(
            runner._attempt_path(
                tmp_path,
                coordinate,
                runner.CONTROL_ARM,
            ),
            binding,
            coordinate,
            runner.CONTROL_ARM,
            allowed_root=tmp_path,
        )
        runner._create_claim(
            runner._claim_path(
                tmp_path,
                coordinate,
                runner.CONTROL_ARM,
            ),
            binding,
            coordinate,
            runner.CONTROL_ARM,
            attempt_digest,
            attempt_file_digest,
            allowed_root=tmp_path,
        )
        return SimpleNamespace(
            returncode=9,
            stdout="",
            stderr="synthetic crash",
        )

    monkeypatch.setattr(runner.subprocess, "run", crash)
    with pytest.raises(RuntimeError, match="failed with 9"):
        runner._run_one(
            tmp_path / "registry.json",
            coordinate,
            runner.CONTROL_ARM,
            tmp_path,
            binding,
        )

    assert calls == ["launched"]
    assert runner._attempt_path(
        tmp_path, coordinate, runner.CONTROL_ARM
    ).is_file()
    assert runner._claim_path(
        tmp_path, coordinate, runner.CONTROL_ARM
    ).is_file()
    assert not runner._spool_path(
        tmp_path, coordinate, runner.CONTROL_ARM
    ).exists()
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "a spent coordinate must never relaunch"
        ),
    )
    with pytest.raises(RuntimeError, match="permanently invalid"):
        runner._run_one(
            tmp_path / "registry.json",
            coordinate,
            runner.CONTROL_ARM,
            tmp_path,
            binding,
        )


def test_decision_worker_crash_before_claim_remains_permanently_invalid(
    tmp_path,
    monkeypatch,
):
    coordinate = _coordinate()
    arm = runner.CONTROL_ARM
    binding = {"campaign": "panel3"}
    monkeypatch.setattr(
        runner,
        "_guard_parent_campaign_boundary",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(runner, "_worker_environment", lambda: {})
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=9,
            stdout="",
            stderr="synthetic pre-claim crash",
        ),
    )

    with pytest.raises(RuntimeError, match="invalid panel-3 worker claim"):
        runner._run_one(
            tmp_path / "registry.json",
            coordinate,
            arm,
            tmp_path,
            binding,
        )

    assert runner._attempt_path(tmp_path, coordinate, arm).is_file()
    assert not runner._claim_path(tmp_path, coordinate, arm).exists()
    assert not runner._spool_path(tmp_path, coordinate, arm).exists()
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "a spent coordinate must never relaunch"
        ),
    )
    with pytest.raises(RuntimeError, match="permanently invalid"):
        runner._run_one(
            tmp_path / "registry.json",
            coordinate,
            arm,
            tmp_path,
            binding,
        )


def test_parent_refuses_worker_output_without_consumed_claim(
    tmp_path,
    monkeypatch,
):
    coordinate = _coordinate()
    arm = runner.CONTROL_ARM
    binding = {"campaign": "panel3"}
    result = _synthetic_result(coordinate, arm, 1.0)
    monkeypatch.setattr(
        runner,
        "_guard_parent_campaign_boundary",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(runner, "_worker_environment", lambda: {})
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=runner.WORKER_PREFIX
            + json.dumps(result, sort_keys=True)
            + "\n",
            stderr="",
        ),
    )
    monkeypatch.setattr(
        runner,
        "_create_spool",
        lambda *_args, **_kwargs: pytest.fail(
            "unclaimed output reached spool publication"
        ),
    )

    with pytest.raises(RuntimeError, match="invalid panel-3 worker claim"):
        runner._run_one(
            tmp_path / "registry.json",
            coordinate,
            arm,
            tmp_path,
            binding,
        )


def test_comparator_launch_failure_consumes_claim_and_persists_failure(
    tmp_path,
    monkeypatch,
):
    coordinate = _coordinate()
    arm = "chimeraboost_0_15_0"
    binding = {"campaign": "panel3"}
    monkeypatch.setattr(
        runner,
        "_guard_parent_campaign_boundary",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(runner, "_worker_environment", lambda: {})
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("synthetic launch failure")
        ),
    )

    (
        result,
        _spool_digest,
        _spool_file_digest,
        _attempt_digest,
        _attempt_file_digest,
        _claim_digest,
        _claim_file_digest,
        resumed,
    ) = runner._run_one(
        tmp_path / "registry.json",
        coordinate,
        arm,
        tmp_path,
        binding,
    )

    assert result["status"] == "failed"
    assert result["failure_kind"] == "worker_launch_failure"
    assert resumed is False
    assert runner._claim_path(tmp_path, coordinate, arm).is_file()
    assert runner._spool_path(tmp_path, coordinate, arm).is_file()


def test_comparator_process_failure_before_claim_persists_and_resumes(
    tmp_path,
    monkeypatch,
):
    coordinate = _coordinate()
    arm = "chimeraboost_0_15_0"
    binding = {"campaign": "panel3"}
    monkeypatch.setattr(
        runner,
        "_guard_parent_campaign_boundary",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(runner, "_worker_environment", lambda: {})
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=9,
            stdout="",
            stderr="synthetic pre-claim crash",
        ),
    )

    first = runner._run_one(
        tmp_path / "registry.json",
        coordinate,
        arm,
        tmp_path,
        binding,
    )

    result = first[0]
    assert result["status"] == "failed"
    assert result["failure_kind"] == "worker_process_failure"
    assert first[-1] is False
    assert runner._attempt_path(tmp_path, coordinate, arm).is_file()
    assert runner._claim_path(tmp_path, coordinate, arm).is_file()
    assert runner._spool_path(tmp_path, coordinate, arm).is_file()
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "a completed comparator must resume without relaunch"
        ),
    )

    resumed = runner._run_one(
        tmp_path / "registry.json",
        coordinate,
        arm,
        tmp_path,
        binding,
    )

    assert resumed[:-1] == first[:-1]
    assert resumed[-1] is True


def test_comparator_process_failure_uses_existing_valid_claim(
    tmp_path,
    monkeypatch,
):
    coordinate = _coordinate()
    arm = "chimeraboost_0_15_0"
    binding = {"campaign": "panel3"}
    create_claim = runner._create_claim
    claim_create_count = 0
    monkeypatch.setattr(
        runner,
        "_guard_parent_campaign_boundary",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(runner, "_worker_environment", lambda: {})

    def claimed_crash(*_args, **_kwargs):
        nonlocal claim_create_count
        attempt_hash, attempt_file_hash = runner._load_attempt(
            runner._attempt_path(tmp_path, coordinate, arm),
            binding,
            coordinate,
            arm,
            allowed_root=tmp_path,
        )
        claim_create_count += 1
        create_claim(
            runner._claim_path(tmp_path, coordinate, arm),
            binding,
            coordinate,
            arm,
            attempt_hash,
            attempt_file_hash,
            allowed_root=tmp_path,
        )
        return SimpleNamespace(
            returncode=9,
            stdout="",
            stderr="synthetic post-claim crash",
        )

    monkeypatch.setattr(runner.subprocess, "run", claimed_crash)

    result = runner._run_one(
        tmp_path / "registry.json",
        coordinate,
        arm,
        tmp_path,
        binding,
    )[0]

    assert result["failure_kind"] == "worker_process_failure"
    assert claim_create_count == 1


def test_comparator_process_failure_rejects_existing_symlink_claim(
    tmp_path,
    monkeypatch,
):
    coordinate = _coordinate()
    arm = "chimeraboost_0_15_0"
    binding = {"campaign": "panel3"}
    monkeypatch.setattr(
        runner,
        "_guard_parent_campaign_boundary",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(runner, "_worker_environment", lambda: {})

    def symlinked_claim_crash(*_args, **_kwargs):
        target = tmp_path / "claim-target.json"
        target.write_text("{}")
        runner._claim_path(
            tmp_path,
            coordinate,
            arm,
        ).symlink_to(target)
        return SimpleNamespace(
            returncode=9,
            stdout="",
            stderr="synthetic post-claim crash",
        )

    monkeypatch.setattr(runner.subprocess, "run", symlinked_claim_crash)

    with pytest.raises(RuntimeError, match="invalid panel-3 worker claim"):
        runner._run_one(
            tmp_path / "registry.json",
            coordinate,
            arm,
            tmp_path,
            binding,
        )

    assert not runner._spool_path(tmp_path, coordinate, arm).exists()


def test_comparator_process_failure_rejects_claim_creation_race(
    tmp_path,
    monkeypatch,
):
    coordinate = _coordinate()
    arm = "chimeraboost_0_15_0"
    binding = {"campaign": "panel3"}
    create_claim = runner._create_claim
    monkeypatch.setattr(
        runner,
        "_guard_parent_campaign_boundary",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(runner, "_worker_environment", lambda: {})
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=9,
            stdout="",
            stderr="synthetic pre-claim crash",
        ),
    )

    def race(path, *args, **kwargs):
        create_claim(path, *args, **kwargs)
        raise FileExistsError(path)

    monkeypatch.setattr(runner, "_create_claim", race)

    with pytest.raises(RuntimeError, match="appeared concurrently"):
        runner._run_one(
            tmp_path / "registry.json",
            coordinate,
            arm,
            tmp_path,
            binding,
        )

    assert runner._claim_path(tmp_path, coordinate, arm).is_file()
    assert not runner._spool_path(tmp_path, coordinate, arm).exists()


def test_concurrent_worker_attempt_claim_refuses_second_parent(
    tmp_path,
    monkeypatch,
):
    coordinate = _coordinate()
    arm = runner.CONTROL_ARM
    binding = {"campaign": "panel3"}
    original_create = runner._create_attempt
    monkeypatch.setattr(
        runner,
        "_guard_parent_campaign_boundary",
        lambda *_args, **_kwargs: None,
    )

    def concurrent_claim(path, *args, **kwargs):
        original_create(path, *args, **kwargs)
        raise FileExistsError(path)

    monkeypatch.setattr(runner, "_create_attempt", concurrent_claim)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "the losing parent must not launch a worker"
        ),
    )

    with pytest.raises(RuntimeError, match="claimed concurrently"):
        runner._run_one(
            tmp_path / "registry.json",
            coordinate,
            arm,
            tmp_path,
            binding,
        )


def test_completed_worker_resumes_only_with_matching_attempt(
    tmp_path,
    monkeypatch,
):
    coordinate = _coordinate()
    arm = runner.CONTROL_ARM
    binding = {"campaign": "panel3"}
    result = _synthetic_result(coordinate, arm, 1.0)
    attempt_digest, attempt_file_digest = runner._create_attempt(
        runner._attempt_path(tmp_path, coordinate, arm),
        binding,
        coordinate,
        arm,
        allowed_root=tmp_path,
    )
    claim_digest, claim_file_digest = runner._create_claim(
        runner._claim_path(tmp_path, coordinate, arm),
        binding,
        coordinate,
        arm,
        attempt_digest,
        attempt_file_digest,
        allowed_root=tmp_path,
    )
    _created, spool_digest, spool_file_digest = runner._create_spool(
        runner._spool_path(tmp_path, coordinate, arm),
        binding,
        coordinate,
        arm,
        result,
        allowed_root=tmp_path,
    )
    monkeypatch.setattr(
        runner,
        "_guard_parent_campaign_boundary",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "completed work must be resumed"
        ),
    )

    (
        observed,
        observed_spool,
        observed_spool_file,
        observed_attempt,
        observed_attempt_file,
        observed_claim,
        observed_claim_file,
        resumed,
    ) = runner._run_one(
        tmp_path / "registry.json",
        coordinate,
        arm,
        tmp_path,
        binding,
    )

    assert observed == result
    assert observed_spool == spool_digest
    assert observed_spool_file == spool_file_digest
    assert observed_attempt == attempt_digest
    assert observed_attempt_file == attempt_file_digest
    assert observed_claim == claim_digest
    assert observed_claim_file == claim_file_digest
    assert resumed is True


def test_source_drift_marker_survives_revert_and_blocks_restart(
    tmp_path,
    monkeypatch,
):
    binding = {"campaign": "panel3"}
    monkeypatch.setattr(
        runner,
        "_validate_parent_campaign_boundary",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic source changed")
        ),
    )

    with pytest.raises(RuntimeError, match="source changed"):
        runner._guard_parent_campaign_boundary(
            tmp_path / "registry.json",
            tmp_path,
            binding,
        )

    marker = runner._campaign_invalidation_path(tmp_path)
    assert marker.is_file()
    monkeypatch.setattr(
        runner,
        "_validate_parent_campaign_boundary",
        lambda *_args, **_kwargs: {"reverted": True},
    )
    with pytest.raises(RuntimeError, match="permanently invalidated"):
        runner._run_one(
            tmp_path / "registry.json",
            _coordinate(),
            runner.CONTROL_ARM,
            tmp_path,
            binding,
        )


def test_spool_resume_rejects_symlink_ancestor(tmp_path):
    coordinate = _coordinate()
    arm = runner.CONTROL_ARM
    binding = {"campaign": "panel3"}
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    result = _synthetic_result(coordinate, arm, 1.0)
    runner._create_spool(
        outside / "record.json",
        binding,
        coordinate,
        arm,
        result,
        allowed_root=outside,
    )
    (allowed / "redirect").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="invalid panel-3 spool file"):
        runner._load_spool(
            allowed / "redirect" / "record.json",
            binding,
            coordinate,
            arm,
            allowed_root=allowed,
        )
