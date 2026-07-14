"""Focused tests for the frozen TabArena safe-ordinal confirmation."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import (
    analyze_tabarena_regression_ordinal_confirmation as analysis,
)
from benchmarks import run_tabarena_regression_ordinal_confirmation as confirmation


AIRFOIL_COLUMNS = (
    "frequency",
    "chord-length",
    "free-stream-velocity",
    "suction-side-displacement-thickness",
    "attack-angle",
)


def _fake_jobs():
    experiments = {}
    for arm, spec in confirmation.ARM_SPECS.items():
        model_cls = type(spec["model_cls"], (), {})
        experiments[arm] = SimpleNamespace(
            name=f"DarkoFit_c1_ordinal_confirm_{arm}_BAG_L1",
            method_kwargs={
                "model_cls": model_cls,
                "model_hyperparameters": {
                    **spec["config"],
                    "ag_args": {
                        "name_suffix": f"_c1_ordinal_confirm_{arm}"
                    },
                    "ag_args_ensemble": confirmation.expected_ag_ensemble_config(),
                },
            },
        )
    return [
        SimpleNamespace(
            experiment=experiments[arm],
            task=SimpleNamespace(dataset=dataset, repeat=repeat, fold=fold),
        )
        for dataset, repeat, fold, arm in sorted(
            confirmation.expected_grid(), reverse=True
        )
    ]


def test_confirmation_grid_excludes_discovery_coordinates():
    assert confirmation.TASK_SPLIT_COUNTS == {
        "airfoil_self_noise": (363612, 30),
        "diamonds": (363631, 9),
    }
    assert confirmation.EXPECTED_COORDINATES == 33
    assert confirmation.EXPECTED_JOBS == 99
    assert confirmation.EXPECTED_CHILD_FITS == 792
    assert confirmation.EXPECTED_CONTRAST_PAIRS == 99

    coordinates = confirmation.expected_coordinates()
    assert len(coordinates) == len(set(coordinates)) == 33
    assert sum(row[0] == "airfoil_self_noise" for row in coordinates) == 27
    assert sum(row[0] == "diamonds" for row in coordinates) == 6
    assert set(coordinates).isdisjoint(confirmation.EXCLUDED_COORDINATES)
    assert confirmation.EXCLUDED_COORDINATES == {
        (dataset, repeat, fold)
        for dataset in confirmation.TASK_SPLIT_COUNTS
        for repeat, fold in ((0, 0), (1, 1), (2, 2))
    }
    assert len(confirmation.expected_grid()) == 99
    assert all(
        len(confirmation.expected_arm_coordinates(arm)) == 33
        for arm in confirmation.ARM_SPECS
    )


def test_exact_six_permutation_cycle_restarts_for_each_dataset():
    assert confirmation.ARM_ORDER_CYCLE == (
        ("P", "B", "O"),
        ("B", "O", "P"),
        ("O", "P", "B"),
        ("O", "B", "P"),
        ("P", "O", "B"),
        ("B", "P", "O"),
    )
    ordered = confirmation.expected_ordered_grid()
    assert len(ordered) == 99
    groups = [ordered[index : index + 3] for index in range(0, len(ordered), 3)]
    by_dataset = {
        dataset: [group for group in groups if group[0][0] == dataset]
        for dataset in confirmation.TASK_SPLIT_COUNTS
    }
    for dataset, dataset_groups in by_dataset.items():
        for index, group in enumerate(dataset_groups):
            assert [confirmation.ARM_SPECS[row[3]]["code"] for row in group] == list(
                confirmation.ARM_ORDER_CYCLE[
                    index % len(confirmation.ARM_ORDER_CYCLE)
                ]
            )
            assert len({row[:3] for row in group}) == 1
            assert group[0][0] == dataset

    # A global cycle would start Diamonds at OBP after Airfoil's 27 groups;
    # the predeclared per-dataset restart must start it at PBO instead.
    assert [confirmation.ARM_SPECS[row[3]]["code"] for row in by_dataset["diamonds"][0]] == [
        "P",
        "B",
        "O",
    ]


def test_ordering_is_position_balanced_and_bound_by_digest():
    ordered = confirmation.order_confirmation_jobs(_fake_jobs())
    audit = confirmation.ordering_audit(ordered)
    assert audit["overall_position_counts"] == {
        arm: {"first": 11, "second": 11, "third": 11}
        for arm in confirmation.ARM_SPECS
    }
    assert audit["per_dataset_position_counts"] == {
        "airfoil_self_noise": {
            arm: {"first": 9, "second": 9, "third": 9}
            for arm in confirmation.ARM_SPECS
        },
        "diamonds": {
            arm: {"first": 2, "second": 2, "third": 2}
            for arm in confirmation.ARM_SPECS
        },
    }
    payload = [
        {"dataset": dataset, "repeat": repeat, "fold": fold, "arm": arm}
        for dataset, repeat, fold, arm in confirmation.expected_ordered_grid()
    ]
    expected_digest = hashlib.sha256(
        confirmation.hardened._canonical_json(payload)
    ).hexdigest()
    assert confirmation.job_order_sha256() == expected_digest
    assert audit["job_order_sha256"] == expected_digest


def test_product_fixed_base_and_ordinal_arms_are_not_conflated():
    product = confirmation.ARM_SPECS["product_default_native"]
    base = confirmation.ARM_SPECS["fixed_base_native"]
    ordinal = confirmation.ARM_SPECS["fixed_base_safe_ordinal"]

    assert product["code"] == "P"
    assert product["config"] == {}
    assert product["representation"] == "native"
    assert base["code"] == "B" and ordinal["code"] == "O"
    assert base["config"] == ordinal["config"] == {
        "iterations": 1_000,
        "tree_mode": "catboost",
        "l2_leaf_reg": 3.0,
        "max_bins": 128,
        "learning_rate": 0.1,
        "ts_permutations": 1,
        "linear_residual": False,
        "early_stopping": True,
        "use_best_model": True,
    }
    assert product["model_cls"] == base["model_cls"]
    assert ordinal["model_cls"] != base["model_cls"]
    assert ordinal["representation"] == "safe_ordinal"

    product_method = confirmation.expected_resolved_method_hyperparameters(
        "product_default_native"
    )
    assert set(product_method) == {"ag_args_ensemble"}
    for arm in ("fixed_base_native", "fixed_base_safe_ordinal"):
        resolved = confirmation.expected_resolved_method_hyperparameters(arm)
        assert resolved["learning_rate"] == 0.1
        assert resolved["max_bins"] == 128
        assert resolved["ag_args_ensemble"] == product_method["ag_args_ensemble"]


def test_product_child_keeps_auto_lr_while_fixed_children_request_point_one():
    product = confirmation.expected_child_hyperparameters(
        "product_default_native", 7
    )
    assert product == {
        "iterations": 1_000,
        "early_stopping": True,
        "tree_mode": "catboost",
        "diagnostic_warnings": "never",
        "random_state": 7,
    }
    assert "learning_rate" not in product
    for arm in ("fixed_base_native", "fixed_base_safe_ordinal"):
        child = confirmation.expected_child_hyperparameters(arm, 7)
        assert child["learning_rate"] == 0.1
        assert child["random_state"] == 7
    for value in (-1, 8, True, 1.5, "1"):
        with pytest.raises(RuntimeError, match="child_fold"):
            confirmation.expected_child_hyperparameters(
                "product_default_native", value
            )


def _valid_child_fit_metadata(*, learning_rate: float) -> dict:
    completed = 80
    return {
        "iterations_requested": 1_000,
        "iterations_attempted": 81,
        "rounds_completed": completed,
        "rounds_retained": 72,
        "best_iteration": 72,
        "resolved_learning_rate": learning_rate,
        "requested_tree_mode": "catboost",
        "selected_tree_mode": "catboost",
        "selected_lane": "boosting",
        "linear_residual_active": False,
        "early_stopping_rounds": max(20, min(200, math.ceil(5.0 / learning_rate))),
        "stop_reason": "early_stopping",
        "wall_clock_limit_seconds": 100.0,
        "wall_clock_safety_margin_seconds": 5.0,
        "wall_clock_effective_seconds": 95.0,
        "wall_clock_elapsed_seconds": 10.0,
        "deadline_hit": False,
        "deadline_is_soft": True,
    }


def test_child_metadata_enforces_auto_lr_and_fixed_lr_separately():
    product = _valid_child_fit_metadata(learning_rate=0.07)
    normalized = confirmation._validate_child_fit_metadata(
        product, arm="product_default_native", field="product"
    )
    assert normalized["resolved_learning_rate"] == 0.07
    assert {
        name: normalized[name]
        for name in (
            "wall_clock_limit_seconds",
            "wall_clock_safety_margin_seconds",
            "wall_clock_effective_seconds",
            "wall_clock_elapsed_seconds",
            "deadline_is_soft",
        )
    } == {
        "wall_clock_limit_seconds": 100.0,
        "wall_clock_safety_margin_seconds": 5.0,
        "wall_clock_effective_seconds": 95.0,
        "wall_clock_elapsed_seconds": 10.0,
        "deadline_is_soft": True,
    }

    fixed = _valid_child_fit_metadata(learning_rate=0.1)
    for arm in ("fixed_base_native", "fixed_base_safe_ordinal"):
        normalized = confirmation._validate_child_fit_metadata(
            fixed, arm=arm, field=arm
        )
        assert normalized["resolved_learning_rate"] == 0.1

    wrong_fixed = deepcopy(fixed)
    wrong_fixed["resolved_learning_rate"] = 0.07
    wrong_fixed["early_stopping_rounds"] = math.ceil(5.0 / 0.07)
    with pytest.raises(RuntimeError, match="learning.rate"):
        confirmation._validate_child_fit_metadata(
            wrong_fixed, arm="fixed_base_native", field="fixed"
        )

    leaked_product = deepcopy(product)
    leaked_product["resolved_learning_rate"] = 0.1
    leaked_product["early_stopping_rounds"] = 50
    # 0.1 is a valid possible automatic resolution; the absence of a manual
    # request is proved by the initialized-parameter checks above. The fitted
    # validator must still accept it rather than inventing a distinct value.
    confirmation._validate_child_fit_metadata(
        leaked_product, arm="product_default_native", field="product"
    )

    invalid = deepcopy(product)
    invalid["resolved_learning_rate"] = float("nan")
    with pytest.raises(RuntimeError, match="learning.rate"):
        confirmation._validate_child_fit_metadata(
            invalid, arm="product_default_native", field="product"
        )


def _valid_safe_analyzer_child(*, arm: str = "fixed_base_native") -> dict:
    learning_rate = 0.1 if arm != "product_default_native" else 0.07
    fitted = confirmation._validate_child_fit_metadata(
        _valid_child_fit_metadata(learning_rate=learning_rate),
        arm=arm,
        field="analyzer child",
    )
    return {
        "arm": arm,
        "child_fold": 0,
        "user_hyperparameters": deepcopy(confirmation.ARM_SPECS[arm]["config"]),
        "initial_hyperparameters": confirmation.expected_child_hyperparameters(
            arm, 0
        ),
        "effective_hyperparameters": (
            confirmation.expected_effective_child_hyperparameters(arm, 0)
        ),
        **fitted,
        "refit_params": {
            "iterations": fitted["best_iteration"],
            "learning_rate": learning_rate,
            "tree_mode": "catboost",
            "depth": 6,
            "num_leaves": None,
            "l2_leaf_reg": 3.0,
            "min_child_samples": 20,
            "min_child_weight": 1.0,
            "cat_smoothing": 1.0,
            "early_stopping": False,
            "early_stopping_rounds": None,
            "use_best_model": False,
            "refit": False,
        },
    }


def test_safe_analyzer_independently_revalidates_complete_wall_clock_state():
    valid = _valid_safe_analyzer_child()
    analysis._validate_child_policy(valid, index=0)

    mutations = {
        "wall_clock_limit_seconds": 3_601.0,
        "wall_clock_safety_margin_seconds": 4.0,
        "wall_clock_effective_seconds": 94.0,
        "wall_clock_elapsed_seconds": -1.0,
        "deadline_is_soft": False,
    }
    for field, invalid_value in mutations.items():
        invalid = deepcopy(valid)
        invalid[field] = invalid_value
        with pytest.raises(RuntimeError):
            analysis._validate_child_policy(invalid, index=0)


def _valid_native_representation(dataset: str, features: list[str]) -> dict:
    categorical = confirmation.EXPECTED_NATIVE_CATEGORICAL_COLUMNS[dataset]
    digest = confirmation.followon._feature_schema_sha256(features, "test features")
    return {
        "schema_version": 2,
        "kind": "native",
        "fit_scope": "darkofit_child_training_fold",
        "feature_alignment_policy": "autogluon_child_drop_unique",
        "target_used_by_representation": True,
        "input_feature_count": len(features),
        "output_feature_count": len(features),
        "external_feature_schema_sha256": digest,
        "fitted_feature_schema_sha256": digest,
        "categorical_input_columns": list(categorical),
        "fitted_categorical_input_columns": list(categorical),
        "dropped_constant_input_columns": [],
        "dropped_constant_input_unique_counts": [],
    }


def _valid_airfoil_ordinal_representation() -> dict:
    return {
        "kind": "safe_ordinal",
        "fit_scope": "child_training_rows_only",
        "target_used_by_representation": False,
        "fit_calls": 1,
        "eval_transform_calls_during_fit": 1,
        "eval_unknown_counts": [0],
        "input_feature_count": len(AIRFOIL_COLUMNS),
        "output_feature_count": len(AIRFOIL_COLUMNS),
        "categorical_input_positions": [4],
        "category_schema_sha256": (
            confirmation.followon.EXPECTED_ORDINAL_SCHEMA_SHA256[
                "airfoil_self_noise"
            ]
        ),
        "domain": "airfoil_attack_angle_numeric",
        "mapping_source": "source_frozen_before_campaign",
        "observed_training_category_counts": [27],
        "compact_category_domains": {"attack-angle": list(range(27))},
        "missing_policy": "numeric_nan",
        "unknown_policy": "fail_closed",
        "remaining_native_target_stat_positions": [],
    }


def test_representation_validator_binds_native_schema_and_safe_ordinal_policy():
    features = list(AIRFOIL_COLUMNS)
    native = _valid_native_representation("airfoil_self_noise", features)
    for arm in ("product_default_native", "fixed_base_native"):
        normalized = confirmation._validate_representation_metadata(
            native,
            arm=arm,
            dataset="airfoil_self_noise",
            field=arm,
            child_features=features,
        )
        assert normalized["external_feature_schema_sha256"] == native[
            "external_feature_schema_sha256"
        ]

    wrong_schema = deepcopy(native)
    wrong_schema["external_feature_schema_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="external schema"):
        confirmation._validate_representation_metadata(
            wrong_schema,
            arm="fixed_base_native",
            dataset="airfoil_self_noise",
            field="wrong native",
            child_features=features,
        )

    ordinal = _valid_airfoil_ordinal_representation()
    confirmation._validate_representation_metadata(
        ordinal,
        arm="fixed_base_safe_ordinal",
        dataset="airfoil_self_noise",
        field="ordinal",
        child_features=features,
    )
    target_leak = deepcopy(ordinal)
    target_leak["target_used_by_representation"] = True
    with pytest.raises(RuntimeError, match="target-free"):
        confirmation._validate_representation_metadata(
            target_leak,
            arm="fixed_base_safe_ordinal",
            dataset="airfoil_self_noise",
            field="target leak",
            child_features=features,
        )
    unknown = deepcopy(ordinal)
    unknown["eval_unknown_counts"] = [1]
    with pytest.raises(RuntimeError, match="safety policy"):
        confirmation._validate_representation_metadata(
            unknown,
            arm="fixed_base_safe_ordinal",
            dataset="airfoil_self_noise",
            field="unknown ordinal",
            child_features=features,
        )


def test_frozen_protocol_binds_provenance_order_and_representation_safety():
    protocol = confirmation.frozen_protocol()
    assert protocol["expected_coordinates"] == 33
    assert protocol["expected_jobs"] == 99
    assert protocol["expected_child_fits"] == 792
    assert protocol["order_cycle_scope"] == "restart_for_each_dataset"
    assert protocol["ordered_job_sha256"] == confirmation.job_order_sha256()
    assert protocol["representation_safety"] == {
        "native_categorical_columns": {
            "airfoil_self_noise": ["attack-angle"],
            "diamonds": ["cut", "color", "clarity"],
        },
        "native_metadata_schema_version": 2,
        "native_pair_child_count": 264,
        "ordinal_fit_scope": "child_training_rows_only",
        "ordinal_mapping_source": "source_frozen_before_campaign",
        "ordinal_target_used": False,
        "ordinal_unknown_policy": "fail_closed",
    }
    assert protocol["evidence_boundary"] == {
        "mechanism_unused_coordinates": 33,
        "cap_campaign_previously_inspected_coordinates": True,
        "globally_unseen_confirmation": False,
        "semantics_scope": "dataset_specific_airfoil_and_diamonds_only",
    }
    assert len(confirmation.protocol_sha256()) == 64
    source_files = {str(path) for path in confirmation.SOURCE_FILES}
    assert {
        "benchmarks/run_tabarena_regression_ordinal_confirmation.py",
        "benchmarks/analyze_tabarena_regression_ordinal_confirmation.py",
        "benchmarks/tabarena_regression_ordinal_confirmation_protocol.md",
        "benchmarks/tabarena_screen_adapters.py",
        "benchmarks/run_tabarena_regression_followon_screen.py",
    } <= source_files


def test_resume_invalidates_the_entire_three_arm_coordinate_group(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "ordinal-confirmation"
    output_dir.mkdir()
    jobs = confirmation.order_confirmation_jobs(_fake_jobs())
    coordinate = confirmation.expected_coordinates()[0]
    triad = [job for job in jobs if confirmation._job_coordinate(job) == coordinate]
    assert len(triad) == 3
    for job in triad:
        path = confirmation._result_path(output_dir, job)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(confirmation._job_arm(job).encode())

    def synthetic_issue(path, job):
        del path
        return (
            "incomplete_or_mismatched"
            if confirmation._job_arm(job) == "fixed_base_safe_ordinal"
            else None
        )

    monkeypatch.setattr(confirmation, "_cached_result_issue", synthetic_issue)
    record = confirmation.prepare_grouped_resume(output_dir, jobs, resume=True)
    assert record["invalidated_coordinate_count"] == 1
    assert record["invalidated_result_count"] == 3
    [entry] = record["invalidated_coordinates"]
    assert (entry["dataset"], entry["repeat"], entry["fold"]) == coordinate
    assert entry["arm_status"] == {
        "fixed_base_native": "valid",
        "fixed_base_safe_ordinal": "incomplete_or_mismatched",
        "product_default_native": "valid",
    }
    assert len(entry["archived"]) == 3
    assert all((output_dir / relative).is_file() for relative in entry["archived"])
    assert all(not confirmation._result_path(output_dir, job).exists() for job in triad)


def test_contrasts_and_gate_thresholds_are_frozen_and_attribution_is_report_only():
    assert analysis.BOOTSTRAP_SEED == 20260718
    assert analysis.BOOTSTRAP_DRAWS == 10_000
    assert analysis.CONTRAST_SPECS == {
        "ordinal_vs_fixed_base": {
            "code": "O/B",
            "numerator": "fixed_base_safe_ordinal",
            "denominator": "fixed_base_native",
            "role": "primary_causal",
        },
        "ordinal_vs_product_default": {
            "code": "O/P",
            "numerator": "fixed_base_safe_ordinal",
            "denominator": "product_default_native",
            "role": "deployment",
        },
        "fixed_base_vs_product_default": {
            "code": "B/P",
            "numerator": "fixed_base_native",
            "denominator": "product_default_native",
            "role": "attribution_only",
        },
    }
    assert set(analysis.GATE_THRESHOLDS) == {"O/B", "O/P"}
    assert analysis.GATE_THRESHOLDS["O/B"] == {
        "test_ratio_max": 0.995,
        "bootstrap_upper95_max": 1.0,
        "each_dataset_test_ratio_max": 0.995,
        "sign_test_p_max": 0.05,
        "validation_ratio_max": 1.002,
        "each_dataset_validation_ratio_max": 1.005,
        "train_time_ratio_max": 1.50,
        "infer_time_ratio_max": 1.25,
        "peak_memory_ratio_max": 1.25,
    }
    assert analysis.GATE_THRESHOLDS["O/P"] == {
        "test_ratio_max": 0.995,
        "bootstrap_upper95_max": 1.0,
        "each_dataset_test_ratio_max": 1.005,
        "sign_test_p_max": 0.05,
        "validation_ratio_max": 1.002,
        "train_time_ratio_max": 1.50,
        "infer_time_ratio_max": 1.25,
        "peak_memory_ratio_max": 1.25,
    }


def test_exact_sign_test_uses_all_thirteen_repeat_blocks_and_strict_boundary():
    p_ten = analysis.exact_one_sided_sign_test_pvalue(10, 3)
    p_nine = analysis.exact_one_sided_sign_test_pvalue(9, 4)
    assert p_ten == pytest.approx(378 / 8192, abs=0.0)
    assert p_nine == pytest.approx(1093 / 8192, abs=0.0)
    assert p_ten < 0.05 < p_nine
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        analysis.exact_one_sided_sign_test_pvalue(10, 2)


def _gate_metrics() -> dict[str, dict[str, float]]:
    return {
        "test_rmse": {"ratio": 0.995},
        "val_rmse": {"ratio": 1.002},
        "train_time_s": {"ratio": 1.50},
        "infer_time_s": {"ratio": 1.25},
        "peak_memory_bytes": {"ratio": 1.25},
    }


def test_gate_inclusivity_strictness_and_contrast_specific_guardrails():
    primary_datasets = [
        {"dataset": "airfoil_self_noise", "test_rmse_ratio": 0.995, "val_rmse_ratio": 1.005},
        {"dataset": "diamonds", "test_rmse_ratio": 0.995, "val_rmse_ratio": 1.005},
    ]
    primary = analysis._contrast_gates(
        "O/B",
        _gate_metrics(),
        primary_datasets,
        bootstrap_upper95=0.999999,
        sign_p=0.049999,
        campaign_clean=True,
    )
    assert primary["advance"] is True

    strict_bootstrap = analysis._contrast_gates(
        "O/B",
        _gate_metrics(),
        primary_datasets,
        bootstrap_upper95=1.0,
        sign_p=0.049999,
        campaign_clean=True,
    )
    assert strict_bootstrap["one_sided_bootstrap_upper95_below_1"] is False
    strict_sign = analysis._contrast_gates(
        "O/B",
        _gate_metrics(),
        primary_datasets,
        bootstrap_upper95=0.999999,
        sign_p=0.05,
        campaign_clean=True,
    )
    assert strict_sign["repeat_block_sign_test_p_below_0_05"] is False

    deployment_datasets = [
        {"dataset": "airfoil_self_noise", "test_rmse_ratio": 1.005, "val_rmse_ratio": 2.0},
        {"dataset": "diamonds", "test_rmse_ratio": 0.98, "val_rmse_ratio": 2.0},
    ]
    deployment = analysis._contrast_gates(
        "O/P",
        _gate_metrics(),
        deployment_datasets,
        bootstrap_upper95=0.999999,
        sign_p=0.049999,
        campaign_clean=True,
    )
    assert deployment["dataset_test_guardrail"] is True
    assert "each_dataset_validation_ratio_at_most_1_005" not in deployment
    assert analysis._contrast_gates(
        "B/P",
        _gate_metrics(),
        deployment_datasets,
        bootstrap_upper95=0.5,
        sign_p=0.0,
        campaign_clean=True,
    ) is None


def _hierarchy_rows() -> list[dict]:
    rows = []
    # Airfoil has ten repeats; Diamonds has three. Deliberately give Airfoil a
    # large positive log ratio and Diamonds a larger negative one. Correct
    # equal-dataset aggregation is -0.1, whereas pooling all 13 repeats is >0.
    for dataset, repeat, fold in confirmation.expected_coordinates():
        rows.append(
            {
                "contrast": "ordinal_vs_fixed_base",
                "contrast_code": "O/B",
                "dataset": dataset,
                "repeat": repeat,
                "fold": fold,
                "test_rmse_log_ratio": (
                    0.1 if dataset == "airfoil_self_noise" else -0.3
                ),
            }
        )
    return rows


def test_hierarchical_point_estimate_keeps_datasets_fixed_and_equal_weighted():
    overall, dataset_logs, repeat_logs = analysis.hierarchical_point_log_ratio(
        _hierarchy_rows(), "test_rmse_log_ratio"
    )
    assert dataset_logs == {
        "airfoil_self_noise": pytest.approx(0.1),
        "diamonds": pytest.approx(-0.3),
    }
    assert overall == pytest.approx(-0.1)
    assert sum(len(repeats) for repeats in repeat_logs.values()) == 13


def test_hierarchical_bootstrap_is_seeded_and_does_not_resample_datasets():
    rows = _hierarchy_rows()
    first = analysis.hierarchical_bootstrap_log_ratios(
        rows, draws=64, seed=20260718
    )
    second = analysis.hierarchical_bootstrap_log_ratios(
        rows, draws=64, seed=20260718
    )
    np.testing.assert_array_equal(first, second)
    # Every fold within each fixed dataset is constant, so resampling repeats
    # and folds cannot move the equal-dataset result. Resampling datasets would.
    np.testing.assert_allclose(first, -0.1, rtol=0.0, atol=1e-15)


def test_worst_coordinate_is_diagnostic_and_cannot_become_a_gate():
    for thresholds in analysis.GATE_THRESHOLDS.values():
        assert all("coordinate" not in key and "split" not in key for key in thresholds)
    assert "B/P" not in analysis.GATE_THRESHOLDS
    assert analysis.CONTRAST_SPECS["fixed_base_vs_product_default"]["role"] == (
        "attribution_only"
    )


def test_analyzer_rejects_output_paths_outside_the_campaign_root():
    with pytest.raises(SystemExit):
        analysis.parse_args(
            [
                "--input-dir",
                "/tmp/ordinal-confirmation",
                "--summary-json",
                "/tmp/outside-summary.json",
            ]
        )


def test_protocol_document_contains_the_registered_caveats_and_gates():
    path = Path("benchmarks/tabarena_regression_ordinal_confirmation_protocol.md")
    text = path.read_text(encoding="utf-8")
    for fragment in (
        "27",
        "6",
        "99 outer jobs",
        "792 child fits",
        "seed **20260718**",
        "already inspected them",
        "not globally",
        "dataset-specific",
        "cycle restarts",
        "Primary causal contrast",
        "Deployment contrast",
        "Attribution contrast",
        "report-only",
    ):
        assert fragment in text
    # Guard against accidentally changing the protocol through malformed JSON
    # in a future frozen-protocol refactor.
    json.dumps(confirmation.frozen_protocol(), sort_keys=True, allow_nan=False)
